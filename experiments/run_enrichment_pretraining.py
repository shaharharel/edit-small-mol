#!/usr/bin/env python3
"""
Enrichment Pretraining for FiLMDelta Extrapolation on ZAP70 Candidates.

Goal: Improve FiLMDelta predictions for 19 novel ZAP70 candidates that are
structurally distant from the training set (max Tanimoto 0.177-0.280).

Approaches:
  1. Chemical Neighbor Enrichment — find ChEMBL molecules similar to candidates
  2. Kinase Panel Pretraining — pretrain on all kinase MMP pairs
  3. Scaffold-Bridging — find molecules with same scaffolds as candidates
  4. Curriculum Transfer Learning — general → kinase → ZAP70+enrichment

Validation: extrapolation test on 16 distant ZAP70 molecules (same regime).

Usage:
    conda run -n quris python -u experiments/run_enrichment_pretraining.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

import gc
import json
import time
import warnings
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from collections import defaultdict

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

# MPS is available
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "enrichment_pretraining"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
SHARED_PAIRS = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
CHEMBL_DB = PROJECT_ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
CANDIDATES_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "19_molecules_scoring.json"
EMB_CACHE = PROJECT_ROOT / "data" / "embedding_cache"

# Experiment params
ZAP70_ID = "CHEMBL2803"
TC_THRESHOLD = 0.3  # Train/test split threshold
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_SEEDS = 3
HIDDEN_DIMS = [1024, 512, 256]
EMB_DIM = 2048
FP_RADIUS = 2
FP_NBITS = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4

# ====================================================================
# Utility functions
# ====================================================================

def compute_morgan_fp(smi, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Compute Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_fp_obj(smi, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Compute Morgan fingerprint as RDKit object (for Tanimoto)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_tanimoto_matrix(smiles_list):
    """Compute pairwise Tanimoto similarity matrix."""
    fps = []
    for smi in smiles_list:
        fp = compute_morgan_fp_obj(smi)
        fps.append(fp)
    n = len(fps)
    tc_matrix = np.zeros((n, n))
    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i + 1, n):
            if fps[j] is None:
                continue
            tc = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            tc_matrix[i, j] = tc
            tc_matrix[j, i] = tc
    return tc_matrix


def bulk_tanimoto(query_fps, target_fps):
    """Compute max Tanimoto of each query to any target. Returns (max_tc, best_idx)."""
    max_tc = np.zeros(len(query_fps))
    best_idx = np.zeros(len(query_fps), dtype=int)
    for i, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qfp, target_fps)
        max_tc[i] = max(sims) if sims else 0
        best_idx[i] = np.argmax(sims) if sims else 0
    return max_tc, best_idx


def get_murcko_scaffold(smi):
    """Get Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except:
        return None


def generate_all_pairs(smiles, pIC50):
    """Generate all-vs-all pairs for a set of molecules."""
    n = len(smiles)
    mol_a, mol_b, deltas = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                mol_a.append(smiles[i])
                mol_b.append(smiles[j])
                deltas.append(float(pIC50[j] - pIC50[i]))
    return pd.DataFrame({'mol_a': mol_a, 'mol_b': mol_b, 'delta': deltas})


def load_zap70_molecules():
    """Load ZAP70 molecule-level data."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  ZAP70: {len(mol_data)} molecules, "
          f"pIC50 {mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f}")
    return mol_data


def load_candidates():
    """Load the 19 candidate molecules."""
    with open(CANDIDATES_FILE) as f:
        data = json.load(f)
    candidates = []
    for mol in data['results']:
        candidates.append({
            'idx': mol['idx'],
            'smiles': mol['smiles_clean'],
            'max_tanimoto': mol['max_tanimoto'],
            'film_delta_pred': mol['film_delta'],
            'consensus_pred': mol['consensus'],
        })
    return pd.DataFrame(candidates)


def build_embedding_dict(smiles_list):
    """Build embedding dictionary for a list of SMILES."""
    emb_dict = {}
    for smi in smiles_list:
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)
    return emb_dict


# ====================================================================
# FiLMDelta training and prediction
# ====================================================================

def train_film_delta(
    train_pairs, val_pairs, emb_dict, seed,
    pretrained_state=None, lr=LR, epochs=MAX_EPOCHS, patience=PATIENCE,
    freeze_delta_encoder=False, label="FiLMDelta"
):
    """
    Train a FiLMDelta model on MMP-style pairs.

    Args:
        train_pairs: DataFrame with mol_a, mol_b, delta columns
        val_pairs: DataFrame for validation
        emb_dict: dict mapping SMILES -> np array
        seed: random seed
        pretrained_state: optional state_dict to initialize from
        lr: learning rate
        epochs: max epochs
        patience: early stopping patience
        freeze_delta_encoder: if True, freeze delta encoder during fine-tuning
        label: label for logging

    Returns:
        Trained model (on CPU), scaler
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in train_pairs['mol_a']])
    train_b = np.array([get_emb(s) for s in train_pairs['mol_b']])
    train_y = train_pairs['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_pairs['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs['mol_b']])
    val_y = val_pairs['delta'].values.astype(np.float32)

    # Standardize embeddings
    scaler = StandardScaler()
    all_embs = np.vstack([train_a, train_b, val_a, val_b])
    scaler.fit(all_embs)
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_y))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_y))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=EMB_DIM, hidden_dims=HIDDEN_DIMS, dropout=0.2).to(device)

    # Load pretrained weights if available
    if pretrained_state is not None:
        try:
            model.load_state_dict(pretrained_state, strict=True)
        except RuntimeError:
            # Try partial load
            own_state = model.state_dict()
            for name, param in pretrained_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
            model.load_state_dict(own_state)
        model = model.to(device)

    # Optionally freeze delta encoder
    if freeze_delta_encoder:
        for name, param in model.named_parameters():
            if 'delta_encoder' in name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_vl, pat_counter, best_st = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl_sum, nv = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(device) for t in batch]
                vl_sum += criterion(model(a, b), y).item()
                nv += 1
        vl = vl_sum / max(nv, 1)
        scheduler.step(vl)

        if vl < best_vl:
            best_vl = vl
            pat_counter = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_counter += 1
            if pat_counter >= patience:
                break

    if best_st:
        model.load_state_dict(best_st)
    model.cpu().eval()

    return model, scaler


def anchor_predict(model, train_smiles, train_y, test_smiles, emb_dict, scaler):
    """Anchor-based prediction: pIC50(j) = median(pIC50(i) + delta(i->j))."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

    n_train = len(train_smiles)
    n_test = len(test_smiles)

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(
        np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

    preds = np.zeros(n_test)
    model.eval()
    with torch.no_grad():
        for j in range(n_test):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (n_train, 1)))
            deltas = model(a, b).numpy().flatten()
            preds[j] = np.median(train_y + deltas)

    return preds


def evaluate_predictions(y_true, y_pred):
    """Compute evaluation metrics."""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    if len(y_true) < 3 or np.std(y_pred) < 1e-6:
        return {'mae': mae, 'rmse': rmse, 'spearman': 0.0, 'pearson': 0.0, 'r2': 0.0}

    spr, _ = scipy_stats.spearmanr(y_pred, y_true)
    prs, _ = scipy_stats.pearsonr(y_pred, y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': mae, 'rmse': rmse,
        'spearman': float(spr), 'pearson': float(prs), 'r2': float(r2)
    }


# ====================================================================
# Extrapolation test setup
# ====================================================================

def setup_extrapolation_test():
    """
    Set up the extrapolation test: split ZAP70 molecules into
    train (max Tc >= 0.3 to any other mol) and test (max Tc < 0.3).
    Returns train_smiles, train_y, test_smiles, test_y, test_tc_to_train, emb_dict
    """
    print("\n" + "=" * 70)
    print("Setting up extrapolation test")
    print("=" * 70)

    mol_data = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n_total = len(all_smiles)

    print("  Computing Tanimoto matrix...")
    tc_matrix = compute_tanimoto_matrix(all_smiles)
    np.fill_diagonal(tc_matrix, 0)
    max_tc = tc_matrix.max(axis=1)

    test_mask = max_tc < TC_THRESHOLD
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    test_smiles = [all_smiles[i] for i in test_idx]
    test_y = all_y[test_idx]
    train_smiles = [all_smiles[i] for i in train_idx]
    train_y = all_y[train_idx]

    # Max Tc of test to train
    test_tc_to_train = np.zeros(len(test_idx))
    for i, ti in enumerate(test_idx):
        test_tc_to_train[i] = tc_matrix[ti, train_idx].max()

    print(f"  Train: {len(train_smiles)} molecules")
    print(f"  Test:  {len(test_smiles)} molecules (max Tc < {TC_THRESHOLD})")
    print(f"  Test max Tc to train: {test_tc_to_train.min():.3f} - {test_tc_to_train.max():.3f}")

    # Build embeddings
    all_smi_unique = list(set(train_smiles + test_smiles))
    emb_dict = build_embedding_dict(all_smi_unique)

    # Naive baseline
    naive_mae = float(np.mean(np.abs(train_y.mean() - test_y)))
    print(f"  Naive baseline (predict mean={train_y.mean():.2f}): MAE={naive_mae:.3f}")

    return train_smiles, train_y, test_smiles, test_y, test_tc_to_train, emb_dict, naive_mae


def generate_enriched_pairs(zap_smiles, zap_y, extra_smiles, extra_y, max_pairs=100000, seed=42):
    """
    Generate pairs combining ZAP70 molecules with enrichment molecules.
    Strategy:
    - All ZAP70-vs-ZAP70 pairs (always included)
    - Cross pairs: ZAP70-vs-enrichment (sampled to limit)
    - No enrichment-vs-enrichment pairs (too many, less relevant)
    """
    np.random.seed(seed)

    # ZAP70 internal pairs (always include all)
    zap_pairs = generate_all_pairs(zap_smiles, zap_y)
    n_zap_pairs = len(zap_pairs)

    # Cross pairs: each ZAP70 mol paired with enrichment mols
    cross_a, cross_b, cross_delta = [], [], []
    for i, (zsmi, zy) in enumerate(zip(zap_smiles, zap_y)):
        for j, (esmi, ey) in enumerate(zip(extra_smiles, extra_y)):
            # Both directions
            cross_a.append(zsmi)
            cross_b.append(esmi)
            cross_delta.append(float(ey - zy))
            cross_a.append(esmi)
            cross_b.append(zsmi)
            cross_delta.append(float(zy - ey))

    cross_df = pd.DataFrame({'mol_a': cross_a, 'mol_b': cross_b, 'delta': cross_delta})

    # Sample cross pairs if too many
    remaining_budget = max_pairs - n_zap_pairs
    if remaining_budget > 0 and len(cross_df) > remaining_budget:
        cross_df = cross_df.sample(n=remaining_budget, random_state=seed)
    elif remaining_budget <= 0:
        cross_df = cross_df.sample(n=max(10000, n_zap_pairs), random_state=seed)

    all_pairs = pd.concat([zap_pairs, cross_df], ignore_index=True)
    print(f"    Pairs: {n_zap_pairs} ZAP70 + {len(cross_df)} cross = {len(all_pairs)} total")

    return all_pairs


def run_extrapolation_test(
    train_smiles, train_y, test_smiles, test_y,
    emb_dict, pretrained_state=None, lr=LR, epochs=MAX_EPOCHS,
    patience=PATIENCE, n_seeds=N_SEEDS, label="Baseline",
    extra_train_smiles=None, extra_train_y=None,
    freeze_delta_encoder=False,
):
    """
    Run extrapolation test with optional pretrained weights and enrichment data.

    Args:
        extra_train_smiles, extra_train_y: additional molecules to add as anchors
            (their pairs with ZAP70 train mols are added to training)
    """
    print(f"\n  Running extrapolation test: {label} ({n_seeds} seeds)...")

    has_enrichment = extra_train_smiles is not None and len(extra_train_smiles) > 0

    if has_enrichment:
        print(f"    Enrichment: {len(extra_train_smiles)} additional molecules")
        # Ensure embeddings exist
        for smi in extra_train_smiles:
            if smi not in emb_dict:
                emb_dict[smi] = compute_morgan_fp(smi)

    all_preds = np.zeros((n_seeds, len(test_smiles)))

    for seed_i in range(n_seeds):
        t0 = time.time()
        actual_seed = seed_i * 17 + 5

        # Generate training pairs
        if has_enrichment:
            pairs = generate_enriched_pairs(
                train_smiles, train_y,
                extra_train_smiles, extra_train_y,
                max_pairs=150000, seed=actual_seed
            )
        else:
            pairs = generate_all_pairs(train_smiles, train_y)

        n_val = max(int(len(pairs) * 0.1), 100)
        val_pairs = pairs.sample(n=n_val, random_state=actual_seed)
        trn_pairs = pairs.drop(val_pairs.index)

        model, scaler = train_film_delta(
            trn_pairs, val_pairs, emb_dict, actual_seed,
            pretrained_state=pretrained_state, lr=lr, epochs=epochs,
            patience=patience, freeze_delta_encoder=freeze_delta_encoder,
            label=label,
        )

        # Use only original ZAP70 train molecules as anchors
        preds = anchor_predict(model, train_smiles, train_y, test_smiles, emb_dict, scaler)
        all_preds[seed_i] = preds

        mae = np.mean(np.abs(preds - test_y))
        spr = scipy_stats.spearmanr(preds, test_y).correlation if len(test_y) > 3 else 0
        elapsed = time.time() - t0
        print(f"    Seed {seed_i+1}/{n_seeds}: MAE={mae:.3f}, Spr={spr:.3f} ({elapsed:.0f}s)")

        del model
        gc.collect()

    # Aggregate
    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)

    metrics = evaluate_predictions(test_y, mean_preds)
    per_seed_mae = [float(np.mean(np.abs(all_preds[s] - test_y))) for s in range(n_seeds)]
    per_seed_spr = [float(scipy_stats.spearmanr(all_preds[s], test_y).correlation)
                    for s in range(n_seeds)]

    result = {
        'label': label,
        'metrics': metrics,
        'per_seed_mae': per_seed_mae,
        'per_seed_mae_mean': float(np.mean(per_seed_mae)),
        'per_seed_mae_std': float(np.std(per_seed_mae)),
        'per_seed_spr': per_seed_spr,
        'per_seed_spr_mean': float(np.mean(per_seed_spr)),
        'per_seed_spr_std': float(np.std(per_seed_spr)),
        'predictions': mean_preds.tolist(),
        'pred_stds': std_preds.tolist(),
    }

    print(f"    {label} ensemble: MAE={metrics['mae']:.3f}, "
          f"Spr={metrics['spearman']:.3f}, R2={metrics['r2']:.3f}")

    return result


# ====================================================================
# Approach 0: Baseline (no enrichment)
# ====================================================================

def run_baseline(train_smiles, train_y, test_smiles, test_y, emb_dict):
    """Run baseline FiLMDelta without any enrichment."""
    print("\n" + "=" * 70)
    print("APPROACH 0: Baseline FiLMDelta (no enrichment)")
    print("=" * 70)

    result = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        label="Baseline"
    )

    save_result('baseline', result)
    return result


# ====================================================================
# Approach 1: Chemical Neighbor Enrichment from ChEMBL
# ====================================================================

def find_chemical_neighbors_chembl(candidate_smiles, tc_threshold=0.3, max_per_candidate=200):
    """
    Search ChEMBL for molecules similar to candidates.
    Returns DataFrame with smiles, pIC50, target info.
    """
    print(f"\n  Searching ChEMBL for neighbors (Tc > {tc_threshold})...")

    # Get candidate fingerprints
    cand_fps = []
    for smi in candidate_smiles:
        fp = compute_morgan_fp_obj(smi)
        cand_fps.append(fp)

    conn = sqlite3.connect(str(CHEMBL_DB))

    # Get all molecules with IC50/Ki data on kinase-like targets
    query = """
    SELECT DISTINCT cs.canonical_smiles, act.standard_value, act.standard_type,
           act.standard_units, td.chembl_id as target_id, td.pref_name as target_name
    FROM compound_structures cs
    JOIN activities act ON cs.molregno = act.molregno
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    WHERE act.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
    AND act.standard_value IS NOT NULL
    AND act.standard_value > 0
    AND act.standard_units = 'nM'
    AND act.standard_relation IN ('=', '<', '<=')
    AND td.target_type = 'SINGLE PROTEIN'
    AND cs.canonical_smiles IS NOT NULL
    """

    print("  Querying ChEMBL database (this may take a few minutes)...")
    t0 = time.time()

    # Process in chunks to avoid memory issues
    chunk_size = 500000
    neighbors = []
    seen_smiles = set()

    for chunk_df in pd.read_sql(query, conn, chunksize=chunk_size):
        print(f"    Processing chunk of {len(chunk_df)} rows...")

        for _, row in chunk_df.iterrows():
            smi = row['canonical_smiles']
            if smi in seen_smiles:
                continue

            fp = compute_morgan_fp_obj(smi)
            if fp is None:
                continue

            # Check Tanimoto to any candidate
            max_tc = 0
            for cfp in cand_fps:
                if cfp is not None:
                    tc = DataStructs.TanimotoSimilarity(fp, cfp)
                    max_tc = max(max_tc, tc)

            if max_tc >= tc_threshold:
                # Convert to pIC50
                val = row['standard_value']
                if val > 0:
                    pIC50 = -np.log10(val * 1e-9)
                else:
                    continue

                neighbors.append({
                    'smiles': smi,
                    'pIC50': pIC50,
                    'target_id': row['target_id'],
                    'target_name': row['target_name'],
                    'max_tc_to_candidates': max_tc,
                    'standard_type': row['standard_type'],
                })
                seen_smiles.add(smi)

        if len(neighbors) > max_per_candidate * len(candidate_smiles):
            print(f"    Reached {len(neighbors)} neighbors, stopping search")
            break

    conn.close()
    elapsed = time.time() - t0
    print(f"  Found {len(neighbors)} neighbor molecules in {elapsed:.0f}s")

    if len(neighbors) == 0:
        return pd.DataFrame()

    return pd.DataFrame(neighbors)


def find_chemical_neighbors_from_dataset(candidate_smiles, tc_threshold=0.25):
    """
    Find molecules in the existing dataset (molecule_pIC50_minimal.csv)
    that are similar to the 19 candidates. Faster than ChEMBL search.
    """
    print(f"\n  Searching existing dataset for neighbors (Tc > {tc_threshold})...")

    cand_fps = [compute_morgan_fp_obj(s) for s in candidate_smiles]

    # Load molecule data
    raw = pd.read_csv(RAW_FILE)
    mol_data = raw.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
        "target_chembl_id": lambda x: list(x.unique()),
    }).reset_index()

    print(f"  Dataset has {len(mol_data)} unique molecules across {raw['target_chembl_id'].nunique()} targets")

    neighbors = []
    for _, row in mol_data.iterrows():
        smi = row['smiles']
        fp = compute_morgan_fp_obj(smi)
        if fp is None:
            continue

        max_tc = 0
        for cfp in cand_fps:
            if cfp is not None:
                tc = DataStructs.TanimotoSimilarity(fp, cfp)
                max_tc = max(max_tc, tc)

        if max_tc >= tc_threshold:
            neighbors.append({
                'smiles': smi,
                'pIC50': row['pIC50'],
                'targets': row['target_chembl_id'],
                'max_tc_to_candidates': max_tc,
                'molecule_chembl_id': row['molecule_chembl_id'],
            })

    print(f"  Found {len(neighbors)} neighbor molecules")
    return pd.DataFrame(neighbors)


def run_chemical_neighbor_enrichment(
    train_smiles, train_y, test_smiles, test_y, emb_dict,
    candidate_smiles
):
    """Approach 1: Enrich training with chemical neighbors of candidates."""
    print("\n" + "=" * 70)
    print("APPROACH 1: Chemical Neighbor Enrichment")
    print("=" * 70)

    # First try the existing dataset (fast)
    neighbors = find_chemical_neighbors_from_dataset(candidate_smiles, tc_threshold=0.25)

    if len(neighbors) == 0:
        print("  No neighbors found in existing dataset. Trying ChEMBL...")
        neighbors = find_chemical_neighbors_chembl(candidate_smiles, tc_threshold=0.3)

    if len(neighbors) == 0:
        print("  No neighbors found. Skipping this approach.")
        return None

    # Filter out ZAP70 training molecules (to avoid leakage)
    train_set = set(train_smiles)
    test_set = set(test_smiles)
    neighbors = neighbors[~neighbors['smiles'].isin(train_set | test_set)]

    print(f"  After removing ZAP70 molecules: {len(neighbors)} neighbors")
    print(f"  Max Tc distribution: min={neighbors['max_tc_to_candidates'].min():.3f}, "
          f"max={neighbors['max_tc_to_candidates'].max():.3f}, "
          f"mean={neighbors['max_tc_to_candidates'].mean():.3f}")

    # Add neighbor embeddings
    for smi in neighbors['smiles'].unique():
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)

    extra_smiles = neighbors['smiles'].tolist()
    extra_y = neighbors['pIC50'].values.astype(np.float32)

    result = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        label="ChemNeighbor_Enrichment",
        extra_train_smiles=extra_smiles,
        extra_train_y=extra_y,
    )

    result['n_neighbors'] = len(neighbors)
    result['tc_threshold'] = 0.25
    save_result('chemical_neighbors', result)
    return result


# ====================================================================
# Approach 2: Kinase Panel Pretraining
# ====================================================================

def get_kinase_target_ids():
    """Get all kinase target ChEMBL IDs from the database."""
    conn = sqlite3.connect(str(CHEMBL_DB))

    # Get kinase protein class IDs (class_level 2, id=6 is 'Kinase')
    query = """
    SELECT DISTINCT td.chembl_id
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_class cc ON tc.component_id = cc.component_id
    JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
    WHERE pc.protein_class_id IN (
        SELECT protein_class_id FROM protein_classification
        WHERE protein_class_id = 6
        OR parent_id = 6
        OR parent_id IN (SELECT protein_class_id FROM protein_classification WHERE parent_id = 6)
        OR parent_id IN (
            SELECT protein_class_id FROM protein_classification
            WHERE parent_id IN (
                SELECT protein_class_id FROM protein_classification WHERE parent_id = 6
            )
        )
    )
    AND td.target_type = 'SINGLE PROTEIN'
    """

    kinase_ids = pd.read_sql(query, conn)['chembl_id'].tolist()
    conn.close()
    print(f"  Found {len(kinase_ids)} kinase targets in ChEMBL")
    return set(kinase_ids)


def load_kinase_pairs(kinase_ids, max_pairs=500000):
    """Load MMP pairs for kinase targets from shared_pairs dataset."""
    print(f"  Loading kinase pairs from shared_pairs dataset...")

    kinase_pairs = []
    chunk_size = 200000
    total_read = 0

    for chunk in pd.read_csv(SHARED_PAIRS, chunksize=chunk_size):
        kinase_chunk = chunk[chunk['target_chembl_id'].isin(kinase_ids)]
        # Prefer within-assay pairs
        within = kinase_chunk[kinase_chunk['is_within_assay'] == True]
        if len(within) > 0:
            kinase_pairs.append(within[['mol_a', 'mol_b', 'delta', 'target_chembl_id']])
        else:
            kinase_pairs.append(kinase_chunk[['mol_a', 'mol_b', 'delta', 'target_chembl_id']])

        total_read += len(chunk)
        total_kinase = sum(len(kp) for kp in kinase_pairs)

        if total_kinase >= max_pairs:
            print(f"    Reached {total_kinase} kinase pairs, stopping")
            break

    if len(kinase_pairs) == 0:
        return pd.DataFrame()

    result = pd.concat(kinase_pairs, ignore_index=True)
    print(f"  Loaded {len(result)} kinase pairs across {result['target_chembl_id'].nunique()} targets")
    return result


def pretrain_on_kinase(kinase_pairs, emb_dict, seed=42):
    """
    Pretrain FiLMDelta on kinase MMP pairs.
    Returns pretrained state_dict.
    """
    print(f"\n  Pretraining on {len(kinase_pairs)} kinase pairs...")

    # Ensure all molecules have embeddings
    all_smiles = set(kinase_pairs['mol_a'].tolist() + kinase_pairs['mol_b'].tolist())
    new_count = 0
    for smi in all_smiles:
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)
            new_count += 1
    print(f"    Computed {new_count} new embeddings (total: {len(emb_dict)})")

    # Split into train/val
    n_val = max(int(len(kinase_pairs) * 0.1), 1000)
    np.random.seed(seed)
    val_idx = np.random.choice(len(kinase_pairs), size=n_val, replace=False)
    train_mask = np.ones(len(kinase_pairs), dtype=bool)
    train_mask[val_idx] = False

    trn_pairs = kinase_pairs[train_mask]
    val_pairs = kinase_pairs[~train_mask]

    model, scaler = train_film_delta(
        trn_pairs, val_pairs, emb_dict, seed,
        lr=1e-3, epochs=80, patience=10,
        label="Kinase Pretrain"
    )

    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

    del model, scaler
    gc.collect()

    return pretrained_state


def run_kinase_pretraining(
    train_smiles, train_y, test_smiles, test_y, emb_dict,
    max_kinase_pairs=200000
):
    """Approach 2: Pretrain on kinase pairs, then fine-tune on ZAP70."""
    print("\n" + "=" * 70)
    print("APPROACH 2: Kinase Panel Pretraining")
    print("=" * 70)

    # Get kinase target IDs
    kinase_ids = get_kinase_target_ids()

    # Also include kinase targets from existing dataset
    raw = pd.read_csv(RAW_FILE, usecols=['target_chembl_id', 'target_name'])
    target_names = raw.groupby('target_chembl_id')['target_name'].first()
    kinase_from_names = set()
    for tid, tname in target_names.items():
        if tname and ('kinase' in str(tname).lower() or 'ZAP' in str(tname)):
            kinase_from_names.add(tid)
    kinase_ids = kinase_ids | kinase_from_names
    print(f"  Total kinase target IDs (DB + name matching): {len(kinase_ids)}")

    # Load kinase pairs
    kinase_pairs = load_kinase_pairs(kinase_ids, max_pairs=max_kinase_pairs)

    if len(kinase_pairs) == 0:
        print("  No kinase pairs found. Skipping.")
        return None

    # Pretrain
    pretrained_state = pretrain_on_kinase(kinase_pairs, emb_dict)

    # Save pretrained weights
    torch.save(pretrained_state, RESULTS_DIR / "kinase_pretrained_state.pt")
    print(f"  Saved pretrained weights to kinase_pretrained_state.pt")

    # Fine-tune on ZAP70 with extrapolation test
    results = {}

    # Variant A: Fine-tune all parameters
    result_a = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        pretrained_state=pretrained_state,
        lr=5e-4,  # Lower LR for fine-tuning
        label="Kinase_Pretrain_FineTune_All",
    )
    results['finetune_all'] = result_a
    save_result('kinase_pretrain_finetune_all', result_a)

    # Variant B: Fine-tune with frozen delta encoder
    result_b = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        pretrained_state=pretrained_state,
        lr=5e-4,
        freeze_delta_encoder=True,
        label="Kinase_Pretrain_FrozenDelta",
    )
    results['finetune_frozen_delta'] = result_b
    save_result('kinase_pretrain_frozen_delta', result_b)

    # Variant C: Lower LR for fine-tuning
    result_c = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        pretrained_state=pretrained_state,
        lr=1e-4,
        label="Kinase_Pretrain_LowLR",
    )
    results['finetune_low_lr'] = result_c
    save_result('kinase_pretrain_low_lr', result_c)

    return results


# ====================================================================
# Approach 3: Scaffold-Bridging
# ====================================================================

def run_scaffold_bridging(
    train_smiles, train_y, test_smiles, test_y, emb_dict,
    candidate_smiles
):
    """
    Approach 3: Find molecules with same scaffolds as candidates,
    add them to training.
    """
    print("\n" + "=" * 70)
    print("APPROACH 3: Scaffold-Bridging")
    print("=" * 70)

    # Get candidate scaffolds
    cand_scaffolds = set()
    for smi in candidate_smiles:
        scaf = get_murcko_scaffold(smi)
        if scaf:
            cand_scaffolds.add(scaf)
    print(f"  Candidate scaffolds: {len(cand_scaffolds)}")
    for scaf in list(cand_scaffolds)[:10]:
        print(f"    {scaf}")

    # Search existing dataset for molecules with matching scaffolds
    raw = pd.read_csv(RAW_FILE)
    mol_data = raw.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()

    bridge_mols = []
    for _, row in mol_data.iterrows():
        smi = row['smiles']
        scaf = get_murcko_scaffold(smi)
        if scaf and scaf in cand_scaffolds:
            bridge_mols.append({
                'smiles': smi,
                'pIC50': row['pIC50'],
                'scaffold': scaf,
            })

    train_set = set(train_smiles)
    test_set = set(test_smiles)

    if len(bridge_mols) > 0:
        bridge_df = pd.DataFrame(bridge_mols)
        bridge_df = bridge_df[~bridge_df['smiles'].isin(train_set | test_set)]
    else:
        bridge_df = pd.DataFrame(columns=['smiles', 'pIC50', 'scaffold'])

    print(f"  Found {len(bridge_df)} bridge molecules (same scaffold as candidates)")

    if len(bridge_df) == 0:
        # Try generic scaffolds (more abstracted)
        print("  Trying generic scaffolds...")
        cand_generic_scaffolds = set()
        for smi in candidate_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    generic = MurckoScaffold.MakeScaffoldGeneric(core)
                    cand_generic_scaffolds.add(Chem.MolToSmiles(generic))
                except:
                    pass

        print(f"  Candidate generic scaffolds: {len(cand_generic_scaffolds)}")

        for _, row in mol_data.iterrows():
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    generic = MurckoScaffold.MakeScaffoldGeneric(core)
                    gen_smi = Chem.MolToSmiles(generic)
                    if gen_smi in cand_generic_scaffolds:
                        bridge_mols.append({
                            'smiles': smi,
                            'pIC50': row['pIC50'],
                            'scaffold': gen_smi,
                        })
                except:
                    pass

        if len(bridge_mols) > 0:
            bridge_df = pd.DataFrame(bridge_mols)
            bridge_df = bridge_df[~bridge_df['smiles'].isin(train_set | test_set)]
            bridge_df = bridge_df.drop_duplicates(subset='smiles')
        else:
            bridge_df = pd.DataFrame(columns=['smiles', 'pIC50', 'scaffold'])
        print(f"  Found {len(bridge_df)} bridge molecules with generic scaffolds")

    if len(bridge_df) == 0:
        print("  No bridge molecules found. Skipping.")
        return None

    # Add embeddings
    for smi in bridge_df['smiles'].unique():
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)

    result = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        label="Scaffold_Bridge",
        extra_train_smiles=bridge_df['smiles'].tolist(),
        extra_train_y=bridge_df['pIC50'].values.astype(np.float32),
    )

    result['n_bridge_molecules'] = len(bridge_df)
    result['n_scaffolds_matched'] = bridge_df['scaffold'].nunique()
    save_result('scaffold_bridge', result)
    return result


# ====================================================================
# Approach 4: Curriculum Transfer Learning
# ====================================================================

def pretrain_on_diverse_pairs(emb_dict, max_pairs=300000, seed=42):
    """
    Phase 1 of curriculum: pretrain on diverse MMP pairs from all targets.
    Uses within-assay pairs from shared_pairs dataset.
    """
    print(f"\n  Phase 1: Pretraining on diverse MMP pairs...")

    pairs = []
    chunk_size = 200000

    for chunk in pd.read_csv(SHARED_PAIRS, chunksize=chunk_size):
        within = chunk[chunk['is_within_assay'] == True]
        pairs.append(within[['mol_a', 'mol_b', 'delta', 'target_chembl_id']].sample(
            n=min(len(within), max_pairs // 5), random_state=seed))

        total = sum(len(p) for p in pairs)
        if total >= max_pairs:
            break

    all_pairs = pd.concat(pairs, ignore_index=True)
    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs.sample(n=max_pairs, random_state=seed)

    print(f"    Loaded {len(all_pairs)} diverse pairs across "
          f"{all_pairs['target_chembl_id'].nunique()} targets")

    # Ensure embeddings
    all_smiles = set(all_pairs['mol_a'].tolist() + all_pairs['mol_b'].tolist())
    new_count = 0
    for smi in all_smiles:
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)
            new_count += 1
    print(f"    Computed {new_count} new embeddings")

    # Train/val split
    n_val = max(int(len(all_pairs) * 0.1), 1000)
    np.random.seed(seed)
    val_idx = np.random.choice(len(all_pairs), size=n_val, replace=False)
    train_mask = np.ones(len(all_pairs), dtype=bool)
    train_mask[val_idx] = False

    model, scaler = train_film_delta(
        all_pairs[train_mask], all_pairs[~train_mask], emb_dict, seed,
        lr=1e-3, epochs=60, patience=8,
        label="Diverse Pretrain"
    )

    state = {k: v.clone() for k, v in model.state_dict().items()}
    del model, scaler
    gc.collect()

    return state


def run_curriculum_transfer(
    train_smiles, train_y, test_smiles, test_y, emb_dict,
    kinase_pretrained_state=None
):
    """
    Approach 4: Curriculum transfer learning.
    Phase 1: Diverse pairs → Phase 2: Kinase pairs → Phase 3: ZAP70
    """
    print("\n" + "=" * 70)
    print("APPROACH 4: Curriculum Transfer Learning")
    print("=" * 70)

    # Phase 1: Diverse pretraining
    diverse_state = pretrain_on_diverse_pairs(emb_dict, max_pairs=100000)
    torch.save(diverse_state, RESULTS_DIR / "diverse_pretrained_state.pt")

    # Phase 2: Fine-tune on kinase (if kinase state not provided)
    if kinase_pretrained_state is None:
        kinase_ids = get_kinase_target_ids()
        raw = pd.read_csv(RAW_FILE, usecols=['target_chembl_id', 'target_name'])
        target_names = raw.groupby('target_chembl_id')['target_name'].first()
        for tid, tname in target_names.items():
            if tname and ('kinase' in str(tname).lower() or 'ZAP' in str(tname)):
                kinase_ids.add(tid)

        kinase_pairs = load_kinase_pairs(kinase_ids, max_pairs=100000)

        if len(kinase_pairs) > 0:
            # Start from diverse pretrained state
            all_smiles = set(kinase_pairs['mol_a'].tolist() + kinase_pairs['mol_b'].tolist())
            for smi in all_smiles:
                if smi not in emb_dict:
                    emb_dict[smi] = compute_morgan_fp(smi)

            n_val = max(int(len(kinase_pairs) * 0.1), 500)
            np.random.seed(42)
            val_idx = np.random.choice(len(kinase_pairs), size=n_val, replace=False)
            train_mask = np.ones(len(kinase_pairs), dtype=bool)
            train_mask[val_idx] = False

            model, _ = train_film_delta(
                kinase_pairs[train_mask], kinase_pairs[~train_mask], emb_dict, 42,
                pretrained_state=diverse_state,
                lr=5e-4, epochs=40, patience=8,
                label="Kinase Fine-tune (from diverse)"
            )
            kinase_pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
            del model
            gc.collect()

    # Phase 3: Fine-tune on ZAP70 with different strategies
    results = {}

    # Variant A: Diverse → ZAP70
    result_a = run_extrapolation_test(
        train_smiles, train_y, test_smiles, test_y, emb_dict,
        pretrained_state=diverse_state,
        lr=5e-4,
        label="Curriculum_Diverse_ZAP70",
    )
    results['diverse_then_zap70'] = result_a
    save_result('curriculum_diverse_zap70', result_a)

    # Variant B: Diverse → Kinase → ZAP70
    if kinase_pretrained_state is not None:
        result_b = run_extrapolation_test(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            pretrained_state=kinase_pretrained_state,
            lr=3e-4,
            label="Curriculum_Diverse_Kinase_ZAP70",
        )
        results['diverse_kinase_zap70'] = result_b
        save_result('curriculum_diverse_kinase_zap70', result_b)

        # Variant C: With very low LR
        result_c = run_extrapolation_test(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            pretrained_state=kinase_pretrained_state,
            lr=1e-4,
            label="Curriculum_DKZ_LowLR",
        )
        results['diverse_kinase_zap70_lowlr'] = result_c
        save_result('curriculum_dkz_lowlr', result_c)

    return results


# ====================================================================
# Approach 5: Similarity-Weighted Anchor Prediction
# ====================================================================

def similarity_weighted_anchor_predict(model, train_smiles, train_y, test_smiles, emb_dict, scaler, power=2):
    """
    Anchor-based prediction weighted by Tanimoto similarity.
    Instead of median, use weighted average where weights = Tc^power.
    Gives more influence to more similar anchors.
    """
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

    n_train = len(train_smiles)
    n_test = len(test_smiles)

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(
        np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

    # Compute Tanimoto similarities between test and train molecules
    train_fps = [compute_morgan_fp_obj(s) for s in train_smiles]
    test_fps = [compute_morgan_fp_obj(s) for s in test_smiles]

    preds = np.zeros(n_test)
    model.eval()
    with torch.no_grad():
        for j in range(n_test):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (n_train, 1)))
            deltas = model(a, b).numpy().flatten()
            abs_preds = train_y + deltas

            # Compute Tanimoto weights
            if test_fps[j] is not None:
                tcs = np.array([
                    DataStructs.TanimotoSimilarity(test_fps[j], tfp) if tfp is not None else 0
                    for tfp in train_fps
                ])
            else:
                tcs = np.ones(n_train) / n_train

            weights = tcs ** power
            w_sum = weights.sum()
            if w_sum > 0:
                preds[j] = np.sum(weights * abs_preds) / w_sum
            else:
                preds[j] = np.median(abs_preds)

    return preds


def run_similarity_weighted(train_smiles, train_y, test_smiles, test_y, emb_dict):
    """Approach 5: Use similarity-weighted anchor prediction."""
    print("\n" + "=" * 70)
    print("APPROACH 5: Similarity-Weighted Anchor Prediction")
    print("=" * 70)

    n_seeds = N_SEEDS
    results = {}

    for power in [1, 2, 4]:
        label = f"SimWeighted_p{power}"
        print(f"\n  Running {label} ({n_seeds} seeds)...")

        all_preds = np.zeros((n_seeds, len(test_smiles)))

        for seed_i in range(n_seeds):
            t0 = time.time()
            actual_seed = seed_i * 17 + 5

            pairs = generate_all_pairs(train_smiles, train_y)
            n_val = max(int(len(pairs) * 0.1), 100)
            val_pairs = pairs.sample(n=n_val, random_state=actual_seed)
            trn_pairs = pairs.drop(val_pairs.index)

            model, scaler = train_film_delta(
                trn_pairs, val_pairs, emb_dict, actual_seed,
                label=label,
            )

            preds = similarity_weighted_anchor_predict(
                model, train_smiles, train_y, test_smiles, emb_dict, scaler,
                power=power
            )
            all_preds[seed_i] = preds

            mae = np.mean(np.abs(preds - test_y))
            spr = scipy_stats.spearmanr(preds, test_y).correlation if len(test_y) > 3 else 0
            elapsed = time.time() - t0
            print(f"    Seed {seed_i+1}/{n_seeds}: MAE={mae:.3f}, Spr={spr:.3f} ({elapsed:.0f}s)")

            del model
            gc.collect()

        mean_preds = all_preds.mean(axis=0)
        metrics = evaluate_predictions(test_y, mean_preds)
        per_seed_mae = [float(np.mean(np.abs(all_preds[s] - test_y))) for s in range(n_seeds)]
        per_seed_spr = [float(scipy_stats.spearmanr(all_preds[s], test_y).correlation)
                        for s in range(n_seeds)]

        result = {
            'label': label,
            'metrics': metrics,
            'per_seed_mae': per_seed_mae,
            'per_seed_mae_mean': float(np.mean(per_seed_mae)),
            'per_seed_mae_std': float(np.std(per_seed_mae)),
            'per_seed_spr': per_seed_spr,
            'per_seed_spr_mean': float(np.mean(per_seed_spr)),
            'per_seed_spr_std': float(np.std(per_seed_spr)),
            'predictions': mean_preds.tolist(),
            'pred_stds': all_preds.std(axis=0).tolist(),
            'power': power,
        }

        print(f"    {label} ensemble: MAE={metrics['mae']:.3f}, Spr={metrics['spearman']:.3f}")
        results[f'power_{power}'] = result
        save_result(f'sim_weighted_p{power}', result)

    return results


# ====================================================================
# Approach 6: Top-K Anchor Prediction
# ====================================================================

def topk_anchor_predict(model, train_smiles, train_y, test_smiles, emb_dict, scaler, k=20):
    """
    Use only the K most similar training molecules as anchors.
    """
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

    n_train = len(train_smiles)
    n_test = len(test_smiles)

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(
        np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

    train_fps = [compute_morgan_fp_obj(s) for s in train_smiles]
    test_fps = [compute_morgan_fp_obj(s) for s in test_smiles]

    preds = np.zeros(n_test)
    model.eval()
    with torch.no_grad():
        for j in range(n_test):
            # Compute similarities
            if test_fps[j] is not None:
                tcs = np.array([
                    DataStructs.TanimotoSimilarity(test_fps[j], tfp) if tfp is not None else 0
                    for tfp in train_fps
                ])
            else:
                tcs = np.ones(n_train) / n_train

            # Top-K indices
            topk_idx = np.argsort(-tcs)[:k]

            a = torch.FloatTensor(train_embs[topk_idx])
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (len(topk_idx), 1)))
            deltas = model(a, b).numpy().flatten()
            abs_preds = train_y[topk_idx] + deltas

            # Weighted average by similarity
            weights = tcs[topk_idx] ** 2
            w_sum = weights.sum()
            if w_sum > 0:
                preds[j] = np.sum(weights * abs_preds) / w_sum
            else:
                preds[j] = np.median(abs_preds)

    return preds


def run_topk_anchor(train_smiles, train_y, test_smiles, test_y, emb_dict):
    """Approach 6: Use only top-K most similar anchors."""
    print("\n" + "=" * 70)
    print("APPROACH 6: Top-K Anchor Prediction")
    print("=" * 70)

    n_seeds = N_SEEDS
    results = {}

    for k in [10, 20, 50]:
        label = f"TopK_{k}"
        print(f"\n  Running {label} ({n_seeds} seeds)...")

        all_preds = np.zeros((n_seeds, len(test_smiles)))

        for seed_i in range(n_seeds):
            t0 = time.time()
            actual_seed = seed_i * 17 + 5

            pairs = generate_all_pairs(train_smiles, train_y)
            n_val = max(int(len(pairs) * 0.1), 100)
            val_pairs = pairs.sample(n=n_val, random_state=actual_seed)
            trn_pairs = pairs.drop(val_pairs.index)

            model, scaler = train_film_delta(
                trn_pairs, val_pairs, emb_dict, actual_seed,
                label=label,
            )

            preds = topk_anchor_predict(
                model, train_smiles, train_y, test_smiles, emb_dict, scaler, k=k
            )
            all_preds[seed_i] = preds

            mae = np.mean(np.abs(preds - test_y))
            spr = scipy_stats.spearmanr(preds, test_y).correlation if len(test_y) > 3 else 0
            elapsed = time.time() - t0
            print(f"    Seed {seed_i+1}/{n_seeds}: MAE={mae:.3f}, Spr={spr:.3f} ({elapsed:.0f}s)")

            del model
            gc.collect()

        mean_preds = all_preds.mean(axis=0)
        metrics = evaluate_predictions(test_y, mean_preds)
        per_seed_mae = [float(np.mean(np.abs(all_preds[s] - test_y))) for s in range(n_seeds)]
        per_seed_spr = [float(scipy_stats.spearmanr(all_preds[s], test_y).correlation)
                        for s in range(n_seeds)]

        result = {
            'label': label,
            'metrics': metrics,
            'per_seed_mae': per_seed_mae,
            'per_seed_mae_mean': float(np.mean(per_seed_mae)),
            'per_seed_mae_std': float(np.std(per_seed_mae)),
            'per_seed_spr': per_seed_spr,
            'per_seed_spr_mean': float(np.mean(per_seed_spr)),
            'per_seed_spr_std': float(np.std(per_seed_spr)),
            'predictions': mean_preds.tolist(),
            'pred_stds': all_preds.std(axis=0).tolist(),
            'k': k,
        }

        print(f"    {label} ensemble: MAE={metrics['mae']:.3f}, Spr={metrics['spearman']:.3f}")
        results[f'k_{k}'] = result
        save_result(f'topk_{k}', result)

    return results


# ====================================================================
# Approach 7: Antisymmetric Augmentation + Regularization
# ====================================================================

def run_antisymmetric(train_smiles, train_y, test_smiles, test_y, emb_dict):
    """
    Approach 7: Train with antisymmetric augmentation and/or regularization.
    Enforces f(A->B) = -f(B->A), which improves consistency.
    """
    print("\n" + "=" * 70)
    print("APPROACH 7: Antisymmetric Training")
    print("=" * 70)

    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

    n_seeds = N_SEEDS
    results = {}

    for variant, antisym_aug, antisym_reg in [
        ('aug_only', True, 0.0),
        ('reg_only', False, 0.1),
        ('aug_and_reg', True, 0.1),
    ]:
        label = f"Antisym_{variant}"
        print(f"\n  Running {label} ({n_seeds} seeds)...")

        all_preds = np.zeros((n_seeds, len(test_smiles)))

        for seed_i in range(n_seeds):
            t0 = time.time()
            actual_seed = seed_i * 17 + 5
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)

            pairs = generate_all_pairs(train_smiles, train_y)
            n_val = max(int(len(pairs) * 0.1), 100)
            val_pairs = pairs.sample(n=n_val, random_state=actual_seed)
            trn_pairs = pairs.drop(val_pairs.index)

            def get_emb(smi):
                return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

            train_a = np.array([get_emb(s) for s in trn_pairs['mol_a']])
            train_b = np.array([get_emb(s) for s in trn_pairs['mol_b']])
            train_yp = trn_pairs['delta'].values.astype(np.float32)
            val_a = np.array([get_emb(s) for s in val_pairs['mol_a']])
            val_b = np.array([get_emb(s) for s in val_pairs['mol_b']])
            val_yp = val_pairs['delta'].values.astype(np.float32)

            # Standardize
            scaler = StandardScaler()
            scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
            train_a = scaler.transform(train_a).astype(np.float32)
            train_b = scaler.transform(train_b).astype(np.float32)
            val_a = scaler.transform(val_a).astype(np.float32)
            val_b = scaler.transform(val_b).astype(np.float32)

            predictor = FiLMDeltaPredictor(
                hidden_dims=HIDDEN_DIMS, dropout=0.2,
                learning_rate=LR, batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                device=DEVICE,
            )
            predictor.fit(
                train_a, train_b, train_yp,
                val_a, val_b, val_yp,
                verbose=False,
                antisymmetric_aug=antisym_aug,
                antisym_reg_weight=antisym_reg,
            )

            # Anchor prediction
            train_embs_scaled = scaler.transform(
                np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
            test_embs_scaled = scaler.transform(
                np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

            preds_j = np.zeros(len(test_smiles))
            for j in range(len(test_smiles)):
                pred_deltas = predictor.predict(
                    train_embs_scaled,
                    np.tile(test_embs_scaled[j:j+1], (len(train_smiles), 1))
                )
                preds_j[j] = np.median(train_y + pred_deltas)

            all_preds[seed_i] = preds_j

            mae = np.mean(np.abs(preds_j - test_y))
            spr = scipy_stats.spearmanr(preds_j, test_y).correlation if len(test_y) > 3 else 0
            elapsed = time.time() - t0
            print(f"    Seed {seed_i+1}/{n_seeds}: MAE={mae:.3f}, Spr={spr:.3f} ({elapsed:.0f}s)")

            del predictor
            gc.collect()

        mean_preds = all_preds.mean(axis=0)
        metrics = evaluate_predictions(test_y, mean_preds)
        per_seed_mae = [float(np.mean(np.abs(all_preds[s] - test_y))) for s in range(n_seeds)]
        per_seed_spr = [float(scipy_stats.spearmanr(all_preds[s], test_y).correlation)
                        for s in range(n_seeds)]

        result = {
            'label': label,
            'metrics': metrics,
            'per_seed_mae': per_seed_mae,
            'per_seed_mae_mean': float(np.mean(per_seed_mae)),
            'per_seed_spr': per_seed_spr,
            'per_seed_spr_mean': float(np.mean(per_seed_spr)),
            'predictions': mean_preds.tolist(),
            'pred_stds': all_preds.std(axis=0).tolist(),
        }

        print(f"    {label} ensemble: MAE={metrics['mae']:.3f}, Spr={metrics['spearman']:.3f}")
        results[variant] = result
        save_result(f'antisym_{variant}', result)

    return results


# ====================================================================
# Candidate scoring
# ====================================================================

def score_candidates(
    train_smiles, train_y, emb_dict, candidate_smiles,
    pretrained_state=None, label="baseline", n_seeds=N_SEEDS,
    extra_train_smiles=None, extra_train_y=None,
    lr=LR,
):
    """Score the 19 candidates using a trained model."""
    print(f"\n  Scoring 19 candidates with {label}...")

    effective_train_smiles = list(train_smiles)
    effective_train_y = np.array(train_y, dtype=np.float32)

    if extra_train_smiles is not None:
        effective_train_smiles = list(train_smiles) + list(extra_train_smiles)
        effective_train_y = np.concatenate([train_y, extra_train_y])

    for smi in candidate_smiles:
        if smi not in emb_dict:
            emb_dict[smi] = compute_morgan_fp(smi)

    all_preds = np.zeros((n_seeds, len(candidate_smiles)))

    for seed_i in range(n_seeds):
        actual_seed = seed_i * 17 + 5

        pairs = generate_all_pairs(effective_train_smiles, effective_train_y)
        n_val = max(int(len(pairs) * 0.1), 100)
        val_pairs = pairs.sample(n=n_val, random_state=actual_seed)
        trn_pairs = pairs.drop(val_pairs.index)

        model, scaler = train_film_delta(
            trn_pairs, val_pairs, emb_dict, actual_seed,
            pretrained_state=pretrained_state,
            lr=lr, label=label,
        )

        preds = anchor_predict(
            model, train_smiles, train_y,
            candidate_smiles, emb_dict, scaler
        )
        all_preds[seed_i] = preds

        del model
        gc.collect()

    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)

    return mean_preds, std_preds


# ====================================================================
# Save / report
# ====================================================================

def save_result(name, result):
    """Save a single result to JSON."""
    path = RESULTS_DIR / f"{name}_result.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved {path}")


def save_all_results(all_results):
    """Save all results to a comprehensive JSON file."""
    path = RESULTS_DIR / "enrichment_results.json"
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved all results to {path}")


def generate_html_report(all_results, test_smiles, test_y, test_tc, candidate_df):
    """Generate comprehensive HTML report."""
    html_parts = []

    html_parts.append("""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <title>Enrichment Pretraining Report</title>
    <style>
    body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #2980b9; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
    th { background-color: #3498db; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .best { background-color: #d4edda !important; font-weight: bold; }
    .worse { background-color: #f8d7da !important; }
    .metric { font-family: monospace; }
    .summary-box { background: #eef; border: 2px solid #88a; border-radius: 8px; padding: 15px; margin: 15px 0; }
    </style></head><body>
    """)

    html_parts.append("<h1>Enrichment Pretraining for FiLMDelta Extrapolation</h1>")
    html_parts.append(f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M')}</p>")

    # Summary table
    html_parts.append("<h2>Approach Comparison</h2>")
    html_parts.append("<table><tr><th>Approach</th><th>MAE</th><th>Spearman</th>"
                       "<th>Pearson</th><th>R2</th><th>vs Baseline MAE</th></tr>")

    baseline_mae = None
    for name, result in all_results.items():
        if result is None:
            continue
        # Handle nested results (approaches that return dicts of variants)
        if isinstance(result, dict) and 'metrics' in result:
            mae = result['metrics']['mae']
            spr = result['metrics']['spearman']
            prs = result['metrics']['pearson']
            r2 = result['metrics']['r2']

            if name == 'baseline':
                baseline_mae = mae

            delta_str = ""
            css_class = ""
            if baseline_mae is not None and name != 'baseline':
                delta = (1 - mae / baseline_mae) * 100
                delta_str = f"{delta:+.1f}%"
                css_class = ' class="best"' if delta > 0 else ' class="worse"'

            html_parts.append(
                f"<tr{css_class}><td>{result.get('label', name)}</td>"
                f'<td class="metric">{mae:.3f}</td>'
                f'<td class="metric">{spr:.3f}</td>'
                f'<td class="metric">{prs:.3f}</td>'
                f'<td class="metric">{r2:.3f}</td>'
                f'<td class="metric">{delta_str}</td></tr>'
            )
        elif isinstance(result, dict):
            # Nested results (e.g., kinase pretraining variants)
            for subname, subresult in result.items():
                if subresult is None or not isinstance(subresult, dict) or 'metrics' not in subresult:
                    continue
                mae = subresult['metrics']['mae']
                spr = subresult['metrics']['spearman']
                prs = subresult['metrics']['pearson']
                r2 = subresult['metrics']['r2']

                delta_str = ""
                css_class = ""
                if baseline_mae is not None:
                    delta = (1 - mae / baseline_mae) * 100
                    delta_str = f"{delta:+.1f}%"
                    css_class = ' class="best"' if delta > 0 else ' class="worse"'

                html_parts.append(
                    f"<tr{css_class}><td>{subresult.get('label', subname)}</td>"
                    f'<td class="metric">{mae:.3f}</td>'
                    f'<td class="metric">{spr:.3f}</td>'
                    f'<td class="metric">{prs:.3f}</td>'
                    f'<td class="metric">{r2:.3f}</td>'
                    f'<td class="metric">{delta_str}</td></tr>'
                )

    html_parts.append("</table>")

    # Candidate predictions comparison
    if 'candidate_scores' in all_results:
        html_parts.append("<h2>Candidate Predictions Comparison</h2>")
        html_parts.append("<table><tr><th>Mol</th><th>Max Tc</th>")
        for approach_name in all_results['candidate_scores']:
            html_parts.append(f"<th>{approach_name}<br>pIC50</th><th>Std</th>")
        html_parts.append("</tr>")

        for i in range(len(candidate_df)):
            row = candidate_df.iloc[i]
            html_parts.append(f"<tr><td>{row['idx']}</td>"
                            f'<td class="metric">{row["max_tanimoto"]:.3f}</td>')
            for approach_name, scores in all_results['candidate_scores'].items():
                pred = scores['predictions'][i]
                std = scores['stds'][i]
                html_parts.append(f'<td class="metric">{pred:.2f}</td>'
                                f'<td class="metric">{std:.2f}</td>')
            html_parts.append("</tr>")
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    report_path = RESULTS_DIR / "enrichment_report.html"
    with open(report_path, 'w') as f:
        f.write("\n".join(html_parts))
    print(f"  HTML report saved to {report_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("ENRICHMENT PRETRAINING FOR FILMDELTA EXTRAPOLATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"N seeds: {N_SEEDS}")
    print(f"TC threshold: {TC_THRESHOLD}")

    # Load candidates
    candidate_df = load_candidates()
    candidate_smiles = candidate_df['smiles'].tolist()
    print(f"\n19 Candidates: max Tc range {candidate_df['max_tanimoto'].min():.3f} - "
          f"{candidate_df['max_tanimoto'].max():.3f}")

    # Setup extrapolation test
    (train_smiles, train_y, test_smiles, test_y,
     test_tc, emb_dict, naive_mae) = setup_extrapolation_test()

    all_results = {
        'metadata': {
            'n_train': len(train_smiles),
            'n_test': len(test_smiles),
            'n_candidates': len(candidate_smiles),
            'tc_threshold': TC_THRESHOLD,
            'n_seeds': N_SEEDS,
            'naive_baseline_mae': naive_mae,
            'device': DEVICE,
            'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        },
        'test_molecules': {
            'smiles': test_smiles,
            'pIC50': test_y.tolist(),
            'max_tc_to_train': test_tc.tolist(),
        },
    }

    # ============================================================
    # Approach 0: Baseline (try to resume from saved)
    # ============================================================
    baseline_result_path = RESULTS_DIR / "baseline_result.json"
    if baseline_result_path.exists():
        with open(baseline_result_path) as f:
            baseline_result = json.load(f)
        print(f"\n  Resumed baseline: MAE={baseline_result['metrics']['mae']:.3f}, "
              f"Spr={baseline_result['metrics']['spearman']:.3f}")
    else:
        baseline_result = run_baseline(
            train_smiles, train_y, test_smiles, test_y, emb_dict
        )
    all_results['baseline'] = baseline_result
    save_all_results(all_results)

    # ============================================================
    # Approach 1: Chemical Neighbor Enrichment
    # ============================================================
    chem_result_path = RESULTS_DIR / "chemical_neighbors_result.json"
    if chem_result_path.exists():
        with open(chem_result_path) as f:
            neighbor_result = json.load(f)
        print(f"\n  Resumed chemical neighbors: MAE={neighbor_result['metrics']['mae']:.3f}")
    else:
        neighbor_result = run_chemical_neighbor_enrichment(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            candidate_smiles
        )
    all_results['chemical_neighbors'] = neighbor_result
    save_all_results(all_results)

    # ============================================================
    # Approach 2: Kinase Panel Pretraining
    # ============================================================
    kinase_ft_path = RESULTS_DIR / "kinase_pretrain_finetune_all_result.json"
    if kinase_ft_path.exists():
        with open(kinase_ft_path) as f:
            kinase_results = {'finetune_all': json.load(f)}
        # Also try loading other variants
        for var in ['kinase_pretrain_frozen_delta', 'kinase_pretrain_low_lr']:
            vp = RESULTS_DIR / f"{var}_result.json"
            if vp.exists():
                with open(vp) as f:
                    kinase_results[var.replace('kinase_pretrain_', 'finetune_')] = json.load(f)
        print(f"\n  Resumed kinase pretraining results")
    else:
        kinase_results = run_kinase_pretraining(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            max_kinase_pairs=100000
        )
    all_results['kinase_pretraining'] = kinase_results
    save_all_results(all_results)

    # ============================================================
    # Approach 3: Scaffold Bridging
    # ============================================================
    scaffold_path = RESULTS_DIR / "scaffold_bridge_result.json"
    if scaffold_path.exists():
        with open(scaffold_path) as f:
            scaffold_result = json.load(f)
        print(f"\n  Resumed scaffold bridge results")
    else:
        scaffold_result = run_scaffold_bridging(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            candidate_smiles
        )
    all_results['scaffold_bridge'] = scaffold_result
    save_all_results(all_results)

    # ============================================================
    # Approach 4: Curriculum Transfer Learning
    # ============================================================
    curriculum_path = RESULTS_DIR / "curriculum_diverse_zap70_result.json"
    if curriculum_path.exists():
        curriculum_results = {}
        for var in ['curriculum_diverse_zap70', 'curriculum_diverse_kinase_zap70', 'curriculum_dkz_lowlr']:
            vp = RESULTS_DIR / f"{var}_result.json"
            if vp.exists():
                with open(vp) as f:
                    key = var.replace('curriculum_', '')
                    curriculum_results[key] = json.load(f)
        print(f"\n  Resumed curriculum results")
    else:
        # Reuse kinase pretrained state if available
        kinase_state = None
        kinase_state_path = RESULTS_DIR / "kinase_pretrained_state.pt"
        if kinase_state_path.exists():
            kinase_state = torch.load(kinase_state_path, map_location='cpu')

        curriculum_results = run_curriculum_transfer(
            train_smiles, train_y, test_smiles, test_y, emb_dict,
            kinase_pretrained_state=kinase_state
        )
    all_results['curriculum'] = curriculum_results
    save_all_results(all_results)

    # ============================================================
    # Approach 5: Similarity-Weighted Anchor Prediction
    # ============================================================
    sim_path = RESULTS_DIR / "sim_weighted_p2_result.json"
    if sim_path.exists():
        sim_results = {}
        for p in [1, 2, 4]:
            vp = RESULTS_DIR / f"sim_weighted_p{p}_result.json"
            if vp.exists():
                with open(vp) as f:
                    sim_results[f'power_{p}'] = json.load(f)
        print(f"\n  Resumed similarity-weighted results")
    else:
        sim_results = run_similarity_weighted(
            train_smiles, train_y, test_smiles, test_y, emb_dict
        )
    all_results['similarity_weighted'] = sim_results
    save_all_results(all_results)

    # ============================================================
    # Approach 6: Top-K Anchor Prediction
    # ============================================================
    topk_path = RESULTS_DIR / "topk_20_result.json"
    if topk_path.exists():
        topk_results = {}
        for k in [10, 20, 50]:
            vp = RESULTS_DIR / f"topk_{k}_result.json"
            if vp.exists():
                with open(vp) as f:
                    topk_results[f'k_{k}'] = json.load(f)
        print(f"\n  Resumed top-K results")
    else:
        topk_results = run_topk_anchor(
            train_smiles, train_y, test_smiles, test_y, emb_dict
        )
    all_results['topk_anchor'] = topk_results
    save_all_results(all_results)

    # ============================================================
    # Approach 7: Antisymmetric Training
    # ============================================================
    antisym_path = RESULTS_DIR / "antisym_aug_only_result.json"
    if antisym_path.exists():
        antisym_results = {}
        for var in ['aug_only', 'reg_only', 'aug_and_reg']:
            vp = RESULTS_DIR / f"antisym_{var}_result.json"
            if vp.exists():
                with open(vp) as f:
                    antisym_results[var] = json.load(f)
        print(f"\n  Resumed antisymmetric results")
    else:
        antisym_results = run_antisymmetric(
            train_smiles, train_y, test_smiles, test_y, emb_dict
        )
    all_results['antisymmetric'] = antisym_results
    save_all_results(all_results)

    # ============================================================
    # Find best approach and score candidates
    # ============================================================
    print("\n" + "=" * 70)
    print("SCORING 19 CANDIDATES WITH BEST APPROACHES")
    print("=" * 70)

    # Collect all approaches and their MAEs
    approach_maes = []

    if baseline_result:
        approach_maes.append(('baseline', baseline_result['metrics']['mae'], baseline_result, None, LR))

    if neighbor_result:
        approach_maes.append(('chem_neighbor', neighbor_result['metrics']['mae'], neighbor_result, None, LR))

    if kinase_results:
        for key, res in kinase_results.items():
            if res and 'metrics' in res:
                state = torch.load(RESULTS_DIR / "kinase_pretrained_state.pt", map_location='cpu')
                lr_map = {'finetune_all': 5e-4, 'finetune_frozen_delta': 5e-4, 'finetune_low_lr': 1e-4}
                approach_maes.append((f'kinase_{key}', res['metrics']['mae'], res, state, lr_map.get(key, 5e-4)))

    if curriculum_results:
        for key, res in curriculum_results.items():
            if res and isinstance(res, dict) and 'metrics' in res:
                if 'kinase' in key:
                    state = kinase_state if 'kinase_state' in dir() else None
                else:
                    state = torch.load(RESULTS_DIR / "diverse_pretrained_state.pt", map_location='cpu') if (RESULTS_DIR / "diverse_pretrained_state.pt").exists() else None
                lr_map = {'diverse_then_zap70': 5e-4, 'diverse_kinase_zap70': 3e-4, 'diverse_kinase_zap70_lowlr': 1e-4}
                approach_maes.append((f'curriculum_{key}', res['metrics']['mae'], res, state, lr_map.get(key, 3e-4)))

    # Add similarity-weighted and top-K results
    if sim_results:
        for key, res in sim_results.items():
            if res and isinstance(res, dict) and 'metrics' in res:
                approach_maes.append((f'sim_weighted_{key}', res['metrics']['mae'], res, None, LR))

    if topk_results:
        for key, res in topk_results.items():
            if res and isinstance(res, dict) and 'metrics' in res:
                approach_maes.append((f'topk_{key}', res['metrics']['mae'], res, None, LR))

    if antisym_results:
        for key, res in antisym_results.items():
            if res and isinstance(res, dict) and 'metrics' in res:
                approach_maes.append((f'antisym_{key}', res['metrics']['mae'], res, None, LR))

    # Sort by MAE
    approach_maes.sort(key=lambda x: x[1])

    print("\n  Ranking of approaches by MAE:")
    for i, (name, mae, _, _, _) in enumerate(approach_maes):
        print(f"    {i+1}. {name}: MAE={mae:.3f}")

    # Score candidates with top 3 approaches
    candidate_scores = {}
    n_to_score = min(3, len(approach_maes))

    # Always score baseline
    baseline_preds, baseline_stds = score_candidates(
        train_smiles, train_y, emb_dict, candidate_smiles,
        pretrained_state=None, label="baseline", n_seeds=3,
        lr=LR,
    )
    candidate_scores['baseline'] = {
        'predictions': baseline_preds.tolist(),
        'stds': baseline_stds.tolist(),
    }

    for i in range(n_to_score):
        name, mae, result, state, lr = approach_maes[i]
        if name == 'baseline':
            continue

        preds, stds = score_candidates(
            train_smiles, train_y, emb_dict, candidate_smiles,
            pretrained_state=state, label=name, n_seeds=3,
            lr=lr,
        )
        candidate_scores[name] = {
            'predictions': preds.tolist(),
            'stds': stds.tolist(),
        }

    all_results['candidate_scores'] = candidate_scores
    save_all_results(all_results)

    # ============================================================
    # Generate report
    # ============================================================
    generate_html_report(all_results, test_smiles, test_y, test_tc, candidate_df)

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"  Baseline MAE: {baseline_result['metrics']['mae']:.3f}")
    print(f"  Best approach: {approach_maes[0][0]} (MAE={approach_maes[0][1]:.3f})")
    improvement = (1 - approach_maes[0][1] / baseline_result['metrics']['mae']) * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"\n  All results saved to: {RESULTS_DIR}")
    print(f"  HTML report: {RESULTS_DIR / 'enrichment_report.html'}")

    # Print candidate predictions
    print(f"\n  Candidate predictions (best approach: {approach_maes[0][0]}):")
    best_name = approach_maes[0][0]
    if best_name in candidate_scores:
        best_preds = candidate_scores[best_name]['predictions']
        best_stds = candidate_scores[best_name]['stds']
    else:
        best_preds = baseline_preds.tolist()
        best_stds = baseline_stds.tolist()

    print(f"  {'Mol':>4s} {'MaxTc':>6s} {'Baseline':>9s} {'Best':>9s} {'Std':>6s}")
    for i in range(len(candidate_df)):
        row = candidate_df.iloc[i]
        bl_pred = candidate_scores['baseline']['predictions'][i]
        b_pred = best_preds[i]
        b_std = best_stds[i]
        print(f"  {row['idx']:4d} {row['max_tanimoto']:6.3f} {bl_pred:9.3f} {b_pred:9.3f} {b_std:6.3f}")


if __name__ == "__main__":
    main()
