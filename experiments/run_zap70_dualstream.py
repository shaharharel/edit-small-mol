#!/usr/bin/env python3
"""
ZAP70 DualStreamFiLM scoring — apply best edit-aware architecture to 19 candidates.

Trains DualStreamFiLM on ZAP70 all-pairs with DRFP + edit features,
then scores 19 candidates via anchor-based prediction.

Also includes:
- Scaffold-aware CV (Butina clustering)
- Distant-molecule CV (lowest Tanimoto test set)
- Comparison with standard FiLMDelta

Usage:
    conda run -n quris python -u experiments/run_zap70_dualstream.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.SaltRemover import SaltRemover

warnings.filterwarnings("ignore")
# MPS is fine for simple MLPs (FiLMDelta, DualStreamFiLM) — only ChemBERTa crashes on MPS
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import RESULTS_DIR, CACHE_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints, compute_absolute_metrics,
    aggregate_cv_results,
)
from src.models.predictors.edit_aware_film_predictor import DualStreamFiLMDeltaMLP
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from src.data.utils.chemistry import compute_edit_features

PROJECT_ROOT = Path(__file__).parent.parent
N_FOLDS = 5
CV_SEED = 42
BATCH_SIZE = 256  # Larger batch for MPS
MAX_EPOCHS = 150
PATIENCE = 15
# Re-enable MPS (run_zap70_v3 import disables it, but FiLMDelta/DualStream MLPs work fine on MPS)
torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SMILES_19 = [
    'C=CC(=O)N1CC=2C=CC=C(C(=O)NC3=CN(C=N3)C(C)C)C2C1',
    'C=CC(=O)N1CC2(CCC(=O)NC3=CNN=C3C(=O)NCC(C)O)CCC1CC2',
    'C=CC(=O)NC[C@H]1C[C@H]2C[C@@H]1CN2C=3N=CN=C(N)C3Cl',
    'C=CC(=O)N1CCCC1(C(=O)NC2=CNN=C2OCC(F)F)C=3C=CC=CC3',
    'C=CC(=O)N1CCC(CC(=O)NC2=CNN=C2C=3C=NC=CN3)CC41CC4',
    'C=CC(=O)N1CC2(CCC2)CC1CC(=O)NC3=CNN=C3C(=O)NCC(C)O',
    'C=CC(=O)N1C[C@H](CC(C)(C)C)[C@H](C1)C(=O)NC2=CNN=C2C(=O)NCC(C)O',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C3=CNC=4C=C(F)C(Cl)=CC34)S2',
    'C=CC(=O)NC1C2C3C[C@@H]1[C@H](C(=O)NC4=CNN=C4C=5C=NC=CN5)C32',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C=3C=CC(F)=C(C3)S(=O)(=O)N(C)C)S2',
    'C=CC(=O)NCC1=CN(N=N1)[C@@H]2C[C@H](C2)C(=O)NC=3C=CC=NC3NC(C)=O',
    'C=CC(=O)N1CC2(CC1CCC2)NC=3N=CC=C(N3)OC=4C=CC=C(C#N)C4',
    'C=CC(=O)NC(C)C=1N=CC(=CN1)NC(=O)C=2N=CN=C3NC=C(C)C23',
    'C=CC(=O)N(C)C1(CNC=2N=CN=C(N)C2C(=O)OCC)CCC1',
    'C=CC(=O)N1CCC[C@H]1C(C)NC(=O)C=2C=NNC2C=3C=CN(C)N3',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CC(CCNC=2C=C(N=CN2)NC=3C=CC=CC3)(C1)N(C)C',
    'O=C(O)C(F)(F)F.C=CC(=O)N(C)C1(CNC=2N=CN=C3NC=C(C(N)=O)C23)CCC1',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CCN(CC1)C=2N=CN=C(N2)NC3(CC)CCNCC3',
    'C=CC(=O)N1CCCC(NC(=O)C2=CNN=C2C3=CC=4C=CC=CC4O3)C51CCC5',
]


def clean_smiles(smi):
    smi_clean = smi.split(' |')[0] if ' |' in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)


def compute_drfp_for_pairs(mol_a_list, mol_b_list):
    """Compute DRFP reaction fingerprints for pairs."""
    from drfp import DrfpEncoder

    rxn_smiles = [f"{a}>>{b}" for a, b in zip(mol_a_list, mol_b_list)]
    # Process in batches
    all_fps = []
    bs = 2000
    n_total = len(rxn_smiles)
    for i in range(0, n_total, bs):
        batch = rxn_smiles[i:i+bs]
        fps = DrfpEncoder.encode(batch, n_folded_length=2048)
        all_fps.extend(fps)
        print(f"      DRFP: {min(i+bs, n_total)}/{n_total} pairs", flush=True)
    return np.array(all_fps, dtype=np.float32)


def _compute_ef_single(args):
    """Worker function for parallel edit feature computation."""
    smi_a, smi_b, es = args
    try:
        return compute_edit_features(smi_a, smi_b, es)
    except Exception:
        return np.zeros(28, dtype=np.float32)


def compute_edit_features_batch(mol_a_list, mol_b_list, edit_smiles_list=None):
    """Compute 28-dim edit features for a batch of pairs (parallelized)."""
    from multiprocessing import Pool, cpu_count
    n_total = len(mol_a_list)
    es_list = edit_smiles_list if edit_smiles_list else [""] * n_total
    args = list(zip(mol_a_list, mol_b_list, es_list))

    n_workers = min(cpu_count(), 8)
    print(f"      Computing {n_total} edit features ({n_workers} workers)...")

    if n_total < 500:
        # Small batch — serial is faster (no pool overhead)
        feats = [_compute_ef_single(a) for a in args]
    else:
        with Pool(n_workers) as pool:
            feats = pool.map(_compute_ef_single, args, chunksize=500)

    result = np.array(feats, dtype=np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"      Edit features: done ({n_total} pairs)")
    return result


def generate_all_pairs(smiles, pIC50):
    """Generate all-pairs from a set of molecules."""
    pairs = []
    for (i, j) in combinations(range(len(smiles)), 2):
        pairs.append({
            'mol_a': smiles[i], 'mol_b': smiles[j],
            'delta': pIC50[j] - pIC50[i],
        })
        pairs.append({
            'mol_a': smiles[j], 'mol_b': smiles[i],
            'delta': pIC50[i] - pIC50[j],
        })
    return pd.DataFrame(pairs)


def train_dualstream(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0,
                     precomputed=None):
    """Train DualStreamFiLM on pairs, return trained model + scaler.

    precomputed: dict with 'train_drfp', 'val_drfp', 'train_ef', 'val_ef' numpy arrays
                 If None, computes from scratch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Compute embeddings
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    # Training data
    train_a = np.array([get_emb(s) for s in train_pairs_df['mol_a']])
    train_b = np.array([get_emb(s) for s in train_pairs_df['mol_b']])
    train_y = train_pairs_df['delta'].values.astype(np.float32)

    # Validation data
    val_a = np.array([get_emb(s) for s in val_pairs_df['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs_df['mol_b']])
    val_y = val_pairs_df['delta'].values.astype(np.float32)

    # Scale embeddings
    scaler = StandardScaler()
    all_embs = np.vstack([train_a, train_b, val_a, val_b])
    scaler.fit(all_embs)
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    if precomputed is not None:
        train_drfp = precomputed['train_drfp']
        val_drfp = precomputed['val_drfp']
        train_ef = precomputed['train_ef']
        val_ef = precomputed['val_ef']
        print("    Using precomputed DRFP + edit features", flush=True)
    else:
        # Compute DRFP
        print("    Computing DRFP for training pairs...", flush=True)
        train_drfp = compute_drfp_for_pairs(
            train_pairs_df['mol_a'].tolist(), train_pairs_df['mol_b'].tolist())
        val_drfp = compute_drfp_for_pairs(
            val_pairs_df['mol_a'].tolist(), val_pairs_df['mol_b'].tolist())

        # Compute edit features
        print("    Computing edit features...", flush=True)
        train_ef = compute_edit_features_batch(
            train_pairs_df['mol_a'].tolist(), train_pairs_df['mol_b'].tolist())
        val_ef = compute_edit_features_batch(
            val_pairs_df['mol_a'].tolist(), val_pairs_df['mol_b'].tolist())

    # Tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b),
        torch.FloatTensor(train_drfp), torch.FloatTensor(train_ef),
        torch.FloatTensor(train_y))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b),
        torch.FloatTensor(val_drfp), torch.FloatTensor(val_ef),
        torch.FloatTensor(val_y))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    device = torch.device(DEVICE)
    model = DualStreamFiLMDeltaMLP(
        mol_dim=emb_dim, drfp_dim=2048, edit_feat_dim=28, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    print(f"    Training on {DEVICE}", flush=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            a, b, drfp, ef, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            pred = model(a, b, drfp, ef)
            loss = criterion(pred, y) + 0.1 * model.aux_loss(ef)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, drfp, ef, y = [t.to(device) for t in batch]
                pred = model(a, b, drfp, ef)
                val_loss += criterion(pred, y).item()
                n_val += 1

        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"    Trained {epoch+1} epochs, best val loss: {best_val_loss:.4f}")
    return model, scaler


def train_filmdelta(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0):
    """Train standard FiLMDelta for comparison."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in train_pairs_df['mol_a']])
    train_b = np.array([get_emb(s) for s in train_pairs_df['mol_b']])
    train_y = train_pairs_df['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_pairs_df['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs_df['mol_b']])
    val_y = val_pairs_df['delta'].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b),
        torch.FloatTensor(train_y))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b),
        torch.FloatTensor(val_y))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=emb_dim, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            pred = model(a, b)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(device) for t in batch]
                pred = model(a, b)
                val_loss += criterion(pred, y).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"    Trained {epoch+1} epochs, best val loss: {best_val_loss:.4f}")
    return model, scaler


def score_candidates_dualstream(model, scaler, train_smiles, train_pIC50,
                                 cand_smiles, emb_dict, emb_dim):
    """Score candidates via anchor-based prediction with DualStreamFiLM.
    Uses global PAIR_DRFP_CACHE/PAIR_EF_CACHE if available."""
    global PAIR_DRFP_CACHE, PAIR_EF_CACHE
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    cand_embs = scaler.transform(
        np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

    n_train = len(train_smiles)
    n_cand = len(cand_smiles)
    abs_preds = np.zeros(n_cand)

    model.cpu().eval()
    with torch.no_grad():
        for j in range(n_cand):
            anchor_embs = torch.FloatTensor(train_embs)
            target_embs = torch.FloatTensor(
                np.tile(cand_embs[j:j+1], (n_train, 1)))

            # Try cache first
            if PAIR_DRFP_CACHE is not None:
                drfp_arr = np.zeros((n_train, 2048), dtype=np.float32)
                ef_arr = np.zeros((n_train, 28), dtype=np.float32)
                for i in range(n_train):
                    key = (train_smiles[i], cand_smiles[j])
                    if key in PAIR_DRFP_CACHE:
                        drfp_arr[i] = PAIR_DRFP_CACHE[key]
                        ef_arr[i] = PAIR_EF_CACHE[key]
                drfp = torch.FloatTensor(drfp_arr)
                ef = torch.FloatTensor(ef_arr)
            else:
                mol_a_list = list(train_smiles)
                mol_b_list = [cand_smiles[j]] * n_train
                drfp = torch.FloatTensor(compute_drfp_for_pairs(mol_a_list, mol_b_list))
                ef = torch.FloatTensor(compute_edit_features_batch(mol_a_list, mol_b_list))

            deltas = model(anchor_embs, target_embs, drfp, ef).numpy().flatten()
            abs_preds[j] = np.median(train_pIC50 + deltas)

    return abs_preds


def score_candidates_dualstream_cached(model, scaler, train_smiles, train_pIC50,
                                        cand_smiles, emb_dict, emb_dim,
                                        cand_drfp_cache, cand_ef_cache):
    """Score candidates using precomputed DRFP + edit features."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    cand_embs = scaler.transform(
        np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

    n_train = len(train_smiles)
    n_cand = len(cand_smiles)
    abs_preds = np.zeros(n_cand)

    model.cpu().eval()
    with torch.no_grad():
        for j in range(n_cand):
            anchor_embs = torch.FloatTensor(train_embs)
            target_embs = torch.FloatTensor(
                np.tile(cand_embs[j:j+1], (n_train, 1)))
            drfp = torch.FloatTensor(cand_drfp_cache[j])
            ef = torch.FloatTensor(cand_ef_cache[j])
            deltas = model(anchor_embs, target_embs, drfp, ef).numpy().flatten()
            abs_preds[j] = np.median(train_pIC50 + deltas)

    return abs_preds


def score_candidates_filmdelta(model, scaler, train_smiles, train_pIC50,
                                cand_smiles, emb_dict, emb_dim):
    """Score candidates via anchor-based prediction with standard FiLMDelta."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    cand_embs = scaler.transform(
        np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

    n_train = len(train_smiles)
    n_cand = len(cand_smiles)
    abs_preds = np.zeros(n_cand)

    model.cpu().eval()
    with torch.no_grad():
        for j in range(n_cand):
            anchor_embs = torch.FloatTensor(train_embs)
            target_embs = torch.FloatTensor(
                np.tile(cand_embs[j:j+1], (n_train, 1)))
            deltas = model(anchor_embs, target_embs).numpy().flatten()
            abs_preds[j] = np.median(train_pIC50 + deltas)

    return abs_preds


def scaffold_cv_splits(mol_data, n_folds=5, seed=42):
    """Butina-clustering based scaffold CV splits."""
    smiles = mol_data['smiles'].tolist()
    n = len(smiles)

    # Compute Morgan FPs for clustering
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)

    # Tanimoto distance matrix
    from rdkit.ML.Cluster import Butina
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    # Butina clustering
    clusters = Butina.ClusterData(dists, n, 0.4, isDistData=True)

    # Assign molecules to folds by cluster
    np.random.seed(seed)
    cluster_order = list(range(len(clusters)))
    np.random.shuffle(cluster_order)

    fold_assignments = np.zeros(n, dtype=int)
    for ci, cluster_idx in enumerate(cluster_order):
        fold = ci % n_folds
        for mol_idx in clusters[cluster_idx]:
            fold_assignments[mol_idx] = fold

    splits = []
    for fold in range(n_folds):
        test_mask = fold_assignments == fold
        train_mask = ~test_mask
        splits.append((
            f"scaffold_fold_{fold}",
            mol_data.iloc[np.where(train_mask)[0]].copy(),
            mol_data.iloc[np.where(test_mask)[0]].copy(),
        ))
    return splits


def distant_molecule_cv_splits(mol_data, n_folds=5, test_frac=0.2, seed=42):
    """CV splits where test set has lowest Tanimoto to training set."""
    smiles = mol_data['smiles'].tolist()
    n = len(smiles)

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)

    # Compute all-pairs Tanimoto similarity
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_matrix[i] = sims

    # Average similarity to all others
    avg_sim = (sim_matrix.sum(axis=1) - 1) / (n - 1)  # exclude self

    # Sort by avg similarity (ascending = most distant first)
    np.random.seed(seed)
    sorted_idx = np.argsort(avg_sim)

    n_test = max(int(n * test_frac), 10)
    splits = []
    for fold in range(n_folds):
        # Rotate the "distant" selection
        start = (fold * n_test) % n
        test_indices = set()
        for i in range(n_test):
            test_indices.add(sorted_idx[(start + i) % n])

        train_idx = [i for i in range(n) if i not in test_indices]
        test_idx = sorted(test_indices)

        splits.append((
            f"distant_fold_{fold}",
            mol_data.iloc[train_idx].copy(),
            mol_data.iloc[test_idx].copy(),
        ))
    return splits


def build_pair_cache(all_smiles):
    """Precompute DRFP and edit features for ALL possible pairs from the molecule set.

    Returns dicts keyed by (smi_a, smi_b) → feature vector.
    """
    n = len(all_smiles)
    print(f"  Building pair cache for {n} molecules ({n*(n-1)} ordered pairs)...")

    # Generate all ordered pairs
    mol_a_list = []
    mol_b_list = []
    for i in range(n):
        for j in range(n):
            if i != j:
                mol_a_list.append(all_smiles[i])
                mol_b_list.append(all_smiles[j])

    n_pairs = len(mol_a_list)
    print(f"    Computing DRFP for {n_pairs} pairs...")
    drfp_all = compute_drfp_for_pairs(mol_a_list, mol_b_list)
    print(f"    Computing edit features for {n_pairs} pairs...")
    ef_all = compute_edit_features_batch(mol_a_list, mol_b_list)

    drfp_cache = {}
    ef_cache = {}
    for idx in range(n_pairs):
        key = (mol_a_list[idx], mol_b_list[idx])
        drfp_cache[key] = drfp_all[idx]
        ef_cache[key] = ef_all[idx]

    print(f"    Pair cache built: {len(drfp_cache)} entries")
    return drfp_cache, ef_cache


# Global pair cache
PAIR_DRFP_CACHE = None
PAIR_EF_CACHE = None


def get_precomputed_from_cache(pairs_df):
    """Look up precomputed DRFP + edit features from global cache."""
    global PAIR_DRFP_CACHE, PAIR_EF_CACHE
    if PAIR_DRFP_CACHE is None:
        return None

    mol_a_list = pairs_df['mol_a'].tolist()
    mol_b_list = pairs_df['mol_b'].tolist()
    n = len(mol_a_list)

    drfp = np.zeros((n, 2048), dtype=np.float32)
    ef = np.zeros((n, 28), dtype=np.float32)
    found = 0

    for i in range(n):
        key = (mol_a_list[i], mol_b_list[i])
        if key in PAIR_DRFP_CACHE:
            drfp[i] = PAIR_DRFP_CACHE[key]
            ef[i] = PAIR_EF_CACHE[key]
            found += 1

    if found < n * 0.9:
        return None  # Too many misses, compute from scratch

    return {'train_drfp': drfp, 'train_ef': ef}


def run_cv_evaluation(mol_data, emb_dict, emb_dim, split_fn, split_name, n_seeds=3):
    """Run CV evaluation for both DualStreamFiLM and FiLMDelta."""
    results = {}

    for method in ["DualStreamFiLM", "FiLMDelta"]:
        print(f"\n  === {method} ({split_name}) ===")
        all_fold_metrics = []

        for seed in range(n_seeds):
            splits = split_fn(mol_data, seed=CV_SEED + seed)
            fold_metrics = []

            for fold_name, train_data, test_data in splits:
                train_smiles = train_data['smiles'].tolist()
                train_y = train_data['pIC50'].values
                test_smiles = test_data['smiles'].tolist()
                test_y = test_data['pIC50'].values

                if len(test_smiles) < 5:
                    continue

                # Generate pairs from training molecules
                train_pairs = generate_all_pairs(train_smiles, train_y)

                # Split pairs into train/val (90/10)
                n_val = max(int(len(train_pairs) * 0.1), 100)
                val_pairs = train_pairs.sample(n=n_val, random_state=seed)
                trn_pairs = train_pairs.drop(val_pairs.index)

                print(f"    {fold_name} seed={seed}: {len(trn_pairs)} train, {len(val_pairs)} val pairs, {len(test_smiles)} test mols")

                if method == "DualStreamFiLM":
                    # Look up precomputed features from cache
                    trn_pre = get_precomputed_from_cache(trn_pairs)
                    val_pre = get_precomputed_from_cache(val_pairs)
                    if trn_pre and val_pre:
                        precomputed = {
                            'train_drfp': trn_pre['train_drfp'],
                            'val_drfp': val_pre['train_drfp'],
                            'train_ef': trn_pre['train_ef'],
                            'val_ef': val_pre['train_ef'],
                        }
                    else:
                        precomputed = None
                    model, scaler = train_dualstream(trn_pairs, val_pairs, emb_dict, emb_dim, seed,
                                                      precomputed=precomputed)
                    preds = score_candidates_dualstream(
                        model, scaler, train_smiles, train_y, test_smiles, emb_dict, emb_dim)
                else:
                    model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed)
                    preds = score_candidates_filmdelta(
                        model, scaler, train_smiles, train_y, test_smiles, emb_dict, emb_dim)

                metrics = compute_absolute_metrics(test_y, preds)
                fold_metrics.append(metrics)
                print(f"      MAE={metrics['mae']:.3f}, Spr={metrics['spearman_r']:.3f}")

                del model, scaler
                gc.collect()

            all_fold_metrics.extend(fold_metrics)

        agg = aggregate_cv_results(all_fold_metrics)
        results[method] = agg
        print(f"  {method} ({split_name}): MAE={agg.get('mae_mean', 0):.3f}±{agg.get('mae_std', 0):.3f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}±{agg.get('spearman_r_std', 0):.3f}")

    return results


def main():
    print("=" * 70)
    print("ZAP70 DualStreamFiLM — Scoring 19 Candidates + Challenging Splits")
    print("=" * 70)

    # Load data
    print("\n[1] Loading ZAP70 data...")
    mol_data, _ = load_zap70_molecules()
    train_smiles = mol_data['smiles'].tolist()
    train_y = mol_data['pIC50'].values

    # Clean 19 candidate SMILES
    cand_smiles = [clean_smiles(s) for s in SMILES_19]
    print(f"  19 candidates cleaned")

    # Compute Morgan FP embeddings
    print("\n[2] Computing Morgan FP embeddings...")
    all_smiles = list(set(train_smiles + cand_smiles))
    X = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smiles)}
    print(f"  Embeddings: {len(emb_dict)} molecules, {emb_dim}d")

    # =========================================================
    # PART A: Score 19 candidates (train on ALL 280 molecules)
    # =========================================================
    print("\n" + "=" * 70)
    print("[3] SCORING 19 CANDIDATES (DualStreamFiLM + FiLMDelta)")
    print("=" * 70)

    all_pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(all_pairs) * 0.1), 100)
    val_pairs = all_pairs.sample(n=n_val, random_state=42)
    trn_pairs = all_pairs.drop(val_pairs.index)
    print(f"  All-pairs: {len(all_pairs)} total, {len(trn_pairs)} train, {len(val_pairs)} val")

    candidate_results = {"candidates": []}
    n_seeds = 5

    # Precompute DRFP + edit features ONCE for all seeds
    print("\n  Precomputing DRFP for training pairs...", flush=True)
    train_drfp = compute_drfp_for_pairs(trn_pairs['mol_a'].tolist(), trn_pairs['mol_b'].tolist())
    val_drfp = compute_drfp_for_pairs(val_pairs['mol_a'].tolist(), val_pairs['mol_b'].tolist())
    print("  Precomputing edit features for training pairs...", flush=True)
    train_ef = compute_edit_features_batch(trn_pairs['mol_a'].tolist(), trn_pairs['mol_b'].tolist())
    val_ef = compute_edit_features_batch(val_pairs['mol_a'].tolist(), val_pairs['mol_b'].tolist())
    precomputed_train = {
        'train_drfp': train_drfp, 'val_drfp': val_drfp,
        'train_ef': train_ef, 'val_ef': val_ef,
    }
    print(f"  Precomputed: DRFP {train_drfp.shape}, edit_feats {train_ef.shape}", flush=True)

    # Precompute DRFP + edit features for candidate scoring (280 anchors × 19 candidates)
    print("  Precomputing DRFP + edit features for candidate scoring...", flush=True)
    cand_drfp_cache = {}
    cand_ef_cache = {}
    for j, csmi in enumerate(cand_smiles):
        mol_a_list = list(train_smiles)
        mol_b_list = [csmi] * len(train_smiles)
        cand_drfp_cache[j] = compute_drfp_for_pairs(mol_a_list, mol_b_list)
        cand_ef_cache[j] = compute_edit_features_batch(mol_a_list, mol_b_list)
        print(f"    Candidate {j+1}/19 done", flush=True)

    # DualStreamFiLM scoring (5 seeds)
    print(f"\n  Training DualStreamFiLM ({n_seeds} seeds)...", flush=True)
    ds_preds_all = np.zeros((19, n_seeds))
    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---", flush=True)
        model, scaler = train_dualstream(trn_pairs, val_pairs, emb_dict, emb_dim, seed,
                                          precomputed=precomputed_train)
        preds = score_candidates_dualstream_cached(
            model, scaler, train_smiles, train_y, cand_smiles, emb_dict, emb_dim,
            cand_drfp_cache, cand_ef_cache)
        ds_preds_all[:, seed] = preds
        print(f"  Predictions: {preds.min():.3f} to {preds.max():.3f}", flush=True)
        del model, scaler
        gc.collect()

    ds_mean = ds_preds_all.mean(axis=1)
    ds_std = ds_preds_all.std(axis=1)

    # FiLMDelta scoring (5 seeds)
    print(f"\n  Training FiLMDelta ({n_seeds} seeds)...")
    fd_preds_all = np.zeros((19, n_seeds))
    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---")
        model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed)
        preds = score_candidates_filmdelta(
            model, scaler, train_smiles, train_y, cand_smiles, emb_dict, emb_dim)
        fd_preds_all[:, seed] = preds
        print(f"  Predictions: {preds.min():.3f} to {preds.max():.3f}")
        del model, scaler
        gc.collect()

    fd_mean = fd_preds_all.mean(axis=1)
    fd_std = fd_preds_all.std(axis=1)

    # Print results
    print("\n" + "=" * 70)
    print("CANDIDATE SCORING RESULTS")
    print("=" * 70)
    print(f"{'Idx':>3} {'DS-FiLM':>8} {'DS_σ':>6} {'FiLMΔ':>8} {'FD_σ':>6} {'Diff':>7}")
    print("-" * 45)

    for j in range(19):
        diff = ds_mean[j] - fd_mean[j]
        candidate_results["candidates"].append({
            "idx": j + 1,
            "smiles": cand_smiles[j],
            "dualstream_mean": float(ds_mean[j]),
            "dualstream_std": float(ds_std[j]),
            "filmdelta_mean": float(fd_mean[j]),
            "filmdelta_std": float(fd_std[j]),
        })
        print(f"{j+1:>3} {ds_mean[j]:>8.3f} {ds_std[j]:>6.3f} {fd_mean[j]:>8.3f} {fd_std[j]:>6.3f} {diff:>+7.3f}")

    # Rankings
    ds_ranking = np.argsort(-ds_mean)
    fd_ranking = np.argsort(-fd_mean)
    print(f"\nDualStreamFiLM top-5: {[r+1 for r in ds_ranking[:5]]}")
    print(f"FiLMDelta top-5:      {[r+1 for r in fd_ranking[:5]]}")

    candidate_results["dualstream_ranking"] = [int(r+1) for r in ds_ranking]
    candidate_results["filmdelta_ranking"] = [int(r+1) for r in fd_ranking]

    # =========================================================
    # PART B: Cross-validation with challenging splits
    # =========================================================
    print("\n" + "=" * 70)
    print("[4] CHALLENGING SPLITS — CV EVALUATION")
    print("=" * 70)

    # Build pair-level DRFP + edit features cache for ALL 280 molecules
    # This avoids recomputing DRFP per fold (78K pairs computed once, reused ~45 times)
    global PAIR_DRFP_CACHE, PAIR_EF_CACHE
    PAIR_DRFP_CACHE, PAIR_EF_CACHE = build_pair_cache(train_smiles)

    cv_results = {}

    # Random CV (baseline)
    print("\n  --- Random 5-fold CV ---")
    def random_splits(mol_data, n_folds=5, seed=42):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            splits.append((f"random_fold_{i}", mol_data.iloc[train_idx].copy(), mol_data.iloc[test_idx].copy()))
        return splits
    cv_results["random"] = run_cv_evaluation(mol_data, emb_dict, emb_dim, random_splits, "random", n_seeds=3)

    # Scaffold CV
    print("\n  --- Scaffold CV (Butina clustering) ---")
    cv_results["scaffold"] = run_cv_evaluation(mol_data, emb_dict, emb_dim, scaffold_cv_splits, "scaffold", n_seeds=3)

    # Distant-molecule CV
    print("\n  --- Distant-molecule CV ---")
    cv_results["distant"] = run_cv_evaluation(mol_data, emb_dict, emb_dim, distant_molecule_cv_splits, "distant", n_seeds=3)

    candidate_results["cv_evaluation"] = cv_results

    # =========================================================
    # SAVE
    # =========================================================
    out_path = RESULTS_DIR / "19_molecules_dualstream.json"
    with open(out_path, "w") as f:
        json.dump(candidate_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("CV SUMMARY")
    print("=" * 70)
    print(f"{'Split':>15} {'DualStream MAE':>15} {'FiLMDelta MAE':>15} {'Improvement':>12}")
    print("-" * 60)
    for split_name in ["random", "scaffold", "distant"]:
        if split_name in cv_results:
            ds = cv_results[split_name].get("DualStreamFiLM", {})
            fd = cv_results[split_name].get("FiLMDelta", {})
            ds_mae = ds.get("mae_mean", 0)
            fd_mae = fd.get("mae_mean", 0)
            ds_std = ds.get("mae_std", 0)
            fd_std = fd.get("mae_std", 0)
            imp = (fd_mae - ds_mae) / fd_mae * 100 if fd_mae > 0 else 0
            print(f"{split_name:>15} {ds_mae:.3f}±{ds_std:.3f}    {fd_mae:.3f}±{fd_std:.3f}    {imp:>+.1f}%")


if __name__ == "__main__":
    main()
