#!/usr/bin/env python3
"""
Full CV evaluation: 5 methods × 3 splits for ZAP70 case study.

Methods:
  1. Subtraction baseline (independent pIC50 predictions, then subtract)
  2. FiLMDelta (no pretrain)
  3. FiLMDelta + kinase pretrain (32K kinase within-assay pairs → ZAP70 fine-tune)
  4. DualStreamFiLM (no pretrain)
  5. DualStreamFiLM + kinase pretrain

Splits: random, scaffold (Butina), distant-molecule

3 seeds × 5 folds each.

Usage:
    conda run -n quris python -u experiments/run_zap70_full_cv.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Re-enable MPS (run_zap70_v3 import disables it)
torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints, compute_absolute_metrics,
    aggregate_cv_results,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from src.models.predictors.edit_aware_film_predictor import DualStreamFiLMDeltaMLP

# Re-enable MPS after imports
torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).parent.parent
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_SEEDS = 3
N_FOLDS = 5
CV_SEED = 42

# ============================================================
# DRFP + Edit Features (for DualStream)
# ============================================================
from drfp import DrfpEncoder

def compute_drfp_for_pairs(mol_a_list, mol_b_list, n_bits=2048):
    """Compute DRFP reaction fingerprints for pairs."""
    rxn_smiles = [f"{a}>>{b}" for a, b in zip(mol_a_list, mol_b_list)]
    fps = DrfpEncoder.encode(rxn_smiles, n_folded_length=n_bits)
    return np.array(fps, dtype=np.float32)


def _compute_ef_single(args):
    smi_a, smi_b, es = args
    try:
        from src.data.utils.chemistry import compute_edit_features
        return compute_edit_features(smi_a, smi_b, es)
    except Exception:
        return np.zeros(28, dtype=np.float32)


def compute_edit_features_batch(mol_a_list, mol_b_list, edit_smiles_list=None):
    """Compute 28-dim edit features, parallelized for large batches."""
    from multiprocessing import Pool, cpu_count
    n_total = len(mol_a_list)
    if edit_smiles_list is None:
        edit_smiles_list = [None] * n_total
    args = list(zip(mol_a_list, mol_b_list, edit_smiles_list))
    n_workers = min(cpu_count(), 8)
    if n_total < 500:
        feats = [_compute_ef_single(a) for a in args]
    else:
        with Pool(n_workers) as pool:
            feats = pool.map(_compute_ef_single, args, chunksize=500)
    return np.array(feats, dtype=np.float32)


# ============================================================
# Pair generation
# ============================================================
def generate_all_pairs(smiles, pIC50):
    """Generate all ordered (i,j) pairs with delta = pIC50[j] - pIC50[i]."""
    pairs = []
    n = len(smiles)
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append({
                    'mol_a': smiles[i], 'mol_b': smiles[j],
                    'delta': float(pIC50[j] - pIC50[i]),
                    'value_a': float(pIC50[i]), 'value_b': float(pIC50[j]),
                })
    return pd.DataFrame(pairs)


# ============================================================
# Split functions
# ============================================================
def random_cv_splits(mol_data, n_folds=N_FOLDS, seed=CV_SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        splits.append((
            f"random_fold_{fold}",
            mol_data.iloc[train_idx].copy(),
            mol_data.iloc[test_idx].copy(),
        ))
    return splits


def scaffold_cv_splits(mol_data, n_folds=N_FOLDS, seed=CV_SEED):
    from rdkit.ML.Cluster import Butina
    smiles = mol_data['smiles'].tolist()
    n = len(smiles)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles]
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, 0.4, isDistData=True)
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


def distant_molecule_cv_splits(mol_data, n_folds=N_FOLDS, test_frac=0.2, seed=CV_SEED):
    smiles = mol_data['smiles'].tolist()
    n = len(smiles)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_matrix[i] = sims
    avg_sim = (sim_matrix.sum(axis=1) - 1) / (n - 1)
    np.random.seed(seed)
    sorted_idx = np.argsort(avg_sim)
    n_test = max(int(n * test_frac), 10)
    splits = []
    for fold in range(n_folds):
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


# ============================================================
# Pair cache for DualStream
# ============================================================
PAIR_DRFP_CACHE = None
PAIR_EF_CACHE = None

def build_pair_cache(all_smiles):
    global PAIR_DRFP_CACHE, PAIR_EF_CACHE
    n = len(all_smiles)
    print(f"  Building pair cache for {n} molecules ({n*(n-1)} ordered pairs)...")
    mol_a_list, mol_b_list = [], []
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
    PAIR_DRFP_CACHE = {}
    PAIR_EF_CACHE = {}
    for idx in range(n_pairs):
        key = (mol_a_list[idx], mol_b_list[idx])
        PAIR_DRFP_CACHE[key] = drfp_all[idx]
        PAIR_EF_CACHE[key] = ef_all[idx]
    print(f"    Pair cache built: {len(PAIR_DRFP_CACHE)} entries")


def get_precomputed_from_cache(pairs_df):
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
        return None
    return {'drfp': drfp, 'ef': ef}


# ============================================================
# Training functions
# ============================================================

def train_filmdelta(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0,
                    pretrained_state=None, pretrained_scaler=None):
    """Train FiLMDelta. If pretrained_state provided, fine-tune from it."""
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

    if pretrained_scaler is not None:
        scaler = pretrained_scaler
    else:
        scaler = StandardScaler()
        scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))

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
    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state)
        model.to(device)
        lr = 1e-4  # lower LR for fine-tuning
    else:
        lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(device) for t in batch]
                vl += criterion(model(a, b), y).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best_val_loss:
            best_val_loss = vl; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model, scaler


def train_dualstream(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0,
                     precomputed=None, pretrained_state=None, pretrained_scaler=None):
    """Train DualStreamFiLM. If pretrained_state provided, fine-tune from it."""
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

    if pretrained_scaler is not None:
        scaler = pretrained_scaler
    else:
        scaler = StandardScaler()
        scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))

    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    if precomputed is not None:
        train_drfp = precomputed['train_drfp']
        val_drfp = precomputed['val_drfp']
        train_ef = precomputed['train_ef']
        val_ef = precomputed['val_ef']
    else:
        train_drfp = compute_drfp_for_pairs(
            train_pairs_df['mol_a'].tolist(), train_pairs_df['mol_b'].tolist())
        val_drfp = compute_drfp_for_pairs(
            val_pairs_df['mol_a'].tolist(), val_pairs_df['mol_b'].tolist())
        train_ef = compute_edit_features_batch(
            train_pairs_df['mol_a'].tolist(), train_pairs_df['mol_b'].tolist())
        val_ef = compute_edit_features_batch(
            val_pairs_df['mol_a'].tolist(), val_pairs_df['mol_b'].tolist())

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b),
        torch.FloatTensor(train_drfp), torch.FloatTensor(train_ef),
        torch.FloatTensor(train_y))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b),
        torch.FloatTensor(val_drfp), torch.FloatTensor(val_ef),
        torch.FloatTensor(val_y))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE)
    model = DualStreamFiLMDeltaMLP(
        mol_dim=emb_dim, drfp_dim=2048, edit_feat_dim=28, dropout=0.2).to(device)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state)
        model.to(device)
        lr = 1e-4
    else:
        lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, drfp, ef, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(a, b, drfp, ef), y) + 0.1 * model.aux_loss(ef)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, drfp, ef, y = [t.to(device) for t in batch]
                vl += criterion(model(a, b, drfp, ef), y).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best_val_loss:
            best_val_loss = vl; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model, scaler


# ============================================================
# Scoring (anchor-based prediction)
# ============================================================

def score_filmdelta(model, scaler, train_smiles, train_pIC50, test_smiles, emb_dict, emb_dim):
    """Anchor-based prediction: pIC50(j) = median(pIC50(i) + delta(i→j))."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))
    train_embs = scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(np.array([get_emb(s) for s in test_smiles])).astype(np.float32)
    n_train = len(train_smiles)
    preds = np.zeros(len(test_smiles))
    model.cpu().eval()
    with torch.no_grad():
        for j in range(len(test_smiles)):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (n_train, 1)))
            deltas = model(a, b).numpy().flatten()
            preds[j] = np.median(train_pIC50 + deltas)
    return preds


def score_dualstream(model, scaler, train_smiles, train_pIC50, test_smiles, emb_dict, emb_dim):
    """Anchor-based prediction with DualStreamFiLM."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))
    train_embs = scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(np.array([get_emb(s) for s in test_smiles])).astype(np.float32)
    n_train = len(train_smiles)
    preds = np.zeros(len(test_smiles))
    model.cpu().eval()
    with torch.no_grad():
        for j in range(len(test_smiles)):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (n_train, 1)))
            # Get DRFP + edit features from cache
            drfp_arr = np.zeros((n_train, 2048), dtype=np.float32)
            ef_arr = np.zeros((n_train, 28), dtype=np.float32)
            if PAIR_DRFP_CACHE is not None:
                for i in range(n_train):
                    key = (train_smiles[i], test_smiles[j])
                    if key in PAIR_DRFP_CACHE:
                        drfp_arr[i] = PAIR_DRFP_CACHE[key]
                        ef_arr[i] = PAIR_EF_CACHE[key]
            drfp = torch.FloatTensor(drfp_arr)
            ef = torch.FloatTensor(ef_arr)
            deltas = model(a, b, drfp, ef).numpy().flatten()
            preds[j] = np.median(train_pIC50 + deltas)
    return preds


def score_subtraction(train_smiles, train_pIC50, test_smiles, emb_dict, emb_dim):
    """Subtraction baseline: train independent pIC50 predictor, predict test, compare."""
    from sklearn.ensemble import GradientBoostingRegressor

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    X_train = np.array([get_emb(s) for s in train_smiles])
    X_test = np.array([get_emb(s) for s in test_smiles])

    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42, n_iter_no_change=15)
    reg.fit(X_train, train_pIC50)
    return reg.predict(X_test)


# ============================================================
# Kinase pretraining
# ============================================================

def pretrain_filmdelta_on_kinase(emb_dict, emb_dim):
    """Pretrain FiLMDelta on kinase within-assay pairs. Returns (state_dict, scaler)."""
    print("  Pretraining FiLMDelta on kinase pairs...")
    kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    all_kinase_smi = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
    # Compute fps for kinase molecules not already in emb_dict
    extra = [s for s in all_kinase_smi if s not in emb_dict]
    if extra:
        efps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
        for i, s in enumerate(extra):
            emb_dict[s] = efps[i]

    mask = kinase_pairs["mol_a"].apply(lambda s: s in emb_dict) & \
           kinase_pairs["mol_b"].apply(lambda s: s in emb_dict)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    print(f"    Kinase pairs: {len(kinase_pairs):,}")

    ea = np.array([emb_dict[s] for s in kinase_pairs["mol_a"]])
    eb = np.array([emb_dict[s] for s in kinase_pairs["mol_b"]])
    d = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([ea, eb]))
    Xa = torch.FloatTensor(scaler.transform(ea))
    Xb = torch.FloatTensor(scaler.transform(eb))
    yd = torch.FloatTensor(d)
    del ea, eb, d, kinase_pairs; gc.collect()

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    n_val = len(Xa) // 10
    best_vl, best_st, wait = float("inf"), None, 0

    for ep in range(100):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), BATCH_SIZE):
            bi = perm[s:s+BATCH_SIZE]
            xa_b = Xa[bi].to(device); xb_b = Xb[bi].to(device); yd_b = yd[bi].to(device)
            opt.zero_grad()
            crit(model(xa_b, xb_b), yd_b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val].to(device), Xb[:n_val].to(device)), yd[:n_val].to(device)).item()
        if vl < best_vl:
            best_vl, best_st, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15:
                print(f"    Pretrain early stop ep {ep+1}, val_loss={best_vl:.4f}")
                break
    else:
        print(f"    Pretrain finished 100 epochs, val_loss={best_vl:.4f}")

    del Xa, Xb, yd; gc.collect()
    return best_st, scaler


def pretrain_dualstream_on_kinase(emb_dict, emb_dim):
    """Pretrain DualStreamFiLM on kinase within-assay pairs. Returns (state_dict, scaler)."""
    print("  Pretraining DualStreamFiLM on kinase pairs...")
    kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    all_kinase_smi = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
    extra = [s for s in all_kinase_smi if s not in emb_dict]
    if extra:
        efps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
        for i, s in enumerate(extra):
            emb_dict[s] = efps[i]

    mask = kinase_pairs["mol_a"].apply(lambda s: s in emb_dict) & \
           kinase_pairs["mol_b"].apply(lambda s: s in emb_dict)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    print(f"    Kinase pairs: {len(kinase_pairs):,}")

    ea = np.array([emb_dict[s] for s in kinase_pairs["mol_a"]])
    eb = np.array([emb_dict[s] for s in kinase_pairs["mol_b"]])
    d = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([ea, eb]))

    # Compute DRFP + edit features for kinase pairs
    print(f"    Computing DRFP for {len(kinase_pairs)} kinase pairs...")
    drfp = compute_drfp_for_pairs(
        kinase_pairs["mol_a"].tolist(), kinase_pairs["mol_b"].tolist())
    print(f"    Computing edit features for {len(kinase_pairs)} kinase pairs...")
    ef = compute_edit_features_batch(
        kinase_pairs["mol_a"].tolist(), kinase_pairs["mol_b"].tolist())

    Xa = torch.FloatTensor(scaler.transform(ea))
    Xb = torch.FloatTensor(scaler.transform(eb))
    Xdrfp = torch.FloatTensor(drfp)
    Xef = torch.FloatTensor(ef)
    yd = torch.FloatTensor(d)
    del ea, eb, d, drfp, ef, kinase_pairs; gc.collect()

    device = torch.device(DEVICE)
    model = DualStreamFiLMDeltaMLP(
        mol_dim=emb_dim, drfp_dim=2048, edit_feat_dim=28, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    n_val = len(Xa) // 10
    best_vl, best_st, wait = float("inf"), None, 0

    for ep in range(100):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), BATCH_SIZE):
            bi = perm[s:s+BATCH_SIZE]
            xa_b = Xa[bi].to(device); xb_b = Xb[bi].to(device)
            dr_b = Xdrfp[bi].to(device); ef_b = Xef[bi].to(device)
            yd_b = yd[bi].to(device)
            opt.zero_grad()
            loss = crit(model(xa_b, xb_b, dr_b, ef_b), yd_b) + 0.1 * model.aux_loss(ef_b)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val].to(device), Xb[:n_val].to(device),
                            Xdrfp[:n_val].to(device), Xef[:n_val].to(device)),
                       yd[:n_val].to(device)).item()
        if vl < best_vl:
            best_vl, best_st, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15:
                print(f"    Pretrain early stop ep {ep+1}, val_loss={best_vl:.4f}")
                break
    else:
        print(f"    Pretrain finished 100 epochs, val_loss={best_vl:.4f}")

    del Xa, Xb, Xdrfp, Xef, yd; gc.collect()
    return best_st, scaler


# ============================================================
# Main CV loop
# ============================================================

METHODS = [
    "Subtraction",
    "FiLMDelta",
    "FiLMDelta+KinasePT",
    "DualStreamFiLM",
    "DualStreamFiLM+KinasePT",
]

SPLITS = {
    "random": random_cv_splits,
    "scaffold": scaffold_cv_splits,
    "distant": distant_molecule_cv_splits,
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("ZAP70 FULL CV: 5 methods × 3 splits × 3 seeds × 5 folds")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    print(f"  {len(all_smiles)} ZAP70 molecules")

    # Compute embeddings
    print("\n[2] Computing Morgan FP embeddings...")
    X = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smiles)}

    # Build pair cache for DualStream
    print("\n[3] Building DRFP + edit feature pair cache...")
    build_pair_cache(all_smiles)

    # Kinase pretraining (once, reuse across folds)
    print("\n[4] Kinase pretraining...")
    film_pt_state, film_pt_scaler = pretrain_filmdelta_on_kinase(emb_dict, emb_dim)
    print("  FiLMDelta pretrain done")
    gc.collect()

    ds_pt_state, ds_pt_scaler = pretrain_dualstream_on_kinase(emb_dict, emb_dim)
    print("  DualStreamFiLM pretrain done")
    gc.collect()

    # Run all combinations
    print(f"\n{'='*70}")
    print("RUNNING CV EVALUATION")
    print(f"{'='*70}")

    all_results = {}

    for split_name, split_fn in SPLITS.items():
        print(f"\n{'='*70}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*70}")

        split_results = {}

        for method in METHODS:
            print(f"\n  --- {method} ({split_name}) ---")
            all_fold_metrics = []

            for seed in range(N_SEEDS):
                splits = split_fn(mol_data, seed=CV_SEED + seed)

                for fold_name, train_data, test_data in splits:
                    train_smi = train_data['smiles'].tolist()
                    train_y = train_data['pIC50'].values
                    test_smi = test_data['smiles'].tolist()
                    test_y = test_data['pIC50'].values

                    if len(test_smi) < 5:
                        continue

                    if method == "Subtraction":
                        preds = score_subtraction(train_smi, train_y, test_smi, emb_dict, emb_dim)

                    elif method == "FiLMDelta":
                        pairs = generate_all_pairs(train_smi, train_y)
                        n_val = max(int(len(pairs) * 0.1), 100)
                        val_pairs = pairs.sample(n=n_val, random_state=seed)
                        trn_pairs = pairs.drop(val_pairs.index)
                        model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed)
                        preds = score_filmdelta(model, scaler, train_smi, train_y, test_smi, emb_dict, emb_dim)
                        del model, scaler

                    elif method == "FiLMDelta+KinasePT":
                        pairs = generate_all_pairs(train_smi, train_y)
                        n_val = max(int(len(pairs) * 0.1), 100)
                        val_pairs = pairs.sample(n=n_val, random_state=seed)
                        trn_pairs = pairs.drop(val_pairs.index)
                        model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed,
                                                         pretrained_state=film_pt_state, pretrained_scaler=film_pt_scaler)
                        preds = score_filmdelta(model, scaler, train_smi, train_y, test_smi, emb_dict, emb_dim)
                        del model, scaler

                    elif method == "DualStreamFiLM":
                        pairs = generate_all_pairs(train_smi, train_y)
                        n_val = max(int(len(pairs) * 0.1), 100)
                        val_pairs = pairs.sample(n=n_val, random_state=seed)
                        trn_pairs = pairs.drop(val_pairs.index)
                        trn_pre = get_precomputed_from_cache(trn_pairs)
                        val_pre = get_precomputed_from_cache(val_pairs)
                        precomputed = None
                        if trn_pre and val_pre:
                            precomputed = {
                                'train_drfp': trn_pre['drfp'], 'val_drfp': val_pre['drfp'],
                                'train_ef': trn_pre['ef'], 'val_ef': val_pre['ef'],
                            }
                        model, scaler = train_dualstream(trn_pairs, val_pairs, emb_dict, emb_dim, seed,
                                                          precomputed=precomputed)
                        preds = score_dualstream(model, scaler, train_smi, train_y, test_smi, emb_dict, emb_dim)
                        del model, scaler

                    elif method == "DualStreamFiLM+KinasePT":
                        pairs = generate_all_pairs(train_smi, train_y)
                        n_val = max(int(len(pairs) * 0.1), 100)
                        val_pairs = pairs.sample(n=n_val, random_state=seed)
                        trn_pairs = pairs.drop(val_pairs.index)
                        trn_pre = get_precomputed_from_cache(trn_pairs)
                        val_pre = get_precomputed_from_cache(val_pairs)
                        precomputed = None
                        if trn_pre and val_pre:
                            precomputed = {
                                'train_drfp': trn_pre['drfp'], 'val_drfp': val_pre['drfp'],
                                'train_ef': trn_pre['ef'], 'val_ef': val_pre['ef'],
                            }
                        model, scaler = train_dualstream(trn_pairs, val_pairs, emb_dict, emb_dim, seed,
                                                          precomputed=precomputed,
                                                          pretrained_state=ds_pt_state,
                                                          pretrained_scaler=ds_pt_scaler)
                        preds = score_dualstream(model, scaler, train_smi, train_y, test_smi, emb_dict, emb_dim)
                        del model, scaler

                    metrics = compute_absolute_metrics(test_y, preds)
                    all_fold_metrics.append(metrics)
                    print(f"    {fold_name} s{seed}: MAE={metrics['mae']:.3f} Spr={metrics['spearman_r']:.3f}")
                    gc.collect()

            agg = aggregate_cv_results(all_fold_metrics)
            split_results[method] = agg
            print(f"  >> {method}: MAE={agg['mae_mean']:.3f}±{agg['mae_std']:.3f}, "
                  f"Spr={agg['spearman_r_mean']:.3f}±{agg['spearman_r_std']:.3f}")

        all_results[split_name] = split_results

        # Save incrementally
        out_file = RESULTS_DIR / "19_molecules_full_cv.json"
        with open(out_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved incrementally to {out_file}]")

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY (elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n{'Method':<28s} {'Random MAE':>12s} {'Scaffold MAE':>14s} {'Distant MAE':>13s}")
    print("-" * 70)
    for method in METHODS:
        row = f"{method:<28s}"
        for split_name in ["random", "scaffold", "distant"]:
            if split_name in all_results and method in all_results[split_name]:
                m = all_results[split_name][method]
                row += f"  {m['mae_mean']:.3f}±{m['mae_std']:.3f}"
            else:
                row += "  ---"
        print(row)

    print(f"\n{'Method':<28s} {'Random Spr':>12s} {'Scaffold Spr':>14s} {'Distant Spr':>13s}")
    print("-" * 70)
    for method in METHODS:
        row = f"{method:<28s}"
        for split_name in ["random", "scaffold", "distant"]:
            if split_name in all_results and method in all_results[split_name]:
                m = all_results[split_name][method]
                row += f"  {m['spearman_r_mean']:.3f}±{m['spearman_r_std']:.3f}"
            else:
                row += "  ---"
        print(row)

    print(f"\nResults saved to {out_file}")
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
