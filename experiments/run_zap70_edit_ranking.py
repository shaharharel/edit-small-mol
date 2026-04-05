#!/usr/bin/env python3
"""
ZAP70 Model-Based Edit Ranking.

Trains FiLMDelta on 1.7M within-assay MMP pairs, fine-tunes on ZAP70 all-pairs,
then evaluates edit ranking: for each ZAP70 molecule, predict which "edits"
(other ZAP70 molecules) would most improve pIC50.

Compares:
  1. FiLMDelta pretrained (200K within-assay) + fine-tuned on ZAP70
  2. FiLMDelta trained only on ZAP70 (no pretraining)
  3. XGB subtraction baseline (predict absolute pIC50, subtract)

Metrics: per-molecule Spearman of predicted vs actual delta rankings,
         top-K precision (does model find best partner in top-K?)

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_zap70_edit_ranking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import os
import time
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.mps.is_available = lambda: False

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ═══════════════════════════════════════════════════════════════════════════
# Paths & Config
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"
CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
SHARED_PAIRS_FILE = DATA_DIR / "shared_pairs_deduped.csv"
RESULTS_FILE = RESULTS_DIR / "zap70_model_edit_ranking.json"

DEVICE = "cpu"
ZAP70_ID = "CHEMBL2803"
EMBEDDER_NAME = "chemprop-dmpnn"  # Morgan FP 2048d
N_PRETRAIN_SAMPLE = 200_000  # Sample from within-assay pairs for speed
BATCH_SIZE = 128
SEED = 42

# XGB best params from v3 Phase 4 Optuna
XGB_BEST_PARAMS = {
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.991,
    "colsample_bytree": 0.774,
    "learning_rate": 0.185,
    "n_estimators": 710,
    "reg_alpha": 0.196,
    "reg_lambda": 0.784,
    "gamma": 0.002,
    "random_state": 42,
    "n_jobs": 8,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_zap70_molecules():
    """Load ZAP70 molecule-level data (averaged across assays)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  ZAP70: {len(mol_data)} molecules, pIC50 {mol_data['pIC50'].min():.2f}-"
          f"{mol_data['pIC50'].max():.2f} (mean={mol_data['pIC50'].mean():.2f}, "
          f"std={mol_data['pIC50'].std():.2f})")
    return mol_data


def load_embeddings(smiles_list):
    """Load Morgan FP embeddings from cache, compute on-the-fly for missing."""
    cache_file = CACHE_DIR / f"{EMBEDDER_NAME}.npz"
    print(f"  Loading cached embeddings from {cache_file.name}...")
    data = np.load(cache_file, allow_pickle=True)
    cached_smiles = data["smiles"].tolist()
    cached_embs = data["embeddings"]
    emb_dim = int(data["emb_dim"])
    cached_dict = {smi: cached_embs[i] for i, smi in enumerate(cached_smiles)}

    missing = [s for s in smiles_list if s not in cached_dict]
    if missing:
        print(f"    {len(missing)} molecules not in cache, computing Morgan FP on-the-fly...")
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        for smi in missing:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=emb_dim)
                arr = np.zeros(emb_dim, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                cached_dict[smi] = arr
            else:
                cached_dict[smi] = np.zeros(emb_dim, dtype=np.float32)
        print(f"    Computed {len(missing)} missing embeddings")

    emb_dict = {s: cached_dict.get(s, np.zeros(emb_dim, dtype=np.float32))
                for s in smiles_list}
    print(f"    Loaded {len(emb_dict):,} embeddings (dim={emb_dim})")
    return emb_dict, emb_dim


def load_within_assay_pairs(n_sample=N_PRETRAIN_SAMPLE):
    """Load within-assay pairs from shared_pairs_deduped.csv, sample for speed."""
    print(f"\n  Loading shared pairs from {SHARED_PAIRS_FILE.name}...")
    t0 = time.time()
    df = pd.read_csv(SHARED_PAIRS_FILE)
    within = df[df["is_within_assay"] == True].copy()
    print(f"    Total within-assay pairs: {len(within):,}")

    if n_sample and len(within) > n_sample:
        within = within.sample(n=n_sample, random_state=SEED)
        print(f"    Sampled to {n_sample:,} pairs")

    elapsed = time.time() - t0
    print(f"    Loaded in {elapsed:.1f}s")
    return within


def generate_zap70_allpairs(mol_data):
    """Generate all ordered pairs from ZAP70 molecules."""
    rows = []
    smiles_list = mol_data["smiles"].values
    pic50_list = mol_data["pIC50"].values
    mol_ids = mol_data["molecule_chembl_id"].values

    for i in range(len(mol_data)):
        for j in range(len(mol_data)):
            if i == j:
                continue
            rows.append({
                "mol_a": smiles_list[i],
                "mol_b": smiles_list[j],
                "mol_a_id": mol_ids[i],
                "mol_b_id": mol_ids[j],
                "value_a": pic50_list[i],
                "value_b": pic50_list[j],
                "delta": pic50_list[j] - pic50_list[i],
                "target_chembl_id": ZAP70_ID,
            })

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df):,} all-pairs from {len(mol_data)} ZAP70 molecules")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FiLMDelta Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_pair_tensors(df, emb_dict, emb_dim):
    """Convert pair dataframe to tensors."""
    zero = np.zeros(emb_dim, dtype=np.float32)
    emb_a = np.array([emb_dict.get(s, zero) for s in df["mol_a"]], dtype=np.float32)
    emb_b = np.array([emb_dict.get(s, zero) for s in df["mol_b"]], dtype=np.float32)
    delta = df["delta"].values.astype(np.float32)
    return (
        torch.from_numpy(emb_a).float(),
        torch.from_numpy(emb_b).float(),
        torch.from_numpy(delta).float(),
    )


def train_film_delta(train_a, train_b, train_y, val_a, val_b, val_y,
                     emb_dim, max_epochs=150, patience=15, lr=1e-3,
                     hidden_dims=None, existing_model=None):
    """Train or fine-tune a FiLMDeltaMLP model.

    If existing_model is provided, fine-tune it (warm start).
    Returns trained model.
    """
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    if hidden_dims is None:
        hidden_dims = [512, 256, 128]

    if existing_model is not None:
        model = existing_model
    else:
        model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=hidden_dims)

    model = model.to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            pred = model(a, b)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(DEVICE) for t in batch]
                pred = model(a, b)
                val_losses.append(criterion(pred, y).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model


def predict_film_delta(model, emb_a, emb_b):
    """Predict deltas with FiLMDeltaMLP."""
    model = model.to(DEVICE)
    model.eval()
    all_preds = []
    dataset = TensorDataset(emb_a, emb_b)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            a, b = [t.to(DEVICE) for t in batch]
            preds = model(a, b)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


# ═══════════════════════════════════════════════════════════════════════════
# XGB Subtraction Baseline
# ═══════════════════════════════════════════════════════════════════════════

def train_xgb_absolute(X_train, y_train):
    """Train XGB to predict absolute pIC50."""
    import xgboost as xgb
    model = xgb.XGBRegressor(**XGB_BEST_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Edit Ranking Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_edit_ranking(mol_data, emb_dict, emb_dim, predict_fn, method_name):
    """Evaluate edit ranking for each molecule.

    For each molecule m_i, pair it with all other molecules m_j (j != i).
    Predict delta(m_i -> m_j) and compare ranking to actual delta.

    Returns per-molecule and aggregate metrics.
    """
    smiles = mol_data["smiles"].values
    pIC50s = mol_data["pIC50"].values
    n = len(mol_data)

    print(f"\n  Evaluating {method_name} edit ranking ({n} molecules, {n*(n-1)} pairs)...")
    t0 = time.time()

    per_mol_metrics = []
    all_true_deltas = []
    all_pred_deltas = []

    # For each molecule, predict deltas to all others
    for i in range(n):
        # Build pairs: (m_i, m_j) for all j != i
        others = [j for j in range(n) if j != i]
        true_deltas = np.array([pIC50s[j] - pIC50s[i] for j in others])

        zero = np.zeros(emb_dim, dtype=np.float32)
        emb_a_arr = np.array([emb_dict.get(smiles[i], zero)] * len(others), dtype=np.float32)
        emb_b_arr = np.array([emb_dict.get(smiles[j], zero) for j in others], dtype=np.float32)

        emb_a_t = torch.from_numpy(emb_a_arr).float()
        emb_b_t = torch.from_numpy(emb_b_arr).float()

        pred_deltas = predict_fn(emb_a_t, emb_b_t)

        all_true_deltas.extend(true_deltas)
        all_pred_deltas.extend(pred_deltas)

        # Per-molecule ranking metrics
        if np.std(true_deltas) > 1e-8 and np.std(pred_deltas) > 1e-8:
            spr, spr_p = spearmanr(true_deltas, pred_deltas)
            pr, pr_p = pearsonr(true_deltas, pred_deltas)
            tau, tau_p = kendalltau(true_deltas, pred_deltas)
        else:
            spr, pr, tau = 0.0, 0.0, 0.0

        # Top-K precision: is the actual best partner in the model's top-K?
        true_rank = np.argsort(-true_deltas)  # descending by actual improvement
        pred_rank = np.argsort(-pred_deltas)  # descending by predicted improvement

        top_k_metrics = {}
        for k in [1, 3, 5, 10]:
            if k <= len(others):
                # What fraction of the true top-K appears in predicted top-K?
                true_top_k = set(true_rank[:k])
                pred_top_k = set(pred_rank[:k])
                precision = len(true_top_k & pred_top_k) / k
                top_k_metrics[f"top_{k}_precision"] = precision

        # Is the actual best partner in the top-K predictions?
        actual_best = true_rank[0]
        for k in [1, 3, 5, 10]:
            if k <= len(others):
                top_k_metrics[f"best_in_top_{k}"] = float(actual_best in set(pred_rank[:k]))

        per_mol_metrics.append({
            "smiles": smiles[i],
            "pIC50": float(pIC50s[i]),
            "spearman": float(spr) if not np.isnan(spr) else 0.0,
            "pearson": float(pr) if not np.isnan(pr) else 0.0,
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            **top_k_metrics,
        })

    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f}s")

    # Aggregate metrics
    all_true = np.array(all_true_deltas)
    all_pred = np.array(all_pred_deltas)

    global_mae = float(np.mean(np.abs(all_true - all_pred)))
    global_spr, _ = spearmanr(all_true, all_pred)
    global_pr, _ = pearsonr(all_true, all_pred)
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    global_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Averaged per-molecule metrics
    avg_spr = float(np.mean([m["spearman"] for m in per_mol_metrics]))
    avg_pr = float(np.mean([m["pearson"] for m in per_mol_metrics]))
    avg_tau = float(np.mean([m["kendall_tau"] for m in per_mol_metrics]))

    top_k_avgs = {}
    for k in [1, 3, 5, 10]:
        key_p = f"top_{k}_precision"
        key_b = f"best_in_top_{k}"
        vals_p = [m[key_p] for m in per_mol_metrics if key_p in m]
        vals_b = [m[key_b] for m in per_mol_metrics if key_b in m]
        if vals_p:
            top_k_avgs[f"avg_{key_p}"] = float(np.mean(vals_p))
        if vals_b:
            top_k_avgs[f"avg_{key_b}"] = float(np.mean(vals_b))

    aggregate = {
        "method": method_name,
        "n_molecules": n,
        "n_pairs": n * (n - 1),
        "global_mae": global_mae,
        "global_spearman": float(global_spr),
        "global_pearson": float(global_pr),
        "global_r2": global_r2,
        "avg_per_mol_spearman": avg_spr,
        "avg_per_mol_pearson": avg_pr,
        "avg_per_mol_kendall_tau": avg_tau,
        **top_k_avgs,
    }

    print(f"    Global MAE: {global_mae:.4f}")
    print(f"    Global Spearman: {global_spr:.4f}")
    print(f"    Avg per-mol Spearman: {avg_spr:.4f}")
    print(f"    Avg per-mol Kendall tau: {avg_tau:.4f}")
    for k in [1, 3, 5, 10]:
        if f"avg_best_in_top_{k}" in top_k_avgs:
            print(f"    Avg best-in-top-{k}: {top_k_avgs[f'avg_best_in_top_{k}']:.3f}")

    return aggregate, per_mol_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ZAP70 Model-Based Edit Ranking")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {"started": datetime.now().isoformat()}

    # ── Step 1: Load ZAP70 data ──
    print("\n[1] Loading ZAP70 molecules...")
    mol_data = load_zap70_molecules()
    n_mols = len(mol_data)

    # ── Step 2: Load embeddings ──
    print("\n[2] Loading Morgan FP embeddings...")
    all_smiles = mol_data["smiles"].tolist()

    # Also need SMILES from pretrain data for embedding lookup
    print("  Loading shared pairs SMILES for embedding coverage...")
    sp_df = pd.read_csv(SHARED_PAIRS_FILE, usecols=["mol_a", "mol_b", "is_within_assay"])
    within_smiles = set(sp_df[sp_df["is_within_assay"] == True]["mol_a"].tolist() +
                        sp_df[sp_df["is_within_assay"] == True]["mol_b"].tolist())
    all_needed_smiles = list(set(all_smiles) | within_smiles)
    del sp_df
    gc.collect()

    emb_dict, emb_dim = load_embeddings(all_needed_smiles)
    results["emb_dim"] = emb_dim
    results["n_zap70_molecules"] = n_mols

    # ── Step 3: Generate ZAP70 all-pairs ──
    print("\n[3] Generating ZAP70 all-pairs...")
    zap70_pairs = generate_zap70_allpairs(mol_data)

    # ── Step 4: 5-fold CV on ZAP70 molecules ──
    # Split molecules, then pairs follow
    print("\n[4] Setting up 5-fold CV on ZAP70 molecules...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    mol_indices = np.arange(n_mols)
    smiles_arr = mol_data["smiles"].values

    fold_results = {
        "film_pretrained_finetuned": [],
        "film_zap70_only": [],
        "xgb_subtraction": [],
    }

    for fold_i, (train_mol_idx, test_mol_idx) in enumerate(kf.split(mol_indices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_i + 1}/5 (train={len(train_mol_idx)} mols, test={len(test_mol_idx)} mols)")
        print(f"{'='*60}")

        train_smiles = set(smiles_arr[train_mol_idx])
        test_smiles = set(smiles_arr[test_mol_idx])

        # Train/val split from training molecules
        n_train = len(train_mol_idx)
        val_size = max(1, int(n_train * 0.15))
        rng = np.random.RandomState(SEED + fold_i)
        perm = rng.permutation(train_mol_idx)
        val_mol_idx = perm[:val_size]
        actual_train_mol_idx = perm[val_size:]

        actual_train_smiles = set(smiles_arr[actual_train_mol_idx])
        val_smiles = set(smiles_arr[val_mol_idx])

        # Build ZAP70 pair sets
        # Training pairs: both molecules in actual_train set
        zap_train = zap70_pairs[
            zap70_pairs["mol_a"].isin(actual_train_smiles) &
            zap70_pairs["mol_b"].isin(actual_train_smiles)
        ].copy()
        zap_val = zap70_pairs[
            zap70_pairs["mol_a"].isin(val_smiles) &
            zap70_pairs["mol_b"].isin(val_smiles)
        ].copy()

        # Test: pairs where mol_a is a test molecule, mol_b can be anything
        # This evaluates: for each test molecule, can we rank all others?
        test_mol_data = mol_data.iloc[test_mol_idx].copy()

        print(f"  ZAP70 train pairs: {len(zap_train):,}, val pairs: {len(zap_val):,}")
        print(f"  Test molecules: {len(test_mol_idx)}")

        zap_train_a, zap_train_b, zap_train_y = get_pair_tensors(zap_train, emb_dict, emb_dim)
        zap_val_a, zap_val_b, zap_val_y = get_pair_tensors(zap_val, emb_dict, emb_dim)

        # ── Method 1: FiLMDelta pretrained + fine-tuned ──
        print(f"\n  [4a] FiLMDelta pretrained on 200K within-assay + fine-tuned on ZAP70")
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)

        # Pretrain on within-assay pairs
        print(f"    Pretraining on within-assay pairs...")
        within_pairs = load_within_assay_pairs(n_sample=N_PRETRAIN_SAMPLE)

        # Split pretrain data into train/val (90/10)
        n_pretrain = len(within_pairs)
        pretrain_val_size = int(n_pretrain * 0.1)
        pretrain_perm = np.random.permutation(n_pretrain)
        pretrain_val_df = within_pairs.iloc[pretrain_perm[:pretrain_val_size]]
        pretrain_train_df = within_pairs.iloc[pretrain_perm[pretrain_val_size:]]

        pt_train_a, pt_train_b, pt_train_y = get_pair_tensors(pretrain_train_df, emb_dict, emb_dim)
        pt_val_a, pt_val_b, pt_val_y = get_pair_tensors(pretrain_val_df, emb_dict, emb_dim)

        t0 = time.time()
        pretrained_model = train_film_delta(
            pt_train_a, pt_train_b, pt_train_y,
            pt_val_a, pt_val_b, pt_val_y,
            emb_dim=emb_dim, max_epochs=100, patience=10, lr=1e-3,
        )
        pretrain_time = time.time() - t0
        print(f"    Pretrain done in {pretrain_time:.1f}s")

        # Free pretrain tensors
        del pt_train_a, pt_train_b, pt_train_y, pt_val_a, pt_val_b, pt_val_y
        del within_pairs, pretrain_train_df, pretrain_val_df
        gc.collect()

        # Fine-tune on ZAP70
        print(f"    Fine-tuning on ZAP70 ({len(zap_train)} pairs)...")
        t0 = time.time()
        finetuned_model = train_film_delta(
            zap_train_a, zap_train_b, zap_train_y,
            zap_val_a, zap_val_b, zap_val_y,
            emb_dim=emb_dim, max_epochs=50, patience=10, lr=1e-4,
            existing_model=pretrained_model,
        )
        finetune_time = time.time() - t0
        print(f"    Fine-tune done in {finetune_time:.1f}s")

        # Evaluate
        def predict_fn_pretrained(a, b):
            return predict_film_delta(finetuned_model, a, b)

        agg_pt, per_mol_pt = evaluate_edit_ranking(
            test_mol_data, emb_dict, emb_dim, predict_fn_pretrained,
            "FiLMDelta_pretrained_finetuned"
        )
        agg_pt["pretrain_time_s"] = pretrain_time
        agg_pt["finetune_time_s"] = finetune_time
        fold_results["film_pretrained_finetuned"].append(agg_pt)

        del pretrained_model, finetuned_model
        gc.collect()

        # ── Method 2: FiLMDelta ZAP70-only ──
        print(f"\n  [4b] FiLMDelta trained only on ZAP70")
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)

        t0 = time.time()
        zap70_only_model = train_film_delta(
            zap_train_a, zap_train_b, zap_train_y,
            zap_val_a, zap_val_b, zap_val_y,
            emb_dim=emb_dim, max_epochs=150, patience=15, lr=1e-3,
        )
        zap70_only_time = time.time() - t0
        print(f"    Training done in {zap70_only_time:.1f}s")

        def predict_fn_zap70(a, b):
            return predict_film_delta(zap70_only_model, a, b)

        agg_z, per_mol_z = evaluate_edit_ranking(
            test_mol_data, emb_dict, emb_dim, predict_fn_zap70,
            "FiLMDelta_zap70_only"
        )
        agg_z["train_time_s"] = zap70_only_time
        fold_results["film_zap70_only"].append(agg_z)

        del zap70_only_model
        gc.collect()

        # ── Method 3: XGB subtraction baseline ──
        print(f"\n  [4c] XGB subtraction baseline")
        t0 = time.time()

        # Train XGB on absolute pIC50 using training molecules
        train_mol = mol_data.iloc[actual_train_mol_idx]
        zero = np.zeros(emb_dim, dtype=np.float32)
        X_train_xgb = np.array([emb_dict.get(s, zero) for s in train_mol["smiles"]], dtype=np.float32)
        y_train_xgb = train_mol["pIC50"].values.astype(np.float32)

        xgb_model = train_xgb_absolute(X_train_xgb, y_train_xgb)
        xgb_time = time.time() - t0
        print(f"    XGB training done in {xgb_time:.1f}s")

        # Predict deltas via subtraction: delta = f(mol_b) - f(mol_a)
        def predict_fn_xgb(a_tensor, b_tensor):
            a_np = a_tensor.numpy()
            b_np = b_tensor.numpy()
            pred_a = xgb_model.predict(a_np)
            pred_b = xgb_model.predict(b_np)
            return pred_b - pred_a

        agg_x, per_mol_x = evaluate_edit_ranking(
            test_mol_data, emb_dict, emb_dim, predict_fn_xgb,
            "XGB_subtraction"
        )
        agg_x["train_time_s"] = xgb_time
        fold_results["xgb_subtraction"].append(agg_x)

        del xgb_model
        gc.collect()

        # Free fold tensors
        del zap_train_a, zap_train_b, zap_train_y
        del zap_val_a, zap_val_b, zap_val_y
        gc.collect()

    # ── Step 5: Aggregate across folds ──
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5-fold CV)")
    print("=" * 70)

    summary = {}
    for method, fold_list in fold_results.items():
        if not fold_list:
            continue
        # Average numeric metrics across folds
        metric_keys = [k for k in fold_list[0].keys()
                       if isinstance(fold_list[0][k], (int, float))]
        agg = {"n_folds": len(fold_list)}
        for k in metric_keys:
            vals = [f[k] for f in fold_list if k in f]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

        summary[method] = agg

        print(f"\n  {method}:")
        print(f"    Global MAE: {agg.get('global_mae_mean', 0):.4f} +/- {agg.get('global_mae_std', 0):.4f}")
        print(f"    Global Spearman: {agg.get('global_spearman_mean', 0):.4f} +/- {agg.get('global_spearman_std', 0):.4f}")
        print(f"    Avg per-mol Spearman: {agg.get('avg_per_mol_spearman_mean', 0):.4f} +/- {agg.get('avg_per_mol_spearman_std', 0):.4f}")
        print(f"    Avg per-mol Kendall tau: {agg.get('avg_per_mol_kendall_tau_mean', 0):.4f} +/- {agg.get('avg_per_mol_kendall_tau_std', 0):.4f}")
        for k_val in [1, 3, 5, 10]:
            bk = f"avg_best_in_top_{k_val}_mean"
            pk = f"avg_top_{k_val}_precision_mean"
            if bk in agg:
                print(f"    Best-in-top-{k_val}: {agg[bk]:.3f}")
            if pk in agg:
                print(f"    Top-{k_val} precision: {agg[pk]:.3f}")

    # ── Step 6: Save results ──
    results["summary"] = summary
    results["fold_results"] = {k: v for k, v in fold_results.items()}
    results["completed"] = datetime.now().isoformat()
    results["config"] = {
        "embedder": EMBEDDER_NAME,
        "emb_dim": emb_dim,
        "n_pretrain_sample": N_PRETRAIN_SAMPLE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "target": ZAP70_ID,
        "xgb_params": XGB_BEST_PARAMS,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    main()
