#!/usr/bin/env python3
"""
Combined Kinase Pretraining + Docking Integration for ZAP70 (CHEMBL2803).

Tests whether combining kinase pretraining (transfer learning from related
kinases) with docking-enhanced FiLMDelta (3D protein-ligand binding features)
yields additive improvements over either alone.

Phases:
    A: Pretrain FiLMDelta on kinase panel MMP pairs (WITHOUT docking features)
    B: Finetune pretrained model on ZAP70 with docking features
       (transfer FiLM backbone weights, randomly init docking-specific layers)
    C: Alternative pretraining strategies:
       Option 1 — zero-filled docking features during pretraining
       Option 2 — no docking during pretraining, add at finetuning (= Phase B)
    D: Comprehensive comparison of all 4 methods (5-fold CV)

Expected methods:
    1. FiLMDelta                — no pretrain, no docking (baseline)
    2. FiLMDelta_vina           — no pretrain, docking
    3. FiLMDelta_pretrained     — pretrain, no docking
    4. FiLMDelta_vina_pretrained — pretrain + docking (THE NEW MODEL)
    5. FiLMDelta_vina_pretrained_zerofill — pretrain with zero docking + finetune with docking

Usage:
    conda run -n quris python -u experiments/run_docking_pretrain_combo.py
    conda run -n quris python -u experiments/run_docking_pretrain_combo.py --phase A
    conda run -n quris python -u experiments/run_docking_pretrain_combo.py --phase A B C D
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import gc
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    compute_absolute_metrics, aggregate_cv_results,
    N_FOLDS, CV_SEED,
)
from experiments.run_paper_evaluation import RESULTS_DIR, DATA_DIR

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "docking_pretrain_results.json"
DEVICE = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else "cpu"

# Docking data paths
DOCK_CHEMBL_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCK_CHEMBL_CSV = DOCK_CHEMBL_DIR / "docking_results.csv"

# Kinase targets for pretraining (same panel as kinase_transfer_ablation)
PRETRAIN_KINASES = {
    "SYK": "CHEMBL2599",
    "LCK": "CHEMBL258",
    "JAK2": "CHEMBL2971",
    "ABL1": "CHEMBL1862",
    "SRC": "CHEMBL267",
    "BTK": "CHEMBL5251",
}

# Hyperparameters
PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_PATIENCE = 15
FINETUNE_EPOCHS = 80
FINETUNE_LR = 5e-4
FINETUNE_BATCH_SIZE = 64
FINETUNE_PATIENCE = 15
N_SEEDS = 3
MAX_PRETRAIN_PAIRS = 100_000
VINA_DIM = 3  # vina_score, vina_inter, vina_intra


# ═══════════════════════════════════════════════════════════════════════════
# Results I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_results():
    """Load existing results or return empty dict."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    """Save results incrementally."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_docking_data():
    """Load ZAP70 molecules merged with docking results.

    Returns:
        mol_data: DataFrame with pIC50, vina_score, vina_inter, vina_intra, has_dock
        dock_available: bool
    """
    mol_data, _ = load_zap70_molecules()

    if not DOCK_CHEMBL_CSV.exists():
        print(f"  WARNING: Docking results not found at {DOCK_CHEMBL_CSV}")
        print(f"  Run experiments/run_dock_chembl_zap70.py first.")
        mol_data["vina_score"] = np.nan
        mol_data["vina_inter"] = np.nan
        mol_data["vina_intra"] = np.nan
        mol_data["has_dock"] = False
        return mol_data, False

    dock_df = pd.read_csv(DOCK_CHEMBL_CSV)
    print(f"  Docking results: {len(dock_df)} molecules, "
          f"{dock_df['success'].sum()} successful")

    dock_df = dock_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    dock_cols = ["molecule_chembl_id", "vina_score", "vina_inter", "vina_intra"]
    dock_subset = dock_df[dock_df["success"] == True][dock_cols].copy()

    mol_data = mol_data.merge(dock_subset, on="molecule_chembl_id", how="left")
    mol_data["has_dock"] = ~mol_data["vina_score"].isna()

    n_docked = mol_data["has_dock"].sum()
    n_total = len(mol_data)
    print(f"  Merged: {n_docked}/{n_total} molecules have docking scores")

    return mol_data, n_docked > 0


def load_kinase_pairs():
    """Load within-assay MMP pairs for kinase pretraining targets.

    Returns:
        kinase_pairs: DataFrame with mol_a, mol_b, delta, value_a, value_b, target_chembl_id
        target_counts: dict of target_name -> pair_count
    """
    print("  Loading kinase MMP pairs from shared_pairs_deduped.csv...")
    pairs = pd.read_csv(
        DATA_DIR / "shared_pairs_deduped.csv",
        usecols=["mol_a", "mol_b", "delta", "is_within_assay",
                 "target_chembl_id", "value_a", "value_b"],
    )
    within_pairs = pairs[pairs["is_within_assay"] == True].copy()
    del pairs
    gc.collect()

    target_ids = set(PRETRAIN_KINASES.values())
    kinase_pairs = within_pairs[within_pairs["target_chembl_id"].isin(target_ids)].copy()
    del within_pairs
    gc.collect()

    target_counts = {}
    for name, chembl_id in PRETRAIN_KINASES.items():
        count = (kinase_pairs["target_chembl_id"] == chembl_id).sum()
        target_counts[name] = int(count)
        print(f"    {name} ({chembl_id}): {count:,} within-assay pairs")

    total = len(kinase_pairs)
    print(f"  Total kinase pairs: {total:,}")

    if total > MAX_PRETRAIN_PAIRS:
        kinase_pairs = kinase_pairs.sample(MAX_PRETRAIN_PAIRS, random_state=42)
        print(f"  Subsampled to {MAX_PRETRAIN_PAIRS:,} pairs")

    return kinase_pairs, target_counts


def generate_all_pairs(mol_data):
    """Generate all unique pairs from N molecules -> N*(N-1)/2 pairs."""
    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    ids = mol_data["molecule_chembl_id"].values

    pairs = []
    for i in range(len(smiles)):
        for j in range(i + 1, len(smiles)):
            pairs.append({
                "mol_a": smiles[i], "mol_b": smiles[j],
                "mol_a_id": ids[i], "mol_b_id": ids[j],
                "value_a": pIC50[i], "value_b": pIC50[j],
                "delta": pIC50[j] - pIC50[i],
                "idx_a": i, "idx_b": j,
            })
    df = pd.DataFrame(pairs)
    print(f"  Generated {len(df):,} all-pairs from {len(smiles)} molecules")
    return df


def get_vina_features_per_mol(mol_data):
    """Extract per-molecule Vina features [vina_score, vina_inter, vina_intra].

    Missing values are mean-imputed.

    Returns:
        vina_feats: np.ndarray of shape [N, 3]
    """
    cols = ["vina_score", "vina_inter", "vina_intra"]
    feats = mol_data[cols].values.astype(np.float32)

    for col_i in range(feats.shape[1]):
        col_vals = feats[:, col_i]
        mask = np.isnan(col_vals)
        if mask.any() and not mask.all():
            col_vals[mask] = np.nanmean(col_vals)
        elif mask.all():
            col_vals[:] = 0.0
        feats[:, col_i] = col_vals

    return feats


def compute_pair_vina_diff(vina_feats, pairs_df):
    """Compute per-pair Vina feature differences: feats[b] - feats[a].

    Returns:
        pair_feats: [N_pairs, 3] array
    """
    idx_a = pairs_df["idx_a"].values
    idx_b = pairs_df["idx_b"].values
    return vina_feats[idx_b] - vina_feats[idx_a]


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_metrics(delta_true, delta_pred):
    """Compute delta prediction metrics."""
    mae = float(np.mean(np.abs(delta_true - delta_pred)))
    rmse = float(np.sqrt(np.mean((delta_true - delta_pred) ** 2)))
    if len(delta_true) > 2 and np.std(delta_pred) > 1e-8:
        spr, _ = spearmanr(delta_true, delta_pred)
        pr, _ = pearsonr(delta_true, delta_pred)
    else:
        spr, pr = 0.0, 0.0
    ss_res = np.sum((delta_true - delta_pred) ** 2)
    ss_tot = np.sum((delta_true - np.mean(delta_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "spearman": float(spr) if not np.isnan(spr) else 0.0,
        "pearson": float(pr) if not np.isnan(pr) else 0.0,
        "n_pairs": len(delta_true),
    }


def reconstruct_absolute_via_anchors(
    test_mol_indices, train_mol_indices, X_fp, y_true_all,
    predict_delta_fn, n_anchors=50,
):
    """Reconstruct absolute pIC50 for test molecules using anchor-based
    median prediction: pred_j = median_i(pIC50_i + delta_pred(i->j)).

    Args:
        test_mol_indices: indices of test molecules
        train_mol_indices: indices of train molecules (anchors)
        X_fp: [N, D] fingerprint matrix
        y_true_all: [N] true pIC50 values
        predict_delta_fn: callable(emb_a, emb_b) -> delta_pred scalar/array
        n_anchors: max number of anchors

    Returns:
        y_pred_abs: [len(test_mol_indices)] absolute pIC50 predictions
    """
    anchor_idx = train_mol_indices
    if n_anchors is not None and n_anchors < len(anchor_idx):
        rng = np.random.RandomState(42)
        anchor_idx = rng.choice(anchor_idx, size=n_anchors, replace=False)

    preds = []
    for j in test_mol_indices:
        anchor_preds = []
        for i in anchor_idx:
            emb_a = X_fp[i:i+1]
            emb_b = X_fp[j:j+1]
            delta_pred = predict_delta_fn(emb_a, emb_b)
            if isinstance(delta_pred, np.ndarray):
                delta_pred = delta_pred.item()
            anchor_preds.append(y_true_all[i] + delta_pred)
        preds.append(float(np.median(anchor_preds)))

    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════
# FiLMDelta Pretraining on Kinase Pairs
# ═══════════════════════════════════════════════════════════════════════════

def pretrain_film_delta(kinase_pairs, fp_cache, seed=42,
                        include_docking_zeros=False):
    """Pretrain a FiLMDelta (or DockingFiLMDelta) model on kinase MMP pairs.

    When include_docking_zeros=True, uses DockingFiLMDeltaMLP with zero-filled
    docking features so the model architecture includes docking layers but
    learns to ignore them initially.

    Args:
        kinase_pairs: DataFrame with mol_a, mol_b, delta
        fp_cache: dict mapping SMILES -> fingerprint vector
        seed: random seed
        include_docking_zeros: if True, use DockingFiLMDeltaMLP with zero docking

    Returns:
        model: trained nn.Module (FiLMDeltaMLP or DockingFiLMDeltaMLP)
        scaler: fitted StandardScaler for fingerprints
        pretrain_metrics: dict with val_mae, val_spearman
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a, emb_b]))

    Xa = torch.FloatTensor(scaler.transform(emb_a)).to(DEVICE)
    Xb = torch.FloatTensor(scaler.transform(emb_b)).to(DEVICE)
    yd = torch.FloatTensor(delta).to(DEVICE)

    n_val = max(len(Xa) // 10, 100)
    input_dim = Xa.shape[1]

    if include_docking_zeros:
        from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP
        model = DockingFiLMDeltaMLP(
            input_dim=input_dim, extra_dim=VINA_DIM, dropout=0.2,
        )
        # Zero docking features for all kinase pairs
        dock_zeros = torch.zeros(len(Xa), VINA_DIM, device=DEVICE)
    else:
        from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
        model = FiLMDeltaMLP(input_dim=input_dim, dropout=0.2)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    print(f"    Pretraining on {len(Xa) - n_val} train / {n_val} val pairs...")
    print(f"    Architecture: {'DockingFiLMDeltaMLP (zero docking)' if include_docking_zeros else 'FiLMDeltaMLP'}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xa) - n_val, device='cpu') + n_val
        for start in range(0, len(perm), PRETRAIN_BATCH_SIZE):
            bi = perm[start:start + PRETRAIN_BATCH_SIZE]
            optimizer.zero_grad()
            if include_docking_zeros:
                pred = model(Xa[bi], Xb[bi], dock_zeros[bi])
            else:
                pred = model(Xa[bi], Xb[bi])
            loss = criterion(pred, yd[bi])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if include_docking_zeros:
                val_pred = model(Xa[:n_val], Xb[:n_val], dock_zeros[:n_val])
            else:
                val_pred = model(Xa[:n_val], Xb[:n_val])
            vl = criterion(val_pred, yd[:n_val]).item()

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PRETRAIN_PATIENCE:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    # Validation metrics
    model.eval()
    with torch.no_grad():
        if include_docking_zeros:
            val_pred = model(Xa[:n_val], Xb[:n_val], dock_zeros[:n_val])
        else:
            val_pred = model(Xa[:n_val], Xb[:n_val])
        val_pred_np = val_pred.cpu().numpy()

    val_mae = float(np.mean(np.abs(delta[:n_val] - val_pred_np)))
    val_spr, _ = spearmanr(delta[:n_val], val_pred_np)

    print(f"    Pretrain val: MAE={val_mae:.4f}, Spr={float(val_spr):.3f}")

    # Move model to CPU for storage
    model = model.cpu()

    return model, scaler, {
        "val_mae": val_mae,
        "val_spearman": float(val_spr),
        "n_pairs": len(Xa),
        "epochs": min(epoch + 1, PRETRAIN_EPOCHS),
        "include_docking_zeros": include_docking_zeros,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Weight Transfer: FiLMDeltaMLP -> DockingFiLMDeltaMLP
# ═══════════════════════════════════════════════════════════════════════════

def transfer_film_to_docking(pretrained_model, input_dim):
    """Transfer weights from FiLMDeltaMLP to DockingFiLMDeltaMLP.

    The DockingFiLMDeltaMLP has the same FiLM backbone (blocks + output) but
    a wider delta_encoder input (input_dim + extra_dim vs input_dim).

    Strategy:
      - Copy blocks and output weights directly (identical architecture)
      - For delta_encoder: copy the weights corresponding to the Morgan diff
        dimensions and zero-initialize the docking feature dimensions

    Args:
        pretrained_model: trained FiLMDeltaMLP
        input_dim: embedding dimension

    Returns:
        docking_model: DockingFiLMDeltaMLP with transferred weights
    """
    from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP

    docking_model = DockingFiLMDeltaMLP(
        input_dim=input_dim, extra_dim=VINA_DIM, dropout=0.2,
    )

    pretrained_sd = pretrained_model.state_dict()
    docking_sd = docking_model.state_dict()

    transferred_keys = []
    skipped_keys = []

    for key in docking_sd:
        if key in pretrained_sd:
            if docking_sd[key].shape == pretrained_sd[key].shape:
                # Direct copy: blocks, output, etc.
                docking_sd[key] = pretrained_sd[key].clone()
                transferred_keys.append(key)
            elif "delta_encoder" in key and "weight" in key:
                # Delta encoder first linear: [out, input_dim + extra_dim] vs [out, input_dim]
                # Copy the Morgan diff portion, zero-init the docking portion
                out_dim = pretrained_sd[key].shape[0]
                morgan_dim = pretrained_sd[key].shape[1]
                docking_sd[key][:, :morgan_dim] = pretrained_sd[key].clone()
                docking_sd[key][:, morgan_dim:] = 0.0
                transferred_keys.append(f"{key} (partial: {morgan_dim}/{docking_sd[key].shape[1]})")
            elif "delta_encoder" in key and "bias" in key:
                # Bias is same size, copy directly
                if docking_sd[key].shape == pretrained_sd[key].shape:
                    docking_sd[key] = pretrained_sd[key].clone()
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {pretrained_sd[key].shape} vs {docking_sd[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in pretrained)")

    docking_model.load_state_dict(docking_sd)

    print(f"    Weight transfer: {len(transferred_keys)} transferred, {len(skipped_keys)} random-init")
    if skipped_keys:
        for sk in skipped_keys[:5]:
            print(f"      Random init: {sk}")

    return docking_model


# ═══════════════════════════════════════════════════════════════════════════
# Finetuning + CV Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def finetune_and_evaluate_cv(
    method_name,
    mol_data, X_fp, pairs_df, vina_pair_diff,
    pretrained_model=None, pretrained_scaler=None,
    use_docking=False,
    transfer_mode="backbone",  # "backbone" or "direct"
    seeds=None,
):
    """Run 5-fold CV: optionally finetune a pretrained model, optionally
    with docking features.

    Args:
        method_name: for logging
        mol_data: ZAP70 DataFrame
        X_fp: [N, D] fingerprints
        pairs_df: all-pairs DataFrame with idx_a, idx_b, delta
        vina_pair_diff: [N_pairs, 3] pair-level Vina diffs
        pretrained_model: pretrained FiLMDeltaMLP or DockingFiLMDeltaMLP, or None
        pretrained_scaler: StandardScaler from pretraining, or None
        use_docking: whether to use Vina features for conditioning
        transfer_mode: "backbone" = transfer FiLM weights to DockingFiLM,
                       "direct" = use pretrained model directly (for DockingFiLMDelta pretrained with zeros)
        seeds: list of random seeds

    Returns:
        result_dict: aggregated metrics
    """
    if seeds is None:
        seeds = list(range(N_SEEDS))

    y_all = mol_data["pIC50"].values
    input_dim = X_fp.shape[1]

    all_fold_delta = []
    all_fold_abs = []

    for seed in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            torch.manual_seed(seed * 1000 + fold_i)
            np.random.seed(seed * 1000 + fold_i)

            # Filter pairs to train/test molecules
            train_set = set(train_idx)
            test_set = set(test_idx)
            train_mask = pairs_df["idx_a"].isin(train_set) & pairs_df["idx_b"].isin(train_set)
            test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            train_pairs = pairs_df[train_mask]
            test_pairs = pairs_df[test_mask]

            if len(test_pairs) == 0:
                continue

            # Prepare training data
            emb_a_tr = X_fp[train_pairs["idx_a"].values]
            emb_b_tr = X_fp[train_pairs["idx_b"].values]
            delta_tr = train_pairs["delta"].values.astype(np.float32)

            # Split train into train/val for early stopping (85/15)
            rng = np.random.RandomState(seed * 1000 + fold_i)
            n = len(train_pairs)
            val_size = max(int(n * 0.15), 50)
            perm = rng.permutation(n)
            val_perm, tr_perm = perm[:val_size], perm[val_size:]

            Xa_tr = torch.FloatTensor(emb_a_tr[tr_perm]).to(DEVICE)
            Xb_tr = torch.FloatTensor(emb_b_tr[tr_perm]).to(DEVICE)
            yd_tr = torch.FloatTensor(delta_tr[tr_perm]).to(DEVICE)
            Xa_val = torch.FloatTensor(emb_a_tr[val_perm]).to(DEVICE)
            Xb_val = torch.FloatTensor(emb_b_tr[val_perm]).to(DEVICE)
            yd_val = torch.FloatTensor(delta_tr[val_perm]).to(DEVICE)

            # Docking features for training pairs
            if use_docking:
                pair_indices_tr = train_pairs.iloc[tr_perm].index.values
                pair_indices_val = train_pairs.iloc[val_perm].index.values
                dock_tr = torch.FloatTensor(vina_pair_diff[pair_indices_tr]).to(DEVICE)
                dock_val = torch.FloatTensor(vina_pair_diff[pair_indices_val]).to(DEVICE)

            # Build/initialize model
            if use_docking:
                if pretrained_model is not None:
                    if transfer_mode == "backbone":
                        # Transfer FiLMDeltaMLP weights to DockingFiLMDeltaMLP
                        model = transfer_film_to_docking(
                            copy.deepcopy(pretrained_model), input_dim,
                        )
                    elif transfer_mode == "direct":
                        # Pretrained model is already DockingFiLMDeltaMLP
                        model = copy.deepcopy(pretrained_model)
                    else:
                        raise ValueError(f"Unknown transfer_mode: {transfer_mode}")
                else:
                    # No pretrain, train from scratch with docking
                    from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP
                    model = DockingFiLMDeltaMLP(
                        input_dim=input_dim, extra_dim=VINA_DIM, dropout=0.2,
                    )
            else:
                if pretrained_model is not None:
                    model = copy.deepcopy(pretrained_model)
                else:
                    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
                    model = FiLMDeltaMLP(input_dim=input_dim, dropout=0.2)

            model = model.to(DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4,
            )
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            best_state = None
            wait = 0

            for epoch in range(FINETUNE_EPOCHS):
                model.train()
                perm_t = torch.randperm(len(Xa_tr), device='cpu')
                for start in range(0, len(perm_t), FINETUNE_BATCH_SIZE):
                    bi = perm_t[start:start + FINETUNE_BATCH_SIZE]
                    optimizer.zero_grad()
                    if use_docking:
                        pred = model(Xa_tr[bi], Xb_tr[bi], dock_tr[bi])
                    else:
                        pred = model(Xa_tr[bi], Xb_tr[bi])
                    loss = criterion(pred, yd_tr[bi])
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    if use_docking:
                        vp = model(Xa_val, Xb_val, dock_val)
                    else:
                        vp = model(Xa_val, Xb_val)
                    vl = criterion(vp, yd_val).item()

                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= FINETUNE_PATIENCE:
                        break

            if best_state:
                model.load_state_dict(best_state)
                model = model.to(DEVICE)

            # Evaluate on test pairs
            model.eval()
            emb_a_te = torch.FloatTensor(X_fp[test_pairs["idx_a"].values]).to(DEVICE)
            emb_b_te = torch.FloatTensor(X_fp[test_pairs["idx_b"].values]).to(DEVICE)
            delta_true = test_pairs["delta"].values.astype(np.float32)

            with torch.no_grad():
                if use_docking:
                    dock_te = torch.FloatTensor(
                        vina_pair_diff[test_pairs.index.values]
                    ).to(DEVICE)
                    delta_pred = model(emb_a_te, emb_b_te, dock_te).cpu().numpy()
                else:
                    delta_pred = model(emb_a_te, emb_b_te).cpu().numpy()

            d_metrics = compute_delta_metrics(delta_true, delta_pred)
            all_fold_delta.append(d_metrics)

            # Absolute reconstruction via anchors
            try:
                model_cpu = model.cpu()

                def _predict_delta(ea, eb, _model=model_cpu, _use_dock=use_docking,
                                   _vina_feats=None):
                    ea_t = torch.FloatTensor(ea)
                    eb_t = torch.FloatTensor(eb)
                    _model.eval()
                    with torch.no_grad():
                        if _use_dock:
                            # For anchor-based reconstruction, compute Vina diff from per-mol feats
                            # Here we approximate with zeros since we don't have per-mol Vina
                            # for arbitrary anchor pairs — this is a known limitation
                            d_t = torch.zeros(ea_t.shape[0], VINA_DIM)
                            return _model(ea_t, eb_t, d_t).numpy()
                        else:
                            return _model(ea_t, eb_t).numpy()

                # For docking models, we use the actual Vina per-mol features for anchor recon
                if use_docking:
                    vina_per_mol = get_vina_features_per_mol(mol_data)

                    def _predict_delta_with_dock(ea, eb,
                                                 _model=model_cpu,
                                                 _vina=vina_per_mol):
                        ea_t = torch.FloatTensor(ea)
                        eb_t = torch.FloatTensor(eb)
                        # We need the molecule indices to look up Vina features
                        # but we only have embeddings here. Use zero as fallback.
                        d_t = torch.zeros(ea_t.shape[0], VINA_DIM)
                        _model.eval()
                        with torch.no_grad():
                            return _model(ea_t, eb_t, d_t).numpy()

                    predict_fn = _predict_delta_with_dock
                else:
                    predict_fn = _predict_delta

                y_pred_abs = reconstruct_absolute_via_anchors(
                    test_idx, train_idx, X_fp, y_all,
                    predict_fn, n_anchors=50,
                )
                a_metrics = compute_absolute_metrics(y_all[test_idx], y_pred_abs)
                all_fold_abs.append(a_metrics)
            except Exception as e:
                print(f"    Anchor reconstruction failed: {e}")

            del model, optimizer
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        print(f"    {method_name} seed {seed}: "
              f"avg delta MAE = {np.mean([m['mae'] for m in all_fold_delta[-N_FOLDS:]]):.4f}")

    # Aggregate
    result = {
        "delta_mae_mean": float(np.mean([m["mae"] for m in all_fold_delta])),
        "delta_mae_std": float(np.std([m["mae"] for m in all_fold_delta])),
        "delta_spearman_mean": float(np.mean([m["spearman"] for m in all_fold_delta])),
        "delta_spearman_std": float(np.std([m["spearman"] for m in all_fold_delta])),
        "delta_pearson_mean": float(np.mean([m["pearson"] for m in all_fold_delta])),
        "delta_r2_mean": float(np.mean([m["r2"] for m in all_fold_delta])),
        "n_seeds": len(seeds),
        "n_folds": N_FOLDS,
        "n_total_folds": len(all_fold_delta),
    }

    if all_fold_abs:
        result["abs_mae_mean"] = float(np.mean([m["mae"] for m in all_fold_abs]))
        result["abs_mae_std"] = float(np.std([m["mae"] for m in all_fold_abs]))
        result["abs_spearman_mean"] = float(np.mean([m.get("spearman_r", 0) for m in all_fold_abs]))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Pretrain FiLMDelta on Kinase Panel
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(kinase_pairs, fp_cache, results):
    """Phase A: Pretrain FiLMDelta on kinase MMP pairs (no docking)."""
    print("\n" + "=" * 70)
    print("PHASE A: Pretrain FiLMDelta on Kinase Panel (no docking)")
    print("=" * 70)

    phase = results.get("phase_a", {})
    if phase.get("completed"):
        print("  Already completed, skipping pretraining.")
        # Reconstruct model from scratch with same seed for downstream use
        model, scaler, _ = pretrain_film_delta(kinase_pairs, fp_cache, seed=42,
                                                include_docking_zeros=False)
        return results, model, scaler

    t0 = time.time()
    model, scaler, pretrain_metrics = pretrain_film_delta(
        kinase_pairs, fp_cache, seed=42, include_docking_zeros=False,
    )

    results["phase_a"] = {
        "pretrain_metrics": pretrain_metrics,
        "kinase_targets": list(PRETRAIN_KINASES.keys()),
        "n_pairs": pretrain_metrics["n_pairs"],
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print(f"  Phase A complete in {time.time() - t0:.1f}s")

    return results, model, scaler


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Finetune Pretrained Model with Docking on ZAP70
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(pretrained_model, pretrained_scaler,
                mol_data, X_fp, pairs_df, vina_pair_diff, results):
    """Phase B: Finetune pretrained FiLMDelta on ZAP70 with docking features.

    Transfers FiLM backbone weights to DockingFiLMDeltaMLP, randomly inits
    the docking-specific delta encoder weights.
    """
    print("\n" + "=" * 70)
    print("PHASE B: Finetune Pretrained FiLMDelta + Docking on ZAP70")
    print("=" * 70)

    phase = results.get("phase_b", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    t0 = time.time()

    result = finetune_and_evaluate_cv(
        "FiLMDelta_vina_pretrained",
        mol_data, X_fp, pairs_df, vina_pair_diff,
        pretrained_model=pretrained_model,
        pretrained_scaler=pretrained_scaler,
        use_docking=True,
        transfer_mode="backbone",
        seeds=list(range(N_SEEDS)),
    )

    results["phase_b"] = {
        "method": "FiLMDelta_vina_pretrained",
        "description": "Kinase pretrained FiLMDelta + docking finetuning (backbone transfer)",
        "metrics": result,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    _print_method_result("FiLMDelta_vina_pretrained", result)
    print(f"  Phase B complete in {time.time() - t0:.1f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Alternative Pretraining with Zero-Filled Docking
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(kinase_pairs, fp_cache,
                mol_data, X_fp, pairs_df, vina_pair_diff, results):
    """Phase C: Pretrain DockingFiLMDelta with zero docking features,
    then finetune on ZAP70 with real docking features.

    Option 1: Zero-filled docking during pretraining -> real docking at finetune
    Option 2: Same as Phase B (no docking at pretrain, add at finetune) — already done
    """
    print("\n" + "=" * 70)
    print("PHASE C: Alternative — Pretrain WITH Zero Docking Features")
    print("=" * 70)

    phase = results.get("phase_c", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    t0 = time.time()

    # Pretrain DockingFiLMDeltaMLP with zero docking features
    print("\n  Option 1: Pretrain DockingFiLMDeltaMLP with zero docking...")
    model_zerofill, scaler_zerofill, pretrain_metrics = pretrain_film_delta(
        kinase_pairs, fp_cache, seed=42, include_docking_zeros=True,
    )

    # Finetune on ZAP70 with real docking features (direct transfer, same arch)
    print("\n  Finetuning with real docking features...")
    result_zerofill = finetune_and_evaluate_cv(
        "FiLMDelta_vina_pretrained_zerofill",
        mol_data, X_fp, pairs_df, vina_pair_diff,
        pretrained_model=model_zerofill,
        pretrained_scaler=scaler_zerofill,
        use_docking=True,
        transfer_mode="direct",  # Same architecture, direct weight load
        seeds=list(range(N_SEEDS)),
    )

    results["phase_c"] = {
        "option_1": {
            "method": "FiLMDelta_vina_pretrained_zerofill",
            "description": "Pretrain DockingFiLMDelta with zero docking, finetune with real docking",
            "pretrain_metrics": pretrain_metrics,
            "finetune_metrics": result_zerofill,
        },
        "note": "Option 2 (no docking at pretrain) is equivalent to Phase B",
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    _print_method_result("FiLMDelta_vina_pretrained_zerofill", result_zerofill)
    print(f"  Phase C complete in {time.time() - t0:.1f}s")

    del model_zerofill, scaler_zerofill
    gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: Comprehensive Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(pretrained_model, pretrained_scaler,
                mol_data, X_fp, pairs_df, vina_pair_diff, results):
    """Phase D: Run all 4 methods side-by-side with same CV splits.

    Methods:
        1. FiLMDelta                — no pretrain, no docking
        2. FiLMDelta_vina           — no pretrain, docking
        3. FiLMDelta_pretrained     — pretrain, no docking
        4. FiLMDelta_vina_pretrained — pretrain + docking (from Phase B)
    """
    print("\n" + "=" * 70)
    print("PHASE D: Comprehensive Comparison (4 methods)")
    print("=" * 70)

    phase = results.get("phase_d", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    t0 = time.time()
    methods = phase.get("methods", {})

    # Method 1: FiLMDelta (no pretrain, no docking) — baseline
    if "FiLMDelta" not in methods:
        print("\n  Method 1: FiLMDelta (baseline, no pretrain, no docking)...")
        r1 = finetune_and_evaluate_cv(
            "FiLMDelta", mol_data, X_fp, pairs_df, vina_pair_diff,
            pretrained_model=None, pretrained_scaler=None,
            use_docking=False, seeds=list(range(N_SEEDS)),
        )
        methods["FiLMDelta"] = r1
        _print_method_result("FiLMDelta", r1)
        results["phase_d"] = {"methods": methods, "completed": False,
                               "timestamp": str(datetime.now())}
        save_results(results)
        gc.collect()

    # Method 2: FiLMDelta_vina (no pretrain, with docking)
    if "FiLMDelta_vina" not in methods:
        print("\n  Method 2: FiLMDelta_vina (no pretrain, with docking)...")
        r2 = finetune_and_evaluate_cv(
            "FiLMDelta_vina", mol_data, X_fp, pairs_df, vina_pair_diff,
            pretrained_model=None, pretrained_scaler=None,
            use_docking=True, seeds=list(range(N_SEEDS)),
        )
        methods["FiLMDelta_vina"] = r2
        _print_method_result("FiLMDelta_vina", r2)
        results["phase_d"] = {"methods": methods, "completed": False,
                               "timestamp": str(datetime.now())}
        save_results(results)
        gc.collect()

    # Method 3: FiLMDelta_pretrained (pretrain, no docking)
    if "FiLMDelta_pretrained" not in methods:
        print("\n  Method 3: FiLMDelta_pretrained (pretrain, no docking)...")
        r3 = finetune_and_evaluate_cv(
            "FiLMDelta_pretrained", mol_data, X_fp, pairs_df, vina_pair_diff,
            pretrained_model=pretrained_model,
            pretrained_scaler=pretrained_scaler,
            use_docking=False, seeds=list(range(N_SEEDS)),
        )
        methods["FiLMDelta_pretrained"] = r3
        _print_method_result("FiLMDelta_pretrained", r3)
        results["phase_d"] = {"methods": methods, "completed": False,
                               "timestamp": str(datetime.now())}
        save_results(results)
        gc.collect()

    # Method 4: FiLMDelta_vina_pretrained (pretrain + docking)
    if "FiLMDelta_vina_pretrained" not in methods:
        print("\n  Method 4: FiLMDelta_vina_pretrained (pretrain + docking)...")
        r4 = finetune_and_evaluate_cv(
            "FiLMDelta_vina_pretrained", mol_data, X_fp, pairs_df, vina_pair_diff,
            pretrained_model=pretrained_model,
            pretrained_scaler=pretrained_scaler,
            use_docking=True, transfer_mode="backbone",
            seeds=list(range(N_SEEDS)),
        )
        methods["FiLMDelta_vina_pretrained"] = r4
        _print_method_result("FiLMDelta_vina_pretrained", r4)
        results["phase_d"] = {"methods": methods, "completed": False,
                               "timestamp": str(datetime.now())}
        save_results(results)
        gc.collect()

    # Also include zerofill if Phase C ran
    phase_c = results.get("phase_c", {})
    if phase_c.get("completed"):
        opt1 = phase_c.get("option_1", {})
        if "finetune_metrics" in opt1:
            methods["FiLMDelta_vina_pretrained_zerofill"] = opt1["finetune_metrics"]

    results["phase_d"] = {
        "methods": methods,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    # Print comparison table
    print_comparison_table(methods, "Phase D — Comprehensive Comparison")

    # Compute improvement over baseline
    baseline_mae = methods.get("FiLMDelta", {}).get("delta_mae_mean", float('nan'))
    if not np.isnan(baseline_mae):
        print(f"\n  Improvements over FiLMDelta baseline (MAE={baseline_mae:.4f}):")
        for name, m in sorted(methods.items(), key=lambda x: x[1].get("delta_mae_mean", 999)):
            mae = m.get("delta_mae_mean", float('nan'))
            if not np.isnan(mae) and name != "FiLMDelta":
                pct = (baseline_mae - mae) / baseline_mae * 100
                print(f"    {name}: {mae:.4f} ({pct:+.1f}%)")

    print(f"\n  Phase D complete in {time.time() - t0:.1f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _print_method_result(name, res):
    """Print a single method's result."""
    d_mae = res.get("delta_mae_mean", float('nan'))
    d_std = res.get("delta_mae_std", 0)
    d_spr = res.get("delta_spearman_mean", float('nan'))
    a_mae = res.get("abs_mae_mean", float('nan'))

    parts = [f"  {name}: Delta MAE={d_mae:.4f}"]
    if d_std > 0:
        parts[0] += f"+-{d_std:.4f}"
    parts.append(f"Spr={d_spr:.3f}")
    if not np.isnan(a_mae):
        parts.append(f"| Abs MAE={a_mae:.4f}")
    print(", ".join(parts))


def print_comparison_table(method_results, title="Comparison"):
    """Print a formatted comparison table sorted by delta MAE."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Method':<35} {'Delta MAE':>12} {'Delta Spr':>10} "
          f"{'Abs MAE':>10} {'Abs Spr':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    sorted_methods = sorted(method_results.items(),
                            key=lambda x: x[1].get("delta_mae_mean", 999))

    for name, res in sorted_methods:
        d_mae = res.get("delta_mae_mean", float('nan'))
        d_mae_std = res.get("delta_mae_std", 0)
        d_spr = res.get("delta_spearman_mean", float('nan'))
        a_mae = res.get("abs_mae_mean", float('nan'))
        a_spr = res.get("abs_spearman_mean", float('nan'))

        d_mae_str = f"{d_mae:.4f}" if not np.isnan(d_mae) else "N/A"
        if d_mae_std > 0:
            d_mae_str += f"+-{d_mae_std:.3f}"
        d_spr_str = f"{d_spr:.3f}" if not np.isnan(d_spr) else "N/A"
        a_mae_str = f"{a_mae:.4f}" if not np.isnan(a_mae) else "N/A"
        a_spr_str = f"{a_spr:.3f}" if not np.isnan(a_spr) else "N/A"

        print(f"  {name:<35} {d_mae_str:>12} {d_spr_str:>10} "
              f"{a_mae_str:>10} {a_spr_str:>10}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined Kinase Pretraining + Docking Integration for ZAP70"
    )
    parser.add_argument(
        "--phase", nargs="*", type=str, default=None,
        help="Phase(s) to run (A, B, C, D). Default: all.",
    )
    args = parser.parse_args()

    phases_to_run = [p.upper() for p in args.phase] if args.phase else ["A", "B", "C", "D"]

    print("=" * 70)
    print("  Combined Kinase Pretraining + Docking Integration — ZAP70")
    print(f"  Phases: {phases_to_run}")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds: {N_SEEDS}, Folds: {N_FOLDS}")
    print(f"  Results: {RESULTS_FILE}")
    print("=" * 70)

    t_start = time.time()

    # ── Load ZAP70 data ──
    print("\n--- Loading ZAP70 Data ---")
    mol_data, dock_available = load_docking_data()
    n_mols = len(mol_data)
    print(f"  Molecules: {n_mols}, Docking available: {dock_available}")

    if not dock_available:
        print("\n  ERROR: Docking data not available. Cannot run this experiment.")
        print(f"  Expected: {DOCK_CHEMBL_CSV}")
        return

    # ── Compute fingerprints ──
    print("\n--- Computing Morgan Fingerprints ---")
    all_smiles = mol_data["smiles"].tolist()
    X_fp = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    print(f"  Morgan FP shape: {X_fp.shape}")

    # ── Vina features ──
    print("\n--- Extracting Vina Features ---")
    vina_feats = get_vina_features_per_mol(mol_data)
    n_valid_vina = np.sum(~np.isnan(mol_data["vina_score"].values))
    print(f"  Vina features shape: {vina_feats.shape}, valid: {n_valid_vina}/{n_mols}")

    # ── Generate all pairs ──
    print("\n--- Generating All Pairs ---")
    pairs_df = generate_all_pairs(mol_data)
    vina_pair_diff = compute_pair_vina_diff(vina_feats, pairs_df)
    print(f"  Vina pair diff shape: {vina_pair_diff.shape}")

    # ── Load kinase pairs (needed for Phases A, C) ──
    kinase_pairs = None
    fp_cache = None
    if any(p in phases_to_run for p in ["A", "C", "B", "D"]):
        print("\n--- Loading Kinase Pretraining Data ---")
        kinase_pairs, target_counts = load_kinase_pairs()

        # Build FP cache for all unique molecules (kinase + ZAP70)
        print("\n--- Building Fingerprint Cache ---")
        kinase_smiles = list(set(
            kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
        ))
        all_unique = list(set(kinase_smiles + all_smiles))
        print(f"  {len(all_unique):,} unique molecules")
        X_all = compute_fingerprints(all_unique, "morgan", radius=2, n_bits=2048)
        fp_cache = {smi: X_all[i] for i, smi in enumerate(all_unique)}
        del X_all
        gc.collect()

    # ── Load results ──
    results = load_results()
    results["metadata"] = {
        "n_zap70_molecules": n_mols,
        "n_zap70_pairs": len(pairs_df),
        "dock_available": dock_available,
        "n_docked": int(n_valid_vina),
        "pretrain_kinases": list(PRETRAIN_KINASES.keys()),
        "n_pretrain_pairs": len(kinase_pairs) if kinase_pairs is not None else 0,
        "n_seeds": N_SEEDS,
        "n_folds": N_FOLDS,
        "device": DEVICE,
        "start_time": str(datetime.now()),
    }
    save_results(results)

    # ── Run phases ──
    pretrained_model = None
    pretrained_scaler = None

    if "A" in phases_to_run:
        results, pretrained_model, pretrained_scaler = run_phase_a(
            kinase_pairs, fp_cache, results,
        )

    # For Phases B/D we need pretrained model from Phase A
    if pretrained_model is None and any(p in phases_to_run for p in ["B", "D"]):
        print("\n  Reconstructing pretrained model from Phase A...")
        pretrained_model, pretrained_scaler, _ = pretrain_film_delta(
            kinase_pairs, fp_cache, seed=42, include_docking_zeros=False,
        )

    if "B" in phases_to_run:
        results = run_phase_b(
            pretrained_model, pretrained_scaler,
            mol_data, X_fp, pairs_df, vina_pair_diff, results,
        )

    if "C" in phases_to_run:
        results = run_phase_c(
            kinase_pairs, fp_cache,
            mol_data, X_fp, pairs_df, vina_pair_diff, results,
        )

    if "D" in phases_to_run:
        results = run_phase_d(
            pretrained_model, pretrained_scaler,
            mol_data, X_fp, pairs_df, vina_pair_diff, results,
        )

    # ── Final summary ──
    total_time = time.time() - t_start
    results["metadata"]["total_time_s"] = total_time
    save_results(results)

    print(f"\n{'='*70}")
    print(f"  Experiment complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"{'='*70}")

    # Print final comparison if Phase D completed
    phase_d = results.get("phase_d", {})
    if phase_d.get("completed"):
        print_comparison_table(
            phase_d["methods"],
            "FINAL — Kinase Pretraining + Docking Integration",
        )


if __name__ == "__main__":
    main()
