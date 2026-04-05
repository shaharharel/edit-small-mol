#!/usr/bin/env python3
"""
Docking Integration — Iteration 3: Pretrain DockingFiLM with REAL docking features.

Previous iterations pretrained FiLMDelta on kinase data WITHOUT docking features,
then transferred weights to DockingFiLM. The docking-specific weights were random-init
at finetuning time, meaning the model had to learn docking signal from only 280 ZAP70 mols.

**Key insight**: The kinase panel docking is COMPLETE (34K molecules docked against ZAP70).
91% of kinase pretraining molecules have docking scores. We can pretrain DockingFiLMDeltaMLP
directly with real Vina features so the model learns to use 3D info from the start.

Methods:
  1. DockFiLM_real_pretrained — pretrain DockingFiLM WITH real Vina diffs on kinase pairs
  2. DockFiLM_real_pretrained_ensemble — 5-seed ensemble of method 1
  3. FeatureGated_real_pretrained — pretrain FeatureGatedFiLM with real Vina features
  4. ResidCorr_real_pretrained — pretrain base FiLMDelta + residual correction with docking
  5. MultiSeed_mixed_ensemble — ensemble of pretrained DockFiLM + FeatureGated + ResidCorr

Usage:
    conda run -n quris python -u experiments/run_docking_iteration3.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    N_FOLDS, CV_SEED,
)
from experiments.run_paper_evaluation import RESULTS_DIR

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "docking_iteration3_results.json"
DEVICE = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else "cpu"

DOCK_CHEMBL_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCK_CHEMBL_CSV = DOCK_CHEMBL_DIR / "docking_results.csv"

N_SEEDS = 3
SEEDS = [42, 123, 456]
VINA_DIM = 3

# Kinase targets for pretraining
PRETRAIN_KINASES = {
    "SYK": "CHEMBL2599", "LCK": "CHEMBL258", "JAK2": "CHEMBL2971",
    "ABL1": "CHEMBL1862", "SRC": "CHEMBL267", "BTK": "CHEMBL5251",
}


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def compute_delta_metrics(delta_true, delta_pred):
    mae = float(np.mean(np.abs(delta_true - delta_pred)))
    if len(delta_true) > 2 and np.std(delta_pred) > 1e-8:
        spr, _ = spearmanr(delta_true, delta_pred)
        pr, _ = pearsonr(delta_true, delta_pred)
    else:
        spr, pr = 0.0, 0.0
    ss_res = np.sum((delta_true - delta_pred) ** 2)
    ss_tot = np.sum((delta_true - np.mean(delta_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "spearman": float(spr) if not np.isnan(spr) else 0.0,
            "pearson": float(pr) if not np.isnan(pr) else 0.0, "r2": r2}


def reconstruct_absolute(test_idx, train_idx, X_fp, y_all, predict_fn, n_anchors=50):
    anchor_idx = train_idx
    if n_anchors < len(anchor_idx):
        rng = np.random.RandomState(42)
        anchor_idx = rng.choice(anchor_idx, size=n_anchors, replace=False)
    preds = []
    for j in test_idx:
        apreds = []
        for i in anchor_idx:
            dp = predict_fn(X_fp[i:i+1], X_fp[j:j+1])
            if isinstance(dp, np.ndarray):
                dp = dp.item()
            apreds.append(y_all[i] + dp)
        preds.append(float(np.median(apreds)))
    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_zap70_data():
    """Load ZAP70 molecules, fingerprints, docking, and generate all pairs."""
    mol_data, _ = load_zap70_molecules()
    dock_df = pd.read_csv(DOCK_CHEMBL_CSV)
    dock_df = dock_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    dock_cols = ["molecule_chembl_id", "vina_score", "vina_inter", "vina_intra"]
    dock_subset = dock_df[dock_df["success"] == True][dock_cols].copy()
    mol_data = mol_data.merge(dock_subset, on="molecule_chembl_id", how="left")
    mol_data["has_dock"] = ~mol_data["vina_score"].isna()

    X_fp = compute_fingerprints(mol_data["smiles"].tolist(), fp_type="morgan", n_bits=2048)

    # Vina features per molecule (fill NaN with mean)
    vina_cols = ["vina_score", "vina_inter", "vina_intra"]
    vina_per_mol = mol_data[vina_cols].values.astype(np.float32)
    for ci in range(vina_per_mol.shape[1]):
        mask = np.isnan(vina_per_mol[:, ci])
        if mask.any() and not mask.all():
            vina_per_mol[mask, ci] = np.nanmean(vina_per_mol[:, ci])

    # All pairs
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
    pairs_df = pd.DataFrame(pairs)

    # Pair-level Vina diffs
    pair_dock_feats = (vina_per_mol[pairs_df["idx_b"].values] -
                       vina_per_mol[pairs_df["idx_a"].values])

    print(f"  ZAP70: {len(mol_data)} mols, {mol_data['has_dock'].sum()} docked, {len(pairs_df)} pairs")
    print(f"  FP: {X_fp.shape}, Dock feats: {pair_dock_feats.shape}")

    return mol_data, X_fp, vina_per_mol, pairs_df, pair_dock_feats


def load_kinase_pretrain_data_with_docking():
    """Load kinase MMP pairs WITH real docking features from kinase panel.

    Returns pairs where BOTH molecules have real Vina docking scores.
    """
    # Load kinase MMP pairs
    pretrain_path = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
    full_df = pd.read_csv(pretrain_path, usecols=[
        "mol_a", "mol_b", "delta", "target_chembl_id", "is_within_assay"
    ])
    kinase_df = full_df[
        full_df["target_chembl_id"].isin(PRETRAIN_KINASES.values()) &
        (full_df["is_within_assay"] == True)
    ].copy()
    del full_df
    gc.collect()

    print(f"  Kinase pairs (all): {len(kinase_df)}")

    # Load unified docking caches
    vina_cache = dict(np.load(
        PROJECT_ROOT / "data" / "embedding_cache" / "docking_vina_scores.npz",
        allow_pickle=True
    ))
    print(f"  Docking cache: {len(vina_cache)} molecules")

    # Filter to pairs where BOTH molecules have docking data
    has_dock_a = kinase_df["mol_a"].isin(set(vina_cache.keys()))
    has_dock_b = kinase_df["mol_b"].isin(set(vina_cache.keys()))
    docked_df = kinase_df[has_dock_a & has_dock_b].copy()
    print(f"  Kinase pairs (both docked): {len(docked_df)} ({100*len(docked_df)/len(kinase_df):.1f}%)")

    # Compute fingerprints for all unique molecules
    all_smiles = list(set(docked_df["mol_a"].tolist() + docked_df["mol_b"].tolist()))
    print(f"  Unique molecules: {len(all_smiles)}")
    all_fps = compute_fingerprints(all_smiles, fp_type="morgan", n_bits=2048)
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}

    # Build tensors
    fps_a = np.array([all_fps[smi_to_idx[s]] for s in docked_df["mol_a"]])
    fps_b = np.array([all_fps[smi_to_idx[s]] for s in docked_df["mol_b"]])
    delta = docked_df["delta"].values.astype(np.float32)

    # Build Vina diff features (vina_b - vina_a), using first 3 features (score, inter, intra)
    vina_a = np.array([vina_cache[s][:3] for s in docked_df["mol_a"]], dtype=np.float32)
    vina_b = np.array([vina_cache[s][:3] for s in docked_df["mol_b"]], dtype=np.float32)
    vina_diff = vina_b - vina_a

    del vina_cache, all_fps
    gc.collect()

    print(f"  Pretrain data: fps={fps_a.shape}, vina_diff={vina_diff.shape}, delta={delta.shape}")
    return fps_a, fps_b, delta, vina_diff, docked_df


# ═══════════════════════════════════════════════════════════════════════════
# Pretraining Functions
# ═══════════════════════════════════════════════════════════════════════════

def pretrain_docking_film(fps_a, fps_b, delta, vina_diff, device=DEVICE):
    """Pretrain DockingFiLMDeltaMLP on kinase data WITH real Vina features."""
    from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP

    input_dim = fps_a.shape[1]
    model = DockingFiLMDeltaMLP(input_dim=input_dim, extra_dim=VINA_DIM, dropout=0.2)
    model = model.to(device)

    # Train/val split
    n = len(delta)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = max(int(n * 0.15), 200)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]

    train_ds = TensorDataset(
        torch.FloatTensor(fps_a[tr_idx]),
        torch.FloatTensor(fps_b[tr_idx]),
        torch.FloatTensor(vina_diff[tr_idx]),
        torch.FloatTensor(delta[tr_idx]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(fps_a[val_idx]),
        torch.FloatTensor(fps_b[val_idx]),
        torch.FloatTensor(vina_diff[val_idx]),
        torch.FloatTensor(delta[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(50):
        model.train()
        for ba, bb, bv, bd in train_loader:
            ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
            optimizer.zero_grad()
            pred = model(ba, bb, bv)
            loss = criterion(pred, bd)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bv, bd in val_loader:
                ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
                pred = model(ba, bb, bv)
                val_losses.append(criterion(pred, bd).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"    Pretrain early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Val metrics
    model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for ba, bb, bv, bd in val_loader:
            ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
            pred = model(ba, bb, bv)
            val_preds.extend(pred.cpu().numpy().tolist())
            val_trues.extend(bd.cpu().numpy().tolist())
    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)
    val_mae = float(np.mean(np.abs(val_preds - val_trues)))
    val_spr = float(spearmanr(val_preds, val_trues)[0])
    print(f"    Pretrain val: MAE={val_mae:.4f}, Spr={val_spr:.3f}, epochs={min(epoch+1, 50)}")

    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return pretrained_state, {"val_mae": val_mae, "val_spr": val_spr, "epochs": min(epoch+1, 50)}


def pretrain_film_backbone(fps_a, fps_b, delta, device=DEVICE):
    """Pretrain plain FiLMDeltaMLP on kinase data (no docking features)."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

    n = len(delta)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = max(int(n * 0.15), 200)

    model = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=128,
        max_epochs=50, patience=10, device=device,
    )
    model.fit(
        fps_a[perm[val_n:]], fps_b[perm[val_n:]], delta[perm[val_n:]],
        fps_a[perm[:val_n]], fps_b[perm[:val_n]], delta[perm[:val_n]],
        verbose=True,
    )
    pretrained_state = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}
    del model
    gc.collect()
    return pretrained_state


def pretrain_feature_gated(fps_a, fps_b, delta, vina_diff, device=DEVICE):
    """Pretrain FeatureGatedFiLM on kinase data WITH real Vina features."""
    from src.models.predictors.advanced_docking_film import FeatureGatedFiLM

    input_dim = fps_a.shape[1]
    model = FeatureGatedFiLM(input_dim=input_dim, dock_dim=VINA_DIM, dropout=0.2)
    model = model.to(device)

    n = len(delta)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = max(int(n * 0.15), 200)

    train_ds = TensorDataset(
        torch.FloatTensor(fps_a[perm[val_n:]]),
        torch.FloatTensor(fps_b[perm[val_n:]]),
        torch.FloatTensor(vina_diff[perm[val_n:]]),
        torch.FloatTensor(delta[perm[val_n:]]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(fps_a[perm[:val_n]]),
        torch.FloatTensor(fps_b[perm[:val_n]]),
        torch.FloatTensor(vina_diff[perm[:val_n]]),
        torch.FloatTensor(delta[perm[:val_n]]),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(50):
        model.train()
        for ba, bb, bv, bd in train_loader:
            ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
            optimizer.zero_grad()
            pred = model(ba, bb, bv)
            loss = criterion(pred, bd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bv, bd in val_loader:
                ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
                pred = model(ba, bb, bv)
                val_losses.append(criterion(pred, bd).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"    FeatureGated pretrain early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return pretrained_state


def pretrain_residual_correction(fps_a, fps_b, delta, vina_diff, device=DEVICE):
    """Pretrain ResidualCorrectionFiLM on kinase data WITH real Vina features."""
    from src.models.predictors.advanced_docking_film import ResidualCorrectionFiLM

    input_dim = fps_a.shape[1]
    model = ResidualCorrectionFiLM(input_dim=input_dim, dock_dim=VINA_DIM, dropout=0.2)
    model = model.to(device)

    n = len(delta)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = max(int(n * 0.15), 200)

    train_ds = TensorDataset(
        torch.FloatTensor(fps_a[perm[val_n:]]),
        torch.FloatTensor(fps_b[perm[val_n:]]),
        torch.FloatTensor(vina_diff[perm[val_n:]]),
        torch.FloatTensor(delta[perm[val_n:]]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(fps_a[perm[:val_n]]),
        torch.FloatTensor(fps_b[perm[:val_n]]),
        torch.FloatTensor(vina_diff[perm[:val_n]]),
        torch.FloatTensor(delta[perm[:val_n]]),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(50):
        model.train()
        for ba, bb, bv, bd in train_loader:
            ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
            optimizer.zero_grad()
            pred = model(ba, bb, bv)
            loss = criterion(pred, bd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bv, bd in val_loader:
                ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
                pred = model(ba, bb, bv)
                val_losses.append(criterion(pred, bd).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"    ResidCorr pretrain early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return pretrained_state


# ═══════════════════════════════════════════════════════════════════════════
# Finetuning (raw training loops to avoid fit() rebuild bug)
# ═══════════════════════════════════════════════════════════════════════════

def finetune_docking_film(train_idx, X_fp, pairs_df, pair_dock_feats,
                          pretrained_state, seed, model_class=None,
                          extra_dim=VINA_DIM, device=DEVICE,
                          lr=5e-4, batch_size=64, max_epochs=100, patience=15):
    """Finetune a pretrained DockingFiLM-like model on ZAP70 data.

    Works with DockingFiLMDeltaMLP, FeatureGatedFiLM, ResidualCorrectionFiLM.
    """
    if model_class is None:
        from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP
        model_class = DockingFiLMDeltaMLP

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_set = set(train_idx)
    mask = pairs_df["idx_a"].isin(train_set) & pairs_df["idx_b"].isin(train_set)
    tp = pairs_df[mask]
    if len(tp) == 0:
        return None

    rng = np.random.RandomState(seed)
    n = len(tp)
    val_size = max(int(n * 0.15), 100)
    perm = rng.permutation(n)
    vi, ti = perm[:val_size], perm[val_size:]
    pair_indices_tr = tp.iloc[ti].index.values
    pair_indices_val = tp.iloc[vi].index.values

    input_dim = X_fp.shape[1]

    # Build model and load pretrained weights
    if model_class.__name__ == "DockingFiLMDeltaMLP":
        model = model_class(input_dim=input_dim, extra_dim=extra_dim, dropout=0.2)
    elif model_class.__name__ == "FeatureGatedFiLM":
        model = model_class(input_dim=input_dim, dock_dim=extra_dim, dropout=0.2)
    elif model_class.__name__ == "ResidualCorrectionFiLM":
        model = model_class(input_dim=input_dim, dock_dim=extra_dim, dropout=0.2)
    else:
        model = model_class(input_dim=input_dim, extra_dim=extra_dim, dropout=0.2)

    # Load pretrained weights (matching keys)
    model_sd = model.state_dict()
    loaded = 0
    for k in model_sd:
        if k in pretrained_state and model_sd[k].shape == pretrained_state[k].shape:
            model_sd[k] = pretrained_state[k].clone()
            loaded += 1
    model.load_state_dict(model_sd)
    model = model.to(device)

    # Prepare data tensors
    emb_a_tr = torch.FloatTensor(X_fp[tp.iloc[ti]["idx_a"].values])
    emb_b_tr = torch.FloatTensor(X_fp[tp.iloc[ti]["idx_b"].values])
    dock_tr = torch.FloatTensor(pair_dock_feats[pair_indices_tr])
    delta_tr = torch.FloatTensor(tp.iloc[ti]["delta"].values.astype(np.float32))

    emb_a_val = torch.FloatTensor(X_fp[tp.iloc[vi]["idx_a"].values])
    emb_b_val = torch.FloatTensor(X_fp[tp.iloc[vi]["idx_b"].values])
    dock_val = torch.FloatTensor(pair_dock_feats[pair_indices_val])
    delta_val = torch.FloatTensor(tp.iloc[vi]["delta"].values.astype(np.float32))

    train_loader = DataLoader(TensorDataset(emb_a_tr, emb_b_tr, dock_tr, delta_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(emb_a_val, emb_b_val, dock_val, delta_val),
                            batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    pat_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        for ba, bb, bv, bd in train_loader:
            ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
            optimizer.zero_grad()
            pred = model(ba, bb, bv)
            loss = criterion(pred, bd)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bv, bd in val_loader:
                ba, bb, bv, bd = ba.to(device), bb.to(device), bv.to(device), bd.to(device)
                pred = model(ba, bb, bv)
                val_losses.append(criterion(pred, bd).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model


def predict_with_model(model, emb_a, emb_b, dock_feats, device=DEVICE):
    """Predict deltas with a docking model."""
    model.eval()
    with torch.no_grad():
        a = torch.FloatTensor(emb_a).to(device)
        b = torch.FloatTensor(emb_b).to(device)
        d = torch.FloatTensor(dock_feats).to(device)
        return model(a, b, d).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# CV Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_cv(method_name, mol_data, X_fp, pairs_df, pair_dock_feats,
           pretrained_state, model_class, seeds, y_all, extra_dim=VINA_DIM):
    """Run multi-seed CV for a pretrained docking model."""
    all_delta, all_abs = [], []

    for seed in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            model = finetune_docking_film(
                train_idx, X_fp, pairs_df, pair_dock_feats,
                pretrained_state, seed, model_class=model_class,
                extra_dim=extra_dim,
            )
            if model is None:
                continue

            test_set = set(test_idx)
            test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            tp = pairs_df[test_mask]
            if len(tp) == 0:
                continue

            delta_pred = predict_with_model(
                model, X_fp[tp["idx_a"].values], X_fp[tp["idx_b"].values],
                pair_dock_feats[tp.index.values])
            delta_true = tp["delta"].values.astype(np.float32)
            all_delta.append(compute_delta_metrics(delta_true, delta_pred))

            try:
                def anchor_fn(a, b):
                    dummy = np.zeros((len(a), extra_dim), dtype=np.float32)
                    return predict_with_model(model, a, b, dummy)
                y_pred = reconstruct_absolute(test_idx, train_idx, X_fp, y_all, anchor_fn, 50)
                all_abs.append({"mae": float(np.mean(np.abs(y_pred - y_all[test_idx]))),
                    "spearman": float(spearmanr(y_pred, y_all[test_idx])[0])})
            except:
                pass

            del model
            gc.collect()

        avg_mae = np.mean([f["mae"] for f in all_delta[-N_FOLDS:]]) if len(all_delta) >= N_FOLDS else 0
        print(f"    {method_name} seed {seed}: avg MAE = {avg_mae:.4f}")

    result = {
        "delta_mae_mean": float(np.mean([f["mae"] for f in all_delta])),
        "delta_mae_std": float(np.std([f["mae"] for f in all_delta])),
        "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_delta])),
        "delta_spearman_std": float(np.std([f["spearman"] for f in all_delta])),
        "delta_pearson_mean": float(np.mean([f["pearson"] for f in all_delta])),
        "delta_r2_mean": float(np.mean([f["r2"] for f in all_delta])),
        "abs_mae_mean": float(np.mean([f["mae"] for f in all_abs])) if all_abs else 0,
        "abs_spearman_mean": float(np.mean([f["spearman"] for f in all_abs])) if all_abs else 0,
        "n_seeds": len(seeds), "n_folds": N_FOLDS,
    }
    print(f"  {method_name}: MAE={result['delta_mae_mean']:.4f}±{result['delta_mae_std']:.3f} "
          f"Spr={result['delta_spearman_mean']:.3f}")
    return result


def run_ensemble_cv(method_name, mol_data, X_fp, pairs_df, pair_dock_feats,
                    pretrained_states, model_classes, ensemble_seeds, y_all,
                    extra_dim=VINA_DIM):
    """Run ensemble CV: for each fold, train multiple models and average predictions."""
    all_delta, all_abs = [], []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        models = []
        for ps, mc, s in zip(pretrained_states, model_classes, ensemble_seeds):
            m = finetune_docking_film(
                train_idx, X_fp, pairs_df, pair_dock_feats,
                ps, s, model_class=mc, extra_dim=extra_dim,
            )
            if m is not None:
                models.append(m)

        if not models:
            continue

        test_set = set(test_idx)
        test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        tp = pairs_df[test_mask]
        if len(tp) == 0:
            continue

        # Average predictions
        preds = []
        for m in models:
            p = predict_with_model(
                m, X_fp[tp["idx_a"].values], X_fp[tp["idx_b"].values],
                pair_dock_feats[tp.index.values])
            preds.append(p)
        delta_pred = np.mean(preds, axis=0)
        delta_true = tp["delta"].values.astype(np.float32)
        all_delta.append(compute_delta_metrics(delta_true, delta_pred))

        try:
            def anchor_fn(a, b):
                dummy = np.zeros((len(a), extra_dim), dtype=np.float32)
                ps_list = [predict_with_model(m, a, b, dummy) for m in models]
                return np.mean(ps_list, axis=0)
            y_pred = reconstruct_absolute(test_idx, train_idx, X_fp, y_all, anchor_fn, 50)
            all_abs.append({"mae": float(np.mean(np.abs(y_pred - y_all[test_idx]))),
                "spearman": float(spearmanr(y_pred, y_all[test_idx])[0])})
        except:
            pass

        for m in models:
            del m
        gc.collect()

        print(f"    fold {fold_i}: MAE = {all_delta[-1]['mae']:.4f}")

    result = {
        "delta_mae_mean": float(np.mean([f["mae"] for f in all_delta])),
        "delta_mae_std": float(np.std([f["mae"] for f in all_delta])),
        "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_delta])),
        "abs_mae_mean": float(np.mean([f["mae"] for f in all_abs])) if all_abs else 0,
        "abs_spearman_mean": float(np.mean([f["spearman"] for f in all_abs])) if all_abs else 0,
        "n_models_per_fold": len(pretrained_states), "n_folds": N_FOLDS,
    }
    print(f"  {method_name}: MAE={result['delta_mae_mean']:.4f}±{result['delta_mae_std']:.3f} "
          f"Spr={result['delta_spearman_mean']:.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*70}")
    print(f"  DOCKING INTEGRATION — ITERATION 3")
    print(f"  Pretrain DockingFiLM with REAL docking features")
    print(f"  Device: {DEVICE}, Seeds: {SEEDS}")
    print(f"  Started: {datetime.now()}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    results = load_results()

    # ── Load ZAP70 data ──
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    mol_data, X_fp, vina_per_mol, pairs_df, pair_dock_feats = load_zap70_data()
    y_all = mol_data["pIC50"].values

    # ── Load kinase pretraining data WITH docking ──
    print("\n  Loading kinase pretraining data with docking features...")
    pt_fps_a, pt_fps_b, pt_delta, pt_vina_diff, pt_df = load_kinase_pretrain_data_with_docking()

    # ── Phase A: Pretraining ──
    print(f"\n{'='*70}")
    print(f"  PHASE A: PRETRAINING")
    print(f"{'='*70}")

    # A1: Pretrain DockingFiLM with REAL Vina features
    print("\n  A1: Pretraining DockingFiLMDeltaMLP with real Vina features...")
    dock_pretrained_state, dock_pt_metrics = pretrain_docking_film(
        pt_fps_a, pt_fps_b, pt_delta, pt_vina_diff)
    print(f"    DockFiLM pretrained: val MAE={dock_pt_metrics['val_mae']:.4f}")

    # A2: Pretrain plain FiLMDelta (for comparison & backbone transfer baseline)
    print("\n  A2: Pretraining FiLMDelta backbone (no docking)...")
    backbone_pretrained_state = pretrain_film_backbone(pt_fps_a, pt_fps_b, pt_delta)

    # A3: Pretrain FeatureGatedFiLM with real Vina features
    print("\n  A3: Pretraining FeatureGatedFiLM with real Vina features...")
    fg_pretrained_state = pretrain_feature_gated(
        pt_fps_a, pt_fps_b, pt_delta, pt_vina_diff)

    # A4: Pretrain ResidualCorrectionFiLM with real Vina features
    print("\n  A4: Pretraining ResidualCorrectionFiLM with real Vina features...")
    rc_pretrained_state = pretrain_residual_correction(
        pt_fps_a, pt_fps_b, pt_delta, pt_vina_diff)

    # Free pretraining data
    n_pretrain_pairs = len(pt_df)
    del pt_fps_a, pt_fps_b, pt_delta, pt_vina_diff, pt_df
    gc.collect()

    results["phase_a"] = {
        "dock_pretrain_metrics": dock_pt_metrics,
        "n_pretrain_pairs_with_docking": n_pretrain_pairs,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    # ── Phase B: Single model evaluations (3 seeds) ──
    print(f"\n{'='*70}")
    print(f"  PHASE B: FINETUNING & EVALUATION (3 seeds × 5 folds)")
    print(f"{'='*70}")

    from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP
    from src.models.predictors.advanced_docking_film import FeatureGatedFiLM, ResidualCorrectionFiLM

    methods = {}

    # B1: DockFiLM pretrained with REAL docking (the main experiment)
    print(f"\n  B1: DockFiLM_real_pretrained (DockingFiLM pretrained WITH real Vina)...")
    methods["DockFiLM_real_pretrained"] = run_cv(
        "DockFiLM_real_pretrained", mol_data, X_fp, pairs_df, pair_dock_feats,
        dock_pretrained_state, DockingFiLMDeltaMLP, SEEDS, y_all)
    save_results({**results, "phase_b": {"methods": methods, "completed": False}})

    # B2: DockFiLM with backbone-only pretraining (previous approach, as reference)
    print(f"\n  B2: DockFiLM_backbone_pretrained (FiLMDelta pretrained, transfer to DockFiLM)...")
    # Transfer backbone weights to DockingFiLM (zero-init docking dims)
    from experiments.run_docking_iter2_phaseD_fixed import transfer_film_to_docking as tf_weights
    transferred_state = {}
    # Build a temporary DockingFiLM and transfer backbone weights
    tmp_model = DockingFiLMDeltaMLP(input_dim=X_fp.shape[1], extra_dim=VINA_DIM, dropout=0.2)
    tmp_sd = tmp_model.state_dict()
    for k in tmp_sd:
        if k in backbone_pretrained_state:
            if tmp_sd[k].shape == backbone_pretrained_state[k].shape:
                tmp_sd[k] = backbone_pretrained_state[k].clone()
            elif "delta_encoder" in k and "weight" in k:
                morgan_dim = backbone_pretrained_state[k].shape[1]
                tmp_sd[k][:, :morgan_dim] = backbone_pretrained_state[k].clone()
                tmp_sd[k][:, morgan_dim:] = 0.0
            elif tmp_sd[k].shape == backbone_pretrained_state[k].shape:
                tmp_sd[k] = backbone_pretrained_state[k].clone()
    transferred_state = {k: v.clone() for k, v in tmp_sd.items()}
    del tmp_model, tmp_sd

    methods["DockFiLM_backbone_pretrained"] = run_cv(
        "DockFiLM_backbone_pretrained", mol_data, X_fp, pairs_df, pair_dock_feats,
        transferred_state, DockingFiLMDeltaMLP, SEEDS, y_all)
    save_results({**results, "phase_b": {"methods": methods, "completed": False}})

    # B3: FeatureGated pretrained with REAL docking
    print(f"\n  B3: FeatureGated_real_pretrained...")
    methods["FeatureGated_real_pretrained"] = run_cv(
        "FeatureGated_real_pretrained", mol_data, X_fp, pairs_df, pair_dock_feats,
        fg_pretrained_state, FeatureGatedFiLM, SEEDS, y_all)
    save_results({**results, "phase_b": {"methods": methods, "completed": False}})

    # B4: ResidualCorrection pretrained with REAL docking
    print(f"\n  B4: ResidCorr_real_pretrained...")
    methods["ResidCorr_real_pretrained"] = run_cv(
        "ResidCorr_real_pretrained", mol_data, X_fp, pairs_df, pair_dock_feats,
        rc_pretrained_state, ResidualCorrectionFiLM, SEEDS, y_all)

    results["phase_b"] = {
        "methods": methods,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    # Print Phase B summary
    print(f"\n{'='*70}")
    print(f"  PHASE B RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':<40} {'ΔMAE':>10} {'ΔSpr':>8} {'AbsMAE':>10} {'AbsSpr':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    for k, v in sorted(methods.items(), key=lambda x: x[1]["delta_mae_mean"]):
        print(f"  {k:<40} {v['delta_mae_mean']:.4f}±{v.get('delta_mae_std',0):.3f} "
              f"   {v['delta_spearman_mean']:.3f}    {v.get('abs_mae_mean',0):.4f} "
              f"   {v.get('abs_spearman_mean',0):.3f}")
    print(f"{'='*70}")

    # ── Phase C: Ensembles ──
    print(f"\n{'='*70}")
    print(f"  PHASE C: ENSEMBLES")
    print(f"{'='*70}")

    ensemble_methods = {}

    # C1: Multi-seed ensemble of DockFiLM real pretrained (5 seeds)
    print(f"\n  C1: DockFiLM_real_pretrained_5seed_ensemble...")
    ensemble_methods["DockFiLM_real_5seed"] = run_ensemble_cv(
        "DockFiLM_real_5seed",
        mol_data, X_fp, pairs_df, pair_dock_feats,
        [dock_pretrained_state] * 5,
        [DockingFiLMDeltaMLP] * 5,
        [42, 123, 456, 789, 1024],
        y_all,
    )
    save_results({**results, "phase_c": {"methods": ensemble_methods, "completed": False}})

    # C2: Mixed architecture ensemble (DockFiLM + FeatureGated + ResidCorr, all real-pretrained)
    print(f"\n  C2: Mixed_arch_real_pretrained_ensemble...")
    ensemble_methods["Mixed_arch_real_pretrained"] = run_ensemble_cv(
        "Mixed_arch_real_pretrained",
        mol_data, X_fp, pairs_df, pair_dock_feats,
        [dock_pretrained_state, fg_pretrained_state, rc_pretrained_state],
        [DockingFiLMDeltaMLP, FeatureGatedFiLM, ResidualCorrectionFiLM],
        [42, 123, 456],
        y_all,
    )
    save_results({**results, "phase_c": {"methods": ensemble_methods, "completed": False}})

    # C3: Real pretrained + backbone pretrained ensemble (diversity)
    print(f"\n  C3: Real_plus_backbone_pretrained_ensemble...")
    ensemble_methods["Real_plus_backbone_ensemble"] = run_ensemble_cv(
        "Real_plus_backbone_ensemble",
        mol_data, X_fp, pairs_df, pair_dock_feats,
        [dock_pretrained_state, dock_pretrained_state, transferred_state, transferred_state],
        [DockingFiLMDeltaMLP] * 4,
        [42, 123, 456, 789],
        y_all,
    )

    results["phase_c"] = {
        "methods": ensemble_methods,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)

    # Print Phase C summary
    print(f"\n{'='*70}")
    print(f"  PHASE C RESULTS")
    print(f"{'='*70}")
    for k, v in sorted(ensemble_methods.items(), key=lambda x: x[1]["delta_mae_mean"]):
        print(f"  {k:<40} MAE={v['delta_mae_mean']:.4f}±{v.get('delta_mae_std',0):.3f} "
              f"Spr={v['delta_spearman_mean']:.3f}")
    print(f"{'='*70}")

    # ── Final Summary ──
    print(f"\n{'#'*70}")
    print(f"  ITERATION 3 — FINAL RESULTS")
    print(f"{'#'*70}")
    all_methods = {**methods, **ensemble_methods}
    # Add reference baselines from iteration 2
    all_methods["[ref] FiLMDelta_baseline"] = {"delta_mae_mean": 0.738, "delta_mae_std": 0.042,
        "delta_spearman_mean": 0.758, "abs_mae_mean": 0.496, "abs_spearman_mean": 0.783}
    all_methods["[ref] DockFiLM_vina_pretrained"] = {"delta_mae_mean": 0.726, "delta_mae_std": 0.049,
        "delta_spearman_mean": 0.766, "abs_mae_mean": 0.489, "abs_spearman_mean": 0.792}
    all_methods["[ref] MultiSeed_ensemble_iter2"] = {"delta_mae_mean": 0.727, "delta_mae_std": 0.042,
        "delta_spearman_mean": 0.764, "abs_mae_mean": 0.498, "abs_spearman_mean": 0.788}

    print(f"\n  {'Method':<45} {'ΔMAE':>10} {'ΔSpr':>8} {'AbsMAE':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*10}")
    for k, v in sorted(all_methods.items(), key=lambda x: x[1]["delta_mae_mean"]):
        print(f"  {k:<45} {v['delta_mae_mean']:.4f}±{v.get('delta_mae_std',0):.3f} "
              f"   {v['delta_spearman_mean']:.3f}    {v.get('abs_mae_mean',0):.4f}")

    print(f"\n  Total time: {time.time() - t0:.0f}s ({(time.time()-t0)/3600:.1f}h)")
    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
