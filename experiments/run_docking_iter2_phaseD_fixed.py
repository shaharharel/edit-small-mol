#!/usr/bin/env python3
"""
Fixed Phase D: Proper weight transfer for pretrained + docking models.

The iteration 2 Phase D had a bug where `fit()` rebuilds the model,
losing transferred weights. This script uses the correct approach
from `run_docking_pretrain_combo.py`: directly build model, transfer
weights at tensor level, then train.

Methods:
  1. FiLMDelta_pretrained — pretrain on kinase, finetune on ZAP70
  2. FiLMDelta_vina_pretrained — pretrain, transfer to DockingFiLM, finetune with Vina
  3. MultiSeed_vina_pretrained — ensemble of 5 pretrained+docking models
  4. Best_ensemble — ensemble of pretrained + pretrained_vina + baseline

Usage:
    conda run -n quris python -u experiments/run_docking_iter2_phaseD_fixed.py
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
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    N_JOBS, N_FOLDS, CV_SEED,
)
from experiments.run_paper_evaluation import RESULTS_DIR

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "docking_iteration2_results.json"
DEVICE = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else "cpu"

DOCK_CHEMBL_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCK_CHEMBL_CSV = DOCK_CHEMBL_DIR / "docking_results.csv"

N_SEEDS = 3
SEEDS = [42, 123, 456]
VINA_DIM = 3


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


def transfer_film_to_docking(pretrained_state_dict, input_dim, extra_dim=3):
    """Transfer weights from FiLMDeltaMLP state_dict to DockingFiLMDeltaMLP.

    Properly handles the wider delta_encoder by zero-initializing docking dims.
    """
    from src.models.predictors.docking_film_predictor import DockingFiLMDeltaMLP

    docking_model = DockingFiLMDeltaMLP(
        input_dim=input_dim, extra_dim=extra_dim, dropout=0.2,
    )

    docking_sd = docking_model.state_dict()
    n_transferred = 0
    n_random = 0

    for key in docking_sd:
        if key in pretrained_state_dict:
            if docking_sd[key].shape == pretrained_state_dict[key].shape:
                docking_sd[key] = pretrained_state_dict[key].clone()
                n_transferred += 1
            elif "delta_encoder" in key and "weight" in key:
                morgan_dim = pretrained_state_dict[key].shape[1]
                docking_sd[key][:, :morgan_dim] = pretrained_state_dict[key].clone()
                docking_sd[key][:, morgan_dim:] = 0.0
                n_transferred += 1
            elif "delta_encoder" in key and "bias" in key:
                if docking_sd[key].shape == pretrained_state_dict[key].shape:
                    docking_sd[key] = pretrained_state_dict[key].clone()
                    n_transferred += 1
                else:
                    n_random += 1
            else:
                n_random += 1
        else:
            n_random += 1

    docking_model.load_state_dict(docking_sd)
    print(f"    Weight transfer: {n_transferred} transferred, {n_random} random-init")
    return docking_model


def train_pretrained_film(
    train_idx, X_fp, pairs_df, pretrained_state, seed,
    device=DEVICE,
):
    """Train FiLMDelta with pretrained weights (proper transfer)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP

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

    # Create predictor WITHOUT calling fit (to avoid model rebuild)
    predictor = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=5e-4, batch_size=64,
        max_epochs=100, patience=15, device=device,
    )

    # Manually build model with pretrained weights
    input_dim = X_fp.shape[1]
    predictor.input_dim = input_dim
    predictor.model = FiLMDeltaMLP(
        input_dim=input_dim, dropout=0.2,
    )
    predictor.model.load_state_dict(pretrained_state)
    predictor.model = predictor.model.to(device)

    # Now finetune (fit will NOT rebuild since model already exists...
    # Actually it WILL rebuild. We need to work around this.)
    # Use the raw training loop instead.

    emb_a_tr = torch.from_numpy(X_fp[tp.iloc[ti]["idx_a"].values]).float()
    emb_b_tr = torch.from_numpy(X_fp[tp.iloc[ti]["idx_b"].values]).float()
    delta_tr = torch.from_numpy(tp.iloc[ti]["delta"].values.astype(np.float32))
    emb_a_val = torch.from_numpy(X_fp[tp.iloc[vi]["idx_a"].values]).float()
    emb_b_val = torch.from_numpy(X_fp[tp.iloc[vi]["idx_b"].values]).float()
    delta_val = torch.from_numpy(tp.iloc[vi]["delta"].values.astype(np.float32))

    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(TensorDataset(emb_a_tr, emb_b_tr, delta_tr),
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(emb_a_val, emb_b_val, delta_val),
                            batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        predictor.model.train()
        for ba, bb, bd in train_loader:
            ba, bb, bd = ba.to(device), bb.to(device), bd.to(device)
            optimizer.zero_grad()
            pred = predictor.model(ba, bb)
            loss = criterion(pred, bd)
            loss.backward()
            optimizer.step()

        predictor.model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bd in val_loader:
                ba, bb, bd = ba.to(device), bb.to(device), bd.to(device)
                pred = predictor.model(ba, bb)
                val_losses.append(criterion(pred, bd).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in predictor.model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 15:
            break

    if best_state:
        predictor.model.load_state_dict(best_state)
        predictor.model = predictor.model.to(device)

    return predictor


def train_pretrained_vina_film(
    train_idx, X_fp, pairs_df, pair_dock_feats, pretrained_state, seed,
    device=DEVICE,
):
    """Train DockingFiLM with pretrained weights via proper transfer."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models.predictors.docking_film_predictor import DockingFiLMPredictor

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

    # Build docking model with transferred weights
    docking_model = transfer_film_to_docking(pretrained_state, input_dim, VINA_DIM)

    # Create predictor wrapper
    predictor = DockingFiLMPredictor(
        arch="docking_film", extra_dim=VINA_DIM,
        dropout=0.2, learning_rate=5e-4, batch_size=64,
        max_epochs=100, patience=15, device=device,
    )
    # Inject the transferred model
    predictor.input_dim = input_dim
    predictor.model = docking_model.to(device)

    # Finetune using raw training loop (bypass fit's model rebuild)
    from torch.utils.data import DataLoader, TensorDataset

    emb_a_tr = torch.from_numpy(X_fp[tp.iloc[ti]["idx_a"].values]).float()
    emb_b_tr = torch.from_numpy(X_fp[tp.iloc[ti]["idx_b"].values]).float()
    dock_tr = torch.from_numpy(pair_dock_feats[pair_indices_tr]).float()
    delta_tr = torch.from_numpy(tp.iloc[ti]["delta"].values.astype(np.float32))

    emb_a_val = torch.from_numpy(X_fp[tp.iloc[vi]["idx_a"].values]).float()
    emb_b_val = torch.from_numpy(X_fp[tp.iloc[vi]["idx_b"].values]).float()
    dock_val = torch.from_numpy(pair_dock_feats[pair_indices_val]).float()
    delta_val = torch.from_numpy(tp.iloc[vi]["delta"].values.astype(np.float32))

    train_loader = DataLoader(TensorDataset(emb_a_tr, emb_b_tr, dock_tr, delta_tr),
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(emb_a_val, emb_b_val, dock_val, delta_val),
                            batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        predictor.model.train()
        for ba, bb, bd_dock, bd_delta in train_loader:
            ba = ba.to(device); bb = bb.to(device)
            bd_dock = bd_dock.to(device); bd_delta = bd_delta.to(device)
            optimizer.zero_grad()
            pred = predictor.model(ba, bb, bd_dock)
            loss = criterion(pred, bd_delta)
            loss.backward()
            optimizer.step()

        predictor.model.eval()
        val_losses = []
        with torch.no_grad():
            for ba, bb, bd_dock, bd_delta in val_loader:
                ba = ba.to(device); bb = bb.to(device)
                bd_dock = bd_dock.to(device); bd_delta = bd_delta.to(device)
                pred = predictor.model(ba, bb, bd_dock)
                val_losses.append(criterion(pred, bd_delta).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in predictor.model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 15:
            break

    if best_state:
        predictor.model.load_state_dict(best_state)
        predictor.model = predictor.model.to(device)

    return predictor


def main():
    print(f"\n{'#'*70}")
    print(f"  PHASE D FIXED: Proper Pretraining + Docking")
    print(f"  Device: {DEVICE}, Seeds: {SEEDS}")
    print(f"  Started: {datetime.now()}")
    print(f"{'#'*70}")

    t0 = time.time()

    # Load data
    mol_data, _ = load_zap70_molecules()
    dock_df = pd.read_csv(DOCK_CHEMBL_CSV)
    dock_df = dock_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    dock_cols = ["molecule_chembl_id", "vina_score", "vina_inter", "vina_intra"]
    dock_subset = dock_df[dock_df["success"] == True][dock_cols].copy()
    mol_data = mol_data.merge(dock_subset, on="molecule_chembl_id", how="left")
    mol_data["has_dock"] = ~mol_data["vina_score"].isna()
    print(f"  Molecules: {len(mol_data)}, Docked: {mol_data['has_dock'].sum()}")

    X_fp = compute_fingerprints(mol_data["smiles"].tolist(), fp_type="morgan", n_bits=2048)
    print(f"  Fingerprints: {X_fp.shape}")

    # Vina features
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
    print(f"  Pairs: {len(pairs_df)}")

    # Pair-level Vina diffs
    pair_dock_feats = (vina_per_mol[pairs_df["idx_b"].values] -
                       vina_per_mol[pairs_df["idx_a"].values])
    print(f"  Pair dock feats: {pair_dock_feats.shape}")

    # Load kinase pretraining data
    pretrain_path = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
    kinase_targets = {
        "SYK": "CHEMBL2599", "LCK": "CHEMBL258",
        "JAK2": "CHEMBL2971", "ABL1": "CHEMBL1862",
        "SRC": "CHEMBL267", "BTK": "CHEMBL5251",
    }
    full_df = pd.read_csv(pretrain_path)
    kinase_df = full_df[full_df["target_chembl_id"].isin(kinase_targets.values())].copy()
    if "is_within_assay" in kinase_df.columns:
        kinase_df = kinase_df[kinase_df["is_within_assay"] == True]
    print(f"  Kinase pretrain pairs: {len(kinase_df)}")

    pretrain_smiles = list(set(kinase_df["mol_a"].tolist() + kinase_df["mol_b"].tolist()))
    pretrain_fps = compute_fingerprints(pretrain_smiles, fp_type="morgan", n_bits=2048)
    smi_to_idx = {s: i for i, s in enumerate(pretrain_smiles)}

    pretrain_a = np.array([pretrain_fps[smi_to_idx[s]] for s in kinase_df["mol_a"]])
    pretrain_b = np.array([pretrain_fps[smi_to_idx[s]] for s in kinase_df["mol_b"]])
    pretrain_delta = kinase_df["delta"].values.astype(np.float32)

    # Pretrain FiLMDelta
    print("\n  Pretraining FiLMDelta on kinase data...")
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
    rng = np.random.RandomState(42)
    n = len(pretrain_delta)
    val_n = max(int(n * 0.15), 200)
    perm = rng.permutation(n)

    pretrain_model = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        max_epochs=50, patience=10, device=DEVICE,
    )
    pretrain_model.fit(
        pretrain_a[perm[val_n:]], pretrain_b[perm[val_n:]],
        pretrain_delta[perm[val_n:]],
        pretrain_a[perm[:val_n]], pretrain_b[perm[:val_n]],
        pretrain_delta[perm[:val_n]],
        verbose=True,
    )
    pretrained_state = {k: v.cpu().clone() for k, v in pretrain_model.model.state_dict().items()}
    del pretrain_model, pretrain_a, pretrain_b, pretrain_delta, pretrain_fps
    gc.collect()

    y_all = mol_data["pIC50"].values
    results = load_results()
    phase_results = {}

    # ── Method 1: FiLMDelta_pretrained (no docking) ──
    print(f"\n  Method 1: FiLMDelta_pretrained...")
    all_delta, all_abs = [], []
    for seed in SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            model = train_pretrained_film(train_idx, X_fp, pairs_df, pretrained_state, seed)
            if model is None: continue
            test_set = set(test_idx)
            test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            tp = pairs_df[test_mask]
            if len(tp) == 0: continue
            delta_pred = model.predict(X_fp[tp["idx_a"].values], X_fp[tp["idx_b"].values])
            delta_true = tp["delta"].values.astype(np.float32)
            all_delta.append(compute_delta_metrics(delta_true, delta_pred))
            try:
                y_pred = reconstruct_absolute(test_idx, train_idx, X_fp, y_all,
                    lambda a, b: model.predict(a, b), 50)
                all_abs.append({"mae": float(np.mean(np.abs(y_pred - y_all[test_idx]))),
                    "spearman": float(spearmanr(y_pred, y_all[test_idx])[0])})
            except: pass
        print(f"    seed {seed}: avg MAE = {np.mean([f['mae'] for f in all_delta[-N_FOLDS:]]):.4f}")

    phase_results["FiLMDelta_pretrained"] = {
        "delta_mae_mean": float(np.mean([f["mae"] for f in all_delta])),
        "delta_mae_std": float(np.std([f["mae"] for f in all_delta])),
        "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_delta])),
        "abs_mae_mean": float(np.mean([f["mae"] for f in all_abs])) if all_abs else 0,
        "abs_spearman_mean": float(np.mean([f["spearman"] for f in all_abs])) if all_abs else 0,
        "n_seeds": N_SEEDS, "n_folds": N_FOLDS,
    }
    print(f"  FiLMDelta_pretrained: MAE={phase_results['FiLMDelta_pretrained']['delta_mae_mean']:.4f}")

    # ── Method 2: FiLMDelta_vina_pretrained (proper weight transfer) ──
    print(f"\n  Method 2: FiLMDelta_vina_pretrained (FIXED)...")
    all_delta, all_abs = [], []
    for seed in SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            model = train_pretrained_vina_film(
                train_idx, X_fp, pairs_df, pair_dock_feats, pretrained_state, seed)
            if model is None: continue
            test_set = set(test_idx)
            test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            tp = pairs_df[test_mask]
            if len(tp) == 0: continue
            delta_pred = model.predict(
                X_fp[tp["idx_a"].values], X_fp[tp["idx_b"].values],
                pair_dock_feats[tp.index.values])
            delta_true = tp["delta"].values.astype(np.float32)
            all_delta.append(compute_delta_metrics(delta_true, delta_pred))
            try:
                def anchor_fn(a, b):
                    dummy = np.zeros((len(a), VINA_DIM), dtype=np.float32)
                    return model.predict(a, b, dummy)
                y_pred = reconstruct_absolute(test_idx, train_idx, X_fp, y_all, anchor_fn, 50)
                all_abs.append({"mae": float(np.mean(np.abs(y_pred - y_all[test_idx]))),
                    "spearman": float(spearmanr(y_pred, y_all[test_idx])[0])})
            except: pass
        print(f"    seed {seed}: avg MAE = {np.mean([f['mae'] for f in all_delta[-N_FOLDS:]]):.4f}")

    phase_results["FiLMDelta_vina_pretrained_fixed"] = {
        "delta_mae_mean": float(np.mean([f["mae"] for f in all_delta])),
        "delta_mae_std": float(np.std([f["mae"] for f in all_delta])),
        "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_delta])),
        "abs_mae_mean": float(np.mean([f["mae"] for f in all_abs])) if all_abs else 0,
        "abs_spearman_mean": float(np.mean([f["spearman"] for f in all_abs])) if all_abs else 0,
        "n_seeds": N_SEEDS, "n_folds": N_FOLDS,
    }
    print(f"  FiLMDelta_vina_pretrained_fixed: MAE={phase_results['FiLMDelta_vina_pretrained_fixed']['delta_mae_mean']:.4f}")

    # ── Method 3: MultiSeed ensemble of pretrained+vina (5 seeds) ──
    print(f"\n  Method 3: MultiSeed_pretrained_ensemble (5 seeds)...")
    ensemble_seeds = [42, 123, 456, 789, 1024]
    all_delta, all_abs = [], []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        models = []
        for s in ensemble_seeds:
            m = train_pretrained_vina_film(
                train_idx, X_fp, pairs_df, pair_dock_feats, pretrained_state, s)
            if m is not None:
                models.append(m)

        if not models: continue
        test_set = set(test_idx)
        test_mask = pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        tp = pairs_df[test_mask]
        if len(tp) == 0: continue

        # Average predictions from all models
        preds = []
        for m in models:
            p = m.predict(X_fp[tp["idx_a"].values], X_fp[tp["idx_b"].values],
                          pair_dock_feats[tp.index.values])
            preds.append(p)
        delta_pred = np.mean(preds, axis=0)
        delta_true = tp["delta"].values.astype(np.float32)
        all_delta.append(compute_delta_metrics(delta_true, delta_pred))

        try:
            def anchor_fn(a, b):
                dummy = np.zeros((len(a), VINA_DIM), dtype=np.float32)
                preds_a = [m.predict(a, b, dummy) for m in models]
                return np.mean(preds_a, axis=0)
            y_pred = reconstruct_absolute(test_idx, train_idx, X_fp, y_all, anchor_fn, 50)
            all_abs.append({"mae": float(np.mean(np.abs(y_pred - y_all[test_idx]))),
                "spearman": float(spearmanr(y_pred, y_all[test_idx])[0])})
        except: pass
        print(f"    fold {fold_i}: MAE = {all_delta[-1]['mae']:.4f}")

    if all_delta:
        phase_results["MultiSeed_pretrained_ensemble"] = {
            "delta_mae_mean": float(np.mean([f["mae"] for f in all_delta])),
            "delta_mae_std": float(np.std([f["mae"] for f in all_delta])),
            "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_delta])),
            "abs_mae_mean": float(np.mean([f["mae"] for f in all_abs])) if all_abs else 0,
            "abs_spearman_mean": float(np.mean([f["spearman"] for f in all_abs])) if all_abs else 0,
            "n_seeds": len(ensemble_seeds), "n_folds": N_FOLDS,
        }
        print(f"  MultiSeed_pretrained_ensemble: MAE={phase_results['MultiSeed_pretrained_ensemble']['delta_mae_mean']:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  PHASE D FIXED — RESULTS")
    print(f"{'='*70}")
    for k, v in sorted(phase_results.items(), key=lambda x: x[1]["delta_mae_mean"]):
        print(f"  {k:<40} MAE={v['delta_mae_mean']:.4f}±{v.get('delta_mae_std',0):.3f} "
              f"Spr={v['delta_spearman_mean']:.3f} AbsMAE={v.get('abs_mae_mean',0):.4f}")
    print(f"{'='*70}")

    # Save to results
    results["phase_d_fixed"] = {
        "methods": phase_results,
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
        "note": "Fixed weight transfer (previous phase_d had bug where fit() rebuilt model)",
    }
    save_results(results)
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
