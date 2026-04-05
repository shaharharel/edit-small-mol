#!/usr/bin/env python3
"""
Docking Integration Iteration 2 — Smart Architecture & Feature Engineering.

Informed by Iteration 1 results:
  - FiLMDelta_vina (3d Vina diff) = best single model (MAE=0.726)
  - Pretraining adds ~1% improvement (MAE=0.726 → pretrained)
  - Complex features (SE3 83d, interaction 17d) HURT on 280 molecules
  - Simple features > complex features at this data scale

Iteration 2 strategies:
  Phase A: Feature Engineering (4 feature sets)
    1. vina_diff (3d) — baseline from iter 1
    2. vina_engineered (9d) — diff + absolute means + ratios
    3. vina_selected (11d) — diff + means + top-5 interaction features
    4. full_engineered (26d) — all engineered + full interaction diff

  Phase B: Architecture Improvements (3 architectures, best feature set)
    1. ResidualCorrectionFiLM — base FiLM + learned docking residual
    2. MultiTaskDockingFiLM — joint delta + Vina score prediction
    3. FeatureGatedFiLM — per-feature learned importance gates

  Phase C: Ensemble & Combination
    1. Multi-seed ensemble (5 seeds, average)
    2. Architecture ensemble (top 2 architectures averaged)
    3. Stacked: XGB on (FiLM_pred, docking_feats) as meta-features

  Phase D: Best + Pretraining (combine best from B/C with kinase pretraining)

Usage:
    conda run -n quris python -u experiments/run_docking_iteration2.py
    conda run -n quris python -u experiments/run_docking_iteration2.py --phase A
    conda run -n quris python -u experiments/run_docking_iteration2.py --phase A B C D
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
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, get_cv_splits, compute_fingerprints,
    compute_absolute_metrics, aggregate_cv_results, train_xgboost,
    N_JOBS, N_FOLDS, CV_SEED,
)
from experiments.run_paper_evaluation import RESULTS_DIR

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "docking_iteration2_results.json"
DEVICE = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else "cpu"

# Docking paths
DOCK_CHEMBL_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCK_CHEMBL_CSV = DOCK_CHEMBL_DIR / "docking_results.csv"
DOCK_CHEMBL_POSES = DOCK_CHEMBL_DIR / "poses"
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"

N_SEEDS = 3
SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════════════════════════════════════
# Results I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (reuse from run_docking_integration.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load molecule data, fingerprints, docking features, interaction features."""
    print("\n" + "="*70)
    print("  LOADING DATA")
    print("="*70)

    # Molecules
    mol_data, _ = load_zap70_molecules()
    print(f"  Molecules: {len(mol_data)}")

    # Docking results
    dock_df = pd.read_csv(DOCK_CHEMBL_CSV)
    dock_df = dock_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    dock_cols = ["molecule_chembl_id", "vina_score", "vina_inter", "vina_intra"]
    dock_subset = dock_df[dock_df["success"] == True][dock_cols].copy()
    mol_data = mol_data.merge(dock_subset, on="molecule_chembl_id", how="left")
    mol_data["has_dock"] = ~mol_data["vina_score"].isna()
    n_docked = mol_data["has_dock"].sum()
    print(f"  Docked: {n_docked}/{len(mol_data)}")

    # Fingerprints
    X_fp = compute_fingerprints(mol_data["smiles"].tolist(), fp_type="morgan", n_bits=2048)
    print(f"  Fingerprints: {X_fp.shape}")

    # Per-molecule Vina features (mean-imputed)
    vina_cols = ["vina_score", "vina_inter", "vina_intra"]
    vina_per_mol = mol_data[vina_cols].values.astype(np.float32)
    for col_i in range(vina_per_mol.shape[1]):
        mask = np.isnan(vina_per_mol[:, col_i])
        if mask.any() and not mask.all():
            vina_per_mol[mask, col_i] = np.nanmean(vina_per_mol[:, col_i])
        elif mask.all():
            vina_per_mol[:, col_i] = 0.0

    # Interaction features
    interact_per_mol = None
    from src.data.utils.interaction_features import (
        compute_all_interaction_features, INTERACTION_FEAT_DIM,
    )
    cache_path = str(DOCK_CHEMBL_DIR / "interaction_features_cache.npz")
    mol_ids = mol_data["molecule_chembl_id"].tolist()
    interact_dict = compute_all_interaction_features(
        poses_dir=str(DOCK_CHEMBL_POSES),
        receptor_path=str(RECEPTOR_PDBQT),
        mol_ids=mol_ids,
        pose_filename_template="{mol_id}_pose.pdbqt",
        cache_path=cache_path,
    )
    interact_per_mol = np.zeros((len(mol_ids), INTERACTION_FEAT_DIM), dtype=np.float32)
    for i, mid in enumerate(mol_ids):
        if mid in interact_dict:
            interact_per_mol[i] = interact_dict[mid]
    for col_i in range(interact_per_mol.shape[1]):
        mask = np.isnan(interact_per_mol[:, col_i])
        if mask.any() and not mask.all():
            interact_per_mol[mask, col_i] = np.nanmean(interact_per_mol[:, col_i])
        elif mask.all():
            interact_per_mol[:, col_i] = 0.0

    n_valid_int = sum(1 for v in interact_dict.values() if not np.all(np.isnan(v)))
    print(f"  Interaction features: {n_valid_int}/{len(mol_ids)} valid, {INTERACTION_FEAT_DIM}d")

    # Generate all pairs
    pairs_df = generate_all_pairs(mol_data)

    return mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df


def generate_all_pairs(mol_data):
    """Generate all unique pairs from N molecules."""
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
    print(f"  All-pairs: {len(df)} from {len(smiles)} molecules")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Metrics & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

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
    return {
        "mae": mae, "spearman": float(spr) if not np.isnan(spr) else 0.0,
        "pearson": float(pr) if not np.isnan(pr) else 0.0, "r2": r2,
    }


def reconstruct_absolute(test_idx, train_idx, X_fp, y_all, predict_fn, n_anchors=50):
    """Anchor-based absolute prediction."""
    anchor_idx = train_idx
    if n_anchors < len(anchor_idx):
        rng = np.random.RandomState(42)
        anchor_idx = rng.choice(anchor_idx, size=n_anchors, replace=False)

    preds = []
    for j in test_idx:
        anchor_preds = []
        for i in anchor_idx:
            dp = predict_fn(X_fp[i:i+1], X_fp[j:j+1])
            if isinstance(dp, np.ndarray):
                dp = dp.item()
            anchor_preds.append(y_all[i] + dp)
        preds.append(float(np.median(anchor_preds)))
    return np.array(preds)


def run_cv_evaluation(
    method_name, mol_data, X_fp, pairs_df,
    train_fn, predict_delta_fn_factory,
    seeds=None, n_anchors=50, verbose=True,
):
    """Run multi-seed 5-fold CV, return metrics dict."""
    if seeds is None:
        seeds = SEEDS

    y_all = mol_data["pIC50"].values

    all_fold_delta = []
    all_fold_abs = []

    for seed in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            try:
                model = train_fn(train_idx, fold_i, seed)
            except Exception as e:
                print(f"    {method_name} fold {fold_i} seed {seed} FAILED: {e}")
                import traceback; traceback.print_exc()
                continue

            if model is None:
                continue

            # Delta metrics on test pairs
            test_set = set(test_idx)
            test_mask = (
                pairs_df["idx_a"].isin(test_set) &
                pairs_df["idx_b"].isin(test_set)
            )
            test_pairs = pairs_df[test_mask]
            if len(test_pairs) == 0:
                continue

            delta_true = test_pairs["delta"].values.astype(np.float32)
            delta_pred_fn, anchor_pred_fn = predict_delta_fn_factory(model, test_pairs)
            delta_pred = delta_pred_fn()

            fold_metrics = compute_delta_metrics(delta_true, delta_pred)
            all_fold_delta.append(fold_metrics)

            # Absolute metrics via anchor reconstruction
            try:
                y_pred_abs = reconstruct_absolute(
                    test_idx, train_idx, X_fp, y_all, anchor_pred_fn, n_anchors,
                )
                y_true_abs = y_all[test_idx]
                abs_mae = float(np.mean(np.abs(y_pred_abs - y_true_abs)))
                abs_spr, _ = spearmanr(y_pred_abs, y_true_abs)
                all_fold_abs.append({"mae": abs_mae, "spearman": float(abs_spr)})
            except Exception:
                pass

        if verbose:
            avg_mae = np.mean([f["mae"] for f in all_fold_delta[-N_FOLDS:]])
            print(f"    {method_name} seed {seed}: avg delta MAE = {avg_mae:.4f}")

    if not all_fold_delta:
        return None

    result = {
        "delta_mae_mean": float(np.mean([f["mae"] for f in all_fold_delta])),
        "delta_mae_std": float(np.std([f["mae"] for f in all_fold_delta])),
        "delta_spearman_mean": float(np.mean([f["spearman"] for f in all_fold_delta])),
        "delta_spearman_std": float(np.std([f["spearman"] for f in all_fold_delta])),
        "delta_pearson_mean": float(np.mean([f["pearson"] for f in all_fold_delta])),
        "delta_r2_mean": float(np.mean([f["r2"] for f in all_fold_delta])),
        "n_seeds": len(seeds),
        "n_folds": N_FOLDS,
        "n_total_folds": len(all_fold_delta),
    }

    if all_fold_abs:
        result["abs_mae_mean"] = float(np.mean([f["mae"] for f in all_fold_abs]))
        result["abs_mae_std"] = float(np.std([f["mae"] for f in all_fold_abs]))
        result["abs_spearman_mean"] = float(np.mean([f["spearman"] for f in all_fold_abs]))

    return result


def print_comparison(method_results, title="Comparison"):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Method':<35} {'ΔMAE':>10} {'ΔSpr':>8} "
          f"{'AbsMAE':>10} {'AbsSpr':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    sorted_methods = sorted(method_results.items(),
                            key=lambda x: x[1].get("delta_mae_mean", 999))

    for name, res in sorted_methods:
        d_mae = res.get("delta_mae_mean", float('nan'))
        d_std = res.get("delta_mae_std", 0)
        d_spr = res.get("delta_spearman_mean", float('nan'))
        a_mae = res.get("abs_mae_mean", float('nan'))
        a_spr = res.get("abs_spearman_mean", float('nan'))

        d_str = f"{d_mae:.4f}±{d_std:.3f}" if d_std > 0 else f"{d_mae:.4f}"
        print(f"  {name:<35} {d_str:>10} {d_spr:>8.3f} "
              f"{a_mae:>10.4f} {a_spr:>8.3f}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_standard_film(train_idx, X_fp, pairs_df, seed, device=DEVICE):
    """Train standard FiLMDelta (no docking features)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

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

    predictor = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        max_epochs=100, patience=15, device=device,
    )
    predictor.fit(
        X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
        tp.iloc[ti]["delta"].values.astype(np.float32),
        X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
        tp.iloc[vi]["delta"].values.astype(np.float32),
        verbose=False,
    )
    return predictor


def train_docking_film(
    train_idx, X_fp, pairs_df, pair_dock_feats, seed,
    arch="docking_film", extra_dim=3, device=DEVICE,
):
    """Train a docking-aware FiLM model."""
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

    predictor = DockingFiLMPredictor(
        arch=arch, extra_dim=extra_dim,
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        max_epochs=100, patience=15, device=device,
    )
    predictor.fit(
        X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
        pair_dock_feats[pair_indices_tr],
        tp.iloc[ti]["delta"].values.astype(np.float32),
        X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
        pair_dock_feats[pair_indices_val],
        tp.iloc[vi]["delta"].values.astype(np.float32),
        verbose=False,
    )
    return predictor


def train_advanced_film(
    train_idx, X_fp, pairs_df, pair_dock_feats, seed,
    arch="residual", extra_dim=3, device=DEVICE,
    vina_per_mol=None, aux_weight=0.1,
    snapshot_ensemble=False,
):
    """Train an advanced docking-aware FiLM model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models.predictors.advanced_docking_film import AdvancedDockingFiLMPredictor

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

    predictor = AdvancedDockingFiLMPredictor(
        arch=arch, extra_dim=extra_dim,
        dropout=0.2, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=64, max_epochs=100, patience=15, device=device,
        grad_clip=1.0, aux_weight=aux_weight,
        snapshot_ensemble=snapshot_ensemble,
    )

    # Build kwargs
    kwargs = dict(
        emb_a_train=X_fp[tp.iloc[ti]["idx_a"].values],
        emb_b_train=X_fp[tp.iloc[ti]["idx_b"].values],
        dock_feats_train=pair_dock_feats[pair_indices_tr],
        delta_train=tp.iloc[ti]["delta"].values.astype(np.float32),
        emb_a_val=X_fp[tp.iloc[vi]["idx_a"].values],
        emb_b_val=X_fp[tp.iloc[vi]["idx_b"].values],
        dock_feats_val=pair_dock_feats[pair_indices_val],
        delta_val=tp.iloc[vi]["delta"].values.astype(np.float32),
        verbose=False,
    )

    # Multi-task: provide Vina scores for mol_a and mol_b
    if arch == "multitask" and vina_per_mol is not None:
        # Vina score (column 0) for each mol in pair
        kwargs["vina_a_train"] = vina_per_mol[tp.iloc[ti]["idx_a"].values, 0]
        kwargs["vina_b_train"] = vina_per_mol[tp.iloc[ti]["idx_b"].values, 0]
        kwargs["vina_a_val"] = vina_per_mol[tp.iloc[vi]["idx_a"].values, 0]
        kwargs["vina_b_val"] = vina_per_mol[tp.iloc[vi]["idx_b"].values, 0]

    predictor.fit(**kwargs)
    return predictor


# ═══════════════════════════════════════════════════════════════════════════
# PHASE A: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results):
    print("\n" + "="*70)
    print("  PHASE A: Feature Engineering (4 feature sets)")
    print("="*70)
    t0 = time.time()

    from src.models.predictors.advanced_docking_film import engineer_docking_features

    # Compute all feature sets
    feature_sets = {
        "vina_diff": ("vina_diff", 3),
        "vina_engineered": ("vina_engineered", 9),
        "vina_selected": ("vina_selected", 11),
        "full_engineered": ("full_engineered", 26),
    }

    all_pair_feats = {}
    for name, (fset, dim) in feature_sets.items():
        feats = engineer_docking_features(
            vina_per_mol, pairs_df, interact_per_mol, feature_set=fset,
        )
        all_pair_feats[name] = feats
        print(f"  Feature set '{name}': {feats.shape}")

    phase_results = {}

    for fs_name, (fset, dim) in feature_sets.items():
        pair_feats = all_pair_feats[fs_name]
        method_name = f"DockFiLM_{fs_name}"
        print(f"\n  Method: {method_name} ({dim}d features)...")

        def make_train_fn(pf, d):
            def train_fn(train_idx, fold_i, seed):
                return train_docking_film(
                    train_idx, X_fp, pairs_df, pf, seed,
                    arch="docking_film", extra_dim=d,
                )
            return train_fn

        def make_pred_factory(pf):
            def pred_factory(model, test_pairs):
                pair_indices = test_pairs.index.values
                def delta_fn():
                    return model.predict(
                        X_fp[test_pairs["idx_a"].values],
                        X_fp[test_pairs["idx_b"].values],
                        pf[pair_indices],
                    )
                def anchor_fn(emb_a, emb_b):
                    # For anchor reconstruction, no pair dock feats — use zeros
                    dummy = np.zeros((len(emb_a), pf.shape[1]), dtype=np.float32)
                    return model.predict(emb_a, emb_b, dummy)
                return delta_fn, anchor_fn
            return pred_factory

        res = run_cv_evaluation(
            method_name, mol_data, X_fp, pairs_df,
            make_train_fn(pair_feats, dim),
            make_pred_factory(pair_feats),
            seeds=SEEDS, verbose=True,
        )

        if res:
            phase_results[method_name] = res
            save_results({**results, "phase_a": {"methods": phase_results,
                          "completed": False, "timestamp": str(datetime.now())}})

        gc.collect()

    # Also run baseline FiLMDelta (no docking)
    print(f"\n  Method: FiLMDelta_baseline (no docking)...")

    def base_train_fn(train_idx, fold_i, seed):
        return train_standard_film(train_idx, X_fp, pairs_df, seed)

    def base_pred_factory(model, test_pairs):
        def delta_fn():
            return model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
            )
        def anchor_fn(emb_a, emb_b):
            return model.predict(emb_a, emb_b)
        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "FiLMDelta_baseline", mol_data, X_fp, pairs_df,
        base_train_fn, base_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["FiLMDelta_baseline"] = res

    print_comparison(phase_results, "Phase A: Feature Engineering Results")

    # Determine best feature set
    best_fs = min(
        [(k, v["delta_mae_mean"]) for k, v in phase_results.items() if k != "FiLMDelta_baseline"],
        key=lambda x: x[1],
    )
    print(f"\n  Best feature set: {best_fs[0]} (MAE={best_fs[1]:.4f})")

    results["phase_a"] = {
        "methods": phase_results,
        "best_feature_set": best_fs[0],
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE B: Architecture Improvements
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results):
    print("\n" + "="*70)
    print("  PHASE B: Architecture Improvements")
    print("="*70)
    t0 = time.time()

    from src.models.predictors.advanced_docking_film import engineer_docking_features

    # Use best feature set from Phase A (or default to vina_diff)
    best_fs_name = "vina_diff"
    if "phase_a" in results and "best_feature_set" in results["phase_a"]:
        best_fs_key = results["phase_a"]["best_feature_set"]
        # Extract feature set name from method name (e.g., "DockFiLM_vina_engineered" -> "vina_engineered")
        best_fs_name = best_fs_key.replace("DockFiLM_", "")

    # Also try vina_diff for all architectures (since it won in iter 1)
    feature_sets_to_try = [best_fs_name]
    if best_fs_name != "vina_diff":
        feature_sets_to_try.append("vina_diff")

    fs_dim_map = {
        "vina_diff": 3, "vina_engineered": 9,
        "vina_selected": 11, "full_engineered": 26,
    }

    phase_results = {}

    for fs_name in feature_sets_to_try:
        dim = fs_dim_map[fs_name]
        pair_feats = engineer_docking_features(
            vina_per_mol, pairs_df, interact_per_mol, feature_set=fs_name,
        )

        architectures = [
            ("Residual", "residual"),
            ("MultiTask", "multitask"),
            ("FeatureGated", "feature_gated"),
        ]

        for arch_label, arch_name in architectures:
            method_name = f"{arch_label}_{fs_name}"
            print(f"\n  Method: {method_name} ({arch_name}, {dim}d)...")

            def make_train_fn(pf, d, an, fsn):
                def train_fn(train_idx, fold_i, seed):
                    return train_advanced_film(
                        train_idx, X_fp, pairs_df, pf, seed,
                        arch=an, extra_dim=d,
                        vina_per_mol=vina_per_mol if an == "multitask" else None,
                    )
                return train_fn

            def make_pred_factory(pf):
                def pred_factory(model, test_pairs):
                    pair_indices = test_pairs.index.values
                    def delta_fn():
                        return model.predict(
                            X_fp[test_pairs["idx_a"].values],
                            X_fp[test_pairs["idx_b"].values],
                            pf[pair_indices],
                        )
                    def anchor_fn(emb_a, emb_b):
                        dummy = np.zeros((len(emb_a), pf.shape[1]), dtype=np.float32)
                        return model.predict(emb_a, emb_b, dummy)
                    return delta_fn, anchor_fn
                return pred_factory

            res = run_cv_evaluation(
                method_name, mol_data, X_fp, pairs_df,
                make_train_fn(pair_feats, dim, arch_name, fs_name),
                make_pred_factory(pair_feats),
                seeds=SEEDS, verbose=True,
            )

            if res:
                phase_results[method_name] = res
                save_results({**results, "phase_b": {"methods": phase_results,
                              "completed": False, "timestamp": str(datetime.now())}})

            gc.collect()

    print_comparison(phase_results, "Phase B: Architecture Improvement Results")

    # Determine overall best
    all_methods = {}
    if "phase_a" in results and "methods" in results["phase_a"]:
        all_methods.update(results["phase_a"]["methods"])
    all_methods.update(phase_results)

    best = min(all_methods.items(), key=lambda x: x[1].get("delta_mae_mean", 999))
    print(f"\n  Overall best (A+B): {best[0]} (MAE={best[1]['delta_mae_mean']:.4f})")

    results["phase_b"] = {
        "methods": phase_results,
        "best_overall": best[0],
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE C: Ensemble & Stacking
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results):
    print("\n" + "="*70)
    print("  PHASE C: Ensemble & Stacking")
    print("="*70)
    t0 = time.time()

    from src.models.predictors.advanced_docking_film import (
        engineer_docking_features, EnsemblePredictor,
    )
    from xgboost import XGBRegressor

    # Use vina_diff (3d) as the reliable feature set
    pair_feats_3d = engineer_docking_features(
        vina_per_mol, pairs_df, interact_per_mol, feature_set="vina_diff",
    )

    phase_results = {}
    y_all = mol_data["pIC50"].values

    # Method 1: Multi-seed ensemble (5 seeds, docking_film with vina_diff)
    print(f"\n  Method: MultiSeed_ensemble (5 seeds, averaged)...")
    ensemble_seeds = [42, 123, 456, 789, 1024]

    def ensemble_train_fn(train_idx, fold_i, seed):
        """Train 5 models with different seeds, return ensemble."""
        models = []
        for s in ensemble_seeds:
            m = train_docking_film(
                train_idx, X_fp, pairs_df, pair_feats_3d, s,
                arch="docking_film", extra_dim=3,
            )
            if m is not None:
                models.append(m)
        if not models:
            return None
        return EnsemblePredictor(models)

    def ensemble_pred_factory(model, test_pairs):
        pair_indices = test_pairs.index.values
        def delta_fn():
            return model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
                pair_feats_3d[pair_indices],
            )
        def anchor_fn(emb_a, emb_b):
            dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
            return model.predict(emb_a, emb_b, dummy)
        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "MultiSeed_ensemble", mol_data, X_fp, pairs_df,
        ensemble_train_fn, ensemble_pred_factory,
        seeds=[42],  # Only 1 outer seed since ensemble is already multi-seed
        verbose=True,
    )
    if res:
        phase_results["MultiSeed_ensemble"] = res
        save_results({**results, "phase_c": {"methods": phase_results,
                      "completed": False, "timestamp": str(datetime.now())}})

    # Method 2: Architecture ensemble (DockingFiLM + ResidualFiLM + base FiLM)
    print(f"\n  Method: Arch_ensemble (DockFiLM + Residual + base)...")

    def arch_ensemble_train_fn(train_idx, fold_i, seed):
        # Standard FiLMDelta
        m1 = train_standard_film(train_idx, X_fp, pairs_df, seed)
        # DockingFiLM with Vina diff
        m2 = train_docking_film(
            train_idx, X_fp, pairs_df, pair_feats_3d, seed,
            arch="docking_film", extra_dim=3,
        )
        # Residual correction FiLM
        m3 = train_advanced_film(
            train_idx, X_fp, pairs_df, pair_feats_3d, seed,
            arch="residual", extra_dim=3,
        )
        models = [m for m in [m1, m2, m3] if m is not None]
        if not models:
            return None
        return (models, "arch_ensemble")

    def arch_ensemble_pred_factory(model_tuple, test_pairs):
        models, _ = model_tuple
        pair_indices = test_pairs.index.values

        def delta_fn():
            preds = []
            for m in models:
                try:
                    p = m.predict(
                        X_fp[test_pairs["idx_a"].values],
                        X_fp[test_pairs["idx_b"].values],
                        pair_feats_3d[pair_indices],
                    )
                except TypeError:
                    p = m.predict(
                        X_fp[test_pairs["idx_a"].values],
                        X_fp[test_pairs["idx_b"].values],
                    )
                preds.append(p)
            return np.mean(preds, axis=0)

        def anchor_fn(emb_a, emb_b):
            preds = []
            for m in models:
                try:
                    dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
                    p = m.predict(emb_a, emb_b, dummy)
                except TypeError:
                    p = m.predict(emb_a, emb_b)
                preds.append(p)
            return np.mean(preds, axis=0)

        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "Arch_ensemble", mol_data, X_fp, pairs_df,
        arch_ensemble_train_fn, arch_ensemble_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["Arch_ensemble"] = res
        save_results({**results, "phase_c": {"methods": phase_results,
                      "completed": False, "timestamp": str(datetime.now())}})

    # Method 3: Stacked XGB (FiLM predictions + docking feats as meta-features)
    print(f"\n  Method: Stacked_XGB (FiLM pred + dock feats → XGB)...")

    def stacked_train_fn(train_idx, fold_i, seed):
        """Two-stage: (1) train FiLM, (2) train XGB on FiLM residuals + dock."""
        # Stage 1: FiLMDelta with Vina
        film_model = train_docking_film(
            train_idx, X_fp, pairs_df, pair_feats_3d, seed,
            arch="docking_film", extra_dim=3,
        )
        if film_model is None:
            return None

        # Get FiLM predictions on train pairs for stage 2
        train_set = set(train_idx)
        mask = pairs_df["idx_a"].isin(train_set) & pairs_df["idx_b"].isin(train_set)
        tp = pairs_df[mask]

        film_preds = film_model.predict(
            X_fp[tp["idx_a"].values],
            X_fp[tp["idx_b"].values],
            pair_feats_3d[tp.index.values],
        )
        delta_true = tp["delta"].values.astype(np.float32)
        residuals = delta_true - film_preds

        # Stage 2: XGB on (film_pred, dock_diff) -> residual
        meta_feats = np.column_stack([
            film_preds.reshape(-1, 1),
            pair_feats_3d[tp.index.values],
        ])

        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            n_jobs=N_JOBS, random_state=seed,
        )
        xgb.fit(meta_feats, residuals)

        return (film_model, xgb, "stacked")

    def stacked_pred_factory(model_tuple, test_pairs):
        film_model, xgb, _ = model_tuple
        pair_indices = test_pairs.index.values

        def delta_fn():
            film_preds = film_model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
                pair_feats_3d[pair_indices],
            )
            meta_feats = np.column_stack([
                film_preds.reshape(-1, 1),
                pair_feats_3d[pair_indices],
            ])
            corrections = xgb.predict(meta_feats)
            return film_preds + corrections

        def anchor_fn(emb_a, emb_b):
            dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
            film_pred = film_model.predict(emb_a, emb_b, dummy)
            meta_feats = np.column_stack([
                np.atleast_1d(film_pred).reshape(-1, 1),
                dummy,
            ])
            correction = xgb.predict(meta_feats)
            return np.atleast_1d(film_pred) + correction

        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "Stacked_XGB", mol_data, X_fp, pairs_df,
        stacked_train_fn, stacked_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["Stacked_XGB"] = res

    # Method 4: Snapshot ensemble with docking FiLM
    print(f"\n  Method: Snapshot_ensemble (cosine annealing, 5 snapshots)...")

    def snapshot_train_fn(train_idx, fold_i, seed):
        return train_advanced_film(
            train_idx, X_fp, pairs_df, pair_feats_3d, seed,
            arch="feature_gated", extra_dim=3,
            snapshot_ensemble=True,
        )

    def snapshot_pred_factory(model, test_pairs):
        pair_indices = test_pairs.index.values
        def delta_fn():
            return model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
                pair_feats_3d[pair_indices],
            )
        def anchor_fn(emb_a, emb_b):
            dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
            return model.predict(emb_a, emb_b, dummy)
        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "Snapshot_ensemble", mol_data, X_fp, pairs_df,
        snapshot_train_fn, snapshot_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["Snapshot_ensemble"] = res

    print_comparison(phase_results, "Phase C: Ensemble & Stacking Results")

    # Overall best
    all_methods = {}
    for phase in ["phase_a", "phase_b"]:
        if phase in results and "methods" in results[phase]:
            all_methods.update(results[phase]["methods"])
    all_methods.update(phase_results)

    best = min(all_methods.items(), key=lambda x: x[1].get("delta_mae_mean", 999))
    print(f"\n  Overall best (A+B+C): {best[0]} (MAE={best[1]['delta_mae_mean']:.4f})")

    results["phase_c"] = {
        "methods": phase_results,
        "best_overall": best[0],
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE D: Best + Pretraining
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results):
    print("\n" + "="*70)
    print("  PHASE D: Best Architecture + Kinase Pretraining")
    print("="*70)
    t0 = time.time()

    from src.models.predictors.advanced_docking_film import engineer_docking_features
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP
    from src.models.predictors.docking_film_predictor import DockingFiLMPredictor, DockingFiLMDeltaMLP

    # Determine best approach from prior phases
    best_approach = "DockFiLM_vina_diff"  # Default
    best_mae = 999
    for phase in ["phase_a", "phase_b", "phase_c"]:
        if phase in results and "methods" in results[phase]:
            for name, res in results[phase]["methods"].items():
                mae = res.get("delta_mae_mean", 999)
                if mae < best_mae:
                    best_mae = mae
                    best_approach = name

    print(f"  Best approach from prior phases: {best_approach} (MAE={best_mae:.4f})")

    # Load kinase pretraining data
    pretrain_data_path = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
    if not pretrain_data_path.exists():
        print("  WARNING: Kinase pretraining data not found, skipping Phase D")
        return results

    print("  Loading kinase pretraining data...")
    kinase_targets = {
        "SYK": "CHEMBL2599", "LCK": "CHEMBL258",
        "JAK2": "CHEMBL2971", "ABL1": "CHEMBL1862",
        "SRC": "CHEMBL267", "BTK": "CHEMBL5251",
    }

    full_df = pd.read_csv(pretrain_data_path)
    kinase_df = full_df[full_df["target_chembl_id"].isin(kinase_targets.values())].copy()

    if "is_within_assay" in kinase_df.columns:
        kinase_df = kinase_df[kinase_df["is_within_assay"] == True]

    print(f"  Kinase pretraining: {len(kinase_df)} pairs from {kinase_df['target_chembl_id'].nunique()} targets")

    if len(kinase_df) == 0:
        print("  No kinase pairs found, skipping Phase D")
        return results

    # Get unique molecules and embeddings
    pretrain_smiles = list(set(kinase_df["mol_a"].tolist() + kinase_df["mol_b"].tolist()))
    print(f"  Unique pretrain molecules: {len(pretrain_smiles)}")

    pretrain_fps = compute_fingerprints(pretrain_smiles, fp_type="morgan", n_bits=2048)
    smi_to_idx = {s: i for i, s in enumerate(pretrain_smiles)}

    # Build pretrain tensors
    pretrain_a = np.array([pretrain_fps[smi_to_idx[s]] for s in kinase_df["mol_a"]])
    pretrain_b = np.array([pretrain_fps[smi_to_idx[s]] for s in kinase_df["mol_b"]])
    pretrain_delta = kinase_df["delta"].values.astype(np.float32)

    # Train/val split for pretraining
    rng = np.random.RandomState(42)
    n = len(pretrain_delta)
    val_n = max(int(n * 0.15), 200)
    perm = rng.permutation(n)
    pretrain_val_idx = perm[:val_n]
    pretrain_tr_idx = perm[val_n:]

    # Pretrain standard FiLMDelta
    print("\n  Pretraining FiLMDelta on kinase data...")
    pretrain_model = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        max_epochs=50, patience=10, device=DEVICE,
    )
    pretrain_model.fit(
        pretrain_a[pretrain_tr_idx], pretrain_b[pretrain_tr_idx],
        pretrain_delta[pretrain_tr_idx],
        pretrain_a[pretrain_val_idx], pretrain_b[pretrain_val_idx],
        pretrain_delta[pretrain_val_idx],
        verbose=True,
    )

    pretrained_state = {k: v.cpu().clone() for k, v in pretrain_model.model.state_dict().items()}
    print(f"  Pretraining complete. Weights saved.")

    phase_results = {}

    # Use vina_diff (3d) for docking features
    pair_feats_3d = engineer_docking_features(
        vina_per_mol, pairs_df, interact_per_mol, feature_set="vina_diff",
    )

    # Method 1: Pretrained FiLMDelta (no docking)
    print(f"\n  Method: FiLMDelta_pretrained (no docking)...")

    def pretrained_train_fn(train_idx, fold_i, seed):
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

        predictor = FiLMDeltaPredictor(
            dropout=0.2, learning_rate=5e-4, batch_size=64,
            max_epochs=100, patience=15, device=DEVICE,
        )
        # Initialize with pretrained weights
        predictor.fit(
            X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
            tp.iloc[ti]["delta"].values.astype(np.float32),
            X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
            tp.iloc[vi]["delta"].values.astype(np.float32),
            verbose=False,
        )

        # Reload pretrained weights and finetune
        try:
            predictor.model.load_state_dict(pretrained_state)
            predictor.model = predictor.model.to(DEVICE)
            # Finetune with lower LR
            predictor.learning_rate = 5e-4
            predictor.fit(
                X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
                tp.iloc[ti]["delta"].values.astype(np.float32),
                X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
                tp.iloc[vi]["delta"].values.astype(np.float32),
                verbose=False,
            )
        except Exception as e:
            print(f"    Weight transfer failed: {e}")

        return predictor

    def pretrained_pred_factory(model, test_pairs):
        def delta_fn():
            return model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
            )
        def anchor_fn(emb_a, emb_b):
            return model.predict(emb_a, emb_b)
        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "FiLMDelta_pretrained", mol_data, X_fp, pairs_df,
        pretrained_train_fn, pretrained_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["FiLMDelta_pretrained"] = res
        save_results({**results, "phase_d": {"methods": phase_results,
                      "completed": False, "timestamp": str(datetime.now())}})

    # Method 2: Pretrained + Vina docking
    print(f"\n  Method: FiLMDelta_vina_pretrained...")

    def pretrained_vina_train_fn(train_idx, fold_i, seed):
        """Transfer pretrained weights to DockingFiLMDeltaMLP, finetune with docking."""
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

        predictor = DockingFiLMPredictor(
            arch="docking_film", extra_dim=3,
            dropout=0.2, learning_rate=5e-4, batch_size=64,
            max_epochs=100, patience=15, device=DEVICE,
        )

        # First build the model by calling fit briefly
        predictor.fit(
            X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
            pair_feats_3d[pair_indices_tr],
            tp.iloc[ti]["delta"].values.astype(np.float32),
            X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
            pair_feats_3d[pair_indices_val],
            tp.iloc[vi]["delta"].values.astype(np.float32),
            verbose=False,
        )

        # Transfer pretrained weights (matching keys only)
        dock_state = predictor.model.state_dict()
        n_transferred = 0
        for k, v in pretrained_state.items():
            if k in dock_state and dock_state[k].shape == v.shape:
                dock_state[k] = v.clone()
                n_transferred += 1
        predictor.model.load_state_dict(dock_state)
        predictor.model = predictor.model.to(DEVICE)

        # Finetune
        predictor.fit(
            X_fp[tp.iloc[ti]["idx_a"].values], X_fp[tp.iloc[ti]["idx_b"].values],
            pair_feats_3d[pair_indices_tr],
            tp.iloc[ti]["delta"].values.astype(np.float32),
            X_fp[tp.iloc[vi]["idx_a"].values], X_fp[tp.iloc[vi]["idx_b"].values],
            pair_feats_3d[pair_indices_val],
            tp.iloc[vi]["delta"].values.astype(np.float32),
            verbose=False,
        )
        return predictor

    def pretrained_vina_pred_factory(model, test_pairs):
        pair_indices = test_pairs.index.values
        def delta_fn():
            return model.predict(
                X_fp[test_pairs["idx_a"].values],
                X_fp[test_pairs["idx_b"].values],
                pair_feats_3d[pair_indices],
            )
        def anchor_fn(emb_a, emb_b):
            dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
            return model.predict(emb_a, emb_b, dummy)
        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "FiLMDelta_vina_pretrained", mol_data, X_fp, pairs_df,
        pretrained_vina_train_fn, pretrained_vina_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["FiLMDelta_vina_pretrained"] = res

    # Method 3: Pretrained + Ensemble (multi-seed FiLM + pretrained vina)
    print(f"\n  Method: Pretrained_ensemble...")

    def pretrained_ensemble_train_fn(train_idx, fold_i, seed):
        """Ensemble of pretrained FiLMDelta + pretrained DockFiLM."""
        m1 = pretrained_train_fn(train_idx, fold_i, seed)
        m2 = pretrained_vina_train_fn(train_idx, fold_i, seed)
        models = [m for m in [m1, m2] if m is not None]
        if not models:
            return None
        return (models, "pretrained_ensemble")

    def pretrained_ensemble_pred_factory(model_tuple, test_pairs):
        models, _ = model_tuple
        pair_indices = test_pairs.index.values

        def delta_fn():
            preds = []
            for m in models:
                try:
                    p = m.predict(
                        X_fp[test_pairs["idx_a"].values],
                        X_fp[test_pairs["idx_b"].values],
                        pair_feats_3d[pair_indices],
                    )
                except TypeError:
                    p = m.predict(
                        X_fp[test_pairs["idx_a"].values],
                        X_fp[test_pairs["idx_b"].values],
                    )
                preds.append(p)
            return np.mean(preds, axis=0)

        def anchor_fn(emb_a, emb_b):
            preds = []
            for m in models:
                try:
                    dummy = np.zeros((len(emb_a), 3), dtype=np.float32)
                    p = m.predict(emb_a, emb_b, dummy)
                except TypeError:
                    p = m.predict(emb_a, emb_b)
                preds.append(p)
            return np.mean(preds, axis=0)

        return delta_fn, anchor_fn

    res = run_cv_evaluation(
        "Pretrained_ensemble", mol_data, X_fp, pairs_df,
        pretrained_ensemble_train_fn, pretrained_ensemble_pred_factory,
        seeds=SEEDS, verbose=True,
    )
    if res:
        phase_results["Pretrained_ensemble"] = res

    print_comparison(phase_results, "Phase D: Pretraining Results")

    # Grand summary
    all_methods = {}
    for phase in ["phase_a", "phase_b", "phase_c", "phase_d"]:
        if phase in results and "methods" in results[phase]:
            all_methods.update(results[phase]["methods"])
    all_methods.update(phase_results)

    best = min(all_methods.items(), key=lambda x: x[1].get("delta_mae_mean", 999))
    print(f"\n  GRAND BEST: {best[0]} (MAE={best[1]['delta_mae_mean']:.4f})")

    print_comparison(all_methods, "ALL METHODS — Final Ranking")

    results["phase_d"] = {
        "methods": phase_results,
        "grand_best": best[0],
        "completed": True,
        "time_s": time.time() - t0,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", nargs="+", default=["A", "B", "C", "D"],
                        help="Phases to run (A, B, C, D)")
    args = parser.parse_args()

    phases = [p.upper() for p in args.phase]
    print(f"\n{'#'*70}")
    print(f"  DOCKING INTEGRATION — ITERATION 2")
    print(f"  Phases: {', '.join(phases)}")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Started: {datetime.now()}")
    print(f"{'#'*70}")

    results = load_results()

    # Load all data once
    mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df = load_all_data()

    results["metadata"] = {
        "n_molecules": len(mol_data),
        "n_pairs": len(pairs_df),
        "device": DEVICE,
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "start_time": str(datetime.now()),
    }

    if "A" in phases:
        results = run_phase_a(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results)

    if "B" in phases:
        results = run_phase_b(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results)

    if "C" in phases:
        results = run_phase_c(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results)

    if "D" in phases:
        results = run_phase_d(mol_data, X_fp, vina_per_mol, interact_per_mol, pairs_df, results)

    print(f"\n{'#'*70}")
    print(f"  ITERATION 2 COMPLETE")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Finished: {datetime.now()}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
