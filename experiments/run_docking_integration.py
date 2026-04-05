#!/usr/bin/env python3
"""
Docking-Integrated ZAP70 Prediction Experiment.

Comprehensive comparison of XGB subtraction vs FiLMDelta with and without
docking features from AutoDock Vina. Tests whether 3D protein-ligand
binding information improves edit-effect prediction beyond 2D fingerprints.

Phases:
    1: Simple docking features (Vina score/inter/intra)
    2: Full interaction features (H-bonds, hydrophobic contacts, burial, etc.)
    3: Advanced architectures (DualStream, Hierarchical, PoseConditioned)
    3.5: SE(3)-invariant geometric features + antisymmetric augmentation
    4: Comprehensive comparison (all methods, 3 seeds)
    5: Ensemble & application to 509 generated molecules

Usage:
    conda run -n quris python -u experiments/run_docking_integration.py
    conda run -n quris python -u experiments/run_docking_integration.py --phase 1
    conda run -n quris python -u experiments/run_docking_integration.py --phase 1 2 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
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
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
# Use MPS if available for PyTorch acceleration

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, get_cv_splits, compute_fingerprints,
    compute_absolute_metrics, aggregate_cv_results, train_xgboost,
    N_JOBS, N_FOLDS, CV_SEED,
)
from experiments.run_paper_evaluation import RESULTS_DIR

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "docking_integration_results.json"
# Force MPS for FiLM MLPs (the MPS disable in run_zap70_v3 is for ChemBERTa, not MLPs)
DEVICE = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else "cpu"

# Docking data paths
DOCK_CHEMBL_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCK_CHEMBL_CSV = DOCK_CHEMBL_DIR / "docking_results.csv"
DOCK_CHEMBL_POSES = DOCK_CHEMBL_DIR / "poses"
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"

# Generated molecules docking
DOCK_GEN_DIR = PROJECT_ROOT / "data" / "docking_500"

# Best XGB hyperparams from v3
BEST_XGB_PARAMS = {
    "max_depth": 6, "min_child_weight": 2,
    "subsample": 0.605, "colsample_bytree": 0.520,
    "learning_rate": 0.0197, "n_estimators": 749,
    "reg_alpha": 1.579, "reg_lambda": 7.313,
}


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
    """Load docking results and merge with ZAP70 molecule data.

    Returns:
        mol_data: DataFrame with columns: molecule_chembl_id, smiles, pIC50,
                  vina_score, vina_inter, vina_intra, has_dock
        dock_available: bool, whether docking results exist
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

    # Merge on SMILES (dock CSV has 'smiles' and 'chembl_id')
    dock_df = dock_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    dock_cols = ["molecule_chembl_id", "vina_score", "vina_inter", "vina_intra"]
    dock_subset = dock_df[dock_df["success"] == True][dock_cols].copy()

    mol_data = mol_data.merge(dock_subset, on="molecule_chembl_id", how="left")
    mol_data["has_dock"] = ~mol_data["vina_score"].isna()

    n_docked = mol_data["has_dock"].sum()
    n_total = len(mol_data)
    print(f"  Merged: {n_docked}/{n_total} molecules have docking scores")

    if n_docked > 0:
        docked = mol_data[mol_data["has_dock"]]
        print(f"  Vina scores: {docked['vina_score'].mean():.2f} +/- "
              f"{docked['vina_score'].std():.2f} kcal/mol")

    return mol_data, n_docked > 0


def load_interaction_features(mol_data):
    """Compute interaction features from docked poses.

    Returns:
        interact_dict: dict mapping molecule_chembl_id -> feature vector (17d)
        interact_available: bool
    """
    if not DOCK_CHEMBL_POSES.exists() or not RECEPTOR_PDBQT.exists():
        print("  WARNING: Pose files or receptor not available for interaction features")
        return {}, False

    from src.data.utils.interaction_features import (
        compute_all_interaction_features, INTERACTION_FEAT_DIM,
    )

    mol_ids = mol_data["molecule_chembl_id"].tolist()
    cache_path = str(DOCK_CHEMBL_DIR / "interaction_features_cache.npz")

    print(f"  Computing interaction features for {len(mol_ids)} molecules...")
    interact_dict = compute_all_interaction_features(
        poses_dir=str(DOCK_CHEMBL_POSES),
        receptor_path=str(RECEPTOR_PDBQT),
        mol_ids=mol_ids,
        pose_filename_template="{mol_id}_pose.pdbqt",
        cache_path=cache_path,
    )

    n_valid = sum(1 for v in interact_dict.values() if not np.all(np.isnan(v)))
    print(f"  Interaction features: {n_valid}/{len(mol_ids)} valid")

    return interact_dict, n_valid > 0


def load_geometric_features(mol_data):
    """Compute SE(3)-invariant geometric features from docked poses.

    Returns:
        geo_dict: dict mapping molecule_chembl_id -> feature vector (~50d)
        geo_available: bool
        pocket_residues: list of pocket residue tuples (for feature names)
    """
    if not DOCK_CHEMBL_POSES.exists() or not RECEPTOR_PDBQT.exists():
        print("  WARNING: Pose files or receptor not available for geometric features")
        return {}, False, []

    from src.data.utils.interaction_features import (
        compute_all_geometric_features,
        get_pocket_residues,
        get_geometric_feature_names,
    )

    mol_ids = mol_data["molecule_chembl_id"].tolist()
    cache_path = str(DOCK_CHEMBL_DIR / "geometric_features_cache.npz")
    pocket_cache = str(DOCK_CHEMBL_DIR / "geometric_features_cache_pocket.json")

    # Get pocket residues
    print(f"  Detecting binding pocket residues...")
    pocket_residues = get_pocket_residues(
        str(RECEPTOR_PDBQT),
        str(DOCK_CHEMBL_POSES),
        n_samples=50,
        cache_path=pocket_cache,
    )
    print(f"  Pocket residues: {len(pocket_residues)}")

    if not pocket_residues:
        print("  WARNING: No pocket residues detected")
        return {}, False, []

    print(f"  Computing geometric features for {len(mol_ids)} molecules...")
    geo_dict = compute_all_geometric_features(
        poses_dir=str(DOCK_CHEMBL_POSES),
        receptor_path=str(RECEPTOR_PDBQT),
        mol_ids=mol_ids,
        pocket_residues=pocket_residues,
        pose_filename_template="{mol_id}_pose.pdbqt",
        cache_path=cache_path,
    )

    n_valid = sum(1 for v in geo_dict.values() if not np.all(np.isnan(v)))
    feat_names = get_geometric_feature_names(pocket_residues)
    print(f"  Geometric features: {n_valid}/{len(mol_ids)} valid, "
          f"{len(feat_names)} dims")

    return geo_dict, n_valid > 0, pocket_residues


def get_geometric_features_matrix(mol_data, geo_dict, pocket_residues):
    """Build per-molecule geometric feature matrix.

    Missing values are mean-imputed.

    Returns:
        geo_feats: np.ndarray of shape [N, geo_dim]
    """
    ids = mol_data["molecule_chembl_id"].values
    geo_dim = len(pocket_residues) + 10  # per-residue contacts + 10 global

    feats = np.full((len(ids), geo_dim), np.nan, dtype=np.float32)

    for i, mid in enumerate(ids):
        if mid in geo_dict:
            vec = geo_dict[mid]
            if len(vec) == geo_dim:
                feats[i] = vec

    # Mean-impute NaNs column by column
    for col_i in range(feats.shape[1]):
        col_vals = feats[:, col_i]
        mask = np.isnan(col_vals)
        if mask.any() and not mask.all():
            col_vals[mask] = np.nanmean(col_vals)
        elif mask.all():
            col_vals[:] = 0.0
        feats[:, col_i] = col_vals

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Pair Generation & Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

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
    print(f"  Generated {len(df)} all-pairs from {len(smiles)} molecules")
    return df


def get_vina_features_per_mol(mol_data):
    """Extract per-molecule Vina features [vina_score, vina_inter, vina_intra].

    Missing values are mean-imputed.

    Returns:
        vina_feats: np.ndarray of shape [N, 3]
    """
    cols = ["vina_score", "vina_inter", "vina_intra"]
    feats = mol_data[cols].values.astype(np.float32)

    # Mean-impute missing
    for col_i in range(feats.shape[1]):
        col_vals = feats[:, col_i]
        mask = np.isnan(col_vals)
        if mask.any() and not mask.all():
            col_vals[mask] = np.nanmean(col_vals)
        elif mask.all():
            col_vals[:] = 0.0
        feats[:, col_i] = col_vals

    return feats


def get_interaction_features_matrix(mol_data, interact_dict):
    """Build per-molecule interaction feature matrix.

    Missing values are mean-imputed.

    Returns:
        interact_feats: np.ndarray of shape [N, INTERACTION_FEAT_DIM]
    """
    from src.data.utils.interaction_features import INTERACTION_FEAT_DIM

    ids = mol_data["molecule_chembl_id"].values
    feats = np.zeros((len(ids), INTERACTION_FEAT_DIM), dtype=np.float32)

    for i, mid in enumerate(ids):
        if mid in interact_dict:
            feats[i] = interact_dict[mid]

    # Mean-impute NaNs column by column
    for col_i in range(feats.shape[1]):
        col_vals = feats[:, col_i]
        mask = np.isnan(col_vals)
        if mask.any() and not mask.all():
            col_vals[mask] = np.nanmean(col_vals)
        elif mask.all():
            col_vals[:] = 0.0
        feats[:, col_i] = col_vals

    return feats


def compute_pair_dock_diff(feats_per_mol, pairs_df):
    """Compute per-pair feature differences: feats[b] - feats[a].

    Args:
        feats_per_mol: [N_mols, D] array of per-molecule features
        pairs_df: DataFrame with 'idx_a' and 'idx_b' columns

    Returns:
        pair_feats: [N_pairs, D] array of feature differences
    """
    idx_a = pairs_df["idx_a"].values
    idx_b = pairs_df["idx_b"].values
    return feats_per_mol[idx_b] - feats_per_mol[idx_a]


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_metrics(delta_true, delta_pred):
    """Compute metrics on delta (pairwise difference) prediction."""
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
    test_mol_indices, train_mol_indices, all_smiles, X_fp, y_true_all,
    predict_delta_fn, n_anchors=None,
):
    """Reconstruct absolute pIC50 for test molecules using anchor-based
    median prediction: pred_j = median_i(pIC50_i + delta_pred(i->j))
    over train anchors.

    Args:
        test_mol_indices: indices of test molecules
        train_mol_indices: indices of train molecules (anchors)
        all_smiles: list of all SMILES
        X_fp: [N, D] fingerprint matrix
        y_true_all: [N] true pIC50 values
        predict_delta_fn: callable(emb_a, emb_b, ...) -> delta_pred array.
            Must accept (emb_a_batch, emb_b_batch) at minimum.
        n_anchors: max number of anchors to use (None = all train mols)

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


def print_comparison_table(method_results, title="Comparison"):
    """Print a formatted comparison table sorted by delta MAE."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Method':<30} {'Delta MAE':>10} {'Delta Spr':>10} "
          f"{'Abs MAE':>10} {'Abs Spr':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Sort by delta MAE
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

        print(f"  {name:<30} {d_mae_str:>10} {d_spr_str:>10} "
              f"{a_mae_str:>10} {a_spr_str:>10}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════
# FiLM Training Helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_film_on_pairs(
    train_mol_idx, X_fp, y_all, pairs_df,
    extra_feats_per_pair=None, arch="standard",
    extra_dim=0, seed=42,
):
    """Train a FiLM model on all-pairs from train molecules.

    Args:
        train_mol_idx: indices of training molecules
        X_fp: [N, D] fingerprint array for ALL molecules
        y_all: [N] pIC50 for ALL molecules
        pairs_df: DataFrame with all pairs (idx_a, idx_b, delta)
        extra_feats_per_pair: [N_pairs, extra_dim] extra features, or None
        arch: 'standard', 'docking_film', 'dual_stream', 'hierarchical'
        extra_dim: dimension of extra features
        seed: random seed

    Returns:
        predictor: trained predictor with .predict(emb_a, emb_b[, dock_feats])
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_smiles_idx_set = set(train_mol_idx)

    # Filter pairs to train molecules only
    mask = (
        pairs_df["idx_a"].isin(train_smiles_idx_set) &
        pairs_df["idx_b"].isin(train_smiles_idx_set)
    )
    train_pairs = pairs_df[mask]

    if len(train_pairs) == 0:
        return None

    # Split into train/val (85/15) for early stopping
    rng = np.random.RandomState(seed)
    n = len(train_pairs)
    val_size = max(int(n * 0.15), 100)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    emb_a_tr = X_fp[train_pairs.iloc[tr_idx]["idx_a"].values]
    emb_b_tr = X_fp[train_pairs.iloc[tr_idx]["idx_b"].values]
    delta_tr = train_pairs.iloc[tr_idx]["delta"].values.astype(np.float32)
    emb_a_val = X_fp[train_pairs.iloc[val_idx]["idx_a"].values]
    emb_b_val = X_fp[train_pairs.iloc[val_idx]["idx_b"].values]
    delta_val = train_pairs.iloc[val_idx]["delta"].values.astype(np.float32)

    if arch == "standard":
        from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
        predictor = FiLMDeltaPredictor(
            dropout=0.2, learning_rate=1e-3, batch_size=64,
            max_epochs=100, patience=15, device=DEVICE,
        )
        predictor.fit(emb_a_tr, emb_b_tr, delta_tr,
                      emb_a_val, emb_b_val, delta_val, verbose=False)
        return predictor
    else:
        from src.models.predictors.docking_film_predictor import DockingFiLMPredictor

        # Get extra features for train pairs
        if extra_feats_per_pair is not None:
            pair_indices_tr = train_pairs.iloc[tr_idx].index.values
            pair_indices_val = train_pairs.iloc[val_idx].index.values
            dock_train = extra_feats_per_pair[pair_indices_tr]
            dock_val = extra_feats_per_pair[pair_indices_val]
        else:
            dock_train = np.zeros((len(tr_idx), max(extra_dim, 1)), dtype=np.float32)
            dock_val = np.zeros((len(val_idx), max(extra_dim, 1)), dtype=np.float32)

        predictor = DockingFiLMPredictor(
            arch=arch, extra_dim=extra_dim,
            dropout=0.2, learning_rate=1e-3, batch_size=64,
            max_epochs=100, patience=15, device=DEVICE,
        )
        predictor.fit(emb_a_tr, emb_b_tr, dock_train, delta_tr,
                      emb_a_val, emb_b_val, dock_val, delta_val, verbose=False)
        return predictor


def evaluate_method_cv(
    method_name, mol_data, X_fp, pairs_df,
    train_fn, predict_delta_fn_factory,
    seeds=None, n_anchors=50,
):
    """Run 5-fold CV for a method, computing both delta and absolute metrics.

    Args:
        method_name: name for reporting
        mol_data: DataFrame with pIC50
        X_fp: [N, D] fingerprint matrix
        pairs_df: all pairs DataFrame
        train_fn: callable(train_idx, fold_i, seed) -> model or None
        predict_delta_fn_factory: callable(model, test_pairs) -> delta_pred array
            Also: callable(model) -> callable(emb_a, emb_b) for anchor recon
        seeds: list of seeds to run (default [42])
        n_anchors: number of anchors for absolute reconstruction

    Returns:
        result_dict with delta/absolute mean/std metrics
    """
    if seeds is None:
        seeds = [42]

    y_all = mol_data["pIC50"].values
    all_smiles = mol_data["smiles"].tolist()

    seed_results = []
    for seed in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        fold_delta_metrics = []
        fold_abs_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            try:
                model = train_fn(train_idx, fold_i, seed)
            except Exception as e:
                print(f"    {method_name} fold {fold_i} seed {seed} FAILED: {e}")
                continue

            if model is None:
                continue

            # Get test pairs (both mols in test set)
            test_set = set(test_idx)
            test_mask = (
                pairs_df["idx_a"].isin(test_set) &
                pairs_df["idx_b"].isin(test_set)
            )
            test_pairs = pairs_df[test_mask]

            if len(test_pairs) == 0:
                continue

            # Delta evaluation
            delta_true = test_pairs["delta"].values.astype(np.float32)
            try:
                delta_pred = predict_delta_fn_factory(model, test_pairs)
            except Exception as e:
                print(f"    {method_name} fold {fold_i} predict FAILED: {e}")
                continue

            d_metrics = compute_delta_metrics(delta_true, delta_pred)
            fold_delta_metrics.append(d_metrics)

            # Absolute reconstruction via anchors
            try:
                pred_fn = predict_delta_fn_factory(model, None)  # get raw predict fn
                if pred_fn is not None:
                    y_pred_abs = reconstruct_absolute_via_anchors(
                        test_idx, train_idx, all_smiles, X_fp, y_all,
                        pred_fn, n_anchors=n_anchors,
                    )
                    a_metrics = compute_absolute_metrics(y_all[test_idx], y_pred_abs)
                    fold_abs_metrics.append(a_metrics)
            except Exception:
                pass  # Absolute reconstruction is optional

        if fold_delta_metrics:
            seed_results.append({
                "delta": fold_delta_metrics,
                "abs": fold_abs_metrics,
            })

    # Aggregate across seeds and folds
    if not seed_results:
        return {"error": "No successful runs"}

    all_delta_mae = []
    all_delta_spr = []
    all_abs_mae = []
    all_abs_spr = []

    for sr in seed_results:
        for dm in sr["delta"]:
            all_delta_mae.append(dm["mae"])
            all_delta_spr.append(dm["spearman"])
        for am in sr["abs"]:
            all_abs_mae.append(am["mae"])
            all_abs_spr.append(am.get("spearman_r", 0))

    result = {
        "delta_mae_mean": float(np.mean(all_delta_mae)),
        "delta_mae_std": float(np.std(all_delta_mae)),
        "delta_spearman_mean": float(np.mean(all_delta_spr)),
        "delta_spearman_std": float(np.std(all_delta_spr)),
        "n_seeds": len(seeds),
        "n_folds": N_FOLDS,
    }
    if all_abs_mae:
        result["abs_mae_mean"] = float(np.mean(all_abs_mae))
        result["abs_mae_std"] = float(np.std(all_abs_mae))
        result["abs_spearman_mean"] = float(np.mean(all_abs_spr))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Simple Docking Features
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_1(mol_data, X_fp, pairs_df, vina_feats, results):
    """Phase 1: XGB and FiLM with simple Vina features."""
    print("\n" + "=" * 70)
    print("PHASE 1: Simple Docking Features (Vina score/inter/intra)")
    print("=" * 70)

    phase = results.get("phase_1", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    has_dock = not np.all(np.isnan(vina_feats))
    methods = {}

    # Compute per-pair vina diffs for FiLM
    vina_pair_diff = compute_pair_dock_diff(vina_feats, pairs_df)  # [N_pairs, 3]

    # ── Method 1: XGB Subtraction (baseline, no docking) ──
    print("\n  1. XGB_subtraction (baseline)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_delta = []
    fold_abs = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        preds_abs, _ = train_xgboost(
            X_fp[train_idx], y_all[train_idx], X_fp[test_idx],
            **BEST_XGB_PARAMS,
        )
        fold_abs.append(compute_absolute_metrics(y_all[test_idx], preds_abs))

        # Delta from subtraction
        abs_map = dict(zip(test_idx, preds_abs))
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            delta_true = test_pairs["delta"].values
            delta_pred = np.array([
                abs_map[b] - abs_map[a]
                for a, b in zip(test_pairs["idx_a"], test_pairs["idx_b"])
            ])
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    methods["XGB_subtraction"] = {
        "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
        "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
        "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        "abs_mae_mean": float(np.mean([m["mae"] for m in fold_abs])),
        "abs_spearman_mean": float(np.mean([m.get("spearman_r", 0) for m in fold_abs])),
    }
    _print_method_result("XGB_subtraction", methods["XGB_subtraction"])

    # ── Method 2: XGB Subtraction + Vina ──
    if has_dock:
        print("\n  2. XGB_subtraction_vina (Morgan + 3 Vina features)...")
        X_fp_vina = np.hstack([X_fp, vina_feats])
        fold_delta = []
        fold_abs = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            preds_abs, _ = train_xgboost(
                X_fp_vina[train_idx], y_all[train_idx], X_fp_vina[test_idx],
                **BEST_XGB_PARAMS,
            )
            fold_abs.append(compute_absolute_metrics(y_all[test_idx], preds_abs))

            abs_map = dict(zip(test_idx, preds_abs))
            test_set = set(test_idx)
            test_pairs = pairs_df[
                pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            ]
            if len(test_pairs) > 0:
                delta_true = test_pairs["delta"].values
                delta_pred = np.array([
                    abs_map[b] - abs_map[a]
                    for a, b in zip(test_pairs["idx_a"], test_pairs["idx_b"])
                ])
                fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

        methods["XGB_subtraction_vina"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
            "abs_mae_mean": float(np.mean([m["mae"] for m in fold_abs])),
            "abs_spearman_mean": float(np.mean([m.get("spearman_r", 0) for m in fold_abs])),
        }
        _print_method_result("XGB_subtraction_vina", methods["XGB_subtraction_vina"])
    else:
        print("\n  2. XGB_subtraction_vina — SKIPPED (no docking data)")

    # ── Method 3: FiLMDelta (baseline, no docking) ──
    print("\n  3. FiLMDelta (baseline)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df, arch="standard",
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            delta_pred = predictor.predict(emb_a, emb_b)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["FiLMDelta"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("FiLMDelta", methods["FiLMDelta"])

    # ── Method 4: FiLMDelta + Vina ──
    if has_dock:
        print("\n  4. FiLMDelta_vina (Morgan diff + delta Vina features)...")
        fold_delta = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            predictor = train_film_on_pairs(
                train_idx, X_fp, y_all, pairs_df,
                extra_feats_per_pair=vina_pair_diff,
                arch="docking_film", extra_dim=3,
            )
            if predictor is None:
                continue
            test_set = set(test_idx)
            test_pairs = pairs_df[
                pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            ]
            if len(test_pairs) > 0:
                emb_a = X_fp[test_pairs["idx_a"].values]
                emb_b = X_fp[test_pairs["idx_b"].values]
                dock_diff = vina_pair_diff[test_pairs.index.values]
                delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
                delta_true = test_pairs["delta"].values.astype(np.float32)
                fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

        if fold_delta:
            methods["FiLMDelta_vina"] = {
                "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
                "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
                "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
            }
            _print_method_result("FiLMDelta_vina", methods["FiLMDelta_vina"])
    else:
        print("\n  4. FiLMDelta_vina — SKIPPED (no docking data)")

    results["phase_1"] = {
        "methods": methods,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print_comparison_table(methods, "Phase 1 Results")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Interaction Features
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_2(mol_data, X_fp, pairs_df, vina_feats, interact_feats, results):
    """Phase 2: Full interaction features (H-bonds, hydrophobic, burial, etc.)."""
    print("\n" + "=" * 70)
    print("PHASE 2: Interaction Features")
    print("=" * 70)

    phase = results.get("phase_2", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    methods = {}

    has_interact = interact_feats is not None and interact_feats.shape[1] > 0
    interact_dim = interact_feats.shape[1] if has_interact else 0

    if not has_interact:
        print("  WARNING: No interaction features available. Skipping Phase 2.")
        results["phase_2"] = {"methods": {}, "completed": True, "skipped": True}
        save_results(results)
        return results

    interact_pair_diff = compute_pair_dock_diff(interact_feats, pairs_df)

    # ── Method 5: XGB_subtraction_full (Morgan + Vina + interaction) ──
    print("\n  5. XGB_subtraction_full (Morgan + Vina + interaction FP)...")
    X_full = np.hstack([X_fp, vina_feats, interact_feats])
    fold_delta = []
    fold_abs = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        preds_abs, _ = train_xgboost(
            X_full[train_idx], y_all[train_idx], X_full[test_idx],
            **BEST_XGB_PARAMS,
        )
        fold_abs.append(compute_absolute_metrics(y_all[test_idx], preds_abs))

        abs_map = dict(zip(test_idx, preds_abs))
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            delta_true = test_pairs["delta"].values
            delta_pred = np.array([
                abs_map[b] - abs_map[a]
                for a, b in zip(test_pairs["idx_a"], test_pairs["idx_b"])
            ])
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    methods["XGB_subtraction_full"] = {
        "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
        "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
        "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        "abs_mae_mean": float(np.mean([m["mae"] for m in fold_abs])),
        "abs_spearman_mean": float(np.mean([m.get("spearman_r", 0) for m in fold_abs])),
    }
    _print_method_result("XGB_subtraction_full", methods["XGB_subtraction_full"])

    # ── Method 6: FiLMDelta_interact (interaction feature diffs as conditioning) ──
    print("\n  6. FiLMDelta_interact (interaction feature diffs)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df,
            extra_feats_per_pair=interact_pair_diff,
            arch="docking_film", extra_dim=interact_dim,
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = interact_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["FiLMDelta_interact"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("FiLMDelta_interact", methods["FiLMDelta_interact"])

    # ── Method 7: XGB_delta_direct (XGB trained on pair features directly) ──
    print("\n  7. XGB_delta_direct (XGB on pair features: morgan_a, morgan_b, diff, vina, interact)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        train_set = set(train_idx)
        test_set = set(test_idx)
        train_pairs = pairs_df[
            pairs_df["idx_a"].isin(train_set) & pairs_df["idx_b"].isin(train_set)
        ]
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) == 0:
            continue

        def _build_pair_feats(p_df):
            fa = X_fp[p_df["idx_a"].values]
            fb = X_fp[p_df["idx_b"].values]
            fdiff = fb - fa
            va = vina_feats[p_df["idx_a"].values]
            vb = vina_feats[p_df["idx_b"].values]
            ia = interact_feats[p_df["idx_a"].values]
            ib = interact_feats[p_df["idx_b"].values]
            return np.hstack([fa, fb, fdiff, va, vb, ia, ib])

        X_tr = _build_pair_feats(train_pairs)
        y_tr = train_pairs["delta"].values.astype(np.float32)
        X_te = _build_pair_feats(test_pairs)

        preds, _ = train_xgboost(X_tr, y_tr, X_te, **BEST_XGB_PARAMS)
        delta_true = test_pairs["delta"].values.astype(np.float32)
        fold_delta.append(compute_delta_metrics(delta_true, preds))

    if fold_delta:
        methods["XGB_delta_direct"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("XGB_delta_direct", methods["XGB_delta_direct"])

    results["phase_2"] = {
        "methods": methods,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print_comparison_table(methods, "Phase 2 Results")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Advanced Architectures
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_3(mol_data, X_fp, pairs_df, vina_feats, interact_feats, results):
    """Phase 3: DualStream, Hierarchical, and PoseConditioned FiLM."""
    print("\n" + "=" * 70)
    print("PHASE 3: Advanced Architectures")
    print("=" * 70)

    phase = results.get("phase_3", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    methods = {}

    has_interact = interact_feats is not None and interact_feats.shape[1] > 0
    has_vina = not np.all(np.isnan(vina_feats))

    if not has_vina:
        print("  WARNING: No docking data available. Skipping Phase 3.")
        results["phase_3"] = {"methods": {}, "completed": True, "skipped": True}
        save_results(results)
        return results

    # Combine vina + interaction as the full docking feature set
    if has_interact:
        full_dock_feats = np.hstack([vina_feats, interact_feats])
    else:
        full_dock_feats = vina_feats
    full_dock_dim = full_dock_feats.shape[1]
    full_dock_pair_diff = compute_pair_dock_diff(full_dock_feats, pairs_df)

    # Interaction-only pair diff (for PoseConditioned)
    if has_interact:
        interact_pair_diff = compute_pair_dock_diff(interact_feats, pairs_df)
        interact_dim = interact_feats.shape[1]
    else:
        interact_pair_diff = None
        interact_dim = 0

    # ── Method 8: DualStream_dock ──
    print("\n  8. DualStream_dock (gated 2D+3D fusion)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df,
            extra_feats_per_pair=full_dock_pair_diff,
            arch="dual_stream", extra_dim=full_dock_dim,
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = full_dock_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["DualStream_dock"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("DualStream_dock", methods["DualStream_dock"])

    # ── Method 9: Hierarchical_dock ──
    print("\n  9. Hierarchical_dock (two-level FiLM: chemical then docking)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df,
            extra_feats_per_pair=full_dock_pair_diff,
            arch="hierarchical", extra_dim=full_dock_dim,
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = full_dock_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["Hierarchical_dock"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("Hierarchical_dock", methods["Hierarchical_dock"])

    # ── Method 10: PoseConditioned (interaction features only, no raw Vina) ──
    if has_interact:
        print("\n  10. PoseConditioned (interaction-only conditioning)...")
        fold_delta = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            predictor = train_film_on_pairs(
                train_idx, X_fp, y_all, pairs_df,
                extra_feats_per_pair=interact_pair_diff,
                arch="docking_film", extra_dim=interact_dim,
            )
            if predictor is None:
                continue
            test_set = set(test_idx)
            test_pairs = pairs_df[
                pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
            ]
            if len(test_pairs) > 0:
                emb_a = X_fp[test_pairs["idx_a"].values]
                emb_b = X_fp[test_pairs["idx_b"].values]
                dock_diff = interact_pair_diff[test_pairs.index.values]
                delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
                delta_true = test_pairs["delta"].values.astype(np.float32)
                fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

        if fold_delta:
            methods["PoseConditioned"] = {
                "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
                "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
                "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
            }
            _print_method_result("PoseConditioned", methods["PoseConditioned"])
    else:
        print("\n  10. PoseConditioned — SKIPPED (no interaction features)")

    results["phase_3"] = {
        "methods": methods,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print_comparison_table(methods, "Phase 3 Results")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3.5: SE(3)-Invariant Geometric Features + Antisymmetric Aug
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_3_5(mol_data, X_fp, pairs_df, vina_feats, interact_feats,
                  geo_feats, results):
    """Phase 3.5: SE(3)-invariant geometric features and antisymmetric augmentation.

    Three new methods:
        SE3_FiLM — DockingFiLMDeltaMLP conditioned on geometric features
        SE3_DualStream — DockingDualStreamFiLM with geometric features as 3D stream
        SE3_antisym — SE3_FiLM with antisymmetric data augmentation
    """
    print("\n" + "=" * 70)
    print("PHASE 3.5: SE(3)-Invariant Geometric Features + Antisymmetric Aug")
    print("=" * 70)

    phase = results.get("phase_3_5", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    methods = {}

    has_geo = geo_feats is not None and geo_feats.shape[1] > 0
    has_interact = interact_feats is not None and interact_feats.shape[1] > 0

    if not has_geo:
        print("  WARNING: No geometric features available. Skipping Phase 3.5.")
        results["phase_3_5"] = {"methods": {}, "completed": True, "skipped": True}
        save_results(results)
        return results

    geo_dim = geo_feats.shape[1]
    geo_pair_diff = compute_pair_dock_diff(geo_feats, pairs_df)

    # Combined: geometric + interaction features for richer conditioning
    if has_interact:
        combined_feats = np.hstack([geo_feats, interact_feats])
        combined_pair_diff = compute_pair_dock_diff(combined_feats, pairs_df)
        combined_dim = combined_feats.shape[1]
    else:
        combined_feats = geo_feats
        combined_pair_diff = geo_pair_diff
        combined_dim = geo_dim

    print(f"  Geometric feature dim: {geo_dim}")
    print(f"  Combined feature dim: {combined_dim}")

    # ── Method: SE3_FiLM ──
    print("\n  SE3_FiLM (DockingFiLM conditioned on geometric features)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df,
            extra_feats_per_pair=geo_pair_diff,
            arch="docking_film", extra_dim=geo_dim,
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = geo_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["SE3_FiLM"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("SE3_FiLM", methods["SE3_FiLM"])

    # ── Method: SE3_DualStream ──
    print("\n  SE3_DualStream (DualStream with geometric features as 3D stream)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        predictor = train_film_on_pairs(
            train_idx, X_fp, y_all, pairs_df,
            extra_feats_per_pair=combined_pair_diff,
            arch="dual_stream", extra_dim=combined_dim,
        )
        if predictor is None:
            continue
        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = combined_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["SE3_DualStream"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("SE3_DualStream", methods["SE3_DualStream"])

    # ── Method: SE3_antisym ──
    print("\n  SE3_antisym (SE3_FiLM + antisymmetric augmentation)...")
    fold_delta = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        # Train with antisymmetric augmentation using DockingFiLMPredictor
        from src.models.predictors.docking_film_predictor import DockingFiLMPredictor

        torch.manual_seed(42)
        np.random.seed(42)

        train_smiles_idx_set = set(train_idx)
        mask = (
            pairs_df["idx_a"].isin(train_smiles_idx_set) &
            pairs_df["idx_b"].isin(train_smiles_idx_set)
        )
        train_pairs = pairs_df[mask]

        if len(train_pairs) == 0:
            continue

        emb_a_tr = X_fp[train_pairs["idx_a"].values]
        emb_b_tr = X_fp[train_pairs["idx_b"].values]
        delta_tr = train_pairs["delta"].values.astype(np.float32)
        dock_tr = geo_pair_diff[train_pairs.index.values]

        predictor = DockingFiLMPredictor(
            arch="docking_film", extra_dim=geo_dim,
            dropout=0.2, learning_rate=1e-3, batch_size=64,
            max_epochs=100, patience=15, device=DEVICE,
        )
        predictor.fit(
            emb_a_tr, emb_b_tr, dock_tr, delta_tr,
            verbose=False,
            antisymmetric_aug=True,
            antisym_reg_weight=0.1,
        )

        test_set = set(test_idx)
        test_pairs = pairs_df[
            pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
        ]
        if len(test_pairs) > 0:
            emb_a = X_fp[test_pairs["idx_a"].values]
            emb_b = X_fp[test_pairs["idx_b"].values]
            dock_diff = geo_pair_diff[test_pairs.index.values]
            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)
            delta_true = test_pairs["delta"].values.astype(np.float32)
            fold_delta.append(compute_delta_metrics(delta_true, delta_pred))

    if fold_delta:
        methods["SE3_antisym"] = {
            "delta_mae_mean": float(np.mean([m["mae"] for m in fold_delta])),
            "delta_mae_std": float(np.std([m["mae"] for m in fold_delta])),
            "delta_spearman_mean": float(np.mean([m["spearman"] for m in fold_delta])),
        }
        _print_method_result("SE3_antisym", methods["SE3_antisym"])

    results["phase_3_5"] = {
        "methods": methods,
        "geo_feature_dim": geo_dim,
        "combined_feature_dim": combined_dim,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print_comparison_table(methods, "Phase 3.5 Results: SE(3) Geometric + Antisymmetric")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Comprehensive Comparison (3 seeds)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_4(mol_data, X_fp, pairs_df, vina_feats, interact_feats, results):
    """Phase 4: Run all methods from phases 1-3 with 3 seeds."""
    print("\n" + "=" * 70)
    print("PHASE 4: Comprehensive Comparison (3 seeds)")
    print("=" * 70)

    phase = results.get("phase_4", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    seeds = [42, 123, 456]
    methods = {}

    has_vina = not np.all(np.isnan(vina_feats))
    has_interact = interact_feats is not None and interact_feats.shape[1] > 0

    vina_pair_diff = compute_pair_dock_diff(vina_feats, pairs_df)
    if has_interact:
        interact_pair_diff = compute_pair_dock_diff(interact_feats, pairs_df)
        interact_dim = interact_feats.shape[1]
        full_dock_feats = np.hstack([vina_feats, interact_feats])
        full_dock_pair_diff = compute_pair_dock_diff(full_dock_feats, pairs_df)
        full_dock_dim = full_dock_feats.shape[1]
    else:
        interact_pair_diff = None
        interact_dim = 0
        full_dock_feats = vina_feats
        full_dock_pair_diff = vina_pair_diff
        full_dock_dim = vina_feats.shape[1]

    # Define all method configs
    method_configs = [
        # (name, type, kwargs)
        ("XGB_subtraction", "xgb_abs", {"X_extra": None}),
        ("FiLMDelta", "film", {"arch": "standard", "extra_feats": None, "extra_dim": 0}),
    ]

    if has_vina:
        method_configs.extend([
            ("XGB_sub_vina", "xgb_abs", {"X_extra": vina_feats}),
            ("FiLM_vina", "film", {
                "arch": "docking_film", "extra_feats": vina_pair_diff, "extra_dim": 3,
            }),
            ("DualStream", "film", {
                "arch": "dual_stream", "extra_feats": full_dock_pair_diff,
                "extra_dim": full_dock_dim,
            }),
            ("Hierarchical", "film", {
                "arch": "hierarchical", "extra_feats": full_dock_pair_diff,
                "extra_dim": full_dock_dim,
            }),
        ])

    if has_interact:
        method_configs.extend([
            ("XGB_sub_full", "xgb_abs", {
                "X_extra": np.hstack([vina_feats, interact_feats]),
            }),
            ("FiLM_interact", "film", {
                "arch": "docking_film", "extra_feats": interact_pair_diff,
                "extra_dim": interact_dim,
            }),
            ("PoseCond", "film", {
                "arch": "docking_film", "extra_feats": interact_pair_diff,
                "extra_dim": interact_dim,
            }),
        ])

    for method_name, method_type, kwargs in method_configs:
        print(f"\n  Running {method_name} ({len(seeds)} seeds)...")
        all_delta_mae = []
        all_delta_spr = []
        all_abs_mae = []
        all_abs_spr = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

            for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
                test_set = set(test_idx)
                test_pairs = pairs_df[
                    pairs_df["idx_a"].isin(test_set) & pairs_df["idx_b"].isin(test_set)
                ]
                if len(test_pairs) == 0:
                    continue

                try:
                    if method_type == "xgb_abs":
                        X_extra = kwargs.get("X_extra")
                        if X_extra is not None:
                            X_train = np.hstack([X_fp[train_idx], X_extra[train_idx]])
                            X_test = np.hstack([X_fp[test_idx], X_extra[test_idx]])
                        else:
                            X_train = X_fp[train_idx]
                            X_test = X_fp[test_idx]

                        preds_abs, _ = train_xgboost(
                            X_train, y_all[train_idx], X_test, **BEST_XGB_PARAMS,
                        )
                        # Absolute metrics
                        a_met = compute_absolute_metrics(y_all[test_idx], preds_abs)
                        all_abs_mae.append(a_met["mae"])
                        all_abs_spr.append(a_met.get("spearman_r", 0))

                        # Delta from subtraction
                        abs_map = dict(zip(test_idx, preds_abs))
                        delta_true = test_pairs["delta"].values
                        delta_pred = np.array([
                            abs_map[b] - abs_map[a]
                            for a, b in zip(test_pairs["idx_a"], test_pairs["idx_b"])
                        ])
                        d_met = compute_delta_metrics(delta_true, delta_pred)
                        all_delta_mae.append(d_met["mae"])
                        all_delta_spr.append(d_met["spearman"])

                    elif method_type == "film":
                        arch = kwargs["arch"]
                        extra_feats = kwargs.get("extra_feats")
                        extra_dim = kwargs.get("extra_dim", 0)

                        predictor = train_film_on_pairs(
                            train_idx, X_fp, y_all, pairs_df,
                            extra_feats_per_pair=extra_feats,
                            arch=arch, extra_dim=extra_dim, seed=seed,
                        )
                        if predictor is None:
                            continue

                        emb_a = X_fp[test_pairs["idx_a"].values]
                        emb_b = X_fp[test_pairs["idx_b"].values]
                        delta_true = test_pairs["delta"].values.astype(np.float32)

                        if arch == "standard":
                            delta_pred = predictor.predict(emb_a, emb_b)
                        else:
                            dock_diff = extra_feats[test_pairs.index.values]
                            delta_pred = predictor.predict(emb_a, emb_b, dock_diff)

                        d_met = compute_delta_metrics(delta_true, delta_pred)
                        all_delta_mae.append(d_met["mae"])
                        all_delta_spr.append(d_met["spearman"])

                except Exception as e:
                    print(f"    {method_name} fold {fold_i} seed {seed}: {e}")
                    continue

        if all_delta_mae:
            methods[method_name] = {
                "delta_mae_mean": float(np.mean(all_delta_mae)),
                "delta_mae_std": float(np.std(all_delta_mae)),
                "delta_spearman_mean": float(np.mean(all_delta_spr)),
                "delta_spearman_std": float(np.std(all_delta_spr)),
                "n_runs": len(all_delta_mae),
            }
            if all_abs_mae:
                methods[method_name]["abs_mae_mean"] = float(np.mean(all_abs_mae))
                methods[method_name]["abs_mae_std"] = float(np.std(all_abs_mae))
                methods[method_name]["abs_spearman_mean"] = float(np.mean(all_abs_spr))
            _print_method_result(method_name, methods[method_name])

        gc.collect()

    results["phase_4"] = {
        "methods": methods,
        "seeds": seeds,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    print_comparison_table(methods, "Phase 4: Comprehensive Comparison (3 seeds)")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Ensemble & Application
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_5(mol_data, X_fp, pairs_df, vina_feats, interact_feats, results):
    """Phase 5: Stacking ensemble + apply to 509 generated molecules."""
    print("\n" + "=" * 70)
    print("PHASE 5: Ensemble & Application to Generated Molecules")
    print("=" * 70)

    phase = results.get("phase_5", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    y_all = mol_data["pIC50"].values
    all_smiles = mol_data["smiles"].tolist()
    has_vina = not np.all(np.isnan(vina_feats))
    has_interact = interact_feats is not None and interact_feats.shape[1] > 0
    methods = {}

    # ── 5A: Stacking ensemble of top 3 diverse models ──
    print("\n  5A. Stacking Ensemble (XGB_sub + FiLMDelta + best docking method)...")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_abs_ensemble = []

    if has_vina and has_interact:
        full_dock_feats = np.hstack([vina_feats, interact_feats])
        full_dock_dim = full_dock_feats.shape[1]
        full_dock_pair_diff = compute_pair_dock_diff(full_dock_feats, pairs_df)
    elif has_vina:
        full_dock_feats = vina_feats
        full_dock_dim = vina_feats.shape[1]
        full_dock_pair_diff = compute_pair_dock_diff(vina_feats, pairs_df)
    else:
        full_dock_feats = None
        full_dock_dim = 0
        full_dock_pair_diff = None

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        # Model 1: XGB subtraction (best absolute predictor)
        X_train_xgb = X_fp[train_idx]
        if has_vina:
            X_train_xgb = np.hstack([X_train_xgb, vina_feats[train_idx]])
        X_test_xgb = X_fp[test_idx]
        if has_vina:
            X_test_xgb = np.hstack([X_test_xgb, vina_feats[test_idx]])

        try:
            preds_xgb, _ = train_xgboost(
                X_train_xgb, y_all[train_idx], X_test_xgb, **BEST_XGB_PARAMS,
            )
        except Exception as e:
            print(f"    XGB failed fold {fold_i}: {e}")
            continue

        # Model 2: FiLMDelta (best delta predictor)
        try:
            film_predictor = train_film_on_pairs(
                train_idx, X_fp, y_all, pairs_df, arch="standard",
            )
        except Exception as e:
            print(f"    FiLMDelta failed fold {fold_i}: {e}")
            film_predictor = None

        # Model 3: DualStream if docking available
        dock_predictor = None
        if full_dock_pair_diff is not None:
            try:
                dock_predictor = train_film_on_pairs(
                    train_idx, X_fp, y_all, pairs_df,
                    extra_feats_per_pair=full_dock_pair_diff,
                    arch="dual_stream", extra_dim=full_dock_dim,
                )
            except Exception as e:
                print(f"    DualStream failed fold {fold_i}: {e}")

        # Reconstruct absolute predictions for each model via anchors
        n_anchors = min(50, len(train_idx))
        anchor_idx = np.random.RandomState(42).choice(
            train_idx, size=n_anchors, replace=False
        )

        preds_list = [preds_xgb]  # XGB already produces absolute

        if film_predictor is not None:
            film_abs = []
            for j in test_idx:
                anchors_preds = []
                for i in anchor_idx:
                    d = film_predictor.predict(X_fp[i:i+1], X_fp[j:j+1]).item()
                    anchors_preds.append(y_all[i] + d)
                film_abs.append(float(np.median(anchors_preds)))
            preds_list.append(np.array(film_abs))

        if dock_predictor is not None:
            dock_abs = []
            for j in test_idx:
                anchors_preds = []
                for i in anchor_idx:
                    dock_diff_ij = (full_dock_feats[j:j+1] - full_dock_feats[i:i+1])
                    d = dock_predictor.predict(
                        X_fp[i:i+1], X_fp[j:j+1], dock_diff_ij
                    ).item()
                    anchors_preds.append(y_all[i] + d)
                dock_abs.append(float(np.median(anchors_preds)))
            preds_list.append(np.array(dock_abs))

        # Simple average ensemble
        ensemble_pred = np.mean(preds_list, axis=0)
        fold_abs_ensemble.append(
            compute_absolute_metrics(y_all[test_idx], ensemble_pred)
        )

    if fold_abs_ensemble:
        agg = aggregate_cv_results(fold_abs_ensemble)
        methods["Ensemble_stacking"] = {
            "abs_mae_mean": agg.get("mae_mean", float('nan')),
            "abs_mae_std": agg.get("mae_std", 0),
            "abs_spearman_mean": agg.get("spearman_r_mean", 0),
            "n_models": len(preds_list),
        }
        print(f"    Ensemble: MAE={agg.get('mae_mean', 0):.4f}+-{agg.get('mae_std', 0):.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")

    # ── 5B: Apply to 509 generated molecules ──
    print("\n  5B. Applying to generated molecules...")
    gen_results_csv = DOCK_GEN_DIR / "docking_results.csv"

    if gen_results_csv.exists():
        gen_df = pd.read_csv(gen_results_csv)
        gen_docked = gen_df[gen_df.get("success", gen_df.columns[0]) == True] \
            if "success" in gen_df.columns else gen_df
        print(f"  Generated molecules: {len(gen_df)} total, "
              f"{len(gen_docked)} successfully docked")

        if "smiles" in gen_docked.columns and len(gen_docked) > 0:
            gen_smiles = gen_docked["smiles"].tolist()

            # Compute Morgan FPs for generated molecules
            gen_fps = compute_fingerprints(gen_smiles, "morgan", radius=2, n_bits=2048)

            # Train models on ALL ChEMBL data
            all_idx = np.arange(len(mol_data))

            # FiLMDelta predictions via anchor-based approach
            film_full = train_film_on_pairs(
                all_idx, X_fp, y_all, pairs_df, arch="standard",
            )

            if film_full is not None:
                n_anchors = min(50, len(mol_data))
                anchor_idx = np.random.RandomState(42).choice(
                    all_idx, size=n_anchors, replace=False
                )

                gen_preds_film = []
                for j in range(len(gen_smiles)):
                    anchors = []
                    for i in anchor_idx:
                        d = film_full.predict(X_fp[i:i+1], gen_fps[j:j+1]).item()
                        anchors.append(y_all[i] + d)
                    gen_preds_film.append(float(np.median(anchors)))

                gen_preds_film = np.array(gen_preds_film)
                print(f"    FiLMDelta pIC50 predictions: "
                      f"{gen_preds_film.mean():.2f}+-{gen_preds_film.std():.2f} "
                      f"(range: {gen_preds_film.min():.2f} - {gen_preds_film.max():.2f})")

                # Vina-only rankings
                if "vina_score" in gen_docked.columns:
                    vina_scores = gen_docked["vina_score"].values
                    valid_mask = ~np.isnan(vina_scores) & ~np.isnan(gen_preds_film)

                    if valid_mask.sum() > 10:
                        spr_film_vina, _ = spearmanr(
                            gen_preds_film[valid_mask],
                            -vina_scores[valid_mask],  # Negate: lower Vina = better
                        )
                        print(f"    FiLMDelta vs Vina ranking agreement: "
                              f"Spearman={spr_film_vina:.3f}")

                        methods["generated_mols"] = {
                            "n_molecules": len(gen_smiles),
                            "film_pred_mean": float(gen_preds_film.mean()),
                            "film_pred_std": float(gen_preds_film.std()),
                            "film_vina_spearman": float(spr_film_vina),
                        }
    else:
        print(f"  Generated molecule docking results not found at {gen_results_csv}")
        print(f"  Skipping application step.")

    results["phase_5"] = {
        "methods": methods,
        "completed": True,
        "timestamp": str(datetime.now()),
    }
    save_results(results)
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Utility
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Docking-Integrated ZAP70 Prediction Experiment"
    )
    parser.add_argument(
        "--phase", nargs="*", type=int, default=None,
        help="Phase(s) to run (1-5, 35 for 3.5). Default: all phases.",
    )
    args = parser.parse_args()

    phases_to_run = args.phase if args.phase else [1, 2, 3, 35, 4, 5]

    print("=" * 70)
    print("  Docking-Integrated ZAP70 Prediction Experiment")
    print(f"  Phases: {phases_to_run}")
    print(f"  Device: {DEVICE}")
    print(f"  Results: {RESULTS_FILE}")
    print("=" * 70)

    t_start = time.time()

    # Load data
    print("\n--- Loading Data ---")
    mol_data, dock_available = load_docking_data()
    n_mols = len(mol_data)
    print(f"  Molecules: {n_mols}")

    # Compute fingerprints
    print("\n--- Computing Morgan Fingerprints ---")
    all_smiles = mol_data["smiles"].tolist()
    X_fp = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    print(f"  Morgan FP shape: {X_fp.shape}")

    # Vina features
    print("\n--- Extracting Vina Features ---")
    vina_feats = get_vina_features_per_mol(mol_data)
    n_valid_vina = np.sum(~np.isnan(mol_data["vina_score"].values))
    print(f"  Vina features shape: {vina_feats.shape}, "
          f"valid: {n_valid_vina}/{n_mols}")

    # Interaction features
    print("\n--- Computing Interaction Features ---")
    interact_dict, interact_available = load_interaction_features(mol_data)
    if interact_available:
        interact_feats = get_interaction_features_matrix(mol_data, interact_dict)
        print(f"  Interaction features shape: {interact_feats.shape}")
    else:
        interact_feats = np.zeros((n_mols, 0), dtype=np.float32)
        print("  Interaction features not available.")

    # Geometric features (SE(3)-invariant)
    print("\n--- Computing SE(3) Geometric Features ---")
    geo_dict, geo_available, pocket_residues = load_geometric_features(mol_data)
    if geo_available:
        geo_feats = get_geometric_features_matrix(mol_data, geo_dict, pocket_residues)
        print(f"  Geometric features shape: {geo_feats.shape}")
    else:
        geo_feats = np.zeros((n_mols, 0), dtype=np.float32)
        pocket_residues = []
        print("  Geometric features not available.")

    # Generate all pairs
    print("\n--- Generating All Pairs ---")
    pairs_df = generate_all_pairs(mol_data)

    # Load existing results
    results = load_results()
    results["metadata"] = {
        "n_molecules": n_mols,
        "n_pairs": len(pairs_df),
        "dock_available": dock_available,
        "interact_available": interact_available,
        "geo_available": geo_available,
        "geo_feature_dim": geo_feats.shape[1] if geo_available else 0,
        "n_pocket_residues": len(pocket_residues),
        "start_time": str(datetime.now()),
    }

    # Run phases
    if 1 in phases_to_run:
        results = run_phase_1(mol_data, X_fp, pairs_df, vina_feats, results)

    if 2 in phases_to_run:
        results = run_phase_2(
            mol_data, X_fp, pairs_df, vina_feats, interact_feats, results
        )

    if 3 in phases_to_run:
        results = run_phase_3(
            mol_data, X_fp, pairs_df, vina_feats, interact_feats, results
        )

    if 35 in phases_to_run:
        results = run_phase_3_5(
            mol_data, X_fp, pairs_df, vina_feats, interact_feats,
            geo_feats, results
        )

    if 4 in phases_to_run:
        results = run_phase_4(
            mol_data, X_fp, pairs_df, vina_feats, interact_feats, results
        )

    if 5 in phases_to_run:
        results = run_phase_5(
            mol_data, X_fp, pairs_df, vina_feats, interact_feats, results
        )

    # Final summary
    total_time = time.time() - t_start
    results["metadata"]["total_time_s"] = total_time
    save_results(results)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE ({total_time/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_FILE}")

    # Print combined comparison from all phases
    all_methods = {}
    for phase_key in ["phase_1", "phase_2", "phase_3", "phase_3_5", "phase_4"]:
        if phase_key in results and "methods" in results[phase_key]:
            all_methods.update(results[phase_key]["methods"])
    if all_methods:
        print_comparison_table(all_methods, "All Methods Combined")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
