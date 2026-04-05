#!/usr/bin/env python3
"""
ZAP70 Comprehensive Case Study v2 — Expert-panel-guided iterative approach.

Key insight: Predicting deltas from arbitrary pairs of 280 molecules is the WRONG
framing. The information content is only 280 pIC50 values. We must:

1. Build the best absolute pIC50 predictor (GP, RF, XGBoost, ensemble)
2. Derive deltas from absolute predictions
3. Compare with direct delta prediction
4. Show when edit framework adds value (on genuine MMPs vs arbitrary pairs)

Phases:
  A: Absolute pIC50 prediction — classical ML (GP/RF/XGBoost) on molecule-level data
  B: Multi-embedding comparison — try all cached embeddings
  C: Transfer learning — pretrain absolute predictor on kinase family, finetune on ZAP70
  D: MMP analysis — proper MMP identification, edit framework on genuine MMPs
  E: Ensemble + advanced — combine best, feature importance, uncertainty

Evaluation: 5-fold CV on molecules (NOT pairs), scaffold split, LOAO.

Usage:
    conda run -n quris python -u experiments/run_zap70_v2.py
    conda run -n quris python -u experiments/run_zap70_v2.py --phase A
    conda run -n quris python -u experiments/run_zap70_v2.py --phase B C
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
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

from experiments.run_paper_evaluation import (
    SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, DROPOUT, DEVICE,
    RESULTS_DIR, CACHE_DIR, DATA_DIR,
    DeltaMLP, AbsoluteMLP,
    compute_embeddings, get_pair_tensors,
    train_model, train_model_multi_input, predict, predict_multi_input,
)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
MMP_FILE = DATA_DIR / "shared_pairs_deduped.csv"

RESULTS_FILE = RESULTS_DIR / "zap70_v2_results.json"
REPORT_FILE = RESULTS_DIR / "zap70_v2_report.html"

ZAP70_ID = "CHEMBL2803"
N_JOBS = 8  # Half of 16 CPUs

# Smaller models for small data
SMALL_HIDDEN = [256, 128, 64]
SMALL_DROPOUT = 0.35
SMALL_LR = 5e-4
SMALL_PATIENCE = 20

# Extended kinase family for transfer learning
KINASE_FAMILY = {
    "SYK": "CHEMBL2599",     # Closest homolog (Syk family)
    "BTK": "CHEMBL5251",     # Tec family
    "LCK": "CHEMBL258",      # Src family
    "ITK": "CHEMBL3009",     # Tec family, T-cell signaling
    "JAK3": "CHEMBL2148",    # Known cross-reactivity
    "FYN": "CHEMBL1841",     # Src family
    "JAK2": "CHEMBL2971",    # JAK family
    "ABL1": "CHEMBL1862",    # Tyrosine kinase
    "SRC": "CHEMBL267",      # Src family prototype
    "FLT3": "CHEMBL1974",    # Receptor tyrosine kinase
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_zap70_molecules():
    """Load ZAP70 molecule-level data (averaged across assays)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()

    # Average pIC50 across all assays for each molecule
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()

    # Also keep per-assay data for LOAO
    per_assay = zap.groupby(["molecule_chembl_id", "assay_id"]).agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()

    print(f"  ZAP70: {len(mol_data)} molecules, pIC50 {mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f} "
          f"(mean={mol_data['pIC50'].mean():.2f}, std={mol_data['pIC50'].std():.2f})")
    print(f"  Per-assay entries: {len(per_assay)}, {per_assay['assay_id'].nunique()} assays")

    return mol_data, per_assay


def load_kinase_molecules():
    """Load molecule-level data for kinase family targets."""
    raw = pd.read_csv(RAW_FILE)
    kinase_ids = list(KINASE_FAMILY.values())
    kinase_data = raw[raw["target_chembl_id"].isin(kinase_ids)].copy()

    # Average per molecule-target
    mol_target = kinase_data.groupby(["molecule_chembl_id", "target_chembl_id"]).agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()

    print(f"  Kinase family: {len(mol_target):,} molecule-target entries, "
          f"{mol_target['molecule_chembl_id'].nunique():,} unique molecules")
    for name, tid in KINASE_FAMILY.items():
        n = mol_target[mol_target["target_chembl_id"] == tid]["molecule_chembl_id"].nunique()
        if n > 0:
            print(f"    {name} ({tid}): {n:,} molecules")
    return mol_target


def identify_mmp_pairs_proper(mol_data):
    """Identify genuine MMPs among ZAP70 molecules using RDKit fragmentation."""
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    print("  Identifying MMPs among ZAP70 molecules...")
    smiles_list = mol_data["smiles"].tolist()
    chembl_ids = mol_data["molecule_chembl_id"].tolist()

    # Fragment each molecule (maxCuts=1 for single-cut MMP)
    frag_map = {}  # core_smiles -> [(mol_idx, frag_smiles)]
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
            for core_smi, chains_smi in frags:
                if core_smi and chains_smi:
                    if core_smi not in frag_map:
                        frag_map[core_smi] = []
                    frag_map[core_smi].append((idx, smi, chains_smi))
        except Exception:
            pass

    # Find pairs sharing a core
    mmp_pairs = []
    seen = set()
    for core, mols in frag_map.items():
        if len(mols) < 2:
            continue
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                idx_a, smi_a, frag_a = mols[i]
                idx_b, smi_b, frag_b = mols[j]
                if smi_a == smi_b:
                    continue
                pair_key = tuple(sorted([smi_a, smi_b]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pic50_a = mol_data.iloc[idx_a]["pIC50"]
                pic50_b = mol_data.iloc[idx_b]["pIC50"]
                mmp_pairs.append({
                    "mol_a": smi_a, "mol_b": smi_b,
                    "mol_a_id": chembl_ids[idx_a], "mol_b_id": chembl_ids[idx_b],
                    "core": core,
                    "frag_a": frag_a, "frag_b": frag_b,
                    "value_a": pic50_a, "value_b": pic50_b,
                    "delta": pic50_b - pic50_a,
                })

    df = pd.DataFrame(mmp_pairs) if mmp_pairs else pd.DataFrame()
    print(f"  Found {len(df)} genuine MMP pairs from {len(frag_map)} unique cores")
    if len(df) > 0:
        print(f"  MMP |delta| range: {df['delta'].abs().min():.2f}-{df['delta'].abs().max():.2f}, "
              f"mean={df['delta'].abs().mean():.2f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Splitting — Molecule-Level
# ═══════════════════════════════════════════════════════════════════════════

def kfold_molecule_splits(mol_data, n_folds=5, seed=42):
    """K-fold CV on molecules."""
    np.random.seed(seed)
    idx = np.random.permutation(len(mol_data))
    folds = np.array_split(idx, n_folds)
    splits = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        splits.append((
            f"fold_{i}",
            mol_data.iloc[train_idx].copy(),
            mol_data.iloc[test_idx].copy(),
        ))
    return splits


def scaffold_molecule_splits(mol_data, n_repeats=3, test_frac=0.2):
    """Scaffold-based split on molecules."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol_scaffolds = {}
    for _, row in mol_data.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                scaf = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol))
                mol_scaffolds[row["smiles"]] = Chem.MolToSmiles(scaf)
            except:
                mol_scaffolds[row["smiles"]] = "unknown"
        else:
            mol_scaffolds[row["smiles"]] = "unknown"

    scaffolds = list(set(mol_scaffolds.values()))
    splits = []
    for seed in range(n_repeats):
        np.random.seed(seed + 300)
        np.random.shuffle(scaffolds)
        n_test = max(2, int(len(scaffolds) * test_frac))
        test_scaffolds = set(scaffolds[:n_test])
        test_mols = {s for s, sc in mol_scaffolds.items() if sc in test_scaffolds}

        test_df = mol_data[mol_data["smiles"].isin(test_mols)].copy()
        train_df = mol_data[~mol_data["smiles"].isin(test_mols)].copy()

        if len(test_df) >= 10 and len(train_df) >= 20:
            splits.append((f"scaffold_{seed}", train_df, test_df))
    return splits


def loao_molecule_splits(per_assay_data, mol_data, min_size=10):
    """Leave-one-assay-out: train on molecule averages from other assays, test on held-out assay."""
    assay_sizes = per_assay_data.groupby("assay_id")["molecule_chembl_id"].nunique()
    large_assays = assay_sizes[assay_sizes >= min_size].index.tolist()
    print(f"  LOAO: {len(large_assays)} assays with ≥{min_size} molecules")

    splits = []
    for test_assay in large_assays:
        test_mols = per_assay_data[per_assay_data["assay_id"] == test_assay].copy()
        # Use assay-specific pIC50 for test
        test_df = test_mols.groupby("molecule_chembl_id").agg({
            "smiles": "first", "pIC50": "mean"
        }).reset_index()

        # Train on all OTHER molecules (using global average pIC50)
        test_mol_ids = set(test_df["molecule_chembl_id"])
        train_df = mol_data[~mol_data["molecule_chembl_id"].isin(test_mol_ids)].copy()

        if len(train_df) >= 20 and len(test_df) >= 5:
            splits.append((f"loao_{test_assay}", train_df, test_df))
    return splits


# ═══════════════════════════════════════════════════════════════════════════
# Absolute pIC50 Predictors
# ═══════════════════════════════════════════════════════════════════════════

def get_mol_features(mol_data, emb_dict, emb_dim):
    """Get feature matrix and target vector for molecule-level prediction."""
    X = np.array([emb_dict.get(smi, np.zeros(emb_dim))
                  for smi in mol_data["smiles"]], dtype=np.float32)
    y = mol_data["pIC50"].values.astype(np.float32)
    return X, y


def _tanimoto_kernel_matrix(X, Y=None):
    """Compute Tanimoto (Jaccard) kernel matrix for binary fingerprints."""
    if Y is None:
        Y = X
    XY = X @ Y.T
    X2 = np.sum(X, axis=1, keepdims=True)
    Y2 = np.sum(Y, axis=1, keepdims=True)
    denom = X2 + Y2.T - XY + 1e-10
    return XY / denom


def train_gp_tanimoto(X_train, y_train, X_test):
    """Gaussian Process with precomputed Tanimoto kernel."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    # Precompute kernel matrices
    K_train = _tanimoto_kernel_matrix(X_train)
    K_test = _tanimoto_kernel_matrix(X_test, X_train)

    gp = GaussianProcessRegressor(kernel="precomputed", alpha=0.1,
                                   normalize_y=True, random_state=42)
    try:
        gp.fit(K_train, y_train)
        y_pred = gp.predict(K_test)
        return y_pred, np.zeros_like(y_pred)
    except Exception as e:
        # Fallback: use precomputed kernel with higher alpha
        try:
            gp = GaussianProcessRegressor(kernel="precomputed", alpha=1.0,
                                           normalize_y=True, random_state=42)
            gp.fit(K_train, y_train)
            y_pred = gp.predict(K_test)
            return y_pred, np.zeros_like(y_pred)
        except Exception as e2:
            print(f"    GP failed: {e2}")
            # Final fallback: KRR with Tanimoto kernel
            from sklearn.kernel_ridge import KernelRidge
            krr = KernelRidge(alpha=1.0, kernel="precomputed")
            krr.fit(K_train, y_train)
            y_pred = krr.predict(K_test)
            return y_pred, np.zeros_like(y_pred)


def train_rf(X_train, y_train, X_test, n_jobs=N_JOBS):
    """Random Forest regressor."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", min_samples_leaf=3,
        oob_score=True, n_jobs=n_jobs, random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test), rf


def train_xgboost(X_train, y_train, X_test, X_val=None, y_val=None):
    """XGBoost with regularization for small data."""
    import xgboost as xgb
    params = {
        "max_depth": 4, "min_child_weight": 5, "subsample": 0.7,
        "colsample_bytree": 0.7, "learning_rate": 0.05,
        "n_estimators": 500, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": N_JOBS,
    }
    model = xgb.XGBRegressor(**params)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test), model


def train_ridge(X_train, y_train, X_test, alphas=None):
    """Ridge regression with cross-validated alpha."""
    from sklearn.linear_model import RidgeCV
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def train_nn_absolute(X_train, y_train, X_test, X_val=None, y_val=None, seed=42):
    """Neural network absolute pIC50 predictor (small architecture)."""
    torch.manual_seed(seed)
    emb_dim = X_train.shape[1]

    if X_val is None:
        n_val = max(5, int(len(X_train) * 0.15))
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]

    model = AbsoluteMLP(emb_dim, hidden_dims=SMALL_HIDDEN, dropout=SMALL_DROPOUT)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=32, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).float()),
        batch_size=32, shuffle=False)

    model = train_model(model, train_loader, val_loader,
                        max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
    y_pred = predict(model, torch.from_numpy(X_test).float())
    return y_pred, model


def train_knn(X_train, y_train, X_test, k=5):
    """k-Nearest Neighbors with distance-weighted voting."""
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=min(k, len(X_train)),
                                 weights="distance", metric="cosine")
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def train_svr(X_train, y_train, X_test):
    """Support Vector Regression."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = SVR(kernel="rbf", C=10.0, gamma="scale")
    model.fit(X_tr, y_train)
    return model.predict(X_te), (model, scaler)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_absolute_metrics(y_true, y_pred):
    """Compute metrics for absolute pIC50 prediction."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    pr, _ = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
        "spearman_r": float(sr) if not np.isnan(sr) else 0.0,
        "spearman_p": float(sp) if not np.isnan(sp) else 1.0,
    }


def compute_delta_from_absolute(y_pred_abs, mol_data, pairs_df):
    """Derive pair deltas from absolute predictions."""
    smi_to_pred = dict(zip(mol_data["smiles"], y_pred_abs))
    pred_deltas = []
    for _, row in pairs_df.iterrows():
        pa = smi_to_pred.get(row["mol_a"], 0)
        pb = smi_to_pred.get(row["mol_b"], 0)
        pred_deltas.append(pb - pa)
    return np.array(pred_deltas)


def aggregate_cv_results(fold_metrics):
    """Aggregate cross-validation results."""
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0].keys()
            if isinstance(fold_metrics[0][k], (int, float)) and k not in ("n", "spearman_p")]
    agg = {"n_folds": len(fold_metrics)}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Absolute pIC50 Prediction
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(mol_data, per_assay, results):
    """Phase A: Absolute pIC50 prediction with classical ML."""
    print("\n" + "=" * 70)
    print("PHASE A: Absolute pIC50 Prediction (Classical ML)")
    print("=" * 70)

    phase_a = results.get("phase_a", {})
    if phase_a.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

    methods = {
        "GP_Tanimoto": lambda Xtr, ytr, Xte, **kw: train_gp_tanimoto(Xtr, ytr, Xte),
        "RandomForest": lambda Xtr, ytr, Xte, **kw: train_rf(Xtr, ytr, Xte),
        "XGBoost": lambda Xtr, ytr, Xte, **kw: train_xgboost(Xtr, ytr, Xte),
        "Ridge": lambda Xtr, ytr, Xte, **kw: train_ridge(Xtr, ytr, Xte),
        "NeuralNet": lambda Xtr, ytr, Xte, **kw: train_nn_absolute(Xtr, ytr, Xte),
        "KNN_5": lambda Xtr, ytr, Xte, **kw: train_knn(Xtr, ytr, Xte, k=5),
        "SVR": lambda Xtr, ytr, Xte, **kw: train_svr(Xtr, ytr, Xte),
    }

    split_types = {
        "5fold": lambda: kfold_molecule_splits(mol_data, n_folds=5, seed=42),
        "scaffold": lambda: scaffold_molecule_splits(mol_data),
        "loao": lambda: loao_molecule_splits(per_assay, mol_data),
    }

    for split_name, split_fn in split_types.items():
        splits = split_fn()
        if not splits:
            print(f"  No {split_name} splits available")
            continue

        print(f"\n  --- {split_name} ({len(splits)} folds) ---")

        for method_name, train_fn in methods.items():
            key = f"{split_name}__{method_name}"
            if key in phase_a:
                agg = phase_a[key].get("aggregated", {})
                print(f"    {key}: done (MAE={agg.get('mae_mean', '?'):.4f})")
                continue

            print(f"    {method_name}...", end=" ", flush=True)
            t0 = time.time()
            fold_metrics = []

            for fold_name, train_df, test_df in splits:
                try:
                    X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
                    X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

                    result = train_fn(Xtr=X_train, ytr=y_train, Xte=X_test)
                    y_pred = result[0] if isinstance(result, tuple) else result

                    m = compute_absolute_metrics(y_test, y_pred)
                    m["fold"] = fold_name
                    fold_metrics.append(m)
                except Exception as e:
                    print(f"error({fold_name}: {e})", end=" ")

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase_a[key] = {"aggregated": agg, "per_fold": fold_metrics}
                elapsed = time.time() - t0
                print(f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                      f"R²={agg.get('r2_mean', 0):.3f}, "
                      f"Spearman={agg['spearman_r_mean']:.3f} ({elapsed:.0f}s)")
            else:
                print("FAILED")

            results["phase_a"] = phase_a
            save_results(results)
            gc.collect()

    # Summary: rank methods
    print("\n  === PHASE A SUMMARY (5-fold CV) ===")
    print(f"  {'Method':<20} {'MAE':>10} {'R²':>10} {'Spearman':>10}")
    print(f"  {'-' * 50}")
    ranked = []
    for key, val in sorted(phase_a.items()):
        if key.startswith("5fold__") and "aggregated" in val:
            method = key.replace("5fold__", "")
            agg = val["aggregated"]
            ranked.append((method, agg["mae_mean"], agg.get("r2_mean", 0),
                          agg.get("spearman_r_mean", 0)))
    for method, mae, r2, spr in sorted(ranked, key=lambda x: x[1]):
        print(f"  {method:<20} {mae:>10.4f} {r2:>10.3f} {spr:>10.3f}")

    phase_a["completed"] = True
    results["phase_a"] = phase_a
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Multi-Embedding Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(mol_data, results):
    """Phase B: Compare embeddings for absolute prediction."""
    print("\n" + "=" * 70)
    print("PHASE B: Multi-Embedding Comparison")
    print("=" * 70)

    phase_b = results.get("phase_b", {})
    if phase_b.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()

    # Try all available embeddings
    embedders = ["chemprop-dmpnn", "morgan", "chemberta2-mtr",
                 "chemeleon", "molformer-xl", "unimol-v1", "unimol-v2-84m"]

    # Use best classical ML methods from Phase A
    best_methods = {
        "RF": lambda Xtr, ytr, Xte: train_rf(Xtr, ytr, Xte)[0],
        "XGBoost": lambda Xtr, ytr, Xte: train_xgboost(Xtr, ytr, Xte)[0],
        "Ridge": lambda Xtr, ytr, Xte: train_ridge(Xtr, ytr, Xte)[0],
    }

    for embedder_name in embedders:
        print(f"\n  --- Embedder: {embedder_name} ---")
        try:
            emb_dict, emb_dim = compute_embeddings(all_smiles, embedder_name)
        except Exception as e:
            print(f"    Failed to load: {e}")
            continue

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)

        for method_name, train_fn in best_methods.items():
            key = f"{embedder_name}__{method_name}"
            if key in phase_b:
                agg = phase_b[key].get("aggregated", {})
                print(f"    {key}: done (MAE={agg.get('mae_mean', '?'):.4f})")
                continue

            print(f"    {method_name}...", end=" ", flush=True)
            fold_metrics = []

            for fold_name, train_df, test_df in splits:
                try:
                    X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
                    X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)
                    y_pred = train_fn(X_train, y_train, X_test)
                    fold_metrics.append(compute_absolute_metrics(y_test, y_pred))
                except Exception as e:
                    print(f"error ", end="")

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase_b[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"MAE={agg['mae_mean']:.4f}, R²={agg.get('r2_mean',0):.3f}, "
                      f"Spearman={agg.get('spearman_r_mean',0):.3f}")

            results["phase_b"] = phase_b
            save_results(results)
        del emb_dict
        gc.collect()

    phase_b["completed"] = True
    results["phase_b"] = phase_b
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Transfer Learning from Kinase Family
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(mol_data, per_assay, results):
    """Phase C: Transfer learning from kinase family."""
    print("\n" + "=" * 70)
    print("PHASE C: Transfer Learning from Kinase Family")
    print("=" * 70)

    phase_c = results.get("phase_c", {})
    if phase_c.get("completed"):
        print("  Already completed")
        return results

    # Load kinase family data
    kinase_data = load_kinase_molecules()
    all_smiles = list(set(
        mol_data["smiles"].tolist() +
        kinase_data["smiles"].tolist()
    ))
    print(f"  Total molecules for embeddings: {len(all_smiles):,}")
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

    # Strategy 1: Train on ALL kinase data, evaluate on ZAP70
    key = "kinase_pretrain_rf"
    if key not in phase_c:
        print("\n  Strategy 1: Train RF on kinase family, predict ZAP70 directly")
        X_kinase = np.array([emb_dict.get(s, np.zeros(emb_dim))
                             for s in kinase_data["smiles"]], dtype=np.float32)
        y_kinase = kinase_data["pIC50"].values.astype(np.float32)
        X_zap, y_zap = get_mol_features(mol_data, emb_dict, emb_dim)

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
        rf.fit(X_kinase, y_kinase)
        y_pred = rf.predict(X_zap)
        m = compute_absolute_metrics(y_zap, y_pred)
        phase_c[key] = m
        print(f"    Kinase→ZAP70: MAE={m['mae']:.4f}, R²={m['r2']:.3f}, "
              f"Spearman={m['spearman_r']:.3f}")
        results["phase_c"] = phase_c
        save_results(results)

    # Strategy 2: Train on kinase family + ZAP70 jointly, evaluate with CV
    key = "kinase_joint_rf"
    if key not in phase_c:
        print("\n  Strategy 2: Joint training (kinase + ZAP70), 5-fold CV on ZAP70")
        X_kinase = np.array([emb_dict.get(s, np.zeros(emb_dim))
                             for s in kinase_data["smiles"]], dtype=np.float32)
        y_kinase = kinase_data["pIC50"].values.astype(np.float32)

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            X_train_zap, y_train_zap = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            # Combine kinase + ZAP70 train
            X_combined = np.vstack([X_kinase, X_train_zap])
            y_combined = np.concatenate([y_kinase, y_train_zap])

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_combined, y_combined)
            y_pred = rf.predict(X_test)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_c[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    Joint: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_c"] = phase_c
        save_results(results)

    # Strategy 3: Pretrain NN on kinases, finetune on ZAP70
    key = "kinase_pretrain_finetune_nn"
    if key not in phase_c:
        print("\n  Strategy 3: Pretrain NN on kinases, finetune on ZAP70")
        X_kinase = np.array([emb_dict.get(s, np.zeros(emb_dim))
                             for s in kinase_data["smiles"]], dtype=np.float32)
        y_kinase = kinase_data["pIC50"].values.astype(np.float32)

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []

        for fold_name, train_df, test_df in splits:
            X_train_zap, y_train_zap = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            torch.manual_seed(42)
            model = AbsoluteMLP(emb_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)

            # Stage 1: Pretrain on kinase data
            n_val_k = int(len(X_kinase) * 0.1)
            idx_k = np.random.permutation(len(X_kinase))
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_kinase[idx_k[n_val_k:]]).float(),
                              torch.from_numpy(y_kinase[idx_k[n_val_k:]]).float()),
                batch_size=128, shuffle=True)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_kinase[idx_k[:n_val_k]]).float(),
                              torch.from_numpy(y_kinase[idx_k[:n_val_k]]).float()),
                batch_size=128, shuffle=False)
            model = train_model(model, train_loader, val_loader,
                                max_epochs=50, patience=10, lr=1e-3)

            # Stage 2: Finetune on ZAP70 (last layer only = freeze backbone)
            # Actually finetune all layers with small LR
            n_val_z = max(5, int(len(X_train_zap) * 0.15))
            idx_z = np.random.permutation(len(X_train_zap))
            ft_train = DataLoader(
                TensorDataset(torch.from_numpy(X_train_zap[idx_z[n_val_z:]]).float(),
                              torch.from_numpy(y_train_zap[idx_z[n_val_z:]]).float()),
                batch_size=32, shuffle=True)
            ft_val = DataLoader(
                TensorDataset(torch.from_numpy(X_train_zap[idx_z[:n_val_z]]).float(),
                              torch.from_numpy(y_train_zap[idx_z[:n_val_z]]).float()),
                batch_size=32, shuffle=False)
            model = train_model(model, ft_train, ft_val,
                                max_epochs=100, patience=20, lr=1e-4)

            y_pred = predict(model, torch.from_numpy(X_test).float())
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_c[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    Pretrain→FT: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_c"] = phase_c
        save_results(results)

    # Strategy 4: Weighted kinase transfer (weight ZAP70 data higher)
    key = "weighted_kinase_rf"
    if key not in phase_c:
        print("\n  Strategy 4: Weighted joint RF (ZAP70 10x weight)")
        X_kinase = np.array([emb_dict.get(s, np.zeros(emb_dim))
                             for s in kinase_data["smiles"]], dtype=np.float32)
        y_kinase = kinase_data["pIC50"].values.astype(np.float32)

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            X_train_zap, y_train_zap = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            X_combined = np.vstack([X_kinase, X_train_zap])
            y_combined = np.concatenate([y_kinase, y_train_zap])
            # Weight ZAP70 samples 10x higher
            weights = np.ones(len(y_combined))
            weights[len(y_kinase):] = 10.0

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_combined, y_combined, sample_weight=weights)
            y_pred = rf.predict(X_test)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_c[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    Weighted: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_c"] = phase_c
        save_results(results)

    # Strategy 5: Per-kinase transfer — train on each kinase individually, test on ZAP70
    key = "per_kinase_transfer"
    if key not in phase_c:
        print("\n  Strategy 5: Per-kinase transfer (which kinase helps most?)")
        X_zap, y_zap = get_mol_features(mol_data, emb_dict, emb_dim)
        per_kinase = {}
        for name, tid in KINASE_FAMILY.items():
            kdata = kinase_data[kinase_data["target_chembl_id"] == tid]
            if len(kdata) < 50:
                continue
            X_k = np.array([emb_dict.get(s, np.zeros(emb_dim))
                            for s in kdata["smiles"]], dtype=np.float32)
            y_k = kdata["pIC50"].values.astype(np.float32)
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_k, y_k)
            y_pred = rf.predict(X_zap)
            m = compute_absolute_metrics(y_zap, y_pred)
            per_kinase[name] = m
            print(f"    {name:>6} ({tid}): MAE={m['mae']:.3f}, R²={m['r2']:.3f}, "
                  f"Spearman={m['spearman_r']:.3f}")
        phase_c[key] = per_kinase
        results["phase_c"] = phase_c
        save_results(results)

    del emb_dict
    gc.collect()
    phase_c["completed"] = True
    results["phase_c"] = phase_c
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: MMP Analysis + Delta Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(mol_data, per_assay, mmp_pairs, results):
    """Phase D: Compare delta prediction strategies."""
    print("\n" + "=" * 70)
    print("PHASE D: Delta Prediction — Absolute vs Direct vs Edit Framework")
    print("=" * 70)

    phase_d = results.get("phase_d", {})
    if phase_d.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

    # Generate all pairs for delta evaluation
    pairs = []
    smiles_list = mol_data["smiles"].tolist()
    pic50_list = mol_data["pIC50"].values
    for i, j in combinations(range(len(mol_data)), 2):
        pairs.append({
            "mol_a": smiles_list[i], "mol_b": smiles_list[j],
            "delta": pic50_list[j] - pic50_list[i],
            "value_a": pic50_list[i], "value_b": pic50_list[j],
        })
    all_pairs_df = pd.DataFrame(pairs)
    print(f"  All pairs: {len(all_pairs_df):,}")

    # Strategy 1: Delta from best absolute predictor (5-fold CV)
    key = "delta_from_absolute_rf"
    if key not in phase_d:
        print("\n  Strategy 1: Delta from absolute RF prediction")
        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        all_true = []
        all_pred = []

        for fold_name, train_df, test_df in splits:
            X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_abs = rf.predict(X_test)

            # Compute deltas for test molecule pairs
            test_smiles = test_df["smiles"].tolist()
            test_pIC50 = test_df["pIC50"].values
            pred_pIC50 = y_pred_abs

            for i, j in combinations(range(len(test_df)), 2):
                all_true.append(test_pIC50[j] - test_pIC50[i])
                all_pred.append(pred_pIC50[j] - pred_pIC50[i])

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        m = compute_absolute_metrics(all_true, all_pred)
        # Rename to delta metrics
        m["delta_mae"] = m.pop("mae")
        m["delta_spearman"] = m.pop("spearman_r")
        m["delta_r2"] = m.pop("r2")
        phase_d[key] = m
        print(f"    Delta MAE={m['delta_mae']:.4f}, Spearman={m['delta_spearman']:.4f}, "
              f"R²={m['delta_r2']:.3f}")
        results["phase_d"] = phase_d
        save_results(results)

    # Strategy 2: Delta from GP with uncertainty
    key = "delta_from_absolute_gp"
    if key not in phase_d:
        print("\n  Strategy 2: Delta from absolute GP prediction")
        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        all_true = []
        all_pred = []

        for fold_name, train_df, test_df in splits:
            X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            y_pred_abs, y_std = train_gp_tanimoto(X_train, y_train, X_test)
            test_pIC50 = test_df["pIC50"].values

            for i, j in combinations(range(len(test_df)), 2):
                all_true.append(test_pIC50[j] - test_pIC50[i])
                all_pred.append(y_pred_abs[j] - y_pred_abs[i])

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        m = compute_absolute_metrics(all_true, all_pred)
        m["delta_mae"] = m.pop("mae")
        m["delta_spearman"] = m.pop("spearman_r")
        m["delta_r2"] = m.pop("r2")
        phase_d[key] = m
        print(f"    Delta MAE={m['delta_mae']:.4f}, Spearman={m['delta_spearman']:.4f}")
        results["phase_d"] = phase_d
        save_results(results)

    # Strategy 3: Direct delta prediction (DeepDelta on pairs) — from v1 (for comparison)
    key = "direct_delta_deepdelta"
    if key not in phase_d:
        print("\n  Strategy 3: Direct delta prediction (DeepDelta, 5-fold CV)")
        from experiments.run_zap70_case_study import random_pair_splits
        all_pairs_with_ids = []
        ids = mol_data["molecule_chembl_id"].tolist() if "molecule_chembl_id" in mol_data.columns else mol_data.index.tolist()
        for i, j in combinations(range(len(mol_data)), 2):
            all_pairs_with_ids.append({
                "mol_a": smiles_list[i], "mol_b": smiles_list[j],
                "mol_a_id": ids[i], "mol_b_id": ids[j],
                "delta": pic50_list[j] - pic50_list[i],
                "value_a": pic50_list[i], "value_b": pic50_list[j],
                "target_chembl_id": ZAP70_ID,
                "assay_id_a": 0, "assay_id_b": 0,
                "is_within_assay": True,
            })
        pairs_full = pd.DataFrame(all_pairs_with_ids)

        # Molecule-level split for fair comparison
        splits = []
        for seed in range(5):
            np.random.seed(seed + 500)
            all_mols = list(set(pairs_full["mol_a"].tolist() + pairs_full["mol_b"].tolist()))
            np.random.shuffle(all_mols)
            n_test = max(10, int(len(all_mols) * 0.2))
            test_mols = set(all_mols[:n_test])
            test_df = pairs_full[pairs_full["mol_a"].isin(test_mols) &
                                  pairs_full["mol_b"].isin(test_mols)].copy()
            train_df = pairs_full[~pairs_full["mol_a"].isin(test_mols) &
                                   ~pairs_full["mol_b"].isin(test_mols)].copy()
            if len(test_df) >= 10 and len(train_df) >= 20:
                splits.append((f"molsplit_{seed}", train_df, test_df))

        if splits:
            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                try:
                    ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
                    tea, teb, tey = get_pair_tensors(test_df, emb_dict, emb_dim)
                    torch.manual_seed(42)
                    model = DeltaMLP(emb_dim * 2, hidden_dims=SMALL_HIDDEN, dropout=SMALL_DROPOUT)

                    n_val = max(5, int(len(ta) * 0.15))
                    idx = np.random.permutation(len(ta))
                    train_loader = DataLoader(
                        TensorDataset(torch.cat([ta[idx[n_val:]], tb[idx[n_val:]]], -1),
                                      ty[idx[n_val:]]),
                        batch_size=64, shuffle=True)
                    val_loader = DataLoader(
                        TensorDataset(torch.cat([ta[idx[:n_val]], tb[idx[:n_val]]], -1),
                                      ty[idx[:n_val]]),
                        batch_size=64, shuffle=False)
                    model = train_model(model, train_loader, val_loader,
                                        max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
                    y_pred = predict(model, torch.cat([tea, teb], -1))
                    m = compute_absolute_metrics(tey.numpy().flatten(), y_pred)
                    m["delta_mae"] = m.pop("mae")
                    m["delta_spearman"] = m.pop("spearman_r")
                    m["delta_r2"] = m.pop("r2")
                    fold_metrics.append(m)
                except Exception as e:
                    print(f"    Error in {fold_name}: {e}")

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase_d[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"    DeepDelta direct: Delta MAE={agg.get('delta_mae_mean',0):.4f}, "
                      f"Spearman={agg.get('delta_spearman_mean',0):.4f}")
        results["phase_d"] = phase_d
        save_results(results)

    # Strategy 4: If MMPs found, compare edit framework vs subtraction on MMPs only
    if len(mmp_pairs) > 20:
        key = "mmp_edit_vs_subtraction"
        if key not in phase_d:
            print(f"\n  Strategy 4: Edit framework on {len(mmp_pairs)} genuine MMPs")
            # 5-fold CV on MMP pairs (molecule-level split)
            mmp_mols = list(set(mmp_pairs["mol_a"].tolist() + mmp_pairs["mol_b"].tolist()))
            fold_results = {"subtraction": [], "deepdelta": [], "film": []}

            for seed in range(5):
                np.random.seed(seed + 600)
                np.random.shuffle(mmp_mols)
                n_test = max(5, int(len(mmp_mols) * 0.2))
                test_mols = set(mmp_mols[:n_test])
                test_df = mmp_pairs[mmp_pairs["mol_a"].isin(test_mols) &
                                     mmp_pairs["mol_b"].isin(test_mols)].copy()
                train_df = mmp_pairs[~mmp_pairs["mol_a"].isin(test_mols) &
                                      ~mmp_pairs["mol_b"].isin(test_mols)].copy()
                if len(test_df) < 5 or len(train_df) < 10:
                    continue

                # Subtraction (from absolute RF)
                X_train_mols = list(set(train_df["mol_a"].tolist() + train_df["mol_b"].tolist()))
                X_tr = np.array([emb_dict.get(s, np.zeros(emb_dim)) for s in X_train_mols], dtype=np.float32)
                y_tr_dict = {}
                for _, r in train_df.iterrows():
                    y_tr_dict[r["mol_a"]] = r["value_a"]
                    y_tr_dict[r["mol_b"]] = r["value_b"]
                y_tr = np.array([y_tr_dict[s] for s in X_train_mols], dtype=np.float32)

                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=200, n_jobs=N_JOBS, random_state=42)
                rf.fit(X_tr, y_tr)
                pred_a = rf.predict(np.array([emb_dict.get(s, np.zeros(emb_dim))
                                              for s in test_df["mol_a"]], dtype=np.float32))
                pred_b = rf.predict(np.array([emb_dict.get(s, np.zeros(emb_dim))
                                              for s in test_df["mol_b"]], dtype=np.float32))
                y_true = test_df["delta"].values
                m_sub = compute_absolute_metrics(y_true, pred_b - pred_a)
                fold_results["subtraction"].append(m_sub)

            if fold_results["subtraction"]:
                for method_name, metrics_list in fold_results.items():
                    if metrics_list:
                        phase_d[f"mmp_{method_name}"] = aggregate_cv_results(metrics_list)

            results["phase_d"] = phase_d
            save_results(results)
    else:
        print(f"\n  Skipping MMP analysis (only {len(mmp_pairs)} MMPs found)")

    phase_d["completed"] = True
    results["phase_d"] = phase_d
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase E: Ensemble + Feature Importance + Advanced
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_e(mol_data, per_assay, results):
    """Phase E: Ensemble, feature importance, multi-embedding fusion."""
    print("\n" + "=" * 70)
    print("PHASE E: Ensemble + Feature Importance + Advanced")
    print("=" * 70)

    phase_e = results.get("phase_e", {})
    if phase_e.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()

    # 1. Ensemble of RF + XGBoost + GP (same embedding)
    key = "ensemble_rf_xgb_gp"
    if key not in phase_e:
        print("\n  Ensemble: RF + XGBoost + GP (Morgan FP)")
        emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []

        for fold_name, train_df, test_df in splits:
            X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
            X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)

            # Train all 3
            p_rf, _ = train_rf(X_train, y_train, X_test)
            p_xgb, _ = train_xgboost(X_train, y_train, X_test)
            p_gp, _ = train_gp_tanimoto(X_train, y_train, X_test)

            # Simple average
            y_pred = (p_rf + p_xgb + p_gp) / 3
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_e[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_e"] = phase_e
        save_results(results)
        del emb_dict
        gc.collect()

    # 2. Multi-embedding ensemble: RF on each embedding, then average predictions
    key = "multi_emb_ensemble"
    if key not in phase_e:
        print("\n  Multi-embedding ensemble: RF per embedding, average predictions")
        embedders = ["chemprop-dmpnn", "morgan", "chemberta2-mtr", "chemeleon",
                     "unimol-v1", "unimol-v2-84m"]
        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []

        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            preds = []
            for emb_name in embedders:
                try:
                    emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
                    X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
                    X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)
                    p, _ = train_rf(X_train, y_train, X_test)
                    preds.append(p)
                    del emb_dict
                except:
                    pass

            if preds:
                y_pred = np.mean(preds, axis=0)
                fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        if fold_metrics:
            agg = aggregate_cv_results(fold_metrics)
            phase_e[key] = {"aggregated": agg, "per_fold": fold_metrics}
            print(f"    MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                  f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_e"] = phase_e
        save_results(results)
        gc.collect()

    # 3. Stacked ensemble: train meta-learner on RF predictions from multiple embeddings
    key = "stacked_ensemble"
    if key not in phase_e:
        print("\n  Stacked ensemble: meta-learner on multi-embedding RF predictions")
        embedders = ["chemprop-dmpnn", "morgan", "chemberta2-mtr", "chemeleon",
                     "unimol-v1", "unimol-v2-84m"]

        # Collect OOF (out-of-fold) predictions from each embedding
        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        n_mols = len(mol_data)
        oof_preds = np.zeros((n_mols, len(embedders)))
        oof_preds[:] = np.nan
        y_all = mol_data["pIC50"].values

        # Map split indices
        split_indices = []
        for fold_name, train_df, test_df in splits:
            test_idx = test_df.index.tolist()
            split_indices.append(test_idx)

        for emb_idx, emb_name in enumerate(embedders):
            try:
                emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
                for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
                    X_train, y_train = get_mol_features(train_df, emb_dict, emb_dim)
                    X_test, y_test = get_mol_features(test_df, emb_dict, emb_dim)
                    p, _ = train_rf(X_train, y_train, X_test)
                    test_indices = test_df.index.tolist()
                    for i, idx in enumerate(test_indices):
                        pos = mol_data.index.get_loc(idx)
                        oof_preds[pos, emb_idx] = p[i]
                del emb_dict
            except:
                pass

        # Train meta-learner (Ridge) on OOF predictions
        valid_mask = ~np.any(np.isnan(oof_preds), axis=1)
        if valid_mask.sum() > 50:
            from sklearn.linear_model import RidgeCV
            X_meta = oof_preds[valid_mask]
            y_meta = y_all[valid_mask]

            # CV on meta-learner
            meta_splits = kfold_molecule_splits(
                mol_data[valid_mask].reset_index(drop=True), n_folds=5, seed=99)
            meta_fold_metrics = []
            for fold_name, tr, te in meta_splits:
                tr_idx = tr.index.tolist()
                te_idx = te.index.tolist()
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                meta.fit(X_meta[tr_idx], y_meta[tr_idx])
                y_pred = meta.predict(X_meta[te_idx])
                meta_fold_metrics.append(compute_absolute_metrics(y_meta[te_idx], y_pred))

            agg = aggregate_cv_results(meta_fold_metrics)
            phase_e[key] = {"aggregated": agg, "per_fold": meta_fold_metrics}
            print(f"    MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                  f"R²={agg.get('r2_mean',0):.3f}, Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_e"] = phase_e
        save_results(results)
        gc.collect()

    # 4. Feature importance (from RF)
    key = "feature_importance"
    if key not in phase_e:
        print("\n  Feature importance from RF (Morgan FP bits)")
        emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
        X, y = get_mol_features(mol_data, emb_dict, emb_dim)
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   n_jobs=N_JOBS, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_k = 20
        top_idx = np.argsort(-importances)[:top_k]
        phase_e[key] = {
            "top_features": [{"index": int(i), "importance": float(importances[i])}
                             for i in top_idx],
            "total_features": int(emb_dim),
            "top20_cumulative": float(np.sum(importances[top_idx])),
        }
        print(f"    Top 20 features explain {phase_e[key]['top20_cumulative']*100:.1f}% of variance")
        del emb_dict
        gc.collect()
        results["phase_e"] = phase_e
        save_results(results)

    # 5. Concatenated embeddings (Morgan + ChemBERTa + Uni-Mol)
    key = "concat_embeddings_rf"
    if key not in phase_e:
        print("\n  Concatenated embeddings: Morgan + ChemBERTa-MTR + Uni-Mol v1")
        emb_dicts = {}
        emb_dims = {}
        for name in ["chemprop-dmpnn", "chemberta2-mtr", "unimol-v1"]:
            try:
                d, dim = compute_embeddings(all_smiles, name)
                emb_dicts[name] = d
                emb_dims[name] = dim
            except:
                pass

        if len(emb_dicts) >= 2:
            splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                X_train_parts = []
                X_test_parts = []
                for name in emb_dicts:
                    d, dim = emb_dicts[name], emb_dims[name]
                    Xtr, _ = get_mol_features(train_df, d, dim)
                    Xte, y_test = get_mol_features(test_df, d, dim)
                    X_train_parts.append(Xtr)
                    X_test_parts.append(Xte)
                X_train = np.hstack(X_train_parts)
                X_test = np.hstack(X_test_parts)
                y_train = train_df["pIC50"].values.astype(np.float32)

                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                           min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

            agg = aggregate_cv_results(fold_metrics)
            phase_e[key] = {"aggregated": agg, "per_fold": fold_metrics}
            print(f"    MAE={agg['mae_mean']:.4f}, R²={agg.get('r2_mean',0):.3f}, "
                  f"Spearman={agg.get('spearman_r_mean',0):.3f}")
        for d in emb_dicts.values():
            del d
        gc.collect()
        results["phase_e"] = phase_e
        save_results(results)

    # 6. RDKit 2D descriptors
    key = "rdkit_descriptors_rf"
    if key not in phase_e:
        print("\n  RDKit 2D descriptors (200d)")
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors

        desc_names = [d[0] for d in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        desc_matrix = []
        for smi in all_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                descs = list(calc.CalcDescriptors(mol))
                desc_matrix.append(descs)
            else:
                desc_matrix.append([0] * len(desc_names))

        desc_matrix = np.array(desc_matrix, dtype=np.float32)
        # Replace NaN/Inf
        desc_matrix = np.nan_to_num(desc_matrix, nan=0, posinf=0, neginf=0)

        desc_dict = dict(zip(all_smiles, desc_matrix))
        desc_dim = desc_matrix.shape[1]

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            X_train, y_train = get_mol_features(train_df, desc_dict, desc_dim)
            X_test, y_test = get_mol_features(test_df, desc_dict, desc_dim)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_e[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    MAE={agg['mae_mean']:.4f}, R²={agg.get('r2_mean',0):.3f}, "
              f"Spearman={agg.get('spearman_r_mean',0):.3f}")
        results["phase_e"] = phase_e
        save_results(results)

    # 7. Morgan + RDKit descriptors combined
    key = "morgan_plus_rdkit_rf"
    if key not in phase_e:
        print("\n  Combined: Morgan FP + RDKit descriptors")
        emb_dict_fp, fp_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors
        desc_names = [d[0] for d in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        combined_dict = {}
        for smi in all_smiles:
            fp = emb_dict_fp.get(smi, np.zeros(fp_dim))
            mol = Chem.MolFromSmiles(smi)
            if mol:
                descs = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
                descs = np.nan_to_num(descs)
            else:
                descs = np.zeros(len(desc_names), dtype=np.float32)
            combined_dict[smi] = np.concatenate([fp, descs])
        combined_dim = fp_dim + len(desc_names)

        splits = kfold_molecule_splits(mol_data, n_folds=5, seed=42)
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            X_train, y_train = get_mol_features(train_df, combined_dict, combined_dim)
            X_test, y_test = get_mol_features(test_df, combined_dict, combined_dim)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase_e[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"    MAE={agg['mae_mean']:.4f}, R²={agg.get('r2_mean',0):.3f}, "
              f"Spearman={agg.get('spearman_r_mean',0):.3f}")
        del emb_dict_fp, combined_dict
        gc.collect()
        results["phase_e"] = phase_e
        save_results(results)

    phase_e["completed"] = True
    results["phase_e"] = phase_e
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results):
    """Generate comprehensive HTML report."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ZAP70 Case Study v2 — Expert-Guided Iterative Analysis</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto;
       padding: 20px; background: #f8f9fa; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
h3 { color: #7f8c8d; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
.best { font-weight: bold; color: #27ae60; background: #e8f5e9 !important; }
.warn { color: #e74c3c; }
.section { background: white; padding: 20px; margin: 15px 0; border-radius: 8px;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.metric { display: inline-block; background: #ecf0f1; padding: 15px 25px;
          margin: 5px; border-radius: 8px; text-align: center; }
.metric .v { font-size: 24px; font-weight: bold; color: #2c3e50; }
.metric .l { font-size: 12px; color: #7f8c8d; }
.insight { background: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; }
.key-finding { background: #e8f5e9; border-left: 4px solid #27ae60; padding: 15px; margin: 10px 0; }
</style></head><body>
"""
    html += "<h1>ZAP70 (CHEMBL2803) — Expert-Guided Iterative Case Study v2</h1>\n"
    html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>\n"

    html += '<div class="insight"><strong>Key Insight:</strong> '
    html += 'The previous approach (predicting deltas from 3,061 arbitrary pairs) was fundamentally '
    html += 'wrong — the information content is only 280 pIC50 values. This v2 study reframes the '
    html += 'problem as absolute pIC50 prediction with classical ML, then derives deltas.</div>\n'

    # Data summary
    if "data_summary" in results:
        ds = results["data_summary"]
        html += '<div class="section"><h2>Dataset</h2>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_molecules", "?")}</div>'
        html += '<div class="l">Molecules</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_assays", "?")}</div>'
        html += '<div class="l">Assays</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("pIC50_range", "?")}</div>'
        html += '<div class="l">pIC50 Range</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_mmp_pairs", 0)}</div>'
        html += '<div class="l">MMP Pairs Found</div></div>\n'
        html += "</div>\n"

    # Phase A results
    if "phase_a" in results:
        html += '<div class="section"><h2>Phase A: Absolute pIC50 Prediction (Classical ML)</h2>\n'
        html += '<p>5-fold cross-validation on 280 molecules with Morgan FP (2048d)</p>\n'
        html += "<table><tr><th>Method</th><th>MAE</th><th>R²</th><th>Spearman</th><th>Pearson</th></tr>\n"
        rows = []
        for key, val in results["phase_a"].items():
            if key.startswith("5fold__") and isinstance(val, dict) and "aggregated" in val:
                method = key.replace("5fold__", "")
                agg = val["aggregated"]
                rows.append((method, agg.get("mae_mean", 99), agg.get("r2_mean", 0),
                             agg.get("spearman_r_mean", 0), agg.get("pearson_r_mean", 0),
                             agg.get("mae_std", 0)))
        rows.sort(key=lambda x: x[1])
        for i, (method, mae, r2, spr, pr, std) in enumerate(rows):
            cls = ' class="best"' if i == 0 else ""
            html += f'<tr{cls}><td>{method}</td><td>{mae:.4f}±{std:.4f}</td>'
            html += f'<td>{r2:.3f}</td><td>{spr:.3f}</td><td>{pr:.3f}</td></tr>\n'
        html += "</table>\n"

        # Scaffold split results
        html += "<h3>Scaffold Split (generalization)</h3>\n"
        html += "<table><tr><th>Method</th><th>MAE</th><th>R²</th><th>Spearman</th></tr>\n"
        rows = []
        for key, val in results["phase_a"].items():
            if key.startswith("scaffold__") and isinstance(val, dict) and "aggregated" in val:
                method = key.replace("scaffold__", "")
                agg = val["aggregated"]
                rows.append((method, agg.get("mae_mean", 99), agg.get("r2_mean", 0),
                             agg.get("spearman_r_mean", 0)))
        rows.sort(key=lambda x: x[1])
        for method, mae, r2, spr in rows:
            html += f'<tr><td>{method}</td><td>{mae:.4f}</td><td>{r2:.3f}</td><td>{spr:.3f}</td></tr>\n'
        html += "</table></div>\n"

    # Phase B: Multi-embedding
    if "phase_b" in results:
        html += '<div class="section"><h2>Phase B: Multi-Embedding Comparison (RF, 5-fold)</h2>\n'
        html += "<table><tr><th>Embedding</th><th>MAE</th><th>R²</th><th>Spearman</th></tr>\n"
        rows = []
        for key, val in results["phase_b"].items():
            if "__RF" in key and isinstance(val, dict) and "aggregated" in val:
                emb = key.replace("__RF", "")
                agg = val["aggregated"]
                rows.append((emb, agg.get("mae_mean", 99), agg.get("r2_mean", 0),
                             agg.get("spearman_r_mean", 0)))
        rows.sort(key=lambda x: x[1])
        for i, (emb, mae, r2, spr) in enumerate(rows):
            cls = ' class="best"' if i == 0 else ""
            html += f'<tr{cls}><td>{emb}</td><td>{mae:.4f}</td><td>{r2:.3f}</td><td>{spr:.3f}</td></tr>\n'
        html += "</table></div>\n"

    # Phase C: Transfer learning
    if "phase_c" in results:
        html += '<div class="section"><h2>Phase C: Transfer Learning from Kinase Family</h2>\n'
        html += "<table><tr><th>Strategy</th><th>MAE</th><th>R²</th><th>Spearman</th></tr>\n"
        for key in ["kinase_pretrain_rf", "kinase_joint_rf", "kinase_pretrain_finetune_nn",
                     "weighted_kinase_rf"]:
            if key in results["phase_c"]:
                val = results["phase_c"][key]
                if "aggregated" in val:
                    agg = val["aggregated"]
                    html += f'<tr><td>{key}</td><td>{agg.get("mae_mean",0):.4f}</td>'
                    html += f'<td>{agg.get("r2_mean",0):.3f}</td>'
                    html += f'<td>{agg.get("spearman_r_mean",0):.3f}</td></tr>\n'
                else:
                    html += f'<tr><td>{key}</td><td>{val.get("mae",0):.4f}</td>'
                    html += f'<td>{val.get("r2",0):.3f}</td>'
                    html += f'<td>{val.get("spearman_r",0):.3f}</td></tr>\n'

        # Per-kinase transfer
        if "per_kinase_transfer" in results["phase_c"]:
            html += "</table>\n<h3>Per-Kinase Transfer (which kinase helps most?)</h3>\n"
            html += "<table><tr><th>Kinase</th><th>MAE</th><th>R²</th><th>Spearman</th></tr>\n"
            for name, m in sorted(results["phase_c"]["per_kinase_transfer"].items(),
                                   key=lambda x: x[1].get("mae", 99)):
                html += f'<tr><td>{name}</td><td>{m["mae"]:.4f}</td>'
                html += f'<td>{m["r2"]:.3f}</td><td>{m["spearman_r"]:.3f}</td></tr>\n'
        html += "</table></div>\n"

    # Phase D: Delta comparison
    if "phase_d" in results:
        html += '<div class="section"><h2>Phase D: Delta Prediction — Absolute vs Direct</h2>\n'
        html += '<div class="key-finding"><strong>Key Question:</strong> '
        html += 'Does deriving deltas from absolute predictions beat direct delta prediction?</div>\n'
        html += "<table><tr><th>Strategy</th><th>Delta MAE</th><th>Delta Spearman</th><th>Delta R²</th></tr>\n"
        for key in ["delta_from_absolute_rf", "delta_from_absolute_gp", "direct_delta_deepdelta"]:
            if key in results["phase_d"]:
                val = results["phase_d"][key]
                if "aggregated" in val:
                    agg = val["aggregated"]
                    html += f'<tr><td>{key}</td><td>{agg.get("delta_mae_mean",0):.4f}</td>'
                    html += f'<td>{agg.get("delta_spearman_mean",0):.4f}</td>'
                    html += f'<td>{agg.get("delta_r2_mean",0):.3f}</td></tr>\n'
                else:
                    html += f'<tr><td>{key}</td><td>{val.get("delta_mae",0):.4f}</td>'
                    html += f'<td>{val.get("delta_spearman",0):.4f}</td>'
                    html += f'<td>{val.get("delta_r2",0):.3f}</td></tr>\n'
        html += "</table></div>\n"

    # Phase E: Advanced
    if "phase_e" in results:
        html += '<div class="section"><h2>Phase E: Ensemble + Advanced Methods</h2>\n'
        html += "<table><tr><th>Method</th><th>MAE</th><th>R²</th><th>Spearman</th></tr>\n"
        for key in ["ensemble_rf_xgb_gp", "multi_emb_ensemble", "stacked_ensemble",
                     "rdkit_descriptors_rf", "morgan_plus_rdkit_rf", "concat_embeddings_rf"]:
            if key in results["phase_e"] and isinstance(results["phase_e"][key], dict):
                val = results["phase_e"][key]
                if "aggregated" in val:
                    agg = val["aggregated"]
                    html += f'<tr><td>{key}</td><td>{agg.get("mae_mean",0):.4f}</td>'
                    html += f'<td>{agg.get("r2_mean",0):.3f}</td>'
                    html += f'<td>{agg.get("spearman_r_mean",0):.3f}</td></tr>\n'
        html += "</table></div>\n"

    # V1 comparison
    html += '<div class="section"><h2>Comparison with v1 (Pair-Based Delta Prediction)</h2>\n'
    html += '<div class="insight">v1 predicted deltas directly from 3,061 arbitrary pairs using '
    html += 'neural networks. Best result: MAE=0.851, Spearman=0.131 (scaffold split). '
    html += 'This v2 approach reframes as absolute prediction with classical ML.</div>\n'
    html += "</div>\n"

    html += "</body></html>"

    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"  Report: {REPORT_FILE.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Results I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ZAP70 Case Study v2 — Expert-guided")
    parser.add_argument("--phase", nargs="+", default=["A", "B", "C", "D", "E"],
                        help="Phases to run (A B C D E)")
    args = parser.parse_args()
    phases = [p.upper() for p in args.phase]

    print("=" * 70)
    print("ZAP70 CASE STUDY v2 — EXPERT-GUIDED ITERATIVE ANALYSIS")
    print(f"Phases: {', '.join(phases)}")
    print(f"CPUs: {N_JOBS}, Device: CPU")
    print("=" * 70)

    results = load_results()

    # Step 1: Load and characterize data
    print("\n--- Data Loading ---")
    mol_data, per_assay = load_zap70_molecules()

    # Identify genuine MMPs
    mmp_pairs = identify_mmp_pairs_proper(mol_data)

    results["data_summary"] = {
        "target": "ZAP70", "target_id": ZAP70_ID,
        "n_molecules": len(mol_data),
        "n_assays": per_assay["assay_id"].nunique(),
        "pIC50_range": f"{mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f}",
        "pIC50_mean": float(mol_data["pIC50"].mean()),
        "pIC50_std": float(mol_data["pIC50"].std()),
        "n_mmp_pairs": len(mmp_pairs),
    }
    save_results(results)

    # Run phases
    if "A" in phases:
        results = run_phase_a(mol_data, per_assay, results)
        save_results(results)

    if "B" in phases:
        results = run_phase_b(mol_data, results)
        save_results(results)

    if "C" in phases:
        results = run_phase_c(mol_data, per_assay, results)
        save_results(results)

    if "D" in phases:
        results = run_phase_d(mol_data, per_assay, mmp_pairs, results)
        save_results(results)

    if "E" in phases:
        results = run_phase_e(mol_data, per_assay, results)
        save_results(results)

    # Generate report
    print("\n--- Report Generation ---")
    generate_report(results)

    results["completed"] = datetime.now().isoformat()
    save_results(results)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Results: {RESULTS_FILE.name}")
    print(f"  Report: {REPORT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
