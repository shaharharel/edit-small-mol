#!/usr/bin/env python3
"""
ZAP70 Case Study v3 — Deep iterative optimization.

Building on v2 (MAE=0.603, Spr=0.698), this script pushes performance via:

Phase 0: Diagnostics — fold analysis, residual analysis, label noise estimation
Phase 1: Fingerprint diversity — 10+ FP types screened individually
Phase 2: Proper pretrained embeddings — compute ChemBERTa/MoLFormer/UniMol for ZAP70
Phase 3: Feature selection — Boruta on Morgan, curated descriptor sets
Phase 4: Hyperparameter optimization — Optuna for XGBoost/RF
Phase 5: Advanced ensembles — diverse FPs, optimized weights, stacking
Phase 6: 3D descriptors — conformer-based shape/pharmacophore features
Phase 7: Conformal prediction — uncertainty quantification

Usage:
    conda run -n quris python -u experiments/run_zap70_v3.py
    conda run -n quris python -u experiments/run_zap70_v3.py --phase 0 1 2
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
os.environ['RDK_DEPRECATION_WARNING'] = 'off'  # Suppress RDKit deprecation warnings
torch.backends.mps.is_available = lambda: False

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import (
    RESULTS_DIR, CACHE_DIR, DATA_DIR,
    compute_embeddings,
)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
RESULTS_FILE = RESULTS_DIR / "zap70_v3_results.json"
REPORT_FILE = RESULTS_DIR / "zap70_v3_report.html"
ZAP70_ID = "CHEMBL2803"
N_JOBS = 8
N_FOLDS = 5
CV_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading & Splitting (shared with v2)
# ═══════════════════════════════════════════════════════════════════════════

def load_zap70_molecules():
    """Load ZAP70 molecule-level data (averaged across assays)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    per_assay = zap.groupby(["molecule_chembl_id", "assay_id"]).agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  ZAP70: {len(mol_data)} molecules, pIC50 {mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f} "
          f"(mean={mol_data['pIC50'].mean():.2f}, std={mol_data['pIC50'].std():.2f})")
    print(f"  Per-assay entries: {len(per_assay)}, {per_assay['assay_id'].nunique()} assays")
    return mol_data, per_assay


def get_cv_splits(mol_data, n_folds=N_FOLDS, seed=CV_SEED):
    """Fixed K-fold CV splits (same across all experiments)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for i, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
        splits.append((
            f"fold_{i}",
            mol_data.iloc[train_idx].copy(),
            mol_data.iloc[test_idx].copy(),
        ))
    return splits


def compute_absolute_metrics(y_true, y_pred):
    """Compute metrics for absolute pIC50 prediction."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    sr, sp = spearmanr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    return {
        "n": len(y_true),
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
        "spearman_r": float(sr) if not np.isnan(sr) else 0.0,
        "spearman_p": float(sp) if not np.isnan(sp) else 1.0,
    }


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


def save_results(results):
    """Save results to JSON file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _tanimoto_kernel_matrix(X, Y=None):
    """Tanimoto kernel for binary fingerprints."""
    if Y is None:
        Y = X
    XY = X @ Y.T
    X2 = np.sum(X, axis=1, keepdims=True)
    Y2 = np.sum(Y, axis=1, keepdims=True)
    denom = X2 + Y2.T - XY + 1e-10
    return XY / denom


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint & Feature Computing
# ═══════════════════════════════════════════════════════════════════════════

def compute_fingerprints(smiles_list, fp_type, radius=2, n_bits=2048, use_features=False):
    """Compute fingerprints for a list of SMILES. Returns (X, dim)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
    from rdkit.Chem import rdMolDescriptors

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if fp_type == "maccs":
                results.append(np.zeros(167, dtype=np.float32))
            else:
                results.append(np.zeros(n_bits, dtype=np.float32))
            continue

        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits, useFeatures=use_features)
        elif fp_type == "morgan_count":
            fp_obj = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float32)
            for idx, val in fp_obj.GetNonzeroElements().items():
                arr[idx % n_bits] = val
            results.append(arr)
            continue
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        elif fp_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == "atompair":
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        elif fp_type == "torsion":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        elif fp_type == "avalon":
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        elif fp_type == "pattern":
            fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
        elif fp_type == "layered":
            fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"Unknown fp_type: {fp_type}")

        arr = np.zeros(len(fp), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)

    X = np.array(results, dtype=np.float32)
    return X


def compute_rdkit_descriptors(smiles_list):
    """Compute all RDKit 2D descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    desc_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(len(desc_names), dtype=np.float32))
            continue
        try:
            vals = calc.CalcDescriptors(mol)
            arr = np.array(vals, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(arr)
        except Exception:
            results.append(np.zeros(len(desc_names), dtype=np.float32))

    X = np.array(results, dtype=np.float32)
    # Remove zero-variance columns
    variances = np.var(X, axis=0)
    good_cols = variances > 1e-10
    X = X[:, good_cols]
    desc_names_filtered = [n for n, g in zip(desc_names, good_cols) if g]
    return X, desc_names_filtered


def compute_3d_descriptors(smiles_list, n_confs=10):
    """Compute 3D descriptors from generated conformers."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(12, dtype=np.float32))
            continue

        try:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

            if len(cids) == 0:
                # Fallback: try with more flexibility
                params.useRandomCoords = True
                cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

            if len(cids) == 0:
                results.append(np.zeros(12, dtype=np.float32))
                continue

            # Optimize best conformer
            energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
            if energies:
                best_conf = min(range(len(energies)), key=lambda i: energies[i][1] if energies[i][0] == 0 else 1e10)
            else:
                best_conf = 0

            mol_noH = Chem.RemoveHs(mol)

            # 3D descriptors
            feats = []
            try:
                # PMI ratios (rod/disc/sphere)
                pmi1 = Descriptors3D.PMI1(mol, confId=cids[best_conf])
                pmi2 = Descriptors3D.PMI2(mol, confId=cids[best_conf])
                pmi3 = Descriptors3D.PMI3(mol, confId=cids[best_conf])
                feats.extend([pmi1, pmi2, pmi3])
                # NPR (normalized PMI ratios)
                npr1 = Descriptors3D.NPR1(mol, confId=cids[best_conf])
                npr2 = Descriptors3D.NPR2(mol, confId=cids[best_conf])
                feats.extend([npr1, npr2])
            except Exception:
                feats.extend([0.0] * 5)

            try:
                feats.append(Descriptors3D.Asphericity(mol, confId=cids[best_conf]))
                feats.append(Descriptors3D.Eccentricity(mol, confId=cids[best_conf]))
                feats.append(Descriptors3D.InertialShapeFactor(mol, confId=cids[best_conf]))
                feats.append(Descriptors3D.RadiusOfGyration(mol, confId=cids[best_conf]))
                feats.append(Descriptors3D.SpherocityIndex(mol, confId=cids[best_conf]))
            except Exception:
                feats.extend([0.0] * 5)

            try:
                # Plane of Best Fit
                feats.append(rdMolDescriptors.CalcPBF(mol, confId=cids[best_conf]))
            except Exception:
                feats.append(0.0)

            try:
                # Autocorrelation 3D
                feats.append(Descriptors3D.GetUSR(mol, confId=cids[best_conf])[0]
                             if hasattr(Descriptors3D, 'GetUSR') else 0.0)
            except Exception:
                feats.append(0.0)

            arr = np.array(feats[:12], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(arr)
        except Exception:
            results.append(np.zeros(12, dtype=np.float32))

    X = np.array(results, dtype=np.float32)
    return X


def compute_pretrained_embeddings(smiles_list, embedder_name):
    """Compute pretrained embeddings fresh (not from cache)."""
    print(f"    Computing {embedder_name} embeddings for {len(smiles_list)} molecules...")
    t0 = time.time()

    if embedder_name == "chemberta2-mtr":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name='chemberta2-mtr', device='cpu', batch_size=32)
        embs = embedder.encode(smiles_list)
        dim = embs.shape[1]
    elif embedder_name == "chemberta2-mlm":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name='chemberta2-mlm', device='cpu', batch_size=32)
        embs = embedder.encode(smiles_list)
        dim = embs.shape[1]
    elif embedder_name == "molformer-xl":
        from src.embedding.molformer import MoLFormerEmbedder
        embedder = MoLFormerEmbedder(device='cpu', batch_size=32)
        embs = embedder.encode(smiles_list)
        dim = embs.shape[1]
    elif embedder_name == "unimol-v1":
        from src.embedding.unimol import UniMolEmbedder
        embedder = UniMolEmbedder(model_name='unimolv1', device='cpu')
        embs = embedder.encode(smiles_list)
        dim = embs.shape[1]
    elif embedder_name == "unimol-v2-84m":
        from src.embedding.unimol import UniMolEmbedder
        embedder = UniMolEmbedder(model_name='unimolv2', model_size='84m', device='cpu')
        embs = embedder.encode(smiles_list)
        dim = embs.shape[1]
    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")

    elapsed = time.time() - t0
    print(f"    Done: shape={embs.shape}, time={elapsed:.1f}s")
    emb_dict = {smi: embs[i] for i, smi in enumerate(smiles_list)}
    return emb_dict, dim


def update_embedding_cache(smiles_list, embedder_name):
    """Compute and merge new embeddings into existing cache."""
    cache_file = CACHE_DIR / f"{embedder_name}.npz"

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        cached_smiles = set(data['smiles'].tolist())
        missing = [s for s in smiles_list if s not in cached_smiles]
        if not missing:
            print(f"    {embedder_name}: all {len(smiles_list)} molecules already cached")
            return
        print(f"    {embedder_name}: {len(missing)} missing from cache ({len(cached_smiles)} cached)")
    else:
        missing = smiles_list
        data = None
        print(f"    {embedder_name}: no cache, computing {len(missing)} molecules")

    emb_dict, dim = compute_pretrained_embeddings(missing, embedder_name)

    # Merge
    if data is not None:
        all_smi = list(data['smiles']) + missing
        new_embs = np.array([emb_dict[s] for s in missing], dtype=np.float32)
        all_emb = np.vstack([data['embeddings'], new_embs])
    else:
        all_smi = missing
        all_emb = np.array([emb_dict[s] for s in missing], dtype=np.float32)
        dim = all_emb.shape[1]

    np.savez_compressed(cache_file,
                        smiles=np.array(all_smi),
                        embeddings=all_emb,
                        emb_dim=np.array(dim))
    print(f"    Updated {embedder_name} cache: {len(all_smi)} total molecules")


# ═══════════════════════════════════════════════════════════════════════════
# ML Trainers
# ═══════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train, X_test, **kwargs):
    """Random Forest with configurable hyperparameters."""
    from sklearn.ensemble import RandomForestRegressor
    params = {
        "n_estimators": kwargs.get("n_estimators", 500),
        "max_features": kwargs.get("max_features", "sqrt"),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 3),
        "max_depth": kwargs.get("max_depth", None),
        "min_samples_split": kwargs.get("min_samples_split", 2),
        "n_jobs": N_JOBS,
        "random_state": 42,
    }
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    return rf.predict(X_test), rf


def train_xgboost(X_train, y_train, X_test, **kwargs):
    """XGBoost with configurable hyperparameters."""
    import xgboost as xgb
    params = {
        "max_depth": kwargs.get("max_depth", 4),
        "min_child_weight": kwargs.get("min_child_weight", 5),
        "subsample": kwargs.get("subsample", 0.7),
        "colsample_bytree": kwargs.get("colsample_bytree", 0.7),
        "learning_rate": kwargs.get("learning_rate", 0.05),
        "n_estimators": kwargs.get("n_estimators", 500),
        "reg_alpha": kwargs.get("reg_alpha", 0.1),
        "reg_lambda": kwargs.get("reg_lambda", 1.0),
        "gamma": kwargs.get("gamma", 0.0),
        "random_state": 42,
        "n_jobs": N_JOBS,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test), model


def train_ridge(X_train, y_train, X_test, **kwargs):
    """Ridge regression."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model.fit(X_tr, y_train)
    return model.predict(X_te), (model, scaler)


def train_gp_tanimoto(X_train, y_train, X_test, alpha=1.0):
    """KRR with Tanimoto kernel (GP-like)."""
    from sklearn.kernel_ridge import KernelRidge
    K_train = _tanimoto_kernel_matrix(X_train)
    K_test = _tanimoto_kernel_matrix(X_test, X_train)
    krr = KernelRidge(alpha=alpha, kernel="precomputed")
    krr.fit(K_train, y_train)
    return krr.predict(K_test), krr


def train_svr(X_train, y_train, X_test, **kwargs):
    """SVR with RBF kernel."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = SVR(kernel="rbf",
                C=kwargs.get("C", 10.0),
                gamma=kwargs.get("gamma", "scale"),
                epsilon=kwargs.get("epsilon", 0.1))
    model.fit(X_tr, y_train)
    return model.predict(X_te), (model, scaler)


def train_elasticnet(X_train, y_train, X_test, **kwargs):
    """Elastic Net with CV."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                         alphas=np.logspace(-4, 2, 20),
                         cv=3, max_iter=10000, n_jobs=N_JOBS)
    model.fit(X_tr, y_train)
    return model.predict(X_te), (model, scaler)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_0(mol_data, per_assay, results):
    """Diagnostics: fold analysis, residual analysis, label noise estimation."""
    print("\n" + "=" * 70)
    print("PHASE 0: Diagnostics")
    print("=" * 70)

    phase = results.get("phase_0", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
    splits = get_cv_splits(mol_data)

    # 0a. Per-fold analysis with RF
    print("\n  0a. Per-fold RF analysis...")
    fold_details = []
    all_preds = np.zeros(len(mol_data))
    all_errors = np.zeros(len(mol_data))

    for fold_name, train_df, test_df in splits:
        X_train = np.array([emb_dict[s] for s in train_df["smiles"]], dtype=np.float32)
        y_train = train_df["pIC50"].values.astype(np.float32)
        X_test = np.array([emb_dict[s] for s in test_df["smiles"]], dtype=np.float32)
        y_test = test_df["pIC50"].values.astype(np.float32)

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Store OOF predictions
        test_indices = test_df.index.tolist()
        for i, idx in enumerate(test_indices):
            pos = mol_data.index.get_loc(idx)
            all_preds[pos] = y_pred[i]
            all_errors[pos] = abs(y_test[i] - y_pred[i])

        m = compute_absolute_metrics(y_test, y_pred)
        m["fold"] = fold_name
        m["test_size"] = len(y_test)
        m["y_test_mean"] = float(y_test.mean())
        m["y_test_std"] = float(y_test.std())
        m["y_test_range"] = float(y_test.max() - y_test.min())

        # Worst-predicted molecules
        errors = np.abs(y_test - y_pred)
        worst_idx = np.argsort(errors)[-5:]
        m["worst_molecules"] = [
            {"smiles": test_df.iloc[i]["smiles"][:80],
             "true": float(y_test[i]), "pred": float(y_pred[i]),
             "error": float(errors[i])}
            for i in worst_idx
        ]
        fold_details.append(m)
        print(f"    {fold_name}: MAE={m['mae']:.4f}, R²={m['r2']:.3f}, Spr={m['spearman_r']:.3f}, "
              f"range={m['y_test_range']:.1f}, n={m['test_size']}")

    phase["fold_analysis"] = fold_details

    # 0b. Residual analysis
    print("\n  0b. Residual analysis...")
    y_all = mol_data["pIC50"].values
    residuals = y_all - all_preds
    abs_errors = np.abs(residuals)

    # Bin by pIC50 range
    bins = [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0)]
    binned_errors = {}
    for lo, hi in bins:
        mask = (y_all >= lo) & (y_all < hi)
        if mask.sum() > 0:
            binned_errors[f"{lo:.0f}-{hi:.0f}"] = {
                "n": int(mask.sum()),
                "mae": float(abs_errors[mask].mean()),
                "rmse": float(np.sqrt(np.mean(residuals[mask]**2))),
                "bias": float(residuals[mask].mean()),
            }
            print(f"    pIC50 [{lo:.0f},{hi:.0f}): n={mask.sum()}, MAE={abs_errors[mask].mean():.3f}, "
                  f"bias={residuals[mask].mean():.3f}")
    phase["residual_analysis"] = binned_errors

    # 0c. Label noise estimation from per-assay data
    print("\n  0c. Label noise estimation...")
    multi_assay = per_assay.groupby("molecule_chembl_id").filter(lambda x: len(x) > 1)
    if len(multi_assay) > 0:
        within_mol_var = multi_assay.groupby("molecule_chembl_id")["pIC50"].var()
        noise_stats = {
            "n_multi_assay_mols": int(multi_assay["molecule_chembl_id"].nunique()),
            "mean_within_mol_var": float(within_mol_var.mean()),
            "median_within_mol_var": float(within_mol_var.median()),
            "mean_within_mol_std": float(np.sqrt(within_mol_var.mean())),
            "estimated_label_noise_mae": float(np.sqrt(within_mol_var.mean()) * np.sqrt(2/np.pi)),
        }
        print(f"    {noise_stats['n_multi_assay_mols']} molecules measured in >1 assay")
        print(f"    Mean within-molecule σ: {noise_stats['mean_within_mol_std']:.3f} pIC50")
        print(f"    Estimated label noise MAE: {noise_stats['estimated_label_noise_mae']:.3f}")
        phase["label_noise"] = noise_stats
    else:
        print("    No multi-assay molecules for noise estimation")

    # 0d. Overall OOF metrics
    oof_metrics = compute_absolute_metrics(y_all, all_preds)
    phase["oof_metrics"] = oof_metrics
    print(f"\n  OOF overall: MAE={oof_metrics['mae']:.4f}, R²={oof_metrics['r2']:.3f}, "
          f"Spr={oof_metrics['spearman_r']:.3f}")

    phase["completed"] = True
    results["phase_0"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Fingerprint Diversity
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_1(mol_data, results):
    """Screen diverse fingerprint types."""
    print("\n" + "=" * 70)
    print("PHASE 1: Fingerprint Diversity Screening")
    print("=" * 70)

    phase = results.get("phase_1", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)

    # Define fingerprint types to screen
    fp_configs = [
        # (name, fp_type, kwargs)
        ("ECFP4_2048", "morgan", {"radius": 2, "n_bits": 2048}),
        ("ECFP6_2048", "morgan", {"radius": 3, "n_bits": 2048}),
        ("ECFP4_4096", "morgan", {"radius": 2, "n_bits": 4096}),
        ("FCFP4_2048", "morgan", {"radius": 2, "n_bits": 2048, "use_features": True}),
        ("FCFP6_2048", "morgan", {"radius": 3, "n_bits": 2048, "use_features": True}),
        ("RDKit_2048", "rdkit", {"n_bits": 2048}),
        ("MACCS_167", "maccs", {}),
        ("AtomPair_2048", "atompair", {"n_bits": 2048}),
        ("TopTorsion_2048", "torsion", {"n_bits": 2048}),
        ("Avalon_1024", "avalon", {"n_bits": 1024}),
        ("Avalon_2048", "avalon", {"n_bits": 2048}),
        ("MorganCount_2048", "morgan_count", {"radius": 2, "n_bits": 2048}),
        ("Pattern_2048", "pattern", {"n_bits": 2048}),
        ("Layered_2048", "layered", {"n_bits": 2048}),
    ]

    methods = {
        "RF": lambda Xtr, ytr, Xte: train_rf(Xtr, ytr, Xte)[0],
        "XGB": lambda Xtr, ytr, Xte: train_xgboost(Xtr, ytr, Xte)[0],
        "KRR_Tan": lambda Xtr, ytr, Xte: train_gp_tanimoto(Xtr, ytr, Xte)[0],
    }

    for fp_name, fp_type, fp_kwargs in fp_configs:
        print(f"\n  --- {fp_name} ---")
        try:
            X_all = compute_fingerprints(all_smiles, fp_type, **fp_kwargs)
            print(f"    Shape: {X_all.shape}, density: {(X_all > 0).mean():.3f}")
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        for method_name, train_fn in methods.items():
            key = f"{fp_name}__{method_name}"
            if key in phase:
                agg = phase[key].get("aggregated", {})
                print(f"    {method_name}: done (MAE={agg.get('mae_mean', '?'):.4f})")
                continue

            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                try:
                    train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
                    test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
                    X_train, X_test = X_all[train_idx], X_all[test_idx]
                    y_train = train_df["pIC50"].values.astype(np.float32)
                    y_test = test_df["pIC50"].values.astype(np.float32)

                    y_pred = train_fn(X_train, y_train, X_test)
                    fold_metrics.append(compute_absolute_metrics(y_test, y_pred))
                except Exception as e:
                    pass

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"    {method_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                      f"Spr={agg.get('spearman_r_mean', 0):.3f}")

            results["phase_1"] = phase
            save_results(results)

    # Also test RDKit 2D descriptors
    print("\n  --- RDKit 2D Descriptors ---")
    key_rdkit = "RDKit2D__RF"
    if key_rdkit not in phase:
        X_desc, desc_names = compute_rdkit_descriptors(all_smiles)
        print(f"    Shape: {X_desc.shape} ({len(desc_names)} descriptors after filtering)")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_desc_scaled = scaler.fit_transform(X_desc)

        for method_name, suffix in [("RF", ""), ("XGB", ""), ("Ridge", ""),
                                     ("ElasticNet", "")]:
            key = f"RDKit2D__{method_name}"
            if key in phase:
                continue

            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                try:
                    train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
                    test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
                    X_train = X_desc_scaled[train_idx]
                    X_test = X_desc_scaled[test_idx]
                    y_train = train_df["pIC50"].values.astype(np.float32)
                    y_test = test_df["pIC50"].values.astype(np.float32)

                    if method_name == "RF":
                        y_pred = train_rf(X_train, y_train, X_test)[0]
                    elif method_name == "XGB":
                        y_pred = train_xgboost(X_train, y_train, X_test)[0]
                    elif method_name == "Ridge":
                        y_pred = train_ridge(X_train, y_train, X_test)[0]
                    elif method_name == "ElasticNet":
                        y_pred = train_elasticnet(X_train, y_train, X_test)[0]

                    fold_metrics.append(compute_absolute_metrics(y_test, y_pred))
                except Exception:
                    pass

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"    {method_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                      f"Spr={agg.get('spearman_r_mean', 0):.3f}")

            results["phase_1"] = phase
            save_results(results)

    # Summary
    print("\n  === PHASE 1 SUMMARY ===")
    print(f"  {'FP + Method':<35} {'MAE':>8} {'Spr':>8}")
    print(f"  {'-' * 55}")
    ranked = []
    for key, val in phase.items():
        if isinstance(val, dict) and "aggregated" in val:
            agg = val["aggregated"]
            ranked.append((key, agg["mae_mean"], agg.get("spearman_r_mean", 0)))
    for name, mae, spr in sorted(ranked, key=lambda x: x[1]):
        print(f"  {name:<35} {mae:>8.4f} {spr:>8.3f}")

    phase["completed"] = True
    results["phase_1"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Compute Proper Pretrained Embeddings
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_2(mol_data, results):
    """Compute and test pretrained embeddings."""
    print("\n" + "=" * 70)
    print("PHASE 2: Pretrained Embeddings (compute + evaluate)")
    print("=" * 70)

    phase = results.get("phase_2", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)

    # Step 1: Update embedding caches
    embedders_to_update = ["chemberta2-mtr", "chemberta2-mlm", "molformer-xl"]
    # UniMol requires special handling, try separately
    for emb_name in embedders_to_update:
        key_update = f"cache_updated_{emb_name}"
        if key_update not in phase:
            try:
                update_embedding_cache(all_smiles, emb_name)
                phase[key_update] = True
                results["phase_2"] = phase
                save_results(results)
            except Exception as e:
                print(f"    {emb_name} cache update FAILED: {e}")
                phase[key_update] = f"failed: {str(e)[:100]}"
                results["phase_2"] = phase
                save_results(results)

    # Try UniMol separately
    for emb_name in ["unimol-v1"]:
        key_update = f"cache_updated_{emb_name}"
        if key_update not in phase:
            try:
                update_embedding_cache(all_smiles, emb_name)
                phase[key_update] = True
            except Exception as e:
                print(f"    {emb_name} FAILED: {e}")
                phase[key_update] = f"failed: {str(e)[:100]}"
            results["phase_2"] = phase
            save_results(results)

    # Step 2: Evaluate each embedding
    all_embedders = ["chemprop-dmpnn", "morgan", "chemberta2-mtr", "chemberta2-mlm",
                     "molformer-xl", "unimol-v1", "chemeleon"]
    methods = {
        "RF": lambda Xtr, ytr, Xte: train_rf(Xtr, ytr, Xte)[0],
        "XGB": lambda Xtr, ytr, Xte: train_xgboost(Xtr, ytr, Xte)[0],
        "Ridge": lambda Xtr, ytr, Xte: train_ridge(Xtr, ytr, Xte)[0],
    }

    for emb_name in all_embedders:
        print(f"\n  --- {emb_name} ---")
        try:
            emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
            # Check for zero vectors
            n_zero = sum(1 for s in all_smiles if np.all(emb_dict.get(s, np.zeros(1)) == 0))
            print(f"    dim={emb_dim}, zero_vectors={n_zero}/{len(all_smiles)}")
            if n_zero > len(all_smiles) * 0.5:
                print(f"    SKIPPING: too many zero vectors")
                phase[f"{emb_name}__status"] = f"skipped: {n_zero} zero vectors"
                continue
        except Exception as e:
            print(f"    Load failed: {e}")
            continue

        for method_name, train_fn in methods.items():
            key = f"{emb_name}__{method_name}"
            if key in phase and isinstance(phase[key], dict):
                agg = phase[key].get("aggregated", {})
                print(f"    {method_name}: done (MAE={agg.get('mae_mean', '?'):.4f})")
                continue

            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                try:
                    X_train = np.array([emb_dict[s] for s in train_df["smiles"]], dtype=np.float32)
                    y_train = train_df["pIC50"].values.astype(np.float32)
                    X_test = np.array([emb_dict[s] for s in test_df["smiles"]], dtype=np.float32)
                    y_test = test_df["pIC50"].values.astype(np.float32)

                    y_pred = train_fn(X_train, y_train, X_test)
                    fold_metrics.append(compute_absolute_metrics(y_test, y_pred))
                except Exception as e:
                    pass

            if fold_metrics:
                agg = aggregate_cv_results(fold_metrics)
                phase[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"    {method_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                      f"Spr={agg.get('spearman_r_mean', 0):.3f}")

            results["phase_2"] = phase
            save_results(results)

        del emb_dict
        gc.collect()

    phase["completed"] = True
    results["phase_2"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Feature Selection
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_3(mol_data, results):
    """Feature selection: Boruta + curated descriptor sets."""
    print("\n" + "=" * 70)
    print("PHASE 3: Feature Selection")
    print("=" * 70)

    phase = results.get("phase_3", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)
    y_all = mol_data["pIC50"].values.astype(np.float32)

    # 3a. RF importance-based selection on Morgan FP
    print("\n  3a. RF importance-based feature selection (Morgan FP)...")
    X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)

    from sklearn.ensemble import RandomForestRegressor
    rf_full = RandomForestRegressor(n_estimators=1000, max_features="sqrt",
                                    min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
    rf_full.fit(X_morgan, y_all)
    importances = rf_full.feature_importances_

    # Select top features at different thresholds
    for top_k in [50, 100, 200, 500]:
        key = f"Morgan_top{top_k}__RF"
        if key in phase:
            continue

        top_idx = np.argsort(importances)[-top_k:]
        X_selected = X_morgan[:, top_idx]

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_selected[train_idx], y_train, X_selected[test_idx])[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "n_features": top_k}
        print(f"    top-{top_k}: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_3"] = phase
        save_results(results)

    # 3b. Combined FP + descriptors with feature selection
    print("\n  3b. Combined Morgan + RDKit descriptors with importance selection...")
    X_desc, desc_names = compute_rdkit_descriptors(all_smiles)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_desc_scaled = scaler.fit_transform(X_desc)

    # Combine Morgan + scaled descriptors
    X_combined = np.hstack([X_morgan, X_desc_scaled])
    print(f"    Combined shape: {X_combined.shape}")

    # Train RF on combined, get importances
    rf_comb = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
    rf_comb.fit(X_combined, y_all)
    comb_imp = rf_comb.feature_importances_

    for top_k in [100, 200, 500]:
        key = f"Morgan+RDKit_top{top_k}__RF"
        if key in phase:
            continue

        top_idx = np.argsort(comb_imp)[-top_k:]
        X_sel = X_combined[:, top_idx]

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_sel[train_idx], y_train, X_sel[test_idx])[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "n_features": top_k}
        print(f"    top-{top_k}: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_3"] = phase
        save_results(results)

    # 3c. Multi-FP concatenation (best FPs from Phase 1)
    print("\n  3c. Multi-FP concatenation (diverse FP types)...")
    key = "MultiFP_concat__RF"
    if key not in phase:
        fp_sets = [
            ("ECFP4", compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)),
            ("FCFP4", compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048, use_features=True)),
            ("AtomPair", compute_fingerprints(all_smiles, "atompair", n_bits=2048)),
            ("TopTorsion", compute_fingerprints(all_smiles, "torsion", n_bits=2048)),
            ("MACCS", compute_fingerprints(all_smiles, "maccs")),
        ]
        X_multi = np.hstack([fp for _, fp in fp_sets])
        print(f"    Multi-FP shape: {X_multi.shape}")

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_multi[train_idx], y_train, X_multi[test_idx])[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Multi-FP RF: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_3"] = phase
        save_results(results)

    phase["completed"] = True
    results["phase_3"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_4(mol_data, results):
    """Bayesian hyperparameter optimization with Optuna."""
    print("\n" + "=" * 70)
    print("PHASE 4: Hyperparameter Optimization (Optuna)")
    print("=" * 70)

    phase = results.get("phase_4", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed, skipping")
        phase["completed"] = True
        phase["status"] = "optuna_not_installed"
        results["phase_4"] = phase
        save_results(results)
        return results

    all_smiles = mol_data["smiles"].tolist()
    y_all = mol_data["pIC50"].values.astype(np.float32)

    # Use best FP from Phase 1 (default to ECFP4 if Phase 1 not done)
    X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    # Also prepare RDKit descriptors for combined optimization
    X_desc, _ = compute_rdkit_descriptors(all_smiles)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_desc_scaled = scaler.fit_transform(X_desc)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # 4a. XGBoost optimization
    key = "xgboost_optimized"
    if key not in phase:
        print("\n  4a. Optimizing XGBoost...")
        import xgboost as xgb

        def xgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0, 5.0),
                "random_state": 42,
                "n_jobs": N_JOBS,
            }
            maes = []
            for train_idx, test_idx in kf.split(X_morgan):
                model = xgb.XGBRegressor(**params)
                model.fit(X_morgan[train_idx], y_all[train_idx], verbose=False)
                y_pred = model.predict(X_morgan[test_idx])
                maes.append(np.mean(np.abs(y_all[test_idx] - y_pred)))
            return np.mean(maes)

        study = optuna.create_study(direction="minimize")
        study.optimize(xgb_objective, n_trials=200, timeout=600)

        best_params = study.best_params
        best_mae = study.best_value
        print(f"    Best XGBoost MAE: {best_mae:.4f}")
        print(f"    Best params: {best_params}")

        # Evaluate with best params
        fold_metrics = []
        splits = get_cv_splits(mol_data)
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_xgboost(X_morgan[train_idx], y_train, X_morgan[test_idx],
                                   **best_params)[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "best_params": best_params,
                      "optuna_best_mae": float(best_mae), "n_trials": 200}
        print(f"    Optimized XGB: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_4"] = phase
        save_results(results)

    # 4b. RF optimization
    key = "rf_optimized"
    if key not in phase:
        print("\n  4b. Optimizing RF...")

        def rf_objective(trial):
            from sklearn.ensemble import RandomForestRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.2, 0.3]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "n_jobs": N_JOBS,
                "random_state": 42,
            }
            maes = []
            for train_idx, test_idx in kf.split(X_morgan):
                rf = RandomForestRegressor(**params)
                rf.fit(X_morgan[train_idx], y_all[train_idx])
                y_pred = rf.predict(X_morgan[test_idx])
                maes.append(np.mean(np.abs(y_all[test_idx] - y_pred)))
            return np.mean(maes)

        study = optuna.create_study(direction="minimize")
        study.optimize(rf_objective, n_trials=150, timeout=600)

        best_params = study.best_params
        best_mae = study.best_value
        print(f"    Best RF MAE: {best_mae:.4f}")

        fold_metrics = []
        splits = get_cv_splits(mol_data)
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_morgan[train_idx], y_train, X_morgan[test_idx],
                              **best_params)[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "best_params": {k: str(v) for k, v in best_params.items()},
                      "optuna_best_mae": float(best_mae), "n_trials": 150}
        print(f"    Optimized RF: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_4"] = phase
        save_results(results)

    # 4c. SVR optimization
    key = "svr_optimized"
    if key not in phase:
        print("\n  4c. Optimizing SVR...")

        def svr_objective(trial):
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
            epsilon = trial.suggest_float("epsilon", 0.01, 0.5)

            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler

            maes = []
            for train_idx, test_idx in kf.split(X_morgan):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_morgan[train_idx])
                X_te = sc.transform(X_morgan[test_idx])
                model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
                model.fit(X_tr, y_all[train_idx])
                y_pred = model.predict(X_te)
                maes.append(np.mean(np.abs(y_all[test_idx] - y_pred)))
            return np.mean(maes)

        study = optuna.create_study(direction="minimize")
        study.optimize(svr_objective, n_trials=100, timeout=300)

        best_params = study.best_params
        best_mae = study.best_value
        print(f"    Best SVR MAE: {best_mae:.4f}")

        fold_metrics = []
        splits = get_cv_splits(mol_data)
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_svr(X_morgan[train_idx], y_train, X_morgan[test_idx],
                               **best_params)[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "best_params": best_params,
                      "optuna_best_mae": float(best_mae)}
        print(f"    Optimized SVR: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_4"] = phase
        save_results(results)

    # 4d. KRR alpha optimization
    key = "krr_optimized"
    if key not in phase:
        print("\n  4d. Optimizing KRR (Tanimoto kernel)...")
        best_alpha = None
        best_mae = 999

        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            maes = []
            for train_idx, test_idx in kf.split(X_morgan):
                y_pred = train_gp_tanimoto(X_morgan[train_idx], y_all[train_idx],
                                           X_morgan[test_idx], alpha=alpha)[0]
                maes.append(np.mean(np.abs(y_all[test_idx] - y_pred)))
            mean_mae = np.mean(maes)
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_alpha = alpha

        print(f"    Best KRR alpha: {best_alpha}, MAE: {best_mae:.4f}")

        fold_metrics = []
        splits = get_cv_splits(mol_data)
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)
            y_pred = train_gp_tanimoto(X_morgan[train_idx], y_train,
                                        X_morgan[test_idx], alpha=best_alpha)[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "best_alpha": best_alpha}
        print(f"    Optimized KRR: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_4"] = phase
        save_results(results)

    phase["completed"] = True
    results["phase_4"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Advanced Ensembles
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_5(mol_data, results):
    """Build advanced ensembles from best components."""
    print("\n" + "=" * 70)
    print("PHASE 5: Advanced Ensembles")
    print("=" * 70)

    phase = results.get("phase_5", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)
    y_all = mol_data["pIC50"].values.astype(np.float32)

    # Get best hyperparams from Phase 4
    p4 = results.get("phase_4", {})
    xgb_params = p4.get("xgboost_optimized", {}).get("best_params", {})
    rf_params_raw = p4.get("rf_optimized", {}).get("best_params", {})
    svr_params = p4.get("svr_optimized", {}).get("best_params", {})
    krr_alpha = p4.get("krr_optimized", {}).get("best_alpha", 1.0)

    # Convert RF params from string
    rf_params = {}
    for k, v in rf_params_raw.items():
        try:
            rf_params[k] = eval(v) if isinstance(v, str) else v
        except Exception:
            rf_params[k] = v

    # Prepare diverse feature sets
    X_ecfp4 = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    X_ecfp6 = compute_fingerprints(all_smiles, "morgan", radius=3, n_bits=2048)
    X_fcfp4 = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048, use_features=True)
    X_desc, _ = compute_rdkit_descriptors(all_smiles)
    from sklearn.preprocessing import StandardScaler
    desc_scaler = StandardScaler()
    X_desc_scaled = desc_scaler.fit_transform(X_desc)

    # Try loading pretrained embeddings
    pretrained_embs = {}
    for emb_name in ["chemberta2-mtr", "chemberta2-mlm", "molformer-xl", "unimol-v1"]:
        try:
            emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
            n_zero = sum(1 for s in all_smiles if np.all(emb_dict.get(s, np.zeros(1)) == 0))
            if n_zero < len(all_smiles) * 0.5:
                X_emb = np.array([emb_dict[s] for s in all_smiles], dtype=np.float32)
                pretrained_embs[emb_name] = X_emb
                print(f"  Loaded {emb_name}: dim={emb_dim}, zeros={n_zero}")
            del emb_dict
        except Exception:
            pass

    # 5a. Diverse-FP ensemble (different FPs × different methods)
    key = "diverse_fp_ensemble"
    if key not in phase:
        print("\n  5a. Diverse FP ensemble...")
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            preds = []
            # RF on ECFP4
            preds.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            # XGB on ECFP4
            preds.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            # KRR on ECFP4
            preds.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            # RF on ECFP6
            preds.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx])[0])
            # RF on FCFP4
            preds.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            # RF on RDKit descriptors
            preds.append(train_rf(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])

            # Add pretrained embedding models
            for emb_name, X_emb in pretrained_embs.items():
                try:
                    preds.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    pass

            # Simple average
            y_pred = np.mean(preds, axis=0)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "n_models": len(preds)}
        print(f"    Diverse ensemble ({len(preds)} models): "
              f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_5"] = phase
        save_results(results)

    # 5b. Median ensemble
    key = "diverse_fp_median_ensemble"
    if key not in phase:
        print("\n  5b. Diverse FP median ensemble...")
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            preds = []
            preds.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            preds.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            preds.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            preds.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx])[0])
            preds.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            preds.append(train_rf(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])

            for emb_name, X_emb in pretrained_embs.items():
                try:
                    preds.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    pass

            y_pred = np.median(preds, axis=0)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Median ensemble: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_5"] = phase
        save_results(results)

    # 5c. Stacking (OOF predictions → Ridge meta-learner)
    key = "stacked_ensemble"
    if key not in phase:
        print("\n  5c. Stacked ensemble (Ridge meta-learner)...")

        # Generate OOF predictions for each base model
        n_base = 6 + len(pretrained_embs)
        oof_preds = np.zeros((len(mol_data), n_base))
        oof_valid = np.zeros(len(mol_data), dtype=bool)

        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)

            base_preds = []
            base_preds.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            base_preds.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            base_preds.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            base_preds.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx])[0])
            base_preds.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            base_preds.append(train_rf(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])

            for emb_name, X_emb in pretrained_embs.items():
                try:
                    base_preds.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    base_preds.append(np.full(len(test_idx), y_train.mean()))

            for i, idx in enumerate(test_idx):
                for j in range(min(len(base_preds), n_base)):
                    oof_preds[idx, j] = base_preds[j][i]
                oof_valid[idx] = True

        # Now train meta-learner and evaluate via nested CV
        fold_metrics = []
        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            # Use OOF from OTHER folds as meta-training
            meta_train_mask = oof_valid.copy()
            for idx in test_idx:
                meta_train_mask[idx] = False

            X_meta_train = oof_preds[meta_train_mask]
            y_meta_train = y_all[meta_train_mask]
            X_meta_test = oof_preds[test_idx]

            from sklearn.linear_model import RidgeCV
            meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            meta.fit(X_meta_train, y_meta_train)
            y_pred = meta.predict(X_meta_test)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "n_base_models": n_base}
        print(f"    Stacked ensemble: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_5"] = phase
        save_results(results)

    # 5d. Optimized weight ensemble
    key = "optimized_weight_ensemble"
    if key not in phase:
        print("\n  5d. Optimized-weight ensemble...")
        from scipy.optimize import minimize

        # Get OOF predictions (reuse from stacking)
        fold_metrics = []
        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            # Train base models
            preds_train = []
            preds_test = []
            fp_sets = [
                (X_ecfp4, rf_params, "rf"),
                (X_ecfp4, xgb_params, "xgb"),
                (X_ecfp4, {}, "krr"),
                (X_ecfp6, {}, "rf"),
                (X_fcfp4, {}, "rf"),
                (X_desc_scaled, {}, "rf"),
            ]
            for X_fp, params, model_type in fp_sets:
                if model_type == "rf":
                    p_tr = train_rf(X_fp[train_idx], y_train, X_fp[train_idx], **params)[0]
                    p_te = train_rf(X_fp[train_idx], y_train, X_fp[test_idx], **params)[0]
                elif model_type == "xgb":
                    p_tr = train_xgboost(X_fp[train_idx], y_train, X_fp[train_idx], **params)[0]
                    p_te = train_xgboost(X_fp[train_idx], y_train, X_fp[test_idx], **params)[0]
                elif model_type == "krr":
                    p_tr = train_gp_tanimoto(X_fp[train_idx], y_train, X_fp[train_idx], alpha=krr_alpha)[0]
                    p_te = train_gp_tanimoto(X_fp[train_idx], y_train, X_fp[test_idx], alpha=krr_alpha)[0]
                preds_train.append(p_tr)
                preds_test.append(p_te)

            preds_train = np.array(preds_train).T  # [n_train, n_models]
            preds_test = np.array(preds_test).T    # [n_test, n_models]

            # Optimize weights on training set
            def obj(w):
                w = np.abs(w) / np.sum(np.abs(w))
                y_p = preds_train @ w
                return np.mean(np.abs(y_train - y_p))

            n_models = preds_train.shape[1]
            w0 = np.ones(n_models) / n_models
            res = minimize(obj, w0, method='Nelder-Mead', options={"maxiter": 1000})
            w_opt = np.abs(res.x) / np.sum(np.abs(res.x))

            y_pred = preds_test @ w_opt
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Optimized-weight ensemble: MAE={agg['mae_mean']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_5"] = phase
        save_results(results)

    phase["completed"] = True
    results["phase_5"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: 3D Descriptors
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_6(mol_data, results):
    """3D conformer-based descriptors."""
    print("\n" + "=" * 70)
    print("PHASE 6: 3D Descriptors")
    print("=" * 70)

    phase = results.get("phase_6", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)

    # Compute 3D descriptors
    print("  Computing 3D conformer descriptors...")
    t0 = time.time()
    X_3d = compute_3d_descriptors(all_smiles, n_confs=10)
    elapsed = time.time() - t0
    print(f"  3D descriptors shape: {X_3d.shape}, time: {elapsed:.0f}s")
    print(f"  Non-zero fraction: {(X_3d != 0).mean():.3f}")

    # 3D only
    for method_name, train_fn in [("RF", lambda Xtr, ytr, Xte: train_rf(Xtr, ytr, Xte)[0]),
                                   ("XGB", lambda Xtr, ytr, Xte: train_xgboost(Xtr, ytr, Xte)[0])]:
        key = f"3D_only__{method_name}"
        if key in phase:
            continue

        fold_metrics = []
        from sklearn.preprocessing import StandardScaler
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_3d[train_idx])
            X_te = sc.transform(X_3d[test_idx])

            y_pred = train_fn(X_tr, y_train, X_te)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    3D-only {method_name}: MAE={agg['mae_mean']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_6"] = phase
        save_results(results)

    # Morgan + 3D
    key = "Morgan_plus_3D__RF"
    if key not in phase:
        X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
        from sklearn.preprocessing import StandardScaler
        sc_3d = StandardScaler()
        X_3d_scaled = sc_3d.fit_transform(X_3d)
        X_combined = np.hstack([X_morgan, X_3d_scaled])

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_combined[train_idx], y_train, X_combined[test_idx])[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Morgan+3D RF: MAE={agg['mae_mean']:.4f}, Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_6"] = phase
        save_results(results)

    # Morgan + RDKit2D + 3D
    key = "Morgan_RDKit2D_3D__RF"
    if key not in phase:
        X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
        X_desc, _ = compute_rdkit_descriptors(all_smiles)
        from sklearn.preprocessing import StandardScaler
        sc_d = StandardScaler()
        sc_3 = StandardScaler()
        X_combined = np.hstack([X_morgan, sc_d.fit_transform(X_desc), sc_3.fit_transform(X_3d)])

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            y_pred = train_rf(X_combined[train_idx], y_train, X_combined[test_idx])[0]
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Morgan+RDKit2D+3D RF: MAE={agg['mae_mean']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_6"] = phase
        save_results(results)

    phase["completed"] = True
    results["phase_6"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: Conformal Prediction
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_7(mol_data, results):
    """Conformal prediction for uncertainty quantification."""
    print("\n" + "=" * 70)
    print("PHASE 7: Conformal Prediction")
    print("=" * 70)

    phase = results.get("phase_7", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)
    y_all = mol_data["pIC50"].values.astype(np.float32)
    X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)

    # Collect OOF predictions and RF tree predictions for uncertainty
    print("  Computing OOF predictions with tree-level uncertainty...")

    oof_preds = np.zeros(len(mol_data))
    oof_tree_std = np.zeros(len(mol_data))
    oof_errors = np.zeros(len(mol_data))
    oof_nn_sim = np.zeros(len(mol_data))  # nearest neighbor similarity

    for fold_name, train_df, test_df in splits:
        train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
        test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
        y_train = train_df["pIC50"].values.astype(np.float32)
        y_test = test_df["pIC50"].values.astype(np.float32)

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=3, n_jobs=N_JOBS, random_state=42)
        rf.fit(X_morgan[train_idx], y_train)

        # Get individual tree predictions
        tree_preds = np.array([tree.predict(X_morgan[test_idx]) for tree in rf.estimators_])
        y_pred = tree_preds.mean(axis=0)
        y_std = tree_preds.std(axis=0)

        # Nearest neighbor Tanimoto similarity
        K = _tanimoto_kernel_matrix(X_morgan[test_idx], X_morgan[train_idx])
        nn_sim = K.max(axis=1)

        for i, idx in enumerate(test_idx):
            oof_preds[idx] = y_pred[i]
            oof_tree_std[idx] = y_std[i]
            oof_errors[idx] = abs(y_test[i] - y_pred[i])
            oof_nn_sim[idx] = nn_sim[i]

    # Conformal calibration
    alpha_levels = [0.1, 0.2, 0.5]
    conformal_results = {}
    for alpha in alpha_levels:
        # Split-conformal: use OOF residuals
        q = np.quantile(oof_errors, 1 - alpha)
        coverage = np.mean(oof_errors <= q)
        avg_width = 2 * q
        conformal_results[f"alpha_{alpha}"] = {
            "coverage": float(coverage),
            "target_coverage": float(1 - alpha),
            "q_threshold": float(q),
            "avg_interval_width": float(avg_width),
        }
        print(f"    α={alpha}: coverage={coverage:.3f} (target={1-alpha:.1f}), "
              f"interval=±{q:.3f}")

    # Correlation between uncertainty and error
    from scipy.stats import spearmanr
    sr_std, _ = spearmanr(oof_tree_std, oof_errors)
    sr_sim, _ = spearmanr(oof_nn_sim, -oof_errors)

    phase["conformal"] = conformal_results
    phase["uncertainty_correlation"] = {
        "tree_std_vs_error_spearman": float(sr_std),
        "nn_sim_vs_neg_error_spearman": float(sr_sim),
    }
    print(f"\n  Uncertainty calibration:")
    print(f"    Tree std vs error: Spearman={sr_std:.3f}")
    print(f"    NN similarity vs -error: Spearman={sr_sim:.3f}")

    # Applicability domain analysis
    ad_thresholds = [0.2, 0.3, 0.4, 0.5]
    ad_results = {}
    for thresh in ad_thresholds:
        in_domain = oof_nn_sim >= thresh
        if in_domain.sum() > 0 and (~in_domain).sum() > 0:
            mae_in = float(oof_errors[in_domain].mean())
            mae_out = float(oof_errors[~in_domain].mean())
            ad_results[f"sim_{thresh}"] = {
                "n_in": int(in_domain.sum()),
                "n_out": int((~in_domain).sum()),
                "mae_in": mae_in,
                "mae_out": mae_out,
                "ratio": float(mae_out / mae_in) if mae_in > 0 else 0,
            }
            print(f"    AD thresh={thresh}: in={in_domain.sum()} (MAE={mae_in:.3f}), "
                  f"out={int((~in_domain).sum())} (MAE={mae_out:.3f})")
    phase["applicability_domain"] = ad_results

    phase["completed"] = True
    results["phase_7"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8: Grand Ensemble — Best of Everything
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_8(mol_data, results):
    """Grand ensemble combining best models from all phases."""
    print("\n" + "=" * 70)
    print("PHASE 8: Grand Ensemble — Combining Best of Everything")
    print("=" * 70)

    phase = results.get("phase_8", {})
    if phase.get("completed"):
        print("  Already completed")
        return results

    all_smiles = mol_data["smiles"].tolist()
    splits = get_cv_splits(mol_data)
    y_all = mol_data["pIC50"].values.astype(np.float32)

    # Get optimized params from Phase 4
    p4 = results.get("phase_4", {})
    xgb_params = p4.get("xgboost_optimized", {}).get("best_params", {})
    rf_params_raw = p4.get("rf_optimized", {}).get("best_params", {})
    krr_alpha = p4.get("krr_optimized", {}).get("best_alpha", 0.05)

    rf_params = {}
    for k, v in rf_params_raw.items():
        try:
            rf_params[k] = eval(v) if isinstance(v, str) else v
        except Exception:
            rf_params[k] = v

    # Prepare all feature sets
    print("  Preparing feature sets...")
    X_ecfp4 = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)
    X_ecfp6 = compute_fingerprints(all_smiles, "morgan", radius=3, n_bits=2048)
    X_fcfp4 = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048, use_features=True)
    X_atompair = compute_fingerprints(all_smiles, "atompair", n_bits=2048)
    X_rdkit_fp = compute_fingerprints(all_smiles, "rdkit", n_bits=2048)
    X_torsion = compute_fingerprints(all_smiles, "torsion", n_bits=2048)

    X_desc, _ = compute_rdkit_descriptors(all_smiles)
    from sklearn.preprocessing import StandardScaler
    desc_scaler = StandardScaler()
    X_desc_scaled = desc_scaler.fit_transform(X_desc)

    # Load pretrained embeddings
    pretrained = {}
    for emb_name in ["chemberta2-mtr", "molformer-xl", "unimol-v1"]:
        try:
            emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
            n_zero = sum(1 for s in all_smiles if np.all(emb_dict.get(s, np.zeros(1)) == 0))
            if n_zero < len(all_smiles) * 0.5:
                pretrained[emb_name] = np.array([emb_dict[s] for s in all_smiles], dtype=np.float32)
            del emb_dict
        except Exception:
            pass
    gc.collect()

    # Multi-FP concat
    X_multifp = np.hstack([X_ecfp4, X_fcfp4, X_atompair, X_torsion,
                           compute_fingerprints(all_smiles, "maccs")])

    # 8a. Grand ensemble: 10+ diverse models
    key = "grand_ensemble_mean"
    if key not in phase:
        print("\n  8a. Grand ensemble (mean of diverse models)...")
        fold_metrics = []
        n_models_used = []

        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            preds = []
            # Optimized XGB on ECFP4
            preds.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            # Optimized RF on ECFP4
            preds.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            # KRR on ECFP4
            preds.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            # XGB on AtomPair (Phase 1 winner)
            preds.append(train_xgboost(X_atompair[train_idx], y_train, X_atompair[test_idx], **xgb_params)[0])
            # RF on RDKit FP (Phase 1 runner-up)
            preds.append(train_rf(X_rdkit_fp[train_idx], y_train, X_rdkit_fp[test_idx], **rf_params)[0])
            # RF on ECFP6
            preds.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx], **rf_params)[0])
            # RF on Multi-FP concat
            preds.append(train_rf(X_multifp[train_idx], y_train, X_multifp[test_idx])[0])
            # XGB on RDKit descriptors
            preds.append(train_xgboost(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])
            # RF on FCFP4
            preds.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            # KRR on AtomPair
            preds.append(train_gp_tanimoto(X_atompair[train_idx], y_train, X_atompair[test_idx], alpha=krr_alpha)[0])

            # Pretrained embedding models
            for emb_name, X_emb in pretrained.items():
                try:
                    preds.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                    preds.append(train_xgboost(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    pass

            n_models_used.append(len(preds))
            y_pred = np.mean(preds, axis=0)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "avg_models": float(np.mean(n_models_used))}
        print(f"    Grand mean ({np.mean(n_models_used):.0f} models): "
              f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_8"] = phase
        save_results(results)

    # 8b. Grand ensemble with median
    key = "grand_ensemble_median"
    if key not in phase:
        print("\n  8b. Grand ensemble (median)...")
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            preds = []
            preds.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            preds.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            preds.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            preds.append(train_xgboost(X_atompair[train_idx], y_train, X_atompair[test_idx], **xgb_params)[0])
            preds.append(train_rf(X_rdkit_fp[train_idx], y_train, X_rdkit_fp[test_idx], **rf_params)[0])
            preds.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx], **rf_params)[0])
            preds.append(train_rf(X_multifp[train_idx], y_train, X_multifp[test_idx])[0])
            preds.append(train_xgboost(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])
            preds.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            preds.append(train_gp_tanimoto(X_atompair[train_idx], y_train, X_atompair[test_idx], alpha=krr_alpha)[0])

            for emb_name, X_emb in pretrained.items():
                try:
                    preds.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                    preds.append(train_xgboost(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    pass

            y_pred = np.median(preds, axis=0)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Grand median: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_8"] = phase
        save_results(results)

    # 8c. Grand stacked ensemble
    key = "grand_stacked"
    if key not in phase:
        print("\n  8c. Grand stacked ensemble (Ridge meta-learner)...")
        n_base = 10 + 2 * len(pretrained)

        # Generate OOF predictions
        oof_preds = np.zeros((len(mol_data), n_base))
        oof_valid = np.zeros(len(mol_data), dtype=bool)

        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)

            base = []
            base.append(train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0])
            base.append(train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0])
            base.append(train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0])
            base.append(train_xgboost(X_atompair[train_idx], y_train, X_atompair[test_idx], **xgb_params)[0])
            base.append(train_rf(X_rdkit_fp[train_idx], y_train, X_rdkit_fp[test_idx], **rf_params)[0])
            base.append(train_rf(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx], **rf_params)[0])
            base.append(train_rf(X_multifp[train_idx], y_train, X_multifp[test_idx])[0])
            base.append(train_xgboost(X_desc_scaled[train_idx], y_train, X_desc_scaled[test_idx])[0])
            base.append(train_rf(X_fcfp4[train_idx], y_train, X_fcfp4[test_idx])[0])
            base.append(train_gp_tanimoto(X_atompair[train_idx], y_train, X_atompair[test_idx], alpha=krr_alpha)[0])

            for emb_name, X_emb in pretrained.items():
                try:
                    base.append(train_rf(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                    base.append(train_xgboost(X_emb[train_idx], y_train, X_emb[test_idx])[0])
                except Exception:
                    base.append(np.full(len(test_idx), y_train.mean()))
                    base.append(np.full(len(test_idx), y_train.mean()))

            for i, idx in enumerate(test_idx):
                for j in range(min(len(base), n_base)):
                    oof_preds[idx, j] = base[j][i]
                oof_valid[idx] = True

        # Train meta-learner
        fold_metrics = []
        for fold_idx, (fold_name, train_df, test_df) in enumerate(splits):
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_test = test_df["pIC50"].values.astype(np.float32)

            meta_train_mask = oof_valid.copy()
            for idx in test_idx:
                meta_train_mask[idx] = False

            from sklearn.linear_model import RidgeCV
            meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            meta.fit(oof_preds[meta_train_mask], y_all[meta_train_mask])
            y_pred = meta.predict(oof_preds[test_idx])
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg, "n_base_models": n_base}
        print(f"    Grand stacked ({n_base} models): MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_8"] = phase
        save_results(results)

    # 8d. Top-3 optimized XGB (different FPs)
    key = "top3_xgb_mean"
    if key not in phase:
        print("\n  8d. Top-3 optimized XGB (ECFP4 + AtomPair + ECFP6)...")
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            p1 = train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0]
            p2 = train_xgboost(X_atompair[train_idx], y_train, X_atompair[test_idx], **xgb_params)[0]
            p3 = train_xgboost(X_ecfp6[train_idx], y_train, X_ecfp6[test_idx], **xgb_params)[0]
            y_pred = (p1 + p2 + p3) / 3
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Top-3 XGB mean: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_8"] = phase
        save_results(results)

    # 8e. Top-5 mixed (XGB+RF on best FPs)
    key = "top5_mixed_mean"
    if key not in phase:
        print("\n  8e. Top-5 mixed (XGB+RF on best FPs)...")
        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            train_idx = [all_smiles.index(s) for s in train_df["smiles"]]
            test_idx = [all_smiles.index(s) for s in test_df["smiles"]]
            y_train = train_df["pIC50"].values.astype(np.float32)
            y_test = test_df["pIC50"].values.astype(np.float32)

            preds = [
                train_xgboost(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **xgb_params)[0],
                train_rf(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], **rf_params)[0],
                train_xgboost(X_atompair[train_idx], y_train, X_atompair[test_idx], **xgb_params)[0],
                train_rf(X_rdkit_fp[train_idx], y_train, X_rdkit_fp[test_idx], **rf_params)[0],
                train_gp_tanimoto(X_ecfp4[train_idx], y_train, X_ecfp4[test_idx], alpha=krr_alpha)[0],
            ]
            y_pred = np.mean(preds, axis=0)
            fold_metrics.append(compute_absolute_metrics(y_test, y_pred))

        agg = aggregate_cv_results(fold_metrics)
        phase[key] = {"aggregated": agg}
        print(f"    Top-5 mixed: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg.get('spearman_r_mean', 0):.3f}")
        results["phase_8"] = phase
        save_results(results)

    phase["completed"] = True
    results["phase_8"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ZAP70 v3 — deep iterative optimization")
    parser.add_argument("--phase", nargs="+", type=int, default=None,
                        help="Phases to run (0-7). Default: all")
    args = parser.parse_args()

    phases_to_run = set(args.phase) if args.phase else set(range(9))

    print("=" * 70)
    print("ZAP70 Case Study v3 — Deep Iterative Optimization")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Phases: {sorted(phases_to_run)}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    mol_data, per_assay = load_zap70_molecules()

    # Load existing results
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"  Loaded existing results from {RESULTS_FILE.name}")
    else:
        results = {
            "data_summary": {
                "target": "ZAP70", "target_id": ZAP70_ID,
                "n_molecules": len(mol_data),
                "n_assays": per_assay["assay_id"].nunique(),
                "pIC50_range": f"{mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f}",
                "pIC50_mean": float(mol_data['pIC50'].mean()),
                "pIC50_std": float(mol_data['pIC50'].std()),
            },
            "v2_best": {"mae": 0.603, "r2": 0.449, "spearman": 0.698,
                        "method": "Ensemble RF+XGB+GP (Morgan 2048)"},
        }

    t_start = time.time()

    if 0 in phases_to_run:
        results = run_phase_0(mol_data, per_assay, results)
    if 1 in phases_to_run:
        results = run_phase_1(mol_data, results)
    if 2 in phases_to_run:
        results = run_phase_2(mol_data, results)
    if 3 in phases_to_run:
        results = run_phase_3(mol_data, results)
    if 4 in phases_to_run:
        results = run_phase_4(mol_data, results)
    if 5 in phases_to_run:
        results = run_phase_5(mol_data, results)
    if 6 in phases_to_run:
        results = run_phase_6(mol_data, results)
    if 7 in phases_to_run:
        results = run_phase_7(mol_data, results)
    if 8 in phases_to_run:
        results = run_phase_8(mol_data, results)

    elapsed = time.time() - t_start
    results["total_time_seconds"] = elapsed
    results["completed"] = datetime.now().isoformat()
    save_results(results)

    # Final summary
    print("\n" + "=" * 70)
    print(f"COMPLETE — Total time: {elapsed/3600:.1f} hours")
    print(f"Results saved to {RESULTS_FILE}")
    print("=" * 70)

    # Print best results across all phases
    print("\n  === BEST RESULTS ACROSS ALL PHASES ===")
    print(f"  {'Source':<40} {'MAE':>8} {'Spr':>8}")
    print(f"  {'-' * 60}")
    all_results = []
    for phase_name, phase_data in results.items():
        if not isinstance(phase_data, dict) or phase_name in ("data_summary", "v2_best",
                                                                "total_time_seconds", "completed"):
            continue
        for key, val in phase_data.items():
            if isinstance(val, dict) and "aggregated" in val:
                agg = val["aggregated"]
                all_results.append((f"{phase_name}/{key}",
                                   agg.get("mae_mean", 999),
                                   agg.get("spearman_r_mean", 0)))
    for name, mae, spr in sorted(all_results, key=lambda x: x[1])[:20]:
        marker = " ***" if mae < 0.603 else ""
        print(f"  {name:<40} {mae:>8.4f} {spr:>8.3f}{marker}")


if __name__ == "__main__":
    main()
