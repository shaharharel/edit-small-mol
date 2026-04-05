#!/usr/bin/env python3
"""
ZAP70 vs SYK Selectivity Analysis

Multi-target selectivity optimization: jointly predicting ZAP70 potency and
selectivity over SYK (both Syk family kinases).

Key outputs:
- Overlap characterization (100 molecules tested on both targets)
- XGBoost models for ZAP70 pIC50, SYK pIC50, and direct selectivity
- Ranked ZAP70 molecules by predicted selectivity
- Structural feature importance for selectivity

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_zap70_syk_selectivity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "zap70_syk_selectivity.json"

ZAP70_ID = "CHEMBL2803"
SYK_ID = "CHEMBL2599"
N_FOLDS = 5
CV_SEED = 42
N_JOBS = 8


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pr = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0
    sr = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0
    if np.isnan(pr): pr = 0.0
    if np.isnan(sr): sr = 0.0
    return {"n": len(y_true), "mae": mae, "rmse": rmse, "r2": r2,
            "pearson_r": pr, "spearman_r": sr}


def compute_morgan_fp(smiles_list, radius=2, nbits=2048):
    """Compute Morgan fingerprints as numpy array."""
    fps = []
    valid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros(nbits, dtype=np.float32)
            fp.ToBitString()
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            fps.append(arr)
            valid.append(i)
        else:
            fps.append(np.zeros(nbits, dtype=np.float32))
            valid.append(i)
    return np.array(fps), valid


def get_xgb_model():
    """Standard XGBoost config for pIC50 prediction."""
    return XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=N_JOBS, random_state=CV_SEED, verbosity=0,
    )


def load_target_molecules(raw, target_id, target_name):
    """Load molecule-level data for a target (averaged across assays)."""
    subset = raw[raw["target_chembl_id"] == target_id].copy()
    mol_data = subset.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  {target_name} ({target_id}): {len(mol_data)} molecules, "
          f"pIC50 {mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f} "
          f"(mean={mol_data['pIC50'].mean():.2f}, std={mol_data['pIC50'].std():.2f})")
    return mol_data


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Characterize the overlap
# ═══════════════════════════════════════════════════════════════════════════

def characterize_overlap(zap_data, syk_data):
    """Analyze the ZAP70-SYK overlap set."""
    print("\n" + "=" * 70)
    print("PHASE 1: Characterize ZAP70-SYK Overlap")
    print("=" * 70)

    overlap_ids = set(zap_data.molecule_chembl_id) & set(syk_data.molecule_chembl_id)
    n_overlap = len(overlap_ids)
    print(f"\n  ZAP70 molecules: {len(zap_data)}")
    print(f"  SYK molecules:   {len(syk_data)}")
    print(f"  Overlap:         {n_overlap} molecules tested on both")
    print(f"  ZAP70-only:      {len(zap_data) - n_overlap}")
    print(f"  SYK-only:        {len(syk_data) - n_overlap}")

    # Build overlap dataframe
    zap_overlap = zap_data[zap_data.molecule_chembl_id.isin(overlap_ids)].set_index("molecule_chembl_id")
    syk_overlap = syk_data[syk_data.molecule_chembl_id.isin(overlap_ids)].set_index("molecule_chembl_id")
    overlap_df = pd.DataFrame({
        "smiles": zap_overlap["smiles"],
        "pIC50_ZAP70": zap_overlap["pIC50"],
        "pIC50_SYK": syk_overlap.loc[zap_overlap.index, "pIC50"],
    }).reset_index()
    overlap_df["selectivity"] = overlap_df["pIC50_ZAP70"] - overlap_df["pIC50_SYK"]

    # Correlation
    pr = pearsonr(overlap_df.pIC50_ZAP70, overlap_df.pIC50_SYK)[0]
    sr = spearmanr(overlap_df.pIC50_ZAP70, overlap_df.pIC50_SYK)[0]
    print(f"\n  ZAP70-SYK pIC50 correlation (n={n_overlap}):")
    print(f"    Pearson r:  {pr:.3f}")
    print(f"    Spearman r: {sr:.3f}")

    # Selectivity stats
    sel = overlap_df.selectivity
    print(f"\n  Selectivity = pIC50_ZAP70 - pIC50_SYK:")
    print(f"    Mean:   {sel.mean():.3f}")
    print(f"    Std:    {sel.std():.3f}")
    print(f"    Median: {sel.median():.3f}")
    print(f"    Range:  [{sel.min():.3f}, {sel.max():.3f}]")
    print(f"    ZAP70-selective (>0): {(sel > 0).sum()} ({100*(sel>0).mean():.1f}%)")
    print(f"    SYK-selective (<0):   {(sel < 0).sum()} ({100*(sel<0).mean():.1f}%)")
    print(f"    Highly ZAP70-sel (>1): {(sel > 1).sum()}")
    print(f"    Highly SYK-sel (<-1):  {(sel < -1).sum()}")

    stats = {
        "n_zap70": len(zap_data),
        "n_syk": len(syk_data),
        "n_overlap": n_overlap,
        "zap70_only": len(zap_data) - n_overlap,
        "syk_only": len(syk_data) - n_overlap,
        "correlation_pearson": float(pr),
        "correlation_spearman": float(sr),
        "selectivity_mean": float(sel.mean()),
        "selectivity_std": float(sel.std()),
        "selectivity_median": float(sel.median()),
        "selectivity_min": float(sel.min()),
        "selectivity_max": float(sel.max()),
        "n_zap70_selective": int((sel > 0).sum()),
        "n_syk_selective": int((sel < 0).sum()),
        "n_highly_zap70_selective": int((sel > 1).sum()),
        "n_highly_syk_selective": int((sel < -1).sum()),
    }
    return overlap_df, stats


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Build prediction models
# ═══════════════════════════════════════════════════════════════════════════

def build_models(zap_data, syk_data, overlap_df):
    """Build XGBoost models for ZAP70, SYK, and selectivity prediction."""
    print("\n" + "=" * 70)
    print("PHASE 2: Build Prediction Models")
    print("=" * 70)

    model_results = {}

    # --- Model 1: ZAP70 pIC50 (all 280 molecules) ---
    print("\n  Model 1: ZAP70 pIC50 (full dataset, 5-fold CV)")
    X_zap, _ = compute_morgan_fp(zap_data.smiles.tolist())
    y_zap = zap_data.pIC50.values

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    y_pred_zap = cross_val_predict(get_xgb_model(), X_zap, y_zap, cv=kf, n_jobs=1)
    m1 = compute_metrics(y_zap, y_pred_zap)
    print(f"    MAE={m1['mae']:.3f}, RMSE={m1['rmse']:.3f}, R2={m1['r2']:.3f}, "
          f"Pearson={m1['pearson_r']:.3f}, Spearman={m1['spearman_r']:.3f}")
    model_results["zap70_pIC50"] = m1

    # --- Model 2: SYK pIC50 (all 5208 molecules) ---
    print(f"\n  Model 2: SYK pIC50 (full dataset, {len(syk_data)} mols, 5-fold CV)")
    X_syk, _ = compute_morgan_fp(syk_data.smiles.tolist())
    y_syk = syk_data.pIC50.values

    y_pred_syk = cross_val_predict(get_xgb_model(), X_syk, y_syk, cv=kf, n_jobs=1)
    m2 = compute_metrics(y_syk, y_pred_syk)
    print(f"    MAE={m2['mae']:.3f}, RMSE={m2['rmse']:.3f}, R2={m2['r2']:.3f}, "
          f"Pearson={m2['pearson_r']:.3f}, Spearman={m2['spearman_r']:.3f}")
    model_results["syk_pIC50"] = m2

    # --- Model 3: Direct selectivity (overlap set, 100 molecules) ---
    print(f"\n  Model 3: Selectivity (overlap set, {len(overlap_df)} mols, 5-fold CV)")
    X_overlap, _ = compute_morgan_fp(overlap_df.smiles.tolist())
    y_selectivity = overlap_df.selectivity.values

    y_pred_sel = cross_val_predict(get_xgb_model(), X_overlap, y_selectivity, cv=kf, n_jobs=1)
    m3 = compute_metrics(y_selectivity, y_pred_sel)
    print(f"    MAE={m3['mae']:.3f}, RMSE={m3['rmse']:.3f}, R2={m3['r2']:.3f}, "
          f"Pearson={m3['pearson_r']:.3f}, Spearman={m3['spearman_r']:.3f}")
    model_results["selectivity_direct"] = m3

    # --- Model 4: Indirect selectivity (predict ZAP70 & SYK separately, subtract) ---
    print(f"\n  Model 4: Indirect selectivity (separate ZAP70 & SYK models, evaluated on overlap)")
    # Train ZAP70 model on non-overlap, predict overlap (or use CV predictions)
    # Simpler: use CV predictions from models 1 & 2 for overlap molecules
    overlap_idx_in_zap = zap_data.molecule_chembl_id.isin(overlap_df.molecule_chembl_id)
    overlap_idx_in_syk = syk_data.molecule_chembl_id.isin(overlap_df.molecule_chembl_id)

    # Map CV predictions back to overlap molecules
    zap_cv_df = zap_data.copy()
    zap_cv_df["pred_pIC50"] = y_pred_zap
    syk_cv_df = syk_data.copy()
    syk_cv_df["pred_pIC50"] = y_pred_syk

    zap_cv_overlap = zap_cv_df[zap_cv_df.molecule_chembl_id.isin(overlap_df.molecule_chembl_id)].set_index("molecule_chembl_id")
    syk_cv_overlap = syk_cv_df[syk_cv_df.molecule_chembl_id.isin(overlap_df.molecule_chembl_id)].set_index("molecule_chembl_id")

    common_idx = zap_cv_overlap.index.intersection(syk_cv_overlap.index)
    pred_sel_indirect = zap_cv_overlap.loc[common_idx, "pred_pIC50"] - syk_cv_overlap.loc[common_idx, "pred_pIC50"]

    # Ground truth selectivity for these molecules
    overlap_indexed = overlap_df.set_index("molecule_chembl_id")
    true_sel = overlap_indexed.loc[common_idx, "selectivity"]

    m4 = compute_metrics(true_sel.values, pred_sel_indirect.values)
    print(f"    MAE={m4['mae']:.3f}, RMSE={m4['rmse']:.3f}, R2={m4['r2']:.3f}, "
          f"Pearson={m4['pearson_r']:.3f}, Spearman={m4['spearman_r']:.3f}")
    model_results["selectivity_indirect"] = m4

    print(f"\n  Direct vs Indirect selectivity comparison:")
    print(f"    Direct MAE:   {m3['mae']:.3f}  (trained on selectivity directly)")
    print(f"    Indirect MAE: {m4['mae']:.3f}  (ZAP70 pred - SYK pred)")
    better = "Direct" if m3['mae'] < m4['mae'] else "Indirect"
    print(f"    Winner: {better}")

    return model_results, y_pred_zap, y_pred_syk, y_pred_sel


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Rank ZAP70 molecules by predicted selectivity
# ═══════════════════════════════════════════════════════════════════════════

def rank_by_selectivity(zap_data, syk_data, overlap_df):
    """Train SYK model on all SYK data, predict SYK pIC50 for all ZAP70 molecules."""
    print("\n" + "=" * 70)
    print("PHASE 3: Rank ZAP70 Molecules by Predicted Selectivity")
    print("=" * 70)

    # Train full SYK model
    X_syk_full, _ = compute_morgan_fp(syk_data.smiles.tolist())
    y_syk_full = syk_data.pIC50.values
    syk_model = get_xgb_model()
    syk_model.fit(X_syk_full, y_syk_full)

    # Train full ZAP70 model
    X_zap_full, _ = compute_morgan_fp(zap_data.smiles.tolist())
    y_zap_full = zap_data.pIC50.values
    zap_model = get_xgb_model()
    zap_model.fit(X_zap_full, y_zap_full)

    # Predict SYK pIC50 for all ZAP70 molecules
    pred_syk_for_zap = syk_model.predict(X_zap_full)

    # Compute predicted selectivity
    ranking_df = zap_data[["molecule_chembl_id", "smiles", "pIC50"]].copy()
    ranking_df.columns = ["molecule_chembl_id", "smiles", "pIC50_ZAP70"]
    ranking_df["pred_pIC50_SYK"] = pred_syk_for_zap
    ranking_df["pred_selectivity"] = ranking_df["pIC50_ZAP70"] - ranking_df["pred_pIC50_SYK"]

    # Mark which molecules have measured SYK data
    overlap_ids = set(overlap_df.molecule_chembl_id)
    ranking_df["has_measured_SYK"] = ranking_df.molecule_chembl_id.isin(overlap_ids)

    # For overlap molecules, add measured values
    overlap_indexed = overlap_df.set_index("molecule_chembl_id")
    for idx, row in ranking_df.iterrows():
        if row["molecule_chembl_id"] in overlap_indexed.index:
            ranking_df.loc[idx, "measured_pIC50_SYK"] = overlap_indexed.loc[row["molecule_chembl_id"], "pIC50_SYK"]
            ranking_df.loc[idx, "measured_selectivity"] = overlap_indexed.loc[row["molecule_chembl_id"], "selectivity"]

    ranking_df = ranking_df.sort_values("pred_selectivity", ascending=False).reset_index(drop=True)

    # Print top and bottom
    print(f"\n  Top 10 most ZAP70-selective molecules (predicted):")
    print(f"  {'Rank':<5} {'ChEMBL ID':<16} {'ZAP70':<7} {'SYK(pred)':<10} {'Sel(pred)':<10} {'Meas?':<6}")
    for i, row in ranking_df.head(10).iterrows():
        meas = "Yes" if row["has_measured_SYK"] else "No"
        print(f"  {i+1:<5} {row['molecule_chembl_id']:<16} {row['pIC50_ZAP70']:<7.2f} "
              f"{row['pred_pIC50_SYK']:<10.2f} {row['pred_selectivity']:<10.2f} {meas:<6}")

    print(f"\n  Top 10 most SYK-selective (least ZAP70-selective) molecules:")
    for i, (_, row) in enumerate(ranking_df.tail(10).iloc[::-1].iterrows()):
        meas = "Yes" if row["has_measured_SYK"] else "No"
        print(f"  {len(ranking_df)-i:<5} {row['molecule_chembl_id']:<16} {row['pIC50_ZAP70']:<7.2f} "
              f"{row['pred_pIC50_SYK']:<10.2f} {row['pred_selectivity']:<10.2f} {meas:<6}")

    # Validate predictions on overlap set
    overlap_ranked = ranking_df[ranking_df.has_measured_SYK].copy()
    syk_pred_err = compute_metrics(
        overlap_ranked.measured_pIC50_SYK.values,
        overlap_ranked.pred_pIC50_SYK.values,
    )
    sel_pred_err = compute_metrics(
        overlap_ranked.measured_selectivity.values,
        overlap_ranked.pred_selectivity.values,
    )
    print(f"\n  SYK prediction quality on overlap set (n={len(overlap_ranked)}):")
    print(f"    MAE={syk_pred_err['mae']:.3f}, Spearman={syk_pred_err['spearman_r']:.3f}")
    print(f"  Selectivity prediction quality on overlap set:")
    print(f"    MAE={sel_pred_err['mae']:.3f}, Spearman={sel_pred_err['spearman_r']:.3f}")

    # Distribution summary
    print(f"\n  Selectivity distribution (all {len(ranking_df)} ZAP70 mols):")
    print(f"    Predicted ZAP70-selective (>0): {(ranking_df.pred_selectivity > 0).sum()}")
    print(f"    Predicted SYK-selective (<0):   {(ranking_df.pred_selectivity < 0).sum()}")
    print(f"    Predicted highly ZAP70-sel (>1): {(ranking_df.pred_selectivity > 1).sum()}")

    return ranking_df, syk_pred_err, sel_pred_err, zap_model, syk_model


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Structural feature analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_selectivity_features(zap_data, syk_data, overlap_df):
    """Identify Morgan FP bits correlated with selectivity."""
    print("\n" + "=" * 70)
    print("PHASE 4: Structural Feature Analysis for Selectivity")
    print("=" * 70)

    # Train selectivity model on overlap set to get feature importances
    X_overlap, _ = compute_morgan_fp(overlap_df.smiles.tolist())
    y_sel = overlap_df.selectivity.values

    model = get_xgb_model()
    model.fit(X_overlap, y_sel)

    importances = model.feature_importances_
    top_bits = np.argsort(importances)[::-1][:20]

    print(f"\n  Top 20 Morgan FP bits for selectivity prediction:")
    print(f"  {'Rank':<5} {'Bit':<6} {'Importance':<12}")
    for rank, bit in enumerate(top_bits):
        if importances[bit] > 0:
            print(f"  {rank+1:<5} {bit:<6} {importances[bit]:.4f}")

    # Analyze bit presence correlation with selectivity
    print(f"\n  Bit-selectivity correlations (Pearson r):")
    bit_correlations = []
    for bit in range(X_overlap.shape[1]):
        if X_overlap[:, bit].sum() >= 5:  # at least 5 molecules with this bit
            r = pearsonr(X_overlap[:, bit], y_sel)[0]
            if not np.isnan(r):
                bit_correlations.append((bit, r, int(X_overlap[:, bit].sum())))

    bit_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Bit':<6} {'Pearson r':<12} {'#Mols with bit':<15} {'Direction':<12}")
    for bit, r, count in bit_correlations[:15]:
        direction = "ZAP70-sel" if r > 0 else "SYK-sel"
        print(f"  {bit:<6} {r:+.3f}       {count:<15} {direction}")

    # Try to decode top bits into substructure info
    print(f"\n  Decoding top correlated bits to substructures:")
    for bit, r, count in bit_correlations[:10]:
        # Find example molecules with this bit ON
        examples_on = []
        for i, smi in enumerate(overlap_df.smiles.tolist()):
            if X_overlap[i, bit] == 1 and len(examples_on) < 3:
                examples_on.append(smi)

        # Use RDKit to get bit info
        mol = Chem.MolFromSmiles(examples_on[0]) if examples_on else None
        if mol is not None:
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)
            if bit in bi:
                atom_idx, radius = bi[bit][0]
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                if submol.GetNumAtoms() > 0:
                    smi_frag = Chem.MolToSmiles(submol)
                    direction = "ZAP70-sel" if r > 0 else "SYK-sel"
                    print(f"    Bit {bit} (r={r:+.3f}, {direction}): {smi_frag} (radius={radius})")

    feature_results = {
        "top_importance_bits": [
            {"bit": int(bit), "importance": float(importances[bit])}
            for bit in top_bits if importances[bit] > 0
        ],
        "top_correlated_bits": [
            {"bit": int(bit), "pearson_r": float(r), "n_molecules": int(count),
             "direction": "ZAP70_selective" if r > 0 else "SYK_selective"}
            for bit, r, count in bit_correlations[:20]
        ],
    }
    return feature_results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Potency-selectivity relationship
# ═══════════════════════════════════════════════════════════════════════════

def potency_selectivity_analysis(ranking_df, overlap_df):
    """Analyze the relationship between potency and selectivity."""
    print("\n" + "=" * 70)
    print("PHASE 5: Potency-Selectivity Relationship")
    print("=" * 70)

    # For overlap molecules: measured potency vs measured selectivity
    pr_meas = pearsonr(overlap_df.pIC50_ZAP70, overlap_df.selectivity)[0]
    sr_meas = spearmanr(overlap_df.pIC50_ZAP70, overlap_df.selectivity)[0]
    print(f"\n  Measured ZAP70 potency vs measured selectivity (n={len(overlap_df)}):")
    print(f"    Pearson r:  {pr_meas:.3f}")
    print(f"    Spearman r: {sr_meas:.3f}")

    # For all ZAP70 molecules: predicted selectivity vs potency
    pr_pred = pearsonr(ranking_df.pIC50_ZAP70, ranking_df.pred_selectivity)[0]
    sr_pred = spearmanr(ranking_df.pIC50_ZAP70, ranking_df.pred_selectivity)[0]
    print(f"\n  ZAP70 potency vs predicted selectivity (n={len(ranking_df)}):")
    print(f"    Pearson r:  {pr_pred:.3f}")
    print(f"    Spearman r: {sr_pred:.3f}")

    # Potency bins
    print(f"\n  Selectivity by ZAP70 potency quartile:")
    ranking_df["potency_q"] = pd.qcut(ranking_df.pIC50_ZAP70, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
        sub = ranking_df[ranking_df.potency_q == q]
        print(f"    {q}: mean_sel={sub.pred_selectivity.mean():.3f}, "
              f"n_ZAP70_sel={int((sub.pred_selectivity > 0).sum())}/{len(sub)}")

    # Identify "sweet spot" molecules: potent AND selective
    potent_threshold = ranking_df.pIC50_ZAP70.quantile(0.75)
    selective_threshold = ranking_df.pred_selectivity.quantile(0.75)
    sweet_spot = ranking_df[
        (ranking_df.pIC50_ZAP70 >= potent_threshold) &
        (ranking_df.pred_selectivity >= selective_threshold)
    ]
    print(f"\n  Sweet spot (top 25% potency AND top 25% selectivity): {len(sweet_spot)} molecules")
    print(f"    Potency threshold: pIC50 >= {potent_threshold:.2f}")
    print(f"    Selectivity threshold: pred_sel >= {selective_threshold:.2f}")
    if len(sweet_spot) > 0:
        print(f"    Top 5 sweet spot molecules:")
        for _, row in sweet_spot.sort_values("pred_selectivity", ascending=False).head(5).iterrows():
            meas = "Yes" if row["has_measured_SYK"] else "No"
            print(f"      {row['molecule_chembl_id']}: ZAP70={row['pIC50_ZAP70']:.2f}, "
                  f"sel={row['pred_selectivity']:.2f}, meas_SYK={meas}")

    results = {
        "measured_potency_vs_selectivity": {
            "pearson_r": float(pr_meas), "spearman_r": float(sr_meas), "n": len(overlap_df),
        },
        "predicted_potency_vs_selectivity": {
            "pearson_r": float(pr_pred), "spearman_r": float(sr_pred), "n": len(ranking_df),
        },
        "sweet_spot_n": len(sweet_spot),
        "potency_threshold": float(potent_threshold),
        "selectivity_threshold": float(selective_threshold),
        "sweet_spot_molecules": [
            {"molecule_chembl_id": row["molecule_chembl_id"],
             "pIC50_ZAP70": float(row["pIC50_ZAP70"]),
             "pred_selectivity": float(row["pred_selectivity"]),
             "has_measured_SYK": bool(row["has_measured_SYK"])}
            for _, row in sweet_spot.sort_values("pred_selectivity", ascending=False).iterrows()
        ],
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("ZAP70 vs SYK Multi-Target Selectivity Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    raw = pd.read_csv(RAW_FILE)
    zap_data = load_target_molecules(raw, ZAP70_ID, "ZAP70")
    syk_data = load_target_molecules(raw, SYK_ID, "SYK")

    results = {"timestamp": datetime.now().isoformat(), "targets": {"ZAP70": ZAP70_ID, "SYK": SYK_ID}}

    # Phase 1: Overlap characterization
    overlap_df, overlap_stats = characterize_overlap(zap_data, syk_data)
    results["overlap_statistics"] = overlap_stats

    # Phase 2: Build models
    model_results, y_pred_zap, y_pred_syk, y_pred_sel = build_models(zap_data, syk_data, overlap_df)
    results["model_performance"] = model_results

    # Phase 3: Rank by selectivity
    ranking_df, syk_pred_quality, sel_pred_quality, zap_model, syk_model = rank_by_selectivity(
        zap_data, syk_data, overlap_df
    )
    results["ranking"] = {
        "syk_prediction_on_overlap": syk_pred_quality,
        "selectivity_prediction_on_overlap": sel_pred_quality,
        "top_selective_molecules": [
            {"rank": i + 1,
             "molecule_chembl_id": row["molecule_chembl_id"],
             "smiles": row["smiles"],
             "pIC50_ZAP70": float(row["pIC50_ZAP70"]),
             "pred_pIC50_SYK": float(row["pred_pIC50_SYK"]),
             "pred_selectivity": float(row["pred_selectivity"]),
             "has_measured_SYK": bool(row["has_measured_SYK"]),
             "measured_selectivity": float(row["measured_selectivity"]) if pd.notna(row.get("measured_selectivity")) else None}
            for i, (_, row) in enumerate(ranking_df.head(20).iterrows())
        ],
        "bottom_selective_molecules": [
            {"rank": len(ranking_df) - i,
             "molecule_chembl_id": row["molecule_chembl_id"],
             "smiles": row["smiles"],
             "pIC50_ZAP70": float(row["pIC50_ZAP70"]),
             "pred_pIC50_SYK": float(row["pred_pIC50_SYK"]),
             "pred_selectivity": float(row["pred_selectivity"]),
             "has_measured_SYK": bool(row["has_measured_SYK"]),
             "measured_selectivity": float(row["measured_selectivity"]) if pd.notna(row.get("measured_selectivity")) else None}
            for i, (_, row) in enumerate(ranking_df.tail(10).iloc[::-1].iterrows())
        ],
        "selectivity_distribution": {
            "mean": float(ranking_df.pred_selectivity.mean()),
            "std": float(ranking_df.pred_selectivity.std()),
            "median": float(ranking_df.pred_selectivity.median()),
            "n_zap70_selective": int((ranking_df.pred_selectivity > 0).sum()),
            "n_syk_selective": int((ranking_df.pred_selectivity < 0).sum()),
        }
    }

    # Phase 4: Feature analysis
    feature_results = analyze_selectivity_features(zap_data, syk_data, overlap_df)
    results["feature_analysis"] = feature_results

    # Phase 5: Potency-selectivity relationship
    ps_results = potency_selectivity_analysis(ranking_df, overlap_df)
    results["potency_selectivity"] = ps_results

    # Save
    elapsed = time.time() - t0
    results["runtime_seconds"] = round(elapsed, 1)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
