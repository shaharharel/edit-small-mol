#!/usr/bin/env python3
"""
ZAP70 (CHEMBL2803) Case Study v5 — Virtual Screening & Molecule Design.

Target: Tyrosine-protein kinase ZAP-70 (CHEMBL2803).

Building on v3 best model (MAE=0.555, Spr=0.713), this script provides:

Phase A: Score & Rank ChEMBL compounds NOT in training set
Phase B: Similarity search — find related molecules in ChEMBL kinase space
Phase C: SAR-guided enumeration — R-group expansion on top scaffolds
Phase D: MMP-inspired optimization — apply beneficial edits from MMP database
Phase E: Molecular property optimization — multi-objective filtering
Phase F: Comprehensive HTML report with ranked candidates

Usage:
    conda run -n quris python -u experiments/run_zap70_v5.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import os
import sqlite3
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem import MACCSkeys, Scaffolds, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from experiments.run_paper_evaluation import RESULTS_DIR, CACHE_DIR, DATA_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, get_cv_splits, compute_absolute_metrics,
    aggregate_cv_results, compute_fingerprints, compute_rdkit_descriptors,
    train_rf, train_xgboost, train_ridge,
    _tanimoto_kernel_matrix, N_JOBS, N_FOLDS, CV_SEED,
)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
CHEMBL_DB = PROJECT_ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
RESULTS_FILE = RESULTS_DIR / "zap70_v5_results.json"
REPORT_FILE = RESULTS_DIR / "zap70_v5_report.html"
TARGET_ID = "CHEMBL2803"  # ZAP70
TARGET_NAME = "ZAP70 (Tyrosine-protein kinase ZAP-70)"

# Best hyperparameters from v3 Optuna
BEST_XGB_PARAMS = {
    "max_depth": 6, "min_child_weight": 2,
    "subsample": 0.605, "colsample_bytree": 0.520,
    "learning_rate": 0.0197, "n_estimators": 749,
    "reg_alpha": 1.579, "reg_lambda": 7.313,
    "gamma": 0.014,
}
BEST_RF_PARAMS = {
    "n_estimators": 614, "max_depth": 16,
    "max_features": 0.3, "min_samples_leaf": 2,
    "min_samples_split": 3,
}


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def compute_multi_fp(smiles_list):
    """Compute the same FP types used in v3 grand ensemble."""
    fps = {}
    for fp_type in ["morgan", "atompair", "rdkit"]:
        fps[fp_type] = compute_fingerprints(smiles_list, fp_type, radius=2 if fp_type == "morgan" else 2, n_bits=2048)
    # ECFP6
    fps["ecfp6"] = compute_fingerprints(smiles_list, "morgan", radius=3, n_bits=2048)
    return fps


def train_ensemble(X_train_dict, y_train, X_test_dict):
    """Train the v3-style grand ensemble (top-5 mixed models)."""
    preds = {}
    models = {}

    # 1. XGB on AtomPair
    p, m = train_xgboost(X_train_dict["atompair"], y_train, X_test_dict["atompair"], **BEST_XGB_PARAMS)
    preds["xgb_atompair"] = p
    models["xgb_atompair"] = m

    # 2. RF on RDKit FP
    p, m = train_rf(X_train_dict["rdkit"], y_train, X_test_dict["rdkit"], **BEST_RF_PARAMS)
    preds["rf_rdkit"] = p
    models["rf_rdkit"] = m

    # 3. XGB on ECFP6
    p, m = train_xgboost(X_train_dict["ecfp6"], y_train, X_test_dict["ecfp6"], **BEST_XGB_PARAMS)
    preds["xgb_ecfp6"] = p
    models["xgb_ecfp6"] = m

    # 4. XGB on Morgan
    p, m = train_xgboost(X_train_dict["morgan"], y_train, X_test_dict["morgan"], **BEST_XGB_PARAMS)
    preds["xgb_morgan"] = p
    models["xgb_morgan"] = m

    # 5. RF on AtomPair
    p, m = train_rf(X_train_dict["atompair"], y_train, X_test_dict["atompair"], **BEST_RF_PARAMS)
    preds["rf_atompair"] = p
    models["rf_atompair"] = m

    # Ensemble = mean of all 5
    ensemble_pred = np.mean(list(preds.values()), axis=0)
    # Also compute std for uncertainty
    ensemble_std = np.std(list(preds.values()), axis=0)

    return ensemble_pred, ensemble_std, preds, models


def compute_druglikeness(smi):
    """Compute druglikeness properties for a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "Rings": Descriptors.RingCount(mol),
            "AromaticRings": Descriptors.NumAromaticRings(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "QED": Descriptors.qed(mol),
            "SA_score": _sa_score(mol),
            "Lipinski_violations": _lipinski_violations(mol),
        }
    except Exception:
        return {}


def _lipinski_violations(mol):
    """Count Lipinski rule of 5 violations."""
    v = 0
    if Descriptors.MolWt(mol) > 500: v += 1
    if Descriptors.MolLogP(mol) > 5: v += 1
    if Descriptors.NumHAcceptors(mol) > 10: v += 1
    if Descriptors.NumHDonors(mol) > 5: v += 1
    return v


def _sa_score(mol):
    """Synthetic accessibility score (1=easy, 10=hard). Simplified version."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score', 'sascorer.py')
        if os.path.exists(sa_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("sascorer", sa_path)
            sascorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sascorer)
            return sascorer.calculateScore(mol)
    except Exception:
        pass
    # Fallback: rough estimate based on complexity
    n_rings = Descriptors.RingCount(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_heavy = mol.GetNumHeavyAtoms()
    return min(10, max(1, 1 + n_heavy * 0.05 + n_rings * 0.3 + n_stereo * 0.5))


def tanimoto_similarity(smi1, smi2, radius=2, n_bits=2048):
    """Compute Tanimoto similarity between two SMILES."""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Score & Rank ChEMBL Compounds
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(mol_data, results):
    """Score all ChEMBL compounds for ZAP70 not in our training set."""
    print("\n" + "=" * 70)
    print("PHASE A: Score & Rank ChEMBL Compounds")
    print("=" * 70)

    if not CHEMBL_DB.exists():
        print("  ChEMBL database not found, skipping.")
        results["phase_a"] = {"error": "ChEMBL DB not found"}
        return results

    # 1. Query ChEMBL for all ZAP70 compounds
    db = sqlite3.connect(str(CHEMBL_DB))
    query = """
        SELECT DISTINCT cs.canonical_smiles, md.chembl_id as mol_chembl_id,
               a.standard_type, a.standard_value, a.standard_units,
               a.pchembl_value
        FROM activities a
        JOIN assays ass ON a.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE td.chembl_id = ?
        AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
        AND a.standard_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
    """
    chembl_df = pd.read_sql_query(query, db, params=[TARGET_ID])
    db.close()

    print(f"  ChEMBL activities: {len(chembl_df)}, unique molecules: {chembl_df['mol_chembl_id'].nunique()}")

    # 2. Identify molecules NOT in training set
    train_smiles = set(mol_data["smiles"].values)
    train_ids = set(mol_data["molecule_chembl_id"].values)

    # Canonicalize for matching
    train_canonical = set()
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    chembl_df["canonical"] = chembl_df["canonical_smiles"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    new_mols = chembl_df[~chembl_df["canonical"].isin(train_canonical)].copy()
    new_mols = new_mols.drop_duplicates(subset=["canonical"])

    # Also get molecules in training for validation
    known_mols = chembl_df[chembl_df["canonical"].isin(train_canonical)].copy()
    known_mols = known_mols.drop_duplicates(subset=["canonical"])

    print(f"  Known (in training): {len(known_mols)}")
    print(f"  New (not in training): {len(new_mols)}")

    if len(new_mols) == 0:
        print("  No new molecules to score.")
        results["phase_a"] = {"n_new": 0, "message": "No new molecules found"}
        return results

    # 3. Train ensemble on ALL training data
    print("\n  Training ensemble on full training set...")
    train_smiles_list = mol_data["smiles"].tolist()
    y_train = mol_data["pIC50"].values

    # Also prepare new molecules
    new_smiles_list = new_mols["canonical"].tolist()
    all_smiles = train_smiles_list + new_smiles_list

    # Compute FPs for all molecules at once
    fps_all = compute_multi_fp(all_smiles)
    n_train = len(train_smiles_list)

    X_train_dict = {k: v[:n_train] for k, v in fps_all.items()}
    X_new_dict = {k: v[n_train:] for k, v in fps_all.items()}

    # Train ensemble
    new_preds, new_stds, individual_preds, models = train_ensemble(
        X_train_dict, y_train, X_new_dict
    )

    # 4. Compute nearest neighbor similarity to training set
    print("  Computing similarity to training set...")
    train_morgan = X_train_dict["morgan"]
    new_morgan = X_new_dict["morgan"]

    nn_sims = []
    nn_smiles = []
    for i in range(len(new_smiles_list)):
        sims = _tanimoto_kernel_matrix(new_morgan[i:i+1], train_morgan)[0]
        best_idx = np.argmax(sims)
        nn_sims.append(float(sims[best_idx]))
        nn_smiles.append(train_smiles_list[best_idx])

    # 5. Compute druglikeness
    print("  Computing druglikeness properties...")
    drug_props = [compute_druglikeness(smi) for smi in new_smiles_list]

    # 6. Build ranked candidate table
    candidates = []
    for i in range(len(new_smiles_list)):
        # Conformal interval (from v3: 90% coverage ≈ ±1.35)
        ci_90 = new_stds[i] * 2.5  # Approximate 90% CI from ensemble spread
        dp = drug_props[i] if drug_props[i] else {}

        cand = {
            "rank": 0,  # filled later
            "smiles": new_smiles_list[i],
            "chembl_id": new_mols.iloc[i]["mol_chembl_id"],
            "predicted_pIC50": round(float(new_preds[i]), 3),
            "uncertainty": round(float(new_stds[i]), 3),
            "ci_90_lower": round(float(new_preds[i] - ci_90), 3),
            "ci_90_upper": round(float(new_preds[i] + ci_90), 3),
            "nn_similarity": round(nn_sims[i], 3),
            "nn_train_smiles": nn_smiles[i],
            "known_value": None,  # Fill if available from ChEMBL pchembl_value
            "MW": dp.get("MW", None),
            "LogP": dp.get("LogP", None),
            "TPSA": dp.get("TPSA", None),
            "QED": dp.get("QED", None),
            "SA_score": dp.get("SA_score", None),
            "Lipinski_violations": dp.get("Lipinski_violations", None),
        }

        # Check if ChEMBL has a pchembl_value
        matching = new_mols[new_mols["canonical"] == new_smiles_list[i]]
        pchembl_vals = matching["pchembl_value"].dropna()
        if len(pchembl_vals) > 0:
            cand["known_value"] = round(float(pchembl_vals.mean()), 2)

        candidates.append(cand)

    # Sort by predicted pIC50 (highest first)
    candidates.sort(key=lambda x: x["predicted_pIC50"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    # 7. Summary statistics
    pred_vals = np.array([c["predicted_pIC50"] for c in candidates])
    n_potent = sum(1 for c in candidates if c["predicted_pIC50"] >= 7.0)
    n_moderate = sum(1 for c in candidates if 6.0 <= c["predicted_pIC50"] < 7.0)
    n_weak = sum(1 for c in candidates if c["predicted_pIC50"] < 6.0)

    # Filter high-confidence potent
    high_conf_potent = [c for c in candidates
                        if c["predicted_pIC50"] >= 7.0 and c["nn_similarity"] >= 0.3]

    print(f"\n  === Screening Results ===")
    print(f"  Total new candidates: {len(candidates)}")
    print(f"  Predicted potent (pIC50≥7): {n_potent}")
    print(f"  Predicted moderate (6-7): {n_moderate}")
    print(f"  Predicted weak (<6): {n_weak}")
    print(f"  High-confidence potent (pIC50≥7, sim≥0.3): {len(high_conf_potent)}")

    # Print top 20
    print(f"\n  Top 20 candidates:")
    print(f"  {'Rank':>4} {'ChEMBL ID':>14} {'Pred pIC50':>10} {'±Unc':>6} {'NN Sim':>7} {'Known':>6} {'QED':>5}")
    for c in candidates[:20]:
        known_str = f"{c['known_value']:.1f}" if c["known_value"] else "   -"
        qed_str = f"{c['QED']:.2f}" if c["QED"] else "  -"
        print(f"  {c['rank']:4d} {c['chembl_id']:>14s} {c['predicted_pIC50']:10.2f} "
              f"±{c['uncertainty']:.2f} {c['nn_similarity']:7.3f} {known_str:>6s} {qed_str:>5s}")

    # Validate predictions where we have known values
    with_known = [c for c in candidates if c["known_value"] is not None]
    if with_known:
        known_true = np.array([c["known_value"] for c in with_known])
        known_pred = np.array([c["predicted_pIC50"] for c in with_known])
        from scipy.stats import spearmanr as spr
        mae = np.mean(np.abs(known_true - known_pred))
        sr, _ = spr(known_true, known_pred) if len(known_true) > 2 else (0, 1)
        print(f"\n  Validation on {len(with_known)} molecules with known pChEMBL values:")
        print(f"  MAE={mae:.3f}, Spearman={sr:.3f}")

    results["phase_a"] = {
        "n_candidates": len(candidates),
        "n_potent": n_potent,
        "n_moderate": n_moderate,
        "n_weak": n_weak,
        "n_high_conf_potent": len(high_conf_potent),
        "candidates": candidates[:50],  # Top 50 for report
        "validation": {
            "n_with_known": len(with_known),
            "mae": float(np.mean(np.abs(
                np.array([c["known_value"] for c in with_known]) -
                np.array([c["predicted_pIC50"] for c in with_known])
            ))) if with_known else None,
        },
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Similarity Search in Kinase Space
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(mol_data, results):
    """Search for related molecules from similar kinase targets."""
    print("\n" + "=" * 70)
    print("PHASE B: Kinase Similarity Search")
    print("=" * 70)

    if not CHEMBL_DB.exists():
        print("  ChEMBL database not found, skipping.")
        results["phase_b"] = {"error": "ChEMBL DB not found"}
        return results

    # Related kinases (from v2 transfer learning)
    RELATED_KINASES = {
        "ITK": "CHEMBL3009",    # Most similar (Spr=0.355)
        "SYK": "CHEMBL2599",    # Same family
        "FYN": "CHEMBL1841",    # Src family
        "RAF1": "CHEMBL1906",   # MAP kinase pathway
        "MEK1": "CHEMBL399",    # Downstream of ZAP70
        "MEK2": "CHEMBL4045",   # Downstream of ZAP70
    }

    db = sqlite3.connect(str(CHEMBL_DB))

    # Get potent compounds (pchembl >= 7) from related kinases
    kinase_compounds = {}
    for name, chembl_id in RELATED_KINASES.items():
        query = """
            SELECT DISTINCT cs.canonical_smiles, md.chembl_id, a.pchembl_value
            FROM activities a
            JOIN assays ass ON a.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN molecule_dictionary md ON a.molregno = md.molregno
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE td.chembl_id = ?
            AND a.pchembl_value >= 7.0
            AND cs.canonical_smiles IS NOT NULL
            LIMIT 500
        """
        df = pd.read_sql_query(query, db, params=[chembl_id])
        kinase_compounds[name] = df
        print(f"  {name} ({chembl_id}): {len(df)} potent compounds")

    db.close()

    # Find compounds from related kinases that are NOVEL to our ZAP70 set
    train_smiles = set(mol_data["smiles"].values)
    train_canonical = set()
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    # Also exclude compounds already scored in Phase A
    phase_a_smiles = set()
    if "phase_a" in results and "candidates" in results["phase_a"]:
        phase_a_smiles = set(c["smiles"] for c in results["phase_a"]["candidates"])

    novel_kinase_mols = []
    for kinase_name, df in kinase_compounds.items():
        for _, row in df.iterrows():
            smi = row["canonical_smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol)
            if can_smi in train_canonical or can_smi in phase_a_smiles:
                continue
            novel_kinase_mols.append({
                "smiles": can_smi,
                "chembl_id": row["chembl_id"],
                "source_kinase": kinase_name,
                "source_pchembl": float(row["pchembl_value"]) if pd.notna(row["pchembl_value"]) else None,
            })

    # Deduplicate
    seen = set()
    unique_kinase_mols = []
    for m in novel_kinase_mols:
        if m["smiles"] not in seen:
            seen.add(m["smiles"])
            unique_kinase_mols.append(m)

    print(f"\n  Novel kinase molecules (not in training or Phase A): {len(unique_kinase_mols)}")

    if len(unique_kinase_mols) == 0:
        results["phase_b"] = {"n_novel": 0}
        return results

    # Score with our ensemble
    print("  Scoring novel kinase molecules with ZAP70 ensemble...")
    train_smiles_list = mol_data["smiles"].tolist()
    y_train = mol_data["pIC50"].values

    novel_smiles_list = [m["smiles"] for m in unique_kinase_mols]
    all_smiles = train_smiles_list + novel_smiles_list
    fps_all = compute_multi_fp(all_smiles)
    n_train = len(train_smiles_list)

    X_train_dict = {k: v[:n_train] for k, v in fps_all.items()}
    X_novel_dict = {k: v[n_train:] for k, v in fps_all.items()}

    novel_preds, novel_stds, _, _ = train_ensemble(X_train_dict, y_train, X_novel_dict)

    # Compute similarity to training set
    train_morgan = X_train_dict["morgan"]
    novel_morgan = X_novel_dict["morgan"]

    # Build results
    kinase_candidates = []
    for i, m in enumerate(unique_kinase_mols):
        sims = _tanimoto_kernel_matrix(novel_morgan[i:i+1], train_morgan)[0]
        best_idx = np.argmax(sims)
        nn_sim = float(sims[best_idx])

        dp = compute_druglikeness(m["smiles"])
        kinase_candidates.append({
            "rank": 0,
            "smiles": m["smiles"],
            "chembl_id": m["chembl_id"],
            "source_kinase": m["source_kinase"],
            "source_pchembl": m["source_pchembl"],
            "predicted_map3k8_pIC50": round(float(novel_preds[i]), 3),
            "uncertainty": round(float(novel_stds[i]), 3),
            "nn_similarity_to_map3k8": round(nn_sim, 3),
            "QED": round(dp.get("QED", 0), 3) if dp else None,
            "MW": round(dp.get("MW", 0), 1) if dp else None,
        })

    # Sort by predicted pIC50
    kinase_candidates.sort(key=lambda x: x["predicted_map3k8_pIC50"], reverse=True)
    for i, c in enumerate(kinase_candidates):
        c["rank"] = i + 1

    n_repurpose = sum(1 for c in kinase_candidates if c["predicted_map3k8_pIC50"] >= 6.5)
    print(f"\n  Kinase compounds predicted active on ZAP70 (pIC50≥6.5): {n_repurpose}")
    print(f"\n  Top 15 repurposing candidates:")
    print(f"  {'Rank':>4} {'Source':>6} {'Pred':>5} {'±Unc':>5} {'Sim':>5} {'Source pIC50':>11}")
    for c in kinase_candidates[:15]:
        src_val = f"{c['source_pchembl']:.1f}" if c["source_pchembl"] else "  -"
        print(f"  {c['rank']:4d} {c['source_kinase']:>6s} {c['predicted_map3k8_pIC50']:5.2f} "
              f"±{c['uncertainty']:.2f} {c['nn_similarity_to_map3k8']:.3f} {src_val:>11s}")

    results["phase_b"] = {
        "n_novel_kinase_mols": len(unique_kinase_mols),
        "n_predicted_active": n_repurpose,
        "kinase_sources": {k: len(v) for k, v in kinase_compounds.items()},
        "candidates": kinase_candidates[:30],
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: SAR-Guided Enumeration
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(mol_data, results):
    """Enumerate R-group variations on top scaffolds."""
    print("\n" + "=" * 70)
    print("PHASE C: SAR-Guided Enumeration")
    print("=" * 70)

    # Identify top scaffolds from potent molecules
    potent = mol_data[mol_data["pIC50"] >= 7.0].copy()
    print(f"  Potent molecules (pIC50≥7): {len(potent)}")

    # Get Murcko scaffolds
    scaffold_groups = defaultdict(list)
    for _, row in potent.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_smi = Chem.MolToSmiles(scaf)
                scaffold_groups[scaf_smi].append({
                    "smiles": row["smiles"],
                    "pIC50": row["pIC50"],
                })
            except Exception:
                continue

    # Top scaffolds by count
    top_scaffolds = sorted(scaffold_groups.items(), key=lambda x: -len(x[1]))[:5]
    print(f"  Top scaffolds with potent molecules:")
    for scaf, mols in top_scaffolds:
        mean_pic = np.mean([m["pIC50"] for m in mols])
        print(f"    {scaf[:60]}... (n={len(mols)}, mean pIC50={mean_pic:.2f})")

    # BRICS decomposition of top molecules
    print("\n  BRICS decomposition of top molecules...")
    all_fragments = Counter()
    for _, row in potent.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                for f in frags:
                    all_fragments[f] += 1
            except Exception:
                continue

    print(f"  Unique BRICS fragments: {len(all_fragments)}")
    print(f"  Top 10 fragments:")
    for frag, count in all_fragments.most_common(10):
        print(f"    {frag} (n={count})")

    # SAR-guided R-group enumeration
    # Use SMARTS patterns from v4 that boost potency
    # From v4: nitrile (+0.91), fluorine (+0.81), pyridine (+1.20), piperidine (+1.06)
    beneficial_groups = {
        "nitrile": "[C:1]#N",
        "fluorine": "[F:1]",
        "chlorine": "[Cl:1]",
        "pyridine": "c1ccncc1",
    }

    # For each top scaffold, try BRICS-based recombination
    print("\n  Generating analogs via BRICS recombination...")
    generated_mols = set()
    train_canonical = set()
    for smi in mol_data["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    # Take top 20 potent molecules and try BRICS rebuild
    top_mols = potent.nlargest(20, "pIC50")
    all_brics_frags = set()
    for _, row in top_mols.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_brics_frags.update(frags)
            except Exception:
                continue

    # Limited BRICS build from fragments of potent molecules
    if all_brics_frags:
        print(f"  BRICS fragments from top-20: {len(all_brics_frags)}")
        try:
            # Convert fragment SMILES to mol objects for BRICSBuild
            frag_mols = []
            for frag_smi in list(all_brics_frags)[:20]:
                fmol = Chem.MolFromSmiles(frag_smi)
                if fmol is not None:
                    frag_mols.append(fmol)
            # Use generator with limit
            builder = BRICS.BRICSBuild(frag_mols)
            for count, mol in enumerate(builder):
                if count >= 500:  # Cap at 500 generated molecules
                    break
                try:
                    smi = Chem.MolToSmiles(mol)
                    if smi not in train_canonical and len(smi) < 200:
                        # Basic druglikeness filter
                        if mol.GetNumHeavyAtoms() <= 50 and mol.GetNumHeavyAtoms() >= 15:
                            generated_mols.add(smi)
                except Exception:
                    continue
            print(f"  Generated {len(generated_mols)} novel molecules via BRICS recombination")
        except Exception as e:
            print(f"  BRICS build failed: {e}")

    if not generated_mols:
        print("  No novel molecules generated.")
        results["phase_c"] = {"n_generated": 0, "completed": True}
        save_results(results)
        return results

    # Score generated molecules
    print(f"\n  Scoring {len(generated_mols)} generated molecules...")
    gen_smiles_list = list(generated_mols)
    train_smiles_list = mol_data["smiles"].tolist()
    y_train = mol_data["pIC50"].values

    all_smiles = train_smiles_list + gen_smiles_list
    fps_all = compute_multi_fp(all_smiles)
    n_train = len(train_smiles_list)

    X_train_dict = {k: v[:n_train] for k, v in fps_all.items()}
    X_gen_dict = {k: v[n_train:] for k, v in fps_all.items()}

    gen_preds, gen_stds, _, _ = train_ensemble(X_train_dict, y_train, X_gen_dict)

    # Compute similarity + druglikeness
    train_morgan = X_train_dict["morgan"]
    gen_morgan = X_gen_dict["morgan"]

    gen_candidates = []
    for i, smi in enumerate(gen_smiles_list):
        sims = _tanimoto_kernel_matrix(gen_morgan[i:i+1], train_morgan)[0]
        best_idx = np.argmax(sims)
        nn_sim = float(sims[best_idx])
        dp = compute_druglikeness(smi)

        gen_candidates.append({
            "rank": 0,
            "smiles": smi,
            "predicted_pIC50": round(float(gen_preds[i]), 3),
            "uncertainty": round(float(gen_stds[i]), 3),
            "nn_similarity": round(nn_sim, 3),
            "nn_train_smiles": train_smiles_list[best_idx],
            "QED": round(dp.get("QED", 0), 3) if dp else None,
            "MW": round(dp.get("MW", 0), 1) if dp else None,
            "SA_score": round(dp.get("SA_score", 3), 2) if dp else None,
            "Lipinski_violations": dp.get("Lipinski_violations"),
        })

    # Sort by predicted pIC50
    gen_candidates.sort(key=lambda x: x["predicted_pIC50"], reverse=True)
    for i, c in enumerate(gen_candidates):
        c["rank"] = i + 1

    # Filter for drug-like
    druglike = [c for c in gen_candidates
                if c.get("Lipinski_violations", 5) <= 1 and
                c.get("QED", 0) >= 0.3 and
                c.get("SA_score", 10) <= 5]

    n_potent_gen = sum(1 for c in gen_candidates if c["predicted_pIC50"] >= 7.0)
    print(f"\n  === Generated Molecule Results ===")
    print(f"  Total generated: {len(gen_candidates)}")
    print(f"  Predicted potent (pIC50≥7): {n_potent_gen}")
    print(f"  Drug-like (Lipinski≤1, QED≥0.3, SA≤5): {len(druglike)}")

    print(f"\n  Top 15 generated candidates:")
    print(f"  {'Rank':>4} {'Pred pIC50':>10} {'±Unc':>5} {'Sim':>5} {'QED':>5} {'SA':>5}")
    for c in gen_candidates[:15]:
        qed_str = f"{c['QED']:.2f}" if c["QED"] else "  -"
        sa_str = f"{c['SA_score']:.1f}" if c["SA_score"] else "  -"
        print(f"  {c['rank']:4d} {c['predicted_pIC50']:10.2f} ±{c['uncertainty']:.2f} "
              f"{c['nn_similarity']:.3f} {qed_str:>5s} {sa_str:>5s}")

    results["phase_c"] = {
        "n_brics_fragments": len(all_brics_frags),
        "n_generated": len(gen_candidates),
        "n_potent": n_potent_gen,
        "n_druglike": len(druglike),
        "top_scaffolds": [{"scaffold": s, "n_mols": len(m), "mean_pIC50": round(np.mean([x["pIC50"] for x in m]), 2)}
                          for s, m in top_scaffolds],
        "top_fragments": [{"fragment": f, "count": c} for f, c in all_fragments.most_common(10)],
        "candidates": gen_candidates[:30],
        "druglike_candidates": druglike[:20] if druglike else [],
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: MMP-Inspired Optimization
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(mol_data, results):
    """Apply beneficial edits from MMP database to top molecules."""
    print("\n" + "=" * 70)
    print("PHASE D: MMP-Inspired Optimization")
    print("=" * 70)

    # Load MMP pairs from the shared dataset to find beneficial edits
    print("  Loading MMP edits from shared pairs dataset...")
    pairs = pd.read_csv(DATA_DIR / "shared_pairs_deduped.csv",
                        usecols=["mol_a", "mol_b", "edit_smiles", "delta", "is_within_assay", "target_chembl_id"])

    # Focus on within-assay pairs for cleaner signal
    within = pairs[pairs["is_within_assay"] == True].copy()
    print(f"  Within-assay pairs: {len(within):,}")

    # Find edits that consistently IMPROVE potency (positive delta = B more potent than A)
    edit_stats = within.groupby("edit_smiles").agg(
        n_pairs=("delta", "count"),
        mean_delta=("delta", "mean"),
        std_delta=("delta", "std"),
        n_targets=("target_chembl_id", "nunique"),
    ).reset_index()

    # Filter: edits seen in ≥10 pairs, positive mean delta (consistently beneficial)
    beneficial_edits = edit_stats[
        (edit_stats["n_pairs"] >= 10) &
        (edit_stats["mean_delta"] > 0.3) &
        (edit_stats["n_targets"] >= 2)
    ].sort_values("mean_delta", ascending=False)

    print(f"  Beneficial edits (n≥10, Δ>0.3, targets≥2): {len(beneficial_edits)}")
    print(f"\n  Top 15 beneficial edits:")
    print(f"  {'Edit SMILES':>50} {'N':>5} {'Mean Δ':>7} {'Targets':>7}")
    for _, row in beneficial_edits.head(15).iterrows():
        edit_smi = str(row["edit_smiles"])
        if len(edit_smi) > 50:
            edit_smi = edit_smi[:47] + "..."
        print(f"  {edit_smi:>50s} {row['n_pairs']:5d} {row['mean_delta']:+7.2f} {row['n_targets']:7d}")

    # Take top molecules as starting points
    top_mols = mol_data.nlargest(15, "pIC50")
    moderate_mols = mol_data[(mol_data["pIC50"] >= 5.5) & (mol_data["pIC50"] < 7.0)].sample(
        min(10, len(mol_data[(mol_data["pIC50"] >= 5.5) & (mol_data["pIC50"] < 7.0)])),
        random_state=42
    )
    seed_mols = pd.concat([top_mols, moderate_mols]).drop_duplicates(subset=["smiles"])

    print(f"\n  Seed molecules for optimization: {len(seed_mols)}")

    # For each seed molecule, look for applicable edits in the MMP database
    # An edit is applicable if the seed molecule contains the leaving fragment
    print("  Finding applicable edits for seed molecules...")
    optimization_results = []

    for _, seed_row in seed_mols.iterrows():
        seed_smi = seed_row["smiles"]
        seed_pic = seed_row["pIC50"]
        seed_mol = Chem.MolFromSmiles(seed_smi)
        if seed_mol is None:
            continue

        # Find all MMP pairs where this molecule (or similar) is mol_a
        # Since ZAP70 mols have 0 direct MMPs, we look for
        # structural analogs in the MMP database
        seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, nBits=2048)

        # Check if any pair in the dataset has a similar mol_a
        # This is too slow for 1.7M pairs, so we use a smarter approach:
        # Apply known beneficial edits as SMIRKS transforms
        for _, edit_row in beneficial_edits.head(30).iterrows():
            edit_smi = edit_row["edit_smiles"]
            # Try to interpret edit as leaving>>incoming fragment
            if ">>" in str(edit_smi):
                parts = str(edit_smi).split(">>")
                if len(parts) == 2:
                    leaving, incoming = parts
                    leaving_mol = Chem.MolFromSmiles(leaving)
                    incoming_mol = Chem.MolFromSmiles(incoming)
                    if leaving_mol and incoming_mol:
                        # Check if seed contains the leaving group
                        if seed_mol.HasSubstructMatch(leaving_mol):
                            optimization_results.append({
                                "seed_smiles": seed_smi,
                                "seed_pIC50": round(float(seed_pic), 2),
                                "edit": str(edit_smi),
                                "expected_delta": round(float(edit_row["mean_delta"]), 3),
                                "expected_new_pIC50": round(float(seed_pic + edit_row["mean_delta"]), 3),
                                "edit_confidence": round(float(edit_row["n_pairs"]), 0),
                                "edit_n_targets": int(edit_row["n_targets"]),
                            })

    print(f"\n  Applicable optimization suggestions: {len(optimization_results)}")

    # Score by expected improvement
    optimization_results.sort(key=lambda x: x["expected_new_pIC50"], reverse=True)

    if optimization_results:
        print(f"\n  Top 15 optimization suggestions:")
        print(f"  {'Seed pIC50':>10} {'Expected':>8} {'Δ':>6} {'Edit':>40} {'Confidence':>10}")
        for o in optimization_results[:15]:
            edit_str = o["edit"][:40]
            print(f"  {o['seed_pIC50']:10.2f} {o['expected_new_pIC50']:8.2f} "
                  f"{o['expected_delta']:+6.2f} {edit_str:>40s} {o['edit_confidence']:10.0f}")

    results["phase_d"] = {
        "n_beneficial_edits": len(beneficial_edits),
        "n_seed_molecules": len(seed_mols),
        "n_applicable_optimizations": len(optimization_results),
        "top_edits": beneficial_edits.head(20).to_dict(orient="records"),
        "optimization_suggestions": optimization_results[:30],
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase E: Multi-Objective Filtering & Final Rankings
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_e(mol_data, results):
    """Combine all candidates with multi-objective filtering."""
    print("\n" + "=" * 70)
    print("PHASE E: Multi-Objective Ranking & Final Candidates")
    print("=" * 70)

    all_candidates = []

    # Collect from Phase A
    if "phase_a" in results and "candidates" in results["phase_a"]:
        for c in results["phase_a"]["candidates"]:
            c["source"] = "ChEMBL (untested)"
            all_candidates.append(c)

    # Collect from Phase B
    if "phase_b" in results and "candidates" in results["phase_b"]:
        for c in results["phase_b"]["candidates"]:
            c["source"] = f"Kinase ({c.get('source_kinase', '?')})"
            c["predicted_pIC50"] = c.get("predicted_map3k8_pIC50", 0)
            all_candidates.append(c)

    # Collect from Phase C
    if "phase_c" in results and "candidates" in results["phase_c"]:
        for c in results["phase_c"]["candidates"]:
            c["source"] = "BRICS generated"
            all_candidates.append(c)

    print(f"  Total candidates from all phases: {len(all_candidates)}")

    if not all_candidates:
        results["phase_e"] = {"message": "No candidates to rank"}
        return results

    # Multi-objective scoring
    # Weights: predicted potency (50%), druglikeness/QED (20%),
    #          novelty/similarity (15%), uncertainty (15% penalty)
    for c in all_candidates:
        potency_score = min(10, max(0, c.get("predicted_pIC50", 0) - 4.0)) / 6.0  # 0-1
        qed_score = c.get("QED", 0.5) if c.get("QED") else 0.5
        # Novel but not too dissimilar
        sim = c.get("nn_similarity", c.get("nn_similarity_to_map3k8", 0.3))
        novelty_score = 1.0 - abs(sim - 0.5) * 2  # Peaks at sim=0.5
        unc = c.get("uncertainty", 0.5)
        unc_penalty = max(0, 1.0 - unc)  # Lower uncertainty = higher score

        c["composite_score"] = round(
            0.50 * potency_score +
            0.20 * qed_score +
            0.15 * novelty_score +
            0.15 * unc_penalty,
            4
        )

    # Sort by composite score
    all_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, c in enumerate(all_candidates):
        c["final_rank"] = i + 1

    # Print final ranking
    print(f"\n  === FINAL SCREENING CANDIDATES ===")
    print(f"  {'Rank':>4} {'Source':>20} {'Pred pIC50':>10} {'Score':>6} {'QED':>5}")
    for c in all_candidates[:25]:
        src = c.get("source", "?")[:20]
        qed_str = f"{c['QED']:.2f}" if c.get("QED") else "  -"
        print(f"  {c['final_rank']:4d} {src:>20s} {c.get('predicted_pIC50', 0):10.2f} "
              f"{c['composite_score']:.3f} {qed_str:>5s}")

    results["phase_e"] = {
        "n_total_candidates": len(all_candidates),
        "top_candidates": all_candidates[:50],
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase F: Comprehensive HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_f(mol_data, results):
    """Generate comprehensive virtual screening HTML report."""
    print("\n" + "=" * 70)
    print("PHASE F: Generating Virtual Screening Report")
    print("=" * 70)

    # Clinical context for ZAP70
    target_info = {
        "name": "ZAP70 (Tyrosine-protein kinase ZAP-70)",
        "full_name": "Zeta-chain-associated protein kinase 70",
        "gene": "ZAP70",
        "uniprot": "P43403",
        "function": "Non-receptor tyrosine kinase essential for T-cell receptor (TCR) signaling",
        "disease_relevance": [
            "Autoimmune diseases (T-cell mediated inflammation)",
            "Transplant rejection (T-cell activation)",
            "CLL prognosis (ZAP-70 expression = aggressive disease)",
            "Severe combined immunodeficiency (ZAP-70 deficiency causes SCID)",
        ],
        "clinical_significance": (
            "ZAP-70 is a critical proximal kinase in TCR signaling. Inhibiting ZAP-70 "
            "selectively blocks T-cell activation without affecting other immune cells. "
            "ZAP-70 expression is a prognostic biomarker in CLL, where it correlates with "
            "unmutated IgVH status and aggressive disease course."
        ),
    }

    html = []
    html.append("<!DOCTYPE html><html><head>")
    html.append("<title>ZAP70 Virtual Screening Report</title>")
    html.append("""<style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto;
               padding: 20px; background: #f5f5f5; color: #333; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #1a5276, #2980b9); color: white;
                  padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 28px; }
        .header p { margin: 5px 0 0; opacity: 0.9; font-size: 16px; }
        .section { background: white; padding: 25px; border-radius: 8px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }
        h2 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 8px; }
        h3 { color: #2c3e50; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
        th { background: #2c3e50; color: white; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f0f7ff; }
        .metric-box { display: inline-block; background: #eaf2f8; padding: 15px 25px;
                      border-radius: 8px; margin: 5px; text-align: center; min-width: 120px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #1a5276; }
        .metric-label { font-size: 12px; color: #666; margin-top: 4px; }
        .highlight { background: #d5f5e3; padding: 2px 6px; border-radius: 3px; }
        .warning { background: #fdebd0; padding: 2px 6px; border-radius: 3px; }
        .danger { background: #fadbd8; padding: 2px 6px; border-radius: 3px; }
        .good { color: #27ae60; font-weight: bold; }
        .smi { font-family: monospace; font-size: 11px; word-break: break-all;
               max-width: 300px; display: inline-block; }
        .note { background: #fef9e7; padding: 15px; border-left: 4px solid #f39c12;
                border-radius: 4px; margin: 15px 0; }
        .clinical { background: #eaf2f8; padding: 15px; border-left: 4px solid #2980b9;
                    border-radius: 4px; margin: 15px 0; }
        ul { margin: 5px 0; padding-left: 20px; }
    </style>""")
    html.append("</head><body>")

    # Header
    html.append('<div class="header">')
    html.append(f'<h1>Virtual Screening Report: {TARGET_NAME}</h1>')
    html.append(f'<p>Computational screening and molecule design for {target_info["full_name"]}</p>')
    html.append(f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
                f'Training set: {len(mol_data)} molecules | '
                f'Best model MAE: 0.555 pIC50</p>')
    html.append('</div>')

    # Section 1: Target Overview
    html.append('<div class="section">')
    html.append('<h2>1. Target Overview</h2>')
    html.append('<div class="clinical">')
    html.append(f'<strong>{target_info["name"]}</strong> ({target_info["gene"]}, UniProt: {target_info["uniprot"]})<br>')
    html.append(f'{target_info["function"]}<br><br>')
    html.append(f'<strong>Clinical Significance:</strong> {target_info["clinical_significance"]}')
    html.append('</div>')
    html.append('<h3>Disease Relevance</h3><ul>')
    for d in target_info["disease_relevance"]:
        html.append(f'<li>{d}</li>')
    html.append('</ul>')

    # Training set summary
    html.append('<h3>Training Data Summary</h3>')
    html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
    for label, value in [
        ("Molecules", str(len(mol_data))),
        ("pIC50 Range", f"{mol_data['pIC50'].min():.1f}–{mol_data['pIC50'].max():.1f}"),
        ("Mean pIC50", f"{mol_data['pIC50'].mean():.2f}"),
        ("Model MAE", "0.555"),
        ("Spearman", "0.713"),
        ("Noise Floor", "≈0.461"),
    ]:
        html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>')
    html.append('</div>')
    html.append('</div>')

    # Section 2: Phase A — ChEMBL Screening
    if "phase_a" in results and results["phase_a"].get("completed"):
        pa = results["phase_a"]
        html.append('<div class="section">')
        html.append('<h2>2. ChEMBL Compound Screening</h2>')
        html.append(f'<p>Scored <strong>{pa["n_candidates"]}</strong> ChEMBL compounds not in training set.</p>')

        html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
        for label, value in [
            ("Total Screened", str(pa["n_candidates"])),
            ("Predicted Potent", f'{pa["n_potent"]} (pIC50≥7)'),
            ("Moderate", f'{pa["n_moderate"]} (6-7)'),
            ("High Confidence", str(pa.get("n_high_conf_potent", 0))),
        ]:
            html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div></div>')
        html.append('</div>')

        if pa.get("validation", {}).get("n_with_known", 0) > 0:
            html.append(f'<div class="note"><strong>Validation:</strong> '
                        f'{pa["validation"]["n_with_known"]} molecules had known pChEMBL values. '
                        f'Prediction MAE = {pa["validation"]["mae"]:.3f}</div>')

        # Top candidates table
        if pa.get("candidates"):
            html.append('<h3>Top Screening Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>ChEMBL ID</th><th>Predicted pIC50</th>'
                        '<th>±Uncertainty</th><th>NN Similarity</th><th>Known Value</th>'
                        '<th>QED</th><th>SMILES</th></tr>')
            for c in pa["candidates"][:30]:
                known = f"{c['known_value']:.1f}" if c.get("known_value") else "—"
                qed = f"{c['QED']:.2f}" if c.get("QED") else "—"
                potency_class = "highlight" if c["predicted_pIC50"] >= 7.0 else ""
                html.append(f'<tr><td>{c["rank"]}</td><td>{c["chembl_id"]}</td>'
                            f'<td class="{potency_class}">{c["predicted_pIC50"]:.2f}</td>'
                            f'<td>±{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity"]:.3f}</td><td>{known}</td>'
                            f'<td>{qed}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 3: Phase B — Kinase Repurposing
    if "phase_b" in results and results["phase_b"].get("completed"):
        pb = results["phase_b"]
        html.append('<div class="section">')
        html.append('<h2>3. Kinase Compound Repurposing</h2>')
        html.append(f'<p>Screened potent compounds from related kinases for potential ZAP70 activity.</p>')

        html.append('<h3>Source Kinases</h3>')
        html.append('<table><tr><th>Kinase</th><th>Potent Compounds Available</th></tr>')
        for k, n in pb.get("kinase_sources", {}).items():
            html.append(f'<tr><td>{k}</td><td>{n}</td></tr>')
        html.append('</table>')

        html.append(f'<p>Novel kinase compounds scored: <strong>{pb["n_novel_kinase_mols"]}</strong>, '
                    f'predicted active (pIC50≥6.5): <strong class="good">{pb["n_predicted_active"]}</strong></p>')

        if pb.get("candidates"):
            html.append('<h3>Top Repurposing Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>Source</th><th>Pred ZAP70 pIC50</th>'
                        '<th>±Unc</th><th>Similarity</th><th>Source pIC50</th><th>SMILES</th></tr>')
            for c in pb["candidates"][:20]:
                src_val = f"{c['source_pchembl']:.1f}" if c.get("source_pchembl") else "—"
                html.append(f'<tr><td>{c["rank"]}</td><td>{c["source_kinase"]}</td>'
                            f'<td>{c["predicted_map3k8_pIC50"]:.2f}</td>'
                            f'<td>±{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity_to_map3k8"]:.3f}</td>'
                            f'<td>{src_val}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 4: Phase C — Generated Molecules
    if "phase_c" in results and results["phase_c"].get("completed"):
        pc = results["phase_c"]
        html.append('<div class="section">')
        html.append('<h2>4. BRICS-Generated Molecule Candidates</h2>')
        html.append(f'<p>Generated <strong>{pc["n_generated"]}</strong> novel molecules via BRICS '
                    f'recombination of fragments from top-20 potent molecules.</p>')

        html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
        for label, value in [
            ("Generated", str(pc["n_generated"])),
            ("Predicted Potent", str(pc["n_potent"])),
            ("Drug-like", str(pc["n_druglike"])),
            ("BRICS Fragments", str(pc["n_brics_fragments"])),
        ]:
            html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div></div>')
        html.append('</div>')

        if pc.get("top_scaffolds"):
            html.append('<h3>Source Scaffolds (from potent molecules)</h3>')
            html.append('<table><tr><th>Scaffold</th><th>N Mols</th><th>Mean pIC50</th></tr>')
            for s in pc["top_scaffolds"]:
                html.append(f'<tr><td><span class="smi">{s["scaffold"][:80]}</span></td>'
                            f'<td>{s["n_mols"]}</td><td>{s["mean_pIC50"]:.2f}</td></tr>')
            html.append('</table>')

        if pc.get("candidates"):
            html.append('<h3>Top Generated Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>Pred pIC50</th><th>±Unc</th>'
                        '<th>Similarity</th><th>QED</th><th>SA Score</th><th>SMILES</th></tr>')
            for c in pc["candidates"][:20]:
                qed = f"{c['QED']:.2f}" if c.get("QED") else "—"
                sa = f"{c['SA_score']:.1f}" if c.get("SA_score") else "—"
                html.append(f'<tr><td>{c["rank"]}</td>'
                            f'<td>{c["predicted_pIC50"]:.2f}</td>'
                            f'<td>±{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity"]:.3f}</td>'
                            f'<td>{qed}</td><td>{sa}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 5: Phase D — MMP Optimization
    if "phase_d" in results and results["phase_d"].get("completed"):
        pd_res = results["phase_d"]
        html.append('<div class="section">')
        html.append('<h2>5. MMP-Guided Optimization</h2>')
        html.append(f'<p>Identified <strong>{pd_res["n_beneficial_edits"]}</strong> consistently beneficial '
                    f'edits from the MMP database (≥10 pairs, mean Δ>0.3 pIC50, ≥2 targets).</p>')

        if pd_res.get("top_edits"):
            html.append('<h3>Top Beneficial Edits (across all kinases)</h3>')
            html.append('<table><tr><th>Edit (leaving→incoming)</th><th>N Pairs</th>'
                        '<th>Mean Δ pIC50</th><th>N Targets</th></tr>')
            for e in pd_res["top_edits"][:15]:
                edit_str = str(e.get("edit_smiles", "?"))
                if len(edit_str) > 60:
                    edit_str = edit_str[:57] + "..."
                html.append(f'<tr><td><span class="smi">{edit_str}</span></td>'
                            f'<td>{e.get("n_pairs", 0)}</td>'
                            f'<td class="good">{e.get("mean_delta", 0):+.2f}</td>'
                            f'<td>{e.get("n_targets", 0)}</td></tr>')
            html.append('</table>')

        if pd_res.get("optimization_suggestions"):
            html.append(f'<h3>Applicable Optimizations ({pd_res["n_applicable_optimizations"]} found)</h3>')
            html.append('<table><tr><th>Seed pIC50</th><th>Edit</th><th>Expected Δ</th>'
                        '<th>Expected pIC50</th><th>Confidence</th></tr>')
            for o in pd_res["optimization_suggestions"][:15]:
                html.append(f'<tr><td>{o["seed_pIC50"]:.2f}</td>'
                            f'<td><span class="smi">{o["edit"][:50]}</span></td>'
                            f'<td class="good">{o["expected_delta"]:+.2f}</td>'
                            f'<td><strong>{o["expected_new_pIC50"]:.2f}</strong></td>'
                            f'<td>{o["edit_confidence"]:.0f} pairs</td></tr>')
            html.append('</table>')
        else:
            html.append('<div class="note">No directly applicable edits found for seed molecules. '
                        'This is expected when training molecules have complex scaffolds that '
                        'don\'t match common MMP fragmentation patterns.</div>')
        html.append('</div>')

    # Section 6: Phase E — Final Rankings
    if "phase_e" in results and results["phase_e"].get("completed"):
        pe = results["phase_e"]
        html.append('<div class="section">')
        html.append('<h2>6. Final Multi-Objective Rankings</h2>')
        html.append('<p>Combined candidates from all sources, scored by: '
                    'predicted potency (50%), druglikeness (20%), '
                    'novelty/similarity balance (15%), prediction confidence (15%).</p>')

        html.append(f'<p>Total candidates evaluated: <strong>{pe["n_total_candidates"]}</strong></p>')

        if pe.get("top_candidates"):
            html.append('<table><tr><th>Rank</th><th>Source</th><th>Pred pIC50</th>'
                        '<th>Composite Score</th><th>QED</th><th>SMILES</th></tr>')
            for c in pe["top_candidates"][:30]:
                src = c.get("source", "?")
                qed = f"{c['QED']:.2f}" if c.get("QED") else "—"
                html.append(f'<tr><td>{c["final_rank"]}</td><td>{src}</td>'
                            f'<td><strong>{c.get("predicted_pIC50", 0):.2f}</strong></td>'
                            f'<td>{c["composite_score"]:.3f}</td>'
                            f'<td>{qed}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 7: Methodology & Limitations
    html.append('<div class="section">')
    html.append('<h2>7. Methodology & Limitations</h2>')
    html.append('''
    <h3>Prediction Model</h3>
    <ul>
        <li><strong>Architecture</strong>: Grand ensemble of 5 models (XGBoost + Random Forest)
            on diverse fingerprints (Morgan ECFP4, AtomPair, RDKit topological, ECFP6)</li>
        <li><strong>Training</strong>: 280 ZAP70 molecules with measured IC50/Ki values from ChEMBL</li>
        <li><strong>Validation</strong>: 5-fold CV, MAE=0.555 pIC50, Spearman=0.713</li>
        <li><strong>Label noise floor</strong>: ≈0.461 pIC50 (from 24 multi-assay molecules)</li>
        <li><strong>Uncertainty</strong>: Ensemble standard deviation across 5 models</li>
    </ul>

    <h3>Screening Strategies</h3>
    <ul>
        <li><strong>ChEMBL screening</strong>: Score untested ZAP70 compounds from ChEMBL 36</li>
        <li><strong>Kinase repurposing</strong>: Score potent compounds from related kinases
            (ITK, SYK, FYN, RAF1, MEK1, MEK2)</li>
        <li><strong>BRICS generation</strong>: Recombine fragments from top-20 potent molecules
            using BRICS decomposition/build algorithm</li>
        <li><strong>MMP optimization</strong>: Apply statistically beneficial edits from 1.7M MMP pairs</li>
    </ul>

    <h3>Limitations</h3>
    <div class="note">
        <ul>
            <li><strong>Applicability domain</strong>: Predictions are most reliable for molecules
                structurally similar to the training set (Tanimoto ≥ 0.3). Low-similarity predictions
                should be treated with extra caution.</li>
            <li><strong>Ensemble uncertainty</strong>: Reflects model disagreement, not true Bayesian
                uncertainty. May underestimate true prediction error.</li>
            <li><strong>BRICS molecules</strong>: Generated molecules may not be synthetically accessible.
                SA scores are approximate.</li>
            <li><strong>MMP edits</strong>: Beneficial edits from cross-target data may not transfer
                to ZAP70 specifically. Expected deltas are averages across diverse targets.</li>
            <li><strong>No 3D/docking</strong>: This is a ligand-based screen. Structure-based
                validation (docking to PDB structures) recommended for top candidates.</li>
        </ul>
    </div>
    ''')
    html.append('</div>')

    # Footer
    html.append('<div style="text-align: center; color: #999; padding: 20px; font-size: 12px;">')
    html.append(f'Generated by Edit Effect Framework | {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
                f'ZAP70 Virtual Screening Pipeline v5')
    html.append('</div>')

    html.append("</body></html>")

    report_text = "\n".join(html)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    print(f"  Report saved to {REPORT_FILE}")
    print(f"  Report size: {len(report_text):,} characters")

    results["report"] = {"path": str(REPORT_FILE), "size": len(report_text)}
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("=" * 70)
    print(f"ZAP70 (CHEMBL2803) Virtual Screening & Molecule Design — v5")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    mol_data, per_assay = load_zap70_molecules()

    # Load or initialize results
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"  Loaded existing results: {list(results.keys())}")

    results["data_summary"] = {
        "target": TARGET_ID,
        "target_name": TARGET_NAME,
        "n_molecules": len(mol_data),
        "pIC50_range": [float(mol_data["pIC50"].min()), float(mol_data["pIC50"].max())],
    }

    # Run all phases
    results = run_phase_a(mol_data, results)
    gc.collect()

    results = run_phase_b(mol_data, results)
    gc.collect()

    results = run_phase_c(mol_data, results)
    gc.collect()

    results = run_phase_d(mol_data, results)
    gc.collect()

    results = run_phase_e(mol_data, results)
    gc.collect()

    results = run_phase_f(mol_data, results)

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed
    results["completed"] = True
    save_results(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print(f"Results: {RESULTS_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
