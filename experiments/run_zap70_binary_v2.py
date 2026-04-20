#!/usr/bin/env python3
"""
Improved Binary ZAP70 Classifier (v2) — Hard Negative Sampling.

v1 issues: AUROC=0.963 was inflated by 500 trivially-easy diverse non-kinase negatives.
Model learned "is kinase?" not "is ZAP70 binder?". All 19 candidates got P(binder) < 0.21.

v2 strategy (from expert panel):
  - Remove all diverse non-kinase negatives (Tc < 0.3)
  - Tier 1: ZAP70 inactives (pIC50 < 5.0) — 49 mols (gold standard)
  - Tier 2: Cross-kinase hard negatives — kinase actives (pIC50 >= 6.0) with
            structural similarity to ZAP70 binders (Tc >= 0.25), from non-ZAP70 targets
  - Tier 3: Near-miss decoys — non-kinase mols with Tc 0.3-0.5 to ZAP70 binders
  - Tier 4: BTK-active, ZAP70-weak discordant molecules (from overlap set)

Also adds:
  - Retrospective OOD validation (hold out structurally dissimilar ZAP70 binders)
  - Multiple classifiers: RF, GradientBoosting, logistic regression
  - Comparison with v1 on same data

Usage:
    conda run -n quris python -u experiments/run_zap70_binary_v2.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.SaltRemover import SaltRemover

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"

SMILES_19 = [
    'C=CC(=O)N1CC=2C=CC=C(C(=O)NC3=CN(C=N3)C(C)C)C2C1',
    'C=CC(=O)N1CC2(CCC(=O)NC3=CNN=C3C(=O)NCC(C)O)CCC1CC2',
    'C=CC(=O)NC[C@H]1C[C@H]2C[C@@H]1CN2C=3N=CN=C(N)C3Cl',
    'C=CC(=O)N1CCCC1(C(=O)NC2=CNN=C2OCC(F)F)C=3C=CC=CC3',
    'C=CC(=O)N1CCC(CC(=O)NC2=CNN=C2C=3C=NC=CN3)CC41CC4',
    'C=CC(=O)N1CC2(CCC2)CC1CC(=O)NC3=CNN=C3C(=O)NCC(C)O',
    'C=CC(=O)N1C[C@H](CC(C)(C)C)[C@H](C1)C(=O)NC2=CNN=C2C(=O)NCC(C)O',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C3=CNC=4C=C(F)C(Cl)=CC34)S2',
    'C=CC(=O)NC1C2C3C[C@@H]1[C@H](C(=O)NC4=CNN=C4C=5C=NC=CN5)C32',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C=3C=CC(F)=C(C3)S(=O)(=O)N(C)C)S2',
    'C=CC(=O)NCC1=CN(N=N1)[C@@H]2C[C@H](C2)C(=O)NC=3C=CC=NC3NC(C)=O',
    'C=CC(=O)N1CC2(CC1CCC2)NC=3N=CC=C(N3)OC=4C=CC=C(C#N)C4',
    'C=CC(=O)NC(C)C=1N=CC(=CN1)NC(=O)C=2N=CN=C3NC=C(C)C23',
    'C=CC(=O)N(C)C1(CNC=2N=CN=C(N)C2C(=O)OCC)CCC1',
    'C=CC(=O)N1CCC[C@H]1C(C)NC(=O)C=2C=NNC2C=3C=CN(C)N3',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CC(CCNC=2C=C(N=CN2)NC=3C=CC=CC3)(C1)N(C)C',
    'O=C(O)C(F)(F)F.C=CC(=O)N(C)C1(CNC=2N=CN=C3NC=C(C(N)=O)C23)CCC1',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CCN(CC1)C=2N=CN=C(N2)NC3(CC)CCNCC3',
    'C=CC(=O)N1CCCC(NC(=O)C2=CNN=C2C3=CC=4C=CC=CC4O3)C51CCC5',
]


def clean_smiles(smi):
    """Clean SMILES: remove ChEMBL extended info and salts."""
    smi_clean = smi.split(' |')[0] if ' |' in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return smi_clean
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)


def compute_fps_bulk(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints as RDKit objects for Tanimoto comparison."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
            valid_idx.append(i)
    return fps, valid_idx


def max_tanimoto_to_set(fp, ref_fps, n_check=None):
    """Compute max Tanimoto similarity of fp to a set of reference fps."""
    if n_check is not None:
        ref_fps = ref_fps[:n_check]
    if not ref_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
    return max(sims) if sims else 0.0


def build_negative_tiers(raw, zap_pos_smiles, zap_neg_smiles, zap_pos_fps):
    """Build tiered hard negatives for ZAP70 binary classifier."""
    results = {}

    # ---- Tier 1: ZAP70 inactives (already have these) ----
    tier1 = list(zap_neg_smiles)
    results['tier1_zap70_inactive'] = tier1
    print(f"  Tier 1 (ZAP70 inactives): {len(tier1)}")

    # ---- Tier 2: Cross-kinase hard negatives ----
    # Kinase actives (pIC50 >= 6.0) NOT tested on ZAP70, with Tc >= 0.25 to ZAP70 binders
    kinase_ids = {"CHEMBL2803", "CHEMBL5251", "CHEMBL2599", "CHEMBL258",
                  "CHEMBL1841", "CHEMBL3009", "CHEMBL2971", "CHEMBL1862", "CHEMBL267"}
    # Get all kinase actives except ZAP70
    other_kinase = raw[(raw["target_chembl_id"].isin(kinase_ids - {"CHEMBL2803"})) &
                       (raw["pIC50"] >= 6.0)]
    other_kinase_mols = other_kinase.groupby("smiles")["pIC50"].mean().reset_index()

    # Remove any molecules that are in ZAP70 set
    zap_set = set(zap_pos_smiles) | set(zap_neg_smiles)
    other_kinase_mols = other_kinase_mols[~other_kinase_mols['smiles'].isin(zap_set)]
    print(f"  Tier 2 candidates: {len(other_kinase_mols)} kinase actives (pIC50 >= 6.0)")

    # Filter by Tc >= 0.25 to ZAP70 binders
    tier2 = []
    tier2_tc = []
    for _, row in other_kinase_mols.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        tc = max_tanimoto_to_set(fp, zap_pos_fps, n_check=100)
        if tc >= 0.25:
            tier2.append(row['smiles'])
            tier2_tc.append(tc)
        if len(tier2) >= 300:
            break

    results['tier2_cross_kinase'] = tier2
    print(f"  Tier 2 (cross-kinase hard neg, Tc>=0.25): {len(tier2)}")
    if tier2_tc:
        print(f"    Tc range: [{min(tier2_tc):.3f}, {max(tier2_tc):.3f}], mean={np.mean(tier2_tc):.3f}")

    # ---- Tier 3: Near-miss decoys (Tc 0.3-0.5 to ZAP70 binders) ----
    non_kinase = raw[~raw["target_chembl_id"].isin(kinase_ids)]
    non_kinase_mols = non_kinase.groupby("smiles").first().reset_index()
    non_kinase_mols = non_kinase_mols[~non_kinase_mols['smiles'].isin(zap_set)]

    # Sample and filter
    sample = non_kinase_mols['smiles'].sample(n=min(10000, len(non_kinase_mols)),
                                               random_state=42).tolist()
    tier3 = []
    tier3_tc = []
    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        tc = max_tanimoto_to_set(fp, zap_pos_fps, n_check=100)
        if 0.3 <= tc <= 0.5:
            tier3.append(smi)
            tier3_tc.append(tc)
        if len(tier3) >= 150:
            break

    results['tier3_near_miss'] = tier3
    print(f"  Tier 3 (near-miss decoys, Tc 0.3-0.5): {len(tier3)}")
    if tier3_tc:
        print(f"    Tc range: [{min(tier3_tc):.3f}, {max(tier3_tc):.3f}], mean={np.mean(tier3_tc):.3f}")

    # ---- Tier 4: BTK-active, ZAP70-weak discordant ----
    # Molecules tested on both BTK and ZAP70 where BTK pIC50 >> ZAP70 pIC50
    btk_data = raw[raw["target_chembl_id"] == "CHEMBL5251"].groupby("smiles")["pIC50"].mean()
    zap_data = raw[raw["target_chembl_id"] == "CHEMBL2803"].groupby("smiles")["pIC50"].mean()
    overlap = pd.DataFrame({"btk": btk_data, "zap70": zap_data}).dropna()

    tier4 = []
    if len(overlap) > 0:
        # BTK-active but ZAP70-weak
        discordant = overlap[(overlap['btk'] >= 6.0) & (overlap['zap70'] < 5.5)]
        tier4 = discordant.index.tolist()
        print(f"  Tier 4 (BTK-active, ZAP70-weak): {len(tier4)} (from {len(overlap)} overlap)")
    else:
        print(f"  Tier 4 (BTK-ZAP70 discordant): 0 overlap molecules")
    results['tier4_btk_discordant'] = tier4

    return results


def run_classifier(X_train, y_train, X_test, y_test, X_cand, clf_name="RF"):
    """Train classifier and return metrics + candidate predictions."""
    if clf_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=4)
    elif clf_name == "GB":
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
    elif clf_name == "LR":
        clf = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "auprc": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    # Candidate predictions
    cand_probs = clf.predict_proba(X_cand)[:, 1] if X_cand is not None else None
    return metrics, cand_probs, clf


def run_cv_and_score(X, y, X_cand, clf_name="RF", n_splits=5):
    """Run stratified CV + final model scoring of candidates."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = []
    cand_probs_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m, _, _ = run_classifier(X_tr, y_tr, X_te, y_te, None, clf_name)
        cv_metrics.append(m)
        print(f"    Fold {fold_idx}: AUROC={m['auroc']:.3f}, AUPRC={m['auprc']:.3f}, F1={m['f1']:.3f}")

    # Final model on all data
    _, cand_probs, clf_final = run_classifier(X, y, X[:10], y[:10], X_cand, clf_name)

    avg = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0]}
    std = {k: float(np.std([m[k] for m in cv_metrics])) for k in cv_metrics[0]}

    return avg, std, cv_metrics, cand_probs, clf_final


def retrospective_ood_test(X, y, smiles_list, pos_fps, clf_name="RF"):
    """Hold out structurally dissimilar positives to test OOD performance."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Compute max Tc of each positive to all other positives
    pos_sims = []
    for i, pi in enumerate(pos_idx):
        other_fps = [pos_fps[j] for j in range(len(pos_fps)) if j != i]
        tc = max_tanimoto_to_set(pos_fps[i], other_fps, n_check=50)
        pos_sims.append((pi, tc))

    # Hold out bottom 10% (most structurally dissimilar)
    pos_sims.sort(key=lambda x: x[1])
    n_holdout = max(5, len(pos_idx) // 10)
    holdout_idx = [ps[0] for ps in pos_sims[:n_holdout]]
    holdout_tc = [ps[1] for ps in pos_sims[:n_holdout]]

    train_pos_idx = [pi for pi in pos_idx if pi not in holdout_idx]
    train_idx = list(train_pos_idx) + list(neg_idx)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_ood = X[holdout_idx]
    y_ood = y[holdout_idx]

    # Train and predict
    if clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                     class_weight='balanced', random_state=42, n_jobs=4)
    elif clf_name == "GB":
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    else:
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    ood_probs = clf.predict_proba(X_ood)[:, 1]

    return {
        "n_holdout": n_holdout,
        "holdout_tc_range": [float(min(holdout_tc)), float(max(holdout_tc))],
        "holdout_tc_mean": float(np.mean(holdout_tc)),
        "ood_probs_mean": float(np.mean(ood_probs)),
        "ood_probs_std": float(np.std(ood_probs)),
        "ood_recall_at_50": float(np.mean(ood_probs >= 0.5)),
        "ood_recall_at_30": float(np.mean(ood_probs >= 0.3)),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("BINARY ZAP70 CLASSIFIER v2 — Hard Negative Sampling")
    print("=" * 70)

    # Load data
    raw = pd.read_csv(RAW_FILE)
    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    # ZAP70 molecules
    mol_data, _ = load_zap70_molecules()
    zap_smiles = mol_data['smiles'].tolist()
    zap_y = mol_data['pIC50'].values

    pos_mask = zap_y >= 5.0
    pos_smiles = [s for s, m in zip(zap_smiles, pos_mask) if m]
    neg_zap_smiles = [s for s, m in zip(zap_smiles, pos_mask) if not m]
    print(f"\nZAP70 positives (pIC50 >= 5.0): {len(pos_smiles)}")
    print(f"ZAP70 inactives (pIC50 < 5.0): {len(neg_zap_smiles)}")

    # Compute positive fingerprints
    print("\nComputing ZAP70 positive fingerprints...")
    pos_fps, _ = compute_fps_bulk(pos_smiles)
    print(f"  {len(pos_fps)} valid fingerprints")

    # =========================================================
    # Build tiered negatives
    # =========================================================
    print(f"\n{'='*70}")
    print("BUILDING TIERED HARD NEGATIVES")
    print(f"{'='*70}")

    tiers = build_negative_tiers(raw, pos_smiles, neg_zap_smiles, pos_fps)

    # Combine all negatives
    all_neg_smiles = []
    neg_tier_labels = []
    for tier_name, tier_smiles in tiers.items():
        all_neg_smiles.extend(tier_smiles)
        neg_tier_labels.extend([tier_name] * len(tier_smiles))

    # Deduplicate negatives
    seen = set()
    deduped_neg = []
    deduped_tier = []
    for smi, tier in zip(all_neg_smiles, neg_tier_labels):
        if smi not in seen and smi not in set(pos_smiles):
            seen.add(smi)
            deduped_neg.append(smi)
            deduped_tier.append(tier)

    tier_counts = Counter(deduped_tier)
    print(f"\nFinal negative composition (after dedup):")
    for tier, cnt in tier_counts.items():
        print(f"  {tier}: {cnt}")
    print(f"  Total negatives: {len(deduped_neg)}")
    print(f"  Positive:Negative ratio: 1:{len(deduped_neg)/len(pos_smiles):.1f}")

    # =========================================================
    # Prepare features
    # =========================================================
    print(f"\n{'='*70}")
    print("CLASSIFIER EVALUATION")
    print(f"{'='*70}")

    all_clf_smiles = list(pos_smiles) + list(deduped_neg)
    all_clf_labels = np.array([1]*len(pos_smiles) + [0]*len(deduped_neg))
    X_clf = compute_fingerprints(all_clf_smiles, "morgan", radius=2, n_bits=2048)
    X_cand = compute_fingerprints(cand_smiles, "morgan", radius=2, n_bits=2048)

    print(f"\nDataset: {len(pos_smiles)} pos + {len(deduped_neg)} neg = {len(all_clf_smiles)}")

    results = {
        "version": "v2_hard_negatives",
        "positives": len(pos_smiles),
        "negatives": len(deduped_neg),
        "neg_composition": dict(tier_counts),
    }

    # =========================================================
    # Run v2 classifiers (hard negatives)
    # =========================================================
    classifiers = ["RF", "GB", "LR"]
    v2_results = {}

    for clf_name in classifiers:
        print(f"\n--- {clf_name} (v2 hard negatives) ---")
        avg, std, cv_m, cand_probs, clf_final = run_cv_and_score(
            X_clf, all_clf_labels, X_cand, clf_name)
        print(f"  Mean: AUROC={avg['auroc']:.3f}±{std['auroc']:.3f}, "
              f"AUPRC={avg['auprc']:.3f}±{std['auprc']:.3f}, "
              f"F1={avg['f1']:.3f}±{std['f1']:.3f}")

        ranking = np.argsort(-cand_probs)
        print(f"  Top-5 candidates:")
        for rank, j in enumerate(ranking[:5]):
            print(f"    {rank+1}. Mol {j+1}: P(binder) = {cand_probs[j]:.3f}")

        v2_results[clf_name] = {
            "cv_auroc": avg['auroc'], "cv_auroc_std": std['auroc'],
            "cv_auprc": avg['auprc'], "cv_auprc_std": std['auprc'],
            "cv_f1": avg['f1'], "cv_f1_std": std['f1'],
            "cv_metrics": cv_m,
            "candidates": [
                {"idx": j+1, "p_binder": float(cand_probs[j])}
                for j in range(19)
            ],
            "ranking": [int(r+1) for r in ranking],
        }

    results["v2_classifiers"] = v2_results

    # =========================================================
    # Run v1 baseline (with diverse non-kinase negatives) for comparison
    # =========================================================
    print(f"\n{'='*70}")
    print("v1 BASELINE (diverse non-kinase negatives) for comparison")
    print(f"{'='*70}")

    # Recreate v1 negatives
    kinase_ids = {"CHEMBL2803", "CHEMBL5251", "CHEMBL2599", "CHEMBL258",
                  "CHEMBL1841", "CHEMBL3009", "CHEMBL2971", "CHEMBL1862", "CHEMBL267"}
    non_kinase = raw[~raw["target_chembl_id"].isin(kinase_ids)]
    non_kinase_mols = non_kinase.groupby("smiles").first().reset_index()

    v1_neg_candidates = non_kinase_mols['smiles'].sample(
        n=min(5000, len(non_kinase_mols)), random_state=42).tolist()
    v1_diverse = []
    for smi in v1_neg_candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        tc = max_tanimoto_to_set(fp, pos_fps, n_check=50)
        if tc < 0.3:
            v1_diverse.append(smi)
        if len(v1_diverse) >= 500:
            break

    v1_neg = list(neg_zap_smiles) + v1_diverse
    v1_all_smiles = list(pos_smiles) + v1_neg
    v1_labels = np.array([1]*len(pos_smiles) + [0]*len(v1_neg))
    X_v1 = compute_fingerprints(v1_all_smiles, "morgan", radius=2, n_bits=2048)

    print(f"\nv1 dataset: {len(pos_smiles)} pos + {len(v1_neg)} neg "
          f"({len(neg_zap_smiles)} ZAP70 weak + {len(v1_diverse)} diverse)")

    print(f"\n--- RF (v1 baseline) ---")
    v1_avg, v1_std, v1_cv, v1_cand_probs, _ = run_cv_and_score(
        X_v1, v1_labels, X_cand, "RF")
    print(f"  Mean: AUROC={v1_avg['auroc']:.3f}±{v1_std['auroc']:.3f}, "
          f"AUPRC={v1_avg['auprc']:.3f}±{v1_std['auprc']:.3f}")

    v1_ranking = np.argsort(-v1_cand_probs)
    print(f"  Top-5 candidates:")
    for rank, j in enumerate(v1_ranking[:5]):
        print(f"    {rank+1}. Mol {j+1}: P(binder) = {v1_cand_probs[j]:.3f}")

    results["v1_baseline"] = {
        "negatives": len(v1_neg),
        "neg_composition": {"zap70_weak": len(neg_zap_smiles), "diverse_non_kinase": len(v1_diverse)},
        "cv_auroc": v1_avg['auroc'], "cv_auprc": v1_avg['auprc'],
        "cv_metrics": v1_cv,
        "candidates": [
            {"idx": j+1, "p_binder": float(v1_cand_probs[j])}
            for j in range(19)
        ],
        "ranking": [int(r+1) for r in v1_ranking],
    }

    # =========================================================
    # Retrospective OOD validation
    # =========================================================
    print(f"\n{'='*70}")
    print("RETROSPECTIVE OOD VALIDATION")
    print(f"{'='*70}")
    print("Hold out 10% most structurally dissimilar ZAP70 positives")

    # Need fps aligned with the classifier dataset
    pos_fps_aligned, _ = compute_fps_bulk(pos_smiles)

    ood_results = {}
    for clf_name in ["RF", "GB"]:
        print(f"\n  {clf_name}:")
        ood = retrospective_ood_test(X_clf, all_clf_labels, all_clf_smiles,
                                     pos_fps_aligned, clf_name)
        print(f"    Held out {ood['n_holdout']} OOD positives (Tc range: "
              f"[{ood['holdout_tc_range'][0]:.3f}, {ood['holdout_tc_range'][1]:.3f}])")
        print(f"    OOD P(binder): {ood['ood_probs_mean']:.3f} ± {ood['ood_probs_std']:.3f}")
        print(f"    Recall@0.5: {ood['ood_recall_at_50']:.1%}, "
              f"Recall@0.3: {ood['ood_recall_at_30']:.1%}")
        ood_results[clf_name] = ood

    results["ood_validation"] = ood_results

    # =========================================================
    # Candidate analysis
    # =========================================================
    print(f"\n{'='*70}")
    print("CANDIDATE ANALYSIS — v1 vs v2 comparison")
    print(f"{'='*70}")

    best_v2 = "RF"  # Use RF as primary
    v2_probs = np.array([r['p_binder'] for r in v2_results[best_v2]['candidates']])
    v1_probs = np.array([r['p_binder'] for r in results['v1_baseline']['candidates']])

    print(f"\n  {'Mol':>4s}  {'v1 P(bind)':>10s}  {'v2 P(bind)':>10s}  {'Change':>8s}")
    print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>10s}  {'------':>8s}")
    for j in range(19):
        change = v2_probs[j] - v1_probs[j]
        print(f"  {j+1:4d}  {v1_probs[j]:10.3f}  {v2_probs[j]:10.3f}  {change:+8.3f}")

    r_v1v2, p_v1v2 = spearmanr(v1_probs, v2_probs)
    print(f"\n  v1 vs v2 ranking correlation: Spearman={r_v1v2:.3f} (p={p_v1v2:.3e})")
    print(f"  v1 P(binder) range: [{v1_probs.min():.3f}, {v1_probs.max():.3f}]")
    print(f"  v2 P(binder) range: [{v2_probs.min():.3f}, {v2_probs.max():.3f}]")

    # Compare with BTK predictions if available
    try:
        with open(RESULTS_DIR / "19_molecules_cross_kinase.json") as f:
            cross_data = json.load(f)
        btk_preds = [c['pred_mean'] for c in cross_data['BTK']['candidates']]
        r_btk, p_btk = spearmanr(v2_probs, btk_preds)
        print(f"\n  v2 P(binder) vs BTK pred: Spearman={r_btk:.3f} (p={p_btk:.3e})")
        results["correlations"] = {
            "v2_vs_btk": {"spearman": float(r_btk), "p": float(p_btk)},
            "v1_vs_v2": {"spearman": float(r_v1v2), "p": float(p_v1v2)},
        }
    except Exception:
        results["correlations"] = {
            "v1_vs_v2": {"spearman": float(r_v1v2), "p": float(p_v1v2)},
        }

    # =========================================================
    # Tanimoto analysis of candidates to training set
    # =========================================================
    print(f"\n{'='*70}")
    print("CANDIDATE TANIMOTO TO TRAINING SET")
    print(f"{'='*70}")

    cand_fps, _ = compute_fps_bulk(cand_smiles)
    all_train_fps = pos_fps_aligned
    # Also compute fps for negatives
    neg_fps, _ = compute_fps_bulk(deduped_neg[:200])  # Sample to save time

    cand_tc_to_pos = []
    cand_tc_to_neg = []
    for j, cfp in enumerate(cand_fps):
        tc_pos = max_tanimoto_to_set(cfp, all_train_fps)
        tc_neg = max_tanimoto_to_set(cfp, neg_fps)
        cand_tc_to_pos.append(tc_pos)
        cand_tc_to_neg.append(tc_neg)
        print(f"  Mol {j+1}: Tc_pos={tc_pos:.3f}, Tc_neg={tc_neg:.3f}, "
              f"v2 P(bind)={v2_probs[j]:.3f}")

    results["candidate_tanimoto"] = [
        {"idx": j+1, "tc_to_pos": float(cand_tc_to_pos[j]),
         "tc_to_neg": float(cand_tc_to_neg[j])}
        for j in range(19)
    ]

    # =========================================================
    # Save results
    # =========================================================
    out_file = RESULTS_DIR / "19_molecules_binary_v2.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s — saved to {out_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
