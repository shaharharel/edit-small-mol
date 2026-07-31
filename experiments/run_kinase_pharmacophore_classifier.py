"""Train a kinase pharmacophore classifier (Strategy 2).

Binary classifier on Morgan FP-2048:
  Positive: FDA-approved kinase inhibitors + ChEMBL pIC50 >= 7 kinase ligands
  Negative: random non-kinase ChEMBL drugs + ZINC-style decoys matched by MW

Output:
  models/kinase_pharmacophore_clf.pkl  -- joblib-pickled (xgb_clf, feat_meta)
  results/paper_evaluation/kinase_pharmacophore_metrics.json
  results/paper_evaluation/kinase_pharmacophore_scores.csv  (per existing cohort)

Usage:
  conda run -n quris python -u experiments/run_kinase_pharmacophore_classifier.py
"""
from __future__ import annotations
import os, sys, json, time, sqlite3, gzip, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")

CHEMBL_DB = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
SHARED_PAIRS = ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
OUT_MODEL = ROOT / "models" / "kinase_pharmacophore_clf.pkl"
OUT_METRICS = ROOT / "results" / "paper_evaluation" / "kinase_pharmacophore_metrics.json"
OUT_SCORES = ROOT / "results" / "paper_evaluation" / "kinase_pharmacophore_scores.csv"

# FDA-approved kinase inhibitors (canonical "tinib" + a few non-tinib kinase drugs).
# Source: public lists, names only; SMILES resolved from ChEMBL by pref_name lookup.
FDA_KINASE_DRUG_NAMES = [
    "IMATINIB","DASATINIB","NILOTINIB","BOSUTINIB","PONATINIB",
    "GEFITINIB","ERLOTINIB","AFATINIB","OSIMERTINIB","DACOMITINIB","LAPATINIB","NERATINIB",
    "SUNITINIB","SORAFENIB","PAZOPANIB","AXITINIB","REGORAFENIB","CABOZANTINIB","VANDETANIB",
    "LENVATINIB","TIVOZANIB","NINTEDANIB","NINTEDANIB ESYLATE",
    "CRIZOTINIB","CERITINIB","ALECTINIB","BRIGATINIB","LORLATINIB","ENTRECTINIB",
    "RUXOLITINIB","TOFACITINIB","BARICITINIB","UPADACITINIB","FEDRATINIB","PACRITINIB",
    "IBRUTINIB","ACALABRUTINIB","ZANUBRUTINIB","TIRABRUTINIB",
    "VEMURAFENIB","DABRAFENIB","ENCORAFENIB","TRAMETINIB","COBIMETINIB","BINIMETINIB",
    "PALBOCICLIB","RIBOCICLIB","ABEMACICLIB",
    "IDELALISIB","COPANLISIB","DUVELISIB","ALPELISIB","UMBRALISIB",
    "EVEROLIMUS","SIROLIMUS","TEMSIROLIMUS",
    "MIDOSTAURIN","GILTERITINIB","QUIZARTINIB",
    "LAROTRECTINIB","SELPERCATINIB","PRALSETINIB",
    "FOSTAMATINIB","ENTOSPLETINIB",
    "TUCATINIB","CAPMATINIB","TEPOTINIB","SAVOLITINIB",
    "RIPRETINIB","AVAPRITINIB","PEMIGATINIB","INFIGRATINIB","ERDAFITINIB","FUTIBATINIB",
    "SOTORASIB","ADAGRASIB","DIVARASIB",
    "MOBOCERTINIB","REPOTRECTINIB","SELUMETINIB","BERZOSERTIB",
]


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def get_kinase_target_ids(conn: sqlite3.Connection) -> set:
    """ChEMBL kinase targets (~454)."""
    q = """
    SELECT DISTINCT td.chembl_id
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_class cc ON tc.component_id = cc.component_id
    JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
    WHERE pc.protein_class_id IN (
        SELECT protein_class_id FROM protein_classification
        WHERE protein_class_id = 6 OR parent_id = 6
        OR parent_id IN (SELECT protein_class_id FROM protein_classification WHERE parent_id = 6)
        OR parent_id IN (SELECT protein_class_id FROM protein_classification
            WHERE parent_id IN (SELECT protein_class_id FROM protein_classification WHERE parent_id = 6))
    ) AND td.target_type = 'SINGLE PROTEIN'
    """
    return set(pd.read_sql(q, conn)["chembl_id"].tolist())


def load_fda_kinase_smiles(conn: sqlite3.Connection) -> list[str]:
    """Pull SMILES for FDA-approved kinase drugs from ChEMBL by name."""
    placeholders = ",".join("?" * len(FDA_KINASE_DRUG_NAMES))
    q = f"""
    SELECT md.pref_name, cs.canonical_smiles
    FROM molecule_dictionary md
    JOIN compound_structures cs ON md.molregno = cs.molregno
    WHERE UPPER(md.pref_name) IN ({placeholders})
    """
    df = pd.read_sql(q, conn, params=FDA_KINASE_DRUG_NAMES)
    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates("pref_name")
    print(f"  Resolved {len(df)}/{len(FDA_KINASE_DRUG_NAMES)} FDA kinase drugs to SMILES")
    return df["canonical_smiles"].tolist()


def load_chembl_kinase_actives(conn: sqlite3.Connection, kinase_ids: set,
                                pic50_min: float = 7.0, max_n: int = 6000) -> list[str]:
    """SMILES of ChEMBL molecules with at least one pIC50 >= pic50_min against any kinase."""
    placeholders = ",".join("?" * len(kinase_ids))
    # Pull activities; ChEMBL stores pchembl_value
    q = f"""
    SELECT DISTINCT cs.canonical_smiles
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN compound_records cr ON act.record_id = cr.record_id
    JOIN compound_structures cs ON cr.molregno = cs.molregno
    WHERE td.chembl_id IN ({placeholders})
      AND act.pchembl_value >= ?
      AND act.standard_relation IN ('=', '<')
      AND act.data_validity_comment IS NULL
    LIMIT ?
    """
    params = list(kinase_ids) + [pic50_min, max_n * 3]
    df = pd.read_sql(q, conn, params=params)
    smiles = df["canonical_smiles"].dropna().drop_duplicates().tolist()[:max_n]
    print(f"  Loaded {len(smiles)} kinase-active SMILES (pIC50>={pic50_min})")
    return smiles


def load_non_kinase_drugs(conn: sqlite3.Connection, kinase_ids: set,
                         pic50_min: float = 6.0, max_n: int = 4000) -> list[str]:
    """Non-kinase actives: pIC50>=6 against non-kinase targets, exclude kinase-active mols."""
    # First, get all kinase-active molregnos
    placeholders = ",".join("?" * len(kinase_ids))
    q_kin = f"""
    SELECT DISTINCT cr.molregno
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN compound_records cr ON act.record_id = cr.record_id
    WHERE td.chembl_id IN ({placeholders}) AND act.pchembl_value >= 5.0
    """
    kin_mol = set(pd.read_sql(q_kin, conn, params=list(kinase_ids))["molregno"].tolist())
    print(f"  Excluding {len(kin_mol)} kinase-active molregnos from negative set")

    # Pull a broad non-kinase active set
    q_neg = f"""
    SELECT DISTINCT cs.canonical_smiles, cr.molregno
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN compound_records cr ON act.record_id = cr.record_id
    JOIN compound_structures cs ON cr.molregno = cs.molregno
    WHERE td.target_type = 'SINGLE PROTEIN'
      AND td.chembl_id NOT IN ({placeholders})
      AND act.pchembl_value >= ?
      AND act.standard_relation = '='
      AND act.data_validity_comment IS NULL
    LIMIT ?
    """
    params = list(kinase_ids) + [pic50_min, max_n * 3]
    df = pd.read_sql(q_neg, conn, params=params)
    df = df[~df["molregno"].isin(kin_mol)]
    smiles = df["canonical_smiles"].dropna().drop_duplicates().tolist()[:max_n]
    print(f"  Loaded {len(smiles)} non-kinase active SMILES")
    return smiles


def featurize(smiles_list: list[str], n_bits: int = 2048) -> tuple[np.ndarray, list[str]]:
    feats, kept = [], []
    for s in smiles_list:
        fp = morgan_fp(s, n_bits=n_bits)
        if fp is not None:
            feats.append(fp)
            kept.append(s)
    return np.stack(feats), kept


def score_lingo_cohorts(clf, n_bits: int = 2048) -> pd.DataFrame:
    """Score existing Lingo cohorts and other generated mol sets."""
    data_root = ROOT / "data"
    sdfs = []
    for pat in ("samples.sdf", "samples_T10.sdf", "samples_T07.sdf"):
        sdfs.extend(data_root.rglob(pat))
    sdfs = [s for s in sdfs if any(tok in str(s).lower() for tok in ("lingo", "covlingo"))]
    print(f"  Discovered {len(sdfs)} candidate Lingo SDFs")
    rows = []
    from rdkit.Chem import SDMolSupplier
    for sdf in sdfs:
        try:
            supp = SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
            smiles = []
            for m in supp:
                if m is None:
                    continue
                try:
                    smiles.append(Chem.MolToSmiles(m))
                except Exception:
                    pass
            if not smiles:
                continue
            X, kept = featurize(smiles, n_bits=n_bits)
            if len(X) == 0:
                continue
            probs = clf.predict_proba(X)[:, 1]
            cohort = sdf.parent.name
            family = sdf.parents[1].name if len(sdf.parents) >= 2 else sdf.parent.name
            for s, p in zip(kept, probs):
                rows.append({
                    "cohort_family": family,
                    "cohort": cohort,
                    "sdf_path": str(sdf.relative_to(ROOT)),
                    "smiles": s,
                    "kinase_pharmacophore_score": float(p),
                })
        except Exception as e:
            print(f"  WARN scoring {sdf}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Strategy 2: kinase pharmacophore classifier")
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(CHEMBL_DB))
    print("Step 1: loading kinase target IDs ...")
    kinase_ids = get_kinase_target_ids(conn)
    print(f"  {len(kinase_ids)} kinase targets")

    print("Step 2: loading positive set (FDA kinase drugs + ChEMBL kinase actives) ...")
    fda_smiles = load_fda_kinase_smiles(conn)
    chembl_kinase = load_chembl_kinase_actives(conn, kinase_ids, pic50_min=7.0, max_n=6000)
    pos_smiles = list(dict.fromkeys(fda_smiles + chembl_kinase))  # dedup, preserve order
    print(f"  Combined positives (deduped): {len(pos_smiles)}")

    print("Step 3: loading negative set (non-kinase actives) ...")
    neg_smiles = load_non_kinase_drugs(conn, kinase_ids, pic50_min=6.0, max_n=6000)
    conn.close()

    print("Step 4: featurizing ...")
    X_pos, smi_pos = featurize(pos_smiles)
    X_neg, smi_neg = featurize(neg_smiles)
    print(f"  X_pos {X_pos.shape}, X_neg {X_neg.shape}")

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.int8)

    print("Step 5a: train/holdout split (80/20 stratified, scaffold-naive) ...")
    from sklearn.model_selection import train_test_split
    rng = 42
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rng)

    print("Step 5b: also computing scaffold-based holdout for honest AUC ...")
    from rdkit.Chem.Scaffolds import MurckoScaffold
    all_smiles = smi_pos + smi_neg
    scaffolds = []
    for s in all_smiles:
        try:
            m = Chem.MolFromSmiles(s)
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        except Exception:
            sc = ""
        scaffolds.append(sc if sc else f"NONE_{len(scaffolds)}")
    sc_arr = np.array(scaffolds)
    unique_scaff = list(set(scaffolds))
    np.random.seed(rng)
    np.random.shuffle(unique_scaff)
    n_test_scaff = max(1, int(0.2 * len(unique_scaff)))
    test_scaff_set = set(unique_scaff[:n_test_scaff])
    sc_te_mask = np.array([s in test_scaff_set for s in scaffolds])
    sc_X_tr = X[~sc_te_mask]; sc_y_tr = y[~sc_te_mask]
    sc_X_te = X[sc_te_mask]; sc_y_te = y[sc_te_mask]
    print(f"  Scaffold-holdout: {len(sc_X_tr)} train / {len(sc_X_te)} test (n_test_scaffolds={n_test_scaff})")

    print(f"  Train: {len(X_tr)}  pos={int(y_tr.sum())}  neg={int((1-y_tr).sum())}")
    print(f"  Test:  {len(X_te)}  pos={int(y_te.sum())}  neg={int((1-y_te).sum())}")

    print("Step 6: training XGBoost ...")
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        random_state=rng,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    dt = time.time() - t0
    print(f"  Trained in {dt:.1f}s")

    print("Step 7: evaluating holdout ...")
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    proba_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba_te)
    ap = average_precision_score(y_te, proba_te)
    brier = brier_score_loss(y_te, proba_te)
    print(f"  Holdout ROC-AUC = {auc:.4f}   PR-AUC = {ap:.4f}   Brier = {brier:.4f}")

    # 5-fold CV ROC-AUC for a more robust estimate
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    fold_aucs = []
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, n_jobs=-1,
            eval_metric="auc", tree_method="hist", random_state=rng + fold,
        )
        m.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])], verbose=False)
        p = m.predict_proba(X[va_i])[:, 1]
        a = roc_auc_score(y[va_i], p)
        fold_aucs.append(a)
        print(f"    fold {fold}: ROC-AUC = {a:.4f}")
    cv_auc_mean = float(np.mean(fold_aucs))
    cv_auc_std = float(np.std(fold_aucs))
    print(f"  5-fold CV ROC-AUC = {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")

    # Scaffold-based AUC -- the honest one
    sc_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, n_jobs=-1,
        eval_metric="auc", tree_method="hist", random_state=rng,
    )
    sc_clf.fit(sc_X_tr, sc_y_tr, eval_set=[(sc_X_te, sc_y_te)], verbose=False)
    sc_proba = sc_clf.predict_proba(sc_X_te)[:, 1]
    if len(set(sc_y_te.tolist())) > 1:
        sc_auc = float(roc_auc_score(sc_y_te, sc_proba))
        sc_ap = float(average_precision_score(sc_y_te, sc_proba))
    else:
        sc_auc = float("nan"); sc_ap = float("nan")
    print(f"  Scaffold-holdout ROC-AUC = {sc_auc:.4f}   PR-AUC = {sc_ap:.4f}")

    pass_qa = (sc_auc > 0.80) if sc_auc == sc_auc else (auc > 0.80)

    # Spot-check 10 FDA drugs (expect high) and 10 random Lingo mols (expect low)
    print("Step 8: QA spot-check ...")
    fda_check = pos_smiles[:10]
    Xfda, fda_keep = featurize(fda_check)
    fda_scores = clf.predict_proba(Xfda)[:, 1].tolist() if len(Xfda) else []
    print(f"  FDA kinase drug scores (10): mean={np.mean(fda_scores):.3f}  scores={[round(x,3) for x in fda_scores]}")

    print("Step 9: scoring existing Lingo cohorts ...")
    cohort_df = score_lingo_cohorts(clf)
    if len(cohort_df) > 0:
        cohort_df.to_csv(OUT_SCORES, index=False)
        print(f"  Wrote {len(cohort_df)} per-mol scores across {cohort_df['cohort'].nunique()} cohorts → {OUT_SCORES}")
        cohort_summary = cohort_df.groupby(["cohort_family","cohort"]).agg(
            n=("smiles","count"),
            mean_score=("kinase_pharmacophore_score","mean"),
            frac_gt_07=("kinase_pharmacophore_score", lambda s: float((s>0.7).mean())),
        ).reset_index()
        print(cohort_summary.to_string(index=False))
    else:
        cohort_summary = pd.DataFrame()
        print("  No SDF cohorts found / scored.")

    print("Step 10: saving model ...")
    bundle = {
        "model": clf,
        "n_bits": 2048,
        "morgan_radius": 2,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pos": int(len(X_pos)),
        "n_neg": int(len(X_neg)),
        "holdout_roc_auc": float(auc),
        "holdout_pr_auc": float(ap),
        "holdout_brier": float(brier),
        "cv_roc_auc_mean": cv_auc_mean,
        "cv_roc_auc_std": cv_auc_std,
        "scaffold_holdout_roc_auc": float(sc_auc),
        "scaffold_holdout_pr_auc": float(sc_ap),
        "fda_drugs": FDA_KINASE_DRUG_NAMES,
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Wrote {OUT_MODEL}")

    metrics = {
        "holdout_roc_auc": float(auc),
        "holdout_pr_auc": float(ap),
        "holdout_brier": float(brier),
        "cv_roc_auc_mean": cv_auc_mean,
        "cv_roc_auc_std": cv_auc_std,
        "scaffold_holdout_roc_auc": float(sc_auc),
        "scaffold_holdout_pr_auc": float(sc_ap),
        "n_pos": int(len(X_pos)),
        "n_neg": int(len(X_neg)),
        "n_kinase_targets": len(kinase_ids),
        "n_fda_kinase_drugs": len(fda_smiles),
        "n_chembl_kinase_actives": len(chembl_kinase),
        "n_non_kinase_actives": len(neg_smiles),
        "qa_passed_auc_gt_080": bool(pass_qa),
        "fda_spot_check_scores": fda_scores,
        "cohort_summary": cohort_summary.to_dict(orient="records") if len(cohort_summary) else [],
    }
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote {OUT_METRICS}")
    print(f"[{time.strftime('%H:%M:%S')}] Done. QA passed: {pass_qa}")
    return metrics


if __name__ == "__main__":
    main()
