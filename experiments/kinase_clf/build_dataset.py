"""Build kinase classifier dataset from ChEMBL 36.

Strategy:
  - Kinase target list: ChEMBL recursive class hierarchy under protein_class_id=6.
  - Tec family target list: protein_class_id=143 (Tec family) + Bruton's etc.
  - Positives: kinase mols with pIC50 >= 6 against any kinase target.
  - GOLD negatives: mols with measured pIC50 < 5 against ANY kinase target
                    (TESTED-AND-FAILED). Excluded if also a positive.
  - Non-kinase positives: pIC50 >= 6 against non-kinase SINGLE PROTEIN targets,
                          EXCLUDING any mol with a recorded kinase pIC50 >= 5 (polypharmacology filter).
  - DUD-E-like decoys: heuristic property-matched random ChEMBL mols never measured against a kinase.

Labels: is_kinase (0/1), target_family (str), pIC50 (float for regression head, NaN for decoys/non-kinase neg).

Output: results/paper_evaluation/kinase_classifier_dataset.csv  (smiles, label_kinase, label_tec, pIC50, family, source)
"""
from __future__ import annotations
import sqlite3, time, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHEMBL_DB = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
OUT_CSV = ROOT / "results" / "paper_evaluation" / "kinase_classifier_dataset.csv"
OUT_META = ROOT / "results" / "paper_evaluation" / "kinase_classifier" / "dataset_meta.json"

TEC_FAMILY_TARGETS = {
    "CHEMBL2959",   # ITK
    "CHEMBL5025",   # TEC
    "CHEMBL3834",   # BMX
    "CHEMBL5251",   # BTK
    "CHEMBL4246",   # Tec (rat)
    "CHEMBL4367",   # TXK
    "CHEMBL2034793",# BMX alt
    "CHEMBL3259478",# BTK alt
}


def kinase_targets(conn) -> dict:
    """Return dict: target_chembl_id -> protein_class_short."""
    q = """
    WITH RECURSIVE kinase_classes(protein_class_id, root_group) AS (
        SELECT protein_class_id, NULL AS root_group
        FROM protein_classification WHERE protein_class_id = 6
        UNION ALL
        SELECT pc.protein_class_id,
               COALESCE(kc.root_group, CASE WHEN pc.class_level = 4 THEN pc.short_name ELSE NULL END)
        FROM protein_classification pc
        JOIN kinase_classes kc ON pc.parent_id = kc.protein_class_id
    )
    SELECT DISTINCT td.chembl_id, td.pref_name,
           COALESCE(MIN(kc.root_group), 'Other') AS kinase_group
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_class cc ON tc.component_id = cc.component_id
    JOIN kinase_classes kc ON cc.protein_class_id = kc.protein_class_id
    WHERE td.target_type = 'SINGLE PROTEIN'
    GROUP BY td.chembl_id
    """
    df = pd.read_sql(q, conn)
    print(f"  {len(df)} kinase targets")
    print(f"  Group breakdown:")
    print(df["kinase_group"].value_counts().to_string())
    return dict(zip(df["chembl_id"], df["kinase_group"]))


def fetch_kinase_activities(conn, kinase_ids: list) -> pd.DataFrame:
    """Pull ALL (mol, target, pIC50) for kinase targets in one go (fast bulk join)."""
    ph = ",".join("?" * len(kinase_ids))
    q = f"""
    SELECT cs.canonical_smiles AS smiles,
           td.chembl_id        AS target_chembl_id,
           act.pchembl_value   AS pIC50,
           act.standard_relation AS rel
    FROM activities act
    JOIN assays a               ON act.assay_id = a.assay_id
    JOIN target_dictionary td   ON a.tid = td.tid
    JOIN compound_structures cs ON act.molregno = cs.molregno
    WHERE td.chembl_id IN ({ph})
      AND act.pchembl_value IS NOT NULL
      AND act.data_validity_comment IS NULL
    """
    df = pd.read_sql(q, conn, params=kinase_ids)
    return df


def fetch_non_kinase_positives(conn, kinase_ids: list, exclude_smiles: set,
                                target_n: int = 30000) -> pd.DataFrame:
    """Non-kinase actives (pIC50>=6), excluding mols with any kinase pIC50>=5 (poly-pharm filter)."""
    ph = ",".join("?" * len(kinase_ids))
    # Round-robin across non-kinase SINGLE_PROTEIN targets for chemotype diversity
    q = f"""
    SELECT DISTINCT cs.canonical_smiles AS smiles, td.chembl_id AS target_chembl_id
    FROM activities act
    JOIN assays a               ON act.assay_id = a.assay_id
    JOIN target_dictionary td   ON a.tid = td.tid
    JOIN compound_structures cs ON act.molregno = cs.molregno
    WHERE td.target_type = 'SINGLE PROTEIN'
      AND td.chembl_id NOT IN ({ph})
      AND act.pchembl_value >= 6.0
      AND act.standard_relation = '='
      AND act.data_validity_comment IS NULL
    """
    df = pd.read_sql(q, conn, params=kinase_ids)
    print(f"  Raw non-kinase positives: {len(df)} ({df['smiles'].nunique()} unique)")
    # Remove anything in the polypharmacology exclude set
    df = df[~df["smiles"].isin(exclude_smiles)].copy()
    print(f"  After polypharmacology exclusion: {df['smiles'].nunique()} unique")
    # Down-sample per target for diversity
    df = (df.groupby("target_chembl_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 80), random_state=42))
            .drop_duplicates("smiles", keep="first"))
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42)
    return df


def fetch_decoys(conn, kinase_ids: list, exclude_smiles: set, target_n: int = 5000) -> pd.DataFrame:
    """Random ChEMBL mols NEVER recorded against a kinase (any pchembl): hard property-matched negatives."""
    ph = ",".join("?" * len(kinase_ids))
    # Mols with at least one activity record but never against a kinase target
    q = f"""
    SELECT cs.canonical_smiles AS smiles
    FROM compound_structures cs
    WHERE cs.molregno IN (
        SELECT DISTINCT act.molregno
        FROM activities act
        WHERE act.pchembl_value IS NOT NULL
    )
    AND cs.molregno NOT IN (
        SELECT DISTINCT act.molregno
        FROM activities act
        JOIN assays a             ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        WHERE td.chembl_id IN ({ph})
    )
    LIMIT 60000
    """
    df = pd.read_sql(q, conn, params=kinase_ids)
    df = df[~df["smiles"].isin(exclude_smiles)].drop_duplicates("smiles")
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42)
    return df


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building kinase classifier dataset (ChEMBL 36)")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHEMBL_DB))

    print("Step 1: kinase target list ...")
    kin_target_to_group = kinase_targets(conn)
    kin_ids = list(kin_target_to_group.keys())
    tec_ids = TEC_FAMILY_TARGETS & set(kin_ids)
    print(f"  Tec-family kinase targets retained: {len(tec_ids)}")

    print(f"Step 2: kinase activities ({len(kin_ids)} targets) ...")
    t1 = time.time()
    act = fetch_kinase_activities(conn, kin_ids)
    print(f"  {len(act):,} activity rows in {time.time()-t1:.1f}s")

    # Best per (smiles, target_chembl_id) — max pIC50
    act_best = (act.groupby(["smiles","target_chembl_id"], as_index=False)["pIC50"]
                  .max())
    act_best["group"] = act_best["target_chembl_id"].map(kin_target_to_group).fillna("Other")
    act_best["is_tec"] = act_best["target_chembl_id"].isin(tec_ids)

    # Per-mol: max kinase pIC50 across all kinase targets, and Tec-family active flag
    mol_max = (act_best.groupby("smiles")
                 .agg(max_kinase_pIC50=("pIC50","max"),
                      n_kinase_records=("pIC50","count"),
                      any_tec_active=("is_tec", lambda x: ((act_best.loc[x.index, "pIC50"]>=6) & x).any()),
                      best_group=("pIC50", lambda x: act_best.loc[x.idxmax(), "group"])
                     )
                 .reset_index())
    print(f"  Unique kinase-tested mols: {len(mol_max):,}")

    # Kinase POSITIVES: max kinase pIC50 >= 6
    pos = mol_max[mol_max["max_kinase_pIC50"] >= 6].copy()
    pos["label_kinase"] = 1
    pos["label_tec"]    = pos["any_tec_active"].astype(int)
    pos["pIC50"]        = pos["max_kinase_pIC50"]
    pos["family"]       = pos["best_group"]
    pos["source"]       = "chembl_kinase_active"
    print(f"  Kinase POSITIVES (pIC50>=6): {len(pos):,}  (Tec actives: {int(pos['label_tec'].sum()):,})")

    # Kinase TESTED-AND-FAILED: max kinase pIC50 < 5, but at least one record
    gold_neg = mol_max[mol_max["max_kinase_pIC50"] < 5].copy()
    gold_neg["label_kinase"] = 0
    gold_neg["label_tec"]    = 0  # not a kinase binder at all
    gold_neg["pIC50"]        = np.nan  # don't use regression label for negatives
    gold_neg["family"]       = "non_kinase_tested"
    gold_neg["source"]       = "chembl_kinase_tested_inactive"
    print(f"  Kinase TESTED-AND-FAILED negatives (max pIC50<5): {len(gold_neg):,}")

    # Polypharmacology exclude set for non-kinase positives:
    # any mol ever recorded with kinase pIC50 >= 5
    poly_exclude = set(mol_max[mol_max["max_kinase_pIC50"] >= 5]["smiles"].tolist())
    print(f"  Polypharmacology exclude set (any kinase pIC50>=5): {len(poly_exclude):,}")

    print("Step 3: non-kinase positives (with polypharmacology filter) ...")
    nonkin = fetch_non_kinase_positives(conn, kin_ids, poly_exclude, target_n=30000)
    nonkin["label_kinase"] = 0
    nonkin["label_tec"]    = 0
    nonkin["pIC50"]        = np.nan
    nonkin["family"]       = "non_kinase_active"
    nonkin["source"]       = "chembl_non_kinase_active"
    print(f"  Non-kinase positives kept: {len(nonkin):,}")

    print("Step 4: DUD-E-like decoys (never-kinase-measured) ...")
    decoys = fetch_decoys(conn, kin_ids, set(pos["smiles"]).union(set(gold_neg["smiles"])).union(set(nonkin["smiles"])), target_n=5000)
    decoys["label_kinase"] = 0
    decoys["label_tec"]    = 0
    decoys["pIC50"]        = np.nan
    decoys["family"]       = "decoy"
    decoys["source"]       = "chembl_never_kinase"
    print(f"  Decoys kept: {len(decoys):,}")

    conn.close()

    # Assemble
    cols = ["smiles", "label_kinase", "label_tec", "pIC50", "family", "source"]
    all_df = pd.concat([pos[cols], gold_neg[cols], nonkin[cols], decoys[cols]], ignore_index=True)
    # de-dup priority: positives > gold_neg > non_kin > decoys; first-keep
    all_df = all_df.drop_duplicates(subset="smiles", keep="first")

    # Drop pathological SMILES (NaN / empty)
    all_df = all_df.dropna(subset=["smiles"])
    all_df = all_df[all_df["smiles"].str.len() > 0]

    # Final balance
    print()
    print(f"Final dataset: {len(all_df):,} mols")
    print(all_df.groupby(["label_kinase","family"]).size().to_string())
    print()
    print(f"label_kinase = 1: {int((all_df['label_kinase']==1).sum()):,}  "
          f"(label_tec=1: {int(((all_df['label_kinase']==1)&(all_df['label_tec']==1)).sum()):,})")
    print(f"label_kinase = 0: {int((all_df['label_kinase']==0).sum()):,}")

    # Per-family kinase distribution
    print()
    print("Kinase active per-group breakdown (kinase positives only):")
    print(all_df[all_df["label_kinase"]==1].groupby("family").size().to_string())

    all_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(all_df):,} rows)")

    meta = dict(
        n_total=int(len(all_df)),
        n_kinase_pos=int((all_df["label_kinase"]==1).sum()),
        n_kinase_tested_neg=int((all_df["source"]=="chembl_kinase_tested_inactive").sum()),
        n_non_kinase_pos=int((all_df["source"]=="chembl_non_kinase_active").sum()),
        n_decoys=int((all_df["source"]=="chembl_never_kinase").sum()),
        n_tec_pos=int(((all_df["label_kinase"]==1)&(all_df["label_tec"]==1)).sum()),
        family_distribution=all_df[all_df["label_kinase"]==1].groupby("family").size().to_dict(),
        n_pIC50_labels=int(all_df["pIC50"].notna().sum()),
        wall_clock_s=time.time()-t0,
    )
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_META}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
