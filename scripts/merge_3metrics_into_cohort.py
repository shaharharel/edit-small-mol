"""Additive merge of the 3 covalent-screening metric CSVs into the cohort CSV.

Reads:
  data/tier4_scored/propka_pKa_Cys346_3597.csv          (col: pKa_Cys346)
  data/tier4_scored/rdkit_strain_kcal_mol_3597.csv      (col: strain_kcal_mol → rdkit_strain_kcal_mol)
  data/tier4_scored/xtb_pred_log_k2_GSH_3597.csv        (col: pred_log_k2_GSH)

Reads QA verdicts at /tmp/<metric>_qa.json and SKIPS any stream whose
verdict is FAIL (does not merge bad data into the cohort).

Writes (additive, with .bak3 backup):
  data/tier4_scored/boltz2_cohort_A_relaxed.csv

Usage: python scripts/merge_3metrics_into_cohort.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER4_DIR = PROJECT_ROOT / "data/tier4_scored"
COHORT_CSV = TIER4_DIR / "boltz2_cohort_A_relaxed.csv"

STREAMS = [
    # (metric_name, csv_filename, src_col, dst_col, qa_path)
    ("pKa_Cys346",
     TIER4_DIR / "propka_pKa_Cys346_3597.csv",
     "pKa_Cys346", "pKa_Cys346",
     Path("/tmp/pKa_Cys346_qa.json")),
    ("rdkit_strain_kcal_mol",
     TIER4_DIR / "rdkit_strain_kcal_mol_3597.csv",
     "strain_kcal_mol", "rdkit_strain_kcal_mol",
     Path("/tmp/rdkit_strain_kcal_mol_qa.json")),
    ("pred_log_k2_GSH",
     TIER4_DIR / "xtb_pred_log_k2_GSH_3597.csv",
     "pred_log_k2_GSH", "pred_log_k2_GSH",
     Path("/tmp/pred_log_k2_GSH_qa.json")),
]


def main() -> int:
    if not COHORT_CSV.exists():
        print(f"Cohort CSV missing: {COHORT_CSV}")
        return 1
    cohort = pd.read_csv(COHORT_CSV, low_memory=False)
    print(f"Cohort: {len(cohort):,} rows, {len(cohort.columns)} cols")
    bak = COHORT_CSV.with_suffix(".csv.bak3_3metrics")
    shutil.copy2(COHORT_CSV, bak)
    print(f"  backup → {bak}")

    cohort["row_id"] = cohort["row_id"].astype(int)
    n_merged = 0
    skipped = []
    for metric, csv_path, src_col, dst_col, qa_path in STREAMS:
        if not csv_path.exists():
            print(f"\n[{metric}] CSV missing: {csv_path} — SKIP")
            skipped.append((metric, "csv_missing"))
            continue
        if qa_path.exists():
            verdict = json.loads(qa_path.read_text()).get("verdict", "UNKNOWN")
            if verdict == "FAIL":
                print(f"\n[{metric}] QA verdict=FAIL ({qa_path.name}) — SKIPPING merge to avoid bad data")
                skipped.append((metric, "qa_fail"))
                continue
            print(f"\n[{metric}] QA verdict={verdict} — proceeding with merge")
        else:
            print(f"\n[{metric}] no QA file at {qa_path} — proceeding cautiously")

        df = pd.read_csv(csv_path)
        if src_col not in df.columns:
            print(f"  src col '{src_col}' not in CSV — SKIP")
            skipped.append((metric, "src_col_missing"))
            continue
        df["row_id"] = df["row_id"].astype(int)
        df = df[["row_id", src_col]].rename(columns={src_col: dst_col})
        df = df.drop_duplicates(subset="row_id", keep="last")
        # Drop col if already present (replace).
        if dst_col in cohort.columns:
            print(f"  col '{dst_col}' already in cohort — replacing")
            cohort = cohort.drop(columns=[dst_col])
        before = len(cohort.columns)
        cohort = cohort.merge(df, on="row_id", how="left")
        coverage = cohort[dst_col].notna().sum()
        print(f"  merged {len(df):,} rows; coverage in cohort = {coverage:,}/{len(cohort):,} ({100*coverage/len(cohort):.1f}%)")
        # Quick stats
        s = pd.to_numeric(cohort[dst_col], errors="coerce").dropna()
        if len(s) > 0:
            print(f"  stats: median={s.median():.3f} p10={s.quantile(0.10):.3f} p90={s.quantile(0.90):.3f} sigma={s.std():.3f}")
        n_merged += 1

    cohort.to_csv(COHORT_CSV, index=False)
    print(f"\nWrote {COHORT_CSV} ({len(cohort):,} rows, {len(cohort.columns)} cols)")
    print(f"Merged: {n_merged}/{len(STREAMS)} streams")
    if skipped:
        print("Skipped streams:")
        for m, reason in skipped:
            print(f"  {m}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
