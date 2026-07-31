#!/usr/bin/env python3
"""Build F2-survivor (row_id, smiles, cohort) list across all 16 tier4 cohort CSVs.

F1 thresholds (Lipinski-like + safety + warhead):
  MW<=700, -1<=LogP<=6, TPSA<=180, HBD<=7, RotBonds<=14,
  PAINS_alerts<1, warhead_intact==True, Tc_to_Mol1<=0.85,
  Lipinski_violations<3

F2 thresholds (drug-like + structural cleanliness):
  MW<=500, LogP<=5, TPSA<=140, HBA<=12, HBD<=5, RotBonds<=11,
  QED>=0.25, HeavyAtoms<=50, Brenk_alerts<3

Output: /tmp/f2_survivors.csv with row_id, smiles, cohort
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TIER4 = PROJECT / "data/tier4_scored"
OUT = Path("/tmp/f2_survivors.csv")


COLS = [
    "smiles", "MW", "LogP", "TPSA", "HBA", "HBD", "RotBonds",
    "QED", "HeavyAtoms", "PAINS_alerts", "Brenk_alerts",
    "warhead_intact", "Tc_to_Mol1", "Lipinski_violations",
]


def main():
    csvs = sorted(glob.glob(str(TIER4 / "*_scored.csv")))
    rows = []
    print(f"scanning {len(csvs)} cohort CSVs", flush=True)
    for p in csvs:
        path = Path(p)
        cohort = path.stem.replace("_scored", "")
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception as e:
            print(f"  ERR {cohort}: {e}")
            continue
        use = [c for c in COLS if c in head.columns]
        # Need row_id-equivalent; if absent, synthesize from position
        df = pd.read_csv(path, usecols=use)
        df["row_id"] = [f"{cohort}_{i}" for i in range(len(df))]

        def col(name, default=0):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        # F1
        m1 = (
            (col("MW", 0) <= 700)
            & (col("LogP", 0) >= -1) & (col("LogP", 0) <= 6)
            & (col("TPSA", 0) <= 180)
            & (col("HBD", 0) <= 7)
            & (col("RotBonds", 0) <= 14)
            & (col("PAINS_alerts", 0) < 1)
            & (col("Tc_to_Mol1", 0) <= 0.85)
            & (col("Lipinski_violations", 0) < 3)
        )
        # warhead_intact may be bool/str/int
        if "warhead_intact" in df.columns:
            wi = df["warhead_intact"]
            if wi.dtype == bool:
                m1 = m1 & wi
            else:
                m1 = m1 & (wi.astype(str).str.lower().isin(["true", "1", "t"]))
        else:
            # if no warhead col, treat as fail
            m1 = m1 & False

        # F2
        m2 = (
            (col("MW", 0) <= 500)
            & (col("LogP", 0) <= 5)
            & (col("TPSA", 0) <= 140)
            & (col("HBA", 0) <= 12)
            & (col("HBD", 0) <= 5)
            & (col("RotBonds", 0) <= 11)
            & (col("QED", 0) >= 0.25)
            & (col("HeavyAtoms", 0) <= 50)
            & (col("Brenk_alerts", 0) < 3)
        )

        sub = df[m1 & m2][["row_id", "smiles"]].copy()
        sub["cohort"] = cohort
        rows.append(sub)
        print(f"  {cohort:<35} n_total={len(df):>7,}  F1={int(m1.sum()):>6,}  "
              f"F2={int(m2.sum()):>6,}  F1&F2={len(sub):>6,}", flush=True)

    all_surv = pd.concat(rows, ignore_index=True)
    # de-dup on (cohort, row_id) just in case
    all_surv = all_surv.drop_duplicates(subset=["cohort", "row_id"])
    all_surv.to_csv(OUT, index=False)
    print(f"\n[done] F2 survivors total = {len(all_surv):,} -> {OUT}")


if __name__ == "__main__":
    main()
