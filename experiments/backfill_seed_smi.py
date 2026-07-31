"""Backfill seed_smi + seed_idx columns into Tier 4 scored CSVs.

Mapping is done by exact SMILES match against each cohort's raw
`*_5k_per_seed/cohort_all.csv`. Legacy EXP2/EXP6 cohorts (500 rows, no raw)
get seed_smi=Mol1 since those cohorts were pure-sampling-from-FT-prior
anchored on Mol1.

Writes back to data/tier4_scored/{cohort}_scored.csv (preserving a .bak).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SCORED = PROJECT / "data" / "tier4_scored"

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

# (scored_csv, raw_cohort_all_csv or None for legacy)
COHORTS = [
    ("exp6_v3_scored.csv",     PROJECT / "data/reinvent4_mol2mol_exp6_v3_5k_per_seed/cohort_all.csv"),
    ("exp6_v4_scored.csv",     PROJECT / "data/exp6_v4_5k_per_seed/cohort_all.csv"),
    ("exp6_v5_scored.csv",     PROJECT / "data/exp6_v5_5k_per_seed/cohort_all.csv"),
    ("exp2_v2_rl_v2_scored.csv", PROJECT / "data/exp2_v2_rl_v2_5k_per_seed/cohort_all.csv"),
    # Legacy 500-row cohorts: pure sampling anchored on Mol1 → seed = Mol1
    ("exp2_scored.csv", None),
    ("exp6_scored.csv", None),
]


def backfill_one(scored_path: Path, raw_path: Path | None) -> None:
    if not scored_path.exists():
        print(f"  [skip] {scored_path.name} missing")
        return

    df = pd.read_csv(scored_path)
    n_before = len(df)
    had_seed = "seed_smi" in df.columns

    if raw_path is None:
        # Legacy: pure sampling on Mol1
        df["seed_smi"] = MOL1
        df["seed_idx"] = 0
        coverage = 1.0
        seed_count = 1
    else:
        if not raw_path.exists():
            print(f"  [skip] {scored_path.name}: raw {raw_path.name} missing")
            return
        raw = pd.read_csv(raw_path)
        if "seed_smi" not in raw.columns:
            print(f"  [skip] {scored_path.name}: raw lacks seed_smi")
            return
        # Map smiles → first seed (in case of duplicates pick first)
        raw_uniq = raw.drop_duplicates(subset=["smiles"], keep="first")[
            ["smiles", "seed_smi", "seed_idx"] if "seed_idx" in raw.columns else ["smiles", "seed_smi"]
        ]
        if had_seed:
            df = df.drop(columns=["seed_smi", "seed_idx"], errors="ignore")
        df = df.merge(raw_uniq, on="smiles", how="left")
        coverage = df["seed_smi"].notna().mean()
        seed_count = df["seed_smi"].nunique(dropna=True)

    # Backup + write
    if not scored_path.with_suffix(".csv.bak_seed").exists():
        shutil.copy(scored_path, scored_path.with_suffix(".csv.bak_seed"))
    df.to_csv(scored_path, index=False)

    status = "UPDATE" if had_seed else "NEW"
    print(f"  [{status}] {scored_path.name}: {n_before:,} rows, "
          f"seed_smi coverage={coverage:.1%}, unique seeds={seed_count}")


def main():
    print(f"Backfilling seed_smi into {len(COHORTS)} cohort files...")
    print(f"  Tier 4 scored dir: {SCORED.relative_to(PROJECT)}\n")
    for scored_name, raw_path in COHORTS:
        scored_path = SCORED / scored_name
        backfill_one(scored_path, raw_path)
    print("\nDone. Backups: *.csv.bak_seed (one per cohort)")


if __name__ == "__main__":
    main()
