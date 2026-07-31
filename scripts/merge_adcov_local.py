#!/usr/bin/env python3
"""Safe-merge adcov_local_kcalmol into an existing scored CSV by row_id.

Usage:
    merge_adcov_local.py <adcov_csv> <scored_csv>

Behavior:
  - Reads scored_csv (must have row_id column).
  - Reads adcov_csv (row_id,smiles,adcov_local_kcalmol,status).
  - Adds/overwrites column 'adcov_local_kcalmol' on scored_csv via left-merge on row_id.
  - NaN-clamps the adcov_local_kcalmol column for sentinel failure values
    (adcov --local_only returns ~+199.9 when local minimization fails;
    threshold > +5.0 kcal/mol catches the sentinel and any other absurd
    positives; preserves true bind scores; F4 cascade NaN-pass semantics
    will then treat these rows as 'unknown, pass through' instead of
    'very weak binder').
  - Backs up scored_csv to scored_csv + '.bak.adcov.<ts>' before overwrite.
  - Prints a summary: total rows, joined rows, non-NaN count, pct non-NaN,
    sentinel-clipped count.
"""
import sys
import time
from pathlib import Path
import pandas as pd

# AD-CovDock --local_only sentinel-clip threshold (kcal/mol).
# Values above this indicate failed local minimization, not weak binding.
ADCOV_SENTINEL_CLIP_KCALMOL = 5.0

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    adcov_csv = Path(sys.argv[1])
    scored_csv = Path(sys.argv[2])
    if not adcov_csv.exists():
        print(f"missing adcov csv: {adcov_csv}"); sys.exit(1)
    if not scored_csv.exists():
        print(f"missing scored csv: {scored_csv}"); sys.exit(1)

    df = pd.read_csv(scored_csv)
    ad = pd.read_csv(adcov_csv)
    if "row_id" not in df.columns:
        print(f"scored_csv lacks row_id: {scored_csv}"); sys.exit(1)
    if "row_id" not in ad.columns:
        print(f"adcov_csv lacks row_id: {adcov_csv}"); sys.exit(1)

    # backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = scored_csv.with_suffix(scored_csv.suffix + f".bak.adcov.{ts}")
    df.to_csv(bak, index=False)

    # safe merge: only bring in adcov_local_kcalmol, drop existing column if present
    if "adcov_local_kcalmol" in df.columns:
        df = df.drop(columns=["adcov_local_kcalmol"])
    ad_min = ad[["row_id", "adcov_local_kcalmol"]].drop_duplicates(subset="row_id")

    # NaN-clamp sentinel failures BEFORE merging so downstream filters
    # (F4 cascade etc.) treat them as 'unknown' not 'very weak binder'.
    raw_pos = (ad_min["adcov_local_kcalmol"] > ADCOV_SENTINEL_CLIP_KCALMOL).sum()
    ad_min = ad_min.copy()
    ad_min["adcov_local_kcalmol"] = ad_min["adcov_local_kcalmol"].where(
        ad_min["adcov_local_kcalmol"] <= ADCOV_SENTINEL_CLIP_KCALMOL,
        other=pd.NA,
    )

    merged = df.merge(ad_min, on="row_id", how="left")

    assert len(merged) == len(df), f"row count changed: {len(df)} -> {len(merged)}"
    merged.to_csv(scored_csv, index=False)

    nn = merged["adcov_local_kcalmol"].notna().sum()
    pct = 100.0 * nn / len(merged) if len(merged) else 0.0
    finite = merged["adcov_local_kcalmol"].dropna()
    max_v = finite.max() if len(finite) else float("nan")
    med_v = finite.median() if len(finite) else float("nan")
    print(
        f"[merge] {scored_csv.name}: rows={len(merged)}  adcov_non_nan={nn} ({pct:.1f}%)  "
        f"sentinel_clipped={raw_pos}  max={max_v:.3f}  median={med_v:.3f}  backup={bak.name}"
    )

if __name__ == "__main__":
    main()
