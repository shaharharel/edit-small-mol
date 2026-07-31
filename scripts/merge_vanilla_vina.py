#!/usr/bin/env python3
"""Safe-merge vanilla_vina_kcalmol into existing scored CSV by row_id.

Usage:
    merge_vanilla_vina.py <vina_csv> <scored_csv>

Behavior:
  - Reads scored_csv (must have row_id column).
  - Reads vina_csv (row_id,smiles,vina_kcalmol,status).
  - Sentinel-clips vina_kcalmol > +5.0 to NaN (failed dock).
  - Adds/overwrites column 'vanilla_vina_kcalmol' on scored_csv via left-merge on row_id.
  - Backs up scored_csv to scored_csv + '.bak.vvina.<ts>' before overwrite.
  - Prints summary.
"""
import sys
import time
from pathlib import Path
import pandas as pd

VINA_SENTINEL_CLIP_KCALMOL = 5.0


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    vina_csv = Path(sys.argv[1])
    scored_csv = Path(sys.argv[2])
    if not vina_csv.exists():
        print(f"missing vina csv: {vina_csv}")
        sys.exit(1)
    if not scored_csv.exists():
        print(f"missing scored csv: {scored_csv}")
        sys.exit(1)

    df = pd.read_csv(scored_csv)
    vn = pd.read_csv(vina_csv)
    if "row_id" not in df.columns:
        print(f"scored_csv lacks row_id: {scored_csv}")
        sys.exit(1)
    if "row_id" not in vn.columns:
        print(f"vina_csv lacks row_id: {vina_csv}")
        sys.exit(1)
    if "vina_kcalmol" not in vn.columns:
        print(f"vina_csv lacks vina_kcalmol col: {vina_csv}")
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = scored_csv.with_suffix(scored_csv.suffix + f".bak.vvina.{ts}")
    df.to_csv(bak, index=False)

    if "vanilla_vina_kcalmol" in df.columns:
        df = df.drop(columns=["vanilla_vina_kcalmol"])
    vn_min = vn[["row_id", "vina_kcalmol"]].drop_duplicates(subset="row_id").copy()

    raw_pos = (vn_min["vina_kcalmol"] > VINA_SENTINEL_CLIP_KCALMOL).sum()
    vn_min["vina_kcalmol"] = vn_min["vina_kcalmol"].where(
        vn_min["vina_kcalmol"] <= VINA_SENTINEL_CLIP_KCALMOL,
        other=pd.NA,
    )
    vn_min = vn_min.rename(columns={"vina_kcalmol": "vanilla_vina_kcalmol"})

    merged = df.merge(vn_min, on="row_id", how="left")
    assert len(merged) == len(df), f"row count changed: {len(df)} -> {len(merged)}"

    # atomic write via .tmp
    tmp = scored_csv.with_suffix(scored_csv.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(scored_csv)

    nn = merged["vanilla_vina_kcalmol"].notna().sum()
    pct = 100.0 * nn / len(merged) if len(merged) else 0.0
    finite = merged["vanilla_vina_kcalmol"].dropna()
    if len(finite):
        print(
            f"[merge] {scored_csv.name}: rows={len(merged)} vvina_nn={nn} ({pct:.1f}%) "
            f"clip={raw_pos} max={finite.max():.3f} med={finite.median():.3f} bak={bak.name}"
        )
    else:
        print(
            f"[merge] {scored_csv.name}: rows={len(merged)} vvina_nn=0 (0.0%) clip={raw_pos} bak={bak.name}"
        )


if __name__ == "__main__":
    main()
