"""Safely merge pIC50_* columns from a source CSV into a target CSV using SMILES join.

Used to avoid clobbering concurrent writes (e.g. ai-chem writing xTB columns).
Reads source, computes the new columns, then re-reads target just before write
to minimize the read-write window.

Usage:
    python scripts/merge_pic50_cols.py \
        --source /path/to/source_with_pIC50.csv \
        --target /path/to/target_to_merge_into.csv \
        --cols pIC50_film pIC50_mean pIC50_method pIC50_std anchor_wins anchor_wins_ge7 \
               delta_vs_mol1 direct_delta_from_mol1 LE LLE BEI SEI SILE \
        --smiles-col smiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    args = ap.parse_args()

    print(f"[merge] source={args.source}, target={args.target}", flush=True)
    src = pd.read_csv(args.source, low_memory=False)
    print(f"[merge] source: {len(src):,} rows, {len(src.columns)} cols", flush=True)
    have = [c for c in args.cols if c in src.columns]
    miss = [c for c in args.cols if c not in src.columns]
    if miss:
        print(f"[merge] WARNING: cols missing from source: {miss}", flush=True)
    if not have:
        print(f"[merge] ERROR: none of requested cols exist in source", flush=True)
        return 1
    # Build smiles -> {col: val} map from source
    src_unique = src[[args.smiles_col] + have].drop_duplicates(args.smiles_col)
    print(f"[merge] {len(src_unique):,} unique smiles with new cols", flush=True)

    # Re-read target just before write (minimize race window)
    tgt = pd.read_csv(args.target, low_memory=False)
    print(f"[merge] target: {len(tgt):,} rows, {len(tgt.columns)} cols", flush=True)
    # Drop any pre-existing target copies of the cols we're merging in
    overlap = [c for c in have if c in tgt.columns]
    if overlap:
        print(f"[merge] overwriting existing target cols: {overlap}", flush=True)
        tgt = tgt.drop(columns=overlap)
    out = tgt.merge(src_unique, on=args.smiles_col, how="left", validate="m:1")
    print(f"[merge] merged: {len(out):,} rows, {len(out.columns)} cols", flush=True)
    for c in have:
        cov = out[c].notna().mean()
        print(f"[merge]   {c}: {cov:.0%}", flush=True)
    out.to_csv(args.target, index=False)
    print(f"[merge] DONE wrote {args.target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
