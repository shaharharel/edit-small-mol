#!/usr/bin/env python3
"""Finalize rescue_78_full.csv → exact column parity with F4_838 + 100% coverage.

Pipeline:
  1. Merge vanilla_vina_kcalmol from rescue_78_docking/vina_output.csv (45 missing)
  2. Merge adcov_local_kcalmol from rescue_78_docking/adcov_output.csv (18 missing)
  3. Drop columns NOT present in F4_boltz_full.csv (rescue-only extras, e.g.
     desirability_score / combined_score / P_potency / mPAE family / etc.)
     EXCEPT keep rescue_row_id as a separate identifier (slim backend strips it).
  4. Reorder columns to match F4 ordering.
  5. QA: assert 100% coverage on cols where F4 has 0% NaN.

Output: overwrites data/tier4_scored/rescue_78_full.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tier4_scored"
RESCUE_CSV = DATA / "rescue_78_full.csv"
F4_CSV     = DATA / "F4_boltz_full.csv"
VIS_SMIS   = DATA / "visible_838_smiles.json"
VINA_OUT   = ROOT / "data" / "rescue_78_docking" / "vina_output.csv"
ADCOV_OUT  = ROOT / "data" / "rescue_78_docking" / "adcov_output.csv"

print(f"[finalize] loading rescue_78 + F4_838")
rescue = pd.read_csv(RESCUE_CSV, low_memory=False)
f4 = pd.read_csv(F4_CSV, low_memory=False)
vis = set(json.loads(VIS_SMIS.read_text()))
f4_838 = f4[f4["smiles"].isin(vis)].copy()
print(f"  rescue: {rescue.shape}")
print(f"  f4_838: {f4_838.shape}")

# ── 1. Merge vanilla_vina_kcalmol ──────────────────────────────────────────
if VINA_OUT.exists():
    v = pd.read_csv(VINA_OUT)
    v_map = v.set_index("row_id")["vina_kcalmol"].dropna().to_dict()
    n_filled = 0
    for i, row in rescue.iterrows():
        rid = row["rescue_row_id"]
        if pd.isna(row.get("vanilla_vina_kcalmol")) and rid in v_map:
            rescue.at[i, "vanilla_vina_kcalmol"] = float(v_map[rid])
            n_filled += 1
    print(f"  filled vanilla_vina_kcalmol: {n_filled} rows")
else:
    print(f"  [warn] {VINA_OUT.name} not found — vanilla_vina_kcalmol gap remains")

# ── 2. Merge adcov_local_kcalmol ───────────────────────────────────────────
if ADCOV_OUT.exists():
    a = pd.read_csv(ADCOV_OUT)
    # adcov_local_scorer header is (row_id, smiles, vina_kcalmol, status) — col name "vina_kcalmol" is actually adcov-local
    score_col = "vina_kcalmol" if "vina_kcalmol" in a.columns else "adcov_local_kcalmol"
    a_map = a.set_index("row_id")[score_col].dropna().to_dict()
    n_filled = 0
    for i, row in rescue.iterrows():
        rid = row["rescue_row_id"]
        if pd.isna(row.get("adcov_local_kcalmol")) and rid in a_map:
            rescue.at[i, "adcov_local_kcalmol"] = float(a_map[rid])
            n_filled += 1
    print(f"  filled adcov_local_kcalmol: {n_filled} rows")
else:
    print(f"  [warn] {ADCOV_OUT.name} not found — adcov_local_kcalmol gap remains")

# ── 3. Trim to F4-cols + keep rescue_row_id for identification ─────────────
f4_cols_list = list(f4.columns)
keep_set = set(f4_cols_list) | {"rescue_row_id"}
drop_cols = [c for c in rescue.columns if c not in keep_set]
add_cols  = [c for c in f4_cols_list if c not in rescue.columns]
if drop_cols:
    print(f"  dropping {len(drop_cols)} rescue-only cols: {drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
    rescue = rescue.drop(columns=drop_cols)
if add_cols:
    print(f"  adding {len(add_cols)} missing-F4 cols as NaN: {add_cols}")
    for c in add_cols:
        rescue[c] = pd.NA

# Reorder: rescue_row_id first (identifier), then F4 col order
ordered = ["rescue_row_id"] + f4_cols_list
rescue = rescue[[c for c in ordered if c in rescue.columns]]
print(f"  final shape: {rescue.shape}")

# ── 4. QA: assert 100% on cols where F4 has 0% NaN ────────────────────────
print()
print("[QA] checking 100% coverage on cols where F4_838 has 0% NaN")
strict = []
for c in f4_cols_list:
    if c == "rescue_row_id": continue
    f4_nan = f4_838[c].isna().sum()
    r_nan  = rescue[c].isna().sum() if c in rescue.columns else len(rescue)
    if f4_nan == 0 and r_nan > 0:
        strict.append((c, r_nan, len(rescue)))
if strict:
    print(f"  ⚠️ {len(strict)} cols still have NaN where F4 has none:")
    for c, n, tot in strict:
        print(f"     {c}: {n}/{tot} NaN")
else:
    print(f"  ✓ all cols with F4 0% NaN are also 0% NaN on rescue")

# ── 5. Save ───────────────────────────────────────────────────────────────
rescue.to_csv(RESCUE_CSV, index=False)
print(f"\nsaved {RESCUE_CSV.name}: {rescue.shape[0]} rows × {rescue.shape[1]} cols")
