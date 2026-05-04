#!/usr/bin/env python3
"""
Tier 3 — Constrained Generative.

Take the existing 39,203 Phase 1+2 candidates (BRICS/CReM/MMP/Mol2Mol/De Novo/LibInvent)
and apply the warhead-intact constraint as a hard filter. Then re-score everything with
the common scoring suite (SAScore + 3D shape Tc + warhead vector deviation + Tc to train).

This gives us:
  (a) the warhead retention rate per generator (diagnostic)
  (b) a properly-constrained Tier 3 candidate set comparable to Tier 1/2

Notes:
  - The original LibInvent scaffold decorated the warhead position [*:0], so many
    LibInvent products won't have the warhead. This is expected and is part of
    what we measure here.
  - A full REINVENT4 re-run with a properly locked warhead scaffold + custom
    warhead-intact scoring component is a separate follow-up (Tier 3b).

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tier2_5_postfilter.py
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_film_predictor, load_zap70_train_smiles,
)


PROJECT_ROOT = Path(__file__).parent.parent
EXPANSION_CSV = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_expansion" / "all_scored_candidates.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_tier2_5_postfilter"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "tier2_5_results.json"
OUT_CSV = RESULTS_DIR / "tier2_5_candidates.csv"

# How many top-of-each-method to keep + score with the heavy 3D pipeline
TOP_PER_METHOD = 200


def main():
    print("=" * 70)
    print("MOL 1 — TIER 2.5: WARHEAD POST-FILTER (DIAGNOSTIC) (warhead-intact filter)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not EXPANSION_CSV.exists():
        print(f"ERROR: {EXPANSION_CSV} not found. Run mol1_expansion + mol1_reinvent4 first.")
        return

    df = pd.read_csv(EXPANSION_CSV)
    print(f"[+] Loaded {len(df):,} Phase 1+2 candidates")

    # Apply warhead filter
    print("[+] Applying warhead-intact SMARTS filter...")
    df["warhead_intact"] = df["smiles"].apply(warhead_intact)
    pre = len(df)
    df_kept = df[df["warhead_intact"]].copy()
    print(f"  Warhead-intact: {len(df_kept):,} / {pre:,} ({100*len(df_kept)/pre:.1f}%)")

    # Per-method retention diagnostic
    retention = {}
    for method, sub in df.groupby("method"):
        retained = int(sub["warhead_intact"].sum())
        retention[method] = {
            "total": int(len(sub)),
            "warhead_intact": retained,
            "retention_pct": float(100 * retained / max(1, len(sub))),
        }
    print("\n  Warhead retention per generator:")
    for m, r in sorted(retention.items(), key=lambda x: -x[1]["retention_pct"]):
        print(f"    {m:<12s} {r['warhead_intact']:>6d} / {r['total']:>6d}  ({r['retention_pct']:5.1f}%)")

    # Pick top-N-per-method (to keep heavy 3D scoring tractable)
    df_kept = df_kept.sort_values("pIC50", ascending=False)
    top_groups = []
    for method, sub in df_kept.groupby("method"):
        top_groups.append(sub.head(TOP_PER_METHOD))
    df_top = pd.concat(top_groups, ignore_index=True)
    df_top = df_top.sort_values("pIC50", ascending=False).reset_index(drop=True)
    print(f"\n[+] Top-{TOP_PER_METHOD} per method (warhead-intact): {len(df_top)} candidates")

    # Score with full 3D suite
    print("[+] Scoring (SAScore + PAINS + shape Tc + warhead vector)...")
    train = load_zap70_train_smiles()
    pred = load_film_predictor()
    df_scored = score_dataframe(
        df_top.rename(columns={'pIC50': 'pIC50_phase'}),
        smiles_col="smiles",
        train_smiles=train,
        compute_3d=True,
        pIC50_predictor=pred,
    )
    df_scored = df_scored.sort_values("pIC50", ascending=False).reset_index(drop=True)

    # Save
    df_scored.to_csv(OUT_CSV, index=False)
    out = {
        "tier": 2.5, "method": "constrained_generative_warhead_filter",
        "seed": MOL1_SMILES,
        "input_total": pre,
        "warhead_intact_total": int(df["warhead_intact"].sum()),
        "n_candidates_scored": len(df_scored),
        "top_per_method": TOP_PER_METHOD,
        "retention_per_method": retention,
        "timestamp": datetime.now().isoformat(),
        "candidates": df_scored.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2))
    print(f"[+] Saved → {OUT_JSON}")

    # Print top-10
    print("\nTop 10 by FiLMDelta pIC50 (warhead-intact):")
    print(f"{'#':<3}{'pIC50':<7}{'SAS':<6}{'shape_Tc':<10}{'wrhd_dev':<10}{'method':<12s}")
    for i, r in df_scored.head(10).iterrows():
        print(f"{i+1:<3}{r['pIC50']:<7.3f}{r['SAScore']:<6.2f}"
              f"{r['shape_Tc_seed']:<10.3f}{r['warhead_dev_deg']:<10.1f}"
              f"{r['method']:<12s}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
