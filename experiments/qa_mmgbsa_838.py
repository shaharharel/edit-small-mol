"""QA validation for MM-GBSA 838 single-frame results.

Checks:
  A. Coverage: how many of the 838 visible mols have populated dG_GB_kcalmol?
  B. Distribution: are dG_GB values in plausible range (-100 to +20 kcal/mol)?
  C. Smiles match rate when joining to F4_boltz_full.csv (target >95%).
  D. Spearman correlations with pIC50_mean and boltz_ligand_iptm.
  E. Backend live coverage (after merge + restart).

Usage:
  python experiments/qa_mmgbsa_838.py --mmgbsa results/.../mmgbsa_838_singleframe.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmgbsa", required=True, type=Path)
    ap.add_argument("--f4", default=Path("data/tier4_scored/F4_boltz_full.csv"), type=Path)
    ap.add_argument("--visible-json", default=Path("/tmp/visible_838.json"), type=Path)
    args = ap.parse_args()

    mm = pd.read_csv(args.mmgbsa)
    n_total = len(mm)
    n_ok = int((mm["status"] == "OK").sum())
    n_fail = n_total - n_ok
    print(f"[A] MMGBSA results: {n_total} rows, OK={n_ok}, FAIL={n_fail}, ratio={100*n_ok/max(1,n_total):.1f}%")

    ok = mm[mm["status"] == "OK"]
    vals = ok["dG_GB_kcalmol"].dropna()
    if len(vals) > 0:
        print(f"[B] dG_GB_kcalmol distribution (N={len(vals)}):")
        print(f"    min   = {vals.min():.2f}")
        print(f"    p10   = {vals.quantile(0.10):.2f}")
        print(f"    p25   = {vals.quantile(0.25):.2f}")
        print(f"    med   = {vals.median():.2f}")
        print(f"    p75   = {vals.quantile(0.75):.2f}")
        print(f"    p90   = {vals.quantile(0.90):.2f}")
        print(f"    max   = {vals.max():.2f}")
        print(f"    mean  = {vals.mean():.2f}")
        print(f"    std   = {vals.std():.2f}")
        in_range = ((vals >= -100) & (vals <= 20)).sum()
        print(f"    in [-100, +20] range: {in_range}/{len(vals)} ({100*in_range/len(vals):.1f}%)")

    # Visible cohort coverage
    if args.visible_json.exists():
        with open(args.visible_json) as f:
            visible = json.load(f)
        vis_smi = set(v["smiles"] for v in visible)
        ok_smi = set(ok["smiles"])
        cov = len(vis_smi & ok_smi)
        print(f"[C] Visible-838 coverage: {cov}/{len(vis_smi)} ({100*cov/max(1,len(vis_smi)):.1f}%)")

    # SMILES match rate against F4
    f4 = pd.read_csv(args.f4, low_memory=False)
    f4_smi = set(f4["smiles"].dropna())
    ok_smi = set(ok["smiles"])
    match = len(ok_smi & f4_smi)
    print(f"[D] F4 SMILES match: {match}/{len(ok_smi)} ({100*match/max(1,len(ok_smi)):.1f}%)")

    # Spearman correlations
    if "pIC50_mean" in f4.columns:
        merged = f4.merge(ok[["smiles","dG_GB_kcalmol","boltz_ligand_iptm" if "boltz_ligand_iptm" in ok.columns else "smiles"]].drop_duplicates("smiles"),
                          on="smiles", how="inner")
        # Pull boltz_ligand_iptm from F4 (where it's already populated)
        m2 = merged.dropna(subset=["dG_GB_kcalmol","pIC50_mean"])
        if len(m2) >= 10:
            r, p = spearmanr(m2["dG_GB_kcalmol"], m2["pIC50_mean"])
            # Note: more negative dG_GB = tighter binding = higher pIC50 = expect negative Spearman
            print(f"[E] Spearman(dG_GB_kcalmol, pIC50_mean) = {r:.3f} (p={p:.2e}, N={len(m2)})")
            print(f"    Note: more negative dG_GB = tighter; expect negative Spearman if MM-GBSA aligns with pIC50_mean")

        if "boltz_ligand_iptm" in f4.columns:
            m3 = merged.dropna(subset=["dG_GB_kcalmol", "boltz_ligand_iptm"])
            if len(m3) >= 10:
                r, p = spearmanr(m3["dG_GB_kcalmol"], m3["boltz_ligand_iptm"])
                print(f"[F] Spearman(dG_GB_kcalmol, boltz_ligand_iptm) = {r:.3f} (p={p:.2e}, N={len(m3)})")

    # Top 5 by dG_GB (most negative)
    print("\n[G] Top-5 most-negative dG_GB_kcalmol candidates:")
    top = ok.dropna(subset=["dG_GB_kcalmol"]).sort_values("dG_GB_kcalmol").head(5)
    for _, r in top.iterrows():
        print(f"  dG_GB={r['dG_GB_kcalmol']:.2f}  E_vdw={r['E_vdw_kcalmol']:.2f}  E_eel={r['E_eel_kcalmol']:.2f}  smiles={r['smiles'][:80]}")


if __name__ == "__main__":
    main()
