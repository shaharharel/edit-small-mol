#!/usr/bin/env python3
"""Analyze L2 scaffold-anchor C1 vs C5 docking results and write a markdown report.

Inputs:
  data/lingo3dmol_docking/C1_vina.csv
  data/lingo3dmol_docking/C5_vina.csv
  data/lingo3dmol_docking/combined_results.json

Output:
  /tmp/l2_c1_vs_c5_docking.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "lingo3dmol_docking"
REPORT_PATH = Path("/tmp/l2_c1_vs_c5_docking.md")


def percentiles(arr, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def summarize_col(df, col):
    vals = df[col].dropna().astype(float).values
    if len(vals) == 0:
        return {"n": 0}
    out = {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
    out.update(percentiles(vals))
    return out


def main():
    c1 = pd.read_csv(OUT_DIR / "C1_vina.csv")
    c5 = pd.read_csv(OUT_DIR / "C5_vina.csv")

    c1_ok = c1[c1["success"] == True].copy()
    c5_ok = c5[c5["success"] == True].copy()

    summaries = {}
    for name, df in [("C1", c1_ok), ("C5", c5_ok)]:
        summaries[name] = {
            "n_input": len(c1) if name == "C1" else len(c5),
            "n_success": len(df),
            "vina_kcalmol": summarize_col(df, "vina_kcalmol"),
            "d_warhead_Cbeta_to_Cys346SG": summarize_col(df, "d_warhead_Cbeta_to_Cys346SG"),
            "d_recognition_arm_to_Met414N": summarize_col(df, "d_recognition_arm_to_Met414N"),
            "d_recognition_arm_to_Met414N_min": summarize_col(df, "d_recognition_arm_to_Met414N_min"),
            "warhead_found_pct": float(100 * df["warhead_found"].fillna(False).astype(bool).mean()) if "warhead_found" in df.columns else None,
        }

    # Statistical tests
    tests = {}
    v1 = c1_ok["vina_kcalmol"].dropna().astype(float).values
    v5 = c5_ok["vina_kcalmol"].dropna().astype(float).values

    if len(v1) > 5 and len(v5) > 5:
        ks_stat, ks_p = stats.ks_2samp(v1, v5)
        mw_stat, mw_p = stats.mannwhitneyu(v1, v5, alternative="greater")  # H1: C1 > C5 (C1 weaker)
        tests["vina_kcalmol"] = {
            "ks": {"stat": float(ks_stat), "p": float(ks_p)},
            "mannwhitney_C1_gt_C5": {"stat": float(mw_stat), "p": float(mw_p)},
            "mean_diff_C1_minus_C5": float(v1.mean() - v5.mean()),
            "median_diff_C1_minus_C5": float(np.median(v1) - np.median(v5)),
        }

    d1 = c1_ok["d_warhead_Cbeta_to_Cys346SG"].dropna().astype(float).values
    d5 = c5_ok["d_warhead_Cbeta_to_Cys346SG"].dropna().astype(float).values
    if len(d1) > 5 and len(d5) > 5:
        ks_stat, ks_p = stats.ks_2samp(d1, d5)
        mw_stat, mw_p = stats.mannwhitneyu(d1, d5, alternative="greater")  # H1: C1 farther
        tests["d_warhead_to_Cys346SG"] = {
            "ks": {"stat": float(ks_stat), "p": float(ks_p)},
            "mannwhitney_C1_gt_C5": {"stat": float(mw_stat), "p": float(mw_p)},
            "median_diff_C1_minus_C5": float(np.median(d1) - np.median(d5)),
            "C1_pct_within_5A": float(100 * (d1 <= 5.0).mean()),
            "C5_pct_within_5A": float(100 * (d5 <= 5.0).mean()),
        }

    m1 = c1_ok["d_recognition_arm_to_Met414N_min"].dropna().astype(float).values
    m5 = c5_ok["d_recognition_arm_to_Met414N_min"].dropna().astype(float).values
    if len(m1) > 5 and len(m5) > 5:
        ks_stat, ks_p = stats.ks_2samp(m1, m5)
        mw_stat, mw_p = stats.mannwhitneyu(m1, m5, alternative="greater")  # H1: C1 farther from hinge
        tests["d_arm_to_Met414N_min"] = {
            "ks": {"stat": float(ks_stat), "p": float(ks_p)},
            "mannwhitney_C1_gt_C5": {"stat": float(mw_stat), "p": float(mw_p)},
            "median_diff_C1_minus_C5": float(np.median(m1) - np.median(m5)),
            "C1_pct_within_6A": float(100 * (m1 <= 6.0).mean()),
            "C5_pct_within_6A": float(100 * (m5 <= 6.0).mean()),
        }

    # Top 5 per cohort by Vina
    def top5(df):
        return df.dropna(subset=["vina_kcalmol"]).nsmallest(5, "vina_kcalmol")[
            ["name", "smi", "vina_kcalmol", "d_warhead_Cbeta_to_Cys346SG",
             "d_recognition_arm_to_Met414N_min"]
        ]

    top_c1 = top5(c1_ok)
    top_c5 = top5(c5_ok)

    # ------- Write markdown -------
    def fmt(v, prec=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.{prec}f}"

    s1, s5 = summaries["C1"], summaries["C5"]
    md = []
    md.append("# L2 Scaffold-Anchor Docking: C1 (aliphatic) vs C5 (aromatic) — ZAP70 4K2R\n")
    md.append("**Setup.** AutoDock Vina v1.2.7, exhaustiveness=8, num_modes=9, "
              "box 20×20×20 Å centered on Cys346 SG (18.888, −3.650, −29.979). "
              "Receptor reused from `data/docking_500/receptor.pdbqt` (PDB 4K2R, "
              "prepared previously). Ligands prepared via meeko/MoleculePreparation "
              "from the inpaint SDF 3D coordinates (fallback ETKDGv3 re-embed if "
              "meeko fails on the SDF). CPU-only.\n")

    md.append("## Vina docking score (kcal/mol, lower is better)\n")
    md.append("| Cohort | n_in | n_dock | median | p25 | p75 | best | worst |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, s in [("C1", s1), ("C5", s5)]:
        v = s["vina_kcalmol"]
        md.append(f"| {name} | {s['n_input']} | {s['n_success']} | "
                  f"{fmt(v.get('median'))} | {fmt(v.get('p25'))} | "
                  f"{fmt(v.get('p75'))} | {fmt(v.get('min'))} | "
                  f"{fmt(v.get('max'))} |")
    if "vina_kcalmol" in tests:
        t = tests["vina_kcalmol"]
        md.append("")
        md.append(f"- Mean(C1) − Mean(C5) = **{t['mean_diff_C1_minus_C5']:+.3f}** kcal/mol "
                  f"(positive ⇒ C5 binds tighter on average)")
        md.append(f"- Median(C1) − Median(C5) = **{t['median_diff_C1_minus_C5']:+.3f}** kcal/mol")
        md.append(f"- Mann-Whitney one-sided (H₁: C1 > C5, i.e. C5 better): "
                  f"U={t['mannwhitney_C1_gt_C5']['stat']:.1f}, "
                  f"p={t['mannwhitney_C1_gt_C5']['p']:.3g}")
        md.append(f"- KS two-sample: D={t['ks']['stat']:.3f}, p={t['ks']['p']:.3g}\n")

    md.append("## Warhead C-β → Cys346 SG distance (Å, ≤5 Å = prereactive complex)\n")
    md.append("| Cohort | median | p25 | p75 | % ≤5 Å | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    for name, s in [("C1", s1), ("C5", s5)]:
        d = s["d_warhead_Cbeta_to_Cys346SG"]
        pct_5 = (tests.get("d_warhead_to_Cys346SG", {}).get(f"{name}_pct_within_5A"))
        md.append(f"| {name} | {fmt(d.get('median'))} | {fmt(d.get('p25'))} | "
                  f"{fmt(d.get('p75'))} | {fmt(pct_5, 1) if pct_5 is not None else 'n/a'} | "
                  f"{fmt(d.get('min'))} | {fmt(d.get('max'))} |")
    if "d_warhead_to_Cys346SG" in tests:
        t = tests["d_warhead_to_Cys346SG"]
        md.append("")
        md.append(f"- Median(C1) − Median(C5) = **{t['median_diff_C1_minus_C5']:+.3f}** Å")
        md.append(f"- Mann-Whitney one-sided (H₁: C1 farther): "
                  f"p={t['mannwhitney_C1_gt_C5']['p']:.3g}\n")

    md.append("## Recognition arm → Met414 backbone N distance (Å, hinge proxy)\n")
    md.append("Min distance from any non-warhead heavy atom to Met414 N. "
              "Lower ⇒ recognition arm directed toward the hinge.\n")
    md.append("| Cohort | median | p25 | p75 | % ≤6 Å | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    for name, s in [("C1", s1), ("C5", s5)]:
        d = s["d_recognition_arm_to_Met414N_min"]
        pct_6 = (tests.get("d_arm_to_Met414N_min", {}).get(f"{name}_pct_within_6A"))
        md.append(f"| {name} | {fmt(d.get('median'))} | {fmt(d.get('p25'))} | "
                  f"{fmt(d.get('p75'))} | {fmt(pct_6, 1) if pct_6 is not None else 'n/a'} | "
                  f"{fmt(d.get('min'))} | {fmt(d.get('max'))} |")
    if "d_arm_to_Met414N_min" in tests:
        t = tests["d_arm_to_Met414N_min"]
        md.append("")
        md.append(f"- Median(C1) − Median(C5) = **{t['median_diff_C1_minus_C5']:+.3f}** Å")
        md.append(f"- Mann-Whitney one-sided (H₁: C1 farther from hinge): "
                  f"p={t['mannwhitney_C1_gt_C5']['p']:.3g}\n")

    md.append("## Top-5 docked molecules per cohort\n")
    for name, df in [("C1", top_c1), ("C5", top_c5)]:
        md.append(f"**{name} cohort — best 5 by Vina score**\n")
        md.append("| Rank | Vina | d(Cβ,SG) Å | d(arm,Met414N) Å | SMILES |")
        md.append("|---|---|---|---|---|")
        for i, (_, r) in enumerate(df.iterrows(), 1):
            md.append(f"| {i} | {fmt(r['vina_kcalmol'])} | "
                      f"{fmt(r['d_warhead_Cbeta_to_Cys346SG'])} | "
                      f"{fmt(r['d_recognition_arm_to_Met414N_min'])} | "
                      f"`{r['smi']}` |")
        md.append("")

    # ----- Verdict -----
    md.append("## Verdict\n")
    verdict_lines = []
    if "vina_kcalmol" in tests:
        t = tests["vina_kcalmol"]
        better_vina = t["median_diff_C1_minus_C5"] > 0
        sig_vina = t["mannwhitney_C1_gt_C5"]["p"] < 0.05
        verdict_lines.append(
            f"- **Affinity:** C5 docks "
            f"{'tighter' if better_vina else 'NOT tighter'} than C1 "
            f"(ΔmedianVina = {t['median_diff_C1_minus_C5']:+.2f} kcal/mol, "
            f"Mann-Whitney p={t['mannwhitney_C1_gt_C5']['p']:.3g}, "
            f"{'significant' if sig_vina else 'not significant'} at α=0.05).")
    if "d_warhead_to_Cys346SG" in tests:
        t = tests["d_warhead_to_Cys346SG"]
        c1_pct = t.get("C1_pct_within_5A", 0)
        c5_pct = t.get("C5_pct_within_5A", 0)
        verdict_lines.append(
            f"- **Covalent reach:** {c5_pct:.0f}% of C5 vs {c1_pct:.0f}% of C1 "
            f"poses place Cβ within 5 Å of Cys346 SG "
            f"(prereactive-complex distance).")
    if "d_arm_to_Met414N_min" in tests:
        t = tests["d_arm_to_Met414N_min"]
        c1_pct = t.get("C1_pct_within_6A", 0)
        c5_pct = t.get("C5_pct_within_6A", 0)
        better_hinge = t["median_diff_C1_minus_C5"] > 0
        verdict_lines.append(
            f"- **Hinge orientation:** {c5_pct:.0f}% of C5 vs {c1_pct:.0f}% of C1 "
            f"poses bring the recognition arm within 6 Å of Met414 N "
            f"(Δmedian = {t['median_diff_C1_minus_C5']:+.2f} Å; "
            f"C5 arm "
            f"{'closer to hinge' if better_hinge else 'NOT closer to hinge'} "
            f"than C1).")
    if not verdict_lines:
        verdict_lines.append("- Insufficient successful docks for statistical comparison.")
    md.extend(verdict_lines)

    md.append("\n**Conclusion.** ")
    if "vina_kcalmol" in tests and "d_arm_to_Met414N_min" in tests:
        affinity_win = tests["vina_kcalmol"]["median_diff_C1_minus_C5"] > 0
        hinge_win = tests["d_arm_to_Met414N_min"]["median_diff_C1_minus_C5"] > 0
        if affinity_win and hinge_win:
            md.append("Vina and hinge-geometry evidence both favor C5 → the "
                      "regiochemistry fix is empirically validated.")
        elif affinity_win and not hinge_win:
            md.append("Vina favors C5 but the arm is not closer to Met414 — "
                      "C5 binds tighter via other interactions, not specifically "
                      "via hinge reach.")
        elif (not affinity_win) and hinge_win:
            md.append("C5 arm is closer to the hinge but Vina doesn't separate "
                      "the cohorts — the rigid-receptor scoring may be "
                      "insensitive to the regiochemistry.")
        else:
            md.append("Neither Vina nor hinge geometry favors C5 — CHR may "
                      "have over-discounted C1, or the C1 cohort happens to "
                      "fit the pocket in some other valid orientation.")
    md.append("\n")

    REPORT_PATH.write_text("\n".join(md))
    print(f"Report written: {REPORT_PATH}")
    # Also save analysis JSON next to CSVs
    analysis = {"summaries": summaries, "tests": tests}
    (OUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"Analysis JSON: {OUT_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
