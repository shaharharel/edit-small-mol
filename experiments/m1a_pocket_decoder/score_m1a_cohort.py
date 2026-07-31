"""Score the M1a cohort using the geometry pipeline from
experiments/covft_geometric_options_bc.py, then run pairwise tests vs covft,
RL, and LibInvent cohorts.

Outputs:
  results/paper_evaluation/m1a_geometry.json
  results/paper_evaluation/m1a_geometry.md
  results/paper_evaluation/covft_geometric_2d_m1a.csv
  results/paper_evaluation/covft_geometric_covvina_m1a.csv

This is run LOCALLY (not on the VM) because the receptor + tools/vina + scoring
infrastructure is already wired up here.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.covft_geometric_options_bc import (
    OUT_DIR,
    WORK_DIR,
    ACRYLAMIDE_SMARTS,
    PLANARITY_THRESH_DEG,
    N_COV_VINA,
    SUBSAMPLE_SEED,
    WORKERS_2D,
    WORKERS_COVDOCK,
    COHORT_SOURCES,
    run_2d_panel,
    run_cov_vina,
    cohort_summary,
    pairwise_stats,
    pairwise_verdict,
    mw_summary,
    cliffs_delta,
    iqr,
    fmt,
    fmt_iqr,
    load_or_compute_cohort,
)
from experiments.run_covalent_docking import prepare_stripped_receptor


M1A_COHORT_NAME = "m1a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1a-csv", required=True,
                     help="Path to M1a sampled cohort CSV (columns: SMILES, Input_SMILES, NLL)")
    ap.add_argument("--workers-2d", type=int, default=WORKERS_2D)
    ap.add_argument("--workers-cov", type=int, default=WORKERS_COVDOCK)
    ap.add_argument("--n-cov-vina", type=int, default=N_COV_VINA)
    ap.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    ap.add_argument("--baselines", nargs="+", default=["covft", "rl", "libinvent"])
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing receptor...", flush=True)
    prepare_stripped_receptor()

    # ---- Score M1a cohort ----
    m1a_2d_path = OUT_DIR / "covft_geometric_2d_m1a.csv"
    m1a_cov_path = OUT_DIR / "covft_geometric_covvina_m1a.csv"
    if (not args.force_recompute) and m1a_2d_path.exists() and m1a_cov_path.exists():
        print(f"[load] m1a from cached CSVs", flush=True)
        df_2d_m = pd.read_csv(m1a_2d_path)
        df_cov_m = pd.read_csv(m1a_cov_path)
        for col in ("acryl_match", "embed_ok"):
            if col in df_2d_m.columns:
                df_2d_m[col] = df_2d_m[col].astype(bool)
        for col in ("warhead_found", "lig_prep_ok", "vina_ok", "pose_converged"):
            if col in df_cov_m.columns:
                df_cov_m[col] = df_cov_m[col].astype(bool)
    else:
        print(f"[compute] m1a from {args.m1a_csv}", flush=True)
        df = pd.read_csv(args.m1a_csv)
        assert "SMILES" in df.columns, f"missing SMILES col in {args.m1a_csv}"
        df_2d_m = run_2d_panel(df, M1A_COHORT_NAME, args.workers_2d)
        df_2d_m.to_csv(m1a_2d_path, index=False)
        acryl = df_2d_m[df_2d_m["acryl_match"] == True].reset_index(drop=True)  # noqa: E712
        df_cov_m = run_cov_vina(acryl, M1A_COHORT_NAME, args.n_cov_vina, args.seed,
                                  args.workers_cov)
        df_cov_m.to_csv(m1a_cov_path, index=False)

    # ---- Load baselines ----
    cohort_data = {M1A_COHORT_NAME: (df_2d_m, df_cov_m)}
    for b in args.baselines:
        src = COHORT_SOURCES.get(b)
        if src is None or not src.exists():
            print(f"WARNING: baseline {b} source missing", flush=True)
            continue
        df2d, dfcov = load_or_compute_cohort(
            b, src, args.workers_2d, args.workers_cov,
            args.n_cov_vina, args.seed, force=False)
        cohort_data[b] = (df2d, dfcov)

    # ---- Per-cohort summaries ----
    summaries = {c: cohort_summary(df2d, dfcov)
                  for c, (df2d, dfcov) in cohort_data.items()}

    # ---- Pairwise tests: m1a vs each baseline ----
    pairwise = {}
    for b in args.baselines:
        if b not in cohort_data:
            continue
        df2a, dfca = cohort_data[M1A_COHORT_NAME]
        df2b, dfcb = cohort_data[b]
        stats = pairwise_stats(M1A_COHORT_NAME, df2a, dfca, b, df2b, dfcb)
        verdict = pairwise_verdict(stats, M1A_COHORT_NAME, b)
        pairwise[f"m1a_vs_{b}"] = {"stats": stats, "verdict": verdict}

    # ---- Headline verdicts ----
    def _pos(key):
        return pairwise.get(key, {}).get("verdict", {}).get("positive_axes_count", 0)
    n_pos_vs_covft = _pos("m1a_vs_covft")
    n_pos_vs_rl    = _pos("m1a_vs_rl")
    n_pos_vs_lib   = _pos("m1a_vs_libinvent")

    headline = {
        "m1a_vs_covft_positive_axes": n_pos_vs_covft,
        "m1a_vs_rl_positive_axes": n_pos_vs_rl,
        "m1a_vs_libinvent_positive_axes": n_pos_vs_lib,
    }
    if n_pos_vs_covft >= 2:
        headline["m1a_beats_covft"] = True
        beats_covft_str = (
            f"M1a IS significantly more geometrically pre-reactive than covFT "
            f"(positive axes {n_pos_vs_covft}/3)."
        )
    elif n_pos_vs_covft == 1:
        headline["m1a_beats_covft"] = "mixed"
        beats_covft_str = f"Mixed: M1a beats covFT on only 1 axis ({n_pos_vs_covft}/3)."
    else:
        headline["m1a_beats_covft"] = False
        beats_covft_str = (
            f"M1a is NOT significantly more geometrically pre-reactive than covFT "
            f"(positive axes {n_pos_vs_covft}/3)."
        )
    if n_pos_vs_lib >= 2:
        headline["m1a_beats_libinvent"] = True
        beats_lib_str = (
            f"M1a matches/beats the LibInvent geometric ceiling ({n_pos_vs_lib}/3)."
        )
    elif n_pos_vs_lib == 1:
        headline["m1a_beats_libinvent"] = "mixed"
        beats_lib_str = f"Mixed vs LibInvent ({n_pos_vs_lib}/3)."
    else:
        headline["m1a_beats_libinvent"] = False
        beats_lib_str = f"M1a does NOT match the LibInvent geometric ceiling ({n_pos_vs_lib}/3)."

    out = {
        "config": {
            "m1a_csv": args.m1a_csv,
            "cohorts": [M1A_COHORT_NAME] + args.baselines,
            "alpha": 0.05,
            "cov_vina_subsample_n": args.n_cov_vina,
            "subsample_seed": args.seed,
            "acrylamide_smarts": ACRYLAMIDE_SMARTS,
            "planarity_threshold_deg": PLANARITY_THRESH_DEG,
        },
        "cohort_summaries": summaries,
        "pairwise_tests": pairwise,
        "headline_verdict": {
            "summary_vs_covft": beats_covft_str,
            "summary_vs_libinvent": beats_lib_str,
            **headline,
        },
    }

    out_json = OUT_DIR / "m1a_geometry.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_json}", flush=True)

    # ---- Markdown ----
    md_lines = []
    md_lines.append("# M1a (pocket+warhead-pose conditioned mol2mol) geometry results\n")
    md_lines.append(f"Source CSV: `{args.m1a_csv}`\n")
    md_lines.append(f"N samples scored: {summaries[M1A_COHORT_NAME]['n_total']}, "
                     f"cov-Vina subsample N: {args.n_cov_vina}\n")
    md_lines.append("\n## Cohort summaries\n")
    md_lines.append("| Metric | M1a | " + " | ".join(args.baselines) + " |")
    md_lines.append("|---|---|" + "|".join(["---"] * len(args.baselines)) + "|")

    def row(label, key, nd=3):
        cells = [label]
        for c in [M1A_COHORT_NAME] + args.baselines:
            v = summaries.get(c, {}).get(key)
            cells.append(fmt(v, nd))
        return "| " + " | ".join(cells) + " |"

    md_lines.append(row("n_total", "n_total", 0))
    md_lines.append(row("frac_acrylamide_match", "frac_acrylamide_match", 3))
    md_lines.append(row("pre_reactivity_score_mean", "pre_reactivity_score_mean", 3))
    md_lines.append(row("pre_reactivity_score_frac>=0.5", "pre_reactivity_score_frac_ge_0p5", 3))
    md_lines.append(row("planar_dev_deg_median", "planar_dev_deg_median", 2))
    md_lines.append(row("cov_vina_n_ok", "cov_vina_n_ok", 0))
    md_lines.append(row("vina_affinity_median", "vina_affinity_median", 3))
    md_lines.append(row("warhead_sg_distance_median_A", "warhead_sg_distance_median_A", 3))
    md_lines.append(row("bd_angle_median_deg", "bd_angle_median_deg", 2))
    md_lines.append(row("pose_converged_rate", "pose_converged_rate", 3))
    md_lines.append("")

    md_lines.append("\n## Pairwise tests (M1a vs baseline)\n")
    md_lines.append("Mann-Whitney U two-sided, Cliff's δ. Positive δ means M1a tends to be larger.")
    md_lines.append("- pre_reactivity_score: larger is better (more frequently planar warhead)")
    md_lines.append("- planar_dev_deg: smaller is better (closer to planar)")
    md_lines.append("- vina_affinity: smaller is better (kcal/mol, more negative)")
    md_lines.append("- warhead_sg_distance_A: closer to 1.85 is better\n")
    md_lines.append("| Baseline | metric | p | δ | M1a_med | base_med |")
    md_lines.append("|---|---|---|---|---|---|")
    for b in args.baselines:
        key = f"m1a_vs_{b}"
        if key not in pairwise:
            continue
        s = pairwise[key]["stats"]
        for metric in ("mw_pre_reactivity_score", "mw_planar_dev_deg",
                        "mw_vina_affinity", "mw_warhead_sg_distance"):
            m = s.get(metric, {})
            md_lines.append(
                f"| {b} | {metric.replace('mw_','')} | "
                f"{fmt(m.get('p_value'), 4)} | {fmt(m.get('cliffs_delta_a_vs_b'), 3)} | "
                f"{fmt(m.get('median_a'), 3)} | {fmt(m.get('median_b'), 3)} |"
            )
    md_lines.append("")
    md_lines.append("\n## Headline verdict\n")
    md_lines.append(f"- vs covFT: **{beats_covft_str}**")
    md_lines.append(f"- vs LibInvent: **{beats_lib_str}**")
    md_lines.append(f"- vs RL: positive_axes={n_pos_vs_rl}/3")

    out_md = OUT_DIR / "m1a_geometry.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Wrote {out_md}", flush=True)

    print("\n" + "=" * 60)
    print("M1a vs covFT:", beats_covft_str)
    print("M1a vs RL: positive_axes =", n_pos_vs_rl, "/3")
    print("M1a vs LibInvent:", beats_lib_str)
    print("=" * 60)


if __name__ == "__main__":
    main()
