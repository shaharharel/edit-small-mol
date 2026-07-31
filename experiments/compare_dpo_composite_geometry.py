"""Pairwise geometric comparison: DPO-composite cohort vs baselines.

Reuses `covft_geometric_options_bc` machinery:
  - 2D acrylamide planarity (pre-reactivity score, planar dev)
  - Cov-Vina docking (vina affinity, warhead-Sγ distance, BD angle)
  - Mann-Whitney + Cliff's δ

Baselines:
  rl         — experiments/exp_geom_bc/samples_rl.csv         (un-DPO'd covFT+RL)
  covft      — experiments/exp_covft_value/samples_covft.csv
  libinvent  — experiments/exp_geom_bc/samples_libinvent.csv

Optional baselines (if present):
  m1a        — data/m1a_cohorts/cohort_mol1_zap70.csv (Stream A's M1a cohort)

Subject:
  dpo_composite — data/dpo_composite_cohort/cohort_mol1.csv (this stream's cohort)

Outputs:
  results/paper_evaluation/dpo_composite_geometry.json
  results/paper_evaluation/dpo_composite_geometry.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import covft_geometric_options_bc as G  # type: ignore  # noqa: E402

DEFAULT_BASELINES = ["rl", "covft", "libinvent"]
DEFAULT_DPO_SOURCE = PROJECT_ROOT / "data/dpo_composite_cohort/cohort_mol1.csv"
DEFAULT_M1A_SOURCE = PROJECT_ROOT / "data/m1a_cohorts/cohort_mol1_zap70.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_source", default=str(DEFAULT_DPO_SOURCE))
    ap.add_argument("--m1a_source", default=str(DEFAULT_M1A_SOURCE))
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--workers-2d", type=int, default=G.WORKERS_2D)
    ap.add_argument("--workers-cov", type=int, default=G.WORKERS_COVDOCK)
    ap.add_argument("--n-cov-vina", type=int, default=G.N_COV_VINA)
    ap.add_argument("--seed", type=int, default=G.SUBSAMPLE_SEED)
    ap.add_argument("--out_dir", default=str(G.OUT_DIR))
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    G.OUT_DIR = OUT_DIR  # honored by load_or_compute_cohort

    G.prepare_stripped_receptor()

    # Register the dpo and (optional) m1a cohorts
    dpo_path = Path(args.dpo_source)
    if not dpo_path.exists():
        sys.exit(f"DPO source missing: {dpo_path}")
    G.COHORT_SOURCES["dpo_composite"] = dpo_path

    extra = ["dpo_composite"]
    m1a_path = Path(args.m1a_source)
    if m1a_path.exists():
        G.COHORT_SOURCES["m1a"] = m1a_path
        extra.append("m1a")
        print(f"[info] including M1a cohort at {m1a_path}")
    else:
        print(f"[info] M1a cohort not found ({m1a_path}); comparison will skip m1a")

    requested = list(dict.fromkeys(extra + args.baselines))
    print(f"[info] cohorts to load/compute: {requested}")
    cohort_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for c in requested:
        src = G.COHORT_SOURCES.get(c)
        if src is None or not Path(src).exists():
            print(f"[warn] skipping {c}: source missing ({src})")
            continue
        df2d, dfcov = G.load_or_compute_cohort(
            c, Path(src),
            args.workers_2d, args.workers_cov,
            args.n_cov_vina, args.seed,
            force=args.force_recompute,
        )
        cohort_data[c] = (df2d, dfcov)

    cohort_summaries = {c: G.cohort_summary(d2, dcov) for c, (d2, dcov) in cohort_data.items()}

    # Pairwise: dpo_composite vs each baseline + (optionally) vs m1a
    pairwise = {}
    if "dpo_composite" not in cohort_data:
        sys.exit("dpo_composite cohort not loaded")

    df2d_a, dfcov_a = cohort_data["dpo_composite"]
    for b in args.baselines + (["m1a"] if "m1a" in cohort_data else []):
        if b not in cohort_data:
            continue
        df2d_b, dfcov_b = cohort_data[b]
        stats = G.pairwise_stats("dpo_composite", df2d_a, dfcov_a, b, df2d_b, dfcov_b)
        verdict = G.pairwise_verdict(stats, "dpo_composite", b)
        pairwise[f"dpo_composite_vs_{b}"] = {"stats": stats, "verdict": verdict}

    # Headline: count how many baselines DPO beats on >=2 axes
    n_beats = sum(1 for k, v in pairwise.items() if v["verdict"]["positive_axes_count"] >= 2)
    n_mixed = sum(1 for k, v in pairwise.items() if v["verdict"]["positive_axes_count"] == 1)
    n_none  = sum(1 for k, v in pairwise.items() if v["verdict"]["positive_axes_count"] == 0)
    n_total = len(pairwise)
    if n_beats >= max(2, n_total - 1):
        headline = (
            f"DPO-composite IS significantly more geometrically pre-reactive than "
            f"{n_beats}/{n_total} baselines (clean win)."
        )
    elif n_beats >= 1:
        headline = (
            f"DPO-composite beats {n_beats}/{n_total} baselines on >=2 axes, "
            f"mixed on {n_mixed}/{n_total}, indistinct on {n_none}/{n_total}."
        )
    else:
        headline = (
            f"No evidence DPO-composite improves geometry over baselines "
            f"(0/{n_total} clean wins; mixed={n_mixed}, indistinct={n_none})."
        )

    out = {
        "config": {
            "subject": "dpo_composite",
            "baselines": args.baselines + (["m1a"] if "m1a" in cohort_data else []),
            "alpha": 0.05,
            "cov_vina_subsample_n": args.n_cov_vina,
            "subsample_seed": args.seed,
            "acrylamide_smarts": G.ACRYLAMIDE_SMARTS,
            "planarity_threshold_deg": G.PLANARITY_THRESH_DEG,
            "dpo_source": str(dpo_path),
        },
        "cohort_summaries": cohort_summaries,
        "pairwise_tests": pairwise,
        "headline_verdict": {"summary": headline,
                              "clean_wins": n_beats,
                              "mixed": n_mixed,
                              "indistinct": n_none,
                              "n_baselines": n_total},
    }

    out_json = OUT_DIR / "dpo_composite_geometry.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_json}")

    # Markdown
    lines = []
    lines.append("# DPO-composite (Stream B) geometric comparison\n")
    lines.append(f"**Headline**: {headline}\n")
    lines.append(f"**DPO source**: `{dpo_path}`\n")
    lines.append("## Per-cohort summary\n")
    keys = ["n_total", "n_acrylamide_match", "frac_acrylamide_match",
            "pre_reactivity_score_mean", "pre_reactivity_score_frac_ge_0p5",
            "planar_dev_deg_median",
            "vina_affinity_median", "warhead_sg_distance_median_A",
            "bd_angle_median_deg", "pose_converged_rate",
            "cov_vina_n_attempted", "cov_vina_n_ok"]
    lines.append("| metric | " + " | ".join(cohort_summaries.keys()) + " |")
    lines.append("|" + "---|" * (len(cohort_summaries) + 1))
    for k in keys:
        row = [k]
        for cn in cohort_summaries.keys():
            v = cohort_summaries[cn].get(k)
            if isinstance(v, float):
                row.append(f"{v:.3f}")
            elif v is None:
                row.append("-")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Pairwise tests (dpo_composite vs baseline)\n")
    for key, payload in pairwise.items():
        lines.append(f"### {key}\n")
        lines.append(f"- Verdict: **{payload['verdict']['verdict']}**")
        lines.append(f"- Positive axes (a-favors): {payload['verdict']['positive_axes_count']}/3")
        for axis in ("mw_pre_reactivity_score", "mw_planar_dev_deg",
                     "mw_vina_affinity", "mw_warhead_sg_distance"):
            mw = payload["stats"][axis]
            lines.append(
                f"  - {axis}: median_a={mw['median_a']}, median_b={mw['median_b']}, "
                f"U={mw['U']}, p={mw['p_value']}, δ={mw['cliffs_delta_a_vs_b']}"
            )
        lines.append("")
    out_md = OUT_DIR / "dpo_composite_geometry.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
