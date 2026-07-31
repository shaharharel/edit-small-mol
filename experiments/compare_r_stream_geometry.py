"""Generic geometry compare: R-stream cohort vs baselines.

Reuses covft_geometric_options_bc machinery for 2D ETKDG dihedral + cov-Vina
docking, then runs Mann-Whitney + Cliff's δ pairwise.

Usage:
    conda run --no-capture-output -n quris python experiments/compare_r_stream_geometry.py \
        --subject r1 \
        --subject_csv data/r1_regularized_dpo_cohort/cohort_mol1.csv \
        --baselines covft rl libinvent m1a dpo_composite \
        --out_dir results/paper_evaluation
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import covft_geometric_options_bc as G  # type: ignore

DEFAULT_BASELINES = ["covft", "rl", "libinvent"]
EXTRA_SOURCES = {
    "m1a": PROJECT_ROOT / "data/m1a_cohorts/cohort_mol1_zap70.csv",
    "dpo_composite": PROJECT_ROOT / "data/dpo_composite_cohort/cohort_mol1.csv",
    "dpo_geometry": PROJECT_ROOT / "data/dpo_geometry_cohort/cohort_mol1.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Short tag, e.g. r1/r2/r3")
    ap.add_argument("--subject_csv", required=True)
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--workers-2d", type=int, default=G.WORKERS_2D)
    ap.add_argument("--workers-cov", type=int, default=G.WORKERS_COVDOCK)
    ap.add_argument("--n-cov-vina", type=int, default=G.N_COV_VINA)
    ap.add_argument("--seed", type=int, default=G.SUBSAMPLE_SEED)
    ap.add_argument("--out_dir", default=str(G.OUT_DIR))
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    G.OUT_DIR = OUT_DIR

    subj_path = Path(args.subject_csv)
    if not subj_path.exists():
        sys.exit(f"Subject CSV missing: {subj_path}")

    G.prepare_stripped_receptor()

    # Register subject
    G.COHORT_SOURCES[args.subject] = subj_path
    # Register extras present on disk
    for name, p in EXTRA_SOURCES.items():
        if p.exists() and name not in G.COHORT_SOURCES:
            G.COHORT_SOURCES[name] = p

    requested = list(dict.fromkeys([args.subject] + args.baselines))
    print(f"[info] cohorts to load/compute: {requested}")
    cohort_data = {}
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

    if args.subject not in cohort_data:
        sys.exit(f"{args.subject} cohort failed to load")
    df2d_a, dfcov_a = cohort_data[args.subject]

    pairwise = {}
    for b in args.baselines:
        if b not in cohort_data:
            continue
        df2d_b, dfcov_b = cohort_data[b]
        stats = G.pairwise_stats(args.subject, df2d_a, dfcov_a, b, df2d_b, dfcov_b)
        verdict = G.pairwise_verdict(stats, args.subject, b)
        pairwise[f"{args.subject}_vs_{b}"] = {"stats": stats, "verdict": verdict}

    n_beats = sum(1 for v in pairwise.values() if v["verdict"]["positive_axes_count"] >= 2)
    n_mixed = sum(1 for v in pairwise.values() if v["verdict"]["positive_axes_count"] == 1)
    n_none  = sum(1 for v in pairwise.values() if v["verdict"]["positive_axes_count"] == 0)
    n_total = len(pairwise)
    if n_beats >= max(2, n_total - 1):
        headline = f"{args.subject} IS significantly more geometrically pre-reactive than {n_beats}/{n_total} baselines (clean win)."
    elif n_beats >= 1:
        headline = f"{args.subject} beats {n_beats}/{n_total} baselines on >=2 axes, mixed on {n_mixed}/{n_total}, indistinct on {n_none}/{n_total}."
    else:
        headline = f"No evidence {args.subject} improves geometry over baselines (0/{n_total} clean wins; mixed={n_mixed}, indistinct={n_none})."

    out = dict(
        config=dict(
            subject=args.subject,
            subject_csv=str(subj_path),
            baselines=[b for b in args.baselines if b in cohort_data],
            alpha=0.05,
            cov_vina_subsample_n=args.n_cov_vina,
            subsample_seed=args.seed,
            acrylamide_smarts=G.ACRYLAMIDE_SMARTS,
        ),
        cohort_summaries=cohort_summaries,
        pairwise_tests=pairwise,
        headline_verdict=dict(summary=headline, clean_wins=n_beats, mixed=n_mixed, indistinct=n_none, n_baselines=n_total),
    )

    out_json = OUT_DIR / f"{args.subject}_geometry.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_json}")

    lines = []
    lines.append(f"# R-stream geometric comparison: {args.subject}\n")
    lines.append(f"**Headline**: {headline}\n")
    lines.append(f"**Subject source**: `{subj_path}`\n")
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
    lines.append(f"\n## Pairwise tests ({args.subject} vs baseline)\n")
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
    out_md = OUT_DIR / f"{args.subject}_geometry.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
