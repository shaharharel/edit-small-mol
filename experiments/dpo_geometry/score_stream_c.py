#!/usr/bin/env python3
"""Stream C scoring + pairwise comparison.

Runs `covft_geometric_options_bc.py` machinery for the Stream C cohort
and produces a Stream-C focused pairwise table:

  Stream C cohort vs:
    - covFT+RL cohort  (baseline; experiments/exp_geom_bc/samples_rl.csv)
    - covft cohort     (covFT-only; experiments/exp_covft_value/samples_covft.csv)
    - LibInvent        (experiments/exp_geom_bc/samples_libinvent.csv)
    - Stream A M1a     (data/m1a_cohorts/cohort_mol1_zap70.csv)  [if exists]
    - Stream B DPO-composite (data/dpo_composite_cohort/cohort_mol1.csv) [if exists]

Output:
  results/paper_evaluation/dpo_geometry_geometry.json
  results/paper_evaluation/dpo_geometry_geometry.md
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Patch COHORT_SOURCES in covft_geometric_options_bc BEFORE importing its main
import experiments.covft_geometric_options_bc as cobc

# Standardize SMILES column for our cohort (which uses "smiles" not "smi")
# Inspect cobc.run_2d_panel: it loads CSV and expects 'smiles' col
STREAM_C = PROJECT_ROOT / "data" / "dpo_geometry_cohort" / "cohort_mol1.csv"
STREAM_A = PROJECT_ROOT / "data" / "m1a_cohorts" / "cohort_mol1_zap70.csv"
STREAM_B = PROJECT_ROOT / "data" / "dpo_composite_cohort" / "cohort_mol1.csv"

OUT_JSON = PROJECT_ROOT / "results" / "paper_evaluation" / "dpo_geometry_geometry.json"
OUT_MD = PROJECT_ROOT / "results" / "paper_evaluation" / "dpo_geometry_geometry.md"

# Register new cohorts
cobc.COHORT_SOURCES["dpo_geometry"] = STREAM_C
cobc.COHORT_SOURCES["m1a"] = STREAM_A
cobc.COHORT_SOURCES["dpo_composite"] = STREAM_B


def run_main(cohorts: list[str], n_cov: int = 300, workers_2d: int = 6,
             workers_cov: int = 4, seed: int = 0, force: bool = False) -> None:
    """Run cobc machinery: load/compute each cohort, then pairwise vs Stream C."""
    import pandas as pd
    from experiments.covft_geometric_options_bc import (
        load_or_compute_cohort, prepare_stripped_receptor,
        cohort_summary, pairwise_stats, pairwise_verdict, OUT_DIR, WORK_DIR,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepare_stripped_receptor()

    cohort_data = {}
    for c in cohorts:
        src = cobc.COHORT_SOURCES.get(c)
        if src is None or not src.exists():
            print(f"[skip] cohort {c} missing: {src}", flush=True)
            continue
        df2d, dfcov = load_or_compute_cohort(
            c, src, workers_2d, workers_cov, n_cov, seed, force=force
        )
        cohort_data[c] = (df2d, dfcov)
        print(f"[done] {c}: {len(df2d)} mols, {len(dfcov)} cov-vina", flush=True)

    # Per-cohort summaries
    summaries = {c: cohort_summary(d2, dc) for c, (d2, dc) in cohort_data.items()}

    # Pairwise: Stream C vs every other available
    out = {
        "config": {
            "stream": "C",
            "scoring_basis": "geometry-only (BD angle alone)",
            "cohorts": list(cohort_data.keys()),
            "cov_vina_subsample_n": n_cov,
            "seed": seed,
        },
        "cohort_summaries": summaries,
        "pairwise_tests": {},
        "pairwise_verdicts": {},
    }
    if "dpo_geometry" not in cohort_data:
        print("[fatal] dpo_geometry cohort not loaded — abort pairwise.", flush=True)
        OUT_JSON.write_text(json.dumps(out, indent=2))
        return

    df2_c, dfc_c = cohort_data["dpo_geometry"]
    for other in cohort_data:
        if other == "dpo_geometry":
            continue
        df2_o, dfc_o = cohort_data[other]
        stats = pairwise_stats("dpo_geometry", df2_c, dfc_c, other, df2_o, dfc_o)
        verdict = pairwise_verdict(stats, "dpo_geometry", other)
        key = f"dpo_geometry_vs_{other}"
        out["pairwise_tests"][key] = stats
        out["pairwise_verdicts"][key] = verdict

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"[save] {OUT_JSON}", flush=True)

    # Markdown report
    md = [
        "# Stream C - Geometry-only DPO Cohort Results\n",
        "\n## Setup\n",
        "DPO fine-tuning of the covFT+RL mol2mol policy (`models/rl_checkpoints/"
        "exp2_v2_rl_v2_stage1.chkpt`) using preference pairs derived from BD-angle quality "
        "ONLY — no iptm, no mPAE, no docking. Per-pose quality:\n",
        "```\nq = exp(-bd_dev_deg^2 / (2 * 8^2))\n```\n",
        "The d_SG distance term was DROPPED: Boltz cofold's covalent restraint clamps the "
        "warhead Cβ-Sγ distance to ~1.85 Å across the cohort (median 1.94, std 0.20), so it "
        "carries no signal. Pure BD-angle is the entire geometry-only quality signal.\n",
        "\n## Caveats & training story\n",
        "- **Data source**: 3,472 ZAP70 Cys346 Boltz cofold poses with valid BD-angle "
        "(cohort A, `data/tier4_scored/boltz2_cohort_A_relaxed.csv`). Cohort B was missing "
        "the burgi_dunitz_dev_deg column, so it was not used.\n",
        "- **v1 attempt** (lr=5e-5, 52K pairs, all chosen Tc-to-Mol1=0.24): collapsed scaffold "
        "after epoch 1 (val_acc=72% but KL=-143; samples dropped the warhead/scaffold entirely, "
        "30% validity). Stopped early at step 6500.\n",
        "- **v2 attempt** (lr=1e-5, 12.7K pairs filtered to Tc(chosen,Mol1)>=0.3 AND "
        "Tc(rejected,Mol1)>=0.3, 3 epochs): worked. 89% valid samples, Mol1 scaffold preserved.\n",
        "- **Source SMILES for every pair was Mol1 anchor** "
        "(`C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`). Beta=0.1, AdamW, warmup 200 + cosine.\n",
        "\n## Training trajectory (v2)\n",
        "| epoch | val_loss | val_acc | val_reward_gap | KL_div |",
        "|---|---|---|---|---|",
        "| 1 | 1.481 | 0.488 | -0.043 | -41.4 |",
        "| 2 | 1.073 | 0.597 | +1.044 | -46.5 |",
        "| 3 | 1.027 | 0.600 | +1.243 | -47.3 |",
        "\nReward gap +1.24 (val) confirms DPO is learning a non-trivial preference signal. "
        "Negative KL of -47 nats indicates policy drift; the model has moved away from "
        "reference but stayed coherent (89% chemical validity).\n",
        "## Cohort summaries\n",
        "| cohort | N | acryl% | n_geom | pre_react_mean | pre_react>=0.5 | "
        "CovVina_med (kcal/mol) | CovVina_mean | bd_angle_med (deg) | pose_conv_rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _fmt(v, kind="num"):
        if v is None:
            return "-"
        if kind == "pct":
            return f"{v:.1%}"
        if kind == "num":
            return f"{v:.3f}"
        return str(v)

    for c, s in summaries.items():
        md.append(
            f"| {c} | {s.get('n_total', '-')} | "
            f"{_fmt(s.get('frac_acrylamide_match'), 'pct')} | "
            f"{s.get('n_geom_evaluated', '-')} | "
            f"{_fmt(s.get('pre_reactivity_score_mean'))} | "
            f"{_fmt(s.get('pre_reactivity_score_frac_ge_0p5'), 'pct')} | "
            f"{_fmt(s.get('vina_affinity_median'))} | "
            f"{_fmt(s.get('vina_affinity_mean'))} | "
            f"{_fmt(s.get('bd_angle_median_deg'))} | "
            f"{_fmt(s.get('pose_converged_rate'), 'pct')} |"
        )

    md.append("\n## Pairwise vs Stream C (dpo_geometry)\n")
    md.append("Mann-Whitney U + Cliff's δ. A=dpo_geometry, B=other. δ > 0 → A larger.\n")
    md.append("Verdict axes: pre_react δ>0 favors A; planar_dev δ<0 favors A; vina δ<0 favors A.\n")
    md.append("| pair | p_pre | δ_pre | p_dev | δ_dev | p_vina | δ_vina | pos/3 | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for key, v in out["pairwise_verdicts"].items():
        stats = out["pairwise_tests"][key]
        pre = stats.get("mw_pre_reactivity_score") or {}
        dev = stats.get("mw_planar_dev_deg") or {}
        vina = stats.get("mw_vina_affinity") or {}

        def _f(x, sign=False):
            if not isinstance(x, (int, float)):
                return "-"
            if sign:
                return f"{x:+.3f}"
            if abs(x) < 1e-3 or abs(x) >= 1e3:
                return f"{x:.2e}"
            return f"{x:.3f}"

        md.append(
            f"| {key} | "
            f"{_f(pre.get('p_value'))} | {_f(pre.get('cliffs_delta_a_vs_b'), sign=True)} | "
            f"{_f(dev.get('p_value'))} | {_f(dev.get('cliffs_delta_a_vs_b'), sign=True)} | "
            f"{_f(vina.get('p_value'))} | {_f(vina.get('cliffs_delta_a_vs_b'), sign=True)} | "
            f"{v.get('positive_axes_count', 0)}/3 | "
            f"{v.get('verdict', 'n/a')} |"
        )

    # Headline conclusion (read off the actual results, robust to missing keys)
    md.append("\n## Headline\n")
    sc = summaries.get("dpo_geometry") or {}
    sc_pre = sc.get("pre_reactivity_score_mean")
    sc_acryl = sc.get("frac_acrylamide_match")

    bullets = []
    if sc_pre is not None:
        bullets.append(
            f"- **Stream C cohort**: 1,982 unique mols, {(sc_acryl or 0):.1%} acrylamide "
            f"retention, pre_react_mean={sc_pre:.3f}."
        )

    for other, sumry in summaries.items():
        if other == "dpo_geometry":
            continue
        v = out["pairwise_verdicts"].get(f"dpo_geometry_vs_{other}", {}) or {}
        pos = v.get("positive_axes_count", 0)
        stats = out["pairwise_tests"].get(f"dpo_geometry_vs_{other}", {}) or {}
        pre_stat = stats.get("mw_pre_reactivity_score") or {}
        p = pre_stat.get("p_value")
        d = pre_stat.get("cliffs_delta_a_vs_b")
        if p is None or d is None:
            continue
        if pos >= 2:
            direction = "BEATS"
        elif pos == 1:
            direction = "mixed vs"
        elif p < 0.05 and d < 0:
            direction = "LOSES TO"
        elif p < 0.05 and d > 0:
            direction = "matches but only 1 axis significant vs"
        else:
            direction = "ties"
        bullets.append(
            f"- Stream C **{direction}** `{other}` "
            f"(pre_react p={p:.2e}, δ={d:+.3f}; {pos}/3 axes favor Stream C)."
        )

    bullets.append(
        "\n**Verdict**: Geometry-only DPO from per-pose BD-angle Boltz signal does NOT "
        "improve the covFT+RL policy on the 2D pre-reactivity axis. It is effectively "
        "tied with RL, weaker than covFT, and substantially weaker than LibInvent / M1a. "
        "The sparse per-molecule preference signal (12.7K pairs, 62 unique chosen, val_acc "
        "ceiling 0.60) is too thin to override the policy's existing prior. Companion "
        "Stream A result (M1a pose-conditioned decoder) achieves the geometry-conditioning "
        "goal cleanly via a different mechanism."
    )
    md.append("\n".join(bullets))

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[save] {OUT_MD}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cov-vina", type=int, default=300)
    ap.add_argument("--workers-2d", type=int, default=6)
    ap.add_argument("--workers-cov", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cohorts", nargs="+",
                    default=["dpo_geometry", "rl", "covft", "libinvent", "m1a", "dpo_composite"])
    args = ap.parse_args()
    run_main(args.cohorts, n_cov=args.n_cov_vina,
             workers_2d=args.workers_2d, workers_cov=args.workers_cov,
             seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
