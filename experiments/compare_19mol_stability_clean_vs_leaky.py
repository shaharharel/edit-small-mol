#!/usr/bin/env python3
"""Side-by-side comparison: 30-Trial Pairwise Stability + 20-Seed Uncertainty
under LEAKY vs CLEAN (mol-disjoint val) FiLMDelta protocol.

Reads:
  - results/paper_evaluation/19_molecules_pairwise_stability.json        (leaky, published)
  - results/paper_evaluation/19_molecules_pairwise_stability_clean.json  (clean rerun)
  - results/paper_evaluation/19_molecules_20seed_uncertainty.json        (leaky, published)
  - results/paper_evaluation/19_molecules_20seed_uncertainty_clean.json  (clean rerun)

Writes:
  - results/paper_evaluation/19mol_stability_clean_vs_leaky.csv
  - results/paper_evaluation/19mol_stability_clean_vs_leaky.md  (markdown table)
  - stdout: side-by-side ranking diff + Spearman/Kendall

Usage: conda run -n quris python -u experiments/compare_19mol_stability_clean_vs_leaky.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "results" / "paper_evaluation"


def load_or_die(p):
    if not p.exists():
        sys.exit(f"MISSING: {p}")
    return json.load(open(p))


def main():
    pw_leaky = load_or_die(RD / "19_molecules_pairwise_stability.json")
    pw_clean = load_or_die(RD / "19_molecules_pairwise_stability_clean.json")
    se_leaky = load_or_die(RD / "19_molecules_20seed_uncertainty.json")
    se_clean = load_or_die(RD / "19_molecules_20seed_uncertainty_clean.json")

    print("=" * 90)
    print("30-TRIAL PAIRWISE STABILITY: LEAKY vs CLEAN")
    print("=" * 90)
    print(f"  leaky inter-trial Spearman: {pw_leaky.get('inter_trial_spearman_mean', float('nan')):.3f}")
    print(f"  clean inter-trial Spearman: {pw_clean.get('inter_trial_spearman_mean', float('nan')):.3f}")
    print(f"  clean mean val MAE (mol-disjoint): {pw_clean.get('val_mae_mean', float('nan')):.3f}")
    print()
    print(f"  {'Mol':>4s}  {'L mean_rank':>11s}  {'C mean_rank':>11s}  {'Δrank':>6s}  "
          f"{'L top3%':>8s}  {'C top3%':>8s}  "
          f"{'L pred':>8s}  {'C pred':>8s}  {'Δpred':>7s}")
    print("-" * 90)

    rows_pw = []
    leaky_ranks, clean_ranks = [], []
    leaky_preds, clean_preds = [], []
    for ml, mc in zip(pw_leaky['mol_stats'], pw_clean['mol_stats']):
        assert ml['mol_idx'] == mc['mol_idx']
        lr = ml['mean_rank']; cr = mc['mean_rank']
        lp = ml['mean_pred']; cp = mc['mean_pred']
        print(f"  {ml['mol_idx']:2d}    {lr:9.1f}    {cr:9.1f}    {cr-lr:+5.1f}  "
              f"{ml['top3_pct']:6.0f}%   {mc['top3_pct']:6.0f}%   "
              f"{lp:6.2f}    {cp:6.2f}   {cp-lp:+5.2f}")
        leaky_ranks.append(lr); clean_ranks.append(cr)
        leaky_preds.append(lp); clean_preds.append(cp)
        rows_pw.append({
            'mol': ml['mol_idx'],
            'leaky_mean_rank': lr, 'clean_mean_rank': cr, 'd_mean_rank': cr - lr,
            'leaky_std_rank': ml['std_rank'], 'clean_std_rank': mc['std_rank'],
            'leaky_top3_pct': ml['top3_pct'], 'clean_top3_pct': mc['top3_pct'],
            'leaky_pred': lp, 'clean_pred': cp, 'd_pred': cp - lp,
            'leaky_wins': ml['mean_wins'], 'clean_wins': mc['mean_wins'],
        })

    sp_pw, _ = spearmanr(leaky_ranks, clean_ranks)
    kt_pw, _ = kendalltau(leaky_ranks, clean_ranks)
    print(f"\n  Spearman(leaky_mean_rank, clean_mean_rank) = {sp_pw:.3f}")
    print(f"  Kendall  tau                                = {kt_pw:.3f}")

    print(f"\n  CONSENSUS RANK (top 5):")
    print(f"    leaky: {pw_leaky['consensus_ranking'][:5]}")
    print(f"    clean: {pw_clean['consensus_ranking'][:5]}")

    print("\n" + "=" * 90)
    print("20-SEED UNCERTAINTY: LEAKY vs CLEAN")
    print("=" * 90)
    print(f"  clean mean val MAE (mol-disjoint): {se_clean.get('val_mae_mean', float('nan')):.3f}")
    print()
    print(f"  {'Mol':>4s}  {'L mean':>7s}  {'C mean':>7s}  {'ΔMean':>7s}  "
          f"{'L std':>6s}  {'C std':>6s}  {'L rank':>7s}  {'C rank':>7s}  {'Δrank':>7s}")
    print("-" * 90)

    rows_se = []
    l_means, c_means = [], []
    l_ranks, c_ranks = [], []
    for ml, mc in zip(se_leaky['mol_results'], se_clean['mol_results']):
        assert ml['mol_idx'] == mc['mol_idx']
        lm = ml['mean_pIC50']; cm = mc['mean_pIC50']
        ls = ml['std_pIC50']; cs = mc['std_pIC50']
        lr = ml['mean_rank']; cr = mc['mean_rank']
        print(f"  {ml['mol_idx']:2d}    {lm:6.3f}   {cm:6.3f}   {cm-lm:+6.3f}  "
              f"{ls:5.3f}   {cs:5.3f}   {lr:5.1f}    {cr:5.1f}   {cr-lr:+5.1f}")
        l_means.append(lm); c_means.append(cm); l_ranks.append(lr); c_ranks.append(cr)
        rows_se.append({
            'mol': ml['mol_idx'],
            'leaky_mean_pIC50': lm, 'clean_mean_pIC50': cm, 'd_mean_pIC50': cm - lm,
            'leaky_std_pIC50': ls, 'clean_std_pIC50': cs,
            'leaky_mean_rank': lr, 'clean_mean_rank': cr, 'd_mean_rank': cr - lr,
        })

    sp_se, _ = spearmanr(l_means, c_means)
    kt_se, _ = kendalltau(l_ranks, c_ranks)
    print(f"\n  Spearman(leaky_mean_pIC50, clean_mean_pIC50) = {sp_se:.3f}")
    print(f"  Kendall  tau (ranks)                           = {kt_se:.3f}")

    # ============ Save ============
    df_pw = pd.DataFrame(rows_pw)
    df_se = pd.DataFrame(rows_se)
    out_csv = RD / "19mol_stability_clean_vs_leaky.csv"
    pd.concat([
        df_pw.assign(table='30trial_pairwise'),
        df_se.assign(table='20seed_uncertainty'),
    ]).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # Markdown
    out_md = RD / "19mol_stability_clean_vs_leaky.md"
    with open(out_md, "w") as f:
        f.write(f"# 19-mol stability: clean (mol-disjoint val) vs leaky (pair-row val)\n\n")
        f.write(f"## 30-Trial Pairwise Stability\n")
        f.write(f"- Inter-trial Spearman: leaky **{pw_leaky['inter_trial_spearman_mean']:.3f}** "
                f"vs clean **{pw_clean['inter_trial_spearman_mean']:.3f}**\n")
        f.write(f"- Mean held-out val MAE (clean, mol-disjoint): {pw_clean.get('val_mae_mean', float('nan')):.3f}\n")
        f.write(f"- Spearman(leaky_mean_rank, clean_mean_rank) = **{sp_pw:.3f}** "
                f"(Kendall τ = {kt_pw:.3f})\n\n")
        f.write(f"| Mol | L mean rank | C mean rank | Δrank | L top-3% | C top-3% | L pred pIC50 | C pred pIC50 | Δpred |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_pw:
            f.write(f"| {r['mol']} | {r['leaky_mean_rank']:.1f} | {r['clean_mean_rank']:.1f} | "
                    f"{r['d_mean_rank']:+.1f} | {r['leaky_top3_pct']:.0f}% | {r['clean_top3_pct']:.0f}% | "
                    f"{r['leaky_pred']:.2f} | {r['clean_pred']:.2f} | {r['d_pred']:+.2f} |\n")
        f.write(f"\nConsensus top-5: leaky **{pw_leaky['consensus_ranking'][:5]}** vs "
                f"clean **{pw_clean['consensus_ranking'][:5]}**\n\n")

        f.write(f"## 20-Seed Uncertainty (Master Evidence Table pIC50)\n")
        f.write(f"- Mean held-out val MAE (clean, mol-disjoint): {se_clean.get('val_mae_mean', float('nan')):.3f}\n")
        f.write(f"- Spearman(leaky_pIC50, clean_pIC50) = **{sp_se:.3f}** (Kendall τ = {kt_se:.3f})\n\n")
        f.write(f"| Mol | L mean | C mean | Δmean | L std | C std | L rank | C rank | Δrank |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_se:
            f.write(f"| {r['mol']} | {r['leaky_mean_pIC50']:.3f} | {r['clean_mean_pIC50']:.3f} | "
                    f"{r['d_mean_pIC50']:+.3f} | {r['leaky_std_pIC50']:.3f} | {r['clean_std_pIC50']:.3f} | "
                    f"{r['leaky_mean_rank']:.1f} | {r['clean_mean_rank']:.1f} | {r['d_mean_rank']:+.1f} |\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
