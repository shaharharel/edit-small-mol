#!/usr/bin/env python3
"""Statistical comparison + plots for covalent_docking_results.csv.

Outputs:
  - covalent_docking_stats.csv   (pairwise Mann-Whitney + Cliff's δ)
  - covalent_docking_per_cohort.csv (already by run_covalent_docking.py)
  - figures/box_AD_CovDock_score.png
  - figures/box_AD_CovDock_d_sg.png
  - figures/box_Restrained_Vina_d_sg.png
  - figures/box_NonCovVina_d_sg_OLD_vs_NEW.png  (broken vs fixed metric)
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cohort ordering (Lingo first, then sequence-only baselines)
COHORT_ORDER = [
    "L0_vanilla", "L_locked", "H1", "H2", "H3", "C5", "L1_FT_H2",  # Lingo
    "DeNovo_warhead_gate", "Mol2Mol_warhead_gate", "LibInvent_locked", "Amine_Replacements",  # seq
]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ: P(x>y) - P(x<y). Range [-1, 1]. Vector of x vs vector of y."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x = np.asarray(x); y = np.asarray(y)
    # Pairwise sign matrix
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    n = len(x) * len(y)
    return (gt - lt) / n


def pairwise_stats(df: pd.DataFrame, metric: str, cohorts: list) -> pd.DataFrame:
    """Mann-Whitney U + Cliff's δ for every pair of cohorts on the given metric."""
    rows = []
    for c1, c2 in itertools.combinations(cohorts, 2):
        x = df.loc[df['cohort'] == c1, metric].dropna().values
        y = df.loc[df['cohort'] == c2, metric].dropna().values
        if len(x) < 5 or len(y) < 5:
            rows.append({
                "metric": metric, "cohort_a": c1, "cohort_b": c2,
                "n_a": len(x), "n_b": len(y),
                "median_a": float(np.median(x)) if len(x) else None,
                "median_b": float(np.median(y)) if len(y) else None,
                "U": None, "p": None, "cliffs_d": None,
            })
            continue
        try:
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            U, p = None, None
        d = cliffs_delta(x, y)
        rows.append({
            "metric": metric, "cohort_a": c1, "cohort_b": c2,
            "n_a": len(x), "n_b": len(y),
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "U": float(U) if U is not None else None,
            "p": float(p) if p is not None else None,
            "cliffs_d": float(d),
        })
    s = pd.DataFrame(rows)
    # BH-FDR within this metric
    valid = s["p"].notna()
    if valid.any():
        _, p_bh, _, _ = multipletests(s.loc[valid, "p"].values, method="fdr_bh")
        s.loc[valid, "p_bh"] = p_bh
    else:
        s["p_bh"] = np.nan
    return s


def boxplot(df: pd.DataFrame, metric: str, out: Path, cohorts: list, title: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [df.loc[df['cohort'] == c, metric].dropna().values for c in cohorts]
    n = [len(d) for d in data]
    labels = [f"{c}\n(N={ni})" for c, ni in zip(cohorts, n)]
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        # Lingo cohorts are first 7, sequence baselines are last 4
        col = "#4682B4" if cohorts[i] in {
            "L0_vanilla", "L_locked", "H1", "H2", "H3", "C5", "L1_FT_H2"} else "#CD5C5C"
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title(title or metric)
    ax.set_ylabel(metric)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def cohort_summary(df: pd.DataFrame, cohorts: list) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        sub = df[df['cohort'] == c]
        if len(sub) == 0:
            continue
        rows.append({
            "cohort": c,
            "N": len(sub),
            "AD_CovDock_score_median": float(np.nanmedian(sub['AD_CovDock_score'])),
            "AD_CovDock_score_iqr": (
                float(np.nanpercentile(sub['AD_CovDock_score'], 75) - np.nanpercentile(sub['AD_CovDock_score'], 25))
                if sub['AD_CovDock_score'].notna().any() else None
            ),
            "AD_CovDock_d_sg_median": float(np.nanmedian(sub['AD_CovDock_d_sg'])),
            "AD_CovDock_bd_median": float(np.nanmedian(sub['AD_CovDock_bd_angle'])),
            "AD_CovDock_pose_valid_pct": float(np.nanmean(sub['AD_CovDock_pose_valid'])) * 100,
            "Restrained_score_median": float(np.nanmedian(sub['Restrained_Vina_score'])),
            "Restrained_d_sg_median": float(np.nanmedian(sub['Restrained_Vina_d_sg'])),
            "Restrained_bd_median": float(np.nanmedian(sub['Restrained_Vina_bd_angle'])),
            "Restrained_pose_valid_pct": float(np.nanmean(sub['Restrained_Vina_pose_valid'])) * 100,
            "Restrained_any_pose_valid_pct": float(np.nanmean(sub['Restrained_Vina_any_pose_valid'])) * 100,
            "NonCovVina_score_median": float(np.nanmedian(sub['NonCovVina_score'])),
            "NonCovVina_d_sg_OLD_median": float(np.nanmedian(sub['NonCovVina_d_sg_OLD'])),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison/covalent_docking_results.csv"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures_covalent"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cohorts_present = [c for c in COHORT_ORDER if c in df['cohort'].unique()]
    print(f"Loaded {len(df)} rows; cohorts present: {cohorts_present}")

    # Per-cohort summary
    summ = cohort_summary(df, cohorts_present)
    summ.to_csv(out_dir / "covalent_docking_per_cohort.csv", index=False)
    print(f"\nPer-cohort summary (covalent_docking_per_cohort.csv):")
    print(summ.to_string(index=False))

    # Pairwise stats on AD-CovDock score
    print("\nComputing pairwise Mann-Whitney + Cliff's δ ...")
    metrics = [
        "AD_CovDock_score", "AD_CovDock_d_sg", "AD_CovDock_bd_angle",
        "Restrained_Vina_score", "Restrained_Vina_d_sg", "Restrained_Vina_bd_angle",
    ]
    all_stats = []
    for m in metrics:
        all_stats.append(pairwise_stats(df, m, cohorts_present))
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(out_dir / "covalent_docking_stats.csv", index=False)
    print(f"wrote {out_dir/'covalent_docking_stats.csv'}  rows={len(stats_df)}")

    # Plots
    print("\nPlots ...")
    boxplot(df, "AD_CovDock_score", fig_dir / "box_AD_CovDock_score.png", cohorts_present,
            title="AD-CovDock score (tethered, score_only) — lower = more covalent-compatible")
    boxplot(df, "AD_CovDock_d_sg", fig_dir / "box_AD_CovDock_d_sg.png", cohorts_present,
            title="AD-CovDock d(Cβ-SG) — target 1.85 Å")
    boxplot(df, "AD_CovDock_bd_angle", fig_dir / "box_AD_CovDock_bd_angle.png", cohorts_present,
            title="AD-CovDock Bürgi–Dunitz angle — target 107°")
    boxplot(df, "Restrained_Vina_d_sg", fig_dir / "box_Restrained_Vina_d_sg.png", cohorts_present,
            title="Restrained Vina d(Cβ-SG) — re-measured with corrected SMARTS atom-index rule")
    boxplot(df, "Restrained_Vina_bd_angle", fig_dir / "box_Restrained_Vina_bd_angle.png", cohorts_present,
            title="Restrained Vina BD angle — re-measured with corrected SMARTS")
    boxplot(df, "NonCovVina_score", fig_dir / "box_NonCovVina_score.png", cohorts_present,
            title="Non-covalent Vina score (eval pipeline; for comparison)")
    print(f"wrote {fig_dir}/*.png")

    # Cohort rank agreement
    print("\nRank-agreement: AD-CovDock vs Restrained-Vina vs NonCovVina")
    a = summ.set_index('cohort')['AD_CovDock_score_median']
    b = summ.set_index('cohort')['Restrained_Vina_score_median'] if 'Restrained_score_median' in summ.columns else summ.set_index('cohort')['Restrained_score_median']
    c = summ.set_index('cohort')['NonCovVina_score_median']
    common = list(a.index)
    s_ab = stats.spearmanr(a.loc[common], b.loc[common])
    s_ac = stats.spearmanr(a.loc[common], c.loc[common])
    s_bc = stats.spearmanr(b.loc[common], c.loc[common])
    print(f"  AD-CovDock vs Restrained-Vina: Spearman = {s_ab.correlation:.3f} (p={s_ab.pvalue:.3f})")
    print(f"  AD-CovDock vs NonCovVina:      Spearman = {s_ac.correlation:.3f} (p={s_ac.pvalue:.3f})")
    print(f"  Restrained-Vina vs NonCovVina: Spearman = {s_bc.correlation:.3f} (p={s_bc.pvalue:.3f})")

    json.dump({
        "ad_cov_vs_restrained_spearman": float(s_ab.correlation),
        "ad_cov_vs_noncov_spearman": float(s_ac.correlation),
        "restrained_vs_noncov_spearman": float(s_bc.correlation),
    }, open(out_dir / "covalent_docking_rank_agreement.json", "w"), indent=2)


if __name__ == "__main__":
    main()
