#!/usr/bin/env python3
"""Dedup covalent docking results by canonical SMILES per cohort, re-aggregate.

This corrects the major confound discovered after the full 2712-mol AD-CovDock
pass: Lingo cohorts have severe mode collapse (H1=4 unique, H3=36, H2=66,
L0=1), so the "Lingo cohort wins" finding from the means was dominated by
replicate counting, not by genuinely better unique mols.

Outputs:
  - covalent_docking_results_unique.csv  (1 row per (cohort, canonical_smiles))
  - covalent_docking_per_cohort_unique.csv
  - covalent_docking_stats_unique.csv (pairwise MW + Cliff's δ)
  - covalent_docking_groupAB_unique.csv (sequence vs Lingo group stats)
  - figures_covalent/box_*_unique.png  (re-plotted on unique sets)
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from scipy import stats
from statsmodels.stats.multitest import multipletests

RDLogger.DisableLog('rdApp.*')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COHORT_ORDER = [
    "L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2",
    "DeNovo_warhead_gate", "Mol2Mol_warhead_gate", "LibInvent_locked", "Amine_Replacements",
]
LINGO_COHORTS = {"L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2"}
SEQ_COHORTS = {"DeNovo_warhead_gate", "Mol2Mol_warhead_gate", "LibInvent_locked", "Amine_Replacements"}

# Sequence-only baselines for the formal Group A vs Group B test
# (LibInvent_locked now has warhead post-2026-06-01 source-CSV fix, but still
# kept as hybrid group C per the unique-stats script convention.)
GROUP_A_SEQ = {"DeNovo_warhead_gate", "Mol2Mol_warhead_gate"}
GROUP_B_LINGO = {"L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2"}


def canon_smi(s):
    if pd.isna(s):
        return None
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def per_cohort_unique(df_unique: pd.DataFrame, cohorts: list) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        sub = df_unique[df_unique['cohort'] == c]
        if len(sub) == 0:
            continue
        rows.append({
            "cohort": c,
            "N_unique": len(sub),
            "AD_CovDock_score_median": float(np.nanmedian(sub['AD_CovDock_score'])),
            "AD_CovDock_score_mean": float(np.nanmean(sub['AD_CovDock_score'])),
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
        })
    return pd.DataFrame(rows)


def pairwise_stats_unique(df_unique: pd.DataFrame, metric: str, cohorts: list) -> pd.DataFrame:
    rows = []
    for c1, c2 in itertools.combinations(cohorts, 2):
        x = df_unique.loc[df_unique['cohort'] == c1, metric].dropna().values
        y = df_unique.loc[df_unique['cohort'] == c2, metric].dropna().values
        if len(x) < 3 or len(y) < 3:
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
    valid = s["p"].notna()
    if valid.any():
        _, p_bh, _, _ = multipletests(s.loc[valid, "p"].values, method="fdr_bh")
        s.loc[valid, "p_bh"] = p_bh
    else:
        s["p_bh"] = np.nan
    return s


def group_AB_test(df_unique: pd.DataFrame, metric: str) -> dict:
    """Mann-Whitney + Cliff's δ for Group A (sequence) vs Group B (Lingo)
    on unique mols per cohort. Pools across all cohorts within each group."""
    a = df_unique[df_unique['cohort'].isin(GROUP_A_SEQ)][metric].dropna().values
    b = df_unique[df_unique['cohort'].isin(GROUP_B_LINGO)][metric].dropna().values
    if len(a) < 5 or len(b) < 5:
        return {"metric": metric, "n_a": len(a), "n_b": len(b),
                "median_a": float(np.median(a)) if len(a) else None,
                "median_b": float(np.median(b)) if len(b) else None,
                "U": None, "p": None, "cliffs_d": None}
    try:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        U, p = None, None
    return {
        "metric": metric, "n_a": len(a), "n_b": len(b),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "U": float(U) if U is not None else None,
        "p": float(p) if p is not None else None,
        "cliffs_d": float(cliffs_delta(a, b)),
    }


def boxplot(df: pd.DataFrame, metric: str, out: Path, cohorts: list, title: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [df.loc[df['cohort'] == c, metric].dropna().values for c in cohorts]
    n = [len(d) for d in data]
    labels = [f"{c}\n(N={ni})" for c, ni in zip(cohorts, n)]
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        col = "#4682B4" if cohorts[i] in LINGO_COHORTS else "#CD5C5C"
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title(title or metric)
    ax.set_ylabel(metric)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison/covalent_docking_results.csv"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison"))
    ap.add_argument("--keep-libinvent", action="store_true",
                    help="Include LibInvent_locked (default: skip — source-path bug, no warhead)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures_covalent"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows (full set)")

    # Canonical SMILES
    print("Canonicalizing SMILES ...")
    df['canon'] = df['smi'].apply(canon_smi)
    n_bad = df['canon'].isna().sum()
    if n_bad:
        print(f"  {n_bad} rows with un-parseable SMILES dropped")
        df = df[df['canon'].notna()].copy()

    # Per-cohort dedup count
    print("\n--- Mode collapse per cohort ---")
    collapse_rows = []
    for c, sub in df.groupby('cohort'):
        n_unique = sub['canon'].nunique()
        n_total = len(sub)
        collapse_rows.append({
            "cohort": c, "N_total": n_total, "N_unique": n_unique,
            "pct_unique": 100 * n_unique / n_total,
        })
        print(f"  {c:30s} {n_total:>4d} total, {n_unique:>4d} unique ({100*n_unique/n_total:.1f}%)")
    pd.DataFrame(collapse_rows).to_csv(out_dir / "covalent_docking_mode_collapse.csv", index=False)

    # Skip LibInvent_locked unless flagged
    if not args.keep_libinvent:
        df = df[df['cohort'] != 'LibInvent_locked'].copy()
        print(f"\nSkipping LibInvent_locked (no warhead, source-path bug) -> {len(df)} rows")

    # Dedup: keep FIRST row per (cohort, canon) — AD-CovDock score depends only on
    # the tethered 3D geometry which differs per replicate (different ETKDG seeds),
    # so we take the median across replicates within each unique canonical SMILES
    print("\nDeduplicating: median across replicates per (cohort, canonical_smiles) ...")
    agg = {
        "AD_CovDock_score": "median",
        "AD_CovDock_d_sg": "median",
        "AD_CovDock_bd_angle": "median",
        "AD_CovDock_pose_valid": "max",  # at least one replicate valid?
        "Restrained_Vina_score": "median",
        "Restrained_Vina_d_sg": "median",
        "Restrained_Vina_bd_angle": "median",
        "Restrained_Vina_pose_valid": "max",
        "Restrained_Vina_any_pose_valid": "max",
        "NonCovVina_score": "median",
        "NonCovVina_d_sg_OLD": "median",
        "smi": "first",   # representative SMILES
    }
    df_unique = df.groupby(['cohort', 'canon']).agg(agg).reset_index()
    df_unique.to_csv(out_dir / "covalent_docking_results_unique.csv", index=False)
    print(f"  unique rows: {len(df_unique)} (median across replicates)")

    cohorts_present = [c for c in COHORT_ORDER if c in df_unique['cohort'].unique()]

    # Per-cohort summary
    summ = per_cohort_unique(df_unique, cohorts_present)
    summ.to_csv(out_dir / "covalent_docking_per_cohort_unique.csv", index=False)
    print(f"\n--- Per-cohort summary on UNIQUE mols ---")
    print(summ[['cohort', 'N_unique', 'AD_CovDock_score_median', 'AD_CovDock_d_sg_median',
                'AD_CovDock_bd_median', 'Restrained_score_median', 'NonCovVina_score_median']].to_string(index=False))

    # Pairwise stats
    print("\nComputing pairwise Mann-Whitney + Cliff's δ on UNIQUE sets...")
    metrics = [
        "AD_CovDock_score", "AD_CovDock_d_sg", "AD_CovDock_bd_angle",
        "Restrained_Vina_score", "Restrained_Vina_d_sg", "Restrained_Vina_bd_angle",
        "NonCovVina_score",
    ]
    all_stats = [pairwise_stats_unique(df_unique, m, cohorts_present) for m in metrics]
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(out_dir / "covalent_docking_stats_unique.csv", index=False)
    print(f"wrote covalent_docking_stats_unique.csv  rows={len(stats_df)}")

    # Group A (seq) vs Group B (Lingo)
    print("\n--- Group A (DeNovo+Mol2Mol) vs Group B (Lingo cohorts) on UNIQUE mols ---")
    group_rows = []
    for m in metrics:
        r = group_AB_test(df_unique, m)
        group_rows.append(r)
        if r['p'] is not None:
            sig = "***" if r['p'] < 0.001 else ("**" if r['p'] < 0.01 else ("*" if r['p'] < 0.05 else "ns"))
            print(f"  {m:30s} N_seq={r['n_a']:>4d} med={r['median_a']:>7.2f} | "
                  f"N_lingo={r['n_b']:>4d} med={r['median_b']:>7.2f} | "
                  f"δ={r['cliffs_d']:+.3f}  p={r['p']:.2e} {sig}")
    pd.DataFrame(group_rows).to_csv(out_dir / "covalent_docking_groupAB_unique.csv", index=False)

    # Plots on unique set
    print("\nPlotting on UNIQUE set ...")
    boxplot(df_unique, "AD_CovDock_score", fig_dir / "box_AD_CovDock_score_unique.png",
            cohorts_present, title="AD-CovDock score (UNIQUE mols only) — lower = better covalent fit")
    boxplot(df_unique, "AD_CovDock_d_sg", fig_dir / "box_AD_CovDock_d_sg_unique.png",
            cohorts_present, title="AD-CovDock d(Cβ-SG) [Å] (UNIQUE) — target 1.85 Å")
    boxplot(df_unique, "AD_CovDock_bd_angle", fig_dir / "box_AD_CovDock_bd_angle_unique.png",
            cohorts_present, title="AD-CovDock Bürgi-Dunitz angle [°] (UNIQUE)")
    boxplot(df_unique, "Restrained_Vina_d_sg", fig_dir / "box_Restrained_Vina_d_sg_unique.png",
            cohorts_present, title="Restrained Vina d(Cβ-SG) [Å] (UNIQUE) — corrected SMARTS")
    boxplot(df_unique, "NonCovVina_score", fig_dir / "box_NonCovVina_score_unique.png",
            cohorts_present, title="Non-covalent Vina score (UNIQUE) — for reference")
    print(f"wrote {fig_dir}/box_*_unique.png")

    # Rank agreement on unique
    a = summ.set_index('cohort')['AD_CovDock_score_median']
    b = summ.set_index('cohort')['Restrained_score_median']
    c = summ.set_index('cohort')['NonCovVina_score_median']
    common = list(a.dropna().index.intersection(b.dropna().index).intersection(c.dropna().index))
    s_ab = stats.spearmanr(a.loc[common], b.loc[common])
    s_ac = stats.spearmanr(a.loc[common], c.loc[common])
    s_bc = stats.spearmanr(b.loc[common], c.loc[common])
    rank = {
        "ad_cov_vs_restrained_spearman_unique": float(s_ab.correlation),
        "ad_cov_vs_noncov_spearman_unique": float(s_ac.correlation),
        "restrained_vs_noncov_spearman_unique": float(s_bc.correlation),
        "cohorts_compared": common,
    }
    json.dump(rank, open(out_dir / "covalent_docking_rank_agreement_unique.json", "w"), indent=2)
    print(f"\nRank agreement (cohort medians, UNIQUE):")
    print(f"  AD-CovDock vs Restrained: ρ={s_ab.correlation:+.3f}")
    print(f"  AD-CovDock vs NonCovVina: ρ={s_ac.correlation:+.3f}")
    print(f"  Restrained vs NonCovVina: ρ={s_bc.correlation:+.3f}")


if __name__ == "__main__":
    main()
