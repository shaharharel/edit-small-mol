#!/usr/bin/env python3
"""Aggregate per-cohort per_mol.csv files, compute statistics (Mann-Whitney U,
Cliff's delta, BH-FDR) and produce CSV outputs + matplotlib figures.

Inputs:
  results/paper_evaluation/cohort_comparison/per_cohort/<cohort>/per_mol.csv

Outputs:
  results/paper_evaluation/cohort_comparison/
    all_cohorts_metrics.csv         (one row per molecule)
    per_cohort_summary.csv          (mean / std / N per cohort × metric)
    comparison_pvalues.csv          (BH-corrected Mann-Whitney U p-values)
    comparison_effect_sizes.csv     (Cliff's delta)
    figures/                        box/violin plots + p-value heatmap

Optionally include results from a remote results dir (e.g. fetched from
ai-chem2 by the caller).
"""
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison"
PER_COHORT_DIR = RESULTS_DIR / "per_cohort"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Metrics (key, label, direction (lower=better -> -1, higher=better -> +1))
METRICS = [
    # Post-dock (Vina) metrics
    ("vina_score", "Vina score (kcal/mol)", -1),
    ("ligand_efficiency", "Ligand efficiency", -1),
    ("hinge_hbond_top1", "Hinge H-bond present (<5A)", +1),
    ("d_hinge_top1", "d(polar->Met414 hinge) post-dock (A)", -1),
    ("d_cb_sg_top1", "d(Cb-Cys346 SG) post-dock (A)", "abs_185"),
    ("clash_count_top1", "Clash count (heavy-atom < 2.0 A)", -1),
    ("warhead_largest_frag", "Warhead on largest frag", +1),
    ("bd_angle_top1", "BD angle post-dock (deg)", "abs_107"),
    ("any_pose_feasible", "BD + d_SG feasible (any pose)", +1),
    ("d_cb_sg_diff_from_185", "|d(Cb-SG)_post - 1.85 A|", -1),
    ("bd_angle_diff_from_107", "|BD_post - 107|", -1),
    # Pre-dock (input pose) metrics — primary covalent-awareness claim
    ("d_cb_sg_input", "d(Cb-SG) input pose (A)", "abs_185"),
    ("bd_angle_input", "BD angle input pose (deg)", "abs_107"),
    ("d_cb_sg_input_diff_from_185", "|d(Cb-SG)_input - 1.85|", -1),
    ("bd_angle_input_diff_from_107", "|BD_input - 107|", -1),
]

# Cohort group labels — for cosmetic ordering / coloring
GROUP_MAP = {
    "L0_vanilla": "lingo_pcov",
    "L_locked": "lingo_pcov",
    "H1": "lingo_pcov",
    "H2": "lingo_pcov",
    "H3": "lingo_pcov",
    "C1": "lingo_pcov",
    "C5": "lingo_pcov",
    "L1_FT_H2": "lingo_pcov",
    "DeNovo_warhead_gate": "seqonly",
    "Mol2Mol_warhead_gate": "seqonly",
    "LibInvent_locked": "seqonly",
    "Amine_Replacements": "seqonly",
}

COHORT_ORDER = [
    "L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2",
    "DeNovo_warhead_gate", "Mol2Mol_warhead_gate", "LibInvent_locked",
    "Amine_Replacements",
]


def load_all_cohorts(extra_dirs=()):
    rows = []
    seen = set()
    dirs = [PER_COHORT_DIR, *[Path(p) for p in extra_dirs]]
    for d in dirs:
        if not d.exists():
            continue
        for cohort_dir in sorted(d.iterdir()):
            if not cohort_dir.is_dir():
                continue
            csv = cohort_dir / "per_mol.csv"
            if not csv.exists():
                continue
            name = cohort_dir.name
            if name in seen:
                continue
            df = pd.read_csv(csv)
            df["cohort"] = name
            df["group"] = GROUP_MAP.get(name, "unknown")
            rows.append(df)
            seen.add(name)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def cliffs_delta(x, y):
    """Cliff's delta — fraction of (x[i] > y[j]) minus fraction of (x[i] < y[j])."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), 0, 0
    # Faster via rank-based formula: U = sum_i sum_j sign(x_i - y_j); delta = U / (nx*ny)
    # Use a stable approach with broadcasting
    if len(x) * len(y) > 1_000_000:
        # subsample to keep this tractable
        rng = np.random.default_rng(0)
        if len(x) > 1000:
            x = rng.choice(x, 1000, replace=False)
        if len(y) > 1000:
            y = rng.choice(y, 1000, replace=False)
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.mean()), len(x), len(y)


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. pvals: array of p-values (NaN-safe).
    Returns array of BH-corrected p-values, same shape, NaN preserved.
    """
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    mask = ~np.isnan(flat)
    valid = flat[mask]
    n = len(valid)
    if n == 0:
        return p
    order = np.argsort(valid)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = valid * n / ranks
    # Enforce monotonicity (working from largest rank down)
    adj_sorted = adj[order]
    for i in range(len(adj_sorted) - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj_corrected = np.empty(n)
    adj_corrected[order] = np.minimum(adj_sorted, 1.0)
    out = np.full_like(flat, np.nan, dtype=float)
    out[mask] = adj_corrected
    return out.reshape(p.shape)


def main():
    extra_dirs = []
    if len(sys.argv) > 1:
        extra_dirs = sys.argv[1:]
    df = load_all_cohorts(extra_dirs=extra_dirs)
    if df is None:
        print("No per_mol.csv files found.")
        sys.exit(1)

    # Ensure transformed metrics exist
    if "d_cb_sg_diff_from_185" not in df.columns and "d_cb_sg_top1" in df.columns:
        df["d_cb_sg_diff_from_185"] = (df["d_cb_sg_top1"] - 1.85).abs()
    if "bd_angle_diff_from_107" not in df.columns and "bd_angle_top1" in df.columns:
        df["bd_angle_diff_from_107"] = (df["bd_angle_top1"] - 107.0).abs()

    out_csv = RESULTS_DIR / "all_cohorts_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(df)} rows, {df['cohort'].nunique()} cohorts")

    cohorts_present = [c for c in COHORT_ORDER if c in df["cohort"].unique()]
    others = sorted(set(df["cohort"].unique()) - set(cohorts_present))
    cohorts_present.extend(others)
    print(f"Cohorts: {cohorts_present}")

    # Per-cohort summary
    summary_rows = []
    for cohort in cohorts_present:
        sub = df[df["cohort"] == cohort]
        row = {"cohort": cohort, "group": GROUP_MAP.get(cohort, "unknown"), "N": len(sub),
               "N_success": int(sub["success"].sum()) if "success" in sub.columns else len(sub)}
        for key, label, _ in METRICS:
            if key not in sub.columns:
                continue
            v = sub[key].dropna()
            if len(v) == 0:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_std"] = np.nan
                row[f"{key}_median"] = np.nan
                row[f"{key}_N"] = 0
                continue
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_std"] = float(v.std())
            row[f"{key}_median"] = float(v.median())
            row[f"{key}_N"] = int(len(v))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "per_cohort_summary.csv", index=False)
    print(f"Wrote per_cohort_summary.csv")

    # Pairwise comparisons — Mann-Whitney U and Cliff's delta
    metric_keys = [m[0] for m in METRICS]
    pair_rows = []
    raw_p_matrix = []
    metric_columns = []  # for heatmap
    for key, label, _ in METRICS:
        if key not in df.columns:
            continue
        metric_columns.append(key)
        for a, b in combinations(cohorts_present, 2):
            va = df.loc[df["cohort"] == a, key].dropna().to_numpy()
            vb = df.loc[df["cohort"] == b, key].dropna().to_numpy()
            if len(va) < 3 or len(vb) < 3:
                pair_rows.append({"metric": key, "cohort_a": a, "cohort_b": b,
                                  "n_a": len(va), "n_b": len(vb),
                                  "u_stat": np.nan, "pvalue_raw": np.nan,
                                  "cliffs_delta": np.nan})
                continue
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            cd, _, _ = cliffs_delta(va, vb)
            pair_rows.append({"metric": key, "cohort_a": a, "cohort_b": b,
                              "n_a": len(va), "n_b": len(vb),
                              "u_stat": float(stat) if not np.isnan(stat) else np.nan,
                              "pvalue_raw": float(p) if not np.isnan(p) else np.nan,
                              "cliffs_delta": cd})
    pair_df = pd.DataFrame(pair_rows)
    pair_df["pvalue_bh"] = bh_correct(pair_df["pvalue_raw"].to_numpy())
    pair_df.to_csv(RESULTS_DIR / "comparison_pvalues.csv", index=False)
    print(f"Wrote comparison_pvalues.csv: {len(pair_df)} pairs")
    pair_df[["metric", "cohort_a", "cohort_b", "n_a", "n_b", "cliffs_delta"]].to_csv(
        RESULTS_DIR / "comparison_effect_sizes.csv", index=False
    )

    # -------- Figures --------
    palette = {"lingo_pcov": "#1f77b4", "seqonly": "#d62728", "unknown": "#7f7f7f"}

    # 1. Box plots per metric
    for key, label, _ in METRICS:
        if key not in df.columns:
            continue
        data_by_cohort = []
        labels = []
        colors = []
        for cohort in cohorts_present:
            v = df.loc[df["cohort"] == cohort, key].dropna()
            if len(v) == 0:
                continue
            data_by_cohort.append(v.values)
            labels.append(f"{cohort}\nN={len(v)}")
            colors.append(palette[GROUP_MAP.get(cohort, "unknown")])
        if not data_by_cohort:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(data_by_cohort)), 4.5))
        bp = ax.boxplot(data_by_cohort, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by cohort")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"box_{key}.png", dpi=110)
        plt.close()
    print(f"Wrote box plots for {sum(1 for k,_,_ in METRICS if k in df.columns)} metrics")

    # 2. P-value heatmap (cohort-pairs x metrics)
    pairs_list = list(combinations(cohorts_present, 2))
    n_pairs = len(pairs_list)
    n_metrics = len(metric_columns)
    if n_pairs > 0 and n_metrics > 0:
        mat = np.full((n_pairs, n_metrics), np.nan)
        for i, (a, b) in enumerate(pairs_list):
            for j, key in enumerate(metric_columns):
                rows_match = pair_df[(pair_df["metric"] == key) &
                                     (pair_df["cohort_a"] == a) &
                                     (pair_df["cohort_b"] == b)]
                if len(rows_match):
                    mat[i, j] = rows_match["pvalue_bh"].iloc[0]
        # log10
        mat_log = -np.log10(np.where(mat > 0, mat, np.nan))
        h = max(6, 0.18 * n_pairs)
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * n_metrics), h))
        im = ax.imshow(mat_log, aspect="auto", cmap="viridis", vmin=0, vmax=8)
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(metric_columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs_list], fontsize=7)
        ax.set_title("-log10 BH-corrected p-value (Mann-Whitney U)")
        plt.colorbar(im, ax=ax, label="-log10(p_BH)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "pvalue_heatmap.png", dpi=110)
        plt.close()
        print(f"Wrote pvalue_heatmap.png")

    # 3. Summary bar chart: mean per cohort for each headline metric
    headline_metrics = ["vina_score", "d_hinge_top1", "warhead_largest_frag",
                        "d_cb_sg_input", "bd_angle_input_diff_from_107", "any_pose_feasible"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, key in zip(axes.flat, headline_metrics):
        if key not in df.columns:
            ax.axis("off"); continue
        means = []; stds = []; labels = []; colors = []
        for c in cohorts_present:
            v = df.loc[df["cohort"] == c, key].dropna()
            if len(v) == 0: continue
            means.append(v.mean()); stds.append(v.std()/np.sqrt(max(1,len(v))))
            labels.append(c); colors.append(palette[GROUP_MAP.get(c, "unknown")])
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(key)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "summary_bars.png", dpi=110)
    plt.close()
    print(f"Wrote summary_bars.png")

    print(f"\nAll figures -> {FIG_DIR}")
    print(f"All tables -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
