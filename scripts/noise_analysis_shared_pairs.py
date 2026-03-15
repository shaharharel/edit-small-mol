"""
Noise analysis on shared pairs dataset (Claim 1).

Compares within-assay vs cross-assay delta variance to quantify
lab-to-lab noise in bioactivity measurements.

Reference: Landrum & Riniker, JCIM 2024
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


DATA_PATH = Path("data/overlapping_assays/extracted/shared_pairs_deduped.csv")
OUTPUT_PATH = Path("results/paper_evaluation/noise_analysis.json")

MIN_PAIRS_PER_TARGET = 50


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / DATA_PATH
    output_path = project_root / OUTPUT_PATH

    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path, usecols=[
        "mol_a", "mol_b", "delta", "is_within_assay", "target_chembl_id",
        "value_a", "value_b"
    ])
    print(f"  Loaded {len(df):,} pairs")

    within = df[df["is_within_assay"] == True]["delta"].values
    cross = df[df["is_within_assay"] == False]["delta"].values
    print(f"  Within-assay: {len(within):,} pairs")
    print(f"  Cross-assay:  {len(cross):,} pairs")

    # --- 1. Variance ---
    var_within = float(np.var(within, ddof=1))
    var_cross = float(np.var(cross, ddof=1))
    print(f"\n{'='*60}")
    print("1. VARIANCE OF DELTA")
    print(f"{'='*60}")
    print(f"  Var(within) = {var_within:.6f}")
    print(f"  Var(cross)  = {var_cross:.6f}")

    # --- 2. Variance ratio ---
    ratio = var_cross / var_within
    print(f"\n{'='*60}")
    print("2. VARIANCE RATIO (cross / within)")
    print(f"{'='*60}")
    print(f"  Ratio = {ratio:.4f}x")

    # --- 3. Levene's test ---
    levene_stat, levene_p = stats.levene(within, cross)
    print(f"\n{'='*60}")
    print("3. LEVENE'S TEST FOR EQUALITY OF VARIANCES")
    print(f"{'='*60}")
    print(f"  Statistic = {levene_stat:.4f}")
    print(f"  p-value   = {levene_p:.4e}")
    print(f"  Significant (p < 0.05): {levene_p < 0.05}")

    # --- 4. Per-target breakdown ---
    print(f"\n{'='*60}")
    print("4. PER-TARGET VARIANCE RATIO (top 15)")
    print(f"{'='*60}")

    target_stats = []
    for target, grp in df.groupby("target_chembl_id"):
        w = grp[grp["is_within_assay"] == True]["delta"]
        c = grp[grp["is_within_assay"] == False]["delta"]
        if len(w) >= MIN_PAIRS_PER_TARGET and len(c) >= MIN_PAIRS_PER_TARGET:
            vw = float(np.var(w, ddof=1))
            vc = float(np.var(c, ddof=1))
            if vw == 0 and vc == 0:
                r = 1.0  # no variance in either => no difference
            elif vw == 0:
                r = float("inf")
            else:
                r = vc / vw
            target_stats.append({
                "target": target,
                "n_within": int(len(w)),
                "n_cross": int(len(c)),
                "var_within": round(vw, 6),
                "var_cross": round(vc, 6),
                "variance_ratio": round(r, 4),
            })

    # Separate finite and infinite ratios
    finite_stats = [t for t in target_stats if np.isfinite(t["variance_ratio"])]
    inf_stats = [t for t in target_stats if not np.isfinite(t["variance_ratio"])]
    finite_stats.sort(key=lambda x: x["variance_ratio"], reverse=True)
    print(f"  Targets with >= {MIN_PAIRS_PER_TARGET} within AND cross pairs: {len(target_stats)}")
    if inf_stats:
        print(f"  (Excluding {len(inf_stats)} targets with zero within-assay variance)")
    print()
    print(f"  {'Target':<20} {'N_within':>8} {'N_cross':>8} {'Var_w':>10} {'Var_c':>10} {'Ratio':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for t in finite_stats[:15]:
        print(f"  {t['target']:<20} {t['n_within']:>8} {t['n_cross']:>8} "
              f"{t['var_within']:>10.4f} {t['var_cross']:>10.4f} {t['variance_ratio']:>8.3f}")

    # Summary stats across targets (finite ratios only)
    ratios_all = [t["variance_ratio"] for t in finite_stats]
    median_ratio = float(np.median(ratios_all))
    mean_ratio = float(np.mean(ratios_all))
    frac_above_1 = sum(1 for r in ratios_all if r > 1.0) / len(ratios_all)
    print(f"\n  Per-target summary ({len(target_stats)} targets):")
    print(f"    Median ratio: {median_ratio:.4f}")
    print(f"    Mean ratio:   {mean_ratio:.4f}")
    print(f"    Fraction > 1: {frac_above_1:.1%}")

    # --- 5. Noise decomposition ---
    print(f"\n{'='*60}")
    print("5. NOISE DECOMPOSITION")
    print(f"{'='*60}")

    # Var(cross) = Var(SAR) + Var(lab)
    # Var(within) = Var(SAR)
    # ratio = Var(cross)/Var(within) = 1 + Var(lab)/Var(SAR)
    # => (sigma_lab / sigma_SAR)^2 = ratio - 1
    sigma_ratio_sq = ratio - 1
    sigma_ratio = float(np.sqrt(max(sigma_ratio_sq, 0)))
    frac_from_lab = 1 - 1 / ratio

    print(f"  Model: Var(cross) = Var(SAR) + Var(lab)")
    print(f"  (sigma_lab / sigma_SAR)^2 = {sigma_ratio_sq:.4f}")
    print(f"  sigma_lab / sigma_SAR      = {sigma_ratio:.4f}")
    print(f"  Fraction of cross-assay variance from lab noise = {frac_from_lab:.1%}")
    print(f"  => Cross-assay deltas are {sigma_ratio_sq*100:.1f}% noisier (in variance)")

    # --- Save results ---
    results = {
        "dataset": str(data_path.name),
        "n_total": int(len(df)),
        "n_within": int(len(within)),
        "n_cross": int(len(cross)),
        "variance": {
            "within": round(var_within, 6),
            "cross": round(var_cross, 6),
            "ratio": round(ratio, 4),
        },
        "levene_test": {
            "statistic": round(levene_stat, 4),
            "p_value": float(levene_p),
            "significant": bool(levene_p < 0.05),
        },
        "noise_decomposition": {
            "sigma_lab_over_sigma_sar_squared": round(sigma_ratio_sq, 4),
            "sigma_lab_over_sigma_sar": round(sigma_ratio, 4),
            "frac_cross_variance_from_lab": round(frac_from_lab, 4),
        },
        "per_target": {
            "n_targets": len(target_stats),
            "min_pairs_threshold": MIN_PAIRS_PER_TARGET,
            "median_ratio": round(median_ratio, 4),
            "mean_ratio": round(mean_ratio, 4),
            "frac_above_1": round(frac_above_1, 4),
            "n_inf_targets": len(inf_stats),
            "top_15": finite_stats[:15],
            "bottom_5": finite_stats[-5:],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
