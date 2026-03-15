#!/usr/bin/env python3
"""
Noise-Performance Analysis: 2x2 factorial design.

Tests whether the edit effect framework's advantage grows with measurement noise.

Design:
  - Architecture: FiLMDelta vs Subtraction
  - Training data: within-assay only vs all pairs (within + cross)
  - Evaluation: per-target MAE on within-assay test pairs (clean ground truth)
  - Stratification: by per-target variance ratio (noise level)

Usage:
    python -u experiments/run_noise_performance_analysis.py
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from run_paper_evaluation import (
    load_data, compute_embeddings, run_single_experiment,
    SEEDS, RESULTS_DIR,
)
from src.utils.splits import get_splitter


def compute_per_target_noise(df, min_pairs=30):
    """Compute variance ratio (cross/within) per target."""
    within = df[df['is_within_assay']]
    cross = df[~df['is_within_assay']]

    targets = df['target_chembl_id'].unique()
    noise_stats = {}

    for t in targets:
        w = within[within['target_chembl_id'] == t]['delta']
        c = cross[cross['target_chembl_id'] == t]['delta']
        if len(w) >= min_pairs and len(c) >= min_pairs:
            var_w = w.var()
            var_c = c.var()
            if var_w > 0:
                noise_stats[t] = {
                    'var_within': float(var_w),
                    'var_cross': float(var_c),
                    'ratio': float(var_c / var_w),
                    'n_within': len(w),
                    'n_cross': len(c),
                }

    return noise_stats


def split_within_only(df, seed):
    """Split using assay splitter, return only within-assay pairs."""
    splitter = get_splitter("assay", random_state=seed, scenario="within_assay")
    return splitter.split(df)


def split_all_data(df, seed):
    """Split using assay splitter with mixed scenario (within + cross training)."""
    splitter = get_splitter("assay", random_state=seed, scenario="mixed")
    return splitter.split(df)


def compute_per_target_mae(test_df, predictions):
    """Compute MAE per target."""
    test_df = test_df.copy()
    test_df['pred'] = predictions
    test_df['error'] = (test_df['pred'] - test_df['delta']).abs()

    per_target = test_df.groupby('target_chembl_id').agg(
        mae=('error', 'mean'),
        n=('error', 'count'),
    ).to_dict('index')

    return per_target


def log(msg, log_file=None):
    """Print and write to log file."""
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def main():
    log_file = RESULTS_DIR / "noise_analysis_progress.log"
    results_file = RESULTS_DIR / "noise_performance_analysis.json"

    # Clear log
    with open(log_file, "w") as f:
        f.write("")

    log("=" * 60, log_file)
    log("Noise-Performance Analysis (2x2 Factorial)", log_file)
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log("=" * 60, log_file)

    df = load_data()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
    log(f"Data loaded: {len(df):,} pairs, {len(all_smiles):,} mols, dim={emb_dim}", log_file)

    # Step 1: Compute per-target noise levels
    log("\n--- Step 1: Per-target noise characterization ---", log_file)
    noise_stats = compute_per_target_noise(df, min_pairs=30)
    log(f"  {len(noise_stats)} targets with >=30 pairs in each context", log_file)

    ratios = [v['ratio'] for v in noise_stats.values()]
    log(f"  Variance ratio: mean={np.mean(ratios):.3f}, median={np.median(ratios):.3f}", log_file)
    log(f"  Low noise (<1.2): {sum(1 for r in ratios if r < 1.2)} targets", log_file)
    log(f"  Medium noise (1.2-2.0): {sum(1 for r in ratios if 1.2 <= r < 2.0)} targets", log_file)
    log(f"  High noise (>=2.0): {sum(1 for r in ratios if r >= 2.0)} targets", log_file)

    # Step 2: Run 2x2 factorial (2 architectures x 2 data conditions x 3 seeds)
    conditions = [
        ("FiLMDelta", "within", "FiLMDelta + within-assay"),
        ("Subtraction", "within", "Subtraction + within-assay"),
        ("FiLMDelta", "all", "FiLMDelta + all data"),
        ("Subtraction", "all", "Subtraction + all data"),
    ]
    total_runs = len(SEEDS) * len(conditions)
    log(f"\n--- Step 2: 2x2 Factorial ({total_runs} runs) ---", log_file)

    # Collect per-target MAE across seeds
    all_per_target = {label: defaultdict(list) for _, _, label in conditions}
    run_count = 0

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)})", log_file)

        # Get splits — free previous ones first
        t0 = time.time()
        train_w, val_w, test_w = split_within_only(df, seed)
        train_a, val_a, test_a = split_all_data(df, seed)
        log(f"    Splits ready ({time.time()-t0:.0f}s): within train={len(train_w):,}, mixed train={len(train_a):,}", log_file)

        for arch, data_type, label in conditions:
            run_count += 1
            t0 = time.time()
            try:
                if data_type == "within":
                    train_df, val_df, test_df = train_w, val_w, test_w
                else:
                    train_df, val_df, test_df = train_a, val_a, test_a

                metrics, preds = run_single_experiment(
                    train_df, val_df, test_df, emb_dict, emb_dim, arch, seed
                )
                elapsed = time.time() - t0
                log(f"    [{run_count}/{total_runs}] {label}: MAE={metrics['mae']:.4f} ({elapsed:.0f}s)", log_file)

                # Per-target MAE
                pt_mae = compute_per_target_mae(test_df, preds)
                for t, st in pt_mae.items():
                    if st['n'] >= 20:
                        all_per_target[label][t].append(st['mae'])

            except Exception as e:
                elapsed = time.time() - t0
                log(f"    [{run_count}/{total_runs}] {label}: ERROR ({elapsed:.0f}s): {e}", log_file)
                import traceback; traceback.print_exc()
            finally:
                gc.collect()

        # Save intermediate results after each seed
        intermediate = {
            'noise_stats': noise_stats,
            'completed_seeds': seed_idx + 1,
            'total_seeds': len(SEEDS),
            'completed_runs': run_count,
            'total_runs': total_runs,
            'per_target_mae_partial': {
                label: {t: float(np.mean(maes)) for t, maes in all_per_target[label].items()}
                for label in all_per_target
            },
        }
        with open(results_file, "w") as f:
            json.dump(intermediate, f, indent=2, default=str)
        log(f"    Intermediate results saved (seed {seed} done)", log_file)

    # Step 3: Analyze noise vs advantage
    log("\n--- Step 3: Noise vs Advantage Analysis ---", log_file)

    # Average per-target MAE across seeds
    avg_mae = {}
    for label in all_per_target:
        avg_mae[label] = {
            t: np.mean(maes) for t, maes in all_per_target[label].items()
            if len(maes) == len(SEEDS)  # only targets with all seeds
        }

    # The headline comparison: FiLMDelta-within vs Subtraction-all
    headline_label_a = "FiLMDelta + within-assay"
    headline_label_b = "Subtraction + all data"
    common_targets = set(avg_mae[headline_label_a].keys()) & set(avg_mae[headline_label_b].keys())
    common_targets = common_targets & set(noise_stats.keys())
    log(f"  {len(common_targets)} targets with all conditions + noise stats", log_file)

    results = {
        'noise_stats': noise_stats,
        'per_target_mae': {label: dict(m) for label, m in avg_mae.items()},
        'analysis': {},
        'completed': True,
    }

    if len(common_targets) > 10:
        targets_sorted = sorted(common_targets)
        noise_ratios = [noise_stats[t]['ratio'] for t in targets_sorted]
        advantages = [
            avg_mae[headline_label_b][t] - avg_mae[headline_label_a][t]
            for t in targets_sorted
        ]
        rel_advantages = [
            (avg_mae[headline_label_b][t] - avg_mae[headline_label_a][t]) / avg_mae[headline_label_b][t]
            for t in targets_sorted
        ]

        # Correlation
        spearman_r, spearman_p = stats.spearmanr(noise_ratios, advantages)
        pearson_r, pearson_p = stats.pearsonr(noise_ratios, advantages)

        log(f"\n  Headline: FiLMDelta(within) vs Subtraction(all)", log_file)
        log(f"  Spearman(noise_ratio, advantage): r={spearman_r:.4f}, p={spearman_p:.4e}", log_file)
        log(f"  Pearson(noise_ratio, advantage): r={pearson_r:.4f}, p={pearson_p:.4e}", log_file)

        # Tier analysis
        for tier_name, lo, hi in [("Low (<1.2)", 0, 1.2), ("Med (1.2-2)", 1.2, 2.0), ("High (>=2)", 2.0, 999)]:
            tier_targets = [t for t in targets_sorted if lo <= noise_stats[t]['ratio'] < hi]
            if tier_targets:
                tier_adv = [avg_mae[headline_label_b][t] - avg_mae[headline_label_a][t] for t in tier_targets]
                tier_mae_a = [avg_mae[headline_label_a][t] for t in tier_targets]
                tier_mae_b = [avg_mae[headline_label_b][t] for t in tier_targets]
                log(f"  {tier_name}: {len(tier_targets)} targets, "
                    f"FiLM MAE={np.mean(tier_mae_a):.4f}, Sub MAE={np.mean(tier_mae_b):.4f}, "
                    f"D={np.mean(tier_adv):.4f} ({100*np.mean(tier_adv)/np.mean(tier_mae_b):.1f}%)", log_file)

        results['analysis'] = {
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'n_targets': len(common_targets),
        }

        # Also decompose: data effect vs architecture effect
        log(f"\n  Decomposition:", log_file)
        for label_a, label_b, desc in [
            ("Subtraction + within-assay", "Subtraction + all data", "Data effect (Sub within vs Sub all)"),
            ("FiLMDelta + within-assay", "Subtraction + within-assay", "Arch effect (FiLM vs Sub, same data)"),
            ("FiLMDelta + all data", "Subtraction + all data", "Arch effect (FiLM vs Sub, all data)"),
        ]:
            ct = set(avg_mae.get(label_a, {}).keys()) & set(avg_mae.get(label_b, {}).keys()) & set(noise_stats.keys())
            if len(ct) > 10:
                nr = [noise_stats[t]['ratio'] for t in sorted(ct)]
                adv = [avg_mae[label_b][t] - avg_mae[label_a][t] for t in sorted(ct)]
                sr, sp = stats.spearmanr(nr, adv)
                log(f"    {desc}: Spearman r={sr:.4f}, p={sp:.4e}", log_file)

    # Save final
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log(f"Saved to {results_file}", log_file)
    log("=" * 60, log_file)


if __name__ == "__main__":
    main()
