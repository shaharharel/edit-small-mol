#!/usr/bin/env python3
"""
Realistic Noise Tier Experiment.

Compares how practitioners would ACTUALLY use each method:
  - FiLMDelta: trains on within-assay pairs (edit-effect approach)
  - Subtraction: trains on ALL available pairs (traditional approach)

Targets grouped by natural noise level (variance ratio cross/within).
Test on held-out within-assay pairs (clean ground truth).

This tests: does FiLMDelta's advantage grow with target noise level?

Design:
  - ~40 targets spanning noise ratios 0.3x to 13x
  - FiLMDelta trains on within-assay pairs only
  - Subtraction trains on ALL pairs (within + cross-assay)
  - Test on held-out within-assay pairs
  - 3 seeds per condition
  - Results stratified by noise tier

Usage:
    python -u experiments/run_fair_noise_tiers.py
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path

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
    compute_embeddings, run_single_experiment,
    RESULTS_DIR,
)

SEEDS = [42, 123, 456]
MIN_WITHIN_PAIRS = 200
MIN_CROSS_PAIRS = 30
LOG_FILE = RESULTS_DIR / "fair_noise_tiers_progress.log"
RESULTS_FILE = RESULTS_DIR / "fair_noise_tiers_results.json"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def select_targets(df, max_low=20):
    """Select targets with stratified sampling: ALL high/medium, subsample low.

    Tiers:
      - High (>=3x):   take ALL
      - Medium (1.5-3x): take ALL
      - Low (<1.5x):   subsample evenly across range, up to max_low
    """
    within = df[df['is_within_assay']]
    cross = df[~df['is_within_assay']]

    candidates = []
    for tgt in df['target_chembl_id'].unique():
        w = within[within['target_chembl_id'] == tgt]
        c = cross[cross['target_chembl_id'] == tgt]
        if len(w) < MIN_WITHIN_PAIRS or len(c) < MIN_CROSS_PAIRS:
            continue
        var_w = w['delta'].var()
        var_c = c['delta'].var()
        if var_w <= 0:
            continue
        ratio = var_c / var_w
        candidates.append({
            'target': tgt,
            'n_within': len(w),
            'n_cross': len(c),
            'n_all': len(w) + len(c),
            'var_within': float(var_w),
            'var_cross': float(var_c),
            'noise_ratio': float(ratio),
        })

    candidates.sort(key=lambda x: x['noise_ratio'])

    # Stratified: all high + all medium + subsampled low
    high = [c for c in candidates if c['noise_ratio'] >= 3.0]
    medium = [c for c in candidates if 1.5 <= c['noise_ratio'] < 3.0]
    low = [c for c in candidates if c['noise_ratio'] < 1.5]

    # Subsample low tier evenly
    if len(low) > max_low:
        step = len(low) / max_low
        low = [low[int(i * step)] for i in range(max_low)]

    selected = low + medium + high
    selected.sort(key=lambda x: x['noise_ratio'])
    return selected


def split_target_data(within_df, all_df, seed):
    """Split: test/val from within-assay. Return within-train and all-train separately.

    FiLMDelta trains on within-assay train split only.
    Subtraction trains on within-assay train split + ALL cross-assay pairs.
    """
    rng = np.random.RandomState(seed)
    n = len(within_df)
    indices = rng.permutation(n)
    n_test = max(20, int(n * 0.15))
    n_val = max(10, int(n * 0.15))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_within_idx = indices[n_test + n_val:]

    test_df = within_df.iloc[test_idx]
    val_df = within_df.iloc[val_idx]
    train_within = within_df.iloc[train_within_idx]

    # For Subtraction: within-assay train + ALL cross-assay pairs
    cross_df = all_df[~all_df['is_within_assay']]
    train_all = pd.concat([train_within, cross_df], ignore_index=True)

    return train_within, train_all, val_df, test_df


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 60)
    log("Fair Noise Tier Experiment")
    log("Both methods train on SAME all-data per target")
    log("Started: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
    log("=" * 60)

    t_start = time.time()

    # Load data
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
    )
    log("Dataset: {:,} pairs".format(len(df)))

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
    log("Embeddings: {:,} molecules, dim={}".format(len(emb_dict), emb_dim))

    # Select targets
    targets = select_targets(df)
    log("\nSelected {} targets (ratio range: {:.2f}x - {:.2f}x)".format(
        len(targets), targets[0]['noise_ratio'], targets[-1]['noise_ratio']))

    for i, t in enumerate(targets):
        log("  {:2d}. {} ratio={:.2f}x, n_within={:,}, n_cross={:,}".format(
            i + 1, t['target'], t['noise_ratio'], t['n_within'], t['n_cross']))

    # Run experiments
    methods = ["FiLMDelta", "Subtraction"]
    total_runs = len(targets) * len(methods) * len(SEEDS)
    run_count = 0

    results = {
        'targets': targets,
        'per_target_results': {},
    }

    for ti, target_info in enumerate(targets):
        tgt = target_info['target']
        tgt_df = df[df['target_chembl_id'] == tgt]
        within_df = tgt_df[tgt_df['is_within_assay']].reset_index(drop=True)
        all_df = tgt_df.reset_index(drop=True)

        log("\n--- Target {}/{}: {} (ratio={:.2f}x, n_within={:,}, n_cross={:,}) ---".format(
            ti + 1, len(targets), tgt, target_info['noise_ratio'],
            target_info['n_within'], target_info['n_cross']))

        target_results = {'seeds': {}}

        for seed in SEEDS:
            # FiLMDelta trains on within-assay only; Subtraction on all data
            train_within, train_all, val_df, test_df = split_target_data(
                within_df, all_df, seed
            )

            seed_results = {}
            for method in methods:
                run_count += 1
                t0 = time.time()

                # FiLMDelta uses within-assay pairs; Subtraction uses all
                if method == "FiLMDelta":
                    train_df = train_within
                else:
                    train_df = train_all

                try:
                    metrics, preds = run_single_experiment(
                        train_df, val_df, test_df, emb_dict, emb_dim, method, seed
                    )
                    elapsed = time.time() - t0
                    seed_results[method] = {
                        'mae': float(metrics['mae']),
                        'rmse': float(metrics.get('rmse', 0)),
                        'spearman_r': float(metrics.get('spearman_r', 0)),
                        'pearson_r': float(metrics.get('pearson_r', 0)),
                        'r2': float(metrics.get('r2', 0)),
                        'n_train': len(train_df),
                        'n_test': len(test_df),
                    }
                    log("  [{}/{}] {}: MAE={:.4f}, n_train={:,} ({:.0f}s)".format(
                        run_count, total_runs, method,
                        metrics['mae'], len(train_df), elapsed))
                except Exception as e:
                    elapsed = time.time() - t0
                    log("  [{}/{}] {} ERROR ({:.0f}s): {}".format(
                        run_count, total_runs, method, elapsed, e))
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()

            target_results['seeds'][seed] = seed_results

        # Aggregate
        film_maes = [target_results['seeds'][s].get('FiLMDelta', {}).get('mae')
                     for s in SEEDS if 'FiLMDelta' in target_results['seeds'].get(s, {})]
        sub_maes = [target_results['seeds'][s].get('Subtraction', {}).get('mae')
                    for s in SEEDS if 'Subtraction' in target_results['seeds'].get(s, {})]

        film_maes = [m for m in film_maes if m is not None]
        sub_maes = [m for m in sub_maes if m is not None]

        if film_maes and sub_maes:
            target_results['aggregated'] = {
                'film_mae_mean': float(np.mean(film_maes)),
                'film_mae_std': float(np.std(film_maes)),
                'sub_mae_mean': float(np.mean(sub_maes)),
                'sub_mae_std': float(np.std(sub_maes)),
                'advantage': float(np.mean(sub_maes) - np.mean(film_maes)),
                'advantage_pct': float(
                    (np.mean(sub_maes) - np.mean(film_maes)) / np.mean(sub_maes) * 100
                ) if np.mean(sub_maes) > 0 else 0,
                'noise_ratio': target_info['noise_ratio'],
            }
            agg = target_results['aggregated']
            log("  => FiLM={:.4f}±{:.4f}, Sub={:.4f}±{:.4f}, Δ={:+.4f} ({:+.1f}%)".format(
                agg['film_mae_mean'], agg['film_mae_std'],
                agg['sub_mae_mean'], agg['sub_mae_std'],
                agg['advantage'], agg['advantage_pct']))

        results['per_target_results'][tgt] = target_results

        # Save after each target
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final analysis
    log("\n" + "=" * 60)
    log("Final Analysis")
    log("=" * 60)

    noise_ratios = []
    advantages = []
    advantage_pcts = []

    for tgt, tr in results['per_target_results'].items():
        agg = tr.get('aggregated')
        if agg:
            noise_ratios.append(agg['noise_ratio'])
            advantages.append(agg['advantage'])
            advantage_pcts.append(agg['advantage_pct'])

    noise_ratios = np.array(noise_ratios)
    advantages = np.array(advantages)
    advantage_pcts = np.array(advantage_pcts)

    # Correlation
    if len(noise_ratios) >= 5:
        sp_r, sp_p = stats.spearmanr(noise_ratios, advantages)
        pe_r, pe_p = stats.pearsonr(noise_ratios, advantages)
        sp_r_pct, sp_p_pct = stats.spearmanr(noise_ratios, advantage_pcts)
        pe_r_pct, pe_p_pct = stats.pearsonr(noise_ratios, advantage_pcts)

        log("\nCorrelation: noise_ratio vs absolute advantage")
        log("  Spearman r={:.4f}, p={:.4e}".format(sp_r, sp_p))
        log("  Pearson  r={:.4f}, p={:.4e}".format(pe_r, pe_p))
        log("\nCorrelation: noise_ratio vs advantage %")
        log("  Spearman r={:.4f}, p={:.4e}".format(sp_r_pct, sp_p_pct))
        log("  Pearson  r={:.4f}, p={:.4e}".format(pe_r_pct, pe_p_pct))

    film_wins = sum(1 for a in advantages if a > 0)
    log("\nFiLMDelta wins: {}/{} targets ({:.0f}%)".format(
        film_wins, len(advantages), film_wins / len(advantages) * 100 if len(advantages) > 0 else 0))

    # Tier analysis
    tiers = [
        ("Low (<1.5x)", 0, 1.5),
        ("Medium (1.5-3x)", 1.5, 3.0),
        ("High (≥3x)", 3.0, 999),
    ]
    log("\nPer-Tier Summary:")
    log("{:<18} {:>8} {:>10} {:>10} {:>10} {:>8}".format(
        "Tier", "Targets", "FiLM MAE", "Sub MAE", "ΔMAE", "FiLM%"))
    for tier_name, lo, hi in tiers:
        tier_idx = [i for i, r in enumerate(noise_ratios) if lo <= r < hi]
        if tier_idx:
            tier_film = np.mean([advantages[i] + np.mean([
                results['per_target_results'][list(results['per_target_results'].keys())[i]]['aggregated']['sub_mae_mean']
            ]) - advantages[i] for i in tier_idx])
            # Simpler: just get the values
            tier_data = []
            for tgt, tr in results['per_target_results'].items():
                agg = tr.get('aggregated')
                if agg and lo <= agg['noise_ratio'] < hi:
                    tier_data.append(agg)
            if tier_data:
                fm = np.mean([t['film_mae_mean'] for t in tier_data])
                sm = np.mean([t['sub_mae_mean'] for t in tier_data])
                wins = sum(1 for t in tier_data if t['advantage'] > 0)
                log("{:<18} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>7.0f}%".format(
                    tier_name, len(tier_data), fm, sm, sm - fm,
                    wins / len(tier_data) * 100))

    results['analysis'] = {
        'n_targets': len(advantages),
        'film_wins': int(film_wins),
        'mean_advantage_pct': float(np.mean(advantage_pcts)) if len(advantage_pcts) > 0 else 0,
    }
    if len(noise_ratios) >= 5:
        results['analysis']['spearman_r'] = float(sp_r)
        results['analysis']['spearman_p'] = float(sp_p)
        results['analysis']['pearson_r'] = float(pe_r)
        results['analysis']['pearson_p'] = float(pe_p)
        results['analysis']['spearman_r_pct'] = float(sp_r_pct)
        results['analysis']['spearman_p_pct'] = float(sp_p_pct)
        results['analysis']['pearson_r_pct'] = float(pe_r_pct)
        results['analysis']['pearson_p_pct'] = float(pe_p_pct)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    total_time = time.time() - t_start
    log("\nCompleted: {} ({:.1f}h)".format(time.strftime('%Y-%m-%d %H:%M:%S'), total_time / 3600))
    log("Results saved to {}".format(RESULTS_FILE))
    log("=" * 60)


if __name__ == "__main__":
    main()
