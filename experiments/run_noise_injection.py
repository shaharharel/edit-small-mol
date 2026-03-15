#!/usr/bin/env python3
"""
Controlled Noise Injection Experiment.

Tests noise robustness by adding synthetic Gaussian noise to within-assay
delta labels at varying levels. Compares how FiLMDelta vs Subtraction
degrade under increasing label noise.

Design:
  - Base data: within-assay pairs only (clean ground truth)
  - Noise levels: 0.0, 0.1, 0.3, 0.5, 1.0, 1.5 pIC50 std
  - Test set always uses CLEAN labels (held out before noise injection)
  - 3 seeds per condition

This directly tests: does the edit effect framework degrade more gracefully
under label noise than the subtraction baseline?

Usage:
    python -u experiments/run_noise_injection.py
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
    load_data, compute_embeddings, run_single_experiment,
    SEEDS, RESULTS_DIR,
)
from src.utils.splits import get_splitter

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5]
METHODS = ["FiLMDelta", "EditDiff", "Subtraction"]
LOG_FILE = RESULTS_DIR / "noise_injection_progress.log"
RESULTS_FILE = RESULTS_DIR / "noise_injection_results.json"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def inject_noise(df, noise_std, seed):
    """Add Gaussian noise to delta labels. Also adjust value_b accordingly."""
    if noise_std == 0.0:
        return df.copy()

    rng = np.random.RandomState(seed + 999)  # different from split seed
    noisy_df = df.copy()
    noise = rng.normal(0, noise_std, size=len(noisy_df))
    noisy_df['delta'] = noisy_df['delta'] + noise
    noisy_df['value_b'] = noisy_df['value_a'] + noisy_df['delta']
    return noisy_df


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 60)
    log("Controlled Noise Injection Experiment")
    log("Started: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
    log("=" * 60)

    t_start = time.time()

    # Load data — within-assay only, subsample for speed
    df = load_data()
    within_df = df[df['is_within_assay']].reset_index(drop=True)
    MAX_PAIRS = 50000
    if len(within_df) > MAX_PAIRS:
        within_df = within_df.sample(n=MAX_PAIRS, random_state=42).reset_index(drop=True)
    log("Within-assay pairs: {:,}".format(len(within_df)))

    # Load embeddings
    all_smiles = list(set(within_df["mol_a"].tolist() + within_df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
    log("Embeddings: {:,} molecules, dim={}".format(len(emb_dict), emb_dim))

    total_runs = len(NOISE_LEVELS) * len(METHODS) * len(SEEDS)
    run_count = 0

    results = {"noise_levels": NOISE_LEVELS, "methods": METHODS, "results": {}}

    for noise_std in NOISE_LEVELS:
        log("\n--- Noise level: sigma={:.1f} ---".format(noise_std))
        noise_results = {}

        for method in METHODS:
            seed_maes = []
            for seed in SEEDS:
                run_count += 1
                t0 = time.time()

                # Split FIRST with clean data, then inject noise into train/val only
                splitter = get_splitter("assay", random_state=seed, scenario="within_assay")
                train_df, val_df, test_df = splitter.split(within_df)

                # Inject noise into train and val (test stays clean!)
                train_noisy = inject_noise(train_df, noise_std, seed)
                val_noisy = inject_noise(val_df, noise_std, seed)

                try:
                    metrics, _ = run_single_experiment(
                        train_noisy, val_noisy, test_df, emb_dict, emb_dim, method, seed
                    )
                    elapsed = time.time() - t0
                    seed_maes.append(float(metrics['mae']))
                    log("  [{}/{}] {} sigma={:.1f} seed={}: MAE={:.4f} ({:.0f}s)".format(
                        run_count, total_runs, method, noise_std, seed, metrics['mae'], elapsed))
                except Exception as e:
                    elapsed = time.time() - t0
                    log("  [{}/{}] {} ERROR ({:.0f}s): {}".format(
                        run_count, total_runs, method, elapsed, e))
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()

            if seed_maes:
                noise_results[method] = {
                    "mae_mean": float(np.mean(seed_maes)),
                    "mae_std": float(np.std(seed_maes)),
                    "per_seed": seed_maes,
                }
                log("  {} sigma={:.1f}: MAE={:.4f}±{:.4f}".format(
                    method, noise_std, np.mean(seed_maes), np.std(seed_maes)))

        results["results"][str(noise_std)] = noise_results

        # Save after each noise level
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final analysis
    log("\n" + "=" * 60)
    log("Summary")
    log("=" * 60)

    log("\n{:<12} ".format("Method") + "  ".join(
        "{:>10}".format("s={:.1f}".format(n)) for n in NOISE_LEVELS))
    for method in METHODS:
        vals = []
        for ns in NOISE_LEVELS:
            r = results["results"].get(str(ns), {}).get(method, {})
            vals.append(r.get("mae_mean", float("nan")))
        log("{:<12} ".format(method) + "  ".join("{:>10.4f}".format(v) for v in vals))

    # Compute degradation rates
    log("\nDegradation (MAE increase from sigma=0):")
    baseline_maes = {}
    for method in METHODS:
        base = results["results"].get("0.0", {}).get(method, {}).get("mae_mean")
        if base:
            baseline_maes[method] = base
            log("  {} baseline: {:.4f}".format(method, base))
            for ns in NOISE_LEVELS[1:]:
                noisy_mae = results["results"].get(str(ns), {}).get(method, {}).get("mae_mean")
                if noisy_mae:
                    degradation = (noisy_mae - base) / base * 100
                    log("    sigma={:.1f}: {:.4f} (+{:.1f}%)".format(ns, noisy_mae, degradation))

    # Store degradation analysis
    degradation_analysis = {}
    for method in METHODS:
        base = baseline_maes.get(method)
        if base:
            degradations = []
            for ns in NOISE_LEVELS[1:]:
                noisy_mae = results["results"].get(str(ns), {}).get(method, {}).get("mae_mean")
                if noisy_mae:
                    degradations.append((ns, (noisy_mae - base) / base * 100))
            degradation_analysis[method] = {
                "baseline_mae": base,
                "degradation_pct": degradations,
            }

    results["degradation_analysis"] = degradation_analysis
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    total_time = time.time() - t_start
    log("\nCompleted: {} ({:.1f}h)".format(time.strftime('%Y-%m-%d %H:%M:%S'), total_time / 3600))
    log("Results saved to {}".format(RESULTS_FILE))
    log("=" * 60)


if __name__ == "__main__":
    main()
