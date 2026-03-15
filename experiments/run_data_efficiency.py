#!/usr/bin/env python3
"""
Data efficiency analysis: learning curves for edit effect methods.

Compares FiLMDelta, EditDiff, and Subtraction baseline at different
training data fractions (1%, 5%, 10%, 25%, 50%, 100%) to show where
the edit framework provides the most advantage.

Uses within-assay split on shared pairs dataset with ChemProp D-MPNN embeddings.

Usage:
    conda run -n quris python -u experiments/run_data_efficiency.py
"""

import sys
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

# Import from main evaluation script
from run_paper_evaluation import (
    load_data, compute_embeddings, split_data, run_single_experiment,
    ARCHITECTURES, SEEDS, RESULTS_DIR, compute_per_target_metrics,
)

EMBEDDER = "chemprop-dmpnn"
METHODS = ["Subtraction", "FiLMDelta", "EditDiff"]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
SPLIT = "assay_within"
RESULTS_FILE = RESULTS_DIR / "data_efficiency.json"


def run_data_efficiency():
    print(f"{'='*60}")
    print(f"Data Efficiency Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Fractions: {FRACTIONS}")
    print(f"{'='*60}")

    # Load data
    df = load_data()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, EMBEDDER)

    results = {}

    for frac in FRACTIONS:
        frac_key = f"frac_{frac}"
        print(f"\n{'='*60}")
        print(f"Training fraction: {frac:.0%}")
        print(f"{'='*60}")

        frac_results = {}

        for method in METHODS:
            seed_runs = []
            for seed in SEEDS:
                print(f"  {method}, seed {seed}, frac {frac:.0%}...", end=" ", flush=True)
                try:
                    train_df, val_df, test_df = split_data(df, SPLIT, seed)

                    # Subsample training data
                    if frac < 1.0:
                        n_train = max(50, int(len(train_df) * frac))
                        train_df = train_df.sample(n=n_train, random_state=seed).reset_index(drop=True)
                        # Also subsample val proportionally
                        n_val = max(20, int(len(val_df) * frac))
                        val_df = val_df.sample(n=min(n_val, len(val_df)), random_state=seed).reset_index(drop=True)

                    if len(train_df) < 50 or len(test_df) < 20:
                        print(f"too few samples")
                        continue

                    metrics, _ = run_single_experiment(
                        train_df, val_df, test_df, emb_dict, emb_dim, method, seed
                    )
                    seed_runs.append(metrics)
                    print(f"MAE={metrics['mae']:.4f} (n_train={len(train_df):,})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    gc.collect()

            if seed_runs:
                # Aggregate
                agg = {}
                for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_r"]:
                    vals = [r[metric] for r in seed_runs if metric in r]
                    if vals:
                        agg[f"{metric}_mean"] = float(np.mean(vals))
                        agg[f"{metric}_std"] = float(np.std(vals))
                agg["n_seeds"] = len(seed_runs)
                agg["n_train"] = int(seed_runs[0].get("n", 0))

                frac_results[method] = {
                    "aggregated": agg,
                    "per_seed": seed_runs,
                }

        results[frac_key] = {
            "fraction": frac,
            "methods": frac_results,
        }

        # Save incrementally
        save_results(results)

    # Print summary
    print(f"\n{'='*60}")
    print("DATA EFFICIENCY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Fraction':<10}", end="")
    for m in METHODS:
        print(f"  {m:>15}", end="")
    print()

    for frac in FRACTIONS:
        frac_key = f"frac_{frac}"
        if frac_key not in results:
            continue
        print(f"{frac:<10.0%}", end="")
        for m in METHODS:
            mae = results[frac_key]["methods"].get(m, {}).get("aggregated", {}).get("mae_mean")
            if mae:
                print(f"  {mae:>15.4f}", end="")
            else:
                print(f"  {'—':>15}", end="")
        print()


def save_results(results):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_FILE.name}]")


if __name__ == "__main__":
    run_data_efficiency()
