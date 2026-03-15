#!/usr/bin/env python3
"""
Run Phase 3 evaluation on new splits: strict_scaffold and pair_random.

Usage:
    python -u experiments/run_new_splits_phase3.py
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from run_paper_evaluation import (
    load_data, split_data, run_single_experiment, compute_embeddings,
    SEEDS, RESULTS_DIR,
)


def aggregate_seeds(seed_runs):
    agg = {}
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_r"]:
        vals = [r[metric] for r in seed_runs if metric in r]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
    agg["n_seeds"] = len(seed_runs)
    return agg


def main():
    print("=" * 60)
    print("Phase 3: New Splits (strict_scaffold, pair_random)")
    print("=" * 60)

    df = load_data()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")
    print(f"  Dataset: {len(df):,} pairs, {len(all_smiles):,} molecules, dim={emb_dim}")

    splits = [
        ("strict_scaffold", "Strict Scaffold"),
        ("pair_random", "Pair-Aware Random"),
    ]
    methods = ["FiLMDelta", "EditDiff", "DeepDelta", "Subtraction"]

    results_file = RESULTS_DIR / "all_results.json"

    for split_idx, (split_name, split_label) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Split {split_idx+1}/{len(splits)}: {split_label}")
        print(f"{'='*60}")

        split_results = {"label": split_label, "methods": {}}

        for method in methods:
            seed_runs = []
            for seed_idx, seed in enumerate(SEEDS):
                print(f"  {method}, seed {seed}...", end=" ", flush=True)
                try:
                    train_df, val_df, test_df = split_data(df, split_name, seed)
                    metrics, _ = run_single_experiment(
                        train_df, val_df, test_df, emb_dict, emb_dim, method, seed
                    )
                    seed_runs.append(metrics)
                    print(f"MAE={metrics['mae']:.4f}, Sp={metrics['spearman_r']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    gc.collect()

            if seed_runs:
                split_results["methods"][method] = {
                    "aggregated": aggregate_seeds(seed_runs),
                    "per_seed": seed_runs,
                }

        # Save incrementally
        try:
            with open(results_file) as f:
                all_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_results = {}

        if "phase3" not in all_results:
            all_results["phase3"] = {}
        all_results["phase3"][split_name] = split_results

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved {split_name} results to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    try:
        with open(results_file) as f:
            all_results = json.load(f)
        for split_name, _ in splits:
            if split_name in all_results.get("phase3", {}):
                methods_data = all_results["phase3"][split_name].get("methods", {})
                print(f"\n{split_name}:")
                for m, data in methods_data.items():
                    agg = data["aggregated"]
                    print(f"  {m:15s} MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}  "
                          f"Sp={agg['spearman_r_mean']:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
