#!/usr/bin/env python3
"""
Next-generation architecture experiments based on ML architecture review.

Tests two novel ideas that address the ACTUAL bottleneck (label noise and
directional bias in training data) rather than adding redundant edit signals:

1. TargetCondFiLM: FiLMDelta + target identity conditioning (target_chembl_id
   as a learned embedding). Different targets have different SARs, so the same
   edit should produce different conditioning per target.

2. FiLMDelta+Aug: FiLMDelta with antisymmetric data augmentation. Training data
   has 0% reversed pairs — adding (mol_b, mol_a, -delta) doubles training data
   and forces the delta conditioning pathway to be antisymmetric.

Usage:
    conda run -n quris python -u experiments/run_next_gen_architectures.py
    conda run -n quris python -u experiments/run_next_gen_architectures.py --phase screen
    conda run -n quris python -u experiments/run_next_gen_architectures.py --phase full
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import time
from datetime import datetime

import numpy as np

from experiments.run_paper_evaluation import (
    ARCHITECTURES, SEEDS, RESULTS_DIR,
    load_data, split_data, compute_embeddings,
    run_single_experiment, aggregate_seeds,
)

RESULTS_FILE = RESULTS_DIR / "next_gen_results.json"


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  → Saved to {RESULTS_FILE.name}")


def run_screen(df, embedder="chemprop-dmpnn"):
    """Screen: 1 seed, assay_within, compare FiLMDelta vs TargetCondFiLM vs FiLMDelta+Aug."""
    print("\n" + "=" * 70)
    print("SCREEN: Next-Gen Architectures (1 seed, assay_within)")
    print("=" * 70)

    results = load_results()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, embedder)

    seed = SEEDS[0]
    train_df, val_df, test_df = split_data(df, "assay_within", seed)
    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    methods = ["FiLMDelta", "TargetCondFiLM", "FiLMDelta+Aug"]
    screen = results.get("screen", {})

    for method in methods:
        if method in screen:
            print(f"\n  {method}: already done (MAE={screen[method]['mae']:.4f})")
            continue

        print(f"\n--- {method} ---")
        t0 = time.time()
        try:
            metrics, _ = run_single_experiment(
                train_df, val_df, test_df, emb_dict, emb_dim, method, seed)
            elapsed = time.time() - t0
            screen[method] = metrics
            screen[method]["elapsed_s"] = elapsed
            print(f"  MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                  f"R²={metrics['r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        results["screen"] = screen
        save_results(results)
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("SCREEN RESULTS:")
    print(f"  {'Method':<25} {'MAE':>10} {'Spearman':>10} {'R²':>10}")
    print(f"  {'-' * 55}")
    ranked = sorted(
        [(k, v["mae"]) for k, v in screen.items() if "mae" in v],
        key=lambda x: x[1])
    for name, mae in ranked:
        m = screen[name]
        print(f"  {name:<25} {m['mae']:>10.4f} {m.get('spearman_r', 0):>10.4f} "
              f"{m.get('r2', 0):>10.4f}")

    results["screen_completed"] = datetime.now().isoformat()
    save_results(results)
    del emb_dict
    gc.collect()
    return results


def run_full(df, embedder="chemprop-dmpnn"):
    """Full evaluation: 3 seeds, assay_within + strict_scaffold."""
    print("\n" + "=" * 70)
    print("FULL: Next-Gen Architectures (3 seeds × 2 splits)")
    print("=" * 70)

    results = load_results()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, embedder)

    splits = ["assay_within", "strict_scaffold"]
    methods = ["FiLMDelta", "Subtraction", "TargetCondFiLM", "FiLMDelta+Aug"]

    full = results.get("full", {})

    for split_name in splits:
        for method in methods:
            key = f"{split_name}__{method}"
            if key in full:
                agg = full[key].get("aggregated", {})
                print(f"  {key}: already done (MAE={agg.get('mae_mean', '?')})")
                continue

            print(f"\n--- {method} on {split_name} ---")
            seed_runs = []
            for seed_idx, seed in enumerate(SEEDS):
                print(f"  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...", end=" ", flush=True)
                try:
                    train_df, val_df, test_df = split_data(df, split_name, seed)
                    metrics, _ = run_single_experiment(
                        train_df, val_df, test_df, emb_dict, emb_dim, method, seed)
                    seed_runs.append(metrics)
                    print(f"MAE={metrics['mae']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")

            if seed_runs:
                full[key] = {
                    "aggregated": aggregate_seeds(seed_runs),
                    "per_seed": seed_runs,
                }

            results["full"] = full
            save_results(results)
            gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("FULL RESULTS:")
    print("=" * 70)
    for split_name in splits:
        print(f"\n  Split: {split_name}")
        print(f"  {'Method':<25} {'MAE':>15} {'Spearman':>15}")
        print(f"  {'-' * 55}")
        for method in methods:
            key = f"{split_name}__{method}"
            if key in full:
                a = full[key]["aggregated"]
                mae_s = f"{a.get('mae_mean',0):.4f}±{a.get('mae_std',0):.4f}"
                spr_s = f"{a.get('spearman_r_mean',0):.4f}±{a.get('spearman_r_std',0):.4f}"
                print(f"  {method:<25} {mae_s:>15} {spr_s:>15}")

    results["completed"] = datetime.now().isoformat()
    save_results(results)
    del emb_dict
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description="Next-gen architecture experiments")
    parser.add_argument("--phase", choices=["screen", "full", "all"], default="all")
    parser.add_argument("--embedder", default="chemprop-dmpnn")
    args = parser.parse_args()

    df = load_data()

    if args.phase in ("screen", "all"):
        run_screen(df, args.embedder)
    if args.phase in ("full", "all"):
        run_full(df, args.embedder)


if __name__ == "__main__":
    main()
