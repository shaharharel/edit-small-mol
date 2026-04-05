#!/usr/bin/env python3
"""
Smart iteration loop for edit-aware FiLM architecture search.

Phase A (single seed, assay_within):
  1. FiLMDelta baseline
  2. DRFP-FiLM
  3. DualStream-FiLM
  4. Fragment-Anchored FiLM
  → Pick top 2

Phase B (single seed, assay_within, top 2 + MultiModal):
  5. MultiModal-FiLM
  → Pick winner

Phase C (3 seeds, assay_within + strict_scaffold):
  6. Winner with 3 seeds on 2 splits
  → Final result

Usage:
    conda run -n quris python -u experiments/run_edit_iteration.py
    conda run -n quris python -u experiments/run_edit_iteration.py --phase A
    conda run -n quris python -u experiments/run_edit_iteration.py --phase B
    conda run -n quris python -u experiments/run_edit_iteration.py --phase C
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

# Import everything from the canonical evaluation script
from experiments.run_paper_evaluation import (
    ARCHITECTURES, BATCH_SIZE, DEVICE, LR, MAX_EPOCHS, PATIENCE, SEEDS,
    RESULTS_DIR, CACHE_DIR,
    load_data, split_data, compute_embeddings, compute_drfp_cache,
    compute_frag_delta_cache, run_single_experiment, aggregate_seeds,
)

RESULTS_FILE = RESULTS_DIR / "edit_iteration_results.json"


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  → Saved results to {RESULTS_FILE.name}")


def run_phase_a(df, embedder="chemprop-dmpnn"):
    """Phase A: Compare baseline FiLMDelta vs 3 new edit-aware variants (1 seed)."""
    print("\n" + "=" * 70)
    print("PHASE A: Edit-Aware Architecture Screening (1 seed, assay_within)")
    print("=" * 70)

    results = load_results()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, embedder)

    seed = SEEDS[0]
    train_df, val_df, test_df = split_data(df, "assay_within", seed)
    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Precompute caches
    print("\n  Precomputing edit representations...")
    compute_drfp_cache(df)
    compute_frag_delta_cache(df)

    archs = ["FiLMDelta", "DrfpFiLM", "DualStreamFiLM", "FragAnchoredFiLM"]
    phase_a = results.get("phase_a", {})

    for arch_name in archs:
        if arch_name in phase_a:
            print(f"\n  {arch_name}: already computed (MAE={phase_a[arch_name]['mae']:.4f})")
            continue

        print(f"\n--- {arch_name} ---")
        t0 = time.time()
        try:
            metrics, _ = run_single_experiment(
                train_df, val_df, test_df, emb_dict, emb_dim, arch_name, seed)
            elapsed = time.time() - t0
            phase_a[arch_name] = metrics
            phase_a[arch_name]["elapsed_s"] = elapsed
            print(f"  MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                  f"R²={metrics['r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        results["phase_a"] = phase_a
        save_results(results)
        gc.collect()

    # Rank and pick top 2
    ranked = sorted(
        [(k, v["mae"]) for k, v in phase_a.items() if "mae" in v],
        key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("PHASE A RESULTS:")
    print(f"  {'Architecture':<25} {'MAE':>10} {'Spearman':>10} {'R²':>10}")
    print(f"  {'-' * 55}")
    for name, mae in ranked:
        m = phase_a[name]
        print(f"  {name:<25} {m['mae']:>10.4f} {m.get('spearman_r', 0):>10.4f} "
              f"{m.get('r2', 0):>10.4f}")

    top2 = [name for name, _ in ranked[:2]]
    print(f"\n  → Top 2: {top2}")
    results["phase_a_top2"] = top2
    save_results(results)

    del emb_dict
    gc.collect()
    return results


def run_phase_b(df, embedder="chemprop-dmpnn"):
    """Phase B: Top 2 from Phase A + MultiModal (1 seed)."""
    print("\n" + "=" * 70)
    print("PHASE B: MultiModal + Top 2 (1 seed, assay_within)")
    print("=" * 70)

    results = load_results()
    top2 = results.get("phase_a_top2")
    if not top2:
        print("  ERROR: Phase A not complete. Run --phase A first.")
        return results

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, embedder)

    seed = SEEDS[0]
    train_df, val_df, test_df = split_data(df, "assay_within", seed)

    # Precompute caches
    compute_drfp_cache(df)
    compute_frag_delta_cache(df)

    archs = list(set(top2 + ["MultiModalFiLM"]))
    phase_b = results.get("phase_b", {})

    for arch_name in archs:
        if arch_name in phase_b:
            print(f"\n  {arch_name}: already computed (MAE={phase_b[arch_name]['mae']:.4f})")
            continue

        # Copy Phase A result if available
        if arch_name in results.get("phase_a", {}) and arch_name != "MultiModalFiLM":
            phase_b[arch_name] = results["phase_a"][arch_name]
            print(f"\n  {arch_name}: reusing Phase A result (MAE={phase_b[arch_name]['mae']:.4f})")
            continue

        print(f"\n--- {arch_name} ---")
        t0 = time.time()
        try:
            metrics, _ = run_single_experiment(
                train_df, val_df, test_df, emb_dict, emb_dim, arch_name, seed)
            elapsed = time.time() - t0
            phase_b[arch_name] = metrics
            phase_b[arch_name]["elapsed_s"] = elapsed
            print(f"  MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                  f"R²={metrics['r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        results["phase_b"] = phase_b
        save_results(results)
        gc.collect()

    # Pick winner
    ranked = sorted(
        [(k, v["mae"]) for k, v in phase_b.items() if "mae" in v],
        key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("PHASE B RESULTS:")
    for name, mae in ranked:
        m = phase_b[name]
        print(f"  {name:<25} MAE={m['mae']:.4f} Spearman={m.get('spearman_r', 0):.4f}")

    winner = ranked[0][0]
    print(f"\n  → Winner: {winner}")
    results["phase_b_winner"] = winner
    save_results(results)

    del emb_dict
    gc.collect()
    return results


def run_phase_c(df, embedder="chemprop-dmpnn"):
    """Phase C: Winner with 3 seeds on assay_within + strict_scaffold."""
    print("\n" + "=" * 70)
    print("PHASE C: Final Evaluation (3 seeds × 2 splits)")
    print("=" * 70)

    results = load_results()
    winner = results.get("phase_b_winner")
    if not winner:
        print("  ERROR: Phase B not complete. Run --phase B first.")
        return results

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, embedder)

    # Precompute caches
    compute_drfp_cache(df)
    compute_frag_delta_cache(df)

    splits = ["assay_within", "strict_scaffold"]
    methods = ["FiLMDelta", "Subtraction", winner]
    methods = list(dict.fromkeys(methods))  # deduplicate

    phase_c = results.get("phase_c", {})

    for split_name in splits:
        for method in methods:
            key = f"{split_name}__{method}"
            if key in phase_c:
                agg = phase_c[key].get("aggregated", {})
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
                phase_c[key] = {
                    "aggregated": aggregate_seeds(seed_runs),
                    "per_seed": seed_runs,
                }

            results["phase_c"] = phase_c
            save_results(results)
            gc.collect()

    # Print final summary
    print("\n" + "=" * 70)
    print("PHASE C FINAL RESULTS:")
    print("=" * 70)
    for split_name in splits:
        print(f"\n  Split: {split_name}")
        print(f"  {'Method':<25} {'MAE':>15} {'Spearman':>15}")
        print(f"  {'-' * 55}")
        for method in methods:
            key = f"{split_name}__{method}"
            if key in phase_c:
                a = phase_c[key]["aggregated"]
                mae_s = f"{a.get('mae_mean',0):.4f}±{a.get('mae_std',0):.4f}"
                spr_s = f"{a.get('spearman_r_mean',0):.4f}±{a.get('spearman_r_std',0):.4f}"
                print(f"  {method:<25} {mae_s:>15} {spr_s:>15}")

    results["completed"] = datetime.now().isoformat()
    save_results(results)

    del emb_dict
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Edit-aware FiLM architecture iteration")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--embedder", default="chemprop-dmpnn")
    args = parser.parse_args()

    df = load_data()

    if args.phase in ("A", "all"):
        run_phase_a(df, args.embedder)
    if args.phase in ("B", "all"):
        run_phase_b(df, args.embedder)
    if args.phase in ("C", "all"):
        run_phase_c(df, args.embedder)


if __name__ == "__main__":
    main()
