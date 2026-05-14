#!/usr/bin/env python3
"""
ChemBERTa-FiLMDelta vs Morgan-FP-FiLMDelta head-to-head (within-assay).

Trains :class:`ChemBERTaFiLMDeltaMLP` on the canonical shared-pairs
within-assay benchmark with the same protocol as Phase 2 of
``run_paper_evaluation.py`` (assay_within split, 3 seeds, identical
hyperparameters). Outputs a CSV that can be compared directly against
the Phase 2 FiLMDelta numbers in ``all_results.json``.

Usage:
    conda run --no-capture-output -n quris python \
        experiments/run_chemberta_filmdelta_phase1.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU — MPS crashes with ChemBERTa after prolonged use.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import gc
import json
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.backends.mps.is_available = lambda: False

from torch.utils.data import DataLoader, TensorDataset

# Reuse the canonical evaluation harness — keep protocols byte-identical.
from experiments.run_paper_evaluation import (  # noqa: E402
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
    compute_embeddings,
    compute_metrics,
    compute_per_target_metrics,
    get_pair_tensors,
    load_data,
    predict_multi_input,
    split_data,
    train_model_multi_input,
)
from src.models.predictors.chemberta_film_delta_predictor import (  # noqa: E402
    ChemBERTaFiLMDeltaMLP,
)

EMBEDDER = "chemberta2-mtr"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
CSV_OUT = RESULTS_DIR / "chemberta_filmdelta_phase1.csv"
JSON_OUT = RESULTS_DIR / "chemberta_filmdelta_phase1.json"

# Morgan-FP-FiLMDelta numbers from results/paper_evaluation/all_results.json
# (Phase 2, assay_within, 3 seeds, ChemProp/Morgan FP 2048d).
MORGAN_FP_FILMDELTA_BASELINE = {
    "mae_mean": 0.6162,
    "mae_std": 0.0215,
    "spearman_r_mean": 0.3997,
    "pearson_r_mean": 0.4508,
    "r2_mean": 0.1962,
}


def train_and_predict_chemberta_film_delta(
    train_df, val_df, test_df, emb_dict, emb_dim, seed
):
    """ChemBERTa-FiLMDelta: same FiLMDelta head, ChemBERTa-2-MTR embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return emb_a, emb_b, delta

    train_a, train_b, train_y = make_datasets(train_df)
    val_a, val_b, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ChemBERTaFiLMDeltaMLP(input_dim=emb_dim)

    def forward_fn(m, a, b):
        return m(a, b)

    model = train_model_multi_input(
        model, train_loader, val_loader, forward_fn,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
    )

    test_a, test_b, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output CSV: {CSV_OUT}")
    print(f"Output JSON: {JSON_OUT}")

    df = load_data()

    # Cache ChemBERTa embeddings once (npz already exists per memory).
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    print(f"\nLoading {EMBEDDER} embeddings for {len(all_smiles):,} molecules...")
    emb_dict, emb_dim = compute_embeddings(all_smiles, EMBEDDER)
    print(f"  Embedding dim: {emb_dim}")

    seed_runs = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")
        t0 = time.time()
        train_df, val_df, test_df = split_data(df, "assay_within", seed)
        print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

        y_pred = train_and_predict_chemberta_film_delta(
            train_df, val_df, test_df, emb_dict, emb_dim, seed
        )
        y_true = test_df["delta"].values
        metrics = compute_metrics(y_true, y_pred)
        if "target_chembl_id" in test_df.columns:
            pt = compute_per_target_metrics(y_true, y_pred,
                                            test_df["target_chembl_id"].values)
            metrics["per_target_avg"] = {k: v for k, v in pt.items()
                                         if k != "per_target"}

        elapsed = time.time() - t0
        print(f"  MAE={metrics['mae']:.4f}  Spearman={metrics['spearman_r']:.4f}  "
              f"Pearson={metrics['pearson_r']:.4f}  R²={metrics['r2']:.4f}  "
              f"({elapsed/60:.1f} min)")
        metrics["seed"] = seed
        seed_runs.append(metrics)
        gc.collect()

    # Write CSV.
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "MAE", "Spearman", "Pearson", "R2"])
        for r in seed_runs:
            writer.writerow([
                "ChemBERTa-FiLMDelta",
                r["seed"],
                f"{r['mae']:.4f}",
                f"{r['spearman_r']:.4f}",
                f"{r['pearson_r']:.4f}",
                f"{r['r2']:.4f}",
            ])

    # Aggregate.
    agg = {}
    for k in ("mae", "spearman_r", "pearson_r", "r2"):
        vals = [r[k] for r in seed_runs]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    with JSON_OUT.open("w") as f:
        json.dump({
            "method": "ChemBERTa-FiLMDelta",
            "embedder": EMBEDDER,
            "emb_dim": emb_dim,
            "split": "assay_within",
            "seeds": SEEDS,
            "per_seed": seed_runs,
            "aggregated": agg,
            "morgan_fp_filmdelta_baseline": MORGAN_FP_FILMDELTA_BASELINE,
        }, f, indent=2, default=float)

    # 5-line summary.
    base_mae = MORGAN_FP_FILMDELTA_BASELINE["mae_mean"]
    delta_mae = agg["mae_mean"] - base_mae
    pct = 100.0 * delta_mae / base_mae
    print("\n" + "=" * 70)
    print("SUMMARY: ChemBERTa-FiLMDelta vs Morgan-FP-FiLMDelta (assay_within)")
    print("=" * 70)
    print(f"  ChemBERTa-FiLMDelta  : MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}  "
          f"Spearman={agg['spearman_r_mean']:.4f}  R²={agg['r2_mean']:.4f}")
    print(f"  Morgan-FP-FiLMDelta  : MAE={base_mae:.4f}±"
          f"{MORGAN_FP_FILMDELTA_BASELINE['mae_std']:.4f}  "
          f"Spearman={MORGAN_FP_FILMDELTA_BASELINE['spearman_r_mean']:.4f}  "
          f"R²={MORGAN_FP_FILMDELTA_BASELINE['r2_mean']:.4f}")
    print(f"  Δ MAE                : {delta_mae:+.4f} ({pct:+.1f}% vs Morgan baseline; "
          f"negative=ChemBERTa wins)")
    print(f"  CSV  : {CSV_OUT}")
    print(f"  JSON : {JSON_OUT}")


if __name__ == "__main__":
    main()
