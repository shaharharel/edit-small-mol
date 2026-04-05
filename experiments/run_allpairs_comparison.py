#!/usr/bin/env python3
"""
All-Pairs vs MMP Comparison Experiment.

Compares architectures trained on MMP-only vs all-pairs (non-MMP) data:

Conditions:
  A) MMP-trained, MMP-tested (reuse existing Phase 2 results)
  B) All-pairs-trained, MMP-tested
  C) All-pairs-trained, all-pairs-tested
  D) MMP-trained, all-pairs-tested
  E) Data-matched: subsample all-pairs to same N as MMP, train, test on MMP
  F) Curriculum: pretrain all-pairs → finetune MMP, test on MMP

Architectures: FiLMDelta, Subtraction, DeepDelta
(DualStreamFiLM excluded — DRFP can't encode non-MMP pairs)

Usage:
    conda run -n quris python -u experiments/run_allpairs_comparison.py
    conda run -n quris python -u experiments/run_allpairs_comparison.py --condition B
    conda run -n quris python -u experiments/run_allpairs_comparison.py --condition F
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_paper_evaluation import (
    SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, DROPOUT, DEVICE,
    RESULTS_DIR, CACHE_DIR,
    DeltaMLP, AbsoluteMLP,
    compute_embeddings, compute_metrics, compute_per_target_metrics,
    get_pair_tensors, split_data, aggregate_seeds,
    train_model, train_model_multi_input, predict, predict_multi_input,
)

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"

MMP_FILE = DATA_DIR / "shared_pairs_deduped.csv"
ALLPAIRS_FILE = DATA_DIR / "all_pairs_within_assay.csv"
RESULTS_FILE = RESULTS_DIR / "allpairs_comparison_results.json"

METHODS = ["FiLMDelta", "Subtraction", "DeepDelta"]
MAX_ALLPAIRS = 1_700_000  # Cap all-pairs to match MMP dataset size (memory + time)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_mmp_data():
    """Load MMP within-assay pairs."""
    print("Loading MMP data...")
    df = pd.read_csv(MMP_FILE)
    # Filter to real MMPs and within-assay
    if "mol_a_id" in df.columns and "mol_b_id" in df.columns:
        df = df[df["mol_a_id"] != df["mol_b_id"]].copy()
    df = df[df["is_within_assay"] == True].copy()
    print(f"  {len(df):,} MMP within-assay pairs, {df['target_chembl_id'].nunique()} targets")
    return df


def load_allpairs_data(max_pairs=MAX_ALLPAIRS):
    """Load all within-assay pairs, with stratified subsampling if needed."""
    print("Loading all-pairs data...")
    if not ALLPAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{ALLPAIRS_FILE.name} not found. Run: python scripts/generate_all_pairs.py")
    df = pd.read_csv(ALLPAIRS_FILE)
    print(f"  {len(df):,} all-pairs, {df['target_chembl_id'].nunique()} targets")

    if max_pairs and len(df) > max_pairs:
        print(f"  Subsampling to {max_pairs:,} pairs (stratified by assay)...")
        # Stratified sample: proportional to assay pair count
        frac = max_pairs / len(df)
        sampled = df.groupby("assay_id_a", group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42) if len(x) * frac >= 1
            else x.sample(n=max(1, int(len(x) * frac)), random_state=42)
        )
        # If we overshot/undershot, adjust
        if len(sampled) > max_pairs:
            sampled = sampled.sample(n=max_pairs, random_state=42)
        df = sampled.reset_index(drop=True)
        n_mmp = df["is_mmp"].sum() if "is_mmp" in df.columns else 0
        print(f"  Subsampled: {len(df):,} pairs ({n_mmp:,} MMP, {len(df)-n_mmp:,} non-MMP)")
        print(f"  Assays retained: {df['assay_id_a'].nunique()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Training Functions (reuse from run_paper_evaluation via import)
# ═══════════════════════════════════════════════════════════════════════════

def train_and_predict_subtraction(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Subtraction baseline: train F(mol)→property, predict F(B)-F(A)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mol_values = dict(zip(train_df["mol_a"], train_df["value_a"]))
    mol_values.update(dict(zip(train_df["mol_b"], train_df["value_b"])))

    zero = np.zeros(emb_dim)
    smiles_list = list(mol_values.keys())
    y_vals = np.array([mol_values[s] for s in smiles_list], dtype=np.float32)
    X = np.array([emb_dict.get(s, zero) for s in smiles_list], dtype=np.float32)

    val_mol_values = dict(zip(val_df["mol_a"], val_df["value_a"]))
    val_mol_values.update(dict(zip(val_df["mol_b"], val_df["value_b"])))
    val_smiles = list(val_mol_values.keys())
    val_y = np.array([val_mol_values[s] for s in val_smiles], dtype=np.float32)
    val_X = np.array([emb_dict.get(s, zero) for s in val_smiles], dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_vals).float()),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float()),
        batch_size=BATCH_SIZE, shuffle=False)

    model = AbsoluteMLP(emb_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict(model, emb_b) - predict(model, emb_a), model


def train_and_predict_deepdelta(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """DeepDelta: MLP on [emb_a, emb_b] → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_input(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return torch.cat([emb_a, emb_b], dim=-1), delta

    train_x, train_y = make_input(train_df)
    val_x, val_y = make_input(val_df)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False)

    model = DeltaMLP(emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict(model, torch.cat([emb_a, emb_b], dim=-1)), model


def train_and_predict_film_delta(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """FiLM-conditioned delta prediction on (emb_a, emb_b) → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return emb_a, emb_b, delta

    train_a, train_b, train_y = make_datasets(train_df)
    val_a, val_b, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_y),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False)

    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])

    def forward_fn(m, a, b):
        return m(a, b)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, _ = make_datasets(test_df)
    preds = predict_multi_input(model, forward_fn, test_a, test_b)
    return preds, model


def train_and_predict_curriculum(train_allpairs_df, val_allpairs_df,
                                 train_mmp_df, val_mmp_df,
                                 test_df, emb_dict, emb_dim, seed, method):
    """Curriculum: pretrain on all-pairs, finetune on MMP."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if method == "FiLMDelta":
        from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

        def make_datasets(df):
            emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
            return emb_a, emb_b, delta

        def forward_fn(m, a, b):
            return m(a, b)

        model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])

        # Stage 1: Pretrain on all-pairs
        print("    Stage 1: pretrain on all-pairs...", end=" ", flush=True)
        ap_a, ap_b, ap_y = make_datasets(train_allpairs_df)
        vap_a, vap_b, vap_y = make_datasets(val_allpairs_df)
        ap_train_loader = DataLoader(TensorDataset(ap_a, ap_b, ap_y), batch_size=BATCH_SIZE, shuffle=True)
        ap_val_loader = DataLoader(TensorDataset(vap_a, vap_b, vap_y), batch_size=BATCH_SIZE, shuffle=False)
        model = train_model_multi_input(model, ap_train_loader, ap_val_loader, forward_fn)
        print("done")

        # Stage 2: Finetune on MMP
        print("    Stage 2: finetune on MMP...", end=" ", flush=True)
        m_a, m_b, m_y = make_datasets(train_mmp_df)
        vm_a, vm_b, vm_y = make_datasets(val_mmp_df)
        m_train_loader = DataLoader(TensorDataset(m_a, m_b, m_y), batch_size=BATCH_SIZE, shuffle=True)
        m_val_loader = DataLoader(TensorDataset(vm_a, vm_b, vm_y), batch_size=BATCH_SIZE, shuffle=False)
        model = train_model_multi_input(model, m_train_loader, m_val_loader, forward_fn,
                                        max_epochs=MAX_EPOCHS // 3, lr=LR / 10)
        print("done")

        test_a, test_b, _ = make_datasets(test_df)
        return predict_multi_input(model, forward_fn, test_a, test_b)

    elif method == "DeepDelta":
        def make_input(df):
            emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
            return torch.cat([emb_a, emb_b], dim=-1), delta

        model = DeltaMLP(emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)

        # Stage 1
        print("    Stage 1: pretrain on all-pairs...", end=" ", flush=True)
        ap_x, ap_y = make_input(train_allpairs_df)
        vap_x, vap_y = make_input(val_allpairs_df)
        model = train_model(model,
                            DataLoader(TensorDataset(ap_x, ap_y), batch_size=BATCH_SIZE, shuffle=True),
                            DataLoader(TensorDataset(vap_x, vap_y), batch_size=BATCH_SIZE, shuffle=False))
        print("done")

        # Stage 2
        print("    Stage 2: finetune on MMP...", end=" ", flush=True)
        m_x, m_y = make_input(train_mmp_df)
        vm_x, vm_y = make_input(val_mmp_df)
        model = train_model(model,
                            DataLoader(TensorDataset(m_x, m_y), batch_size=BATCH_SIZE, shuffle=True),
                            DataLoader(TensorDataset(vm_x, vm_y), batch_size=BATCH_SIZE, shuffle=False),
                            max_epochs=MAX_EPOCHS // 3, lr=LR / 10)
        print("done")

        emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict(model, torch.cat([emb_a, emb_b], dim=-1))

    elif method == "Subtraction":
        model = AbsoluteMLP(emb_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)

        def make_mol_data(df):
            mv = dict(zip(df["mol_a"], df["value_a"]))
            mv.update(dict(zip(df["mol_b"], df["value_b"])))
            zero = np.zeros(emb_dim)
            sl = list(mv.keys())
            y = np.array([mv[s] for s in sl], dtype=np.float32)
            X = np.array([emb_dict.get(s, zero) for s in sl], dtype=np.float32)
            return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        # Stage 1
        print("    Stage 1: pretrain on all-pairs...", end=" ", flush=True)
        ap_X, ap_y = make_mol_data(train_allpairs_df)
        vap_X, vap_y = make_mol_data(val_allpairs_df)
        model = train_model(model,
                            DataLoader(TensorDataset(ap_X, ap_y), batch_size=BATCH_SIZE, shuffle=True),
                            DataLoader(TensorDataset(vap_X, vap_y), batch_size=BATCH_SIZE, shuffle=False))
        print("done")

        # Stage 2
        print("    Stage 2: finetune on MMP...", end=" ", flush=True)
        m_X, m_y = make_mol_data(train_mmp_df)
        vm_X, vm_y = make_mol_data(val_mmp_df)
        model = train_model(model,
                            DataLoader(TensorDataset(m_X, m_y), batch_size=BATCH_SIZE, shuffle=True),
                            DataLoader(TensorDataset(vm_X, vm_y), batch_size=BATCH_SIZE, shuffle=False),
                            max_epochs=MAX_EPOCHS // 3, lr=LR / 10)
        print("done")

        emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict(model, emb_b) - predict(model, emb_a)


TRAIN_FNS = {
    "FiLMDelta": train_and_predict_film_delta,
    "Subtraction": train_and_predict_subtraction,
    "DeepDelta": train_and_predict_deepdelta,
}


# ═══════════════════════════════════════════════════════════════════════════
# Results management
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Splitting with alignment
# ═══════════════════════════════════════════════════════════════════════════

def aligned_split(mmp_df, allpairs_df, seed):
    """Split both datasets using the same assay-level split.

    Uses the all-pairs dataset for splitting (larger), then filters MMP to match.
    """
    # Split all-pairs
    ap_train, ap_val, ap_test = split_data(allpairs_df, "assay_within", seed)
    test_assays = set(ap_test["assay_id_a"].unique())
    train_assays = set(ap_train["assay_id_a"].unique())
    val_assays = set(ap_val["assay_id_a"].unique())

    # Filter MMP to same assay split
    mmp_train = mmp_df[mmp_df["assay_id_a"].isin(train_assays)].copy()
    mmp_val = mmp_df[mmp_df["assay_id_a"].isin(val_assays)].copy()
    mmp_test = mmp_df[mmp_df["assay_id_a"].isin(test_assays)].copy()

    print(f"  Aligned split (seed={seed}):")
    print(f"    All-pairs: train={len(ap_train):,}, val={len(ap_val):,}, test={len(ap_test):,}")
    print(f"    MMP:       train={len(mmp_train):,}, val={len(mmp_val):,}, test={len(mmp_test):,}")
    print(f"    Test assays: {len(test_assays)}")

    return {
        "ap_train": ap_train, "ap_val": ap_val, "ap_test": ap_test,
        "mmp_train": mmp_train, "mmp_val": mmp_val, "mmp_test": mmp_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(condition, splits, emb_dict, emb_dim, seed, method, results):
    """Run a single condition-method-seed combination."""
    key = f"{condition}__{method}"

    if condition == "A":
        # MMP-trained, MMP-tested
        train_df = splits["mmp_train"]
        val_df = splits["mmp_val"]
        test_df = splits["mmp_test"]
    elif condition == "B":
        # All-pairs-trained, MMP-tested
        train_df = splits["ap_train"]
        val_df = splits["ap_val"]
        test_df = splits["mmp_test"]
    elif condition == "C":
        # All-pairs-trained, all-pairs-tested
        train_df = splits["ap_train"]
        val_df = splits["ap_val"]
        test_df = splits["ap_test"]
    elif condition == "D":
        # MMP-trained, all-pairs-tested
        train_df = splits["mmp_train"]
        val_df = splits["mmp_val"]
        test_df = splits["ap_test"]
    elif condition == "E":
        # Data-matched: subsample all-pairs to MMP size, MMP-tested
        n_mmp = len(splits["mmp_train"])
        ap_train = splits["ap_train"]
        if len(ap_train) > n_mmp:
            train_df = ap_train.sample(n=n_mmp, random_state=seed)
        else:
            train_df = ap_train
        n_mmp_val = len(splits["mmp_val"])
        ap_val = splits["ap_val"]
        val_df = ap_val.sample(n=min(n_mmp_val, len(ap_val)), random_state=seed)
        test_df = splits["mmp_test"]
    elif condition == "F":
        # Curriculum: pretrain all-pairs → finetune MMP
        y_true = splits["mmp_test"]["delta"].values
        t0 = time.time()
        y_pred = train_and_predict_curriculum(
            splits["ap_train"], splits["ap_val"],
            splits["mmp_train"], splits["mmp_val"],
            splits["mmp_test"], emb_dict, emb_dim, seed, method)
        elapsed = time.time() - t0
        metrics = compute_metrics(y_true, y_pred)
        metrics["elapsed_s"] = elapsed
        return metrics
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Standard train-predict for conditions A-E
    y_true = test_df["delta"].values
    if len(test_df) == 0:
        print(f"    WARNING: empty test set for {key}")
        return None

    t0 = time.time()
    train_fn = TRAIN_FNS[method]
    y_pred, _ = train_fn(train_df, val_df, test_df, emb_dict, emb_dim, seed)
    elapsed = time.time() - t0

    metrics = compute_metrics(y_true, y_pred)
    metrics["elapsed_s"] = elapsed
    metrics["train_size"] = len(train_df)
    metrics["test_size"] = len(test_df)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(conditions=None):
    """Run all-pairs vs MMP comparison experiment."""
    if conditions is None:
        conditions = ["A", "B", "C", "D", "E", "F"]

    print("=" * 70)
    print("ALL-PAIRS vs MMP COMPARISON")
    print(f"  Conditions: {conditions}")
    print(f"  Methods: {METHODS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    # Load data
    mmp_df = load_mmp_data()
    allpairs_df = load_allpairs_data()

    # Compute embeddings for all molecules
    all_smiles = list(set(
        mmp_df["mol_a"].tolist() + mmp_df["mol_b"].tolist() +
        allpairs_df["mol_a"].tolist() + allpairs_df["mol_b"].tolist()
    ))
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

    results = load_results()

    # Delta distribution analysis (do once)
    if "delta_stats" not in results:
        print("\n--- Delta Distribution Analysis ---")
        mmp_within = mmp_df[mmp_df["is_within_assay"] == True]
        mmp_deltas = mmp_within["delta"].values
        ap_deltas = allpairs_df["delta"].values
        results["delta_stats"] = {
            "mmp_mean": float(np.mean(np.abs(mmp_deltas))),
            "mmp_std": float(np.std(mmp_deltas)),
            "mmp_median_abs": float(np.median(np.abs(mmp_deltas))),
            "allpairs_mean": float(np.mean(np.abs(ap_deltas))),
            "allpairs_std": float(np.std(ap_deltas)),
            "allpairs_median_abs": float(np.median(np.abs(ap_deltas))),
            "mmp_n": len(mmp_deltas),
            "allpairs_n": len(ap_deltas),
        }
        print(f"  MMP: mean|δ|={results['delta_stats']['mmp_mean']:.3f}, "
              f"std={results['delta_stats']['mmp_std']:.3f}, n={len(mmp_deltas):,}")
        print(f"  All-pairs: mean|δ|={results['delta_stats']['allpairs_mean']:.3f}, "
              f"std={results['delta_stats']['allpairs_std']:.3f}, n={len(ap_deltas):,}")
        save_results(results)

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*70}")

        splits = aligned_split(mmp_df, allpairs_df, seed)

        for condition in conditions:
            for method in METHODS:
                key = f"{condition}__{method}"
                seed_key = f"seed_{seed}"

                # Check if already done
                if key in results and seed_key in results.get(key, {}):
                    m = results[key][seed_key]
                    print(f"  {key} seed={seed}: already done (MAE={m.get('mae','?')})")
                    continue

                print(f"\n--- Condition {condition}: {method} (seed={seed}) ---")
                try:
                    metrics = run_condition(condition, splits, emb_dict, emb_dim, seed, method, results)
                    if metrics:
                        if key not in results:
                            results[key] = {}
                        results[key][seed_key] = metrics
                        print(f"  MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()

                save_results(results)
                gc.collect()

    # Aggregate across seeds
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    for condition in conditions:
        print(f"\n  Condition {condition}:")
        print(f"  {'Method':<20} {'MAE':>15} {'Spearman':>15}")
        print(f"  {'-'*50}")
        for method in METHODS:
            key = f"{condition}__{method}"
            if key in results:
                seed_metrics = [v for k, v in results[key].items()
                                if k.startswith("seed_") and isinstance(v, dict)]
                if seed_metrics:
                    agg = aggregate_seeds(seed_metrics)
                    results[key]["aggregated"] = agg
                    mae_s = f"{agg.get('mae_mean',0):.4f}±{agg.get('mae_std',0):.4f}"
                    spr_s = f"{agg.get('spearman_r_mean',0):.4f}±{agg.get('spearman_r_std',0):.4f}"
                    print(f"  {method:<20} {mae_s:>15} {spr_s:>15}")

    results["completed"] = datetime.now().isoformat()
    save_results(results)


def main():
    parser = argparse.ArgumentParser(description="All-pairs vs MMP comparison")
    parser.add_argument("--condition", nargs="+", choices=["A", "B", "C", "D", "E", "F"],
                        default=None, help="Run specific conditions only")
    parser.add_argument("--embedder", default="chemprop-dmpnn")
    args = parser.parse_args()

    run_experiment(conditions=args.condition)


if __name__ == "__main__":
    main()
