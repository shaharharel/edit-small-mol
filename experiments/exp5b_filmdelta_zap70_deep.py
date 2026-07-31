#!/usr/bin/env python3
"""
EXP5b - ZAP70 deep comparison: FiLMDelta vs direct pIC50 predictor.

Extends EXP5 with:
  1) Three split protocols (pair-disjoint, mol-disjoint single-side,
     mol-disjoint both-sides) -- isolates memorization artifacts.
  2) Sample-size sweep N in {25, 50, 100, 200, 500, ALL}.
  3) Five random seeds per (split, N, method) combination.
  4) Both delta-MAE and absolute-pIC50-MAE (informational for direct).

Outputs
-------
results/paper_evaluation/exp5b_filmdelta_zap70_deep.json
results/paper_evaluation/exp5b_filmdelta_zap70_deep.png
results/paper_evaluation/exp5b_filmdelta_zap70_deep_summary.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False  # CPU only

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

# Reuse DirectMLP / DirectPropertyPredictor from EXP5
from experiments.exp5_zap70_smalldata import (  # noqa: E402
    DirectPropertyPredictor,
    smiles_to_morgan,
    stack_pair_fps,
    build_direct_training_set,
)

PAIRS_FILE_DEFAULT = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_FILE = RESULTS_DIR / "exp5b_filmdelta_zap70_deep.json"
PLOT_FILE = RESULTS_DIR / "exp5b_filmdelta_zap70_deep.png"
SUMMARY_MD = RESULTS_DIR / "exp5b_filmdelta_zap70_deep_summary.md"

N_BITS = 2048
RADIUS = 2
TRAIN_SIZES = [25, 50, 100, 200, 500]  # "all" appended at runtime
SEEDS = [0, 1, 2, 3, 4]
TEST_MOL_FRAC = 0.20  # fraction of unique molecules held out for mol-disjoint splits
TEST_PAIR_FRAC = 0.20  # fraction of pairs held out for pair-disjoint split

# Training hyperparameters (kept light for small data + CPU)
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-3
DROPOUT = 0.2
HIDDEN_DIMS = [512, 256, 128]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def delta_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"),
                "r2": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        pr = 0.0; sr = 0.0
    else:
        pr, _ = scipy_stats.pearsonr(y_pred, y_true)
        sr, _ = scipy_stats.spearmanr(y_pred, y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae": mae, "rmse": rmse,
        "pearson": float(pr) if not np.isnan(pr) else 0.0,
        "spearman": float(sr) if not np.isnan(sr) else 0.0,
        "r2": r2,
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------
def split_pair_disjoint(pairs: pd.DataFrame, seed: int, test_frac: float = TEST_PAIR_FRAC
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classic pair-disjoint: shuffle pairs, hold out test_frac."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pairs)))
    test_pairs = pairs.iloc[idx[:n_test]].reset_index(drop=True)
    train_pool = pairs.iloc[idx[n_test:]].reset_index(drop=True)
    return train_pool, test_pairs


def split_mol_disjoint_single_side(pairs: pd.DataFrame, seed: int,
                                   mol_test_frac: float = TEST_MOL_FRAC
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a set of candidate molecules M_test.
    Test pairs: m_b in M_test AND m_a not in M_test (candidate side novel).
    Train pool: neither m_a nor m_b in M_test (clean train).
    """
    rng = np.random.RandomState(seed)
    mols = sorted(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    perm = rng.permutation(len(mols))
    n_test_mols = int(round(mol_test_frac * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test_mols])

    test_mask = (pairs["mol_b_id"].isin(test_mols) & ~pairs["mol_a_id"].isin(test_mols))
    train_mask = (~pairs["mol_a_id"].isin(test_mols) & ~pairs["mol_b_id"].isin(test_mols))
    train_pool = pairs[train_mask].reset_index(drop=True)
    test_pairs = pairs[test_mask].reset_index(drop=True)

    # Assert m_b leakage cleared
    train_mols = set(train_pool["mol_a_id"]).union(train_pool["mol_b_id"])
    test_b_mols = set(test_pairs["mol_b_id"])
    assert test_b_mols.isdisjoint(train_mols), \
        f"single-side leakage: {len(test_b_mols & train_mols)} test-mb mols leaked"
    return train_pool, test_pairs


def split_mol_disjoint_both_sides(pairs: pd.DataFrame, seed: int,
                                  mol_test_frac: float = TEST_MOL_FRAC
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict mol-disjoint: hold out M_test. Test pairs have BOTH endpoints in M_test.
    Train pool has NEITHER endpoint in M_test.
    """
    rng = np.random.RandomState(seed)
    mols = sorted(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    perm = rng.permutation(len(mols))
    n_test_mols = int(round(mol_test_frac * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test_mols])

    test_mask = (pairs["mol_a_id"].isin(test_mols) & pairs["mol_b_id"].isin(test_mols))
    train_mask = (~pairs["mol_a_id"].isin(test_mols) & ~pairs["mol_b_id"].isin(test_mols))
    train_pool = pairs[train_mask].reset_index(drop=True)
    test_pairs = pairs[test_mask].reset_index(drop=True)

    train_mols = set(train_pool["mol_a_id"]).union(train_pool["mol_b_id"])
    test_mols_seen = set(test_pairs["mol_a_id"]).union(test_pairs["mol_b_id"])
    assert test_mols_seen.isdisjoint(train_mols), \
        f"both-sides leakage: {len(test_mols_seen & train_mols)} test mols in train"
    return train_pool, test_pairs


SPLITS = {
    "pair_disjoint": split_pair_disjoint,
    "mol_disjoint_single_side": split_mol_disjoint_single_side,
    "mol_disjoint_both_sides": split_mol_disjoint_both_sides,
}


# ---------------------------------------------------------------------------
# One (split, N, seed) run -> dict of both methods
# ---------------------------------------------------------------------------
def run_one_setting(
    train_pool: pd.DataFrame,
    test_pairs: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    N: int,
    seed: int,
    split_name: str,
) -> Dict[str, dict]:
    if len(train_pool) == 0 or len(test_pairs) == 0:
        return {"error": "empty_split",
                "n_pair_train": int(len(train_pool)),
                "n_test_pairs": int(len(test_pairs))}

    rng = np.random.RandomState(seed * 1000 + N)
    pool_idx = np.arange(len(train_pool))
    rng.shuffle(pool_idx)
    use_idx = pool_idx[:N]
    train_pairs = train_pool.iloc[use_idx].copy().reset_index(drop=True)

    # Internal val split (10% of training set, only when N >= 50)
    if N >= 50 and len(train_pairs) >= 20:
        n_val = max(5, int(0.1 * len(train_pairs)))
        val_pairs = train_pairs.iloc[-n_val:].copy().reset_index(drop=True)
        fit_pairs = train_pairs.iloc[:-n_val].copy().reset_index(drop=True)
    else:
        val_pairs = None
        fit_pairs = train_pairs

    # --- FiLMDelta
    A_fit, B_fit, d_fit = stack_pair_fps(fit_pairs, fp_cache)
    A_test, B_test, d_test = stack_pair_fps(test_pairs, fp_cache)
    film = FiLMDeltaPredictor(
        hidden_dims=HIDDEN_DIMS, dropout=DROPOUT, learning_rate=LR,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device="cpu",
    )
    if val_pairs is not None:
        A_v, B_v, d_v = stack_pair_fps(val_pairs, fp_cache)
        film.fit(A_fit, B_fit, d_fit, A_v, B_v, d_v, verbose=False)
    else:
        film.fit(A_fit, B_fit, d_fit, verbose=False)
    film_pred = film.predict(A_test, B_test)
    film_metrics = delta_metrics(d_test, film_pred)

    # --- Direct property predictor
    X_mol, y_mol = build_direct_training_set(fit_pairs, fp_cache)
    n_direct_train = int(len(X_mol))
    direct = DirectPropertyPredictor(input_dim=N_BITS, device="cpu")
    if val_pairs is not None:
        Xv_mol, yv_mol = build_direct_training_set(val_pairs, fp_cache)
        direct.fit(X_mol, y_mol, Xv_mol, yv_mol)
    else:
        direct.fit(X_mol, y_mol)

    pred_b = direct.predict(B_test)
    pred_a = direct.predict(A_test)
    direct_delta_pred = pred_b - pred_a
    direct_delta_metrics = delta_metrics(d_test, direct_delta_pred)

    # Absolute pIC50 MAE (test-set endpoints, deduped by molecule_id)
    abs_rows = []
    abs_rows.extend(zip(test_pairs["mol_a_id"], test_pairs["value_a"], pred_a))
    abs_rows.extend(zip(test_pairs["mol_b_id"], test_pairs["value_b"], pred_b))
    abs_df = pd.DataFrame(abs_rows, columns=["mol_id", "y_true", "y_pred"])
    abs_dedup = abs_df.groupby("mol_id").agg({"y_true": "first", "y_pred": "mean"}).reset_index()
    abs_true = abs_dedup["y_true"].values.astype(np.float32)
    abs_pred = abs_dedup["y_pred"].values.astype(np.float32)
    abs_mae = float(np.mean(np.abs(abs_pred - abs_true)))
    if np.std(abs_pred) > 1e-9 and np.std(abs_true) > 1e-9:
        abs_pearson = float(scipy_stats.pearsonr(abs_pred, abs_true)[0])
        abs_spearman = float(scipy_stats.spearmanr(abs_pred, abs_true)[0])
    else:
        abs_pearson = 0.0; abs_spearman = 0.0
    abs_metrics = {
        "mae": abs_mae,
        "pearson": abs_pearson if not np.isnan(abs_pearson) else 0.0,
        "spearman": abs_spearman if not np.isnan(abs_spearman) else 0.0,
        "n_unique_test_mols": int(len(abs_dedup)),
    }

    out = {
        "split": split_name,
        "seed": seed,
        "N_requested": N,
        "filmdelta": film_metrics,
        "direct_reconstructed_delta": direct_delta_metrics,
        "direct_abs_pIC50": abs_metrics,
        "n_pair_train": int(len(fit_pairs)),
        "n_direct_train_mols": n_direct_train,
        "n_test_pairs": int(len(test_pairs)),
        "n_train_pool": int(len(train_pool)),
    }

    del film, direct, A_fit, B_fit, A_test, B_test, X_mol
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Worker for parallel execution (one full (split, N, seed) cell)
# ---------------------------------------------------------------------------
def worker_run(args_tuple):
    (pairs_csv_path, split_name, seed, N, fp_cache_path) = args_tuple
    # Re-set torch threads inside worker for fair CPU sharing
    torch.set_num_threads(1)
    try:
        pairs = pd.read_csv(pairs_csv_path)
        fp_cache = np.load(fp_cache_path, allow_pickle=True)["cache"].item()
        split_fn = SPLITS[split_name]
        train_pool, test_pairs = split_fn(pairs, seed=seed)
        actual_N = min(N, len(train_pool))
        out = run_one_setting(train_pool, test_pairs, fp_cache,
                              N=actual_N, seed=seed, split_name=split_name)
        out["N_requested_raw"] = N
        out["N_capped_to"] = actual_N
        return out
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(),
                "split": split_name, "seed": seed, "N_requested": N}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(PAIRS_FILE_DEFAULT))
    parser.add_argument("--out", default=str(RESULTS_FILE))
    parser.add_argument("--plot", default=str(PLOT_FILE))
    parser.add_argument("--summary", default=str(SUMMARY_MD))
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel processes (n1-standard-8 = 8 vCPU; "
                             "default 4 to leave headroom for OS).")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 seeds, only N in {25,100}, single split.")
    parser.add_argument("--resume", action="store_true",
                        help="If output JSON exists, skip (split, seed, N) cells already recorded.")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    plot_path = Path(args.plot)
    summary_path = Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_path)
    all_mols = set(pairs["mol_a_id"]).union(pairs["mol_b_id"])
    print(f"PROGRESS: Loaded {len(pairs)} ZAP70 within-assay pairs "
          f"({len(all_mols)} unique mols).", flush=True)

    # Build FP cache once, dump to disk, then workers load it
    t0 = time.time()
    smi_set = pd.concat([pairs["mol_a"], pairs["mol_b"]]).unique()
    fp_cache = {smi: smiles_to_morgan(smi) for smi in smi_set}
    fp_cache_path = out_path.parent / "_fp_cache_exp5b.npz"
    np.savez(fp_cache_path, cache=np.array(fp_cache, dtype=object))
    print(f"PROGRESS: Built Morgan FP cache for {len(fp_cache)} mols "
          f"in {time.time()-t0:.1f}s.", flush=True)

    # Report split sizes for each protocol & seed (sanity)
    print("\nPROGRESS: Split-size preview (seed=0):")
    for split_name, fn in SPLITS.items():
        tp, te = fn(pairs, seed=0)
        n_train_mols = len(set(tp["mol_a_id"]).union(tp["mol_b_id"]))
        n_test_mols = len(set(te["mol_a_id"]).union(te["mol_b_id"]))
        print(f"  {split_name:30s} train_pool={len(tp):5d} ({n_train_mols} mols) "
              f"test={len(te):5d} ({n_test_mols} mols)", flush=True)

    seeds = SEEDS if not args.quick else [0, 1]
    sizes = TRAIN_SIZES + ["ALL"]
    splits_to_run = list(SPLITS.keys())
    if args.quick:
        sizes = [25, 100]
        splits_to_run = ["pair_disjoint"]

    # Compose work list. "ALL" => N = max possible (we pass a giant number, worker caps)
    BIG = 10**9
    work = []
    for split_name in splits_to_run:
        for seed in seeds:
            for n in sizes:
                N_arg = BIG if n == "ALL" else int(n)
                work.append((str(pairs_path), split_name, seed, N_arg, str(fp_cache_path)))

    # Resume: load prior results, skip cells already done
    prior_runs: List[dict] = []
    if args.resume and out_path.exists():
        try:
            with open(out_path) as f:
                prior = json.load(f)
            prior_runs = [r for r in prior.get("runs", []) if "error" not in r]
            done_keys = {(r["split"], r["seed"], r.get("N_requested_raw", r["N_requested"]))
                         for r in prior_runs}
            before = len(work)
            work = [w for w in work if (w[1], w[2], w[3]) not in done_keys]
            print(f"PROGRESS: RESUME from {out_path}: {len(prior_runs)} runs loaded, "
                  f"{before - len(work)} cells skipped, {len(work)} remaining.", flush=True)
        except Exception as e:
            print(f"PROGRESS: resume parse failed ({e}); running all.", flush=True)
            prior_runs = []

    print(f"\nPROGRESS: scheduling {len(work)} runs across {args.workers} workers...", flush=True)

    results = {
        "config": {
            "target": "ZAP70 (CHEMBL2803)",
            "pairs_file": str(pairs_path),
            "n_pairs_total": int(len(pairs)),
            "n_unique_mols": int(len(all_mols)),
            "splits": splits_to_run,
            "test_mol_frac": TEST_MOL_FRAC,
            "test_pair_frac": TEST_PAIR_FRAC,
            "seeds": seeds,
            "train_sizes": sizes,
            "morgan": {"n_bits": N_BITS, "radius": RADIUS},
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "workers": args.workers,
        },
        "runs": list(prior_runs),
    }

    completed = 0
    t_global = time.time()
    last_heartbeat = time.time()

    if args.workers <= 1:
        for w in work:
            t0 = time.time()
            out = worker_run(w)
            out["wall_sec"] = round(time.time() - t0, 1)
            results["runs"].append(out)
            completed += 1
            _log_progress(out, completed, len(work), t_global)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(worker_run, w): w for w in work}
            for fut in as_completed(futures):
                out = fut.result()
                results["runs"].append(out)
                completed += 1
                _log_progress(out, completed, len(work), t_global)
                # Incremental save
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                if time.time() - last_heartbeat > 1200:
                    print(f"PROGRESS: heartbeat - {completed}/{len(work)} runs done, "
                          f"elapsed {(time.time()-t_global)/60:.1f} min", flush=True)
                    last_heartbeat = time.time()

    # Build summary
    summary = summarize(results["runs"])
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"PROGRESS: Wrote results to {out_path}", flush=True)

    try:
        make_plot(summary, plot_path)
        print(f"PROGRESS: Wrote plot to {plot_path}", flush=True)
    except Exception as e:
        print(f"PROGRESS: plotting failed: {e}", flush=True)

    try:
        write_summary_md(summary, results["config"], summary_path)
        print(f"PROGRESS: Wrote summary to {summary_path}", flush=True)
    except Exception as e:
        print(f"PROGRESS: summary write failed: {e}", flush=True)

    print_summary_table(summary)
    print("PROGRESS: done.", flush=True)


def _log_progress(out, completed, total, t_global):
    if "error" in out:
        print(f"PROGRESS: [{completed}/{total}] ERROR split={out.get('split')} "
              f"seed={out.get('seed')} N={out.get('N_requested')}: {out['error']}",
              flush=True)
        return
    fm = out["filmdelta"]; dm = out["direct_reconstructed_delta"]
    elapsed = (time.time() - t_global) / 60.0
    print(f"PROGRESS: [{completed}/{total}] {elapsed:5.1f}min "
          f"split={out['split']:25s} seed={out['seed']} "
          f"N={out.get('N_capped_to', out['N_requested']):4d} "
          f"n_test={out['n_test_pairs']:4d} "
          f"film_mae={fm['mae']:.3f} direct_mae={dm['mae']:.3f} "
          f"adv={(dm['mae']-fm['mae'])/dm['mae']*100:+.1f}%", flush=True)


# ---------------------------------------------------------------------------
# Summary / plot / markdown
# ---------------------------------------------------------------------------
def summarize(runs: List[dict]) -> List[dict]:
    good = [r for r in runs if "error" not in r]
    if not good:
        return []
    rows = []
    for r in good:
        rows.append({
            "split": r["split"],
            "seed": r["seed"],
            "N": r.get("N_capped_to", r["N_requested"]),
            "N_requested_raw": r.get("N_requested_raw", r["N_requested"]),
            "film_mae": r["filmdelta"]["mae"],
            "film_pearson": r["filmdelta"]["pearson"],
            "film_spearman": r["filmdelta"]["spearman"],
            "film_r2": r["filmdelta"]["r2"],
            "direct_delta_mae": r["direct_reconstructed_delta"]["mae"],
            "direct_delta_pearson": r["direct_reconstructed_delta"]["pearson"],
            "direct_delta_spearman": r["direct_reconstructed_delta"]["spearman"],
            "direct_delta_r2": r["direct_reconstructed_delta"]["r2"],
            "direct_abs_mae": r["direct_abs_pIC50"]["mae"],
            "direct_abs_pearson": r["direct_abs_pIC50"]["pearson"],
            "direct_abs_spearman": r["direct_abs_pIC50"]["spearman"],
            "n_pair_train": r["n_pair_train"],
            "n_test_pairs": r["n_test_pairs"],
            "n_unique_test_mols": r["direct_abs_pIC50"]["n_unique_test_mols"],
        })
    df = pd.DataFrame(rows)

    summary = []
    # Bucket by (split, requested N) -- map "ALL" to its post-cap N per split.
    for (split, N_req), grp in df.groupby(["split", "N_requested_raw"]):
        bucket = {
            "split": split,
            "N_requested": int(N_req) if N_req < 10**8 else "ALL",
            "N_actual_mean": float(grp["N"].mean()),
            "n_seeds": int(len(grp)),
            "n_test_pairs_mean": float(grp["n_test_pairs"].mean()),
            "n_unique_test_mols_mean": float(grp["n_unique_test_mols"].mean()),
        }
        for metric in ["mae", "pearson", "spearman", "r2"]:
            for method in ["film", "direct_delta"]:
                col = f"{method}_{metric}"
                vals = grp[col].values
                bucket[f"{col}_mean"] = float(np.mean(vals))
                bucket[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for metric in ["mae", "pearson", "spearman"]:
            col = f"direct_abs_{metric}"
            vals = grp[col].values
            bucket[f"{col}_mean"] = float(np.mean(vals))
            bucket[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        bucket["mae_advantage_film_over_direct"] = (
            bucket["direct_delta_mae_mean"] - bucket["film_mae_mean"]
        )
        if bucket["direct_delta_mae_mean"] > 0:
            bucket["mae_advantage_pct"] = (
                100.0 * (bucket["direct_delta_mae_mean"] - bucket["film_mae_mean"])
                / bucket["direct_delta_mae_mean"]
            )
        else:
            bucket["mae_advantage_pct"] = 0.0
        summary.append(bucket)

    def _sort_key(b):
        n = b["N_requested"]
        n_val = 10**9 if n == "ALL" else int(n)
        return (b["split"], n_val)
    summary.sort(key=_sort_key)
    return summary


def make_plot(summary: List[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_names = ["pair_disjoint", "mol_disjoint_single_side", "mol_disjoint_both_sides"]
    split_titles = ["(a) Pair-disjoint", "(b) Mol-disjoint single-side",
                    "(c) Mol-disjoint both-sides"]
    metrics_to_plot = [("mae", "Delta MAE (pIC50)"),
                       ("pearson", "Delta Pearson r"),
                       ("spearman", "Delta Spearman rho")]

    fig, axes = plt.subplots(len(split_names), len(metrics_to_plot),
                             figsize=(15, 11), sharex=False)

    for i, split_name in enumerate(split_names):
        rows = [r for r in summary if r["split"] == split_name]
        if not rows:
            continue
        Ns = []
        for r in rows:
            n = r["N_requested"]
            Ns.append(r["N_actual_mean"] if n == "ALL" else int(n))
        order = np.argsort(Ns)
        Ns = [Ns[k] for k in order]
        rows = [rows[k] for k in order]
        for j, (metric, ylabel) in enumerate(metrics_to_plot):
            ax = axes[i, j]
            film_y = [r[f"film_{metric}_mean"] for r in rows]
            film_e = [r[f"film_{metric}_std"] for r in rows]
            direct_y = [r[f"direct_delta_{metric}_mean"] for r in rows]
            direct_e = [r[f"direct_delta_{metric}_std"] for r in rows]
            ax.errorbar(Ns, film_y, yerr=film_e, marker="o", capsize=3,
                        label="FiLMDelta", color="#1f77b4", linewidth=2)
            ax.errorbar(Ns, direct_y, yerr=direct_e, marker="s", capsize=3,
                        label="Direct (reconstructed delta)", color="#d62728", linewidth=2)
            ax.set_xscale("log")
            ax.set_xlabel("Training pairs (N)")
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(metrics_to_plot[j][1])
            if j == 0:
                ax.set_ylabel(f"{split_titles[i]}\n{ylabel}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=9)

    fig.suptitle("EXP5b: FiLMDelta vs direct pIC50 -- ZAP70 (CHEMBL2803), 5 seeds",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(summary: List[dict], cfg: dict, path: Path):
    lines = []
    lines.append("# EXP5b: FiLMDelta vs Direct pIC50 -- ZAP70 deep comparison\n")
    lines.append(f"- Target: {cfg['target']}\n")
    lines.append(f"- Pairs total: {cfg['n_pairs_total']}, unique mols: {cfg['n_unique_mols']}\n")
    lines.append(f"- Seeds: {cfg['seeds']}\n")
    lines.append(f"- Train sizes: {cfg['train_sizes']}\n")
    lines.append(f"- Splits: {', '.join(cfg['splits'])}\n\n")

    # Per-split table
    for split in cfg["splits"]:
        rows = [r for r in summary if r["split"] == split]
        if not rows:
            continue
        lines.append(f"## Split: `{split}`\n\n")
        lines.append("| N (req) | N (actual) | n_test | "
                     "film MAE | direct delta MAE | adv % | "
                     "film Spr | direct Spr | direct abs MAE |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            adv = r["mae_advantage_pct"]
            adv_str = f"**{adv:+.1f}**" if adv > 0 else f"{adv:+.1f}"
            lines.append(
                f"| {r['N_requested']} | {r['N_actual_mean']:.0f} | "
                f"{r['n_test_pairs_mean']:.0f} | "
                f"{r['film_mae_mean']:.3f}±{r['film_mae_std']:.3f} | "
                f"{r['direct_delta_mae_mean']:.3f}±{r['direct_delta_mae_std']:.3f} | "
                f"{adv_str} | "
                f"{r['film_spearman_mean']:.3f} | "
                f"{r['direct_delta_spearman_mean']:.3f} | "
                f"{r['direct_abs_mae_mean']:.3f} |\n"
            )
        lines.append("\n")

    # Interpretation
    lines.append("## Interpretation\n\n")
    lines.append(_auto_interpret(summary, cfg))

    with open(path, "w") as f:
        f.writelines(lines)


def _auto_interpret(summary: List[dict], cfg: dict) -> str:
    """Generate a short interpretation paragraph from numbers (heuristic)."""
    text = []
    for split in cfg["splits"]:
        rows = [r for r in summary if r["split"] == split]
        if not rows:
            continue
        wins_film = sum(1 for r in rows if r["mae_advantage_pct"] > 0)
        any_win = "yes" if wins_film == len(rows) else ("partial" if wins_film > 0 else "no")
        adv_avg = float(np.mean([r["mae_advantage_pct"] for r in rows]))
        adv_at_all = next((r["mae_advantage_pct"] for r in rows if r["N_requested"] == "ALL"), None)
        adv_str = f"avg adv={adv_avg:+.1f}%"
        if adv_at_all is not None:
            adv_str += f", at ALL N adv={adv_at_all:+.1f}%"
        text.append(f"- **{split}**: FiLMDelta wins MAE in {wins_film}/{len(rows)} N buckets ({any_win}); {adv_str}.\n")
    text.append("\n")
    text.append("**Headline candidate**: see the mol-disjoint single-side row -- "
                "this is the LO-realistic regime (novel candidate vs known anchor). "
                "If FiLMDelta wins there at all N, that is the paper claim.\n")
    return "".join(text)


def print_summary_table(summary: List[dict]):
    print("\n=== SUMMARY ===")
    for split in sorted(set(r["split"] for r in summary)):
        rows = [r for r in summary if r["split"] == split]
        print(f"\n[{split}]")
        cols = ["N_requested", "N_actual_mean", "n_test_pairs_mean",
                "film_mae_mean", "direct_delta_mae_mean", "mae_advantage_pct",
                "film_spearman_mean", "direct_delta_spearman_mean", "direct_abs_mae_mean"]
        print(" | ".join(f"{c:>22s}" for c in cols))
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, (int, str)):
                    vals.append(f"{v:>22}")
                else:
                    vals.append(f"{v:>22.4f}")
            print(" | ".join(vals))


if __name__ == "__main__":
    main()
