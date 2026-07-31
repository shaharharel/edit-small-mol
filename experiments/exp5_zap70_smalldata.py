#!/usr/bin/env python3
"""
EXP5 - ZAP70 small-data: FiLMDelta vs direct property predictor.

Question
--------
For ZAP70 (CHEMBL2803, ~hundreds of within-assay MMP pairs), how does a
pair-trained delta predictor (FiLMDelta) compare to a direct molecular
property (pIC50) predictor as the training data size N varies?

Setup
-----
- Within-assay ZAP70 pairs constructed from molecule_pIC50_minimal.csv
  (all pairs of molecules sharing the same assay_id).
- 20% of pairs held out as test set (pair-disjoint by random seed).
- For each N in {25, 50, 100, 200, all-available}, both models train on
  the SAME N pairs from the train pool. The direct predictor uses both
  endpoints of each pair as labeled (mol, pIC50) examples (so it sees
  effectively 2N labeled molecules, with duplicates if a mol appears in
  multiple pairs - we dedupe by molecule_id).
- Predictions on the test set are evaluated on the *delta*:
    * FiLMDelta: direct delta prediction
    * Direct: predict(m_b) - predict(m_a)
- Three random seeds for error bars.

Encoder
-------
Morgan FP, 2048 bits, radius 2 (RDKit). Same encoder for both methods.

Architectures
-------------
- FiLMDelta: src.models.predictors.FiLMDeltaPredictor (existing)
- Direct: a matching 3-layer MLP on Morgan FP -> pIC50

Outputs
-------
- results/paper_evaluation/exp5_zap70_smalldata_filmdelta_vs_direct.json
- results/paper_evaluation/exp5_zap70_smalldata_filmdelta_vs_direct.png
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

PAIRS_FILE = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_FILE = RESULTS_DIR / "exp5_zap70_smalldata_filmdelta_vs_direct.json"
PLOT_FILE = RESULTS_DIR / "exp5_zap70_smalldata_filmdelta_vs_direct.png"

N_BITS = 2048
RADIUS = 2
TRAIN_SIZES = [25, 50, 100, 200]  # "all" appended at runtime
SEEDS = [0, 1, 2]
TEST_FRAC = 0.20

# Training hyperparameters (kept light for small data + CPU)
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-3
DROPOUT = 0.2
HIDDEN_DIMS = [512, 256, 128]


# ---------------------------------------------------------------------------
# Morgan FP cache
# ---------------------------------------------------------------------------
def smiles_to_morgan(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def cache_morgan(smiles_iter) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for smi in set(smiles_iter):
        cache[smi] = smiles_to_morgan(smi)
    return cache


# ---------------------------------------------------------------------------
# Direct pIC50 predictor (matching architecture to FiLMDelta backbone)
# ---------------------------------------------------------------------------
class DirectMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DirectPropertyPredictor:
    """Single-molecule pIC50 regressor (Morgan FP -> MLP -> pIC50)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = DROPOUT,
        lr: float = LR,
        batch_size: int = BATCH_SIZE,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        device: str = "cpu",
    ):
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.model: DirectMLP | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> Dict[str, list]:
        self.model = DirectMLP(self.input_dim, self.hidden_dims, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X_train).float()
        yt = torch.from_numpy(y_train).float()
        train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xv = torch.from_numpy(X_val).float()
            yv = torch.from_numpy(y_val).float()
            val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=self.batch_size, shuffle=False)

        hist = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        bad = 0

        for ep in range(self.max_epochs):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            hist["train_loss"].append(float(np.mean(losses)))

            if val_loader is not None:
                self.model.eval()
                vlosses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device); yb = yb.to(self.device)
                        pred = self.model(xb)
                        vlosses.append(loss_fn(pred, yb).item())
                vl = float(np.mean(vlosses))
                hist["val_loss"].append(vl)
                sched.step(vl)
                if vl < best_val:
                    best_val = vl
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= self.patience:
                    break
            else:
                hist["val_loss"].append(hist["train_loss"][-1])

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).float().to(self.device)
            return self.model(Xt).cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def delta_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
# Main experiment
# ---------------------------------------------------------------------------
def make_pair_splits(pairs: pd.DataFrame, seed: int, test_frac: float):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pairs)))
    test_idx = idx[:n_test]
    train_pool_idx = idx[n_test:]
    return train_pool_idx, test_idx


def build_direct_training_set(train_pairs: pd.DataFrame, fp_cache: Dict[str, np.ndarray]):
    """Return (X_mol, y_mol) deduped by molecule_id, averaging pIC50 across appearances."""
    rows = []
    rows.extend(zip(train_pairs["mol_a_id"], train_pairs["mol_a"], train_pairs["value_a"]))
    rows.extend(zip(train_pairs["mol_b_id"], train_pairs["mol_b"], train_pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "pIC50"])
    agg = df.groupby("mol_id").agg({"smiles": "first", "pIC50": "mean"}).reset_index()
    X = np.stack([fp_cache[s] for s in agg["smiles"]])
    y = agg["pIC50"].values.astype(np.float32)
    return X, y


def stack_pair_fps(pairs: pd.DataFrame, fp_cache: Dict[str, np.ndarray]):
    A = np.stack([fp_cache[s] for s in pairs["mol_a"]])
    B = np.stack([fp_cache[s] for s in pairs["mol_b"]])
    d = pairs["delta"].values.astype(np.float32)
    return A.astype(np.float32), B.astype(np.float32), d


def run_one_setting(
    train_pool: pd.DataFrame,
    test_pairs: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    N: int,
    seed: int,
) -> Dict[str, dict]:
    rng = np.random.RandomState(seed * 1000 + N)
    pool_idx = np.arange(len(train_pool))
    rng.shuffle(pool_idx)
    use_idx = pool_idx[:N]
    train_pairs = train_pool.iloc[use_idx].copy()

    # Internal val split (10% of training pool, fixed seed) - only when N >= 50
    n_val = max(5, int(0.1 * len(train_pairs)))
    if N >= 50:
        val_pairs = train_pairs.iloc[-n_val:].copy()
        fit_pairs = train_pairs.iloc[:-n_val].copy()
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
    direct_metrics = delta_metrics(d_test, direct_delta_pred)
    # Also record absolute pIC50 metrics for diagnostic
    y_abs_true = np.concatenate([test_pairs["value_a"].values, test_pairs["value_b"].values]).astype(np.float32)
    y_abs_pred = np.concatenate([pred_a, pred_b])
    abs_mae = float(np.mean(np.abs(y_abs_true - y_abs_pred)))

    out = {
        "filmdelta": film_metrics,
        "direct_reconstructed_delta": direct_metrics,
        "direct_abs_pIC50_mae": abs_mae,
        "n_pair_train": int(len(fit_pairs)),
        "n_direct_train_mols": n_direct_train,
        "n_test_pairs": int(len(test_pairs)),
    }

    # cleanup
    del film, direct, A_fit, B_fit, A_test, B_test, X_mol
    gc.collect()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(PAIRS_FILE))
    parser.add_argument("--out", default=str(RESULTS_FILE))
    parser.add_argument("--plot", default=str(PLOT_FILE))
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    plot_path = Path(args.plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_path)
    print(f"PROGRESS: Loaded {len(pairs)} ZAP70 within-assay pairs "
          f"({len(set(pairs['mol_a_id']).union(pairs['mol_b_id']))} unique mols, "
          f"{pairs['assay_id'].nunique()} assays).", flush=True)

    # FP cache
    t0 = time.time()
    all_smiles = pd.concat([pairs["mol_a"], pairs["mol_b"]]).unique()
    fp_cache = {smi: smiles_to_morgan(smi) for smi in all_smiles}
    print(f"PROGRESS: Built Morgan FP cache for {len(fp_cache)} unique molecules "
          f"in {time.time()-t0:.1f}s.", flush=True)

    results = {
        "config": {
            "target": "ZAP70 (CHEMBL2803)",
            "pairs_file": str(pairs_path),
            "n_pairs_total": int(len(pairs)),
            "n_unique_mols": int(len(fp_cache)),
            "test_frac": TEST_FRAC,
            "seeds": SEEDS,
            "train_sizes_requested": TRAIN_SIZES + ["all"],
            "morgan": {"n_bits": N_BITS, "radius": RADIUS},
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
        "runs": [],
    }

    last_progress = time.time()
    total_runs = 0
    for seed in SEEDS:
        train_pool_idx, test_idx = make_pair_splits(pairs, seed=seed, test_frac=TEST_FRAC)
        train_pool = pairs.iloc[train_pool_idx].reset_index(drop=True)
        test_pairs = pairs.iloc[test_idx].reset_index(drop=True)
        sizes = [n for n in TRAIN_SIZES if n <= len(train_pool)]
        if len(train_pool) not in sizes:
            sizes.append(len(train_pool))  # "all"
        print(f"PROGRESS: seed={seed} train_pool={len(train_pool)} test={len(test_pairs)} "
              f"sizes={sizes}", flush=True)

        for N in sizes:
            t_run = time.time()
            try:
                out = run_one_setting(train_pool, test_pairs, fp_cache, N, seed)
            except Exception as e:
                print(f"PROGRESS: FAILED seed={seed} N={N}: {e}", flush=True)
                continue
            run_record = {
                "seed": seed,
                "N_requested": N,
                "is_all": N == len(train_pool),
                "filmdelta": out["filmdelta"],
                "direct_reconstructed_delta": out["direct_reconstructed_delta"],
                "direct_abs_pIC50_mae": out["direct_abs_pIC50_mae"],
                "n_pair_train": out["n_pair_train"],
                "n_direct_train_mols": out["n_direct_train_mols"],
                "n_test_pairs": out["n_test_pairs"],
                "wall_sec": round(time.time() - t_run, 1),
            }
            results["runs"].append(run_record)
            total_runs += 1

            fm = run_record["filmdelta"]
            dm = run_record["direct_reconstructed_delta"]
            print(
                f"PROGRESS: seed={seed} N={N} "
                f"film_mae={fm['mae']:.3f} film_spr={fm['spearman']:.3f} "
                f"direct_mae={dm['mae']:.3f} direct_spr={dm['spearman']:.3f} "
                f"in {run_record['wall_sec']:.1f}s",
                flush=True,
            )

            # Incremental save
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            if time.time() - last_progress > 900:
                print(f"PROGRESS: heartbeat after {total_runs} runs", flush=True)
                last_progress = time.time()

    # Summary table
    summary = summarize(results["runs"])
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"PROGRESS: Wrote results to {out_path}", flush=True)

    # Plot
    try:
        make_plot(summary, plot_path)
        print(f"PROGRESS: Wrote plot to {plot_path}", flush=True)
    except Exception as e:
        print(f"PROGRESS: plotting failed: {e}", flush=True)

    print_summary(summary)
    print("PROGRESS: done.", flush=True)


def summarize(runs: List[dict]) -> List[dict]:
    """Aggregate runs across seeds by N (using N_requested as bucket; treat 'all' separately if multiple)."""
    df = pd.DataFrame([{
        "seed": r["seed"],
        "N": r["N_requested"],
        "is_all": r["is_all"],
        "film_mae": r["filmdelta"]["mae"],
        "film_pearson": r["filmdelta"]["pearson"],
        "film_spearman": r["filmdelta"]["spearman"],
        "film_r2": r["filmdelta"]["r2"],
        "direct_mae": r["direct_reconstructed_delta"]["mae"],
        "direct_pearson": r["direct_reconstructed_delta"]["pearson"],
        "direct_spearman": r["direct_reconstructed_delta"]["spearman"],
        "direct_r2": r["direct_reconstructed_delta"]["r2"],
        "n_pair_train": r["n_pair_train"],
        "n_direct_train_mols": r["n_direct_train_mols"],
    } for r in runs])
    summary = []
    for N, grp in df.groupby("N"):
        row = {"N": int(N), "n_seeds": int(len(grp)),
               "is_all_any": bool(grp["is_all"].any()),
               "n_pair_train_mean": float(grp["n_pair_train"].mean()),
               "n_direct_train_mols_mean": float(grp["n_direct_train_mols"].mean())}
        for metric in ["mae", "pearson", "spearman", "r2"]:
            for method in ["film", "direct"]:
                vals = grp[f"{method}_{metric}"].values
                row[f"{method}_{metric}_mean"] = float(np.mean(vals))
                row[f"{method}_{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["mae_advantage_film_over_direct"] = row["direct_mae_mean"] - row["film_mae_mean"]
        row["mae_advantage_pct"] = (
            100.0 * (row["direct_mae_mean"] - row["film_mae_mean"]) / row["direct_mae_mean"]
            if row["direct_mae_mean"] > 0 else 0.0
        )
        summary.append(row)
    summary.sort(key=lambda x: x["N"])
    return summary


def make_plot(summary: List[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [s["N"] for s in summary]
    film_mae = [s["film_mae_mean"] for s in summary]
    film_mae_err = [s["film_mae_std"] for s in summary]
    direct_mae = [s["direct_mae_mean"] for s in summary]
    direct_mae_err = [s["direct_mae_std"] for s in summary]
    film_spr = [s["film_spearman_mean"] for s in summary]
    film_spr_err = [s["film_spearman_std"] for s in summary]
    direct_spr = [s["direct_spearman_mean"] for s in summary]
    direct_spr_err = [s["direct_spearman_std"] for s in summary]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ax = axes[0]
    ax.errorbar(Ns, film_mae, yerr=film_mae_err, marker="o", capsize=4, label="FiLMDelta", color="#1f77b4", linewidth=2)
    ax.errorbar(Ns, direct_mae, yerr=direct_mae_err, marker="s", capsize=4, label="Direct (pIC50, reconstructed delta)", color="#d62728", linewidth=2)
    ax.set_xlabel("Training pairs (N)")
    ax.set_ylabel("Delta MAE (pIC50 units)")
    ax.set_title("ZAP70 small-data: Delta MAE vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xscale("log")

    ax = axes[1]
    ax.errorbar(Ns, film_spr, yerr=film_spr_err, marker="o", capsize=4, label="FiLMDelta", color="#1f77b4", linewidth=2)
    ax.errorbar(Ns, direct_spr, yerr=direct_spr_err, marker="s", capsize=4, label="Direct (pIC50, reconstructed delta)", color="#d62728", linewidth=2)
    ax.set_xlabel("Training pairs (N)")
    ax.set_ylabel("Delta Spearman rho")
    ax.set_title("ZAP70 small-data: Delta Spearman vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xscale("log")

    fig.suptitle("FiLMDelta vs direct pIC50 predictor (ZAP70 / CHEMBL2803, within-assay pairs, 3 seeds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(summary: List[dict]):
    print("\n=== SUMMARY ===")
    cols = ["N", "n_pair_train_mean", "n_direct_train_mols_mean",
            "film_mae_mean", "direct_mae_mean", "mae_advantage_pct",
            "film_spearman_mean", "direct_spearman_mean"]
    print(" | ".join(f"{c:>22s}" for c in cols))
    for row in summary:
        print(" | ".join(f"{row.get(c, ''):>22}" if isinstance(row.get(c), (str, int))
                         else f"{row.get(c, 0):>22.4f}"
                         for c in cols))


if __name__ == "__main__":
    main()
