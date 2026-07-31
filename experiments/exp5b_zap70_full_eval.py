#!/usr/bin/env python3
"""
EXP5b ZAP70 FULL EVAL — paper-centerpiece FiLMDelta benchmark.

Matrix:
  - 2 splits: mol_disjoint_both_sides (PRIMARY), lab_disjoint (SECONDARY)
  - 3 predictors: FiLMDelta, DirectDeltaMLP (Morgan diff -> delta),
                  DirectAbsoluteMLP (single Morgan FP -> pIC50)
  - 2 pretrain regimes: None vs Kinase-pretrain -> ZAP70 finetune
  - 2 evaluation modes:
      Delta-prediction: MAE, Pearson, Spearman on test pair delta
      Absolute-prediction: anchor-averaged predicted pIC50 vs true mol pIC50,
        deduped by molecule

K-fold CV (k=5) on mol_disjoint_both_sides for the primary split,
seeded folds on lab_disjoint for the secondary.

Outputs:
  results/paper_evaluation/exp5b_zap70_full_eval.json
  results/paper_evaluation/exp5b_zap70_full_eval.csv
  results/paper_evaluation/exp5b_zap70_full_eval_summary.md
  results/paper_evaluation/exp5b_zap70_full_eval.png
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
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False  # CPU only (avoid MPS instability)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP  # noqa: E402

ZAP70_PAIRS = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
KINASE_PAIRS = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_JSON = RESULTS_DIR / "exp5b_zap70_full_eval.json"
OUT_CSV = RESULTS_DIR / "exp5b_zap70_full_eval.csv"
OUT_MD = RESULTS_DIR / "exp5b_zap70_full_eval_summary.md"
OUT_PNG = RESULTS_DIR / "exp5b_zap70_full_eval.png"
FP_CACHE_DIR = RESULTS_DIR / "_fp_cache_exp5b_full"

N_BITS = 2048
RADIUS = 2
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.2
LR = 1e-3
PRETRAIN_LR = 5e-4
FINETUNE_LR = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 25
PRETRAIN_EPOCHS = 5
KFOLDS = 5
N_LABFOLDS = 5
MAX_KINASE_PAIRS = 30000  # cap for speed; ~32K total available
ANCHOR_AGG = "mean"  # mean-of-anchor-implied-abs


# ---------------------------------------------------------------------------
# Morgan FP cache
# ---------------------------------------------------------------------------
def smi_to_morgan(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def build_fp_cache(smiles_iter) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for smi in set(smiles_iter):
        cache[smi] = smi_to_morgan(smi)
    return cache


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
class DirectDeltaMLP(nn.Module):
    """Maps Morgan FP diff (B - A, 2048d) -> predicted delta."""

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

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.net(diff).squeeze(-1)


class DirectAbsoluteMLP(nn.Module):
    """Maps single Morgan FP (2048d) -> pIC50."""

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


def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    lr: float,
    max_epochs: int,
    patience: int,
    forward_fn,
    device: str = "cpu",
) -> nn.Module:
    """Generic train loop; forward_fn(model, batch) -> (pred, target)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            pred, target = forward_fn(model, [t.to(device) for t in batch])
            opt.zero_grad()
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if val_loader is not None:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for batch in val_loader:
                    pred, target = forward_fn(model, [t.to(device) for t in batch])
                    vlosses.append(loss_fn(pred, target).item())
            vl = float(np.mean(vlosses))
            sched.step(vl)
            if vl < best - 1e-5:
                best = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "pearson": float("nan"),
                "spearman": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(y_pred - y_true)))
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        pr, sr = 0.0, 0.0
    else:
        pr = float(scipy_stats.pearsonr(y_pred, y_true)[0])
        sr = float(scipy_stats.spearmanr(y_pred, y_true)[0])
    return {
        "mae": mae,
        "pearson": pr if not np.isnan(pr) else 0.0,
        "spearman": sr if not np.isnan(sr) else 0.0,
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def kfold_mol_disjoint_both_sides(pairs: pd.DataFrame, k: int, seed: int = 0
                                  ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """K-fold over UNIQUE MOLECULES; for each fold:
       test pairs  = pairs where BOTH mols in fold,
       train pairs = pairs where NEITHER mol in fold.
    """
    rng = np.random.RandomState(seed)
    mols = sorted(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    perm = rng.permutation(len(mols))
    folds_mols = np.array_split(perm, k)
    folds_mols = [set(mols[i] for i in idxs) for idxs in folds_mols]

    out = []
    for fi, test_mols in enumerate(folds_mols):
        test_mask = pairs["mol_a_id"].isin(test_mols) & pairs["mol_b_id"].isin(test_mols)
        train_mask = ~pairs["mol_a_id"].isin(test_mols) & ~pairs["mol_b_id"].isin(test_mols)
        train_pairs = pairs[train_mask].reset_index(drop=True)
        test_pairs = pairs[test_mask].reset_index(drop=True)
        train_mols = set(train_pairs["mol_a_id"]).union(train_pairs["mol_b_id"])
        test_mols_seen = set(test_pairs["mol_a_id"]).union(test_pairs["mol_b_id"])
        # Strict: test mols should be disjoint from train mols.
        assert test_mols_seen.isdisjoint(train_mols), (
            f"fold {fi}: leak {len(test_mols_seen & train_mols)} mols")
        out.append((train_pairs, test_pairs))
    return out


def lab_disjoint_folds(pairs: pd.DataFrame, n_folds: int = 5, seed: int = 0
                       ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Hold out 1/n_folds of assays (by pair-weight) as test; others train.
    Repeats with different random partitions of assays."""
    folds = []
    rng = np.random.RandomState(seed)
    assay_counts = pairs["assay_id"].value_counts()
    assays = sorted(assay_counts.index.tolist())
    n_assays = len(assays)
    target_test_frac = 1.0 / n_folds

    for fi in range(n_folds):
        rng_fold = np.random.RandomState(seed + fi * 1000)
        # Greedy partition: shuffle assays then pick first chunk by pair-weight.
        perm = rng_fold.permutation(n_assays)
        total = len(pairs)
        target_test_pairs = int(target_test_frac * total)
        test_assays = set()
        cum = 0
        for j in perm:
            a = assays[j]
            if cum >= target_test_pairs:
                break
            test_assays.add(a)
            cum += assay_counts[a]
        train_assays = set(assays) - test_assays
        test_pairs = pairs[pairs["assay_id"].isin(test_assays)].reset_index(drop=True)
        train_pairs = pairs[pairs["assay_id"].isin(train_assays)].reset_index(drop=True)
        # Note: lab-disjoint does NOT enforce mol-disjoint; that's by design.
        folds.append((train_pairs, test_pairs))
    return folds


# ---------------------------------------------------------------------------
# Anchor-based absolute prediction
# ---------------------------------------------------------------------------
def anchor_based_absolute_preds(
    predict_delta_fn,
    train_anchors: pd.DataFrame,  # cols: mol_id, smiles, value
    test_mols: pd.DataFrame,       # cols: mol_id, smiles, true_value
    fp_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """For each test mol, predict abs pIC50 as mean over train anchors of
       (anchor_value + predicted_delta(anchor, test_mol)).
       Returns (y_true, y_pred, mol_ids).
    """
    if len(train_anchors) == 0 or len(test_mols) == 0:
        return np.array([]), np.array([]), []

    A_fp = np.stack([fp_cache[s] for s in train_anchors["smiles"]]).astype(np.float32)
    anchor_vals = train_anchors["value"].values.astype(np.float32)
    n_anchors = len(train_anchors)

    abs_preds, abs_true, ids = [], [], []
    # Batch over test mols (one mol at a time, but vectorized over anchors).
    for _, row in test_mols.iterrows():
        smi = row["smiles"]
        if smi not in fp_cache:
            continue
        b_fp = fp_cache[smi].astype(np.float32)
        B_fp = np.tile(b_fp, (n_anchors, 1))
        delta_pred = predict_delta_fn(A_fp, B_fp)
        implied_abs = anchor_vals + delta_pred  # value_a + delta(a->b) = predicted value_b
        if ANCHOR_AGG == "mean":
            agg = float(np.mean(implied_abs))
        else:
            agg = float(np.median(implied_abs))
        abs_preds.append(agg)
        abs_true.append(float(row["true_value"]))
        ids.append(row["mol_id"])
    return np.array(abs_true, dtype=np.float32), np.array(abs_preds, dtype=np.float32), ids


def collect_endpoint_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert pair df -> (mol_id, smiles, value) with mol_id-deduped mean."""
    rows = []
    rows.extend(zip(pairs["mol_a_id"], pairs["mol_a"], pairs["value_a"]))
    rows.extend(zip(pairs["mol_b_id"], pairs["mol_b"], pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"])
    return df.groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})


# ---------------------------------------------------------------------------
# Predictor wrappers (provide common .fit() and .predict_delta(A, B))
# ---------------------------------------------------------------------------
class FiLMDeltaWrapper:
    NAME = "FiLMDelta"

    def __init__(self, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                 lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.predictor = FiLMDeltaPredictor(
            hidden_dims=hidden_dims, dropout=dropout, learning_rate=lr,
            batch_size=BATCH_SIZE, max_epochs=max_epochs, patience=patience,
            device="cpu",
        )

    def fit(self, A_train, B_train, d_train, A_val=None, B_val=None, d_val=None,
            pretrained_state: dict | None = None):
        # Build model first if pretraining state provided.
        if pretrained_state is not None:
            self.predictor.input_dim = A_train.shape[1]
            self.predictor.model = FiLMDeltaMLP(
                input_dim=A_train.shape[1], hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            self.predictor.model.load_state_dict(pretrained_state)
            # Manual finetune loop to use a lower LR.
            self._finetune(A_train, B_train, d_train, A_val, B_val, d_val)
        else:
            if A_val is not None:
                self.predictor.fit(A_train, B_train, d_train,
                                   A_val, B_val, d_val, verbose=False)
            else:
                self.predictor.fit(A_train, B_train, d_train, verbose=False)
        return self

    def _finetune(self, A, B, d, Av, Bv, dv):
        m = self.predictor.model.to("cpu")
        opt = torch.optim.Adam(m.parameters(), lr=FINETUNE_LR, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        At, Bt, dt = (torch.from_numpy(x).float() for x in (A, B, d))
        ds = TensorDataset(At, Bt, dt)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        if Av is not None:
            Avt, Bvt, dvt = (torch.from_numpy(x).float() for x in (Av, Bv, dv))
            vds = TensorDataset(Avt, Bvt, dvt)
            vld = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False)
        else:
            vld = None
        best, bad, best_state = float("inf"), 0, None
        for ep in range(self.max_epochs):
            m.train()
            for a, b, y in ld:
                opt.zero_grad()
                loss = loss_fn(m(a, b), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            if vld is not None:
                m.eval()
                vls = []
                with torch.no_grad():
                    for a, b, y in vld:
                        vls.append(loss_fn(m(a, b), y).item())
                vl = float(np.mean(vls))
                if vl < best - 1e-5:
                    best = vl
                    best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            m.load_state_dict(best_state)
        self.predictor.model = m

    def predict_delta(self, A, B):
        return self.predictor.predict(A, B)


class DirectDeltaWrapper:
    NAME = "DirectDeltaMLP"

    def __init__(self, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                 lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: DirectDeltaMLP | None = None
        self.input_dim: int | None = None

    def _make_loader(self, A, B, d, shuffle):
        diff = (B - A).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(diff), torch.from_numpy(d.astype(np.float32)))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    def fit(self, A_train, B_train, d_train, A_val=None, B_val=None, d_val=None,
            pretrained_state: dict | None = None):
        self.input_dim = A_train.shape[1]
        if self.model is None:
            self.model = DirectDeltaMLP(self.input_dim, self.hidden_dims, self.dropout)
        if pretrained_state is not None:
            self.model.load_state_dict(pretrained_state)
        train_loader = self._make_loader(A_train, B_train, d_train, shuffle=True)
        val_loader = self._make_loader(A_val, B_val, d_val, shuffle=False) if A_val is not None else None

        def fwd(m, batch):
            x, y = batch
            return m(x), y

        lr_use = FINETUNE_LR if pretrained_state is not None else self.lr
        self.model = _train_loop(self.model, train_loader, val_loader,
                                 lr_use, self.max_epochs, self.patience, fwd)
        return self

    def predict_delta(self, A, B):
        assert self.model is not None
        self.model.eval()
        diff = torch.from_numpy((B - A).astype(np.float32))
        with torch.no_grad():
            return self.model(diff).cpu().numpy()


class DirectAbsoluteWrapper:
    """Single-mol pIC50 regressor; reconstructs delta as pred(B) - pred(A)."""

    NAME = "DirectAbsoluteMLP"

    def __init__(self, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                 lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: DirectAbsoluteMLP | None = None
        self.input_dim: int | None = None

    def _build_training_set(self, pairs: pd.DataFrame, fp_cache):
        """Dedup by mol_id, average pIC50 across appearances."""
        rows = []
        rows.extend(zip(pairs["mol_a_id"], pairs["mol_a"], pairs["value_a"]))
        rows.extend(zip(pairs["mol_b_id"], pairs["mol_b"], pairs["value_b"]))
        df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"])
        agg = df.groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})
        X = np.stack([fp_cache[s] for s in agg["smiles"]]).astype(np.float32)
        y = agg["value"].values.astype(np.float32)
        return X, y

    def fit_from_pairs(self, train_pairs, val_pairs, fp_cache, pretrained_state=None):
        X, y = self._build_training_set(train_pairs, fp_cache)
        Xv, yv = self._build_training_set(val_pairs, fp_cache) if (val_pairs is not None and len(val_pairs) > 0) else (None, None)
        self.input_dim = X.shape[1]
        if self.model is None:
            self.model = DirectAbsoluteMLP(self.input_dim, self.hidden_dims, self.dropout)
        if pretrained_state is not None:
            self.model.load_state_dict(pretrained_state)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = None
        if Xv is not None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)),
                batch_size=BATCH_SIZE, shuffle=False,
            )

        def fwd(m, batch):
            x, y = batch
            return m(x), y

        lr_use = FINETUNE_LR if pretrained_state is not None else self.lr
        self.model = _train_loop(self.model, train_loader, val_loader,
                                 lr_use, self.max_epochs, self.patience, fwd)
        return self

    def predict_abs(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X.astype(np.float32))).cpu().numpy()

    def predict_delta(self, A, B):
        return self.predict_abs(B) - self.predict_abs(A)


# ---------------------------------------------------------------------------
# Kinase pretraining (one-time per predictor type)
# ---------------------------------------------------------------------------
def kinase_pretrain_state(predictor_cls, kinase_df: pd.DataFrame,
                          fp_cache: Dict[str, np.ndarray], epochs: int = PRETRAIN_EPOCHS,
                          verbose: bool = True) -> dict:
    """Pretrain the relevant model and return its state_dict."""
    t0 = time.time()
    A = np.stack([fp_cache[s] for s in kinase_df["mol_a"]]).astype(np.float32)
    B = np.stack([fp_cache[s] for s in kinase_df["mol_b"]]).astype(np.float32)
    d = kinase_df["delta"].values.astype(np.float32)

    # 10% val for early stopping
    n_val = max(500, int(0.1 * len(d)))
    A_tr, A_v = A[:-n_val], A[-n_val:]
    B_tr, B_v = B[:-n_val], B[-n_val:]
    d_tr, d_v = d[:-n_val], d[-n_val:]

    if predictor_cls is FiLMDeltaWrapper:
        m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(A_tr), torch.from_numpy(B_tr), torch.from_numpy(d_tr)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(A_v), torch.from_numpy(B_v), torch.from_numpy(d_v)),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        def fwd(model, batch):
            a, b, y = batch
            return model(a, b), y
        m = _train_loop(m, train_loader, val_loader,
                        PRETRAIN_LR, epochs, patience=3, forward_fn=fwd)
    elif predictor_cls is DirectDeltaWrapper:
        m = DirectDeltaMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
        diff_tr = (B_tr - A_tr).astype(np.float32)
        diff_v = (B_v - A_v).astype(np.float32)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(diff_tr), torch.from_numpy(d_tr)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(diff_v), torch.from_numpy(d_v)),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        def fwd(model, batch):
            x, y = batch
            return model(x), y
        m = _train_loop(m, train_loader, val_loader,
                        PRETRAIN_LR, epochs, patience=3, forward_fn=fwd)
    elif predictor_cls is DirectAbsoluteWrapper:
        m = DirectAbsoluteMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
        # Dedup by molecule id within kinase set
        rows = []
        rows.extend(zip(kinase_df["mol_a_id"], kinase_df["mol_a"], kinase_df["value_a"]))
        rows.extend(zip(kinase_df["mol_b_id"], kinase_df["mol_b"], kinase_df["value_b"]))
        df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"]) \
                .groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})
        X = np.stack([fp_cache[s] for s in df["smiles"]]).astype(np.float32)
        y = df["value"].values.astype(np.float32)
        n_v = max(200, int(0.1 * len(y)))
        X_tr, X_v = X[:-n_v], X[-n_v:]
        y_tr, y_v = y[:-n_v], y[-n_v:]
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v)),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        def fwd(model, batch):
            x, y = batch
            return model(x), y
        m = _train_loop(m, train_loader, val_loader,
                        PRETRAIN_LR, epochs, patience=3, forward_fn=fwd)
    else:
        raise ValueError(predictor_cls)

    state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if verbose:
        print(f"  [pretrain] {predictor_cls.__name__} done in {time.time()-t0:.1f}s "
              f"on {len(d)} kinase pairs", flush=True)
    return state


# ---------------------------------------------------------------------------
# One fold runner: returns metrics for one (split, predictor, pretrain, fold)
# ---------------------------------------------------------------------------
def run_fold(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    predictor_cls,
    pretrained_state: dict | None,
    fold_id: int,
) -> Dict:
    """Train predictor on train_pairs, evaluate on test_pairs.
       Returns dict with delta_* and abs_* metrics."""
    # Internal val split for early stopping (10% of training pairs)
    n_val = max(20, int(0.1 * len(train_pairs)))
    rng = np.random.RandomState(fold_id * 7 + 3)
    perm = rng.permutation(len(train_pairs))
    val_pairs = train_pairs.iloc[perm[:n_val]].reset_index(drop=True)
    fit_pairs = train_pairs.iloc[perm[n_val:]].reset_index(drop=True)

    # Stack FPs
    def stack(pairs):
        A = np.stack([fp_cache[s] for s in pairs["mol_a"]]).astype(np.float32)
        B = np.stack([fp_cache[s] for s in pairs["mol_b"]]).astype(np.float32)
        d = pairs["delta"].values.astype(np.float32)
        return A, B, d

    A_fit, B_fit, d_fit = stack(fit_pairs)
    A_val, B_val, d_val = stack(val_pairs)
    A_test, B_test, d_test = stack(test_pairs)

    # Train predictor
    wrapper = predictor_cls()
    if predictor_cls is DirectAbsoluteWrapper:
        wrapper.fit_from_pairs(fit_pairs, val_pairs, fp_cache,
                               pretrained_state=pretrained_state)
    else:
        wrapper.fit(A_fit, B_fit, d_fit, A_val, B_val, d_val,
                    pretrained_state=pretrained_state)

    # Delta-prediction metrics
    delta_pred = wrapper.predict_delta(A_test, B_test)
    delta_m = regression_metrics(d_test, delta_pred)

    # Anchor-based absolute prediction
    # Anchors = train endpoints (FIT pairs only, to avoid leaking val-set values; conservative)
    # Test mols = endpoints of test pairs (deduped)
    train_anchors = collect_endpoint_table(fit_pairs)
    test_mols = collect_endpoint_table(test_pairs)
    test_mols = test_mols.rename(columns={"value": "true_value"})

    abs_true, abs_pred, abs_ids = anchor_based_absolute_preds(
        wrapper.predict_delta, train_anchors, test_mols, fp_cache,
    )
    abs_m = regression_metrics(abs_true, abs_pred)

    out = {
        "fold": fold_id,
        "n_train_pairs": int(len(fit_pairs)),
        "n_val_pairs": int(len(val_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "n_train_anchors": int(len(train_anchors)),
        "n_test_mols": int(len(abs_ids)),
        "delta_mae": delta_m["mae"],
        "delta_pearson": delta_m["pearson"],
        "delta_spearman": delta_m["spearman"],
        "abs_mae": abs_m["mae"],
        "abs_pearson": abs_m["pearson"],
        "abs_spearman": abs_m["spearman"],
    }
    # cleanup
    del wrapper, A_fit, B_fit, A_val, B_val, A_test, B_test
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: 2 folds per split, skip pretrain regimes.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    t_global = time.time()

    # --- Load ZAP70 pairs
    pairs = pd.read_csv(ZAP70_PAIRS)
    n_mols = len(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    print(f"PROGRESS: ZAP70: {len(pairs)} pairs, {n_mols} mols, "
          f"{pairs['assay_id'].nunique()} assays", flush=True)

    # --- Load kinase pretrain pairs (exclude ZAP70 — which it already is)
    kinase_df = pd.read_csv(KINASE_PAIRS)
    print(f"PROGRESS: kinase: {len(kinase_df)} total pairs, "
          f"targets={sorted(kinase_df['target_chembl_id'].unique())}", flush=True)
    if len(kinase_df) > MAX_KINASE_PAIRS:
        kinase_df = kinase_df.sample(n=MAX_KINASE_PAIRS, random_state=42).reset_index(drop=True)
        print(f"PROGRESS: kinase subsampled to {len(kinase_df)}", flush=True)

    # --- Build combined Morgan FP cache (ZAP70 + kinase)
    t0 = time.time()
    smis = list(
        set(pairs["mol_a"]).union(pairs["mol_b"])
        | set(kinase_df["mol_a"]).union(kinase_df["mol_b"])
    )
    fp_cache = {s: smi_to_morgan(s) for s in smis}
    print(f"PROGRESS: FP cache built for {len(fp_cache)} unique mols in "
          f"{time.time()-t0:.1f}s", flush=True)

    # --- Define splits
    if args.quick:
        kfolds = 2
        n_lab_folds = 2
        pretrain_epochs = 1
    else:
        kfolds = KFOLDS
        n_lab_folds = N_LABFOLDS
        pretrain_epochs = PRETRAIN_EPOCHS

    print("PROGRESS: building folds...", flush=True)
    mol_folds = kfold_mol_disjoint_both_sides(pairs, k=kfolds, seed=0)
    lab_folds = lab_disjoint_folds(pairs, n_folds=n_lab_folds, seed=0)
    for fi, (tr, te) in enumerate(mol_folds):
        print(f"  mol_disjoint fold {fi}: train={len(tr)} test={len(te)}", flush=True)
    for fi, (tr, te) in enumerate(lab_folds):
        train_assays = sorted(tr["assay_id"].unique())[:3]
        test_assays = sorted(te["assay_id"].unique())[:3]
        print(f"  lab_disjoint fold {fi}: train={len(tr)} test={len(te)} "
              f"(train_assays_head={train_assays} test_assays_head={test_assays})",
              flush=True)

    splits = {
        "mol_disjoint_both_sides": mol_folds,
        "lab_disjoint": lab_folds,
    }

    predictors = [FiLMDeltaWrapper, DirectDeltaWrapper, DirectAbsoluteWrapper]
    pretrain_regimes = ["none", "kinase"]

    # --- Pretrain checkpoints (one per predictor type, reused)
    print("PROGRESS: kinase pretraining (3 predictor types)...", flush=True)
    pretrain_states: Dict[str, dict] = {}
    for pcls in predictors:
        st = kinase_pretrain_state(pcls, kinase_df, fp_cache,
                                   epochs=pretrain_epochs, verbose=True)
        pretrain_states[pcls.__name__] = st

    # --- Resume support
    results = {"runs": []}
    if args.resume and out_path.exists():
        try:
            prior = json.load(open(out_path))
            results["runs"] = prior.get("runs", [])
            print(f"PROGRESS: resume — loaded {len(results['runs'])} prior runs",
                  flush=True)
        except Exception as e:
            print(f"PROGRESS: resume parse failed: {e}", flush=True)

    done_keys = {(r["split"], r["predictor"], r["pretrain"], r["fold"])
                 for r in results["runs"]}
    total_cells = sum(len(f) for f in splits.values()) * len(predictors) * len(pretrain_regimes)
    completed = len(done_keys)
    print(f"\nPROGRESS: total cells = {total_cells}; resuming with {completed} done.",
          flush=True)

    config = {
        "target": "ZAP70 (CHEMBL2803)",
        "zap70_pairs": str(ZAP70_PAIRS),
        "kinase_pairs": str(KINASE_PAIRS),
        "n_pairs_zap70": int(len(pairs)),
        "n_mols_zap70": int(n_mols),
        "n_kinase_pairs_used": int(len(kinase_df)),
        "kinase_targets": sorted(kinase_df["target_chembl_id"].unique().tolist()),
        "k_mol_folds": kfolds,
        "n_lab_folds": n_lab_folds,
        "pretrain_epochs": pretrain_epochs,
        "morgan": {"n_bits": N_BITS, "radius": RADIUS},
        "hidden_dims": HIDDEN_DIMS,
        "lr": LR,
        "pretrain_lr": PRETRAIN_LR,
        "finetune_lr": FINETUNE_LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "anchor_aggregation": ANCHOR_AGG,
    }
    results["config"] = config

    # --- Main loop
    for split_name, folds in splits.items():
        for pcls in predictors:
            for pretrain in pretrain_regimes:
                for fi, (train_pairs, test_pairs) in enumerate(folds):
                    key = (split_name, pcls.NAME, pretrain, fi)
                    if key in done_keys:
                        continue
                    t_cell = time.time()
                    pretrained = pretrain_states[pcls.__name__] if pretrain == "kinase" else None
                    try:
                        out = run_fold(train_pairs, test_pairs, fp_cache,
                                       pcls, pretrained, fold_id=fi)
                    except Exception as e:
                        import traceback
                        print(f"PROGRESS: ERROR {key}: {e}\n{traceback.format_exc()}",
                              flush=True)
                        out = {"fold": fi, "error": str(e)}
                    out["split"] = split_name
                    out["predictor"] = pcls.NAME
                    out["pretrain"] = pretrain
                    out["wall_sec"] = round(time.time() - t_cell, 1)
                    results["runs"].append(out)
                    completed += 1
                    if "error" in out:
                        print(f"PROGRESS [{completed}/{total_cells}] {key} ERROR "
                              f"({out['wall_sec']:.0f}s)", flush=True)
                    else:
                        print(f"PROGRESS [{completed}/{total_cells}] "
                              f"split={split_name:25s} pred={pcls.NAME:18s} "
                              f"pre={pretrain:6s} fold={fi} "
                              f"d_mae={out['delta_mae']:.3f} d_spr={out['delta_spearman']:.3f} "
                              f"a_mae={out['abs_mae']:.3f} a_spr={out['abs_spearman']:.3f} "
                              f"({out['wall_sec']:.0f}s)", flush=True)
                    # Incremental save
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)

    # --- Summarize
    summary = summarize(results["runs"])
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPROGRESS: wrote {out_path}", flush=True)

    # Tidy CSV
    write_csv(results["runs"], OUT_CSV)
    print(f"PROGRESS: wrote {OUT_CSV}", flush=True)

    # MD summary
    write_md(summary, config, OUT_MD)
    print(f"PROGRESS: wrote {OUT_MD}", flush=True)

    # Plot
    try:
        make_plot(summary, OUT_PNG)
        print(f"PROGRESS: wrote {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"PROGRESS: plot failed: {e}", flush=True)

    print(f"\nPROGRESS: TOTAL TIME {(time.time()-t_global)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Summarize & report
# ---------------------------------------------------------------------------
def summarize(runs: List[dict]) -> List[dict]:
    good = [r for r in runs if "error" not in r]
    if not good:
        return []
    df = pd.DataFrame(good)
    bucket_cols = ["split", "predictor", "pretrain"]
    metric_cols = ["delta_mae", "delta_pearson", "delta_spearman",
                   "abs_mae", "abs_pearson", "abs_spearman"]
    out = []
    for keys, grp in df.groupby(bucket_cols):
        rec = dict(zip(bucket_cols, keys))
        rec["n_folds"] = int(len(grp))
        rec["n_test_pairs_mean"] = float(grp["n_test_pairs"].mean())
        rec["n_test_mols_mean"] = float(grp["n_test_mols"].mean())
        rec["n_train_pairs_mean"] = float(grp["n_train_pairs"].mean())
        for m in metric_cols:
            rec[f"{m}_mean"] = float(grp[m].mean())
            rec[f"{m}_std"] = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
        out.append(rec)
    return out


def write_csv(runs: List[dict], path: Path):
    rows = []
    for r in runs:
        if "error" in r:
            continue
        for metric in ["delta_mae", "delta_pearson", "delta_spearman",
                       "abs_mae", "abs_pearson", "abs_spearman"]:
            rows.append({
                "split": r["split"],
                "predictor": r["predictor"],
                "pretrain": r["pretrain"],
                "fold": r["fold"],
                "metric": metric,
                "value": r[metric],
            })
    df = pd.DataFrame(rows)
    # Add aggregated mean/std rows for downstream consumption
    if len(df) > 0:
        agg = df.groupby(["split", "predictor", "pretrain", "metric"]).agg(
            mean=("value", "mean"),
            std=("value", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            n=("value", "size"),
        ).reset_index()
        agg.to_csv(path.parent / (path.stem + "_aggregated.csv"), index=False)
    df.to_csv(path, index=False)


def write_md(summary: List[dict], cfg: dict, path: Path):
    lines = []
    lines.append("# EXP5b ZAP70 FULL EVAL — paper centerpiece\n\n")
    lines.append(f"- Target: {cfg['target']}\n")
    lines.append(f"- ZAP70 pairs: {cfg['n_pairs_zap70']} ({cfg['n_mols_zap70']} mols)\n")
    lines.append(f"- Kinase pretrain pool: {cfg['n_kinase_pairs_used']} pairs across "
                 f"{len(cfg['kinase_targets'])} targets {cfg['kinase_targets']}\n")
    lines.append(f"- K-fold (mol_disjoint_both_sides): k={cfg['k_mol_folds']}\n")
    lines.append(f"- Lab-disjoint folds: {cfg['n_lab_folds']}\n")
    lines.append(f"- Pretrain epochs: {cfg['pretrain_epochs']}\n")
    lines.append(f"- Anchor aggregation: {cfg['anchor_aggregation']}\n\n")

    splits = sorted(set(r["split"] for r in summary))
    predictors = ["FiLMDelta", "DirectDeltaMLP", "DirectAbsoluteMLP"]
    pretrains = ["none", "kinase"]

    for split in splits:
        rows = [r for r in summary if r["split"] == split]
        lines.append(f"## Split: `{split}`\n\n")
        lines.append("### Delta prediction (predicted Δ vs true Δ on test pairs)\n\n")
        lines.append("| Predictor | Pretrain | n_folds | n_test_pairs | MAE | Pearson | Spearman |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for p in predictors:
            for pt in pretrains:
                r = next((x for x in rows if x["predictor"] == p and x["pretrain"] == pt), None)
                if r is None:
                    continue
                lines.append(
                    f"| {p} | {pt} | {r['n_folds']} | {r['n_test_pairs_mean']:.0f} | "
                    f"{r['delta_mae_mean']:.3f}±{r['delta_mae_std']:.3f} | "
                    f"{r['delta_pearson_mean']:.3f}±{r['delta_pearson_std']:.3f} | "
                    f"{r['delta_spearman_mean']:.3f}±{r['delta_spearman_std']:.3f} |\n"
                )
        lines.append("\n### Absolute prediction (anchor-averaged predicted pIC50 vs true)\n\n")
        lines.append("| Predictor | Pretrain | n_test_mols | MAE | Pearson | Spearman |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for p in predictors:
            for pt in pretrains:
                r = next((x for x in rows if x["predictor"] == p and x["pretrain"] == pt), None)
                if r is None:
                    continue
                lines.append(
                    f"| {p} | {pt} | {r['n_test_mols_mean']:.0f} | "
                    f"{r['abs_mae_mean']:.3f}±{r['abs_mae_std']:.3f} | "
                    f"{r['abs_pearson_mean']:.3f}±{r['abs_pearson_std']:.3f} | "
                    f"{r['abs_spearman_mean']:.3f}±{r['abs_spearman_std']:.3f} |\n"
                )
        lines.append("\n")

    # Headline
    lines.append("## Headline\n\n")
    headline_split = "mol_disjoint_both_sides"
    rows = [r for r in summary if r["split"] == headline_split]
    if rows:
        film_kp = next((r for r in rows if r["predictor"] == "FiLMDelta" and r["pretrain"] == "kinase"), None)
        film_none = next((r for r in rows if r["predictor"] == "FiLMDelta" and r["pretrain"] == "none"), None)
        dd_none = next((r for r in rows if r["predictor"] == "DirectDeltaMLP" and r["pretrain"] == "none"), None)
        da_none = next((r for r in rows if r["predictor"] == "DirectAbsoluteMLP" and r["pretrain"] == "none"), None)
        if film_kp and film_none and dd_none and da_none:
            lines.append(
                f"On {headline_split}, FiLMDelta (no pretrain) delta MAE = "
                f"{film_none['delta_mae_mean']:.3f}±{film_none['delta_mae_std']:.3f}, "
                f"vs DirectDeltaMLP {dd_none['delta_mae_mean']:.3f}±{dd_none['delta_mae_std']:.3f} "
                f"and DirectAbsoluteMLP reconstructed Δ {da_none['delta_mae_mean']:.3f}±"
                f"{da_none['delta_mae_std']:.3f}.\n"
            )
            lines.append(
                f"Kinase pretrain on FiLMDelta: delta MAE "
                f"{film_kp['delta_mae_mean']:.3f}±{film_kp['delta_mae_std']:.3f} "
                f"({(film_none['delta_mae_mean'] - film_kp['delta_mae_mean']):+.3f} vs no-pretrain).\n"
            )
            lines.append(
                f"Anchor-averaged absolute MAE: FiLMDelta(+kinase) "
                f"{film_kp['abs_mae_mean']:.3f}±{film_kp['abs_mae_std']:.3f}, "
                f"FiLMDelta(no pre) {film_none['abs_mae_mean']:.3f}±{film_none['abs_mae_std']:.3f}, "
                f"DirectAbs {da_none['abs_mae_mean']:.3f}±{da_none['abs_mae_std']:.3f}.\n"
            )
    with open(path, "w") as f:
        f.writelines(lines)


def make_plot(summary: List[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted(set(r["split"] for r in summary))
    predictors = ["FiLMDelta", "DirectDeltaMLP", "DirectAbsoluteMLP"]
    pretrains = ["none", "kinase"]
    # 6-panel grid: rows = (delta, absolute), cols = (MAE, Pearson, Spearman)
    panels = [
        ("delta_mae", "Delta MAE (lower better)"),
        ("delta_pearson", "Delta Pearson r"),
        ("delta_spearman", "Delta Spearman rho"),
        ("abs_mae", "Abs MAE (lower better)"),
        ("abs_pearson", "Abs Pearson r"),
        ("abs_spearman", "Abs Spearman rho"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    axes = axes.flatten()

    bar_w = 0.13
    # 6 bars per split: 3 predictors × 2 pretrain regimes
    color_map = {
        ("FiLMDelta", "none"): "#1f77b4",
        ("FiLMDelta", "kinase"): "#08306b",
        ("DirectDeltaMLP", "none"): "#d62728",
        ("DirectDeltaMLP", "kinase"): "#67000d",
        ("DirectAbsoluteMLP", "none"): "#2ca02c",
        ("DirectAbsoluteMLP", "kinase"): "#00441b",
    }
    x = np.arange(len(splits))

    for i, (m, ylabel) in enumerate(panels):
        ax = axes[i]
        offset = 0
        for pi, p in enumerate(predictors):
            for pti, pt in enumerate(pretrains):
                ys, errs = [], []
                for s in splits:
                    r = next((rec for rec in summary
                              if rec["split"] == s and rec["predictor"] == p
                              and rec["pretrain"] == pt), None)
                    if r is None:
                        ys.append(np.nan); errs.append(0)
                    else:
                        ys.append(r[f"{m}_mean"]); errs.append(r[f"{m}_std"])
                xs = x + (offset - 2.5) * bar_w
                label = f"{p} ({pt})" if i == 0 else None
                ax.bar(xs, ys, bar_w, yerr=errs, capsize=2,
                       label=label, color=color_map[(p, pt)])
                offset += 1
        ax.set_xticks(x)
        ax.set_xticklabels(splits, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, axis="y")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("EXP5b ZAP70 full eval — 2 splits × 3 predictors × 2 pretrain regimes",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
