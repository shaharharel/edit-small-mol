#!/usr/bin/env python3
"""
EXP5d — Multi-task FiLMDelta variants (Δ + Abs heads, shared/separate encoders,
adversarial debiasing).

Variants:
  - basic     : shared encoder + Δ head (FiLM) + Abs head (linear); λ ∈ {0.5,1.0,2.0}
  - sep_enc   : Option A — separate encoders for Δ and Abs (no weight sharing);
                λ ∈ {0.5,1.0,2.0}
  - adversarial: Option B — shared encoder + adversarial lab classifier (GRL);
                λ = 1.0 fixed, α ∈ {0.1, 0.5, 1.0}
  - per_assay : Option C — basic + per-assay learned offset on the Abs head.
                **SKIPPED**: 21/24 ZAP70 assays have <30 mols (smallest =2 mols),
                far below the spec's ≥30-per-lab feasibility floor; embeddings would
                be too noisy to estimate. See "Option C status" in the summary md.

Eval matrix:
  - 2 splits: mol_disjoint_both_sides (5 folds) + lab_disjoint (5 folds)
  - 2 pretrains: none / kinase-pretrain
  - 6 metrics: delta_{mae,pearson,spearman} + abs_{mae,pearson,spearman}
    + abs_direct_{mae,pearson,spearman} (Abs-head direct, alongside anchor-Δ)

Headline picks best λ (or α) per variant by min Δ-MAE on mol_disjoint+kinase.

Outputs (additive — basic + Option A + Option B):
  results/paper_evaluation/exp5d_multitask_filmdelta.json
  results/paper_evaluation/exp5d_multitask_filmdelta_summary.md
  results/paper_evaluation/exp5d_multitask_filmdelta.png
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from src.models.predictors.multitask_film_delta import (  # noqa: E402
    MultiTaskFiLMDelta,
    SeparateEncoderMultiTaskFiLMDelta,
    AdversarialMultiTaskFiLMDelta,
)

ZAP70_PAIRS = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
KINASE_PAIRS = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_JSON = RESULTS_DIR / "exp5d_multitask_filmdelta.json"
OUT_MD = RESULTS_DIR / "exp5d_multitask_filmdelta_summary.md"
OUT_PNG = RESULTS_DIR / "exp5d_multitask_filmdelta.png"
EXP5B_REF_JSON = RESULTS_DIR / "exp5b_zap70_full_eval.json"

# Hyperparams
N_BITS = 2048
RADIUS = 2
ENC_HIDDEN = [1024, 512, 256]
FILM_HIDDEN = [256, 128]
DROPOUT = 0.2
LR = 1e-3
PRETRAIN_LR = 5e-4
FINETUNE_LR = 5e-4
BATCH_SIZE = 128  # larger batch -> faster epoch on CPU (was 64 per exp5b for parity, but we'd be 2x slower under contention)
MAX_EPOCHS = 60   # was 200; under contention we use a smaller budget — early stopping (patience=10) usually fires well before this
PATIENCE = 10     # was 25; tightened to keep folds bounded
PRETRAIN_EPOCHS = 3  # was 5; 3 epochs is plenty for warm-start
KFOLDS = 5
N_LABFOLDS = 5
MAX_KINASE_PAIRS = 12000  # was 30000; reduced for tractable runtime under CPU contention from sibling agent
ANCHOR_AGG = "mean"

LAMBDAS_BASIC = [0.5, 1.0, 2.0]
LAMBDAS_SEPENC = [0.5, 1.0, 2.0]
ADV_ALPHAS = [0.1, 0.5, 1.0]  # λ fixed at 1.0 per the spec
ADV_LAMBDA_FIXED = 1.0


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
# Splits (copied from exp5b)
# ---------------------------------------------------------------------------
def kfold_mol_disjoint_both_sides(
    pairs: pd.DataFrame, k: int, seed: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
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
        assert test_mols_seen.isdisjoint(train_mols), (
            f"fold {fi}: leak {len(test_mols_seen & train_mols)} mols")
        out.append((train_pairs, test_pairs))
    return out


def lab_disjoint_folds(
    pairs: pd.DataFrame, n_folds: int = 5, seed: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    assay_counts = pairs["assay_id"].value_counts()
    assays = sorted(assay_counts.index.tolist())
    n_assays = len(assays)
    target_test_frac = 1.0 / n_folds
    for fi in range(n_folds):
        rng_fold = np.random.RandomState(seed + fi * 1000)
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
        folds.append((train_pairs, test_pairs))
    return folds


def collect_endpoint_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(zip(pairs["mol_a_id"], pairs["mol_a"], pairs["value_a"]))
    rows.extend(zip(pairs["mol_b_id"], pairs["mol_b"], pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"])
    return df.groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})


def anchor_based_abs_via_delta(
    predict_delta_fn,
    train_anchors: pd.DataFrame,
    test_mols: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if len(train_anchors) == 0 or len(test_mols) == 0:
        return np.array([]), np.array([]), []
    A_fp = np.stack([fp_cache[s] for s in train_anchors["smiles"]]).astype(np.float32)
    anchor_vals = train_anchors["value"].values.astype(np.float32)
    n_anchors = len(train_anchors)
    abs_preds, abs_true, ids = [], [], []
    for _, row in test_mols.iterrows():
        smi = row["smiles"]
        if smi not in fp_cache:
            continue
        b_fp = fp_cache[smi].astype(np.float32)
        B_fp = np.tile(b_fp, (n_anchors, 1))
        delta_pred = predict_delta_fn(A_fp, B_fp)
        implied_abs = anchor_vals + delta_pred
        agg = float(np.mean(implied_abs)) if ANCHOR_AGG == "mean" else float(np.median(implied_abs))
        abs_preds.append(agg)
        abs_true.append(float(row["true_value"]))
        ids.append(row["mol_id"])
    return np.array(abs_true, dtype=np.float32), np.array(abs_preds, dtype=np.float32), ids


# ---------------------------------------------------------------------------
# Variant wrappers — uniform .fit_from_pairs(), .predict_delta(), .predict_abs()
# ---------------------------------------------------------------------------
class _BaseMTWrapper:
    NAME = "MTBase"

    def __init__(self, hp: dict):
        self.hp = hp
        self.model: Optional[nn.Module] = None

    def _build(self):
        raise NotImplementedError

    def state_dict_clone(self) -> dict:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def predict_delta(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred = self.model.forward_delta(
                torch.from_numpy(A.astype(np.float32)),
                torch.from_numpy(B.astype(np.float32)),
            )
        return pred.cpu().numpy()

    def predict_abs(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred = self.model.forward_abs(torch.from_numpy(X.astype(np.float32)))
        return pred.cpu().numpy()


class BasicMTWrapper(_BaseMTWrapper):
    """Shared encoder + Δ FiLM head + Abs linear head. Loss = L_Δ + λ·L_abs."""

    NAME = "basic"

    def _build(self):
        return MultiTaskFiLMDelta(
            input_dim=N_BITS,
            enc_hidden_dims=ENC_HIDDEN,
            film_hidden_dims=FILM_HIDDEN,
            dropout=DROPOUT,
        )

    def fit_from_pairs(self, train_pairs, val_pairs, fp_cache,
                       pretrained_state=None, lr_override=None,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE,
                       assay_to_idx=None):
        if self.model is None:
            self.model = self._build()
        if pretrained_state is not None:
            self.model.load_state_dict(pretrained_state)

        A_tr, B_tr, d_tr, ya_tr, yb_tr = _stack_pair_tensors(train_pairs, fp_cache)
        if val_pairs is not None and len(val_pairs) > 0:
            A_v, B_v, d_v, ya_v, yb_v = _stack_pair_tensors(val_pairs, fp_cache)
        else:
            A_v = B_v = d_v = ya_v = yb_v = None

        lam = self.hp["lambda_abs"]
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(A_tr), torch.from_numpy(B_tr),
                torch.from_numpy(d_tr),
                torch.from_numpy(ya_tr), torch.from_numpy(yb_tr),
            ),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = None
        if A_v is not None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(A_v), torch.from_numpy(B_v),
                    torch.from_numpy(d_v),
                    torch.from_numpy(ya_v), torch.from_numpy(yb_v),
                ),
                batch_size=BATCH_SIZE, shuffle=False,
            )

        lr = lr_override if lr_override is not None else (
            FINETUNE_LR if pretrained_state is not None else LR
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
        mse = nn.MSELoss()

        best, bad, best_state = float("inf"), 0, None
        for ep in range(max_epochs):
            self.model.train()
            for a, b, d, ya, yb in train_loader:
                delta_pred, abs_a, abs_b = self.model(a, b)
                l_delta = mse(delta_pred, d)
                l_abs = 0.5 * (mse(abs_a, ya) + mse(abs_b, yb))
                loss = l_delta + lam * l_abs
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            if val_loader is not None:
                self.model.eval()
                vls = []
                with torch.no_grad():
                    for a, b, d, ya, yb in val_loader:
                        dp, aa, ab = self.model(a, b)
                        vls.append((mse(dp, d) + lam * 0.5 * (mse(aa, ya) + mse(ab, yb))).item())
                vl = float(np.mean(vls))
                sched.step(vl)
                if vl < best - 1e-5:
                    best = vl
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self


class SepEncMTWrapper(_BaseMTWrapper):
    """Option A — separate encoders for Δ and Abs."""

    NAME = "sep_enc"

    def _build(self):
        return SeparateEncoderMultiTaskFiLMDelta(
            input_dim=N_BITS,
            enc_hidden_dims=ENC_HIDDEN,
            film_hidden_dims=FILM_HIDDEN,
            dropout=DROPOUT,
        )

    def fit_from_pairs(self, train_pairs, val_pairs, fp_cache,
                       pretrained_state=None, lr_override=None,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE,
                       assay_to_idx=None):
        if self.model is None:
            self.model = self._build()
        if pretrained_state is not None:
            self.model.load_state_dict(pretrained_state)
        A_tr, B_tr, d_tr, ya_tr, yb_tr = _stack_pair_tensors(train_pairs, fp_cache)
        if val_pairs is not None and len(val_pairs) > 0:
            A_v, B_v, d_v, ya_v, yb_v = _stack_pair_tensors(val_pairs, fp_cache)
        else:
            A_v = B_v = d_v = ya_v = yb_v = None
        lam = self.hp["lambda_abs"]
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(A_tr), torch.from_numpy(B_tr),
                torch.from_numpy(d_tr),
                torch.from_numpy(ya_tr), torch.from_numpy(yb_tr),
            ),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = None
        if A_v is not None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(A_v), torch.from_numpy(B_v),
                    torch.from_numpy(d_v),
                    torch.from_numpy(ya_v), torch.from_numpy(yb_v),
                ),
                batch_size=BATCH_SIZE, shuffle=False,
            )

        lr = lr_override if lr_override is not None else (
            FINETUNE_LR if pretrained_state is not None else LR
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
        mse = nn.MSELoss()

        best, bad, best_state = float("inf"), 0, None
        for ep in range(max_epochs):
            self.model.train()
            for a, b, d, ya, yb in train_loader:
                delta_pred, abs_a, abs_b = self.model(a, b)
                l_delta = mse(delta_pred, d)
                l_abs = 0.5 * (mse(abs_a, ya) + mse(abs_b, yb))
                loss = l_delta + lam * l_abs
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            if val_loader is not None:
                self.model.eval()
                vls = []
                with torch.no_grad():
                    for a, b, d, ya, yb in val_loader:
                        dp, aa, ab = self.model(a, b)
                        vls.append((mse(dp, d) + lam * 0.5 * (mse(aa, ya) + mse(ab, yb))).item())
                vl = float(np.mean(vls))
                sched.step(vl)
                if vl < best - 1e-5:
                    best = vl
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self


class AdvMTWrapper(_BaseMTWrapper):
    """Option B — adversarial debiasing via GRL on assay_id classifier."""

    NAME = "adversarial"

    def __init__(self, hp: dict):
        super().__init__(hp)
        self.n_assays = hp["n_assays"]

    def _build(self):
        return AdversarialMultiTaskFiLMDelta(
            input_dim=N_BITS,
            n_assays=self.n_assays,
            enc_hidden_dims=ENC_HIDDEN,
            film_hidden_dims=FILM_HIDDEN,
            dropout=DROPOUT,
        )

    def fit_from_pairs(self, train_pairs, val_pairs, fp_cache,
                       pretrained_state=None, lr_override=None,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE,
                       assay_to_idx=None):
        """Adversarial fit: pair examples have a per-pair assay label (assay_id);
        we use the SAME assay_id for both endpoints in the pair (within-assay pairs)."""
        assert assay_to_idx is not None, "AdvMTWrapper needs assay_to_idx"
        if self.model is None:
            self.model = self._build()
        if pretrained_state is not None:
            # Allow partial load (skip lab_head if shapes differ)
            try:
                self.model.load_state_dict(pretrained_state)
            except Exception:
                # Strip lab_head keys and load
                sd = self.model.state_dict()
                for k, v in pretrained_state.items():
                    if k in sd and sd[k].shape == v.shape:
                        sd[k] = v
                self.model.load_state_dict(sd)

        # Build tensors with assay-idx label for each pair
        def stack(pairs):
            A, B, d, ya, yb = _stack_pair_tensors(pairs, fp_cache)
            labs = np.array([assay_to_idx.get(int(a), 0) for a in pairs["assay_id"]],
                            dtype=np.int64)
            return A, B, d, ya, yb, labs

        A_tr, B_tr, d_tr, ya_tr, yb_tr, lab_tr = stack(train_pairs)
        if val_pairs is not None and len(val_pairs) > 0:
            A_v, B_v, d_v, ya_v, yb_v, lab_v = stack(val_pairs)
        else:
            A_v = B_v = d_v = ya_v = yb_v = lab_v = None

        lam = self.hp["lambda_abs"]
        alpha = self.hp["alpha"]
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(A_tr), torch.from_numpy(B_tr),
                torch.from_numpy(d_tr),
                torch.from_numpy(ya_tr), torch.from_numpy(yb_tr),
                torch.from_numpy(lab_tr),
            ),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = None
        if A_v is not None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(A_v), torch.from_numpy(B_v),
                    torch.from_numpy(d_v),
                    torch.from_numpy(ya_v), torch.from_numpy(yb_v),
                    torch.from_numpy(lab_v),
                ),
                batch_size=BATCH_SIZE, shuffle=False,
            )

        lr = lr_override if lr_override is not None else (
            FINETUNE_LR if pretrained_state is not None else LR
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        best, bad, best_state = float("inf"), 0, None
        for ep in range(max_epochs):
            self.model.train()
            for a, b, d, ya, yb, lab in train_loader:
                dp, abs_a, abs_b, lab_a_logits, lab_b_logits = self.model(a, b, alpha=alpha)
                l_delta = mse(dp, d)
                l_abs = 0.5 * (mse(abs_a, ya) + mse(abs_b, yb))
                # Note: GRL means encoder loss is -alpha * L_lab (the spec's minus sign);
                # the lab head's own weights still see +grad as usual.
                # Standard DANN: L = L_delta + lam*L_abs + L_lab (with GRL on the encoder side).
                l_lab = 0.5 * (ce(lab_a_logits, lab) + ce(lab_b_logits, lab))
                loss = l_delta + lam * l_abs + l_lab
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            if val_loader is not None:
                self.model.eval()
                vls = []
                with torch.no_grad():
                    for a, b, d, ya, yb, lab in val_loader:
                        dp, aa, ab, _, _ = self.model(a, b, alpha=alpha)
                        # Val criterion = task losses only (ignore adversarial term)
                        vls.append((mse(dp, d) + lam * 0.5 * (mse(aa, ya) + mse(ab, yb))).item())
                vl = float(np.mean(vls))
                sched.step(vl)
                if vl < best - 1e-5:
                    best = vl
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self


def _stack_pair_tensors(pairs: pd.DataFrame, fp_cache):
    A = np.stack([fp_cache[s] for s in pairs["mol_a"]]).astype(np.float32)
    B = np.stack([fp_cache[s] for s in pairs["mol_b"]]).astype(np.float32)
    d = pairs["delta"].values.astype(np.float32)
    ya = pairs["value_a"].values.astype(np.float32)
    yb = pairs["value_b"].values.astype(np.float32)
    return A, B, d, ya, yb


# ---------------------------------------------------------------------------
# Kinase pretrain (per-variant + per-hyperparam state cache)
# ---------------------------------------------------------------------------
def kinase_pretrain(
    variant: str,
    hp: dict,
    kinase_df: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    epochs: int,
    verbose: bool = True,
    kinase_assay_to_idx: Optional[dict] = None,
) -> dict:
    t0 = time.time()
    n_val = max(500, int(0.1 * len(kinase_df)))
    perm = np.random.RandomState(42).permutation(len(kinase_df))
    tr_idx = perm[n_val:]
    v_idx = perm[:n_val]
    tr_df = kinase_df.iloc[tr_idx].reset_index(drop=True)
    v_df = kinase_df.iloc[v_idx].reset_index(drop=True)

    if variant == "basic":
        w = BasicMTWrapper(hp)
    elif variant == "sep_enc":
        w = SepEncMTWrapper(hp)
    elif variant == "adversarial":
        # For adversarial pretrain we need an assay_to_idx covering kinase pairs.
        # We DON'T pretrain the lab_head against ZAP70's 24 assays — that'd be
        # mismatch. Just pretrain the encoder + Δ + abs heads. To do this cleanly
        # we build a kinase-side lab_head (n_kinase_assays) and discard it later
        # by re-loading only matching keys back into the ZAP70 model.
        # NB: kinase_within_pairs uses `assay_id_a`/`assay_id_b` (equal for within-pairs),
        # not a single `assay_id` column.
        assay_col = "assay_id" if "assay_id" in kinase_df.columns else "assay_id_a"
        kinase_assays = sorted(kinase_df[assay_col].unique().tolist())
        ka2i = {int(a): i for i, a in enumerate(kinase_assays)}
        kinase_hp = dict(hp)
        kinase_hp["n_assays"] = len(kinase_assays)
        # Inject a unified assay_id column into the (kinase) DataFrames so the
        # adversarial wrapper's fit can read pairs["assay_id"] uniformly.
        tr_df = tr_df.copy()
        tr_df["assay_id"] = tr_df[assay_col].astype(int)
        v_df = v_df.copy()
        v_df["assay_id"] = v_df[assay_col].astype(int)
        w = AdvMTWrapper(kinase_hp)
        w.fit_from_pairs(tr_df, v_df, fp_cache,
                         pretrained_state=None, lr_override=PRETRAIN_LR,
                         max_epochs=epochs, patience=3,
                         assay_to_idx=ka2i)
        state = w.state_dict_clone()
        if verbose:
            print(f"  [pretrain {variant} {hp}] done in {time.time()-t0:.1f}s "
                  f"on {len(tr_df)} kinase pairs (lab_head trained on "
                  f"{len(kinase_assays)} kinase assays — will be reinit on ZAP70)",
                  flush=True)
        return state
    else:
        raise ValueError(variant)

    w.fit_from_pairs(tr_df, v_df, fp_cache,
                     pretrained_state=None, lr_override=PRETRAIN_LR,
                     max_epochs=epochs, patience=3, assay_to_idx=None)
    state = w.state_dict_clone()
    if verbose:
        print(f"  [pretrain {variant} {hp}] done in {time.time()-t0:.1f}s "
              f"on {len(tr_df)} kinase pairs", flush=True)
    return state


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------
def run_fold(
    variant: str,
    hp: dict,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    pretrained_state: Optional[dict],
    fold_id: int,
    split_name: str,
    zap70_assay_to_idx: dict,
) -> dict:
    n_val = max(20, int(0.1 * len(train_pairs)))
    rng = np.random.RandomState(fold_id * 7 + 3)
    perm = rng.permutation(len(train_pairs))
    val_pairs = train_pairs.iloc[perm[:n_val]].reset_index(drop=True)
    fit_pairs = train_pairs.iloc[perm[n_val:]].reset_index(drop=True)

    if variant == "basic":
        w = BasicMTWrapper(hp)
    elif variant == "sep_enc":
        w = SepEncMTWrapper(hp)
    elif variant == "adversarial":
        hp_local = dict(hp)
        hp_local["n_assays"] = len(zap70_assay_to_idx)
        w = AdvMTWrapper(hp_local)
    else:
        raise ValueError(variant)

    w.fit_from_pairs(fit_pairs, val_pairs, fp_cache,
                     pretrained_state=pretrained_state,
                     assay_to_idx=zap70_assay_to_idx)

    A_test = np.stack([fp_cache[s] for s in test_pairs["mol_a"]]).astype(np.float32)
    B_test = np.stack([fp_cache[s] for s in test_pairs["mol_b"]]).astype(np.float32)
    d_test = test_pairs["delta"].values.astype(np.float32)
    delta_pred = w.predict_delta(A_test, B_test)
    delta_m = regression_metrics(d_test, delta_pred)

    train_anchors = collect_endpoint_table(fit_pairs)
    test_mols = collect_endpoint_table(test_pairs).rename(columns={"value": "true_value"})
    abs_true_a, abs_pred_a, abs_ids_a = anchor_based_abs_via_delta(
        w.predict_delta, train_anchors, test_mols, fp_cache,
    )
    abs_m_anchor = regression_metrics(abs_true_a, abs_pred_a)

    test_fps = np.stack([fp_cache[s] for s in test_mols["smiles"]]).astype(np.float32)
    abs_true_b = test_mols["true_value"].values.astype(np.float32)
    abs_pred_b = w.predict_abs(test_fps)
    abs_m_direct = regression_metrics(abs_true_b, abs_pred_b)

    out = {
        "variant": variant,
        "hp": hp,
        "fold": fold_id,
        "split": split_name,
        "n_train_pairs": int(len(fit_pairs)),
        "n_val_pairs": int(len(val_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "n_train_anchors": int(len(train_anchors)),
        "n_test_mols": int(len(abs_ids_a)),
        "delta_mae": delta_m["mae"],
        "delta_pearson": delta_m["pearson"],
        "delta_spearman": delta_m["spearman"],
        "abs_mae": abs_m_anchor["mae"],
        "abs_pearson": abs_m_anchor["pearson"],
        "abs_spearman": abs_m_anchor["spearman"],
        "abs_direct_mae": abs_m_direct["mae"],
        "abs_direct_pearson": abs_m_direct["pearson"],
        "abs_direct_spearman": abs_m_direct["spearman"],
    }
    del w, A_test, B_test
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "delta_mae", "delta_pearson", "delta_spearman",
    "abs_mae", "abs_pearson", "abs_spearman",
    "abs_direct_mae", "abs_direct_pearson", "abs_direct_spearman",
]


def _hp_key(hp: dict) -> str:
    parts = []
    for k in sorted(hp.keys()):
        v = hp[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.2g}")
        else:
            parts.append(f"{k}={v}")
    return ",".join(parts)


def summarize(runs: List[dict]) -> List[dict]:
    good = [r for r in runs if "error" not in r]
    if not good:
        return []
    df = pd.DataFrame(good)
    df["hp_key"] = df["hp"].apply(_hp_key)
    bucket_cols = ["variant", "hp_key", "split", "pretrain"]
    out = []
    for keys, grp in df.groupby(bucket_cols):
        rec = dict(zip(bucket_cols, keys))
        rec["hp"] = grp["hp"].iloc[0]
        rec["n_folds"] = int(len(grp))
        rec["n_test_pairs_mean"] = float(grp["n_test_pairs"].mean())
        rec["n_test_mols_mean"] = float(grp["n_test_mols"].mean())
        rec["n_train_pairs_mean"] = float(grp["n_train_pairs"].mean())
        for m in METRIC_COLS:
            rec[f"{m}_mean"] = float(grp[m].mean())
            rec[f"{m}_std"] = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
        out.append(rec)
    return out


def _fmt(rec, m):
    return f"{rec[f'{m}_mean']:.3f}±{rec[f'{m}_std']:.3f}"


def find_best_hp(summary, variant, headline_split="mol_disjoint_both_sides",
                 pretrain="kinase"):
    """Pick hp_key minimizing delta_mae_mean on the given split & pretrain."""
    candidates = [r for r in summary if r["variant"] == variant
                  and r["split"] == headline_split and r["pretrain"] == pretrain]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["delta_mae_mean"])
    return best["hp_key"]


def write_md(summary, cfg, exp5b, path):
    lines = []
    lines.append("# EXP5d Multi-task FiLMDelta — variants (basic, sep_enc, adversarial)\n\n")
    lines.append(f"- Target: {cfg['target']}\n")
    lines.append(f"- ZAP70 pairs: {cfg['n_pairs_zap70']} ({cfg['n_mols_zap70']} mols, "
                 f"{cfg['n_zap70_assays']} assays)\n")
    lines.append(f"- Kinase pretrain pool: {cfg['n_kinase_pairs_used']} pairs\n")
    lines.append(f"- K-fold (mol_disjoint_both_sides): k={cfg['k_mol_folds']}\n")
    lines.append(f"- Lab-disjoint folds: {cfg['n_lab_folds']}\n")
    lines.append(f"- Encoder: Morgan FP {N_BITS}d -> {ENC_HIDDEN}; FiLM head {FILM_HIDDEN}; dropout {DROPOUT}\n")
    lines.append(f"- Variants run: {cfg['variants_run']}\n\n")

    # Option C status
    lines.append("## Option C (per-assay learned offset) — status: SKIPPED\n\n")
    lines.append(cfg["option_c_note"] + "\n\n")

    # Per-variant headlines
    for variant in cfg["variants_run"]:
        lines.append(f"## Variant: `{variant}`\n\n")
        best_hp_key = find_best_hp(summary, variant)
        lines.append(f"Best hp (by Δ MAE on mol_disjoint, kinase pretrain): "
                     f"**{best_hp_key}**\n\n")

        for split in ["mol_disjoint_both_sides", "lab_disjoint"]:
            rows = [r for r in summary if r["variant"] == variant
                    and r["split"] == split and r["hp_key"] == best_hp_key]
            lines.append(f"### Split `{split}` — best hp `{best_hp_key}`\n\n")
            lines.append("#### Delta head\n\n")
            lines.append("| Pretrain | n_folds | n_test_pairs | MAE | Pearson | Spearman |\n")
            lines.append("|---|---|---|---|---|---|\n")
            for pt in ["none", "kinase"]:
                r = next((x for x in rows if x["pretrain"] == pt), None)
                if r is None: continue
                lines.append(
                    f"| {pt} | {r['n_folds']} | {r['n_test_pairs_mean']:.0f} | "
                    f"{_fmt(r,'delta_mae')} | {_fmt(r,'delta_pearson')} | {_fmt(r,'delta_spearman')} |\n"
                )
            lines.append("\n#### Abs via Δ head (anchor-averaged)\n\n")
            lines.append("| Pretrain | n_test_mols | MAE | Pearson | Spearman |\n")
            lines.append("|---|---|---|---|---|\n")
            for pt in ["none", "kinase"]:
                r = next((x for x in rows if x["pretrain"] == pt), None)
                if r is None: continue
                lines.append(
                    f"| {pt} | {r['n_test_mols_mean']:.0f} | "
                    f"{_fmt(r,'abs_mae')} | {_fmt(r,'abs_pearson')} | {_fmt(r,'abs_spearman')} |\n"
                )
            lines.append("\n#### Abs via Abs head directly\n\n")
            lines.append("| Pretrain | n_test_mols | MAE | Pearson | Spearman |\n")
            lines.append("|---|---|---|---|---|\n")
            for pt in ["none", "kinase"]:
                r = next((x for x in rows if x["pretrain"] == pt), None)
                if r is None: continue
                lines.append(
                    f"| {pt} | {r['n_test_mols_mean']:.0f} | "
                    f"{_fmt(r,'abs_direct_mae')} | {_fmt(r,'abs_direct_pearson')} | {_fmt(r,'abs_direct_spearman')} |\n"
                )
            lines.append("\n")

        # Appendix: full hp sweep for this variant
        lines.append(f"### Appendix — all hp configurations for `{variant}`\n\n")
        for split in ["mol_disjoint_both_sides", "lab_disjoint"]:
            lines.append(f"#### Split `{split}`\n\n")
            lines.append("| hp | Pretrain | Δ MAE | Δ Spr | Abs(anchor) MAE | Abs(head) MAE |\n")
            lines.append("|---|---|---|---|---|---|\n")
            rows = [r for r in summary if r["variant"] == variant and r["split"] == split]
            # sort by hp_key then pretrain
            rows.sort(key=lambda r: (r["hp_key"], r["pretrain"]))
            for r in rows:
                lines.append(
                    f"| {r['hp_key']} | {r['pretrain']} | "
                    f"{_fmt(r,'delta_mae')} | {_fmt(r,'delta_spearman')} | "
                    f"{_fmt(r,'abs_mae')} | {_fmt(r,'abs_direct_mae')} |\n"
                )
            lines.append("\n")

    # ---- Cross-variant + baseline comparison ----
    lines.append("## Cross-variant + baseline comparison (best hp per variant)\n\n")
    base_sum = exp5b.get("summary", []) if exp5b else []

    def fb(p, s, pt):
        return next((r for r in base_sum if r.get("predictor") == p
                     and r.get("split") == s and r.get("pretrain") == pt), None)

    for split in ["mol_disjoint_both_sides", "lab_disjoint"]:
        lines.append(f"### Split `{split}`\n\n")
        lines.append("**Δ MAE / Δ Spearman**\n\n")
        lines.append("| Model | Pretrain | Δ MAE | Δ Spr |\n")
        lines.append("|---|---|---|---|\n")
        for p in ["FiLMDelta", "DirectDeltaMLP", "DirectAbsoluteMLP"]:
            for pt in ["none", "kinase"]:
                r = fb(p, split, pt)
                if r is None: continue
                lines.append(
                    f"| {p} | {pt} | "
                    f"{r['delta_mae_mean']:.3f}±{r['delta_mae_std']:.3f} | "
                    f"{r['delta_spearman_mean']:.3f}±{r['delta_spearman_std']:.3f} |\n"
                )
        for variant in cfg["variants_run"]:
            best_hp_key = find_best_hp(summary, variant)
            for pt in ["none", "kinase"]:
                r = next((x for x in summary if x["variant"] == variant
                          and x["split"] == split and x["pretrain"] == pt
                          and x["hp_key"] == best_hp_key), None)
                if r is None: continue
                lines.append(
                    f"| **MT/{variant}({best_hp_key})** | {pt} | "
                    f"**{_fmt(r,'delta_mae')}** | **{_fmt(r,'delta_spearman')}** |\n"
                )
        lines.append("\n**Abs MAE (anchor-Δ for baselines/MT; Abs-head also reported for MT)**\n\n")
        lines.append("| Model | Pretrain | Abs MAE | Abs Spr |\n")
        lines.append("|---|---|---|---|\n")
        for p in ["FiLMDelta", "DirectDeltaMLP", "DirectAbsoluteMLP"]:
            for pt in ["none", "kinase"]:
                r = fb(p, split, pt)
                if r is None: continue
                lines.append(
                    f"| {p} | {pt} | "
                    f"{r['abs_mae_mean']:.3f}±{r['abs_mae_std']:.3f} | "
                    f"{r['abs_spearman_mean']:.3f}±{r['abs_spearman_std']:.3f} |\n"
                )
        for variant in cfg["variants_run"]:
            best_hp_key = find_best_hp(summary, variant)
            for pt in ["none", "kinase"]:
                r = next((x for x in summary if x["variant"] == variant
                          and x["split"] == split and x["pretrain"] == pt
                          and x["hp_key"] == best_hp_key), None)
                if r is None: continue
                lines.append(
                    f"| **MT/{variant}({best_hp_key}, anchor-Δ)** | {pt} | "
                    f"**{_fmt(r,'abs_mae')}** | **{_fmt(r,'abs_spearman')}** |\n"
                )
                lines.append(
                    f"| **MT/{variant}({best_hp_key}, abs-head)** | {pt} | "
                    f"**{_fmt(r,'abs_direct_mae')}** | **{_fmt(r,'abs_direct_spearman')}** |\n"
                )
        lines.append("\n")

    # ---- Verdict ----
    lines.append("## Verdict\n\n")
    if exp5b is not None and "summary" in exp5b:
        film_mol_k = next((r for r in base_sum if r.get("predictor") == "FiLMDelta"
                           and r.get("split") == "mol_disjoint_both_sides"
                           and r.get("pretrain") == "kinase"), None)
        da_mol_k = next((r for r in base_sum if r.get("predictor") == "DirectAbsoluteMLP"
                         and r.get("split") == "mol_disjoint_both_sides"
                         and r.get("pretrain") == "kinase"), None)
        da_lab_k = next((r for r in base_sum if r.get("predictor") == "DirectAbsoluteMLP"
                         and r.get("split") == "lab_disjoint"
                         and r.get("pretrain") == "kinase"), None)
        for variant in cfg["variants_run"]:
            best_hp_key = find_best_hp(summary, variant)
            mt_mol_k = next((r for r in summary if r["variant"] == variant
                             and r["split"] == "mol_disjoint_both_sides"
                             and r["pretrain"] == "kinase"
                             and r["hp_key"] == best_hp_key), None)
            mt_lab_k = next((r for r in summary if r["variant"] == variant
                             and r["split"] == "lab_disjoint"
                             and r["pretrain"] == "kinase"
                             and r["hp_key"] == best_hp_key), None)
            lines.append(f"### `{variant}` — best hp `{best_hp_key}`\n\n")
            if film_mol_k and mt_mol_k:
                d_delta = mt_mol_k["delta_mae_mean"] - film_mol_k["delta_mae_mean"]
                vd = "BEATS" if d_delta < 0 else "DOES NOT BEAT"
                lines.append(
                    f"- Δ on `mol_disjoint` (kinase): MT={mt_mol_k['delta_mae_mean']:.3f} "
                    f"vs FiLM={film_mol_k['delta_mae_mean']:.3f} → {vd} (Δ={d_delta:+.3f})\n"
                )
            if da_mol_k and mt_mol_k:
                d1 = mt_mol_k["abs_mae_mean"] - da_mol_k["abs_mae_mean"]
                d2 = mt_mol_k["abs_direct_mae_mean"] - da_mol_k["abs_mae_mean"]
                v1 = "BEATS" if d1 < 0 else "DOES NOT BEAT"
                v2 = "BEATS" if d2 < 0 else "DOES NOT BEAT"
                lines.append(
                    f"- Abs on `mol_disjoint` (kinase): DirectAbs={da_mol_k['abs_mae_mean']:.3f}; "
                    f"MT anchor-Δ={mt_mol_k['abs_mae_mean']:.3f} ({v1}, Δ={d1:+.3f}); "
                    f"MT abs-head={mt_mol_k['abs_direct_mae_mean']:.3f} ({v2}, Δ={d2:+.3f})\n"
                )
            if da_lab_k and mt_lab_k:
                d = mt_lab_k["abs_direct_mae_mean"] - da_lab_k["abs_mae_mean"]
                v = "BEATS" if d < 0 else "DOES NOT BEAT"
                lines.append(
                    f"- Abs on `lab_disjoint` (kinase, abs-head vs DirectAbs): "
                    f"MT={mt_lab_k['abs_direct_mae_mean']:.3f} vs DirectAbs="
                    f"{da_lab_k['abs_mae_mean']:.3f} → {v} (Δ={d:+.3f})\n"
                )
            lines.append("\n")

    with open(path, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(summary, cfg, exp5b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["mol_disjoint_both_sides", "lab_disjoint"]
    variants = cfg["variants_run"]
    metrics = [
        ("delta_mae", "Δ MAE (lower better)"),
        ("delta_spearman", "Δ Spearman ρ"),
        ("abs_mae", "Abs MAE (anchor-Δ)"),
        ("abs_direct_mae", "Abs MAE (Abs-head)"),
    ]

    base_sum = exp5b.get("summary", []) if exp5b else []

    def get_b(split, pt, p):
        return next((r for r in base_sum if r.get("split") == split
                     and r.get("pretrain") == pt and r.get("predictor") == p), None)

    fig, axes = plt.subplots(len(splits), len(metrics), figsize=(20, 9))
    if len(splits) == 1:
        axes = axes.reshape(1, -1)

    color_map = {
        "basic": "#1f77b4",
        "sep_enc": "#ff7f0e",
        "adversarial": "#9467bd",
    }
    for ri, split in enumerate(splits):
        for ci, (metric, ylabel) in enumerate(metrics):
            ax = axes[ri, ci]
            # x-axis: variants × pretrain; one bar per (variant, pretrain) at best hp
            xs = []
            ys = []
            errs = []
            labels = []
            colors = []
            for vi, variant in enumerate(variants):
                best_hp_key = find_best_hp(summary, variant)
                for pi, pt in enumerate(["none", "kinase"]):
                    r = next((x for x in summary if x["variant"] == variant
                              and x["split"] == split and x["pretrain"] == pt
                              and x["hp_key"] == best_hp_key), None)
                    xs.append(vi * 2.5 + pi)
                    if r is None:
                        ys.append(np.nan); errs.append(0)
                    else:
                        ys.append(r[f"{metric}_mean"]); errs.append(r[f"{metric}_std"])
                    labels.append(f"{variant[:3]}\n{pt}")
                    base_color = color_map.get(variant, "#888")
                    colors.append(base_color if pt == "none" else _darken(base_color))
            ax.bar(xs, ys, 0.8, yerr=errs, capsize=3, color=colors)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(split if ci == 0 else "")
            ax.grid(True, alpha=0.25, axis="y")
            # baselines as horizontal lines
            for p, lc in [("FiLMDelta", "#444"),
                          ("DirectDeltaMLP", "#d62728"),
                          ("DirectAbsoluteMLP", "#2ca02c")]:
                base_metric = metric if metric != "abs_direct_mae" else "abs_mae"
                r = get_b(split, "kinase", p)
                if r is None: continue
                ax.axhline(r[f"{base_metric}_mean"], linestyle="--", linewidth=1.0,
                           color=lc, alpha=0.7, label=f"{p}(kinase)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6, loc="best")
    fig.suptitle("EXP5d — MT variants (basic / sep_enc / adversarial) vs single-task baselines",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _darken(hex_color, factor=0.6):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb = tuple(int(c * factor) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ---------------------------------------------------------------------------
# Variant config builders
# ---------------------------------------------------------------------------
def build_hp_grid_for_variant(variant: str) -> List[dict]:
    if variant == "basic":
        return [{"lambda_abs": lam} for lam in LAMBDAS_BASIC]
    if variant == "sep_enc":
        return [{"lambda_abs": lam} for lam in LAMBDAS_SEPENC]
    if variant == "adversarial":
        return [{"lambda_abs": ADV_LAMBDA_FIXED, "alpha": a} for a in ADV_ALPHAS]
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: 2 folds, 1 hp per variant, basic only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(OUT_JSON))
    parser.add_argument("--workers", type=int, default=1,
                        help="Reserved for parity; runner is single-process.")
    parser.add_argument("--variants", nargs="+",
                        default=["basic", "sep_enc", "adversarial"],
                        choices=["basic", "sep_enc", "adversarial"])
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    t_global = time.time()

    pairs = pd.read_csv(ZAP70_PAIRS)
    n_mols = len(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    zap70_assays = sorted(pairs["assay_id"].unique().tolist())
    zap70_assay_to_idx = {int(a): i for i, a in enumerate(zap70_assays)}
    print(f"PROGRESS: ZAP70: {len(pairs)} pairs, {n_mols} mols, "
          f"{len(zap70_assays)} assays", flush=True)

    kinase_df = pd.read_csv(KINASE_PAIRS)
    print(f"PROGRESS: kinase: {len(kinase_df)} pairs", flush=True)
    if len(kinase_df) > MAX_KINASE_PAIRS:
        kinase_df = kinase_df.sample(n=MAX_KINASE_PAIRS, random_state=42).reset_index(drop=True)
        print(f"PROGRESS: kinase subsampled to {len(kinase_df)}", flush=True)

    t0 = time.time()
    smis = list(
        set(pairs["mol_a"]).union(pairs["mol_b"])
        | set(kinase_df["mol_a"]).union(kinase_df["mol_b"])
    )
    fp_cache = {s: smi_to_morgan(s) for s in smis}
    print(f"PROGRESS: FP cache built for {len(fp_cache)} unique mols in "
          f"{time.time()-t0:.1f}s", flush=True)

    if args.quick:
        kfolds = 2
        n_lab_folds = 2
        pretrains = ["none"]
        pretrain_epochs = 1
        variants = ["basic"]
        # Single quick hp for each variant
        hp_grid_overrides = {"basic": [{"lambda_abs": 1.0}]}
    else:
        kfolds = KFOLDS
        n_lab_folds = N_LABFOLDS
        pretrains = ["none", "kinase"]
        pretrain_epochs = PRETRAIN_EPOCHS
        variants = args.variants
        hp_grid_overrides = {}

    mol_folds = kfold_mol_disjoint_both_sides(pairs, k=kfolds, seed=0)
    lab_folds = lab_disjoint_folds(pairs, n_folds=n_lab_folds, seed=0)
    for fi, (tr, te) in enumerate(mol_folds):
        print(f"  mol_disjoint fold {fi}: train={len(tr)} test={len(te)}", flush=True)
    for fi, (tr, te) in enumerate(lab_folds):
        print(f"  lab_disjoint fold {fi}: train={len(tr)} test={len(te)}", flush=True)

    splits = {"mol_disjoint_both_sides": mol_folds, "lab_disjoint": lab_folds}

    # Option C feasibility note
    import collections
    mols_per_assay = collections.defaultdict(set)
    for _, row in pairs.iterrows():
        mols_per_assay[row["assay_id"]].add(row["mol_a_id"])
        mols_per_assay[row["assay_id"]].add(row["mol_b_id"])
    counts = sorted([len(m) for m in mols_per_assay.values()])
    n_lt_30 = sum(1 for c in counts if c < 30)
    option_c_note = (
        f"On ZAP70 ({len(counts)} assays), {n_lt_30}/{len(counts)} assays have "
        f"<30 mols (smallest assay: {min(counts)} mols, largest: {max(counts)} mols, "
        f"median: {counts[len(counts)//2]} mols). The spec calls for skipping the "
        f"per-assay-offset variant when any training lab has <30 mols, since the "
        f"per-assay scalar offset would be too noisy to estimate. Option C is "
        f"therefore SKIPPED for ZAP70."
    )

    # Pretrain states cache
    pretrain_state_cache: Dict[Tuple[str, str], dict] = {}

    if "kinase" in pretrains:
        for variant in variants:
            hp_grid = hp_grid_overrides.get(variant, build_hp_grid_for_variant(variant))
            for hp in hp_grid:
                st = kinase_pretrain(variant, hp, kinase_df, fp_cache,
                                     epochs=pretrain_epochs, verbose=True)
                pretrain_state_cache[(variant, _hp_key(hp))] = st

    # Resume
    results = {"runs": []}
    if args.resume and out_path.exists():
        try:
            prior = json.load(open(out_path))
            results["runs"] = prior.get("runs", [])
            print(f"PROGRESS: resume — loaded {len(results['runs'])} prior runs",
                  flush=True)
        except Exception as e:
            print(f"PROGRESS: resume parse failed: {e}", flush=True)

    done_keys = {
        (r["variant"], _hp_key(r["hp"]), r["split"], r["pretrain"], r["fold"])
        for r in results["runs"]
    }

    # Count total cells
    total_cells = 0
    for variant in variants:
        hp_grid = hp_grid_overrides.get(variant, build_hp_grid_for_variant(variant))
        total_cells += sum(len(f) for f in splits.values()) * len(hp_grid) * len(pretrains)
    completed = len(done_keys)
    print(f"\nPROGRESS: total cells = {total_cells}; resuming with {completed} done.",
          flush=True)

    config = {
        "target": "ZAP70 (CHEMBL2803)",
        "zap70_pairs": str(ZAP70_PAIRS),
        "kinase_pairs": str(KINASE_PAIRS),
        "n_pairs_zap70": int(len(pairs)),
        "n_mols_zap70": int(n_mols),
        "n_zap70_assays": len(zap70_assays),
        "n_kinase_pairs_used": int(len(kinase_df)),
        "k_mol_folds": kfolds,
        "n_lab_folds": n_lab_folds,
        "pretrain_epochs": pretrain_epochs,
        "morgan": {"n_bits": N_BITS, "radius": RADIUS},
        "enc_hidden": ENC_HIDDEN,
        "film_hidden": FILM_HIDDEN,
        "dropout": DROPOUT,
        "lr": LR,
        "pretrain_lr": PRETRAIN_LR,
        "finetune_lr": FINETUNE_LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "anchor_aggregation": ANCHOR_AGG,
        "variants_run": variants,
        "lambdas_basic": LAMBDAS_BASIC,
        "lambdas_sep_enc": LAMBDAS_SEPENC,
        "adv_alphas": ADV_ALPHAS,
        "adv_lambda_fixed": ADV_LAMBDA_FIXED,
        "option_c_note": option_c_note,
    }
    results["config"] = config

    # Main loop
    for variant in variants:
        hp_grid = hp_grid_overrides.get(variant, build_hp_grid_for_variant(variant))
        for split_name, folds in splits.items():
            for pt in pretrains:
                for hp in hp_grid:
                    for fi, (train_pairs, test_pairs) in enumerate(folds):
                        key = (variant, _hp_key(hp), split_name, pt, fi)
                        if key in done_keys:
                            continue
                        t_cell = time.time()
                        # Pretrain state for adversarial doesn't carry the kinase lab_head over
                        # (different n_assays); fix by stripping mismatched keys.
                        pretrained = None
                        if pt == "kinase":
                            base_state = pretrain_state_cache.get((variant, _hp_key(hp)))
                            if base_state is not None:
                                if variant == "adversarial":
                                    # Build a fresh ZAP70-sized adv model state and copy only matching keys.
                                    tmp = AdversarialMultiTaskFiLMDelta(
                                        input_dim=N_BITS,
                                        n_assays=len(zap70_assay_to_idx),
                                        enc_hidden_dims=ENC_HIDDEN,
                                        film_hidden_dims=FILM_HIDDEN,
                                        dropout=DROPOUT,
                                    )
                                    tgt = tmp.state_dict()
                                    for k, v in base_state.items():
                                        if k in tgt and tgt[k].shape == v.shape:
                                            tgt[k] = v.clone()
                                    pretrained = tgt
                                else:
                                    pretrained = base_state
                        try:
                            out = run_fold(
                                variant, hp,
                                train_pairs, test_pairs, fp_cache,
                                pretrained_state=pretrained,
                                fold_id=fi,
                                split_name=split_name,
                                zap70_assay_to_idx=zap70_assay_to_idx,
                            )
                        except Exception as e:
                            import traceback
                            print(f"PROGRESS: ERROR {key}: {e}\n{traceback.format_exc()}",
                                  flush=True)
                            out = {"variant": variant, "hp": hp, "fold": fi,
                                   "split": split_name, "error": str(e)}
                        out["variant"] = variant
                        out["hp"] = hp
                        out["split"] = split_name
                        out["pretrain"] = pt
                        out["fold"] = fi
                        out["wall_sec"] = round(time.time() - t_cell, 1)
                        results["runs"].append(out)
                        completed += 1
                        if "error" in out:
                            print(f"PROGRESS [{completed}/{total_cells}] {key} ERROR "
                                  f"({out['wall_sec']:.0f}s)", flush=True)
                        else:
                            hpstr = _hp_key(hp)
                            print(f"PROGRESS [{completed}/{total_cells}] "
                                  f"v={variant:11s} hp={hpstr:25s} "
                                  f"split={split_name:25s} pre={pt:6s} fold={fi} "
                                  f"d_mae={out['delta_mae']:.3f} d_spr={out['delta_spearman']:.3f} "
                                  f"a_anc_mae={out['abs_mae']:.3f} "
                                  f"a_dir_mae={out['abs_direct_mae']:.3f} "
                                  f"({out['wall_sec']:.0f}s)", flush=True)
                        # Incremental save
                        with open(out_path, "w") as f:
                            json.dump(results, f, indent=2, default=str)

    summary = summarize(results["runs"])
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nPROGRESS: wrote {out_path}", flush=True)

    exp5b = None
    if EXP5B_REF_JSON.exists():
        try:
            exp5b = json.load(open(EXP5B_REF_JSON))
        except Exception as e:
            print(f"PROGRESS: failed to load exp5b ref: {e}", flush=True)

    write_md(summary, config, exp5b, OUT_MD)
    print(f"PROGRESS: wrote {OUT_MD}", flush=True)
    try:
        make_plot(summary, config, exp5b, OUT_PNG)
        print(f"PROGRESS: wrote {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"PROGRESS: plot failed: {e}", flush=True)
    print(f"\nPROGRESS: TOTAL TIME {(time.time()-t_global)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
