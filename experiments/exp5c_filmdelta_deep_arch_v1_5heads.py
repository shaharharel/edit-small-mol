#!/usr/bin/env python3
"""
EXP5c - ZAP70 small-data deep architecture comparison.

Compares richer edit-aware architectures vs the basic Morgan-only FiLMDelta
and direct property prediction on ZAP70 within-assay MMP pairs (CHEMBL2803,
3001 pairs, 257 mols, 24 assays).

Hypothesis (verbatim from user):
  "edit delta representation should have some more power from explicitly
   representing the delta between the molecules, and seeing the delta label
   (potentially in addition to predicting absolute as well). maybe we need
   more than morgan finger prints. ... we should be able to use also the
   regular absolute label in the loss, as well as all the features /
   embedding that the single molecule model has."

Design:
  - 3 encoders:  Morgan (2048), ChemBERTa-2 MTR (384), MolFormer-XL (768)
  - 5 heads:     T1 direct-absolute, T2 FiLMDelta delta-only, T3 multi-task FiLMDelta,
                 T4 DeepDelta-style concat, T5 multi-task FiLMDelta + DRFP edit conditioning
  - 6 N:         {25, 50, 100, 200, 500, all}
  - 5 seeds

Split: mol-disjoint single-side (test mol_b never appears as mol_b in any
training pair — most LO-relevant for prospective design).

All runs use the same MLP backbone (hidden [256, 128], dropout 0.2) and
training schedule (AdamW lr=1e-3, max 200 epochs, early stop patience 25).

Outputs:
  results/paper_evaluation/exp5c_filmdelta_deep_arch_zap70.json
  results/paper_evaluation/exp5c_filmdelta_deep_arch_zap70.png
  results/paper_evaluation/exp5c_filmdelta_deep_arch_summary.md
  models/filmdelta_zap70_best.pt   (strongest config, marked in summary)
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
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PAIRS_FILE = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
EMB_DIR = PROJECT_ROOT / "data" / "exp5c_embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp5c_filmdelta_deep_arch_zap70.json"
PLOT_FILE = RESULTS_DIR / "exp5c_filmdelta_deep_arch_zap70.png"
SUMMARY_FILE = RESULTS_DIR / "exp5c_filmdelta_deep_arch_summary.md"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_CKPT = MODELS_DIR / "filmdelta_zap70_best.pt"

# Fairness: same backbone for all heads
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
LR = 1e-3
WD = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 25
TRAIN_SIZES = [25, 50, 100, 200, 500]  # "all" appended at runtime
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.20

DEVICE = "cpu"
torch.set_num_threads(max(1, os.cpu_count() // 2))


# ---------------------------------------------------------------------------
# Embedding cache builders (all run once, then reused)
# ---------------------------------------------------------------------------
def _save_emb_npz(path: Path, smiles: List[str], emb: np.ndarray):
    np.savez_compressed(path, smiles=np.array(smiles, dtype=object), embeddings=emb.astype(np.float32))
    print(f"  saved {path.name}: {emb.shape}")


def _load_emb_npz(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {s: d["embeddings"][i] for i, s in enumerate(d["smiles"])}


def build_morgan(smiles: List[str], out: Path, n_bits: int = 2048, radius: int = 2):
    if out.exists():
        print(f"  {out.name} cached.")
        return
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.DisableLog("rdApp.*")
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        X[i] = arr.astype(np.float32)
    _save_emb_npz(out, smiles, X)


def build_chemberta(smiles: List[str], out: Path, batch: int = 64):
    if out.exists():
        print(f"  {out.name} cached.")
        return
    from transformers import AutoTokenizer, AutoModel
    print("  loading ChemBERTa-2 MTR ...")
    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    mdl = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(DEVICE).eval()
    X = np.zeros((len(smiles), 384), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(smiles), batch):
            chunk = smiles[b: b + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out_h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out_h * mask).sum(1) / mask.sum(1).clamp(min=1)
            X[b: b + batch] = pooled.cpu().numpy()
    print(f"  ChemBERTa pass {time.time()-t0:.1f}s")
    _save_emb_npz(out, smiles, X)
    del mdl


def build_molformer(smiles: List[str], out: Path, batch: int = 64):
    if out.exists():
        print(f"  {out.name} cached.")
        return
    from transformers import AutoTokenizer, AutoModel
    print("  loading MolFormer-XL ...")
    tok = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        trust_remote_code=True,
        deterministic_eval=True,
    ).to(DEVICE).eval()
    X = np.zeros((len(smiles), 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(smiles), batch):
            chunk = smiles[b: b + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out_h = mdl(**enc).pooler_output
            X[b: b + batch] = out_h.cpu().numpy()
    print(f"  MolFormer pass {time.time()-t0:.1f}s")
    _save_emb_npz(out, smiles, X)
    del mdl


def build_drfp(pairs: pd.DataFrame, out: Path):
    """Build DRFP for every unique (mol_a, mol_b) pair (used for edit conditioning T5)."""
    if out.exists():
        print(f"  {out.name} cached.")
        return
    from drfp import DrfpEncoder
    rxns = [f"{a}>>{b}" for a, b in zip(pairs["mol_a"], pairs["mol_b"])]
    print(f"  building DRFP for {len(rxns)} reactions ...")
    t0 = time.time()
    X = DrfpEncoder.encode(rxns, n_folded_length=2048, radius=3, rings=True)
    X = np.asarray(X, dtype=np.float32)
    print(f"  drfp pass {time.time()-t0:.1f}s -> shape {X.shape}")
    np.savez_compressed(
        out,
        mol_a=np.array(pairs["mol_a"].tolist(), dtype=object),
        mol_b=np.array(pairs["mol_b"].tolist(), dtype=object),
        embeddings=X,
    )
    print(f"  saved {out.name}")


def load_drfp_cache(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {(a, b): d["embeddings"][i] for i, (a, b) in enumerate(zip(d["mol_a"], d["mol_b"]))}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def mol_disjoint_single_side_split(pairs: pd.DataFrame, test_frac: float, seed: int):
    """Hold out test pairs whose mol_b never appears as mol_b in any train pair.

    Implementation: pick a random subset of unique mol_b ids covering ~test_frac
    of pairs, all pairs with that mol_b go to test, rest to train.
    """
    rng = np.random.RandomState(seed)
    b_ids = pairs["mol_b_id"].unique().tolist()
    rng.shuffle(b_ids)

    counts = pairs.groupby("mol_b_id").size().to_dict()
    target = int(round(test_frac * len(pairs)))
    chosen = set()
    cum = 0
    for bid in b_ids:
        chosen.add(bid)
        cum += counts[bid]
        if cum >= target:
            break

    test_mask = pairs["mol_b_id"].isin(chosen).values
    return ~test_mask, test_mask


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MLPBackbone(nn.Module):
    """Shared trunk: in_dim -> hidden_dims -> out_dim (hidden_dims[-1])."""
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class DirectAbsModel(nn.Module):
    """T1: single-mol pIC50 regressor."""
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        self.trunk = MLPBackbone(in_dim, hidden_dims, dropout)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)


class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        nn.init.xavier_uniform_(self.gamma.weight, gain=0.1)
        nn.init.constant_(self.gamma.bias, 1.0)
        nn.init.xavier_uniform_(self.beta.weight, gain=0.1)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h, cond):
        return self.gamma(cond) * h + self.beta(cond)


class FiLMTrunk(nn.Module):
    """Shared FiLM-conditioned trunk: maps emb -> latent (conditioned on edit cond)."""
    def __init__(self, in_dim, hidden_dims, cond_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        prev = in_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)))
            self.films.append(FiLMLayer(h, cond_dim))
            prev = h
        self.out_dim = prev

    def forward(self, x, cond):
        h = x
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            h = film(h, cond)
        return h


class EditConditionEncoder(nn.Module):
    """Encodes the raw edit vector (Morgan diff, or DRFP, or concat) into cond_dim."""
    def __init__(self, in_dim, cond_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cond_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, e):
        return self.net(e)


class FiLMDeltaOnlyModel(nn.Module):
    """T2: FiLM trunk + single absolute head; loss only on delta = f(B)-f(A)."""
    def __init__(self, in_dim, hidden_dims, dropout, cond_in_dim, cond_dim=128):
        super().__init__()
        self.cond_enc = EditConditionEncoder(cond_in_dim, cond_dim, dropout)
        self.trunk = FiLMTrunk(in_dim, hidden_dims, cond_dim, dropout)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def predict_abs(self, x, edit):
        cond = self.cond_enc(edit)
        return self.head(self.trunk(x, cond)).squeeze(-1)

    def forward(self, emb_a, emb_b, edit_ab):
        cond = self.cond_enc(edit_ab)
        ha = self.trunk(emb_a, cond)
        hb = self.trunk(emb_b, cond)
        ya = self.head(ha).squeeze(-1)
        yb = self.head(hb).squeeze(-1)
        return yb - ya, ya, yb


class MultiTaskFiLMModel(nn.Module):
    """T3: shared FiLM trunk + abs head; loss = w_abs*(|y_a-ya|+|y_b-yb|) + w_delta*|delta-(yb-ya)|."""
    def __init__(self, in_dim, hidden_dims, dropout, cond_in_dim, cond_dim=128):
        super().__init__()
        self.cond_enc = EditConditionEncoder(cond_in_dim, cond_dim, dropout)
        self.trunk = FiLMTrunk(in_dim, hidden_dims, cond_dim, dropout)
        self.head_abs = nn.Linear(self.trunk.out_dim, 1)
        # Separate delta head improves performance (delta sees both reps)
        self.head_delta = nn.Sequential(
            nn.Linear(self.trunk.out_dim * 2, hidden_dims[-1]),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1),
        )

    def predict_abs(self, x, edit):
        cond = self.cond_enc(edit)
        return self.head_abs(self.trunk(x, cond)).squeeze(-1)

    def forward(self, emb_a, emb_b, edit_ab):
        cond = self.cond_enc(edit_ab)
        ha = self.trunk(emb_a, cond)
        hb = self.trunk(emb_b, cond)
        ya = self.head_abs(ha).squeeze(-1)
        yb = self.head_abs(hb).squeeze(-1)
        delta_pred = self.head_delta(torch.cat([ha, hb], dim=-1)).squeeze(-1)
        return delta_pred, ya, yb


class DeepDeltaConcatModel(nn.Module):
    """T4: DeepDelta style — concat(emb_a, emb_b) -> MLP -> delta. No FiLM, no abs head."""
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        self.trunk = MLPBackbone(in_dim * 2, hidden_dims, dropout)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, emb_a, emb_b):
        z = torch.cat([emb_a, emb_b], dim=-1)
        return self.head(self.trunk(z)).squeeze(-1)


# T5 = MultiTaskFiLMModel with cond_in_dim = morgan_diff_dim + drfp_dim (concat edit)
# (so the same MultiTaskFiLMModel class is reused, just bigger cond input)

# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------
def _to_t(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def train_direct(
    X_train, y_train, X_val, y_val,
    in_dim, hidden_dims, dropout, lr, wd, batch, max_ep, pat,
):
    """T1 trainer."""
    model = DirectAbsModel(in_dim, hidden_dims, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    Xt = _to_t(X_train); yt = _to_t(y_train)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    has_val = X_val is not None
    if has_val:
        Xv = _to_t(X_val); yv = _to_t(y_val)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(max_ep):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb.to(DEVICE))
            loss = loss_fn(pred, yb.to(DEVICE))
            loss.backward()
            opt.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(Xv.to(DEVICE)), yv.to(DEVICE)).item())
            if vl < best - 1e-5:
                best = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= pat:
                    break
    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    return model


def _pair_loaders(emb_dict_a, emb_dict_b, edit_dict, y_delta, y_a, y_b, batch, shuffle):
    """Stacks tensors for pair training."""
    A = torch.from_numpy(np.stack(emb_dict_a)).float()
    B = torch.from_numpy(np.stack(emb_dict_b)).float()
    E = torch.from_numpy(np.stack(edit_dict)).float() if edit_dict is not None else None
    yd = torch.from_numpy(y_delta).float()
    ya = torch.from_numpy(y_a).float()
    yb = torch.from_numpy(y_b).float()
    if E is None:
        ds = TensorDataset(A, B, yd, ya, yb)
    else:
        ds = TensorDataset(A, B, E, yd, ya, yb)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_filmdelta(
    A_train, B_train, E_train, dy_train, ya_train, yb_train,
    A_val, B_val, E_val, dy_val, ya_val, yb_val,
    in_dim, hidden_dims, dropout, lr, wd, batch, max_ep, pat,
    multi_task: bool, cond_in_dim: int,
):
    """T2 (multi_task=False) or T3/T5 (multi_task=True) trainer."""
    if multi_task:
        model = MultiTaskFiLMModel(in_dim, hidden_dims, dropout, cond_in_dim).to(DEVICE)
    else:
        model = FiLMDeltaOnlyModel(in_dim, hidden_dims, dropout, cond_in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    dl = _pair_loaders(A_train, B_train, E_train, dy_train, ya_train, yb_train, batch, True)
    has_val = A_val is not None
    if has_val:
        dl_val = _pair_loaders(A_val, B_val, E_val, dy_val, ya_val, yb_val, batch, False)

    W_DELTA = 1.0
    W_ABS = 0.5 if multi_task else 0.0

    best = float("inf"); best_state = None; bad = 0
    for ep in range(max_ep):
        model.train()
        for batch_data in dl:
            A, B, E, yd, ya, yb = batch_data
            A = A.to(DEVICE); B = B.to(DEVICE); E = E.to(DEVICE)
            yd = yd.to(DEVICE); ya = ya.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            d_pred, ya_pred, yb_pred = model(A, B, E)
            loss = W_DELTA * F.mse_loss(d_pred, yd)
            if multi_task:
                loss = loss + W_ABS * (F.mse_loss(ya_pred, ya) + F.mse_loss(yb_pred, yb))
            loss.backward()
            opt.step()
        if has_val:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch_data in dl_val:
                    A, B, E, yd, ya, yb = batch_data
                    A = A.to(DEVICE); B = B.to(DEVICE); E = E.to(DEVICE)
                    yd = yd.to(DEVICE); ya = ya.to(DEVICE); yb = yb.to(DEVICE)
                    d_pred, _, _ = model(A, B, E)
                    losses.append(F.mse_loss(d_pred, yd).item())
            vl = float(np.mean(losses))
            if vl < best - 1e-5:
                best = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= pat:
                    break
    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    return model


def train_deepdelta(
    A_train, B_train, dy_train, A_val, B_val, dy_val,
    in_dim, hidden_dims, dropout, lr, wd, batch, max_ep, pat,
):
    model = DeepDeltaConcatModel(in_dim, hidden_dims, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    A = torch.from_numpy(np.stack(A_train)).float()
    B = torch.from_numpy(np.stack(B_train)).float()
    yd = torch.from_numpy(dy_train).float()
    dl = DataLoader(TensorDataset(A, B, yd), batch_size=batch, shuffle=True)
    has_val = A_val is not None
    if has_val:
        Av = torch.from_numpy(np.stack(A_val)).float()
        Bv = torch.from_numpy(np.stack(B_val)).float()
        ydv = torch.from_numpy(dy_val).float()
        dl_val = DataLoader(TensorDataset(Av, Bv, ydv), batch_size=batch, shuffle=False)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(max_ep):
        model.train()
        for Ab, Bb, yb in dl:
            opt.zero_grad()
            pred = model(Ab.to(DEVICE), Bb.to(DEVICE))
            loss = F.mse_loss(pred, yb.to(DEVICE))
            loss.backward()
            opt.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                losses = []
                for Ab, Bb, yb in dl_val:
                    pred = model(Ab.to(DEVICE), Bb.to(DEVICE))
                    losses.append(F.mse_loss(pred, yb.to(DEVICE)).item())
                vl = float(np.mean(losses))
            if vl < best - 1e-5:
                best = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= pat:
                    break
    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# Eval helpers
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
        "mae": mae, "rmse": rmse, "pearson": float(pr if not np.isnan(pr) else 0),
        "spearman": float(sr if not np.isnan(sr) else 0), "r2": r2, "n": int(len(y_true))
    }


def predict_filmdelta(model, A_test, B_test, E_test):
    model.eval()
    with torch.no_grad():
        A = torch.from_numpy(np.stack(A_test)).float().to(DEVICE)
        B = torch.from_numpy(np.stack(B_test)).float().to(DEVICE)
        E = torch.from_numpy(np.stack(E_test)).float().to(DEVICE)
        d_pred, ya_pred, yb_pred = model(A, B, E)
        return d_pred.cpu().numpy(), ya_pred.cpu().numpy(), yb_pred.cpu().numpy()


def predict_direct(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy()


def predict_deepdelta(model, A, B):
    model.eval()
    with torch.no_grad():
        A_t = torch.from_numpy(np.stack(A)).float().to(DEVICE)
        B_t = torch.from_numpy(np.stack(B)).float().to(DEVICE)
        return model(A_t, B_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Build the dataset of pair embeddings
# ---------------------------------------------------------------------------
def stack_pairs(pairs: pd.DataFrame, mol_emb: Dict[str, np.ndarray],
                drfp_cache: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
                cond_kind: str = "morgan_diff",
                morgan_emb: Optional[Dict[str, np.ndarray]] = None):
    """Stack arrays needed for training/eval:
       A, B : encoder-specific mol embeddings for mol_a, mol_b
       E    : edit conditioning vector (morgan_diff | drfp | morgan_diff + drfp)
       returns (A, B, E, dy, ya, yb)
    """
    A = np.stack([mol_emb[s] for s in pairs["mol_a"]]).astype(np.float32)
    B = np.stack([mol_emb[s] for s in pairs["mol_b"]]).astype(np.float32)
    dy = pairs["delta"].values.astype(np.float32)
    ya = pairs["value_a"].values.astype(np.float32)
    yb = pairs["value_b"].values.astype(np.float32)

    if cond_kind == "morgan_diff":
        assert morgan_emb is not None
        Ma = np.stack([morgan_emb[s] for s in pairs["mol_a"]]).astype(np.float32)
        Mb = np.stack([morgan_emb[s] for s in pairs["mol_b"]]).astype(np.float32)
        E = Mb - Ma
    elif cond_kind == "drfp":
        assert drfp_cache is not None
        E = np.stack([drfp_cache[(a, b)] for a, b in zip(pairs["mol_a"], pairs["mol_b"])]).astype(np.float32)
    elif cond_kind == "morgan_diff_plus_drfp":
        assert drfp_cache is not None and morgan_emb is not None
        Ma = np.stack([morgan_emb[s] for s in pairs["mol_a"]]).astype(np.float32)
        Mb = np.stack([morgan_emb[s] for s in pairs["mol_b"]]).astype(np.float32)
        D = np.stack([drfp_cache[(a, b)] for a, b in zip(pairs["mol_a"], pairs["mol_b"])]).astype(np.float32)
        E = np.concatenate([Mb - Ma, D], axis=-1)
    else:
        raise ValueError(cond_kind)
    return A, B, E, dy, ya, yb


def build_direct_training_set(pairs: pd.DataFrame, mol_emb: Dict[str, np.ndarray]):
    """Dedup by mol_id, average pIC50 across appearances."""
    rows = []
    rows.extend(zip(pairs["mol_a_id"], pairs["mol_a"], pairs["value_a"]))
    rows.extend(zip(pairs["mol_b_id"], pairs["mol_b"], pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "pIC50"])
    agg = df.groupby("mol_id").agg({"smiles": "first", "pIC50": "mean"}).reset_index()
    X = np.stack([mol_emb[s] for s in agg["smiles"]]).astype(np.float32)
    y = agg["pIC50"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# One run = (encoder, head, N, seed) -> metrics
# ---------------------------------------------------------------------------
def run_one(
    encoder: str,
    head: str,
    N: int,
    seed: int,
    train_pool: pd.DataFrame,
    test_pairs: pd.DataFrame,
    mol_emb: Dict[str, np.ndarray],
    morgan_emb: Dict[str, np.ndarray],
    drfp_cache: Dict[Tuple[str, str], np.ndarray],
):
    rng = np.random.RandomState(seed * 1000 + N)
    pool_idx = np.arange(len(train_pool))
    rng.shuffle(pool_idx)
    use = pool_idx[: min(N, len(pool_idx))]
    fit_pairs = train_pool.iloc[use].copy().reset_index(drop=True)

    # Internal val (10% of fit, fixed)
    n_val = max(5, int(0.1 * len(fit_pairs)))
    if len(fit_pairs) >= 50:
        val_pairs = fit_pairs.iloc[-n_val:].copy()
        fit_pairs = fit_pairs.iloc[:-n_val].copy()
    else:
        val_pairs = None

    in_dim = next(iter(mol_emb.values())).shape[0]
    morgan_dim = next(iter(morgan_emb.values())).shape[0]
    drfp_dim = next(iter(drfp_cache.values())).shape[0]

    cond_in_dim = None
    if head in ("filmdelta", "multitask_film"):
        cond_in_dim = morgan_dim  # Morgan diff
    elif head == "multitask_film_drfp":
        cond_in_dim = morgan_dim + drfp_dim  # Morgan diff + DRFP

    out = {"head": head, "encoder": encoder, "N": int(N), "seed": int(seed),
           "n_fit": int(len(fit_pairs)), "n_val": int(len(val_pairs) if val_pairs is not None else 0),
           "n_test": int(len(test_pairs))}

    if head == "direct":
        # T1: train on dedup'd mol-level set, then predict delta = pred(B) - pred(A)
        X_mol, y_mol = build_direct_training_set(fit_pairs, mol_emb)
        if val_pairs is not None:
            Xv, yv = build_direct_training_set(val_pairs, mol_emb)
        else:
            Xv = yv = None
        model = train_direct(
            X_mol, y_mol, Xv, yv,
            in_dim, HIDDEN_DIMS, DROPOUT, LR, WD, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
        )
        A_test = np.stack([mol_emb[s] for s in test_pairs["mol_a"]]).astype(np.float32)
        B_test = np.stack([mol_emb[s] for s in test_pairs["mol_b"]]).astype(np.float32)
        pred_a = predict_direct(model, A_test)
        pred_b = predict_direct(model, B_test)
        d_pred = pred_b - pred_a
        d_true = test_pairs["delta"].values.astype(np.float32)
        y_abs_true = np.concatenate([test_pairs["value_a"].values, test_pairs["value_b"].values]).astype(np.float32)
        y_abs_pred = np.concatenate([pred_a, pred_b])
        out["delta_metrics"] = delta_metrics(d_true, d_pred)
        out["abs_mae"] = float(np.mean(np.abs(y_abs_true - y_abs_pred)))
        out["n_direct_train_mols"] = int(len(X_mol))

    elif head == "deepdelta":
        A_fit = np.stack([mol_emb[s] for s in fit_pairs["mol_a"]]).astype(np.float32)
        B_fit = np.stack([mol_emb[s] for s in fit_pairs["mol_b"]]).astype(np.float32)
        if val_pairs is not None:
            A_v = np.stack([mol_emb[s] for s in val_pairs["mol_a"]]).astype(np.float32)
            B_v = np.stack([mol_emb[s] for s in val_pairs["mol_b"]]).astype(np.float32)
            dy_v = val_pairs["delta"].values.astype(np.float32)
        else:
            A_v = B_v = dy_v = None
        model = train_deepdelta(
            A_fit, B_fit, fit_pairs["delta"].values.astype(np.float32),
            A_v, B_v, dy_v,
            in_dim, HIDDEN_DIMS, DROPOUT, LR, WD, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
        )
        A_t = np.stack([mol_emb[s] for s in test_pairs["mol_a"]]).astype(np.float32)
        B_t = np.stack([mol_emb[s] for s in test_pairs["mol_b"]]).astype(np.float32)
        d_pred = predict_deepdelta(model, A_t, B_t)
        out["delta_metrics"] = delta_metrics(test_pairs["delta"].values.astype(np.float32), d_pred)

    elif head in ("filmdelta", "multitask_film", "multitask_film_drfp"):
        cond_kind = "morgan_diff" if head != "multitask_film_drfp" else "morgan_diff_plus_drfp"
        A_fit, B_fit, E_fit, dy_fit, ya_fit, yb_fit = stack_pairs(
            fit_pairs, mol_emb, drfp_cache, cond_kind, morgan_emb
        )
        if val_pairs is not None:
            A_v, B_v, E_v, dy_v, ya_v, yb_v = stack_pairs(
                val_pairs, mol_emb, drfp_cache, cond_kind, morgan_emb
            )
        else:
            A_v = B_v = E_v = dy_v = ya_v = yb_v = None

        model = train_filmdelta(
            A_fit, B_fit, E_fit, dy_fit, ya_fit, yb_fit,
            A_v, B_v, E_v, dy_v, ya_v, yb_v,
            in_dim, HIDDEN_DIMS, DROPOUT, LR, WD, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
            multi_task=(head != "filmdelta"), cond_in_dim=cond_in_dim,
        )

        A_t, B_t, E_t, dy_t, ya_t, yb_t = stack_pairs(
            test_pairs, mol_emb, drfp_cache, cond_kind, morgan_emb
        )
        d_pred, ya_pred, yb_pred = predict_filmdelta(model, A_t, B_t, E_t)
        out["delta_metrics"] = delta_metrics(dy_t, d_pred)
        if head != "filmdelta":
            y_abs_true = np.concatenate([ya_t, yb_t])
            y_abs_pred = np.concatenate([ya_pred, yb_pred])
            out["abs_mae"] = float(np.mean(np.abs(y_abs_true - y_abs_pred)))
        out["_model"] = model  # so caller can save best
    else:
        raise ValueError(head)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="*", default=["morgan", "chemberta", "molformer"])
    parser.add_argument("--heads", nargs="*",
                        default=["direct", "filmdelta", "multitask_film", "deepdelta", "multitask_film_drfp"])
    parser.add_argument("--sizes", nargs="*", type=int, default=TRAIN_SIZES)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--quick", action="store_true", help="Small smoke set")
    args = parser.parse_args()

    if args.quick:
        args.sizes = [25, 100]
        args.seeds = [0, 1]

    print(f"[{time.strftime('%H:%M:%S')}] Loading data: {PAIRS_FILE}")
    pairs = pd.read_csv(PAIRS_FILE)
    print(f"  pairs: {len(pairs)}, mols: {len(set(pairs.mol_a_id).union(pairs.mol_b_id))}, assays: {pairs.assay_id.nunique()}")

    all_smiles = sorted(set(pairs.mol_a).union(pairs.mol_b))
    print(f"  unique smiles: {len(all_smiles)}")

    # --- Build encoder caches (per-mol) ---
    print(f"[{time.strftime('%H:%M:%S')}] Building encoder caches under {EMB_DIR}/ ...")
    morgan_path = EMB_DIR / "morgan.npz"
    chemberta_path = EMB_DIR / "chemberta2_mtr.npz"
    molformer_path = EMB_DIR / "molformer_xl.npz"
    drfp_path = EMB_DIR / "drfp_pair.npz"

    build_morgan(all_smiles, morgan_path)
    if "chemberta" in args.encoders:
        build_chemberta(all_smiles, chemberta_path)
    if "molformer" in args.encoders:
        build_molformer(all_smiles, molformer_path)
    build_drfp(pairs[["mol_a", "mol_b"]].drop_duplicates(), drfp_path)

    morgan_emb = _load_emb_npz(morgan_path)
    encoder_caches = {"morgan": morgan_emb}
    if "chemberta" in args.encoders:
        encoder_caches["chemberta"] = _load_emb_npz(chemberta_path)
    if "molformer" in args.encoders:
        encoder_caches["molformer"] = _load_emb_npz(molformer_path)
    drfp_cache = load_drfp_cache(drfp_path)
    print(f"  loaded {len(encoder_caches)} encoder caches; drfp pair entries: {len(drfp_cache)}")

    # --- Make splits per seed (mol-disjoint single-side) ---
    print(f"[{time.strftime('%H:%M:%S')}] Splits: mol-disjoint single-side (test mol_b distinct)")

    train_sizes = list(args.sizes) + ["all"]

    results = {
        "config": {
            "target": "ZAP70 (CHEMBL2803)",
            "pairs_file": str(PAIRS_FILE),
            "n_pairs": int(len(pairs)),
            "encoders": args.encoders,
            "heads": args.heads,
            "train_sizes": train_sizes,
            "seeds": args.seeds,
            "split": "mol_disjoint_single_side",
            "test_frac": TEST_FRAC,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "lr": LR, "wd": WD,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "w_abs": 0.5, "w_delta": 1.0,
        },
        "runs": [],
    }

    # Track best multi-task FiLM model for checkpoint
    best_delta_mae = float("inf")
    best_model_info = None

    total = len(args.encoders) * len(args.heads) * len(train_sizes) * len(args.seeds)
    done = 0
    t_start = time.time()

    # Precompute per-seed splits (so all settings share the same split per seed)
    splits = {}
    for seed in args.seeds:
        tr_mask, te_mask = mol_disjoint_single_side_split(pairs, TEST_FRAC, seed)
        splits[seed] = (pairs[tr_mask].reset_index(drop=True), pairs[te_mask].reset_index(drop=True))
        print(f"  seed={seed} train_pool={tr_mask.sum()} test={te_mask.sum()} "
              f"test_mol_b_unique={pairs[te_mask].mol_b_id.nunique()}")

    for seed in args.seeds:
        train_pool, test_pairs = splits[seed]
        for encoder in args.encoders:
            mol_emb = encoder_caches[encoder]
            for head in args.heads:
                for N in train_sizes:
                    if N == "all":
                        n_use = len(train_pool)
                    else:
                        n_use = int(N)
                        if n_use > len(train_pool):
                            done += 1
                            continue
                    t0 = time.time()
                    try:
                        out = run_one(encoder, head, n_use, seed,
                                      train_pool, test_pairs, mol_emb, morgan_emb, drfp_cache)
                        model = out.pop("_model", None)
                        out["dt"] = float(time.time() - t0)
                        out["N_label"] = str(N)
                        results["runs"].append(out)

                        # Track best for checkpoint: T3 multitask_film at full N
                        if head == "multitask_film" and N == "all":
                            mae = out["delta_metrics"]["mae"]
                            if mae < best_delta_mae:
                                best_delta_mae = mae
                                if model is not None:
                                    torch.save({
                                        "state_dict": model.state_dict(),
                                        "config": {
                                            "encoder": encoder, "head": head,
                                            "in_dim": next(iter(mol_emb.values())).shape[0],
                                            "hidden_dims": HIDDEN_DIMS,
                                            "dropout": DROPOUT,
                                            "cond_in_dim": next(iter(morgan_emb.values())).shape[0],
                                        },
                                        "metrics": out["delta_metrics"],
                                        "seed": seed,
                                    }, BEST_CKPT)
                                    best_model_info = {"encoder": encoder, "head": head, "seed": seed,
                                                       "mae": mae}

                        if done % 10 == 0 or out.get("delta_metrics", {}).get("mae", 99) < 0.65:
                            print(f"  [{done+1:3d}/{total}] enc={encoder:9s} head={head:22s} N={str(N):4s} "
                                  f"seed={seed} mae={out['delta_metrics']['mae']:.3f} "
                                  f"spr={out['delta_metrics']['spearman']:.3f} ({out['dt']:.1f}s)",
                                  flush=True)
                    except Exception as e:
                        print(f"  FAIL enc={encoder} head={head} N={N} seed={seed}: {e}")
                    done += 1
                    if done % 25 == 0:
                        # incremental save
                        with open(RESULTS_FILE, "w") as f:
                            json.dump(results, f, indent=2)
                    gc.collect()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {RESULTS_FILE} ({len(results['runs'])} runs, {time.time()-t_start:.1f}s)")

    # --- Plot ---
    try:
        plot_results(results, PLOT_FILE)
        print(f"  Wrote plot {PLOT_FILE}")
    except Exception as e:
        print(f"  PLOT FAIL: {e}")

    # --- Summary ---
    try:
        write_summary(results, SUMMARY_FILE, best_model_info)
        print(f"  Wrote summary {SUMMARY_FILE}")
    except Exception as e:
        print(f"  SUMMARY FAIL: {e}")


def plot_results(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([
        {
            "encoder": r["encoder"], "head": r["head"], "N": r["N_label"], "seed": r["seed"],
            "mae": r["delta_metrics"]["mae"], "spearman": r["delta_metrics"]["spearman"],
            "pearson": r["delta_metrics"]["pearson"], "r2": r["delta_metrics"]["r2"],
        }
        for r in results["runs"]
    ])
    encoders = results["config"]["encoders"]
    metrics = ["mae", "spearman", "r2"]
    fig, axes = plt.subplots(len(encoders), len(metrics), figsize=(4 * len(metrics), 3.2 * len(encoders)),
                             squeeze=False)
    head_colors = {"direct": "#888888", "filmdelta": "#1f77b4", "multitask_film": "#d62728",
                   "deepdelta": "#2ca02c", "multitask_film_drfp": "#9467bd"}
    for i, enc in enumerate(encoders):
        for j, met in enumerate(metrics):
            ax = axes[i][j]
            sub = df[df.encoder == enc]
            for head, sub_h in sub.groupby("head"):
                g = sub_h.groupby("N")[met].agg(["mean", "std"]).reset_index()
                # order N
                def n_key(s):
                    return int(s) if s.isdigit() else 10**9
                g = g.sort_values(by="N", key=lambda c: c.apply(n_key))
                xs = list(range(len(g)))
                ax.errorbar(xs, g["mean"], yerr=g["std"], marker="o", label=head,
                            color=head_colors.get(head, None), capsize=3, lw=1.5, ms=5)
                ax.set_xticks(xs); ax.set_xticklabels(g["N"], fontsize=8, rotation=30)
            ax.set_title(f"{enc} | {met}", fontsize=10)
            if met == "mae":
                ax.set_ylabel("Δ-MAE (lower=better)")
            elif met == "spearman":
                ax.set_ylabel("Spearman ρ (Δ)")
            elif met == "r2":
                ax.set_ylabel("R² (Δ)")
                ax.set_ylim(-1, 1)
            if i == 0 and j == len(metrics) - 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("EXP5c — ZAP70 small-data deep architecture comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_summary(results, out_path, best_model_info):
    df = pd.DataFrame([
        {
            "encoder": r["encoder"], "head": r["head"], "N": r["N_label"], "seed": r["seed"],
            "mae": r["delta_metrics"]["mae"], "spearman": r["delta_metrics"]["spearman"],
            "pearson": r["delta_metrics"]["pearson"], "r2": r["delta_metrics"]["r2"],
            "abs_mae": r.get("abs_mae", float("nan")),
        }
        for r in results["runs"]
    ])
    # Aggregate per (encoder, head, N): mean across seeds
    agg = df.groupby(["encoder", "head", "N"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        spr_mean=("spearman", "mean"), spr_std=("spearman", "std"),
        n_seeds=("mae", "count"),
    ).reset_index()

    def n_key(c):
        return c.apply(lambda s: int(s) if s.isdigit() else 10**9)

    lines = ["# EXP5c — ZAP70 small-data deep architecture comparison\n"]
    lines.append("Task: predict delta pIC50 between matched molecular pairs (3001 pairs, 257 mols, 24 assays).")
    lines.append("Split: mol-disjoint single-side (held-out test mol_b never appears as mol_b in training).\n")

    # Q1: T3 vs T2
    lines.append("## Q1: Multi-task vs delta-only (T3 vs T2)\n")
    for enc in results["config"]["encoders"]:
        rows = []
        for N in sorted(agg.N.unique(), key=lambda s: int(s) if s.isdigit() else 10**9):
            t2 = agg[(agg.encoder == enc) & (agg.head == "filmdelta") & (agg.N == N)]
            t3 = agg[(agg.encoder == enc) & (agg.head == "multitask_film") & (agg.N == N)]
            if t2.empty or t3.empty:
                continue
            d2 = t2.mae_mean.iloc[0]; d3 = t3.mae_mean.iloc[0]
            rows.append((N, d2, d3, (d3 - d2) / d2 * 100))
        lines.append(f"### Encoder = {enc}\n")
        lines.append("| N | T2 FiLMDelta MAE | T3 MultiTask MAE | Δ% |")
        lines.append("|---|---|---|---|")
        for N, d2, d3, pct in rows:
            lines.append(f"| {N} | {d2:.3f} | {d3:.3f} | {pct:+.1f}% |")
        lines.append("")

    # Q2: richer encoders (E2/E3 vs E1) per head
    lines.append("## Q2: Richer encoders (ChemBERTa/MolFormer) vs Morgan, per head\n")
    for head in results["config"]["heads"]:
        rows = []
        for N in sorted(agg.N.unique(), key=lambda s: int(s) if s.isdigit() else 10**9):
            base = agg[(agg.encoder == "morgan") & (agg.head == head) & (agg.N == N)]
            cb = agg[(agg.encoder == "chemberta") & (agg.head == head) & (agg.N == N)]
            mf = agg[(agg.encoder == "molformer") & (agg.head == head) & (agg.N == N)]
            if base.empty:
                continue
            rows.append((N, base.mae_mean.iloc[0],
                         cb.mae_mean.iloc[0] if not cb.empty else float("nan"),
                         mf.mae_mean.iloc[0] if not mf.empty else float("nan")))
        lines.append(f"### Head = {head}\n")
        lines.append("| N | Morgan | ChemBERTa | MolFormer |")
        lines.append("|---|---|---|---|")
        for N, m, c, f in rows:
            lines.append(f"| {N} | {m:.3f} | {c:.3f} | {f:.3f} |")
        lines.append("")

    # Q3: headline — best edit vs best direct
    lines.append("## Q3: HEADLINE — best edit-aware vs best direct, per N\n")
    direct_subset = agg[agg.head == "direct"]
    edit_subset = agg[agg.head.isin(["filmdelta", "multitask_film", "multitask_film_drfp", "deepdelta"])]
    lines.append("| N | Best direct (enc) MAE | Best edit (enc, head) MAE | Δ% (edit-direct)/direct |")
    lines.append("|---|---|---|---|")
    for N in sorted(agg.N.unique(), key=lambda s: int(s) if s.isdigit() else 10**9):
        d = direct_subset[direct_subset.N == N]
        e = edit_subset[edit_subset.N == N]
        if d.empty or e.empty:
            continue
        d_best = d.loc[d.mae_mean.idxmin()]
        e_best = e.loc[e.mae_mean.idxmin()]
        pct = (e_best.mae_mean - d_best.mae_mean) / d_best.mae_mean * 100
        lines.append(f"| {N} | {d_best.mae_mean:.3f} ({d_best.encoder}) | "
                     f"{e_best.mae_mean:.3f} ({e_best.encoder}, {e_best.head}) | {pct:+.1f}% |")
    lines.append("")

    # Q4: where does edit advantage live? regime analysis
    lines.append("## Q4: Where does edit-aware advantage live?\n")
    lines.append("Per-N % advantage of best edit-aware method over best direct (negative = edit wins):\n")
    lines.append("| N | best direct (mean MAE) | best edit (mean MAE) | edit advantage % |")
    lines.append("|---|---|---|---|")
    for N in sorted(agg.N.unique(), key=lambda s: int(s) if s.isdigit() else 10**9):
        d = direct_subset[direct_subset.N == N]
        e = edit_subset[edit_subset.N == N]
        if d.empty or e.empty:
            continue
        d_mae = d.mae_mean.min(); e_mae = e.mae_mean.min()
        adv = (d_mae - e_mae) / d_mae * 100
        lines.append(f"| {N} | {d_mae:.3f} | {e_mae:.3f} | {adv:+.1f}% |")
    lines.append("")

    # Q5: does multi-task improve abs prediction (positive transfer)?
    lines.append("## Q5: Multi-task positive transfer on abs-pIC50 prediction\n")
    abs_df = df[(df.head.isin(["direct", "multitask_film", "multitask_film_drfp"]))
                & df.abs_mae.notna()]
    abs_agg = abs_df.groupby(["encoder", "head", "N"]).abs_mae.mean().reset_index()
    for enc in results["config"]["encoders"]:
        rows = []
        for N in sorted(abs_agg.N.unique(), key=lambda s: int(s) if s.isdigit() else 10**9):
            d = abs_agg[(abs_agg.encoder == enc) & (abs_agg.head == "direct") & (abs_agg.N == N)]
            t3 = abs_agg[(abs_agg.encoder == enc) & (abs_agg.head == "multitask_film") & (abs_agg.N == N)]
            if d.empty or t3.empty:
                continue
            rows.append((N, d.abs_mae.iloc[0], t3.abs_mae.iloc[0]))
        lines.append(f"### Encoder = {enc}\n")
        lines.append("| N | direct abs-MAE | multitask abs-MAE |")
        lines.append("|---|---|---|")
        for N, d, t3 in rows:
            lines.append(f"| {N} | {d:.3f} | {t3:.3f} |")
        lines.append("")

    if best_model_info:
        lines.append(f"## Saved checkpoint\n`models/filmdelta_zap70_best.pt` — "
                     f"{best_model_info['encoder']} / {best_model_info['head']} / seed={best_model_info['seed']}, "
                     f"delta-MAE={best_model_info['mae']:.3f}\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
