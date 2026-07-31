#!/usr/bin/env python3
"""
EXP5c - ZAP70 deep edit-aware architecture matrix (sequential encoder loading).

Refactored to avoid OOM by computing each encoder's cache one at a time, saving
to disk, and freeing the encoder model between caches. At training time, all
features are loaded from disk (no encoder models in memory).

Architectures tested (vs FiLM baseline from EXP5b):
  1) DrfpFiLM             — DRFP (2048d) conditions FiLM
  2) DualStreamFiLM       — Gated DRFP + Morgan diff + 28d edit feats
  3) FragAnchoredFiLM     — Fragment FP delta (1024d) + 28d edit feats
  4) MultiModalFiLM       — DRFP + fragment delta + edit feats fused
  5) EditHypernetFiLM     — LoRA-style weight perturbations from edit encoding

Backbone (mol encoder): Morgan FP (2048d) for all archs. Edit conditioning varies.

Splits (matched to EXP5b for direct comparison):
  - pair_disjoint
  - mol_disjoint_single_side
  - mol_disjoint_both_sides

Train sizes: 50, 200, ALL (1933 in mol_disjoint_both_sides train_pool)
Seeds: [0, 1, 2]

Total: 3 splits × 5 archs × 3 sizes × 3 seeds = 135 runs

Outputs:
  results/paper_evaluation/exp5c_filmdelta_deep_arch.json
  results/paper_evaluation/exp5c_filmdelta_deep_arch_summary.md
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
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force CPU (MPS has known issues with these models)
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.edit_aware_film_predictor import (
    DrfpFiLMDeltaMLP,
    DualStreamFiLMDeltaMLP,
    FragAnchoredFiLMDeltaMLP,
    MultiModalEditFiLMDeltaMLP,
    EditHypernetFiLMDeltaMLP,
)

PAIRS_FILE = PROJECT_ROOT / "data" / "exp5_zap70_within_assay_pairs.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp5c_filmdelta_deep_arch.json"
SUMMARY_FILE = RESULTS_DIR / "exp5c_filmdelta_deep_arch_summary.md"
EXP5B_RESULTS = RESULTS_DIR / "exp5b_filmdelta_zap70_deep.json"

# Per-target caches (ZAP70-specific, smaller than global caches)
MORGAN_CACHE = CACHE_DIR / "zap70_morgan_2048.npz"
DRFP_CACHE = CACHE_DIR / "zap70_drfp_pair_2048.npz"
FRAG_CACHE = CACHE_DIR / "zap70_frag_delta_1024.npz"
FEAT_CACHE = CACHE_DIR / "zap70_edit_feats_28.npz"

# Hyperparameters (match EXP5b for fair comparison)
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.2
LR = 1e-3
WD = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 25
SEEDS = [0, 1, 2]
TRAIN_SIZES = [50, 200]  # "ALL" appended
TEST_MOL_FRAC = 0.20
TEST_PAIR_FRAC = 0.20

DEVICE = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

ARCHITECTURES = ["DrfpFiLM", "DualStreamFiLM", "FragAnchoredFiLM",
                 "MultiModalFiLM", "EditHypernetFiLM"]


# ---------------------------------------------------------------------------
# Encoder cache builders (each frees its model after writing to disk)
# ---------------------------------------------------------------------------
def build_morgan_cache(smiles: List[str], out: Path,
                       n_bits: int = 2048, radius: int = 2):
    if out.exists():
        print(f"  [cache] {out.name} exists ({len(smiles)} smiles)")
        return
    print(f"  [build] morgan {n_bits}b r={radius} for {len(smiles)} smiles ...")
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
    np.savez_compressed(out, smiles=np.array(smiles, dtype=object), embeddings=X)
    print(f"  [save] {out.name}: {X.shape}")
    gc.collect()


def build_drfp_cache(pair_keys: List[Tuple[str, str]], out: Path):
    """DRFP per (mol_a, mol_b) pair."""
    if out.exists():
        print(f"  [cache] {out.name} exists")
        return
    print(f"  [build] drfp 2048 r=3 for {len(pair_keys)} pairs ...")
    from drfp import DrfpEncoder
    rxns = [f"{a}>>{b}" for a, b in pair_keys]
    t0 = time.time()
    X = DrfpEncoder.encode(rxns, n_folded_length=2048, radius=3, rings=True)
    X = np.asarray(X, dtype=np.float32)
    print(f"  [build] drfp done {time.time()-t0:.1f}s, shape={X.shape}")
    np.savez_compressed(
        out,
        mol_a=np.array([p[0] for p in pair_keys], dtype=object),
        mol_b=np.array([p[1] for p in pair_keys], dtype=object),
        embeddings=X,
    )
    print(f"  [save] {out.name}: {X.shape}")
    gc.collect()


def _derive_edit_smiles_via_mcs(mol_a_smi: str, mol_b_smi: str) -> str:
    """Derive an edit_smiles 'leaving>>incoming' by MCS deletion.

    Returns "" if it cannot be derived.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    m_a = Chem.MolFromSmiles(mol_a_smi)
    m_b = Chem.MolFromSmiles(mol_b_smi)
    if m_a is None or m_b is None:
        return ""
    try:
        res = rdFMCS.FindMCS([m_a, m_b], timeout=2,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                             ringMatchesRingOnly=True)
        if res.canceled or res.numAtoms < 2:
            return ""
        common = Chem.MolFromSmarts(res.smartsString)
        if common is None:
            return ""
        match_a = m_a.GetSubstructMatch(common)
        match_b = m_b.GetSubstructMatch(common)
        if not match_a or not match_b:
            return ""

        def make_frag(m, keep_idx):
            atoms_to_remove = [a.GetIdx() for a in m.GetAtoms()
                               if a.GetIdx() not in set(keep_idx)]
            if not atoms_to_remove:
                # No edit fragment (identical mols) — return wildcard
                return "[H]"
            edit = Chem.RWMol(m)
            # remove atoms (sort descending to keep indices stable)
            for idx in sorted(atoms_to_remove, reverse=True):
                # keep removed atoms — actually we want the OPPOSITE: keep removed
                pass
            # Strategy: keep only atoms_to_remove, drop the rest
            edit = Chem.RWMol(m)
            keep_set = set(atoms_to_remove)
            for idx in sorted(range(m.GetNumAtoms()), reverse=True):
                if idx not in keep_set:
                    edit.RemoveAtom(idx)
            try:
                Chem.SanitizeMol(edit)
                return Chem.MolToSmiles(edit)
            except Exception:
                return ""

        leaving = make_frag(m_a, match_a)
        incoming = make_frag(m_b, match_b)
        if not leaving or not incoming:
            return ""
        return f"{leaving}>>{incoming}"
    except Exception:
        return ""


def build_frag_and_feat_caches(pair_keys: List[Tuple[str, str]],
                               frag_out: Path, feat_out: Path,
                               n_bits: int = 1024):
    """Compute fragment FP delta (1024d) + 28-dim edit features per pair.

    Uses MCS-derived edit_smiles since the input CSV has no edit_smiles column.
    """
    frag_exists = frag_out.exists()
    feat_exists = feat_out.exists()
    if frag_exists and feat_exists:
        print(f"  [cache] {frag_out.name} & {feat_out.name} exist")
        return

    from src.data.utils.chemistry import (
        compute_fragment_fps, compute_edit_features,
    )
    n = len(pair_keys)
    print(f"  [build] frag deltas + edit feats for {n} pairs ...")

    frag_delta = np.zeros((n, n_bits), dtype=np.float32)
    edit_feats = np.zeros((n, 28), dtype=np.float32)

    t0 = time.time()
    n_ok = 0
    for i, (a, b) in enumerate(pair_keys):
        es = _derive_edit_smiles_via_mcs(a, b)
        if not es:
            continue
        try:
            fp_leaving, fp_incoming = compute_fragment_fps(es, radius=2, n_bits=n_bits)
            frag_delta[i] = fp_incoming - fp_leaving
            ef = compute_edit_features(a, b, es)
            edit_feats[i] = ef
            n_ok += 1
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s, {n_ok} ok)", flush=True)
    print(f"  [build] done {time.time()-t0:.0f}s, {n_ok}/{n} pairs derived")

    mol_a_arr = np.array([p[0] for p in pair_keys], dtype=object)
    mol_b_arr = np.array([p[1] for p in pair_keys], dtype=object)

    if not frag_exists:
        np.savez_compressed(frag_out, mol_a=mol_a_arr, mol_b=mol_b_arr,
                            embeddings=frag_delta)
        print(f"  [save] {frag_out.name}: {frag_delta.shape}")
    if not feat_exists:
        np.savez_compressed(feat_out, mol_a=mol_a_arr, mol_b=mol_b_arr,
                            features=edit_feats)
        print(f"  [save] {feat_out.name}: {edit_feats.shape}")
    gc.collect()


# ---------------------------------------------------------------------------
# Cache loaders (return dict-like for fast lookup)
# ---------------------------------------------------------------------------
def load_morgan(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {s: d["embeddings"][i] for i, s in enumerate(d["smiles"])}


def load_pair_emb(path: Path, key: str = "embeddings") -> Dict[Tuple[str, str], np.ndarray]:
    d = np.load(path, allow_pickle=True)
    val_key = key if key in d.keys() else "features"
    return {(a, b): d[val_key][i] for i, (a, b) in enumerate(zip(d["mol_a"], d["mol_b"]))}


# ---------------------------------------------------------------------------
# Splits (matched to EXP5b)
# ---------------------------------------------------------------------------
def split_pair_disjoint(pairs: pd.DataFrame, seed: int,
                        test_frac: float = TEST_PAIR_FRAC):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pairs)))
    test = pairs.iloc[idx[:n_test]].reset_index(drop=True)
    train = pairs.iloc[idx[n_test:]].reset_index(drop=True)
    return train, test


def split_mol_disjoint_single_side(pairs: pd.DataFrame, seed: int,
                                   mol_test_frac: float = TEST_MOL_FRAC):
    rng = np.random.RandomState(seed)
    mols = sorted(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    perm = rng.permutation(len(mols))
    n_test = int(round(mol_test_frac * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test])
    test_mask = (pairs["mol_b_id"].isin(test_mols) & ~pairs["mol_a_id"].isin(test_mols))
    train_mask = (~pairs["mol_a_id"].isin(test_mols) & ~pairs["mol_b_id"].isin(test_mols))
    return pairs[train_mask].reset_index(drop=True), pairs[test_mask].reset_index(drop=True)


def split_mol_disjoint_both_sides(pairs: pd.DataFrame, seed: int,
                                  mol_test_frac: float = TEST_MOL_FRAC):
    rng = np.random.RandomState(seed)
    mols = sorted(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    perm = rng.permutation(len(mols))
    n_test = int(round(mol_test_frac * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test])
    test_mask = (pairs["mol_a_id"].isin(test_mols) & pairs["mol_b_id"].isin(test_mols))
    train_mask = (~pairs["mol_a_id"].isin(test_mols) & ~pairs["mol_b_id"].isin(test_mols))
    return pairs[train_mask].reset_index(drop=True), pairs[test_mask].reset_index(drop=True)


SPLITS = {
    "pair_disjoint": split_pair_disjoint,
    "mol_disjoint_single_side": split_mol_disjoint_single_side,
    "mol_disjoint_both_sides": split_mol_disjoint_both_sides,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def delta_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    n = int(len(y_true))
    if n == 0:
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
    return {"mae": mae, "rmse": rmse,
            "pearson": float(pr if not np.isnan(pr) else 0),
            "spearman": float(sr if not np.isnan(sr) else 0),
            "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# Stack pair tensors from caches
# ---------------------------------------------------------------------------
def stack_features(pairs: pd.DataFrame,
                   morgan_emb: Dict[str, np.ndarray],
                   drfp_cache: Dict[Tuple[str, str], np.ndarray],
                   frag_cache: Dict[Tuple[str, str], np.ndarray],
                   feat_cache: Dict[Tuple[str, str], np.ndarray]):
    """Returns dict of arrays needed by every architecture."""
    A = np.stack([morgan_emb[s] for s in pairs["mol_a"]]).astype(np.float32)
    B = np.stack([morgan_emb[s] for s in pairs["mol_b"]]).astype(np.float32)
    drfp = np.stack([drfp_cache[(a, b)] for a, b in zip(pairs["mol_a"], pairs["mol_b"])]).astype(np.float32)
    frag = np.stack([frag_cache[(a, b)] for a, b in zip(pairs["mol_a"], pairs["mol_b"])]).astype(np.float32)
    feats = np.stack([feat_cache[(a, b)] for a, b in zip(pairs["mol_a"], pairs["mol_b"])]).astype(np.float32)
    y = pairs["delta"].values.astype(np.float32)
    return {"A": A, "B": B, "drfp": drfp, "frag": frag, "feats": feats, "y": y}


# ---------------------------------------------------------------------------
# Architecture instantiation
# ---------------------------------------------------------------------------
def make_model(arch: str, mol_dim: int) -> nn.Module:
    if arch == "DrfpFiLM":
        return DrfpFiLMDeltaMLP(mol_dim=mol_dim, drfp_dim=2048,
                                hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if arch == "DualStreamFiLM":
        return DualStreamFiLMDeltaMLP(mol_dim=mol_dim, drfp_dim=2048,
                                      edit_feat_dim=28,
                                      hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if arch == "FragAnchoredFiLM":
        return FragAnchoredFiLMDeltaMLP(mol_dim=mol_dim, frag_dim=1024,
                                        edit_feat_dim=28,
                                        hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if arch == "MultiModalFiLM":
        return MultiModalEditFiLMDeltaMLP(mol_dim=mol_dim, drfp_dim=2048,
                                          frag_dim=1024, edit_feat_dim=28,
                                          use_rxnfp=False,
                                          hidden_dims=HIDDEN_DIMS,
                                          dropout=DROPOUT)
    if arch == "EditHypernetFiLM":
        # Use DRFP as edit_enc (project to 256d inside model)
        # The hypernet expects pre-encoded edit_enc; we encode DRFP→256d here
        return _EditHypernetWrapper(mol_dim=mol_dim, drfp_dim=2048,
                                    edit_enc_dim=256, dropout=DROPOUT)
    raise ValueError(arch)


class _EditHypernetWrapper(nn.Module):
    """Wraps EditHypernetFiLMDeltaMLP with a DRFP→edit_enc projector,
    so it can be trained end-to-end like the others."""
    def __init__(self, mol_dim: int, drfp_dim: int = 2048,
                 edit_enc_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.edit_proj = nn.Sequential(
            nn.Linear(drfp_dim, edit_enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hyper = EditHypernetFiLMDeltaMLP(
            mol_dim=mol_dim, edit_enc_dim=edit_enc_dim,
            hidden_dims=HIDDEN_DIMS, rank=8, lora_scale=0.1, dropout=dropout)

    def forward(self, emb_a, emb_b, drfp):
        edit_enc = self.edit_proj(drfp)
        return self.hyper(emb_a, emb_b, edit_enc)


# ---------------------------------------------------------------------------
# Training loop (architecture-agnostic via signature dispatch)
# ---------------------------------------------------------------------------
def _forward(arch: str, model: nn.Module, batch: dict) -> torch.Tensor:
    A, B = batch["A"], batch["B"]
    if arch == "DrfpFiLM":
        return model(A, B, batch["drfp"])
    if arch == "DualStreamFiLM":
        return model(A, B, batch["drfp"], batch["feats"])
    if arch == "FragAnchoredFiLM":
        return model(A, B, batch["frag"], batch["feats"])
    if arch == "MultiModalFiLM":
        return model(A, B, batch["drfp"], batch["frag"], batch["feats"])
    if arch == "EditHypernetFiLM":
        return model(A, B, batch["drfp"])
    raise ValueError(arch)


def _batch_iter(stacks: dict, batch_size: int, shuffle: bool, rng=None):
    n = stacks["y"].shape[0]
    idx = np.arange(n)
    if shuffle:
        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)
    for s in range(0, n, batch_size):
        e = idx[s: s + batch_size]
        yield {
            "A": torch.from_numpy(stacks["A"][e]).float(),
            "B": torch.from_numpy(stacks["B"][e]).float(),
            "drfp": torch.from_numpy(stacks["drfp"][e]).float(),
            "frag": torch.from_numpy(stacks["frag"][e]).float(),
            "feats": torch.from_numpy(stacks["feats"][e]).float(),
            "y": torch.from_numpy(stacks["y"][e]).float(),
        }


def train_arch(arch: str, train_stacks: dict, val_stacks: Optional[dict],
               mol_dim: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arch, mol_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    rng = np.random.RandomState(seed)
    best = float("inf"); best_state = None; bad = 0
    has_val = val_stacks is not None and val_stacks["y"].shape[0] > 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in _batch_iter(train_stacks, BATCH_SIZE, shuffle=True, rng=rng):
            opt.zero_grad()
            pred = _forward(arch, model, batch)
            loss = F.mse_loss(pred, batch["y"])
            # Add aux loss for DualStream / MultiModal
            if arch in ("DualStreamFiLM", "MultiModalFiLM"):
                if hasattr(model, "aux_loss"):
                    try:
                        loss = loss + 0.1 * model.aux_loss(batch["feats"])
                    except Exception:
                        pass
            loss.backward()
            opt.step()

        if has_val:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in _batch_iter(val_stacks, BATCH_SIZE, shuffle=False):
                    pred = _forward(arch, model, batch)
                    losses.append(F.mse_loss(pred, batch["y"]).item())
            vl = float(np.mean(losses))
            if vl < best - 1e-5:
                best = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def predict_arch(arch: str, model: nn.Module, stacks: dict) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in _batch_iter(stacks, BATCH_SIZE, shuffle=False):
            pred = _forward(arch, model, batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Worker (one (arch, split, N, seed) cell)
# ---------------------------------------------------------------------------
def run_one_cell_inproc(args_tuple, cached: dict):
    """In-process version: skip cache reloads, use pre-loaded dicts."""
    (arch, split_name, seed, N, _, _, _, _, _) = args_tuple
    try:
        return _do_one_cell(arch, split_name, seed, N,
                             cached["pairs"], cached["morgan"], cached["drfp"],
                             cached["frag"], cached["feat"])
    except Exception as e:
        import traceback
        return {"arch": arch, "split": split_name, "seed": seed, "N": str(N),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:2000]}


def _do_one_cell(arch, split_name, seed, N, pairs, morgan_emb,
                 drfp_cache, frag_cache, feat_cache):
    split_fn = SPLITS[split_name]
    train_pool, test_pairs = split_fn(pairs, seed=seed)

    if len(train_pool) == 0 or len(test_pairs) == 0:
        return {"arch": arch, "split": split_name, "seed": seed, "N": str(N),
                "error": "empty_split"}

    # Subset training
    rng = np.random.RandomState(seed * 1000 + (0 if N == "ALL" else int(N)))
    idx = np.arange(len(train_pool)); rng.shuffle(idx)
    n_use = len(train_pool) if N == "ALL" else min(int(N), len(train_pool))
    fit_pairs = train_pool.iloc[idx[:n_use]].copy().reset_index(drop=True)

    # 10% internal val (only when N >= 50)
    if len(fit_pairs) >= 50:
        n_val = max(5, int(0.1 * len(fit_pairs)))
        val_pairs = fit_pairs.iloc[-n_val:].copy()
        fit_pairs = fit_pairs.iloc[:-n_val].copy()
    else:
        val_pairs = None

    mol_dim = next(iter(morgan_emb.values())).shape[0]

    train_stacks = stack_features(fit_pairs, morgan_emb, drfp_cache, frag_cache, feat_cache)
    val_stacks = (stack_features(val_pairs, morgan_emb, drfp_cache, frag_cache, feat_cache)
                  if val_pairs is not None else None)
    test_stacks = stack_features(test_pairs, morgan_emb, drfp_cache, frag_cache, feat_cache)

    t0 = time.time()
    model = train_arch(arch, train_stacks, val_stacks, mol_dim, seed)
    train_dt = time.time() - t0

    y_pred = predict_arch(arch, model, test_stacks)
    y_true = test_stacks["y"]
    metrics = delta_metrics(y_true, y_pred)

    return {
        "arch": arch, "split": split_name, "seed": seed, "N": str(N),
        "n_fit": int(len(fit_pairs)),
        "n_val": int(len(val_pairs) if val_pairs is not None else 0),
        "n_test": int(len(test_pairs)),
        "n_train_pool": int(len(train_pool)),
        "metrics": metrics,
        "train_dt": float(train_dt),
    }


def run_one_cell(args_tuple):
    (arch, split_name, seed, N, pairs_path,
     morgan_path, drfp_path, frag_path, feat_path) = args_tuple

    # Re-set CPU thread count in worker
    torch.set_num_threads(1)
    try:
        pairs = pd.read_csv(pairs_path)
        morgan_emb = load_morgan(Path(morgan_path))
        drfp_cache = load_pair_emb(Path(drfp_path))
        frag_cache = load_pair_emb(Path(frag_path))
        feat_cache = load_pair_emb(Path(feat_path), key="features")
        return _do_one_cell(arch, split_name, seed, N, pairs, morgan_emb,
                             drfp_cache, frag_cache, feat_cache)
    except Exception as e:
        import traceback
        return {"arch": arch, "split": split_name, "seed": seed, "N": str(N),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:2000]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="*", default=ARCHITECTURES)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS.keys()))
    parser.add_argument("--sizes", nargs="*", default=[50, 200, "ALL"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--build-only", action="store_true",
                        help="Only build caches, don't run experiments")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [0]
        args.sizes = [50, "ALL"]
        args.archs = ["DrfpFiLM", "DualStreamFiLM"]

    # Normalize sizes
    sizes_norm = []
    for s in args.sizes:
        if isinstance(s, str) and s.upper() == "ALL":
            sizes_norm.append("ALL")
        else:
            sizes_norm.append(int(s))
    args.sizes = sizes_norm

    print(f"[{time.strftime('%H:%M:%S')}] Loading pairs: {PAIRS_FILE}")
    pairs = pd.read_csv(PAIRS_FILE)
    print(f"  pairs={len(pairs)} mols={len(set(pairs.mol_a_id) | set(pairs.mol_b_id))} "
          f"assays={pairs.assay_id.nunique()}")

    all_smiles = sorted(set(pairs.mol_a).union(pairs.mol_b))
    pair_keys = list(zip(pairs.mol_a, pairs.mol_b))
    print(f"  unique smiles: {len(all_smiles)}, pair keys: {len(pair_keys)}")

    # --- SEQUENTIAL CACHE BUILDING (free each encoder before next) ---
    print(f"\n[{time.strftime('%H:%M:%S')}] === Building caches sequentially ===")

    print(f"[{time.strftime('%H:%M:%S')}] 1/3 Morgan ...")
    build_morgan_cache(all_smiles, MORGAN_CACHE)
    gc.collect()

    print(f"[{time.strftime('%H:%M:%S')}] 2/3 DRFP ...")
    build_drfp_cache(pair_keys, DRFP_CACHE)
    gc.collect()

    print(f"[{time.strftime('%H:%M:%S')}] 3/3 Fragment FP + edit features ...")
    build_frag_and_feat_caches(pair_keys, FRAG_CACHE, FEAT_CACHE)
    gc.collect()

    if args.build_only:
        print("--build-only set, exiting")
        return

    # --- Print split sizes ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Split sizes (seed=0 sanity):")
    for name, fn in SPLITS.items():
        tp, te = fn(pairs, 0)
        print(f"  {name:30s} train_pool={len(tp):4d} test={len(te):4d}")

    # --- Build worklist ---
    work = []
    for arch in args.archs:
        for split in args.splits:
            for seed in args.seeds:
                for N in args.sizes:
                    work.append((arch, split, seed, N,
                                 str(PAIRS_FILE),
                                 str(MORGAN_CACHE), str(DRFP_CACHE),
                                 str(FRAG_CACHE), str(FEAT_CACHE)))
    total = len(work)
    print(f"\n[{time.strftime('%H:%M:%S')}] Total runs: {total} "
          f"({len(args.archs)} archs × {len(args.splits)} splits "
          f"× {len(args.sizes)} sizes × {len(args.seeds)} seeds)")

    # --- Resume support: skip cells already in JSON ---
    existing = {"runs": [], "config": {}}
    if RESULTS_FILE.exists():
        try:
            existing = json.load(open(RESULTS_FILE))
            print(f"  loaded {len(existing.get('runs', []))} existing runs")
        except Exception:
            existing = {"runs": [], "config": {}}
    done_keys = {(r["arch"], r["split"], r["seed"], r["N"]) for r in existing.get("runs", [])
                 if "error" not in r}
    work = [w for w in work if (w[0], w[1], w[2], (str(w[3]))) not in done_keys]
    print(f"  {len(work)} cells to run (skipping {total - len(work)} done)")

    results = existing
    results["config"] = {
        "target": "ZAP70 (CHEMBL2803)",
        "pairs_file": str(PAIRS_FILE),
        "n_pairs": int(len(pairs)),
        "archs": args.archs,
        "splits": args.splits,
        "sizes": [str(s) for s in args.sizes],
        "seeds": args.seeds,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "lr": LR, "wd": WD,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "test_mol_frac": TEST_MOL_FRAC, "test_pair_frac": TEST_PAIR_FRAC,
        "workers": args.workers,
    }

    if not work:
        print("Nothing to run — proceeding to summary")
    else:
        t0 = time.time()
        completed = 0
        # Pre-load caches once (workers=1 path) so we don't reload per cell
        if args.workers <= 1:
            print(f"  [main] pre-loading caches for in-process runs ...")
            cached = {
                "pairs": pd.read_csv(PAIRS_FILE),
                "morgan": load_morgan(MORGAN_CACHE),
                "drfp": load_pair_emb(DRFP_CACHE),
                "frag": load_pair_emb(FRAG_CACHE),
                "feat": load_pair_emb(FEAT_CACHE, key="features"),
            }
            for w in work:
                out = run_one_cell_inproc(w, cached)
                completed += 1
                results["runs"].append(out)
                _log_progress(out, completed, len(work), time.time() - t0)
                if completed % 5 == 0:
                    _save(results)
                gc.collect()
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                fut_map = {ex.submit(run_one_cell, w): w for w in work}
                for fut in as_completed(fut_map):
                    out = fut.result()
                    completed += 1
                    results["runs"].append(out)
                    _log_progress(out, completed, len(work), time.time() - t0)
                    if completed % 5 == 0:
                        _save(results)

        _save(results)
        print(f"[{time.strftime('%H:%M:%S')}] Done. Wrote {RESULTS_FILE} "
              f"(total runs: {len(results['runs'])}, dt={time.time()-t0:.1f}s)")

    # Always (re)write summary
    write_summary(results, SUMMARY_FILE)
    print(f"Wrote summary {SUMMARY_FILE}")


def _log_progress(out, completed, total, dt):
    if "error" in out:
        print(f"  [{completed}/{total}] FAIL arch={out['arch']:18s} split={out['split']:25s} "
              f"seed={out['seed']} N={out['N']:>3s}: {out['error'][:100]}", flush=True)
    else:
        m = out["metrics"]
        print(f"  [{completed}/{total}] arch={out['arch']:18s} split={out['split']:25s} "
              f"seed={out['seed']} N={out['N']:>3s} "
              f"mae={m['mae']:.3f} spr={m['spearman']:.3f} "
              f"({out['train_dt']:.1f}s, elapsed={dt/60:.1f}m)", flush=True)


def _save(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _load_exp5b_baseline() -> Dict[Tuple[str, int, str], float]:
    """Return {(split, seed, N): filmdelta_mae} from EXP5b."""
    out = {}
    if not EXP5B_RESULTS.exists():
        return out
    try:
        d = json.load(open(EXP5B_RESULTS))
        for r in d.get("runs", []):
            if "filmdelta" not in r or "mae" not in r.get("filmdelta", {}):
                continue
            N_raw = r.get("N_requested_raw", r.get("N_requested"))
            N_key = str(N_raw)
            out[(r["split"], int(r["seed"]), N_key)] = r["filmdelta"]["mae"]
    except Exception as e:
        print(f"  WARN: failed to load EXP5b baseline: {e}")
    return out


def write_summary(results, out_path):
    runs = [r for r in results.get("runs", []) if "error" not in r]
    if not runs:
        with open(out_path, "w") as f:
            f.write("# EXP5c — no successful runs to summarize.\n")
        return

    df = pd.DataFrame([{
        "arch": r["arch"], "split": r["split"], "seed": r["seed"], "N": r["N"],
        "mae": r["metrics"]["mae"],
        "spearman": r["metrics"]["spearman"],
        "pearson": r["metrics"]["pearson"],
        "r2": r["metrics"]["r2"],
    } for r in runs])

    # Baseline FiLM from exp5b
    baseline = _load_exp5b_baseline()

    def n_sort_key(n):
        try:
            return int(n)
        except (ValueError, TypeError):
            return 10**9

    lines = ["# EXP5c — ZAP70 deep edit-aware architecture matrix\n"]
    lines.append("Target: ZAP70 (CHEMBL2803), 3001 within-assay MMP pairs, 257 mols.")
    lines.append(f"Architectures: {', '.join(sorted(df.arch.unique()))}")
    lines.append(f"Splits: {', '.join(sorted(df.split.unique()))}")
    lines.append(f"Sizes: {sorted(df.N.unique(), key=n_sort_key)}")
    lines.append(f"Seeds: {sorted(df.seed.unique())}")
    lines.append("\nBaseline (for comparison): FiLM (Morgan-diff conditioning) from EXP5b.\n")

    # === Table 1: arch × split mean ± std MAE (averaged across N and seeds) ===
    lines.append("## Table 1: Mean MAE by (arch, split), averaged over N and seeds\n")
    pivot = df.groupby(["arch", "split"]).mae.agg(["mean", "std", "count"]).reset_index()
    splits_sorted = sorted(df.split.unique())
    lines.append("| Arch | " + " | ".join(splits_sorted) + " |")
    lines.append("|---|" + "|".join(["---"] * len(splits_sorted)) + "|")
    for arch in sorted(df.arch.unique()):
        row = [arch]
        for split in splits_sorted:
            sub = pivot[(pivot.arch == arch) & (pivot.split == split)]
            if sub.empty:
                row.append("-")
            else:
                m = sub["mean"].iloc[0]; s = sub["std"].iloc[0]; n = sub["count"].iloc[0]
                row.append(f"{m:.3f}±{s:.3f} (n={n})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Table 2: arch × N × split detail ===
    lines.append("## Table 2: MAE detail by (arch, split, N), mean ± std across seeds\n")
    for split in splits_sorted:
        lines.append(f"### Split = {split}\n")
        sub_df = df[df.split == split]
        sizes = sorted(sub_df.N.unique(), key=n_sort_key)
        lines.append("| Arch | " + " | ".join([f"N={n}" for n in sizes]) + " |")
        lines.append("|---|" + "|".join(["---"] * len(sizes)) + "|")
        for arch in sorted(sub_df.arch.unique()):
            row = [arch]
            for N in sizes:
                cell = sub_df[(sub_df.arch == arch) & (sub_df.N == N)]
                if cell.empty:
                    row.append("-")
                else:
                    m = cell.mae.mean(); s = cell.mae.std()
                    row.append(f"{m:.3f}±{s:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # === Table 3: advantage vs FiLM baseline ===
    if baseline:
        lines.append("## Table 3: Advantage % over FiLM baseline (matched split/seed/N)\n")
        lines.append("Negative % = improvement over FiLM (lower MAE).")
        lines.append("Computed per (split, seed, N) cell, then averaged.\n")
        adv_rows = []
        for _, r in df.iterrows():
            bkey = (r["split"], int(r["seed"]), r["N"])
            if bkey not in baseline:
                continue
            base_mae = baseline[bkey]
            if base_mae <= 0:
                continue
            adv_pct = (r["mae"] - base_mae) / base_mae * 100
            adv_rows.append({
                "arch": r["arch"], "split": r["split"], "N": r["N"], "seed": r["seed"],
                "arch_mae": r["mae"], "film_mae": base_mae, "adv_pct": adv_pct,
            })
        if adv_rows:
            adv_df = pd.DataFrame(adv_rows)
            lines.append("| Arch | " + " | ".join(splits_sorted) + " | OVERALL |")
            lines.append("|---|" + "|".join(["---"] * (len(splits_sorted) + 1)) + "|")
            for arch in sorted(adv_df.arch.unique()):
                row = [arch]
                arch_sub = adv_df[adv_df.arch == arch]
                for split in splits_sorted:
                    cell = arch_sub[arch_sub.split == split]
                    if cell.empty:
                        row.append("-")
                    else:
                        m = cell.adv_pct.mean()
                        row.append(f"{m:+.1f}%")
                m_overall = arch_sub.adv_pct.mean()
                row.append(f"{m_overall:+.1f}%")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

            # === FLAG: arch beats FiLM by ≥10% on mol_disjoint_both_sides ===
            lines.append("## FLAG check (≥10% improvement over FiLM on mol_disjoint_both_sides)\n")
            both_sides = adv_df[adv_df.split == "mol_disjoint_both_sides"]
            flagged_any = False
            for arch in sorted(both_sides.arch.unique()):
                a_sub = both_sides[both_sides.arch == arch]
                m = a_sub.adv_pct.mean()
                if m <= -10.0:
                    flagged_any = True
                    lines.append(f"**  {arch}: {m:+.1f}% (n={len(a_sub)}) — MEETS ≥10% FLAG  **\n")
                else:
                    lines.append(f"- {arch}: {m:+.1f}% (n={len(a_sub)}) — does not meet 10% threshold\n")
            if not flagged_any:
                lines.append("\n**No architecture beat FiLM by ≥10% on mol_disjoint_both_sides.**")
        else:
            lines.append("(no matched cells found in EXP5b baseline)")
    else:
        lines.append("## Table 3: baseline comparison unavailable (no EXP5b results found)\n")

    # === Best per split ===
    lines.append("\n## Best arch per split (lowest mean MAE)\n")
    lines.append("| Split | Best arch | MAE | Spearman |")
    lines.append("|---|---|---|---|")
    for split in splits_sorted:
        sub = df[df.split == split].groupby("arch").agg(
            mae_mean=("mae", "mean"), spr_mean=("spearman", "mean")).reset_index()
        if sub.empty:
            continue
        best = sub.loc[sub.mae_mean.idxmin()]
        lines.append(f"| {split} | {best.arch} | {best.mae_mean:.3f} | {best.spr_mean:.3f} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
