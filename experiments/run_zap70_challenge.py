#!/usr/bin/env python3
"""
ZAP70 Challenge — Stress-test FiLMDelta against 9 alternative models.

Models on identical 5-fold CV (random KFold + Butina cluster GroupKFold):
  1. FiLMDelta (delta, all-pairs, anchor-median reconstruction)
  2. Direct Morgan MLP (absolute)
  3. Morgan MLP + kinase pretrain (dual-objective)
  4. ChemBERTa-2 MTR frozen + MLP head (absolute)
  5. ChemProp featurizer (= Morgan via chemprop) + MLP (absolute)
  6. DeepDelta (concat MLP, delta, anchor-median reconstruction)
  7. Bootstrap ensemble of FiLMDelta (B=20)
  8. Classification cascade: XGB binary active + XGB regressor on positives
  9. Multi-task with target embedding (kinase + ZAP70 jointly)
 10. XGBoost multi-FP (Morgan + RDKit + MACCS + AtomPair)

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_zap70_challenge.py --models all
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_zap70_challenge.py --models 1,2,10
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_zap70_challenge.py --models 0  # Phase 0 reproduction only
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
# Let torch.cuda.is_available() decide. (On Mac there's no CUDA anyway; on V100 it's enabled.)
torch.backends.mps.is_available = lambda: False  # MPS issues per CLAUDE.md

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
TARGET_ID = "CHEMBL2803"  # ZAP70
N_FOLDS = 5
CV_SEED = 42
_RAW_DEFAULT = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
_RAW_ZAP_ONLY = PROJECT_ROOT / "data" / "overlapping_assays" / "zap70_only.csv"

def _pick_raw_file():
    """Prefer whichever file has MORE ZAP70 unique mols."""
    candidates = []
    for p in (_RAW_ZAP_ONLY, _RAW_DEFAULT):
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, usecols=["target_chembl_id", "molecule_chembl_id"])
            n_zap = df[df["target_chembl_id"] == TARGET_ID]["molecule_chembl_id"].nunique()
            candidates.append((p, n_zap))
        except Exception:
            continue
    if not candidates:
        return _RAW_DEFAULT
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]

RAW_FILE = _pick_raw_file()
print(f"[INFO] Using RAW_FILE = {RAW_FILE}")
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "zap70_challenge"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "all_results.json"
RESULTS_CSV = RESULTS_DIR / "all_results.csv"
PREDS_DIR = RESULTS_DIR / "per_model_logs"
PREDS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RESULTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def save_torch_ckpt(model, model_key, fold, split_name, seed, extra=None):
    """Save a torch model checkpoint."""
    import pickle
    out = CKPT_DIR / f"{model_key}_{split_name}_fold{fold}_seed{seed}.pt"
    try:
        ckpt = {"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "extra": extra or {}}
        torch.save(ckpt, out)
    except Exception as e:
        print(f"  ⚠ ckpt save failed: {e}")


def save_sklearn_ckpt(model, model_key, fold, split_name, seed, extra=None):
    import pickle
    out = CKPT_DIR / f"{model_key}_{split_name}_fold{fold}_seed{seed}.pkl"
    try:
        with open(out, "wb") as f:
            pickle.dump({"model": model, "extra": extra or {}}, f)
    except Exception as e:
        print(f"  ⚠ ckpt save failed: {e}")

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"[INFO] Device = {DEVICE}")


# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────
def load_zap70_mols() -> pd.DataFrame:
    """Load 280 unique ZAP70 mols with averaged pIC50."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == TARGET_ID].copy()
    mol = (
        zap.groupby("molecule_chembl_id")
        .agg({"smiles": "first", "pIC50": "mean"})
        .reset_index()
    )
    mol = mol.reset_index(drop=True)
    print(f"[DATA] ZAP70: {len(mol)} unique mols, pIC50 {mol.pIC50.min():.2f}-{mol.pIC50.max():.2f}")
    return mol


def load_kinase_pretrain() -> pd.DataFrame:
    """Load kinase MMP pretrain pairs (no ZAP70 leakage)."""
    df = pd.read_csv(KINASE_PAIRS_FILE)
    assert (df["target_chembl_id"] == TARGET_ID).sum() == 0, "ZAP70 leakage in kinase pretrain!"
    print(f"[DATA] Kinase pretrain: {len(df)} MMP pairs across {df.target_chembl_id.nunique()} targets")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Fingerprints / embeddings
# ────────────────────────────────────────────────────────────────────────────
def compute_morgan(smiles_list, radius=2, n_bits=2048) -> np.ndarray:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out.append(arr)
    return np.array(out, dtype=np.float32)


def compute_rdkit_fp(smiles_list, n_bits=2048) -> np.ndarray:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out.append(arr)
    return np.array(out, dtype=np.float32)


def compute_maccs(smiles_list) -> np.ndarray:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(np.zeros(167, dtype=np.float32))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out.append(arr)
    return np.array(out, dtype=np.float32)


def compute_atompair(smiles_list, n_bits=2048) -> np.ndarray:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out.append(arr)
    return np.array(out, dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Splits
# ────────────────────────────────────────────────────────────────────────────
def random_kfold(mol_df: pd.DataFrame, n_splits=N_FOLDS, seed=CV_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(len(mol_df))))


def butina_groupkfold(mol_df: pd.DataFrame, n_splits=N_FOLDS, cutoff=0.35, seed=CV_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Cluster mols with Butina, then GroupKFold using clusters as groups."""
    smiles = mol_df["smiles"].tolist()
    fps = []
    valid = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        valid.append(i)
    n = len(fps)
    dists = []
    for i in range(n):
        for j in range(i):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # cluster_id per mol
    cluster_of = np.zeros(len(mol_df), dtype=int)
    for cid, members in enumerate(clusters):
        for local in members:
            cluster_of[valid[local]] = cid
    print(f"[SPLIT-BUTINA] {len(clusters)} clusters; sizes: min={min(len(c) for c in clusters)}, max={max(len(c) for c in clusters)}, mean={n/len(clusters):.1f}")
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.arange(len(mol_df)), groups=cluster_of))


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────
def compute_abs_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if len(y_true) > 2 and y_pred.std() > 1e-9:
        pr, _ = pearsonr(y_true, y_pred)
        sr, _ = spearmanr(y_true, y_pred)
    else:
        pr, sr = 0.0, 0.0
    return {
        "n": int(len(y_true)),
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": float(pr if not np.isnan(pr) else 0.0),
        "spearman_r": float(sr if not np.isnan(sr) else 0.0),
    }


def aggregate_folds(fold_metrics: List[Dict]) -> Dict[str, float]:
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float)) and k != "n"]
    out = {"n_folds": len(fold_metrics)}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Anchor reconstruction: delta predictions → absolute pIC50
# ────────────────────────────────────────────────────────────────────────────
def delta_to_abs_via_anchors(
    predict_delta_fn,  # callable(emb_a [Na,D], emb_b [Nb,D]) -> delta [Nab]
    train_emb: np.ndarray,
    train_pIC50: np.ndarray,
    test_emb: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """For each test mol b, predict pIC50_b = aggregate_over_anchors a in train of (pIC50_a + delta_pred(a,b))."""
    n_tr = train_emb.shape[0]
    n_te = test_emb.shape[0]
    # Build all (a in train, b in test) pairs
    emb_a_all = np.repeat(train_emb, n_te, axis=0)  # [Na*Nb, D]
    emb_b_all = np.tile(test_emb, (n_tr, 1))  # [Na*Nb, D]
    delta_all = predict_delta_fn(emb_a_all, emb_b_all)  # [Na*Nb]
    delta_mat = delta_all.reshape(n_tr, n_te)
    anchor_pred = train_pIC50[:, None] + delta_mat  # [Na, Nb]
    if method == "median":
        return np.median(anchor_pred, axis=0)
    return np.mean(anchor_pred, axis=0)


# ────────────────────────────────────────────────────────────────────────────
# Model 1: FiLMDelta
# ────────────────────────────────────────────────────────────────────────────
def train_eval_filmdelta(
    train_idx: np.ndarray, test_idx: np.ndarray, mol_df: pd.DataFrame, X_morgan: np.ndarray,
    max_epochs=50, batch_size=256, lr=1e-3, dropout=0.2, seed=42,
) -> Tuple[Dict, np.ndarray]:
    """Train FiLMDelta on all train-train pairs, predict abs pIC50 on test via anchor reconstruction."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
    torch.manual_seed(seed); np.random.seed(seed)

    train_smiles = mol_df.iloc[train_idx]["smiles"].values
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_smiles = mol_df.iloc[test_idx]["smiles"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values

    train_emb = X_morgan[train_idx]
    test_emb = X_morgan[test_idx]

    # Build train pairs: all i<j in train fold
    n_tr = len(train_idx)
    ii, jj = np.triu_indices(n_tr, k=1)
    emb_a_tr = train_emb[ii]
    emb_b_tr = train_emb[jj]
    delta_tr = (train_y[jj] - train_y[ii]).astype(np.float32)

    predictor = FiLMDeltaPredictor(
        dropout=dropout, learning_rate=lr, batch_size=batch_size,
        max_epochs=max_epochs, patience=12, device=DEVICE,
    )
    predictor.fit(emb_a_tr, emb_b_tr, delta_tr, verbose=False)

    # Anchor reconstruction
    def pred_fn(a, b):
        return predictor.predict(a, b)
    abs_pred = delta_to_abs_via_anchors(pred_fn, train_emb, train_y, test_emb, method="median")

    # Also evaluate delta predictions on the test-test pairs (for parity with v6 Phase A)
    n_te = len(test_idx)
    ii_t, jj_t = np.triu_indices(n_te, k=1)
    delta_te_true = (test_y[jj_t] - test_y[ii_t]).astype(np.float32)
    delta_te_pred = predictor.predict(test_emb[ii_t], test_emb[jj_t])
    delta_mae = float(np.mean(np.abs(delta_te_true - delta_te_pred)))
    delta_spr, _ = spearmanr(delta_te_true, delta_te_pred) if len(delta_te_true) > 2 else (0, 1)

    metrics = compute_abs_metrics(test_y, abs_pred)
    metrics["delta_mae"] = delta_mae
    metrics["delta_spearman"] = float(delta_spr if not np.isnan(delta_spr) else 0)
    metrics["n_train_pairs"] = int(len(delta_tr))
    return metrics, abs_pred


# ────────────────────────────────────────────────────────────────────────────
# Model 2: Direct Morgan MLP
# ────────────────────────────────────────────────────────────────────────────
class DirectMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_direct_mlp(X_train, y_train, X_test, hidden=(512, 256, 128), dropout=0.3,
                     lr=1e-3, epochs=200, batch_size=32, patience=20, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)
    n_val = max(10, len(X_tr) // 5)
    perm = np.random.RandomState(seed).permutation(len(X_tr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = DirectMLP(X_tr.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.FloatTensor(X_tr).to(DEVICE)
    ytr_t = torch.FloatTensor(y_train).to(DEVICE)
    Xte_t = torch.FloatTensor(X_te).to(DEVICE)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm_tr = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), batch_size):
            bi = tr_idx[perm_tr[start:start + batch_size]]
            opt.zero_grad()
            pred = model(Xtr_t[bi])
            ls = loss_fn(pred, ytr_t[bi])
            ls.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xtr_t[val_idx]), ytr_t[val_idx]).item())
        sch.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).cpu().numpy()
    return preds


def train_eval_direct_morgan_mlp(train_idx, test_idx, mol_df, X_morgan, seed=42):
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    preds = train_direct_mlp(X_morgan[train_idx], train_y, X_morgan[test_idx], seed=seed)
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 3: Morgan MLP + kinase pretrain (dual-objective)
# ────────────────────────────────────────────────────────────────────────────
class DualObjective(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.3):
        super().__init__()
        encoder = []
        d = in_dim
        for h in hidden:
            encoder += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*encoder)
        self.enc_dim = hidden[-1]
        self.delta_head = nn.Sequential(
            nn.Linear(self.enc_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        self.abs_head = nn.Sequential(
            nn.Linear(self.enc_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, xa, xb):
        ea, eb = self.encoder(xa), self.encoder(xb)
        delta = self.delta_head(torch.cat([ea, eb], dim=-1)).squeeze(-1)
        return delta, self.abs_head(ea).squeeze(-1), self.abs_head(eb).squeeze(-1)

    def predict_abs(self, x):
        return self.abs_head(self.encoder(x)).squeeze(-1)


def pretrain_dual_kinase(kinase_df, X_emb_by_smiles, in_dim, epochs=20, batch_size=512, lr=1e-3, seed=42):
    """Pretrain DualObjective on kinase MMPs."""
    torch.manual_seed(seed); np.random.seed(seed)
    smiles_a = kinase_df["mol_a"].values
    smiles_b = kinase_df["mol_b"].values
    delta = kinase_df["delta"].values.astype(np.float32)
    val_a = kinase_df["value_a"].values.astype(np.float32)
    val_b = kinase_df["value_b"].values.astype(np.float32)

    # Embed
    Xa = np.stack([X_emb_by_smiles[s] for s in smiles_a])
    Xb = np.stack([X_emb_by_smiles[s] for s in smiles_b])

    scaler = StandardScaler().fit(np.vstack([Xa, Xb]))
    Xa = scaler.transform(Xa).astype(np.float32)
    Xb = scaler.transform(Xb).astype(np.float32)

    model = DualObjective(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xa_t = torch.FloatTensor(Xa).to(DEVICE)
    Xb_t = torch.FloatTensor(Xb).to(DEVICE)
    d_t = torch.FloatTensor(delta).to(DEVICE)
    va_t = torch.FloatTensor(val_a).to(DEVICE)
    vb_t = torch.FloatTensor(val_b).to(DEVICE)

    n = len(delta)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            bi = perm[start:start + batch_size]
            opt.zero_grad()
            d_pred, a_pred, b_pred = model(Xa_t[bi], Xb_t[bi])
            ls = loss_fn(d_pred, d_t[bi]) + loss_fn(a_pred, va_t[bi]) + loss_fn(b_pred, vb_t[bi])
            ls.backward()
            opt.step()
    return model, scaler


def train_eval_pretrain_mlp(train_idx, test_idx, mol_df, X_morgan, kinase_df, seed=42):
    """Pretrain dual-obj on kinase, finetune absolute head on ZAP70 train fold."""
    torch.manual_seed(seed); np.random.seed(seed)

    # Build embedding lookup for kinase
    kinase_smiles = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    X_kinase = compute_morgan(kinase_smiles)
    emb_by_smi = {s: X_kinase[i] for i, s in enumerate(kinase_smiles)}

    # Pretrain
    model, scaler = pretrain_dual_kinase(kinase_df, emb_by_smi, X_morgan.shape[1], epochs=15, seed=seed)

    # Finetune on ZAP70 train fold (absolute head only, keep encoder warm)
    train_y = mol_df.iloc[train_idx]["pIC50"].values.astype(np.float32)
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    X_tr = scaler.transform(X_morgan[train_idx]).astype(np.float32)
    X_te = scaler.transform(X_morgan[test_idx]).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xtr_t = torch.FloatTensor(X_tr).to(DEVICE)
    ytr_t = torch.FloatTensor(train_y).to(DEVICE)
    Xte_t = torch.FloatTensor(X_te).to(DEVICE)

    n_val = max(10, len(X_tr) // 5)
    perm = np.random.RandomState(seed).permutation(len(X_tr))
    val_idx, tr_only_idx = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(100):
        model.train()
        perm_tr = np.random.permutation(len(tr_only_idx))
        for start in range(0, len(tr_only_idx), 32):
            bi = tr_only_idx[perm_tr[start:start + 32]]
            opt.zero_grad()
            pred = model.predict_abs(Xtr_t[bi])
            ls = loss_fn(pred, ytr_t[bi])
            ls.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model.predict_abs(Xtr_t[val_idx]), ytr_t[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state); model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model.predict_abs(Xte_t).cpu().numpy()
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 4: ChemBERTa-2 MTR + MLP head
# ────────────────────────────────────────────────────────────────────────────
_CHEMBERTA_CACHE = None
def compute_chemberta_emb(smiles_list, model_name="DeepChem/ChemBERTa-77M-MTR"):
    global _CHEMBERTA_CACHE
    cache_path = RESULTS_DIR / "chemberta_zap70_emb.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        cache_smiles = d["smiles"].tolist()
        cache_emb = d["emb"]
        cache_map = {s: cache_emb[i] for i, s in enumerate(cache_smiles)}
        if all(s in cache_map for s in smiles_list):
            return np.stack([cache_map[s] for s in smiles_list])

    from transformers import AutoTokenizer, AutoModel
    print(f"[CB] Computing ChemBERTa-2 MTR embeddings for {len(smiles_list)} mols...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), 32):
            batch = smiles_list[i:i + 32]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            h = model(**enc).last_hidden_state  # [B, L, D]
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    np.savez(cache_path, smiles=np.array(smiles_list), emb=emb)
    return emb


def train_eval_chemberta(train_idx, test_idx, mol_df, seed=42):
    smiles_all = mol_df["smiles"].tolist()
    X_cb = compute_chemberta_emb(smiles_all)
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    preds = train_direct_mlp(X_cb[train_idx], train_y, X_cb[test_idx],
                              hidden=(256, 128), dropout=0.3, seed=seed)
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 5: ChemProp featurizer (= Morgan via chemprop) + MLP
# ────────────────────────────────────────────────────────────────────────────
def train_eval_chemprop_mlp(train_idx, test_idx, mol_df, X_morgan, seed=42):
    """Per Phase 1 finding, ChemProp featurizer = Morgan. We use the same Morgan X but with a different MLP config."""
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    preds = train_direct_mlp(X_morgan[train_idx], train_y, X_morgan[test_idx],
                              hidden=(1024, 512, 128), dropout=0.4, seed=seed)
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 6: DeepDelta (concat MLP on delta)
# ────────────────────────────────────────────────────────────────────────────
class DeepDeltaMLP(nn.Module):
    def __init__(self, in_dim, hidden=(1024, 512, 256), dropout=0.3):
        super().__init__()
        layers = []
        d = in_dim * 3  # concat [a, b, b-a]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xa, xb):
        x = torch.cat([xa, xb, xb - xa], dim=-1)
        return self.net(x).squeeze(-1)


def train_eval_deepdelta(train_idx, test_idx, mol_df, X_morgan, max_epochs=80, lr=1e-3, batch_size=128, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    train_emb = X_morgan[train_idx]
    test_emb = X_morgan[test_idx]

    n_tr = len(train_idx)
    ii, jj = np.triu_indices(n_tr, k=1)
    emb_a_tr = train_emb[ii]; emb_b_tr = train_emb[jj]
    delta_tr = (train_y[jj] - train_y[ii]).astype(np.float32)

    scaler = StandardScaler().fit(np.vstack([emb_a_tr, emb_b_tr]))
    Xa_tr = scaler.transform(emb_a_tr).astype(np.float32)
    Xb_tr = scaler.transform(emb_b_tr).astype(np.float32)

    model = DeepDeltaMLP(X_morgan.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xa_t = torch.FloatTensor(Xa_tr).to(DEVICE)
    Xb_t = torch.FloatTensor(Xb_tr).to(DEVICE)
    d_t = torch.FloatTensor(delta_tr).to(DEVICE)
    n = len(delta_tr)
    # Train/val split
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(50, n // 10)
    val_idx = perm[:n_val]; tr_only = perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        perm_tr = np.random.permutation(len(tr_only))
        for start in range(0, len(tr_only), batch_size):
            bi = tr_only[perm_tr[start:start + batch_size]]
            opt.zero_grad()
            pred = model(Xa_t[bi], Xb_t[bi])
            ls = loss_fn(pred, d_t[bi])
            ls.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xa_t[val_idx], Xb_t[val_idx]), d_t[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state); model.to(DEVICE)
    model.eval()

    # Anchor reconstruction
    def pred_fn(a_np, b_np):
        a_s = scaler.transform(a_np).astype(np.float32)
        b_s = scaler.transform(b_np).astype(np.float32)
        with torch.no_grad():
            preds = []
            for i in range(0, len(a_s), 4096):
                A = torch.FloatTensor(a_s[i:i + 4096]).to(DEVICE)
                B = torch.FloatTensor(b_s[i:i + 4096]).to(DEVICE)
                preds.append(model(A, B).cpu().numpy())
            return np.concatenate(preds)

    abs_pred = delta_to_abs_via_anchors(pred_fn, train_emb, train_y, test_emb, method="median")

    # Delta test metric
    n_te = len(test_idx)
    ii_t, jj_t = np.triu_indices(n_te, k=1)
    delta_te_true = (test_y[jj_t] - test_y[ii_t]).astype(np.float32)
    delta_te_pred = pred_fn(test_emb[ii_t], test_emb[jj_t])
    delta_mae = float(np.mean(np.abs(delta_te_true - delta_te_pred)))

    metrics = compute_abs_metrics(test_y, abs_pred)
    metrics["delta_mae"] = delta_mae
    return metrics, abs_pred


# ────────────────────────────────────────────────────────────────────────────
# Model 7: Bootstrap ensemble of FiLMDelta
# ────────────────────────────────────────────────────────────────────────────
def train_eval_filmdelta_bootstrap(train_idx, test_idx, mol_df, X_morgan, n_boot=20, seed=42):
    """Bootstrap n_boot FiLMDelta predictors, take median absolute prediction."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    train_emb = X_morgan[train_idx]; test_emb = X_morgan[test_idx]
    n_tr = len(train_idx)
    ii_full, jj_full = np.triu_indices(n_tr, k=1)
    delta_full = (train_y[jj_full] - train_y[ii_full]).astype(np.float32)
    emb_a_full = train_emb[ii_full]; emb_b_full = train_emb[jj_full]
    n_pairs = len(delta_full)

    abs_preds_all = []
    for b in range(n_boot):
        rng = np.random.RandomState(seed + b)
        boot_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        predictor = FiLMDeltaPredictor(
            dropout=0.2, learning_rate=1e-3, batch_size=128,
            max_epochs=50, patience=10, device=DEVICE,
        )
        predictor.fit(emb_a_full[boot_idx], emb_b_full[boot_idx], delta_full[boot_idx], verbose=False)
        def pf(a, b_):
            return predictor.predict(a, b_)
        abs_pred_b = delta_to_abs_via_anchors(pf, train_emb, train_y, test_emb, method="median")
        abs_preds_all.append(abs_pred_b)
        del predictor; gc.collect()

    abs_preds_all = np.stack(abs_preds_all, axis=0)  # [B, Nte]
    abs_pred = np.median(abs_preds_all, axis=0)
    abs_std = abs_preds_all.std(axis=0)
    metrics = compute_abs_metrics(test_y, abs_pred)
    metrics["mean_uncertainty_std"] = float(abs_std.mean())
    return metrics, abs_pred


# ────────────────────────────────────────────────────────────────────────────
# Model 8: Classification cascade
# ────────────────────────────────────────────────────────────────────────────
def train_eval_classifier_cascade(train_idx, test_idx, mol_df, X_morgan, kinase_df, seed=42, active_thresh=6.0):
    """Step 1: binary 'kinase-active' classifier (XGB) on 32K kinase + ZAP70 train.
    Step 2: XGB regressor on positives only.
    Final: pIC50 = p(active) * pIC50_pred + (1 - p) * baseline(=mean inactive train pIC50).
    """
    from xgboost import XGBClassifier, XGBRegressor

    # Build the kinase background mols (use mol_a values as activity proxies)
    kinase_smi = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    kinase_val = {}
    for _, row in kinase_df.iterrows():
        kinase_val.setdefault(row["mol_a"], []).append(row["value_a"])
        kinase_val.setdefault(row["mol_b"], []).append(row["value_b"])
    kinase_y = np.array([np.mean(kinase_val[s]) for s in kinase_smi], dtype=np.float32)
    X_kinase_pool = compute_morgan(kinase_smi)

    # Step 1: binary classifier
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    train_X = X_morgan[train_idx]
    pool_X = np.vstack([X_kinase_pool, train_X])
    pool_y = np.concatenate([kinase_y, train_y])
    pool_label = (pool_y >= active_thresh).astype(int)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        verbosity=0, n_jobs=4, random_state=seed,
    )
    clf.fit(pool_X, pool_label)

    # Step 2: regressor on positives only
    pos_mask = pool_label == 1
    if pos_mask.sum() < 20:
        # Fallback
        reg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, verbosity=0, n_jobs=4, random_state=seed)
        reg.fit(pool_X, pool_y)
        baseline = float(pool_y[~pos_mask].mean()) if (~pos_mask).any() else 5.0
    else:
        reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=4, random_state=seed)
        reg.fit(pool_X[pos_mask], pool_y[pos_mask])
        baseline = float(pool_y[~pos_mask].mean()) if (~pos_mask).any() else 5.0

    # Predict
    test_X = X_morgan[test_idx]
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    p_active = clf.predict_proba(test_X)[:, 1]
    yhat_pos = reg.predict(test_X)
    preds = p_active * yhat_pos + (1 - p_active) * baseline

    metrics = compute_abs_metrics(test_y, preds)
    metrics["frac_predicted_active"] = float((p_active >= 0.5).mean())
    return metrics, preds


# ────────────────────────────────────────────────────────────────────────────
# Model 9: Multi-task with target embedding
# ────────────────────────────────────────────────────────────────────────────
class MultiTaskTargetMLP(nn.Module):
    def __init__(self, in_dim, n_targets, target_emb_dim=32, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.target_emb = nn.Embedding(n_targets, target_emb_dim)
        layers = []
        d = in_dim + target_emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, tgt_idx):
        te = self.target_emb(tgt_idx)
        return self.net(torch.cat([x, te], dim=-1)).squeeze(-1)


def train_eval_multitask(train_idx, test_idx, mol_df, X_morgan, kinase_df, seed=42):
    """Joint training on 32K kinase (per-target) + 280 ZAP70 train fold. Test on ZAP70 test fold with ZAP70 target embedding."""
    torch.manual_seed(seed); np.random.seed(seed)

    # Build full training pool: kinase (mol_a only with value_a) + ZAP70 train
    kinase_smi = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    kinase_to_target = {}  # smi -> list of (target, value)
    for _, r in kinase_df.iterrows():
        kinase_to_target.setdefault(r["mol_a"], []).append((r["target_chembl_id"], r["value_a"]))
        kinase_to_target.setdefault(r["mol_b"], []).append((r["target_chembl_id"], r["value_b"]))
    # Average per (mol, target)
    rows = []
    for smi, lst in kinase_to_target.items():
        df_l = pd.DataFrame(lst, columns=["t", "v"]).groupby("t").v.mean().reset_index()
        for _, r in df_l.iterrows():
            rows.append((smi, r["t"], r["v"]))
    kinase_pool = pd.DataFrame(rows, columns=["smiles", "target", "pIC50"])

    zap_train_pool = pd.DataFrame({
        "smiles": mol_df.iloc[train_idx]["smiles"].values,
        "target": TARGET_ID,
        "pIC50": mol_df.iloc[train_idx]["pIC50"].values,
    })

    full_pool = pd.concat([kinase_pool, zap_train_pool], ignore_index=True)
    all_targets = sorted(full_pool["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    zap_idx = target_to_idx[TARGET_ID]

    # Compute morgan for full pool (cache via dict)
    pool_smi = full_pool["smiles"].tolist()
    pool_emb = compute_morgan(pool_smi)
    pool_y = full_pool["pIC50"].values.astype(np.float32)
    pool_tgt = np.array([target_to_idx[t] for t in full_pool["target"]], dtype=np.int64)

    scaler = StandardScaler().fit(pool_emb)
    pool_emb_s = scaler.transform(pool_emb).astype(np.float32)

    model = MultiTaskTargetMLP(pool_emb_s.shape[1], len(all_targets)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Hold out small validation from the ZAP70 portion (not test) to early-stop
    zap_pool_mask = pool_tgt == zap_idx
    zap_pool_idx_global = np.where(zap_pool_mask)[0]
    rng = np.random.RandomState(seed)
    val_zap = rng.choice(zap_pool_idx_global, size=max(10, len(zap_pool_idx_global) // 5), replace=False)
    train_pool_mask = np.ones(len(pool_y), dtype=bool); train_pool_mask[val_zap] = False
    train_pool_idx = np.where(train_pool_mask)[0]

    X_t = torch.FloatTensor(pool_emb_s).to(DEVICE)
    y_t = torch.FloatTensor(pool_y).to(DEVICE)
    tgt_t = torch.LongTensor(pool_tgt).to(DEVICE)

    best_val, best_state, wait, bs = float("inf"), None, 0, 256
    for ep in range(30):
        model.train()
        perm = np.random.permutation(len(train_pool_idx))
        for start in range(0, len(perm), bs):
            bi = train_pool_idx[perm[start:start + bs]]
            opt.zero_grad()
            pred = model(X_t[bi], tgt_t[bi])
            ls = loss_fn(pred, y_t[bi])
            ls.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(X_t[val_zap], tgt_t[val_zap]), y_t[val_zap]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state); model.to(DEVICE)
    model.eval()

    # Predict on ZAP70 test fold with ZAP70 target embedding
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    test_emb_s = scaler.transform(X_morgan[test_idx]).astype(np.float32)
    Xte_t = torch.FloatTensor(test_emb_s).to(DEVICE)
    tgt_te = torch.LongTensor(np.full(len(test_idx), zap_idx, dtype=np.int64)).to(DEVICE)
    with torch.no_grad():
        preds = model(Xte_t, tgt_te).cpu().numpy()
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 10: XGBoost multi-FP
# ────────────────────────────────────────────────────────────────────────────
def train_eval_xgb_multifp(train_idx, test_idx, mol_df, X_morgan, X_rdkit, X_maccs, X_atompair, seed=42):
    from xgboost import XGBRegressor
    X_all = np.hstack([X_morgan, X_rdkit, X_maccs, X_atompair])
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    reg = XGBRegressor(
        n_estimators=749, max_depth=6, min_child_weight=2,
        subsample=0.605, colsample_bytree=0.520, learning_rate=0.0197,
        reg_alpha=1.579, reg_lambda=7.313,
        verbosity=0, n_jobs=4, random_state=seed,
    )
    reg.fit(X_all[train_idx], train_y)
    preds = reg.predict(X_all[test_idx])
    return compute_abs_metrics(test_y, preds), preds


# ────────────────────────────────────────────────────────────────────────────
# Model 11: FiLMDelta + kinase pretrain
# ────────────────────────────────────────────────────────────────────────────
def train_eval_filmdelta_pretrain(train_idx, test_idx, mol_df, X_morgan, kinase_df, seed=42,
                                   pretrain_epochs=8, finetune_epochs=40):
    """Pretrain FiLMDelta on 32K kinase MMP pairs (delta head), then finetune on ZAP70 all-pairs."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP
    torch.manual_seed(seed); np.random.seed(seed)

    # Build kinase pretrain data
    kinase_smi = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    X_kinase = compute_morgan(kinase_smi)
    smi_to_emb = {s: X_kinase[i] for i, s in enumerate(kinase_smi)}
    emb_a_pre = np.stack([smi_to_emb[s] for s in kinase_df["mol_a"]])
    emb_b_pre = np.stack([smi_to_emb[s] for s in kinase_df["mol_b"]])
    delta_pre = kinase_df["delta"].values.astype(np.float32)

    # Create FiLMDelta and pretrain
    pretrain = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=256,
        max_epochs=pretrain_epochs, patience=pretrain_epochs, device=DEVICE,
    )
    pretrain.fit(emb_a_pre, emb_b_pre, delta_pre, verbose=False)

    # Now finetune on ZAP70 all-pairs (within train fold)
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    train_emb = X_morgan[train_idx]; test_emb = X_morgan[test_idx]
    n_tr = len(train_idx)
    ii, jj = np.triu_indices(n_tr, k=1)
    emb_a_tr = train_emb[ii]; emb_b_tr = train_emb[jj]
    delta_tr = (train_y[jj] - train_y[ii]).astype(np.float32)

    # Reuse the pretrained model: do a few more epochs of training on ZAP70 pairs (transfer init)
    pretrain.max_epochs = finetune_epochs
    pretrain.patience = 8
    # Keep current model state (do not reinit); we call fit again which re-creates from scratch in this class.
    # So instead: manually run a finetune loop on the existing model.
    model = pretrain.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    emb_a_t = torch.FloatTensor(emb_a_tr).to(DEVICE)
    emb_b_t = torch.FloatTensor(emb_b_tr).to(DEVICE)
    d_t = torch.FloatTensor(delta_tr).to(DEVICE)
    n = len(delta_tr)
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(50, n // 10)
    val_idx_p, tr_only_p = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(finetune_epochs):
        model.train()
        pt = np.random.permutation(len(tr_only_p))
        for start in range(0, len(tr_only_p), 128):
            bi = tr_only_p[pt[start:start + 128]]
            opt.zero_grad()
            pred = model(emb_a_t[bi], emb_b_t[bi])
            ls = loss_fn(pred, d_t[bi])
            ls.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(emb_a_t[val_idx_p], emb_b_t[val_idx_p]), d_t[val_idx_p]).item())
        sch.step(vl)
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state); model.to(DEVICE)
    model.eval()

    # Anchor reconstruction
    def pred_fn(a_np, b_np):
        with torch.no_grad():
            outs = []
            for i in range(0, len(a_np), 8192):
                A = torch.FloatTensor(a_np[i:i + 8192]).to(DEVICE)
                B = torch.FloatTensor(b_np[i:i + 8192]).to(DEVICE)
                outs.append(model(A, B).cpu().numpy())
            return np.concatenate(outs)

    abs_pred = delta_to_abs_via_anchors(pred_fn, train_emb, train_y, test_emb, method="median")

    # Delta test metric
    n_te = len(test_idx)
    ii_t, jj_t = np.triu_indices(n_te, k=1)
    delta_te_true = (test_y[jj_t] - test_y[ii_t]).astype(np.float32)
    delta_te_pred = pred_fn(test_emb[ii_t], test_emb[jj_t])
    delta_mae = float(np.mean(np.abs(delta_te_true - delta_te_pred)))

    metrics = compute_abs_metrics(test_y, abs_pred)
    metrics["delta_mae"] = delta_mae
    metrics["n_pretrain_pairs"] = int(len(delta_pre))
    return metrics, abs_pred


# ────────────────────────────────────────────────────────────────────────────
# v7 reproduction trainers
# ────────────────────────────────────────────────────────────────────────────
def train_eval_v7g1_xgb_morgan(train_idx, test_idx, mol_df, X_morgan, seed=42):
    """v7 Phase G1: XGBoost on Morgan binary (Phase B's XGBoost_baseline = same)."""
    from xgboost import XGBRegressor
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    reg = XGBRegressor(
        n_estimators=749, max_depth=6, min_child_weight=2,
        subsample=0.605, colsample_bytree=0.520, learning_rate=0.0197,
        reg_alpha=1.579, reg_lambda=7.313,
        verbosity=0, n_jobs=4, random_state=seed,
    )
    reg.fit(X_morgan[train_idx], train_y)
    preds = reg.predict(X_morgan[test_idx])
    return compute_abs_metrics(test_y, preds), preds


def train_eval_v7h1_morgan_knn10(train_idx, test_idx, mol_df, X_morgan, seed=42):
    """v7 Phase H1: Tanimoto-weighted kNN-10 on Morgan binary."""
    # Tanimoto distance: D(x,y) = 1 - |x AND y| / |x OR y|
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    X_tr = (X_morgan[train_idx] > 0).astype(np.float32)
    X_te = (X_morgan[test_idx] > 0).astype(np.float32)
    K = 10
    preds = []
    for i in range(len(X_te)):
        x = X_te[i]
        inter = (X_tr * x).sum(axis=1)
        union = ((X_tr + x) > 0).astype(np.float32).sum(axis=1)
        sim = inter / np.clip(union, 1, None)
        # top-K by similarity
        top = np.argsort(-sim)[:K]
        w = sim[top]
        if w.sum() < 1e-6:
            preds.append(float(train_y.mean()))
        else:
            preds.append(float(np.sum(train_y[top] * w) / w.sum()))
    preds = np.array(preds)
    return compute_abs_metrics(test_y, preds), preds


def train_eval_v7b_mlp_morgan(train_idx, test_idx, mol_df, X_morgan, seed=42):
    """v7 Phase B: Direct MLP on Morgan (3-layer [512,256,128], dropout 0.3)."""
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    preds = train_direct_mlp(X_morgan[train_idx], train_y, X_morgan[test_idx],
                              hidden=(512, 256, 128), dropout=0.3,
                              lr=1e-3, epochs=200, batch_size=32, patience=20, seed=seed)
    return compute_abs_metrics(test_y, preds), preds


def train_eval_v7c_xgb_interpretable(train_idx, test_idx, mol_df, X_morgan, seed=42):
    """v7 Phase C: XGBoost on RDKit 2D + functional groups + MACCS (interpretable feature set)."""
    from xgboost import XGBRegressor
    smiles_all = mol_df["smiles"].tolist()

    # Build interpretable feature set
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rdkit_feats = []
    for s in smiles_all:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rdkit_feats.append(np.zeros(len(desc_names), dtype=np.float32))
        else:
            try:
                rdkit_feats.append(np.array(calc.CalcDescriptors(m), dtype=np.float32))
            except Exception:
                rdkit_feats.append(np.zeros(len(desc_names), dtype=np.float32))
    X_rd = np.nan_to_num(np.array(rdkit_feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_macc = compute_maccs(smiles_all)
    X_all = np.hstack([X_rd, X_macc])
    train_y = mol_df.iloc[train_idx]["pIC50"].values
    test_y = mol_df.iloc[test_idx]["pIC50"].values
    reg = XGBRegressor(
        n_estimators=500, max_depth=5, min_child_weight=2,
        subsample=0.7, colsample_bytree=0.6, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=5.0,
        verbosity=0, n_jobs=4, random_state=seed,
    )
    reg.fit(X_all[train_idx], train_y)
    preds = reg.predict(X_all[test_idx])
    return compute_abs_metrics(test_y, preds), preds


# v7A = same as Model 1 FiLMDelta
# v7D = same as Model 3 pretrain_mlp (dual-objective MLP + kinase pretrain)


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # v7 reproduction track
    "v7A_FiLMDelta_repro": "v7a_filmdelta",
    "v7B_MLP_Morgan_repro": "v7b_mlp_morgan",
    "v7C_XGB_Interpretable_repro": "v7c_xgb_interp",
    "v7D_DualObj_Pretrain_repro": "v7d_dualobj_pretrain",
    "v7G1_XGB_Morgan_repro": "v7g1_xgb_morgan",
    "v7H1_Morgan_KNN10_repro": "v7h1_morgan_knn10",
    # 11 challenge models
    "1_FiLMDelta": "filmdelta",
    "2_DirectMorganMLP": "direct_morgan",
    "3_MorganMLP_KinasePretrain": "pretrain_mlp",
    "4_ChemBERTa_MTR": "chemberta",
    "5_ChemPropMLP": "chemprop_mlp",
    "6_DeepDelta": "deepdelta",
    "7_FiLMDelta_Bootstrap": "filmdelta_boot",
    "8_ClassificationCascade": "cls_cascade",
    "9_MultiTask_TargetEmb": "multitask",
    "10_XGB_MultiFP": "xgb_multifp",
    "11_FiLMDelta_KinasePretrain": "filmdelta_pretrain",
}


def run_model(model_key: str, mol_df, X_morgan, X_rdkit, X_maccs, X_atompair, kinase_df,
              splits_random, splits_butina, seed=42):
    model_id = MODEL_REGISTRY[model_key]
    print(f"\n{'=' * 70}\n[MODEL] {model_key}\n{'=' * 70}")
    fold_results = {"random": [], "butina": []}
    fold_preds = {"random": [], "butina": []}

    for split_name, splits in [("random", splits_random), ("butina", splits_butina)]:
        print(f"  Split = {split_name}")
        for fi, (tr_i, te_i) in enumerate(splits):
            t0 = time.time()
            try:
                if model_id in ("filmdelta", "v7a_filmdelta"):
                    m, p = train_eval_filmdelta(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "direct_morgan":
                    m, p = train_eval_direct_morgan_mlp(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id in ("pretrain_mlp", "v7d_dualobj_pretrain"):
                    m, p = train_eval_pretrain_mlp(tr_i, te_i, mol_df, X_morgan, kinase_df, seed=seed + fi)
                elif model_id == "chemberta":
                    m, p = train_eval_chemberta(tr_i, te_i, mol_df, seed=seed + fi)
                elif model_id == "chemprop_mlp":
                    m, p = train_eval_chemprop_mlp(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "deepdelta":
                    m, p = train_eval_deepdelta(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "filmdelta_boot":
                    # B=10 on V100 (3.5 min/model × 10 × 10 folds ≈ 6 h); reduce to 10 to fit time budget
                    m, p = train_eval_filmdelta_bootstrap(tr_i, te_i, mol_df, X_morgan, n_boot=10, seed=seed + fi)
                elif model_id == "cls_cascade":
                    m, p = train_eval_classifier_cascade(tr_i, te_i, mol_df, X_morgan, kinase_df, seed=seed + fi)
                elif model_id == "multitask":
                    m, p = train_eval_multitask(tr_i, te_i, mol_df, X_morgan, kinase_df, seed=seed + fi)
                elif model_id == "xgb_multifp":
                    m, p = train_eval_xgb_multifp(tr_i, te_i, mol_df, X_morgan, X_rdkit, X_maccs, X_atompair, seed=seed + fi)
                elif model_id == "filmdelta_pretrain":
                    m, p = train_eval_filmdelta_pretrain(tr_i, te_i, mol_df, X_morgan, kinase_df, seed=seed + fi)
                elif model_id == "v7g1_xgb_morgan":
                    m, p = train_eval_v7g1_xgb_morgan(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "v7h1_morgan_knn10":
                    m, p = train_eval_v7h1_morgan_knn10(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "v7b_mlp_morgan":
                    m, p = train_eval_v7b_mlp_morgan(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                elif model_id == "v7c_xgb_interp":
                    m, p = train_eval_v7c_xgb_interpretable(tr_i, te_i, mol_df, X_morgan, seed=seed + fi)
                else:
                    raise ValueError(f"Unknown model_id {model_id}")
                dt = time.time() - t0
                m["wall_time_s"] = dt
                m["fold"] = fi
                m["split"] = split_name
                fold_results[split_name].append(m)
                fold_preds[split_name].append({"test_idx": te_i.tolist(), "preds": p.tolist(),
                                                "y_true": mol_df.iloc[te_i]["pIC50"].values.tolist()})
                print(f"    fold {fi}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.3f}, "
                      f"Spr={m['spearman_r']:.3f}, Pr={m['pearson_r']:.3f} [{dt:.1f}s]")
            except Exception as e:
                import traceback
                print(f"    fold {fi} FAILED: {e}\n{traceback.format_exc()}")
                fold_results[split_name].append({"fold": fi, "split": split_name, "error": str(e)})

    # Aggregate
    aggregated = {}
    for split_name in ("random", "butina"):
        ok = [m for m in fold_results[split_name] if "mae" in m]
        if ok:
            aggregated[split_name] = aggregate_folds(ok)
            print(f"  [{split_name}] MAE = {aggregated[split_name]['mae_mean']:.4f}±{aggregated[split_name]['mae_std']:.4f}, "
                  f"Spr = {aggregated[split_name]['spearman_r_mean']:.3f}")

    # Save predictions
    with open(PREDS_DIR / f"{model_key}_preds.json", "w") as f:
        json.dump({"per_fold_metrics": fold_results, "predictions": fold_preds}, f)

    return {"per_fold": fold_results, "aggregated": aggregated}


def load_existing_results() -> Dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: Dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Also export CSV
    rows = []
    for model, info in results.items():
        if not isinstance(info, dict) or "per_fold" not in info:
            continue
        for split_name, fold_list in info["per_fold"].items():
            for fm in fold_list:
                if "mae" not in fm:
                    continue
                # Emit canonical metric names: mae, rmse, r2, spearman, pearson
                metric_map = {"mae": "mae", "rmse": "rmse", "r2": "r2",
                              "spearman": "spearman_r", "pearson": "pearson_r"}
                for csv_metric, key in metric_map.items():
                    rows.append({"model": model, "fold": fm["fold"], "split_type": split_name,
                                 "metric": csv_metric, "value": fm.get(key, np.nan)})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"[SAVE] CSV → {RESULTS_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="comma-separated model numbers (1-10), or 'all', or '0' for reproduction smoke test")
    ap.add_argument("--seed", type=int, default=CV_SEED)
    args = ap.parse_args()

    print(f"\n{'#' * 72}\n# ZAP70 Challenge — 10-model bake-off (seed={args.seed})\n{'#' * 72}\n")
    t_total = time.time()

    # Load
    mol_df = load_zap70_mols()
    kinase_df = load_kinase_pretrain()
    smiles = mol_df["smiles"].tolist()
    print("[FP] Computing Morgan / RDKit / MACCS / AtomPair for 280 ZAP70 mols...")
    X_morgan = compute_morgan(smiles)
    X_rdkit = compute_rdkit_fp(smiles)
    X_maccs = compute_maccs(smiles)
    X_atompair = compute_atompair(smiles)
    print(f"  X_morgan: {X_morgan.shape}, X_rdkit: {X_rdkit.shape}, X_maccs: {X_maccs.shape}, X_atompair: {X_atompair.shape}")

    # Splits (precompute)
    splits_random = random_kfold(mol_df)
    splits_butina = butina_groupkfold(mol_df)
    print(f"[SPLIT] Random fold sizes: {[len(t) for t,_ in splits_random]} / {[len(te) for _,te in splits_random]}")
    print(f"[SPLIT] Butina fold sizes: {[len(t) for t,_ in splits_butina]} / {[len(te) for _,te in splits_butina]}")

    # Save fold definitions for reproducibility
    fold_def = {
        "random": [{"fold": i, "train_idx": tr.tolist(), "test_idx": te.tolist()} for i, (tr, te) in enumerate(splits_random)],
        "butina": [{"fold": i, "train_idx": tr.tolist(), "test_idx": te.tolist()} for i, (tr, te) in enumerate(splits_butina)],
    }
    with open(RESULTS_DIR / "fold_definitions.json", "w") as f:
        json.dump(fold_def, f, indent=2)

    # Determine which models
    if args.models == "all":
        wanted = list(MODEL_REGISTRY.keys())
    elif args.models == "0":
        # Smoke-test mode: FiLMDelta only on the first random fold
        print("\n[PHASE 0] Reproduction smoke test: FiLMDelta on fold 0 (random split)")
        tr_i, te_i = splits_random[0]
        t0 = time.time()
        m, _ = train_eval_filmdelta(tr_i, te_i, mol_df, X_morgan, seed=args.seed)
        print(f"\n  fold 0: MAE={m['mae']:.4f} (abs), delta_MAE={m['delta_mae']:.4f}, "
              f"Spr={m['spearman_r']:.3f}, R²={m['r2']:.3f}, time={time.time()-t0:.1f}s")
        print(f"\n[GATE] v7 baseline delta MAE = 0.877 (pair-level); current run = {m['delta_mae']:.4f}")
        print(f"       Within ±0.05? {abs(m['delta_mae'] - 0.877) <= 0.05}")
        return
    elif args.models == "v7":
        # v7 reproduction track only
        wanted = [k for k in MODEL_REGISTRY if k.startswith("v7")]
    elif args.models == "v7_gate":
        # Quick gate: G1 + H1 only (~10 min CPU)
        wanted = ["v7G1_XGB_Morgan_repro", "v7H1_Morgan_KNN10_repro"]
    else:
        parts = [x.strip() for x in args.models.split(",")]
        wanted = []
        for p in parts:
            if p.isdigit():
                num = int(p)
                wanted.extend([k for k in MODEL_REGISTRY if not k.startswith("v7") and int(k.split("_")[0]) == num])
            else:
                # v7 phase key like "v7G1" or "G1"
                pp = p.lower().replace("v7", "")
                wanted.extend([k for k in MODEL_REGISTRY if k.startswith("v7") and pp in k.lower().split("_")[0].replace("v7","")])
    print(f"\n[RUN] Models: {wanted}")

    results = load_existing_results()
    for model_key in wanted:
        if model_key in results and "aggregated" in results.get(model_key, {}):
            print(f"\n[SKIP] {model_key} already complete.")
            continue
        info = run_model(model_key, mol_df, X_morgan, X_rdkit, X_maccs, X_atompair, kinase_df,
                          splits_random, splits_butina, seed=args.seed)
        results[model_key] = info
        save_results(results)
        gc.collect()

    print(f"\n[DONE] Total wall time: {(time.time() - t_total)/60:.1f} min")
    print(f"[SAVE] Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
