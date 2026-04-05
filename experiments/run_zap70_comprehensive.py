#!/usr/bin/env python3
"""
ZAP70 (CHEMBL2803) Comprehensive Method Comparison.

Iterative experiment comparing prediction methods on ZAP70:
  - Iteration 1: Data characterization + baselines (MMP fraction, noise analysis)
  - Iteration 2: All methods comparison (abs, delta, ranking metrics)
  - Iteration 3: Virtual screening comparison (which molecules does each method prioritize?)

Methods:
  1. Subtraction (XGBoost): Train on absolute pIC50, delta = f(B) - f(A)
  2. FiLMDelta: Neural FiLM-conditioned f(B|δ)-f(A|δ)
  3. Dual-objective: Shared encoder, joint abs+delta loss
  4. XGB + pairwise ranking: Custom objective MSE + concordance
  5. FiLMDelta + ranking loss: FiLM with added concordance loss

Evaluation dimensions:
  - Absolute IC50: MAE, Spearman, Pearson on held-out molecules
  - Delta prediction: MAE, Spearman on test molecule pairs
  - Pair ranking: Concordance index, top-K precision
  - Virtual screening: Recall@K for identifying most potent molecules

Usage:
    conda run -n quris python -u experiments/run_zap70_comprehensive.py
    conda run -n quris python -u experiments/run_zap70_comprehensive.py --iteration 1
    conda run -n quris python -u experiments/run_zap70_comprehensive.py --iteration 2
    conda run -n quris python -u experiments/run_zap70_comprehensive.py --iteration 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
torch.backends.mps.is_available = lambda: False

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFMCS

from experiments.run_paper_evaluation import RESULTS_DIR, CACHE_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    N_FOLDS, CV_SEED,
)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "zap70_comprehensive_results.json"
DEVICE = "cpu"
N_JOBS = 8

# Best XGB hyperparameters from v3 Optuna search
BEST_XGB_PARAMS = {
    "max_depth": 6, "min_child_weight": 2,
    "subsample": 0.605, "colsample_bytree": 0.520,
    "learning_rate": 0.0197, "n_estimators": 749,
    "reg_alpha": 1.579, "reg_lambda": 7.313,
}


def save_results(results):
    """Save results atomically."""
    import fcntl
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_file = RESULTS_FILE.with_suffix(".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if RESULTS_FILE.exists():
                with open(RESULTS_FILE) as f:
                    existing = json.load(f)
                existing.update(results)
                results = existing
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, default=str)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def concordance_index(y_true, y_pred):
    """Compute concordance index (C-index) for pairwise ranking."""
    n = len(y_true)
    if n < 2:
        return 0.5
    concordant = 0
    discordant = 0
    tied = 0
    # Sample if too many pairs
    if n > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 500, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = 500
    for i in range(n):
        for j in range(i + 1, n):
            true_diff = y_true[i] - y_true[j]
            pred_diff = y_pred[i] - y_pred[j]
            if abs(true_diff) < 1e-8:
                continue
            if true_diff * pred_diff > 0:
                concordant += 1
            elif true_diff * pred_diff < 0:
                discordant += 1
            else:
                tied += 1
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def top_k_precision(y_true, y_pred, k=10):
    """Precision: what fraction of top-K predicted are in top-K true?"""
    if len(y_true) < k:
        k = len(y_true)
    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])
    return len(true_top_k & pred_top_k) / k


def evaluate_absolute(y_true, y_pred):
    """Evaluate absolute pIC50 predictions."""
    mae = np.mean(np.abs(y_true - y_pred))
    spr, _ = spearmanr(y_true, y_pred)
    pear, _ = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    top10 = top_k_precision(y_true, y_pred, k=10)
    top20 = top_k_precision(y_true, y_pred, k=20)
    return {
        "mae": float(mae), "spearman": float(spr), "pearson": float(pear),
        "concordance_index": float(ci),
        "top10_precision": float(top10), "top20_precision": float(top20),
        "n": int(len(y_true)),
    }


def evaluate_delta(delta_true, delta_pred):
    """Evaluate delta (pairwise difference) predictions."""
    mae = np.mean(np.abs(delta_true - delta_pred))
    spr, _ = spearmanr(delta_true, delta_pred)
    pear, _ = pearsonr(delta_true, delta_pred)
    ci = concordance_index(delta_true, delta_pred)
    # Sign accuracy: how often do we get the direction right?
    sign_acc = np.mean(np.sign(delta_true) == np.sign(delta_pred))
    # Large delta accuracy: for pairs with |delta| > 1 pIC50
    large_mask = np.abs(delta_true) > 1.0
    large_sign_acc = np.mean(np.sign(delta_true[large_mask]) == np.sign(delta_pred[large_mask])) if large_mask.sum() > 10 else float('nan')
    return {
        "mae": float(mae), "spearman": float(spr), "pearson": float(pear),
        "concordance_index": float(ci),
        "sign_accuracy": float(sign_acc),
        "large_delta_sign_accuracy": float(large_sign_acc),
        "n_pairs": int(len(delta_true)),
        "n_large_delta": int(large_mask.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h, cond):
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        return gamma * h + beta


class FiLMBlock(nn.Module):
    """Linear → ReLU → FiLM → Dropout."""
    def __init__(self, in_dim, out_dim, cond_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = FiLMLayer(cond_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, cond):
        h = F.relu(self.linear(h))
        h = self.film(h, cond)
        return self.dropout(h)


class FiLMDeltaNet(nn.Module):
    """FiLM-conditioned delta predictor: f(B|δ) - f(A|δ)."""
    def __init__(self, input_dim, hidden_dims=[512, 256], cond_dim=256, dropout=0.2):
        super().__init__()
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.blocks = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.blocks.append(FiLMBlock(prev, h, cond_dim, dropout))
            prev = h
        self.output = nn.Linear(prev, 1)

    def forward_single(self, x, cond):
        h = x
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h).squeeze(-1)

    def forward(self, emb_a, emb_b):
        delta_cond = self.delta_encoder(emb_b - emb_a)
        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        return pred_b - pred_a

    def predict_absolute(self, emb):
        """Predict absolute pIC50 using zero conditioning (no edit)."""
        zero_cond = self.delta_encoder(torch.zeros_like(emb))
        return self.forward_single(emb, zero_cond)


class FiLMDeltaWithRanking(FiLMDeltaNet):
    """FiLMDelta with additional concordance ranking loss."""
    def __init__(self, *args, ranking_weight=0.5, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_weight = ranking_weight
        self.margin = margin

    def compute_ranking_loss(self, delta_pred, delta_true):
        """Soft concordance loss on predicted deltas."""
        n = len(delta_pred)
        if n < 4:
            return torch.tensor(0.0)
        n_samples = min(n * 4, n * (n - 1) // 2, 2000)
        idx_i = torch.randint(0, n, (n_samples,))
        idx_j = torch.randint(0, n, (n_samples,))
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        diff_true = delta_true[idx_i] - delta_true[idx_j]
        diff_pred = delta_pred[idx_i] - delta_pred[idx_j]
        sign = torch.sign(diff_true)
        losses = torch.clamp(self.margin - sign * diff_pred, min=0)
        return losses.mean()


class DualObjectiveNet(nn.Module):
    """Shared encoder with abs + delta heads."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout),
            ])
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_dim = hidden_dims[-1]
        self.delta_head = nn.Sequential(
            nn.Linear(self.enc_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.abs_head = nn.Sequential(
            nn.Linear(self.enc_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x_a, x_b):
        enc_a, enc_b = self.encoder(x_a), self.encoder(x_b)
        delta_pred = self.delta_head(torch.cat([enc_a, enc_b], dim=-1)).squeeze(-1)
        abs_a = self.abs_head(enc_a).squeeze(-1)
        abs_b = self.abs_head(enc_b).squeeze(-1)
        return delta_pred, abs_a, abs_b

    def predict_absolute(self, x):
        return self.abs_head(self.encoder(x)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Training functions
# ═══════════════════════════════════════════════════════════════════════════

def train_xgb_absolute(X_train, y_train, X_test, params=None):
    """Train XGBoost for absolute pIC50 prediction."""
    import xgboost as xgb
    p = dict(BEST_XGB_PARAMS) if params is None else dict(params)
    p.update({"n_jobs": N_JOBS, "random_state": 42, "verbosity": 0})
    model = xgb.XGBRegressor(**p)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return pred_train, pred_test, model


def xgb_pairwise_objective(alpha=1.0, n_pair_samples=500):
    """Custom XGBoost objective: MSE + alpha * pairwise concordance."""
    def obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        n = len(y_true)
        residual = y_pred - y_true
        grad = 2 * residual
        hess = 2 * np.ones_like(residual)

        grad_rank = np.zeros_like(residual)
        rng = np.random.RandomState(42)
        n_samp = min(n_pair_samples, n * (n - 1) // 2)
        idx_i = rng.randint(0, n, n_samp)
        idx_j = rng.randint(0, n, n_samp)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        diff_true = y_true[idx_i] - y_true[idx_j]
        diff_pred = y_pred[idx_i] - y_pred[idx_j]
        discordant = np.sign(diff_true) != np.sign(diff_pred)
        for k in range(len(idx_i)):
            if discordant[k] and abs(diff_true[k]) > 0.1:
                sign = np.sign(diff_true[k])
                grad_rank[idx_i[k]] -= sign * 0.01
                grad_rank[idx_j[k]] += sign * 0.01

        return grad + alpha * grad_rank, hess
    return obj


def train_xgb_pairwise(X_train, y_train, X_test, alpha=2.0):
    """Train XGBoost with pairwise ranking objective."""
    import xgboost as xgb
    p = dict(BEST_XGB_PARAMS)
    p.pop('n_estimators', None)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    p.update({"nthread": N_JOBS, "seed": 42, "verbosity": 0})
    obj = xgb_pairwise_objective(alpha=alpha)
    model = xgb.train(p, dtrain, num_boost_round=749, obj=obj)
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    return pred_train, pred_test, model


def train_film_delta(X_train_a, X_train_b, delta_train,
                     X_test_a, X_test_b, delta_test=None,
                     epochs=200, lr=1e-3, batch_size=64, patience=25,
                     ranking_weight=0.0, ranking_margin=0.1):
    """Train FiLMDelta (optionally with ranking loss)."""
    input_dim = X_train_a.shape[1]
    scaler = StandardScaler()
    # Fit scaler on concatenated A and B
    all_train = np.vstack([X_train_a, X_train_b])
    scaler.fit(all_train)
    X_a_s = scaler.transform(X_train_a)
    X_b_s = scaler.transform(X_train_b)

    # Train/val split (80/20 of training pairs)
    n = len(X_a_s)
    perm = np.random.RandomState(42).permutation(n)
    n_val = max(20, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    if ranking_weight > 0:
        model = FiLMDeltaWithRanking(input_dim, ranking_weight=ranking_weight, margin=ranking_margin)
    else:
        model = FiLMDeltaNet(input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    A_tr = torch.FloatTensor(X_a_s[tr_idx])
    B_tr = torch.FloatTensor(X_b_s[tr_idx])
    D_tr = torch.FloatTensor(delta_train[tr_idx])
    A_val = torch.FloatTensor(X_a_s[val_idx])
    B_val = torch.FloatTensor(X_b_s[val_idx])
    D_val = torch.FloatTensor(delta_train[val_idx])

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        perm_tr = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), batch_size):
            bidx = perm_tr[start:start + batch_size]
            pred = model(A_tr[bidx], B_tr[bidx])
            mse_loss = F.mse_loss(pred, D_tr[bidx])
            loss = mse_loss
            if ranking_weight > 0:
                rank_loss = model.compute_ranking_loss(pred, D_tr[bidx])
                loss = mse_loss + ranking_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(A_val, B_val)
            val_loss = F.mse_loss(val_pred, D_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Predict on test
    X_te_a = torch.FloatTensor(scaler.transform(X_test_a))
    X_te_b = torch.FloatTensor(scaler.transform(X_test_b))
    with torch.no_grad():
        delta_pred = model(X_te_a, X_te_b).numpy()

    # Also predict absolute (zero-conditioned)
    all_test = np.unique(np.vstack([X_test_a, X_test_b]), axis=0)
    X_all_s = torch.FloatTensor(scaler.transform(all_test if len(all_test) > 0 else X_test_a))

    return delta_pred, model, scaler


def train_dual_objective(X_train_a, X_train_b, delta_train, abs_a_train, abs_b_train,
                         X_test_a, X_test_b,
                         epochs=200, lr=1e-3, batch_size=64, patience=25,
                         delta_weight=1.0, abs_weight=1.0):
    """Train dual-objective model."""
    input_dim = X_train_a.shape[1]
    scaler = StandardScaler()
    all_train = np.vstack([X_train_a, X_train_b])
    scaler.fit(all_train)
    X_a_s = scaler.transform(X_train_a)
    X_b_s = scaler.transform(X_train_b)

    n = len(X_a_s)
    perm = np.random.RandomState(42).permutation(n)
    n_val = max(20, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = DualObjectiveNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    A_tr = torch.FloatTensor(X_a_s[tr_idx])
    B_tr = torch.FloatTensor(X_b_s[tr_idx])
    D_tr = torch.FloatTensor(delta_train[tr_idx])
    AbsA_tr = torch.FloatTensor(abs_a_train[tr_idx])
    AbsB_tr = torch.FloatTensor(abs_b_train[tr_idx])
    A_val = torch.FloatTensor(X_a_s[val_idx])
    B_val = torch.FloatTensor(X_b_s[val_idx])
    D_val = torch.FloatTensor(delta_train[val_idx])

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm_tr = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), batch_size):
            bidx = perm_tr[start:start + batch_size]
            delta_pred, abs_a_pred, abs_b_pred = model(A_tr[bidx], B_tr[bidx])
            loss_delta = F.mse_loss(delta_pred, D_tr[bidx])
            loss_abs = (F.mse_loss(abs_a_pred, AbsA_tr[bidx]) + F.mse_loss(abs_b_pred, AbsB_tr[bidx])) / 2
            loss = delta_weight * loss_delta + abs_weight * loss_abs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_delta, _, _ = model(A_val, B_val)
            val_loss = F.mse_loss(val_delta, D_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    X_te_a = torch.FloatTensor(scaler.transform(X_test_a))
    X_te_b = torch.FloatTensor(scaler.transform(X_test_b))
    with torch.no_grad():
        delta_pred, abs_a_pred, abs_b_pred = model(X_te_a, X_te_b)

    return delta_pred.numpy(), abs_a_pred.numpy(), abs_b_pred.numpy(), model, scaler


# ═══════════════════════════════════════════════════════════════════════════
# ITERATION 1: Data Characterization + Baselines
# ═══════════════════════════════════════════════════════════════════════════

def run_iteration_1(mol_data, per_assay):
    """Characterize ZAP70 data: noise, MMP fraction, pair structure."""
    print("\n" + "=" * 70)
    print("ITERATION 1: Data Characterization + Baseline Establishment")
    print("=" * 70)
    results = {}

    # --- 1a: Inter-assay noise ---
    print("\n--- 1a: Inter-Assay Noise Analysis ---")
    mol_assay_counts = per_assay.groupby("molecule_chembl_id")["assay_id"].nunique()
    multi_assay_mols = mol_assay_counts[mol_assay_counts > 1]
    noise_stats = {"n_molecules": len(mol_data), "n_assays": per_assay["assay_id"].nunique(),
                   "n_multi_assay_mols": int(len(multi_assay_mols)),
                   "pct_multi_assay": float(len(multi_assay_mols) / len(mol_data) * 100)}

    if len(multi_assay_mols) > 0:
        variances = []
        ranges = []
        for mol_id in multi_assay_mols.index:
            vals = per_assay[per_assay["molecule_chembl_id"] == mol_id]["pIC50"].values
            if len(vals) > 1:
                variances.append(np.var(vals))
                ranges.append(np.ptp(vals))
        noise_stats["inter_assay_var_mean"] = float(np.mean(variances))
        noise_stats["inter_assay_var_median"] = float(np.median(variances))
        noise_stats["inter_assay_std_mean"] = float(np.sqrt(np.mean(variances)))
        noise_stats["inter_assay_range_mean"] = float(np.mean(ranges))
        noise_stats["inter_assay_range_max"] = float(np.max(ranges))
        print(f"  Multi-assay molecules: {len(multi_assay_mols)}/{len(mol_data)} ({noise_stats['pct_multi_assay']:.1f}%)")
        print(f"  Inter-assay std: {noise_stats['inter_assay_std_mean']:.3f} pIC50")
        print(f"  Inter-assay range: mean={noise_stats['inter_assay_range_mean']:.3f}, max={noise_stats['inter_assay_range_max']:.2f}")
    else:
        print("  No multiply-measured molecules — cannot estimate inter-assay noise.")

    results["noise_analysis"] = noise_stats

    # --- 1b: MMP fraction analysis ---
    print("\n--- 1b: MMP / Structural Similarity Analysis ---")
    smiles_list = mol_data["smiles"].values
    n_mols = len(smiles_list)

    # Compute Morgan FPs for Tanimoto
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        else:
            fps.append(None)

    # Sample pairs for Tanimoto distribution
    n_total_pairs = n_mols * (n_mols - 1) // 2
    rng = np.random.RandomState(42)
    n_sample = min(5000, n_total_pairs)

    tanimotos = []
    sampled = 0
    if n_sample < n_total_pairs:
        while sampled < n_sample:
            i, j = rng.randint(0, n_mols, 2)
            if i != j and fps[i] is not None and fps[j] is not None:
                tanimotos.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                sampled += 1
    else:
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                if fps[i] is not None and fps[j] is not None:
                    tanimotos.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    tanimotos = np.array(tanimotos)
    mmp_thresholds = {
        "tanimoto_0.5": float(np.mean(tanimotos >= 0.5)),
        "tanimoto_0.6": float(np.mean(tanimotos >= 0.6)),
        "tanimoto_0.7": float(np.mean(tanimotos >= 0.7)),
        "tanimoto_0.8": float(np.mean(tanimotos >= 0.8)),
        "tanimoto_0.9": float(np.mean(tanimotos >= 0.9)),
        "tanimoto_1.0": float(np.mean(tanimotos >= 1.0)),
    }
    print(f"  Total possible pairs: {n_total_pairs}")
    print(f"  Tanimoto distribution (sampled {len(tanimotos)}):")
    print(f"    Mean: {tanimotos.mean():.3f}, Median: {np.median(tanimotos):.3f}")
    for thresh, frac in mmp_thresholds.items():
        est_pairs = int(frac * n_total_pairs)
        print(f"    {thresh}: {frac*100:.1f}% ({est_pairs} est. pairs)")

    results["pair_structure"] = {
        "n_total_pairs": n_total_pairs,
        "tanimoto_mean": float(tanimotos.mean()),
        "tanimoto_median": float(np.median(tanimotos)),
        "mmp_fractions": mmp_thresholds,
    }

    # --- 1c: Absolute prediction baseline ---
    print("\n--- 1c: XGBoost Absolute Prediction Baseline (5-fold CV) ---")
    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    X = compute_fingerprints(smiles, fp_type="morgan", n_bits=2048)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_abs_results = []
    fold_delta_results = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_mols))):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pIC50[train_idx], pIC50[test_idx]

        pred_train, pred_test, _ = train_xgb_absolute(X_train, y_train, X_test)

        # Absolute metrics
        abs_metrics = evaluate_absolute(y_test, pred_test)
        fold_abs_results.append(abs_metrics)

        # Delta metrics: all test-vs-test pairs
        n_test = len(test_idx)
        delta_true = []
        delta_pred = []
        for i in range(n_test):
            for j in range(i + 1, n_test):
                delta_true.append(y_test[j] - y_test[i])
                delta_pred.append(pred_test[j] - pred_test[i])
        delta_true = np.array(delta_true)
        delta_pred = np.array(delta_pred)
        delta_metrics = evaluate_delta(delta_true, delta_pred)
        fold_delta_results.append(delta_metrics)

        print(f"  Fold {fold_i}: Abs MAE={abs_metrics['mae']:.3f}, Abs Spr={abs_metrics['spearman']:.3f} | "
              f"Δ MAE={delta_metrics['mae']:.3f}, Δ Spr={delta_metrics['spearman']:.3f}, CI={delta_metrics['concordance_index']:.3f}")

    # Aggregate
    def agg(results_list, key):
        vals = [r[key] for r in results_list if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": float('nan'), "std": 0}

    abs_agg = {k: agg(fold_abs_results, k) for k in fold_abs_results[0].keys() if k != 'n'}
    delta_agg = {k: agg(fold_delta_results, k) for k in fold_delta_results[0].keys() if k not in ['n_pairs', 'n_large_delta']}

    print(f"\n  XGB Subtraction Baseline:")
    print(f"    Absolute: MAE={abs_agg['mae']['mean']:.3f}±{abs_agg['mae']['std']:.3f}, "
          f"Spr={abs_agg['spearman']['mean']:.3f}±{abs_agg['spearman']['std']:.3f}")
    print(f"    Delta:    MAE={delta_agg['mae']['mean']:.3f}±{delta_agg['mae']['std']:.3f}, "
          f"Spr={delta_agg['spearman']['mean']:.3f}±{delta_agg['spearman']['std']:.3f}")
    print(f"    C-index:  {delta_agg['concordance_index']['mean']:.3f}, "
          f"Sign acc: {delta_agg['sign_accuracy']['mean']:.3f}")

    results["xgb_subtraction_baseline"] = {
        "absolute": abs_agg,
        "delta": delta_agg,
        "per_fold_absolute": fold_abs_results,
        "per_fold_delta": fold_delta_results,
    }

    save_results({"iteration_1": results})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ITERATION 2: Full Method Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_iteration_2(mol_data, per_assay, iter1_results=None):
    """Compare all methods: abs prediction, delta prediction, pair ranking."""
    print("\n" + "=" * 70)
    print("ITERATION 2: Full Method Comparison (5 methods × 5 folds)")
    print("=" * 70)

    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n_mols = len(smiles)
    X = compute_fingerprints(smiles, fp_type="morgan", n_bits=2048)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # Results storage: method → list of fold results
    method_results = defaultdict(lambda: {"absolute": [], "delta": []})

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_mols))):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} (train={len(train_idx)}, test={len(test_idx)}) ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pIC50[train_idx], pIC50[test_idx]

        # Build test pairs (all test-vs-test) using vectorized triu_indices
        n_test = len(test_idx)
        pair_i, pair_j = np.triu_indices(n_test, k=1)
        delta_true = y_test[pair_j] - y_test[pair_i]

        # Build training pairs (all train-vs-train) using vectorized triu_indices
        n_train = len(train_idx)
        tr_pair_i, tr_pair_j = np.triu_indices(n_train, k=1)
        delta_train = y_train[tr_pair_j] - y_train[tr_pair_i]
        # Subsample training pairs for neural models (cap at MAX_TRAIN_PAIRS)
        MAX_TRAIN_PAIRS = 25000  # Use all pairs (was 10K, now full)
        if len(tr_pair_i) > MAX_TRAIN_PAIRS:
            sub_idx = np.random.RandomState(42).choice(len(tr_pair_i), MAX_TRAIN_PAIRS, replace=False)
            tr_pair_i_sub = tr_pair_i[sub_idx]
            tr_pair_j_sub = tr_pair_j[sub_idx]
            delta_train_sub = delta_train[sub_idx]
        else:
            tr_pair_i_sub, tr_pair_j_sub = tr_pair_i, tr_pair_j
            delta_train_sub = delta_train
        print(f"  Training pairs: {len(tr_pair_i)} total, {len(tr_pair_i_sub)} used for neural models")

        # ── Method 1: XGBoost Subtraction ──
        print(f"  [1/5] XGB Subtraction...", end=" ", flush=True)
        _, pred_test_abs, _ = train_xgb_absolute(X_train, y_train, X_test)
        delta_pred_sub = pred_test_abs[pair_j] - pred_test_abs[pair_i]
        m1_abs = evaluate_absolute(y_test, pred_test_abs)
        m1_delta = evaluate_delta(delta_true, delta_pred_sub)
        method_results["1_XGB_Subtraction"]["absolute"].append(m1_abs)
        method_results["1_XGB_Subtraction"]["delta"].append(m1_delta)
        print(f"Abs MAE={m1_abs['mae']:.3f}, Δ Spr={m1_delta['spearman']:.3f}")

        # ── Method 2: XGBoost + Pairwise Ranking ──
        print(f"  [2/5] XGB Pairwise Ranking (α=2.0)...", end=" ", flush=True)
        _, pred_test_rank, _ = train_xgb_pairwise(X_train, y_train, X_test, alpha=2.0)
        delta_pred_rank = pred_test_rank[pair_j] - pred_test_rank[pair_i]
        m2_abs = evaluate_absolute(y_test, pred_test_rank)
        m2_delta = evaluate_delta(delta_true, delta_pred_rank)
        method_results["2_XGB_PairwiseRanking"]["absolute"].append(m2_abs)
        method_results["2_XGB_PairwiseRanking"]["delta"].append(m2_delta)
        print(f"Abs MAE={m2_abs['mae']:.3f}, Δ Spr={m2_delta['spearman']:.3f}")

        # ── Method 3: FiLMDelta (MSE only) ──
        print(f"  [3/5] FiLMDelta...", end=" ", flush=True)
        X_tr_a = X_train[tr_pair_i_sub]
        X_tr_b = X_train[tr_pair_j_sub]
        X_te_a = X_test[pair_i]
        X_te_b = X_test[pair_j]
        delta_pred_film, film_model, film_scaler = train_film_delta(
            X_tr_a, X_tr_b, delta_train_sub, X_te_a, X_te_b, ranking_weight=0.0)
        m3_delta = evaluate_delta(delta_true, delta_pred_film)
        # For absolute: use zero-conditioned prediction
        film_model.eval()
        with torch.no_grad():
            X_test_s = torch.FloatTensor(film_scaler.transform(X_test))
            pred_abs_film = film_model.predict_absolute(X_test_s).numpy()
        m3_abs = evaluate_absolute(y_test, pred_abs_film)
        method_results["3_FiLMDelta"]["absolute"].append(m3_abs)
        method_results["3_FiLMDelta"]["delta"].append(m3_delta)
        print(f"Abs MAE={m3_abs['mae']:.3f}, Δ Spr={m3_delta['spearman']:.3f}")
        del film_model; gc.collect()

        # ── Method 4: FiLMDelta + Ranking Loss ──
        print(f"  [4/5] FiLMDelta + Ranking (λ=0.5)...", end=" ", flush=True)
        delta_pred_filmr, filmr_model, filmr_scaler = train_film_delta(
            X_tr_a, X_tr_b, delta_train_sub, X_te_a, X_te_b,
            ranking_weight=0.5, ranking_margin=0.1)
        m4_delta = evaluate_delta(delta_true, delta_pred_filmr)
        filmr_model.eval()
        with torch.no_grad():
            X_test_s = torch.FloatTensor(filmr_scaler.transform(X_test))
            pred_abs_filmr = filmr_model.predict_absolute(X_test_s).numpy()
        m4_abs = evaluate_absolute(y_test, pred_abs_filmr)
        method_results["4_FiLMDelta_Ranking"]["absolute"].append(m4_abs)
        method_results["4_FiLMDelta_Ranking"]["delta"].append(m4_delta)
        print(f"Abs MAE={m4_abs['mae']:.3f}, Δ Spr={m4_delta['spearman']:.3f}")
        del filmr_model; gc.collect()

        # ── Method 5: Dual-Objective ──
        print(f"  [5/5] Dual-Objective...", end=" ", flush=True)
        abs_a_tr = y_train[tr_pair_i_sub]
        abs_b_tr = y_train[tr_pair_j_sub]
        delta_pred_dual, abs_a_pred_dual, abs_b_pred_dual, dual_model, dual_scaler = \
            train_dual_objective(X_tr_a, X_tr_b, delta_train_sub, abs_a_tr, abs_b_tr,
                                X_te_a, X_te_b)
        m5_delta = evaluate_delta(delta_true, delta_pred_dual)
        dual_model.eval()
        with torch.no_grad():
            X_test_s = torch.FloatTensor(dual_scaler.transform(X_test))
            pred_abs_dual = dual_model.predict_absolute(X_test_s).numpy()
        m5_abs = evaluate_absolute(y_test, pred_abs_dual)
        method_results["5_DualObjective"]["absolute"].append(m5_abs)
        method_results["5_DualObjective"]["delta"].append(m5_delta)
        print(f"Abs MAE={m5_abs['mae']:.3f}, Δ Spr={m5_delta['spearman']:.3f}")
        del dual_model; gc.collect()

        # ── Method 6: FiLMDelta on 10K subsample (ablation) ──
        if len(tr_pair_i) > 10000:
            print(f"  [6/6] FiLMDelta (10K subsample)...", end=" ", flush=True)
            sub10k = np.random.RandomState(99).choice(len(tr_pair_i), 10000, replace=False)
            delta_pred_10k, _, _ = train_film_delta(
                X_train[tr_pair_i[sub10k]], X_train[tr_pair_j[sub10k]],
                delta_train[sub10k], X_te_a, X_te_b, ranking_weight=0.0)
            m6_delta = evaluate_delta(delta_true, delta_pred_10k)
            m6_abs = {"mae": float('nan'), "spearman": float('nan'), "pearson": float('nan'),
                      "concordance_index": float('nan'), "top10_precision": float('nan'),
                      "top20_precision": float('nan'), "n": n_test}
            method_results["6_FiLMDelta_10K"]["absolute"].append(m6_abs)
            method_results["6_FiLMDelta_10K"]["delta"].append(m6_delta)
            print(f"Δ MAE={m6_delta['mae']:.3f}, Δ Spr={m6_delta['spearman']:.3f}")

    # ── Aggregate and print comparison ──
    print("\n" + "=" * 70)
    print("ITERATION 2 RESULTS: Method Comparison")
    print("=" * 70)

    summary = {}
    print(f"\n{'Method':<30} {'Abs MAE':>8} {'Abs Spr':>8} {'Δ MAE':>8} {'Δ Spr':>8} {'Δ CI':>8} {'Sign%':>8} {'LgSign%':>8}")
    print("-" * 100)

    for method_name in sorted(method_results.keys()):
        mr = method_results[method_name]
        abs_results = mr["absolute"]
        delta_results = mr["delta"]

        def agg(lst, key):
            vals = [r[key] for r in lst if not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.mean(vals)) if vals else float('nan'), float(np.std(vals)) if vals else 0

        abs_mae_m, abs_mae_s = agg(abs_results, "mae")
        abs_spr_m, abs_spr_s = agg(abs_results, "spearman")
        d_mae_m, d_mae_s = agg(delta_results, "mae")
        d_spr_m, d_spr_s = agg(delta_results, "spearman")
        d_ci_m, d_ci_s = agg(delta_results, "concordance_index")
        d_sign_m, _ = agg(delta_results, "sign_accuracy")
        d_lsign_m, _ = agg(delta_results, "large_delta_sign_accuracy")

        print(f"{method_name:<30} {abs_mae_m:>7.3f}  {abs_spr_m:>7.3f}  {d_mae_m:>7.3f}  {d_spr_m:>7.3f}  "
              f"{d_ci_m:>7.3f}  {d_sign_m:>6.1%}  {d_lsign_m:>6.1%}")

        summary[method_name] = {
            "absolute": {k: {"mean": agg(abs_results, k)[0], "std": agg(abs_results, k)[1]}
                         for k in abs_results[0].keys() if k != 'n'},
            "delta": {k: {"mean": agg(delta_results, k)[0], "std": agg(delta_results, k)[1]}
                      for k in delta_results[0].keys() if k not in ['n_pairs', 'n_large_delta']},
            "per_fold_absolute": abs_results,
            "per_fold_delta": delta_results,
        }

    save_results({"iteration_2": summary})
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# ITERATION 3: Virtual Screening Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_iteration_3(mol_data, per_assay, iter2_results=None):
    """Virtual screening: which molecules does each method prioritize?"""
    print("\n" + "=" * 70)
    print("ITERATION 3: Virtual Screening Comparison")
    print("=" * 70)

    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    mol_ids = mol_data["molecule_chembl_id"].values
    n_mols = len(smiles)
    X = compute_fingerprints(smiles, fp_type="morgan", n_bits=2048)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    results = {}

    # Use a single representative fold for detailed analysis
    # (fold 0 for reproducibility)
    train_idx, test_idx = list(kf.split(np.arange(n_mols)))[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = pIC50[train_idx], pIC50[test_idx]
    test_smiles = smiles[test_idx]
    test_ids = mol_ids[test_idx]

    # Reference molecule: weakest in training set (simulates lead optimization)
    ref_idx_in_train = np.argmin(y_train)
    X_ref = X_train[ref_idx_in_train:ref_idx_in_train+1]
    y_ref = y_train[ref_idx_in_train]
    ref_smi = smiles[train_idx[ref_idx_in_train]]
    print(f"\n  Reference molecule: {smiles[train_idx[ref_idx_in_train]][:60]}...")
    print(f"  Reference pIC50: {y_ref:.2f} (weakest in training set)")

    # True ranking: test molecules ranked by pIC50
    true_rank = np.argsort(-y_test)  # descending
    true_top10 = set(true_rank[:10])
    true_top20 = set(true_rank[:20])

    print(f"  Test set: {len(test_idx)} molecules, pIC50 range {y_test.min():.2f}-{y_test.max():.2f}")

    # --- Train all methods and predict absolute pIC50 for test molecules ---
    method_predictions = {}

    # Method 1: XGB Subtraction
    print("\n  Training Method 1: XGB Subtraction...")
    _, pred_sub, model_sub = train_xgb_absolute(X_train, y_train, X_test)
    method_predictions["1_XGB_Subtraction"] = pred_sub

    # Method 2: XGB Pairwise Ranking
    print("  Training Method 2: XGB Pairwise Ranking...")
    _, pred_rank, _ = train_xgb_pairwise(X_train, y_train, X_test, alpha=2.0)
    method_predictions["2_XGB_PairwiseRanking"] = pred_rank

    # Method 3: FiLMDelta (need pair-wise training)
    print("  Training Method 3: FiLMDelta...")
    n_train = len(train_idx)
    tr_pi, tr_pj = np.triu_indices(n_train, k=1)
    delta_train = y_train[tr_pj] - y_train[tr_pi]
    # Subsample for speed
    if len(tr_pi) > 10000:
        sub_idx = np.random.RandomState(42).choice(len(tr_pi), 10000, replace=False)
        tr_pi, tr_pj, delta_train = tr_pi[sub_idx], tr_pj[sub_idx], delta_train[sub_idx]
    X_tr_a, X_tr_b = X_train[tr_pi], X_train[tr_pj]

    # For FiLM: predict delta relative to reference, then add reference pIC50
    X_ref_rep = np.tile(X_ref, (len(test_idx), 1))
    _, film_model, film_scaler = train_film_delta(
        X_tr_a, X_tr_b, delta_train, X_ref_rep, X_test)
    film_model.eval()
    with torch.no_grad():
        X_test_s = torch.FloatTensor(film_scaler.transform(X_test))
        pred_film_abs = film_model.predict_absolute(X_test_s).numpy()
    method_predictions["3_FiLMDelta"] = pred_film_abs
    del film_model; gc.collect()

    # Method 4: FiLMDelta + Ranking
    print("  Training Method 4: FiLMDelta + Ranking...")
    _, filmr_model, filmr_scaler = train_film_delta(
        X_tr_a, X_tr_b, delta_train, X_ref_rep, X_test,
        ranking_weight=0.5)
    filmr_model.eval()
    with torch.no_grad():
        X_test_s = torch.FloatTensor(filmr_scaler.transform(X_test))
        pred_filmr_abs = filmr_model.predict_absolute(X_test_s).numpy()
    method_predictions["4_FiLMDelta_Ranking"] = pred_filmr_abs
    del filmr_model; gc.collect()

    # Method 5: Dual-Objective
    print("  Training Method 5: Dual-Objective...")
    abs_a_tr = y_train[tr_pi]
    abs_b_tr = y_train[tr_pj]
    _, _, _, dual_model, dual_scaler = train_dual_objective(
        X_tr_a, X_tr_b, delta_train, abs_a_tr, abs_b_tr, X_ref_rep, X_test)
    dual_model.eval()
    with torch.no_grad():
        X_test_s = torch.FloatTensor(dual_scaler.transform(X_test))
        pred_dual_abs = dual_model.predict_absolute(X_test_s).numpy()
    method_predictions["5_DualObjective"] = pred_dual_abs
    del dual_model; gc.collect()

    # --- Compare virtual screening performance ---
    print("\n" + "-" * 70)
    print("VIRTUAL SCREENING RESULTS")
    print("-" * 70)
    print(f"\n{'Method':<30} {'Abs MAE':>8} {'Abs Spr':>8} {'CI':>8} {'R@10':>8} {'R@20':>8}")
    print("-" * 78)

    screening_results = {}
    for method_name, pred in sorted(method_predictions.items()):
        abs_metrics = evaluate_absolute(y_test, pred)
        pred_rank = np.argsort(-pred)
        pred_top10 = set(pred_rank[:10])
        pred_top20 = set(pred_rank[:20])
        recall_10 = len(true_top10 & pred_top10) / 10
        recall_20 = len(true_top20 & pred_top20) / 20
        ci = concordance_index(y_test, pred)

        print(f"{method_name:<30} {abs_metrics['mae']:>7.3f}  {abs_metrics['spearman']:>7.3f}  "
              f"{ci:>7.3f}  {recall_10:>6.0%}  {recall_20:>6.0%}")

        screening_results[method_name] = {
            "absolute_metrics": abs_metrics,
            "concordance_index": float(ci),
            "recall_at_10": float(recall_10),
            "recall_at_20": float(recall_20),
        }

    # --- Detailed: top-10 molecules per method ---
    print("\n" + "-" * 70)
    print("TOP 10 MOLECULES BY METHOD (compared to ground truth)")
    print("-" * 70)

    print(f"\n  TRUE Top 10 (by pIC50):")
    for rank, idx in enumerate(true_rank[:10]):
        print(f"    #{rank+1}: {test_ids[idx]} pIC50={y_test[idx]:.2f} {test_smiles[idx][:50]}...")

    for method_name, pred in sorted(method_predictions.items()):
        pred_rank = np.argsort(-pred)
        print(f"\n  {method_name} Top 10:")
        for rank, idx in enumerate(pred_rank[:10]):
            in_true = "✓" if idx in true_top10 else " "
            print(f"    #{rank+1} [{in_true}] {test_ids[idx]} pred={pred[idx]:.2f} true={y_test[idx]:.2f} Δ={pred[idx]-y_test[idx]:+.2f}")

    # --- Agreement analysis: how much do methods agree? ---
    print("\n" + "-" * 70)
    print("METHOD AGREEMENT (Spearman between predicted rankings)")
    print("-" * 70)
    method_names = sorted(method_predictions.keys())
    agreement = {}
    print(f"\n{'':>30}", end="")
    for mn in method_names:
        print(f" {mn[2:12]:>12}", end="")
    print()
    for mn1 in method_names:
        print(f"{mn1:<30}", end="")
        for mn2 in method_names:
            spr, _ = spearmanr(method_predictions[mn1], method_predictions[mn2])
            print(f" {spr:>11.3f}", end="")
            agreement[f"{mn1}_vs_{mn2}"] = float(spr)
        print()

    screening_results["method_agreement"] = agreement
    screening_results["reference_molecule"] = {
        "smiles": ref_smi, "pIC50": float(y_ref),
    }
    screening_results["test_set_stats"] = {
        "n_test": int(len(test_idx)),
        "pIC50_min": float(y_test.min()),
        "pIC50_max": float(y_test.max()),
        "pIC50_mean": float(y_test.mean()),
    }

    # --- Molecules uniquely identified by each method ---
    print("\n" + "-" * 70)
    print("UNIQUE PICKS: Molecules in one method's top-10 but NOT in XGB Subtraction's top-10")
    print("-" * 70)
    sub_top10 = set(np.argsort(-method_predictions["1_XGB_Subtraction"])[:10])
    for method_name, pred in sorted(method_predictions.items()):
        if method_name == "1_XGB_Subtraction":
            continue
        pred_rank = np.argsort(-pred)
        pred_top10 = set(pred_rank[:10])
        unique = pred_top10 - sub_top10
        if unique:
            print(f"\n  {method_name} unique picks:")
            for idx in unique:
                in_true = "✓" if idx in true_top10 else " "
                print(f"    [{in_true}] {test_ids[idx]} pred={pred[idx]:.2f} true={y_test[idx]:.2f}")

    save_results({"iteration_3": screening_results})
    return screening_results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ZAP70 Comprehensive Method Comparison")
    parser.add_argument("--iteration", type=int, nargs="+", default=[1, 2, 3],
                        help="Which iterations to run (1, 2, 3)")
    args = parser.parse_args()

    print("=" * 70)
    print("ZAP70 (CHEMBL2803) COMPREHENSIVE METHOD COMPARISON")
    print("=" * 70)

    mol_data, per_assay = load_zap70_molecules()
    print(f"Loaded: {len(mol_data)} molecules, {per_assay['assay_id'].nunique()} assays")

    t0 = time.time()
    iter1_results = None
    iter2_results = None

    if 1 in args.iteration:
        iter1_results = run_iteration_1(mol_data, per_assay)

    if 2 in args.iteration:
        iter2_results = run_iteration_2(mol_data, per_assay, iter1_results)

    if 3 in args.iteration:
        run_iteration_3(mol_data, per_assay, iter2_results)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"TOTAL TIME: {elapsed/60:.1f} minutes")
    save_results({"total_time_seconds": elapsed, "completed": datetime.now().isoformat()})


if __name__ == "__main__":
    main()
