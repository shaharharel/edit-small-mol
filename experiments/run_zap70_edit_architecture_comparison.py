#!/usr/bin/env python3
"""
ZAP70 Edit Architecture Comparison with Kinase-Specific Pretraining.

Compares multiple edit-aware neural architectures against XGB subtraction on
ZAP70 (CHEMBL2803, 280 molecules), all using kinase-specific pretraining from
8 related kinase targets (~32K within-assay MMP pairs).

Models:
1. XGB Subtraction (baseline) — no pretraining
2. FiLMDelta + Kinase Pretraining
3. DualObjective + Kinase Pretraining
4. FiLMDelta + Edit Features (11d pair features, no pretraining)
5. FiLMDelta + Edit Features + Kinase Pretraining

Usage:
    conda run -n quris python -u experiments/run_zap70_edit_architecture_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import gc
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import RESULTS_DIR, DATA_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    compute_absolute_metrics, aggregate_cv_results,
)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "zap70_edit_architecture_comparison.json"

ZAP70_ID = "CHEMBL2803"
N_FOLDS = 5
CV_SEED = 42

KINASE_TARGETS = {
    "SYK": "CHEMBL2599", "FYN": "CHEMBL1841", "LCK": "CHEMBL258",
    "BTK": "CHEMBL5251", "ITK": "CHEMBL3009", "JAK2": "CHEMBL2971",
    "ABL1": "CHEMBL1862", "SRC": "CHEMBL267",
}

# ═══════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════

class DualObjectiveModel(nn.Module):
    """Model with dual heads: delta prediction + absolute prediction."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h), nn.ReLU(),
                nn.BatchNorm1d(h), nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_dim = hidden_dims[-1]
        self.delta_head = nn.Sequential(
            nn.Linear(self.enc_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.abs_head = nn.Sequential(
            nn.Linear(self.enc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_a, x_b):
        enc_a = self.encoder(x_a)
        enc_b = self.encoder(x_b)
        delta_pred = self.delta_head(torch.cat([enc_a, enc_b], dim=-1)).squeeze(-1)
        abs_pred_a = self.abs_head(enc_a).squeeze(-1)
        abs_pred_b = self.abs_head(enc_b).squeeze(-1)
        return delta_pred, abs_pred_a, abs_pred_b

    def predict_absolute(self, x):
        enc = self.encoder(x)
        return self.abs_head(enc).squeeze(-1)


class FiLMDeltaWithFeats(nn.Module):
    """FiLMDelta where conditioning = [emb_b - emb_a, edit_features]."""
    def __init__(self, emb_dim, feat_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Import FiLM components
        from src.models.predictors.film_delta_predictor import FiLMBlock

        # Project concatenated [delta_emb, feats] to conditioning dim
        cond_dim = emb_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(emb_dim + feat_dim, cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Delta encoder (matches FiLMDeltaMLP)
        delta_hidden = max(emb_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(cond_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM blocks
        self.blocks = nn.ModuleList()
        prev = emb_dim
        for h in hidden_dims:
            self.blocks.append(FiLMBlock(prev, h, delta_hidden, dropout=dropout))
            prev = h
        self.output = nn.Linear(prev, 1)

    def forward(self, emb_a, emb_b, edit_feats=None):
        delta = emb_b - emb_a
        if edit_feats is not None:
            cond_input = torch.cat([delta, edit_feats], dim=-1)
        else:
            cond_input = torch.cat([
                delta,
                torch.zeros(delta.shape[0], self.feat_dim, device=delta.device)
            ], dim=-1)

        delta_cond = self.cond_proj(cond_input)
        delta_enc = self.delta_encoder(delta_cond)

        # Process both molecules with FiLM conditioning: f(B|delta) - f(A|delta)
        pred_b = emb_b
        for block in self.blocks:
            pred_b = block(pred_b, delta_enc)
        pred_b = self.output(pred_b).squeeze(-1)

        pred_a = emb_a
        for block in self.blocks:
            pred_a = block(pred_a, delta_enc)
        pred_a = self.output(pred_a).squeeze(-1)

        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Pair Feature Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_pair_features(smi_a, smi_b):
    """Compute 11-dim pair features without edit SMILES."""
    feats = np.zeros(11, dtype=np.float32)
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return feats

    # Descriptor deltas (6)
    for i, func in enumerate([
        Descriptors.MolLogP, Descriptors.MolWt, Descriptors.TPSA,
        Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.NumRotatableBonds
    ]):
        feats[i] = func(mol_b) - func(mol_a)

    # Tanimoto similarity (1)
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
    feats[6] = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    # Number of differing bits (1)
    arr_a = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(fp_a, arr_a)
    arr_b = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(fp_b, arr_b)
    feats[7] = float(np.sum(arr_a != arr_b))

    # Sign indicators (3)
    feats[8] = np.sign(feats[2])   # sign(delta_TPSA)
    feats[9] = np.sign(feats[0])   # sign(delta_logP)
    feats[10] = np.sign(feats[1])  # sign(delta_MW)

    return feats


def compute_all_pair_features(pairs_df):
    """Compute pair features for a DataFrame of pairs. Returns (N, 11) array."""
    n = len(pairs_df)
    feats = np.zeros((n, 11), dtype=np.float32)
    for i, (_, row) in enumerate(pairs_df.iterrows()):
        feats[i] = compute_pair_features(row["mol_a"], row["mol_b"])
        if (i + 1) % 10000 == 0:
            print(f"    Pair features: {i+1}/{n}")
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Data Helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_pairs(mol_data):
    """Generate all directed pairwise combinations from molecule data."""
    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n = len(smiles)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append({
                "mol_a": smiles[i], "mol_b": smiles[j],
                "delta": float(pIC50[j] - pIC50[i]),
                "value_a": float(pIC50[i]), "value_b": float(pIC50[j]),
            })
    return pd.DataFrame(rows)


def compute_delta_metrics(y_true, y_pred):
    """Compute metrics for delta prediction."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    sr, _ = spearmanr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    return {
        "n": len(y_true),
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
        "spearman_r": float(sr) if not np.isnan(sr) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training Routines
# ═══════════════════════════════════════════════════════════════════════════

def pretrain_dual_objective(kinase_pairs, fp_cache, hidden_dims=[512, 256],
                            epochs=100, batch_size=256, lr=1e-3, patience=15,
                            lambda_abs=0.3):
    """Pretrain DualObjectiveModel on kinase within-assay pairs."""
    print("  Pretraining DualObjective on kinase pairs...")
    emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta = kinase_pairs["delta"].values.astype(np.float32)
    val_a = kinase_pairs["value_a"].values.astype(np.float32)
    val_b = kinase_pairs["value_b"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a, emb_b]))

    Xa = torch.FloatTensor(scaler.transform(emb_a))
    Xb = torch.FloatTensor(scaler.transform(emb_b))
    yd = torch.FloatTensor(delta)
    ya = torch.FloatTensor(val_a)
    yb = torch.FloatTensor(val_b)

    n_val = max(len(Xa) // 10, 1)
    model = DualObjectiveModel(Xa.shape[1], hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            bi = perm[start:start + batch_size]
            optimizer.zero_grad()
            d_pred, a_pred, b_pred = model(Xa[bi], Xb[bi])
            loss = criterion(d_pred, yd[bi]) + lambda_abs * (
                criterion(a_pred, ya[bi]) + criterion(b_pred, yb[bi]))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vd, _, _ = model(Xa[:n_val], Xb[:n_val])
            vl = criterion(vd, yd[:n_val]).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f}, val_loss={vl:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred, _, _ = model(Xa[:n_val], Xb[:n_val])
    val_mae = float(np.mean(np.abs(delta[:n_val] - val_pred.numpy())))
    val_spr, _ = spearmanr(delta[:n_val], val_pred.numpy()) if n_val > 2 else (0, 1)
    print(f"    Pretrain done: val_MAE={val_mae:.4f}, val_Spr={float(val_spr):.3f}")

    return model, scaler, {"val_mae": val_mae, "val_spearman": float(val_spr)}


def pretrain_film_delta(kinase_pairs, fp_cache, hidden_dims=[1024, 512, 256],
                        epochs=100, batch_size=256, lr=1e-3, patience=15):
    """Pretrain FiLMDeltaMLP on kinase within-assay pairs."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    print("  Pretraining FiLMDelta on kinase pairs...")
    emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a, emb_b]))

    Xa = torch.FloatTensor(scaler.transform(emb_a))
    Xb = torch.FloatTensor(scaler.transform(emb_b))
    yd = torch.FloatTensor(delta)

    n_val = max(len(Xa) // 10, 1)
    model = FiLMDeltaMLP(input_dim=Xa.shape[1], hidden_dims=hidden_dims, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            bi = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(Xa[bi], Xb[bi])
            loss = criterion(pred, yd[bi])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vp = model(Xa[:n_val], Xb[:n_val])
            vl = criterion(vp, yd[:n_val]).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f}, val_loss={vl:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(Xa[:n_val], Xb[:n_val]).numpy()
    val_mae = float(np.mean(np.abs(delta[:n_val] - val_pred)))
    val_spr, _ = spearmanr(delta[:n_val], val_pred) if n_val > 2 else (0, 1)
    print(f"    Pretrain done: val_MAE={val_mae:.4f}, val_Spr={float(val_spr):.3f}")

    return model, scaler, {"val_mae": val_mae, "val_spearman": float(val_spr)}


def pretrain_film_with_feats(kinase_pairs, fp_cache, kinase_feats,
                             hidden_dims=[1024, 512, 256],
                             epochs=100, batch_size=256, lr=1e-3, patience=15):
    """Pretrain FiLMDeltaWithFeats on kinase within-assay pairs."""
    print("  Pretraining FiLMDelta+EditFeats on kinase pairs...")
    emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a, emb_b]))

    feat_scaler = StandardScaler()
    feat_scaler.fit(kinase_feats)

    Xa = torch.FloatTensor(scaler.transform(emb_a))
    Xb = torch.FloatTensor(scaler.transform(emb_b))
    Xf = torch.FloatTensor(feat_scaler.transform(kinase_feats))
    yd = torch.FloatTensor(delta)

    n_val = max(len(Xa) // 10, 1)
    feat_dim = kinase_feats.shape[1]
    model = FiLMDeltaWithFeats(emb_dim=Xa.shape[1], feat_dim=feat_dim,
                                hidden_dims=hidden_dims, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            bi = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(Xa[bi], Xb[bi], Xf[bi])
            loss = criterion(pred, yd[bi])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vp = model(Xa[:n_val], Xb[:n_val], Xf[:n_val])
            vl = criterion(vp, yd[:n_val]).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f}, val_loss={vl:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(Xa[:n_val], Xb[:n_val], Xf[:n_val]).numpy()
    val_mae = float(np.mean(np.abs(delta[:n_val] - val_pred)))
    val_spr, _ = spearmanr(delta[:n_val], val_pred) if n_val > 2 else (0, 1)
    print(f"    Pretrain done: val_MAE={val_mae:.4f}, val_Spr={float(val_spr):.3f}")

    return model, scaler, feat_scaler, {"val_mae": val_mae, "val_spearman": float(val_spr)}


# ═══════════════════════════════════════════════════════════════════════════
# CV Evaluation Routines
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_xgb_subtraction(mol_data, fp_cache):
    """Model 1: XGB Subtraction baseline (no pretraining)."""
    from xgboost import XGBRegressor

    print("\n--- Model 1: XGB Subtraction ---")
    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    delta_folds = []
    abs_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        train_smi = smiles_list[train_idx]
        test_smi = smiles_list[test_idx]
        train_y = pIC50[train_idx]
        test_y = pIC50[test_idx]

        X_train = np.array([fp_cache[s] for s in train_smi])
        X_test = np.array([fp_cache[s] for s in test_smi])

        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=CV_SEED, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X_train, train_y)
        abs_pred = xgb.predict(X_test)
        abs_folds.append(compute_absolute_metrics(test_y, abs_pred))

        # Delta: all test-test pairs
        n_te = len(test_idx)
        if n_te < 2:
            continue
        # Compute all directed test pairs
        delta_true_list = []
        delta_pred_list = []
        pred_vals = xgb.predict(X_test)
        for i in range(n_te):
            for j in range(n_te):
                if i == j:
                    continue
                delta_true_list.append(test_y[j] - test_y[i])
                delta_pred_list.append(pred_vals[j] - pred_vals[i])
        delta_true_arr = np.array(delta_true_list)
        delta_pred_arr = np.array(delta_pred_list)
        delta_folds.append(compute_delta_metrics(delta_true_arr, delta_pred_arr))

        print(f"  Fold {fold_i+1}: abs MAE={abs_folds[-1]['mae']:.4f}, "
              f"delta MAE={delta_folds[-1]['mae']:.4f}, delta Spr={delta_folds[-1]['spearman_r']:.3f}")

    return {
        "delta_folds": delta_folds,
        "absolute_folds": abs_folds,
        "delta_summary": aggregate_cv_results(delta_folds),
        "absolute_summary": aggregate_cv_results(abs_folds),
    }


def _finetune_film_cv(pretrained_model, pretrained_scaler, zap70_pairs, fp_cache,
                       mol_data, epochs=50, batch_size=256, lr=1e-4, patience=15):
    """Fine-tune a FiLMDeltaMLP on ZAP70 with molecule-level CV."""
    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    delta_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        train_smi = set(smiles_list[train_idx])
        test_smi = set(smiles_list[test_idx])

        train_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(train_smi) & zap70_pairs["mol_b"].isin(train_smi)
        ]
        test_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(test_smi) & zap70_pairs["mol_b"].isin(test_smi)
        ]
        if len(test_p) == 0:
            continue

        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        delta_tr = train_p["delta"].values.astype(np.float32)

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        delta_te = test_p["delta"].values.astype(np.float32)

        # Fine-tune copy of pretrained model
        ft_model = copy.deepcopy(pretrained_model)
        ft_scaler = copy.deepcopy(pretrained_scaler)

        Xa_tr = torch.FloatTensor(ft_scaler.transform(emb_a_tr))
        Xb_tr = torch.FloatTensor(ft_scaler.transform(emb_b_tr))
        yd = torch.FloatTensor(delta_tr)

        Xa_te = torch.FloatTensor(ft_scaler.transform(emb_a_te))
        Xb_te = torch.FloatTensor(ft_scaler.transform(emb_b_te))

        optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr))
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                optimizer.zero_grad()
                pred = ft_model(Xa_tr[bi], Xb_tr[bi])
                loss = criterion(pred, yd[bi])
                loss.backward()
                optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                val_pred = ft_model(Xa_te, Xb_te)
                vl = criterion(val_pred, torch.FloatTensor(delta_te)).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            ft_model.load_state_dict(best_state)

        ft_model.eval()
        with torch.no_grad():
            pred = ft_model(Xa_te, Xb_te).numpy()

        delta_folds.append(compute_delta_metrics(delta_te, pred))
        print(f"  Fold {fold_i+1}: delta MAE={delta_folds[-1]['mae']:.4f}, "
              f"delta Spr={delta_folds[-1]['spearman_r']:.3f}")

        del ft_model
        gc.collect()

    return {
        "delta_folds": delta_folds,
        "delta_summary": aggregate_cv_results(delta_folds),
    }


def _finetune_dual_cv(pretrained_model, pretrained_scaler, zap70_pairs, fp_cache,
                       mol_data, epochs=50, batch_size=256, lr=1e-4, patience=15,
                       lambda_abs=0.3):
    """Fine-tune DualObjectiveModel on ZAP70 with molecule-level CV."""
    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    delta_folds = []
    abs_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        train_smi = set(smiles_list[train_idx])
        test_smi = set(smiles_list[test_idx])

        train_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(train_smi) & zap70_pairs["mol_b"].isin(train_smi)
        ]
        test_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(test_smi) & zap70_pairs["mol_b"].isin(test_smi)
        ]
        if len(test_p) == 0:
            continue

        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        delta_tr = train_p["delta"].values.astype(np.float32)
        val_a_tr = train_p["value_a"].values.astype(np.float32)
        val_b_tr = train_p["value_b"].values.astype(np.float32)

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        delta_te = test_p["delta"].values.astype(np.float32)

        # Fine-tune copy
        ft_model = copy.deepcopy(pretrained_model)
        ft_scaler = copy.deepcopy(pretrained_scaler)

        Xa_tr = torch.FloatTensor(ft_scaler.transform(emb_a_tr))
        Xb_tr = torch.FloatTensor(ft_scaler.transform(emb_b_tr))
        yd = torch.FloatTensor(delta_tr)
        ya = torch.FloatTensor(val_a_tr)
        yb = torch.FloatTensor(val_b_tr)

        Xa_te = torch.FloatTensor(ft_scaler.transform(emb_a_te))
        Xb_te = torch.FloatTensor(ft_scaler.transform(emb_b_te))

        optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr))
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                optimizer.zero_grad()
                d_p, a_p, b_p = ft_model(Xa_tr[bi], Xb_tr[bi])
                loss = criterion(d_p, yd[bi]) + lambda_abs * (
                    criterion(a_p, ya[bi]) + criterion(b_p, yb[bi]))
                loss.backward()
                optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                d_te, _, _ = ft_model(Xa_te, Xb_te)
                vl = criterion(d_te, torch.FloatTensor(delta_te)).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            ft_model.load_state_dict(best_state)

        ft_model.eval()
        with torch.no_grad():
            d_pred, _, _ = ft_model(Xa_te, Xb_te)
            d_pred = d_pred.numpy()

        delta_folds.append(compute_delta_metrics(delta_te, d_pred))

        # Absolute prediction from dual model
        test_embs = np.array([fp_cache[s] for s in smiles_list[test_idx]])
        test_y = pIC50[test_idx]
        with torch.no_grad():
            abs_preds = ft_model.predict_absolute(
                torch.FloatTensor(ft_scaler.transform(test_embs))
            ).numpy()
        abs_folds.append(compute_absolute_metrics(test_y, abs_preds))

        print(f"  Fold {fold_i+1}: delta MAE={delta_folds[-1]['mae']:.4f}, "
              f"delta Spr={delta_folds[-1]['spearman_r']:.3f}, "
              f"abs MAE={abs_folds[-1]['mae']:.4f}")

        del ft_model
        gc.collect()

    return {
        "delta_folds": delta_folds,
        "absolute_folds": abs_folds,
        "delta_summary": aggregate_cv_results(delta_folds),
        "absolute_summary": aggregate_cv_results(abs_folds),
    }


def _finetune_film_feats_cv(pretrained_model, pretrained_scaler, pretrained_feat_scaler,
                              zap70_pairs, fp_cache, zap70_feats, mol_data,
                              epochs=50, batch_size=256, lr=1e-4, patience=15):
    """Fine-tune FiLMDeltaWithFeats on ZAP70 with molecule-level CV."""
    smiles_list = mol_data["smiles"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    delta_folds = []

    # Build pair index for feature lookup
    pair_to_idx = {}
    for i, (_, row) in enumerate(zap70_pairs.iterrows()):
        pair_to_idx[(row["mol_a"], row["mol_b"])] = i

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        train_smi = set(smiles_list[train_idx])
        test_smi = set(smiles_list[test_idx])

        train_mask = zap70_pairs["mol_a"].isin(train_smi) & zap70_pairs["mol_b"].isin(train_smi)
        test_mask = zap70_pairs["mol_a"].isin(test_smi) & zap70_pairs["mol_b"].isin(test_smi)

        train_p = zap70_pairs[train_mask]
        test_p = zap70_pairs[test_mask]
        if len(test_p) == 0:
            continue

        train_feat_idx = np.where(train_mask.values)[0]
        test_feat_idx = np.where(test_mask.values)[0]

        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        delta_tr = train_p["delta"].values.astype(np.float32)
        feats_tr = zap70_feats[train_feat_idx]

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        delta_te = test_p["delta"].values.astype(np.float32)
        feats_te = zap70_feats[test_feat_idx]

        ft_model = copy.deepcopy(pretrained_model)
        ft_scaler = copy.deepcopy(pretrained_scaler)
        ft_feat_scaler = copy.deepcopy(pretrained_feat_scaler)

        Xa_tr = torch.FloatTensor(ft_scaler.transform(emb_a_tr))
        Xb_tr = torch.FloatTensor(ft_scaler.transform(emb_b_tr))
        Xf_tr = torch.FloatTensor(ft_feat_scaler.transform(feats_tr))
        yd = torch.FloatTensor(delta_tr)

        Xa_te = torch.FloatTensor(ft_scaler.transform(emb_a_te))
        Xb_te = torch.FloatTensor(ft_scaler.transform(emb_b_te))
        Xf_te = torch.FloatTensor(ft_feat_scaler.transform(feats_te))

        optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr))
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                optimizer.zero_grad()
                pred = ft_model(Xa_tr[bi], Xb_tr[bi], Xf_tr[bi])
                loss = criterion(pred, yd[bi])
                loss.backward()
                optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                vp = ft_model(Xa_te, Xb_te, Xf_te)
                vl = criterion(vp, torch.FloatTensor(delta_te)).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            ft_model.load_state_dict(best_state)

        ft_model.eval()
        with torch.no_grad():
            pred = ft_model(Xa_te, Xb_te, Xf_te).numpy()

        delta_folds.append(compute_delta_metrics(delta_te, pred))
        print(f"  Fold {fold_i+1}: delta MAE={delta_folds[-1]['mae']:.4f}, "
              f"delta Spr={delta_folds[-1]['spearman_r']:.3f}")

        del ft_model
        gc.collect()

    return {
        "delta_folds": delta_folds,
        "delta_summary": aggregate_cv_results(delta_folds),
    }


def evaluate_film_feats_no_pretrain(zap70_pairs, fp_cache, zap70_feats, mol_data,
                                      epochs=50, batch_size=256, lr=1e-3, patience=15):
    """Model 4: FiLMDelta + Edit Features, NO pretraining (train from scratch per fold)."""
    print("\n--- Model 4: FiLMDelta + Edit Features (no pretrain) ---")
    smiles_list = mol_data["smiles"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    feat_dim = zap70_feats.shape[1]

    delta_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        train_smi = set(smiles_list[train_idx])
        test_smi = set(smiles_list[test_idx])

        train_mask = zap70_pairs["mol_a"].isin(train_smi) & zap70_pairs["mol_b"].isin(train_smi)
        test_mask = zap70_pairs["mol_a"].isin(test_smi) & zap70_pairs["mol_b"].isin(test_smi)

        train_p = zap70_pairs[train_mask]
        test_p = zap70_pairs[test_mask]
        if len(test_p) == 0:
            continue

        train_feat_idx = np.where(train_mask.values)[0]
        test_feat_idx = np.where(test_mask.values)[0]

        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        delta_tr = train_p["delta"].values.astype(np.float32)
        feats_tr = zap70_feats[train_feat_idx]

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        delta_te = test_p["delta"].values.astype(np.float32)
        feats_te = zap70_feats[test_feat_idx]

        # Fit scaler on train fold
        scaler = StandardScaler()
        scaler.fit(np.vstack([emb_a_tr, emb_b_tr]))
        feat_scaler = StandardScaler()
        feat_scaler.fit(feats_tr)

        Xa_tr = torch.FloatTensor(scaler.transform(emb_a_tr))
        Xb_tr = torch.FloatTensor(scaler.transform(emb_b_tr))
        Xf_tr = torch.FloatTensor(feat_scaler.transform(feats_tr))
        yd = torch.FloatTensor(delta_tr)

        Xa_te = torch.FloatTensor(scaler.transform(emb_a_te))
        Xb_te = torch.FloatTensor(scaler.transform(emb_b_te))
        Xf_te = torch.FloatTensor(feat_scaler.transform(feats_te))

        model = FiLMDeltaWithFeats(emb_dim=Xa_tr.shape[1], feat_dim=feat_dim,
                                    hidden_dims=[1024, 512, 256], dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            model.train()
            perm = np.random.permutation(len(Xa_tr))
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                optimizer.zero_grad()
                pred = model(Xa_tr[bi], Xb_tr[bi], Xf_tr[bi])
                loss = criterion(pred, yd[bi])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vp = model(Xa_te, Xb_te, Xf_te)
                vl = criterion(vp, torch.FloatTensor(delta_te)).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred = model(Xa_te, Xb_te, Xf_te).numpy()

        delta_folds.append(compute_delta_metrics(delta_te, pred))
        print(f"  Fold {fold_i+1}: delta MAE={delta_folds[-1]['mae']:.4f}, "
              f"delta Spr={delta_folds[-1]['spearman_r']:.3f}")

        del model
        gc.collect()

    return {
        "delta_folds": delta_folds,
        "delta_summary": aggregate_cv_results(delta_folds),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-molecule edit ranking metric
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_molecule_ranking(pairs_df, delta_true, delta_pred, mol_data):
    """Compute per-molecule edit ranking Spearman: for each molecule, rank its
    edits by predicted delta vs true delta."""
    smiles_list = mol_data["smiles"].values
    mol_a_arr = pairs_df["mol_a"].values
    rankings = []
    for smi in smiles_list:
        mask = mol_a_arr == smi
        if mask.sum() < 3:
            continue
        true_sub = delta_true[mask]
        pred_sub = delta_pred[mask]
        sr, _ = spearmanr(true_sub, pred_sub)
        if not np.isnan(sr):
            rankings.append(sr)
    return float(np.mean(rankings)) if rankings else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ZAP70 EDIT ARCHITECTURE COMPARISON — KINASE PRETRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded existing results: {list(results.keys())}")

    # ── 1. Load ZAP70 molecules ──
    print("\n1. Loading ZAP70 molecules...")
    mol_data, _ = load_zap70_molecules()
    zap70_smiles = list(mol_data["smiles"].values)
    n_mols = len(mol_data)
    print(f"   {n_mols} molecules")

    # ── 2. Generate ZAP70 all-pairs (directed) ──
    print("\n2. Generating ZAP70 all-pairs (directed)...")
    zap70_pairs = generate_all_pairs(mol_data)
    print(f"   {len(zap70_pairs):,} directed pairs from {n_mols} molecules")

    # ── 3. Load kinase within-assay pairs ──
    print("\n3. Loading kinase within-assay pairs from shared_pairs_deduped.csv...")
    pairs = pd.read_csv(
        DATA_DIR / "shared_pairs_deduped.csv",
        usecols=["mol_a", "mol_b", "delta", "is_within_assay",
                 "target_chembl_id", "value_a", "value_b"]
    )
    kinase_ids = set(KINASE_TARGETS.values())
    kinase_pairs = pairs[
        (pairs["is_within_assay"] == True) &
        (pairs["target_chembl_id"].isin(kinase_ids))
    ].copy()
    del pairs
    gc.collect()
    print(f"   {len(kinase_pairs):,} kinase within-assay pairs")
    for name, cid in KINASE_TARGETS.items():
        ct = (kinase_pairs["target_chembl_id"] == cid).sum()
        print(f"   {name} ({cid}): {ct:,}")

    # ── 4. Compute fingerprints ──
    print("\n4. Computing Morgan FP (2048d) for all molecules...")
    all_kinase_smi = list(set(
        kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
    ))
    all_smi = list(set(all_kinase_smi + zap70_smiles))
    print(f"   {len(all_smi):,} unique molecules")
    X_all = compute_fingerprints(all_smi, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: X_all[i] for i, smi in enumerate(all_smi)}
    del X_all
    gc.collect()

    # ── 5. Compute pair features ──
    print("\n5. Computing pair features (11d) for ZAP70 pairs...")
    t0 = time.time()
    zap70_feats = compute_all_pair_features(zap70_pairs)
    print(f"   Done in {time.time()-t0:.1f}s, shape={zap70_feats.shape}")

    print("   Computing pair features for kinase pairs...")
    t0 = time.time()
    kinase_feats = compute_all_pair_features(kinase_pairs)
    print(f"   Done in {time.time()-t0:.1f}s, shape={kinase_feats.shape}")

    # ── 6. Run models ──

    # --- Model 1: XGB Subtraction ---
    if "xgb_subtraction" not in results:
        res1 = evaluate_xgb_subtraction(mol_data, fp_cache)
        results["xgb_subtraction"] = res1
        print(f"\n  XGB Subtraction: delta MAE={res1['delta_summary']['mae_mean']:.4f}±{res1['delta_summary']['mae_std']:.4f}, "
              f"Spr={res1['delta_summary']['spearman_r_mean']:.3f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved.")
    else:
        print(f"\n  XGB Subtraction: already done, skipping.")

    # --- Model 2: FiLMDelta + Kinase Pretraining ---
    if "film_delta_kinase" not in results:
        print("\n--- Model 2: FiLMDelta + Kinase Pretraining ---")
        film_model, film_scaler, film_pretrain_metrics = pretrain_film_delta(
            kinase_pairs, fp_cache,
            hidden_dims=[1024, 512, 256], epochs=100, batch_size=256, lr=1e-3, patience=15
        )
        res2 = _finetune_film_cv(
            film_model, film_scaler, zap70_pairs, fp_cache, mol_data,
            epochs=50, batch_size=256, lr=1e-4, patience=15
        )
        res2["pretrain_metrics"] = film_pretrain_metrics
        results["film_delta_kinase"] = res2
        print(f"\n  FiLMDelta+Kinase: delta MAE={res2['delta_summary']['mae_mean']:.4f}±{res2['delta_summary']['mae_std']:.4f}, "
              f"Spr={res2['delta_summary']['spearman_r_mean']:.3f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved.")
        del film_model, film_scaler
        gc.collect()
    else:
        print(f"\n  FiLMDelta+Kinase: already done, skipping.")

    # --- Model 3: DualObjective + Kinase Pretraining ---
    if "dual_objective_kinase" not in results:
        print("\n--- Model 3: DualObjective + Kinase Pretraining ---")
        dual_model, dual_scaler, dual_pretrain_metrics = pretrain_dual_objective(
            kinase_pairs, fp_cache,
            hidden_dims=[512, 256], epochs=100, batch_size=256, lr=1e-3, patience=15,
            lambda_abs=0.3
        )
        res3 = _finetune_dual_cv(
            dual_model, dual_scaler, zap70_pairs, fp_cache, mol_data,
            epochs=50, batch_size=256, lr=1e-4, patience=15, lambda_abs=0.3
        )
        res3["pretrain_metrics"] = dual_pretrain_metrics
        results["dual_objective_kinase"] = res3
        print(f"\n  DualObj+Kinase: delta MAE={res3['delta_summary']['mae_mean']:.4f}±{res3['delta_summary']['mae_std']:.4f}, "
              f"Spr={res3['delta_summary']['spearman_r_mean']:.3f}")
        if "absolute_summary" in res3 and res3["absolute_summary"]:
            print(f"                  abs MAE={res3['absolute_summary']['mae_mean']:.4f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved.")
        del dual_model, dual_scaler
        gc.collect()
    else:
        print(f"\n  DualObj+Kinase: already done, skipping.")

    # --- Model 4: FiLMDelta + Edit Features (no pretrain) ---
    if "film_delta_feats" not in results:
        res4 = evaluate_film_feats_no_pretrain(
            zap70_pairs, fp_cache, zap70_feats, mol_data,
            epochs=50, batch_size=256, lr=1e-3, patience=15
        )
        results["film_delta_feats"] = res4
        print(f"\n  FiLMDelta+Feats: delta MAE={res4['delta_summary']['mae_mean']:.4f}±{res4['delta_summary']['mae_std']:.4f}, "
              f"Spr={res4['delta_summary']['spearman_r_mean']:.3f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved.")
    else:
        print(f"\n  FiLMDelta+Feats: already done, skipping.")

    # --- Model 5: FiLMDelta + Edit Features + Kinase Pretraining ---
    if "film_delta_feats_kinase" not in results:
        print("\n--- Model 5: FiLMDelta + Edit Features + Kinase Pretraining ---")
        feats_model, feats_scaler, feats_feat_scaler, feats_pretrain_metrics = pretrain_film_with_feats(
            kinase_pairs, fp_cache, kinase_feats,
            hidden_dims=[1024, 512, 256], epochs=100, batch_size=256, lr=1e-3, patience=15
        )
        res5 = _finetune_film_feats_cv(
            feats_model, feats_scaler, feats_feat_scaler,
            zap70_pairs, fp_cache, zap70_feats, mol_data,
            epochs=50, batch_size=256, lr=1e-4, patience=15
        )
        res5["pretrain_metrics"] = feats_pretrain_metrics
        results["film_delta_feats_kinase"] = res5
        print(f"\n  FiLMDelta+Feats+Kinase: delta MAE={res5['delta_summary']['mae_mean']:.4f}±{res5['delta_summary']['mae_std']:.4f}, "
              f"Spr={res5['delta_summary']['spearman_r_mean']:.3f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved.")
        del feats_model, feats_scaler, feats_feat_scaler
        gc.collect()
    else:
        print(f"\n  FiLMDelta+Feats+Kinase: already done, skipping.")

    # ── 7. Final Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'delta MAE':>12} {'delta Spr':>12} {'abs MAE':>12}")
    print("-" * 71)

    model_order = [
        ("xgb_subtraction", "XGB Subtraction"),
        ("film_delta_kinase", "FiLMDelta + Kinase PT"),
        ("dual_objective_kinase", "DualObjective + Kinase PT"),
        ("film_delta_feats", "FiLMDelta + EditFeats"),
        ("film_delta_feats_kinase", "FiLMDelta + EditFeats + Kinase PT"),
    ]

    for key, label in model_order:
        if key not in results:
            continue
        r = results[key]
        ds = r.get("delta_summary", {})
        a_s = r.get("absolute_summary", {})
        d_mae = f"{ds.get('mae_mean', float('nan')):.4f}+-{ds.get('mae_std', 0):.4f}"
        d_spr = f"{ds.get('spearman_r_mean', float('nan')):.3f}"
        a_mae = f"{a_s.get('mae_mean', float('nan')):.4f}" if a_s else "N/A"
        print(f"{label:<35} {d_mae:>12} {d_spr:>12} {a_mae:>12}")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
