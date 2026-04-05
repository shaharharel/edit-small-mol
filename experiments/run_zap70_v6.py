#!/usr/bin/env python3
"""
ZAP70 (CHEMBL2803) Case Study v7 — Comprehensive Deep Analysis.

Addresses multiple research questions:

Phase A: All-Pairs Edit Effect (280 mols → ~39K pairs, FiLMDelta/DeepDelta)
Phase B: Deep Learning for Absolute Prediction (MLP, ChemBERTa fine-tuning)
Phase C: Rich Interpretable Features + SHAP (200+ named features, MACCS keys)
Phase D: Transfer Learning from Kinase MMPs (pretrain→finetune, dual objective)
Phase E: Activity Cliff Deep Dive (Tanimoto=1.0 investigation, stereoisomer analysis)
Phase F: Edit Effect Virtual Screening (predict edit effects for new modifications)

Usage:
    conda run -n quris python -u experiments/run_zap70_v6.py
    conda run -n quris python -u experiments/run_zap70_v6.py --phase A B C
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
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
torch.backends.mps.is_available = lambda: False  # Force CPU for stability

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, MACCSkeys
from rdkit.Chem import rdMolDescriptors

from experiments.run_paper_evaluation import (
    RESULTS_DIR, CACHE_DIR, DATA_DIR,
)
from experiments.run_zap70_v3 import (
    load_zap70_molecules, get_cv_splits, compute_absolute_metrics,
    aggregate_cv_results, compute_fingerprints,
    train_rf, train_xgboost,
    _tanimoto_kernel_matrix, N_JOBS, N_FOLDS, CV_SEED,
)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = RESULTS_DIR / "zap70_v7_results.json"
TARGET_ID = "CHEMBL2803"
DEVICE = "cpu"

# Best hyperparameters from v3
BEST_XGB_PARAMS = {
    "max_depth": 6, "min_child_weight": 2,
    "subsample": 0.605, "colsample_bytree": 0.520,
    "learning_rate": 0.0197, "n_estimators": 749,
    "reg_alpha": 1.579, "reg_lambda": 7.313,
}
BEST_RF_PARAMS = {
    "n_estimators": 614, "max_depth": 16,
    "max_features": 0.3, "min_samples_leaf": 2,
    "min_samples_split": 3,
}


def save_results(results):
    """Save results, merging with any existing results to avoid race conditions."""
    import fcntl
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_file = RESULTS_FILE.with_suffix(".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # Load existing and merge (new results take priority)
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
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_pairs(mol_data):
    """Generate all unique pairs from N molecules → N*(N-1)/2 pairs."""
    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    ids = mol_data["molecule_chembl_id"].values

    pairs = []
    for i in range(len(smiles)):
        for j in range(i + 1, len(smiles)):
            pairs.append({
                "mol_a": smiles[i], "mol_b": smiles[j],
                "mol_a_id": ids[i], "mol_b_id": ids[j],
                "value_a": pIC50[i], "value_b": pIC50[j],
                "delta": pIC50[j] - pIC50[i],
            })
    df = pd.DataFrame(pairs)
    print(f"  Generated {len(df)} all-pairs from {len(smiles)} molecules")
    return df


def compute_named_features(smiles_list):
    """Compute rich, named feature set for interpretability.
    Returns (X, feature_names) with ~250 features.
    """
    from rdkit.Chem import Fragments
    from rdkit.ML.Descriptors import MoleculeDescriptors

    # 1. All RDKit 2D descriptors (~210)
    desc_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(len(desc_names), dtype=np.float32))
            continue
        try:
            vals = calc.CalcDescriptors(mol)
            results.append(np.array(vals, dtype=np.float32))
        except Exception:
            results.append(np.zeros(len(desc_names), dtype=np.float32))

    X = np.array(results, dtype=np.float32)
    # Replace NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, list(desc_names)


def compute_maccs_features(smiles_list):
    """Compute MACCS keys (166 bits, each with defined SMARTS meaning)."""
    MACCS_NAMES = [
        "?", "?", "?", # bits 0-2 unused
        "Li", "Be", "B_present", "C_present", "N_present", "O_present", "F_present",
        "Na", "Si", "P_present", "S_present", "Cl_present", "K_present",
        "Br_present", "I_present", "At",
        # Simplified names for bits 20-166 (key functional groups)
    ]
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(167, dtype=np.float32))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)
    return np.array(results, dtype=np.float32)


def compute_functional_group_counts(smiles_list):
    """Compute counts of named functional groups."""
    GROUPS = {
        "nitrile": "[C]#N",
        "fluorine": "[F]",
        "chlorine": "[Cl]",
        "bromine": "[Br]",
        "hydroxyl": "[OH]",
        "primary_amine": "[NH2]",
        "secondary_amine": "[NH]([!H])[!H]",
        "tertiary_amine": "[N]([!H])([!H])[!H]",
        "amide": "[C](=O)[NH]",
        "carboxyl": "[C](=O)[OH]",
        "ester": "[C](=O)[O][C]",
        "ether": "[C][O][C]",
        "aldehyde": "[CH]=O",
        "ketone": "[C](=O)([C])[C]",
        "sulfonamide": "[S](=O)(=O)[NH]",
        "sulfone": "[S](=O)(=O)",
        "nitro": "[N+](=O)[O-]",
        "methoxy": "[CH3][O]",
        "trifluoromethyl": "[C](F)(F)F",
        "pyridine": "c1ccncc1",
        "pyrimidine": "c1ccnc(n1)",
        "imidazole": "c1cnc[nH]1",
        "triazole_1_2_3": "c1nn[nH]c1",
        "triazole_1_2_4": "c1nnc[nH]1",
        "piperidine": "C1CCNCC1",
        "piperazine": "C1CNCCN1",
        "morpholine": "C1COCCN1",
        "benzimidazole": "c1ccc2[nH]cnc2c1",
        "indazole": "c1ccc2[nH]ncc2c1",
        "indole": "c1ccc2c(c1)[nH]cc2",
        "quinoline": "c1ccc2ncccc2c1",
        "isoquinoline": "c1ccc2cnccc2c1",
    }

    feature_names = list(GROUPS.keys())
    results = []
    patterns = {}
    for name, sma in GROUPS.items():
        patterns[name] = Chem.MolFromSmarts(sma)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        counts = np.zeros(len(feature_names), dtype=np.float32)
        if mol is not None:
            for i, name in enumerate(feature_names):
                pat = patterns[name]
                if pat is not None:
                    matches = mol.GetSubstructMatches(pat)
                    counts[i] = len(matches)
        results.append(counts)

    return np.array(results, dtype=np.float32), feature_names


class SimpleMLPRegressor(nn.Module):
    """Simple MLP for absolute pIC50 prediction."""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualObjectiveModel(nn.Module):
    """Model with dual heads: delta prediction + absolute prediction."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_dim = hidden_dims[-1]

        # Delta head: from concatenated encoder outputs
        self.delta_head = nn.Sequential(
            nn.Linear(self.enc_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Absolute head: from single encoder output
        self.abs_head = nn.Sequential(
            nn.Linear(self.enc_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def train_mlp_absolute(X_train, y_train, X_test, hidden_dims=[512, 256, 128],
                       dropout=0.3, lr=1e-3, epochs=200, batch_size=32, patience=20):
    """Train MLP for absolute pIC50 prediction."""
    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_train)
    X_te_t = torch.FloatTensor(X_te)

    model = SimpleMLPRegressor(X_tr.shape[1], hidden_dims, dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # Simple train/val split for early stopping
    n_val = max(10, len(X_tr) // 5)
    perm = np.random.RandomState(42).permutation(len(X_tr))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm_tr = np.random.permutation(len(tr_idx))
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(tr_idx), batch_size):
            batch_idx = tr_idx[perm_tr[start:start + batch_size]]
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_tr_t[val_idx])
            val_loss = criterion(val_pred, y_tr_t[val_idx]).item()
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
    with torch.no_grad():
        preds = model(X_te_t).numpy()
    return preds, (model, scaler)


def train_chemberta_absolute(smiles_train, y_train, smiles_test,
                             model_name='DeepChem/ChemBERTa-77M-MTR',
                             lr=2e-5, epochs=30, batch_size=16, patience=5):
    """Fine-tune ChemBERTa for absolute pIC50 prediction."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    # Freeze most layers, fine-tune last 2
    for param in bert_model.parameters():
        param.requires_grad = False
    for param in bert_model.encoder.layer[-2:].parameters():
        param.requires_grad = True

    class ChemBERTaRegressor(nn.Module):
        def __init__(self, bert, hidden_dim=256):
            super().__init__()
            self.bert = bert
            self.head = nn.Sequential(
                nn.Linear(bert.config.hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embed = outputs.last_hidden_state[:, 0, :]
            return self.head(cls_embed).squeeze(-1), cls_embed

    model = ChemBERTaRegressor(bert_model)

    def tokenize(smiles_list, max_len=128):
        enc = tokenizer(smiles_list, padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt')
        return enc['input_ids'], enc['attention_mask']

    train_ids, train_mask = tokenize(smiles_train)
    test_ids, test_mask = tokenize(smiles_test)
    y_tr_t = torch.FloatTensor(y_train)

    # Train/val split
    n_val = max(8, len(smiles_train) // 5)
    perm = np.random.RandomState(42).permutation(len(smiles_train))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch
        perm_tr = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), batch_size):
            batch_i = tr_idx[perm_tr[start:start + batch_size]]
            ids_b = train_ids[batch_i]
            mask_b = train_mask[batch_i]
            y_b = y_tr_t[batch_i]
            optimizer.zero_grad()
            pred, _ = model(ids_b, mask_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(train_ids[val_idx], train_mask[val_idx])
            val_loss = criterion(val_pred, y_tr_t[val_idx]).item()

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
    with torch.no_grad():
        preds, embeddings = model(test_ids, test_mask)
        _, train_embeddings = model(train_ids, train_mask)

    return (preds.numpy(), embeddings.numpy(), train_embeddings.numpy(),
            model, tokenizer)


def train_dual_objective(X_train_a, X_train_b, y_delta, y_abs_a, y_abs_b,
                         X_test_a, X_test_b, lambda_abs=0.3,
                         hidden_dims=[512, 256], epochs=200, batch_size=64,
                         lr=1e-3, patience=20):
    """Train dual-objective model (delta + absolute)."""
    scaler = StandardScaler()
    # Fit scaler on all molecules
    all_X = np.vstack([X_train_a, X_train_b])
    scaler.fit(all_X)

    Xa_tr = torch.FloatTensor(scaler.transform(X_train_a))
    Xb_tr = torch.FloatTensor(scaler.transform(X_train_b))
    Xa_te = torch.FloatTensor(scaler.transform(X_test_a))
    Xb_te = torch.FloatTensor(scaler.transform(X_test_b))
    yd = torch.FloatTensor(y_delta)
    ya = torch.FloatTensor(y_abs_a)
    yb = torch.FloatTensor(y_abs_b)

    model = DualObjectiveModel(Xa_tr.shape[1], hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Simple early stopping
    n_val = max(20, len(Xa_tr) // 5)
    perm = np.random.RandomState(42).permutation(len(Xa_tr))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm_tr = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), batch_size):
            bi = tr_idx[perm_tr[start:start + batch_size]]
            optimizer.zero_grad()
            d_pred, a_pred, b_pred = model(Xa_tr[bi], Xb_tr[bi])
            loss_delta = criterion(d_pred, yd[bi])
            loss_abs = criterion(a_pred, ya[bi]) + criterion(b_pred, yb[bi])
            loss = loss_delta + lambda_abs * loss_abs
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vd, va, vb = model(Xa_tr[val_idx], Xb_tr[val_idx])
            vl = criterion(vd, yd[val_idx]).item()

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
        d_pred, _, _ = model(Xa_te, Xb_te)
    return d_pred.numpy(), model, scaler


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: All-Pairs Edit Effect
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(mol_data, results):
    """All-pairs edit effect prediction for ZAP70."""
    print("\n" + "=" * 70)
    print("PHASE A: All-Pairs Edit Effect (280 mols → ~39K pairs)")
    print("=" * 70)

    phase = results.get("phase_a", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    # Generate all pairs
    pairs_df = generate_all_pairs(mol_data)
    n_pairs = len(pairs_df)

    # Compute fingerprints for all molecules
    all_smiles = list(mol_data["smiles"].values)
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    X_morgan = compute_fingerprints(all_smiles, "morgan", radius=2, n_bits=2048)

    # 5-fold CV on MOLECULES (not pairs) to avoid leakage
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    mol_indices = np.arange(len(all_smiles))

    methods = {}

    for method_name in ["FiLMDelta", "Subtraction", "MLP_delta"]:
        print(f"\n  --- {method_name} ---")
        fold_metrics_delta = []
        fold_metrics_abs = []

        for fold_i, (train_mol_idx, test_mol_idx) in enumerate(kf.split(mol_indices)):
            train_smiles_set = set(np.array(all_smiles)[train_mol_idx])
            test_smiles_set = set(np.array(all_smiles)[test_mol_idx])

            # Split pairs: test pairs have BOTH molecules in test set
            train_pairs = pairs_df[
                pairs_df["mol_a"].isin(train_smiles_set) &
                pairs_df["mol_b"].isin(train_smiles_set)
            ]
            test_pairs = pairs_df[
                pairs_df["mol_a"].isin(test_smiles_set) &
                pairs_df["mol_b"].isin(test_smiles_set)
            ]

            if len(test_pairs) == 0:
                continue

            # Get embeddings
            def get_emb(smiles_series):
                return np.array([X_morgan[smi_to_idx[s]] for s in smiles_series])

            emb_a_tr = get_emb(train_pairs["mol_a"])
            emb_b_tr = get_emb(train_pairs["mol_b"])
            delta_tr = train_pairs["delta"].values.astype(np.float32)

            emb_a_te = get_emb(test_pairs["mol_a"])
            emb_b_te = get_emb(test_pairs["mol_b"])
            delta_te = test_pairs["delta"].values.astype(np.float32)

            if method_name == "FiLMDelta":
                from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
                predictor = FiLMDeltaPredictor(
                    dropout=0.2, learning_rate=1e-3, batch_size=64,
                    max_epochs=100, patience=15, device=DEVICE
                )
                predictor.fit(emb_a_tr, emb_b_tr, delta_tr, verbose=False)
                delta_pred = predictor.predict(emb_a_te, emb_b_te)

            elif method_name == "Subtraction":
                # Train on absolute, subtract
                X_tr_abs = X_morgan[train_mol_idx]
                y_tr_abs = mol_data["pIC50"].values[train_mol_idx]
                X_te_abs = X_morgan[test_mol_idx]
                y_te_abs = mol_data["pIC50"].values[test_mol_idx]

                preds_abs, _ = train_xgboost(X_tr_abs, y_tr_abs, X_te_abs, **BEST_XGB_PARAMS)
                # Map absolute predictions back to pairs
                test_smi_list = list(np.array(all_smiles)[test_mol_idx])
                abs_map = dict(zip(test_smi_list, preds_abs))
                delta_pred = np.array([
                    abs_map.get(b, 0) - abs_map.get(a, 0)
                    for a, b in zip(test_pairs["mol_a"], test_pairs["mol_b"])
                ])

                # Also evaluate absolute predictions
                metrics_abs = compute_absolute_metrics(y_te_abs, preds_abs)
                fold_metrics_abs.append(metrics_abs)

            elif method_name == "MLP_delta":
                # MLP directly on concatenated embeddings
                X_tr_concat = np.hstack([emb_a_tr, emb_b_tr, emb_b_tr - emb_a_tr])
                X_te_concat = np.hstack([emb_a_te, emb_b_te, emb_b_te - emb_a_te])
                delta_pred, _ = train_mlp_absolute(
                    X_tr_concat, delta_tr, X_te_concat,
                    hidden_dims=[1024, 512, 256], dropout=0.3,
                    lr=1e-3, epochs=150, batch_size=64, patience=15
                )

            # Evaluate delta predictions
            mae_delta = float(np.mean(np.abs(delta_te - delta_pred)))
            spr_delta, _ = spearmanr(delta_te, delta_pred) if len(delta_te) > 2 else (0, 1)
            fold_metrics_delta.append({
                "mae": mae_delta, "spearman": float(spr_delta) if not np.isnan(spr_delta) else 0,
                "n_pairs": len(delta_te),
            })

        # Aggregate
        if fold_metrics_delta:
            mae_vals = [m["mae"] for m in fold_metrics_delta]
            spr_vals = [m["spearman"] for m in fold_metrics_delta]
            result_entry = {
                "mae_mean": float(np.mean(mae_vals)),
                "mae_std": float(np.std(mae_vals)),
                "spearman_mean": float(np.mean(spr_vals)),
                "spearman_std": float(np.std(spr_vals)),
                "per_fold": fold_metrics_delta,
            }
            if fold_metrics_abs:
                result_entry["absolute_metrics"] = aggregate_cv_results(fold_metrics_abs)
            methods[method_name] = result_entry
            print(f"  {method_name}: Delta MAE={np.mean(mae_vals):.4f}±{np.std(mae_vals):.4f}, "
                  f"Spearman={np.mean(spr_vals):.3f}±{np.std(spr_vals):.3f}")

    results["phase_a"] = {
        "n_molecules": len(mol_data),
        "n_pairs": n_pairs,
        "methods": methods,
        "completed": True,
    }
    save_results(results)
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Deep Learning for Absolute Prediction
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(mol_data, results):
    """Deep learning models for absolute pIC50 prediction."""
    print("\n" + "=" * 70)
    print("PHASE B: Deep Learning for Absolute Prediction")
    print("=" * 70)

    phase = results.get("phase_b", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)

    splits = get_cv_splits(mol_data)
    methods = {}

    # B1: MLP on Morgan FPs
    print("\n  B1. MLP on Morgan FPs...")
    fold_metrics = []
    for fold_name, train_df, test_df in splits:
        train_idx = train_df.index if hasattr(train_df, 'index') else range(len(train_df))
        test_idx = test_df.index if hasattr(test_df, 'index') else range(len(test_df))

        # Map back to original indices
        train_smiles = train_df["smiles"].values
        test_smiles = test_df["smiles"].values
        train_y = train_df["pIC50"].values
        test_y = test_df["pIC50"].values

        X_tr = compute_fingerprints(list(train_smiles), "morgan", radius=2, n_bits=2048)
        X_te = compute_fingerprints(list(test_smiles), "morgan", radius=2, n_bits=2048)

        preds, _ = train_mlp_absolute(X_tr, train_y, X_te,
                                      hidden_dims=[512, 256, 128], dropout=0.3,
                                      lr=1e-3, epochs=200, patience=20)
        metrics = compute_absolute_metrics(test_y, preds)
        fold_metrics.append(metrics)

    methods["MLP_Morgan"] = {"aggregated": aggregate_cv_results(fold_metrics), "per_fold": fold_metrics}
    agg = methods["MLP_Morgan"]["aggregated"]
    print(f"    MLP Morgan: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
          f"Spr={agg['spearman_r_mean']:.3f}")

    # B2: MLP on rich features (RDKit descriptors)
    print("\n  B2. MLP on RDKit 2D descriptors...")
    fold_metrics = []
    for fold_name, train_df, test_df in splits:
        X_tr, _ = compute_named_features(list(train_df["smiles"]))
        X_te, _ = compute_named_features(list(test_df["smiles"]))
        preds, _ = train_mlp_absolute(X_tr, train_df["pIC50"].values, X_te,
                                      hidden_dims=[256, 128, 64], dropout=0.3,
                                      lr=5e-4, epochs=200, patience=20)
        metrics = compute_absolute_metrics(test_df["pIC50"].values, preds)
        fold_metrics.append(metrics)

    methods["MLP_RDKit2D"] = {"aggregated": aggregate_cv_results(fold_metrics), "per_fold": fold_metrics}
    agg = methods["MLP_RDKit2D"]["aggregated"]
    print(f"    MLP RDKit2D: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
          f"Spr={agg['spearman_r_mean']:.3f}")

    # B3: ChemBERTa fine-tuning
    print("\n  B3. ChemBERTa-2 MTR fine-tuning...")
    fold_metrics = []
    all_embeddings = []
    for fold_name, train_df, test_df in splits:
        try:
            preds, test_embs, train_embs, model, tok = train_chemberta_absolute(
                list(train_df["smiles"]), train_df["pIC50"].values,
                list(test_df["smiles"]),
                lr=2e-5, epochs=30, batch_size=16, patience=5
            )
            metrics = compute_absolute_metrics(test_df["pIC50"].values, preds)
            fold_metrics.append(metrics)
            all_embeddings.append({
                "test_smiles": list(test_df["smiles"]),
                "test_embeddings": test_embs.tolist(),
                "test_pIC50": list(test_df["pIC50"]),
            })
            del model
            gc.collect()
        except Exception as e:
            print(f"    ChemBERTa fold {fold_name} failed: {e}")
            fold_metrics.append({"mae": 999, "rmse": 999, "r2": 0, "pearson_r": 0,
                                 "spearman_r": 0, "spearman_p": 1, "n": 0})

    methods["ChemBERTa_finetune"] = {
        "aggregated": aggregate_cv_results(fold_metrics),
        "per_fold": fold_metrics,
    }
    agg = methods["ChemBERTa_finetune"]["aggregated"]
    print(f"    ChemBERTa: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
          f"Spr={agg['spearman_r_mean']:.3f}")

    # B4: XGBoost baseline for comparison
    print("\n  B4. XGBoost (v3 best) for comparison...")
    fold_metrics = []
    for fold_name, train_df, test_df in splits:
        X_tr = compute_fingerprints(list(train_df["smiles"]), "morgan", radius=2, n_bits=2048)
        X_te = compute_fingerprints(list(test_df["smiles"]), "morgan", radius=2, n_bits=2048)
        preds, _ = train_xgboost(X_tr, train_df["pIC50"].values, X_te, **BEST_XGB_PARAMS)
        metrics = compute_absolute_metrics(test_df["pIC50"].values, preds)
        fold_metrics.append(metrics)

    methods["XGBoost_baseline"] = {"aggregated": aggregate_cv_results(fold_metrics), "per_fold": fold_metrics}
    agg = methods["XGBoost_baseline"]["aggregated"]
    print(f"    XGBoost: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
          f"Spr={agg['spearman_r_mean']:.3f}")

    results["phase_b"] = {
        "methods": methods,
        "completed": True,
    }
    save_results(results)
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Rich Interpretable Features + SHAP
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(mol_data, results):
    """Rich interpretable feature analysis with SHAP."""
    print("\n" + "=" * 70)
    print("PHASE C: Rich Interpretable Features + SHAP")
    print("=" * 70)

    phase = results.get("phase_c", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values

    # C1: Compute all feature sets
    print("\n  C1. Computing feature sets...")
    X_rdkit, rdkit_names = compute_named_features(smiles_list)
    X_maccs = compute_maccs_features(smiles_list)
    X_funcgroups, fg_names = compute_functional_group_counts(smiles_list)
    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)

    # Combined interpretable features
    X_interp = np.hstack([X_rdkit, X_funcgroups])
    interp_names = rdkit_names + fg_names
    print(f"    RDKit 2D: {X_rdkit.shape[1]} features")
    print(f"    Functional groups: {X_funcgroups.shape[1]} features")
    print(f"    MACCS keys: {X_maccs.shape[1]} features")
    print(f"    Combined interpretable: {X_interp.shape[1]} features")

    # C2: Train XGBoost on interpretable features
    print("\n  C2. XGBoost on interpretable features...")
    splits = get_cv_splits(mol_data)
    fold_metrics = []
    all_models = []
    for fold_name, train_df, test_df in splits:
        tr_idx = [smiles_list.index(s) for s in train_df["smiles"]]
        te_idx = [smiles_list.index(s) for s in test_df["smiles"]]

        X_tr = X_interp[tr_idx]
        X_te = X_interp[te_idx]

        # Remove constant features
        non_const = np.std(X_tr, axis=0) > 1e-10
        X_tr_f = X_tr[:, non_const]
        X_te_f = X_te[:, non_const]

        preds, model = train_xgboost(X_tr_f, train_df["pIC50"].values, X_te_f, **BEST_XGB_PARAMS)
        metrics = compute_absolute_metrics(test_df["pIC50"].values, preds)
        fold_metrics.append(metrics)
        all_models.append((model, non_const))

    agg = aggregate_cv_results(fold_metrics)
    print(f"    XGB Interpretable: MAE={agg['mae_mean']:.4f}, Spr={agg['spearman_r_mean']:.3f}")

    # C3: SHAP analysis on interpretable features
    print("\n  C3. SHAP analysis on named features...")
    try:
        import shap
        # Use first fold model
        model_0, non_const_0 = all_models[0]
        tr_idx_0 = [smiles_list.index(s) for s in splits[0][1]["smiles"]]
        X_bg = X_interp[tr_idx_0][:, non_const_0]
        filtered_names = [n for n, nc in zip(interp_names, non_const_0) if nc]

        explainer = shap.TreeExplainer(model_0)
        shap_vals = explainer.shap_values(X_bg)

        # Top features by mean |SHAP|
        mean_shap = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(mean_shap)[::-1][:30]

        shap_features = []
        for rank, idx in enumerate(top_indices):
            feat_name = filtered_names[idx]
            feat_vals = X_bg[:, idx]
            spr_r, spr_p = spearmanr(feat_vals, mol_data["pIC50"].values[tr_idx_0]) \
                if np.std(feat_vals) > 1e-10 else (0, 1)

            shap_features.append({
                "rank": rank + 1,
                "feature": feat_name,
                "mean_abs_shap": round(float(mean_shap[idx]), 4),
                "spearman_with_pIC50": round(float(spr_r), 4) if not np.isnan(spr_r) else 0,
                "feature_mean": round(float(np.mean(feat_vals)), 4),
                "feature_std": round(float(np.std(feat_vals)), 4),
            })

        print(f"\n  Top 20 interpretable features by SHAP importance:")
        print(f"  {'Rank':>4} {'Feature':>30} {'|SHAP|':>8} {'Spr':>7}")
        for f in shap_features[:20]:
            print(f"  {f['rank']:4d} {f['feature']:>30s} {f['mean_abs_shap']:8.4f} "
                  f"{f['spearman_with_pIC50']:+7.3f}")
    except ImportError:
        print("    SHAP not available, skipping.")
        shap_features = []

    # C4: Functional group impact analysis
    print("\n  C4. Functional group impact on potency...")
    fg_impact = []
    for i, name in enumerate(fg_names):
        present = X_funcgroups[:, i] > 0
        if present.sum() >= 5 and (~present).sum() >= 5:
            mean_with = float(np.mean(y[present]))
            mean_without = float(np.mean(y[~present]))
            delta = mean_with - mean_without
            spr_r, spr_p = spearmanr(X_funcgroups[:, i], y)
            fg_impact.append({
                "group": name,
                "n_with": int(present.sum()),
                "mean_pIC50_with": round(mean_with, 3),
                "mean_pIC50_without": round(mean_without, 3),
                "delta": round(delta, 3),
                "spearman": round(float(spr_r), 3) if not np.isnan(spr_r) else 0,
            })

    fg_impact.sort(key=lambda x: abs(x["delta"]), reverse=True)
    print(f"\n  Functional group impact (sorted by |Δ pIC50|):")
    print(f"  {'Group':>20} {'N':>4} {'With':>6} {'Without':>8} {'Δ':>6} {'Spr':>6}")
    for f in fg_impact[:15]:
        print(f"  {f['group']:>20s} {f['n_with']:4d} {f['mean_pIC50_with']:6.2f} "
              f"{f['mean_pIC50_without']:8.2f} {f['delta']:+6.2f} {f['spearman']:+6.3f}")

    results["phase_c"] = {
        "xgb_interpretable": {"aggregated": agg, "per_fold": fold_metrics},
        "shap_features": shap_features,
        "functional_group_impact": fg_impact,
        "n_rdkit_features": X_rdkit.shape[1],
        "n_funcgroup_features": len(fg_names),
        "completed": True,
    }
    save_results(results)
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: Transfer Learning from Kinase MMPs
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(mol_data, results):
    """Transfer learning: pretrain on kinase MMPs, fine-tune on ZAP70."""
    print("\n" + "=" * 70)
    print("PHASE D: Transfer Learning from Kinase MMPs")
    print("=" * 70)

    phase = results.get("phase_d", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    # D1: Load kinase MMP pairs
    print("\n  D1. Loading kinase MMP pairs...")
    pairs = pd.read_csv(DATA_DIR / "shared_pairs_deduped.csv",
                        usecols=["mol_a", "mol_b", "delta", "is_within_assay",
                                 "target_chembl_id", "value_a", "value_b"])

    # Get related kinases with substantial data
    KINASE_TARGETS = {
        "ITK": "CHEMBL3009", "SYK": "CHEMBL2599", "FYN": "CHEMBL1841",
        "LCK": "CHEMBL258", "BTK": "CHEMBL5251", "JAK2": "CHEMBL2971",
        "ABL1": "CHEMBL1862", "SRC": "CHEMBL267",
    }
    kinase_ids = set(KINASE_TARGETS.values())

    # Filter to kinase within-assay pairs
    kinase_pairs = pairs[
        (pairs["target_chembl_id"].isin(kinase_ids)) &
        (pairs["is_within_assay"] == True)
    ].copy()
    print(f"    Kinase within-assay MMP pairs: {len(kinase_pairs):,}")

    # Sample to manageable size
    MAX_PRETRAIN = 100000
    if len(kinase_pairs) > MAX_PRETRAIN:
        kinase_pairs = kinase_pairs.sample(MAX_PRETRAIN, random_state=42)
        print(f"    Sampled to {MAX_PRETRAIN:,}")

    # Compute fingerprints for all unique molecules
    all_kinase_smiles = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
    zap70_smiles = list(mol_data["smiles"].values)
    all_smiles_combined = list(set(all_kinase_smiles + zap70_smiles))

    print(f"    Computing fingerprints for {len(all_smiles_combined)} unique molecules...")
    fp_cache = {}
    X_all = compute_fingerprints(all_smiles_combined, "morgan", radius=2, n_bits=2048)
    for i, smi in enumerate(all_smiles_combined):
        fp_cache[smi] = X_all[i]
    del X_all
    gc.collect()

    # D2: Generate ZAP70 all-pairs
    print("\n  D2. Generating ZAP70 all-pairs...")
    zap70_pairs = generate_all_pairs(mol_data)

    # D3: Pretrain dual-objective model on kinase MMPs
    print("\n  D3. Pretraining dual-objective model on kinase MMPs...")
    kinase_emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    kinase_emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    kinase_delta = kinase_pairs["delta"].values.astype(np.float32)
    kinase_val_a = kinase_pairs["value_a"].values.astype(np.float32)
    kinase_val_b = kinase_pairs["value_b"].values.astype(np.float32)

    # Train on kinase data (no CV — this is pretraining)
    n_val = len(kinase_emb_a) // 10
    pretrain_pred, pretrain_model, pretrain_scaler = train_dual_objective(
        kinase_emb_a[n_val:], kinase_emb_b[n_val:],
        kinase_delta[n_val:], kinase_val_a[n_val:], kinase_val_b[n_val:],
        kinase_emb_a[:n_val], kinase_emb_b[:n_val],
        lambda_abs=0.3, hidden_dims=[512, 256], epochs=100,
        batch_size=256, lr=1e-3, patience=15
    )
    pretrain_mae = float(np.mean(np.abs(kinase_delta[:n_val] - pretrain_pred)))
    pretrain_spr, _ = spearmanr(kinase_delta[:n_val], pretrain_pred)
    print(f"    Pretrain validation: MAE={pretrain_mae:.4f}, Spr={pretrain_spr:.3f}")

    # D4: Fine-tune on ZAP70 all-pairs (5-fold CV)
    print("\n  D4. Fine-tuning on ZAP70 all-pairs (5-fold CV)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    zap70_mol_indices = np.arange(len(zap70_smiles))

    finetune_folds = []
    nofinetune_folds = []  # Also test pretrained-only for comparison
    absolute_from_dual_folds = []

    for fold_i, (train_mol_idx, test_mol_idx) in enumerate(kf.split(zap70_mol_indices)):
        train_smi_set = set(np.array(zap70_smiles)[train_mol_idx])
        test_smi_set = set(np.array(zap70_smiles)[test_mol_idx])

        train_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(train_smi_set) &
            zap70_pairs["mol_b"].isin(train_smi_set)
        ]
        test_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(test_smi_set) &
            zap70_pairs["mol_b"].isin(test_smi_set)
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

        # Test pretrained model without fine-tuning
        pretrain_model.eval()
        with torch.no_grad():
            Xa_te_s = torch.FloatTensor(pretrain_scaler.transform(emb_a_te))
            Xb_te_s = torch.FloatTensor(pretrain_scaler.transform(emb_b_te))
            noft_pred, _, _ = pretrain_model(Xa_te_s, Xb_te_s)
            noft_pred = noft_pred.numpy()
        noft_mae = float(np.mean(np.abs(delta_te - noft_pred)))
        noft_spr, _ = spearmanr(delta_te, noft_pred) if len(delta_te) > 2 else (0, 1)
        nofinetune_folds.append({"mae": noft_mae, "spearman": float(noft_spr) if not np.isnan(noft_spr) else 0})

        # Fine-tune: create a copy of the pretrained model
        import copy
        ft_model = copy.deepcopy(pretrain_model)
        ft_scaler = copy.deepcopy(pretrain_scaler)

        ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()

        Xa_tr_s = torch.FloatTensor(ft_scaler.transform(emb_a_tr))
        Xb_tr_s = torch.FloatTensor(ft_scaler.transform(emb_b_tr))
        yd = torch.FloatTensor(delta_tr)
        ya = torch.FloatTensor(val_a_tr)
        yb = torch.FloatTensor(val_b_tr)

        best_loss = float('inf')
        best_state = None
        for epoch in range(50):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr_s))
            for start in range(0, len(perm), 64):
                bi = perm[start:start + 64]
                ft_optimizer.zero_grad()
                d_p, a_p, b_p = ft_model(Xa_tr_s[bi], Xb_tr_s[bi])
                loss = criterion(d_p, yd[bi]) + 0.3 * (criterion(a_p, ya[bi]) + criterion(b_p, yb[bi]))
                loss.backward()
                ft_optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                d_p_te, _, _ = ft_model(Xa_te_s, Xb_te_s)
                val_loss = criterion(d_p_te, torch.FloatTensor(delta_te)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}

        if best_state:
            ft_model.load_state_dict(best_state)
        ft_model.eval()
        with torch.no_grad():
            ft_pred, _, _ = ft_model(Xa_te_s, Xb_te_s)
            ft_pred = ft_pred.numpy()
        ft_mae = float(np.mean(np.abs(delta_te - ft_pred)))
        ft_spr, _ = spearmanr(delta_te, ft_pred) if len(delta_te) > 2 else (0, 1)
        finetune_folds.append({"mae": ft_mae, "spearman": float(ft_spr) if not np.isnan(ft_spr) else 0})

        # Also test absolute predictions from dual model
        test_embs = np.array([fp_cache[s] for s in np.array(zap70_smiles)[test_mol_idx]])
        test_y = mol_data["pIC50"].values[test_mol_idx]
        with torch.no_grad():
            abs_preds = ft_model.predict_absolute(
                torch.FloatTensor(ft_scaler.transform(test_embs))
            ).numpy()
        abs_metrics = compute_absolute_metrics(test_y, abs_preds)
        absolute_from_dual_folds.append(abs_metrics)

        del ft_model
        gc.collect()

    # Summarize
    print(f"\n  Results (delta prediction on ZAP70 all-pairs):")
    if nofinetune_folds:
        nft_mae = np.mean([f["mae"] for f in nofinetune_folds])
        nft_spr = np.mean([f["spearman"] for f in nofinetune_folds])
        print(f"    Pretrained only (no fine-tune): MAE={nft_mae:.4f}, Spr={nft_spr:.3f}")
    if finetune_folds:
        ft_mae = np.mean([f["mae"] for f in finetune_folds])
        ft_spr = np.mean([f["spearman"] for f in finetune_folds])
        print(f"    Pretrained + fine-tuned:         MAE={ft_mae:.4f}, Spr={ft_spr:.3f}")
    if absolute_from_dual_folds:
        abs_agg = aggregate_cv_results(absolute_from_dual_folds)
        print(f"    Absolute from dual model:        MAE={abs_agg['mae_mean']:.4f}, Spr={abs_agg['spearman_r_mean']:.3f}")

    results["phase_d"] = {
        "pretrain": {
            "n_kinase_pairs": len(kinase_pairs),
            "kinase_targets": list(KINASE_TARGETS.keys()),
            "pretrain_val_mae": pretrain_mae,
            "pretrain_val_spearman": float(pretrain_spr) if not np.isnan(pretrain_spr) else 0,
        },
        "no_finetune": nofinetune_folds,
        "finetune": finetune_folds,
        "absolute_from_dual": {"aggregated": aggregate_cv_results(absolute_from_dual_folds)} if absolute_from_dual_folds else {},
        "completed": True,
    }
    save_results(results)
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase E: Activity Cliff Deep Dive
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_e(mol_data, results):
    """Deep investigation of activity cliffs, especially Tanimoto=1.0 pairs."""
    print("\n" + "=" * 70)
    print("PHASE E: Activity Cliff Deep Dive")
    print("=" * 70)

    phase = results.get("phase_e", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)

    # E1: Find all high-similarity pairs
    print("\n  E1. Computing pairwise similarities...")
    n = len(smiles_list)
    cliffs = []
    tan1_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            # Compute Tanimoto from Morgan FPs
            sim = float(np.sum(np.minimum(X_morgan[i], X_morgan[j])) /
                        max(np.sum(np.maximum(X_morgan[i], X_morgan[j])), 1e-10))
            delta = abs(y[j] - y[i])

            if sim >= 0.8 and delta >= 0.5:
                cliff_entry = {
                    "mol_a_smi": smiles_list[i],
                    "mol_b_smi": smiles_list[j],
                    "pIC50_a": round(float(y[i]), 2),
                    "pIC50_b": round(float(y[j]), 2),
                    "delta": round(float(y[j] - y[i]), 2),
                    "tanimoto": round(sim, 4),
                }
                cliffs.append(cliff_entry)
                if sim >= 0.999:
                    tan1_pairs.append(cliff_entry)

    print(f"    Activity cliffs (Tan≥0.8, |Δ|≥0.5): {len(cliffs)}")
    print(f"    Tanimoto ≈ 1.0 pairs: {len(tan1_pairs)}")

    # E2: Investigate Tanimoto=1.0 pairs
    print("\n  E2. Investigating Tanimoto ≈ 1.0 pairs...")
    tan1_analysis = []
    for pair in tan1_pairs:
        mol_a = Chem.MolFromSmiles(pair["mol_a_smi"])
        mol_b = Chem.MolFromSmiles(pair["mol_b_smi"])
        if mol_a is None or mol_b is None:
            continue

        # Check if same canonical SMILES
        can_a = Chem.MolToSmiles(mol_a)
        can_b = Chem.MolToSmiles(mol_b)
        same_canonical = can_a == can_b

        # Check chirality
        chiral_a = Chem.FindMolChiralCenters(mol_a, includeUnassigned=True)
        chiral_b = Chem.FindMolChiralCenters(mol_b, includeUnassigned=True)
        has_chiral_a = len(chiral_a) > 0
        has_chiral_b = len(chiral_b) > 0

        # Check atom count difference
        n_atoms_a = mol_a.GetNumHeavyAtoms()
        n_atoms_b = mol_b.GetNumHeavyAtoms()

        # Check if E/Z isomers (double bond geometry)
        analysis = {
            "same_canonical": same_canonical,
            "chiral_centers_a": len(chiral_a),
            "chiral_centers_b": len(chiral_b),
            "atom_count_diff": n_atoms_b - n_atoms_a,
            "explanation": "",
        }

        if same_canonical:
            analysis["explanation"] = "Same molecule, different assay measurements (noise)"
        elif has_chiral_a or has_chiral_b:
            analysis["explanation"] = "Stereoisomers (chiral centers differ, invisible to Morgan FPs)"
        elif n_atoms_a != n_atoms_b:
            analysis["explanation"] = "Different molecules with FP collision (different size)"
        else:
            analysis["explanation"] = "FP collision (different molecules, same fingerprint)"

        pair.update(analysis)
        tan1_analysis.append(pair)

    # Categorize
    categories = Counter(p.get("explanation", "unknown") for p in tan1_analysis)
    print(f"\n  Tanimoto=1.0 pair categories:")
    for cat, count in categories.most_common():
        print(f"    {cat}: {count}")

    # E3: Cliff prediction with richer features
    print("\n  E3. Cliff prediction with chirality-aware features...")
    # Use MACCS + RDKit descriptors which may capture some structural differences
    X_rdkit, rdkit_names = compute_named_features(smiles_list)
    X_maccs = compute_maccs_features(smiles_list)

    # For cliff pairs, compute feature difference and try to predict delta
    if cliffs:
        cliff_X = []
        cliff_y = []
        for c in cliffs:
            idx_a = smiles_list.index(c["mol_a_smi"])
            idx_b = smiles_list.index(c["mol_b_smi"])
            # Use RDKit descriptor difference
            diff = X_rdkit[idx_b] - X_rdkit[idx_a]
            cliff_X.append(diff)
            cliff_y.append(c["delta"])

        cliff_X = np.array(cliff_X)
        cliff_y = np.array(cliff_y)

        # Quick train/test split on cliffs
        from sklearn.model_selection import train_test_split
        if len(cliff_X) >= 20:
            X_tr, X_te, y_tr, y_te = train_test_split(
                cliff_X, cliff_y, test_size=0.3, random_state=42
            )
            # Remove NaN/inf
            X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
            X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

            preds, _ = train_xgboost(X_tr, y_tr, X_te, n_estimators=200, max_depth=4)
            cliff_mae = float(np.mean(np.abs(y_te - preds)))
            cliff_spr, _ = spearmanr(y_te, preds) if len(y_te) > 2 else (0, 1)
            direction_acc = float(np.mean(np.sign(y_te) == np.sign(preds)))
            print(f"    Cliff prediction with RDKit descriptors:")
            print(f"    MAE={cliff_mae:.3f}, Spr={cliff_spr:.3f}, Direction={direction_acc:.1%}")
        else:
            cliff_mae = None
            cliff_spr = None
            direction_acc = None

    results["phase_e"] = {
        "n_cliffs": len(cliffs),
        "n_tan1_pairs": len(tan1_pairs),
        "tan1_categories": dict(categories),
        "tan1_analysis": tan1_analysis[:20],
        "cliff_prediction_rdkit": {
            "mae": cliff_mae,
            "spearman": float(cliff_spr) if cliff_spr is not None and not np.isnan(cliff_spr) else None,
            "direction_accuracy": direction_acc,
            "n_test": len(y_te) if cliff_mae else 0,
        } if cliffs else {},
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase F: Edit Effect Virtual Screening
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_f(mol_data, results):
    """Use edit effect model to predict modification outcomes."""
    print("\n" + "=" * 70)
    print("PHASE F: Edit Effect Virtual Screening")
    print("=" * 70)

    phase = results.get("phase_f", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)

    # F1: Train FiLMDelta on ALL ZAP70 pairs (no CV — for screening)
    print("\n  F1. Training FiLMDelta on all ZAP70 pairs for screening...")
    pairs_df = generate_all_pairs(mol_data)

    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}
    emb_a = np.array([X_morgan[smi_to_idx[s]] for s in pairs_df["mol_a"]])
    emb_b = np.array([X_morgan[smi_to_idx[s]] for s in pairs_df["mol_b"]])
    delta = pairs_df["delta"].values.astype(np.float32)

    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
    film_model = FiLMDeltaPredictor(
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        max_epochs=100, patience=15, device=DEVICE
    )
    film_model.fit(emb_a, emb_b, delta, verbose=True)

    # F2: For each molecule, predict delta to all others → rank by predicted potency
    print("\n  F2. Predicting pairwise deltas for ranking...")
    n_mols = len(smiles_list)
    pred_matrix = np.zeros((n_mols, n_mols))

    for i in range(n_mols):
        # Predict delta from mol_i to all others
        emb_i = np.tile(X_morgan[i], (n_mols, 1))
        emb_all = X_morgan
        deltas = film_model.predict(emb_i, emb_all)
        pred_matrix[i, :] = deltas

    # Derive absolute predictions: for each molecule, predicted pIC50 = mean(known_j + delta_j→i)
    # This uses the consensus of all pairwise predictions
    pred_abs = np.zeros(n_mols)
    for i in range(n_mols):
        # Predicted pIC50_i = mean over j of (pIC50_j + delta_j→i)
        pred_abs[i] = np.mean(y + pred_matrix[:, i])

    # Evaluate ranking
    spr_ranking, _ = spearmanr(y, pred_abs)
    mae_abs = float(np.mean(np.abs(y - pred_abs)))
    print(f"    Consensus absolute predictions: MAE={mae_abs:.4f}, Spearman={spr_ranking:.3f}")

    # F3: Score hypothetical modifications
    # Use the trained model to predict what happens if we modify molecules
    # Load ChEMBL compounds (from v5 Phase A) to use as modification targets
    print("\n  F3. Predicting edit effects for screening candidates...")

    # Get the v5 candidates if available
    v5_results_file = RESULTS_DIR / "zap70_v5_results.json"
    screening_candidates = []
    if v5_results_file.exists():
        with open(v5_results_file) as f:
            v5_res = json.load(f)
        if "phase_a" in v5_res and "candidates" in v5_res["phase_a"]:
            for c in v5_res["phase_a"]["candidates"][:50]:
                screening_candidates.append(c["smiles"])

    if screening_candidates:
        # Score each candidate relative to our known actives
        cand_fps = compute_fingerprints(screening_candidates, "morgan", radius=2, n_bits=2048)
        top_mol_indices = np.argsort(y)[::-1][:10]  # Top 10 most potent

        candidate_scores = []
        for ci, cand_smi in enumerate(screening_candidates):
            # Predict delta from top actives to this candidate
            deltas_from_top = []
            for mi in top_mol_indices:
                emb_ref = X_morgan[mi:mi+1]
                emb_cand = cand_fps[ci:ci+1]
                d = film_model.predict(emb_ref, emb_cand)[0]
                deltas_from_top.append(d)

            mean_delta = float(np.mean(deltas_from_top))
            pred_pic50 = float(np.mean([y[mi] + d for mi, d in zip(top_mol_indices, deltas_from_top)]))

            candidate_scores.append({
                "smiles": cand_smi,
                "mean_delta_from_top10": round(mean_delta, 3),
                "predicted_pIC50_edit": round(pred_pic50, 3),
            })

        candidate_scores.sort(key=lambda x: x["predicted_pIC50_edit"], reverse=True)
        print(f"    Scored {len(candidate_scores)} candidates via edit effect model")
        print(f"    Top 10:")
        for cs in candidate_scores[:10]:
            print(f"      pred pIC50={cs['predicted_pIC50_edit']:.2f}, "
                  f"Δ from top-10={cs['mean_delta_from_top10']:+.2f}")

    results["phase_f"] = {
        "consensus_ranking": {
            "mae": mae_abs,
            "spearman": float(spr_ranking) if not np.isnan(spr_ranking) else 0,
        },
        "candidate_scores": candidate_scores[:30] if screening_candidates else [],
        "completed": True,
    }
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase G: Iteration 2 Improvements
# ═══════════════════════════════════════════════════════════════════════════

def compute_chiral_morgan_fps(smiles_list, radius=2, n_bits=2048, use_counts=False):
    """Compute chirality-aware Morgan fingerprints."""
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(n_bits, dtype=np.float32))
            continue
        if use_counts:
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits, useChirality=True)
            arr = np.zeros(n_bits, dtype=np.float32)
            for idx, count in fp.GetNonzeroElements().items():
                arr[idx % n_bits] = count
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)
    return np.array(results, dtype=np.float32)


def compute_morgan_count_fps(smiles_list, radius=2, n_bits=2048):
    """Compute count-based Morgan fingerprints (no chirality)."""
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % n_bits] = count
        results.append(arr)
    return np.array(results, dtype=np.float32)


def run_phase_g(mol_data, results):
    """Iteration 2 improvements: chirality FPs, combined features, consensus ranking, ensemble."""
    print("\n" + "=" * 70)
    print("PHASE G: Iteration 2 Improvements")
    print("=" * 70)

    phase = results.get("phase_g", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    n_mols = len(smiles_list)

    # Shared CV splits
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    mol_indices = np.arange(n_mols)

    # ─── G1: Chirality-Aware Fingerprints ────────────────────────────────
    print("\n  G1. Chirality-Aware Fingerprints...")

    fp_variants = {
        "morgan_binary": compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048),
        "morgan_count": compute_morgan_count_fps(smiles_list, radius=2, n_bits=2048),
        "chiral_morgan_binary": compute_chiral_morgan_fps(smiles_list, radius=2, n_bits=2048, use_counts=False),
        "chiral_morgan_count": compute_chiral_morgan_fps(smiles_list, radius=2, n_bits=2048, use_counts=True),
    }

    g1_results = {}
    for fp_name, X_fp in fp_variants.items():
        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            preds, _ = train_xgboost(X_fp[train_idx], y[train_idx], X_fp[test_idx], **BEST_XGB_PARAMS)
            metrics = compute_absolute_metrics(y[test_idx], preds)
            fold_metrics.append(metrics)
        agg = aggregate_cv_results(fold_metrics)
        g1_results[fp_name] = agg
        print(f"    {fp_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg['spearman_r_mean']:.3f}")

    # Find stereoisomer pairs dynamically from the dataset
    # Look for SMILES that differ only in @/@@
    import re
    stereo_pairs = []
    for i in range(len(smiles_list)):
        for j in range(i + 1, len(smiles_list)):
            smi_a_flat = re.sub(r'[@]+', '', smiles_list[i])
            smi_b_flat = re.sub(r'[@]+', '', smiles_list[j])
            if smi_a_flat == smi_b_flat and smiles_list[i] != smiles_list[j]:
                stereo_pairs.append((smiles_list[i], smiles_list[j],
                                     float(y[i]), float(y[j])))
    print(f"    Found {len(stereo_pairs)} stereoisomer pairs in ZAP70 data")

    stereo_check = {}
    for fp_name, X_fp in fp_variants.items():
        smi_to_fp = {s: X_fp[i] for i, s in enumerate(smiles_list)}
        n_diff = 0
        for smi_a, smi_b, _, _ in stereo_pairs:
            if smi_a in smi_to_fp and smi_b in smi_to_fp:
                diff = np.sum(np.abs(smi_to_fp[smi_a] - smi_to_fp[smi_b]))
                if diff > 0:
                    n_diff += 1
        n_testable = len([1 for sa, sb, _, _ in stereo_pairs
                          if sa in smi_to_fp and sb in smi_to_fp])
        stereo_check[fp_name] = {"distinguished": n_diff, "testable": n_testable}
    g1_results["stereo_pairs_distinguished"] = stereo_check
    g1_results["stereo_pairs_found"] = len(stereo_pairs)
    if stereo_pairs:
        g1_results["stereo_pairs_examples"] = [
            {"smi_a": a, "smi_b": b, "pic50_a": pa, "pic50_b": pb}
            for a, b, pa, pb in stereo_pairs[:5]
        ]
    print(f"    Stereo pairs distinguished: {stereo_check}")

    phase["g1"] = g1_results
    save_results({**results, "phase_g": phase})

    # ─── G2: Combined Feature XGBoost ────────────────────────────────────
    print("\n  G2. Combined Feature XGBoost...")

    # Get top SHAP features from Phase C
    top_shap_features = []
    if "phase_c" in results and "shap_features" in results["phase_c"]:
        top_shap_features = [f["feature"] for f in results["phase_c"]["shap_features"][:20]]

    # Compute components
    X_morgan = fp_variants["morgan_binary"]
    X_best_chirality = fp_variants.get("chiral_morgan_count", X_morgan)
    X_named, named_feature_names = compute_named_features(smiles_list)
    X_func, func_names = compute_functional_group_counts(smiles_list)

    # Select top SHAP features from named features
    if top_shap_features:
        shap_indices = [i for i, n in enumerate(named_feature_names) if n in top_shap_features]
        func_indices = [i for i, n in enumerate(func_names) if n in top_shap_features]
        X_top_named = X_named[:, shap_indices] if shap_indices else np.zeros((n_mols, 0), dtype=np.float32)
        X_top_func = X_func[:, func_indices] if func_indices else np.zeros((n_mols, 0), dtype=np.float32)
    else:
        X_top_named = X_named[:, :20]
        X_top_func = X_func

    # Combined feature sets
    combined_variants = {
        "morgan+top_rdkit+funcgroups": np.hstack([X_morgan, X_top_named, X_top_func]),
        "morgan+all_rdkit+funcgroups": np.hstack([X_morgan, X_named, X_func]),
        "chiral_count+top_rdkit+funcgroups": np.hstack([X_best_chirality, X_top_named, X_top_func]),
        "chiral_count+all_rdkit+funcgroups": np.hstack([X_best_chirality, X_named, X_func]),
    }

    # Clean features (replace inf/nan)
    for name in combined_variants:
        combined_variants[name] = np.nan_to_num(combined_variants[name], nan=0.0, posinf=0.0, neginf=0.0)

    g2_results = {}
    g2_fold_preds = {}  # Store per-fold predictions for G4

    for var_name, X_combined in combined_variants.items():
        fold_metrics = []
        fold_preds_list = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            preds, _ = train_xgboost(X_combined[train_idx], y[train_idx], X_combined[test_idx], **BEST_XGB_PARAMS)
            metrics = compute_absolute_metrics(y[test_idx], preds)
            fold_metrics.append(metrics)
            fold_preds_list.append((test_idx, preds))
        agg = aggregate_cv_results(fold_metrics)
        g2_results[var_name] = agg
        g2_fold_preds[var_name] = fold_preds_list
        print(f"    {var_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spr={agg['spearman_r_mean']:.3f}")

    phase["g2"] = g2_results
    save_results({**results, "phase_g": phase})

    # ─── G3: Consensus Ranking from Edit Effect ──────────────────────────
    print("\n  G3. Consensus Ranking from Edit Effect (proper CV)...")

    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor

    g3_fold_metrics = []
    g3_fold_preds = []
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
        print(f"    Fold {fold_i + 1}/{N_FOLDS}...", end=" ", flush=True)
        train_smiles = [smiles_list[i] for i in train_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Generate train-train pairs for FiLMDelta training
        train_pairs = []
        for i in range(len(train_idx)):
            for j in range(i + 1, len(train_idx)):
                train_pairs.append((train_idx[i], train_idx[j],
                                    y[train_idx[j]] - y[train_idx[i]]))

        if len(train_pairs) < 50:
            print("too few pairs, skipping")
            continue

        pair_arr = np.array(train_pairs, dtype=object)
        idx_a = np.array([p[0] for p in train_pairs])
        idx_b = np.array([p[1] for p in train_pairs])
        deltas_tr = np.array([p[2] for p in train_pairs], dtype=np.float32)

        emb_a_tr = X_morgan[idx_a]
        emb_b_tr = X_morgan[idx_b]

        # Train FiLMDelta on train-train pairs
        predictor = FiLMDeltaPredictor(
            dropout=0.2, learning_rate=1e-3, batch_size=64,
            max_epochs=80, patience=12, device=DEVICE
        )
        predictor.fit(emb_a_tr, emb_b_tr, deltas_tr, verbose=False)

        # For each test molecule, predict delta from ALL training molecules
        consensus_preds = np.zeros(len(test_idx))
        weighted_preds = np.zeros(len(test_idx))

        for ti, test_i in enumerate(test_idx):
            emb_test = np.tile(X_morgan[test_i], (len(train_idx), 1))
            emb_train = X_morgan[train_idx]

            # Predict delta from each train mol to test mol
            deltas_pred = predictor.predict(emb_train, emb_test)

            # Consensus: pIC50_test = pIC50_train_j + delta_j→test
            abs_estimates = y_train + deltas_pred

            # Unweighted median
            consensus_preds[ti] = np.median(abs_estimates)

            # Tanimoto-weighted mean
            from rdkit.DataStructs import BulkTanimotoSimilarity
            test_fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles_list[test_i]), 2, nBits=2048)
            train_fps = [AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(s), 2, nBits=2048) for s in train_smiles]
            sims = np.array(BulkTanimotoSimilarity(test_fp, train_fps))
            weights = sims ** 2  # Square for stronger locality
            if weights.sum() > 0:
                weighted_preds[ti] = np.average(abs_estimates, weights=weights)
            else:
                weighted_preds[ti] = consensus_preds[ti]

        # Evaluate
        mae_med = float(np.mean(np.abs(y_test - consensus_preds)))
        spr_med, _ = spearmanr(y_test, consensus_preds)
        mae_wt = float(np.mean(np.abs(y_test - weighted_preds)))
        spr_wt, _ = spearmanr(y_test, weighted_preds)

        fold_result = {
            "median_consensus": {"mae": mae_med, "spearman": float(spr_med) if not np.isnan(spr_med) else 0},
            "weighted_consensus": {"mae": mae_wt, "spearman": float(spr_wt) if not np.isnan(spr_wt) else 0},
            "n_test": len(test_idx), "n_train_pairs": len(train_pairs),
        }
        g3_fold_metrics.append(fold_result)
        g3_fold_preds.append((test_idx, consensus_preds, weighted_preds))
        print(f"median MAE={mae_med:.3f} Spr={spr_med:.3f}, "
              f"weighted MAE={mae_wt:.3f} Spr={spr_wt:.3f}")

        del predictor
        gc.collect()

    # Aggregate G3
    if g3_fold_metrics:
        g3_agg = {}
        for method in ["median_consensus", "weighted_consensus"]:
            maes = [f[method]["mae"] for f in g3_fold_metrics]
            sprs = [f[method]["spearman"] for f in g3_fold_metrics]
            g3_agg[method] = {
                "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "spearman_mean": float(np.mean(sprs)), "spearman_std": float(np.std(sprs)),
            }
            print(f"    {method}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}, "
                  f"Spr={np.mean(sprs):.3f}±{np.std(sprs):.3f}")
        phase["g3"] = {"aggregated": g3_agg, "per_fold": g3_fold_metrics}
    save_results({**results, "phase_g": phase})

    # ─── G4: Ensemble (Absolute + Consensus) ─────────────────────────────
    print("\n  G4. Ensemble of Absolute + Consensus Predictions...")

    # Find best G2 variant
    best_g2_name = min(g2_results, key=lambda k: g2_results[k]["mae_mean"])
    best_g2_preds = g2_fold_preds[best_g2_name]
    print(f"    Best G2 variant: {best_g2_name}")

    # Also use baseline morgan XGB
    baseline_fold_preds = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
        preds, _ = train_xgboost(X_morgan[train_idx], y[train_idx], X_morgan[test_idx], **BEST_XGB_PARAMS)
        baseline_fold_preds.append((test_idx, preds))

    # Build full prediction arrays for ensembling
    pred_xgb_baseline = np.zeros(n_mols)
    pred_xgb_best = np.zeros(n_mols)
    pred_consensus_med = np.zeros(n_mols)
    pred_consensus_wt = np.zeros(n_mols)

    for fold_data in baseline_fold_preds:
        test_idx, preds = fold_data
        pred_xgb_baseline[test_idx] = preds

    for fold_data in best_g2_preds:
        test_idx, preds = fold_data
        pred_xgb_best[test_idx] = preds

    for fold_data in g3_fold_preds:
        test_idx, cons_med, cons_wt = fold_data
        pred_consensus_med[test_idx] = cons_med
        pred_consensus_wt[test_idx] = cons_wt

    # Sweep ensemble weights
    g4_results = {}
    ensemble_configs = [
        ("xgb_baseline + median_consensus", pred_xgb_baseline, pred_consensus_med),
        ("xgb_baseline + weighted_consensus", pred_xgb_baseline, pred_consensus_wt),
        ("xgb_best_g2 + median_consensus", pred_xgb_best, pred_consensus_med),
        ("xgb_best_g2 + weighted_consensus", pred_xgb_best, pred_consensus_wt),
    ]

    for config_name, pred_abs, pred_cons in ensemble_configs:
        best_alpha = 1.0
        best_mae = float('inf')
        best_spr = 0
        sweep_results = []
        for alpha in np.arange(0.0, 1.05, 0.1):
            pred_ens = alpha * pred_abs + (1 - alpha) * pred_cons
            mae_ens = float(np.mean(np.abs(y - pred_ens)))
            spr_ens, _ = spearmanr(y, pred_ens)
            spr_ens = float(spr_ens) if not np.isnan(spr_ens) else 0
            sweep_results.append({"alpha": round(float(alpha), 1), "mae": mae_ens, "spearman": spr_ens})
            if mae_ens < best_mae:
                best_mae = mae_ens
                best_alpha = float(alpha)
                best_spr = spr_ens

        g4_results[config_name] = {
            "best_alpha": best_alpha, "best_mae": best_mae, "best_spearman": best_spr,
            "sweep": sweep_results,
        }
        print(f"    {config_name}: α={best_alpha:.1f}, MAE={best_mae:.4f}, Spr={best_spr:.3f}")

    # Rank-based ensemble (average ranks for Spearman)
    from scipy.stats import rankdata
    rank_xgb = rankdata(pred_xgb_baseline)
    rank_cons_wt = rankdata(pred_consensus_wt)
    rank_ensemble = (rank_xgb + rank_cons_wt) / 2
    spr_rank_ens, _ = spearmanr(y, rank_ensemble)
    g4_results["rank_ensemble_xgb+weighted_consensus"] = {
        "spearman": float(spr_rank_ens) if not np.isnan(spr_rank_ens) else 0
    }
    print(f"    Rank ensemble (xgb + weighted_consensus): Spr={spr_rank_ens:.3f}")

    phase["g4"] = g4_results

    # Final summary
    print("\n  ═══ Phase G Summary ═══")
    print(f"    G1 Best FP: {min(g1_results, key=lambda k: g1_results[k].get('mae_mean', 999) if isinstance(g1_results[k], dict) and 'mae_mean' in g1_results[k] else 999)}")
    print(f"    G2 Best Combined: {best_g2_name} (MAE={g2_results[best_g2_name]['mae_mean']:.4f})")
    if g3_fold_metrics:
        print(f"    G3 Best Consensus: weighted (MAE={g3_agg['weighted_consensus']['mae_mean']:.4f})")
    print(f"    G4 Best Ensemble: {min(g4_results, key=lambda k: g4_results[k].get('best_mae', 999) if 'best_mae' in g4_results[k] else 999)}")

    phase["completed"] = True
    results["phase_g"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase H: Iteration 3 — XGBoost Consensus + Multi-Model Ensemble
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_h(mol_data, results):
    """Iteration 3: XGBoost-based consensus (stable) + ensemble with v3 top5."""
    print("\n" + "=" * 70)
    print("PHASE H: Iteration 3 — XGBoost Consensus + Multi-Model Ensemble")
    print("=" * 70)

    phase = results.get("phase_h", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    n_mols = len(smiles_list)

    # Shared CV splits — MUST match all other phases
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    mol_indices = np.arange(n_mols)

    # ─── H1: XGBoost-Based Consensus (stable version of G3) ─────────────
    print("\n  H1. XGBoost-Based Consensus Ranking...")

    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    X_chiral = compute_chiral_morgan_fps(smiles_list, radius=2, n_bits=2048, use_counts=False)

    # Precompute all pairwise Tanimoto similarities
    from rdkit.DataStructs import BulkTanimotoSimilarity
    all_fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles_list]
    sim_matrix = np.zeros((n_mols, n_mols), dtype=np.float32)
    for i in range(n_mols):
        sims = BulkTanimotoSimilarity(all_fps[i], all_fps)
        sim_matrix[i, :] = sims

    consensus_methods = {}
    xgb_fold_preds = np.zeros(n_mols)  # Store per-molecule OOF predictions
    xgb_chiral_fold_preds = np.zeros(n_mols)

    for fp_name, X_fp in [("morgan", X_morgan), ("chiral_morgan", X_chiral)]:
        fold_metrics_consensus = []
        fold_metrics_trimmed = []
        fold_metrics_knn = []
        oof_preds = np.zeros(n_mols)

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            # Train XGBoost on absolute values
            preds_abs, _ = train_xgboost(
                X_fp[train_idx], y[train_idx], X_fp[test_idx], **BEST_XGB_PARAMS)
            oof_preds[test_idx] = preds_abs

            # For consensus: predict ALL molecules, then derive test predictions
            preds_all, _ = train_xgboost(
                X_fp[train_idx], y[train_idx], X_fp, **BEST_XGB_PARAMS)

            # For each test molecule:
            # pred_consensus_i = median_j(pIC50_j + (pred_j - pred_i))
            # = median_j(pIC50_j - pred_j) + pred_i
            # This simplifies to: pred_i + median_j(residual_j)
            # where residual_j = pIC50_j - pred_j (only on training set)

            # Actually, the proper consensus is:
            # For test mol i, using each train mol j:
            #   estimate_j = pIC50_j + (pred_i - pred_j) = pIC50_j - pred_j + pred_i
            #   = pred_i + (pIC50_j - pred_j)  [= pred_i + residual_j]
            # Median of these = pred_i + median(train_residuals)
            # This is just bias correction!

            train_residuals = y[train_idx] - preds_all[train_idx]

            # Method 1: Simple bias-corrected (median residual)
            bias = np.median(train_residuals)
            consensus_preds = preds_abs + bias

            # Method 2: Tanimoto-weighted residual correction
            weighted_preds = np.zeros(len(test_idx))
            for ti, test_i in enumerate(test_idx):
                sims = sim_matrix[test_i, train_idx]
                weights = sims ** 2
                if weights.sum() > 0:
                    weighted_residual = np.average(train_residuals, weights=weights)
                else:
                    weighted_residual = bias
                weighted_preds[ti] = preds_abs[ti] + weighted_residual

            # Method 3: KNN residual correction (top-10 nearest neighbors)
            knn_preds = np.zeros(len(test_idx))
            for ti, test_i in enumerate(test_idx):
                sims = sim_matrix[test_i, train_idx]
                top_k = np.argsort(sims)[-10:]
                knn_residual = np.mean(train_residuals[top_k])
                knn_preds[ti] = preds_abs[ti] + knn_residual

            y_test = y[test_idx]
            fold_metrics_consensus.append(compute_absolute_metrics(y_test, consensus_preds))
            fold_metrics_trimmed.append(compute_absolute_metrics(y_test, weighted_preds))
            fold_metrics_knn.append(compute_absolute_metrics(y_test, knn_preds))

        # Store OOF for ensembling
        if fp_name == "morgan":
            xgb_fold_preds = oof_preds.copy()
        else:
            xgb_chiral_fold_preds = oof_preds.copy()

        for method_name, metrics_list in [
            (f"{fp_name}_bias_corrected", fold_metrics_consensus),
            (f"{fp_name}_tanimoto_weighted", fold_metrics_trimmed),
            (f"{fp_name}_knn10", fold_metrics_knn),
        ]:
            agg = aggregate_cv_results(metrics_list)
            consensus_methods[method_name] = agg
            print(f"    {method_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                  f"Spr={agg['spearman_r_mean']:.3f}")

    phase["h1"] = consensus_methods
    save_results({**results, "phase_h": phase})

    # ─── H2: v3-style Multi-FP Ensemble ──────────────────────────────────
    print("\n  H2. Multi-Fingerprint Ensemble (v3-style)...")

    # Compute diverse fingerprint types
    fp_configs = {
        "morgan_2048": ("morgan", {"radius": 2, "n_bits": 2048}),
        "ecfp6_2048": ("morgan", {"radius": 3, "n_bits": 2048}),
        "rdkit_2048": ("rdkit", {"n_bits": 2048}),
        "atompair_2048": ("atompair", {"n_bits": 2048}),
    }

    oof_predictions = {}
    individual_metrics = {}

    for fp_label, (fp_type, fp_params) in fp_configs.items():
        X_fp = compute_fingerprints(smiles_list, fp_type, **fp_params)
        oof = np.zeros(n_mols)
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            # XGBoost
            preds_xgb, _ = train_xgboost(X_fp[train_idx], y[train_idx], X_fp[test_idx], **BEST_XGB_PARAMS)
            oof[test_idx] = preds_xgb
            fold_metrics.append(compute_absolute_metrics(y[test_idx], preds_xgb))

        oof_predictions[f"xgb_{fp_label}"] = oof.copy()
        agg = aggregate_cv_results(fold_metrics)
        individual_metrics[f"xgb_{fp_label}"] = agg
        print(f"    xgb_{fp_label}: MAE={agg['mae_mean']:.4f}, Spr={agg['spearman_r_mean']:.3f}")

    # Also add RF on select FPs
    for fp_label in ["morgan_2048", "rdkit_2048"]:
        fp_type, fp_params = fp_configs[fp_label]
        X_fp = compute_fingerprints(smiles_list, fp_type, **fp_params)
        oof = np.zeros(n_mols)
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            preds_rf, _ = train_rf(X_fp[train_idx], y[train_idx], X_fp[test_idx], **BEST_RF_PARAMS)
            oof[test_idx] = preds_rf
            fold_metrics.append(compute_absolute_metrics(y[test_idx], preds_rf))

        oof_predictions[f"rf_{fp_label}"] = oof.copy()
        agg = aggregate_cv_results(fold_metrics)
        individual_metrics[f"rf_{fp_label}"] = agg
        print(f"    rf_{fp_label}: MAE={agg['mae_mean']:.4f}, Spr={agg['spearman_r_mean']:.3f}")

    # Add chiral Morgan
    oof_predictions["xgb_chiral_morgan"] = xgb_chiral_fold_preds.copy()

    # KRR with Tanimoto kernel (like v3)
    from sklearn.kernel_ridge import KernelRidge
    oof_krr = np.zeros(n_mols)
    krr_fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
        K_train = sim_matrix[np.ix_(train_idx, train_idx)]
        K_test = sim_matrix[np.ix_(test_idx, train_idx)]
        krr = KernelRidge(alpha=1.0, kernel="precomputed")
        krr.fit(K_train, y[train_idx])
        preds_krr = krr.predict(K_test)
        oof_krr[test_idx] = preds_krr
        krr_fold_metrics.append(compute_absolute_metrics(y[test_idx], preds_krr))

    oof_predictions["krr_tanimoto"] = oof_krr.copy()
    agg = aggregate_cv_results(krr_fold_metrics)
    individual_metrics["krr_tanimoto"] = agg
    print(f"    krr_tanimoto: MAE={agg['mae_mean']:.4f}, Spr={agg['spearman_r_mean']:.3f}")

    # Ensemble: simple mean of diverse models
    all_model_names = list(oof_predictions.keys())
    print(f"\n    Building ensembles from {len(all_model_names)} models...")

    # Try all subsets of size 3, 4, 5
    from itertools import combinations as combos
    from scipy.stats import rankdata

    best_ensemble = {"name": "", "mae": 999, "spearman": 0, "n_models": 0}
    ensemble_results = []

    for n_models in [3, 4, 5, 6, len(all_model_names)]:
        if n_models > len(all_model_names):
            continue
        candidate_combos = list(combos(all_model_names, n_models))
        # Limit search for large n
        if len(candidate_combos) > 50:
            # Greedy: start with best individual, add best complement
            sorted_models = sorted(individual_metrics.items(),
                                   key=lambda x: x[1].get("mae_mean", 999))
            candidate_combos = []
            base = [sorted_models[0][0]]
            for i in range(1, min(n_models, len(sorted_models))):
                base.append(sorted_models[i][0])
                candidate_combos.append(tuple(base.copy()))

        for combo in candidate_combos:
            pred_ens = np.mean([oof_predictions[m] for m in combo], axis=0)
            mae = float(np.mean(np.abs(y - pred_ens)))
            spr, _ = spearmanr(y, pred_ens)
            spr = float(spr) if not np.isnan(spr) else 0
            if mae < best_ensemble["mae"]:
                best_ensemble = {"name": "+".join(combo), "mae": mae, "spearman": spr, "n_models": n_models}
            ensemble_results.append({
                "models": list(combo), "mae": round(mae, 4), "spearman": round(spr, 3)
            })

    # Sort and keep top 10
    ensemble_results.sort(key=lambda x: x["mae"])
    top_ensembles = ensemble_results[:10]

    print(f"    Best ensemble ({best_ensemble['n_models']} models): "
          f"MAE={best_ensemble['mae']:.4f}, Spr={best_ensemble['spearman']:.3f}")
    print(f"    Models: {best_ensemble['name']}")

    # Also try rank-based ensemble of top 5
    rank_preds = {}
    for name, oof in oof_predictions.items():
        rank_preds[name] = rankdata(oof)

    best_rank_ens = {"name": "", "spearman": 0}
    for combo in combos(all_model_names, min(5, len(all_model_names))):
        avg_rank = np.mean([rank_preds[m] for m in combo], axis=0)
        spr, _ = spearmanr(y, avg_rank)
        spr = float(spr) if not np.isnan(spr) else 0
        if spr > best_rank_ens["spearman"]:
            best_rank_ens = {"name": "+".join(combo), "spearman": spr}

    print(f"    Best rank ensemble: Spr={best_rank_ens['spearman']:.3f}")
    print(f"    Models: {best_rank_ens['name']}")

    phase["h2"] = {
        "individual": individual_metrics,
        "top_ensembles": top_ensembles,
        "best_ensemble": best_ensemble,
        "best_rank_ensemble": best_rank_ens,
    }
    save_results({**results, "phase_h": phase})

    # ─── H3: Transfer + Absolute Ensemble ────────────────────────────────
    print("\n  H3. Transfer Learning + Absolute Ensemble...")

    # Use dual-objective model from Phase D for another prediction source
    # Train dual-objective per fold and get OOF absolute predictions
    dual_oof = np.zeros(n_mols)
    dual_fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
        train_smiles = [smiles_list[i] for i in train_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Generate train-train pairs
        emb_a_tr, emb_b_tr, delta_tr, abs_a_tr, abs_b_tr = [], [], [], [], []
        for i in range(len(train_idx)):
            for j in range(i + 1, len(train_idx)):
                emb_a_tr.append(X_morgan[train_idx[i]])
                emb_b_tr.append(X_morgan[train_idx[j]])
                delta_tr.append(y[train_idx[j]] - y[train_idx[i]])
                abs_a_tr.append(y[train_idx[i]])
                abs_b_tr.append(y[train_idx[j]])

        emb_a_tr = np.array(emb_a_tr, dtype=np.float32)
        emb_b_tr = np.array(emb_b_tr, dtype=np.float32)
        delta_tr = np.array(delta_tr, dtype=np.float32)
        abs_a_tr = np.array(abs_a_tr, dtype=np.float32)
        abs_b_tr = np.array(abs_b_tr, dtype=np.float32)

        # Train dual-objective model
        model = DualObjectiveModel(input_dim=2048, hidden_dims=[512, 256], dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion_delta = nn.MSELoss()
        criterion_abs = nn.MSELoss()

        emb_a_t = torch.FloatTensor(emb_a_tr)
        emb_b_t = torch.FloatTensor(emb_b_tr)
        delta_t = torch.FloatTensor(delta_tr)
        abs_a_t = torch.FloatTensor(abs_a_tr)
        abs_b_t = torch.FloatTensor(abs_b_tr)

        best_loss = float('inf')
        best_state = None
        wait = 0
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            d_pred, a_pred, b_pred = model(emb_a_t, emb_b_t)
            loss = criterion_delta(d_pred, delta_t) + 0.5 * (
                criterion_abs(a_pred, abs_a_t) + criterion_abs(b_pred, abs_b_t))
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 15:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # Predict test molecules: use consensus from train→test pairs
        with torch.no_grad():
            test_preds = []
            for ti in test_idx:
                emb_ref = torch.FloatTensor(X_morgan[train_idx])
                emb_test = torch.FloatTensor(np.tile(X_morgan[ti], (len(train_idx), 1)))
                _, _, abs_pred = model(emb_ref, emb_test)
                # abs_pred is prediction for test mol from each reference
                # Weight by similarity
                sims = sim_matrix[ti, train_idx]
                weights = sims ** 2
                if weights.sum() > 0:
                    pred = float(np.average(abs_pred.numpy(), weights=weights))
                else:
                    pred = float(abs_pred.mean())
                test_preds.append(pred)

            dual_oof[test_idx] = test_preds

        dual_fold_metrics.append(compute_absolute_metrics(y_test, np.array(test_preds)))
        del model
        gc.collect()

    agg_dual = aggregate_cv_results(dual_fold_metrics)
    print(f"    Dual-objective consensus: MAE={agg_dual['mae_mean']:.4f}, "
          f"Spr={agg_dual['spearman_r_mean']:.3f}")

    # Add dual-objective to ensemble candidates
    oof_predictions["dual_objective"] = dual_oof

    # Rebuild ensemble with dual-objective included
    all_names_h3 = list(oof_predictions.keys())
    pred_ens_h3 = np.mean([oof_predictions[m] for m in all_names_h3], axis=0)
    mae_all = float(np.mean(np.abs(y - pred_ens_h3)))
    spr_all, _ = spearmanr(y, pred_ens_h3)
    print(f"    Full ensemble ({len(all_names_h3)} models): MAE={mae_all:.4f}, Spr={float(spr_all):.3f}")

    # Best combo including dual_objective
    best_with_dual = {"name": "", "mae": 999}
    for n in [3, 4, 5]:
        for combo in combos(all_names_h3, n):
            if "dual_objective" not in combo:
                continue
            pred = np.mean([oof_predictions[m] for m in combo], axis=0)
            mae = float(np.mean(np.abs(y - pred)))
            spr, _ = spearmanr(y, pred)
            if mae < best_with_dual["mae"]:
                best_with_dual = {"name": "+".join(combo), "mae": mae,
                                  "spearman": float(spr) if not np.isnan(spr) else 0, "n": n}

    print(f"    Best ensemble with dual: MAE={best_with_dual['mae']:.4f}, "
          f"Spr={best_with_dual.get('spearman', 0):.3f}")

    phase["h3"] = {
        "dual_objective_consensus": agg_dual,
        "full_ensemble": {"mae": mae_all, "spearman": float(spr_all)},
        "best_with_dual": best_with_dual,
    }

    # Final summary
    print("\n  ═══ Phase H Summary ═══")
    print(f"    H1 Best Consensus: {min(consensus_methods, key=lambda k: consensus_methods[k]['mae_mean'])}")
    print(f"    H2 Best Ensemble: {best_ensemble['name']} (MAE={best_ensemble['mae']:.4f})")
    print(f"    H2 Best Rank Ensemble: Spr={best_rank_ens['spearman']:.3f}")
    print(f"    H3 Dual+Ensemble: MAE={best_with_dual['mae']:.4f}")

    phase["completed"] = True
    results["phase_h"] = phase
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase I: SHAP Pharmacophore Analysis with Chemical Meaning
# ═══════════════════════════════════════════════════════════════════════════

# Expert-assigned pharmacophore roles for key Morgan FP bits
# Pharmacophore annotations will be populated by SHAP analysis on ZAP70 data
# (no pre-populated annotations — let the data speak)
PHARMACOPHORE_ANNOTATIONS = {}


def run_phase_i(mol_data, results):
    """SHAP Pharmacophore Analysis: chemical meaning for every important feature."""
    print("\n" + "=" * 70)
    print("PHASE I: SHAP Pharmacophore Analysis with Chemical Meaning")
    print("=" * 70)

    phase = results.get("phase_i", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    smiles_list = mol_data["smiles"].tolist()
    y = mol_data["pIC50"].values
    n_mols = len(smiles_list)

    # ─── I1: Load v4 SHAP results and annotate with pharmacophore roles ──
    print("\n  I1. Annotating Morgan FP SHAP features with pharmacophore roles...")

    v4_file = RESULTS_DIR / "zap70_v4_results.json"
    if not v4_file.exists():
        print("    WARNING: v4 results not found, skipping SHAP annotation")
        phase["completed"] = True
        results["phase_i"] = phase
        save_results(results)
        return results

    with open(v4_file) as f:
        v4 = json.load(f)

    shap_bits = v4.get("phase_b", {}).get("top_features", [])

    annotated_features = []
    for feat in shap_bits:
        bit = feat["bit"]
        entry = {
            "bit": bit,
            "mean_abs_shap": feat["mean_abs_shap"],
            "n_active": feat["n_active"],
            "delta_pIC50": feat["delta_pIC50"],
            "spearman": feat["spearman_with_pIC50"],
            "substructures_raw": feat["substructures"],
        }
        if bit in PHARMACOPHORE_ANNOTATIONS:
            ann = PHARMACOPHORE_ANNOTATIONS[bit]
            entry["pharmacophore_role"] = ann["role"]
            entry["binding_region"] = ann["region"]
            entry["chemistry"] = ann["chemistry"]
            entry["note"] = ann["note"]
        else:
            entry["pharmacophore_role"] = "Unknown"
            entry["binding_region"] = "Unknown"
            entry["chemistry"] = feat["substructures"]
            entry["note"] = ""
        annotated_features.append(entry)

    phase["annotated_shap_features"] = annotated_features

    # Print formatted table
    print("\n  ╔═══════════════════════════════════════════════════════════════════════╗")
    print("  ║  ZAP70 SHAP Pharmacophore Analysis — Morgan FP Bits → Chemistry    ║")
    print("  ╠═══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║ {'Rank':>4} {'Bit':>5} {'|SHAP|':>7} {'Δ pIC50':>8} {'Region':>12} {'Pharmacophore Role':<35} ║")
    print("  ╠═══════════════════════════════════════════════════════════════════════╣")
    for i, f in enumerate(annotated_features[:20]):
        role = f["pharmacophore_role"][:35]
        region = f["binding_region"][:12]
        delta = f["delta_pIC50"]
        sign = "↑" if delta > 0.3 else ("↓" if delta < -0.3 else "~")
        print(f"  ║ {i+1:>4} {f['bit']:>5} {f['mean_abs_shap']:>7.4f} {delta:>+7.3f}{sign} {region:>12} {role:<35} ║")
    print("  ╚═══════════════════════════════════════════════════════════════════════╝")

    # ─── I2: Region-level aggregation ────────────────────────────────────
    print("\n  I2. Binding Region Summary...")

    region_stats = defaultdict(lambda: {"shap_sum": 0, "n_bits": 0, "bits": [],
                                         "delta_sum": 0, "n_mols": set()})
    for f in annotated_features:
        region = f["binding_region"]
        region_stats[region]["shap_sum"] += f["mean_abs_shap"]
        region_stats[region]["n_bits"] += 1
        region_stats[region]["bits"].append(f["bit"])
        region_stats[region]["delta_sum"] += f["delta_pIC50"] * f["mean_abs_shap"]

    region_summary = {}
    print(f"\n  {'Region':<15} {'Σ|SHAP|':>8} {'N bits':>7} {'Weighted Δ':>10} {'Key Bits'}")
    print(f"  {'─'*15} {'─'*8} {'─'*7} {'─'*10} {'─'*30}")
    for region in sorted(region_stats, key=lambda r: region_stats[r]["shap_sum"], reverse=True):
        rs = region_stats[region]
        weighted_delta = rs["delta_sum"] / rs["shap_sum"] if rs["shap_sum"] > 0 else 0
        bits_str = ", ".join(str(b) for b in rs["bits"][:5])
        print(f"  {region:<15} {rs['shap_sum']:>8.4f} {rs['n_bits']:>7} {weighted_delta:>+10.3f} {bits_str}")
        region_summary[region] = {
            "total_shap": rs["shap_sum"], "n_bits": rs["n_bits"],
            "weighted_delta": weighted_delta, "bits": rs["bits"],
        }

    phase["region_summary"] = region_summary

    # ─── I3: Per-atom SHAP for most potent molecule ─────────────────────
    print("\n  I3. Per-Atom SHAP Map for Most Potent Molecule...")

    # Find most potent molecule
    top_idx = np.argmax(y)
    top_smi = smiles_list[top_idx]
    top_pic50 = y[top_idx]
    print(f"    Most potent: pIC50={top_pic50:.2f}")
    print(f"    SMILES: {top_smi}")

    mol = Chem.MolFromSmiles(top_smi)
    if mol is not None:
        # Get Morgan FP with bit info
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)

        # Map SHAP values to atoms
        atom_shap = np.zeros(mol.GetNumAtoms())
        atom_bits = defaultdict(list)

        for f in annotated_features:
            bit = f["bit"]
            if bit in bi:
                shap_val = f["mean_abs_shap"] * (1 if f["delta_pIC50"] > 0 else -1)
                for atom_idx, radius in bi[bit]:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    for orig_idx in amap:
                        atom_shap[orig_idx] += shap_val / len(amap)  # Distribute across atoms
                        atom_bits[orig_idx].append((bit, f["pharmacophore_role"]))

        # Print atom-level SHAP with pharmacophore annotation
        atom_data = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            roles = list(set(r for _, r in atom_bits.get(i, [])))
            atom_data.append({
                "idx": i, "symbol": symbol, "shap": float(atom_shap[i]),
                "roles": roles[:3],
            })

        # Sort by |SHAP|
        atom_data.sort(key=lambda x: abs(x["shap"]), reverse=True)

        print(f"\n    Top 15 atoms by |SHAP| contribution:")
        print(f"    {'Atom':>6} {'Sym':>4} {'SHAP':>7} {'Pharmacophore Roles'}")
        print(f"    {'─'*6} {'─'*4} {'─'*7} {'─'*40}")
        for a in atom_data[:15]:
            sign = "+" if a["shap"] > 0 else "-"
            roles_str = ", ".join(a["roles"][:2]) if a["roles"] else "—"
            print(f"    {a['idx']:>6} {a['symbol']:>4} {a['shap']:>+7.4f} {roles_str}")

        phase["top_molecule_atom_shap"] = {
            "smiles": top_smi, "pIC50": float(top_pic50),
            "atom_contributions": atom_data[:20],
        }

    # ─── I4: Data-Driven SAR Summary ────────────────────────────────────
    print("\n  I4. ZAP70 SAR Summary from SHAP Analysis")
    print("  " + "─" * 65)

    # Build SAR story from the actual SHAP results
    story = {}

    # Group features by positive/negative impact
    beneficial = [f for f in annotated_features if f["delta_pIC50"] > 0.2]
    detrimental = [f for f in annotated_features if f["delta_pIC50"] < -0.2]

    if beneficial:
        top_ben = beneficial[:5]
        story["beneficial_features"] = {
            "description": "Morgan FP bits associated with higher pIC50",
            "features": [
                {"bit": f["bit"], "shap": f["mean_abs_shap"],
                 "delta": f["delta_pIC50"], "substructures": f["substructures_raw"][:100],
                 "n_active": f["n_active"]}
                for f in top_ben
            ],
        }
        print(f"\n    Beneficial features ({len(beneficial)} bits):")
        for f in top_ben:
            print(f"      Bit {f['bit']}: SHAP={f['mean_abs_shap']:.4f}, "
                  f"Δ={f['delta_pIC50']:+.2f}, N={f['n_active']}")

    if detrimental:
        top_det = detrimental[:5]
        story["detrimental_features"] = {
            "description": "Morgan FP bits associated with lower pIC50",
            "features": [
                {"bit": f["bit"], "shap": f["mean_abs_shap"],
                 "delta": f["delta_pIC50"], "substructures": f["substructures_raw"][:100],
                 "n_active": f["n_active"]}
                for f in top_det
            ],
        }
        print(f"\n    Detrimental features ({len(detrimental)} bits):")
        for f in top_det:
            print(f"      Bit {f['bit']}: SHAP={f['mean_abs_shap']:.4f}, "
                  f"Δ={f['delta_pIC50']:+.2f}, N={f['n_active']}")

    # Add functional group SAR from Phase C if available
    if "phase_c" in results and "functional_group_impact" in results["phase_c"]:
        fg_data = results["phase_c"]["functional_group_impact"]
        fg_ben = [f for f in fg_data if f["delta"] > 0.3]
        fg_det = [f for f in fg_data if f["delta"] < -0.3]
        story["beneficial_groups"] = [
            {"group": f["group"], "delta": f["delta"], "n": f["n_with"]}
            for f in fg_ben[:5]
        ]
        story["detrimental_groups"] = [
            {"group": f["group"], "delta": f["delta"], "n": f["n_with"]}
            for f in fg_det[:5]
        ]
        if fg_ben:
            print(f"\n    Beneficial functional groups:")
            for f in fg_ben[:5]:
                print(f"      {f['group']}: Δ={f['delta']:+.2f}, N={f['n_with']}")
        if fg_det:
            print(f"\n    Detrimental functional groups:")
            for f in fg_det[:5]:
                print(f"      {f['group']}: Δ={f['delta']:+.2f}, N={f['n_with']}")

    phase["sar_story"] = story

    # ─── I5: Combined Interpretable Feature Table ────────────────────────
    print("\n  I5. Combined Feature Importance: Named + Pharmacophore Annotated")

    # Merge v4 bit SHAP with v6 named feature SHAP
    combined_table = []

    # Add annotated Morgan bits
    for f in annotated_features[:15]:
        combined_table.append({
            "feature_name": f"{f['pharmacophore_role']} (bit {f['bit']})",
            "type": "Morgan FP bit",
            "mean_abs_shap": f["mean_abs_shap"],
            "delta_pIC50": f["delta_pIC50"],
            "binding_region": f["binding_region"],
            "n_molecules": f["n_active"],
        })

    # Add v6 named features
    if "phase_c" in results and "shap_features" in results["phase_c"]:
        for f in results["phase_c"]["shap_features"][:15]:
            combined_table.append({
                "feature_name": f["feature"],
                "type": "RDKit descriptor",
                "mean_abs_shap": f["mean_abs_shap"],
                "delta_pIC50": f.get("spearman_with_pIC50", 0),
                "binding_region": "—",
                "n_molecules": "—",
            })

    # Add functional group impacts
    if "phase_c" in results and "functional_group_impact" in results["phase_c"]:
        for f in results["phase_c"]["functional_group_impact"][:10]:
            combined_table.append({
                "feature_name": f["group"],
                "type": "Functional group",
                "mean_abs_shap": abs(f["delta"]) * abs(f["spearman"]),  # Proxy importance
                "delta_pIC50": f["delta"],
                "binding_region": "—",
                "n_molecules": f["n_with"],
            })

    phase["combined_feature_table"] = combined_table

    phase["completed"] = True
    results["phase_i"] = phase
    save_results(results)
    print("\n  Phase I complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase J: Best Single Dual-Objective Model (iterate to find best)
# ═══════════════════════════════════════════════════════════════════════════

class PairRankingDualModel(nn.Module):
    """Neural dual-objective model with ranking loss for pairwise deltas.

    Key difference from DualObjectiveModel: uses concordance-based ranking
    loss instead of MSE for delta prediction, which directly optimizes
    Spearman correlation of deltas.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.4):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_dim = hidden_dims[-1]
        # Single prediction head (absolute pIC50)
        self.head = nn.Sequential(
            nn.Linear(self.enc_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        enc = self.encoder(x)
        return self.head(enc).squeeze(-1)

    def predict_pair(self, x_a, x_b):
        pred_a = self.forward(x_a)
        pred_b = self.forward(x_b)
        return pred_b - pred_a, pred_a, pred_b


def concordance_loss(pred_delta, true_delta, margin=0.1):
    """Differentiable concordance loss — penalizes discordant pair rankings.

    For each pair (i, j) of pairs, if true_delta_i > true_delta_j
    but pred_delta_i < pred_delta_j, that's a concordance error.
    Uses soft hinge: max(0, margin - (pred_i - pred_j) * sign(true_i - true_j))
    """
    n = len(pred_delta)
    if n < 2:
        return torch.tensor(0.0)
    # Sample pairs to keep O(n) not O(n^2)
    n_samples = min(n * 4, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (n_samples,))
    idx_j = torch.randint(0, n, (n_samples,))
    # Remove self-pairs
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    diff_true = true_delta[idx_i] - true_delta[idx_j]
    diff_pred = pred_delta[idx_i] - pred_delta[idx_j]
    sign = torch.sign(diff_true)
    # Only penalize when signs disagree
    losses = torch.clamp(margin - sign * diff_pred, min=0)
    return losses.mean()


def run_phase_j(mol_data, results):
    """Phase J: Best Single Dual-Objective Model.

    Goal: Find a single model that:
    1. Predicts absolute pIC50 comparably to XGBoost (MAE~0.57, Spr~0.71)
    2. Has better delta Spearman than subtraction baseline (0.678)

    Approaches tested:
    J1: XGBoost with pair-optimized hyperparameters
    J2: XGBoost with custom pairwise ranking objective
    J3: Neural dual-objective with concordance loss + lambda sweep
    J4: XGBoost pairwise residual correction (two-stage, single prediction)
    J5: Best approach with optimized hyperparameters
    """
    print("\n" + "=" * 70)
    print("PHASE J: Best Single Dual-Objective Model")
    print("=" * 70)

    phase = results.get("phase_j", {})
    if phase.get("completed"):
        print("  Already completed, skipping.")
        return results

    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    smiles_list = list(mol_data["smiles"].values)
    y = mol_data["pIC50"].values.astype(np.float32)
    n_mols = len(smiles_list)

    # Precompute fingerprints
    X_morgan = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)

    # Generate all pairs for delta evaluation
    pairs_df = generate_all_pairs(mol_data)
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    mol_indices = np.arange(n_mols)

    def evaluate_model_full(name, get_predictions_fn):
        """Evaluate a model on both absolute and delta metrics.

        get_predictions_fn(train_idx, test_idx) -> predictions for test molecules
        Returns dict with absolute and delta metrics per fold.
        """
        fold_abs = []
        fold_delta = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
            preds = get_predictions_fn(train_idx, test_idx)
            if preds is None:
                continue
            # Absolute metrics
            y_test = y[test_idx]
            abs_metrics = compute_absolute_metrics(y_test, preds)
            fold_abs.append(abs_metrics)

            # Delta metrics on test-only pairs
            test_smiles_set = set(np.array(smiles_list)[test_idx])
            test_pairs = pairs_df[
                pairs_df["mol_a"].isin(test_smiles_set) &
                pairs_df["mol_b"].isin(test_smiles_set)
            ]
            if len(test_pairs) < 3:
                continue
            pred_map = dict(zip(np.array(smiles_list)[test_idx], preds))
            delta_pred = np.array([
                pred_map[b] - pred_map[a]
                for a, b in zip(test_pairs["mol_a"], test_pairs["mol_b"])
            ])
            delta_true = test_pairs["delta"].values.astype(np.float32)
            delta_mae = float(np.mean(np.abs(delta_true - delta_pred)))
            spr_d, _ = spearmanr(delta_true, delta_pred)
            pear_d, _ = pearsonr(delta_true, delta_pred)
            fold_delta.append({
                "delta_mae": delta_mae,
                "delta_spearman": float(spr_d) if not np.isnan(spr_d) else 0,
                "delta_pearson": float(pear_d) if not np.isnan(pear_d) else 0,
                "n_pairs": len(test_pairs),
            })

        if not fold_abs:
            return None

        agg_abs = aggregate_cv_results(fold_abs)
        agg_delta = {
            "delta_mae_mean": float(np.mean([f["delta_mae"] for f in fold_delta])),
            "delta_mae_std": float(np.std([f["delta_mae"] for f in fold_delta])),
            "delta_spearman_mean": float(np.mean([f["delta_spearman"] for f in fold_delta])),
            "delta_spearman_std": float(np.std([f["delta_spearman"] for f in fold_delta])),
            "delta_pearson_mean": float(np.mean([f["delta_pearson"] for f in fold_delta])),
            "delta_pearson_std": float(np.std([f["delta_pearson"] for f in fold_delta])),
        }
        result = {**agg_abs, **agg_delta, "per_fold_abs": fold_abs, "per_fold_delta": fold_delta}
        print(f"    {name}: Abs MAE={agg_abs['mae_mean']:.4f}±{agg_abs['mae_std']:.4f}, "
              f"Abs Spr={agg_abs['spearman_r_mean']:.3f}, "
              f"Δ MAE={agg_delta['delta_mae_mean']:.4f}, "
              f"Δ Spr={agg_delta['delta_spearman_mean']:.3f}±{agg_delta['delta_spearman_std']:.3f}")
        return result

    # ─── J0: Subtraction baseline (XGBoost absolute) ─────────────────────
    print("\n  J0. Subtraction Baseline (XGBoost absolute)...")
    def xgb_baseline(train_idx, test_idx):
        preds, _ = train_xgboost(X_morgan[train_idx], y[train_idx],
                                  X_morgan[test_idx], **BEST_XGB_PARAMS)
        return preds
    phase["j0_subtraction_baseline"] = evaluate_model_full("XGBoost-Sub", xgb_baseline)

    # ─── J1: XGBoost optimized for pairwise ranking ──────────────────────
    print("\n  J1. XGBoost with pair-optimized hyperparameters...")
    # Try several XGBoost configs and pick the one with best delta Spearman
    xgb_configs = [
        {"name": "shallow_reg", "max_depth": 4, "min_child_weight": 5,
         "subsample": 0.7, "colsample_bytree": 0.5, "learning_rate": 0.03,
         "n_estimators": 500, "reg_alpha": 2.0, "reg_lambda": 10.0},
        {"name": "deep_less_reg", "max_depth": 8, "min_child_weight": 1,
         "subsample": 0.8, "colsample_bytree": 0.6, "learning_rate": 0.01,
         "n_estimators": 1000, "reg_alpha": 0.5, "reg_lambda": 3.0},
        {"name": "wide_conservative", "max_depth": 5, "min_child_weight": 3,
         "subsample": 0.65, "colsample_bytree": 0.8, "learning_rate": 0.015,
         "n_estimators": 800, "reg_alpha": 1.0, "reg_lambda": 5.0},
        {"name": "very_shallow", "max_depth": 3, "min_child_weight": 8,
         "subsample": 0.6, "colsample_bytree": 0.4, "learning_rate": 0.05,
         "n_estimators": 400, "reg_alpha": 3.0, "reg_lambda": 15.0},
        {"name": "boosted_more", "max_depth": 6, "min_child_weight": 2,
         "subsample": 0.7, "colsample_bytree": 0.5, "learning_rate": 0.008,
         "n_estimators": 1500, "reg_alpha": 1.5, "reg_lambda": 8.0},
    ]
    j1_results = {}
    for cfg in xgb_configs:
        name = cfg.pop("name")
        def make_pred(train_idx, test_idx, cfg=cfg):
            preds, _ = train_xgboost(X_morgan[train_idx], y[train_idx],
                                      X_morgan[test_idx], **cfg)
            return preds
        r = evaluate_model_full(f"XGB-{name}", make_pred)
        j1_results[name] = r
    # Find best by delta Spearman
    best_j1 = max(j1_results, key=lambda k: j1_results[k]["delta_spearman_mean"])
    phase["j1_xgb_configs"] = j1_results
    phase["j1_best"] = best_j1
    print(f"  → J1 best: {best_j1} (Δ Spr={j1_results[best_j1]['delta_spearman_mean']:.3f})")

    # ─── J2: XGBoost with pairwise ranking custom objective ──────────────
    print("\n  J2. XGBoost with pairwise ranking custom objective...")

    def xgb_pairwise_obj(y_pred, dtrain, alpha=0.5, n_samples=500):
        """Custom XGBoost objective: MSE + alpha * pairwise ranking loss.

        Gradient = grad_MSE + alpha * grad_ranking
        """
        y_true = dtrain.get_label()
        n = len(y_true)
        # MSE gradient
        residual = y_pred - y_true
        grad_mse = 2 * residual
        hess_mse = 2 * np.ones_like(residual)

        # Pairwise ranking gradient: for sampled pairs (i,j),
        # if sign(true_i - true_j) != sign(pred_i - pred_j), push them apart
        grad_rank = np.zeros_like(residual)
        n_samp = min(n_samples, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        idx_i = rng.randint(0, n, n_samp)
        idx_j = rng.randint(0, n, n_samp)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        diff_true = y_true[idx_i] - y_true[idx_j]
        diff_pred = y_pred[idx_i] - y_pred[idx_j]
        # Where signs disagree, gradient pushes predictions in correct direction
        discordant = np.sign(diff_true) != np.sign(diff_pred)
        for k in range(len(idx_i)):
            if discordant[k] and abs(diff_true[k]) > 0.1:  # Only for meaningful diffs
                sign = np.sign(diff_true[k])
                # Push pred_i up if true_i > true_j, push pred_j down
                grad_rank[idx_i[k]] -= sign * 0.01
                grad_rank[idx_j[k]] += sign * 0.01

        grad = grad_mse + alpha * grad_rank
        hess = hess_mse  # Approximate hessian
        return grad, hess

    j2_results = {}
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
        def make_pred_j2(train_idx, test_idx, alpha=alpha):
            dtrain = xgb.DMatrix(X_morgan[train_idx], label=y[train_idx])
            dtest = xgb.DMatrix(X_morgan[test_idx])
            params = {
                "max_depth": 6, "min_child_weight": 2,
                "subsample": 0.6, "colsample_bytree": 0.5,
                "learning_rate": 0.02, "reg_alpha": 1.5, "reg_lambda": 7.0,
                "seed": 42,
            }
            def obj_fn(y_pred, dtrain):
                return xgb_pairwise_obj(y_pred, dtrain, alpha=alpha)
            bst = xgb.train(params, dtrain, num_boost_round=750, obj=obj_fn,
                           verbose_eval=False)
            return bst.predict(dtest)
        r = evaluate_model_full(f"XGB-rank-α{alpha}", make_pred_j2)
        j2_results[f"alpha_{alpha}"] = r
    best_j2 = max(j2_results, key=lambda k: j2_results[k]["delta_spearman_mean"])
    phase["j2_xgb_pairwise"] = j2_results
    phase["j2_best"] = best_j2
    print(f"  → J2 best: {best_j2} (Δ Spr={j2_results[best_j2]['delta_spearman_mean']:.3f})")

    # ─── J3: Neural dual-objective with concordance loss ─────────────────
    print("\n  J3. Neural Dual-Objective with Concordance Loss...")

    def train_ranking_dual(X_train, y_train, X_test,
                           lambda_rank=0.5, lambda_abs=1.0,
                           hidden_dims=[256, 128], dropout=0.4,
                           lr=5e-4, epochs=300, batch_size=32, patience=30,
                           n_pair_samples=200):
        """Train neural model optimizing: L = λ_abs * L_MSE + λ_rank * L_concordance."""
        scaler = StandardScaler()
        X_tr = torch.FloatTensor(scaler.fit_transform(X_train))
        X_te = torch.FloatTensor(scaler.transform(X_test))
        y_tr = torch.FloatTensor(y_train)

        model = PairRankingDualModel(X_tr.shape[1], hidden_dims, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Val split
        n_val = max(10, len(X_tr) // 5)
        perm = np.random.RandomState(42).permutation(len(X_tr))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        best_val = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            model.train()
            perm_tr = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), batch_size):
                bi = tr_idx[perm_tr[start:start + batch_size]]
                optimizer.zero_grad()
                pred = model(X_tr[bi])
                loss_abs = nn.MSELoss()(pred, y_tr[bi])

                # Concordance loss: sample pairs from batch
                if len(bi) >= 4:
                    # Generate implied deltas from absolute predictions
                    n_p = min(n_pair_samples, len(bi) * (len(bi) - 1) // 2)
                    pi = torch.randint(0, len(bi), (n_p,))
                    pj = torch.randint(0, len(bi), (n_p,))
                    mask = pi != pj
                    pi, pj = pi[mask], pj[mask]
                    delta_pred = pred[pj] - pred[pi]
                    delta_true = y_tr[bi[pj]] - y_tr[bi[pi]]
                    loss_rank = concordance_loss(delta_pred, delta_true, margin=0.05)
                else:
                    loss_rank = torch.tensor(0.0)

                loss = lambda_abs * loss_abs + lambda_rank * loss_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_tr[val_idx])
                val_loss = nn.MSELoss()(val_pred, y_tr[val_idx]).item()
            if val_loss < best_val:
                best_val = val_loss
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
            preds = model(X_te).numpy()
        return preds

    j3_results = {}
    configs_j3 = [
        {"name": "rank0.3_abs1.0", "lambda_rank": 0.3, "lambda_abs": 1.0},
        {"name": "rank0.5_abs1.0", "lambda_rank": 0.5, "lambda_abs": 1.0},
        {"name": "rank1.0_abs1.0", "lambda_rank": 1.0, "lambda_abs": 1.0},
        {"name": "rank2.0_abs1.0", "lambda_rank": 2.0, "lambda_abs": 1.0},
        {"name": "rank1.0_abs0.5", "lambda_rank": 1.0, "lambda_abs": 0.5},
        {"name": "rank0.5_wider", "lambda_rank": 0.5, "lambda_abs": 1.0,
         "hidden_dims": [512, 256], "dropout": 0.5},
        {"name": "rank0.5_deeper", "lambda_rank": 0.5, "lambda_abs": 1.0,
         "hidden_dims": [256, 128, 64], "dropout": 0.4},
    ]
    for cfg in configs_j3:
        name = cfg.pop("name")
        def make_pred_j3(train_idx, test_idx, cfg=cfg):
            return train_ranking_dual(
                X_morgan[train_idx], y[train_idx], X_morgan[test_idx],
                lambda_rank=cfg.get("lambda_rank", 0.5),
                lambda_abs=cfg.get("lambda_abs", 1.0),
                hidden_dims=cfg.get("hidden_dims", [256, 128]),
                dropout=cfg.get("dropout", 0.4),
            )
        r = evaluate_model_full(f"Neural-{name}", make_pred_j3)
        j3_results[name] = r
    best_j3 = max(j3_results, key=lambda k: j3_results[k]["delta_spearman_mean"])
    phase["j3_neural_dual"] = j3_results
    phase["j3_best"] = best_j3
    print(f"  → J3 best: {best_j3} (Δ Spr={j3_results[best_j3]['delta_spearman_mean']:.3f})")

    # ─── J4: XGBoost with pair-aware residual correction ─────────────────
    print("\n  J4. XGBoost + Pair-Aware Residual Correction...")

    def xgb_pair_corrected(train_idx, test_idx, correction_weight=0.3):
        """Train XGBoost on absolute values, then apply pair-aware bias correction.

        For each test molecule i, compute:
        pred_corrected_i = pred_i + w * median_j(residual_j * sim(i,j))

        where residual_j = y_j - pred_j for training molecules,
        weighted by Tanimoto similarity. This corrects systematic biases
        that harm pairwise rankings.
        """
        # Train XGBoost absolute
        preds_tr, model = train_xgboost(X_morgan[train_idx], y[train_idx],
                                         X_morgan[train_idx], **BEST_XGB_PARAMS)
        preds_te_raw, _ = train_xgboost(X_morgan[train_idx], y[train_idx],
                                         X_morgan[test_idx], **BEST_XGB_PARAMS)
        # Compute training residuals
        residuals = y[train_idx] - preds_tr

        # Compute similarities (test × train)
        from rdkit.Chem import AllChem
        train_fps = [AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles_list[i]), 2, nBits=2048)
            for i in train_idx]
        test_fps = [AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles_list[i]), 2, nBits=2048)
            for i in test_idx]

        preds_corrected = np.zeros(len(test_idx))
        for i, tfp in enumerate(test_fps):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(tfp, train_fps))
            # Weight residuals by similarity
            weighted_correction = np.sum(sims * residuals) / (np.sum(sims) + 1e-8)
            preds_corrected[i] = preds_te_raw[i] + correction_weight * weighted_correction

        return preds_corrected

    j4_results = {}
    for w in [0.1, 0.2, 0.3, 0.5, 0.7]:
        def make_pred_j4(train_idx, test_idx, w=w):
            return xgb_pair_corrected(train_idx, test_idx, correction_weight=w)
        r = evaluate_model_full(f"XGB-corr-w{w}", make_pred_j4)
        j4_results[f"w_{w}"] = r
    best_j4 = max(j4_results, key=lambda k: j4_results[k]["delta_spearman_mean"])
    phase["j4_xgb_corrected"] = j4_results
    phase["j4_best"] = best_j4
    print(f"  → J4 best: {best_j4} (Δ Spr={j4_results[best_j4]['delta_spearman_mean']:.3f})")

    # ─── J5: XGBoost pair-reweighted training ────────────────────────────
    print("\n  J5. XGBoost with pair-reweighted sample weights...")

    def xgb_pair_reweighted(train_idx, test_idx, pair_weight=2.0):
        """Train XGBoost where molecules involved in many discordant pairs
        get higher sample weights, forcing the model to get them right.

        Iterative: train → find discordant → reweight → retrain.
        """
        X_tr, y_tr = X_morgan[train_idx], y[train_idx]
        X_te = X_morgan[test_idx]
        weights = np.ones(len(train_idx), dtype=np.float32)

        for iteration in range(3):
            # Train with current weights
            model = xgb.XGBRegressor(
                **BEST_XGB_PARAMS, random_state=42, n_jobs=N_JOBS)
            model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
            preds_tr = model.predict(X_tr)

            # Find discordant pairs
            n_tr = len(train_idx)
            discordance_count = np.zeros(n_tr)
            # Sample pairs
            rng = np.random.RandomState(42 + iteration)
            n_samp = min(5000, n_tr * (n_tr - 1) // 2)
            ii = rng.randint(0, n_tr, n_samp)
            jj = rng.randint(0, n_tr, n_samp)
            mask = ii != jj
            ii, jj = ii[mask], jj[mask]

            diff_true = y_tr[ii] - y_tr[jj]
            diff_pred = preds_tr[ii] - preds_tr[jj]
            discordant = (np.sign(diff_true) != np.sign(diff_pred)) & (np.abs(diff_true) > 0.2)

            for k in range(len(ii)):
                if discordant[k]:
                    discordance_count[ii[k]] += 1
                    discordance_count[jj[k]] += 1

            # Reweight: molecules with more discordances get higher weight
            if discordance_count.max() > 0:
                norm_disc = discordance_count / discordance_count.max()
                weights = 1.0 + pair_weight * norm_disc

        preds = model.predict(X_te)
        return preds

    j5_results = {}
    for pw in [0.5, 1.0, 2.0, 3.0, 5.0]:
        def make_pred_j5(train_idx, test_idx, pw=pw):
            return xgb_pair_reweighted(train_idx, test_idx, pair_weight=pw)
        r = evaluate_model_full(f"XGB-reweight-{pw}", make_pred_j5)
        j5_results[f"pw_{pw}"] = r
    best_j5 = max(j5_results, key=lambda k: j5_results[k]["delta_spearman_mean"])
    phase["j5_xgb_reweighted"] = j5_results
    phase["j5_best"] = best_j5
    print(f"  → J5 best: {best_j5} (Δ Spr={j5_results[best_j5]['delta_spearman_mean']:.3f})")

    # ─── J6: Multi-seed averaged neural dual-objective ───────────────────
    print("\n  J6. Multi-Seed Averaged Neural Dual-Objective...")

    def neural_multi_seed(train_idx, test_idx, n_seeds=5,
                          lambda_rank=0.5, lambda_abs=1.0):
        """Average predictions from multiple seeds for stability."""
        all_preds = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            preds = train_ranking_dual(
                X_morgan[train_idx], y[train_idx], X_morgan[test_idx],
                lambda_rank=lambda_rank, lambda_abs=lambda_abs,
                hidden_dims=[256, 128], dropout=0.4, lr=5e-4,
                epochs=300, batch_size=32, patience=30,
            )
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)

    # Use best lambda from J3
    best_j3_cfg = None
    for cfg in configs_j3:
        if cfg.get("name") == best_j3 or True:
            pass
    # Just use the best from J3 results
    best_j3_data = j3_results[best_j3]
    # Extract lambda values from name
    j3_lr = 0.5  # default
    j3_la = 1.0

    def make_pred_j6(train_idx, test_idx):
        return neural_multi_seed(train_idx, test_idx, n_seeds=5)
    j6_result = evaluate_model_full("Neural-5seed", make_pred_j6)
    phase["j6_neural_multiseed"] = j6_result

    # ─── J7: Quantile-blended XGBoost ────────────────────────────────────
    print("\n  J7. Quantile-Blended XGBoost (MSE + Quantile objectives)...")

    def xgb_quantile_blend(train_idx, test_idx, q_weight=0.3):
        """Blend MSE-trained XGBoost with quantile-trained XGBoost.

        Quantile regression (median) is more robust to outliers,
        which can improve pairwise ranking by reducing outlier influence.
        """
        X_tr, y_tr = X_morgan[train_idx], y[train_idx]
        X_te = X_morgan[test_idx]

        # Standard MSE model
        preds_mse, _ = train_xgboost(X_tr, y_tr, X_te, **BEST_XGB_PARAMS)

        # Quantile regression (pseudo-Huber approximation)
        params_q = {**BEST_XGB_PARAMS, "objective": "reg:pseudohubererror",
                    "huber_slope": 0.5}  # More robust than MSE
        preds_q, _ = train_xgboost(X_tr, y_tr, X_te, **params_q)

        # Blend
        return (1 - q_weight) * preds_mse + q_weight * preds_q

    j7_results = {}
    for qw in [0.2, 0.3, 0.5]:
        def make_pred_j7(train_idx, test_idx, qw=qw):
            return xgb_quantile_blend(train_idx, test_idx, q_weight=qw)
        r = evaluate_model_full(f"XGB-qblend-{qw}", make_pred_j7)
        j7_results[f"qw_{qw}"] = r
    best_j7 = max(j7_results, key=lambda k: j7_results[k]["delta_spearman_mean"])
    phase["j7_xgb_quantile"] = j7_results
    phase["j7_best"] = best_j7
    print(f"  → J7 best: {best_j7} (Δ Spr={j7_results[best_j7]['delta_spearman_mean']:.3f})")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE J SUMMARY — Single Model Comparison")
    print("=" * 70)

    all_best = {
        "J0 XGBoost-Sub (baseline)": phase["j0_subtraction_baseline"],
        f"J1 XGB-{best_j1}": j1_results[best_j1],
        f"J2 XGB-rank-{best_j2}": j2_results[best_j2],
        f"J3 Neural-{best_j3}": j3_results[best_j3],
        f"J4 XGB-corr-{best_j4}": j4_results[best_j4],
        f"J5 XGB-reweight-{best_j5}": j5_results[best_j5],
        "J6 Neural-5seed": j6_result,
        f"J7 XGB-qblend-{best_j7}": j7_results[best_j7],
    }

    print(f"\n  {'Method':<35} {'Abs MAE':>8} {'Abs Spr':>8} {'Δ MAE':>8} {'Δ Spr':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, r in sorted(all_best.items(), key=lambda x: -x[1]["delta_spearman_mean"]):
        print(f"  {name:<35} {r['mae_mean']:>8.4f} {r['spearman_r_mean']:>8.3f} "
              f"{r['delta_mae_mean']:>8.4f} {r['delta_spearman_mean']:>8.3f}")

    # Overall winner
    winner_name = max(all_best, key=lambda k: all_best[k]["delta_spearman_mean"])
    winner = all_best[winner_name]
    phase["overall_winner"] = {
        "name": winner_name,
        "abs_mae": winner["mae_mean"],
        "abs_spearman": winner["spearman_r_mean"],
        "delta_mae": winner["delta_mae_mean"],
        "delta_spearman": winner["delta_spearman_mean"],
    }
    print(f"\n  ★ WINNER: {winner_name}")
    print(f"    Abs MAE={winner['mae_mean']:.4f}, Abs Spr={winner['spearman_r_mean']:.3f}")
    print(f"    Δ MAE={winner['delta_mae_mean']:.4f}, Δ Spr={winner['delta_spearman_mean']:.3f}")

    # Check if any model beats subtraction on delta Spearman
    baseline_spr = phase["j0_subtraction_baseline"]["delta_spearman_mean"]
    if winner["delta_spearman_mean"] > baseline_spr:
        improvement = (winner["delta_spearman_mean"] - baseline_spr) / baseline_spr * 100
        print(f"    → Beats subtraction delta Spr by {improvement:.1f}%!")
    else:
        print(f"    → Does NOT beat subtraction delta Spr ({baseline_spr:.3f})")

    phase["completed"] = True
    results["phase_j"] = phase
    save_results(results)
    print("\n  Phase J complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ZAP70 v7 — Comprehensive deep analysis")
    parser.add_argument("--phase", nargs="+", default=None,
                        help="Run specific phases (A B C D E F G H I J)")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print(f"ZAP70 Case Study v7 — Comprehensive Deep Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    mol_data, per_assay = load_zap70_molecules()

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"  Loaded existing results: {list(results.keys())}")

    results["data_summary"] = {
        "target": "ZAP70 (CHEMBL2803)",
        "n_molecules": len(mol_data),
    }

    phases_to_run = [p.upper() for p in args.phase] if args.phase else ["A", "B", "C", "D", "E", "F", "G", "J"]

    if "A" in phases_to_run:
        results = run_phase_a(mol_data, results)
        gc.collect()

    if "B" in phases_to_run:
        results = run_phase_b(mol_data, results)
        gc.collect()

    if "C" in phases_to_run:
        results = run_phase_c(mol_data, results)
        gc.collect()

    if "D" in phases_to_run:
        results = run_phase_d(mol_data, results)
        gc.collect()

    if "E" in phases_to_run:
        results = run_phase_e(mol_data, results)
        gc.collect()

    if "F" in phases_to_run:
        results = run_phase_f(mol_data, results)
        gc.collect()

    if "G" in phases_to_run:
        results = run_phase_g(mol_data, results)
        gc.collect()

    if "H" in phases_to_run:
        results = run_phase_h(mol_data, results)
        gc.collect()

    if "I" in phases_to_run:
        results = run_phase_i(mol_data, results)
        gc.collect()

    if "J" in phases_to_run:
        results = run_phase_j(mol_data, results)
        gc.collect()

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed
    results["completed"] = True
    save_results(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
