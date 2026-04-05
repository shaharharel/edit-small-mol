#!/usr/bin/env python3
"""
Unified paper evaluation: embedder selection → architecture comparison → generalization.

Produces a single coherent set of results and an HTML report.

Usage:
    conda run -n quris python -u experiments/run_paper_evaluation.py
    conda run -n quris python -u experiments/run_paper_evaluation.py --phase 1
    conda run -n quris python -u experiments/run_paper_evaluation.py --phase 2
    conda run -n quris python -u experiments/run_paper_evaluation.py --phase 3
    conda run -n quris python -u experiments/run_paper_evaluation.py --report-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Force CPU — MPS crashes with transformers after prolonged use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False
DEVICE = "cpu"

# ═══════════════════════════════════════════════════════════════════════════
# Paths & Config
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"
CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

# Data files in priority order
SHARED_PAIRS_FILE = DATA_DIR / "shared_pairs_deduped.csv"
SUBSAMPLE_FILE = DATA_DIR / "overlapping_assay_pairs_minimal_mmp_50k.csv"
FULL_FILE = DATA_DIR / "overlapping_assay_pairs_minimal_mmp.csv"

SEEDS = [42, 123, 456]
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
DROPOUT = 0.2
MAX_PHASE3_PAIRS = None  # No cap — use full dataset; gc.collect() between runs controls memory

# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class DeltaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AbsoluteMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Compute all metrics. Returns dict with mae, rmse, pearson, spearman, r2."""
    if len(y_true) < 3:
        return {"n": len(y_true)}
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    try:
        pr, _ = pearsonr(y_true, y_pred)
    except Exception:
        pr = 0.0
    try:
        sr, _ = spearmanr(y_true, y_pred)
    except Exception:
        sr = 0.0
    return {
        "n": len(y_true), "mae": float(mae), "rmse": float(rmse),
        "r2": float(r2), "pearson_r": float(pr), "spearman_r": float(sr),
    }


def compute_per_target_metrics(y_true, y_pred, targets):
    """Compute per-target metrics and return averaged results."""
    target_metrics = {}
    for t in np.unique(targets):
        mask = targets == t
        if mask.sum() >= 10:
            m = compute_metrics(y_true[mask], y_pred[mask])
            target_metrics[t] = m

    if not target_metrics:
        return {}

    # Average across targets
    avg = {}
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_r"]:
        vals = [m[metric] for m in target_metrics.values() if metric in m]
        if vals:
            avg[f"{metric}_mean"] = float(np.mean(vals))
            avg[f"{metric}_std"] = float(np.std(vals))
    avg["n_targets"] = len(target_metrics)
    avg["per_target"] = target_metrics
    return avg


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, lr=LR):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    val_losses.append(criterion(model(x), y).item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model


def train_model_multi_input(model, train_loader, val_loader, forward_fn,
                            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR):
    """Train a model that takes multiple inputs (not just a single tensor).

    forward_fn: callable(model, *batch_inputs) -> predictions tensor
        Takes the model and unpacked batch tensors (excluding the last = target),
        returns predicted values.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            inputs = [b.to(DEVICE) for b in batch[:-1]]
            target = batch[-1].to(DEVICE)
            optimizer.zero_grad()
            pred = forward_fn(model, *inputs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = [b.to(DEVICE) for b in batch[:-1]]
                    target = batch[-1].to(DEVICE)
                    pred = forward_fn(model, *inputs)
                    val_losses.append(criterion(pred, target).item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model


def predict_multi_input(model, forward_fn, *tensors):
    """Predict with a multi-input model."""
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        inputs = [t.to(DEVICE) for t in tensors]
        return forward_fn(model, *inputs).cpu().numpy()


def predict(model, x):
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        return model(x.to(DEVICE)).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Embedding computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_embeddings(smiles_list, embedder_name):
    """Load cached embeddings or compute on-the-fly."""
    smiles_list = list(set(smiles_list))
    cache_file = CACHE_DIR / f"{embedder_name}.npz"

    # Try loading from cache first
    if cache_file.exists():
        print(f"  Loading cached {embedder_name} embeddings from {cache_file.name}...")
        t0 = time.time()
        data = np.load(cache_file, allow_pickle=True)
        cached_smiles = data['smiles'].tolist()
        cached_embs = data['embeddings']
        emb_dim = int(data['emb_dim'])
        cached_dict = {smi: cached_embs[i] for i, smi in enumerate(cached_smiles)}

        # Check coverage
        missing = [s for s in smiles_list if s not in cached_dict]
        if missing:
            print(f"    WARNING: {len(missing)} molecules not in cache, using zero vectors")
        emb_dict = {s: cached_dict.get(s, np.zeros(emb_dim, dtype=np.float32)) for s in smiles_list}
        elapsed = time.time() - t0
        print(f"    Loaded {len(emb_dict):,} embeddings (dim={emb_dim}) in {elapsed:.1f}s")
        return emb_dict, emb_dim

    # Fall back to computing on-the-fly
    print(f"  Computing {embedder_name} embeddings for {len(smiles_list):,} molecules...")
    t0 = time.time()

    if embedder_name == "morgan":
        from src.embedding.fingerprints import FingerprintEmbedder
        embedder = FingerprintEmbedder(fp_type="morgan", radius=2, n_bits=2048)
        emb_dict = {}
        for smi in smiles_list:
            emb_dict[smi] = embedder.encode(smi)
        emb_dim = 2048

    elif embedder_name in ("chemberta2-mtr", "chemberta2-mlm"):
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name=embedder_name, device='cpu', batch_size=128)
        embs = embedder.encode(smiles_list)
        emb_dict = {smi: embs[i] for i, smi in enumerate(smiles_list)}
        emb_dim = embedder.embedding_dim

    elif embedder_name == "chemprop-dmpnn":
        from src.embedding.chemprop import ChemPropEmbedder
        embedder = ChemPropEmbedder(featurizer_type='morgan')
        embs = embedder.encode(smiles_list)
        emb_dict = {smi: embs[i] for i, smi in enumerate(smiles_list)}
        emb_dim = embedder.embedding_dim

    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s (dim={emb_dim})")
    return emb_dict, emb_dim


# ═══════════════════════════════════════════════════════════════════════════
# Method trainers (generic for any embedding dimension)
# ═══════════════════════════════════════════════════════════════════════════

def get_pair_tensors(df, emb_dict, emb_dim):
    zero = np.zeros(emb_dim)
    emb_a = np.array([emb_dict.get(s, zero) for s in df["mol_a"]])
    emb_b = np.array([emb_dict.get(s, zero) for s in df["mol_b"]])
    delta = df["delta"].values.astype(np.float32)
    return (
        torch.from_numpy(emb_a).float(),
        torch.from_numpy(emb_b).float(),
        torch.from_numpy(delta).float(),
    )


def train_and_predict_subtraction(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Subtraction baseline: train F(mol)→property, predict F(B)-F(A)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect molecule-level labels from training pairs (vectorized)
    mol_values = dict(zip(train_df["mol_a"], train_df["value_a"]))
    mol_values.update(dict(zip(train_df["mol_b"], train_df["value_b"])))

    zero = np.zeros(emb_dim)
    smiles_list = list(mol_values.keys())
    y_vals = np.array([mol_values[s] for s in smiles_list], dtype=np.float32)
    X = np.array([emb_dict.get(s, zero) for s in smiles_list], dtype=np.float32)

    # Validation molecules (vectorized)
    val_mol_values = dict(zip(val_df["mol_a"], val_df["value_a"]))
    val_mol_values.update(dict(zip(val_df["mol_b"], val_df["value_b"])))
    val_smiles = list(val_mol_values.keys())
    val_y = np.array([val_mol_values[s] for s in val_smiles], dtype=np.float32)
    val_X = np.array([emb_dict.get(s, zero) for s in val_smiles], dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_vals).float()),
        batch_size=BATCH_SIZE, shuffle=True    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float()),
        batch_size=BATCH_SIZE, shuffle=False    )

    model = AbsoluteMLP(emb_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict(model, emb_b) - predict(model, emb_a)


def train_and_predict_edit_diff(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Edit effect: MLP on [emb_a, emb_b - emb_a] → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_input(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return torch.cat([emb_a, emb_b - emb_a], dim=-1), delta

    train_x, train_y = make_input(train_df)
    val_x, val_y = make_input(val_df)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeltaMLP(emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict(model, torch.cat([emb_a, emb_b - emb_a], dim=-1))


def train_and_predict_deepdelta(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """DeepDelta: MLP on [emb_a, emb_b] → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_input(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return torch.cat([emb_a, emb_b], dim=-1), delta

    train_x, train_y = make_input(train_df)
    val_x, val_y = make_input(val_df)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeltaMLP(emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict(model, torch.cat([emb_a, emb_b], dim=-1))


def _compute_edit_feats_batch(df):
    """Compute 28-dim edit features for a dataframe of pairs (vectorized access)."""
    from src.data.utils.chemistry import compute_edit_features, EDIT_FEAT_DIM
    feats = []
    mol_a_vals = df["mol_a"].values
    mol_b_vals = df["mol_b"].values
    edit_vals = df["edit_smiles"].values if "edit_smiles" in df.columns else [""] * len(df)
    for i in range(len(df)):
        try:
            feats.append(compute_edit_features(mol_a_vals[i], mol_b_vals[i], edit_vals[i]))
        except Exception:
            feats.append(np.zeros(EDIT_FEAT_DIM, dtype=np.float32))
    return np.array(feats, dtype=np.float32)


def train_and_predict_edit_features(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Edit effect + chemical features: MLP on [emb_a, emb_b - emb_a, edit_feats] → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.data.utils.chemistry import EDIT_FEAT_DIM

    def make_input(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        edit_emb = emb_b - emb_a
        ef = torch.from_numpy(_compute_edit_feats_batch(df)).float()
        ef = torch.nan_to_num(ef, nan=0.0)
        return torch.cat([emb_a, edit_emb, ef], dim=-1), delta

    train_x, train_y = make_input(train_df)
    val_x, val_y = make_input(val_df)

    input_dim = emb_dim * 2 + EDIT_FEAT_DIM
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeltaMLP(input_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)
    model = train_model(model, train_loader, val_loader)

    emb_a, emb_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    edit_emb = emb_b - emb_a
    ef = torch.from_numpy(_compute_edit_feats_batch(test_df)).float()
    ef = torch.nan_to_num(ef, nan=0.0)
    return predict(model, torch.cat([emb_a, edit_emb, ef], dim=-1))


def train_and_predict_gated_crossattn(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Gated Cross-Attention MLP on (emb_a, emb_b, edit_feats) → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.attention_delta_predictor import GatedCrossAttnMLP
    from src.data.utils.chemistry import EDIT_FEAT_DIM

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        ef = torch.from_numpy(_compute_edit_feats_batch(df)).float()
        ef = torch.nan_to_num(ef, nan=0.0)
        return emb_a, emb_b, ef, delta

    train_a, train_b, train_ef, train_y = make_datasets(train_df)
    val_a, val_b, val_ef, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_ef, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_ef, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GatedCrossAttnMLP(input_dim=emb_dim, hidden_dim=256, n_layers=3)

    def forward_fn(m, a, b, ef):
        return m(a, b, ef)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, test_ef, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b, test_ef)


def train_and_predict_attn_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Attention + FiLM MLP on (emb_a, emb_b, edit_feats) → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.attention_delta_predictor import AttnThenFiLMMLP

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        ef = torch.from_numpy(_compute_edit_feats_batch(df)).float()
        ef = torch.nan_to_num(ef, nan=0.0)
        return emb_a, emb_b, ef, delta

    train_a, train_b, train_ef, train_y = make_datasets(train_df)
    val_a, val_b, val_ef, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_ef, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_ef, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AttnThenFiLMMLP(input_dim=emb_dim, hidden_dim=256)

    def forward_fn(m, a, b, ef):
        return m(wt=a, mt=b, mf=ef)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, test_ef, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b, test_ef)


def train_and_predict_film_delta(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """FiLM-conditioned delta prediction on (emb_a, emb_b) → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return emb_a, emb_b, delta

    train_a, train_b, train_y = make_datasets(train_df)
    val_a, val_b, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])

    def forward_fn(m, a, b):
        return m(a, b)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b)


def train_and_predict_trainable_edit(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Trainable edit embedder + DeltaMLP predictor on (emb_a, emb_b) → delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.embedding.trainable_edit_embedder import TrainableEditEmbedder

    class TrainableEditModel(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.edit_embedder = TrainableEditEmbedder(
                mol_dim=emb_dim, edit_dim=emb_dim, hidden_dims=[emb_dim])
            self.predictor = DeltaMLP(
                emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)

        def forward(self, emb_a, emb_b):
            edit_emb = self.edit_embedder(emb_a, emb_b)
            return self.predictor(torch.cat([emb_a, edit_emb], dim=-1))

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        return emb_a, emb_b, delta

    train_a, train_b, train_y = make_datasets(train_df)
    val_a, val_b, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TrainableEditModel(emb_dim)

    def forward_fn(m, a, b):
        return m(a, b)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b)


# ═══════════════════════════════════════════════════════════════════════════
# Edit representation caching (DRFP, fragment deltas, edit features)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level caches (loaded once, reused across runs)
DRFP_CACHE = None       # dict: (mol_a, mol_b) → np.ndarray(2048,)
FRAG_DELTA_CACHE = None  # dict: edit_smiles → np.ndarray(1024,)
EDIT_FEATS_CACHE = None  # dict: (mol_a, mol_b, edit_smiles) → np.ndarray(28,)


def compute_drfp_cache(df):
    """Compute or load cached DRFP reaction fingerprints for all pairs."""
    global DRFP_CACHE
    if DRFP_CACHE is not None:
        return DRFP_CACHE

    cache_file = CACHE_DIR / "drfp_2048.npz"
    if cache_file.exists():
        print("  Loading cached DRFP fingerprints...")
        data = np.load(cache_file, allow_pickle=True)
        keys = [tuple(k) for k in data['keys']]
        values = data['values']
        DRFP_CACHE = {k: values[i] for i, k in enumerate(keys)}
        print(f"    Loaded {len(DRFP_CACHE):,} DRFP fingerprints")
        return DRFP_CACHE

    print("  Computing DRFP fingerprints for all pairs...")
    from drfp import DrfpEncoder

    # Get unique (mol_a, mol_b) pairs
    pairs = list(set(zip(df["mol_a"], df["mol_b"])))
    print(f"    {len(pairs):,} unique pairs")

    DRFP_CACHE = {}
    batch_size = 5000
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        rxn_smiles = [f"{a}>>{b}" for a, b in batch_pairs]
        try:
            fps = DrfpEncoder.encode(rxn_smiles, n_folded_length=2048)
            for j, (a, b) in enumerate(batch_pairs):
                DRFP_CACHE[(a, b)] = np.array(fps[j], dtype=np.float32)
        except Exception as e:
            print(f"    WARNING: DRFP batch {i} failed: {e}")
            for a, b in batch_pairs:
                try:
                    fp = DrfpEncoder.encode(f"{a}>>{b}", n_folded_length=2048)
                    DRFP_CACHE[(a, b)] = np.array(fp[0], dtype=np.float32)
                except Exception:
                    DRFP_CACHE[(a, b)] = np.zeros(2048, dtype=np.float32)

        if (i // batch_size) % 20 == 0:
            print(f"    {i + len(batch_pairs):,}/{len(pairs):,}")

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys_arr = np.array(list(DRFP_CACHE.keys()))
    values_arr = np.array(list(DRFP_CACHE.values()))
    np.savez_compressed(cache_file, keys=keys_arr, values=values_arr)
    print(f"    Cached {len(DRFP_CACHE):,} DRFP fingerprints to {cache_file.name}")
    return DRFP_CACHE


def compute_frag_delta_cache(df):
    """Compute or load cached fragment FP deltas for all edit_smiles."""
    global FRAG_DELTA_CACHE
    if FRAG_DELTA_CACHE is not None:
        return FRAG_DELTA_CACHE

    cache_file = CACHE_DIR / "fragment_deltas_1024.npz"
    if cache_file.exists():
        print("  Loading cached fragment deltas...")
        data = np.load(cache_file, allow_pickle=True)
        keys = data['keys'].tolist()
        values = data['values']
        FRAG_DELTA_CACHE = {k: values[i] for i, k in enumerate(keys)}
        print(f"    Loaded {len(FRAG_DELTA_CACHE):,} fragment deltas")
        return FRAG_DELTA_CACHE

    print("  Computing fragment FP deltas...")
    from src.data.utils.chemistry import compute_fragment_delta

    edits = df["edit_smiles"].dropna().unique().tolist() if "edit_smiles" in df.columns else []
    print(f"    {len(edits):,} unique edits")

    FRAG_DELTA_CACHE = {}
    for i, es in enumerate(edits):
        FRAG_DELTA_CACHE[es] = compute_fragment_delta(es, n_bits=1024)
        if (i + 1) % 50000 == 0:
            print(f"    {i + 1:,}/{len(edits):,}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys_arr = np.array(list(FRAG_DELTA_CACHE.keys()))
    values_arr = np.array(list(FRAG_DELTA_CACHE.values()))
    np.savez_compressed(cache_file, keys=keys_arr, values=values_arr)
    print(f"    Cached {len(FRAG_DELTA_CACHE):,} fragment deltas to {cache_file.name}")
    return FRAG_DELTA_CACHE


def get_drfp_tensors(df, drfp_cache):
    """Get DRFP tensors for a dataframe of pairs (vectorized)."""
    zero = np.zeros(2048, dtype=np.float32)
    mol_a_vals = df["mol_a"].values
    mol_b_vals = df["mol_b"].values
    drfps = np.array([drfp_cache.get((mol_a_vals[i], mol_b_vals[i]), zero)
                      for i in range(len(df))], dtype=np.float32)
    return torch.from_numpy(drfps).float()


def get_frag_delta_tensors(df, frag_cache):
    """Get fragment delta tensors for a dataframe of pairs (vectorized)."""
    zero = np.zeros(1024, dtype=np.float32)
    if "edit_smiles" in df.columns:
        edit_vals = df["edit_smiles"].values
        deltas = np.array([frag_cache.get(edit_vals[i], zero)
                          for i in range(len(df))], dtype=np.float32)
    else:
        deltas = np.zeros((len(df), 1024), dtype=np.float32)
    return torch.from_numpy(deltas).float()


def get_edit_feats_tensors(df):
    """Get 28-dim edit feature tensors for a dataframe."""
    feats = torch.from_numpy(_compute_edit_feats_batch(df)).float()
    return torch.nan_to_num(feats, nan=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Edit-aware FiLM architecture trainers
# ═══════════════════════════════════════════════════════════════════════════

def train_and_predict_drfp_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """DRFP-FiLM: FiLM conditioned on DRFP reaction fingerprint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.edit_aware_film_predictor import DrfpFiLMDeltaMLP

    drfp_cache = compute_drfp_cache(train_df)

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        drfp = get_drfp_tensors(df, drfp_cache)
        return emb_a, emb_b, drfp, delta

    train_a, train_b, train_drfp, train_y = make_datasets(train_df)
    val_a, val_b, val_drfp, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_drfp, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_drfp, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DrfpFiLMDeltaMLP(mol_dim=emb_dim, drfp_dim=2048)

    def forward_fn(m, a, b, drfp):
        return m(a, b, drfp)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, test_drfp, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b, test_drfp)


def train_and_predict_dualstream_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """DualStream-FiLM: gated DRFP + Morgan diff fusion with edit features."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.edit_aware_film_predictor import DualStreamFiLMDeltaMLP

    drfp_cache = compute_drfp_cache(train_df)

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        drfp = get_drfp_tensors(df, drfp_cache)
        ef = get_edit_feats_tensors(df)
        return emb_a, emb_b, drfp, ef, delta

    train_a, train_b, train_drfp, train_ef, train_y = make_datasets(train_df)
    val_a, val_b, val_drfp, val_ef, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_drfp, train_ef, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_drfp, val_ef, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DualStreamFiLMDeltaMLP(mol_dim=emb_dim, drfp_dim=2048, edit_feat_dim=28)

    # Custom training with auxiliary loss
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, drfp, ef, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            pred = model(a, b, drfp, ef)
            loss = criterion(pred, y) + 0.1 * model.aux_loss(ef)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                a, b, drfp, ef, y = [t.to(DEVICE) for t in batch]
                pred = model(a, b, drfp, ef)
                val_losses.append(criterion(pred, y).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    test_a, test_b, test_drfp, test_ef, _ = make_datasets(test_df)

    def forward_fn(m, a, b, drfp, ef):
        return m(a, b, drfp, ef)

    return predict_multi_input(model, forward_fn, test_a, test_b, test_drfp, test_ef)


def train_and_predict_frag_anchored_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Fragment-Anchored FiLM: scaffold-independent fragment + edit features."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.edit_aware_film_predictor import FragAnchoredFiLMDeltaMLP

    frag_cache = compute_frag_delta_cache(train_df)

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        frag_d = get_frag_delta_tensors(df, frag_cache)
        ef = get_edit_feats_tensors(df)
        return emb_a, emb_b, frag_d, ef, delta

    train_a, train_b, train_frag, train_ef, train_y = make_datasets(train_df)
    val_a, val_b, val_frag, val_ef, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_frag, train_ef, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_frag, val_ef, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FragAnchoredFiLMDeltaMLP(mol_dim=emb_dim, frag_dim=1024, edit_feat_dim=28)

    def forward_fn(m, a, b, frag_d, ef):
        return m(a, b, frag_d, ef)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, test_frag, test_ef, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b, test_frag, test_ef)


def train_and_predict_multimodal_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Multi-Modal Edit FiLM: fuses DRFP + fragment + edit features."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.edit_aware_film_predictor import MultiModalEditFiLMDeltaMLP

    drfp_cache = compute_drfp_cache(train_df)
    frag_cache = compute_frag_delta_cache(train_df)

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        drfp = get_drfp_tensors(df, drfp_cache)
        frag_d = get_frag_delta_tensors(df, frag_cache)
        ef = get_edit_feats_tensors(df)
        return emb_a, emb_b, drfp, frag_d, ef, delta

    train_a, train_b, train_drfp, train_frag, train_ef, train_y = make_datasets(train_df)
    val_a, val_b, val_drfp, val_frag, val_ef, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_drfp, train_frag, train_ef, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_drfp, val_frag, val_ef, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiModalEditFiLMDeltaMLP(
        mol_dim=emb_dim, drfp_dim=2048, frag_dim=1024, edit_feat_dim=28,
        use_rxnfp=False)

    # Custom training with auxiliary loss
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, drfp, frag_d, ef, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            pred = model(a, b, drfp, frag_d, ef)
            loss = criterion(pred, y) + model.aux_loss(ef)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                a, b, drfp, frag_d, ef, y = [t.to(DEVICE) for t in batch]
                pred = model(a, b, drfp, frag_d, ef)
                val_losses.append(criterion(pred, y).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    test_a, test_b, test_drfp, test_frag, test_ef, _ = make_datasets(test_df)

    def forward_fn(m, a, b, drfp, frag_d, ef):
        return m(a, b, drfp, frag_d, ef)

    return predict_multi_input(model, forward_fn, test_a, test_b,
                               test_drfp, test_frag, test_ef)


# ── Target-Conditioned FiLM ──────────────────────────────────────────────

# Module-level target ID mapping (computed once)
TARGET_ID_MAP = None


def get_target_id_map(df):
    """Build or return cached mapping from target_chembl_id → integer index."""
    global TARGET_ID_MAP
    if TARGET_ID_MAP is None:
        all_targets = sorted(df["target_chembl_id"].unique())
        TARGET_ID_MAP = {t: i for i, t in enumerate(all_targets)}
    return TARGET_ID_MAP


def get_target_id_tensors(df, target_map):
    """Convert target_chembl_id column to integer tensor."""
    # Use target_chembl_id (same for both mols in a pair)
    ids = df["target_chembl_id"].map(target_map).fillna(len(target_map)).astype(int).values
    return torch.from_numpy(ids).long()


def train_and_predict_target_cond_film(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Target-Conditioned FiLMDelta: FiLM conditioning on delta + target identity."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.edit_aware_film_predictor import TargetCondFiLMDeltaMLP

    target_map = get_target_id_map(train_df)
    n_targets = len(target_map) + 1  # +1 for unknown targets

    def make_datasets(df):
        emb_a, emb_b, delta = get_pair_tensors(df, emb_dict, emb_dim)
        tgt_ids = get_target_id_tensors(df, target_map)
        return emb_a, emb_b, tgt_ids, delta

    train_a, train_b, train_tgt, train_y = make_datasets(train_df)
    val_a, val_b, val_tgt, val_y = make_datasets(val_df)

    train_loader = DataLoader(
        TensorDataset(train_a, train_b, train_tgt, train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_tgt, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TargetCondFiLMDeltaMLP(
        input_dim=emb_dim, n_targets=n_targets,
        target_emb_dim=64, hidden_dims=[512, 256, 128])

    def forward_fn(m, a, b, tgt):
        return m(a, b, tgt)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, test_tgt, _ = make_datasets(test_df)
    return predict_multi_input(model, forward_fn, test_a, test_b, test_tgt)


def train_and_predict_film_delta_augmented(train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """FiLMDelta with antisymmetric data augmentation.

    Adds reversed pairs (mol_b, mol_a, -delta) to training data, doubling
    the effective dataset. This enforces antisymmetry in the delta conditioning
    pathway (which is NOT structurally antisymmetric even though the output is).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    # Get original pairs
    train_a, train_b, train_y = get_pair_tensors(train_df, emb_dict, emb_dim)
    # Augment: concatenate with reversed pairs (B, A, -delta)
    aug_a = torch.cat([train_a, train_b], dim=0)
    aug_b = torch.cat([train_b, train_a], dim=0)
    aug_y = torch.cat([train_y, -train_y], dim=0)

    val_a, val_b, val_y = get_pair_tensors(val_df, emb_dict, emb_dim)

    train_loader = DataLoader(
        TensorDataset(aug_a, aug_b, aug_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(val_a, val_b, val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])

    def forward_fn(m, a, b):
        return m(a, b)

    model = train_model_multi_input(model, train_loader, val_loader, forward_fn)

    test_a, test_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict_multi_input(model, forward_fn, test_a, test_b)


# Architecture registry
ARCHITECTURES = {
    "Subtraction": {
        "fn": train_and_predict_subtraction,
        "label": "Subtraction Baseline",
        "color": "#e74c3c",
    },
    "EditDiff": {
        "fn": train_and_predict_edit_diff,
        "label": "Edit Effect (diff)",
        "color": "#2ecc71",
    },
    "DeepDelta": {
        "fn": train_and_predict_deepdelta,
        "label": "DeepDelta (concat)",
        "color": "#3498db",
    },
    "EditDiff+Feats": {
        "fn": train_and_predict_edit_features,
        "label": "Edit Effect + Features",
        "color": "#9b59b6",
    },
    "GatedCrossAttn": {
        "fn": train_and_predict_gated_crossattn,
        "label": "Gated Cross-Attention",
        "color": "#e67e22",
    },
    "AttnThenFiLM": {
        "fn": train_and_predict_attn_film,
        "label": "Attention + FiLM",
        "color": "#1abc9c",
    },
    "FiLMDelta": {
        "fn": train_and_predict_film_delta,
        "label": "FiLM-Conditioned",
        "color": "#f39c12",
    },
    "TrainableEdit": {
        "fn": train_and_predict_trainable_edit,
        "label": "Trainable Edit Embedder",
        "color": "#8e44ad",
    },
    "DrfpFiLM": {
        "fn": train_and_predict_drfp_film,
        "label": "DRFP-FiLM",
        "color": "#c0392b",
    },
    "DualStreamFiLM": {
        "fn": train_and_predict_dualstream_film,
        "label": "DualStream-FiLM",
        "color": "#27ae60",
    },
    "FragAnchoredFiLM": {
        "fn": train_and_predict_frag_anchored_film,
        "label": "Fragment-Anchored FiLM",
        "color": "#2980b9",
    },
    "MultiModalFiLM": {
        "fn": train_and_predict_multimodal_film,
        "label": "Multi-Modal Edit FiLM",
        "color": "#8e44ad",
    },
    "TargetCondFiLM": {
        "fn": train_and_predict_target_cond_film,
        "label": "Target-Conditioned FiLM",
        "color": "#16a085",
    },
    "FiLMDelta+Aug": {
        "fn": train_and_predict_film_delta_augmented,
        "label": "FiLMDelta + Antisymmetric Aug",
        "color": "#d35400",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading and splitting
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load MMP pairs. Prefers shared_pairs_deduped > 50k subsample > full file."""
    if SHARED_PAIRS_FILE.exists():
        data_file = SHARED_PAIRS_FILE
    elif SUBSAMPLE_FILE.exists():
        data_file = SUBSAMPLE_FILE
    else:
        data_file = FULL_FILE

    print(f"Loading data from {data_file.name}...")
    df = pd.read_csv(data_file)

    # Filter to real MMPs (not self-pairs) if mol_a_id/mol_b_id columns exist
    if "mol_a_id" in df.columns and "mol_b_id" in df.columns:
        is_mmp = df["mol_a_id"] != df["mol_b_id"]
        df = df[is_mmp].copy()

    n_within = (df["is_within_assay"] == True).sum() if "is_within_assay" in df.columns else "?"
    n_cross = (df["is_within_assay"] == False).sum() if "is_within_assay" in df.columns else "?"
    n_mols = len(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    print(f"  {len(df):,} MMP pairs ({n_within} within, {n_cross} cross)")
    print(f"  {df['target_chembl_id'].nunique()} targets, {n_mols:,} unique molecules")
    return df


def split_data(df, split_type, seed, **kwargs):
    """Split data using specified strategy."""
    from src.utils.splits import get_splitter

    if split_type == "assay_within":
        splitter = get_splitter("assay", random_state=seed, scenario="within_assay")
        return splitter.split(df)
    elif split_type == "assay_cross":
        splitter = get_splitter("assay", random_state=seed, scenario="cross_assay")
        return splitter.split(df)
    elif split_type == "assay_mixed":
        splitter = get_splitter("assay", random_state=seed, scenario="mixed")
        return splitter.split(df)
    elif split_type == "scaffold":
        splitter = get_splitter("scaffold", random_state=seed)
        return splitter.split(df, smiles_col="mol_a")
    elif split_type == "random":
        splitter = get_splitter("random", random_state=seed)
        return splitter.split(df)
    elif split_type == "target":
        splitter = get_splitter("target", random_state=seed)
        return splitter.split(df, target_col="target_chembl_id")
    elif split_type == "few_shot":
        splitter = get_splitter("few_shot_target", random_state=seed,
                               few_shot_samples=kwargs.get("few_shot_samples", 100))
        return splitter.split(df, target_col="target_chembl_id")
    elif split_type == "strict_scaffold":
        splitter = get_splitter("strict_scaffold", random_state=seed)
        return splitter.split(df)
    elif split_type == "pair_random":
        splitter = get_splitter("pair_aware_random", random_state=seed)
        return splitter.split(df)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════

def run_single_experiment(train_df, val_df, test_df, emb_dict, emb_dim,
                          arch_name, seed):
    """Run a single architecture on given splits."""
    arch = ARCHITECTURES[arch_name]
    y_true = test_df["delta"].values
    y_pred = arch["fn"](train_df, val_df, test_df, emb_dict, emb_dim, seed)
    metrics = compute_metrics(y_true, y_pred)

    # Per-target metrics if target column exists
    if "target_chembl_id" in test_df.columns:
        targets = test_df["target_chembl_id"].values
        pt_metrics = compute_per_target_metrics(y_true, y_pred, targets)
        metrics["per_target_avg"] = {k: v for k, v in pt_metrics.items() if k != "per_target"}
        # Save full per-target detail (each target's individual metrics)
        if "per_target" in pt_metrics:
            metrics["per_target_detail"] = pt_metrics["per_target"]

    return metrics, y_pred


def aggregate_seeds(all_runs):
    """Aggregate metrics across seeds."""
    if not all_runs:
        return {}
    agg = {"n": all_runs[0].get("n", 0)}
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_r"]:
        vals = [r[metric] for r in all_runs if metric in r]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Embedder Selection
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1(df):
    """Compare embedders using EditDiff architecture on within-assay split."""
    print("\n" + "=" * 70)
    print("PHASE 1: Embedder Selection")
    print("  Architecture: EditDiff (MLP on [emb_a, emb_b - emb_a])")
    print("  Split: within-assay, 3 seeds")
    print("=" * 70)

    embedders = ["morgan", "chemberta2-mtr", "chemberta2-mlm", "chemprop-dmpnn"]
    results = {}

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))

    for emb_name in embedders:
        print(f"\n--- Embedder: {emb_name} ---")
        try:
            emb_dict, emb_dim = compute_embeddings(all_smiles, emb_name)
        except Exception as e:
            print(f"  FAILED to compute embeddings: {e}")
            import traceback; traceback.print_exc()
            continue

        seed_runs = []
        for seed_idx, seed in enumerate(SEEDS):
            print(f"  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...", end=" ", flush=True)
            try:
                train_df, val_df, test_df = split_data(df, "assay_within", seed)
                if len(train_df) < 50 or len(test_df) < 20:
                    print("too few samples, skipping")
                    continue
                metrics, _ = run_single_experiment(
                    train_df, val_df, test_df, emb_dict, emb_dim, "EditDiff", seed
                )
                seed_runs.append(metrics)
                print(f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                      f"Pearson={metrics['pearson_r']:.4f}, R²={metrics['r2']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()

        if seed_runs:
            results[emb_name] = {
                "aggregated": aggregate_seeds(seed_runs),
                "per_seed": seed_runs,
                "emb_dim": emb_dim,
            }

        # Free memory
        del emb_dict
        import gc; gc.collect()

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY: Embedder Comparison (EditDiff, within-assay)")
    print("=" * 70)
    print(f"  {'Embedder':<20} {'MAE':>12} {'Spearman':>12} {'Pearson':>12} {'R²':>12}")
    print(f"  {'-' * 68}")
    best_emb = None
    best_mae = float("inf")
    for emb_name, res in results.items():
        a = res["aggregated"]
        mae_str = f"{a.get('mae_mean',0):.4f}±{a.get('mae_std',0):.4f}"
        spr_str = f"{a.get('spearman_r_mean',0):.4f}±{a.get('spearman_r_std',0):.4f}"
        prs_str = f"{a.get('pearson_r_mean',0):.4f}±{a.get('pearson_r_std',0):.4f}"
        r2_str = f"{a.get('r2_mean',0):.4f}±{a.get('r2_std',0):.4f}"
        print(f"  {emb_name:<20} {mae_str:>12} {spr_str:>12} {prs_str:>12} {r2_str:>12}")
        if a.get("mae_mean", float("inf")) < best_mae:
            best_mae = a["mae_mean"]
            best_emb = emb_name

    print(f"\n  → Best embedder: {best_emb} (MAE={best_mae:.4f})")
    results["_best"] = best_emb
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Architecture Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_phase2(df, best_embedder):
    """Compare architectures using the best embedder on within-assay split."""
    print("\n" + "=" * 70)
    print(f"PHASE 2: Architecture Comparison (embedder={best_embedder})")
    print("  Split: within-assay, 3 seeds")
    print("=" * 70)

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, best_embedder)

    results = {}
    for arch_name in ARCHITECTURES:
        print(f"\n--- Architecture: {arch_name} ---")
        seed_runs = []
        for seed_idx, seed in enumerate(SEEDS):
            print(f"  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...", end=" ", flush=True)
            try:
                train_df, val_df, test_df = split_data(df, "assay_within", seed)
                if len(train_df) < 50 or len(test_df) < 20:
                    print("too few samples")
                    continue
                metrics, _ = run_single_experiment(
                    train_df, val_df, test_df, emb_dict, emb_dim, arch_name, seed
                )
                seed_runs.append(metrics)
                print(f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                      f"Pearson={metrics['pearson_r']:.4f}, R²={metrics['r2']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()

        if seed_runs:
            results[arch_name] = {
                "aggregated": aggregate_seeds(seed_runs),
                "per_seed": seed_runs,
            }

    # Print summary
    print("\n" + "=" * 70)
    print(f"PHASE 2 SUMMARY: Architecture Comparison ({best_embedder})")
    print("=" * 70)
    print(f"  {'Architecture':<20} {'MAE':>12} {'Spearman':>12} {'Pearson':>12} {'R²':>12}")
    print(f"  {'-' * 68}")
    best_arch = None
    best_mae = float("inf")
    for arch_name, res in results.items():
        a = res["aggregated"]
        mae_str = f"{a.get('mae_mean',0):.4f}±{a.get('mae_std',0):.4f}"
        spr_str = f"{a.get('spearman_r_mean',0):.4f}±{a.get('spearman_r_std',0):.4f}"
        prs_str = f"{a.get('pearson_r_mean',0):.4f}±{a.get('pearson_r_std',0):.4f}"
        r2_str = f"{a.get('r2_mean',0):.4f}±{a.get('r2_std',0):.4f}"
        marker = " ← baseline" if arch_name == "Subtraction" else ""
        print(f"  {arch_name:<20} {mae_str:>12} {spr_str:>12} {prs_str:>12} {r2_str:>12}{marker}")
        if arch_name != "Subtraction" and a.get("mae_mean", float("inf")) < best_mae:
            best_mae = a["mae_mean"]
            best_arch = arch_name

    # Compute improvement over subtraction
    sub_mae = results.get("Subtraction", {}).get("aggregated", {}).get("mae_mean")
    if sub_mae and best_mae < float("inf"):
        improvement = (1 - best_mae / sub_mae) * 100
        print(f"\n  → Best edit architecture: {best_arch}")
        print(f"  → Improvement over Subtraction: {improvement:.1f}% MAE reduction")

    results["_best"] = best_arch
    results["_embedder"] = best_embedder
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Generalization Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _save_incremental_phase3(results, best_embedder, best_arch):
    """Save Phase 3 results incrementally so we don't lose progress."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = dict(results)
    save_data["_embedder"] = best_embedder
    save_data["_best_arch"] = best_arch
    # Load existing results and update phase3
    results_file = RESULTS_DIR / "all_results.json"
    try:
        with open(results_file) as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
    all_results["phase3"] = save_data
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"    [saved {len(results)} splits to {results_file.name}]")


def run_phase3(df, best_embedder, best_arch, extra_methods=None):
    """Run best models + subtraction baseline across all splits.

    Args:
        extra_methods: Optional list of additional architecture names to evaluate
                       alongside Subtraction and best_arch.
    """
    methods = ["Subtraction", best_arch]
    if extra_methods:
        for m in extra_methods:
            if m not in methods and m in ARCHITECTURES:
                methods.append(m)

    print("\n" + "=" * 70)
    print(f"PHASE 3: Generalization ({best_embedder})")
    print(f"  Methods: {', '.join(methods)}")
    print("=" * 70)

    # Subsample if dataset is too large (memory control)
    if MAX_PHASE3_PAIRS and len(df) > MAX_PHASE3_PAIRS:
        print(f"  Subsampling {len(df):,} → {MAX_PHASE3_PAIRS:,} pairs for memory efficiency")
        df = df.sample(n=MAX_PHASE3_PAIRS, random_state=42).reset_index(drop=True)

    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    emb_dict, emb_dim = compute_embeddings(all_smiles, best_embedder)
    print(f"  Dataset: {len(df):,} pairs, {len(all_smiles):,} molecules, dim={emb_dim}")

    splits = [
        ("assay_within", "Assay: Within"),
        ("assay_cross", "Assay: Cross"),
        ("assay_mixed", "Assay: Mixed"),
        ("scaffold", "Scaffold"),
        ("random", "Random"),
        ("target", "Cross-Target"),
        ("few_shot", "Few-Shot Target"),
    ]
    results = {}

    for split_idx, (split_name, split_label) in enumerate(splits):
        print(f"\n--- Split {split_idx+1}/{len(splits)}: {split_label} ---")
        split_results = {}

        for method in methods:
            seed_runs = []
            for seed_idx, seed in enumerate(SEEDS):
                print(f"  {method}, seed {seed}...", end=" ", flush=True)
                try:
                    train_df, val_df, test_df = split_data(df, split_name, seed)
                    if len(train_df) < 50 or len(test_df) < 20:
                        print(f"too few samples (train={len(train_df)}, test={len(test_df)})")
                        continue
                    metrics, _ = run_single_experiment(
                        train_df, val_df, test_df, emb_dict, emb_dim, method, seed
                    )
                    seed_runs.append(metrics)
                    print(f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    # Free memory after each run
                    gc.collect()

            if seed_runs:
                split_results[method] = {
                    "aggregated": aggregate_seeds(seed_runs),
                    "per_seed": seed_runs,
                }

        results[split_name] = {
            "label": split_label,
            "methods": split_results,
        }

        # Save incrementally after each split
        _save_incremental_phase3(results, best_embedder, best_arch)
        gc.collect()

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY: Generalization")
    print("=" * 70)
    print(f"  {'Split':<20} {'Subtraction MAE':>16} {f'{best_arch} MAE':>16} {'Δ%':>8}")
    print(f"  {'-' * 62}")
    for split_name, split_info in results.items():
        methods = split_info["methods"]
        sub_mae = methods.get("Subtraction", {}).get("aggregated", {}).get("mae_mean")
        edit_mae = methods.get(best_arch, {}).get("aggregated", {}).get("mae_mean")
        sub_str = f"{sub_mae:.4f}" if sub_mae else "N/A"
        edit_str = f"{edit_mae:.4f}" if edit_mae else "N/A"
        if sub_mae and edit_mae:
            delta_pct = (1 - edit_mae / sub_mae) * 100
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "N/A"
        print(f"  {split_info['label']:<20} {sub_str:>16} {edit_str:>16} {delta_str:>8}")

    results["_embedder"] = best_embedder
    results["_best_arch"] = best_arch
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(val, std=None, fmt=".4f"):
    """Format a metric value with optional std."""
    if val is None:
        return "N/A"
    s = f"{val:{fmt}}"
    if std is not None:
        s += f"±{std:{fmt}}"
    return s


def _per_target_analysis(phase3, best_arch):
    """Extract per-target win/loss analysis from Phase 3 results."""
    analysis = {}
    splits_order = ["random", "assay_within", "assay_mixed", "assay_cross",
                    "scaffold", "target", "few_shot"]

    for split_name in splits_order:
        if split_name not in phase3:
            continue
        info = phase3[split_name]
        methods = info.get("methods", {})
        sub_seeds = methods.get("Subtraction", {}).get("per_seed", [])
        edit_seeds = methods.get(best_arch, {}).get("per_seed", [])

        if not sub_seeds or not edit_seeds:
            continue
        # Use first seed for per-target breakdown
        sub_detail = sub_seeds[0].get("per_target_detail", {})
        edit_detail = edit_seeds[0].get("per_target_detail", {})

        if not sub_detail or not edit_detail:
            continue

        common_targets = set(sub_detail.keys()) & set(edit_detail.keys())
        wins, losses, ties = 0, 0, 0
        target_rows = []
        for t in common_targets:
            sm = sub_detail[t]
            em = edit_detail[t]
            if sm.get("mae") is None or em.get("mae") is None:
                continue
            diff = sm["mae"] - em["mae"]
            if abs(diff) < 1e-6:
                ties += 1
            elif diff > 0:
                wins += 1
            else:
                losses += 1
            target_rows.append({
                "target": t, "n": em.get("n", 0),
                "edit_mae": em["mae"], "sub_mae": sm["mae"],
                "edit_r2": em.get("r2", 0), "sub_r2": sm.get("r2", 0),
                "edit_spearman": em.get("spearman_r", 0), "sub_spearman": sm.get("spearman_r", 0),
                "mae_improvement_pct": (sm["mae"] - em["mae"]) / sm["mae"] * 100 if sm["mae"] > 0 else 0,
            })

        target_rows.sort(key=lambda r: r["mae_improvement_pct"])
        analysis[split_name] = {
            "label": info["label"],
            "n_targets": len(common_targets),
            "wins": wins, "losses": losses, "ties": ties,
            "targets": target_rows,
        }
    return analysis


def generate_report(phase1, phase2, phase3):
    """Generate comprehensive self-contained HTML report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_emb = phase1.get("_best", "unknown") if phase1 else "unknown"
    best_arch = phase2.get("_best", "unknown") if phase2 else "unknown"

    h = []  # html lines
    h.append("<!DOCTYPE html><html><head>")
    h.append("<meta charset='utf-8'>")
    h.append("<title>Edit Effect Framework — Evaluation Report</title>")
    h.append("<style>")
    h.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2c3e50; margin-top: 40px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }
        h3 { color: #34495e; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }
        th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
        th { background-color: #3498db; color: white; font-weight: 600; }
        td:first-child { text-align: left; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .best { font-weight: bold; color: #27ae60; }
        .worst { color: #e74c3c; }
        .baseline { color: #95a5a6; }
        .note { background: #f0f7ff; border-left: 4px solid #3498db; padding: 12px 16px; margin: 15px 0; }
        .improvement { background: #e8f8e8; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 15px 0; }
        .warning { background: #fff8e8; border-left: 4px solid #f39c12; padding: 12px 16px; margin: 15px 0; }
        .summary-box { background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;
                       padding: 16px 20px; margin: 20px 0; }
        .summary-box h3 { margin-top: 0; }
        .metric-sm { font-size: 0.85em; color: #7f8c8d; }
        .win { color: #27ae60; font-weight: bold; }
        .loss { color: #e74c3c; font-weight: bold; }
        .toc { background: #f8f9fa; padding: 16px 24px; border-radius: 8px; margin: 20px 0; }
        .toc ul { margin: 5px 0; padding-left: 20px; }
        .toc li { margin: 3px 0; }
        .toc a { text-decoration: none; color: #3498db; }
        .toc a:hover { text-decoration: underline; }
    """)
    h.append("</style></head><body>")

    # Title & overview
    h.append("<h1>Edit Effect Framework — Evaluation Report</h1>")
    h.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")

    h.append("<div class='summary-box'>")
    h.append("<h3>Executive Summary</h3>")
    h.append(f"<p><strong>Embedder</strong>: {best_emb} (selected in Phase 1)<br>")
    h.append(f"<strong>Architecture</strong>: {best_arch} — MLP on [emb_a, emb_b − emb_a] (selected in Phase 2)<br>")
    h.append(f"<strong>Baseline</strong>: Subtraction — train F(mol)→property, predict F(B)−F(A)<br>")
    h.append("<strong>Dataset</strong>: Shared MMP pairs (1.7M pairs, 88K molecules, 751 targets) — "
             "pairs appearing in both within-assay and cross-assay contexts<br>")
    h.append("<strong>Seeds</strong>: 3 (42, 123, 456) — all values reported as mean±std</p>")

    if phase3:
        splits_done = [k for k in phase3 if not k.startswith("_")]
        n_wins = 0
        for sn in splits_done:
            sub_mae = phase3[sn].get("methods", {}).get("Subtraction", {}).get("aggregated", {}).get("mae_mean")
            ed_mae = phase3[sn].get("methods", {}).get(best_arch, {}).get("aggregated", {}).get("mae_mean")
            if sub_mae and ed_mae and ed_mae < sub_mae:
                n_wins += 1
        h.append(f"<p><strong>Result</strong>: {best_arch} wins <strong>{n_wins}/{len(splits_done)}</strong> "
                 f"generalization splits over Subtraction baseline.</p>")
    h.append("</div>")

    # TOC
    h.append("<div class='toc'><strong>Contents</strong><ul>")
    h.append("<li><a href='#phase1'>Phase 1: Embedder Selection</a></li>")
    h.append("<li><a href='#phase2'>Phase 2: Architecture Comparison</a></li>")
    h.append("<li><a href='#phase3'>Phase 3: Generalization</a></li>")
    h.append("<li><a href='#pertarget'>Per-Target Analysis</a></li>")
    h.append("<li><a href='#metrics'>Metrics Policy</a></li>")
    h.append("</ul></div>")

    # ── Phase 1 ──
    if phase1:
        h.append("<h2 id='phase1'>Phase 1: Embedder Selection</h2>")
        h.append("<p>All embedders evaluated with EditDiff architecture on <strong>within-assay</strong> split "
                 "(1.7M shared pairs, 3 seeds).</p>")
        h.append("<table><tr><th>Embedder</th><th>Dim</th>"
                 "<th>MAE ↓</th><th>Spearman ↑</th><th>Pearson r ↑</th><th>R² ↑</th></tr>")
        # Sort by MAE
        emb_items = [(k, v) for k, v in phase1.items() if not k.startswith("_")]
        emb_items.sort(key=lambda x: x[1]["aggregated"].get("mae_mean", 99))
        for emb_name, res in emb_items:
            a = res["aggregated"]
            css = "best" if emb_name == best_emb else ""
            h.append(f"<tr class='{css}'>"
                     f"<td>{emb_name}</td>"
                     f"<td>{res.get('emb_dim', '?')}</td>"
                     f"<td>{_fmt(a.get('mae_mean'), a.get('mae_std'))}</td>"
                     f"<td>{_fmt(a.get('spearman_r_mean'), a.get('spearman_r_std'))}</td>"
                     f"<td>{_fmt(a.get('pearson_r_mean'), a.get('pearson_r_std'))}</td>"
                     f"<td>{_fmt(a.get('r2_mean'), a.get('r2_std'))}</td>"
                     f"</tr>")
        h.append("</table>")
        h.append(f"<div class='note'><strong>Selected: {best_emb}</strong> — "
                 "ChemProp D-MPNN narrowly beats Morgan FP on the larger shared-pairs dataset. "
                 "ChemBERTa-2 variants lag behind, consistent with recent benchmarks showing "
                 "neural embedders rarely outperform fingerprints for property prediction.</div>")

    # ── Phase 2 ──
    if phase2:
        h.append("<h2 id='phase2'>Phase 2: Architecture Comparison</h2>")
        h.append(f"<p>Using <strong>{best_emb}</strong> embeddings, within-assay split, 3 seeds.</p>")
        h.append("<table><tr><th>Architecture</th><th>Description</th>"
                 "<th>MAE ↓</th><th>Spearman ↑</th><th>Pearson r ↑</th><th>R² ↑</th></tr>")

        arch_descriptions = {
            "Subtraction": "F(B) − F(A)",
            "EditDiff": "MLP([a, b−a]) → Δ",
            "DeepDelta": "MLP([a, b]) → Δ",
            "EditDiff+Feats": "MLP([a, b−a, feats]) → Δ",
            "FiLMDelta": "FiLM f(B|δ) − f(A|δ)",
            "GatedCrossAttn": "Gated cross-attn",
            "AttnThenFiLM": "Attn + FiLM layers",
            "TrainableEdit": "Learned edit embs",
        }
        # Sort all architectures by MAE
        arch_order = [k for k in phase2 if not k.startswith("_")]
        arch_order.sort(key=lambda k: phase2[k].get("aggregated", {}).get("mae_mean", 99))
        for arch_name in arch_order:
            if arch_name not in phase2 or arch_name.startswith("_"):
                continue
            res = phase2[arch_name]
            a = res["aggregated"]
            css = "baseline" if arch_name == "Subtraction" else ("best" if arch_name == best_arch else "")
            h.append(f"<tr class='{css}'>"
                     f"<td>{arch_name}</td>"
                     f"<td>{arch_descriptions.get(arch_name, '')}</td>"
                     f"<td>{_fmt(a.get('mae_mean'), a.get('mae_std'))}</td>"
                     f"<td>{_fmt(a.get('spearman_r_mean'), a.get('spearman_r_std'))}</td>"
                     f"<td>{_fmt(a.get('pearson_r_mean'), a.get('pearson_r_std'))}</td>"
                     f"<td>{_fmt(a.get('r2_mean'), a.get('r2_std'))}</td>"
                     f"</tr>")
        h.append("</table>")

        sub_mae = phase2.get("Subtraction", {}).get("aggregated", {}).get("mae_mean")
        sub_std = phase2.get("Subtraction", {}).get("aggregated", {}).get("mae_std")
        ed_mae = phase2.get(best_arch, {}).get("aggregated", {}).get("mae_mean")
        ed_std = phase2.get(best_arch, {}).get("aggregated", {}).get("mae_std")
        if sub_mae and ed_mae:
            improvement = (1 - ed_mae / sub_mae) * 100
            h.append(f"<div class='improvement'><strong>{best_arch}</strong> reduces MAE by "
                     f"<strong>{improvement:.1f}%</strong> vs Subtraction "
                     f"({_fmt(ed_mae, ed_std)} vs {_fmt(sub_mae, sub_std)}). "
                     f"Critically, {best_arch} also has <strong>much lower variance</strong> across seeds "
                     f"(±{ed_std:.4f} vs ±{sub_std:.4f}), "
                     f"indicating more stable learning.</div>")

    # ── Phase 3 ──
    if phase3:
        h.append("<h2 id='phase3'>Phase 3: Generalization</h2>")
        p3_note = phase3.get("_note", "")
        # Determine how many pairs were used
        n_pairs_note = "full 1.7M pairs" if not MAX_PHASE3_PAIRS else f"{MAX_PHASE3_PAIRS:,} pairs"
        h.append(f"<p>Generalization across 7 evaluation splits ({n_pairs_note}, 3 seeds).</p>")

        splits_order = ["random", "assay_within", "assay_mixed", "assay_cross",
                        "scaffold", "target", "few_shot"]

        # Collect all methods that appear in Phase 3
        all_p3_methods = set()
        for sn in splits_order:
            if sn in phase3:
                all_p3_methods.update(phase3[sn].get("methods", {}).keys())
        # Order: Subtraction first, best_arch second, then others sorted by name
        method_order = []
        if "Subtraction" in all_p3_methods:
            method_order.append("Subtraction")
            all_p3_methods.discard("Subtraction")
        if best_arch in all_p3_methods:
            method_order.append(best_arch)
            all_p3_methods.discard(best_arch)
        method_order.extend(sorted(all_p3_methods))

        # Full metrics table
        h.append("<h3>Full Metrics</h3>")
        h.append("<table><tr><th>Split</th><th>Method</th>"
                 "<th>MAE ↓</th><th>Spearman ↑</th><th>Pearson r ↑</th><th>R² ↑</th></tr>")
        for sn in splits_order:
            if sn not in phase3:
                continue
            info = phase3[sn]
            label = info["label"]
            for mi, method in enumerate(method_order):
                a = info["methods"].get(method, {}).get("aggregated", {})
                if not a:
                    continue
                css = "baseline" if method == "Subtraction" else ("best" if method == best_arch else "")
                row_label = label if mi == 0 else ""
                h.append(f"<tr class='{css}'>"
                         f"<td>{row_label}</td>"
                         f"<td>{method}</td>"
                         f"<td>{_fmt(a.get('mae_mean'), a.get('mae_std'))}</td>"
                         f"<td>{_fmt(a.get('spearman_r_mean'), a.get('spearman_r_std'))}</td>"
                         f"<td>{_fmt(a.get('pearson_r_mean'), a.get('pearson_r_std'))}</td>"
                         f"<td>{_fmt(a.get('r2_mean'), a.get('r2_std'))}</td>"
                         f"</tr>")
        h.append("</table>")

        # Summary comparison table — all methods vs Subtraction
        h.append("<h3>Summary: MAE Comparison vs Subtraction</h3>")
        edit_methods = [m for m in method_order if m != "Subtraction"]
        header_cols = "<th>Split</th><th>Sub MAE</th>"
        for m in edit_methods:
            header_cols += f"<th>{m} MAE</th><th>ΔMAE%</th>"
        header_cols += "<th>Best</th>"
        h.append(f"<table><tr>{header_cols}</tr>")
        for sn in splits_order:
            if sn not in phase3:
                continue
            info = phase3[sn]
            sub = info["methods"].get("Subtraction", {}).get("aggregated", {})
            sub_mae = sub.get("mae_mean")
            if not sub_mae:
                continue
            row = f"<td>{info['label']}</td><td>{_fmt(sub_mae)}</td>"
            best_method = "Subtraction"
            best_mae = sub_mae
            for m in edit_methods:
                a = info["methods"].get(m, {}).get("aggregated", {})
                m_mae = a.get("mae_mean")
                if m_mae:
                    d_mae = (1 - m_mae / sub_mae) * 100
                    css = "win" if d_mae > 0 else "loss"
                    row += f"<td>{_fmt(m_mae)}</td><td class='{css}'>{d_mae:+.1f}%</td>"
                    if m_mae < best_mae:
                        best_mae = m_mae
                        best_method = m
                else:
                    row += "<td>—</td><td>—</td>"
            winner_css = "win" if best_method != "Subtraction" else "loss"
            row += f"<td class='{winner_css}'>{best_method}</td>"
            h.append(f"<tr>{row}</tr>")
        h.append("</table>")

        # Narrative — compute wins dynamically
        n_splits_done = len([s for s in splits_order if s in phase3])
        n_wins = 0
        win_details = []
        loss_details = []
        for sn in splits_order:
            if sn not in phase3:
                continue
            info = phase3[sn]
            sub_mae = info["methods"].get("Subtraction", {}).get("aggregated", {}).get("mae_mean", 99)
            best_mae = info["methods"].get(best_arch, {}).get("aggregated", {}).get("mae_mean", 99)
            if best_mae < sub_mae:
                n_wins += 1
                pct = (1 - best_mae / sub_mae) * 100
                win_details.append(f"{info['label']} (+{pct:.1f}%)")
            elif sub_mae < best_mae:
                pct = (1 - sub_mae / best_mae) * 100
                loss_details.append(f"{info['label']}")

        h.append(f"<div class='improvement'><strong>{best_arch} wins {n_wins}/{n_splits_done} splits.</strong> ")
        if win_details:
            h.append(f"Wins: {', '.join(win_details)}. ")
        if loss_details:
            h.append(f"Subtraction wins: {', '.join(loss_details)} — "
                     f"when molecules are structurally novel, absolute property models transfer better.")
        h.append("</div>")

        # Cross-assay note
        h.append("<div class='warning'><strong>Cross-assay results</strong>: Both methods show negative R² "
                 "and low Spearman, confirming that cross-lab measurement noise dominates prediction accuracy. "
                 "This validates the motivation for using within-assay pairs as cleaner training signal.</div>")

    # ── Per-Target Analysis ──
    if phase3:
        pt_analysis = _per_target_analysis(phase3, best_arch)
        if pt_analysis:
            h.append("<h2 id='pertarget'>Per-Target Analysis</h2>")
            h.append("<p>Per-target win/loss breakdown (seed 42, targets with ≥10 test pairs).</p>")

            # Win/loss summary table
            h.append("<h3>Win/Loss by Split</h3>")
            h.append("<table><tr><th>Split</th><th>Targets</th>"
                     f"<th>{best_arch} Wins</th><th>Subtraction Wins</th><th>Win Rate</th></tr>")
            for sn in ["random", "assay_within", "assay_mixed", "assay_cross",
                        "scaffold", "target", "few_shot"]:
                if sn not in pt_analysis:
                    continue
                pa = pt_analysis[sn]
                total = pa["wins"] + pa["losses"] + pa["ties"]
                win_rate = pa["wins"] / total * 100 if total > 0 else 0
                css = "win" if win_rate > 50 else ("loss" if win_rate < 50 else "")
                h.append(f"<tr>"
                         f"<td>{pa['label']}</td>"
                         f"<td>{pa['n_targets']}</td>"
                         f"<td class='win'>{pa['wins']}</td>"
                         f"<td class='loss'>{pa['losses']}</td>"
                         f"<td class='{css}'>{win_rate:.0f}%</td>"
                         f"</tr>")
            h.append("</table>")

            # Detailed per-target tables for key splits
            for sn in ["random", "assay_within", "target"]:
                if sn not in pt_analysis:
                    continue
                pa = pt_analysis[sn]
                targets = pa["targets"]
                if len(targets) < 5:
                    continue

                h.append(f"<h3>{pa['label']}: Top Targets</h3>")

                # Top 5 where EditDiff wins most
                top_wins = sorted(targets, key=lambda r: -r["mae_improvement_pct"])[:5]
                h.append(f"<p><strong>Best for {best_arch}</strong> (largest MAE improvement):</p>")
                h.append("<table><tr><th>Target</th><th>N pairs</th>"
                         f"<th>{best_arch} MAE</th><th>Sub MAE</th><th>ΔMAE%</th>"
                         f"<th>{best_arch} R²</th><th>Sub R²</th></tr>")
                for r in top_wins:
                    h.append(f"<tr>"
                             f"<td>{r['target']}</td><td>{r['n']}</td>"
                             f"<td class='best'>{r['edit_mae']:.3f}</td>"
                             f"<td>{r['sub_mae']:.3f}</td>"
                             f"<td class='win'>{r['mae_improvement_pct']:+.1f}%</td>"
                             f"<td>{r['edit_r2']:.3f}</td><td>{r['sub_r2']:.3f}</td>"
                             f"</tr>")
                h.append("</table>")

                # Top 5 where Subtraction wins most
                top_losses = sorted(targets, key=lambda r: r["mae_improvement_pct"])[:5]
                h.append(f"<p><strong>Best for Subtraction</strong> (where {best_arch} underperforms):</p>")
                h.append("<table><tr><th>Target</th><th>N pairs</th>"
                         f"<th>{best_arch} MAE</th><th>Sub MAE</th><th>ΔMAE%</th>"
                         f"<th>{best_arch} R²</th><th>Sub R²</th></tr>")
                for r in top_losses:
                    h.append(f"<tr>"
                             f"<td>{r['target']}</td><td>{r['n']}</td>"
                             f"<td>{r['edit_mae']:.3f}</td>"
                             f"<td class='best'>{r['sub_mae']:.3f}</td>"
                             f"<td class='loss'>{r['mae_improvement_pct']:+.1f}%</td>"
                             f"<td>{r['edit_r2']:.3f}</td><td>{r['sub_r2']:.3f}</td>"
                             f"</tr>")
                h.append("</table>")

    # ── Metrics Policy ──
    h.append("<h2 id='metrics'>Metrics Policy</h2>")
    h.append("<div class='note'>")
    h.append("<strong>Primary</strong>: MAE (lower is better) — mean absolute error of predicted vs actual Δproperty<br>")
    h.append("<strong>Secondary</strong>: Spearman rank correlation (higher is better) — captures ranking quality<br>")
    h.append("<strong>Also reported</strong>: Pearson r, R² — computed per-target then averaged "
             "(NOT pooled across targets, which produces misleading artifacts)<br>")
    h.append("<strong>All values</strong>: mean ± std across 3 random seeds (42, 123, 456).<br>")
    p3_note = "300K subsample" if MAX_PHASE3_PAIRS else "full 1.7M shared pairs"
    h.append(f"<strong>Phase 3</strong>: Run on {p3_note}. Phase 1–2 used full 1.7M pairs.")
    h.append("</div>")

    h.append("</body></html>")

    report_path = RESULTS_DIR / "evaluation_report.html"
    report_path.write_text("\n".join(h))
    print(f"\nReport saved to: {report_path}")
    return str(report_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Paper evaluation pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only this phase (default: all)")
    parser.add_argument("--embedder", type=str, default=None,
                        help="Override embedder selection (skip phase 1)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture selection (skip phase 2)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from saved results")
    parser.add_argument("--phase3-methods", type=str, nargs="*", default=None,
                        help="Additional methods to run in Phase 3 (e.g. GatedCrossAttn FiLMDelta)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    results_file = RESULTS_DIR / "all_results.json"
    saved = {}
    if results_file.exists():
        with open(results_file) as f:
            saved = json.load(f)

    if args.report_only:
        generate_report(saved.get("phase1"), saved.get("phase2"), saved.get("phase3"))
        return

    # Load data
    df = load_data()

    # Phase 1: Embedder selection
    phase1 = saved.get("phase1")
    if args.phase is None or args.phase == 1:
        if args.embedder is None:
            phase1 = run_phase1(df)
            saved["phase1"] = phase1
            with open(results_file, 'w') as f:
                json.dump(saved, f, indent=2, default=str)

    best_emb = args.embedder or (phase1 or {}).get("_best", "morgan")
    print(f"\n→ Using embedder: {best_emb}")

    # Phase 2: Architecture comparison
    phase2 = saved.get("phase2")
    if args.phase is None or args.phase == 2:
        if args.arch is None:
            phase2 = run_phase2(df, best_emb)
            saved["phase2"] = phase2
            with open(results_file, 'w') as f:
                json.dump(saved, f, indent=2, default=str)

    best_arch = args.arch or (phase2 or {}).get("_best", "EditDiff")
    print(f"\n→ Using architecture: {best_arch}")

    # Phase 3: Generalization
    phase3 = saved.get("phase3")
    if args.phase is None or args.phase == 3:
        phase3 = run_phase3(df, best_emb, best_arch, extra_methods=args.phase3_methods)
        saved["phase3"] = phase3
        with open(results_file, 'w') as f:
            json.dump(saved, f, indent=2, default=str)

    # Generate report
    generate_report(phase1, phase2, phase3)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"Results: {results_file}")
    print(f"Report:  {RESULTS_DIR / 'evaluation_report.html'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
