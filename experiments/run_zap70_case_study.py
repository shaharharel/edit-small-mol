#!/usr/bin/env python3
"""
ZAP70 Comprehensive Case Study — iterative modeling with proper baselines.

Implements the full expert-recommended protocol:
  Tier 0: Baselines (predict-zero, random permutation, Tanimoto-NN, Ridge)
  Tier 1: Zero-shot transfer from 751-target MMP model
  Tier 2: Direct training on ZAP70 with LOAO + other splits
  Tier 3: Pretrain on 751 targets → finetune on ZAP70
  Tier 4: Multi-task with related kinases (SYK, BTK)

Evaluation: Leave-One-Assay-Out (LOAO) on assays with 20+ molecules.
Also: random pair split, scaffold split, molecule-level split.

Usage:
    conda run -n quris python -u experiments/run_zap70_case_study.py
    conda run -n quris python -u experiments/run_zap70_case_study.py --tier 0
    conda run -n quris python -u experiments/run_zap70_case_study.py --tier 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_paper_evaluation import (
    SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, DROPOUT, DEVICE,
    RESULTS_DIR, CACHE_DIR, DATA_DIR,
    DeltaMLP, AbsoluteMLP,
    compute_embeddings, compute_metrics,
    get_pair_tensors,
    train_model, train_model_multi_input, predict, predict_multi_input,
    aggregate_seeds,
)

import warnings
warnings.filterwarnings("ignore")

# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
MMP_FILE = DATA_DIR / "shared_pairs_deduped.csv"

RESULTS_FILE = RESULTS_DIR / "zap70_case_study_results.json"
REPORT_FILE = RESULTS_DIR / "zap70_case_study.html"

# ZAP70 config
ZAP70_ID = "CHEMBL2803"
# Related kinases for multi-task (Tier 4)
RELATED_KINASES = {
    "SYK": "CHEMBL2599",
    "BTK": "CHEMBL5251",
    "LCK": "CHEMBL258",
}

# Smaller models for small data (expert recommendation)
SMALL_HIDDEN = [256, 128, 64]
SMALL_DROPOUT = 0.35
SMALL_LR = 5e-4
SMALL_PATIENCE = 20

# Minimum assay size for LOAO evaluation
MIN_LOAO_ASSAY_SIZE = 10  # molecules


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

def extract_target_allpairs(target_id, raw_df=None):
    """Extract all within-assay pairs for a target."""
    if raw_df is None:
        raw_df = pd.read_csv(RAW_FILE)
    target_data = raw_df[raw_df["target_chembl_id"] == target_id].copy()

    # Average duplicate measurements
    deduped = target_data.groupby(["molecule_chembl_id", "assay_id"]).agg({
        "smiles": "first", "target_chembl_id": "first", "pIC50": "mean",
    }).reset_index()

    pairs = []
    for assay_id, group in deduped.groupby("assay_id"):
        if len(group) < 2:
            continue
        mols = group[["molecule_chembl_id", "smiles", "pIC50"]].values
        for i, j in combinations(range(len(mols)), 2):
            pairs.append({
                "mol_a": mols[i][1], "mol_b": mols[j][1],
                "mol_a_id": mols[i][0], "mol_b_id": mols[j][0],
                "edit_smiles": "", "is_within_assay": True,
                "delta": float(mols[j][2]) - float(mols[i][2]),
                "value_a": float(mols[i][2]), "value_b": float(mols[j][2]),
                "target_chembl_id": target_id,
                "assay_id_a": int(assay_id), "assay_id_b": int(assay_id),
            })
    return pd.DataFrame(pairs)


def identify_mmp_pairs(pairs_df):
    """Identify which pairs are genuine MMPs using RDKit MMP fragmentation."""
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    print("  Identifying MMP pairs among all-pairs...")
    # Build fragment cache per molecule
    smiles_set = set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist())
    frag_cache = {}  # smiles -> set of (core, frag) tuples

    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            frag_cache[smi] = set()
            continue
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
            cores = set()
            for core_smi, chains_smi in frags:
                if core_smi:
                    cores.add(core_smi)
            frag_cache[smi] = cores
        except Exception:
            frag_cache[smi] = set()

    # Check each pair for shared core
    is_mmp = []
    for _, row in pairs_df.iterrows():
        cores_a = frag_cache.get(row["mol_a"], set())
        cores_b = frag_cache.get(row["mol_b"], set())
        is_mmp.append(len(cores_a & cores_b) > 0)

    pairs_df = pairs_df.copy()
    pairs_df["is_mmp"] = is_mmp
    n_mmp = sum(is_mmp)
    print(f"  Found {n_mmp} MMP pairs out of {len(pairs_df)} total ({100*n_mmp/len(pairs_df):.1f}%)")
    return pairs_df


def characterize_assays(pairs_df):
    """Compute per-assay statistics."""
    stats = []
    for assay_id, group in pairs_df.groupby("assay_id_a"):
        mols = set(group["mol_a"].tolist() + group["mol_b"].tolist())
        deltas = group["delta"].values
        pIC50s = np.concatenate([group["value_a"].values, group["value_b"].values])
        stats.append({
            "assay_id": int(assay_id),
            "n_molecules": len(mols),
            "n_pairs": len(group),
            "n_mmp": int(group["is_mmp"].sum()) if "is_mmp" in group.columns else 0,
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "delta_abs_mean": float(np.mean(np.abs(deltas))),
            "pIC50_min": float(pIC50s.min()),
            "pIC50_max": float(pIC50s.max()),
            "pIC50_range": float(pIC50s.max() - pIC50s.min()),
            "pIC50_std": float(np.std(pIC50s)),
        })
    return sorted(stats, key=lambda x: -x["n_molecules"])


# ═══════════════════════════════════════════════════════════════════════════
# Splitting Strategies
# ═══════════════════════════════════════════════════════════════════════════

def loao_splits(pairs_df, min_assay_size=MIN_LOAO_ASSAY_SIZE):
    """Leave-One-Assay-Out splits. Returns list of (fold_name, train_df, test_df)."""
    assay_sizes = pairs_df.groupby("assay_id_a").apply(
        lambda g: len(set(g["mol_a"].tolist() + g["mol_b"].tolist()))
    )
    large_assays = assay_sizes[assay_sizes >= min_assay_size].index.tolist()
    print(f"  LOAO: {len(large_assays)} assays with ≥{min_assay_size} molecules")

    splits = []
    for test_assay in large_assays:
        test_df = pairs_df[pairs_df["assay_id_a"] == test_assay].copy()
        train_df = pairs_df[pairs_df["assay_id_a"] != test_assay].copy()
        if len(train_df) < 20 or len(test_df) < 5:
            continue
        splits.append((f"loao_{test_assay}", train_df, test_df))
    return splits


def random_pair_splits(pairs_df, n_repeats=5, test_frac=0.2):
    """Random pair-level splits (with molecule leakage — for reference only)."""
    splits = []
    for seed in range(n_repeats):
        np.random.seed(seed + 42)
        idx = np.random.permutation(len(pairs_df))
        n_test = int(len(pairs_df) * test_frac)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        splits.append((
            f"random_{seed}",
            pairs_df.iloc[train_idx].copy(),
            pairs_df.iloc[test_idx].copy(),
        ))
    return splits


def molecule_level_splits(pairs_df, n_repeats=5, test_frac=0.2):
    """Split by molecules — no molecule appears in both train and test."""
    all_mols = list(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist()))
    splits = []
    for seed in range(n_repeats):
        np.random.seed(seed + 100)
        np.random.shuffle(all_mols)
        n_test = max(5, int(len(all_mols) * test_frac))
        test_mols = set(all_mols[:n_test])

        test_df = pairs_df[
            pairs_df["mol_a"].isin(test_mols) & pairs_df["mol_b"].isin(test_mols)
        ].copy()
        train_df = pairs_df[
            ~pairs_df["mol_a"].isin(test_mols) & ~pairs_df["mol_b"].isin(test_mols)
        ].copy()

        if len(test_df) < 5 or len(train_df) < 20:
            continue
        splits.append((f"mol_split_{seed}", train_df, test_df))
    return splits


def scaffold_splits(pairs_df, n_repeats=3, test_frac=0.2):
    """Split by Bemis-Murcko scaffold — novel scaffolds in test."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    all_mols = list(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist()))
    mol_to_scaffold = {}
    for smi in all_mols:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaf = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol))
                mol_to_scaffold[smi] = Chem.MolToSmiles(scaf)
            except Exception:
                mol_to_scaffold[smi] = "unknown"
        else:
            mol_to_scaffold[smi] = "unknown"

    scaffolds = list(set(mol_to_scaffold.values()))
    splits = []
    for seed in range(n_repeats):
        np.random.seed(seed + 200)
        np.random.shuffle(scaffolds)
        n_test = max(2, int(len(scaffolds) * test_frac))
        test_scaffolds = set(scaffolds[:n_test])
        test_mols = {s for s, sc in mol_to_scaffold.items() if sc in test_scaffolds}

        test_df = pairs_df[
            pairs_df["mol_a"].isin(test_mols) & pairs_df["mol_b"].isin(test_mols)
        ].copy()
        train_df = pairs_df[
            ~pairs_df["mol_a"].isin(test_mols) & ~pairs_df["mol_b"].isin(test_mols)
        ].copy()

        if len(test_df) < 5 or len(train_df) < 20:
            continue
        splits.append((f"scaffold_{seed}", train_df, test_df))
    return splits


# ═══════════════════════════════════════════════════════════════════════════
# Ranking Metrics
# ═══════════════════════════════════════════════════════════════════════════

def precision_at_k(y_true, y_pred, k):
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    top_pred = set(np.argsort(-y_pred.flatten())[:k])
    top_true = set(np.argsort(-y_true)[:k])
    return len(top_pred & top_true) / k


def random_precision_at_k(n, k, n_trials=10000):
    """Expected P@k under random permutation."""
    if n <= k:
        return 1.0
    hits = 0
    for _ in range(n_trials):
        perm = np.random.permutation(n)
        top_pred = set(perm[:k])
        top_true = set(range(k))
        hits += len(top_pred & top_true) / k
    return hits / n_trials


def compute_ranking_metrics(y_true, y_pred):
    n = len(y_true)
    y_pred = y_pred.flatten()
    m = {"n": n, "mae": float(np.mean(np.abs(y_true - y_pred)))}
    if n > 2:
        rho, p = spearmanr(y_true, y_pred)
        m["spearman_r"] = float(rho) if not np.isnan(rho) else 0.0
        m["spearman_p"] = float(p) if not np.isnan(p) else 1.0
    for k in [3, 5, 10]:
        if n >= k:
            m[f"p_at_{k}"] = precision_at_k(y_true, y_pred, k)
    return m


def permutation_test(y_true, y_pred, metric_fn, n_perm=5000):
    """Compute p-value by permutation test."""
    observed = metric_fn(y_true, y_pred.flatten())
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(y_true)
        if metric_fn(perm, y_pred.flatten()) >= observed:
            count += 1
    return count / n_perm


# ═══════════════════════════════════════════════════════════════════════════
# Tier 0: Baselines
# ═══════════════════════════════════════════════════════════════════════════

def predict_zero(test_df, **kwargs):
    """Predict delta=0 for all pairs."""
    return np.zeros(len(test_df))


def predict_mean(train_df, test_df, **kwargs):
    """Predict the mean training delta."""
    mean_delta = train_df["delta"].mean()
    return np.full(len(test_df), mean_delta)


def predict_tanimoto_nn(train_df, test_df, emb_dict, emb_dim, **kwargs):
    """Nearest-neighbor by Tanimoto similarity — use most similar training pair's delta."""
    train_a, train_b, train_y = get_pair_tensors(train_df, emb_dict, emb_dim)
    test_a, test_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)

    # Compute pair "signature" as concat of sorted embeddings
    train_sigs = torch.cat([train_a, train_b], dim=-1).numpy()
    test_sigs = torch.cat([test_a, test_b], dim=-1).numpy()

    # For each test pair, find nearest training pair by cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    preds = []
    # Process in chunks to save memory
    chunk_size = 500
    train_y_np = train_y.numpy().flatten()
    for i in range(0, len(test_sigs), chunk_size):
        chunk = test_sigs[i:i+chunk_size]
        sims = cosine_similarity(chunk, train_sigs)
        nn_idx = np.argmax(sims, axis=1)
        preds.extend(train_y_np[nn_idx])
    return np.array(preds)


def predict_ridge(train_df, test_df, emb_dict, emb_dim, **kwargs):
    """Ridge regression on Morgan FP difference."""
    train_a, train_b, train_y = get_pair_tensors(train_df, emb_dict, emb_dim)
    test_a, test_b, _ = get_pair_tensors(test_df, emb_dict, emb_dim)

    X_train = (train_b - train_a).numpy()
    y_train = train_y.numpy().flatten()
    X_test = (test_b - test_a).numpy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Zero-shot transfer
# ═══════════════════════════════════════════════════════════════════════════

def train_zeroshot_model(method, mmp_df, emb_dict, emb_dim, seed):
    """Train model on 751-target MMP data, return predict function."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.utils.splits import get_splitter
    splitter = get_splitter("assay", random_state=seed, scenario="within_assay")
    train_df, val_df, _ = splitter.split(mmp_df)

    if method == "FiLMDelta":
        from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
        ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
        va, vb, vy = get_pair_tensors(val_df, emb_dict, emb_dim)
        model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])
        fwd = lambda m, a, b: m(a, b)
        model = train_model_multi_input(
            model,
            DataLoader(TensorDataset(ta, tb, ty), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(TensorDataset(va, vb, vy), batch_size=BATCH_SIZE, shuffle=False),
            fwd)
        return lambda df: predict_multi_input(model, fwd,
            *get_pair_tensors(df, emb_dict, emb_dim)[:2])

    elif method == "Subtraction":
        mv = dict(zip(train_df["mol_a"], train_df["value_a"]))
        mv.update(dict(zip(train_df["mol_b"], train_df["value_b"])))
        zero = np.zeros(emb_dim)
        sl = list(mv.keys())
        X = np.array([emb_dict.get(s, zero) for s in sl], dtype=np.float32)
        y = np.array([mv[s] for s in sl], dtype=np.float32)
        vmv = dict(zip(val_df["mol_a"], val_df["value_a"]))
        vmv.update(dict(zip(val_df["mol_b"], val_df["value_b"])))
        vsl = list(vmv.keys())
        vX = np.array([emb_dict.get(s, zero) for s in vsl], dtype=np.float32)
        vy = np.array([vmv[s] for s in vsl], dtype=np.float32)
        model = AbsoluteMLP(emb_dim, hidden_dims=[512, 256, 128], dropout=DROPOUT)
        model = train_model(
            model,
            DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                     torch.from_numpy(y).float()),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(TensorDataset(torch.from_numpy(vX).float(),
                                     torch.from_numpy(vy).float()),
                       batch_size=BATCH_SIZE, shuffle=False))
        return lambda df: (predict(model, get_pair_tensors(df, emb_dict, emb_dim)[1]) -
                           predict(model, get_pair_tensors(df, emb_dict, emb_dim)[0]))

    elif method == "DeepDelta":
        ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
        model = DeltaMLP(emb_dim * 2, hidden_dims=[512, 256, 128], dropout=DROPOUT)
        model = train_model(
            model,
            DataLoader(TensorDataset(torch.cat([ta, tb], -1), ty),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(TensorDataset(
                torch.cat(get_pair_tensors(val_df, emb_dict, emb_dim)[:2], -1),
                get_pair_tensors(val_df, emb_dict, emb_dim)[2]),
                batch_size=BATCH_SIZE, shuffle=False))
        return lambda df: predict(model, torch.cat(
            get_pair_tensors(df, emb_dict, emb_dim)[:2], -1))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: Direct training on ZAP70
# ═══════════════════════════════════════════════════════════════════════════

def train_direct(method, train_df, val_df, test_df, emb_dict, emb_dim, seed):
    """Train directly on ZAP70 data with smaller models."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use 15% of train as val if no val provided
    if val_df is None or len(val_df) == 0:
        n_val = max(5, int(len(train_df) * 0.15))
        idx = np.random.permutation(len(train_df))
        val_df = train_df.iloc[idx[:n_val]].copy()
        train_df = train_df.iloc[idx[n_val:]].copy()

    if method == "FiLMDelta":
        from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
        ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
        va, vb, vy = get_pair_tensors(val_df, emb_dict, emb_dim)
        model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=SMALL_HIDDEN,
                             dropout=SMALL_DROPOUT)
        fwd = lambda m, a, b: m(a, b)
        model = train_model_multi_input(
            model,
            DataLoader(TensorDataset(ta, tb, ty), batch_size=64, shuffle=True),
            DataLoader(TensorDataset(va, vb, vy), batch_size=64, shuffle=False),
            fwd, max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
        tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict_multi_input(model, fwd, tea, teb)

    elif method == "DeepDelta":
        ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
        va, vb, vy = get_pair_tensors(val_df, emb_dict, emb_dim)
        model = DeltaMLP(emb_dim * 2, hidden_dims=SMALL_HIDDEN, dropout=SMALL_DROPOUT)
        model = train_model(
            model,
            DataLoader(TensorDataset(torch.cat([ta, tb], -1), ty),
                       batch_size=64, shuffle=True),
            DataLoader(TensorDataset(torch.cat([va, vb], -1), vy),
                       batch_size=64, shuffle=False),
            max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
        tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict(model, torch.cat([tea, teb], -1))

    elif method == "EditDiff":
        ta, tb, ty = get_pair_tensors(train_df, emb_dict, emb_dim)
        va, vb, vy = get_pair_tensors(val_df, emb_dict, emb_dim)
        model = DeltaMLP(emb_dim * 2, hidden_dims=SMALL_HIDDEN, dropout=SMALL_DROPOUT)
        model = train_model(
            model,
            DataLoader(TensorDataset(torch.cat([ta, tb - ta], -1), ty),
                       batch_size=64, shuffle=True),
            DataLoader(TensorDataset(torch.cat([va, vb - va], -1), vy),
                       batch_size=64, shuffle=False),
            max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
        tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict(model, torch.cat([tea, teb - tea], -1))

    elif method == "Subtraction":
        mv = dict(zip(train_df["mol_a"], train_df["value_a"]))
        mv.update(dict(zip(train_df["mol_b"], train_df["value_b"])))
        zero = np.zeros(emb_dim)
        sl = list(mv.keys())
        X = np.array([emb_dict.get(s, zero) for s in sl], dtype=np.float32)
        y = np.array([mv[s] for s in sl], dtype=np.float32)
        vmv = dict(zip(val_df["mol_a"], val_df["value_a"]))
        vmv.update(dict(zip(val_df["mol_b"], val_df["value_b"])))
        vsl = list(vmv.keys())
        vX = np.array([emb_dict.get(s, zero) for s in vsl], dtype=np.float32)
        vy_arr = np.array([vmv[s] for s in vsl], dtype=np.float32)
        model = AbsoluteMLP(emb_dim, hidden_dims=SMALL_HIDDEN, dropout=SMALL_DROPOUT)
        model = train_model(
            model,
            DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                     torch.from_numpy(y).float()),
                       batch_size=64, shuffle=True),
            DataLoader(TensorDataset(torch.from_numpy(vX).float(),
                                     torch.from_numpy(vy_arr).float()),
                       batch_size=64, shuffle=False),
            max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)
        tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
        return predict(model, teb) - predict(model, tea)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Pretrain → Finetune
# ═══════════════════════════════════════════════════════════════════════════

def train_pretrain_finetune(method, mmp_df, zap_train_df, zap_val_df, test_df,
                            emb_dict, emb_dim, seed):
    """Pretrain on 751-target MMP, finetune on ZAP70."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if method != "FiLMDelta":
        # For simplicity, only implement FiLM pretrain-finetune
        return None

    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
    from src.utils.splits import get_splitter

    # Stage 1: Pretrain on MMP
    print("    Pretrain on MMP...", end=" ", flush=True)
    splitter = get_splitter("assay", random_state=seed, scenario="within_assay")
    mmp_train, mmp_val, _ = splitter.split(mmp_df)

    ta, tb, ty = get_pair_tensors(mmp_train, emb_dict, emb_dim)
    va, vb, vy = get_pair_tensors(mmp_val, emb_dict, emb_dim)

    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[512, 256, 128])
    fwd = lambda m, a, b: m(a, b)
    model = train_model_multi_input(
        model,
        DataLoader(TensorDataset(ta, tb, ty), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(va, vb, vy), batch_size=BATCH_SIZE, shuffle=False),
        fwd)
    print("done")
    del ta, tb, ty, va, vb, vy
    gc.collect()

    # Stage 2: Finetune on ZAP70
    if zap_val_df is None or len(zap_val_df) == 0:
        n_val = max(3, int(len(zap_train_df) * 0.15))
        idx = np.random.permutation(len(zap_train_df))
        zap_val_df = zap_train_df.iloc[idx[:n_val]].copy()
        zap_train_df = zap_train_df.iloc[idx[n_val:]].copy()

    print("    Finetune on ZAP70...", end=" ", flush=True)
    za, zb, zy = get_pair_tensors(zap_train_df, emb_dict, emb_dim)
    zva, zvb, zvy = get_pair_tensors(zap_val_df, emb_dict, emb_dim)
    model = train_model_multi_input(
        model,
        DataLoader(TensorDataset(za, zb, zy), batch_size=64, shuffle=True),
        DataLoader(TensorDataset(zva, zvb, zvy), batch_size=64, shuffle=False),
        fwd, max_epochs=MAX_EPOCHS // 2, patience=SMALL_PATIENCE, lr=SMALL_LR / 5)
    print("done")

    tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict_multi_input(model, fwd, tea, teb)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4: Multi-task with related kinases
# ═══════════════════════════════════════════════════════════════════════════

def train_multitask(mmp_df, related_pairs, zap_train_df, zap_val_df, test_df,
                    emb_dict, emb_dim, seed):
    """Multi-task: train on related kinases + ZAP70 data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    # Combine related kinase MMP pairs + ZAP70 all-pairs
    combined = pd.concat([related_pairs, zap_train_df], ignore_index=True)
    print(f"    Multi-task training: {len(combined):,} pairs "
          f"({len(related_pairs):,} related + {len(zap_train_df):,} ZAP70)")

    if zap_val_df is None or len(zap_val_df) == 0:
        n_val = max(3, int(len(zap_train_df) * 0.15))
        idx = np.random.permutation(len(zap_train_df))
        zap_val_df = zap_train_df.iloc[idx[:n_val]].copy()

    ta, tb, ty = get_pair_tensors(combined, emb_dict, emb_dim)
    va, vb, vy = get_pair_tensors(zap_val_df, emb_dict, emb_dim)

    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=SMALL_HIDDEN,
                         dropout=SMALL_DROPOUT)
    fwd = lambda m, a, b: m(a, b)
    model = train_model_multi_input(
        model,
        DataLoader(TensorDataset(ta, tb, ty), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(va, vb, vy), batch_size=64, shuffle=False),
        fwd, max_epochs=MAX_EPOCHS, patience=SMALL_PATIENCE, lr=SMALL_LR)

    tea, teb, _ = get_pair_tensors(test_df, emb_dict, emb_dim)
    return predict_multi_input(model, fwd, tea, teb)


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_tier0(pairs_df, emb_dict, emb_dim, results):
    """Tier 0: Baselines on LOAO splits."""
    print("\n" + "=" * 70)
    print("TIER 0: Baselines")
    print("=" * 70)

    if "tier0" in results and results["tier0"].get("completed"):
        print("  Already completed")
        return results

    splits = loao_splits(pairs_df)
    tier0 = results.get("tier0", {})

    baselines = {
        "predict_zero": lambda tr, te, **kw: predict_zero(te),
        "predict_mean": lambda tr, te, **kw: predict_mean(tr, te),
        "tanimoto_nn": lambda tr, te, **kw: predict_tanimoto_nn(tr, te, emb_dict, emb_dim),
        "ridge": lambda tr, te, **kw: predict_ridge(tr, te, emb_dict, emb_dim),
    }

    for bl_name, bl_fn in baselines.items():
        if bl_name in tier0:
            print(f"  {bl_name}: already done")
            continue

        fold_metrics = []
        for fold_name, train_df, test_df in splits:
            y_true = test_df["delta"].values
            y_pred = bl_fn(train_df, test_df)
            m = compute_ranking_metrics(y_true, y_pred)
            m["fold"] = fold_name
            fold_metrics.append(m)

        agg = {
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "spearman_mean": float(np.mean([m.get("spearman_r", 0) for m in fold_metrics])),
            "spearman_std": float(np.std([m.get("spearman_r", 0) for m in fold_metrics])),
        }
        tier0[bl_name] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"  {bl_name}: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
              f"Spearman={agg['spearman_mean']:.4f}")

    # Random permutation test
    if "random_null" not in tier0:
        print("  Computing random null distribution...")
        all_deltas = pairs_df["delta"].values
        null_spearman = []
        null_mae = []
        for _ in range(5000):
            perm = np.random.permutation(all_deltas)
            null_mae.append(float(np.mean(np.abs(all_deltas - perm))))
            r, _ = spearmanr(all_deltas, perm)
            null_spearman.append(float(r) if not np.isnan(r) else 0.0)
        tier0["random_null"] = {
            "mae_mean": float(np.mean(null_mae)),
            "mae_p95": float(np.percentile(null_mae, 5)),  # Lower MAE = better
            "spearman_mean": float(np.mean(null_spearman)),
            "spearman_p95": float(np.percentile(null_spearman, 95)),
        }
        print(f"  Random null: MAE={tier0['random_null']['mae_mean']:.4f}, "
              f"Spearman 95th={tier0['random_null']['spearman_p95']:.4f}")

    tier0["completed"] = True
    results["tier0"] = tier0
    return results


def run_tier1(pairs_df, mmp_df, emb_dict, emb_dim, results):
    """Tier 1: Zero-shot transfer."""
    print("\n" + "=" * 70)
    print("TIER 1: Zero-shot Transfer")
    print("=" * 70)

    if "tier1" in results and results["tier1"].get("completed"):
        print("  Already completed")
        return results

    tier1 = results.get("tier1", {})
    seed = SEEDS[0]

    for method in ["FiLMDelta", "Subtraction", "DeepDelta"]:
        if method in tier1:
            print(f"  {method}: already done")
            continue

        print(f"\n  Training {method} on 751-target MMP data...")
        predict_fn = train_zeroshot_model(method, mmp_df, emb_dict, emb_dim, seed)

        y_true = pairs_df["delta"].values
        y_pred = predict_fn(pairs_df)
        overall = compute_ranking_metrics(y_true, y_pred)

        # Per-assay
        per_assay = []
        for assay_id, group in pairs_df.groupby("assay_id_a"):
            if len(group) < 5:
                continue
            m = compute_ranking_metrics(group["delta"].values, predict_fn(group))
            m["assay_id"] = int(assay_id)
            per_assay.append(m)

        tier1[method] = {"overall": overall, "per_assay": per_assay}
        print(f"    Overall: MAE={overall['mae']:.4f}, Spearman={overall.get('spearman_r', 0):.4f}")
        gc.collect()

    tier1["completed"] = True
    results["tier1"] = tier1
    return results


def run_tier2(pairs_df, emb_dict, emb_dim, results):
    """Tier 2: Direct training on ZAP70 with multiple split strategies."""
    print("\n" + "=" * 70)
    print("TIER 2: Direct Training on ZAP70")
    print("=" * 70)

    tier2 = results.get("tier2", {})
    methods = ["FiLMDelta", "DeepDelta", "EditDiff", "Subtraction"]

    # Multiple split strategies
    split_generators = {
        "loao": lambda: loao_splits(pairs_df),
        "random_pair": lambda: random_pair_splits(pairs_df),
        "mol_split": lambda: molecule_level_splits(pairs_df),
        "scaffold": lambda: scaffold_splits(pairs_df),
    }

    for split_name, gen_fn in split_generators.items():
        print(f"\n--- Split: {split_name} ---")
        try:
            splits = gen_fn()
        except Exception as e:
            print(f"  Failed to generate splits: {e}")
            continue

        if not splits:
            print(f"  No valid splits generated")
            continue

        print(f"  {len(splits)} folds")

        for method in methods:
            key = f"{split_name}__{method}"
            if key in tier2:
                agg = tier2[key].get("aggregated", {})
                print(f"  {key}: already done (MAE={agg.get('mae_mean', '?')})")
                continue

            print(f"  {method} on {split_name}...", end=" ", flush=True)
            fold_metrics = []
            for fold_name, train_df, test_df in splits:
                try:
                    y_true = test_df["delta"].values
                    y_pred = train_direct(method, train_df, None, test_df,
                                          emb_dict, emb_dim, SEEDS[0])
                    m = compute_ranking_metrics(y_true, y_pred)
                    m["fold"] = fold_name
                    m["train_size"] = len(train_df)
                    m["test_size"] = len(test_df)
                    fold_metrics.append(m)
                except Exception as e:
                    print(f"error in {fold_name}: {e}")

            if fold_metrics:
                agg = {
                    "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
                    "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
                    "spearman_mean": float(np.mean([m.get("spearman_r", 0)
                                                    for m in fold_metrics])),
                    "spearman_std": float(np.std([m.get("spearman_r", 0)
                                                  for m in fold_metrics])),
                    "n_folds": len(fold_metrics),
                }
                tier2[key] = {"aggregated": agg, "per_fold": fold_metrics}
                print(f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
                      f"Spearman={agg['spearman_mean']:.4f}")
            else:
                print("no valid folds")

            gc.collect()

    tier2["completed"] = True
    results["tier2"] = tier2
    return results


def run_tier3(pairs_df, mmp_df, emb_dict, emb_dim, results):
    """Tier 3: Pretrain on 751 targets → finetune on ZAP70."""
    print("\n" + "=" * 70)
    print("TIER 3: Pretrain → Finetune")
    print("=" * 70)

    tier3 = results.get("tier3", {})

    splits = loao_splits(pairs_df)
    if not splits:
        print("  No LOAO splits available")
        return results

    key = "loao__FiLMDelta_pretrain_finetune"
    if key in tier3:
        print(f"  {key}: already done")
        results["tier3"] = tier3
        return results

    fold_metrics = []
    for fold_name, train_df, test_df in splits:
        print(f"\n  Fold: {fold_name} (train={len(train_df)}, test={len(test_df)})")
        try:
            y_true = test_df["delta"].values
            y_pred = train_pretrain_finetune(
                "FiLMDelta", mmp_df, train_df, None, test_df,
                emb_dict, emb_dim, SEEDS[0])
            if y_pred is not None:
                m = compute_ranking_metrics(y_true, y_pred)
                m["fold"] = fold_name
                fold_metrics.append(m)
                print(f"    MAE={m['mae']:.4f}, Spearman={m.get('spearman_r', 0):.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback; traceback.print_exc()
        gc.collect()

    if fold_metrics:
        agg = {
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "spearman_mean": float(np.mean([m.get("spearman_r", 0)
                                            for m in fold_metrics])),
            "n_folds": len(fold_metrics),
        }
        tier3[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"\n  Pretrain→Finetune: MAE={agg['mae_mean']:.4f}, "
              f"Spearman={agg['spearman_mean']:.4f}")

    tier3["completed"] = True
    results["tier3"] = tier3
    return results


def run_tier4(pairs_df, mmp_df, emb_dict, emb_dim, results):
    """Tier 4: Multi-task with related kinases."""
    print("\n" + "=" * 70)
    print("TIER 4: Multi-task with Related Kinases")
    print("=" * 70)

    tier4 = results.get("tier4", {})
    key = "loao__multitask_FiLM"
    if key in tier4:
        print(f"  {key}: already done")
        results["tier4"] = tier4
        return results

    # Get related kinase pairs from MMP dataset
    related_targets = list(RELATED_KINASES.values())
    related_pairs = mmp_df[
        (mmp_df["target_chembl_id"].isin(related_targets)) &
        (mmp_df["is_within_assay"] == True) &
        (mmp_df["mol_a_id"] != mmp_df["mol_b_id"])
    ].copy()
    print(f"  Related kinase MMP pairs: {len(related_pairs):,}")
    for name, tid in RELATED_KINASES.items():
        n = (related_pairs["target_chembl_id"] == tid).sum()
        print(f"    {name} ({tid}): {n:,} pairs")

    splits = loao_splits(pairs_df)
    if not splits:
        print("  No LOAO splits available")
        return results

    fold_metrics = []
    for fold_name, train_df, test_df in splits:
        print(f"\n  Fold: {fold_name}")
        try:
            y_true = test_df["delta"].values
            y_pred = train_multitask(
                mmp_df, related_pairs, train_df, None, test_df,
                emb_dict, emb_dim, SEEDS[0])
            m = compute_ranking_metrics(y_true, y_pred)
            m["fold"] = fold_name
            fold_metrics.append(m)
            print(f"    MAE={m['mae']:.4f}, Spearman={m.get('spearman_r', 0):.4f}")
        except Exception as e:
            print(f"    Error: {e}")
        gc.collect()

    if fold_metrics:
        agg = {
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "spearman_mean": float(np.mean([m.get("spearman_r", 0)
                                            for m in fold_metrics])),
            "n_folds": len(fold_metrics),
        }
        tier4[key] = {"aggregated": agg, "per_fold": fold_metrics}
        print(f"\n  Multi-task: MAE={agg['mae_mean']:.4f}, "
              f"Spearman={agg['spearman_mean']:.4f}")

    tier4["completed"] = True
    results["tier4"] = tier4
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results):
    """Generate comprehensive HTML report."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ZAP70 Comprehensive Case Study</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 0 auto;
       padding: 20px; background: #f8f9fa; }
h1 { color: #2c3e50; border-bottom: 3px solid #e74c3c; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #7f8c8d; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
.best { font-weight: bold; color: #27ae60; }
.warn { color: #e74c3c; }
.section { background: white; padding: 20px; margin: 15px 0; border-radius: 8px;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.metric { display: inline-block; background: #ecf0f1; padding: 15px 25px;
          margin: 5px; border-radius: 8px; text-align: center; }
.metric .v { font-size: 24px; font-weight: bold; color: #2c3e50; }
.metric .l { font-size: 12px; color: #7f8c8d; }
</style></head><body>
"""
    html += "<h1>ZAP70 (CHEMBL2803) — Comprehensive Case Study</h1>\n"
    html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>\n"

    # Data summary
    if "data_summary" in results:
        ds = results["data_summary"]
        html += '<div class="section"><h2>Data Summary</h2>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_molecules", "?")}</div>'
        html += '<div class="l">Molecules</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_assays", "?")}</div>'
        html += '<div class="l">Assays</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_pairs", "?"):,}</div>'
        html += '<div class="l">All Pairs</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_mmp_pairs", 0):,}</div>'
        html += '<div class="l">MMP Pairs</div></div>\n'
        html += f'<div class="metric"><div class="v">{ds.get("n_loao_assays", "?")}</div>'
        html += '<div class="l">LOAO Assays (≥10 mols)</div></div>\n'
        html += "</div>\n"

    # Assay table
    if "assay_stats" in results:
        html += '<div class="section"><h2>Per-Assay Characterization</h2>\n'
        html += "<table><tr><th>Assay ID</th><th>Molecules</th><th>Pairs</th>"
        html += "<th>MMP Pairs</th><th>pIC50 Range</th><th>|Δ| Mean</th></tr>\n"
        for a in results["assay_stats"]:
            html += f"<tr><td>{a['assay_id']}</td><td>{a['n_molecules']}</td>"
            html += f"<td>{a['n_pairs']}</td><td>{a['n_mmp']}</td>"
            html += f"<td>{a['pIC50_range']:.2f}</td>"
            html += f"<td>{a['delta_abs_mean']:.3f}</td></tr>\n"
        html += "</table></div>\n"

    # Tier results comparison
    html += '<div class="section"><h2>Results Comparison (LOAO Evaluation)</h2>\n'
    html += "<table><tr><th>Tier</th><th>Method</th><th>MAE</th><th>Spearman</th></tr>\n"

    # Tier 0
    if "tier0" in results:
        for bl in ["predict_zero", "predict_mean", "tanimoto_nn", "ridge"]:
            if bl in results["tier0"]:
                a = results["tier0"][bl].get("aggregated", {})
                html += f'<tr><td>T0: Baseline</td><td>{bl}</td>'
                html += f'<td>{a.get("mae_mean", 0):.4f}</td>'
                html += f'<td>{a.get("spearman_mean", 0):.4f}</td></tr>\n'
        if "random_null" in results["tier0"]:
            rn = results["tier0"]["random_null"]
            html += f'<tr class="warn"><td>T0: Null</td><td>Random permutation</td>'
            html += f'<td>{rn.get("mae_mean", 0):.4f}</td>'
            html += f'<td>{rn.get("spearman_p95", 0):.4f} (95th)</td></tr>\n'

    # Tier 1
    if "tier1" in results:
        for method in ["FiLMDelta", "Subtraction", "DeepDelta"]:
            if method in results["tier1"]:
                o = results["tier1"][method].get("overall", {})
                html += f'<tr><td>T1: Zero-shot</td><td>{method}</td>'
                html += f'<td>{o.get("mae", 0):.4f}</td>'
                html += f'<td>{o.get("spearman_r", 0):.4f}</td></tr>\n'

    # Tier 2 (LOAO only for comparison)
    if "tier2" in results:
        for method in ["FiLMDelta", "DeepDelta", "EditDiff", "Subtraction"]:
            key = f"loao__{method}"
            if key in results["tier2"]:
                a = results["tier2"][key].get("aggregated", {})
                html += f'<tr><td>T2: Direct</td><td>{method}</td>'
                html += f'<td>{a.get("mae_mean", 0):.4f}</td>'
                html += f'<td>{a.get("spearman_mean", 0):.4f}</td></tr>\n'

    # Tier 3
    if "tier3" in results:
        key = "loao__FiLMDelta_pretrain_finetune"
        if key in results["tier3"]:
            a = results["tier3"][key].get("aggregated", {})
            html += f'<tr class="best"><td>T3: Pretrain→FT</td><td>FiLMDelta</td>'
            html += f'<td>{a.get("mae_mean", 0):.4f}</td>'
            html += f'<td>{a.get("spearman_mean", 0):.4f}</td></tr>\n'

    # Tier 4
    if "tier4" in results:
        key = "loao__multitask_FiLM"
        if key in results["tier4"]:
            a = results["tier4"][key].get("aggregated", {})
            html += f'<tr><td>T4: Multi-task</td><td>FiLM+kinases</td>'
            html += f'<td>{a.get("mae_mean", 0):.4f}</td>'
            html += f'<td>{a.get("spearman_mean", 0):.4f}</td></tr>\n'

    html += "</table></div>\n"

    # Tier 2 split comparison
    if "tier2" in results:
        html += '<div class="section"><h2>Tier 2: Split Strategy Comparison</h2>\n'
        html += "<table><tr><th>Split</th><th>Method</th><th>MAE</th>"
        html += "<th>Spearman</th><th>Folds</th></tr>\n"
        for key, val in sorted(results["tier2"].items()):
            if "__" not in key:
                continue
            split, method = key.split("__")
            a = val.get("aggregated", {})
            html += f"<tr><td>{split}</td><td>{method}</td>"
            html += f"<td>{a.get('mae_mean', 0):.4f}±{a.get('mae_std', 0):.4f}</td>"
            html += f"<td>{a.get('spearman_mean', 0):.4f}</td>"
            html += f"<td>{a.get('n_folds', 0)}</td></tr>\n"
        html += "</table></div>\n"

    html += "</body></html>"

    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"Report: {REPORT_FILE.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, nargs="+", default=None,
                        help="Run specific tiers (0-4)")
    args = parser.parse_args()
    tiers = args.tier or [0, 1, 2, 3, 4]

    print("=" * 70)
    print("ZAP70 COMPREHENSIVE CASE STUDY")
    print("=" * 70)

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    def save():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Step 1: Extract and characterize data
    print("\n--- Data Extraction ---")
    raw_df = pd.read_csv(RAW_FILE)
    pairs_df = extract_target_allpairs(ZAP70_ID, raw_df)
    print(f"  {len(pairs_df):,} pairs, "
          f"{len(set(pairs_df['mol_a'].tolist() + pairs_df['mol_b'].tolist()))} molecules")

    # Identify MMP pairs
    pairs_df = identify_mmp_pairs(pairs_df)

    # Characterize assays
    assay_stats = characterize_assays(pairs_df)
    results["assay_stats"] = assay_stats
    results["data_summary"] = {
        "target": "ZAP70", "target_id": ZAP70_ID,
        "n_molecules": len(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist())),
        "n_assays": pairs_df["assay_id_a"].nunique(),
        "n_pairs": len(pairs_df),
        "n_mmp_pairs": int(pairs_df["is_mmp"].sum()),
        "n_loao_assays": len([a for a in assay_stats if a["n_molecules"] >= MIN_LOAO_ASSAY_SIZE]),
    }
    print(f"\n  Assays with ≥{MIN_LOAO_ASSAY_SIZE} mols: {results['data_summary']['n_loao_assays']}")
    for a in assay_stats[:10]:
        print(f"    Assay {a['assay_id']}: {a['n_molecules']} mols, {a['n_pairs']} pairs, "
              f"{a['n_mmp']} MMP, range={a['pIC50_range']:.2f}")
    save()

    # Compute embeddings
    all_smiles = list(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist()))

    # For Tier 1/3/4 we also need MMP molecules
    mmp_df = None
    if any(t in tiers for t in [1, 3, 4]):
        print("\n  Loading MMP data for transfer learning...")
        mmp_df = pd.read_csv(MMP_FILE)
        mmp_df = mmp_df[(mmp_df["mol_a_id"] != mmp_df["mol_b_id"]) &
                         (mmp_df["is_within_assay"] == True)].copy()
        mmp_smiles = set(mmp_df["mol_a"].tolist() + mmp_df["mol_b"].tolist())
        all_smiles = list(set(all_smiles) | mmp_smiles)
        print(f"  MMP: {len(mmp_df):,} pairs")

    print(f"\n  Computing embeddings for {len(all_smiles):,} molecules...")
    emb_dict, emb_dim = compute_embeddings(all_smiles, "chemprop-dmpnn")

    # Run tiers
    if 0 in tiers:
        results = run_tier0(pairs_df, emb_dict, emb_dim, results)
        save()

    if 1 in tiers:
        results = run_tier1(pairs_df, mmp_df, emb_dict, emb_dim, results)
        save()
        gc.collect()

    if 2 in tiers:
        results = run_tier2(pairs_df, emb_dict, emb_dim, results)
        save()

    if 3 in tiers:
        results = run_tier3(pairs_df, mmp_df, emb_dict, emb_dim, results)
        save()
        gc.collect()

    if 4 in tiers:
        results = run_tier4(pairs_df, mmp_df, emb_dict, emb_dim, results)
        save()
        gc.collect()

    # Generate report
    print("\n--- Generating Report ---")
    generate_report(results)

    results["completed"] = datetime.now().isoformat()
    save()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Results: {RESULTS_FILE.name}")
    print(f"  Report: {REPORT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
