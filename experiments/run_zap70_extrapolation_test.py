#!/usr/bin/env python3
"""
Extrapolation test: How well does FiLMDelta predict ZAP70 molecules
that are structurally distant (max Tc < 0.3) from the training set?

This directly simulates the same regime as the 19 candidates.

Usage:
    conda run -n quris python -u experiments/run_zap70_extrapolation_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_SEEDS = 10
TC_THRESHOLD = 0.3


def compute_tanimoto_matrix(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
    n = len(fps)
    tc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tc = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            tc_matrix[i, j] = tc
            tc_matrix[j, i] = tc
    return tc_matrix


def generate_all_pairs(smiles, pIC50):
    pairs = []
    n = len(smiles)
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append({
                    'mol_a': smiles[i], 'mol_b': smiles[j],
                    'delta': float(pIC50[j] - pIC50[i]),
                })
    return pd.DataFrame(pairs)


def train_and_predict(train_smiles, train_y, test_smiles, test_y, emb_dict, emb_dim, seed):
    """Train FiLMDelta on train set, predict test set via anchor-based scoring."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate training pairs
    pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(pairs) * 0.1), 100)
    val_pairs = pairs.sample(n=n_val, random_state=seed)
    trn_pairs = pairs.drop(val_pairs.index)

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in trn_pairs['mol_a']])
    train_b = np.array([get_emb(s) for s in trn_pairs['mol_b']])
    train_yp = trn_pairs['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_pairs['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs['mol_b']])
    val_yp = val_pairs['delta'].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_yp))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_yp))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_vl, pat, best_st = float('inf'), 0, None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            criterion(model(a, b), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(device) for t in batch]
                vl += criterion(model(a, b), y).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best_vl:
            best_vl = vl; pat = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best_st:
        model.load_state_dict(best_st)
    model.cpu().eval()

    # Anchor-based prediction for test molecules
    n_train = len(train_smiles)
    n_test = len(test_smiles)
    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(
        np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

    preds = np.zeros(n_test)
    with torch.no_grad():
        for j in range(n_test):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(test_embs[j:j+1], (n_train, 1)))
            deltas = model(a, b).numpy().flatten()
            preds[j] = np.median(train_y + deltas)

    del model
    gc.collect()
    return preds, scaler


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXTRAPOLATION TEST: FiLMDelta on Distant ZAP70 Molecules")
    print("=" * 70)

    # Load data
    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n_total = len(all_smiles)
    print(f"  Total ZAP70 molecules: {n_total}")

    # Compute Tanimoto matrix
    print("  Computing Tanimoto similarity matrix...")
    tc_matrix = compute_tanimoto_matrix(all_smiles)
    np.fill_diagonal(tc_matrix, 0)
    max_tc = tc_matrix.max(axis=1)

    # Split: test = molecules with max Tc < threshold
    test_mask = max_tc < TC_THRESHOLD
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    test_smiles = [all_smiles[i] for i in test_idx]
    test_y = all_y[test_idx]
    train_smiles = [all_smiles[i] for i in train_idx]
    train_y = all_y[train_idx]

    n_test = len(test_smiles)
    n_train = len(train_smiles)

    # Compute max Tc of each test mol to TRAINING set only
    test_tc_to_train = np.zeros(n_test)
    for i, ti in enumerate(test_idx):
        test_tc_to_train[i] = tc_matrix[ti, train_idx].max()

    print(f"\n  Train: {n_train} molecules (max Tc >= {TC_THRESHOLD})")
    print(f"  Test:  {n_test} molecules (max Tc < {TC_THRESHOLD} to ANY other mol)")
    print(f"  Test max Tc to training set: {test_tc_to_train.min():.3f} - {test_tc_to_train.max():.3f} "
          f"(mean {test_tc_to_train.mean():.3f})")
    print(f"  Test pIC50: {test_y.min():.2f} - {test_y.max():.2f} (mean {test_y.mean():.2f})")
    print(f"  Train pIC50: {train_y.min():.2f} - {train_y.max():.2f} (mean {train_y.mean():.2f})")

    print(f"\n  Test molecules:")
    for i in range(n_test):
        print(f"    {i+1:2d}. pIC50={test_y[i]:.2f}, maxTc_to_train={test_tc_to_train[i]:.3f}, "
              f"SMILES={test_smiles[i][:55]}")

    # Compute embeddings
    all_smi_unique = list(set(train_smiles + test_smiles))
    X = compute_fingerprints(all_smi_unique, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smi_unique)}

    # Train N_SEEDS models
    print(f"\n  Training {N_SEEDS} FiLMDelta models...")
    all_preds = np.zeros((N_SEEDS, n_test))

    for seed in range(N_SEEDS):
        seed_t0 = time.time()
        actual_seed = seed * 17 + 5
        preds, _ = train_and_predict(train_smiles, train_y, test_smiles, test_y,
                                      emb_dict, emb_dim, actual_seed)
        all_preds[seed] = preds
        mae = np.mean(np.abs(preds - test_y))
        spr = stats.spearmanr(preds, test_y).correlation if n_test > 3 else 0
        elapsed = time.time() - seed_t0
        print(f"    Seed {seed+1:2d}/{N_SEEDS}: MAE={mae:.3f}, Spearman={spr:.3f} ({elapsed:.0f}s)")

    # =========================================================
    # Aggregate results
    # =========================================================
    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)

    mae_per_seed = np.array([np.mean(np.abs(all_preds[s] - test_y)) for s in range(N_SEEDS)])
    spr_per_seed = np.array([stats.spearmanr(all_preds[s], test_y).correlation for s in range(N_SEEDS)])

    mae_mean_pred = np.mean(np.abs(mean_preds - test_y))
    spr_mean_pred = stats.spearmanr(mean_preds, test_y).correlation if n_test > 3 else 0
    pearson_mean_pred = stats.pearsonr(mean_preds, test_y).statistic if n_test > 3 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: Extrapolation Test (max Tc < {TC_THRESHOLD})")
    print(f"{'='*70}")
    print(f"  Per-seed MAE:      {mae_per_seed.mean():.3f} +/- {mae_per_seed.std():.3f}")
    print(f"  Per-seed Spearman: {spr_per_seed.mean():.3f} +/- {spr_per_seed.std():.3f}")
    print(f"  Ensemble (mean of {N_SEEDS} seeds):")
    print(f"    MAE:     {mae_mean_pred:.3f}")
    print(f"    Spearman: {spr_mean_pred:.3f}")
    print(f"    Pearson:  {pearson_mean_pred:.3f}")

    # Per-molecule breakdown
    print(f"\n{'Mol':>4s} {'True pIC50':>11s} {'Pred pIC50':>11s} {'Error':>7s} {'Std':>6s} {'MaxTc':>6s}")
    print("-" * 55)
    for i in range(n_test):
        err = mean_preds[i] - test_y[i]
        print(f"  {i+1:2d}  {test_y[i]:9.2f}  {mean_preds[i]:9.3f}  {err:+6.3f}  {std_preds[i]:5.3f}  {test_tc_to_train[i]:.3f}")

    # Compare with training set mean prediction (naive baseline)
    train_mean = train_y.mean()
    naive_mae = np.mean(np.abs(train_mean - test_y))
    print(f"\n  Naive baseline (predict train mean={train_mean:.2f}): MAE={naive_mae:.3f}")
    print(f"  FiLMDelta improvement over naive: {(1 - mae_mean_pred/naive_mae)*100:.1f}%")

    # Rank accuracy
    true_ranking = np.argsort(-test_y)
    pred_ranking = np.argsort(-mean_preds)
    print(f"\n  True top-3:      {[i+1 for i in true_ranking[:3]]}")
    print(f"  Predicted top-3: {[i+1 for i in pred_ranking[:3]]}")
    top3_overlap = len(set(true_ranking[:3]) & set(pred_ranking[:3]))
    print(f"  Top-3 overlap:   {top3_overlap}/3")

    if n_test >= 5:
        top5_overlap = len(set(true_ranking[:5]) & set(pred_ranking[:5]))
        print(f"  True top-5:      {[i+1 for i in true_ranking[:5]]}")
        print(f"  Predicted top-5: {[i+1 for i in pred_ranking[:5]]}")
        print(f"  Top-5 overlap:   {top5_overlap}/5")

    # Also test with broader thresholds
    print(f"\n{'='*70}")
    print("SENSITIVITY: Different Tc thresholds")
    print(f"{'='*70}")
    for tc_thresh in [0.25, 0.28, 0.30, 0.35, 0.40]:
        mask = max_tc < tc_thresh
        n_t = mask.sum()
        if n_t < 3:
            print(f"  Tc < {tc_thresh}: {n_t} molecules (too few)")
            continue
        t_idx = np.where(mask)[0]
        t_smiles = [all_smiles[i] for i in t_idx]
        t_y = all_y[t_idx]
        tr_idx = np.where(~mask)[0]
        tr_smiles = [all_smiles[i] for i in tr_idx]
        tr_y = all_y[tr_idx]

        # Quick single-seed test
        preds_quick, _ = train_and_predict(tr_smiles, tr_y, t_smiles, t_y,
                                            emb_dict, emb_dim, seed=42)
        mae_q = np.mean(np.abs(preds_quick - t_y))
        spr_q = stats.spearmanr(preds_quick, t_y).correlation if n_t > 3 else 0
        naive_q = np.mean(np.abs(tr_y.mean() - t_y))

        # Max Tc of test to train
        tc_to_tr = np.array([tc_matrix[ti, tr_idx].max() for ti in t_idx])

        print(f"  Tc < {tc_thresh}: {n_t} test mols, maxTc_to_train={tc_to_tr.mean():.3f}, "
              f"MAE={mae_q:.3f} (naive={naive_q:.3f}), Spr={spr_q:.3f}")

    # Save results
    output = {
        'tc_threshold': TC_THRESHOLD,
        'n_train': n_train,
        'n_test': n_test,
        'n_seeds': N_SEEDS,
        'test_max_tc_to_train': test_tc_to_train.tolist(),
        'test_pIC50': test_y.tolist(),
        'ensemble_mae': float(mae_mean_pred),
        'ensemble_spearman': float(spr_mean_pred),
        'ensemble_pearson': float(pearson_mean_pred),
        'per_seed_mae_mean': float(mae_per_seed.mean()),
        'per_seed_mae_std': float(mae_per_seed.std()),
        'per_seed_spearman_mean': float(spr_per_seed.mean()),
        'per_seed_spearman_std': float(spr_per_seed.std()),
        'naive_baseline_mae': float(naive_mae),
        'per_molecule': [
            {
                'idx': int(i + 1),
                'smiles': test_smiles[i],
                'true_pIC50': float(test_y[i]),
                'pred_pIC50': float(mean_preds[i]),
                'pred_std': float(std_preds[i]),
                'error': float(mean_preds[i] - test_y[i]),
                'max_tc_to_train': float(test_tc_to_train[i]),
            }
            for i in range(n_test)
        ],
    }

    out_file = RESULTS_DIR / "extrapolation_test_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
