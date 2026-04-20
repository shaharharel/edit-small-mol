#!/usr/bin/env python3
"""
ZAP70 Embedding-based Models — ChemBERTa + MoLFormer on ZAP70 task.

Trains FiLMDelta with pretrained embeddings on ZAP70 all-pairs,
then scores 19 candidates. Also runs challenging CV splits.

Usage:
    conda run -n quris python -u experiments/run_zap70_embeddings.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
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
from itertools import combinations
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

warnings.filterwarnings("ignore")
torch.backends.mps.is_available = lambda: False  # MPS crashes with ChemBERTa
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import RESULTS_DIR, CACHE_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_absolute_metrics, aggregate_cv_results,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
N_FOLDS = 5
CV_SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 15
DEVICE = "cpu"

SMILES_19 = [
    'C=CC(=O)N1CC=2C=CC=C(C(=O)NC3=CN(C=N3)C(C)C)C2C1',
    'C=CC(=O)N1CC2(CCC(=O)NC3=CNN=C3C(=O)NCC(C)O)CCC1CC2',
    'C=CC(=O)NC[C@H]1C[C@H]2C[C@@H]1CN2C=3N=CN=C(N)C3Cl',
    'C=CC(=O)N1CCCC1(C(=O)NC2=CNN=C2OCC(F)F)C=3C=CC=CC3',
    'C=CC(=O)N1CCC(CC(=O)NC2=CNN=C2C=3C=NC=CN3)CC41CC4',
    'C=CC(=O)N1CC2(CCC2)CC1CC(=O)NC3=CNN=C3C(=O)NCC(C)O',
    'C=CC(=O)N1C[C@H](CC(C)(C)C)[C@H](C1)C(=O)NC2=CNN=C2C(=O)NCC(C)O',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C3=CNC=4C=C(F)C(Cl)=CC34)S2',
    'C=CC(=O)NC1C2C3C[C@@H]1[C@H](C(=O)NC4=CNN=C4C=5C=NC=CN5)C32',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C=3C=CC(F)=C(C3)S(=O)(=O)N(C)C)S2',
    'C=CC(=O)NCC1=CN(N=N1)[C@@H]2C[C@H](C2)C(=O)NC=3C=CC=NC3NC(C)=O',
    'C=CC(=O)N1CC2(CC1CCC2)NC=3N=CC=C(N3)OC=4C=CC=C(C#N)C4',
    'C=CC(=O)NC(C)C=1N=CC(=CN1)NC(=O)C=2N=CN=C3NC=C(C)C23',
    'C=CC(=O)N(C)C1(CNC=2N=CN=C(N)C2C(=O)OCC)CCC1',
    'C=CC(=O)N1CCC[C@H]1C(C)NC(=O)C=2C=NNC2C=3C=CN(C)N3',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CC(CCNC=2C=C(N=CN2)NC=3C=CC=CC3)(C1)N(C)C',
    'O=C(O)C(F)(F)F.C=CC(=O)N(C)C1(CNC=2N=CN=C3NC=C(C(N)=O)C23)CCC1',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CCN(CC1)C=2N=CN=C(N2)NC3(CC)CCNCC3',
    'C=CC(=O)N1CCCC(NC(=O)C2=CNN=C2C3=CC=4C=CC=CC4O3)C51CCC5',
]


def clean_smiles(smi):
    smi_clean = smi.split(' |')[0] if ' |' in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)


def generate_all_pairs(smiles, pIC50):
    pairs = []
    for (i, j) in combinations(range(len(smiles)), 2):
        pairs.append({'mol_a': smiles[i], 'mol_b': smiles[j], 'delta': pIC50[j] - pIC50[i]})
        pairs.append({'mol_a': smiles[j], 'mol_b': smiles[i], 'delta': pIC50[i] - pIC50[j]})
    return pd.DataFrame(pairs)


def compute_embeddings_for_smiles(smiles_list, embedder_name):
    """Compute embeddings using specified embedder."""
    if embedder_name == "chemberta2-mlm":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name="chemberta2-mlm")
    elif embedder_name == "chemberta2-mtr":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name="chemberta2-mtr")
    elif embedder_name == "molformer":
        from src.embedding.molformer import MoLFormerEmbedder
        embedder = MoLFormerEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")

    embeddings = embedder.encode(smiles_list)
    emb_dim = embeddings.shape[1]
    emb_dict = {smi: embeddings[i] for i, smi in enumerate(smiles_list)}
    return emb_dict, emb_dim


def train_filmdelta(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0):
    """Train FiLMDelta on pairs with given embeddings."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in train_pairs_df['mol_a']])
    train_b = np.array([get_emb(s) for s in train_pairs_df['mol_b']])
    train_y = train_pairs_df['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_pairs_df['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs_df['mol_b']])
    val_y = val_pairs_df['delta'].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_y))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_y))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FiLMDeltaMLP(input_dim=emb_dim, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, y = batch
            optimizer.zero_grad()
            pred = model(a, b)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = batch
                pred = model(a, b)
                val_loss += criterion(pred, y).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"      Trained {epoch+1} epochs, best val loss: {best_val_loss:.4f}")
    return model, scaler


def score_candidates(model, scaler, train_smiles, train_pIC50, cand_smiles, emb_dict, emb_dim):
    """Anchor-based absolute prediction."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_embs = scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    cand_embs = scaler.transform(np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

    n_train = len(train_smiles)
    n_cand = len(cand_smiles)
    abs_preds = np.zeros(n_cand)

    model.eval()
    with torch.no_grad():
        for j in range(n_cand):
            anchor_embs = torch.FloatTensor(train_embs)
            target_embs = torch.FloatTensor(np.tile(cand_embs[j:j+1], (n_train, 1)))
            deltas = model(anchor_embs, target_embs).numpy().flatten()
            abs_preds[j] = np.median(train_pIC50 + deltas)

    return abs_preds


def main():
    print("=" * 70)
    print("ZAP70 EMBEDDING-BASED MODELS — Scoring 19 Candidates")
    print("=" * 70)

    # Load data
    print("\n[1] Loading ZAP70 data...")
    mol_data, _ = load_zap70_molecules()
    train_smiles = mol_data['smiles'].tolist()
    train_y = mol_data['pIC50'].values
    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    all_smiles = list(set(train_smiles + cand_smiles))
    print(f"  Total unique SMILES: {len(all_smiles)}")

    # Generate all-pairs
    all_pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(all_pairs) * 0.1), 100)
    val_pairs = all_pairs.sample(n=n_val, random_state=42)
    trn_pairs = all_pairs.drop(val_pairs.index)
    print(f"  All-pairs: {len(all_pairs)} total, {len(trn_pairs)} train, {len(val_pairs)} val")

    embedders = ["chemberta2-mlm", "chemberta2-mtr"]  # MoLFormer takes too long
    n_seeds = 3
    results = {}

    for emb_name in embedders:
        print(f"\n{'='*70}")
        print(f"[EMBEDDER] {emb_name}")
        print(f"{'='*70}")

        print(f"  Computing embeddings...")
        t0 = time.time()
        emb_dict, emb_dim = compute_embeddings_for_smiles(all_smiles, emb_name)
        print(f"  Embeddings: {emb_dim}d, computed in {time.time()-t0:.1f}s")

        # === PART A: Score 19 candidates ===
        print(f"\n  Scoring 19 candidates ({n_seeds} seeds)...")
        preds_all = np.zeros((19, n_seeds))
        for seed in range(n_seeds):
            print(f"    Seed {seed}...")
            model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed)
            preds = score_candidates(model, scaler, train_smiles, train_y, cand_smiles, emb_dict, emb_dim)
            preds_all[:, seed] = preds
            del model, scaler
            gc.collect()

        pred_mean = preds_all.mean(axis=1)
        pred_std = preds_all.std(axis=1)

        print(f"\n  {emb_name} Rankings:")
        ranking = np.argsort(-pred_mean)
        for rank, j in enumerate(ranking[:10]):
            print(f"    {rank+1}. Mol {j+1}: {pred_mean[j]:.3f} ± {pred_std[j]:.3f}")

        # === PART B: 5-fold CV ===
        print(f"\n  5-fold CV evaluation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(mol_data)):
            train_data = mol_data.iloc[train_idx]
            test_data = mol_data.iloc[test_idx]

            fold_train_smi = train_data['smiles'].tolist()
            fold_train_y = train_data['pIC50'].values
            fold_test_smi = test_data['smiles'].tolist()
            fold_test_y = test_data['pIC50'].values

            fold_pairs = generate_all_pairs(fold_train_smi, fold_train_y)
            n_fv = max(int(len(fold_pairs) * 0.1), 100)
            fv_pairs = fold_pairs.sample(n=n_fv, random_state=42)
            ft_pairs = fold_pairs.drop(fv_pairs.index)

            model, scaler = train_filmdelta(ft_pairs, fv_pairs, emb_dict, emb_dim, seed=fold_idx)
            preds = score_candidates(model, scaler, fold_train_smi, fold_train_y,
                                     fold_test_smi, emb_dict, emb_dim)
            metrics = compute_absolute_metrics(fold_test_y, preds)
            fold_metrics.append(metrics)
            print(f"    Fold {fold_idx}: MAE={metrics['mae']:.3f}, Spr={metrics['spearman_r']:.3f}")
            del model, scaler
            gc.collect()

        agg = aggregate_cv_results(fold_metrics)
        print(f"  {emb_name} CV: MAE={agg['mae_mean']:.3f}±{agg['mae_std']:.3f}, "
              f"Spr={agg['spearman_r_mean']:.3f}±{agg['spearman_r_std']:.3f}")

        results[emb_name] = {
            "emb_dim": emb_dim,
            "candidates": [
                {"idx": j+1, "smiles": cand_smiles[j],
                 "pred_mean": float(pred_mean[j]), "pred_std": float(pred_std[j])}
                for j in range(19)
            ],
            "ranking": [int(r+1) for r in ranking],
            "cv_metrics": agg,
        }

    # Save results
    out_path = RESULTS_DIR / "19_molecules_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Comparison table
    print("\n" + "=" * 70)
    print("EMBEDDING COMPARISON — 19 Candidate Rankings")
    print("=" * 70)
    print(f"{'Idx':>3}", end="")
    for emb_name in embedders:
        print(f" {emb_name:>16}", end="")
    print()
    for j in range(19):
        print(f"{j+1:>3}", end="")
        for emb_name in embedders:
            pred = results[emb_name]["candidates"][j]["pred_mean"]
            print(f" {pred:>16.3f}", end="")
        print()


if __name__ == "__main__":
    main()
