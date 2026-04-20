#!/usr/bin/env python3
"""
20-seed uncertainty estimation for 19 ZAP70 candidates.

Trains FiLMDelta 20 times (different seeds, same full training data) and computes
prediction mean + std + confidence intervals for each candidate.

This gives proper uncertainty bounds: is molecule 9 predicted as 6.25 ± 0.08 or 6.25 ± 0.40?

Usage:
    conda run -n quris python -u experiments/run_zap70_20seed_uncertainty.py
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

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_SEEDS = 20

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
    if mol is None:
        return smi_clean
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)


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


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"20-SEED UNCERTAINTY ESTIMATION for 19 Candidates")
    print("=" * 70)

    # Load data
    mol_data, _ = load_zap70_molecules()
    train_smiles = mol_data['smiles'].tolist()
    train_y = mol_data['pIC50'].values
    n_train = len(train_smiles)
    print(f"  {n_train} ZAP70 training molecules")

    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    # Compute embeddings
    all_smi = list(set(train_smiles + cand_smiles))
    X = compute_fingerprints(all_smi, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smi)}

    # Generate pairs once
    pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(pairs) * 0.1), 100)

    # Train 20 models and collect predictions
    all_preds = np.zeros((N_SEEDS, 19))
    all_rankings = np.zeros((N_SEEDS, 19), dtype=int)

    for seed in range(N_SEEDS):
        seed_t0 = time.time()
        np.random.seed(seed * 17 + 5)
        torch.manual_seed(seed * 17 + 5)

        val_pairs = pairs.sample(n=n_val, random_state=seed * 17 + 5)
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
                if pat >= PATIENCE: break

        if best_st:
            model.load_state_dict(best_st)
        model.cpu().eval()

        # Score candidates
        train_embs_all = scaler.transform(
            np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
        cand_embs_all = scaler.transform(
            np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

        with torch.no_grad():
            for j in range(19):
                a = torch.FloatTensor(train_embs_all)
                b = torch.FloatTensor(np.tile(cand_embs_all[j:j+1], (n_train, 1)))
                deltas = model(a, b).numpy().flatten()
                all_preds[seed, j] = np.median(train_y + deltas)

        ranking = np.argsort(-all_preds[seed])
        all_rankings[seed] = ranking + 1  # 1-indexed

        elapsed = time.time() - seed_t0
        top5 = ranking[:5] + 1
        print(f"  Seed {seed+1:2d}/{N_SEEDS}: top-5={list(top5)}, "
              f"mol9={all_preds[seed,8]:.3f}, mol10={all_preds[seed,9]:.3f}, "
              f"mol17={all_preds[seed,16]:.3f} ({elapsed:.0f}s)")

        del model, scaler
        gc.collect()

    # =========================================================
    # Analysis
    # =========================================================
    print(f"\n{'='*70}")
    print("PREDICTION UNCERTAINTY (20 seeds)")
    print(f"{'='*70}")

    print(f"\n{'Mol':>4s} {'Mean pIC50':>11s} {'Std':>6s} {'95% CI':>16s} {'Range':>16s} {'Mean Rank':>10s}")
    print("-" * 75)

    mol_results = []
    for j in range(19):
        preds = all_preds[:, j]
        mean = np.mean(preds)
        std = np.std(preds)
        ci_lo = np.percentile(preds, 2.5)
        ci_hi = np.percentile(preds, 97.5)
        ranks = np.zeros(N_SEEDS)
        for s in range(N_SEEDS):
            ranks[s] = np.where(np.argsort(-all_preds[s]) == j)[0][0] + 1
        mean_rank = np.mean(ranks)

        marker = " ***" if j+1 in [9, 10, 17] else ""
        print(f"  {j+1:2d}  {mean:9.3f}  {std:5.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  "
              f"[{preds.min():.3f}, {preds.max():.3f}]  {mean_rank:8.1f}{marker}")

        mol_results.append({
            'mol_idx': j + 1,
            'mean_pIC50': float(mean),
            'std_pIC50': float(std),
            'ci_2.5': float(ci_lo),
            'ci_97.5': float(ci_hi),
            'min_pIC50': float(preds.min()),
            'max_pIC50': float(preds.max()),
            'mean_rank': float(mean_rank),
            'all_preds': preds.tolist(),
        })

    # Overlapping CIs
    print(f"\n{'='*70}")
    print("PAIRWISE SEPARABILITY")
    print(f"{'='*70}")
    print("How often does mol A have higher prediction than mol B across 20 seeds?")

    focus_mols = [1, 4, 7, 8, 9, 10, 17, 18, 19]
    print(f"\n  P(row > col):")
    print(f"       ", end="")
    for j in focus_mols:
        print(f"  Mol{j:2d}", end="")
    print()
    for i in focus_mols:
        print(f"  Mol{i:2d}", end="")
        for j in focus_mols:
            if i == j:
                print(f"    ---", end="")
            else:
                p = np.mean(all_preds[:, i-1] > all_preds[:, j-1])
                print(f"   {p:.2f}", end="")
        print()

    # Save
    output = {
        'n_seeds': N_SEEDS,
        'mol_results': mol_results,
        'focus_mols_9_10_17': {
            'mol_9': {'mean': float(np.mean(all_preds[:, 8])),
                      'std': float(np.std(all_preds[:, 8])),
                      'top3_pct': float(np.mean([np.where(np.argsort(-all_preds[s])==8)[0][0] < 3 for s in range(N_SEEDS)]) * 100)},
            'mol_10': {'mean': float(np.mean(all_preds[:, 9])),
                       'std': float(np.std(all_preds[:, 9])),
                       'top3_pct': float(np.mean([np.where(np.argsort(-all_preds[s])==9)[0][0] < 3 for s in range(N_SEEDS)]) * 100)},
            'mol_17': {'mean': float(np.mean(all_preds[:, 16])),
                       'std': float(np.std(all_preds[:, 16])),
                       'top3_pct': float(np.mean([np.where(np.argsort(-all_preds[s])==16)[0][0] < 3 for s in range(N_SEEDS)]) * 100)},
        },
    }

    out_file = RESULTS_DIR / "19_molecules_20seed_uncertainty.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
