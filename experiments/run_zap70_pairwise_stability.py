#!/usr/bin/env python3
"""
Stress-test pairwise FiLMDelta ranking stability for 19 ZAP70 candidates.

For each of N_TRIALS bootstrap/subsample splits of the 280 ZAP70 training molecules:
  1. Train FiLMDelta on a different subset (80% subsample, different seed)
  2. Score all 19 candidates via anchor-based prediction (using only the subsample anchors)
  3. Compute 19×18 pairwise delta matrix
  4. Rank by win count and average delta

Then analyze: how stable are the top-3 (molecules 9, 10, 17)?
- How often does each molecule appear in top-3, top-5?
- What is the rank distribution for each molecule?
- Are there molecules that ALWAYS beat others?

Usage:
    conda run -n quris python -u experiments/run_zap70_pairwise_stability.py
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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints, compute_absolute_metrics,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_TRIALS = 30  # 30 different models from different subsamples
SUBSAMPLE_FRAC = 0.80  # Use 80% of training data each trial

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


def train_filmdelta(train_pairs_df, val_pairs_df, emb_dict, emb_dim, seed=0):
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

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_y))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_y))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            a, b, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                a, b, y = [t.to(device) for t in batch]
                vl += criterion(model(a, b), y).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best_val_loss:
            best_val_loss = vl; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model, scaler


def score_and_rank_pairwise(model, scaler, train_smiles, train_pIC50,
                             cand_smiles, emb_dict, emb_dim):
    """Score 19 candidates and compute pairwise ranking."""
    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

    train_embs = scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    cand_embs = scaler.transform(np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

    n_train = len(train_smiles)
    n_cand = len(cand_smiles)

    # Absolute predictions via anchor-based method
    abs_preds = np.zeros(n_cand)
    model.cpu().eval()
    with torch.no_grad():
        for j in range(n_cand):
            a = torch.FloatTensor(train_embs)
            b = torch.FloatTensor(np.tile(cand_embs[j:j+1], (n_train, 1)))
            deltas = model(a, b).numpy().flatten()
            abs_preds[j] = np.median(train_pIC50 + deltas)

    # Pairwise delta matrix: delta[i][j] = predicted delta from candidate i to candidate j
    delta_matrix = np.zeros((n_cand, n_cand))
    with torch.no_grad():
        for i in range(n_cand):
            for j in range(n_cand):
                if i == j:
                    continue
                a = torch.FloatTensor(cand_embs[i:i+1])
                b = torch.FloatTensor(cand_embs[j:j+1])
                delta_matrix[i][j] = model(a, b).item()

    # Win counts: mol j wins over mol i if delta[i][j] > 0
    win_counts = np.zeros(n_cand, dtype=int)
    for j in range(n_cand):
        for i in range(n_cand):
            if i != j and delta_matrix[i][j] > 0:
                win_counts[j] += 1

    # Average delta (higher = better predicted)
    avg_delta = np.zeros(n_cand)
    for j in range(n_cand):
        deltas_to_j = [delta_matrix[i][j] for i in range(n_cand) if i != j]
        avg_delta[j] = np.mean(deltas_to_j)

    # Rank by avg_delta (descending)
    ranking = np.argsort(-avg_delta)  # 0-indexed

    return {
        'abs_preds': abs_preds,
        'delta_matrix': delta_matrix,
        'win_counts': win_counts,
        'avg_delta': avg_delta,
        'ranking': ranking,  # 0-indexed
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"PAIRWISE RANKING STABILITY TEST ({N_TRIALS} trials)")
    print("=" * 70)

    # Load data
    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n_total = len(all_smiles)
    print(f"  {n_total} ZAP70 training molecules")

    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    # Compute embeddings
    all_smi_set = list(set(all_smiles + cand_smiles))
    X = compute_fingerprints(all_smi_set, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smi_set)}

    # Track rankings across trials
    all_rankings = []  # list of 0-indexed ranking arrays
    all_abs_preds = []
    all_win_counts = []
    all_avg_deltas = []

    n_subsample = int(n_total * SUBSAMPLE_FRAC)

    for trial in range(N_TRIALS):
        trial_t0 = time.time()
        trial_seed = trial * 7 + 13  # Different seed each trial

        # Subsample training molecules
        np.random.seed(trial_seed)
        idx = np.random.choice(n_total, size=n_subsample, replace=False)
        sub_smiles = [all_smiles[i] for i in idx]
        sub_y = all_y[idx]

        # Generate pairs
        pairs = generate_all_pairs(sub_smiles, sub_y)
        n_val = max(int(len(pairs) * 0.1), 100)
        val_pairs = pairs.sample(n=n_val, random_state=trial_seed)
        trn_pairs = pairs.drop(val_pairs.index)

        # Train
        model, scaler = train_filmdelta(trn_pairs, val_pairs, emb_dict, emb_dim, seed=trial_seed)

        # Score and rank
        result = score_and_rank_pairwise(model, scaler, sub_smiles, sub_y,
                                          cand_smiles, emb_dict, emb_dim)

        all_rankings.append(result['ranking'])
        all_abs_preds.append(result['abs_preds'])
        all_win_counts.append(result['win_counts'])
        all_avg_deltas.append(result['avg_delta'])

        # Print progress
        top5 = result['ranking'][:5] + 1  # 1-indexed
        elapsed = time.time() - trial_t0
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: top-5 = {list(top5)} "
              f"| mol9 rank={np.where(result['ranking']==8)[0][0]+1:2d} "
              f"| mol10 rank={np.where(result['ranking']==9)[0][0]+1:2d} "
              f"| mol17 rank={np.where(result['ranking']==16)[0][0]+1:2d} "
              f"({elapsed:.0f}s)")

        del model, scaler
        gc.collect()

    # =========================================================
    # Analysis
    # =========================================================
    print(f"\n{'='*70}")
    print("STABILITY ANALYSIS")
    print(f"{'='*70}")

    rankings_matrix = np.array(all_rankings)  # [N_TRIALS, 19] - position indices
    abs_preds_matrix = np.array(all_abs_preds)  # [N_TRIALS, 19] - predicted pIC50
    win_counts_matrix = np.array(all_win_counts)  # [N_TRIALS, 19]
    avg_deltas_matrix = np.array(all_avg_deltas)  # [N_TRIALS, 19]

    # Compute rank of each molecule across trials
    rank_of_mol = np.zeros((N_TRIALS, 19), dtype=int)
    for t in range(N_TRIALS):
        for pos, mol_idx in enumerate(rankings_matrix[t]):
            rank_of_mol[t, mol_idx] = pos + 1  # 1-indexed rank

    # Summary table
    print(f"\n{'Mol':>4s} {'Mean Rank':>10s} {'Std':>6s} {'Median':>7s} {'Min':>5s} {'Max':>5s} "
          f"{'Top-3%':>7s} {'Top-5%':>7s} {'Wins(mean)':>11s} {'Pred pIC50':>11s}")
    print("-" * 95)

    mol_stats = []
    for mol_idx in range(19):
        ranks = rank_of_mol[:, mol_idx]
        preds = abs_preds_matrix[:, mol_idx]
        wins = win_counts_matrix[:, mol_idx]
        top3_pct = np.mean(ranks <= 3) * 100
        top5_pct = np.mean(ranks <= 5) * 100
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        median_rank = np.median(ranks)

        marker = " ***" if mol_idx + 1 in [9, 10, 17] else ""
        print(f"  {mol_idx+1:2d}  {mean_rank:8.1f}  {std_rank:5.1f}  {median_rank:6.0f}  "
              f"{ranks.min():4d}  {ranks.max():4d}  {top3_pct:5.0f}%  {top5_pct:5.0f}%  "
              f"{wins.mean():9.1f}  {preds.mean():9.3f}±{preds.std():.3f}{marker}")

        mol_stats.append({
            'mol_idx': mol_idx + 1,
            'mean_rank': float(mean_rank),
            'std_rank': float(std_rank),
            'median_rank': float(median_rank),
            'min_rank': int(ranks.min()),
            'max_rank': int(ranks.max()),
            'top3_pct': float(top3_pct),
            'top5_pct': float(top5_pct),
            'mean_wins': float(wins.mean()),
            'mean_pred': float(preds.mean()),
            'std_pred': float(preds.std()),
        })

    # Focus on molecules 9, 10, 17
    print(f"\n{'='*70}")
    print("FOCUS: Molecules 9, 10, 17")
    print(f"{'='*70}")

    for mol_id in [9, 10, 17]:
        mol_idx = mol_id - 1
        ranks = rank_of_mol[:, mol_idx]
        preds = abs_preds_matrix[:, mol_idx]
        print(f"\n  Mol {mol_id}:")
        print(f"    Rank distribution: {sorted(ranks)}")
        print(f"    Mean rank: {np.mean(ranks):.1f} ± {np.std(ranks):.1f}")
        print(f"    Top-3 frequency: {np.mean(ranks<=3)*100:.0f}%")
        print(f"    Top-5 frequency: {np.mean(ranks<=5)*100:.0f}%")
        print(f"    Pred pIC50: {preds.mean():.3f} ± {preds.std():.3f}")

    # Pairwise dominance: how often does mol A beat mol B?
    print(f"\n{'='*70}")
    print("PAIRWISE DOMINANCE (key matchups)")
    print(f"{'='*70}")

    key_mols = [9, 10, 17, 1, 4, 7, 8, 18, 19]  # Include top consensus + challengers
    print(f"\n  P(row beats col) across {N_TRIALS} trials:")
    print(f"       ", end="")
    for j in key_mols:
        print(f"  Mol{j:2d}", end="")
    print()
    for i in key_mols:
        print(f"  Mol{i:2d}", end="")
        for j in key_mols:
            if i == j:
                print(f"    ---", end="")
            else:
                # How often does i beat j (i has lower rank = better)
                p_win = np.mean(rank_of_mol[:, i-1] < rank_of_mol[:, j-1])
                print(f"   {p_win:.2f}", end="")
        print()

    # Overall consensus ranking (by mean rank)
    mean_ranks = np.array([s['mean_rank'] for s in mol_stats])
    consensus_order = np.argsort(mean_ranks)  # 0-indexed
    print(f"\n{'='*70}")
    print("CONSENSUS RANKING (by mean rank across {N_TRIALS} trials)")
    print(f"{'='*70}")
    for pos, mol_idx in enumerate(consensus_order):
        s = mol_stats[mol_idx]
        marker = " <<<" if s['mol_idx'] in [9, 10, 17] else ""
        print(f"  {pos+1:2d}. Mol {s['mol_idx']:2d}: "
              f"mean_rank={s['mean_rank']:.1f}±{s['std_rank']:.1f}, "
              f"top3={s['top3_pct']:.0f}%, wins={s['mean_wins']:.1f}, "
              f"pIC50={s['mean_pred']:.3f}±{s['std_pred']:.3f}{marker}")

    # Rank correlation between trials
    from scipy.stats import spearmanr, kendalltau
    trial_corrs = []
    for i in range(N_TRIALS):
        for j in range(i+1, N_TRIALS):
            r, _ = spearmanr(rank_of_mol[i], rank_of_mol[j])
            trial_corrs.append(r)
    print(f"\nInter-trial rank correlation (Spearman):")
    print(f"  Mean: {np.mean(trial_corrs):.3f}")
    print(f"  Min:  {np.min(trial_corrs):.3f}")
    print(f"  Max:  {np.max(trial_corrs):.3f}")
    print(f"  Std:  {np.std(trial_corrs):.3f}")

    # Save results
    output = {
        'n_trials': N_TRIALS,
        'subsample_frac': SUBSAMPLE_FRAC,
        'mol_stats': mol_stats,
        'consensus_ranking': [int(consensus_order[i] + 1) for i in range(19)],
        'inter_trial_spearman_mean': float(np.mean(trial_corrs)),
        'inter_trial_spearman_std': float(np.std(trial_corrs)),
        'inter_trial_spearman_min': float(np.min(trial_corrs)),
        'rank_distributions': {
            str(mol_idx+1): rank_of_mol[:, mol_idx].tolist()
            for mol_idx in range(19)
        },
        'pred_distributions': {
            str(mol_idx+1): abs_preds_matrix[:, mol_idx].tolist()
            for mol_idx in range(19)
        },
    }

    out_file = RESULTS_DIR / "19_molecules_pairwise_stability.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
