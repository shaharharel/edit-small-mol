#!/usr/bin/env python3
"""CLEAN variant of run_zap70_pairwise_stability.py — molecule-disjoint val split.

For each of N_TRIALS bootstrap/subsample splits of the 280 ZAP70 molecules:
  1. Subsample 80% (224 mols) as the trial pool
  2. Hold out 10% of the pool (~22 mols) as a MOLECULE-DISJOINT val set
  3. All pairs touching the held-out mols go to val; remaining pairs train
  4. Train FiLMDelta on the molecule-disjoint train pairs
  5. Score 19 candidates via anchor prediction over the TRAIN MOLECULES only
  6. Rank by win count and average delta over candidates

Matches the protocol of `reinvent4_film_model_clean.pt` (mol-disjoint val).

Output: results/paper_evaluation/19_molecules_pairwise_stability_clean.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import builtins
_op = builtins.print
def print(*a, **k):
    k.setdefault('flush', True); _op(*a, **k)
builtins.print = print

import gc, json, time, warnings
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
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_TRIALS = 30
SUBSAMPLE_FRAC = 0.80
HOLDOUT_FRAC = 0.10  # within the subsample, fraction held out for mol-disjoint val

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
    return Chem.MolToSmiles(SaltRemover().StripMol(mol))


def build_mol_disjoint_pairs(sub_smiles, sub_y, held_set):
    """Returns (train_pairs_df, val_pairs_df) with molecule-disjoint val split."""
    n = len(sub_smiles)
    train_rows, val_rows = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = {'mol_a': sub_smiles[i], 'mol_b': sub_smiles[j],
                 'delta': float(sub_y[j] - sub_y[i])}
            if (i in held_set) or (j in held_set):
                val_rows.append(r)
            else:
                train_rows.append(r)
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def train_filmdelta(train_df, val_df, emb_dict, emb_dim, seed):
    np.random.seed(seed); torch.manual_seed(seed)

    def get_emb(s):
        return emb_dict.get(s, np.zeros(emb_dim, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in train_df['mol_a']])
    train_b = np.array([get_emb(s) for s in train_df['mol_b']])
    train_y = train_df['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_df['mol_a']])
    val_b = np.array([get_emb(s) for s in val_df['mol_b']])
    val_y = val_df['delta'].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a, train_b]))  # fit on train only (no leak)
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    tr_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_y))
    vl_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_y))
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE)
    model = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    best, pat, best_st = float('inf'), 0, None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in tr_ld:
            a, b, y = [t.to(device) for t in batch]
            opt.zero_grad()
            crit(model(a, b), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in vl_ld:
                a, b, y = [t.to(device) for t in batch]
                vl += crit(model(a, b), y).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best:
            best, pat = vl, 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE: break

    if best_st:
        model.load_state_dict(best_st)
    model.cpu().eval()
    return model, scaler, best


def score_and_rank(model, scaler, anchor_smiles, anchor_y, cand_smiles, emb_dict, emb_dim):
    """Anchor prediction over `anchor_smiles` (train molecules only)."""
    def get_emb(s):
        return emb_dict.get(s, np.zeros(emb_dim, dtype=np.float32))

    anchor_embs = scaler.transform(np.array([get_emb(s) for s in anchor_smiles])).astype(np.float32)
    cand_embs = scaler.transform(np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)
    n_anchor, n_cand = len(anchor_smiles), len(cand_smiles)

    abs_preds = np.zeros(n_cand)
    with torch.no_grad():
        for j in range(n_cand):
            a = torch.FloatTensor(anchor_embs)
            b = torch.FloatTensor(np.tile(cand_embs[j:j+1], (n_anchor, 1)))
            deltas = model(a, b).numpy().flatten()
            abs_preds[j] = np.median(anchor_y + deltas)

    delta_matrix = np.zeros((n_cand, n_cand))
    with torch.no_grad():
        for i in range(n_cand):
            for j in range(n_cand):
                if i == j: continue
                a = torch.FloatTensor(cand_embs[i:i+1])
                b = torch.FloatTensor(cand_embs[j:j+1])
                delta_matrix[i][j] = model(a, b).item()

    win_counts = np.zeros(n_cand, dtype=int)
    for j in range(n_cand):
        for i in range(n_cand):
            if i != j and delta_matrix[i][j] > 0:
                win_counts[j] += 1

    avg_delta = np.array([
        np.mean([delta_matrix[i][j] for i in range(n_cand) if i != j])
        for j in range(n_cand)
    ])
    ranking = np.argsort(-avg_delta)
    return {'abs_preds': abs_preds, 'delta_matrix': delta_matrix,
            'win_counts': win_counts, 'avg_delta': avg_delta, 'ranking': ranking}


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"CLEAN PAIRWISE STABILITY ({N_TRIALS} trials, mol-disjoint val)")
    print("=" * 70)

    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n_total = len(all_smiles)
    print(f"  {n_total} ZAP70 molecules")

    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    all_smi_set = list(set(all_smiles + cand_smiles))
    X = compute_fingerprints(all_smi_set, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {s: X[i] for i, s in enumerate(all_smi_set)}

    all_rankings = []
    all_abs_preds = []
    all_win_counts = []
    all_avg_deltas = []
    all_val_mae = []

    n_sub = int(n_total * SUBSAMPLE_FRAC)

    for trial in range(N_TRIALS):
        t1 = time.time()
        trial_seed = trial * 7 + 13

        rng = np.random.default_rng(trial_seed)
        idx = rng.choice(n_total, size=n_sub, replace=False)
        sub_smiles = [all_smiles[i] for i in idx]
        sub_y = all_y[idx]

        # Molecule-disjoint val split within the trial pool
        n_held = max(1, int(round(n_sub * HOLDOUT_FRAC)))
        held_local = set(rng.choice(n_sub, size=n_held, replace=False).tolist())
        train_local_idx = [i for i in range(n_sub) if i not in held_local]
        train_sub_smiles = [sub_smiles[i] for i in train_local_idx]
        train_sub_y = sub_y[train_local_idx]

        train_df, val_df = build_mol_disjoint_pairs(sub_smiles, sub_y, held_local)

        model, scaler, vmae = train_filmdelta(train_df, val_df, emb_dict, emb_dim,
                                              seed=trial_seed)
        # Anchor pool = TRAIN MOLECULES ONLY (no leak)
        result = score_and_rank(model, scaler, train_sub_smiles, train_sub_y,
                                cand_smiles, emb_dict, emb_dim)

        all_rankings.append(result['ranking'])
        all_abs_preds.append(result['abs_preds'])
        all_win_counts.append(result['win_counts'])
        all_avg_deltas.append(result['avg_delta'])
        all_val_mae.append(vmae)

        top5 = result['ranking'][:5] + 1
        elapsed = time.time() - t1
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: top-5={list(top5)} "
              f"| val_MAE={vmae:.3f} "
              f"| anchors={len(train_sub_smiles)} held={n_held} "
              f"| mol1={np.where(result['ranking']==0)[0][0]+1:2d} "
              f"mol4={np.where(result['ranking']==3)[0][0]+1:2d} "
              f"mol18={np.where(result['ranking']==17)[0][0]+1:2d} "
              f"({elapsed:.0f}s)")

        del model, scaler
        gc.collect()

    rankings_mat = np.array(all_rankings)
    abs_preds_mat = np.array(all_abs_preds)
    win_counts_mat = np.array(all_win_counts)

    rank_of_mol = np.zeros((N_TRIALS, 19), dtype=int)
    for t in range(N_TRIALS):
        for pos, mi in enumerate(rankings_mat[t]):
            rank_of_mol[t, mi] = pos + 1

    print(f"\n{'='*70}")
    print("CLEAN STABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"\n{'Mol':>4s} {'Mean':>7s} {'Std':>6s} {'Median':>7s} {'Min':>5s} {'Max':>5s} "
          f"{'Top3%':>6s} {'Top5%':>6s} {'Wins':>6s} {'Pred':>7s}")
    print("-" * 80)
    mol_stats = []
    for mi in range(19):
        ranks = rank_of_mol[:, mi]; preds = abs_preds_mat[:, mi]; wins = win_counts_mat[:, mi]
        top3 = np.mean(ranks <= 3) * 100
        top5 = np.mean(ranks <= 5) * 100
        print(f"  {mi+1:2d}  {np.mean(ranks):6.1f}  {np.std(ranks):5.1f}  {np.median(ranks):6.0f}  "
              f"{ranks.min():4d}  {ranks.max():4d}  {top3:4.0f}%  {top5:4.0f}%  "
              f"{wins.mean():5.1f}  {preds.mean():5.3f}±{preds.std():.3f}")
        mol_stats.append({
            'mol_idx': mi+1, 'mean_rank': float(np.mean(ranks)),
            'std_rank': float(np.std(ranks)), 'median_rank': float(np.median(ranks)),
            'min_rank': int(ranks.min()), 'max_rank': int(ranks.max()),
            'top3_pct': float(top3), 'top5_pct': float(top5),
            'mean_wins': float(wins.mean()),
            'mean_pred': float(preds.mean()), 'std_pred': float(preds.std()),
        })

    mean_ranks = np.array([s['mean_rank'] for s in mol_stats])
    consensus = np.argsort(mean_ranks)
    print(f"\nCONSENSUS RANKING (clean, mol-disjoint, {N_TRIALS} trials):")
    for pos, mi in enumerate(consensus):
        s = mol_stats[mi]
        print(f"  {pos+1:2d}. Mol {s['mol_idx']:2d}: rank={s['mean_rank']:.1f}±{s['std_rank']:.1f}, "
              f"top3={s['top3_pct']:.0f}%, pIC50={s['mean_pred']:.3f}")

    from scipy.stats import spearmanr
    trial_corrs = []
    for i in range(N_TRIALS):
        for j in range(i+1, N_TRIALS):
            r, _ = spearmanr(rank_of_mol[i], rank_of_mol[j])
            trial_corrs.append(r)
    print(f"\nInter-trial Spearman: mean={np.mean(trial_corrs):.3f}, "
          f"min={np.min(trial_corrs):.3f}, std={np.std(trial_corrs):.3f}")
    print(f"Mean held-out val MAE: {np.mean(all_val_mae):.3f} ± {np.std(all_val_mae):.3f}")

    output = {
        'n_trials': N_TRIALS, 'subsample_frac': SUBSAMPLE_FRAC,
        'holdout_frac': HOLDOUT_FRAC, 'val_split': 'molecule_disjoint',
        'mol_stats': mol_stats,
        'consensus_ranking': [int(consensus[i] + 1) for i in range(19)],
        'inter_trial_spearman_mean': float(np.mean(trial_corrs)),
        'inter_trial_spearman_std': float(np.std(trial_corrs)),
        'inter_trial_spearman_min': float(np.min(trial_corrs)),
        'val_mae_per_trial': [float(v) for v in all_val_mae],
        'val_mae_mean': float(np.mean(all_val_mae)),
        'val_mae_std': float(np.std(all_val_mae)),
        'rank_distributions': {str(mi+1): rank_of_mol[:, mi].tolist() for mi in range(19)},
        'pred_distributions': {str(mi+1): abs_preds_mat[:, mi].tolist() for mi in range(19)},
    }

    out = RESULTS_DIR / "19_molecules_pairwise_stability_clean.json"
    json.dump(output, open(out, 'w'), indent=2)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min  →  {out}")


if __name__ == "__main__":
    main()
