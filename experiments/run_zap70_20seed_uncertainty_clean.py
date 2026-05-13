#!/usr/bin/env python3
"""CLEAN variant of run_zap70_20seed_uncertainty.py — molecule-disjoint val split.

Trains FiLMDelta 20 times (different seeds) with SAME molecule-disjoint val split
(the 28 held-out molecules from `reinvent4_film_model_clean.pt`). Per-seed
prediction uses the 252 train molecules as anchors only (held-out are NEVER used
as anchors — that would be the original leak).

Output: results/paper_evaluation/19_molecules_20seed_uncertainty_clean.json
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
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"
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
    if mol is None: return smi_clean
    return Chem.MolToSmiles(SaltRemover().StripMol(mol))


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"CLEAN 20-SEED UNCERTAINTY (mol-disjoint val from clean ckpt)")
    print("=" * 70)

    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n = len(all_smiles)
    print(f"  {n} ZAP70 molecules")

    cand_smiles = [clean_smiles(s) for s in SMILES_19]

    # Load the FROZEN held-out indices from reinvent4_film_model_clean.pt — these
    # are the 28 mols whose pairs go to val. Reusing the exact same split makes
    # this variant a direct comparator to the cached clean checkpoint.
    if not CLEAN_CKPT.exists():
        raise FileNotFoundError(f"Need {CLEAN_CKPT} for held-out indices")
    ck = torch.load(CLEAN_CKPT, map_location='cpu', weights_only=False)
    held = set(ck['heldout_idx'])
    print(f"  Loaded {len(held)} held-out mol indices from clean ckpt")
    print(f"  held: {sorted(held)[:8]}...")

    train_idx = [i for i in range(n) if i not in held]
    train_smiles = [all_smiles[i] for i in train_idx]
    train_y = all_y[train_idx]
    print(f"  train mols: {len(train_smiles)}  val mols: {len(held)}")

    # Build mol-disjoint pair sets ONCE — same across seeds (only seed changes shuffle)
    pairs_train_a, pairs_train_b, pairs_train_d = [], [], []
    pairs_val_a, pairs_val_b, pairs_val_d = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r = (all_smiles[i], all_smiles[j], float(all_y[j] - all_y[i]))
            if (i in held) or (j in held):
                pairs_val_a.append(r[0]); pairs_val_b.append(r[1]); pairs_val_d.append(r[2])
            else:
                pairs_train_a.append(r[0]); pairs_train_b.append(r[1]); pairs_train_d.append(r[2])
    n_train_pairs, n_val_pairs = len(pairs_train_a), len(pairs_val_a)
    print(f"  train pairs: {n_train_pairs:,}  val pairs: {n_val_pairs:,}")

    all_smi_set = list(set(all_smiles + cand_smiles))
    X = compute_fingerprints(all_smi_set, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {s: X[i] for i, s in enumerate(all_smi_set)}

    def get_emb(s):
        return emb_dict.get(s, np.zeros(emb_dim, dtype=np.float32))

    train_a_raw = np.array([get_emb(s) for s in pairs_train_a])
    train_b_raw = np.array([get_emb(s) for s in pairs_train_b])
    train_y_raw = np.array(pairs_train_d, dtype=np.float32)
    val_a_raw = np.array([get_emb(s) for s in pairs_val_a])
    val_b_raw = np.array([get_emb(s) for s in pairs_val_b])
    val_y_raw = np.array(pairs_val_d, dtype=np.float32)
    del pairs_train_a, pairs_train_b, pairs_train_d
    del pairs_val_a, pairs_val_b, pairs_val_d
    gc.collect()

    # Fit scaler on TRAIN only (no leak)
    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a_raw, train_b_raw]))
    train_a = scaler.transform(train_a_raw).astype(np.float32)
    train_b = scaler.transform(train_b_raw).astype(np.float32)
    val_a = scaler.transform(val_a_raw).astype(np.float32)
    val_b = scaler.transform(val_b_raw).astype(np.float32)
    train_a_t = torch.FloatTensor(train_a); train_b_t = torch.FloatTensor(train_b)
    train_y_t = torch.FloatTensor(train_y_raw)
    val_a_t = torch.FloatTensor(val_a); val_b_t = torch.FloatTensor(val_b)
    val_y_t = torch.FloatTensor(val_y_raw)
    del train_a_raw, train_b_raw, val_a_raw, val_b_raw, train_a, train_b, val_a, val_b
    gc.collect()

    all_preds = np.zeros((N_SEEDS, 19))
    all_rankings = np.zeros((N_SEEDS, 19), dtype=int)
    all_val_mae = np.zeros(N_SEEDS)

    train_anchor_embs_t = torch.FloatTensor(
        scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32))
    cand_embs_t = torch.FloatTensor(
        scaler.transform(np.array([get_emb(s) for s in cand_smiles])).astype(np.float32))
    n_anchors = len(train_smiles)

    for seed in range(N_SEEDS):
        st = time.time()
        np.random.seed(seed * 17 + 5); torch.manual_seed(seed * 17 + 5)

        tr_ds = torch.utils.data.TensorDataset(train_a_t, train_b_t, train_y_t)
        vl_ds = torch.utils.data.TensorDataset(val_a_t, val_b_t, val_y_t)
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

        if best_st: model.load_state_dict(best_st)
        model.cpu().eval()
        all_val_mae[seed] = best

        with torch.no_grad():
            for j in range(19):
                a = train_anchor_embs_t
                b = cand_embs_t[j:j+1].expand(n_anchors, -1)
                deltas = model(a, b).numpy().flatten()
                all_preds[seed, j] = np.median(train_y + deltas)
        ranking = np.argsort(-all_preds[seed])
        all_rankings[seed] = ranking + 1

        elapsed = time.time() - st
        top5 = ranking[:5] + 1
        print(f"  Seed {seed+1:2d}/{N_SEEDS}: top-5={list(top5)}, val_MAE={best:.3f} "
              f"| mol1={all_preds[seed,0]:.3f} mol4={all_preds[seed,3]:.3f} "
              f"mol18={all_preds[seed,17]:.3f} ({elapsed:.0f}s)")
        del model
        gc.collect()

    print(f"\n{'='*70}")
    print("CLEAN PREDICTION UNCERTAINTY (20 seeds, mol-disjoint)")
    print(f"{'='*70}")
    print(f"\n{'Mol':>4s} {'Mean':>7s} {'Std':>6s} {'95% CI':>17s} {'Range':>17s} {'MeanRank':>9s}")
    print("-" * 80)
    mol_results = []
    for j in range(19):
        preds = all_preds[:, j]
        mean = np.mean(preds); std = np.std(preds)
        ci_lo = np.percentile(preds, 2.5); ci_hi = np.percentile(preds, 97.5)
        ranks = np.zeros(N_SEEDS)
        for s in range(N_SEEDS):
            ranks[s] = np.where(np.argsort(-all_preds[s]) == j)[0][0] + 1
        mr = np.mean(ranks)
        print(f"  {j+1:2d}  {mean:6.3f}  {std:5.3f}  [{ci_lo:6.3f},{ci_hi:6.3f}]  "
              f"[{preds.min():.3f},{preds.max():.3f}]  {mr:8.1f}")
        mol_results.append({
            'mol_idx': j+1, 'mean_pIC50': float(mean), 'std_pIC50': float(std),
            'ci_2.5': float(ci_lo), 'ci_97.5': float(ci_hi),
            'min_pIC50': float(preds.min()), 'max_pIC50': float(preds.max()),
            'mean_rank': float(mr),
            'all_preds': preds.tolist(),
        })

    print(f"\nMean held-out val MAE: {all_val_mae.mean():.3f} ± {all_val_mae.std():.3f}")

    output = {
        'n_seeds': N_SEEDS, 'val_split': 'molecule_disjoint',
        'n_train_mols': len(train_smiles), 'n_val_mols': len(held),
        'heldout_idx': sorted(list(held)),
        'val_mae_mean': float(all_val_mae.mean()),
        'val_mae_std': float(all_val_mae.std()),
        'mol_results': mol_results,
    }
    out = RESULTS_DIR / "19_molecules_20seed_uncertainty_clean.json"
    json.dump(output, open(out, 'w'), indent=2)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min  →  {out}")


if __name__ == "__main__":
    main()
