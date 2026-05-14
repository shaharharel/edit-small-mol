#!/usr/bin/env python3
"""20-seed uncertainty + KP + SCAFFOLD-DISJOINT val split (harder than mol-disjoint).

Mol-disjoint = different molecule indices, but molecule pairs can share scaffolds.
Scaffold-disjoint = no shared Bemis-Murcko scaffold between train and val mols.
This is the real generalization test: can FiLMDelta predict pIC50 for a NEW
chemotype it never saw?

Hypothesis: if Mol-1 / Mol-15 are highly similar to high-pIC50 ChEMBL mols by
SCAFFOLD, they'll lose their inflated rank under scaffold-disjoint val.

Output: results/paper_evaluation/19_molecules_20seed_uncertainty_clean_kp_scaffold.json
"""
import sys, gc, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.SaltRemover import SaltRemover

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 10
N_SEEDS = 20
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
N_KINASE_PRETRAIN = 100_000
PRETRAIN_EPOCHS = 30
PRETRAIN_PATIENCE = 5
PRETRAIN_LR = 2e-4
FINETUNE_LR = 1e-4

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


def bm_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return ""
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf) if scaf is not None else ""


def kinase_pretrain(emb_dict, scaler):
    kp = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    if len(kp) > N_KINASE_PRETRAIN:
        kp = kp.sample(n=N_KINASE_PRETRAIN, random_state=42).reset_index(drop=True)
    need = [s for s in set(kp.mol_a.tolist() + kp.mol_b.tolist()) if s not in emb_dict]
    if need:
        fps = compute_fingerprints(need, "morgan", radius=2, n_bits=2048)
        for s, f in zip(need, fps): emb_dict[s] = f
    Xa = np.array([emb_dict[s] for s in kp.mol_a])
    Xb = np.array([emb_dict[s] for s in kp.mol_b])
    Xa = scaler.transform(Xa).astype(np.float32)
    Xb = scaler.transform(Xb).astype(np.float32)
    yd = kp.delta.values.astype(np.float32)
    n_val = len(Xa) // 10
    Xa_t = torch.FloatTensor(Xa); Xb_t = torch.FloatTensor(Xb); yd_t = torch.FloatTensor(yd)
    device = torch.device(DEVICE)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    crit = nn.L1Loss()
    best, best_st, w = float('inf'), None, 0
    for ep in range(PRETRAIN_EPOCHS):
        m.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), BATCH_SIZE):
            bi = perm[s:s+BATCH_SIZE]; opt.zero_grad()
            crit(m(Xa_t[bi].to(device), Xb_t[bi].to(device)), yd_t[bi].to(device)).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xa_t[:n_val].to(device), Xb_t[:n_val].to(device)), yd_t[:n_val].to(device)).item()
        if vl < best:
            best, w = vl, 0
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            w += 1
            if w >= PRETRAIN_PATIENCE: break
    print(f"  [pretrain] kinase MAE: {best:.4f}")
    return best_st


def main():
    t0 = time.time()
    print("=" * 80)
    print(f"CLEAN + KP + SCAFFOLD-DISJOINT 20-SEED UNCERTAINTY")
    print("=" * 80)
    mol_data, _ = load_zap70_molecules()
    all_smiles = mol_data['smiles'].tolist()
    all_y = mol_data['pIC50'].values
    n = len(all_smiles)

    # Compute Bemis-Murcko scaffolds for all 280
    print("Computing scaffolds for 280 ZAP70 mols...")
    scaffolds = [bm_scaffold(s) for s in all_smiles]
    scaf_counts = pd.Series(scaffolds).value_counts()
    print(f"  Unique scaffolds: {len(scaf_counts)}")
    print(f"  Top scaffold appears in: {scaf_counts.iloc[0]} mols ({100*scaf_counts.iloc[0]/n:.1f}%)")

    # Pick scaffolds for val such that ~15% of mols are held out — sample
    # whole scaffold clusters, not individual mols
    rng = np.random.default_rng(42)
    target_n_val = int(0.15 * n)  # ~42 mols
    scaf_order = list(scaf_counts.index)
    rng.shuffle(scaf_order)
    held = set()
    held_scaf = set()
    for s in scaf_order:
        held.update([i for i, sc in enumerate(scaffolds) if sc == s])
        held_scaf.add(s)
        if len(held) >= target_n_val: break
    print(f"  Held-out: {len(held)} mols across {len(held_scaf)} scaffold clusters")
    train_idx = [i for i in range(n) if i not in held]
    train_smiles = [all_smiles[i] for i in train_idx]
    train_y = all_y[train_idx]

    cand_smiles = [clean_smiles(s) for s in SMILES_19]
    cand_scafs = [bm_scaffold(s) for s in cand_smiles]
    cand_held = [int(c in held_scaf) for c in cand_scafs]
    print(f"  19 candidates: {sum(cand_held)} share a scaffold with the held-out training cluster")

    # Build mol-disjoint (scaffold-stratified) pairs
    pairs_train_a, pairs_train_b, pairs_train_d = [], [], []
    pairs_val_a, pairs_val_b, pairs_val_d = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = float(all_y[j] - all_y[i])
            if (i in held) or (j in held):
                pairs_val_a.append(all_smiles[i]); pairs_val_b.append(all_smiles[j]); pairs_val_d.append(d)
            else:
                pairs_train_a.append(all_smiles[i]); pairs_train_b.append(all_smiles[j]); pairs_train_d.append(d)
    print(f"  train pairs: {len(pairs_train_a):,}  val pairs: {len(pairs_val_a):,}")

    all_smi_set = list(set(all_smiles + cand_smiles))
    X = compute_fingerprints(all_smi_set, "morgan", radius=2, n_bits=2048)
    emb_dim = 2048
    emb_dict = {s: X[i] for i, s in enumerate(all_smi_set)}

    scaler = StandardScaler()
    scaler.fit(np.array([emb_dict[s] for s in train_smiles]))

    print("\nPretraining on kinase MMPs…")
    pretrain_state = kinase_pretrain(emb_dict, scaler)

    def get_emb(s):
        return emb_dict.get(s, np.zeros(emb_dim, dtype=np.float32))

    train_a = scaler.transform(np.array([get_emb(s) for s in pairs_train_a])).astype(np.float32)
    train_b = scaler.transform(np.array([get_emb(s) for s in pairs_train_b])).astype(np.float32)
    val_a   = scaler.transform(np.array([get_emb(s) for s in pairs_val_a])).astype(np.float32)
    val_b   = scaler.transform(np.array([get_emb(s) for s in pairs_val_b])).astype(np.float32)
    train_a_t = torch.FloatTensor(train_a); train_b_t = torch.FloatTensor(train_b)
    train_y_t = torch.FloatTensor(np.array(pairs_train_d, dtype=np.float32))
    val_a_t = torch.FloatTensor(val_a); val_b_t = torch.FloatTensor(val_b)
    val_y_t = torch.FloatTensor(np.array(pairs_val_d, dtype=np.float32))
    del train_a, train_b, val_a, val_b
    gc.collect()

    all_preds = np.zeros((N_SEEDS, 19))
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
        m = FiLMDeltaMLP(input_dim=emb_dim, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
        m.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()})
        opt = torch.optim.Adam(m.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
        crit = nn.MSELoss()
        best, pat, best_st = float('inf'), 0, None
        for epoch in range(MAX_EPOCHS):
            m.train()
            for batch in tr_ld:
                a, b, y = [t.to(device) for t in batch]
                opt.zero_grad(); crit(m(a, b), y).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval()
            vl, nv = 0.0, 0
            with torch.no_grad():
                for batch in vl_ld:
                    a, b, y = [t.to(device) for t in batch]
                    vl += crit(m(a, b), y).item(); nv += 1
            vl /= max(nv, 1)
            if vl < best:
                best, pat = vl, 0
                best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE: break
        if best_st: m.load_state_dict(best_st)
        m.cpu().eval()
        all_val_mae[seed] = best
        with torch.no_grad():
            for j in range(19):
                a = train_anchor_embs_t
                b = cand_embs_t[j:j+1].expand(n_anchors, -1)
                deltas = m(a, b).numpy().flatten()
                all_preds[seed, j] = np.median(train_y + deltas)
        ranking = np.argsort(-all_preds[seed])
        elapsed = time.time() - st
        print(f"  Seed {seed+1:2d}/{N_SEEDS}: top-5={[int(i+1) for i in ranking[:5]]} val_MSE={best:.3f} "
              f"| mol1={all_preds[seed,0]:.3f} mol15={all_preds[seed,14]:.3f} ({elapsed:.0f}s)")
        del m; gc.collect()

    print(f"\nMean val MSE: {all_val_mae.mean():.3f} ± {all_val_mae.std():.3f}")
    mol_results = []
    for j in range(19):
        preds = all_preds[:, j]
        ranks = np.zeros(N_SEEDS)
        for s in range(N_SEEDS):
            ranks[s] = np.where(np.argsort(-all_preds[s]) == j)[0][0] + 1
        mol_results.append({
            'mol_idx': j+1, 'mean_pIC50': float(np.mean(preds)),
            'std_pIC50': float(np.std(preds)),
            'mean_rank': float(np.mean(ranks)),
            'cand_in_heldout_scaffold': cand_held[j],
            'cand_scaffold': cand_scafs[j],
        })
    for r in sorted(mol_results, key=lambda r: -r['mean_pIC50'])[:10]:
        flag = " ⚠SCAF_OOD" if r['cand_in_heldout_scaffold'] else ""
        print(f"  Mol {r['mol_idx']:2d}: pIC50={r['mean_pIC50']:.3f}±{r['std_pIC50']:.3f}  rank={r['mean_rank']:.1f}{flag}")
    output = {
        'n_seeds': N_SEEDS, 'val_split': 'scaffold_disjoint',
        'n_train_mols': len(train_smiles), 'n_val_mols': len(held),
        'n_heldout_scaffolds': len(held_scaf),
        'val_mse_mean': float(all_val_mae.mean()),
        'mol_results': mol_results,
    }
    out_path = RESULTS_DIR / "19_molecules_20seed_uncertainty_clean_kp_scaffold.json"
    json.dump(output, open(out_path, 'w'), indent=2)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min  →  {out_path}")


if __name__ == "__main__":
    main()
