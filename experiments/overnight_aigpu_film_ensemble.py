#!/usr/bin/env python3
"""
Train a 3-seed FiLMDelta ensemble for the uncertainty-aware reward signal.

Designed to run on ai-gpu (Tesla T4): GPU-accelerated kinase pretrain + ZAP70
fine-tune. Each seed produces an independent checkpoint that the
uncertainty-aware scorer averages over to compute mean − λ·std reward.

Output: ~/edit-small-mol-rsync/results/paper_evaluation/reinvent4_film_ensemble/
        film_seed{0,1,2}.pt

After completion the orchestrator should rsync the ensemble checkpoints back
to local + ai-chem so they can be used for scoring.

Usage (on ai-gpu):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aigpu_film_ensemble.py
"""

import sys
import os
import gc
import warnings
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import StandardScaler
import pandas as pd
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
ENSEMBLE_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)


def compute_morgan_fps(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan FPs as numpy array."""
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


def load_zap70():
    """Load ZAP70 training molecules. Uses the same loader as run_zap70_v3 for
    consistency with the rest of the pipeline (returns 280 ZAP70 mols)."""
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    return smiles_df['smiles'].tolist(), smiles_df['pIC50'].values.astype(np.float64)


def train_one_film(seed: int):
    """Train one FiLMDelta seed: kinase pretrain → ZAP70 fine-tune. Saves to disk."""
    out_path = ENSEMBLE_DIR / f"film_seed{seed - 42}.pt"
    if out_path.exists():
        print(f"[+] seed{seed - 42}: cached at {out_path}")
        return
    print(f"\n=== Training seed={seed} (output: {out_path}) ===")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load training data
    smiles_list, pIC50_arr = load_zap70()

    # Compute FPs for ZAP70
    print(f"  Computing FPs for {len(smiles_list)} ZAP70 molecules...")
    fps_zap = compute_morgan_fps(smiles_list)
    fp_cache = dict(zip(smiles_list, fps_zap))

    # Load kinase pairs
    print(f"  Loading kinase pairs from {KINASE_PAIRS_FILE}")
    kp = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    print(f"  Loaded {len(kp):,} kinase pairs")

    # Compute FPs for unique kinase mols
    all_kinase = list(set(kp["mol_a"].tolist() + kp["mol_b"].tolist()))
    extra = [s for s in all_kinase if s not in fp_cache]
    if extra:
        print(f"  Computing FPs for {len(extra):,} additional kinase mols...")
        efps = compute_morgan_fps(extra)
        for i, s in enumerate(extra):
            fp_cache[s] = efps[i]

    mask = kp["mol_a"].apply(lambda s: s in fp_cache) & \
           kp["mol_b"].apply(lambda s: s in fp_cache)
    kp = kp[mask].reset_index(drop=True)

    # Build pretrain tensors
    ea = np.array([fp_cache[s] for s in kp["mol_a"]], dtype=np.float32)
    eb = np.array([fp_cache[s] for s in kp["mol_b"]], dtype=np.float32)
    d = kp["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([ea, eb]))
    Xa = torch.FloatTensor(scaler.transform(ea)).to(DEVICE)
    Xb = torch.FloatTensor(scaler.transform(eb)).to(DEVICE)
    yd = torch.FloatTensor(d).to(DEVICE)
    del ea, eb, d, kp; gc.collect()
    print(f"  Pretrain tensors: Xa={Xa.shape}, on {DEVICE}")

    # Pretrain
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    n_val = len(Xa) // 10
    best_vl, best_st, wait = float("inf"), None, 0

    bs = 1024 if torch.cuda.is_available() else 256
    print(f"  Pretrain (bs={bs}, max 100 epochs, patience 15)...")
    for ep in range(100):
        model.train()
        perm = (np.random.permutation(len(Xa) - n_val) + n_val).tolist()
        for s in range(0, len(perm), bs):
            bi = perm[s:s+bs]
            opt.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                print(f"    Early stop at ep {ep+1}, best val={best_vl:.4f}")
                break
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}: val={vl:.4f} (best {best_vl:.4f})")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_st.items()})
    del Xa, Xb, yd; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ZAP70 fine-tune (all-pairs)
    print(f"  Fine-tune on ZAP70 all-pairs...")
    n = len(smiles_list)
    pairs_a, pairs_b, pairs_d = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs_a.append(fp_cache[smiles_list[i]])
                pairs_b.append(fp_cache[smiles_list[j]])
                pairs_d.append(float(pIC50_arr[j] - pIC50_arr[i]))
    Xa = torch.FloatTensor(scaler.transform(np.array(pairs_a, dtype=np.float32))).to(DEVICE)
    Xb = torch.FloatTensor(scaler.transform(np.array(pairs_b, dtype=np.float32))).to(DEVICE)
    yd = torch.FloatTensor(np.array(pairs_d, dtype=np.float32)).to(DEVICE)
    del pairs_a, pairs_b, pairs_d; gc.collect()

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_val2 = len(Xa) // 10
    best_vl2, best_st2, w2 = float("inf"), None, 0
    print(f"  Fine-tune (bs={bs}, max 50 epochs, patience 15)...")
    for ep in range(50):
        model.train()
        perm = (np.random.permutation(len(Xa) - n_val2) + n_val2).tolist()
        for s in range(0, len(perm), bs):
            bi = perm[s:s+bs]
            opt2.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt2.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val2], Xb[:n_val2]), yd[:n_val2]).item()
        if vl < best_vl2:
            best_vl2 = vl
            best_st2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            w2 = 0
        else:
            w2 += 1
            if w2 >= 15:
                break
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}: val={vl:.4f} (best {best_vl2:.4f})")
    if best_st2:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_st2.items()})
    model.eval()
    del Xa, Xb, yd; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Anchor embeddings (on CPU for portability)
    anchor_fps = np.array([fp_cache[s] for s in smiles_list], dtype=np.float32)
    anchor_embs_cpu = torch.FloatTensor(scaler.transform(anchor_fps))

    # Save
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "anchor_embs": anchor_embs_cpu,
        "anchor_pIC50": pIC50_arr.copy(),
        "seed": seed,
        "trained_at": datetime.now().isoformat(),
    }, out_path)
    print(f"  Saved → {out_path}")
    print(f"  Done: {datetime.now().strftime('%H:%M:%S')}")


def main():
    print("=" * 70)
    print("3-SEED FILMDELTA ENSEMBLE TRAINING (ai-gpu T4)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    for seed in (42, 43, 44):
        train_one_film(seed=seed)

    print(f"\nAll seeds done. Output dir: {ENSEMBLE_DIR}")
    print(f"  Files: {list(ENSEMBLE_DIR.glob('*.pt'))}")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
