#!/usr/bin/env python3
"""
FiLMDelta External Scorer for REINVENT4.

Reads SMILES from stdin, outputs JSON with anchor-based absolute pIC50 predictions.
Protocol: ExternalProcess (stdin newline-separated SMILES → stdout JSON).

The scorer trains FiLMDelta on kinase pairs (pretrain) + ZAP70 all-pairs (fine-tune),
then uses anchor-based prediction: pred(j) = mean_i(known_pIC50(i) + FiLMDelta(i→j))

Usage (standalone test):
    echo -e "c1ccccc1\nCCO" | conda run --no-capture-output -n quris python experiments/reinvent4_film_scorer.py

Usage with REINVENT4:
    Set params.executable and params.args in TOML config to invoke this script.
"""

import sys
import json
import os
import gc
import warnings
import logging

# Suppress all warnings and RDKit logs to keep stdout clean for REINVENT4
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
logging.disable(logging.CRITICAL)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# Save MPS, then import modules that disable it
_real_mps = torch.backends.mps.is_available
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
torch.backends.mps.is_available = _real_mps

PROJECT_ROOT = Path(__file__).parent.parent
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"

# Model cache path — train once, reuse across calls
MODEL_CACHE = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def train_film_model():
    """Train FiLMDelta: kinase pretrain → ZAP70 fine-tune. Returns (model, scaler, anchor_embs, anchor_pIC50)."""
    import pandas as pd

    # Load ZAP70 training data (returns DataFrames: 280 unique mols, 305 per-assay)
    smiles_df, _ = load_zap70_molecules()
    smiles_list = smiles_df['smiles'].tolist()
    pIC50_arr = smiles_df['pIC50'].values.astype(np.float64)
    print(f"[scorer] ZAP70: {len(smiles_list)} molecules", file=sys.stderr)

    # Compute fingerprints for training molecules
    fp_cache = {}
    fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    for i, s in enumerate(smiles_list):
        fp_cache[s] = fps[i]

    # Load kinase pairs for pretraining
    kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    all_kinase_smi = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
    extra = [s for s in all_kinase_smi if s not in fp_cache]
    if extra:
        efps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
        for i, s in enumerate(extra):
            fp_cache[s] = efps[i]

    mask = kinase_pairs["mol_a"].apply(lambda s: s in fp_cache) & \
           kinase_pairs["mol_b"].apply(lambda s: s in fp_cache)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    print(f"[scorer] Kinase pairs: {len(kinase_pairs):,}", file=sys.stderr)

    ea = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    eb = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    d = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([ea, eb]))
    Xa = torch.FloatTensor(scaler.transform(ea))
    Xb = torch.FloatTensor(scaler.transform(eb))
    yd = torch.FloatTensor(d)
    del ea, eb, d, kinase_pairs; gc.collect()

    # Pretrain on kinase pairs
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    n_val = len(Xa) // 10
    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(100):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), 256):
            bi = perm[s:s+256]
            opt.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
        if vl < best_vl:
            best_vl, best_st, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15:
                print(f"[scorer] Pretrain early stop ep {ep+1}, val={best_vl:.4f}", file=sys.stderr)
                break
    model.load_state_dict(best_st)
    del Xa, Xb, yd; gc.collect()

    # Fine-tune on ZAP70 all-pairs
    n = len(smiles_list)
    pairs_a, pairs_b, pairs_d = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs_a.append(fp_cache[smiles_list[i]])
                pairs_b.append(fp_cache[smiles_list[j]])
                pairs_d.append(float(pIC50_arr[j] - pIC50_arr[i]))

    print(f"[scorer] Fine-tuning on {len(pairs_a):,} ZAP70 all-pairs", file=sys.stderr)
    Xa = torch.FloatTensor(scaler.transform(np.array(pairs_a)))
    Xb = torch.FloatTensor(scaler.transform(np.array(pairs_b)))
    yd = torch.FloatTensor(np.array(pairs_d, dtype=np.float32))
    del pairs_a, pairs_b, pairs_d; gc.collect()

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_val2 = len(Xa) // 10
    best_vl2, best_st2, w2 = float("inf"), None, 0
    for ep in range(50):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val2) + n_val2
        for s in range(0, len(perm), 256):
            bi = perm[s:s+256]
            opt2.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt2.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val2], Xb[:n_val2]), yd[:n_val2]).item()
        if vl < best_vl2:
            best_vl2, best_st2, w2 = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            w2 += 1
            if w2 >= 15:
                break
    if best_st2:
        model.load_state_dict(best_st2)
    model.eval()
    del Xa, Xb, yd; gc.collect()

    # Anchor embeddings
    anchor_fps = np.array([fp_cache[s] for s in smiles_list])
    anchor_embs = torch.FloatTensor(scaler.transform(anchor_fps))
    anchor_pIC50 = pIC50_arr.copy()

    # Save to cache
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "anchor_embs": anchor_embs,
        "anchor_pIC50": anchor_pIC50,
    }, MODEL_CACHE)
    print(f"[scorer] Model saved to {MODEL_CACHE}", file=sys.stderr)

    return model, scaler, anchor_embs, anchor_pIC50


def load_film_model():
    """Load cached model or train from scratch."""
    if MODEL_CACHE.exists():
        print(f"[scorer] Loading cached model from {MODEL_CACHE}", file=sys.stderr)
        ckpt = torch.load(MODEL_CACHE, map_location="cpu", weights_only=False)
        model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)

        anchor_embs = ckpt["anchor_embs"]
        anchor_pIC50 = ckpt["anchor_pIC50"]
        print(f"[scorer] Model loaded: {len(anchor_pIC50)} anchors", file=sys.stderr)
        return model, scaler, anchor_embs, anchor_pIC50
    else:
        print("[scorer] No cached model, training from scratch...", file=sys.stderr)
        return train_film_model()


def score_smiles(smiles_list, model, scaler, anchor_embs, anchor_pIC50):
    """Score SMILES using anchor-based FiLMDelta prediction.
    Returns list of floats (pIC50 predictions). NaN for invalid SMILES.
    """
    from rdkit.Chem import AllChem

    scores = []
    valid_indices = []
    valid_fps = []

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scores.append(float('nan'))
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            valid_fps.append(arr)
            valid_indices.append(i)
            scores.append(0.0)  # placeholder
        except Exception:
            scores.append(float('nan'))

    if not valid_fps:
        return scores

    # Batch score valid molecules
    fps_arr = np.array(valid_fps)
    batch_embs = torch.FloatTensor(scaler.transform(fps_arr))

    n_anchors = len(anchor_pIC50)
    with torch.no_grad():
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            target_emb = batch_embs[idx_in_valid:idx_in_valid+1].expand(n_anchors, -1)
            deltas = model(anchor_embs, target_emb).numpy().flatten()
            abs_preds = anchor_pIC50 + deltas
            scores[orig_idx] = float(np.mean(abs_preds))

    return scores


from rdkit.Chem import DataStructs

def main():
    """Main entry point: read SMILES from stdin, output JSON scores to stdout."""
    # Load or train model
    model, scaler, anchor_embs, anchor_pIC50 = load_film_model()

    # Read SMILES from stdin
    smiles_list = [line.strip() for line in sys.stdin if line.strip()]

    if not smiles_list:
        output = {"version": 1, "payload": {"pIC50": []}}
        print(json.dumps(output))
        return

    print(f"[scorer] Scoring {len(smiles_list)} molecules...", file=sys.stderr)

    # Score
    scores = score_smiles(smiles_list, model, scaler, anchor_embs, anchor_pIC50)

    # REINVENT4 transform expects scores where higher = better for sigmoid/reverse_sigmoid
    # pIC50 is already higher = more potent, so we output directly
    output = {
        "version": 1,
        "payload": {
            "pIC50": scores
        }
    }

    print(json.dumps(output))
    print(f"[scorer] Done. Mean pIC50: {np.nanmean(scores):.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
