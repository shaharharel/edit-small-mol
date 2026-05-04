#!/usr/bin/env python3
"""
FiLMDelta scorer with uncertainty-aware reward (mean − λ·std).

Same I/O contract as `reinvent4_film_scorer.py` — reads SMILES from stdin,
outputs JSON to stdout. Difference: trains/loads N FiLMDelta seeds, returns
mean - LAMBDA * std as the reward signal so REINVENT4's RL prefers candidates
the model is confident about.

LAMBDA controls penalty weight; default 0.5 (one half-std penalty).

Usage in REINVENT4 TOML:
  [stage.scoring.component.ExternalProcess]
  [[stage.scoring.component.ExternalProcess.endpoint]]
  name = "FiLMDelta pIC50 (uncertainty-aware)"
  weight = 0.6
  params.executable = "/opt/miniconda3/condabin/conda"
  params.args = "run --no-capture-output -n quris python /abs/path/reinvent4_film_scorer_uncertainty.py"
  params.property = "pIC50"
  transform.type = "sigmoid"
  transform.high = 7.5
  transform.low = 5.5
  transform.k = 0.5

Standalone test:
  echo "C=CC(=O)N1CCN(c2ncnc(NC3(CC)CCNCC3)n2)CC1" |
    conda run -n quris python experiments/reinvent4_film_scorer_uncertainty.py
"""

import sys
import json
import os
import gc
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

# Save MPS, then import modules that disable it
_real_mps = torch.backends.mps.is_available
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
torch.backends.mps.is_available = _real_mps

PROJECT_ROOT = Path(__file__).parent.parent
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"

# Cached primary model + N additional seeded models
PRIMARY_CACHE = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
ENSEMBLE_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_ensemble"

LAMBDA = float(os.environ.get("FILM_UNCERTAINTY_LAMBDA", 0.5))
N_SEEDS = int(os.environ.get("FILM_UNCERTAINTY_N_SEEDS", 3))

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def train_one_film(seed: int):
    """Train one FiLMDelta seed: kinase pretrain → ZAP70 fine-tune.
    Returns (model_state, scaler, anchor_embs, anchor_pIC50).
    """
    import pandas as pd
    np.random.seed(seed)
    torch.manual_seed(seed)

    smiles_df, _ = load_zap70_molecules()
    smiles_list = smiles_df['smiles'].tolist()
    pIC50_arr = smiles_df['pIC50'].values.astype(np.float64)

    fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    fp_cache = dict(zip(smiles_list, fps))

    kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    all_kinase = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
    extra = [s for s in all_kinase if s not in fp_cache]
    if extra:
        efps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
        for i, s in enumerate(extra):
            fp_cache[s] = efps[i]

    mask = kinase_pairs["mol_a"].apply(lambda s: s in fp_cache) & \
           kinase_pairs["mol_b"].apply(lambda s: s in fp_cache)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    ea = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    eb = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    d = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([ea, eb]))
    Xa = torch.FloatTensor(scaler.transform(ea))
    Xb = torch.FloatTensor(scaler.transform(eb))
    yd = torch.FloatTensor(d)
    del ea, eb, d, kinase_pairs; gc.collect()

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
                break
    model.load_state_dict(best_st)
    del Xa, Xb, yd; gc.collect()

    n = len(smiles_list)
    pairs_a, pairs_b, pairs_d = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs_a.append(fp_cache[smiles_list[i]])
                pairs_b.append(fp_cache[smiles_list[j]])
                pairs_d.append(float(pIC50_arr[j] - pIC50_arr[i]))
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

    anchor_fps = np.array([fp_cache[s] for s in smiles_list])
    anchor_embs = torch.FloatTensor(scaler.transform(anchor_fps))
    return {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "anchor_embs": anchor_embs,
        "anchor_pIC50": pIC50_arr.copy(),
    }


def get_or_train_ensemble():
    """Load ensemble of N seeded FiLMDelta models. Train if not cached."""
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    scalers = []
    anchor_embs = None
    anchor_pIC50 = None

    for seed_idx in range(N_SEEDS):
        path = ENSEMBLE_DIR / f"film_seed{seed_idx}.pt"
        if not path.exists():
            print(f"[uncertainty_scorer] Training seed {seed_idx} (one-time)...", file=sys.stderr)
            ck = train_one_film(seed=42 + seed_idx)
            torch.save(ck, path)
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
        sc = StandardScaler()
        sc.mean_ = ck["scaler_mean"]
        sc.scale_ = ck["scaler_scale"]
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = len(sc.mean_)
        scalers.append(sc)
        if anchor_embs is None:
            anchor_embs = ck["anchor_embs"]
            anchor_pIC50 = np.asarray(ck["anchor_pIC50"]).astype(np.float64)
    print(f"[uncertainty_scorer] Loaded {len(models)} seeds (λ={LAMBDA})", file=sys.stderr)
    return models, scalers, anchor_embs, anchor_pIC50


def score_ensemble(smiles_list, models, scalers, anchor_embs, anchor_pIC50):
    """Return list of (mean - λ*std) per SMILES. NaN if invalid."""
    n_anchors = anchor_embs.shape[0]
    scores = np.full(len(smiles_list), np.nan)

    valid_idx, fps = [], []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr); valid_idx.append(i)
    if not fps:
        return scores

    fps_arr = np.array(fps, dtype=np.float32)
    seed_preds = np.zeros((N_SEEDS, len(valid_idx)), dtype=np.float64)
    for k, (m, sc) in enumerate(zip(models, scalers)):
        embs = torch.FloatTensor(sc.transform(fps_arr))
        with torch.no_grad():
            for j, orig in enumerate(valid_idx):
                tgt = embs[j:j+1].expand(n_anchors, -1)
                deltas = m(anchor_embs, tgt).numpy().flatten()
                seed_preds[k, j] = float(np.mean(anchor_pIC50 + deltas))

    means = seed_preds.mean(axis=0)
    stds = seed_preds.std(axis=0)
    penalised = means - LAMBDA * stds

    for j, orig in enumerate(valid_idx):
        scores[orig] = float(penalised[j])
    return scores, means, stds


def main():
    models, scalers, anchor_embs, anchor_pIC50 = get_or_train_ensemble()

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        print(json.dumps({"version": 1, "payload": {"pIC50": []}}))
        return

    scores, means, stds = score_ensemble(smiles_list, models, scalers, anchor_embs, anchor_pIC50)
    print(json.dumps({"version": 1, "payload": {"pIC50": scores.tolist()}}))
    valid = ~np.isnan(scores)
    if valid.any():
        print(f"[uncertainty_scorer] N={int(valid.sum())} mean={float(np.nanmean(means)):.3f} "
              f"std-mean={float(np.nanmean(stds)):.3f} reward-mean={float(np.nanmean(scores)):.3f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
