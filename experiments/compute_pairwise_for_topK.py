#!/usr/bin/env python3
"""
Compute pairwise anchor metrics (using 3-seed FiLMDelta ensemble) for the top-K
candidates per method, then merge into all_methods_bulk_scored.csv.

Adds columns:
  - pIC50_mean       : ensemble mean over 3 seeds × 280 anchors
  - pIC50_std        : std across 3 seed-anchor-means (ensemble disagreement)
  - delta_vs_mol1    : pIC50_mean - Mol 1's pIC50_mean
  - direct_delta_from_mol1 : avg over 3 seeds of FiLMDelta(Mol1, candidate)
  - anchor_wins      : avg over 3 seeds of count(anchors where δ > 0); 0..280
  - anchor_wins_ge7  : avg over 3 seeds of count(high-potency anchors beaten)

For candidates not in any method's top-K, columns are NaN.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

RES = PROJECT_ROOT / "results" / "paper_evaluation"
INPUT_CSV = RES / "all_methods_bulk_scored.csv"
OUTPUT_CSV = RES / "all_methods_bulk_scored_v2.csv"
ENSEMBLE_DIR = RES / "reinvent4_film_ensemble"
MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
TOP_K_PER_METHOD = 1000


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ensemble():
    out = []
    for k in range(3):
        path = ENSEMBLE_DIR / f"film_seed{k}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        sc = StandardScaler()
        sc.mean_ = ckpt["scaler_mean"]
        sc.scale_ = ckpt["scaler_scale"]
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = len(sc.mean_)
        out.append({"model": m, "scaler": sc,
                    "anchor_embs": ckpt["anchor_embs"],
                    "anchor_pIC50": np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)})
    return out


def fp_array(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    log("=" * 70)
    log("PAIRWISE ANCHOR METRICS FOR TOP-K PER METHOD")
    log("=" * 70)

    log(f"Loading bulk CSV...")
    df = pd.read_csv(INPUT_CSV)
    log(f"  Loaded {len(df):,}")
    if "method" not in df.columns:
        log("ERROR: no 'method' column"); return

    # Pick top-K per method by pIC50_method (or pIC50_film fallback)
    pic_col = "pIC50_method" if "pIC50_method" in df.columns else "pIC50_film"
    if pic_col not in df.columns:
        log(f"ERROR: no pIC50 column"); return

    log(f"Selecting top-{TOP_K_PER_METHOD} per method by {pic_col}...")
    df = df.sort_values(pic_col, ascending=False, na_position="last").reset_index(drop=True)
    df["row_id"] = df.index  # row_id matches backend
    top_idx = []
    for m, sub in df.groupby("method"):
        top_idx.extend(sub.head(TOP_K_PER_METHOD).index.tolist())
    log(f"  {len(top_idx):,} candidates to score")

    # Init NaN columns for everyone
    for c in ["pIC50_mean", "pIC50_std", "delta_vs_mol1", "direct_delta_from_mol1",
              "anchor_wins", "anchor_wins_ge7"]:
        df[c] = np.nan

    # Load ensemble
    log(f"Loading 3-seed FiLMDelta ensemble...")
    ensemble = load_ensemble()
    n_anchors = ensemble[0]["anchor_embs"].shape[0]
    anchor_pIC50 = ensemble[0]["anchor_pIC50"]
    high_potency_mask = anchor_pIC50 >= 7.0

    # Mol 1 baseline (mean across 3 seeds × 280 anchors)
    log(f"Computing Mol 1 baseline...")
    mol1_fp = fp_array(MOL1_SMILES)
    mol1_seed_means = []
    for ens in ensemble:
        emb = torch.FloatTensor(ens["scaler"].transform(mol1_fp.reshape(1, -1)))
        target = emb.expand(n_anchors, -1)
        with torch.no_grad():
            deltas = ens["model"](ens["anchor_embs"], target).numpy().flatten()
        mol1_seed_means.append(float((ens["anchor_pIC50"] + deltas).mean()))
    mol1_baseline = float(np.mean(mol1_seed_means))
    log(f"  Mol 1 baseline pIC50: {mol1_baseline:.3f}")

    # Score top candidates in chunks (tiled tensor inference for speed)
    CHUNK = 64
    log(f"Scoring top-{len(top_idx):,} in chunks of {CHUNK}...")
    smiles_list = df.loc[top_idx, "smiles"].tolist()

    # Pre-compute FPs in parallel (simple loop here is fine, 12K mols)
    fps = []
    valid_idx_in_top = []
    for j, smi in enumerate(smiles_list):
        fp = fp_array(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx_in_top.append(j)
    log(f"  Valid FPs: {len(fps):,} / {len(smiles_list):,}")
    fps_arr = np.array(fps, dtype=np.float32)

    # Pre-scale per seed (same FPs for all seeds, but each seed has its own scaler)
    # Actually all 3 ensemble seeds share the same scaler in our training pipeline.
    # Verify: yes — same data, just different model seeds. So we can pre-scale once.
    scaled_fps = ensemble[0]["scaler"].transform(fps_arr).astype(np.float32)
    cand_embs = torch.from_numpy(scaled_fps)

    # Storage
    pIC50_mean_arr = np.full(len(smiles_list), np.nan)
    pIC50_std_arr = np.full(len(smiles_list), np.nan)
    direct_delta_arr = np.full(len(smiles_list), np.nan)
    wins_arr = np.full(len(smiles_list), np.nan)
    wins_ge7_arr = np.full(len(smiles_list), np.nan)
    delta_vs_arr = np.full(len(smiles_list), np.nan)

    # For Mol 1 direct delta, all 3 seeds need (anchor=Mol1, target=cand)
    mol1_emb_per_seed = []
    for ens in ensemble:
        mol1_emb_per_seed.append(torch.FloatTensor(ens["scaler"].transform(mol1_fp.reshape(1, -1))))

    n_chunks = (len(cand_embs) + CHUNK - 1) // CHUNK
    t0 = datetime.now()
    with torch.no_grad():
        for c in range(n_chunks):
            chunk_embs = cand_embs[c*CHUNK:(c+1)*CHUNK]
            k = chunk_embs.shape[0]

            # For each seed, tiled forward: anchors × candidates → (k, 280) deltas
            per_seed_deltas = []
            per_seed_anchor_means = []
            per_seed_direct = []
            for ens in ensemble:
                anchor_tiled = ens["anchor_embs"].unsqueeze(0).expand(k, -1, -1).reshape(k * n_anchors, -1)
                cand_tiled = chunk_embs.unsqueeze(1).expand(-1, n_anchors, -1).reshape(k * n_anchors, -1)
                deltas = ens["model"](anchor_tiled, cand_tiled).numpy().reshape(k, n_anchors)
                per_seed_deltas.append(deltas)
                per_seed_anchor_means.append((ens["anchor_pIC50"][None, :] + deltas).mean(axis=1))
                # Mol 1 direct delta: (1, 2048) Mol1 anchor expanded to k candidates
                m1_anchor = mol1_emb_per_seed[ensemble.index(ens)].expand(k, -1)
                d_direct = ens["model"](m1_anchor, chunk_embs).numpy().flatten()
                per_seed_direct.append(d_direct)

            seed_means = np.array(per_seed_anchor_means)  # (3, k)
            all_deltas = np.array(per_seed_deltas)        # (3, k, 280)
            seed_directs = np.array(per_seed_direct)      # (3, k)

            mean = seed_means.mean(axis=0)                 # (k,)
            std = seed_means.std(axis=0)
            wins = (all_deltas > 0).sum(axis=2).mean(axis=0)            # (k,)
            wins_ge7 = ((all_deltas > 0) & high_potency_mask[None, None, :]).sum(axis=2).mean(axis=0)
            direct = seed_directs.mean(axis=0)
            delta_vs = mean - mol1_baseline

            for j in range(k):
                idx_global = c * CHUNK + j
                pIC50_mean_arr[idx_global] = mean[j]
                pIC50_std_arr[idx_global] = std[j]
                direct_delta_arr[idx_global] = direct[j]
                wins_arr[idx_global] = wins[j]
                wins_ge7_arr[idx_global] = wins_ge7[j]
                delta_vs_arr[idx_global] = delta_vs[j]

            if (c + 1) % 20 == 0:
                done = (c + 1) * CHUNK
                el = (datetime.now() - t0).total_seconds()
                rate = done / max(1, el)
                log(f"  chunk {c+1}/{n_chunks} ({done:,}/{len(smiles_list):,}, {rate:.0f} mol/s)")

    # Map back to df rows
    log("Writing results...")
    for j, top_j in enumerate(top_idx):
        df.at[top_j, "pIC50_mean"] = pIC50_mean_arr[j]
        df.at[top_j, "pIC50_std"] = pIC50_std_arr[j]
        df.at[top_j, "direct_delta_from_mol1"] = direct_delta_arr[j]
        df.at[top_j, "anchor_wins"] = wins_arr[j]
        df.at[top_j, "anchor_wins_ge7"] = wins_ge7_arr[j]
        df.at[top_j, "delta_vs_mol1"] = delta_vs_arr[j]

    df.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved → {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1024 / 1024:.0f} MB)")
    n_scored = df["pIC50_mean"].notna().sum()
    log(f"  Pairwise filled: {n_scored:,} / {len(df):,}")


if __name__ == "__main__":
    main()
