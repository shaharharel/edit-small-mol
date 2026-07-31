#!/usr/bin/env python3
"""
Backfill pIC50_mean / pIC50_std / anchor_wins / anchor_wins_ge7 for the rows in

  results/paper_evaluation/all_methods_bulk_scored_v4.csv
  results/paper_evaluation/tier4_overnight_bulk.csv

that are missing them. Uses the canonical 3-seed FiLMDelta ensemble (anchor-mean
reconstruction over 280 ZAP70 training anchors), GPU-accelerated.

Inputs:
  - results/paper_evaluation/reinvent4_film_ensemble/film_seed{0,1,2}.pt
  - --smiles-list path to a newline-separated SMILES file (unique mols to score)

Output:
  - results/paper_evaluation/pIC50_mean_backfill.csv with columns
        smiles, pIC50_mean, pIC50_std, anchor_wins, anchor_wins_ge7

The backend can pick this up at startup and merge by SMILES.

Designed to run on a single V100 (16 GB). FP compute on CPU (RDKit), model
inference batched on GPU with tiled (B, 280) anchor expansion.
"""

import argparse
import os
import sys
import time
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
RDLogger.DisableLog("rdApp.*")

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

RES = PROJECT_ROOT / "results" / "paper_evaluation"
ENSEMBLE_DIR = RES / "reinvent4_film_ensemble"
DEFAULT_OUTPUT = RES / "pIC50_mean_backfill.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ensemble(device):
    """Load 3 FiLMDelta seed checkpoints. Returns list of dicts with model+scaler+anchors."""
    out = []
    for k in range(3):
        path = ENSEMBLE_DIR / f"film_seed{k}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        m.to(device)
        sc = StandardScaler()
        sc.mean_ = ckpt["scaler_mean"]
        sc.scale_ = ckpt["scaler_scale"]
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = len(sc.mean_)
        anchor_embs = ckpt["anchor_embs"].to(device)
        anchor_pIC50 = np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)
        out.append({
            "model": m, "scaler": sc,
            "anchor_embs": anchor_embs,
            "anchor_pIC50": anchor_pIC50,
        })
    return out


def fp_array(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fps_parallel(smiles_list, n_workers=8):
    """Compute Morgan FPs. Returns (fps_array, valid_mask)."""
    n = len(smiles_list)
    fps = np.zeros((n, 2048), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        arr = fp_array(smi)
        if arr is not None:
            fps[i] = arr
            valid[i] = True
        if (i + 1) % 50000 == 0:
            rate = (i + 1) / max(1, time.time() - t0)
            log(f"  FP {i+1:,}/{n:,} ({rate:.0f} mol/s)")
    return fps, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-list", required=True,
                    help="Path to newline-separated SMILES file (unique mols to score)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--chunk", type=int, default=512,
                    help="Number of candidate mols per GPU forward pass")
    ap.add_argument("--smoke", type=int, default=0,
                    help="If >0, score only first N SMILES (sanity test)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("pIC50_mean BACKFILL — 3-seed FiLMDelta ensemble")
    log("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Read SMILES list
    log(f"Reading SMILES from {args.smiles_list}")
    with open(args.smiles_list) as f:
        smiles_list = [ln.strip() for ln in f if ln.strip()]
    log(f"  {len(smiles_list):,} SMILES to score")
    if args.smoke > 0:
        smiles_list = smiles_list[: args.smoke]
        log(f"  SMOKE mode: scoring only {len(smiles_list):,}")

    # Compute FPs
    log("Computing Morgan FPs (radius=2, 2048 bits)...")
    fps, valid = compute_fps_parallel(smiles_list)
    n_valid = valid.sum()
    log(f"  Valid: {n_valid:,} / {len(smiles_list):,}")

    # Load ensemble
    log("Loading 3-seed FiLMDelta ensemble...")
    ensemble = load_ensemble(device)
    n_anchors = ensemble[0]["anchor_embs"].shape[0]
    high_potency_mask = ensemble[0]["anchor_pIC50"] >= 7.0
    high_potency_mask_t = torch.from_numpy(high_potency_mask.astype(np.float32)).to(device)
    log(f"  {n_anchors} anchors, {high_potency_mask.sum()} with pIC50 >= 7.0")

    # All 3 seeds share the same scaler (same data, different model init seeds)
    scaler = ensemble[0]["scaler"]
    # Verify
    for ens in ensemble[1:]:
        if not np.allclose(ens["scaler"].mean_, scaler.mean_):
            log("  [warn] scaler mean mismatch between seeds — using per-seed scalers")
            scaler = None
            break

    # Storage arrays (NaN for invalid mols)
    pIC50_mean_arr = np.full(len(smiles_list), np.nan, dtype=np.float64)
    pIC50_std_arr = np.full(len(smiles_list), np.nan, dtype=np.float64)
    wins_arr = np.full(len(smiles_list), np.nan, dtype=np.float64)
    wins_ge7_arr = np.full(len(smiles_list), np.nan, dtype=np.float64)

    # Pre-stack anchor pIC50 (constant across seeds is the same data → use seed 0)
    anchor_pIC50_t = torch.from_numpy(ensemble[0]["anchor_pIC50"].astype(np.float32)).to(device)

    # Pre-pack anchor embeddings per seed (on device already)
    seed_anchors = [ens["anchor_embs"] for ens in ensemble]
    seed_models = [ens["model"] for ens in ensemble]

    # Pre-scale all valid FPs once (single scaler)
    if scaler is not None:
        scaled_all = scaler.transform(fps[valid]).astype(np.float32)
        scaled_all_t = torch.from_numpy(scaled_all)
    else:
        scaled_all_t = None  # would need per-seed; not expected

    valid_idx = np.where(valid)[0]
    n = len(valid_idx)
    CHUNK = args.chunk
    log(f"Scoring {n:,} candidates in chunks of {CHUNK}...")
    t0 = time.time()
    last_log_t = t0

    with torch.no_grad():
        for c_start in range(0, n, CHUNK):
            c_end = min(c_start + CHUNK, n)
            k = c_end - c_start
            cand_embs = scaled_all_t[c_start:c_end].to(device)  # (k, 2048)

            # Per-seed: tile anchors and cands to (k * 280, 2048) then forward
            seed_means_list = []  # each (k,)
            seed_wins_list = []  # each (k,)
            seed_wins_ge7_list = []  # each (k,)
            for ens_i in range(3):
                anchors = seed_anchors[ens_i]  # (280, 2048)
                model = seed_models[ens_i]
                # tile: (k, 280, 2048)
                anchor_tiled = anchors.unsqueeze(0).expand(k, -1, -1).reshape(k * n_anchors, -1)
                cand_tiled = cand_embs.unsqueeze(1).expand(-1, n_anchors, -1).reshape(k * n_anchors, -1)
                deltas = model(anchor_tiled, cand_tiled).reshape(k, n_anchors)
                # absolute = anchor_pIC50 + delta
                abs_preds = anchor_pIC50_t.unsqueeze(0) + deltas  # (k, 280)
                seed_means_list.append(abs_preds.mean(dim=1))  # (k,)
                wins_per_anchor = (deltas > 0).float()  # (k, 280)
                seed_wins_list.append(wins_per_anchor.sum(dim=1))  # (k,)
                seed_wins_ge7_list.append((wins_per_anchor * high_potency_mask_t.unsqueeze(0)).sum(dim=1))

            seed_means = torch.stack(seed_means_list, dim=0)  # (3, k)
            seed_wins = torch.stack(seed_wins_list, dim=0)
            seed_wins_ge7 = torch.stack(seed_wins_ge7_list, dim=0)

            mean = seed_means.mean(dim=0).cpu().numpy()
            std = seed_means.std(dim=0, unbiased=False).cpu().numpy()
            wins = seed_wins.mean(dim=0).cpu().numpy()
            wins_ge7 = seed_wins_ge7.mean(dim=0).cpu().numpy()

            # Map back to global indices
            orig_idx = valid_idx[c_start:c_end]
            pIC50_mean_arr[orig_idx] = mean
            pIC50_std_arr[orig_idx] = std
            wins_arr[orig_idx] = wins
            wins_ge7_arr[orig_idx] = wins_ge7

            # Progress
            now = time.time()
            if now - last_log_t > 30:
                el = now - t0
                done = c_end
                rate = done / max(1, el)
                eta_s = (n - done) / max(1, rate)
                log(f"  {done:,}/{n:,} ({100*done/n:.1f}%), {rate:.0f} mol/s, "
                    f"ETA {eta_s/60:.1f} min, mean pIC50 so far: "
                    f"{np.nanmean(pIC50_mean_arr):.3f}")
                last_log_t = now

    log(f"Done scoring. Elapsed: {(time.time()-t0)/60:.1f} min")
    log(f"  pIC50_mean range: [{np.nanmin(pIC50_mean_arr):.3f}, "
        f"{np.nanmax(pIC50_mean_arr):.3f}], mean={np.nanmean(pIC50_mean_arr):.3f}")
    log(f"  pIC50_std range: [{np.nanmin(pIC50_std_arr):.3f}, "
        f"{np.nanmax(pIC50_std_arr):.3f}], mean={np.nanmean(pIC50_std_arr):.3f}")
    log(f"  anchor_wins range: [{np.nanmin(wins_arr):.1f}, "
        f"{np.nanmax(wins_arr):.1f}], mean={np.nanmean(wins_arr):.1f}")
    log(f"  Failed FPs (NaN): {np.isnan(pIC50_mean_arr).sum():,}")

    # Write output
    out_df = pd.DataFrame({
        "smiles": smiles_list,
        "pIC50_mean": pIC50_mean_arr,
        "pIC50_std": pIC50_std_arr,
        "anchor_wins": wins_arr,
        "anchor_wins_ge7": wins_ge7_arr,
    })
    out_df.to_csv(out_path, index=False)
    log(f"Wrote {len(out_df):,} rows → {out_path}")
    log(f"  Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
