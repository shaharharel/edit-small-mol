#!/usr/bin/env python3
"""
ai-chem fast batched FiLMDelta scoring on coupled products.

Replaces the per-molecule Python loop with proper tiled-tensor batched inference:
process K candidates × 280 anchors in a single forward pass per chunk. ~50× faster
than the per-mol loop on CPU.

Reads coupled_products_smiles.csv (498K), runs descriptors (parallel) +
batched FiLMDelta scoring, saves products_scored.csv + top50k.

Saves a partial CSV every CHUNK candidates so progress survives a kill.

Usage (on ai-chem):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aichem_score_fast.py
"""

import sys
import os
import gc
import json
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog('rdApp.*')

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "aichem_tier2_scaled"
COUPLED_CSV = RESULTS_DIR / "coupled_products_smiles.csv"
DESCRIPTORS_CSV = RESULTS_DIR / "products_descriptors.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"

N_PROCS = max(1, mp.cpu_count() - 2)
CHUNK = 64           # candidates per batched forward pass
SAVE_EVERY = 50_000  # save partial CSV every N mols scored

print(f"Workers: {N_PROCS}")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _score_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return {
            "smiles": smi,
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "QED": Chem.QED.qed(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "Rings": Descriptors.RingCount(mol),
        }
    except Exception:
        return None


def _fp_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    log("=" * 70)
    log("AI-CHEM FAST BATCHED SCORING (tiled tensor inference)")
    log("=" * 70)
    log(f"Started: {datetime.now().isoformat()}")

    if not COUPLED_CSV.exists():
        log(f"ERROR: {COUPLED_CSV} not found"); return
    df_in = pd.read_csv(COUPLED_CSV)
    log(f"Loaded {len(df_in):,} coupled products")

    # ---- descriptors (or load cached) ----
    if DESCRIPTORS_CSV.exists():
        log(f"Loading cached descriptors from {DESCRIPTORS_CSV}")
        df_desc = pd.read_csv(DESCRIPTORS_CSV)
        log(f"  Cached: {len(df_desc):,}")
    else:
        log(f"Step 1: descriptors (parallel on {N_PROCS} workers)")
        smiles = df_in["smiles"].tolist()
        with mp.Pool(N_PROCS) as pool:
            out = []
            for i, r in enumerate(pool.imap_unordered(_score_descriptors, smiles, chunksize=500)):
                if r is not None:
                    out.append(r)
                if (i + 1) % 100_000 == 0:
                    log(f"  descriptors {i+1:,}/{len(smiles):,}")
        df_desc = pd.DataFrame(out)
        df_desc.to_csv(DESCRIPTORS_CSV, index=False)
        log(f"  Done: {len(df_desc):,}")

    # ---- FPs (parallel) ----
    log(f"\nStep 2: Compute FPs (parallel)")
    desc_smiles = df_desc["smiles"].tolist()
    with mp.Pool(N_PROCS) as pool:
        fps = pool.map(_fp_one, desc_smiles, chunksize=500)
    valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
    log(f"  Valid FPs: {len(valid):,}")

    # ---- FiLMDelta tiled batch inference ----
    log(f"\nStep 3: FiLMDelta TILED-BATCH inference (CHUNK={CHUNK})")
    if not MODEL_PATH.exists():
        log(f"ERROR: {MODEL_PATH} not found"); return

    import torch
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    # Limit torch threads to avoid contention with the main loop
    torch.set_num_threads(N_PROCS)

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    anchor_embs = ckpt["anchor_embs"]                          # (280, 2048)
    anchor_pIC50 = np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)  # (280,)
    n_anchors = anchor_embs.shape[0]
    log(f"  Loaded model: {n_anchors} anchors")

    valid_arr = np.array([v[1] for v in valid], dtype=np.float32)
    valid_idx = [v[0] for v in valid]

    log(f"  Pre-scaling all candidate FPs...")
    valid_scaled = scaler.transform(valid_arr).astype(np.float32)
    valid_embs = torch.from_numpy(valid_scaled)               # (N, 2048)

    log(f"  Tiling anchor embeddings on demand...")
    scores = np.full(len(desc_smiles), np.nan)

    last_save = 0
    last_log_time = datetime.now()
    n_total = len(valid_embs)
    log(f"  Scoring {n_total:,} mols in chunks of {CHUNK}...")

    t_start = datetime.now()
    with torch.no_grad():
        for start in range(0, n_total, CHUNK):
            cand_chunk = valid_embs[start:start + CHUNK]      # (k, 2048)
            k = cand_chunk.shape[0]
            # Tile: each candidate is paired with each of the 280 anchors
            # anchor_tiled: anchor_embs repeated for each candidate → (k*280, 2048)
            # cand_tiled:   each candidate replicated 280 times → (k*280, 2048)
            anchor_tiled = anchor_embs.unsqueeze(0).expand(k, -1, -1).reshape(k * n_anchors, -1)
            cand_tiled = cand_chunk.unsqueeze(1).expand(-1, n_anchors, -1).reshape(k * n_anchors, -1)
            deltas = model(anchor_tiled, cand_tiled).numpy().reshape(k, n_anchors)
            # add anchor_pIC50, mean across anchors
            chunk_scores = (anchor_pIC50[None, :] + deltas).mean(axis=1)
            for j in range(k):
                scores[valid_idx[start + j]] = float(chunk_scores[j])

            done = start + k

            # Periodic snapshot save
            if done - last_save >= SAVE_EVERY or done >= n_total:
                df_partial = df_desc.copy()
                df_partial["pIC50_film"] = scores
                df_partial.to_csv(RESULTS_DIR / "products_scored_partial.csv", index=False)
                last_save = done

                now = datetime.now()
                elapsed = (now - t_start).total_seconds()
                rate = done / max(1, elapsed)
                eta_sec = (n_total - done) / max(1, rate)
                log(f"    {done:,}/{n_total:,} scored ({100*done/n_total:.1f}%, "
                    f"{rate:.0f} mol/s, ETA {eta_sec/60:.1f} min)")

    df_desc["pIC50_film"] = scores
    df_desc = df_desc.sort_values("pIC50_film", ascending=False, na_position='last')
    df_desc.to_csv(RESULTS_DIR / "products_scored.csv", index=False)
    df_desc.head(50_000).to_csv(RESULTS_DIR / "products_top50k.csv", index=False)

    summary = {
        "n_products": len(df_in),
        "n_scored": int(np.isfinite(scores).sum()),
        "pIC50_max": float(np.nanmax(scores)) if np.isfinite(scores).any() else None,
        "pIC50_median": float(np.nanmedian(scores)) if np.isfinite(scores).any() else None,
        "n_potent_7": int((scores >= 7.0).sum()),
        "n_potent_8": int((scores >= 8.0).sum()),
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n{json.dumps(summary, indent=2)}")
    log(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
