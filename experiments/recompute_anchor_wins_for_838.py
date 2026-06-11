"""Recompute anchor_wins + anchor_wins_ge7 for the 838 L5 visible survivors.

The cohort CSVs have buggy anchor_wins_ge7 values (up to 275; max possible
is 57 since only 57 of 280 anchors have pIC50 >= 7). This script reproduces
the canonical scorer logic on the 838 visible SMILES and writes the corrected
values into a side CSV that the backend backfills via SMILES.

Run on CPU — 838 mols × 280 anchors × 3-seed ensemble is small (~30 s).

Checkpoint format (`results/paper_evaluation/reinvent4_film_ensemble/film_seed{k}.pt`):
  - model_state: weights for FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024,512,256])
  - scaler_mean / scaler_scale: StandardScaler params over Morgan FP (2048-dim)
  - anchor_embs: (280, 2048) pre-standardized anchor fingerprints
  - anchor_pIC50: (280,) anchor potencies
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

ENSEMBLE_DIR = PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_ensemble"
MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def log(*a): print("[recompute]", *a, flush=True)


def morgan_fp(smi: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    return np.array(bv, dtype=np.float32)


def load_ensemble():
    """Load the 3 FiLMDelta seed checkpoints (new format)."""
    ensemble = []
    for k in range(3):
        path = ENSEMBLE_DIR / f"film_seed{k}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256])
        model.load_state_dict(ck["model_state"])
        model.eval()
        ensemble.append({
            "model": model,
            "scaler_mean": np.asarray(ck["scaler_mean"], dtype=np.float32),
            "scaler_scale": np.asarray(ck["scaler_scale"], dtype=np.float32),
            "anchor_embs": ck["anchor_embs"].float(),  # (280, 2048) — already standardized
            "anchor_pIC50": np.asarray(ck["anchor_pIC50"], dtype=np.float64),  # (280,)
        })
    n = len(ensemble[0]["anchor_pIC50"])
    n_ge7 = (ensemble[0]["anchor_pIC50"] >= 7).sum()
    log(f"loaded 3-seed FiLMDelta ensemble: {n} anchors, {n_ge7} with pIC50≥7")
    return ensemble


def standardize(fps: np.ndarray, ens: dict) -> torch.Tensor:
    """Apply this ensemble seed's scaler to raw Morgan fingerprints."""
    z = (fps - ens["scaler_mean"]) / ens["scaler_scale"]
    return torch.FloatTensor(z)


def get_visible_smiles():
    import requests
    r = requests.get(
        "http://localhost:5001/api/filter",
        params={"groups": "murcko,thiq,murcko_and_acryl", "length": "20000"},
        timeout=180,
    )
    return [rw["smiles"] for rw in r.json().get("rows", []) if rw.get("smiles")]


def score(cand_smis, ensemble):
    n_anchors = len(ensemble[0]["anchor_pIC50"])
    high_potency_mask = ensemble[0]["anchor_pIC50"] >= 7.0  # (280,)

    # Mol1 baseline
    mol1_fp = morgan_fp(MOL1_SMI)
    mol1_baseline_seeds = []
    mol1_emb_per_seed = []
    for ens in ensemble:
        m1_e = standardize(mol1_fp.reshape(1, -1), ens)  # (1, 2048)
        mol1_emb_per_seed.append(m1_e)
        with torch.no_grad():
            cand_t = m1_e.expand(n_anchors, -1)
            d = ens["model"](ens["anchor_embs"], cand_t).numpy()
            mol1_baseline_seeds.append(float((ens["anchor_pIC50"] + d).mean()))
    mol1_baseline = float(np.mean(mol1_baseline_seeds))
    log(f"Mol1 baseline pIC50 (ensemble mean): {mol1_baseline:.3f}")

    cand_fps = np.stack([morgan_fp(s) for s in cand_smis])
    cand_embs_per_seed = [standardize(cand_fps, ens) for ens in ensemble]
    n_cand = len(cand_smis)

    CHUNK = 64
    n_chunks = (n_cand + CHUNK - 1) // CHUNK
    log(f"scoring {n_cand:,} candidates × {n_anchors} anchors × 3 seeds, {n_chunks} chunks")

    pIC50 = np.zeros(n_cand); pIC50_std = np.zeros(n_cand)
    wins = np.zeros(n_cand); wins_ge7 = np.zeros(n_cand)
    direct = np.zeros(n_cand); delta_vs = np.zeros(n_cand)

    t0 = datetime.now()
    with torch.no_grad():
        for c in range(n_chunks):
            lo, hi = c * CHUNK, min((c + 1) * CHUNK, n_cand)
            k = hi - lo
            ps_means, ps_deltas, ps_direct = [], [], []
            for si, ens in enumerate(ensemble):
                chunk_e = cand_embs_per_seed[si][lo:hi]   # (k, 2048)
                a_tile = ens["anchor_embs"].unsqueeze(0).expand(k, -1, -1).reshape(k * n_anchors, -1)
                c_tile = chunk_e.unsqueeze(1).expand(-1, n_anchors, -1).reshape(k * n_anchors, -1)
                deltas = ens["model"](a_tile, c_tile).numpy().reshape(k, n_anchors)
                ps_deltas.append(deltas)
                ps_means.append((ens["anchor_pIC50"][None, :] + deltas).mean(axis=1))
                m1_a = mol1_emb_per_seed[si].expand(k, -1)
                d_direct = ens["model"](m1_a, chunk_e).numpy().flatten()
                ps_direct.append(d_direct)
            seed_means = np.array(ps_means)        # (3, k)
            all_deltas = np.array(ps_deltas)       # (3, k, 280)
            seed_directs = np.array(ps_direct)     # (3, k)
            mean = seed_means.mean(axis=0)
            std = seed_means.std(axis=0)
            w = (all_deltas > 0).sum(axis=2).mean(axis=0)
            w7 = ((all_deltas > 0) & high_potency_mask[None, None, :]).sum(axis=2).mean(axis=0)
            d_mean = seed_directs.mean(axis=0)
            d_vs = mean - mol1_baseline
            for j in range(k):
                pIC50[lo+j] = mean[j]; pIC50_std[lo+j] = std[j]
                wins[lo+j] = w[j]; wins_ge7[lo+j] = w7[j]
                direct[lo+j] = d_mean[j]; delta_vs[lo+j] = d_vs[j]
            if (c+1) % 4 == 0 or c == n_chunks-1:
                el = (datetime.now() - t0).total_seconds()
                rate = hi / max(1, el)
                log(f"  chunk {c+1}/{n_chunks} ({hi:,}/{n_cand:,}, {rate:.0f} mol/s)")

    return pd.DataFrame({
        "smiles": cand_smis,
        "pIC50_mean_recomp": pIC50,
        "pIC50_std_recomp": pIC50_std,
        "delta_vs_mol1_recomp": delta_vs,
        "direct_delta_from_mol1_recomp": direct,
        "anchor_wins_recomp": wins,
        "anchor_wins_ge7_recomp": wins_ge7,
    })


def main():
    log("fetching 838 visible SMILES...")
    cand_smis = get_visible_smiles()
    log(f"  {len(cand_smis):,} unique SMILES")
    ensemble = load_ensemble()
    out = score(cand_smis, ensemble)

    log("\n=== Recomputed distribution ===")
    for col in ["anchor_wins_recomp", "anchor_wins_ge7_recomp", "pIC50_mean_recomp"]:
        s = out[col]
        log(f"  {col}: min={s.min():.2f}  med={s.median():.2f}  max={s.max():.2f}")
    log(f"  anchor_wins_ge7 max should be ≤57 (= 57 anchors at pIC50≥7)")

    out_path = PROJECT_ROOT / "results/paper_evaluation/anchor_wins_recompute_838.csv"
    out.to_csv(out_path, index=False)
    log(f"\nwrote {out_path}")
    log(f"  rows: {len(out):,}")


if __name__ == "__main__":
    main()
