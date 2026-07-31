"""CUDA-adapted FiLM enricher (V100). Mirrors enrich_scored_with_film_and_metrics.py
but moves model+anchor embeddings to GPU and uses larger chunks (500 mols × 280 anchors
= 140K pair forward, well within V100 16 GB).

Usage: python experiments/enrich_film_gpu.py <scored_csv>
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


MODEL_CACHE = PROJECT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"


def load_film_cuda():
    """Load FiLM model + anchor embeddings directly from cache → CUDA.

    Avoids importing experiments.reinvent4_film_scorer (which hard-disables CUDA
    via CUDA_VISIBLE_DEVICES='' at import time for the REINVENT4 ExternalProcess
    contract).
    """
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    assert torch.cuda.is_available(), "CUDA not available — run on GPU host."
    assert MODEL_CACHE.exists(), f"Model cache missing: {MODEL_CACHE}"
    device = torch.device("cuda")

    ckpt = torch.load(MODEL_CACHE, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    anchor_embs = ckpt["anchor_embs"].to(device)
    anchor_pIC50 = ckpt["anchor_pIC50"]
    return model, scaler, anchor_embs, anchor_pIC50, device


def score_film_proper_batch_gpu(smiles_list, model, scaler, anchor_embs, anchor_pIC50,
                                device, mol_chunk=500):
    """GPU-batched FiLM scorer.

    For each chunk of `mol_chunk` molecules, build a (mol_chunk × n_anchors) pair
    matrix on device and run one forward pass.
    """
    n_anchors = len(anchor_pIC50)
    out = [np.nan] * len(smiles_list)

    valid_idx = []
    fps = []
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_idx.append(i)
        fps.append(arr)

    if not fps:
        return out

    fps_arr = np.array(fps)
    target_embs_all = torch.FloatTensor(scaler.transform(fps_arr))  # CPU
    anchor_pIC50_np = np.asarray(anchor_pIC50, dtype=np.float32)

    t_start = time.time()
    total_mols = len(valid_idx)
    with torch.no_grad():
        for start in range(0, total_mols, mol_chunk):
            end = min(start + mol_chunk, total_mols)
            chunk = target_embs_all[start:end].to(device)
            B = chunk.shape[0]
            anchor_rep = anchor_embs.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            target_rep = chunk.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            deltas = model(anchor_rep, target_rep).detach().cpu().numpy().flatten().reshape(B, n_anchors)
            abs_preds = deltas + anchor_pIC50_np.reshape(1, -1)
            mean_preds = abs_preds.mean(axis=1)
            std_preds = abs_preds.std(axis=1)
            for j, orig_idx in enumerate(valid_idx[start:end]):
                out[orig_idx] = (
                    float(mean_preds[j]),
                    float(std_preds[j]),
                    int((deltas[j] > 0).sum()),
                    int(((deltas[j] > 0) & (abs_preds[j] >= 7.0)).sum()),
                )
            if (start // mol_chunk) % 10 == 0:
                elapsed = time.time() - t_start
                rate = end / elapsed if elapsed > 0 else 0
                eta = (total_mols - end) / rate if rate > 0 else 0
                print(f"  [film-gpu] {end:>7,}/{total_mols:,}  ({rate:.0f} mol/s, ETA {eta/60:.1f}min)",
                      flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored_csv")
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    print(f"[enrich-gpu] Loading {args.scored_csv}", flush=True)
    df = pd.read_csv(args.scored_csv, low_memory=False)
    n = len(df)
    print(f"[enrich-gpu] {n:,} rows, {len(df.columns)} cols", flush=True)

    if "pIC50_film" in df.columns and df["pIC50_film"].notna().sum() > 0.9 * n:
        print(f"[enrich-gpu] Already enriched (pIC50_film present) — skipping")
        return

    print(f"[enrich-gpu] Loading FiLM model on CUDA", flush=True)
    model, scaler, anchor_embs, anchor_pIC50, device = load_film_cuda()
    print(f"[enrich-gpu] FiLM ready on {device}: {len(anchor_pIC50)} anchors", flush=True)

    print(f"[enrich-gpu] Scoring with mol_chunk={args.chunk}", flush=True)
    results = score_film_proper_batch_gpu(df["smiles"].tolist(), model, scaler,
                                          anchor_embs, anchor_pIC50, device,
                                          mol_chunk=args.chunk)

    pic50_film, pic50_std, anchor_wins, anchor_wins_ge7 = [], [], [], []
    for r in results:
        if isinstance(r, tuple):
            pic50_film.append(round(r[0], 4))
            pic50_std.append(round(r[1], 4))
            anchor_wins.append(r[2])
            anchor_wins_ge7.append(r[3])
        else:
            pic50_film.append(None)
            pic50_std.append(None)
            anchor_wins.append(None)
            anchor_wins_ge7.append(None)

    df["pIC50_film"] = pic50_film
    df["pIC50_mean"] = pic50_film
    df["pIC50_method"] = "FiLMDelta"
    df["pIC50_std"] = pic50_std
    df["anchor_wins"] = anchor_wins
    df["anchor_wins_ge7"] = anchor_wins_ge7
    df["delta_vs_mol1"] = [round(p - 6.59, 4) if p is not None else None for p in pic50_film]
    df["direct_delta_from_mol1"] = df["delta_vs_mol1"]

    # Efficiency metrics — only if descriptor columns are present
    pic50_arr = pd.to_numeric(df["pIC50_film"], errors="coerce")
    if {"HeavyAtoms", "LogP", "MW", "TPSA"}.issubset(df.columns):
        print(f"[enrich-gpu] Adding efficiency metrics", flush=True)
        df["LE"] = (1.4 * pic50_arr / df["HeavyAtoms"]).round(3)
        df["LLE"] = (pic50_arr - df["LogP"]).round(3)
        df["BEI"] = (1000 * pic50_arr / df["MW"]).round(2)
        df["SEI"] = (100 * pic50_arr / df["TPSA"]).round(2)
        df["SILE"] = (pic50_arr / (df["HeavyAtoms"] ** 0.3)).round(3)
    else:
        print(f"[enrich-gpu] Skipping efficiency metrics — descriptor cols missing", flush=True)

    df.to_csv(args.scored_csv, index=False)
    valid = pic50_arr.notna().sum()
    print(f"[enrich-gpu] DONE — wrote {args.scored_csv}")
    print(f"  Valid pIC50: {valid:,}/{n:,}")
    print(f"  Mean pIC50:  {pic50_arr.mean():.3f}, median: {pic50_arr.median():.3f}")
    print(f"  % pIC50>=7:  {(pic50_arr>=7.0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
