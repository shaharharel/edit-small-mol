"""Batched FiLM pIC50 scorer for already-scored CSVs.

Reads data/tier4_scored/{cohort}_scored.csv, adds pIC50_film column, writes back.
Uses TORCH BATCHED FiLM evaluation — orders of magnitude faster than per-mol loop.

Run: python experiments/score_film_batch.py data/tier4_scored/<cohort>_scored.csv
"""
from __future__ import annotations

import argparse
import sys
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


def load_film():
    from experiments.reinvent4_film_scorer import load_film_model
    return load_film_model()


def score_film_batched(smiles_list, model, scaler, anchor_embs, anchor_pIC50, batch_size=128):
    """Vectorized FiLM scoring: for each mol, predict delta vs ALL anchors, average.

    Returns list of pIC50 predictions; NaN for invalid SMILES.
    """
    n_anchors = len(anchor_pIC50)
    anchor_pIC50_t = torch.FloatTensor(anchor_pIC50)
    out = [float("nan")] * len(smiles_list)

    # Compute fingerprints for all valid mols
    valid_indices = []
    valid_fps = []
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_indices.append(i)
        valid_fps.append(arr)

    if not valid_fps:
        return out

    fps_arr = np.array(valid_fps)
    target_embs = torch.FloatTensor(scaler.transform(fps_arr))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            end = min(start + batch_size, len(valid_indices))
            chunk = target_embs[start:end]  # (B, 2048)
            B = chunk.shape[0]
            # For each mol in chunk: pair with all anchors
            # anchor_embs: (n_anchors, D); chunk_repeat: (B*n_anchors, D)
            anchor_rep = anchor_embs.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            target_rep = chunk.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            deltas = model(anchor_rep, target_rep).numpy().flatten()  # (B*n_anchors,)
            deltas = deltas.reshape(B, n_anchors)
            abs_preds = deltas + anchor_pIC50.reshape(1, -1)
            mean_preds = abs_preds.mean(axis=1)
            for j, orig_idx in enumerate(valid_indices[start:end]):
                out[orig_idx] = float(mean_preds[j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored_csv")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print(f"[film] Loading scored CSV: {args.scored_csv}")
    df = pd.read_csv(args.scored_csv)
    n = len(df)
    print(f"[film] {n:,} rows")

    if "pIC50_film" in df.columns and df["pIC50_film"].notna().sum() > 0.9 * n:
        print(f"[film] Skipping — already has pIC50_film")
        return

    print(f"[film] Loading FiLM model")
    model, scaler, anchor_embs, anchor_pIC50 = load_film()
    print(f"[film] Model loaded: {len(anchor_embs)} anchors")

    print(f"[film] Scoring (batch_size={args.batch_size})")
    scores = score_film_batched(
        df["smiles"].tolist(),
        model, scaler, anchor_embs, anchor_pIC50,
        batch_size=args.batch_size,
    )
    df["pIC50_film"] = scores
    df["pIC50_mean"] = scores
    df["delta_vs_mol1"] = [
        round(s - 6.59, 4) if s is not None and not np.isnan(s) else None for s in scores
    ]
    df.to_csv(args.scored_csv, index=False)
    valid = df["pIC50_film"].notna().sum()
    print(f"[film] Wrote {args.scored_csv}")
    print(f"[film]   Valid pIC50: {valid:,}/{n:,}")
    print(f"[film]   Mean pIC50:  {df['pIC50_film'].mean():.3f}")
    print(f"[film]   Median pIC50: {df['pIC50_film'].median():.3f}")
    print(f"[film]   % pIC50>=7: {(df['pIC50_film']>=7.0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
