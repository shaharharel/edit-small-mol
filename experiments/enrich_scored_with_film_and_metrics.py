"""Enrich a scored CSV with:
  - FiLM pIC50 (batched, 100 mols × 280 anchors = 28K pairs/forward)
  - delta_vs_mol1, direct_delta_from_mol1
  - anchor_wins, anchor_wins_ge7
  - Efficiency metrics: LE, LLE, BEI, SEI, SILE
  - pIC50 method/mean/std (aliases)

Run: python experiments/enrich_scored_with_film_and_metrics.py <scored_csv>
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


def load_film():
    from experiments.reinvent4_film_scorer import load_film_model
    return load_film_model()


def score_film_proper_batch(smiles_list, model, scaler, anchor_embs, anchor_pIC50, mol_chunk=100):
    """Properly batched FiLM scorer.

    For each chunk of `mol_chunk` molecules, build a (mol_chunk × n_anchors) pair
    matrix and run one forward pass.
    """
    n_anchors = len(anchor_pIC50)
    out_pIC50 = [np.nan] * len(smiles_list)

    # Compute fingerprints for all valid mols
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
        return out_pIC50

    fps_arr = np.array(fps)
    target_embs_all = torch.FloatTensor(scaler.transform(fps_arr))
    anchor_pIC50_t = torch.FloatTensor(anchor_pIC50)

    model.eval()
    t_start = time.time()
    total_mols = len(valid_idx)
    with torch.no_grad():
        for start in range(0, total_mols, mol_chunk):
            end = min(start + mol_chunk, total_mols)
            chunk = target_embs_all[start:end]
            B = chunk.shape[0]
            anchor_rep = anchor_embs.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            target_rep = chunk.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            deltas = model(anchor_rep, target_rep).numpy().flatten().reshape(B, n_anchors)
            abs_preds = deltas + anchor_pIC50.reshape(1, -1)
            mean_preds = abs_preds.mean(axis=1)
            std_preds = abs_preds.std(axis=1)
            for j, orig_idx in enumerate(valid_idx[start:end]):
                out_pIC50[orig_idx] = (float(mean_preds[j]), float(std_preds[j]),
                                       int((deltas[j] > 0).sum()),
                                       int(((deltas[j] > 0) & (abs_preds[j] >= 7.0)).sum()))
            if (start // mol_chunk) % 50 == 0:
                elapsed = time.time() - t_start
                rate = end / elapsed if elapsed > 0 else 0
                eta = (total_mols - end) / rate if rate > 0 else 0
                print(f"  [film] {end:>6,}/{total_mols:,}  ({rate:.0f} mol/s, ETA {eta/60:.1f}min)")
    return out_pIC50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored_csv")
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    print(f"[enrich] Loading {args.scored_csv}")
    df = pd.read_csv(args.scored_csv)
    n = len(df)
    print(f"[enrich] {n:,} rows, {len(df.columns)} cols")

    if "pIC50_film" in df.columns and df["pIC50_film"].notna().sum() > 0.9 * n:
        print(f"[enrich] Already enriched (pIC50_film present)")
        return

    print(f"[enrich] Loading FiLM model")
    model, scaler, anchor_embs, anchor_pIC50 = load_film()
    print(f"[enrich] FiLM ready: {len(anchor_embs)} anchors")

    print(f"[enrich] Scoring with chunk={args.chunk}")
    results = score_film_proper_batch(df["smiles"].tolist(), model, scaler, anchor_embs, anchor_pIC50, mol_chunk=args.chunk)

    pic50_film = []
    pic50_std = []
    anchor_wins = []
    anchor_wins_ge7 = []
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

    # Efficiency metrics
    print(f"[enrich] Adding efficiency metrics")
    pic50_arr = pd.to_numeric(df["pIC50_film"], errors="coerce")
    df["LE"] = (1.4 * pic50_arr / df["HeavyAtoms"]).round(3)  # ligand efficiency
    df["LLE"] = (pic50_arr - df["LogP"]).round(3)              # lipophilic LE
    df["BEI"] = (1000 * pic50_arr / df["MW"]).round(2)         # binding eff index
    df["SEI"] = (100 * pic50_arr / df["TPSA"]).round(2)        # surface eff index
    df["SILE"] = (pic50_arr / (df["HeavyAtoms"] ** 0.3)).round(3)  # size-indep LE

    df.to_csv(args.scored_csv, index=False)
    valid = pic50_arr.notna().sum()
    print(f"[enrich] DONE — wrote {args.scored_csv}")
    print(f"  Valid pIC50: {valid:,}/{n:,}")
    print(f"  Mean pIC50:  {pic50_arr.mean():.3f}, median: {pic50_arr.median():.3f}")
    print(f"  % pIC50>=7:  {(pic50_arr>=7.0).mean()*100:.1f}%")
    print(f"  Median LE: {df['LE'].median():.3f}, LLE: {df['LLE'].median():.2f}, BEI: {df['BEI'].median():.1f}")


if __name__ == "__main__":
    main()
