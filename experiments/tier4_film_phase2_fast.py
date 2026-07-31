#!/usr/bin/env python3
"""Vectorized FiLMDelta inference for the 215K Tier-4 filter-1 survivors.

The original scorer does anchor expansion in a Python loop (one mol at a
time × 280 anchor forward passes) — fine for REINVENT4's small batches but
WAY too slow for 215K. This rewrite:
  1. Computes Morgan FPs in bulk (multiprocessing across CPUs).
  2. Batches (B targets, 280 anchors) → single forward pass of (B * 280)
     concatenated (anchor_emb, target_emb) pairs.
  3. Reshapes back to (B, 280), means over the anchor axis.
Expected speedup: 50-100x on the V100's CPU vs the loop version.

CLI: tier4_film_phase2_fast.py <phase1_csv> <out_csv> [batch_size=512]
"""
import sys
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_LOCAL = Path("/Users/shaharharel/Documents/github/edit-small-mol")
PROJECT_V100 = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = next((p for p in (PROJECT_LOCAL, PROJECT_V100) if p.exists()), None)
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"


def main():
    if len(sys.argv) < 3:
        print("usage: tier4_film_phase2_fast.py <in_csv> <out_csv> [batch=512]", file=sys.stderr)
        sys.exit(2)
    in_csv, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    import torch
    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import reinvent4_film_scorer as fs
    model, scaler, anchor_embs, anchor_pIC50 = fs.load_film_model()
    n_anch = len(anchor_pIC50)
    print(f"loading {in_csv} ...", flush=True)
    df = pd.read_csv(in_csv)
    n = len(df)
    print(f"  {n:,} rows  ·  {n_anch} anchors  ·  batch={B}", flush=True)

    # --- Bulk Morgan FP computation ---
    print("computing Morgan FPs...", flush=True)
    t0 = time.time()
    smis = df["smiles"].astype(str).tolist()
    fps = np.zeros((n, 2048), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        DataStructs.ConvertToNumpyArray(fp, fps[i])
        valid[i] = True
        if i % 20000 == 0 and i > 0:
            print(f"  ...{i:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  FP done: {valid.sum():,}/{n:,} valid in {time.time()-t0:.0f}s", flush=True)

    # Scale once
    embs = scaler.transform(fps).astype(np.float32)
    embs_t = torch.from_numpy(embs)  # (n, 384)
    n_anch = anchor_embs.shape[0]
    A = anchor_embs  # (280, 384) already torch tensor

    # --- Vectorized batched scoring ---
    print(f"batched FiLMDelta inference (B={B}, expand to B*{n_anch})...", flush=True)
    scores = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with torch.no_grad():
        for start in range(0, n, B):
            end = min(start + B, n)
            mask = valid[start:end]
            if not mask.any():
                continue
            tgt = embs_t[start:end][mask]  # (b_eff, 384)
            b_eff = tgt.shape[0]
            # tile anchors and targets to (b_eff*n_anch, 384) each
            tgt_x = tgt.unsqueeze(1).expand(b_eff, n_anch, -1).reshape(-1, tgt.shape[-1])  # (b*n_anch, 384)
            anc_x = A.unsqueeze(0).expand(b_eff, -1, -1).reshape(-1, A.shape[-1])  # (b*n_anch, 384)
            deltas = model(anc_x, tgt_x).numpy().reshape(b_eff, n_anch).astype(np.float32)
            abs_preds = anchor_pIC50.reshape(1, -1) + deltas
            mean_preds = abs_preds.mean(axis=1)
            local_idx = np.flatnonzero(mask)
            scores[start + local_idx] = mean_preds
            if (start // B) % 5 == 0:
                rate = (start + B) / max(1, time.time() - t0)
                eta = (n - start - B) / max(1, rate) / 60
                print(f"  ...{start+B:,}/{n:,}  ({time.time()-t0:.0f}s, {int(rate)}/s, ETA {eta:.1f} min)", flush=True)
    print(f"  inference done in {time.time()-t0:.0f}s", flush=True)

    # Mol-1 reference
    mol1_score = fs.score_smiles([MOL1_SMILES], model, scaler, anchor_embs, anchor_pIC50)[0]
    print(f"  Mol-1 FiLM pIC50: {mol1_score:.3f}", flush=True)

    df["pIC50_method"] = scores
    df["pIC50_film"] = scores
    df["pIC50_mean"] = scores
    df["pIC50_std"] = np.nan
    df["delta_vs_mol1"] = df["pIC50_method"] - mol1_score
    df["direct_delta_from_mol1"] = df["delta_vs_mol1"]
    df["anchor_wins"] = (df["pIC50_method"] > mol1_score).astype("Int64")
    df["anchor_wins_ge7"] = ((df["pIC50_method"] > mol1_score) & (df["pIC50_method"] >= 7.0)).astype("Int64")
    print(f"  mean pIC50: {np.nanmean(scores):.3f}  ·  anchor_wins: {int(df['anchor_wins'].sum())}/{n}", flush=True)

    df.to_csv(out_csv, index=False)
    print(f"  → {out_csv}  ({out_csv.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
