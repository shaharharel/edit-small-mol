#!/usr/bin/env python3
"""GPU FiLMDelta inference. Loads model on cuda:0, batches large.

The reinvent4_film_scorer module forces CPU via CUDA_VISIBLE_DEVICES=''
at import time. We unset that BEFORE importing, so the model can use CUDA.

For 215K mols × 280 anchors, V100 GPU completes in ~30 sec.

CLI: tier4_film_phase2_gpu.py <in_csv> <out_csv> [batch=4096]
"""
import os
# CRITICAL: undo the scorer module's CPU forcing *before* import
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol") if Path("/home/shaharh_quris_ai").exists() \
    else Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"


def main():
    in_csv = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    # B*n_anch*D*4 bytes peak; for D=2048, n_anch=280, B=1024 → ~2.3 GB. Safe on 16 GB V100.
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    import torch
    print(f"torch {torch.__version__}  ·  cuda available: {torch.cuda.is_available()}  ·  device count: {torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        print("CUDA not available — exiting (use CPU script instead).", file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda:0")
    print(f"  using {torch.cuda.get_device_name(0)}", flush=True)

    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    # Re-import scorer. Its top-level sets CUDA_VISIBLE_DEVICES='' at import,
    # so re-pop AFTER import to restore GPU visibility for any later CUDA call.
    import reinvent4_film_scorer as fs
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    model, scaler, anchor_embs, anchor_pIC50 = fs.load_film_model()
    model = model.to(device).eval()
    anchor_embs = anchor_embs.to(device)  # (280, D) — D is 2048 (Morgan FP fed directly, no projection)
    anchor_pIC50_t = torch.from_numpy(np.asarray(anchor_pIC50, dtype=np.float32)).to(device)
    n_anch = anchor_embs.shape[0]
    D = anchor_embs.shape[1]
    print(f"  model + anchors on GPU  ·  n_anchors={n_anch}  ·  emb_dim={D}", flush=True)

    df = pd.read_csv(in_csv)
    n = len(df)
    print(f"loaded {in_csv.name}: {n:,} rows", flush=True)

    # Bulk Morgan FPs on CPU
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
        if i % 50000 == 0 and i > 0:
            print(f"  ...{i:,}/{n:,}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  FPs done: {valid.sum():,}/{n:,} valid in {time.time()-t0:.1f}s", flush=True)

    embs = scaler.transform(fps).astype(np.float32)
    embs_t = torch.from_numpy(embs).to(device)  # (n, 384)

    print(f"GPU batched inference (B={B}, n_anch={n_anch}, D={D})...", flush=True)
    scores = np.full(n, np.nan, dtype=np.float32)
    stds   = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, B):
            end = min(start + B, n)
            mask = valid[start:end]
            if not mask.any():
                continue
            tgt = embs_t[start:end][torch.from_numpy(mask).to(device)]  # (b_eff, D)
            b_eff = tgt.shape[0]
            tgt_x = tgt.unsqueeze(1).expand(b_eff, n_anch, -1).reshape(-1, D)
            anc_x = anchor_embs.unsqueeze(0).expand(b_eff, -1, -1).reshape(-1, D)
            deltas = model(anc_x, tgt_x).view(b_eff, n_anch)
            abs_preds = anchor_pIC50_t.unsqueeze(0) + deltas  # (b_eff, n_anch)
            mean_preds = abs_preds.mean(dim=1).cpu().numpy()
            std_preds  = abs_preds.std(dim=1).cpu().numpy()
            local_idx = np.flatnonzero(mask)
            scores[start + local_idx] = mean_preds
            stds[start + local_idx]   = std_preds
            del tgt_x, anc_x, deltas, abs_preds
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  inference: {dt:.2f}s  ·  {n/dt:.0f} mol/s", flush=True)

    # Mol-1 ref: inline GPU compute (1 mol, reuse path) for consistency.
    m1 = Chem.MolFromSmiles(MOL1_SMILES)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    a1 = np.zeros(2048, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp1, a1)
    e1 = scaler.transform(a1.reshape(1, -1)).astype(np.float32)
    e1_t = torch.from_numpy(e1).to(device).expand(n_anch, -1)
    with torch.no_grad():
        d1 = model(anchor_embs, e1_t).view(-1)
        mol1_score = float((anchor_pIC50_t + d1).mean().cpu())
    print(f"  Mol-1 reference: {mol1_score:.3f}", flush=True)

    df["pIC50_method"] = scores
    df["pIC50_film"] = scores
    df["pIC50_mean"] = scores
    df["pIC50_std"] = stds
    df["delta_vs_mol1"] = df["pIC50_method"] - mol1_score
    df["direct_delta_from_mol1"] = df["delta_vs_mol1"]
    df["anchor_wins"] = (df["pIC50_method"] > mol1_score).astype("Int64")
    df["anchor_wins_ge7"] = ((df["pIC50_method"] > mol1_score) & (df["pIC50_method"] >= 7.0)).astype("Int64")
    print(f"  mean pIC50: {np.nanmean(scores):.3f}  ·  anchor_wins: {int(df['anchor_wins'].sum()):,}/{n:,}  ·  ge7: {int(df['anchor_wins_ge7'].sum()):,}", flush=True)
    df.to_csv(out_csv, index=False)
    print(f"  → {out_csv}  ({out_csv.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
