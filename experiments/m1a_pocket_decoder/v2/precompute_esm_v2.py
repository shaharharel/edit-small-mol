#!/usr/bin/env python3
"""Phase 3b: Pre-compute ESM-2 (320d) per-residue pocket embeddings.

Operates on data/m1a_triples_v2/triples.parquet. Outputs an NPZ keyed by
sequence hash (so repeated pockets share embedding rows).

Outputs:
  data/m1a_triples_v2/esm2_cache.npz
    - seq_hashes: (U,) object  # md5 hexdigest of pocket pseudo-seq
    - residues_emb: (U, R_max, 320) float32
    - residues_mask: (U, R_max) bool
    - row_seq_idx: (N,) int32   # idx into U for each triple row
    - poses: (N, 6) float32
    - smiles: (N,) object
    - sources: (N,) object
    - struct_ids: (N,) object
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/triples.parquet"))
    ap.add_argument("--out", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache.npz"))
    ap.add_argument("--model_id", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    print(f"Loading {args.triples}...", flush=True)
    df = pd.read_parquet(args.triples)
    N = len(df)
    print(f"{N} rows", flush=True)

    seqs = []
    for _, row in df.iterrows():
        pocket = json.loads(row["pocket_residues"])
        pocket = sorted(pocket, key=lambda r: r["idx"])
        seq = "".join([r["aa"] if r["aa"] in "ACDEFGHIKLMNPQRSTVWY" else "X"
                        for r in pocket])
        seqs.append(seq)

    # Hash sequences and find unique
    hashes = [hashlib.md5(s.encode()).hexdigest() for s in seqs]
    unique_hash_to_seq = {}
    for h, s in zip(hashes, seqs):
        if h not in unique_hash_to_seq:
            unique_hash_to_seq[h] = s
    unique_hashes = list(unique_hash_to_seq.keys())
    unique_seqs = [unique_hash_to_seq[h] for h in unique_hashes]
    hash_to_idx = {h: i for i, h in enumerate(unique_hashes)}
    row_seq_idx = np.array([hash_to_idx[h] for h in hashes], dtype=np.int32)
    print(f"Unique pocket sequences: {len(unique_seqs)} (of {N} rows)", flush=True)

    seq_lens = [len(s) for s in unique_seqs]
    r_max = max(seq_lens)
    print(f"Pocket residue counts: min={min(seq_lens)} median={int(np.median(seq_lens))} max={r_max}", flush=True)

    # Load ESM-2 on CPU (Mac with MPS issues per CLAUDE.md)
    from transformers import AutoTokenizer, AutoModel
    torch.backends.mps.is_available = lambda: False
    device = torch.device("cpu")
    print(f"Loading ESM-2 ({args.model_id}) on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    d = model.config.hidden_size
    print(f"ESM-2 hidden size = {d}", flush=True)

    U = len(unique_seqs)
    residues_emb = np.zeros((U, r_max, d), dtype=np.float32)
    residues_mask = np.zeros((U, r_max), dtype=bool)

    with torch.no_grad():
        for start in tqdm(range(0, U, args.batch_size)):
            batch = unique_seqs[start:start + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state  # (B, L+2, d)
            for j, seq in enumerate(batch):
                L = len(seq)
                emb = out[j, 1:1 + L].cpu().numpy().astype(np.float32)
                residues_emb[start + j, :L] = emb
                residues_mask[start + j, :L] = True

    # Build per-row poses + ids
    poses = np.zeros((N, 6), dtype=np.float32)
    smiles_arr = []
    sources_arr = []
    struct_ids_arr = []
    for i, row in df.iterrows():
        poses[i] = np.array(json.loads(row["warhead_pose_6d"]), dtype=np.float32)
        smiles_arr.append(row["smiles"])
        sources_arr.append(row["source"])
        struct_ids_arr.append(row["struct_id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        seq_hashes=np.array(unique_hashes, dtype=object),
        residues_emb=residues_emb,
        residues_mask=residues_mask,
        row_seq_idx=row_seq_idx,
        poses=poses,
        smiles=np.array(smiles_arr, dtype=object),
        sources=np.array(sources_arr, dtype=object),
        struct_ids=np.array(struct_ids_arr, dtype=object),
    )
    print(f"Wrote {out_path}", flush=True)
    print(f"  U={U} unique sequences  R_max={r_max}  D={d}", flush=True)
    print(f"  N={N} rows  size={out_path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
