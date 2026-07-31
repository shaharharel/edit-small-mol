"""Pre-compute ESM-2 residue embeddings for every pocket in triples.parquet.

We cannot reproduce the *true* ZAP70 / CovInDB target sequences from a tiny
pocket window without the full UniProt mapping. Solution: build a per-row
"pseudo sequence" from the pocket residues themselves (one-letter AA codes,
ordered by residue index), then run ESM-2-8M on each pseudo sequence and grab
the residue token embeddings.

This works because the conditioning module only needs a residue-level
representation of the *pocket* environment; the upstream pretrained ESM
weights still inject sensible per-residue biochemistry.

Output: data/m1a_triples/pocket_embeddings.npz
  - residues_emb: (N, R_max, 320) float32 padded with zeros
  - residues_mask: (N, R_max) bool, True where real
  - poses: (N, 6) float32
  - smiles: (N,) object
  - sources, struct_ids: (N,) object
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=str(PROJECT_ROOT / "data/m1a_triples/triples.parquet"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/m1a_triples/pocket_embeddings.npz"))
    ap.add_argument("--model_id", default="facebook/esm2_t6_8M_UR50D",
                     help="ESM-2 8M model id from Hugging Face Hub")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    print(f"Loading {args.triples}...", flush=True)
    df = pd.read_parquet(args.triples)
    print(f"{len(df)} rows", flush=True)

    # Build pseudo-sequence per row: residues sorted by idx, take one-letter aa
    seqs = []
    masks_meta = []
    for _, row in df.iterrows():
        pocket = json.loads(row["pocket_residues"])
        # sort by residue index
        pocket = sorted(pocket, key=lambda r: r["idx"])
        seq = "".join([r["aa"] if r["aa"] in "ACDEFGHIKLMNPQRSTVWY" else "X"
                        for r in pocket])
        seqs.append(seq)
        masks_meta.append(len(seq))
    r_max = max(masks_meta)
    print(f"Pocket residue counts: min={min(masks_meta)} median={int(np.median(masks_meta))}"
           f" max={r_max} mean={np.mean(masks_meta):.1f}", flush=True)

    # Load ESM-2
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM-2 ({args.model_id}) on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    d = model.config.hidden_size
    print(f"ESM-2 hidden size = {d}", flush=True)
    assert d == 320, f"Expected 320, got {d}"

    # Pre-allocate output arrays
    N = len(df)
    residues_emb = np.zeros((N, r_max, d), dtype=np.float32)
    residues_mask = np.zeros((N, r_max), dtype=bool)
    poses = np.zeros((N, 6), dtype=np.float32)
    smiles = []
    sources = []
    struct_ids = []
    for i, row in df.iterrows():
        poses[i] = np.array(json.loads(row["warhead_pose_6d"]), dtype=np.float32)
        smiles.append(row["smiles"])
        sources.append(row["source"])
        struct_ids.append(row["struct_id"])

    # Batch ESM-2 inference
    print(f"Running ESM-2 in batches of {args.batch_size}...", flush=True)
    with torch.no_grad():
        for start in tqdm(range(0, N, args.batch_size)):
            batch = seqs[start:start + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state  # (B, L+2, 320)
            attn_mask = enc["attention_mask"].bool()
            for j, seq in enumerate(batch):
                # ESM-2 prepends CLS, appends EOS
                seq_len = len(seq)
                # token positions 1..1+seq_len (inclusive) are residues
                emb = out[j, 1:1 + seq_len].cpu().numpy().astype(np.float32)
                residues_emb[start + j, :seq_len] = emb
                residues_mask[start + j, :seq_len] = True

    np.savez_compressed(
        args.out,
        residues_emb=residues_emb,
        residues_mask=residues_mask,
        poses=poses,
        smiles=np.array(smiles, dtype=object),
        sources=np.array(sources, dtype=object),
        struct_ids=np.array(struct_ids, dtype=object),
    )
    print(f"Wrote {args.out} (N={N}, R_max={r_max}, D=320)", flush=True)


if __name__ == "__main__":
    main()
