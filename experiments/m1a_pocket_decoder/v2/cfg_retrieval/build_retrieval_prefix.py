"""Build the retrieval prefix for ZAP70.

Given the ZAP70 pocket ESM readout (Mol1's fresh Boltz cofold in
`data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz`), rank all CovInDB v2 pockets by
cosine similarity of their MEAN-POOLED ESM residue embedding to the ZAP70
mean-pooled pocket, LEAVE-ONE-OUT ZAP70 itself (any struct_id containing
'ZAP70' or matching known ZAP70 PDBs), and pick the top K rows.  For each
retained row, emit the SMILES.

Output: data/paper_pair_training/cfg_retrieval/retrieval_top{K}.json with
    {
      "zap70_mean_pool_norm": ...,
      "excluded_struct_ids": [...],
      "top_k": [
         {"rank": 1, "struct_id": ..., "smiles": ..., "cosine": ...},
         ...
      ]
    }
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# The v2 CovInDB cache tags every row with a `source` field.  Rows with
# source == 'boltz_zap70' are the ZAP70 curriculum (Mol1-scaffold acrylamide
# series cofolded against ZAP70), so we exclude them entirely to preserve
# the leave-one-out property.  Additionally exclude any struct_id starting
# with a canonical ZAP70 PDB code, in case the corpus grows.
ZAP70_STRUCT_ID_PREFIXES = (
    "2OZO", "2OQ1", "4K2R", "1U59", "4XZ0", "4XZ1", "4WFG", "4WFH",
)
ZAP70_SOURCES = ("boltz_zap70",)


def pool_pocket(residues_emb: np.ndarray, residues_mask: np.ndarray) -> np.ndarray:
    m = residues_mask.astype(np.float32)[..., None]
    denom = m.sum(axis=-2).clip(min=1e-6)
    return (residues_emb * m).sum(axis=-2) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--zap70_esm", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/retrieval_top5.json"))
    args = ap.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    residues_emb = cache["residues_emb"]       # (U, R_max, 320)
    residues_mask = cache["residues_mask"]     # (U, R_max)
    row_seq_idx = cache["row_seq_idx"]         # (N,)
    smiles = cache["smiles"]                    # (N,)
    struct_ids = cache["struct_ids"]            # (N,)
    sources = cache["sources"]                  # (N,)

    U = residues_emb.shape[0]
    print(f"Cache: U={U} pockets, N={len(smiles)} rows", flush=True)

    zap = np.load(args.zap70_esm, allow_pickle=True)
    zap_emb = zap["residues_emb"][0]           # (R, 320)
    zap_mask = zap["residues_mask"][0]         # (R,)
    zap_pool = pool_pocket(zap_emb[None], zap_mask[None])[0]   # (320,)
    zap_pool = zap_pool / (np.linalg.norm(zap_pool) + 1e-8)
    print(f"ZAP70 pocket: {int(zap_mask.sum())} residues, "
          f"pool ‖·‖={np.linalg.norm(zap_pool):.4f} (should be 1.0)", flush=True)

    # Per-pocket pooled embeddings.
    pool = pool_pocket(residues_emb, residues_mask)            # (U, 320)
    pool = pool / (np.linalg.norm(pool, axis=-1, keepdims=True) + 1e-8)
    cos_per_pocket = pool @ zap_pool                             # (U,)

    # Broadcast to per-row cosine.
    cos_per_row = cos_per_pocket[row_seq_idx]                    # (N,)

    # Build exclusion mask: any row from a ZAP70-tagged source OR a struct_id
    # starting with a canonical ZAP70 PDB code.
    excluded_mask = np.zeros(len(struct_ids), dtype=bool)
    excluded_sids = []
    for i in range(len(struct_ids)):
        sid = str(struct_ids[i]).upper()
        src = str(sources[i]).lower()
        if src in ZAP70_SOURCES:
            excluded_mask[i] = True
            excluded_sids.append(f"{sid} [src={src}]")
            continue
        for pref in ZAP70_STRUCT_ID_PREFIXES:
            if sid.startswith(pref) or f"_{pref}_" in sid or f"_{pref}" in sid \
                    or f"{pref}_" in sid:
                excluded_mask[i] = True
                excluded_sids.append(sid)
                break
    print(f"Excluded {int(excluded_mask.sum())} rows "
          f"(source in {ZAP70_SOURCES} or struct_id starts with "
          f"{ZAP70_STRUCT_ID_PREFIXES})", flush=True)

    # Rank remaining rows by cosine (descending).
    keep_mask = ~excluded_mask
    order = np.argsort(-cos_per_row)
    # Walk in order, take top K UNIQUE canonical SMILES.
    seen_smi = set()
    picks = []
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    for idx in order:
        if not keep_mask[idx]:
            continue
        smi = str(smiles[idx])
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, canonical=True)
        if canon in seen_smi:
            continue
        seen_smi.add(canon)
        picks.append({
            "rank": len(picks) + 1,
            "row_index": int(idx),
            "struct_id": str(struct_ids[idx]),
            "smiles": canon,
            "cosine": float(cos_per_row[idx]),
        })
        if len(picks) >= args.k:
            break

    out = {
        "zap70_pocket_size": int(zap_mask.sum()),
        "excluded_struct_id_count": int(excluded_mask.sum()),
        "excluded_struct_id_sample": excluded_sids[:20],
        "top_k": picks,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Wrote {outp}", flush=True)
    for p in picks:
        print(f"  rank {p['rank']}  cos={p['cosine']:.4f}  "
              f"sid={p['struct_id']}  smi={p['smiles']}", flush=True)


if __name__ == "__main__":
    main()
