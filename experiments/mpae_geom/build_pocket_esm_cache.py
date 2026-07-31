#!/usr/bin/env python3
"""Build ESM-2 pocket embeddings for the 8 mpae-geom kinase targets.

For each of the 8 targets (BMX, BTK, EGFR, FGFR1, FGFR4_477, FGFR4_552, ITK,
JAK3) we:
  1. Read the sequence from any one Boltz output prot.pdb for that target
  2. Locate the cys_res in the sequence
  3. Take 9 residues on each side (18 total) → the same 18-residue pocket
     window used by the m1a v2 training cache
  4. Encode with ESM-2 (esm2_t6_8M for cache compatibility — 320-dim)
  5. Cache to data/paper_pair_training/mpae_geom/esm_cache.npz with format:
     seq_hashes (8,) str
     residues_emb (8, 18, 320) float32
     residues_mask (8, 18) bool
     row_seq_idx (N,) int32       -- per labeled row: target index
     target_names (8,) str
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# 3-letter → 1-letter
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

WINDOW = 9  # residues each side → 18 residues total


def parse_seq_and_cys(pdb_path: Path, cys_res: int) -> tuple[str, dict[int, int]]:
    """Return (sequence_1letter, {resnum -> seq_position})."""
    seq_chars = []
    res_map: dict[int, int] = {}
    seen: set[tuple[str, int]] = set()
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            atom = ln[12:16].strip()
            if atom != "CA":
                continue
            chain = ln[21]
            try:
                resnum = int(ln[22:26])
            except Exception:
                continue
            key = (chain, resnum)
            if key in seen:
                continue
            seen.add(key)
            resname = ln[17:20].strip()
            aa1 = AA3.get(resname, "X")
            res_map[resnum] = len(seq_chars)
            seq_chars.append(aa1)
    return "".join(seq_chars), res_map


def pocket_window(seq: str, res_map: dict[int, int], cys_res: int) -> tuple[str, list[int]]:
    """Return (pocket_seq, list_of_seq_positions_used)."""
    center = res_map.get(cys_res)
    if center is None:
        return "", []
    lo = max(0, center - WINDOW)
    hi = min(len(seq), center + WINDOW)
    return seq[lo:hi], list(range(lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/labeled_mols.csv"))
    ap.add_argument("--warhead_csv", default=str(PROJECT_ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"))
    ap.add_argument("--cifs_root",   default=str(PROJECT_ROOT / "data/covalid_mv_cofolds/cifs"))
    ap.add_argument("--out_npz",     default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/esm_cache.npz"))
    ap.add_argument("--esm_model",   default="esm2_t6_8M_UR50D")
    args = ap.parse_args()

    labeled = pd.read_csv(args.labeled_csv)
    print(f"[in] {len(labeled)} labeled rows")

    # Pick one representative row per target for sequence extraction
    warhead = pd.read_csv(args.warhead_csv)
    reps = warhead.drop_duplicates(subset="target", keep="first")[["target", "name", "cys_res"]]

    target_seqs: dict[str, tuple[str, int]] = {}
    for _, row in reps.iterrows():
        tgt = row["target"]; name = row["name"]; cys_res = int(row["cys_res"])
        # Locate PDB for this rep row
        # Try covalid_mv_cofolds subdir first, then bmx_stratified
        for sub in ["covalid_mv_cofolds", "bmx_stratified_cofolds"]:
            pdb = Path(args.cifs_root) / sub / tgt / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.prot.pdb"
            if pdb.exists():
                break
        if not pdb.exists():
            print(f"[warn] no prot.pdb for {tgt}/{name}"); continue
        seq, res_map = parse_seq_and_cys(pdb, cys_res)
        pocket_seq, positions = pocket_window(seq, res_map, cys_res)
        if not pocket_seq:
            print(f"[warn] no Cys {cys_res} in {tgt} seq"); continue
        target_seqs[tgt] = (pocket_seq, cys_res)
        print(f"  {tgt}: cys{cys_res} pocket len={len(pocket_seq)}  {pocket_seq}")

    targets = sorted(target_seqs.keys())
    print(f"[targets] {targets}")

    # Encode with ESM
    print(f"[esm] loading {args.esm_model}")
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(args.esm_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()

    R_max = 2 * WINDOW  # 18
    residues_emb = np.zeros((len(targets), R_max, 320), dtype=np.float32)
    residues_mask = np.zeros((len(targets), R_max), dtype=bool)
    for t_i, tgt in enumerate(targets):
        pocket_seq, cys = target_seqs[tgt]
        batch = [(f"{tgt}_cys{cys}", pocket_seq)]
        _, _, batch_toks = bc(batch)
        batch_toks = batch_toks.to(device)
        with torch.no_grad():
            out = model(batch_toks, repr_layers=[6], return_contacts=False)
        rep = out["representations"][6][0, 1:1 + len(pocket_seq)].cpu().numpy()
        L = min(rep.shape[0], R_max)
        residues_emb[t_i, :L] = rep[:L]
        residues_mask[t_i, :L] = True
        print(f"  encoded {tgt}: shape={rep.shape}  L={L}")

    # Build row_seq_idx
    tgt_to_idx = {t: i for i, t in enumerate(targets)}
    row_seq_idx = np.array([tgt_to_idx[t] for t in labeled["target"].tolist()], dtype=np.int32)

    seq_hashes = np.array([
        hashlib.md5(target_seqs[t][0].encode()).hexdigest()[:12]
        for t in targets
    ])

    np.savez_compressed(
        args.out_npz,
        seq_hashes=seq_hashes,
        residues_emb=residues_emb,
        residues_mask=residues_mask,
        row_seq_idx=row_seq_idx,
        target_names=np.array(targets),
    )
    print(f"[write] {args.out_npz}  residues_emb={residues_emb.shape}")


if __name__ == "__main__":
    main()
