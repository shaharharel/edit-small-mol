"""Sample from the trained M1a model conditioned on a chosen pocket + warhead pose.

For the headline run:
  anchor SMILES = Mol1 = C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1
  pocket = ZAP70 (Cys346) pocket residues from data/m1a_triples (using one of
            the Boltz ZAP70 cofold poses as a representative pocket
            embedding)
  warhead pose = Mol1's warhead Cβ position relative to Cys346 SG, BD angle, etc

Output: data/m1a_cohorts/cohort_mol1_zap70.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder"))
from m1a_model import load_m1a  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def randomize_smi(smi: str) -> str:
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def pick_zap70_pocket(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Pick a ZAP70 pocket from the training-time triples to use as conditioning.

    Returns (residues_emb, residues_mask, pose, struct_id).
    """
    d = np.load(npz_path, allow_pickle=True)
    sources = d["sources"]
    # Find rows from boltz_zap70 source
    zap_idx = np.where(sources == "boltz_zap70")[0]
    if len(zap_idx) == 0:
        raise RuntimeError("No boltz_zap70 source rows in pocket embeddings npz")
    # Among ZAP70 rows, pick the one with the highest mask coverage (largest
    # pocket = more residues) as the canonical conditioning
    masks = d["residues_mask"][zap_idx]
    n_real = masks.sum(axis=1)
    pick = zap_idx[int(np.argmax(n_real))]
    print(f"Picked struct {d['struct_ids'][pick]} with {n_real.max()} residues",
           flush=True)
    return (d["residues_emb"][pick], d["residues_mask"][pick],
             d["poses"][pick], str(d["struct_ids"][pick]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "models/m1a_checkpoints/m1a_final.ckpt"))
    ap.add_argument("--triples_emb", default=str(PROJECT_ROOT / "data/m1a_triples/pocket_embeddings.npz"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "data/m1a_cohorts/cohort_mol1_zap70.csv"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--randomize_smiles", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model
    print(f"Loading prior+ckpt...", flush=True)
    model = load_m1a(args.prior, device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    model.eval()

    # Pick ZAP70 pocket
    print("Picking ZAP70 pocket conditioning...", flush=True)
    res_emb_np, res_mask_np, pose_np, struct_id = pick_zap70_pocket(Path(args.triples_emb))
    print(f"Pocket residues: {int(res_mask_np.sum())}, pose_d_nuc={pose_np[3]:.2f}, "
           f"bd_angle={pose_np[4]:.1f}", flush=True)

    # Encode anchor SMILES
    vocab = model.base.vocabulary
    tok = model.base.tokenizer

    out_rows = []
    n_done = 0
    while n_done < args.n:
        batch_n = min(args.batch_size, args.n - n_done)
        # Randomize anchor per batch
        anchors = [randomize_smi(args.anchor_smi) if args.randomize_smiles
                    else args.anchor_smi for _ in range(batch_n)]
        # Tokenize and pad to common length
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((batch_n, L), dtype=np.int64)
        for j, s in enumerate(seqs):
            src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        # Replicate conditioning across batch
        res_emb = torch.from_numpy(np.tile(res_emb_np[None], (batch_n, 1, 1))).to(device)
        res_mask = torch.from_numpy(np.tile(res_mask_np[None], (batch_n, 1))).to(device)
        pose = torch.from_numpy(np.tile(pose_np[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial(
            src_t, src_mask, res_emb, res_mask, pose,
            max_length=args.max_length, temperature=args.temperature)
        for s, nll, anchor in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anchor,
                              "NLL": float(nll)})
        n_done += batch_n
        if n_done % (args.batch_size * 10) == 0 or n_done == args.n:
            print(f"  sampled {n_done}/{args.n}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_rows)} samples to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
