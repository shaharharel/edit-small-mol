"""M1a pose-sensitivity ablation: sample 5 cohorts (B/C/D/E/F).

Variant A (baseline) is already on disk at data/m1a_cohorts/cohort_mol1_zap70.csv.

Variants:
  B: NULL pose (zeros)        + ZAP70 pocket
  C: BD angle = 60° (other 5 pose dims kept at Mol1's values) + ZAP70 pocket
  D: BD angle = 150°          + ZAP70 pocket
  E: Mol1 pose                + EGFR (Cys797) pocket from CovInDB PDB 5GTY
  F: Mol1 pose                + Cathepsin B (Cys25) pocket from CovInDB PDB 1AEC (non-kinase)

The ZAP70 baseline pocket and Mol1 pose are taken from the same
boltz_zap70 row used by the existing baseline sampler (max-residue-coverage row).
EGFR/Cathepsin pockets are picked from the training-time covindb_pdb rows
already embedded in pocket_embeddings.npz, so no new ESM-2 inference is needed.

Output: data/m1a_ablation/cohort_{B,C,D,E,F}.csv (5000 mols each)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder"))
from m1a_model import load_m1a  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

# Pocket struct_ids selected from training data
ZAP70_STRUCT_ID = "row01006"   # baseline ZAP70 pick (max residue coverage)
EGFR_STRUCT_ID = "5GTY_A_816_1101"  # EGFR Cys797 covalent inhibitor
CATH_STRUCT_ID = "1AEC_A_E64_219"   # Cathepsin B Cys25 (non-kinase protease)


def randomize_smi(smi: str) -> str:
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def pick_pocket_by_struct_id(npz_d, struct_id: str):
    sids = npz_d["struct_ids"]
    idx = np.where(sids == struct_id)[0]
    if len(idx) == 0:
        raise RuntimeError(f"struct_id {struct_id!r} not found in pocket_embeddings npz")
    i = int(idx[0])
    return (npz_d["residues_emb"][i], npz_d["residues_mask"][i],
            npz_d["poses"][i], struct_id)


def sample_cohort(model, args, anchor_smi: str, res_emb_np: np.ndarray,
                  res_mask_np: np.ndarray, pose_np: np.ndarray,
                  n: int, batch_size: int, max_length: int, temperature: float):
    """Generate n samples from the m1a model with given conditioning."""
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer

    out_rows = []
    n_done = 0
    t0 = time.time()
    while n_done < n:
        batch_n = min(batch_size, n - n_done)
        anchors = [randomize_smi(anchor_smi) for _ in range(batch_n)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((batch_n, L), dtype=np.int64)
        for j, s in enumerate(seqs):
            src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        res_emb = torch.from_numpy(np.tile(res_emb_np[None], (batch_n, 1, 1))).to(device)
        res_mask = torch.from_numpy(np.tile(res_mask_np[None], (batch_n, 1))).to(device)
        pose = torch.from_numpy(np.tile(pose_np[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial(
            src_t, src_mask, res_emb, res_mask, pose,
            max_length=max_length, temperature=temperature)
        for s, nll, anchor in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anchor, "NLL": float(nll)})
        n_done += batch_n
        if n_done % (batch_size * 10) == 0 or n_done == n:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"    sampled {n_done}/{n}  {rate:.0f} mol/s  ETA {eta:.0f}s", flush=True)
    return out_rows


def progress_write(progress_path: Path, current_variant: str, phase: str):
    rec = {"current_variant": current_variant, "phase": phase,
            "timestamp": time.time()}
    progress_path.write_text(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "models/m1a_checkpoints/m1a_final.ckpt"))
    ap.add_argument("--triples_emb", default=str(PROJECT_ROOT / "data/m1a_triples/pocket_embeddings.npz"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "data/m1a_ablation"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--variants", default="B,C,D,E,F",
                     help="Comma list of which variants to sample")
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT / "data/m1a_ablation_progress.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model
    print("Loading prior+ckpt...", flush=True)
    model = load_m1a(args.prior, device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    model.eval()

    # Load pocket embeddings
    print(f"Loading pocket embeddings from {args.triples_emb}...", flush=True)
    npz_d = np.load(args.triples_emb, allow_pickle=True)

    # Get the three pockets we need
    zap_emb, zap_mask, zap_pose, _ = pick_pocket_by_struct_id(npz_d, ZAP70_STRUCT_ID)
    egfr_emb, egfr_mask, egfr_pose, _ = pick_pocket_by_struct_id(npz_d, EGFR_STRUCT_ID)
    cath_emb, cath_mask, cath_pose, _ = pick_pocket_by_struct_id(npz_d, CATH_STRUCT_ID)

    print(f"ZAP70 pocket: {int(zap_mask.sum())} residues, baseline pose={zap_pose}", flush=True)
    print(f"EGFR pocket:  {int(egfr_mask.sum())} residues, native pose={egfr_pose}", flush=True)
    print(f"Cathepsin pocket: {int(cath_mask.sum())} residues, native pose={cath_pose}", flush=True)

    # Build variant configs
    # MOL1_POSE = baseline ZAP70 pose (same as variant A)
    mol1_pose = zap_pose.copy()
    print(f"Mol1 pose (= baseline ZAP70 pose): {mol1_pose}", flush=True)

    # Variant B: null pose
    null_pose = np.zeros_like(mol1_pose)

    # Variant C: bd_angle = 60° (col 4)
    pose_c = mol1_pose.copy()
    pose_c[4] = 60.0

    # Variant D: bd_angle = 150°
    pose_d = mol1_pose.copy()
    pose_d[4] = 150.0

    variants = {
        "B": {"pose": null_pose,   "emb": zap_emb,  "mask": zap_mask,  "label": "null_pose+ZAP70"},
        "C": {"pose": pose_c,      "emb": zap_emb,  "mask": zap_mask,  "label": "bd60+ZAP70"},
        "D": {"pose": pose_d,      "emb": zap_emb,  "mask": zap_mask,  "label": "bd150+ZAP70"},
        "E": {"pose": mol1_pose,   "emb": egfr_emb, "mask": egfr_mask, "label": "mol1pose+EGFR(5GTY)"},
        "F": {"pose": mol1_pose,   "emb": cath_emb, "mask": cath_mask, "label": "mol1pose+Cathepsin(1AEC)"},
    }

    requested = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    for vid in requested:
        if vid not in variants:
            print(f"SKIP unknown variant {vid}", flush=True)
            continue
        cfg = variants[vid]
        out_csv = out_dir / f"cohort_{vid}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 1000:
            print(f"[skip] {out_csv} exists already (rerun w/ rm to redo)", flush=True)
            continue
        print(f"\n=== Variant {vid} ({cfg['label']}) ===", flush=True)
        print(f"   pose={cfg['pose']}", flush=True)
        progress_write(progress_path, vid, "sampling")

        rows = sample_cohort(
            model, args, args.anchor_smi,
            cfg["emb"], cfg["mask"], cfg["pose"],
            n=args.n, batch_size=args.batch_size,
            max_length=args.max_length, temperature=args.temperature,
        )
        # Add metadata
        for r in rows:
            r["variant"] = vid
            r["variant_label"] = cfg["label"]
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"   wrote {len(df)} rows to {out_csv}", flush=True)
        progress_write(progress_path, vid, "wrote_csv")

    progress_write(progress_path, None, "all_variants_sampled")
    print("\nALL VARIANTS SAMPLED.", flush=True)


if __name__ == "__main__":
    main()
