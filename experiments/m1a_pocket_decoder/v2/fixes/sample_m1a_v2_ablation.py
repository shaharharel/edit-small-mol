"""M1a v2 pose-sensitivity ablation: sample 6 (or 7) cohorts.

Variants (raw, unnormalized pose values; the model's persisted normalizer is
applied inside the script so the network sees z-scored inputs):

  A : Mol1 cofold pose + ZAP70 pocket (from fresh Boltz-2 cofold)
  B : NULL pose (all zeros, raw)              + ZAP70 pocket
  C : Mol1 pose with BD-angle dim set to 60°  + ZAP70 pocket
  D : Mol1 pose with BD-angle dim set to 150° + ZAP70 pocket
  E : Mol1 pose                                + EGFR Cys797 pocket (PDB 5GTY)
  F : Mol1 pose                                + Cathepsin Cys25 pocket (PDB 1AEC)
  G : Mol1 pose with planar-dihedral dim set to 90° + ZAP70 pocket  (NEW for v2)

EGFR / Cathepsin pockets are taken from the v2 training corpus by struct_id,
the same way v1 did. The Mol1 pose is loaded from
data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz (Phase 2 fresh cofold) and the
ZAP70 pocket from data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz.

Output: data/m1a_ablation_v2/cohort_{A,B,C,D,E,F,G}.csv (5000 rows each).
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
from m1a_v2_model import load_m1a_v2  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

# struct_ids that point to EGFR / Cathepsin pockets in the v2 corpus
EGFR_STRUCT_ID_CANDIDATES = [
    "5GTY_A_816_1101", "5gty_A_816_1101",
    # any 5GTY HET binding Cys797
]
CATH_STRUCT_ID_CANDIDATES = [
    "1AEC_A_E64_219", "1aec_A_E64_219",
]


def randomize_smi(smi: str) -> str:
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def pick_pocket_by_struct_id(cache_npz: dict, candidates: list[str]):
    """Look up a row in the v2 cache by struct_id, return (residues_emb, mask,
    pose_norm, struct_id_used)."""
    sids = cache_npz["struct_ids"]
    poses = cache_npz["poses"]   # z-scored
    row_seq_idx = cache_npz["row_seq_idx"]
    res_emb = cache_npz["residues_emb"]
    res_mask = cache_npz["residues_mask"]
    for sid in candidates:
        idx = np.where(sids == sid)[0]
        if len(idx) > 0:
            i = int(idx[0])
            seq_idx = int(row_seq_idx[i])
            return (res_emb[seq_idx], res_mask[seq_idx], poses[i], sid)
    # Fallback: search by prefix
    pdb_prefix = candidates[0].split("_")[0].upper()
    for j, s in enumerate(sids):
        if isinstance(s, (str, np.str_)) and str(s).startswith(pdb_prefix):
            seq_idx = int(row_seq_idx[j])
            return (res_emb[seq_idx], res_mask[seq_idx], poses[j], str(s))
    raise RuntimeError(f"No struct_id matching {candidates} in cache")


def sample_cohort(model, anchor_smi: str, res_emb_np: np.ndarray,
                    res_mask_np: np.ndarray, pose_norm_np: np.ndarray,
                    n: int, batch_size: int, max_length: int,
                    temperature: float):
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
        pose = torch.from_numpy(np.tile(pose_norm_np[None], (batch_n, 1))).to(device)
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
            print(f"    sampled {n_done}/{n}  {rate:.0f} mol/s  ETA {eta:.0f}s",
                  flush=True)
    return out_rows


def write_progress(progress_path: str, **fields):
    rec = {"timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    Path(progress_path).write_text(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/m1a_v2_ablation"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--variants", default="A,B,C,D,E,F,G")
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/m1a_v2_progress.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load v2 model (with persisted normalizer)
    print(f"Loading prior+ckpt: {args.ckpt}", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.ckpt)
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"Model pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)
    model.eval()

    # Load v2 cache for EGFR/Cathepsin pocket lookup
    print(f"Loading v2 cache for EGFR/Cathepsin pocket lookup: {args.cache}",
          flush=True)
    cache = np.load(args.cache, allow_pickle=True)

    # Load Mol1 cofold pose + pocket from Phase 2
    print(f"Loading Mol1 cofold pose: {args.mol1_pose_npz}", flush=True)
    pose_d = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_unnorm = pose_d["pose_unnorm"].astype(np.float32)
    mol1_pose_norm = pose_d["pose_norm"].astype(np.float32)
    print(f"Mol1 pose unnorm: {mol1_pose_unnorm}", flush=True)
    print(f"Mol1 pose norm:   {mol1_pose_norm}", flush=True)

    print(f"Loading Mol1 pocket ESM: {args.mol1_pocket_npz}", flush=True)
    pocket_d = np.load(args.mol1_pocket_npz, allow_pickle=True)
    zap70_emb = pocket_d["residues_emb"][0]
    zap70_mask = pocket_d["residues_mask"][0]
    # The model expects fixed R_max; check the v2 cache dimensions and pad
    R_max = cache["residues_emb"].shape[1]
    if zap70_emb.shape[0] < R_max:
        pad = np.zeros((R_max - zap70_emb.shape[0], zap70_emb.shape[1]),
                        dtype=zap70_emb.dtype)
        zap70_emb = np.concatenate([zap70_emb, pad], axis=0)
        mask_pad = np.zeros((R_max - zap70_mask.shape[0],), dtype=bool)
        zap70_mask = np.concatenate([zap70_mask, mask_pad], axis=0)
    elif zap70_emb.shape[0] > R_max:
        zap70_emb = zap70_emb[:R_max]
        zap70_mask = zap70_mask[:R_max]
    print(f"ZAP70 pocket from fresh cofold: {int(zap70_mask.sum())} real residues "
          f"(padded to R_max={R_max})", flush=True)

    # EGFR / Cathepsin pockets
    egfr_emb, egfr_mask, _, egfr_sid = pick_pocket_by_struct_id(
        cache, EGFR_STRUCT_ID_CANDIDATES)
    cath_emb, cath_mask, _, cath_sid = pick_pocket_by_struct_id(
        cache, CATH_STRUCT_ID_CANDIDATES)
    print(f"EGFR pocket from {egfr_sid}: {int(egfr_mask.sum())} residues",
          flush=True)
    print(f"Cathepsin pocket from {cath_sid}: {int(cath_mask.sum())} residues",
          flush=True)

    # Build perturbed Mol1 poses (operate in UNNORMALIZED space then z-score)
    def norm(p):
        return (p - pose_mean) / np.maximum(pose_std, 1e-6)

    mol1_pose = mol1_pose_unnorm.copy()

    pose_b_raw = np.zeros_like(mol1_pose)   # NULL pose (raw zeros)
    pose_b_norm = norm(pose_b_raw)

    # v3 invariant pose schema: dim0=d_b_nuc, dim1=bd_angle_deg, dim2=planar_dihedral_deg
    pose_c_raw = mol1_pose.copy(); pose_c_raw[1] = 60.0   # BD angle -> 60°
    pose_d_raw = mol1_pose.copy(); pose_d_raw[1] = 150.0  # BD angle -> 150°
    pose_g_raw = mol1_pose.copy(); pose_g_raw[2] = 90.0   # planar dihedral -> 90°

    variants = {
        "A": {"label": "Mol1 cofold pose + ZAP70",
              "pose": mol1_pose_norm, "emb": zap70_emb, "mask": zap70_mask},
        "B": {"label": "NULL pose + ZAP70",
              "pose": pose_b_norm.astype(np.float32),
              "emb": zap70_emb, "mask": zap70_mask},
        "C": {"label": "BD=60° + ZAP70",
              "pose": norm(pose_c_raw).astype(np.float32),
              "emb": zap70_emb, "mask": zap70_mask},
        "D": {"label": "BD=150° + ZAP70",
              "pose": norm(pose_d_raw).astype(np.float32),
              "emb": zap70_emb, "mask": zap70_mask},
        "E": {"label": "Mol1 pose + EGFR(5GTY)",
              "pose": mol1_pose_norm, "emb": egfr_emb, "mask": egfr_mask},
        "F": {"label": "Mol1 pose + Cathepsin(1AEC)",
              "pose": mol1_pose_norm, "emb": cath_emb, "mask": cath_mask},
        "G": {"label": "Planar-dihedral=90° + ZAP70 (NEW v2)",
              "pose": norm(pose_g_raw).astype(np.float32),
              "emb": zap70_emb, "mask": zap70_mask},
    }

    requested = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    for vid in requested:
        if vid not in variants:
            print(f"SKIP unknown variant {vid}", flush=True); continue
        cfg = variants[vid]
        out_csv = out_dir / f"cohort_{vid}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 1000:
            print(f"[skip] {out_csv} exists already", flush=True); continue
        print(f"\n=== Variant {vid} ({cfg['label']}) ===", flush=True)
        print(f"  pose_norm = {cfg['pose']}", flush=True)
        write_progress(args.progress_path,
                       phase="phase3_sampling", current_variant=vid)
        rows = sample_cohort(
            model, args.anchor_smi, cfg["emb"], cfg["mask"], cfg["pose"],
            n=args.n, batch_size=args.batch_size,
            max_length=args.max_length, temperature=args.temperature,
        )
        for r in rows:
            r["variant"] = vid; r["variant_label"] = cfg["label"]
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"  wrote {len(df)} rows to {out_csv}", flush=True)

    write_progress(args.progress_path, phase="phase3_done")
    print("\nALL VARIANTS SAMPLED.", flush=True)


if __name__ == "__main__":
    main()
