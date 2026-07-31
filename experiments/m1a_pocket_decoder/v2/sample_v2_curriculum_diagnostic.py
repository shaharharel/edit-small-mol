"""Steering diagnostic (Phase 1e) — sample from RETRAINED v2 with fixed
anchor + ZAP70 pocket + 4 pose conditions.

Cohorts:
  theta_90  : bd_angle_deg = 90°   (all other pose dims from Mol1 cofold)
  theta_105 : bd_angle_deg = 105°  (ideal Burgi-Dunitz)
  theta_130 : bd_angle_deg = 130°  (bent-away)
  null_pose : all pose dims = 0 (raw), then z-scored (control)

Output CSVs: data/paper_pair_training/v2_curriculum/samples_{cohort}.csv
Each row: SMILES, Input_SMILES, NLL, cohort, theta_target, pose_norm_json

After sampling, Boltz-cofold all 800 mols and re-parse bd_angle_deg — the
KEY win metric is that the OUTPUT theta distributions DIFFER between
cohorts (KS-test p<0.01 for at least one pair).
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

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT
LOCAL_REINVENT = Path("/Users/shaharharel/Documents/github/REINVENT4")
A100_REINVENT = Path("/home/shaharh_quris_ai/REINVENT4")
REINVENT4_ROOT = LOCAL_REINVENT if LOCAL_REINVENT.exists() else A100_REINVENT
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

FIXES_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
if str(FIXES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXES_DIR))

from m1a_v2_model import load_m1a_v2  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def randomize_smi(smi: str) -> str:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def sample_cohort(model, anchor_smi: str, res_emb_np, res_mask_np, pose_norm_np,
                  n: int, batch_size: int, max_length: int, temperature: float):
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
        for s, nll, anc in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anc, "NLL": float(nll)})
        n_done += batch_n
        if n_done % (batch_size * 5) == 0 or n_done == n:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(f"    {n_done}/{n}  ({rate:.0f} mol/s)", flush=True)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                    "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT /
                    "models/v2_curriculum/best.chkpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                    "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                    "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/v2_curriculum/steering_samples"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--use_cache_pocket", type=int, default=1,
                    help="If 1, use ZAP70 pocket_idx=5 from ESM cache (works "
                         "even without mol1_zap70_esm.npz).")
    ap.add_argument("--cache_pocket_idx", type=int, default=5,
                    help="Which pocket_idx in the ESM cache to use for ZAP70. "
                         "5 is the largest ZAP70 boltz pocket (449 rows).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}", flush=True)

    # ---- Load model (with persisted normalizer inside ckpt) ----
    print(f"[diag] loading model: prior={args.prior} ckpt={args.ckpt}", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.ckpt)
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"[diag] pose_mean={pose_mean} pose_std={pose_std}", flush=True)
    model.eval()

    # ---- Load ZAP70 pocket ----
    if args.use_cache_pocket and Path(args.cache).exists():
        cache = np.load(args.cache, allow_pickle=True)
        row_seq_idx = cache["row_seq_idx"]
        # Find first row with our target pocket_idx
        rows = np.where(row_seq_idx == args.cache_pocket_idx)[0]
        if len(rows) == 0:
            raise SystemExit(f"[diag] pocket_idx={args.cache_pocket_idx} not in cache")
        seq_idx = args.cache_pocket_idx
        zap70_emb = cache["residues_emb"][seq_idx]
        zap70_mask = cache["residues_mask"][seq_idx]
        mol1_pose_unnorm = cache["poses_unnorm"][rows[0]].astype(np.float32)
        mol1_pose_norm = cache["poses"][rows[0]].astype(np.float32)
        print(f"[diag] using pocket_idx={seq_idx} from ESM cache "
              f"({int(zap70_mask.sum())} residues, {len(rows)} rows total)",
              flush=True)
        print(f"[diag] anchor pose_unnorm = {mol1_pose_unnorm}", flush=True)
    else:
        # Fallback: use mol1_zap70 external file
        pose_d = np.load(args.mol1_pose_npz, allow_pickle=True)
        mol1_pose_unnorm = pose_d["pose_unnorm"].astype(np.float32)
        mol1_pose_norm = pose_d["pose_norm"].astype(np.float32)
        pocket_d = np.load(args.mol1_pocket_npz, allow_pickle=True)
        zap70_emb = pocket_d["residues_emb"][0]
        zap70_mask = pocket_d["residues_mask"][0]
        # Pad to model's R_max if needed
        cache = np.load(args.cache, allow_pickle=True)
        R_max = cache["residues_emb"].shape[1]
        if zap70_emb.shape[0] < R_max:
            pad_e = np.zeros((R_max - zap70_emb.shape[0], zap70_emb.shape[1]), dtype=zap70_emb.dtype)
            zap70_emb = np.concatenate([zap70_emb, pad_e], axis=0)
            pad_m = np.zeros((R_max - zap70_mask.shape[0],), dtype=bool)
            zap70_mask = np.concatenate([zap70_mask, pad_m], axis=0)
        elif zap70_emb.shape[0] > R_max:
            zap70_emb = zap70_emb[:R_max]
            zap70_mask = zap70_mask[:R_max]
        print(f"[diag] using external mol1_zap70 pocket (R={int(zap70_mask.sum())})",
              flush=True)

    # ---- Build 4 pose conditions ----
    def norm(p):
        return (p - pose_mean) / np.maximum(pose_std, 1e-6)

    base = mol1_pose_unnorm.copy()  # [d_b_nuc, bd_angle_deg, planar_dihedral_deg]

    pose_90 = base.copy(); pose_90[1] = 90.0
    pose_105 = base.copy(); pose_105[1] = 105.0
    pose_130 = base.copy(); pose_130[1] = 130.0
    pose_null = np.zeros_like(base)

    variants = {
        "theta_90":  {"raw": pose_90,  "norm": norm(pose_90).astype(np.float32)},
        "theta_105": {"raw": pose_105, "norm": norm(pose_105).astype(np.float32)},
        "theta_130": {"raw": pose_130, "norm": norm(pose_130).astype(np.float32)},
        "null_pose": {"raw": pose_null, "norm": norm(pose_null).astype(np.float32)},
    }

    all_dfs = []
    for name, cfg in variants.items():
        out_csv = out_dir / f"samples_{name}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 500:
            print(f"[diag] skipping (exists): {out_csv}", flush=True)
            df = pd.read_csv(out_csv)
            all_dfs.append(df)
            continue
        print(f"\n[diag] === Cohort {name} ===", flush=True)
        print(f"[diag] pose_raw={cfg['raw']} pose_norm={cfg['norm']}", flush=True)
        rows = sample_cohort(model, args.anchor_smi, zap70_emb, zap70_mask,
                             cfg["norm"], n=args.n, batch_size=args.batch_size,
                             max_length=args.max_length,
                             temperature=args.temperature)
        for r in rows:
            r["cohort"] = name
            r["theta_target"] = float(cfg["raw"][1])
            r["pose_raw"] = json.dumps(cfg["raw"].tolist())
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"[diag] wrote {len(df)} rows -> {out_csv}", flush=True)
        all_dfs.append(df)

    # Merged CSV
    merged = pd.concat(all_dfs, ignore_index=True)
    merged_path = out_dir / "samples_all.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[diag] merged {len(merged)} rows -> {merged_path}", flush=True)


if __name__ == "__main__":
    main()
