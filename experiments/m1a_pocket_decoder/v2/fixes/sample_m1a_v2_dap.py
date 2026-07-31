"""Sample 10k Mol1-anchored SMILES from the DAP-finetuned v2-cond model.

Same conditioning as cohort_A (Mol1 cofold pose + ZAP70 pocket, variant A).
Only difference: model weights come from m1a_v2_dap.ckpt.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
from m1a_v2_model import load_m1a_v2  # noqa: E402
from sample_m1a_v2_ablation import (  # noqa: E402
    randomize_smi, sample_cohort, MOL1_SMI,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "models/m1a_v2_dap.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "data/m1a_v2_dap/cohort_A_10k.csv"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sample_dap] device={device}", flush=True)

    model = load_m1a_v2(args.prior, device, ckpt_path=args.ckpt)
    # If the checkpoint also carries base.network state (produced by DAP training),
    # load it — the wrapper's load_state_dict only touches pocket_enc/pose_enc/buffers.
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "base_network_state" in sd:
        print(f"[sample_dap] loading base.network state from ckpt ({len(sd['base_network_state'])} keys)",
              flush=True)
        model.base.network.load_state_dict(sd["base_network_state"], strict=True)
    else:
        print("[sample_dap] no base_network_state in ckpt — base.network stays as prior", flush=True)
    model.eval()

    pose_d = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_norm = pose_d["pose_norm"].astype(np.float32)
    pocket_d = np.load(args.mol1_pocket_npz, allow_pickle=True)
    zap70_emb = pocket_d["residues_emb"][0]
    zap70_mask = pocket_d["residues_mask"][0]

    cache = np.load(args.cache, allow_pickle=True)
    R_max = cache["residues_emb"].shape[1]
    if zap70_emb.shape[0] < R_max:
        pad = np.zeros((R_max - zap70_emb.shape[0], zap70_emb.shape[1]), dtype=zap70_emb.dtype)
        zap70_emb = np.concatenate([zap70_emb, pad], axis=0)
        mask_pad = np.zeros((R_max - zap70_mask.shape[0],), dtype=bool)
        zap70_mask = np.concatenate([zap70_mask, mask_pad], axis=0)
    elif zap70_emb.shape[0] > R_max:
        zap70_emb = zap70_emb[:R_max]
        zap70_mask = zap70_mask[:R_max]
    print(f"[sample_dap] ZAP70 residues: {int(zap70_mask.sum())} (R_max={R_max})", flush=True)

    t0 = time.time()
    rows = sample_cohort(
        model, args.anchor_smi, zap70_emb, zap70_mask, mol1_pose_norm,
        n=args.n, batch_size=args.batch_size,
        max_length=args.max_length, temperature=args.temperature,
    )
    for r in rows:
        r["variant"] = "A_dap"
        r["variant_label"] = "Mol1 pose + ZAP70 (DAP RL)"
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    dt = time.time() - t0
    print(f"[sample_dap] wrote {len(df)} rows to {args.out_csv}  ({dt/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
