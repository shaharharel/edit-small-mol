"""Strategy 1 — Lingo3DMol L1 kinase fine-tune launcher.

Continues training from `data/covlingo_full_v1/ckpt_phase2_dev.pt` (the "L1 FT v1"
ckpt) on a subset of CovInDB filtered to kinase Cys-bound covalent complexes
(204 complexes intersecting the kinase PDB ID list).

Goal: bias generation toward kinase-pharmacophore chemistry.

Note: 204 complexes is the realistic upper bound for kinase-Cys covalent
complexes in CovInDB; the task's "5-10K" target is unattainable for the
3D-pocket-conditioned Lingo L1 training format (which requires crystal
structures, not just SMILES). See report for rationale.

Output: data/lingo3dmol_L1_kinase_FT/ckpt_phase2_dev.pt

Usage (on V100):
    conda run -n lingo3dmol python -u \\
        experiments/run_lingo3dmol_l1_kinase_ft.py \\
        --device cuda --epochs 8 --batch_size 16
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--complex_csv",
                   default="data/lingo3dmol_kinase_FT/kinase_covindb_complexes.csv")
    p.add_argument("--pdb_dir",
                   default="data/covbinder/raw_covindb2/PDB")
    p.add_argument("--pretrained_ckpt",
                   default="data/covlingo_full_v1/ckpt_phase2_dev.pt")
    p.add_argument("--out_dir", default="data/lingo3dmol_L1_kinase_FT")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--freeze_encoder_layers", type=int, default=3)
    p.add_argument("--pocket_radius", type=float, default=15.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u",
        "experiments/run_lingo3dmol_l1_train.py",
        "--complex_csv", args.complex_csv,
        "--pocket_pdb_dir", args.pdb_dir,
        "--pretrained_ckpt", args.pretrained_ckpt,
        "--out_dir", args.out_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--freeze_encoder_layers", str(args.freeze_encoder_layers),
        "--pocket_radius", str(args.pocket_radius),
        "--device", args.device,
        "--cache_dir", str(Path(args.out_dir) / "cache"),
    ]
    print("[launcher] cmd:")
    print(" \\\n  ".join(cmd))
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
