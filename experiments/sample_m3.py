"""Sample from an M3 (D-soft) finetuned checkpoint.

Our train_d_soft.py saves checkpoints as {"model": state_dict, ...} which is
not in PyTorch-Lightning format. This script loads the PRETRAINED DiffSBDD via
LigandPocketDDPM.load_from_checkpoint, then overwrites the model weights from
our M3 checkpoint, then runs sample_given_pocket.

Usage:
  python sample_m3.py --m3_ckpt ~/runs/m3_jitter_med/final.ckpt \
                      --pdbfile <receptor.pdb> --ref_ligand <ref.sdf> \
                      --outfile samples.sdf --n_samples 100
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path
import torch
import numpy as np

# DiffSBDD repo path
DIFFSBDD = Path.home() / "DiffSBDD"
sys.path.insert(0, str(DIFFSBDD))

from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE
from Bio.PDB import PDBParser
from rdkit import Chem


def parse_receptor_to_pocket(pdb_path: Path, ref_ligand_sdf: Path):
    """Build pocket atom set: protein atoms within 8 Å of any ref ligand atom."""
    # Read ref ligand to get its atom positions
    ref_mol = Chem.SDMolSupplier(str(ref_ligand_sdf), sanitize=False)[0]
    conf = ref_mol.GetConformer()
    ref_pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in range(ref_mol.GetNumAtoms())])
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", str(pdb_path))[0]
    pocket_atoms = []
    for chain in struct:
        for res in chain:
            if res.id[0].strip():  # skip hetatm/water
                continue
            for atom in res.get_atoms():
                if atom.element == "H":
                    continue
                ap = np.array(atom.get_coord())
                if np.min(np.linalg.norm(ref_pos - ap, axis=1)) < 8.0:
                    pocket_atoms.append({
                        "x": ap, "element": atom.element, "name": atom.name,
                        "resname": res.resname, "resi": res.id[1],
                    })
    return pocket_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", type=str,
                    default=str(DIFFSBDD / "checkpoints" / "crossdocked_fullatom_cond.ckpt"))
    ap.add_argument("--m3_ckpt", type=str, required=True,
                    help="Path to M3 final.ckpt (saved via torch.save({'model':...}))")
    ap.add_argument("--pdbfile", type=str, required=True)
    ap.add_argument("--ref_ligand", type=str, required=True)
    ap.add_argument("--outfile", type=str, required=True)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--num_nodes_lig", type=int, default=30)
    ap.add_argument("--sanitize", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load pretrained
    print(f"[1] loading pretrained: {args.pretrained_ckpt}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LigandPocketDDPM.load_from_checkpoint(args.pretrained_ckpt, map_location=device)

    # 2. Overlay our M3 trained weights
    print(f"[2] loading M3 trained weights: {args.m3_ckpt}")
    m3 = torch.load(args.m3_ckpt, map_location=device, weights_only=False)
    sd_key = "model" if "model" in m3 else "state_dict"
    missing, unexpected = model.load_state_dict(m3[sd_key], strict=False)
    print(f"    loaded; missing={len(missing)}  unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"    sample missing keys: {missing[:3]}")
    model.to(device).eval()

    # 3. Reuse DiffSBDD's generate_ligands.py logic via the model's sample method.
    # Easiest: dynamically call generate_ligands functions. Instead, replicate the
    # essential bits inline (parse pocket, set up batch, sample).
    print(f"[3] sampling {args.n_samples} mols ...")
    t0 = time.time()
    # We'll just call generate_ligands as a subprocess on a TEMPORARY PL-style ckpt
    # by re-saving our weights into a fake PL checkpoint structure.
    fake_pl = {
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": "2.0.0",
        "hyper_parameters": getattr(model, "hparams", {}),
        "epoch": 0, "global_step": 0,
    }
    tmp_ckpt = Path(args.m3_ckpt).with_suffix(".pl.ckpt")
    torch.save(fake_pl, tmp_ckpt)
    print(f"    wrote temp PL-format ckpt: {tmp_ckpt}")

    # Now invoke generate_ligands.py via subprocess
    import subprocess
    cmd = [
        "python", str(DIFFSBDD / "generate_ligands.py"),
        str(tmp_ckpt),
        "--pdbfile", args.pdbfile,
        "--ref_ligand", args.ref_ligand,
        "--outfile", args.outfile,
        "--n_samples", str(args.n_samples),
        "--batch_size", str(args.batch_size),
    ]
    if args.sanitize:
        cmd.append("--sanitize")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(f"FAILED: {r.stderr[-2000:]}")
        sys.exit(1)
    print(f"[4] done in {time.time()-t0:.0f}s — output: {args.outfile}")


if __name__ == "__main__":
    main()
