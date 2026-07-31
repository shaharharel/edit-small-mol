"""Sample from a C+D finetuned DrugFlow checkpoint with optional C-arm
token override at inference time.

Loads the .pt produced by `train_dc.py`, restores model + adapter, then runs
the same inpainting code path as `inpaint.py` so the warhead atom+bond
state pinning is also applied (you'd typically want BOTH the learned C-arm
bias AND the explicit warhead inpaint for max control).
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DRUGFLOW_ROOT = None
for cand in [Path.home() / "DrugFlow", Path("/home/shaharh_quris_ai/DrugFlow")]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DRUGFLOW_ROOT = cand
        break

import numpy as np
import torch
from rdkit import Chem
from Bio.PDB import PDBParser

from anchordiff.covind.covalent_token_v2_5 import (
    build_token_v2_5, TOKEN_DIM_V2_5,
)
from anchordiff.covind.cov_adapter_v2_5 import CovalentConditioningAdapterV25
from anchordiff.drugflow_covind.inpaint import (
    build_inpaint_spec_for_zap70,
    make_inpainted_sample_zt_given_zs,
    audit_warhead,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_ckpt", required=True, help=".pt from train_dc.py")
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ref_ligand", required=True)
    ap.add_argument("--warhead", required=True)
    ap.add_argument("--cys_chain", default="A")
    ap.add_argument("--cys_resi", type=int, default=346)
    ap.add_argument("--output", default="samples.sdf")
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--molecule_size", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datadir", default=str(Path.home() / "DrugFlow" / "src" / "default"))
    ap.add_argument("--inpaint", action="store_true",
                    help="also apply warhead atom+bond inpaint hook")
    ap.add_argument("--warhead_class", default="Michael Acceptor")
    ap.add_argument("--reaction_mech", default="Michael Addition")
    ap.add_argument("--strength_pos", type=float, default=0.7)
    ap.add_argument("--strength_h", type=float, default=5.0)
    ap.add_argument("--strength_e", type=float, default=5.0)
    args = ap.parse_args()

    from src.model.lightning import DrugFlow
    from src.data.dataset import ProcessedLigandPocketDataset
    from src.data.data_utils import TensorDict, process_raw_pair
    from torch.utils.data import DataLoader
    from functools import partial
    from src import utils as drugflow_utils
    drugflow_utils.set_deterministic(seed=42)
    drugflow_utils.disable_rdkit_logging()

    print(f"[sample] loading ft ckpt: {args.ft_ckpt}")
    ft = torch.load(args.ft_ckpt, map_location=args.device)
    # We need a base DrugFlow first
    base_ckpt = Path.home() / "DrugFlow" / "checkpoints" / "drugflow.ckpt"
    if not base_ckpt.exists():
        raise RuntimeError("base drugflow.ckpt not found; needed to init module structure")
    model = DrugFlow.load_from_checkpoint(base_ckpt, map_location=args.device, strict=False)
    if args.datadir:
        model.datadir = Path(args.datadir)
    model.setup(stage="generation")
    model.batch_size = model.eval_batch_size = args.batch_size
    # Load finetune state on top
    model.load_state_dict(ft["model_state"], strict=False)
    model.eval().to(args.device)
    model.T = args.n_steps

    # Build pocket batch
    pdb_model = PDBParser(QUIET=True).get_structure("", args.receptor)[0]
    rdmol = Chem.SDMolSupplier(str(args.ref_ligand))[0]
    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=args.receptor if model.dynamics.add_nma_feat else None,
    )
    ligand["name"] = "ligand"

    # Build C-arm adapter and apply token bias on pocket['one_hot'] BEFORE sampling
    pocket_oh_dim = pocket["one_hot"].shape[-1]
    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=pocket_oh_dim).to(args.device)
    adapter.load_state_dict(ft["adapter_state"])
    adapter.eval()

    # Build the token from the user's override (Michael Acceptor by default)
    # Read pocket residue context from receptor
    pocket_resnames = [a.get_parent().get_resname() for a in pdb_model.get_atoms()
                       if a.get_id() == "CA"]
    token = build_token_v2_5(
        d_canonical=1.85, theta_canonical=107.0,
        warhead_class=args.warhead_class,
        cys_context_residues=pocket_resnames,
        reaction_mechanism=args.reaction_mech,
    )
    token_t = torch.from_numpy(token).float().to(args.device)
    with torch.no_grad():
        bias = adapter(token_t)
    pocket["one_hot"] = pocket["one_hot"] + bias.unsqueeze(0).cpu()
    print(f"[sample] applied C-arm bias  mean|Δ|={float(bias.abs().mean()):.4f}")

    # Build batch
    dataset = [{"ligand": ligand, "pocket": pocket} for _ in range(args.batch_size)]
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
        pin_memory=True,
    )

    # Optionally install inpaint hook (warhead atom+bond pin)
    if args.inpaint:
        spec = build_inpaint_spec_for_zap70(
            warhead_sdf=args.warhead, receptor_pdb=args.receptor,
            cys_chain=args.cys_chain, cys_resi=args.cys_resi,
            n_samples=args.batch_size,
            strength_pos=args.strength_pos,
            strength_h=args.strength_h,
            strength_e=args.strength_e,
        )
        model.sample_zt_given_zs = make_inpainted_sample_zt_given_zs(model, spec)
        print(f"[sample] inpaint hook installed: canonical Cβ={spec.canonical_global_pos[0]}")

    # Sample
    all_mols = []
    while len(all_mols) < args.n_samples:
        for data in dataloader:
            new_data = {
                "ligand": TensorDict(**data["ligand"]).to(args.device),
                "pocket": TensorDict(**data["pocket"]).to(args.device),
            }
            rdmols, _, _ = model.sample(
                new_data, n_samples=1, timesteps=args.n_steps,
                num_nodes=args.molecule_size)
            for m in rdmols:
                if m is not None:
                    all_mols.append(m)
                if len(all_mols) >= args.n_samples:
                    break
            if len(all_mols) >= args.n_samples:
                break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    drugflow_utils.write_sdf_file(args.output, all_mols[:args.n_samples])
    print(f"[sample] wrote {len(all_mols[:args.n_samples])} mols to {args.output}")

    if args.inpaint:
        audit = audit_warhead(all_mols[:args.n_samples], spec)
        print(f"[sample] audit: {audit}")


if __name__ == "__main__":
    main()
