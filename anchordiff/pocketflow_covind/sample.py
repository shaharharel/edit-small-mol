"""Generate from a finetuned PocketFlow checkpoint with the warhead seed.

This is `inpaint.py` re-wrapped to accept a finetuned ckpt path. The C-arm
adapter weights, if present in the ckpt under key 'adapter', are loaded and
applied. If not present (smoke-only finetune that didn't save adapter), falls
back to vanilla seeded inpainting.

Usage:
  python ~/edit-small-mol/anchordiff/pocketflow_covind/sample.py \
      --pocket ~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb \
      --warhead_sdf ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
      --ckpt <finetuned.ckpt> \
      -n 25 --name zap70_dc_sample
"""
from __future__ import annotations
import os
import sys
import argparse
import torch

POCKETFLOW_ROOT = os.path.expanduser("~/PocketFlow")
sys.path.insert(0, POCKETFLOW_ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from inpaint import SeededGenerate  # type: ignore
from warhead_seed import build_seed_or_empty  # type: ignore
from cov_token import build_token, CovTokenAdapter  # type: ignore

from pocket_flow import PocketFlow  # type: ignore
from pocket_flow.utils import (  # type: ignore
    Protein, ComplexData, torchify_dict,
    FeaturizeProteinAtom, FeaturizeLigandAtom, AtomComposer,
    RefineData, LigandCountNeighbors, mask_node,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket", required=True)
    ap.add_argument("--warhead_sdf", required=True)
    ap.add_argument("--ckpt", required=True, help="path to finetuned ckpt")
    ap.add_argument("--adapter_state", default=None,
                    help="optional separate adapter state_dict .pt")
    ap.add_argument("-n", "--num_gen", type=int, default=25)
    ap.add_argument("--name", default="zap70_dc_sample")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--atom_temperature", type=float, default=1.0)
    ap.add_argument("--bond_temperature", type=float, default=1.0)
    ap.add_argument("--max_atom_num", type=int, default=40)
    ap.add_argument("--focus_threshold", type=float, default=0.5)
    ap.add_argument("--choose_max", default="1")
    ap.add_argument("--min_dist_inter_mol", type=float, default=3.0)
    ap.add_argument("--bond_length_range", default="(1.0,2.0)")
    ap.add_argument("--max_double_in_6ring", type=int, default=0)
    ap.add_argument("--root_path", default=os.path.expanduser("~/runs/pocketflow_dc_sample"))
    args = ap.parse_args()

    choose_max = args.choose_max.lower() in {"1", "true", "yes", "t", "y"}
    if isinstance(args.bond_length_range, str):
        args.bond_length_range = eval(args.bond_length_range)

    pro_dict = Protein(args.pocket).get_atom_dict(removeHs=True, get_surf=True)
    lig_dict, cb_idx = build_seed_or_empty(args.warhead_sdf)
    n_seed = len(lig_dict.element)
    print(f"[sample] seeded with n={n_seed} atoms, Cβ idx={cb_idx}")

    data = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lig_dict),
    )
    pf = FeaturizeProteinAtom()
    lf = FeaturizeLigandAtom(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
    ac = AtomComposer(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)
    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = pf(data)
    data = lf(data)
    context_idx = torch.arange(data.ligand_pos.size(0))  # all seed atoms in context
    data = mask_node(data, context_idx, torch.empty([0], dtype=torch.long),
                     num_atom_type=9, y_pos_std=0.)
    data = ac.run(data)

    print(f"[sample] loading model: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = ckpt["config"]
    model = PocketFlow(config).to(args.device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # tolerate the 'base.' prefix from CovConditionedPocketFlow wrapping
    state = {k.removeprefix("base."): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[sample] partial load: {e}")

    # Apply C-arm bias to cpx_feature using the trained adapter (if available)
    adapter = None
    if args.adapter_state and os.path.exists(args.adapter_state):
        a_state = torch.load(args.adapter_state, map_location=args.device)
        adapter = CovTokenAdapter(feat_dim=config.ligand_atom_feature_dim).to(args.device)
        adapter.load_state_dict(a_state)
        adapter.eval()
        tok = torch.from_numpy(build_token(1.85, 107.0, "Michael Acceptor",
                                           ["CYS"], "Michael Addition")).float().to(args.device)
        with torch.no_grad():
            bias = adapter(tok)
        lig_idx = data.idx_ligand_ctx_in_cpx
        if lig_idx.numel() > 0:
            K = min(bias.shape[0], data.cpx_feature.shape[1])
            feat = data.cpx_feature.float().clone().to(args.device)
            lig_idx_dev = lig_idx.to(args.device)
            feat[lig_idx_dev, :K] = feat[lig_idx_dev, :K] + bias[:K].unsqueeze(0)
            data.cpx_feature = feat.cpu()  # Generate.run moves to device anyway
        print(f"[sample] applied C-arm bias |bias|_mean={bias.abs().mean().item():.4f}")

    gen = SeededGenerate(
        model, ac.run,
        temperature=[args.atom_temperature, args.bond_temperature],
        atom_type_map=[6, 7, 8, 9, 15, 16, 17, 35, 53],
        num_bond_type=4, max_atom_num=args.max_atom_num,
        focus_threshold=args.focus_threshold,
        max_double_in_6ring=args.max_double_in_6ring,
        min_dist_inter_mol=args.min_dist_inter_mol,
        bond_length_range=args.bond_length_range,
        choose_max=choose_max, device=args.device, n_seed_atoms=n_seed,
    )
    gen.generate(data, num_gen=args.num_gen, rec_name=args.name,
                 with_print=True, root_path=args.root_path)
    print(f"[sample] done — out: {gen.out_dir}")


if __name__ == "__main__":
    main()
