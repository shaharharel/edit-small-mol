"""Sample from the fine-tuned C+D DiffSBDD.

Re-uses DiffSBDD's own inpaint pipeline (utils.get_pocket_from_ligand,
prepare_from_sdf_files, model.ddpm.inpaint) instead of re-implementing it,
so the only NEW behaviour vs vanilla inpaint.py is:
  1. Override the base CrossDocked2020-pretrained weights with the
     fine-tuned model_state from dc_flagship_epN.pt.
  2. Inject the C-arm adapter bias into pocket['one_hot'] before sampling.

CLI:
  python -m anchordiff.covind.sample_dc \
    --ckpt   ~/results/covind/dc_flagship_ep30.pt \
    --pdbfile ~/anchordiff/pockets/zap70_cys346/receptor.pdb \
    --warhead_sdf ~/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
    --warhead_class "Michael Acceptor" \
    --cys_chain A --cys_resi 346 \
    --n_samples 100 --batch_size 10 \
    --outfile ~/results/9h_run/phase4/zap70_finetuned.sdf
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for cand in [Path.home() / "DiffSBDD", Path("/home/shaharh_quris_ai/DiffSBDD")]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DIFFSBDD_ROOT = cand
        break
else:
    DIFFSBDD_ROOT = None

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, Polypeptide
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from anchordiff.covind.cov_adapter import (
    CovalentConditioningAdapter, inject_into_pocket_oh,
)
from anchordiff.covind.covalent_token import build_token, TOKEN_DIM
from anchordiff.covind.covalent_token_v2_5 import build_token_v2_5, TOKEN_DIM_V2_5
from anchordiff.covind.dataset import F_A
# Variant-aware adapter (used for v2.5/J3/J4/J5 ablation ckpts)
try:
    from anchordiff.covind.cov_adapter_ablation import (
        AblationAdapter, inject_into_pocket_oh_spatial,
    )
    _ABLATION_ADAPTER_AVAILABLE = True
except ImportError:
    _ABLATION_ADAPTER_AVAILABLE = False

# Import WARHEAD_GEOM from curate.py to avoid drift (2026-05-13 fix).
# Previously this module had its own (incomplete) dict; missing classes
# silently fell back to Michael Acceptor.
from anchordiff.covind.curate import WARHEAD_GEOM


def collect_cys_context_resnames(pocket_residues):
    return [r.get_resname() for r in pocket_residues]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pdbfile", type=Path, required=True)
    ap.add_argument("--cys_chain", type=str, default="A")
    ap.add_argument("--cys_resi",  type=int, required=True)
    ap.add_argument("--warhead_sdf", type=Path, required=True)
    ap.add_argument("--reference_ligand_sdf", type=Path, default=None,
                    help="Reference ligand pose (e.g. docked parent) used to "
                         "BUILD THE POCKET at inference. Without this, the "
                         "5-atom warhead stub yields only ~9 residues — way "
                         "smaller than the 22-58-residue pockets the model "
                         "saw at training. (B2 fix May-2026 QA.) Falls back "
                         "to warhead_sdf with wider cutoff if not provided.")
    ap.add_argument("--pocket_cutoff", type=float, default=8.0,
                    help="Å cutoff for pocket residue selection. With "
                         "reference_ligand_sdf: keep at 8.0. Without "
                         "(warhead_sdf only): bump to ~15-20 to compensate.")
    ap.add_argument("--warhead_class", type=str, default="Michael Acceptor")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--add_n_nodes", type=int, default=25)
    ap.add_argument("--resamplings", type=int, default=10)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--no_carm", action="store_true",
                    help="skip C-arm adapter injection (D-arm-only ablation)")
    ap.add_argument("--diffsbdd_base_ckpt", type=str,
                    default=str(Path.home() / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"))
    args = ap.parse_args()

    if DIFFSBDD_ROOT is None:
        raise RuntimeError("DiffSBDD repo not found on this host")
    import utils as dsutils
    from lightning_modules import LigandPocketDDPM
    from analysis.molecule_builder import build_molecule, process_molecule
    from inpaint import prepare_from_sdf_files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load base + fine-tuned weights
    print(f"Loading base DiffSBDD from {args.diffsbdd_base_ckpt}")
    model = LigandPocketDDPM.load_from_checkpoint(args.diffsbdd_base_ckpt,
                                                  map_location=device).to(device)
    print(f"Loading fine-tuned weights from {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"  model+adapter loaded from epoch {ck.get('epoch', '?')}")

    # 2. Detect variant from ckpt + load adapter (unless --no_carm)
    # variant = v2 (default) | v2_5 | j4a | j4b | j4c | j3 | j5
    # Detection order:
    #   (a) ck["variant"] (set by train_dc_ablation)
    #   (b) ck["token_variant"] == "v2_5" (set by train_dc_v2_5)
    #   (c) adapter_state shape: "log_scale" + "mlp.0.weight" → v2.5 architecture
    variant = ck.get("variant", "v2")
    if variant == "v2":
        if ck.get("token_variant") == "v2_5":
            variant = "v2_5"
        elif "adapter_state" in ck:
            ak = set(ck["adapter_state"].keys())
            if "log_scale" in ak and "mlp.0.weight" in ak:
                variant = "v2_5"
    saved_args = ck.get("args", {}) if variant != "v2" else {}
    # Token dim per variant
    if variant in ("v2_5",) or saved_args.get("token_variant") == "v2_5":
        token_dim_for_variant = TOKEN_DIM_V2_5
        token_builder = build_token_v2_5
    else:
        token_dim_for_variant = TOKEN_DIM
        token_builder = build_token
    use_spatial_injection = saved_args.get("spatial_decay_lambda", 0.0) > 0
    print(f"  detected variant: {variant!r}  token_dim={token_dim_for_variant}  "
          f"spatial={'on' if use_spatial_injection else 'off'}")

    adapter = None
    if not args.no_carm:
        if variant in ("j4a", "j4b", "j4c", "j3", "j5") and _ABLATION_ADAPTER_AVAILABLE:
            adapter = AblationAdapter(
                token_dim=token_dim_for_variant,
                feat_dim=F_A,
                scale_init=saved_args.get("scale_init", 0.25),
                scale_learnable=saved_args.get("scale_learnable", False),
                hidden_dim=saved_args.get("hidden_dim", 0),
                token_dropout_p=saved_args.get("token_dropout_p", 0.0),
                spatial_decay_lambda=saved_args.get("spatial_decay_lambda", 0.0),
            ).to(device)
        elif variant == "v2_5":
            # v2.5 uses CovalentConditioningAdapterV25 (2-layer MLP, learnable scale)
            from anchordiff.covind.cov_adapter_v2_5 import CovalentConditioningAdapterV25
            adapter = CovalentConditioningAdapterV25(
                token_dim=TOKEN_DIM_V2_5, feat_dim=F_A,
            ).to(device)
        else:
            adapter = CovalentConditioningAdapter(token_dim=TOKEN_DIM, feat_dim=F_A).to(device)
        adapter.load_state_dict(ck["adapter_state"])
        adapter.eval()
        print(f"  C-arm adapter loaded ({sum(p.numel() for p in adapter.parameters())} params)  "
              f"scale={float(adapter.scale) if hasattr(adapter, 'scale') and not callable(adapter.scale) else float(getattr(adapter, 'scale', 0.25)):.3f}")
    else:
        print("  C-arm DISABLED (--no_carm)")

    # 3. Pocket prep via DiffSBDD's OWN function — guaranteed compatible.
    # B2 fix: build pocket from a REFERENCE LIGAND (e.g. docked parent) if
    # given. Falls back to warhead_sdf if not — but logs a warning because
    # the 5-atom warhead stub gives only ~9 residues vs the 22-58 residue
    # pockets the model saw at training.
    pdb_model = PDBParser(QUIET=True).get_structure('', str(args.pdbfile))[0]
    pocket_ref_sdf = args.reference_ligand_sdf if args.reference_ligand_sdf else args.warhead_sdf
    if args.reference_ligand_sdf is None:
        print(f"  [B2 WARNING] no --reference_ligand_sdf — building pocket from "
              f"5-atom warhead stub at cutoff={args.pocket_cutoff} Å. Expect "
              f"a pocket size mismatch vs training (22-58 residues). For a "
              f"clean evaluation pass a docked parent ligand SDF.")
    pocket_residues = dsutils.get_pocket_from_ligand(
        pdb_model, str(pocket_ref_sdf), dist_cutoff=args.pocket_cutoff)
    print(f"\nPocket: {len(pocket_residues)} residues (ref={pocket_ref_sdf.name}, "
          f"cutoff={args.pocket_cutoff} Å)")

    # Build pocket batch (replicated across n_samples)
    pocket = model.prepare_pocket(pocket_residues, repeats=args.n_samples)
    print(f"  pocket['x']: {tuple(pocket['x'].shape)}  one_hot: {tuple(pocket['one_hot'].shape)}")

    # 4. Inject C-arm bias into pocket['one_hot']
    if adapter is not None:
        ctx = collect_cys_context_resnames(pocket_residues)
        geom = WARHEAD_GEOM.get(args.warhead_class, WARHEAD_GEOM["Michael Acceptor"])
        # Map our per-warhead-class mechanism string to a CovInDB2 `Reaction`
        # vocabulary entry so the token's mechanism axis matches the training
        # distribution. (QA fix 2026-05-13: without this, inference defaults
        # to OTHER which the trained model never associates with the warhead.)
        _MECH_MAP = {
            "addition":       "Michael Addition",
            "sn2":            "Nucleophilic Substitution",
            "disulfide":      "Disulfide Formation",
            "hemithioacetal": "Nucleophilic Addition",
            "thioimidate":    "Nucleophilic Addition",
            "sn2_at_S":       "Sulfonylation",
            "sulfonylation":  "Sulfonylation",
            "transesterification": "Nucleophilic Substitution",
        }
        reaction_mech = _MECH_MAP.get(geom.get("mechanism"), "OTHER")
        token = token_builder(d_canonical=geom["d"], theta_canonical=geom["angle"],
                              warhead_class=args.warhead_class,
                              cys_context_residues=ctx,
                              reaction_mechanism=reaction_mech)
        token_t = torch.from_numpy(token).float().to(device)
        pkt_oh_before = pocket["one_hot"].detach().clone()
        with torch.no_grad():
            if use_spatial_injection and _ABLATION_ADAPTER_AVAILABLE:
                pocket["one_hot"] = inject_into_pocket_oh_spatial(
                    pocket["one_hot"], token_t, adapter, pocket["x"])
            else:
                pocket["one_hot"] = inject_into_pocket_oh(pocket["one_hot"], token_t, adapter)
        print(f"  C-arm bias mean|Δ|={float((pocket['one_hot'] - pkt_oh_before).abs().mean()):.4f}")

    # 5. Fixed warhead atoms (DiffSBDD's own loader)
    x_fixed, one_hot_fixed = prepare_from_sdf_files([str(args.warhead_sdf)],
                                                    model.lig_type_encoder)
    x_fixed = x_fixed.to(device)
    one_hot_fixed = one_hot_fixed.to(device)
    n_fixed = len(x_fixed)
    print(f"\nWarhead fixed: {n_fixed} atoms")

    # 6. Generate in batches (matches inpaint.py)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(args.outfile))
    n_written = 0
    inv_enc = {v: k for k, v in model.lig_type_encoder.items()}

    n_done = 0
    while n_done < args.n_samples:
        n_b = min(args.batch_size, args.n_samples - n_done)

        # Build the ligand batch following inpaint.py pattern
        num_nodes_lig = torch.full((n_b,), n_fixed + args.add_n_nodes,
                                   dtype=torch.long)
        ligand_mask = dsutils.num_nodes_to_batch_mask(n_b, num_nodes_lig, device)
        ligand = {
            'x':       torch.zeros((len(ligand_mask), model.x_dims), device=device),
            'one_hot': torch.zeros((len(ligand_mask), model.atom_nf), device=device),
            'size':    num_nodes_lig.to(device),
            'mask':    ligand_mask,
        }
        lig_fixed = torch.zeros(len(ligand_mask), device=device)
        for i in range(n_b):
            sele = (ligand_mask == i).nonzero(as_tuple=True)[0]
            ligand['x'][sele[:n_fixed]] = x_fixed
            ligand['one_hot'][sele[:n_fixed]] = one_hot_fixed.float()
            lig_fixed[sele[:n_fixed]] = 1.0

        # Subset the (replicated) pocket to the first n_b samples
        pkt_b = {
            'x':       pocket['x'][(pocket['mask'] < n_b)],
            'one_hot': pocket['one_hot'][(pocket['mask'] < n_b)],
            'size':    pocket['size'][:n_b],
            'mask':    pocket['mask'][(pocket['mask'] < n_b)],
        }
        print(f"  batch  generating {n_b} samples …", flush=True)
        try:
            with torch.no_grad():
                out_lig, out_pkt, om_lig, om_pkt = model.ddpm.inpaint(
                    ligand, pkt_b, lig_fixed=lig_fixed,
                    resamplings=args.resamplings, return_frames=1)
        except Exception as e:
            print(f"    batch failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            n_done += n_b
            continue

        for i in range(n_b):
            sele = (om_lig == i).nonzero(as_tuple=True)[0]
            x_i = out_lig[sele, :3]
            h_i = out_lig[sele, 3:3+F_A]
            atom_types = h_i.argmax(dim=1)
            try:
                mol = build_molecule(x_i, atom_types, model.dataset_info)
                mol = process_molecule(mol, add_hydrogens=False, sanitize=True,
                                        largest_frag=True, relax_iter=0)
                if mol is None: continue
                mol.SetProp("warhead_class", args.warhead_class)
                mol.SetProp("source", "dc_flagship_finetuned" if not args.no_carm
                                       else "dc_flagship_finetuned_no_carm")
                writer.write(mol)
                n_written += 1
            except Exception:
                continue
        n_done += n_b
        print(f"    cumulative written: {n_written}", flush=True)
    writer.close()
    print(f"\nWrote {n_written}/{args.n_samples} mols to {args.outfile}")


if __name__ == "__main__":
    main()
