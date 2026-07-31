"""L2 INPAINT — T4/GPU variant of run_lingo3dmol_l2_inpaint.py.

Differences vs the CPU variant:
  * Uses `lingo3dmol_gpu_compat` shim (numpy/torch compat fixes only, no `.cuda()` no-op)
  * Loads models with `map_location='cuda'` and runs `.cuda()`
  * Moves dataloader tensors to CUDA before forward

Tokenization and inpainting logic is IDENTICAL to the CPU variant — we
import the building blocks from `run_lingo3dmol_l2_inpaint` so any fix
applies to both.
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    cand = (ORIG_CWD / p).resolve()
    if cand.exists():
        return str(cand)
    cand2 = (ROOT / p).resolve()
    if cand2.exists():
        return str(cand2)
    return str(cand)


# GPU compat (numpy/torch fixes only; .cuda() works for real)
import lingo3dmol_gpu_compat  # noqa: F401

# Pre-register a no-op `lingo3dmol_cpu_shim` so that downstream imports of
# `run_lingo3dmol_l0_smoke` (which does `import lingo3dmol_cpu_shim` at module
# top) DON'T clobber our GPU `.cuda()` calls with the CPU no-op shim.
import sys as _sys
import types as _types
_shim = _types.ModuleType("lingo3dmol_cpu_shim")
_shim.__doc__ = "stubbed by run_lingo3dmol_l2_inpaint_gpu.py — GPU mode active"
_sys.modules["lingo3dmol_cpu_shim"] = _shim

import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from util.fragmol_frag_zyh import FragmolUtil
from model.transformer_v1_res_mp1 import TransformerModel as TransformerModel_contact, topkp_random
from model.transformer_v1_res_fac2 import TransformerModel
from dataloader.dataloader_case_nci_res_merge import mydataset as testdataset
from torch.utils.data import DataLoader
from inference.cube_collision_check import CollisionCheck

# Shared L0 primitives
from run_lingo3dmol_l0_smoke import (
    bonusVSpenalty2, get_partial_to_warehouse, molecular_workflow,
    write_product, go_factory, _open_incremental_writer, _flush_summary,
)
# L2 token/coord builders
from run_lingo3dmol_l2_inpaint import (
    ACRYL_TOKENS, ACRYL_TOKEN_LABELS, ATOM_TOKEN_INDEX_TO_ATOM,
    _build_warhead_xyz, _build_initial_partial,
)


def _to_cuda(x):
    return x.cuda() if isinstance(x, torch.Tensor) and torch.cuda.is_available() else x


def run(args):
    args.pocket_pdb = _abs(args.pocket_pdb)
    args.output = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)
    print(f"[L2 inpaint GPU] pocket={args.pocket_pdb}")
    print(f"[L2 inpaint GPU] anchor={args.anchor_json}")
    print(f"[L2 inpaint GPU] gennums={args.gennums} min_acceptable={args.min_acceptable}")
    print(f"[L2 inpaint GPU] acryl prefix tokens ({len(ACRYL_TOKENS)}) = {ACRYL_TOKENS}")
    print(f"[L2 inpaint GPU] acryl prefix labels = {ACRYL_TOKEN_LABELS}")
    print(f"[L2 inpaint GPU] cuda available = {torch.cuda.is_available()}")

    # Load models on CUDA
    caption_contact = TransformerModel_contact()
    sd_contact = torch.load(args.contact_path, map_location="cpu", weights_only=False)
    caption_contact.load_state_dict(sd_contact, strict=False)
    caption_contact = caption_contact.cuda()
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.eval()

    caption = TransformerModel()
    sd_cap = torch.load(args.caption_path, map_location="cpu", weights_only=False)
    caption.load_state_dict(sd_cap)
    caption = caption.cuda()
    caption = nn.DataParallel(caption)
    caption.eval()

    line = f",,{args.pocket_pdb}"
    testset = testdataset([line])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    for batch in testloader:
        coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
        coc = CollisionCheck(args.pocket_pdb, args.coc_dis, center=center)
        pocket_center = center[0].cpu().numpy()
        break

    initial_partial, warhead_xyz, warhead_grid = _build_initial_partial(args.anchor_json, pocket_center)
    print(f"[L2 inpaint GPU] pocket_center={pocket_center.tolist()}")
    print(f"[L2 inpaint GPU] warhead xyz (Å):")
    for i, p in enumerate(warhead_xyz):
        print(f"    atom {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print(f"[L2 inpaint GPU] warhead grid (voxel):")
    for i, p in enumerate(warhead_grid):
        print(f"    atom {i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    if (warhead_grid.min() < 0) or (warhead_grid.max() >= 240):
        print(f"[L2 inpaint GPU] WARNING grid OOB: min={warhead_grid.min():.2f} max={warhead_grid.max():.2f}")

    results = []
    start = time.time()
    sample_num = args.gen_frag_set
    warehouse = [[], []]

    writer, summary_path = _open_incremental_writer(args.output)
    summary = {
        "n_sampled": 0,
        "elapsed_sec": 0,
        "output_sdf": args.output,
        "pocket_pdb": args.pocket_pdb,
        "status": "running",
        "gennums_target": args.gennums,
        "min_acceptable": args.min_acceptable,
        "plan": "L2_inpaint_GPU",
        "acryl_prefix_tokens": ACRYL_TOKENS,
        "acryl_prefix_labels": ACRYL_TOKEN_LABELS,
        "warhead_xyz_receptor": warhead_xyz.tolist(),
        "warhead_grid_voxel": warhead_grid.tolist(),
        "pocket_center_receptor": pocket_center.tolist(),
    }
    _flush_summary(summary_path, summary)

    try:
        while len(results) < args.gennums and (time.time() - start) < args.max_run_seconds:
            for batch in testloader:
                coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
                # Move inputs to CUDA
                coords = _to_cuda(coords)
                residue = _to_cuda(residue)
                atom_type = _to_cuda(atom_type)
                mask = _to_cuda(mask)
                center = _to_cuda(center)
                index = _to_cuda(index)
                contact_prob = _to_cuda(contact_prob)
                contact_scaffold_prob = _to_cuda(contact_scaffold_prob)

                with torch.no_grad():
                    index = index.repeat(sample_num)
                    center = center.repeat(sample_num, 1)
                    if contact_prob.shape[-1] == 0 or contact_scaffold_prob.shape[-1] == 0:
                        model_cp, model_csp = caption_contact(
                            coords=coords, residue=residue, atom_type=atom_type,
                            src_mask=mask, isTrain=args.isTrain)
                    if contact_prob.shape[-1] == 0:
                        contact_prob = model_cp
                    if contact_scaffold_prob.shape[-1] == 0:
                        contact_scaffold_prob = model_csp

                    contact_prob0 = torch.where(contact_prob > args.nci_thrs, 2, 0)
                    contact_scaffold_prob1 = torch.where(contact_scaffold_prob > 0.9, 1, 0)
                    contact_prob1 = contact_prob0 + contact_scaffold_prob1
                    contact_prob = contact_prob.repeat(sample_num, 1)
                    residue_use = residue.repeat(sample_num, 1)
                    residue_mask = torch.where(residue_use != 5, 1.0, 0.0)
                    src_mask_repeat = mask.squeeze(1).repeat(sample_num, 1)
                    contact_prob = contact_prob.masked_fill(src_mask_repeat == 0, 0)
                    contact_prob = contact_prob.masked_fill(residue_mask == 0, 0)
                    contact_prob = torch.softmax(contact_prob * 5, dim=-1)
                    contact_idx = topkp_random(contact_prob, top_k=args.topk, top_p=0.9, thred=0.0)
                    factory_args = [coords, residue, mask, atom_type, center, caption,
                                    contact_idx, contact_prob1, coc, contact_scaffold_prob1]
                    seeded_partial = [
                        [list(initial_partial[0][0])],
                        [list(initial_partial[1][0])],
                    ]
                    molecular_workflow(0, warehouse, seeded_partial, factory_args, results, args,
                                       writer=writer, summary_path=summary_path,
                                       summary=summary, start_time=start)
                    if len(results) >= args.gennums:
                        break
            if len(results) >= args.gennums:
                break
            elapsed_so_far = time.time() - start
            if (len(results) >= args.min_acceptable
                    and elapsed_so_far >= 0.6 * args.max_run_seconds):
                print(f"[L2 inpaint GPU] min_acceptable={args.min_acceptable} met with "
                      f"n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[L2 inpaint GPU] generated {len(results)} mols in {elapsed:.1f}s status={status}")
    print(f"[L2 inpaint GPU] wrote SDF: {args.output}")

    summary.update({"n_sampled": len(results), "elapsed_sec": elapsed, "status": status})
    _flush_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--anchor_json", default="data/lingo3dmol_L2_inpaint/anchor_zap70_cys346.json")
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--caption_path", default="checkpoint/gen_mol.pkl")
    p.add_argument("--gennums", type=int, default=20)
    p.add_argument("--min_acceptable", type=int, default=5)
    p.add_argument("--gen_frag_set", type=int, default=10)
    p.add_argument("--prod_time", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=0.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=2400)
    p.add_argument("--isTrain", action="store_true")
    p.add_argument("--USE_THRESHOLD", action="store_true", default=True)
    p.add_argument("--isMultiSample", action="store_true", default=True)
    p.add_argument("--isGuideSample", action="store_true", default=True)
    p.add_argument("--OnceMolGen", action="store_true")
    args = p.parse_args()
    run(args)
