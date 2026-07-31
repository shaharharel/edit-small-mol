"""L2 INPAINT (v3) — Lingo3DMol inference with forced acrylamide warhead prefix.

CHANGELOG vs v2 (see data/lingo3dmol_L2_inpaint/INPAINT_PATCH_NOTES_V3.md):

  * v1/v2 used ACRYL_TOKENS = [4, 68, 4, 4, 70, 68, 35, 71, 16, 74]. This
    MISSED the leading `start_0=1` token. In the model loop
    `captions[:,0]=1` is set automatically, but the guidepath array also
    starts at index 0. With the old list, `gudiecap[:, :10] = ACRYL`, the
    model's `start_this_step=10`, and forcing at `i=1` reads
    `guidepath[0][:,1] = 68 (=_0)` instead of `4 (C_0)`. The resulting
    forced prefix was effectively `start, =, C, C, (, =, O, ), N, [*]`
    which decodes to `=CC(=O)N[*]` — invalid SMILES, hence zero mols.

  * v3 uses a 12-token guidepath:
        [start_0=1, C_0=4, =_0=68, C_0=4, C_0=4, (_0=70, =_0=68, O_0=35,
         )_0=71, N_0=16, [*]_0=74, sep_0=3]
    with start_this_step=12. The trailing sep_0 is REQUIRED so that the
    workflow slicing `frag_new_cap = cap[o_index+1:index+1]` (where
    o_index = position of last sep) extracts only the NEW tokens at the
    next recursion level, without duplicating the prefix.

  * Coords are computed by:
      1. RDKit ETKDG/MMFF embed of `C=CC(=O)N` (5 heavy atoms).
      2. Translate so atom 0 (terminal CH2) sits at `cb_pos_target` from
         data/lingo3dmol_anchor_zap70_cys346.json.
      3. Rotate so the atom0->atom1 bond aligns with `anchor_attack_vector`
         (Bürgi–Dunitz attack axis from SG into the Michael acceptor).
      4. Convert each heavy-atom xyz to voxel grid coords:
            grid = (xyz - pocket_center) / 0.1 + 119.5
         where pocket_center is the dataloader-returned center of the
         ZAP70 pocket PDB.

  * Coord guidepath is filled per-token: atom-tokens get their own
    voxel coord; non-atom tokens (=, (, ), [*], sep) carry the coord of
    the preceding atom-token (so geometry conditioning stays coherent).

Token map (FragmolUtil.vocab_c2i_v1_decode_new), verified at runtime:

  'start_0' -> 1   'C_0' -> 4    '=_0' -> 68   '(_0' -> 70   ')_0' -> 71
  'O_0'     -> 35  'N_0' -> 16   '[*]_0' -> 74  'sep_0' -> 3

Run target (local Mac CPU, ~5 GB / 4 h budget):

  conda run -n quris python experiments/run_lingo3dmol_l2_inpaint.py \
      --pocket_pdb data/lingo3dmol_smoke/zap70_pocket_cys346.pdb \
      --output    data/lingo3dmol_L2_inpaint/samples.sdf \
      --gennums 10 --min_acceptable 5 --gen_frag_set 6
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path
import numpy as np

# Install CPU shim FIRST, before any Lingo3DMol model import
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


import lingo3dmol_cpu_shim  # noqa: F401

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

# Re-use the L0 smoke runner's workflow primitives
from run_lingo3dmol_l0_smoke import (
    changepos2, bonusVSpenalty2, get_partial_to_warehouse,
    molecular_workflow, write_product, go_factory,
    _open_incremental_writer, _flush_summary,
)


# Build the warhead prefix from the FSMILES vocab. This must be in sync
# with FragmolUtil.vocab_c2i_v1_decode_new in
# external/Lingo3DMol/util/fragmol_frag_zyh.py.
_FU = FragmolUtil()
_V = _FU.vocab_c2i_v1_decode_new

# Sanity: print key vocab IDs at import time so any drift is immediately visible.
assert _V['start_0'] == 1, f"vocab drift: start_0={_V['start_0']}"
assert _V['C_0'] == 4, f"vocab drift: C_0={_V['C_0']}"
assert _V['=_0'] == 68, f"vocab drift: ={_V['=_0']}"
assert _V['(_0'] == 70, f"vocab drift: (={_V['(_0']}"
assert _V[')_0'] == 71, f"vocab drift: )={_V[')_0']}"
assert _V['O_0'] == 35, f"vocab drift: O={_V['O_0']}"
assert _V['N_0'] == 16, f"vocab drift: N={_V['N_0']}"
assert _V['[*]_0'] == 74, f"vocab drift: [*]={_V['[*]_0']}"
assert _V['sep_0'] == 3, f"vocab drift: sep={_V['sep_0']}"

# 12-token FSMILES warhead prefix (with start_0 lead-in and sep_0 terminator).
# SMILES viewed as: C=CC(=O)N + [*] (open bond) + fragment separator.
ACRYL_TOKEN_LABELS = [
    'start_0', 'C_0', '=_0', 'C_0', 'C_0', '(_0',
    '=_0', 'O_0', ')_0', 'N_0', '[*]_0', 'sep_0',
]
ACRYL_TOKENS = [_V[t] for t in ACRYL_TOKEN_LABELS]

# Index map: which prefix tokens are ATOM tokens (the only ones that
# need an honest 3D coord). Other tokens carry the previous atom's coord
# to keep geometric conditioning monotone.
#   token idx   label       atom idx
#   0           start_0     n/a (model auto-sets gt_coords[:,0] = contact_idx)
#   1           C_0         0  (terminal CH2)
#   2           =_0         -  (->1)
#   3           C_0         1  (Michael acceptor Cα)
#   4           C_0         2  (carbonyl C)
#   5           (_0         -  (->4)
#   6           =_0         -  (->4)
#   7           O_0         3  (carbonyl O)
#   8           )_0         -  (->4)
#   9           N_0         4  (amide N)
#   10          [*]_0       -  (open bond marker; no atom)
#   11          sep_0       -  (fragment separator)
ATOM_TOKEN_INDEX_TO_ATOM = {1: 0, 3: 1, 4: 2, 7: 3, 9: 4}


def _build_warhead_xyz(anchor_json_path):
    """Build receptor-frame xyz for the 5 heavy atoms of C=CC(=O)N.

    Pipeline:
      1. ETKDG/MMFF embed `C=CC(=O)N` in 3D.
      2. Translate atom 0 (terminal CH2) to cb_pos_target.
      3. Rotate so atom0->atom1 bond aligns with anchor_attack_vector
         (post-Michael-addition, the CH2 sits at 1.85 Å from Cys346 SG
         and the molecule extends away from SG along that vector).

    Returns:
        xyz: (5, 3) np.float64 array in receptor (Å) frame.
    """
    with open(anchor_json_path) as f:
        anchor = json.load(f)
    # PREFERRED: use pre-staged 5-atom XYZ (RDKit ETKDG pose, chemically real).
    if "warhead_xyz_at_cys" in anchor:
        xyz = np.array(
            [a["xyz"] for a in anchor["warhead_xyz_at_cys"]],
            dtype=np.float64,
        )
        assert xyz.shape == (5, 3), f"expected (5,3) warhead pose, got {xyz.shape}"
        return xyz
    # Fallback: do ETKDG embed + rotate onto cb_pos_target/anchor_attack_vector.
    cb_target = np.array(anchor["cb_pos_target"], dtype=np.float64)
    av = np.array(anchor["anchor_attack_vector"], dtype=np.float64)
    av_norm = av / np.linalg.norm(av)

    # Embed C=CC(=O)N
    mol = Chem.MolFromSmiles("C=CC(=O)N")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        # Fallback to 2D embed if ETKDG fails on this tiny ligand (shouldn't).
        AllChem.Compute2DCoords(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                      dtype=np.float64)

    # Align atom0->atom1 with av_norm and pin atom 0 to cb_target.
    v0 = coords[1] - coords[0]
    v0_norm = v0 / np.linalg.norm(v0)

    cross = np.cross(v0_norm, av_norm)
    c = float(np.dot(v0_norm, av_norm))
    if np.linalg.norm(cross) < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        s = float(np.linalg.norm(cross))
        K = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        R = np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))

    rotated = (coords - coords[0]) @ R.T + cb_target
    return rotated  # (5, 3)


def _build_initial_partial(anchor_json_path, pocket_center):
    """Construct partial_product = [[token_list], [coords_per_token]].

    The coord list has one entry per prefix token (length = len(ACRYL_TOKENS));
    atom-token coords are real, non-atom-token coords carry the previous
    atom's coord so the conditioning stays geometrically smooth.

    Returns:
        partial_product: list with two sub-lists, each containing exactly
        one fragment (lists, not arrays, to match the workflow API).
    """
    xyz = _build_warhead_xyz(anchor_json_path)  # (5, 3) receptor Å
    # Receptor xyz -> voxel grid (matches inference_avoid_clash.changepos2):
    grid = (xyz - pocket_center) / 0.1 + 119.5
    grid = grid.astype(np.float32)

    coords_per_token = []
    last_atom_grid = grid[0].tolist()  # placeholder for start_0 (model overrides)
    for tok_idx, _ in enumerate(ACRYL_TOKENS):
        if tok_idx in ATOM_TOKEN_INDEX_TO_ATOM:
            last_atom_grid = grid[ATOM_TOKEN_INDEX_TO_ATOM[tok_idx]].tolist()
        coords_per_token.append(list(last_atom_grid))

    # partial_product expects [[frag_list], [coord_list]] where each
    # "fragment" is a list of token IDs and the matching coords.
    return [
        [list(ACRYL_TOKENS)],
        [list(coords_per_token)],
    ], xyz, grid


def run(args):
    args.pocket_pdb = _abs(args.pocket_pdb)
    args.output = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)
    print(f"[L2 inpaint v3] pocket={args.pocket_pdb}")
    print(f"[L2 inpaint v3] anchor={args.anchor_json}")
    print(f"[L2 inpaint v3] gennums={args.gennums} min_acceptable={args.min_acceptable}")
    print(f"[L2 inpaint v3] acryl prefix tokens (12) = {ACRYL_TOKENS}")
    print(f"[L2 inpaint v3] acryl prefix labels      = {ACRYL_TOKEN_LABELS}")

    # Load models on CPU
    caption_contact = TransformerModel_contact()
    dict_ = torch.load(args.contact_path, map_location="cpu", weights_only=False)
    caption_contact.load_state_dict(dict_, strict=False)
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.eval()

    caption = TransformerModel()
    caption.load_state_dict(torch.load(args.caption_path, map_location="cpu", weights_only=False))
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
    print(f"[L2 inpaint v3] pocket_center={pocket_center.tolist()}")
    print(f"[L2 inpaint v3] warhead xyz (Å):")
    for i, p in enumerate(warhead_xyz):
        print(f"    atom {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print(f"[L2 inpaint v3] warhead grid (voxel):")
    for i, p in enumerate(warhead_grid):
        print(f"    atom {i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    # Sanity: grid must fall inside [0, 240) or model breaks
    if (warhead_grid.min() < 0) or (warhead_grid.max() >= 240):
        print(f"[L2 inpaint v3] WARNING: grid coords outside [0,240): "
              f"min={warhead_grid.min():.2f} max={warhead_grid.max():.2f}. "
              f"Pocket center may not align with warhead anchor — generation will likely fail.")
    print(f"[L2 inpaint v3] seeded {len(initial_partial[0][0])} tokens + "
          f"{len(initial_partial[1][0])} per-token coords; "
          f"atom coords = {len(ATOM_TOKEN_INDEX_TO_ATOM)}")

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
        "plan": "L2_inpaint_v3",
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
                    # SEED with acryl prefix (deep-copy each call so recursion can pop)
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
                print(f"[L2 inpaint v3] min_acceptable={args.min_acceptable} met with "
                      f"n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[L2 inpaint v3] generated {len(results)} mols in {elapsed:.1f}s status={status}")
    print(f"[L2 inpaint v3] wrote SDF: {args.output}")

    summary.update({"n_sampled": len(results), "elapsed_sec": elapsed, "status": status})
    _flush_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--anchor_json", default="data/lingo3dmol_anchor_zap70_cys346.json")
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--caption_path", default="checkpoint/gen_mol.pkl")
    p.add_argument("--gennums", type=int, default=10)
    p.add_argument("--min_acceptable", type=int, default=5)
    p.add_argument("--gen_frag_set", type=int, default=6)
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
