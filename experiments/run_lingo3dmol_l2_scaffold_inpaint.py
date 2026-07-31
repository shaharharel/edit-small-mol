"""L2 SCAFFOLD INPAINT — clamp the FULL recognition motif (acrylamide + isoindoline-methylene).

Hypothesis (medchem reviewer): warhead-only anchor (`C=CC(=O)N[*]`) is too narrow.
The model decodes trivial caps because the conditional generation problem is too
under-determined. Clamping the recognition SCAFFOLD `C=CC(=O)N1Cc2ccccc2C1[*]`
should force the decoder to only fill in the hinge-binder arm — a much easier
conditional generation problem.

Scaffold prefix breakdown (FSMILES vocab from util/fragmol_frag_zyh.py):

  SMILES:   C  =  C  C  (  =  O  )  N1 C  c2 c  c  c  c  c2 C1 [*]
  token:    C_0 =_0 C_0 C_0 (_0 =_0 O_0 )_0 N_5 1_0 C_5 c_5 2_0 c_6 c_6 c_6 c_6 c_5 2_0 C_5 1_0 [*]_0
  vocab id: 4   68  4   4   70  68  35  71  17  61  5   11  62  12  12  12  12  11  62  5   61  74

Wrapped: [start_0=1] + scaffold_tokens + [sep_0=3]  → 24 tokens total.

Verified at import time via FragmolUtil.decode3d → `*C1c2ccccc2CN1C(=O)C=C`
(same molecule as `C=CC(=O)N1Cc2ccccc2C1[*]`).

Atom indices in the 2D scaffold (no [*]):

   0  C  (CH2=  terminal, attacks Cys SG)
   1  C  (vinyl Cα)
   2  C  (carbonyl C)
   3  O  (carbonyl O)
   4  N  (amide / isoindoline N, ring atom)
   5  C  (isoindoline CH2, ring atom)
   6  c  (benzene ring junction C)
   7  c  (benzene)
   8  c  (benzene)
   9  c  (benzene)
   10 c  (benzene)
   11 c  (benzene ring junction C)
   12 C  (isoindoline CH, ring atom; carries [*])
   13 = [*]  (open valence to hinge arm — not embedded)

13-heavy-atom scaffold (without [*]) gets ETKDG-embedded, then translated so
atom 0 sits at cb_pos_target (1.85 Å from Cys346 SG) and rotated so the
atom0->atom1 bond aligns with anchor_attack_vector — same recipe as L2 v3.

Run target:
  conda run -n lingo3dmol python experiments/run_lingo3dmol_l2_scaffold_inpaint.py \
      --pocket_pdb data/lingo3dmol_smoke/zap70_pocket_cys346.pdb \
      --output    data/lingo3dmol_L2_scaffold_anchor/samples_T10.sdf \
      --gennums 50 --gen_frag_set 10 --prod_time 3 \
      --coc_dis 0.5 --min_acceptable 15 --tempture 1.0
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

from run_lingo3dmol_l0_smoke import (
    changepos2, bonusVSpenalty2, get_partial_to_warehouse,
    molecular_workflow, write_product, go_factory,
    _open_incremental_writer, _flush_summary,
)
from anchor_geometry import (
    rotate_scaffold_to_bd, assert_bd_angle, measured_bd_angle,
)


_FU = FragmolUtil()
_V = _FU.vocab_c2i_v1_decode_new

# --- Vocab sanity ----------------------------------------------------
assert _V['start_0'] == 1
assert _V['C_0']     == 4
assert _V['C_5']     == 5
assert _V['c_5']     == 11
assert _V['c_6']     == 12
assert _V['N_5']     == 17
assert _V['O_0']     == 35
assert _V['=_0']     == 68
assert _V['(_0']     == 70
assert _V[')_0']     == 71
assert _V['[*]_0']   == 74
assert _V['sep_0']   == 3
assert _V['1_0']     == 61
assert _V['2_0']     == 62

# --- Scaffold token sequence ----------------------------------------
# SMILES: C=CC(=O)N1Cc2ccccc2C1[*]
# Convention for fused 5/6 atoms (idx 6 & 11): tag with `c_5` (smallest ring,
# also the one whose ring-closure digit is currently open at that position).
SCAFFOLD_TOKEN_LABELS = [
    'start_0',
    'C_0',           # atom 0  (terminal CH2= of acrylamide)
    '=_0',
    'C_0',           # atom 1  (vinyl CH)
    'C_0',           # atom 2  (carbonyl C)
    '(_0',
    '=_0',
    'O_0',           # atom 3  (carbonyl O)
    ')_0',
    'N_5',           # atom 4  (amide / isoindoline N)
    '1_0',
    'C_5',           # atom 5  (isoindoline CH2)
    'c_5',           # atom 6  (benzene junction; fused to 5-ring)
    '2_0',
    'c_6',           # atom 7  (benzene)
    'c_6',           # atom 8  (benzene)
    'c_6',           # atom 9  (benzene)
    'c_6',           # atom 10 (benzene)
    'c_5',           # atom 11 (benzene junction; fused to 5-ring)
    '2_0',
    'C_5',           # atom 12 (isoindoline CH carrying [*])
    '1_0',
    '[*]_0',
    'sep_0',
]
SCAFFOLD_TOKENS = [_V[t] for t in SCAFFOLD_TOKEN_LABELS]

# Map token position -> heavy-atom index in the 13-atom embedded scaffold.
# Non-atom tokens (=, (, ), ring digits, [*], sep, start) carry the previous
# atom's coord so the geometric conditioning stays smooth.
ATOM_TOKEN_INDEX_TO_ATOM = {
    1: 0,   # C_0
    3: 1,   # C_0
    4: 2,   # C_0 (carbonyl)
    7: 3,   # O_0
    9: 4,   # N_5
    11: 5,  # C_5
    12: 6,  # c_5 (junction)
    14: 7,  # c_6
    15: 8,  # c_6
    16: 9,  # c_6
    17: 10, # c_6
    18: 11, # c_5 (junction)
    20: 12, # C_5 ([*] carrier)
}

# Sanity: decode the scaffold token sequence at import time to confirm vocab usage.
_test_batch = np.array([SCAFFOLD_TOKENS + [0] * (100 - len(SCAFFOLD_TOKENS))])
_test_pos = np.zeros((1, 100, 3), dtype=np.float32)
_smi_chk, _ss_chk, _mols_chk = _FU.decode3d(_test_batch, _test_pos)
print(f"[L2 scaffold] scaffold decode sanity: {_smi_chk[0] if _smi_chk else None}")
assert _smi_chk and _smi_chk[0] is not None, "scaffold tokens failed to decode"
# Canonicalize and check it contains the expected ring system
_canon = Chem.MolToSmiles(Chem.MolFromSmiles(_smi_chk[0])) if _smi_chk[0] else None
print(f"[L2 scaffold] canonical:                {_canon}")
_expected = Chem.MolToSmiles(Chem.MolFromSmiles('C=CC(=O)N1Cc2ccccc2C1[*]'))
print(f"[L2 scaffold] expected canonical:       {_expected}")
assert _canon == _expected, f"scaffold mismatch: got {_canon!r} expected {_expected!r}"


def _build_scaffold_xyz(anchor_json_path):
    """ETKDG-embed the 13-heavy-atom scaffold, then translate+rotate so atom 0
    sits at cb_pos_target and the angle SG-atom0-atom1 equals the
    Burgi-Dunitz angle (107 deg) at the electrophilic Cbeta carbon.

    Previous (buggy) recipe aligned atom0->atom1 COLINEAR with the attack
    vector, forcing SG-atom0-atom1 = 180 deg (wrong for Michael addition).

    Returns:
        xyz: (13, 3) np.float64 array in receptor (Å) frame.
    """
    with open(anchor_json_path) as f:
        anchor = json.load(f)
    cb_target = np.array(anchor["cb_pos_target"], dtype=np.float64)
    av = np.array(anchor["anchor_attack_vector"], dtype=np.float64)
    sg_pos = np.array(anchor["sg_pos"], dtype=np.float64)
    perp_in_plane = np.array(anchor.get("frame_e2_in_plane",
                                        [0.0, 0.0, 0.0]), dtype=np.float64)
    if np.linalg.norm(perp_in_plane) < 1e-6:
        perp_in_plane = None
    bd_deg = float(anchor.get("burgi_dunitz_angle_deg", 107.0))

    # 13-heavy-atom scaffold (no [*])
    scaffold_smi = "C=CC(=O)N1Cc2ccccc2C1"
    mol = Chem.MolFromSmiles(scaffold_smi)
    assert mol is not None, f"failed to parse {scaffold_smi}"
    assert mol.GetNumAtoms() == 13, f"expected 13 atoms, got {mol.GetNumAtoms()}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        # Retry without random seed; if still failing, fall back to 2D coords.
        params.randomSeed = -1
        res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        AllChem.Compute2DCoords(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        print(f"[L2 scaffold] MMFF relax failed (continuing): {e}")
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )

    rotated = rotate_scaffold_to_bd(
        coords, cb_target=cb_target, av=av,
        perp_in_plane=perp_in_plane, bd_deg=bd_deg,
    )
    measured = assert_bd_angle(rotated, sg_pos, bd_deg=bd_deg, tol_deg=5.0)
    print(f"[L2 scaffold] BD angle at atom0 = {measured:.2f} deg "
          f"(target {bd_deg:.1f} deg)")
    return rotated  # (13, 3)


def _build_initial_partial(anchor_json_path, pocket_center):
    """Build partial_product = [[token_list], [coords_per_token]] for the
    scaffold prefix."""
    xyz = _build_scaffold_xyz(anchor_json_path)
    grid = (xyz - pocket_center) / 0.1 + 119.5
    grid = grid.astype(np.float32)

    coords_per_token = []
    last_atom_grid = grid[0].tolist()
    for tok_idx in range(len(SCAFFOLD_TOKENS)):
        if tok_idx in ATOM_TOKEN_INDEX_TO_ATOM:
            last_atom_grid = grid[ATOM_TOKEN_INDEX_TO_ATOM[tok_idx]].tolist()
        coords_per_token.append(list(last_atom_grid))

    return [
        [list(SCAFFOLD_TOKENS)],
        [list(coords_per_token)],
    ], xyz, grid


def run(args):
    args.pocket_pdb  = _abs(args.pocket_pdb)
    args.output      = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[L2 scaffold] pocket={args.pocket_pdb}")
    print(f"[L2 scaffold] anchor={args.anchor_json}")
    print(f"[L2 scaffold] gennums={args.gennums} min_acceptable={args.min_acceptable}")
    print(f"[L2 scaffold] tempture={args.tempture}")
    print(f"[L2 scaffold] scaffold prefix ({len(SCAFFOLD_TOKENS)} tokens) = {SCAFFOLD_TOKENS}")
    print(f"[L2 scaffold] scaffold labels                                 = {SCAFFOLD_TOKEN_LABELS}")

    caption_contact = TransformerModel_contact()
    dict_ = torch.load(args.contact_path, map_location="cpu", weights_only=False)
    caption_contact.load_state_dict(dict_, strict=False)
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.eval()

    caption = TransformerModel()
    # BUG FIX 2026-05-31 (QA round-2 A1): unwrap dict-wrapped FT checkpoints.
    _raw_ckpt = torch.load(args.caption_path, map_location="cpu", weights_only=False)
    _sd = _raw_ckpt["model"] if isinstance(_raw_ckpt, dict) and "model" in _raw_ckpt else _raw_ckpt
    _info = caption.load_state_dict(_sd, strict=False)
    print(f"[load caption] from {args.caption_path}: missing={len(_info.missing_keys)} unexpected={len(_info.unexpected_keys)}")
    assert len(_info.missing_keys) < 10, f"load_state_dict missing too many params: {_info.missing_keys[:5]}"
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

    initial_partial, scaffold_xyz, scaffold_grid = _build_initial_partial(
        args.anchor_json, pocket_center
    )
    print(f"[L2 scaffold] pocket_center={pocket_center.tolist()}")
    print(f"[L2 scaffold] scaffold xyz (Å):")
    for i, p in enumerate(scaffold_xyz):
        print(f"    atom {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print(f"[L2 scaffold] scaffold grid (voxel):")
    for i, p in enumerate(scaffold_grid):
        print(f"    atom {i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    if (scaffold_grid.min() < 0) or (scaffold_grid.max() >= 240):
        print(f"[L2 scaffold] WARNING: grid coords outside [0,240): "
              f"min={scaffold_grid.min():.2f} max={scaffold_grid.max():.2f}. "
              f"Pocket center may not align with scaffold anchor — generation likely fails.")
    print(f"[L2 scaffold] seeded {len(initial_partial[0][0])} tokens + "
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
        "plan": "L2_scaffold_inpaint",
        "scaffold_prefix_tokens": SCAFFOLD_TOKENS,
        "scaffold_prefix_labels": SCAFFOLD_TOKEN_LABELS,
        "scaffold_xyz_receptor": scaffold_xyz.tolist(),
        "scaffold_grid_voxel": scaffold_grid.tolist(),
        "pocket_center_receptor": pocket_center.tolist(),
        "tempture": args.tempture,
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
                print(f"[L2 scaffold] min_acceptable={args.min_acceptable} met with "
                      f"n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[L2 scaffold] generated {len(results)} mols in {elapsed:.1f}s status={status}")
    print(f"[L2 scaffold] wrote SDF: {args.output}")

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
    p.add_argument("--gennums", type=int, default=50)
    p.add_argument("--min_acceptable", type=int, default=15)
    p.add_argument("--gen_frag_set", type=int, default=10)
    p.add_argument("--prod_time", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=0.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=3000)
    p.add_argument("--isTrain", action="store_true")
    p.add_argument("--USE_THRESHOLD", action="store_true", default=True)
    p.add_argument("--isMultiSample", action="store_true", default=True)
    p.add_argument("--isGuideSample", action="store_true", default=True)
    p.add_argument("--OnceMolGen", action="store_true")
    args = p.parse_args()
    run(args)
