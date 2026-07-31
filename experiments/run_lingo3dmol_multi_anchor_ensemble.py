"""L2 SCAFFOLD INPAINT — multi-chassis ensemble.

Generates Lingo3DMol L2 inpaint cohorts for FOUR different covalent kinase
scaffolds (A/B/C/D) against the ZAP70 Cys346 anchor, then merges them into a
single ensemble cohort. Reuses the C5 attachment recipe (ETKDG embed,
rotate-to-attack-vector, grid coords). Only the FSMILES traversal token list
and the heavy-atom indexing change per chassis.

Chassis catalog:
  A — isoindoline_C5         : C=CC(=O)N1Cc2cccc([*])c2C1
                               Today's headline cohort; skipped here — reuse
                               existing samples_T10.sdf from
                               data/lingo3dmol_L2_scaffold_C5/.
  B — isoindoline_amide_pip  : C=CC(=O)N1Cc2cccc(C(=O)N3CCC([*])CC3)c2C1
                               MM-GBSA #1 winner geometry: warhead + isoindoline
                               + amide + piperidine; [*] on piperidine C4.
  C — THIQ_aryl              : C=CC(=O)N1CCc2cccc([*])c2C1
                               Tetrahydroisoquinoline classical alt-chassis.
  D — azaindoline_C5         : C=CC(=O)N1Cc2cncc([*])c2C1
                               Pyridine-fused isoindoline; extra HBA for
                               αC-helix interaction.

Each chassis runs separately at T=1.0 / gennums=30 / min_acceptable=15 and
outputs to data/lingo3dmol_multi_anchor/{ID}/samples_T10.sdf.

Wall budget: Mac CPU ~10 min/chassis. Total ~30 min for B+C+D.

Run target:
  conda run -n lingo3dmol python experiments/run_lingo3dmol_multi_anchor_ensemble.py \
      --pocket_pdb data/lingo3dmol_smoke/zap70_pocket_cys346.pdb \
      --chassis_id B \
      --output    data/lingo3dmol_multi_anchor/B/samples_T10.sdf \
      --gennums 30 --gen_frag_set 10 --prod_time 3 \
      --coc_dis 0.5 --min_acceptable 15 --tempture 1.0
"""
from __future__ import annotations
import os, sys, time, argparse, json, math
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

# ---------------------------------------------------------------------------
# Chassis definitions
# ---------------------------------------------------------------------------
# Each entry: (smiles_without_star_for_etkdg, token_label_list, atom_token_idx_to_atom_idx_map, expected_canonical_smiles_with_star)
# The ETKDG-embed SMILES OMITS the [*] (since [*] is decoded as a SMARTS hole;
# we ETKDG-embed the scaffold's *real* heavy atoms only). The expected
# canonical SMILES (with [*]) is used as a round-trip assertion.

CHASSIS_REGISTRY: dict = {
    # ---- B: isoindoline + amide + piperidine, [*] on piperidine C4 ----
    "B": {
        "embed_smiles": "C=CC(=O)N1Cc2cccc(C(=O)N3CCCCC3)c2C1",
        "expected_with_star": "C=CC(=O)N1Cc2cccc(C(=O)N3CCC([*])CC3)c2C1",
        "n_atoms_expected": 21,
        "labels": [
            'start_0',
            'C_0',                                    # atom 0  (CH2=)
            '=_0',
            'C_0',                                    # atom 1  (vinyl)
            'C_0',                                    # atom 2  (carbonyl C)
            '(_0', '=_0', 'O_0', ')_0',               # atom 3  (=O)
            'N_5',                                    # atom 4  (isoindoline N)
            '1_0',
            'C_5',                                    # atom 5  (CH2)
            'c_5',                                    # atom 6  (aromatic junction)
            '2_0',
            'c_6',                                    # atom 7
            'c_6',                                    # atom 8
            'c_6',                                    # atom 9
            'c_6',                                    # atom 10 (has amide branch)
            '(_0',
            'C_0',                                    # atom 11 (amide C)
            '(_0', '=_0', 'O_0', ')_0',               # atom 12 (=O of amide)
            'N_6',                                    # atom 13 (piperidine N)
            '3_0',
            'C_6',                                    # atom 14
            'C_6',                                    # atom 15
            'C_6',                                    # atom 16 (piperidine C4 [*])
            '(_0', '[*]_0', ')_0',
            'C_6',                                    # atom 17
            'C_6',                                    # atom 18
            '3_0',
            ')_0',
            'c_5',                                    # atom 19 (aromatic junction)
            '2_0',
            'C_5',                                    # atom 20 (sp3 isoindoline CH)
            '1_0',
            'sep_0',
        ],
        # token index -> heavy atom index in the 21-atom embed
        "tok2atom": {
            1: 0, 3: 1, 4: 2, 7: 3,
            9: 4, 11: 5, 12: 6,
            14: 7, 15: 8, 16: 9, 17: 10,
            19: 11, 22: 12, 24: 13,
            26: 14, 27: 15, 28: 16,
            32: 17, 33: 18,
            36: 19, 38: 20,
        },
    },

    # ---- C: tetrahydroisoquinoline, [*] on aromatic C5 ----
    "C": {
        "embed_smiles": "C=CC(=O)N1CCc2ccccc2C1",
        "expected_with_star": "C=CC(=O)N1CCc2cccc([*])c2C1",
        "n_atoms_expected": 14,
        "labels": [
            'start_0',
            'C_0',                                    # atom 0
            '=_0',
            'C_0',                                    # atom 1
            'C_0',                                    # atom 2
            '(_0', '=_0', 'O_0', ')_0',               # atom 3
            'N_6',                                    # atom 4 (THIQ N, 6-mem)
            '1_0',
            'C_6',                                    # atom 5 (CH2)
            'C_6',                                    # atom 6 (CH2)
            'c_6',                                    # atom 7 (junction)
            '2_0',
            'c_6',                                    # atom 8
            'c_6',                                    # atom 9
            'c_6',                                    # atom 10
            'c_6',                                    # atom 11 ([*])
            '(_0', '[*]_0', ')_0',
            'c_6',                                    # atom 12 (junction)
            '2_0',
            'C_6',                                    # atom 13 (sp3 CH2)
            '1_0',
            'sep_0',
        ],
        "tok2atom": {
            1: 0, 3: 1, 4: 2, 7: 3,
            9: 4, 11: 5, 12: 6,
            13: 7, 15: 8, 16: 9, 17: 10, 18: 11,
            22: 12, 24: 13,
        },
    },

    # ---- D: azaindoline (pyridine-fused), [*] on aromatic C ----
    "D": {
        "embed_smiles": "C=CC(=O)N1Cc2cnccc2C1",
        "expected_with_star": "C=CC(=O)N1Cc2cncc([*])c2C1",
        "n_atoms_expected": 13,
        "labels": [
            'start_0',
            'C_0',                                    # atom 0
            '=_0',
            'C_0',                                    # atom 1
            'C_0',                                    # atom 2
            '(_0', '=_0', 'O_0', ')_0',               # atom 3
            'N_5',                                    # atom 4
            '1_0',
            'C_5',                                    # atom 5
            'c_5',                                    # atom 6 (junction)
            '2_0',
            'c_6',                                    # atom 7
            'n_6',                                    # atom 8 (pyridine N)
            'c_6',                                    # atom 9
            'c_6',                                    # atom 10 ([*])
            '(_0', '[*]_0', ')_0',
            'c_5',                                    # atom 11 (junction)
            '2_0',
            'C_5',                                    # atom 12 (sp3 CH)
            '1_0',
            'sep_0',
        ],
        "tok2atom": {
            1: 0, 3: 1, 4: 2, 7: 3,
            9: 4, 11: 5, 12: 6,
            14: 7, 15: 8, 16: 9, 17: 10,
            21: 11, 23: 12,
        },
    },
}


def _build_token_sequence(chassis_id: str):
    """Build (token_ids, labels, tok2atom, embed_smiles, expected_canonical)."""
    spec = CHASSIS_REGISTRY[chassis_id]
    labels = spec["labels"]
    tokens = [_V[t] for t in labels]
    tok2atom = spec["tok2atom"]
    embed_smiles = spec["embed_smiles"]
    expected_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(spec["expected_with_star"]))
    return tokens, labels, tok2atom, embed_smiles, expected_canonical, spec["n_atoms_expected"]


def _verify_chassis(chassis_id: str):
    """Decode-round-trip and aromaticity sanity check for the chassis."""
    tokens, labels, tok2atom, embed_smiles, expected, n_atoms_expected = \
        _build_token_sequence(chassis_id)

    pad_len = max(200, len(tokens) + 20)
    batch = np.array([tokens + [0] * (pad_len - len(tokens))])
    pos = np.zeros((1, pad_len, 3), dtype=np.float32)
    smi_chk, _, _ = _FU.decode3d(batch, pos)
    assert smi_chk and smi_chk[0] is not None, \
        f"[chassis {chassis_id}] scaffold tokens failed to decode"
    decoded = smi_chk[0]
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(decoded))
    print(f"[chassis {chassis_id}] decoded:   {decoded}")
    print(f"[chassis {chassis_id}] canonical: {canon}")
    print(f"[chassis {chassis_id}] expected:  {expected}")
    assert canon == expected, \
        f"[chassis {chassis_id}] round-trip mismatch:\n  got      {canon!r}\n  expected {expected!r}"

    check_mol = Chem.MolFromSmiles(decoded)
    star_atom = next((a for a in check_mol.GetAtoms() if a.GetSymbol() == '*'), None)
    assert star_atom is not None, f"[chassis {chassis_id}] no [*] in decoded prefix"
    star_nbrs = list(star_atom.GetNeighbors())
    assert len(star_nbrs) == 1, \
        f"[chassis {chassis_id}] expected 1 neighbor for [*], got {len(star_nbrs)}"
    attach = star_nbrs[0]
    print(f"[chassis {chassis_id}] [*] attaches to: {attach.GetSymbol()} "
          f"aromatic={attach.GetIsAromatic()} in_ring={attach.IsInRing()}")
    return tokens, labels, tok2atom, embed_smiles, n_atoms_expected


def _build_scaffold_xyz(chassis_id: str, anchor_json_path: str, pocket_center=None,
                        rng_seed: int = 0):
    """ETKDG-embed the heavy-atom-only scaffold, then translate+rotate so atom 0
    sits at cb_pos_target and the angle SG-atom0-atom1 equals the Bürgi-Dunitz
    angle (~107°) at atom 0 (the chemistry-standard convention for the
    electrophilic Cβ carbon).

    The previous (buggy) recipe aligned atom0->atom1 COLINEAR with the attack
    vector, forcing SG-atom0-atom1 = 180° (wrong for Michael addition).

    After the BD-correct placement, additionally rotate around the (atom0->atom1)
    bond axis (NOT around the attack vector — that would now break the BD
    angle) to push the body centroid (atoms 2..N) as close to pocket_center as
    possible. This keeps the scaffold inside the 0..240 grid for longer-bodied
    chassis like THIQ and amide-piperidine.

    Returns: xyz: (n_atoms, 3) np.float64 array in receptor (Å) frame.
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

    spec = CHASSIS_REGISTRY[chassis_id]
    embed_smiles = spec["embed_smiles"]
    n_atoms_expected = spec["n_atoms_expected"]
    mol = Chem.MolFromSmiles(embed_smiles)
    assert mol is not None, f"failed to parse {embed_smiles}"
    assert mol.GetNumAtoms() == n_atoms_expected, \
        f"expected {n_atoms_expected} atoms for chassis {chassis_id}, got {mol.GetNumAtoms()}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 + rng_seed
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        params.randomSeed = -1
        res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        AllChem.Compute2DCoords(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        print(f"[chassis {chassis_id}] MMFF relax failed (continuing): {e}")
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
    pre_scan_angle = measured_bd_angle(rotated, sg_pos)

    # In-plane scan: rotate around the (atom0 -> atom1) bond axis so the BD
    # angle SG-atom0-atom1 is preserved (the axis is collinear with both
    # endpoints of that angle, so rotation around it leaves both fixed).
    if pocket_center is not None and len(rotated) >= 4:
        best = rotated
        best_score = float('inf')

        bond_vec = rotated[1] - rotated[0]
        bvn = float(np.linalg.norm(bond_vec))
        if bvn < 1e-9:
            return rotated
        k = bond_vec / bvn  # rotation axis = bond axis

        for ang_deg in range(0, 360, 10):
            ang = math.radians(ang_deg)
            cos_a = math.cos(ang); sin_a = math.sin(ang)
            K2 = np.array([
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ])
            R2 = np.eye(3) * cos_a + sin_a * K2 + (1 - cos_a) * np.outer(k, k)
            # rotate around atom0 (anchor point) — preserves atom0 and the
            # direction of (atom0 -> atom1), hence preserves the BD angle.
            candidate = (rotated - cb_target) @ R2.T + cb_target
            body_cent = candidate[2:].mean(axis=0)
            score = float(np.linalg.norm(body_cent - pocket_center))
            if score < best_score:
                best_score = score
                best = candidate
        rotated = best

    measured = assert_bd_angle(rotated, sg_pos, bd_deg=bd_deg, tol_deg=5.0)
    print(f"[chassis {chassis_id}] BD angle at atom0 = {measured:.2f} deg "
          f"(target {bd_deg:.1f} deg; pre-scan {pre_scan_angle:.2f})")
    return rotated


def _build_initial_partial(chassis_id: str, anchor_json_path: str, pocket_center):
    """Build partial_product = [[token_list], [coords_per_token]] for the
    scaffold prefix.

    If the first ETKDG conformer + best in-plane rotation still produces grid
    coords outside [0, 240), retry with up to 8 alternative random-seed embeds.
    """
    tokens, labels, tok2atom, embed_smiles, n_atoms_expected = \
        _verify_chassis(chassis_id)
    pocket_center_arr = np.array(pocket_center, dtype=np.float64)
    best_xyz = None
    best_violation = float('inf')
    for seed in range(0, 9):
        xyz = _build_scaffold_xyz(chassis_id, anchor_json_path,
                                  pocket_center=pocket_center_arr, rng_seed=seed)
        grid_try = (xyz - pocket_center_arr) / 0.1 + 119.5
        # Violation = how far outside [0, 240) the worst voxel is.
        viol = max(
            max(0.0, -float(grid_try.min())),
            max(0.0, float(grid_try.max()) - 239.0),
        )
        if viol < best_violation:
            best_violation = viol
            best_xyz = xyz
        if viol == 0.0:
            print(f"[multi-anchor] seed={seed} produces in-bounds grid for chassis {chassis_id}")
            break
    xyz = best_xyz
    grid = (xyz - pocket_center_arr) / 0.1 + 119.5
    grid = grid.astype(np.float32)
    # Clip into model's [0, 239] voxel range to prevent get_partial_to_warehouse failures.
    grid = np.clip(grid, 0.0, 239.0)
    if best_violation > 0.0:
        print(f"[multi-anchor] WARNING: best ETKDG seed for chassis {chassis_id} still "
              f"violates grid by {best_violation:.2f} voxels — clipped to [0,239].")

    coords_per_token = []
    last_atom_grid = grid[0].tolist()
    for tok_idx in range(len(tokens)):
        if tok_idx in tok2atom:
            last_atom_grid = grid[tok2atom[tok_idx]].tolist()
        coords_per_token.append(list(last_atom_grid))

    return (
        [
            [list(tokens)],
            [list(coords_per_token)],
        ],
        xyz,
        grid,
        tokens,
        labels,
        tok2atom,
    )


def run(args):
    args.pocket_pdb  = _abs(args.pocket_pdb)
    args.output      = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.chassis_id not in CHASSIS_REGISTRY:
        raise SystemExit(
            f"--chassis_id must be one of {sorted(CHASSIS_REGISTRY.keys())}, "
            f"got {args.chassis_id!r}"
        )

    print(f"[multi-anchor] chassis={args.chassis_id}")
    print(f"[multi-anchor] pocket={args.pocket_pdb}")
    print(f"[multi-anchor] anchor={args.anchor_json}")
    print(f"[multi-anchor] gennums={args.gennums} min_acceptable={args.min_acceptable}")
    print(f"[multi-anchor] tempture={args.tempture}")

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

    initial_partial, scaffold_xyz, scaffold_grid, tokens, labels, tok2atom = \
        _build_initial_partial(args.chassis_id, args.anchor_json, pocket_center)
    print(f"[multi-anchor] pocket_center={pocket_center.tolist()}")
    print(f"[multi-anchor] scaffold prefix ({len(tokens)} tokens) = {tokens}")
    print(f"[multi-anchor] scaffold labels = {labels}")
    print(f"[multi-anchor] scaffold xyz (Å):")
    for i, p in enumerate(scaffold_xyz):
        print(f"    atom {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print(f"[multi-anchor] scaffold grid (voxel):")
    for i, p in enumerate(scaffold_grid):
        print(f"    atom {i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    if (scaffold_grid.min() < 0) or (scaffold_grid.max() >= 240):
        print(f"[multi-anchor] WARNING: grid coords outside [0,240): "
              f"min={scaffold_grid.min():.2f} max={scaffold_grid.max():.2f}. "
              f"Pocket center may not align with scaffold anchor.")
    print(f"[multi-anchor] seeded {len(initial_partial[0][0])} tokens + "
          f"{len(initial_partial[1][0])} per-token coords; "
          f"atom coords = {len(tok2atom)}")

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
        "plan": f"L2_multi_anchor_{args.chassis_id}_inpaint",
        "chassis_id": args.chassis_id,
        "embed_smiles": CHASSIS_REGISTRY[args.chassis_id]["embed_smiles"],
        "expected_with_star": CHASSIS_REGISTRY[args.chassis_id]["expected_with_star"],
        "scaffold_prefix_tokens": tokens,
        "scaffold_prefix_labels": labels,
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
                print(f"[multi-anchor] min_acceptable={args.min_acceptable} met with "
                      f"n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[multi-anchor] chassis={args.chassis_id} generated {len(results)} mols "
          f"in {elapsed:.1f}s status={status}")
    print(f"[multi-anchor] wrote SDF: {args.output}")

    summary.update({"n_sampled": len(results), "elapsed_sec": elapsed, "status": status})
    _flush_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chassis_id", required=True, choices=sorted(CHASSIS_REGISTRY.keys()),
                   help="One of A/B/C/D — selects which scaffold to inpaint.")
    p.add_argument("--anchor_json", default="data/lingo3dmol_anchor_zap70_cys346.json")
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--caption_path", default="checkpoint/gen_mol.pkl")
    p.add_argument("--gennums", type=int, default=30)
    p.add_argument("--min_acceptable", type=int, default=15)
    p.add_argument("--gen_frag_set", type=int, default=10)
    p.add_argument("--prod_time", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=0.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=1500)
    p.add_argument("--isTrain", action="store_true")
    p.add_argument("--USE_THRESHOLD", action="store_true", default=True)
    p.add_argument("--isMultiSample", action="store_true", default=True)
    p.add_argument("--isGuideSample", action="store_true", default=True)
    p.add_argument("--OnceMolGen", action="store_true")
    args = p.parse_args()
    run(args)
