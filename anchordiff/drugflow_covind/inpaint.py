"""DrugFlow covalent inpainting: soft joint pinning of warhead atoms + bonds.

Native DrugFlow has a flow over (x_lig, h_lig) and a Markov bridge over
bond_one_hot (e_lig). We hook the integration step (`sample_zt_given_zs`) to:

  1) Pull the warhead atoms' velocity prediction toward the velocity that
     would land them at the canonical position with Gaussian wobble σ_d.
     We compute the velocity-target = (canonical_pos - current_pos) / (1 - t)
     (the closed-form Markov-bridge-style velocity that hits the canonical
     point at t=1), and BLEND it with the model's predicted velocity by a
     schedule-weight `α_pos(t)`. α_pos ramps up early in the schedule (when
     positions are noisy) and ramps down near t=1 (so the model's final
     pose dominates and we avoid hard pinning).

  2) Bias the atom-type logits for the warhead atoms toward the canonical
     element one-hot (C, C, C, O, N) via additive log-bias `α_h(t) * λ * z1`.
     Same schedule shape.

  3) Bias the bond-type logits for the 4 warhead bonds toward their target
     classes (DOUBLE, SINGLE, DOUBLE, SINGLE). This is the KEY novel piece —
     DrugFlow's bond Markov bridge is a NAMED LATENT VARIABLE so we can
     condition on it directly, no post-hoc OpenBabel inference.

The schedule weight α(t) = strength * (1 - t)^pow_anneal (default pow=1.0).
  - strength=1.0 means "fully overwrite at t=0, no bias at t=1" — too hard,
    breaks gradient flow into the model's own prediction. Default 0.7.
  - For HARD pin (smoke test), use strength=8.0 with high log-bias λ_logit
    so the warhead atoms snap to canonical at every step.

CRITICAL: the canonical positions live in the GLOBAL frame of the input
pocket (we transform from the local Cys-SG frame to global using `compute_frame`
on the input Cys-SG/CB/CA atoms — so the frame is derived from INPUT atoms
only, never hardcoded; SE(3) equivariance is preserved as long as the pocket
input is consistent).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

# anchordiff (local-frame, token) — reuse without modification
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from anchordiff.covind.local_frame import compute_frame, to_global

# DrugFlow imports (resolved at runtime — only available on V100)
DRUGFLOW_ROOT = None
for cand in [Path.home() / "DrugFlow", Path("/home/shaharh_quris_ai/DrugFlow")]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DRUGFLOW_ROOT = cand
        break


# --- Warhead canonical specification ----------------------------------------
# Acrylamide warhead: Cβ=Cα-C(=O)-N. Atom ordering MUST match the SDF used
# downstream (warhead.sdf at ~/edit-small-mol/anchordiff/pockets/zap70_cys346).
WARHEAD_ATOM_ELEMENTS = ["C", "C", "C", "O", "N"]   # length 5

# Bonds as (i, j, bond_type_name). Indices into the warhead atom list above.
WARHEAD_BONDS = [
    (0, 1, "DOUBLE"),   # Cβ=Cα
    (1, 2, "SINGLE"),   # Cα-Ccarb
    (2, 3, "DOUBLE"),   # Ccarb=O
    (2, 4, "SINGLE"),   # Ccarb-N
]


def load_warhead_canonical(sdf_path: str) -> np.ndarray:
    """Load (5, 3) canonical positions (in the SDF's own frame) for the
    warhead. Validates atom count and elements match the spec.
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break
    if mol is None:
        raise RuntimeError(f"Could not read warhead SDF: {sdf_path}")
    conf = mol.GetConformer()
    atoms = list(mol.GetAtoms())
    assert len(atoms) == len(WARHEAD_ATOM_ELEMENTS), \
        f"warhead.sdf has {len(atoms)} atoms, expected {len(WARHEAD_ATOM_ELEMENTS)}"
    elems = [a.GetSymbol() for a in atoms]
    assert elems == WARHEAD_ATOM_ELEMENTS, f"warhead elements {elems} != {WARHEAD_ATOM_ELEMENTS}"
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(len(atoms))], dtype=np.float64)
    return pos


@dataclass
class InpaintSpec:
    """Specification for a single covalent inpaint job.

    canonical_global_pos: (5, 3) canonical warhead positions in the GLOBAL
        coordinate frame of the *pocket input* (already transformed if needed).
        We pin against these.
    warhead_atom_idx: list of 5 ints into the ligand atoms (which sampled
        atoms to pin). For DrugFlow's default unconditional init, the ligand
        atoms are unordered/random so we choose the first 5 sampled indices
        of the smallest sample (warhead atoms are at indices 0..4 by convention).
    """
    canonical_global_pos: np.ndarray        # (5, 3)
    warhead_atom_indices_per_sample: list   # list[list[int]], len n_samples
    bond_targets: list                       # list[(i_lig, j_lig, bond_idx)]
    sigma_d: float = 0.1                     # Å, positional Gaussian wobble
    strength_pos: float = 0.7                # 0..N, blend factor for vel.
    # 2026-05-25 NOTE: I tried 0.7→2.0 to tighten warhead position, but QA agent
    # flagged that aggressive position pull combined with a softly-degraded
    # finetuned model overwhelms the model's own velocity, causing "dust + warhead"
    # collapse. Reverted to 0.7. Strength_pos sweep (0.3 / 0.5 / 0.7 / 1.0 / 2.0)
    # is part of Day-2 ablations — test on BASE model vs intermediate finetuned ckpts.
    strength_h: float = 5.0                  # additive log-bias for atom types
    strength_e: float = 5.0                  # additive log-bias for bond types
    pow_anneal: float = 1.0                  # exponent on (1-t) schedule


def build_inpaint_spec_for_zap70(
    warhead_sdf: str,
    receptor_pdb: str,
    cys_chain: str,
    cys_resi: int,
    n_samples: int,
    sigma_d: float = 0.1,
    strength_pos: float = 0.7,
    strength_h: float = 5.0,
    strength_e: float = 5.0,
) -> InpaintSpec:
    """Build the InpaintSpec for ZAP70 Cys346.

    The warhead.sdf is in its own local frame (centered however it was
    extracted). We need its GLOBAL position relative to the input receptor.
    Strategy: re-place the warhead by snapping atom 0 (Cβ) onto the canonical
    1.85 Å offset from Cys-SG along the SG→CB axis (i.e., use Cys-SG/CB from
    the input PDB to define the frame, then place warhead at canonical local
    coords (0, 0, 1.85) for Cβ + rest from the SDF's relative offsets to atom 0).
    This keeps the frame derived from INPUT atoms only.
    """
    from Bio.PDB import PDBParser

    # 1) Load warhead in its own SDF frame
    warhead_local_raw = load_warhead_canonical(warhead_sdf)  # (5, 3)
    # Re-center so atom 0 (Cβ) is at origin
    warhead_centered = warhead_local_raw - warhead_local_raw[0:1]

    # 2) Build the SE(3) local frame from Cys-SG/CB/CA in the receptor
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("", str(receptor_pdb))[0]
    cys = s[cys_chain][int(cys_resi)]
    sg = np.array(cys["SG"].get_coord(), dtype=float)
    cb = np.array(cys["CB"].get_coord(), dtype=float)
    ca = np.array(cys["CA"].get_coord(), dtype=float)
    R, t = compute_frame(sg, cb, ca)

    # 3) Place atom 0 (Cβ) at canonical d=1.85 Å along +z (= SG→CB direction)
    #    Build warhead in local frame: atom 0 at (0,0,1.85), and rotate the
    #    rest so the Cβ=Cα bond direction lies along an arbitrary in-plane
    #    direction. Easiest: take the SDF-relative geometry of the warhead
    #    and ROTATE so that atom1-atom0 (Cα→Cβ) points along the -z axis
    #    (toward SG) — then atom 0 ends up "pointing into" the pocket from SG.
    # Compute SDF's Cα→Cβ direction:
    v_alpha_to_beta = warhead_local_raw[0] - warhead_local_raw[1]
    v_alpha_to_beta = v_alpha_to_beta / (np.linalg.norm(v_alpha_to_beta) + 1e-9)
    # Target direction in local frame: -z (so Cα is on +z side, attacked carbon
    # is the one closer to the pocket — but we want Cβ at the END nearer SG, so
    # Cα→Cβ direction is along -z when projected into local frame).
    target_dir = np.array([0.0, 0.0, -1.0])
    # Rotation that maps v_alpha_to_beta → target_dir
    axis = np.cross(v_alpha_to_beta, target_dir)
    s_ax = np.linalg.norm(axis)
    if s_ax < 1e-6:
        # already aligned (or anti-aligned)
        if np.dot(v_alpha_to_beta, target_dir) > 0:
            R_align = np.eye(3)
        else:
            R_align = np.diag([1.0, -1.0, -1.0])
    else:
        c_ax = np.dot(v_alpha_to_beta, target_dir)
        ax = axis / s_ax
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R_align = np.eye(3) + s_ax * K + (1 - c_ax) * (K @ K)
    warhead_aligned = warhead_centered @ R_align.T  # (5, 3) in a frame where Cα→Cβ ≈ -z

    # Shift atom 0 to (0, 0, 1.85)
    warhead_local = warhead_aligned + np.array([0.0, 0.0, 1.85])

    # 4) Transform to global
    canonical_global = to_global(warhead_local, R, t)  # (5, 3)

    # 5) Bond targets in the bond_encoder index space
    from src.constants import bond_encoder
    bond_targets = []
    for (i, j, bname) in WARHEAD_BONDS:
        bond_targets.append((i, j, bond_encoder[bname]))

    # 6) For each sample, the warhead atoms are at the FIRST 5 ligand atom
    # indices (a convention; init_ligand returns random atoms so we just pick
    # the first 5 per sample). Sample masks tell us which atoms belong to
    # which molecule.
    return InpaintSpec(
        canonical_global_pos=canonical_global,
        warhead_atom_indices_per_sample=[[0, 1, 2, 3, 4] for _ in range(n_samples)],
        bond_targets=bond_targets,
        sigma_d=sigma_d,
        strength_pos=strength_pos,
        strength_h=strength_h,
        strength_e=strength_e,
    )


# ----- The actual integration hook -----------------------------------------
def make_inpainted_sample_zt_given_zs(model, spec: InpaintSpec):
    """Wrap model.sample_zt_given_zs to apply soft warhead pinning.

    Returns a closure that replaces `model.sample_zt_given_zs`.
    """
    from src.constants import atom_encoder, bond_encoder
    original = model.sample_zt_given_zs

    # Build per-warhead-atom global canonical position lookup as a Tensor.
    # We will compute the "true" ligand-atom index per sample at first call,
    # since DrugFlow's batched mask tells us which atoms belong to which sample.
    canonical = torch.tensor(spec.canonical_global_pos, dtype=torch.float32)  # (5, 3)
    warhead_elem_idx = torch.tensor(
        [atom_encoder[e] for e in WARHEAD_ATOM_ELEMENTS], dtype=torch.long
    )  # (5,) class indices
    # Bond targets: list of (i_local, j_local, bond_class)
    bond_target_idx = torch.tensor(
        [b[2] for b in spec.bond_targets], dtype=torch.long
    )  # (num_warhead_bonds,)
    n_warhead = len(WARHEAD_ATOM_ELEMENTS)  # 5

    def hooked(zs_ligand, zs_pocket, s, t, delta_eps_x=None, uncertainty=None):
        # Call the original first — but we need to intercept the model's
        # prediction. We replicate the inner logic here to inject biases.
        sc_transform = model.get_sc_transform_fn(
            zs_pocket.get('chi'), zs_ligand['x'], s, None,
            zs_ligand['mask'], zs_pocket)
        pred_ligand, pred_residues = model.dynamics(
            zs_ligand['x'], zs_ligand['h'], zs_ligand['mask'], zs_pocket, s,
            bonds_ligand=(zs_ligand['bonds'], zs_ligand['e']),
            sc_transform=sc_transform,
        )
        if delta_eps_x is not None:
            pred_ligand['vel'] = pred_ligand['vel'] + delta_eps_x

        # --- COVALENT INPAINT HOOK ---
        device = zs_ligand['x'].device
        canon_d = canonical.to(device)
        warhead_elem_d = warhead_elem_idx.to(device)
        bond_target_d = bond_target_idx.to(device)
        t_scalar = float(t.max().item())  # all samples share t; safe to take scalar
        # Schedule: anneal is HIGH at both ends and decays only near t≈0.95
        # (we want the warhead pinned essentially throughout the trajectory;
        # only release in the very last few steps so the model can fine-tune
        # the local geometry). Previous schedule (1-t)^1.0 decayed too fast.
        # Use a piecewise: 1.0 for t<0.7, linearly decay 1.0→0.3 for 0.7≤t≤0.95,
        # then 0.3 for the rest.
        if t_scalar < 0.7:
            anneal = 1.0
        elif t_scalar < 0.95:
            anneal = 1.0 - (t_scalar - 0.7) / 0.25 * 0.7
        else:
            anneal = 0.3

        # Per-sample warhead atom indices (assume first 5 ligand atoms of each sample)
        lig_mask = zs_ligand['mask']           # (N_lig,) sample-id per atom
        sample_ids = torch.unique(lig_mask)
        warhead_global_idx = []
        for sid in sample_ids:
            atom_idx_in_mask = (lig_mask == sid).nonzero(as_tuple=True)[0]
            if len(atom_idx_in_mask) >= n_warhead:
                warhead_global_idx.append(atom_idx_in_mask[:n_warhead])
            else:
                # Sample too small to host warhead — skip
                pass
        if len(warhead_global_idx) > 0:
            warhead_gidx = torch.cat(warhead_global_idx, dim=0)  # (5 * n_samples,)
            # 1) POSITION inpaint: pull vel toward (canonical - current) / (1 - t)
            # Robust against t→1: clamp 1-t at a small ε so the velocity stays finite.
            current_pos = zs_ligand['x'][warhead_gidx]                # (5*ns, 3)
            # Broadcast canonical across samples
            canon_repeat = canon_d.unsqueeze(0).expand(
                len(sample_ids), -1, -1).reshape(-1, 3)                # (5*ns, 3)
            # Add Gaussian noise σ_d to canonical (soft pin)
            jitter = torch.randn_like(canon_repeat) * spec.sigma_d
            canon_jittered = canon_repeat + jitter
            denom = max(1.0 - t_scalar, 1e-3)
            vel_target = (canon_jittered - current_pos) / denom        # (5*ns, 3)
            # Blend: vel := (1-α) * pred + α * target
            alpha_pos = min(spec.strength_pos * anneal, 0.95)
            pred_ligand['vel'][warhead_gidx] = (
                (1.0 - alpha_pos) * pred_ligand['vel'][warhead_gidx]
                + alpha_pos * vel_target
            )

            # 2) ATOM-TYPE inpaint: bias logits_h toward target element
            #    via additive log-bias. logits[i, class] += λ * (1-t) * onehot[class]
            warhead_elem_rep = warhead_elem_d.unsqueeze(0).expand(
                len(sample_ids), -1).reshape(-1)                       # (5*ns,)
            elem_oh = F.one_hot(warhead_elem_rep,
                                num_classes=pred_ligand['logits_h'].size(-1)).float()
            pred_ligand['logits_h'][warhead_gidx] = (
                pred_ligand['logits_h'][warhead_gidx]
                + spec.strength_h * anneal * elem_oh
            )

        # 3) BOND-TYPE inpaint via the Markov bridge logits
        if len(warhead_global_idx) > 0:
            edge_mask = zs_ligand['edge_mask']                          # (N_edges,)
            bonds_idx = zs_ligand['bonds']                              # (2, N_edges)
            # For each sample, find the edges that connect (sample_warhead_atoms)
            for s_pos, sid in enumerate(sample_ids):
                edge_in_sample = (edge_mask == sid).nonzero(as_tuple=True)[0]
                if len(edge_in_sample) == 0:
                    continue
                wh_atoms = warhead_global_idx[s_pos]                    # (5,) global atom ids
                wh_set = set(int(x) for x in wh_atoms.tolist())
                # For each warhead bond (i_local, j_local, target_class), find the edge
                # global-index whose endpoints map to (wh_atoms[i_local], wh_atoms[j_local])
                # — and edit the bond logit.
                bond_logits_e = pred_ligand['logits_e']
                for k, (i_loc, j_loc, _bname) in enumerate(WARHEAD_BONDS):
                    g_i = int(wh_atoms[i_loc])
                    g_j = int(wh_atoms[j_loc])
                    # bonds in DrugFlow are upper-triangular: (i, j) with i < j
                    g_lo, g_hi = min(g_i, g_j), max(g_i, g_j)
                    src = bonds_idx[0, edge_in_sample]
                    dst = bonds_idx[1, edge_in_sample]
                    match = ((src == g_lo) & (dst == g_hi)).nonzero(as_tuple=True)[0]
                    if len(match) == 0:
                        continue
                    edge_g = edge_in_sample[match[0]]
                    target_cls = int(bond_target_d[k])
                    oh = torch.zeros_like(bond_logits_e[edge_g])
                    oh[target_cls] = 1.0
                    pred_ligand['logits_e'][edge_g] = (
                        pred_ligand['logits_e'][edge_g] + spec.strength_e * anneal * oh
                    )

                # QA fix 2026-05-25: bias non-warhead-internal edges touching
                # warhead atoms toward NO_BOND. Without this, warhead Cβ/Cα/etc
                # accept additional bonds from non-warhead neighbors, producing
                # valence-6 carbons → 22/25 sanitize failures.
                # Allow exactly ONE outgoing bond from Cβ (atom local idx 0, the
                # growth point — Michael β-carbon will be attacked by Cys-Sγ in
                # the protein but for now we let one extra ligand bond grow off
                # it). Block all other warhead-touching non-internal edges.
                NO_BOND_CLS = 0  # DrugFlow's bond_encoder[None]/[BondType.NONE]
                cb_global = int(wh_atoms[0])
                warhead_internal = set()
                for (i_loc, j_loc, _) in WARHEAD_BONDS:
                    g_i, g_j = int(wh_atoms[i_loc]), int(wh_atoms[j_loc])
                    warhead_internal.add((min(g_i, g_j), max(g_i, g_j)))
                # Track edges already permitted out of Cβ (we allow 1)
                cb_growth_count = 0
                CB_GROWTH_MAX = 1
                for e_idx_in_sample in edge_in_sample:
                    src_a = int(bonds_idx[0, e_idx_in_sample])
                    dst_a = int(bonds_idx[1, e_idx_in_sample])
                    key = (min(src_a, dst_a), max(src_a, dst_a))
                    if key in warhead_internal:
                        continue  # internal warhead bonds already pinned above
                    # Does this edge touch a warhead atom?
                    touches_wh = (src_a in wh_set or dst_a in wh_set)
                    if not touches_wh:
                        continue
                    touches_cb = (src_a == cb_global or dst_a == cb_global)
                    if touches_cb and cb_growth_count < CB_GROWTH_MAX:
                        # Allow the first non-internal outgoing bond from Cβ
                        # (growth direction) — don't bias
                        cb_growth_count += 1
                        continue
                    # Bias toward NO_BOND
                    oh_no = torch.zeros_like(bond_logits_e[e_idx_in_sample])
                    oh_no[NO_BOND_CLS] = 1.0
                    pred_ligand['logits_e'][e_idx_in_sample] = (
                        pred_ligand['logits_e'][e_idx_in_sample]
                        + spec.strength_e * anneal * oh_no
                    )
        # ----- END HOOK -----

        # Replicate downstream from original sample_zt_given_zs
        zt_ligand = zs_ligand.copy()
        zt_ligand['x'] = model.module_x.sample_zt_given_zs(
            zs_ligand['x'], pred_ligand['vel'], s, t, zs_ligand['mask'])
        zt_ligand['h'] = model.module_h.sample_zt_given_zs(
            zs_ligand['h'], pred_ligand['logits_h'], s, t, zs_ligand['mask'])
        zt_ligand['e'] = model.module_e.sample_zt_given_zs(
            zs_ligand['e'], pred_ligand['logits_e'], s, t, zs_ligand['edge_mask'])
        # POST-SAMPLE HARD OVERRIDE for warhead bond/atom states. The Markov
        # bridge's β-gated transition prevents fast jumps even with maxed-out
        # logits — we override the resulting categorical state directly for
        # the warhead atoms + bonds (with strength_e>=1.0 we mean "hard").
        # This is still a soft pin in terms of POSITIONS (we used a velocity
        # blend with strength_pos), so the model retains control of fine
        # placement while we lock the chemistry.
        if len(warhead_global_idx) > 0 and spec.strength_e >= 1.0:
            # Atom-type override (one-hot at target element class)
            for s_pos, sid in enumerate(sample_ids):
                wh_atoms = warhead_global_idx[s_pos]
                # Atoms
                if spec.strength_h >= 1.0:
                    new_h = torch.zeros_like(zt_ligand['h'][wh_atoms])
                    elem_repeat = warhead_elem_d
                    new_h.scatter_(1, elem_repeat.unsqueeze(1), 1.0)
                    zt_ligand['h'][wh_atoms] = new_h
                # Bonds
                edge_in_sample = (zs_ligand['edge_mask'] == sid).nonzero(as_tuple=True)[0]
                src = zs_ligand['bonds'][0, edge_in_sample]
                dst = zs_ligand['bonds'][1, edge_in_sample]
                for k, (i_loc, j_loc, _bname) in enumerate(WARHEAD_BONDS):
                    g_i = int(wh_atoms[i_loc])
                    g_j = int(wh_atoms[j_loc])
                    g_lo, g_hi = min(g_i, g_j), max(g_i, g_j)
                    match = ((src == g_lo) & (dst == g_hi)).nonzero(as_tuple=True)[0]
                    if len(match) == 0:
                        continue
                    edge_g = edge_in_sample[match[0]]
                    target_cls = int(bond_target_d[k])
                    oh = torch.zeros_like(zt_ligand['e'][edge_g])
                    oh[target_cls] = 1.0
                    zt_ligand['e'][edge_g] = oh

        zt_pocket = zs_pocket.copy()
        if model.flexible_bb:
            zt_trans = model.module_trans.sample_zt_given_zs(
                zs_pocket['x'], pred_residues['trans'], s, t, zs_pocket['mask'])
            zt_rot = model.module_rot.sample_zt_given_zs(
                zs_pocket['axis_angle'], pred_residues['rot'], s, t, zs_pocket['mask'])
            zt_pocket.set_frame(zt_trans, zt_rot)
        if model.flexible:
            zt_chi = model.module_chi.sample_zt_given_zs(
                zs_pocket['chi'][..., :5], pred_residues['chi'], s, t, zs_pocket['mask'])
            zt_pocket.set_chi(zt_chi)

        if model.predict_confidence:
            from torch.distributions.categorical import Categorical
            assert uncertainty is not None
            dt = (t - s).view(-1)[zt_ligand['mask']]
            uncertainty['sigma_x_squared'] += (dt * pred_ligand['uncertainty_vel']**2)
            uncertainty['entropy_h'] += (
                dt * Categorical(logits=pred_ligand['logits_h']).entropy())

        return zt_ligand, zt_pocket

    return hooked


def run_inpaint(
    checkpoint: str,
    receptor_pdb: str,
    ref_ligand_sdf: str,
    warhead_sdf: str,
    cys_chain: str,
    cys_resi: int,
    output_sdf: str,
    n_samples: int = 25,
    batch_size: int = 25,
    n_steps: int = 100,
    sigma_d: float = 0.1,
    strength_pos: float = 0.7,
    strength_h: float = 5.0,
    strength_e: float = 5.0,
    device: str = "cuda:0",
    pocket_distance_cutoff: float = 8.0,
    molecule_size: int = 25,
    datadir: Optional[str] = None,
    seed: int = 42,
):
    """End-to-end: load DrugFlow, build inpaint spec, sample with hook, write SDF."""
    from Bio.PDB import PDBParser
    from torch.utils.data import DataLoader
    from functools import partial
    from src.data.dataset import ProcessedLigandPocketDataset
    from src.data.data_utils import TensorDict, process_raw_pair
    from src.model.lightning import DrugFlow
    from src import utils as drugflow_utils

    drugflow_utils.set_deterministic(seed=seed)
    drugflow_utils.disable_rdkit_logging()

    model = DrugFlow.load_from_checkpoint(checkpoint, map_location=device, strict=False)
    if datadir is not None:
        model.datadir = Path(datadir)
    model.setup(stage="generation")
    model.batch_size = model.eval_batch_size = batch_size
    model.eval().to(device)
    model.T = n_steps

    # Build the input batch with `process_raw_pair`
    pdb_model = PDBParser(QUIET=True).get_structure("", receptor_pdb)[0]
    rdmol = Chem.SDMolSupplier(str(ref_ligand_sdf))[0]
    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=pocket_distance_cutoff,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=receptor_pdb if model.dynamics.add_nma_feat else None,
    )
    ligand["name"] = "ligand"
    dataset = [{"ligand": ligand, "pocket": pocket} for _ in range(batch_size)]
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
        pin_memory=True,
    )

    # Build inpaint spec (uses the same Cys-SG/CB/CA atoms as input — frame derived from input)
    spec = build_inpaint_spec_for_zap70(
        warhead_sdf=warhead_sdf,
        receptor_pdb=receptor_pdb,
        cys_chain=cys_chain,
        cys_resi=cys_resi,
        n_samples=batch_size,
        sigma_d=sigma_d,
        strength_pos=strength_pos,
        strength_h=strength_h,
        strength_e=strength_e,
    )
    print(f"[inpaint] canonical_global Cβ position: {spec.canonical_global_pos[0]}")

    # Hook the integration step
    model.sample_zt_given_zs = make_inpainted_sample_zt_given_zs(model, spec)

    # Sample
    all_mols = []
    n_done = 0
    while n_done < n_samples:
        for data in dataloader:
            new_data = {
                "ligand": TensorDict(**data["ligand"]).to(device),
                "pocket": TensorDict(**data["pocket"]).to(device),
            }
            rdmols, _, _ = model.sample(
                new_data, n_samples=1, timesteps=n_steps, num_nodes=molecule_size)
            for m in rdmols:
                if m is not None:
                    all_mols.append(m)
                    n_done += 1
                if n_done >= n_samples:
                    break
            if n_done >= n_samples:
                break

    Path(output_sdf).parent.mkdir(parents=True, exist_ok=True)
    drugflow_utils.write_sdf_file(output_sdf, all_mols[:n_samples])
    print(f"[inpaint] wrote {len(all_mols[:n_samples])} mols to {output_sdf}")

    # ---- POST-HOC AUDIT (no fixing! just measurement) -----
    audit = audit_warhead(all_mols[:n_samples], spec)
    print(f"[inpaint] audit: {audit}")
    return all_mols[:n_samples], audit


def audit_warhead(rdmols, spec: InpaintSpec) -> dict:
    """Measure WITHOUT fixing: how close did the first 5 atoms of each
    output mol land to the canonical Cβ=Cα-C(=O)-N pattern?

    Returns:
      mean_d_to_canonical: Å, position error
      pct_correct_elements: fraction of mols whose first 5 atoms are CCCON
      pct_correct_bonds_DSDS: fraction whose 4 warhead bonds match
        (DOUBLE, SINGLE, DOUBLE, SINGLE)
    """
    from src.constants import bond_encoder, atom_encoder
    target_elems = WARHEAD_ATOM_ELEMENTS
    target_bonds = [b[2] for b in WARHEAD_BONDS]
    n_ok = 0; n_elem_ok = 0; n_bond_ok = 0
    sum_d = 0.0
    for m in rdmols:
        if m is None or m.GetNumAtoms() < 5:
            continue
        n_ok += 1
        # Element check
        elems = [m.GetAtomWithIdx(i).GetSymbol() for i in range(5)]
        if elems == target_elems:
            n_elem_ok += 1
        # Position check
        conf = m.GetConformer()
        positions = np.array([list(conf.GetAtomPosition(i)) for i in range(5)])
        d = np.linalg.norm(positions - spec.canonical_global_pos, axis=1).mean()
        sum_d += d
        # Bond check
        bond_types_present = []
        for (i_loc, j_loc, target_name) in WARHEAD_BONDS:
            b = m.GetBondBetweenAtoms(i_loc, j_loc)
            if b is None:
                bond_types_present.append("NOBOND")
            else:
                bond_types_present.append(str(b.GetBondType()).split(".")[-1])
        if bond_types_present == [b[2] for b in WARHEAD_BONDS]:
            n_bond_ok += 1
    return {
        "n_mols": n_ok,
        "mean_d_to_canonical": sum_d / max(n_ok, 1),
        "pct_correct_elements": n_elem_ok / max(n_ok, 1),
        "pct_correct_bonds": n_bond_ok / max(n_ok, 1),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ref_ligand", required=True)
    ap.add_argument("--warhead", required=True)
    ap.add_argument("--cys_chain", default="A")
    ap.add_argument("--cys_resi", type=int, default=346)
    ap.add_argument("--output", default="samples.sdf")
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--sigma_d", type=float, default=0.1)
    ap.add_argument("--strength_pos", type=float, default=0.7)
    ap.add_argument("--strength_h", type=float, default=5.0)
    ap.add_argument("--strength_e", type=float, default=5.0)
    ap.add_argument("--molecule_size", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_inpaint(
        checkpoint=args.checkpoint,
        receptor_pdb=args.receptor,
        ref_ligand_sdf=args.ref_ligand,
        warhead_sdf=args.warhead,
        cys_chain=args.cys_chain,
        cys_resi=args.cys_resi,
        output_sdf=args.output,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        sigma_d=args.sigma_d,
        strength_pos=args.strength_pos,
        strength_h=args.strength_h,
        strength_e=args.strength_e,
        molecule_size=args.molecule_size,
        device=args.device,
        datadir=args.datadir,
        seed=args.seed,
    )
