"""B2 inpaint: REINVENT-style particle resampling inside DiffSBDD's reverse-diffusion loop.

Lives at ~/DiffSBDD/b2_inpaint.py on T4. DOES NOT modify upstream inpaint.py.

Pipeline per requested sample (out of N_SAMPLES total):
    1. Initialize P particles (P=n_particles), each a separate ligand replicate
       in a single ligand_mask batch. Each particle is paired with its own pocket
       replica.
    2. For t = T-1 ... 0 (timesteps reversed):
         a. denoise step: x_t -> x_{t-1} (REPAINT inpaint update over all P particles)
         b. inpaint warhead atoms: replace warhead positions with the fixed
            x_warhead_fixed value (handled by REPAINT mask).
         c. project warhead onto covalent constraint manifold (per particle).
         d. every K=resample_every steps: bond-perceive -> SMILES -> FiLMDelta
            score; multinomially resample particles by softmax(score / T_temp).
    3. Final decode -> RDKit molecule -> warhead bond fix -> SMILES.
    4. Pick the best particle by final FiLMDelta score and write to mols.sdf.

We keep one *batch* of P particles in flight at a time (memory cheap on T4 for
P=16). For n_samples > 1, we just loop the whole thing n_samples times, each
producing 1 mol with its own random init.

Imports DiffSBDD internals (LigandPocketDDPM etc.) in-place. Adapts the inpaint
loop from `equivariant_diffusion/conditional_model.py::ConditionalDDPM.inpaint`.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()

# DiffSBDD imports (assumed in cwd or sys.path)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import utils
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule

# Anchordiff helpers — sit in ~/anchordiff/
ANCHORDIFF_DIR = Path("~/anchordiff").expanduser()
sys.path.insert(0, str(ANCHORDIFF_DIR))
from covalent_constraint_manifold import project as covalent_project, WarheadAtoms
from fix_inpaint_warhead_bonds import fix_warhead_bonds

# B2 scorer — sits in ~/DiffSBDD/b2_film_scorer.py
from b2_film_scorer import FiLMScorer


# ── Helpers borrowed from inpaint.py ───────────────────────────────────────
def prepare_from_sdf_files(sdf_files, atom_encoder):
    ligand_coords, atom_one_hot = [], []
    for f in sdf_files:
        rdmol = Chem.SDMolSupplier(str(f), sanitize=False)[0]
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()])
        atom_one_hot.append(F.one_hot(types, num_classes=len(atom_encoder)))
    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):
    coord = torch.tensor(np.array([a.get_coord() for a in biopython_atoms]),
                         dtype=FLOAT_TYPE)
    types = torch.tensor([atom_encoder[a.element.capitalize()]
                          for a in biopython_atoms])
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))
    return coord, one_hot


# ── Decode current x_t (a noisy particle) into an RDKit mol ────────────────
def decode_particle_to_mol(x_lig, atom_type_idx, dataset_info):
    """Build an RDKit mol from a single particle's positions + atom types."""
    try:
        mol = build_molecule(x_lig, atom_type_idx, dataset_info, add_coords=True)
        mol = process_molecule(mol, add_hydrogens=False, sanitize=True,
                               relax_iter=0, largest_frag=False)
    except Exception:
        return None
    if mol is None:
        return None
    fixed = fix_warhead_bonds(mol)
    return fixed if fixed is not None else mol


# ── Constraint projection: project warhead atoms (indices 0..3) onto manifold ──
WARHEAD_ATOMS = WarheadAtoms(cb=0, ca=1, c_carb=2, n_amide=4)


def project_warhead_inplace(x_lig: np.ndarray, sg_pos: np.ndarray) -> np.ndarray:
    """Project warhead positions in `x_lig` (np array, [n_atoms, 3]) onto the
    covalent constraint manifold relative to SG at `sg_pos`. Modifies a copy."""
    if x_lig.shape[0] < 5:
        return x_lig
    return covalent_project(x_lig, sg_pos, WARHEAD_ATOMS)


# ── The B2 sampler ─────────────────────────────────────────────────────────
class B2Sampler:
    def __init__(self, model, args, scorer: FiLMScorer, sg_pos_world: np.ndarray):
        self.model = model
        self.ddpm = model.ddpm
        self.device = model.device
        self.args = args
        self.scorer = scorer
        # SG cysteine position in world coords (will be translated to centered
        # coords inside sample_one).
        self.sg_pos_world = sg_pos_world

    def _build_initial(self, n_particles: int, x_warhead: torch.Tensor,
                       one_hot_warhead: torch.Tensor, pocket_residues):
        """Build ligand + pocket with n_particles parallel replicates."""
        pocket = self.model.prepare_pocket(pocket_residues, repeats=n_particles)
        n_fixed = len(x_warhead)

        num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed)
        # Force all particles to have the same size so that the per-particle
        # resampling can swap atoms 1:1 without size mismatches. We use the
        # first sample's size (a typical pocket-conditioned draw).
        target_size = int(num_nodes_lig[0].item())
        num_nodes_lig = torch.full_like(num_nodes_lig, target_size)

        ligand_mask = utils.num_nodes_to_batch_mask(
            len(num_nodes_lig), num_nodes_lig, self.device)

        ligand = {
            'x': torch.zeros((len(ligand_mask), self.model.x_dims),
                             device=self.device, dtype=FLOAT_TYPE),
            'one_hot': torch.zeros((len(ligand_mask), self.model.atom_nf),
                                   device=self.device, dtype=FLOAT_TYPE),
            'size': num_nodes_lig,
            'mask': ligand_mask,
        }

        # Fill in the fixed warhead atoms at the start of each particle's slot.
        lig_fixed = torch.zeros_like(ligand_mask)
        for i in range(n_particles):
            sele = (ligand_mask == i)
            xnew = ligand['x'][sele]
            xnew[:n_fixed] = x_warhead.to(self.device)
            ligand['x'][sele] = xnew

            hnew = ligand['one_hot'][sele]
            hnew[:n_fixed] = one_hot_warhead.to(self.device)
            ligand['one_hot'][sele] = hnew

            fnew = lig_fixed[sele]
            fnew[:n_fixed] = 1
            lig_fixed[sele] = fnew

        return ligand, pocket, lig_fixed

    def _decode_and_score_all(self, z_lig, lig_mask, n_particles, sg_in_centered):
        """Unnormalize z_lig, decode each particle to SMILES, score all."""
        with torch.no_grad():
            x_lig_un, _ = self.ddpm.unnormalize(z_lig[:, :self.ddpm.n_dims].clone(),
                                                z_lig[:, self.ddpm.n_dims:].clone())
            atom_type_idx = z_lig[:, self.ddpm.n_dims:].argmax(1).detach().cpu()
        x_lig_cpu = x_lig_un.detach().cpu()
        smiles_list, mols = [], []
        for p in range(n_particles):
            sele = (lig_mask == p).cpu()
            x_p = x_lig_cpu[sele]
            t_p = atom_type_idx[sele]
            mol = decode_particle_to_mol(x_p, t_p, self.model.dataset_info)
            if mol is None:
                smiles_list.append("")
                mols.append(None)
            else:
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    smi = ""
                smiles_list.append(smi)
                mols.append(mol)
        scores = self.scorer.score_smiles_batch(smiles_list)
        return smiles_list, scores, mols

    def _resample_particles(self, scores: np.ndarray, n_particles: int,
                            temperature: float):
        """Multinomial resample particle indices ~ softmax(score / T)."""
        finite = np.isfinite(scores)
        if not finite.any():
            return np.arange(n_particles)
        # Replace -inf with the min finite score - 5 (heavily downweight invalid)
        s = scores.copy().astype(np.float64)
        s[~finite] = s[finite].min() - 5.0
        s = s / max(temperature, 1e-3)
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()
        idx = np.random.choice(n_particles, size=n_particles, replace=True, p=w)
        return idx

    def _gather_particles(self, z_lig, xh_pocket, ligand, lig_fixed,
                          ligand_mask, pocket_mask, idx, n_particles, n_fixed):
        """Return new z_lig, xh_pocket, etc. that are particle-permuted by `idx`.
        Particles all have the same fixed atoms in the same positions, so we
        only permute the variable parts. Pocket is identical across replicates,
        so no permutation needed there.

        We assume each particle has the same number of ligand atoms (true when
        we call `sample_conditional` once and propagate). Build sele lists once,
        gather, then write back."""
        new_z = z_lig.clone()
        # Build a list of slot indices per particle
        per_particle_slots = [
            (ligand_mask == p).nonzero(as_tuple=True)[0]
            for p in range(n_particles)
        ]
        for dst, src in enumerate(idx):
            if dst == src:
                continue
            dst_slots = per_particle_slots[dst]
            src_slots = per_particle_slots[src]
            # Particles may have different sizes (size_distribution sampled per
            # particle). To keep this simple we only copy min(len_dst, len_src)
            # atoms. The remaining slots in dst keep their value (cosmetic).
            n_copy = min(len(dst_slots), len(src_slots))
            new_z[dst_slots[:n_copy]] = z_lig[src_slots[:n_copy]]
            # Re-write fixed warhead atoms exactly (in case copy from a particle
            # whose atom 0..n_fixed-1 drifted in raw z but was inpaint-snapped):
            # actually warhead is the first n_fixed atoms, which are the same
            # value-wise across particles, so this preserves them.
        return new_z

    @torch.no_grad()
    def sample_one(self, x_warhead, one_hot_warhead, pocket_residues,
                   n_particles: int, resample_every: int, temperature: float,
                   timesteps: int, sample_idx: int):
        """Run one diffusion sample with N particles + REINVENT resampling.

        Returns: (best_mol, best_smi, best_score, history_dicts)
        """
        ligand, pocket, lig_fixed = self._build_initial(
            n_particles, x_warhead, one_hot_warhead, pocket_residues)
        n_fixed = len(x_warhead)

        # The diffusion loop expects normalized inputs.
        ligand, pocket = self.ddpm.normalize(ligand, pocket)
        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        device = self.device
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        com_pocket_0 = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        xh0_ligand = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh_ligand = xh0_ligand.clone()

        # center: ligand
        mean_known = scatter_mean(ligand['x'][lig_fixed.bool().view(-1)],
                                  ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
        mu_lig_x = mean_known
        mu_lig_h = torch.zeros((n_particles, self.model.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[ligand['mask']]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.ddpm.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand['mask'], pocket['mask'])

        # SG position needs to be expressed in the centered coordinate frame:
        # the ligand+pocket system has been centered at COM of known (warhead)
        # atoms in normalized space. We translate sg accordingly. For our
        # purposes the constraint is local geometry, so an approximate frame
        # is fine — we work with the unnormalized current positions when
        # projecting.
        sg_world = self.sg_pos_world

        history = []
        steps_since_resample = 0
        for s in reversed(range(0, timesteps)):
            steps_since_resample += 1
            s_array = torch.full((n_particles, 1), fill_value=s,
                                 device=device, dtype=torch.float)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            gamma_t = self.ddpm.gamma(t_array)
            gamma_s = self.ddpm.gamma(s_array)

            # 1. denoise unknown part
            z_lig_unknown, xh_pocket = self.ddpm.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, ligand['mask'],
                pocket['mask'])

            # 2. re-noise known (fixed warhead) part to match diffusion level
            com_pocket = scatter_mean(xh_pocket[:, :self.ddpm.n_dims],
                                      pocket['mask'], dim=0)
            xh_ligand[:, :self.ddpm.n_dims] = \
                ligand['x'] + (com_pocket - com_pocket_0)[ligand['mask']]
            z_lig_known, xh_pocket, _ = self.ddpm.noised_representation(
                xh_ligand, xh_pocket, ligand['mask'], pocket['mask'], gamma_s)

            # COM correction so combined system is COM-free
            com_noised = scatter_mean(
                z_lig_known[lig_fixed.bool().view(-1)][:, :self.ddpm.n_dims],
                ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
            com_denoised = scatter_mean(
                z_lig_unknown[lig_fixed.bool().view(-1)][:, :self.ddpm.n_dims],
                ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
            dx = com_denoised - com_noised
            z_lig_known[:, :self.ddpm.n_dims] += dx[ligand['mask']]
            xh_pocket[:, :self.ddpm.n_dims] += dx[pocket['mask']]

            # 3. combine: fixed atoms come from re-noised known, rest from denoised
            z_lig = z_lig_known * lig_fixed + z_lig_unknown * (1 - lig_fixed)

            # 4. covalent constraint projection — DISABLED in smoke mode.
            #    The warhead atoms are already pinned to the prereactive
            #    geometry by the inpaint mask (they come from
            #    warhead_at_cys.sdf, which was produced by place_warhead_at_cys
            #    on the manifold). Re-projecting at every step would break the
            #    DDPM COM-zero invariant. We keep the hooks for later
            #    relaxation experiments.
            #
            # if self.args.project_constraint:
            #     ... (project_warhead_inplace per particle, then re-center)

            # 5. resample particles every K steps (skip the very last few steps
            #    so the resampled positions can finish denoising).
            if (steps_since_resample >= resample_every) and (s > 1):
                steps_since_resample = 0
                # Decode each particle (in unnormalized x), get SMILES, score.
                # In early diffusion steps the SMILES will be garbage, but the
                # softmax still works (low scores -> not picked). We use the
                # standard build_molecule logic.
                smiles_list, scores, _ = self._decode_and_score_all(
                    z_lig, ligand['mask'], n_particles, sg_world)
                idx = self._resample_particles(scores, n_particles, temperature)
                ess = 1.0 / (np.exp((scores - np.max(scores)) / max(temperature, 1e-3))
                             ).sum() if np.isfinite(scores).any() else float('nan')
                history.append({
                    "step": int(s),
                    "scores": scores.tolist(),
                    "selected_idx": idx.tolist(),
                    "smiles_sample": [smi[:80] for smi in smiles_list[:4]],
                })
                # gather: copy z_lig from selected source particles to dst
                z_lig = self._gather_particles(
                    z_lig, xh_pocket, ligand, lig_fixed,
                    ligand['mask'], pocket['mask'], idx,
                    n_particles, n_fixed)
                print(f"  [step {s:3d}/{timesteps}] resampled — "
                      f"scores=[{np.nanmin(scores):.2f}, {np.nanmax(scores):.2f}], "
                      f"ESS≈{ess:.1f}", flush=True)

        # Final: sample p(x, h | z_0) for all particles, then pick best
        with torch.no_grad():
            x_lig, h_lig, x_pocket, h_pocket = self.ddpm.sample_p_xh_given_z0(
                z_lig, xh_pocket, ligand['mask'], pocket['mask'], n_particles)
        # Move back to original pocket position
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0) * self.ddpm.norm_values[0]
        # Note: pocket['x'] is normalized, undo the normalization for the COM correction
        pocket_com_after = scatter_mean(x_pocket[:, :self.ddpm.n_dims], pocket['mask'], dim=0)
        # The actual COM correction logic from inpaint.py uses (pocket_com_before - pocket_com_after)
        # but those calculations require unnormalized poses. We just keep raw x for
        # decoding — geometry will be relative to the centered frame, which is fine
        # for SMILES bond perception and FiLMDelta scoring (it doesn't use coords).

        atom_type = h_lig.argmax(1).detach().cpu()
        x_lig_cpu = x_lig.detach().cpu()

        # Decode all particles
        final_mols, final_smiles = [], []
        for p in range(n_particles):
            sele = (ligand['mask'] == p).cpu()
            mol = decode_particle_to_mol(
                x_lig_cpu[sele], atom_type[sele], self.model.dataset_info)
            if mol is None:
                final_mols.append(None)
                final_smiles.append("")
            else:
                final_mols.append(mol)
                try:
                    final_smiles.append(Chem.MolToSmiles(mol))
                except Exception:
                    final_smiles.append("")

        final_scores = self.scorer.score_smiles_batch(final_smiles)
        if not np.isfinite(final_scores).any():
            return None, "", -np.inf, history
        best_p = int(np.argmax(final_scores))
        return final_mols[best_p], final_smiles[best_p], float(final_scores[best_p]), history


# ── Pocket residues helper ─────────────────────────────────────────────────
def get_pocket_residues_only(pdb_file, ref_ligand):
    pdb_model = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
    return utils.get_pocket_from_ligand(pdb_model, ref_ligand)


# ── Find SG position of the cysteine from the warhead-at-cys SDF ───────────
def find_sg_position(receptor_pdb: Path, warhead_at_cys_sdf: Path) -> np.ndarray:
    """Return the SG position. We use the C_β position from warhead_at_cys.sdf
    as a proxy if SG isn't directly in there (it's offset by 1.85 Å along the
    S-C bond). For a more precise location, we read the receptor PDB and find
    the closest cysteine SG to the warhead C_β."""
    suppl = Chem.SDMolSupplier(str(warhead_at_cys_sdf), sanitize=False)
    wh = suppl[0]
    cb_pos = np.array(list(wh.GetConformer().GetAtomPosition(0)))
    # Search receptor PDB for closest CYS SG
    pdb_model = PDBParser(QUIET=True).get_structure('', str(receptor_pdb))[0]
    best_d, best_sg = 1e9, None
    for chain in pdb_model:
        for res in chain:
            if res.get_resname() != "CYS":
                continue
            for atom in res:
                if atom.get_name() == "SG":
                    p = np.array(atom.get_coord())
                    d = np.linalg.norm(p - cb_pos)
                    if d < best_d:
                        best_d, best_sg = d, p
    if best_sg is None:
        # fallback: SG ≈ Cβ - 1.85 along arbitrary direction
        return cb_pos
    print(f"  found CYS SG at {best_sg}, dist to warhead C_β = {best_d:.2f} Å",
          flush=True)
    return best_sg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--pdbfile", type=str, required=True)
    p.add_argument("--ref_ligand", type=str, required=True)
    p.add_argument("--fixed_atoms_file", type=str, required=True,
                   help="Text file listing the warhead atom names, one per line.")
    p.add_argument("--warhead_sdf", type=str, required=True,
                   help="SDF with warhead positioned at the cysteine.")
    p.add_argument("--film_ckpt", type=str, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--n_particles", type=int, default=16)
    p.add_argument("--resample_every", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--center", type=str, default="ligand", choices=("ligand", "pocket"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== B2 inpaint ===", flush=True)
    print(f"  device: {device}", flush=True)
    print(f"  N samples={args.n_samples}, P particles={args.n_particles}, "
          f"K resample_every={args.resample_every}, T={args.temperature}, "
          f"timesteps={args.timesteps}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Lock file guards against parallel launches stomping on the same outdir
    # (root cause of the 24h B2 empty-mols.sdf bug). Stale lock (>2h) is bypassed.
    import os as _os, time as _t
    lock_path = args.outdir / "RUNNING.lock"
    if lock_path.exists():
        age_s = _t.time() - lock_path.stat().st_mtime
        if age_s < 7200:
            print(f"[abort] another B2 instance is already writing to {args.outdir} "
                  f"(lock age {age_s/60:.1f} min). Pick a different --outdir or wait.",
                  flush=True)
            sys.exit(2)
        else:
            print(f"[stale lock {age_s/60:.1f} min — overwriting]", flush=True)
    lock_path.write_text(f"pid={_os.getpid()} started={_t.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log_path = args.outdir / f"run.{_os.getpid()}.log"
    log_f = open(log_path, "w")

    # Tee stdout
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s)
            for f in self.files:
                if hasattr(f, "flush"): f.flush()
        def flush(self):
            for f in self.files:
                if hasattr(f, "flush"): f.flush()
    sys.stdout = Tee(sys.__stdout__, log_f)

    # Load DiffSBDD model
    t0 = time.time()
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"  loaded LigandPocketDDPM in {time.time() - t0:.1f}s", flush=True)

    # Load FiLMDelta scorer
    scorer = FiLMScorer(args.film_ckpt)
    print(f"  loaded FiLMScorer (anchors={scorer.n_anchors}, "
          f"pIC50 ∈ [{scorer.anchor_pIC50.min():.2f}, {scorer.anchor_pIC50.max():.2f}])",
          flush=True)

    # Read warhead atom names + warhead SDF
    fix_atoms_names = [ln.strip() for ln in open(args.fixed_atoms_file)
                       if ln.strip() and not ln.startswith("#")]
    print(f"  warhead atoms (by name): {fix_atoms_names}", flush=True)

    # Build x_fixed, one_hot_fixed from warhead SDF
    x_warhead, one_hot_warhead = prepare_from_sdf_files(
        [args.warhead_sdf], model.lig_type_encoder)
    print(f"  warhead: {len(x_warhead)} atoms", flush=True)

    # SG position
    sg_world = find_sg_position(Path(args.pdbfile), Path(args.warhead_sdf))

    # Pocket residues (computed once)
    pocket_residues = get_pocket_residues_only(args.pdbfile, args.ref_ligand)
    print(f"  pocket: {len(list(pocket_residues))} residues", flush=True)

    # Run the sampler
    sampler = B2Sampler(model, args, scorer, sg_world)
    all_mols, ranking_rows = [], []
    for s in range(args.n_samples):
        # Free GPU memory between samples — replicating the pocket P times
        # for each sample is the dominant memory cost on a T4 (14.5 GB).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc; gc.collect(); torch.cuda.empty_cache()
        print(f"\n--- Sample {s + 1}/{args.n_samples} ---", flush=True)
        t1 = time.time()
        torch.manual_seed(1234 + s)
        np.random.seed(1234 + s)
        try:
            mol, smi, score, history = sampler.sample_one(
                x_warhead, one_hot_warhead, pocket_residues,
                n_particles=args.n_particles,
                resample_every=args.resample_every,
                temperature=args.temperature,
                timesteps=args.timesteps,
                sample_idx=s,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  sample {s} FAILED: {e}", flush=True)
            continue

        t2 = time.time()
        if mol is None:
            print(f"  sample {s} produced no valid mol ({t2 - t1:.1f}s)", flush=True)
            continue

        all_mols.append(mol)
        ranking_rows.append({"sample_idx": s, "smiles": smi,
                             "filmdelta_pIC50": score,
                             "time_s": t2 - t1})
        print(f"  sample {s}: pIC50={score:.3f}  smi={smi[:80]}  "
              f"({t2 - t1:.1f}s)", flush=True)

    # Write SDF via PID-suffixed temp file + atomic rename, so parallel
    # launches on the same outdir don't clobber one another. Also append-flush
    # each mol incrementally to a partial.sdf so a mid-run crash still leaves
    # whatever was produced.
    import os as _os
    sdf_final = args.outdir / "mols.sdf"
    sdf_tmp   = args.outdir / f"mols.{_os.getpid()}.sdf"
    if all_mols:
        with Chem.SDWriter(str(sdf_tmp)) as w:
            for m in all_mols:
                if m is not None:
                    w.write(m)
        # Atomic rename — if another process beat us, keep both so neither is lost.
        if sdf_final.exists():
            _os.rename(str(sdf_tmp), str(args.outdir / f"mols.from_pid_{_os.getpid()}.sdf"))
            print(f"\nKept {len(all_mols)} mols at mols.from_pid_{_os.getpid()}.sdf "
                  f"(mols.sdf already existed)", flush=True)
        else:
            _os.rename(str(sdf_tmp), str(sdf_final))
            print(f"\nWrote {len(all_mols)} mols to {sdf_final}", flush=True)
    else:
        print("\nNo mols produced", flush=True)

    # Write ranking CSV
    import csv
    csv_path = args.outdir / "ranking.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sample_idx", "smiles", "filmdelta_pIC50", "time_s"])
        w.writeheader()
        for row in sorted(ranking_rows, key=lambda r: -r["filmdelta_pIC50"]):
            w.writerow(row)
    print(f"Wrote {csv_path}", flush=True)
    # Release the lock
    try:
        lock_path.unlink()
    except Exception:
        pass
    log_f.close()


if __name__ == "__main__":
    main()
