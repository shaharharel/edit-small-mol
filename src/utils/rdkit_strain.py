"""
RDKit MMFF ligand-strain + UFF vdW interaction scorer.

Public functions:
  - score_cofold(pdb_path, sdf_path_or_smiles, smiles, num_conf=5)
      Full pipeline (strain + vdW) for a Boltz cofold complex PDB. If only the
      complex PDB is given (no separate .lig.sdf), the ligand is extracted from
      HETATM (chain B) and bond orders are restored from the SMILES template.

  - score_posefree(smiles, num_conf_free=5)
      Pose-free intrinsic strain (single ETKDGv3 conformer "bound proxy" vs
      best-of-N free reference). For dashboard candidates without a cofold pose.

Fixes vs the /tmp prototype:
  (1) H-only relaxation under MMFF94s (heavy atoms constrained) before computing
      e_bound. Eliminates the H-placement mismatch that produced negative strain.
  (2) Wrap every MMFF energy call in try/except; treat overflow (|E| > 1e6) as
      a failure (NaN, success_flag=0).
  (3) UFF vdW interaction excludes any ligand-protein contact closer than
      2.5 A (treated as a covalent bond partner -- prevents LJ blow-up on
      covalent inhibitor cofolds like ZAP70 Cys346).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdForceFieldHelpers

RDLogger.DisableLog("rdApp.*")


# UFF vdW params (x_i = r_min in A, D_i = well depth in kcal/mol).
_UFF_PARAMS_HARDCODED: dict[str, tuple[float, float]] = {
    "H": (2.886, 0.044),
    "C": (3.851, 0.105),
    "N": (3.660, 0.069),
    "O": (3.500, 0.060),
    "F": (3.364, 0.050),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "Cl": (3.947, 0.227),
    "Br": (4.189, 0.251),
    "I": (4.500, 0.339),
    "B": (4.083, 0.180),
    "Si": (4.295, 0.402),
    "Na": (2.983, 0.030),
    "Mg": (3.021, 0.111),
    "K": (3.812, 0.035),
    "Ca": (3.399, 0.238),
    "Fe": (2.912, 0.013),
    "Zn": (2.763, 0.124),
}

_ENERGY_OVERFLOW = 1.0e6


# ---------------------------- low-level MMFF helpers ------------------------

def _safe_mmff_energy(mol: Chem.Mol, conf_id: int = -1, variant: str = "MMFF94") -> float | None:
    """MMFF single-point energy. Returns None on overflow or failure."""
    try:
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if props is None:
            return None
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return None
        e = float(ff.CalcEnergy())
        if not math.isfinite(e) or abs(e) > _ENERGY_OVERFLOW:
            return None
        return e
    except Exception:
        return None


def _h_only_relax(mol: Chem.Mol, conf_id: int = -1, variant: str = "MMFF94s", max_its: int = 500) -> bool:
    """Relax only hydrogens; heavy atoms held fixed. Returns True on success."""
    try:
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if props is None:
            return False
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return False
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:
                ff.AddFixedPoint(i)
        ff.Minimize(maxIts=max_its)
        return True
    except Exception:
        return False


def _free_min_energy(smiles: str, num_conf: int = 5, seed: int = 42, max_its: int = 300) -> float | None:
    """Min MMFF energy across `num_conf` optimized conformers (free reference)."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(m, numConfs=num_conf, params=params)
        if len(cids) == 0:
            return None
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=max_its, mmffVariant="MMFF94s")
        energies: list[float] = []
        for conv, e in res:
            if conv == 0 and math.isfinite(e) and abs(e) < _ENERGY_OVERFLOW:
                energies.append(float(e))
        if not energies:
            # fall back to any finite energies
            for conv, e in res:
                if math.isfinite(e) and abs(e) < _ENERGY_OVERFLOW:
                    energies.append(float(e))
        return float(min(energies)) if energies else None
    except Exception:
        return None


def _single_conformer_energy(smiles: str, seed: int = 17) -> float | None:
    """E_bound proxy for pose-free candidates: one ETKDGv3 conformer, no MMFF opt.

    Used as the "bound" reference for pose-free strain (constitutional strain).
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(m, params=params)
        if cid < 0:
            return None
        # H-only relax so the H placement matches the free reference convention.
        _h_only_relax(m, conf_id=cid, variant="MMFF94s")
        return _safe_mmff_energy(m, conf_id=cid, variant="MMFF94s")
    except Exception:
        return None


# ---------------------------- ligand extraction -----------------------------

def _read_ligand_from_complex_pdb(pdb_path: Path, smiles: str) -> Chem.Mol | None:
    """Extract HETATM ligand from cofold complex PDB; restore bonds from SMILES."""
    try:
        with open(pdb_path) as fh:
            lines = [ln for ln in fh if ln.startswith("HETATM")]
        if not lines:
            return None
        block = "".join(lines) + "END\n"
        raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
        if raw is None:
            return None
        try:
            Chem.SanitizeMol(raw, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            return None
        fixed = None
        try:
            fixed = AllChem.AssignBondOrdersFromTemplate(template, raw)
        except Exception:
            fixed = None
        # 2026-06-07 fallback: charged species (e.g. [N+] in pyrazolium rings)
        # often break AssignBondOrdersFromTemplate's graph match because Boltz
        # writes neutral atom records to the PDB but the SMILES carries the
        # formal charge. Strip the explicit + / − tokens and retry — gives a
        # reasonable bound-pose geometry good enough for an MMFF SP energy
        # (charge state is forced after AddHs anyway).
        if fixed is None:
            try:
                neut_smiles = (
                    smiles
                    .replace("[N+]", "N").replace("[N-]", "N")
                    .replace("[O+]", "O").replace("[O-]", "O")
                    .replace("[C-]", "C").replace("[C+]", "C")
                    .replace("[S+]", "S").replace("[S-]", "S")
                    .replace("[NH+]", "N").replace("[nH+]", "[nH]")
                    .replace("[NH2+]", "N").replace("[NH3+]", "N")
                )
                neut_template = Chem.MolFromSmiles(neut_smiles)
                if neut_template is not None:
                    fixed = AllChem.AssignBondOrdersFromTemplate(neut_template, raw)
            except Exception:
                fixed = None
        if fixed is None:
            return None
        try:
            Chem.SanitizeMol(fixed)
        except Exception:
            pass
        fixed_h = Chem.AddHs(fixed, addCoords=True)
        return fixed_h
    except Exception:
        return None


def _read_ligand_from_sdf(sdf_path: Path) -> Chem.Mol | None:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for mol in suppl:
        if mol is None:
            continue
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            try:
                mol = Chem.AddHs(mol, addCoords=True)
            except Exception:
                pass
        return mol
    return None


def _read_protein_heavy_atoms(pdb_path: Path) -> list[tuple[str, float, float, float]]:
    out: list[tuple[str, float, float, float]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            elem = line[76:78].strip().upper() or line[12:16].strip()[0]
            if elem == "H":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            out.append((elem.capitalize(), x, y, z))
    return out


def _vdw_interaction(ligand: Chem.Mol,
                     prot_atoms: list[tuple[str, float, float, float]],
                     cutoff: float = 8.0,
                     min_dist: float = 2.5) -> float | None:
    """UFF Lennard-Jones ligand-protein interaction.

    Pairs with r < min_dist are EXCLUDED (treated as bond partners). This is
    essential for covalent inhibitor cofolds, where the warhead-cysteine bond
    (~1.8 A) would otherwise blow up the LJ.
    """
    try:
        conf = ligand.GetConformer()
        n = ligand.GetNumAtoms()
        lig_params: list[tuple[float, float] | None] = []
        lig_coords: list[tuple[float, float, float]] = []
        for i in range(n):
            a = ligand.GetAtomWithIdx(i)
            elem = a.GetSymbol().capitalize()
            if elem == "H":
                lig_params.append(None); lig_coords.append((0.0, 0.0, 0.0)); continue
            lig_params.append(_UFF_PARAMS_HARDCODED.get(elem))
            p = conf.GetAtomPosition(i)
            lig_coords.append((p.x, p.y, p.z))

        cutoff_sq = cutoff * cutoff
        min_sq = min_dist * min_dist
        total = 0.0
        prot_cache: dict[str, tuple[float, float] | None] = {}
        for elem_p, px, py, pz in prot_atoms:
            if elem_p not in prot_cache:
                prot_cache[elem_p] = _UFF_PARAMS_HARDCODED.get(elem_p)
            pp = prot_cache[elem_p]
            if pp is None:
                continue
            xp, Dp = pp
            for i in range(n):
                lp = lig_params[i]
                if lp is None:
                    continue
                xl, Dl = lp
                lx, ly, lz = lig_coords[i]
                dx = lx - px; dy = ly - py; dz = lz - pz
                r2 = dx*dx + dy*dy + dz*dz
                if r2 > cutoff_sq or r2 < min_sq:
                    continue
                x_ij = math.sqrt(xl * xp)
                D_ij = math.sqrt(Dl * Dp)
                r = math.sqrt(r2)
                ratio = x_ij / r
                r6 = ratio ** 6
                r12 = r6 * r6
                total += D_ij * (r12 - 2.0 * r6)
        if not math.isfinite(total) or abs(total) > _ENERGY_OVERFLOW:
            return None
        return float(total)
    except Exception:
        return None


# ---------------------------- public entrypoints ----------------------------

def score_cofold(pdb_path: Path, smiles: str, sdf_path: Path | None = None,
                 num_conf: int = 5, seed: int = 42, compute_interaction: bool = True) -> dict[str, Any]:
    """Full strain + vdW score for one Boltz cofold complex.

    If sdf_path is provided (and exists) the ligand is read from there. Otherwise
    the ligand is extracted from HETATM in pdb_path and bonds are restored from
    the SMILES template.
    """
    t0 = time.perf_counter()
    out: dict[str, Any] = {"success_flag": 0, "error": None,
                            "strain_kcal_mol": float("nan"),
                            "vdw_interaction_kcal_mol": float("nan"),
                            "e_bound_kcal_mol": float("nan"),
                            "e_free_kcal_mol": float("nan")}

    lig = None
    if sdf_path is not None and Path(sdf_path).exists():
        lig = _read_ligand_from_sdf(Path(sdf_path))
    if lig is None:
        lig = _read_ligand_from_complex_pdb(Path(pdb_path), smiles)
    if lig is None:
        out["error"] = "ligand_load_failed"
        return out

    # Fix (1): H-only relax with MMFF94s before SP energy.
    _h_only_relax(lig, conf_id=-1, variant="MMFF94s")

    # Fix (2): safe SP energy w/ overflow guard.
    e_bound = _safe_mmff_energy(lig, conf_id=-1, variant="MMFF94s")
    if e_bound is None:
        out["error"] = "e_bound_failed"
        return out

    e_free = _free_min_energy(smiles, num_conf=num_conf, seed=seed)
    if e_free is None:
        out["error"] = "e_free_failed"
        return out

    out["e_bound_kcal_mol"] = e_bound
    out["e_free_kcal_mol"] = e_free
    out["strain_kcal_mol"] = float(e_bound - e_free)

    if compute_interaction:
        prot_atoms = _read_protein_heavy_atoms(Path(pdb_path))
        e_inter = _vdw_interaction(lig, prot_atoms, cutoff=8.0, min_dist=2.5)
        if e_inter is None:
            out["error"] = "vdw_failed"
            # strain still valid -- partial success
            out["success_flag"] = 1
            out["wallclock_s"] = time.perf_counter() - t0
            return out
        out["vdw_interaction_kcal_mol"] = float(e_inter)

    out["success_flag"] = 1
    out["wallclock_s"] = time.perf_counter() - t0
    return out


def score_posefree(smiles: str, num_conf_free: int = 5, seed: int = 42,
                   free_max_its: int = 300) -> dict[str, Any]:
    """Pose-free intrinsic strain (constitutional, no pocket).

    One ETKDGv3 conformer + H-only relax = "bound proxy". Best-of-N MMFF-optimized
    conformers = "free reference". Difference = pose-free strain.
    """
    t0 = time.perf_counter()
    out: dict[str, Any] = {"success_flag": 0, "error": None,
                            "strain_kcal_mol": float("nan"),
                            "e_bound_kcal_mol": float("nan"),
                            "e_free_kcal_mol": float("nan")}
    e_bound = _single_conformer_energy(smiles, seed=seed + 1)
    if e_bound is None:
        out["error"] = "e_bound_failed"
        return out
    e_free = _free_min_energy(smiles, num_conf=num_conf_free, seed=seed, max_its=free_max_its)
    if e_free is None:
        out["error"] = "e_free_failed"
        return out
    out["e_bound_kcal_mol"] = e_bound
    out["e_free_kcal_mol"] = e_free
    out["strain_kcal_mol"] = float(e_bound - e_free)
    out["success_flag"] = 1
    out["wallclock_s"] = time.perf_counter() - t0
    return out
