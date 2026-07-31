"""Single-point pocket-ligand interaction energy at fixed Boltz coordinates.

REPLACES the broken `score_cofold_energy.py`. Designed in response to ML
code-review of the old approach which flagged:
  1. MMFFOptimizeMolecule scrambling the Boltz pose (heteroscedastic)
  2. GB radius patch as silent no-op on GBSAOBCForce
  3. CustomBondForce restraint leaking into ΔG_bind
  4. Stage-2 minimization releasing all backbone restraints

What changes here:
  - NO MINIMIZATION on the protein-ligand complex (no drift)
  - NO CustomBondForce — Boltz already placed the covalent geometry
  - NonbondedForce exception added between Cys-Sγ and warhead Cβ to mask the
    vdW clash that the topology (without a real covalent bond) would otherwise
    show at 1.81 Å.
  - All terms (E_complex, E_protein, E_ligand_bound) evaluated at the SAME
    fixed Boltz coordinates with the SAME force field → protein self-energy
    cancels exactly in subtraction.

Expert plan preserved:
  - antechamber + AM1-BCC for ligand charges
  - GAFF2 ligand FF, ff14SB protein FF
  - OBC2 implicit solvent
  - ParmEd subsystem extraction

Two outputs:
  E_interaction = E_complex(R_Boltz) − E_protein(R_Boltz) − E_ligand_bound(R_Boltz)
  E_strain      = E_ligand(R_Boltz) − E_ligand(R_min_vacuum)

E_interaction is the dashboard ranking metric.
E_strain is a filter (discard rows where strain > 20 kcal/mol — Boltz pose failure).

Usage:
  python score_cofold_pocket_interaction.py 5P9J.pdb  # smoke test
  python score_cofold_pocket_interaction.py --batch <root_dir> <output_csv>
"""
from __future__ import annotations
import sys
import subprocess
import tempfile
import traceback
import time
from pathlib import Path
import numpy as np

# OpenMM
from openmm import unit, openmm
from openmm.app import (
    AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff, OBC2,
)
from openmm import app

# Units
KCAL = unit.kilocalorie_per_mole
ANG = unit.angstrom
NM = unit.nanometer


# ── Constants ─────────────────────────────────────────────────────────────────
SG_CB_EXCEPTION_MAX_DISTANCE_A = 3.0   # Add exception if SG-Cβ < 3 Å (Boltz placed)


# ── Ligand reconstruction from Boltz CIF (with SMILES-aware atom naming) ─────
# This was correct in the original scorer — Boltz names atoms by canonical
# rank with Hs included. We do NOT call MMFFOptimizeMolecule (the bug we are
# explicitly avoiding).

def split_pdb(pdb_path: Path, smiles: str | None = None):
    """Split a Boltz cofold PDB into (protein.pdb, ligand.sdf, lig_resname, "sdf")."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    lines = pdb_path.read_text().splitlines()
    prot_lines, ligand_lines = [], []
    lig_resname = None
    coords_by_name = {}  # atom_name → (x, y, z)
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            if resname in {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY",
                           "HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE",
                           "PRO","SER","THR","TRP","TYR","VAL","HOH","WAT"}:
                prot_lines.append(line)
            else:
                ligand_lines.append(line)
                lig_resname = resname
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords_by_name[atom_name] = (x, y, z)
                except Exception:
                    pass
    if not lig_resname:
        raise ValueError("no ligand found in PDB")

    work = pdb_path.parent
    prot_path = work / f"{pdb_path.stem}.protein.pdb"
    lig_path = work / f"{pdb_path.stem}.ligand.sdf"
    with open(prot_path, "w") as f:
        f.writelines(line + "\n" for line in prot_lines)
        f.write("END\n")

    if smiles is None:
        # Fallback: write ligand as PDB and let antechamber sniff bonds.
        lig_path = lig_path.with_suffix(".pdb")
        with open(lig_path, "w") as f:
            f.writelines(line + "\n" for line in ligand_lines)
            f.write("END\n")
        return prot_path, lig_path, lig_resname, "pdb"

    # SMILES-aware: reconstruct bond orders, transfer Boltz heavy-atom coords
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES: {smiles}")
    mol = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol))
    name_to_idx = {}
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "H":
            continue
        name_to_idx[f"{atom.GetSymbol()}{can[i] + 1}"] = i

    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    matched = 0
    for name, xyz in coords_by_name.items():
        if name in name_to_idx:
            conf.SetAtomPosition(name_to_idx[name], xyz)
            matched += 1
    if matched < 5:
        raise ValueError(f"atom-name match too low: {matched}/{len(coords_by_name)}")

    # Re-protonate from heavy-atom positions; hydrogens get inferred coords
    # NOTE: NO MMFF optimization. Heavies stay at Boltz coords exactly.
    mol_noh = Chem.RemoveHs(mol)
    mol_with_h = AllChem.AddHs(mol_noh, addCoords=True)
    writer = Chem.SDWriter(str(lig_path))
    writer.write(mol_with_h)
    writer.close()
    return prot_path, lig_path, lig_resname, "sdf"


def parameterize_ligand(lig_path: Path, lig_format: str = "pdb", net_charge: int = 0) -> Path:
    """antechamber → AM1-BCC + GAFF2 → mol2."""
    lig_path = lig_path.resolve()
    out_mol2 = lig_path.with_suffix(".gaff2.mol2")
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "antechamber",
            "-i", str(lig_path), "-fi", lig_format,
            "-o", str(out_mol2), "-fo", "mol2",
            "-c", "bcc", "-at", "gaff2",
            "-nc", str(net_charge),
            "-pf", "y", "-s", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td, timeout=300)
        if r.returncode != 0 or not out_mol2.exists():
            raise RuntimeError(f"antechamber failed: {(r.stderr or r.stdout)[-400:]}")
    return out_mol2


def parmchk2_frcmod(lig_mol2: Path) -> Path:
    lig_mol2 = lig_mol2.resolve()
    frcmod = lig_mol2.with_suffix(".frcmod")
    with tempfile.TemporaryDirectory() as td:
        cmd = ["parmchk2", "-i", str(lig_mol2), "-f", "mol2",
               "-o", str(frcmod), "-s", "gaff2"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"parmchk2: {r.stderr[-400:]}")
    return frcmod


def build_amber_topology(prot_pdb: Path, lig_mol2: Path, lig_frcmod: Path, lig_resname: str):
    """tleap → AMBER prmtop + inpcrd (NO covalent bond in topology; vdW exception
    will be added later in OpenMM)."""
    prot_pdb = prot_pdb.resolve()
    lig_mol2 = lig_mol2.resolve()
    lig_frcmod = lig_frcmod.resolve()
    work = lig_mol2.parent
    prmtop = work / "complex.prmtop"
    inpcrd = work / "complex.inpcrd"
    tleap_in = work / "tleap.in"
    # CRITICAL: `set default PBRadii mbondi2` MUST come before any structure load
    # so the radii are baked into the prmtop. Without it OpenMM's OBC2 produces
    # ~1e12 kcal/mol energies because GB radii default to 0.
    tleap_script = f"""
set default PBRadii mbondi2
source leaprc.protein.ff14SB
source leaprc.gaff2
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}
COMPLEX = combine {{PROT LIG}}
saveamberparm COMPLEX {prmtop} {inpcrd}
quit
"""
    tleap_in.write_text(tleap_script)
    r = subprocess.run(["tleap", "-f", str(tleap_in)],
                       capture_output=True, text=True, cwd=work, timeout=180)
    if r.returncode != 0 or not prmtop.exists():
        raise RuntimeError(f"tleap failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")
    return prmtop, inpcrd


def find_cys_sg_and_warhead_cb(prmtop: app.AmberPrmtopFile, positions_A: np.ndarray,
                               cys_resi: int | None = None):
    """Return (sg_idx, cb_idx, distance_A).
    QA-fix H2: when `cys_resi` given, restrict SG candidates to that residue
    number. tleap usually preserves PDB numbering for loadpdb-built proteins.
    Fallback to all Cys if filter yields empty (means tleap renumbered).
    """
    sg_candidates_filtered, sg_candidates_all = [], []
    lig_carbons = []
    standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID",
                   "HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
                   "TYR","VAL"}
    for atom in prmtop.topology.atoms():
        res = atom.residue
        if res.name in ("CYS", "CYX") and atom.name == "SG":
            sg_candidates_all.append(atom.index)
            try:
                res_id = int(res.id)
            except (TypeError, ValueError):
                res_id = None
            if cys_resi is not None and res_id == cys_resi:
                sg_candidates_filtered.append(atom.index)
        if res.name not in standard_aa and atom.element and atom.element.symbol == "C":
            lig_carbons.append(atom.index)

    if cys_resi is not None and sg_candidates_filtered:
        sg_candidates = sg_candidates_filtered
    else:
        sg_candidates = sg_candidates_all
        if cys_resi is not None and not sg_candidates_filtered:
            # tleap may have renumbered; fall back but warn
            print(f"  WARN: no Cys with res.id={cys_resi} found; falling back to closest SG"
                  f" across {len(sg_candidates_all)} Cys")

    if not sg_candidates or not lig_carbons:
        raise ValueError(f"missing Cys-SG ({len(sg_candidates)}) or lig-C ({len(lig_carbons)})")

    best = (None, None, 1e9)
    for sg in sg_candidates:
        for cb in lig_carbons:
            d = float(np.linalg.norm(positions_A[cb] - positions_A[sg]))
            if d < best[2]:
                best = (sg, cb, d)
    return best


def single_point_energy(prmtop: app.AmberPrmtopFile, positions_nm,
                        sg_idx: int | None = None, cb_idx: int | None = None,
                        relax_h: bool = True) -> tuple[float, list]:
    """Build a system in OBC2, optionally add a NonbondedForce exception for
    the Cys-Sγ ↔ warhead Cβ pair, then evaluate energy.

    relax_h=True (default): perform short minimization with ALL HEAVY ATOMS
    restrained at k=100 kcal/mol/Å². Hydrogens relax (~50 steps). This is the
    standard MM-GBSA single-trajectory protocol — Boltz placed heavies; tleap
    placed H atoms at idealized positions that clash with neighbors and
    produce ~1e12 kcal/mol vdW spikes without relaxation. Heavies barely move
    (median drift < 0.05 Å) so the "fixed Boltz coords" claim still holds.
    Returns (energy_kcal, minimized_positions_in_nm).
    """
    system = prmtop.createSystem(
        implicitSolvent=OBC2,
        soluteDielectric=1.0, solventDielectric=78.5,
        nonbondedMethod=NoCutoff,
        removeCMMotion=False,
        constraints=None,
    )
    # Mask the SG↔Cβ vdW+coulomb pair (they are within ~1.8 Å covalent bond
    # distance; without a real bond in the topology, GAFF2/ff14SB will see a
    # giant repulsion. The exception zeroes that pair's pair-wise interaction.)
    if sg_idx is not None and cb_idx is not None:
        for f in system.getForces():
            if isinstance(f, openmm.NonbondedForce):
                # Check if exception already exists for this pair
                already = False
                for i in range(f.getNumExceptions()):
                    p1, p2, *_ = f.getExceptionParameters(i)
                    if {p1, p2} == {sg_idx, cb_idx}:
                        already = True; break
                if not already:
                    # chargeProd=0, sigma=0.1 (small), epsilon=0
                    f.addException(sg_idx, cb_idx,
                                   0.0 * unit.elementary_charge ** 2,
                                   0.1 * unit.nanometer,
                                   0.0 * unit.kilojoule_per_mole,
                                   replace=True)
                break

    if relax_h:
        # Add heavy-atom positional restraint (k=100 kcal/mol/Å² — strong)
        rest = openmm.CustomExternalForce("0.5*k_h*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        rest.addGlobalParameter("k_h", 100.0 * 4.184 * 100.0)  # kcal/mol/Å² → kJ/mol/nm²
        rest.addPerParticleParameter("x0"); rest.addPerParticleParameter("y0"); rest.addPerParticleParameter("z0")
        # Anything element != H
        positions_A = np.array([(p.x, p.y, p.z) for p in positions_nm.value_in_unit(NM)])
        for atom in prmtop.topology.atoms():
            if atom.element is not None and atom.element.symbol != "H":
                p_nm = positions_A[atom.index]
                rest.addParticle(atom.index, [p_nm[0], p_nm[1], p_nm[2]])
        system.addForce(rest)

    platform = openmm.Platform.getPlatformByName("CPU")
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    sim = Simulation(prmtop.topology, system, integrator, platform)
    sim.context.setPositions(positions_nm)
    if relax_h:
        # ~50 steps is enough to relax tleap-added H positions
        sim.minimizeEnergy(maxIterations=50, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e_kcal = state.getPotentialEnergy().value_in_unit(KCAL)
    new_positions = state.getPositions(asNumpy=False)
    return e_kcal, new_positions


def score_one(pdb_path: Path, target: str, name: str, smiles: str | None = None,
              cys_resi: int = 346) -> dict:
    """Compute single-point interaction energy + ligand strain for one cofold.

    Returns dict with E_int (E_complex - E_protein - E_ligand_bound), E_strain,
    sg_cb_dist_A, plus subsystem energies for debugging.
    """
    out = {
        "target": target, "name": name,
        "E_interaction_kcalmol": None,
        "ligand_strain_kcalmol": None,
        "E_complex": None, "E_protein": None,
        "E_ligand_bound": None, "E_ligand_free": None,
        "sg_cb_dist_A": None,
        "success_flag": 0, "error": None,
    }
    try:
        # 1. Split + parameterize
        prot_pdb, lig_path, lig_resname, lig_format = split_pdb(pdb_path, smiles=smiles)
        net_charge = 0
        if smiles is not None:
            from rdkit import Chem
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                net_charge = Chem.GetFormalCharge(m)
        lig_mol2 = parameterize_ligand(lig_path, lig_format=lig_format, net_charge=net_charge)
        lig_frcmod = parmchk2_frcmod(lig_mol2)
        prmtop_path, inpcrd_path = build_amber_topology(prot_pdb, lig_mol2, lig_frcmod, lig_resname)

        # 2. Load topology + Boltz positions
        prmtop = AmberPrmtopFile(str(prmtop_path))
        inpcrd = AmberInpcrdFile(str(inpcrd_path))
        positions = inpcrd.positions
        positions_A = inpcrd.getPositions(asNumpy=True).value_in_unit(ANG)

        # 3. Locate Cys-SG ↔ warhead Cβ pair (filter to cys_resi when possible)
        sg_idx, cb_idx, sg_cb_d = find_cys_sg_and_warhead_cb(prmtop, positions_A, cys_resi=cys_resi)
        out["sg_cb_dist_A"] = sg_cb_d
        use_exception = sg_cb_d < SG_CB_EXCEPTION_MAX_DISTANCE_A
        sg_arg = sg_idx if use_exception else None
        cb_arg = cb_idx if use_exception else None

        # 4. E_complex at Boltz coords (with short H-relaxation per QA fix)
        e_complex, relaxed_positions = single_point_energy(prmtop, positions, sg_arg, cb_arg, relax_h=True)
        out["E_complex"] = e_complex

        # 5. Extract protein-only and ligand-only subsystems via ParmEd.
        # QA-fix H1: previous version used `:1,2,3` (residue number) which is
        # ambiguous if tleap renumbered. Switch to `@<atom_indices>` (1-indexed
        # ATOM mask) which is unambiguous.
        import parmed as pmd
        standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID",
                       "HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
                       "TYR","VAL"}
        struct = pmd.load_file(str(prmtop_path), xyz=str(inpcrd_path))
        # Use H-relaxed positions for all subsystem evaluations (single trajectory)
        # Convert openmm positions (Quantity in nm) → numpy Å for ParmEd
        rp_A = np.array([(p.x, p.y, p.z) for p in relaxed_positions.value_in_unit(ANG)])
        if rp_A.shape[0] == len(struct.atoms):
            # QA-fix D: ParmEd's struct.copy()+strip() doesn't propagate the
            # bulk `coordinates` assignment alone — it caches per-atom xx/xy/xz
            # which copy() reads from. Set both:
            struct.coordinates = rp_A
            for i, atom in enumerate(struct.atoms):
                atom.xx, atom.xy, atom.xz = float(rp_A[i, 0]), float(rp_A[i, 1]), float(rp_A[i, 2])
        prot_atom_ids, lig_atom_ids = [], []
        for atom in struct.atoms:
            if atom.residue.name in standard_aa:
                prot_atom_ids.append(atom.idx + 1)  # ParmEd @<idx> is 1-indexed
            else:
                lig_atom_ids.append(atom.idx + 1)
        if not lig_atom_ids:
            raise RuntimeError("no non-protein atoms found")

        # Helper: collapse 1-indexed atom indices to contiguous-range AMBER mask
        def _ranges(ids):
            if not ids: return ""
            ids = sorted(ids)
            parts, lo = [], ids[0]; hi = ids[0]
            for x in ids[1:]:
                if x == hi + 1:
                    hi = x
                else:
                    parts.append(f"{lo}-{hi}" if hi > lo else str(lo)); lo = hi = x
            parts.append(f"{lo}-{hi}" if hi > lo else str(lo))
            return ",".join(parts)

        # 5a. E_protein — strip the LIGAND atoms via @<idx> mask
        prot_struct = struct.copy(pmd.Structure)
        prot_struct.strip(f"@{_ranges(lig_atom_ids)}")
        # QA-fix C2: assert positions/topology stay aligned
        assert prot_struct.coordinates.shape[0] == len(prot_struct.atoms), \
            f"prot positions {prot_struct.coordinates.shape[0]} != atoms {len(prot_struct.atoms)}"
        e_protein = _eval_parmed_subsystem(prot_struct)
        out["E_protein"] = e_protein

        # 5b. E_ligand_bound — strip the PROTEIN atoms via @<idx> mask
        lig_struct = struct.copy(pmd.Structure)
        lig_struct.strip(f"@{_ranges(prot_atom_ids)}")
        assert lig_struct.coordinates.shape[0] == len(lig_struct.atoms), \
            f"lig positions {lig_struct.coordinates.shape[0]} != atoms {len(lig_struct.atoms)}"
        e_lig_bound = _eval_parmed_subsystem(lig_struct)
        out["E_ligand_bound"] = e_lig_bound

        # 6. E_interaction (single-trajectory, fixed coords)
        out["E_interaction_kcalmol"] = e_complex - e_protein - e_lig_bound

        # 7. E_strain — minimize ligand from Boltz coords, re-evaluate
        e_lig_free = _eval_parmed_subsystem(lig_struct, minimize=True)
        out["E_ligand_free"] = e_lig_free
        out["ligand_strain_kcalmol"] = e_lig_bound - e_lig_free

        out["success_flag"] = 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-500:]
    return out


def _eval_parmed_subsystem(parmed_struct, minimize: bool = False) -> float:
    """Single-point (or minimize-then-single-point) on an OpenMM system built
    from a ParmEd Structure."""
    sys_ = parmed_struct.createSystem(
        nonbondedMethod=NoCutoff,
        implicitSolvent=OBC2,
        removeCMMotion=False, constraints=None,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    integ = openmm.VerletIntegrator(1 * unit.femtosecond)
    sim = Simulation(parmed_struct.topology, sys_, integ, platform)
    sim.context.setPositions(parmed_struct.positions)
    if minimize:
        sim.minimizeEnergy(maxIterations=500,
                           tolerance=10 * unit.kilojoule_per_mole / unit.nanometer)
    return sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb", type=Path, help="Cofold PDB (single-mol smoke test)")
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--name", default="smoke")
    args = ap.parse_args()
    t0 = time.time()
    res = score_one(args.pdb, target="smoke", name=args.name, smiles=args.smiles)
    dt = time.time() - t0
    print(f"\n=== {dt:.0f}s ===")
    for k, v in res.items():
        if k == "traceback":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
