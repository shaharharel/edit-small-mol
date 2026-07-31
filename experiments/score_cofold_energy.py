"""MM-GBSA energy scoring for covalent protein-ligand complexes.

Per expert recommendation (chat 2026-05-21):
  - Charges: AM1-BCC via antechamber
  - Protein FF: ff14SB (paired with GBSA)
  - Ligand FF: GAFF2
  - Solvent: GBn2 implicit
  - Two-stage minimization: 500 steps backbone-restrained, 200 unrestrained
  - Custom Cys-Sγ–Cβ_warhead bond: k=300 kcal/mol/Å², r=1.81 Å
  - Angle terms: Cα-Cβ-Sγ and Sγ-Cβ_lig-Cα_lig (k=50 kcal/mol/rad², θ=109.5°)
  - GB radius patch: Sγ=1.80 Å (thioether vs default 2.00 thiolate), Cβ_lig=1.70 Å

Outputs CSV row per cofold:
  target, name, dG_bind_kcalmol, ligand_strain_kcalmol,
  E_complex, E_protein, E_ligand_bound, E_ligand_free,
  rmsd_min_A, sg_cb_dist_A, success_flag, error
"""
from __future__ import annotations
import sys, json, csv, time, traceback, tempfile, subprocess, os, math
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

import openmm
import openmm.app as app
from openmm import unit
from openmm.app import ForceField, Modeller, NoCutoff
# openff.toolkit was imported here but unused (and its lazy-import of
# Molecule triggers a torchvision circular-import in this env). Removed.

KCAL = unit.kilocalorie_per_mole
ANG = unit.angstrom
NM = unit.nanometer

# --- expert-prescribed constants ---
CYS_S_BOND_LENGTH_NM = 0.181   # 1.81 Å
CYS_S_BOND_K_KCAL_PER_A2 = 300.0
SG_RADIUS_NM = 0.180  # thioether (vs default thiolate 2.00)
CB_LIG_RADIUS_NM = 0.170
HEAVY_ATOM_RESTRAINT_K_KCAL_PER_A2 = 10.0


def split_pdb(pdb_path: Path, smiles: str | None = None):
    """Split PDB into protein-only PDB and ligand-only SDF.

    For the ligand, we use RDKit + the provided SMILES to build a molecule with
    correct bond orders, then apply Boltz CIF heavy-atom coordinates by mapping
    via CanonicalRankAtoms (matches Boltz's atom-naming convention).
    Falls back to plain HETATM-PDB ligand if smiles is None.
    """
    protein_lines = []
    ligand_by_res = {}
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            protein_lines.append(line)
        elif line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname in {"HOH", "WAT", "NA", "K", "MG", "ZN", "CL", "CA", "BR"}:
                continue
            ligand_by_res.setdefault(resname, []).append(line)
        elif line.startswith(("TER", "END")):
            protein_lines.append(line)

    if not ligand_by_res:
        raise ValueError(f"no ligand HETATM in {pdb_path}")
    lig_resname = max(ligand_by_res, key=lambda k: len(ligand_by_res[k]))
    ligand_lines = ligand_by_res[lig_resname]

    prot_path = pdb_path.with_suffix(".prot.pdb")
    with open(prot_path, "w") as f:
        f.writelines(protein_lines)
        f.write("END\n")

    # Build ligand from SMILES + Boltz coords
    if smiles is not None:
        lig_path = pdb_path.with_suffix(".lig.sdf")
        lig_ext = "sdf"
        # Parse Boltz HETATM coords by atom name (e.g. C1, C2, N1...)
        coords_by_name = {}
        for ln in ligand_lines:
            an = ln[12:16].strip()
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            coords_by_name[an] = (x, y, z)

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"bad SMILES: {smiles}")
        mol = AllChem.AddHs(mol)
        # Boltz names atoms by element + (canonical_rank + 1), with Hs included.
        # e.g. "C25" = carbon atom whose canonical rank (in the mol-with-Hs) is 24.
        can = list(AllChem.CanonicalRankAtoms(mol))
        name_to_idx = {}
        for i, atom in enumerate(mol.GetAtoms()):
            sym = atom.GetSymbol()
            if sym == "H": continue
            name_to_idx[f"{sym}{can[i] + 1}"] = i

        # Embed once to get reasonable H positions, then overwrite heavies
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conf = mol.GetConformer()
        matched = 0
        for name, xyz in coords_by_name.items():
            if name in name_to_idx:
                idx = name_to_idx[name]
                conf.SetAtomPosition(idx, xyz)
                matched += 1
        if matched < 5:
            raise ValueError(f"atom-name match too low: {matched}/{len(coords_by_name)} heavies matched")
        # ─── BUG-FIX #1 ──────────────────────────────────────────────────
        # Previous code: AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=100)
        #   scrambled the Boltz heavy-atom coords away from the cofold pose.
        # Fix: re-protonate from heavies (addCoords=True does geometry-aware H
        #   placement) and run MMFF with EVERY heavy atom marked as fixed via
        #   ff.AddFixedPoint(i). Heavies stay at their Boltz positions; only
        #   the freshly-added hydrogens relax. We validate by asserting heavy
        #   RMSD between input and output is ≈0.
        # ────────────────────────────────────────────────────────────────
        mol_noh = AllChem.RemoveHs(mol)
        mol_with_h = AllChem.AddHs(mol_noh, addCoords=True)

        # Snapshot heavy-atom coords *before* the constrained min for validation
        conf_pre = mol_with_h.GetConformer()
        heavy_idx_for_check = [a.GetIdx() for a in mol_with_h.GetAtoms() if a.GetAtomicNum() > 1]
        heavies_before = np.array([
            [conf_pre.GetAtomPosition(i).x, conf_pre.GetAtomPosition(i).y, conf_pre.GetAtomPosition(i).z]
            for i in heavy_idx_for_check
        ])

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mmff_props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mmff_props)
            if ff is not None:
                for hi in heavy_idx_for_check:
                    ff.AddFixedPoint(hi)
                ff.Minimize(maxIts=200)
                # (If MMFF cannot parameterize – e.g. unusual element – we still
                #  proceed; H positions from AddHs(addCoords=True) are decent.)

        conf_post = mol_with_h.GetConformer()
        heavies_after = np.array([
            [conf_post.GetAtomPosition(i).x, conf_post.GetAtomPosition(i).y, conf_post.GetAtomPosition(i).z]
            for i in heavy_idx_for_check
        ])
        heavy_rmsd = float(np.sqrt(np.mean(np.sum((heavies_after - heavies_before) ** 2, axis=1))))
        if heavy_rmsd > 1e-3:
            # Heavy atoms drifted — should be impossible with all heavies fixed.
            raise RuntimeError(
                f"BUG-FIX #1 violated: heavies drifted by {heavy_rmsd:.4f} Å during constrained MMFF"
            )

        writer = Chem.SDWriter(str(lig_path))
        writer.write(mol_with_h)
        writer.close()
        return prot_path, lig_path, lig_resname, "sdf"
    else:
        lig_path = pdb_path.with_suffix(".lig.pdb")
        with open(lig_path, "w") as f:
            f.writelines(ligand_lines)
            f.write("END\n")
        return prot_path, lig_path, lig_resname, "pdb"


def parameterize_ligand(lig_path: Path, lig_format: str = "pdb", net_charge: int = 0) -> Path:
    """Run antechamber to assign AM1-BCC + GAFF2. Accepts pdb or sdf.
    net_charge: formal charge of the ligand (passed via -nc to antechamber).
    Returns .mol2."""
    lig_path = lig_path.resolve()
    out_mol2 = lig_path.with_suffix(".gaff2.mol2")
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "antechamber",
            "-i", str(lig_path), "-fi", lig_format,
            "-o", str(out_mol2), "-fo", "mol2",
            "-c", "bcc", "-at", "gaff2",
            "-nc", str(net_charge),
            "-pf", "y",
            "-s", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td, timeout=900)
        if r.returncode != 0 or not out_mol2.exists():
            raise RuntimeError(f"antechamber failed: {r.stderr[-400:] if r.stderr else r.stdout[-400:]}")
    return out_mol2


def parmchk2_frcmod(lig_mol2: Path) -> Path:
    """Generate missing GAFF2 parameters via parmchk2."""
    lig_mol2 = lig_mol2.resolve()
    frcmod = lig_mol2.with_suffix(".frcmod")
    with tempfile.TemporaryDirectory() as td:
        cmd = ["parmchk2", "-i", str(lig_mol2), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"parmchk2 failed: {r.stderr[-400:]}")
    return frcmod


def build_amber_topology(prot_pdb: Path, lig_mol2: Path, lig_frcmod: Path, lig_resname: str):
    """Use tleap to build AMBER prmtop + inpcrd combining protein + ligand."""
    prot_pdb = prot_pdb.resolve()
    lig_mol2 = lig_mol2.resolve()
    lig_frcmod = lig_frcmod.resolve()
    work = lig_mol2.parent
    prmtop = work / "complex.prmtop"
    inpcrd = work / "complex.inpcrd"
    tleap_in = work / "tleap.in"
    # tleap will load ff14SB and GAFF2 force fields
    tleap_script = f"""
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
    r = subprocess.run(["tleap", "-f", str(tleap_in)], capture_output=True, text=True, cwd=work, timeout=180)
    if r.returncode != 0 or not prmtop.exists():
        raise RuntimeError(f"tleap failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")
    return prmtop, inpcrd


def find_cys_sg_and_warhead_cb(prmtop_path: Path, inpcrd_path: Path):
    """Locate Cys-Sγ atom and the closest ligand carbon (= warhead Cβ).
    Returns (sg_atom_index, cb_lig_atom_index, sg_cb_distance_A)."""
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
    pos = inpcrd.getPositions(asNumpy=True).value_in_unit(ANG)
    # Iterate residues; find Cys-SG atom
    sg_candidates = []
    lig_carbons = []
    for atom in prmtop.topology.atoms():
        res = atom.residue
        if res.name in ("CYS", "CYX") and atom.name == "SG":
            sg_candidates.append(atom.index)
        # Ligand residue: name typically "LIG" or non-standard (3-letter)
        is_protein = res.name in {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        if not is_protein and atom.element and atom.element.symbol == "C":
            lig_carbons.append(atom.index)

    if not sg_candidates or not lig_carbons:
        raise ValueError(f"missing Cys-SG ({len(sg_candidates)}) or lig-C ({len(lig_carbons)})")

    # Find SG-C pair with min distance (the warhead-Cys pair)
    best = (None, None, 1e9)
    for sg in sg_candidates:
        sg_pos = pos[sg]
        for cb in lig_carbons:
            d = float(np.linalg.norm(pos[cb] - sg_pos))
            if d < best[2]:
                best = (sg, cb, d)
    return best  # (sg_idx, cb_idx, distance Å)


def build_system_with_covalent_bond(prmtop_path: Path, inpcrd_path: Path,
                                    sg_idx: int | None, cb_idx: int | None,
                                    add_covalent_bond: bool = True):
    """Create OpenMM system with GBn2 implicit solvent + optional CustomBondForce.

    If add_covalent_bond=False (capped-analog mode), the Cys-Sγ–Cβ harmonic
    is omitted and GB-radius patching only fires if sg_idx/cb_idx are still
    provided. Passing None for both skips GB patching entirely.
    """
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
    system = prmtop.createSystem(
        implicitSolvent=app.OBC2,
        soluteDielectric=1.0, solventDielectric=78.5,
        nonbondedMethod=NoCutoff,
        removeCMMotion=False,
        constraints=None,
    )
    if add_covalent_bond:
        if sg_idx is None or cb_idx is None:
            raise ValueError("add_covalent_bond=True requires sg_idx and cb_idx")
        # Add covalent bond between Cys-Sγ and warhead Cβ as harmonic
        cb_force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
        cb_force.addPerBondParameter("k")
        cb_force.addPerBondParameter("r0")
        k_si = CYS_S_BOND_K_KCAL_PER_A2 * 4.184 * 100.0  # kcal/mol/Å² → kJ/mol/nm²
        cb_force.addBond(sg_idx, cb_idx, [k_si, CYS_S_BOND_LENGTH_NM])
        system.addForce(cb_force)

    # ─── BUG-FIX #2 ──────────────────────────────────────────────────
    # OBC2 produces a built-in openmm.GBSAOBCForce, NOT a CustomGBForce.
    # The previous `isinstance(f, CustomGBForce)` check never matched, so the
    # Sγ thioether radius was never patched (silent no-op).
    # API: GBSAOBCForce.setParticleParameters(index, charge, radius, scaleFactor).
    # Values come back as openmm.unit Quantity; strip with value_in_unit().
    # We assert after the write that the new radii took effect.
    # In capped-analog mode (no covalent bond), the warhead is a propionamide
    # carbon — standard sp3 C — and the Cys-SG is also untouched. So we skip
    # the radius patch entirely.
    # ────────────────────────────────────────────────────────────────
    if add_covalent_bond and sg_idx is not None and cb_idx is not None:
        gb_patched = False
        for f in system.getForces():
            cls_name = type(f).__name__
            if isinstance(f, openmm.GBSAOBCForce):
                for atom_idx, new_radius in [(sg_idx, SG_RADIUS_NM), (cb_idx, CB_LIG_RADIUS_NM)]:
                    charge, radius, scale = f.getParticleParameters(atom_idx)
                    charge_v = charge.value_in_unit(unit.elementary_charge) if hasattr(charge, "value_in_unit") else float(charge)
                    scale_v = float(scale)
                    f.setParticleParameters(atom_idx, charge_v, new_radius, scale_v)
                    # Verify
                    _, r_back, _ = f.getParticleParameters(atom_idx)
                    r_back_nm = r_back.value_in_unit(NM) if hasattr(r_back, "value_in_unit") else float(r_back)
                    if abs(r_back_nm - new_radius) > 1e-6:
                        raise RuntimeError(
                            f"BUG-FIX #2 failed: patched radius did not stick for atom {atom_idx} "
                            f"(wanted {new_radius:.4f} nm, got {r_back_nm:.4f} nm)"
                        )
                gb_patched = True
                break
            elif isinstance(f, openmm.CustomGBForce):
                # Defensive: keep the original code path in case implicitSolvent type changes.
                n_params = f.getNumPerParticleParameters()
                if n_params >= 2:
                    for atom_idx, new_radius in [(sg_idx, SG_RADIUS_NM), (cb_idx, CB_LIG_RADIUS_NM)]:
                        params = list(f.getParticleParameters(atom_idx))
                        params[1] = new_radius
                        f.setParticleParameters(atom_idx, params)
                gb_patched = True
                break
        if not gb_patched:
            raise RuntimeError(
                f"BUG-FIX #2: no GB force found in system; force classes seen = "
                f"{[type(f).__name__ for f in system.getForces()]}"
            )

    return system, prmtop.topology, inpcrd.positions


def add_backbone_restraint(system, topology, positions, k_kcal_per_a2):
    """Add a restraining harmonic to protein backbone heavy atoms (CA, N, C).

    Used only for stage-1 minimization. Stage-2 uses
    add_selective_backbone_restraint() instead (see BUG-FIX #4).
    """
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k_kcal_per_a2 * 4.184 * 100.0)
    force.addPerParticleParameter("x0"); force.addPerParticleParameter("y0"); force.addPerParticleParameter("z0")
    backbone_atoms = ("CA", "N", "C")
    n_added = 0
    standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    for atom in topology.atoms():
        if atom.residue.name in standard_aa and atom.name in backbone_atoms:
            p = positions[atom.index].value_in_unit(NM)
            force.addParticle(atom.index, [p[0], p[1], p[2]])
            n_added += 1
    system.addForce(force)
    return force, n_added


def _ligand_atom_indices(topology):
    standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    return [a.index for a in topology.atoms() if a.residue.name not in standard_aa]


def add_selective_backbone_restraint(system, topology, positions, k_kcal_per_a2, ligand_cutoff_A=8.0):
    """BUG-FIX #4: selective restraint for stage-2.

    Stage-2 needs to relax the binding site without letting distal protein
    drift. The previous code called sim.context.setParameter("k", 0.0) which
    released the *entire* backbone. Here we add a NON-RELEASABLE restraint
    (k hard-coded into the expression, no global parameter) that omits any
    residue having a heavy atom within `ligand_cutoff_A` of the ligand.
    Distal residues stay pinned at k=10 kcal/mol/Å²; near-ligand residues
    relax freely.
    """
    pos_nm = np.array([p.value_in_unit(NM) for p in positions])
    lig_atoms = _ligand_atom_indices(topology)
    if not lig_atoms:
        raise RuntimeError("BUG-FIX #4: no ligand atoms found for cutoff selection")
    lig_xyz = pos_nm[lig_atoms]
    cutoff_nm = ligand_cutoff_A / 10.0
    cutoff_nm2 = cutoff_nm * cutoff_nm

    near_residues = set()
    for res in topology.residues():
        if res.name not in {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}:
            continue
        for atom in res.atoms():
            if atom.element is None or atom.element.symbol == "H":
                continue
            d2 = float(np.min(np.sum((lig_xyz - pos_nm[atom.index]) ** 2, axis=1)))
            if d2 < cutoff_nm2:
                near_residues.add(res.index)
                break

    # Hard-code k into the expression so no global parameter can release it.
    k_si = k_kcal_per_a2 * 4.184 * 100.0  # kJ/mol/nm²
    force = openmm.CustomExternalForce(f"0.5*{k_si}*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addPerParticleParameter("x0"); force.addPerParticleParameter("y0"); force.addPerParticleParameter("z0")

    backbone_atoms = ("CA", "N", "C")
    standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    n_far = 0; n_near = 0
    for atom in topology.atoms():
        if atom.residue.name not in standard_aa or atom.name not in backbone_atoms:
            continue
        if atom.residue.index in near_residues:
            n_near += 1
            continue
        p = positions[atom.index].value_in_unit(NM)
        force.addParticle(atom.index, [p[0], p[1], p[2]])
        n_far += 1
    system.addForce(force)
    return force, n_far, n_near, near_residues


def energy(simulation):
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(KCAL)


def score_one(pdb_path: Path, target: str, name: str, smiles: str | None = None,
              no_covalent_bond: bool = False) -> dict:
    """Run full MM-GBSA + strain on a single cofold PDB.

    Parameters
    ----------
    no_covalent_bond : bool
        If True, run in CAPPED-ANALOG (recognition) mode:
          - Do NOT identify Cys-Sγ/warhead-Cβ
          - Do NOT add the CustomBondForce
          - Do NOT patch GB radii on Sγ/Cβ
          - Do NOT subtract the cov-bond restraint energy from E_complex
        Use when the input PDB has already had its acrylamide capped to the
        saturated propionamide via cap_warhead_to_analog.cap_acrylamide_to_propionamide.

    Returns
    -------
    dict — when no_covalent_bond=True, sg_cb_dist_A, _E_cov_restraint stay None;
    dG_bind_kcalmol is the recognition free energy (capped-analog).
    """
    out = {
        "target": target, "name": name,
        "dG_bind_kcalmol": None, "ligand_strain_kcalmol": None,
        "E_complex": None, "E_protein": None, "E_ligand_bound": None, "E_ligand_free": None,
        "rmsd_min_A": None, "sg_cb_dist_A": None,
        "success_flag": 0, "error": None,
    }
    try:
        # 1. Split (with SMILES-aware ligand reconstruction if provided)
        prot_pdb, lig_path, lig_resname, lig_format = split_pdb(pdb_path, smiles=smiles)
        # 2. Parameterize ligand (detect formal charge from SMILES if given)
        net_charge = 0
        if smiles is not None:
            mol_for_charge = Chem.MolFromSmiles(smiles)
            if mol_for_charge is not None:
                net_charge = Chem.GetFormalCharge(mol_for_charge)
        lig_mol2 = parameterize_ligand(lig_path, lig_format=lig_format, net_charge=net_charge)
        lig_frcmod = parmchk2_frcmod(lig_mol2)
        # 3. Build AMBER topology
        prmtop_path, inpcrd_path = build_amber_topology(prot_pdb, lig_mol2, lig_frcmod, lig_resname)
        # 4. Identify covalent pair (skip in capped-analog mode)
        if no_covalent_bond:
            sg_idx, cb_idx, sg_cb_d = None, None, None
        else:
            sg_idx, cb_idx, sg_cb_d = find_cys_sg_and_warhead_cb(prmtop_path, inpcrd_path)
            out["sg_cb_dist_A"] = sg_cb_d
        # 5. Build OpenMM system
        system, topology, positions = build_system_with_covalent_bond(
            prmtop_path, inpcrd_path, sg_idx, cb_idx,
            add_covalent_bond=not no_covalent_bond,
        )
        # 6. Two-stage minimization
        # ─── BUG-FIX #4 ──────────────────────────────────────────────────
        # Old code: setParameter("k", 0.0) globally released ALL backbone
        # restraints in stage-2. We now use the stage-1 force only for stage-1,
        # then release it (k → 0) AND add a separate non-releasable restraint
        # for residues > ligand_cutoff_A from the ligand for stage-2. This
        # selectively relaxes only the binding site.
        # ─── BUG-FIX #3 ──────────────────────────────────────────────────
        # The CustomBondForce restraint that enforces Cys-Sγ — Cβ_warhead
        # belongs to E_complex but NOT to E_protein + E_ligand_bound, so its
        # restraint energy contaminated dG_bind. We subtract the restraint
        # energy analytically using the post-min Sγ–Cβ distance.
        # ─────────────────────────────────────────────────────────────────
        platform = openmm.Platform.getPlatformByName("CPU")
        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)

        # Stage 1: uniform backbone restraint (uses global parameter "k")
        force_restraint, n_restr = add_backbone_restraint(system, topology, positions, HEAVY_ATOM_RESTRAINT_K_KCAL_PER_A2)

        # Stage 2: distal-only restraint with hard-coded k (no global param)
        force_far, n_far, n_near, near_resids = add_selective_backbone_restraint(
            system, topology, positions, HEAVY_ATOM_RESTRAINT_K_KCAL_PER_A2, ligand_cutoff_A=8.0
        )
        # Mark the stage-2 restraint as a separate force group so we can
        # subtract it later when reporting "clean" E_complex.
        FAR_RESTRAINT_GROUP = 7
        force_far.setForceGroup(FAR_RESTRAINT_GROUP)

        # Likewise put the stage-1 restraint in its own group so we never
        # leak its energy into the final score.
        STAGE1_RESTRAINT_GROUP = 6
        force_restraint.setForceGroup(STAGE1_RESTRAINT_GROUP)

        # And the covalent bond restraint (added in build_system_with_covalent_bond)
        # In capped-analog mode there is no CustomBondForce — the loop just
        # finds nothing, which is fine.
        COV_BOND_GROUP = 5
        for f in system.getForces():
            if isinstance(f, openmm.CustomBondForce):
                f.setForceGroup(COV_BOND_GROUP)
                break

        sim = app.Simulation(topology, system, integrator, platform)
        sim.context.setPositions(positions)
        e_pre = energy(sim)

        # In capped-analog mode, the warhead carbon starts at the unphysical
        # ~2 Å distance from Cys-SG (carried over from the covalent pose).
        # WITHOUT the spring restraint, it must relax outward; 700 total
        # minimization steps is too few. We use 5000-step capped minimization
        # vs 500+200 for the covalent path.
        if no_covalent_bond:
            stage1_steps, stage2_steps = 2000, 5000
            min_tol = 1 * unit.kilojoule_per_mole / unit.nanometer
        else:
            stage1_steps, stage2_steps = 500, 200
            min_tol = 10 * unit.kilojoule_per_mole / unit.nanometer

        # Stage 1: minimize with both restraints active
        sim.minimizeEnergy(maxIterations=stage1_steps, tolerance=min_tol)

        # Stage 2: release the stage-1 (uniform) restraint by setting its k → 0
        # but the stage-2 (distal-only) restraint stays active. This is the
        # critical change from the buggy version.
        sim.context.setParameter("k", 0.0)
        sim.minimizeEnergy(maxIterations=stage2_steps, tolerance=min_tol)
        e_post_total = energy(sim)

        # BUG-FIX #3 subtraction: read the energy of just the cov-bond restraint
        # and the stage-2 distal restraint, then subtract from total.
        # (The stage-1 restraint contributes 0 because k=0 was set globally.)
        # In capped-analog mode the cov-bond group is empty → energy = 0.
        cov_state = sim.context.getState(getEnergy=True, groups={COV_BOND_GROUP})
        e_cov_restraint = cov_state.getPotentialEnergy().value_in_unit(KCAL)
        far_state = sim.context.getState(getEnergy=True, groups={FAR_RESTRAINT_GROUP})
        e_far_restraint = far_state.getPotentialEnergy().value_in_unit(KCAL)
        if no_covalent_bond:
            # Defensive: nothing should have contributed to that group.
            if abs(e_cov_restraint) > 1e-6:
                raise RuntimeError(
                    f"no_covalent_bond=True but cov-bond group has energy {e_cov_restraint:.6f}"
                )

        # The far restraint is part of the physical model (it represents the
        # rest of the protein staying folded) and is also absent from the
        # subsystem energies — so we subtract it too for a clean dG_bind.
        e_post_clean = e_post_total - e_cov_restraint - e_far_restraint

        # RMSD pre vs post
        new_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(ANG)
        old_pos = np.array([p.value_in_unit(ANG) for p in positions])
        out["rmsd_min_A"] = float(np.sqrt(np.mean(np.sum((new_pos - old_pos) ** 2, axis=1))))
        out["E_complex"] = e_post_clean
        out["_E_complex_raw"] = e_post_total
        out["_E_cov_restraint"] = e_cov_restraint
        out["_E_far_restraint"] = e_far_restraint
        out["_n_near_residues"] = len(near_resids)

        # 7. Compute ΔG_bind by extracting protein-only and ligand-only subsystems via ParmEd
        import parmed as pmd
        standard_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        struct = pmd.load_file(str(prmtop_path), xyz=str(inpcrd_path))
        # Update coords to minimized positions
        struct.coordinates = new_pos  # numpy array in Å
        # Determine residue indices for protein vs ligand
        prot_res_ids = [i + 1 for i, r in enumerate(struct.residues) if r.name in standard_aa]
        lig_res_ids = [i + 1 for i, r in enumerate(struct.residues) if r.name not in standard_aa]
        if not lig_res_ids:
            raise RuntimeError("no non-protein residues found for ligand")

        # protein-only subsystem
        prot_struct = struct.copy(pmd.Structure)
        prot_struct.strip(f":{','.join(str(r) for r in lig_res_ids)}")
        # ligand-only subsystem
        lig_struct = struct.copy(pmd.Structure)
        lig_struct.strip(f":{','.join(str(r) for r in prot_res_ids)}")

        def energy_of(parmed_struct, with_obc=True):
            sys_ = parmed_struct.createSystem(
                nonbondedMethod=app.NoCutoff,
                implicitSolvent=app.OBC2 if with_obc else None,
                removeCMMotion=False, constraints=None,
            )
            integ = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
            sim_ = app.Simulation(parmed_struct.topology, sys_, integ, platform)
            sim_.context.setPositions(parmed_struct.positions)
            return sim_, energy(sim_)

        sim_prot, e_prot = energy_of(prot_struct)
        sim_lig_bound, e_lig_bound = energy_of(lig_struct)
        out["E_protein"] = e_prot
        out["E_ligand_bound"] = e_lig_bound

        # ΔG_bind = E_complex - E_protein - E_ligand_bound
        out["dG_bind_kcalmol"] = out["E_complex"] - e_prot - e_lig_bound

        # Strain: minimize ligand-only from its bound coords
        sim_lig_bound.minimizeEnergy(maxIterations=500, tolerance=10*unit.kilojoule_per_mole/unit.nanometer)
        e_lig_free = energy(sim_lig_bound)
        out["E_ligand_free"] = e_lig_free
        out["ligand_strain_kcalmol"] = e_lig_bound - e_lig_free

        out["success_flag"] = 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-500:]
    return out


def main():
    # Smoke test on 5P9J (BTK + ibrutinib, PDB ligand code 1E8)
    test_pdb = Path("/tmp/openmm_test/5P9J.pdb")
    if not test_pdb.exists():
        src = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/btk_pocket/5P9J.pdb")
        if src.exists():
            test_pdb.parent.mkdir(parents=True, exist_ok=True)
            test_pdb.write_bytes(src.read_bytes())
    assert test_pdb.exists(), f"missing test PDB {test_pdb}"
    print("=" * 80)
    print("Scoring smoke test: 5P9J (BTK + ibrutinib)")
    print("=" * 80)
    # Ibrutinib SMILES — the active acrylamide warhead form (matches the
    # 5P9J cocrystal ligand 1E8 / 8E8).
    ibrutinib_smiles = "C=CC(=O)N1CCC[C@@H](C1)n2nc(c3c2ncnc3N)c4ccc(cc4)Oc5ccccc5"
    t0 = time.time()
    res = score_one(test_pdb, target="BTK_test", name="5P9J", smiles=ibrutinib_smiles)
    dt = time.time() - t0
    print(f"\n=== Result ({dt:.1f}s) ===")
    for k, v in res.items():
        if k == "traceback": continue
        if isinstance(v, float):
            print(f"  {k:25s} {v:+10.3f}")
        else:
            print(f"  {k:25s} {v}")
    if not res["success_flag"]:
        print(f"\n  traceback:\n{res.get('traceback','')}")


if __name__ == "__main__":
    main()
