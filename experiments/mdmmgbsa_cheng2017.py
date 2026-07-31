"""MD-averaged MM-GBSA per Cheng JCTC 2017 protocol for covalent kinase ligands.

Pipeline (per ligand):
  1. Build AMBER topology (ff14SB + GAFF2 + AM1-BCC) — reuse score_cofold_energy.py
  2. Solvate explicitly with TIP3P (10 Å buffer)
  3. NVT/NPT equilibration (50 ps each)
  4. NPT production MD at 300 K, 1 bar, HMR 4 fs (default 1 ns = 250K steps)
  5. Save every 50th frame → ~200 snapshots (5 ps stride)
  6. For each frame: strip waters via ParmEd, single-point GBn2 MM-GBSA
     using the covalent + GB-radius-patched system from score_cofold_energy
  7. Mean across frames = dG_bind_md

Returns dict with dG_bind_md_kcalmol, std, ligand_strain_md, n_frames, etc.

Reuses pure functions from score_cofold_energy:
  - split_pdb, parameterize_ligand, parmchk2_frcmod, build_amber_topology
  - find_cys_sg_and_warhead_cb, CYS_S_BOND_LENGTH_NM, CYS_S_BOND_K_KCAL_PER_A2,
    SG_RADIUS_NM, CB_LIG_RADIUS_NM

Usage:
    python mdmmgbsa_cheng2017.py --smoke         # run 5P9J smoke
    python mdmmgbsa_cheng2017.py --pdb <p> --smiles <s> --name <n>
"""
from __future__ import annotations
import sys, json, csv, time, traceback, tempfile, subprocess, os, argparse, gc
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
from openmm import unit

# Reuse builders from existing scorer (assumed sibling file)
from score_cofold_energy import (
    split_pdb, parameterize_ligand, parmchk2_frcmod, build_amber_topology,
    find_cys_sg_and_warhead_cb,
    CYS_S_BOND_LENGTH_NM, CYS_S_BOND_K_KCAL_PER_A2,
    SG_RADIUS_NM, CB_LIG_RADIUS_NM,
    KCAL, ANG, NM,
)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Force constants
STANDARD_AA = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}


def _discover_leap_resnum_offset(lig_mol2: Path, lig_frcmod: Path, prot_pdb: Path, work: Path) -> int:
    """Find the *leap-internal* sequence number offset of the COMPLEX unit.

    Leap addresses residues via ``UNIT.<seq>.<atom>``, where ``<seq>`` is the
    leap-internal "sequence number" assigned at build time — *not* the 1-based
    position in the residue list and *not* the original PDB residue number.

    The offset depends on the order of object construction inside leap (LIG is
    loaded first as a single residue → gets seq=1; then PROT loaded → PROT's
    residues are sequenced 2..N+1; then COMPLEX = combine{PROT LIG} → COMPLEX
    inherits these seqs). Empirically the offset is +393 for the BTK 5P9J
    case but can differ per system, so we discover it dynamically.

    Strategy: build COMPLEX in a probe leap and ask ``desc COMPLEX.<n>`` for
    a small range of ``n`` until we find the first valid hit; that gives us
    the offset (leap_seq_of_first_residue - 1).

    Returns the integer offset such that leap_seq = parmed_idx + 1 + offset.
    """
    tleap_in = work / "tleap_probe_offset.in"
    # Issue `desc COMPLEX` — its "Contents:" section lists every residue as
    # `R<NAME SEQ>` in order; first line's SEQ tells us the offset.
    script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}
COMPLEX = combine {{PROT LIG}}
desc COMPLEX
quit
"""
    tleap_in.write_text(script)
    r = subprocess.run(["tleap", "-f", str(tleap_in)], capture_output=True, text=True, cwd=work, timeout=120)
    out = r.stdout
    # Find the "Contents:" line that follows the UNIT name we just described,
    # then parse the FIRST `R<NAME SEQ>` after it.
    import re
    m_contents = re.search(r"Contents:\s*\n((?:R<[^>]+>\s*\n)+)", out)
    if not m_contents:
        raise RuntimeError(f"tleap offset-probe failed; could not find Contents block:\n{out[-1500:]}\n{r.stderr[-300:]}")
    first_line = m_contents.group(1).splitlines()[0].strip()
    m_first = re.match(r"R<\s*\S+\s+(\d+)\s*>", first_line)
    if not m_first:
        raise RuntimeError(f"tleap offset-probe failed; could not parse first residue line: {first_line!r}")
    first_seq = int(m_first.group(1))
    # first_seq is the leap-seq of parmed list-idx 0 (i.e. 1-based position 1).
    # So leap_seq = parmed_idx + 1 + offset → offset = first_seq - 1.
    offset = first_seq - 1
    return offset


def _resolve_cys_lig_bond_atoms(prmtop_dry: Path, inpcrd_dry: Path,
                                 lig_mol2: Path, lig_frcmod: Path, prot_pdb: Path) -> tuple[int, int, str, str, float]:
    """Identify the covalent bond endpoints in *leap residue/atom-name terms*.

    Runs the existing distance-based heuristic on a dry-built prmtop to find
    the Cys-Sγ and the closest ligand carbon (= warhead Cβ), then resolves
    leap-friendly identifiers (leap-internal residue seq number in the COMPLEX
    unit, and atom *names* such as ``SG`` / ``CAA``) via ParmEd + leap probe.

    The leap-internal sequence number is **not** the 1-based list position in
    the prmtop — leap maintains its own global symbol counter, so an offset
    is applied (discovered via a leap probe).

    Returns (cys_resnum_leap, lig_resnum_leap, sg_atom_name, cb_atom_name, sg_cb_dist_A).
    """
    import parmed as pmd
    sg_idx, cb_idx, d = find_cys_sg_and_warhead_cb(prmtop_dry, inpcrd_dry)
    struct = pmd.load_file(str(prmtop_dry), xyz=str(inpcrd_dry))
    sg_atom = struct.atoms[sg_idx]
    cb_atom = struct.atoms[cb_idx]
    # Discover leap's residue-number offset for the COMPLEX unit.
    work = prmtop_dry.parent
    offset = _discover_leap_resnum_offset(lig_mol2, lig_frcmod, prot_pdb, work)
    # parmed residue idx is 0-based; leap addresses by (idx + 1 + offset).
    cys_resnum = sg_atom.residue.idx + 1 + offset
    lig_resnum = cb_atom.residue.idx + 1 + offset
    print(f"[_resolve_cys_lig_bond_atoms] leap offset={offset}; "
          f"cys parmed_idx={sg_atom.residue.idx} → leap_seq={cys_resnum}; "
          f"lig parmed_idx={cb_atom.residue.idx} → leap_seq={lig_resnum}")
    return cys_resnum, lig_resnum, sg_atom.name, cb_atom.name, d


def solvate_complex(prot_pdb: Path, lig_mol2: Path, lig_frcmod: Path, lig_resname: str, buffer_A: float = 10.0):
    """Build solvated AMBER topology with TIP3P.

    The Cys-Sγ ↔ warhead-Cβ covalent bond is added *natively* via leap's
    ``bond`` directive (after ``combine``, before ``saveamberparm``) so that
    AMBER sees it as a real 1-2 bonded pair (using GAFF2 default S-C bond
    parameters) and the 1-2 LJ nonbonded interaction is excluded. This avoids
    the ~100 kcal/mol artificial repulsion that arises when the S-C atoms sit
    at the covalent 1.81 Å distance but are treated as a non-bonded pair with
    preferred ~3.5 Å LJ separation.

    Two-pass strategy:
      1. Dry-build (no bond, no solvate) → use distance heuristic + ParmEd to
         resolve the *names* of Cys-SG and warhead-Cβ and their residue indices
         inside the COMPLEX unit.
      2. Final build with the explicit leap ``bond`` directive, producing both
         the dry (saveamberparm before solvate) and solvated topologies.

    Returns (prmtop_solv, inpcrd_solv, dry_prmtop, dry_inpcrd).
    """
    prot_pdb = prot_pdb.resolve()
    lig_mol2 = lig_mol2.resolve()
    lig_frcmod = lig_frcmod.resolve()
    work = lig_mol2.parent
    prmtop_solv = work / "complex_solv.prmtop"
    inpcrd_solv = work / "complex_solv.inpcrd"
    prmtop_dry = work / "complex_dry.prmtop"
    inpcrd_dry = work / "complex_dry.inpcrd"

    # --- Pass 1: dry build (no bond) to discover atom names + residue indices.
    probe_prmtop = work / "probe_dry.prmtop"
    probe_inpcrd = work / "probe_dry.inpcrd"
    tleap_probe = work / "tleap_probe.in"
    probe_script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}
COMPLEX = combine {{PROT LIG}}
saveamberparm COMPLEX {probe_prmtop} {probe_inpcrd}
quit
"""
    tleap_probe.write_text(probe_script)
    r = subprocess.run(["tleap", "-f", str(tleap_probe)], capture_output=True, text=True, cwd=work, timeout=180)
    if r.returncode != 0 or not probe_prmtop.exists():
        raise RuntimeError(f"tleap probe failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    cys_resnum, lig_resnum, sg_name, cb_name, sg_cb_d = _resolve_cys_lig_bond_atoms(
        probe_prmtop, probe_inpcrd, lig_mol2, lig_frcmod, prot_pdb,
    )
    print(f"[solvate_complex] native S-C bond: COMPLEX.{cys_resnum}.{sg_name} -- COMPLEX.{lig_resnum}.{cb_name}  (initial d={sg_cb_d:.2f} Å)")

    # --- Pass 2: final build with native bond + solvate.
    tleap_in = work / "tleap_solv.in"
    tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}
COMPLEX = combine {{PROT LIG}}
bond COMPLEX.{cys_resnum}.{sg_name} COMPLEX.{lig_resnum}.{cb_name}
saveamberparm COMPLEX {prmtop_dry} {inpcrd_dry}
solvateBox COMPLEX TIP3PBOX {buffer_A} iso
addions COMPLEX Na+ 0
addions COMPLEX Cl- 0
saveamberparm COMPLEX {prmtop_solv} {inpcrd_solv}
quit
"""
    tleap_in.write_text(tleap_script)
    r = subprocess.run(["tleap", "-f", str(tleap_in)], capture_output=True, text=True, cwd=work, timeout=300)
    if r.returncode != 0 or not prmtop_solv.exists():
        raise RuntimeError(f"tleap solvate (with native S-C bond) failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")
    return prmtop_solv, inpcrd_solv, prmtop_dry, inpcrd_dry


def add_covalent_bond_force(system, prmtop_path, sg_idx, cb_idx):
    """Add CustomBondForce for Cys-Sγ → Cβ_warhead. Mutates system; returns force.

    Used in BOTH the solvated production MD and the per-frame GBSA single-points.
    """
    cb_force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    cb_force.addPerBondParameter("k")
    cb_force.addPerBondParameter("r0")
    k_si = CYS_S_BOND_K_KCAL_PER_A2 * 4.184 * 100.0  # kcal/mol/Å² → kJ/mol/nm²
    cb_force.addBond(sg_idx, cb_idx, [k_si, CYS_S_BOND_LENGTH_NM])
    system.addForce(cb_force)
    return cb_force


def patch_gb_radii(system, sg_idx, cb_idx):
    """Patch Sγ thioether and warhead Cβ radii in GBSAOBCForce. Returns True if patched."""
    for f in system.getForces():
        if isinstance(f, openmm.GBSAOBCForce):
            for atom_idx, new_radius in [(sg_idx, SG_RADIUS_NM), (cb_idx, CB_LIG_RADIUS_NM)]:
                charge, _radius, scale = f.getParticleParameters(atom_idx)
                ch_v = charge.value_in_unit(unit.elementary_charge) if hasattr(charge, "value_in_unit") else float(charge)
                sc_v = float(scale)
                f.setParticleParameters(atom_idx, ch_v, new_radius, sc_v)
            return True
    return False


def build_solvated_system(prmtop_path: Path, hmr: bool = True):
    """Build OpenMM system for solvated production MD with PME, HBond constraints, optional HMR.

    Returns (system, topology, prmtop_obj). Caller adds covalent bond + integrator.
    """
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    if hmr:
        system = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=True,
            hydrogenMass=1.5 * unit.amu,
        )
    else:
        system = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=True,
        )
    return system, prmtop.topology, prmtop


def run_md(prmtop_solv: Path, inpcrd_solv: Path, sg_idx_solv: int, cb_idx_solv: int,
           out_dcd: Path, ns: float = 1.0, save_every_ps: float = 5.0,
           equil_ps: float = 50.0, dt_fs: float = 4.0,
           platform_name: str = "CUDA") -> tuple[Path, int]:
    """Run NPT MD. HMR + 4 fs timestep, Langevin 300 K, MonteCarloBarostat 1 bar.

    Steps = ns * 1000 / (dt_fs/1000) = ns * 1e6 / dt_fs
    Returns (dcd_path, n_frames_saved).
    """
    system, topology, prm = build_solvated_system(prmtop_solv, hmr=True)
    # Covalent Cys-Sγ–Cβ bond is now baked into the AMBER topology natively
    # (via leap `bond` directive), so we do NOT add a CustomBondForce here.
    # The previous CustomBondForce overlapped with the now-real bond and also
    # left S-C as a 1-2 nonbonded pair (LJ repulsion artifact).
    _ = (sg_idx_solv, cb_idx_solv)  # kept for API compat / diagnostics

    # NPT barostat: 1 bar, 25 step interval
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))

    inpcrd = app.AmberInpcrdFile(str(inpcrd_solv))
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtoseconds
    )
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except Exception:
        platform = openmm.Platform.getPlatformByName("CPU")
    properties = {}
    if platform.getName() == "CUDA":
        properties = {"Precision": "mixed"}
    sim = app.Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # Minimization
    sim.minimizeEnergy(maxIterations=2000, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    # Equilibration (NPT)
    equil_steps = int(equil_ps * 1000.0 / dt_fs)
    if equil_steps > 0:
        sim.step(equil_steps)

    # Production
    prod_steps = int(ns * 1e6 / dt_fs)
    save_every_steps = int(save_every_ps * 1000.0 / dt_fs)
    if save_every_steps < 1:
        save_every_steps = 1
    sim.reporters.append(app.DCDReporter(str(out_dcd), save_every_steps, enforcePeriodicBox=False))

    sim.step(prod_steps)
    n_frames = prod_steps // save_every_steps

    # Release GPU memory
    del sim, system, integrator
    gc.collect()
    return out_dcd, n_frames


def _ligand_atom_indices(topology):
    return [a.index for a in topology.atoms() if a.residue.name not in STANDARD_AA]


def score_trajectory(prmtop_dry: Path, inpcrd_dry: Path, dcd_path: Path,
                     sg_idx_dry: int, cb_idx_dry: int,
                     max_frames: int = 200, platform_name: str = "CUDA") -> list[dict]:
    """For each frame of the SOLVATED trajectory, strip waters/ions and run
    single-point GBn2 (OBC2) MM-GBSA on the dry complex.

    Uses ParmEd to map solvated -> dry. Since solvated topology has same protein+
    ligand atom indices at the front (tleap adds waters/ions AFTER), we can
    simply take the first N atoms of each frame.
    """
    import parmed as pmd

    # Load dry topology to know how many atoms to keep.
    prmtop_dry_amber = app.AmberPrmtopFile(str(prmtop_dry))
    n_dry_atoms = prmtop_dry_amber.topology.getNumAtoms()

    # Build the GB system on the DRY topology. The Cys-Sγ ↔ warhead-Cβ bond is
    # already in the prmtop natively (leap `bond` directive in solvate_complex),
    # so we no longer add a CustomBondForce — its energy is already inside the
    # standard HarmonicBondForce term of the complex.
    inpcrd_dry_amber = app.AmberInpcrdFile(str(inpcrd_dry))
    sys_complex = prmtop_dry_amber.createSystem(
        implicitSolvent=app.OBC2, soluteDielectric=1.0, solventDielectric=78.5,
        nonbondedMethod=app.NoCutoff, removeCMMotion=False, constraints=None,
    )
    if not patch_gb_radii(sys_complex, sg_idx_dry, cb_idx_dry):
        raise RuntimeError("GB radius patch failed in dry complex system")
    # No separate cov-bond force group; e_cov := 0 below.

    # Build protein-only + ligand-only systems via ParmEd (no cov bond on them)
    struct_dry = pmd.load_file(str(prmtop_dry), xyz=str(inpcrd_dry))
    prot_res_ids = [i + 1 for i, r in enumerate(struct_dry.residues) if r.name in STANDARD_AA]
    lig_res_ids = [i + 1 for i, r in enumerate(struct_dry.residues) if r.name not in STANDARD_AA]
    prot_struct = struct_dry.copy(pmd.Structure)
    prot_struct.strip(f":{','.join(str(r) for r in lig_res_ids)}")
    lig_struct = struct_dry.copy(pmd.Structure)
    lig_struct.strip(f":{','.join(str(r) for r in prot_res_ids)}")

    sys_prot = prot_struct.createSystem(
        nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2,
        removeCMMotion=False, constraints=None,
    )
    sys_lig = lig_struct.createSystem(
        nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2,
        removeCMMotion=False, constraints=None,
    )

    # Platforms
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except Exception:
        platform = openmm.Platform.getPlatformByName("CPU")

    def _make_sim(sys_, top_, positions):
        integ = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
        s = app.Simulation(top_, sys_, integ, platform)
        s.context.setPositions(positions)
        return s

    # Persistent sims (reuse contexts across frames)
    sim_complex = _make_sim(sys_complex, prmtop_dry_amber.topology, inpcrd_dry_amber.positions)
    sim_prot = _make_sim(sys_prot, prot_struct.topology, prot_struct.positions)
    sim_lig = _make_sim(sys_lig, lig_struct.topology, lig_struct.positions)

    # Read trajectory with parmed (handles DCD generically)
    traj = pmd.load_file(str(dcd_path), top=str(prmtop_dry_amber).replace(".prmtop", "")) \
        if False else None
    # Better: use MDAnalysis if needed, but parmed can read DCD with explicit topology.
    import MDAnalysis as mda
    u = mda.Universe(str(prmtop_dry), str(dcd_path), topology_format="PRMTOP", format="DCD")
    # The DCD was written from the SOLVATED simulation, which has many more atoms.
    # We need to load the SOLVATED topology to read DCD, then take first n_dry_atoms.
    # Try to find solvated prmtop next to dcd
    prmtop_solv_path = dcd_path.parent / "complex_solv.prmtop"
    if prmtop_solv_path.exists():
        u = mda.Universe(str(prmtop_solv_path), str(dcd_path), topology_format="PRMTOP", format="DCD")
    else:
        raise RuntimeError("could not locate solvated prmtop for DCD reading")

    total_frames = len(u.trajectory)
    if total_frames <= max_frames:
        stride = 1
    else:
        stride = total_frames // max_frames
    sg_cb_dists = []

    per_frame = []
    n_scored = 0
    for ts in u.trajectory[::stride]:
        if n_scored >= max_frames:
            break
        all_pos = u.atoms.positions.copy()  # Å
        dry_pos_A = all_pos[:n_dry_atoms]
        dry_pos_nm = dry_pos_A / 10.0

        # Set positions on all three sims
        sim_complex.context.setPositions(dry_pos_nm * unit.nanometer)
        # protein-only: take first len(prot_struct.atoms) atoms (tleap orders PROT before LIG)
        n_prot = len(prot_struct.atoms)
        sim_prot.context.setPositions(dry_pos_nm[:n_prot] * unit.nanometer)
        sim_lig.context.setPositions(dry_pos_nm[n_prot:n_prot + len(lig_struct.atoms)] * unit.nanometer)

        e_complex_total = sim_complex.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)
        # Native AMBER bond is part of the standard energy; no separate
        # restraint to subtract.
        e_cov = 0.0
        e_complex = e_complex_total
        e_prot = sim_prot.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)
        e_lig = sim_lig.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)
        dG_bind = e_complex - e_prot - e_lig

        # Sγ–Cβ distance this frame
        sg_xyz = dry_pos_A[sg_idx_dry]
        cb_xyz = dry_pos_A[cb_idx_dry]
        sg_cb_d = float(np.linalg.norm(sg_xyz - cb_xyz))
        sg_cb_dists.append(sg_cb_d)

        per_frame.append({
            "frame": int(ts.frame),
            "E_complex": e_complex, "E_protein": e_prot, "E_ligand_bound": e_lig,
            "dG_bind": dG_bind, "sg_cb_dist_A": sg_cb_d,
        })
        n_scored += 1

    # Free contexts
    del sim_complex, sim_prot, sim_lig
    del sys_complex, sys_prot, sys_lig
    gc.collect()
    return per_frame


def score_one_md(pdb_path: Path, smiles: str | None, target: str, name: str,
                 md_ns: float = 1.0, n_frames: int = 200, equil_ps: float = 50.0,
                 platform_name: str = "CUDA",
                 workdir: Path | None = None) -> dict:
    """Full pipeline for one ligand."""
    out = {
        "target": target, "name": name,
        "dG_bind_md_kcalmol": None, "dG_bind_md_std": None,
        "ligand_strain_md_kcalmol": None,
        "n_frames_scored": 0,
        "sg_cb_dist_md_mean": None, "sg_cb_dist_md_std": None,
        "md_ns": md_ns,
        "wall_md_s": None, "wall_gbsa_s": None, "wall_total_s": None,
        "success_flag": 0, "error": None,
    }
    t_total0 = time.time()
    try:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="mdmmgbsa_"))
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        # Stage 1: split + parameterize (reuse from score_cofold_energy)
        # Copy PDB into workdir so split_pdb writes the .prot/.lig files there
        local_pdb = workdir / pdb_path.name
        if local_pdb != pdb_path:
            local_pdb.write_bytes(pdb_path.read_bytes())
        prot_pdb, lig_path, lig_resname, lig_format = split_pdb(local_pdb, smiles=smiles)

        net_charge = 0
        if smiles is not None:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                net_charge = Chem.GetFormalCharge(m)
        lig_mol2 = parameterize_ligand(lig_path, lig_format=lig_format, net_charge=net_charge)
        lig_frcmod = parmchk2_frcmod(lig_mol2)

        # Stage 2: build BOTH dry and solvated topologies
        prmtop_solv, inpcrd_solv, prmtop_dry, inpcrd_dry = solvate_complex(
            prot_pdb, lig_mol2, lig_frcmod, lig_resname, buffer_A=10.0
        )

        # Stage 3: identify Cys-SG / warhead-Cβ in BOTH topologies
        # (atom indexing differs: solv has waters appended, dry doesn't, but
        # protein+ligand indices are identical at the front)
        sg_idx_dry, cb_idx_dry, sg_cb_d0 = find_cys_sg_and_warhead_cb(prmtop_dry, inpcrd_dry)
        # In solv, same indices for protein+ligand (tleap appends solvent)
        sg_idx_solv, cb_idx_solv = sg_idx_dry, cb_idx_dry

        # Stage 4: production MD
        dcd_path = workdir / "prod.dcd"
        t_md0 = time.time()
        run_md(prmtop_solv, inpcrd_solv, sg_idx_solv, cb_idx_solv,
               out_dcd=dcd_path, ns=md_ns, save_every_ps=5.0,
               equil_ps=equil_ps, dt_fs=4.0, platform_name=platform_name)
        out["wall_md_s"] = time.time() - t_md0

        # Stage 5: per-frame GBSA
        t_g0 = time.time()
        per_frame = score_trajectory(
            prmtop_dry, inpcrd_dry, dcd_path,
            sg_idx_dry, cb_idx_dry,
            max_frames=n_frames, platform_name=platform_name,
        )
        out["wall_gbsa_s"] = time.time() - t_g0

        if not per_frame:
            raise RuntimeError("score_trajectory returned 0 frames")
        dGs = np.array([f["dG_bind"] for f in per_frame])
        sgs = np.array([f["sg_cb_dist_A"] for f in per_frame])
        e_lig_bound = np.array([f["E_ligand_bound"] for f in per_frame])

        out["dG_bind_md_kcalmol"] = float(np.mean(dGs))
        out["dG_bind_md_std"] = float(np.std(dGs))
        out["n_frames_scored"] = int(len(per_frame))
        out["sg_cb_dist_md_mean"] = float(np.mean(sgs))
        out["sg_cb_dist_md_std"] = float(np.std(sgs))

        # Strain proxy: difference between mean E_ligand_bound (MD-sampled) and
        # the minimized free-ligand energy. Use the LAST per-frame coords and
        # run a quick minimize-in-vacuum-with-GB.
        import parmed as pmd
        struct_dry = pmd.load_file(str(prmtop_dry), xyz=str(inpcrd_dry))
        prot_res_ids = [i + 1 for i, r in enumerate(struct_dry.residues) if r.name in STANDARD_AA]
        lig_res_ids = [i + 1 for i, r in enumerate(struct_dry.residues) if r.name not in STANDARD_AA]
        lig_struct = struct_dry.copy(pmd.Structure)
        lig_struct.strip(f":{','.join(str(r) for r in prot_res_ids)}")
        sys_lig = lig_struct.createSystem(
            nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2,
            removeCMMotion=False, constraints=None,
        )
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
        integ = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
        sim_lig = app.Simulation(lig_struct.topology, sys_lig, integ, platform)
        sim_lig.context.setPositions(lig_struct.positions)
        sim_lig.minimizeEnergy(maxIterations=2000, tolerance=10*unit.kilojoule_per_mole/unit.nanometer)
        e_lig_free = sim_lig.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)
        out["ligand_strain_md_kcalmol"] = float(np.mean(e_lig_bound) - e_lig_free)
        del sim_lig, sys_lig
        gc.collect()

        out["success_flag"] = 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-800:]
    out["wall_total_s"] = time.time() - t_total0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run 5P9J smoke test")
    ap.add_argument("--pdb", type=str, default=None)
    ap.add_argument("--smiles", type=str, default=None)
    ap.add_argument("--name", type=str, default="unnamed")
    ap.add_argument("--target", type=str, default="TARGET")
    ap.add_argument("--md_ns", type=float, default=1.0)
    ap.add_argument("--equil_ps", type=float, default=50.0)
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--platform", type=str, default="CUDA")
    args = ap.parse_args()

    if args.smoke:
        # Test 5P9J BTK + ibrutinib. PDB atom names are PDB-style (CAA, NAB,
        # OAC...), not Boltz canonical-rank style — pass smiles=None so
        # split_pdb takes the HETATM-direct branch. This is OK because
        # 5P9J's PDB ligand already has correct heavy-atom positions and Hs
        # need only be added by antechamber (which accepts PDB without Hs).
        pdb = Path.home() / "data" / "5P9J.pdb"
        if not pdb.exists():
            # Fallback path used in score_cofold_energy
            pdb = Path("/tmp/openmm_test/5P9J.pdb")
        assert pdb.exists(), f"missing test pdb at {pdb}"
        smiles = None
        target, name = "BTK_test", "5P9J"
    else:
        assert args.pdb and args.smiles, "need --pdb and --smiles"
        pdb = Path(args.pdb)
        smiles = args.smiles
        target = args.target
        name = args.name

    print(f"=== MD MM-GBSA Cheng-2017 ===")
    smiles_preview = (smiles[:60] + "...") if smiles else "None"
    print(f"pdb={pdb} smiles={smiles_preview} md_ns={args.md_ns} equil_ps={args.equil_ps} platform={args.platform}")
    t0 = time.time()
    res = score_one_md(pdb, smiles, target, name, md_ns=args.md_ns,
                       n_frames=args.n_frames, equil_ps=args.equil_ps,
                       platform_name=args.platform)
    dt = time.time() - t0
    print(f"\n--- Result ({dt:.1f}s) ---")
    for k, v in res.items():
        if k == "traceback":
            continue
        if isinstance(v, float):
            print(f"  {k:30s} {v:+10.3f}")
        else:
            print(f"  {k:30s} {v}")
    if not res["success_flag"]:
        print(f"\n  traceback:\n{res.get('traceback','')}")


if __name__ == "__main__":
    main()
