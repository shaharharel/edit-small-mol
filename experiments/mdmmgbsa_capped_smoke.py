"""Standalone capped-analog MM-GBSA smoke test for 5P9J (BTK + ibrutinib).

Replaces dead worker `a4695cbc35cc89115`. Caps the warhead (acrylamide ->
propionamide) so we can run NON-covalent MM-GBSA (no `bond COMPLEX.X.SG ...`
in tleap), removing the cross-component bonded params that have been
breaking gmx_MMPBSA's ante-MMPBSA splitter.

Pipeline (single ligand):
  1. Split 5P9J -> protein.pdb + ligand.pdb + extract original SMILES
  2. Cap acrylamide -> propionamide via cap_acrylamide_to_propionamide
  3. Build NEW complex from protein + capped ligand (no bond directive!)
  4. tleap (ff14SB + GAFF2, NO bond line) -> 4 prmtops
  5. 1 ns MD on T4 (CUDA, dt=2 fs, no HMR), 100 ps NVT heat + 100 ps NPT density
  6. cpptraj DCD -> dry NetCDF
  7. gmx_MMPBSA -> dG_bind, residue contributions
  8. Write ~/results/mdmmgbsa_smoke_capped.csv

If gmx_MMPBSA fails (atom-order mismatch etc.), fallback to native MMPBSA.py
(AmberTools, no GROMACS wrapper) preserves work.

Pass criteria: dG_bind in [-15, -3], success_flag=1, n_frames>=100.
"""
from __future__ import annotations
import sys, json, time, traceback, tempfile, subprocess, os, csv, re
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse existing builders + MD runner
from score_cofold_energy import (
    split_pdb, parameterize_ligand, parmchk2_frcmod,
)
from mdmmgbsa_cheng2017 import run_md
from mdmmgbsa_via_gmxmmpbsa import (
    convert_dcd_to_netcdf_dry, write_mmpbsa_config,
    run_gmx_mmpbsa, parse_final_results, parse_per_frame, parse_decomp,
)
from cap_warhead_to_analog import cap_acrylamide_to_propionamide


# ---- Capped-analog topology builder (NO covalent bond) ---------------------

def build_topologies_no_bond(
    prot_pdb: Path,
    lig_mol2: Path,
    lig_frcmod: Path,
    buffer_A: float = 10.0,
) -> dict[str, Path]:
    """Build the 4 prmtops gmx_MMPBSA needs, with NO cross-component bond.

    This is the capped-analog variant of build_topologies_for_gmxmmpbsa in
    mdmmgbsa_via_gmxmmpbsa.py. The capped ligand (propionamide) is non-covalent,
    so we simply combine PROT + LIG without any `bond` directive.
    """
    prot_pdb = prot_pdb.resolve()
    lig_mol2 = lig_mol2.resolve()
    lig_frcmod = lig_frcmod.resolve()
    work = lig_mol2.parent

    paths = {
        "complex_prmtop": work / "complex.prmtop",
        "complex_inpcrd": work / "complex.inpcrd",
        "receptor_prmtop": work / "receptor.prmtop",
        "receptor_inpcrd": work / "receptor.inpcrd",
        "ligand_prmtop": work / "ligand.prmtop",
        "ligand_inpcrd": work / "ligand.inpcrd",
        "complex_solv_prmtop": work / "complex_solv.prmtop",
        "complex_solv_inpcrd": work / "complex_solv.inpcrd",
        "complex_dry_prmtop": work / "complex.prmtop",
        "complex_dry_inpcrd": work / "complex.inpcrd",
    }

    tleap_in = work / "tleap_extended.in"
    tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p

LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}

# Capped complex: NO covalent bond (analog, non-covalent)
COMPLEX = combine {{PROT LIG}}
saveamberparm COMPLEX {paths["complex_prmtop"]} {paths["complex_inpcrd"]}

RECEPTOR = combine {{PROT}}
saveamberparm RECEPTOR {paths["receptor_prmtop"]} {paths["receptor_inpcrd"]}

LIGAND = combine {{LIG}}
saveamberparm LIGAND {paths["ligand_prmtop"]} {paths["ligand_inpcrd"]}

solvateBox COMPLEX TIP3PBOX {buffer_A} iso
addions COMPLEX Na+ 0
addions COMPLEX Cl- 0
saveamberparm COMPLEX {paths["complex_solv_prmtop"]} {paths["complex_solv_inpcrd"]}

quit
"""
    tleap_in.write_text(tleap_script)
    r = subprocess.run(["tleap", "-f", str(tleap_in)], capture_output=True,
                       text=True, cwd=work, timeout=300)
    if r.returncode != 0 or not paths["complex_solv_prmtop"].exists():
        leap_log = work / "leap.log"
        leap_tail = leap_log.read_text()[-1500:] if leap_log.exists() else "(no leap.log)"
        raise RuntimeError(
            f"tleap (no-bond) failed:\n"
            f"--- stdout ---\n{r.stdout[-800:]}\n"
            f"--- stderr ---\n{r.stderr[-400:]}\n"
            f"--- leap.log tail ---\n{leap_tail}"
        )
    for k, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"tleap did not produce expected file: {k} -> {p}")
    return paths


# ---- MD without forcing covalent restraint --------------------------------

def run_md_noncovalent(solv_prmtop: Path, solv_inpcrd: Path,
                       out_dcd: Path, ns: float = 1.0,
                       equil_ps: float = 100.0, save_every_ps: float = 5.0,
                       dt_fs: float = 2.0, platform_name: str = "CUDA") -> None:
    """1 ns MD with 100 ps NVT heat + 100 ps NPT equil + production.

    Uses OpenMM directly (no HMR, no covalent restraint). Matches the surface
    used by `score_one_md_v2` but without the SG/CB indices since there's no
    covalent bond to harmonically restrain.
    """
    from openmm import (
        unit, Platform, LangevinIntegrator, MonteCarloBarostat,
        AmberPrmtopFile, AmberInpcrdFile,
    )
    from openmm.app import (
        Simulation, DCDReporter, StateDataReporter, PME, HBonds,
    )
    import openmm as mm

    prmtop = AmberPrmtopFile(str(solv_prmtop))
    inpcrd = AmberInpcrdFile(str(solv_inpcrd))

    system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds, rigidWater=True,
    )

    integrator = LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond,
    )

    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception:
        print(f"[run_md_noncovalent] platform {platform_name} unavailable, falling back to CPU")
        platform = Platform.getPlatformByName("CPU")

    sim = Simulation(prmtop.topology, system, integrator, platform)
    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # 1) Minimize
    print("[run_md_noncovalent] minimizing...")
    sim.minimizeEnergy(maxIterations=5000)

    # 2) NVT heat (100 ps)
    print(f"[run_md_noncovalent] NVT heat for {equil_ps} ps...")
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    n_heat_steps = int(equil_ps * 1000.0 / dt_fs)
    sim.step(n_heat_steps)

    # 3) NPT density equil (100 ps) -- add barostat
    print(f"[run_md_noncovalent] NPT density equil for {equil_ps} ps...")
    barostat = MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 25)
    system.addForce(barostat)
    sim.context.reinitialize(preserveState=True)
    n_npt_steps = int(equil_ps * 1000.0 / dt_fs)
    sim.step(n_npt_steps)

    # 4) Production (ns)
    n_prod_steps = int(ns * 1e6 / dt_fs)
    save_stride_steps = max(1, int(save_every_ps * 1000.0 / dt_fs))
    print(f"[run_md_noncovalent] production for {ns} ns "
          f"({n_prod_steps} steps, save every {save_stride_steps} steps)...")
    sim.reporters.append(DCDReporter(str(out_dcd), save_stride_steps))
    sim.reporters.append(StateDataReporter(
        sys.stdout, save_stride_steps,
        step=True, time=True, potentialEnergy=True, temperature=True, speed=True,
    ))
    sim.step(n_prod_steps)
    del sim


# ---- Fallback: native MMPBSA.py (no GROMACS wrapper) ----------------------

def run_native_mmpbsa(complex_prmtop: Path, receptor_prmtop: Path,
                     ligand_prmtop: Path, nc_dry: Path,
                     work: Path, endframe: int = 200,
                     timeout_s: int = 3600) -> dict:
    """AmberTools native MMPBSA.py fallback if gmx_MMPBSA fails."""
    cfg = work / "mmpbsa_native.in"
    cfg.write_text(f"""Single-trajectory MMPBSA via native MMPBSA.py
&general
  startframe=1, endframe={endframe}, interval=1,
  verbose=1, keep_files=0,
/
&gb
  igb=8, saltcon=0.150,
/
""")
    final_dat = work / "FINAL_RESULTS_MMPBSA_native.dat"
    cmd = [
        "MMPBSA.py", "-O",
        "-i", str(cfg),
        "-o", str(final_dat),
        "-cp", str(complex_prmtop),
        "-rp", str(receptor_prmtop),
        "-lp", str(ligand_prmtop),
        "-y", str(nc_dry),
    ]
    print(f"[run_native_mmpbsa] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work, timeout=timeout_s)
    print(f"[run_native_mmpbsa] stdout tail:\n{r.stdout[-1200:]}")
    if r.returncode != 0:
        print(f"[run_native_mmpbsa] stderr tail:\n{r.stderr[-1200:]}")
        raise RuntimeError(f"native MMPBSA.py failed rc={r.returncode}")
    return parse_final_results(final_dat)


# ---- Smoke driver ---------------------------------------------------------

def smoke_one_capped(pdb_path: Path, workdir: Path, md_ns: float = 1.0,
                     equil_ps: float = 100.0, platform_name: str = "CUDA") -> dict:
    # Hardcoded ibrutinib SMILES (5P9J ligand 8E8) -- reading from PDB loses bond orders.
    IBRUTINIB_SMILES = (
        "C=CC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c1ncnc2N"
    )
    # Capped (propionamide) analog of ibrutinib.
    PROPIONAMIDE_IBRUTINIB_SMILES = (
        "CCC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c1ncnc2N"
    )

    out = {
        "name": "5P9J_capped",
        "target": "BTK_capped",
        "dG_bind_md_kcalmol": None,
        "dG_bind_md_std": None,
        "n_frames_scored": 0,
        "ggas_kcalmol": None, "gsolv_kcalmol": None,
        "dG_per_frame_mean": None, "dG_per_frame_std": None,
        "residue_contributions": None,
        "md_ns": md_ns,
        "wall_md_s": None, "wall_mmpbsa_s": None, "wall_total_s": None,
        "success_flag": 0, "error": None, "used_fallback": 0,
        "capped_smiles": None,
    }
    t0 = time.time()
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        local_pdb = workdir / pdb_path.name
        if local_pdb != pdb_path:
            local_pdb.write_bytes(pdb_path.read_bytes())

        # Stage 1: Build capped ligand SDF aligned to crystal pose.
        # 5P9J uses PDB-component naming (CAA, CAD, ...) NOT Boltz CanonicalRank naming,
        # so split_pdb / cap_acrylamide_to_propionamide both fail. Instead we use RDKit
        # substructure-match between the parent (acrylamide) and capped (propionamide)
        # SMILES, embed a fresh 3D conformer of the capped mol, then align it to the
        # crystal acrylamide by mapping common-substructure heavy atoms.
        from rdkit.Chem import AllChem

        out["capped_smiles"] = PROPIONAMIDE_IBRUTINIB_SMILES
        print(f"[smoke] parent SMILES: {IBRUTINIB_SMILES}")
        print(f"[smoke] capped SMILES: {PROPIONAMIDE_IBRUTINIB_SMILES}")

        # Extract crystal ligand atoms (resname 8E8) from 5P9J PDB
        crystal_lig_lines = []
        crystal_prot_lines = []
        lig_resname = None
        for line in local_pdb.read_text().splitlines(keepends=True):
            if line.startswith("ATOM"):
                crystal_prot_lines.append(line)
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname in {"HOH", "WAT", "NA", "K", "MG", "ZN", "CL", "CA", "BR"}:
                    continue
                lig_resname = resname
                crystal_lig_lines.append(line)
            elif line.startswith(("TER", "END")):
                crystal_prot_lines.append(line)
        if not crystal_lig_lines:
            raise RuntimeError("no HETATM ligand found in 5P9J PDB")
        print(f"[smoke] crystal ligand: {lig_resname} ({len(crystal_lig_lines)} atoms)")

        prot_pdb = local_pdb.with_suffix(".prot.pdb")
        prot_pdb.write_text("".join(crystal_prot_lines) + "END\n")

        # Read crystal coords into an RDKit mol with element-only bond perception
        crystal_lig_pdb = local_pdb.with_suffix(".crystal_lig.pdb")
        crystal_lig_pdb.write_text("".join(crystal_lig_lines) + "END\n")
        crystal_mol = Chem.MolFromPDBFile(str(crystal_lig_pdb), removeHs=True,
                                          sanitize=False, proximityBonding=True)
        if crystal_mol is None:
            raise RuntimeError("RDKit could not read crystal ligand PDB")

        # Build capped mol with 3D coords; embed once, then constrained-align to
        # parent crystal pose via the propionamide-vs-acrylamide common substructure
        # (everything except the terminal vinyl CH=CH2).
        capped_mol = Chem.MolFromSmiles(PROPIONAMIDE_IBRUTINIB_SMILES)
        capped_mol = Chem.AddHs(capped_mol)
        if AllChem.EmbedMolecule(capped_mol, randomSeed=42) != 0:
            if AllChem.EmbedMolecule(capped_mol, randomSeed=7) != 0:
                raise RuntimeError("3D embed failed for capped ibrutinib")
        AllChem.MMFFOptimizeMolecule(capped_mol, maxIters=200)

        # Align capped -> crystal via maximum common substructure
        from rdkit.Chem import rdFMCS
        capped_heavy = Chem.RemoveHs(capped_mol)
        mcs = rdFMCS.FindMCS([capped_heavy, crystal_mol],
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             completeRingsOnly=False, timeout=60)
        if mcs.numAtoms < 10:
            raise RuntimeError(f"MCS too small: {mcs.numAtoms} atoms")
        patt = Chem.MolFromSmarts(mcs.smartsString)
        match_capped = capped_heavy.GetSubstructMatch(patt)
        match_crystal = crystal_mol.GetSubstructMatch(patt)
        if not match_capped or not match_crystal:
            raise RuntimeError("MCS match failed")
        atom_map = list(zip(match_capped, match_crystal))
        print(f"[smoke] MCS atoms matched: {len(atom_map)}/{mcs.numAtoms}")

        # Heavy-index in capped_heavy == heavy-index in capped_mol (Hs appended last)
        rms = AllChem.AlignMol(capped_mol, crystal_mol, atomMap=atom_map)
        print(f"[smoke] align RMS to crystal: {rms:.3f} A")

        # Write capped ligand as SDF for parameterize_ligand
        lig_sdf = workdir / "capped_lig.sdf"
        w = Chem.SDWriter(str(lig_sdf)); w.write(capped_mol); w.close()
        print(f"[smoke] wrote capped ligand SDF: {lig_sdf}")

        # Stage 2: Parameterize capped ligand
        m_capped = Chem.MolFromSmiles(PROPIONAMIDE_IBRUTINIB_SMILES)
        net_charge = Chem.GetFormalCharge(m_capped) if m_capped is not None else 0
        lig_mol2 = parameterize_ligand(lig_sdf, lig_format="sdf", net_charge=net_charge)
        lig_frcmod = parmchk2_frcmod(lig_mol2)
        prot_pdb2 = prot_pdb

        # Stage 3: tleap build (NO bond directive)
        paths = build_topologies_no_bond(prot_pdb2, lig_mol2, lig_frcmod, buffer_A=10.0)

        # Stage 4: MD on T4
        dcd_path = workdir / "prod.dcd"
        t_md0 = time.time()
        run_md_noncovalent(
            paths["complex_solv_prmtop"], paths["complex_solv_inpcrd"],
            out_dcd=dcd_path, ns=md_ns, equil_ps=equil_ps,
            save_every_ps=5.0, dt_fs=2.0, platform_name=platform_name,
        )
        out["wall_md_s"] = time.time() - t_md0

        # Stage 5: DCD -> dry NetCDF
        nc_dry = convert_dcd_to_netcdf_dry(
            dcd_path, paths["complex_solv_prmtop"],
            paths["complex_dry_prmtop"], workdir,
        )

        # Stage 6: gmx_MMPBSA, with native fallback
        n_frames_actual = max(1, int(md_ns * 1000.0 / 5.0))
        endframe = min(200, n_frames_actual)
        mmpbsa_in = write_mmpbsa_config(workdir / "mmpbsa.in",
                                        startframe=1, endframe=endframe,
                                        interval=1, decomp=True, decomp_within_A=6.0)
        t_g0 = time.time()
        try:
            final_dat, per_frame_csv, decomp_csv = run_gmx_mmpbsa(
                paths["complex_prmtop"], paths["receptor_prmtop"],
                paths["ligand_prmtop"], nc_dry, mmpbsa_in, workdir, timeout_s=3600,
            )
            fin = parse_final_results(final_dat)
            pf = parse_per_frame(per_frame_csv)
            residues = parse_decomp(decomp_csv)
        except Exception as e_gmx:
            print(f"[smoke] gmx_MMPBSA failed: {e_gmx}; falling back to MMPBSA.py")
            out["used_fallback"] = 1
            fin = run_native_mmpbsa(
                paths["complex_prmtop"], paths["receptor_prmtop"],
                paths["ligand_prmtop"], nc_dry, workdir, endframe=endframe,
            )
            pf = {"n_frames_scored": endframe, "dG_per_frame_mean": fin.get("dG_bind_kcalmol"),
                  "dG_per_frame_std": fin.get("dG_bind_std")}
            residues = {}
        out["wall_mmpbsa_s"] = time.time() - t_g0

        out["dG_bind_md_kcalmol"] = fin["dG_bind_kcalmol"]
        out["dG_bind_md_std"] = fin["dG_bind_std"]
        out["ggas_kcalmol"] = fin.get("ggas_kcalmol")
        out["gsolv_kcalmol"] = fin.get("gsolv_kcalmol")
        out["n_frames_scored"] = pf["n_frames_scored"]
        out["dG_per_frame_mean"] = pf["dG_per_frame_mean"]
        out["dG_per_frame_std"] = pf["dG_per_frame_std"]
        if residues:
            out["residue_contributions"] = json.dumps(residues, sort_keys=True)

        # Pass criteria
        dG = out["dG_bind_md_kcalmol"]
        nf = out["n_frames_scored"]
        if dG is not None and -15.0 <= dG <= -3.0 and nf >= 100:
            out["success_flag"] = 1
        else:
            out["error"] = (
                f"pass criteria not met: dG={dG} (need [-15,-3]), n_frames={nf} (need >=100)"
            )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["traceback"] = traceback.format_exc()[-2000:]
    out["wall_total_s"] = time.time() - t0
    return out


def write_result_csv(res: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop traceback from CSV (too long); print it to log instead
    tb = res.pop("traceback", None)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        w.writeheader()
        w.writerow(res)
    if tb:
        (csv_path.parent / f"{csv_path.stem}_traceback.log").write_text(tb)


def main():
    pdb = Path.home() / "data" / "5P9J.pdb"
    if not pdb.exists():
        alt = Path("/tmp/openmm_test/5P9J.pdb")
        if alt.exists():
            pdb = alt
        else:
            raise SystemExit(f"missing 5P9J PDB at {pdb} or {alt}")

    out_csv = Path.home() / "results" / "mdmmgbsa_smoke_capped.csv"
    workdir = Path(tempfile.mkdtemp(prefix="capsmoke_"))
    print(f"=== capped-analog MM-GBSA smoke ===")
    print(f"pdb={pdb} workdir={workdir} out_csv={out_csv}")

    res = smoke_one_capped(pdb, workdir, md_ns=1.0, equil_ps=100.0, platform_name="CUDA")

    print(f"\n--- Result ---")
    for k, v in res.items():
        if k in ("traceback", "residue_contributions"):
            continue
        if isinstance(v, float):
            print(f"  {k:30s} {v:+10.3f}")
        else:
            print(f"  {k:30s} {v}")
    if res.get("residue_contributions"):
        rc = json.loads(res["residue_contributions"])
        print(f"\n  Top per-residue contributions (kcal/mol):")
        for k, v in sorted(rc.items(), key=lambda kv: kv[1])[:10]:
            print(f"    {k:10s} {v:+8.3f}")

    write_result_csv(res, out_csv)
    print(f"\n[smoke] wrote {out_csv}")
    print(f"[smoke] success_flag={res['success_flag']} dG={res['dG_bind_md_kcalmol']}")
    sys.exit(0 if res["success_flag"] else 1)


if __name__ == "__main__":
    main()
