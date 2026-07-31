"""Capped-analog MD MM-GBSA via gmx_MMPBSA (Cheng JCTC 2017 + Valdes-Tresanco JCTC 2021).

This is the *field-standard fallback* for covalent kinase inhibitors:
  1. Cap the acrylamide warhead (`C=C-C(=O)-N`) to the saturated propionamide
     (`C-C-C(=O)-N`) — see Cheng JCTC 2017 (EGFR) and Zhu 2015 (Schrödinger).
  2. Run STANDARD non-covalent MD with the saturated analog (no S-C bond
     means no GAFF S-C parameter issue, no CustomBondForce, no leap `bond`
     directive, no CYS->CYX patch).
  3. Score with gmx_MMPBSA -> get `dG_recognition_md_kcalmol` (the K_I proxy).

The "recognition" component dG_recognition = what dG_bind would be if the
covalent bond never formed. It is the part of the binding free energy that
the inhibitor's NON-COVALENT scaffold contributes — and is the closest
thing to a virtual-screening-relevant ranking metric for covalent
inhibitors short of full alchemical FEP.

Pipeline (per ligand):
  1. cap_acrylamide_to_propionamide()                          [RDKit]
  2. split_pdb -> protein + capped-ligand SDF                  [RDKit]
  3. antechamber + parmchk2 on capped ligand                   [AmberTools]
  4. Standard NON-COVALENT tleap build (no bond directive)     [tleap]
       - complex.prmtop / .inpcrd
       - receptor.prmtop / .inpcrd
       - ligand.prmtop / .inpcrd
       - complex_solv.prmtop / .inpcrd (TIP3P box, neutralized)
  5. 1 ns NPT MD (HMR, 4 fs dt, 50 ps equil + 1 ns prod)       [OpenMM]
  6. DCD -> dry NetCDF via cpptraj                             [AmberTools]
  7. gmx_MMPBSA (igb=8, PBRadii=4, salt=0.15, idecomp=2)       [gmx_MMPBSA]
  8. Parse FINAL_RESULTS / per_frame / decomp                  [pure python]

Output dict (matches existing scorer shape + adds capped-specific fields):
    dG_recognition_md_kcalmol     <- mean DELTA TOTAL from FINAL_RESULTS
    dG_recognition_md_std         <- DELTA TOTAL std
    ligand_strain_capped_md_kcalmol  (placeholder; not yet computed)
    n_frames_scored
    residue_contributions         (JSON string)
    success_flag, error, traceback
    wall_md_s, wall_mmpbsa_s, wall_total_s
"""
from __future__ import annotations
import sys, json, time, traceback, tempfile, subprocess, os, argparse, gc, csv, re
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Reuse existing helpers
sys.path.insert(0, str(Path(__file__).parent))
from score_cofold_energy import (
    split_pdb, parameterize_ligand, parmchk2_frcmod,
)
from mdmmgbsa_cheng2017 import run_md, STANDARD_AA
from mdmmgbsa_via_gmxmmpbsa import (
    convert_dcd_to_netcdf_dry,
    parse_final_results,
    parse_per_frame,
    parse_decomp,
)
from cap_warhead_to_analog import cap_acrylamide_to_propionamide


def _find_residues_within_A(prmtop: Path, inpcrd: Path, dist_A: float = 6.0) -> str:
    """Find protein residue numbers within `dist_A` of the ligand in the
    complex (uses ParmEd). Returns a comma-separated string like '344-347,401,414'
    suitable for AMBER MMPBSA.py's print_res field (which requires integer
    residue lists, NOT gmx_MMPBSA's "within X" syntax).
    """
    import parmed as pmd
    import numpy as np

    s = pmd.load_file(str(prmtop), xyz=str(inpcrd))
    lig_res_ids = [i for i, r in enumerate(s.residues) if r.name not in STANDARD_AA]
    if not lig_res_ids:
        # Fallback: just print all
        return "1-9999"
    lig_atom_idx = [a.idx for a in s.atoms
                    if any(a.residue.idx == ri for ri in lig_res_ids)]

    coords = np.array(s.coordinates)  # (N, 3)
    lig_xyz = coords[lig_atom_idx]

    near_res = set()
    d2 = dist_A * dist_A
    for ri, res in enumerate(s.residues):
        if res.name not in STANDARD_AA:
            continue
        for a in res.atoms:
            ax, ay, az = coords[a.idx]
            for lx, ly, lz in lig_xyz:
                if (ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2 <= d2:
                    near_res.add(ri + 1)  # 1-based residue index
                    break
            if (ri + 1) in near_res:
                break

    if not near_res:
        return "1-9999"
    # Compact to ranges
    sorted_res = sorted(near_res)
    ranges = []
    start = prev = sorted_res[0]
    for r in sorted_res[1:]:
        if r == prev + 1:
            prev = r
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = r
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def write_mmpbsa_py_config(out_path: Path, startframe: int = 1, endframe: int = 200,
                           interval: int = 4, decomp: bool = True,
                           print_res_str: str = "all",
                           complex_prmtop: Path | None = None,
                           complex_inpcrd: Path | None = None,
                           decomp_within_A: float = 6.0) -> Path:
    """Write AMBER MMPBSA.py-style input file (NOT gmx_MMPBSA).

    MMPBSA.py accepts AMBER .prmtop directly via -cp. We use MMPBSA.py for
    the capped pipeline since we never leave AMBER format.

    Settings:
      - igb=8 (GB-Neck2; recommended for protein-ligand)
      - saltcon=0.150 M
      - idecomp=2 (per-residue decomp, 1-4 added to EEL/VDW)
      - print_res = integer residue list (e.g. "344-347,401,414")

    MMPBSA.py does NOT support gmx_MMPBSA's print_res="within X" syntax;
    we precompute the list of residues within decomp_within_A of the ligand
    via ParmEd (caller can pass `complex_prmtop` + `complex_inpcrd`).
    """
    if decomp and print_res_str == "all" and complex_prmtop is not None and complex_inpcrd is not None:
        print_res_str = _find_residues_within_A(complex_prmtop, complex_inpcrd, decomp_within_A)
        print(f"[write_mmpbsa_py_config] print_res = {print_res_str}", flush=True)

    decomp_block = (
        f"&decomposition\n"
        f"  idecomp=2, dec_verbose=0, print_res=\"{print_res_str}\",\n"
        f"  csv_format=1,\n"
        f"/\n"
    ) if decomp else ""
    cfg = f"""Input for AMBER MMPBSA.py on capped covalent inhibitor complex
&general
  startframe={startframe}, endframe={endframe}, interval={interval},
  verbose=1, netcdf=1,
/
&gb
  igb=8, saltcon=0.150,
/
{decomp_block}"""
    out_path.write_text(cfg)
    return out_path


def run_mmpbsa_py(complex_prmtop: Path, receptor_prmtop: Path, ligand_prmtop: Path,
                  nc_dry: Path, mmpbsa_in: Path, work: Path,
                  timeout_s: int = 3600, n_mpi: int = 4) -> tuple[Path, Path, Path]:
    """Invoke AmberTools MMPBSA.py.MPI (native, accepts .prmtop, parallelizes
    across frames).

    With decomp=True (idecomp=2), single-thread sander does ~50s/frame on
    ~9K-atom complexes. MMPBSA.py.MPI distributes frames across MPI ranks,
    giving ~n_mpi x speedup. Caller should pick n_mpi <= n_cpu_cores - 1.

    Returns (final_results_dat, per_frame_csv, decomp_csv).
    """
    final_dat = work / "FINAL_RESULTS_MMPBSA.dat"
    per_frame_csv = work / "per_frame.csv"
    decomp_csv = work / "FINAL_DECOMP_MMPBSA.csv"

    # mpirun -np N MMPBSA.py.MPI -O -i ... ; uses MPI to distribute frames
    cmd = [
        "mpirun", "--allow-run-as-root", "-np", str(n_mpi),
        "MMPBSA.py.MPI",
        "-O",
        "-i", str(mmpbsa_in),
        "-cp", str(complex_prmtop),
        "-rp", str(receptor_prmtop),
        "-lp", str(ligand_prmtop),
        "-y", str(nc_dry),
        "-o", str(final_dat),
        "-eo", str(per_frame_csv),
        "-do", str(decomp_csv),
    ]
    print(f"[run_mmpbsa_py] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work,
                       timeout=timeout_s)
    print(f"[run_mmpbsa_py] stdout tail:\n{r.stdout[-1500:]}", flush=True)
    if r.returncode != 0:
        print(f"[run_mmpbsa_py] stderr tail:\n{r.stderr[-1500:]}", flush=True)
        # Known issue: MMPBSA.py.MPI 14.0 with idecomp=2 sometimes raises
        # "DecompError: Mismatch in number of decomp terms!" at the
        # final-parse stage AFTER FINAL_RESULTS_MMPBSA.dat is already
        # written. We tolerate this case if the result file exists.
        if final_dat.exists() and final_dat.stat().st_size > 1000:
            print(f"[run_mmpbsa_py] WARN: rc={r.returncode} but FINAL_RESULTS exists "
                  f"({final_dat.stat().st_size} bytes) - tolerating, decomp may be missing",
                  flush=True)
        else:
            raise RuntimeError(f"MMPBSA.py.MPI failed with rc={r.returncode}; see logs above")
    if not final_dat.exists():
        raise RuntimeError(f"MMPBSA.py.MPI did not produce {final_dat}")
    return final_dat, per_frame_csv, decomp_csv


def build_topologies_capped(
    prot_pdb: Path,
    lig_mol2: Path,
    lig_frcmod: Path,
    lig_resname: str,
    buffer_A: float = 10.0,
) -> dict[str, Path]:
    """Build the 4 prmtops gmx_MMPBSA needs, NON-COVALENT (no `bond` directive).

    This is the capped-analog topology: ligand is propionamide (saturated),
    protein is unchanged, no cross-component bond.
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
        # Aliases for compat
        "complex_dry_prmtop": work / "complex.prmtop",
        "complex_dry_inpcrd": work / "complex.inpcrd",
    }

    tleap_in = work / "tleap_capped.in"
    # Use mbondi3 PBRadii so that igb=8 (GB-Neck2) can run on these prmtops.
    # Default PARSE radii contain H atoms with r=0.8 which igb>=7 rejects.
    tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p

set default PBRadii mbondi3

LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}

# Non-covalent complex (capped ligand, NO bond directive)
COMPLEX = combine {{PROT LIG}}
saveamberparm COMPLEX {paths["complex_prmtop"]} {paths["complex_inpcrd"]}

# Receptor alone
RECEPTOR = combine {{PROT}}
saveamberparm RECEPTOR {paths["receptor_prmtop"]} {paths["receptor_inpcrd"]}

# Ligand alone (capped/saturated)
LIGAND = combine {{LIG}}
saveamberparm LIGAND {paths["ligand_prmtop"]} {paths["ligand_inpcrd"]}

# Solvated complex for MD
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
        raise RuntimeError(f"tleap capped build failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    for k, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"tleap did not produce expected file: {k} -> {p}")
    return paths


def score_one_md_capped(pdb_path: Path, smiles: str, target: str, name: str,
                        md_ns: float = 1.0, n_frames: int = 200,
                        equil_ps: float = 50.0, platform_name: str = "CUDA",
                        workdir: Path | None = None) -> dict:
    """Capped-analog gmx_MMPBSA scorer for covalent kinase inhibitors.

    Returns dict with keys:
        target, name,
        dG_recognition_md_kcalmol, dG_recognition_md_std,
        ligand_strain_capped_md_kcalmol,
        n_frames_scored,
        ggas_kcalmol, gsolv_kcalmol,
        dG_per_frame_mean, dG_per_frame_std,
        residue_contributions (JSON string),
        md_ns,
        wall_cap_s, wall_md_s, wall_mmpbsa_s, wall_total_s,
        success_flag, error, traceback,
        capped_smiles,
    """
    out = {
        "target": target, "name": name,
        "dG_recognition_md_kcalmol": None, "dG_recognition_md_std": None,
        "ligand_strain_capped_md_kcalmol": None,
        "n_frames_scored": 0,
        "ggas_kcalmol": None, "gsolv_kcalmol": None,
        "dG_per_frame_mean": None, "dG_per_frame_std": None,
        "residue_contributions": None,
        "md_ns": md_ns,
        "wall_cap_s": None, "wall_md_s": None,
        "wall_mmpbsa_s": None, "wall_total_s": None,
        "success_flag": 0, "error": None, "traceback": None,
        "capped_smiles": None,
    }
    t_total0 = time.time()
    try:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="cappedgmx_"))
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"[score_one_md_capped] workdir = {workdir}", flush=True)

        # --- Stage 1: cap the acrylamide warhead ---
        t_cap0 = time.time()
        local_pdb = workdir / pdb_path.name
        if local_pdb != pdb_path:
            local_pdb.write_bytes(pdb_path.read_bytes())

        capped_smiles, capped_pdb = cap_acrylamide_to_propionamide(smiles, local_pdb)
        if capped_smiles is None or capped_pdb is None:
            raise RuntimeError(
                "No acrylamide warhead found in SMILES — cap_acrylamide_to_propionamide returned None"
            )
        out["capped_smiles"] = capped_smiles
        out["wall_cap_s"] = time.time() - t_cap0
        print(f"[score_one_md_capped] capped: {smiles[:50]}... -> {capped_smiles[:50]}...", flush=True)

        # --- Stage 2/3: build prmtops (with cache to skip antechamber on retry) ---
        # Cache hit requires: all 8 prmtop/inpcrd files exist AND prmtops > 100KB
        # (inpcrds for small ligands can be tiny - don't enforce a size floor on them)
        cache_prmtops = ["complex.prmtop", "receptor.prmtop", "ligand.prmtop", "complex_solv.prmtop"]
        cache_inpcrds = ["complex.inpcrd", "receptor.inpcrd", "ligand.inpcrd", "complex_solv.inpcrd"]
        cache_ok = (
            all((workdir / fn).exists() and (workdir / fn).stat().st_size > (1000 if fn == "ligand.prmtop" else 100000)
                for fn in cache_prmtops)
            and all((workdir / fn).exists() for fn in cache_inpcrds)
        )
        if cache_ok:
            print(f"[score_one_md_capped] reusing cached prmtops in {workdir}", flush=True)
            paths = {
                "complex_prmtop": workdir / "complex.prmtop",
                "complex_inpcrd": workdir / "complex.inpcrd",
                "receptor_prmtop": workdir / "receptor.prmtop",
                "receptor_inpcrd": workdir / "receptor.inpcrd",
                "ligand_prmtop": workdir / "ligand.prmtop",
                "ligand_inpcrd": workdir / "ligand.inpcrd",
                "complex_solv_prmtop": workdir / "complex_solv.prmtop",
                "complex_solv_inpcrd": workdir / "complex_solv.inpcrd",
                "complex_dry_prmtop": workdir / "complex.prmtop",
                "complex_dry_inpcrd": workdir / "complex.inpcrd",
            }
        else:
            prot_pdb, lig_path, lig_resname, lig_format = split_pdb(capped_pdb, smiles=capped_smiles)
            net_charge = 0
            m = Chem.MolFromSmiles(capped_smiles)
            if m is not None:
                net_charge = Chem.GetFormalCharge(m)
            lig_mol2 = parameterize_ligand(lig_path, lig_format=lig_format, net_charge=net_charge)
            lig_frcmod = parmchk2_frcmod(lig_mol2)

            paths = build_topologies_capped(
                prot_pdb, lig_mol2, lig_frcmod, lig_resname, buffer_A=10.0,
            )

        # Sanity: complex.prmtop must be > 100 KB or tleap silently failed
        if paths["complex_prmtop"].stat().st_size < 100000:
            raise RuntimeError(
                f"complex.prmtop is only {paths['complex_prmtop'].stat().st_size} bytes - "
                f"tleap likely failed silently. See {workdir}/leap.log"
            )

        # --- Stage 4: MD (capped ligand, no covalent bond, dummy sg/cb indices) ---
        # run_md no longer uses the sg/cb indices for force-field purposes
        # (they're kept only as diagnostics in the original Cheng pipeline),
        # so we pass (0, 0) here. The covalent bond is NOT in the topology.
        dcd_path = workdir / "prod.dcd"
        nc_dry = workdir / "trajectory_dry.nc"
        # Cache: if MD + cpptraj already done, skip both
        if dcd_path.exists() and dcd_path.stat().st_size > 10_000_000 and \
           nc_dry.exists() and nc_dry.stat().st_size > 1_000_000:
            print(f"[score_one_md_capped] reusing cached MD trajectory ({dcd_path.stat().st_size//1024//1024} MB DCD, "
                  f"{nc_dry.stat().st_size//1024//1024} MB dry NetCDF)", flush=True)
            out["wall_md_s"] = 0.0
        else:
            t_md0 = time.time()
            run_md(paths["complex_solv_prmtop"], paths["complex_solv_inpcrd"],
                   sg_idx_solv=0, cb_idx_solv=0,
                   out_dcd=dcd_path, ns=md_ns, save_every_ps=5.0,
                   equil_ps=equil_ps, dt_fs=4.0, platform_name=platform_name)
            out["wall_md_s"] = time.time() - t_md0

            # --- Stage 5: DCD -> dry NetCDF ---
            nc_dry = convert_dcd_to_netcdf_dry(
                dcd_path, paths["complex_solv_prmtop"], paths["complex_dry_prmtop"],
                workdir,
            )

        # --- Stage 6: MMPBSA.py (AmberTools native; accepts .prmtop directly,
        # unlike gmx_MMPBSA 1.6.5 which now requires GROMACS .top) ---
        n_frames_actual = max(1, int(md_ns * 1000.0 / 5.0))  # 5 ps stride
        endframe = min(n_frames, n_frames_actual)
        # interval=10 -> 20 evenly-spaced frames from 200 (gives converged
        # mean ± std for dG_recognition; MPI x 4 finishes in ~3 min/component).
        mmpbsa_in = write_mmpbsa_py_config(
            workdir / "mmpbsa.in",
            startframe=1, endframe=endframe, interval=10,
            decomp=True, decomp_within_A=6.0,
            complex_prmtop=paths["complex_prmtop"],
            complex_inpcrd=paths["complex_inpcrd"],
        )

        t_g0 = time.time()
        final_dat, per_frame_csv, decomp_csv = run_mmpbsa_py(
            paths["complex_prmtop"], paths["receptor_prmtop"],
            paths["ligand_prmtop"], nc_dry, mmpbsa_in, workdir,
            timeout_s=3600,
        )
        out["wall_mmpbsa_s"] = time.time() - t_g0

        # --- Stage 7: parse results ---
        fin = parse_final_results(final_dat)
        out["dG_recognition_md_kcalmol"] = fin["dG_bind_kcalmol"]
        out["dG_recognition_md_std"] = fin["dG_bind_std"]
        out["ggas_kcalmol"] = fin["ggas_kcalmol"]
        out["gsolv_kcalmol"] = fin["gsolv_kcalmol"]

        pf = parse_per_frame(per_frame_csv)
        out["n_frames_scored"] = pf["n_frames_scored"]
        out["dG_per_frame_mean"] = pf["dG_per_frame_mean"]
        out["dG_per_frame_std"] = pf["dG_per_frame_std"]

        residues = parse_decomp(decomp_csv)
        if residues:
            out["residue_contributions"] = json.dumps(residues, sort_keys=True)

        if out["dG_recognition_md_kcalmol"] is not None:
            out["success_flag"] = 1
        else:
            out["error"] = "could not parse DELTA TOTAL from FINAL_RESULTS_MMPBSA.dat"

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-1500:]
    out["wall_total_s"] = time.time() - t_total0
    return out


def _smoke_5p9j_fetch(work: Path) -> Path:
    """Fetch 5P9J PDB from RCSB if not present locally."""
    pdb = work / "5P9J.pdb"
    if pdb.exists():
        return pdb
    work.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = "https://files.rcsb.org/download/5P9J.pdb"
    urllib.request.urlretrieve(url, str(pdb))
    return pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="5P9J BTK+ibrutinib smoke test")
    ap.add_argument("--pdb", type=str, default=None)
    ap.add_argument("--smiles", type=str, default=None)
    ap.add_argument("--name", type=str, default="unnamed")
    ap.add_argument("--target", type=str, default="TARGET")
    ap.add_argument("--md_ns", type=float, default=1.0)
    ap.add_argument("--equil_ps", type=float, default=50.0)
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--platform", type=str, default="CUDA")
    ap.add_argument("--workdir", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        # Smoke = rank-1 ZAP70 cofold (Boltz-naming, acrylamide warhead).
        # 5P9J ibrutinib was deprecated as smoke because the RCSB PDB uses
        # crystallographic atom names (CAA, CAD...) that don't match the
        # Boltz-naming convention the capper expects.  The rank-1 ZAP70
        # cofold IS a field-relevant validation in its own right.
        zap70_name = "000611_Tier_3_v3_LibInvent_lock_611"
        zap70_root = Path.home() / "data" / "cys346_predictions" / zap70_name
        pdb = zap70_root / f"{zap70_name}_model_0.pdb"
        smiles = "C=CC(=O)N1Cc2cccc(C(=O)NC(=O)NCCN(C)S(C)(=O)=O)c2C1"
        target, name = "ZAP70_C346_smoke", zap70_name
        workdir = Path(args.workdir) if args.workdir else Path("/tmp/cappedgmx_smoke_zap70")
    else:
        assert args.pdb and args.smiles, "need --pdb and --smiles"
        pdb = Path(args.pdb)
        smiles = args.smiles
        target = args.target
        name = args.name
        workdir = Path(args.workdir) if args.workdir else None

    print(f"=== Capped-Analog MD MM-GBSA (gmx_MMPBSA) ===")
    smiles_preview = (smiles[:60] + "...") if smiles else "None"
    print(f"pdb={pdb} smiles={smiles_preview} md_ns={args.md_ns} equil_ps={args.equil_ps} platform={args.platform}")
    t0 = time.time()
    res = score_one_md_capped(pdb, smiles, target, name, md_ns=args.md_ns,
                              n_frames=args.n_frames, equil_ps=args.equil_ps,
                              platform_name=args.platform, workdir=workdir)
    dt = time.time() - t0
    print(f"\n--- Result ({dt:.1f}s) ---")
    for k, v in res.items():
        if k in ("traceback", "residue_contributions"):
            continue
        if isinstance(v, float):
            print(f"  {k:35s} {v:+10.3f}")
        else:
            print(f"  {k:35s} {v}")
    if res.get("residue_contributions"):
        rc = json.loads(res["residue_contributions"])
        print(f"\n  Top 10 most-favorable per-residue contributions (kcal/mol):")
        for k, v in sorted(rc.items(), key=lambda kv: kv[1])[:10]:
            print(f"    {k:10s} {v:+8.3f}")
    if not res["success_flag"]:
        print(f"\n  traceback:\n{res.get('traceback','')}")
    sys.exit(0 if res["success_flag"] else 1)


if __name__ == "__main__":
    main()
