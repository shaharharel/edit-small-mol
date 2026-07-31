"""MD-averaged MM-GBSA via gmx_MMPBSA (Valdes-Tresanco JCTC 2021).

Replaces our custom OpenMM/ParmEd MM-GBSA scorer (which hit a cascade of bugs:
NaN explosion -> LJ repulsion artifact -> POINTERS keyerror) with the
battle-tested gmx_MMPBSA pipeline that natively handles covalent inhibitors
via its `ante-MMPBSA.py` cross-component bond auto-detection.

Pipeline (per ligand):
  1. Build AMBER topologies (extended): complex.prmtop + receptor.prmtop +
     ligand.prmtop + complex_solv.prmtop. All four come out of a single
     extended tleap script. The native Cys-SG -- Cb_warhead bond is added
     in the complex via leap's `bond` directive (as before).
  2. Solvate explicitly with TIP3P (10 A buffer) - reuse existing code.
  3. Run 1 ns MD with heat + soft-ramp (reuse run_md from cheng2017 script;
     the MD itself worked fine through smoke v3 - the bug was post-MD only).
  4. Convert DCD -> AMBER NetCDF via cpptraj, strip waters.
  5. Write mmpbsa.in and invoke `gmx_MMPBSA -O -i mmpbsa.in -cp complex.prmtop
     -rp receptor.prmtop -lp ligand.prmtop -ct trajectory_dry.nc -nogui ...`.
  6. Parse FINAL_RESULTS_MMPBSA.dat (mean dG_bind) + per_frame.csv (std) +
     decomp output (per-residue contributions within 6 A of ligand).
  7. Return dict with same shape as `score_one_md` in mdmmgbsa_cheng2017.py
     plus a new `residue_contributions` JSON field.

Usage:
    python mdmmgbsa_via_gmxmmpbsa.py --smoke --md_ns 1.0
"""
from __future__ import annotations
import sys, json, time, traceback, tempfile, subprocess, os, argparse, gc, csv, re
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Reuse the existing pure-Python builders + MD runner
from score_cofold_energy import (
    split_pdb, parameterize_ligand, parmchk2_frcmod,
)
from mdmmgbsa_cheng2017 import (
    _resolve_cys_lig_bond_atoms,
    run_md,
    STANDARD_AA,
)
from score_cofold_energy import find_cys_sg_and_warhead_cb


def build_topologies_for_gmxmmpbsa(
    prot_pdb: Path,
    lig_mol2: Path,
    lig_frcmod: Path,
    lig_resname: str,
    buffer_A: float = 10.0,
) -> dict[str, Path]:
    """Extended topology builder: produce ALL four prmtops gmx_MMPBSA needs.

    Output files in lig_mol2.parent:
      - complex.prmtop / complex.inpcrd  (protein + ligand, native S-C bond)
      - receptor.prmtop / receptor.inpcrd (protein alone, no ligand)
      - ligand.prmtop / ligand.inpcrd     (ligand alone, no protein)
      - complex_solv.prmtop / complex_solv.inpcrd (with TIP3P box, for MD)
      - complex_dry.prmtop / complex_dry.inpcrd   (same as complex.* - kept
        for compatibility with mdmmgbsa_cheng2017's run_md interface)

    Two-pass strategy (same as solvate_complex in mdmmgbsa_cheng2017):
      Pass 1: dry build without bond -> resolve atom names + leap residue
              numbers for Cys-SG and warhead-Cb.
      Pass 2: full extended build with `bond` directive and all 4 prmtops.

    Note on covalent bond handling:
      - The bond IS added to COMPLEX (so the MD sees it as a real AMBER bond,
        no LJ repulsion at 1.81 A).
      - RECEPTOR and LIGAND are built from independent leap units (`combine`
        of PROT alone, `combine` of LIG alone) - no cross-component bond.
        gmx_MMPBSA's `ante-MMPBSA.py` auto-strips bonded params that span
        components when it splits prmtops, but here we give it pre-split
        prmtops directly via -rp / -lp, so it doesn't need to split.
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
        # Aliases used elsewhere in the codebase
        "complex_dry_prmtop": work / "complex.prmtop",
        "complex_dry_inpcrd": work / "complex.inpcrd",
    }

    # --- Pass 1: dry probe to resolve atom names + leap residue numbers ---
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
    r = subprocess.run(["tleap", "-f", str(tleap_probe)], capture_output=True,
                       text=True, cwd=work, timeout=180)
    if r.returncode != 0 or not probe_prmtop.exists():
        raise RuntimeError(f"tleap probe failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    cys_resnum, lig_resnum, sg_name, cb_name, sg_cb_d = _resolve_cys_lig_bond_atoms(
        probe_prmtop, probe_inpcrd, lig_mol2, lig_frcmod, prot_pdb,
    )
    print(f"[build_topologies_for_gmxmmpbsa] native S-C bond: "
          f"COMPLEX.{cys_resnum}.{sg_name} -- COMPLEX.{lig_resnum}.{cb_name}  "
          f"(initial d={sg_cb_d:.2f} A)")

    # --- Pass 2: extended build (complex with bond, receptor, ligand, solv) ---
    tleap_in = work / "tleap_extended.in"
    tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p

LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
PROT = loadpdb {prot_pdb}

# Complex with native covalent bond (the "dry" complex)
COMPLEX = combine {{PROT LIG}}
bond COMPLEX.{cys_resnum}.{sg_name} COMPLEX.{lig_resnum}.{cb_name}
saveamberparm COMPLEX {paths["complex_prmtop"]} {paths["complex_inpcrd"]}

# Receptor alone (protein, no ligand)
RECEPTOR = combine {{PROT}}
saveamberparm RECEPTOR {paths["receptor_prmtop"]} {paths["receptor_inpcrd"]}

# Ligand alone (no protein)
LIGAND = combine {{LIG}}
saveamberparm LIGAND {paths["ligand_prmtop"]} {paths["ligand_inpcrd"]}

# Solvated complex for MD (uses same bond as COMPLEX above)
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
        raise RuntimeError(f"tleap extended build failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    for k, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"tleap did not produce expected file: {k} -> {p}")

    return paths


def convert_dcd_to_netcdf_dry(dcd_path: Path, solv_prmtop: Path,
                              dry_prmtop: Path, work: Path) -> Path:
    """Convert solvated DCD trajectory -> water-stripped AMBER NetCDF.

    Two-stage cpptraj:
      1. Read DCD with solvated prmtop -> write all-atom .nc
      2. Read .nc + solvated prmtop, strip ":WAT,Na+,Cl-" -> dry .nc

    Returns the dry .nc path (matches dry_prmtop).
    """
    nc_full = work / "trajectory_full.nc"
    nc_dry = work / "trajectory_dry.nc"

    # Stage 1: DCD -> NetCDF
    cpptraj_in1 = work / "cpptraj_to_nc.in"
    cpptraj_in1.write_text(f"""parm {solv_prmtop}
trajin {dcd_path}
trajout {nc_full} netcdf
run
quit
""")
    r = subprocess.run(["cpptraj", "-i", str(cpptraj_in1)], capture_output=True,
                       text=True, cwd=work, timeout=600)
    if r.returncode != 0 or not nc_full.exists():
        raise RuntimeError(f"cpptraj DCD->nc failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    # Stage 2: strip waters/ions -> dry NetCDF
    # The dry prmtop has only protein + ligand atoms (no WAT, no ions).
    # So we strip everything except :1-N where N = number of dry residues.
    # Easier: use `strip :WAT,Na+,Cl-` which leaves protein+ligand.
    cpptraj_in2 = work / "cpptraj_strip_waters.in"
    cpptraj_in2.write_text(f"""parm {solv_prmtop}
trajin {nc_full}
strip :WAT,Na+,Cl-
trajout {nc_dry} netcdf
run
quit
""")
    r = subprocess.run(["cpptraj", "-i", str(cpptraj_in2)], capture_output=True,
                       text=True, cwd=work, timeout=600)
    if r.returncode != 0 or not nc_dry.exists():
        raise RuntimeError(f"cpptraj strip waters failed:\n{r.stdout[-800:]}\n{r.stderr[-300:]}")

    # Sanity check: number of atoms in nc_dry should match dry_prmtop
    return nc_dry


def write_mmpbsa_config(out_path: Path, startframe: int = 1, endframe: int = 200,
                        interval: int = 1, decomp: bool = True,
                        decomp_within_A: float = 6.0) -> Path:
    """Write the gmx_MMPBSA .in configuration file.

    Settings:
      - igb=8 (GBn2, OBC2-like; recommended for protein-ligand by Onufriev)
      - PBRadii=4 (mbondi3, matches igb=8)
      - saltcon=0.15 M (physiological)
      - idecomp=2 (per-residue, bb+sc separately), print residues within 6 A
        of ligand
    """
    decomp_block = (
        f"&decomp\n"
        f"  idecomp=2, dec_verbose=0, print_res=\"within {decomp_within_A}\",\n"
        f"/\n"
    ) if decomp else ""
    cfg = f"""Input file for gmx_MMPBSA on covalent inhibitor complex (auto-generated)
&general
  startframe={startframe}, endframe={endframe}, interval={interval},
  forcefields="leaprc.protein.ff14SB,leaprc.gaff2",
  PBRadii=4,
  verbose=1,
/
&gb
  igb=8, saltcon=0.150,
/
{decomp_block}"""
    out_path.write_text(cfg)
    return out_path


def run_gmx_mmpbsa(complex_prmtop: Path, receptor_prmtop: Path, ligand_prmtop: Path,
                   nc_dry: Path, mmpbsa_in: Path, work: Path,
                   timeout_s: int = 3600) -> tuple[Path, Path, Path]:
    """Invoke gmx_MMPBSA. Returns (final_results_dat, per_frame_csv, decomp_csv).

    Uses `-nogui` to suppress visualization, and `-deo` to dump per-residue
    decomposition to CSV (only meaningful if &decomp section is present).
    """
    final_dat = work / "FINAL_RESULTS_MMPBSA.dat"
    per_frame_csv = work / "per_frame.csv"
    decomp_csv = work / "FINAL_DECOMP_MMPBSA.csv"

    cmd = [
        "gmx_MMPBSA",
        "-O",
        "-i", str(mmpbsa_in),
        "-cp", str(complex_prmtop),
        "-rp", str(receptor_prmtop),
        "-lp", str(ligand_prmtop),
        "-ct", str(nc_dry),
        "-nogui",
        "-o", str(final_dat),
        "-eo", str(per_frame_csv),
        "-do", str(decomp_csv),
    ]
    print(f"[run_gmx_mmpbsa] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work,
                       timeout=timeout_s)
    print(f"[run_gmx_mmpbsa] stdout tail:\n{r.stdout[-1500:]}")
    if r.returncode != 0:
        print(f"[run_gmx_mmpbsa] stderr tail:\n{r.stderr[-1500:]}")
        raise RuntimeError(f"gmx_MMPBSA failed with rc={r.returncode}; see logs above")
    if not final_dat.exists():
        raise RuntimeError(f"gmx_MMPBSA did not produce {final_dat}")
    return final_dat, per_frame_csv, decomp_csv


def parse_final_results(final_dat: Path) -> dict:
    """Extract mean dG_bind (DELTA TOTAL) from FINAL_RESULTS_MMPBSA.dat.

    The file has a section like:
        Delta (Complex - Receptor - Ligand):
        Energy Component            Average              Std. Dev.  Std. Err. of Mean
        -----------------------------------------------------------------------------
        BOND                        0.0000               0.0000              0.0000
        ...
        DELTA TOTAL                -8.5234               1.2345              0.0567

    We pull DELTA TOTAL average + std.
    """
    text = final_dat.read_text()
    # Find the "Delta" section
    out = {"dG_bind_kcalmol": None, "dG_bind_std": None,
           "ggas_kcalmol": None, "gsolv_kcalmol": None}

    delta_match = re.search(
        r"Delta\s*\([^\)]+\)\s*[:\n].*?(?=\n\s*\n|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    section = delta_match.group(0) if delta_match else text

    # DELTA TOTAL row: "DELTA TOTAL  <avg>  <std>  <stderr>"
    m = re.search(r"DELTA\s+TOTAL\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                  section, re.IGNORECASE)
    if m:
        out["dG_bind_kcalmol"] = float(m.group(1))
        out["dG_bind_std"] = float(m.group(2))

    # Also pull DELTA G gas and DELTA G solv if present
    for key, label in [("ggas_kcalmol", "DELTA G gas"),
                       ("gsolv_kcalmol", "DELTA G solv")]:
        m = re.search(rf"{label}\s+(-?\d+\.\d+)", section, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))

    return out


def parse_per_frame(per_frame_csv: Path) -> dict:
    """Parse per-frame CSV emitted by `gmx_MMPBSA -eo`.

    Format (gmx_MMPBSA 1.6.5): header row then per-frame rows with columns
    including "TOTAL" (the delta binding energy per frame). We compute mean/std
    from the TOTAL column to corroborate FINAL_RESULTS values.
    """
    if not per_frame_csv.exists():
        return {"n_frames_scored": 0, "dG_per_frame_mean": None,
                "dG_per_frame_std": None}

    totals = []
    with per_frame_csv.open() as f:
        # gmx_MMPBSA per-frame CSV has multiple multi-line headers; we look for
        # a row whose Frame column is an int and grab the TOTAL column.
        reader = csv.reader(f)
        header = None
        total_idx = None
        for row in reader:
            if not row:
                continue
            # Header detection: row contains "Frame" (case-insensitive)
            if any(c.strip().lower() == "frame" for c in row):
                header = [c.strip() for c in row]
                # Prefer "TOTAL" column (delta in GB section is the rightmost
                # block; we just take the LAST column named "TOTAL").
                total_candidates = [i for i, h in enumerate(header)
                                    if h.upper() == "TOTAL"]
                if total_candidates:
                    total_idx = total_candidates[-1]
                continue
            if header is None or total_idx is None:
                continue
            try:
                int(row[0])
            except (ValueError, IndexError):
                continue
            try:
                totals.append(float(row[total_idx]))
            except (ValueError, IndexError):
                continue

    if not totals:
        return {"n_frames_scored": 0, "dG_per_frame_mean": None,
                "dG_per_frame_std": None}

    arr = np.array(totals)
    return {
        "n_frames_scored": int(len(arr)),
        "dG_per_frame_mean": float(arr.mean()),
        "dG_per_frame_std": float(arr.std()),
    }


def parse_decomp(decomp_csv: Path) -> dict[str, float]:
    """Parse the per-residue decomposition CSV produced by `gmx_MMPBSA -do`.

    gmx_MMPBSA 1.6.5 emits a CSV with multi-section format. We look for rows
    of the form (Residue, Internal, vdW, Eel, GBSOL, Total) and keep the
    DELTA section (last block).

    Returns dict mapping "ResName###" (e.g. "CYS346") -> total delta in kcal/mol.
    """
    out = {}
    if not decomp_csv.exists():
        return out

    text = decomp_csv.read_text()
    # Find the "DELTA" section - everything after a line containing "DELTAS:"
    # or after a "Total Energy Decomposition" header for the delta block.
    # Heuristic: locate last occurrence of "Total Energy Decomposition" and
    # parse from there.
    sections = re.split(r"(?im)^.*total energy decomposition.*$", text)
    target_section = sections[-1] if len(sections) > 1 else text

    # Per-residue rows look like:
    #   CYS 346,...,<numbers>,...,<total>
    # The exact columns vary; we extract the residue name+number from col 0
    # and the LAST numeric column as the total.
    for line in target_section.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        # First column should look like "RES 123" or "RES123" or "RES_123"
        m = re.match(r"^\s*([A-Z]{2,4})[\s_]*(\d+)\s*$", parts[0])
        if not m:
            continue
        res_name = m.group(1)
        res_num = int(m.group(2))
        # Find last numeric value in the row (the Total column)
        total_val = None
        for cell in reversed(parts):
            try:
                total_val = float(cell)
                break
            except ValueError:
                continue
        if total_val is None:
            continue
        key = f"{res_name.capitalize()}{res_num}"  # e.g. "Cys346"
        out[key] = total_val

    return out


def score_one_md_v2(pdb_path: Path, smiles: str | None, target: str, name: str,
                    md_ns: float = 1.0, n_frames: int = 200,
                    equil_ps: float = 50.0, platform_name: str = "CUDA",
                    workdir: Path | None = None) -> dict:
    """gmx_MMPBSA-backed MD MM-GBSA scorer. Same return shape as
    `mdmmgbsa_cheng2017.score_one_md`, plus `residue_contributions` (JSON str)
    and `dG_per_frame_*` fields.
    """
    out = {
        "target": target, "name": name,
        "dG_bind_md_kcalmol": None, "dG_bind_md_std": None,
        "ligand_strain_md_kcalmol": None,  # not computed here; kept for shape
        "n_frames_scored": 0,
        "ggas_kcalmol": None, "gsolv_kcalmol": None,
        "dG_per_frame_mean": None, "dG_per_frame_std": None,
        "residue_contributions": None,  # JSON string
        "md_ns": md_ns,
        "wall_md_s": None, "wall_mmpbsa_s": None, "wall_total_s": None,
        "success_flag": 0, "error": None,
    }
    t_total0 = time.time()
    try:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="gmxmmpbsa_"))
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"[score_one_md_v2] workdir = {workdir}")

        # --- Stage 1: split + parameterize ligand ---
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

        # --- Stage 2: extended topology build (4 prmtops) ---
        paths = build_topologies_for_gmxmmpbsa(
            prot_pdb, lig_mol2, lig_frcmod, lig_resname, buffer_A=10.0,
        )

        # --- Stage 3: locate Cys-SG / warhead-Cb (for diagnostics + run_md) ---
        sg_idx_dry, cb_idx_dry, sg_cb_d0 = find_cys_sg_and_warhead_cb(
            paths["complex_dry_prmtop"], paths["complex_dry_inpcrd"]
        )
        sg_idx_solv, cb_idx_solv = sg_idx_dry, cb_idx_dry  # tleap puts solvent after

        # --- Stage 4: MD (reuse mdmmgbsa_cheng2017.run_md) ---
        dcd_path = workdir / "prod.dcd"
        t_md0 = time.time()
        run_md(paths["complex_solv_prmtop"], paths["complex_solv_inpcrd"],
               sg_idx_solv, cb_idx_solv,
               out_dcd=dcd_path, ns=md_ns, save_every_ps=5.0,
               equil_ps=equil_ps, dt_fs=4.0, platform_name=platform_name)
        out["wall_md_s"] = time.time() - t_md0

        # --- Stage 5: DCD -> dry NetCDF via cpptraj ---
        nc_dry = convert_dcd_to_netcdf_dry(
            dcd_path, paths["complex_solv_prmtop"], paths["complex_dry_prmtop"],
            workdir,
        )

        # --- Stage 6: write mmpbsa.in + run gmx_MMPBSA ---
        # Cap endframe at the actual number of frames in the NC. Quick probe
        # via cpptraj might be overkill; estimate from md_ns / save_every_ps.
        n_frames_actual = max(1, int(md_ns * 1000.0 / 5.0))  # 5 ps stride
        endframe = min(n_frames, n_frames_actual)
        mmpbsa_in = write_mmpbsa_config(workdir / "mmpbsa.in",
                                        startframe=1, endframe=endframe,
                                        interval=1, decomp=True,
                                        decomp_within_A=6.0)

        t_g0 = time.time()
        final_dat, per_frame_csv, decomp_csv = run_gmx_mmpbsa(
            paths["complex_prmtop"], paths["receptor_prmtop"],
            paths["ligand_prmtop"], nc_dry, mmpbsa_in, workdir,
            timeout_s=3600,
        )
        out["wall_mmpbsa_s"] = time.time() - t_g0

        # --- Stage 7: parse ---
        fin = parse_final_results(final_dat)
        out["dG_bind_md_kcalmol"] = fin["dG_bind_kcalmol"]
        out["dG_bind_md_std"] = fin["dG_bind_std"]
        out["ggas_kcalmol"] = fin["ggas_kcalmol"]
        out["gsolv_kcalmol"] = fin["gsolv_kcalmol"]

        pf = parse_per_frame(per_frame_csv)
        out["n_frames_scored"] = pf["n_frames_scored"]
        out["dG_per_frame_mean"] = pf["dG_per_frame_mean"]
        out["dG_per_frame_std"] = pf["dG_per_frame_std"]

        residues = parse_decomp(decomp_csv)
        if residues:
            out["residue_contributions"] = json.dumps(residues, sort_keys=True)

        if out["dG_bind_md_kcalmol"] is not None:
            out["success_flag"] = 1
        else:
            out["error"] = "could not parse DELTA TOTAL from FINAL_RESULTS_MMPBSA.dat"

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-1200:]
    out["wall_total_s"] = time.time() - t_total0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run 5P9J BTK+ibrutinib smoke test")
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
        pdb = Path.home() / "data" / "5P9J.pdb"
        if not pdb.exists():
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

    print(f"=== MD MM-GBSA via gmx_MMPBSA ===")
    smiles_preview = (smiles[:60] + "...") if smiles else "None"
    print(f"pdb={pdb} smiles={smiles_preview} md_ns={args.md_ns} "
          f"equil_ps={args.equil_ps} platform={args.platform}")
    t0 = time.time()
    workdir = Path(args.workdir) if args.workdir else None
    res = score_one_md_v2(pdb, smiles, target, name, md_ns=args.md_ns,
                          n_frames=args.n_frames, equil_ps=args.equil_ps,
                          platform_name=args.platform, workdir=workdir)
    dt = time.time() - t0
    print(f"\n--- Result ({dt:.1f}s) ---")
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
    if not res["success_flag"]:
        print(f"\n  traceback:\n{res.get('traceback','')}")


if __name__ == "__main__":
    main()
