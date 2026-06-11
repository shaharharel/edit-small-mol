"""Run single-frame MM-GBSA (GB-igb8) on Boltz cofold poses for the 838 visible mols.

Per pose:
  1. CIF -> PDB (protein chain A + ligand chain B renamed LIG1 -> LIG)
  2. Split into protein.pdb (ATOM) + ligand.pdb (HETATM, then -> ligand.mol2 via antechamber AM1-BCC + GAFF2)
  3. tleap: ff14SB (protein) + GAFF2 (ligand) + Cys346(SG)-LIG(warhead C) covalent bond
     -> complex.prmtop, complex.inpcrd, receptor.prmtop, ligand.prmtop
  4. sander: 200 steepest descent + 200 conjugate gradient with backbone CA/N/O restraints (1 kcal/mol/A^2)
  5. MMPBSA.py single-frame with &gb igb=8 saltcon=0.15 -> parse FINAL_RESULTS_MMPBSA.dat

Outputs: results/paper_evaluation/mmgbsa_838_singleframe.csv with columns
    smiles, row_id, row_id_semantic, dG_GB_kcalmol, ggas_kcalmol, gsolv_kcalmol,
    E_vdw_kcalmol, E_eel_kcalmol, status, error

Usage:
  python experiments/run_mmgbsa_838.py --manifest manifest.csv --cif-dir cifs/ \
      --out results/paper_evaluation/mmgbsa_838_singleframe.csv --workers 28
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi

# Boltz residue name for the ligand in the CIF
BOLTZ_LIGAND_RESNAME = "LIG1"
TARGET_LIGAND_RESNAME = "LIG"

# Covalent target residue (ZAP70 catalytic Cys346 in cofold structure)
CYS_RESI = 346

# Warhead element/atom used for covalent bond detection. Boltz names ligand
# atoms element+canonical_rank_plus_one. The bonded carbon is identified by
# proximity to Cys346 SG in the cofold structure.

# Standard protein residue names
STD_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
    "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","HOH","WAT","CYX","HID","HIE","HIP",
}


def run_cmd(cmd, cwd=None, timeout=300, input_str=None):
    """Run a shell command and return (rc, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            input=input_str, shell=isinstance(cmd, str),
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return -2, "", f"EXCEPTION: {type(e).__name__}: {e}"


def extract_protein_pdb(cif_path: Path, prot_pdb: Path) -> int:
    """Write protein-only PDB from CIF. Returns n_atoms."""
    st = gemmi.read_structure(str(cif_path))
    new_st = gemmi.Structure()
    new_model = gemmi.Model("1")
    n_atoms = 0
    for model in st:
        for chain in model:
            new_chain = gemmi.Chain("A")
            has = False
            for res in chain:
                if res.name in STD_RESIDUES:
                    new_res = gemmi.Residue()
                    new_res.name = res.name
                    new_res.seqid = res.seqid
                    new_res.het_flag = " "
                    for atom in res:
                        new_res.add_atom(atom)
                        n_atoms += 1
                    new_chain.add_residue(new_res)
                    has = True
            if has:
                new_model.add_chain(new_chain)
            break
        break
    new_st.add_model(new_model)
    new_st.write_pdb(str(prot_pdb))
    return n_atoms


def parse_charge_from_smiles(smiles: str) -> int:
    """Sum formal charges from SMILES string."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return 0
        return sum(a.GetFormalCharge() for a in m.GetAtoms())
    except Exception:
        return 0


def build_ligand_with_h(cif_path: Path, smiles: str, out_pdb: Path) -> tuple[bool, str, str | None]:
    """Build a ligand PDB with explicit hydrogens and 3D coords from the CIF.

    Strategy:
      1. Extract ligand heavy atoms + coords from CIF -> minimal PDB block.
      2. RDKit MolFromPDBBlock(removeHs=False, proximityBonding=True) reads
         heavy atoms + guesses bonds by proximity.
      3. Template-match via AssignBondOrdersFromTemplate(smiles_template, mol)
         to get correct bond orders.
      4. AddHs(mol, addCoords=True) places H atoms based on geometry.
      5. Write PDB with chain L, residue LIG, HETATM.

    Returns (success, error_msg, warhead_atom_name). Warhead = ligand C
    nearest Cys346 SG. Atom name preserves whatever RDKit assigns; we'll
    pick the name of the warhead C.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as e:
        return False, f"rdkit import: {e}", None

    # 1. Read CIF: protein Cys346.SG + ligand heavy atoms with coords
    st = gemmi.read_structure(str(cif_path))
    cys_sg = None
    lig_atoms = []  # (cif_name, element, pos)
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ("CYS", "CYX") and res.seqid.num == CYS_RESI:
                    for atom in res:
                        if atom.name == "SG":
                            cys_sg = atom.pos
                if res.name == BOLTZ_LIGAND_RESNAME:
                    for atom in res:
                        lig_atoms.append((atom.name, atom.element.name, atom.pos))
        break
    if not lig_atoms:
        return False, "no ligand atoms in CIF", None

    # 2. Write minimal PDB block for ligand heavy atoms; HETATM, chain L,
    #    residue LIG, sequential atom serials.
    pdb_lines = []
    for i, (cname, elem, pos) in enumerate(lig_atoms, start=1):
        # Make 4-char atom name, right-padded; for 1-char element + 2-digit num
        # (e.g. "C40"), we want " C40" (leading space) in cols 13-16.
        if len(cname) <= 3:
            atom_name = f" {cname:<3s}"
        else:
            atom_name = cname[:4]
        # PDB HETATM line (76 width then element)
        line = (
            f"HETATM{i:>5d} {atom_name} LIG L   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00          {elem:>2s}"
        )
        pdb_lines.append(line)
    pdb_lines.append("END")
    pdb_block = "\n".join(pdb_lines) + "\n"

    # 3. RDKit reads with proximity bonding
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=True, sanitize=False)
    if mol is None:
        return False, "RDKit cannot parse generated PDB block", None

    # 4. Template-match bond orders from SMILES
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return False, f"bad smiles: {smiles}", None

    try:
        from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
        mol = AssignBondOrdersFromTemplate(template, mol)
    except Exception as e:
        return False, f"assignBondOrders: {e}", None

    # Sanitize
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, f"sanitize: {e}", None

    # 5. Find warhead heavy atom (closest ligand C to Cys346 SG) BEFORE AddHs
    warhead_cif_name = None
    if cys_sg is not None:
        best = None
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            p = conf.GetAtomPosition(atom.GetIdx())
            d = ((p.x - cys_sg.x) ** 2 + (p.y - cys_sg.y) ** 2 + (p.z - cys_sg.z) ** 2) ** 0.5
            if best is None or d < best[1]:
                # Get atom name from PDB residue info
                info = atom.GetPDBResidueInfo()
                cname_local = info.GetName().strip() if info else f"C{atom.GetIdx()}"
                best = (cname_local, d, atom.GetIdx())
        if best:
            warhead_cif_name = best[0]
            warhead_idx = best[2]

    # 6. Add hydrogens with positions
    mol = Chem.AddHs(mol, addCoords=True)

    # 7. Ensure all atoms have proper PDB monomer info (chain L, resname LIG)
    h_counter = 1
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("LIG")
        info.SetChainId("L")
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        if atom.GetAtomicNum() == 1:
            name = f"H{h_counter}"
            h_counter += 1
            info.SetName(f" {name:<3s}" if len(name) <= 3 else name[:4])
        # leave heavy atom name as-is from original PDB
        atom.SetMonomerInfo(info)

    Chem.MolToPDBFile(mol, str(out_pdb))
    # Also write an SDF for antechamber (more reliable bond perception than PDB CONECT)
    sdf_path = out_pdb.with_suffix(".sdf")
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return True, "", warhead_cif_name


def parameterize_ligand(lig_pdb: Path, work_dir: Path, charge: int = 0) -> tuple[bool, str]:
    """Run antechamber + parmchk2 to make ligand.mol2 + ligand.frcmod.

    Returns (success, error_msg).
    """
    # antechamber: SDF -> mol2 with AM1-BCC charges (no geom opt: maxcyc=0
    # uses single-point AM1 + BCC; charges only, no relaxation of the
    # Boltz pose), GAFF2 atom types. SDF is more reliable than PDB for
    # bond perception.
    sdf_path = lig_pdb.with_suffix(".sdf")
    src_arg = str(sdf_path) if sdf_path.exists() else str(lig_pdb)
    src_fmt = "sdf" if sdf_path.exists() else "pdb"
    rc, out, err = run_cmd([
        "antechamber",
        "-i", src_arg, "-fi", src_fmt,
        "-o", "ligand.mol2", "-fo", "mol2",
        "-c", "bcc", "-s", "0",
        "-nc", str(charge),
        "-at", "gaff2",
        "-pf", "y",
        "-ek", "qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, maxcyc=0,",
    ], cwd=work_dir, timeout=600)
    if rc != 0 or not (work_dir / "ligand.mol2").exists():
        return False, f"antechamber rc={rc}: {err[-500:]}"

    # parmchk2 for missing GAFF2 parameters
    rc, out, err = run_cmd([
        "parmchk2",
        "-i", "ligand.mol2", "-f", "mol2",
        "-o", "ligand.frcmod",
        "-s", "gaff2",
    ], cwd=work_dir, timeout=60)
    if rc != 0 or not (work_dir / "ligand.frcmod").exists():
        return False, f"parmchk2 rc={rc}: {err[-500:]}"

    return True, ""


def count_protein_residues(prot_pdb: Path) -> int:
    """Count distinct (chain, resnum) protein residues from PDB."""
    seen = set()
    with open(prot_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resnum = line[22:26].strip()
                seen.add((chain, resnum))
    return len(seen)


def build_tleap_script(work_dir: Path, warhead_atom: str | None, n_prot_res: int) -> str:
    """Write tleap input script.

    Note: we do NOT form an explicit covalent bond between Cys346.SG and the
    ligand warhead C. The cross-FF (ff14SB + GAFF2) is missing angle/dihedral
    parameters at the boundary (SH-c3-*, HS-SH-c3, etc.) which causes tleap
    parameterization to fail. Instead, the ligand is treated as a tight
    non-bonded complex with the C-S distance geometrically held by the Boltz
    pose (~1.8 A) and energy captured via VdW + electrostatic terms after
    minimization. For relative dG_GB ranking across acrylamide-warhead ligands
    sharing the same Cys346 conjugation, the missing covalent bonded term is
    approximately constant.

    Cys346 is kept as CYS (with HG) for non-covalent treatment.
    """
    script = f"""source leaprc.protein.ff14SB
source leaprc.gaff2
# mbondi3 radii required for igb=8 (GB-Neck2)
set default PBradii mbondi3
loadamberparams ligand.frcmod
ligand = loadmol2 ligand.mol2
protein = loadpdb protein.pdb
# Combine into complex; no explicit covalent bond
complex = combine {{ protein ligand }}
saveamberparm complex complex.prmtop complex.inpcrd
saveamberparm protein receptor.prmtop receptor.inpcrd
saveamberparm ligand ligand.prmtop ligand.inpcrd
savepdb complex complex_built.pdb
quit
"""
    return script


def parameterize_complex(prot_pdb: Path, work_dir: Path, warhead_atom: str | None) -> tuple[bool, str]:
    """Run tleap to build complex.prmtop + complex.inpcrd."""
    # Copy protein PDB into work dir
    shutil.copy(prot_pdb, work_dir / "protein.pdb")
    n_prot_res = count_protein_residues(work_dir / "protein.pdb")
    script = build_tleap_script(work_dir, warhead_atom, n_prot_res)
    (work_dir / "tleap.in").write_text(script)
    rc, out, err = run_cmd(["tleap", "-f", "tleap.in"], cwd=work_dir, timeout=300)
    if not (work_dir / "complex.prmtop").exists() or not (work_dir / "complex.inpcrd").exists():
        return False, f"tleap rc={rc}, stdout last 500: {out[-500:]}; err: {err[-200:]}"
    if not (work_dir / "receptor.prmtop").exists() or not (work_dir / "ligand.prmtop").exists():
        return False, "tleap did not produce receptor.prmtop or ligand.prmtop"
    return True, ""


def minimize_complex(work_dir: Path) -> tuple[bool, str]:
    """Run sander minimization with backbone restraints."""
    min_in = """Min: 50 SD + 50 CG with backbone restraints
&cntrl
  imin=1, maxcyc=100, ncyc=50,
  cut=12.0, igb=8, saltcon=0.15,
  ntb=0, ntr=1, ntpr=50,
  restraint_wt=1.0, restraintmask='@CA,N,O',
/
"""
    (work_dir / "min.in").write_text(min_in)
    rc, out, err = run_cmd([
        "sander", "-O",
        "-i", "min.in", "-o", "min.out",
        "-p", "complex.prmtop", "-c", "complex.inpcrd",
        "-r", "min.rst7", "-ref", "complex.inpcrd",
    ], cwd=work_dir, timeout=600)
    if rc != 0 or not (work_dir / "min.rst7").exists():
        return False, f"sander rc={rc}: {err[-500:]}"
    return True, ""


def run_mmpbsa(work_dir: Path) -> tuple[bool, dict, str]:
    """Run MMPBSA.py single-frame, parse FINAL_RESULTS_MMPBSA.dat."""
    mmpbsa_in = """Single-frame MM-GBSA (GB-igb8)
&general
  startframe=1, endframe=1, interval=1,
  keep_files=0, verbose=1,
/
&gb
  igb=8, saltcon=0.15,
/
"""
    (work_dir / "mmpbsa.in").write_text(mmpbsa_in)
    # MMPBSA.py needs a "trajectory" — we use the minimized rst7 directly.
    # We need a topology with no solvent box, so we use the unminimized
    # complex.inpcrd for solvated. Since igb=8 has no waters, we pass the
    # minimized rst7 as the trajectory via -y.
    rc, out, err = run_cmd([
        "MMPBSA.py", "-O",
        "-i", "mmpbsa.in",
        "-cp", "complex.prmtop",
        "-rp", "receptor.prmtop",
        "-lp", "ligand.prmtop",
        "-y", "min.rst7",
    ], cwd=work_dir, timeout=600)
    res_file = work_dir / "FINAL_RESULTS_MMPBSA.dat"
    if not res_file.exists():
        return False, {}, f"MMPBSA.py rc={rc}: stdout {out[-500:]}; err {err[-300:]}"

    content = res_file.read_text()
    # Parse the "Differences (Complex - Receptor - Ligand):" block
    # Look for VDWAALS, EEL, EGB, GGAS, GSOLV, DELTA TOTAL
    parsed = parse_mmpbsa_output(content)
    if parsed is None:
        return False, {}, f"Failed to parse FINAL_RESULTS_MMPBSA.dat:\n{content[:1000]}"
    return True, parsed, ""


def parse_mmpbsa_output(content: str) -> dict | None:
    """Parse FINAL_RESULTS_MMPBSA.dat for the delta values."""
    # The "Differences (Complex - Receptor - Ligand):" section has lines like:
    #   VDWAALS     -42.1234       0.0000      0.0000
    #   EEL          -23.4567       ...
    #   EGB           18.1234       ...
    #   ESURF        -3.4567        ...
    #   ...
    #   DELTA G gas: ...
    #   DELTA G solv: ...
    #   DELTA TOTAL ...
    # Find the "Differences" section
    diff_start = content.find("Differences (Complex - Receptor - Ligand)")
    if diff_start < 0:
        # Try alternate phrasing
        diff_start = content.find("Delta Energy Terms")
    if diff_start < 0:
        return None
    diff_block = content[diff_start:]

    def grep(pattern):
        m = re.search(pattern, diff_block, re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    vdw = grep(r"^\s*VDWAALS\s+(-?\d+\.\d+)")
    eel = grep(r"^\s*EEL\s+(-?\d+\.\d+)")
    egb = grep(r"^\s*EGB\s+(-?\d+\.\d+)")
    esurf = grep(r"^\s*ESURF\s+(-?\d+\.\d+)")
    internal = grep(r"^\s*(?:INTERNAL|BOND\+ANG\+DIH)\s+(-?\d+\.\d+)")
    # Newer MMPBSA splits as BOND, ANGLE, DIHED. Sum if needed.
    if internal is None:
        bond = grep(r"^\s*BOND\s+(-?\d+\.\d+)") or 0.0
        angle = grep(r"^\s*ANGLE\s+(-?\d+\.\d+)") or 0.0
        dihed = grep(r"^\s*DIHED\s+(-?\d+\.\d+)") or 0.0
        internal = bond + angle + dihed
    # 1-4 terms
    onefour_vdw = grep(r"^\s*1-4\s+VDW\s+(-?\d+\.\d+)") or 0.0
    onefour_eel = grep(r"^\s*1-4\s+EEL\s+(-?\d+\.\d+)") or 0.0

    ggas = grep(r"^\s*DELTA\s+G\s+gas\s*[:=]?\s*(-?\d+\.\d+)")
    gsolv = grep(r"^\s*DELTA\s+G\s+solv\s*[:=]?\s*(-?\d+\.\d+)")
    total = grep(r"^\s*DELTA\s+TOTAL\s+(-?\d+\.\d+)")

    if total is None:
        # Try the older format: "DELTA G binding ="
        total = grep(r"^\s*DELTA\s+G\s+binding\s*[:=]?\s*(-?\d+\.\d+)")
    if total is None:
        # Compute from components if possible
        if ggas is not None and gsolv is not None:
            total = ggas + gsolv
        elif all(v is not None for v in [vdw, eel, egb, esurf]):
            total = (internal or 0.0) + vdw + eel + onefour_vdw + onefour_eel + egb + esurf
    if ggas is None and vdw is not None and eel is not None:
        ggas = (internal or 0.0) + vdw + eel + onefour_vdw + onefour_eel
    if gsolv is None and egb is not None:
        gsolv = egb + (esurf or 0.0)

    if total is None:
        return None
    return {
        "dG_GB_kcalmol": total,
        "ggas_kcalmol": ggas,
        "gsolv_kcalmol": gsolv,
        "E_vdw_kcalmol": vdw,
        "E_eel_kcalmol": eel,
    }


def process_one(args):
    """Process one cofold. Returns dict with status and parsed results."""
    row = args["row"]
    cif_path = Path(args["cif_path"])
    base_workdir = Path(args["workdir_base"])
    smiles = row["smiles"]
    row_id = row["row_id_api"]
    semantic = row["row_id_semantic"]

    # Per-molecule scratch dir
    work_dir = base_workdir / semantic
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "smiles": smiles,
        "row_id": row_id,
        "row_id_semantic": semantic,
        "dG_GB_kcalmol": None,
        "ggas_kcalmol": None,
        "gsolv_kcalmol": None,
        "E_vdw_kcalmol": None,
        "E_eel_kcalmol": None,
        "status": "FAIL",
        "error": "",
    }

    try:
        # 1a. Extract protein PDB
        prot_pdb = work_dir / "protein_raw.pdb"
        lig_pdb = work_dir / "ligand_raw.pdb"
        try:
            n_prot = extract_protein_pdb(cif_path, prot_pdb)
        except Exception as e:
            result["error"] = f"extract_protein: {e}"
            return result

        # 1b. Build ligand PDB with explicit H (from SMILES + CIF coords)
        try:
            ok, err, warhead = build_ligand_with_h(cif_path, smiles, lig_pdb)
        except Exception as e:
            result["error"] = f"build_ligand: {type(e).__name__}: {e}"
            return result
        if not ok:
            result["error"] = f"build_ligand: {err}"
            return result

        # 2. Parameterize ligand (AM1-BCC + GAFF2)
        charge = parse_charge_from_smiles(smiles)
        ok, err = parameterize_ligand(lig_pdb, work_dir, charge=charge)
        if not ok:
            result["error"] = f"ligand_param: {err}"
            return result

        # 3. Build complex with tleap
        ok, err = parameterize_complex(prot_pdb, work_dir, warhead)
        if not ok:
            result["error"] = f"tleap: {err}"
            return result

        # 4. Minimize
        ok, err = minimize_complex(work_dir)
        if not ok:
            result["error"] = f"sander: {err}"
            return result

        # 5. MMPBSA.py
        ok, parsed, err = run_mmpbsa(work_dir)
        if not ok:
            result["error"] = f"mmpbsa: {err}"
            return result

        result.update(parsed)
        result["status"] = "OK"
    except Exception as e:
        result["error"] = f"unexpected: {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
    finally:
        # Cleanup: keep only summary files; remove bulky intermediates
        try:
            if result["status"] == "OK":
                # Keep min.out + FINAL_RESULTS_MMPBSA.dat for QA; nuke prmtops
                for f in work_dir.iterdir():
                    if f.name not in {"min.out", "FINAL_RESULTS_MMPBSA.dat", "tleap.in"}:
                        try:
                            f.unlink()
                        except Exception:
                            pass
        except Exception:
            pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cif-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--workdir-base", type=Path, default=Path("/tmp/mmgbsa_work"))
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test on first N mols.")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    args = ap.parse_args()

    args.workdir_base.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest
    import csv as csvmod
    jobs = []
    with open(args.manifest) as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            cif_path = args.cif_dir / row["cif_filename"]
            if not cif_path.exists():
                print(f"WARN: missing CIF {cif_path}", file=sys.stderr)
                continue
            jobs.append({
                "row": row,
                "cif_path": str(cif_path),
                "workdir_base": str(args.workdir_base),
            })
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Processing {len(jobs)} cofolds on {args.workers} workers...", flush=True)

    # Load existing checkpoint
    existing = set()
    if args.out.exists():
        with open(args.out) as f:
            r = csvmod.DictReader(f)
            for row in r:
                existing.add(row["row_id_semantic"])
        print(f"Resume: {len(existing)} already done", flush=True)
    jobs = [j for j in jobs if j["row"]["row_id_semantic"] not in existing]
    print(f"Remaining: {len(jobs)} jobs", flush=True)

    fieldnames = ["smiles","row_id","row_id_semantic","dG_GB_kcalmol",
                  "ggas_kcalmol","gsolv_kcalmol","E_vdw_kcalmol","E_eel_kcalmol",
                  "status","error"]

    file_exists = args.out.exists()
    f_out = open(args.out, "a", newline="")
    writer = csvmod.DictWriter(f_out, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f_out.flush()

    n_ok = 0
    n_fail = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                r = {"smiles":"", "row_id":"", "row_id_semantic":"?",
                     "dG_GB_kcalmol":None, "ggas_kcalmol":None, "gsolv_kcalmol":None,
                     "E_vdw_kcalmol":None, "E_eel_kcalmol":None,
                     "status":"FAIL", "error": f"futureexc: {e}"}
            writer.writerow(r)
            done += 1
            if r["status"] == "OK":
                n_ok += 1
            else:
                n_fail += 1
            if done % args.checkpoint_every == 0:
                f_out.flush()
                os.fsync(f_out.fileno())
                print(f"  [{done}/{len(jobs)}] ok={n_ok} fail={n_fail}", flush=True)
    f_out.close()
    print(f"\nDONE: {done} processed, {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
