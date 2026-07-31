"""AutoDock Vina rescoring (--score_only) for Boltz cofold poses on ZAP70 Cys346.

Per Trott & Olson 2010, Vina's knowledge-based scoring function is orthogonal
to MM-GBSA's force-field physics — it catches MM-GBSA pathologies such as
entropy neglect and clash mis-scoring.

For each cofold dir at
  data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/<name>/
this script:
  1. Splits the full <name>_model_0.pdb into protein PDB and ligand SDF if
     those files are not already present.
  2. Converts protein PDB -> PDBQT via OpenBabel (`obabel -xr` keeps it rigid).
  3. Converts ligand SDF -> PDBQT via Meeko (`mk_prepare_ligand.py`).
  4. Builds a search box centered on Cys346 SG of THAT cofold's protein,
     22 A cubic, expanded to cover all ligand atoms if needed.
  5. Runs `vina --score_only` to score the existing pose (no docking).
  6. Parses Affinity / inter / intra components from stdout.

CLI:
  python experiments/vina_rescore_cofolds.py --cofold-dir <path> [--name <stem>]

When invoked as a library, call `rescore_cofold(cofold_dir)` and consume the
returned dict. Use `vina_rescore_batch.py` for the parallel sweep over all 997
cofolds.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _detect_name(cofold_dir: Path) -> str:
    """Cofold dir name = stem of every produced file."""
    return cofold_dir.name


def _atom_xyz(line: str) -> tuple[float, float, float]:
    """Parse columns 31-54 of a PDB ATOM/HETATM record."""
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def split_pdb_to_protein_and_ligand(
    pdb_path: Path, prot_pdb: Path, lig_sdf: Path
) -> dict:
    """Split the Boltz full-complex PDB into protein PDB + ligand SDF.

    Protein = all ATOM records (chain A).
    Ligand  = all HETATM records (resname LIG); written as a fresh SDF via RDKit
              with bond perception from the 3D coordinates.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    prot_lines = []
    lig_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                prot_lines.append(line)
            elif line.startswith("HETATM") and line[17:20].strip() == "LIG":
                lig_lines.append(line)
            elif line.startswith(("TER", "END")):
                pass  # don't carry across protein/lig boundaries

    if not prot_lines:
        raise RuntimeError(f"no ATOM (protein) records in {pdb_path}")
    if not lig_lines:
        raise RuntimeError(f"no HETATM LIG records in {pdb_path}")

    # Write protein PDB (chain A polymer only).
    prot_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(prot_pdb, "w") as f:
        f.writelines(prot_lines)
        f.write("END\n")

    # Build ligand-only PDB block, hand it to RDKit, perceive bonds, write SDF.
    lig_pdb_block = "".join(lig_lines) + "END\n"
    mol = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=False, sanitize=False)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse ligand HETATM block in {pdb_path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Tolerate non-kekulizable nitrogens — Vina/Meeko only need bond topology.
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
    # Ensure Hs are present so Meeko can assign torsions correctly.
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass

    writer = Chem.SDWriter(str(lig_sdf))
    writer.write(mol)
    writer.close()
    return {"n_prot_atoms": len(prot_lines), "n_lig_atoms": len(lig_lines)}


def find_cys346_sg(prot_pdb: Path) -> Optional[tuple[float, float, float]]:
    """Return (x, y, z) of SG atom of residue CYS 346 chain A, or None."""
    with open(prot_pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "SG":
                continue
            if line[17:20].strip() != "CYS":
                continue
            if line[22:26].strip() != "346":
                continue
            return _atom_xyz(line)
    return None


def ligand_xyz_bounds(lig_sdf_or_pdb: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mins, maxs) array of ligand coordinates."""
    coords = []
    if lig_sdf_or_pdb.suffix == ".sdf":
        # Cheap SDF parse — atoms start on line 4 of V2000.
        with open(lig_sdf_or_pdb) as f:
            lines = f.readlines()
        try:
            counts_line = lines[3]
            n_atoms = int(counts_line[:3])
        except (IndexError, ValueError):
            n_atoms = 0
        for i in range(4, 4 + n_atoms):
            parts = lines[i].split()
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    else:
        with open(lig_sdf_or_pdb) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    coords.append(list(_atom_xyz(line)))
    if not coords:
        raise RuntimeError(f"no ligand coordinates parsed from {lig_sdf_or_pdb}")
    arr = np.array(coords)
    return arr.min(axis=0), arr.max(axis=0)


# --------------------------------------------------------------------------
# PDBQT preparation
# --------------------------------------------------------------------------

def protein_to_pdbqt(prot_pdb: Path, prot_pdbqt: Path) -> None:
    """OpenBabel: protein PDB -> rigid PDBQT (-xr flag)."""
    cmd = [
        "obabel",
        str(prot_pdb),
        "-O",
        str(prot_pdbqt),
        "-xr",  # rigid receptor; no torsions written
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0 or not prot_pdbqt.exists() or prot_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            f"obabel receptor failed (rc={res.returncode}): {res.stderr[:400]}"
        )


def ligand_to_pdbqt(lig_sdf: Path, lig_pdbqt: Path) -> None:
    """Meeko: ligand SDF -> PDBQT."""
    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        str(lig_sdf),
        "-o",
        str(lig_pdbqt),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not lig_pdbqt.exists() or lig_pdbqt.stat().st_size == 0:
        # Fallback: try obabel ligand prep (less robust torsion tree but works).
        cmd2 = ["obabel", str(lig_sdf), "-O", str(lig_pdbqt), "--gen3d", "-h"]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
        if res2.returncode != 0 or not lig_pdbqt.exists() or lig_pdbqt.stat().st_size == 0:
            # One more try without --gen3d (preserve pose) -- this is the priority.
            cmd3 = ["obabel", str(lig_sdf), "-O", str(lig_pdbqt), "-h"]
            res3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=120)
            if res3.returncode != 0 or not lig_pdbqt.exists() or lig_pdbqt.stat().st_size == 0:
                raise RuntimeError(
                    f"meeko + obabel ligand prep both failed.\n"
                    f"meeko stderr: {res.stderr[:200]}\n"
                    f"obabel stderr: {res3.stderr[:200]}"
                )


# --------------------------------------------------------------------------
# Vina scoring
# --------------------------------------------------------------------------

VINA_AFFINITY_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:?\s*([-+]?[0-9]*\.?[0-9]+)"
)
VINA_INTER_RE = re.compile(r"\(1\)\s*Final Intermolecular Energy\s*:?\s*([-+]?[0-9]*\.?[0-9]+)")
VINA_INTRA_RE = re.compile(r"\(2\)\s*Final Total Internal Energy\s*:?\s*([-+]?[0-9]*\.?[0-9]+)")


_NUM_AFTER_COLON = re.compile(r":\s*([-+]?[0-9]*\.?[0-9]+)")


def _num_after_colon(line: str) -> Optional[float]:
    m = _NUM_AFTER_COLON.search(line)
    return float(m.group(1)) if m else None


def parse_vina_score(stdout: str) -> dict:
    """Parse Affinity / inter / intra / torsion / unbound from `vina --score_only`.

    Vina 1.2 stdout (score_only) contains lines like:
      Estimated Free Energy of Binding   : -4.114 (kcal/mol) [=(1)+(2)+(3)-(4)]
      (1) Final Intermolecular Energy    : -6.038 (kcal/mol)
      (2) Final Total Internal Energy    : 0.132 (kcal/mol)
      (3) Torsional Free Energy          : 1.924 (kcal/mol)
      (4) Unbound System's Energy        : 0.132 (kcal/mol)

    We extract the number right after the first `:` so we don't pick up the
    section index "(1)" / "(2)" as the value.
    """
    affinity = None
    inter = None
    intra = None
    torsion = None
    unbound = None
    for raw in stdout.splitlines():
        s = raw.strip()
        if s.startswith("Affinity:"):
            affinity = _num_after_colon(s)
        elif s.startswith("Estimated Free Energy of Binding"):
            affinity = _num_after_colon(s)
        elif "Final Intermolecular Energy" in s:
            inter = _num_after_colon(s)
        elif "Final Total Internal Energy" in s:
            intra = _num_after_colon(s)
        elif "Torsional Free Energy" in s:
            torsion = _num_after_colon(s)
        elif "Unbound System" in s:
            unbound = _num_after_colon(s)
    return {
        "vina_kcalmol": affinity,
        "vina_inter_kcalmol": inter,
        "vina_intra_kcalmol": intra,
        "vina_torsion_kcalmol": torsion,
        "vina_unbound_kcalmol": unbound,
    }


def run_vina_score_only(
    prot_pdbqt: Path,
    lig_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
) -> dict:
    """Invoke the vina binary with --score_only and parse its output."""
    cmd = [
        "vina",
        "--receptor",
        str(prot_pdbqt),
        "--ligand",
        str(lig_pdbqt),
        "--score_only",
        "--center_x",
        f"{center[0]:.3f}",
        "--center_y",
        f"{center[1]:.3f}",
        "--center_z",
        f"{center[2]:.3f}",
        "--size_x",
        f"{box_size[0]:.3f}",
        "--size_y",
        f"{box_size[1]:.3f}",
        "--size_z",
        f"{box_size[2]:.3f}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(
            f"vina score_only failed (rc={res.returncode}): {res.stderr[:400]}"
        )
    parsed = parse_vina_score(res.stdout)
    if parsed["vina_kcalmol"] is None:
        raise RuntimeError(f"vina returned no Affinity in stdout: {res.stdout[-400:]}")
    parsed["raw_stdout"] = res.stdout
    return parsed


# --------------------------------------------------------------------------
# Top-level entry
# --------------------------------------------------------------------------

def rescore_cofold(
    cofold_dir: Path,
    box_pad: float = 8.0,
    box_min: float = 22.0,
) -> dict:
    """Run full rescore pipeline on one cofold directory.

    Box center  = Cys346 SG of that cofold's protein.
    Box size    = max(box_min, ligand-extent + 2 * box_pad) per axis.
                  Vina requires the box to cover the ligand or it errors out.
    """
    name = _detect_name(cofold_dir)
    pdb_path = cofold_dir / f"{name}_model_0.pdb"
    prot_pdb = cofold_dir / f"{name}_model_0.prot.pdb"
    lig_sdf = cofold_dir / f"{name}_model_0.lig.sdf"

    if not pdb_path.exists():
        return {
            "name": name,
            "success_flag": 0,
            "error": f"missing {pdb_path.name}",
            "vina_kcalmol": None,
            "vina_inter_kcalmol": None,
            "vina_intra_kcalmol": None,
        }

    # Split if necessary.
    if not prot_pdb.exists() or not lig_sdf.exists():
        try:
            split_pdb_to_protein_and_ligand(pdb_path, prot_pdb, lig_sdf)
        except Exception as e:
            return {
                "name": name,
                "success_flag": 0,
                "error": f"split_pdb: {type(e).__name__}: {e}",
                "vina_kcalmol": None,
                "vina_inter_kcalmol": None,
                "vina_intra_kcalmol": None,
            }

    # Find Cys346 SG for box center.
    sg = find_cys346_sg(prot_pdb)
    if sg is None:
        return {
            "name": name,
            "success_flag": 0,
            "error": "no Cys346 SG found in protein",
            "vina_kcalmol": None,
            "vina_inter_kcalmol": None,
            "vina_intra_kcalmol": None,
        }

    # PDBQT prep (use a per-cofold temp workspace so concurrent runs don't collide).
    with tempfile.TemporaryDirectory(prefix=f"vina_{name}_") as tmpd:
        tmp = Path(tmpd)
        prot_pdbqt = tmp / "rec.pdbqt"
        lig_pdbqt = tmp / "lig.pdbqt"

        try:
            protein_to_pdbqt(prot_pdb, prot_pdbqt)
        except Exception as e:
            return {
                "name": name,
                "success_flag": 0,
                "error": f"obabel_prot: {type(e).__name__}: {e}",
                "vina_kcalmol": None,
                "vina_inter_kcalmol": None,
                "vina_intra_kcalmol": None,
            }
        try:
            ligand_to_pdbqt(lig_sdf, lig_pdbqt)
        except Exception as e:
            return {
                "name": name,
                "success_flag": 0,
                "error": f"meeko_lig: {type(e).__name__}: {e}",
                "vina_kcalmol": None,
                "vina_inter_kcalmol": None,
                "vina_intra_kcalmol": None,
            }

        # Size box to cover both the SG pocket and the entire ligand.
        try:
            mins, maxs = ligand_xyz_bounds(lig_sdf)
            sg_arr = np.asarray(sg)
            box_lo = np.minimum(mins, sg_arr) - box_pad
            box_hi = np.maximum(maxs, sg_arr) + box_pad
            center = ((box_lo + box_hi) / 2.0).tolist()
            extents = (box_hi - box_lo)
            box_size = np.maximum(extents, box_min).tolist()
        except Exception as e:
            return {
                "name": name,
                "success_flag": 0,
                "error": f"box_size: {type(e).__name__}: {e}",
                "vina_kcalmol": None,
                "vina_inter_kcalmol": None,
                "vina_intra_kcalmol": None,
            }

        try:
            result = run_vina_score_only(
                prot_pdbqt, lig_pdbqt, tuple(center), tuple(box_size)
            )
        except Exception as e:
            return {
                "name": name,
                "success_flag": 0,
                "error": f"vina: {type(e).__name__}: {e}",
                "vina_kcalmol": None,
                "vina_inter_kcalmol": None,
                "vina_intra_kcalmol": None,
            }

    return {
        "name": name,
        "success_flag": 1,
        "error": None,
        "vina_kcalmol": result["vina_kcalmol"],
        "vina_inter_kcalmol": result["vina_inter_kcalmol"],
        "vina_intra_kcalmol": result["vina_intra_kcalmol"],
        "vina_torsion_kcalmol": result.get("vina_torsion_kcalmol"),
        "vina_unbound_kcalmol": result.get("vina_unbound_kcalmol"),
        "box_center_x": center[0],
        "box_center_y": center[1],
        "box_center_z": center[2],
        "box_size_x": box_size[0],
        "box_size_y": box_size[1],
        "box_size_z": box_size[2],
        "sg_x": sg[0],
        "sg_y": sg[1],
        "sg_z": sg[2],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold-dir", type=Path, required=True)
    args = ap.parse_args()
    result = rescore_cofold(args.cofold_dir)
    import json as _json
    print(_json.dumps({k: v for k, v in result.items() if k != "raw_stdout"}, indent=2))
    return 0 if result.get("success_flag") == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
