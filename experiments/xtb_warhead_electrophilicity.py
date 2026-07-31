#!/usr/bin/env python3
"""
xTB GFN2 warhead electrophilicity descriptors for acrylamide-bearing ligands.

For every input SMILES we identify the acrylamide warhead via the SMARTS
    [CH2]=[CH]-[C](=O)-[N]
The match indices are (Cβ, Cα, C=O, N). The Cβ atom is the Michael acceptor.

We then run xTB GFN2 single-point calculations on the free-state (or supplied
3D pose) geometry. From stdout / xtbout.json / `charges` we extract:

    HOMO_eV, LUMO_eV   (frontier orbital energies)
    gap_eV             (HOMO-LUMO gap)
    omega_eV           (Parr global electrophilicity ω = (H+L)²/(2·gap))
    q_Cb               (Mulliken partial charge on Cβ)
    fukui_plus_Cb      (electrophilic Fukui f+ on Cβ, from --vfukui)

We also output a literature-grounded QSAR predicting log(k2) for the
glutathione (GSH) reaction (Flanagan et al. JMC 2014; Awoonor-Williams &
Rowley JCIM 2018). The QSAR uses LUMO and q_Cβ in a simple linear form
fit qualitatively to span the literature 4-log-unit range:

    log k2 (GSH, 1/M/s)  ≈  -2.5 · (LUMO_eV + 7.0)  +  6.0 · q_Cb  -  1.0

Note: xTB GFN2 LUMO eigenvalues are systematically ~4-5 eV lower than
DFT/B3LYP (xTB calibration to an internal reference). The center value
-7.0 eV places the median acrylamide at log k2 ≈ -1. Lower LUMO → more
electrophilic (negative coefficient on LUMO+7); LESS NEGATIVE q_Cb
(closer to zero on the Mulliken scale) → more electrophilic — donor
substitution on the amide N pushes electrons onto Cβ, making q_Cb more
negative and SLOWING the Michael addition (positive coefficient on q_Cb,
so multiplying by a more-negative q gives a smaller log k2). This is a
heuristic, NOT a fitted regression — it ranks compounds but should NOT
be interpreted as an absolute rate. Downstream consumers can refit.

Usage:
    python experiments/xtb_warhead_electrophilicity.py --smiles "C=CC(=O)NCc1ccccc1"
    python experiments/xtb_warhead_electrophilicity.py --batch <manifest.json> \
        --out data/boltz_poses/zap70_cys346_warhead_reactivity.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Acrylamide SMARTS: terminal CH2=CH-C(=O)-N. Idx 0 = Cβ (Michael acceptor).
ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]-[C](=O)-[N]")

XTB_BIN = shutil.which("xtb")


def find_cb_index(mol: Chem.Mol) -> Optional[int]:
    """Return the 0-based atom index of the acrylamide Cβ (terminal =CH2 C)."""
    matches = mol.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None
    # First match, first atom = Cβ
    return matches[0][0]


def mol_to_xyz_string(mol: Chem.Mol, conf_id: int = 0) -> Tuple[str, int]:
    """Return (xyz_string, n_atoms) for the given conformer."""
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    lines = [str(n), "xtb input"]
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        p = conf.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol():<3s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines) + "\n", n


def prepare_3d_mol(smiles: str, sdf_path: Optional[str] = None) -> Tuple[Chem.Mol, int]:
    """Return (mol_with_H_and_3D_coords, cb_idx).

    If sdf_path is provided AND it parses, use its 3D coordinates. The atom
    ordering in SDF is preserved; we identify Cβ via SMARTS on the SDF mol.
    Otherwise embed and MMFF-minimize from SMILES.
    """
    if sdf_path and Path(sdf_path).exists():
        sup = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for cand in sup:
            if cand is None:
                continue
            cand = Chem.AddHs(cand, addCoords=True)
            cb = find_cb_index(cand)
            if cb is not None:
                return cand, cb
        # Fall through to SMILES embedding if SDF didn't yield a Cβ match
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    cb = find_cb_index(mol)
    if cb is None:
        raise ValueError(f"No acrylamide [CH2]=[CH]-[C](=O)-[N] in {smiles}")
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry with random coords
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"Embed failed for {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass
    return mol, cb


# Regex for parsing the Fukui stdout table from `xtb --vfukui`.
# Lines look like:    1C       0.031    0.032    0.031
_FUKUI_LINE = re.compile(
    r"^\s*(\d+)([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


def parse_fukui_stdout(stdout: str) -> Dict[int, Dict[str, float]]:
    """Parse the `--vfukui` table in xtb stdout. Keys are 0-based atom idx."""
    out: Dict[int, Dict[str, float]] = {}
    in_block = False
    for line in stdout.splitlines():
        if "Fukui functions:" in line:
            in_block = True
            continue
        if in_block:
            if "Property Printout" in line or "------" in line:
                # Block typically ends with a new section header
                if out:
                    break
                continue
            m = _FUKUI_LINE.match(line)
            if m:
                idx1 = int(m.group(1))
                out[idx1 - 1] = {
                    "f+": float(m.group(3)),
                    "f-": float(m.group(4)),
                    "f0": float(m.group(5)),
                }
            elif line.strip() == "":
                continue
            else:
                # Non-matching, non-blank line — table ended
                if out:
                    break
    return out


# Regex for the HOMO/LUMO orbital lines in xtb stdout. Two variants:
#   HOMO (occupied):   "  14   2.0000   -0.3936783   -10.7125 (HOMO)"
#   LUMO (unoccupied): "  15            -0.2639900    -7.1835 (LUMO)"  (occ col blank)
# The last numeric token before "(HOMO|LUMO)" is the eigenvalue in eV.
_ORBITAL_LINE = re.compile(
    r"^\s*\d+\s+.*?(-?\d+\.\d+)\s*\((HOMO|LUMO)\)\s*$"
)


def parse_homo_lumo_stdout(stdout: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse HOMO and LUMO eigenvalues (eV) from xtb stdout."""
    homo = lumo = None
    for line in stdout.splitlines():
        m = _ORBITAL_LINE.match(line)
        if m:
            val = float(m.group(1))
            tag = m.group(2)
            if tag == "HOMO":
                homo = val
            else:
                lumo = val
    return homo, lumo


def predict_log_k2_GSH(LUMO_eV: float, q_Cb: float) -> float:
    """Heuristic QSAR for log k2 of GSH addition to acrylamide warhead.

    Coefficients are chosen so that the typical literature range
    (LUMO ≈ -2.5 → -1.5 eV; q_Cβ ≈ 0.0 → +0.2) maps onto log k2 ≈ -3 → +1.
    This is illustrative, NOT a fitted regression. See module docstring.
    """
    return -2.5 * (LUMO_eV + 7.0) + 6.0 * q_Cb - 1.0


def compute_warhead_descriptors(
    smiles: str,
    sdf_path: Optional[str] = None,
    workdir: Optional[str] = None,
    keep_tmp: bool = False,
) -> Dict[str, float]:
    """Compute xTB GFN2 electrophilicity descriptors for one ligand.

    Returns dict with keys:
        q_Cb, LUMO_eV, HOMO_eV, gap_eV, omega_eV, fukui_plus_Cb,
        pred_log_k2_GSH, success_flag, error, cb_idx, n_atoms
    """
    rec: Dict[str, float] = {
        "q_Cb": np.nan,
        "LUMO_eV": np.nan,
        "HOMO_eV": np.nan,
        "gap_eV": np.nan,
        "omega_eV": np.nan,
        "fukui_plus_Cb": np.nan,
        "pred_log_k2_GSH": np.nan,
        "success_flag": 0,
        "error": "",
        "cb_idx": -1,
        "n_atoms": 0,
    }
    if XTB_BIN is None:
        rec["error"] = "xtb binary not found on PATH"
        return rec

    tmpdir_obj = None
    try:
        if workdir is None:
            tmpdir_obj = tempfile.TemporaryDirectory(prefix="xtb_warhead_")
            workdir = tmpdir_obj.name
        os.makedirs(workdir, exist_ok=True)

        mol, cb = prepare_3d_mol(smiles, sdf_path)
        rec["cb_idx"] = int(cb)
        rec["n_atoms"] = int(mol.GetNumAtoms())

        xyz_str, n_atoms = mol_to_xyz_string(mol)
        xyz_path = Path(workdir) / "input.xyz"
        xyz_path.write_text(xyz_str)

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("XTB_NUM_THREADS", "1")

        # Run xtb with --vfukui (which also gives orbital energies & charges)
        proc = subprocess.run(
            [XTB_BIN, "input.xyz", "--gfn", "2", "--vfukui", "--json"],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # "normal termination of xtb" goes to stderr for larger molecules,
        # stdout for smaller ones — check both.
        combined = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0 or "normal termination of xtb" not in combined:
            tail = (proc.stderr or proc.stdout)[-400:]
            rec["error"] = f"xtb failed: {tail.strip()}"
            return rec

        # HOMO / LUMO from stdout
        homo, lumo = parse_homo_lumo_stdout(proc.stdout)
        if homo is None or lumo is None:
            rec["error"] = "could not parse HOMO/LUMO"
            return rec
        rec["HOMO_eV"] = float(homo)
        rec["LUMO_eV"] = float(lumo)
        gap = float(lumo - homo)
        rec["gap_eV"] = gap
        if gap > 1e-6:
            rec["omega_eV"] = ((homo + lumo) ** 2) / (2.0 * gap)
        else:
            rec["omega_eV"] = np.nan

        # Mulliken charges from `charges` file (one per atom, in input order)
        charges_path = Path(workdir) / "charges"
        if charges_path.exists():
            qs = []
            for ln in charges_path.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    qs.append(float(ln.split()[0]))
                except ValueError:
                    pass
            if len(qs) >= n_atoms and cb < len(qs):
                rec["q_Cb"] = float(qs[cb])

        # Fukui indices from stdout (xtb prints a table with --vfukui)
        fukui = parse_fukui_stdout(proc.stdout)
        if cb in fukui:
            rec["fukui_plus_Cb"] = float(fukui[cb]["f+"])

        # Heuristic GSH k2
        if not (math.isnan(rec["LUMO_eV"]) or math.isnan(rec["q_Cb"])):
            rec["pred_log_k2_GSH"] = predict_log_k2_GSH(rec["LUMO_eV"], rec["q_Cb"])

        rec["success_flag"] = 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if tmpdir_obj is not None and not keep_tmp:
            try:
                tmpdir_obj.cleanup()
            except Exception:
                pass
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────


def _worker(job: Tuple[int, str, str, Optional[str]]) -> Dict[str, object]:
    row_id, name, smiles, sdf_path = job
    t0 = time.time()
    rec = compute_warhead_descriptors(smiles, sdf_path=sdf_path)
    return {
        "row_id": int(row_id),
        "name": name,
        "smiles": smiles,
        "q_Cb": rec["q_Cb"],
        "LUMO_eV": rec["LUMO_eV"],
        "HOMO_eV": rec["HOMO_eV"],
        "gap_eV": rec["gap_eV"],
        "omega_eV": rec["omega_eV"],
        "fukui_plus_Cb": rec["fukui_plus_Cb"],
        "pred_log_k2_GSH": rec["pred_log_k2_GSH"],
        "success_flag": int(rec["success_flag"]),
        "error": rec["error"],
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_batch(
    jobs,
    out_csv: Path,
    n_workers: int = 4,
    flush_every: int = 25,
) -> None:
    import pandas as pd

    rows = []
    start = time.time()
    print(f"[batch] {len(jobs)} jobs, {n_workers} workers, out={out_csv}")
    if n_workers == 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % flush_every == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                ok = sum(r["success_flag"] for r in rows)
                el = time.time() - start
                print(
                    f"  [{i}/{len(jobs)}] ok={ok} fail={i-ok}  "
                    f"elapsed={el/60:.1f} min  eta={el/i*(len(jobs)-i)/60:.1f} min"
                )
    else:
        with Pool(n_workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
                rows.append(rec)
                if i % flush_every == 0 or i == len(jobs):
                    pd.DataFrame(rows).sort_values("row_id").to_csv(out_csv, index=False)
                    ok = sum(r["success_flag"] for r in rows)
                    el = time.time() - start
                    eta = (el / i) * (len(jobs) - i) / 60
                    print(
                        f"  [{i}/{len(jobs)}] ok={ok} fail={i-ok}  "
                        f"elapsed={el/60:.1f} min  eta={eta:.1f} min",
                        flush=True,
                    )
    pd.DataFrame(rows).sort_values("row_id").to_csv(out_csv, index=False)
    el = time.time() - start
    ok = sum(r["success_flag"] for r in rows)
    print(f"[batch] DONE — {ok}/{len(rows)} ok, {el/60:.1f} min total, wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", help="Single SMILES to test", default=None)
    ap.add_argument("--sdf", help="Optional SDF pose for --smiles run", default=None)
    ap.add_argument("--batch", help="Path to manifest JSON (row_id -> {yaml_name, smiles})", default=None)
    ap.add_argument("--out", help="Output CSV for --batch", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pose-dir", help="Boltz predictions/ directory for SDFs", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Limit batch to first N (debug)")
    args = ap.parse_args()

    if args.smiles:
        rec = compute_warhead_descriptors(args.smiles, sdf_path=args.sdf)
        print(json.dumps(rec, indent=2, default=str))
        return

    if args.batch:
        assert args.out, "--out required with --batch"
        man = json.loads(Path(args.batch).read_text())
        jobs = []
        for rid_str, m in man.items():
            rid = int(rid_str)
            smi = m.get("smiles")
            name = m.get("yaml_name") or m.get("name") or str(rid)
            sdf = None
            if args.pose_dir:
                cand = Path(args.pose_dir) / name / f"{name}_model_0.lig.sdf"
                if cand.exists():
                    sdf = str(cand)
            jobs.append((rid, name, smi, sdf))
        if args.limit > 0:
            jobs = jobs[: args.limit]
        run_batch(jobs, Path(args.out), n_workers=args.workers)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
