#!/usr/bin/env python3
"""Vanilla AutoDock Vina scorer for ZAP70 (NO covalent tether).

Pipeline per mol:
  RDKit ETKDG embed → MMFF optimize → SDF → obabel PDBQT → vina dock (exh=1) → score

CLI:
  vanilla_vina_scorer.py <input.smi> <output.csv> [N_WORKERS=16]

Input: text file, one SMILES per line. Optional second column = row_id.
Output CSV: row_id, smiles, vina_kcalmol, status
"""
import os
import sys
import csv
import time
import json
import subprocess as sp
import tempfile
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

# Receptor + box (ZAP70 Cys346 pocket — same as AD-CovDock)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECEPTOR_PDBQT = PROJECT_ROOT / "data/docking_500/receptor_cys346_stripped.pdbqt"

# Pocket center: Cys346 SG from PDB 4K2R (matches run_covalent_docking.py line 51)
CENTER = (18.888, -3.650, -29.979)
SIZE = (20.0, 20.0, 20.0)

VINA = "/usr/bin/vina"
if not Path(VINA).exists():
    for alt in ("/usr/local/bin/vina",
                "/opt/miniconda3/envs/quris/bin/vina",
                "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/vina",
                "/home/shaharh_quris_ai/edit-small-mol/tools/vina"):
        if Path(alt).exists():
            VINA = alt
            break


def smi_to_pdbqt(smi: str, out_pdbqt: Path) -> bool:
    """Embed SMILES with RDKit + write PDBQT with meeko MoleculePreparation.

    Uses meeko (NO CovalentBuilder — vanilla non-covalent docking) for correct
    PDBQT atom typing + rotatable-bond detection. obabel is avoided because it
    mis-types aromatic N (per QA agent: ~0.3–0.8 kcal/mol systematic shift).
    """
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=20) < 0:
            return False
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        # meeko vanilla prep
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(m)
        if not setups:
            return False
        pdbqt_str, is_ok, _msg = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            return False
        out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        out_pdbqt.write_text(pdbqt_str)
        return out_pdbqt.exists() and out_pdbqt.stat().st_size > 0
    except Exception:
        return False


def run_vina(lig_pdbqt: Path) -> tuple[float, str]:
    """Run vina dock with exh=1 num_modes=1, return (kcal/mol, status)."""
    if not RECEPTOR_PDBQT.exists():
        return float("nan"), "no_receptor"
    cx, cy, cz = CENTER
    sx, sy, sz = SIZE
    out_pdbqt = lig_pdbqt.with_suffix(".dock.pdbqt")
    cmd = [VINA, "--receptor", str(RECEPTOR_PDBQT),
           "--ligand", str(lig_pdbqt),
           "--out", str(out_pdbqt),
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--exhaustiveness", "1", "--num_modes", "1", "--cpu", "1"]
    try:
        res = sp.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return float("nan"), f"vina_err:{res.returncode}"
        # Parse stdout for the best-pose energy line
        # "REMARK VINA RESULT:   -7.123    0.000    0.000"
        for line in res.stdout.splitlines():
            if "REMARK VINA RESULT" in line:
                parts = line.split()
                for tok in parts:
                    try:
                        score = float(tok)
                        if -30 < score < 30:
                            return score, "ok"
                    except ValueError:
                        continue
        # Try parsing from out_pdbqt
        if out_pdbqt.exists():
            for line in out_pdbqt.read_text().splitlines():
                if line.startswith("REMARK VINA RESULT"):
                    parts = line.split()
                    for tok in parts:
                        try:
                            score = float(tok)
                            if -30 < score < 30:
                                return score, "ok_from_pdbqt"
                        except ValueError:
                            continue
        return float("nan"), "no_score_line"
    except sp.TimeoutExpired:
        return float("nan"), "timeout"
    except Exception as e:
        return float("nan"), f"exc:{type(e).__name__}"
    finally:
        try: out_pdbqt.unlink(missing_ok=True)
        except Exception: pass


_WORK_DIR_GLOBAL = None


def _init_worker(work_dir_str):
    global _WORK_DIR_GLOBAL
    _WORK_DIR_GLOBAL = Path(work_dir_str)


def score_one(args):
    row_id, smi = args
    work = _WORK_DIR_GLOBAL
    lig = work / f"lig_{os.getpid()}_{row_id}.pdbqt"
    try:
        if not smi_to_pdbqt(smi, lig):
            return (row_id, smi, float("nan"), "prep_fail")
        score, status = run_vina(lig)
        return (row_id, smi, score, status)
    finally:
        try: lig.unlink(missing_ok=True)
        except Exception: pass


def main():
    if len(sys.argv) < 3:
        print("usage: vanilla_vina_scorer.py <input.smi> <output.csv> [N_WORKERS=16]", file=sys.stderr)
        sys.exit(2)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    N_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    # Read input
    tasks = []
    with open(inp) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            smi = parts[0]
            row_id = parts[1] if len(parts) > 1 else str(i)
            tasks.append((row_id, smi))
    n_total = len(tasks)

    # Resume: skip row_ids already present in output CSV
    done_ids = set()
    resume_mode = out.exists() and out.stat().st_size > 0
    if resume_mode:
        with open(out) as f_in:
            rdr = csv.reader(f_in)
            try:
                header = next(rdr)
                # validate header
                if header[:1] != ["row_id"]:
                    print(f"  WARN: existing {out} header looks wrong: {header[:4]}; ignoring resume", file=sys.stderr)
                    resume_mode = False
                else:
                    for row in rdr:
                        if row:
                            done_ids.add(row[0])
            except StopIteration:
                resume_mode = False
    tasks = [(rid, smi) for (rid, smi) in tasks if rid not in done_ids]
    n = len(tasks)
    if resume_mode:
        print(f"[vanilla-vina] RESUME: {len(done_ids):,} already done in {out.name}; {n:,} remaining (of {n_total:,})", flush=True)
    print(f"[vanilla-vina] {n:,} mols  ·  {N_WORKERS} workers  ·  receptor={RECEPTOR_PDBQT.name}", flush=True)
    if n == 0:
        print(f"[vanilla-vina] nothing to do (all {n_total:,} rows already in {out})", flush=True)
        sys.exit(0)
    if not RECEPTOR_PDBQT.exists():
        print(f"  ERROR: receptor not found", file=sys.stderr); sys.exit(3)
    if not Path(VINA).exists():
        print(f"  ERROR: vina not found at {VINA}", file=sys.stderr); sys.exit(3)
    print(f"  box center: {CENTER}  ·  size: {SIZE}  ·  vina: {VINA}", flush=True)
    # Verify vina version + receptor sha256 for cross-machine reproducibility
    import hashlib
    sha = hashlib.sha256(RECEPTOR_PDBQT.read_bytes()).hexdigest()[:12]
    v = sp.run([VINA, "--version"], capture_output=True, text=True).stdout.strip().split("\n")[0]
    print(f"  receptor sha256: {sha}  ·  vina: {v}", flush=True)

    # Working dir for pdbqt files (RAM-backed if /dev/shm available)
    work_dir = Path("/dev/shm/vina_work") if Path("/dev/shm").exists() else Path("/tmp/vina_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Streaming CSV output (one row per mol, written as completed).
    # Append mode if resuming so we preserve prior progress on SPOT preemption restart.
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not resume_mode
    f_out = open(out, "a" if resume_mode else "w", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(["row_id", "smiles", "vina_kcalmol", "status"])
        f_out.flush()

    t0 = time.time()
    n_done = 0
    n_ok = 0
    last_log = t0
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(str(work_dir),)) as pool:
        for row_id, smi, score, status in pool.imap_unordered(score_one, tasks, chunksize=4):
            writer.writerow([row_id, smi, "" if (score != score) else score, status])
            f_out.flush()
            n_done += 1
            if status == "ok" or status == "ok_from_pdbqt":
                n_ok += 1
            now = time.time()
            if now - last_log > 30:
                rate = n_done / (now - t0)
                eta_min = (n - n_done) / max(1e-3, rate) / 60
                print(f"  ...{n_done:,}/{n:,}  ({(now-t0):.0f}s, {rate:.2f}/s, ETA {eta_min:.1f} min, ok={n_ok})", flush=True)
                last_log = now
    f_out.close()
    dt = time.time() - t0
    print(f"\n[vanilla-vina] done in {dt:.0f}s ({n/dt:.2f}/s)  ·  ok: {n_ok:,}/{n:,}", flush=True)
    print(f"  → {out}", flush=True)


if __name__ == "__main__":
    main()
