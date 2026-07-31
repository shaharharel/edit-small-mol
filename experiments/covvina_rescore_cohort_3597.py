"""Covalent-Vina --score_only rescore for the cohort_3597 Boltz cofold pool.

Sibling of ``vina_rescore_cohort_3597.py`` but uses the AD-CovDock receptor
(Cys346 SG+CB stripped) and the 40 Å covalent box centred on Cys346 SG.

Per cofold:
  1. Read ``<id>_model_0.lig.sdf`` (Boltz 3D ligand, warhead already covalently
     bonded to Cys346 SG).
  2. Prepare PDBQT from the SDF preserving its 3D coords (no embed, no
     minimisation) via meeko's ``MoleculePreparation``.
  3. Run ``vina --score_only`` against ``RECEPTOR_PDBQT_STRIPPED`` with the
     ``BOX_SIZE_ADCOV`` (40 Å) box.
  4. Parse Affinity / inter / intra / torsions / unbound from stdout.

Writes to ``data/tier4_scored/covvina_rescore_boltz_intermediate.csv`` with the
same 13-column schema as the vanilla pipeline but with ``covvina_*_kcalmol``
column names. Multiprocessing 6 workers. Resumable.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

QURIS_BIN = "/opt/miniconda3/envs/quris/bin"
if QURIS_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = QURIS_BIN + os.pathsep + os.environ.get("PATH", "")

from experiments.run_covalent_docking import (  # noqa: E402
    BOX_SIZE_ADCOV,
    VINA_BIN,
)
from experiments.vina_rescore_cohort_3597 import (  # noqa: E402
    cif_to_full_pdb,
    yaml_to_row_id,
)
from experiments.vina_rescore_cofolds import (  # noqa: E402
    split_pdb_to_protein_and_ligand,
    parse_vina_score,
    find_cys346_sg,
    protein_to_pdbqt,
)

DEFAULT_ROOT = PROJECT_ROOT / "data" / "boltz_results" / "cohort_3597_full"
DEFAULT_OUT = PROJECT_ROOT / "data" / "tier4_scored" / "covvina_rescore_boltz_intermediate.csv"

FIELDS = [
    "yaml_name",
    "row_id",
    "covvina_affinity_kcalmol",
    "covvina_inter_kcalmol",
    "covvina_intra_kcalmol",
    "covvina_torsions_kcalmol",
    "covvina_unbound_kcalmol",
    "success_flag",
    "error",
    "boltz_machine",
    "n_protein_atoms",
    "n_lig_atoms",
    "ts",
]


VINA = str(VINA_BIN) if VINA_BIN.exists() else "vina"


# --------------------------------------------------------------------------
# Pose-preserving ligand prep
# --------------------------------------------------------------------------

def _prepare_ligand_pdbqt_pose(lig_sdf: Path, out_pdbqt: Path) -> None:
    """Convert a Boltz ligand SDF -> PDBQT preserving the 3D coordinates.

    Uses RDKit to load the SDF (with Hs) and meeko to write the PDBQT directly
    from those coords — no CovalentBuilder, no embed.

    Falls back to ``mk_prepare_ligand.py`` (subprocess) and then ``obabel`` if
    the in-process meeko path fails. None of the fall-backs re-embeds.
    """
    from rdkit import Chem

    # --- Attempt 1: in-process meeko on the RDKit Mol -----------------------
    try:
        mol = Chem.MolFromMolFile(str(lig_sdf), removeHs=False, sanitize=True)
        if mol is None:
            raise RuntimeError("RDKit could not parse lig.sdf")
        # Make sure Hs are present (Boltz SDFs include them but be safe).
        if not any(a.GetSymbol() == "H" for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)

        from meeko import MoleculePreparation, PDBQTWriterLegacy

        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        if not setups:
            raise RuntimeError("meeko returned no setups")
        pdbqt_str, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise RuntimeError(f"meeko write fail: {msg}")
        out_pdbqt.write_text(pdbqt_str)
        if out_pdbqt.stat().st_size == 0:
            raise RuntimeError("meeko wrote empty pdbqt")
        return
    except Exception as e_meeko:
        last_err = f"meeko_inproc: {type(e_meeko).__name__}: {e_meeko}"

    # --- Attempt 2: mk_prepare_ligand.py subprocess -------------------------
    try:
        cmd = ["mk_prepare_ligand.py", "-i", str(lig_sdf), "-o", str(out_pdbqt)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
            return
        last_err = f"{last_err}; mk_subproc rc={res.returncode}: {res.stderr[:200]}"
    except Exception as e_sub:
        last_err = f"{last_err}; mk_subproc exc: {e_sub}"

    # --- Attempt 3: obabel (no --gen3d → preserves coords) ------------------
    try:
        cmd = ["obabel", str(lig_sdf), "-O", str(out_pdbqt), "-h"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
            return
        raise RuntimeError(f"obabel rc={res.returncode}: {res.stderr[:200]}")
    except Exception as e_ob:
        raise RuntimeError(f"all ligand preps failed. {last_err}; obabel: {e_ob}")


# --------------------------------------------------------------------------
# Cov-Vina --score_only on existing pose
# --------------------------------------------------------------------------

def _strip_cys346_cb_sg(prot_pdb: Path, stripped_pdb: Path) -> None:
    """Write a copy of ``prot_pdb`` with CYS A 346 SG and CB atoms removed.

    Mirrors ``run_covalent_docking.prepare_stripped_receptor()`` logic but
    against this cofold's own protein PDB (not the 4K2R reference).
    """
    out_lines = []
    for ln in open(prot_pdb):
        if ln.startswith("ATOM") and ln[17:20].strip() == "CYS" and ln[22:26].strip() == "346":
            atom_name = ln[12:16].strip()
            if atom_name in ("CB", "SG"):
                continue
        out_lines.append(ln)
    stripped_pdb.write_text("".join(out_lines))


def _run_cov_vina_score_only(
    rec_pdbqt: Path,
    lig_pdbqt: Path,
    center: tuple[float, float, float],
) -> dict:
    """Invoke vina --score_only against the stripped-receptor with a 40 Å box
    centred on the Cys346 SG of *this* cofold's protein."""
    cmd = [
        VINA,
        "--receptor", str(rec_pdbqt),
        "--ligand", str(lig_pdbqt),
        "--score_only",
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{float(BOX_SIZE_ADCOV[0]):.3f}",
        "--size_y", f"{float(BOX_SIZE_ADCOV[1]):.3f}",
        "--size_z", f"{float(BOX_SIZE_ADCOV[2]):.3f}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(
            f"vina rc={res.returncode}: {res.stderr[:400]}"
        )
    parsed = parse_vina_score(res.stdout)
    if parsed["vina_kcalmol"] is None:
        raise RuntimeError(f"no Affinity in stdout: {res.stdout[-400:]}")
    return parsed


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def _ts() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _worker(args: tuple) -> dict:
    cofold_dir_str, machine_letter = args
    cofold_dir = Path(cofold_dir_str)
    yaml_name = cofold_dir.name
    row = {
        "yaml_name": yaml_name,
        "row_id": yaml_to_row_id(yaml_name),
        "covvina_affinity_kcalmol": None,
        "covvina_inter_kcalmol": None,
        "covvina_intra_kcalmol": None,
        "covvina_torsions_kcalmol": None,
        "covvina_unbound_kcalmol": None,
        "success_flag": 0,
        "error": None,
        "boltz_machine": machine_letter,
        "n_protein_atoms": None,
        "n_lig_atoms": None,
        "ts": _ts(),
    }

    cif_path = cofold_dir / f"{yaml_name}_model_0.cif"
    pdb_path = cofold_dir / f"{yaml_name}_model_0.pdb"
    prot_pdb = cofold_dir / f"{yaml_name}_model_0.prot.pdb"
    lig_sdf = cofold_dir / f"{yaml_name}_model_0.lig.sdf"

    if not cif_path.exists() and not pdb_path.exists():
        row["error"] = "missing CIF and PDB"
        return row

    # Ensure full PDB exists (the vanilla pipeline normally produced this; we
    # regenerate it if absent so we can run independently).
    try:
        if not pdb_path.exists():
            n_prot, n_lig = cif_to_full_pdb(cif_path, pdb_path)
            row["n_protein_atoms"] = n_prot
            row["n_lig_atoms"] = n_lig
    except Exception as e:
        row["error"] = f"cif_to_pdb: {type(e).__name__}: {e}"
        return row

    # Ensure prot.pdb + lig.sdf exist.
    try:
        if not prot_pdb.exists() or not lig_sdf.exists():
            split_pdb_to_protein_and_ligand(pdb_path, prot_pdb, lig_sdf)
        if row["n_protein_atoms"] is None:
            row["n_protein_atoms"] = sum(
                1 for ln in open(pdb_path) if ln.startswith("ATOM")
            )
            row["n_lig_atoms"] = sum(
                1 for ln in open(pdb_path) if ln.startswith("HETATM")
            )
    except Exception as e:
        row["error"] = f"split_pdb: {type(e).__name__}: {e}"
        return row

    # Find this cofold's Cys346 SG (box center) BEFORE stripping it.
    sg = find_cys346_sg(prot_pdb)
    if sg is None:
        row["error"] = "no Cys346 SG in cofold protein"
        return row

    # Prepare stripped receptor + ligand PDBQT, then run cov-Vina --score_only.
    with tempfile.TemporaryDirectory(prefix=f"covvina_{yaml_name}_") as tmpd:
        tmp = Path(tmpd)
        rec_stripped_pdb = tmp / "rec_stripped.pdb"
        rec_pdbqt = tmp / "rec.pdbqt"
        lig_pdbqt = tmp / "lig.pdbqt"

        try:
            _strip_cys346_cb_sg(prot_pdb, rec_stripped_pdb)
        except Exception as e:
            row["error"] = f"strip_rec: {type(e).__name__}: {e}"
            return row

        try:
            protein_to_pdbqt(rec_stripped_pdb, rec_pdbqt)
        except Exception as e:
            row["error"] = f"obabel_prot: {type(e).__name__}: {e}"
            return row

        try:
            _prepare_ligand_pdbqt_pose(lig_sdf, lig_pdbqt)
        except Exception as e:
            row["error"] = f"lig_prep: {type(e).__name__}: {e}"
            return row

        try:
            parsed = _run_cov_vina_score_only(rec_pdbqt, lig_pdbqt, tuple(sg))
        except Exception as e:
            row["error"] = f"vina: {type(e).__name__}: {str(e)[:300]}"
            return row

    row["covvina_affinity_kcalmol"] = parsed["vina_kcalmol"]
    row["covvina_inter_kcalmol"] = parsed["vina_inter_kcalmol"]
    row["covvina_intra_kcalmol"] = parsed["vina_intra_kcalmol"]
    row["covvina_torsions_kcalmol"] = parsed.get("vina_torsion_kcalmol")
    row["covvina_unbound_kcalmol"] = parsed.get("vina_unbound_kcalmol")
    row["success_flag"] = 1
    row["ts"] = _ts()
    return row


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("success_flag")) == "1":
                done.add(row["yaml_name"])
    return done


def discover(root: Path) -> list[tuple[Path, str]]:
    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"from_([a-z])$", sub.name)
        if not m:
            continue
        letter = m.group(1)
        for cof in sorted(sub.iterdir()):
            if not cof.is_dir():
                continue
            cif = cof / f"{cof.name}_model_0.cif"
            pdb = cof / f"{cof.name}_model_0.pdb"
            if cif.exists() or pdb.exists():
                out.append((cof, letter))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cofolds = discover(args.cofold_root)
    print(f"Discovered {len(cofolds)} cofolds under {args.cofold_root}", flush=True)
    if args.limit:
        cofolds = cofolds[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else load_done(args.out)
    todo = [(d, m) for (d, m) in cofolds if d.name not in done]
    print(
        f"Already done: {len(done)} | to process: {len(todo)} | workers: {args.workers}",
        flush=True,
    )
    if not todo:
        print("Nothing to do.")
        return 0

    new_file = (not args.out.exists()) or args.out.stat().st_size == 0
    f_out = open(args.out, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        f_out.flush()

    t0 = time.time()
    n_ok = 0
    n_fail = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = {exc.submit(_worker, (str(d), m)): (d, m) for (d, m) in todo}
        for fut in as_completed(futures):
            d, m = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "yaml_name": d.name,
                    "row_id": yaml_to_row_id(d.name),
                    "covvina_affinity_kcalmol": None,
                    "covvina_inter_kcalmol": None,
                    "covvina_intra_kcalmol": None,
                    "covvina_torsions_kcalmol": None,
                    "covvina_unbound_kcalmol": None,
                    "success_flag": 0,
                    "error": f"future_uncaught: {type(e).__name__}: {e}",
                    "boltz_machine": m,
                    "n_protein_atoms": None,
                    "n_lig_atoms": None,
                    "ts": _ts(),
                }
            writer.writerow({k: row.get(k) for k in FIELDS})
            f_out.flush()
            if row["success_flag"] == 1:
                n_ok += 1
            else:
                n_fail += 1
            n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print(
                    f"  [{n_done}/{len(todo)}] ok={n_ok} fail={n_fail}  "
                    f"{rate:.2f} cof/s  ETA {eta/60:.1f} min",
                    flush=True,
                )

    f_out.close()
    print(
        f"DONE  ok={n_ok} fail={n_fail}  elapsed {(time.time()-t0)/60:.1f} min  -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
