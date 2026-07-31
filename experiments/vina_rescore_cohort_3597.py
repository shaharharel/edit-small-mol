"""Vina --score_only rescore for the cohort_3597 Boltz cofold pool.

Operates over the multi-machine layout:
  data/boltz_results/cohort_3597_full/from_<a|b|c|d|e|f>/<yaml_name>/

Each cofold subdir has only:
  <yaml_name>_model_0.cif
  confidence_<yaml_name>_model_0.json

This script:
  1. Converts the mmCIF -> full-complex PDB (ATOM=protein, HETATM=ligand, resname LIG).
  2. Calls experiments.vina_rescore_cofolds.rescore_cofold on the cofold dir,
     which handles the protein/ligand split, PDBQT prep, and Vina scoring.
  3. Appends a row to data/tier4_scored/vina_rescore_boltz_intermediate.csv
     with the schema requested by the orchestrator.

Resumable: rows whose yaml_name already has success_flag=1 in the output CSV
are skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Make quris bin available for the obabel/vina/meeko subprocess calls.
QURIS_BIN = "/opt/miniconda3/envs/quris/bin"
if QURIS_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = QURIS_BIN + os.pathsep + os.environ.get("PATH", "")

from experiments.vina_rescore_cofolds import rescore_cofold

DEFAULT_ROOT = PROJECT_ROOT / "data" / "boltz_results" / "cohort_3597_full"
DEFAULT_OUT = PROJECT_ROOT / "data" / "tier4_scored" / "vina_rescore_boltz_intermediate.csv"

FIELDS = [
    "yaml_name",
    "row_id",
    "vina_affinity_kcalmol",
    "vina_inter_kcalmol",
    "vina_intra_kcalmol",
    "vina_torsions_kcalmol",
    "vina_unbound_kcalmol",
    "success_flag",
    "error",
    "boltz_machine",
    "n_protein_atoms",
    "n_lig_atoms",
    "ts",
]

_ROW_ID_RE = re.compile(r"(\d+)$")


def yaml_to_row_id(yaml_name: str):
    m = _ROW_ID_RE.search(yaml_name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# CIF -> full complex PDB
# --------------------------------------------------------------------------

def _fmt_atom_name(atom_id: str, sym: str) -> str:
    atom_id = atom_id.strip().strip('"').strip("'")
    if len(atom_id) >= 4:
        return atom_id[:4]
    if len(sym) == 1:
        return f" {atom_id:<3}"
    return f"{atom_id:<4}"


def cif_to_full_pdb(cif_path: Path, pdb_path: Path) -> tuple[int, int]:
    """Parse Boltz mmCIF and write a full-complex PDB.

    ATOM records = polymer (chain A), resname kept.
    HETATM records = ligand, resname forced to LIG so the downstream splitter
    in vina_rescore_cofolds.split_pdb_to_protein_and_ligand can find them.
    Returns (n_protein_atoms, n_lig_atoms).
    """
    with open(cif_path) as f:
        raw = f.readlines()

    # Find _atom_site loop header columns.
    header = []
    body_start = None
    in_atom_loop = False
    for i, line in enumerate(raw):
        s = line.strip()
        if s.startswith("_atom_site."):
            header.append(s)
            in_atom_loop = True
        elif in_atom_loop and not s.startswith("_atom_site."):
            body_start = i
            break
    if not header or body_start is None:
        raise RuntimeError(f"no _atom_site loop in {cif_path}")

    idx = {col.replace("_atom_site.", ""): k for k, col in enumerate(header)}
    required = ["group_PDB", "type_symbol", "label_atom_id", "label_comp_id",
                "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    for n in required:
        if n not in idx:
            raise RuntimeError(f"missing column {n} in {cif_path}")

    asym_idx = idx.get("auth_asym_id", idx.get("label_asym_id"))
    occ_idx = idx.get("occupancy")
    bf_idx = idx.get("B_iso_or_equiv")

    prot = []
    lig = []
    serial = 0
    for ln in raw[body_start:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
            break
        toks = s.split()
        if len(toks) < len(header):
            continue
        group = toks[idx["group_PDB"]]
        if group not in ("ATOM", "HETATM"):
            continue
        serial += 1
        sym = toks[idx["type_symbol"]]
        atom_id = toks[idx["label_atom_id"]]
        comp = toks[idx["label_comp_id"]]
        chain_id = toks[asym_idx][0] if asym_idx is not None else "A"
        try:
            res_seq = int(toks[idx["auth_seq_id"]])
        except (ValueError, IndexError):
            res_seq = 1
        x = float(toks[idx["Cartn_x"]])
        y = float(toks[idx["Cartn_y"]])
        z = float(toks[idx["Cartn_z"]])
        occ = 1.0
        bfac = 0.0
        if occ_idx is not None:
            try:
                occ = float(toks[occ_idx])
            except (ValueError, IndexError):
                occ = 1.0
        if bf_idx is not None:
            try:
                bfac = float(toks[bf_idx])
            except (ValueError, IndexError):
                bfac = 0.0

        if group == "HETATM":
            resname = "LIG"
        else:
            resname = comp[:3]

        name_field = _fmt_atom_name(atom_id, sym)
        rec = (
            f"{group:<6}{serial:>5} {name_field} {resname:>3} "
            f"{chain_id:1}{res_seq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfac:>6.2f}"
            f"          {sym:>2}\n"
        )
        if group == "ATOM":
            prot.append(rec)
        else:
            lig.append(rec)

    if not prot:
        raise RuntimeError(f"no protein ATOM records in {cif_path}")
    if not lig:
        raise RuntimeError(f"no ligand HETATM records in {cif_path}")

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_path, "w") as f:
        f.writelines(prot)
        f.write("TER\n")
        f.writelines(lig)
        f.write("END\n")

    return len(prot), len(lig)


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
        "vina_affinity_kcalmol": None,
        "vina_inter_kcalmol": None,
        "vina_intra_kcalmol": None,
        "vina_torsions_kcalmol": None,
        "vina_unbound_kcalmol": None,
        "success_flag": 0,
        "error": None,
        "boltz_machine": machine_letter,
        "n_protein_atoms": None,
        "n_lig_atoms": None,
        "ts": _ts(),
    }

    cif_path = cofold_dir / f"{yaml_name}_model_0.cif"
    pdb_path = cofold_dir / f"{yaml_name}_model_0.pdb"

    if not cif_path.exists():
        row["error"] = f"missing CIF: {cif_path.name}"
        return row

    # CIF -> full PDB (only if not already produced).
    try:
        if not pdb_path.exists():
            n_prot, n_lig = cif_to_full_pdb(cif_path, pdb_path)
        else:
            # Quick count by grepping.
            n_prot = sum(1 for ln in open(pdb_path) if ln.startswith("ATOM"))
            n_lig = sum(1 for ln in open(pdb_path) if ln.startswith("HETATM"))
        row["n_protein_atoms"] = n_prot
        row["n_lig_atoms"] = n_lig
    except Exception as e:
        row["error"] = f"cif_to_pdb: {type(e).__name__}: {e}"
        return row

    # Call existing rescore pipeline.
    try:
        res = rescore_cofold(cofold_dir)
    except Exception as e:
        row["error"] = f"rescore_uncaught: {type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
        return row

    row["success_flag"] = int(res.get("success_flag", 0))
    row["error"] = res.get("error")
    row["vina_affinity_kcalmol"] = res.get("vina_kcalmol")
    row["vina_inter_kcalmol"] = res.get("vina_inter_kcalmol")
    row["vina_intra_kcalmol"] = res.get("vina_intra_kcalmol")
    row["vina_torsions_kcalmol"] = res.get("vina_torsion_kcalmol")
    row["vina_unbound_kcalmol"] = res.get("vina_unbound_kcalmol")
    row["ts"] = _ts()
    return row


# --------------------------------------------------------------------------
# Main
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
    """Return [(cofold_dir, machine_letter), ...]."""
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
            if cif.exists():
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
    # If file exists but schema mismatches FIELDS, we still append; reader will be lenient.
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
                    "vina_affinity_kcalmol": None,
                    "vina_inter_kcalmol": None,
                    "vina_intra_kcalmol": None,
                    "vina_torsions_kcalmol": None,
                    "vina_unbound_kcalmol": None,
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
