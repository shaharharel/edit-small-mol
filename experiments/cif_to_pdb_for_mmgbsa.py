"""Convert Boltz CIF cofold outputs to PDB files suitable for MM-GBSA scoring.

For each `<name>_model_0.cif` in a predictions directory (one subdir per pose),
writes a sibling `<name>_model_0.pdb` containing:
  - ATOM records for the protein
  - HETATM records for the ligand, renamed from `LIG1` → `LIG` (3-letter)

Atom names, coordinates, chain IDs and atom serial numbers are preserved
verbatim from the CIF, which is what `score_cofold_energy.py:split_pdb()`
expects (Boltz atom-naming convention "C25" = element + canonical_rank + 1).

Cofolds that already have a `_model_0.pdb` are skipped (legacy smoke-test
files are preserved).

Usage:
  python experiments/cif_to_pdb_for_mmgbsa.py <predictions_root> [--workers N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

import gemmi


# Residue name in the source CIF that Boltz uses for the ligand.
BOLTZ_LIGAND_RESNAME = "LIG1"
# Residue name we emit in the PDB (3 letters; this is what split_pdb() uses
# to identify HETATM lines.)
TARGET_LIGAND_RESNAME = "LIG"


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> dict:
    """Convert one Boltz CIF to a PDB suitable for `score_cofold_energy.py`.

    Renames any `LIG1` residue to `LIG` and marks it as a non-polymer HETATM.
    All other residues, atoms, coordinates, chain IDs and serials are passed
    through unchanged.

    Returns a small diagnostic dict.
    """
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError(f"empty structure in {cif_path}")

    n_atom = 0
    n_hetatm = 0
    found_ligand = False
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == BOLTZ_LIGAND_RESNAME:
                    res.name = TARGET_LIGAND_RESNAME
                    res.het_flag = "H"
                    res.entity_type = gemmi.EntityType.NonPolymer
                    found_ligand = True
                    n_hetatm += len(res)
                else:
                    n_atom += len(res)

    if not found_ligand:
        # Some cofolds may use a different ligand resname — try to detect any
        # non-standard residue and convert it.
        standard = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "HOH", "WAT",
        }
        for model in st:
            for chain in model:
                for res in chain:
                    if res.name not in standard and res.name != TARGET_LIGAND_RESNAME:
                        res.name = TARGET_LIGAND_RESNAME
                        res.het_flag = "H"
                        res.entity_type = gemmi.EntityType.NonPolymer
                        found_ligand = True
                        n_hetatm += len(res)

    if not found_ligand:
        raise ValueError(f"no ligand residue found in {cif_path}")

    st.write_pdb(str(pdb_path))
    return {
        "cif": str(cif_path),
        "pdb": str(pdb_path),
        "n_atom": n_atom,
        "n_hetatm": n_hetatm,
    }


def _worker(args):
    cif_path, pdb_path = args
    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path)
    try:
        info = convert_cif_to_pdb(cif_path, pdb_path)
        return {"ok": True, **info}
    except Exception as e:
        return {
            "ok": False,
            "cif": str(cif_path),
            "pdb": str(pdb_path),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-500:],
        }


def find_jobs(predictions_root: Path, force: bool = False) -> list[tuple[Path, Path]]:
    """Return (cif, pdb) pairs for cofolds that need conversion.

    Skips any pose where the PDB already exists unless `force=True`.
    """
    jobs: list[tuple[Path, Path]] = []
    for sub in sorted(predictions_root.iterdir()):
        if not sub.is_dir():
            continue
        cifs = list(sub.glob("*_model_0.cif"))
        if not cifs:
            continue
        if len(cifs) > 1:
            print(f"WARNING: multiple CIFs in {sub.name}: {[c.name for c in cifs]}", file=sys.stderr)
        cif = cifs[0]
        pdb = cif.with_suffix(".pdb")
        if pdb.exists() and not force:
            continue
        jobs.append((cif, pdb))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "predictions_root",
        type=Path,
        help="Boltz `predictions/` directory containing one subdir per cofold.",
    )
    ap.add_argument("--workers", type=int, default=8, help="Multiprocessing workers (default 8).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing PDB files.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N (testing).")
    args = ap.parse_args()

    if not args.predictions_root.exists():
        print(f"ERROR: predictions_root does not exist: {args.predictions_root}", file=sys.stderr)
        sys.exit(2)

    jobs = find_jobs(args.predictions_root, force=args.force)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Found {len(jobs)} CIF files needing conversion.")
    if not jobs:
        return

    n_ok = 0
    failures: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            r = _worker(j)
            if r["ok"]:
                n_ok += 1
            else:
                failures.append(r)
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4)):
                if r["ok"]:
                    n_ok += 1
                else:
                    failures.append(r)
                if (i + 1) % 100 == 0:
                    print(f"  ... {i + 1}/{len(jobs)} done ({n_ok} ok, {len(failures)} failed)")

    print(f"\nDone: {n_ok}/{len(jobs)} converted successfully; {len(failures)} failures.")
    for f in failures[:20]:
        print(f"  FAIL: {Path(f['cif']).name}: {f['error']}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more failures.")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
