"""Batch driver: run score_cofold_energy.py on a directory of cofold CIFs in parallel.

Usage:
  python score_cofold_batch.py <cif_root_dir> <output_csv> [--workers N] [--limit M]

For each CIF, calls score_one() with the corresponding target/name derived from the path.
Outputs CSV incrementally so we can resume if interrupted.
"""
import argparse, csv, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tempfile, shutil

sys.path.insert(0, str(Path(__file__).parent))
from score_cofold_energy import score_one


def cif_to_pdb(cif_path: Path) -> Path:
    """Convert a Boltz CIF to PDB using gemmi or pdbfixer.
    Returns path to converted PDB in a temp location (caller cleans)."""
    pdb_path = cif_path.with_suffix(".pdb")
    if pdb_path.exists():
        return pdb_path
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        st.write_pdb(str(pdb_path))
        return pdb_path
    except ImportError:
        # Fallback: pdbfixer can read both
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        fixer = PDBFixer(filename=str(cif_path))
        with open(pdb_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return pdb_path


def process_one(cif_path: str, target: str, name: str, smiles: str | None = None):
    cif = Path(cif_path)
    try:
        pdb = cif_to_pdb(cif)
        res = score_one(pdb, target=target, name=name, smiles=smiles)
        return res
    except Exception as e:
        return {
            "target": target, "name": name,
            "success_flag": 0, "error": f"setup_error: {e!r}",
            "dG_bind_kcalmol": None, "ligand_strain_kcalmol": None,
            "E_complex": None, "E_protein": None, "E_ligand_bound": None, "E_ligand_free": None,
            "rmsd_min_A": None, "sg_cb_dist_A": None,
        }


def gather_cifs(root: Path):
    """Find all Boltz cofold CIFs under root: */boltz_results_*/predictions/<name>/<name>_model_0.cif"""
    out = []
    for cif in root.rglob("*_model_0.cif"):
        # Path: <root>/<target>/boltz_results_<name>/predictions/<name>/<name>_model_0.cif
        try:
            name = cif.parent.name
            target = cif.parents[3].name
        except Exception:
            continue
        out.append((str(cif), target, name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root dir of Boltz cofolds")
    ap.add_argument("output", type=Path, help="Output CSV")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip rows already in output CSV")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="CSV with target,name,smiles columns to provide SMILES for ligand reconstruction")
    args = ap.parse_args()

    # Load SMILES manifest if provided
    smiles_map = {}
    if args.manifest and args.manifest.exists():
        with open(args.manifest) as f:
            r = csv.DictReader(f)
            for row in r:
                smiles_map[(row["target"], row["name"])] = row["smiles"]
        print(f"Loaded {len(smiles_map)} SMILES from manifest")

    cifs = gather_cifs(args.root)
    print(f"Found {len(cifs)} CIFs under {args.root}")
    if args.limit:
        cifs = cifs[:args.limit]
        print(f"Limiting to first {len(cifs)}")

    # Resume support
    done = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row["target"], row["name"]))
        print(f"Resume: {len(done)} already in {args.output.name}")
        cifs = [c for c in cifs if (c[1], c[2]) not in done]
        print(f"Remaining: {len(cifs)}")

    if not cifs:
        print("Nothing to do.")
        return

    fieldnames = [
        "target", "name", "dG_bind_kcalmol", "ligand_strain_kcalmol",
        "E_complex", "E_protein", "E_ligand_bound", "E_ligand_free",
        "rmsd_min_A", "sg_cb_dist_A", "success_flag", "error",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    f_out = open(args.output, mode)
    w = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
    if mode == "w":
        w.writeheader()

    t0 = time.time()
    n_done = n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, c, t, n, smiles_map.get((t, n))) for c, t, n in cifs]
        for fut in as_completed(futures):
            res = fut.result()
            w.writerow(res)
            f_out.flush()
            n_done += 1
            if res["success_flag"]: n_ok += 1
            else: n_fail += 1
            if n_done % 5 == 0 or n_done == 1:
                elapsed = time.time() - t0
                rate = elapsed / n_done
                eta = (len(cifs) - n_done) * rate / 60
                print(f"[{n_done:4d}/{len(cifs)}] ok={n_ok} fail={n_fail} rate={rate:.1f}s/cofold ETA={eta:.0f}min", flush=True)
    f_out.close()
    print(f"\nDone. ok={n_ok} fail={n_fail}  total time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
