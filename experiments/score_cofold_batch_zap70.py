"""Batch-run the (bug-fixed) MM-GBSA scorer over all 997 ZAP70-Cys346 cofolds.

Writes incrementally to data/boltz_poses/zap70_cys346_energy_scores_v2.csv
(the original v1 is left untouched). Uses multiprocessing because antechamber
/parmchk2/tleap are subprocess-bound. On failure, the row is still written
with success_flag=0 and the error string.

Usage:
    python experiments/score_cofold_batch_zap70.py [--n-workers N] [--limit K]
        [--row-ids 3889,4019,...]

After a clean run, ENABLE_ENERGY_MERGE=True in backend.py can be flipped on.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from score_cofold_energy import score_one  # noqa: E402

MANIFEST = REPO / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
PRED_ROOT = REPO / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
OUT_CSV = REPO / "data" / "boltz_poses" / "zap70_cys346_energy_scores_v2.csv"

FIELDS = [
    "target", "name", "row_id", "smiles",
    "dG_bind_kcalmol", "ligand_strain_kcalmol",
    "E_complex", "E_protein", "E_ligand_bound", "E_ligand_free",
    "rmsd_min_A", "sg_cb_dist_A",
    "E_complex_raw", "E_cov_restraint", "E_far_restraint", "n_near_residues",
    "combined_score",
    "elapsed_s", "success_flag", "error",
]


def _resolve_pdb(yaml_name: str) -> Path:
    """Locate the model_0.pdb under predictions/<yaml_name>/."""
    pdir = PRED_ROOT / yaml_name
    candidates = [
        pdir / f"{yaml_name}_model_0.pdb",
        pdir / f"{yaml_name}.pdb",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: any PDB under the directory.
    for p in pdir.glob("*_model_*.pdb"):
        return p
    raise FileNotFoundError(f"no PDB under {pdir}")


def _score_one_worker(args):
    row_id, yaml_name, smiles, combined_score = args
    t0 = time.time()
    try:
        pdb = _resolve_pdb(yaml_name)
        res = score_one(pdb, target="ZAP70_Cys346", name=yaml_name, smiles=smiles)
    except Exception as e:
        res = {
            "target": "ZAP70_Cys346", "name": yaml_name,
            "success_flag": 0, "error": f"{type(e).__name__}: {str(e)[:300]}",
        }
    elapsed = time.time() - t0
    row = {
        "target": res.get("target"),
        "name": res.get("name"),
        "row_id": row_id,
        "smiles": smiles,
        "dG_bind_kcalmol": res.get("dG_bind_kcalmol"),
        "ligand_strain_kcalmol": res.get("ligand_strain_kcalmol"),
        "E_complex": res.get("E_complex"),
        "E_protein": res.get("E_protein"),
        "E_ligand_bound": res.get("E_ligand_bound"),
        "E_ligand_free": res.get("E_ligand_free"),
        "rmsd_min_A": res.get("rmsd_min_A"),
        "sg_cb_dist_A": res.get("sg_cb_dist_A"),
        "E_complex_raw": res.get("_E_complex_raw"),
        "E_cov_restraint": res.get("_E_cov_restraint"),
        "E_far_restraint": res.get("_E_far_restraint"),
        "n_near_residues": res.get("_n_near_residues"),
        "combined_score": combined_score,
        "elapsed_s": round(elapsed, 1),
        "success_flag": int(bool(res.get("success_flag"))),
        "error": (res.get("error") or "")[:280],
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score the first K cofolds (for smoke testing)")
    ap.add_argument("--row-ids", type=str, default=None,
                    help="Comma-separated row_ids to score (overrides --limit)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present in v2 CSV")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    items = []
    for rid_str, entry in manifest.items():
        items.append((
            int(rid_str),
            entry["yaml_name"],
            entry["smiles"],
            entry.get("combined_score"),
        ))
    items.sort(key=lambda x: x[0])

    if args.row_ids:
        wanted = {int(s) for s in args.row_ids.split(",")}
        items = [it for it in items if it[0] in wanted]
    if args.limit:
        items = items[: args.limit]

    done_ids = set()
    if args.resume and OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                try:
                    done_ids.add(int(r["row_id"]))
                except Exception:
                    pass
        items = [it for it in items if it[0] not in done_ids]
        print(f"Resume: {len(done_ids)} already done, {len(items)} to go")

    print(f"Scoring {len(items)} cofolds with {args.n_workers} workers")
    print(f"Output: {OUT_CSV}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.resume and OUT_CSV.exists())
    open_mode = "a" if (args.resume and OUT_CSV.exists()) else "w"
    f = open(OUT_CSV, open_mode, newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader(); f.flush()

    t0 = time.time()
    n_ok = 0; n_fail = 0
    try:
        with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
            futures = {exe.submit(_score_one_worker, it): it for it in items}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    row = fut.result()
                except Exception as e:
                    it = futures[fut]
                    row = {
                        "target": "ZAP70_Cys346", "name": it[1], "row_id": it[0], "smiles": it[2],
                        "combined_score": it[3], "elapsed_s": -1,
                        "success_flag": 0, "error": f"WorkerCrash: {type(e).__name__}: {str(e)[:200]}",
                    }
                writer.writerow(row); f.flush()
                if row.get("success_flag") == 1:
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 25 == 0 or i == len(items):
                    rate = i / max(time.time() - t0, 1e-6)
                    eta = (len(items) - i) / max(rate, 1e-6)
                    dG = row.get("dG_bind_kcalmol")
                    dG_str = f"{dG:+.1f}" if isinstance(dG, (int, float)) else "—"
                    print(f"[{i:4d}/{len(items)}] ok={n_ok} fail={n_fail} "
                          f"rate={rate:.2f}/s eta={eta/60:.1f}min "
                          f"last={row['name'][:30]:30s} dG={dG_str}")
    finally:
        f.close()

    print(f"\nDone. ok={n_ok} fail={n_fail}  out={OUT_CSV}")


if __name__ == "__main__":
    main()
