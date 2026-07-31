"""Score the ZAP70 Cys346 top-1000 cofolds with MM-GBSA (ΔG_bind + ligand_strain).

Reads:
  data/boltz_poses/top1000_manifest__zap70_cys346.json   (row_id → yaml_name, smiles, …)
  data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/<yaml_name>/<yaml_name>_model_0.cif

Writes:
  data/boltz_poses/zap70_cys346_energy_scores.csv  (target,name,row_id,smiles,dG_bind_kcalmol,ligand_strain_kcalmol,…)

Reuses `score_cofold_batch.process_one` for the CIF→PDB→param→energy pipeline.

Usage:
  python score_zap70_top1000_energy.py            # full 997-mol run
  python score_zap70_top1000_energy.py --limit 5  # smoke test (sequential)
  python score_zap70_top1000_energy.py --workers 6 --resume
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_cofold_batch import process_one  # already handles cif→pdb→param→energy

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_JSON = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
COFOLD_ROOT = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
OUTPUT_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_energy_scores.csv"
TARGET_NAME = "ZAP70_Cys346"

FIELDNAMES = [
    "target", "name", "row_id", "smiles",
    "dG_bind_kcalmol", "ligand_strain_kcalmol",
    "E_complex", "E_protein", "E_ligand_bound", "E_ligand_free",
    "rmsd_min_A", "sg_cb_dist_A", "success_flag", "error",
]


def load_jobs() -> list[tuple[str, str, str, int]]:
    """Return [(cif_path, name, smiles, row_id), ...] for the top1000 cofolds."""
    if not MANIFEST_JSON.exists():
        sys.exit(f"manifest missing: {MANIFEST_JSON}")
    m = json.loads(MANIFEST_JSON.read_text())
    jobs = []
    for rid_str, entry in m.items():
        name = entry["yaml_name"]
        smi = entry["smiles"]
        cif = COFOLD_ROOT / name / f"{name}_model_0.cif"
        if cif.exists():
            jobs.append((str(cif), name, smi, int(rid_str)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    jobs = load_jobs()
    print(f"Found {len(jobs)} cofold CIFs (manifest has {len(json.loads(MANIFEST_JSON.read_text()))} entries)")
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"Limited to first {len(jobs)}")

    done = set()
    if args.resume and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add(row["name"])
        jobs = [j for j in jobs if j[1] not in done]
        print(f"Resume: {len(done)} already scored, {len(jobs)} remaining")

    if not jobs:
        print("nothing to do")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and OUTPUT_CSV.exists() else "w"
    f_out = open(OUTPUT_CSV, mode)
    w = csv.DictWriter(f_out, fieldnames=FIELDNAMES, extrasaction="ignore")
    if mode == "w":
        w.writeheader()

    t0 = time.time()
    n_done = n_ok = n_fail = 0

    def submit(cif, name, smi, rid):
        # process_one(cif_path, target, name, smiles) — defined in score_cofold_batch
        return process_one(cif, TARGET_NAME, name, smi), rid

    if args.workers == 1 or args.limit and args.limit <= 5:
        # Sequential — easier to read smoke-test logs
        for cif, name, smi, rid in jobs:
            res, _rid = submit(cif, name, smi, rid)
            res["row_id"] = rid
            res["smiles"] = smi
            w.writerow(res)
            f_out.flush()
            n_done += 1
            if res["success_flag"]:
                n_ok += 1
            else:
                n_fail += 1
            elapsed = time.time() - t0
            print(f"[{n_done:4d}/{len(jobs)}] {name[:60]:60s} ok={res['success_flag']} dG={res.get('dG_bind_kcalmol')} strain={res.get('ligand_strain_kcalmol')} {elapsed:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, c, TARGET_NAME, n, s): (n, s, r) for c, n, s, r in jobs}
            for fut in as_completed(futures):
                name, smi, rid = futures[fut]
                res = fut.result()
                res["row_id"] = rid
                res["smiles"] = smi
                w.writerow(res)
                f_out.flush()
                n_done += 1
                if res["success_flag"]:
                    n_ok += 1
                else:
                    n_fail += 1
                if n_done % 5 == 0 or n_done == 1:
                    elapsed = time.time() - t0
                    rate = elapsed / n_done
                    eta_h = (len(jobs) - n_done) * rate / 3600
                    print(f"[{n_done:4d}/{len(jobs)}] ok={n_ok} fail={n_fail} rate={rate:.0f}s/mol ETA={eta_h:.1f}h", flush=True)

    f_out.close()
    print(f"\ndone. ok={n_ok} fail={n_fail}  total={time.time()-t0:.0f}s  → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
