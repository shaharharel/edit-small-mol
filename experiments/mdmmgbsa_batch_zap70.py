"""Batch driver: MD MM-GBSA over ZAP70 Cys346 cofolds.

Reads top1000_manifest__zap70_cys346.json (with combined_score), sorts
descending, walks the top-N requested. For each ligand:
  - locates ~/data/cys346_predictions/<name>/<name>_model_0.pdb
  - calls score_one_md (Cheng JCTC 2017 protocol)
  - writes one CSV row to the OUTPUT csv before moving on (durability)
  - handles SIGINT/SIGTERM cleanly

Per-ligand wall-clock timeout enforced via multiprocessing isolation.

Usage:
    python mdmmgbsa_batch_zap70.py --top 100 --md_ns 1.0
    nohup python mdmmgbsa_batch_zap70.py --top 100 > ~/results/mdmmgbsa.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, csv, time, signal, sys, os, gc, traceback, multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np

# Local import (file lives next to mdmmgbsa_cheng2017.py)
sys.path.insert(0, str(Path(__file__).parent))
from mdmmgbsa_cheng2017 import score_one_md


_CSV_COLS = [
    "row_id", "target", "name", "smiles", "combined_score",
    "dG_bind_md_kcalmol", "dG_bind_md_std",
    "ligand_strain_md_kcalmol", "n_frames_scored",
    "sg_cb_dist_md_mean", "sg_cb_dist_md_std",
    "md_ns", "wall_md_s", "wall_gbsa_s", "wall_total_s",
    "success_flag", "error", "timestamp",
]


_STOP = False
def _signal_handler(signum, frame):
    global _STOP
    _STOP = True
    print(f"[signal] {signum} received — finishing current ligand, then exiting", flush=True)


def _worker(args):
    """Subprocess worker: runs one score_one_md and writes result via queue."""
    (pdb_path, smiles, target, name, md_ns, n_frames, equil_ps, platform_name) = args
    try:
        res = score_one_md(
            Path(pdb_path), smiles, target, name,
            md_ns=md_ns, n_frames=n_frames, equil_ps=equil_ps,
            platform_name=platform_name,
        )
        return res
    except Exception as e:
        return {
            "target": target, "name": name,
            "dG_bind_md_kcalmol": None, "dG_bind_md_std": None,
            "ligand_strain_md_kcalmol": None, "n_frames_scored": 0,
            "sg_cb_dist_md_mean": None, "sg_cb_dist_md_std": None,
            "md_ns": md_ns,
            "wall_md_s": None, "wall_gbsa_s": None, "wall_total_s": None,
            "success_flag": 0,
            "error": f"WORKER_EXCEPTION: {type(e).__name__}: {str(e)[:300]}",
            "traceback": traceback.format_exc()[-500:],
        }


def run_one_with_timeout(pdb_path, smiles, target, name,
                         md_ns: float, n_frames: int, equil_ps: float,
                         platform_name: str, timeout_s: int) -> dict:
    """Run score_one_md in a spawned subprocess so we can hard-kill on timeout."""
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)
    try:
        ar = pool.apply_async(
            _worker,
            args=((str(pdb_path), smiles, target, name, md_ns, n_frames, equil_ps, platform_name),),
        )
        try:
            return ar.get(timeout=timeout_s)
        except mp.TimeoutError:
            pool.terminate()
            return {
                "target": target, "name": name,
                "dG_bind_md_kcalmol": None, "dG_bind_md_std": None,
                "ligand_strain_md_kcalmol": None, "n_frames_scored": 0,
                "sg_cb_dist_md_mean": None, "sg_cb_dist_md_std": None,
                "md_ns": md_ns,
                "wall_md_s": None, "wall_gbsa_s": None, "wall_total_s": float(timeout_s),
                "success_flag": 0,
                "error": f"TIMEOUT_{timeout_s}s",
            }
    finally:
        pool.close()
        pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(Path.home() / "data" / "top1000_manifest__zap70_cys346.json"))
    ap.add_argument("--pdb_root", type=str, default=str(Path.home() / "data" / "cys346_predictions"))
    ap.add_argument("--out_csv", type=str, default=str(Path.home() / "results" / "zap70_cys346_mdmmgbsa.csv"))
    ap.add_argument("--top", type=int, default=100, help="ligands to process (by combined_score)")
    ap.add_argument("--start", type=int, default=0, help="skip first N (for resume after top batch)")
    ap.add_argument("--md_ns", type=float, default=1.0)
    ap.add_argument("--equil_ps", type=float, default=50.0)
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--platform", type=str, default="CUDA")
    ap.add_argument("--timeout_min", type=int, default=60)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest if isinstance(manifest, list) else manifest.get("rows", [])
    if not rows:
        print("ERROR: empty manifest", flush=True); sys.exit(1)
    rows_sorted = sorted(rows, key=lambda r: r.get("combined_score", 0.0), reverse=True)
    selected = rows_sorted[args.start:args.start + args.top]
    print(f"[batch] processing {len(selected)} ligands from rank {args.start} (top by combined_score)", flush=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    # Determine already-processed names (resume safety)
    done_names = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as fh:
            r = csv.DictReader(fh)
            for row in r:
                done_names.add(row.get("name"))
        print(f"[batch] {len(done_names)} ligands already in {out_path} — skipping", flush=True)

    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        if write_header:
            writer.writeheader()
            fh.flush()

        t_batch0 = time.time()
        for i, row in enumerate(selected):
            global _STOP
            if _STOP:
                print("[batch] stop signal received — exiting", flush=True)
                break
            name = row.get("name") or row.get("ligand_name") or f"row_{i}"
            if name in done_names:
                print(f"[batch] [{i+1}/{len(selected)}] {name} SKIP (already in csv)", flush=True)
                continue
            smiles = row.get("smiles")
            combined = row.get("combined_score", 0.0)
            row_id = row.get("row_id", i)
            pdb = Path(args.pdb_root) / name / f"{name}_model_0.pdb"
            if not pdb.exists():
                print(f"[batch] [{i+1}/{len(selected)}] {name} MISSING_PDB", flush=True)
                writer.writerow({
                    "row_id": row_id, "target": "ZAP70_C346", "name": name,
                    "smiles": smiles, "combined_score": combined,
                    "dG_bind_md_kcalmol": None, "dG_bind_md_std": None,
                    "ligand_strain_md_kcalmol": None, "n_frames_scored": 0,
                    "sg_cb_dist_md_mean": None, "sg_cb_dist_md_std": None,
                    "md_ns": args.md_ns,
                    "wall_md_s": None, "wall_gbsa_s": None, "wall_total_s": None,
                    "success_flag": 0, "error": "MISSING_PDB",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                fh.flush()
                continue

            t0 = time.time()
            res = run_one_with_timeout(
                pdb, smiles, "ZAP70_C346", name,
                md_ns=args.md_ns, n_frames=args.n_frames, equil_ps=args.equil_ps,
                platform_name=args.platform, timeout_s=args.timeout_min * 60,
            )
            elapsed = time.time() - t0
            dG = res.get("dG_bind_md_kcalmol")
            std = res.get("dG_bind_md_std")
            print(
                f"[batch] [{i+1}/{len(selected)}] {name} "
                f"dG_bind={'NA' if dG is None else f'{dG:+.2f}'} "
                f"std={'NA' if std is None else f'{std:.2f}'} "
                f"frames={res.get('n_frames_scored',0)} "
                f"elapsed={elapsed:.0f}s "
                f"err={res.get('error') or ''}",
                flush=True,
            )

            writer.writerow({
                "row_id": row_id, "target": "ZAP70_C346", "name": name,
                "smiles": smiles, "combined_score": combined,
                "dG_bind_md_kcalmol": res.get("dG_bind_md_kcalmol"),
                "dG_bind_md_std": res.get("dG_bind_md_std"),
                "ligand_strain_md_kcalmol": res.get("ligand_strain_md_kcalmol"),
                "n_frames_scored": res.get("n_frames_scored"),
                "sg_cb_dist_md_mean": res.get("sg_cb_dist_md_mean"),
                "sg_cb_dist_md_std": res.get("sg_cb_dist_md_std"),
                "md_ns": res.get("md_ns"),
                "wall_md_s": res.get("wall_md_s"),
                "wall_gbsa_s": res.get("wall_gbsa_s"),
                "wall_total_s": res.get("wall_total_s"),
                "success_flag": res.get("success_flag"),
                "error": res.get("error"),
                "timestamp": datetime.utcnow().isoformat(),
            })
            fh.flush()
            gc.collect()

        dt = time.time() - t_batch0
        print(f"[batch] DONE: {len(selected)} attempts in {dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
