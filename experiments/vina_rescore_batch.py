"""Batch AutoDock Vina rescore over all Boltz cofold poses (ZAP70 Cys346).

Runs `experiments/vina_rescore_cofolds.rescore_cofold` for every cofold dir in
  data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/
with N workers (default 8) and writes
  data/boltz_poses/zap70_cys346_vina_rescore.csv
with columns:
  row_id, name, vina_kcalmol, vina_inter_kcalmol, vina_intra_kcalmol,
  vina_torsion_kcalmol, vina_unbound_kcalmol, success_flag, error,
  sg_x, sg_y, sg_z, box_center_x, box_center_y, box_center_z,
  box_size_x, box_size_y, box_size_z

row_id is inferred from the cofold name (the suffix integer after the last
underscore in <name>, which matches the row_id in the products / Tier 2 CSV).

Resumable: rows already in the CSV are skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vina_rescore_cofolds import rescore_cofold

PRED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "boltz_poses"
    / "boltz_results_top1000__zap70_cys346"
    / "predictions"
)
OUT_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_vina_rescore.csv"

FIELDS = [
    "row_id",
    "name",
    "vina_kcalmol",
    "vina_inter_kcalmol",
    "vina_intra_kcalmol",
    "vina_torsion_kcalmol",
    "vina_unbound_kcalmol",
    "success_flag",
    "error",
    "sg_x",
    "sg_y",
    "sg_z",
    "box_center_x",
    "box_center_y",
    "box_center_z",
    "box_size_x",
    "box_size_y",
    "box_size_z",
]


_ROW_ID_RE = re.compile(r"_(\d+)$")


def name_to_row_id(name: str) -> int | None:
    m = _ROW_ID_RE.search(name)
    return int(m.group(1)) if m else None


def _worker(cofold_dir: str) -> dict:
    path = Path(cofold_dir)
    try:
        res = rescore_cofold(path)
    except Exception as e:
        res = {
            "name": path.name,
            "success_flag": 0,
            "error": f"worker_uncaught: {type(e).__name__}: {e}",
            "vina_kcalmol": None,
            "vina_inter_kcalmol": None,
            "vina_intra_kcalmol": None,
        }
    rid = name_to_row_id(res.get("name", path.name))
    res["row_id"] = rid
    return res


def _normalize_row(d: dict) -> dict:
    return {k: d.get(k) for k in FIELDS}


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success_flag") == "1":
                done.add(row["name"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=None, help="Test mode: process at most N cofolds")
    ap.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated explicit cofold dir names (overrides discovery)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run even if a successful row already exists in the CSV",
    )
    args = ap.parse_args()

    if args.names:
        dirs = [PRED_ROOT / n for n in args.names.split(",") if n.strip()]
    else:
        dirs = sorted(p for p in PRED_ROOT.iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else load_done(args.out)
    todo = [d for d in dirs if d.name not in done]
    print(
        f"Total cofolds: {len(dirs)}  "
        f"already done: {len(dirs) - len(todo)}  "
        f"to process: {len(todo)}  "
        f"workers: {args.workers}",
        flush=True,
    )
    if not todo:
        print("Nothing to do.")
        return 0

    # Open CSV in append mode; write header if file new/empty.
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    fout = open(args.out, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        fout.flush()

    t0 = time.time()
    n_ok = 0
    n_fail = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = {exc.submit(_worker, str(d)): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "name": d.name,
                    "row_id": name_to_row_id(d.name),
                    "success_flag": 0,
                    "error": f"future_uncaught: {type(e).__name__}: {e}",
                    "vina_kcalmol": None,
                    "vina_inter_kcalmol": None,
                    "vina_intra_kcalmol": None,
                }
            writer.writerow(_normalize_row(res))
            fout.flush()
            if res.get("success_flag") == 1:
                n_ok += 1
            else:
                n_fail += 1
            n_done += 1
            if n_done % 20 == 0 or n_done == len(todo):
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print(
                    f"  [{n_done}/{len(todo)}] ok={n_ok} fail={n_fail}  "
                    f"{rate:.2f} cof/s  ETA {eta/60:.1f} min",
                    flush=True,
                )

    fout.close()
    print(
        f"DONE  ok={n_ok} fail={n_fail}  "
        f"total elapsed {(time.time()-t0)/60:.1f} min  -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
