"""Batch PROPKA3 Cys346 pKa over the 997 Boltz cofolds.

Reads top1000_manifest__zap70_cys346.json for the (row_id, name) list, then
runs PROPKA3 in parallel on each <name>_model_0.pdb. Output CSV columns:
    row_id, name, pKa_Cys346, pKa_model, success_flag, error
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from propka_cys346_pka import compute_cys346_pka  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
PRED_DIR = (
    PROJECT_ROOT
    / "data"
    / "boltz_poses"
    / "boltz_results_top1000__zap70_cys346"
    / "predictions"
)
OUT_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_propka.csv"


def _work_one(item: tuple[int, str]) -> dict:
    row_id, name = item
    pdb = PRED_DIR / name / f"{name}_model_0.pdb"
    res = compute_cys346_pka(pdb)
    return {
        "row_id": row_id,
        "name": name,
        "pKa_Cys346": res["pKa_Cys346"],
        "pKa_model": res["pKa_model"],
        "success_flag": res["success_flag"],
        "error": res["error"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N items (debug)")
    ap.add_argument("--rows", type=str, default=None, help="Comma-separated row_ids to process (debug)")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"Manifest missing: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text())

    items: list[tuple[int, str]] = []
    for rid_str, m in manifest.items():
        name = m.get("yaml_name")
        if not name:
            continue
        items.append((int(rid_str), name))
    items.sort(key=lambda t: t[0])

    if args.rows:
        wanted = {int(x) for x in args.rows.split(",") if x.strip()}
        items = [it for it in items if it[0] in wanted]
    if args.limit:
        items = items[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running PROPKA3 on {len(items)} cofolds with {args.workers} workers")
    print(f"Output: {out_path}")
    t0 = time.time()
    results: list[dict] = []
    done = 0
    succ = 0
    last_log = t0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work_one, it): it for it in items}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                rid, name = futs[fut]
                r = {
                    "row_id": rid,
                    "name": name,
                    "pKa_Cys346": None,
                    "pKa_model": None,
                    "success_flag": 0,
                    "error": f"worker exception: {e}",
                }
            results.append(r)
            done += 1
            succ += int(r["success_flag"])
            now = time.time()
            if now - last_log > 30 or done == len(items):
                rate = done / (now - t0 + 1e-9)
                eta = (len(items) - done) / max(rate, 1e-9)
                print(
                    f"  done={done}/{len(items)} succ={succ} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )
                last_log = now

    results.sort(key=lambda d: d["row_id"])
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["row_id", "name", "pKa_Cys346", "pKa_model", "success_flag", "error"],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    pkas = [r["pKa_Cys346"] for r in results if r["pKa_Cys346"] is not None]
    cov = succ / max(len(results), 1) * 100
    print(f"\nWrote {out_path}: {len(results)} rows, success {succ} ({cov:.1f}%)")
    if pkas:
        pkas_s = sorted(pkas)
        n = len(pkas_s)

        def q(p):
            return pkas_s[min(int(p * n), n - 1)]

        mean = sum(pkas_s) / n
        print(
            "pKa_Cys346 stats: "
            f"mean={mean:.3f} p10={q(0.10):.3f} p25={q(0.25):.3f} "
            f"p50={q(0.50):.3f} p75={q(0.75):.3f} p90={q(0.90):.3f} "
            f"min={pkas_s[0]:.3f} max={pkas_s[-1]:.3f}"
        )
    print(f"Wall time: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
