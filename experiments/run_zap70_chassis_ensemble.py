"""Phase B + C orchestrator — runs smokes (gennums=15) for each ZAP70 chassis
and dispatches N=500 sampling for the ones that pass.

To respect CPU contention with already-running H2/C5 jobs, this script runs
chassis SEQUENTIALLY (one at a time) by default. Use --parallel N to run up to
N concurrent samplers.

Usage:
    # Sequential smoke for all 6 chassis (recommended on Mac CPU when other
    # samplers are running):
    python experiments/run_zap70_chassis_ensemble.py --phase smoke

    # After smokes pass, dispatch N=500 sequentially:
    python experiments/run_zap70_chassis_ensemble.py --phase n500
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHASSIS_IDS = ["ZAP_C1", "ZAP_C2", "ZAP_C3", "ZAP_C4", "ZAP_C5", "ZAP_C6"]


def run_one(chassis_id: str, gennums: int, min_acceptable: int,
            max_run_seconds: int, out_subdir: str, tempture: float = 1.0):
    out_dir = ROOT / "data" / "lingo3dmol_zap70_chassis" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf = out_dir / "samples.sdf"
    log = out_dir / "stdout.log"
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_lingo3dmol_l2_extended_anchor.py"),
        "--pocket_pdb", "data/lingo3dmol_smoke/zap70_pocket_cys346.pdb",
        "--anchor_id", chassis_id,
        "--output", str(sdf),
        "--tempture", str(tempture),
        "--gennums", str(gennums),
        "--min_acceptable", str(min_acceptable),
        "--max_run_seconds", str(max_run_seconds),
    ]
    print(f"[ensemble] [{chassis_id}] running -> {sdf}")
    print(f"[ensemble] [{chassis_id}] log -> {log}")
    print(f"[ensemble] cmd: {' '.join(cmd)}")
    t0 = time.time()
    with log.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"[ensemble] [{chassis_id}] finished in {elapsed:.1f}s rc={proc.returncode}")
    summary_path = out_dir / "samples_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    return {
        "chassis_id": chassis_id,
        "out_sdf": str(sdf),
        "log": str(log),
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "n_sampled": summary.get("n_sampled", 0),
        "status": summary.get("status", "unknown"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "n500"], required=True)
    ap.add_argument("--chassis_ids", nargs="+", default=CHASSIS_IDS)
    ap.add_argument("--gennums_smoke", type=int, default=15)
    ap.add_argument("--min_smoke", type=int, default=5)
    ap.add_argument("--max_smoke_sec", type=int, default=900)
    ap.add_argument("--gennums_full", type=int, default=500)
    ap.add_argument("--min_full", type=int, default=200)
    ap.add_argument("--max_full_sec", type=int, default=14400)
    args = ap.parse_args()

    out_report_path = (
        ROOT / "data" / "lingo3dmol_zap70_chassis"
        / f"ensemble_{args.phase}_report.json"
    )
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for cid in args.chassis_ids:
        if args.phase == "smoke":
            r = run_one(cid, args.gennums_smoke, args.min_smoke,
                        args.max_smoke_sec, out_subdir=f"{cid}_smoke")
        else:
            r = run_one(cid, args.gennums_full, args.min_full,
                        args.max_full_sec, out_subdir=cid)
        results.append(r)
        # incremental save
        out_report_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"[ensemble] cumulative results: {len(results)} chassis processed")

    print("\n=== Phase summary ===")
    for r in results:
        print(f"  {r['chassis_id']:>8}  n_sampled={r['n_sampled']:>4}  "
              f"status={r['status']:<14}  rc={r['returncode']}  "
              f"elapsed={r['elapsed_sec']:.0f}s")
    print(f"\nWrote {out_report_path}")


if __name__ == "__main__":
    main()
