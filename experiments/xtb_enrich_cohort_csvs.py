#!/usr/bin/env python3
"""Enrich cohort scored CSVs with xTB GFN2 warhead reactivity descriptors.

Adds 8 columns: LUMO_eV, HOMO_eV, gap_eV, omega_eV, q_Cb, fukui_plus_Cb,
pred_log_k2_GSH, xtb_status.

Only processes rows where warhead_intact==True; non-acrylamide rows get NaN
descriptors and xtb_status='skipped_nonacryl'. Failed xTB runs get
xtb_status='failed' and NaN descriptors.

Usage:
    python experiments/xtb_enrich_cohort_csvs.py \
        --csv data/tier4_scored/foo_scored.csv \
        [--csv data/tier4_scored/bar_scored.csv ...] \
        --workers 28
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.xtb_warhead_electrophilicity import compute_warhead_descriptors  # noqa: E402

XTB_COLS = [
    "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV",
    "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH", "xtb_status",
]


def _worker(job: Tuple[int, str]) -> Tuple[int, Dict[str, float]]:
    idx, smi = job
    try:
        rec = compute_warhead_descriptors(smi)
        ok = bool(rec.get("success_flag", 0))
        out = {
            "LUMO_eV": rec["LUMO_eV"],
            "HOMO_eV": rec["HOMO_eV"],
            "gap_eV": rec["gap_eV"],
            "omega_eV": rec["omega_eV"],
            "q_Cb": rec["q_Cb"],
            "fukui_plus_Cb": rec["fukui_plus_Cb"],
            "pred_log_k2_GSH": rec["pred_log_k2_GSH"],
            "xtb_status": "ok" if ok else "failed",
        }
    except Exception as e:
        out = {c: np.nan for c in XTB_COLS if c != "xtb_status"}
        out["xtb_status"] = "failed"
    return idx, out


def enrich(csv_path: Path, workers: int) -> Dict[str, float]:
    print(f"\n[{csv_path.name}] loading…", flush=True)
    df = pd.read_csv(csv_path)
    n = len(df)
    assert "smiles" in df.columns, f"no smiles col in {csv_path}"
    assert "warhead_intact" in df.columns, f"no warhead_intact col in {csv_path}"

    # Initialize xTB columns (or leave existing ones — we overwrite)
    for c in XTB_COLS:
        if c == "xtb_status":
            df[c] = "skipped_nonacryl"
        else:
            df[c] = np.nan

    acryl_mask = df["warhead_intact"] == True  # noqa: E712
    acryl_idx = df.index[acryl_mask].tolist()
    jobs = [(int(i), str(df.at[i, "smiles"])) for i in acryl_idx]
    n_jobs = len(jobs)
    print(f"[{csv_path.name}] rows={n}, acrylamide={n_jobs}, workers={workers}", flush=True)

    if n_jobs == 0:
        df.to_csv(csv_path, index=False)
        return {"rows": n, "acryl": 0, "ok": 0, "failed": 0, "elapsed_min": 0.0}

    t0 = time.time()
    results: Dict[int, Dict[str, float]] = {}
    flush_every = max(500, n_jobs // 50)

    with Pool(workers) as pool:
        for i, (idx, rec) in enumerate(
            pool.imap_unordered(_worker, jobs, chunksize=4), 1
        ):
            results[idx] = rec
            if i % flush_every == 0 or i == n_jobs:
                el = time.time() - t0
                eta = (el / i) * (n_jobs - i) / 60
                ok_so_far = sum(1 for v in results.values() if v["xtb_status"] == "ok")
                print(
                    f"[{csv_path.name}] [{i}/{n_jobs}] ok={ok_so_far} "
                    f"fail={i-ok_so_far}  elapsed={el/60:.1f}m  eta={eta:.1f}m",
                    flush=True,
                )

    # Write back
    for idx, rec in results.items():
        for c, v in rec.items():
            df.at[idx, c] = v

    df.to_csv(csv_path, index=False)
    el = time.time() - t0

    n_ok = int((df["xtb_status"] == "ok").sum())
    n_fail = int((df["xtb_status"] == "failed").sum())
    n_skip = int((df["xtb_status"] == "skipped_nonacryl").sum())
    lumo_med = float(df.loc[df["xtb_status"] == "ok", "LUMO_eV"].median())
    logk_med = float(df.loc[df["xtb_status"] == "ok", "pred_log_k2_GSH"].median())
    print(
        f"[{csv_path.name}] DONE rows={n} acryl={n_jobs} ok={n_ok} "
        f"failed={n_fail} skipped={n_skip} elapsed={el/60:.1f}m  "
        f"median LUMO={lumo_med:.3f} eV  median pred_log_k2={logk_med:.3f}",
        flush=True,
    )
    return {
        "rows": n, "acryl": n_jobs, "ok": n_ok, "failed": n_fail,
        "elapsed_min": el / 60.0, "median_LUMO_eV": lumo_med,
        "median_pred_log_k2_GSH": logk_med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="Path to scored CSV (repeat for multiple)")
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()

    overall_t0 = time.time()
    summaries = {}
    for csv_path in args.csv:
        p = Path(csv_path)
        if not p.exists():
            print(f"[skip] {p} does not exist", flush=True)
            continue
        summaries[p.name] = enrich(p, args.workers)

    total = time.time() - overall_t0
    print("\n========== SUMMARY ==========")
    for name, s in summaries.items():
        print(f"{name}: rows={s['rows']} acryl={s['acryl']} ok={s['ok']} "
              f"failed={s['failed']} elapsed={s['elapsed_min']:.1f}m")
    print(f"TOTAL elapsed: {total/60:.1f} min")


if __name__ == "__main__":
    main()
