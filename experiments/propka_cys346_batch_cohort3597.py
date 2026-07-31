"""Batch PROPKA3 Cys346 pKa over the cohort 3,597 cofold CIFs.

Reads `data/boltz_results/cohort_3597_full/from_*/<row_id>/<row_id>_model_0.cif`,
converts each CIF to PDB in a per-worker tempdir, then runs PROPKA3.

Output CSV columns:
    row_id, pKa_Cys346, pKa_model, success_flag, error

Resumable: skips row_ids already present in the output CSV with success_flag=1.

Usage:
    PATH=/opt/miniconda3/envs/quris/bin:$PATH python experiments/propka_cys346_batch_cohort3597.py \
        --workers 14 --out data/tier4_scored/propka_cys346_3597_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from propka_cys346_pka import compute_cys346_pka  # noqa: E402

COHORT3597_ROOT = PROJECT_ROOT / "data/boltz_results/cohort_3597_full"
DEFAULT_OUT = PROJECT_ROOT / "data/tier4_scored/propka_cys346_3597_v2.csv"


def _convert_cif(cif_path: Path, pdb_path: Path) -> None:
    """Lazy import gemmi so we don't pay the cost in the parent process."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    # Rename Boltz's LIG1 -> LIG so propka treats it as a ligand HETATM.
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "LIG1":
                    res.name = "LIG"
                    res.het_flag = "H"
                    res.entity_type = gemmi.EntityType.NonPolymer
    st.write_pdb(str(pdb_path))


def _worker(item: tuple[int, str]) -> dict:
    row_id, cif_path = item
    cif_p = Path(cif_path)
    out = {"row_id": int(row_id), "pKa_Cys346": None, "pKa_model": None,
           "success_flag": 0, "error": ""}
    # First, look for a sibling PDB (saved by an earlier step).
    pdb_sib = cif_p.with_suffix(".pdb")
    try:
        if pdb_sib.exists():
            res = compute_cys346_pka(pdb_sib)
        else:
            tmpdir = Path(tempfile.mkdtemp(prefix="propka_cof_"))
            try:
                pdb_tmp = tmpdir / (cif_p.stem + ".pdb")
                _convert_cif(cif_p, pdb_tmp)
                res = compute_cys346_pka(pdb_tmp)
            finally:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        out.update({
            "pKa_Cys346": res["pKa_Cys346"],
            "pKa_model": res["pKa_model"],
            "success_flag": res["success_flag"],
            "error": res["error"],
        })
    except Exception as e:
        out["error"] = f"worker exception: {type(e).__name__}: {e}"
    return out


def _discover_tasks() -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    seen: set[int] = set()
    for from_dir in sorted(COHORT3597_ROOT.glob("from_*")):
        if not from_dir.is_dir():
            continue
        for sub in sorted(from_dir.iterdir()):
            if not sub.is_dir() or not sub.name.isdigit():
                continue
            rid = int(sub.name)
            if rid in seen:
                continue
            cif = sub / f"{rid}_model_0.cif"
            if cif.exists():
                tasks.append((rid, str(cif)))
                seen.add(rid)
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tasks = _discover_tasks()
    print(f"Discovered {len(tasks)} CIF tasks", flush=True)

    out_path = Path(args.out)
    done_ids: set[int] = set()
    if out_path.exists():
        with out_path.open() as fh:
            r = csv.DictReader(fh)
            for row in r:
                if row.get("success_flag") == "1":
                    try:
                        done_ids.add(int(row["row_id"]))
                    except Exception:
                        pass
        print(f"Resume: {len(done_ids)} rows already done; will skip", flush=True)
        tasks = [t for t in tasks if t[0] not in done_ids]
        print(f"Remaining: {len(tasks)}", flush=True)

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"Limit applied: {len(tasks)}", flush=True)

    if not tasks:
        print("Nothing to do", flush=True)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "pKa_Cys346", "pKa_model", "success_flag", "error"]
    # Preserve prior rows
    prior_rows: list[dict] = []
    if out_path.exists():
        with out_path.open() as fh:
            for r in csv.DictReader(fh):
                prior_rows.append({k: r.get(k) for k in fields})

    print(f"Running PROPKA3 on {len(tasks)} cofolds with {args.workers} workers -> {out_path}", flush=True)
    t0 = time.time()
    n_ok = sum(1 for r in prior_rows if r.get("success_flag") == "1")
    n_err = 0
    new_rows: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        last_log = t0
        last_flush = t0
        for i, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            new_rows.append({k: res.get(k) for k in fields})
            if res["success_flag"] == 1:
                n_ok += 1
            else:
                n_err += 1
            now = time.time()
            if now - last_log > 20 or i == len(tasks):
                rate = i / max(now - t0, 1e-9)
                eta = (len(tasks) - i) / max(rate, 1e-9) / 60
                print(f"  done={i}/{len(tasks)} succ={n_ok} err={n_err} rate={rate:.2f}/s eta={eta:.1f}min", flush=True)
                last_log = now
            if now - last_flush > 60:
                # partial flush
                with out_path.open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    w.writeheader()
                    for r in prior_rows + new_rows:
                        w.writerow(r)
                last_flush = now

    # Final write
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in prior_rows + new_rows:
            w.writerow(r)
    print(f"\nWrote {out_path}  ({len(prior_rows) + len(new_rows)} rows total, "
          f"new ok={n_ok - sum(1 for r in prior_rows if r.get('success_flag') == '1')}, "
          f"new err={n_err})", flush=True)
    print(f"Wall: {(time.time() - t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    # Make sure propka3 binary is on PATH
    propka = "/opt/miniconda3/envs/quris/bin/propka3"
    if Path(propka).exists():
        os.environ["PATH"] = "/opt/miniconda3/envs/quris/bin:" + os.environ.get("PATH", "")
    mp.freeze_support()
    raise SystemExit(main())
