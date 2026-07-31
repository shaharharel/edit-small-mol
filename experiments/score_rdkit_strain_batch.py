"""
Batch RDKit MMFF strain + UFF vdW scoring for the dashboard pipeline.

Two coverage modes:

  cofold     : 997 Boltz cofolds, FULL pipeline (strain + vdW). Reads
               <dir>/<yaml_name>_model_0.pdb (complex). If a .lig.sdf is
               present uses that for the ligand; otherwise extracts HETATM
               from the complex PDB and restores bonds from the SMILES
               template in the top-1000 manifest.
               Output: data/boltz_poses/zap70_cys346_rdkit_strain_cofold.csv

  posefree   : ~520K dashboard candidates. Reads SMILES from
               results/paper_evaluation/all_methods_bulk_scored_v4.csv,
               generates one ETKDGv3 conformer, MMFF SP energy vs best-of-5
               free reference. Pose-free (constitutional) strain only.
               Output: data/paper_evaluation/rdkit_strain_posefree.csv

Usage:
    python experiments/score_rdkit_strain_batch.py --mode cofold [--n 10]
    python experiments/score_rdkit_strain_batch.py --mode posefree [--n 1000]

Multiprocessing: 8 workers (configurable via --workers).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rdkit_strain import score_cofold, score_posefree

COFOLD_ROOT = PROJECT_ROOT / "data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions"
MANIFEST_PATH = PROJECT_ROOT / "data/boltz_poses/top1000_manifest__zap70_cys346.json"
BULK_CSV = PROJECT_ROOT / "results/paper_evaluation/all_methods_bulk_scored_v4.csv"
COFOLD_OUT = PROJECT_ROOT / "data/boltz_poses/zap70_cys346_rdkit_strain_cofold.csv"
POSEFREE_OUT = PROJECT_ROOT / "data/paper_evaluation/rdkit_strain_posefree.csv"

# Cohort 3,597 cofold layout (CIF format, sharded across from_*/<row_id>/).
COHORT3597_ROOT = PROJECT_ROOT / "data/boltz_results/cohort_3597_full"
COHORT3597_CSV = PROJECT_ROOT / "data/tier4_scored/boltz2_cohort_A_relaxed.csv"
COHORT3597_OUT = PROJECT_ROOT / "data/tier4_scored/rdkit_strain_pose_3597_v2.csv"


# ---------------------- cofold workers ----------------------

def _cofold_worker(task: tuple[int, str, str, str, str]) -> dict:
    row_id, yaml_name, smiles, pdb_path, sdf_path = task
    try:
        out = score_cofold(
            pdb_path=Path(pdb_path),
            smiles=smiles,
            sdf_path=Path(sdf_path) if sdf_path else None,
            num_conf=5,
            compute_interaction=True,
        )
    except Exception as e:
        out = {"success_flag": 0, "error": f"exception:{type(e).__name__}",
               "strain_kcal_mol": float("nan"),
               "vdw_interaction_kcal_mol": float("nan"),
               "e_bound_kcal_mol": float("nan"),
               "e_free_kcal_mol": float("nan")}
    out["row_id"] = row_id
    out["yaml_name"] = yaml_name
    return out


def _build_cofold_tasks() -> list[tuple[int, str, str, str, str]]:
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    yaml_to_row: dict[str, tuple[int, str]] = {}
    for row_id_str, rec in manifest.items():
        try:
            row_id = int(row_id_str)
        except ValueError:
            continue
        yaml_to_row[rec["yaml_name"]] = (row_id, rec["smiles"])

    tasks = []
    for sub in sorted(COFOLD_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        yaml_name = sub.name
        if yaml_name not in yaml_to_row:
            continue
        row_id, smiles = yaml_to_row[yaml_name]
        pdb = sub / f"{yaml_name}_model_0.pdb"
        sdf = sub / f"{yaml_name}_model_0.lig.sdf"
        if not pdb.exists():
            continue
        tasks.append((row_id, yaml_name, smiles, str(pdb), str(sdf) if sdf.exists() else ""))
    return tasks


def run_cofold(n_limit: int | None, workers: int, out_path: Path) -> None:
    tasks = _build_cofold_tasks()
    if n_limit is not None:
        tasks = tasks[:n_limit]
    print(f"[cofold] {len(tasks)} tasks, {workers} workers -> {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "yaml_name", "strain_kcal_mol", "vdw_interaction_kcal_mol",
              "e_bound_kcal_mol", "e_free_kcal_mol", "success_flag", "error"]

    t0 = time.perf_counter()
    n_ok = 0
    n_err = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        if workers == 1:
            for i, task in enumerate(tasks):
                res = _cofold_worker(task)
                w.writerow({k: res.get(k) for k in fields})
                if res["success_flag"] == 1:
                    n_ok += 1
                else:
                    n_err += 1
                if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                    elapsed = time.perf_counter() - t0
                    print(f"  [cofold] {i+1}/{len(tasks)} ok={n_ok} err={n_err} "
                          f"elapsed={elapsed:.1f}s", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for i, res in enumerate(pool.imap_unordered(_cofold_worker, tasks, chunksize=4)):
                    w.writerow({k: res.get(k) for k in fields})
                    if res["success_flag"] == 1:
                        n_ok += 1
                    else:
                        n_err += 1
                    if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                        elapsed = time.perf_counter() - t0
                        print(f"  [cofold] {i+1}/{len(tasks)} ok={n_ok} err={n_err} "
                              f"elapsed={elapsed:.1f}s", flush=True)
    print(f"[cofold] done. ok={n_ok} err={n_err} total={len(tasks)} "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------- posefree workers ----------------------

def _posefree_worker(task: tuple[int, str]) -> dict:
    row_id, smiles = task
    try:
        # Fast settings tuned for 620K-candidate sweep: 2 conformers, 100 max iters.
        # The bound proxy (unrelaxed ETKDG) carries large numerical baseline; the
        # free reference must merely be on a similar relative scale to give a
        # rank-meaningful Delta. 5/300 -> 2/100 gives ~3x speedup with strong
        # spearman agreement (>0.95) on a 500-mol smoke test.
        out = score_posefree(smiles, num_conf_free=2, seed=42, free_max_its=100)
    except Exception as e:
        out = {"success_flag": 0, "error": f"exception:{type(e).__name__}",
               "strain_kcal_mol": float("nan"),
               "e_bound_kcal_mol": float("nan"),
               "e_free_kcal_mol": float("nan")}
    out["row_id"] = row_id
    return out


def _build_posefree_tasks(n_limit: int | None,
                          shard: tuple[int, int] | None = None) -> list[tuple[int, str]]:
    """Build (row_id, smiles) task list. If shard=(idx, total) is set, only keep
    rows where row_id % total == idx, allowing independent process partitioning."""
    tasks: list[tuple[int, str]] = []
    with open(BULK_CSV) as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                row_id = int(float(row["row_id"]))
            except (KeyError, ValueError, TypeError):
                continue
            if shard is not None:
                idx, total = shard
                if row_id % total != idx:
                    continue
            smi = row.get("smiles", "").strip()
            if not smi:
                continue
            tasks.append((row_id, smi))
            if n_limit is not None and len(tasks) >= n_limit:
                break
    return tasks


def run_posefree(n_limit: int | None, workers: int, out_path: Path,
                 shard: tuple[int, int] | None = None) -> None:
    tasks = _build_posefree_tasks(n_limit, shard=shard)
    print(f"[posefree] {len(tasks)} tasks, {workers} workers -> {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "strain_kcal_mol", "e_bound_kcal_mol",
              "e_free_kcal_mol", "success_flag", "error"]

    t0 = time.perf_counter()
    n_ok = 0
    n_err = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        if workers == 1:
            for i, task in enumerate(tasks):
                res = _posefree_worker(task)
                w.writerow({k: res.get(k) for k in fields})
                if res["success_flag"] == 1:
                    n_ok += 1
                else:
                    n_err += 1
                if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    print(f"  [posefree] {i+1}/{len(tasks)} ok={n_ok} err={n_err} "
                          f"rate={rate:.1f}/s elapsed={elapsed:.1f}s", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for i, res in enumerate(pool.imap_unordered(_posefree_worker, tasks, chunksize=16)):
                    w.writerow({k: res.get(k) for k in fields})
                    if res["success_flag"] == 1:
                        n_ok += 1
                    else:
                        n_err += 1
                    if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                        elapsed = time.perf_counter() - t0
                        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                        print(f"  [posefree] {i+1}/{len(tasks)} ok={n_ok} err={n_err} "
                              f"rate={rate:.1f}/s elapsed={elapsed:.1f}s", flush=True)
    print(f"[posefree] done. ok={n_ok} err={n_err} total={len(tasks)} "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------- cohort3597 cofold workers ----------------------

def _cohort3597_worker(task: tuple[int, str, str]) -> dict:
    """Score one cohort cofold. Converts CIF to PDB in a tempdir if not present
    next to the CIF, then runs `score_cofold`."""
    row_id, smiles, cif_path = task
    cif_p = Path(cif_path)
    pdb_p = cif_p.with_suffix(".pdb")
    try:
        if not pdb_p.exists():
            # Import inline so the worker doesn't pay the import cost up-front.
            from experiments.cif_to_pdb_for_mmgbsa import convert_cif_to_pdb
            convert_cif_to_pdb(cif_p, pdb_p)
        out = score_cofold(
            pdb_path=pdb_p,
            smiles=smiles,
            sdf_path=None,
            num_conf=3,
            compute_interaction=True,
        )
    except Exception as e:
        out = {"success_flag": 0, "error": f"exception:{type(e).__name__}:{e}",
               "strain_kcal_mol": float("nan"),
               "vdw_interaction_kcal_mol": float("nan"),
               "e_bound_kcal_mol": float("nan"),
               "e_free_kcal_mol": float("nan")}
    out["row_id"] = int(row_id)
    return out


def _build_cohort3597_tasks() -> list[tuple[int, str, str]]:
    """Walk every from_*/<row_id>/<row_id>_model_0.cif, pair with cohort SMILES."""
    import pandas as pd
    smi_df = pd.read_csv(COHORT3597_CSV, low_memory=False, usecols=["row_id", "smiles"])
    smi_df["row_id"] = smi_df["row_id"].astype(int)
    smi_map = dict(zip(smi_df["row_id"], smi_df["smiles"]))

    tasks: list[tuple[int, str, str]] = []
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
            if not cif.exists():
                continue
            smi = smi_map.get(rid)
            if smi is None:
                continue
            tasks.append((rid, smi, str(cif)))
            seen.add(rid)
    return tasks


def run_cohort3597(n_limit: int | None, workers: int, out_path: Path) -> None:
    """Score every cohort 3,597 cofold's MMFF pose-strain. Resumable via prior CSV."""
    import pandas as pd
    tasks = _build_cohort3597_tasks()
    print(f"[cohort3597] discovered {len(tasks)} cofold tasks (CIF on disk)", flush=True)

    # Resume from existing output (don't re-score what's already done & success_flag==1).
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev.loc[prev["success_flag"] == 1, "row_id"].astype(int).tolist())
        print(f"[cohort3597] resume: {len(done)} rows already done; will skip", flush=True)
        tasks = [t for t in tasks if t[0] not in done]
        print(f"[cohort3597] remaining: {len(tasks)}", flush=True)

    if n_limit is not None:
        tasks = tasks[:n_limit]
        print(f"[cohort3597] limit applied: {len(tasks)}", flush=True)

    if not tasks:
        print(f"[cohort3597] nothing to do; current rows in {out_path}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "strain_kcal_mol", "vdw_interaction_kcal_mol",
              "e_bound_kcal_mol", "e_free_kcal_mol", "success_flag", "error"]

    # If resuming, load previous rows so we can rewrite the full CSV at the end.
    accumulated: list[dict] = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        for _, r in prev.iterrows():
            accumulated.append({k: r.get(k) for k in fields})

    t0 = time.perf_counter()
    n_ok = sum(1 for r in accumulated if r.get("success_flag") == 1)
    n_err = sum(1 for r in accumulated if r.get("success_flag") != 1)
    new_rows: list[dict] = []
    if workers == 1:
        for i, task in enumerate(tasks):
            res = _cohort3597_worker(task)
            new_rows.append({k: res.get(k) for k in fields})
            if res["success_flag"] == 1:
                n_ok += 1
            else:
                n_err += 1
            if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = (i+1)/elapsed if elapsed > 0 else 0
                print(f"  [cohort3597] {i+1}/{len(tasks)} ok={n_ok} err={n_err} rate={rate:.2f}/s elapsed={elapsed:.1f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_cohort3597_worker, tasks, chunksize=2)):
                new_rows.append({k: res.get(k) for k in fields})
                if res["success_flag"] == 1:
                    n_ok += 1
                else:
                    n_err += 1
                if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                    elapsed = time.perf_counter() - t0
                    rate = (i+1)/elapsed if elapsed > 0 else 0
                    eta = (len(tasks)-(i+1))/max(rate,1e-6)/60
                    print(f"  [cohort3597] {i+1}/{len(tasks)} ok={n_ok} err={n_err} rate={rate:.2f}/s elapsed={elapsed:.1f}s eta={eta:.1f}min", flush=True)
                    # Partial flush every 100 rows
                    out_df = pd.DataFrame(accumulated + new_rows)
                    out_df.to_csv(out_path, index=False)

    # Final write
    import pandas as pd
    out_df = pd.DataFrame(accumulated + new_rows)
    out_df.to_csv(out_path, index=False)
    print(f"[cohort3597] done. ok={n_ok} err={n_err} total={len(accumulated)+len(new_rows)} "
          f"wall={time.perf_counter()-t0:.1f}s -> {out_path}", flush=True)


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cofold", "posefree", "cohort3597"], required=True)
    ap.add_argument("--n", type=int, default=None, help="Limit number of tasks (for smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="Override output CSV path")
    ap.add_argument("--shard", type=str, default=None,
                    help="Shard spec 'idx/total' for posefree (e.g. '0/4'). Each shard writes its own file.")
    args = ap.parse_args()

    shard = None
    if args.shard is not None:
        idx_s, total_s = args.shard.split("/")
        shard = (int(idx_s), int(total_s))

    if args.mode == "cofold":
        out = args.out if args.out is not None else COFOLD_OUT
        run_cofold(args.n, args.workers, out)
    elif args.mode == "cohort3597":
        out = args.out if args.out is not None else COHORT3597_OUT
        run_cohort3597(args.n, args.workers, out)
    else:
        out = args.out if args.out is not None else POSEFREE_OUT
        run_posefree(args.n, args.workers, out, shard=shard)


if __name__ == "__main__":
    # Prevent BLAS/OpenMP oversubscription with multiprocessing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    mp.freeze_support()
    main()
