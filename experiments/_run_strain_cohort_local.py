"""Run RDKit MMFF94s pose-aware strain over cohort cofold lig.sdf files.

Uses the validated `score_cofold` from `src.utils.rdkit_strain`, which is the
function that produced the legacy 978-cofold reference distribution
(strain median ≈ 33 kcal/mol).
"""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rdkit_strain import score_cofold  # noqa: E402


def _worker(task):
    row_id, pdb_str, sdf_str, smi = task
    try:
        out = score_cofold(
            pdb_path=Path(pdb_str),
            smiles=smi,
            sdf_path=Path(sdf_str) if sdf_str else None,
            num_conf=5,
            compute_interaction=False,
        )
    except Exception as e:
        out = {"success_flag": 0,
               "error": f"exception:{type(e).__name__}:{str(e)[:80]}",
               "strain_kcal_mol": float("nan"),
               "e_bound_kcal_mol": float("nan"),
               "e_free_kcal_mol": float("nan")}
    out["row_id"] = row_id
    return out


def build_tasks(cofold_root: Path, smiles_map: dict):
    tasks = []
    for sub in sorted(cofold_root.glob("from_*/*")):
        if not sub.is_dir():
            continue
        try:
            row_id = int(sub.name)
        except ValueError:
            continue
        sdf = sub / f"{sub.name}_model_0.lig.sdf"
        pdb = sub / f"{sub.name}_model_0.pdb"
        if not sdf.exists() and not pdb.exists():
            continue
        smi = smiles_map.get(row_id, "")
        if not smi:
            continue
        tasks.append((row_id,
                      str(pdb) if pdb.exists() else "",
                      str(sdf) if sdf.exists() else "",
                      smi))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold_root", required=True)
    ap.add_argument("--smiles_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--progress", default="/tmp/strain_progress.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    smiles_map = {}
    with open(args.smiles_csv) as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                rid = int(float(row["row_id"]))
            except (KeyError, ValueError, TypeError):
                continue
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles_map[rid] = smi

    tasks = build_tasks(Path(args.cofold_root), smiles_map)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"[strain] {len(tasks)} tasks, {args.workers} workers", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "strain_kcal_mol", "e_bound_kcal_mol", "e_free_kcal_mol",
              "success_flag", "error"]

    t0 = time.time(); n_ok = 0; last_hb = 0
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            results.append(r)
            n_ok += int(r["success_flag"])
            now = time.time()
            if now - last_hb > 60 or i == len(tasks):
                with open(args.progress, "w") as fh:
                    json.dump({"metric": "rdkit_strain_kcal_mol",
                               "done": i, "total": len(tasks),
                               "ok": n_ok, "elapsed_s": now - t0,
                               "rate_per_s": i / max(now - t0, 1e-9),
                               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh)
                last_hb = now
                print(f"  [strain] {i}/{len(tasks)} ok={n_ok} elapsed={now-t0:.0f}s", flush=True)

    results.sort(key=lambda d: d["row_id"])
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[strain] done. ok={n_ok}/{len(tasks)} wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    mp.freeze_support()
    main()
