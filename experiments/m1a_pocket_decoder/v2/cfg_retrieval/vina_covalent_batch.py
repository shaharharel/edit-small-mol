"""AD-CovDock (Vina covalent-tethered) batch for CFG × retrieval samples.

Wraps `experiments/run_covalent_docking.py` primitives:
  - build_cov_tethered_pdbqt: meeko CovalentBuilder aligns β-C → Cys346 CB
    and α-C → Cys346 CA, then writes ligand PDBQT.
  - vina_dock(mode="score_only"): evaluates the tethered pose against
    Cys346-stripped ZAP70 (4K2R) — the canonical AD-CovDock-Vina recipe.
  - read_tethered_geom: reads d(Cβ-SG) and Bürgi-Dunitz angle from the
    tethered PDBQT (β-C should be ~1.85 Å from SG by construction).

For each valid+acryl sample from covalent_metric_panel we output:
  cell, sample_idx, smi, vina_cov_score (kcal/mol; lower better),
  vina_cov_d_sg, vina_cov_bd_angle, vina_cov_ok, vina_cov_note

BD-ready gate (canonical Bürgi-Dunitz, matching run_covalent_docking.py):
  d_SG ∈ [1.55, 2.15] Å AND bd_angle ∈ [102°, 112°]

For the covalent-tethered pose, d_SG is FIXED near 1.85Å by construction
(meeko sets β-C on Cys346 CB, 1.7Å from SG).  Real variation comes from
the pose's α-C orientation, i.e. the BD angle.

Multiprocessed with mp.Pool.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# Ensure the run_covalent_docking module is importable.
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cov_one(args):
    # Support both 3-tuple (legacy: score_only) and 4-tuple (with mode).
    if len(args) == 3:
        idx, smi, work_dir = args
        mode = "score_only"
    else:
        idx, smi, work_dir, mode = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    result = {"row_idx": idx,
                "vina_cov_score": float("nan"),
                "vina_cov_d_sg": float("nan"),
                "vina_cov_bd_angle": float("nan"),
                "vina_cov_ok": False,
                "vina_cov_note": "",
                "vina_cov_mode": mode}
    try:
        # Late import to keep each worker isolated.
        from experiments.run_covalent_docking import (
            build_cov_tethered_pdbqt, vina_dock, read_tethered_geom,
        )
    except Exception as e:
        result["vina_cov_note"] = f"import fail: {type(e).__name__}:{e}"
        return result
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            result["vina_cov_note"] = "smi parse fail"
            return result
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        lig_pdbqt = work_dir / f"lig_{idx:06d}.pdbqt"
        prep = build_cov_tethered_pdbqt(m, smi, lig_pdbqt)
        if not prep.get("ok", False):
            result["vina_cov_note"] = f"prep_fail:{prep.get('msg', 'unknown')}"
            return result
        pose_pdbqt = work_dir / f"pose_{idx:06d}.pdbqt"
        dock = vina_dock(lig_pdbqt, pose_pdbqt, mode=mode, threads=1)
        if not dock.get("ok", False):
            result["vina_cov_note"] = f"vina_fail:{dock.get('msg', 'unknown')}"
            return result
        result["vina_cov_score"] = float(dock.get("score") or float("nan"))
        geom = read_tethered_geom(lig_pdbqt, smi)
        if geom.get("d_sg") is not None:
            result["vina_cov_d_sg"] = float(geom["d_sg"])
        if geom.get("bd_angle") is not None:
            result["vina_cov_bd_angle"] = float(geom["bd_angle"])
        result["vina_cov_ok"] = True
        result["vina_cov_note"] = "ok"
        # Clean up per-mol files.
        try:
            lig_pdbqt.unlink(missing_ok=True)
            pose_pdbqt.unlink(missing_ok=True)
        except Exception:
            pass
        return result
    except Exception as e:
        result["vina_cov_note"] = f"exc: {type(e).__name__}:{e}"
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_metric_panel_v2.csv"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_vina_cov_panel.csv"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "vina_cov_batch_progress.json"))
    ap.add_argument("--work_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/vina_cov_work"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--incremental_save_every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="score_only",
                    choices=["score_only", "local_only"],
                    help="score_only = eval tethered pose (fast, clash-penalized); "
                          "local_only = light Newton min around tether (slower, "
                          "gives usable absolute affinities per DrugFlow finding).")
    ap.add_argument("--plan_csv", default=None,
                    help="If given, restrict to (cell, sample_idx) pairs in this "
                          "CSV (columns: cell, sample_idx).  Used for the "
                          "top-25/cell follow-up pass.")
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    panel = pd.read_csv(args.panel_csv)
    valid = panel[(panel["valid"] == True) &
                    (panel["acryl_largest"] == True)].copy().reset_index(drop=True)
    print(f"Vina-cov panel: {len(valid)} valid+acryl rows", flush=True)
    if args.limit is not None:
        valid = valid.head(args.limit)
        print(f"Limited to {len(valid)}", flush=True)

    if args.plan_csv is not None and Path(args.plan_csv).exists():
        plan = pd.read_csv(args.plan_csv)
        plan_keys = set(zip(plan["cell"].astype(str),
                             plan["sample_idx"].astype(int)))
        valid = valid[
            valid.apply(lambda r: (str(r["cell"]), int(r["sample_idx"]))
                          in plan_keys, axis=1)
        ].reset_index(drop=True)
        print(f"Plan-restricted to {len(valid)} rows", flush=True)

    done_keys = set()
    prior_rows = []
    if args.resume and Path(args.out_csv).exists():
        try:
            prior = pd.read_csv(args.out_csv)
            ok_mask = prior["vina_cov_ok"].fillna(False).astype(bool)
            for _, r in prior[ok_mask].iterrows():
                done_keys.add((str(r["cell"]), int(r["sample_idx"])))
                prior_rows.append(r.to_dict())
            print(f"[resume] {len(done_keys)} rows already done in {args.out_csv}",
                   flush=True)
        except Exception as e:
            print(f"[resume] load fail: {e}", flush=True)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    row_index_map = {}
    for i, row in valid.iterrows():
        key = (str(row["cell"]), int(row["sample_idx"]))
        if key in done_keys:
            continue
        jobs.append((len(jobs), str(row["largest_frag_SMILES"]),
                        args.work_dir, args.mode))
        row_index_map[len(jobs) - 1] = i
    print(f"Jobs to run: {len(jobs)}", flush=True)

    def dump_partial(results_so_far, done_positions):
        out_rows = list(prior_rows)
        for pos in done_positions:
            res = results_so_far[pos]
            if res is None:
                continue
            i_in_valid = row_index_map[pos]
            r = valid.iloc[i_in_valid].to_dict()
            base = {"cell": r["cell"], "sample_idx": int(r["sample_idx"]),
                      "smi": r["largest_frag_SMILES"]}
            out_rows.append({**base, **{k: v for k, v in res.items()
                                             if k != "row_idx"}})
        df_out = pd.DataFrame(out_rows)
        if len(df_out):
            df_out["bd_ready_vina_cov"] = (
                (df_out["vina_cov_d_sg"] >= 1.55) &
                (df_out["vina_cov_d_sg"] <= 2.15) &
                (df_out["vina_cov_bd_angle"] >= 102.0) &
                (df_out["vina_cov_bd_angle"] <= 112.0)
            ).fillna(False)
            df_out["bd_ready_vina_cov_relaxed"] = (
                (df_out["vina_cov_d_sg"] >= 1.5) &
                (df_out["vina_cov_d_sg"] <= 2.5) &
                (df_out["vina_cov_bd_angle"] >= 90.0) &
                (df_out["vina_cov_bd_angle"] <= 130.0)
            ).fillna(False)
        df_out.to_csv(args.out_csv, index=False)

    t_start = time.time()
    results = [None] * len(jobs)
    completed_positions = []
    if jobs:
        with mp.Pool(processes=args.workers) as pool:
            for k, res in enumerate(pool.imap_unordered(_cov_one, jobs, chunksize=2)):
                results[res["row_idx"]] = res
                completed_positions.append(res["row_idx"])
                if (k + 1) % 20 == 0 or (k + 1) == len(jobs):
                    el = time.time() - t_start
                    rate = (k + 1) / max(el, 1e-6)
                    eta = (len(jobs) - k - 1) / max(rate, 1e-6)
                    n_ok = sum(1 for r in results
                                if r is not None and r["vina_cov_ok"])
                    print(f"[{k+1}/{len(jobs)}] ok={n_ok}  {rate:.2f}/s  "
                           f"ETA={eta/60:.1f}min", flush=True)
                    Path(args.progress_path).write_text(json.dumps({
                        "phase": "vina_cov_batch",
                        "done": k + 1 + len(done_keys),
                        "total": len(jobs) + len(done_keys),
                        "n_ok": n_ok + len(done_keys),
                        "rate_per_sec": rate, "eta_min": eta / 60,
                        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, indent=2))
                if args.incremental_save_every > 0 and \
                        (k + 1) % args.incremental_save_every == 0:
                    dump_partial(results, completed_positions)

    dump_partial(results, completed_positions)
    print(f"Wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
