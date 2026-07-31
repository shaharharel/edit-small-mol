#!/usr/bin/env python3
"""Run cov-Vina on a SMILES CSV shard, reusing the exact protocol from
`experiments/run_covalent_docking.py` (meeko CovalentBuilder tether at Cys346
CA/CB, score_only on Cys346-stripped receptor, 20 Å box for score_only mode).

Writes one row per SMILES to --out CSV, with resume support: if --out already
exists, only SMILES whose row is missing (or has vina_ok=False & retry=True)
are re-run. Checkpoint every N rows.

Same schema as `results/paper_evaluation/covft_geometric_covvina_*.csv`:
    cohort, subsample_idx, original_idx, smi, warhead_found, lig_prep_ok,
    vina_ok, vina_affinity, warhead_sg_distance_A, bd_angle_deg,
    pose_converged, msg, dt_s

Usage on ai-chem:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate quris
    python run_covvina_10k.py --smiles <in.csv> --cohort base \\
        --out out.csv --workers 32 --checkpoint-every 200
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# --- Project setup (works both on Mac laptop and ai-chem VM) --------------
def find_project_root() -> Path:
    for cand in [
        Path("/home/shaharh_quris_ai/edit-small-mol"),
        Path("/Users/shaharharel/Documents/github/edit-small-mol"),
        Path(__file__).resolve().parent.parent.parent,
    ]:
        if (cand / "experiments/run_covalent_docking.py").exists():
            return cand
    raise RuntimeError("Could not find edit-small-mol project root")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# Patch VINA_BIN path for ai-chem (system vina at /usr/local/bin/vina)
import experiments.run_covalent_docking as rcd  # noqa: E402
if not rcd.VINA_BIN.exists():
    # Try common paths in order
    for cand in ["/usr/local/bin/vina", "/usr/bin/vina",
                 str(PROJECT_ROOT / "tools/vina")]:
        if Path(cand).exists():
            rcd.VINA_BIN = Path(cand)
            break
    else:
        raise RuntimeError("vina executable not found")

from experiments.run_covalent_docking import (  # noqa: E402
    build_cov_tethered_pdbqt,
    find_warhead_atoms,
    prepare_stripped_receptor,
    read_tethered_geom,
    vina_dock,
    pose_valid_covalent,
)


# --- Per-molecule worker (mirrors covft_geometric_options_bc._cov_vina_one)
def _cov_vina_one(args):
    cohort, sub_idx, original_idx, smi, work_root_str = args
    work_root = Path(work_root_str) / cohort
    work_root.mkdir(parents=True, exist_ok=True)
    row = {
        "cohort": cohort, "subsample_idx": sub_idx, "original_idx": original_idx,
        "smi": smi, "warhead_found": False, "lig_prep_ok": False,
        "vina_ok": False, "vina_affinity": None,
        "warhead_sg_distance_A": None, "bd_angle_deg": None,
        "pose_converged": False, "msg": "", "dt_s": 0.0,
    }
    t0 = time.time()
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            row["msg"] = "smi_parse_fail"
            return row
        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                row["msg"] = "embed_fail"
                return row
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        info = find_warhead_atoms(smi)
        if info is None:
            row["msg"] = "no_warhead"
            return row
        row["warhead_found"] = True

        lig_pdbqt = work_root / f"{cohort}_{sub_idx:06d}.pdbqt"
        prep = build_cov_tethered_pdbqt(mol_h, smi, lig_pdbqt)
        if not prep.get("ok"):
            row["msg"] = f"prep_fail:{prep.get('msg')}"
            # Clean up if half-written
            try:
                lig_pdbqt.unlink(missing_ok=True)
            except Exception:
                pass
            return row
        row["lig_prep_ok"] = True

        g = read_tethered_geom(lig_pdbqt, smi)
        row["warhead_sg_distance_A"] = g.get("d_sg")
        row["bd_angle_deg"] = g.get("bd_angle")

        pose_path = work_root / f"{cohort}_{sub_idx:06d}_pose.pdbqt"
        dock = vina_dock(lig_pdbqt, pose_path, mode="score_only", threads=1)
        if not dock.get("ok"):
            row["msg"] = f"vina_fail:{dock.get('msg')}"
            return row
        row["vina_ok"] = True
        row["vina_affinity"] = dock.get("score")
        row["pose_converged"] = bool(pose_valid_covalent(
            row["warhead_sg_distance_A"], row["bd_angle_deg"]
        ))
        row["msg"] = "ok"
        # Delete pdbqt to save disk (we already extracted geometry + score)
        try:
            lig_pdbqt.unlink(missing_ok=True)
            pose_path.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        row["msg"] = f"exc:{type(e).__name__}:{str(e)[:120]}"
    finally:
        row["dt_s"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True, help="CSV with SMILES column")
    ap.add_argument("--cohort", required=True, help="Cohort tag written into 'cohort' col")
    ap.add_argument("--out", required=True, help="Output CSV path (resume-aware)")
    ap.add_argument("--work-dir", default=None, help="Scratch dir for pdbqt files")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max rows to process (for smoke test)")
    ap.add_argument("--smi-col", default="SMILES")
    args = ap.parse_args()

    smiles_path = Path(args.smiles).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir) if args.work_dir else (
        PROJECT_ROOT / f"data/covvina_10k_work/{args.cohort}"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    # Ensure stripped receptor exists
    prepare_stripped_receptor()

    # Load SMILES
    df_in = pd.read_csv(smiles_path)
    assert args.smi_col in df_in.columns, (
        f"col '{args.smi_col}' missing in {smiles_path} (cols: {list(df_in.columns)})"
    )
    df_in = df_in.reset_index(drop=True)
    if args.limit:
        df_in = df_in.head(args.limit).copy()
    df_in["_orig_idx"] = df_in.index

    # Resume: if out CSV exists, drop already-done original_idx
    done_idx: set = set()
    prior_rows: list = []
    if out_path.exists():
        try:
            prior = pd.read_csv(out_path)
            done_idx = set(int(x) for x in prior["original_idx"].tolist())
            prior_rows = prior.to_dict("records")
            print(f"[resume] loaded {len(done_idx)} prior rows from {out_path}",
                  flush=True)
        except Exception as e:
            print(f"[resume] failed to read {out_path}: {e}. Starting fresh.",
                  flush=True)
            prior_rows = []
            done_idx = set()

    todo = df_in[~df_in["_orig_idx"].isin(done_idx)].reset_index(drop=True)
    if len(todo) == 0:
        print(f"[done] all {len(df_in)} rows already in {out_path}", flush=True)
        return

    # Build tasks
    tasks = []
    for sub_idx, r in todo.iterrows():
        smi = r[args.smi_col]
        if not isinstance(smi, str) or not smi:
            continue
        tasks.append((
            args.cohort, int(sub_idx), int(r["_orig_idx"]), smi, str(work_root),
        ))
    print(
        f"[start] cohort={args.cohort} N={len(tasks)} workers={args.workers} "
        f"out={out_path}", flush=True,
    )

    rows: list = list(prior_rows)
    t0 = time.time()
    done_this_run = 0
    last_ckpt = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = [exc.submit(_cov_vina_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "cohort": args.cohort, "subsample_idx": -1, "original_idx": -1,
                    "smi": "", "warhead_found": False, "lig_prep_ok": False,
                    "vina_ok": False, "vina_affinity": None,
                    "warhead_sg_distance_A": None, "bd_angle_deg": None,
                    "pose_converged": False, "msg": f"fut_exc:{e}", "dt_s": 0.0,
                })
            done_this_run += 1
            if done_this_run % 20 == 0:
                dt = time.time() - t0
                rate = done_this_run / max(dt, 1e-6)
                eta = (len(tasks) - done_this_run) / max(rate, 1e-6)
                ok = sum(1 for r in rows if r.get("vina_ok"))
                print(
                    f"  [{args.cohort}] {done_this_run}/{len(tasks)} "
                    f"ok_total={ok} {rate:.2f} mol/s ETA {eta/60:.1f} min",
                    flush=True,
                )
            if done_this_run - last_ckpt >= args.checkpoint_every:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                last_ckpt = done_this_run

    df_out = pd.DataFrame(rows).sort_values("original_idx").reset_index(drop=True)
    df_out.to_csv(out_path, index=False)
    dt = time.time() - t0
    ok = int(df_out["vina_ok"].astype(bool).sum())
    print(
        f"[done] cohort={args.cohort} rows={len(df_out)} ok={ok} "
        f"elapsed={dt/60:.1f} min out={out_path}", flush=True,
    )


if __name__ == "__main__":
    main()
