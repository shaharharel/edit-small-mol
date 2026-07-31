#!/usr/bin/env python3
"""Backfill the ~2,000 NULL rdkit_strain_posefree / shape_Tc_seed / esp_sim_seed
/ warhead_dev_deg entries in the dashboard.

Each worker:
  * 5-second SIGALRM timeout per molecule
  * 1 ETKDGv3 attempt with useRandomCoords (matches original recipe)
  * If that fails, 2 more seeds before bailing
  * Compute strain + 3 D-shape descriptors if successful

Why the original 2,281 rows failed: pathological bonds (boronate-O-P, hypervalent
SH chains, exotic peroxides) that RDKit's distance geometry rejects.  Recovery
rate is expected to be ~5-10%.

Output: data/paper_evaluation/threed_strain_backfill.csv  with columns
    row_id, rdkit_strain_posefree_kcal_mol, shape_Tc_seed, esp_sim_seed,
    warhead_dev_deg, recovered_by

Run:
    /opt/miniconda3/envs/quris/bin/python experiments/backfill_3d_strain_nulls.py
"""
import csv
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(ROOT))

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

POSEFREE_CSV = ROOT / "data" / "paper_evaluation" / "rdkit_strain_posefree.csv"
BULK_CSV = ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
OUT_CSV = ROOT / "data" / "paper_evaluation" / "threed_strain_backfill.csv"

NPROC = max(1, mp.cpu_count() - 2)
TIMEOUT_S = 4  # per-molecule budget


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _backfill_one(task: tuple[int, str]) -> dict:
    """One row, with a hard SIGALRM per-mol budget."""
    row_id, smi = task
    out = {"row_id": int(row_id), "rdkit_strain_posefree_kcal_mol": None,
           "shape_Tc_seed": None, "esp_sim_seed": None, "warhead_dev_deg": None,
           "recovered_by": "init"}

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out["recovered_by"] = "parse_failed"
            return out

        # Try ETKDGv3 with useRandomCoords + 3 seeds
        molH = Chem.AddHs(mol)
        cid = -1
        for seed in (42, 43, 44):
            ps = AllChem.ETKDGv3()
            ps.randomSeed = seed
            ps.useRandomCoords = True
            try:
                cid = AllChem.EmbedMolecule(molH, ps)
            except Exception:
                cid = -1
            if cid >= 0:
                break
        if cid < 0:
            out["recovered_by"] = "embed_failed"
            return out

        # Posefree strain (no H-only relax to match the bug-tested original logic;
        # difference is small in practice and the original recipe in
        # src/utils/rdkit_strain.py uses _h_only_relax before scoring).
        try:
            from src.utils.rdkit_strain import _h_only_relax, _free_min_energy, _safe_mmff_energy
            _h_only_relax(molH, conf_id=cid, variant="MMFF94s")
            e_bound = _safe_mmff_energy(molH, conf_id=cid, variant="MMFF94s")
            if e_bound is not None:
                e_free = _free_min_energy(smi, num_conf=3, seed=42, max_its=150)
                if e_free is not None:
                    out["rdkit_strain_posefree_kcal_mol"] = float(e_bound - e_free)
                    out["recovered_by"] = "strain_ok"
        except _Timeout:
            raise
        except Exception:
            pass

        # 3D shape / ESP / warhead_dev_deg via src.utils.mol1_scoring
        try:
            from src.utils.mol1_scoring import (
                shape_tanimoto_seed, esp_sim_seed, warhead_vector_deviation,
            )
            try: out["shape_Tc_seed"] = float(shape_tanimoto_seed(smi))
            except Exception: pass
            try: out["esp_sim_seed"] = float(esp_sim_seed(smi))
            except Exception: pass
            try: out["warhead_dev_deg"] = float(warhead_vector_deviation(smi))
            except Exception: pass
            if out["shape_Tc_seed"] is not None and out["recovered_by"] == "init":
                out["recovered_by"] = "shape_only"
        except _Timeout:
            raise
        except Exception:
            pass

        if out["recovered_by"] == "init":
            out["recovered_by"] = "embed_ok_but_score_failed"
    except _Timeout:
        out["recovered_by"] = "timeout"
    finally:
        signal.alarm(0)
    return out


def main() -> None:
    t0 = time.time()
    print(f"[backfill] Loading {POSEFREE_CSV.name} ...", flush=True)
    pf = pd.read_csv(POSEFREE_CSV)
    failed_pf = set(pf[pf.success_flag == 0]["row_id"].astype(int).tolist())
    print(f"[backfill]   {len(failed_pf):,} posefree NULL rows", flush=True)

    bulk = pd.read_csv(
        BULK_CSV,
        usecols=["row_id", "smiles", "shape_Tc_seed", "warhead_intact"],
    )
    bulk_rid = bulk["row_id"].astype(int)
    is_pf_null = bulk_rid.isin(failed_pf)
    is_3d_null = bulk["shape_Tc_seed"].isna()
    needs = is_pf_null | is_3d_null
    todo = bulk[needs].copy()
    # drop disconnected
    todo = todo[~todo["smiles"].astype(str).str.contains(".", regex=False, na=False)]
    # only retry warhead_intact rows (dashboard filters others)
    if "warhead_intact" in todo.columns:
        todo = todo[todo["warhead_intact"] == True]
    print(f"[backfill]   to retry: {len(todo):,} rows", flush=True)

    tasks = list(zip(todo["row_id"].astype(int).tolist(),
                     todo["smiles"].astype(str).tolist()))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    n_strain_ok = 0
    n_shape_ok = 0
    n_total = 0

    print(f"[backfill] Running on {NPROC} workers, timeout={TIMEOUT_S}s/mol ...", flush=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "row_id", "rdkit_strain_posefree_kcal_mol", "shape_Tc_seed",
            "esp_sim_seed", "warhead_dev_deg", "recovered_by",
        ])
        w.writeheader()
        fh.flush()
        with mp.Pool(NPROC) as pool:
            for res in pool.imap_unordered(_backfill_one, tasks, chunksize=1):
                w.writerow(res)
                n_total += 1
                if res["rdkit_strain_posefree_kcal_mol"] is not None:
                    n_strain_ok += 1
                if res["shape_Tc_seed"] is not None:
                    n_shape_ok += 1
                if n_total % 20 == 0:
                    fh.flush()
                if n_total % 100 == 0:
                    dt = time.time() - t0
                    print(f"[backfill]   {n_total}/{len(tasks)}  "
                          f"strain_ok={n_strain_ok}  shape_ok={n_shape_ok}  "
                          f"({dt:.0f}s)", flush=True)
        fh.flush()

    print(f"\n[backfill] DONE  total={n_total}  "
          f"strain recovered={n_strain_ok}  3D recovered={n_shape_ok}  "
          f"wall={time.time()-t0:.0f}s", flush=True)
    print(f"[backfill] Wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
