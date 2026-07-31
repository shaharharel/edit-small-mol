#!/usr/bin/env python3
"""Compute pose-free 3D shape similarity vs Mol1 for cohort CSVs.

Adds 2 columns:
  - shape_Tc_mol1   : 1 - ShapeTanimotoDist(mol, mol1) after Crippen O3A alignment
  - o3a_score       : Crippen O3A alignment score vs Mol1

Process per row:
  1. SMILES -> AddHs -> ETKDGv3 embed -> MMFF94 optimize (best effort)
  2. GetCrippenO3A(probe=mol, ref=mol1).Align()
  3. Record 1 - ShapeTanimotoDist (so higher = more similar)
  4. Record O3A score

Failures yield NaN for both columns. Uses multiprocessing.Pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

_MOL1_REF = None  # global, initialised in each worker


def _build_mol1() -> Chem.Mol:
    m = Chem.MolFromSmiles(MOL1_SMILES)
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params) == -1:
        # retry with random coords
        params.useRandomCoords = True
        AllChem.EmbedMolecule(m, params)
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    except Exception:
        pass
    return m


def _init_worker():
    global _MOL1_REF
    _MOL1_REF = _build_mol1()


def _process_one(args: Tuple[int, str]) -> Tuple[int, Optional[float], Optional[float]]:
    idx, smi = args
    if not isinstance(smi, str) or not smi:
        return idx, None, None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return idx, None, None
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) == -1:
            return idx, None, None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass  # use unoptimized
        try:
            o3a = rdMolAlign.GetCrippenO3A(m, _MOL1_REF)
            o3a.Align()
            o3a_score = float(o3a.Score())
            sim = 1.0 - float(AllChem.ShapeTanimotoDist(m, _MOL1_REF))
            return idx, sim, o3a_score
        except Exception:
            return idx, None, None
    except Exception:
        return idx, None, None


def process_csv(csv_path: Path, workers: int, smiles_col: str = "smiles") -> dict:
    t0 = time.time()
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        # fall back common alternatives
        for cand in ("SMILES", "smi", "canonical_smiles"):
            if cand in df.columns:
                smiles_col = cand
                break
        else:
            raise RuntimeError(f"no smiles col in {csv_path} (cols: {list(df.columns)[:10]}...)")

    skip = False
    if "shape_Tc_mol1" in df.columns:
        cov = df["shape_Tc_mol1"].notna().mean()
        if cov >= 0.90:
            print(f"[skip] {csv_path.name}: shape_Tc_mol1 already {cov:.1%} covered")
            return {"csv": str(csv_path), "rows": len(df), "skipped": True, "coverage": cov}

    n = len(df)
    jobs = list(zip(range(n), df[smiles_col].tolist()))

    sims = [None] * n
    scores = [None] * n
    done = 0
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for idx, sim, score in pool.imap_unordered(_process_one, jobs, chunksize=64):
            sims[idx] = sim
            scores[idx] = score
            done += 1
            if done % 5000 == 0:
                el = time.time() - t0
                rate = done / el
                eta_s = (n - done) / max(rate, 1e-6)
                print(f"  [{csv_path.name}] {done}/{n} ({100*done/n:.1f}%)  "
                      f"rate={rate:.1f}/s  eta={eta_s/60:.1f} min", flush=True)

    df["shape_Tc_mol1"] = sims
    df["o3a_score"] = scores

    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

    cov = df["shape_Tc_mol1"].notna().mean()
    med = float(np.nanmedian(df["shape_Tc_mol1"].astype(float).values)) if cov > 0 else float("nan")
    med_o3a = float(np.nanmedian(df["o3a_score"].astype(float).values)) if cov > 0 else float("nan")
    el = time.time() - t0
    print(f"[done] {csv_path.name}: n={n} cov={cov:.1%} median_shape_Tc={med:.3f} "
          f"median_o3a={med_o3a:.2f} elapsed={el/60:.1f} min", flush=True)
    return {
        "csv": str(csv_path), "rows": n, "skipped": False,
        "coverage": cov, "median_shape_Tc": med, "median_o3a": med_o3a,
        "elapsed_min": el / 60.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="cohort CSV path (repeatable)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--smiles-col", default="smiles")
    args = ap.parse_args()

    results = []
    for p in args.csv:
        path = Path(p)
        if not path.exists():
            print(f"[miss] {p} not found, skipping")
            continue
        try:
            r = process_csv(path, workers=args.workers, smiles_col=args.smiles_col)
            results.append(r)
        except Exception as e:
            print(f"[err]  {p}: {e}")
            results.append({"csv": p, "error": str(e)})

    print("\n=== summary ===")
    for r in results:
        if "error" in r:
            print(f"  ERR  {r['csv']}: {r['error']}")
        elif r.get("skipped"):
            print(f"  skip {Path(r['csv']).name}  (cov={r['coverage']:.1%})")
        else:
            print(f"  ok   {Path(r['csv']).name}  "
                  f"n={r['rows']} cov={r['coverage']:.1%} "
                  f"med_shape_Tc={r['median_shape_Tc']:.3f} "
                  f"med_o3a={r['median_o3a']:.2f} "
                  f"({r['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
