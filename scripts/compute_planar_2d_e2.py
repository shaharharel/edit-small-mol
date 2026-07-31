#!/usr/bin/env python3
"""Compute 2D planar-dihedral panel for v2-cond+DAP (E2) cohort.

Same logic as experiments/covft_geometric_comparison.py::_compute_2d_one but
run as a standalone script for a single cohort. Output CSV matches the
schema of results/paper_evaluation/covft_geometric_2d_*.csv.

Usage:
  conda run -n quris python scripts/compute_planar_2d_e2.py \
    --samples data/paper_dap_repro/samples_v2cond_E2_10k.csv \
    --cohort v2cond_DAP \
    --out results/paper_evaluation/covft_geometric_2d_v2cond_DAP.csv \
    --workers 6
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
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

RDLogger.DisableLog("rdApp.*")

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
PLANARITY_THRESH_DEG = 20.0


def _planar_dev(d_deg: float) -> float:
    d = abs(d_deg)
    return min(d, abs(180.0 - d))


def _compute_2d_one(args):
    idx, smi = args
    out = {
        "idx": idx,
        "smi": smi,
        "acryl_match": False,
        "embed_ok": False,
        "dihedral_deg": None,
        "planar_dev_deg": None,
        "pre_reactivity_score": None,
        "msg": "",
    }
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out["msg"] = "smi_parse_fail"
            return out
        patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            out["msg"] = "no_acrylamide"
            return out
        out["acryl_match"] = True
        m = matches[0]
        b_idx, a_idx, c_idx, n_idx = int(m[0]), int(m[1]), int(m[2]), int(m[4])

        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                out["msg"] = "embed_fail"
                return out
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        out["embed_ok"] = True
        conf = mol_h.GetConformer()
        d = GetDihedralDeg(conf, b_idx, a_idx, c_idx, n_idx)
        out["dihedral_deg"] = float(d)
        dev = _planar_dev(d)
        out["planar_dev_deg"] = float(dev)
        out["pre_reactivity_score"] = 1.0 if dev <= PLANARITY_THRESH_DEG else 0.0
        out["msg"] = "ok"
    except Exception as e:
        out["msg"] = f"exc:{type(e).__name__}:{str(e)[:80]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cohort", default="v2cond_DAP")
    ap.add_argument("--smi_col", default="SMILES")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.samples)
    if args.smi_col not in df.columns:
        # fallback
        args.smi_col = df.columns[0]
    smiles = df[args.smi_col].astype(str).tolist()
    tasks = list(enumerate(smiles))
    print(f"[2D] cohort={args.cohort} N={len(tasks)} workers={args.workers}", flush=True)

    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = [exc.submit(_compute_2d_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "idx": -1, "smi": "", "acryl_match": False, "embed_ok": False,
                    "dihedral_deg": None, "planar_dev_deg": None,
                    "pre_reactivity_score": None, "msg": f"fut_exc:{e}",
                })
            done += 1
            if done % 500 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"  [2D {args.cohort}] {done}/{len(tasks)} {rate:.1f} mol/s ETA {eta/60:.1f} min", flush=True)

    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    out["cohort"] = args.cohort
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[2D] wrote {len(out)} rows to {args.out}", flush=True)

    ok = out[out["msg"] == "ok"]
    if len(ok):
        planar = ok["planar_dev_deg"].astype(float)
        print(f"[2D] median planar_dev_deg = {planar.median():.2f}°  (n_ok={len(ok)}/{len(out)})", flush=True)
        pre_react = (out["pre_reactivity_score"] >= 0.5).mean()
        print(f"[2D] pre_reactivity_score >=0.5 frac (all rows) = {pre_react*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
