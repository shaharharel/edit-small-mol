"""Batched xTB Fukui f+ computation for all v2 samples.

For each sample in `covalent_metric_panel_v2.csv` that has a valid+acryl SMILES,
compute:
  - RDKit ETKDG lowest-energy conformer (+ MMFF opt)
  - xTB --gfn 2 --vfukui single point
  - Parse `Fukui functions:` block from xtb stdout
  - Extract f+ at the acrylamide β-C (SMARTS `[CH2;X3]=[CH;X3][C;X3](=O)[N]` match 0)

Parallelized via multiprocessing.Pool.  On a V100 machine with 8 CPU cores
this processes ~1600 mols in <15 min.

Outputs: covalent_xtb_panel.csv with columns cell, sample_idx, canonical_smi,
  fukui_fplus (real), xtb_ok, xtb_note.
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")

# xtb regex: matches lines like "     1C       0.039    0.015    0.027".
FUKUI_LINE_RE = re.compile(r"^\s*(\d+)([A-Za-z]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def _xtb_one(args):
    idx, smi = args
    result = {"row_idx": idx, "fukui_fplus": float("nan"),
                "fukui_fminus": float("nan"), "fukui_fzero": float("nan"),
                "xtb_ok": False, "xtb_note": ""}
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            result["xtb_note"] = "smi parse fail"
            return result
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            result["xtb_note"] = "no acryl smarts"
            return result
        beta_rdkit = matches[0][0]
        mH = Chem.AddHs(m)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mH, params)
        if cid < 0:
            result["xtb_note"] = "embed fail"
            return result
        try:
            AllChem.MMFFOptimizeMolecule(mH, maxIters=200)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "mol.xyz"
            conf = mH.GetConformer(cid)
            n = mH.GetNumAtoms()
            lines = [str(n), ""]
            for i in range(n):
                a = mH.GetAtomWithIdx(i)
                p = conf.GetAtomPosition(i)
                lines.append(f"{a.GetSymbol():<3s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
            xyz.write_text("\n".join(lines))
            env = os.environ.copy()
            # Cap xtb to 1 thread so multiprocessing pool works cleanly.
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            try:
                res = subprocess.run(
                    ["xtb", str(xyz), "--gfn", "2", "--vfukui", "--iterations", "150"],
                    cwd=tmp, env=env, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                result["xtb_note"] = "timeout"
                return result
            if res.returncode != 0:
                result["xtb_note"] = f"rc={res.returncode}"
                return result
            # Parse Fukui block.
            in_block = False
            table = {}
            for ln in res.stdout.splitlines():
                if not in_block:
                    if ln.strip().startswith("Fukui functions"):
                        in_block = True
                    continue
                if in_block:
                    if ln.strip().startswith("---"):
                        break
                    m2 = FUKUI_LINE_RE.match(ln)
                    if m2 is None:
                        # skip header line "#   f(+)  f(-)  f(0)"
                        continue
                    atom_no = int(m2.group(1))
                    table[atom_no] = (float(m2.group(3)),
                                       float(m2.group(4)),
                                       float(m2.group(5)))
            # xtb atom indexing is 1-based, matches mH order.
            key = beta_rdkit + 1
            if key not in table:
                result["xtb_note"] = f"beta {key} missing from fukui table"
                return result
            fp, fm, f0 = table[key]
            result["fukui_fplus"] = fp
            result["fukui_fminus"] = fm
            result["fukui_fzero"] = f0
            result["xtb_ok"] = True
            result["xtb_note"] = "ok"
            return result
    except Exception as e:
        result["xtb_note"] = f"exc {type(e).__name__}: {e}"
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_metric_panel_v2.csv"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_xtb_panel.csv"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "xtb_batch_progress.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    panel = pd.read_csv(args.panel_csv)
    valid = panel[(panel["valid"] == True) &
                    (panel["acryl_largest"] == True)].copy().reset_index(drop=True)
    print(f"Panel: {len(valid)} valid+acryl rows", flush=True)
    if args.limit is not None:
        valid = valid.head(args.limit)
        print(f"Limited to {len(valid)}", flush=True)

    # Prepare jobs.
    jobs = [(i, str(row["largest_frag_SMILES"]))
             for i, row in valid.iterrows()]
    n_total = len(jobs)
    t_start = time.time()
    results = [None] * n_total

    print(f"Launching Pool of {args.workers} workers over {n_total} mols",
          flush=True)
    with mp.Pool(processes=args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_xtb_one, jobs, chunksize=4)):
            results[res["row_idx"]] = res
            if (k + 1) % 25 == 0 or (k + 1) == n_total:
                el = time.time() - t_start
                rate = (k + 1) / max(el, 1e-6)
                eta = (n_total - k - 1) / max(rate, 1e-6)
                n_ok = sum(1 for r in results if r is not None and r["xtb_ok"])
                print(f"[{k+1}/{n_total}] ok={n_ok}  {rate:.2f}/s  ETA {eta/60:.1f}min",
                       flush=True)
                Path(args.progress_path).write_text(json.dumps({
                    "phase": "xtb_batch", "done": k + 1, "total": n_total,
                    "n_ok": n_ok, "rate_per_sec": rate, "eta_min": eta / 60,
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, indent=2))

    # Merge back into original panel row context.
    out_rows = []
    for i, res in enumerate(results):
        r = valid.iloc[i].to_dict()
        out_rows.append({
            "cell": r["cell"],
            "sample_idx": int(r["sample_idx"]),
            "smi": r["largest_frag_SMILES"],
            "fukui_fplus": res["fukui_fplus"] if res else float("nan"),
            "fukui_fminus": res["fukui_fminus"] if res else float("nan"),
            "fukui_fzero": res["fukui_fzero"] if res else float("nan"),
            "xtb_ok": bool(res["xtb_ok"]) if res else False,
            "xtb_note": res["xtb_note"] if res else "no_result",
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)", flush=True)
    Path(args.progress_path).write_text(json.dumps({
        "phase": "xtb_batch_done",
        "n_total": len(out_df),
        "n_ok": int(out_df["xtb_ok"].sum()),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
