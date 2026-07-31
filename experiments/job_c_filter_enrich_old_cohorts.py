#!/usr/bin/env python3
"""Job C: Filter & Ranking F1+F2+F3 columns for 4 OLD Mol1-anchored cohorts.

Adds (if missing):
  PAINS_alerts   — RDKit FilterCatalog PAINS hit count
  Brenk_alerts   — RDKit FilterCatalog BRENK hit count
  SAScore        — SA_Score from RDKit Contrib (1=easy, 10=hard synth)
  LLE            — pIC50_film - LogP
  LE             — 1.4 * pIC50_film / HeavyAtoms
  max_pubTc + mean_pubTc + median_pubTc + top10_mean_pubTc + closest_lead
                  (via experiments.add_pubtc_to_mol1_cohorts.compute_pubtc)

Cohorts: exp6_v3, exp2_v2_rl_v2, exp6_v4, exp6_v5 (~213K mols total).

Skips a column block if it already has >=90% coverage.
Writes atomically (.csv.tmp then os.replace).

Usage:
    python experiments/job_c_filter_enrich_old_cohorts.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog as FC

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# SAScore from Contrib
SASCORE_PATH = Path("/opt/miniconda3/envs/quris/share/RDKit/Contrib/SA_Score")
if SASCORE_PATH.exists():
    sys.path.insert(0, str(SASCORE_PATH))

COHORTS = ["exp6_v3", "exp2_v2_rl_v2", "exp6_v4", "exp6_v5"]


# ----------------- workers -----------------

_PAINS = None
_BRENK = None
_SA = None


def _init_filters():
    global _PAINS, _BRENK, _SA
    p = FC.FilterCatalogParams()
    p.AddCatalog(FC.FilterCatalogParams.FilterCatalogs.PAINS)
    _PAINS = FC.FilterCatalog(p)
    b = FC.FilterCatalogParams()
    b.AddCatalog(FC.FilterCatalogParams.FilterCatalogs.BRENK)
    _BRENK = FC.FilterCatalog(b)
    try:
        import sascorer  # type: ignore
        _SA = sascorer.calculateScore
    except Exception as e:
        print(f"  warn: SAScore import failed ({e}) — SAScore col will be NaN", flush=True)
        _SA = None


def _filt_one(smi):
    if not isinstance(smi, str) or not smi:
        return None, None, None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None, None
        pains = len(_PAINS.GetMatches(m))
        brenk = len(_BRENK.GetMatches(m))
        if _SA is not None:
            try:
                sa = float(_SA(m))
            except Exception:
                sa = None
        else:
            sa = None
        return pains, brenk, sa
    except Exception:
        return None, None, None


# ----------------- per-cohort -----------------

def process_cohort(name: str, workers: int) -> dict:
    csv = PROJECT / "data/tier4_scored" / f"{name}_scored.csv"
    if not csv.exists():
        return {"cohort": name, "error": "missing"}

    t0 = time.time()
    df = pd.read_csv(csv)
    n = len(df)
    print(f"\n>>> {name}: n={n:,}", flush=True)

    need_filt = not (
        "PAINS_alerts" in df.columns and df["PAINS_alerts"].notna().mean() >= 0.90
        and "Brenk_alerts" in df.columns and df["Brenk_alerts"].notna().mean() >= 0.90
        and "SAScore" in df.columns and df["SAScore"].notna().mean() >= 0.90
    )

    if need_filt:
        t1 = time.time()
        with Pool(processes=workers, initializer=_init_filters) as pool:
            results = pool.map(_filt_one, df["smiles"].tolist(), chunksize=200)
        pains = [r[0] for r in results]
        brenk = [r[1] for r in results]
        sa = [r[2] for r in results]
        if "PAINS_alerts" not in df.columns or df["PAINS_alerts"].notna().mean() < 0.90:
            df["PAINS_alerts"] = pains
        if "Brenk_alerts" not in df.columns or df["Brenk_alerts"].notna().mean() < 0.90:
            df["Brenk_alerts"] = brenk
        if "SAScore" not in df.columns or df["SAScore"].notna().mean() < 0.90:
            df["SAScore"] = sa
        print(f"  PAINS/Brenk/SA done in {time.time()-t1:.1f}s", flush=True)
    else:
        print("  PAINS/Brenk/SA already present", flush=True)

    # LLE + LE — instant
    if "pIC50_film" in df.columns and "LogP" in df.columns:
        df["LLE"] = (pd.to_numeric(df["pIC50_film"], errors="coerce")
                     - pd.to_numeric(df["LogP"], errors="coerce")).round(3)
    if "pIC50_film" in df.columns and "HeavyAtoms" in df.columns:
        ha = pd.to_numeric(df["HeavyAtoms"], errors="coerce")
        df["LE"] = (1.4 * pd.to_numeric(df["pIC50_film"], errors="coerce") / ha).round(3)
    print(f"  LLE/LE computed", flush=True)

    # pubTc via the existing module
    need_pubtc = not (
        "max_pubTc" in df.columns and df["max_pubTc"].notna().mean() >= 0.90
    )
    if need_pubtc:
        t1 = time.time()
        # late import so it doesn't load pubTc panel for skip-only cohorts
        from experiments.add_pubtc_to_mol1_cohorts import compute_pubtc
        # serial-safe; can multiproc but compute_pubtc reads module-level panel_fps
        with Pool(processes=workers) as pool:
            results = pool.map(compute_pubtc, df["smiles"].tolist(), chunksize=200)
        for col in ("max_pubTc", "mean_pubTc", "median_pubTc",
                    "top10_mean_pubTc", "closest_lead"):
            df[col] = [r[col] for r in results]
        print(f"  pubTc done in {time.time()-t1:.1f}s", flush=True)
    else:
        print("  pubTc already present", flush=True)

    # atomic write
    tmp = csv.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)

    elapsed = time.time() - t0
    summary = {
        "cohort": name,
        "rows": n,
        "elapsed_s": round(elapsed, 1),
        "PAINS_cov": float(df["PAINS_alerts"].notna().mean()) if "PAINS_alerts" in df else 0.0,
        "Brenk_cov": float(df["Brenk_alerts"].notna().mean()) if "Brenk_alerts" in df else 0.0,
        "SAScore_cov": float(df["SAScore"].notna().mean()) if "SAScore" in df else 0.0,
        "max_pubTc_cov": float(df["max_pubTc"].notna().mean()) if "max_pubTc" in df else 0.0,
        "med_max_pubTc": float(df["max_pubTc"].median()) if "max_pubTc" in df and df["max_pubTc"].notna().any() else float("nan"),
        "med_PAINS": float(df["PAINS_alerts"].median()) if "PAINS_alerts" in df else float("nan"),
        "med_Brenk": float(df["Brenk_alerts"].median()) if "Brenk_alerts" in df else float("nan"),
        "med_SA": float(df["SAScore"].median()) if "SAScore" in df else float("nan"),
        "med_LLE": float(df["LLE"].median()) if "LLE" in df else float("nan"),
        "med_LE": float(df["LE"].median()) if "LE" in df else float("nan"),
    }
    print(f"  [done] {name}: {elapsed:.1f}s  "
          f"PAINS_cov={summary['PAINS_cov']:.1%}  "
          f"max_pubTc_cov={summary['max_pubTc_cov']:.1%}  "
          f"med_SA={summary['med_SA']:.2f}  med_LLE={summary['med_LLE']:.2f}  "
          f"med_LE={summary['med_LE']:.2f}  med_max_pubTc={summary['med_max_pubTc']:.3f}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cohort", action="append", default=None,
                    help="override cohort list (repeatable)")
    args = ap.parse_args()

    cohorts = args.cohort if args.cohort else COHORTS
    summaries = []
    t_all = time.time()
    for c in cohorts:
        try:
            s = process_cohort(c, workers=args.workers)
            summaries.append(s)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append({"cohort": c, "error": str(e)})

    print("\n=== JOB C SUMMARY ===")
    for s in summaries:
        if "error" in s:
            print(f"  ERR  {s['cohort']}: {s['error']}")
        else:
            print(f"  ok   {s['cohort']:<20} n={s['rows']:>7,} "
                  f"PAINS_cov={s['PAINS_cov']:.1%} pubTc_cov={s['max_pubTc_cov']:.1%} "
                  f"SA={s['med_SA']:.2f} LLE={s['med_LLE']:.2f} LE={s['med_LE']:.2f} "
                  f"({s['elapsed_s']:.1f}s)")
    print(f"  total elapsed: {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
