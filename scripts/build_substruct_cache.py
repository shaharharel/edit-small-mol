"""Build a per-SMILES substructure flag cache for ALL data sources:

  - results/paper_evaluation/all_methods_bulk_scored_v4.csv      (Tier 1/2/3 + Tier-4 bulk)
  - data/tier4_scored/*_scored.csv                                (Tier 4 cohort CSVs)

Writes results/paper_evaluation/substruct_flags.csv with columns:
  smiles, mol1_murcko_smarts_match, thiq_core, acryl_match

Backend loads this once at startup and merges flags into DF + into every cohort
read inside /api/cohort_smis. Idempotent — re-runs reuse existing rows by SMILES
and only compute for new ones.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "results" / "paper_evaluation" / "substruct_flags.csv"

MURCKO_SMARTS = Chem.MolFromSmarts("O=C(Nc1cncn1)c1cccc2c1CNC2")
THIQ_ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N")


def collect_smiles() -> set[str]:
    smis: set[str] = set()
    main = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
    if main.exists():
        df = pd.read_csv(main, usecols=["smiles"])
        smis.update(s for s in df["smiles"].dropna().astype(str) if s)
        print(f"  +{len(smis):,} from {main.name}")
    tier4_dir = PROJECT_ROOT / "data" / "tier4_scored"
    for csv in sorted(tier4_dir.glob("*_scored.csv")):
        if csv.name.endswith(".bak.pre_substruct"):
            continue
        try:
            df = pd.read_csv(csv, usecols=["smiles"], low_memory=False)
        except Exception:
            continue
        before = len(smis)
        smis.update(s for s in df["smiles"].dropna().astype(str) if s)
        print(f"  +{len(smis) - before:,} from {csv.name} (cum {len(smis):,})")
    return smis


def compute_flags(smis: list[str]) -> pd.DataFrame:
    n = len(smis)
    murcko = [False] * n
    thiq = [False] * n
    acryl = [False] * n
    t0 = time.time()
    for i, s in enumerate(smis):
        if i and i % 50000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate
            print(f"  ... {i:>7,}/{n:,}  {rate:.0f}/s  ETA {eta:.0f}s")
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        murcko[i] = bool(m.HasSubstructMatch(MURCKO_SMARTS))
        thiq[i] = bool(m.HasSubstructMatch(THIQ_ACRYL_SMARTS))
        acryl[i] = bool(m.HasSubstructMatch(ACRYL_SMARTS))
    return pd.DataFrame({
        "smiles": smis,
        "mol1_murcko_smarts_match": murcko,
        "thiq_core": thiq,
        "acryl_match": acryl,
    })


def main():
    print("Collecting unique SMILES across all data sources …")
    all_smis = collect_smiles()
    print(f"Total unique SMILES: {len(all_smis):,}")

    existing = None
    if CACHE_PATH.exists():
        existing = pd.read_csv(CACHE_PATH)
        print(f"Existing cache: {len(existing):,} rows")
        known = set(existing["smiles"].astype(str))
        new = sorted(all_smis - known)
        print(f"New SMILES to compute: {len(new):,}")
        if not new:
            print("Cache up to date.")
            return
        added = compute_flags(new)
        merged = pd.concat([existing, added], ignore_index=True).drop_duplicates(subset=["smiles"], keep="last")
        merged.to_csv(CACHE_PATH, index=False)
        print(f"Wrote {len(merged):,} rows → {CACHE_PATH}")
    else:
        all_sorted = sorted(all_smis)
        out = compute_flags(all_sorted)
        out.to_csv(CACHE_PATH, index=False)
        print(f"Wrote {len(out):,} rows → {CACHE_PATH}")
        for c in ("mol1_murcko_smarts_match", "thiq_core", "acryl_match"):
            print(f"  {c}: {int(out[c].sum()):,} True")


if __name__ == "__main__":
    main()
