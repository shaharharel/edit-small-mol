"""Check which exp7 drugs are in CovInDB v2 corpus + which were first published after 2022.

Output: data/exp7_lo_benchmark/exp7_contamination.json with per-pair flags:
   drug_in_covindb: bool
   drug_year: int or null
   strict_post2022: bool (drug_year > 2022)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs

COVINDB_CSV = PROJECT_ROOT / "data" / "covbinder" / "raw_covindb2" / "CovInDB_All.csv"
OUT_JSON = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_contamination.json"

# Known approval/disclosure years for our 50-pair drug roster
DRUG_YEARS = {
    # EGFR T790M
    "osimertinib": 2014,
    "rociletinib": 2014,
    "naquotinib": 2015,
    "lazertinib": 2018,
    "mavelertinib": 2014,
    "almonertinib": 2015,
    "aumolertinib": 2015,
    # BTK
    "ibrutinib": 2007,
    "acalabrutinib": 2014,
    "zanubrutinib": 2017,
    "tirabrutinib": 2014,
    "spebrutinib": 2014,
    "evobrutinib": 2018,
    "remibrutinib": 2019,
    # JAK3
    "ritlecitinib": 2015,
    "PF-06651600": 2015,  # same as ritlecitinib
    # HER2 / pan-ErbB
    "afatinib": 2008,
    "dacomitinib": 2008,
    "neratinib": 2008,
    "mobocertinib": 2018,
    "poziotinib": 2010,
    "pyrotinib": 2012,
    "tucatinib": 2013,
    "canertinib": 2003,
    # FGFR
    "futibatinib": 2017,
    "fisogatinib": 2017,
    "infigratinib": 2014,
    "erdafitinib": 2014,
    "pemigatinib": 2017,
    "roblitinib": 2018,
    "rogaratinib": 2014,
    "lirafugratinib": 2020,
    "fexagratinib": 2016,
}


def canon(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def main():
    pairs = load_all_pairs()
    print(f"Loaded {len(pairs)} pairs")

    # Build CovInDB canonical SMILES set
    if not COVINDB_CSV.exists():
        print(f"COVINDB not found at {COVINDB_CSV}")
        return
    df = pd.read_csv(COVINDB_CSV, low_memory=False)
    print(f"CovInDB rows: {len(df)}")
    cov_canon = set()
    for s in df["SMILES"].dropna().astype(str):
        c = canon(s)
        if c:
            cov_canon.add(c)
    print(f"Unique canon SMILES in CovInDB: {len(cov_canon)}")

    results = []
    for p in pairs:
        dc = canon(p["drug_smiles"])
        in_db = dc in cov_canon if dc else False
        year = DRUG_YEARS.get(p["drug_name"].lower(), None)
        if year is None:
            # Try without case sensitivity by checking common aliases
            for k, v in DRUG_YEARS.items():
                if k.lower() == p["drug_name"].lower():
                    year = v
                    break
        strict_post2022 = (year is not None) and (year > 2022)
        results.append({
            "pair_id": p["pair_id"],
            "target_key": p["target_key"],
            "drug_name": p["drug_name"],
            "drug_smiles": p["drug_smiles"],
            "drug_canon": dc,
            "drug_in_covindb": in_db,
            "drug_year": year,
            "strict_post2022": strict_post2022,
        })
    out = {
        "n_pairs": len(results),
        "n_drug_in_covindb": sum(1 for r in results if r["drug_in_covindb"]),
        "n_strict_post2022": sum(1 for r in results if r["strict_post2022"]),
        "loose_subset_n": sum(1 for r in results if not r["drug_in_covindb"]),
        "drugs_in_db": sorted(set(r["drug_name"] for r in results if r["drug_in_covindb"])),
        "pairs": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"  drug_in_covindb: {out['n_drug_in_covindb']}/{out['n_pairs']}")
    print(f"  strict_post2022: {out['n_strict_post2022']}/{out['n_pairs']}")
    print(f"  loose_subset_n:  {out['loose_subset_n']}/{out['n_pairs']}")
    print(f"  drugs in DB: {out['drugs_in_db']}")


if __name__ == "__main__":
    main()
