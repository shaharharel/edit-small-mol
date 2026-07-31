"""Backfill substructure columns (mol1_murcko_smarts_match, thiq_core, acryl_match)
into Tier-4 scored CSVs that don't already have them.

Idempotent: skip if all three columns are present. Adds columns IN-PLACE (no schema
loss — only ADDs). Writes a .bak.pre_substruct copy if any column is missing.

Used by experiments/server/report.html's Filter 0 (substructure-group filter), so
every Tier-4 cohort table can be filtered by Mol1 Murcko / THIQ / Murcko+Acryl.
"""

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# SMARTS — same as experiments/analyze_mol1_pharmacophore_coverage.py
MURCKO_SMARTS = Chem.MolFromSmarts("O=C(Nc1cncn1)c1cccc2c1CNC2")
THIQ_ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_DIR = PROJECT_ROOT / "data" / "tier4_scored"

TARGET_FILES = [
    "exp2_scored.csv",
    "exp6_scored.csv",
    "exp6_v3_scored.csv",
    "exp2_v2_rl_v2_scored.csv",
    "exp6_v4_scored.csv",
    "exp6_v5_scored.csv",
    "mol1RL_v5_seed_mol1_only_scored.csv",
    "thiq_rl_mol1only_scored.csv",
    "thiq_rl_exp2_zap70_scored.csv",
    "thiq_rl_exp2_kinase_scored.csv",
    "thiq_rl_exp2_mol1only_scored.csv",
    "thiq_rl_zap70_scored.csv",
    "thiq_rl_kinase_scored.csv",
    "murcko_rl_zap70_scored.csv",
    "murcko_rl_kinase_scored.csv",
    "murcko_rl_exp2_zap70_scored.csv",
    "murcko_rl_exp2_kinase_scored.csv",
]


def compute_cols(smiles_series: pd.Series) -> dict:
    murcko, thiq, acryl = [], [], []
    for smi in smiles_series:
        if not isinstance(smi, str) or not smi:
            murcko.append(False)
            thiq.append(False)
            acryl.append(False)
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            murcko.append(False)
            thiq.append(False)
            acryl.append(False)
            continue
        murcko.append(bool(m.HasSubstructMatch(MURCKO_SMARTS)))
        thiq.append(bool(m.HasSubstructMatch(THIQ_ACRYL_SMARTS)))
        acryl.append(bool(m.HasSubstructMatch(ACRYL_SMARTS)))
    return {
        "mol1_murcko_smarts_match": murcko,
        "thiq_core": thiq,
        "acryl_match": acryl,
    }


def process_file(path: Path) -> dict:
    df = pd.read_csv(path)
    needed = []
    if "mol1_murcko_smarts_match" not in df.columns:
        needed.append("mol1_murcko_smarts_match")
    if "thiq_core" not in df.columns:
        needed.append("thiq_core")
    if "acryl_match" not in df.columns:
        needed.append("acryl_match")
    if not needed:
        return {"file": path.name, "added": [], "n": len(df), "skipped": True}
    if "smiles" not in df.columns:
        return {"file": path.name, "added": [], "n": len(df), "error": "no smiles col"}
    # Backup once
    bak = path.with_suffix(path.suffix + ".bak.pre_substruct")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    cols = compute_cols(df["smiles"])
    for c in needed:
        df[c] = cols[c]
    df.to_csv(path, index=False)
    counts = {c: int(sum(cols[c])) for c in needed}
    return {"file": path.name, "added": needed, "n": len(df), "counts": counts}


def main():
    results = []
    for name in TARGET_FILES:
        path = SCORED_DIR / name
        if not path.exists():
            print(f"[skip] {name} — file not found")
            continue
        try:
            r = process_file(path)
            results.append(r)
            if r.get("skipped"):
                print(f"[skip] {name} — all 3 cols present (n={r['n']:,})")
            elif "error" in r:
                print(f"[ERR ] {name} — {r['error']}")
            else:
                cnt = r["counts"]
                bits = " ".join(f"{c}={cnt[c]:,}" for c in r["added"])
                print(f"[OK  ] {name} (n={r['n']:,}) +{','.join(r['added'])}  {bits}")
        except Exception as e:
            print(f"[ERR ] {name} — {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
