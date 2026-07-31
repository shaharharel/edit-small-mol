"""Build the 3 RL seed files for Track B' Mol1-anchored RL experiment.

Outputs to data/mol1_rl_seeds/:
  - seed_mol1_only.smi:                Mol1 only (1 mol)
  - seed_zap70_all_plus_mol1.smi:      all 280 ChEMBL ZAP70 mols + Mol1 (~281 unique)
  - seed_kinase_zap70x5_mol1x20.smi:   kinase panel (de-ZAP70) + ZAP70 x5 + Mol1 x20

Used by Track B' (Mol1-anchored RL retraining) on the EXP6_v5 config.
Each seed file is sanity-checked before write.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "mol1_rl_seeds"

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def main():
    mol1_canon = canon(MOL1)
    assert mol1_canon is not None, "Mol1 failed to parse"

    zap = pd.read_csv(PROJECT / "data/docking_chembl_zap70/docking_results.csv")
    kin = pd.read_csv(PROJECT / "data/docking_kinase_panel/docking_results.csv")

    zap_canon = []
    for s in zap["smiles"]:
        c = canon(s)
        if c is not None:
            zap_canon.append(c)
    zap_set = set(zap_canon)

    kin_only = []
    for s in kin["smiles"]:
        c = canon(s)
        if c is not None and c not in zap_set:
            kin_only.append(c)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # B1: Mol1 only
    p1 = OUT_DIR / "seed_mol1_only.smi"
    p1.write_text(mol1_canon + "\n")

    # B2: All 280 ZAP70 + Mol1 (Mol1 first)
    seeds_2 = [mol1_canon] + [s for s in dict.fromkeys(zap_canon) if s != mol1_canon]
    p2 = OUT_DIR / "seed_zap70_all_plus_mol1.smi"
    p2.write_text("\n".join(seeds_2) + "\n")

    # B3: All kinase (non-ZAP70) + ZAP70 x5 + Mol1 x20
    seeds_3 = list(dict.fromkeys(kin_only))
    for s in dict.fromkeys(zap_canon):
        if s != mol1_canon:
            seeds_3.extend([s] * 5)
    seeds_3.extend([mol1_canon] * 20)
    p3 = OUT_DIR / "seed_kinase_zap70x5_mol1x20.smi"
    p3.write_text("\n".join(seeds_3) + "\n")

    # QA — verify files
    print("=== Track B' RL seed files ===")
    for p in [p1, p2, p3]:
        lines = [l for l in p.read_text().splitlines() if l]
        uniq = len(set(lines))
        mol1_n = lines.count(mol1_canon)
        # Every line must canonicalize
        bad = sum(1 for l in lines if canon(l) is None)
        # Mol1 must be present at least once
        assert mol1_n >= 1, f"Mol1 missing in {p}"
        assert bad == 0, f"{bad} unparsable SMILES in {p}"
        print(f"  {p.name}: {len(lines):,} lines, {uniq:,} unique, "
              f"Mol1 ×{mol1_n}, parse-ok=100%")


if __name__ == "__main__":
    main()
