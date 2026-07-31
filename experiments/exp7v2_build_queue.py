#!/usr/bin/env python3
"""Build the v2 cell queue for the 4-way comparison (54 pairs x 4 methods = 216 cells).

Reads data/exp7_v2_benchmark/clean_pairs.json and emits CSVs for the two drivers:
  _rl/queue_covft.csv       — for exp7_rl_driver.py (strategies: A=covFT_RL, baseline_prior_anchor=covFT_baseline)
  _rl/queue_mol2mol.csv     — for exp7_mol2mol_driver.py (conditions: mol2mol_baseline, mol2mol_RL)

Columns: pair_id, target_key, anchor_smiles, drug_smiles, strategy/condition, target_chembl_id
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
V2 = PROJ / "data" / "exp7_v2_benchmark"
OUT = V2 / "_rl"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with open(V2 / "clean_pairs.json") as f:
        d = json.load(f)
    pairs = d["pairs"]
    print(f"loaded {len(pairs)} v2 pairs")

    # Build the 4-way cells
    covft_rows, mol2mol_rows = [], []
    for p in pairs:
        base = dict(
            pair_id=p["pair_id"],
            target_key=p["target"],   # using v2 target name as the key (matches phase1 dir)
            target_chembl_id=p["target_chembl_id"],
            anchor_smiles=p["anchor"]["smiles"],
            anchor_chembl_id=p["anchor"]["chembl_id"],
            drug_smiles=p["drug"]["smiles"],
            drug_name=p["drug"]["name"],
            warhead_class=p["warhead_class"],
            hinge_class=p["hinge_class"],
            tier=p["tier"],
            tc_anchor_drug=p["tc_anchor_drug"],
            delta_pic50=p["delta_pic50"],
        )
        # covFT cells (strategies)
        covft_rows.append({**base, "strategy": "A"})                     # covFT_RL
        covft_rows.append({**base, "strategy": "baseline_prior_anchor"}) # covFT_baseline
        # mol2mol cells (conditions)
        mol2mol_rows.append({**base, "condition": "mol2mol_baseline"})
        mol2mol_rows.append({**base, "condition": "mol2mol_RL"})

    pd.DataFrame(covft_rows).to_csv(OUT / "queue_covft.csv", index=False)
    pd.DataFrame(mol2mol_rows).to_csv(OUT / "queue_mol2mol.csv", index=False)
    print(f"wrote {OUT/'queue_covft.csv'}: {len(covft_rows)} cells (covFT_RL + covFT_baseline)")
    print(f"wrote {OUT/'queue_mol2mol.csv'}: {len(mol2mol_rows)} cells (mol2mol_baseline + mol2mol_RL)")
    print(f"TOTAL: {len(covft_rows) + len(mol2mol_rows)} cells")


if __name__ == "__main__":
    main()
