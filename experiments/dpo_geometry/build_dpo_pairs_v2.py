#!/usr/bin/env python3
"""Stream C v2: rebuild DPO preference pairs from the existing v1 set,
filtered to keep ONLY pairs where BOTH chosen and rejected are Mol1-similar
(Morgan Tc >= 0.30 to Mol1 anchor).

Motivation:
  v1 training collapsed to "drop Mol1 scaffold" because chosen mean Tc
  to Mol1 was only 0.24. The mol2mol policy is anchored on Mol1 → giving
  it preference signals over OFF-distribution molecules taught it to
  abandon the scaffold.

Reads:  data/dpo_pairs/geometry_only.parquet (v1)
Writes: data/dpo_pairs/geometry_only_v2.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PARQUET = PROJECT_ROOT / "data" / "dpo_pairs" / "geometry_only.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "dpo_pairs" / "geometry_only_v2.parquet"
ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
TC_THRESHOLD = 0.30


def main():
    df = pd.read_parquet(IN_PARQUET)
    print(f"v1 pairs: {len(df)}")
    mol1 = Chem.MolFromSmiles(ANCHOR_SMI)
    fp_mol1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)

    def tc(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), fp_mol1
        )

    chosen_unique = df["chosen_smiles"].unique()
    rejected_unique = df["rejected_smiles"].unique()
    print(f"computing Tc for {len(chosen_unique)} chosen + {len(rejected_unique)} rejected...")
    chosen_tc = {s: tc(s) for s in chosen_unique}
    rejected_tc = {s: tc(s) for s in rejected_unique}
    df["tc_chosen_to_mol1"] = df["chosen_smiles"].map(chosen_tc)
    df["tc_rejected_to_mol1"] = df["rejected_smiles"].map(rejected_tc)

    mask = (df["tc_chosen_to_mol1"] >= TC_THRESHOLD) & (df["tc_rejected_to_mol1"] >= TC_THRESHOLD)
    v2 = df.loc[mask].copy()
    print(f"v2 pairs (both Tc>={TC_THRESHOLD} to Mol1): {len(v2)}")

    # Re-split train/val
    import numpy as np
    rng = np.random.default_rng(7)
    unique_chosen = v2["chosen_smiles"].unique()
    rng.shuffle(unique_chosen)
    val_chosen = set(unique_chosen[: max(1, len(unique_chosen) // 10)])
    v2["split"] = np.where(v2["chosen_smiles"].isin(val_chosen), "val", "train")
    n_train = (v2["split"] == "train").sum()
    n_val = (v2["split"] == "val").sum()
    print(f"train={n_train}, val={n_val}, n_unique_chosen={len(unique_chosen)}")

    v2.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}")

    summary = {
        "n_pairs_total": int(len(v2)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_unique_chosen": int(v2["chosen_smiles"].nunique()),
        "n_unique_rejected": int(v2["rejected_smiles"].nunique()),
        "tc_threshold_to_mol1": TC_THRESHOLD,
        "tc_chosen_mean": float(v2["tc_chosen_to_mol1"].mean()),
        "tc_rejected_mean": float(v2["tc_rejected_to_mol1"].mean()),
        "delta_q_mean": float(v2["delta_q"].mean()),
        "scoring_formula": "q = exp(-bd_dev_deg^2 / (2 * 8^2))",
        "anchor_smiles": ANCHOR_SMI,
        "v2_rationale": (
            "v1 chose mean Tc=0.24 to Mol1 → DPO pushed away from Mol1 scaffold. "
            "v2 enforces Tc>=0.3 for BOTH chosen and rejected so policy "
            "learns geometry preference WITHIN Mol1-shaped molecules."
        ),
    }
    import json
    (OUT_PARQUET.parent / (OUT_PARQUET.stem + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"summary → {OUT_PARQUET.parent / (OUT_PARQUET.stem + '.summary.json')}")


if __name__ == "__main__":
    main()
