"""Build mol2mol queue CSV for exp7_mol2mol_driver.

Two conditions per pair, ordered:
  Round 1 (priority): all 50 mol2mol_baseline (cheap, ~30s each)
  Round 2: all 50 mol2mol_RL (50 steps + sample, ~5 min each)

Output: data/exp7_lo_benchmark/_rl/mol2mol_queue.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs

OUT_CSV = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_rl" / "mol2mol_queue.csv"


def main():
    pairs = load_all_pairs()
    print(f"Loaded {len(pairs)} pairs")

    # Group by target for round-robin
    by_target = {}
    for p in pairs:
        by_target.setdefault(p["target_key"], []).append(p)

    rows = []
    # Phase 1: all baselines first (cheap, ~30s each = ~25 min total)
    # Phase 2: all RL (5 min each = ~4-5h)
    for condition in ["mol2mol_baseline", "mol2mol_RL"]:
        max_per_target = max(len(ps) for ps in by_target.values())
        for i in range(max_per_target):
            for tk, ps in by_target.items():
                if i >= len(ps):
                    continue
                p = ps[i]
                rows.append({
                    "pair_id": p["pair_id"],
                    "target_key": p["target_key"],
                    "anchor_smiles": p["anchor_smiles"],
                    "drug_smiles": p["drug_smiles"],
                    "condition": condition,
                })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} cells -> {OUT_CSV}")
    print(f"  by condition: {df['condition'].value_counts().to_dict()}")
    print(f"  by target:    {df['target_key'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
