#!/usr/bin/env python3
"""EXP7 LO eval — build the queue CSV for exp7_rl_driver.py.

Output: data/exp7_lo_benchmark/_rl/queue.csv with columns
   pair_id, target_key, anchor_smiles, drug_smiles, strategy

Strategies per pair (in priority order, scheduler runs round-robin across targets):
   1) A                       (RL with single anchor seed)
   2) baseline_prior_anchor   (no RL, prior with anchor seed)
   3) baseline_prior_rand     (no RL, prior with anchor seed + randomize)  -- same toml as anchor; deduplicate
   4) B                       (RL with 100-anchor pool)
   5) baseline_prior_pool     (no RL, prior with 100-anchor pool seed)

We DROP baseline_prior_rand (it's the same call as baseline_prior_anchor — the prior
sample toml always sets randomize_smiles=true). So 4 cells per pair = 200 total.

Round-robin order helps if budget is short: every target gets at least Strategy A
before any target's B is started.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs

OUT_CSV = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_rl" / "queue.csv"
PHASE1_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_phase1"


def vocab_ok(smi: str) -> bool:
    """Cov FT prior vocab — known disallowed tokens from exp6."""
    DISALLOWED = ["[S+]", "[S@@+]", "[S@+]"]
    return all(tok not in smi for tok in DISALLOWED)


def main():
    pairs = load_all_pairs()
    print(f"Loaded {len(pairs)} pairs")

    # Group by target for round-robin order
    by_target = {}
    for p in pairs:
        by_target.setdefault(p["target_key"], []).append(p)

    rows = []
    # Phase 1: all A + all baselines (cheap and important)
    strategy_groups = [
        ("A", "rl"),
        ("baseline_prior_anchor", "sample"),
        ("baseline_prior_pool", "sample"),
        ("B", "rl"),
    ]

    n_skipped_vocab = 0
    for strategy, mode in strategy_groups:
        # round-robin across targets
        max_per_target = max(len(ps) for ps in by_target.values())
        for i in range(max_per_target):
            for tk, ps in by_target.items():
                if i >= len(ps):
                    continue
                pair = ps[i]
                # If strategy uses pool, check audit
                if strategy in ("B", "baseline_prior_pool"):
                    pool_csv = PHASE1_BASE / tk / "anchors" / f"{pair['pair_id']}_b_pool.csv"
                    if not pool_csv.exists():
                        continue
                # Vocab check for A / baseline_prior_anchor (anchor only as seed)
                if strategy in ("A", "baseline_prior_anchor"):
                    if not vocab_ok(pair["anchor_smiles"]):
                        n_skipped_vocab += 1
                        continue
                rows.append({
                    "pair_id": pair["pair_id"],
                    "target_key": pair["target_key"],
                    "anchor_smiles": pair["anchor_smiles"],
                    "drug_smiles": pair["drug_smiles"],
                    "strategy": strategy,
                    "mode": mode,
                })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} cells -> {OUT_CSV}")
    print(f"  by strategy: {df['strategy'].value_counts().to_dict()}")
    print(f"  by target:   {df['target_key'].value_counts().to_dict()}")
    if n_skipped_vocab:
        print(f"  skipped {n_skipped_vocab} cells due to anchor vocab incompatibility")


if __name__ == "__main__":
    main()
