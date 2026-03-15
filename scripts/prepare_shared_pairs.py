#!/usr/bin/env python3
"""
Extract shared MMP pairs (appearing in both within-assay and cross-assay contexts).
Deduplicates by averaging deltas for the same (mol_a_id, mol_b_id, is_within_assay, target) combo.

Output: data/overlapping_assays/extracted/shared_pairs_deduped.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

DATA_DIR = Path("data/overlapping_assays/extracted")
INPUT_FILE = DATA_DIR / "overlapping_assay_pairs_minimal_mmp.csv"
OUTPUT_FILE = DATA_DIR / "shared_pairs_deduped.csv"


def main():
    # Pass 1: Find shared pair keys
    print("Pass 1: Finding shared pairs...")
    within_pairs = set()
    cross_pairs = set()

    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=500_000)):
        is_mmp = chunk['mol_a_id'] != chunk['mol_b_id']
        mmp = chunk[is_mmp]
        w = mmp[mmp['is_within_assay'] == True]
        c = mmp[mmp['is_within_assay'] == False]
        within_pairs.update(zip(w['mol_a_id'], w['mol_b_id']))
        cross_pairs.update(zip(c['mol_a_id'], c['mol_b_id']))
        print(f"  Chunk {i+1}: within={len(within_pairs):,}, cross={len(cross_pairs):,}")

    shared = within_pairs & cross_pairs
    print(f"\n  Shared pairs: {len(shared):,}")

    # Convert to sets of mol_a_id for fast filtering
    shared_a = {p[0] for p in shared}
    shared_b = {p[1] for p in shared}

    # Pass 2: Filter and collect shared pair rows (vectorized)
    print("\nPass 2: Extracting shared pair rows...")
    chunks_out = []
    total_kept = 0

    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=500_000)):
        is_mmp = chunk['mol_a_id'] != chunk['mol_b_id']
        mmp = chunk[is_mmp].copy()

        # Fast pre-filter by mol_a_id (reduces search space)
        mask_a = mmp['mol_a_id'].isin(shared_a)
        candidates = mmp[mask_a]

        # Exact filter: check (mol_a_id, mol_b_id) pair is in shared set
        if len(candidates) > 0:
            pair_keys = list(zip(candidates['mol_a_id'], candidates['mol_b_id']))
            exact_mask = [pk in shared for pk in pair_keys]
            kept = candidates[exact_mask]
            if len(kept) > 0:
                chunks_out.append(kept)
                total_kept += len(kept)

        print(f"  Chunk {i+1}: kept {total_kept:,} rows so far")

    # Combine
    print(f"\nCombining {len(chunks_out)} chunks...")
    df = pd.concat(chunks_out, ignore_index=True)
    print(f"  Total rows before dedup: {len(df):,}")

    # Deduplicate: group by (mol_a_id, mol_b_id, is_within_assay, target_chembl_id)
    # Average delta, value_a, value_b; keep first for SMILES/edit/assay
    print("\nDeduplicating...")
    group_cols = ['mol_a_id', 'mol_b_id', 'is_within_assay', 'target_chembl_id']
    agg_dict = {
        'delta': 'mean',
        'value_a': 'mean',
        'value_b': 'mean',
        'mol_a': 'first',
        'mol_b': 'first',
        'edit_smiles': 'first',
        'assay_id_a': 'first',
        'assay_id_b': 'first',
    }
    # Count observations per group
    counts = df.groupby(group_cols).size().reset_index(name='n_observations')
    deduped = df.groupby(group_cols).agg(agg_dict).reset_index()
    deduped = deduped.merge(counts, on=group_cols)

    n_within = (deduped['is_within_assay'] == True).sum()
    n_cross = (deduped['is_within_assay'] == False).sum()
    n_mols = len(set(deduped['mol_a'].tolist() + deduped['mol_b'].tolist()))
    n_targets = deduped['target_chembl_id'].nunique()

    print(f"\nFinal deduped dataset:")
    print(f"  Total rows: {len(deduped):,}")
    print(f"  Within-assay: {n_within:,}")
    print(f"  Cross-assay: {n_cross:,}")
    print(f"  Unique molecules: {n_mols:,}")
    print(f"  Targets: {n_targets}")
    print(f"  Avg observations per group: {deduped['n_observations'].mean():.1f}")

    deduped.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
