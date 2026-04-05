#!/usr/bin/env python3
"""
Generate ALL within-assay pairwise combinations (not just MMP pairs).

For each assay, creates n*(n-1)/2 ordered pairs of compounds with delta = pIC50_b - pIC50_a.
Uses the same Goldilocks-filtered assay universe as the MMP dataset.

Also extracts ZAP70 (CHEMBL2803) all-pairs separately.

Usage:
    conda run -n quris python -u scripts/generate_all_pairs.py
    conda run -n quris python -u scripts/generate_all_pairs.py --target CHEMBL2803
    conda run -n quris python -u scripts/generate_all_pairs.py --update-cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from itertools import combinations

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays"
EXTRACTED_DIR = DATA_DIR / "extracted"
CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"

RAW_FILE = DATA_DIR / "molecule_pIC50_minimal.csv"
MMP_FILE = EXTRACTED_DIR / "shared_pairs_deduped.csv"
OUTPUT_FILE = EXTRACTED_DIR / "all_pairs_within_assay.csv"
ZAP70_OUTPUT = EXTRACTED_DIR / "zap70_all_pairs.csv"


def get_mmp_assay_ids():
    """Get assay IDs used in the MMP dataset (within-assay only)."""
    print("Loading MMP dataset to identify assays...")
    mmp = pd.read_csv(MMP_FILE, usecols=["assay_id_a", "is_within_assay"])
    within = mmp[mmp["is_within_assay"] == True]
    assay_ids = set(within["assay_id_a"].unique())
    print(f"  {len(assay_ids)} unique within-assay assay IDs")
    return assay_ids


def load_raw_activities(assay_ids=None, target_chembl_id=None):
    """Load raw activity data, optionally filtered to specific assays or target."""
    print(f"Loading raw activities from {RAW_FILE.name}...")
    raw = pd.read_csv(RAW_FILE)
    print(f"  {len(raw):,} total rows, {raw['molecule_chembl_id'].nunique():,} molecules, "
          f"{raw['assay_id'].nunique():,} assays")

    if target_chembl_id:
        raw = raw[raw["target_chembl_id"] == target_chembl_id].copy()
        print(f"  Filtered to {target_chembl_id}: {len(raw):,} rows, "
              f"{raw['molecule_chembl_id'].nunique()} molecules, {raw['assay_id'].nunique()} assays")
    elif assay_ids is not None:
        raw = raw[raw["assay_id"].isin(assay_ids)].copy()
        print(f"  Filtered to MMP assays: {len(raw):,} rows, "
              f"{raw['molecule_chembl_id'].nunique():,} molecules")

    # Average duplicate measurements (same molecule in same assay)
    deduped = raw.groupby(["molecule_chembl_id", "assay_id"]).agg({
        "smiles": "first",
        "target_chembl_id": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  After averaging duplicates: {len(deduped):,} molecule-assay entries")
    return deduped


def generate_all_pairs(activities_df, mmp_pairs_set=None):
    """Generate all within-assay pairs from activity data.

    Args:
        activities_df: DataFrame with columns [molecule_chembl_id, assay_id, smiles, target_chembl_id, pIC50]
        mmp_pairs_set: Optional set of (mol_a_id, mol_b_id) tuples that are MMP pairs

    Returns:
        DataFrame with same schema as shared_pairs_deduped.csv
    """
    print("Generating all within-assay pairs...")
    t0 = time.time()

    all_rows = []
    assay_groups = activities_df.groupby("assay_id")
    n_assays = len(assay_groups)

    for idx, (assay_id, group) in enumerate(assay_groups):
        if len(group) < 2:
            continue

        mols = group[["molecule_chembl_id", "smiles", "pIC50"]].values
        target = group["target_chembl_id"].iloc[0]

        for i, j in combinations(range(len(mols)), 2):
            mol_a_id, smi_a, val_a = mols[i]
            mol_b_id, smi_b, val_b = mols[j]

            is_mmp = False
            if mmp_pairs_set:
                is_mmp = (mol_a_id, mol_b_id) in mmp_pairs_set or \
                         (mol_b_id, mol_a_id) in mmp_pairs_set

            all_rows.append({
                "mol_a": smi_a,
                "mol_b": smi_b,
                "mol_a_id": mol_a_id,
                "mol_b_id": mol_b_id,
                "edit_smiles": "",
                "delta": float(val_b) - float(val_a),
                "value_a": float(val_a),
                "value_b": float(val_b),
                "target_chembl_id": target,
                "assay_id_a": int(assay_id),
                "assay_id_b": int(assay_id),
                "is_within_assay": True,
                "is_mmp": is_mmp,
            })

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx+1}/{n_assays} assays, {len(all_rows):,} pairs ({elapsed:.0f}s)")

    df = pd.DataFrame(all_rows)
    elapsed = time.time() - t0
    print(f"  Generated {len(df):,} pairs from {n_assays} assays in {elapsed:.0f}s")
    return df


def get_mmp_pairs_set():
    """Load MMP pair identifiers for annotation."""
    print("Loading MMP pair identifiers...")
    mmp = pd.read_csv(MMP_FILE, usecols=["mol_a_id", "mol_b_id", "is_within_assay"])
    within = mmp[mmp["is_within_assay"] == True]
    pairs = set(zip(within["mol_a_id"], within["mol_b_id"]))
    print(f"  {len(pairs):,} unique MMP within-assay pairs")
    return pairs


def update_morgan_cache(all_smiles):
    """Compute Morgan FPs for molecules not in cache and update the cache file."""
    cache_file = CACHE_DIR / "chemprop-dmpnn.npz"
    if not cache_file.exists():
        print("  No existing Morgan cache found, skipping update")
        return

    data = np.load(cache_file, allow_pickle=True)
    cached_smiles = set(data["smiles"].tolist())
    missing = [s for s in all_smiles if s not in cached_smiles]

    if not missing:
        print("  All molecules already in cache")
        return

    print(f"  Computing Morgan FPs for {len(missing):,} new molecules...")
    from src.embedding.fingerprints import FingerprintEmbedder
    embedder = FingerprintEmbedder(fp_type="morgan", radius=2, n_bits=2048)

    new_embs = []
    for i, smi in enumerate(missing):
        try:
            new_embs.append(embedder.encode(smi))
        except Exception:
            new_embs.append(np.zeros(2048, dtype=np.float32))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(missing)}")

    # Merge with existing
    all_smi = list(data["smiles"]) + missing
    all_emb = np.vstack([data["embeddings"], np.array(new_embs, dtype=np.float32)])

    np.savez_compressed(cache_file,
                        smiles=np.array(all_smi),
                        embeddings=all_emb,
                        emb_dim=np.array(2048))
    print(f"  Updated cache: {len(all_smi):,} total molecules")


def main():
    parser = argparse.ArgumentParser(description="Generate all within-assay pairs")
    parser.add_argument("--target", default=None, help="Generate pairs for a specific target (e.g., CHEMBL2803)")
    parser.add_argument("--update-cache", action="store_true", help="Update Morgan FP cache with new molecules")
    parser.add_argument("--no-mmp-annotation", action="store_true", help="Skip MMP pair annotation (faster)")
    args = parser.parse_args()

    if args.target:
        # Target-specific extraction (e.g., ZAP70)
        activities = load_raw_activities(target_chembl_id=args.target)
        pairs_df = generate_all_pairs(activities)
        output = EXTRACTED_DIR / f"{args.target.lower()}_all_pairs.csv"
        pairs_df.to_csv(output, index=False)
        print(f"\nSaved {len(pairs_df):,} pairs to {output.name}")
    else:
        # Full extraction using MMP assay universe
        assay_ids = get_mmp_assay_ids()
        activities = load_raw_activities(assay_ids=assay_ids)

        mmp_set = None if args.no_mmp_annotation else get_mmp_pairs_set()
        pairs_df = generate_all_pairs(activities, mmp_pairs_set=mmp_set)

        # Stats
        n_mmp = pairs_df["is_mmp"].sum() if "is_mmp" in pairs_df.columns else 0
        n_targets = pairs_df["target_chembl_id"].nunique()
        n_mols = len(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist()))
        print(f"\nSummary:")
        print(f"  Total pairs: {len(pairs_df):,}")
        print(f"  MMP pairs: {n_mmp:,} ({100*n_mmp/len(pairs_df):.1f}%)")
        print(f"  Non-MMP pairs: {len(pairs_df)-n_mmp:,} ({100*(len(pairs_df)-n_mmp)/len(pairs_df):.1f}%)")
        print(f"  Targets: {n_targets}")
        print(f"  Unique molecules: {n_mols:,}")

        pairs_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE.name}")

    if args.update_cache:
        all_smiles = list(set(pairs_df["mol_a"].tolist() + pairs_df["mol_b"].tolist()))
        update_morgan_cache(all_smiles)

    # Always generate ZAP70 pairs too
    if not args.target:
        print("\n--- Generating ZAP70 pairs ---")
        zap_activities = load_raw_activities(target_chembl_id="CHEMBL2803")
        if len(zap_activities) > 0:
            zap_pairs = generate_all_pairs(zap_activities)
            zap_pairs.to_csv(ZAP70_OUTPUT, index=False)
            print(f"Saved {len(zap_pairs):,} ZAP70 pairs to {ZAP70_OUTPUT.name}")


if __name__ == "__main__":
    main()
