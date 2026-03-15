"""
Extract real ChEMBL IC50 data via the web API for overlapping assay analysis.

Uses chembl_webresource_client to fetch IC50 data for well-studied targets,
identifies overlapping assays (same target, shared compounds), and generates
within-assay MMP pairs.

Usage:
    conda run -n quris python scripts/extraction/extract_chembl_api.py
"""

import os
import sys
import hashlib
import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Well-studied targets with many IC50 assays in ChEMBL
TARGET_IDS = [
    'CHEMBL203',   # EGFR
    'CHEMBL301',   # CDK2
    'CHEMBL279',   # VEGFR2 / KDR
    'CHEMBL4005',  # HDAC1
    'CHEMBL325',   # BRAF
    'CHEMBL2111',  # JAK2
    'CHEMBL267',   # COX-2
    'CHEMBL240',   # DHFR
    'CHEMBL344',   # p38 MAPK
    'CHEMBL4722',  # ABL1
    'CHEMBL1862',  # Aurora A
    'CHEMBL2035',  # PI3Kα
    'CHEMBL3594',  # CDK4
    'CHEMBL4303',  # Sirtuin 1
    'CHEMBL206',   # ERα
]

# Configuration
MIN_COMPOUNDS_PER_ASSAY = 20
MAX_COMPOUNDS_PER_ASSAY = 100
MIN_SHARED_COMPOUNDS = 5
TANIMOTO_THRESHOLD = 0.7
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'overlapping_assays' / 'extracted'


def fetch_target_activities(target_id: str) -> pd.DataFrame:
    """Fetch all IC50 activities for a target via ChEMBL API."""
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    results = activity.filter(
        target_chembl_id=target_id,
        standard_type='IC50',
        standard_relation='=',
        standard_units='nM',
    ).only([
        'molecule_chembl_id', 'canonical_smiles', 'pchembl_value',
        'assay_chembl_id', 'standard_type', 'standard_value',
        'document_chembl_id', 'target_chembl_id',
    ])

    rows = []
    for r in results:
        if r.get('pchembl_value') and r.get('canonical_smiles'):
            try:
                pchembl = float(r['pchembl_value'])
                rows.append({
                    'molecule_chembl_id': r['molecule_chembl_id'],
                    'canonical_smiles': r['canonical_smiles'],
                    'pchembl_value': pchembl,
                    'assay_chembl_id': r['assay_chembl_id'],
                    'standard_type': r['standard_type'],
                    'document_chembl_id': r.get('document_chembl_id', ''),
                    'target_chembl_id': target_id,
                })
            except (ValueError, TypeError):
                continue

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Deduplicate: keep median pchembl per (molecule, assay)
        df = df.groupby(['molecule_chembl_id', 'assay_chembl_id']).agg({
            'canonical_smiles': 'first',
            'pchembl_value': 'median',
            'standard_type': 'first',
            'document_chembl_id': 'first',
            'target_chembl_id': 'first',
        }).reset_index()
    return df


def find_goldilocks_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Find assays with 20-100 distinct compounds."""
    assay_sizes = df.groupby('assay_chembl_id')['molecule_chembl_id'].nunique()
    goldilocks = assay_sizes[
        (assay_sizes >= MIN_COMPOUNDS_PER_ASSAY) &
        (assay_sizes <= MAX_COMPOUNDS_PER_ASSAY)
    ].index
    return df[df['assay_chembl_id'].isin(goldilocks)]


def find_overlapping_assay_pairs(df: pd.DataFrame) -> list:
    """Find pairs of assays for the same target with >= MIN_SHARED_COMPOUNDS shared compounds."""
    pairs = []
    targets = df['target_chembl_id'].unique()

    for target in targets:
        target_df = df[df['target_chembl_id'] == target]
        assays = target_df['assay_chembl_id'].unique()

        # Build molecule sets per assay
        assay_mols = {}
        for assay in assays:
            assay_mols[assay] = set(
                target_df[target_df['assay_chembl_id'] == assay]['molecule_chembl_id']
            )

        # Find overlapping pairs
        for a1, a2 in itertools.combinations(assays, 2):
            shared = assay_mols[a1] & assay_mols[a2]
            if len(shared) >= MIN_SHARED_COMPOUNDS:
                pairs.append({
                    'assay_id_1': a1,
                    'assay_id_2': a2,
                    'target_chembl_id': target,
                    'n_shared': len(shared),
                    'shared_molecules': list(shared),
                })

    return pairs


def generate_mmp_pairs(smiles_list: list) -> list:
    """Generate matched molecular pairs from a list of SMILES using Tanimoto similarity."""
    mols = []
    valid_indices = []
    fps = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    pairs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim >= TANIMOTO_THRESHOLD:
                # Generate edit SMILES (simplified: use canonical diff)
                smi_a = smiles_list[valid_indices[i]]
                smi_b = smiles_list[valid_indices[j]]
                edit_smiles = f"{smi_a}>>{smi_b}"
                pairs.append((valid_indices[i], valid_indices[j], sim, edit_smiles))

    return pairs


def build_within_assay_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Build within-assay MMP pairs using Tanimoto similarity."""
    all_pairs = []
    assays = df['assay_chembl_id'].unique()

    for assay_idx, assay in enumerate(assays):
        assay_df = df[df['assay_chembl_id'] == assay].reset_index(drop=True)
        if len(assay_df) < 2:
            continue

        smiles_list = assay_df['canonical_smiles'].tolist()
        mmp_pairs = generate_mmp_pairs(smiles_list)

        for idx_a, idx_b, sim, edit_smiles in mmp_pairs:
            row_a = assay_df.iloc[idx_a]
            row_b = assay_df.iloc[idx_b]
            all_pairs.append({
                'mol_a': row_a['canonical_smiles'],
                'mol_b': row_b['canonical_smiles'],
                'mol_a_id': row_a['molecule_chembl_id'],
                'mol_b_id': row_b['molecule_chembl_id'],
                'edit_smiles': edit_smiles,
                'value_a': row_a['pchembl_value'],
                'value_b': row_b['pchembl_value'],
                'delta': row_b['pchembl_value'] - row_a['pchembl_value'],
                'assay_id_a': assay,
                'assay_id_b': assay,
                'is_within_assay': True,
                'assay_pair_id': f"{assay}_{assay}",
                'pair_method': 'tanimoto_mmp',
                'target_chembl_id': row_a['target_chembl_id'],
                'property_name': row_a['target_chembl_id'],
                'curation_level': 'minimal',
                'tanimoto_similarity': sim,
            })

        if (assay_idx + 1) % 10 == 0:
            print(f"  Processed {assay_idx + 1}/{len(assays)} assays, {len(all_pairs)} pairs so far")

    return pd.DataFrame(all_pairs)


def build_cross_assay_self_pairs(df: pd.DataFrame, overlapping_pairs: list) -> pd.DataFrame:
    """Build cross-assay self-pairs: same molecule, two different assays."""
    all_pairs = []

    for pair_info in overlapping_pairs:
        a1, a2 = pair_info['assay_id_1'], pair_info['assay_id_2']
        target = pair_info['target_chembl_id']

        df_a1 = df[(df['assay_chembl_id'] == a1)].set_index('molecule_chembl_id')
        df_a2 = df[(df['assay_chembl_id'] == a2)].set_index('molecule_chembl_id')

        shared = df_a1.index.intersection(df_a2.index)

        for mol_id in shared:
            row1 = df_a1.loc[mol_id]
            row2 = df_a2.loc[mol_id]

            # Handle case where mol_id appears multiple times
            if isinstance(row1, pd.DataFrame):
                row1 = row1.iloc[0]
            if isinstance(row2, pd.DataFrame):
                row2 = row2.iloc[0]

            all_pairs.append({
                'mol_a': row1['canonical_smiles'],
                'mol_b': row2['canonical_smiles'] if isinstance(row2['canonical_smiles'], str) else row1['canonical_smiles'],
                'mol_a_id': mol_id,
                'mol_b_id': mol_id,
                'edit_smiles': 'self_pair',
                'value_a': float(row1['pchembl_value']),
                'value_b': float(row2['pchembl_value']),
                'delta': float(row2['pchembl_value']) - float(row1['pchembl_value']),
                'assay_id_a': a1,
                'assay_id_b': a2,
                'is_within_assay': False,
                'assay_pair_id': f"{a1}_{a2}",
                'pair_method': 'cross_assay_self',
                'target_chembl_id': target,
                'property_name': target,
                'curation_level': 'minimal',
                'tanimoto_similarity': 1.0,
            })

    return pd.DataFrame(all_pairs)


def main():
    print("=" * 70)
    print("ChEMBL Overlapping Assay Data Extraction (API)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch activities for all targets
    all_activities = []
    for i, target in enumerate(TARGET_IDS):
        print(f"\n[{i+1}/{len(TARGET_IDS)}] Fetching IC50 data for {target}...")
        try:
            df = fetch_target_activities(target)
            n_assays = df['assay_chembl_id'].nunique() if len(df) > 0 else 0
            print(f"  → {len(df)} activities across {n_assays} assays")
            all_activities.append(df)
        except Exception as e:
            print(f"  → ERROR: {e}")
            continue

    activities_df = pd.concat(all_activities, ignore_index=True)
    print(f"\nTotal activities: {len(activities_df)}")
    print(f"Total unique molecules: {activities_df['molecule_chembl_id'].nunique()}")
    print(f"Total assays: {activities_df['assay_chembl_id'].nunique()}")

    # Save activities
    activities_path = OUTPUT_DIR / 'activities_minimal.csv'
    activities_df.to_csv(activities_path, index=False)
    print(f"Saved activities to {activities_path}")

    # Step 2: Find Goldilocks assays
    print("\nFinding Goldilocks assays (20-100 compounds)...")
    goldilocks_df = find_goldilocks_assays(activities_df)
    n_goldilocks = goldilocks_df['assay_chembl_id'].nunique()
    print(f"  → {n_goldilocks} Goldilocks assays")

    # Step 3: Find overlapping assay pairs
    print("\nFinding overlapping assay pairs (>= 5 shared compounds)...")
    overlapping_pairs = find_overlapping_assay_pairs(goldilocks_df)
    print(f"  → {len(overlapping_pairs)} overlapping assay pairs")

    if len(overlapping_pairs) == 0:
        print("\nNo overlapping pairs found. Relaxing constraints...")
        # Try with ALL assays (not just Goldilocks)
        overlapping_pairs = find_overlapping_assay_pairs(activities_df)
        print(f"  → {len(overlapping_pairs)} overlapping pairs (relaxed)")
        goldilocks_df = activities_df  # Use all data

    # Save assay pairs
    pairs_info_df = pd.DataFrame([
        {k: v for k, v in p.items() if k != 'shared_molecules'}
        for p in overlapping_pairs
    ])
    pairs_info_path = OUTPUT_DIR / 'assay_pairs_minimal.csv'
    pairs_info_df.to_csv(pairs_info_path, index=False)
    print(f"Saved assay pairs to {pairs_info_path}")

    # Step 4: Build within-assay MMP pairs
    print("\nBuilding within-assay pairs (Tanimoto >= 0.7)...")
    within_pairs = build_within_assay_pairs(goldilocks_df)
    print(f"  → {len(within_pairs)} within-assay pairs")

    # Step 5: Build cross-assay self-pairs (for noise measurement)
    print("\nBuilding cross-assay self-pairs (same molecule, different assays)...")
    cross_pairs = build_cross_assay_self_pairs(goldilocks_df, overlapping_pairs)
    print(f"  → {len(cross_pairs)} cross-assay self-pairs")

    # Combine all pairs
    all_pairs = pd.concat([within_pairs, cross_pairs], ignore_index=True)
    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"  Within-assay: {all_pairs['is_within_assay'].sum()}")
    print(f"  Cross-assay: {(~all_pairs['is_within_assay']).sum()}")

    # Save pairs
    pairs_path = OUTPUT_DIR / 'overlapping_assay_pairs_minimal_mmp.csv'
    all_pairs.to_csv(pairs_path, index=False)
    print(f"Saved pairs to {pairs_path}")

    # Also save as the generic name
    all_pairs.to_csv(OUTPUT_DIR / 'overlapping_assay_pairs.csv', index=False)

    # Save molecule-level pIC50 table
    mol_pic50 = goldilocks_df[['molecule_chembl_id', 'canonical_smiles', 'pchembl_value',
                                'assay_chembl_id', 'target_chembl_id']].copy()
    mol_pic50.to_csv(OUTPUT_DIR / 'molecule_pIC50_minimal.csv', index=False)

    # Summary statistics
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Targets queried: {len(TARGET_IDS)}")
    print(f"Total activities: {len(activities_df)}")
    print(f"Goldilocks assays: {n_goldilocks}")
    print(f"Overlapping assay pairs: {len(overlapping_pairs)}")
    print(f"Within-assay MMP pairs: {len(within_pairs)}")
    print(f"Cross-assay self-pairs: {len(cross_pairs)}")

    if len(cross_pairs) > 0:
        abs_deltas = cross_pairs['delta'].abs()
        print(f"\nCross-assay noise statistics:")
        print(f"  Mean |delta|: {abs_deltas.mean():.3f} log units")
        print(f"  Median |delta|: {abs_deltas.median():.3f} log units")
        print(f"  Std: {abs_deltas.std():.3f}")
        print(f"  f > 0.3: {(abs_deltas > 0.3).mean():.1%}")
        print(f"  f > 1.0: {(abs_deltas > 1.0).mean():.1%}")

    if len(within_pairs) > 0:
        within_abs = within_pairs['delta'].abs()
        print(f"\nWithin-assay pair statistics:")
        print(f"  Mean |delta|: {within_abs.mean():.3f} log units")
        print(f"  Std: {within_abs.std():.3f}")
        print(f"  f > 0.3: {(within_abs > 0.3).mean():.1%}")
        print(f"  f > 1.0: {(within_abs > 1.0).mean():.1%}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
