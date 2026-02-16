"""
Prepare data for parallel MMP extraction.

This script:
1. Queries ChEMBL to get target molecule counts
2. Selects top N targets
3. Prepares target counts file for parallel extraction
4. Provides recommended threading configuration

Usage:
    python scripts/prepare_parallel_extraction.py \
        --top-n 10 \
        --output-dir data/pairs \
        --num-threads 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from src.data.chembl_extractor import ChEMBLExtractor, ChEMBLConfig

logger = logging.getLogger(__name__)


def get_target_molecule_counts(
    extractor: ChEMBLExtractor,
    top_n: int = 10,
    min_molecules: int = 1000
) -> pd.DataFrame:
    """
    Query ChEMBL to get target molecule counts.

    Args:
        extractor: ChEMBLExtractor instance
        top_n: Number of top targets to select
        min_molecules: Minimum molecules per target

    Returns:
        DataFrame with columns: target_chembl_id, target_name, molecule_count
    """
    logger.info(f"Querying ChEMBL for target molecule counts (top {top_n})...")

    # Query to get molecule counts per target
    query = """
    SELECT
        target_dictionary.chembl_id AS target_chembl_id,
        target_dictionary.pref_name AS target_name,
        COUNT(DISTINCT compound_structures.canonical_smiles) AS molecule_count
    FROM activities
    JOIN assays ON activities.assay_id = assays.assay_id
    JOIN target_dictionary ON assays.tid = target_dictionary.tid
    JOIN compound_structures ON activities.molregno = compound_structures.molregno
    WHERE
        activities.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
        AND activities.pchembl_value IS NOT NULL
        AND target_dictionary.target_type = 'SINGLE PROTEIN'
        AND compound_structures.canonical_smiles IS NOT NULL
    GROUP BY
        target_dictionary.chembl_id,
        target_dictionary.pref_name
    HAVING
        COUNT(DISTINCT compound_structures.canonical_smiles) >= %(min_molecules)s
    ORDER BY
        molecule_count DESC
    LIMIT %(top_n)s
    """

    df = extractor.chembl.query(
        query,
        params={'min_molecules': min_molecules, 'top_n': top_n}
    )

    logger.info(f"Found {len(df)} targets:")
    for _, row in df.iterrows():
        logger.info(
            f"  {row['target_chembl_id']}: {row['target_name']} "
            f"({row['molecule_count']:,} molecules)"
        )

    return df


def recommend_threading_config(
    target_counts: pd.DataFrame,
    molecules_per_batch: int = 15000
) -> dict:
    """
    Recommend threading configuration based on target counts.

    Args:
        target_counts: DataFrame with target molecule counts
        molecules_per_batch: Target molecules per batch

    Returns:
        Dict with recommendations
    """
    total_molecules = target_counts['molecule_count'].sum()
    max_molecules = target_counts['molecule_count'].max()
    min_molecules = target_counts['molecule_count'].min()

    # Estimate batches using greedy bin-packing simulation
    sorted_targets = target_counts.sort_values('molecule_count', ascending=False)
    batches = []
    current_batch = 0

    for _, row in sorted_targets.iterrows():
        mol_count = row['molecule_count']

        if current_batch + mol_count > molecules_per_batch and current_batch > 0:
            batches.append(current_batch)
            current_batch = 0

        current_batch += mol_count

    if current_batch > 0:
        batches.append(current_batch)

    num_batches = len(batches)

    # Recommended threads = min(num_batches, num_cpu_cores)
    import os
    num_cores = os.cpu_count() or 8
    recommended_threads = min(num_batches, num_cores)

    # Estimate time
    # Empirical: ~500K pairs/hour/core with max_cuts=3
    total_pairs_estimate = total_molecules * 10  # Very rough estimate
    hours_estimate = total_pairs_estimate / (500_000 * recommended_threads)

    return {
        'total_molecules': total_molecules,
        'num_batches': num_batches,
        'batch_sizes': batches,
        'recommended_threads': recommended_threads,
        'available_cores': num_cores,
        'estimated_hours': hours_estimate,
        'molecules_per_batch': molecules_per_batch
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare parallel MMP extraction")
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top targets to select (default: 10)')
    parser.add_argument('--min-molecules', type=int, default=1000,
                       help='Minimum molecules per target (default: 1000)')
    parser.add_argument('--molecules-per-batch', type=int, default=15000,
                       help='Target molecules per batch (default: 15000)')
    parser.add_argument('--output-dir', default='data/pairs',
                       help='Output directory')
    parser.add_argument('--chembl-version', default='chembl_34',
                       help='ChEMBL database version (default: chembl_34)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Connect to ChEMBL
    config = ChEMBLConfig(
        database_name=args.chembl_version,
        top_n_targets=args.top_n,
        min_molecules_per_label=args.min_molecules
    )

    extractor = ChEMBLExtractor(config)

    # Step 2: Get target molecule counts
    target_counts = get_target_molecule_counts(
        extractor,
        top_n=args.top_n,
        min_molecules=args.min_molecules
    )

    # Save target counts
    target_counts_file = output_dir / "target_molecule_counts.csv"
    target_counts.to_csv(target_counts_file, index=False)
    logger.info(f"\n✓ Saved target counts to: {target_counts_file}")

    # Step 3: Generate recommendations
    logger.info("\n" + "=" * 70)
    logger.info(" THREADING CONFIGURATION RECOMMENDATIONS")
    logger.info("=" * 70)

    recommendations = recommend_threading_config(
        target_counts,
        molecules_per_batch=args.molecules_per_batch
    )

    logger.info(f" Total molecules: {recommendations['total_molecules']:,}")
    logger.info(f" Estimated batches: {recommendations['num_batches']}")
    logger.info(f" Batch sizes: {[f'{b:,}' for b in recommendations['batch_sizes']]}")
    logger.info(f" Available CPU cores: {recommendations['available_cores']}")
    logger.info(f" Recommended threads: {recommendations['recommended_threads']}")
    logger.info(f" Estimated time: {recommendations['estimated_hours']:.1f} hours")
    logger.info("=" * 70)

    # Step 4: Print next steps
    logger.info("\nNEXT STEPS:")
    logger.info("1. Extract molecules and bioactivity:")
    logger.info(f"   python scripts/extract_chembl_data.py \\")
    logger.info(f"       --top-n {args.top_n} \\")
    logger.info(f"       --output-dir {args.output_dir}")
    logger.info("")
    logger.info("2. Run parallel MMP extraction:")
    logger.info(f"   python -m src.data.parallel_extraction \\")
    logger.info(f"       --molecules-file {args.output_dir}/molecules.csv \\")
    logger.info(f"       --bioactivity-file {args.output_dir}/bioactivity.csv \\")
    logger.info(f"       --target-counts-file {target_counts_file} \\")
    logger.info(f"       --num-threads {recommendations['recommended_threads']} \\")
    logger.info(f"       --molecules-per-batch {args.molecules_per_batch} \\")
    logger.info(f"       --max-cuts 3 \\")
    logger.info(f"       --output-dir {args.output_dir}")
    logger.info("")

    # Print RAM recommendations
    total_molecules = recommendations['total_molecules']
    if total_molecules < 30000:
        ram_gb = 8
    elif total_molecules < 60000:
        ram_gb = 16
    elif total_molecules < 100000:
        ram_gb = 24
    else:
        ram_gb = 32

    logger.info("MACHINE REQUIREMENTS:")
    logger.info(f" Recommended RAM: {ram_gb} GB")
    logger.info(f" Recommended CPU cores: {recommendations['recommended_threads']}")
    logger.info(f" Storage: ~{total_molecules * 0.5 / 1000:.1f} GB for pairs")
    logger.info("")


if __name__ == '__main__':
    main()
