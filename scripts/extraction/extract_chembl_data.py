"""
Extract molecules and bioactivity from ChEMBL database.

This script queries the ChEMBL database to extract:
1. Top N targets by molecule count
2. All molecules for those targets
3. All bioactivity measurements

Outputs:
- molecules.csv: All molecules with computed properties
- bioactivity.csv: All bioactivity measurements (long format)
- target_molecule_counts.csv: Target statistics for load balancing

Usage:
    python scripts/extract_chembl_data.py \
        --top-n 10 \
        --output-dir data/chembl \
        --database chembl_34
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from src.data.chembl_extractor import ChEMBLExtractor, ChEMBLConfig

logger = logging.getLogger(__name__)


def extract_chembl_data(
    top_n: int = 10,
    min_molecules_per_target: int = 1000,
    database_name: str = 'chembl_34',
    output_dir: str = 'data/chembl',
    specific_targets: list = None,
    target_mix: dict = None
):
    """
    Extract molecules and bioactivity from ChEMBL.

    Args:
        top_n: Number of top targets to extract
        min_molecules_per_target: Minimum molecules per target
        database_name: ChEMBL database name
        output_dir: Output directory
        specific_targets: List of specific target ChEMBL IDs (optional)
        target_mix: Dict of target classes to counts (optional)

    Returns:
        Tuple of (molecules_df, bioactivity_df, target_counts_df)
    """
    logger.info("=" * 70)
    logger.info(" CHEMBL DATA EXTRACTION")
    logger.info("=" * 70)
    logger.info(f" Database: {database_name}")
    logger.info(f" Top N targets: {top_n}")
    logger.info(f" Min molecules per target: {min_molecules_per_target}")
    if specific_targets:
        logger.info(f" Specific targets: {specific_targets}")
    if target_mix:
        logger.info(f" Target mix: {target_mix}")
    logger.info("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure ChEMBL extractor
    config = ChEMBLConfig(
        database_name=database_name,
        top_n_targets=top_n,
        min_molecules_per_label=min_molecules_per_target,
        specific_targets=specific_targets,
        target_mix=target_mix
    )

    # Extract data
    logger.info("\nConnecting to ChEMBL database...")
    extractor = ChEMBLExtractor(config)

    logger.info("Extracting molecules and bioactivity...")
    molecules_df, bioactivity_df = extractor.extract_data()

    logger.info(f"\n✓ Extracted {len(molecules_df):,} unique molecules")
    logger.info(f"✓ Extracted {len(bioactivity_df):,} measurements")

    # Get target statistics
    bioactivity_only = bioactivity_df[bioactivity_df['target_name'] != 'computed'].copy()

    target_counts = bioactivity_only.groupby(['target_chembl_id', 'target_name']).agg(
        molecule_count=('chembl_id', 'nunique'),
        measurement_count=('chembl_id', 'count')
    ).reset_index()

    target_counts = target_counts.sort_values('molecule_count', ascending=False)

    logger.info(f"\nTarget Statistics:")
    for _, row in target_counts.iterrows():
        logger.info(
            f"  {row['target_chembl_id']:12s}: {row['target_name']:50s} "
            f"({row['molecule_count']:,} molecules, {row['measurement_count']:,} measurements)"
        )

    # Save results
    logger.info("\nSaving results...")

    molecules_file = output_path / 'molecules.csv'
    bioactivity_file = output_path / 'bioactivity.csv'
    target_counts_file = output_path / 'target_molecule_counts.csv'

    molecules_df.to_csv(molecules_file, index=False)
    logger.info(f"  ✓ Saved molecules to: {molecules_file}")

    bioactivity_df.to_csv(bioactivity_file, index=False)
    logger.info(f"  ✓ Saved bioactivity to: {bioactivity_file}")

    target_counts.to_csv(target_counts_file, index=False)
    logger.info(f"  ✓ Saved target counts to: {target_counts_file}")

    logger.info("\n" + "=" * 70)
    logger.info(" EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f" Total molecules: {len(molecules_df):,}")
    logger.info(f" Total measurements: {len(bioactivity_df):,}")
    logger.info(f" Total targets: {len(target_counts)}")
    logger.info("\n Next steps:")
    logger.info("  1. (Optional) Analyze overlap:")
    logger.info(f"     python scripts/analyze_target_overlap.py \\")
    logger.info(f"         --molecules-file {molecules_file} \\")
    logger.info(f"         --bioactivity-file {bioactivity_file}")
    logger.info("")
    logger.info("  2. Extract MMP pairs:")
    logger.info(f"     python -m src.data.parallel_extraction \\")
    logger.info(f"         --molecules-file {molecules_file} \\")
    logger.info(f"         --bioactivity-file {bioactivity_file} \\")
    logger.info(f"         --target-counts-file {target_counts_file} \\")
    logger.info(f"         --max-cuts 3 \\")
    logger.info(f"         --output-dir data/pairs")
    logger.info("=" * 70)

    return molecules_df, bioactivity_df, target_counts


def main():
    parser = argparse.ArgumentParser(description="Extract ChEMBL data")
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top targets (default: 10)')
    parser.add_argument('--min-molecules', type=int, default=1000,
                       help='Minimum molecules per target (default: 1000)')
    parser.add_argument('--database', default='chembl_34',
                       help='ChEMBL database name (default: chembl_34)')
    parser.add_argument('--output-dir', default='data/chembl',
                       help='Output directory (default: data/chembl)')
    parser.add_argument('--specific-targets', nargs='+',
                       help='Specific target ChEMBL IDs (e.g., CHEMBL203 CHEMBL1862)')
    parser.add_argument('--target-mix', type=str,
                       help='Target mix as JSON string (e.g., \'{"Kinase": 5, "GPCR": 3}\')')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Parse target mix if provided
    target_mix = None
    if args.target_mix:
        import json
        target_mix = json.loads(args.target_mix)

    extract_chembl_data(
        top_n=args.top_n,
        min_molecules_per_target=args.min_molecules,
        database_name=args.database,
        output_dir=args.output_dir,
        specific_targets=args.specific_targets,
        target_mix=target_mix
    )


if __name__ == '__main__':
    main()
