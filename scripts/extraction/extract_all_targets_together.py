"""
Extract MMP pairs for all targets together (single pass).

This approach processes all targets in a single MMP extraction to avoid
any potential duplicate pair computation across targets.

Strategy:
- Query all target bioactivity together
- Run single MMP extraction with all properties
- Each pair computed exactly once
- Multiple property labels per pair stored in long format

Trade-offs:
+ No duplicate computation
+ Slightly more memory efficient (single core index)
- Cannot parallelize across targets (but can still use multi-core RDKit)
- All-or-nothing (no partial results)

Usage:
    python scripts/extract_all_targets_together.py \\
        --top-n 10 \\
        --max-cuts 3 \\
        --output data/pairs/chembl_pairs_top10.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from src.data.chembl_extractor import ChEMBLExtractor, ChEMBLConfig
from src.data.mmp_long_format import LongFormatMMPExtractor

logger = logging.getLogger(__name__)


def extract_all_targets_together(
    top_n: int = 10,
    max_cuts: int = 3,
    max_mw_delta: float = 200,
    min_similarity: float = 0.4,
    output_file: str = "data/pairs/chembl_pairs.csv",
    checkpoint_dir: str = "data/pairs/checkpoints"
):
    """
    Extract MMP pairs for all targets in single pass.

    Args:
        top_n: Number of top targets to extract
        max_cuts: Maximum bond cuts (1, 2, or 3)
        max_mw_delta: Maximum MW difference
        min_similarity: Minimum Tanimoto similarity
        output_file: Output CSV file path
        checkpoint_dir: Directory for checkpoints
    """
    logger.info("=" * 70)
    logger.info(" ALL-TARGETS-TOGETHER MMP EXTRACTION")
    logger.info("=" * 70)
    logger.info(f" Top N targets: {top_n}")
    logger.info(f" Max cuts: {max_cuts}")
    logger.info(f" Max MW delta: {max_mw_delta}")
    logger.info(f" Min similarity: {min_similarity}")
    logger.info("=" * 70)

    # Step 1: Extract ChEMBL data
    logger.info("\nStep 1: Extracting ChEMBL data...")

    config = ChEMBLConfig(
        database_name='chembl_34',
        top_n_targets=top_n,
        max_cuts=max_cuts
    )

    extractor = ChEMBLExtractor(config)
    molecules_df, bioactivity_df = extractor.extract_data()

    logger.info(f"  Molecules: {len(molecules_df):,}")
    logger.info(f"  Bioactivity measurements: {len(bioactivity_df):,}")
    logger.info(f"  Targets: {bioactivity_df['target_chembl_id'].nunique()}")

    # Step 2: Extract MMP pairs (single pass, all targets)
    logger.info("\nStep 2: Extracting MMP pairs (single pass)...")

    mmp_extractor = LongFormatMMPExtractor(max_cuts=max_cuts)

    pairs_df = mmp_extractor.extract_pairs_long_format(
        molecules_df=molecules_df,
        bioactivity_df=bioactivity_df,
        max_mw_delta=max_mw_delta,
        min_similarity=min_similarity,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1000,
        resume_from_checkpoint=True,
        micro_batch_size=200
    )

    # Step 3: Save results
    logger.info("\nStep 3: Saving results...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_path, index=False)

    logger.info(f"  ✓ Saved to: {output_path}")

    # Step 4: Statistics
    logger.info("\n" + "=" * 70)
    logger.info(" EXTRACTION STATISTICS")
    logger.info("=" * 70)

    n_unique_pairs = pairs_df[['mol_a', 'mol_b']].drop_duplicates().shape[0]
    n_unique_edits = pairs_df['edit_smiles'].nunique()
    n_properties = pairs_df['property_name'].nunique()
    n_targets = pairs_df['target_chembl_id'].nunique()

    logger.info(f" Total rows: {len(pairs_df):,}")
    logger.info(f" Unique molecular pairs: {n_unique_pairs:,}")
    logger.info(f" Unique edits: {n_unique_edits:,}")
    logger.info(f" Unique properties: {n_properties}")
    logger.info(f" Unique targets: {n_targets}")
    logger.info(f" Avg properties per pair: {len(pairs_df) / n_unique_pairs:.2f}")

    # Check for any duplicates (should be none with this approach)
    duplicates = pairs_df.groupby(['mol_a', 'mol_b', 'property_name']).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        logger.warning(f" WARNING: Found {len(duplicates):,} duplicate pair-property combos")
        logger.info("   This can happen if same pair measured multiple times for same target")
    else:
        logger.info(" ✓ No duplicate pair-property combinations")

    logger.info("=" * 70)

    return pairs_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract MMP pairs for all targets together (single pass)"
    )
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top targets (default: 10)')
    parser.add_argument('--max-cuts', type=int, default=3,
                       help='Maximum bond cuts (default: 3)')
    parser.add_argument('--max-mw-delta', type=float, default=200,
                       help='Maximum MW delta (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                       help='Minimum similarity (default: 0.4)')
    parser.add_argument('--output', default='data/pairs/chembl_pairs.csv',
                       help='Output CSV file')
    parser.add_argument('--checkpoint-dir', default='data/pairs/checkpoints',
                       help='Checkpoint directory')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    extract_all_targets_together(
        top_n=args.top_n,
        max_cuts=args.max_cuts,
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity,
        output_file=args.output,
        checkpoint_dir=args.checkpoint_dir
    )

    logger.info("\n✓ Extraction complete!")


if __name__ == '__main__':
    main()
