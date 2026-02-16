#!/usr/bin/env python
"""
Build molecular pairs dataset in long format.

Long format structure:
    mol_a, mol_b, edit_id, core, from_smarts, to_smarts,
    property_name, value_a, value_b, delta

Benefits:
- No NaN/missing values
- Efficient storage
- Easy filtering by property
- SQL-friendly

Usage:
    python build_pairs_long_format.py \\
        --molecules-file data/chembl_bulk/chembl_molecules_1000000.csv \\
        --bioactivity-file data/chembl_bulk/chembl_bioactivity_long_1000000.csv
"""

import logging
import argparse
import pandas as pd
from pathlib import Path
from src.data.mmp_long_format import LongFormatMMPExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build pairs dataset in long format"
    )

    parser.add_argument('--molecules-file', required=True,
                       help='Molecules CSV file (from download step)')
    parser.add_argument('--bioactivity-file', required=True,
                       help='Bioactivity CSV file (from download step)')
    parser.add_argument('--output', default='data/pairs/chembl_pairs_long.csv',
                       help='Output file (default: data/pairs/chembl_pairs_long.csv)')
    parser.add_argument('--max-mw-delta', type=float, default=200,
                       help='Max MW delta (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                       help='Min similarity (default: 0.4)')
    parser.add_argument('--max-cuts', type=int, default=1,
                       help='Max cuts (default: 1 for simpler, closer edits; 2 for more complex)')
    parser.add_argument('--checkpoint-dir', default='data/pairs/checkpoints',
                       help='Checkpoint directory (default: data/pairs/checkpoints)')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                       help='Checkpoint every N cores (default: 1000)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoints')
    parser.add_argument('--properties', nargs='+', default=None,
                       help='Only extract pairs for these properties (e.g., --properties mw alogp IC50_CHEMBL1862)')
    parser.add_argument('--exclude-computed', action='store_true',
                       help='Exclude computed properties (mw, alogp, etc.) from pair extraction')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" LONG-FORMAT PAIRS GENERATION")
    print("=" * 70)
    print()
    print(f" Input files:")
    print(f"   • Molecules: {args.molecules_file}")
    print(f"   • Bioactivity: {args.bioactivity_file}")
    print()
    print(f" Parameters:")
    print(f"   • Max MW delta: {args.max_mw_delta}")
    print(f"   • Min similarity: {args.min_similarity}")
    print(f"   • Max cuts: {args.max_cuts}")
    print()
    print(f" Output: {args.output}")
    print()
    print("=" * 70)
    print()

    # Load data
    logger.info(f"Loading molecules...")
    molecules_df = pd.read_csv(args.molecules_file)
    logger.info(f"  ✓ Loaded {len(molecules_df):,} molecules")

    logger.info(f"Loading bioactivity...")
    bioactivity_df = pd.read_csv(args.bioactivity_file)
    logger.info(f"  ✓ Loaded {len(bioactivity_df):,} bioactivity measurements")
    logger.info("")

    # Extract pairs with checkpoint support
    extractor = LongFormatMMPExtractor(max_cuts=args.max_cuts)

    # Build property filter
    property_filter = None
    if args.properties:
        property_filter = set(args.properties)
    elif args.exclude_computed:
        # Exclude all computed properties, only keep bioactivity
        computed_props = {'mw', 'alogp', 'mw_freebase', 'hbd', 'hba', 'psa', 'rtb',
                         'aromatic_rings', 'heavy_atoms', 'qed_weighted',
                         'num_ro5_violations', 'np_likeness_score'}
        # Get all properties from bioactivity
        all_bio_props = set(bioactivity_df['property_name'].unique())
        property_filter = all_bio_props  # Only biological properties

    df_pairs = extractor.extract_pairs_long_format(
        molecules_df=molecules_df,
        bioactivity_df=bioactivity_df,
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from_checkpoint=not args.no_resume,
        property_filter=property_filter
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print(" SUCCESS!")
    print("=" * 70)
    print(f" Output: {output_path}")
    print(f" Total rows: {len(df_pairs):,}")
    print(f" Unique pairs: {df_pairs[['mol_a', 'mol_b']].drop_duplicates().shape[0]:,}")
    print(f" Unique edits: {df_pairs['edit_smiles'].nunique():,}")
    print(f" Unique properties: {df_pairs['property_name'].nunique():,}")
    print()
    print(" Example rows:")
    print(df_pairs.head(10).to_string())
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
