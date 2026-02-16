#!/usr/bin/env python
"""
Data extraction CLI for generating molecular pairs from various sources.

Currently supports:
- ChEMBL: Small molecule bioactivity data

Future support:
- Other small molecule databases (PubChem, ZINC, etc.)
- Antibody databases (SAbDab, OAS, etc.)
- RNA databases (RNAcentral, Rfam, etc.)

Usage:
    # Extract 10K molecules from ChEMBL with single-cut MMPs
    python extract_data.py chembl --n-molecules 10000 --max-cuts 1

    # Extract with specific properties only
    python extract_data.py chembl --n-molecules 50000 --properties IC50_CHEMBL1862 Ki_CHEMBL220

    # Skip download if data already exists
    python extract_data.py chembl --skip-download --n-molecules 10000

For parallelization on external machines:
    # Run multiple extractions with different parameters
    python extract_data.py chembl --n-molecules 100000 --max-cuts 1 &
    python extract_data.py chembl --n-molecules 100000 --max-cuts 2 &
    wait
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.data.chembl_extractor import ChEMBLPairExtractor, ChEMBLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_chembl(args):
    """Extract pairs from ChEMBL."""

    config = ChEMBLConfig(
        # ChEMBL settings
        chembl_version=args.chembl_version,
        n_molecules=args.n_molecules,
        activity_types=args.activity_types,

        # MMP settings
        max_cuts=args.max_cuts,
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity,

        # Property filtering
        property_filter=args.properties,
        exclude_computed=args.exclude_computed,

        # Checkpointing
        checkpoint_every=args.checkpoint_every,
        resume_from_checkpoint=not args.no_resume,

        # Paths
        data_dir=Path(args.data_dir),
        output_name=args.output_name or "chembl_pairs"
    )

    print()
    print("=" * 80)
    print("ChEMBL PAIR EXTRACTION")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Molecules: {config.n_molecules:,}")
    print(f"  Activity types: {', '.join(config.activity_types)}")
    print(f"  Max cuts: {config.max_cuts}")
    print(f"  Max MW delta: {config.max_mw_delta}")
    print(f"  Min similarity: {config.min_similarity}")
    if config.property_filter:
        print(f"  Properties: {', '.join(config.property_filter)}")
    if config.exclude_computed:
        print(f"  Exclude computed: Yes")
    print()
    print(f"Output:")
    print(f"  Directory: {config.output_dir}")
    print(f"  File: {config.pairs_output.name}")
    print()
    print("=" * 80)
    print()

    # Run extraction
    extractor = ChEMBLPairExtractor(config)
    pairs_df = extractor.run(skip_download=args.skip_download)

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Results:")
    print(f"  Total pairs: {len(pairs_df):,}")
    print(f"  Unique molecule pairs: {pairs_df[['mol_a', 'mol_b']].drop_duplicates().shape[0]:,}")
    print(f"  Unique edits: {pairs_df['edit_smiles'].nunique():,}")
    print(f"  Properties: {pairs_df['property_name'].nunique():,}")
    print()
    print(f"Output: {config.pairs_output}")
    print()
    print("Sample pairs:")
    print(pairs_df.head(5).to_string())
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract paired data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='source', help='Data source')

    # ChEMBL extractor
    chembl_parser = subparsers.add_parser('chembl', help='Extract from ChEMBL database')

    # ChEMBL download settings
    chembl_parser.add_argument('--chembl-version', type=str, default=None,
                              help='ChEMBL version (default: latest)')
    chembl_parser.add_argument('--n-molecules', type=int, default=10000,
                              help='Number of molecules to extract (default: 10000)')
    chembl_parser.add_argument('--activity-types', nargs='+',
                              default=['IC50', 'Ki', 'EC50', 'Kd'],
                              help='Activity types to include (default: IC50 Ki EC50 Kd)')

    # MMP settings
    chembl_parser.add_argument('--max-cuts', type=int, default=1,
                              help='Maximum cuts for MMP (default: 1)')
    chembl_parser.add_argument('--max-mw-delta', type=float, default=200.0,
                              help='Maximum MW difference (default: 200)')
    chembl_parser.add_argument('--min-similarity', type=float, default=0.4,
                              help='Minimum Tanimoto similarity (default: 0.4)')

    # Property filtering
    chembl_parser.add_argument('--properties', nargs='+', default=None,
                              help='Only extract these properties (default: all)')
    chembl_parser.add_argument('--exclude-computed', action='store_true',
                              help='Exclude computed properties (mw, logp, etc.)')

    # Checkpointing
    chembl_parser.add_argument('--checkpoint-every', type=int, default=1000,
                              help='Save checkpoint every N cores (default: 1000)')
    chembl_parser.add_argument('--no-resume', action='store_true',
                              help='Start fresh, ignore checkpoints')

    # Output
    chembl_parser.add_argument('--data-dir', type=str, default='data/chembl',
                              help='Data directory (default: data/chembl)')
    chembl_parser.add_argument('--output-name', type=str, default=None,
                              help='Output filename prefix (default: chembl_pairs)')

    # Execution
    chembl_parser.add_argument('--skip-download', action='store_true',
                              help='Skip download, use existing data')

    args = parser.parse_args()

    if args.source == 'chembl':
        extract_chembl(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
