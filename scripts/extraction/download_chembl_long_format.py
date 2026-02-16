#!/usr/bin/env python
"""
Download ChEMBL in optimized long format.

This gets ALL molecules with ANY bioactivity (no waste!),
then downloads ALL their properties.

Usage:
    python download_chembl_long_format.py --max-molecules 1000000
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import logging
from src.data.legacy.chembl_long_format import ChEMBLLongFormat

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download ChEMBL optimized for long-format pairs"
    )

    parser.add_argument('--max-molecules', type=int, default=None,
                       help='Maximum molecules (default: no limit)')
    parser.add_argument('--max-mw', type=float, default=800,
                       help='Max MW (default: 800)')
    parser.add_argument('--min-mw', type=float, default=100,
                       help='Min MW (default: 100)')
    parser.add_argument('--activity-types', nargs='+',
                       default=['IC50', 'Ki', 'EC50', 'Kd'],
                       help='Activity types (default: IC50 Ki EC50 Kd)')
    parser.add_argument('--all-activities', action='store_true',
                       help='Include ALL activity types with pchembl_value (ignores --activity-types)')
    parser.add_argument('--top-targets', type=int, default=None,
                       help='Only get molecules from top N targets (RECOMMENDED: 100-200 for best pair density)')
    parser.add_argument('--computed-properties', nargs='+',
                       default=['mw', 'alogp', 'hbd', 'hba', 'psa', 'rtb', 'aromatic_rings', 'qed_weighted'],
                       help='Computed properties to include (default: mw alogp hbd hba psa rtb aromatic_rings qed_weighted)')
    parser.add_argument('--download-target-sequences', action='store_true',
                       help='Download protein sequences for all targets (default: False)')
    # Default db-dir relative to project root
    default_db_dir = project_root / 'data' / 'chembl_db'
    parser.add_argument('--db-dir', type=str, default=str(default_db_dir),
                       help=f'Directory containing ChEMBL database (default: {default_db_dir})')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" CHEMBL LONG-FORMAT DOWNLOAD")
    print("=" * 70)
    print()

    if args.top_targets:
        print(f" OPTIMIZED MODE: Top {args.top_targets} targets")
        print(" This focuses on well-tested targets for maximum pair density!")
    else:
        print(" ALL TARGETS MODE: Any bioactivity")
        print(" This gets all molecules, lower pair density.")

    print()
    print(f" Configuration:")
    print(f"   • Max molecules: {args.max_molecules:,}" if args.max_molecules else "   • Max molecules: no limit")
    print(f"   • MW range: {args.min_mw}-{args.max_mw}")
    if args.all_activities:
        print(f"   • Activity types: ALL (any with pchembl_value)")
    else:
        print(f"   • Activity types: {', '.join(args.activity_types)}")

    if args.top_targets:
        print(f"   • Top targets: {args.top_targets}")

    print()
    print("=" * 70)
    print()

    response = input("Start download? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print()

    downloader = ChEMBLLongFormat(db_dir=args.db_dir)

    # If --all-activities, pass None to disable activity type filtering
    activity_types = None if args.all_activities else args.activity_types

    result = downloader.download_complete(
        max_molecules=args.max_molecules,
        max_mw=args.max_mw,
        min_mw=args.min_mw,
        activity_types=activity_types,
        top_targets=args.top_targets,
        computed_properties=args.computed_properties,
        download_target_sequences=args.download_target_sequences
    )

    # Unpack result (may include targets_df if download_target_sequences=True)
    if args.download_target_sequences:
        molecules_df, bioactivity_df, property_lookup, targets_df = result
    else:
        molecules_df, bioactivity_df, property_lookup = result

    print()
    print("=" * 70)
    print(" NEXT STEP")
    print("=" * 70)
    print()
    print("Generate pairs with:")
    print()
    print(f"  python build_pairs_long_format.py \\")
    print(f"      --molecules-file data/chembl_bulk/chembl_molecules_{len(molecules_df)}.csv \\")
    print(f"      --bioactivity-file data/chembl_bulk/chembl_bioactivity_long_{len(molecules_df)}.csv")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
