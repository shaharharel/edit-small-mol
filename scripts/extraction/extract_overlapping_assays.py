#!/usr/bin/env python
"""
Extract overlapping assay data from ChEMBL for cross-lab noise analysis.

Implements Landrum & Riniker (JCIM 2024) methodology to identify assays
measuring the same target with shared compounds, then generates
within-assay and cross-assay molecular pairs.

Usage:
    # Minimal curation, MMP pairs only
    python scripts/extraction/extract_overlapping_assays.py

    # Maximal curation, all pair methods
    python scripts/extraction/extract_overlapping_assays.py \
        --curation maximal \
        --pair-methods mmp tanimoto scaffold

    # Custom assay size range
    python scripts/extraction/extract_overlapping_assays.py \
        --min-compounds 10 --max-compounds 200 \
        --min-shared 3
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import logging

from src.data.overlapping_assay_extractor import (
    OverlappingAssayConfig,
    OverlappingAssayExtractor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract overlapping assay data from ChEMBL"
    )

    parser.add_argument(
        "--curation",
        choices=["minimal", "maximal"],
        default="minimal",
        help="Curation level (default: minimal)",
    )
    parser.add_argument(
        "--activity-types",
        nargs="+",
        default=["IC50", "Ki"],
        help="Activity types to extract (default: IC50 Ki)",
    )
    parser.add_argument(
        "--min-compounds",
        type=int,
        default=20,
        help="Min compounds per assay (default: 20)",
    )
    parser.add_argument(
        "--max-compounds",
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=100,
        help="Max compounds per assay (default: 100, use 'none' for no limit)",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=5,
        help="Min shared compounds between assay pairs (default: 5)",
    )
    parser.add_argument(
        "--pair-methods",
        nargs="+",
        default=["mmp"],
        choices=["mmp", "tanimoto", "scaffold"],
        help="Pair generation methods (default: mmp)",
    )
    parser.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=0.7,
        help="Tanimoto similarity threshold (default: 0.7)",
    )
    parser.add_argument(
        "--max-mw-delta",
        type=float,
        default=200.0,
        help="Max MW difference for MMP pairs (default: 200)",
    )
    parser.add_argument(
        "--max-cuts",
        type=int,
        default=1,
        help="Max bond cuts for MMP (default: 1)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/overlapping_assays",
        help="Output directory (default: data/overlapping_assays)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="data/chembl_db",
        help="ChEMBL database directory (default: data/chembl_db)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use cached activities",
    )

    args = parser.parse_args()

    config = OverlappingAssayConfig(
        curation_level=args.curation,
        activity_types=args.activity_types,
        min_compounds_per_assay=args.min_compounds,
        max_compounds_per_assay=args.max_compounds,
        min_shared_compounds=args.min_shared,
        pair_methods=args.pair_methods,
        tanimoto_threshold=args.tanimoto_threshold,
        max_mw_delta=args.max_mw_delta,
        max_cuts=args.max_cuts,
        data_dir=Path(args.data_dir),
        db_dir=Path(args.db_dir),
        output_name="overlapping_assay_pairs",
    )

    extractor = OverlappingAssayExtractor(config)
    pairs_df = extractor.run(skip_download=args.skip_download)

    print()
    print("=" * 70)
    print(" EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Pairs: {len(pairs_df):,}")
    print(f"  Output: {config.pairs_output}")
    print(f"  Molecule pIC50: {config.molecule_properties_file}")
    print(f"  Assay pairs: {config.assay_pairs_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
