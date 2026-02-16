#!/usr/bin/env python3
"""
Extract molecules.csv and bioactivity.csv from ChEMBL SQLite database.

This script extracts the intermediate CSV files needed for MMP pair extraction,
without running the full MMP extraction pipeline.

Usage:
    # Extract top 5 targets
    python scripts/extraction/extract_chembl_csvs.py \\
        --db-path data/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 5 \\
        --output-dir data/chembl

    # Extract specific targets
    python scripts/extraction/extract_chembl_csvs.py \\
        --db-path data/chembl_db/chembl/36/chembl_36.db \\
        --specific-targets CHEMBL203 CHEMBL217 \\
        --output-dir data/chembl

    # Extract with sampling
    python scripts/extraction/extract_chembl_csvs.py \\
        --db-path data/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 10 \\
        --sample-size 5000 \\
        --output-dir data/chembl

Output:
    - {output-dir}/molecules.csv: chembl_id, smiles
    - {output-dir}/bioactivity.csv: chembl_id, property_name, value, target_name,
                                     target_chembl_id, doc_id, assay_id
"""

import argparse
import sqlite3
import logging
from pathlib import Path
import pandas as pd
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChEMBLCSVExtractor:
    """Extract molecules and bioactivity CSVs from ChEMBL SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def get_top_targets(self, top_n: int = 10, min_molecules: int = 100) -> pd.DataFrame:
        """Get top N targets by molecule count."""
        logger.info(f"Finding top {top_n} targets with ≥{min_molecules} molecules...")

        query = """
        SELECT
            td.chembl_id as target_chembl_id,
            td.pref_name as target_name,
            COUNT(DISTINCT cs.molregno) as molecule_count
        FROM target_dictionary td
        JOIN assays ass ON td.tid = ass.tid
        JOIN activities act ON ass.assay_id = act.assay_id
        JOIN compound_structures cs ON act.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
            AND act.pchembl_value IS NOT NULL
            AND td.target_type = 'SINGLE PROTEIN'
        GROUP BY td.chembl_id, td.pref_name
        HAVING COUNT(DISTINCT cs.molregno) >= ?
        ORDER BY molecule_count DESC
        LIMIT ?
        """

        df = pd.read_sql_query(query, self.conn, params=(min_molecules, top_n))
        logger.info(f"Found {len(df)} targets")

        for idx, row in df.iterrows():
            logger.info(f"  {idx+1}. {row['target_name']} ({row['target_chembl_id']}): "
                       f"{row['molecule_count']:,} molecules")

        return df

    def extract_molecules_for_targets(
        self,
        target_ids: List[str],
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract molecules for specific targets.

        Returns DataFrame with columns:
        - chembl_id: ChEMBL molecule ID (e.g., CHEMBL1234)
        - smiles: Canonical SMILES string
        """
        logger.info(f"Extracting molecules for {len(target_ids)} target(s)...")

        target_list = ','.join([f"'{tid}'" for tid in target_ids])

        query = f"""
        SELECT DISTINCT
            md.chembl_id,
            cs.canonical_smiles as smiles
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        JOIN activities act ON cs.molregno = act.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE cs.canonical_smiles IS NOT NULL
            AND act.pchembl_value IS NOT NULL
            AND td.chembl_id IN ({target_list})
        """

        if sample_size:
            query += f" LIMIT {sample_size}"

        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Extracted {len(df):,} unique molecules")

        return df

    def extract_bioactivity_for_molecules(
        self,
        molecule_ids: List[str]
    ) -> pd.DataFrame:
        """
        Extract bioactivity for specific molecules.

        Returns DataFrame with columns:
        - chembl_id: ChEMBL molecule ID
        - property_name: Always 'pchembl_value' (negative log of activity)
        - value: pChEMBL value (e.g., 6.5 = 10^-6.5 M = 316 nM)
        - target_name: Biological target name (e.g., "Epidermal growth factor receptor")
        - target_chembl_id: ChEMBL target ID (e.g., CHEMBL203)
        - doc_id: Publication/document ID
        - assay_id: Assay protocol ID
        """
        logger.info(f"Extracting bioactivity for {len(molecule_ids):,} molecules...")

        # Split into chunks to avoid SQL parameter limits
        chunk_size = 500
        all_bioactivity = []

        for i in range(0, len(molecule_ids), chunk_size):
            chunk = molecule_ids[i:i + chunk_size]
            mol_list = ','.join([f"'{mid}'" for mid in chunk])

            query = f"""
            SELECT
                md.chembl_id,
                'pchembl_value' as property_name,
                act.pchembl_value as value,
                td.pref_name as target_name,
                td.chembl_id as target_chembl_id,
                act.doc_id,
                act.assay_id
            FROM activities act
            JOIN molecule_dictionary md ON act.molregno = md.molregno
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN compound_structures cs ON act.molregno = cs.molregno
            WHERE cs.canonical_smiles IS NOT NULL
                AND act.pchembl_value IS NOT NULL
                AND md.chembl_id IN ({mol_list})
                AND td.target_type = 'SINGLE PROTEIN'
            """

            chunk_df = pd.read_sql_query(query, self.conn)
            all_bioactivity.append(chunk_df)

            if i > 0 and i % 5000 == 0:
                logger.info(f"  Processed {i:,}/{len(molecule_ids):,} molecules...")

        df = pd.concat(all_bioactivity, ignore_index=True)
        logger.info(f"Extracted {len(df):,} bioactivity measurements")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract molecules.csv and bioactivity.csv from ChEMBL SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--db-path', required=True,
                       help='Path to ChEMBL SQLite database')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for CSV files')

    # Target selection (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--top-n-targets', type=int,
                             help='Extract top N targets by molecule count')
    target_group.add_argument('--specific-targets', nargs='+',
                             help='Specific target ChEMBL IDs (e.g., CHEMBL203 CHEMBL217)')

    # Optional parameters
    parser.add_argument('--min-molecules', type=int, default=100,
                       help='Minimum molecules per target (for --top-n-targets)')
    parser.add_argument('--sample-size', type=int,
                       help='Limit number of molecules extracted')

    args = parser.parse_args()

    # Validate database exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ChEMBL CSV Extraction")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Extract data
    with ChEMBLCSVExtractor(args.db_path) as extractor:
        # Step 1: Get target IDs
        if args.top_n_targets:
            logger.info(f"Step 1: Finding top {args.top_n_targets} targets...")
            targets_df = extractor.get_top_targets(
                top_n=args.top_n_targets,
                min_molecules=args.min_molecules
            )
            target_ids = targets_df['target_chembl_id'].tolist()
        else:
            target_ids = args.specific_targets
            logger.info(f"Step 1: Using {len(target_ids)} specific target(s): {', '.join(target_ids)}")

        logger.info("")

        # Step 2: Extract molecules
        logger.info("Step 2: Extracting molecules...")
        molecules_df = extractor.extract_molecules_for_targets(
            target_ids=target_ids,
            sample_size=args.sample_size
        )

        if len(molecules_df) == 0:
            logger.error("No molecules extracted! Check target IDs and database.")
            return 1

        logger.info("")

        # Step 3: Extract bioactivity
        logger.info("Step 3: Extracting bioactivity...")
        molecule_ids = molecules_df['chembl_id'].tolist()
        bioactivity_df = extractor.extract_bioactivity_for_molecules(molecule_ids)

        if len(bioactivity_df) == 0:
            logger.error("No bioactivity extracted! Check molecule IDs.")
            return 1

        logger.info("")

    # Step 4: Save CSV files
    logger.info("Step 4: Saving CSV files...")

    molecules_file = output_dir / "molecules.csv"
    bioactivity_file = output_dir / "bioactivity.csv"

    molecules_df.to_csv(molecules_file, index=False)
    bioactivity_df.to_csv(bioactivity_file, index=False)

    logger.info(f"  Saved: {molecules_file}")
    logger.info(f"  Saved: {bioactivity_file}")
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Molecules: {len(molecules_df):,}")
    logger.info(f"Bioactivity measurements: {len(bioactivity_df):,}")
    logger.info(f"Avg measurements per molecule: {len(bioactivity_df)/len(molecules_df):.1f}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  {molecules_file}")
    logger.info(f"  {bioactivity_file}")
    logger.info("")
    logger.info("Next step: Run MMP pair extraction")
    logger.info(f"  python scripts/extraction/build_pairs_long_format.py \\")
    logger.info(f"      --molecules-file {molecules_file} \\")
    logger.info(f"      --bioactivity-file {bioactivity_file} \\")
    logger.info(f"      --output data/pairs/chembl_pairs.csv \\")
    logger.info(f"      --max-cuts 1")
    logger.info("")

    return 0


if __name__ == '__main__':
    exit(main())
