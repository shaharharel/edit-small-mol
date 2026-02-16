"""
Run complete ChEMBL MMP pair extraction pipeline.

This script orchestrates all 4 stages:
1. ChEMBL data extraction (or load from existing CSV files)
2. Target overlap analysis (optional)
3. Parallel MMP extraction
4. Results verification

The script intelligently skips stages if output files already exist.

Usage:
    # Extract from database
    python scripts/run_chembl_pair_extraction.py \
        --config configs/extraction_config.yaml

    # Use existing CSV files (skip stage 1)
    python scripts/run_chembl_pair_extraction.py \
        --molecules-file data/chembl/molecules.csv \
        --bioactivity-file data/chembl/bioactivity.csv \
        --output-dir data/pairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import yaml
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for ChEMBL pair extraction pipeline."""

    # ========== Input Sources (mutually exclusive) ==========
    # Option 1: Extract from database
    database_name: Optional[str] = 'chembl_34'
    database_host: Optional[str] = 'localhost'
    database_port: Optional[int] = 5432
    database_user: Optional[str] = None
    database_password: Optional[str] = None

    # Option 2: Use existing CSV files (skips database extraction)
    molecules_file: Optional[str] = None
    bioactivity_file: Optional[str] = None
    target_counts_file: Optional[str] = None

    # ========== Target Selection ==========
    top_n_targets: int = 10
    min_molecules_per_target: int = 1000
    specific_targets: Optional[List[str]] = None
    target_mix: Optional[Dict[str, int]] = None

    # ========== MMP Extraction Parameters ==========
    max_cuts: int = 3
    max_mw_delta: float = 200.0
    min_similarity: float = 0.4

    # ========== Parallelization ==========
    num_threads: Optional[int] = None  # None = auto-detect all cores
    molecules_per_batch: int = 15000

    # ========== Output Options ==========
    output_dir: str = 'data/pairs'
    chembl_cache_dir: str = 'data/chembl'

    # ========== Processing Options ==========
    include_computed_properties: bool = False
    deduplicate_pairs: bool = True
    run_overlap_analysis: bool = True

    # ========== Skip Stages (for resuming) ==========
    skip_chembl_extraction: bool = False
    skip_overlap_analysis: bool = False
    skip_mmp_extraction: bool = False

    # ========== Advanced ==========
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'ExtractionConfig':
        """Load configuration from YAML file."""
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_file: str):
        """Save configuration to YAML file."""
        import dataclasses
        config_dict = dataclasses.asdict(self)
        with open(yaml_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class PipelineRunner:
    """Orchestrates the complete extraction pipeline."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.start_time = None
        self.stage_times = {}

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.chembl_cache_dir).mkdir(parents=True, exist_ok=True)

        if config.checkpoint_dir:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the complete pipeline."""
        logger.info("=" * 70)
        logger.info(" CHEMBL MMP PAIR EXTRACTION PIPELINE")
        logger.info("=" * 70)
        logger.info(f" Output directory: {self.config.output_dir}")
        logger.info(f" Max cuts: {self.config.max_cuts}")
        logger.info(f" Top N targets: {self.config.top_n_targets}")
        logger.info(f" Threads: {self.config.num_threads or 'auto-detect'}")
        logger.info(f" Include computed properties: {self.config.include_computed_properties}")
        logger.info("=" * 70)

        self.start_time = time.time()

        try:
            # Stage 1: ChEMBL data extraction (or load existing)
            molecules_file, bioactivity_file, target_counts_file = self._stage1_chembl_extraction()

            # Stage 2: Overlap analysis (optional)
            if self.config.run_overlap_analysis and not self.config.skip_overlap_analysis:
                self._stage2_overlap_analysis(molecules_file, bioactivity_file)

            # Stage 3: MMP extraction
            if not self.config.skip_mmp_extraction:
                pairs_file = self._stage3_mmp_extraction(
                    molecules_file, bioactivity_file, target_counts_file
                )

                # Stage 4: Verification
                self._stage4_verification(pairs_file)

            # Summary
            self._print_summary()

        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            raise

    def _stage1_chembl_extraction(self):
        """Stage 1: Extract or load ChEMBL data."""
        stage_start = time.time()

        logger.info("\n" + "=" * 70)
        logger.info(" STAGE 1: CHEMBL DATA EXTRACTION")
        logger.info("=" * 70)

        # Check if we should use existing files
        molecules_file = self.config.molecules_file
        bioactivity_file = self.config.bioactivity_file
        target_counts_file = self.config.target_counts_file

        # If files provided, use them
        if molecules_file and bioactivity_file:
            logger.info("Using existing CSV files:")
            logger.info(f"  Molecules: {molecules_file}")
            logger.info(f"  Bioactivity: {bioactivity_file}")

            if target_counts_file:
                logger.info(f"  Target counts: {target_counts_file}")
            else:
                # Generate target counts from bioactivity file
                logger.info("  Generating target counts from bioactivity...")
                target_counts_file = self._generate_target_counts(bioactivity_file)

            # Verify files exist
            for f in [molecules_file, bioactivity_file, target_counts_file]:
                if not Path(f).exists():
                    raise FileNotFoundError(f"File not found: {f}")

            logger.info("✓ Files validated")

        # Otherwise, check cache
        elif self.config.skip_chembl_extraction:
            cache_dir = Path(self.config.chembl_cache_dir)
            molecules_file = str(cache_dir / 'molecules.csv')
            bioactivity_file = str(cache_dir / 'bioactivity.csv')
            target_counts_file = str(cache_dir / 'target_molecule_counts.csv')

            if Path(molecules_file).exists() and Path(bioactivity_file).exists():
                logger.info("Using cached ChEMBL data:")
                logger.info(f"  {molecules_file}")
                logger.info(f"  {bioactivity_file}")
            else:
                raise FileNotFoundError(
                    f"Cache files not found. Run without --skip-chembl-extraction first."
                )

        # Otherwise, extract from database
        else:
            logger.info("Extracting from ChEMBL database...")
            logger.info(f"  Database: {self.config.database_name}")
            logger.info(f"  Top N targets: {self.config.top_n_targets}")

            from src.data.chembl_extractor import ChEMBLExtractor, ChEMBLConfig

            chembl_config = ChEMBLConfig(
                database_name=self.config.database_name,
                host=self.config.database_host,
                port=self.config.database_port,
                user=self.config.database_user,
                password=self.config.database_password,
                top_n_targets=self.config.top_n_targets,
                min_molecules_per_label=self.config.min_molecules_per_target,
                specific_targets=self.config.specific_targets,
                target_mix=self.config.target_mix,
                max_cuts=self.config.max_cuts
            )

            extractor = ChEMBLExtractor(chembl_config)
            molecules_df, bioactivity_df = extractor.extract_data()

            # Save to cache
            cache_dir = Path(self.config.chembl_cache_dir)
            molecules_file = str(cache_dir / 'molecules.csv')
            bioactivity_file = str(cache_dir / 'bioactivity.csv')

            molecules_df.to_csv(molecules_file, index=False)
            bioactivity_df.to_csv(bioactivity_file, index=False)

            logger.info(f"  ✓ Saved molecules: {molecules_file}")
            logger.info(f"  ✓ Saved bioactivity: {bioactivity_file}")

            # Generate target counts
            target_counts_file = self._generate_target_counts(bioactivity_file)

        elapsed = time.time() - stage_start
        self.stage_times['chembl_extraction'] = elapsed
        logger.info(f"\n✓ Stage 1 complete ({elapsed:.1f}s)")

        return molecules_file, bioactivity_file, target_counts_file

    def _generate_target_counts(self, bioactivity_file: str) -> str:
        """Generate target_molecule_counts.csv from bioactivity."""
        bioactivity_df = pd.read_csv(bioactivity_file)

        # Filter to bioactivity only
        bioactivity_only = bioactivity_df[
            bioactivity_df['target_name'] != 'computed'
        ].copy()

        # Count molecules per target
        target_counts = bioactivity_only.groupby(
            ['target_chembl_id', 'target_name']
        ).agg(
            molecule_count=('chembl_id', 'nunique')
        ).reset_index()

        target_counts = target_counts.sort_values('molecule_count', ascending=False)

        # Save
        output_file = str(Path(self.config.chembl_cache_dir) / 'target_molecule_counts.csv')
        target_counts.to_csv(output_file, index=False)

        logger.info(f"  ✓ Generated target counts: {output_file}")
        return output_file

    def _stage2_overlap_analysis(self, molecules_file: str, bioactivity_file: str):
        """Stage 2: Analyze target overlap."""
        stage_start = time.time()

        logger.info("\n" + "=" * 70)
        logger.info(" STAGE 2: TARGET OVERLAP ANALYSIS")
        logger.info("=" * 70)

        from analyze_target_overlap import analyze_target_overlap

        results = analyze_target_overlap(
            molecules_file=molecules_file,
            bioactivity_file=bioactivity_file
        )

        # Save report
        report_file = Path(self.config.output_dir) / 'target_overlap_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(" TARGET OVERLAP ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Molecule overlap: {results['molecule_overlap_pct']:.1f}%\n")
            f.write(f"Pair duplicate computation: {results['duplicate_pair_pct']:.1f}%\n")
            f.write(f"Max pairwise overlap: {results['max_pairwise_overlap_pct']:.1f}%\n")

        logger.info(f"✓ Report saved: {report_file}")

        elapsed = time.time() - stage_start
        self.stage_times['overlap_analysis'] = elapsed
        logger.info(f"\n✓ Stage 2 complete ({elapsed:.1f}s)")

    def _stage3_mmp_extraction(
        self,
        molecules_file: str,
        bioactivity_file: str,
        target_counts_file: str
    ) -> str:
        """Stage 3: Extract MMP pairs."""
        stage_start = time.time()

        logger.info("\n" + "=" * 70)
        logger.info(" STAGE 3: MMP PAIR EXTRACTION")
        logger.info("=" * 70)

        from src.data.parallel_extraction import ParallelExtractor

        # Load data
        logger.info("Loading data...")
        molecules_df = pd.read_csv(molecules_file)
        bioactivity_df = pd.read_csv(bioactivity_file)
        target_counts_df = pd.read_csv(target_counts_file)

        logger.info(f"  Molecules: {len(molecules_df):,}")
        logger.info(f"  Bioactivity: {len(bioactivity_df):,}")
        logger.info(f"  Targets: {len(target_counts_df)}")

        # Create extractor
        extractor = ParallelExtractor(
            num_threads=self.config.num_threads,
            molecules_per_batch=self.config.molecules_per_batch,
            output_dir=self.config.output_dir
        )

        # Extract pairs
        pairs_df = extractor.extract_parallel(
            bioactivity_df=bioactivity_df,
            molecules_df=molecules_df,
            target_molecule_counts=target_counts_df,
            max_cuts=self.config.max_cuts,
            max_mw_delta=self.config.max_mw_delta,
            min_similarity=self.config.min_similarity,
            deduplicate_pairs=self.config.deduplicate_pairs,
            exclude_computed_properties=not self.config.include_computed_properties
        )

        # Output file
        pairs_file = str(Path(self.config.output_dir) / 'pairs_combined.csv')

        elapsed = time.time() - stage_start
        self.stage_times['mmp_extraction'] = elapsed
        logger.info(f"\n✓ Stage 3 complete ({elapsed:.1f}s)")

        return pairs_file

    def _stage4_verification(self, pairs_file: str):
        """Stage 4: Verify results."""
        stage_start = time.time()

        logger.info("\n" + "=" * 70)
        logger.info(" STAGE 4: VERIFICATION")
        logger.info("=" * 70)

        df = pd.read_csv(pairs_file)

        n_unique_pairs = df[['mol_a', 'mol_b']].drop_duplicates().shape[0]
        n_properties = df['property_name'].nunique()
        n_targets = df['target_chembl_id'].nunique()

        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Unique molecular pairs: {n_unique_pairs:,}")
        logger.info(f"  Unique properties: {n_properties}")
        logger.info(f"  Unique targets: {n_targets}")
        logger.info(f"  Avg properties per pair: {len(df) / n_unique_pairs:.2f}")

        # Check data quality
        logger.info("\n  Data quality checks:")

        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"    ⚠️  Columns with NaN: {nan_cols}")
        else:
            logger.info("    ✓ No NaN values")

        # Check delta calculation
        delta_check = (df['delta'] - (df['value_b'] - df['value_a'])).abs().max()
        if delta_check < 1e-6:
            logger.info("    ✓ Delta values correct")
        else:
            logger.warning(f"    ⚠️  Delta calculation error: {delta_check}")

        # Check duplicates
        duplicates = df.duplicated(subset=['mol_a', 'mol_b', 'property_name']).sum()
        if duplicates == 0:
            logger.info("    ✓ No duplicate pairs")
        else:
            logger.warning(f"    ⚠️  {duplicates:,} duplicate pairs found")

        elapsed = time.time() - stage_start
        self.stage_times['verification'] = elapsed
        logger.info(f"\n✓ Stage 4 complete ({elapsed:.1f}s)")

    def _print_summary(self):
        """Print pipeline summary."""
        total_time = time.time() - self.start_time

        logger.info("\n" + "=" * 70)
        logger.info(" PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f" Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
        logger.info("\n Stage timings:")
        for stage, elapsed in self.stage_times.items():
            pct = 100 * elapsed / total_time
            logger.info(f"   {stage:20s}: {elapsed:8.1f}s ({pct:5.1f}%)")

        logger.info("\n Output files:")
        logger.info(f"   Pairs: {self.config.output_dir}/pairs_combined.csv")
        if self.config.run_overlap_analysis:
            logger.info(f"   Overlap report: {self.config.output_dir}/target_overlap_report.txt")

        logger.info("\n Next steps:")
        logger.info("   Train models:")
        logger.info(f"     python experiments/main.py \\")
        logger.info(f"         --pairs-file {self.config.output_dir}/pairs_combined.csv")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ChEMBL MMP pair extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from database
  python scripts/run_chembl_pair_extraction.py --config configs/extraction_config.yaml

  # Use existing CSV files
  python scripts/run_chembl_pair_extraction.py \\
      --molecules-file data/chembl/molecules.csv \\
      --bioactivity-file data/chembl/bioactivity.csv \\
      --output-dir data/pairs

  # Quick test (top 3 targets, max_cuts=1)
  python scripts/run_chembl_pair_extraction.py \\
      --top-n 3 --max-cuts 1 --output-dir data/test
        """
    )

    # Config file
    parser.add_argument('--config', help='Path to YAML config file')

    # Input options (mutually exclusive with database extraction)
    parser.add_argument('--molecules-file', help='Existing molecules.csv file')
    parser.add_argument('--bioactivity-file', help='Existing bioactivity.csv file')
    parser.add_argument('--target-counts-file', help='Existing target_molecule_counts.csv file')

    # Database options
    parser.add_argument('--database', default='chembl_34', help='ChEMBL database name')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--db-user', help='Database user')
    parser.add_argument('--db-password', help='Database password')

    # Target selection
    parser.add_argument('--top-n', type=int, default=10, help='Top N targets')
    parser.add_argument('--min-molecules', type=int, default=1000, help='Min molecules per target')
    parser.add_argument('--specific-targets', nargs='+', help='Specific target IDs')

    # MMP parameters
    parser.add_argument('--max-cuts', type=int, default=3, help='Max bond cuts')
    parser.add_argument('--max-mw-delta', type=float, default=200, help='Max MW delta')
    parser.add_argument('--min-similarity', type=float, default=0.4, help='Min similarity')

    # Parallelization
    parser.add_argument('--num-threads', type=int, default=None, help='Number of threads (default: auto)')
    parser.add_argument('--molecules-per-batch', type=int, default=15000, help='Molecules per batch')

    # Output
    parser.add_argument('--output-dir', default='data/pairs', help='Output directory')
    parser.add_argument('--chembl-cache-dir', default='data/chembl', help='ChEMBL cache directory')

    # Options
    parser.add_argument('--include-computed-properties', action='store_true',
                       help='Include computed properties')
    parser.add_argument('--no-deduplicate', action='store_true',
                       help='Skip deduplication')
    parser.add_argument('--skip-overlap-analysis', action='store_true',
                       help='Skip overlap analysis')

    # Skip stages
    parser.add_argument('--skip-chembl-extraction', action='store_true',
                       help='Skip ChEMBL extraction (use cached files)')
    parser.add_argument('--skip-mmp-extraction', action='store_true',
                       help='Skip MMP extraction (only run analysis)')

    # Save config
    parser.add_argument('--save-config', help='Save configuration to YAML file')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Load or create config
    if args.config:
        config = ExtractionConfig.from_yaml(args.config)
    else:
        config = ExtractionConfig(
            # Input files
            molecules_file=args.molecules_file,
            bioactivity_file=args.bioactivity_file,
            target_counts_file=args.target_counts_file,
            # Database
            database_name=args.database,
            database_host=args.db_host,
            database_port=args.db_port,
            database_user=args.db_user,
            database_password=args.db_password,
            # Target selection
            top_n_targets=args.top_n,
            min_molecules_per_target=args.min_molecules,
            specific_targets=args.specific_targets,
            # MMP parameters
            max_cuts=args.max_cuts,
            max_mw_delta=args.max_mw_delta,
            min_similarity=args.min_similarity,
            # Parallelization
            num_threads=args.num_threads,
            molecules_per_batch=args.molecules_per_batch,
            # Output
            output_dir=args.output_dir,
            chembl_cache_dir=args.chembl_cache_dir,
            # Options
            include_computed_properties=args.include_computed_properties,
            deduplicate_pairs=not args.no_deduplicate,
            run_overlap_analysis=not args.skip_overlap_analysis,
            # Skip stages
            skip_chembl_extraction=args.skip_chembl_extraction,
            skip_mmp_extraction=args.skip_mmp_extraction
        )

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        logger.info(f"✓ Configuration saved to: {args.save_config}")

    # Run pipeline
    runner = PipelineRunner(config)
    runner.run()


if __name__ == '__main__':
    main()
