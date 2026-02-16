"""
Parallel MMP extraction using threading for single-machine optimization.

This module provides thread-level parallelization for ChEMBL extraction,
allowing efficient use of multi-core machines without the overhead of
multiple machines or processes.

Strategy:
- Split targets into batches based on molecule count
- Each thread processes one batch of targets
- Targets are grouped to balance load
- Results are merged at the end

Example:
    # Extract top 10 targets using 8 threads
    from parallel_extraction import ParallelExtractor

    config = ChEMBLConfig(
        top_n_targets=10,
        max_cuts=3
    )

    extractor = ParallelExtractor(
        num_threads=8,
        config=config
    )

    results = extractor.extract_parallel()
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class TargetBatch:
    """Represents a batch of targets to process together."""
    batch_id: int
    target_ids: List[str]
    target_names: List[str]
    total_molecules: int
    estimated_pairs: int


class ParallelExtractor:
    """
    Parallel MMP extractor using thread-level parallelization.

    Designed for single-machine optimization with load balancing.
    """

    def __init__(
        self,
        num_threads: Optional[int] = None,
        molecules_per_batch: int = 15000,
        output_dir: str = "data/pairs"
    ):
        """
        Initialize parallel extractor.

        Args:
            num_threads: Number of worker threads (default: None = use all available cores)
            molecules_per_batch: Target molecules per batch for load balancing (default: 15K)
            output_dir: Directory for output files
        """
        import os
        if num_threads is None:
            num_threads = os.cpu_count() or 8
            logger.info(f"Auto-detected {num_threads} CPU cores")

        self.num_threads = num_threads
        self.molecules_per_batch = molecules_per_batch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_balanced_batches(
        self,
        target_molecule_counts: pd.DataFrame
    ) -> List[TargetBatch]:
        """
        Create balanced batches of targets for parallel processing.

        Strategy:
        1. Sort targets by molecule count (descending)
        2. Use greedy bin-packing to balance load across batches
        3. Group small targets together to avoid thread underutilization

        Args:
            target_molecule_counts: DataFrame with columns:
                - target_chembl_id
                - target_name
                - molecule_count

        Returns:
            List of TargetBatch objects
        """
        # Sort by molecule count (descending)
        sorted_targets = target_molecule_counts.sort_values(
            'molecule_count', ascending=False
        ).reset_index(drop=True)

        logger.info(f"Creating balanced batches for {len(sorted_targets)} targets")
        logger.info(f"Target molecules per batch: ~{self.molecules_per_batch:,}")

        # Greedy bin-packing algorithm
        batches = []
        current_batch_targets = []
        current_batch_names = []
        current_batch_molecules = 0
        batch_id = 0

        for _, row in sorted_targets.iterrows():
            target_id = row['target_chembl_id']
            target_name = row['target_name']
            mol_count = row['molecule_count']

            # If adding this target exceeds threshold, start new batch
            # UNLESS it's the first target (large targets get their own batch)
            if current_batch_molecules + mol_count > self.molecules_per_batch and current_batch_targets:
                # Save current batch
                estimated_pairs = self._estimate_pairs(current_batch_molecules)
                batches.append(TargetBatch(
                    batch_id=batch_id,
                    target_ids=current_batch_targets,
                    target_names=current_batch_names,
                    total_molecules=current_batch_molecules,
                    estimated_pairs=estimated_pairs
                ))

                # Start new batch
                batch_id += 1
                current_batch_targets = []
                current_batch_names = []
                current_batch_molecules = 0

            # Add target to current batch
            current_batch_targets.append(target_id)
            current_batch_names.append(target_name)
            current_batch_molecules += mol_count

        # Add final batch
        if current_batch_targets:
            estimated_pairs = self._estimate_pairs(current_batch_molecules)
            batches.append(TargetBatch(
                batch_id=batch_id,
                target_ids=current_batch_targets,
                target_names=current_batch_names,
                total_molecules=current_batch_molecules,
                estimated_pairs=estimated_pairs
            ))

        # Log batch statistics
        logger.info(f"Created {len(batches)} batches:")
        for batch in batches:
            logger.info(
                f"  Batch {batch.batch_id}: {len(batch.target_ids)} targets, "
                f"{batch.total_molecules:,} molecules, "
                f"~{batch.estimated_pairs:,} pairs"
            )

        return batches

    def _estimate_pairs(self, num_molecules: int) -> int:
        """
        Estimate number of pairs for molecule count.

        Uses empirical formula based on core density.
        Average core has ~20 molecules, so:
        - num_cores ≈ num_molecules / 20
        - pairs_per_core ≈ C(20, 2) = 190
        - total_pairs ≈ num_cores * 190
        """
        avg_molecules_per_core = 20
        num_cores = num_molecules / avg_molecules_per_core
        pairs_per_core = 190  # C(20, 2)
        return int(num_cores * pairs_per_core)

    def extract_batch(
        self,
        batch: TargetBatch,
        bioactivity_df: pd.DataFrame,
        molecules_df: pd.DataFrame,
        max_cuts: int = 3,
        max_mw_delta: float = 200,
        min_similarity: float = 0.4,
        exclude_computed_properties: bool = True
    ) -> Tuple[int, str]:
        """
        Extract pairs for a single batch of targets.

        Args:
            batch: TargetBatch to process
            bioactivity_df: Full bioactivity DataFrame
            molecules_df: Full molecules DataFrame
            max_cuts: Max bond cuts
            max_mw_delta: Max MW difference
            min_similarity: Min Tanimoto similarity
            exclude_computed_properties: Only extract bioactivity (no computed properties)

        Returns:
            Tuple of (batch_id, output_file_path)
        """
        from mmp_long_format import LongFormatMMPExtractor

        logger.info(f"[Batch {batch.batch_id}] Starting extraction...")
        logger.info(f"[Batch {batch.batch_id}] Targets: {', '.join(batch.target_names)}")

        start_time = time.time()

        # Filter bioactivity to only include this batch's targets
        batch_bioactivity = bioactivity_df[
            bioactivity_df['target_chembl_id'].isin(batch.target_ids)
        ].copy()

        # Exclude computed properties if requested
        if exclude_computed_properties:
            # Filter out rows where target_name == 'computed'
            batch_bioactivity = batch_bioactivity[
                batch_bioactivity['target_name'] != 'computed'
            ].copy()
            logger.info(f"[Batch {batch.batch_id}] Excluding computed properties (bioactivity only)")

        # Filter molecules to only those in batch bioactivity
        batch_chembl_ids = batch_bioactivity['chembl_id'].unique()
        batch_molecules = molecules_df[
            molecules_df['chembl_id'].isin(batch_chembl_ids)
        ].copy()

        logger.info(
            f"[Batch {batch.batch_id}] Filtered to {len(batch_molecules):,} molecules, "
            f"{len(batch_bioactivity):,} measurements"
        )

        # Extract pairs
        extractor = LongFormatMMPExtractor(max_cuts=max_cuts)

        checkpoint_dir = self.output_dir / f"checkpoints_batch_{batch.batch_id}"

        pairs_df = extractor.extract_pairs_long_format(
            molecules_df=batch_molecules,
            bioactivity_df=batch_bioactivity,
            max_mw_delta=max_mw_delta,
            min_similarity=min_similarity,
            checkpoint_dir=str(checkpoint_dir),
            micro_batch_size=200,
            resume_from_checkpoint=True
        )

        # Save batch result
        output_file = self.output_dir / f"pairs_batch_{batch.batch_id}.csv"
        pairs_df.to_csv(output_file, index=False)

        elapsed = time.time() - start_time

        logger.info(
            f"[Batch {batch.batch_id}] Complete! "
            f"Extracted {len(pairs_df):,} pairs in {elapsed:.1f}s"
        )

        return batch.batch_id, str(output_file)

    def extract_parallel(
        self,
        bioactivity_df: pd.DataFrame,
        molecules_df: pd.DataFrame,
        target_molecule_counts: pd.DataFrame,
        max_cuts: int = 3,
        max_mw_delta: float = 200,
        min_similarity: float = 0.4,
        deduplicate_pairs: bool = True,
        exclude_computed_properties: bool = True
    ) -> pd.DataFrame:
        """
        Extract pairs in parallel using thread pool.

        Args:
            bioactivity_df: Full bioactivity DataFrame
            molecules_df: Full molecules DataFrame
            target_molecule_counts: DataFrame with target molecule counts
            max_cuts: Max bond cuts
            max_mw_delta: Max MW difference
            min_similarity: Min Tanimoto similarity
            deduplicate_pairs: Whether to deduplicate pairs across targets (default: True)
            exclude_computed_properties: Only extract bioactivity properties (default: True)

        Returns:
            Combined pairs DataFrame
        """
        logger.info("=" * 70)
        logger.info(" PARALLEL MMP EXTRACTION")
        logger.info("=" * 70)
        logger.info(f" Threads: {self.num_threads}")
        logger.info(f" Molecules per batch: ~{self.molecules_per_batch:,}")
        logger.info(f" Max cuts: {max_cuts}")
        logger.info(f" Deduplicate pairs: {deduplicate_pairs}")
        logger.info(f" Exclude computed properties: {exclude_computed_properties}")
        logger.info("=" * 70)

        # Step 1: Create balanced batches
        batches = self.create_balanced_batches(target_molecule_counts)

        # Step 2: Extract in parallel
        logger.info(f"\nStarting parallel extraction with {self.num_threads} threads...")

        batch_results = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batches
            futures = {
                executor.submit(
                    self.extract_batch,
                    batch,
                    bioactivity_df,
                    molecules_df,
                    max_cuts,
                    max_mw_delta,
                    min_similarity,
                    exclude_computed_properties
                ): batch for batch in batches
            }

            # Collect results as they complete
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_id, output_file = future.result()
                    batch_results.append(output_file)
                    logger.info(f"✓ Batch {batch_id} complete")
                except Exception as e:
                    logger.error(f"✗ Batch {batch.batch_id} failed: {e}")

        # Step 3: Merge results
        logger.info(f"\nMerging {len(batch_results)} batch results...")

        all_pairs = []
        for result_file in batch_results:
            df = pd.read_csv(result_file)
            all_pairs.append(df)
            logger.info(f"  Loaded {len(df):,} pairs from {Path(result_file).name}")

        combined_df = pd.concat(all_pairs, ignore_index=True)

        logger.info(f"Combined: {len(combined_df):,} total pairs")

        # Step 4: Deduplicate if requested
        if deduplicate_pairs:
            logger.info("\nDeduplicating pairs across targets...")

            # Count duplicates
            pair_counts = combined_df.groupby(['mol_a', 'mol_b', 'property_name']).size()
            duplicates = pair_counts[pair_counts > 1]

            if len(duplicates) > 0:
                logger.info(f"  Found {len(duplicates):,} duplicate pair-property combinations")
                logger.info(f"  Example duplicates (first 5):")
                for (mol_a, mol_b, prop), count in duplicates.head().items():
                    logger.info(f"    ({mol_a[:20]}..., {mol_b[:20]}..., {prop}): {count} times")

                # Keep first occurrence (arbitrary choice - could use other strategies)
                combined_df = combined_df.drop_duplicates(
                    subset=['mol_a', 'mol_b', 'property_name'],
                    keep='first'
                )
                logger.info(f"  After deduplication: {len(combined_df):,} pairs")
            else:
                logger.info("  No duplicates found across targets!")

        # Step 5: Save final result
        final_output = self.output_dir / "pairs_combined.csv"
        combined_df.to_csv(final_output, index=False)

        logger.info(f"\n✓ Final result saved to: {final_output}")
        logger.info(f"  Total pairs: {len(combined_df):,}")
        logger.info(f"  Unique molecular pairs: {combined_df[['mol_a', 'mol_b']].drop_duplicates().shape[0]:,}")
        logger.info(f"  Unique properties: {combined_df['property_name'].nunique()}")

        return combined_df


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel MMP extraction")
    parser.add_argument('--molecules-file', required=True)
    parser.add_argument('--bioactivity-file', required=True)
    parser.add_argument('--target-counts-file', required=True,
                       help='CSV with target_chembl_id, target_name, molecule_count')
    parser.add_argument('--num-threads', type=int, default=None,
                       help='Number of threads (default: None = use all CPU cores)')
    parser.add_argument('--molecules-per-batch', type=int, default=15000)
    parser.add_argument('--max-cuts', type=int, default=3)
    parser.add_argument('--output-dir', default='data/pairs')
    parser.add_argument('--no-deduplicate', action='store_true',
                       help='Skip deduplication (keep all pairs)')
    parser.add_argument('--include-computed-properties', action='store_true',
                       help='Include computed chemical properties (default: bioactivity only)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load data
    logger.info("Loading data...")
    molecules_df = pd.read_csv(args.molecules_file)
    bioactivity_df = pd.read_csv(args.bioactivity_file)
    target_counts_df = pd.read_csv(args.target_counts_file)

    # Extract
    extractor = ParallelExtractor(
        num_threads=args.num_threads,
        molecules_per_batch=args.molecules_per_batch,
        output_dir=args.output_dir
    )

    result_df = extractor.extract_parallel(
        bioactivity_df=bioactivity_df,
        molecules_df=molecules_df,
        target_molecule_counts=target_counts_df,
        max_cuts=args.max_cuts,
        deduplicate_pairs=not args.no_deduplicate,
        exclude_computed_properties=not args.include_computed_properties
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
