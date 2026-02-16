"""
Analyze molecular overlap across ChEMBL targets.

This script answers the question:
"How often does the same molecular pair appear across multiple targets?"

If overlap is high (>20%), we should extract MMPs on unique molecules first,
then apply property labels. If overlap is low (<10%), per-target extraction
with deduplication is fine.

Usage:
    python scripts/analyze_target_overlap.py \
        --molecules-file data/chembl/molecules.csv \
        --bioactivity-file data/chembl/bioactivity.csv \
        --output-report data/analysis/target_overlap_report.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


def analyze_target_overlap(
    molecules_file: str,
    bioactivity_file: str
) -> dict:
    """
    Analyze molecular overlap across ChEMBL targets.

    Args:
        molecules_file: Path to molecules.csv
        bioactivity_file: Path to bioactivity.csv

    Returns:
        Dict with analysis results
    """
    logger.info("=" * 70)
    logger.info(" MOLECULAR OVERLAP ANALYSIS")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\nStep 1: Loading ChEMBL data...")

    molecules_df = pd.read_csv(molecules_file)
    bioactivity_df = pd.read_csv(bioactivity_file)

    # Filter to bioactivity only (exclude computed properties)
    bioactivity_only = bioactivity_df[bioactivity_df['target_name'] != 'computed'].copy()

    logger.info(f"  Molecules: {len(molecules_df):,}")
    logger.info(f"  Bioactivity measurements: {len(bioactivity_only):,}")
    logger.info(f"  Targets: {bioactivity_only['target_chembl_id'].nunique()}")

    # Step 2: Build molecule-to-targets mapping
    logger.info("\nStep 2: Analyzing molecule-target relationships...")

    # Group by molecule, collect targets
    molecule_targets = defaultdict(set)

    for _, row in bioactivity_only.iterrows():
        chembl_id = row['chembl_id']
        target_id = row['target_chembl_id']
        molecule_targets[chembl_id].add(target_id)

    # Count molecules by number of targets
    molecules_per_target_count = defaultdict(int)
    for chembl_id, targets in molecule_targets.items():
        num_targets = len(targets)
        molecules_per_target_count[num_targets] += 1

    total_molecules = len(molecule_targets)
    molecules_with_multiple_targets = sum(
        count for num_targets, count in molecules_per_target_count.items()
        if num_targets > 1
    )

    logger.info(f"  Total unique molecules: {total_molecules:,}")
    logger.info(f"  Molecules tested on 1 target: {molecules_per_target_count[1]:,}")
    logger.info(f"  Molecules tested on 2+ targets: {molecules_with_multiple_targets:,}")
    logger.info(f"  Overlap rate: {100 * molecules_with_multiple_targets / total_molecules:.1f}%")

    # Step 3: Estimate pair overlap
    logger.info("\nStep 3: Estimating pair overlap...")

    # For each target, count molecules
    target_molecule_counts = bioactivity_only.groupby('target_chembl_id')['chembl_id'].nunique()

    # Build target-to-molecules mapping
    target_molecules = defaultdict(set)
    for _, row in bioactivity_only.iterrows():
        target_id = row['target_chembl_id']
        chembl_id = row['chembl_id']
        target_molecules[target_id].add(chembl_id)

    # Estimate potential pairs per target
    total_potential_pairs = 0
    for target_id, molecules in target_molecules.items():
        n = len(molecules)
        pairs = n * (n - 1) // 2
        total_potential_pairs += pairs

    logger.info(f"  Total potential pairs (sum across targets): {total_potential_pairs:,}")

    # Estimate unique pairs (if we extract all targets together)
    all_molecules = set()
    for molecules in target_molecules.values():
        all_molecules.update(molecules)

    n_unique = len(all_molecules)
    unique_pairs = n_unique * (n_unique - 1) // 2

    logger.info(f"  Unique pairs (if extracted together): {unique_pairs:,}")

    # Estimate duplicate computation
    duplicate_pair_computation = total_potential_pairs - unique_pairs
    duplicate_percentage = 100 * duplicate_pair_computation / total_potential_pairs

    logger.info(f"  Duplicate pair computation: {duplicate_pair_computation:,}")
    logger.info(f"  Duplicate percentage: {duplicate_percentage:.1f}%")

    # Step 4: Pairwise target overlap
    logger.info("\nStep 4: Pairwise target overlap...")

    target_ids = list(target_molecules.keys())
    target_names_map = dict(zip(
        bioactivity_only['target_chembl_id'],
        bioactivity_only['target_name']
    ))

    max_overlap = 0
    max_overlap_pair = None

    logger.info("\n  Top overlapping target pairs:")

    overlaps = []
    for i in range(len(target_ids)):
        for j in range(i + 1, len(target_ids)):
            target_a = target_ids[i]
            target_b = target_ids[j]

            molecules_a = target_molecules[target_a]
            molecules_b = target_molecules[target_b]

            overlap = len(molecules_a & molecules_b)
            overlap_pct = 100 * overlap / min(len(molecules_a), len(molecules_b))

            overlaps.append({
                'target_a': target_a,
                'target_b': target_b,
                'target_a_name': target_names_map[target_a],
                'target_b_name': target_names_map[target_b],
                'molecules_a': len(molecules_a),
                'molecules_b': len(molecules_b),
                'overlap': overlap,
                'overlap_pct': overlap_pct
            })

            if overlap_pct > max_overlap:
                max_overlap = overlap_pct
                max_overlap_pair = (target_a, target_b)

    # Sort by overlap percentage
    overlaps_sorted = sorted(overlaps, key=lambda x: x['overlap_pct'], reverse=True)

    for entry in overlaps_sorted[:5]:
        logger.info(
            f"    {entry['target_a_name'][:30]:30s} ↔ {entry['target_b_name'][:30]:30s}: "
            f"{entry['overlap']:4d} molecules ({entry['overlap_pct']:5.1f}%)"
        )

    # Step 5: Summary and recommendations
    logger.info("\n" + "=" * 70)
    logger.info(" SUMMARY")
    logger.info("=" * 70)

    results = {
        'total_molecules': total_molecules,
        'molecules_with_multiple_targets': molecules_with_multiple_targets,
        'molecule_overlap_pct': 100 * molecules_with_multiple_targets / total_molecules,
        'total_potential_pairs': total_potential_pairs,
        'unique_pairs': unique_pairs,
        'duplicate_pairs': duplicate_pair_computation,
        'duplicate_pair_pct': duplicate_percentage,
        'max_pairwise_overlap_pct': max_overlap,
        'top_overlaps': overlaps_sorted[:10]
    }

    logger.info(f" Molecule-level overlap: {results['molecule_overlap_pct']:.1f}%")
    logger.info(f" Pair-level duplicate computation: {results['duplicate_pair_pct']:.1f}%")
    logger.info(f" Max pairwise target overlap: {results['max_pairwise_overlap_pct']:.1f}%")

    logger.info("\n RECOMMENDATION:")

    if results['duplicate_pair_pct'] > 20:
        logger.info("  ⚠️  HIGH OVERLAP (>20%)")
        logger.info("  → Consider extracting MMPs on unique molecules first")
        logger.info("  → Then apply property labels in post-processing")
        logger.info("  → This avoids duplicate MMP computation")
    elif results['duplicate_pair_pct'] > 10:
        logger.info("  ℹ️  MODERATE OVERLAP (10-20%)")
        logger.info("  → Per-target extraction with deduplication is reasonable")
        logger.info("  → ~10-20% redundant computation, but parallel speedup compensates")
    else:
        logger.info("  ✅ LOW OVERLAP (<10%)")
        logger.info("  → Per-target extraction with deduplication is efficient")
        logger.info("  → Minimal redundant computation")

    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze target overlap in ChEMBL")
    parser.add_argument('--molecules-file', required=True,
                       help='Path to molecules.csv')
    parser.add_argument('--bioactivity-file', required=True,
                       help='Path to bioactivity.csv')
    parser.add_argument('--output-report', default=None,
                       help='Output report file (optional)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    results = analyze_target_overlap(
        molecules_file=args.molecules_file,
        bioactivity_file=args.bioactivity_file
    )

    # Save detailed report if requested
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(" MOLECULAR OVERLAP ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Molecules file: {args.molecules_file}\n")
            f.write(f"Bioactivity file: {args.bioactivity_file}\n\n")

            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Total unique molecules: {results['total_molecules']:,}\n")
            f.write(f"  Molecules with multiple targets: {results['molecules_with_multiple_targets']:,}\n")
            f.write(f"  Molecule overlap rate: {results['molecule_overlap_pct']:.1f}%\n\n")

            f.write(f"  Total potential pairs (sum): {results['total_potential_pairs']:,}\n")
            f.write(f"  Unique pairs: {results['unique_pairs']:,}\n")
            f.write(f"  Duplicate pair computation: {results['duplicate_pairs']:,}\n")
            f.write(f"  Duplicate percentage: {results['duplicate_pair_pct']:.1f}%\n\n")

            f.write(f"  Max pairwise target overlap: {results['max_pairwise_overlap_pct']:.1f}%\n\n")

            f.write("TOP 10 OVERLAPPING TARGET PAIRS:\n")
            for i, entry in enumerate(results['top_overlaps'], 1):
                f.write(f"{i:2d}. {entry['target_a_name']:40s} ↔ {entry['target_b_name']:40s}\n")
                f.write(f"    Overlap: {entry['overlap']:,} molecules ({entry['overlap_pct']:.1f}%)\n")
                f.write(f"    Sizes: {entry['molecules_a']:,} vs {entry['molecules_b']:,}\n\n")

        logger.info(f"\n✓ Detailed report saved to: {output_path}")


if __name__ == '__main__':
    main()
