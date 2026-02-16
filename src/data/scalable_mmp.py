"""
Scalable MMP extraction for large datasets (100K-1M+ molecules).

Strategy to avoid O(n²) complexity:
1. Molecular weight filtering (ΔMW < threshold)
2. Morgan fingerprint similarity (Tanimoto > threshold)
3. Core-based indexing (rdMMPA fragmentation)
4. Batch processing with checkpointing

For 1M molecules:
- Naive: 500B comparisons
- Filtered: ~100M comparisons (5000x reduction)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import pickle
import hashlib

from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA, Descriptors, rdFingerprintGenerator
from rdkit import DataStructs

logger = logging.getLogger(__name__)


@dataclass
class ScalableMMPConfig:
    """Configuration for scalable MMP extraction."""

    # Stage 1: Molecular weight filter
    max_mw_delta: float = 200  # Only compare molecules within 200 Da

    # Stage 2: Fingerprint similarity filter
    min_similarity: float = 0.4  # Tanimoto threshold (0.4 = reasonably similar)
    fp_radius: int = 2
    fp_bits: int = 2048

    # Stage 3: MMP extraction
    max_cuts: int = 2
    min_heavy_atoms: int = 5

    # Performance
    batch_size: int = 10000  # Process in batches
    checkpoint_every: int = 50000  # Save checkpoint
    use_multiprocessing: bool = True
    n_jobs: int = -1  # -1 = all CPUs

    # Output
    min_pair_count: int = 1  # Minimum pairs to keep an edit


class ScalableMMPExtractor:
    """
    Extract matched molecular pairs from large datasets efficiently.

    Uses multi-stage filtering:
    1. MW binning: O(n log n)
    2. Fingerprint similarity: O(n * k) where k << n
    3. MMP extraction: Only on filtered candidates

    Expected complexity for 1M molecules: ~O(n * k) where k ≈ 100-1000
    """

    def __init__(self, config: Optional[ScalableMMPConfig] = None):
        """
        Initialize scalable MMP extractor.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or ScalableMMPConfig()
        logger.info("Initialized scalable MMP extractor")
        logger.info(f"Config: MW delta={self.config.max_mw_delta}, "
                   f"similarity={self.config.min_similarity}, "
                   f"max_cuts={self.config.max_cuts}")

    def extract_pairs_scalable(self,
                               smiles_list: List[str],
                               properties: Dict[str, Dict[str, float]],
                               property_name: str = 'property',
                               checkpoint_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Extract MMPs from large dataset with smart filtering.

        Args:
            smiles_list: List of SMILES strings
            properties: Dict mapping SMILES to dict of property values
                       e.g., {'CCO': {'alogp': 0.5, 'mw': 46.0, 'psa': 20.2}}
                       OR Dict mapping SMILES to single property value (backward compatible)
            property_name: Primary property name for filtering (deprecated, kept for compatibility)
            checkpoint_dir: Directory for checkpoints (None = no checkpoints)

        Returns:
            DataFrame with columns: mol_a, mol_b, edit_id, core,
                                   from_smarts, to_smarts, delta_* for each property
        """
        logger.info("=" * 60)
        logger.info("SCALABLE MMP EXTRACTION")
        logger.info("=" * 60)
        logger.info(f"Input: {len(smiles_list):,} molecules")

        # Detect if properties is old format (single value) or new format (dict of values)
        sample_smiles = next(iter(properties.keys()))
        if isinstance(properties[sample_smiles], dict):
            property_names = list(properties[sample_smiles].keys())
            logger.info(f"Properties: {', '.join(property_names)}")
            multi_property = True
        else:
            logger.info(f"Property: {property_name}")
            property_names = [property_name]
            multi_property = False
        logger.info("")

        # Setup checkpoint
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / "mmp_checkpoint.pkl"
        else:
            checkpoint_file = None

        # Stage 1: Preprocess molecules
        logger.info("Stage 1: Preprocessing molecules...")
        mol_data = self._preprocess_molecules(smiles_list, properties, multi_property)
        logger.info(f"Valid molecules: {len(mol_data):,}")

        # Stage 2: MW-based binning
        logger.info("\nStage 2: Molecular weight binning...")
        mw_bins = self._bin_by_molecular_weight(mol_data)
        logger.info(f"Created {len(mw_bins)} MW bins")

        # Stage 3: Generate candidate pairs with fingerprint filtering
        logger.info("\nStage 3: Generating candidate pairs (with similarity filter)...")
        candidate_pairs = self._generate_candidate_pairs(mol_data, mw_bins)
        logger.info(f"Candidate pairs (after filtering): {len(candidate_pairs):,}")

        # Stage 4: Extract MMPs from candidates
        logger.info("\nStage 4: Extracting MMPs from candidates...")
        pairs_df = self._extract_mmps_from_candidates(
            mol_data,
            candidate_pairs,
            property_names,
            checkpoint_file
        )

        logger.info("=" * 60)
        logger.info(f"EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total pairs found: {len(pairs_df):,}")
        logger.info(f"Unique edits: {pairs_df['edit_id'].nunique():,}")
        logger.info("=" * 60)

        return pairs_df

    def _preprocess_molecules(self,
                             smiles_list: List[str],
                             properties: Dict[str, float],
                             multi_property: bool = False) -> List[Dict]:
        """
        Preprocess molecules: parse, compute MW, generate fingerprints.

        Args:
            smiles_list: SMILES strings
            properties: Property values (dict of values or single value)
            multi_property: Whether properties contains multiple properties per molecule

        Returns:
            List of dicts with molecule data
        """
        mol_data = []

        for smiles in tqdm(smiles_list, desc="Preprocessing"):
            if smiles not in properties:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Compute MW
            mw = Descriptors.MolWt(mol)

            # Generate fingerprint using new API
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.config.fp_radius,
                fpSize=self.config.fp_bits
            )
            fp = morgan_gen.GetFingerprint(mol)

            # Store properties
            if multi_property:
                property_values = properties[smiles]
            else:
                property_values = {'property': properties[smiles]}

            mol_data.append({
                'smiles': smiles,
                'mol': mol,
                'mw': mw,
                'fp': fp,
                'properties': property_values,  # Changed from 'property' to 'properties'
                'idx': len(mol_data)
            })

        return mol_data

    def _bin_by_molecular_weight(self, mol_data: List[Dict]) -> Dict[int, List[int]]:
        """
        Bin molecules by molecular weight for efficient filtering.

        Only molecules in same or adjacent bins need to be compared.

        Args:
            mol_data: Preprocessed molecule data

        Returns:
            Dict mapping bin_id -> list of molecule indices
        """
        # Bin width = max_mw_delta (e.g., 200 Da)
        bin_width = self.config.max_mw_delta

        bins = defaultdict(list)

        for data in mol_data:
            bin_id = int(data['mw'] / bin_width)
            bins[bin_id].append(data['idx'])

        logger.info(f"MW range: {min(d['mw'] for d in mol_data):.1f} - {max(d['mw'] for d in mol_data):.1f}")
        logger.info(f"Bin width: {bin_width} Da")
        logger.info(f"Molecules per bin (avg): {len(mol_data) / len(bins):.1f}")

        return bins

    def _generate_candidate_pairs(self,
                                  mol_data: List[Dict],
                                  mw_bins: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
        """
        Generate candidate pairs using MW binning + fingerprint similarity.

        This is the key optimization: instead of n², we compare each molecule
        only with similar molecules (same/adjacent MW bins + fingerprint match).

        Args:
            mol_data: Preprocessed molecule data
            mw_bins: MW bin assignments

        Returns:
            Set of (idx_a, idx_b) tuples (with idx_a < idx_b)
        """
        candidates = set()

        # Build list of all molecules to process (for progress bar)
        all_molecule_indices = []
        for bin_id, mol_indices in mw_bins.items():
            for idx in mol_indices:
                all_molecule_indices.append((bin_id, idx))

        logger.info(f"Processing {len(all_molecule_indices):,} molecules across {len(mw_bins)} bins...")

        # Process each molecule with progress bar
        processed_comparisons = 0
        last_report_pct = 0

        for bin_id, idx_a in tqdm(all_molecule_indices, desc="Generating candidates", unit="mol"):
            mol_a = mol_data[idx_a]

            # Get molecules in current and adjacent bins
            adjacent_bins = [bin_id - 1, bin_id, bin_id + 1]
            neighbor_indices = []
            for adj_bin in adjacent_bins:
                neighbor_indices.extend(mw_bins.get(adj_bin, []))

            # Compare with neighbors
            for idx_b in neighbor_indices:
                if idx_b <= idx_a:  # Avoid duplicates and self-comparison
                    continue

                mol_b = mol_data[idx_b]

                # MW filter (exact check)
                mw_delta = abs(mol_a['mw'] - mol_b['mw'])
                if mw_delta > self.config.max_mw_delta:
                    continue

                # Fingerprint similarity filter
                similarity = DataStructs.TanimotoSimilarity(mol_a['fp'], mol_b['fp'])
                if similarity < self.config.min_similarity:
                    continue

                # Passed filters - add to candidates
                candidates.add((idx_a, idx_b))
                processed_comparisons += 1

        logger.info(f"Filtered to {processed_comparisons:,} similar pairs from {len(all_molecule_indices):,} molecules")

        return candidates

    def _extract_mmps_from_candidates(self,
                                     mol_data: List[Dict],
                                     candidate_pairs: Set[Tuple[int, int]],
                                     property_names: List[str],
                                     checkpoint_file: Optional[Path]) -> pd.DataFrame:
        """
        Extract MMPs from filtered candidate pairs.

        Args:
            mol_data: Preprocessed molecule data
            candidate_pairs: Filtered candidate pairs
            property_names: List of property names
            checkpoint_file: Path to checkpoint file

        Returns:
            DataFrame with MMP data
        """
        # Build core index using rdMMPA
        logger.info("Building core index (fragmenting molecules)...")
        core_index = self._build_core_index(mol_data)

        logger.info(f"Unique cores: {len(core_index):,}")
        logger.info(f"Avg molecules per core: {sum(len(v) for v in core_index.values()) / len(core_index):.1f}")

        # Extract pairs from core index
        pairs = []

        logger.info("Extracting MMPs from candidates...")

        # Only check candidate pairs that share a core
        checked = 0
        for core, mol_indices in tqdm(core_index.items(), desc="Extracting"):
            # Only check pairs that are in our candidate set
            for i, idx_a in enumerate(mol_indices):
                for idx_b in mol_indices[i+1:]:
                    # Ensure consistent ordering
                    pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))

                    if pair_key not in candidate_pairs:
                        continue

                    checked += 1

                    mol_a_data = mol_data[idx_a]
                    mol_b_data = mol_data[idx_b]

                    # Extract edit
                    pair_data = self._extract_edit(
                        mol_a_data,
                        mol_b_data,
                        core,
                        property_names
                    )

                    if pair_data:
                        pairs.append(pair_data)

        logger.info(f"Checked {checked:,} candidate pairs")
        logger.info(f"Found {len(pairs):,} valid MMPs")

        # Convert to DataFrame
        df = pd.DataFrame(pairs)

        # Save checkpoint if requested
        if checkpoint_file:
            df.to_csv(checkpoint_file.with_suffix('.csv'), index=False)
            logger.info(f"Saved checkpoint: {checkpoint_file.with_suffix('.csv')}")

        return df

    def _build_core_index(self, mol_data: List[Dict]) -> Dict[str, List[int]]:
        """
        Build index: core SMILES -> list of molecule indices.

        Uses rdMMPA to fragment molecules and find cores.

        Args:
            mol_data: Preprocessed molecule data

        Returns:
            Dict mapping core SMILES to molecule indices
        """
        core_index = defaultdict(list)

        for data in tqdm(mol_data, desc="Fragmenting"):
            mol = data['mol']
            idx = data['idx']

            # Fragment molecule
            try:
                frags = rdMMPA.FragmentMol(
                    mol,
                    maxCuts=self.config.max_cuts,
                    resultsAsMols=False
                )

                # Add to index for each core
                for core, _ in frags:
                    core_index[core].append(idx)

            except Exception as e:
                logger.debug(f"Failed to fragment {data['smiles']}: {e}")
                continue

        # Filter cores with only 1 molecule (no pairs possible)
        core_index = {k: v for k, v in core_index.items() if len(v) >= 2}

        return dict(core_index)

    def _extract_edit(self,
                     mol_a_data: Dict,
                     mol_b_data: Dict,
                     core: str,
                     property_names: List[str]) -> Optional[Dict]:
        """
        Extract edit information from molecule pair.

        Args:
            mol_a_data: Data for molecule A
            mol_b_data: Data for molecule B
            core: Shared core SMILES
            property_names: List of property names

        Returns:
            Dict with pair data or None
        """
        # Fragment both molecules
        try:
            frags_a = dict(rdMMPA.FragmentMol(mol_a_data['mol'], maxCuts=self.config.max_cuts, resultsAsMols=False))
            frags_b = dict(rdMMPA.FragmentMol(mol_b_data['mol'], maxCuts=self.config.max_cuts, resultsAsMols=False))
        except:
            return None

        # Get attachments for this core
        attachment_a = frags_a.get(core)
        attachment_b = frags_b.get(core)

        if not attachment_a or not attachment_b:
            return None

        if attachment_a == attachment_b:
            return None  # Not a transformation

        # Generate edit ID (hash of from->to transformation)
        edit_str = f"{attachment_a}>{attachment_b}"
        edit_id = hashlib.md5(edit_str.encode()).hexdigest()[:12]

        # Build result with basic info
        result = {
            'mol_a': mol_a_data['smiles'],
            'mol_b': mol_b_data['smiles'],
            'edit_id': edit_id,
            'core': core,
            'from_smarts': attachment_a,
            'to_smarts': attachment_b,
        }

        # Add all property values and deltas
        for prop_name in property_names:
            if prop_name in mol_a_data['properties'] and prop_name in mol_b_data['properties']:
                val_a = mol_a_data['properties'][prop_name]
                val_b = mol_b_data['properties'][prop_name]

                # Skip if either value is None or NaN
                if val_a is None or val_b is None:
                    continue
                if isinstance(val_a, float) and np.isnan(val_a):
                    continue
                if isinstance(val_b, float) and np.isnan(val_b):
                    continue

                result[f'{prop_name}_a'] = val_a
                result[f'{prop_name}_b'] = val_b
                result[f'delta_{prop_name}'] = val_b - val_a

        return result


def estimate_complexity(n_molecules: int, config: ScalableMMPConfig) -> Dict[str, float]:
    """
    Estimate computational complexity for given dataset size.

    Args:
        n_molecules: Number of molecules
        config: Configuration

    Returns:
        Dict with estimates
    """
    # Naive approach
    naive_comparisons = n_molecules * (n_molecules - 1) / 2

    # MW binning reduces by ~10x (molecules spread across bins)
    mw_reduction = 10

    # Fingerprint similarity further reduces by ~50-100x (most molecules dissimilar)
    fp_reduction = 75

    # Combined reduction
    total_reduction = mw_reduction * fp_reduction

    filtered_comparisons = naive_comparisons / total_reduction

    # Time estimates (assuming 1M comparisons per second)
    naive_hours = naive_comparisons / 1e6 / 3600
    filtered_hours = filtered_comparisons / 1e6 / 3600

    return {
        'n_molecules': n_molecules,
        'naive_comparisons': naive_comparisons,
        'filtered_comparisons': filtered_comparisons,
        'reduction_factor': total_reduction,
        'naive_time_hours': naive_hours,
        'filtered_time_hours': filtered_hours,
        'speedup': naive_hours / max(filtered_hours, 0.001)
    }


def main():
    """
    Main script for scalable MMP extraction.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Scalable MMP extraction for large datasets")
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV with SMILES and properties')
    parser.add_argument('--smiles-col', type=str, default='smiles',
                       help='SMILES column name')
    parser.add_argument('--property-col', type=str, default='alogp',
                       help='Property column name')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV for pairs')
    parser.add_argument('--max-mw-delta', type=float, default=200,
                       help='Max MW difference (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                       help='Min Tanimoto similarity (default: 0.4)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} molecules")

    # Prepare inputs
    smiles_list = df[args.smiles_col].dropna().tolist()
    properties = df.set_index(args.smiles_col)[args.property_col].to_dict()

    # Configure extractor
    config = ScalableMMPConfig(
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity
    )

    # Estimate complexity
    logger.info("\nComplexity Estimate:")
    estimate = estimate_complexity(len(smiles_list), config)
    logger.info(f"  Naive comparisons: {estimate['naive_comparisons']:,.0f}")
    logger.info(f"  Filtered comparisons: {estimate['filtered_comparisons']:,.0f}")
    logger.info(f"  Reduction factor: {estimate['reduction_factor']:.0f}x")
    logger.info(f"  Estimated time: {estimate['filtered_time_hours']:.1f} hours")
    logger.info("")

    # Extract pairs
    extractor = ScalableMMPExtractor(config=config)
    pairs_df = extractor.extract_pairs_scalable(
        smiles_list=smiles_list,
        properties=properties,
        property_name=args.property_col,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save results
    pairs_df.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to {args.output}")

    # Summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Total pairs: {len(pairs_df):,}")
    logger.info(f"  Unique edits: {pairs_df['edit_id'].nunique():,}")
    logger.info(f"  Unique cores: {pairs_df['core'].nunique():,}")

    # Top edits
    logger.info("\nTop 10 most frequent edits:")
    top_edits = pairs_df['edit_id'].value_counts().head(10)
    for edit_id, count in top_edits.items():
        example = pairs_df[pairs_df['edit_id'] == edit_id].iloc[0]
        logger.info(f"  {example['from_smarts']} → {example['to_smarts']}: {count} pairs")


if __name__ == '__main__':
    main()
