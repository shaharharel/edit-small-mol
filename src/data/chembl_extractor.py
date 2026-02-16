"""
ChEMBL data extraction for molecular pairs.

This module provides a unified pipeline for:
1. Downloading ChEMBL molecules and bioactivity data
2. Generating matched molecular pairs (MMPs)
3. Outputting pairs in long format

Usage:
    from src.data import ChEMBLPairExtractor, ChEMBLConfig

    config = ChEMBLConfig(n_molecules=10000, max_cuts=1)
    extractor = ChEMBLPairExtractor(config)
    pairs_df = extractor.run()
"""

import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from typing import List

from .base_extractor import PairDataExtractor, ExtractionConfig
from .mmp_long_format import LongFormatMMPExtractor

logger = logging.getLogger(__name__)


@dataclass
class ChEMBLConfig(ExtractionConfig):
    """
    Configuration for ChEMBL molecule download and pair extraction.
    """

    # === ChEMBL Download Settings ===
    chembl_version: Optional[str] = None  # None = latest
    n_molecules: int = 50000
    activity_types: List[str] = field(default_factory=lambda: ['IC50', 'Ki', 'EC50', 'Kd'])
    min_assays_per_molecule: int = 1

    # === Target/Label Selection ===
    min_molecules_per_label: Optional[int] = None  # Minimum molecules per target (None = no filter)
    top_n_targets: int = 20  # Select top N targets by molecule count
    specific_targets: Optional[List[str]] = None  # Specific target ChEMBL IDs (e.g., ['CHEMBL203', 'CHEMBL1862'])
    target_mix: Optional[dict] = None  # Mix of target classes (e.g., {'Kinase': 5, 'GPCR': 3})

    # === MMP Extraction Settings ===
    max_cuts: int = 1
    max_mw_delta: float = 200.0
    min_similarity: float = 0.4

    # === Property Filtering ===
    property_filter: Optional[List[str]] = None
    exclude_computed: bool = False

    # === Checkpointing ===
    checkpoint_every: int = 1000
    resume_from_checkpoint: bool = True

    # === Paths ===
    data_dir: Path = field(default_factory=lambda: Path("data/chembl"))
    output_name: str = "chembl_pairs"

    @property
    def db_dir(self) -> Path:
        """Directory for ChEMBL SQLite database."""
        return self.data_dir / "db"

    @property
    def molecules_file(self) -> Path:
        return self.data_dir / f"molecules_{self.n_molecules}.csv"

    @property
    def bioactivity_file(self) -> Path:
        return self.data_dir / f"bioactivity_{self.n_molecules}.csv"

    @property
    def pairs_output(self) -> Path:
        return self.output_dir / f"{self.output_name}_n{self.n_molecules}_cuts{self.max_cuts}.csv"

    def __post_init__(self):
        super().__post_init__()
        self.db_dir.mkdir(parents=True, exist_ok=True)


class ChEMBLPairExtractor(PairDataExtractor):
    """
    Extract molecular pairs from ChEMBL database.

    This extractor:
    1. Downloads ChEMBL SQLite database (if not cached)
    2. Extracts molecules and bioactivity data
    3. Generates matched molecular pairs using MMPs
    4. Outputs in long format
    """

    def __init__(self, config: ChEMBLConfig):
        super().__init__(config)
        self.config: ChEMBLConfig = config

        # Setup chembl-downloader
        os.environ['PYSTOW_HOME'] = str(self.config.db_dir.absolute())

        try:
            import chembl_downloader
            self.chembl = chembl_downloader
            logger.info("✓ chembl-downloader loaded")
        except ImportError:
            logger.error("✗ chembl-downloader not installed!")
            logger.error("  Install with: pip install chembl-downloader")
            raise

    def _check_database_cached(self) -> bool:
        """Check if ChEMBL database is downloaded."""
        version = self.config.chembl_version or self.chembl.latest()
        db_path = self.config.db_dir / 'chembl' / str(version) / f'chembl_{version}.db'
        return db_path.exists()

    def _check_molecules_cached(self) -> bool:
        """Check if molecules and bioactivity CSVs exist."""
        return (
            self.config.molecules_file.exists() and
            self.config.bioactivity_file.exists()
        )

    def _filter_targets(self, bioactivity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter bioactivity data based on target selection criteria.

        Priority order (only one method is applied):
        1. specific_targets - exact target list
        2. target_mix - balanced mix by target class
        3. top_n_targets + min_molecules_per_label - top targets with minimum threshold

        Args:
            bioactivity_df: DataFrame with bioactivity measurements

        Returns:
            Filtered bioactivity DataFrame
        """
        if bioactivity_df.empty:
            return bioactivity_df

        # Skip filtering for computed properties
        bioactivity_only = bioactivity_df[bioactivity_df['target_name'] != 'computed'].copy()
        computed_only = bioactivity_df[bioactivity_df['target_name'] == 'computed'].copy()

        if bioactivity_only.empty:
            logger.info("  No bioactivity targets to filter")
            return bioactivity_df

        original_targets = bioactivity_only['target_chembl_id'].nunique()
        original_measurements = len(bioactivity_only)

        # Method 1: Specific targets
        if self.config.specific_targets:
            logger.info(f"  Filtering to {len(self.config.specific_targets)} specific targets...")
            bioactivity_only = bioactivity_only[
                bioactivity_only['target_chembl_id'].isin(self.config.specific_targets)
            ]
            logger.info(f"    Kept {bioactivity_only['target_chembl_id'].nunique()} targets, "
                       f"{len(bioactivity_only):,} measurements")

        # Method 2: Target mix (balanced sampling by class)
        elif self.config.target_mix:
            logger.info(f"  Applying target mix: {self.config.target_mix}")

            # Query target classes from ChEMBL
            target_ids = "', '".join(bioactivity_only['target_chembl_id'].unique().tolist())
            class_query = f"""
            SELECT
                td.chembl_id,
                td.pref_name,
                tc.protein_class_desc
            FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_class cc ON tc.component_id = cc.component_id
            WHERE td.chembl_id IN ('{target_ids}')
            """

            try:
                target_classes = self.chembl.query(class_query)

                selected_targets = []
                for class_name, count in self.config.target_mix.items():
                    # Find targets matching this class (case-insensitive substring match)
                    class_targets = target_classes[
                        target_classes['protein_class_desc'].str.contains(class_name, case=False, na=False)
                    ]['chembl_id'].unique()

                    # Count molecules per target
                    target_mol_counts = bioactivity_only[
                        bioactivity_only['target_chembl_id'].isin(class_targets)
                    ].groupby('target_chembl_id')['chembl_id'].nunique().sort_values(ascending=False)

                    # Select top N targets from this class
                    top_class_targets = target_mol_counts.head(count).index.tolist()
                    selected_targets.extend(top_class_targets)

                    logger.info(f"    {class_name}: selected {len(top_class_targets)}/{len(class_targets)} targets")

                bioactivity_only = bioactivity_only[
                    bioactivity_only['target_chembl_id'].isin(selected_targets)
                ]
                logger.info(f"    Total: {len(selected_targets)} targets, {len(bioactivity_only):,} measurements")

            except Exception as e:
                logger.warning(f"  Failed to apply target mix: {e}")
                logger.warning("  Falling back to top_n_targets method")
                # Fall through to Method 3

        # Method 3: Top N targets (default)
        if not self.config.specific_targets and not self.config.target_mix:
            # Count molecules per target
            target_mol_counts = bioactivity_only.groupby('target_chembl_id')['chembl_id'].nunique()

            # Apply minimum threshold if specified
            if self.config.min_molecules_per_label:
                valid_targets = target_mol_counts[
                    target_mol_counts >= self.config.min_molecules_per_label
                ].index.tolist()
                logger.info(f"  Filtering targets with ≥{self.config.min_molecules_per_label} molecules...")
                logger.info(f"    {len(valid_targets)}/{original_targets} targets passed threshold")
                target_mol_counts = target_mol_counts[target_mol_counts.index.isin(valid_targets)]

            # Select top N targets
            top_targets = target_mol_counts.nlargest(self.config.top_n_targets).index.tolist()
            bioactivity_only = bioactivity_only[
                bioactivity_only['target_chembl_id'].isin(top_targets)
            ]

            logger.info(f"  Selected top {len(top_targets)} targets:")
            for target_id in top_targets[:5]:  # Show top 5
                target_name = bioactivity_only[bioactivity_only['target_chembl_id'] == target_id]['target_name'].iloc[0]
                mol_count = target_mol_counts[target_id]
                logger.info(f"    {target_id} ({target_name}): {mol_count} molecules")
            if len(top_targets) > 5:
                logger.info(f"    ... and {len(top_targets) - 5} more")

        # Combine filtered bioactivity with computed properties
        filtered_df = pd.concat([bioactivity_only, computed_only], ignore_index=True)

        logger.info(f"  Filtering summary:")
        logger.info(f"    Targets: {original_targets} → {bioactivity_only['target_chembl_id'].nunique()}")
        logger.info(f"    Measurements: {original_measurements:,} → {len(bioactivity_only):,}")

        return filtered_df

    def download_if_needed(self) -> bool:
        """
        Download ChEMBL data if not already cached.

        Returns:
            True if download was performed
        """
        # Check if CSVs already exist
        if self._check_molecules_cached():
            logger.info(f"✓ Using cached data:")
            logger.info(f"  Molecules: {self.config.molecules_file}")
            logger.info(f"  Bioactivity: {self.config.bioactivity_file}")
            return False

        # Check database
        if not self._check_database_cached():
            logger.info("Downloading ChEMBL database (this may take a while)...")
            version = self.config.chembl_version or self.chembl.latest()
            logger.info(f"  Version: {version}")
            logger.info(f"  Database dir: {self.config.db_dir}")

        # Extract molecules and bioactivity
        logger.info(f"Extracting {self.config.n_molecules:,} molecules with bioactivity...")

        # Query for molecules with bioactivity
        query = f"""
        SELECT DISTINCT
            md.chembl_id,
            cs.canonical_smiles,
            cp.mw_freebase as mw,
            cp.alogp,
            cp.hbd,
            cp.hba,
            cp.psa,
            cp.rtb,
            cp.aromatic_rings,
            cp.heavy_atoms,
            cp.qed_weighted,
            cp.num_ro5_violations,
            cp.np_likeness_score
        FROM
            molecule_dictionary md
        JOIN
            compound_structures cs ON md.molregno = cs.molregno
        JOIN
            compound_properties cp ON md.molregno = cp.molregno
        JOIN
            activities act ON md.molregno = act.molregno
        WHERE
            cs.canonical_smiles IS NOT NULL
            AND cp.mw_freebase < 1000
            AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
        LIMIT {self.config.n_molecules}
        """

        molecules_df = self.chembl.query(query)
        logger.info(f"  ✓ Extracted {len(molecules_df):,} molecules")

        # Get all bioactivity for these molecules
        chembl_ids = "', '".join(molecules_df['chembl_id'].tolist())

        bioactivity_query = f"""
        SELECT
            md.chembl_id,
            act.standard_type || '_' || td.chembl_id as property_name,
            act.standard_value,
            act.standard_units,
            td.pref_name as target_name,
            td.chembl_id as target_chembl_id,
            act.doc_id,
            act.assay_id
        FROM
            activities act
        JOIN
            molecule_dictionary md ON act.molregno = md.molregno
        JOIN
            target_dictionary td ON act.tid = td.tid
        WHERE
            md.chembl_id IN ('{chembl_ids}')
            AND act.standard_value IS NOT NULL
            AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
            AND act.standard_units = 'nM'
        """

        bioactivity_df = self.chembl.query(bioactivity_query)
        logger.info(f"  ✓ Extracted {len(bioactivity_df):,} bioactivity measurements")

        # Apply target filtering
        bioactivity_df = self._filter_targets(bioactivity_df)

        # Add computed properties as "bioactivity" for consistency
        computed_props = ['mw', 'alogp', 'hbd', 'hba', 'psa', 'rtb',
                         'aromatic_rings', 'heavy_atoms', 'qed_weighted',
                         'num_ro5_violations', 'np_likeness_score']

        computed_bio = []
        for prop in computed_props:
            if prop in molecules_df.columns:
                prop_df = molecules_df[['chembl_id', prop]].copy()
                prop_df = prop_df.dropna(subset=[prop])
                prop_df['property_name'] = prop
                prop_df['standard_value'] = prop_df[prop]
                prop_df['standard_units'] = 'computed'
                prop_df['target_name'] = 'computed'
                prop_df['target_chembl_id'] = ''
                prop_df['doc_id'] = None
                prop_df['assay_id'] = None
                prop_df = prop_df[['chembl_id', 'property_name', 'standard_value', 'standard_units', 'target_name', 'target_chembl_id', 'doc_id', 'assay_id']]
                computed_bio.append(prop_df)

        if computed_bio:
            bioactivity_df = pd.concat([bioactivity_df] + computed_bio, ignore_index=True)

        # Save
        molecules_df.to_csv(self.config.molecules_file, index=False)
        bioactivity_df.to_csv(self.config.bioactivity_file, index=False)

        logger.info(f"✓ Saved to:")
        logger.info(f"  {self.config.molecules_file}")
        logger.info(f"  {self.config.bioactivity_file}")

        return True

    def extract_pairs(self, **kwargs) -> pd.DataFrame:
        """
        Extract molecular pairs using MMP algorithm.

        Returns:
            DataFrame with columns: mol_a, mol_b, edit_smiles, property_name, value_a, value_b, delta, ...
        """
        # Load data
        logger.info("Loading molecules and bioactivity...")
        molecules_df = pd.read_csv(self.config.molecules_file)
        bioactivity_df = pd.read_csv(self.config.bioactivity_file)

        logger.info(f"  Molecules: {len(molecules_df):,}")
        logger.info(f"  Bioactivity: {len(bioactivity_df):,}")

        # Property filter
        property_filter = None
        if self.config.property_filter:
            property_filter = set(self.config.property_filter)
        elif self.config.exclude_computed:
            computed_props = {'mw', 'alogp', 'hbd', 'hba', 'psa', 'rtb',
                             'aromatic_rings', 'heavy_atoms', 'qed_weighted',
                             'num_ro5_violations', 'np_likeness_score'}
            all_props = set(bioactivity_df['property_name'].unique())
            property_filter = all_props - computed_props

        # Extract pairs
        logger.info("Extracting matched molecular pairs...")
        extractor = LongFormatMMPExtractor(max_cuts=self.config.max_cuts)

        pairs_df = extractor.extract_pairs_long_format(
            molecules_df=molecules_df,
            bioactivity_df=bioactivity_df,
            max_mw_delta=self.config.max_mw_delta,
            min_similarity=self.config.min_similarity,
            checkpoint_dir=str(self.config.checkpoint_dir),
            checkpoint_every=self.config.checkpoint_every,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
            property_filter=property_filter
        )

        # Save
        pairs_df.to_csv(self.config.pairs_output, index=False)
        logger.info(f"✓ Saved pairs to: {self.config.pairs_output}")

        return pairs_df
