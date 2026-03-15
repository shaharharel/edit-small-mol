"""
Overlapping assay extractor for cross-lab noise analysis.

Implements the methodology from Landrum & Riniker (JCIM 2024):
"Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise"

This module extracts bioactivity data from ChEMBL with proper curation,
identifies overlapping assay pairs (same target, shared compounds),
and generates within-assay vs cross-assay molecular pairs for
demonstrating that the edit effect framework overcomes inter-assay noise.

Key concepts:
- "Goldilocks" assays: 20-100 compounds (avoids HTS screening noise)
- Overlapping assay pairs: same target, >=5 shared compounds
- Minimal curation: basic quality filters only
- Maximal curation: strict metadata matching with MD5 conditions hash
  (assay_type, assay_organism, assay_cell_type, assay_subcellular_fraction,
   assay_tissue, assay_strain, assay_tax_id, assay_category, bao_format,
   standard_type), confidence_score=9, mutant exclusion,
   same-publication duplicate removal
- Within-assay MMPs: pairs where both molecules measured in same assay
- Cross-assay MMPs: pairs where molecules measured in different assays

The edit effect framework should show that within-assay delta predictions
are more consistent than cross-assay ones, and that learning edits
explicitly is more robust to inter-assay noise than subtraction baselines.

Reference: https://github.com/rinikerlab/overlapping_assays
"""

import hashlib
import logging
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

from .base_extractor import ExtractionConfig, PairDataExtractor
from .mmp_long_format import LongFormatMMPExtractor

# Metadata fields used to build the MD5 conditions hash for maximal curation.
# Two assays are considered "same conditions" only when all of these match.
_MAXIMAL_CURATION_FIELDS = [
    "assay_type",
    "assay_organism",
    "assay_cell_type",
    "assay_subcellular_fraction",
    "assay_tissue",
    "assay_strain",
    "assay_tax_id",
    "assay_category",
    "bao_format",
    "standard_type",
]

logger = logging.getLogger(__name__)


def _compute_conditions_hash(row: pd.Series) -> str:
    """
    Compute MD5 hash of assay conditions metadata.

    This creates a deterministic fingerprint of the experimental conditions
    so that two assays can be compared for protocol compatibility by
    simply comparing their hash values (maximal curation).

    Fields used (from Landrum & Riniker 2024):
    assay_type, assay_organism, assay_cell_type, assay_subcellular_fraction,
    assay_tissue, assay_strain, assay_tax_id, assay_category, bao_format,
    standard_type
    """
    parts = []
    for field in _MAXIMAL_CURATION_FIELDS:
        val = row.get(field, "")
        parts.append(str(val) if pd.notna(val) else "")
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OverlappingAssayConfig(ExtractionConfig):
    """
    Configuration for overlapping assay extraction.

    Two curation levels following Landrum & Riniker (2024):
    - minimal: basic quality filters (non-null pchembl, nM units, no validity comments)
    - maximal: additionally require matching assay metadata across pairs
    """

    # --- ChEMBL connection ---
    chembl_version: Optional[str] = None
    db_dir: Path = field(default_factory=lambda: Path("data/chembl_db"))

    # --- Activity filters ---
    activity_types: List[str] = field(default_factory=lambda: ["IC50", "Ki"])
    curation_level: str = "minimal"  # "minimal" or "maximal"

    # --- Assay selection ---
    min_compounds_per_assay: int = 20
    max_compounds_per_assay: Optional[int] = 100  # None = no upper limit
    min_shared_compounds: int = 5
    target_types: List[str] = field(default_factory=lambda: ["SINGLE PROTEIN"])
    min_confidence_score: int = 9  # ChEMBL target confidence (maximal only)

    # --- Pair generation ---
    pair_methods: List[str] = field(default_factory=lambda: ["mmp"])
    # Supported: "mmp", "tanimoto", "scaffold"
    tanimoto_threshold: float = 0.7
    max_mw_delta: float = 200.0
    max_cuts: int = 1

    # --- Output ---
    data_dir: Path = field(default_factory=lambda: Path("data/overlapping_assays"))
    output_name: str = "overlapping_assay_pairs"

    @property
    def activities_file(self) -> Path:
        return self.data_dir / f"activities_{self.curation_level}.csv"

    @property
    def assay_pairs_file(self) -> Path:
        return self.data_dir / f"assay_pairs_{self.curation_level}.csv"

    @property
    def molecule_properties_file(self) -> Path:
        return self.data_dir / f"molecule_pIC50_{self.curation_level}.csv"

    @property
    def pairs_output(self) -> Path:
        methods = "_".join(sorted(self.pair_methods))
        return (
            self.output_dir
            / f"{self.output_name}_{self.curation_level}_{methods}.csv"
        )

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.db_dir, str):
            object.__setattr__(self, "db_dir", Path(self.db_dir))
        self.db_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class OverlappingAssayExtractor(PairDataExtractor):
    """
    Extract molecular pairs from overlapping ChEMBL assays.

    Pipeline:
    1. Query ChEMBL for IC50/Ki activities with curation filters
    2. Identify "Goldilocks" assays (20-100 compounds)
    3. Find overlapping assay pairs (same target, >=5 shared compounds)
    4. Generate within-assay and cross-assay molecular pairs
    5. Store absolute pIC50 values per molecule for baseline comparison
    """

    def __init__(self, config: OverlappingAssayConfig):
        super().__init__(config)
        self.config: OverlappingAssayConfig = config

        os.environ["PYSTOW_HOME"] = str(self.config.db_dir.absolute())

        try:
            import chembl_downloader
            self.chembl = chembl_downloader
            logger.info("chembl-downloader loaded")
        except ImportError:
            logger.error("chembl-downloader not installed. Install with: pip install chembl-downloader")
            raise

    # ------------------------------------------------------------------
    # Download / query
    # ------------------------------------------------------------------

    def download_if_needed(self) -> bool:
        """Download ChEMBL activities if not cached."""
        if self.config.activities_file.exists():
            logger.info(f"Using cached activities: {self.config.activities_file}")
            return False

        logger.info("Querying ChEMBL for bioactivity data...")
        activities_df = self._query_activities()
        activities_df.to_csv(self.config.activities_file, index=False)
        logger.info(f"Saved {len(activities_df):,} activities to {self.config.activities_file}")
        return True

    def _query_activities(self) -> pd.DataFrame:
        """
        Query ChEMBL for IC50/Ki activities with curation filters.

        Minimal curation:
        - pchembl_value not null
        - standard_units = 'nM'
        - data_validity_comment is null
        - standard_relation = '='
        - single protein targets
        - has associated document

        Maximal curation adds:
        - confidence_score = 9
        - exclude mutant/variant assay descriptions
        - same-publication duplicate removal (keep largest assay per doc_id)
        - MD5 conditions hash from full assay metadata for pair matching
        """
        activity_types_sql = ", ".join(f"'{t}'" for t in self.config.activity_types)
        target_types_sql = ", ".join(f"'{t}'" for t in self.config.target_types)

        base_query = f"""
        SELECT
            act.activity_id,
            md.chembl_id AS molecule_chembl_id,
            cs.canonical_smiles AS smiles,
            act.pchembl_value,
            act.standard_type,
            act.standard_value,
            act.standard_units,
            act.data_validity_comment,
            a.assay_id,
            a.chembl_id AS assay_chembl_id,
            a.description AS assay_description,
            a.assay_type,
            a.assay_organism,
            a.assay_cell_type,
            a.assay_subcellular_fraction,
            a.assay_tissue,
            a.assay_strain,
            a.assay_tax_id,
            a.assay_category,
            a.bao_format,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.target_type,
            a.confidence_score,
            docs.doc_id,
            docs.year AS doc_year
        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON act.molregno = cs.molregno
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        LEFT JOIN docs ON act.doc_id = docs.doc_id
        WHERE
            act.pchembl_value IS NOT NULL
            AND act.standard_units = 'nM'
            AND act.data_validity_comment IS NULL
            AND act.standard_type IN ({activity_types_sql})
            AND td.target_type IN ({target_types_sql})
            AND act.standard_relation = '='
            AND docs.doc_id IS NOT NULL
        """

        if self.config.curation_level == "maximal":
            base_query += f"""
            AND a.confidence_score >= {self.config.min_confidence_score}
            AND a.description NOT LIKE '%mutant%'
            AND a.description NOT LIKE '%Mutant%'
            AND a.description NOT LIKE '%MUTANT%'
            AND a.description NOT LIKE '%variant%'
            AND a.description NOT LIKE '%Variant%'
            AND a.description NOT LIKE '%VARIANT%'
            """

        logger.info(f"Running ChEMBL query (curation={self.config.curation_level})...")
        df = self.chembl.query(base_query)
        logger.info(f"  Raw results: {len(df):,} activities")

        # Deduplicate: keep one measurement per (molecule, assay, standard_type)
        # When duplicates exist, keep the one with the median pchembl_value
        before = len(df)
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["pchembl_value"])
        df = (
            df.groupby(["molecule_chembl_id", "assay_id", "standard_type"])
            .apply(lambda g: g.iloc[(g["pchembl_value"] - g["pchembl_value"].median()).abs().argsort()[:1]])
            .reset_index(drop=True)
        )
        logger.info(f"  After dedup: {len(df):,} (removed {before - len(df):,} duplicates)")

        # Maximal curation: remove same-publication duplicates
        if self.config.curation_level == "maximal":
            df = self._remove_same_publication_duplicates(df)

        # Compute conditions hash for maximal curation pair matching
        if self.config.curation_level == "maximal":
            df["conditions_hash"] = df.apply(
                lambda row: _compute_conditions_hash(row), axis=1
            )
            logger.info(f"  Computed conditions hash for {df['conditions_hash'].nunique():,} unique conditions")

        return df

    @staticmethod
    def _remove_same_publication_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        For maximal curation: when multiple assays from the same publication
        target the same protein, keep only the largest assay (most compounds).

        This avoids counting redundant measurements from the same lab/paper
        as independent evidence.
        """
        before = len(df)
        before_assays = df["assay_id"].nunique()

        # Count compounds per assay
        assay_sizes = df.groupby("assay_id")["molecule_chembl_id"].nunique()

        # Group assays by (doc_id, target_chembl_id, standard_type)
        assay_doc_target = (
            df[["assay_id", "doc_id", "target_chembl_id", "standard_type"]]
            .drop_duplicates(subset=["assay_id"])
        )

        # For each (doc, target, type) group, keep only the largest assay
        assays_to_drop = set()
        for _, group in assay_doc_target.groupby(["doc_id", "target_chembl_id", "standard_type"]):
            if len(group) <= 1:
                continue
            group_assay_ids = group["assay_id"].tolist()
            sizes = {aid: assay_sizes.get(aid, 0) for aid in group_assay_ids}
            largest = max(sizes, key=sizes.get)
            assays_to_drop.update(aid for aid in group_assay_ids if aid != largest)

        if assays_to_drop:
            df = df[~df["assay_id"].isin(assays_to_drop)]
            logger.info(
                f"  Same-publication dedup: removed {len(assays_to_drop)} assays "
                f"({before_assays} -> {df['assay_id'].nunique()}), "
                f"activities {before:,} -> {len(df):,}"
            )

        return df

    # ------------------------------------------------------------------
    # Assay pair identification
    # ------------------------------------------------------------------

    def _find_goldilocks_assays(self, activities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify 'Goldilocks' assays with 20-100 compounds.

        These assays are large enough for meaningful statistics but small
        enough to avoid HTS screening assays that introduce systematic noise.
        """
        min_c = self.config.min_compounds_per_assay
        max_c = self.config.max_compounds_per_assay

        assay_sizes = activities_df.groupby("assay_id")["molecule_chembl_id"].nunique()
        mask = assay_sizes >= min_c
        if max_c is not None:
            mask = mask & (assay_sizes <= max_c)
        goldilocks = assay_sizes[mask]

        upper_label = str(max_c) if max_c is not None else "∞"
        logger.info(f"  Total assays: {len(assay_sizes):,}")
        logger.info(f"  Goldilocks assays ({min_c}-{upper_label} compounds): {len(goldilocks):,}")

        return goldilocks

    def _find_overlapping_pairs(
        self, activities_df: pd.DataFrame, goldilocks_assay_ids: Set[int]
    ) -> pd.DataFrame:
        """
        Find pairs of assays targeting the same protein with shared compounds.

        For maximal curation, pairs must also share the same conditions hash
        (MD5 of all assay metadata fields).

        Returns DataFrame with columns:
        - assay_id_1, assay_id_2: the two assay IDs
        - target_chembl_id: shared target
        - n_shared: number of shared compounds
        - n_total_1, n_total_2: total compounds in each assay
        - conditions_hash_1, conditions_hash_2: (maximal only) MD5 hashes
        """
        # Filter to goldilocks assays
        gl_df = activities_df[activities_df["assay_id"].isin(goldilocks_assay_ids)].copy()

        # Group molecules by (assay, target)
        assay_target_mols = (
            gl_df.groupby(["assay_id", "target_chembl_id"])["molecule_chembl_id"]
            .apply(set)
            .reset_index()
        )
        assay_target_mols.columns = ["assay_id", "target_chembl_id", "molecules"]

        # Build per-assay conditions hash for maximal curation
        assay_conditions_hash: Dict[int, str] = {}
        if self.config.curation_level == "maximal" and "conditions_hash" in gl_df.columns:
            # Each assay should have one consistent hash (use first row per assay)
            for assay_id, grp in gl_df.groupby("assay_id"):
                assay_conditions_hash[assay_id] = grp["conditions_hash"].iloc[0]

        # Find overlapping pairs per target
        pairs = []
        for target_id, group in assay_target_mols.groupby("target_chembl_id"):
            if len(group) < 2:
                continue

            assay_list = group.to_dict("records")
            for i in range(len(assay_list)):
                for j in range(i + 1, len(assay_list)):
                    a1 = assay_list[i]
                    a2 = assay_list[j]

                    shared = a1["molecules"] & a2["molecules"]
                    if len(shared) < self.config.min_shared_compounds:
                        continue

                    aid1 = a1["assay_id"]
                    aid2 = a2["assay_id"]

                    # Maximal curation: require matching conditions hash
                    hash1, hash2 = "", ""
                    if self.config.curation_level == "maximal":
                        hash1 = assay_conditions_hash.get(aid1, "")
                        hash2 = assay_conditions_hash.get(aid2, "")
                        if not hash1 or not hash2 or hash1 != hash2:
                            continue

                    pair_entry = {
                        "assay_id_1": aid1,
                        "assay_id_2": aid2,
                        "target_chembl_id": target_id,
                        "n_shared": len(shared),
                        "n_total_1": len(a1["molecules"]),
                        "n_total_2": len(a2["molecules"]),
                    }
                    if self.config.curation_level == "maximal":
                        pair_entry["conditions_hash"] = hash1

                    pairs.append(pair_entry)

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"  Found {len(pairs_df):,} overlapping assay pairs")

        if not pairs_df.empty:
            logger.info(
                f"  Shared compounds per pair: "
                f"min={pairs_df['n_shared'].min()}, "
                f"median={pairs_df['n_shared'].median():.0f}, "
                f"max={pairs_df['n_shared'].max()}"
            )
            logger.info(f"  Unique targets: {pairs_df['target_chembl_id'].nunique()}")

        return pairs_df

    # ------------------------------------------------------------------
    # Molecule-level pIC50 table
    # ------------------------------------------------------------------

    def _build_molecule_properties(self, activities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build molecule-level pIC50 table (absolute values for baseline).

        Returns DataFrame with columns:
        - molecule_chembl_id, smiles, assay_id, target_chembl_id,
          standard_type, pIC50 (= pchembl_value)

        This table is needed so the baseline predictor can train on
        absolute pIC50 values and compute deltas via subtraction.
        """
        cols = [
            "molecule_chembl_id",
            "smiles",
            "assay_id",
            "assay_chembl_id",
            "target_chembl_id",
            "target_name",
            "standard_type",
            "pchembl_value",
            "doc_id",
        ]
        existing_cols = [c for c in cols if c in activities_df.columns]
        mol_props = activities_df[existing_cols].copy()
        mol_props = mol_props.rename(columns={"pchembl_value": "pIC50"})
        return mol_props

    # ------------------------------------------------------------------
    # Pair generation methods
    # ------------------------------------------------------------------

    def _generate_mmp_pairs(
        self,
        activities_df: pd.DataFrame,
        assay_pairs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate within-assay MMPs grouped by assay_id.

        For each assay, find MMP pairs among the molecules measured in
        that assay.  Then annotate whether both molecules are in the
        same assay pair (is_within_assay=True) or different assays.
        """
        logger.info("Generating MMP pairs (within-assay grouping)...")

        # Collect all assay IDs involved in overlapping pairs
        all_assay_ids = set(assay_pairs_df["assay_id_1"]) | set(assay_pairs_df["assay_id_2"])

        # Filter activities to those assays
        relevant = activities_df[activities_df["assay_id"].isin(all_assay_ids)].copy()
        logger.info(f"  Relevant activities: {len(relevant):,} across {len(all_assay_ids)} assays")

        # Build molecules_df for MMP extractor (unique SMILES)
        unique_mols = relevant[["molecule_chembl_id", "smiles"]].drop_duplicates(subset=["smiles"])
        unique_mols = unique_mols.rename(columns={"molecule_chembl_id": "chembl_id"})
        logger.info(f"  Unique molecules: {len(unique_mols):,}")

        # Build per-assay bioactivity in the format expected by LongFormatMMPExtractor
        # Group by assay and generate pairs within each assay
        all_pair_rows = []
        assay_groups = relevant.groupby("assay_id")

        # Build assay pair lookup: assay_id -> set of partner assay_ids
        assay_pair_lookup: Dict[int, Set[int]] = defaultdict(set)
        assay_pair_id_lookup: Dict[Tuple[int, int], int] = {}
        for idx, row in assay_pairs_df.iterrows():
            a1, a2 = row["assay_id_1"], row["assay_id_2"]
            assay_pair_lookup[a1].add(a2)
            assay_pair_lookup[a2].add(a1)
            pair_key = (min(a1, a2), max(a1, a2))
            assay_pair_id_lookup[pair_key] = idx

        # For efficiency: fragment molecules once
        mmp_extractor = LongFormatMMPExtractor(max_cuts=self.config.max_cuts)

        # Build global fragment index
        logger.info("  Fragmenting molecules...")
        smiles_to_frags = {}
        for smiles in unique_mols["smiles"]:
            frags = mmp_extractor.fragment_molecule(smiles)
            if frags:
                smiles_to_frags[smiles] = frags

        logger.info(f"  Fragmented {len(smiles_to_frags):,} molecules")

        # Build core -> smiles index
        core_to_smiles: Dict[str, List[str]] = defaultdict(list)
        for smiles, frags in smiles_to_frags.items():
            for core in frags:
                core_to_smiles[core].append(smiles)

        # Now generate pairs per assay
        logger.info("  Generating within-assay MMP pairs...")
        pair_count = 0
        for assay_id, assay_group in assay_groups:
            assay_smiles = set(assay_group["smiles"].unique())
            if len(assay_smiles) < 2:
                continue

            # Build pIC50 lookup for this assay
            pic50_lookup = {}
            for _, row in assay_group.iterrows():
                pic50_lookup[row["smiles"]] = {
                    "pIC50": row["pchembl_value"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row.get("target_name", ""),
                    "standard_type": row.get("standard_type", ""),
                    "doc_id": row.get("doc_id"),
                }

            # Find MMP pairs within this assay
            seen_pairs = set()
            for core, core_smiles_list in core_to_smiles.items():
                # Filter to molecules in this assay
                in_assay = [s for s in core_smiles_list if s in assay_smiles]
                if len(in_assay) < 2:
                    continue

                for i in range(len(in_assay)):
                    for j in range(i + 1, len(in_assay)):
                        s_a, s_b = in_assay[i], in_assay[j]
                        pair_key = (min(s_a, s_b), max(s_a, s_b))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        # MW filter
                        try:
                            mol_a = Chem.MolFromSmiles(s_a)
                            mol_b = Chem.MolFromSmiles(s_b)
                            if mol_a is None or mol_b is None:
                                continue
                            mw_a = Descriptors.MolWt(mol_a)
                            mw_b = Descriptors.MolWt(mol_b)
                            if abs(mw_a - mw_b) > self.config.max_mw_delta:
                                continue
                        except Exception:
                            continue

                        # Build edit SMILES from fragments
                        frags_a = smiles_to_frags.get(s_a, {})
                        frags_b = smiles_to_frags.get(s_b, {})
                        if core not in frags_a or core not in frags_b:
                            continue
                        chains_a = frags_a[core]
                        chains_b = frags_b[core]
                        if chains_a == chains_b:
                            continue

                        edit_smiles = self._compute_edit_smiles(chains_a, chains_b)
                        if not edit_smiles:
                            continue

                        info_a = pic50_lookup.get(s_a)
                        info_b = pic50_lookup.get(s_b)
                        if info_a is None or info_b is None:
                            continue

                        delta = info_b["pIC50"] - info_a["pIC50"]
                        prop_name = f"{info_a['standard_type']}_{info_a['target_chembl_id']}"

                        row = {
                            "mol_a": s_a,
                            "mol_b": s_b,
                            "mol_a_id": info_a["molecule_chembl_id"],
                            "mol_b_id": info_b["molecule_chembl_id"],
                            "edit_smiles": edit_smiles,
                            "num_cuts": 1,
                            "property_name": prop_name,
                            "value_a": info_a["pIC50"],
                            "value_b": info_b["pIC50"],
                            "delta": delta,
                            "target_name": info_a["target_name"],
                            "target_chembl_id": info_a["target_chembl_id"],
                            "doc_id_a": info_a.get("doc_id"),
                            "doc_id_b": info_b.get("doc_id"),
                            "assay_id_a": assay_id,
                            "assay_id_b": assay_id,
                            "is_within_assay": True,
                            "assay_pair_id": None,
                        }
                        all_pair_rows.append(row)
                        pair_count += 1

        logger.info(f"  Within-assay MMP pairs: {pair_count:,}")

        # Generate cross-assay pairs for overlapping compounds
        logger.info("  Generating cross-assay MMP pairs...")
        cross_count = 0
        for _, ap_row in assay_pairs_df.iterrows():
            a1 = ap_row["assay_id_1"]
            a2 = ap_row["assay_id_2"]
            pair_key = (min(a1, a2), max(a1, a2))
            pair_id = assay_pair_id_lookup.get(pair_key)

            # Get molecules in each assay
            if a1 not in assay_groups.groups or a2 not in assay_groups.groups:
                continue
            grp1 = assay_groups.get_group(a1)
            grp2 = assay_groups.get_group(a2)

            # Build pIC50 lookups
            pic50_1 = {
                row["smiles"]: {
                    "pIC50": row["pchembl_value"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row.get("target_name", ""),
                    "standard_type": row.get("standard_type", ""),
                    "doc_id": row.get("doc_id"),
                }
                for _, row in grp1.iterrows()
            }
            pic50_2 = {
                row["smiles"]: {
                    "pIC50": row["pchembl_value"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row.get("target_name", ""),
                    "standard_type": row.get("standard_type", ""),
                    "doc_id": row.get("doc_id"),
                }
                for _, row in grp2.iterrows()
            }

            smiles_1 = set(pic50_1.keys())
            smiles_2 = set(pic50_2.keys())

            # For shared compounds: create cross-assay "self-pairs"
            # (same molecule measured in two assays - directly measures noise)
            shared = smiles_1 & smiles_2
            for smi in shared:
                info1 = pic50_1[smi]
                info2 = pic50_2[smi]
                prop_name = f"{info1['standard_type']}_{info1['target_chembl_id']}"
                delta = info2["pIC50"] - info1["pIC50"]

                row = {
                    "mol_a": smi,
                    "mol_b": smi,
                    "mol_a_id": info1["molecule_chembl_id"],
                    "mol_b_id": info2["molecule_chembl_id"],
                    "edit_smiles": "",  # identity edit
                    "num_cuts": 0,
                    "property_name": prop_name,
                    "value_a": info1["pIC50"],
                    "value_b": info2["pIC50"],
                    "delta": delta,
                    "target_name": info1["target_name"],
                    "target_chembl_id": info1["target_chembl_id"],
                    "doc_id_a": info1.get("doc_id"),
                    "doc_id_b": info2.get("doc_id"),
                    "assay_id_a": a1,
                    "assay_id_b": a2,
                    "is_within_assay": False,
                    "assay_pair_id": pair_id,
                }
                all_pair_rows.append(row)
                cross_count += 1

            # Cross-assay MMP pairs: molecule from assay 1, molecule from assay 2
            seen_cross = set()
            for core, core_smiles_list in core_to_smiles.items():
                in_a1 = [s for s in core_smiles_list if s in smiles_1]
                in_a2 = [s for s in core_smiles_list if s in smiles_2]
                if not in_a1 or not in_a2:
                    continue

                for s_a in in_a1:
                    for s_b in in_a2:
                        if s_a == s_b:
                            continue
                        cross_key = (min(s_a, s_b), max(s_a, s_b))
                        if cross_key in seen_cross:
                            continue
                        seen_cross.add(cross_key)

                        # MW filter
                        try:
                            mol_a = Chem.MolFromSmiles(s_a)
                            mol_b = Chem.MolFromSmiles(s_b)
                            if mol_a is None or mol_b is None:
                                continue
                            if abs(Descriptors.MolWt(mol_a) - Descriptors.MolWt(mol_b)) > self.config.max_mw_delta:
                                continue
                        except Exception:
                            continue

                        frags_a = smiles_to_frags.get(s_a, {})
                        frags_b = smiles_to_frags.get(s_b, {})
                        if core not in frags_a or core not in frags_b:
                            continue
                        chains_a = frags_a[core]
                        chains_b = frags_b[core]
                        if chains_a == chains_b:
                            continue

                        edit_smiles = self._compute_edit_smiles(chains_a, chains_b)
                        if not edit_smiles:
                            continue

                        info_a = pic50_1.get(s_a)
                        info_b = pic50_2.get(s_b)
                        if info_a is None or info_b is None:
                            continue

                        delta = info_b["pIC50"] - info_a["pIC50"]
                        prop_name = f"{info_a['standard_type']}_{info_a['target_chembl_id']}"

                        row = {
                            "mol_a": s_a,
                            "mol_b": s_b,
                            "mol_a_id": info_a["molecule_chembl_id"],
                            "mol_b_id": info_b["molecule_chembl_id"],
                            "edit_smiles": edit_smiles,
                            "num_cuts": 1,
                            "property_name": prop_name,
                            "value_a": info_a["pIC50"],
                            "value_b": info_b["pIC50"],
                            "delta": delta,
                            "target_name": info_a["target_name"],
                            "target_chembl_id": info_a["target_chembl_id"],
                            "doc_id_a": info_a.get("doc_id"),
                            "doc_id_b": info_b.get("doc_id"),
                            "assay_id_a": a1,
                            "assay_id_b": a2,
                            "is_within_assay": False,
                            "assay_pair_id": pair_id,
                        }
                        all_pair_rows.append(row)
                        cross_count += 1

        logger.info(f"  Cross-assay pairs: {cross_count:,}")

        if all_pair_rows:
            return pd.DataFrame(all_pair_rows)
        return pd.DataFrame()

    def _generate_tanimoto_pairs(
        self,
        activities_df: pd.DataFrame,
        assay_pairs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate pairs based on Tanimoto similarity (Morgan FP > threshold).

        Alternative to MMP that captures structurally similar but not
        necessarily MMP-related molecules.
        """
        logger.info(f"Generating Tanimoto similarity pairs (threshold={self.config.tanimoto_threshold})...")

        all_assay_ids = set(assay_pairs_df["assay_id_1"]) | set(assay_pairs_df["assay_id_2"])
        relevant = activities_df[activities_df["assay_id"].isin(all_assay_ids)].copy()

        # Compute Morgan fingerprints
        unique_smiles = relevant["smiles"].unique()
        fps = {}
        for smi in unique_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        logger.info(f"  Computed fingerprints for {len(fps):,} molecules")

        # Generate pairs per assay
        all_rows = []
        assay_groups = relevant.groupby("assay_id")

        for assay_id in all_assay_ids:
            if assay_id not in assay_groups.groups:
                continue
            grp = assay_groups.get_group(assay_id)
            assay_smiles = [s for s in grp["smiles"].unique() if s in fps]
            if len(assay_smiles) < 2:
                continue

            pic50_lookup = {}
            for _, row in grp.iterrows():
                pic50_lookup[row["smiles"]] = {
                    "pIC50": row["pchembl_value"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row.get("target_name", ""),
                    "standard_type": row.get("standard_type", ""),
                    "doc_id": row.get("doc_id"),
                }

            for i in range(len(assay_smiles)):
                for j in range(i + 1, len(assay_smiles)):
                    s_a, s_b = assay_smiles[i], assay_smiles[j]
                    sim = DataStructs.TanimotoSimilarity(fps[s_a], fps[s_b])
                    if sim < self.config.tanimoto_threshold:
                        continue

                    info_a = pic50_lookup.get(s_a)
                    info_b = pic50_lookup.get(s_b)
                    if info_a is None or info_b is None:
                        continue

                    delta = info_b["pIC50"] - info_a["pIC50"]
                    prop_name = f"{info_a['standard_type']}_{info_a['target_chembl_id']}"

                    all_rows.append(
                        {
                            "mol_a": s_a,
                            "mol_b": s_b,
                            "mol_a_id": info_a["molecule_chembl_id"],
                            "mol_b_id": info_b["molecule_chembl_id"],
                            "edit_smiles": "",
                            "num_cuts": 0,
                            "property_name": prop_name,
                            "value_a": info_a["pIC50"],
                            "value_b": info_b["pIC50"],
                            "delta": delta,
                            "target_name": info_a["target_name"],
                            "target_chembl_id": info_a["target_chembl_id"],
                            "doc_id_a": info_a.get("doc_id"),
                            "doc_id_b": info_b.get("doc_id"),
                            "assay_id_a": assay_id,
                            "assay_id_b": assay_id,
                            "is_within_assay": True,
                            "assay_pair_id": None,
                            "tanimoto_similarity": sim,
                            "pair_method": "tanimoto",
                        }
                    )

        logger.info(f"  Tanimoto pairs: {len(all_rows):,}")
        if all_rows:
            return pd.DataFrame(all_rows)
        return pd.DataFrame()

    def _generate_scaffold_pairs(
        self,
        activities_df: pd.DataFrame,
        assay_pairs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate pairs of molecules sharing the same Bemis-Murcko scaffold.

        This captures molecules with the same core structure but different
        substituents - a natural grouping for medicinal chemistry SAR.
        """
        logger.info("Generating scaffold-matched pairs...")

        all_assay_ids = set(assay_pairs_df["assay_id_1"]) | set(assay_pairs_df["assay_id_2"])
        relevant = activities_df[activities_df["assay_id"].isin(all_assay_ids)].copy()

        # Compute scaffolds
        unique_smiles = relevant["smiles"].unique()
        scaffolds: Dict[str, str] = {}
        for smi in unique_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds[smi] = Chem.MolToSmiles(scaffold)
                except Exception:
                    pass

        logger.info(f"  Computed scaffolds for {len(scaffolds):,} molecules")
        logger.info(f"  Unique scaffolds: {len(set(scaffolds.values())):,}")

        # Group by scaffold
        scaffold_to_smiles: Dict[str, List[str]] = defaultdict(list)
        for smi, scaffold in scaffolds.items():
            scaffold_to_smiles[scaffold].append(smi)

        # Generate pairs per assay
        all_rows = []
        assay_groups = relevant.groupby("assay_id")

        for assay_id in all_assay_ids:
            if assay_id not in assay_groups.groups:
                continue
            grp = assay_groups.get_group(assay_id)
            assay_smiles_set = set(grp["smiles"].unique())

            pic50_lookup = {}
            for _, row in grp.iterrows():
                pic50_lookup[row["smiles"]] = {
                    "pIC50": row["pchembl_value"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row.get("target_name", ""),
                    "standard_type": row.get("standard_type", ""),
                    "doc_id": row.get("doc_id"),
                }

            seen = set()
            for scaffold, scaffold_mols in scaffold_to_smiles.items():
                in_assay = [s for s in scaffold_mols if s in assay_smiles_set]
                if len(in_assay) < 2:
                    continue

                for i in range(len(in_assay)):
                    for j in range(i + 1, len(in_assay)):
                        s_a, s_b = in_assay[i], in_assay[j]
                        pkey = (min(s_a, s_b), max(s_a, s_b))
                        if pkey in seen:
                            continue
                        seen.add(pkey)

                        info_a = pic50_lookup.get(s_a)
                        info_b = pic50_lookup.get(s_b)
                        if info_a is None or info_b is None:
                            continue

                        delta = info_b["pIC50"] - info_a["pIC50"]
                        prop_name = f"{info_a['standard_type']}_{info_a['target_chembl_id']}"

                        all_rows.append(
                            {
                                "mol_a": s_a,
                                "mol_b": s_b,
                                "mol_a_id": info_a["molecule_chembl_id"],
                                "mol_b_id": info_b["molecule_chembl_id"],
                                "edit_smiles": "",
                                "num_cuts": 0,
                                "property_name": prop_name,
                                "value_a": info_a["pIC50"],
                                "value_b": info_b["pIC50"],
                                "delta": delta,
                                "target_name": info_a["target_name"],
                                "target_chembl_id": info_a["target_chembl_id"],
                                "doc_id_a": info_a.get("doc_id"),
                                "doc_id_b": info_b.get("doc_id"),
                                "assay_id_a": assay_id,
                                "assay_id_b": assay_id,
                                "is_within_assay": True,
                                "assay_pair_id": None,
                                "scaffold": scaffold,
                                "pair_method": "scaffold",
                            }
                        )

        logger.info(f"  Scaffold pairs: {len(all_rows):,}")
        if all_rows:
            return pd.DataFrame(all_rows)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_edit_smiles(chains_a: str, chains_b: str) -> str:
        """
        Compute canonical edit SMILES from MMP fragment chains.

        Converts attachment chains like "[*:1]C" and "[*:1]CC" into
        reaction SMILES "C>>CC" by stripping attachment points and
        canonicalizing with RDKit.
        """
        parts_a = set(chains_a.split("."))
        parts_b = set(chains_b.split("."))

        edit_from_parts = parts_a - parts_b
        edit_to_parts = parts_b - parts_a

        if len(edit_from_parts) != 1 or len(edit_to_parts) != 1:
            return ""

        edit_from_raw = (
            list(edit_from_parts)[0]
            .replace("[*:1]", "[H]")
            .replace("[*:2]", "[H]")
            .replace("[*:3]", "[H]")
        )
        edit_to_raw = (
            list(edit_to_parts)[0]
            .replace("[*:1]", "[H]")
            .replace("[*:2]", "[H]")
            .replace("[*:3]", "[H]")
        )

        mol_from = Chem.MolFromSmiles(edit_from_raw)
        mol_to = Chem.MolFromSmiles(edit_to_raw)
        if mol_from is None or mol_to is None:
            return ""

        edit_from = Chem.MolToSmiles(mol_from)
        edit_to = Chem.MolToSmiles(mol_to)

        return f"{edit_from}>>{edit_to}"

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    def extract_pairs(self, **kwargs) -> pd.DataFrame:
        """
        Full extraction pipeline:
        1. Load/query activities
        2. Find Goldilocks assays and overlapping pairs
        3. Generate molecular pairs (MMP, Tanimoto, and/or scaffold)
        4. Save molecule-level pIC50 table
        """
        # Load activities
        logger.info("Loading activities...")
        activities_df = pd.read_csv(self.config.activities_file)
        logger.info(f"  Activities: {len(activities_df):,}")
        logger.info(f"  Unique molecules: {activities_df['molecule_chembl_id'].nunique():,}")
        logger.info(f"  Unique assays: {activities_df['assay_id'].nunique():,}")
        logger.info(f"  Unique targets: {activities_df['target_chembl_id'].nunique():,}")

        # Find Goldilocks assays
        logger.info("Finding Goldilocks assays...")
        goldilocks = self._find_goldilocks_assays(activities_df)

        # Find overlapping assay pairs
        logger.info("Finding overlapping assay pairs...")
        assay_pairs_df = self._find_overlapping_pairs(
            activities_df, set(goldilocks.index)
        )

        if assay_pairs_df.empty:
            logger.warning("No overlapping assay pairs found!")
            return pd.DataFrame()

        # Save assay pairs
        assay_pairs_df.to_csv(self.config.assay_pairs_file, index=False)
        logger.info(f"  Saved assay pairs to {self.config.assay_pairs_file}")

        # Build and save molecule-level pIC50 table
        logger.info("Building molecule-level pIC50 table...")
        mol_props = self._build_molecule_properties(activities_df)
        mol_props.to_csv(self.config.molecule_properties_file, index=False)
        logger.info(f"  Saved {len(mol_props):,} molecule properties to {self.config.molecule_properties_file}")

        # Generate pairs using requested methods
        all_pairs = []

        if "mmp" in self.config.pair_methods:
            mmp_pairs = self._generate_mmp_pairs(activities_df, assay_pairs_df)
            if not mmp_pairs.empty:
                mmp_pairs["pair_method"] = "mmp"
                all_pairs.append(mmp_pairs)

        if "tanimoto" in self.config.pair_methods:
            tan_pairs = self._generate_tanimoto_pairs(activities_df, assay_pairs_df)
            if not tan_pairs.empty:
                all_pairs.append(tan_pairs)

        if "scaffold" in self.config.pair_methods:
            scaffold_pairs = self._generate_scaffold_pairs(activities_df, assay_pairs_df)
            if not scaffold_pairs.empty:
                all_pairs.append(scaffold_pairs)

        if not all_pairs:
            logger.warning("No pairs generated!")
            return pd.DataFrame()

        pairs_df = pd.concat(all_pairs, ignore_index=True)

        # Fill missing columns
        if "pair_method" not in pairs_df.columns:
            pairs_df["pair_method"] = "mmp"
        if "tanimoto_similarity" not in pairs_df.columns:
            pairs_df["tanimoto_similarity"] = np.nan
        if "scaffold" not in pairs_df.columns:
            pairs_df["scaffold"] = ""

        # Add curation level to every row
        pairs_df["curation_level"] = self.config.curation_level

        # Save
        pairs_df.to_csv(self.config.pairs_output, index=False)
        logger.info(f"Saved {len(pairs_df):,} pairs to {self.config.pairs_output}")

        # Summary statistics
        self._log_summary(pairs_df)

        return pairs_df

    def _log_summary(self, pairs_df: pd.DataFrame) -> None:
        """Log summary statistics."""
        logger.info("=" * 70)
        logger.info(" OVERLAPPING ASSAY EXTRACTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total pairs: {len(pairs_df):,}")
        logger.info(f"  Unique molecules: {pd.concat([pairs_df['mol_a'], pairs_df['mol_b']]).nunique():,}")
        logger.info(f"  Unique targets: {pairs_df['target_chembl_id'].nunique():,}")

        if "is_within_assay" in pairs_df.columns:
            within = pairs_df["is_within_assay"].sum()
            cross = (~pairs_df["is_within_assay"]).sum()
            logger.info(f"  Within-assay pairs: {within:,}")
            logger.info(f"  Cross-assay pairs: {cross:,}")

        if "pair_method" in pairs_df.columns:
            for method, count in pairs_df["pair_method"].value_counts().items():
                logger.info(f"  {method} pairs: {count:,}")

        # Delta distribution
        logger.info(f"  Delta (pIC50) stats:")
        logger.info(f"    mean: {pairs_df['delta'].mean():.3f}")
        logger.info(f"    std:  {pairs_df['delta'].std():.3f}")
        logger.info(f"    |delta| > 0.3: {(pairs_df['delta'].abs() > 0.3).mean():.1%}")
        logger.info(f"    |delta| > 1.0: {(pairs_df['delta'].abs() > 1.0).mean():.1%}")
        logger.info("=" * 70)
