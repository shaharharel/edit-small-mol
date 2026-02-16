#!/usr/bin/env python3
"""
MMP Pair Extraction using mmpdb.

This script uses the mmpdb package (https://github.com/rdkit/mmpdb) to efficiently
generate matched molecular pairs from ChEMBL data. mmpdb is optimized for large-scale
MMP generation and handles the combinatorial complexity better than custom implementations.

Input files (from ChEMBL extraction):
    - molecules_top_targets_*.csv: chembl_id, smiles
    - bioactivity_top_targets_*.csv: chembl_id, property_name, value, target_name,
                                     target_chembl_id, doc_id, assay_id

Output schema (matching existing format):
    mol_a, mol_b, edit_smiles, num_cuts, property_name, value_a, value_b, delta,
    target_name, target_chembl_id, doc_id_a, doc_id_b, assay_id_a, assay_id_b

Usage:
    python extract_pairs_mmpdb.py --molecules molecules.csv --bioactivity bioactivity.csv --output pairs.csv

    # Or with default paths:
    python extract_pairs_mmpdb.py
"""

import argparse
import logging
import os
import re
import subprocess
import tempfile
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOLECULES = PROJECT_ROOT / "data/chembl/molecules_top_targets_5_all_molecules.csv"
DEFAULT_BIOACTIVITY = PROJECT_ROOT / "data/chembl/bioactivity_top_targets_5_all_molecules.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/pairs/mmpdb"


class MMPDBExtractor:
    """
    Extract matched molecular pairs using mmpdb.

    Pipeline:
    1. Convert molecules CSV to mmpdb SMILES format
    2. Fragment molecules using mmpdb fragment
    3. Index fragments to create MMP database (can output CSV of pairs)
    4. Join pairs with bioactivity data to create full output
    """

    def __init__(
        self,
        mmpdb_path: str = "mmpdb",
        num_cuts: int = 3,
        max_heavies: int = 100,
        max_rotatable_bonds: int = 10,
        max_variable_heavies: int = 10,
        num_jobs: int = 4,
        symmetric: bool = False,
        cut_smarts: Optional[str] = None
    ):
        """
        Initialize the mmpdb extractor.

        Args:
            mmpdb_path: Path to mmpdb executable
            num_cuts: Maximum number of cuts for fragmentation (1, 2, or 3)
            max_heavies: Maximum heavy atoms in a molecule
            max_rotatable_bonds: Maximum rotatable bonds
            max_variable_heavies: Maximum heavy atoms in variable fragment
            num_jobs: Number of parallel jobs for fragmentation
            symmetric: Whether to output both A>>B and B>>A transformations
            cut_smarts: SMARTS pattern for cutting. Options:
                - None/'default': Drug-like bonds only (excludes CH2-CH2, amides)
                - 'cut_AlkylChains': Also cuts CH2-CH2 bonds
                - 'cut_Amides': Also cuts amide bonds
                - 'cut_all': Cuts ALL C-[!H] bonds (like rdMMPA)
                - 'exocyclic': Only exocyclic bonds
        """
        self.mmpdb_path = mmpdb_path
        self.num_cuts = num_cuts
        self.max_heavies = max_heavies
        self.max_rotatable_bonds = max_rotatable_bonds
        self.max_variable_heavies = max_variable_heavies
        self.num_jobs = num_jobs
        self.symmetric = symmetric
        self.cut_smarts = cut_smarts
        self.verbose = False  # Will be set by extract_pairs

        # Verify mmpdb is available
        self._verify_mmpdb()

    def _verify_mmpdb(self):
        """Verify mmpdb is installed and accessible."""
        try:
            result = subprocess.run(
                [self.mmpdb_path, "--version"],
                capture_output=True,
                text=True
            )
            logger.info(f"Using mmpdb: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"mmpdb not found at '{self.mmpdb_path}'. "
                "Install with: pip install mmpdb"
            )

    def _run_mmpdb(self, args: List[str], desc: str = "Running mmpdb") -> subprocess.CompletedProcess:
        """Run mmpdb command with logging."""
        cmd = [self.mmpdb_path] + args
        logger.info(f"{desc}: {' '.join(cmd)}")

        # Suppress RDKit warnings in subprocess
        env = os.environ.copy()
        env['RDKIT_LOG_LEVEL'] = 'ERROR'

        if self.verbose:
            # Stream output in real-time for verbose mode
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            output_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)
            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"mmpdb command failed with return code {process.returncode}")

            # Create a mock CompletedProcess for compatibility
            result = subprocess.CompletedProcess(cmd, process.returncode, ''.join(output_lines), '')
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode != 0:
                logger.error(f"mmpdb error: {result.stderr}")
                raise RuntimeError(f"mmpdb command failed: {result.stderr}")

            if result.stdout:
                logger.debug(f"mmpdb output: {result.stdout}")

        return result

    def prepare_smiles_file(
        self,
        molecules_df: pd.DataFrame,
        output_path: Path
    ) -> Path:
        """
        Convert molecules DataFrame to mmpdb SMILES format.

        mmpdb expects: SMILES<whitespace>ID format
        """
        logger.info(f"Preparing SMILES file with {len(molecules_df)} molecules")

        # Ensure we have the required columns
        if 'smiles' not in molecules_df.columns:
            raise ValueError("molecules_df must have 'smiles' column")
        if 'chembl_id' not in molecules_df.columns:
            raise ValueError("molecules_df must have 'chembl_id' column")

        # Write SMILES file (tab-separated: SMILES ID)
        with open(output_path, 'w') as f:
            for _, row in molecules_df.iterrows():
                smiles = row['smiles']
                chembl_id = row['chembl_id']
                # Skip invalid SMILES
                if pd.isna(smiles) or not smiles.strip():
                    continue
                f.write(f"{smiles}\t{chembl_id}\n")

        logger.info(f"Wrote SMILES file to {output_path}")
        return output_path

    def fragment_molecules(
        self,
        smiles_file: Path,
        output_fragdb: Path
    ) -> Path:
        """
        Fragment molecules using mmpdb fragment.

        This is the most time-consuming step but mmpdb is highly optimized.
        """
        cut_info = f", cut_smarts={self.cut_smarts}" if self.cut_smarts else ""
        logger.info(f"Fragmenting molecules (num_cuts={self.num_cuts}, jobs={self.num_jobs}{cut_info})")

        args = [
            "fragment",
            str(smiles_file),
            "--num-cuts", str(self.num_cuts),
            "--max-heavies", str(self.max_heavies),
            "--max-rotatable-bonds", str(self.max_rotatable_bonds),
            "--num-jobs", str(self.num_jobs),
            "--delimiter", "tab",
            "-o", str(output_fragdb)
        ]

        # Add custom cut SMARTS if specified
        if self.cut_smarts:
            args.extend(["--cut-smarts", self.cut_smarts])

        self._run_mmpdb(args, "Fragmenting molecules")
        logger.info(f"Fragment database created: {output_fragdb}")
        return output_fragdb

    def index_fragments_to_csv(
        self,
        fragdb_file: Path,
        output_csv: Path
    ) -> Path:
        """
        Index fragments and output pairs as CSV.

        The CSV output has columns: SMILES1, SMILES2, id1, id2, V1>>V2, C
        """
        logger.info("Indexing fragments to CSV")

        args = [
            "index",
            str(fragdb_file),
            "--max-variable-heavies", str(self.max_variable_heavies),
            "--out", "csv",
            "-o", str(output_csv)
        ]

        if self.symmetric:
            args.append("--symmetric")

        self._run_mmpdb(args, "Indexing to CSV")
        logger.info(f"Pairs CSV created: {output_csv}")
        return output_csv

    def index_fragments_to_mmpdb(
        self,
        fragdb_file: Path,
        output_mmpdb: Path
    ) -> Path:
        """
        Index fragments and create mmpdb database.

        This creates a SQLite database that can be queried for more details.
        """
        logger.info("Indexing fragments to mmpdb database")

        args = [
            "index",
            str(fragdb_file),
            "--max-variable-heavies", str(self.max_variable_heavies),
            "-o", str(output_mmpdb)
        ]

        if self.symmetric:
            args.append("--symmetric")

        self._run_mmpdb(args, "Indexing to mmpdb")
        logger.info(f"mmpdb database created: {output_mmpdb}")
        return output_mmpdb

    def parse_mmpdb_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Parse mmpdb CSV output.

        mmpdb CSV format (tab-separated):
        SMILES1  SMILES2  id1  id2  V1>>V2  C

        Returns DataFrame with columns:
        - mol_a: SMILES of molecule A
        - mol_b: SMILES of molecule B
        - chembl_id_a: ChEMBL ID of molecule A
        - chembl_id_b: ChEMBL ID of molecule B
        - edit_smiles: Transformation V1>>V2
        - constant: Constant (core) SMILES
        """
        logger.info(f"Parsing mmpdb CSV: {csv_path}")

        # Read the tab-separated file
        df = pd.read_csv(
            csv_path,
            sep='\t',
            names=['mol_a', 'mol_b', 'chembl_id_a', 'chembl_id_b', 'edit_smiles', 'constant'],
            header=None
        )

        logger.info(f"Loaded {len(df)} pairs from mmpdb CSV")
        return df

    def extract_num_cuts(self, constant_smiles: str) -> int:
        """
        Extract number of cuts from constant SMILES.

        The constant part has [*] or [*:N] attachment points.
        Number of cuts = number of attachment points.
        """
        if pd.isna(constant_smiles):
            return 1

        # Count attachment points
        # Match [*], [*:1], [*:2], etc.
        attachments = re.findall(r'\[\*(?::\d+)?\]', constant_smiles)
        return max(1, len(attachments))

    def compute_atom_mapping_fast(
        self,
        mol_a_smiles: str,
        mol_b_smiles: str,
        constant_smiles: str,
        edit_smiles: str
    ) -> Dict[str, any]:
        """
        Compute atom-level mapping using mmpdb output directly.

        This is fast because mmpdb already gives us the core (constant) and
        the transformation - no expensive MCS computation needed.

        Args:
            mol_a_smiles: SMILES of molecule A
            mol_b_smiles: SMILES of molecule B
            constant_smiles: Core/constant SMILES from mmpdb (e.g., "[*:1]c1ccccc1")
            edit_smiles: Transformation (e.g., "[*:1]O>>[*:1]OC")

        Returns:
            Dict with:
            - removed_atoms_A: Atom indices in A of the leaving fragment
            - added_atoms_B: Atom indices in B of the incoming fragment
            - attach_atoms_A: Attachment point atom indices in A
            - mapped_pairs: List of (idx_A, idx_B) for atoms in common core
        """
        if not RDKIT_AVAILABLE:
            return self._empty_mapping()

        try:
            mol_a = Chem.MolFromSmiles(mol_a_smiles)
            mol_b = Chem.MolFromSmiles(mol_b_smiles)

            if mol_a is None or mol_b is None:
                return self._empty_mapping()

            if pd.isna(constant_smiles) or not constant_smiles:
                return self._empty_mapping()

            # Clean up the core SMILES by replacing attachment points with H
            # This gives us the common scaffold that we can match
            core_clean = re.sub(r'\[\*(?::\d+)?\]', '[H]', constant_smiles)

            # Handle disconnected cores (e.g., "[*:1]C.[*:2]N")
            core_mol = Chem.MolFromSmiles(core_clean)

            if core_mol is None:
                # Try without the hydrogens (some cores don't parse well)
                core_clean = re.sub(r'\[\*(?::\d+)?\]', '*', constant_smiles)
                core_mol = Chem.MolFromSmiles(core_clean)
                if core_mol is None:
                    return self._empty_mapping()

            # Fast substructure matching to find core atoms in both molecules
            core_match_a = mol_a.GetSubstructMatch(core_mol)
            core_match_b = mol_b.GetSubstructMatch(core_mol)

            if not core_match_a or not core_match_b:
                return self._empty_mapping()

            # Atoms not in core = removed/added fragments
            all_atoms_a = set(range(mol_a.GetNumAtoms()))
            all_atoms_b = set(range(mol_b.GetNumAtoms()))

            core_atoms_a = set(core_match_a)
            core_atoms_b = set(core_match_b)

            removed_atoms_A = sorted(all_atoms_a - core_atoms_a)
            added_atoms_B = sorted(all_atoms_b - core_atoms_b)

            # Find attachment points (core atoms bonded to removed atoms)
            attach_atoms_A = []
            for core_atom_idx in core_match_a:
                atom = mol_a.GetAtomWithIdx(core_atom_idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in removed_atoms_A:
                        if core_atom_idx not in attach_atoms_A:
                            attach_atoms_A.append(core_atom_idx)
                        break

            # Create mapped pairs from core correspondences
            mapped_pairs = list(zip(core_match_a, core_match_b))

            return {
                'removed_atoms_A': removed_atoms_A,
                'added_atoms_B': added_atoms_B,
                'attach_atoms_A': attach_atoms_A,
                'mapped_pairs': mapped_pairs
            }

        except Exception as e:
            logger.debug(f"Atom mapping failed for {mol_a_smiles[:30]}: {e}")
            return self._empty_mapping()

    def _empty_mapping(self) -> Dict[str, any]:
        """Return empty mapping structure."""
        return {
            'removed_atoms_A': [],
            'added_atoms_B': [],
            'attach_atoms_A': [],
            'mapped_pairs': []
        }

    def _serialize_list(self, lst: List[int]) -> str:
        """Convert list to string: [1,2,3] -> '1;2;3'"""
        if not lst:
            return ''
        return ';'.join(map(str, lst))

    def _serialize_pairs(self, pairs: List[Tuple[int, int]]) -> str:
        """Convert pairs to string: [(1,2),(3,4)] -> '1,2;3,4'"""
        if not pairs:
            return ''
        return ';'.join(f"{a},{b}" for a, b in pairs)

    def add_atom_mapping_to_df(
        self,
        df: pd.DataFrame,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Add atom mapping columns to a pairs DataFrame.

        Args:
            df: DataFrame with mol_a, mol_b, constant, edit_smiles columns
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with additional columns:
            - removed_atoms_A, added_atoms_B, attach_atoms_A, mapped_pairs
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping atom mapping")
            df['removed_atoms_A'] = ''
            df['added_atoms_B'] = ''
            df['attach_atoms_A'] = ''
            df['mapped_pairs'] = ''
            return df

        logger.info(f"Computing atom mapping for {len(df)} pairs...")

        # Get unique (mol_a, mol_b, constant) combinations to avoid redundant computation
        unique_pairs = df[['mol_a', 'mol_b', 'constant', 'edit_smiles']].drop_duplicates()
        logger.info(f"Computing mappings for {len(unique_pairs)} unique pairs")

        # Compute mappings
        mappings = {}
        iterator = unique_pairs.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(unique_pairs), desc="Atom mapping")

        for _, row in iterator:
            key = (row['mol_a'], row['mol_b'])
            if key not in mappings:
                mapping = self.compute_atom_mapping_fast(
                    row['mol_a'],
                    row['mol_b'],
                    row['constant'],
                    row['edit_smiles']
                )
                mappings[key] = {
                    'removed_atoms_A': self._serialize_list(mapping['removed_atoms_A']),
                    'added_atoms_B': self._serialize_list(mapping['added_atoms_B']),
                    'attach_atoms_A': self._serialize_list(mapping['attach_atoms_A']),
                    'mapped_pairs': self._serialize_pairs(mapping['mapped_pairs'])
                }

        # Apply mappings to DataFrame
        df['removed_atoms_A'] = df.apply(
            lambda row: mappings.get((row['mol_a'], row['mol_b']), {}).get('removed_atoms_A', ''),
            axis=1
        )
        df['added_atoms_B'] = df.apply(
            lambda row: mappings.get((row['mol_a'], row['mol_b']), {}).get('added_atoms_B', ''),
            axis=1
        )
        df['attach_atoms_A'] = df.apply(
            lambda row: mappings.get((row['mol_a'], row['mol_b']), {}).get('attach_atoms_A', ''),
            axis=1
        )
        df['mapped_pairs'] = df.apply(
            lambda row: mappings.get((row['mol_a'], row['mol_b']), {}).get('mapped_pairs', ''),
            axis=1
        )

        success_count = sum(1 for m in mappings.values() if m['removed_atoms_A'] or m['added_atoms_B'])
        logger.info(f"Computed atom mappings: {success_count}/{len(mappings)} successful")

        return df

    def join_with_bioactivity(
        self,
        pairs_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame,
        property_filter: Optional[set] = None
    ) -> pd.DataFrame:
        """
        Join pairs with bioactivity data to create full output.

        For each pair (A, B), we need to find properties where BOTH molecules
        were tested on the SAME target. Then we compute delta = value_b - value_a.

        Args:
            pairs_df: DataFrame with mol_a, mol_b, chembl_id_a, chembl_id_b, edit_smiles
            bioactivity_df: DataFrame with chembl_id, property_name, value/pchembl_value,
                           target_name, target_chembl_id, and optionally doc_id, assay_id
            property_filter: Optional set of property names to include

        Returns:
            DataFrame with full schema
        """
        logger.info("Joining pairs with bioactivity data")

        # Filter properties if specified
        if property_filter:
            bioactivity_df = bioactivity_df[
                bioactivity_df['property_name'].isin(property_filter)
            ].copy()
            logger.info(f"Filtered to {len(bioactivity_df)} measurements for {property_filter}")

        # Detect value column name (pchembl_value or value)
        if 'pchembl_value' in bioactivity_df.columns:
            value_col = 'pchembl_value'
        elif 'value' in bioactivity_df.columns:
            value_col = 'value'
        else:
            raise ValueError("Bioactivity DataFrame must have 'value' or 'pchembl_value' column")

        # Check for optional columns
        has_doc_id = 'doc_id' in bioactivity_df.columns
        has_assay_id = 'assay_id' in bioactivity_df.columns

        # Create lookup for molecule -> bioactivity
        logger.info("Creating bioactivity lookup...")

        # Build rename mapping dynamically
        rename_a = {'chembl_id': 'chembl_id_a', value_col: 'value_a'}
        rename_b = {'chembl_id': 'chembl_id_b', value_col: 'value_b'}

        if has_doc_id:
            rename_a['doc_id'] = 'doc_id_a'
            rename_b['doc_id'] = 'doc_id_b'
        if has_assay_id:
            rename_a['assay_id'] = 'assay_id_a'
            rename_b['assay_id'] = 'assay_id_b'

        bio_a = bioactivity_df.copy()
        bio_a = bio_a.rename(columns=rename_a)

        bio_b = bioactivity_df.copy()
        bio_b = bio_b.rename(columns=rename_b)

        # Build column lists for merge
        merge_cols_a = ['chembl_id_a', 'property_name', 'value_a', 'target_name', 'target_chembl_id']
        merge_cols_b = ['chembl_id_b', 'property_name', 'value_b', 'target_chembl_id']

        if has_doc_id:
            merge_cols_a.append('doc_id_a')
            merge_cols_b.append('doc_id_b')
        if has_assay_id:
            merge_cols_a.append('assay_id_a')
            merge_cols_b.append('assay_id_b')

        # Join pairs with bioactivity for molecule A
        logger.info("Joining with molecule A bioactivity...")
        merged = pairs_df.merge(
            bio_a[merge_cols_a],
            on='chembl_id_a',
            how='inner'
        )
        logger.info(f"After join with A: {len(merged)} rows")

        # Join with bioactivity for molecule B (must match property AND target)
        logger.info("Joining with molecule B bioactivity...")
        merged = merged.merge(
            bio_b[merge_cols_b],
            on=['chembl_id_b', 'property_name', 'target_chembl_id'],
            how='inner'
        )
        logger.info(f"After join with B: {len(merged)} rows")

        # Compute delta
        merged['delta'] = merged['value_b'] - merged['value_a']

        # Extract num_cuts from constant SMILES
        if 'constant' in merged.columns:
            merged['num_cuts'] = merged['constant'].apply(self.extract_num_cuts)
        else:
            merged['num_cuts'] = 1

        # Select and order columns for output
        # Keep 'constant' for atom mapping computation later
        output_columns = [
            'mol_a', 'mol_b', 'edit_smiles', 'num_cuts', 'constant',
            'property_name', 'value_a', 'value_b', 'delta',
            'target_name', 'target_chembl_id',
            'doc_id_a', 'doc_id_b', 'assay_id_a', 'assay_id_b'
        ]

        # Keep only columns that exist
        output_columns = [c for c in output_columns if c in merged.columns]
        result = merged[output_columns].copy()

        logger.info(f"Final output: {len(result)} pair-property rows")
        return result

    def deduplicate_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate pairs.

        A pair is considered duplicate if (mol_a, mol_b, property_name, target_chembl_id)
        is the same. Keep the first occurrence.
        """
        before = len(df)

        dedup_columns = ['mol_a', 'mol_b', 'property_name', 'target_chembl_id']
        dedup_columns = [c for c in dedup_columns if c in df.columns]

        df = df.drop_duplicates(subset=dedup_columns, keep='first')

        after = len(df)
        if before != after:
            logger.info(f"Deduplicated: {before} -> {after} rows (removed {before - after})")

        return df

    def extract_pairs(
        self,
        molecules_path: Path,
        bioactivity_path: Path,
        output_path: Path,
        work_dir: Optional[Path] = None,
        property_filter: Optional[set] = None,
        keep_intermediate: bool = False,
        compute_atom_mapping: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Full pipeline to extract MMP pairs.

        Args:
            molecules_path: Path to molecules CSV
            bioactivity_path: Path to bioactivity CSV
            output_path: Path for output pairs CSV
            work_dir: Working directory for intermediate files
            property_filter: Optional set of property names to include
            keep_intermediate: Whether to keep intermediate files
            compute_atom_mapping: Whether to compute atom-level mapping
            verbose: Whether to stream mmpdb output in real-time

        Returns:
            DataFrame with extracted pairs
        """
        self.verbose = verbose
        logger.info("=" * 60)
        logger.info("Starting MMP extraction with mmpdb")
        logger.info("=" * 60)

        # Load input data
        logger.info(f"Loading molecules from {molecules_path}")
        molecules_df = pd.read_csv(molecules_path)
        logger.info(f"Loaded {len(molecules_df)} molecules")

        logger.info(f"Loading bioactivity from {bioactivity_path}")
        bioactivity_df = pd.read_csv(bioactivity_path)
        logger.info(f"Loaded {len(bioactivity_df)} bioactivity measurements")

        # Create work directory
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="mmpdb_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Working directory: {work_dir}")

        try:
            # Step 1: Prepare SMILES file
            smiles_file = work_dir / "molecules.smi"
            self.prepare_smiles_file(molecules_df, smiles_file)

            # Step 2: Fragment molecules
            fragdb_file = work_dir / "molecules.fragdb"
            self.fragment_molecules(smiles_file, fragdb_file)

            # Step 3: Index to CSV
            pairs_csv = work_dir / "pairs_raw.csv"
            self.index_fragments_to_csv(fragdb_file, pairs_csv)

            # Step 4: Parse mmpdb output
            pairs_df = self.parse_mmpdb_csv(pairs_csv)

            # Step 5: Join with bioactivity
            result_df = self.join_with_bioactivity(
                pairs_df,
                bioactivity_df,
                property_filter=property_filter
            )

            # Step 6: Deduplicate
            result_df = self.deduplicate_pairs(result_df)

            # Step 7: Compute atom mapping (optional)
            if compute_atom_mapping:
                result_df = self.add_atom_mapping_to_df(result_df)

            # Remove 'constant' column from final output (only needed for atom mapping)
            if 'constant' in result_df.columns and not compute_atom_mapping:
                result_df = result_df.drop(columns=['constant'])

            # Save output
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(result_df)} pairs to {output_path}")

            return result_df

        finally:
            # Cleanup
            if not keep_intermediate:
                import shutil
                if work_dir.exists() and str(work_dir).startswith(tempfile.gettempdir()):
                    shutil.rmtree(work_dir)
                    logger.info(f"Cleaned up working directory: {work_dir}")

    def extract_pairs_chunked(
        self,
        molecules_path: Path,
        bioactivity_path: Path,
        output_path: Path,
        chunk_size: int = 50000,
        work_dir: Optional[Path] = None,
        property_filter: Optional[set] = None
    ) -> Path:
        """
        Extract pairs in chunks for very large datasets.

        This processes molecules in chunks to manage memory usage.
        The mmpdb fragmentation is still done on the full dataset for
        best pair coverage, but joining is done in chunks.
        """
        logger.info("=" * 60)
        logger.info("Starting CHUNKED MMP extraction with mmpdb")
        logger.info("=" * 60)

        # Load molecules (we need all for fragmentation)
        logger.info(f"Loading molecules from {molecules_path}")
        molecules_df = pd.read_csv(molecules_path)
        logger.info(f"Loaded {len(molecules_df)} molecules")

        # Create work directory
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="mmpdb_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Working directory: {work_dir}")

        # Step 1-3: Fragment and index (full dataset)
        smiles_file = work_dir / "molecules.smi"
        self.prepare_smiles_file(molecules_df, smiles_file)

        fragdb_file = work_dir / "molecules.fragdb"
        self.fragment_molecules(smiles_file, fragdb_file)

        pairs_csv = work_dir / "pairs_raw.csv"
        self.index_fragments_to_csv(fragdb_file, pairs_csv)

        pairs_df = self.parse_mmpdb_csv(pairs_csv)
        logger.info(f"Generated {len(pairs_df)} raw pairs")

        # Process bioactivity in chunks
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load bioactivity
        logger.info(f"Loading bioactivity from {bioactivity_path}")
        bioactivity_df = pd.read_csv(bioactivity_path)

        if property_filter:
            bioactivity_df = bioactivity_df[
                bioactivity_df['property_name'].isin(property_filter)
            ].copy()

        logger.info(f"Loaded {len(bioactivity_df)} bioactivity measurements")

        # Join with bioactivity (this is memory intensive for large datasets)
        result_df = self.join_with_bioactivity(
            pairs_df,
            bioactivity_df,
            property_filter=None  # Already filtered
        )

        # Deduplicate
        result_df = self.deduplicate_pairs(result_df)

        # Save
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(result_df)} pairs to {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract MMP pairs using mmpdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--molecules", "-m",
        type=Path,
        default=DEFAULT_MOLECULES,
        help="Path to molecules CSV (chembl_id, smiles)"
    )

    parser.add_argument(
        "--bioactivity", "-b",
        type=Path,
        default=DEFAULT_BIOACTIVITY,
        help="Path to bioactivity CSV"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for pairs CSV"
    )

    parser.add_argument(
        "--work-dir", "-w",
        type=Path,
        default=None,
        help="Working directory for intermediate files"
    )

    parser.add_argument(
        "--num-cuts",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Maximum number of cuts for fragmentation"
    )

    parser.add_argument(
        "--max-variable-heavies",
        type=int,
        default=10,
        help="Maximum heavy atoms in variable fragment"
    )

    parser.add_argument(
        "--num-jobs", "-j",
        type=int,
        default=4,
        help="Number of parallel jobs for fragmentation"
    )

    parser.add_argument(
        "--property-filter", "-p",
        type=str,
        nargs="+",
        default=None,
        help="Property names to include (e.g., pchembl_value)"
    )

    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files"
    )

    parser.add_argument(
        "--atom-mapping",
        action="store_true",
        help="Compute atom-level mapping (adds removed_atoms_A, added_atoms_B, etc.)"
    )

    parser.add_argument(
        "--mmpdb-path",
        type=str,
        default="mmpdb",
        help="Path to mmpdb executable"
    )

    parser.add_argument(
        "--cut-smarts",
        type=str,
        default=None,
        choices=['default', 'cut_AlkylChains', 'cut_Amides', 'cut_all', 'exocyclic'],
        help="SMARTS pattern for bond cutting (default: mmpdb default which is drug-like bonds only)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Stream mmpdb output in real-time (shows progress)"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / "chembl_pairs_mmpdb.csv"

    # Create extractor
    extractor = MMPDBExtractor(
        mmpdb_path=args.mmpdb_path,
        num_cuts=args.num_cuts,
        max_variable_heavies=args.max_variable_heavies,
        num_jobs=args.num_jobs,
        cut_smarts=args.cut_smarts
    )

    # Convert property filter to set
    property_filter = set(args.property_filter) if args.property_filter else None

    # Run extraction
    result_df = extractor.extract_pairs(
        molecules_path=args.molecules,
        bioactivity_path=args.bioactivity,
        output_path=args.output,
        work_dir=args.work_dir,
        property_filter=property_filter,
        keep_intermediate=args.keep_intermediate,
        compute_atom_mapping=args.atom_mapping,
        verbose=args.verbose
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total pairs: {len(result_df)}")
    print(f"Unique molecular pairs: {result_df[['mol_a', 'mol_b']].drop_duplicates().shape[0]}")
    print(f"Unique edits: {result_df['edit_smiles'].nunique()}")
    if 'property_name' in result_df.columns:
        print(f"Properties: {result_df['property_name'].unique().tolist()}")
    if 'target_chembl_id' in result_df.columns:
        print(f"Targets: {result_df['target_chembl_id'].nunique()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
