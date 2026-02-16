"""
Long-format MMP extraction optimized for efficiency.

SCHEMA:
    mol_a, mol_b, edit_smiles, num_cuts, property_name, value_a, value_b, delta,
    target_name, target_chembl_id, doc_id_a, doc_id_b, assay_id_a, assay_id_b,
    removed_atoms_A, added_atoms_B, attach_atoms_A, mapped_pairs

Fields:
    - mol_a, mol_b: Full molecule SMILES (for debugging, will convert to IDs later)
    - edit_smiles: Canonical reaction SMILES (e.g., "C>>CC") for encoding with ChemBERTa
                   Computed using RDKit MMP fragmentation + canonical SMILES
    - num_cuts: Number of bond cuts used to generate this pair (1, 2, or 3)
    - property_name: Property being compared (e.g., "IC50_EGFR")
    - value_a, value_b: Property values
    - delta: value_b - value_a (observed change)
    - target_name, target_chembl_id: Target info (for bioactivity only)
    - doc_id_a, doc_id_b: Document/publication IDs (for batch effect control)
    - assay_id_a, assay_id_b: Assay IDs (for experimental protocol tracking)
    - removed_atoms_A: Atom indices in mol_a of leaving fragment (semicolon-separated)
    - added_atoms_B: Atom indices in mol_b of incoming fragment (semicolon-separated)
    - attach_atoms_A: Attachment point atom indices in mol_a (semicolon-separated)
    - mapped_pairs: Atom mapping tuples (a,b;c,d format) for changed region

For edit embeddings:
    Option 1: Use edit_smiles directly with ChemBERTa/transformers
    Option 2: Compute fingerprint difference on-the-fly: fp(mol_b) - fp(mol_a)
    Option 3: Combined embedding: F(mol, edit) -> property_change
    Option 4: Structured edit embedding using atom-level mapping (NEW)

This eliminates NaN/missing issues and only stores what exists.
"""

import logging
import pandas as pd
import numpy as np
import hashlib
import gc
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA, Descriptors
from tqdm import tqdm
from .mmp_atom_mapping_fast import extract_atom_mapping_fast, serialize_mapping

logger = logging.getLogger(__name__)


def _worker_wrapper_for_imap(args):
    """
    Module-level wrapper for multiprocessing.Pool.imap().

    This function is picklable (unlike nested functions) and unpacks
    arguments to call the static worker method.
    """
    chunk_id, chunk, core_index, property_lookup, fragments, max_mw_delta, checkpoint_dir, micro_batch_size, property_filter = args
    return LongFormatMMPExtractor._process_core_chunk_worker(
        chunk_id=chunk_id,
        chunk=chunk,
        core_index=core_index,
        property_lookup=property_lookup,
        fragments=fragments,
        max_mw_delta=max_mw_delta,
        checkpoint_dir=checkpoint_dir,
        micro_batch_size=micro_batch_size,
        property_filter=property_filter
    )


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class LongFormatMMPExtractor:
    """
    Extract matched molecular pairs and output in long format.

    Long format: one row per pair-property combination.
    More efficient storage, no NaN issues, easy filtering.
    """

    def __init__(self, max_cuts: int = 1):
        """
        Initialize extractor.

        Args:
            max_cuts: Maximum number of cuts for MMP fragmentation (default: 1)
                     1 = single attachment point (simpler, closer edits)
                     2 = two attachment points (more complex edits)
        """
        self.max_cuts = max_cuts

    def fragment_molecule(self, smiles: str) -> Dict[str, str]:
        """
        Fragment molecule using rdMMPA.

        Args:
            smiles: SMILES string

        Returns:
            Dict mapping core -> attachment
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            # For max_cuts=1, we actually fragment with maxCuts=2 to get proper cores for grouping
            # but will filter to only single-cut pairs during extraction
            fragment_max_cuts = max(2, self.max_cuts)
            frags = rdMMPA.FragmentMol(mol, maxCuts=fragment_max_cuts, resultsAsMols=False)

            core_to_attachment = {}
            for core, chains in frags:
                if core and chains:  # Both must be non-empty
                    core_to_attachment[core] = chains

            return core_to_attachment

        except Exception as e:
            logger.debug(f"Fragmentation failed for {smiles}: {e}")
            return {}

    def extract_pairs_long_format(
        self,
        molecules_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame,
        max_mw_delta: float = 200,
        min_similarity: float = 0.4,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1000,
        resume_from_checkpoint: bool = True,
        micro_batch_size: int = 1_000_000,  # Flush every 1M pairs to reduce disk I/O
        property_filter: Optional[set] = None,  # NEW: Only extract these properties
        n_jobs: int = -1,  # NEW: Number of CPU cores for parallelization (-1 = all cores)
        min_molecules_per_core: int = 10,  # NEW: Minimum molecules per core to keep
        max_molecules_per_core_sample: int = 1000  # NEW: Sample large cores to this size
    ) -> pd.DataFrame:
        """
        Extract molecular pairs in long format with MEMORY-EFFICIENT streaming.

        Args:
            molecules_df: DataFrame with molecules and computed properties
            bioactivity_df: DataFrame with bioactivity (long format)
            max_mw_delta: Maximum MW difference for pairs
            min_similarity: Minimum Tanimoto similarity
            checkpoint_dir: Optional directory for checkpoints
            checkpoint_every: Report progress every N cores (default: 1000)
            resume_from_checkpoint: Try to resume from checkpoint if exists (default: True)
            micro_batch_size: Flush to disk every N pairs (default: 200)
            property_filter: Optional set of property names to extract (default: None = all)
            n_jobs: Number of CPU cores for parallelization (default: -1 = all cores, 1 = single-threaded)
            min_molecules_per_core: Minimum molecules per core to keep (default: 10)
            max_molecules_per_core_sample: Maximum molecules to sample per core (default: 1000).
                Cores with more molecules will be randomly sampled to this size to prevent
                computational explosion. Use None to disable sampling.

        Returns:
            Long-format DataFrame with pairs
        """
        mem_start = get_memory_usage_mb()
        logger.info("=" * 70)
        logger.info(" LONG-FORMAT PAIR EXTRACTION (MEMORY-EFFICIENT)")
        logger.info("=" * 70)
        logger.info(f" Molecules: {len(molecules_df):,}")
        logger.info(f" Bioactivity: {len(bioactivity_df):,} measurements")
        logger.info(f" Max MW delta: {max_mw_delta}")
        logger.info(f" Min similarity: {min_similarity}")
        logger.info(f" Micro-batch size: {micro_batch_size} pairs")
        if property_filter:
            logger.info(f" Property filter: {sorted(property_filter)}")
        logger.info(f" Memory at start: {mem_start:.1f} MB")
        logger.info("=" * 70)
        logger.info("")

        # Step 1: Create SMILES index and property lookup
        logger.info("Step 1: Creating property lookup...")

        property_lookup = self._create_property_lookup(molecules_df, bioactivity_df)
        logger.info(f"  ✓ Created lookup for {len(property_lookup):,} molecules")
        mem_after_lookup = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_lookup:.1f} MB (+{mem_after_lookup - mem_start:.1f} MB)")
        logger.info("")

        # Step 2: Fragment molecules (with caching)
        logger.info("Step 2: Fragmenting molecules...")

        # Check for cached fragments
        fragments_cache_file = None
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            fragments_cache_file = checkpoint_path / "fragments_cache.pkl"

        if fragments_cache_file and fragments_cache_file.exists() and resume_from_checkpoint:
            logger.info("  Loading cached fragments...")
            import pickle
            with open(fragments_cache_file, 'rb') as f:
                fragments = pickle.load(f)
            logger.info(f"  ✓ Loaded {len(fragments):,} cached fragments")
        else:
            logger.info("  Computing fragments (this may take a while)...")
            smiles_list = molecules_df['smiles'].tolist()

            # Parallelize fragmentation across CPU cores
            import multiprocessing as mp

            # Determine number of cores to use
            if n_jobs == -1:
                n_cores = mp.cpu_count()
            elif n_jobs <= 0:
                n_cores = max(1, mp.cpu_count() + n_jobs)
            else:
                n_cores = min(n_jobs, mp.cpu_count())

            if n_cores == 1:
                # Single-threaded mode (useful for debugging)
                logger.info("  Using single-threaded mode...")
                results = [self.fragment_molecule(smiles) for smiles in tqdm(smiles_list, desc="Fragmenting")]
            else:
                # Multi-threaded mode
                logger.info(f"  Using {n_cores} CPU cores for parallel fragmentation...")

                # Use multiprocessing pool to fragment in parallel
                with mp.Pool(processes=n_cores) as pool:
                    results = list(tqdm(
                        pool.imap(self.fragment_molecule, smiles_list, chunksize=100),
                        total=len(smiles_list),
                        desc="Fragmenting"
                    ))

            # Build fragments dict from results
            fragments = {}
            for smiles, frags in zip(smiles_list, results):
                if frags:
                    fragments[smiles] = frags

            logger.info(f"  ✓ Fragmented {len(fragments):,} molecules")

            # Cache fragments for next time
            if fragments_cache_file:
                logger.info("  Saving fragments cache...")
                import pickle
                with open(fragments_cache_file, 'wb') as f:
                    pickle.dump(fragments, f)
                logger.info("  ✓ Cached fragments for future runs")

        mem_after_frag = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_frag:.1f} MB (+{mem_after_frag - mem_after_lookup:.1f} MB)")
        logger.info("")

        # Step 3: Index by core for efficient pairing
        logger.info("Step 3: Indexing by core...")

        core_index = defaultdict(list)
        for smiles, frags in fragments.items():
            for core in frags.keys():
                core_index[core].append(smiles)

        logger.info(f"  ✓ Found {len(core_index):,} total cores")

        # Filter: only keep cores with enough molecules for better pair density
        cores_to_keep = {core: mols for core, mols in core_index.items() if len(mols) >= min_molecules_per_core}

        num_deleted = len(core_index) - len(cores_to_keep)
        core_index = cores_to_keep
        del cores_to_keep
        gc.collect()

        filtered_cores = sorted(core_index.keys(), key=lambda c: len(core_index[c]))

        if len(filtered_cores) == 0:
            logger.warning(f"  No cores with {min_molecules_per_core}+ molecules found!")
            return pd.DataFrame(columns=[
                'mol_a', 'mol_b', 'edit_smiles', 'num_cuts', 'property_name', 'value_a', 'value_b',
                'delta', 'target_name', 'target_chembl_id', 'doc_id_a', 'doc_id_b', 'assay_id_a', 'assay_id_b',
                'removed_atoms_A', 'added_atoms_B', 'attach_atoms_A', 'mapped_pairs'
            ])

        core_sizes = [len(core_index[c]) for c in filtered_cores]
        logger.info(f"  ✓ Kept {len(filtered_cores):,} cores with {min_molecules_per_core}+ molecules")
        logger.info(f"  ✓ Deleted {num_deleted:,} cores to save memory")
        logger.info(f"  ✓ Core sizes: min={min(core_sizes)}, max={max(core_sizes)}, avg={sum(core_sizes)/len(core_sizes):.1f}")

        mem_after_index = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_index:.1f} MB (+{mem_after_index - mem_after_frag:.1f} MB)")
        logger.info("")

        # Step 3.5: Sample large cores to prevent computational explosion
        if max_molecules_per_core_sample:
            import random

            logger.info(f"Step 3.5: Sampling large cores (max_molecules_per_core_sample={max_molecules_per_core_sample:,})...")

            cores_sampled = 0
            total_molecules_before = sum(len(core_index[c]) for c in filtered_cores)
            largest_cores_sampled = []

            for core in filtered_cores:
                mols = core_index[core]
                if len(mols) > max_molecules_per_core_sample:
                    # Deterministic sampling using core hash as seed for reproducibility
                    random.seed(hash(core))
                    sampled_mols = random.sample(list(mols), max_molecules_per_core_sample)
                    core_index[core] = set(sampled_mols)
                    cores_sampled += 1

                    # Track largest cores for logging
                    if len(largest_cores_sampled) < 10:
                        largest_cores_sampled.append((core, len(mols), max_molecules_per_core_sample))

            if cores_sampled > 0:
                total_molecules_after = sum(len(core_index[c]) for c in filtered_cores)
                logger.info(f"  ✓ Sampled {cores_sampled:,} large cores")
                logger.info(f"  ✓ Total molecules in cores: {total_molecules_before:,} → {total_molecules_after:,} ({100*total_molecules_after/total_molecules_before:.1f}%)")
                logger.info(f"  ✓ Top sampled cores:")
                for i, (core, orig_size, sampled_size) in enumerate(largest_cores_sampled[:5], 1):
                    potential_pairs_before = orig_size * (orig_size - 1) // 2
                    potential_pairs_after = sampled_size * (sampled_size - 1) // 2
                    logger.info(f"      {i}. {orig_size:,} → {sampled_size:,} molecules ({potential_pairs_before:,} → {potential_pairs_after:,} potential pairs)")
            else:
                logger.info(f"  ✓ No cores exceeded {max_molecules_per_core_sample:,} molecules - no sampling needed")
            logger.info("")

        # Step 4: Extract pairs with PARALLEL PROCESSING
        logger.info("Step 4: Extracting pairs (PARALLEL MODE)...")

        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            # Use project-level cache directory instead of /tmp
            import uuid
            from pathlib import Path
            project_cache = Path(__file__).parent.parent.parent / ".cache" / "mmp_extraction"
            project_cache.mkdir(parents=True, exist_ok=True)
            checkpoint_path = project_cache / f"mmp_parallel_{uuid.uuid4().hex[:8]}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created checkpoint dir: {checkpoint_path}")

        # Create balanced chunks
        logger.info(f"  Creating balanced chunks (target: 500 molecules per chunk)...")
        chunks = self._create_balanced_core_chunks(
            core_index=core_index,
            target_molecules_per_chunk=500
        )
        logger.info(f"  ✓ Created {len(chunks)} chunks")
        logger.info("")

        # Check chunk progress (for resume support)
        incomplete_chunks, completed_files = self._check_chunk_progress(
            chunks=chunks,
            checkpoint_dir=str(checkpoint_path)
        )

        if completed_files:
            logger.info(f"  ✓ Found {len(completed_files)} completed chunks (resuming)")

        logger.info(f"  ✓ {len(incomplete_chunks)} chunks to process")
        logger.info("")

        # Determine number of workers
        import multiprocessing as mp
        if n_jobs == -1:
            num_workers = mp.cpu_count()
        elif n_jobs < -1:
            num_workers = max(1, mp.cpu_count() + n_jobs + 1)
        else:
            num_workers = max(1, n_jobs)

        logger.info(f"  Using {num_workers} parallel workers")
        logger.info(f"  Micro-batch size: {micro_batch_size} pairs")
        logger.info("")

        # Create property lookup by chembl_id (not SMILES) for workers
        # Keep full structure: {chembl_id: {'chembl_id': ..., 'properties': {...}}}
        property_lookup_by_id = {}
        for smiles, data in property_lookup.items():
            chembl_id = data['chembl_id']
            property_lookup_by_id[chembl_id] = data

        # Create fragments lookup by chembl_id
        fragments_by_id = {}
        for smiles, frags in fragments.items():
            chembl_id = property_lookup[smiles]['chembl_id']
            fragments_by_id[chembl_id] = frags

        # Convert core_index from SMILES to chembl_id
        core_index_by_id = {}
        for core, smiles_list in core_index.items():
            chembl_ids = [property_lookup[s]['chembl_id'] for s in smiles_list if s in property_lookup]
            if chembl_ids:
                core_index_by_id[core] = chembl_ids

        mem_peak = mem_after_index

        # Process chunks in parallel
        if num_workers == 1 or len(incomplete_chunks) == 1:
            # Single-threaded mode (for debugging)
            logger.info("  Running in single-threaded mode...")
            worker_files = []
            for chunk_id, chunk in incomplete_chunks:
                output_file = self._process_core_chunk_worker(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    core_index=core_index_by_id,
                    property_lookup=property_lookup_by_id,
                    fragments=fragments_by_id,
                    max_mw_delta=max_mw_delta,
                    checkpoint_dir=str(checkpoint_path),
                    micro_batch_size=micro_batch_size,
                    property_filter=property_filter
                )
                worker_files.append(output_file)
        else:
            # Multi-process mode
            logger.info("  Starting parallel processing...")
            worker_files = []

            # Prepare arguments for imap (each element is a tuple of arguments)
            worker_args = [
                (
                    chunk_id,
                    chunk,
                    core_index_by_id,
                    property_lookup_by_id,
                    fragments_by_id,
                    max_mw_delta,
                    str(checkpoint_path),
                    micro_batch_size,
                    property_filter
                )
                for chunk_id, chunk in incomplete_chunks
            ]

            with mp.Pool(processes=num_workers) as pool:
                # Use imap_unordered for parallel processing with small chunksize for progress updates
                results = list(tqdm(
                    pool.imap_unordered(_worker_wrapper_for_imap, worker_args, chunksize=1),
                    total=len(worker_args),
                    desc="Processing chunks"
                ))
                worker_files.extend(results)

        # Add completed files
        worker_files.extend(completed_files)

        logger.info(f"  ✓ All chunks complete!")
        logger.info("")

        # Merge worker files
        final_output = checkpoint_path / "pairs_final.csv"
        self._merge_worker_files(worker_files=worker_files, final_output=final_output)

        # Count total pairs
        import subprocess
        try:
            result = subprocess.run(['wc', '-l', str(final_output)],
                                  capture_output=True, text=True, check=True)
            line_count = int(result.stdout.split()[0])
            total_pairs_written = max(0, line_count - 1)
        except:
            with open(final_output, 'r') as f:
                total_pairs_written = sum(1 for _ in f) - 1

        logger.info(f"  ✓ Extracted {total_pairs_written:,} pair-property combinations")
        mem_after_extraction = get_memory_usage_mb()
        logger.info(f"  Memory after extraction: {mem_after_extraction:.1f} MB")
        logger.info(f"  Peak memory during extraction: {mem_peak:.1f} MB")
        logger.info("")

        # Read final result
        logger.info("  Reading final result from disk...")
        df_pairs_long = pd.read_csv(final_output)

        # Clean up if using temporary directory
        if not checkpoint_dir:
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"  Cleaned up temporary directory")
        logger.info("")

        if len(df_pairs_long) == 0:
            logger.warning("  No pairs found!")
            return df_pairs_long

        # Statistics
        n_unique_pairs = df_pairs_long[['mol_a', 'mol_b']].drop_duplicates().shape[0]
        n_unique_edits = df_pairs_long['edit_smiles'].nunique()
        n_unique_properties = df_pairs_long['property_name'].nunique()

        mem_final = get_memory_usage_mb()

        logger.info("=" * 70)
        logger.info(" EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f" Total rows: {len(df_pairs_long):,}")
        logger.info(f" Unique pairs: {n_unique_pairs:,}")
        logger.info(f" Unique edits: {n_unique_edits:,}")
        logger.info(f" Unique properties: {n_unique_properties:,}")
        logger.info(f" Avg properties per pair: {len(df_pairs_long) / n_unique_pairs:.1f}")
        logger.info("")
        logger.info(f" MEMORY STATS:")
        logger.info(f"   Start: {mem_start:.1f} MB")
        logger.info(f"   Peak: {mem_peak:.1f} MB")
        logger.info(f"   Final: {mem_final:.1f} MB")
        logger.info(f"   Peak increase: +{mem_peak - mem_start:.1f} MB")
        logger.info("=" * 70)
        logger.info("")

        return df_pairs_long

    def _create_property_lookup(
        self,
        molecules_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Create property lookup: {smiles: {properties: {name: value}}}
        """
        lookup = {}

        # Computed properties
        computed_props = [
            'alogp', 'mw', 'mw_freebase', 'hbd', 'hba', 'psa', 'rtb',
            'aromatic_rings', 'heavy_atoms', 'qed_weighted',
            'num_ro5_violations', 'np_likeness_score'
        ]

        # Create chembl_id -> smiles mapping for O(1) lookup
        chembl_to_smiles = dict(zip(molecules_df['chembl_id'], molecules_df['smiles']))

        for _, row in molecules_df.iterrows():
            smiles = row['smiles']
            chembl_id = row['chembl_id']

            lookup[smiles] = {
                'chembl_id': chembl_id,
                'properties': {}
            }

            for prop in computed_props:
                if prop in row and pd.notna(row[prop]):
                    lookup[smiles]['properties'][prop] = row[prop]

            # CRITICAL: Compute MW on-the-fly if not provided in molecules_df
            # This is required for the MW filter in pair extraction
            if 'mw' not in lookup[smiles]['properties']:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        lookup[smiles]['properties']['mw'] = Descriptors.MolWt(mol)
                except Exception as e:
                    logger.debug(f"MW computation failed for {smiles[:30]}: {e}")
                    pass  # Skip if MW computation fails

        # Bioactivity properties (with target info)
        # IMPORTANT: Use composite key (property_name, target_chembl_id) to preserve
        # multiple measurements per molecule across different targets
        for _, row in bioactivity_df.iterrows():
            chembl_id = row['chembl_id']

            # O(1) lookup instead of O(n) scan
            smiles = chembl_to_smiles.get(chembl_id)
            if not smiles or smiles not in lookup:
                continue

            prop_name = row['property_name']
            # Get value from 'value' column (long format has this column)
            # Also support 'pchembl_value' and 'standard_value' for backwards compatibility
            value = row.get('value') or row.get('pchembl_value') or row.get('standard_value')
            target_name = row.get('target_name', '')
            target_chembl_id = row.get('target_chembl_id', '')
            doc_id = row.get('doc_id')
            assay_id = row.get('assay_id')

            if pd.notna(value):
                # Use composite key: (property_name, target_chembl_id)
                # This preserves ALL measurements (same molecule tested on multiple targets)
                composite_key = f"{prop_name}@{target_chembl_id}"

                lookup[smiles]['properties'][composite_key] = {
                    'value': value,
                    'property_name': prop_name,  # Store original property name
                    'target_name': target_name,
                    'target_chembl_id': target_chembl_id,
                    'doc_id': doc_id if pd.notna(doc_id) else None,
                    'assay_id': assay_id if pd.notna(assay_id) else None
                }

        # DEBUG: Check how many molecules have MW computed
        molecules_with_mw = sum(1 for data in lookup.values() if 'mw' in data['properties'])
        logger.info(f"  MW computed for {molecules_with_mw}/{len(lookup)} molecules")

        # Count bioactivity measurements (composite keys)
        total_measurements = sum(
            1 for data in lookup.values()
            for key in data['properties'].keys()
            if '@' in str(key)  # Composite keys have '@'
        )
        logger.info(f"  Bioactivity: {total_measurements} measurements across {len(lookup)} molecules")

        return lookup

    def _extract_single_pair(
        self,
        smiles_a: str,
        smiles_b: str,
        frags_a: Dict[str, str],
        frags_b: Dict[str, str],
        property_lookup: Dict,
        property_filter: Optional[set] = None
    ) -> List[Dict]:
        """
        Extract pair and return list of rows (one per property).

        MINIMAL SCHEMA: Only stores mol_a, mol_b, property values, and target info.
        Edit embeddings should be computed on-the-fly: fp(mol_b) - fp(mol_a)

        Args:
            property_filter: Optional set of property names to include (default: all)

        Returns:
            List of dicts, each representing one row in long format
        """
        # Find common cores
        common_cores = set(frags_a.keys()) & set(frags_b.keys())

        if not common_cores:
            # Try all combinations of fragments to find if any share common parts
            # Find the LARGEST common core (most atoms)
            attachment_a = None
            attachment_b = None
            best_core_size = 0

            for core_a, chains_a in frags_a.items():
                for core_b, chains_b in frags_b.items():
                    # Split chains into parts
                    parts_a = set(chains_a.split('.'))
                    parts_b = set(chains_b.split('.'))

                    # Find common parts (these form the core)
                    common_parts = parts_a & parts_b

                    if common_parts and len(common_parts) > 0:
                        # Calculate core size (number of atoms in common parts)
                        core_size = 0
                        for part in common_parts:
                            part_clean = part.replace('[*:1]', '[H]').replace('[*:2]', '[H]').replace('[*:3]', '[H]')
                            try:
                                mol_part = Chem.MolFromSmiles(part_clean)
                                if mol_part:
                                    core_size += mol_part.GetNumAtoms()
                            except:
                                pass

                        # Keep the largest core
                        if core_size > best_core_size:
                            best_core_size = core_size
                            attachment_a = chains_a
                            attachment_b = chains_b

            if not attachment_a:
                return []  # No valid MMP found
        else:
            # Take first common core
            core = list(common_cores)[0]
            attachment_a = frags_a[core]
            attachment_b = frags_b[core]

        if attachment_a == attachment_b:
            return []  # Not a transformation

        # Extract pure edit (what actually changed) for canonical representation
        # Split fragments by '.' to get individual pieces
        parts_a = set(attachment_a.split('.'))
        parts_b = set(attachment_b.split('.'))

        # Find what's unique to each side (the actual edit)
        edit_from_parts = parts_a - parts_b
        edit_to_parts = parts_b - parts_a

        # Calculate number of cuts (max of parts on either side)
        num_cuts = max(len(edit_from_parts), len(edit_to_parts))

        # For max_cuts=1: filter to only accept single-cut pairs
        # (where only one fragment differs on each side)
        if self.max_cuts == 1:
            if len(edit_from_parts) != 1 or len(edit_to_parts) != 1:
                return []  # Not a single-cut pair

        # Convert to canonical SMILES (replace attachment points with H)
        # Use RDKit to canonicalize for consistency
        if edit_from_parts:
            edit_from_raw = '.'.join(sorted(edit_from_parts)).replace('[*:1]', '[H]').replace('[*:2]', '[H]')
            mol_from = Chem.MolFromSmiles(edit_from_raw)
            edit_from = Chem.MolToSmiles(mol_from) if mol_from else edit_from_raw
        else:
            edit_from = ''

        if edit_to_parts:
            edit_to_raw = '.'.join(sorted(edit_to_parts)).replace('[*:1]', '[H]').replace('[*:2]', '[H]')
            mol_to = Chem.MolFromSmiles(edit_to_raw)
            edit_to = Chem.MolToSmiles(mol_to) if mol_to else edit_to_raw
        else:
            edit_to = ''

        # Create reaction SMILES (CANONICAL format for encoding with ChemBERTa, etc.)
        # Format: reactant>>product
        # This IS the RDKit canonical way - MMP fragmentation + canonical SMILES
        edit_smiles = f"{edit_from}>>{edit_to}" if edit_from and edit_to else ''

        # Extract atom-level mapping for structured edit predictor
        # Use FAST version that leverages MMP core data directly (84x faster!)
        common_parts = parts_a & parts_b

        # Join multiple common parts with '.' (for multi-cut pairs)
        # Sort for consistency
        core_smiles = '.'.join(sorted(common_parts)) if common_parts else ""
        removed_fragment = '.'.join(sorted(edit_from_parts)) if edit_from_parts else ""
        added_fragment = '.'.join(sorted(edit_to_parts)) if edit_to_parts else ""

        atom_mapping = extract_atom_mapping_fast(
            smiles_a, smiles_b, core_smiles, removed_fragment, added_fragment, num_cuts
        )
        atom_mapping_serialized = serialize_mapping(atom_mapping)

        # Get properties for both molecules
        props_a = property_lookup[smiles_a]['properties']
        props_b = property_lookup[smiles_b]['properties']

        # Find shared properties
        shared_props = set(props_a.keys()) & set(props_b.keys())

        if not shared_props:
            return []

        # Create one row per property
        rows = []

        for composite_key in shared_props:
            value_a_raw = props_a[composite_key]
            value_b_raw = props_b[composite_key]

            # Extract value and target info if dict (bioactivity), else just use value (computed prop)
            if isinstance(value_a_raw, dict):
                value_a = value_a_raw['value']
                # Extract original property_name from dict (for composite keys)
                property_name = value_a_raw.get('property_name', composite_key)
                target_name = value_a_raw.get('target_name', '')
                target_chembl_id = value_a_raw.get('target_chembl_id', '')
                doc_id_a = value_a_raw.get('doc_id')
                assay_id_a = value_a_raw.get('assay_id')
            else:
                value_a = value_a_raw
                property_name = composite_key  # Computed properties use simple keys
                target_name = ''
                target_chembl_id = ''
                doc_id_a = None
                assay_id_a = None

            # Skip if property not in filter (check AFTER extracting property_name)
            if property_filter and property_name not in property_filter:
                continue

            if isinstance(value_b_raw, dict):
                value_b = value_b_raw['value']
                doc_id_b = value_b_raw.get('doc_id')
                assay_id_b = value_b_raw.get('assay_id')
            else:
                value_b = value_b_raw
                doc_id_b = None
                assay_id_b = None

            # Skip if either is None/NaN
            if value_a is None or value_b is None:
                continue
            if isinstance(value_a, float) and np.isnan(value_a):
                continue
            if isinstance(value_b, float) and np.isnan(value_b):
                continue

            delta = value_b - value_a

            # MINIMAL SCHEMA - only essential fields + canonical edit representation + batch tracking
            row = {
                'mol_a': smiles_a,
                'mol_b': smiles_b,
                'edit_smiles': edit_smiles,  # ⭐ CANONICAL: "C>>CC" for ChemBERTa encoding
                'num_cuts': num_cuts,  # Number of bond cuts (1, 2, or 3)
                'property_name': property_name,  # Use extracted property_name, not composite_key
                'value_a': value_a,
                'value_b': value_b,
                'delta': delta,
                'target_name': target_name,
                'target_chembl_id': target_chembl_id,
                'doc_id_a': doc_id_a,  # Publication/study ID for molecule A measurement
                'doc_id_b': doc_id_b,  # Publication/study ID for molecule B measurement
                'assay_id_a': assay_id_a,  # Assay ID for molecule A measurement
                'assay_id_b': assay_id_b,  # Assay ID for molecule B measurement
                # Atom-level mapping for structured edit predictor
                'removed_atoms_A': atom_mapping_serialized['removed_atoms_A'],
                'added_atoms_B': atom_mapping_serialized['added_atoms_B'],
                'attach_atoms_A': atom_mapping_serialized['attach_atoms_A'],
                'mapped_pairs': atom_mapping_serialized['mapped_pairs']
            }

            rows.append(row)

        return rows

    def _create_balanced_core_chunks(
        self,
        core_index: Dict[str, List[str]],
        target_molecules_per_chunk: int = 500
    ) -> List[Dict]:
        """
        Create balanced chunks where each chunk has ~target_molecules_per_chunk molecules.

        This ensures balanced workload across workers, regardless of core size distribution.

        Args:
            core_index: Dict mapping core SMILES to list of molecule SMILES
            target_molecules_per_chunk: Target number of molecules per chunk (default: 500)

        Returns:
            List of chunk dicts with 'cores' and 'molecule_count' keys
        """
        chunks = []
        current_chunk_cores = []
        current_molecule_count = 0

        # Sort cores by size (largest first) for better load balancing
        sorted_cores = sorted(core_index.keys(), key=lambda c: len(core_index[c]), reverse=True)

        for core in sorted_cores:
            core_molecules = len(core_index[core])

            # If adding this core exceeds target, start new chunk
            # (unless current chunk is empty - avoid tiny cores getting their own chunk)
            if current_molecule_count + core_molecules > target_molecules_per_chunk and current_chunk_cores:
                chunks.append({
                    'cores': current_chunk_cores,
                    'molecule_count': current_molecule_count
                })
                current_chunk_cores = []
                current_molecule_count = 0

            current_chunk_cores.append(core)
            current_molecule_count += core_molecules

        # Add final chunk
        if current_chunk_cores:
            chunks.append({
                'cores': current_chunk_cores,
                'molecule_count': current_molecule_count
            })

        return chunks

    @staticmethod
    def _check_chunk_progress(chunks: List[Dict], checkpoint_dir: Optional[str]) -> Tuple[List[Tuple], List[str]]:
        """
        Check which chunks are already complete (for resume support).

        Args:
            chunks: List of chunk dicts
            checkpoint_dir: Checkpoint directory path

        Returns:
            Tuple of (incomplete_chunks, completed_files)
            - incomplete_chunks: List of (chunk_id, chunk_dict) tuples to process
            - completed_files: List of paths to completed worker files
        """
        if not checkpoint_dir:
            # No checkpointing, process all chunks
            return [(i, chunk) for i, chunk in enumerate(chunks)], []

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        incomplete_chunks = []
        completed_files = []

        for i, chunk in enumerate(chunks):
            checkpoint_file = checkpoint_path / f"checkpoint_chunk_{i}.pkl"
            output_file = checkpoint_path / f"pairs_chunk_{i}.csv"

            if checkpoint_file.exists() and output_file.exists():
                try:
                    import pickle
                    with open(checkpoint_file, 'rb') as f:
                        state = pickle.load(f)
                        if state.get('completed', False):
                            logger.info(f"  Chunk {i} already complete ({state.get('pairs_written', 0):,} pairs)")
                            completed_files.append(str(output_file))
                            continue
                except Exception as e:
                    logger.warning(f"  Failed to load checkpoint for chunk {i}: {e}")

            incomplete_chunks.append((i, chunk))

        return incomplete_chunks, completed_files

    @staticmethod
    def _merge_worker_files(worker_files: List[str], final_output: Path) -> None:
        """
        Merge worker output files into final CSV with deduplication.

        Since molecules can appear in multiple cores (from fragmentation),
        the same pair can be extracted multiple times. We deduplicate based on:
        (mol_a, mol_b, property_name, target_chembl_id)

        Args:
            worker_files: List of paths to worker CSV files
            final_output: Path to final merged CSV file
        """
        logger.info(f"Merging {len(worker_files)} worker files with deduplication...")

        import csv

        # Track seen pairs to deduplicate
        seen_pairs = set()
        duplicates_skipped = 0
        rows_written = 0

        with open(final_output, 'w', newline='', buffering=1024*1024) as out:
            csv_writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)

            # Write header
            csv_writer.writerow([
                'mol_a', 'mol_b', 'edit_smiles', 'num_cuts', 'property_name',
                'value_a', 'value_b', 'delta', 'target_name', 'target_chembl_id',
                'doc_id_a', 'doc_id_b', 'assay_id_a', 'assay_id_b',
                'removed_atoms_A', 'added_atoms_B', 'attach_atoms_A', 'mapped_pairs'
            ])

            # Read and deduplicate worker files
            for worker_file in tqdm(worker_files, desc="Merging"):
                if not Path(worker_file).exists():
                    logger.warning(f"Worker file not found: {worker_file}")
                    continue

                with open(worker_file, 'r', newline='') as f:
                    csv_reader = csv.reader(f)

                    # Skip header if it exists
                    first_row = next(csv_reader, None)
                    if first_row and first_row[0] != 'mol_a':
                        # Not a header, process this row
                        mol_a, mol_b, edit_smiles, num_cuts, property_name = first_row[:5]
                        target_chembl_id = first_row[9] if len(first_row) > 9 else ''

                        # Create deduplication key
                        pair_key = (mol_a, mol_b, property_name, target_chembl_id)

                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            csv_writer.writerow(first_row)
                            rows_written += 1
                        else:
                            duplicates_skipped += 1

                    # Process remaining rows
                    for row in csv_reader:
                        if not row:
                            continue

                        mol_a, mol_b, edit_smiles, num_cuts, property_name = row[:5]
                        target_chembl_id = row[9] if len(row) > 9 else ''

                        # Create deduplication key
                        pair_key = (mol_a, mol_b, property_name, target_chembl_id)

                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            csv_writer.writerow(row)
                            rows_written += 1
                        else:
                            duplicates_skipped += 1

        logger.info(f"✓ Merged to {final_output}")
        logger.info(f"  Rows written: {rows_written:,}")
        logger.info(f"  Duplicates skipped: {duplicates_skipped:,}")

    @staticmethod
    def _process_core_chunk_worker(
        chunk_id: int,
        chunk: Dict,
        core_index: Dict[str, List[str]],
        property_lookup: Dict,
        fragments: Dict,
        max_mw_delta: float,
        checkpoint_dir: str,
        micro_batch_size: int = 1_000_000,
        property_filter: Optional[set] = None
    ) -> str:
        """
        Process a chunk of cores and extract all pairs.

        This is the worker function that runs in parallel.
        Each worker processes its assigned cores and writes to its own file.

        Args:
            chunk_id: Unique ID for this chunk
            chunk: Dict with 'cores' and 'molecule_count'
            core_index: Full core index mapping (chembl_id-based)
            property_lookup: Property lookup table (chembl_id-based)
            fragments: Fragment lookup table (chembl_id-based)
            max_mw_delta: Max MW difference filter
            checkpoint_dir: Directory for checkpoints and output
            micro_batch_size: Flush to disk every N pairs (default: 1000)
            property_filter: Optional set of property names to include (default: None = all)

        Returns:
            Path to worker output file
        """
        import pickle
        import csv
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        checkpoint_path = Path(checkpoint_dir)
        output_file = checkpoint_path / f"pairs_chunk_{chunk_id}.csv"
        checkpoint_file = checkpoint_path / f"checkpoint_chunk_{chunk_id}.pkl"

        # Print start message (commented out to reduce output)
        # print(f"[Chunk {chunk_id}] Starting: {len(chunk['cores'])} cores, ~{chunk['molecule_count']} molecules", flush=True)

        # Extract MW from property_lookup for fast filtering
        mw_lookup = {}
        for chembl_id, data in property_lookup.items():
            mw = data['properties'].get('mw')
            if mw is not None:
                mw_lookup[chembl_id] = mw

        # Open output file with CSV writer for proper escaping
        pair_buffer = []
        pairs_written = 0
        cores_processed = 0

        with open(output_file, 'w', newline='', buffering=1024*1024) as f:
            # Create CSV writer with proper quoting
            csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # Write header
            csv_writer.writerow([
                'mol_a', 'mol_b', 'edit_smiles', 'num_cuts', 'property_name',
                'value_a', 'value_b', 'delta', 'target_name', 'target_chembl_id',
                'doc_id_a', 'doc_id_b', 'assay_id_a', 'assay_id_b',
                'removed_atoms_A', 'added_atoms_B', 'attach_atoms_A', 'mapped_pairs'
            ])

            # Process each core in the chunk
            total_cores = len(chunk['cores'])
            for core_idx, core in enumerate(chunk['cores']):
                chembl_ids = core_index[core]

                # Skip small cores
                if len(chembl_ids) < 2:
                    continue

                # Log progress every 10% of cores (commented out to reduce output)
                # if core_idx % max(1, total_cores // 10) == 0:
                #     progress_pct = (core_idx / total_cores) * 100
                #     print(f"[Chunk {chunk_id}] Progress: {progress_pct:.0f}% ({core_idx}/{total_cores} cores, {pairs_written} pairs written)", flush=True)

                # Extract all pairs from this core
                for i in range(len(chembl_ids)):
                    for j in range(i + 1, len(chembl_ids)):
                        chembl_id_a = chembl_ids[i]
                        chembl_id_b = chembl_ids[j]

                        # MW filter
                        mw_a = mw_lookup.get(chembl_id_a)
                        mw_b = mw_lookup.get(chembl_id_b)
                        if mw_a is None or mw_b is None:
                            continue
                        if abs(mw_a - mw_b) > max_mw_delta:
                            continue

                        # Get properties for both molecules
                        data_a = property_lookup.get(chembl_id_a, {})
                        data_b = property_lookup.get(chembl_id_b, {})

                        props_a = data_a.get('properties', {})
                        props_b = data_b.get('properties', {})

                        # Find common properties
                        common_properties = set(props_a.keys()) & set(props_b.keys())

                        if not common_properties:
                            continue

                        # Get fragments for this core
                        frags_a = fragments.get(chembl_id_a, {})
                        frags_b = fragments.get(chembl_id_b, {})

                        # Check if both molecules have this core
                        if core not in frags_a or core not in frags_b:
                            continue

                        chains_a = frags_a[core]
                        chains_b = frags_b[core]

                        # Extract pair for each common property
                        for composite_key in common_properties:
                            # Get property values
                            prop_a = props_a[composite_key]
                            prop_b = props_b[composite_key]

                            # Handle both dict (bioactivity) and scalar (computed) properties
                            if isinstance(prop_a, dict):
                                value_a = prop_a['value']
                                # Extract original property_name from dict (for composite keys)
                                property_name = prop_a.get('property_name', composite_key)
                                target_name = prop_a.get('target_name', '')
                                target_chembl_id = prop_a.get('target_chembl_id', '')
                                doc_id_a = prop_a.get('doc_id', '')
                                assay_id_a = prop_a.get('assay_id', '')
                            else:
                                value_a = prop_a
                                property_name = composite_key  # Computed properties use simple keys
                                target_name = ''
                                target_chembl_id = ''
                                doc_id_a = ''
                                assay_id_a = ''

                            # Skip if property not in filter (check AFTER extracting property_name)
                            if property_filter and property_name not in property_filter:
                                continue

                            if isinstance(prop_b, dict):
                                value_b = prop_b['value']
                                doc_id_b = prop_b.get('doc_id', '')
                                assay_id_b = prop_b.get('assay_id', '')
                            else:
                                value_b = prop_b
                                doc_id_b = ''
                                assay_id_b = ''

                            # Calculate delta
                            try:
                                delta = float(value_b) - float(value_a)
                            except (ValueError, TypeError):
                                delta = None

                            # Create edit SMILES (just use chains_b for now - simplified)
                            edit_smiles = chains_b

                            # Count cuts (number of attachment points in chains)
                            num_cuts = chains_a.count('[*:')

                            # Create row
                            row = {
                                'mol_a': chembl_id_a,
                                'mol_b': chembl_id_b,
                                'edit_smiles': edit_smiles,
                                'num_cuts': num_cuts,
                                'property_name': property_name,  # Use extracted property_name
                                'value_a': value_a,
                                'value_b': value_b,
                                'delta': delta,
                                'target_name': target_name,
                                'target_chembl_id': target_chembl_id,
                                'doc_id_a': doc_id_a,
                                'doc_id_b': doc_id_b,
                                'assay_id_a': assay_id_a,
                                'assay_id_b': assay_id_b,
                                'removed_atoms_A': '',  # Simplified - not computing atom mapping in parallel
                                'added_atoms_B': '',
                                'attach_atoms_A': '',
                                'mapped_pairs': ''
                            }

                            pair_buffer.append(row)
                            pairs_written += 1

                            # Flush buffer if needed (write to CSV but don't checkpoint every time)
                            if len(pair_buffer) >= micro_batch_size:
                                for pair in pair_buffer:
                                    csv_writer.writerow([
                                        pair['mol_a'], pair['mol_b'], pair['edit_smiles'],
                                        pair['num_cuts'], pair['property_name'], pair['value_a'],
                                        pair['value_b'], pair['delta'], pair['target_name'],
                                        pair['target_chembl_id'], pair['doc_id_a'], pair['doc_id_b'],
                                        pair['assay_id_a'], pair['assay_id_b'], pair['removed_atoms_A'],
                                        pair['added_atoms_B'], pair['attach_atoms_A'], pair['mapped_pairs']
                                    ])
                                pair_buffer = []

                cores_processed += 1

            # Flush remaining pairs
            if pair_buffer:
                for pair in pair_buffer:
                    csv_writer.writerow([
                        pair['mol_a'], pair['mol_b'], pair['edit_smiles'],
                        pair['num_cuts'], pair['property_name'], pair['value_a'],
                        pair['value_b'], pair['delta'], pair['target_name'],
                        pair['target_chembl_id'], pair['doc_id_a'], pair['doc_id_b'],
                        pair['assay_id_a'], pair['assay_id_b'], pair['removed_atoms_A'],
                        pair['added_atoms_B'], pair['attach_atoms_A'], pair['mapped_pairs']
                    ])

        # Mark as complete
        with open(checkpoint_file, 'wb') as cp:
            pickle.dump({
                'chunk_id': chunk_id,
                'pairs_written': pairs_written,
                'cores_processed': cores_processed,
                'completed': True
            }, cp)

        # print(f"[Chunk {chunk_id}] Complete: {pairs_written} pairs written from {cores_processed} cores", flush=True)
        return str(output_file)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract pairs in long format")
    parser.add_argument('--molecules-file', required=True,
                       help='Molecules CSV file')
    parser.add_argument('--bioactivity-file', required=True,
                       help='Bioactivity CSV file (long format)')
    parser.add_argument('--output', default='data/pairs/chembl_pairs_long.csv',
                       help='Output file')
    parser.add_argument('--max-mw-delta', type=float, default=200,
                       help='Max MW delta (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                       help='Min similarity (default: 0.4)')
    parser.add_argument('--max-cuts', type=int, default=2,
                       help='Max cuts (default: 2)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Load data
    logger.info(f"Loading molecules from {args.molecules_file}...")
    molecules_df = pd.read_csv(args.molecules_file)

    logger.info(f"Loading bioactivity from {args.bioactivity_file}...")
    bioactivity_df = pd.read_csv(args.bioactivity_file)

    # Extract pairs
    extractor = LongFormatMMPExtractor(max_cuts=args.max_cuts)

    df_pairs = extractor.extract_pairs_long_format(
        molecules_df=molecules_df,
        bioactivity_df=bioactivity_df,
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(output_path, index=False)

    logger.info(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
