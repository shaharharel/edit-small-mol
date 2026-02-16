"""
Structured MMP Dataset for training with StructuredEditEmbedder.

This dataset provides the atom-level embeddings and MMP structural
information required by the StructuredEditEmbedder.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem

from .mmp_parser import parse_mmp_info


class StructuredMMPDataset(Dataset):
    """
    Dataset for structured MMP edit prediction.

    Provides atom-level embeddings and MMP structural information
    for each molecular pair.

    The dataset stores:
    - Atom embeddings (H_A, H_B) as variable-length arrays
    - Molecule embeddings (h_A_global, h_B_global)
    - MMP structural info (removed atoms, added atoms, attachment points, mapping)
    - Target delta values

    Args:
        df: DataFrame with MMP data. Required columns:
            - mol_a, mol_b: SMILES strings
            - removed_atoms_A, added_atoms_B, attach_atoms_A, mapped_pairs: MMP info
            - property_name: Property being predicted
            - delta: Target value (property change)
        atom_embeddings: Dict mapping SMILES -> (H, h_global) where
            H is [n_atoms, gnn_dim] and h_global is [gnn_dim]
        task_names: List of task/property names for multi-task learning
        task_col: Column name for task/property identifier (default: 'property_name')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        atom_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
        task_names: List[str],
        task_col: str = 'property_name'
    ):
        self.df = df.reset_index(drop=True)
        self.atom_embeddings = atom_embeddings
        self.task_names = task_names
        self.task_col = task_col
        self.n_tasks = len(task_names)

        # Create task name to index mapping
        self.task_to_idx = {name: i for i, name in enumerate(task_names)}

        # Pre-parse MMP info for efficiency
        self._mmp_info_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_mmp_info(self, idx: int) -> Dict:
        """Get parsed MMP info for sample, with caching."""
        if idx not in self._mmp_info_cache:
            row = self.df.iloc[idx]
            self._mmp_info_cache[idx] = parse_mmp_info(row)
        return self._mmp_info_cache[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with:
            - 'H_A': Atom embeddings for molecule A [n_atoms_A, gnn_dim]
            - 'H_B': Atom embeddings for molecule B [n_atoms_B, gnn_dim]
            - 'h_A_global': Global embedding for A [gnn_dim]
            - 'h_B_global': Global embedding for B [gnn_dim]
            - 'mol_a_smiles': SMILES of molecule A
            - 'mol_b_smiles': SMILES of molecule B
            - 'mmp_info': Dict with structural info
            - 'delta': Target value (multi-task format)
            - 'task_idx': Index of the task this sample belongs to
        """
        row = self.df.iloc[idx]

        # Get SMILES
        mol_a_smiles = row['mol_a']
        mol_b_smiles = row['mol_b']

        # Get embeddings
        H_A, h_A_global = self.atom_embeddings[mol_a_smiles]
        H_B, h_B_global = self.atom_embeddings[mol_b_smiles]

        # Get MMP info
        mmp_info = self._get_mmp_info(idx)

        # Get task index and delta
        task_name = row[self.task_col]
        task_idx = self.task_to_idx.get(task_name, -1)

        # Create multi-task delta (NaN for other tasks)
        delta = np.full(self.n_tasks, np.nan, dtype=np.float32)
        if task_idx >= 0:
            delta[task_idx] = row['delta']

        return {
            'H_A': torch.from_numpy(H_A).float(),
            'H_B': torch.from_numpy(H_B).float(),
            'h_A_global': torch.from_numpy(h_A_global).float(),
            'h_B_global': torch.from_numpy(h_B_global).float(),
            'mol_a_smiles': mol_a_smiles,
            'mol_b_smiles': mol_b_smiles,
            'mmp_info': mmp_info,
            'delta': torch.from_numpy(delta).float(),
            'task_idx': task_idx
        }


def structured_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for StructuredMMPDataset.

    Handles variable-length atom embeddings by keeping them as lists.

    Args:
        batch: List of sample dicts from __getitem__

    Returns:
        Batched dict with:
        - 'H_A_list': List of atom embedding tensors
        - 'H_B_list': List of atom embedding tensors
        - 'h_A_global': Stacked global embeddings [batch_size, gnn_dim]
        - 'h_B_global': Stacked global embeddings [batch_size, gnn_dim]
        - 'mol_A_list': List of RDKit Mol objects
        - 'mol_B_list': List of RDKit Mol objects
        - 'removed_atoms_list': List of index lists
        - 'added_atoms_list': List of index lists
        - 'attach_atoms_list': List of index lists
        - 'mapped_pairs_list': List of pair lists
        - 'delta': Stacked delta tensor [batch_size, n_tasks]
    """
    # Variable-length: keep as lists
    H_A_list = [sample['H_A'] for sample in batch]
    H_B_list = [sample['H_B'] for sample in batch]

    # Fixed-size: stack into tensors
    h_A_global = torch.stack([sample['h_A_global'] for sample in batch])
    h_B_global = torch.stack([sample['h_B_global'] for sample in batch])
    delta = torch.stack([sample['delta'] for sample in batch])

    # Create RDKit Mol objects from SMILES
    mol_A_list = [Chem.MolFromSmiles(sample['mol_a_smiles']) for sample in batch]
    mol_B_list = [Chem.MolFromSmiles(sample['mol_b_smiles']) for sample in batch]

    # Extract MMP info lists
    removed_atoms_list = [sample['mmp_info']['removed_atom_indices_A'] for sample in batch]
    added_atoms_list = [sample['mmp_info']['added_atom_indices_B'] for sample in batch]
    attach_atoms_list = [sample['mmp_info']['attach_atom_indices_A'] for sample in batch]
    mapped_pairs_list = [sample['mmp_info']['mapped_atom_pairs'] for sample in batch]

    return {
        'H_A_list': H_A_list,
        'H_B_list': H_B_list,
        'h_A_global': h_A_global,
        'h_B_global': h_B_global,
        'mol_A_list': mol_A_list,
        'mol_B_list': mol_B_list,
        'removed_atoms_list': removed_atoms_list,
        'added_atoms_list': added_atoms_list,
        'attach_atoms_list': attach_atoms_list,
        'mapped_pairs_list': mapped_pairs_list,
        'delta': delta
    }


def prepare_structured_embeddings(
    df: pd.DataFrame,
    embedder,
    cache_dir: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare atom-level embeddings for all molecules in DataFrame.

    Args:
        df: DataFrame with 'mol_a' and 'mol_b' columns
        embedder: ChemPropEmbedder with encode_with_atom_embeddings method
        cache_dir: Optional directory for caching (not implemented yet)

    Returns:
        Dict mapping SMILES -> (H, h_global) tuples
    """
    # Collect unique SMILES
    all_smiles = set(df['mol_a'].unique()) | set(df['mol_b'].unique())
    print(f"Computing atom embeddings for {len(all_smiles)} unique molecules...")

    embeddings = {}
    for i, smiles in enumerate(all_smiles):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(all_smiles)}")

        try:
            H, h_global = embedder.encode_with_atom_embeddings(smiles)
            embeddings[smiles] = (H, h_global)
        except Exception as e:
            print(f"  Warning: Failed to embed {smiles}: {e}")
            # Fallback: zeros
            mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms() if mol else 1
            embeddings[smiles] = (
                np.zeros((n_atoms, embedder.embedding_dim), dtype=np.float32),
                np.zeros(embedder.embedding_dim, dtype=np.float32)
            )

    print(f"  Done! Embedded {len(embeddings)} molecules.")
    return embeddings
