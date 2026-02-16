"""
Base interface for structured edit embedders.

Defines a common interface for extracting local edit representations from
different molecule embedding methods (ChemBERTa-2, Graphormer, MolFM, D-MPNN).

The key insight is that different models provide different granularities:
- D-MPNN/Graphormer: Atom-level embeddings → extract by atom indices
- ChemBERTa-2/MolFM: Token-level embeddings → map tokens to atom indices

All implementations should provide:
1. Local environment extraction (k-hop neighborhood around edit site)
2. Fragment embeddings (leaving/incoming fragments)
3. Global context embedding (full molecule)
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem


class StructuredEditEmbedderBase(nn.Module, ABC):
    """
    Abstract base class for structured edit embedders.

    All structured edit embedders should:
    1. Extract local embeddings around edit sites
    2. Compute fragment embeddings (leaving/incoming)
    3. Provide global context
    4. Support both frozen and trainable modes

    Args:
        embedding_dim: Dimension of the underlying embeddings
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether the embedder is trainable (default: False)
    """

    def __init__(
        self,
        embedding_dim: int,
        k_hop_env: int = 2,
        trainable: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k_hop_env = k_hop_env
        self.trainable = trainable

    @abstractmethod
    def get_atom_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level embeddings for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - atom_embeddings: [n_atoms, embedding_dim]
            - global_embedding: [embedding_dim]
        """
        pass

    @abstractmethod
    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        Get mapping from token indices to atom indices.

        Only relevant for sequence-based models (ChemBERTa, MolFM).
        For graph-based models, returns None.

        Args:
            smiles: SMILES string
            mol: Optional RDKit molecule (to avoid re-parsing)

        Returns:
            Dict mapping token_idx → list of atom indices, or None
        """
        pass

    def compute_k_hop_neighbors(
        self,
        mol: Chem.Mol,
        seed_indices: List[int]
    ) -> List[int]:
        """
        Compute k-hop neighborhood of seed atoms.

        Args:
            mol: RDKit molecule
            seed_indices: Starting atom indices

        Returns:
            List of atom indices in k-hop neighborhood (including seeds)
        """
        if not seed_indices:
            return []

        env_indices = set(seed_indices)

        for _ in range(self.k_hop_env):
            new_indices = []
            for idx in env_indices:
                if idx < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(idx)
                    for neighbor in atom.GetNeighbors():
                        new_indices.append(neighbor.GetIdx())
            env_indices.update(new_indices)

        return sorted(list(env_indices))

    def get_local_environment_embedding(
        self,
        atom_embeddings: torch.Tensor,
        mol: Chem.Mol,
        seed_indices: List[int],
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Extract local environment embedding around seed atoms.

        Args:
            atom_embeddings: [n_atoms, embedding_dim]
            mol: RDKit molecule
            seed_indices: Indices of atoms at edit site (attachment points)
            aggregation: How to aggregate ('mean', 'sum', 'max')

        Returns:
            Local environment embedding [embedding_dim]
        """
        device = atom_embeddings.device

        if not seed_indices:
            return torch.zeros(self.embedding_dim, device=device)

        # Get k-hop neighborhood
        env_indices = self.compute_k_hop_neighbors(mol, seed_indices)

        # Filter valid indices
        valid_indices = [i for i in env_indices if i < atom_embeddings.shape[0]]

        if not valid_indices:
            return torch.zeros(self.embedding_dim, device=device)

        # Extract embeddings
        env_embeddings = atom_embeddings[valid_indices]  # [n_env, embedding_dim]

        # Aggregate
        if aggregation == 'mean':
            return env_embeddings.mean(dim=0)
        elif aggregation == 'sum':
            return env_embeddings.sum(dim=0)
        elif aggregation == 'max':
            return env_embeddings.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_fragment_embedding(
        self,
        atom_embeddings: torch.Tensor,
        fragment_indices: List[int],
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Extract embedding for a molecular fragment.

        Args:
            atom_embeddings: [n_atoms, embedding_dim]
            fragment_indices: Indices of atoms in fragment
            aggregation: How to aggregate ('mean', 'sum', 'max')

        Returns:
            Fragment embedding [embedding_dim]
        """
        device = atom_embeddings.device

        if not fragment_indices:
            return torch.zeros(self.embedding_dim, device=device)

        # Filter valid indices
        valid_indices = [i for i in fragment_indices if i < atom_embeddings.shape[0]]

        if not valid_indices:
            return torch.zeros(self.embedding_dim, device=device)

        # Extract embeddings
        frag_embeddings = atom_embeddings[valid_indices]  # [n_frag, embedding_dim]

        # Aggregate
        if aggregation == 'mean':
            return frag_embeddings.mean(dim=0)
        elif aggregation == 'sum':
            return frag_embeddings.sum(dim=0)
        elif aggregation == 'max':
            return frag_embeddings.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def forward(
        self,
        smiles_a: str,
        smiles_b: str,
        removed_atom_indices_a: List[int],
        added_atom_indices_b: List[int],
        attach_atom_indices_a: List[int],
        attach_atom_indices_b: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute structured edit embedding for a molecular pair.

        Args:
            smiles_a: SMILES of molecule A (before edit)
            smiles_b: SMILES of molecule B (after edit)
            removed_atom_indices_a: Indices of atoms removed from A
            added_atom_indices_b: Indices of atoms added in B
            attach_atom_indices_a: Attachment points in A
            attach_atom_indices_b: Attachment points in B (optional)

        Returns:
            Dictionary with:
            - 'h_a_global': Global embedding of A [embedding_dim]
            - 'h_b_global': Global embedding of B [embedding_dim]
            - 'h_env_a': Local environment in A [embedding_dim]
            - 'h_env_b': Local environment in B [embedding_dim] (if attach_b provided)
            - 'h_out': Leaving fragment embedding [embedding_dim]
            - 'h_in': Incoming fragment embedding [embedding_dim]
        """
        # Get atom embeddings for both molecules
        H_a, h_a_global = self.get_atom_embeddings(smiles_a)
        H_b, h_b_global = self.get_atom_embeddings(smiles_b)

        device = H_a.device

        # Parse molecules
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            # Return zeros if parsing fails
            zero = torch.zeros(self.embedding_dim, device=device)
            return {
                'h_a_global': h_a_global if mol_a else zero,
                'h_b_global': h_b_global if mol_b else zero,
                'h_env_a': zero,
                'h_out': zero,
                'h_in': zero,
            }

        # Local environment in A (around attachment points)
        h_env_a = self.get_local_environment_embedding(H_a, mol_a, attach_atom_indices_a)

        # Leaving fragment embedding
        h_out = self.get_fragment_embedding(H_a, removed_atom_indices_a)

        # Incoming fragment embedding
        h_in = self.get_fragment_embedding(H_b, added_atom_indices_b)

        result = {
            'h_a_global': h_a_global,
            'h_b_global': h_b_global,
            'h_env_a': h_env_a,
            'h_out': h_out,
            'h_in': h_in,
        }

        # Optional: environment in B
        if attach_atom_indices_b:
            h_env_b = self.get_local_environment_embedding(H_b, mol_b, attach_atom_indices_b)
            result['h_env_b'] = h_env_b

        return result

    def forward_batch(
        self,
        smiles_a_list: List[str],
        smiles_b_list: List[str],
        removed_atoms_list: List[List[int]],
        added_atoms_list: List[List[int]],
        attach_atoms_a_list: List[List[int]],
        attach_atoms_b_list: Optional[List[List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Batched forward pass.

        Returns:
            Dictionary with batched tensors [batch_size, embedding_dim]
        """
        batch_size = len(smiles_a_list)

        if attach_atoms_b_list is None:
            attach_atoms_b_list = [None] * batch_size

        # Process each sample
        results = []
        for i in range(batch_size):
            result = self.forward(
                smiles_a=smiles_a_list[i],
                smiles_b=smiles_b_list[i],
                removed_atom_indices_a=removed_atoms_list[i],
                added_atom_indices_b=added_atoms_list[i],
                attach_atom_indices_a=attach_atoms_a_list[i],
                attach_atom_indices_b=attach_atoms_b_list[i]
            )
            results.append(result)

        # Stack into batched tensors
        batched = {}
        for key in results[0].keys():
            batched[key] = torch.stack([r[key] for r in results], dim=0)

        return batched

    def freeze(self):
        """Freeze embedder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.trainable = False

    def unfreeze(self):
        """Unfreeze embedder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.trainable = True

    @property
    def output_dim(self) -> int:
        """Total output dimension when all components are concatenated."""
        # h_a_global + h_env_a + h_out + h_in = 4 * embedding_dim
        return 4 * self.embedding_dim

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this embedder."""
        pass
