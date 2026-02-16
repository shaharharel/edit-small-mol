"""
Uni-Mol structured edit embedder.

Uses Uni-Mol's 3D molecular representations for structured edit embedding.
Uni-Mol provides atom-level embeddings that align with RDKit atom ordering,
making it suitable for structured molecular pair analysis.

Key features:
- 3D-aware atom-level embeddings
- Direct atom index mapping (aligned with RDKit)
- Supports k-hop neighborhood extraction
- Multiple model sizes available

Installation:
    pip install unimol-tools

References:
- Paper: Uni-Mol: A Universal 3D Molecular Representation Learning Framework (ICLR 2023)
- GitHub: https://github.com/deepmodeling/Uni-Mol
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem

from .structured_edit_base import StructuredEditEmbedderBase


class UniMolStructuredEditEmbedder(StructuredEditEmbedderBase):
    """
    Uni-Mol based structured edit embedder.

    Uses Uni-Mol's atom-level embeddings to extract local edit representations
    around molecular edit sites. Since Uni-Mol atom embeddings are aligned with
    RDKit atom ordering, no token-to-atom mapping is needed.

    Args:
        model_name: Model variant - 'unimolv1' or 'unimolv2' (default: 'unimolv1')
        model_size: For v2 only - '84m', '164m', '310m', '570m', '1.1B' (default: '84m')
        device: Device to use - 'cuda', 'cpu', or 'auto' (default: 'auto')
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether to enable gradient updates (default: False)
        remove_hs: Whether to remove hydrogens (default: False)
    """

    # Embedding dimensions for each model variant
    EMBEDDING_DIMS = {
        'unimolv1': 512,
        'unimolv2_84m': 480,
        'unimolv2_164m': 640,
        'unimolv2_310m': 800,
        'unimolv2_570m': 960,
        'unimolv2_1.1B': 1280,
    }

    def __init__(
        self,
        model_name: str = 'unimolv1',
        model_size: str = '84m',
        device: Optional[str] = None,
        k_hop_env: int = 2,
        trainable: bool = False,
        remove_hs: bool = False
    ):
        try:
            from unimol_tools import UniMolRepr
        except ImportError:
            raise ImportError(
                "unimol_tools not installed. Install with: pip install unimol-tools"
            )

        # Determine device
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Determine embedding dimension
        if model_name == 'unimolv1':
            embedding_dim = self.EMBEDDING_DIMS['unimolv1']
        else:
            key = f'unimolv2_{model_size}'
            if key not in self.EMBEDDING_DIMS:
                raise ValueError(f"Unknown model_size: {model_size}. Choose from: 84m, 164m, 310m, 570m, 1.1B")
            embedding_dim = self.EMBEDDING_DIMS[key]

        # Initialize base class
        super().__init__(
            embedding_dim=embedding_dim,
            k_hop_env=k_hop_env,
            trainable=trainable
        )

        self.model_name = model_name
        self.model_size = model_size
        self.device = device
        self.remove_hs = remove_hs

        # GPU configuration for unimol
        if device == 'cuda':
            use_gpu = '0'
        else:
            use_gpu = False

        # Initialize UniMolRepr
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading Uni-Mol structured embedder ({model_name}, {trainable_str})...")

        self.model = UniMolRepr(
            data_type='molecule',
            remove_hs=remove_hs,
            model_name=model_name,
            model_size=model_size if model_name == 'unimolv2' else None,
            use_gpu=use_gpu
        )

        print(f"  Model: {model_name}" + (f" ({model_size})" if model_name == 'unimolv2' else ""))
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  k-hop env: {k_hop_env}")

        # Cache for embeddings (optional optimization)
        self._cache = {}
        self._cache_enabled = True
        self._max_cache_size = 10000

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
        # Check cache
        if self._cache_enabled and smiles in self._cache:
            return self._cache[smiles]

        try:
            result = self.model.get_repr([smiles], return_atomic_reprs=True)

            mol_emb = torch.tensor(
                result['cls_repr'][0],
                dtype=torch.float32,
                device=self.device
            )
            atom_embs = torch.tensor(
                result['atomic_reprs'][0],
                dtype=torch.float32,
                device=self.device
            )

            # Cache result
            if self._cache_enabled and len(self._cache) < self._max_cache_size:
                self._cache[smiles] = (atom_embs, mol_emb)

            return atom_embs, mol_emb

        except Exception as e:
            print(f"Warning: Failed to get embeddings for {smiles}: {e}")
            zero_atom = torch.zeros(1, self.embedding_dim, device=self.device)
            zero_mol = torch.zeros(self.embedding_dim, device=self.device)
            return zero_atom, zero_mol

    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        Uni-Mol atom embeddings are already aligned with RDKit atom ordering.
        No token-to-atom mapping needed.

        Returns:
            None (not applicable for Uni-Mol)
        """
        return None

    def embed_batch(
        self,
        smiles_list: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Efficient batch embedding of multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of:
            - List of atom embeddings (one per molecule)
            - Global embeddings [batch_size, embedding_dim]
        """
        # Check which are in cache
        cached_indices = []
        uncached_indices = []
        uncached_smiles = []

        for i, smiles in enumerate(smiles_list):
            if self._cache_enabled and smiles in self._cache:
                cached_indices.append(i)
            else:
                uncached_indices.append(i)
                uncached_smiles.append(smiles)

        # Process uncached molecules
        if uncached_smiles:
            try:
                result = self.model.get_repr(uncached_smiles, return_atomic_reprs=True)

                for j, smiles in enumerate(uncached_smiles):
                    mol_emb = torch.tensor(
                        result['cls_repr'][j],
                        dtype=torch.float32,
                        device=self.device
                    )
                    atom_embs = torch.tensor(
                        result['atomic_reprs'][j],
                        dtype=torch.float32,
                        device=self.device
                    )

                    if self._cache_enabled and len(self._cache) < self._max_cache_size:
                        self._cache[smiles] = (atom_embs, mol_emb)

            except Exception as e:
                print(f"Warning: Batch embedding failed: {e}")
                # Process individually
                for smiles in uncached_smiles:
                    self.get_atom_embeddings(smiles)

        # Collect all results
        atom_embs_list = []
        global_embs_list = []

        for smiles in smiles_list:
            if smiles in self._cache:
                atom_embs, mol_emb = self._cache[smiles]
            else:
                atom_embs, mol_emb = self.get_atom_embeddings(smiles)
            atom_embs_list.append(atom_embs)
            global_embs_list.append(mol_emb)

        global_embs = torch.stack(global_embs_list, dim=0)
        return atom_embs_list, global_embs

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def disable_cache(self):
        """Disable embedding cache."""
        self._cache_enabled = False
        self._cache.clear()

    def enable_cache(self, max_size: int = 10000):
        """Enable embedding cache."""
        self._cache_enabled = True
        self._max_cache_size = max_size

    @property
    def name(self) -> str:
        """Return the name of this embedder."""
        if self.model_name == 'unimolv2':
            return f"UniMol-v2-{self.model_size}-Structured"
        return "UniMol-v1-Structured"

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.trainable else "frozen"
        return f"UniMolStructuredEditEmbedder(model={self.model_name}, dim={self.embedding_dim}, {trainable_str})"


# Convenience function
def unimol_structured_embedder(
    version: str = 'v1',
    size: str = '84m',
    device: str = 'auto',
    k_hop: int = 2,
    **kwargs
) -> UniMolStructuredEditEmbedder:
    """
    Create a Uni-Mol structured edit embedder with simplified parameters.

    Args:
        version: 'v1' or 'v2'
        size: Model size for v2 ('84m', '164m', '310m', '570m', '1.1B')
        device: 'cuda', 'cpu', or 'auto'
        k_hop: k-hop neighborhood size
        **kwargs: Additional arguments

    Returns:
        UniMolStructuredEditEmbedder instance
    """
    model_name = 'unimolv1' if version == 'v1' else 'unimolv2'
    return UniMolStructuredEditEmbedder(
        model_name=model_name,
        model_size=size,
        device=device,
        k_hop_env=k_hop,
        **kwargs
    )


if __name__ == "__main__":
    # Test the embedder
    print("Testing Uni-Mol Structured Edit Embedder")
    print("=" * 60)

    try:
        embedder = UniMolStructuredEditEmbedder(model_name='unimolv1')

        # Test single molecule
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        print(f"\nTest molecule: {smiles}")

        atom_embs, global_emb = embedder.get_atom_embeddings(smiles)
        print(f"Atom embeddings shape: {atom_embs.shape}")
        print(f"Global embedding shape: {global_emb.shape}")

        # Test structured edit
        smiles_a = "CCO"
        smiles_b = "CCCO"
        print(f"\nEdit: {smiles_a} -> {smiles_b}")

        result = embedder.forward(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            removed_atom_indices_a=[],
            added_atom_indices_b=[2],
            attach_atom_indices_a=[1],
        )

        print("Edit representation shapes:")
        for key, val in result.items():
            print(f"  {key}: {val.shape}")

        print("\n✓ All tests passed!")

    except ImportError as e:
        print(f"Cannot test: {e}")
        print("Install with: pip install unimol-tools")
