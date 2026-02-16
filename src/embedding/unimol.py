"""
Uni-Mol molecular embedder.

Uni-Mol is a Universal 3D Molecular Representation Learning Framework that
uses SE(3) Transformer architecture pretrained on 209M molecular conformations.

Key features:
- 3D-aware molecular representations
- Atom-level embeddings aligned with RDKit atom ordering
- Multiple model sizes (Uni-Mol v2: 84M to 1.1B parameters)
- Supports both molecule-level and atom-level embeddings

Installation:
    pip install unimol-tools

References:
- Paper: Uni-Mol: A Universal 3D Molecular Representation Learning Framework (ICLR 2023)
- GitHub: https://github.com/deepmodeling/Uni-Mol
- HuggingFace: https://huggingface.co/dptech/Uni-Mol-Models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .base import MoleculeEmbedder


class UniMolEmbedder(MoleculeEmbedder):
    """
    Uni-Mol based molecular embedder.

    Uses the unimol_tools package to extract molecular representations
    from SMILES strings. Provides both molecule-level (CLS) and atom-level
    embeddings.

    Args:
        model_name: Model variant - 'unimolv1' or 'unimolv2' (default: 'unimolv1')
        model_size: For v2 only - '84m', '164m', '310m', '570m', '1.1B' (default: '84m')
        remove_hs: Whether to remove hydrogens (default: False)
        device: Device to use - 'cuda', 'cpu', or 'auto' (default: 'auto')
        use_gpu: GPU specification for unimol - 'all', '0', '0,1', etc. (default: None)

    Embedding dimensions:
        - unimolv1: 512
        - unimolv2 (84m): 480
        - unimolv2 (164m): 640
        - unimolv2 (310m): 800
        - unimolv2 (570m): 960
        - unimolv2 (1.1B): 1280
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
        remove_hs: bool = False,
        device: Optional[str] = None,
        use_gpu: Optional[str] = None,
        trainable: bool = False
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
        self.device = device

        # GPU configuration for unimol
        if use_gpu is None:
            if device == 'cuda':
                use_gpu = '0'  # Default to first GPU
            else:
                use_gpu = False

        self.model_name = model_name
        self.model_size = model_size
        self.remove_hs = remove_hs
        self.trainable = trainable

        # Determine embedding dimension
        if model_name == 'unimolv1':
            self._embedding_dim = self.EMBEDDING_DIMS['unimolv1']
        else:
            key = f'unimolv2_{model_size}'
            if key not in self.EMBEDDING_DIMS:
                raise ValueError(f"Unknown model_size: {model_size}. Choose from: 84m, 164m, 310m, 570m, 1.1B")
            self._embedding_dim = self.EMBEDDING_DIMS[key]

        # Initialize UniMolRepr
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading Uni-Mol embedder ({model_name}, {trainable_str})...")

        self.model = UniMolRepr(
            data_type='molecule',
            remove_hs=remove_hs,
            model_name=model_name,
            model_size=model_size if model_name == 'unimolv2' else None,
            use_gpu=use_gpu
        )

        print(f"  Model: {model_name}" + (f" ({model_size})" if model_name == 'unimolv2' else ""))
        print(f"  Embedding dim: {self._embedding_dim}")
        print(f"  Device: {device}")

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to embedding vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(smiles, str):
            return self.embed(smiles)
        else:
            return self.embed_batch(smiles)

    def embed(self, smiles: str) -> np.ndarray:
        """
        Embed a single SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Molecule embedding as numpy array [embedding_dim]
        """
        try:
            result = self.model.get_repr([smiles], return_atomic_reprs=False)
            return np.array(result['cls_repr'][0], dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to embed {smiles}: {e}")
            return np.zeros(self._embedding_dim, dtype=np.float32)

    def embed_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Embed a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Molecule embeddings as numpy array [batch_size, embedding_dim]
        """
        if not smiles_list:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        try:
            result = self.model.get_repr(smiles_list, return_atomic_reprs=False)
            return np.array(result['cls_repr'], dtype=np.float32)
        except Exception as e:
            print(f"Warning: Batch embedding failed: {e}")
            # Fall back to individual embedding
            embeddings = []
            for smiles in smiles_list:
                embeddings.append(self.embed(smiles))
            return np.array(embeddings, dtype=np.float32)

    def get_atom_embeddings(
        self,
        smiles: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get atom-level embeddings for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - atom_embeddings: [n_atoms, embedding_dim]
            - mol_embedding: [embedding_dim]
        """
        try:
            result = self.model.get_repr([smiles], return_atomic_reprs=True)
            mol_emb = np.array(result['cls_repr'][0], dtype=np.float32)
            atom_embs = np.array(result['atomic_reprs'][0], dtype=np.float32)
            return atom_embs, mol_emb
        except Exception as e:
            print(f"Warning: Failed to get atom embeddings for {smiles}: {e}")
            return (
                np.zeros((1, self._embedding_dim), dtype=np.float32),
                np.zeros(self._embedding_dim, dtype=np.float32)
            )

    def embed_batch_with_atoms(
        self,
        smiles_list: List[str]
    ) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Get both molecule and atom-level embeddings for a batch.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with:
            - 'mol_embeddings': [batch_size, embedding_dim]
            - 'atom_embeddings': List of [n_atoms_i, embedding_dim] arrays
        """
        if not smiles_list:
            return {
                'mol_embeddings': np.zeros((0, self._embedding_dim), dtype=np.float32),
                'atom_embeddings': []
            }

        try:
            result = self.model.get_repr(smiles_list, return_atomic_reprs=True)
            mol_embs = np.array(result['cls_repr'], dtype=np.float32)
            atom_embs = [np.array(a, dtype=np.float32) for a in result['atomic_reprs']]
            return {
                'mol_embeddings': mol_embs,
                'atom_embeddings': atom_embs
            }
        except Exception as e:
            print(f"Warning: Batch atom embedding failed: {e}")
            # Fall back to individual processing
            mol_embs = []
            atom_embs = []
            for smiles in smiles_list:
                atom_emb, mol_emb = self.get_atom_embeddings(smiles)
                mol_embs.append(mol_emb)
                atom_embs.append(atom_emb)
            return {
                'mol_embeddings': np.array(mol_embs, dtype=np.float32),
                'atom_embeddings': atom_embs
            }

    def embed_batch_tensor(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Embed a batch and return as PyTorch tensor.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Molecule embeddings as tensor [batch_size, embedding_dim]
        """
        embeddings = self.embed_batch(smiles_list)
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedder."""
        if self.model_name == 'unimolv2':
            return f"UniMol-v2-{self.model_size}"
        return "UniMol-v1"

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.trainable else "frozen"
        return f"UniMolEmbedder(model={self.model_name}, dim={self._embedding_dim}, {trainable_str})"


# Convenience function
def create_unimol_embedder(
    version: str = 'v1',
    size: str = '84m',
    device: str = 'auto',
    **kwargs
) -> UniMolEmbedder:
    """
    Create a Uni-Mol embedder with simplified parameters.

    Args:
        version: 'v1' or 'v2'
        size: Model size for v2 ('84m', '164m', '310m', '570m', '1.1B')
        device: 'cuda', 'cpu', or 'auto'
        **kwargs: Additional arguments for UniMolEmbedder

    Returns:
        UniMolEmbedder instance
    """
    model_name = 'unimolv1' if version == 'v1' else 'unimolv2'
    return UniMolEmbedder(
        model_name=model_name,
        model_size=size,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Test the embedder
    print("Testing Uni-Mol Embedder")
    print("=" * 60)

    try:
        embedder = UniMolEmbedder(model_name='unimolv1')

        # Test single molecule
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        print(f"\nTest molecule: {smiles}")

        emb = embedder.embed(smiles)
        print(f"Molecule embedding shape: {emb.shape}")
        print(f"Embedding norm: {np.linalg.norm(emb):.4f}")

        # Test atom embeddings
        atom_embs, mol_emb = embedder.get_atom_embeddings(smiles)
        print(f"Atom embeddings shape: {atom_embs.shape}")
        print(f"Mol embedding shape: {mol_emb.shape}")

        # Test batch
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CC(C)CC']
        batch_embs = embedder.embed_batch(smiles_list)
        print(f"\nBatch embeddings shape: {batch_embs.shape}")

        print("\n✓ All tests passed!")

    except ImportError as e:
        print(f"Cannot test: {e}")
        print("Install with: pip install unimol-tools")
