"""
Base interface for molecule embedders.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List


class MoleculeEmbedder(ABC):
    """
    Abstract base class for molecule embedding methods.

    All embedders should implement encode() to convert SMILES to vectors.
    """

    @abstractmethod
    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to embedding vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Embedding vector(s) as numpy array
            - Single SMILES: shape (embedding_dim,)
            - List of SMILES: shape (n_molecules, embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this embedding method."""
        pass
