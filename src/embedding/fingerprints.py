"""
Fingerprint-based molecule embeddings.

Supports:
- Morgan (ECFP) fingerprints
- RDKit topological fingerprints
- MACCS keys
- Atom pair fingerprints
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from typing import Union, List
from .base import MoleculeEmbedder


class FingerprintEmbedder(MoleculeEmbedder):
    """
    Fingerprint-based molecule embedder.

    Args:
        fp_type: Fingerprint type ('morgan', 'rdkit', 'maccs', 'atompair')
        radius: Radius for Morgan fingerprints (default: 2 for ECFP4)
        n_bits: Number of bits (default: 2048)
        use_features: Use feature-based Morgan FP (FCFP) instead of ECFP
    """

    def __init__(
        self,
        fp_type: str = 'morgan',
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False
    ):
        self.fp_type = fp_type.lower()
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features

        if self.fp_type not in ['morgan', 'rdkit', 'maccs', 'atompair']:
            raise ValueError(
                f"Invalid fp_type: {fp_type}. "
                f"Must be one of: morgan, rdkit, maccs, atompair"
            )

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to fingerprint vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Fingerprint vector(s) as numpy array (binary or count-based)
        """
        if isinstance(smiles, str):
            return self._encode_single(smiles)
        else:
            return np.array([self._encode_single(s) for s in smiles])

    def _encode_single(self, smiles: str) -> np.ndarray:
        """Encode a single SMILES to fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return zero vector for invalid SMILES
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self.fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits,
                useFeatures=self.use_features
            )
        elif self.fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        elif self.fp_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif self.fp_type == 'atompair':
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=self.n_bits
            )

        # Convert to numpy array
        arr = np.zeros(len(fp), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of fingerprints."""
        if self.fp_type == 'maccs':
            return 167  # MACCS keys have fixed size
        else:
            return self.n_bits

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        if self.fp_type == 'morgan':
            feat_str = '_feat' if self.use_features else ''
            return f"morgan_r{self.radius}{feat_str}_{self.n_bits}bit"
        else:
            return f"{self.fp_type}_{self.n_bits}bit"


# Convenience constructors
def morgan_embedder(radius: int = 2, n_bits: int = 2048) -> FingerprintEmbedder:
    """Create Morgan (ECFP) fingerprint embedder."""
    return FingerprintEmbedder(fp_type='morgan', radius=radius, n_bits=n_bits)


def maccs_embedder() -> FingerprintEmbedder:
    """Create MACCS keys embedder (167-bit)."""
    return FingerprintEmbedder(fp_type='maccs')


def rdkit_embedder(n_bits: int = 2048) -> FingerprintEmbedder:
    """Create RDKit topological fingerprint embedder."""
    return FingerprintEmbedder(fp_type='rdkit', n_bits=n_bits)
