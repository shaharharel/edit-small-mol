"""
Edit embedding via difference of molecule embeddings.

Computes edit embeddings as:
    edit_embedding = embedding(product) - embedding(reactant)

This works with any molecule embedder (fingerprints, ChemBERTa, ChemProp, etc.)
"""

import numpy as np
from typing import Union, List, Tuple
from .base import MoleculeEmbedder


class EditEmbedder:
    """
    Edit embedder using difference of molecule embeddings with two modes.

    **Two Modes:**

    Mode 1 (use_edit_fragments=False): Full Molecule Embeddings
        edit = Embed(full_mol_b) - Embed(full_mol_a)
        - Context-aware: Different scaffolds produce different edit vectors
        - Works with SMILES pairs (mol_a, mol_b)

    Mode 2 (use_edit_fragments=True): Edit Fragment Embeddings
        edit = Embed(edit_fragment_product) - Embed(edit_fragment_reactant)
        - Scaffold-independent: Same transformation always same vector
        - Works with edit_smiles format "fragment_a>>fragment_b"
        - Better transfer learning

    This approach:
    - Works with ANY molecule embedder (fingerprints, transformers, GNNs)
    - Preserves the same dimensionality as the molecule embeddings
    - Captures the change in molecular properties
    - Can be used directly for ML without additional processing

    Args:
        molecule_embedder: Any MoleculeEmbedder instance
                          (FingerprintEmbedder, ChemBERTaEmbedder, etc.)
        use_edit_fragments: If True, use edit_smiles fragments (Mode 2)
                           If False, use full molecules (Mode 1, default)

    Example (Mode 1 - Full molecules):
        >>> from src.embedding import FingerprintEmbedder, EditEmbedder
        >>> mol_emb = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=2048)
        >>> edit_emb = EditEmbedder(mol_emb, use_edit_fragments=False)
        >>> edit_vec = edit_emb.encode_from_smiles('CCO', 'CC(=O)O')
        >>> print(edit_vec.shape)  # (2048,)

    Example (Mode 2 - Edit fragments):
        >>> edit_emb = EditEmbedder(mol_emb, use_edit_fragments=True)
        >>> edit_vec = edit_emb.encode_from_edit_smiles('[*]O>>[*]C(=O)O')
        >>> print(edit_vec.shape)  # (2048,)
    """

    def __init__(self, molecule_embedder: MoleculeEmbedder, use_edit_fragments: bool = False):
        self.molecule_embedder = molecule_embedder
        self.use_edit_fragments = use_edit_fragments

    def encode_from_smiles(
        self,
        mol_a: Union[str, List[str]],
        mol_b: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Encode edit(s) from molecule A and molecule B SMILES.

        Args:
            mol_a: Molecule A SMILES (single or list)
            mol_b: Molecule B SMILES (single or list)

        Returns:
            Edit embedding(s) as numpy array
            - Single pair: shape (embedding_dim,)
            - Multiple pairs: shape (n_edits, embedding_dim)
        """
        # Handle single vs batch
        if isinstance(mol_a, str):
            assert isinstance(mol_b, str), "mol_a and mol_b must both be strings or lists"
            mol_a_list = [mol_a]
            mol_b_list = [mol_b]
            return_single = True
        else:
            assert len(mol_a) == len(mol_b), "mol_a and mol_b lists must have same length"
            mol_a_list = mol_a
            mol_b_list = mol_b
            return_single = False

        # Encode molecules
        mol_a_emb = self.molecule_embedder.encode(mol_a_list)
        mol_b_emb = self.molecule_embedder.encode(mol_b_list)

        # Compute difference
        edit_emb = mol_b_emb - mol_a_emb

        if return_single:
            return edit_emb[0]
        else:
            return edit_emb

    def encode_from_edit_smiles(
        self,
        edit_smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Encode edit(s) from reaction SMILES format.

        Args:
            edit_smiles: Reaction SMILES "mol_a>>mol_b" (single or list)

        Returns:
            Edit embedding(s) as numpy array

        """
        if isinstance(edit_smiles, str):
            edit_smiles = [edit_smiles]
            return_single = True
        else:
            return_single = False

        # Parse reaction SMILES
        mol_a_list = []
        mol_b_list = []
        for edit in edit_smiles:
            if '>>' not in edit:
                raise ValueError(f"Invalid edit SMILES (missing '>>'): {edit}")

            mol_a_str, mol_b_str = edit.split('>>')
            mol_a_list.append(mol_a_str)
            mol_b_list.append(mol_b_str)

        # Encode
        edit_emb = self.encode_from_smiles(mol_a_list, mol_b_list)

        if return_single:
            return edit_emb[0] if edit_emb.ndim > 1 else edit_emb
        else:
            return edit_emb

    def encode_from_pair_df(self, pairs_df) -> np.ndarray:
        """
        Encode edits from pairs DataFrame.

        Args:
            pairs_df: DataFrame with 'mol_a' and 'mol_b' columns

        Returns:
            Edit embeddings array of shape (n_pairs, embedding_dim)

        """
        return self.encode_from_smiles(
            pairs_df['mol_a'].tolist(),
            pairs_df['mol_b'].tolist()
        )

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of edit embeddings."""
        return self.molecule_embedder.embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this edit embedding method."""
        return f"edit_diff_{self.molecule_embedder.name}"


# Convenience constructors
def edit_embedder_morgan(radius: int = 2, n_bits: int = 2048) -> EditEmbedder:
    """
    Create edit embedder using Morgan fingerprint differences.

    This is the recommended baseline approach.

    Args:
        radius: Morgan fingerprint radius (default: 2 for ECFP4)
        n_bits: Number of bits (default: 2048)

    Returns:
        EditEmbedder instance

    """
    from .fingerprints import FingerprintEmbedder
    mol_emb = FingerprintEmbedder(fp_type='morgan', radius=radius, n_bits=n_bits)
    return EditEmbedder(mol_emb)


def edit_embedder_chemberta(model_name: str = 'chemberta', pooling: str = 'mean') -> EditEmbedder:
    """
    Create edit embedder using ChemBERTa embedding differences.

    Args:
        model_name: ChemBERTa model variant
        pooling: Pooling strategy

    Returns:
        EditEmbedder instance

    """
    try:
        from .chemberta import ChemBERTaEmbedder
        mol_emb = ChemBERTaEmbedder(model_name=model_name, pooling=pooling)
        return EditEmbedder(mol_emb)
    except ImportError as e:
        raise ImportError(
            "ChemBERTa requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e
