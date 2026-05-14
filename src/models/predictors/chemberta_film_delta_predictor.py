"""
ChemBERTa-encoded FiLM-conditioned delta predictor.

Identical architecture to :class:`FiLMDeltaMLP` (Phase 2 winner) but operating
on frozen ChemBERTa-2-MTR sequence embeddings (default 384-dim) instead of
Morgan fingerprints (2048-dim). The point is a clean head-to-head: same
predictor head + FiLM conditioning, different molecule representation.

Usage mirrors FiLMDeltaMLP — caller supplies (emb_a, emb_b) tensors of
shape [batch, input_dim]. The class does not own the encoder; embeddings
are precomputed and cached by the caller (see
``data/embedding_cache/chemberta2-mtr.npz``).
"""

from typing import List, Optional

from .film_delta_predictor import FiLMDeltaMLP


# Default ChemBERTa-2 hidden size (DeepChem/ChemBERTa-77M-MTR).
CHEMBERTA2_MTR_DIM = 384


class ChemBERTaFiLMDeltaMLP(FiLMDeltaMLP):
    """
    FiLMDeltaMLP wired for ChemBERTa-2-MTR frozen embeddings (384-d default).

    Reuses the parent class verbatim — only difference is a defaulted
    ``input_dim`` of 384 and a slightly trimmer auto-generated hidden stack
    that is appropriate for the smaller embedding (Morgan FP variant uses
    [512, 256, 128]; with a 384-d input the [256, 128, 64] stack keeps the
    same shrink-by-half pattern and parameter count).

    Args:
        input_dim: Embedding dimension (default 384 for ChemBERTa-2-MTR).
        hidden_dims: Hidden layer dims; if None, uses [256, 128, 64].
        dropout: Dropout probability.
        spectral: If True, use spectral normalization.
        modulation_strength: Scale factor for FiLM modulation (0.0-1.0).
        use_batchnorm: If True, use BatchNorm in FiLM blocks.
    """

    def __init__(
        self,
        input_dim: int = CHEMBERTA2_MTR_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        spectral: bool = False,
        modulation_strength: float = 1.0,
        use_batchnorm: bool = False,
    ):
        if hidden_dims is None:
            # Mirror Morgan-FP FiLMDelta's [512, 256, 128] shrink pattern but
            # scaled to the 384-d input. Each layer halves the previous dim.
            hidden_dims = [256, 128, 64]

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            spectral=spectral,
            modulation_strength=modulation_strength,
            use_batchnorm=use_batchnorm,
        )
