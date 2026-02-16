"""
Trainable edit embeddings that can be refined during multi-task learning.

Instead of using simple difference (product - reactant), this module learns
to transform the molecular embeddings into edit representations that are
optimized for predicting property changes across multiple tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class TrainableEditEmbedder(nn.Module):
    """
    Trainable edit embedding network with support for two embedding strategies.

    Takes raw molecule embeddings (reactant, product) and learns to produce
    edit embeddings that are refined through gradient descent during training.

    **Two Modes:**

    Mode 1 (use_edit_fragments=False): Full Molecule Embeddings
        diff = Embed(full_mol_b) - Embed(full_mol_a)
        - Context-aware: Same edit produces different vectors on different scaffolds
        - Continuous generalization to novel molecules

    Mode 2 (use_edit_fragments=True): Edit Fragment Embeddings
        diff = Embed(edit_fragment_product) - Embed(edit_fragment_reactant)
        - Scaffold-independent: Same transformation always same edit vector
        - Better transfer learning across scaffolds
        - Requires edit_smiles column in data

    Architecture:
    1. Initial embedding: difference baseline (product - reactant or fragments)
    2. Learnable transformation: MLP that refines the difference
    3. Skip connection: Preserves initial difference signal

    This allows edit representations to be learned jointly across multiple
    tasks, enabling the model to discover better edit features.

    Args:
        mol_dim: Dimension of input molecule embeddings
        edit_dim: Dimension of output edit embeddings (default: same as mol_dim)
        hidden_dims: Hidden layer dimensions for transformation network
        dropout: Dropout probability
        use_skip_connection: Add skip connection from input difference
        activation: Activation function ('relu', 'elu', 'gelu')
        use_edit_fragments: If True, use edit fragment embeddings (Mode 2)
                           If False, use full molecule embeddings (Mode 1, default)

    Example (Mode 1 - Full molecules):
        >>> edit_embedder = TrainableEditEmbedder(
        ...     mol_dim=512,
        ...     edit_dim=512,
        ...     hidden_dims=[256, 256],
        ...     use_edit_fragments=False
        ... )
        >>> mol_a_emb = torch.randn(32, 512)
        >>> mol_b_emb = torch.randn(32, 512)
        >>> edit_emb = edit_embedder(mol_a_emb, mol_b_emb)

    Example (Mode 2 - Edit fragments):
        >>> edit_embedder = TrainableEditEmbedder(
        ...     mol_dim=512,
        ...     use_edit_fragments=True
        ... )
        >>> mol_a_emb = torch.randn(32, 512)  # Full molecule A (context)
        >>> mol_b_emb = torch.randn(32, 512)  # Full molecule B (not used)
        >>> edit_frag_a_emb = torch.randn(32, 512)  # Edit fragment A
        >>> edit_frag_b_emb = torch.randn(32, 512)  # Edit fragment B
        >>> edit_emb = edit_embedder(
        ...     mol_a_emb, mol_b_emb,
        ...     edit_frag_a_emb=edit_frag_a_emb,
        ...     edit_frag_b_emb=edit_frag_b_emb
        ... )
    """

    def __init__(
        self,
        mol_dim: int,
        edit_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_skip_connection: bool = True,
        activation: str = 'relu',
        use_edit_fragments: bool = False
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.edit_dim = edit_dim if edit_dim is not None else mol_dim
        self.use_skip_connection = use_skip_connection
        self.use_edit_fragments = use_edit_fragments

        # Default architecture: one hidden layer
        if hidden_dims is None:
            hidden_dims = [mol_dim]

        self.hidden_dims = hidden_dims

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build transformation network
        # Input: difference (product - reactant)
        # Output: refined edit embedding
        layers = []
        prev_dim = mol_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output projection
        layers.append(nn.Linear(prev_dim, self.edit_dim))

        self.transform = nn.Sequential(*layers)

        # Skip connection projection (if dimensions don't match)
        if use_skip_connection and mol_dim != self.edit_dim:
            self.skip_projection = nn.Linear(mol_dim, self.edit_dim)
        else:
            self.skip_projection = None

    def forward(
        self,
        mol_a_emb: torch.Tensor,
        mol_b_emb: torch.Tensor,
        edit_frag_a_emb: Optional[torch.Tensor] = None,
        edit_frag_b_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform molecule pair into trainable edit embedding.

        Args:
            mol_a_emb: Molecule A embeddings [batch_size, mol_dim]
            mol_b_emb: Molecule B embeddings [batch_size, mol_dim]
            edit_frag_a_emb: Edit fragment A embeddings [batch_size, mol_dim]
                            Only used if use_edit_fragments=True
            edit_frag_b_emb: Edit fragment B embeddings [batch_size, mol_dim]
                            Only used if use_edit_fragments=True

        Returns:
            Edit embeddings [batch_size, edit_dim]
        """
        # Compute difference (baseline edit representation)
        if self.use_edit_fragments:
            # Mode 2: Use edit fragment embeddings
            if edit_frag_a_emb is None or edit_frag_b_emb is None:
                raise ValueError(
                    "use_edit_fragments=True requires edit_frag_a_emb and edit_frag_b_emb"
                )
            diff = edit_frag_b_emb - edit_frag_a_emb
        else:
            # Mode 1: Use full molecule embeddings (original behavior)
            diff = mol_b_emb - mol_a_emb


        # Transform difference through learnable network
        edit_emb = self.transform(diff)

        # Add skip connection if enabled
        if self.use_skip_connection:
            if self.skip_projection is not None:
                skip = self.skip_projection(diff)
            else:
                skip = diff
            edit_emb = edit_emb + skip

        return edit_emb

    def freeze(self):
        """Freeze all parameters (stop learning edit representations)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (resume learning edit representations)."""
        for param in self.parameters():
            param.requires_grad = True


class ConcatenationEditEmbedder(nn.Module):
    """
    Alternative trainable edit embedder using concatenation.

    Instead of taking difference, concatenates [reactant, product] and
    learns to extract edit features directly.

    This can capture more complex edit patterns but requires more parameters.

    Args:
        mol_dim: Dimension of input molecule embeddings
        edit_dim: Dimension of output edit embeddings
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        mol_dim: int,
        edit_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.edit_dim = edit_dim

        # Default architecture
        if hidden_dims is None:
            hidden_dims = [mol_dim, mol_dim // 2]

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        # Input: [reactant, product] concatenated
        layers = []
        prev_dim = mol_dim * 2  # Concatenation doubles dimension

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, edit_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, mol_a_emb: torch.Tensor, mol_b_emb: torch.Tensor) -> torch.Tensor:
        """
        Extract edit embedding from molecule pair.

        Args:
            mol_a_emb: Molecule A embeddings [batch_size, mol_dim]
            mol_b_emb: Molecule B embeddings [batch_size, mol_dim]

        Returns:
            Edit embeddings [batch_size, edit_dim]
        """
        # Concatenate molecule A and molecule B
        x = torch.cat([mol_a_emb, mol_b_emb], dim=-1)

        # Extract edit features
        edit_emb = self.network(x)

        return edit_emb

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
