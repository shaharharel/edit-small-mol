"""
Critic (value) network for PPO-based molecular generation.

Estimates expected reward from Mol2Mol transformer encoder hidden states.
Used alongside the Mol2Mol policy (actor) to compute advantages for PPO updates.

Architecture:
    - Input: encoder output from Mol2Mol transformer (mean-pooled, d_model=256)
    - Network: Linear(256, 128) -> ReLU -> Linear(128, 1)
    - Mean-pool over non-padding positions using src_mask
"""

import torch
import torch.nn as nn
from typing import Optional


class SmilesCritic(nn.Module):
    """
    Value network that estimates expected reward from encoder hidden states.

    Takes the encoder output of a Mol2Mol transformer and produces a scalar
    value estimate via mean-pooling followed by a small MLP.

    Args:
        d_model: Dimension of the transformer encoder output (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate value from encoder hidden states.

        Args:
            encoder_output: Encoder output tensor [batch, seq_len, d_model]
            src_mask: Source mask [batch, 1, seq_len] (True where tokens exist)

        Returns:
            Value estimates [batch] (scalar per sequence)
        """
        # src_mask shape: [batch, 1, seq_len] -> [batch, seq_len, 1]
        mask = src_mask.squeeze(1).unsqueeze(-1).float()  # [batch, seq_len, 1]

        # Mean-pool over non-padding positions
        masked_output = encoder_output * mask  # [batch, seq_len, d_model]
        lengths = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
        pooled = masked_output.sum(dim=1) / lengths  # [batch, d_model]

        # MLP to scalar value
        value = self.net(pooled).squeeze(-1)  # [batch]
        return value
