"""Configurable C-arm adapter for J3/J4/J5 ablations.

Combines features from cov_adapter.py (v2) and cov_adapter_v2_5.py:
  - Configurable scale (fixed scalar OR learnable parameter)
  - Configurable hidden dim (0 = single Linear, >0 = two-layer MLP w/ ReLU)
  - Configurable token dropout (zero out entire token with prob p)
  - Configurable spatial decay (J5): bias to atom i scaled by
    exp(-||pos_i|| / lambda) where ||pos_i|| is the L2 distance to Sγ
    (= origin in the D-arm local frame).

Token-dropout: zero the entire token with prob p; scale others by 1/(1-p)
to preserve expected magnitude.

Spatial decay: ONLY applied via `inject_into_pocket_oh_spatial()` which
needs the pocket coordinates. The standard `inject_into_pocket_oh()` does
the uniform-broadcast injection (v2/v2.5 behavior).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AblationAdapter(nn.Module):
    def __init__(
        self,
        token_dim: int,
        feat_dim: int = 10,
        scale_init: float = 0.25,
        scale_learnable: bool = False,
        hidden_dim: int = 0,
        token_dropout_p: float = 0.0,
        spatial_decay_lambda: float = 0.0,
    ):
        """
        Args:
          token_dim: input cov-token width (52 for v2, 290 for v2.5).
          feat_dim:  output bias width (= pocket_one_hot dim, 10).
          scale_init: initial scale; for fixed mode this is the constant.
          scale_learnable: if True, scale = exp(log_scale) where log_scale
            is a trainable parameter. Init: log(scale_init).
          hidden_dim: 0 = single Linear(token→feat); >0 = MLP with ReLU.
          token_dropout_p: train-time prob of zeroing the entire token.
          spatial_decay_lambda: if > 0, the inject_into_pocket_oh_spatial()
            function uses this decay constant. Stored here for downstream use.
        """
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.token_dropout_p = token_dropout_p
        self.spatial_decay_lambda = spatial_decay_lambda
        self.scale_learnable = scale_learnable

        # Build the head: single linear or MLP
        if hidden_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feat_dim),
            )
        else:
            self.mlp = nn.Sequential(nn.Linear(token_dim, feat_dim))

        # Init small so untrained ≈ identity
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        # Scale: fixed scalar or learnable log-scale (always positive)
        if scale_learnable:
            self.log_scale = nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(scale_init))))
            )
            self._scale_const = None
        else:
            self.register_buffer("_scale_const", torch.tensor(float(scale_init)))
            self.log_scale = None

    @property
    def scale(self) -> torch.Tensor:
        if self.scale_learnable:
            return torch.exp(self.log_scale)
        return self._scale_const

    def forward(self, cov_token: torch.Tensor) -> torch.Tensor:
        """cov_token: (token_dim,) or (B, token_dim). Returns bias (feat_dim,)."""
        if self.training and self.token_dropout_p > 0:
            mask_shape = (1,) if cov_token.dim() == 1 else (cov_token.shape[0], 1)
            keep = torch.bernoulli(
                torch.full(mask_shape, 1.0 - self.token_dropout_p, device=cov_token.device)
            )
            cov_token = cov_token * keep / (1.0 - self.token_dropout_p)
        h = self.mlp(cov_token)
        h = torch.tanh(h)
        return self.scale * h


def inject_into_pocket_oh(
    pkt_oh: torch.Tensor, cov_token: torch.Tensor, adapter: AblationAdapter,
) -> torch.Tensor:
    """Uniform-broadcast injection (v2/v2.5 behavior). pkt_oh: (N_pocket, feat_dim)."""
    bias = adapter(cov_token)
    return pkt_oh + bias.unsqueeze(0)


def inject_into_pocket_oh_spatial(
    pkt_oh: torch.Tensor,
    cov_token: torch.Tensor,
    adapter: AblationAdapter,
    pkt_pos: torch.Tensor,
) -> torch.Tensor:
    """Spatially-decayed injection (J5).

    bias_i = adapter(token) × exp(-||pkt_pos_i|| / lambda)

    In the D-arm local frame, Sγ is at the origin, so ||pkt_pos_i|| is the
    Euclidean distance from atom i to the Cys Sγ. lambda controls the decay
    range (4 Å covers the immediate Cys environment).

    pkt_pos: (N_pocket, 3) in local frame (Sγ at origin).
    """
    bias = adapter(cov_token)  # (feat_dim,)
    lam = adapter.spatial_decay_lambda
    if lam <= 0:
        return pkt_oh + bias.unsqueeze(0)
    dist = torch.norm(pkt_pos, dim=1)  # (N_pocket,)
    weight = torch.exp(-dist / lam)  # (N_pocket,)
    # bias_per_atom = bias × weight_per_atom  → outer-product (N_pocket, feat_dim)
    return pkt_oh + weight.unsqueeze(1) * bias.unsqueeze(0)


if __name__ == "__main__":
    # Smoke test all variant configurations
    import sys
    configs = [
        ("J4a scale=1.0", dict(token_dim=52, scale_init=1.0, hidden_dim=0)),
        ("J4b warhead FP", dict(token_dim=290, scale_init=0.25, hidden_dim=0)),
        ("J4c dropout", dict(token_dim=52, scale_init=0.25, hidden_dim=0, token_dropout_p=0.1)),
        ("J5 spatial",   dict(token_dim=290, scale_init=0.5, scale_learnable=True,
                               hidden_dim=64, token_dropout_p=0.1, spatial_decay_lambda=4.0)),
    ]
    for label, kw in configs:
        a = AblationAdapter(**kw)
        n_params = sum(p.numel() for p in a.parameters())
        tok = torch.randn(kw["token_dim"])
        a.eval()
        bias = a(tok)
        print(f"{label:25s}  params={n_params:6d}  scale={float(a.scale):.3f}  "
              f"|bias|mean={bias.abs().mean().item():.4f}  learnable={a.scale_learnable}")
