"""C-arm: covalent-token conditioning adapter.

The DiffSBDD `crossdocked_fullatom_cond` checkpoint exposes only a 10-d
atom-element feature per pocket residue (no native residue-type or
covalent-context channel). We inject the C-arm by training a tiny
adapter that maps the 37-d covalent token → a per-pocket-node bias added
elementwise to the pocket one-hot features:

    pkt_oh_aug[i, :]  =  pkt_oh[i, :]  +  adapter(cov_token)

The adapter is a single `nn.Linear(37, 10)` followed by `Tanh()` (bounded
output prevents overwhelming the existing atom-type signal). It is trained
jointly with the diffusion model. ~370 parameters.

Why this works:
  - SE(3)-equivariance is preserved because we modify only invariant
    (atom-type) features, never coordinates.
  - Per-node-equal bias means the token affects every pocket node identically,
    which is the right inductive bias for "global covalent context."
  - Easy to ablate: zero the adapter → exact equivalence to the D-only model.

For inference, the adapter is loaded with the rest of the fine-tuned ckpt.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from anchordiff.covind.covalent_token import TOKEN_DIM


class CovalentConditioningAdapter(nn.Module):
    def __init__(self, token_dim: int = TOKEN_DIM, feat_dim: int = 10, scale: float = 0.25):
        """token_dim: input cov-token width (default 37)
        feat_dim:  output bias width (= pocket_one_hot dim, 10 for crossdock_full)
        scale:     fixed multiplier applied AFTER Tanh, controls injection
                   strength. 0.25 keeps the bias well below the unit one-hot
                   amplitudes during early training."""
        super().__init__()
        self.linear = nn.Linear(token_dim, feat_dim)
        self.scale = scale
        # initialise small so untrained adapter ~= identity
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cov_token: torch.Tensor) -> torch.Tensor:
        """token: (token_dim,) or (B, token_dim).  Returns (feat_dim,) or (B, feat_dim)."""
        return self.scale * torch.tanh(self.linear(cov_token))


def inject_into_pocket_oh(pkt_oh: torch.Tensor, cov_token: torch.Tensor,
                          adapter: CovalentConditioningAdapter) -> torch.Tensor:
    """pkt_oh: (N_pocket, feat_dim).  cov_token: (token_dim,).
    Returns the augmented pkt_oh (same shape)."""
    bias = adapter(cov_token)             # (feat_dim,)
    return pkt_oh + bias.unsqueeze(0)     # broadcast over N_pocket


if __name__ == "__main__":
    a = CovalentConditioningAdapter()
    print(f"adapter params: {sum(p.numel() for p in a.parameters())}")
    tok = torch.randn(TOKEN_DIM)
    bias = a(tok)
    print(f"token → bias  shape={tuple(bias.shape)}  range=[{bias.min().item():+.3f}, {bias.max().item():+.3f}]")
    # injection
    pkt = torch.zeros(50, 10)
    pkt[:, 0] = 1.0   # all-C pocket
    pkt_aug = inject_into_pocket_oh(pkt, tok, a)
    print(f"pkt_oh injection: mean Δ|.|={float((pkt_aug - pkt).abs().mean()):.4f}")
    # gradient flow check
    tok2 = torch.randn(TOKEN_DIM, requires_grad=True)
    pkt_aug2 = inject_into_pocket_oh(pkt, tok2, a)
    pkt_aug2.sum().backward()
    print(f"grad through adapter+token: token grad ‖.‖={float(tok2.grad.norm()):.4f}")
