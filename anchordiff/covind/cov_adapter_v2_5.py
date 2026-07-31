"""C-arm v2.5 adapter — learnable scale + larger MLP + token dropout.

Three changes vs v2 (cov_adapter.py):
  1. LEARNABLE scale (was fixed 0.25). v2 ran at mean |Δ|=0.026 — the adapter
     was effectively a no-op. Letting the model learn its own injection strength
     removes that ceiling and tells us whether the C-arm has real signal to add.
  2. LARGER adapter: Linear(290→64) → ReLU → Linear(64→10) instead of
     single Linear(52→10). The token grew 52→290 (Morgan FP); a deeper
     network has the capacity to actually use the richer signal.
  3. TOKEN DROPOUT (train-time only): zero out the entire token with prob p.
     Forces the diffusion backbone to be robust without the token, so the
     model can't silently ignore the C-arm conditioning.

Architecture:
    token (290,) → [optional dropout @ train] → Linear(290→64) → ReLU
                  → Linear(64→10) → tanh
                  → × learnable_scale (scalar, init 0.5)
                  → broadcast bias to all pocket atoms (same as v2)

Parameter count: 290*64 + 64 + 64*10 + 10 + 1 = ~19,265 params (vs v2's 530).
Still tiny vs the 1M-param diffusion backbone.

SE(3) equivariance preserved: we modify only scalar (atom-type) features,
never coordinates — same property as v2.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from anchordiff.covind.covalent_token_v2_5 import TOKEN_DIM_V2_5


class CovalentConditioningAdapterV25(nn.Module):
    def __init__(
        self,
        token_dim: int = TOKEN_DIM_V2_5,
        feat_dim: int = 10,
        hidden_dim: int = 64,
        init_scale: float = 0.5,
        token_dropout_p: float = 0.1,
    ):
        """token_dim: input cov-token width (default 290 for v2.5)
        feat_dim:  output bias width (= pocket one_hot dim, 10 for crossdock_full)
        hidden_dim: MLP hidden width
        init_scale: initial value for learnable scale (will be optimized)
        token_dropout_p: probability of zeroing entire token during training
        """
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.token_dropout_p = token_dropout_p

        # Two-layer MLP — meaningful capacity to use the 290-d input
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # Learnable injection scale (the v2 ceiling fix)
        self.log_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_scale)))))

        # Init: small weights so untrained adapter ≈ identity
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    @property
    def scale(self) -> torch.Tensor:
        """Always-positive scale (exp keeps it >0 during training)."""
        return torch.exp(self.log_scale)

    def forward(self, cov_token: torch.Tensor) -> torch.Tensor:
        """cov_token: (token_dim,) or (B, token_dim).
        Returns bias: (feat_dim,) or (B, feat_dim).
        """
        # Train-time token dropout: zero entire token with prob p, scale others
        # by 1/(1-p) so expected value is preserved (standard dropout).
        if self.training and self.token_dropout_p > 0:
            # Drop the whole token (not per-element) — we want the model to
            # learn to be robust to "no covalent conditioning provided".
            mask_shape = (1,) if cov_token.dim() == 1 else (cov_token.shape[0], 1)
            keep = torch.bernoulli(
                torch.full(mask_shape, 1.0 - self.token_dropout_p, device=cov_token.device)
            )
            cov_token = cov_token * keep / (1.0 - self.token_dropout_p)

        h = self.mlp(cov_token)
        h = torch.tanh(h)
        return self.scale * h


def inject_into_pocket_oh_v25(
    pkt_oh: torch.Tensor,
    cov_token: torch.Tensor,
    adapter: CovalentConditioningAdapterV25,
) -> torch.Tensor:
    """pkt_oh: (N_pocket, feat_dim).  cov_token: (token_dim,).
    Returns the augmented pkt_oh (same shape).
    """
    bias = adapter(cov_token)            # (feat_dim,)
    return pkt_oh + bias.unsqueeze(0)    # broadcast over N_pocket


if __name__ == "__main__":
    a = CovalentConditioningAdapterV25()
    n_params = sum(p.numel() for p in a.parameters())
    print(f"adapter params: {n_params}  (v2 was 530)")
    print(f"learnable scale: init={float(a.scale):.3f}")

    tok = torch.randn(TOKEN_DIM_V2_5)
    a.eval()
    bias_eval = a(tok)
    print(f"  eval bias  shape={tuple(bias_eval.shape)}  "
          f"range=[{bias_eval.min().item():+.3f}, {bias_eval.max().item():+.3f}]  "
          f"|mean|={bias_eval.abs().mean().item():.4f}")

    # Train mode — dropout active
    a.train()
    drops = []
    for _ in range(100):
        bias_train = a(tok)
        drops.append(float(bias_train.abs().mean()))
    import statistics
    print(f"  train-mode |bias| over 100 forwards: "
          f"min={min(drops):.4f}  med={statistics.median(drops):.4f}  max={max(drops):.4f}")
    print(f"    (zeros from token-dropout p={a.token_dropout_p}; non-zeros from learnable scale)")

    # Injection sanity
    pkt = torch.zeros(50, 10); pkt[:, 0] = 1.0
    pkt_aug = inject_into_pocket_oh_v25(pkt, tok, a)
    print(f"  pkt_oh injection mean|Δ|={float((pkt_aug - pkt).abs().mean()):.4f}")

    # Gradient flow
    tok2 = torch.randn(TOKEN_DIM_V2_5, requires_grad=True)
    a.train()
    bias2 = a(tok2)
    bias2.sum().backward()
    print(f"  token grad ‖.‖={float(tok2.grad.norm()):.4f}  "
          f"log_scale grad={float(a.log_scale.grad):.4f}")
