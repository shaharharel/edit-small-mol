"""
Multi-task FiLMDelta predictor with two heads sharing a single encoder backbone.

Architecture
------------
              Morgan FP 2048d
                    |
            Shared encoder (MLP, hidden_dims = [1024, 512, 256], dropout 0.2)
                    |
              +-----+-----+
              |           |
            Delta head    Abs head
            (FiLM)        (Linear -> 1)
        uses enc(A), enc(B), raw delta cond
        outputs pred delta
                        outputs pred pIC50

Joint loss
----------
L = L_Delta + lambda * L_abs

The model exposes:
  - forward_delta(fp_a, fp_b) -> predicted delta
  - forward_abs(fp)          -> predicted pIC50
  - forward(fp_a, fp_b)      -> (predicted_delta, predicted_abs_a, predicted_abs_b)

This is the "best of both worlds" model — a single shared encoder that can serve
both Delta queries (pair input) and Abs queries (single-mol input).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .film_delta_predictor import FiLMBlock


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (DANN-style). Forward = identity; backward = -alpha * grad."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


class SharedMolEncoder(nn.Module):
    """Shared MLP encoder mapping Morgan FP 2048d -> hidden representation.

    The encoder is shared between the Delta head and the Abs head. Both inputs
    in a pair are encoded by the SAME weights (siamese-style), which is what
    we want: the encoder learns molecule-level features useful for both tasks.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskFiLMDelta(nn.Module):
    """Multi-task model with shared encoder + Delta (FiLM) head + Abs head.

    Args:
        input_dim: Morgan FP dimension (default 2048).
        enc_hidden_dims: Hidden sizes for the shared encoder. Default
            [1024, 512, 256].
        film_hidden_dims: Hidden sizes for the FiLM Delta head (operating on
            the encoded representation). Default [256, 128].
        dropout: Dropout in both encoder and FiLM head.
        film_cond_dim: Internal dimension used to encode the raw delta
            (fp_b - fp_a) before broadcasting as FiLM conditioning. Default 256.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        enc_hidden_dims: Optional[List[int]] = None,
        film_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        film_cond_dim: int = 256,
    ):
        super().__init__()
        if enc_hidden_dims is None:
            enc_hidden_dims = [1024, 512, 256]
        if film_hidden_dims is None:
            film_hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.film_hidden_dims = film_hidden_dims
        self.film_cond_dim = film_cond_dim
        self.dropout = dropout

        # Shared encoder (used by both heads)
        self.encoder = SharedMolEncoder(
            input_dim=input_dim,
            hidden_dims=enc_hidden_dims,
            dropout=dropout,
        )
        enc_out = self.encoder.output_dim

        # ---- Abs head: linear from shared encoder's last layer ----
        # Per the spec, "simple linear projection from the shared encoder's last layer".
        self.abs_head = nn.Linear(enc_out, 1)

        # ---- Delta head (FiLM-conditioned) ----
        # The delta cond is built from the RAW Morgan FP difference (fp_b - fp_a),
        # encoded through a small MLP. This preserves the edit-as-reaction signal.
        self.delta_cond_encoder = nn.Sequential(
            nn.Linear(input_dim, film_cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM blocks operate on the encoded molecule representation, conditioned
        # on the delta cond. We share the FiLM stack: pred_a = f(enc_a | cond),
        # pred_b = f(enc_b | cond), then delta = pred_b - pred_a.
        self.film_blocks = nn.ModuleList()
        prev = enc_out
        for hdim in film_hidden_dims:
            self.film_blocks.append(
                FiLMBlock(
                    input_dim=prev,
                    hidden_dim=hdim,
                    cond_dim=film_cond_dim,
                    dropout=dropout,
                    spectral=False,
                    use_batchnorm=False,
                )
            )
            prev = hdim
        self.delta_output = nn.Linear(prev, 1)

    # ----- public forwards -----
    def forward_abs(self, fp: torch.Tensor) -> torch.Tensor:
        """Predict pIC50 for a single molecule from its Morgan FP."""
        h = self.encoder(fp)
        return self.abs_head(h).squeeze(-1)

    def _film_pred_single(self, enc_x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = enc_x
        for blk in self.film_blocks:
            h = blk(h, cond)
        return self.delta_output(h).squeeze(-1)

    def forward_delta(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        """Predict delta = pIC50(B) - pIC50(A) for a pair."""
        enc_a = self.encoder(fp_a)
        enc_b = self.encoder(fp_b)
        raw_delta = fp_b - fp_a
        cond = self.delta_cond_encoder(raw_delta)
        pred_a = self._film_pred_single(enc_a, cond)
        pred_b = self._film_pred_single(enc_b, cond)
        return pred_b - pred_a

    def forward(
        self, fp_a: torch.Tensor, fp_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both heads. Returns (delta_pred, abs_pred_a, abs_pred_b)."""
        enc_a = self.encoder(fp_a)
        enc_b = self.encoder(fp_b)
        raw_delta = fp_b - fp_a
        cond = self.delta_cond_encoder(raw_delta)
        pred_a_delta = self._film_pred_single(enc_a, cond)
        pred_b_delta = self._film_pred_single(enc_b, cond)
        delta_pred = pred_b_delta - pred_a_delta
        abs_pred_a = self.abs_head(enc_a).squeeze(-1)
        abs_pred_b = self.abs_head(enc_b).squeeze(-1)
        return delta_pred, abs_pred_a, abs_pred_b


# ---------------------------------------------------------------------------
# Option A — Separate encoders for Δ vs Abs (no weight sharing)
# ---------------------------------------------------------------------------
class SeparateEncoderMultiTaskFiLMDelta(nn.Module):
    """Two parallel encoders: Encoder_main for Δ head, Encoder_abs for Abs head.

    Trained JOINTLY (same optimizer, same data loader) but no shared weights —
    keeps the Δ-side encoder clean of per-mol lab-bias signals.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        enc_hidden_dims=None,
        film_hidden_dims=None,
        dropout: float = 0.2,
        film_cond_dim: int = 256,
    ):
        super().__init__()
        if enc_hidden_dims is None:
            enc_hidden_dims = [1024, 512, 256]
        if film_hidden_dims is None:
            film_hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.film_hidden_dims = film_hidden_dims
        self.film_cond_dim = film_cond_dim

        # Two separate encoders
        self.encoder_main = SharedMolEncoder(input_dim, enc_hidden_dims, dropout)
        self.encoder_abs = SharedMolEncoder(input_dim, enc_hidden_dims, dropout)
        enc_out = self.encoder_main.output_dim

        self.abs_head = nn.Linear(enc_out, 1)

        self.delta_cond_encoder = nn.Sequential(
            nn.Linear(input_dim, film_cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.film_blocks = nn.ModuleList()
        prev = enc_out
        for hdim in film_hidden_dims:
            self.film_blocks.append(
                FiLMBlock(prev, hdim, film_cond_dim, dropout=dropout)
            )
            prev = hdim
        self.delta_output = nn.Linear(prev, 1)

    def forward_abs(self, fp):
        h = self.encoder_abs(fp)
        return self.abs_head(h).squeeze(-1)

    def _film_pred_single(self, enc_x, cond):
        h = enc_x
        for blk in self.film_blocks:
            h = blk(h, cond)
        return self.delta_output(h).squeeze(-1)

    def forward_delta(self, fp_a, fp_b):
        enc_a = self.encoder_main(fp_a)
        enc_b = self.encoder_main(fp_b)
        cond = self.delta_cond_encoder(fp_b - fp_a)
        pa = self._film_pred_single(enc_a, cond)
        pb = self._film_pred_single(enc_b, cond)
        return pb - pa

    def forward(self, fp_a, fp_b):
        # Δ path uses encoder_main; Abs path uses encoder_abs
        enc_a_main = self.encoder_main(fp_a)
        enc_b_main = self.encoder_main(fp_b)
        cond = self.delta_cond_encoder(fp_b - fp_a)
        pa = self._film_pred_single(enc_a_main, cond)
        pb = self._film_pred_single(enc_b_main, cond)
        delta_pred = pb - pa
        # Abs path: independent encoder
        abs_a = self.abs_head(self.encoder_abs(fp_a)).squeeze(-1)
        abs_b = self.abs_head(self.encoder_abs(fp_b)).squeeze(-1)
        return delta_pred, abs_a, abs_b


# ---------------------------------------------------------------------------
# Option B — Adversarial debiasing (shared encoder + lab classifier via GRL)
# ---------------------------------------------------------------------------
class AdversarialMultiTaskFiLMDelta(nn.Module):
    """Shared encoder + 3 heads:
      - Δ head (FiLM)
      - Abs head (linear)
      - Lab classifier head (softmax over assay_ids, attached via gradient
        reversal so the encoder is encouraged to be lab-invariant).

    Joint loss outside the module: L = L_Δ + λ·L_abs + α·L_lab_via_GRL
    (gradients to encoder are negated by the GRL; the lab head's own
    parameters still update normally to maximize its own classification accuracy).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_assays: int = 24,
        enc_hidden_dims=None,
        film_hidden_dims=None,
        dropout: float = 0.2,
        film_cond_dim: int = 256,
        lab_hidden: int = 64,
    ):
        super().__init__()
        if enc_hidden_dims is None:
            enc_hidden_dims = [1024, 512, 256]
        if film_hidden_dims is None:
            film_hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.n_assays = n_assays
        self.enc_hidden_dims = enc_hidden_dims

        self.encoder = SharedMolEncoder(input_dim, enc_hidden_dims, dropout)
        enc_out = self.encoder.output_dim

        self.abs_head = nn.Linear(enc_out, 1)

        self.delta_cond_encoder = nn.Sequential(
            nn.Linear(input_dim, film_cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.film_blocks = nn.ModuleList()
        prev = enc_out
        for hdim in film_hidden_dims:
            self.film_blocks.append(
                FiLMBlock(prev, hdim, film_cond_dim, dropout=dropout)
            )
            prev = hdim
        self.delta_output = nn.Linear(prev, 1)

        # Lab classifier head
        self.lab_head = nn.Sequential(
            nn.Linear(enc_out, lab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lab_hidden, n_assays),
        )

    def forward_abs(self, fp):
        return self.abs_head(self.encoder(fp)).squeeze(-1)

    def _film_pred_single(self, enc_x, cond):
        h = enc_x
        for blk in self.film_blocks:
            h = blk(h, cond)
        return self.delta_output(h).squeeze(-1)

    def forward_delta(self, fp_a, fp_b):
        enc_a = self.encoder(fp_a)
        enc_b = self.encoder(fp_b)
        cond = self.delta_cond_encoder(fp_b - fp_a)
        return self._film_pred_single(enc_b, cond) - self._film_pred_single(enc_a, cond)

    def forward(self, fp_a, fp_b, alpha: float = 1.0):
        enc_a = self.encoder(fp_a)
        enc_b = self.encoder(fp_b)
        cond = self.delta_cond_encoder(fp_b - fp_a)
        pa = self._film_pred_single(enc_a, cond)
        pb = self._film_pred_single(enc_b, cond)
        delta_pred = pb - pa
        abs_a = self.abs_head(enc_a).squeeze(-1)
        abs_b = self.abs_head(enc_b).squeeze(-1)
        # Lab classifier via gradient reversal — applied to BOTH endpoints
        lab_a_logits = self.lab_head(grad_reverse(enc_a, alpha))
        lab_b_logits = self.lab_head(grad_reverse(enc_b, alpha))
        return delta_pred, abs_a, abs_b, lab_a_logits, lab_b_logits
