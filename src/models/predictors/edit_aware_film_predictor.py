"""
Edit-aware FiLM predictor variants.

All variants share the FiLM prediction backbone from film_delta_predictor.py
but differ in how the conditioning vector (delta_cond) is computed:

1. DrfpFiLMDeltaMLP — DRFP reaction fingerprint conditioning
2. DualStreamFiLMDeltaMLP — Gated DRFP + Morgan diff fusion
3. FragAnchoredFiLMDeltaMLP — Fragment FP + edit features conditioning
4. MultiModalEditFiLMDeltaMLP — All representations fused
5. EditHypernetFiLMDeltaMLP — Low-rank hypernetwork (edit generates weight perturbations)
6. TargetCondFiLMDeltaMLP — FiLMDelta + target identity conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

from src.models.predictors.film_delta_predictor import FiLMLayer, FiLMBlock


# ═══════════════════════════════════════════════════════════════════════════
# Shared FiLM prediction backbone
# ═══════════════════════════════════════════════════════════════════════════

class FiLMPredictionBackbone(nn.Module):
    """Shared FiLM backbone: takes (mol_emb, delta_cond) → scalar prediction.

    Used by all edit-aware variants. Identical to FiLMDeltaMLP.forward_single
    but factored out for reuse.
    """

    def __init__(self, input_dim: int, cond_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.blocks.append(FiLMBlock(prev_dim, h, cond_dim, dropout=dropout))
            prev_dim = h

        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass: x is mol embedding, cond is edit conditioning."""
        h = x
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 1: DRFP-FiLM
# ═══════════════════════════════════════════════════════════════════════════

class DrfpFiLMDeltaMLP(nn.Module):
    """FiLM-conditioned delta predictor using DRFP reaction fingerprints.

    Instead of conditioning on emb_b - emb_a (Morgan diff in hashed space),
    conditions on DRFP which encodes the symmetric difference of atom
    environments — exactly what changed — cleanly before hashing.

    forward(emb_a, emb_b, drfp) → predicted delta
    """

    def __init__(self, mol_dim: int, drfp_dim: int = 2048,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()
        self.mol_dim = mol_dim
        self.drfp_dim = drfp_dim

        cond_hidden = max(mol_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(drfp_dim, cond_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = FiLMPredictionBackbone(
            mol_dim, cond_hidden, hidden_dims, dropout)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                drfp: torch.Tensor) -> torch.Tensor:
        delta_cond = self.delta_encoder(drfp)
        pred_a = self.backbone(emb_a, delta_cond)
        pred_b = self.backbone(emb_b, delta_cond)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 2: DualStream-FiLM
# ═══════════════════════════════════════════════════════════════════════════

class DualStreamFiLMDeltaMLP(nn.Module):
    """Gated fusion of DRFP + Morgan diff, optionally with edit features.

    Two streams:
      - stream_drfp: reaction-level signal (scaffold-independent)
      - stream_mol: context-aware signal (emb_b - emb_a)

    A learned gate blends them. Optional edit feature reconstruction
    auxiliary loss forces interpretability.

    forward(emb_a, emb_b, drfp, edit_feats=None) → predicted delta
    """

    def __init__(self, mol_dim: int, drfp_dim: int = 2048,
                 edit_feat_dim: int = 28,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()
        self.mol_dim = mol_dim
        self.edit_feat_dim = edit_feat_dim

        stream_h = max(mol_dim // 2, 128)

        self.drfp_proj = nn.Sequential(
            nn.Linear(drfp_dim, stream_h),
            nn.LayerNorm(stream_h),
        )
        self.mol_diff_proj = nn.Sequential(
            nn.Linear(mol_dim, stream_h),
            nn.LayerNorm(stream_h),
        )

        # Learned gate
        self.gate = nn.Sequential(
            nn.Linear(stream_h * 2, stream_h),
            nn.Sigmoid(),
        )

        # Optional edit features projection
        cond_input_dim = stream_h
        if edit_feat_dim > 0:
            self.feat_proj = nn.Linear(edit_feat_dim, 64)
            cond_input_dim = stream_h + 64
        else:
            self.feat_proj = None

        cond_hidden = max(mol_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(cond_input_dim, cond_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = FiLMPredictionBackbone(
            mol_dim, cond_hidden, hidden_dims, dropout)

        # Auxiliary: reconstruct edit features from edit_enc
        self.aux_reconstructor = nn.Linear(stream_h, edit_feat_dim) if edit_feat_dim > 0 else None

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                drfp: torch.Tensor,
                edit_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        stream_drfp = self.drfp_proj(drfp)
        stream_mol = self.mol_diff_proj(emb_b - emb_a)

        g = self.gate(torch.cat([stream_drfp, stream_mol], dim=-1))
        edit_enc = g * stream_drfp + (1 - g) * stream_mol

        # Store for auxiliary loss access
        self._edit_enc = edit_enc

        if self.feat_proj is not None and edit_feats is not None:
            feat_emb = self.feat_proj(edit_feats)
            cond_input = torch.cat([edit_enc, feat_emb], dim=-1)
        else:
            cond_input = edit_enc

        delta_cond = self.delta_encoder(cond_input)

        pred_a = self.backbone(emb_a, delta_cond)
        pred_b = self.backbone(emb_b, delta_cond)
        return pred_b - pred_a

    def aux_loss(self, edit_feats: torch.Tensor) -> torch.Tensor:
        """Edit feature reconstruction loss (call after forward)."""
        if self.aux_reconstructor is None or not hasattr(self, '_edit_enc'):
            return torch.tensor(0.0)
        pred_feats = self.aux_reconstructor(self._edit_enc)
        return F.mse_loss(pred_feats, edit_feats)


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 3: Fragment-Anchored FiLM
# ═══════════════════════════════════════════════════════════════════════════

class FragAnchoredFiLMDeltaMLP(nn.Module):
    """Fully scaffold-independent conditioning via fragment FP deltas.

    Replaces mol difference with:
      frag_delta = fp(incoming_frag) - fp(leaving_frag)  [scaffold-independent]
      edit_feats = 28-dim handcrafted features

    Same F→Cl edit produces the same conditioning regardless of scaffold.

    forward(emb_a, emb_b, frag_delta, edit_feats) → predicted delta
    """

    def __init__(self, mol_dim: int, frag_dim: int = 1024,
                 edit_feat_dim: int = 28,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()
        self.mol_dim = mol_dim

        cond_hidden = max(mol_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(frag_dim + edit_feat_dim, cond_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = FiLMPredictionBackbone(
            mol_dim, cond_hidden, hidden_dims, dropout)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                frag_delta: torch.Tensor,
                edit_feats: torch.Tensor) -> torch.Tensor:
        edit_signal = torch.cat([frag_delta, edit_feats], dim=-1)
        delta_cond = self.delta_encoder(edit_signal)

        pred_a = self.backbone(emb_a, delta_cond)
        pred_b = self.backbone(emb_b, delta_cond)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 4: Multi-Modal Edit Encoder
# ═══════════════════════════════════════════════════════════════════════════

class MultiModalEditFiLMDeltaMLP(nn.Module):
    """Fuses ALL available edit representations via MLP (no attention).

    Four parallel streams:
      - DRFP (2048d) → 128d
      - Fragment FP delta (1024d) → 128d
      - Edit features (28d) → 64d
      - rxnfp (256d) → 64d (optional)

    Concatenated and fused via MLP into edit_enc, then FiLM conditioning.

    forward(emb_a, emb_b, drfp, frag_delta, edit_feats, rxnfp=None) → delta
    """

    def __init__(self, mol_dim: int, drfp_dim: int = 2048,
                 frag_dim: int = 1024, edit_feat_dim: int = 28,
                 rxnfp_dim: int = 256, use_rxnfp: bool = False,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2,
                 aux_feat_weight: float = 0.1,
                 aux_symmetry_weight: float = 0.05):
        super().__init__()
        self.mol_dim = mol_dim
        self.use_rxnfp = use_rxnfp
        self.aux_feat_weight = aux_feat_weight
        self.aux_symmetry_weight = aux_symmetry_weight
        self.edit_feat_dim = edit_feat_dim

        # Four parallel streams
        self.drfp_stream = nn.Sequential(
            nn.Linear(drfp_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.frag_stream = nn.Sequential(
            nn.Linear(frag_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.feat_stream = nn.Sequential(
            nn.Linear(edit_feat_dim, 64), nn.ReLU(), nn.Dropout(dropout))

        fusion_dim = 128 + 128 + 64
        if use_rxnfp:
            self.rxnfp_stream = nn.Sequential(
                nn.Linear(rxnfp_dim, 64), nn.ReLU(), nn.Dropout(dropout))
            fusion_dim += 64

        edit_enc_dim = 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, edit_enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        cond_hidden = max(mol_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(edit_enc_dim, cond_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = FiLMPredictionBackbone(
            mol_dim, cond_hidden, hidden_dims, dropout)

        # Auxiliary: edit feature reconstruction
        self.aux_feat_head = nn.Linear(edit_enc_dim, edit_feat_dim)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                drfp: torch.Tensor, frag_delta: torch.Tensor,
                edit_feats: torch.Tensor,
                rxnfp: Optional[torch.Tensor] = None) -> torch.Tensor:
        e_drfp = self.drfp_stream(drfp)
        e_frag = self.frag_stream(frag_delta)
        e_feat = self.feat_stream(edit_feats)

        streams = [e_drfp, e_frag, e_feat]
        if self.use_rxnfp and rxnfp is not None:
            streams.append(self.rxnfp_stream(rxnfp))

        edit_enc = self.fusion(torch.cat(streams, dim=-1))
        self._edit_enc = edit_enc

        delta_cond = self.delta_encoder(edit_enc)
        pred_a = self.backbone(emb_a, delta_cond)
        pred_b = self.backbone(emb_b, delta_cond)
        return pred_b - pred_a

    def aux_loss(self, edit_feats: torch.Tensor) -> torch.Tensor:
        """Combined auxiliary losses: feature reconstruction + symmetry."""
        loss = torch.tensor(0.0, device=edit_feats.device)
        if hasattr(self, '_edit_enc'):
            # Feature reconstruction
            pred_feats = self.aux_feat_head(self._edit_enc)
            loss = loss + self.aux_feat_weight * F.mse_loss(pred_feats, edit_feats)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 5: Edit Hypernetwork with Low-Rank Generation
# ═══════════════════════════════════════════════════════════════════════════

class LoRAFiLMBlock(nn.Module):
    """FiLM block where the edit generates low-rank weight perturbations.

    For each layer, the edit encoding generates A and B matrices (rank r)
    such that W_delta = A @ B. The effective weight is W_base + scale * W_delta.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 edit_enc_dim: int, rank: int = 8,
                 lora_scale: float = 0.1, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.lora_scale = lora_scale

        self.base_linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # LoRA generators from edit encoding
        self.gen_A = nn.Linear(edit_enc_dim, hidden_dim * rank)
        self.gen_B = nn.Linear(edit_enc_dim, rank * input_dim)

        # FiLM scale/shift (still used alongside LoRA)
        self.gamma_proj = nn.Linear(edit_enc_dim, hidden_dim)
        self.beta_proj = nn.Linear(edit_enc_dim, hidden_dim)

        # Initialize small
        nn.init.zeros_(self.gen_A.weight)
        nn.init.zeros_(self.gen_B.weight)
        nn.init.xavier_uniform_(self.gamma_proj.weight, gain=0.1)
        nn.init.constant_(self.gamma_proj.bias, 1.0)
        nn.init.xavier_uniform_(self.beta_proj.weight, gain=0.1)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, edit_enc: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Base transformation
        h = self.base_linear(x)

        # LoRA perturbation: edit_enc → A, B → W_delta = A @ B → h_delta = x @ W_delta^T
        A = self.gen_A(edit_enc).view(B, self.hidden_dim, self.rank)  # [B, H, r]
        B_mat = self.gen_B(edit_enc).view(B, self.rank, x.shape[1])  # [B, r, D]

        # h_delta = (A @ B_mat @ x^T)^T per sample → bmm
        h_delta = torch.bmm(A, torch.bmm(B_mat, x.unsqueeze(-1))).squeeze(-1)
        h = h + self.lora_scale * h_delta

        # FiLM modulation on top
        gamma = self.gamma_proj(edit_enc)
        beta = self.beta_proj(edit_enc)
        h = gamma * self.activation(h) + beta

        return self.dropout(h)


class EditHypernetFiLMDeltaMLP(nn.Module):
    """Edit-as-instruction: edit encoding generates LoRA weight perturbations
    per FiLM layer, so different edits get different weight matrices.

    Takes the best edit encoder from iterations 1-4 as input.

    forward(emb_a, emb_b, edit_enc) → predicted delta
    where edit_enc is pre-computed by the best edit encoder.
    """

    def __init__(self, mol_dim: int, edit_enc_dim: int = 256,
                 hidden_dims: Optional[List[int]] = None,
                 rank: int = 8, lora_scale: float = 0.1,
                 dropout: float = 0.2):
        super().__init__()
        self.mol_dim = mol_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.blocks = nn.ModuleList()
        prev_dim = mol_dim
        for h in hidden_dims:
            self.blocks.append(LoRAFiLMBlock(
                prev_dim, h, edit_enc_dim, rank, lora_scale, dropout))
            prev_dim = h

        self.output = nn.Linear(prev_dim, 1)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                edit_enc: torch.Tensor) -> torch.Tensor:
        def forward_single(x, enc):
            h = x
            for block in self.blocks:
                h = block(h, enc)
            return self.output(h).squeeze(-1)

        pred_a = forward_single(emb_a, edit_enc)
        pred_b = forward_single(emb_b, edit_enc)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Target-Conditioned FiLM
# ═══════════════════════════════════════════════════════════════════════════

class TargetCondFiLMDeltaMLP(nn.Module):
    """FiLMDelta with target identity conditioning.

    Concatenates a learned target embedding with the delta encoding before
    FiLM projection. This allows the model to learn target-specific SAR rules:
    the same F→Cl edit affects kinases differently than GPCRs.

    forward(emb_a, emb_b, target_ids) → predicted delta

    Args:
        input_dim: Molecule embedding dimension
        n_targets: Number of unique targets (for embedding table)
        target_emb_dim: Dimension of target embedding (default 64)
        hidden_dims: FiLM backbone hidden dimensions
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, n_targets: int,
                 target_emb_dim: int = 64,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.target_emb_dim = target_emb_dim

        # Target embedding lookup
        self.target_embedding = nn.Embedding(n_targets, target_emb_dim)
        nn.init.normal_(self.target_embedding.weight, std=0.01)

        # Delta encoder: transform delta before using for conditioning
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fuse delta_cond + target_emb → final conditioning
        cond_dim = delta_hidden + target_emb_dim
        fused_cond_dim = delta_hidden  # keep same dim as FiLMDelta for backbone
        self.cond_fuser = nn.Sequential(
            nn.Linear(cond_dim, fused_cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = FiLMPredictionBackbone(
            input_dim, fused_cond_dim, hidden_dims, dropout)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a: [batch, input_dim]
            emb_b: [batch, input_dim]
            target_ids: [batch] long tensor of target indices
        """
        delta = emb_b - emb_a
        delta_cond = self.delta_encoder(delta)

        target_emb = self.target_embedding(target_ids)  # [batch, target_emb_dim]
        fused = torch.cat([delta_cond, target_emb], dim=-1)
        cond = self.cond_fuser(fused)

        pred_a = self.backbone(emb_a, cond)
        pred_b = self.backbone(emb_b, cond)
        return pred_b - pred_a
