"""
3D-aware surrogate that imitates FiLMDelta predictions.

Goal: produce a *differentiable* pIC50 estimator w.r.t. atom positions, so we
can compute ∇_x pIC50 for gradient-guided diffusion sampling (Path B1
ablation). The ground-truth signal is FiLMDelta's *predicted* delta on the
ZAP70 anchor set — NOT raw labels. We imitate the existing scoring pipeline.

Architecture
------------
A small SchNet-style continuous-filter convolutional network (CFConv) with a
manual O(N²) radius graph (small molecules, ~30-60 atoms — torch-cluster is
not required and is also a pain to install on macOS). Output is a scalar
graph embedding per molecule. Two molecules are encoded with shared weights
(siamese), then a small MLP combines [feat_a, feat_b, feat_b - feat_a] into a
scalar Δ pIC50.

Inputs per molecule:
    z   :  [N] long tensor of atomic numbers
    pos :  [N, 3] float tensor of 3D coordinates (in Å)
    batch: [N] long tensor of graph indices

Output: scalar Δ pIC50 (predicted b - a).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _radius_graph_dense(
    pos: torch.Tensor,
    batch: torch.Tensor,
    r: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """O(N²) radius graph — fine for small molecules.

    Returns (edge_index [2, E], edge_dist [E]). Self-loops excluded.
    """
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)            # [N, N, 3]
    dist = torch.linalg.norm(diff, dim=-1)                # [N, N]
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    eye = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    mask = (dist < r) & same_graph & ~eye
    edge_index = mask.nonzero(as_tuple=False).t()         # [2, E]
    edge_dist = dist[edge_index[0], edge_index[1]]
    return edge_index, edge_dist


class _GaussianSmearing(nn.Module):
    def __init__(self, start: float = 0.0, stop: float = 6.0, num_gaussians: int = 25):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        x = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * x.pow(2))


class _ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


class _CFConv(nn.Module):
    """Continuous-filter convolution from SchNet."""

    def __init__(self, hidden: int, num_gaussians: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden, bias=False)
        self.lin2 = nn.Linear(hidden, hidden)
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden),
            _ShiftedSoftplus(),
            nn.Linear(hidden, hidden),
        )
        self.act = _ShiftedSoftplus()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_basis: torch.Tensor,
    ) -> torch.Tensor:
        # Continuous filter
        W = self.filter_net(edge_basis)             # [E, hidden]
        x = self.lin1(h)                            # [N, hidden]
        # Source features times filter, scattered to destination
        src, dst = edge_index[0], edge_index[1]
        msg = x[src] * W                            # [E, hidden]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        out = self.lin2(agg)
        return self.act(out)


class _Interaction(nn.Module):
    def __init__(self, hidden: int, num_gaussians: int):
        super().__init__()
        self.cfconv = _CFConv(hidden, num_gaussians)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_basis: torch.Tensor,
    ) -> torch.Tensor:
        return h + self.cfconv(h, edge_index, edge_basis)


class _SchNetEncoder(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 25,
        cutoff: float = 6.0,
        out_dim: int = 128,
        max_z: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(max_z, hidden)
        self.smearing = _GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList(
            [_Interaction(hidden, num_gaussians) for _ in range(num_interactions)]
        )
        self.lin1 = nn.Linear(hidden, hidden // 2)
        self.act = _ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden // 2, out_dim)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(z)
        edge_index, edge_dist = _radius_graph_dense(pos, batch, self.cutoff)
        edge_basis = self.smearing(edge_dist)
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_basis)
        h = self.act(self.lin1(h))
        h = self.lin2(h)                                  # [N, out_dim]
        # Mean-pool per graph
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, h)
        counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        out = out / counts.clamp(min=1.0).unsqueeze(-1)
        return out                                        # [num_graphs, out_dim]


class FiLMDelta3DSurrogate(nn.Module):
    """Siamese SchNet surrogate for FiLMDelta predictions.

    Args:
        hidden: SchNet hidden width.
        num_interactions: number of message-passing layers.
        cutoff: radius cutoff (Å).
        feature_dim: graph-level embedding dim.
        head_hidden: hidden width of the delta MLP head.
        dropout: dropout in the head.
    """

    def __init__(
        self,
        hidden: int = 128,
        num_interactions: int = 3,
        cutoff: float = 6.0,
        feature_dim: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = _SchNetEncoder(
            hidden=hidden,
            num_interactions=num_interactions,
            num_gaussians=25,
            cutoff=cutoff,
            out_dim=feature_dim,
        )
        self.feature_dim = feature_dim
        self.head = nn.Sequential(
            nn.Linear(3 * feature_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def encode(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(z, pos, batch)

    def forward(
        self,
        z_a: torch.Tensor,
        pos_a: torch.Tensor,
        batch_a: torch.Tensor,
        z_b: torch.Tensor,
        pos_b: torch.Tensor,
        batch_b: torch.Tensor,
    ) -> torch.Tensor:
        feat_a = self.encode(z_a, pos_a, batch_a)
        feat_b = self.encode(z_b, pos_b, batch_b)
        h = torch.cat([feat_a, feat_b, feat_b - feat_a], dim=-1)
        return self.head(h).squeeze(-1)

    def predict_abs_pIC50(
        self,
        z_cand: torch.Tensor,
        pos_cand: torch.Tensor,
        anchor_z_list: List[torch.Tensor],
        anchor_pos_list: List[torch.Tensor],
        anchor_pIC50: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor-based absolute pIC50 prediction (matches FiLMDelta scorer).

        For a single candidate, average over all anchors:
            pred = mean_i ( anchor_pIC50[i] + surrogate_delta(anchor_i, cand) )

        ``pos_cand`` may require_grad; the result is differentiable w.r.t. it.
        """
        # Build a single batched candidate graph copied N_anchors times, plus a
        # batched anchor graph stacked across the anchor list. Loop is fine for
        # 280 anchors and small molecules; CPU runtime is tens of ms.
        n_anchors = len(anchor_z_list)
        deltas = []
        for i in range(n_anchors):
            za = anchor_z_list[i]
            pa = anchor_pos_list[i]
            ba = torch.zeros(za.shape[0], dtype=torch.long, device=za.device)
            zb = z_cand
            pb = pos_cand
            bb = torch.zeros(zb.shape[0], dtype=torch.long, device=zb.device)
            d = self.forward(za, pa, ba, zb, pb, bb)
            deltas.append(d)
        deltas = torch.stack(deltas, dim=0).squeeze(-1)
        abs_preds = anchor_pIC50.to(deltas.device) + deltas
        return abs_preds.mean()
