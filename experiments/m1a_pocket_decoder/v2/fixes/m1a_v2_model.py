"""M1a v2 pocket+warhead-pose conditioned mol2mol decoder.

Same architecture as v1 (m1a_model.py) but:
  - Persists a pose normalizer (z-score mean/std) inside the checkpoint so
    inference uses identical normalization (Fix 3).
  - Pose vector schema (v3 — 3-dim rotation/translation-invariant):
       [d_b_nuc_A, bd_angle_deg, planar_vinylamide_dihedral_deg]
    Each dim is an intrinsic geometric invariant (distance or angle between
    specified atoms), so the pose is rotation/translation invariant by
    construction — no local frame is built and Fix 2 holds trivially.
    Z-scoring is applied at training-time AND inference-time.

The wrapper exposes:
  - .pose_normalizer  = {'mean': (3,), 'std': (3,)} numpy arrays
  - .normalize_pose(p) and .denormalize_pose(p)  (numpy or torch)
  - save_m1a_v2()/load_m1a_v2() persist normalizer inside the state dict.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

REINVENT4_ROOT = Path("/home/shaharh_quris_ai/REINVENT4")
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel  # noqa: E402
from reinvent.models.model_mode_enum import ModelModeEnum  # noqa: E402


D_MODEL = 256
ESM_DIM = 320
N_POCKET_TR_LAYERS = 3
N_POCKET_TR_HEADS = 4
POSE_DIM = 3  # v3 invariant pose: (d_b_nuc, bd_angle_deg, planar_dihedral_deg). Was 6 in earlier v2.


class PocketEncoder(nn.Module):
    def __init__(self, d_in: int = ESM_DIM, d_model: int = D_MODEL,
                  n_layers: int = N_POCKET_TR_LAYERS,
                  n_heads: int = N_POCKET_TR_HEADS, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, residue_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(residue_emb)
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        pooled = (x * m).sum(dim=1) / denom
        return self.out_norm(pooled)


class WarheadPoseEncoder(nn.Module):
    def __init__(self, d_in: int = POSE_DIM, d_hidden: int = 64,
                  d_out: int = D_MODEL, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(pose))


class M1aV2ConditionedModel(nn.Module):
    """Wraps a Mol2MolModel; adds 2 conditioning tokens to encoder memory.

    Carries a pose normalizer as a non-parameter buffer so it persists with the
    checkpoint.
    """

    def __init__(self, base_model: Mol2MolModel,
                  pose_mean: torch.Tensor | None = None,
                  pose_std: torch.Tensor | None = None):
        super().__init__()
        self.base = base_model
        self.pocket_enc = PocketEncoder()
        self.pose_enc = WarheadPoseEncoder()
        if pose_mean is None:
            pose_mean = torch.zeros(POSE_DIM)
        if pose_std is None:
            pose_std = torch.ones(POSE_DIM)
        # Register as buffers so .state_dict() carries them
        self.register_buffer("pose_mean", pose_mean.float())
        self.register_buffer("pose_std", pose_std.float())

    @property
    def network(self) -> nn.Module:
        return self.base.network

    @property
    def device(self) -> torch.device:
        return next(self.base.network.parameters()).device

    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        # pose: (..., POSE_DIM)
        return (pose - self.pose_mean) / self.pose_std.clamp_min(1e-6)

    def denormalize_pose(self, pose_norm: torch.Tensor) -> torch.Tensor:
        return pose_norm * self.pose_std + self.pose_mean

    def _build_conditioned_memory(self, src, src_mask,
                                    residue_emb, residue_mask, pose_norm):
        memory = self.base.network.encoder(self.base.network.src_embed(src), src_mask)
        B, L, D = memory.shape
        pocket_vec = self.pocket_enc(residue_emb, residue_mask)
        pose_vec = self.pose_enc(pose_norm)
        cond = torch.stack([pocket_vec, pose_vec], dim=1)
        memory_ext = torch.cat([cond, memory], dim=1)
        if src_mask.dim() == 2:
            extra = torch.ones(B, 2, dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=1)
        elif src_mask.dim() == 3:
            extra = torch.ones(B, src_mask.shape[1], 2,
                                dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=2)
        else:
            raise ValueError(f"Unexpected src_mask shape: {src_mask.shape}")
        return memory_ext, mask_ext

    def likelihood(self, src, src_mask, trg, trg_mask,
                    residue_emb, residue_mask, pose):
        """pose may be ALREADY normalized (training path) or raw — caller decides.

        We DO NOT auto-normalize here: training loader applies normalization
        once into the cached tensor, inference scripts apply it explicitly.
        """
        memory_ext, src_mask_ext = self._build_conditioned_memory(
            src, src_mask, residue_emb, residue_mask, pose)
        trg_y = trg[:, 1:]
        trg_in = trg[:, :-1]
        out = self.base.network.decoder(
            self.base.network.tgt_embed(trg_in), memory_ext, src_mask_ext, trg_mask)
        log_prob = self.base.network.generator(out, self.base.temperature).transpose(1, 2)
        nll = self.base._nll_loss(log_prob, trg_y).sum(dim=1)
        return nll

    @torch.no_grad()
    def sample_multinomial(self, src, src_mask, residue_emb, residue_mask, pose,
                              max_length: int = 128, temperature: float = 1.0):
        from torch.autograd import Variable
        from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask
        memory_ext, src_mask_ext = self._build_conditioned_memory(
            src, src_mask, residue_emb, residue_mask, pose)
        batch_size = src.shape[0]
        ys = torch.ones(1, device=src.device).repeat(batch_size, 1).long()
        break_cond = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        nlls = torch.zeros(batch_size, device=src.device)
        end_token = self.base.vocabulary["$"]
        for i in range(max_length - 1):
            out = self.base.network.decode(
                memory_ext, src_mask_ext, Variable(ys),
                Variable(subsequent_mask(ys.size(1)).type_as(src)))
            log_prob = self.base.network.generator(out[:, -1], temperature)
            prob = torch.exp(log_prob)
            mask_prop = self.base.mask_property_tokens(batch_size)
            prob = prob.masked_fill(mask_prop, 0)
            next_word = torch.multinomial(prob, 1)
            break_t = torch.unsqueeze(break_cond, 1)
            next_word = next_word.masked_fill(break_t, 0)
            ys = torch.cat([ys, next_word], dim=1)
            nw = next_word.reshape(-1)
            nlls += self.base._nll_loss(log_prob, nw)
            break_cond = break_cond | (nw == end_token)
            if bool(break_cond.all()):
                break
        from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
        tok = SMILESTokenizer()
        out_smiles = [tok.untokenize(self.base.vocabulary.decode(seq))
                       for seq in ys.detach().cpu().numpy()]
        return out_smiles, nlls.detach().cpu().numpy()


def load_base_model(prior_path: str, device: torch.device) -> Mol2MolModel:
    save_dict = torch.load(prior_path, map_location=device, weights_only=False)
    model = Mol2MolModel.create_from_dict(save_dict, ModelModeEnum().TRAINING, device)
    return model


def load_m1a_v2(prior_path: str, device: torch.device,
                  ckpt_path: str | None = None,
                  pose_mean=None, pose_std=None) -> M1aV2ConditionedModel:
    base = load_base_model(prior_path, device)
    if pose_mean is not None:
        pm = torch.as_tensor(pose_mean, dtype=torch.float32)
    else:
        pm = None
    if pose_std is not None:
        ps = torch.as_tensor(pose_std, dtype=torch.float32)
    else:
        ps = None
    wrapper = M1aV2ConditionedModel(base, pm, ps).to(device)
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state" in sd:
            wrapper.load_state_dict(sd["model_state"], strict=True)
        else:
            wrapper.load_state_dict(sd, strict=True)
    return wrapper


def save_m1a_v2(wrapper: M1aV2ConditionedModel, ckpt_path: str,
                  extra: dict | None = None) -> None:
    payload = {"model_state": wrapper.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)
