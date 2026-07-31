"""M1a pocket+warhead-pose conditioned mol2mol decoder.

Wraps the REINVENT4 Mol2MolModel by:
  - keeping its EncoderDecoder untouched
  - adding a small pocket Transformer encoder (3-layer) over ESM-2 residue
    embeddings (320-dim) -> 256-dim pocket vector
  - adding a 6 -> 64 -> 256 MLP for the warhead pose feature
  - prepending [pocket_vec, warhead_vec] (2 tokens, 256-d) to the encoder
    memory before the decoder cross-attends.

The base mol2mol prior is loaded as-is and continues to be fine-tuned (small lr).
The new modules are trained from random init.
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

REINVENT4_ROOT = Path("/home/shaharh_quris_ai/REINVENT4")
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel  # noqa: E402
from reinvent.models.model_mode_enum import ModelModeEnum  # noqa: E402


D_MODEL = 256
ESM_DIM = 320  # esm2_t6_8M_UR50D
N_POCKET_TR_LAYERS = 3
N_POCKET_TR_HEADS = 4
POSE_DIM = 6


class PocketEncoder(nn.Module):
    """3-layer Transformer encoder + mean-pool over residue tokens.

    Input:  (B, R, ESM_DIM) residue embeddings, (B, R) bool mask (True=real)
    Output: (B, D_MODEL) pocket vector
    """

    def __init__(self, d_in: int = ESM_DIM, d_model: int = D_MODEL,
                  n_layers: int = N_POCKET_TR_LAYERS,
                  n_heads: int = N_POCKET_TR_HEADS,
                  dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, residue_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # residue_emb: (B, R, ESM_DIM); mask: (B, R) bool, True = real
        x = self.proj(residue_emb)  # (B, R, D)
        # nn.TransformerEncoder uses src_key_padding_mask = True for *PAD*
        pad_mask = ~mask  # True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # masked mean pool
        m = mask.float().unsqueeze(-1)  # (B, R, 1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        pooled = (x * m).sum(dim=1) / denom  # (B, D)
        return self.out_norm(pooled)


class WarheadPoseEncoder(nn.Module):
    """6-d pose feature -> 256-d via 6 -> 64 -> 256."""

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
        # pose: (B, POSE_DIM)
        return self.out_norm(self.net(pose))


class M1aConditionedModel(nn.Module):
    """Wraps a Mol2MolModel; adds 2 conditioning tokens to its encoder memory."""

    def __init__(self, base_model: Mol2MolModel):
        super().__init__()
        self.base = base_model  # has .network (nn.Module) and .vocabulary/.tokenizer
        self.pocket_enc = PocketEncoder()
        self.pose_enc = WarheadPoseEncoder()

    @property
    def network(self) -> nn.Module:
        return self.base.network

    @property
    def device(self) -> torch.device:
        return next(self.base.network.parameters()).device

    def parameters(self, recurse: bool = True):
        # nn.Module.parameters() already iterates base + pocket_enc + pose_enc
        return super().parameters(recurse)

    def _build_conditioned_memory(self, src, src_mask,
                                    residue_emb, residue_mask, pose):
        """Run encoder on src, prepend 2 condition tokens.

        Returns: memory (B, 2+L, D), extended_src_mask (B, 1, 2+L)
        """
        # src_embed -> encoder -> memory
        memory = self.base.network.encoder(self.base.network.src_embed(src), src_mask)
        # (B, L, D)
        B, L, D = memory.shape
        pocket_vec = self.pocket_enc(residue_emb, residue_mask)  # (B, D)
        pose_vec = self.pose_enc(pose)  # (B, D)
        cond = torch.stack([pocket_vec, pose_vec], dim=1)  # (B, 2, D)
        memory_ext = torch.cat([cond, memory], dim=1)  # (B, 2+L, D)
        # Extend src_mask: (B, 1, L) -> (B, 1, 2+L)
        if src_mask.dim() == 2:
            extra = torch.ones(B, 2, dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=1)
        elif src_mask.dim() == 3:
            # (B, 1, L)
            extra = torch.ones(B, src_mask.shape[1], 2,
                                dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=2)
        else:
            raise ValueError(f"Unexpected src_mask shape: {src_mask.shape}")
        return memory_ext, mask_ext

    def likelihood(self, src, src_mask, trg, trg_mask,
                    residue_emb, residue_mask, pose):
        """Conditional NLL: -log P(trg | src, pocket, pose)."""
        # Build conditioned memory
        memory_ext, src_mask_ext = self._build_conditioned_memory(
            src, src_mask, residue_emb, residue_mask, pose)
        # Decode
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
        """Multinomial sampling. Returns (output_smiles, nlls)."""
        from torch.autograd import Variable
        from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask
        memory_ext, src_mask_ext = self._build_conditioned_memory(
            src, src_mask, residue_emb, residue_mask, pose)
        batch_size = src.shape[0]
        ys = torch.ones(1, device=src.device).repeat(batch_size, 1).long()  # start tok 1
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
        # Decode
        from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
        tok = SMILESTokenizer()
        out_smiles = [tok.untokenize(self.base.vocabulary.decode(seq))
                       for seq in ys.detach().cpu().numpy()]
        return out_smiles, nlls.detach().cpu().numpy()


def load_base_model(prior_path: str, device: torch.device) -> Mol2MolModel:
    """Load mol2mol from a .prior checkpoint."""
    save_dict = torch.load(prior_path, map_location=device, weights_only=False)
    model = Mol2MolModel.create_from_dict(save_dict, ModelModeEnum().TRAINING, device)
    return model


def load_m1a(prior_path: str, device: torch.device,
              ckpt_path: str | None = None) -> M1aConditionedModel:
    base = load_base_model(prior_path, device)
    wrapper = M1aConditionedModel(base).to(device)
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # New keys: pocket_enc.*, pose_enc.*; base.* (network weights)
        wrapper.load_state_dict(sd, strict=True)
    return wrapper


def save_m1a(wrapper: M1aConditionedModel, ckpt_path: str) -> None:
    torch.save(wrapper.state_dict(), ckpt_path)
