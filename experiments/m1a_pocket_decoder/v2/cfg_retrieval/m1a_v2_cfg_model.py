"""M1a v2 model with Classifier-Free Guidance + optional retrieval prefix.

Architecture is identical to the covFT + pocket/pose conditioning v2 model
(experiments/m1a_pocket_decoder/v2/fixes/m1a_v2_model.py).  Two additions:

  1. CFG null token: a single learned vector of shape (D_MODEL,) that
     REPLACES BOTH the pocket and pose conditioning tokens with a single
     unconditional token when guidance is dropped.  The drop is applied
     together for pocket+pose (never independently), as specified.

     During training, with probability p_drop (default 0.15) the whole
     conditioning is dropped for a training sample; the collator marks
     samples with a mask which likelihood() consumes.  During sampling we
     compute both branches and combine logits: s * cond + (1-s) * uncond.

  2. Retrieval prefix: 0..K extra encoder-memory tokens injected *before*
     the [POCKET]/[POSE] tokens.  These are produced by re-embedding retrieved
     SMILES through the base mol2mol encoder and mean-pooling their memory
     to a single per-SMILES vector (already in D_MODEL space).  This makes
     the retrieval prefix use exactly the same latent geometry as the target
     encoder memory.

The wrapper stores pose_mean/pose_std buffers exactly like the v2 model so
inference normalization matches training.  A `null_emb` (D_MODEL,) parameter
is added.
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
POSE_DIM = 3


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


class M1aV2CfgModel(nn.Module):
    """v2 conditioned mol2mol + CFG null token + optional retrieval prefix.

    Attribute compatibility with the v2 M1aV2ConditionedModel wrapper:
      - .base  -> Mol2MolModel
      - .pocket_enc, .pose_enc, .pose_mean, .pose_std
      - .likelihood(...) / .sample_multinomial(...)

    New:
      - .null_emb  (D_MODEL,) learnable
      - .likelihood(..., drop_cond=BoolTensor(B,))
          If drop_cond[i] is True, the conditioning tokens for sample i are
          replaced by a single [null_emb] token (memory length shrinks by 1).
          To keep batching simple we still emit 2 conditioning tokens per
          sample; the second null token is a copy of null_emb.  Both attend
          to the same learned "uncond" vector — equivalent to a single
          repeated null token.
      - .sample_multinomial_cfg(..., cfg_scale, retrieval_prefix_embs)
          Runs two forward passes per token and combines logits.
    """

    def __init__(self, base_model: Mol2MolModel,
                  pose_mean: torch.Tensor | None = None,
                  pose_std: torch.Tensor | None = None,
                  init_from_v2: dict | None = None):
        super().__init__()
        self.base = base_model
        self.pocket_enc = PocketEncoder()
        self.pose_enc = WarheadPoseEncoder()
        if pose_mean is None:
            pose_mean = torch.zeros(POSE_DIM)
        if pose_std is None:
            pose_std = torch.ones(POSE_DIM)
        self.register_buffer("pose_mean", pose_mean.float())
        self.register_buffer("pose_std", pose_std.float())
        # CFG null token: a single learnable vector in D_MODEL space.
        # Initialize small so it starts near the mean of pocket/pose tokens.
        self.null_emb = nn.Parameter(torch.zeros(D_MODEL))
        nn.init.normal_(self.null_emb, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.base.network.parameters()).device

    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        return (pose - self.pose_mean) / self.pose_std.clamp_min(1e-6)

    def _cond_tokens(self, residue_emb, residue_mask, pose_norm,
                       drop_cond: torch.Tensor | None):
        """Return (B, 2, D) conditioning tokens.  If drop_cond[i], both slots
        become copies of self.null_emb."""
        B = residue_emb.shape[0]
        pocket_vec = self.pocket_enc(residue_emb, residue_mask)  # (B, D)
        pose_vec = self.pose_enc(pose_norm)                       # (B, D)
        cond = torch.stack([pocket_vec, pose_vec], dim=1)         # (B, 2, D)
        if drop_cond is not None:
            null = self.null_emb.view(1, 1, -1).expand(B, 2, -1)
            m = drop_cond.view(B, 1, 1).float()
            cond = (1.0 - m) * cond + m * null
        return cond

    def _build_memory(self, src, src_mask,
                        residue_emb, residue_mask, pose_norm,
                        drop_cond: torch.Tensor | None,
                        retrieval_prefix_embs: torch.Tensor | None,
                        retrieval_prefix_mask: torch.Tensor | None):
        """Build encoder memory + src_mask with (optional) retrieval prefix
        then pocket/pose conditioning tokens then encoded source memory.

        retrieval_prefix_embs: (B, K, D_MODEL) or None
        retrieval_prefix_mask: (B, K) bool (True = valid) or None
        """
        memory = self.base.network.encoder(
            self.base.network.src_embed(src), src_mask)  # (B, L, D)
        B, L, D = memory.shape
        cond = self._cond_tokens(residue_emb, residue_mask, pose_norm, drop_cond)
        pieces = []
        mask_pieces = []
        if retrieval_prefix_embs is not None and retrieval_prefix_embs.shape[1] > 0:
            K = retrieval_prefix_embs.shape[1]
            pieces.append(retrieval_prefix_embs)
            # Build (B, 1, K) mask matching src_mask 3D convention below.
            if retrieval_prefix_mask is None:
                retrieval_prefix_mask = torch.ones(B, K,
                    dtype=torch.bool, device=src.device)
            mask_pieces.append(retrieval_prefix_mask.long())
        pieces.append(cond)
        cond_mask = torch.ones(B, 2, dtype=torch.long, device=src.device)
        mask_pieces.append(cond_mask)
        pieces.append(memory)
        # src_mask is (B, 1, L) long; extract (B, L).
        if src_mask.dim() == 3:
            base_mask_1d = src_mask.squeeze(1).long()
        else:
            base_mask_1d = src_mask.long()
        mask_pieces.append(base_mask_1d)
        memory_ext = torch.cat(pieces, dim=1)                # (B, K+2+L, D)
        mask_1d = torch.cat(mask_pieces, dim=1)               # (B, K+2+L)
        # Keep the (B, 1, T) shape that reinvent's decoder expects for src.
        src_mask_ext = mask_1d.unsqueeze(1)
        return memory_ext, src_mask_ext

    def likelihood(self, src, src_mask, trg, trg_mask,
                    residue_emb, residue_mask, pose,
                    drop_cond: torch.Tensor | None = None,
                    retrieval_prefix_embs: torch.Tensor | None = None,
                    retrieval_prefix_mask: torch.Tensor | None = None):
        memory_ext, src_mask_ext = self._build_memory(
            src, src_mask, residue_emb, residue_mask, pose,
            drop_cond, retrieval_prefix_embs, retrieval_prefix_mask)
        trg_y = trg[:, 1:]
        trg_in = trg[:, :-1]
        out = self.base.network.decoder(
            self.base.network.tgt_embed(trg_in), memory_ext, src_mask_ext, trg_mask)
        log_prob = self.base.network.generator(
            out, self.base.temperature).transpose(1, 2)
        nll = self.base._nll_loss(log_prob, trg_y).sum(dim=1)
        return nll

    @torch.no_grad()
    def encode_smiles_to_prefix_embs(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode a list of SMILES through the base mol2mol encoder and
        mean-pool each into a single (D_MODEL,) vector.  Returns (K, D_MODEL).
        Invalid SMILES are silently dropped."""
        vocab = self.base.vocabulary
        tok = self.base.tokenizer
        seqs = []
        for smi in smiles_list:
            try:
                s = tok.tokenize(smi)
                enc = vocab.encode(s)
                if len(enc) < 2 or len(enc) > 128:
                    continue
                seqs.append(enc)
            except Exception:
                continue
        if len(seqs) == 0:
            return torch.zeros(0, D_MODEL, device=self.device)
        import numpy as np
        L = max(len(s) for s in seqs)
        arr = np.zeros((len(seqs), L), dtype=np.int64)
        for j, s in enumerate(seqs):
            arr[j, :len(s)] = s
        src = torch.from_numpy(arr).to(self.device)
        src_mask = (src != 0).unsqueeze(-2).long()
        mem = self.base.network.encoder(
            self.base.network.src_embed(src), src_mask)  # (K, L, D)
        mask_f = (src != 0).float().unsqueeze(-1)
        pooled = (mem * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)
        return pooled

    @torch.no_grad()
    def sample_multinomial_cfg(self, src, src_mask,
                                residue_emb, residue_mask, pose,
                                cfg_scale: float = 1.0,
                                retrieval_prefix_embs: torch.Tensor | None = None,
                                max_length: int = 128,
                                temperature: float = 1.0):
        """CFG sampling.  If cfg_scale == 1.0, only the conditional branch is
        used (equivalent to baseline sampling).  Otherwise both branches are
        computed per token and combined:
            logit_final = logit_uncond + cfg_scale * (logit_cond - logit_uncond)

        retrieval_prefix_embs: (K, D_MODEL) tensor on device, or None.  The
        same prefix is broadcast across the batch.  The prefix is used for
        BOTH branches (retrieval prefix is not part of the CFG drop).
        """
        from torch.autograd import Variable
        from reinvent.models.transformer.core.network.module.subsequent_mask \
            import subsequent_mask

        B = src.shape[0]
        # Broadcast retrieval prefix.
        prefix_embs = None
        prefix_mask = None
        if retrieval_prefix_embs is not None and retrieval_prefix_embs.numel() > 0:
            K = retrieval_prefix_embs.shape[0]
            prefix_embs = retrieval_prefix_embs.unsqueeze(0).expand(B, K, -1)
            prefix_mask = torch.ones(B, K, dtype=torch.bool, device=src.device)

        drop_none = torch.zeros(B, dtype=torch.bool, device=src.device)
        drop_all = torch.ones(B, dtype=torch.bool, device=src.device)

        mem_c, msk_c = self._build_memory(
            src, src_mask, residue_emb, residue_mask, pose,
            drop_none, prefix_embs, prefix_mask)
        if cfg_scale != 1.0:
            mem_u, msk_u = self._build_memory(
                src, src_mask, residue_emb, residue_mask, pose,
                drop_all, prefix_embs, prefix_mask)

        ys = torch.ones(1, device=src.device).repeat(B, 1).long()
        break_cond = torch.zeros(B, dtype=torch.bool, device=src.device)
        nlls = torch.zeros(B, device=src.device)
        end_token = self.base.vocabulary["$"]

        for i in range(max_length - 1):
            tgt_msk = Variable(subsequent_mask(ys.size(1)).type_as(src))
            out_c = self.base.network.decode(mem_c, msk_c, Variable(ys), tgt_msk)
            log_prob_c = self.base.network.generator(out_c[:, -1], temperature)
            if cfg_scale == 1.0:
                log_prob = log_prob_c
            else:
                out_u = self.base.network.decode(mem_u, msk_u, Variable(ys), tgt_msk)
                log_prob_u = self.base.network.generator(out_u[:, -1], temperature)
                # Combine in logit space and renormalize.
                fused = log_prob_u + cfg_scale * (log_prob_c - log_prob_u)
                log_prob = fused - torch.logsumexp(fused, dim=-1, keepdim=True)
            prob = torch.exp(log_prob)
            mask_prop = self.base.mask_property_tokens(B)
            prob = prob.masked_fill(mask_prop, 0)
            # Numerical guard against all-zero rows (can happen at high s).
            row_sum = prob.sum(dim=-1, keepdim=True)
            prob = torch.where(row_sum > 1e-8, prob,
                                torch.ones_like(prob) / prob.shape[-1])
            next_word = torch.multinomial(prob, 1)
            break_t = torch.unsqueeze(break_cond, 1)
            next_word = next_word.masked_fill(break_t, 0)
            ys = torch.cat([ys, next_word], dim=1)
            nw = next_word.reshape(-1)
            # NLL under the CONDITIONAL branch (audit metric); this is not the
            # CFG-fused log prob because we want a stable per-sample score.
            nlls += self.base._nll_loss(log_prob_c, nw)
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


def load_cfg_model(prior_path: str, device: torch.device,
                     ckpt_path: str | None = None,
                     pose_mean=None, pose_std=None,
                     init_from_v2_ckpt: str | None = None,
                     strict: bool = True) -> M1aV2CfgModel:
    """Load M1aV2CfgModel.

    If `init_from_v2_ckpt` is given, load the v2 wrapper state dict INTO the
    CFG wrapper (non-strict; null_emb stays at its Gaussian init).  Used to
    warm-start CFG training from the v2 covFT-trained checkpoint.

    If `ckpt_path` is given, load a CFG-formatted checkpoint (strict).
    """
    base = load_base_model(prior_path, device)
    pm = torch.as_tensor(pose_mean, dtype=torch.float32) if pose_mean is not None else None
    ps = torch.as_tensor(pose_std, dtype=torch.float32) if pose_std is not None else None
    wrapper = M1aV2CfgModel(base, pm, ps).to(device)
    if init_from_v2_ckpt is not None and Path(init_from_v2_ckpt).exists():
        sd = torch.load(init_from_v2_ckpt, map_location=device, weights_only=False)
        v2_sd = sd["model_state"] if "model_state" in sd else sd
        # v2 state dict has no 'null_emb' key.  Load non-strict.
        missing, unexpected = wrapper.load_state_dict(v2_sd, strict=False)
        # Only allow null_emb to be missing.
        allowed_missing = {"null_emb"}
        real_missing = [k for k in missing if k not in allowed_missing]
        if real_missing:
            raise RuntimeError(f"Missing keys not in {allowed_missing}: {real_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = sd["model_state"] if "model_state" in sd else sd
        wrapper.load_state_dict(state, strict=strict)
    return wrapper


def save_cfg_model(wrapper: M1aV2CfgModel, ckpt_path: str,
                     extra: dict | None = None) -> None:
    payload = {"model_state": wrapper.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)
