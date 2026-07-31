"""M1a v2 with per-layer FiLM conditioning (v4 architecture).

Change vs v3 (CFG-token) approach:
  - CFG-token relied on encoder-memory cross-attention alone; v3 showed
    this coupling is too weak for CFG guidance to steer generation
    (see cfg_retrieval_report_v3.md).
  - v4: keep the memory-token pathway (belt-and-suspenders) AND add
    per-layer FiLM at the output of every decoder block.

Architecture additions on top of v3 M1aV2CfgModel:
  1. `ConditionPooler`: takes pocket_vec (D=256) and pose_vec (D=256), concats
     and MLPs to a single conditioning vector `c` of size D=256.
  2. `LayerFiLM`: for each decoder layer, a small MLP `(256 -> d_model)` for
     `gamma` (initialised to ZERO so at init FiLM is identity), and another
     for `beta` (also zero-init).  Applied at the OUTPUT of each decoder
     block: `h' = h * (1 + gamma(c)) + beta(c)`.
  3. `film_decoder_forward`: replaces the decoder's forward pass to apply
     the per-layer FiLM.  We monkey-patch the decoder at wrapper init time
     (safe because the wrapper owns the base model's lifetime).

Everything else — pocket_enc, pose_enc, memory-token conditioning, sampling
loop — is inherited unchanged.  This means at init the model is EXACTLY the
v3 model (zero FiLM); training warms the FiLM modules to do the work.

Training uses the base + new-modules two-LR scheme from v3 (base LR=1e-5,
new LR=5e-5).  No CFG dropout — FiLM doesn't need it.

The wrapper still supports the CFG null-token drop path for A/B testing,
but the default training doesn't use it.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

REINVENT4_ROOT = Path("/home/shaharh_quris_ai/REINVENT4")
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel  # noqa
from reinvent.models.model_mode_enum import ModelModeEnum  # noqa


D_MODEL = 256
ESM_DIM = 320
N_POCKET_TR_LAYERS = 3
N_POCKET_TR_HEADS = 4
POSE_DIM = 3


class PocketEncoder(nn.Module):
    def __init__(self, d_in=ESM_DIM, d_model=D_MODEL,
                  n_layers=N_POCKET_TR_LAYERS, n_heads=N_POCKET_TR_HEADS, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, residue_emb, mask):
        x = self.proj(residue_emb)
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        pooled = (x * m).sum(dim=1) / denom
        return self.out_norm(pooled)


class WarheadPoseEncoder(nn.Module):
    def __init__(self, d_in=POSE_DIM, d_hidden=64, d_out=D_MODEL, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, pose):
        return self.out_norm(self.net(pose))


class ConditionPooler(nn.Module):
    """Pool [POCKET] and [POSE] into a single conditioning vector `c`.

    Concat (256 + 256) -> MLP (512 -> 256).  LayerNorm on output.
    """
    def __init__(self, d_model=D_MODEL, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, pocket_vec, pose_vec):
        x = torch.cat([pocket_vec, pose_vec], dim=-1)
        return self.out_norm(self.net(x))


class LayerFiLM(nn.Module):
    """FiLM parameters (gamma, beta) for one decoder layer.

    gamma_mlp: 256 -> d_model. beta_mlp: 256 -> d_model.
    ZERO-INIT both output layers so at init: gamma = 0, beta = 0,
    thus `h' = h * (1 + 0) + 0 = h` (identity).
    """
    def __init__(self, d_cond=D_MODEL, d_model=D_MODEL, dropout=0.0):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_model),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_model),
        )
        # ZERO-INIT the output projections so FiLM starts as identity.
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(self, h, c):
        """h: (B, T, d_model), c: (B, d_cond).  Returns FiLM(h)."""
        gamma = self.gamma_mlp(c).unsqueeze(1)   # (B, 1, d_model)
        beta = self.beta_mlp(c).unsqueeze(1)     # (B, 1, d_model)
        return h * (1.0 + gamma) + beta


class M1aV2FiLMModel(nn.Module):
    """Wraps a Mol2MolModel; adds:
      - pocket_enc, pose_enc (same as v3)
      - null_emb (D_MODEL,) for optional CFG dropout (unused by v4 default)
      - condition_pooler
      - LayerFiLM per decoder layer (auto-detected count)

    Memory-token pathway (v3) is KEPT for belt-and-suspenders — the encoder
    memory still gets a [POCKET]+[POSE] prefix.  FiLM is ADDITIVE on top.
    """

    def __init__(self, base_model: Mol2MolModel,
                  pose_mean=None, pose_std=None,
                  film_enabled=True):
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
        # CFG null token, kept for backward compat (unused by v4 default).
        self.null_emb = nn.Parameter(torch.zeros(D_MODEL))
        nn.init.normal_(self.null_emb, mean=0.0, std=0.02)
        # FiLM modules.
        self.condition_pooler = ConditionPooler()
        n_layers = len(self.base.network.decoder.layers)
        self.film_layers = nn.ModuleList([LayerFiLM() for _ in range(n_layers)])
        self.film_enabled = film_enabled
        # Store the current conditioning tensor for the monkey-patched decoder.
        self._current_c = None
        # Monkey-patch the decoder's forward to apply FiLM.
        self._install_film_hook()

    def _install_film_hook(self):
        """Replace self.base.network.decoder.forward with a version that
        applies FiLM after each layer, using self._current_c."""
        decoder = self.base.network.decoder
        film_layers = self.film_layers

        _wrapper = self  # closure over the wrapper for _current_c access

        def _film_forward(x, memory, src_mask, tgt_mask):
            c = _wrapper._current_c
            for i, layer in enumerate(decoder.layers):
                x = layer(x, memory, src_mask, tgt_mask)
                if _wrapper.film_enabled and c is not None:
                    x = film_layers[i](x, c)
            return decoder.norm(x)

        decoder.forward = _film_forward

    @property
    def device(self):
        return next(self.base.network.parameters()).device

    def normalize_pose(self, pose):
        return (pose - self.pose_mean) / self.pose_std.clamp_min(1e-6)

    def _cond_tokens(self, pocket_vec, pose_vec, drop_cond=None):
        """Return (B, 2, D) memory-conditioning tokens."""
        cond = torch.stack([pocket_vec, pose_vec], dim=1)   # (B, 2, D)
        if drop_cond is not None:
            null = self.null_emb.view(1, 1, -1).expand(cond.shape[0], 2, -1)
            m = drop_cond.view(-1, 1, 1).float()
            cond = (1.0 - m) * cond + m * null
        return cond

    def _build_memory(self, src, src_mask, pocket_vec, pose_vec,
                        drop_cond=None,
                        retrieval_prefix_embs=None,
                        retrieval_prefix_mask=None):
        """Same shape as v3: [retrieval..., [POCKET][POSE], encoder_memory]."""
        memory = self.base.network.encoder(
            self.base.network.src_embed(src), src_mask)   # (B, L, D)
        B, L, D = memory.shape
        cond = self._cond_tokens(pocket_vec, pose_vec, drop_cond)
        pieces = []
        mask_pieces = []
        if retrieval_prefix_embs is not None and retrieval_prefix_embs.shape[1] > 0:
            K = retrieval_prefix_embs.shape[1]
            pieces.append(retrieval_prefix_embs)
            if retrieval_prefix_mask is None:
                retrieval_prefix_mask = torch.ones(B, K, dtype=torch.bool,
                                                       device=src.device)
            mask_pieces.append(retrieval_prefix_mask.long())
        pieces.append(cond)
        cond_mask = torch.ones(B, 2, dtype=torch.long, device=src.device)
        mask_pieces.append(cond_mask)
        pieces.append(memory)
        base_mask_1d = src_mask.squeeze(1).long() if src_mask.dim() == 3 \
                            else src_mask.long()
        mask_pieces.append(base_mask_1d)
        memory_ext = torch.cat(pieces, dim=1)
        mask_1d = torch.cat(mask_pieces, dim=1)
        src_mask_ext = mask_1d.unsqueeze(1)
        return memory_ext, src_mask_ext

    def likelihood(self, src, src_mask, trg, trg_mask,
                    residue_emb, residue_mask, pose,
                    drop_cond=None,
                    retrieval_prefix_embs=None,
                    retrieval_prefix_mask=None):
        pocket_vec = self.pocket_enc(residue_emb, residue_mask)
        pose_vec = self.pose_enc(pose)
        # Set FiLM conditioning for the monkey-patched decoder.
        self._current_c = self.condition_pooler(pocket_vec, pose_vec)
        try:
            memory_ext, src_mask_ext = self._build_memory(
                src, src_mask, pocket_vec, pose_vec, drop_cond,
                retrieval_prefix_embs, retrieval_prefix_mask)
            trg_y = trg[:, 1:]
            trg_in = trg[:, :-1]
            out = self.base.network.decoder(
                self.base.network.tgt_embed(trg_in), memory_ext,
                src_mask_ext, trg_mask)
            log_prob = self.base.network.generator(
                out, self.base.temperature).transpose(1, 2)
            nll = self.base._nll_loss(log_prob, trg_y).sum(dim=1)
        finally:
            self._current_c = None
        return nll

    @torch.no_grad()
    def encode_smiles_to_prefix_embs(self, smiles_list):
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
        if not seqs:
            return torch.zeros(0, D_MODEL, device=self.device)
        import numpy as np
        L = max(len(s) for s in seqs)
        arr = np.zeros((len(seqs), L), dtype=np.int64)
        for j, s in enumerate(seqs):
            arr[j, :len(s)] = s
        src = torch.from_numpy(arr).to(self.device)
        src_mask = (src != 0).unsqueeze(-2).long()
        mem = self.base.network.encoder(
            self.base.network.src_embed(src), src_mask)
        mask_f = (src != 0).float().unsqueeze(-1)
        pooled = (mem * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)
        return pooled

    @torch.no_grad()
    def sample_multinomial(self, src, src_mask,
                              residue_emb, residue_mask, pose,
                              retrieval_prefix_embs=None,
                              max_length=128, temperature=1.0):
        """FiLM sampling — no CFG combine, just standard sampling with FiLM."""
        from torch.autograd import Variable
        from reinvent.models.transformer.core.network.module.subsequent_mask \
            import subsequent_mask
        B = src.shape[0]
        prefix_embs = None; prefix_mask = None
        if retrieval_prefix_embs is not None and retrieval_prefix_embs.numel() > 0:
            K = retrieval_prefix_embs.shape[0]
            prefix_embs = retrieval_prefix_embs.unsqueeze(0).expand(B, K, -1)
            prefix_mask = torch.ones(B, K, dtype=torch.bool, device=src.device)
        pocket_vec = self.pocket_enc(residue_emb, residue_mask)
        pose_vec = self.pose_enc(pose)
        self._current_c = self.condition_pooler(pocket_vec, pose_vec)
        try:
            mem, msk = self._build_memory(src, src_mask, pocket_vec, pose_vec,
                                              None, prefix_embs, prefix_mask)
            ys = torch.ones(1, device=src.device).repeat(B, 1).long()
            break_cond = torch.zeros(B, dtype=torch.bool, device=src.device)
            nlls = torch.zeros(B, device=src.device)
            end_token = self.base.vocabulary["$"]
            for i in range(max_length - 1):
                tgt_msk = Variable(subsequent_mask(ys.size(1)).type_as(src))
                out = self.base.network.decode(mem, msk, Variable(ys), tgt_msk)
                log_prob = self.base.network.generator(out[:, -1], temperature)
                prob = torch.exp(log_prob)
                mask_prop = self.base.mask_property_tokens(B)
                prob = prob.masked_fill(mask_prop, 0)
                row_sum = prob.sum(dim=-1, keepdim=True)
                prob = torch.where(row_sum > 1e-8, prob,
                                        torch.ones_like(prob) / prob.shape[-1])
                next_word = torch.multinomial(prob, 1)
                break_t = torch.unsqueeze(break_cond, 1)
                next_word = next_word.masked_fill(break_t, 0)
                ys = torch.cat([ys, next_word], dim=1)
                nw = next_word.reshape(-1)
                nlls += self.base._nll_loss(log_prob, nw)
                break_cond = break_cond | (nw == end_token)
                if bool(break_cond.all()):
                    break
        finally:
            self._current_c = None
        from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
        tok = SMILESTokenizer()
        out_smiles = [tok.untokenize(self.base.vocabulary.decode(seq))
                       for seq in ys.detach().cpu().numpy()]
        return out_smiles, nlls.detach().cpu().numpy()

    @torch.no_grad()
    def sample_multinomial_film_toggle(self, src, src_mask,
                                            residue_emb, residue_mask, pose,
                                            film_enabled: bool,
                                            retrieval_prefix_embs=None,
                                            max_length=128, temperature=1.0):
        """Temporarily enable/disable FiLM for the sampling call."""
        prev = self.film_enabled
        self.film_enabled = film_enabled
        try:
            return self.sample_multinomial(
                src, src_mask, residue_emb, residue_mask, pose,
                retrieval_prefix_embs=retrieval_prefix_embs,
                max_length=max_length, temperature=temperature)
        finally:
            self.film_enabled = prev


def load_base_model(prior_path, device):
    save_dict = torch.load(prior_path, map_location=device, weights_only=False)
    model = Mol2MolModel.create_from_dict(save_dict, ModelModeEnum().TRAINING, device)
    return model


def load_film_model(prior_path, device,
                      ckpt_path=None, pose_mean=None, pose_std=None,
                      init_from_v2_ckpt=None, init_from_v3_cfg_ckpt=None,
                      strict=True):
    """Load M1aV2FiLMModel.

    - `init_from_v2_ckpt`: v2 wrapper checkpoint (has pocket_enc, pose_enc,
      pose_mean/std; missing null_emb, film_layers, condition_pooler).
    - `init_from_v3_cfg_ckpt`: v3 CFG checkpoint (has pocket_enc, pose_enc,
      null_emb; missing film_layers, condition_pooler).
    - `ckpt_path`: v4 FiLM checkpoint (strict).
    """
    base = load_base_model(prior_path, device)
    pm = torch.as_tensor(pose_mean, dtype=torch.float32) if pose_mean is not None else None
    ps = torch.as_tensor(pose_std, dtype=torch.float32) if pose_std is not None else None
    wrapper = M1aV2FiLMModel(base, pm, ps).to(device)

    def _load_partial(sd, source_name):
        missing, unexpected = wrapper.load_state_dict(sd, strict=False)
        # allowed missing: our new modules (condition_pooler, film_layers, null_emb).
        allowed_missing = {"null_emb"}
        real_missing = [k for k in missing
                          if not (k in allowed_missing
                                    or k.startswith("condition_pooler.")
                                    or k.startswith("film_layers."))]
        if real_missing:
            raise RuntimeError(f"[{source_name}] missing keys not allowed: "
                                  f"{real_missing[:10]}")
        if unexpected:
            raise RuntimeError(f"[{source_name}] unexpected keys: {unexpected[:10]}")

    if init_from_v2_ckpt is not None and Path(init_from_v2_ckpt).exists():
        sd = torch.load(init_from_v2_ckpt, map_location=device, weights_only=False)
        state = sd["model_state"] if "model_state" in sd else sd
        _load_partial(state, f"v2:{init_from_v2_ckpt}")
    if init_from_v3_cfg_ckpt is not None and Path(init_from_v3_cfg_ckpt).exists():
        sd = torch.load(init_from_v3_cfg_ckpt, map_location=device, weights_only=False)
        state = sd["model_state"] if "model_state" in sd else sd
        _load_partial(state, f"v3_cfg:{init_from_v3_cfg_ckpt}")
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = sd["model_state"] if "model_state" in sd else sd
        wrapper.load_state_dict(state, strict=strict)
        # CRITICAL FIX (QA #8): Mol2MolModel is not an nn.Module so
        # wrapper.state_dict() does NOT include the 17.46M base transformer
        # weights.  If the checkpoint has a separate "base_network_state"
        # key (written by the fixed save_film_model), load it into
        # wrapper.base.network so DPO/SFT updates to the transformer are
        # actually restored at inference time.
        if isinstance(sd, dict) and "base_network_state" in sd:
            base_missing, base_unexpected = wrapper.base.network.load_state_dict(
                sd["base_network_state"], strict=False)
            if base_missing:
                print(f"[load base] MISSING keys: {base_missing[:5]}"
                       f"{'...' if len(base_missing) > 5 else ''}",
                       flush=True)
            if base_unexpected:
                print(f"[load base] UNEXPECTED keys: {base_unexpected[:5]}"
                       f"{'...' if len(base_unexpected) > 5 else ''}",
                       flush=True)
    return wrapper


def save_film_model(wrapper, ckpt_path, extra=None):
    """QA #8 fix: also save wrapper.base.network.state_dict() so the 17.46M
    base transformer weights (which receive DPO/SFT gradients but are NOT
    included in wrapper.state_dict() because Mol2MolModel is not an
    nn.Module) are actually persisted.
    """
    payload = {
        "model_state": wrapper.state_dict(),
        "base_network_state": wrapper.base.network.state_dict(),
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)
