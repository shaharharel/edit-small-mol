"""Reimplementation of REINVENT4 Mol2Mol architecture (Harvard Annotated Transformer style)
compatible with loading `reinvent4_mol2mol_covalent_ft.prior` state_dict.

Key architectural details (from checkpoint inspection):
    - vocab_size=128, max_seq_len=128
    - 6 layers each in encoder/decoder, 8 heads, d=256, ff=2048, dropout=0.1
    - src_embed = [Embedding, PositionalEncoding] (Sequential)
    - tgt_embed = [Embedding, PositionalEncoding]
    - generator = Linear(256, vocab_size) with softmax outside
    - encoder.layers[i]:
        self_attn.linears.0..3 = Q, K, V, output_projection (each 256x256)
        feed_forward.w_1 (256->2048), w_2 (2048->256)
        sublayer[0..1] = residual+LayerNorm (a_2, b_2)  -- annotated Transformer style
        norm at encoder root
    - decoder.layers[i]: adds src_attn (cross-attn), sublayer[0..2]

Only supports LOAD/INFERENCE-compatible forward + coord conditioning extension.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Matches keys `a_2` (gamma) and `b_2` (beta)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d))
        self.b_2 = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mu) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual + LayerNorm (pre-norm-then-residual as in Annotated Transformer)."""

    def __init__(self, d, dropout):
        super().__init__()
        self.norm = LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)
    return torch.matmul(p, v), p


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d, dropout=0.1):
        super().__init__()
        assert d % h == 0
        self.d_k = d // h
        self.h = h
        self.linears = clones(nn.Linear(d, d), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        B = q.size(0)
        q_, k_, v_ = [
            l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (q, k, v))
        ]
        x, self.attn = attention(q_, k_, v_, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[3](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d, d_ff)
        self.w_2 = nn.Linear(d_ff, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d, dropout), 2)
        self.size = d

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d, dropout), 3)
        self.size = d

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d)
        self.d = d

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d)


class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].detach()
        return self.dropout(x)


class Generator(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.proj = nn.Linear(d, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Mol2MolTransformer(nn.Module):
    """Full Mol2Mol matching REINVENT4 covFT checkpoint."""

    def __init__(self, vocab_size=128, N=6, d=256, h=8, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        attn = MultiHeadAttention(h, d, dropout)
        ff = PositionwiseFeedForward(d, d_ff, dropout)
        pos = PositionalEncoding(d, dropout, max_len=max_len)
        self.encoder = Encoder(EncoderLayer(d, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(d, vocab_size), copy.deepcopy(pos))
        self.tgt_embed = nn.Sequential(Embeddings(d, vocab_size), copy.deepcopy(pos))
        self.generator = Generator(d, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        h = self.decode(memory, src_mask, tgt, tgt_mask)
        return self.generator(h), h  # log-probs and hidden


def subsequent_mask(size):
    return torch.tril(torch.ones(1, size, size, dtype=torch.uint8))


def load_mol2mol_prior(ckpt_path, device="cpu"):
    """Load prior weights into a Mol2MolTransformer."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net_state = ck["network_state"]
    vocab = ck["vocabulary"]
    np_ = ck["network_parameter"]
    max_seq_len = ck["max_sequence_length"]
    # Prior stores PE with 5000 rows even though max_sequence_length=128; reproduce
    model = Mol2MolTransformer(
        vocab_size=np_["vocabulary_size"],
        N=np_["num_layers"],
        d=np_["model_dimension"],
        h=np_["num_heads"],
        d_ff=np_["feedforward_dimension"],
        dropout=np_["dropout"],
        max_len=5000,
    )
    missing, unexpected = model.load_state_dict(net_state, strict=False)
    return model, vocab, max_seq_len, missing, unexpected


# =============== Coord-conditioned wrapper ================


class CoordConditionedMol2Mol(nn.Module):
    """Wraps Mol2MolTransformer, adding 5-vector coordinate conditioning
    by prepending K learned "coord tokens" (from an MLP) to the encoder memory.

    Also has an auxiliary MLP that reconstructs the 5-vector from the pooled
    decoder hidden state (auxiliary loss during training).
    """

    def __init__(self, base: Mol2MolTransformer, coord_dim=5, n_coord_tokens=4):
        super().__init__()
        self.base = base
        d = base.encoder.norm.a_2.shape[0]
        self.d = d
        self.n_coord_tokens = n_coord_tokens
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, 64), nn.GELU(),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, n_coord_tokens * d),
        )
        # Aux head that predicts coord from mean-pooled decoder hidden
        self.aux_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, coord_dim),
        )

    def build_memory(self, src, src_mask, coord_vec):
        """Encode src, then prepend n_coord_tokens conditioning tokens to memory."""
        # src: (B, S), src_mask: (B, 1, S), coord_vec: (B, 5)
        memory = self.base.encode(src, src_mask)  # (B, S, d)
        B = memory.shape[0]
        coord_tokens = self.coord_mlp(coord_vec).view(B, self.n_coord_tokens, self.d)  # (B, K, d)
        memory_ext = torch.cat([coord_tokens, memory], dim=1)  # (B, K+S, d)
        # extend mask: coord tokens always visible (1)
        # src_mask shape (B, 1, S) → (B, 1, K+S)
        ext_ones = torch.ones(B, 1, self.n_coord_tokens, dtype=src_mask.dtype, device=src_mask.device)
        src_mask_ext = torch.cat([ext_ones, src_mask], dim=-1)
        return memory_ext, src_mask_ext

    def forward(self, src, tgt, src_mask, tgt_mask, coord_vec):
        memory_ext, src_mask_ext = self.build_memory(src, src_mask, coord_vec)
        h = self.base.decode(memory_ext, src_mask_ext, tgt, tgt_mask)
        logp = self.base.generator(h)  # (B, T, V)
        # Aux: mean-pool valid tgt positions
        # tgt_mask: (B, 1, T, T) with lower-triangular; simplest: use last-token hidden or mean
        pooled = h.mean(dim=1)  # (B, d)
        aux_pred = self.aux_head(pooled)  # (B, 5)
        return logp, aux_pred, h

    def sample(self, src, src_mask, coord_vec, bos, eos, max_len=128, temperature=1.0, top_k=0):
        """Greedy/multinomial sampling. Returns (B, T) token ids."""
        device = src.device
        memory_ext, src_mask_ext = self.build_memory(src, src_mask, coord_vec)
        B = src.shape[0]
        ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(max_len - 1):
            tm = subsequent_mask(ys.size(1)).to(device)
            h = self.base.decode(memory_ext, src_mask_ext, ys, tm)
            logp = self.base.generator(h[:, -1])  # (B, V)
            if temperature != 1.0:
                logp = logp / temperature
            probs = logp.exp()
            # numerical safety: replace nan/inf, ensure non-neg, non-zero row sums
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs = probs.clamp_min(0)
            row_sum = probs.sum(-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs, torch.ones_like(probs) / probs.size(-1))
            if top_k and top_k > 0:
                v, ix = probs.topk(top_k, dim=-1)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, ix, v)
                probs = mask / mask.sum(-1, keepdim=True).clamp_min(1e-9)
            nxt = torch.multinomial(probs, 1)  # (B, 1)
            nxt = torch.where(done.unsqueeze(-1), torch.full_like(nxt, eos), nxt)
            ys = torch.cat([ys, nxt], dim=1)
            done = done | (nxt.squeeze(-1) == eos)
            if done.all():
                break
        return ys


# =============== Tokenizer helpers ================


def tokenize_smiles(smi: str, tokens: dict, bos=1, eos=2, pad=0, max_len=128):
    """Greedy-longest tokenization using multi-char tokens in `tokens` dict."""
    # sort tokens by length desc for greedy matching
    keys = sorted([k for k in tokens.keys() if k not in ("^", "$", "*", "?")], key=len, reverse=True)
    i = 0
    out = [bos]
    while i < len(smi):
        matched = False
        for k in keys:
            if smi.startswith(k, i):
                if k in tokens:
                    out.append(tokens[k])
                    i += len(k)
                    matched = True
                    break
        if not matched:
            # unknown char: skip (or map to * = 0)
            out.append(tokens.get("*", 0))
            i += 1
    out.append(eos)
    out = out[:max_len]
    # pad
    while len(out) < max_len:
        out.append(pad)
    return out[:max_len]


def detokenize(ids, tokens_inv: dict, bos=1, eos=2, pad=0):
    parts = []
    for i in ids:
        i = int(i)
        if i == bos:
            continue
        if i == eos:
            break
        if i == pad:
            continue
        parts.append(tokens_inv.get(i, ""))
    return "".join(parts)
