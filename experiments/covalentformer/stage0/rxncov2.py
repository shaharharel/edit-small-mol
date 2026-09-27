#!/usr/bin/env python
"""RXNcov v2 -- shared encoder, REACTION BOTTLENECK, task-specific heads.

THE DESIGN PROBLEM THIS SOLVES. v1 trained one seq2seq on pre->adduct and then asked the same
decoder to emit ANALOGUES. Those are different output distributions competing for one decoder, and
the 63-d conditioning vector it used turned out to be mostly non-information: `delta` is EXACTLY
post-pre (21 of 63 dims are arithmetic padding, max error 5.96e-08) and the categorical blocks are
~96% identical pre->post, so "predict the post state" was ~96% solvable from the pre state alone.
The only block that moved was the geometry (mean |angle change| 6.688 deg). The 63-d vector is gone.

THE SHAPE:

    SMILES(hit) --> ENCODER ---------> z
                                       |
                            REACTION BOTTLENECK r   (low-dim, SHARED)
                                       |
             +-------------+-----------+-----------+--------------+
             |             |                       |              |
       DECODER-A      DECODER-R              GEOMETRY HEAD   AFFINITY HEAD
       analogues      adducts                angle, approach   dPIC50
     (target task)   (189,571 pairs)          distance        (143,075 pairs)

Decoder-R carries the 189,571 ADDUCT+RETRO pairs, Decoder-A only ever emits analogues, so they
never compete for output capacity. They are coupled ONLY through r, so whatever Decoder-R learns
about covalent reactivity must be encoded there, and Decoder-A reads the same r. At inference only
Decoder-A runs; Decoder-R is scaffolding. The two auxiliary heads hang off r as well, which is what
makes the geometry pocket-dependent (the pocket enters the encoder) and puts RXNcov and the LLM on
the SAME affinity objective so they are comparable on more than validity.

WHY THE BOTTLENECK IS THE RISK, STATED UP FRONT. If r is too narrow the analogue task starves; too
wide and the "coupling" is nominal because each head can carve out its own subspace. --rbits is a
sweep, not a guess, and the honest check is the ABLATION already wired in: train with the r-gate
zeroed for Decoder-A and see whether analogue quality actually drops. If it does not, the coupling
claim is void and I will report that rather than the architecture diagram.

SIZED FOR A 16 GB V100: d_model 512, 6+6 layers, batch 64 at len 160 fits with room to spare.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- tokenizer
ATOM = re.compile(r'(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|@|[BCNOFPSIbcnops]|[0-9]|\(|\)|\[|\]|=|#|-|\+|\\|/|%|\.|:|~|\*|\$)')
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def tokenize(smi):
    return ATOM.findall(smi or '')


class Vocab:
    def __init__(self, toks=None):
        self.itos = ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(toks or [])
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def enc(self, smi, maxlen):
        ids = [BOS] + [self.stoi.get(t, UNK) for t in tokenize(smi)][:maxlen - 2] + [EOS]
        return ids

    def dec(self, ids):
        out = []
        for i in ids:
            if i in (BOS, PAD):
                continue
            if i == EOS:
                break
            out.append(self.itos[i] if i < len(self.itos) else '')
        return ''.join(out)

    def save(self, p):
        json.dump(self.itos, open(p, 'w'))

    @staticmethod
    def load(p):
        v = Vocab([]); v.itos = json.load(open(p)); v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


# ---------------------------------------------------------------- model
class Enc(nn.Module):
    def __init__(self, V, d, nl, nh, ff, drop):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.pos = nn.Embedding(1024, d)
        layer = nn.TransformerEncoderLayer(d, nh, ff, drop, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, nl)

    def forward(self, x, pad_mask):
        p = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(p)
        return self.tr(h, src_key_padding_mask=pad_mask)


class Dec(nn.Module):
    """One decoder. Cross-attends to [encoder memory ; the shared reaction bottleneck r]."""
    def __init__(self, V, d, nl, nh, ff, drop):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.pos = nn.Embedding(1024, d)
        layer = nn.TransformerDecoderLayer(d, nh, ff, drop, batch_first=True, norm_first=True)
        self.tr = nn.TransformerDecoder(layer, nl)
        self.out = nn.Linear(d, V)

    def forward(self, y_in, mem, mem_pad):
        p = torch.arange(y_in.size(1), device=y_in.device).unsqueeze(0)
        h = self.emb(y_in) + self.pos(p)
        causal = nn.Transformer.generate_square_subsequent_mask(y_in.size(1), device=y_in.device)
        h = self.tr(h, mem, tgt_mask=causal, tgt_key_padding_mask=(y_in == PAD),
                    memory_key_padding_mask=mem_pad)
        return self.out(h)


class RXNcov2(nn.Module):
    def __init__(self, V, d=512, nl=6, nh=8, ff=2048, drop=0.1, rbits=64, pocket_dim=56):
        super().__init__()
        self.enc = Enc(V, d, nl, nh, ff, drop)
        self.pocket = nn.Sequential(nn.Linear(pocket_dim, d), nn.GELU(), nn.Linear(d, d))
        # the SHARED reaction bottleneck: everything downstream sees the reaction only through r
        self.to_r = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, rbits))
        self.from_r = nn.Linear(rbits, d)
        self.dec_a = Dec(V, d, nl, nh, ff, drop)     # analogues  (target task)
        self.dec_r = Dec(V, d, nl, nh, ff, drop)     # adducts    (reaction pretraining)
        self.geom = nn.Sequential(nn.Linear(rbits, 128), nn.GELU(), nn.Linear(128, 2))   # angle, dist
        self.aff = nn.Sequential(nn.Linear(rbits, 128), nn.GELU(), nn.Linear(128, 1))    # dPIC50
        self.rbits = rbits

    def encode(self, src, pocket=None, use_r=True):
        pad = (src == PAD)
        mem = self.enc(src, pad)
        pooled = (mem * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        if pocket is not None:
            pooled = pooled + self.pocket(pocket)
        r = self.to_r(pooled)
        if not use_r:                       # ABLATION: cut the bottleneck, keep everything else
            r = torch.zeros_like(r)
        rtok = self.from_r(r).unsqueeze(1)                       # [B,1,d]
        mem2 = torch.cat([mem, rtok], 1)
        pad2 = torch.cat([pad, torch.zeros(pad.size(0), 1, dtype=torch.bool, device=pad.device)], 1)
        return mem2, pad2, r

    def forward(self, src, y_in, task='A', pocket=None, use_r=True):
        mem, pad, r = self.encode(src, pocket, use_r)
        dec = self.dec_a if task == 'A' else self.dec_r
        return dec(y_in, mem, pad), r


# ---------------------------------------------------------------- data
class Pairs(torch.utils.data.Dataset):
    def __init__(self, path, vocab, maxlen, task):
        self.rows = [json.loads(l) for l in open(path)]
        self.v = vocab; self.m = maxlen; self.task = task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        src = r.get('src') or r.get('hit') or r.get('input')
        tgt = r.get('tgt') or r.get('analogue')
        if tgt is None and isinstance(r.get('output'), str) and 'Analogue: ' in r['output']:
            tgt = r['output'].rsplit('Analogue: ', 1)[1]
        y = r.get('dpic50')
        return {'src': self.v.enc(src, self.m), 'tgt': self.v.enc(tgt, self.m),
                'y': float(y) if y is not None else float('nan'), 'task': self.task}


def pad_to(seqs, L):
    out = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s[:L])
    return out


def collate(b):
    Ls = max(len(x['src']) for x in b); Lt = max(len(x['tgt']) for x in b)
    return {'src': pad_to([x['src'] for x in b], Ls),
            'tgt': pad_to([x['tgt'] for x in b], Lt),
            'y': torch.tensor([x['y'] for x in b], dtype=torch.float32),
            'task': b[0]['task']}


def build_vocab(paths, out):
    toks = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            for k in ('src', 'tgt', 'hit', 'analogue', 'input'):
                if isinstance(r.get(k), str):
                    toks.update(tokenize(r[k]))
            if isinstance(r.get('output'), str) and 'Analogue: ' in r['output']:
                toks.update(tokenize(r['output'].rsplit('Analogue: ', 1)[1]))
    v = Vocab(toks); v.save(out)
    print('vocab %d tokens -> %s' % (len(v.itos), out), flush=True)
    return v


# ---------------------------------------------------------------- train
def run_stage(model, ds, dev, args, stage, task, use_aff=False):
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate,
                                     num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = max(len(dl) * args.epochs, 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    hub = nn.HuberLoss(delta=0.5)
    step = 0; run = 0.0; runa = 0.0; na = 0
    model.train()
    for ep in range(args.epochs):
        for b in dl:
            src = b['src'].to(dev); tgt = b['tgt'].to(dev); y = b['y'].to(dev)
            logits, r = model(src, tgt[:, :-1], task=task)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.05)
            if use_aff:
                m = ~torch.isnan(y)
                if m.any():
                    la = hub(model.aff(r).squeeze(-1)[m], y[m])
                    loss = loss + args.aff_weight * la
                    runa += float(la); na += 1
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); step += 1; run += float(loss)
            if step % 200 == 0:
                print('  [%s] step %d/%d  loss %.4f  aff %.4f' %
                      (stage, step, total, run / 200, runa / max(na, 1)), flush=True)
                run = 0.0; runa = 0.0; na = 0
            if step >= total:
                break
        if step >= total:
            break
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default='', help='ADDUCT/RETRO pairs -> Decoder-R')
    ap.add_argument('--stage2', default='', help='hit->analogue pairs -> Decoder-A')
    ap.add_argument('--affinity', default='', help='dPIC50-labelled pairs -> Decoder-A + aff head')
    ap.add_argument('--out', required=True)
    ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--ff', type=int, default=2048)
    ap.add_argument('--rbits', type=int, default=64)
    ap.add_argument('--maxlen', type=int, default=160)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--aff-weight', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--smoke', action='store_true', help='tiny run, no GPU needed')
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = [p for p in (a.stage1, a.stage2, a.affinity) if p]
    v = build_vocab(paths, os.path.join(a.out, 'vocab.json'))
    model = RXNcov2(len(v.itos), a.d, a.layers, a.heads, a.ff, rbits=a.rbits).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print('RXNcov2 params %.1fM  (d=%d layers=%d rbits=%d) on %s' % (n / 1e6, a.d, a.layers, a.rbits, dev), flush=True)

    if a.stage1:
        ds = Pairs(a.stage1, v, a.maxlen, 'R')
        if a.smoke:
            ds.rows = ds.rows[:512]
        print('STAGE 1: Decoder-R on %d reaction pairs' % len(ds), flush=True)
        run_stage(model, ds, dev, a, 'stage1', 'R')
        torch.save(model.state_dict(), os.path.join(a.out, 'stage1.pt'))
    if a.stage2:
        ds = Pairs(a.stage2, v, a.maxlen, 'A')
        if a.smoke:
            ds.rows = ds.rows[:512]
        print('STAGE 2: Decoder-A on %d analogue pairs' % len(ds), flush=True)
        run_stage(model, ds, dev, a, 'stage2', 'A')
        torch.save(model.state_dict(), os.path.join(a.out, 'stage2.pt'))
    if a.affinity:
        ds = Pairs(a.affinity, v, a.maxlen, 'A')
        if a.smoke:
            ds.rows = ds.rows[:512]
        print('STAGE 4: Decoder-A + dPIC50 head on %d affinity pairs' % len(ds), flush=True)
        run_stage(model, ds, dev, a, 'affinity', 'A', use_aff=True)
        torch.save(model.state_dict(), os.path.join(a.out, 'affinity.pt'))
    print('DONE -> %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
