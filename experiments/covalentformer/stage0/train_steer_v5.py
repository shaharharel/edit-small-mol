#!/usr/bin/env python
"""Train ONE steering arm: Mol2Mol + a direction-instruction token.

    mode='none'   no instruction at all                  CONTROL
    mode='instr'  one row, embedding over {UP,DOWN,SAME} PRIMARY

Every arm starts from the MOL2MOL PRIOR, never from scratch -- the covalent chemistry and the
SMILES grammar come for free, so any difference between mode=none and mode=instr is
attributable to the instruction and not to the chemistry lesson.

WHAT IS REPORTED EACH EPOCH, and why all three are needed:

  valid       loss with the TRUE instruction
  GAP_perm    loss with instructions PERMUTED within the batch, minus valid.
              This is DEPENDENCE: does the model's output change when the instruction
              changes? It is the arm's OWN null and it is the only honest baseline --
              a CI against zero is not a null, which is how `extent` and `resid` were
              once reported as small positives when they were non-directional.
  GAP_flip    loss with each instruction replaced by its OPPOSITE (UP<->DOWN, SAME kept),
              minus valid. This is DIRECTIONALITY: a model that merely notices the token
              scores on GAP_perm; only a model that knows which WAY to move scores here.

DEPENDENCE IS NOT BENEFIT, and neither is directionality. The benefit is the mode=none
contrast, which is why the control arm is trained for every param.

NO CLASS WEIGHTING IN THE LOSS, DELIBERATELY. An earlier draft of this docstring claimed
inverse-frequency weights; the code never had them, which is the same assert-what-the-code-
does-not-do defect that has landed ten times in this project. Class balance is handled
UPSTREAM in build_steer_v5.py, which downsamples SAME to the mean of the moving classes in
the TRAIN set only. The printed instr mix below is the check on that.

SEEDED. torch, numpy and python random are all fixed, and the seed is printed in the header --
an unseeded run cost this project two headline numbers that did not replicate.
"""
from __future__ import annotations
import os, sys, csv, json, time, math, random, argparse, collections
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
REINVENT = '/Users/shaharharel/Documents/github/REINVENT4'
sys.path.insert(0, REINVENT)
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder  # noqa
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer      # noqa

# FOUR levels up, not three: train_phaseA.py lives in covalentformer/ but this file lives in
# covalentformer/stage0/, so copying its path arithmetic lands on experiments/models/ and
# every arm dies with FileNotFoundError before a single step runs.
PRIOR = os.path.join(os.path.dirname(os.path.dirname(ROOT)),
                     'models/reinvent4_mol2mol_warhead_tokens.prior')
assert os.path.exists(PRIOR), 'prior not found at %s' % PRIOR
INSTR = ['SAME', 'UP', 'DOWN', 'CHANGED']
I2N = {s: i for i, s in enumerate(INSTR)}
OPPOSITE = {'UP': 'DOWN', 'DOWN': 'UP', 'SAME': 'SAME', 'CHANGED': 'SAME'}


def subsequent_mask(size, device):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0)


class SteerNet(EncoderDecoder):
    def __init__(self, *a, mode='instr', **kw):
        super().__init__(*a, **kw)
        self.mode = mode
        if mode == 'instr':
            self.instr_emb = nn.Embedding(len(INSTR), self.model_dimension)

    def encode_cond(self, src, src_mask, cond):
        mem = self.encode(src, src_mask)
        if self.mode == 'none':
            return mem, src_mask
        row = self.instr_emb(cond.long()).unsqueeze(1)
        mem = torch.cat([row, mem], dim=1)
        pad = torch.ones(mem.size(0), 1, 1, dtype=src_mask.dtype, device=src_mask.device)
        return mem, torch.cat([pad, src_mask], dim=-1)

    def forward_cond(self, src, src_mask, tgt, tgt_mask, cond):
        mem, m2 = self.encode_cond(src, src_mask, cond)
        return self.decode(mem, m2, tgt, tgt_mask)


class SteerData(torch.utils.data.Dataset):
    def __init__(self, path, vocab, tok, max_len=128, limit=0):
        self.rows = [r for r in csv.DictReader(open(path))
                     if r.get('anchor', '').strip() and r.get('target', '').strip()]
        if limit:
            self.rows = self.rows[:limit]
        self.v, self.t, self.max_len = vocab, tok, max_len

    def __len__(self):
        return len(self.rows)

    def enc(self, s):
        return torch.tensor(np.asarray(self.v.encode(self.t.tokenize(s)))
                            .astype(np.int64)[:self.max_len], dtype=torch.long)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            s, t = self.enc(r['anchor']), self.enc(r['target'])
        except Exception:
            return None
        d = r.get('instr', 'SAME')
        return s, t, I2N.get(d, 0), I2N.get(OPPOSITE.get(d, 'SAME'), 0)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    ss, ts, cs, fs = zip(*batch)
    B, Ks, Kt = len(ss), max(len(x) for x in ss), max(len(x) for x in ts)
    src = torch.zeros(B, Ks, dtype=torch.long)
    tgt = torch.zeros(B, Kt, dtype=torch.long)
    sm = torch.zeros(B, 1, Ks, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(ss, ts)):
        src[i, :len(a)] = a
        sm[i, 0, :len(a)] = True
        tgt[i, :len(b)] = b
    return (src, sm, tgt, torch.tensor(cs, dtype=torch.long),
            torch.tensor(fs, dtype=torch.long))


def batch_loss(model, crit, src, sm, tgt, cond, dev):
    src, sm, tgt, cond = src.to(dev), sm.to(dev), tgt.to(dev), cond.to(dev)
    ti, to = tgt[:, :-1], tgt[:, 1:]
    tm = (ti != 0).unsqueeze(-2) & subsequent_mask(ti.size(1), dev)
    out = model.forward_cond(src, sm, ti, tm, cond)
    logits = model.generator(out)
    return crit(logits.reshape(-1, logits.size(-1)), to.reshape(-1))


@torch.no_grad()
def evaluate(model, dl, crit, dev, seed):
    """Returns (valid, gap_perm, gap_flip). Permutation is SEEDED so the null is
    reproducible; an unseeded null is a different number every time it is quoted."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = totp = totf = n = 0.0
    for b in dl:
        if b is None:
            continue
        src, sm, tgt, cond, flip = b
        tot += batch_loss(model, crit, src, sm, tgt, cond, dev).item()
        perm = cond[torch.randperm(cond.size(0), generator=g)]
        totp += batch_loss(model, crit, src, sm, tgt, perm, dev).item()
        totf += batch_loss(model, crit, src, sm, tgt, flip, dev).item()
        n += 1
    model.train()
    if not n:
        return None, None, None
    return tot / n, (totp - tot) / n, (totf - tot) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/steer_v5')
    ap.add_argument('--param', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['none', 'instr'], default='instr')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--init', default='', help='checkpoint to warm-start from (pocket arms)')
    a = ap.parse_args()

    dev = (('cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
           if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    ck = torch.load(a.prior, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    npm = dict(ck['network_parameter'])
    model = SteerNet(mode=a.mode, **npm)
    missing, unexpected = model.load_state_dict(ck['network_state'], strict=False)
    bad = [k for k in missing if not k.startswith('instr_emb')]
    if bad:
        print('  FATAL: prior did not load for %s' % bad[:5]); return 2
    if a.init:
        w = torch.load(a.init, map_location='cpu', weights_only=False)
        model.load_state_dict(w['network_state'], strict=False)
        print('  warm-started from %s' % a.init)
    model.to(dev)

    trf = os.path.join(a.data, '%s_train.csv' % a.param)
    vaf = os.path.join(a.data, '%s_valid.csv' % a.param)
    tr = SteerData(trf, vocab, tok, limit=a.limit)
    va = SteerData(vaf, vocab, tok, limit=(a.limit // 10 if a.limit else 0))
    print('PARAM %s | mode=%s device=%s seed=%d | train %d valid %d | new tensors %d'
          % (a.param, a.mode, dev, a.seed, len(tr), len(va), len(missing)))

    cnt = collections.Counter(r.get('instr', 'SAME') for r in tr.rows)
    print('  train instr mix: %s' % dict(cnt))

    gtr = torch.Generator().manual_seed(a.seed)
    dtr = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True,
                                      collate_fn=collate, generator=gtr)
    dva = torch.utils.data.DataLoader(va, batch_size=a.bs, shuffle=False, collate_fn=collate)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    hist = []
    for ep in range(a.epochs):
        t0 = time.time(); run = 0.0; nb = 0
        for b in dtr:
            if b is None:
                continue
            src, sm, tgt, cond, _ = b
            loss = batch_loss(model, crit, src, sm, tgt, cond, dev)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1
            if nb % 400 == 0:
                print('    ep%d step %d  train %.4f' % (ep, nb, run / nb)); sys.stdout.flush()
        v, gp, gf = evaluate(model, dva, crit, dev, a.seed)
        rec = {'epoch': ep, 'train': run / max(nb, 1), 'valid': v,
               'gap_perm': gp, 'gap_flip': gf, 'secs': round(time.time() - t0, 1)}
        hist.append(rec)
        print('  ep%d  train %.4f  valid %.4f  GAP_perm %+.4f  GAP_flip %+.4f  (%.0fs)'
              % (ep, rec['train'], v, gp, gf, rec['secs'])); sys.stdout.flush()
        torch.save({'network_state': model.state_dict(), 'network_parameter': npm,
                    'vocabulary': ck['vocabulary'], 'mode': a.mode, 'param': a.param,
                    'seed': a.seed, 'epoch': ep, 'instr_vocab': INSTR,
                    'train_file': trf, 'valid_file': vaf, 'history': hist},
                   '%s_ep%d.pt' % (a.out, ep))
    with open('%s_history.json' % a.out, 'w') as fo:
        json.dump({'param': a.param, 'mode': a.mode, 'seed': a.seed,
                   'train_rows': len(tr), 'valid_rows': len(va),
                   'instr_mix': dict(cnt), 'history': hist}, fo, indent=2)
    print('DONE %s' % a.out)


if __name__ == '__main__':
    main()
