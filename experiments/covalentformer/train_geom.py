"""COVALENT SFT: geometry-conditioned Mol2Mol.

WHAT IS NEW, AND WHAT IS NOT. The network is deliberately the SAME shape as CovaCraft-v2-cond -- a
Mol2Mol encoder-decoder with extra rows prepended to the cross-attention memory. No architectural
claim is being made. The contribution is what those rows MEAN and how the data is built.

THE DEFECT THIS EXISTS TO FIX. v2-cond conditions on the warhead's pose p = (d_Sgamma-Cbeta,
theta_BD, phi_planar) and at inference pins it to Mol1's own pose: d ~ 1.85 A, theta ~ 105, phi ~ 0.
1.85 A is a FORMED C-S bond -- that vector describes the product, after the reaction. Worse, every
real covalent adduct in the training corpus carries essentially the same triple, so the conditioning
variable has almost no variance in training and is a CONSTANT across all samples at inference. A
2.47M-parameter module conditioned on a near-constant cannot differentiate anything; it can only act
as a bias term. There is no shuffle ablation in that work showing otherwise.

Here the conditioning is a REQUIREMENT computed before any candidate exists:

    d           |SG - attachment atom|, the PRE-reactive gap. Spans 2-12 A across pockets.
    cos theta   direction from the scaffold's exit vector to SG. Ortho vs para at fixed length.
    v_free      occlusion along the path. PINNED AT 1.0 IN TIER A -- there is no pocket, so this
                channel carries no information tonight and must not be interpreted.

and the label flips with the gap: the same fragment is a positive at d=5 A and a negative at d=8 A.
That is why no fragment-only descriptor can predict it (measured: all at 0.500-0.503, combined
logreg 0.492), and why the model has no shortcut but to read the geometry input.

BOTH LABELS ARE TRAINED ON. r=0 examples are not waste -- they are where the decision boundary is.
A model shown only reaching fragments learns "what covalent molecules look like", not "which ones
reach". Conditioning on r also buys bidirectional control, which is the honest test of whether the
channel is a condition or merely a bias: ask for r=0 and see whether it complies.

WHY SFT AND NOT RL FROM THE PRIOR. h_geom and h_label are NEW input channels with randomly
initialised weights. RL supplies one scalar reward per sequence; learning what a 256-d conditioning
vector MEANS from scalar rewards is hopeless sample-efficiency-wise, whereas cross-entropy supplies
dense per-token supervision. RL reweights a distribution the model can already express; SFT teaches
it to express something new. CovaCraft's own numbers agree: SFT moved acrylamide retention 0.4% ->
94.9% (+94.5 points) and the DAP RL stage then moved it 94.9% -> 99.3% (+4.4). There is a third
reason: DAP/DPO pull toward pi_prior via KL, so running RL from a geometry-blind prior would spend
the budget fighting that prior.
"""
import os, sys, csv, json, math, time, argparse, random
import numpy as np
import torch
import torch.nn as nn

REINVENT = '/Users/shaharharel/Documents/github/REINVENT4'
sys.path.insert(0, REINVENT)
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer

PRIOR = 'models/reinvent4_mol2mol_warhead_tokens.prior'


def subsequent_mask(size, device):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0)


class GeomEncoderDecoder(EncoderDecoder):
    """EncoderDecoder + two conditioning rows prepended to the cross-attention memory.

    memory   (B, K, d)      ->  (B, K+2, d)
    src_mask (B, 1, K)      ->  (B, 1, K+2)

    The base weights come from the Mol2Mol prior; only geom_mlp and label_emb are new. Keeping the
    injection at the MEMORY rather than inside the encoder means the anchor-SMILES pathway is
    numerically identical to the prior's at initialisation, so any behavioural change is
    attributable to the new rows.
    """

    def __init__(self, *a, cond_repeat=1, geom_rbf=0, **kw):
        self._pending_rbf = int(geom_rbf)
        super().__init__(*a, **kw)
        self.geom_rbf = self._pending_rbf
        d = self.model_dimension
        # INPUT ENCODING OF THE CONTINUOUS GEOMETRY. --geom-rbf switches d and cos(theta) from raw
        # scalars to RBF features before the MLP.
        #
        # WHY THIS IS THE NEXT TEST, registered BEFORE running it. Tonight's three hypotheses
        # disagreed in an informative way:
        #   H3 (discrete label bit) PASSED at +17.5 pts, p<0.0001, through THIS SAME injection point
        #   H2 (continuous d)       moves only linker LENGTH: Q1 +0.515 all length, Q2 +0.097 shape
        #   H1 (shuffle)            +4.3 pts p=0.123 at k=1, +1.8 pts p=0.555 at k=8
        # A one-bit condition steers generation decisively at the same site where a continuous
        # distance barely registers. That exonerates the injection point and the attention budget
        # (k=8 raised GAP +42% and Q2 by only 2.1 SE, and its gain was mostly geometry-INDEPENDENT:
        # its shuffled arm 43.5% beat k=1's true arm 41.0%). What is left is the REPRESENTATION: a
        # raw scalar through a 3->64->256 MLP gives the decoder no basis in which nearby distances
        # are distinguishable, so it can resolve d only coarsely enough to pick a bond count.
        #
        # PREDICTION: RBF features must raise Q2 (SHAPE at fixed length) above the k=1 baseline of
        # +0.097 by more than 2 SE, AND must do so in the 3-4 bond bin (k=1 +0.071, k=8 +0.037),
        # where 64/139 real covalent inhibitors sit. A gain that again lands only at 7-9 bonds
        # REFUTES the encoding hypothesis -- it would mean the model resolves long linkers better
        # for reasons unrelated to input basis, and the limit is elsewhere.
        gin = 3
        if getattr(self, 'geom_rbf', 0):
            n = self.geom_rbf
            # centres spanning the observed gap range (d 1.5-13 A) and cos(theta) in [-1, 1]
            self.register_buffer('rbf_d', torch.linspace(1.5, 13.0, n))
            self.register_buffer('rbf_c', torch.linspace(-1.0, 1.0, n))
            self.rbf_sigma_d = (13.0 - 1.5) / (n - 1)
            self.rbf_sigma_c = 2.0 / (n - 1)
            gin = 2 * n + 1
        # HIDDEN WIDTH STAYS 64 IN BOTH ARMS. Widening it to 128 alongside the RBF change did two
        # damaging things at once: it broke loading of every k=1/k=8 checkpoint already on disk
        # (geom_mlp.0 shape mismatch), and it made the RBF run differ from k=1 in TWO ways --
        # input basis AND capacity -- so a win could not be attributed to the encoding, which is
        # the entire registered claim. Only the input dimension may differ between arms.
        self.geom_mlp = nn.Sequential(nn.Linear(gin, 64), nn.ReLU(), nn.Linear(64, d))
        self.label_emb = nn.Embedding(2, d)
        # PRE-REGISTERED FALSIFIER for the attention-starvation hypothesis (task #21).
        # MEASURED on geom_ep1: the decoder places 0.0101 of its cross-attention mass on the two
        # conditioning rows against a uniform share of 0.0244 -- i.e. 0.41x uniform, collapsing to
        # 0.20x in layer 2. Meanwhile gradient boosting extracts the same label from (ECFP4, d,
        # cos_theta) at AUROC 0.913 on a FRAGMENT-DISJOINT split. So the mapping is learnable and
        # the transformer is not learning it; the hypothesis is that two rows cannot compete with
        # ~35 anchor tokens that already predict most of the output.
        #
        # PREDICTION, FIXED BEFORE RUNNING: repeating the conditioning rows k times raises their
        # share of the memory from 2/(K+2) to 2k/(K+2k). If starvation is the bottleneck, the
        # shuffled-g GAP must RISE materially above the k=1 trajectory (+0.0029 ep0, +0.0066 ep1).
        # If the GAP does NOT rise, attention starvation is REFUTED as the explanation and the
        # cause lies elsewhere -- most likely that copying the anchor is simply a better loss
        # bargain than reading geometry, which repetition cannot fix and FiLM might.
        # This is a crude instrument on purpose: it tests the hypothesis without changing the
        # architecture, so a null here is informative rather than confounded by a redesign.
        self.cond_repeat = int(cond_repeat)

    def encode_cond(self, src, src_mask, geom, label):
        mem = self.encode(src, src_mask)                       # (B, K, d)
        B = mem.size(0)
        k = max(1, self.cond_repeat)
        g = self.geom_mlp(self._geom_feat(geom)).unsqueeze(1)  # (B, 1, d)
        r = self.label_emb(label).unsqueeze(1)                 # (B, 1, d)
        cond = torch.cat([g, r], dim=1)                        # (B, 2, d)
        if k > 1:
            cond = cond.repeat(1, k, 1)                        # (B, 2k, d)
        mem = torch.cat([cond, mem], dim=1)                    # (B, K+2k, d)
        pad = torch.ones(B, 1, 2 * k, dtype=src_mask.dtype, device=src_mask.device)
        return mem, torch.cat([pad, src_mask], dim=-1)

    def _geom_feat(self, geom):
        """raw (d, cos_theta, v_free) -> RBF basis, or pass through unchanged when disabled."""
        if not getattr(self, 'geom_rbf', 0):
            return geom
        d = geom[:, 0:1]
        c = geom[:, 1:2]
        fd = torch.exp(-((d - self.rbf_d[None, :]) ** 2) / (2 * self.rbf_sigma_d ** 2))
        fc = torch.exp(-((c - self.rbf_c[None, :]) ** 2) / (2 * self.rbf_sigma_c ** 2))
        return torch.cat([fd, fc, geom[:, 2:3]], dim=-1)

    def forward_cond(self, src, src_mask, tgt, tgt_mask, geom, label):
        mem, m2 = self.encode_cond(src, src_mask, geom, label)
        return self.decode(mem, m2, tgt, tgt_mask)


class TierA(torch.utils.data.Dataset):
    def __init__(self, path, vocab, tok, max_len=128, limit=0):
        self.rows = list(csv.DictReader(open(path)))
        if limit:
            self.rows = self.rows[:limit]
        self.v, self.t, self.max_len = vocab, tok, max_len

    def __len__(self):
        return len(self.rows)

    def enc(self, smi):
        ids = self.v.encode(self.t.tokenize(smi))
        return torch.tensor(ids[:self.max_len], dtype=torch.long)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            s, t = self.enc(r['anchor']), self.enc(r['target'])
        except Exception:
            return None
        g = torch.tensor([float(r['d']), float(r['cos_theta']), float(r['v_free'])],
                         dtype=torch.float)
        return s, t, g, int(r['label'])


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    ss, ts, gs, ls = zip(*batch)
    B, Ks, Kt = len(ss), max(len(x) for x in ss), max(len(x) for x in ts)
    src = torch.zeros(B, Ks, dtype=torch.long)
    tgt = torch.zeros(B, Kt, dtype=torch.long)
    src_mask = torch.zeros(B, 1, Ks, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(ss, ts)):
        src[i, :len(a)] = a
        src_mask[i, 0, :len(a)] = True
        tgt[i, :len(b)] = b
    return src, src_mask, tgt, torch.stack(gs), torch.tensor(ls, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='experiments/covalentformer/data/tierA')
    ap.add_argument('--out', default='experiments/covalentformer/ckpt')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--geom-rbf', type=int, default=0,
                    help='encode d and cos(theta) as N RBF features instead of raw scalars')
    ap.add_argument('--cond-repeat', type=int, default=1,
                    help='repeat the 2 conditioning rows k times in the cross-attention memory; '
                         'pre-registered falsifier for the attention-starvation hypothesis')
    a = ap.parse_args()

    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    torch.manual_seed(20260913)
    random.seed(20260913)
    np.random.seed(20260913)

    ck = torch.load(a.prior, map_location='cpu', weights_only=False)
    # The checkpoint stores the vocabulary as a WRAPPER dict -- {'tokens': {...}, 'pad_token': 0,
    # 'bos_token': 1, ...} -- not as the bare token->id map Vocabulary() consumes. Passing the
    # wrapper straight in fails on the unhashable inner dict.
    _v = ck['vocabulary']
    if isinstance(_v, Vocabulary):
        vocab = _v
    else:
        vocab = Vocabulary(_v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    npm = dict(ck['network_parameter'])
    print('prior %s | vocab %d | %s | device %s | cond_repeat %d'
          % (os.path.basename(a.prior), npm['vocabulary_size'], npm, dev, a.cond_repeat))

    model = GeomEncoderDecoder(cond_repeat=a.cond_repeat, geom_rbf=a.geom_rbf, **npm)
    missing, unexpected = model.load_state_dict(ck['network_state'], strict=False)
    # The ONLY parameters that may be missing are the two new conditioning modules. Anything else
    # missing means the prior did not actually load and we would be training from scratch while
    # reporting a fine-tune -- fail loudly rather than silently produce a weaker baseline.
    # The allowlist must name EVERY tensor the prior legitimately lacks. rbf_d and rbf_c are
    # registered BUFFERS, so they show up in missing_keys exactly like parameters do -- the guard
    # correctly refused to run until they were declared. Widening the allowlist is right; relaxing
    # the guard to "ignore anything missing" would have let a genuinely unloaded prior through and
    # turned a fine-tune into a from-scratch run reported as a fine-tune.
    new = [k for k in missing
           if k.startswith('geom_mlp') or k.startswith('label_emb') or k.startswith('rbf_')]
    bad = [k for k in missing if k not in new]
    print('  loaded prior. new params: %d tensors | unexpected: %d' % (len(new), len(unexpected)))
    if bad:
        print('  FATAL: prior weights did not load for %d tensors: %s' % (len(bad), bad[:6]))
        return 2
    model.to(dev)

    tr = TierA(os.path.join(a.data, 'train.csv'), vocab, tok, limit=a.limit)
    va = TierA(os.path.join(a.data, 'valid.csv'), vocab, tok,
               limit=(a.limit // 10 if a.limit else 0))
    print('  train %d | valid %d' % (len(tr), len(va)))
    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate,
                                     num_workers=0, drop_last=True)
    dlv = torch.utils.data.DataLoader(va, batch_size=a.bs, shuffle=False, collate_fn=collate,
                                      num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    hist = []

    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for step, b in enumerate(dl):
            if b is None:
                continue
            src, src_mask, tgt, geom, lab = [x.to(dev) for x in b]
            ti, to = tgt[:, :-1], tgt[:, 1:]
            tm = subsequent_mask(ti.size(1), dev)
            out = model.forward_cond(src, src_mask, ti, tm, geom, lab)
            logits = model.generator(out)
            loss = lossf(logits.reshape(-1, logits.size(-1)), to.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
            if step % 200 == 0:
                print('  ep%d step %5d/%d loss %.4f  (%.1f min)'
                      % (ep, step, len(dl), tot / max(n, 1), (time.time() - t0) / 60), flush=True)

        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for b in dlv:
                if b is None:
                    continue
                src, src_mask, tgt, geom, lab = [x.to(dev) for x in b]
                ti, to = tgt[:, :-1], tgt[:, 1:]
                out = model.forward_cond(src, src_mask, ti, subsequent_mask(ti.size(1), dev),
                                         geom, lab)
                lg = model.generator(out)
                vt += lossf(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).item()
                vn += 1
        # SHUFFLED-GEOMETRY VALIDATION LOSS, computed every epoch alongside the real one.
        # This is the cheap in-training form of the pre-registered kill criterion: if permuting the
        # geometry vector across the batch does not RAISE validation loss, the decoder is not using
        # that channel and no amount of further training or data will change it.
        vs, vsn = 0.0, 0
        with torch.no_grad():
            for b in dlv:
                if b is None:
                    continue
                src, src_mask, tgt, geom, lab = [x.to(dev) for x in b]
                perm = torch.randperm(geom.size(0), device=dev)
                ti, to = tgt[:, :-1], tgt[:, 1:]
                out = model.forward_cond(src, src_mask, ti, subsequent_mask(ti.size(1), dev),
                                         geom[perm], lab[perm])
                lg = model.generator(out)
                vs += lossf(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).item()
                vsn += 1
        tr_l, va_l, sh_l = tot / max(n, 1), vt / max(vn, 1), vs / max(vsn, 1)
        gap = sh_l - va_l
        print('EPOCH %d  train %.4f | valid %.4f | valid(shuffled g) %.4f | GAP %+.4f  [%.1f min]'
              % (ep, tr_l, va_l, sh_l, gap, (time.time() - t0) / 60), flush=True)
        print('   gap > 0 means the model IS using the geometry channel. gap ~ 0 means it is dead.',
              flush=True)
        hist.append(dict(epoch=ep, train=tr_l, valid=va_l, valid_shuffled=sh_l, gap=gap))
        torch.save({'model_state': model.state_dict(), 'network_parameter': npm,
                    'vocabulary': ck['vocabulary'], 'epoch': ep, 'history': hist,
                    'cond_repeat': a.cond_repeat, 'geom_rbf': a.geom_rbf},
                   os.path.join(a.out, 'geom_ep%d.ckpt' % ep))
        json.dump(hist, open(os.path.join(a.out, 'history.json'), 'w'), indent=1)

    print('\ndone in %.1f min' % ((time.time() - t0) / 60))
    return 0


if __name__ == '__main__':
    sys.exit(main())
