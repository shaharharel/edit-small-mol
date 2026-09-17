#!/usr/bin/env python
"""COHORT SHIFT: does the steered model produce a DIFFERENT POPULATION than the baseline?

This is the product claim, and none of the loss metrics test it.

  GAP_perm / GAP_flip  one model, instruction scrambled/reversed  -> does it USE the token
  BENEFIT              instr model vs none model, held-out loss   -> is the token WORTH anything
  obedience            paired, same anchor+seed, UP vs DOWN       -> null 0.5, WITHIN one model

All four are blind to the thing a chemist cares about: point the model at 5,000 anchors, ask
for UP, and see whether the resulting COHORT has a different parameter distribution than a
model that was never given the instruction. That is what this file measures.

THREE ARMS, because "more planar than baseline" is ambiguous about which baseline:

  prior    the raw mol2mol prior             -- no covalent fine-tuning at all
  none     the covalent-FT arm, no token     -- isolates the INSTRUCTION from the FINE-TUNING
  instr    the steered arm, under UP and DOWN

The manuscript's planarity effect was measured against the PRIOR. A claim of "our steering
makes molecules more planar" that only beats the prior may be entirely the covalent
fine-tuning, which the `none` arm already has. Both contrasts get reported.

DISTRIBUTIONS, NOT JUST MEANS. A mean shift can hide a bimodal split, and planarity is KNOWN
to be bimodal here (corpus median 24.9 deg with 41.9% over 30 and 21.4% under 5). So this
reports median and IQR alongside the mean, and runs a two-sample KS test on the full
distributions. A significant KS with an unmoved mean is a real and reportable outcome.

EXCLUSIONS ARE COUNTED, NEVER SILENT. Rows where a molecule is invalid or the param is not
computable are reported separately -- a previous "21% invalid SMILES" headline in this project
was ~1% model failure and the rest scorer failure.
"""
from __future__ import annotations
import os, sys, csv, json, math, argparse, random, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_and_score import sample, param_value                      # noqa: E402
from rdkit import Chem, RDLogger                                        # noqa: E402
RDLogger.DisableLog('rdApp.*')
import torch                                                            # noqa: E402
from train_steer_v5 import SteerNet, I2N                                # noqa: E402
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer  # noqa


def load(ckpt, dev):
    """Load a steering OR a role checkpoint.

    THE GUARD BELOW IS THE MOST IMPORTANT LINE IN THIS FILE. Role checkpoints are `PhaseA`
    with a `role_emb` table; steering checkpoints are `SteerNet` with `instr_emb`. Both call
    load_state_dict(strict=False), so loading one into the other SUCCEEDS SILENTLY and leaves
    the conditioning embedding RANDOMLY INITIALISED. The model still generates valid
    molecules, the run still finishes, and every conditioning number it produces is noise.
    That has already happened once here (a probe that "measured" a random model). So: after
    loading, assert that the conditioning tensor actually came from the file.
    """
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    # KEY NAME DIFFERS BY VINTAGE: the steering trainer writes 'network_state', the older
    # Phase A role trainer writes 'model_state'. Resolve it rather than assume -- assuming
    # raised KeyError here, which is the GOOD failure; the bad one is a .get() default of {}
    # that would leave the whole network random and still generate valid-looking molecules.
    sd = ck.get('network_state', ck.get('model_state'))
    assert sd is not None, 'no weights in %s (keys: %s)' % (ckpt, list(ck)[:8])
    is_role = any(k.startswith('role_emb') for k in sd)
    # THE RAW PRIOR HAS NEITHER CONDITIONING TABLE -- it IS the unconditioned baseline, so
    # force mode='none' rather than asserting. The assert below is for a checkpoint that
    # CLAIMS to be conditioned and is not; this is a model that never claimed to be.
    if not is_role and not any(k.startswith('instr_emb') for k in sd):
        m = SteerNet(mode='none', **dict(ck['network_parameter']))
        m.load_state_dict(sd, strict=False)
        return m.to(dev).eval(), vocab, SMILESTokenizer(), 'none', None
    if is_role:
        from train_phaseA import PhaseA, ROLES
        mode = ck.get('mode', 'role')
        m = PhaseA(mode=mode, **dict(ck['network_parameter']))
        cond_key = 'role_emb.weight'
        vocab_labels = ck.get('roles') or ROLES
    else:
        mode = ck.get('mode', 'instr')
        m = SteerNet(mode=mode, **dict(ck['network_parameter']))
        cond_key = 'instr_emb.weight'
        vocab_labels = ck.get('instr_vocab')
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if mode not in ('none',):
        assert cond_key in sd, (
            'CONDITIONING NOT IN CHECKPOINT: %s has no %s. Refusing to run -- strict=False '
            'would leave it RANDOM and every number below would be noise.' % (ckpt, cond_key))
        assert cond_key not in missing, (
            'CONDITIONING DID NOT LOAD: %s stayed at its random init for %s.' % (cond_key, ckpt))
    return m.to(dev).eval(), vocab, SMILESTokenizer(), mode, vocab_labels


def ks_2samp(a, b):
    """Two-sample KS. Hand-rolled to avoid a scipy dependency; returns (D, approx p)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return None, None
    allv = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, allv, side='right') / n1
    cdf2 = np.searchsorted(b, allv, side='right') / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def describe(vals):
    v = [x for x in vals if x is not None and not isinstance(x, str)]
    if not v:
        return None
    a = np.asarray(v, float)
    return {'n': len(a), 'mean': round(float(a.mean()), 4),
            'median': round(float(np.median(a)), 4),
            'p25': round(float(np.percentile(a, 25)), 4),
            'p75': round(float(np.percentile(a, 75)), 4),
            'sd': round(float(a.std()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', required=True)
    ap.add_argument('--instr-ckpt', required=True)
    ap.add_argument('--none-ckpt', default='')
    ap.add_argument('--prior-ckpt', default='')
    ap.add_argument('--valid', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=20260916)
    a = ap.parse_args()
    dev = (('cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
           if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    rows = list(csv.DictReader(open(a.valid)))
    # SHUFFLE BEFORE SLICING. The valid CSVs are written in MW-ASCENDING corpus order, so
    # taking the head samples the smallest molecules. That single mistake produced three
    # separate wrong conclusions in this project (a phantom truncation bug, a bogus
    # "correction" of it, and an unrepresentative tie rate).
    random.Random(a.seed).shuffle(rows)
    anchors = [r['anchor'] for r in rows][:a.n]
    print('param %s | anchors %d | device %s' % (a.param, len(anchors), dev))
    sys.stdout.flush()

    cohorts = {}
    mi, vocab, tok, mode, labels = load(a.instr_ckpt, dev)
    if mode == 'role':
        # ROLE MODE: 4 tokens, not a direction. One cohort PER ROLE, each from the SAME
        # anchors and seed, so the only thing that differs between cohorts is the token.
        from train_phaseA import ROLES
        labels = labels or ROLES
        for i, r in enumerate(labels):
            cohorts['role_%s' % r] = sample(mi, vocab, tok, anchors, i, dev, seed=a.seed)
            print('  %s done' % r); sys.stdout.flush()
    else:
        cohorts['instr_UP'] = sample(mi, vocab, tok, anchors, I2N['UP'], dev, seed=a.seed)
        print('  instr UP done'); sys.stdout.flush()
        cohorts['instr_DOWN'] = sample(mi, vocab, tok, anchors, I2N['DOWN'], dev, seed=a.seed)
        print('  instr DOWN done'); sys.stdout.flush()
    for tag, ck in (('none', a.none_ckpt), ('prior', a.prior_ckpt)):
        if not ck or not os.path.exists(ck):
            print('  %s: SKIPPED (no checkpoint)' % tag); continue
        m2, v2, t2, _m2mode, _ = load(ck, dev)
        # The cond id is ignored when mode='none'; passed only to keep one call signature.
        cohorts[tag] = sample(m2, v2, t2, anchors, 0, dev, seed=a.seed)
        print('  %s done' % tag); sys.stdout.flush()

    vals, excl = {}, {}
    for tag, smis in cohorts.items():
        vv, bad = [], collections.Counter()
        for idx, s in enumerate(smis):
            m = Chem.MolFromSmiles(s) if s else None
            if m is None:
                bad['invalid_smiles'] += 1; continue
            if a.param == 'role':
                # ROLE IS CATEGORICAL -- there is no scalar to order. The cohort-level
                # quantity that DOES differ by role is HOW MUCH OF THE MOLECULE MOVED:
                # heavy-atom delta against the anchor. EDIT_WARHEAD should shift a small,
                # specific region; EDIT_SCAFFOLD a large one. Reported as |delta HAC|.
                am = Chem.MolFromSmiles(anchors[idx]) if idx < len(anchors) else None
                if am is None:
                    bad['anchor_unparseable'] += 1; continue
                vv.append(abs(m.GetNumHeavyAtoms() - am.GetNumHeavyAtoms()))
                continue
            p = param_value(s, a.param)
            if p is None:
                bad['param_not_computable'] += 1; continue
            vv.append(p)
        vals[tag] = vv; excl[tag] = dict(bad)

    out = {'param': a.param, 'anchors': len(anchors), 'seed': a.seed,
           'cohorts': {t: describe(v) for t, v in vals.items()},
           'excluded': excl, 'contrasts': {}}

    def contrast(x, y):
        dx, dy = describe(vals.get(x, [])), describe(vals.get(y, []))
        if not dx or not dy:
            return None
        d, p = ks_2samp([v for v in vals[x] if not isinstance(v, str)],
                        [v for v in vals[y] if not isinstance(v, str)])
        return {'mean_shift': round(dx['mean'] - dy['mean'], 4),
                'median_shift': round(dx['median'] - dy['median'], 4),
                'ks_D': None if d is None else round(d, 4),
                'ks_p': None if p is None else float('%.3g' % p)}

    if mode == 'role':
        # PER ROLE, NEVER POOLED. valid_strat is role-REBALANCED by design (LINKER ~10x
        # enriched vs train, 45x spread in train itself), so any pooled role number is a
        # MIX ARTIFACT, not a model property. That trap has already corrupted several
        # figures in this project.
        base = [x for x in vals.get('none', []) if not isinstance(x, str)]
        for t in [k for k in vals if k.startswith('role_')]:
            out['contrasts']['%s_vs_none' % t] = contrast(t, 'none')
        out['role_note'] = ('Reported PER ROLE. Do NOT pool: the valid split is '
                            'role-rebalanced, so a pooled figure measures the mix.')
        json.dump(out, open(a.out, 'w'), indent=2)
        print('\n=== PER-ROLE COHORTS (%s) -- NEVER POOL THESE ===' % a.param)
        for t, d in out['cohorts'].items():
            if d:
                print('  %-24s n=%5d mean %8.3f median %8.3f' % (t, d['n'], d['mean'], d['median']))
        for k, c in out['contrasts'].items():
            if c:
                print('  %-24s mean %+8.4f  KS D=%s p=%s' % (k, c['mean_shift'], c['ks_D'], c['ks_p']))
        print('\nwrote %s' % a.out)
        return

    # THE INSTRUCTION's own effect: UP vs DOWN from identical anchors and seed.
    out['contrasts']['UP_vs_DOWN'] = contrast('instr_UP', 'instr_DOWN')
    # Is the steered cohort different from a model with NO instruction but the SAME training?
    out['contrasts']['UP_vs_none'] = contrast('instr_UP', 'none')
    out['contrasts']['DOWN_vs_none'] = contrast('instr_DOWN', 'none')
    # Is ANY of it just the covalent fine-tuning rather than the instruction?
    out['contrasts']['none_vs_prior'] = contrast('none', 'prior')
    out['contrasts']['UP_vs_prior'] = contrast('instr_UP', 'prior')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('\n=== COHORT DISTRIBUTIONS (%s) ===' % a.param)
    print('%-12s %6s %9s %9s %9s %9s' % ('cohort', 'n', 'mean', 'median', 'p25', 'p75'))
    for t, d in out['cohorts'].items():
        if d:
            print('%-12s %6d %9.3f %9.3f %9.3f %9.3f'
                  % (t, d['n'], d['mean'], d['median'], d['p25'], d['p75']))
    print('\n=== CONTRASTS ===')
    for k, c in out['contrasts'].items():
        if c:
            print('  %-16s mean %+8.4f  median %+8.4f  KS D=%s p=%s'
                  % (k, c['mean_shift'], c['median_shift'], c['ks_D'], c['ks_p']))
    print('\nUP_vs_none and DOWN_vs_none are the PRODUCT claim.')
    print('none_vs_prior tells you how much is the covalent FINE-TUNING rather than steering.')
    print('A significant KS with an unmoved mean is a REAL result -- report it, do not bury it.')
    print('wrote %s' % a.out)


if __name__ == '__main__':
    main()
