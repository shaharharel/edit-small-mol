"""H3 BIDIRECTIONAL CONTROL: does the label token r actually steer generation?

Identical anchors, identical geometry. ONLY r changes. A conditioned model must reach MORE when
asked for r=1 than for r=0; a model that merely carries a bias shows no difference.

WHY THIS EXISTS AS A FILE. H3 was first run as an inline one-off and reported +17.5 points
(p<0.0001) as the night's strongest conditioning result. That command seeded numpy but NOT torch,
so the sampled molecules differed run to run, and it was never replicated. The cost of that omission
is measured: best-of-N gave +16.5 points on one draw and +4.1 on the next, and H1's shuffle gave
+4.3 unseeded but +0.3 (McNemar 96 vs 95, p=1.0000) once torch was seeded. A single draw from this
harness is not a result. This file seeds torch, takes --seed, and is meant to be run several times.

THE TWO CONTROLS THAT MAKE IT INTERPRETABLE, both computed here rather than assumed:
  DEGENERATION CHECK. r=0 must produce equally VALID and equally SCORABLE molecules. If r=0 merely
  makes the model emit garbage, a reach drop is degeneration scored as non-reach, not steering. The
  original run passed this (validity 96.2% vs 98.8%, scorable 93.8% both) and it is kept here.
  RANDOM FLOOR. A random in-vocabulary fragment satisfies a random gap 24.6% of the time, because
  the oracle tolerance admits ~28% of the vocabulary per gap. An r=0 arm at ~25% has fallen to
  chance, which is the strongest form of compliance; reading it against 0 would overstate the effect.

CIRCULARITY: scored by the envelope oracle the model was trained against. This measures whether the
conditioning is WIRED, not whether the molecules are good.
"""
import os, sys, csv, json, math, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from shuffle_test import load_model, sample, reaches

RANDOM_FLOOR = 0.246


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--valid', default='experiments/covalentformer/data/tierA_v3/valid.csv')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    m, vocab, tok, ep = load_model(a.ckpt, dev)
    rows = [r for r in csv.DictReader(open(a.valid)) if r['label'] == '1']
    rows = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)]
    anc = [r['anchor'] for r in rows]
    sca = [r['scaffold'] for r in rows]
    G = [[float(r['d']), float(r['cos_theta']), float(r['v_free'])] for r in rows]
    TH = [float(r['theta']) for r in rows]
    print('H3 (epoch %d, seed %d, n=%d, %s)' % (ep, a.seed, len(rows), dev))

    res = {}
    for lab in (1, 0):
        gens = []
        for i in range(0, len(rows), a.bs):
            gens += sample(m, vocab, tok, anc[i:i + a.bs], G[i:i + a.bs],
                           [lab] * len(anc[i:i + a.bs]), dev, temp=a.temp)
        valid = sum(1 for s in gens if s and Chem.MolFromSmiles(s) is not None)
        hits, scorable = [], 0
        for s, sc, gg, th in zip(gens, sca, G, TH):
            if not s or Chem.MolFromSmiles(s) is None:
                hits.append(False)
                continue
            h, _e = reaches(s, sc, gg[0], th)
            if h is None:
                hits.append(False)
            else:
                scorable += 1
                hits.append(bool(h))
        n = max(len(gens), 1)
        res[lab] = dict(hits=hits, validity=valid / n, scorable=scorable / n,
                        reach=float(np.mean(hits)))
        print('  r=%d  validity %5.1f%%  scorable %5.1f%%  reach %5.1f%%'
              % (lab, 100 * valid / n, 100 * scorable / n, 100 * np.mean(hits)))

    A, B = np.array(res[1]['hits']), np.array(res[0]['hits'])
    b = int((A & ~B).sum())
    c = int((B & ~A).sum())
    chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    p = math.erfc(math.sqrt(chi / 2.0)) if (b + c) else 1.0
    diff = A.mean() - B.mean()
    print('  difference %+.1f pts | McNemar r1-only %d, r0-only %d, chi2 %.2f, p=%.4f'
          % (100 * diff, b, c, chi, p))
    # DEGENERATION CONTROL, stated explicitly rather than left implicit
    dv = res[1]['validity'] - res[0]['validity']
    ds = res[1]['scorable'] - res[0]['scorable']
    degen = abs(dv) > 0.10 or abs(ds) > 0.10
    print('  validity gap %+.1f pts | scorable gap %+.1f pts -> %s'
          % (100 * dv, 100 * ds,
             'DEGENERATION: r=0 is emitting worse molecules, not non-reaching ones'
             if degen else 'no degeneration; r=0 molecules are equally well formed'))
    print('  r=0 vs the %.1f%% random floor: %+.1f pts'
          % (100 * RANDOM_FLOOR, 100 * (res[0]['reach'] - RANDOM_FLOOR)))
    verdict = ('COMPLIES' if (diff > 0.05 and p < 0.05 and not degen) else
               'NO CONTROL' if p >= 0.05 else 'AMBIGUOUS')
    print('  VERDICT: %s' % verdict)
    if a.out:
        json.dump({'seed': a.seed, 'epoch': ep, 'n': len(rows), 'diff': float(diff),
                   'b': b, 'c': c, 'p': p, 'verdict': verdict,
                   'r1': {k: v for k, v in res[1].items() if k != 'hits'},
                   'r0': {k: v for k, v in res[0].items() if k != 'hits'}},
                  open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
