"""HOW MANY LENGTH-MATCHED DPO PAIRS SURVIVE THE ORACLE'S RESOLUTION LIMIT?

WHY THIS HAS TO BE MEASURED BEFORE THE DPO STAGE IS PLANNED. The preference pairs are the only part
of the pipeline where a free descriptor genuinely cannot help: matched on bond count, warhead class
and heavy-atom count, two fragments differ only in topology, and the question "which one reaches
this gap" has no 2D answer. That is the whole reason DPO is worth running.

But the boundary calibration puts a floor under how fine a distinction the oracle can actually
resolve. Displacing the target cysteine and measuring pass rate gave:

    delta   0A     1A     2A     3A     4A     6A     8A
    pass  62.8%  61.2%  50.4%  32.6%  20.9%  20.9%   7.0%        half-max 4.0 A

So a pair whose members differ by less than ~4 A is inside the oracle's noise: it will label them
differently, but that label is not reliable, and DPO trained on it learns the oracle's jitter.

THIS INVALIDATES THE EXAMPLE THIS PROJECT HAS BEEN QUOTING. The showcase pair -- 4-bond
ortho-phenyl (median reach 4.99 A) versus 4-bond para-phenyl (8.21 A) -- separates by 3.2 A, which
is BELOW 4.0 A. It is not usable. An earlier estimate of ~3 A resolution (from a smaller boundary
sample) would have admitted it; the fuller sample does not.

WHAT IS COUNTED HERE. For each candidate pair (A, B) matched on bond-count bin and warhead class,
and each gap g drawn from A's envelope:
    A reaches g            (witnessed by a real conformer)
    B misses g by MARGIN   = min over B's conformers of the (d, theta) distance to g, in Angstrom
                             with theta converted at 0.05 A/degree so one number orders both axes
A pair counts only if margin > MIN_MARGIN. The output is the surviving pair count, which decides
whether the DPO stage has data at all or whether the fragment vocabulary must be enlarged first.

Reported at several margins so the cost of the threshold is visible rather than asserted.
"""
import os, sys, json, argparse, collections, itertools
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import electrophile_index

D_TOL, A_TOL = 0.5, 20.0
DEG_TO_A = 0.05          # 20 deg of angular slack ~ 1 A of positional slack
MARGINS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def bond_count(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    star = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    el, _ = electrophile_index(m)
    if not star or el is None:
        return None
    try:
        return len(Chem.GetShortestPath(m, star[0], el)) - 1
    except Exception:
        return None


def bin_of(nb):
    return ('3-4' if nb <= 4 else '5-6' if nb <= 6 else '7-9' if nb <= 9 else
            '10-13' if nb <= 13 else '14+')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='experiments/covalentformer/data/envelopes.jsonl')
    ap.add_argument('--out', default='experiments/covalentformer/data/dpo_pairs.json')
    ap.add_argument('--max-frag', type=int, default=900)
    ap.add_argument('--gaps', type=int, default=6)
    a = ap.parse_args()
    rng = np.random.default_rng(20260913)

    E = {}
    for line in open(a.env):
        e = json.loads(line)
        E[e['frag']] = (np.array(e['d'], float), np.array(e['theta'], float),
                        e.get('cls', '?'), e.get('n_heavy', 0))
    frags = sorted(E)
    if len(frags) > a.max_frag:
        frags = [frags[i] for i in rng.choice(len(frags), a.max_frag, replace=False)]
    print('fragments considered: %d' % len(frags))

    meta = {}
    for f in frags:
        nb = bond_count(f)
        if nb is None:
            continue
        meta[f] = (bin_of(nb), E[f][2], E[f][3])
    print('with a resolvable attachment->electrophile path: %d' % len(meta))

    groups = collections.defaultdict(list)
    for f, (b, cls, nh) in meta.items():
        groups[(b, cls)].append(f)
    print('matched groups (bond-bin x warhead class): %d  sizes %s'
          % (len(groups), sorted((len(v) for v in groups.values()), reverse=True)[:8]))

    counts = {m: 0 for m in MARGINS}
    by_bin = collections.defaultdict(lambda: {m: 0 for m in MARGINS})
    considered = 0
    for (b, cls), fs in groups.items():
        if len(fs) < 2:
            continue
        pairs = list(itertools.combinations(fs, 2))
        if len(pairs) > 4000:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 4000, replace=False)]
        for A, B in pairs:
            dA, tA, _c, nhA = E[A]
            dB, tB, _c2, nhB = E[B]
            if abs(nhA - nhB) > 4:          # also match on size, not just bond count
                continue
            considered += 1
            idx = rng.choice(len(dA), min(a.gaps, len(dA)), replace=False)
            for j in idx:
                g_d, g_t = float(dA[j]), float(tA[j])
                # A reaches g by construction (g is one of A's own conformer points).
                # How badly does B miss it? Distance in a combined (d, theta) metric.
                sep = np.sqrt((dB - g_d) ** 2 + ((tB - g_t) * DEG_TO_A) ** 2)
                margin = float(sep.min())
                if margin <= D_TOL:
                    continue                # B reaches it too: no contrast, not a pair
                for m in MARGINS:
                    if margin > m:
                        counts[m] += 1
                        by_bin[b][m] += 1

    print('\nLENGTH-MATCHED PAIRS SURVIVING EACH MARGIN (pair x gap instances)')
    print('  %10s %12s   %s' % ('margin A', 'surviving', 'note'))
    for m in MARGINS:
        note = ''
        if m == 4.0:
            note = '<-- REQUIRED by the 4.0 A half-max resolution'
        elif m == 3.0:
            note = '(the ~3 A figure quoted earlier, now known to be too lenient)'
        print('  %10.1f %12d   %s' % (m, counts[m], note))
    print('\n  candidate pairs examined: %d' % considered)

    print('\n  BY BOND-COUNT BIN at the required 4.0 A margin:')
    for b in sorted(by_bin):
        print('    %-6s %8d' % (b, by_bin[b][4.0]))

    surv = counts[4.0]
    print()
    if surv < 2000:
        print('  VERDICT: TOO FEW. %d usable pairs will not support a DPO stage; the fragment'
              % surv)
        print('  vocabulary has to be enlarged before preference learning is worth running.')
    else:
        print('  VERDICT: %d usable pairs. Sufficient to attempt DPO, with the caveat that they'
              % surv)
        print('  concentrate in whichever bins dominate above.')
    json.dump({'counts': {str(k): v for k, v in counts.items()},
               'by_bin': {k: {str(m): v for m, v in d.items()} for k, d in by_bin.items()},
               'considered': considered, 'min_margin_required': 4.0},
              open(a.out, 'w'), indent=1)
    print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
