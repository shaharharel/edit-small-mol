#!/usr/bin/env python
"""Score the steerability probe: does flipping one direction word move the property?

The test is PAIRED ON THE INPUT MOLECULE, not on the prompt. This val set asks about the same
molecule several times, and a test paired on prompts treats up to ten correlated rows as ten
observations -- the mistake that made an earlier +0.033 look significant at p=0.014 when the real n
was 94 inputs over 16 targets.

Per axis we report:
  more, less      mean signed property change (generated minus input) under each direction word
  more-less       the paired difference, which is the steerability effect, with its expected sign
  neutral         the same quantity with no direction stated, which is where zero actually sits
  agree           fraction of inputs where the paired difference has the expected sign
  p               paired Wilcoxon over inputs, then Holm-corrected across the axes

A model steerable on an axis shows a paired difference of the expected sign on a majority of
inputs. A model steerable on nothing shows differences centred on the neutral arm.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

HBD = Chem.MolFromSmarts('[$([NX3;H2,H1;!$(NC=O)]),$([OX2H])]')


def props(s):
    m = Chem.MolFromSmiles(s or '')
    if m is None:
        return None
    return {'cLogP': Crippen.MolLogP(m), 'TPSA': rdMolDescriptors.CalcTPSA(m),
            'MW': Descriptors.MolWt(m), 'RotB': rdMolDescriptors.CalcNumRotatableBonds(m),
            'RingCount': rdMolDescriptors.CalcNumRings(m),
            'HBD': len(m.GetSubstructMatches(HBD))}


def gen_smiles(r):
    return r.get('gen_smiles_canonical') or r.get('gen_smiles') or ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()

    meta = {}
    for line in open(a.prompts):
        r = json.loads(line)
        meta[(r['input_smiles'], r['arm'])] = r
    # generations are keyed the same way; an eval harness that drops `arm` cannot be scored here
    got = {}
    n_missing_arm = 0
    for line in open(a.gens):
        r = json.loads(line)
        arm, inp = r.get('arm'), r.get('input_smiles')
        if not arm or not inp:
            n_missing_arm += 1
            continue
        got[(inp, arm)] = r
    if n_missing_arm:
        print('WARNING %d generations carry no arm/input and cannot be scored' % n_missing_arm)

    axes = sorted({m['axis'] for m in meta.values()} - {'neutral'})
    valid = sum(1 for k in got if Chem.MolFromSmiles(gen_smiles(got[k])))
    print('generations scored: %d (valid %d, %.1f%%)\n' % (len(got), valid,
                                                           100 * valid / max(len(got), 1)))

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    rows, pvals = [], []
    for axis in axes:
        prop = next(m['prop'] for m in meta.values() if m['axis'] == axis)
        d_more, d_less, diffs, neu = [], [], [], []
        for inp in {m['input_smiles'] for m in meta.values()}:
            ip = props(inp)
            if ip is None:
                continue
            gm = got.get((inp, '%s_more' % axis))
            gl = got.get((inp, '%s_less' % axis))
            gn = got.get((inp, 'neutral'))
            pm = props(gen_smiles(gm)) if gm else None
            pl = props(gen_smiles(gl)) if gl else None
            pn = props(gen_smiles(gn)) if gn else None
            if pn is not None:
                neu.append(pn[prop] - ip[prop])
            if pm is None or pl is None:
                continue
            dm, dl = pm[prop] - ip[prop], pl[prop] - ip[prop]
            d_more.append(dm)
            d_less.append(dl)
            diffs.append(dm - dl)
        if len(diffs) < 8:
            print('%-14s too few paired inputs (%d)' % (axis, len(diffs)))
            continue
        agree = sum(1 for x in diffs if x > 0) / len(diffs)
        pv = float('nan')
        if wilcoxon is not None and any(x != 0 for x in diffs):
            try:
                pv = wilcoxon(diffs, alternative='greater').pvalue
            except Exception:
                pv = float('nan')
        rows.append((axis, prop, len(diffs), st.mean(d_more), st.mean(d_less),
                     st.mean(diffs), st.mean(neu) if neu else float('nan'), agree, pv))
        if pv == pv:
            pvals.append((axis, pv))

    print('%-14s %-10s %5s %9s %9s %11s %9s %7s %10s' %
          ('axis', 'prop', 'n', 'more', 'less', 'more-less', 'neutral', 'agree', 'p'))
    for axis, prop, n, dm, dl, dd, nu, ag, pv in rows:
        print('%-14s %-10s %5d %+9.2f %+9.2f %+11.2f %+9.2f %6.0f%% %10.3g'
              % (axis, prop, n, dm, dl, dd, nu, 100 * ag, pv))

    if pvals:
        pvals.sort(key=lambda x: x[1])
        print('\nHolm across %d axes:' % len(pvals))
        for i, (axis, pv) in enumerate(pvals):
            thr = 0.05 / (len(pvals) - i)
            print('  %-14s p=%-10.3g threshold %.4f  %s'
                  % (axis, pv, thr, 'STEERABLE' if pv < thr else 'not shown'))
    print('\n"more-less" is the effect; "neutral" is where zero sits when no direction is stated.')

    if a.report:
        with open(a.report, 'w') as fh:
            json.dump([{'axis': r[0], 'prop': r[1], 'n': r[2], 'more': r[3], 'less': r[4],
                        'more_minus_less': r[5], 'neutral': r[6], 'agree': r[7], 'p': r[8]}
                       for r in rows], fh, indent=1)
        print('written to %s' % a.report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
