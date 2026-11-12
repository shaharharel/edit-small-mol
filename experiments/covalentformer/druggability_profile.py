#!/usr/bin/env python
"""Druggability profile of our own generations, against the molecules chemists actually made.

WHY THIS EXISTS WITHOUT A BASELINE ATTACHED. Every pocket-conditioned generator we were asked to
compare against -- Delete, Diffleop, PROBE, PockLigGPT -- reports QED, synthetic accessibility and a
docking score, and three of them optimise one of those directly. We report none of them. That makes
"you never checked whether these molecules are drug-like or makeable" a free hit for any reviewer,
and it is free to answer, because the numbers need no GPU and no baseline: RDKit computes them.

WHAT THE RIGHT CEILING IS. A generator that beats the reference analogues on QED is not thereby
better -- it may simply be ignoring the series. So every arm is reported beside two anchors:

  input       the hit the model was given, i.e. where the series already sits
  reference   the analogue the chemists actually made from that hit

The question a chemist asks is not "is QED high" but "did the edit move QED and SA the way the
chemists' own edit did, or did it wander off". So the table reports the DELTA against the input for
every arm, and the reference's delta is the target.

SA SCORE IS THE RDKit CONTRIB IMPLEMENTATION (Ertl & Schuffenhauer), loaded from the RDKit install
rather than reimplemented; if it is unavailable the column is omitted rather than silently replaced
by a different scale, because an SA number on the wrong scale is worse than no SA number.

Docking is deliberately NOT here. A Vina column belongs in a run with a prepared receptor per
target and is a separate job; this covers everything that needs nothing but the SMILES.

    python experiments/covalentformer/druggability_profile.py \\
        --bench <sft_pocket.jsonl> \\
        --arm reference:<M_oracle.jsonl> --arm R->M:<generations.jsonl> \\
        --arm mismatch:<M_mismatch.jsonl> --arm joint:<pocket21k_on_21kval.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

_SA = None
try:                                     # RDKit ships sascorer under Contrib, not in the package
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer as _SA               # noqa: N813
except Exception:
    _SA = None


def profile(smi):
    m = Chem.MolFromSmiles(smi or '')
    if m is None:
        return None
    d = {'QED': QED.qed(m), 'MW': Descriptors.MolWt(m), 'cLogP': Crippen.MolLogP(m),
         'TPSA': rdMolDescriptors.CalcTPSA(m),
         'HBD': rdMolDescriptors.CalcNumHBD(m), 'HBA': rdMolDescriptors.CalcNumHBA(m),
         'RotB': rdMolDescriptors.CalcNumRotatableBonds(m),
         'Rings': rdMolDescriptors.CalcNumRings(m),
         'heavy': m.GetNumHeavyAtoms(),
         'Fsp3': rdMolDescriptors.CalcFractionCSP3(m)}
    if _SA is not None:
        d['SA'] = _SA.calculateScore(m)
    # Lipinski and the common lead-likeness gate, as pass/fail so the table can carry a rate
    d['ro5_ok'] = float(sum([d['MW'] > 500, d['cLogP'] > 5, d['HBD'] > 5, d['HBA'] > 10]) <= 1)
    d['leadlike_ok'] = float(250 <= d['MW'] <= 350 and d['cLogP'] <= 3.5 and d['RotB'] <= 7)
    return d


KEYS = ['QED', 'SA', 'MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotB', 'Rings', 'Fsp3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--arm', action='append', default=[], metavar='NAME:PATH')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    bench = {}
    for line in open(a.bench):
        r = json.loads(line)
        if r.get('split') == a.split:
            bench[r['pair']] = r

    arms = {}
    for spec in a.arm:
        name, path = spec.split(':', 1)
        if not os.path.exists(path):
            print('MISSING %s (%s)' % (name, path))
            continue
        arms[name] = {json.loads(l)['pair']: json.loads(l) for l in open(path)}

    # one row set: the pairs every arm answered, so the anchors and the arms are on identical rows
    shared = sorted(set(bench) & (set.intersection(*[set(v) for v in arms.values()])
                                 if arms else set()))
    print('pairs scored on every arm: %d\n' % len(shared))
    if not shared:
        return 1

    cols = [k for k in KEYS if k != 'SA' or _SA is not None]
    if _SA is None:
        print('NOTE: RDKit sascorer unavailable, SA column omitted rather than substituted\n')

    def row(name, get):
        vals = {k: [] for k in cols}
        ro5 = lead = n = 0
        for p in shared:
            pr = profile(get(p))
            if pr is None:
                continue
            n += 1
            for k in cols:
                vals[k].append(pr[k])
            ro5 += pr['ro5_ok']
            lead += pr['leadlike_ok']
        if not n:
            return None
        out = {'arm': name, 'n': n, 'ro5_pct': 100 * ro5 / n, 'leadlike_pct': 100 * lead / n}
        out.update({k: st.mean(vals[k]) for k in cols})
        return out

    def gen_of(arm):
        def f(p):
            r = arms[arm].get(p) or {}
            return r.get('gen_smiles_canonical') or r.get('gen_smiles') or ''
        return f

    rows = [row('input (the hit)', lambda p: bench[p].get('input_smiles')),
            row('reference analogue', lambda p: bench[p].get('analogue_smiles'))]
    for name in arms:
        rows.append(row(name, gen_of(name)))
    rows = [r for r in rows if r]

    hdr = '%-22s %5s' % ('arm', 'n') + ''.join('%8s' % k for k in cols) + '%8s%10s' % ('Ro5%', 'leadlike%')
    print(hdr)
    for r in rows:
        print('%-22s %5d' % (r['arm'], r['n']) + ''.join('%8.2f' % r[k] for k in cols)
              + '%8.0f%10.0f' % (r['ro5_pct'], r['leadlike_pct']))

    # the comparison that matters: delta against the input, with the reference's delta as target
    base = {k: rows[0][k] for k in cols}
    ref = {k: rows[1][k] for k in cols}
    print('\nchange from the hit (the reference row is the target, not a number to beat)')
    print('%-22s' % 'arm' + ''.join('%8s' % k for k in cols))
    for r in rows[1:]:
        print('%-22s' % r['arm'] + ''.join('%+8.2f' % (r[k] - base[k]) for k in cols))
    print('\ndistance from the reference\'s own move, |arm delta - reference delta|')
    print('%-22s' % 'arm' + ''.join('%8s' % k for k in cols))
    for r in rows[2:]:
        print('%-22s' % r['arm'] + ''.join('%8.2f' % abs((r[k] - base[k]) - (ref[k] - base[k]))
                                           for k in cols))
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
        print('\nwritten to %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
