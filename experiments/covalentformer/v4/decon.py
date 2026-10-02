#!/usr/bin/env python
"""Correct decontamination against the rung-3 eval analogues, plus a measurement of what the broken
version let through.

THE BUG IN build_joint.clean(). It did:
    ci = canon(r.get('input'));  co = canon(r.get('output'))
and dropped the row if either landed in the eval set. Both lookups fail silently on v4's schema:
  * v4 rows carry 'hit'/'analogue', not a bare 'input' -> ci is always None
  * 'output' is 'Change: REPLACE\\nAnalogue: CC...' -> MolFromSmiles returns None -> co is always None
So the filter compared None against the eval set and kept everything. It printed
"79241 rows -> 79241 kept, 0 dropped" on every run, which is what a working filter would never do.

CONSEQUENCE, MEASURED SEPARATELY: v4's rung-3 training rows contain 99.5% of the R3_COV test fold,
98.5% of valid and 99.9% of mpro on the exact (hit, analogue) key, and 439 of 440 eval analogues.
Any rung-3 generalisation number from this model would have been memorisation.

THE FIX. Pull the analogue out of wherever it actually lives -- the 'analogue' field, or the text
after 'Analogue: ' in a formatted response -- and compare THAT. Also compare the hit, since a pair
whose hit is an eval analogue leaks the molecule too.
"""
from __future__ import annotations
import json, os, re, sys

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ANA = re.compile(r'Analogue:\s*(\S+)')
EVAL_FOLDS = ('test', 'valid', 'mpro')
EVAL_VARIANTS = ('R3_COV', 'R3_COVPOCK', 'R3_SMILES')


def canon(smi):
    """Canonical, stereo-stripped. None propagates -- never fall back to the raw string, or an
    unparseable row compares equal only to itself and silently passes the filter."""
    if not smi or not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def row_molecules(r):
    """Every molecule a row exposes, however the row is shaped."""
    out = set()
    for k in ('analogue', 'hit', 'input', 'src', 'tgt'):
        c = canon(r.get(k))
        if c:
            out.add(c)
    o = r.get('output')
    if isinstance(o, str):
        m = ANA.search(o)
        if m:
            c = canon(m.group(1))
            if c:
                out.add(c)
        else:
            c = canon(o)            # bare-SMILES outputs still work
            if c:
                out.add(c)
    return out


def load_eval_analogues(ladder_dir):
    out, n = set(), 0
    for v in EVAL_VARIANTS:
        for f in EVAL_FOLDS:
            p = os.path.join(ladder_dir, 'rung3_%s_%s.jsonl' % (v, f))
            if not os.path.exists(p):
                continue
            n += 1
            for line in open(p):
                r = json.loads(line)
                for c in row_molecules(r):
                    out.add(c)
    return out, n


def clean(rows, evalset, label, verbose=True):
    kept, dropped = [], 0
    for r in rows:
        if row_molecules(r) & evalset:
            dropped += 1
            continue
        kept.append(r)
    if verbose:
        print('  %-10s %7d rows -> %7d kept, %7d DROPPED as eval-analogue contact (%.2f%%)'
              % (label, len(rows), len(kept), dropped, 100.0 * dropped / max(len(rows), 1)))
    return kept


if __name__ == '__main__':
    ladder = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ladder_v3')
    ev, nf = load_eval_analogues(ladder)
    print('eval molecule set: %d distinct (from %d fold files)' % (len(ev), nf))
    print('  (the broken loader found 440 -- it only read the `analogue` field)')
    for name in sys.argv[2:]:
        rows = [json.loads(l) for l in open(name)]
        clean(rows, ev, os.path.basename(name))
