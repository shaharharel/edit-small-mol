#!/usr/bin/env python
"""Canonical, salt-aware pair validation. This is the check whose absence produced the bug.

WHAT WENT WRONG. The pair builders compared hit and analogue as RAW SMILES STRINGS. Two SMILES can
spell the same molecule -- different atom order, different ring-closure digits, different kekulisation,
and '/C=C/' vs '\\C=C\\' (both trans). So 9 rung-3 rows and 1 rung-3 test row carried hit == analogue
while the raw strings differed, and the op classifier labelled them NONE because it compared heavy-atom
COUNTS (difference 0) and never compared the molecules.

WHAT THIS ENFORCES, in order:
  1. both sides parse
  2. largest-fragment (salt/counterion) stripped before comparison -- a hydrochloride and its free base
     are the same design, and covPROG had 11 such pairs
  3. canonical SMILES of the PARENTS must differ
Returns (ok, reason) so callers can count rejections instead of silently dropping rows.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_LFC = rdMolStandardize.LargestFragmentChooser()

def parent_canon(smi, min_heavy=6):
    m = Chem.MolFromSmiles(smi or '')
    if m is None:
        return None
    m = _LFC.choose(m)
    if m is None or m.GetNumHeavyAtoms() < min_heavy:
        return None
    return Chem.MolToSmiles(m)

def check_pair(hit, analogue):
    ph, pa = parent_canon(hit), parent_canon(analogue)
    if ph is None:
        return False, 'hit_unparseable'
    if pa is None:
        return False, 'analogue_unparseable'
    if ph == pa:
        return False, 'identical_after_canonicalisation'
    return True, 'ok'

def filter_jsonl(src, dst, hit_key='hit', ana_key='analogue'):
    import json
    from collections import Counter
    kept = 0; why = Counter()
    with open(dst, 'w') as o:
        for line in open(src):
            r = json.loads(line)
            ok, reason = check_pair(r.get(hit_key), r.get(ana_key))
            why[reason] += 1
            if ok:
                o.write(json.dumps(r) + '\n'); kept += 1
    return kept, why
