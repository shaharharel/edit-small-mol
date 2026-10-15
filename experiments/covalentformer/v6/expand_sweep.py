#!/usr/bin/env python3
"""How many NEW pairs would we get by grouping on TARGET instead of PAPER, at various Tanimoto?

The current corpus requires both ligands to come from the same PAPER. Dropping that to "same
protein, any paper" is the largest available lever on corpus size. Raising the Tanimoto floor at
the same time trades volume for edits that are closer to true matched pairs.

Dedup against the existing corpus is on the canonical SMILES PAIR, not on ligand codes: the same
molecule appears under different three-letter codes across entries, so a code-pair check would
report molecules we already have as new.

Breadth is reported separately from volume, because they are not the same thing and only one of
them helps a target-split benchmark: a thousand more pairs on SARS-CoV-2 Mpro deepen one target,
they do not teach generalisation across targets.
"""
import collections
import itertools
import json
import os

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
src = open(os.path.join(HERE, 'build_pairs_v7.py')).read()
ns = {}
exec(compile(src.split('def ')[0], 'c', 'exec'), ns)
COFACTOR, NOTMED = ns['COFACTOR'], ns['NOT_MEDCHEM']
HMIN, HMAX, CAP = ns['HEAVY_MIN'], ns['HEAVY_MAX'], ns['PER_TARGET_CAP']

meta = json.load(open(D + '/meta.json'))
lig = json.load(open(D + '/ligand_smiles.json'))

by_t = collections.defaultdict(set)
for pid, v in meta.items():
    for u in (v.get('unp') or []):
        for l in (v.get('lig') or []):
            if l not in COFACTOR and l not in NOTMED:
                by_t[u].add(l)

fp, canon = {}, {}
def prep(code):
    if code in fp:
        return fp[code]
    s = lig.get(code)
    m = Chem.MolFromSmiles(s) if s else None
    if m is None or not (HMIN <= m.GetNumHeavyAtoms() <= HMAX):
        fp[code] = None
        return None
    canon[code] = Chem.MolToSmiles(m)
    fp[code] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    return fp[code]

# existing corpus, keyed by canonical SMILES pair
have_smi, have_targets = set(), set()
for line in open(D + '/v7_pairs.jsonl'):
    r = json.loads(line)
    a, b = Chem.MolFromSmiles(r['input']), Chem.MolFromSmiles(r['analogue'])
    if a and b:
        have_smi.add(frozenset((Chem.MolToSmiles(a), Chem.MolToSmiles(b))))
    have_targets.add(r['target'])
print('existing corpus: %s unique SMILES pairs over %s targets'
      % (format(len(have_smi), ','), format(len(have_targets), ',')))

# precompute per-target eligible ligands once
elig = {t: [c for c in sorted(ls) if prep(c) is not None] for t, ls in by_t.items()}
elig = {t: ls for t, ls in elig.items() if len(ls) >= 2}
print('targets with >=2 eligible ligands: %s\n' % format(len(elig), ','))

print('%-6s %10s %10s %10s %10s %9s %s' % ('TC', 'pairs', 'NEW', 'NEW@cap', 'new tgts',
                                           'top8 sh.', 'biggest targets (new pairs)'))
for TC in (0.50, 0.60, 0.70):
    tot = 0
    new_by_t = collections.Counter()
    seen = set()
    for t, ls in elig.items():
        for a, b in itertools.combinations(ls, 2):
            if DataStructs.TanimotoSimilarity(fp[a], fp[b]) < TC:
                continue
            tot += 1
            key = frozenset((canon[a], canon[b]))
            if key in have_smi or key in seen:
                continue
            seen.add(key)
            new_by_t[t] += 1
    new = sum(new_by_t.values())
    capped = sum(min(n, CAP) for n in new_by_t.values())
    new_tgts = sum(1 for t in new_by_t if t not in have_targets)
    top8 = sum(n for _, n in new_by_t.most_common(8))
    print('%-6.2f %10s %10s %10s %10s %8.0f%% %s'
          % (TC, format(tot, ','), format(new, ','), format(capped, ','),
             format(new_tgts, ','), 100 * top8 / max(new, 1),
             ', '.join('%s:%s' % (t, n) for t, n in new_by_t.most_common(4))))
