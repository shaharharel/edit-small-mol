#!/usr/bin/env python3
"""Freeze two NEW labelling queues with pair-selection criteria that differ from g1-g4.

g1-g4 all used: same paper/series + Tanimoto >= 0.50 + same protein. g5 and g6 change the SELECTION
axis, which is why they get their own directories, their own stamps and their own manifests. A
future training run must not be able to pool a cross-paper pair with a same-paper one by accident.

  g5  CROSS-PAPER, same protein, Tanimoto >= 0.70. Breadth: adds targets the corpus does not have.
  g6  COVALENT-ENRICHED. Same cross-paper rule, but the input ligand must be covalently bound
      (a heavy atom within 2.1 A of a protein N/O/S, metals excluded) or carry a known warhead.
      Built from the g5 pool first, then topped up from TC >= 0.60 if the strict band is thin --
      the covalent slice is small and worth widening slightly rather than shipping 200 pairs.

Every candidate is deduplicated against ALL of g1-g4 on the canonical SMILES pair, so no pair is
labelled twice across generations.
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
COF, NOTMED, HMIN, HMAX = ns['COFACTOR'], ns['NOT_MEDCHEM'], ns['HEAVY_MIN'], ns['HEAVY_MAX']
import covalent_stats as C   # warhead SMARTS + the geometric covalency test

G5_TC, G6_TC = 0.70, 0.60
METALS = C.METALS

meta = json.load(open(D + '/meta.json'))
lig = json.load(open(D + '/ligand_smiles.json'))

# ---- what is already labelled anywhere in g1-g4 -------------------------------------------
import glob
labelled = set()
for f in glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')) + \
         glob.glob(os.path.join(HERE, 'label_out_g4', 'batch_*.json')):
    labelled |= set(json.load(open(f)))
print('pair ids already labelled in g1-g4: %s' % format(len(labelled), ','))

have_smi = set()
for line in open(D + '/v7_pairs.jsonl'):
    r = json.loads(line)
    a, b = Chem.MolFromSmiles(r['input']), Chem.MolFromSmiles(r['analogue'])
    if a and b:
        have_smi.add(frozenset((Chem.MolToSmiles(a), Chem.MolToSmiles(b))))
print('canonical SMILES pairs already in the built corpus: %s' % format(len(have_smi), ','))

# ---- covalency of an input ligand, from the evidence we already computed -------------------
cov_pdbs = set()
for line in open(D + '/v7_evidence.jsonl'):
    r = json.loads(line)
    c, _m = C.geometric(r)
    if c:
        cov_pdbs.add(r['pair'].split()[0])
print('PDB entries observed covalent in existing evidence: %s' % format(len(cov_pdbs), ','))

cache = {}
def prep(code):
    if code in cache:
        return cache[code]
    s = lig.get(code)
    m = Chem.MolFromSmiles(s) if s else None
    if m is None or not (HMIN <= m.GetNumHeavyAtoms() <= HMAX):
        cache[code] = None
    else:
        cache[code] = (AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048),
                       Chem.MolToSmiles(m), C.warheads_in(s), m.GetNumHeavyAtoms())
    return cache[code]

by_t = collections.defaultdict(set)
lig_pdbs = collections.defaultdict(set)
for pid, v in meta.items():
    for l in (v.get('lig') or []):
        lig_pdbs[l].add(pid)
    for u in (v.get('unp') or []):
        for l in (v.get('lig') or []):
            if l not in COF and l not in NOTMED:
                by_t[u].add(l)

g5, g6 = [], []
for t, ls in by_t.items():
    ok = [c for c in sorted(ls) if prep(c)]
    if len(ok) < 2:
        continue
    for a, b in itertools.combinations(ok, 2):
        fa, ca, wa, ha = cache[a]
        fb, cb, wb, hb = cache[b]
        s = DataStructs.TanimotoSimilarity(fa, fb)
        if s < G6_TC:
            continue
        key = frozenset((ca, cb))
        if key in have_smi:
            continue
        rec = {'target': t, 'input_code': a, 'analogue_code': b, 'tanimoto': round(s, 3),
               'input_smiles': lig[a], 'analogue_smiles': lig[b],
               'input_pdbs': sorted(lig_pdbs[a])[:4], 'analogue_pdbs': sorted(lig_pdbs[b])[:4],
               'heavy_input': ha, 'heavy_analogue': hb,
               'input_warheads': sorted(wa or []), 'analogue_warheads': sorted(wb or [])}
        covalent = bool(wa) or bool(set(lig_pdbs[a]) & cov_pdbs)
        if s >= G5_TC:
            g5.append(rec)
        if covalent:
            rec = dict(rec, covalent_evidence=('warhead' if wa else 'geometric'))
            g6.append(rec)

# g6 must not duplicate g5
g5keys = {frozenset((r['input_code'], r['analogue_code'])) for r in g5}
g6_only = [r for r in g6 if frozenset((r['input_code'], r['analogue_code'])) not in g5keys]
g6_in_g5 = len(g6) - len(g6_only)

print('\ng5  cross-paper TC>=%.2f            : %s pairs, %s targets'
      % (G5_TC, format(len(g5), ','), format(len({r['target'] for r in g5}), ',')))
print('g6  covalent, TC>=%.2f, NOT in g5   : %s pairs, %s targets  (%s covalent pairs already sit in g5)'
      % (G6_TC, format(len(g6_only), ','), format(len({r['target'] for r in g6_only}), ','),
         format(g6_in_g5, ',')))
wh = collections.Counter(w for r in g6_only for w in r['input_warheads'])
print('    warheads in g6 inputs: %s' % dict(wh.most_common(8)))

for name, rows, tc in (('g5', g5, G5_TC), ('g6', g6_only, G6_TC)):
    out = os.path.join(HERE, 'label_queue_%s' % name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'candidates.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    json.dump({
        'generation': '%s_crosspaper_tc%02d%s' % (name, int(tc * 100),
                                                  '_covalent' if name == 'g6' else ''),
        'n_pairs': len(rows),
        'n_targets': len({r['target'] for r in rows}),
        'selection': {
            'series_key': 'TARGET (UniProt accession) -- NOT the paper, unlike g1-g4',
            'tanimoto_floor': tc,
            'heavy_atom_range': [HMIN, HMAX],
            'cofactor_lipid_excluded': True,
            'per_target_cap': None,
            'covalent_required': name == 'g6',
        },
        'dedup': 'canonical SMILES pair, against the whole built v7 corpus (g1-g4)',
        'sentence_rules': 'unchanged from g3/g4: v2 input-anchored + pocket envelope',
        'note': ('These are same-series-like analogue pairs, NOT matched molecular pairs: no '
                 'fragmentation, no single-transformation constraint.'),
    }, open(os.path.join(out, 'manifest.json'), 'w'), indent=2)
    print('wrote %s/candidates.jsonl' % out)
