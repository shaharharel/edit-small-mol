#!/usr/bin/env python3
"""Self-tests for the evidence builder. Run BEFORE generation; non-zero exit blocks it.

Each test exists because the corresponding bug actually reached the data:
  T1  `role` was constant per pair, making the site of change meaningless in 67% of rows
  T2  a 1.78 A heavy-atom "contact" was reported as a polar interaction
  T3  pure additions had no marked attachment point, so reviewers guessed
  T4  "H-bond" was asserted from distance alone, with no angle
  T5  a 14-direction room probe disagreed with 64 by a median of 3.8 A
  T6  atom names must come from the CCD or the table is unanchored to the molecule
  T7  MODEL_INPUT must leak nothing derived from the lead -- at inference there is no lead
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence_v3 as E

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
rows = [json.loads(l) for l in open(D + '/v6_pairs_tc50.jsonl')]
sub = [r for r in rows if r['changed_hit'] > 0][:14]
add = [r for r in rows if r['changed_hit'] == 0][:14]
built_sub = [e for e in (E.build(r) for r in sub) if e]
built_add = [e for e in (E.build(r) for r in add) if e]
fail = []


def t(name, cond, detail=''):
    print('  %-54s %s%s' % (name, 'PASS' if cond else 'FAIL',
                            ('  <- ' + detail) if detail and not cond else ''))
    if not cond:
        fail.append(name)


def soc(e):
    return e.get('LABEL_ONLY', {}).get('site_of_change', [])


print('T1 the site of change is a strict subset of the atoms')
bad = sum(1 for e in built_sub if not soc(e) or len(soc(e)) >= len(e['MODEL_INPUT']['atoms']))
t('substitutions mark some but not all atoms', bad == 0, '%d of %d wrong' % (bad, len(built_sub)))

print('T2 no NON-BONDED heavy-atom contact below 2.4 A')
# Covalent bonds (1.55-2.05 A) and metal coordination (1.8-2.4 A) are SHORTER than any clash
# threshold by physics, so they are exempt -- the original form of this test is what drove the
# filter that deleted 737 covalent contacts. What must not exist is a short contact presented as
# an ordinary polar interaction.
def bonded(a):
    u = a['interaction'].upper()
    return 'COVALENT' in u or 'METAL' in u or 'cofactor' in a['interaction']
nb = [a for e in built_sub + built_add for a in e['MODEL_INPUT']['atoms'] if not bonded(a)]
worst = min((a['distance_A'] for a in nb), default=9.9)
t('minimum non-bonded contact >= 2.4 A', worst >= 2.4, 'min %.2f' % worst)
mis = [a for a in nb if a['distance_A'] < 2.4 and a['interaction'].startswith('H-bond')]
t('no sub-2.4 A contact is called an H-bond', not mis, '%d mislabelled' % len(mis))

print('T3 pure additions carry a marked attachment point')
m = sum(1 for e in built_add if any(a['role'].startswith('ATTACHMENT') for a in soc(e)))
t('pure additions with an attachment point', m >= 0.8 * max(1, len(built_add)),
  '%d/%d' % (m, len(built_add)))

print('T4 H-bond is never claimed without an angle')
bad = sum(1 for e in built_sub + built_add for a in e['MODEL_INPUT']['atoms']
          if a['interaction'].startswith('H-bond') and not a.get('angle_deg'))
t('H-bond claims carry an angle', bad == 0, '%d without' % bad)

print('T5 the probe resolution is sufficient (64 vs 14 directions)')
ref = {}
for r, e in zip(sub[:6], (E.build(x) for x in sub[:6])):
    if e:
        ref[r['pdb']] = {a['atom']: a['pocket_room_A'] for a in soc(e)
                         if a['pocket_room_A'] is not None}
orig = E.PROBE
E.PROBE = E.fib(14)
diffs = []
for r in sub[:6]:
    e = E.build(r)
    if not e:
        continue
    for a in soc(e):
        rv = ref.get(r['pdb'], {}).get(a['atom'])
        if rv is not None and a['pocket_room_A'] is not None:
            diffs.append(abs(rv - a['pocket_room_A']))
E.PROBE = orig
md = float(np.median(diffs)) if diffs else 9.9
t('coarse probe would degrade room by > 0.6 A (so 64 is justified)',
  md > 0.6, 'median diff %.2f - 14 dirs would be fine, 64 is wasted' % md)

print('T6 every reported atom name came from the CCD')
bad = 0
for r, e in zip(sub[:8], (E.build(x) for x in sub[:8])):
    if not e:
        continue
    names = {x[0] for x in E.CCD.get(r['hit_code'], {}).get('atoms', [])}
    if names:
        bad += sum(1 for a in e['MODEL_INPUT']['atoms'] if a['atom'] not in names)
t('atom names present in the CCD', bad == 0, '%d unmapped' % bad)

print('T7 MODEL_INPUT leaks nothing derived from the lead')
LEAK = {'role', 'pocket_room_A', 'pocket_room_lining', 'slot_shape', 'site_of_change'}
bad = 0
for e in built_sub + built_add:
    for a in e['MODEL_INPUT']['atoms']:
        if LEAK & set(a.keys()):
            bad += 1
    # Scan the DATA, not the prose: MODEL_INPUT['note'] legitimately contains the word "lead"
    # in the sentence "no lead information", and scanning it made this test fail on its own
    # disclaimer for three regenerations.
    blob = json.dumps({k: v for k, v in e['MODEL_INPUT'].items() if k != 'note'}).lower()
    if 'analogue' in blob or 'lead' in blob:
        bad += 1
t('no lead-derived field in MODEL_INPUT', bad == 0, '%d leaks' % bad)

print('T8 metal coordination is only ever claimed for an element that can coordinate')
# 108 rows reported a ligand CARBON at 2.5-2.7 A from a manganese as "METAL COORDINATION". A carbon
# bonded to a coordinating oxygen sits that far from the metal by geometry; it coordinates nothing.
badm = [(e['pair'], a['atom'], a['element'], a['faces'], a['distance_A'])
        for e in built_sub + built_add for a in e['MODEL_INPUT']['atoms']
        if 'METAL COORDINATION' in a['interaction'] and a['element'].upper() not in E.COORDINATING]
t('no impossible metal coordination', not badm, '%d rows, e.g. %s' % (len(badm), badm[:2]))

print('T9 conserved atoms are not marked as the site of change')
# A single connected MCS marked a 4-chlorophenyl present in BOTH molecules as changed, while the
# ring that actually changed went unmarked. Verify against an independent substructure test: a
# group present in hit AND lead must not be reported as the edit.
from rdkit import Chem as _C
bad9 = []
for e in built_sub + built_add:
    h = _C.MolFromSmiles(e['MODEL_INPUT']['hit_smiles'] or '')
    l = _C.MolFromSmiles(e['ANSWER']['analogue_smiles'] or '')
    if not h or not l:
        continue
    ka, _kb, _am = E.conserved_multi(h, l)
    n_changed = h.GetNumHeavyAtoms() - len(ka)
    n_marked = sum(1 for a in soc(e) if a['role'] == 'CHANGED')
    # marking more atoms CHANGED than actually differ means conserved atoms were included
    if n_marked > n_changed + 1:
        bad9.append((e['pair'], n_marked, n_changed))
t('no conserved atom reported as CHANGED', not bad9,
  '%d pairs over-marked, e.g. %s' % (len(bad9), bad9[:2]))

print('\n%d/9 tests passed' % (9 - len(fail)))
if fail:
    print('FAILED: %s' % '; '.join(fail))
sys.exit(1 if fail else 0)