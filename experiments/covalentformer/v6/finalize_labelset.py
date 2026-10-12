#!/usr/bin/env python3
"""Apply the MODEL_INPUT whitelist to the corpus on disk and add what the labeller still lacks.

Three findings from the pre-labelling audit, fixed here:

1. LEAK. `room_reading` and `_exit` are computed only for site-of-change atoms, so they were
   non-null exactly on the edited atoms -- `_exit` identified the edit site in 93% of rows. They
   are deleted from MODEL_INPUT. This is a pure field deletion, identical to what the whitelist in
   evidence_v3.py now produces, so no geometry is recomputed and nothing else changes.

2. EDIT SIZE WAS NEVER STATED. 84% of the cohort adds atoms, but the evidence only said
   "pure addition" / "substitution/removal" -- categorical, no magnitude. A labeller cannot reason
   about whether a group fits a pocket without knowing how big the group is. `heavy_atoms_added`
   goes in LABEL_ONLY (it is derived from the lead, so it must never reach MODEL_INPUT) next to a
   plain-language statement of the edit.

3. ROOM CONTRADICTS THE EDIT. In 431 growth pairs the marked site reports 0.0 A of room while the
   lead adds atoms there. The room probe measures enclosure around an atom that is already bonded
   into the molecule, which is not the same question as "can a substituent grow from here" -- so a
   labeller told "blocked by protein" would write the opposite of what the chemists did. These rows
   are flagged and held out of the label-ready cohort rather than silently labelled.
"""
import json, collections
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
MODEL_FIELDS = ('atom', 'element', 'faces', 'distance_A', 'interaction', 'angle_deg',
                'open_fraction', 'reading')

rows = [json.loads(l) for l in open(D + '/v6_labelset.jsonl')]
stat = collections.Counter()
ready, held = [], []

for r in rows:
    # ---- 1. whitelist MODEL_INPUT
    before = set(r['MODEL_INPUT']['atoms'][0].keys()) if r['MODEL_INPUT']['atoms'] else set()
    r['MODEL_INPUT']['atoms'] = [{k: a[k] for k in MODEL_FIELDS if k in a}
                                 for a in r['MODEL_INPUT']['atoms']]
    stat['fields dropped from MODEL_INPUT'] += len(before - set(MODEL_FIELDS))

    # ---- 2. state the edit size on the label side
    h = Chem.MolFromSmiles(r['MODEL_INPUT']['hit_smiles'] or '')
    l = Chem.MolFromSmiles(r['ANSWER']['analogue_smiles'] or '')
    lo = r['LABEL_ONLY']
    if h and l:
        d = l.GetNumHeavyAtoms() - h.GetNumHeavyAtoms()
        lo['heavy_atoms_added'] = d
        lo['edit_size_note'] = (
            'the lead adds %d heavy atoms to the hit (%d -> %d)' % (d, h.GetNumHeavyAtoms(),
                                                                    l.GetNumHeavyAtoms())
            if d > 0 else
            'the lead removes %d heavy atoms from the hit (%d -> %d)' % (-d, h.GetNumHeavyAtoms(),
                                                                         l.GetNumHeavyAtoms())
            if d < 0 else
            'the lead is the same heavy-atom count as the hit (%d): a substitution, not a change '
            'in size' % h.GetNumHeavyAtoms())
    else:
        lo['heavy_atoms_added'] = None
        lo['edit_size_note'] = 'one of the two SMILES does not parse; edit size unknown'

    # ---- 3. flag room that contradicts the edit
    grew = (lo.get('heavy_atoms_added') or 0) > 0
    rm = [s.get('pocket_room_A') for s in lo['site_of_change'] if s.get('pocket_room_A') is not None]
    contradiction = grew and rm and max(rm) == 0.0
    lo['room_contradicts_edit'] = bool(contradiction)
    if contradiction:
        lo['room_caveat'] = (
            'The room probe reports 0.0 A at every marked atom, yet the lead adds %d heavy atoms '
            'here. The probe measures enclosure around an atom that is already bonded into the '
            'molecule, which is not the same question as whether a substituent can grow from it. '
            'Do NOT write that there is no room at this site.' % lo['heavy_atoms_added'])
        stat['held out: room contradicts the edit'] += 1
        held.append(r)
    else:
        ready.append(r)

with open(D + '/v6_labelset.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
with open(D + '/v6_labelready.jsonl', 'w') as f:
    for r in ready:
        f.write(json.dumps(r) + '\n')
with open(D + '/v6_held_room_contradiction.jsonl', 'w') as f:
    for r in held:
        f.write(json.dumps(r) + '\n')

print('rows processed                 %6d' % len(rows))
for k, v in stat.most_common():
    print('%-34s %6d' % (k, v))
print()
print('MODEL_INPUT fields now: %s' % ', '.join(MODEL_FIELDS))
ex = rows[0]['MODEL_INPUT']['atoms'][0]
print('example atom row: %s' % json.dumps(ex))
print()
print('v6_labelready.jsonl            %6d   <- cohort to label' % len(ready))
print('v6_held_room_contradiction     %6d   <- needs a growth-vector probe, not a label' % len(held))

# verify the leak is actually gone, on the real corpus rather than a sample
import numpy as np
agg = collections.defaultdict(lambda: [[], []])
for r in rows:
    sn = {s['atom'] for s in r['LABEL_ONLY']['site_of_change']}
    for a in r['MODEL_INPUT']['atoms']:
        for k, v in a.items():
            if k in ('atom', 'faces'):
                continue
            x = (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                 else (0.0 if v is None else 1.0))
            agg[k][0 if a['atom'] in sn else 1].append(x)
print('\nleak check on the full corpus (z = site mean vs unchanged spread):')
worst = 0.0
for k, (s, o) in sorted(agg.items()):
    if len(s) < 20 or len(o) < 20:
        continue
    z = abs(np.mean(s) - np.mean(o)) / (np.std(o) or 1e-9)
    worst = max(worst, z)
    print('   %-16s site %8.3f  other %8.3f  z=%5.2f  %s'
          % (k, np.mean(s), np.mean(o), z, 'LEAK' if z > 1.0 else 'ok'))
print('\nworst z across all fields: %.2f  (%s)' % (worst, 'CLEAN' if worst <= 1.0 else 'STILL LEAKING'))
