#!/usr/bin/env python3
"""Render a stratified audit sample exactly as a labeller reads it.

Stratified across distinct targets and forced to contain a covalent and a metal case: an earlier
audit sample was three HIV-1 protease pairs out of ten and could not certify anything.
"""
import json, collections, random, os, sys

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SRC = sys.argv[1] if len(sys.argv) > 1 else 'v6_labelcohort.jsonl'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 11
OUT = sys.argv[4] if len(sys.argv) > 4 else 'audit'


def render(r):
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    L = ['=' * 86,
         'PAIR %s   target %s   tanimoto %.3f   edit_type %s'
         % (r['pair'], r['target'], r['tanimoto'], r['edit_type']),
         '=' * 86, '',
         '--- MODEL_INPUT (all the model gets at inference: the hit and its own pocket) ---',
         'hit SMILES: %s' % mi['hit_smiles'], '',
         '%-6s %-3s %-16s %6s %6s %-52s %5s %s'
         % ('atom', 'el', 'faces', 'dist', 'angle', 'interaction', 'open', 'reading')]
    for a in mi['atoms']:
        # angle_deg MUST be rendered: hard rule 2 turns on whether a polar contact qualifies as an
        # H-bond, and a labeller cannot apply it from the interaction string alone.
        ang = '%6.0f' % a['angle_deg'] if a.get('angle_deg') else '     -'
        L.append('%-6s %-3s %-16s %6.2f %s %-52s %5.2f %s'
                 % (a['atom'], a['element'], a['faces'], a['distance_A'], ang,
                    a['interaction'][:52], a['open_fraction'], a['reading']))
    L += ['', '--- LABEL_ONLY (for writing the label; the model never sees this) ---',
          'lead PDB: %s   numbering_consistent: %s   numbering_confidence: %s'
          % (lo['lead_pdb'], lo['numbering_consistent'], lo.get('numbering_confidence')),
          'EDIT: %s' % lo['edit_size_note'],
          'room_contradicts_edit: %s' % lo.get('room_contradicts_edit'),
          'lead SMILES: %s' % r['ANSWER']['analogue_smiles']]
    if r.get('n_other_crystals'):
        L.append('this edit also appears in %d other crystal(s)' % r['n_other_crystals'])
    if lo.get('lead_contact_warning'):
        L.append('!! %s' % lo['lead_contact_warning'])
    L += ['', 'site_of_change (which rows of the atom table the edit touches):']
    for s in lo['site_of_change']:
        L.append('   %-6s %-62s room %-5s %-28s lining %s'
                 % (s['atom'], s['role'], s.get('pocket_room_A'),
                    s.get('room_reading') or '-',
                    ','.join(s.get('pocket_room_lining') or []) or '-'))
    lc = lo.get('lead_contacts') or []
    L += ['', "lead-side contacts in the LEAD's own crystal (%d):" % len(lc)]
    for c in lc[:40]:
        flag = '   <-- UNCERTAIN, do not cite' if c.get('uncertain') else ''
        L.append('   %-6s %-3s %-16s %6.2f%s' % (c['atom'], c['el'], c['faces'], c['distance_A'], flag))
    return '\n'.join(L)


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(os.path.join(D, SRC))]
    random.seed(SEED)

    def has(r, s):
        return any(s in a['interaction'].upper() for a in r['MODEL_INPUT']['atoms'])

    bytgt = collections.defaultdict(list)
    for r in rows:
        bytgt[r['target']].append(r)
    pick, used = [], set()
    for pool, tag in (([r for r in rows if has(r, 'COVALENT')], 'covalent'),
                      ([r for r in rows if has(r, 'METAL COORDINATION')], 'metal'),
                      ([r for r in rows if r['LABEL_ONLY'].get('lead_contact_warning')], 'uncertain-flag')):
        random.shuffle(pool)
        for r in pool:
            if r['target'] not in used:
                pick.append((tag, r)); used.add(r['target']); break
    tgs = [t for t in bytgt if t not in used]
    random.shuffle(tgs)
    for t in tgs:
        if len(pick) >= N: break
        pick.append(('general', random.choice(bytgt[t]))); used.add(t)

    with open(OUT + '.jsonl', 'w') as f:
        for t, r in pick:
            f.write(json.dumps(dict(r, _tag=t)) + '\n')
    with open(OUT + '.txt', 'w') as f:
        f.write('STRATIFIED PRE-LABELLING AUDIT - %d pairs, %d distinct targets\n\n'
                % (len(pick), len({r['target'] for _, r in pick})))
        for t, r in pick:
            f.write('[%s]\n%s\n\n' % (t, render(r)))
    print('%d pairs, %d distinct targets -> %s.txt (%d KB)'
          % (len(pick), len({r['target'] for _, r in pick}), OUT,
             os.path.getsize(OUT + '.txt') / 1024))
    for t, r in pick:
        print('   %-14s %-26s %s' % (t, r['pair'], r['target']))
