#!/usr/bin/env python3
"""Final label-ready cohort, plus a stratified audit sample rendered as a labeller sees it.

The cohort gates are all "can a labeller write a POCKET-GROUNDED sentence about this pair", which
is the point of the corpus -- not "is the row well formed". A pair where the edit sits in solvent,
or where hit and lead residue numbering disagree so the before/after contacts cannot be compared,
is a pair whose label would fall back to SMILES diffing.

The sample is stratified across distinct targets and forces in a covalent and a metal case, because
the previous audit sample was three HIV-1 protease pairs out of ten and could not certify anything.
"""
import json, collections, random, os

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SP = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(D + '/v6_labelset.jsonl')]

why = collections.Counter()
ok = []
for r in rows:
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    ats = {a['atom']: a for a in mi['atoms']}
    soc = lo['site_of_change']
    sa = [ats[s['atom']] for s in soc if s['atom'] in ats]
    f = []
    if not soc:
        f.append('no site_of_change')
    if soc and not sa:
        f.append('site atoms not in the atom table')
    if sa and not any(a['distance_A'] <= 4.5 for a in sa):
        f.append('edit sits in solvent (no residue within 4.5 A)')
    if not lo.get('numbering_consistent'):
        f.append('numbering inconsistent: lead contacts not citable')
    if not (lo.get('lead_contacts') or []):
        f.append('no lead_contacts')
    if sa and len(sa) / max(len(ats), 1) > 0.45:
        f.append('site covers >45% of the molecule')
    if lo.get('room_contradicts_edit'):
        f.append('room probe contradicts the edit')
    if lo.get('heavy_atoms_added') is None:
        f.append('edit size unknown (SMILES parse)')
    for x in f:
        why[x] += 1
    if not f:
        ok.append(r)

with open(D + '/v6_labelready.jsonl', 'w') as f:
    for r in ok:
        f.write(json.dumps(r) + '\n')

n = len(rows)
print('unique chemical edits             %6d' % n)
print('LABEL-READY cohort                %6d   %.0f%%' % (len(ok), 100 * len(ok) / n))
print('\nheld back (a row may fail several gates):')
for k, v in why.most_common():
    print('   %-50s %5d  %4.1f%%' % (k, v, 100 * v / n))

tg = collections.Counter(r['target'] for r in ok)
print('\ncohort spread: %d targets; largest series %d rows (%.1f%%)'
      % (len(tg), tg.most_common(1)[0][1], 100 * tg.most_common(1)[0][1] / max(len(ok), 1)))


def render(r):
    """The evidence exactly as a labeller reads it: hit + pocket, then the label-side material."""
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    L = ['=' * 78, 'PAIR %s   target %s   tanimoto %.3f' % (r['pair'], r['target'], r['tanimoto']),
         '=' * 78, '',
         '--- MODEL_INPUT (all the model gets at inference: hit + its own pocket) ---',
         'hit SMILES: %s' % mi['hit_smiles'], '',
         '%-6s %-3s %-16s %6s %-46s %5s %s' % ('atom', 'el', 'faces', 'dist', 'interaction',
                                               'open', 'reading')]
    for a in mi['atoms']:
        L.append('%-6s %-3s %-16s %6.2f %-46s %5.2f %s'
                 % (a['atom'], a['element'], a['faces'], a['distance_A'],
                    a['interaction'][:46], a['open_fraction'], a['reading']))
    L += ['', '--- LABEL_ONLY (for writing the label; never shown to the model) ---',
          'lead PDB: %s   numbering_consistent: %s' % (lo['lead_pdb'], lo['numbering_consistent']),
          'EDIT: %s' % lo['edit_size_note'],
          'lead SMILES: %s' % r['ANSWER']['analogue_smiles'], '',
          'site_of_change (which atom-table rows the edit touches):']
    for s in lo['site_of_change']:
        L.append('   %-6s %-42s room %s  %s  lining %s'
                 % (s['atom'], s['role'], s.get('pocket_room_A'), s.get('room_reading') or '-',
                    ','.join(s.get('pocket_room_lining') or []) or '-'))
    L += ['', 'lead-side contacts in the lead crystal (%d):' % len(lo.get('lead_contacts') or [])]
    for c in (lo.get('lead_contacts') or [])[:40]:
        L.append('   %-6s %-3s %-16s %.2f' % (c['atom'], c['el'], c['faces'], c['distance_A']))
    return '\n'.join(L)


random.seed(11)
bytgt = collections.defaultdict(list)
for r in ok:
    bytgt[r['target']].append(r)


def has(r, s):
    return any(s in a['interaction'].upper() for a in r['MODEL_INPUT']['atoms'])


cov = [r for r in ok if has(r, 'COVALENT')]
met = [r for r in ok if has(r, 'METAL')]
print('cohort contains %d covalent-bond rows and %d metal-coordination rows' % (len(cov), len(met)))

pick, used = [], set()
for pool, tag in ((cov, 'covalent'), (met, 'metal')):
    random.shuffle(pool)
    for r in pool:
        if r['target'] not in used:
            pick.append((tag, r))
            used.add(r['target'])
            break
tgs = [t for t in bytgt if t not in used]
random.shuffle(tgs)
for t in tgs:
    if len(pick) >= 10:
        break
    pick.append(('general', random.choice(bytgt[t])))
    used.add(t)

with open(SP + '/audit10.jsonl', 'w') as f:
    for t, r in pick:
        f.write(json.dumps(dict(r, _tag=t)) + '\n')
with open(SP + '/audit10.txt', 'w') as f:
    f.write('STRATIFIED PRE-LABELLING AUDIT - 10 pairs, 10 distinct targets\n\n')
    for t, r in pick:
        f.write('[%s]\n%s\n\n' % (t, render(r)))

print('\naudit sample (10 distinct targets) -> audit10.txt (%d KB)'
      % (os.path.getsize(SP + '/audit10.txt') / 1024))
for t, r in pick:
    print('   %-9s %-20s %s' % (t, r['pair'], r['target']))
