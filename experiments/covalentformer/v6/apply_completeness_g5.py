#!/usr/bin/env python3
"""Declare, for each pair, how much of each molecule actually has a row.

Separate from the generator because it needs the analogue SMILES heavy-atom count, and inline in a
shell heredoc because that is where it started -- which is how it got lost on two rebuilds. It is a
script now so a rebuild cannot silently skip it.

g5 variant: identical logic, paths repointed (the original stays reproducible for g1-g4).
Run AFTER clean_g5.py.

The INCOMPLETE warning is suppressed when the only unlisted atom is a HALOGEN on a pair that has a
covalent bond: that is the leaving group of the reaction, chemically required to be absent once the
adduct is modelled, and warning about it suppressed legitimate loss claims on 8ARW -- an otherwise
complete 29-of-30 structure.
"""
import json, os, sys
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/g5_evidence.jsonl'
MIN_ROWS = 1200  # g5 is a 1,249-pair build; 5000 was the 8,359-pair floor

rows = [json.loads(l) for l in open(EV)]
if len(rows) < MIN_ROWS:
    sys.exit('REFUSING: corpus holds %d pairs (floor %d)' % (len(rows), MIN_ROWS))

inc = suppressed = 0
for r in rows:
    lo, mi = r['LABEL_ONLY'], r['MODEL_INPUT']
    cov = bool(lo.pop('_covalent_present', False))
    m = Chem.MolFromSmiles((r.get('ANSWER') or {}).get('analogue_smiles') or '')
    ac = lo.get('analogue_contacts') or []
    if m is not None:
        heavy = m.GetNumHeavyAtoms()
        gap = heavy - len(ac)
        lo['analogue_contacts_complete'] = bool(gap <= 0)
        lo['analogue_atoms_listed'] = ('%d of %d analogue heavy atoms appear in the table'
                                       % (len(ac), heavy))
        listed = {c.get('atom') for c in ac}
        hal = [a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() in ('F', 'CL', 'BR', 'I', 'Cl', 'Br')]
        leaving = cov and gap == 1 and hal
        if gap > 0 and not leaving:
            inc += 1
            lo['analogue_contact_warning'] = (
                'INCOMPLETE: only %d of this analogue\'s %d heavy atoms appear in the table, so an '
                'ABSENCE IS NOT EVIDENCE here. Do NOT write that the analogue loses, drops or no '
                'longer makes a contact; the atom may simply not be modelled. Gains that ARE '
                'listed, and the input side, remain usable.' % (len(ac), heavy))
        lo['analogue_polar_caveat'] = (
            'EACH ANALOGUE ROW SHOWS ONLY THE NEAREST PROTEIN ATOM, and this table has no '
            'polar_partner column. So a polar atom whose nearest neighbour is a carbon has its '
            'hydrogen-bond partner in NO field here. For an apolar contact an absence in a complete '
            'table is evidence; for a POLAR one it is not -- you cannot conclude from this table '
            'that a hydrogen bond was lost, only that it is not the nearest contact.')
        if leaving:
            suppressed += 1
            lo['analogue_atoms_listed'] += (
                ' -- the one unlisted atom is a halogen on a covalent pair, i.e. the leaving group, '
                'so the table is effectively complete for APOLAR contacts (see '
                'analogue_polar_caveat for why polar absences still prove nothing).')
    mim = Chem.MolFromSmiles(mi['input_smiles'])
    if mim is not None:
        ih, il = mim.GetNumHeavyAtoms(), len(mi['atoms'])
        nmod = len(mi.get('atoms_not_modelled') or [])
        ncl = len(mi.get('atoms_dropped_as_clashes') or [])
        mi['input_atoms_listed'] = ('%d of %d input heavy atoms have a row (%d not modelled, %d '
                                    'omitted as clashes)' % (il, ih, nmod, ncl))
        if il + nmod + ncl < ih:
            mi['input_atom_warning'] = (
                'INCOMPLETE INPUT TABLE: %d heavy atoms are unaccounted for. An absence is not '
                'evidence on the input side either.' % (ih - il - nmod - ncl))

tmp = EV + '.tmp'
with open(tmp, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
assert sum(1 for _ in open(tmp)) == len(rows)
os.replace(tmp, EV)
print('%d pairs: %d analogue tables INCOMPLETE, %d suppressed as a covalent leaving group'
      % (len(rows), inc, suppressed))
