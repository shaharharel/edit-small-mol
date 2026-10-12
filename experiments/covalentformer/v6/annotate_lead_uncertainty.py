#!/usr/bin/env python3
"""Mirror the hit side's short-contact disclosure onto the lead side.

The hit's atom table already annotates an implausibly short contact in place
("unusually short polar contact (2.10 A) -- treat the crystal model as uncertain") rather than
deleting it, because a flat distance filter is what silently removed 737 real covalent contacts.
`lead_contacts` never got the same treatment, so 157 contacts across 130 pairs are presented to a
labeller as plain measurement at distances below any chemical bond -- 0.52 A in one case, where the
ligand and SER509 are modelled in the same place (altloc / partial occupancy).

A labeller quoting those would write "the lead's C23 reaches SER509.CB at 0.52 A", which is not a
design rationale, it is a refinement artifact. So: annotate, never drop. Nothing is removed; the
pair stays labellable and the reviewer is told which numbers not to trust.

Thresholds, by element pair rather than one flat cut:
  < 1.30 A   shorter than any heavy-atom bond            -> uncertain
  < 0.80 x the ideal bond length for that element pair   -> uncertain
Write this into evidence_v3.build() before the next full regeneration; this pass exists so the
cohort can be labelled without waiting on one.
"""
import json, os, sys

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SRC = sys.argv[1] if len(sys.argv) > 1 else 'v6_labelready.jsonl'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'v6_labelready_annot.jsonl'

HARD_FLOOR = 1.30
IDEAL = {frozenset('CC'): 1.54, frozenset('CN'): 1.47, frozenset('CO'): 1.43,
         frozenset('CS'): 1.81, frozenset('NO'): 1.40, frozenset('NN'): 1.45,
         frozenset('OO'): 1.48, frozenset('OS'): 1.57, frozenset('NS'): 1.70,
         frozenset('SS'): 2.05}
NOTE = 'distance below a physical bond (%.2f A) -- refinement artifact, do not cite this number'


def face_element(faces):
    """'SER509.OG' -> 'O'; 'ZN301.ZN' -> 'ZN'. The atom name's leading element symbol."""
    atom = faces.split('.', 1)[1] if '.' in faces else faces
    return atom[:2] if atom[:2] in ('ZN', 'MG', 'MN', 'FE', 'CU', 'NA', 'CL', 'BR', 'SE') else atom[:1]


def suspect(c):
    d = c['distance_A']
    if d < HARD_FLOOR:
        return True
    a, b = (c.get('el') or '?').upper(), face_element(c['faces']).upper()
    if len(a) == 1 and len(b) == 1:
        ideal = IDEAL.get(frozenset(a + b))
        if ideal and d < 0.80 * ideal:
            return True
    return False


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(os.path.join(D, SRC))]
    npair = ncontact = 0
    with open(os.path.join(D, OUT), 'w') as f:
        for e in rows:
            hit = 0
            for c in (e['LABEL_ONLY'].get('lead_contacts') or []):
                if suspect(c):
                    c['uncertain'] = NOTE % c['distance_A']
                    hit += 1
            if hit:
                npair += 1
                ncontact += hit
                e['LABEL_ONLY']['lead_contact_warning'] = (
                    '%d of these contacts are flagged `uncertain` -- the crystal models the ligand '
                    'and the residue too close together. Base the label on the others.' % hit)
            f.write(json.dumps(e) + '\n')
    print('pairs in                     %6d' % len(rows))
    print('pairs with a flagged contact %6d  (%.1f%%)' % (npair, 100.0 * npair / len(rows)))
    print('contacts flagged             %6d' % ncontact)
    print('pairs dropped                     0   (annotation only)')
    print('-> %s' % os.path.join(D, OUT))
