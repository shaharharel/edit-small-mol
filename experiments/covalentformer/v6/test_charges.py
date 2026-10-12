#!/usr/bin/env python3
"""Unit tests for the two chemistry models that decide what an interaction is called.

Part 1 -- the ligand charge model, which decides every SALT BRIDGE row.
Part 2 -- which ligand sulfurs may accept a hydrogen bond.

It exists because this component was wrong twice in one hour. First a thioamide nitrogen was typed
+1 (the pattern excluded N-C=O but not N-C=S), which produced an ion pair that cannot exist AND
suppressed the row's correct `H-bond, good geometry` typing, because the salt-bridge branch returns
before the hydrogen-bond branch. Then my fix for that excluded amidine and guanidine as well --
which would have deleted the guanidinium salt bridges of every arginine mimetic in the corpus, the
ones that matter most. A charge model is exactly the kind of component where a plausible-looking
SMARTS edit does quiet damage, so it gets a table of cases instead of a spot check.

Run: python -I test_charges.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from evidence_v4 import ligand_charges, sulfur_hb_ok

# (smiles, label, expected overall sign)
CASES = [
    # --- must be CATIONIC: these are protonated at physiological pH and really do form ion pairs
    ('CCN',                  'primary amine',        '+1'),
    ('CNC',                  'secondary amine',      '+1'),
    ('C[N+](C)(C)C',         'quaternary ammonium',  '+1'),
    ('CNC(=N)N',             'guanidine',            '+1'),
    ('CC(=N)N',              'amidine',              '+1'),
    ('C1CCNCC1',             'piperidine',           '+1'),

    # --- the ANILINE class: an amine conjugated into an aromatic ring is NOT protonated at pH 7.4
    #     (aniline pKa ~4.6). These are extremely common -- aminopyrimidine and aminopurine hinge
    #     binders -- and every one of them was being reported as a salt bridge partner.
    ('Nc1ccccc1',            'aniline',              'neutral'),
    ('Nc1ncnc2[nH]cnc12',    'adenine exocyclic NH2','neutral'),
    ('Nc1nccnc1',            '2-aminopyrimidine',    'neutral'),
    ('Nc1ccncc1',            '4-aminopyridine NH2',  'neutral'),
    ('NCc1ccccc1',           'benzylamine',          '+1'),
    ('C1CCNCC1',             'piperidine (again)',   '+1'),

    # --- must be ANIONIC
    ('CC(=O)O',              'carboxylate',          '-1'),
    ('CS(=O)(=O)O',          'sulfonate',            '-1'),
    ('CP(=O)(O)O',           'phosphonate',          '-1'),

    # --- must be NEUTRAL: not basic, not acidic, and not an ion pair partner
    ('CC(=O)N',              'amide',                'neutral'),
    ('CC(=S)N',              'thioamide',            'neutral'),
    ('CS(=O)(=O)N',          'sulfonamide',          'neutral'),
    ('c1cc[nH]c1',           'pyrrole',              'neutral'),
    ('c1ccncc1',             'pyridine',             'neutral'),
    ('c1ccc2[nH]ccc2c1',     'indole',               'neutral'),
    ('CO',                   'alcohol',              'neutral'),

    # --- charge-separated but NET NEUTRAL: the formal charges in the SMILES are real but the group
    #     is neither acidic nor basic, so reading them literally invents salt bridges
    ('C[N+](=O)[O-]',        'alkyl nitro',          'neutral'),
    ('c1ccccc1[N+](=O)[O-]', 'aryl nitro',           'neutral'),
]


def sign(q):
    q = {k: v for k, v in q.items() if v}
    if any(v > 0 for v in q.values()):
        return '+1'
    return '-1' if q else 'neutral'


# Part 2. A ligand sulfur is an H-bond partner only when it has a lone pair to offer: a thiol,
# thioether, thione or aromatic sulfur. A fully substituted S(VI) -- sulfone, sulfonamide,
# sulfonate -- cannot accept, yet 178 rows typed `H-bond, good geometry` on one, and 8IW6's was the
# geometric shadow of an oxygen placement the pipeline already distrusted.
# A SULFOXIDE is deliberately refused too: it has a lone pair, but in structural practice the
# hydrogen bond goes to its oxygen, not its sulfur. Conservative is right here -- this feeds labels.
S_CASES = [
    ('CS(=O)(=O)N', 'sulfonamide S', False),
    ('CS(=O)(=O)O', 'sulfonate S',   False),
    ('CS(=O)C',     'sulfoxide S',   False),
    ('CSC',         'thioether S',   True),
    ('CS',          'thiol S',       True),
    ('CC(=S)N',     'thione S',      True),
    ('c1ccsc1',     'thiophene S',   True),
]


def main():
    bad = []
    print('%-22s %-24s %-10s %-10s' % ('group', 'smiles', 'got', 'expected'))
    for smi, label, want in CASES:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad.append((label, 'unparseable', want))
            print('%-22s %-24s UNPARSEABLE' % (label, smi))
            continue
        got = sign(ligand_charges(m))
        ok = got == want
        if not ok:
            bad.append((label, got, want))
        print('%-22s %-24s %-10s %-10s %s' % (label, smi, got, want, '' if ok else '*** WRONG ***'))
    print()
    print('%-22s %-24s %-10s %-10s' % ('sulfur', 'smiles', 'accepts?', 'expected'))
    for smi, label, want in S_CASES:
        m = Chem.MolFromSmiles(smi)
        i = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S'][0]
        got = sulfur_hb_ok(m, i)
        ok = got == want
        if not ok:
            bad.append((label, got, want))
        print('%-22s %-24s %-10s %-10s %s'
              % (label, smi, got, want, '' if ok else '*** WRONG ***'))
    print()
    print('%d of %d correct' % (len(CASES) + len(S_CASES) - len(bad), len(CASES) + len(S_CASES)))
    if bad:
        for label, got, want in bad:
            print('  FAIL %-22s got %s, expected %s' % (label, got, want))
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
