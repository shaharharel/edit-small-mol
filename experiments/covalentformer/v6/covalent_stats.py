#!/usr/bin/env python3
"""How covalent is g3, and is the warhead preserved across the edit?

Two independent readings, because neither alone is trustworthy:

  GEOMETRIC -- a ligand heavy atom within 2.1 A of a protein N/O/S is a bond, not a contact. This
  catches covalency regardless of chemotype, but also catches metal coordination, so ZN/MG/FE/MN
  partners are excluded and counted separately.

  CHEMOTYPE -- SMARTS for the electrophiles actually used in covalent medicinal chemistry. This
  catches warheads that are present but not engaged in the deposited model (unreacted, or modelled
  in the apo form), which geometry misses.

"Retained" is then the question the generator has to get right: of the pairs whose INPUT carries a
warhead, in how many does the ANALOGUE still carry one? A method that silently destroys the
warhead while optimising the rest of the molecule is useless for covalent design.
"""
import collections
import json

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
GEN = 'g3_v2_input_anchored_envelope'
METALS = {'ZN', 'MG', 'FE', 'MN', 'CU', 'NI', 'CO', 'CA', 'NA', 'K', 'CD', 'HG'}

WARHEADS = [
    ('chloroacetamide',      '[NX3][CX3](=O)[CH2][Cl,Br,I]'),
    ('acrylamide',           '[NX3][CX3](=O)[CH]=[CH2]'),
    ('cyanoacrylamide',      '[NX3][CX3](=O)C(=C)C#N'),
    ('vinyl sulfone',        '[SX4](=O)(=O)[CH]=[CH2]'),
    ('sulfonyl fluoride',    '[SX4](=O)(=O)F'),
    ('fluorosulfate',        '[OX2][SX4](=O)(=O)F'),
    ('nitrile (Cys/Ser)',    '[#6][CX2]#[NX1]'),
    ('aldehyde',             '[CX3H1](=O)[#6]'),
    ('boronic acid/ester',   '[BX3]([OX2])[OX2]'),
    ('epoxide',              'C1OC1'),
    ('aziridine',            'C1NC1'),
    ('alpha-ketoamide',      '[NX3][CX3](=O)[CX3](=O)'),
    ('beta-lactam',          'C1C(=O)NC1'),
    ('maleimide',            'O=C1C=CC(=O)N1'),
    ('propargyl amide',      '[NX3][CX3](=O)C#C'),
    ('disulfide',            '[SX2][SX2]'),
    ('isocyanate/thio',      '[NX2]=[CX2]=[OS]'),
]
PATTS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEADS]
for n, p in PATTS:
    assert p is not None, 'bad SMARTS for %s' % n


def warheads_in(smiles):
    m = Chem.MolFromSmiles(smiles or '')
    if m is None:
        return None
    return {n for n, p in PATTS if m.HasSubstructMatch(p)}


def geometric(rec):
    """(covalent_partners, metal_partners) from the input-side table."""
    cov, met = set(), set()
    for a in rec['MODEL_INPUT'].get('atoms') or []:
        d, faces = a.get('distance_A'), a.get('faces') or ''
        inter = (a.get('interaction') or '').upper()
        if d is None:
            continue
        res = faces.split('.')[0] if '.' in faces else faces
        if 'METAL' in inter or res[:2] in METALS or res[:3] in METALS:
            if d <= 2.6:
                met.add(faces)
            continue
        if d <= 2.1:
            cov.add(faces)
    return cov, met


def main():
    n = 0
    geo_cov = metal = 0
    wh_input = wh_analogue = wh_retained = wh_lost = wh_gained = 0
    wh_names = collections.Counter()
    retained_by_wh = collections.Counter()
    lost_by_wh = collections.Counter()
    partners = collections.Counter()
    both_signals = 0
    for line in open(EV):
        rec = json.loads(line)
        # the generation stamp lives on the LABEL, not the evidence, so re-derive from batch range
        n += 1
        cov, met = geometric(rec)
        if cov:
            geo_cov += 1
            for f in cov:
                partners[f.split('.')[0]] += 1
        if met:
            metal += 1
        wi = warheads_in(rec['MODEL_INPUT'].get('input_smiles'))
        wa = warheads_in((rec.get('ANSWER') or {}).get('analogue_smiles'))
        if wi is None or wa is None:
            continue
        if wi:
            wh_input += 1
            for x in wi:
                wh_names[x] += 1
            if wa & wi:
                wh_retained += 1
                for x in (wa & wi):
                    retained_by_wh[x] += 1
            else:
                wh_lost += 1
                for x in wi:
                    lost_by_wh[x] += 1
            if cov:
                both_signals += 1
        if wa:
            wh_analogue += 1
        if wa and not wi:
            wh_gained += 1
    print('records scanned: %s\n' % format(n, ','))
    print('GEOMETRIC (input heavy atom <= 2.1 A of protein N/O/S, metals excluded)')
    print('  covalently bound input ligands : %s (%.1f%%)' % (format(geo_cov, ','), 100 * geo_cov / n))
    print('  metal-coordinating             : %s (%.1f%%)' % (format(metal, ','), 100 * metal / n))
    print('  top covalent partners          : %s' % ', '.join('%s %d' % kv for kv in partners.most_common(6)))
    print('\nCHEMOTYPE (warhead SMARTS)')
    print('  input carries a warhead        : %s (%.1f%%)' % (format(wh_input, ','), 100 * wh_input / n))
    print('  analogue carries a warhead     : %s (%.1f%%)' % (format(wh_analogue, ','), 100 * wh_analogue / n))
    if wh_input:
        print('  WARHEAD RETAINED across edit   : %s / %s (%.1f%% of warhead-bearing inputs)'
              % (format(wh_retained, ','), format(wh_input, ','), 100 * wh_retained / wh_input))
        print('  warhead LOST across edit       : %s (%.1f%%)' % (format(wh_lost, ','), 100 * wh_lost / wh_input))
    print('  warhead GAINED (none -> one)   : %s' % format(wh_gained, ','))
    print('  both geometric and chemotype   : %s' % format(both_signals, ','))
    print('\n  warheads present in inputs, and how often retained:')
    for name, c in wh_names.most_common():
        print('    %-22s n=%-5d retained %-5d (%.0f%%)'
              % (name, c, retained_by_wh[name], 100 * retained_by_wh[name] / c))


if __name__ == '__main__':
    main()
