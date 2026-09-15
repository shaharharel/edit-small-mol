"""IS THIS MOLECULE A PLAUSIBLE TARGETED COVALENT INHIBITOR? One place, version-stamped.

WHY THIS EXISTS. build_covalent_union.py used '[CX3]=[CX3][CX3](=O)[NX3,OX2]' for
"acrylamide/Michael". That pattern does not distinguish a TERMINAL vinyl (a designed TCI warhead)
from a BETA-SUBSTITUTED or ARYL-CONJUGATED alkene. Measured on 16,348 unique molecules of the
corpus it built: 37.69% match a contaminant class against 15.16% genuine terminal acrylamide --
2.5 contaminants per real warhead. It admitted SUNITINIB (a marketed NON-covalent kinase
inhibitor), nintedanib's oxindole, rhodanines/TZDs (textbook PAINS) and N-ethylmaleimide (a
thiol-capping reagent). I wrote patterns and never looked at what they matched; a chemist found it
by looking at molecules.

THE RULE. A molecule is covalent-plausible if it carries at least one ACCEPTED warhead and no
REJECTED motif. Conjugation status is returned SEPARATELY, never folded into a class, because
holding it fixed between A and B is a precondition for any linker/placement instruction -- 16/20
sampled DOWN pairs flipped conjugation status.

VERSION IS PART OF THE PARAM DEFINITION. Changing this panel silently relabels every span, flex
and motif value downstream, so two corpora built under different PANEL_VERSION are NOT poolable.
Bump on any edit.
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PANEL_VERSION = 'cf-warheads-v1-2026-09-15'

ACCEPTED = {
    # Michael acceptors, TERMINAL or alkyl-beta only -- the designed-TCI kind
    'acrylamide_terminal':   '[CH2]=[CH1][CX3](=O)[NX3]',
    'acrylate_terminal':     '[CH2]=[CH1][CX3](=O)[OX2]',
    'crotonamide_alkyl':     '[CX4,NX3][CH1]=[CH1][CX3](=O)[NX3]',   # afatinib/neratinib/dacomitinib
    'propiolamide':          'C#C[CX3](=O)[NX3]',
    'vinyl_sulfonamide':     '[CH2]=[CH1][SX4](=O)(=O)[NX3,CX4]',
    # non-Michael electrophiles
    'haloacetamide':         '[Cl,Br,I][CH2][CX3](=O)[NX3]',
    'sulfonyl_fluoride':     '[SX4](=O)(=O)[F]',
    'epoxide':               '[CX4]1[OX2][CX4]1',
    'aziridine':             '[CX4]1[NX3][CX4]1',
    'beta_lactam':           'O=C1[NX3][CX4][CX4]1',
    'ketoamide':             '[CX3](=O)[CX3](=O)[NX3]',              # nirmatrelvir-like, reversible
    'aldehyde_peptidic':     '[CX3H1](=O)[CX4][NX3][CX3]=O',
    'boronic_acid':          '[BX3]([OX2H])[OX2H]',
    'cyanamide':             '[NX3][CX2]#[NX1]',
    'nitrile_activated':     '[CX4;$(C[NX3])][CX2]#[NX1]',           # alpha-amino nitrile only
}

# Motifs that DISQUALIFY regardless of what else matched. Each one is a named contaminant class
# the chemist identified in our own corpus, with the molecule that exposed it.
REJECTED = {
    'arylidene_conjugated':  '[c,n][CH1]=[CX3][CX3](=O)[NX3,OX2]',   # cinnamamide / chalcone / coumaroyl
    'arylidene_oxindole':    'O=C1Nc2ccccc2C1=[CX3]',                # SUNITINIB, nintedanib
    'rhodanine_a':           'O=C1[NX3]C(=[OX1,SX1])C(=[CX3])S1',
    'rhodanine_b':           'O=C1[NX3]C(=[OX1,SX1])[SX2]C1=[CX3]',
    'maleimide':             'O=C1C=CC(=O)N1',                       # N-ethylmaleimide: a REAGENT
    'barbiturate_ylidene':   'O=C1NC(=O)NC(=O)C1=[CX3]',
    'quinone':               'O=C1C=CC(=O)C=C1',
}

CONJUGATED = '[c,n][CH1]=[CX3][CX3](=O)'   # reported separately, never folded into the class

_A = {k: Chem.MolFromSmarts(v) for k, v in ACCEPTED.items()}
_R = {k: Chem.MolFromSmarts(v) for k, v in REJECTED.items()}
_C = Chem.MolFromSmarts(CONJUGATED)


def classify(smi_or_mol):
    """-> dict(ok, accepted, rejected, conjugated) or None if the SMILES will not parse."""
    m = Chem.MolFromSmiles(smi_or_mol) if isinstance(smi_or_mol, str) else smi_or_mol
    if m is None:
        return None
    acc = sorted(k for k, p in _A.items() if p is not None and m.HasSubstructMatch(p))
    rej = sorted(k for k, p in _R.items() if p is not None and m.HasSubstructMatch(p))
    return dict(ok=bool(acc) and not rej, accepted=acc, rejected=rej,
                conjugated=bool(_C is not None and m.HasSubstructMatch(_C)),
                panel=PANEL_VERSION)


if __name__ == '__main__':
    # THE TEST THAT MATTERS: named drugs the panel MUST get right. Writing patterns without
    # looking at what they match is exactly how sunitinib got into a covalent corpus.
    CASES = [
        ('ibrutinib',      'C=CC(=O)N1CCCC1c1ncnc2c1cnn2-c1ccc(Oc2ccccc2)cc1', True),
        ('osimertinib',    'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C', True),
        ('afatinib',       'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1', True),
        ('sotorasib_core', 'C=CC(=O)N1CCC(F)C(C1)n1c(=O)n(-c2cccc3cccnc23)c2cc(O)ccc2c1=O', True),
        ('nirmatrelvir',   'CC(C)(C)NC(=O)C1CC2(CC2)CN1C(=O)C(NC(=O)C(F)(F)F)C(C)(C)C', False),
        ('SUNITINIB',      'CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C', False),
        ('cinnamamide',    'O=C(/C=C/c1ccc(O)cc1)NCCc1ccccc1', False),
        ('N-ethylmaleimide','CCN1C(=O)C=CC1=O', False),
        ('rhodanine_PAINS','O=C1NC(=S)SC1=Cc1ccccc1', False),
    ]
    print('panel: %s' % PANEL_VERSION)
    bad = 0
    for name, smi, want in CASES:
        r = classify(smi)
        got = bool(r and r['ok'])
        mark = 'OK ' if got == want else '*** WRONG ***'
        if got != want: bad += 1
        print('  %-13s want %-5s got %-5s  %s   acc=%s rej=%s conj=%s'
              % (name, want, got, mark, (r or {}).get('accepted'), (r or {}).get('rejected'),
                 (r or {}).get('conjugated')))
    print('\n  %d/%d correct' % (len(CASES) - bad, len(CASES)))
