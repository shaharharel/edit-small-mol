"""CovalentFormer V1 training data: scaffold-with-attachment-point -> warhead-bearing fragment.

THE TASK. V1 is deliberately narrow: given a good binder with a known pose and a nearby ligandable
cysteine, generate the linker+electrophile that converts it into a covalent candidate. In LibInvent
terms that is scaffold decoration:
    input   scaffold containing one attachment point [*]
    output  the fragment that was removed, which MUST carry the warhead

WHY NO <KEEP>/<EDIT> TOKENS. The infilling formulation already enforces the constraint: the scaffold
never enters the decoder's output space, so it cannot be modified. Control tokens marking protected
atoms would spend vocabulary restating what the attachment point already says. The only conditioning
that is NOT recoverable from the input string is the reaction class, which is emitted as a separate
column rather than a token so it can be routed to the pocket encoder later.

THE CUT IS WARHEAD-DIRECTED, NOT RANDOM. A generic BRICS enumeration would mostly produce examples
where the warhead sits in the FROZEN half, teaching the model to decorate a covalent molecule with
inert groups -- the opposite of the intent. Every example here is cut so the electrophile lands in
the GENERATED fragment. The model therefore only ever learns "produce the reactive end".

CLASS-CONDITIONAL WARHEAD DETECTION. Michael acceptors and haloacetamides react through different
coordinates (conjugate 1,4-addition at C-beta, approach perpendicular to the alkene plane; versus
SN2 backside attack at ~180 degrees to the leaving group). They are labelled separately here because
a single geometric descriptor cannot serve both -- an error we made and measured: comparing an
in-plane angle against the Burgi-Dunitz 107 degree reference, which describes 1,2-addition to a
CARBONYL and is simply the wrong reaction for an acrylamide.

LEAKAGE CONTROL. Splits are by BEMIS-MURCKO SCAFFOLD, not at random. The covalent corpus is dense
with congeneric series, so a random split puts near-identical molecules on both sides and reports a
memorisation score. Scaffold-disjoint splits are the minimum honest standard here.
"""
import os, sys, json, csv, collections, random
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

WARHEADS = [
    ('michael_terminal',   '[CH2]=[CH]C(=O)[NX3]'),
    ('michael_substituted', '[CX3]=[CX3][CX3](=O)[NX3]'),
    ('haloacetamide',      '[F,Cl,Br,I][CH2]C(=O)[NX3]'),
    ('vinylsulfone',       '[CX3]=[CX3][SX4](=O)(=O)'),
    ('propiolamide',       'C#CC(=O)[NX3]'),
]
PATS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEADS]
MIN_FRAG_HEAVY, MAX_FRAG_HEAVY = 4, 24
MIN_SCAF_HEAVY = 10


def warhead_atoms(mol):
    """-> (set of atom indices in any warhead, class name) or (None, None)."""
    for name, p in PATS:
        if p is None:
            continue
        mt = mol.GetSubstructMatches(p)
        if mt:
            return set(mt[0]), name
    return None, None


def cuts_putting_warhead_in_fragment(mol):
    """Yield (scaffold_smiles_with_star, fragment_smiles_with_star) for BRICS bonds that
    separate the molecule so the WARHEAD ends up in the fragment."""
    wh, cls = warhead_atoms(mol)
    if wh is None:
        return
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return
    seen = set()
    for (a1, a2), _labels in bonds:
        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is None:
            continue
        try:
            frag = Chem.FragmentOnBonds(mol, [b.GetIdx()], addDummies=True)
            pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        if len(pieces) != 2:
            continue
        # the piece containing the warhead is the target; the other is the frozen scaffold
        tagged = []
        for p in pieces:
            ids = set()
            for at in p.GetAtoms():
                if at.HasProp('molAtomMapNumber'):
                    pass
            ids = None
            tagged.append(p)
        # identify by substructure rather than index bookkeeping, which FragmentOnBonds scrambles
        withwh = [p for p in pieces if warhead_atoms(p)[0] is not None]
        without = [p for p in pieces if warhead_atoms(p)[0] is None]
        if len(withwh) != 1 or len(without) != 1:
            continue
        fragment, scaffold = withwh[0], without[0]
        nf = fragment.GetNumHeavyAtoms() - 1        # minus the dummy
        ns = scaffold.GetNumHeavyAtoms() - 1
        if not (MIN_FRAG_HEAVY <= nf <= MAX_FRAG_HEAVY) or ns < MIN_SCAF_HEAVY:
            continue
        try:
            s_smi = Chem.MolToSmiles(scaffold)
            f_smi = Chem.MolToSmiles(fragment)
        except Exception:
            continue
        key = (s_smi, f_smi)
        if key in seen:
            continue
        seen.add(key)
        yield s_smi, f_smi, cls


def bm_scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        return None


def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    smis = []
    with open(src) as fh:
        r = csv.DictReader(fh)
        col = 'Target_Mol' if 'Target_Mol' in (r.fieldnames or []) else (r.fieldnames or [None])[0]
        for row in r:
            v = (row.get(col) or '').strip()
            if v:
                smis.append(v)
    smis = list(dict.fromkeys(smis))
    print('unique input molecules: %d  (column %r of %s)' % (len(smis), col, os.path.basename(src)))

    rows, stats = [], collections.Counter()
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            stats['unparseable'] += 1
            continue
        _wh, cls = warhead_atoms(m)
        if cls is None:
            stats['no_warhead'] += 1
            continue
        got = 0
        for s_smi, f_smi, c in cuts_putting_warhead_in_fragment(m):
            rows.append(dict(parent=smi, scaffold=s_smi, fragment=f_smi, warhead_class=c,
                             parent_scaffold=bm_scaffold(smi) or ''))
            got += 1
        stats['molecules_with_cuts' if got else 'no_valid_cut'] += 1
        stats['examples'] += got

    print('examples: %d from %d molecules' % (len(rows), stats['molecules_with_cuts']))
    print('  %s' % {k: v for k, v in stats.items() if k != 'examples'})
    print('  by warhead class: %s'
          % dict(collections.Counter(r['warhead_class'] for r in rows)))
    if not rows:
        print('NOTHING BUILT'); return 1
    n = [len(Chem.MolFromSmiles(r['fragment']).GetAtoms()) for r in rows[:2000]]
    print('  fragment heavy atoms (first 2000): median %d' % sorted(n)[len(n) // 2])

    # SCAFFOLD-DISJOINT SPLIT
    scafs = sorted({r['parent_scaffold'] for r in rows})
    random.Random(0).shuffle(scafs)
    n_val = max(1, int(0.1 * len(scafs)))
    val_s = set(scafs[:n_val])
    tr = [r for r in rows if r['parent_scaffold'] not in val_s]
    va = [r for r in rows if r['parent_scaffold'] in val_s]
    print('\nscaffold-disjoint split: %d train / %d val over %d scaffolds (%d held out)'
          % (len(tr), len(va), len(scafs), len(val_s)))
    ov = {r['parent_scaffold'] for r in tr} & {r['parent_scaffold'] for r in va}
    print('  scaffold overlap between splits: %d  (must be 0)' % len(ov))
    assert not ov, 'scaffold leakage'

    for name, part in (('train', tr), ('valid', va)):
        p = os.path.join(out_dir, '%s.smi' % name)
        with open(p, 'w') as fh:
            fh.write('Input\tOutput\n')
            for r in part:
                fh.write('%s\t%s\n' % (r['scaffold'], r['fragment']))
        print('  wrote %s (%d rows)' % (p, len(part)))
    with open(os.path.join(out_dir, 'meta.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['parent', 'scaffold', 'fragment', 'warhead_class',
                                'parent_scaffold'])
        w.writeheader()
        w.writerows(rows)
    print('  wrote meta.csv')
    return 0


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'models/train.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'experiments/covalentformer/data'
    sys.exit(main(src, out))
