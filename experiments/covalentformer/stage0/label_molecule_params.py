#!/usr/bin/env python
"""Label the three SMARTS/topology steering params, PER MOLECULE.

    acyl_N_motif        categorical  -- what the acyl nitrogen is attached to
    michael_subst_class categorical  -- what decorates the Michael acceptor
    linker_atom_count   integer      -- warhead electrophile -> first ring, DIRECTED

Per MOLECULE (~20k), not per pair (~1M): every param is a property of one structure, and the
pair label is the pair of molecule labels. Labelling pairs would recompute each molecule ~75x.

THREE RULES, each from a bug that shipped tonight:

1. A FAILED LABEL IS None, NEVER A DEFAULT. A bare `except` emitting 0 turns an RDKit failure
   into a confident "zero linker atoms" and a cohort of those reads as a real distribution.
2. THE ANCHOR IS THE WARHEAD ELECTROPHILE, resolved from covalent_filter's accepted match --
   not "the longest appendage tip", not an arbitrary terminal atom.
3. linker_atom_count IS DIRECTED. It walks from the electrophile THROUGH the carbonyl and the
   acyl heteroatom toward the Murcko scaffold. This is what killed warhead_span: an undirected
   `min` over shortest paths ran toward the scaffold only 49.6% of the time, silently switching
   between two opposite quantities. A directed walk makes that impossible BY CONSTRUCTION
   rather than filtering it out afterwards at 97% row cost.
"""
from __future__ import annotations
import os, sys, json, argparse, csv
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covalent_filter import classify, ACCEPTED, PANEL_VERSION   # noqa: E402

PARAM_VERSION = 'molparams-v1-2026-09-15'

# ---------------------------------------------------------------- acyl_N_motif
# The real named hit-to-lead move: exocyclic N-H -> endocyclic ring nitrogen. That single
# change is the ibrutinib / sotorasib / futibatinib warhead presentation, and it is the thing
# a covalent chemist actually instructs. Ordered: FIRST match wins, so keep specific first.
ACYL_N_MOTIF = [
    ('fused_indoline',  '[$([NX3;R2](@[CX3]=O))]'),                  # N in TWO rings (indoline etc)
    ('ring_N_4',        '[$([NX3;R;r4](@[#6])(@[#6])[CX3]=O)]'),     # azetidine
    ('ring_N_5',        '[$([NX3;R;r5](@[#6])(@[#6])[CX3]=O)]'),     # pyrrolidine
    ('ring_N_6',        '[$([NX3;R;r6](@[#6])(@[#6])[CX3]=O)]'),     # piperidine/piperazine
    ('N_methyl_aryl',   '[$([NX3]([CH3])(c)[CX3]=O)]'),              # tertiary N-Me anilide
    ('N_dialkyl',       '[$([NX3]([CX4])([CX4])[CX3]=O)]'),
    ('aryl_NH',         '[$([NX3;H1](c)[CX3]=O)]'),                  # anilide -- osimertinib class
    ('alkyl_NH',        '[$([NX3;H1]([CX4])[CX3]=O)]'),
    ('primary_NH2',     '[$([NX3;H2][CX3]=O)]'),
]

# ------------------------------------------------------- michael_subst_class
# Separates ibrutinib from a rhodanine. This is currently COLLAPSED INTO span, which is one
# reason span appeared to move: a "span" edit was often a warhead-class swap wearing a
# geometry label. Only meaningful for C=C-C(=O) acceptors; returns None otherwise, which is a
# genuine not-applicable and MUST NOT be pooled with a measured value.
MICHAEL_CLASS = [
    ('alpha_cyano',     '[CH1,CH0]=[CX3]([CX2]#[NX1])[CX3](=O)[NX3]'),
    ('beta_amino',      '[NX3][CH1]=[CH0,CH1][CX3](=O)[NX3]'),
    ('beta_aryl',       '[c,n][CH1]=[CH0,CH1][CX3](=O)[NX3]'),
    ('ring_embedded',   '[CX3;R]=[CX3;R][CX3](=O)[NX3]'),
    ('beta_alkyl',      '[CX4][CH1]=[CH1][CX3](=O)[NX3]'),           # crotonamide (afatinib)
    ('alpha_subst',     '[CH2]=[CX3;!$([CH1])][CX3](=O)[NX3]'),
    ('terminal',        '[CH2]=[CH1][CX3](=O)[NX3]'),                # the clean TCI acrylamide
]

_PAT = {}
def _pat(sma):
    if sma not in _PAT:
        _PAT[sma] = Chem.MolFromSmarts(sma)
    return _PAT[sma]


def electrophile_idx(mol, accepted):
    """Index of the electrophilic atom, from the warhead class the filter actually accepted.

    NOT a guess and NOT position 0 of an arbitrary match: for each accepted class we take its
    own SMARTS' first atom, which is written to BE the electrophile in every pattern in the
    panel. Returns None if no accepted class re-matches (tautomer drift between the filter call
    and here) -- that is a real failure and is reported, not defaulted.
    """
    for cls in accepted:
        p = _pat(ACCEPTED[cls])
        if p is None:
            continue
        m = mol.GetSubstructMatch(p)
        if m:
            return m[0], cls
    return None, None


def acyl_n_motif(mol):
    for name, sma in ACYL_N_MOTIF:
        p = _pat(sma)
        if p is not None and mol.HasSubstructMatch(p):
            return name
    return None


def michael_class(mol):
    if not mol.HasSubstructMatch(_pat('[CX3]=[CX3][CX3](=O)[NX3]')):
        return None          # NOT-APPLICABLE, not "unclassified". Never pool with a value.
    for name, sma in MICHAEL_CLASS:
        p = _pat(sma)
        if p is not None and mol.HasSubstructMatch(p):
            return name
    return 'other_michael'


def linker_atom_count(mol, e_idx):
    """Atoms strictly between the electrophile and the first ring, walking TOWARD the scaffold.

    Direction is fixed by the Murcko scaffold, not by a `min` over paths. We BFS out from the
    electrophile and return the distance to the nearest atom that is BOTH in a ring AND in the
    Murcko scaffold. The warhead tail is not in the Murcko scaffold, so a path running outward
    along it cannot terminate -- the 49.6%-of-the-time direction flip that killed warhead_span
    is structurally unreachable here.

    Returns None when the molecule HAS no Murcko ring system (acyclic), which is a genuine
    not-applicable.
    """
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    scaf_match = mol.GetSubstructMatch(scaf)
    if not scaf_match:
        return None
    target = set(i for i in scaf_match if mol.GetAtomWithIdx(i).IsInRing())
    if not target:
        return None

    # EXCLUDE THE RING SYSTEM THE ELECTROPHILE BELONGS TO OR IS FUSED TO.
    # Without this, a CYCLIC warhead terminates the BFS on its own ring at distance 0 and the
    # param is a CONSTANT. Measured on the v2 corpus: beta_lactam 814/814 = 100.0% zero with
    # ONE distinct value, epoxide 266/266 = 100.0% zero with ONE distinct value, 1,140
    # molecules in total carrying a label that cannot move. Those rows can only ever produce
    # ties at generation time, which is what a 34/37 tie rate on the first cohort showed.
    # This is the SAME defect already fixed in label_pocket_pairs.d_cys_scaffold; it was fixed
    # there and not carried across to here.
    ri = mol.GetRingInfo()
    e_rings = [set(r) for r in ri.AtomRings() if e_idx in r]
    if not e_rings:
        # electrophile is acyclic but may sit ON a ring atom's neighbour; exclude any ring it
        # is directly bonded into, otherwise the same collapse happens one bond out.
        nb = {n.GetIdx() for n in mol.GetAtomWithIdx(e_idx).GetNeighbors()}
        e_rings = [set(r) for r in ri.AtomRings() if nb & set(r)]
    banned = set()
    changed = True
    while changed:                       # grow across FUSED rings
        changed = False
        for r in (set(x) for x in ri.AtomRings()):
            if r & banned or any(r & er for er in e_rings):
                if not r <= banned:
                    banned |= r; changed = True
    target = target - banned
    if not target:
        return None      # the only ring system IS the warhead -> genuinely not applicable
    # BFS from the electrophile
    seen, frontier, dist = {e_idx}, [e_idx], 0
    while frontier:
        if any(i in target for i in frontier):
            return dist - 1 if dist > 0 else 0   # atoms STRICTLY BETWEEN
        nxt = []
        for i in frontier:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j not in seen:
                    seen.add(j); nxt.append(j)
        frontier = nxt; dist += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--union', default='data/covalent_union_v2/pairs.jsonl')
    ap.add_argument('--out', default='data/labels/molecule_params_v2.csv')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    mols = []
    seen = set()
    with open(a.union) as fh:
        for line in fh:
            r = json.loads(line)
            for s in (r.get('input_smiles', r.get('a')), r.get('output_smiles', r.get('b'))):
                if s not in seen:
                    seen.add(s); mols.append(s)
    if a.limit:
        mols = mols[:a.limit]
    print('unique molecules: %d' % len(mols)); sys.stdout.flush()

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    n_ok = 0
    fail = {'parse': 0, 'not_accepted': 0, 'no_electrophile': 0}
    got = {'acyl_N_motif': 0, 'michael_subst_class': 0, 'linker_atom_count': 0}
    with open(a.out, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['smiles', 'warhead_class', 'acyl_N_motif',
                    'michael_subst_class', 'linker_atom_count',
                    'param_version', 'panel_version'])
        for k, smi in enumerate(mols):
            if k and k % 5000 == 0:
                print('  %d/%d' % (k, len(mols))); sys.stdout.flush()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fail['parse'] += 1; continue
            cl = classify(mol)
            if not (cl and cl['ok']):
                fail['not_accepted'] += 1; continue
            e_idx, wcls = electrophile_idx(mol, cl['accepted'])
            if e_idx is None:
                fail['no_electrophile'] += 1; continue
            an = acyl_n_motif(mol)
            mc = michael_class(mol)
            lk = linker_atom_count(mol, e_idx)
            for nm, v in (('acyl_N_motif', an), ('michael_subst_class', mc),
                          ('linker_atom_count', lk)):
                if v is not None:
                    got[nm] += 1
            w.writerow([smi, wcls,
                        an if an is not None else '',
                        mc if mc is not None else '',
                        lk if lk is not None else '',
                        PARAM_VERSION, PANEL_VERSION])
            n_ok += 1

    print('\n=== MOLECULE PARAM LABELLING ===')
    print('  param_version %s   panel %s' % (PARAM_VERSION, PANEL_VERSION))
    print('  labelled           %8d / %d' % (n_ok, len(mols)))
    for k2, v in fail.items():
        print('  dropped %-16s %8d' % (k2, v))
    print('  -- coverage among labelled rows (blank = genuine not-applicable) --')
    for nm, c in got.items():
        print('    %-22s %8d  %5.1f%%' % (nm, c, 100.0 * c / max(n_ok, 1)))
    print('  wrote %s' % a.out)


if __name__ == '__main__':
    main()
