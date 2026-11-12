#!/usr/bin/env python
"""Does the RATIONALE TEXT talk about chemistry that is actually in this pair?

check_change_consistency.py already cross-examines the structured `change_from`/`change_to`
fields. This asks the blunter question a chemist asks on sight: the sentence says "replace the
trifluoromethyl" -- is there a trifluoromethyl in either molecule at all?

FOUR CLASSES OF CLAIM, all decidable from the two SMILES alone:

  GROUP     a named substructure ("sulfonamide", "piperazine", "fluorine"). If it appears in
            NEITHER molecule, the sentence is not about this pair. Hard fail.
  DIRECTION "add/install X" must increase X's count; "remove/delete X" must decrease it;
            "replace X with Y" must do both. A reversed direction is recoverable and is
            reported separately from a claim that is simply false.
  STEREO    "invert the configuration", "epimer", "cis/trans" require the pair to differ in
            stereochemistry and nowhere else. A stereo claim on a constitutional change is a
            mismatch.
  RING      "ring expansion/contraction" requires the ring sizes to differ.

WHY THIS IS A FILTER AND NOT A SCORE. Every verdict here is a statement about the TEXT against
the STRUCTURES; it cannot see whether the reasoning is good. It catches the specific defect
that makes a reviewer stop trusting the whole set -- a rationale about a group that is not
there -- and nothing else. `no_claim` is the commonest verdict by design: silence, not a pass.

A pair whose two molecules are IDENTICAL after canonicalisation fails regardless of text: there
is no edit to explain.

Usage:
  python scripts/extraction/check_claims_vs_structures.py \
      --resolved <resolved.csv> --intents <submissions dir> --out <annotated csv>
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_change_consistency import LEX  # noqa: E402

# a few the other lexicon does not carry, in the words rationales actually use
# Ring SMARTS are rewritten with a generic aromatic nitrogen. `[nH]` demands a free N-H, so every
# N-methyl / N-aryl / N-acyl azole failed to match its own name: doc_11198's molecules ARE indoles
# (N-methylindoles), doc_127331's IS a pyrazole (N-acylated). 826 of 3,041 absent_group flags flip
# to present under this one change.
NH_FIX = {
    'indole': 'c1ccc2[n]ccc2c1', 'pyrrole': 'c1cc[n]c1', 'imidazole': 'c1c[n]c[n]1',
    'pyrazole': 'c1c[n][n]c1', 'tetrazole': 'c1[n][n][n][n]1', 'triazole': 'c1c[n][n][n]1',
    'benzimidazole': 'c1ccc2[n]c[n]c2c1',
}

EXTRA = {
    'trifluoromethyl': 'C(F)(F)F', 'cf3': 'C(F)(F)F', 'methyl': '[CH3]',
    'ethyl': '[CH2][CH3]', 'isopropyl': '[CH]([CH3])[CH3]', 'tert-butyl': 'C([CH3])([CH3])[CH3]',
    'phenol': 'c[OX2H]', 'aniline': 'c[NX3;H2]', 'ketone': '[#6]C(=O)[#6]',
    'aldehyde': '[CX3H1](=O)', 'ether': '[#6]O[#6]', 'thioether': '[#6]S[#6]',
    'alkyne': 'C#C', 'alkene': 'C=C', 'acrylamide': 'C=CC(=O)N',
    'boronic acid': 'B(O)O', 'phosphonate': 'P(=O)(O)O', 'sulfonyl': 'S(=O)(=O)',
    'carbonyl': '[CX3]=[OX1]', 'quinoline': 'c1ccc2ncccc2c1', 'benzimidazole': 'c1ccc2[nH]cnc2c1',
    'nitrogen': '[#7]', 'sulfur': '[#16]', 'oxygen': '[#8]', 'halogen': '[F,Cl,Br,I]',
}
# "an additional hydrogen bond with the phenol of residue Tyr238" is a claim about the PROTEIN.
# Testing it against the ligand flagged 455 correct rationales. If the sentence is talking about
# the binding site, group presence in the ligand is not the question.
PROTEIN_SIDE = re.compile(r'\bresidue|\bside[- ]chain|\bamino acid|\bbackbone|\bpocket\b|'
                          r'\b(ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|'
                          r'THR|TRP|TYR|VAL)\s?\d+', re.I)
# a group named as the LOCATION of a change, or explicitly retained, must not be required to move
ANCHOR = re.compile(r'\b(on|at|to|of|onto|into|within|around|flanking) the [^.,;]{0,30}$', re.I)
RETAIN = re.compile(r'\b(retain\w*|keep\w*|preserv\w*|maintain\w*|unchanged|conserve\w*)\b', re.I)

ADD = re.compile(r'\b(add|adds|adding|install|installs|installing|introduce|introduces|'
                 r'introducing|append|incorporat\w+)\b', re.I)
DEL = re.compile(r'\b(remove|removes|removing|delete|deletes|deleting|excise|excises|'
                 r'truncat\w+|drop|drops|dropping|strip\w*)\b', re.I)
REPL = re.compile(r'\b(replac\w+|swap\w*|exchang\w+|substitut\w+)\b', re.I)
STEREO = re.compile(r'\b(invert\w*|inversion|configuration|epimer\w*|enantiomer\w*|'
                    r'stereochem\w*|\(R\)|\(S\)|cis|trans)\b', re.I)
RING = re.compile(r'\bring (expansion|contraction|expand\w*|contract\w*)\b|\b(expand|contract)\w* '
                  r'the ring\b', re.I)


def count(mol, smarts, cache={}):
    if smarts not in cache:
        cache[smarts] = Chem.MolFromSmarts(smarts)
    p = cache[smarts]
    return len(mol.GetSubstructMatches(p)) if (mol and p) else 0


def flat(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if not m:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=False)


def ring_sizes(mol):
    return sorted(len(r) for r in mol.GetRingInfo().AtomRings()) if mol else []


def judge(text, ma, mb):
    """-> (verdict, detail). verdict in {ok, reversed, absent_group, wrong_direction,
    stereo_mismatch, ring_mismatch, no_edit, no_claim}"""
    t = (text or '').lower()
    lex = dict(LEX)
    lex.update(EXTRA)
    lex.update(NH_FIX)

    if ma and mb and Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb):
        return 'no_edit', 'the two molecules are identical'

    named = []
    for name, sma in lex.items():
        # word-boundary match so 'amine' does not fire inside 'examine'
        if re.search(r'\b' + re.escape(name) + r'(s|es)?\b', t):
            named.append((name, sma))
    # prefer the most specific name when several overlap (trifluoromethyl over methyl)
    named.sort(key=lambda kv: -len(kv[0]))
    seen_smarts, uniq = set(), []
    for n, s in named:
        if s in seen_smarts:
            continue
        seen_smarts.add(s)
        uniq.append((n, s))
    if not uniq:
        if STEREO.search(t):
            fa, fb = flat(Chem.MolToSmiles(ma)), flat(Chem.MolToSmiles(mb))
            if fa and fb and fa == fb:
                return 'ok', 'stereochemistry-only change, as claimed'
            return 'stereo_mismatch', 'claims a stereochemical change; the skeletons differ'
        if RING.search(t):
            return ('ok', 'ring sizes differ') if ring_sizes(ma) != ring_sizes(mb) else \
                   ('ring_mismatch', 'claims a ring change; ring sizes identical')
        return 'no_claim', 'no nameable group in the sentence'

    # every named group must exist somewhere in the pair
    absent = [n for n, s in uniq if count(ma, s) == 0 and count(mb, s) == 0]
    if absent and PROTEIN_SIDE.search(t):
        return 'no_claim', 'the absent group is named on the protein side, not the ligand'
    if absent:
        return 'absent_group', 'named but in neither molecule: ' + ', '.join(sorted(set(absent))[:4])

    # direction, when the sentence states one
    moved = [(n, count(ma, s), count(mb, s)) for n, s in uniq]
    changed = [x for x in moved if x[1] != x[2]]
    if ADD.search(t) and not REPL.search(t):
        if changed and all(b < a for _, a, b in changed):
            return 'reversed', 'says add; every named group decreases'
    if DEL.search(t) and not REPL.search(t):
        if changed and all(b > a for _, a, b in changed):
            return 'reversed', 'says remove; every named group increases'
    if RETAIN.search(t) and not changed:
        return 'ok', 'an explicit retention claim; the named group correctly does not move'
    if (ADD.search(t) or DEL.search(t) or REPL.search(t)) and not changed:
        # the group may be the ANCHOR the change was made to ("substituents introduced to the
        # phenyl A group"), in which case its count is supposed to stay put. 850 of 1,320
        # wrong_direction flags were this.
        for nm, _, _ in moved:
            if re.search(r'\b(to|on|at|of|onto|into) the [^.,;]{0,25}\b' + re.escape(nm), t):
                return 'ok', 'the named group is the site of the change, not the thing changed'
        return 'wrong_direction', ('names %s and states a change, but their counts are identical '
                                   'in both molecules' % ', '.join(n for n, _, _ in moved[:3]))
    return 'ok', 'named groups present; counts move consistently'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resolved', required=True)
    ap.add_argument('--intents', required=True)
    ap.add_argument('--out')
    ap.add_argument('--min-tc', type=float, default=0.0)
    a = ap.parse_args()

    eps = {}
    for f in Path(os.path.expanduser(a.intents)).glob('*.json'):
        d = json.loads(f.read_text())
        for i, e in enumerate(d.get('episodes') or d.get('intents') or []):
            eps['%s#%d' % (f.stem, i)] = e

    rows = list(csv.DictReader(open(os.path.expanduser(a.resolved))))
    stat = collections.Counter()
    out = []
    for r in rows:
        ep = eps.get(r.get('intent_id')) or {}
        text = ep.get('design_rationale') or ''
        ma, mb = Chem.MolFromSmiles(r['mol_a']), Chem.MolFromSmiles(r['mol_b'])
        if ma is None or mb is None:
            v, d = 'unparseable', 'a molecule does not parse'
        else:
            v, d = judge(text, ma, mb)
        stat[v] += 1
        rr = dict(r)
        rr['claim_check'] = v
        rr['claim_detail'] = d
        out.append(rr)

    n = len(rows)
    print('%d triplets checked\n' % n)
    order = ['ok', 'no_claim', 'absent_group', 'wrong_direction', 'reversed',
             'stereo_mismatch', 'ring_mismatch', 'no_edit', 'unparseable']
    for k in order:
        if stat[k]:
            print('  %-16s %6d  %5.1f%%' % (k, stat[k], 100 * stat[k] / n))
    hard = sum(stat[k] for k in ('absent_group', 'wrong_direction', 'stereo_mismatch',
                                 'ring_mismatch', 'no_edit', 'unparseable'))
    print('\n  would be filtered out: %d (%.1f%%)' % (hard, 100 * hard / n))
    print('  survive: %d' % (n - hard))
    print('\nexamples of what gets caught:')
    shown = collections.Counter()
    for rr in out:
        v = rr['claim_check']
        if v in ('absent_group', 'wrong_direction', 'stereo_mismatch', 'no_edit') and shown[v] < 2:
            shown[v] += 1
            ep = eps.get(rr.get('intent_id')) or {}
            print('  [%s] %s' % (v, rr['claim_detail']))
            print('      %s' % (ep.get('design_rationale') or '')[:150])
    if a.out:
        with open(os.path.expanduser(a.out), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
