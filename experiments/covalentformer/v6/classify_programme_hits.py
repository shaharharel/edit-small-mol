#!/usr/bin/env python3
"""Classify every programme generation against what the target's chemistry actually did.

Four levels, strongest first, and each one answers a different question:

  EXACT      the proposed molecule IS the withheld reference analogue of its own pair.
  KNOWN      it IS some other compound measured against that target (ChEMBL or a PDB ligand).
  SAR_MOVE   it is a NEW molecule, but the TRANSFORMATION it applies -- the fragment removed and
             the fragment added -- is one that real pairs for that target also made. Two compounds
             can be far apart in Tanimoto while sharing the same move, so this catches a real SAR
             step applied to a different starting point, which a molecule-level test scores as a
             miss. This is the level the user asked for and the one a molecule comparison misses.
  NEAR       Tanimoto >= the threshold to a known compound, and closer to it than to its own input.

WHY `edit_smiles` IS NOT USED AS THE KEY. The pair file carries edit_smiles as `before>>after` over
a fragmentation this pipeline does not reproduce, so joining on it would silently miss or
mis-match. The transformation is recomputed by MCS on both sides instead -- our generations and the
target's known pairs -- so both keys come from the same procedure.

THE INPUT IS EXCLUDED EVERYWHERE. An earlier version left it in the comparison pool and its five
"hits" were all a generation reproducing its own input (7KX->7KX, A1IJC->A1IJC, INA->INA). A NEAR
hit must also be closer to the known compound than to the input, so "looks like what it was given"
cannot pass.

READ A MISS AS WEAK EVIDENCE. A pocket admits many reasonable edits, the programme made some of
them, and often for synthesis, patent or PK reasons the model is given no sight of. The hit count
is the headline; there is no honest miss rate here.
"""
import argparse
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'data', 'overlapping_assays', 'extracted',
    'overlapping_assay_pairs_minimal_mmp.csv'))
CHEMBL_OF = {'P52333': 'CHEMBL2148', 'P04626': 'CHEMBL1824', 'P43235': 'CHEMBL268',
             'P28074': 'CHEMBL4662', 'P26664': 'CHEMBL4296325'}


HEAVY = __import__('re').compile(r'(?:Cl|Br|[CNOSPFIcnops])')


def move_size(key):
    """Heavy atoms in a (removed, added) transformation."""
    return sum(len(HEAVY.findall(x or '')) for x in key)


def transform_key(a_smi, b_smi, Chem, rdFMCS):
    """(removed, added) canonical fragment SMILES for one a->b change, by MCS. None if unusable."""
    a, b = Chem.MolFromSmiles(a_smi or ''), Chem.MolFromSmiles(b_smi or '')
    if a is None or b is None:
        return None
    m = rdFMCS.FindMCS([a, b], timeout=3, ringMatchesRingOnly=True)
    core = Chem.MolFromSmarts(m.smartsString) if m.smartsString else None
    if core is None:
        return None
    ka, kb = set(a.GetSubstructMatch(core)), set(b.GetSubstructMatch(core))
    rm = frag(a, [i for i in range(a.GetNumAtoms()) if i not in ka], Chem)
    ad = frag(b, [i for i in range(b.GetNumAtoms()) if i not in kb], Chem)
    if rm is None or ad is None or (not rm and not ad):
        return None
    return (rm, ad)


def frag(mol, idxs, Chem):
    """Canonical SMILES of the submolecule on those atom indices, '' when empty."""
    if not idxs:
        return ''
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(idxs), canonical=True)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', nargs='+', required=True)
    ap.add_argument('--prompts', default=os.path.join(HERE, 'programme_prompts.jsonl'))
    ap.add_argument('--known', default=os.path.join(HERE, 'known_sar_hits_v2.json'),
                    help='only used to reuse the molecule-level pools if present')
    ap.add_argument('--near', type=float, default=0.85)
    ap.add_argument('--min-move', type=int, default=5,
                    help='minimum heavy atoms in removed+added fragments for a SAR_MOVE to count. '
                         'Without a floor the match is meaningless: on v2, 50 of 77 matches were '
                         '1-2 atom moves such as ">>C" or ">>Cl", and "add a methyl" is guaranteed '
                         'to appear among 4,316 known JAK3 transformations. It says the move is '
                         'common, not that the model reproduced real SAR.')
    ap.add_argument('--out', default=os.path.join(HERE, 'programme_hits_classified.json'))
    a = ap.parse_args()
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFMCS, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s or ''))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    prompts = {r['pair']: r for r in (json.loads(l) for l in open(a.prompts))}
    targets = {r['target'] for r in prompts.values()}
    want = {CHEMBL_OF[t]: t for t in targets if t in CHEMBL_OF}

    # known compounds and known TRANSFORMATIONS for each target, from the pair file
    known_mol = collections.defaultdict(dict)
    known_tf = collections.defaultdict(set)
    n_pairs = collections.Counter()
    with open(PAIRS) as fh:
        for row in csv.DictReader(fh):
            uni = want.get(row.get('target_chembl_id'))
            if not uni:
                continue
            n_pairs[uni] += 1
            for col in ('mol_a', 'mol_b'):
                cs = canon(row.get(col))
                if cs:
                    known_mol[uni].setdefault(cs, 'ChEMBL')
            k = transform_key(row.get('mol_a'), row.get('mol_b'), Chem, rdFMCS)
            if k:
                known_tf[uni].add(k)
                known_tf[uni].add((k[1], k[0]))      # the reverse move is equally real
    cache = os.path.join(HERE, 'known_transforms_cache.json')
    json.dump({u: sorted('%s>>%s' % k for k in v) for u, v in known_tf.items()},
              open(cache, 'w'))
    print('cached known transformations -> %s' % cache)
    print('target pools:')
    for uni in sorted(known_mol):
        print('  %-9s %6d pairs  %6d compounds  %6d distinct transformations'
              % (uni, n_pairs[uni], len(known_mol[uni]), len(known_tf[uni])))

    rows, counts = [], collections.Counter()
    for path in a.gens:
        if not os.path.exists(path):
            continue
        for r in (json.loads(l) for l in open(path)):
            pr = prompts.get(r['pair'])
            if pr is None:
                continue
            uni = pr['target']
            inp, ref = canon(pr['input_smiles']), canon(pr['reference_smiles'])
            kmol = {k: v for k, v in known_mol.get(uni, {}).items() if k != inp}
            ktf = known_tf.get(uni, set())
            fpi = gen.GetFingerprint(Chem.MolFromSmiles(inp)) if inp else None
            for g in r['generations']:
                cs = canon(g.get('gen_smiles'))
                if not cs or cs == inp:
                    continue
                kind, matched, score = None, None, None
                if ref and cs == ref:
                    kind, matched, score = 'EXACT', pr['analogue_code'], 1.0
                elif cs in kmol:
                    kind, matched, score = 'KNOWN', kmol[cs], 1.0
                else:
                    k = transform_key(pr['input_smiles'], cs, Chem, rdFMCS)
                    if k and k in ktf and move_size(k) >= a.min_move:
                        kind, matched, score = 'SAR_MOVE', '%s>>%s' % k, 1.0
                    else:
                        fp = gen.GetFingerprint(Chem.MolFromSmiles(cs))
                        tci = DataStructs.TanimotoSimilarity(fp, fpi) if fpi is not None else 0.0
                        best, who = 0.0, None
                        for km in kmol:
                            t = DataStructs.TanimotoSimilarity(
                                fp, gen.GetFingerprint(Chem.MolFromSmiles(km)))
                            if t > best:
                                best, who = t, kmol[km]
                        if best >= a.near and best > tci:
                            kind, matched, score = 'NEAR', who, round(best, 3)
                if kind:
                    counts[kind] += 1
                    rows.append({'kind': kind, 'pair': r['pair'], 'programme': pr['programme'],
                                 'site': pr['site'], 'pdb': pr['pdb'],
                                 'input_smiles': pr['input_smiles'], 'proposed_smiles': cs,
                                 'sentence': ' '.join((g.get('gen_sentence') or '').split()),
                                 'matched': matched, 'score': score, 'decode': g['decode']})
    order = {'EXACT': 0, 'KNOWN': 1, 'SAR_MOVE': 2, 'NEAR': 3}
    rows.sort(key=lambda r: (order[r['kind']], -(r['score'] or 0)))
    json.dump(rows, open(a.out, 'w'), indent=1)
    print('\n%s   total %d' % (dict(counts), len(rows)))
    for r in rows[:25]:
        print('  %-9s %-11s %-22s %s' % (r['kind'], r['programme'], r['pair'],
                                         str(r['matched'])[:48]))
    print('\nwrote %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
