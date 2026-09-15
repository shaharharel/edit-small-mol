"""FINAL PAIR SET: TC >= 0.4, |dMW| <= 100 Da, MW <= 700, tagged by source. No stratification.

THE RULE, AND IT IS THE WHOLE RULE. Two filters and a mass cap:
  TC >= 0.4      similarity, so the pair is a plausible single edit rather than two molecules
  |dMW| <= 100   ONE med-chem move. phenyl +76, pyridyl +77, piperazine +84, morpholine +85 all
                 fit; two rings at once do not. Measured cost of the choice: 1,508,169 pairs at
                 100 Da against 1,713,387 at 150, so the tighter bound gives up 12% of the mass
                 and buys the "one edit" interpretation.
  MW <= 700      the pool runs to 1715 Da because CovInDB carries peptidic covalent ligands. A
                 1700 Da peptide edited by 100 Da is not the same operation as a 350 Da hit edited
                 by 100, and the model would be learning both under one instruction.
NO same-target constraint, NO single-cut MMP, NO shared-scaffold requirement. Those were inherited
from three different extractors and together kept 0.029% of the pair space, with EGFR alone
supplying 35,381 of 199,602 pairs. "Lengthen the linker" is chemistry, not a target property.
NO stratification and no descriptor-null gate: obedience is judged on GENERATIONS, not against a
precomputed floor.
"""
import os, sys, json, argparse
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/covalent_allvsall/pairs_tc0.40_mw150.jsonl')
    ap.add_argument('--union', default='data/covalent_union/pairs.jsonl')
    ap.add_argument('--max-dmw', type=float, default=100.0)
    ap.add_argument('--max-mw', type=float, default=700.0)
    ap.add_argument('--out', default='data/covalent_final/pairs.jsonl')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # source tags, so a cross-source pair is visible rather than pooled away
    cov, che = set(), set()
    for line in open(a.union):
        r = json.loads(line)
        t = cov if r.get('source') == 'covindb' else che
        t.add(r['input_smiles']); t.add(r['output_smiles'])

    mw = {}
    def ok_mw(s):
        if s not in mw:
            m = Chem.MolFromSmiles(s)
            mw[s] = Descriptors.MolWt(m) if m else 1e9
        return mw[s] <= a.max_mw

    n = kept = 0
    tags = {'covindb-covindb': 0, 'chembl-chembl': 0, 'CROSS': 0}
    drop_dmw = drop_mw = 0
    with open(a.out, 'w') as out:
        for line in open(a.pairs):
            r = json.loads(line); n += 1
            if r['dmw'] > a.max_dmw:
                drop_dmw += 1; continue
            if not (ok_mw(r['a']) and ok_mw(r['b'])):
                drop_mw += 1; continue
            sa = 'covindb' if r['a'] in cov else 'chembl'
            sb = 'covindb' if r['b'] in cov else 'chembl'
            tag = 'CROSS' if sa != sb else '%s-%s' % (sa, sb)
            tags[tag] += 1; kept += 1
            r['pair_source'] = tag
            out.write(json.dumps(r) + '\n')

    print('=== FINAL COVALENT PAIR SET ===')
    print('  read                     %9d' % n)
    print('  dropped |dMW| > %3d Da   %9d' % (int(a.max_dmw), drop_dmw))
    print('  dropped MW  > %3d Da     %9d' % (int(a.max_mw), drop_mw))
    print('  KEPT                     %9d' % kept)
    print()
    for k, v in sorted(tags.items(), key=lambda t: -t[1]):
        print('    %-18s %9d  %5.1f%%' % (k, v, 100 * v / max(kept, 1)))
    print('\n  wrote %s' % a.out)
    json.dump(dict(kept=kept, read=n, max_dmw=a.max_dmw, max_mw=a.max_mw,
                   by_pair_source=tags, tc_min=0.4,
                   filters='TC>=0.4 + |dMW|<=100 + MW<=700; no target, no MMP, no scaffold, '
                           'no stratification'),
              open(a.out.replace('.jsonl', '_meta.json'), 'w'), indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
