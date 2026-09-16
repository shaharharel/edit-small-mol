"""CovInDB PAIRS -- and the fix for wclass being a nitrile detector.

WHY CovInDB MATTERS HERE, specifically. Our wclass label is inferred by SMARTS matching, and that
inference is badly broken: QA measured nitrile present in 78.53% of wclass-labelled rows and
ordinary (non-warhead) nitriles at 70.43% -- so the channel is largely a nitrile-presence detector
rather than a warhead-class channel. CovInDB ships a CURATED `War_head` column. We stop guessing
and read the annotation. That removes the inference step that was the defect.

It is also the source I measured varying wclass/path/flex at 16-590x the rate of our generated
pairs, on genuine warhead swaps -- which is the other half of why wclass has been starving.

NO SINGLE-CUT MMP HERE EITHER, for the reasons in extract_similarity_pairs.py: the labels need a
ring, not a shared core, and TC>=0.4 is the similarity guarantee that single-cut MMP pairs already
satisfy anyway (measured: their TC min is 0.400, median 0.769).

SCOPE LIMIT, up front: CovInDB is ~8k unique ligands, so this is a SMALL, HIGH-QUALITY corpus, not
a replacement for ChEMBL. Its job is to supply warhead DIVERSITY that ChEMBL's acrylamide-heavy
pool does not.
"""
import os, sys, json, argparse
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ring_attach_atom(m):
    ring = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing()]
    if not ring:
        return None
    dm = Chem.GetDistanceMatrix(m)
    best, bd = None, -1
    for a in m.GetAtoms():
        if a.IsInRing():
            continue
        d = min(dm[a.GetIdx()][r] for r in ring)
        if d > bd:
            best, bd = a.GetIdx(), d
    return best


def labels(m):
    # FLEX RULE MUST MATCH v3 EXACTLY. extract_chembl_pairs.py:94 requires BOTH bond atoms to be
    # in `near` (within 3 bonds of the reference), i.e. max(d_i,d_j)<=3. I originally wrote
    # min(...)<=3 -- EITHER atom -- which is a strictly looser rule and disagrees with v3 on
    # 48.40%% of molecules. Two corpora labelled by different rules cannot be pooled or compared,
    # and every v3-vs-this-corpus flex number computed before this fix is invalid.
    at = ring_attach_atom(m)
    if at is None:
        return None
    ring = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing()]
    dm = Chem.GetDistanceMatrix(m)
    path = int(min(dm[at][r] for r in ring))
    flex = 0
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if max(dm[at][i], dm[at][j]) <= 3 and \
           b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1:
            flex += 1
    return path, flex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='../../data/covbinder/raw_covindb2/CovInDB_All.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tc', type=float, default=0.4)
    ap.add_argument('--mw', type=float, default=100.0)
    a = ap.parse_args()

    df = pd.read_csv(a.src)
    df = df.dropna(subset=['SMILES'])
    print('CovInDB rows: %d' % len(df))
    # one record per unique ligand; keep the CURATED warhead annotation
    recs = {}
    for _, r in df.iterrows():
        s = str(r['SMILES']).strip()
        if not s or s in recs:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        lab = labels(m)
        if lab is None:
            continue
        wh = str(r.get('War_head', '')).strip()
        recs[s] = dict(smiles=Chem.MolToSmiles(m), mol=m, wclass=wh if wh and wh != 'nan' else '',
                       target=str(r.get('Target', '')), path=lab[0], flex=lab[1],
                       mw=Descriptors.MolWt(m))
    items = list(recs.values())
    print('unique parsable ligands with a ring: %d' % len(items))
    from collections import Counter
    wc = Counter(i['wclass'] for i in items if i['wclass'])
    print('CURATED warhead classes: %d distinct' % len(wc))
    for k, v in wc.most_common(12):
        print('   %-34s %5d (%.1f%%)' % (k[:34], v, 100 * v / max(sum(wc.values()), 1)))

    fps = [GEN.GetFingerprint(i['mol']) for i in items]
    out = []
    for i in range(len(items)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, tc in enumerate(sims):
            j = i + 1 + k
            if tc < a.tc or abs(items[i]['mw'] - items[j]['mw']) > a.mw:
                continue
            A, B = items[i], items[j]
            try:
                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(A['mol']))
            except Exception:
                sc = ''
            out.append(dict(
                source='covindb', target=A['target'], scaffold=sc,
                input_smiles=A['smiles'], output_smiles=B['smiles'],
                tc=round(tc, 4), mw_delta=round(B['mw'] - A['mw'], 2),
                attach_path_a=A['path'], attach_path_b=B['path'],
                attach_path_dir='SAME' if A['path'] == B['path'] else ('UP' if B['path'] > A['path'] else 'DOWN'),
                attach_flex_a=A['flex'], attach_flex_b=B['flex'],
                attach_flex_dir='SAME' if A['flex'] == B['flex'] else ('UP' if B['flex'] > A['flex'] else 'DOWN'),
                wclass_a=A['wclass'], wclass_b=B['wclass'],
                wclass_dir=('' if not (A['wclass'] or B['wclass'])
                            else ('SAME' if A['wclass'] == B['wclass'] else 'CHANGED'))))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    print('\nPAIRS at TC>=%.2f, |dMW|<=%.0f : %d' % (a.tc, a.mw, len(out)))
    for p in ('attach_path', 'attach_flex', 'wclass'):
        c = Counter(r[p + '_dir'] for r in out if r.get(p + '_dir'))
        tot = sum(c.values()); st = tot - c.get('SAME', 0)
        if tot:
            print('  %-12s labelled=%-7d steerable=%-7d (%.1f%%)  %s'
                  % (p, tot, st, 100 * st / tot, dict(c)))
    print('wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
