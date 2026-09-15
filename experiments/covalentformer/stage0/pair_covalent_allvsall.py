"""ALL-VS-ALL PAIRING WITHIN THE COVALENT SET: TC >= 0.4 and a MW window. Nothing else.

WHY THIS REPLACES THE INHERITED PAIRING. The union corpus keeps 199,602 pairs from 37,035
molecules -- 0.029% of the 6.86e8 unordered pair space -- because every pair was INHERITED from an
extractor that also required single-cut MMP (ChEMBL v3), or a shared Murcko scaffold (CovInDB),
AND the same target in all cases. EGFR alone supplies 35,381 of those pairs. None of those
constraints is needed to teach "lengthen the linker": that is a chemistry instruction, not a
target-specific one.

THE MW WINDOW, AND WHY. A pair must represent ONE plausible med-chem move, because that is what the
model is being asked to perform. Reference points for a single substituent change:
    methyl +14   ethyl +28   methoxy +30   Cl +34   CF3 +68   phenyl +76   pyridyl +77
    morpholine +85   piperazine +84
So ~100 Da covers any single standard hit-to-lead move including adding a whole ring. Below ~50 Da
excludes ring additions, which are the commonest potency play. Above ~150 Da you are merging
fragments, not making one edit. Several windows are reported so the choice is made on the numbers.

COST CONTROL. Sorting by MW turns the O(n^2) sweep into a sliding window: a molecule is only
compared against those within the MW window, which is a tiny fraction of 37,035.
"""
import os, sys, json, argparse, itertools
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/covalent_union/pairs.jsonl')
    ap.add_argument('--tc', type=float, default=0.4)
    ap.add_argument('--windows', default='30,50,100,150')
    ap.add_argument('--out', default='data/covalent_allvsall')
    a = ap.parse_args()

    smis = set()
    for line in open(a.src):
        r = json.loads(line); smis.add(r['input_smiles']); smis.add(r['output_smiles'])
    smis = sorted(smis)
    print('unique covalent molecules: %d' % len(smis), flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    keep, fps, mws = [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        keep.append(s); fps.append(gen.GetFingerprint(m)); mws.append(Descriptors.MolWt(m))
    order = np.argsort(mws)
    keep = [keep[i] for i in order]; fps = [fps[i] for i in order]
    mws = np.array([mws[i] for i in order])
    n = len(keep)
    print('parsed %d, MW range %.1f - %.1f' % (n, mws[0], mws[-1]), flush=True)

    wins = sorted(float(w) for w in a.windows.split(','))
    counts = {w: 0 for w in wins}
    biggest = max(wins)
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, 'pairs_tc%.2f_mw%d.jsonl' % (a.tc, int(biggest))), 'w')
    for i in range(n):
        j = i + 1
        while j < n and mws[j] - mws[i] <= biggest:
            j += 1
        if j <= i + 1: continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:j])
        for off, s in enumerate(sims):
            if s < a.tc: continue
            k = i + 1 + off
            dmw = mws[k] - mws[i]
            for w in wins:
                if dmw <= w: counts[w] += 1
            fh.write(json.dumps(dict(a=keep[i], b=keep[k], tc=round(s, 4),
                                     dmw=round(float(dmw), 2))) + '\n')
        if i % 2000 == 0:
            print('  %6d/%d   pairs@%d Da so far %d' % (i, n, int(biggest), counts[biggest]), flush=True)
    fh.close()
    print('\n=== ALL-VS-ALL, TC >= %.2f ===' % a.tc)
    for w in wins:
        print('  MW window +/- %3d Da : %10d unordered pairs   (%.1fx the inherited 199,602)'
              % (int(w), counts[w], counts[w] / 199602.0))
    json.dump(dict(n_molecules=n, tc=a.tc, counts={str(int(w)): counts[w] for w in wins}),
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
