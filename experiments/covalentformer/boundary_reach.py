"""BOUNDARY CALIBRATION: what is the oracle's actual spatial resolution?

WHY THE SPECIFICITY TEST WAS NOT ENOUGH. specificity_reach.py rejected 0 of 214 decoy cysteines --
true 69.7% vs decoy 0.0%. That looked decisive and is not, for a reason visible in its own numbers:
the best-conformer distance to the TRUE cysteine is a median 2.63 A, while the nearest decoy band
starts 8 A away FROM THE TRUE SG. Every decoy therefore demanded the electrophile travel roughly
three times its normal reach. 0% is mechanically guaranteed by that gap; it is evidence of
non-degeneracy, nothing more.

It matters because the training pairs we intend to mint sit exactly at the boundary. A
length-matched pair -- ortho-phenyl reaches at 5 A, para-phenyl does not -- differs by ~2-3 A. If the
oracle cannot resolve 2-3 A, every such pair is coin-flip noise and the DPO stage learns nothing.

THE TEST. Displace the true SG by a controlled delta along random unit vectors and measure the pass
rate as a function of delta. This gives a dose-response curve rather than a single number:

    delta = 0        -> 69.7% by construction (the recall rate)
    delta small      -> should stay high; the electrophile can still get there
    delta large      -> should fall to 0

The WIDTH of that decline is the oracle's effective resolution, and it is the number that decides
whether length-matched pairs carry signal. Read it as:

    falls to ~0 by 2 A   -> resolution is finer than our pair separation. Pairs are informative.
    still high at 6 A    -> the oracle is loose; "reachable" means "has a floppy linker" and the
                            matched pairs are noise. Tighten D_MAX/ANGLE_TOL before minting labels.

ONE KNOWN CONFOUND, and it biases AGAINST the conclusion we want. reachable() excludes protein atoms
within 3 A of the target point from the clash test (so the bonded cysteine is not counted as a
clash). When the target is displaced, that exclusion sphere moves with it and can mask genuine
protein atoms, making a displaced target EASIER to satisfy than it should be. A sharp decline
measured despite that leniency is therefore a lower bound on the true sharpness.
"""
import os, sys, json, collections, argparse
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import reachable, electrophile_index
from specificity_reach import load, scaffold_anchor, SCR

DELTAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
N_DIR = 3          # random displacement directions per delta
N_CONF = 120
RNG_SEED = 20260913


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='experiments/covalentformer/data/boundary.jsonl')
    a = ap.parse_args()

    P = {}
    for f in ('P_rung_final.json', 'P_rung_expanded.json'):
        p = os.path.join(SCR, 'ladder', f)
        if os.path.exists(p):
            for e in json.load(open(p))['entries']:
                P[e['pid']] = e
    items = sorted(P.items())
    if a.limit:
        items = items[:a.limit]
    if a.nshards > 1:
        items = items[a.shard::a.nshards]
    print('entries %d [shard %d/%d] deltas %s x %d directions'
          % (len(items), a.shard, a.nshards, DELTAS, N_DIR), flush=True)

    rng = np.random.default_rng(RNG_SEED)
    skip = collections.Counter()
    n_ok = 0
    with open(a.out, 'w') as fh:
        for pid, e in items:
            pre, cys, ch, lig = e.get('pre'), e.get('cys'), e.get('cys_ch', 'A'), e.get('lig')
            if not all([pre, cys, lig]):
                skip['no_fields'] += 1
                continue
            mol = Chem.MolFromSmiles(pre)
            if mol is None:
                skip['unparseable'] += 1
                continue
            if electrophile_index(mol)[0] is None:
                skip['no_warhead'] += 1
                continue
            try:
                sg, _all_sg, ligatoms, prot, pelem = load(pid, cys, ch, lig)
            except Exception:
                skip['cif'] += 1
                continue
            if sg is None or len(ligatoms) < 8:
                skip['no_partners'] += 1
                continue
            scaf, ref, err = scaffold_anchor(mol, ligatoms, sg)
            if err:
                skip[err] += 1
                continue

            row = {'pid': pid, 'hits': {}}
            for dl in DELTAS:
                hs = []
                ndir = 1 if dl == 0.0 else N_DIR
                for _k in range(ndir):
                    if dl == 0.0:
                        tgt = sg
                    else:
                        v = rng.normal(size=3)
                        tgt = sg + dl * v / np.linalg.norm(v)
                    r = reachable(mol, scaf, ref, tgt, prot, protein_elem=pelem, n_conf=N_CONF)
                    hs.append(bool(r.get('hit')))
                row['hits']['%.1f' % dl] = hs
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            n_ok += 1
            if n_ok % 5 == 0:
                print('  %d done' % n_ok, flush=True)

    print('scored %d ; skipped %s' % (n_ok, dict(skip)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
