"""warhead_planarity, PER MOLECULE, using the VERIFIED manuscript definition.

WHY PER MOLECULE AND NOT PER PAIR. Planarity is a property of ONE molecule -- the Michael-acceptor
dihedral C=C-C(=O)-N. There are 37,035 unique molecules behind 1,470,338 pairs, so labelling
molecules and JOINING is ~40x less work than labelling pairs. The pair delta is then a lookup.
That is also why the pair yield is not something to choose: it FALLS OUT of how many molecules
carry an acrylamide and embed successfully.

DEFINITION IS PINNED, NOT REINVENTED. This imports scripts/compute_planar_2d_e2.py, the producer
that generated the stored manuscript columns. Verified this session: re-running it against
paper/reproducibility/metrics/planar_2d/v2_cond.csv reproduces planar_dev_deg on 215/215 comparable
rows to 1e-6, max abs diff 0.000000. steer_params.phi_planar_dev is a DIFFERENT metric I wrote and
its own docstring says its numbers are not comparable to the manuscript panel -- it is not used here.

TWO FACTS ABOUT THE STORED COLUMN THAT MUST TRAVEL WITH ANY NUMBER FROM IT:
  * 2.62 deg for v2_cond is the MEDIAN. The MEAN is 26.02. The distribution is heavily skewed --
    most generations near-planar, a minority badly twisted. Quote the median.
  * embed_ok is 7,174/10,000 = 71.7% on that cohort. ETKDG fails on ~28%, and a pair needs BOTH
    sides, so the pair yield is roughly the square of the per-molecule rate.
FAILURE RETURNS None, NEVER A DEFAULT. A bare except emitting 0.0 turns an RDKit failure into a
confident "perfectly planar" label, and a cohort of those reads as a real result.
"""
import os, sys, json, csv, argparse, importlib.util
from multiprocessing import Pool

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
spec = importlib.util.spec_from_file_location('p2d', os.path.join(ROOT, 'scripts/compute_planar_2d_e2.py'))
P2D = importlib.util.module_from_spec(spec); spec.loader.exec_module(P2D)


def one(t):
    i, smi = t
    try:
        return P2D._compute_2d_one((i, smi))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/covalent_final/pairs_mw800.jsonl')
    ap.add_argument('--out', default='data/labels/planarity_by_molecule.csv')
    ap.add_argument('--procs', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    mols = set()
    for line in open(a.pairs):
        r = json.loads(line); mols.add(r['a']); mols.add(r['b'])
    mols = sorted(mols)
    if a.limit:
        mols = mols[:a.limit]
    print('unique molecules to label: %d  (from %s)' % (len(mols), a.pairs), flush=True)

    n_ok = n_acr = 0
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['smiles', 'acryl_match', 'embed_ok', 'dihedral_deg', 'planar_dev_deg'])
        with Pool(a.procs) as pool:
            for k, row in enumerate(pool.imap(one, enumerate(mols), chunksize=64)):
                smi = mols[k]
                if row is None:
                    w.writerow([smi, 0, 0, '', '']); continue
                # _compute_2d_one RETURNS A DICT, NOT A TUPLE. My first version did
                # `_, _, acr, ok, dih, dev = list(row)`, which iterates the dict's KEYS -- so every
                # row got the literal strings 'dihedral_deg'/'planar_dev_deg' written into the value
                # columns, and bool('acryl_match') is always True. The 200-molecule smoke reported
                # 100.00% acrylamide and 100.00% embed_ok, which is what gave it away: a perfect
                # rate on both is not a result, it is a parse error. Ninth instance tonight of code
                # that runs, writes a valid-looking file, and reports a confident wrong number.
                acr = row.get('acryl_match'); ok = row.get('embed_ok')
                dih = row.get('dihedral_deg'); dev = row.get('planar_dev_deg')
                n_acr += 1 if acr else 0; n_ok += 1 if ok else 0
                w.writerow([smi, int(bool(acr)), int(bool(ok)),
                            '' if dih is None else dih, '' if dev is None else dev])
                if (k + 1) % 2000 == 0:
                    print('  %6d/%d   acryl %5.1f%%   embed_ok %5.1f%%'
                          % (k + 1, len(mols), 100 * n_acr / (k + 1), 100 * n_ok / (k + 1)), flush=True)

    print('\n=== PER-MOLECULE PLANARITY DONE ===')
    print('  molecules          %7d' % len(mols))
    print('  with acrylamide    %7d  %5.2f%%' % (n_acr, 100 * n_acr / len(mols)))
    print('  embed_ok           %7d  %5.2f%%  <- a PAIR needs BOTH sides' % (n_ok, 100 * n_ok / len(mols)))
    print('  wrote %s' % a.out)
    print('  NEXT: join onto pairs; expected pair yield ~ (per-molecule usable rate)^2 x 1,470,338')
    return 0


if __name__ == '__main__':
    sys.exit(main())
