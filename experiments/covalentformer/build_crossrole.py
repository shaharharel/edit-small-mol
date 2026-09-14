"""CROSS-ROLE EVALUATION SET: the rows that make "steering" a measurable quantity.

THE PROBLEM THIS FIXES. build_roles draws the replacement fragment from global_pool[SAME role] as
the one it removed, so on every row in the corpus

    classify(frag_from) == classify(frag_to) == role        4,691 / 4,691

The requested role is therefore identical to the role of the fragment already sitting in the anchor,
in 100% of rows. The model is never once asked to CHANGE a fragment's class. That makes the +0.2795
conditioning gap un-decomposable: "steers toward the requested role" and "points at the anchor
fragment of that class" predict exactly the same thing on exactly every row, so no measurement on
this corpus can separate them. The 4x4 matrix is the best available proxy and it is still a proxy,
because every off-diagonal cell asks for something the FIXED target does not contain.

WHAT THIS BUILDS. Rows where the two differ:

    anchor  = keep + fragment of role X
    target  = keep + fragment of role Y        Y != X
    request = Y

Now "the model emits a role-Y fragment" is evidence of steering and NOT of pointing, because the
fragment being replaced is role X. The reciprocal (request X, i.e. leave it alone) is the control.

This is an EVALUATION set, never a training set. Training on it would teach the mapping we are
trying to test for. It is built from the same retained halves so the chemistry is unchanged; only
the pairing differs.

HONEST LIMIT. A cross-role target is a molecule nobody made -- keep+frag_Y is a valid but synthetic
combination, same as every row in the corpus. It measures whether the token steers the decoder, not
whether the product is a good molecule.
"""
import os, sys, csv, json, random, argparse, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_roles import classify, join
from reachability import electrophile_index

ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='experiments/covalentformer/data/roles/valid_clean.csv')
    ap.add_argument('--out', default='experiments/covalentformer/data/roles/cross_role.csv')
    ap.add_argument('--max-mw', type=float, default=600.0)
    ap.add_argument('--seed', type=int, default=20260914)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = list(csv.DictReader(open(a.src)))
    print('source rows: %d' % len(rows))
    # pool of fragments per role, taken from the same held-out slice so nothing from train leaks in
    pool = collections.defaultdict(set)
    for r in rows:
        for k in ('frag_from', 'frag_to'):
            f = r.get(k)
            if f:
                pool[classify(Chem.MolFromSmiles(f), set(), set())].add(f)
    pool = {k: sorted(v) for k, v in pool.items()}
    print('  fragment pool by role: %s' % {k: len(v) for k, v in pool.items()})

    out, stats = [], collections.Counter()
    for r in rows:
        keep = r.get('keep')
        x = r.get('role')
        if not keep or x not in ROLES:
            continue
        others = [y for y in ROLES if y != x and pool.get(y)]
        if not others:
            stats['no_other_role'] += 1
            continue
        y = others[rng.randrange(len(others))]
        fy = pool[y][rng.randrange(len(pool[y]))]
        anc, tgt = r.get('anchor'), join(keep, fy)
        if not anc or not tgt or anc == tgt:
            stats['join_failed'] += 1
            continue
        m = Chem.MolFromSmiles(tgt)
        if m is None or Descriptors.MolWt(m) > a.max_mw:
            stats['mw_or_parse'] += 1
            continue
        # the point of the file: the anchor's fragment is role X, the target's is role Y
        out.append(dict(anchor=anc, target=tgt, keep=keep,
                        role=y,                 # what we REQUEST -- the target's class
                        anchor_role=x,          # what the anchor currently has
                        frag_from=r.get('frag_to', ''), frag_to=fy))
        stats['built'] += 1

    print('  built %d cross-role rows | %s' % (len(out), dict(stats)))
    if not out:
        print('NOTHING BUILT')
        return 1
    # verify the property this file exists to create
    same = sum(1 for r in out if r['role'] == r['anchor_role'])
    chk = sum(1 for r in out
              if classify(Chem.MolFromSmiles(r['frag_to']), set(), set()) == r['role'])
    print('  rows where requested role == anchor role: %d  (MUST be 0)' % same)
    print('  rows where classify(frag_to) == requested role: %d / %d' % (chk, len(out)))
    print('  requested-role distribution: %s'
          % dict(collections.Counter(r['role'] for r in out)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
