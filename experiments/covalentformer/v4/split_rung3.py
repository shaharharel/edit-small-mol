#!/usr/bin/env python
"""Build a MOLECULE-DISJOINT train/test split of the rung-3 pairs.

WHY A NEW SPLIT RATHER THAN THE EXISTING FOLDS. The rung3_R3_*_{test,valid,mpro} folds were derived
from cdl.jsonl, and v4's rung-3 corpus is built from that same cdl.jsonl. Measured overlap of v4's
training rows against those folds: 99.5% of test, 98.5% of valid, 99.9% of mpro on the exact
(hit, analogue) key. Decontaminating against them instead leaves only 1,083 of 5,518 training rows
(80.37% touch an eval molecule), which cannot teach a pocket-conditioned task. The folds are not a
held-out set for this corpus and no filtering makes them one.

THE SPLIT KEY IS A CONNECTED COMPONENT, NOT A ROW. Splitting on the analogue alone still leaks: the
same hit can appear with a test analogue and a train analogue, so the model sees the test hit's
pocket during training. Treat (hit, analogue) as an edge in a bipartite molecule graph, take
connected components, and assign whole components. Then NO molecule -- hit or analogue -- appears on
both sides, which is the guarantee a generation metric needs.

Components are assigned largest-first to whichever side is under quota, so one giant component
cannot blow the ratio; the resulting fraction is reported rather than assumed.
"""
from __future__ import annotations
import argparse, collections, json

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def canon(s):
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True)
    ap.add_argument('--train-out', required=True)
    ap.add_argument('--test-out', required=True)
    ap.add_argument('--test-frac', type=float, default=0.18)
    ap.add_argument('--seed', type=int, default=20261001)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp)]
    keyed = []
    for r in rows:
        h, an = canon(r.get('hit')), canon(r.get('analogue'))
        if h and an:
            keyed.append((r, h, an))
    print('rows %d, usable (both sides parse) %d' % (len(rows), len(keyed)))

    # union-find over molecules
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _, h, an in keyed:
        union(h, an)

    comp = collections.defaultdict(list)
    for t in keyed:
        comp[find(t[1])].append(t)
    print('connected components: %d  (largest %d rows)'
          % (len(comp), max(len(v) for v in comp.values())))

    target = a.test_frac * len(keyed)
    test, train = [], []
    for _, members in sorted(comp.items(), key=lambda kv: -len(kv[1])):
        if len(test) < target and len(members) <= max(target - len(test), 1) * 1.5:
            test.extend(members)
        else:
            train.extend(members)

    tr_mols = {h for _, h, _ in train} | {an for _, _, an in train}
    te_mols = {h for _, h, _ in test} | {an for _, _, an in test}
    inter = tr_mols & te_mols
    print('\ntrain %d rows (%d molecules) | test %d rows (%d molecules) -> test is %.1f%%'
          % (len(train), len(tr_mols), len(test), len(te_mols),
             100.0 * len(test) / max(len(keyed), 1)))
    print('MOLECULE INTERSECTION: %d  (must be 0)' % len(inter))
    assert not inter, 'split is not molecule-disjoint -- refusing to write'

    tr_pairs = {(h, an) for _, h, an in train}
    te_pairs = {(h, an) for _, h, an in test}
    print('PAIR INTERSECTION:     %d  (must be 0)' % len(tr_pairs & te_pairs))
    assert not (tr_pairs & te_pairs)

    # the test side must still be big enough and chemically varied enough to measure on
    te_nuc = collections.Counter(r.get('nucleophile') for r, _, _ in test)
    te_op = collections.Counter(r.get('op') for r, _, _ in test)
    print('test nucleophiles: %s' % dict(te_nuc.most_common(6)))
    print('test ops:          %s' % dict(te_op.most_common(6)))
    assert len(test) >= 200, 'test fold too small to report a rate on (%d)' % len(test)

    for path, part in ((a.train_out, train), (a.test_out, test)):
        with open(path, 'w') as f:
            for r, _, _ in part:
                f.write(json.dumps(r) + '\n')
        print('wrote %-34s %6d rows' % (path, len(part)))
    print('SPLIT_DONE')


if __name__ == '__main__':
    main()
