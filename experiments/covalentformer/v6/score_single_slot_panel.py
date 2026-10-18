#!/usr/bin/env python3
"""Pooled rates for a single-slot chemist panel.

One rate per field, pooled over every reviewer judgement -- three reviewers on the same item are
three judgements, which is the convention the `_codex` external columns were computed with. Also
prints the per-reviewer spread, because `would_show_a_chemist` has moved 20+ points between
reviewers on the same items and a single pooled number hides that.

Optionally restricts to a set of item_ids (--only), which is how the ChemDFM g3+g4 external round
was reported: that package was built before the content-dedup fix and held 126 distinct items in
150 slots, so its rates are over the distinct subset.
"""
import json, os, sys, argparse, collections

FIELDS = ['edit_identified_correctly', 'atom_attribution_ok', 'invented_chemistry',
          'chemically_sensible_edit', 'would_show_a_chemist']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('score_dir')
    ap.add_argument('--only', default=None, help='json list of item_ids to keep')
    a = ap.parse_args()
    keep = set(json.load(open(a.only))) if a.only else None

    per_rev = collections.defaultdict(lambda: collections.defaultdict(list))
    pooled = collections.defaultdict(list)
    items = set()
    for fn in sorted(os.listdir(a.score_dir)):
        if not fn.endswith('.json'):
            continue
        rev = fn.rsplit('_', 1)[1].replace('.json', '')
        d = json.load(open(os.path.join(a.score_dir, fn)))
        for iid, v in d.items():
            if keep is not None and iid not in keep:
                continue
            items.add(iid)
            for f in FIELDS:
                if f in v and isinstance(v[f], bool):
                    pooled[f].append(v[f])
                    per_rev[rev][f].append(v[f])

    print('%d distinct items, %d judgements per field (max)' %
          (len(items), max(len(v) for v in pooled.values())))
    print('%-28s %7s   %s' % ('field', 'pooled', 'per reviewer'))
    print('-' * 62)
    for f in FIELDS:
        v = pooled[f]
        spread = '  '.join('%s %.0f%%' % (r, 100.0 * sum(per_rev[r][f]) / len(per_rev[r][f]))
                           for r in sorted(per_rev) if per_rev[r][f])
        print('%-28s %6.1f%%   %s' % (f, 100.0 * sum(v) / len(v), spread))


if __name__ == '__main__':
    main()
