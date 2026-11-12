#!/usr/bin/env python3
"""Score a reviewer's verdicts, and compare them against our automated QA.

LAST LINE WINS per (reviewer, item): the server appends, so a changed verdict is a second line.

THE COMPARISON IS THE POINT. We have automated checks on every one of these rows -- Tanimoto,
whether the rationale is grounded in its own quote, whether it reads as an instruction. None of
them was shown to the reviewer. So their verdicts are an independent read, and the question this
script answers is which automated check (if any) predicts a rejection. A check that does not
separate approvals from rejections is not measuring quality, whatever its own pass rate.

Usage:
  python score_review.py --reviews reviews.jsonl --items app/items.json
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', required=True)
    ap.add_argument('--items', required=True)
    a = ap.parse_args()

    items = {x['id']: x for x in json.load(open(os.path.expanduser(a.items)))}
    last = {}
    n_lines = 0
    for line in open(os.path.expanduser(a.reviews)):
        if not line.strip():
            continue
        n_lines += 1
        r = json.loads(line)
        last[(r.get('reviewer'), r.get('id'))] = r
    print('%d submissions -> %d current verdicts' % (n_lines, len(last)))

    by_rev = collections.defaultdict(list)
    for (rev, _), r in last.items():
        by_rev[rev].append(r)
    for rev, rs in sorted(by_rev.items()):
        ap_n = sum(1 for r in rs if r['verdict'] == 'approve')
        with_c = sum(1 for r in rs if (r.get('comment') or '').strip())
        print('  %-18s %3d judged, approve %3d (%.0f%%), %d with a comment'
              % (rev, len(rs), ap_n, 100 * ap_n / len(rs), with_c))

    # pooled, and the stratum break-down: the sample was built to the full set's delta mix
    pooled = {}
    for (rev, iid), r in last.items():
        pooled.setdefault(iid, []).append(r['verdict'])
    agree = [i for i, v in pooled.items() if len(set(v)) == 1 and len(v) > 1]
    split = [i for i, v in pooled.items() if len(set(v)) > 1]
    if len(pooled) != sum(len(v) for v in pooled.values()):
        print('\ndouble-reviewed items: %d agreed, %d split' % (len(agree), len(split)))

    def rate(sel, label):
        sel = [i for i in sel if i in pooled]
        if not sel:
            return
        ok = sum(1 for i in sel if pooled[i].count('approve') > len(pooled[i]) / 2)
        print('  %-34s %3d items, approved %.0f%%' % (label, len(sel), 100 * ok / len(sel)))

    print('\nby potency stratum (hidden from the reviewer):')
    for s in ('big', 'mid', 'small'):
        rate([i for i in pooled if items.get(i, {}).get('after_submit', {}).get('stratum') == s],
             '|delta| %s' % s)
    print('\nby our automated checks (none shown to the reviewer):')
    rate([i for i in pooled
          if (items.get(i, {}).get('after_submit', {}).get('tanimoto') or 0) >= 0.6],
         'Tanimoto >= 0.6')
    rate([i for i in pooled
          if (items.get(i, {}).get('after_submit', {}).get('tanimoto') or 0) < 0.6],
         'Tanimoto < 0.6')
    print('\nrejections with comments:')
    for (rev, iid), r in sorted(last.items()):
        if r['verdict'] == 'reject' and (r.get('comment') or '').strip():
            it = items.get(iid, {})
            print('  [%s] %s' % (rev, (r['comment'] or '').strip()[:140]))
            print('        reason was: %s' % (it.get('reason') or '')[:140])


if __name__ == '__main__':
    main()
