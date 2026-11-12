#!/usr/bin/env python
"""Export design episodes to a CSV a chemist can review, with two-extractor agreement marked.

WHY AGREEMENT IS A COLUMN AND NOT A FILTER. Two extractors read each paper independently. On
the v1 corpus, cross-extractor agreement turned out to be a perfect fabrication filter -- the
agreed stratum carried 0.0% fabricated rationales against 10-12% for records only one pass
found -- so whether both extractors found an episode is the single most useful signal we have
about it. But agreement is not correctness: both can miss the same thing, and a solo episode
from a rationale-dense paper may be perfectly good. So it is surfaced as `agreement` for the
reviewer to weigh rather than used to silently drop rows.

Matching episodes across extractors: same paper, and either the compound sets overlap or the
evidence quotes overlap substantially. Quotes are the stronger key because both extractors are
quoting the same paper text; compounds are the fallback when they quote different sentences
about one argument.

The `grounded` column is word overlap between the rationale and its own evidence quote. It is
a SHAPE check, not a truth check -- a rationale can reuse every word of its quote and still
misattribute the chemistry, which is exactly what a chemist panel found at 44% ("overstated")
while automated grounding looked fine. Read it as "obviously invented or not", nothing more.

Usage:
  python scripts/extraction/export_episodes_csv.py \
      --returned <scratch>/episode_v3/returned --out ~/Downloads/lit10_episodes_v3.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_intent_sentence_quality import grounded, score  # noqa: E402

STOP = set('the a an of to in on at and or for with that this it its'.split())


def toks(s):
    return {w for w in re.findall(r'[a-z][a-z0-9-]{2,}', (s or '').lower()) if w not in STOP}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), '*.json'))):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        # accept e1/e2 AND ex1/ex2: the work package names tasks __ex1/__ex2, and the
        # narrower pattern silently labelled every row 'e1', reporting 0% agreement
        # on a two-extractor set -- a wrong number, not a missing one.
        m = re.search(r'__(ex?\d+)\.json$', os.path.basename(f))
        ex = m.group(1) if m else (j.get('extractor') or 'e1')
        for e in (j.get('episodes') or []):
            rows.append(dict(paper=j.get('paper') or Path(f).stem, extractor=ex, **e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--returned', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--match-threshold', type=float, default=0.30)
    a = ap.parse_args()

    rows = load(a.returned)
    if not rows:
        sys.exit(f'no episodes found in {a.returned}')

    # match across extractors within a paper
    for r in rows:
        r['_q'] = toks(r.get('evidence_quote'))
        r['_c'] = {str(x).strip().lower() for x in
                   (r.get('compounds_before') or []) + (r.get('compounds_after') or [])
                   if str(x).strip()}
        r['_match'] = None
    by_paper = {}
    for r in rows:
        by_paper.setdefault(r['paper'], []).append(r)
    for p, rs in by_paper.items():
        exs = sorted({r['extractor'] for r in rs})
        if len(exs) < 2:
            continue
        A = [r for r in rs if r['extractor'] == exs[0]]
        B = [r for r in rs if r['extractor'] == exs[1]]
        pairs = []
        for i, x in enumerate(A):
            for j, y in enumerate(B):
                s = max(jac(x['_q'], y['_q']), jac(x['_c'], y['_c']))
                if s >= a.match_threshold:
                    pairs.append((s, i, j))
        pairs.sort(reverse=True)
        ua, ub = set(), set()
        for s, i, j in pairs:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            A[i]['_match'] = B[j]['_match'] = round(s, 2)

    # v5.3 fields ride along: without them the CSV silently drops `endpoint`,
    # `reference_source`, the per-analogue what_changed/outcome_direction and `secondhand`,
    # which is most of what the last three brief revisions added.
    fields = ['paper', 'extractor', 'agreement', 'match_score', 'intent_class', 'basis',
              'endpoint', 'endpoint_verbatim', 'secondhand',
              'evidence_section', 'change_from', 'change_to',
              'reference', 'reference_source', 'analogues_n', 'analogues',
              'pairs_n', 'pairs',
              'compounds_before', 'compounds_after', 'same_span',
              'reason_verbatim', 'design_rationale', 'outcome',
              'rationale_words', 'grounded_in_quote', 'briefable_shape', 'evidence_quote']
    out = os.path.expanduser(a.out)
    n_agree = 0
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x['paper'], x['extractor'])):
            q = r.get('evidence_quote') or ''
            sc = score(r.get('design_rationale') or '', q) or {}
            comps = [str(x) for x in (r.get('compounds_before') or [])
                     + (r.get('compounds_after') or [])]
            agree = 'both' if r['_match'] is not None else 'one_only'
            n_agree += (agree == 'both')
            w.writerow({
                'paper': r['paper'], 'extractor': r['extractor'],
                'agreement': agree, 'match_score': r['_match'] if r['_match'] else '',
                'intent_class': r.get('intent_class'), 'basis': r.get('basis'),
                'evidence_section': r.get('evidence_section'),
                'change_from': r.get('change_from'), 'change_to': r.get('change_to'),
                'compounds_before': '; '.join(str(x) for x in (r.get('compounds_before') or [])),
                'compounds_after': '; '.join(str(x) for x in (r.get('compounds_after') or [])),
                'endpoint': r.get('endpoint'),
                'endpoint_verbatim': r.get('endpoint_verbatim'),
                'secondhand': 'yes' if r.get('secondhand') else '',
                'reference': r.get('reference'),
                'reference_source': r.get('reference_source'),
                'analogues_n': len(r.get('analogues') or []),
                'analogues': ' | '.join(
                    '%s: %s [%s]' % (a.get('label'), (a.get('what_changed') or '')[:90],
                                     a.get('outcome_direction') or '')
                    for a in (r.get('analogues') or [])),
                'pairs_n': len(r.get('pairs') or []),
                'pairs': '; '.join('%s->%s' % (p[0], p[1])
                                   for p in (r.get('pairs') or []) if len(p) == 2),
                'same_span': 'yes' if any(c and c in q for c in comps) else 'no',
                'reason_verbatim': r.get('reason_verbatim'),
                'design_rationale': r.get('design_rationale'),
                'outcome': r.get('outcome'),
                'rationale_words': sc.get('words'),
                'grounded_in_quote': (f"{sc['grounded']:.2f}"
                                      if sc.get('grounded') is not None else ''),
                'briefable_shape': 'yes' if sc.get('briefable') else 'no',
                'evidence_quote': re.sub(r'\s+', ' ', q)[:1200],
            })
    exs = sorted({r['extractor'] for r in rows})
    print(f'wrote {out}')
    print(f'  {len(rows)} episode rows from {len(by_paper)} papers, extractors {exs}')
    for e in exs:
        print(f'    {e}: {sum(1 for r in rows if r["extractor"]==e)} episodes')
    print(f'  found by BOTH extractors: {n_agree} rows '
          f'({n_agree/len(rows):.0%}) -> {n_agree//2} matched episodes')
    print(f'  one extractor only: {len(rows)-n_agree} rows')
    sm = sum(1 for r in rows
             if any(str(c) and str(c) in (r.get('evidence_quote') or '')
                    for c in (r.get('compounds_before') or []) + (r.get('compounds_after') or [])))
    print(f'  compound label inside the quoted reason (same_span): {sm}/{len(rows)} '
          f'= {sm/len(rows):.0%}')


if __name__ == '__main__':
    main()
