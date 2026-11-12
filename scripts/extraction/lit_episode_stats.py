#!/usr/bin/env python
"""Count what an episode return is actually worth for training: doubles and triplets.

THE THREE QUANTITIES ARE NOT INTERCHANGEABLE and the headline episode count is the least
useful of them:

  episodes   a design rationale extracted from the paper. Trains nothing on its own.
  doubles    (one molecule + its rationale). Usable as an absolute-frame example: "here is a
             molecule and why it was made". Needs >=1 compound label to resolve.
  triplets   (before molecule + after molecule + the rationale for the change). This is the
             edit-effect unit, and it needs BOTH sides to resolve to a ChEMBL structure.

A paper can be rich in episodes and poor in triplets -- a review names hundreds of compounds
and explains none of its own; a rationale written about a series names one compound and a
property. So the conversion rate from episodes to triplets is a property of the paper, which is
why it is reported per paper as well as in total.

Usage:
  python scripts/extraction/lit_episode_stats.py \
      --subs ~/Downloads/.../lit_episodes_1k_v4/submissions --pairs <resolved pairs csv>
"""
import argparse
import collections
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_resolve_compounds import build_index, resolve  # noqa: E402
CHEMBL = 'data/chembl_db/chembl/36/chembl_36.db'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', required=True)
    ap.add_argument('--pairs', help='resolved-pairs CSV from lit_resolve_compounds.py')
    ap.add_argument('--chembl-db', default=CHEMBL)
    ap.add_argument('--out', help='write per-paper CSV here')
    a = ap.parse_args()

    cx = sqlite3.connect(a.chembl_db)
    st = collections.Counter()
    per_paper = {}
    for f in sorted(Path(os.path.expanduser(a.subs)).glob('*.json')):
        d = json.loads(f.read_text())
        pf = d.get('paper')
        doc = pf.get('doc_id') if isinstance(pf, dict) else (pf or '')
        doc = str(doc)[4:] if str(doc).startswith('doc_') else str(doc)
        eps = d.get('episodes') or d.get('intents') or d.get('records') or []
        bare, scoped = build_index(cx, doc)
        p = per_paper.setdefault(doc, collections.Counter())
        st['papers'] += 1
        p['episodes'] += len(eps)
        st['episodes'] += len(eps)
        for e in eps:
            pre = [str(x).strip() for x in (e.get('compounds_before') or []) if str(x).strip()]
            post = [str(x).strip() for x in (e.get('compounds_after') or []) if str(x).strip()]
            rpre = {resolve(x, bare, scoped)[0] for x in pre}
            rpost = {resolve(x, bare, scoped)[0] for x in post}
            rpre.discard(None)
            rpost.discard(None)
            if e.get('design_rationale'):
                st['episodes_with_rationale'] += 1
            # a double needs ONE resolved molecule, either side
            n_double = len(rpre | rpost)
            if n_double:
                st['episodes_yielding_a_double'] += 1
                st['doubles'] += n_double
                p['doubles'] += n_double
            if rpre and rpost:
                st['episodes_yielding_a_triplet'] += 1
            # labels that never resolved: the ceiling on everything above
            st['labels_named'] += len(pre) + len(post)
            st['labels_unresolved'] += (len(pre) - len([x for x in pre if resolve(x, bare, scoped)[0]])
                                        + len(post) - len([x for x in post if resolve(x, bare, scoped)[0]]))
    print('papers                        %6d' % st['papers'])
    print('episodes                      %6d  (%.1f per paper)'
          % (st['episodes'], st['episodes'] / max(st['papers'], 1)))
    print('  with a design rationale     %6d  (%.1f%%)'
          % (st['episodes_with_rationale'], 100 * st['episodes_with_rationale'] / max(st['episodes'], 1)))
    print('  yielding >=1 DOUBLE         %6d  (%.1f%%)'
          % (st['episodes_yielding_a_double'],
             100 * st['episodes_yielding_a_double'] / max(st['episodes'], 1)))
    print('  yielding >=1 TRIPLET        %6d  (%.1f%%)'
          % (st['episodes_yielding_a_triplet'],
             100 * st['episodes_yielding_a_triplet'] / max(st['episodes'], 1)))
    print('DOUBLES (mol + rationale)     %6d  (%.1f per paper)'
          % (st['doubles'], st['doubles'] / max(st['papers'], 1)))
    print('compound labels named         %6d, unresolved %d (%.1f%%)'
          % (st['labels_named'], st['labels_unresolved'],
             100 * st['labels_unresolved'] / max(st['labels_named'], 1)))

    if a.pairs and os.path.exists(os.path.expanduser(a.pairs)):
        rows = list(csv.DictReader(open(os.path.expanduser(a.pairs))))
        withd = [r for r in rows if (r.get('delta') or '').strip()]
        pp = collections.Counter(r['doc_id'] for r in rows)
        print('TRIPLETS (before+after+why)   %6d  (%.1f per paper)'
              % (len(rows), len(rows) / max(st['papers'], 1)))
        print('  with a same-assay delta     %6d  (%.1f%%)'
              % (len(withd), 100 * len(withd) / max(len(rows), 1)))
        print('  distinct molecules          %6d'
              % len({r['mol_a_id'] for r in rows} | {r['mol_b_id'] for r in rows}))
        print('  distinct targets            %6d' % len({r['target_chembl_id'] for r in rows
                                                         if r.get('target_chembl_id')}))
        print('  papers contributing         %6d' % len(pp))
        for doc, c in pp.most_common(1):
            print('  most from one paper         %6d  (doc %s)' % (c, doc))
        for doc in pp:
            per_paper.setdefault(doc, collections.Counter())['triplets'] = pp[doc]
    if a.out:
        with open(os.path.expanduser(a.out), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['doc_id', 'episodes', 'doubles', 'triplets'])
            for doc, c in sorted(per_paper.items(),
                                 key=lambda kv: -kv[1].get('triplets', 0)):
                w.writerow([doc, c['episodes'], c['doubles'], c.get('triplets', 0)])
        print('\nwrote', a.out)


if __name__ == '__main__':
    main()
