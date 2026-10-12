#!/usr/bin/env python3
"""Build v7 evidence from v7_pairs.jsonl with evidence_v4, recording WHY every drop happened.

Two differences from gen_ev4 besides the builder:

* every rejection reason is a named counter that reconciles to the input count, because the v6
  funnel could not be reconciled -- `v6_pairs_clean.jsonl` lost 10,381 pairs to a rule that is in
  no surviving script and cannot be inferred from the data.
* each pair is built TWICE and the two results compared. The v6 site-of-change depended on wall
  clock (rdFMCS is bounded by a timeout), so the same pair could be stored as "all 31 atoms
  changed" and recompute as "all 31 conserved". A corpus whose labels are not reproducible cannot
  be QA'd, so reproducibility is asserted during the build rather than hoped for.
"""
import json, os, sys, collections
from multiprocessing import Pool, cpu_count
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_v4 import build

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
MAX_EDIT = 15


def one(r):
    # a +15 heavy-atom change is not one design decision; reviewer 2: "compressing a 21-heavy-atom
    # dimerisation into one pocket sentence is a fiction regardless of the geometry fields"
    if abs(r['heavy'][1] - r['heavy'][0]) > MAX_EDIT:
        return {'_err': 'edit larger than %d heavy atoms' % MAX_EDIT}
    try:
        e = build(r)
    except Exception as ex:
        return {'_err': 'exception: %s' % type(ex).__name__}
    if not e:
        return {'_err': 'no usable complex (missing cif, too few pocket atoms, or pervasive clashes)'}
    if not e['LABEL_ONLY'].get('site_of_change'):
        # named, not silent: see the note on `unmodelled_site_atoms` in evidence_v4.build
        if e['QA'].get('unmodelled_site_atoms'):
            return {'_err': 'the edited atom is not modelled in the hit crystal'}
        return {'_err': 'no site of change could be located'}
    try:
        e2 = build(r)
        if json.dumps(e, sort_keys=True) != json.dumps(e2, sort_keys=True):
            e['QA']['NOT_REPRODUCIBLE'] = True
    except Exception:
        e['QA']['NOT_REPRODUCIBLE'] = True
    return e


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(D + '/v7_pairs.jsonl')]
    print('candidate pairs: %s' % format(len(rows), ','), flush=True)
    ok = nonrep = 0
    drop = collections.Counter()
    out = open(D + '/v7_evidence.NEW.jsonl', 'w')
    with Pool(max(1, cpu_count() - 1)) as p:
        for n, e in enumerate(p.imap_unordered(one, rows, chunksize=8), 1):
            if '_err' in e:
                drop[e['_err']] += 1
            else:
                if e['QA'].get('NOT_REPRODUCIBLE'):
                    nonrep += 1
                out.write(json.dumps(e) + '\n')
                ok += 1
            if n % 1000 == 0:
                print('  %d/%d   kept %d' % (n, len(rows), ok), flush=True)
    out.close()
    print('\nkept %s -> v7_evidence.NEW.jsonl' % format(ok, ','))
    for k, v in drop.most_common():
        print('  dropped  %-62s %s' % (k, format(v, ',')))
    print('  %-71s %s' % ('reconciles to input', format(ok + sum(drop.values()), ',')))
    print('\nnot reproducible across two builds: %d  (%s)'
          % (nonrep, 'MUST BE ZERO' if nonrep else 'zero, as required'))

    # REPRODUCIBILITY IS NOT CORRECTNESS. A build that raised UnboundLocalError on 6,933 of 8,359
    # pairs kept 189 and still printed "not reproducible: 0" -- because both runs failed in exactly
    # the same place, so they agreed perfectly. The check proves determinism, nothing more. These
    # two assertions are the ones that would have caught it.
    EXPECT_MIN = 6000
    exc = sum(v for k, v in drop.items() if k.startswith('exception:'))
    if exc:
        print('\n*** %s PAIRS DIED OF AN EXCEPTION. That is a code fault, not a data drop: %s'
              % (format(exc, ','),
                 ', '.join('%s %s' % (k, format(v, ',')) for k, v in drop.most_common()
                           if k.startswith('exception:'))))
    if ok < EXPECT_MIN or exc:
        print('*** BUILD REJECTED: kept %s (floor %s), %s lost to exceptions. Do NOT run clean_v7 '
              'against this file.' % (format(ok, ','), format(EXPECT_MIN, ','), format(exc, ',')))
        sys.exit(1)
