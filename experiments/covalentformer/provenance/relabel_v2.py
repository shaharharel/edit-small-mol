"""READ-ONLY: re-label valid_strat.csv rows with the CURRENT (v2) electrophile classifier.

Stored `role` column was written by v1 (reachability.CORPUS_LABELLED_WITH == 1).
classify() in build_roles.py is unchanged; only electrophile_index() changed (v1 -> v2).
So re-running classify() today = the v2 label.
"""
import sys, csv, collections, json
sys.path.insert(0, '/Users/shaharharel/Documents/github/edit-small-mol')
sys.path.insert(0, '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from build_roles import classify
from reachability import electrophile_index, CLASSIFIER_VERSION, CORPUS_LABELLED_WITH

SRC = '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data/roles/valid_strat.csv'
print('reachability CLASSIFIER_VERSION=%s  CORPUS_LABELLED_WITH=%s'
      % (CLASSIFIER_VERSION, CORPUS_LABELLED_WITH))

rows = list(csv.DictReader(open(SRC)))
print('rows: %d' % len(rows))

cache = {}
def v2role(smi):
    if smi in cache:
        return cache[smi]
    m = Chem.MolFromSmiles(smi)
    r = None if m is None else classify(m, set(), set())
    cache[smi] = r
    return r

conf_to = collections.Counter()
conf_from = collections.Counter()
bad = 0
out = []
for r in rows:
    old = r['role']
    nt = v2role(r['frag_to'])
    nf = v2role(r['frag_from'])
    if nt is None or nf is None:
        bad += 1
        continue
    conf_to[(old, nt)] += 1
    conf_from[(old, nf)] += 1
    out.append(dict(anchor=r['anchor'], keep=r['keep'], old=old, v2_to=nt, v2_from=nf))
print('unparseable fragments: %d' % bad)

ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']
for label, conf in (('frag_to (the GENERATED half)', conf_to),
                    ('frag_from (the anchor half)', conf_from)):
    print('\n=== v1 stored role -> v2 role, classified on %s ===' % label)
    print('  %-16s %8s %8s  %s' % ('old_role', 'n', 'changed', 'breakdown'))
    tot = chg = 0
    for o in ROLES:
        n = sum(v for (oo, nn), v in conf.items() if oo == o)
        if not n:
            continue
        c = sum(v for (oo, nn), v in conf.items() if oo == o and nn != o)
        bd = ', '.join('%s:%d' % (nn, v) for (oo, nn), v in sorted(conf.items())
                       if oo == o and nn != o) or '-'
        print('  %-16s %8d %8d  %s' % (o, n, c, bd))
        tot += n; chg += c
    print('  %-16s %8d %8d  (%.2f%%)' % ('ALL', tot, chg, 100.0 * chg / max(tot, 1)))

json.dump(out, open('/tmp/qa29_t4/v2_labels.json', 'w'))
print('\nWROTE /tmp/qa29_t4/v2_labels.json')
