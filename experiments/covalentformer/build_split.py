"""ROLE-STRATIFIED, LEAKAGE-FREE TRAIN/HELD-OUT SPLIT for Phase A.

TWO DEFECTS THIS FIXES.

(1) NO BUILDER EXISTED. valid_clean.csv -- the file every Phase A headline is measured on -- was
    produced by an ad-hoc script that is not in the repo. The set could not be rebuilt, audited, or
    re-derived with a different seed. A measurement whose evaluation set cannot be regenerated from
    source is not reproducible, whatever its p-value.

(2) THE OLD SPLIT COULD NOT POWER THREE OF THE FOUR ROLES. The original split was on the retained
    half and was then filtered for leakage post hoc. Leakage survival is strongly role-dependent:

        role              valid   valid_clean   survival
        EDIT_WARHEAD      14997       3421        22.8%
        EDIT_DECORATION    7580        791        10.4%
        EDIT_SCAFFOLD      3678        466        12.7%
        EDIT_LINKER         352         13         3.7%

    13 clean LINKER rows are not a small sample, they are the ENTIRE clean supply -- rerunning
    cannot produce a 14th. Any per-role statistic on LINKER is unreportable, and the reciprocal
    steering test (hold TRUE=LINKER, request DECORATION) cannot be run at all. Post-hoc filtering
    of a role-blind split is the cause: the filter removes whatever it removes, and nothing
    constrains what is left.

WHY GROUPING BY `keep` IS NOT ENOUGH. One molecule cut at several BRICS bonds yields several rows
with DIFFERENT retained halves but the SAME anchor or target. Splitting on `keep` alone therefore
leaves the same molecule on both sides under a different cut -- which is exactly how valid.csv ended
up 82.4% anchor-leaked while looking like a clean split. Rows are grouped here by CONNECTED
COMPONENT under "shares an anchor, a target, or a retained half" (union-find), and whole components
are assigned to one side. Disjointness on all three keys is then true by construction rather than
being filtered for afterwards, and is asserted at the end regardless.

*** THIS SPLIT IS ONLY VALID FOR A MODEL TRAINED ON train_strat.csv. ***
It re-splits the POOLED corpus, so rows that sat in the old train.csv can land in valid_strat. Scored
against the existing ckpt_A_role -- which was trained on the OLD train.csv -- valid_strat is
99.7% anchor-leaked, 90.2% target-leaked, and exactly 8 of its 3,169 rows are legal. Measured, not
assumed. Using it on a checkpoint trained on the old split would produce a spectacular and entirely
fake result. Retrain on train_strat.csv or do not use this file.

STRATIFICATION IS BEST-EFFORT AND REPORTED AS SUCH. Components are indivisible and many are
role-mixed, so an exact per-role quota is not generally achievable. Components are selected by
whichever role is furthest from its quota, and the achieved counts are printed next to the requested
ones. A role that cannot reach its quota is named explicitly -- it is a property of the corpus, not
a failure to be papered over.
"""
import os, sys, csv, json, random, argparse, collections

ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:            # path compression
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='experiments/covalentformer/data/roles')
    ap.add_argument('--srcs', default='train.csv,valid.csv',
                    help='pooled and re-split; the existing split is NOT trusted')
    ap.add_argument('--per-role', type=int, default=400,
                    help='requested held-out rows per role')
    ap.add_argument('--out-train', default='train_strat.csv')
    ap.add_argument('--out-valid', default='valid_strat.csv')
    ap.add_argument('--seed', type=int, default=20260914)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []
    for s in a.srcs.split(','):
        p = os.path.join(a.data, s.strip())
        if not os.path.exists(p):
            print('  MISSING %s' % p)
            return 2
        n0 = len(rows)
        rows += list(csv.DictReader(open(p)))
        print('  %-16s %7d rows' % (s.strip(), len(rows) - n0))
    print('pooled: %d rows' % len(rows))
    print('  role mix: %s' % dict(collections.Counter(r['role'] for r in rows)))

    # ---- group by connected component over {anchor, target, keep} ----
    d = DSU()
    for i, r in enumerate(rows):
        rid = ('row', i)
        for k in ('anchor', 'target', 'keep'):
            v = r.get(k)
            if v:
                d.union(rid, (k[0], v))
    comp = collections.defaultdict(list)
    for i in range(len(rows)):
        comp[d.find(('row', i))].append(i)
    comps = list(comp.values())
    rng.shuffle(comps)
    sizes = sorted((len(c) for c in comps), reverse=True)
    print('  %d connected components | largest %d rows | median %d'
          % (len(comps), sizes[0], sizes[len(sizes) // 2]))
    if sizes[0] > 0.25 * len(rows):
        print('  WARNING: one component holds %.1f%% of all rows -- it cannot be split, so the '
              'held-out set can never contain it.' % (100.0 * sizes[0] / len(rows)))

    # ---- greedy stratified selection over whole components ----
    quota = {r: a.per_role for r in ROLES}
    have = {r: 0 for r in ROLES}
    val_idx, used = [], set()

    def deficit(role):
        return quota[role] - have[role]

    # index components by which roles they contain, so we can go looking for scarce roles
    by_role = collections.defaultdict(list)
    for ci, c in enumerate(comps):
        for role in {rows[i]['role'] for i in c}:
            by_role[role].append(ci)

    while True:
        need = [r for r in ROLES if deficit(r) > 0 and by_role[r]]
        if not need:
            break
        # serve the role that is furthest from its quota first -- scarce roles would otherwise
        # never be reached, which is precisely how LINKER ended up at 13
        role = max(need, key=lambda r: deficit(r))
        cand = None
        while by_role[role]:
            ci = by_role[role].pop()
            if ci not in used:
                cand = ci
                break
        if cand is None:
            continue
        used.add(cand)
        val_idx += comps[cand]
        for i in comps[cand]:
            have[rows[i]['role']] += 1

    val_set = set(val_idx)
    tr = [rows[i] for i in range(len(rows)) if i not in val_set]
    va = [rows[i] for i in val_idx]

    print()
    print('  %-16s %9s %9s   %s' % ('role', 'requested', 'achieved', 'note'))
    short = []
    for r in ROLES:
        note = ''
        if have[r] < quota[r]:
            note = 'SHORT -- corpus cannot supply it'
            short.append(r)
        print('  %-16s %9d %9d   %s' % (r, quota[r], have[r], note))
    if short:
        print('  roles that could not reach quota: %s' % ', '.join(short))
        print('  this is a corpus property (whole components are indivisible), not a filter bug')

    # ---- disjointness is ASSERTED, not assumed ----
    print()
    ok = True
    for k in ('anchor', 'target', 'keep'):
        t = {r.get(k, '') for r in tr}
        bad = sum(1 for r in va if r.get(k, '') in t)
        print('  held-out rows sharing a %-7s with train: %d  (MUST be 0)' % (k, bad))
        ok = ok and bad == 0
    if not ok:
        print('  FATAL: union-find grouping did not achieve disjointness -- do NOT use this split')
        return 3

    # the limit that survives even a perfect split: fragments are reused across components
    tr_fragto = {r.get('frag_to', '') for r in tr}
    seen = sum(1 for r in va if r.get('frag_to', '') in tr_fragto)
    print('  held-out rows whose frag_to WAS seen in train: %d/%d (%.1f%%)'
          % (seen, len(va), 100.0 * seen / max(len(va), 1)))
    print('  -> generalisation claims are scoped to NOVEL COMBINATION, never novel chemistry')

    for name, data in ((a.out_train, tr), (a.out_valid, va)):
        p = os.path.join(a.data, name)
        with open(p, 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0].keys()))
            w.writeheader()
            w.writerows(data)
        print('  wrote %s (%d rows)' % (p, len(data)))
    json.dump({'seed': a.seed, 'per_role_requested': quota, 'per_role_achieved': have,
               'n_train': len(tr), 'n_valid': len(va), 'short_roles': short,
               'frag_to_seen_in_train_pct': 100.0 * seen / max(len(va), 1)},
              open(os.path.join(a.data, 'split_provenance.json'), 'w'), indent=1)
    print('  wrote split_provenance.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
