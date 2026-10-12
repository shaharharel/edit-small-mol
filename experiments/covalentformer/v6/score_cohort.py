#!/usr/bin/env python3
"""Summarise ten independent medicinal-chemist reads of the same label packet.

The decision this feeds is binary: scale to the full corpus, or run another round of fixes. So the
headline number is not the mean score -- it is how many pairs a MAJORITY of chemists call WRONG,
because those are the labels that would teach a model something false.

Inter-rater agreement matters as much as the score. If the ten chemists disagree wildly, the
verdict distribution is not measuring our labels, it is measuring taste, and a further round of
"fixes" aimed at it would be chasing noise. Fleiss' kappa is reported for that reason.
"""
import collections, glob, json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
CATS = ['GOOD', 'WEAK', 'WRONG']

reads = {}
for p in sorted(glob.glob(os.path.join(SP, 'cohort_out', 'chemist_*.json'))):
    name = os.path.basename(p)[:-5]
    try:
        reads[name] = json.load(open(p))
    except Exception as e:
        print('UNPARSEABLE %s: %s' % (name, e))

if not reads:
    sys.exit('no cohort_out/chemist_*.json found')

meta = json.load(open(os.path.join(SP, 'cohort_pairs.json')))
pairs = meta['pairs']
print('chemists reporting   %d' % len(reads))
print('pairs in packet      %d' % len(pairs))

def verdict(rec):
    v = (rec.get('verdict') if isinstance(rec, dict) else rec) or ''
    v = str(v).strip().upper()
    for c in CATS:
        if v.startswith(c):
            return c
    return None

# ---- per-chemist totals: a chemist who marks everything GOOD is not reviewing, and one who marks
# everything WRONG is applying a different standard than the brief. Both are visible here.
print('\nper chemist')
print('  %-14s %5s %5s %5s %6s' % ('', 'GOOD', 'WEAK', 'WRONG', 'given'))
for name in sorted(reads):
    c = collections.Counter(verdict(reads[name].get(p)) for p in pairs)
    print('  %-14s %5d %5d %5d %6d'
          % (name, c['GOOD'], c['WEAK'], c['WRONG'], sum(c[k] for k in CATS)))

# ---- per-pair distribution
rows = []
for p in pairs:
    c = collections.Counter()
    for name in reads:
        v = verdict(reads[name].get(p))
        if v:
            c[v] += 1
    n = sum(c[k] for k in CATS)
    rows.append((p, c, n))

maj_wrong = [(p, c, n) for p, c, n in rows if n and c['WRONG'] * 2 > n]
any_wrong = [(p, c, n) for p, c, n in rows if c['WRONG']]
unan_good = [(p, c, n) for p, c, n in rows if n and c['GOOD'] == n]
split = [(p, c, n) for p, c, n in rows if n and max(c[k] for k in CATS) * 2 <= n]

tot = collections.Counter()
for _p, c, _n in rows:
    tot.update(c)
N = sum(tot[k] for k in CATS) or 1
print('\nall %d verdicts pooled' % N)
for k in CATS:
    print('  %-6s %5d  %5.1f%%' % (k, tot[k], 100.0 * tot[k] / N))

print('\nby pair (of %d)' % len(rows))
print('  unanimously GOOD                 %3d' % len(unan_good))
print('  at least one WRONG               %3d' % len(any_wrong))
print('  MAJORITY WRONG  <- the defects   %3d' % len(maj_wrong))
print('  no majority view (ambiguous)     %3d' % len(split))

# ---- Fleiss' kappa over the three categories
def fleiss(table):
    table = [r for r in table if sum(r) > 1]
    if not table:
        return float('nan')
    n = sum(table[0])
    if any(sum(r) != n for r in table):          # unequal raters per item: kappa undefined
        return float('nan')
    N_ = len(table)
    p = [sum(r[j] for r in table) / float(N_ * n) for j in range(len(CATS))]
    Pi = [(sum(x * x for x in r) - n) / float(n * (n - 1)) for r in table]
    Pbar = sum(Pi) / N_
    Pe = sum(x * x for x in p)
    return (Pbar - Pe) / (1 - Pe) if Pe != 1 else float('nan')

k = fleiss([[c[j] for j in CATS] for _p, c, _n in rows])
print('\nFleiss kappa  %.3f   (<0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, >0.6 substantial)' % k)

print('\n' + '=' * 96)
print('MAJORITY-WRONG PAIRS -- each of these is a label that would teach a model something false')
print('=' * 96)
for p, c, n in sorted(maj_wrong, key=lambda t: -t[1]['WRONG']):
    print('\n%-26s  GOOD %d / WEAK %d / WRONG %d' % (p, c['GOOD'], c['WEAK'], c['WRONG']))
    print('   "%s"' % meta['sentences'].get(p, '?'))
    for name in sorted(reads):
        r = reads[name].get(p)
        if isinstance(r, dict) and verdict(r) == 'WRONG':
            why = (r.get('why') or r.get('reason') or '').strip().replace('\n', ' ')
            print('   %-12s %s' % (name, why[:150]))

print('\n' + '=' * 96)
print('AMBIGUOUS PAIRS -- no majority; these measure taste, not correctness')
print('=' * 96)
for p, c, n in split:
    print('%-26s  GOOD %d / WEAK %d / WRONG %d' % (p, c['GOOD'], c['WEAK'], c['WRONG']))

with open(os.path.join(SP, 'cohort_summary.json'), 'w') as f:
    json.dump({'chemists': sorted(reads), 'n_pairs': len(rows), 'pooled': dict(tot),
               'kappa': k,
               'majority_wrong': [{'pair': p, 'dist': dict(c)} for p, c, _n in maj_wrong],
               'ambiguous': [{'pair': p, 'dist': dict(c)} for p, c, _n in split],
               'unanimous_good': [p for p, _c, _n in unan_good]}, f, indent=1)
print('\nwrote cohort_summary.json')
