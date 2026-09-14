"""Re-score run 2 and run 3 under the v3 filter (#162's pre-registered test).

Answers ONE question: does the Phase A -> Phase B transfer effect survive removing the ~28% of
held-out rows whose `regrow` fragment the model has already seen as a training target? H1 says the
leak is symmetric across arms and cancels in the paired difference; H2 says it does not.

RE-SCORING, NOT RETRAINING: de_leak_holdout touches only va_rows.
"""
import sys, os, json, statistics as st
CF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CF, 'nulls'))
from replay import replay, load_model, eval_loss

SEEDS = (404, 505, 606)
RUNS = {'run2': ('ckpt_Bconv_from_a_s%d', 'ckpt_Bconv_scratch_s%d'),
        'run3': ('ckpt_B4c_from_a_s%d',   'ckpt_B4c_scratch_s%d')}
_c = {}
def split(seed, v):
    if (seed, v) not in _c:
        _c[(seed, v)] = replay(seed, deleak_version=v)
    return _c[(seed, v)]

def score(path, seed, v):
    d = split(seed, v)
    m, ck, vocab, tok = load_model(os.path.join(CF, path))
    if ck.get('gzscore'):
        raise SystemExit('FATAL %s is z-scored' % path)
    return eval_loss(m, d['va'], vocab, tok, gstats=None)

out = {}
for v in (2, 3):
    print('=== de-leak v%d  (n held-out: %s) ===' % (v, {s: len(split(s, v)['va']) for s in SEEDS}))
    for label, (fa, sc) in RUNS.items():
        ds = []
        for s in SEEDS:
            a = score((sc % s) + '/ep0.ckpt', s, v)
            b = score((fa % s) + '/ep0.ckpt', s, v)
            ds.append(a - b)
            print('  %s s%d  scratch %.6f  from_a %.6f  delta %+.6f' % (label, s, a, b, a - b))
        out['%s_v%d' % (label, v)] = ds
        print('  %s v%d mean %+.6f  all positive %s\n' % (label, v, st.mean(ds), all(x > 0 for x in ds)))

print('=== #162 VERDICT ===')
for label in RUNS:
    d2, d3 = out['%s_v2' % label], out['%s_v3' % label]
    print('  %s  v2 %+.6f -> v3 %+.6f   change %+.6f   per-seed v3 %s'
          % (label, st.mean(d2), st.mean(d3), st.mean(d3) - st.mean(d2),
             ['%+.4f' % x for x in d3]))
json.dump(out, open(os.path.join(CF, 'results', 'rescore_runs_v3.json'), 'w'), indent=1)
print('\nwritten -> results/rescore_runs_v3.json')
