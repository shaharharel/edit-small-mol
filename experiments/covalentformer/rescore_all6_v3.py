"""#123's headline (6/6 seeds positive, exact two-sided p=0.0312) recomputed under the v3 filter.

#123 is a v1 number. #163 showed the transfer effect GROWS as the filter tightens, so the mean is
expected to rise -- but the HEADLINE is not the mean, it is the sign count and the exact p, and
those can move either way: v3 removes 8-44% of held-out rows per seed, and a seed with few rows
left could flip. The test can therefore still fail, which is the only reason to run it.

Reports BOTH selection rules, because QA showed #134's 6/6 was argmin-only and flipped to 5/6 at
fixed ep0 -- a result quoted as "6/6, p=0.0312" that is 5/6, NS under the other equally defensible
rule. Any 6/6 claim on this project must now name its rule.
"""
import sys, os, json, itertools, statistics as st
CF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CF, 'nulls'))
from replay import replay, load_model, eval_loss

ARMS = {101: ('ckpt_Bconv_from_a_s101', 'ckpt_Bconv_scratch_s101', 12),
        202: ('ckpt_Bconv_from_a_s202', 'ckpt_Bconv_scratch_s202', 12),
        303: ('ckpt_Bconv_from_a_s303', 'ckpt_Bconv_scratch_s303', 12),
        404: ('ckpt_Bconv_from_a_s404', 'ckpt_Bconv_scratch_s404', 3),
        505: ('ckpt_Bconv_from_a_s505', 'ckpt_Bconv_scratch_s505', 3),
        606: ('ckpt_Bconv_from_a_s606', 'ckpt_Bconv_scratch_s606', 3)}
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

def exact_two_sided_sign_p(deltas):
    """Exact sign-flip permutation on the MEAN, two-sided. n=6 floor is 2/64 = 0.03125."""
    n = len(deltas); obs = abs(st.mean(deltas)); hit = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(st.mean([s * d for s, d in zip(signs, deltas)])) >= obs - 1e-12:
            hit += 1
    return hit / 2 ** n

out = {}
for v in (1, 3):
    print('=== de-leak v%d ===' % v)
    ep0, arg = [], []
    for seed in sorted(ARMS):
        fa, sc, neps = ARMS[seed]
        cs = [score('%s/ep%d.ckpt' % (sc, e), seed, v) for e in range(neps)]
        cf = [score('%s/ep%d.ckpt' % (fa, e), seed, v) for e in range(neps)]
        ep0.append(cs[0] - cf[0]); arg.append(min(cs) - min(cf))
        print('  s%-4d n=%-4d ep0 %.6f-%.6f=%+.6f | argmin(ep%d) %.6f-(ep%d) %.6f=%+.6f'
              % (seed, len(split(seed, v)['va']), cs[0], cf[0], cs[0] - cf[0],
                 cs.index(min(cs)), min(cs), cf.index(min(cf)), min(cf), min(cs) - min(cf)))
    for name, ds in (('ep0', ep0), ('argmin', arg)):
        p = exact_two_sided_sign_p(ds)
        print('  %-7s %d/6 positive  mean %+.6f  sd %.6f  exact two-sided p = %.5f'
              % (name, sum(1 for x in ds if x > 0), st.mean(ds), st.stdev(ds), p))
        out['v%d_%s' % (v, name)] = dict(deltas=ds, n_pos=sum(1 for x in ds if x > 0),
                                         mean=st.mean(ds), p=p)
    print()
json.dump(out, open(os.path.join(CF, 'results', 'rescore_all6_v3.json'), 'w'), indent=1)
print('written -> results/rescore_all6_v3.json')
