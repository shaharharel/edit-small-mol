"""H2 SWEEP: does the model change SHAPE with the gap, or only LENGTH?

WHY H1 IS NOT ENOUGH. The shuffle test proves the decoder reads the geometry channel. It does not
prove it learned anything interesting. A model that memorised "small d -> short linker" would pass
H1 outright, and that mapping is a lookup table: linker length is a free 2D descriptor, so a model
that only learned it has added nothing a SMILES-only baseline could not do with a length prior.

The measured linker distribution says this shortcut is available and cheap: over 139 real covalent
inhibitors the electrophile-to-ring-atom distance is 2-15 bonds, median 4, with 64/139 at exactly 4.
Length alone is nearly a constant -- so a length-only policy is easy to learn and nearly useless.

WHAT ACTUALLY CARRIES THE SIGNAL is shape at fixed length. Measured over 200 conformers each, four-
bond fragments span a 3.2 A range in median reach purely by topology:

    4-bond ortho-phenyl      reach 4.99 A   (rigid, can only reach NEAR)
    4-bond flexible chain    reach 6.21 A
    4-bond trans-cyclohexyl  reach 7.68 A
    4-bond para-phenyl       reach 8.21 A   (rigid, can only reach FAR)

Same bond count. No 2D descriptor distinguishes which one satisfies a 5 A gap. That is the part the
geometry channel must supply, and the only part worth a paper.

THE TEST. Hold the anchor fixed, sweep the required d from 4 to 9 A, and ask two questions:

    Q1 (weak)   does ACHIEVED reach track REQUESTED d?            -> could be pure length
    Q2 (strong) does it still track WITHIN a bond-count stratum?  -> must be shape

Q2 is the result. Q1 passing while Q2 fails means we built a length lookup table and should say so.

Reported as Spearman rho: pooled for Q1, and computed within each bond-count stratum then combined
for Q2. Strata with fewer than MIN_STRATUM generations are dropped and REPORTED as dropped, never
silently, because a stratum that vanishes is exactly where a length-only model hides.
"""
import os, sys, csv, json, time, argparse, collections
import math as _math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from shuffle_test import load_model, sample, recut
from reach_envelope import envelope
from reachability import electrophile_index

D_SWEEP = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
MIN_STRATUM = 12


def bond_count(frag_smi):
    """Bonds from the attachment dummy to the electrophilic carbon. The 'length' axis."""
    m = Chem.MolFromSmiles(frag_smi)
    if m is None:
        return None
    star = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    el, _ = electrophile_index(m)
    if not star or el is None:
        return None
    try:
        return len(Chem.GetShortestPath(m, star[0], el)) - 1
    except Exception:
        return None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--valid',
                    # DEFAULT CHANGED from data/tierA/valid.csv, which is the VOID v1 set:
                    # 99.49% of its anchors have exactly one target, so I(target; g, r | anchor)=0
                    # and any conditioning result scored against it is guaranteed null BY THE DATA.
                    # A run launched without an explicit --valid was silently scoring on a dead
                    # dataset and would have looked like a model failure.
                    default='experiments/covalentformer/data/tierA_v3/valid.csv')
    ap.add_argument('--n-anchor', type=int, default=120)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--theta', type=float, default=60.0)
    ap.add_argument('--out', default='experiments/covalentformer/data/sweep_test.json')
    # MPS is ONE serialising queue on this machine; running an eval on it alongside a training
    # job throttled training from 143 to under 50 steps/min. Evals go on CPU by default here.
    ap.add_argument('--device', default='auto')
    # SEEDED. I wrongly described the sweep as unaffected by the torch-seeding defect "because it
    # does not sample". It does -- it imports sample() from shuffle_test and draws generations with
    # torch.multinomial. Q1 +0.515 and Q2 +0.097 were therefore single unseeded draws, exactly like
    # H1's +4.3 which collapsed to a null (+1.5, n=4 seeds) once seeded. The sweep uses n=2265
    # generations and reports rank correlations rather than a difference of proportions, so it
    # should be far more stable -- but "should be" is not a measurement, which is the whole lesson.
    ap.add_argument('--seed', type=int, default=20260913)
    # THE CONTROL I FAILED TO RUN. rho(requested d, achieved reach) = +0.510 was reported as
    # "Phase A demonstrably works" with NO control. --control feeds the model a CONSTANT d while
    # still LABELLING each row with the sweep value, so the correlation is computed against a
    # requirement the model never received. If rho stays high, the correlation is an artifact of
    # how the sweep is scored, not evidence the model responds to d.
    ap.add_argument('--control', action='store_true',
                    help='feed a constant d regardless of the swept value (null control)')
    a = ap.parse_args()
    # THE SHAPE HEADLINE CROSSES THE CLASSIFIER BOUNDARY AND NEVER SAID SO.
    # bond_count() below calls electrophile_index, and bond_count IS the length
    # variable in both the binned Q2 and the partial rho. A v2 relabel moves which
    # atom is the electrophile, hence the attachment->electrophile path, hence the
    # strata AND the partial's control variable. 4 of the 19 files that call
    # electrophile_index announced this; this was not one of them.
    from reachability import announce_classifier_version
    announce_classifier_version('sweep_test')
    torch.manual_seed(a.seed)
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    model, vocab, tok, ep = load_model(a.ckpt, dev)

    rows = [r for r in csv.DictReader(open(a.valid)) if r['label'] == '1']
    rng = np.random.default_rng(20260913)
    rows = [rows[i] for i in rng.choice(len(rows), min(a.n_anchor, len(rows)), replace=False)]
    anchors = [r['anchor'] for r in rows]
    scaffolds = [r['scaffold'] for r in rows]
    ct = float(np.cos(np.radians(a.theta)))
    # RECORD THE PROVENANCE. Two 3-hour runs were spent on a module version that predated the
    # partial_rho patch, and their output was indistinguishable from a current run: same filename
    # shape, same fields, silently missing the only statistic that matters. Stamp the source file's
    # mtime and the flags into the result so a stale artifact announces itself.
    _src_mtime = time.strftime('%Y-%m-%d %H:%M:%S',
                               time.localtime(os.path.getmtime(os.path.abspath(__file__))))
    print('model %s (epoch %d) | %d anchors | theta %.0f deg | sweep d=%s | control=%s | '
          'sweep_test.py mtime %s'
          % (os.path.basename(a.ckpt), ep, len(rows), a.theta, D_SWEEP, a.control, _src_mtime))

    recs = []
    for d_req in D_SWEEP:
        gens = []
        for i in range(0, len(rows), 50):
            d_fed = 6.5 if a.control else d_req
            gens += sample(model, vocab, tok, anchors[i:i + 50],
                           [[d_fed, ct, 1.0]] * len(anchors[i:i + 50]),
                           [1] * len(anchors[i:i + 50]), dev, temp=a.temp)
        ok = 0
        for s, sc in zip(gens, scaffolds):
            if not s:
                continue
            frag, err = recut(s, sc)
            if frag is None:
                continue
            nb = bond_count(frag)
            e = envelope(frag, n_conf=60)
            if nb is None or 'err' in e:
                continue
            recs.append(dict(d_req=d_req, bonds=nb,
                             reach_med=float(np.median(e['d'])),
                             reach_max=float(np.max(e['d'])), frag=frag))
            ok += 1
        print('  d_req %.1f A -> %d/%d scorable' % (d_req, ok, len(gens)), flush=True)

    if len(recs) < 40:
        print('\nINCONCLUSIVE: only %d scorable generations across the whole sweep.' % len(recs))
        # Same key discipline as the main dump below: nothing in this file writes a key called
        # 'verdict', so a consumer cannot read one by default and get an unqualified answer.
        json.dump({'n': len(recs), 'binned_verdict_NOT_AUTHORITATIVE': 'inconclusive',
                   'partial_rho_shape': None}, open(a.out, 'w'), indent=1)
        return 1

    dq = [r['d_req'] for r in recs]
    print('\nQ1 (WEAK -- could be a pure length lookup table)')
    print('   requested d  vs  achieved median reach : rho = %+.3f  (n=%d)'
          % (spearman(dq, [r['reach_med'] for r in recs]), len(recs)))
    print('   requested d  vs  bond count            : rho = %+.3f'
          % spearman(dq, [r['bonds'] for r in recs]))

    print('\nQ2 (STRONG -- shape at FIXED length; this is the result)')
    # BINNED, NOT EXACT, BOND COUNTS. Generations spread over 3-19 bonds, so exact-count strata
    # leave every bucket under n=12 and the whole test returns inconclusive regardless of what the
    # model learned. Bins keep length nearly fixed while giving usable n: within a 3-4 bond bin the
    # reach spread attributable to SHAPE alone is >3 A (ortho-phenyl 4.99 A vs para-phenyl 8.21 A at
    # the same 4 bonds), which is far larger than the sub-bond length variation the bin admits.
    # Reported as bins, never as "fixed bond count", because they are not.
    def bin_of(nb):
        return ('3-4' if nb <= 4 else '5-6' if nb <= 6 else '7-9' if nb <= 9 else
                '10-13' if nb <= 13 else '14+')

    by = collections.defaultdict(list)
    for r in recs:
        by[bin_of(r['bonds'])].append(r)
    rhos, dropped, tot = [], [], 0
    for nb in sorted(by):
        grp = by[nb]
        if len(grp) < MIN_STRATUM:
            dropped.append((nb, len(grp)))
            continue
        rho = spearman([g['d_req'] for g in grp], [g['reach_med'] for g in grp])
        rhos.append((nb, len(grp), rho))
        tot += len(grp)
        print('   %-6s bonds  n=%4d   rho(requested d, achieved reach) = %+.3f' % (nb, len(grp), rho))
    if dropped:
        print('   DROPPED strata (n < %d): %s' % (MIN_STRATUM, dropped))
    # HOISTED OUT OF THE else:. This constant was assigned only inside the "some stratum survived"
    # branch but referenced UNCONDITIONALLY in the final json.dump, so the `not rhos` path died with
    # UnboundLocalError AFTER doing every expensive thing -- full sweep, all generations, all
    # envelopes -- and AFTER printing the partial rho this file calls "the shape statistic", but
    # BEFORE writing it. Printed only, then gone. The trigger window is 40-55 scorable generations:
    # above the n>=40 early exit, below 5 bins x MIN_STRATUM=12. It sits exactly BETWEEN the two
    # guards. Worse, it left a HALF-ARTIFACT -- _recs.json written, summary absent -- which to a
    # glob looks like a completed run, and a stale summary from a previous checkpoint would survive
    # beside it with no marker that the two belong to different runs.
    _LENGTH_MATCHED_NULL_AT_MODEL_RHO = (0.0971, 0.1495)
    if not rhos:
        print('\n   NO STRATUM SURVIVED. Cannot separate shape from length at this n.')
        verdict = 'inconclusive'
        pooled = float('nan')
    else:
        pooled = float(np.average([r for _n, _c, r in rhos],
                                  weights=[c for _n, c, _r in rhos]))
        print('\n   n-weighted mean within-stratum rho = %+.3f  over %d generations' % (pooled, tot))
        # THE OLD PASS BAR SAT BELOW ITS OWN NULL, and the sentence under it denied the very
        # hypothesis that passes. `pooled >= 0.20` printed "not a lookup table" -- while a
        # shape-blind pure-length lookup policy scores +0.2917 on this same binned metric (measured;
        # see the comment below). The null does not merely reach the bar, it CLEARS it. So the
        # statistic awarded "not a lookup table" to a lookup table. No threshold on the binned
        # metric can be rescued, because the null outscores the model on it.
        # The binned verdict is retained ONLY as a descriptive label and is explicitly marked
        # non-authoritative here and in the JSON. The shape question is settled by PARTIAL rho below.
        # 0.2917 WAS NOT DERIVED AND IT IS NOT THE RIGHT NULL. It was hardcoded here; no script on
        # disk produces it and no artifact records the policy that did. Derived from the fragment
        # population, the binned metric turns out to be MONOTONE in the null policy's own length
        # response (tau sweep: rho(d,bonds) 0.97 -> null 0.396; 0.82 -> 0.275; 0.51 -> 0.121;
        # 0.07 -> 0.009). A null is therefore only meaningful LENGTH-MATCHED to the model it judges.
        # At the model's measured rho(d,bonds) = +0.4992 the shape-blind null is +0.0992..+0.1198
        # across three pool constructions, seeded, 200 replicates each. Reaching 0.2917 needs a
        # policy at rho(d,bonds) = +0.8394 -- 1.68x as length-obedient as the model -- so the old
        # constant compared the model against a strictly stronger policy and read the difference as
        # failure. A hard (deterministic) lookup cannot produce it at all: b* collapses two bins to a
        # single d level, spearman returns nan and the pooled value is NaN.
        LOOKUP_NULL_BINNED = None  # must be supplied length-matched; see nulls/README.md
        # THE INTERVAL SPANS POLICY FAMILY, NOT JUST POOL AND n -- and that is why it is this wide.
        # My first replacement for 0.2917 was (0.0992, 0.1198), derived by sweeping POOL and n. Both
        # are DATASET knobs. The POLICY -- what "shape-blind length-matched lookup" means as a
        # sampler -- was fixed at one implementation and never perturbed, so a dataset-sensitivity
        # interval was being read as a total-uncertainty interval. An independent implementation of
        # the same six words (Laplacian rather than Gaussian kernel, bond counts weighted by natural
        # abundance rather than by kernel alone), calibrated to the same rho(d,bonds)=+0.4992,
        # returns +0.0971 / +0.1153 / +0.1495. On the SAME envelopes_20k pool the two implementations
        # disagree by 0.05 -- more than twice the model's margin over the narrow range. So the
        # verdict was a property of whose null you use, not of the model.
        # Note also that the narrow range (0.0206 wide) was NARROWER THAN ONE REPLICATE'S SD (~0.021).
        _LENGTH_MATCHED_NULL_AT_MODEL_RHO = (0.0971, 0.1495)  # union of two implementations
        lo, hi = _LENGTH_MATCHED_NULL_AT_MODEL_RHO
        # NaN MUST NOT FALL THROUGH TO 'within-null'. Every comparison against NaN is False, so a
        # degenerate pooled rho would take the else-branch and report "INDISTINGUISHABLE from the
        # shape-blind null" -- the headline conclusion of this whole test -- when the truth is "not
        # computable". spearman() returns nan when a stratum has <2 distinct x or y, and MIN_STRATUM
        # does NOT prevent that: a bond bin populated at a single d_req level is enough, which is
        # plausible for '14+' since it only fills when long reach is requested. np.average then
        # propagates the nan to `pooled`. The 'not rhos' branch already says 'inconclusive'; this
        # path silently said the opposite.
        # json.dump also emits a BARE NaN token, which Python accepts and strict JSON parsers reject,
        # so the artifact would be unreadable to anything but Python.
        if pooled is None or _math.isnan(pooled):
            verdict = 'inconclusive_nan_stratum'
            print('   BINNED VERDICT INCONCLUSIVE: pooled rho is NaN, which means at least one '
                  'stratum had <2 distinct values and spearman() could not be computed. This is '
                  'NOT "indistinguishable from the null" -- it is not measured at all.')
            pooled = None   # keep the artifact valid JSON rather than writing a bare NaN token
        else:
            verdict = ('above-null' if pooled > hi
                       else 'below-null' if pooled < lo else 'within-null')
        if pooled is not None:
            print('   binned pooled rho %+.3f vs the LENGTH-MATCHED shape-blind null %+.4f..%+.4f -- %s'
                  % (pooled, lo, hi,
                     'ABOVE it' if pooled > hi else 'BELOW it' if pooled < lo else 'INSIDE it'))
        print('   CAP: only 5.8%% of reach variance is WITHIN bond count (rho(bonds,reach)=+0.9665),')
        print('   so a small value here is largely the ceiling, not necessarily a model failure.')
        print('   THIS IS NOT A VERDICT ON SHAPE. The binned metric cannot separate shape from')
        print('   length in either direction; see PARTIAL rho below, which is the shape statistic.')
    # PARTIAL CORRELATION, controlling for bond count CONTINUOUSLY rather than by wide bins.
    # The binned Q2 does not hold length fixed: rho(bonds, reach) INSIDE the bins runs +0.53 to
    # +0.77, and the 7-9 bin that carries the whole result is among the leakiest. A shape-blind
    # pure-length lookup policy scores +0.2917 on the binned metric -- higher than the model and
    # higher than the "ceiling" I quoted -- so the binned statistic cannot separate shape from
    # length in either direction. Partial rho removes the linear length component instead of
    # hoping a bin does it.
    def _resid(y, x):
        y, x = np.asarray(y, float), np.asarray(x, float)
        if len(set(x.tolist())) < 2:
            return y - y.mean()
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)
    bonds_v = [r['bonds'] for r in recs]
    reach_v = [r['reach_med'] for r in recs]
    partial = spearman(_resid(dq, bonds_v), _resid(reach_v, bonds_v))
    print('\n   PARTIAL rho(requested d, achieved reach | bond count) = %+.3f' % partial)
    print('   This is the shape statistic. The binned Q2 above is NOT -- a shape-blind length')
    print('   lookup table scores +0.292 on it.')
    # PERSIST THE RECORDS. sweep_test never dumped `recs`, so the one number that settles the
    # shape question was unrecoverable from any artifact after the fact.
    json.dump(recs, open(a.out.replace('.json', '_recs.json'), 'w'))
    # SANITISE EVERY rho FIELD, NOT JUST THE TWO I NOTICED. The first pass guarded
    # within_stratum and pooled_within_rho -- both labelled NOT_AUTHORITATIVE -- and left
    # partial_rho_shape unguarded, which is the field this file's own print calls "the shape
    # statistic". A NaN there serialises as bare `NaN`, which is invalid JSON: strict parsers
    # reject the file and Python's json accepts it silently, so the authoritative number is the
    # one that fails least visibly. Guarding the decorations and not the headline is the same
    # half-fix as hoisting a constant while leaving its use behind a branch.
    def _nn(x):
        return None if (x is None or _math.isnan(x)) else x
    json.dump({'partial_rho_shape': _nn(partial), 'control': bool(a.control),
               'seed': a.seed, 'ckpt': a.ckpt, 'valid': a.valid, 'src_mtime': _src_mtime,
               'q1_rho_reach': _nn(spearman(dq, [r['reach_med'] for r in recs])),
               'q1_rho_bonds': _nn(spearman(dq, [r['bonds'] for r in recs])),
               'within_stratum': [{'bond_bin': n, 'n': c,
                                   'rho': (None if (r is None or _math.isnan(r)) else r)}
                                  for n, c, r in rhos],
               # SANITISE AT THE CONVERGENCE POINT, NOT IN ONE BRANCH. My first attempt put the
               # NaN->None conversion inside the `else:` and hoisted only the null CONSTANT out, so
               # the `not rhos` path -- which sets pooled=float('nan') at :208 -- could reach the
               # constant but never the sanitiser, and still wrote a bare NaN token. I also added
               # `import math as _math` at module scope while leaving a local one inside main(),
               # which symtable confirms made _math LOCAL and the hoist INERT. Two half-fixes that
               # together looked like a fix. Both paths now converge here, so one conversion covers
               # both, and there is no branch left for a future edit to miss.
               'dropped_strata': dropped,
               'pooled_within_rho': (None if (pooled is None or _math.isnan(pooled)) else pooled),
               # KEY RENAMED DELIBERATELY. It used to be 'verdict', so any downstream consumer
               # reading the obvious key got the binned PASS/WEAK/FAIL that this file's own comment
               # says cannot separate shape from length. There is now no key called 'verdict':
               # a consumer must choose one of these two and cannot pick the wrong one by default.
               # SANITISE within_stratum TOO. I converted `pooled` to None so the artifact would
               # not carry a bare NaN token -- and left the IDENTICAL NaN in the per-stratum list
               # one field over, which is the natural thing for a consumer to re-average. A reader
               # who sanity-checks pooled_within_rho sees a clean "inconclusive" and has no reason
               # to distrust the list beside it. Fixed the wrong number and left the same wrong
               # number one field away, which is the pattern this file already documents twice.
               'binned_verdict_NOT_AUTHORITATIVE': verdict,
               # The artifact must not carry the undrivable 0.2917 either. Record the LENGTH-MATCHED
               # null as a RANGE, together with the model rho it is matched at, so a reader can see
               # that the comparison is conditional on a length response and not an absolute bar.
               'binned_lookup_null_length_matched': list(_LENGTH_MATCHED_NULL_AT_MODEL_RHO),
               'binned_lookup_null_matched_at_rho_d_bonds': 0.4992,
               'binned_null_note': ('the binned metric is monotone in the null policy length '
                                    'response; an unmatched null is not interpretable'),
               # No PASS threshold is asserted on partial rho. I do not have a pre-registered bar
               # for it, and inventing one here would be hardcoding a conclusion into the artifact
               # -- the exact pattern that produced the defect being fixed above.
               'n': len(recs), 'theta': a.theta},
              open(a.out, 'w'), indent=1)
    print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
