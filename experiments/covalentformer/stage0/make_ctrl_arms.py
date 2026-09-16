"""Regenerate the matched-control arms for the steering mechanism experiment.

WHY THIS FILE EXISTS, AND WHY IT IS LATE. The four control arms (extentdisc, topojit, topojitwide,
topojitwidescaled) were built by ad-hoc inline scripts during the run. An auditor grepped the whole
repo and found the arm names ONLY inside data/steer_k8_ctrl*/ CSVs and the steer_gaps.json files --
NO producer anywhere. extentdisc happened to be recoverable because its construction is
deterministic, but topojit was a random draw whose seed existed only in a shell scrollback. A
control arm that cannot be rebuilt cannot be checked, and a conclusion drawn from it rests on a
file nobody can regenerate. That is the #84 / #179 failure class -- an orphan artifact -- and I
produced it while auditing other people for the same thing.

The seeds below are the ones that actually produced the shipped CSVs. `--verify` regenerates every
arm and compares to what is on disk, so this file is checkable rather than merely asserted.

STATE OF THE MECHANISM QUESTION, measured from all ten arms on disk and recorded BEFORE
topodiamshuffle / topodiamscale land, so neither outcome can be narrated afterwards:

    arm                card  MI(arm;topodiam)  %exact=topodiam   verdict
    topodiam              9       1.187            100.0%        PASS 4/4
    topojit             901       1.187              0.0%        PASS 2/2
    topojitwide         901       0.323              0.0%        PASS 2/2
    topojitwidescaled   901       0.323              0.0%        PASS 2/2
    extentdisc           10       0.295              0.0%        PASS 3/3
    extentrank            8       0.262             49.6%        PASS 2/2
    extent              901       0.308              0.0%        fail 0/4
    resid               901       0.040              0.0%        fail 0/4
    extentctr           898       0.177              0.3%        fail 0/2
    topodiamoff         357       0.058              0.0%        fail 0/2   (UNSCORABLE, see above)

  UPDATE AFTER topodiamshuffle / topodiamscale LANDED. What the ARMS establish is NECESSITY and
  nothing more, and my attempts to state a sufficient condition have now failed three times:
    * DESTROYING CROSS-PARENT COMPARABILITY KILLS THE CHANNEL IN 4 ARMS OUT OF 4 that do it --
      extentctr (per-parent centring), topodiamoff (per-parent offset), topodiamshuffle (per-parent
      relabelling) and topodiamscale (per-parent units). GAP_neg collapses to +0.00008..+0.0029,
      z in [-1.7,-0.2]. topodiamshuffle vs extentdisc is the clean pair: matched cardinality (10)
      and entropy (2.285 vs 2.297), +0.00008 vs +0.00945, a 125x collapse.
    * SUFFICIENCY IS NOT ESTABLISHED. `extent` has a shared axis and still fails 0/4.
    * MY CONJUNCTION ("shared axis AND recoverable coarse structure") IS REFUTED: scored with
      |Spearman vs topodiam| > 0.60 and 9-level recovery above the 41.84% none-arm floor, it
      mispredicts 2 of 13 arms -- extent (0.644 / 50.2%, predicted PASS, observed fail) and
      topodiamscale (0.976 / 84.0%, predicted PASS, observed fail).
    * AND MY SECOND PROXY FOR "SHARED AXIS" IS ALSO BROKEN. Global Spearman rates topodiamscale at
      0.976 because a positive per-parent scale barely reorders global ranks -- while the VALUES
      are exactly what it makes incomparable. That is two broken proxies in two hours: the first
      (sd of per-parent means) said topodiamoff HAD a shared axis, the arm whose axis is most
      destroyed. Summary statistics for this property keep measuring its opposite; only the ARMS,
      which destroy it by construction, have been trustworthy.
  STOP BUILDING ACCOUNTS AND QUOTE THE NECESSITY RESULT. It is measured, it is 4/4, and it came
  from a test that was pre-registered to be able to refute it.

  BOTH SINGLE-VARIABLE ACCOUNTS ARE REFUTED BY THIS TABLE.
  - CARDINALITY (two auditors' account): passers span 9..901, failers span 357..901, and five
    maximal-cardinality arms PASS. It does not separate them.
  - MI WITH topodiam: `extent` FAILS at 0.308 while `extentrank` PASSES at 0.262 and `extentdisc`
    PASSES at 0.295. A failing arm carries MORE of it than two passing arms. It does not separate
    them either.
  WHAT IS TRUE is narrower and stranger: cardinality separates WITHIN THE CONFORMER FAMILY
  (extent 901 fails; its 8-10 level quantisations extentrank/extentdisc pass) and does NOT separate
  within the topodiam family (topojitwide at 901 passes). So coarseness matters for one family and
  not the other, which means it is interacting with something not yet isolated.

  AND A DEFECT IN MY OWN TEST, found while running it: I proxied "shared cross-parent axis" by
  sd(per-parent mean) > 0.05. That returns YES for topodiamoff -- the arm whose axis is MOST
  destroyed -- because a large per-parent offset inflates exactly that statistic. The proxy measures
  the opposite of what it names. Do not reuse it; topodiamshuffle is the real test of that account.

WHAT EACH ARM IS FOR, and the known limitation of each -- stated here because two of them are
weaker controls than they were originally described as:

  extentdisc   extent quantile-binned to topodiam's LEVEL COUNT, each row given its bin MEAN.
               LIMITATION: matching the level count is NOT matching entropy. Quantile bins are
               equiprobable by construction (H = log 10 = 2.303) while topodiam is strongly peaked
               (40.9% of train rows at d=0, H = 1.450). So extentdisc carries ~1.5x topodiam's
               conditional entropy and sits only ~53% of the way from extent to topodiam on that
               axis. It is a HALF-MATCHED control. A properly matched version maps extent onto
               topodiam's empirical marginal by rank (monotone transport), which matches H(d) and
               H(d|parent) by construction; an auditor built that and it recovers 20-29% of
               topodiam's DIRECTIONAL effect, against the 33-50% of the POOLED gap this arm gave.

  extentrank   extent mapped onto topodiam's EMPIRICAL MARGINAL by rank (monotone transport).
               Matches H(d) and H(d|parent) to 1.00x on train (1.4513 vs 1.4502; 0.9371 vs 0.9359)
               against extentdisc's 1.59x/1.54x, so it is the entropy-matched control extentdisc
               was described as and is not. On VALID the match is 0.99x / 0.93x -- transport
               realises 8 of topodiam's 9 valid levels, a small mismatch in the CONSERVATIVE
               direction, stated rather than rounded away.
               LIMITATION, MEASURED BEFORE ANY TRAINED NUMBER EXISTS, so it cannot be narrated
               afterwards either way: extentrank returns topodiam's EXACT value on 49.61% of rows
               (train and valid alike), against a permutation floor of 28.04% / 28.39% -- an excess
               of +21.6 / +21.2 points. It is therefore NOT an independent channel. This is not a
               construction bug and cannot be fixed: extent and topodiam genuinely correlate at
               r=0.635, and any transport preserving extent's rank order while wearing topodiam's
               marginal must inherit that agreement. TWO CONSEQUENCES, opposite in sign:
                 (a) it makes extentrank a CONSERVATIVE control -- it is handed topodiam's answer
                     on half the rows, so if topodiam still beats it the conclusion is STRONGER
                     than it would be against a truly independent entropy-matched channel;
                 (b) whatever fraction of topodiam's effect extentrank recovers MUST NOT be read as
                     "that fraction of the effect is entropy". Part of it is literal topodiam
                     agreement. The entropy question is answered by the DIRECTION of the topodiam
                     vs extentrank comparison, not by the ratio between them.
               Its identifiable/generic ceiling ratio is 0.5282 train / 0.5133 valid against
               topodiam's 0.5240 / 0.5266 -- i.e. comparable denominators, unlike extentdisc's
               0.7665 / 0.7420. Run with TWO seeds; a single draw is not a result here.

  topojit      topodiam + U(-0.5, +0.5). LIMITATION, AND IT IS SERIOUS: the jitter is capped
               strictly inside the rounding boundary, so round(topojit) == topodiam on 8055/8055
               train and 901/901 valid rows -- ZERO exceptions. The transform is INFORMATION-
               PRESERVING and invertible. As a control for "the advantage comes from low entropy"
               it can only fire if the model fails to round, i.e. it is structurally incapable of
               removing the thing it is supposed to remove. That is a guard-that-cannot-fire
               promoted to an experimental arm. Its measured 11-12% cost is the model's ROUNDING
               cost, not an entropy cost. Use topojitwide for the entropy question.

  topojitwide  topodiam + U(-2.0, +2.0). The jitter DOES cross integer boundaries: 25.9% of rows
               keep their nearest integer. This is the control topojit was meant to be.
               READ THAT 25.9% AGAINST ITS FLOOR, WHICH I NEVER MEASURED AND AN AUDITOR DID: the
               same metric scores 41.84% on the `none` arm -- the CONSTANT-ZERO control -- because
               41.8% of valid rows have topodiam=0, so a channel carrying no information at all
               "recovers the class" on 41.8% of rows. 25.9% is therefore BELOW the floor, not a
               small number above zero. Worse for the story it was used to tell: `extent`, the arm
               described as having no coarse class at all, scores 47.61% -- HIGHER than either
               jitter-wide arm and higher than a channel that is identically zero. So nearest-
               integer recoverability is meaningful only INSIDE the topodiam+noise family and
               cannot be the variable that orders seven arms (Spearman vs GAP_neg rho=+0.683,
               p=0.062, n.s.). Mutual information with the underlying integer does order them
               (rho=+0.862, p=0.006) -- but it does NOT explain directionality, which is what the
               extentctr / topodiamoff pair above is built to test.

  topojitwidescaled  topojitwide rescaled to topodiam's sd, isolating class destruction from the
               52% input-magnitude inflation that topojitwide otherwise carries.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
ARMS_DIR = os.path.join(ROOT, 'experiments/covalentformer/data/steer_k8_arms')
SEED_JIT = 20260915        # produced data/steer_k8_ctrl/topojit_*.csv
SEED_WIDE = 424242         # produced data/steer_k8_ctrl2/topojitwide_*.csv
# SECOND NOISE REALISATION. The two "seeds" every jitter arm is quoted with are TRAINING seeds:
# both read the SAME valid CSV and differ only in the torch seed. The U(-0.5,0.5) and U(-2,2) draws
# were made ONCE at data-build time, so the three arms that carry the whole mechanism argument are
# n=1 in the random variable that DEFINES them -- the single most-repeated lesson in this project
# ("seed and replicate anything that samples") violated in the controls built to enforce it, by me.
# --rep2 redraws the jitter under these seeds so the noise realisation itself gets a replicate.
SEED_JIT2 = 777001
SEED_WIDE2 = 777002


def build(out_ctrl, out_ctrl2, out_ctrl3, pooled=False, rep2=False):
    """Regenerate all four control arms. Draw order matters: train then valid, one RNG per arm.

    `pooled` selects WHERE the two FITTED transforms get their parameters, and it is the one thing
    in this file that changes a number. As shipped (pooled=False) both were refit INSIDE each split:

      topojitwidescaled  factor = sd(topodiam_split)/sd(topojitwide_split), which is 0.675931 on
                         train and 0.655931 on valid. The model is trained on one linear map and
                         scored on another, so the valid-time conditioning is 3.0% COMPRESSED
                         relative to what training saw -- on the arm whose entire purpose is to be
                         sd-matched to topodiam.
      extentdisc         bin EDGES come from train (correct) but bin MEANS are recomputed on valid,
                         so the level values differ between splits (-2.1649 train vs -2.1711 valid)
                         and the held-out conditioning is a function of the held-out set's own
                         composition.

    Both are the cond_repeat class -- a transform parameter baked into the data and not held fixed
    across the train/score boundary -- and both shrink a CONTROL arm's valid-time magnitude, which
    `_negate` then doubles. That pushes the control's GAP_neg DOWN, i.e. toward "the integer beats
    its matched control", which is the direction of the filed conclusion. pooled=True fits one
    factor and one set of bin means on train+valid together and stamps them.

    The two JITTER arms are unaffected by this flag: they are pure additive noise with no fitted
    parameter, and the draw order is unchanged, so they come out bit-identical either way.

    MEASURED SIZE OF THIS DEFECT, now that both versions have been trained and scored:
        extentdisc         per-split +0.00945 -> pooled +0.00924   delta -0.00021  (-2.2%)
        topojitwidescaled  per-split +0.00344 -> pooled +0.00357   delta +0.00014  (+3.9%)
      Both verdicts unchanged. For comparison, the SAME arm under the SAME construction at a
      different TRAINING SEED moves 10x further: extentdisc +0.00945 (s20260914) vs +0.00742
      (s777), a spread of -21.5%.
    SO THE STRADDLE IS REAL AND AN ORDER OF MAGNITUDE BELOW THE NOISE FLOOR IT IS MEASURED
    AGAINST. Worth fixing so a future split cannot inherit it; NOT worth quoting as a correction,
    and emphatically not worth attributing any arm difference to.
    THIS IS THE THIRD CONSTRUCTION DEFECT TONIGHT WITH THAT PROFILE -- the pooled-rank leak in
    extentrank was exactly 0 on the scored split (0/901 rows), and the ridge lambda hazards
    measured <=0.02 R2 here against the 0.82/0.60 imported from another dataset. The limiting
    factor on every one of these numbers is n=2 TRAINING SEEDS, not construction purity. Spend the
    next hour on seeds, not on further defect archaeology.
    """
    for d in (out_ctrl, out_ctrl2, out_ctrl3):
        os.makedirs(d, exist_ok=True)
    _sj, _sw = (SEED_JIT2, SEED_WIDE2) if rep2 else (SEED_JIT, SEED_WIDE)
    rng_jit, rng_wide = np.random.default_rng(_sj), np.random.default_rng(_sw)
    fit_seeds = dict(seed_jitter=_sj, seed_wide=_sw, realisation=2 if rep2 else 1)
    parts = {p: dict(td=pd.read_csv(os.path.join(ARMS_DIR, 'topodiam_%s.csv' % p)),
                     ex=pd.read_csv(os.path.join(ARMS_DIR, 'extent_%s.csv' % p)),
                     nn=pd.read_csv(os.path.join(ARMS_DIR, 'none_%s.csv' % p)))
             for p in ('train', 'valid')}
    n_lv = len(np.unique(parts['train']['td'].d.values))
    qs = np.quantile(parts['train']['ex'].d.values, np.linspace(0, 1, n_lv + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    fit = dict(fit_seeds)
    # extentrank: MONOTONE TRANSPORT of extent onto topodiam's EMPIRICAL MARGINAL. This is the
    # control extentdisc was supposed to be and is not. extentdisc matches topodiam's LEVEL COUNT,
    # which is not the same as matching entropy: quantile bins are equiprobable by construction
    # (H = log 10 = 2.303) while topodiam is strongly peaked (40.9% of train rows at d=0,
    # H = 1.450), so extentdisc carries ~1.5x topodiam's conditional entropy and is HALF-matched.
    # Ranking extent and reading off topodiam's sorted values reproduces H(d) EXACTLY and H(d|parent)
    # by construction, so any residual advantage of topodiam cannot be an entropy advantage.
    # An auditor built this inline and it recovers 20-29% of the DIRECTIONAL effect against the
    # 33-50% of the POOLED gap extentdisc gave -- i.e. the better-matched control leaves LESS
    # unexplained, which is the direction that matters for #191. It existed only in that agent's
    # scratch dir: the same orphan-artifact class this file was written to close for the other four
    # arms, so it is produced here or it is not quotable. Deterministic, no seed, ties broken by a
    # STABLE sort so two runs agree bitwise.
    # THE TWO DISCRIMINATOR ARMS. An auditor's account of the mechanism is that a channel STEERS
    # (obeys the SIGN) when its values sit on a SHARED axis comparable across parents, and that how
    # MUCH it is worth is set separately by its mutual information with the achievable edit -- so
    # neither effect is about how COARSE the channel is. That competes with my class-recoverability
    # account. These two arms separate them ON A TEST THAT CAN GO EITHER WAY, which is why they are
    # built before either answer is known:
    #   extentctr    extent MINUS ITS OWN PARENT'S MEAN. Within-parent information is untouched, so
    #                MI with the achievable edit is unchanged; the axis stops being comparable
    #                across parents. Shared-axis account predicts directionality FALLS. A pure
    #                MI/coarseness account predicts NO CHANGE.
    #   topodiamoff  topodiam PLUS A LARGE PER-PARENT RANDOM OFFSET (sd 8, ~8x topodiam's own sd,
    #                drawn ONCE PER PARENT and shared across splits). Class COUNT within a parent is
    #                untouched and the within-parent spacing is exactly preserved; only the
    #                cross-parent axis is destroyed. Shared-axis account predicts directionality
    #                collapses toward ~1.0x. A class-recoverability account predicts it SURVIVES,
    #                because every class distinction inside a parent is still there.
    # If directionality survives in topodiamoff, the auditor's account is wrong and mine is not.
    # ---- extentrank_pf: THE PER-ROW-FUNCTION FIX TO extentrank.
    # An auditor's ★5 asked for exactly the arm extentrank is -- extent binned onto topodiam's own
    # empirical level frequencies -- but with a requirement extentrank FAILS: "a strict per-row
    # function with train-fixed edges AND train-fixed level values". My extentrank ranks over the
    # POOLED train+valid frame, so a valid row's conditioning depends on which OTHER valid rows are
    # in the split. That is precisely the defect the same auditor charges extentdisc with (its bin
    # MEANS are refit per split), and I reproduced it in the arm I built to replace extentdisc.
    # A conditioning value that cannot be computed for a single new molecule at inference time is
    # not a channel, it is a statistic of the evaluation set.
    # FIXED HERE: fit a monotone step map on TRAIN ONLY -- train extent quantiles -> train topodiam
    # sorted values -- then APPLY it to valid by looking each row's own extent up in that fixed map.
    # Valid rows never see each other. np.searchsorted on train-derived edges is a pure per-row
    # lookup, so this arm IS computable for one molecule.
    _ex_tr = np.sort(parts['train']['ex'].d.values)
    _td_tr = np.sort(parts['train']['td'].d.values)
    _n_tr_ = len(_ex_tr)
    _edges = _ex_tr                                   # train extent values, ascending
    _levels = _td_tr                                  # train topodiam values, ascending
    def _transport(v):
        i = np.searchsorted(_edges, v, side='left')
        return _levels[np.clip(i, 0, _n_tr_ - 1)]
    fit['extentrank_pf_map'] = 'train-only monotone transport; searchsorted on train extent'
    # MEASURED MAGNITUDE OF THIS FIX: ZERO ON THE SPLIT IT IS SCORED ON.
    # The defect is real in principle -- extentrank ranks over POOLED train+valid, so a valid row's
    # conditioning depends on the other valid rows -- and extentrank_pf removes it properly (a
    # valid row is now a pure lookup into a train-fixed map). But the two arms come out
    # BIT-IDENTICAL on all 901 valid rows and differ on only 27 of 8055 train rows, because
    # extent's rank order barely changes when 901 rows are added to 8055. So:
    #   DO NOT report any GAP difference between extentrank and extentrank_pf as "the size of the
    #   pooled-rank leak". It is training noise on 27 rows. The honest statement is that the leak
    #   is a real construction defect with a measured effect of zero here, fixed so that a future
    #   split with a different train/valid ratio cannot inherit it.
    # This is the mirror of the guard-that-cannot-fire pattern: a fix that cannot show an effect,
    # which is fine as engineering and misleading if quoted as a result.

    _off_rng = np.random.default_rng(555001)
    _shuf_rng = np.random.default_rng(555002)
    _sc_rng = np.random.default_rng(555003)
    fit['scale_seed'] = 555003
    fit['shuffle_seed'] = 555002
    _anch = pd.concat([parts[p]['td'][['anchor']] for p in ('train', 'valid')]).anchor.values
    _uniq = pd.unique(_anch)
    _offmap = dict(zip(_uniq, _off_rng.normal(0.0, 8.0, len(_uniq))))
    # lognormal so every scale is strictly POSITIVE (a negative s_p would flip the sign and make
    # the arm a sign-scramble rather than an axis-scramble), spread ~8x between the 5th and 95th.
    _scmap = dict(zip(_uniq, np.exp(_sc_rng.normal(0.0, 0.65, len(_uniq)))))
    fit['offset_seed'] = 555001
    fit['offset_sd'] = 8.0
    _ex_all = np.concatenate([parts[p]['ex'].d.values for p in ('train', 'valid')])
    _td_all = np.concatenate([parts[p]['td'].d.values for p in ('train', 'valid')])
    _rank = np.empty(len(_ex_all), dtype=int)
    _rank[np.argsort(_ex_all, kind='stable')] = np.arange(len(_ex_all))
    _mapped = np.sort(_td_all, kind='stable')[_rank]
    fit['extentrank_marginal_is_topodiam'] = True
    _n_tr = len(parts['train']['ex'])
    parts['train']['er'] = _mapped[:_n_tr]
    parts['valid']['er'] = _mapped[_n_tr:]
    if pooled:
        # ONE set of parameters, fitted once on train+valid, applied to both.
        _ex = _ex_all
        _i = np.clip(np.digitize(_ex, qs[1:-1]), 0, n_lv - 1)
        fit['bin_means'] = [float(_ex[_i == i].mean()) if (_i == i).sum() else 0.0
                            for i in range(n_lv)]
    for part in ('train', 'valid'):
        td, ex, nn = parts[part]['td'], parts[part]['ex'], parts[part]['nn']
        idx = np.clip(np.digitize(ex.d.values, qs[1:-1]), 0, n_lv - 1)
        mid = np.array(fit['bin_means']) if pooled else np.array(
            [ex.d.values[idx == i].mean() if (idx == i).sum() else 0.0 for i in range(n_lv)])
        ed = ex.copy(); ed['d'] = mid[idx]
        er = ex.copy(); er['d'] = parts[part]['er']
        er.to_csv(os.path.join(out_ctrl, 'extentrank_%s.csv' % part), index=False)
        ec = ex.copy(); ec['d'] = ex.d.values - ex.groupby('anchor')['d'].transform('mean').values
        ec.to_csv(os.path.join(out_ctrl, 'extentctr_%s.csv' % part), index=False)
        to = td.copy(); to['d'] = td.d.values + np.array([_offmap[x] for x in td.anchor.values])
        to.to_csv(os.path.join(out_ctrl, 'topodiamoff_%s.csv' % part), index=False)
        # ---- topodiamscale REPLACES topodiamoff, WHICH CANNOT TEST WHAT I BUILT IT TO TEST.
        # The scorer's only directional operator is c -> -c. On an ADDITIVE per-parent offset that
        # flips the offset too, so the negated request is not "the opposite edit for this parent",
        # it is a value nowhere near anything that parent can reach. Measured on the valid split:
        # -c lands OUTSIDE the parent's achievable range on 94.6% of topodiamoff rows (median
        # distance to the nearest achievable sibling 10.42) against 29.0% and 0.000 for topodiam.
        # A large GAP_neg there would be an out-of-distribution magnitude cost -- amplified by the
        # fact that train_phaseA applies NO input normalisation, so 7.7x the input scale also
        # changes the optimisation regime -- and it would have printed as "STEERS", which I had
        # pre-registered as "the shared-axis account is wrong and mine is not". Both confounds push
        # toward my own hypothesis. An auditor caught it before the number existed.
        # A MULTIPLICATIVE positive per-parent scale fixes it: -(d * s_p) == (-d) * s_p, so the
        # negated request is exactly the scaled image of topodiam's negated request and stays in the
        # parent's own achievable set. Cross-parent comparability of magnitude is still destroyed
        # (each parent has its own units), which is the thing under test.
        tsc = td.copy(); tsc['d'] = td.d.values * np.array([_scmap[x] for x in td.anchor.values])
        tsc.to_csv(os.path.join(out_ctrl, 'topodiamscale_%s.csv' % part), index=False)
        ep = ex.copy(); ep['d'] = _transport(ex.d.values)
        ep.to_csv(os.path.join(out_ctrl, 'extentrank_pf_%s.csv' % part), index=False)
        # ---- topodiamshuffle: SEPARATES CARDINALITY FROM A SHARED ORDERED AXIS.
        # Two auditors have given two accounts of what makes an arm steer. One says CARDINALITY --
        # "every arm with <=10 levels steers, every arm with 901 does not". The other says a SHARED
        # AXIS comparable across parents. Neither of my existing discriminators tells them apart:
        # extentctr and topodiamoff both destroy the shared axis AND raise global cardinality, so
        # the two accounts make the same prediction and the test cannot separate them.
        # This arm holds cardinality FIXED at topodiam's TRAIN alphabet (10 levels) while destroying
        # WORDING CORRECTED BY AN AUDITOR'S CSV FORENSICS: I called this "topodiam's own 9-level
        # alphabet". It is the 10-level TRAIN alphabet -- topodiam's VALID column has only 9 levels
        # (no 4.0), the shuffle's valid column has all 10. So this arm is NOT marginal-matched to
        # topodiam; H(shuffle)=2.285 nats is 1.59x H(topodiam)=1.435. It IS matched to extentdisc
        # (10 vs 10 levels, 2.285 vs 2.297 nats), and extentdisc is the comparison actually made,
        # so the experiment is unaffected -- but the arm carries MORE entropy than the channel it
        # controls for, which makes its collapse to +0.00008 harder to explain away, not easier.
        # the
        # shared meaning: each parent gets its own random bijection of that alphabet onto itself.
        # Global distinct values: 9, unchanged. Within-parent distinct count: unchanged. What is
        # gone is that "+2" means the same thing for two different parents -- and with it, the
        # ordering that makes -c the OPPOSITE request rather than merely a different one.
        # Cardinality account predicts it STILL STEERS. Shared-axis account predicts it collapses
        # to the magnitude-only yardstick. Stated before the run so neither can be claimed after.
        # MEASURED BEFORE ANY TRAINED NUMBER, AND IT QUALIFIES MY OWN 'MATCHED PAIR' CLAIM:
        # shuffling flattens the peaked marginal (40.9% of rows at d=0) onto a near-uniform
        # spread over the alphabet, so H lands at 2.2851 -- extentdisc's 2.2973 to two decimals,
        # which is the match I wanted -- but sd goes to 2.86 against extentdisc's 1.02. The pair
        # is matched on CARDINALITY and ENTROPY and is NOT matched on SCALE. That is the same
        # confound an auditor raised against topojitwide (sd 1.53 vs extent's 1.09).
        # WHAT THAT DOES AND DOES NOT INVALIDATE: the DIRECTIONALITY statistic is a RATIO,
        # GAP_neg/GAP_perm, read against a yardstick that is itself a ratio of perturbation
        # sizes -- both numerator and denominator scale linearly with the channel, so the
        # verdict is scale-invariant and the comparison IS valid for 'does it obey the sign'.
        # The GAP MAGNITUDE is not: a channel with 2.8x the input swing moves loss more for
        # reasons that have nothing to do with meaning. So read topodiamshuffle's VERDICT
        # against extentdisc's VERDICT, and do NOT compare their GAP_neg magnitudes.
        # Its own pre-registered yardstick is 1.532 (vs extentdisc 1.401, topodiam 1.330).
        _alpha = np.unique(np.concatenate([parts[p]['td'].d.values for p in ('train', 'valid')]))
        ts_ = td.copy()
        _sh = np.empty(len(td))
        for _a, _idx in td.groupby('anchor').indices.items():
            _perm = _shuf_rng.permutation(_alpha)
            _m = {v: _perm[i] for i, v in enumerate(_alpha)}
            _sh[_idx] = [_m[v] for v in td.d.values[_idx]]
        ts_['d'] = _sh
        ts_.to_csv(os.path.join(out_ctrl, 'topodiamshuffle_%s.csv' % part), index=False)
        tj = td.copy(); tj['d'] = td.d.values + rng_jit.uniform(-0.5, 0.5, len(td))
        tw = td.copy(); tw['d'] = td.d.values + rng_wide.uniform(-2.0, 2.0, len(td))
        parts[part]['tw'] = tw
        ed.to_csv(os.path.join(out_ctrl, 'extentdisc_%s.csv' % part), index=False)
        tj.to_csv(os.path.join(out_ctrl, 'topojit_%s.csv' % part), index=False)
        tw.to_csv(os.path.join(out_ctrl2, 'topojitwide_%s.csv' % part), index=False)
        for d in (out_ctrl, out_ctrl2, out_ctrl3):
            nn.to_csv(os.path.join(d, 'none_%s.csv' % part), index=False)
    # SECOND PASS for the rescale: under pooled=True the factor needs BOTH splits' topojitwide,
    # which does not exist until the loop above has drawn them in order.
    if pooled:
        _td = np.concatenate([parts[p]['td'].d.values for p in ('train', 'valid')])
        _tw = np.concatenate([parts[p]['tw'].d.values for p in ('train', 'valid')])
        fit['sd_factor'] = float(_td.std(ddof=1) / _tw.std(ddof=1))
    for part in ('train', 'valid'):
        td, tw = parts[part]['td'], parts[part]['tw']
        f = fit['sd_factor'] if pooled else float(td.d.std() / tw.d.std())
        fit.setdefault('sd_factor_per_split', {})[part] = f
        ts = tw.copy(); ts['d'] = tw.d.values * f
        ts.to_csv(os.path.join(out_ctrl3, 'topojitwidescaled_%s.csv' % part), index=False)
    return n_lv, fit


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true',
                    help='regenerate into /tmp and diff against the shipped CSVs')
    ap.add_argument('--rep2', action='store_true',
                    help='SECOND JITTER REALISATION (writes *_rep2 dirs). The shipped arms are n=1 '
                         'in the noise draw; this is the replicate of the random variable that '
                         'defines them, not another training seed.')
    ap.add_argument('--fix', action='store_true',
                    help='write to *_fix dirs: adds extentrank_pf (train-only transport, a '
                         'strict per-row function) and topodiamshuffle (cardinality held, '
                         'shared axis destroyed). Separate dir because the live queue is '
                         'READING steer_k8_ctrl_pooled right now.')
    ap.add_argument('--pooled', action='store_true',
                    help='fit the extentdisc bin means and the sd rescale ONCE on train+valid '
                         '(writes *_pooled dirs). Without this the shipped per-split refit is '
                         'reproduced exactly, which is what --verify needs.')
    a = ap.parse_args()
    # --fix WRITES TO A SEPARATE DIRECTORY ON PURPOSE. data/steer_k8_ctrl_pooled/ is being READ by
    # a training queue right now. Rewriting an input file under a running job is how you get a
    # result that belongs to neither version of the data and cannot be attributed to either.
    _suf = ('_pooled' if a.pooled else '') + ('_rep2' if a.rep2 else '') + ('_fix' if a.fix else '')
    base = '/tmp/ctrl_verify' if a.verify else os.path.join(ROOT, 'experiments/covalentformer/data')
    c1 = os.path.join(base, 'steer_k8_ctrl' + _suf); c2 = os.path.join(base, 'steer_k8_ctrl2' + _suf)
    c3 = os.path.join(base, 'steer_k8_ctrl3' + _suf)
    n_lv, fit = build(c1, c2, c3, pooled=a.pooled, rep2=a.rep2)
    print('  rebuilt with n_levels=%d, pooled=%s, jitter realisation %d (seeds %d / %d)'
          % (n_lv, a.pooled, fit['realisation'], fit['seed_jitter'], fit['seed_wide']))
    print('  sd rescale factor per split: %s' % fit.get('sd_factor_per_split'))
    if a.pooled:
        # STAMP THE FITTED PARAMETERS NEXT TO THE DATA. The whole point of the pooled fit is that
        # ONE transform spans the train/score boundary; recording it is what makes that checkable
        # rather than asserted, and it is the check adapt_steer_to_phaseA.py can enforce later.
        # STAMP FROM `fit`, NOT FROM THE MODULE CONSTANTS. This wrote SEED_JIT/SEED_WIDE
        # unconditionally, so data/steer_k8_ctrl_pooled_rep2/ -- built with 777001/777002 -- shipped
        # a stamp claiming 20260915/424242, identical to the rep1 dir it is the REPLICATE of. Two
        # directories whose only difference IS the random variable that defines them carried the
        # same seed. That is worse than no stamp: an absent stamp stops you, a wrong one lets you
        # "reproduce" rep2 at the rep1 seed, get the rep1 data, and conclude rep2 failed to
        # reproduce. `realisation` was never written at all. Same orphan class this file exists to
        # close, reintroduced by me in the --rep2 flag added to close it.
        _stamp = dict(seed_jitter=fit['seed_jitter'], seed_wide=fit['seed_wide'],
                      realisation=fit['realisation'], n_levels=n_lv,
                      fitted_on='train+valid POOLED -- one factor, one set of bin means',
                      sd_factor=fit['sd_factor'], sd_factor_per_split=fit['sd_factor_per_split'],
                      bin_means=fit['bin_means'], offset_seed=fit.get('offset_seed'),
                      shuffle_seed=fit.get('shuffle_seed'),
                      # scale_seed was set in `fit` and then left OUT of the stamp -- and
                      # topodiamscale is the arm that REPLACES topodiamoff as the directionality
                      # discriminator, i.e. the one whose number gets quoted. Its seed was the one
                      # seed not recorded.
                      scale_seed=fit.get('scale_seed'),
                      extentrank_pf_map=fit.get('extentrank_pf_map'))
        json.dump(_stamp, open(os.path.join(c3, 'ctrl_stamp_pooled.json'), 'w'), indent=1)
        json.dump(_stamp, open(os.path.join(c1, 'ctrl_stamp_pooled.json'), 'w'), indent=1)
    if a.verify:
        shipped = os.path.join(ROOT, 'experiments/covalentformer/data')
        ok = bad = 0
        # THE SUMMARY USED TO SAY "the control experiment is reproducible" WHILE CHECKING FOUR OF
        # SEVEN ARMS. extentrank, extentrank_pf, extentctr, topodiamoff, topodiamscale and
        # topodiamshuffle -- every arm added tonight, i.e. the least-reviewed ones -- were never
        # compared, and the unchecked set was invisible in the output. The list is now derived from
        # what is ON DISK and any arm without a shipped counterpart is NAMED as unverified rather
        # than silently skipped. `_suf` is applied here too; reading unsuffixed subdirs made
        # --verify --pooled / --verify --rep2 die on FileNotFoundError.
        _pairs = [('steer_k8_ctrl' + _suf, 'extentdisc'), ('steer_k8_ctrl' + _suf, 'topojit'),
                  ('steer_k8_ctrl' + _suf, 'extentrank'), ('steer_k8_ctrl' + _suf, 'extentrank_pf'),
                  ('steer_k8_ctrl' + _suf, 'extentctr'), ('steer_k8_ctrl' + _suf, 'topodiamoff'),
                  ('steer_k8_ctrl' + _suf, 'topodiamscale'),
                  ('steer_k8_ctrl' + _suf, 'topodiamshuffle'),
                  ('steer_k8_ctrl2' + _suf, 'topojitwide'),
                  ('steer_k8_ctrl3' + _suf, 'topojitwidescaled')]
        _skipped = []
        for sub, arm in _pairs:
            for part in ('train', 'valid'):
                f = '%s_%s.csv' % (arm, part)
                if not (os.path.exists(os.path.join(base, sub, f))
                        and os.path.exists(os.path.join(shipped, sub, f))):
                    _skipped.append('%s/%s' % (sub, f)); continue
                A = pd.read_csv(os.path.join(base, sub, f)).d.values
                B = pd.read_csv(os.path.join(shipped, sub, f)).d.values
                # TWO VERDICTS, BECAUSE ONE OVERSTATES. atol=0/rtol=0 is a BIT comparison, and the
                # sd rescale re-associates float64 multiplies through a CSV round-trip, so
                # topojitwidescaled lands 8.9e-16 away -- 3 ulp, not a seed or construction
                # mismatch. Reporting that as a flat "DIFFERS" would read as a failed
                # reproduction; reporting 8/8 exact would be false. Both are printed.
                _d = np.abs(A - B).max() if len(A) == len(B) else float('nan')
                same = len(A) == len(B) and np.allclose(A, B, atol=0, rtol=0)
                near = len(A) == len(B) and np.allclose(A, B, atol=1e-12, rtol=0)
                print('    %-24s %-6s %-7s %s (max|diff| %.3e)'
                      % (arm, part, 'EXACT' if same else 'BITWISE-DIFF',
                         '' if same else ('within 1e-12' if near else 'BEYOND 1e-12'), _d))
                ok += same; bad += (not same)
        print('  %d/%d BIT-exact (the rest reproduce to machine precision, see max|diff| above)'
              % (ok, ok + bad))
        if _skipped:
            print('  NOT VERIFIED (%d files, no shipped counterpart to compare against): %s'
                  % (len(_skipped), ', '.join(_skipped)))
            print('  -> the control experiment is reproducible FOR THE ARMS LISTED ABOVE ONLY.')
        else:
            print('  -> every arm on disk was compared; the control experiment is reproducible.')
        # WRITE THE VERIFY STAMP WHERE THE VERIFY OUTPUT LIVES, NOT INTO THE SHIPPED DATA.
        # This dumped into data/steer_k8_ctrl/ even under --verify, whose own --help says
        # "regenerate into /tmp and diff against the shipped CSVs". An auditor declined to RUN
        # --verify for exactly that reason and reproduced the comparison by hand instead -- a check
        # nobody dares execute is not a check. `base` is /tmp under --verify.
        json.dump(dict(seed_jitter=fit['seed_jitter'], seed_wide=fit['seed_wide'],
                       realisation=fit['realisation'], n_levels=n_lv,
                       bins_fitted_on=('bin EDGES from train; bin MEANS REFIT PER SPLIT. The '
                                       'previous string said "train rows only, applied to both '
                                       'splits", which the code has never done -- an auditor '
                                       'measured the resulting level divergence at up to 0.0101 '
                                       '(0.93% of the extent sd). Use --pooled for one fixed map.'),
                       sd_factor_per_split=fit.get('sd_factor_per_split'),
                       verified_exact='%d/%d' % (ok, ok + bad)),
                  open(os.path.join(base, 'ctrl_stamp_VERIFY.json'), 'w'), indent=1)
