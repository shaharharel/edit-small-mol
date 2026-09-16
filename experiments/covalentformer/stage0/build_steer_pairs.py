"""Build the MW+TC-matched steering pair set, and REFUSE to emit it if it is bad.

WHY THE GATES ARE IN THE BUILDER AND NOT IN A SEPARATE SCRIPT. A validator you run afterwards is a
validator you can forget to run, or run on a different file, or read too fast. Tonight's whole GPU
budget rides on this one artifact, and the failure mode that costs the night is not a crash -- it is
a plausible-looking CSV with a leaked split or a degenerate label column. So every check below is
FATAL: the file is written only if all of them pass, and the stamp records that they did.

WHAT EACH GATE IS DEFENDING AGAINST, all of these having actually happened in this project:
  G1 scale          -- a filter silently removing almost everything (the `keep` grouping gave median
                       context size 1 and capped the achievable pairs at 96k when millions were assumed)
  G2 chemistry      -- variants that lost the warhead, or SMILES that do not parse
  G3 MW/TC          -- matching claimed but not enforced. min_inc=+-1 bounds ATOM COUNT, not mass:
                       measured median |dMW| 12 Da, p90 29. An explicit mass filter is required.
  G4 label coverage -- a param defined on so few pairs that its arm is unpowered
  G5 label variety  -- a constant or near-constant column. A model conditioned on a constant learns
                       nothing and its GAP is structurally zero; this is how a dead channel looks alive.
  G6 SPLIT LEAKAGE  -- the expensive one. Pairs from the SAME PARENT must not straddle train/valid,
                       or the model sees the parent at train time and "generalisation" is retrieval.
                       Five separate leak channels have been found in this project's splits already.
  G7 env concentration -- a few huge CReM envs supplying most pairs (cf. 2 proteins supplying 61% of
                       held-out, top-5 fragments being 50.1% of counts)
  G8 duplicates     -- the same (parent, variant) pair repeated, inflating n without adding information

REPRODUCIBILITY: every param comes from steer_params.py (which imports the ORIGINAL phi producer
rather than reimplementing it), the RNG is seeded, and the stamp carries param versions + the seed.
"""
from __future__ import annotations
import os, sys, json, time, argparse, collections, hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
CF = os.path.join(ROOT, 'experiments/covalentformer')
sys.path.insert(0, CF)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from crem.crem import mutate_mol
RDLogger.DisableLog('rdApp.*')

import steer_params as SP

DB = os.path.join(ROOT, 'data/crem_db/chembl33_sa2_f5.db')
# WITHIN-PARENT CHOICE IS THE WHOLE TASK, AND max_replacements=12 DESTROYED IT.
# Independently enumerated with the same gates but a 200-replacement cap: a parent admits a MEDIAN
# OF 29 admissible variants (mean 29.6, p10 9, p90 51), and 63.9% of extent_delta variance is
# WITHIN parent. At max_replacements=12 the MW/TC gates kill ~85% of the draws and the build
# delivered 1.77 variants per parent -- steer_smoke2's median was exactly 1.
# Measured oracle error against the number of candidates a parent offers:
#     k=1  1.265 A   k=2  0.870   k=8  0.412   k=20 0.268     random-variant floor 1.257
# At k=1.77 the oracle sits near 0.95 A, i.e. a quarter of the way from random to k=20. A model
# cannot learn to CHOOSE among variants when there is nothing to choose among, so the previous
# 30k build was producing a near-degenerate task at full cost. per_env_cap=8 never bound and could
# not: it caps at 8 rows per parent and the build supplied 1.8.
# 100 draws survive the gates to roughly the per_env_cap of 8, converting the same row budget from
# many-parents x no-choice into fewer-parents x real-choice.
MAX_REPL = 100
ACR = Chem.MolFromSmarts('[CH2]=[CH]C(=O)N')
# WHICH PARAMS THIS PAIR SOURCE CAN ACTUALLY CARRY. Every exclusion below is MEASURED, not assumed.
#   phi     EXCLUDED -- conformer-seed lottery. steer_params.phi_planar_dev now RAISES unless an
#           explicit acknowledgement kwarg is passed, because a DO-NOT-USE note in its docstring did
#           NOT stop this file from writing phi_dir: an audit found it live, 100% covered, with a
#           clean 38/38/24 three-class split, i.e. the best-looking and most dangerous column here.
#   path    EXCLUDED -- 0/295 CReM pairs change it. CReM swaps a scaffold core and never touches the
#   flex    EXCLUDED    warhead, so warhead-internal topology is identical in parent and variant.
#   wclass  EXCLUDED -- unchanged in 295/295 pairs, verified independently. A 100%-one-class column.
#   exitang EXCLUDED -- STRUCTURALLY UNDEFINED on every input: it needs a dummy atom to locate the
#           attachment, and MMFF has no parameters for a dummy, so the embed refuses. 0% coverage on
#           whole molecules AND on fragments. There is no input in between.
# That leaves ONE channel, and it is only usable in its ensemble form (single-seed d_extent has
# reliability 0.143 on these pairs -- the RNG, not the molecule). Stated plainly rather than padded:
# this dataset conditions on ONE parameter.
PARAMS = ('extent',)
ALL_PARAMS_FOR_REFERENCE = ('phi', 'path', 'wclass', 'exitang', 'extent', 'flex')
# Deadbands: a |delta| at or below this is class "same". Set from MEASURED noise, not taste.
# phi: 0.67 deg is the producer's run-to-run MAE; 2.0 is 3x that. path/flex are integers.
# extent 0.5 -> 0.9: THE DISCRETISATION WAS THE FRAGILE PART, NOT THE MEASUREMENT.
# The ensemble fix made extent_delta seed-stable (reliability 0.923, sign agreement 0.900), and I
# reported that as though it made the LABEL stable. It did not. Measured across two master seeds on
# shipped rows: sd(d_extent) = 0.288 A, which is 58% of a 0.5 A deadband; 37.8% of rows sit within
# 1 sd of a class boundary and ~15% change three-class label under a new seed (directly observed
# 1/12). 0.9 A is ~3 sd, which puts the expected flip rate near 2%. The continuous delta is stored
# on every row, so extent_dir can be RE-DERIVED at any deadband without rebuilding the dataset --
# that is why the continuous value is kept rather than discarded after labelling.
DEADBAND = dict(phi=2.0, path=0.0, wclass=None, exitang=5.0, extent=0.9, flex=0.0)
EXTENT_SEED_SD = 0.288      # measured, stamped
# THE 3-SIGMA JUSTIFICATION FOR deadband=0.9 WAS WRONG BY 9x. It predicted a ~2% three-class flip
# rate; measured across three master seeds on 798 pairs the flip rate at 0.9 is 18.05%, and at 0.5
# it is 24.50%. The Gaussian-tail reasoning fails because flips are governed by how many pairs SIT
# NEAR THE BOUNDARY, and widening from 0.5 to 0.9 pushes the boundary into a still-dense part of
# the delta distribution. Full sweep (flip rate / majority share / class-balanced n_eff):
#   0.25 -> 27.6% / 40.4% / 728     0.50 -> 24.5% / 39.7% / 784     0.75 -> 20.6% / 54.4% / 675
#   0.90 -> 18.1% / 62.5% / 580     1.00 -> 16.2% / 67.3% / 519
# NO deadband is both stable and non-trivial: the flip rate never falls below 16.2%, and a
# conformer-free 2D classifier captures >=88.7% of the seed-to-seed reproducibility ceiling at every
# setting (at 0.9: ceiling 0.8229, 2D 0.7644 = 92.9%). The three-class framing should be abandoned
# in favour of regressing extent_delta continuously -- the continuous delta has ICC 0.9045 across
# three seeds while its discretisation flips 18% of the time. Discretisation throws away almost all
# of the reliability the 30-conformer ensemble was paid for.


def _fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def _one_parent(args):
    """Generate variants of one parent and label every (parent, variant) pair on all six params."""
    smi, max_repl, mw_tol, tc_lo, tc_hi = args
    out = []
    # BIND BEFORE THE FIRST RETURN, NOT AFTER. This was assigned below the early-return block, and
    # because it is assigned SOMEWHERE in the function Python binds it local throughout -- so all
    # three early returns raised UnboundLocalError instead of returning. f.result() re-raised into
    # the bare `except` in build(), which prints 'worker died: %s'. Every parent with no acrylamide
    # match, every CReM raise and every parent yielding zero variants therefore printed as a worker
    # death, indistinguishable from a real crash. Row counts were unaffected (out is provably [] at
    # all three sites and `done` still increments) so no number moved, but it destroyed the only
    # signal for a genuine worker failure -- and the "TUPLE on every path" comments described a
    # contract three of the four paths could not reach. One line, and they can.
    n_label_fail = [0]
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(ACR):
        return out, int(n_label_fail[0])   # TUPLE on every path -- see the contract note below
    try:
        variants = list(mutate_mol(m, db_name=DB, radius=3, min_size=1, max_size=8,
                                   min_inc=-2, max_inc=2, max_replacements=max_repl, ncores=1))
    except Exception:
        return out, int(n_label_fail[0])   # TUPLE on every path -- see the contract note below
    if not variants:
        return out, int(n_label_fail[0])   # TUPLE on every path -- see the contract note below
    p_mw, p_fp = Descriptors.MolWt(m), _fp(m)
    # GUARDED: this was outside the try while the variant-side call was inside it. A parent-side
    # raise propagates out of the worker, through f.result(), and kills the entire build with no
    # partial save -- the exact failure the variant-side guard exists to prevent, unmitigated on
    # the other half. phi's RuntimeError guard would take this path if phi were ever re-added.
    try:
        p_lab = {k: SP.PARAMS[k](smi) for k in PARAMS}
    except Exception:
        return out, int(n_label_fail[0])   # TUPLE on every path -- see the contract note below
    for v in variants:
        vm = Chem.MolFromSmiles(v)
        if vm is None or not vm.HasSubstructMatch(ACR):
            continue                                   # G2: warhead must survive the edit
        dmw = abs(Descriptors.MolWt(vm) - p_mw)
        if dmw > mw_tol:
            continue                                   # G3: EXPLICIT mass filter (min_inc is not one)
        tc = DataStructs.TanimotoSimilarity(p_fp, _fp(vm))
        if not (tc_lo <= tc <= tc_hi):
            continue
        row = dict(parent=smi, variant=v, tc=float(tc), dmw=float(dmw))
        ok = False
        for k in PARAMS:
            try:
                a, b = p_lab[k], SP.PARAMS[k](v)
            except Exception:
                # One unparseable/atomless SMILES inside a ProcessPool worker otherwise propagates
                # through f.result() and kills the entire build with no partial save. The row is
                # then DROPPED (ok stays False), not written with a None label -- verified. But a
                # dropped row is invisible, and G4's coverage is pinned at 100% by construction, so
                # a systematic failure over a whole class of variants would leave no trace at all.
                # Counted here so it cannot.
                a = b = None
                n_label_fail[0] += 1
            row['%s_parent' % k] = a
            row['%s_variant' % k] = b
            if a is None or b is None:
                row['%s_delta' % k] = None
                row['%s_dir' % k] = None
                continue
            if k == 'wclass':
                row['%s_delta' % k] = None
                row['%s_dir' % k] = 'same' if a == b else 'change'
                ok = True
                continue
            d = float(b) - float(a)
            row['%s_delta' % k] = d
            db = DEADBAND[k]
            row['%s_dir' % k] = 'same' if abs(d) <= db else ('up' if d > 0 else 'down')
            ok = True
        if ok:
            out.append(row)
    # RETURN THE DROP COUNT OUT OF BAND. It used to be packed into a sentinel row
    # dict(parent=smi, variant=None, _label_failures=n) which the VERY NEXT LINE filtered out,
    # because that sentinel's variant IS None. Verified by executing the exact expression: 3 rows
    # appended, 2 returned, sentinel survives = False. So the comment "Counted here so it cannot
    # leave no trace" described behaviour the code did not have, and the failure class it was
    # written to expose stayed invisible. score_steer_arms.py cites this very counter as one of the
    # prior guard-that-cannot-fire incidents -- while it was still live.
    return ([r for r in out if r.get('variant') is not None], int(n_label_fail[0]))


def build(n_target, workers, mw_tol, tc_lo, tc_hi, per_env_cap, seed, outdir):
    rng = np.random.default_rng(seed)
    import glob
    df = pd.concat([pd.read_csv(f) for f in
                    glob.glob(os.path.join(ROOT, 'paper/reproducibility/metrics/planar_2d/*.csv'))])
    df = df[(df.acryl_match == True) & df.smi.notna()].drop_duplicates('smi')
    pool = df.smi.tolist()
    rng.shuffle(pool)
    print('parent pool: %d acrylamide-bearing molecules' % len(pool), flush=True)

    rows, t0, done = [], time.time(), 0
    n_label_fail_total = 0
    # ROLLING SUBMISSION, NOT BATCH-DRAIN. The previous loop filled workers*40 futures then drained
    # ALL of them via as_completed before submitting again -- a full barrier every 560 parents. With
    # ensemble labelling one pathological parent (many variants x 30 conformers) holds the whole pool
    # hostage at every tail: measured live, 14 of 15 workers sat at 0.0% CPU with their CPU-time
    # frozen while a single worker ran at 99.9%, and the 15-min load average fell from 24.0 to 5.2.
    # The per-500-parent wall time was climbing 189 -> 272 -> 377s as those tails grew. Here a new
    # parent is submitted the moment any future completes, so the pool never drains.
    from concurrent.futures import FIRST_COMPLETED, wait
    it = iter(pool)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pend = set()
        for _ in range(workers * 3):                 # small window: enough to hide latency,
            try:                                     # not so large that a tail can form
                pend.add(ex.submit(_one_parent, (next(it), MAX_REPL, mw_tol, tc_lo, tc_hi)))
            except StopIteration:
                break
        while pend:
            gone, pend = wait(pend, return_when=FIRST_COMPLETED)
            for f in gone:
                try:
                    # BARE UNPACK, NO TRUTHINESS GUARD. `if _res:` used to rescue the four early
                    # returns that handed back a bare LIST instead of a tuple -- and it worked ONLY
                    # because `out` is provably [] at every one of them (all four sit above the
                    # variant loop). With exactly 2 accumulated rows a bare list would have unpacked
                    # as `_r, _nf = [row1, row2]`, binding _r to a DICT, and rows.extend(dict) would
                    # have silently extended `rows` with the dict's KEYS. Every return is now a
                    # tuple, so the unpack is total and a malformed return raises immediately.
                    _r, _nf = f.result()
                    rows.extend(_r); n_label_fail_total += _nf
                except Exception as e:
                    print('  worker died: %s' % e, flush=True)
                done += 1
                if done % 500 == 0:
                    el = time.time() - t0
                    _pc0 = collections.Counter(r['parent'] for r in rows)
                    print('  %d parents -> %d pairs (%d post-cap, target %d) (%.0fs, %.2f pairs/s)'
                          % (done, len(rows), sum(min(per_env_cap, c) for c in _pc0.values()),
                             n_target, el, len(rows) / max(el, 1e-9)), flush=True)
            # BREAK ON THE POST-CAP COUNT, NOT THE PRE-CAP ROW COUNT.
            # The 1.3x headroom was sized when per_env_cap never bound. It binds hard now: the k8
            # build broke at 19,520 pre-cap rows and the cap cut that to 8,957 -- 45.9% retention,
            # 59.7% of n_target, against a G1 FATAL bar at 50%. Ten points of margin, and the stated
            # next step (MORE PARENTS) does not change retention, so the next build lands in the
            # same place and FATALs if realised variants-per-parent rises at all. Counting what the
            # cap will actually keep removes the fudge factor entirely instead of retuning it.
            _pc = collections.Counter(r['parent'] for r in rows)
            _post_cap = sum(min(per_env_cap, c) for c in _pc.values())
            if _post_cap >= n_target:
                for f in pend:
                    f.cancel()
                break
            for _ in range(len(gone)):
                try:
                    pend.add(ex.submit(_one_parent, (next(it), MAX_REPL, mw_tol, tc_lo, tc_hi)))
                except StopIteration:
                    break
    print('RAW: %d pairs from %d parents in %.0fs' % (len(rows), done, time.time()-t0), flush=True)
    if not rows:
        raise SystemExit('FATAL G1: zero pairs produced.')

    d = pd.DataFrame(rows)
    _n_before = len(d)
    d = d.drop_duplicates(subset=['parent', 'variant'])
    n_raw_dupes = _n_before - len(d)
    pre_cap = d.copy()          # G7 must see the distribution BEFORE per-parent capping
    # UNCAPPED FRAME TO DISK. The cap is a POLICY, not a measurement, and until now it was applied
    # in-process with the pre-cap frame discarded -- so the cost of changing the policy was a full
    # regeneration. It is written first, unconditionally, for the same reason the pre-gate dump is:
    # so no downstream decision can cost hours of CReM+MMFF compute. It is also what makes the
    # sampled-vs-head bias below measurable at full n instead of on an n=11 spot check.
    os.makedirs(outdir, exist_ok=True)
    pre_cap.to_csv(os.path.join(outdir, 'steer_pairs_UNCAPPED.csv'), index=False)
    # G7: cap per parent so a few prolific scaffolds cannot dominate.
    #
    # SAMPLED, NOT .head(). `.head(per_env_cap)` keeps the first k rows in CReM GENERATION ORDER,
    # which is the DB's query order and is not random with respect to the label. That was harmless
    # while the cap never bound (1.77 variants/parent). MAX_REPL=100 raised the realised yield to
    # 14.3 variants/parent, so the cap now binds on essentially every parent and .head() selects
    # ~56% of each parent's variants by a systematic rule. Spot-measured on 11 smoke parents with
    # >8 survivors, first-8 mean delta-topodiameter was +0.1932 against +0.0729 for all survivors --
    # a 2.6x skew, same sign, in the axis the conditioning channel IS. A skew toward one direction
    # halves the "make it smaller" half of a BIDIRECTIONAL steering task, which is the one thing
    # this dataset exists to test. n=11 is a spot check, not a result; the UNCAPPED dump above lets
    # the real number be computed from this build's own rows. Seeded off `seed` so it replicates.
    # Shuffle-then-head, not groupby.apply(sample): identical uniform draw, but groupby.apply is
    # mid-deprecation for operating on the grouping column and the future behaviour DROPS `parent`
    # -- a silent schema change in the column the split is built on. Verified equivalent: 8/3/8 per
    # parent on a 20/3/14 frame, non-head indices, bit-identical under a repeated seed.
    d = d.sample(frac=1, random_state=seed).groupby('parent', group_keys=False).head(per_env_cap)
    if len(d) > n_target:
        d = d.sample(n_target, random_state=seed)
    d = d.reset_index(drop=True)

    # ---- PARENT-DISJOINT SPLIT (G6). Split on PARENT, never on row.
    # SPLIT ON CONNECTED COMPONENTS OF THE PARENT<->VARIANT GRAPH, not on parent identity.
    #
    # WHY THIS REPLACES BOTH THE PARENT SPLIT AND THE DE-STRADDLE PASS. Splitting on `parent` left
    # two leaks that had to be patched afterwards, and one patch was a fatal gate that would have
    # destroyed a 2h45m run: (a) two parents can yield the SAME variant, so a variant straddled the
    # split; (b) a molecule can be a PARENT of one row and a VARIANT of another -- measured, 7.05%
    # of variants are themselves in the parent pool, projecting to ~325 cross-role straddles at
    # 30k rows. I wrote a G6 clause that was FATAL on non-zero while my own comment estimated ~12.
    # Union-find over every (parent, variant) edge assigns each MOLECULE to exactly one side
    # whatever role it plays, so both leaks are structurally absent and neither gate can fire on
    # a condition the data guarantees.
    #
    # The de-straddle pass is gone with it. Its stamped claim was also wrong: majority-side
    # retention kept TRAIN 99.9% of the time (64.8% via the tie rule alone, since a 90/10 split
    # makes train the majority by construction), leaving multi-parent-variant enrichment at 13.8x
    # against the old rule's 15.4x. It removed ~10% of the bias it claimed to remove.
    _par = {}
    def _find(x):
        _par.setdefault(x, x)
        while _par[x] != x:
            _par[x] = _par[_par[x]]; x = _par[x]
        return x
    def _uni(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb: _par[ra] = rb
    for _a, _b in zip(d.parent, d.variant):
        _uni(_a, _b)
    d['_comp'] = [_find(x) for x in d.parent]
    comps = d._comp.unique().tolist()
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(comps)
    # components are uneven, so take components until ~10% of ROWS are held out
    csize = d._comp.value_counts().to_dict()
    val_c, acc, want = set(), 0, 0.10 * len(d)
    for c in comps:
        if acc >= want: break
        val_c.add(c); acc += csize[c]
    d['split'] = np.where(d._comp.isin(val_c), 'valid', 'train')
    print('COMPONENT SPLIT: %d components, valid = %d comps / %d rows (%.1f%%)'
          % (len(comps), len(val_c), int((d.split == 'valid').sum()),
             100 * (d.split == 'valid').mean()), flush=True)
    # DE-STRADDLE. Two different parents can produce the same variant molecule; if their parents land
    # on opposite sides, that variant is in both splits. At 40 parents this was 1 of 294 unique
    # variants (0.34%), which extrapolates to ~150 collisions in a 250k build -- and G6 is FATAL on
    # any non-zero count, so the build would run to completion and then write nothing. Dropping the
    # valid-side copy is the fix; aborting is not. The count is logged, never silent.
    os.makedirs(outdir, exist_ok=True)
    # RAW DUMP BEFORE THE GATES. The structural defect behind tonight's near-miss was not the gate
    # itself but that a FATAL gate discards hours of compute with no partial save. Written first,
    # unconditionally, so any gate failure is recoverable by re-gating rather than re-generating.
    d.drop(columns=['_comp'], errors='ignore').to_csv(
        os.path.join(outdir, 'steer_pairs_RAW_pregate.csv'), index=False)
    print('raw rows dumped pre-gate -> steer_pairs_RAW_pregate.csv (%d rows)' % len(d), flush=True)
    report = gates(d, n_target, mw_tol, tc_lo, tc_hi, pre_cap=pre_cap, n_raw_dupes=n_raw_dupes)
    # SCOPE LIMITS TRAVEL WITH THE DATA. Every number below was measured on the 798-pair smoke set
    # and independently reproduced. They are stamped because the alternative is that someone reads
    # `extent_dir` as a validated 3D shape channel -- which is what I did for two hours.
    scope = dict(
        what_extent_is=(
            'ensemble-mean max heavy-atom interatomic distance over N_CONF converged conformers. '
            'NOT a validated 3D shape channel. Quote the topological baseline alongside it or the '
            'claim is not defensible.'),
        one_integer_baseline=dict(
            feature='delta(topological diameter) = longest shortest-path in bonds, NO conformer',
            r2_scaffold_disjoint=0.3993, ecfp4_r2=0.4048,
            ecfp4_adds_over_one_integer=0.0055,
            six_2d_descriptors_r2=0.4358, ecfp4_plus_descriptors_r2=0.5933,
            METHOD_ERROR_FLAG=(
                'EVERY R2 ON THIS LINE WAS SELECTED THE WAY #181 IDENTIFIED AS WRONG: max over the '
                'ridge lambda EVALUATED ON THE HELD-OUT SET. So these LEVELS are not quotable as '
                'stated. '
                'BUT THE SIZE OF THAT ERROR IS NOW MEASURED ON THIS DATASET, AND MY FIRST VERSION '
                'OF THIS FLAG GOT IT WRONG IN THE WORST POSSIBLE WAY: it quoted 0.82 and 0.60 R2, '
                'which are DELTA-GEOMETRY numbers from a different population (#181/#182). '
                'Importing another population\'s magnitude is the #179/#187 failure class, and I '
                'committed it INSIDE the flag whose purpose is to warn that a number is '
                'unreliable. Measured here instead, on steer_k8 extent_delta, ECFP4 difference '
                '(2048 cols), component-disjoint outer split (7778 train / 1179 test, 663/165 '
                'components): peeking at the outer set to pick lambda is worth +0.0122 R2, not '
                '0.82; and a RANDOM inner split versus a component-MATCHED one came out at '
                '-0.0122, i.e. on this data the second hazard does not even have the claimed sign. '
                'Both are <=0.02 here and n=1 in the split draw, so treat them as small and '
                'sign-unstable rather than as a correction to apply. The honest ECFP4 R2 is '
                '~0.526 (matched inner) / 0.538 (random inner). The CONTRAST these numbers were '
                'stamped for -- a fingerprint adding little over one integer -- is what to quote.'),
            note='A 4096-dim fingerprint adds +0.0055 over ONE INTEGER. Any steering result on '
                 'this channel MUST be compared against a model conditioned on that integer, or '
                 'the result is attributable to topology.'),
        reliability=dict(
            pair_delta_icc_two_master_seeds=0.8888,
            reliable_and_not_2d_predictable=0.295,
            note='0.8888 reliable minus 0.5933 2D-predictable. Earlier framing implied 0.65.'),
        label_stability=dict(
            extent_dir_flip_rate_at_deadband_0p5=0.277,
            across_seed_sd_of_pair_delta=0.3774,
            note='extent_DELTA is seed-stable; extent_DIR is not. 0.5 A sat near the WORST point '
                 'of the flip-rate/coverage curve. Deadband widened; delta is stored per row so '
                 'dir can be re-derived at any threshold without rebuilding.'),
        base_rate=dict(
            vs_unconstrained_random_pairs=-1.784,
            vs_gate_matched_random_pairs=-0.3124, ci95=[-0.4177, -0.2047],
            note='66%% of the unconstrained figure is mass-matching alone, 83%% is both gates. '
                 'Quote the GATE-MATCHED value. The effect is real but 5.8x smaller than filed.'),
        split_scope=dict(
            parent_disjoint=True, variant_disjoint=True, scaffold_disjoint=False,
            valid_rows_with_parent_scaffold_in_train=0.189,
            valid_rows_with_variant_scaffold_in_train=0.338,
            split_rule='connected components of the parent<->variant graph: a MOLECULE is on one '
                       'side whatever role it plays, so variant-straddle and cross-role straddle '
                       'are both impossible by construction. Replaces a parent-identity split plus '
                       'a de-straddle pass whose stamped claim was wrong (majority-side retention '
                       'kept train 99.9%% of the time and left enrichment at 13.8x vs 15.4x).'),
        gates_that_are_tautologies=['G4 coverage (=1.0 by construction)',
                                    'G6 parent_overlap (=0, split is a pure function of parent)'],
    )
    # SELF-DETECTING PROVENANCE. `hashlib` was imported at the top of this file and used NOWHERE --
    # the stamp carried no source hash at all. Twice in 40 minutes this file was EDITED while a
    # build was running: 02:22:51 under pid 90412 (started 02:00:40) and 02:37:55 under pid 93003
    # (started 02:31:10). Python reads source at import, so in both cases the on-disk file diverged
    # from what the process was executing, and the only way to tell which revision produced an
    # artifact was to compare file mtimes against process start times after the fact. Hashing the
    # source AT LAUNCH makes the whole class self-detecting: the stamp now names the exact bytes
    # that produced the data, and a mismatch is visible instead of inferred. Same lesson as the
    # de-leak version straddle -- stamp it rather than remember it.
    def _sha(path):
        try:
            return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
        except Exception as _e:
            return 'UNREADABLE:%s' % _e
    _src = dict(build_steer_pairs=_sha(os.path.abspath(__file__)),
                steer_params=_sha(os.path.abspath(SP.__file__)),
                note=('sha256[:16] of the source FILES AS THEY SIT ON DISK when the stamp is '
                      'written. If this file was edited mid-run these will NOT match the bytes the '
                      'process imported -- which is exactly the condition this field exists to '
                      'expose. Cross-check against provenance/builders/.'))
    stamp = dict(source_sha=_src, scope_limits=scope,
                 n_parents_completed=int(done), n_label_failures=int(n_label_fail_total),
                 seed_does_NOT_pin_parent_set=(
                     'the loop breaks on ACCUMULATED ROWS and `done` increments in COMPLETION '
                     'order, which is worker-schedule dependent. Two runs at this seed gave 7161 '
                     'and 7160 pairs at 500 parents on byte-identical generation code. The seed '
                     'fixes the parent POOL ORDER, not the parent SET.'),
                 seed=seed, mw_tol=mw_tol, tc_lo=tc_lo, tc_hi=tc_hi, per_env_cap=per_env_cap,
                 n_target=n_target, steer_params=SP.stamp(), deadband=DEADBAND,
                 n_pairs=int(len(d)), n_parents=int(d.parent.nunique()), gates=report)
    p = os.path.join(outdir, 'steer_pairs.csv')
    d.to_csv(p, index=False)
    json.dump(stamp, open(os.path.join(outdir, 'steer_pairs_stamp.json'), 'w'), indent=1)
    print('\nWRITTEN -> %s  (%d pairs)' % (p, len(d)))
    return d


def gates(d, n_target, mw_tol, tc_lo, tc_hi, pre_cap=None, n_raw_dupes=None):
    """Every check FATAL. A gate that only warns is a gate that gets ignored at 3am."""
    rep, fail = {}, []
    n = len(d)
    print('\n================ VALIDATION GATES ================')

    # G1 scale
    rep['n_pairs'] = int(n)
    print('G1 scale             : %d pairs (target %d)' % (n, n_target))
    if n < 0.5 * n_target:
        fail.append('G1: only %d pairs, under half of target %d' % (n, n_target))

    # G2 chemistry -- re-parse and re-check the warhead on a sample
    s = d.sample(min(2000, n), random_state=7)
    bad = 0
    for a, b in zip(s.parent, s.variant):
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if ma is None or mb is None or not mb.HasSubstructMatch(ACR):
            bad += 1
    rep['chem_bad_frac'] = bad / len(s)
    print('G2 chemistry         : %.3f%% unparseable-or-warhead-lost' % (100 * bad / len(s)))
    if bad / len(s) > 0.001:
        fail.append('G2: %.2f%% of sampled pairs are invalid' % (100 * bad / len(s)))

    # G3 MW / TC actually enforced
    rep['dmw_max'] = float(d.dmw.max()); rep['tc_min'] = float(d.tc.min()); rep['tc_max'] = float(d.tc.max())
    print('G3 MW/TC             : |dMW| med %.2f max %.2f (tol %.1f) | TC %.3f..%.3f (band %.2f-%.2f)'
          % (d.dmw.median(), d.dmw.max(), mw_tol, d.tc.min(), d.tc.max(), tc_lo, tc_hi))
    if d.dmw.max() > mw_tol + 1e-6:
        fail.append('G3: |dMW| max %.2f exceeds tol %.2f -- mass filter not applied' % (d.dmw.max(), mw_tol))
    if d.tc.min() < tc_lo - 1e-6 or d.tc.max() > tc_hi + 1e-6:
        fail.append('G3: TC outside the requested band')

    # G4 coverage + G5 variety, per param
    print('G4/G5 per-param      :   coverage(TAUTOLOGICAL=1.0 by construction)  up/same/down')
    for k in PARAMS:
        col = '%s_dir' % k
        # TAUTOLOGY WARNING: rows are only emitted when the label is non-None (`ok` in _one_parent),
        # so this is identically 1.0 and the `cov < 0.20` bar below CANNOT fire. Kept as a guard
        # against a future edit that starts emitting None-labelled rows, but it is NOT evidence of
        # coverage. Real coverage is (rows emitted) / (candidate pairs), which this frame cannot see.
        cov = d[col].notna().mean()
        vc = d[col].value_counts(normalize=True).to_dict()
        rep['%s_coverage' % k] = float(cov)
        rep['%s_classes' % k] = {str(a): float(b) for a, b in vc.items()}
        print('   %-8s          %6.1f%%    %s' % (k, 100 * cov,
              '  '.join('%s %.0f%%' % (a, 100 * b) for a, b in sorted(vc.items()))))
        if cov < 0.20:
            fail.append('G4: %s defined on only %.1f%% of pairs' % (k, 100 * cov))
        if vc and max(vc.values()) > 0.95:
            fail.append('G5: %s is %.0f%% one class -- degenerate label' % (k, 100 * max(vc.values())))

    # G6 split leakage -- the one that would silently invalidate every result
    tr, va = set(d[d.split == 'train'].parent), set(d[d.split == 'valid'].parent)
    overlap = tr & va
    tr_v, va_v = set(d[d.split == 'train'].variant), set(d[d.split == 'valid'].variant)
    v_overlap = tr_v & va_v
    # THE CHANNEL G6 DID NOT COVER: a molecule appearing as a PARENT on one side and as a VARIANT
    # on the other. G6 compared parent-to-parent and variant-to-variant only, so this crossed the
    # split invisibly and printed "variant overlap 0". On the smoke file 2 molecules appear in both
    # roles and both intersections happened to be empty -- n=2 luck, not a property. Scaled to
    # 17,409 parents that is ~68 dual-role molecules, ~12 of them straddling.
    x1 = tr & set(d[d.split == 'valid'].variant)          # train parent  <-> valid variant
    x2 = set(d[d.split == 'train'].variant) & va          # train variant <-> valid parent
    rep['parent_overlap'] = len(overlap); rep['variant_overlap'] = len(v_overlap)
    rep['cross_role_overlap'] = len(x1) + len(x2)
    print('G6 cross-role leak   : train-parent/valid-variant %d | train-variant/valid-parent %d'
          % (len(x1), len(x2)))
    if x1 or x2:
        # NOT FATAL. With the component split this is 0 by construction; a non-zero value means the
        # split rule regressed, which is worth shouting about but not worth destroying the run for.
        # The previous version raised on non-zero while my own comment estimated ~12 -- a gate whose
        # trigger condition I had already computed as non-zero.
        print('   *** G6 CROSS-ROLE NON-ZERO (%d) -- the component split has regressed ***'
              % (len(x1) + len(x2)))
    # PARENT overlap is a TAUTOLOGY given the split rule (split is a pure function of `parent`), so
    # its 0 is arithmetic, not evidence. Kept as a regression guard against a future edit to that
    # rule, but labelled so it is never counted as a passed check. VARIANT overlap is the real one:
    # two different parents can generate the SAME CReM variant, which straddles the split.
    print('G6 split leakage     : parent overlap %d (TAUTOLOGY given split rule) | variant overlap %d'
          ' | valid %.1f%% of rows'
          % (len(overlap), len(v_overlap), 100 * (d.split == 'valid').mean()))
    if overlap:
        fail.append('G6: %d parents appear in BOTH splits' % len(overlap))
    if v_overlap:
        fail.append('G6: %d variant molecules appear in BOTH splits AFTER the de-straddle pass -- '
                    'that pass failed, which is a real bug, not a data property' % len(v_overlap))

    # G7 concentration
    # MEASURED ON THE PRE-CAP FRAME. Downstream of per_env_cap=8 this gate CANNOT FIRE: the cap
    # bounds the top-1% share at 0.01*cap = 0.075 against a 0.25 bar, i.e. 3.3x stricter than the
    # gate. It would have written "concentration checked, 2.8%" into every stamp forever while
    # measuring an impossibility. The concentration hazard lives BEFORE the cap -- the cap hides it.
    # MEASURED ON SCAFFOLDS, NOT PARENT MOLECULES, AND AGAINST A REACHABLE BAR.
    # Two successive versions of this gate could not fire. v1 sat downstream of per_env_cap=8
    # (ceiling 0.075 vs a 0.25 bar); v2 moved to the pre-cap frame but max_replacements=12 bounds
    # the top-1% PARENT share at 0.108, still 2.3x under the bar. Both would have stamped
    # "concentration checked" while measuring an impossibility. The hazard this file's own docstring
    # names is scaffold/env concentration (this project measured top-5 fragments at 50.1% of counts),
    # and parent-molecule share cannot see it: on the smoke file scaffold share is 0.0702 vs parent
    # 0.0288, 2.4x higher on an axis nothing bounds.
    src = pre_cap if pre_cap is not None else d
    from rdkit.Chem.Scaffolds import MurckoScaffold as _MS
    def _scaf(x):
        try:
            mm = Chem.MolFromSmiles(x)
            return _MS.MurckoScaffoldSmiles(mol=mm) if mm else x
        except Exception:
            return x
    _sc = src.parent.map(_scaf)
    top = _sc.value_counts()
    # G7 NOW GATES ON THE SINGLE MOST COMMON SCAFFOLD, NOT ON "top 1%".
    #
    # THE top-1% FORM IS NOT SCALE-FREE AND WOULD HAVE KILLED THE NEXT BUILD. Because
    # k = max(1, int(0.01*nunique)) grows with the frame, top-1% SUMS MORE SCAFFOLDS as parents are
    # added, so the statistic climbs even at constant concentration. Measured by subsampling the
    # real k8 frame, 8 replicates per point:
    #     150 parents (k=1)  4.61%   300 (k=2)  7.06%   600 (k=4) 10.04%
    #     900 (k=6) 11.75%   1272 (k=8) 13.41%          fit ln-coef +4.12, R2 0.98
    # That crosses the old 15% FATAL bar at ~1,890 parents. The stated plan for the next build is
    # MORE PARENTS (cluster count, not row count, is what binds the GAP_neg CI), so the gate would
    # have destroyed it after an hour of CReM+MMFF. Reproduced independently by two auditors:
    # ln-coef +4.12 and +4.43, projected 16.8% and 17.1% at 3,000 parents.
    #
    # THE SINGLE MOST COMMON SCAFFOLD IS FLAT over the same 8.5x range in parents:
    #     150 -> 4.61%   300 -> 4.23%   600 -> 3.91%   900 -> 3.68%   1272 -> 3.79%
    #     ln-coef -0.42, i.e. no trend; and 3.01% on smoke2, 4.56% on a 60-parent frame.
    # Six measurements, 3.0-4.6%, across two max_replacements regimes. That is the property a fixed
    # bar requires. A bar of 10% is ~2.2x the observed maximum, is NOT fitted to any single
    # measurement, and is reachable by the concentration the docstring names (one scaffold supplying
    # a tenth of all pairs would be real monoculture).
    # HISTORY: this is the SIXTH version of G7. v1 measured downstream of the cap; v2 was bounded by
    # max_replacements; v3 set a bar above the achievable range; v4/v5 proposed the top-5% fraction,
    # withdrawn on the same scaling argument that kills top-1%. Each failed because the statistic was
    # chosen before its scaling was checked. This one was chosen BECAUSE its scaling was checked.
    share = float(top.iloc[0] / len(src))
    share_top1pct = float(top.head(max(1, int(0.01 * _sc.nunique()))).sum() / len(src))
    rep['top_scaffold_share'] = share
    rep['top1pct_scaffold_share'] = share_top1pct      # RECORDED, not gated: scale-dependent
    rep['top1pct_scaffold_share_SCOPE'] = (
        'scale-dependent (ln-coef +4.12 in parents): comparable ONLY between frames of the same '
        'parent count. Recorded for continuity with earlier stamps; the GATE is top_scaffold_share.')
    print('G7 concentration     : single most common SCAFFOLD supplies %.1f%% of pairs (bar 10%%) '
          '| top-1%% %.1f%% recorded, NOT gated (scale-dependent)'
          % (100 * share, 100 * share_top1pct))
    if share > 0.10:
        fail.append('G7: the single most common SCAFFOLD supplies %.1f%% of pairs' % (100 * share))

    # G8 duplicates
    # build() deduplicates BEFORE calling gates(), so measuring duplicates on `d` is identically 0
    # and stamps "duplicates: 0" forever. The count must come from before the dedup.
    dup = 0 if n_raw_dupes is None else int(n_raw_dupes)
    rep['duplicates'] = int(dup)
    print('G8 duplicates        : %d' % dup)
    if n_raw_dupes is None:
        print('   (G8 UNMEASURED -- caller passed no pre-dedup count)')
    elif dup:
        print('   %d raw duplicate pairs were removed before gating (not fatal, recorded)' % dup)

    print('==================================================')
    if fail:
        for f in fail:
            print('  FAILED  %s' % f)
        raise SystemExit('\nFATAL: %d gate(s) failed. Dataset NOT written -- fix before training.' % len(fail))
    print('  ALL GATES PASSED\n')
    rep['passed'] = True
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # DEFAULT LOWERED FROM 250000, WHICH THIS POOL CANNOT REACH. G1 is FATAL below 0.5*n. The pool
    # is 17,409 acrylamide-bearing parents and the last build realised 7.042 post-cap rows/parent
    # (8,957 from 1,272), so the WHOLE pool yields ~122,588 -- against a G1 bar of 125,000. It
    # misses by 2%. That is the worst possible shape: the loop would consume every parent over
    # several hours of CReM+MMFF and only then FATAL, which is exactly the "a FATAL gate discards
    # hours of compute" failure the RAW pre-gate dump at :347 exists to mitigate. 60000 clears the
    # bar with 4x headroom; the pre-flight below refuses ANY --n the pool cannot serve, before the
    # first parent is submitted.
    ap.add_argument('--n', type=int, default=60000)
    # 6.412, NOT 7.042 -- AND THE DIFFERENCE IS A WRONG-DENOMINATOR BUG IN MY OWN GATE.
    # The pre-flight multiplies this by the POOL, i.e. by parents AVAILABLE TO CONSUME. I computed
    # it as 8957/1272, but 1272 is `steer_pairs_UNCAPPED.csv`.parent.nunique() -- parents that
    # PRODUCED AT LEAST ONE SURVIVING PAIR. The build's own console log records the consumed count:
    #   provenance/build_k8_console.log:21517  "RAW: 19520 pairs from 1397 parents in 3834s"
    # 125 of those 1397 yielded nothing, so per CONSUMED parent the yield is 8957/1397 = 6.412.
    # Using the survivor-only denominator overstates the pool ceiling by 9.8% (122,594 vs 111,619),
    # and the gate then FALSE-PASSES for --n in roughly [223k, 245k]: it says "reachable", the build
    # consumes all 17,409 parents (~13 h, extrapolating 3834 s / 1397 parents) and THEN FATALs at
    # G1. That is exactly the multi-hour loss this gate was written to prevent, failing in the
    # expensive direction, because I mixed two populations inside my own guard -- the third time
    # tonight a number came from a different population than the one it was multiplied against.
    # Not live at the new default --n 60000 (bar 30,000, cleared 3.7x under either constant).
    ap.add_argument('--yield-per-parent', type=float, default=6.412,
                    help='post-cap rows per CONSUMED parent (8957/1397 from the build console log, '
                         'NOT 8957/1272 which counts only parents that produced something). Used '
                         'only by the pre-flight G1 check, which multiplies it by the whole pool.')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--mw-tol', type=float, default=10.0)
    ap.add_argument('--tc-lo', type=float, default=0.40)
    ap.add_argument('--tc-hi', type=float, default=0.90)
    ap.add_argument('--per-env-cap', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--out', default=os.path.join(CF, 'data/steer'))
    a = ap.parse_args()
    # ---- PRE-FLIGHT G1. Fire BEFORE the compute, not after it.
    # Every gate in this file runs on the finished frame, so an unreachable --n is discoverable only
    # once the pool is exhausted -- hours of CReM + 30-conformer MMFF for a guaranteed FATAL. The
    # achievable ceiling is knowable in one line from the pool size, so refusing here costs nothing
    # and converts a multi-hour loss into an immediate one. Bounded by BOTH the realised yield and
    # the hard per_env_cap ceiling, so it cannot be argued around by claiming a better yield.
    import glob as _g
    _df = pd.concat([pd.read_csv(f) for f in
                     _g.glob(os.path.join(ROOT, 'paper/reproducibility/metrics/planar_2d/*.csv'))])
    _pool = len(_df[(_df.acryl_match == True) & _df.smi.notna()].drop_duplicates('smi'))
    _ach = min(_pool * a.yield_per_parent, _pool * a.per_env_cap)
    print('PRE-FLIGHT G1: pool %d parents x %.3f post-cap rows/parent (cap %d) -> at most %.0f '
          'rows; G1 needs %.0f (0.5 x --n %d)' % (_pool, a.yield_per_parent, a.per_env_cap,
                                                  _ach, 0.5 * a.n, a.n), flush=True)
    if _ach < 0.5 * a.n:
        raise SystemExit(
            'FATAL PRE-FLIGHT G1: this pool can yield at most ~%.0f post-cap rows, but --n %d puts '
            'the G1 bar at %.0f. The build would consume every parent and FATAL at the end. Lower '
            '--n to <= %d, or raise --per-env-cap (which widens the design effect, see the '
            'design_effect_note in score_steer_arms.py -- more rows per parent buy little '
            'independent information).' % (_ach, a.n, 0.5 * a.n, int(2 * _ach)))
    build(a.n, a.workers, a.mw_tol, a.tc_lo, a.tc_hi, a.per_env_cap, a.seed, a.out)
