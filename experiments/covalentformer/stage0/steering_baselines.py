"""THE FOUR BASELINES ANY OBEDIENCE NUMBER MUST BE QUOTED AGAINST.

WHY THIS FILE EXISTS. A QA audit measured that steering direction is predictable from the INPUT
MOLECULE ALONE at ~0.62-0.70 on attach_path, against a majority-class floor of 0.5106. Every
obedience figure quoted against 51.1% is therefore overstated by 12-19 points, and an obedience
number below ~0.70 on attach_path demonstrates NOTHING. That audit ran from /tmp. A harness that
lives only in /tmp is the unsourced-artifact defect (#84/#139): the number outlives the code that
made it and then cannot be checked. So the baseline computation lives HERE, in the repo, and the
scorer is expected to call it before printing any obedience figure.

THE MECHANISM, because it is not obvious that an INPUT alone should predict a DIRECTION.
It is regression to the mean. Pairs are (input, output) with a label of UP or DOWN on some integer
property. A molecule whose own value is LOW has more room above it than below, so its partner is
more often higher -- the label is partly a restatement of where the input sits in the distribution.
Stratification fixes this for the INTEGER (baseline C lands at or below chance) but does NOT fix it
for the MOLECULE (baseline A stays high), because the fingerprint encodes the value the integer was
binned from, plus everything else about the molecule.

THE FOUR BASELINES, and what each one rules out:
  A  input-molecule-only ECFP4        THE FLOOR THAT MATTERS. Obedience must beat this, not 0.5.
  B  shuffled-label control           Must land ~0.500. If it does not, the HARNESS is broken and
                                      every other row on the table is uninterpretable. This is the
                                      only baseline that can invalidate the others.
  C  stratifying integer alone        Should be ~chance if stratification worked. Checks the fix.
  D  both sides concatenated          A sanity CEILING, not a target. Should be high; if it is not,
                                      the label is not learnable at all and nothing else matters.
Plus, requested specifically because ECFP4 alone is not the only free descriptor:
  A2 scaffold identity alone          Does the Bemis-Murcko scaffold carry the direction on its own?
  A3 bond count alone                 ONE INTEGER has out-predicted ECFP4 four times on this
                                      project. It is checked every time now.
  A4 ECFP4 + scaffold + bond count    The combined free-descriptor floor.

DISCIPLINE. One learner, one seed, one split, fixed everywhere -- LogisticRegression(liblinear,
C=1, random_state=0). Magnitudes move with learner strength (a prior pass got 0.6243 where a
stronger learner got 0.6980), so the learner is STAMPED into the output and two numbers from
different learners must never be compared. Nothing here samples, so there is no replication to do;
the shuffled control uses a seeded RNG so it replicates exactly.
"""
import os, sys, json, argparse
import numpy as np
from scipy import sparse

# EVERYTHING IS SPARSE. The scaffold one-hot is ~30,543 columns with exactly ONE set per row --
# 99.997% zeros -- and ECFP4 is 2048 bits of which ~50 are on. Built dense and handed to sklearn,
# the A4 block (ECFP4 + scaffold + bonds) is 92,954 x 32,592, which sklearn upcasts to float64:
#   v3   24.24 GB      v4d   82.17 GB
# I watched one of these reach 40.9 GB RSS before I killed it; it would have swapped the machine
# and taken the three training arms down with it. An independent auditor's harness was "sparse
# rather than dense" -- not a style preference, it is the reason theirs returned and mine did not.

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog('rdApp.*')

# SOLVER CHANGED liblinear -> lbfgs. liblinear is BINARY-ONLY: on a 3-class corpus it
# raises 'does not support multiclass classification (n_classes >= 3)'. I shipped the
# 3-class rewrite while leaving the binary-only solver in place, so the very first
# multi-class fit died and I reported the run as 'computing' when it was already dead.
# CONSEQUENCE FOR COMPARISONS: this is a DIFFERENT LEARNER, and this file's own
# docstring says magnitudes move with learner strength and two numbers from different
# learners must never be compared. So every v4c/v4d figure from here is lbfgs, and the
# v3 figures (0.6544 etc.) are liblinear. To compare v3 against v4d, v3 must be RE-RUN
# under lbfgs. The stamp below is what makes that checkable rather than silent.
# The stamp is DERIVED from MAX_ITER, not retyped. It said max_iter=1000 for the first
# minutes after MAX_ITER moved to 20000 -- a stamp one revision behind the code is the
# exact failure mode that cost 170 s of RDKit to answer a provenance question tonight.
LEARNER = 'LogisticRegression(solver=lbfgs, C=1, random_state=0, max_iter=%d)'
# (the old module-level UP=('UP','CHANGED') was deleted: after the 3-class rewrite it was
#  never read, but it still READ as live configuration, which is how a reviewer concludes
#  the binariser is present when it is not.)


# THE ORIGINAL BINARISED A THREE-CLASS CORPUS AND ITS OWN CONTROLS STILL "PASSED".
# `1 if d in UP else 0` with UP=('UP','CHANGED') silently folded DOWN and SAME into one class. On
# v3 that was harmless (two labels). On v4c, where SAME is a THIRD steering instruction and 35.7%
# of rows carry it, it reported a BINARY floor for a THREE-CLASS problem -- and the shuffled
# control still landed near 0.50, so the harness looked healthy while measuring the wrong thing.
# A guard that returns PASS because its input has been silently reshaped is worse than no guard.
LABELS_ORDER = ('DOWN', 'SAME', 'UP', 'CHANGED')


def load(path, param):
    """Rows with a usable direction label, as (input_smiles, output_smiles, y, strat_key).

    y is now the CLASS INDEX over whatever labels the file actually contains, not a forced 0/1."""
    key = param + '_dir'
    ak = param + '_a'
    raw = []
    for line in open(path):
        r = json.loads(line)
        d = r.get(key)
        if d is None:
            continue
        raw.append((r['input_smiles'], r['output_smiles'], d, r.get(ak)))
    return raw


def encode_classes(train_raw, valid_raw):
    """Class index built from the TRAIN labels; a valid-only label would otherwise shift indices."""
    labs = sorted({r[2] for r in train_raw}, key=lambda x: (LABELS_ORDER.index(x)
                                                           if x in LABELS_ORDER else 99, x))
    idx = {l: i for i, l in enumerate(labs)}
    tr = [(a, b, idx[d], k) for a, b, d, k in train_raw if d in idx]
    va = [(a, b, idx[d], k) for a, b, d, k in valid_raw if d in idx]
    return tr, va, labs


def fp_matrix(smis, gen):
    rows, cols = [], []
    bad = 0
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            continue
        on = np.nonzero(np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.int8))[0]
        rows.extend([i] * len(on))
        cols.extend(on.tolist())
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(smis), 2048))
    return X, bad


def scaffold_onehot(smis, vocab=None):
    """Bemis-Murcko scaffold as a one-hot over scaffolds SEEN IN TRAIN. Unseen -> all-zero row,
    which is the honest encoding: a scaffold the model never saw carries no information."""
    scs = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        try:
            scs.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else '')
        except Exception:
            scs.append('')
    if vocab is None:
        vocab = {s: i for i, s in enumerate(sorted(set(scs)))}
    rows, cols = [], []
    for i, s in enumerate(scs):
        j = vocab.get(s)
        if j is not None:
            rows.append(i)
            cols.append(j)
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(scs), max(len(vocab), 1)))
    return X, vocab


# BOND COUNT IS SCALED. It is the ONLY non-{0,1} feature in the whole harness -- ECFP4 bits and
# both one-hots are binary -- and on its raw scale (tens of bonds) it dominates the lbfgs gradient
# and the A4 block FAILS TO CONVERGE at max_iter=1000. That is not a cosmetic warning: on v4c
# attach_path it INVERTS the row's meaning.
#     A  (ECFP4 alone)          0.4043   converged in 147 iters
#     A4 (ECFP4+scaffold+bonds) 0.3970   HIT max_iter, NOT converged   -> reads "bonds HURT"
#     A4 with bondcount/100     0.4059   converged in 427 iters        -> reads "bonds help"
# A4 minus A flips sign, -0.0073 -> +0.0016. A4 exists precisely to ask whether the free
# descriptors ADD over ECFP4, so an unconverged A4 answers that question backwards. A is
# unaffected (it never contains this column), so the quoted FLOOR is unharmed -- but A4 was wrong
# on both v3 and v4c and I printed it as a finished number.
BOND_SCALE = 100.0


def bondcount(smis):
    v = np.zeros((len(smis), 1), dtype=np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        v[i, 0] = (m.GetNumBonds() if m else 0) / BOND_SCALE
    return sparse.csr_matrix(v)


# CONVERGENCE IS NOW RECORDED, NOT ASSUMED. An auditor argued that BOND_SCALE breaks lbfgs and
# that the fix is to drop it. I measured all four scalings myself on v3 attach_path (cap 20,000,
# max_iter 1,000, everything else identical; /tmp/bondscale_check.py):
#     A   ECFP4 alone            0.606900   117 iters   CONVERGED
#     A4  raw bond count         0.609000  1000 iters   HIT max_iter
#     A4  bonds/100  (SHIPPED)   0.609800   191 iters   CONVERGED
#     A4  bonds*100              0.612400  1000 iters   HIT max_iter
#     A4  bonds z-scored         0.609000   189 iters   CONVERGED
# BOND_SCALE = 100.0 DIVIDES (line 165: GetNumBonds() / BOND_SCALE), so the shipped column is
# ~0.3 and converges; the auditor's "x100" is the MULTIPLIED column, the opposite regime, and
# dropping BOND_SCALE restores the RAW column, which is the one that fails. So the recommendation
# was aimed the wrong way -- but the finding UNDER it is real and is fixed here: nothing in this
# harness ever checked whether a fit converged, and an unconverged A4 is exactly how the "bonds
# HURT" sign inversion got printed as a finished number (see the BOND_SCALE block above). A
# max_iter hit is now carried out of fit() and into the JSON, and the FLOOR takes its max only
# over CONVERGED arms -- an unconverged accuracy has no fixed sign of bias, so it must not be
# allowed to win a max.
MAX_ITER = 20000


def fit(Xtr, ytr, Xva, yva):
    """(accuracy, n_iter, converged). NaN accuracy for an empty feature block."""
    if Xtr.shape[1] == 0:
        return float('nan'), 0, True
    clf = LogisticRegression(solver='lbfgs', C=1, random_state=0, max_iter=MAX_ITER)
    clf.fit(Xtr, ytr)
    n_iter = int(clf.n_iter_[0])
    return float((clf.predict(Xva) == yva).mean()), n_iter, n_iter < MAX_ITER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--param', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--seed', type=int, default=0)
    # BOUNDED COST, AND THE BOUND IS REPORTED. The full v4d run is 3.4x the rows of v3 AND 3-class
    # (one-vs-rest = 3x the fits), i.e. ~10x the work, ~80 min under contention -- for a floor that
    # a 60k-row subsample estimates to about +-0.002. A cap that is not printed reads as full
    # coverage (#184); this one is printed in the header and written into the JSON.
    ap.add_argument('--max-train', type=int, default=0, help='0 = all rows; else seeded subsample')
    a = ap.parse_args()

    tr_raw = load(os.path.join(a.split_dir, '%s_train.jsonl' % a.param), a.param)
    va_raw = load(os.path.join(a.split_dir, '%s_valid.jsonl' % a.param), a.param)
    if not tr_raw or not va_raw:
        print('FATAL: empty split for %s' % a.param)
        return 2
    tr, va, LABELS = encode_classes(tr_raw, va_raw)
    # THE CAP MUST BIND BOTH SIDES. My first version capped TRAIN only, and valid is 78,363 rows
    # of Bemis-Murcko scaffold perception at ~2ms each -- so a run I believed was bounded to 8,000
    # rows still spent minutes on the uncapped valid set, produced no output (the header prints
    # after the matrices), and looked exactly like a hang. That is the fourth failure of this one
    # measurement and the third that was my own harness rather than the data. Valid is capped at
    # the same seed, and BOTH pre-cap sizes are reported so the shrink is never invisible.
    n_train_full, n_valid_full = len(tr), len(va)
    if a.max_train:
        import random as _r
        if len(tr) > a.max_train:
            tr = _r.Random(a.seed).sample(tr, a.max_train)
        _vcap = max(2000, a.max_train // 2)
        if len(va) > _vcap:
            va = _r.Random(a.seed + 1).sample(va, _vcap)
    K = len(LABELS)
    ytr = np.array([r[2] for r in tr])
    yva = np.array([r[2] for r in va])

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    Atr, bad_tr = fp_matrix([r[0] for r in tr], gen)
    Ava, bad_va = fp_matrix([r[0] for r in va], gen)
    Otr, _ = fp_matrix([r[1] for r in tr], gen)
    Ova, _ = fp_matrix([r[1] for r in va], gen)

    Str, vocab = scaffold_onehot([r[0] for r in tr])
    Sva, _ = scaffold_onehot([r[0] for r in va], vocab)
    Btr, Bva = bondcount([r[0] for r in tr]), bondcount([r[0] for r in va])

    # the stratifying integer, one-hot over values seen in train
    kt = [r[3] for r in tr]
    kv = [r[3] for r in va]
    kvals = {k: i for i, k in enumerate(sorted({x for x in kt if x is not None},
                                              key=lambda z: str(z)))}
    def kmat(ks):
        rr, cc = [], []
        for i, k in enumerate(ks):
            j = kvals.get(k)
            if j is not None:
                rr.append(i)
                cc.append(j)
        return sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                                 shape=(len(ks), max(len(kvals), 1)))
    Ktr, Kva = kmat(kt), kmat(kv)

    rng = np.random.RandomState(a.seed)
    y_shuf = ytr.copy()
    rng.shuffle(y_shuf)

    # TWO DIFFERENT FLOORS, AND THEY ARE NOT THE SAME NUMBER. Chance (1/K) is what a coin-flipping
    # classifier gets; the MAJORITY floor is what a classifier that always predicts the biggest
    # class gets. The shuffled control converges to the MAJORITY floor, NOT to 1/K, because
    # logistic regression on scrambled labels still learns the prior. Quoting chance as "the null"
    # for a shuffled control is how a broken harness passes its own check.
    counts = np.bincount(yva, minlength=K).astype(float)
    maj = float(counts.max() / counts.sum())
    chance = 1.0 / K
    # Each fitted row now carries (accuracy, n_iter, converged); the two ANALYTIC rows carry
    # (value, 0, True) because there is no optimiser behind them and they cannot fail to converge.
    rows = [
        ('chance (1/K, K=%d)' % K, (chance, 0, True)),
        ('majority-class floor (valid)', (maj, 0, True)),
        ('B  shuffled-label control', fit(Atr, y_shuf, Ava, yva)),
        ('C  stratifying integer alone', fit(Ktr, ytr, Kva, yva)),
        ('A2 scaffold identity alone', fit(Str, ytr, Sva, yva)),
        ('A3 bond count alone (ONE INTEGER)', fit(Btr, ytr, Bva, yva)),
        ('A  INPUT-ONLY ECFP4  <- THE FLOOR', fit(Atr, ytr, Ava, yva)),
        ('A4 ECFP4 + scaffold + bonds', fit(sparse.hstack([Atr, Str, Btr]).tocsr(), ytr,
                                            sparse.hstack([Ava, Sva, Bva]).tocsr(), yva)),
        ('D  both sides concat (ceiling)', fit(sparse.hstack([Atr, Otr]).tocsr(), ytr,
                                               sparse.hstack([Ava, Ova]).tocsr(), yva)),
    ]

    print('=== STEERING BASELINES: %s ===' % a.param)
    print('learner: %s' % (LEARNER % MAX_ITER))
    print('n_train %d%s | n_valid %d | invalid SMILES %d/%d'
          % (len(tr), ('  (SUBSAMPLED from %d, seed %d)' % (n_train_full, a.seed))
             if a.max_train and n_train_full > a.max_train else '', len(va), bad_tr, bad_va))
    if a.max_train and n_valid_full > len(va):
        print('  valid SUBSAMPLED from %d -> %d (seed %d)' % (n_valid_full, len(va), a.seed + 1))
    print('scaffold vocab %d | strat-integer levels %d' % (len(vocab), len(kvals)))
    for nm, (v, it, ok) in rows:
        print('  %-38s %.4f%s' % (nm, v,
              '' if ok else '   *** HIT max_iter=%d, NOT CONVERGED -- excluded from the floor ***'
              % MAX_ITER))

    # THE FLOOR IS THE MAX OVER EVERY BASELINE THAT A MODEL COULD BEAT WITHOUT STEERING -- NOT
    # JUST ECFP4. The original took max(A, majority), i.e. two of seven rows, and DISCARDED A4
    # (ECFP4+scaffold+bonds) which this file's own docstring calls "The combined free-descriptor
    # floor". On both artifacts on disk A4 BEATS A:
    #     baselines_attach_path.json        A 0.6543592  A4 0.6678286  quoted 0.6543592
    #     baselines_attach_path_lbfgs.json  A 0.6550047  A4 0.6686462  quoted 0.6550047
    # So the quoted floor was 1.35-1.36 points too LOW, and any obedience figure in
    # [0.6544, 0.6678] was being certified as beating a floor it does not beat. The correct,
    # higher number was printed two lines above the wrong one in every single run.
    # INCLUDED: A, A2, A3, A4, C, majority -- all reachable without using the steering instruction.
    # EXCLUDED: chance (not achievable-by-a-model), B (shuffled, a harness check), D (both-sides
    # ceiling -- it SEES the answer, so it is an upper bound, not a floor).
    # AND THE MAX IS TAKEN OVER CONVERGED ARMS ONLY. An unconverged fit's bias has no fixed sign
    # (measured: +0.0028/+0.0045/+0.0282 on three corpora, -0.0033 on a fourth), so letting one
    # win a max is letting optimiser noise set the bar a steering result must clear. majority is
    # always eligible -- it is a count, not a fit -- so the floor can never be empty.
    _d = {k: v for k, (v, _i, _ok) in rows}
    _ok = {k: ok for k, (_v, _i, ok) in rows}
    _floor_keys = [k for k in _d if (k.startswith(('A ', 'A2', 'A3', 'A4', 'C '))
                                     or k.startswith('majority'))
                   and _ok[k] and _d[k] == _d[k]]
    _dropped = [k for k, (v, _i, ok) in rows
                if (k.startswith(('A ', 'A2', 'A3', 'A4', 'C ')) or k.startswith('majority'))
                and not ok]
    A = _d['A  INPUT-ONLY ECFP4  <- THE FLOOR']
    B = _d['B  shuffled-label control']
    floor = max([_d[k] for k in _floor_keys] + [maj])
    floor_src = max(((_d[k], k) for k in _floor_keys), key=lambda t: t[0])[1] \
        if _floor_keys else 'majority-class floor (valid)'
    if _dropped:
        print('')
        print('  EXCLUDED FROM THE FLOOR (hit max_iter=%d, bias has no fixed sign): %s'
              % (MAX_ITER, ', '.join(s.strip() for s in _dropped)))
    print('')
    # The shuffled control must sit at the MAJORITY floor, not at chance. Tolerance is +-0.04.
    if abs(B - maj) > 0.04:
        print('  HARNESS BROKEN: shuffled control is %.4f but the majority floor is %.4f.'
              ' Every row above is uninterpretable. Do NOT quote any of these numbers.' % (B, maj))
    else:
        print('  shuffled control %.4f sits at the majority floor %.4f -- table is interpretable.'
              % (B, maj))
    print('  classes (%d): %s   counts %s' % (K, list(LABELS), counts.astype(int).tolist()))
    print('  QUOTE OBEDIENCE AGAINST %.4f  (highest free baseline: %s), NOT AGAINST %.4f.'
          % (floor, floor_src.strip(), maj))
    print('  An obedience figure at or below %.4f demonstrates nothing about steering.' % floor)

    if a.out:
        json.dump(dict(param=a.param, learner=LEARNER % MAX_ITER, seed=a.seed,
                       n_train_full=n_train_full, n_valid_full=n_valid_full,
                       max_train_cap=a.max_train,
                       n_train=len(tr), n_valid=len(va),
                       invalid_train=bad_tr, invalid_valid=bad_va,
                       baselines={k: v for k, (v, _i, _o) in rows},
                       # n_iter and converged are recorded per arm so a future reader can tell a
                       # real number from an optimiser artifact WITHOUT re-running the fit --
                       # which is exactly what the A4 sign inversion cost before.
                       n_iter={k: i for k, (_v, i, _o) in rows},
                       converged={k: bool(o) for k, (_v, _i, o) in rows},
                       max_iter=MAX_ITER,
                       floor_to_quote=floor, floor_source=floor_src.strip()),
                  open(a.out, 'w'), indent=2)
        print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
