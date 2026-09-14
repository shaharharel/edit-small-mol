"""TIER A v3: a dataset where the TARGET DEPENDS ON THE REQUIREMENT.

WHY v1 AND v2 ARE BOTH VOID. In those builds the anchor and the target were fixed per pair
(anchor = scaffold+frag_j, target = scaffold+frag_i) and only (d, theta, label) varied across the
20 rows. Measured on the produced file: 19,473 of 19,572 anchors (99.49%) had exactly ONE distinct
target. So for essentially every anchor, all twenty rows -- ten r=1 and ten r=0, at ten different
gaps -- demanded the SAME output string.

    I(target ; d, cos_theta, r | anchor) = 0

Cross-entropy is then fully minimised by learning anchor -> target and ignoring both conditioning
rows. Three consequences, all fatal to the experiment as it stood:
  - training optimises a copy function, whatever the architecture does;
  - the pre-registered shuffle test CANNOT discriminate, because permuting g leaves the loss-optimal
    target unchanged for 99.5% of rows. A null result was guaranteed BY THE DATA, independent of
    the model. It was not a valid kill criterion on that file;
  - conditioning on r=0 at inference could never work: the model was never once shown a different
    answer for r=0.

This is precisely the defect we criticised in CovaCraft-v2-cond -- a conditioning channel with
variance but no information -- arrived at by a different route. The theta-matched-negative fix does
not touch it; both builds share the flaw.

THE FIX. Hold the anchor fixed and make the TARGET a function of the requirement:

    for a scaffold S and a sampled gap g:
        r=1  ->  target = S + a fragment whose envelope CONTAINS g
        r=0  ->  target = S + a fragment whose envelope MISSES   g

Same anchor, same scaffold; change g and the correct answer changes. Now the only way to place
probability on the right string is to read g, and r genuinely selects between two different outputs,
so bidirectional control (H3) becomes a meaningful question rather than a vacuous one.

The gap is drawn from the union of the candidate fragments' envelopes so that BOTH a reacher and a
misser exist among them -- otherwise the pair cannot be formed and the gap is discarded.

SPLIT IS FRAGMENT-DISJOINT AS WELL AS SCAFFOLD-DISJOINT. In v1 the label depended only on
(fragment, d, theta), the scaffold never entered it, and the split was scaffold-based -- so 99.5% of
validation fragments also appeared in training and 38,360 of 38,520 validation rows had their exact
(fragment, d, theta) present in train with an identical label. That validation set measured
generalisation to new scaffolds and could not measure the geometry mapping at all. Holding out
FRAGMENTS is what makes a reach metric on the validation split mean something.
"""
import os, sys, csv, json, random, argparse, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tierA import join, null_battery

D_TOL, A_TOL = 0.5, 20.0
GAPS_PER_SCAFFOLD = 14
SEED = 20260913


def reaches(dd, tt, d, th):
    return bool(np.any((np.abs(dd - d) <= D_TOL) & (np.abs(tt - th) <= A_TOL)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='experiments/covalentformer/data/envelopes.jsonl')
    ap.add_argument('--meta', default='experiments/covalentformer/data/meta.csv')
    ap.add_argument('--out', default='experiments/covalentformer/data/tierA_v3')
    ap.add_argument('--n-scaf', type=int, default=2500)
    ap.add_argument('--pool', type=int, default=14, help='candidate fragments per scaffold')
    # PROPERTY GATE. Measured on the shipped v3 build: joining a scaffold to an ARBITRARY pool
    # fragment drifts the targets well off real covalent chemistry --
    #   tierA_v3 targets      QED 0.28  MW 578.5  logP 4.46  heavy 41.3
    #   real CovInDB parents  QED 0.39  MW 494.3  logP 4.08  heavy 35.4
    # Both pieces are real; their cross-product is not a molecule anyone made. The model reproduces
    # its targets exactly (QED 0.28, MW 571), so this is a DATA defect I introduced, and it is the
    # Tier-3 property drift the eval paradigm explicitly warns about. Capping MW keeps recombinations
    # inside the distribution the fragments were cut from.
    ap.add_argument('--max-mw', type=float, default=0.0,
                    help='drop joined targets above this MW (0 = off, reproduces v3)')
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)

    envs = {}
    for line in open(a.env):
        e = json.loads(line)
        envs[e['frag']] = (np.array(e['d'], float), np.array(e['theta'], float), e.get('cls', '?'))
    frags = sorted(envs)
    print('envelopes: %d fragments' % len(frags))

    meta = list(csv.DictReader(open(a.meta)))
    scafs = sorted({r['scaffold'] for r in meta})
    par = {}
    for r in meta:
        par.setdefault(r['scaffold'], r.get('parent_scaffold', ''))
    pyrng.shuffle(scafs)
    scafs = scafs[:a.n_scaf]

    rows = []
    bal = collections.Counter()   # fragment -> (times used as r=1) minus (times used as r=0)
    stats = collections.Counter()
    for sc in scafs:
        pool = pyrng.sample(frags, min(a.pool, len(frags)))
        built = {}
        for f in pool:
            full = join(sc, f)
            if full is None:
                continue
            if a.max_mw:
                mj = Chem.MolFromSmiles(full)
                if mj is None or Descriptors.MolWt(mj) > a.max_mw:
                    stats['dropped_mw'] += 1
                    continue
            built[f] = full
        if len(built) < 4:
            stats['scaffold_too_few_joins'] += 1
            continue
        keys = sorted(built)
        # the anchor is ONE fixed molecule on this scaffold; it never changes with g
        af = keys[0]
        anchor = built[af]
        cand = keys[1:]
        made = 0
        for _ in range(GAPS_PER_SCAFFOLD * 4):
            if made >= GAPS_PER_SCAFFOLD:
                break
            # draw the gap from a RANDOM CANDIDATE'S OWN ENVELOPE, so at least one reacher is
            # guaranteed to exist; uniform sampling over (d, theta) mostly produces gaps no
            # fragment in the pool can satisfy and the pair is wasted.
            src = cand[int(rng.integers(len(cand)))]
            dd, tt, _c = envs[src]
            j = int(rng.integers(len(dd)))
            d = float(dd[j] + rng.normal(0, 0.2))
            th = float(tt[j] + rng.normal(0, 5.0))
            if not (1.5 <= d <= 13.0 and 0.0 <= th <= 180.0):
                continue
            hit = [f for f in cand if reaches(envs[f][0], envs[f][1], d, th)]
            miss = [f for f in cand if f not in hit]
            if not hit or not miss:
                stats['no_contrast'] += 1
                continue
            # GREEDY PER-FRAGMENT BALANCING. Picking uniformly from `hit` and `miss` reintroduces a
            # descriptor leak the first v3 draft measured at logreg 0.604 (HBA 0.569, fsp3 0.565):
            # across fragments, the floppy high-rotatable-bond ones land in `hit` far more often, so
            # "is this fragment flexible" predicts the label without any geometry being read.
            # v1 avoided this by giving every fragment exactly 10 positives and 10 negatives; that
            # construction is unavailable here because reacher/misser is decided by the gap. Instead
            # drive each fragment toward balance greedily: among reachers prefer whichever is
            # currently most NEGATIVE-heavy, among missers whichever is most POSITIVE-heavy. Over
            # the build every fragment converges to roughly equal counts, so fragment identity stops
            # carrying label information while the target still depends on g.
            fp = min(hit, key=lambda f: (bal[f], rng.random()))
            fn = max(miss, key=lambda f: (bal[f], rng.random()))
            bal[fp] += 1
            bal[fn] -= 1
            ct = round(float(np.cos(np.radians(th))), 4)
            for f, lab in ((fp, 1), (fn, 0)):
                rows.append(dict(anchor=anchor, target=built[f], scaffold=sc, fragment=f,
                                 anchor_fragment=af, d=round(d, 3), cos_theta=ct,
                                 theta=round(th, 2), v_free=1.0, label=lab,
                                 warhead_class=envs[f][2], parent_scaffold=par.get(sc, '')))
            made += 1
        stats['scaffolds_used' if made else 'scaffold_no_pairs'] += 1
        stats['pairs'] += made

    print('rows %d from %d scaffolds (%d gap-pairs) | %s'
          % (len(rows), stats['scaffolds_used'], stats['pairs'],
             {k: v for k, v in stats.items() if k != 'pairs'}))
    if not rows:
        print('NOTHING BUILT'); return 1

    # THE CHECK THAT v1 FAILED. Anchors must map to MORE THAN ONE target, or the conditioning
    # carries no information about the output and the whole exercise is a copy task again.
    by = collections.defaultdict(set)
    for r in rows:
        by[r['anchor']].add(r['target'])
    dist = collections.Counter(len(v) for v in by.values())
    one = dist.get(1, 0)
    print('\nCONDITIONING INFORMATION CHECK')
    print('  distinct targets per anchor: %s' % dict(sorted(dist.items())[:8]))
    print('  anchors with only ONE target: %d / %d (%.2f%%)  -- v1 was 99.49%%'
          % (one, len(by), 100 * one / len(by)))
    if one / max(len(by), 1) > 0.25:
        print('  BUILD FAILED: conditioning still carries no information. Not writing.')
        return 2
    # and the label must actually select between DIFFERENT targets at the SAME gap
    same = collections.defaultdict(dict)
    for r in rows:
        same[(r['anchor'], r['d'], r['theta'])][r['label']] = r['target']
    both = [v for v in same.values() if len(v) == 2]
    diff = sum(1 for v in both if v[0] != v[1])
    print('  (anchor, gap) seen with BOTH labels: %d ; of those the two targets DIFFER: %d (%.1f%%)'
          % (len(both), diff, 100 * diff / max(len(both), 1)))

    print('\nNULL BATTERY (fragment-only descriptors -> label; must be ~0.500):')
    nb = null_battery(rows, rng)
    worst = max(nb.values())
    for k, v in sorted(nb.items(), key=lambda kv: -kv[1])[:5]:
        print('   %-14s %.3f%s' % (k, v, '  <-- LEAK' if v > 0.60 else ''))
    if worst > 0.60:
        print('  BUILD FAILED: a free descriptor predicts the label. Not writing.')
        return 2

    ths = np.array([r['theta'] for r in rows])
    ys = np.array([r['label'] for r in rows])
    from sklearn.metrics import roc_auc_score
    ct = np.cos(np.radians(ths))
    print('   %-14s %.3f   (v1 was 0.657)'
          % ('cos_theta', max(roc_auc_score(ys, ct), roc_auc_score(ys, -ct))))

    # FRAGMENT-DISJOINT *AND* SCAFFOLD-DISJOINT SPLIT
    # PRIMARY AXIS IS THE FRAGMENT, not the scaffold. The label is a function of
    # (fragment, d, theta) and the scaffold never enters it, so holding out fragments is the only
    # split that tests the geometry mapping at all. v1 split on scaffolds instead, which left 99.5%
    # of validation fragments present in training and made its reach metric a memorisation readout.
    # Requiring BOTH axes disjoint at once is stricter but leaves ~2% of rows (179 in the smoke
    # build) -- too few to measure anything -- so the doubly-disjoint rows are carved out and
    # REPORTED as a secondary, harder slice rather than used as the whole validation set.
    fr = sorted({r['fragment'] for r in rows})
    pyrng.shuffle(fr)
    val_f = set(fr[:max(1, int(0.15 * len(fr)))])
    tr = [r for r in rows if r['fragment'] not in val_f]
    va = [r for r in rows if r['fragment'] in val_f]
    fo = {r['fragment'] for r in tr} & {r['fragment'] for r in va}
    print('\nsplit (FRAGMENT-disjoint): %d train / %d valid' % (len(tr), len(va)))
    print('  fragment overlap %d (must be 0)' % len(fo))
    assert not fo
    tr_sc = {r['parent_scaffold'] for r in tr}
    hard = [r for r in va if r['parent_scaffold'] not in tr_sc]
    print('  of which ALSO scaffold-disjoint (harder slice): %d rows' % len(hard))

    os.makedirs(a.out, exist_ok=True)
    cols = ['anchor', 'target', 'scaffold', 'fragment', 'anchor_fragment', 'd', 'cos_theta',
            'theta', 'v_free', 'label', 'warhead_class', 'parent_scaffold']
    for nm, part in (('train', tr), ('valid', va), ('valid_hard', hard)):
        with open(os.path.join(a.out, '%s.csv' % nm), 'w', newline='') as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(part)
        print('  wrote %s/%s.csv (%d)' % (a.out, nm, len(part)))
    json.dump({'null_battery': nb, 'anchors_one_target_frac': one / max(len(by), 1),
               'pairs_with_differing_targets': diff, 'n_train': len(tr), 'n_valid': len(va)},
              open(os.path.join(a.out, 'build_report.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
