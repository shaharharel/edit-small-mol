"""TIER A TRAINING SET: (anchor SMILES, geometry requirement, oracle label) -> target SMILES.

THE ONE DESIGN DECISION THAT MATTERS: PER-FRAGMENT LABEL BALANCE.

The naive construction samples gaps from one global range for every fragment. That is broken, and
broken in the exact way this campaign has been burned by four times already: long floppy fragments
would collect mostly r=1 and short rigid ones mostly r=0, so BOND COUNT PREDICTS THE LABEL. The model
would then learn "emit a long linker", score well, and have internalised nothing about geometry -- a
free descriptor solving the task, dressed up as a structural result.

The fix is to sample each fragment's gaps FROM ITS OWN ENVELOPE:

    K_POS gaps drawn from INSIDE  the fragment's reach envelope   -> r = 1
    K_NEG gaps drawn from OUTSIDE the fragment's reach envelope   -> r = 0

Every fragment then carries exactly K_POS positives and K_NEG negatives. Marginalised over gaps, the
label is independent of every fragment-only descriptor -- MW, bond count, rigidity, fsp3, ECFP4 all
sit at 0.500 BY CONSTRUCTION. The label is a joint property of (fragment, gap) and the ONLY way to
predict it is to read the geometry input. That is the whole point of the architecture, so the data
has to enforce it rather than hope for it.

This is checked, not asserted: null_battery() trains a classifier on fragment descriptors alone and
the build FAILS if any of them exceeds AUROC 0.60.

WHY THE LABEL IS TRUSTWORTHY HERE. A positive is WITNESSED -- an actual embedded, MMFF-minimised
conformer sits at that (d, theta). Adding conformers can only ADD points to an envelope, so the error
is one-sided: some fragments are called unreachable that could reach with better sampling (false
negatives), and essentially none are called reachable that cannot be. Tier A uses only this exact
part of the oracle. The inferential part -- clash and free volume against a rigid receptor -- is NOT
used here, which is why this set can be built before boundary_reach.py returns its verdict.

V_FREE IS A PLACEHOLDER IN TIER A. With no pocket there is no occlusion to measure, so the third
conditioning channel is pinned at 1.0 (fully open) for every example. It carries no information
tonight and MUST NOT be interpreted; Tier B supplies the real values.
"""
import os, sys, json, csv, random, argparse, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

D_TOL, A_TOL = 0.5, 20.0      # a gap is MATCHED if some conformer is within both
D_LO, D_HI = 2.0, 12.0        # plausible attachment->SG gaps seen across real pockets
K_POS, K_NEG = 10, 10
SEED = 20260913


def join(scaf_smi, frag_smi):
    """Bond a [*]-tagged scaffold to a [*]-tagged fragment, returning a full molecule."""
    a, b = Chem.MolFromSmiles(scaf_smi), Chem.MolFromSmiles(frag_smi)
    if a is None or b is None:
        return None
    combo = Chem.RWMol(Chem.CombineMols(a, b))
    stars = [at.GetIdx() for at in combo.GetAtoms() if at.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nbrs = []
    for s in stars:
        n = [x.GetIdx() for x in combo.GetAtomWithIdx(s).GetNeighbors()]
        if len(n) != 1:
            return None
        nbrs.append(n[0])
    try:
        combo.AddBond(nbrs[0], nbrs[1], Chem.BondType.SINGLE)
        for s in sorted(stars, reverse=True):
            combo.RemoveAtom(s)
        m = combo.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def sample_gaps(d_arr, th_arr, rng, theta_matched=True):
    """-> [(d, theta, label), ...] with K_POS inside the envelope and K_NEG outside it.

    THETA-MATCHED NEGATIVES (the fix; theta_matched=False reproduces the original build).

    The first version drew both classes from a uniform theta on [0,180]. Real envelopes concentrate
    at LOW theta -- the electrophile points away from the attachment -- so "inside" draws landed
    mostly at low theta and "outside" at high theta. Measured on the resulting 432k set: positives
    median theta 58.6 deg, negatives 96.8 deg, and cos_theta alone predicted the label at AUROC
    0.657. That is not a molecule-descriptor leak (ECFP4 stayed at 0.487) but it starves the model
    of high-theta positives, which are precisely the hard cases where the electrophile must fold
    BACK toward the scaffold.

    The fix pairs each negative with a positive at the SAME theta and a MISSING d. Two consequences:
      - theta becomes uninformative about the label by construction, not by hope;
      - the negatives get harder. "Right direction, wrong distance" forces the model to learn the
        distance mapping instead of settling for the direction.
    """
    pos, guard = [], 0
    while len(pos) < K_POS and guard < 4000:
        guard += 1
        d = float(rng.uniform(D_LO, D_HI))
        th = float(rng.uniform(0.0, 180.0))
        if bool(np.any((np.abs(d_arr - d) <= D_TOL) & (np.abs(th_arr - th) <= A_TOL))):
            pos.append((d, th, 1))
    if len(pos) < K_POS:
        return None

    neg = []
    if theta_matched:
        for (_dp, thp, _l) in pos:
            for _try in range(200):
                dn = float(rng.uniform(D_LO, D_HI))
                if not np.any((np.abs(d_arr - dn) <= D_TOL) & (np.abs(th_arr - thp) <= A_TOL)):
                    neg.append((dn, thp, 0))
                    break
    else:
        g2 = 0
        while len(neg) < K_NEG and g2 < 4000:
            g2 += 1
            d = float(rng.uniform(D_LO, D_HI))
            th = float(rng.uniform(0.0, 180.0))
            if not np.any((np.abs(d_arr - d) <= D_TOL) & (np.abs(th_arr - th) <= A_TOL)):
                neg.append((d, th, 0))
    # Only emit a fragment if BOTH classes filled. A fragment contributing 10 positives and 2
    # negatives would reintroduce exactly the descriptor->label leak this design exists to remove.
    if len(pos) < K_POS or len(neg) < K_NEG:
        return None
    return pos + neg


def null_battery(rows, rng):
    """Can any FRAGMENT-ONLY descriptor predict the label? By construction it must not."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sub = rows if len(rows) <= 40000 else [rows[i] for i in
                                           rng.choice(len(rows), 40000, replace=False)]
    cache, feats, ys = {}, [], []
    for r in sub:
        f = r['fragment']
        if f not in cache:
            m = Chem.MolFromSmiles(f)
            if m is None:
                cache[f] = None
            else:
                cache[f] = [m.GetNumHeavyAtoms(), Descriptors.MolWt(m), Descriptors.MolLogP(m),
                            Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m),
                            Descriptors.FractionCSP3(m), Descriptors.RingCount(m),
                            Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m)]
        if cache[f] is None:
            continue
        feats.append(cache[f])
        ys.append(r['label'])
    X, y = np.array(feats), np.array(ys)
    names = ['heavy', 'MW', 'logP', 'TPSA', 'rotB', 'fsp3', 'rings', 'HBA', 'HBD']
    out = {}
    for i, n in enumerate(names):
        try:
            out[n] = float(max(roc_auc_score(y, X[:, i]), roc_auc_score(y, -X[:, i])))
        except Exception:
            out[n] = float('nan')
    try:
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        tr, te = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        out['ALL (logreg)'] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    except Exception:
        out['ALL (logreg)'] = float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='experiments/covalentformer/data/envelopes.jsonl')
    ap.add_argument('--meta', default='experiments/covalentformer/data/meta.csv')
    ap.add_argument('--out', default='experiments/covalentformer/data/tierA')
    ap.add_argument('--n-scaf', type=int, default=1200, help='scaffolds sampled as anchors')
    ap.add_argument('--frag-per-scaf', type=int, default=18)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)

    envs = {}
    if os.path.isdir(os.path.dirname(a.env)):
        shard_dir = os.path.join(os.path.dirname(a.env), 'shards')
        srcs = ([os.path.join(shard_dir, f) for f in sorted(os.listdir(shard_dir))]
                if os.path.isdir(shard_dir) else [])
        if os.path.exists(a.env):
            srcs.append(a.env)
    for s in srcs:
        for line in open(s):
            try:
                e = json.loads(line)
            except Exception:
                continue
            envs[e['frag']] = (np.array(e['d'], dtype=float), np.array(e['theta'], dtype=float),
                               e.get('cls', '?'))
    print('envelopes loaded: %d fragments' % len(envs))
    if len(envs) < 200:
        print('TOO FEW ENVELOPES -- is reach_envelope.py still running?')
        return 1

    meta = list(csv.DictReader(open(a.meta)))
    scafs = sorted({r['scaffold'] for r in meta})
    par_scaf = {}
    for r in meta:
        par_scaf.setdefault(r['scaffold'], r.get('parent_scaffold', ''))
    pyrng.shuffle(scafs)
    scafs = scafs[:a.n_scaf]
    frag_list = sorted(envs.keys())

    # PRE-SAMPLE GAPS ONCE PER FRAGMENT so every scaffold that uses a fragment sees the SAME
    # balanced 10/10 split. Resampling per (scaffold, fragment) would let the pos:neg ratio drift
    # per fragment and quietly reopen the descriptor leak.
    gaps, dropped = {}, 0
    for f in frag_list:
        d_arr, th_arr, _c = envs[f]
        g = sample_gaps(d_arr, th_arr, rng)
        if g is None:
            dropped += 1
        else:
            gaps[f] = g
    print('fragments with a balanced %d/%d gap set: %d  (dropped %d)'
          % (K_POS, K_NEG, len(gaps), dropped))
    usable = sorted(gaps.keys())

    rows, joinfail = [], 0
    for sc in scafs:
        picks = pyrng.sample(usable, min(a.frag_per_scaf, len(usable)))
        built = []
        for f in picks:
            full = join(sc, f)
            if full is None:
                joinfail += 1
                continue
            built.append((f, full))
        if len(built) < 2:
            continue
        for i, (f, full) in enumerate(built):
            # ANCHOR = the SAME scaffold carrying a DIFFERENT fragment. The edit the model must
            # learn is therefore confined to the reactive end, which is the hit-to-lead move.
            f0, anchor = built[(i + 1) % len(built)]
            for (d, th, lab) in gaps[f]:
                rows.append(dict(anchor=anchor, target=full, scaffold=sc, fragment=f,
                                 anchor_fragment=f0, d=round(d, 3),
                                 cos_theta=round(float(np.cos(np.radians(th))), 4),
                                 theta=round(th, 2), v_free=1.0, label=lab,
                                 warhead_class=envs[f][2], parent_scaffold=par_scaf.get(sc, '')))
    print('examples built: %d  (join failures %d)' % (len(rows), joinfail))
    if not rows:
        print('NOTHING BUILT'); return 1
    print('  label balance: %s' % dict(collections.Counter(r['label'] for r in rows)))

    print('\nNULL BATTERY -- fragment-only descriptors predicting the label (must be ~0.500):')
    nb = null_battery(rows, rng)
    worst, worstn = 0.0, ''
    for k, v in sorted(nb.items(), key=lambda kv: -kv[1]):
        flag = '  <-- LEAK' if v > 0.60 else ''
        print('    %-14s AUROC %.3f%s' % (k, v, flag))
        if v > worst:
            worst, worstn = v, k
    if worst > 0.60:
        print('\n  BUILD FAILED: %s predicts the label at %.3f. A free descriptor solves the task,'
              % (worstn, worst))
        print('  so any model trained on this set can ignore the geometry channel. Not writing.')
        return 2
    print('  PASS: no fragment descriptor exceeds 0.60. Label requires the geometry input.')

    # SCAFFOLD-DISJOINT SPLIT on the parent Bemis-Murcko scaffold
    keys = sorted({r['parent_scaffold'] for r in rows})
    pyrng.shuffle(keys)
    val_keys = set(keys[:max(1, int(0.1 * len(keys)))])
    tr = [r for r in rows if r['parent_scaffold'] not in val_keys]
    va = [r for r in rows if r['parent_scaffold'] in val_keys]
    ov = {r['parent_scaffold'] for r in tr} & {r['parent_scaffold'] for r in va}
    assert not ov, 'scaffold leakage: %d' % len(ov)
    print('\nscaffold-disjoint split: %d train / %d valid  (overlap %d, must be 0)'
          % (len(tr), len(va), len(ov)))

    os.makedirs(a.out, exist_ok=True)
    cols = ['anchor', 'target', 'scaffold', 'fragment', 'anchor_fragment', 'd', 'cos_theta',
            'theta', 'v_free', 'label', 'warhead_class', 'parent_scaffold']
    for nm, part in (('train', tr), ('valid', va)):
        p = os.path.join(a.out, '%s.csv' % nm)
        with open(p, 'w', newline='') as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(part)
        print('  wrote %s (%d rows)' % (p, len(part)))
    json.dump({'null_battery': nb, 'n_train': len(tr), 'n_valid': len(va),
               'k_pos': K_POS, 'k_neg': K_NEG, 'd_tol': D_TOL, 'a_tol': A_TOL,
               'n_fragments': len(gaps)},
              open(os.path.join(a.out, 'build_report.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
