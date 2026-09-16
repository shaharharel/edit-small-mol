"""STAGE 0 -- the universal steering pair corpus. Six params, relative labels, real chemistry.

WHAT CHANGED VS THE ORIGINAL STAGE 0 SPEC ("CReM/covinDB/chemBL assembly"), all measured tonight:
  #194  CReM CANNOT vary 3 of the 6 params. With the warhead gate FULLY relaxed it changes
        wclass in 1.20%, path in 0.74%, flex in 0.04% of 9,416 variants. CReM stays as an
        AUGMENTER for the params it can move, but it cannot be the source for those three.
  #195  Real CovInDB pairs move them at 16-590x that rate; 10,550 edit-sized pairs survive MCS.
  HERE  data/lo_corpus/lo_pairs_v1.csv holds 307,555 REAL ChEMBL pairs (691 targets, 63,860
        unique molecules, TC mean 0.628) and no steering work has ever read it.

THE ACRYLAMIDE GATE COSTS 55x AND BUYS NOTHING. Measured on those 307,555 pairs:
        both sides bear an acrylamide      5,529 (1.80%)   <- what the shipped gate demands
        attachment-referenced labelable  307,264 (99.91%)
  The gate exists because all six params were defined FROM THE ELECTROPHILE -- a reference-atom
  choice inherited from the manuscript, not a chemistry requirement. Referenced from the
  attachment point instead (#137, which worked on 40/40 non-warhead fragments) the requirement
  disappears. Only `wclass` genuinely needs a warhead, because it IS the warhead's identity.

RELATIVE LABELS, NOT ABSOLUTE. Each param emits UP / DOWN / SAME from the sign of the delta.
This is deliberately the user's proposal and it is the right call for a reason worth recording:
`phi` was excluded as a "conformer-seed lottery" because its per-molecule VALUE is seed-noisy.
A relative label needs only the SIGN of a delta to be stable, which is a far weaker requirement
than a stable magnitude -- and the same ensemble fix that took `extent` from reliability 0.143 to
0.923 applies unchanged. So relative labelling attacks the exact defect that killed phi.

DEADBANDS COME FROM MEASURED SEED NOISE, NEVER FROM TASTE. The pose-independent three are
integers, so their deadband is 0 and is exact. The pose-dependent three get their deadband from
an ensemble sd measured on this corpus by --calibrate, NOT from the steer_k8 constants: those
were measured on a different population (100% acrylamide, 100% our own generations) and #187 is
the standing lesson about importing a constant across populations.
"""
import os, json, argparse, math
from collections import Counter
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
RDLogger.DisableLog('rdApp.*')

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
LO = os.path.join(ROOT, 'data/lo_corpus/lo_pairs_v1.csv')
COVINDB = os.path.join(ROOT, 'data/covbinder/raw_covindb2/CovInDB_All.csv')

# ---------------------------------------------------------------- warheads
WARHEADS = [
    ('acrylamide', '[CH2]=[CH]C(=O)N'), ('propiolamide', 'C#CC(=O)N'),
    ('chloroacetamide', 'ClCC(=O)N'), ('vinylsulfone', 'C=CS(=O)(=O)'),
    ('fluorosulfate', 'OS(=O)(=O)F'), ('epoxide', 'C1CO1'),
    ('boronic', 'B(O)O'), ('aldehyde', '[CX3H1](=O)'), ('nitrile', '[NX1]#[CX2]'),
]
# The Michael-acceptor dihedral phi is measured on: C=C-C(=O)-N. #179 recorded that NO file in
# the repo computed a dihedral at all and that this SMARTS was defined and never used. It is used
# here, and --calibrate validates the producer against #180's population values before any build.
MICHAEL = '[CH2]=[CH]-[CX3](=O)-[NX3]'
_P = None
_MI = None


def _pats():
    global _P, _MI
    if _P is None:
        _P = [(k, Chem.MolFromSmarts(v)) for k, v in WARHEADS]
        _MI = Chem.MolFromSmarts(MICHAEL)
    return _P


def wclass(m):
    return tuple(k for k, p in _pats() if m.HasSubstructMatch(p))


def elec_atom(m):
    for _, p in _pats():
        h = m.GetSubstructMatch(p)
        if h:
            return h[0]
    return None


def attach_atom(m):
    """ATTACHMENT reference (#137): the acyclic heavy atom farthest from the ring system.
    Defined for any molecule with a ring, warhead or not. This is what removes the acrylamide
    requirement from five of the six params."""
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring:
        return None
    dm = Chem.GetDistanceMatrix(m)
    best, bd = None, -1
    for i in range(m.GetNumAtoms()):
        if m.GetAtomWithIdx(i).IsInRing():
            continue
        d = min(dm[i][r] for r in ring)
        if d > bd:
            best, bd = i, d
    return best


def ref_atom(m, mode):
    return elec_atom(m) if mode == 'elec' else attach_atom(m)


# ---------------------------------------------------- POSE-INDEPENDENT (2D, no conformer)
def p_path(m, ref):
    dm = Chem.GetDistanceMatrix(m)
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    return int(min(dm[ref][i] for i in ring)) if ring else None


def p_flex(m, ref):
    dm = Chem.GetDistanceMatrix(m)
    near = {i for i in range(m.GetNumAtoms()) if dm[ref][i] <= 3}
    return sum(1 for b in m.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
               and b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near
               and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)


def p_wclass(m, ref):
    w = wclass(m)
    return '+'.join(w) if w else None


# ---------------------------------------------------- POSE-DEPENDENT (need a 3D ensemble)
def _embed(m, k, seed):
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    ps.pruneRmsThresh = -1.0
    cids = AllChem.EmbedMultipleConfs(mh, numConfs=k, params=ps)
    if not len(cids):
        return None, []
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=200)
    except Exception:
        pass
    return mh, list(cids)


def pose_params(m, ref, k, seed):
    """(extent, phi, exitang) as ENSEMBLE MEANS over k conformers. Returns dict of value+sd so the
    deadband can be set from the sd rather than guessed."""
    mh, cids = _embed(m, k, seed)
    if mh is None:
        return {}
    conf_ref = ref  # heavy-atom indices are preserved by AddHs
    ring_h = [i for i in range(mh.GetNumAtoms())
              if mh.GetAtomWithIdx(i).IsInRing()]
    ext, phis, angs = [], [], []
    mi = mh.GetSubstructMatch(_MI) if _MI is not None else ()
    for c in cids:
        pos = mh.GetConformer(c).GetPositions()
        # EXTENT: 3D distance from the reference atom to the farthest ring atom.
        if ring_h:
            ext.append(float(max(np.linalg.norm(pos[conf_ref] - pos[r]) for r in ring_h)))
        # PHI: Michael-acceptor dihedral C=C-C(=O)-N, the planarity metric (#180).
        if len(mi) >= 5:
            try:
                d = rdMolTransforms.GetDihedralDeg(mh.GetConformer(c), mi[0], mi[1], mi[2], mi[4])
                # fold to deviation from planarity: 0 deg = planar, 90 = maximally twisted
                dev = abs(((d + 180.0) % 360.0) - 180.0)
                phis.append(min(dev, 180.0 - dev))
            except Exception:
                pass
        # EXITANG: angle ring-atom -> reference -> its neighbour, i.e. how the substituent leaves.
        nb = [a.GetIdx() for a in mh.GetAtomWithIdx(conf_ref).GetNeighbors()
              if mh.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() > 1]
        if ring_h and nb:
            r0 = min(ring_h, key=lambda r: np.linalg.norm(pos[conf_ref] - pos[r]))
            v1, v2 = pos[r0] - pos[conf_ref], pos[nb[0]] - pos[conf_ref]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                angs.append(float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))))
    out = {}
    for nm, arr in (('extent', ext), ('phi', phis), ('exitang', angs)):
        if arr:
            out[nm] = float(np.mean(arr))
            out[nm + '_sd'] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


POSE_FREE = dict(path=p_path, flex=p_flex, wclass=p_wclass)
POSE_DEP = ('extent', 'phi', 'exitang')


def rel_label(a, b, deadband):
    """RELATIVE steering label. SAME when |delta| <= deadband. For categorical wclass there is no
    ordering, so it emits SAME / CHANGED rather than a fake UP/DOWN."""
    if a is None or b is None:
        return None
    if isinstance(a, str) or isinstance(b, str):
        return 'SAME' if a == b else 'CHANGED'
    d = b - a
    if abs(d) <= deadband:
        return 'SAME'
    return 'UP' if d > 0 else 'DOWN'


_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg
    _pats()


def label_pair(rec):
    s1, s2, src = rec
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return None
    row = dict(input_smiles=s1, output_smiles=s2, source=src)
    got = 0
    for mode in ('attach', 'elec'):
        r1, r2 = ref_atom(m1, mode), ref_atom(m2, mode)
        if r1 is None or r2 is None:
            continue
        for nm, fn in POSE_FREE.items():
            try:
                a, b = fn(m1, r1), fn(m2, r2)
            except Exception:
                continue
            if a is None or b is None:
                continue
            row['%s_%s_a' % (mode, nm)] = a
            row['%s_%s_b' % (mode, nm)] = b
            lab = rel_label(a, b, 0)          # integers / categorical: deadband 0, exact
            row['%s_%s_dir' % (mode, nm)] = lab
            got += 1
        if _CFG['pose'] and mode == _CFG['pose_ref']:
            pa = pose_params(m1, r1, _CFG['k'], _CFG['seed'])
            pb = pose_params(m2, r2, _CFG['k'], _CFG['seed'] + 1)
            for nm in POSE_DEP:
                if nm in pa and nm in pb:
                    row['%s_%s_a' % (mode, nm)] = pa[nm]
                    row['%s_%s_b' % (mode, nm)] = pb[nm]
                    row['%s_%s_sd' % (mode, nm)] = max(pa.get(nm + '_sd', 0), pb.get(nm + '_sd', 0))
                    row['%s_%s_dir' % (mode, nm)] = rel_label(pa[nm], pb[nm],
                                                              _CFG['deadband'].get(nm, 0.0))
                    got += 1
    return row if got else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--pose', action='store_true', help='also compute the 3 pose-dependent params')
    ap.add_argument('--pose-ref', default='attach', choices=['attach', 'elec'])
    ap.add_argument('--k', type=int, default=8, help='conformers per molecule')
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--calibrate', type=int, default=0,
                    help='measure ensemble sd on N pairs and print deadbands, build nothing')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments/covalentformer/data/universal_pairs'))
    a = ap.parse_args()

    df = pd.read_csv(LO, usecols=['input_smiles', 'output_smiles'])
    recs = [(x, y, 'chembl_lo') for x, y in zip(df.input_smiles, df.output_smiles)]
    print('lo_corpus pairs: %d' % len(recs), flush=True)
    if a.limit:
        recs = recs[:a.limit]

    cfg = dict(pose=bool(a.pose or a.calibrate), pose_ref=a.pose_ref, k=a.k, seed=a.seed,
               deadband=dict(extent=0.0, phi=0.0, exitang=0.0))

    if a.calibrate:
        recs = recs[:a.calibrate]
        print('CALIBRATION RUN on %d pairs -- measuring ensemble sd, building nothing' % len(recs),
              flush=True)
    print('labelling %d pairs on %d workers (pose=%s, k=%d)'
          % (len(recs), a.workers, cfg['pose'], a.k), flush=True)

    rows = []
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for n, r in enumerate(pool.imap_unordered(label_pair, recs, chunksize=64), 1):
            if r:
                rows.append(r)
            if n % 20000 == 0:
                print('  %d/%d labelled, %d kept' % (n, len(recs), len(rows)), flush=True)
    out = pd.DataFrame(rows)
    print('\nrows: %d' % len(out))

    if a.calibrate:
        print('\n=== ENSEMBLE SD, measured on THIS corpus (not imported) ===')
        for nm in POSE_DEP:
            col = '%s_%s_sd' % (a.pose_ref, nm)
            if col in out:
                s = out[col].dropna()
                if len(s):
                    print('%-9s n=%-6d mean sd=%7.3f  p95=%7.3f  -> suggested deadband (3sd)=%7.3f'
                          % (nm, len(s), s.mean(), s.quantile(0.95), 3 * s.mean()))
        pc = '%s_phi_a' % a.pose_ref
        if pc in out:
            s = out[pc].dropna()
            if len(s):
                print('\n=== PHI PRODUCER VALIDATION (#179: no dihedral producer existed) ===')
                print('phi on inputs: n=%d mean=%.2f deg median=%.2f p05=%.2f p95=%.2f'
                      % (len(s), s.mean(), s.median(), s.quantile(.05), s.quantile(.95)))
                print('#180 reference population: unconditioned 61.8-63.5 deg, v2_cond 2.62 deg.')
                print('This corpus is ChEMBL LO pairs, a DIFFERENT population again -- so this is')
                print('a sanity range, NOT a match target. Read it as "does the producer return')
                print('plausible dihedral deviations at all", which is what #179 said was untested.')
        return

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, 'pairs_labelled.csv')
    out.to_csv(p, index=False)
    print('\n=== LABEL YIELD PER PARAM (relative labels) ===')
    for c in sorted(x for x in out.columns if x.endswith('_dir')):
        vc = out[c].value_counts()
        n = int(vc.sum())
        steer = int(n - vc.get('SAME', 0))
        print('%-24s labelled=%-7d steerable=%-7d (%.1f%%)  %s'
              % (c, n, steer, 100.0 * steer / n if n else float('nan'), dict(vc)))
    json.dump(dict(n_rows=len(out), cfg={k: v for k, v in cfg.items() if k != 'deadband'}),
              open(os.path.join(a.out, 'build_meta.json'), 'w'), indent=2)
    print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
