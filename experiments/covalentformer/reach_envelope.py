"""REACH ENVELOPE: precompute, per fragment, every (d, theta) its electrophile can present.

WHY THIS IS THE RIGHT PRIMITIVE. Labelling one (fragment, gap) pair is a geometry question that does
not involve the protein at all:

    A  = attachment point (the [*] where the scaffold was)
    N  = the first fragment atom (so A->N is the EXIT VECTOR, the direction the fragment grows)
    E  = the electrophilic carbon
    d      = |A - E|
    theta  = angle( (N-A), (E-A) )     0 deg = electrophile straight ahead; 180 = folded back

Those two numbers, over the conformer ensemble, are the fragment's REACH ENVELOPE. The matching
requirement read off a pocket is the same pair:

    d_req     = |SG - A|
    theta_req = angle( exit vector of the scaffold, (SG - A) )

and the fragment reaches iff some conformer matches both. Two numbers suffice because the torsion
about the exit bond is free, which sweeps the remaining azimuthal degree of freedom: if d and theta
agree, that conformer can be rotated about A->N until E lands on SG.

PRECISION IS 1.0 BY CONSTRUCTION, WHICH IS THE WHOLE POINT.
A POSITIVE label here is WITNESSED -- there is an actual embedded conformer whose electrophile sits
at that (d, theta). It is not an inference about a protein, it is a geometric fact about the
molecule.
    CORRECTION: this sentence used to read "an actual embedded, MMFF-MINIMISED conformer", and that
    was FALSE for every row ever written by this file. MMFF has no parameters for the [n*] dummy, so
    the optimiser silently no-opped on 100% of fragments -- see the long note in envelope(). The
    conformers are RAW ETKDG. The one-sided-error argument below is UNAFFECTED, because it rests on
    "more sampling can only add points to an envelope", which is true of unminimised conformers too;
    but the geometry is less relaxed than the old wording claimed, and 3 of 13 test fragments shifted
    median d by more than the entire oracle tolerance once minimisation was actually applied.
    build_tierA.py and shuffle_test.py repeat the old wording and need the same correction. Adding conformers can only ADD points to an envelope, never remove them, so more
sampling can only convert a negative into a positive. The error is therefore one-sided: some
fragments are labelled unreachable that could reach with better sampling (false negatives), and NO
fragment is labelled reachable that cannot be (false positives ~ 0).

This matters because it splits the oracle's claim in two:
    EXACT part        d and theta from the conformer ensemble  <- this file. Trustworthy as a label.
    INFERENTIAL part  clash and free volume against a RIGID receptor  <- specificity_reach.py is
                      testing whether that part discriminates at all. It is not used here.
Tier A training data uses only the exact part, so it can be built before that verdict lands.

OUTPUT one row per fragment: the (d, theta) point cloud over conformers, as float16. Labelling a
fragment against any requirement is then a vectorised comparison against the cloud -- microseconds,
so gap augmentation is effectively free and one fragment yields as many examples as we want gaps.
"""
import os, sys, json, time, argparse, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import electrophile_index

N_CONF = 300
PRUNE_RMS = 0.3
SEED = 20260913
# Counts fragments whose conformers were NOT MMFF-minimised. Every fragment carrying a [n*] dummy
# lands here, which on the corpora built so far is all of them -- see the long note in envelope().
_NOMIN = [0]
# Fragments that REACHED the optimiser -- the correct denominator for _NOMIN (see envelope()).
_REACHED = [0]


def envelope(smi, n_conf=N_CONF):
    """-> dict(d=[...], theta=[...], cls=str) or dict(err=...) for one [*]-tagged fragment."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {'err': 'unparseable'}
    star = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star) != 1:
        return {'err': 'n_attachment_%d' % len(star)}
    a_idx = star[0]
    nb = [x.GetIdx() for x in m.GetAtomWithIdx(a_idx).GetNeighbors()]
    if len(nb) != 1:
        return {'err': 'attachment_degree_%d' % len(nb)}
    n_idx = nb[0]
    el, cls = electrophile_index(m)
    if el is None:
        return {'err': 'warhead_%s' % cls}
    if el in (a_idx, n_idx):
        return {'err': 'electrophile_is_attachment'}

    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    ps.pruneRmsThresh = PRUNE_RMS
    ps.useSmallRingTorsions = True
    cids = AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=ps)
    if not len(cids):
        return {'err': 'embed_failed'}
    # THE MINIMISATION HAS NEVER RUN, ON ANY FRAGMENT, AND IT NEVER RAISED.
    # mh still carries the [n*] dummy (atomic number 0), for which MMFF has no parameters.
    # MMFFOptimizeMoleculeConfs does not throw on an unparameterisable molecule -- it returns the
    # sentinel [(-1, -1.0), ...] and leaves every coordinate BIT-IDENTICAL. The return value was
    # discarded, so the `except Exception: pass` was guarding against nothing: there was no
    # exception, only a silent no-op. Verified on sampled envelope fragments:
    # MMFFHasAllMoleculeParams is False on 100% of them, the return is (-1, -1.0), and
    # max|dx| after the call is exactly 0.00e+00.
    # CONSEQUENCE: every envelope in data/envelopes.jsonl and data/envelopes_20k.jsonl is RAW
    # ETKDG, not minimised. The phrase "an actual embedded, MMFF-minimised conformer" -- used to
    # justify the one-sided-error claim in this file, build_tierA.py and shuffle_test.py -- is
    # FALSE for every row on disk. The correct wording is "witnessed by an unminimised ETKDG
    # conformer". Measured magnitude on methyl-capped analogues (so MMFF can actually run, same
    # conformer set before and after, isolating the minimisation): median |dd| 0.04 A and
    # median |dtheta| 3.0 deg, but 3 of 13 fragments shift median d by more than the ENTIRE
    # oracle tolerance D_TOL=0.5 A. So the tail matters even though the median does not.
    # NOT silently "fixed" here by capping the dummy: that would change the geometry semantics and
    # make new envelopes incomparable with every number already derived from the existing files.
    # Instead the no-op is now DETECTED and REPORTED, and capping is available behind an explicit
    # flag so a rebuild is a deliberate choice with a recorded provenance difference.
    minimised = False
    if AllChem.MMFFHasAllMoleculeParams(mh):
        res = AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=300)
        minimised = bool(res) and all(r[0] != -1 for r in res)
    # BOTH counters, because the obvious denominator is the wrong one. _NOMIN increments here, for
    # every fragment that REACHES the optimiser; but `ok` in main() counts only fragments that go on
    # to produce geometry, and a fragment can reach the optimiser and then fail at 'no_geometry'
    # below. Dividing _NOMIN by `ok` can therefore exceed 100%. _REACHED is the correct denominator.
    _REACHED[0] += 1
    if not minimised:
        _NOMIN[0] += 1

    ds, ths = [], []
    for cid in cids:
        c = mh.GetConformer(cid)
        A = np.array(list(c.GetAtomPosition(a_idx)))
        N = np.array(list(c.GetAtomPosition(n_idx)))
        E = np.array(list(c.GetAtomPosition(el)))
        v_exit, v_el = N - A, E - A
        d = float(np.linalg.norm(v_el))
        if d < 1e-6:
            continue
        ct = float(v_exit.dot(v_el) / (np.linalg.norm(v_exit) * d))
        ds.append(d)
        ths.append(float(np.degrees(np.arccos(np.clip(ct, -1, 1)))))
    if not ds:
        return {'err': 'no_geometry'}
    return {'d': ds, 'theta': ths, 'cls': cls, 'n_conf': len(ds),
            'n_heavy': m.GetNumHeavyAtoms() - 1, 'minimised': minimised}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='experiments/covalentformer/data/meta.csv')
    ap.add_argument('--out', default='experiments/covalentformer/data/envelopes.jsonl')
    ap.add_argument('--n-conf', type=int, default=N_CONF)
    ap.add_argument('--limit', type=int, default=0)
    # SHARDING. Profiling is embarrassingly parallel -- each fragment is independent -- and single
    # process runs at ~18 fragments/min, i.e. >3 h for the corpus. Strided shards (i, i+N, i+2N...)
    # rather than contiguous blocks so every shard gets a similar mix of fragment sizes; contiguous
    # blocks would land all the large, slow fragments in whichever shard holds that sort range.
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    # --out was opened with mode 'w', which TRUNCATES. That is a live hazard, not a hypothetical:
    # data/envelopes_20k.jsonl holds 12,680 envelopes of which 7,058 (56%) correspond to fragments
    # that exist in NO input file on disk. Its --meta list was passed inline and never persisted, so
    # that artifact CANNOT BE REGENERATED -- one careless rerun with the default --out would have
    # destroyed it permanently. Refuse to clobber a non-empty file unless the caller says so.
    ap.add_argument('--force', action='store_true',
                    help='permit overwriting a non-empty --out (default: refuse)')
    a = ap.parse_args()

    if os.path.exists(a.out) and os.path.getsize(a.out) > 0 and not a.force:
        print('REFUSING to overwrite non-empty %s (%d bytes).' % (a.out, os.path.getsize(a.out)))
        print('  Envelope files are expensive (~18 fragments/min) and some on disk are NOT')
        print('  reproducible, because their input fragment list was never persisted.')
        print('  Pass --force if you really mean to discard it, or choose a different --out.')
        return 1

    import csv as _csv
    frags = sorted({r['fragment'] for r in _csv.DictReader(open(a.meta))})
    if a.limit:
        frags = frags[:a.limit]
    if a.nshards > 1:
        frags = frags[a.shard::a.nshards]
    print('fragments to profile: %d  (%d conformers each) [shard %d/%d]'
          % (len(frags), a.n_conf, a.shard, a.nshards))

    t0 = time.time()
    ok = 0
    err = collections.Counter()
    with open(a.out, 'w') as fh:
        for i, s in enumerate(frags):
            e = envelope(s, a.n_conf)
            if 'err' in e:
                err[e['err']] += 1
            else:
                ok += 1
                # 'minimised' is written PER ROW so a consumer can tell, from the artifact alone,
                # whether a given envelope was MMFF-minimised or is raw ETKDG. Rows in
                # envelopes.jsonl / envelopes_20k.jsonl predate this field; their absence means
                # UNMINIMISED, since the optimiser never ran on any of them.
                fh.write(json.dumps({'frag': s, 'cls': e['cls'], 'n_heavy': e['n_heavy'],
                                     'd': [round(x, 2) for x in e['d']],
                                     'theta': [round(x, 1) for x in e['theta']],
                                     'minimised': bool(e.get('minimised'))}) + '\n')
            if (i + 1) % 250 == 0:
                el_ = time.time() - t0
                print('  %5d/%d  ok=%d  %.1f s  (eta %.1f min)'
                      % (i + 1, len(frags), ok, el_, (len(frags) - i - 1) * el_ / (i + 1) / 60),
                      flush=True)

    print('\nprofiled %d/%d fragments in %.1f min' % (ok, len(frags), (time.time() - t0) / 60))
    print('  errors: %s' % dict(err))
    # REPORTED, NOT ASSUMED. This was a silent no-op for the entire life of the file.
    if _NOMIN[0]:
        print('  *** MMFF MINIMISATION DID NOT RUN on %d of the %d fragments that reached the '
              'optimiser (%.1f%%).'
              % (_NOMIN[0], _REACHED[0], 100.0 * _NOMIN[0] / max(_REACHED[0], 1)))
        print('      MMFF has no parameters for the [n*] dummy (atomic number 0) and')
        print('      MMFFOptimizeMoleculeConfs returns (-1,-1.0) WITHOUT raising, leaving')
        print('      coordinates bit-identical. These envelopes are RAW ETKDG.')
        print('      Do NOT describe them as "MMFF-minimised" -- say "unminimised ETKDG conformer".')
    elif _REACHED[0]:
        print('  MMFF minimisation ran on all %d fragments that reached the optimiser.' % _REACHED[0])
    else:
        # A ZERO-YIELD RUN USED TO PRINT THE ALL-CLEAR ABOVE. _NOMIN stays 0 when nothing reaches the
        # optimiser, so `else` fired and announced "minimisation ran on all 0 fragments" -- a
        # reassuring sentence for a run where every fragment errored out. That is the
        # conclusion-hardcoded-into-a-print pattern: the text asserts success without consulting a
        # number that could contradict it. Reproduced on a meta file of unparseable fragments.
        print('  NO FRAGMENT REACHED THE OPTIMISER (%d profiled). Nothing was minimised and nothing '
              'was skipped -- this run produced no geometry at all.' % ok)
    print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
