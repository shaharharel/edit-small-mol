"""PAIRS WITHOUT THE SINGLE-CUT MMP CONSTRAINT: TC >= 0.4 and a molecular-weight window.

WHY. The single-cut MMP requirement -- cut exactly ONE acyclic bond in each molecule, demand both
halves leave the SAME core -- is the most aggressive filter in the pipeline, and it was never
required by anything we actually measure:
  * The LABELS do not need it. attach_path/attach_flex are defined from "the acyclic heavy atom
    farthest from a ring", computable on ANY molecule with a ring. No shared core is involved.
  * The SIMILARITY does not need it. Measured on the v3 corpus: single-cut MMP pairs have TC
    min 0.400, median 0.769, mean 0.753, and 0.00% fall below 0.4 -- because the extractor ALREADY
    applies a TC>=0.4 gate. So "TC>=0.4" is not an alternative to MMP, it is a filter MMP pairs
    already satisfy. Dropping single-cut while KEEPING TC>=0.4 is therefore a STRICT SUPERSET:
    every pair we have now is retained, plus every multi-substituent pair of the same similarity.
  * What single-cut DOES buy is ATTRIBUTION: with one substituent changed, a delta in the parameter
    is caused by that edit. With a multi-point pair, several things move at once and the steering
    signal competes with the rest. That is an argument about LEARNABILITY, not validity -- and it
    is a hypothesis to test against the MMP arm, not a reason to discard 80% of the data up front.

THE COST BEING PAID NOW, measured: of 815,701 MMP pairs, attach_path keeps 230,359 as UP/DOWN
(28.2%) and the trainer sees 92,954 (11.4%).

SPLIT KEY. Without a shared core there is no `core` field to split on, so the disjointness key
becomes the BEMIS-MURCKO SCAFFOLD of the input. That is weaker than a core -- two different
scaffolds can still be near-duplicates -- so the Tanimoto near-duplicate check (which found a
3.54% stereoisomer leak in v3 that exact-string matching reported as 0.00%) is MANDATORY on this
corpus, not optional.

COMBINATORICS, stated because it is the thing that will bite. Within-target all-pairs is O(n^2).
A target with 10,000 molecules is 50M candidate pairs before any filter. Molecules per target are
therefore capped and the cap is REPORTED, not silent -- a silent cap reads as "we covered
everything" when it did not (#184).
"""
import os, sys, json, argparse, sqlite3, itertools
from multiprocessing import Pool

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ARGS = {}


def ring_attach_atom(m):
    """The acyclic heavy atom farthest (in bonds) from any ring. Works without a warhead."""
    ring = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing()]
    if not ring:
        return None
    dm = Chem.GetDistanceMatrix(m)
    best, bd = None, -1
    for a in m.GetAtoms():
        if a.IsInRing():
            continue
        d = min(dm[a.GetIdx()][r] for r in ring)
        if d > bd:
            best, bd = a.GetIdx(), d
    return best


def labels(m):
    # FLEX RULE MUST MATCH v3 EXACTLY. extract_chembl_pairs.py:94 requires BOTH bond atoms to be
    # in `near` (within 3 bonds of the reference), i.e. max(d_i,d_j)<=3. I originally wrote
    # min(...)<=3 -- EITHER atom -- which is a strictly looser rule and disagrees with v3 on
    # 48.40%% of molecules. Two corpora labelled by different rules cannot be pooled or compared,
    # and every v3-vs-this-corpus flex number computed before this fix is invalid.
    """attach_path and attach_flex for one molecule, or None."""
    at = ring_attach_atom(m)
    if at is None:
        return None
    ring = [a.GetIdx() for a in m.GetAtoms() if a.IsInRing()]
    dm = Chem.GetDistanceMatrix(m)
    path = int(min(dm[at][r] for r in ring))
    flex = 0
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if max(dm[at][i], dm[at][j]) <= 3:
            if b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1:
                flex += 1
    return path, flex


def do_target(item):
    tgt, smis = item
    tc_min, mw_win, cap_pairs = ARGS['tc'], ARGS['mw'], ARGS['cap_pairs']
    mols, fps, props = [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        lab = labels(m)
        if lab is None:
            continue
        mols.append((s, m))
        fps.append(GEN.GetFingerprint(m))
        props.append((Descriptors.MolWt(m), lab))
    out, n_cand = [], 0
    for i in range(len(mols)):
        if len(out) >= cap_pairs:
            break
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, tc in enumerate(sims):
            j = i + 1 + k
            n_cand += 1
            if tc < tc_min:
                continue
            if abs(props[i][0] - props[j][0]) > mw_win:
                continue
            (pi, fi), (pj, fj) = props[i][1], props[j][1]
            si, sj = mols[i][0], mols[j][0]
            try:
                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mols[i][1]))
            except Exception:
                sc = ''
            out.append(dict(target=tgt, input_smiles=si, output_smiles=sj, tc=round(tc, 4),
                            mw_delta=round(props[j][0] - props[i][0], 2), scaffold=sc,
                            attach_path_a=pi, attach_path_b=pj,
                            attach_path_dir='SAME' if pj == pi else ('UP' if pj > pi else 'DOWN'),
                            attach_flex_a=fi, attach_flex_b=fj,
                            attach_flex_dir='SAME' if fj == fi else ('UP' if fj > fi else 'DOWN')))
            if len(out) >= cap_pairs:
                break
    return tgt, out, len(mols), n_cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tc', type=float, default=0.4)
    ap.add_argument('--mw', type=float, default=100.0)
    ap.add_argument('--min-mols', type=int, default=20)
    ap.add_argument('--max-mols', type=int, default=1200)
    ap.add_argument('--cap-pairs', type=int, default=60000)
    ap.add_argument('--max-targets', type=int, default=400)
    ap.add_argument('--procs', type=int, default=30)
    a = ap.parse_args()
    ARGS.update(tc=a.tc, mw=a.mw, cap_pairs=a.cap_pairs)

    con = sqlite3.connect(a.db)
    cur = con.cursor()
    print('ranking targets...', flush=True)
    cur.execute("""SELECT td.chembl_id, COUNT(DISTINCT act.molregno) c
                   FROM target_dictionary td
                   JOIN assays ass ON ass.tid = td.tid
                   JOIN activities act ON act.assay_id = ass.assay_id
                   WHERE td.target_type='SINGLE PROTEIN'
                   GROUP BY td.chembl_id HAVING c >= ? ORDER BY c DESC LIMIT ?""",
                (a.min_mols, a.max_targets))
    tgts = [r[0] for r in cur.fetchall()]
    print('targets: %d' % len(tgts), flush=True)

    # PROGRESS OUTPUT IN THE FETCH LOOP. Without it this loop is SILENT for over an hour on a 30GB
    # DB, and a healthy job is indistinguishable from a hung one -- which cost me two wrong
    # diagnoses tonight and nearly a wrong kill. Any loop that can run longer than a monitoring
    # interval must say so while it runs.
    items = []
    import time as _t
    _t0 = _t.time()
    for _k, t in enumerate(tgts, 1):
        if _k % 20 == 0:
            print('  fetch %d/%d targets  (%d with enough molecules, %.1f min elapsed)'
                  % (_k, len(tgts), len(items), (_t.time() - _t0) / 60), flush=True)
        cur.execute("""SELECT DISTINCT cs.canonical_smiles
                       FROM target_dictionary td
                       JOIN assays ass ON ass.tid=td.tid
                       JOIN activities act ON act.assay_id=ass.assay_id
                       JOIN compound_structures cs ON cs.molregno=act.molregno
                       WHERE td.chembl_id=? LIMIT ?""", (t, a.max_mols))
        s = sorted({r[0] for r in cur.fetchall()})
        if len(s) >= a.min_mols:
            items.append((t, s))
    con.close()
    print('targets with molecules: %d  (max_mols cap = %d -- REPORTED, not silent)'
          % (len(items), a.max_mols), flush=True)

    n = 0
    capped = 0
    with open(a.out, 'w') as fh, Pool(a.procs) as pool:
        for k, (tgt, rows, nmol, ncand) in enumerate(pool.imap(do_target, items, chunksize=1), 1):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            n += len(rows)
            if len(rows) >= a.cap_pairs:
                capped += 1
            if k % 25 == 0:
                print('  %d/%d targets  pairs=%d  (last: %s n_mol=%d cand=%d kept=%d)'
                      % (k, len(items), n, tgt, nmol, ncand, len(rows)), flush=True)
    print('TOTAL PAIRS: %d' % n, flush=True)
    print('targets that HIT the %d-pair cap: %d (their yield is truncated, not complete)'
          % (a.cap_pairs, capped), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
