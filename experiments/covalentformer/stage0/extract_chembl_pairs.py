"""STAGE 0 -- FRESH MMP pair extraction straight from chembl_36.db, labelled for all six params.

Nothing here reuses an earlier pair set. The old ~/edit-chem extractor is not called; the LO
corpus is not read. Molecules come out of ChEMBL 36 and pairs are built here.

WHY A NEW EXTRACTOR RATHER THAN THE OLD ONE: the old run was single-process against a 29.7 GB
SQLite on the boot disk, went I/O-bound, was pkill'ed, and left data/pairs EMPTY. This box has 32
cores and 125 GB RAM, so the whole database fits in page cache and the work shards cleanly by
target. Single-cut MMP indexing is O(atoms) per molecule, not O(n^2) pairwise, so 400 targets are
tractable in minutes rather than hours.

THE ACRYLAMIDE GATE IS GONE. Measured earlier today on 307,555 real ChEMBL pairs: demanding an
acrylamide on both sides keeps 1.80% of them, while referencing the geometry from the ATTACHMENT
point instead of the electrophile keeps 99.91% -- a 55x difference that buys nothing, because the
warhead requirement was a consequence of choosing the electrophilic carbon as the reference atom,
not a fact about the chemistry. Only `wclass` genuinely needs a warhead, since it IS the warhead's
identity. Every pair is kept and labelled with whatever it can carry.

SIX PARAMS, 3 POSE-FREE + 3 POSE-DEPENDENT. This pass computes the three pose-free ones, which
need no conformer and are exact:
    path    bond distance reference-atom -> nearest ring atom   (how far the reactive tip sits
            from the scaffold: whether the electrophile can reach the cysteine)
    flex    rotatable single bonds within 3 bonds of the reference (conformational freedom of the
            tip: the entropic cost of presenting it correctly)
    wclass  which electrophile class is present (the covalent mechanism itself)
The three pose-dependent ones -- extent, phi (Michael-acceptor planarity), exitang -- need a
conformer ensemble and are computed in a second pass over these rows, because embedding is ~1000x
the cost per molecule and must not block pair generation.

LABELS ARE RELATIVE (UP / DOWN / SAME), not absolute. A relative label needs only the SIGN of a
delta to be stable, which is a far weaker requirement than a stable magnitude -- and instability
of the per-molecule magnitude is precisely why phi was excluded before.
"""
import os, sys, json, argparse, sqlite3, itertools
from collections import defaultdict, Counter
from multiprocessing import Pool
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

WARHEADS = [
    ('acrylamide', '[CH2]=[CH]C(=O)N'), ('propiolamide', 'C#CC(=O)N'),
    ('chloroacetamide', 'ClCC(=O)N'), ('vinylsulfone', 'C=CS(=O)(=O)'),
    ('fluorosulfate', 'OS(=O)(=O)F'), ('epoxide', 'C1CO1'),
    ('boronic', 'B(O)O'), ('aldehyde', '[CX3H1](=O)'), ('nitrile', '[NX1]#[CX2]'),
]
_P = None
_GEN = None


def _init_pats():
    global _P, _GEN
    if _P is None:
        _P = [(k, Chem.MolFromSmarts(v)) for k, v in WARHEADS]
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def wclass(m):
    _init_pats()
    return tuple(k for k, p in _P if m.HasSubstructMatch(p))


def elec_atom(m):
    _init_pats()
    for _, p in _P:
        h = m.GetSubstructMatch(p)
        if h:
            return h[0]
    return None


def attach_atom(m):
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


def pose_free(m, ref):
    dm = Chem.GetDistanceMatrix(m)
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring:
        return None, None
    path = int(min(dm[ref][i] for i in ring))
    near = {i for i in range(m.GetNumAtoms()) if dm[ref][i] <= 3}
    flex = sum(1 for b in m.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
               and b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near
               and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)
    return path, flex


def frag_keys(smi, max_heavy_rgroup=14):
    """Single-cut MMP index: cut each acyclic single bond between two heavy atoms, emit
    (core_smiles, rgroup_smiles). Two molecules sharing a core differ at exactly one site."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    out = []
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if b.GetBeginAtom().GetDegree() == 1 or b.GetEndAtom().GetDegree() == 1:
            continue
        try:
            # dummyLabels=[(0,0)] PINS BOTH DUMMY ISOTOPES TO 0. Without it RDKit sets each
            # dummy's isotope to the ATOM INDEX it replaced, MolToSmiles writes that into the
            # string, and the core key becomes a function of atom NUMBERING, not of the core's
            # chemical identity. Two molecules with the SAME core hash to DIFFERENT keys
            # whenever the attachment sits at a different index -- the normal case for an MMP,
            # since the halves differ in size upstream of the cut. Measured on 2,564 real
            # CHEMBL220 molecules: 14,830 pair slots as shipped vs 35,300 with isotopes pinned,
            # i.e. the extractor was finding 42.0%% of available single-cut MMP pairs.
            # COVERAGE ONLY -- the key holds the full core SMILES, so distinct cores could not
            # collide and no false pair was ever emitted. Same root cause as the isotope split
            # leak fixed earlier tonight.
            fm = Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
            parts = Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False)
        except Exception:
            continue
        if len(parts) != 2:
            continue
        a, c = sorted(parts, key=lambda x: -x.GetNumHeavyAtoms())
        if c.GetNumHeavyAtoms() > max_heavy_rgroup:
            continue
        try:
            out.append((Chem.MolToSmiles(a), Chem.MolToSmiles(c)))
        except Exception:
            continue
    return out


_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg
    _init_pats()


def do_target(tgt_and_mols):
    """One target -> its MMP pairs, labelled. Returns (target, rows, stats)."""
    tgt, mols = tgt_and_mols
    _init_pats()
    core_idx = defaultdict(list)
    info = {}
    for smi in mols:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        hv = m.GetNumHeavyAtoms()
        if hv < 8 or hv > 70:
            continue
        info[smi] = (m, Descriptors.MolWt(m), _GEN.GetFingerprint(m))
        for core, rg in frag_keys(smi, _CFG['max_rgroup']):
            core_idx[core].append(smi)

    rows, seen = [], set()
    n_raw = 0
    for core, members in core_idx.items():
        members = list(dict.fromkeys(members))
        if len(members) < 2 or len(members) > _CFG['max_per_core']:
            continue
        for s1, s2 in itertools.combinations(members, 2):
            key = (s1, s2) if s1 < s2 else (s2, s1)
            if key in seen:
                continue
            seen.add(key)
            n_raw += 1
            m1, mw1, fp1 = info[s1]
            m2, mw2, fp2 = info[s2]
            if abs(mw1 - mw2) > _CFG['max_mw_delta']:
                continue
            tc = DataStructs.TanimotoSimilarity(fp1, fp2)
            if tc < _CFG['min_tc'] or tc > _CFG['max_tc']:
                continue
            row = dict(target=tgt, input_smiles=s1, output_smiles=s2, tc=round(tc, 4),
                       mw_delta=round(abs(mw1 - mw2), 2), core=core)
            got = False
            for mode, fn in (('attach', attach_atom), ('elec', elec_atom)):
                r1, r2 = fn(m1), fn(m2)
                if r1 is None or r2 is None:
                    continue
                p1, f1 = pose_free(m1, r1)
                p2, f2 = pose_free(m2, r2)
                if p1 is None or p2 is None:
                    continue
                row['%s_path_a' % mode], row['%s_path_b' % mode] = p1, p2
                row['%s_flex_a' % mode], row['%s_flex_b' % mode] = f1, f2
                row['%s_path_dir' % mode] = 'SAME' if p1 == p2 else ('UP' if p2 > p1 else 'DOWN')
                row['%s_flex_dir' % mode] = 'SAME' if f1 == f2 else ('UP' if f2 > f1 else 'DOWN')
                got = True
            w1, w2 = wclass(m1), wclass(m2)
            row['wclass_a'] = '+'.join(w1) if w1 else ''
            row['wclass_b'] = '+'.join(w2) if w2 else ''
            # GAIN and LOSS ARE CHANGES. The old guard `if w1 and w2` set wclass_dir only when
            # BOTH sides bore a recognised electrophile, so every pair that GAINED or LOST a warhead
            # -- the most chemically interesting transitions in the set -- got no label and dropped
            # out of the arm entirely. Sampled 20,000 such rows: 714 gains + 549 losses,
            # extrapolating to ~17,940 pairs (5.84%% of the corpus) discarded against 1,015 CHANGED
            # retained, a 17.7x loss. The build then printed a "steerable" percentage over a
            # denominator that had already excluded every gain and loss event.
            if w1 or w2:
                row['wclass_dir'] = 'SAME' if w1 == w2 else 'CHANGED'
            if got:
                rows.append(row)
    return tgt, rows, dict(n_mols=len(info), n_cores=len(core_idx), n_raw=n_raw, n_kept=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.expanduser('~/chembl_36.db'))
    ap.add_argument('--targets', type=int, default=400)
    ap.add_argument('--min-mols-per-target', type=int, default=200)
    ap.add_argument('--max-per-core', type=int, default=60)
    ap.add_argument('--max-rgroup', type=int, default=14)
    ap.add_argument('--min-tc', type=float, default=0.40)
    ap.add_argument('--max-tc', type=float, default=0.99)
    ap.add_argument('--max-mw-delta', type=float, default=200.0)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--out', default=os.path.expanduser('~/extract_run'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    con = sqlite3.connect(a.db)
    print('db: %s' % a.db, flush=True)
    con.execute('PRAGMA cache_size=-8000000')       # ~8 GB of page cache; the box has 125 GB
    # TWO QUERIES, NOT ONE. The first version had the top-targets GROUP BY as a nested subquery
    # inside the molecule fetch, so SQLite scanned `activities` (tens of millions of rows) TWICE
    # and the process sat in uninterruptible-sleep at 10% CPU -- the identical I/O stall that got
    # the previous attempt killed rather than fixed. Step 1 ranks targets, step 2 fetches only
    # those targets' molecules through an explicit IN list that the tid index can serve.
    print('step 1/2: ranking targets ...', flush=True)
    t_rows = con.execute("""
        SELECT td.chembl_id, COUNT(DISTINCT act.molregno) n
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        WHERE act.standard_value IS NOT NULL AND td.target_type = 'SINGLE PROTEIN'
        GROUP BY td.chembl_id HAVING n >= ? ORDER BY n DESC LIMIT ?
    """, (a.min_mols_per_target, a.targets)).fetchall()
    tids = [r[0] for r in t_rows]
    print('  %d targets, %d molecule-activities' % (len(tids), sum(r[1] for r in t_rows)), flush=True)

    print('step 2/2: fetching molecules ...', flush=True)
    by_t = defaultdict(set)
    CH = 50
    for i in range(0, len(tids), CH):
        chunk = tids[i:i + CH]
        ph = ','.join('?' * len(chunk))
        for tgt, smi in con.execute("""
            SELECT td.chembl_id, cs.canonical_smiles
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            JOIN target_dictionary td ON a.tid = td.tid
            JOIN compound_structures cs ON act.molregno = cs.molregno
            WHERE td.chembl_id IN (%s)
              AND act.standard_type IN ('IC50','Ki','Kd','EC50')
              AND act.standard_value IS NOT NULL
        """ % ph, chunk):
            if smi:
                by_t[tgt].add(smi)
        print('  %d/%d targets fetched, %d molecules'
              % (min(i + CH, len(tids)), len(tids), sum(len(v) for v in by_t.values())), flush=True)
    con.close()
    # sorted(), NOT list(). by_t is a defaultdict(set) keyed by SMILES STRINGS and str hashing
    # is PYTHONHASHSEED-randomised, so list(s) returns a different order every run. That order
    # decides which molecule of a pair becomes input_smiles, so EVERY UP/DOWN LABEL FLIPPED RUN
    # TO RUN. The labels were never wrong -- an MMP pair is symmetric and the balance stayed
    # ~50/50 -- but the dataset could not be regenerated, which is why a committed rebuild of
    # the splits drifted by 24 and 11 rows against the files on disk. balance_meta.json
    # advertised "seed": 20260915 for a producer that had no seeding at all. Shape of #185.
    items = [(t, sorted(s)) for t, s in by_t.items() if len(s) >= 20]
    items.sort(key=lambda x: -len(x[1]))
    print('targets: %d   molecules: %d' % (len(items), sum(len(v) for _, v in items)), flush=True)

    cfg = dict(max_per_core=a.max_per_core, max_rgroup=a.max_rgroup, min_tc=a.min_tc,
               max_tc=a.max_tc, max_mw_delta=a.max_mw_delta)
    outf = os.path.join(a.out, 'chembl36_pairs.jsonl')
    n_pairs = 0
    stats = []
    import time
    t0 = time.time()
    with open(outf, 'w') as fh, Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        # imap, NOT imap_unordered. THIS IS THE OTHER HALF OF F15 AND I SHIPPED WITHOUT IT.
        # sorted(s) pinned molecule order WITHIN a target; it did NOT pin the order the TARGET
        # BLOCKS are written in, because imap_unordered yields by worker COMPLETION TIME. Row
        # order decides which rows survive stratification and which land in valid, so two runs
        # of byte-identical code under PYTHONHASHSEED=0 print identical counts and are DIFFERENT
        # DATASETS -- demonstrated by reordering target blocks with every within-target row held
        # byte-identical: row count stayed 79,006 and held-out overlap fell to CHANCE (13.9%).
        # `items` is already sorted largest-first above, so imap costs nothing but ordering.
        for k, (tgt, rows, st) in enumerate(pool.imap(do_target, items, chunksize=1), 1):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            n_pairs += len(rows)
            stats.append(dict(target=tgt, **st))
            if k % 10 == 0 or k == len(items):
                el = time.time() - t0
                print('  %d/%d targets  %d pairs  %.0fs  (%.0f pairs/s)'
                      % (k, len(items), n_pairs, el, n_pairs / el if el else 0), flush=True)
                fh.flush()
    print('\nTOTAL PAIRS: %d' % n_pairs)
    json.dump(dict(n_pairs=n_pairs, n_targets=len(items), cfg=cfg,
                   per_target=sorted(stats, key=lambda x: -x['n_kept'])[:50]),
              open(os.path.join(a.out, 'extract_meta.json'), 'w'), indent=2)
    print('wrote %s' % outf)


if __name__ == '__main__':
    main()
