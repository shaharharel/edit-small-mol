#!/usr/bin/env python
"""Extract protein-ligand complexes WITHOUT requiring a covalent bond -- the v4b half.

WHY THIS EXISTS. extract_complexes.py demands a covalent link between a HET atom and a nucleophilic
side chain, and on the 76,840-structure mirror it reported `no_covalent_link` for 71,295 of them
(92.8%). Those are not failures; they are ordinary noncovalent complexes, and they are the entire
reason v4b can be larger than v4. The covalent requirement existed only to supply the frame -- an
origin (the attacked atom) and a +z (the attack axis). v4b's encoding prints interatomic DISTANCES
and nothing else, so it needs no origin, no axis, and therefore no bond. Dropping the requirement
costs nothing and multiplies the corpus by ~13x.

WHAT IS DELIBERATELY NOT COMPUTED. No frame. extract_complexes.py spends its hardest code on
build_frame and still had 13.6-19.8% of azimuths decided by an atom-name tie-break until tonight.
None of that applies here: there is no coordinate system to choose, so the failure mode cannot
occur. Coordinates are written RAW, straight from the file, because the only consumer
(fmt_atoms mode='invariant') reduces them to distances and never reads an absolute position.

LIGAND SELECTION is the one real judgement. JUNK is reused verbatim from extract_complexes.py so
the two halves of v4b agree on what counts as a ligand -- without it the "ligand" is routinely a
sulfate or a cryoprotectant. Multi-residue ligands (peptidyl inhibitors) are merged on the same
1.85 A rule. Structures with several distinct ligands yield one record per ligand, each keyed
pdb:resname:chain:resnum, since each is a separate pocket question.
"""
from __future__ import annotations
import argparse, collections, glob, json, math, multiprocessing as mp, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE0 = '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/stage0'
sys.path.insert(0, STAGE0)
from pocket_param_gate import parse_pdb, dist                      # noqa: E402

JUNK = {
    'HOH', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'TRS', 'MES', 'EPE',
    'IMD', 'FMT', 'ACT', 'ACY', 'DMS', 'CIT', 'NO3', 'MPD', 'BME', 'IPA', 'ETA', 'TFA', 'AZI',
    'ZN', 'MG', 'CA', 'NA', 'K', 'CL', 'BR', 'IOD', 'MN', 'FE', 'NI', 'CU', 'CD', 'HG', 'CO',
    'NAG', 'BMA', 'MAN', 'FUC', 'GAL', 'GLC', 'SIA', 'XYP', 'EOH', 'URE', 'SCN', 'FLC',
}
MIN_LIG_ATOMS = 6
MERGE_BOND = 1.85
POCKET_R = 12.0
HYDROPHOBIC, HBOND, IONIC = 4.5, 3.5, 4.0
AROM_RES = {'PHE', 'TYR', 'TRP', 'HIS'}


def interaction_fingerprint(prot, lig):
    """Per-residue contact types. No covalent exclusion: there is no warhead to exclude, so every
    contact is a recognition contact. Same thresholds and same type vocabulary as the covalent
    extractor, so contacts rows from both halves are the same task."""
    fp = collections.defaultdict(set)
    for pa in prot:
        pe = pa[7].upper()
        for la in lig:
            d = dist(pa[4:7], la[4:7])
            if d > HYDROPHOBIC:
                continue
            le = la[7].upper()
            if pe == 'C' and le == 'C':
                fp['%s.%s.%s.hydrophobic' % (pa[1], pa[2], pa[3])].add(1)
            if pe in ('N', 'O') and le in ('N', 'O') and d <= HBOND:
                fp['%s.%s.%s.hbond' % (pa[1], pa[2], pa[3])].add(1)
            if pa[1] in AROM_RES and le == 'C':
                fp['%s.%s.%s.aromatic' % (pa[1], pa[2], pa[3])].add(1)
            if ((pa[1] in ('ASP', 'GLU') and le == 'N') or
                    (pa[1] in ('LYS', 'ARG') and le == 'O')) and d <= IONIC:
                fp['%s.%s.%s.ionic' % (pa[1], pa[2], pa[3])].add(1)
    return sorted(fp.keys())


def _work(t):
    pdb_id, path = t
    try:
        prot, het = parse_pdb(path)
    except Exception:
        return [{'pdb': pdb_id, 'status': 'parse_error'}]
    if not prot or not het:
        return [{'pdb': pdb_id, 'status': 'no_protein_or_het'}]

    groups = collections.defaultdict(list)
    for h in het:
        if h[1] in JUNK:
            continue
        groups[(h[1], h[2], h[3])].append(h)
    if not groups:
        return [{'pdb': pdb_id, 'status': 'no_candidate_ligand'}]

    # merge HET residues that are covalently close to each other (peptidyl / multi-residue ligands)
    keys = list(groups)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if find(ka) == find(kb):
                continue
            close = any(dist(a[4:7], b[4:7]) < MERGE_BOND
                        for a in groups[ka] for b in groups[kb])
            if close:
                parent[find(ka)] = find(kb)
    merged = collections.defaultdict(list)
    for k in keys:
        merged[find(k)].extend(groups[k])

    out = []
    for key, atoms in merged.items():
        if len(atoms) < MIN_LIG_ATOMS:
            out.append({'pdb': pdb_id, 'status': 'ligand_too_small'})
            continue
        poc = [p for p in prot
               if any(dist(p[4:7], a[4:7]) <= POCKET_R for a in atoms)]
        if not poc:
            out.append({'pdb': pdb_id, 'status': 'no_pocket'})
            continue
        ifp = interaction_fingerprint(poc, atoms)
        if not ifp:
            out.append({'pdb': pdb_id, 'status': 'no_interactions'})
            continue
        out.append({
            'pdb': '%s:%s:%s:%s' % (pdb_id, key[0], key[1], key[2]),
            'src_pdb': pdb_id,
            'status': 'ok',
            'ligand': {'resname': key[0], 'chain': key[1], 'resnum': key[2],
                       'n_atoms': len(atoms)},
            # RAW coordinates, no frame. The invariant encoder reduces these to distances.
            'ligand_atoms': [{'name': a[0], 'el': a[7], 'xyz': [round(c, 3) for c in a[4:7]]}
                             for a in atoms],
            'pocket': [{'name': p[0], 'res': p[1], 'resnum': p[3], 'el': p[7],
                        'xyz': [round(c, 3) for c in p[4:7]]} for p in poc],
            'ifp': ifp, 'n_ifp': len(ifp), 'n_pocket': len(poc),
        })
    return out or [{'pdb': pdb_id, 'status': 'no_usable_ligand'}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pdb_dir, '*.pdb')))
    if a.limit:
        files = files[:a.limit]
    tasks = [(os.path.basename(f).split('.')[0].upper(), f) for f in files]
    print('structures to process: %d' % len(tasks), flush=True)

    tally = collections.Counter()
    n_ok = 0
    # fork, not spawn: spawn re-imports __main__ and on macOS silently produced a pool that never
    # started a worker (parent at 0% CPU for an hour).
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as pool, open(a.out, 'w') as fh:
        for i, recs in enumerate(pool.imap_unordered(_work, tasks, chunksize=16)):
            for r in recs:
                tally[r['status']] += 1
                if r['status'] == 'ok':
                    n_ok += 1
                fh.write(json.dumps(r) + '\n')
            if (i + 1) % 2000 == 0:
                print('  %d/%d structures, %d ligand records ok' % (i + 1, len(tasks), n_ok),
                      flush=True)
    print('\nDONE %s' % a.out)
    print('ligand records ok: %d from %d structures' % (n_ok, len(tasks)))
    print('status breakdown: %s' % dict(tally.most_common()))
    print('FAILURES ARE IN THE FILE with a status field -- coverage is read from it, not assumed.')


if __name__ == '__main__':
    main()
