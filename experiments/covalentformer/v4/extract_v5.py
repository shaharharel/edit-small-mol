#!/usr/bin/env python
"""PER-COMPLEX EXTRACTOR — the shared foundation for CovRXN and CDL.

For each covalent co-crystal, emit ONE json record carrying everything both architectures need:
  reactive residue + nucleophilic atom, electrophilic ligand atom, covalent bond geometry,
  pocket atoms expressed in the REACTION-CENTRIC FRAME, and the non-covalent INTERACTION
  FINGERPRINT with the covalent bond and its flanking atoms excluded.

THE FRAME (CovRXN's core trick, and it removes any need for equivariant layers):
    origin = C_E (electrophilic carbon)
    z      = unit(C_E -> Nu)          the attack vector
    x      = unit(C_E -> O_carbonyl), Gram-Schmidt orthogonalised against z
    y      = z x x
Coordinates in this frame are invariant to global rotation/translation BY CONSTRUCTION.

WHAT IS DELIBERATELY NOT HERE: d(C_E-Nu) as a training target. It is a bond length (sd ~0.11 A)
and CovDocker concedes the point by SAMPLING it from N(mu,sigma) rather than predicting it. It is
recorded for QA -- if the distribution is not tight, the covalent-bond detection is wrong -- but it
is not a label.

FAILURES ARE RECORDED, NOT DROPPED. Every input id appears in the output with a `status`, so
coverage is read off the file rather than inferred from how many records happen to exist.
"""
from __future__ import annotations
import os, sys, json, math, glob, argparse, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_param_gate import parse_pdb, dist, angle               # noqa: E402

# EXTENDED nucleophile table. pocket_param_gate.NUC covers only CYS/SER/THR/TYR/LYS, which
# silently excluded every glycosidase (Asp/Glu mechanism-based inhibitors) and His-nucleophile
# complex. Verified by hand on PDB LINK records: 6Q6K OE2 GLU->C32, 1JZ2 OE2 GLU->C1,
# 5NPB OD2 ASP->C1, 1HNE NE2 HIS->C1 were all reported as `no_covalent_link`.
NUC = {'CYS': ['SG'], 'SER': ['OG'], 'THR': ['OG1'], 'TYR': ['OH'], 'LYS': ['NZ'],
       'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2'], 'HIS': ['NE2', 'ND1'], 'SEC': ['SE']}

# HET groups that are never the covalent ligand. Without this the "ligand" is frequently a
# sulfate or a cryoprotectant that happens to sit near a Cys.
JUNK = {
    'HOH', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'TRS', 'MES', 'EPE',
    'IMD', 'FMT', 'ACT', 'ACY', 'DMS', 'CIT', 'NO3', 'MPD', 'BME', 'IPA', 'ETA', 'TFA', 'AZI',
    'ZN', 'MG', 'CA', 'NA', 'K', 'CL', 'BR', 'IOD', 'MN', 'FE', 'NI', 'CU', 'CD', 'HG', 'CO',
    'NAG', 'BMA', 'MAN', 'FUC', 'GAL', 'GLC', 'SIA', 'XYP', 'EOH', 'URE', 'SCN', 'FLC',
}
# PER-ELEMENT covalent cutoffs. A single 2.15 A cutoff plus a C/N/S/P element whitelist deleted
# EVERY BORON electrophile from the corpus -- verified: 1,456 extracted complexes contained
# {C:1353, P:63, S:33, N:7} and ZERO B, while covalent_filter.ACCEPTED lists boronic_acid. That
# is a chemically coherent deletion (beta-lactamases, thrombin, proteasome), not random loss.
ELEC_MAX = {'C': 2.15, 'N': 2.15, 'S': 2.30, 'P': 2.30, 'B': 1.80, 'SI': 2.30, 'SE': 2.40}
COV_MAX = max(ELEC_MAX.values())
MIN_LIG_ATOMS = 6
MERGE_BOND = 1.85       # HET residues closer than this are ONE ligand (peptidyl inhibitors)
POCKET_R = 12.0

# The atom used as the far reference for the attack angle, PER RESIDUE. The previous code took
# the first of ('CB','CD','CZ','CE') in FILE order, which is CB for every residue -- giving LYS a
# 3-bond-distant reference. Measured consequence: CYS angles were correct (mean 105.3 deg,
# Burgi-Dunitz) but LYS ran to 177.0 deg and TYR to 149.1 deg, i.e. impossible valence angles.
NUC_CB = {'CYS': 'CB', 'SER': 'CB', 'THR': 'CB', 'TYR': 'CZ', 'LYS': 'CE',
          'ASP': 'CG', 'GLU': 'CD', 'HIS': 'CE1', 'SEC': 'CB'}

IFP_EXCLUDE_R = 2.6     # ~2 bonds from C_E
HYDROPHOBIC = 4.5       # C...C
HBOND = 3.5             # N/O...N/O
IONIC = 4.0
AROM_RES = {'PHE', 'TYR', 'TRP', 'HIS'}


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _norm(v):
    n = math.sqrt(sum(t*t for t in v))
    return (v[0]/n, v[1]/n, v[2]/n) if n > 1e-9 else None


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(a[i]*b[i] for i in range(3))


def dihedral(p0, p1, p2, p3):
    b0 = _sub(p0, p1); b1 = _sub(p2, p1); b2 = _sub(p3, p2)
    b1n = _norm(b1)
    if b1n is None:
        return None
    v = tuple(b0[i] - _dot(b0, b1n)*b1n[i] for i in range(3))
    w = tuple(b2[i] - _dot(b2, b1n)*b1n[i] for i in range(3))
    x = _dot(v, w); y = _dot(_cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))


def find_covalent_link(prot, het):
    """Return (nuc_atom, elec_atom, ligand_atoms, lig_key) for the best covalent link, or None.

    'Best' = shortest qualifying nucleophile->ligand distance. Multiple Cys can sit near a
    ligand; only one is bonded, and the bonded one is decisively closer.
    """
    groups = collections.defaultdict(list)
    for a in het:
        if a[1] in JUNK:
            continue
        groups[(a[1], a[2], a[3])].append(a)

    # MERGE BONDED HET RESIDUES BEFORE SIZE-FILTERING. Peptidyl inhibitors are deposited as
    # several HET residues in one chain (e.g. 5PAD: PHQ-GLY-GLY-0QE, with the covalent link on
    # 0QE which has ONE atom). Filtering per residue shredded them: each fragment fell under
    # MIN_LIG_ATOMS and the whole complex was recorded as `no_covalent_link`.
    keys = list(groups)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k

    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[1] != kb[1]:                    # same chain only
                continue
            if any(dist(x[4:7], y[4:7]) <= MERGE_BOND for x in groups[ka] for y in groups[kb]):
                parent[find(ka)] = find(kb)
    merged = collections.defaultdict(list)
    for k in keys:
        merged[find(k)].extend(groups[k])
    ligs = {k: v for k, v in merged.items() if len(v) >= MIN_LIG_ATOMS}
    if not ligs:
        return None

    nucs = [a for a in prot if a[1] in NUC and a[0] in NUC[a[1]]]
    cands = []
    for nu in nucs:
        for key, atoms in ligs.items():
            for la in atoms:
                el = la[7].upper()
                lim = ELEC_MAX.get(el)
                if lim is None:                   # not a plausible electrophile element
                    continue
                d = dist(nu[4:7], la[4:7])
                if d <= lim:
                    # TIE-BREAK on (distance, atom name, residue key). A bare `<` min with no
                    # tie-break makes the chosen link -- and therefore the whole frame -- depend
                    # on dict/file ordering when two distances are equal.
                    cands.append((round(d, 4), la[0], str(key), nu, la, atoms, key, d))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1], t[2]))
    b = cands[0]
    return b[3], b[4], b[5], b[6], b[7]


def _ligand_core(lig_atoms):
    """Murcko-like graph core: iteratively delete degree-1 atoms from the ligand's geometric graph.
    What survives is the ring systems plus the linkers between them, i.e. the scaffold. Returns an
    empty set for a fully acyclic ligand."""
    n = len(lig_atoms)
    adj = collections.defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if dist(lig_atoms[i][4:7], lig_atoms[j][4:7]) < 1.8:
                adj[i].add(j); adj[j].add(i)
    alive = set(range(n))
    changed = True
    while changed:
        changed = False
        for i in list(alive):
            if len(adj[i] & alive) <= 1:
                alive.discard(i); changed = True
    return alive


def _centroid(pts):
    k = len(pts)
    return (sum(p[0] for p in pts) / k, sum(p[1] for p in pts) / k, sum(p[2] for p in pts) / k)


def build_frame(elec, nuc, lig_atoms):
    """Orthonormal frame from the reaction itself. Returns (origin, (ex,ey,ez), ref_name).

    +x NOW POINTS AT THE SCAFFOLD CENTROID, NOT AT A CHOSEN NEIGHBOUR ATOM. The previous rule picked
    the neighbour of C_E whose connected component was largest, tie-broken on (-size, distance, atom
    name). That tie-break is arbitrary, and it fired more often than anyone checked: measured on the
    extracted corpora, the winning component size was TIED with a runner-up in 13.6% of the original
    1,693 complexes and 19.8% of the 5,256 newly parsed ones. In those cases +x was decided by PDB
    atom naming, so the azimuth -- the one channel carrying pocket signal -- was noise. The failure
    mode is structural: when C_E sits in a ring, deleting it disconnects nothing, so every neighbour
    reports the same component and the rule cannot discriminate even in principle.

    A centroid has no tie-break. It is a continuous function of the coordinates, so exact ties have
    measure zero, and it is defined whether or not C_E is in a ring. Precedence:
      1. centroid of the Murcko-like graph core (ring systems + linkers)   -- the chemist's scaffold
      2. centroid of all other ligand heavy atoms                          -- acyclic ligands
      3. the old largest-component neighbour rule                          -- last resort
    Each is recorded in frame_x_ref so the corpus can be audited by which rule fired.
    """
    z = _norm(_sub(nuc[4:7], elec[4:7]))
    if z is None:
        return None
    # ---- preferred: scaffold centroid, projected perpendicular to z ----
    for pts, nm in (
        ([lig_atoms[i][4:7] for i in _ligand_core(lig_atoms)], 'murcko_core_centroid'),
        ([a[4:7] for a in lig_atoms if a is not elec], 'ligand_centroid'),
    ):
        if not pts:
            continue
        raw = _sub(_centroid(pts), elec[4:7])
        proj = _dot(raw, z)
        perp = tuple(raw[i] - proj * z[i] for i in range(3))
        # ILL-CONDITIONING GUARD, on the UNNORMALISED length: if the scaffold centroid sits almost on
        # the attack axis, its perpendicular component is tiny and its direction is dominated by
        # coordinate error. 0.30 A is ~2x typical crystallographic coordinate error.
        if math.sqrt(sum(c * c for c in perp)) < 0.30:
            continue
        x = _norm(perp)
        if x is None:
            continue
        return elec[4:7], (x, _cross(z, x), z), '%s(n=%d)' % (nm, len(pts))
    # ---- last resort: the original discrete rule, kept verbatim below ----
    # x reference = C_E -> SCAFFOLD-BEARING SUBSTITUENT: of the atoms bonded to C_E, the one
    # whose connected component (with C_E deleted) is LARGEST. That is the exit vector a chemist
    # reasons about, and unlike "the carbonyl oxygen" it is defined for EVERY electrophile --
    # boronates, nitriles, phosphorus, epoxides included.
    #
    # WHY THIS REPLACED THE CARBONYL RULE: the old rule found a carbonyl O for only 54.3% of
    # complexes and fell back to an ARBITRARY nearest neighbour for the other 45.7%. Those frames
    # had their octants randomly rotated about z -- and azimuth is exactly where the surviving
    # pocket signal lives (directional u_y delta +0.1462, axial u_z +0.0189). Half the corpus was
    # contributing noise to the one channel that works.
    nbrs = sorted([a for a in lig_atoms
                   if a is not elec and dist(elec[4:7], a[4:7]) < 1.8],
                  key=lambda a: (round(dist(elec[4:7], a[4:7]), 4), a[0]))
    ref, refname = None, None
    if nbrs:
        # connected components of the ligand graph with C_E removed
        rest = [a for a in lig_atoms if a is not elec]
        adj = collections.defaultdict(list)
        for i, a in enumerate(rest):
            for j in range(i + 1, len(rest)):
                if dist(a[4:7], rest[j][4:7]) < 1.8:
                    adj[i].append(j); adj[j].append(i)
        idx = {id(a): i for i, a in enumerate(rest)}

        def comp_size(start):
            seen, stack = {start}, [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            return len(seen)
        best = None
        for a in nbrs:
            i = idx.get(id(a))
            if i is None:
                continue
            s = comp_size(i)
            # tie-break on (-size, distance, name) so the choice is deterministic
            key = (-s, round(dist(elec[4:7], a[4:7]), 4), a[0])
            if best is None or key < best[0]:
                best = (key, a, s)
        if best is not None:
            ref, refname = best[1], 'scaffold_substituent(n=%d)' % best[2]
    if ref is None:
        return None
    # DEGENERACY GUARD: if C_E->ref is near-collinear with z, Gram-Schmidt is ill-conditioned and
    # the azimuth is meaningless. Fall back to the next neighbour; record it either way.
    _z = _norm(_sub(nuc[4:7], elec[4:7]))
    _r = _norm(_sub(ref[4:7], elec[4:7]))
    if _z and _r and abs(_dot(_z, _r)) > 0.9:
        alt = [a for a in nbrs if a is not ref]
        if alt:
            ref, refname = alt[0], refname + '+degenerate_fallback'
        else:
            refname = refname + '+DEGENERATE'
    raw = _sub(ref[4:7], elec[4:7])
    proj = _dot(raw, z)
    x = _norm(tuple(raw[i] - proj*z[i] for i in range(3)))
    if x is None:
        return None
    y = _cross(z, x)
    return elec[4:7], (x, y, z), refname


def to_frame(pt, origin, axes):
    v = _sub(pt, origin)
    return [round(_dot(v, axes[0]), 3), round(_dot(v, axes[1]), 3), round(_dot(v, axes[2]), 3)]


def interaction_fingerprint(prot, lig_atoms, elec, nuc):
    """Per-residue non-covalent contact types. THE COVALENT BOND AND ITS FLANKING ATOMS ARE
    EXCLUDED -- that is what makes this the RECOGNITION state rather than a restatement of the
    reaction. Excluded: the electrophile, the nucleophilic atom, and the nucleophile's residue."""
    nuc_res = (nuc[1], nuc[2], nuc[3])
    # EXCLUDE THE WHOLE WARHEAD NEIGHBOURHOOD, not just C_E. Removing one atom left the carbonyl
    # O, the vinyl carbons and the leaving group all scored, so a warhead SWAP mechanically
    # perturbed the fingerprint -- making "recognition" partly a restatement of the reaction,
    # which is precisely the confound this channel exists to avoid.
    lig = [a for a in lig_atoms
           if a is not elec and dist(a[4:7], elec[4:7]) > IFP_EXCLUDE_R]
    fp = collections.defaultdict(set)
    for pa in prot:
        pres = (pa[1], pa[2], pa[3])
        if pres == nuc_res:
            continue                                   # the reacting residue is not recognition
        pe = pa[7].upper()
        for la in lig:
            d = dist(pa[4:7], la[4:7])
            if d > HYDROPHOBIC:
                continue
            le = la[7].upper()
            if pe == 'C' and le == 'C' and d <= HYDROPHOBIC:
                fp['%s.%s.%s.hydrophobic' % (pa[1], pa[2], pa[3])].add(1)
            if pe in ('N', 'O') and le in ('N', 'O') and d <= HBOND:
                fp['%s.%s.%s.hbond' % (pa[1], pa[2], pa[3])].add(1)
            if pa[1] in AROM_RES and le == 'C' and d <= HYDROPHOBIC:
                fp['%s.%s.%s.aromatic' % (pa[1], pa[2], pa[3])].add(1)
            if ((pa[1] in ('ASP', 'GLU') and le == 'N') or
                    (pa[1] in ('LYS', 'ARG') and le == 'O')) and d <= IONIC:
                fp['%s.%s.%s.ionic' % (pa[1], pa[2], pa[3])].add(1)
    return sorted(fp.keys())


def extract_one(pdb_id, path):
    rec = {'pdb': pdb_id, 'status': 'ok'}
    try:
        prot, het = parse_pdb(path)
    except Exception as e:
        rec['status'] = 'parse_fail:%s' % type(e).__name__
        return rec
    if not prot or not het:
        rec['status'] = 'no_protein_or_het'
        return rec

    link = find_covalent_link(prot, het)
    if link is None:
        rec['status'] = 'no_covalent_link'
        return rec
    nuc, elec, lig_atoms, lig_key, d_cov = link

    fr = build_frame(elec, nuc, lig_atoms)
    if fr is None:
        rec['status'] = 'frame_fail'
        return rec
    origin, axes, refname = fr

    # Bond geometry. d_cov is recorded for QA ONLY -- it is a bond length, not a label (see header).
    want = NUC_CB.get(nuc[1])
    cb = next((a for a in prot
               if (a[1], a[2], a[3]) == (nuc[1], nuc[2], nuc[3]) and a[0] == want), None)
    rec['nuc'] = {'res': nuc[1], 'chain': nuc[2], 'resnum': nuc[3], 'atom': nuc[0]}
    rec['elec'] = {'name': elec[0], 'element': elec[7]}
    rec['ligand'] = {'resname': lig_key[0], 'chain': lig_key[1], 'resnum': lig_key[2],
                     'n_atoms': len(lig_atoms)}
    rec['bond'] = {
        'd_cov_QA_ONLY': round(d_cov, 3),
        'angle_elec_nuc_cb': (round(angle(elec[4:7], nuc[4:7], cb[4:7]), 2)
                              if cb and angle(elec[4:7], nuc[4:7], cb[4:7]) is not None else None),
        'frame_x_ref': refname,
    }
    if cb is not None:
        nbr = [a for a in lig_atoms if a is not elec and dist(elec[4:7], a[4:7]) < 1.8]
        if nbr:
            dh = dihedral(nbr[0][4:7], elec[4:7], nuc[4:7], cb[4:7])
            rec['bond']['dihedral_nbr_elec_nuc_cb'] = round(dh, 2) if dh is not None else None

    pocket = [a for a in prot if dist(a[4:7], origin) <= POCKET_R]
    rec['pocket'] = [{'name': a[0], 'res': a[1], 'resnum': a[3], 'el': a[7],
                      'xyz': to_frame(a[4:7], origin, axes),
                      'is_nuc_res': (a[1], a[2], a[3]) == (nuc[1], nuc[2], nuc[3])}
                     for a in pocket]
    rec['ligand_atoms'] = [{'name': a[0], 'el': a[7], 'xyz': to_frame(a[4:7], origin, axes),
                            'is_elec': a is elec} for a in lig_atoms]
    rec['ifp'] = interaction_fingerprint(prot, lig_atoms, elec, nuc)
    rec['n_pocket'] = len(pocket)
    rec['n_ifp'] = len(rec['ifp'])
    return rec


def _work(f):
    pid = os.path.basename(f)[:-4].upper()
    try:
        return extract_one(pid, f)
    except Exception as e:
        return {'pdb': pid, 'status': 'exception:%s' % type(e).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb-dir', default='data/pdb')
    ap.add_argument('--out', default='data/complexes.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', default='', help='comma list of PDB ids (for the synchronous probe)')
    ap.add_argument('--workers', type=int, default=30)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pdb_dir, '*.pdb')))
    if a.only:
        keep = {x.strip().upper() for x in a.only.split(',') if x.strip()}
        files = [f for f in files if os.path.basename(f)[:-4].upper() in keep]
    if a.limit:
        files = files[:a.limit]
    print('structures to process: %d' % len(files)); sys.stdout.flush()

    stat = collections.Counter()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    # PARALLEL. The serial loop leaves 31 of 32 cores idle; at 76,840 structures that is the
    # difference between an afternoon and a day. Each structure is independent -- no shared state --
    # so a Pool over files is safe. 'fork' is used explicitly because the default start method
    # re-imports the module per worker and can deadlock.
    import multiprocessing as mp
    with open(a.out, 'w') as fo:
        ctx = mp.get_context('fork')
        with ctx.Pool(a.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_work, files, chunksize=16)):
                stat[rec['status']] += 1
                fo.write(json.dumps(rec) + '\n')
            if (i + 1) % 100 == 0 or i + 1 == len(files):
                print('  %4d/%4d  %s' % (i + 1, len(files), dict(stat))); sys.stdout.flush()

    ok = stat['ok']
    print('\nDONE %s  |  ok %d/%d (%.1f%%)' % (a.out, ok, len(files), 100.0*ok/max(len(files), 1)))
    print('status breakdown: %s' % dict(stat))
    print('FAILURES ARE IN THE FILE with a status field -- coverage is read from it, not assumed.')


if __name__ == '__main__':
    main()
