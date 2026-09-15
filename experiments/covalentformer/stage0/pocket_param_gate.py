#!/usr/bin/env python
"""GATE, run BEFORE building the pocket param pipeline: do these quantities actually VARY?

PRE-REGISTERED FALSIFIER, stated before any number is printed:

  A param is DEAD as a steering target if its spread across the corpus is at or below the
  spread of the underlying geometric constant it is built from.

  d_cys died exactly this way: d(electrophile->Cys) looked like a rich 3D descriptor and is
  the formed C-S BOND, sd 0.106 A, with every residue type sitting on its textbook value.
  There was nothing to steer. theta_BD is the same shape of risk -- a covalent adduct's
  Nu-C-C angle is a VALENCE ANGLE, and valence angles are stiff.

  PASS for theta_BD  : sd >= 5 deg AND p95-p05 >= 15 deg  (real conformational range)
  FAIL for theta_BD  : sd <  5 deg                        (it is a valence angle, kill it)

  PASS for burial    : p95/p05 >= 2.0 across ligands in the SAME pocket
                       (same-pocket is the honest denominator: across different proteins
                        burial varies because POCKETS differ, which the model cannot steer)

I am writing the thresholds down now so that neither outcome can be narrated after the fact.

theta_BD here is the angle S(gamma) - C(electrophile) - C(carbonyl), i.e. the nucleophile's
approach axis against the attacked carbon. In a formed adduct this is the post-reaction angle;
that is the correct thing to measure, because the corpus is crystal structures of ADDUCTS.
"""
from __future__ import annotations
import os, sys, json, math, argparse, collections, statistics

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
PDB_DIR = os.path.join(ROOT, 'data/covbinder/raw_covindb2/PDB')

NUC = {'CYS': 'SG', 'SER': 'OG', 'THR': 'OG1', 'TYR': 'OH', 'LYS': 'NZ'}


def parse_pdb(path):
    """FIRST MODEL ONLY. 18/3445 files are multi-model NMR entries; stacking their MODELs
    inflated CYS burial sd to 58.8 against a mean of 31.8 last time. Break on ENDMDL."""
    prot, het = [], []
    with open(path, errors='ignore') as fh:
        for line in fh:
            rec = line[:6]
            if rec == 'ENDMDL':
                break
            if rec in ('ATOM  ', 'HETATM'):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                name = line[12:16].strip(); res = line[17:20].strip()
                el = line[76:78].strip() or name[:1]
                if el == 'H':
                    continue
                rec_t = (name, res, line[21], line[22:26].strip(), x, y, z, el)
                (prot if rec == 'ATOM  ' else het).append(rec_t)
    return prot, het


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def angle(a, b, c):
    """Angle a-b-c in degrees."""
    v1 = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    v2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    n1 = math.sqrt(sum(t*t for t in v1)); n2 = math.sqrt(sum(t*t for t in v2))
    if n1 == 0 or n2 == 0:
        return None
    cs = sum(v1[i]*v2[i] for i in range(3)) / (n1*n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cs))))


# Shrake-Rupley sphere points (fixed, deterministic -- no RNG, so no seed to forget)
def sphere_points(n=92):
    pts = []
    inc = math.pi * (3 - math.sqrt(5)); off = 2.0 / n
    for k in range(n):
        y = k * off - 1 + off / 2
        r = math.sqrt(max(0.0, 1 - y*y)); phi = k * inc
        pts.append((math.cos(phi)*r, y, math.sin(phi)*r))
    return pts


SPH = sphere_points(92)
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'F': 1.47,
       'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'SE': 1.90}


def sasa(atoms, context=None, probe=1.4):
    """Solvent-accessible surface area of `atoms`, optionally occluded by `context`."""
    allat = atoms + (context or [])
    total = 0.0
    for i, a in enumerate(atoms):
        ra = VDW.get(a[7].upper(), 1.70) + probe
        nb = [b for b in allat if b is not a and dist(a[4:7], b[4:7]) < ra + VDW.get(b[7].upper(), 1.70) + probe]
        acc = 0
        for p in SPH:
            px = (a[4] + ra*p[0], a[5] + ra*p[1], a[6] + ra*p[2])
            if all(dist(px, b[4:7]) >= VDW.get(b[7].upper(), 1.70) + probe for b in nb):
                acc += 1
        total += 4 * math.pi * ra * ra * acc / len(SPH)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/pocket_pairs/pairs.jsonl')
    ap.add_argument('--max-pdb', type=int, default=260)
    a = ap.parse_args()

    want = []
    seen = set()
    for line in open(a.pairs):
        r = json.loads(line)
        for k in ('pdb_a', 'pdb_b'):
            if r[k] not in seen:
                seen.add(r[k]); want.append((r[k], r['protein']))
    have = set(f[:-4].upper() for f in os.listdir(PDB_DIR) if f.lower().endswith('.pdb'))
    want = [(p, pr) for p, pr in want if p.upper() in have][:a.max_pdb]
    print('PDBs available for the pocket pair set: %d of %d needed'
          % (len(seen & have), len(seen)))
    print('measuring %d ...' % len(want)); sys.stdout.flush()

    thetas = collections.defaultdict(list)
    burial_by_prot = collections.defaultdict(list)
    n_done = 0
    for pdb, prot_id in want:
        try:
            prot, het = parse_pdb(os.path.join(PDB_DIR, pdb + '.pdb'))
        except Exception:
            continue
        if not prot or not het:
            continue
        # ligand = largest HETATM residue that is not water/ion
        groups = collections.defaultdict(list)
        for h in het:
            if h[1] in ('HOH', 'WAT', 'SO4', 'PO4', 'GOL', 'EDO', 'NAG', 'ZN', 'MG', 'NA', 'CL'):
                continue
            groups[(h[1], h[2], h[3])].append(h)
        if not groups:
            continue
        lig = max(groups.values(), key=len)
        if len(lig) < 8:
            continue

        # --- theta_BD: nucleophile - attacked C - neighbouring C
        best = None
        for p in prot:
            if p[1] in NUC and p[0] == NUC[p[1]]:
                for lat in lig:
                    if lat[7].upper() != 'C':
                        continue
                    d = dist(p[4:7], lat[4:7])
                    if d < 2.2 and (best is None or d < best[0]):
                        best = (d, p, lat)
        if best is not None:
            _, nuc, catom = best
            nbrs = sorted((l for l in lig if l is not catom and dist(l[4:7], catom[4:7]) < 1.75),
                          key=lambda l: dist(l[4:7], catom[4:7]))
            if nbrs:
                th = angle(nuc[4:7], catom[4:7], nbrs[0][4:7])
                if th is not None:
                    thetas[nuc[1]].append(th)

        # --- burial: fraction of ligand SASA occluded by the protein
        near = [p for p in prot if any(dist(p[4:7], l[4:7]) < 12.0 for l in lig[:6])]
        free = sasa(lig)
        bound = sasa(lig, context=near)
        if free > 1.0:
            burial_by_prot[prot_id].append(100.0 * (free - bound) / free)
        n_done += 1
        if n_done % 40 == 0:
            print('  %d/%d' % (n_done, len(want))); sys.stdout.flush()

    print('\n================ PRE-REGISTERED GATE ================')
    print('--- theta_BD (nucleophile - attacked C - alpha C), degrees ---')
    print('%-6s %6s %8s %8s %8s %8s' % ('res', 'n', 'mean', 'sd', 'p05', 'p95'))
    allth = []
    for res, v in sorted(thetas.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 3:
            continue
        v2 = sorted(v); allth += v
        print('%-6s %6d %8.2f %8.3f %8.2f %8.2f'
              % (res, len(v), statistics.mean(v), statistics.pstdev(v),
                 v2[int(.05*len(v2))], v2[int(.95*len(v2))]))
    if allth:
        s = statistics.pstdev(allth); q = sorted(allth)
        rng = q[int(.95*len(q))] - q[int(.05*len(q))]
        print('POOLED n=%d  sd=%.3f  p95-p05=%.2f' % (len(allth), s, rng))
        print('VERDICT theta_BD: %s  (PASS needs sd>=5 AND p95-p05>=15)'
              % ('PASS -- real conformational range' if (s >= 5 and rng >= 15)
                 else 'FAIL -- this is a VALENCE ANGLE, kill it like d_cys'))

    print('\n--- ligand burial (%% of ligand SASA occluded), WITHIN the same protein ---')
    ratios = []
    print('%-10s %5s %8s %8s %8s' % ('protein', 'n', 'mean', 'p05', 'p95'))
    for prot_id, v in sorted(burial_by_prot.items(), key=lambda kv: -len(kv[1]))[:8]:
        if len(v) < 3:
            continue
        v2 = sorted(v); lo = v2[int(.05*len(v2))] or 1e-9; hi = v2[int(.95*len(v2))]
        ratios.append(hi / lo)
        print('%-10s %5d %8.1f %8.1f %8.1f' % (prot_id, len(v), statistics.mean(v), lo, hi))
    if ratios:
        m = statistics.median(ratios)
        print('median within-pocket p95/p05 = %.2fx   (PASS needs >= 2.0)' % m)
        print('VERDICT burial: %s' % ('PASS -- steerable within a fixed pocket'
                                      if m >= 2.0 else
                                      'WEAK -- mostly a property of the POCKET, not the ligand'))


if __name__ == '__main__':
    main()
