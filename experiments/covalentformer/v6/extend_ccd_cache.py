#!/usr/bin/env python3
"""Add missing ligands to the CCD atom/bond cache, from the full local components.cif.gz.

data/pdb_mirror/ccd_atoms.json was built for the ligands the corpus already uses -- 8,354 codes --
so a new programme's ligands are simply absent: 91 of the 95 ligands across the five unseen covalent
programmes. evidence_v4.ccd_mol() then returns (None, None), no ligand atom NAMES can be assigned,
and build() produces no contact table at all. The failure is silent at the call site: build()
returns None and says nothing about why.

Atom names matter here specifically. The contact table is keyed by the deposition's own atom names
(C01, N10, O13), the model is trained on those, and the reward resolves a cited distance back to the
atom that owns it. Without the CCD definition there is nothing to key the table on.

Writes a SEPARATE file rather than editing the corpus cache in place, and the caller merges it, so
nothing the corpus depends on is mutated by a run of this script.
"""
import argparse
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'pdb_mirror'))


def parse_ccd(path, want):
    """-> {code: {'atoms': [[name, element], ...], 'bonds': [[a1, a2, order], ...]}}

    Walks the mmCIF component dictionary one data_ block at a time, tracking which loop_ is open
    and the column order inside it. Column order is NOT fixed across the file, so the header is
    read every time rather than assumed.
    """
    out, cur, loop, cols = {}, None, None, []
    with gzip.open(path, 'rt', errors='ignore') as fh:
        for ln in fh:
            if ln.startswith('data_'):
                cur = ln[5:].strip()
                loop, cols = None, []
                continue
            if cur not in want:
                continue
            t = ln.strip()
            if t == 'loop_':
                loop, cols = 'pending', []
                continue
            if t.startswith('_chem_comp_atom.'):
                loop, cols = ('atom', cols + [t.split('.', 1)[1]]) if loop in ('pending', 'atom') \
                    else (loop, cols)
                continue
            if t.startswith('_chem_comp_bond.'):
                loop, cols = ('bond', cols + [t.split('.', 1)[1]]) if loop in ('pending', 'bond') \
                    else (loop, cols)
                continue
            if t.startswith('_') or t.startswith('#'):
                if loop in ('atom', 'bond'):
                    loop, cols = None, []
                continue
            if loop not in ('atom', 'bond') or not t:
                continue
            p = _split_cif(t)
            if len(p) != len(cols):
                continue
            row = dict(zip(cols, p))
            e = out.setdefault(cur, {'atoms': [], 'bonds': []})
            if loop == 'atom':
                nm, el = row.get('atom_id'), row.get('type_symbol')
                if nm and el:
                    e['atoms'].append([nm, el])
            else:
                a1, a2, vo = row.get('atom_id_1'), row.get('atom_id_2'), row.get('value_order')
                if a1 and a2:
                    e['bonds'].append([a1, a2, vo or 'SING'])
    return out


def _split_cif(line):
    """mmCIF value split that respects single and double quotes."""
    out, buf, q = [], '', None
    for ch in line:
        if q:
            if ch == q:
                q = None
            else:
                buf += ch
        elif ch in '"\'':
            q = ch
        elif ch.isspace():
            if buf:
                out.append(buf); buf = ''
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes-from', default=os.path.join(HERE, 'covalent_programmes_full.json'))
    ap.add_argument('--out', default=os.path.join(MIRROR, 'ccd_atoms_ext.json'))
    a = ap.parse_args()

    base = json.load(open(os.path.join(MIRROR, 'ccd_atoms.json')))
    prog = json.load(open(a.codes_from))
    codes = {h['ligand'] for v in prog.values() for h in v['covalent']}
    missing = sorted(c for c in codes if c not in base)
    print('ligands %d, already cached %d, missing %d'
          % (len(codes), len(codes) - len(missing), len(missing)))
    if not missing:
        print('nothing to do'); return 0

    got = parse_ccd(os.path.join(MIRROR, 'components.cif.gz'), set(missing))
    got = {k: v for k, v in got.items() if v['atoms'] and v['bonds']}
    json.dump(got, open(a.out, 'w'))
    print('extracted %d/%d  -> %s' % (len(got), len(missing), a.out))
    still = sorted(set(missing) - set(got))
    if still:
        print('no definition found for: %s' % ', '.join(still[:12]))
    for c in sorted(got)[:4]:
        print('  %-7s %3d atoms %3d bonds' % (c, len(got[c]['atoms']), len(got[c]['bonds'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
