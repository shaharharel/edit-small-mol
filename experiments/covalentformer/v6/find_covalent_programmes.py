#!/usr/bin/env python3
"""Find deposited COVALENT complexes for a list of UniProt targets, from the local mirror.

WHY THIS EXISTS
---------------
The corpus covers 700 targets and ZAP70 is not one of them, so mol1 -- the only case study the
report shows -- runs on an unseen target AND a predicted pose at the same time. Those two shifts
cannot be separated from each other there. A programme with a real deposited covalent crystal and no
rows in the corpus separates them: unseen target, measured geometry.

Picking those programmes from memory is how wrong PDB codes get into a report. Web search returned
conflicting and low-authority IDs for the same compounds, so this asks the PDB instead: RCSB resolves
UniProt -> entry ids, and the covalent bond is then verified locally, from coordinates, with the same
cov_check that built the training corpus. Nothing here is asserted from recall.

WHAT COUNTS AS COVALENT
-----------------------
A ligand heavy atom within bonding distance of a side-chain nucleophile, judged by
evidence_v4.cov_check, which compares against the ideal bond length for that element pair rather
than a flat cutoff and excludes backbone atoms as nucleophiles. That is the same test the corpus
used, so a hit here is covalent by the project's own definition and not by a looser one.
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = None            # set in main() from evidence_v4's own D
SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'

# Side-chain nucleophiles that form the covalent bonds this project cares about. Backbone N and O
# are deliberately absent: they are not nucleophiles in this chemistry and including them turns
# every close packing contact into a false covalent call.
NUCLEOPHILE = {
    ('CYS', 'SG'), ('LYS', 'NZ'), ('SER', 'OG'), ('THR', 'OG1'), ('TYR', 'OH'),
    ('HIS', 'NE2'), ('HIS', 'ND1'), ('ASP', 'OD2'), ('GLU', 'OE2'), ('MET', 'SD'),
}
SKIP_LIG = {'HOH', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'CL', 'NA', 'MG', 'ZN',
            'CA', 'MN', 'K', 'IOD', 'BR', 'NO3', 'DMS', 'TRS', 'MES', 'EPE', 'FMT', 'ACY',
            'NAG', 'BMA', 'MAN', 'FUC', 'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NAP',
            'FAD', 'SAH', 'SAM', 'COA', 'HEM', 'MYR', 'PLM', 'OLA', 'STE'}


STD_RES = set()          # filled in main() from evidence_v4.STD


def load_ev4():
    spec = importlib.util.spec_from_file_location('ev4', os.path.join(HERE, 'evidence_v4.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def entries_for(uniprot, limit=400):
    """PDB entry ids for a UniProt accession, from RCSB. Network, once per target."""
    q = {'query': {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_polymer_entity_container_identifiers.'
                         'reference_sequence_identifiers.database_accession',
            'operator': 'exact_match', 'value': uniprot}},
         'return_type': 'entry',
         'request_options': {'paginate': {'start': 0, 'rows': limit}}}
    url = SEARCH + '?json=' + urllib.parse.quote(json.dumps(q))
    with urllib.request.urlopen(url, timeout=60) as fh:
        d = json.load(fh)
    return [r['identifier'] for r in d.get('result_set', [])], d.get('total_count', 0)


def fetch(ids, workers=6):
    """Pull missing CIFs into the mirror. Same source and layout as data/pdb_mirror/fetch_local.sh."""
    import concurrent.futures
    def one(pid):
        dst = os.path.join(MIRROR, '%s.cif.gz' % pid)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        try:
            url = 'https://files.rcsb.org/download/%s.cif.gz' % pid
            with urllib.request.urlopen(url, timeout=90) as r, open(dst + '.part', 'wb') as w:
                w.write(r.read())
            os.replace(dst + '.part', dst)
            return True
        except Exception:
            for junk in (dst + '.part',):
                if os.path.exists(junk):
                    os.remove(junk)
            return False
    if not ids:
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(1 for ok in ex.map(one, ids) if ok)


def read_atoms(path):
    """Heavy atoms of one CIF: (protein side-chain nucleophiles, ligand atoms by comp/chain/seq)."""
    nuc, lig, hdr, idx = [], collections.defaultdict(list), [], None
    with gzip.open(path, 'rt', errors='ignore') as fh:
        for ln in fh:
            if ln.startswith('_atom_site.'):
                hdr.append(ln.strip().split('.', 1)[1]); idx = None; continue
            if hdr and idx is None:
                try:
                    idx = {k: hdr.index(k) for k in
                           ('type_symbol', 'label_atom_id', 'label_comp_id',
                            'auth_seq_id', 'auth_asym_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')}
                except ValueError:
                    return [], {}
            if not ln.startswith(('ATOM', 'HETATM')):
                if nuc or lig:
                    break
                continue
            p = ln.split()
            if idx is None or len(p) <= max(idx.values()):
                continue
            el = p[idx['type_symbol']]
            if el in ('H', 'D'):
                continue
            comp = p[idx['label_comp_id']]
            name = p[idx['label_atom_id']].strip('"')
            try:
                xyz = (float(p[idx['Cartn_x']]), float(p[idx['Cartn_y']]), float(p[idx['Cartn_z']]))
            except ValueError:
                continue
            if (comp, name) in NUCLEOPHILE:
                nuc.append({'res': comp, 'name': name, 'rn': p[idx['auth_seq_id']],
                            'ch': p[idx['auth_asym_id']], 'el': el, 'xyz': xyz})
            elif comp not in SKIP_LIG and comp not in STD_RES and len(comp) >= 2 \
                    and not comp.startswith('UNK'):
                lig[(comp, p[idx['auth_asym_id']], p[idx['auth_seq_id']])].append(
                    {'el': el, 'name': name, 'xyz': xyz})
    return nuc, lig


def covalent_hits(path, ev4, min_heavy=8, accept_probable=False):
    """-> [(ligand_code, residue, resnum, chain, ligand_atom, protein_atom, distance)]"""
    nuc, lig = read_atoms(path)
    if not nuc or not lig:
        return []
    out = []
    for (comp, ch, rn), atoms in lig.items():
        if len(atoms) < min_heavy:
            continue
        best = None
        for a in atoms:
            for p in nuc:
                dx = a['xyz'][0] - p['xyz'][0]
                dy = a['xyz'][1] - p['xyz'][1]
                dz = a['xyz'][2] - p['xyz'][2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > 9.0:                      # 3 A, generous pre-filter before the real test
                    continue
                d = d2 ** 0.5
                # cov_check -> (is_bond, ideal_length, confidence). Testing the TUPLE is
                # always true; only the first element is the verdict, and REVIEWER.md rule 3
                # permits calling a bond covalent only at confidence 'confirmed'.
                ok, ideal, conf = ev4.cov_check(a['el'], p['el'], d, p['name'])
                good = conf == 'confirmed' or (accept_probable and conf == 'probable')
                if ok and good and (best is None or d < best[-1]):
                    best = (comp, p['res'], p['rn'], p['ch'], a['name'], p['name'], round(d, 2))
        if best:
            out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True,
                    help='comma-separated UniProt:label pairs, e.g. P52333:JAK3,P04626:HER2')
    ap.add_argument('--out', default=os.path.join(HERE, 'covalent_programmes.json'))
    ap.add_argument('--max-entries', type=int, default=400)
    ap.add_argument('--fetch', action='store_true',
                    help='download entries RCSB lists that the mirror lacks. The mirror was built '
                         "for the corpus's own target list, so a new programme is only partly "
                         'covered: HER2 had 3 of 63 entries locally, which is unlikely to include '
                         'the complex of interest.')
    ap.add_argument('--probable', action='store_true',
                    help="also report confidence 'probable' -- a stretched adduct. REVIEWER.md "
                         'rule 3 forbids calling these covalent in a label, but a reversible '
                         'covalent warhead (a ketoamide, a nitrile) is often modelled long, so a '
                         'programme that returns nothing under the strict bar may still be real.')
    a = ap.parse_args()

    ev4 = load_ev4()
    global MIRROR, STD_RES
    STD_RES = set(ev4.STD) | set(ev4.METALS) | set(ev4.COFACTOR_HET)
    MIRROR = os.path.join(ev4.D, 'cif')
    assert os.path.isdir(MIRROR), 'no cif mirror at %s' % MIRROR

    found = {}
    for spec in a.targets.split(','):
        uni, _, label = spec.partition(':')
        uni, label = uni.strip(), (label.strip() or uni.strip())
        try:
            ids, total = entries_for(uni, a.max_entries)
        except Exception as e:
            print('%-9s %-12s RCSB lookup FAILED: %s' % (uni, label, e)); continue
        local = [i for i in ids if os.path.exists(os.path.join(MIRROR, '%s.cif.gz' % i))]
        if a.fetch:
            missing = [i for i in ids if i not in set(local)]
            got = fetch(missing)
            local = [i for i in ids if os.path.exists(os.path.join(MIRROR, '%s.cif.gz' % i))]
            if missing:
                print('%-9s %-12s fetched %d/%d missing entries' % (uni, label, got, len(missing)))
        hits = []
        for pid in local:
            for h in covalent_hits(os.path.join(MIRROR, '%s.cif.gz' % pid), ev4,
                                   accept_probable=a.probable):
                hits.append({'pdb': pid, 'ligand': h[0], 'residue': h[1], 'resnum': h[2],
                             'chain': h[3], 'lig_atom': h[4], 'prot_atom': h[5], 'dist': h[6]})
        ligs = collections.Counter(h['ligand'] for h in hits)
        res = collections.Counter('%s%s' % (h['residue'], h['resnum']) for h in hits)
        print('%-9s %-12s RCSB %4d entries, %4d in mirror -> %3d covalent complexes, '
              '%2d distinct ligands  %s'
              % (uni, label, total, len(local), len(hits), len(ligs),
                 ', '.join('%s(%d)' % kv for kv in res.most_common(4))))
        found[uni] = {'label': label, 'rcsb_entries': total, 'in_mirror': len(local),
                      'covalent': sorted(hits, key=lambda h: (h['ligand'], h['pdb']))}
    json.dump(found, open(a.out, 'w'), indent=1)
    print('\nwrote %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
