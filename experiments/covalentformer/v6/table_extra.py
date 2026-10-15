#!/usr/bin/env python3
"""The columns big_table.py does not cover: scaffold retention, attribution tiers, self-fidelity.

  scaffold_kept   Bemis-Murcko scaffold of the generated molecule equals the input's. Separates
                  "analogue" from "different molecule"; the reference analogues keep it 1.000 by
                  construction of a matched pair, so anything below that is scaffold drift.
  attrib_1a       every number in the sentence appears somewhere in the INPUT contact table
  attrib_1b       every (residue, distance) the sentence asserts CO-OCCURS in one input table row.
                  This is the "real distance, wrong residue" class. Input-side only, so it is valid
                  whichever analogue the model proposed. The reference labels score 0.71 under this
                  heuristic (nearest-number-after-residue mis-pairs across clauses), so 0.71 is the
                  practical ceiling, not 1.0.
  elem_self       element-count fidelity of the sentence against the MODEL'S OWN molecule
  dir_self        direction fidelity against the model's own molecule
                  Both reuse eval_axes' scorer with analogue_smiles swapped for the generation, so
                  a model is not charged for proposing a different valid analogue.
"""
import json
import os
import re
import subprocess
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'sft_out_g2', 'sft_pocket.jsonl')
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
SCRATCH = ('/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/'
           'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad')

NUM = re.compile(r'\d+\.\d+')
AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
      'PHE','PRO','SER','THR','TRP','TYR','VAL'}
RES = re.compile(r'\b([A-Z]{3})(\d+)(?:\.([A-Z0-9]+))?')


def scaffold(smi):
    m = Chem.MolFromSmiles(smi or '')
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:                                                # noqa: BLE001
        return None


def load_bench():
    out = {}
    for line in open(BENCH):
        r = json.loads(line)
        if r['split'] == 'val':
            out[r['pair']] = r
    return out


def load_tables(pairs):
    out = {}
    want = set(pairs)
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] not in want:
            continue
        tbl = []
        for a in r['MODEL_INPUT'].get('atoms') or []:
            for dk, fk in (('distance_A', 'faces'), ('polar_distance_A', 'polar_partner')):
                if a.get(dk) is not None and a.get(fk):
                    tbl.append((a['atom'], a[fk], round(a[dk], 2)))
        out[r['pair']] = tbl
    return out


def attrib(sent, tbl):
    nums = {round(float(x), 2) for x in NUM.findall(sent or '')}
    tbl_nums = {d for _, _, d in tbl}
    a_ok = all(n in tbl_nums for n in nums) if nums else None
    pairs = []
    for m in RES.finditer((sent or '').upper()):
        if m.group(1) not in AA:
            continue
        res = '%s%s' % (m.group(1), m.group(2))
        nm = NUM.search((sent or '')[m.end():m.end() + 80])
        if nm:
            pairs.append((res, round(float(nm.group(0)), 2)))
    ok = sum(1 for res, d in pairs
             if any(f.split('.')[0] == res and abs(dd - d) < 0.005 for _, f, dd in tbl))
    return a_ok, ok, len(pairs)


def self_fidelity(gen_path):
    """elem/dir fidelity against the model's own molecule, via the existing eval_axes path."""
    tmp = os.path.join(SCRATCH, 'self_%s.jsonl' % os.path.basename(os.path.dirname(gen_path)))
    sub = subprocess.run([sys.executable, os.path.join(SCRATCH, 'selfconsist.py'),
                          BENCH, gen_path, tmp], capture_output=True, text=True)
    if sub.returncode != 0:
        return None, None
    out = subprocess.run([sys.executable, os.path.join(HERE, 'eval_axes.py'), gen_path,
                          '--data', tmp], capture_output=True, text=True)
    el = dr = None
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] == 'edit_element_fidelity':
            el = float(p[1])
        if len(p) >= 2 and p[0] == 'edit_direction_fidelity':
            dr = float(p[1])
    return el, dr


def main():
    bench = load_bench()
    tables = load_tables(bench)
    arms = sys.argv[1:]
    print('%-22s %-9s %-9s %-9s %-10s %-9s' %
          ('arm', 'scaffold', 'attrib1a', 'attrib1b', 'elem_self', 'dir_self'))
    print('-' * 74)
    # the reference row first, as the ceiling
    rows = [('REFERENCE', None)] + [(os.path.basename(os.path.dirname(a)), a) for a in arms]
    for name, path in rows:
        if path is None:
            src = {p: (r['reference_sentence'], r['analogue_smiles']) for p, r in bench.items()}
        else:
            src = {}
            for line in open(path):
                r = json.loads(line)
                src[r['pair']] = (r.get('gen_sentence') or '',
                                  r.get('gen_smiles_canonical') or r.get('gen_smiles'))
        sc_ok = sc_n = 0
        a_ok = a_n = 0
        b_ok = b_n = 0
        for pid, (sent, smi) in src.items():
            b = bench.get(pid)
            if b is None:
                continue
            si, sg = scaffold(b['input_smiles']), scaffold(smi)
            if si is not None and sg is not None:
                sc_n += 1
                sc_ok += (si == sg)
            tbl = tables.get(pid) or []
            ao, bo, bn = attrib(sent, tbl)
            if ao is not None:
                a_n += 1
                a_ok += ao
            b_ok += bo
            b_n += bn
        el = dr = None
        if path:
            el, dr = self_fidelity(path)
        f = lambda x, y: ('%.3f' % (x / y)) if y else '-'           # noqa: E731
        print('%-22s %-9s %-9s %-9s %-10s %-9s' %
              (name, f(sc_ok, sc_n), f(a_ok, a_n), f(b_ok, b_n),
               '-' if el is None else '%.3f' % el, '-' if dr is None else '%.3f' % dr))


if __name__ == '__main__':
    main()
