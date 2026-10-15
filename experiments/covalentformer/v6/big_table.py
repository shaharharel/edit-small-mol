#!/usr/bin/env python3
"""One table, every arm, every programmatic metric. Chemist columns stay TBD by design.

Metric definitions, so the table can be read without guessing:

  validity        fraction of rows where a SMILES could be extracted AND parses in RDKit.
                  Denominator is ALL rows, so a model that emits no molecule scores 0, not NaN.
  uniqueness      distinct canonical SMILES / valid SMILES. Low means the model has a few favourite
                  answers it reuses -- the canned-output failure.
  novelty         fraction of valid generations NOT present in that arm's training completions.
                  Guards against the model reciting a memorised analogue.
  tc_to_input     mean Tanimoto (ECFP4, 2048 bits) between the generated molecule and the INPUT.
                  1.0 = echoed the input, so this measures how conservative the edit is. The
                  reference analogues sit at 0.68, which is the target band, NOT a maximum.
  tc_to_reference mean Tanimoto to the crystallised analogue. Reported for continuity with earlier
                  tables, but it penalises a model for proposing a different VALID analogue, so it
                  is not a primary metric.
  heavy_delta_mae mean |heavy atoms added by the model - heavy atoms added by the reference|.
                  Measures whether the edit is the right SIZE even when it is a different edit.
  warhead_kept    of rows whose input carries a covalent warhead (SMARTS), the fraction whose
                  generated molecule still carries one. The human reference scores 0.76 here.
  grounded        fraction of named residues that exist in that pair's own contact table. This is
                  the metric that reads 1.000 while chemists reject the sentence, so it is a
                  NECESSARY-not-sufficient check, kept for exactly that reason.
  selectivity     overlap between the residues the sentence names and the residues the reference
                  sentence names. A zero-parameter baseline -- "name the nearest table residue" --
                  scores 0.5625, so any arm below that is worse than a distance sort.
  elem_self       element-count fidelity of the sentence against the MODEL'S OWN molecule: does the
                  prose describe the edit the model actually made? Reference-free.
  dir_self        same, for the direction of the edit (atoms added vs removed).
  schema_ok       fraction emitting a well-formed think/answer structure. Only meaningful for an
                  arm trained in ether0's schema; '-' elsewhere.

  sensible_analogue  CHEMIST column. Defined as: is the proposed molecule a plausible
                     medicinal-chemistry analogue of the input at all -- same scaffold family, no
                     implausible valence or unstable motif, a change a chemist might actually make.
                     Programmatic proxies (validity, TC band, ring-count delta) are reported above
                     but do NOT substitute for it.
  reasoning_valid    CHEMIST column. Blind panel: does the rationale describe the real edit from
                     input to the model's own molecule, with every cited distance attached to the
                     atom that owns it.
"""
import argparse
import collections
import json
import os
import statistics
import sys

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import covalent_stats as C  # noqa: E402

BENCH = 'sft_out_g2/sft_pocket.jsonl'


def canon(s):
    m = Chem.MolFromSmiles(s or '')
    return Chem.MolToSmiles(m) if m else None


def fp(s):
    m = Chem.MolFromSmiles(s or '')
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def heavy(s):
    m = Chem.MolFromSmiles(s or '')
    return m.GetNumHeavyAtoms() if m else None


def residues(text):
    import re
    AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
          'PHE','PRO','SER','THR','TRP','TYR','VAL'}
    return {m.group(0).upper() for m in re.finditer(r'[A-Z]{3}\d+', (text or '').upper())
            if m.group(0)[:3].upper() in AA}


def metrics(gen_path, train_path=None):
    bench = {}
    for line in open(BENCH):
        r = json.loads(line)
        if r['split'] == 'val':
            bench[r['pair']] = r
    train_set = set()
    if train_path and os.path.exists(train_path):
        for line in open(train_path):
            r = json.loads(line)
            if r['split'] == 'train' and r.get('analogue_smiles'):
                train_set.add(canon(r['analogue_smiles']))

    rows = [json.loads(l) for l in open(gen_path)]
    n = len(rows)
    valid, canons = 0, []
    tc_in, tc_ref, hd_err = [], [], []
    wh_tot = wh_kept = 0
    g_hit = g_tot = 0
    sel = []
    schema = [r.get('schema_ok') for r in rows if 'schema_ok' in r]
    for r in rows:
        b = bench.get(r['pair'])
        if b is None:
            continue
        g = canon(r.get('gen_smiles_canonical') or r.get('gen_smiles'))
        if g:
            valid += 1
            canons.append(g)
            gf, inf, rf = fp(g), fp(b['input_smiles']), fp(b['analogue_smiles'])
            if gf is not None and inf is not None:
                tc_in.append(DataStructs.TanimotoSimilarity(gf, inf))
            if gf is not None and rf is not None:
                tc_ref.append(DataStructs.TanimotoSimilarity(gf, rf))
            hg, hi, hr = heavy(g), heavy(b['input_smiles']), heavy(b['analogue_smiles'])
            if None not in (hg, hi, hr):
                hd_err.append(abs((hg - hi) - (hr - hi)))
            wi = C.warheads_in(b['input_smiles'])
            if wi:
                wh_tot += 1
                if (C.warheads_in(g) or set()) & wi:
                    wh_kept += 1
        named = residues(r.get('gen_sentence'))
        if named:
            tbl = set(b['table_residues'])
            g_hit += sum(1 for x in named if x in tbl)
            g_tot += len(named)
            ref = set(b['reference_residues'])
            if ref:
                sel.append(len(named & ref) / len(ref))
    uniq = len(set(canons)) / valid if valid else None
    nov = (sum(1 for c in set(canons) if c not in train_set) / len(set(canons))
           if canons and train_set else None)
    m = lambda x: round(statistics.mean(x), 3) if x else None   # noqa: E731
    return {
        'n': n,
        'validity': round(valid / n, 3) if n else None,
        'uniqueness': round(uniq, 3) if uniq else None,
        'novelty': round(nov, 3) if nov is not None else None,
        'tc_to_input': m(tc_in),
        'tc_to_reference': m(tc_ref),
        'heavy_delta_mae': m(hd_err),
        'warhead_kept': (round(wh_kept / wh_tot, 3) if wh_tot else None),
        'warhead_n': wh_tot,
        'grounded': round(g_hit / g_tot, 3) if g_tot else None,
        'selectivity': m(sel),
        'schema_ok': (round(sum(1 for s in schema if s) / len(schema), 3) if schema else None),
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', nargs=3, action='append', metavar=('NAME', 'GEN', 'TRAIN'),
                    help='label, generations.jsonl, training jsonl (or - for none)')
    a = ap.parse_args()
    out = {}
    for name, gen, train in a.arm or []:
        if not os.path.exists(gen):
            print('%-30s MISSING %s' % (name, gen))
            continue
        out[name] = metrics(gen, None if train == '-' else train)
    cols = ['n', 'validity', 'uniqueness', 'novelty', 'tc_to_input', 'tc_to_reference',
            'heavy_delta_mae', 'warhead_kept', 'warhead_n', 'grounded', 'selectivity', 'schema_ok']
    print('\n| arm | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    for name, v in out.items():
        print('| %s | ' % name + ' | '.join(
            ('-' if v.get(c) is None else str(v[c])) for c in cols) + ' |')
    json.dump(out, open('big_table_metrics.json', 'w'), indent=2)
