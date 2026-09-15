#!/usr/bin/env python
"""Turn labelled MOLECULES into steering INSTRUCTIONS on PAIRS.

This is the step that makes the corpus trainable. A molecule-level value is not an
instruction; the DELTA across a pair is:

    acyl_N_motif      aryl_NH -> ring_N_5        (a categorical MOVE)
    linker_atom_count 3 -> 2                     (DOWN)
    warhead_planarity 41.2 deg -> 8.7 deg        (DOWN, and by how much)

The model is trained to read the instruction and produce B from A. That is the whole
mini covalent-medchem instruction set.

THREE RULES, each from something that went wrong earlier:

1. PRECONDITION: warhead class AND conjugation status must be IDENTICAL between A and B for
   any linker/placement instruction. In a sampled DOWN set, 16/20 pairs had flipped
   conjugation -- that is a warhead SWAP wearing a linker label. Rows that fail this are
   emitted with instructable=0 per param, never silently dropped and never silently kept.

2. SAME means SAME. A pair where the param does not move is a legitimate NO-OP instruction and
   is kept as its own level, because "do not change the linker" is a real thing to ask. It is
   NOT pooled with UP/DOWN and it is NOT discarded.

3. CONTINUOUS PARAMS GET A DEADBAND. A 0.3 deg dihedral difference is conformer noise, not an
   instruction. Below the deadband the label is SAME, not a tiny UP. The deadband is stamped.

Emits BOTH encodings per param so the training code can choose without a rebuild:
    <param>_dir    categorical  UP / DOWN / SAME   (or the explicit A->B move)
    <param>_delta  float        b - a             (continuous, scale-only -- never mean-shifted)
"""
from __future__ import annotations
import os, sys, json, argparse, csv, collections

INSTR_VERSION = 'instr-v1-2026-09-15'

# Deadbands for continuous params, in the param's own units. Below this, the pair is SAME.
DEADBAND = {
    'warhead_planarity': 5.0,     # degrees. ETKDG conformer noise on a locked enone is ~1-2 deg;
                                  # 5 deg is comfortably outside it and still far below the
                                  # 61.8 -> 2.62 effect the param is being trained to steer.
}


def load_mol_params(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r['smiles']] = r
    return out


def load_planarity(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get('acryl_match') in ('1', 'True', 'true') and \
               r.get('embed_ok') in ('1', 'True', 'true') and r.get('planar_dev_deg'):
                try:
                    out[r['smiles']] = float(r['planar_dev_deg'])
                except ValueError:
                    pass
    return out


def cat_instruction(va, vb):
    """Categorical param -> (dir, move). Blank on either side = not instructable."""
    if not va or not vb:
        return None, None
    if va == vb:
        return 'SAME', '%s->%s' % (va, va)
    return 'CHANGED', '%s->%s' % (va, vb)


def num_instruction(va, vb, deadband):
    """Numeric param -> (dir, delta). Deadband collapses noise to SAME."""
    if va is None or vb is None:
        return None, None
    d = vb - va
    if abs(d) < deadband:
        return 'SAME', d
    return ('UP' if d > 0 else 'DOWN'), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/covalent_final_v2/pairs.jsonl')
    ap.add_argument('--molparams', default='data/labels/molecule_params_v2.csv')
    ap.add_argument('--planarity', default='data/labels/planarity_v2.csv')
    ap.add_argument('--out', default='data/instructions/pairs_v2.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    mp = load_mol_params(a.molparams)
    pl = load_planarity(a.planarity)
    print('molecule params %d   planarity usable %d' % (len(mp), len(pl))); sys.stdout.flush()

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    n = 0
    stats = collections.defaultdict(collections.Counter)
    precond_fail = 0
    with open(a.pairs) as fh, open(a.out, 'w') as fo:
        for line in fh:
            if a.limit and n >= a.limit:
                break
            r = json.loads(line)
            ra, rb = mp.get(r['a']), mp.get(r['b'])
            if ra is None or rb is None:
                continue

            # ---- RULE 1: the precondition, evaluated ONCE and recorded on the row.
            same_warhead = (ra['warhead_class'] == rb['warhead_class'])
            row = {'a': r['a'], 'b': r['b'], 'tc': r.get('tc'), 'dmw': r.get('dmw'),
                   'pair_source': r.get('pair_source'),
                   'warhead_a': ra['warhead_class'], 'warhead_b': rb['warhead_class'],
                   'same_warhead': int(same_warhead),
                   'instr_version': INSTR_VERSION}
            if not same_warhead:
                precond_fail += 1

            # ---- acyl_N_motif : categorical
            d, mv = cat_instruction(ra['acyl_N_motif'], rb['acyl_N_motif'])
            row['acyl_N_motif_dir'] = d
            row['acyl_N_motif_move'] = mv
            row['acyl_N_motif_instructable'] = int(d is not None and same_warhead)
            if d and same_warhead:
                stats['acyl_N_motif'][d] += 1

            # ---- linker_atom_count : ordinal
            try:
                la = int(ra['linker_atom_count']) if ra['linker_atom_count'] != '' else None
                lb = int(rb['linker_atom_count']) if rb['linker_atom_count'] != '' else None
            except ValueError:
                la = lb = None
            d, dl = num_instruction(la, lb, 1)      # integer param: deadband 1 => any change counts
            row['linker_atom_count_dir'] = d
            row['linker_atom_count_delta'] = dl
            row['linker_atom_count_instructable'] = int(d is not None and same_warhead)
            if d and same_warhead:
                stats['linker_atom_count'][d] += 1

            # ---- warhead_planarity : continuous, deadbanded
            d, dl = num_instruction(pl.get(r['a']), pl.get(r['b']),
                                    DEADBAND['warhead_planarity'])
            row['warhead_planarity_dir'] = d
            row['warhead_planarity_delta'] = round(dl, 4) if dl is not None else None
            row['warhead_planarity_instructable'] = int(d is not None and same_warhead)
            if d and same_warhead:
                stats['warhead_planarity'][d] += 1

            fo.write(json.dumps(row) + '\n')
            n += 1

    print('\n=== INSTRUCTION SET %s ===' % INSTR_VERSION)
    print('  pair rows written        %9d' % n)
    print('  precondition FAILED (warhead class differs A vs B)  %9d  %5.1f%%'
          % (precond_fail, 100.0 * precond_fail / max(n, 1)))
    print('  deadbands: %s' % DEADBAND)
    print('\n  -- INSTRUCTABLE rows per param (precondition passed AND both sides labelled) --')
    for p in ('acyl_N_motif', 'linker_atom_count', 'warhead_planarity'):
        c = stats[p]; tot = sum(c.values())
        moved = tot - c.get('SAME', 0)
        print('    %-20s %9d rows   MOVED %8d (%5.1f%%)   %s'
              % (p, tot, moved, 100.0 * moved / max(tot, 1), dict(c)))
    print('\n  wrote %s' % a.out)


if __name__ == '__main__':
    main()
