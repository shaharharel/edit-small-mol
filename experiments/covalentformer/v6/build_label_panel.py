#!/usr/bin/env python3
"""Blind, PAIRED chemist panel: the same pair labelled by g3 and by g4.

2,600 pairs carry a usable label from both generations and the sentences differ on 100% of them,
so this is a true paired comparison -- same input, same contact table, same analogue, two
independent human labels. Side is randomised per pair and the generation is recorded only in a key
file, because a reviewer who knows which pass is "the new one" will find it better.

Unlike the model panels, the real analogue IS shown: these are labels of a known edit, and the
question is whether the sentence describes that edit and attributes its numbers correctly.
"""
import argparse, glob, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
G4 = os.path.expanduser('~/Downloads/covalentformer_labelling_task/label_out')
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'


def usable(d):
    return {p: v for p, v in d.items()
            if v.get('verdict') in ('ADOPT', 'REWRITE') and (v.get('final') or '').strip()}


def fmt_atoms(atoms, limit=36):
    out = []
    for a in atoms[:limit]:
        bits = ['%-5s %-2s' % (a.get('atom', '?'), a.get('element', '?'))]
        if a.get('faces'):
            bits.append('faces %-12s' % a['faces'])
        if a.get('distance_A') is not None:
            bits.append('%.2f A' % a['distance_A'])
        if a.get('interaction'):
            bits.append(str(a['interaction']))
        out.append('    ' + '  '.join(bits))
    if len(atoms) > limit:
        out.append('    ... %d more rows ...' % (len(atoms) - limit))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    g3 = {}
    for f in glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')):
        for p, v in json.load(open(f)).items():
            if v.get('label_generation') == 'g3_v2_input_anchored_envelope':
                g3[p] = v
    g4 = {}
    for f in sorted(glob.glob(os.path.join(G4, 'batch_*.json'))):
        g4.update(json.load(open(f)))
    g3, g4 = usable(g3), usable(g4)
    both = sorted(set(g3) & set(g4))
    rng = random.Random(a.seed)
    chosen = sorted(rng.sample(both, min(a.n, len(both))))

    recs = {}
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in set(chosen):
            recs[r['pair']] = r

    os.makedirs(a.out, exist_ok=True)
    key, blocks = {}, []
    for i, pid in enumerate(chosen, 1):
        rec = recs[pid]
        sides = ['g3', 'g4']
        rng.shuffle(sides)
        key[pid] = {'A': sides[0], 'B': sides[1]}
        src = {'g3': g3, 'g4': g4}
        mi, lo = rec['MODEL_INPUT'], rec.get('LABEL_ONLY') or {}
        ac = []
        for c in (lo.get('analogue_contacts') or [])[:20]:
            ac.append('    %-5s faces %-12s %s' % (
                c.get('atom', '?'), c.get('faces', '?'),
                ('%.2f A' % c['distance_A']) if c.get('distance_A') is not None else ''))
        blocks.append("""
================================================================================
CASE {i} of {n}   pair {pid}   target {tgt}
================================================================================
INPUT LIGAND   {ins}

INPUT CONTACT TABLE -- each heavy atom's NEAREST protein atom, measured in the input crystal
{atoms}

THE ANALOGUE THAT WAS ACTUALLY MADE   {ana}
  heavy atoms added: {hadd}     site of change: {soc}

ANALOGUE'S OWN MEASURED CONTACTS
{acon}

--------------------------------------------------------------------------------
TWO HUMAN-WRITTEN LABELS for this same pair. Both are candidate TRAINING TARGETS.

  [A] {sa}

  [B] {sb}
--------------------------------------------------------------------------------
Judge EACH of A and B:
  describes_real_edit    does it describe the edit that was actually made above?
  atom_attribution_ok    is every distance it cites a real row of the INPUT table AND attached to
                         the atom that owns it? Check number by number. This is the failure that
                         matters most: a real distance on the wrong atom reads perfectly and is
                         false.
  invented_chemistry     any contact, bond or interaction the evidence does not support?
  input_anchored         are the MEASURED facts taken from the input side, with the consequence on
                         the analogue side stated qualitatively and WITHOUT numbers? (A measured
                         analogue distance is a violation: at inference no analogue exists.)
  would_train_on_it      would you accept this as a training target for a model that must learn to
                         explain pocket consequences truthfully?
  one_line_why
Then: which_is_better (A | B | neither) and a one-line reason.
""".format(i=i, n=len(chosen), pid=pid, tgt=rec.get('target', '?'),
           ins=mi.get('input_smiles', '?'), atoms=fmt_atoms(mi.get('atoms') or []),
           ana=(rec.get('ANSWER') or {}).get('analogue_smiles', '?'),
           hadd=lo.get('heavy_atoms_added', '?'),
           soc=', '.join(x['atom'] for x in (lo.get('site_of_change') or [])) or '?',
           acon='\n'.join(ac) or '    (none recorded)',
           sa=src[sides[0]][pid]['final'].strip(), sb=src[sides[1]][pid]['final'].strip()))

    open(os.path.join(a.out, 'panel.txt'), 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    print('wrote %s/panel.txt: %d paired cases; g3_as_A=%d'
          % (a.out, len(key), sum(1 for v in key.values() if v['A'] == 'g3')))


if __name__ == '__main__':
    main()
