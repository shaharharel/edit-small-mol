#!/usr/bin/env python3
"""Build a BLIND A/B packet so a medicinal chemist can judge two models' rationales.

Why: every comparison we have between SmileyLlama-8B and ChemDFM-14B rests on automated axes, and
we have already proved those axes blind to the error class that actually matters. `6BRY GK9->E3Y`
cited a REAL input-side distance attached to the WRONG atom and scored groundedness 1.000 while
passing r1, r2 and r3. So "ChemDFM is slightly ahead" is, as it stands, unvalidated chemistry.

Design decisions that keep this honest:

  * BLIND. Which model wrote A and which wrote B is drawn per pair from a fixed seed and kept in a
    separate key file the reviewer never reads. Otherwise the reviewer's prior about an 8B vs a 14B
    does the judging.
  * SIDE-RANDOMISED PER PAIR, not per packet, so a reviewer who drifts toward "A" cannot favour
    one model.
  * The REFERENCE sentence is withheld. A reviewer shown the human label grades similarity to it
    instead of chemical truth, and the reference is itself only one of several defensible edits.
  * The EVIDENCE IS INCLUDED -- the input contact table and the analogue's own contacts -- because
    the question is not "does this read well" but "is this what the edit did", and that cannot be
    answered from the sentences alone.

Usage:
    python build_chemist_panel.py --n 40 --out panel_chemdfm_vs_smiley
"""
import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'

MODELS = {
    'smiley': os.path.join(HERE, 'sft_2k', 'smiley_pocket_g2', 'generations.jsonl'),
    'chemdfm': os.path.join(HERE, 'sft_2k', 'chemdfm_pocket_g2', 'generations.jsonl'),
}


def load_gens(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r['pair']] = r
    return out


def fmt_atoms(atoms, limit=40):
    lines = []
    for a in atoms[:limit]:
        bits = ['%-5s %-2s' % (a.get('atom', '?'), a.get('element', '?'))]
        if a.get('faces'):
            bits.append('faces %-12s' % a['faces'])
        if a.get('distance_A') is not None:
            bits.append('%.2f A' % a['distance_A'])
        if a.get('interaction'):
            bits.append(str(a['interaction']))
        lines.append('    ' + '  '.join(bits))
    if len(atoms) > limit:
        lines.append('    ... %d more atom rows' % (len(atoms) - limit))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    gens = {k: load_gens(v) for k, v in MODELS.items()}
    pairs = sorted(set(gens['smiley']) & set(gens['chemdfm']))
    rng = random.Random(a.seed)
    chosen = sorted(rng.sample(pairs, min(a.n, len(pairs))))

    recs = {}
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in set(chosen):
            recs[r['pair']] = r

    os.makedirs(a.out, exist_ok=True)
    key = {}
    blocks = []
    for i, pid in enumerate(chosen, 1):
        rec = recs.get(pid)
        if rec is None:
            continue
        sides = ['smiley', 'chemdfm']
        rng.shuffle(sides)
        key[pid] = {'A': sides[0], 'B': sides[1]}
        mi, lo = rec['MODEL_INPUT'], rec.get('LABEL_ONLY') or {}
        sa = (gens[sides[0]][pid].get('gen_sentence') or '').strip()
        sb = (gens[sides[1]][pid].get('gen_sentence') or '').strip()
        ana = rec.get('ANSWER', {}).get('analogue_smiles', '')

        ac_lines = []
        for c in (lo.get('analogue_contacts') or [])[:25]:
            ac_lines.append('    %-5s faces %-12s %s' % (
                c.get('atom', '?'), c.get('faces', '?'),
                ('%.2f A' % c['distance_A']) if c.get('distance_A') is not None else ''))

        blocks.append("""
================================================================================
PAIR {i} of {n}   id {pid}   target {tgt}
================================================================================
INPUT LIGAND (what the chemist started from)
  SMILES: {ins}

INPUT CONTACT TABLE -- each atom's NEAREST protein atom, measured in the input crystal
{atoms}

WHAT WAS ACTUALLY MADE (the analogue, from its own crystal)
  SMILES: {ana}
  heavy atoms added: {hadd}
  site of change: {soc}

ANALOGUE'S OWN CONTACTS (ground truth for what the edit did)
{acon}

--------------------------------------------------------------------------------
TWO CANDIDATE RATIONALES. One sentence each. Judge them, do not rank politely.

  [A] {sa}

  [B] {sb}
--------------------------------------------------------------------------------
For EACH of A and B answer, in the JSON described in your brief:
  describes_real_edit   does the sentence describe the edit that was ACTUALLY made above?
  atom_attribution_ok   is every distance/contact attached to the atom that OWNS it? This is the
                        failure we care about most: a sentence can quote a real input distance and
                        hang it on the wrong atom, which reads perfectly and is false.
  invented_chemistry    does it assert a contact, bond or interaction the evidence does not support?
  would_show_a_chemist  would you put this in front of a medicinal chemistry audience?
  one_line_why
Then: which_is_better  (A | B | neither) and why.
""".format(i=i, n=len(chosen), pid=pid, tgt=rec.get('target', '?'),
           ins=mi.get('input_smiles', '?'), atoms=fmt_atoms(mi.get('atoms') or []),
           ana=ana, hadd=lo.get('heavy_atoms_added', '?'),
           soc=', '.join(x['atom'] for x in (lo.get('site_of_change') or [])) or '?',
           acon='\n'.join(ac_lines) or '    (none recorded)',
           sa=sa or '(empty)', sb=sb or '(empty)'))

    packet = os.path.join(a.out, 'panel.txt')
    open(packet, 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    counts = {'smiley_as_A': sum(1 for v in key.values() if v['A'] == 'smiley'),
              'chemdfm_as_A': sum(1 for v in key.values() if v['A'] == 'chemdfm')}
    print('wrote %s: %d pairs' % (packet, len(key)))
    print('side balance: %s' % counts)


if __name__ == '__main__':
    main()
