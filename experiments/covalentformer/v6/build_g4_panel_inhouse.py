#!/usr/bin/env python3
"""Blind FOUR-ARM self-consistency panel -- the sensible/reasoning columns for every
untested arm, scored on one scale by one set of reviewers.

The first panel showed the chemists the input, the contact table, and THE ANALOGUE THAT WAS
ACTUALLY CRYSTALLISED, then asked whether the sentence described that edit. So it measured
sentence-vs-REFERENCE. That is unfair as the primary read: a molecule has many valid analogues,
and 37% of benchmark rows share a prompt with a different "correct" answer, so a model that
proposes a different reasonable analogue was scored wrong.

This panel replaces the reference with THE MOLECULE THE MODEL ITSELF EMITTED. The question becomes
self-consistency: does the rationale describe the edit from the input to the model's own output?
That has a single defensible answer, needs no reference, and is the property the model must have
for its reasoning to mean anything.

Only rows where the model emitted a parseable, different molecule are eligible -- self-consistency
is undefined with no molecule, and trivial if the model echoed the input.
"""
import argparse, json, os, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
SCR = ('/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/'
       'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad/alltab/sft_runs')
# The five arms that have NEVER had a chemist panel. All in ONE panel, not five pairwise ones:
# the ChemDFM g2-vs-g3 panel put would_show_a_chemist at 12.5% for an arm the earlier panel had
# at 25.6%, so panel-to-panel drift on these columns is larger than most differences between
# arms. One panel, one set of reviewers, one scale -- then the arms are comparable to each other.
# The two g4 arms -- same models and schema as their g3 counterparts, trained on the independent
# second labelling pass. attrib-1b roughly doubled on both (ChemDFM 0.290 -> 0.594, ether0 schema B
# 0.330 -> 0.693 against a 0.709 ceiling), so the question for the chemists is whether that
# automated gain is visible in the rationales they would actually put in front of an audience.
G4 = '/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad/g4res'
MODELS = {'chemdfm_g4':     G4 + '/chemdfm/generations.jsonl',
          'ether0_schemaB_g4': G4 + '/ether0/generations.jsonl'}
SLOTS = 'AB' 


def canon(s):
    m = Chem.MolFromSmiles(s or '')
    return Chem.MolToSmiles(m) if m else None


def fmt_atoms(atoms, limit=40):
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
        out.append('    ... %d more atom rows' % (len(atoms) - limit))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    gens = {}
    for k, p in MODELS.items():
        path = p if os.path.isabs(p) else os.path.join(HERE, p)
        gens[k] = {json.loads(l)['pair']: json.loads(l) for l in open(path)}

    eligible = []
    common = set.intersection(*[set(g) for g in gens.values()])
    for pid in sorted(common):
        ok = True
        for k in MODELS:
            r = gens[k][pid]
            g = canon(r.get('gen_smiles_canonical') or r.get('gen_smiles'))
            if not g or not (r.get('gen_sentence') or '').strip():
                ok = False
        if ok:
            eligible.append(pid)
    # Draw ONLY from the cases the external g4 panel scores, so the in-house and external rates
    # describe the same molecules and any difference is scorer calibration, not sampling.
    import glob as _g, re as _re, os as _os
    ext = set()
    for _f in _g.glob(_os.path.expanduser(
            '~/Downloads/covalentformer_labelling_task/metrics_eval_g4/batches/*.txt')):
        ext |= set(_re.findall(r'id (\S+ \S+->\S+)', open(_f).read()))
    if ext:
        before = len(eligible)
        eligible = [p for p in eligible if p in ext]
        print('eligible restricted to the external g4 panel: %d of %d' % (len(eligible), before))
    rng = random.Random(a.seed)
    chosen = sorted(rng.sample(eligible, min(a.n, len(eligible))))
    print('eligible pairs (both models emitted a molecule and a sentence): %d; sampled %d'
          % (len(eligible), len(chosen)))

    recs = {}
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in set(chosen):
            recs[r['pair']] = r

    os.makedirs(a.out, exist_ok=True)
    key, blocks = {}, []
    for i, pid in enumerate(chosen, 1):
        rec = recs[pid]
        sides = list(MODELS)
        rng.shuffle(sides)
        key[pid] = {SLOTS[j]: sides[j] for j in range(len(sides))}
        mi = rec['MODEL_INPUT']
        cards = []
        for slot, mname in zip(SLOTS, sides):
            r = gens[mname][pid]
            cards.append("""  [{slot}] PROPOSED MOLECULE: {smi}
      RATIONALE: {sent}""".format(slot=slot,
                                  smi=canon(r.get('gen_smiles_canonical') or r.get('gen_smiles')),
                                  sent=(r.get('gen_sentence') or '').strip()))
        blocks.append("""
================================================================================
CASE {i} of {n}   id {pid}   target {tgt}
================================================================================
INPUT LIGAND
  SMILES: {ins}

INPUT CONTACT TABLE -- each atom's NEAREST protein atom, measured in the input crystal
{atoms}

--------------------------------------------------------------------------------
TWO INDEPENDENT PROPOSALS. Each is one model's OWN proposed molecule together with
its OWN rationale. There is no reference answer and you are NOT being asked whether
the proposed molecule is the one a crystallographer made -- many analogues are valid.

{cards}
--------------------------------------------------------------------------------
For EACH of A and B, judge ONLY self-consistency and groundedness:

  edit_identified_correctly  Compare the INPUT SMILES with that proposal's OWN molecule. Does the
                             rationale describe the change that actually separates those two
                             structures? If the rationale talks about removing a group that is
                             present in both, or ignores the group that actually changed, this is
                             false.
  atom_attribution_ok        Every distance or contact the rationale cites: is it a real row of
                             the input contact table, AND attached to the atom that owns it?
  invented_chemistry         Does it assert a contact, bond or interaction the table does not
                             support?
  chemically_sensible_edit   Is the proposed molecule a reasonable medicinal-chemistry analogue of
                             the input at all (not a different scaffold, not nonsense)?
  would_show_a_chemist       Would you put this molecule-plus-rationale in front of a medicinal
                             chemistry audience?
  one_line_why

Then: best_overall (A | B | none) and why.

Return ONE json object keyed by the case id, each value holding a key per slot A-B with the five
booleans plus one_line_why, and best_overall. Score every slot on every case: a slot you skip is
indistinguishable from a slot you judged false, and the rates are what the table reports.
""".format(i=i, n=len(chosen), pid=pid, tgt=rec.get('target', '?'),
           ins=mi.get('input_smiles', '?'), atoms=fmt_atoms(mi.get('atoms') or []),
           cards='\n\n'.join(cards)))

    open(os.path.join(a.out, 'panel.txt'), 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    print('wrote %s/panel.txt: %d cases; side balance smiley_as_A=%d'
          % (a.out, len(key), sum(1 for v in key.values() if v['A'] == 'smiley')))


if __name__ == '__main__':
    main()
