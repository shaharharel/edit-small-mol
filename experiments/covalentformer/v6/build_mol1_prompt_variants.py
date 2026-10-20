#!/usr/bin/env python3
"""Build the mol1 prompt variants that tilt the context for a PREDICTED pose.

WHY mol1 NEEDS A DIFFERENT PROMPT FROM A CRYSTAL PAIR
-----------------------------------------------------
Every training row anchors on geometry measured from a deposited crystal. mol1 has no crystal: its
contact table comes from a Boltz-2 covalent cofold against ZAP70 Cys346 (ligand ipTM 0.976,
confidence 0.844). A cofold carries 1-2 A coordinate error, and the error is not uniform -- it
falls hardest on exactly the number the prompt was most confident about:

    the cofold models the C-S covalent bond at 1.40 A, against a ~1.81 A ideal for C-S.

So the original prompt both FLAGGED that length as unreliable and PRINTED IT FOUR TIMES -- once in
the dist column and three times in the interaction text. The brief forbids asserting it (rule 9a:
never build a claim on a contact the typing flags uncertain). The model repeated it anyway, and
three external chemists failed 142 of 181 round-1 candidates on exactly that assertion. A model
shown a number four times and told not to use it will use it; that is a prompt defect, not a model
defect.

WHAT THE `no_length` VARIANT CHANGES
------------------------------------
One row. The dist cell for the covalent contact becomes `-` and the interaction text asserts the
BOND while stating the length is not measurable from a cofold and giving no figure. Nothing else in
the prompt moves, so the comparison is clean.

MEASURED EFFECT, on 251 samples per variant at temperature 0.9:

                                    as_is      no_length
    cleared the automated funnel     23/213     60/202        11%  ->  30%
    caveat failures                    169         80
    chemists' would_make_it            3/23      13/60        13%  ->  22%

Tripling the usable yield and nearly doubling chemist endorsement by deleting one number. The 80
residual caveat failures under `no_length` are the two kinds a prompt change cannot reach -- building
on the 2.74 A C2 row the table itself flags uncertain, and calling an angle-less polar contact a
hydrogen bond.

WHAT THIS IS NOT
----------------
This is an INFERENCE-time intervention: a different prompt for a case whose geometry is predicted.
It does not retrain anything and it does not change the reward. The training-side counterpart --
perturbing distances on a fraction of training rows, flagging the table as predicted, and dropping
the number from the target sentence, so the model learns "cite the contact, not the measurement"
as a conditional -- is a separate and unbuilt piece of work. Noise alone would not achieve it: the
reward validates cited pairs against whatever table it is given, so a perturbed table teaches the
model to copy perturbed numbers and nothing about hedging.

FURTHER VARIANTS AVAILABLE AND NOT YET USED
-------------------------------------------
Boltz produced THREE poses for mol1 (mol1_model_0/1/2, confidence 0.844/0.837/0.837, ligand ipTM
0.976/0.977/0.978) and only model_0 was ever used. The other two are genuinely different predicted
poses, not synthetic noise, and under 1-2 A error which protein atom is NEAREST a given ligand atom
can flip -- so a second pose redistributes contact OWNERSHIP, which is the failure mode that
dominates mol1 (29% ownership-correct against 70% on crystals). Only model_0 has the split
prot.pdb/lig.sdf that build_mol1_prompt.py reads; models 1 and 2 would need splitting out of their
.cif first. --poses records that intent and refuses rather than silently using model_0.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'mol1_chemdfm', 'mol1_prompt_covalent.jsonl')
POSE_DIR = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'data', 'boltz_poses', 'mol1__zap70_cys346', 'mol1'))

# The replacement row. The BOND is still asserted -- it comes from the pose setup, not from this
# distance -- while no figure is offered for a length a cofold cannot measure.
NO_LENGTH_ROW = ('C1    C  CYS346.SG       A         -     - | COVALENT BOND to CYS346.SG '
                 '[asserted by the pose provenance (cys346_boltz_v2_2026-05-13); this is a cofold, '
                 'not a crystal, so the BOND is asserted while its LENGTH is not measurable here '
                 'and no distance is given for it]')


def variants(prompt):
    """-> {name: prompt}. `as_is` is the original, byte for byte."""
    rows = [l for l in prompt.split('\n') if 'COVALENT BOND to CYS346.SG' in l]
    assert len(rows) == 1, 'expected exactly one covalent row, found %d' % len(rows)
    out = {'as_is': prompt, 'no_length': prompt.replace(rows[0], NO_LENGTH_ROW)}
    body = out['no_length'].split('### Response')[0]
    assert '1.40' not in body, (
        'the unreliable length survives in the no_length body: %s'
        % [l for l in body.split('\n') if '1.40' in l])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'mol1_prompt_variants.jsonl'))
    ap.add_argument('--poses', type=int, default=1,
                    help='how many Boltz poses to emit variants for. Only 1 is supported: '
                         'models 1 and 2 exist (and are of equal confidence) but ship only as '
                         '.cif, with no split prot.pdb/lig.sdf for build_mol1_prompt.py to read.')
    a = ap.parse_args()

    if a.poses != 1:
        have = sorted(f for f in os.listdir(POSE_DIR) if f.endswith('.prot.pdb')) \
            if os.path.isdir(POSE_DIR) else []
        raise SystemExit(
            '--poses %d requested but only %d pose(s) have the split protein/ligand files this '
            'pipeline reads (%s). Split mol1_model_1.cif and mol1_model_2.cif first; they are '
            'real alternative poses of equal confidence and are the right way to vary contact '
            'OWNERSHIP rather than just distances.' % (a.poses, len(have), have or 'none'))

    rec = json.loads(open(SRC).readline())
    vs = variants(rec['prompt'])
    with open(a.out, 'w') as fh:
        for name, prompt in vs.items():
            q = dict(rec)
            q.update({'prompt': prompt, 'variant': name,
                      'pair': 'MOL1 zap70_cys346 %s' % name,
                      'programme': 'ZAP70_mol1', 'site': 'CYS346',
                      'reference_smiles': '',          # mol1 has no reference analogue
                      'pose': 'mol1_model_0',
                      'PREDICTED_GEOMETRY': True})
            fh.write(json.dumps(q) + '\n')
    for name, prompt in vs.items():
        print('%-10s %5d chars  "1.40" x%d' % (name, len(prompt), prompt.count('1.40')))
    print('\nwrote %d variants -> %s' % (len(vs), a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
