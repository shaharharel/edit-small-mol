#!/usr/bin/env python3
"""Rewrite a built arm into ether0's native think/answer schema.

ether0 adds four tokens to the vocabulary -- <|think_start|> 131072, <|think_end|> 131073,
<|answer_start|> 131074, <|answer_end|> 131075 -- and its GRPO reward was computed ONLY on the
answer span. Two things follow, and both were visible in the g2 run that scored format compliance
0.000 with a 99-word mean:

  * It has a strong prior to emit reasoning before answering, and nothing we did asked for it.
  * It learned the think span is free -- unbounded length, no penalty -- so left to itself it
    rambles and often never closes with a clean answer.

So this builder stops fighting the prior. The prompt is unchanged (the pocket table still goes in
exactly as before); the completion becomes

    <|think_start|>
    changing: <site-of-change atoms>
    <atom> owns <RES.ATOM> at <d> A        <- one line per changed atom that HAS a contact
    <atom> no contact within 4.0 A         <- explicit, because this clause is abused as filler
    <|think_end|><|answer_start|>
    <the one-sentence rationale>
    <the analogue SMILES>
    <|answer_end|>

THE THINK SPAN IS NOT DECORATION. It is the atom-attribution statement the chemists said was
missing. All three reviewers independently found the dominant defect to be a real input-side
distance attached to an atom that does not own it, and reviewer 3 specified the fix as: every
quoted distance must match a contact-table row AND its owning atom must be in site_of_change for a
loss claim. Making the model emit that mapping BEFORE the prose does two things -- it conditions
the sentence on the atoms being changed rather than on the whole input ligand, which is the
conditioning defect we measured; and it produces a machine-checkable intermediate, so a verifier
can score the attribution directly instead of inferring it from free text.

site_of_change is LABEL_ONLY, which is correct here: it is in the COMPLETION, never the prompt.
At inference the model must decide for itself which atoms it is changing, which is exactly the
decision we want it to make explicit.

    python build_ether0_schema.py --in-dir sft_out_g3 --out-dir sft_out_g3_ether0
"""
import argparse
import json
import os

THINK_START, THINK_END = '<|think_start|>', '<|think_end|>'
ANS_START, ANS_END = '<|answer_start|>', '<|answer_end|>'

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'


def load_evidence(pairs):
    out = {}
    want = set(pairs)
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in want:
            out[r['pair']] = r
    return out


def think_span(rec):
    """The attribution statement: which atoms change, and what each one actually owns."""
    mi = rec['MODEL_INPUT']
    lo = rec.get('LABEL_ONLY') or {}
    soc = [x['atom'] for x in (lo.get('site_of_change') or [])]
    by_atom = {a.get('atom'): a for a in (mi.get('atoms') or [])}
    lines = ['changing: %s' % (', '.join(soc) if soc else 'unspecified')]
    for name in soc:
        a = by_atom.get(name)
        if a is None:
            # The edited atom is not in the input table at all -- a pure addition. Say so rather
            # than invent a contact for it, which is the single most common fabrication we measured.
            lines.append('%s is new, no input-side contact' % name)
            continue
        res, d = a.get('faces'), a.get('distance_A')
        if res and d is not None:
            lines.append('%s owns %s at %.2f A' % (name, res, d))
        else:
            lines.append('%s no contact within 4.0 A' % name)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    for arm in ('sft_pocket.jsonl', 'sft_blind.jsonl'):
        src = os.path.join(a.in_dir, arm)
        if not os.path.exists(src):
            continue
        rows = [json.loads(l) for l in open(src)]
        recs = load_evidence([r['pair'] for r in rows])
        n_think = n_plain = 0
        with open(os.path.join(a.out_dir, arm), 'w') as fh:
            for r in rows:
                if r['split'] == 'excluded':
                    # REJECTs keep their empty completion; they are never trained or scored.
                    fh.write(json.dumps(r) + '\n')
                    continue
                rec = recs.get(r['pair'])
                sentence = r['reference_sentence'].strip()
                smiles = r['analogue_smiles'].strip()
                if rec is not None:
                    t = think_span(rec)
                    n_think += 1
                else:
                    t = 'changing: unspecified'
                    n_plain += 1
                # The SMILES keeps its `SMILES:` prefix INSIDE the answer span. That is not
                # cosmetic: eval_sft's format_compliance requires the extractor to reach the
                # molecule by the trained `smiles_line` route, so emitting a bare SMILES here
                # would score 0.000 for shape reasons and we would misread it as the same
                # instruction-following failure we are trying to fix.
                r['completion'] = ('%s\n%s\n%s\n%s\nSMILES: %s\n%s'
                                   % (THINK_START, t, THINK_END + ANS_START,
                                      sentence, smiles, ANS_END))
                fh.write(json.dumps(r) + '\n')
        print('%s: %d rows (%d with an attribution think span, %d without evidence)'
              % (arm, len(rows), n_think, n_plain))

    stats_src = os.path.join(a.in_dir, 'split_stats.json')
    if os.path.exists(stats_src):
        s = json.load(open(stats_src))
        s['schema'] = 'ether0_think_answer'
        s['think_span'] = 'site_of_change atoms with the contact each one owns, from MODEL_INPUT'
        s['derived_from'] = a.in_dir
        json.dump(s, open(os.path.join(a.out_dir, 'split_stats.json'), 'w'), indent=2)
    print('wrote %s' % a.out_dir)


if __name__ == '__main__':
    main()
