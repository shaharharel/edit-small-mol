#!/usr/bin/env python3
"""ether0 schema variant B: reasoning in the THINK span, the molecule in the ANSWER span.

Variant A (build_ether0_schema.py) put our rationale in the answer span and a derived attribution
table in think. That inverts ether0's own semantics: its GRPO reward ran RDKit validity, a
purchasability Bloom filter and a Tanimoto >= 0.7 check on the ANSWER span, so that span was
trained to hold a VERIFIABLE MOLECULE, not prose. Variant A therefore asks the rewarded slot to
emit a sentence, and puts the reasoning where the reward never looked.

Variant B aligns with how the base was actually trained:

    <|think_start|>
    changing: C2, C6, C9                  <- attribution scaffold, kept: it is what makes
    C6 owns LEU418.O at 3.45 A               checking a cited distance a lookup, not an NLP task
    <the rationale sentence>              <- OUR LABEL, in the reasoning slot
    <|think_end|><|answer_start|>
    SMILES: <analogue>                    <- the molecule, in the slot the reward scored
    <|answer_end|>

Two deliberate differences from every other arm, recorded in split_stats.json rather than hidden:

  * The final INSTRUCTION line changes. The shared prompt says "Answer with the sentence, then a
    line of the form 'SMILES: <smiles>'", which directly contradicts this target. An instruction
    that disagrees with the completion teaches the model to ignore instructions, so it is rewritten
    for this arm only -- and the prompt is consequently NOT byte-identical to the other arms.
  * The `SMILES:` prefix is kept inside the answer span even though ether0's own answers are bare
    molecules, because eval_sft's format_compliance requires the extractor to reach the molecule by
    the trained `smiles_line` route. Dropping the prefix would score 0.000 for shape reasons and be
    misread as the instruction-following failure we are trying to measure.

    python build_ether0_schema_v2.py --in-dir sft_out_g3 --out-dir sft_out_g3_ether0_v2
"""
import argparse
import json
import os

THINK_START, THINK_END = '<|think_start|>', '<|think_end|>'
ANS_START, ANS_END = '<|answer_start|>', '<|answer_end|>'

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'

OLD_INSTRUCTION = "Answer with the sentence, then a line of the form 'SMILES: <smiles>'."
NEW_INSTRUCTION = ("Reason first, then give the analogue as your answer on a line of the form "
                   "'SMILES: <smiles>'.")


def load_evidence(pairs):
    want = set(pairs)
    out = {}
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in want:
            out[r['pair']] = r
    return out


def attribution_lines(rec):
    """Which atoms change, and what each one actually owns on the input side."""
    mi = rec['MODEL_INPUT']
    lo = rec.get('LABEL_ONLY') or {}
    soc = [x['atom'] for x in (lo.get('site_of_change') or [])]
    by_atom = {a.get('atom'): a for a in (mi.get('atoms') or [])}
    lines = ['changing: %s' % (', '.join(soc) if soc else 'unspecified')]
    for name in soc:
        a = by_atom.get(name)
        if a is None:
            lines.append('%s is new, no input-side contact' % name)
            continue
        res, d = a.get('faces'), a.get('distance_A')
        if res and d is not None:
            lines.append('%s owns %s at %.2f A' % (name, res, d))
        else:
            lines.append('%s no contact within 4.0 A' % name)
    return lines


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
        n_instr = n_done = 0
        with open(os.path.join(a.out_dir, arm), 'w') as fh:
            for r in rows:
                if OLD_INSTRUCTION in r['prompt']:
                    r['prompt'] = r['prompt'].replace(OLD_INSTRUCTION, NEW_INSTRUCTION)
                    n_instr += 1
                if r['split'] == 'excluded':
                    fh.write(json.dumps(r) + '\n')
                    continue
                # NO attribution scaffold here, deliberately. Variant A carries one; it has never
                # been tested, no other arm has it, and measured on variant A it was 188 of 237
                # supervised tokens -- 79% of the gradient spent reciting a generated table rather
                # than on the reasoning or the molecule. Variant B's think span is the rationale
                # and nothing else, so the comparison against variant A isolates one change.
                think = [r['reference_sentence'].strip()]
                r['completion'] = ('%s\n%s\n%s\nSMILES: %s\n%s'
                                   % (THINK_START, '\n'.join(think), THINK_END + ANS_START,
                                      r['analogue_smiles'].strip(), ANS_END))
                n_done += 1
                fh.write(json.dumps(r) + '\n')
        print('%s: %d rows, %d completions rewritten, %d prompts re-instructed'
              % (arm, len(rows), n_done, n_instr))

    stats_src = os.path.join(a.in_dir, 'split_stats.json')
    if os.path.exists(stats_src):
        s = json.load(open(stats_src))
        s['schema'] = 'ether0_think_answer_v2'
        s['think_span'] = 'attribution lines THEN the rationale sentence'
        s['answer_span'] = 'the analogue SMILES only -- the slot ether0 GRPO actually scored'
        s['prompt_differs_from_other_arms'] = True
        s['prompt_change'] = {'from': OLD_INSTRUCTION, 'to': NEW_INSTRUCTION}
        s['derived_from'] = a.in_dir
        json.dump(s, open(os.path.join(a.out_dir, 'split_stats.json'), 'w'), indent=2)
    print('wrote %s' % a.out_dir)


if __name__ == '__main__':
    main()
