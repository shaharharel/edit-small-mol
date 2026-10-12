#!/usr/bin/env python3
"""Score a trained arm on the held-out targets, and score the ablation against the same yardstick.

    python eval_sft.py --arm pocket --run sft_runs/pocket --out sft_runs/pocket/eval
    python eval_sft.py --arm blind  --run sft_runs/blind  --out sft_runs/blind/eval
    python eval_sft.py --arm pocket --base-only --out sft_runs/base_pocket/eval

Four numbers per arm, and one of them matters more than the rest.

1. HELD-OUT LOSS / PERPLEXITY, teacher-forced on the assistant turn only, with exactly the mask
   training used (`ChatSFT` is imported from train_sft.py rather than reimplemented, so the two can
   never drift). Comparable across arms because the assistant turns are byte-identical between the
   two JSONL files -- only the prompt differs, so the loss difference is attributable to the prompt.

2. SMILES VALIDITY -- RDKit parses the generated SMILES -- and ECHO RATE, how often the model just
   handed back its input. Validity alone is gameable by copying the input, which is always valid and
   always wrong, so neither number means anything without the other. Echo is measured on canonical
   SMILES, since a re-spelled input is still an echo.

3. GROUNDEDNESS: of the residues the generated sentence names, the fraction that appear in that
   pair's MODEL_INPUT contact table. This is the number that matters. A fluent sentence citing
   LEU218.CD1 at 3.4 A is worthless if LEU218 is not in the pocket, and only this metric can tell
   the difference.

   It is reported against the REFERENCE CEILING, which is ~0.90, not 1.0. The chair's own sentences
   name a residue the input's crystal never touched about one time in ten, because the analogue's
   new atoms reach residues the input molecule did not -- genuinely unknowable from MODEL_INPUT.
   Scoring a model at 0.90 as "10% hallucination" would be reading the ceiling as a failure.

   The blind arm is scored against the same tables it was never shown. That is the point: pocket
   dependence means the pocket-aware arm's citations are grounded BECAUSE it read the table, and the
   comparison is only meaningful if both arms are held to the pocket's actual contents.

Per-row generations are written to `generations.jsonl` so every aggregate here can be recomputed, or
disputed, without re-running the model.
"""
import argparse
import json
import os
import re
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sft_data import chat_ids, residue_tokens  # noqa: E402  (shared so neither can drift)
from train_sft import CFG, ChatSFT, load_rows  # noqa: E402


def canon(smi):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def parse_completion(text):
    """Split a completion into (sentence, smiles).

    The trained format is the sentence then a `SMILES: ...` line. A model that drifts off format is
    not rescued by clever parsing -- that would hide the drift -- so this takes the last SMILES: line
    and whatever preceded it, and records nothing if the line is absent.
    """
    smi, sent_lines = None, []
    for line in text.strip().splitlines():
        m = re.match(r'\s*SMILES:\s*(\S+)', line)
        if m:
            smi = m.group(1)
        elif smi is None:
            sent_lines.append(line)
    return ' '.join(sent_lines).strip(), smi


@torch.no_grad()
def heldout_loss(model, ds, device):
    """Mean per-token NLL over assistant tokens only. Batch 1, so no padding can skew it."""
    tot_nll, tot_tok = 0.0, 0
    per_example = []
    for i in range(len(ds)):
        e = ds[i]
        ids = torch.tensor([e['input_ids']], device=device)
        lab = torch.tensor([e['labels']], device=device)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = out.logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        nll = torch.nn.functional.cross_entropy(
            logits[mask], tgt[mask], reduction='sum').item()
        n = int(mask.sum())
        tot_nll += nll
        tot_tok += n
        per_example.append(nll / max(1, n))
    return {'loss': tot_nll / tot_tok,
            'perplexity': float(torch.exp(torch.tensor(tot_nll / tot_tok))),
            'n_label_tokens': tot_tok,
            'mean_per_example_loss': statistics.mean(per_example)}


@torch.no_grad()
def generate(model, tok, rows, device, max_new=192):
    outs = []
    for r in rows:
        ids = torch.tensor([chat_ids(tok, r['messages'][:2], add_generation_prompt=True)],
                           device=device)
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        outs.append(text)
    return outs


def score(rows, completions):
    recs, g_hit, g_tot, g_macro = [], 0, 0, []
    n_valid = n_smiles = n_echo = n_named = n_fully = 0
    for r, text in zip(rows, completions):
        sent, smi = parse_completion(text)
        cs, ci = canon(smi), canon(r['input_smiles'])
        valid = cs is not None
        echo = bool(valid and ci and cs == ci)
        tbl = set(r['table_residues'])
        named = residue_tokens(sent)
        h = sum(1 for n in named if n in tbl)
        if smi:
            n_smiles += 1
        n_valid += valid
        n_echo += echo
        if named:
            n_named += 1
            g_hit += h
            g_tot += len(named)
            g_macro.append(h / len(named))
            n_fully += (h == len(named))
        recs.append({'pair': r['pair'], 'target': r['target'], 'split': r['split'],
                     'completion': text, 'gen_sentence': sent, 'gen_smiles': smi,
                     'gen_smiles_canonical': cs, 'valid': valid, 'echo': echo,
                     'named_residues': sorted(named),
                     'grounded_residues': sorted(n for n in named if n in tbl),
                     'ungrounded_residues': sorted(n for n in named if n not in tbl),
                     'groundedness': (h / len(named)) if named else None,
                     'reference_sentence': r['reference_sentence'],
                     'reference_smiles': r['analogue_smiles'],
                     'reference_residues': r['reference_residues'],
                     'n_table_residues': len(tbl)})
    n = len(rows)
    summary = {
        'n_val': n,
        'smiles_line_emitted': round(n_smiles / n, 4),
        'validity': round(n_valid / n, 4),
        'validity_of_emitted': round(n_valid / n_smiles, 4) if n_smiles else None,
        'echo_rate': round(n_echo / n, 4),
        'valid_and_not_echo': round(sum(1 for x in recs if x['valid'] and not x['echo']) / n, 4),
        'sentences_naming_a_residue': round(n_named / n, 4),
        'groundedness_micro': round(g_hit / g_tot, 4) if g_tot else None,
        'groundedness_macro': round(statistics.mean(g_macro), 4) if g_macro else None,
        'fully_grounded_frac': round(n_fully / n_named, 4) if n_named else None,
        'residues_named_per_sentence': round(g_tot / n_named, 2) if n_named else None,
    }
    return summary, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['pocket', 'blind'], required=True)
    ap.add_argument('--run', default=None, help='run dir containing adapter/ (omit with --base-only)')
    ap.add_argument('--data-dir', default='sft_out')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-only', action='store_true',
                    help='score the untuned base model, the zero-shot floor for both arms')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-len', type=int, default=CFG['max_len'])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.data_dir, 'sft_%s.jsonl' % a.arm)
    _, va_rows = load_rows(path, a.arm)
    if a.limit:
        va_rows = va_rows[:a.limit]

    src = os.path.join(a.run, 'adapter') if a.run else CFG['model_id']
    tok = AutoTokenizer.from_pretrained(src if a.run else CFG['model_id'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CFG['model_id'], dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            CFG['model_id'], torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda')
    if not a.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(a.run, 'adapter'))
        model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    ds = ChatSFT(va_rows, tok, a.max_len)
    loss = heldout_loss(model, ds, device)
    comps = generate(model, tok, va_rows, device)
    summary, recs = score(va_rows, comps)
    summary.update(loss)
    summary['arm'] = a.arm
    summary['base_only'] = a.base_only
    summary['run'] = a.run

    # The ceiling, recomputed here from the same rows so the comparison is self-contained.
    ref_macro, rh, rt, rfull, rnamed = [], 0, 0, 0, 0
    for r in va_rows:
        tbl, named = set(r['table_residues']), set(r['reference_residues'])
        if not named:
            continue
        rnamed += 1
        h = sum(1 for n in named if n in tbl)
        rh += h
        rt += len(named)
        ref_macro.append(h / len(named))
        rfull += (h == len(named))
    summary['reference_ceiling'] = {
        'groundedness_micro': round(rh / rt, 4) if rt else None,
        'groundedness_macro': round(statistics.mean(ref_macro), 4) if ref_macro else None,
        'fully_grounded_frac': round(rfull / rnamed, 4) if rnamed else None,
        'residues_named_per_sentence': round(rt / rnamed, 2) if rnamed else None}

    with open(os.path.join(a.out, 'generations.jsonl'), 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    json.dump(summary, open(os.path.join(a.out, 'eval_summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print('\n' + '=' * 90)
    for r in recs[:5]:
        print('PAIR %s  target %s' % (r['pair'], r['target']))
        print('  GEN : %s' % r['gen_sentence'][:300])
        print('  GEN SMILES: %s   valid=%s echo=%s' % (r['gen_smiles'], r['valid'], r['echo']))
        print('  REF : %s' % r['reference_sentence'][:300])
        print('  REF SMILES: %s' % r['reference_smiles'])
        print('  grounded %s / ungrounded %s' % (r['grounded_residues'], r['ungrounded_residues']))
        print('-' * 90)


if __name__ == '__main__':
    main()
