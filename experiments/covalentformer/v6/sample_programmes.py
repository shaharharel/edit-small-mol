#!/usr/bin/env python3
"""Sample candidate analogues for the unseen covalent programmes.

Greedy decoding gives one answer per prompt, which is not how this model gets used: the mol1 pool
was 120 samples from one prompt, and a chemist picks from a pool. So this samples N per prompt at
the training temperature, and ALSO records the greedy answer, because the two differ materially --
v4's training reward rose +0.117 on sampled rollouts while its greedy holdout stayed flat, so
sampled and argmax quality are not interchangeable and a report should not quietly mix them.

The reference analogue is carried through to the output but was never in the prompt. It is what
makes these programmes worth more than mol1: mol1 had no reference, so nothing it proposed could be
scored against what the programme actually did.
"""
import argparse
import json
import os
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {'chemdfm': 'OpenDFM/ChemDFM-v2.0-14B'}


def split_completion(text):
    """Same strip as train_grpo._split_completion: special tokens off BEFORE the SMILES is read."""
    t = re.sub(r'<\|[^|>]*\|>', '', text).replace('</s>', '')
    sent = t.split('SMILES:')[0].strip()
    smi = t.split('SMILES:')[-1].strip().split()[0] if 'SMILES:' in t else ''
    return ' '.join(sent.split()), smi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='chemdfm')
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=24, help='samples per prompt')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', dest='top_p', type=float, default=0.95)
    ap.add_argument('--max-new', dest='max_new', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8, help='samples per generate() call')
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts)]
    print('[gen] %d prompts, %d samples each, temp %.2f' % (len(rows), a.n, a.temperature))

    tok = AutoTokenizer.from_pretrained(MODELS[a.model], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[a.model], torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True
    dev = next(model.parameters()).device
    # tok.eos_token_id, not the EOS_ID constant in train_sft: that constant is 128009, a LLAMA-3
    # token id, while ChemDFM is Qwen-based and stops on <|im_end|> = 151645. Passing the wrong one
    # lets generation run past the real end of the answer.
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    print('[gen] eos=%s pad=%s' % (eos, pad))

    out = open(a.out, 'w')
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        ids = torch.tensor([tok(r['prompt'], add_special_tokens=True)['input_ids']], device=dev)
        mask = torch.ones_like(ids)
        got = []
        with torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                               max_new_tokens=a.max_new, pad_token_id=pad, eos_token_id=eos)
            sent, smi = split_completion(tok.decode(g[0][ids.shape[1]:],
                                                    skip_special_tokens=False))
            got.append({'decode': 'greedy', 'gen_sentence': sent, 'gen_smiles': smi})
            done = 0
            while done < a.n:
                k = min(a.batch, a.n - done)
                g = model.generate(input_ids=ids, attention_mask=mask, do_sample=True,
                                   temperature=a.temperature, top_p=a.top_p,
                                   num_return_sequences=k, max_new_tokens=a.max_new,
                                   pad_token_id=pad, eos_token_id=eos)
                for row in g:
                    sent, smi = split_completion(tok.decode(row[ids.shape[1]:],
                                                            skip_special_tokens=False))
                    got.append({'decode': 'sampled', 'gen_sentence': sent, 'gen_smiles': smi})
                done += k
        rec = dict(r)
        rec.pop('prompt', None)          # the prompt is already in programme_prompts.jsonl
        rec['generations'] = got
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print('[gen] %2d/%d  %-22s %-11s %d generations  %.0fs'
              % (i, len(rows), r['pair'], r['programme'], len(got), time.time() - t0))
    out.close()
    print('[gen] done -> %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
