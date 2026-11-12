#!/usr/bin/env python
"""Generate from a prompts JSONL, with sampling. Two things eval_sft.py cannot do.

WHY THIS EXISTS. `eval_sft.py` builds its prompts from a corpus directory and hardcodes
`do_sample=False`, which is right for a reproducible benchmark and wrong for two questions we now
need answered:

  1. TEACHER-FORCED INTENTS. The training target is `<intent sentence>\\nSMILES: <smiles>`, so the
     joint model emits the reason first and decodes the molecule conditioned on its own reason
     tokens. Replacing that sentence with one we wrote and letting the model continue is therefore
     not a trick -- it is the same mechanism the model already uses on itself. It needs a prompt
     that already ends mid-response, which only a prompts file can express.
  2. A DIVERSE POPULATION PER MOLECULE. Every number we hold was produced greedily, one generation
     per prompt, so an identical prompt gave an identical molecule and "uniqueness" measured the
     redundancy of the evaluation set rather than the model. A chemist wants several options for
     one hit, so we need `n` samples at a temperature.

Each input row is echoed into the output with the generation attached, so downstream scoring joins
on whatever keys the prompt file carried (`arm`, `axis`, `input_smiles`, ...) without a second
lookup. With --num-samples > 1 a row appears once per sample, carrying `sample_idx`.

SMILES EXTRACTION IS SHARED, NOT REIMPLEMENTED. It imports eval_sft's extractor, because a second
dialect of "find the SMILES in the completion" is how two runs come to disagree about validity.

    python gen_from_prompts.py --model chemdfm --adapter <ckpt> \\
        --prompts steer_prompts.jsonl --out steer_gens.jsonl           # greedy, 1 per prompt
    python gen_from_prompts.py ... --num-samples 10 --temperature 0.9 --top-p 0.95
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_sft as E           # noqa: E402
import train_sft               # noqa: E402


@torch.no_grad()
def run(model, tok, rows, device, max_new, n_samples, temperature, top_p, seed):
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = torch.tensor([tok(r['prompt'], add_special_tokens=True)['input_ids']], device=device)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                  pad_token_id=E.PAD_ID, eos_token_id=E.EOS_ID)
        if n_samples > 1 or temperature is not None:
            # A sampled run is only reproducible if the seed is set per ROW: setting it once at
            # the top makes row k depend on how many tokens rows 0..k-1 happened to draw, so a
            # resumed or reordered run silently produces different molecules.
            torch.manual_seed(seed + i)
            kw.update(do_sample=True, temperature=(temperature if temperature is not None else 1.0),
                      top_p=(top_p if top_p is not None else 1.0),
                      num_return_sequences=n_samples)
        else:
            kw.update(do_sample=False)
        gen = model.generate(**kw)
        for s in range(gen.shape[0]):
            txt = tok.decode(gen[s][ids.shape[1]:], skip_special_tokens=True)
            smi, how = E.extract_smiles(txt)
            rec = dict(r)
            rec.pop('prompt', None)          # the prompt is in the input file; do not duplicate it
            rec.update(sample_idx=s, completion=txt, gen_smiles=smi, extraction_method=how)
            out.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print('[gen] %d/%d prompts  %.2fs/prompt  elapsed %.1f min  eta %.1f min'
                  % (i + 1, len(rows), el / (i + 1), el / 60,
                     (len(rows) - i - 1) * el / (i + 1) / 60), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(E.MODELS), required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--base-adapter', dest='base_adapter', default=None)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-new', dest='max_new', type=int, default=256)
    ap.add_argument('--num-samples', dest='num_samples', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=None,
                    help='omit for greedy. Setting it switches sampling on.')
    ap.add_argument('--top-p', dest='top_p', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-merge', dest='no_merge', action='store_true')
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts)]
    if a.limit:
        rows = rows[:a.limit]
    assert all('prompt' in r for r in rows), 'every row of the prompts file needs a `prompt`'
    print('prompts: %d | samples each: %d | temperature %s | top_p %s'
          % (len(rows), a.num_samples, a.temperature, a.top_p))

    model_path = E.MODELS[a.model]
    E.set_special_ids(a.model)
    E.PAD_ID, E.EOS_ID = train_sft.PAD_ID, train_sft.EOS_ID
    tok = E.build_tokenizer(model_path, a.model)
    model = E.load_model(model_path, E.CFG['bf16'], device_map='cuda')
    if a.base_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.base_adapter).merge_and_unload()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        if not a.no_merge:
            model = model.merge_and_unload()
    model.eval()

    out = run(model, tok, rows, 'cuda', a.max_new, a.num_samples, a.temperature, a.top_p, a.seed)
    with open(a.out, 'w') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    meta = {'prompts': len(rows), 'generations': len(out), 'model': a.model,
            'adapter': a.adapter, 'num_samples': a.num_samples,
            'temperature': a.temperature, 'top_p': a.top_p, 'seed': a.seed,
            'greedy': a.num_samples == 1 and a.temperature is None}
    json.dump(meta, open(a.out + '.meta.json', 'w'), indent=1)
    print('wrote %d generations to %s' % (len(out), a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
