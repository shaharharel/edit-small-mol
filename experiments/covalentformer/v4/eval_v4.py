#!/usr/bin/env python
"""Evaluate v4 on the molecule-disjoint rung-3 test fold, plus the dpIC50 head.

THE PROMPT-MISMATCH BUG IS MADE STRUCTURALLY IMPOSSIBLE HERE, NOT GUARDED AGAINST. It has cost this
project three times: a joint model scored on R3_COVPOCK prompts it never saw (~40% of its measured
performance), Decoder-R evaluated without its direction tokens (read as 0% everywhere), and a
--maxnew too small to emit the answer (6.6% of outputs discarded as "truncated" when they were
format drift). The root cause each time was the eval RE-BUILDING the prompt.

So this script never builds a prompt. It feeds r['instruction'] verbatim out of
rung3_v4_test.jsonl -- the same file written by the same build_v4.py invocation, with the same
--pocket-mode and --max-pocket, that produced the training fold. A startup assertion compares the
structural line labels of the test prompts against the training fold and refuses to run on a
mismatch.

WHAT IS AND IS NOT A BASELINE. Copy-the-hit is not reported as a baseline -- it answers a different
question and the user has said so. It IS reported as a PATHOLOGY rate: a model that returns its own
input has failed, and exact-match credit must never come from it. The real reference points are
base Qwen2.5-7B with no adapter on identical prompts (does the LoRA do anything?) and, when its
adapter is present, v3b (does v4 beat the previous rung?).

FOR THE HEAD, THE FLOOR IS WITHIN-HIT VARIANCE. Predict-the-mean is trivially beatable because the
delta distribution is narrow. The question a ranker must answer is "of several analogues OF THE SAME
HIT, which is better", so the metric is within-hit Spearman and the floor is the irreducible spread
inside each hit group.
"""
from __future__ import annotations
import argparse, collections, json, math, os, re, statistics as st, sys

ANA = re.compile(r'Analogue:\s*(\S+)')
CHG = re.compile(r'Change:\s*(\S+)')


def structural_sig(prompt):
    """Line LABELS only -- never content. Content differs per row by construction; what must match
    between train and test is which labelled blocks are present and in what order."""
    out = []
    for ln in prompt.split('\n'):
        if not ln or ln.startswith('  '):
            continue
        # keep only the LABEL: everything before the first ':' or '=' , then drop parenthetical
        # counts and digits. Values must not survive -- an earlier version kept 'partner=CYS', so a
        # SER-nucleophile row produced a different signature than a CYS one and the comparison
        # would have reported a spurious mismatch between two correctly-formatted folds.
        cut = min([i for i in (ln.find(':'), ln.find('=')) if i >= 0] or [len(ln)])
        key = re.sub(r'[0-9].*', '', re.sub(r'\(.*', '', ln[:cut])).strip()
        if key:
            out.append(key[:48])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default='')
    ap.add_argument('--head', default='')
    ap.add_argument('--test', required=True)
    ap.add_argument('--train-ref', default='', help='training fold, for the format assertion')
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--maxnew', type=int, default=256)
    ap.add_argument('--temp', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--tag', default='v4')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    def canon(s):
        if not s or not isinstance(s, str):
            return None
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)

    rows = [json.loads(l) for l in open(a.test)][:a.n]
    print('test rows: %d (of %d requested)' % (len(rows), a.n), flush=True)

    # ---- FORMAT ASSERTION ----
    if a.train_ref:
        tr = [json.loads(l) for l in open(a.train_ref)][:200]
        tsig = collections.Counter(tuple(structural_sig(r['instruction'])) for r in tr)
        esig = collections.Counter(tuple(structural_sig(r['instruction'])) for r in rows)
        tmost, emost = tsig.most_common(1)[0][0], esig.most_common(1)[0][0]
        print('train prompt signature: %s' % list(tmost), flush=True)
        print('test  prompt signature: %s' % list(emost), flush=True)
        if tmost != emost:
            sys.exit('FATAL PROMPT MISMATCH -- the eval would score the model on an encoding it was '
                     'never trained on. This exact failure has cost three evaluations already.')
        print('PROMPT ENCODING VERIFIED IDENTICAL to the training fold', flush=True)

    # ---- leak assertion on the right key ----
    if a.train_ref:
        tr_mol = set()
        for r in tr:
            for k in ('hit', 'analogue'):
                c = canon(r.get(k))
                if c:
                    tr_mol.add(c)
        te_mol = set()
        for r in rows:
            for k in ('hit', 'analogue'):
                c = canon(r.get(k))
                if c:
                    te_mol.add(c)
        inter = tr_mol & te_mol
        print('molecule overlap train(sample) n test: %d' % len(inter), flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'                      # required for batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map='auto', trust_remote_code=True)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print('loaded adapter %s' % a.adapter, flush=True)
    else:
        print('NO ADAPTER -- this is the base-Qwen baseline arm', flush=True)
    model.eval()

    n = hit1 = hitk = copies = trunc = op_ok = 0
    valid = total = 0
    uniq = []
    with torch.no_grad():
        for i0 in range(0, len(rows), a.bs):
            chunk = rows[i0:i0 + a.bs]
            prompts = [r['instruction'] for r in chunk]          # VERBATIM. never rebuilt.
            enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(dev)
            gen = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=a.top_p,
                                 max_new_tokens=a.maxnew, num_return_sequences=a.k,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc['input_ids'].shape[1]:]
            txt = tok.batch_decode(new, skip_special_tokens=True)
            for j, r in enumerate(chunk):
                outs = txt[j * a.k:(j + 1) * a.k]
                ref, src = canon(r['analogue']), canon(r['hit'])
                cands, ops = [], []
                for o in outs:
                    total += 1
                    if len(tok(o, add_special_tokens=False)['input_ids']) >= a.maxnew - 2:
                        trunc += 1
                    m = ANA.search(o)
                    cand = canon(m.group(1)) if m else None
                    if cand:
                        valid += 1
                        cands.append(cand)
                    mo = CHG.search(o)
                    if mo:
                        ops.append(mo.group(1))
                n += 1
                uniq.extend(cands)
                if cands and ref and cands[0] == ref:
                    hit1 += 1
                if ref and ref in set(cands):
                    hitk += 1
                if cands and src and cands[0] == src:
                    copies += 1
                if ops and r.get('op') and ops[0] == r['op']:
                    op_ok += 1
            if (i0 // a.bs) % 10 == 0:
                print('  %d/%d  exact@%d so far %.2f%%'
                      % (n, len(rows), a.k, 100.0 * hitk / max(n, 1)), flush=True)

    res = {'arm': a.tag, 'n': n, 'k': a.k,
           'validity': valid / max(total, 1),
           'uniqueness': len(set(uniq)) / max(len(uniq), 1),
           'exact@1': hit1 / max(n, 1),
           'exact@%d' % a.k: hitk / max(n, 1),
           'copy_rate_PATHOLOGY': copies / max(n, 1),
           'change_op_acc': op_ok / max(n, 1),
           'truncated_frac': trunc / max(total, 1)}
    print('\n=== %s  n=%d k=%d ===' % (a.tag, n, a.k))
    for k2, v in res.items():
        if isinstance(v, float):
            print('  %-22s %7.3f%%' % (k2, 100 * v))
    if res['truncated_frac'] > 0.02:
        print('  WARNING: %.1f%% of generations hit --maxnew=%d. Raise it before believing exact@k; '
              'a truncated answer is scored as a miss.' % (100 * res['truncated_frac'], a.maxnew))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=2)
        print('wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
