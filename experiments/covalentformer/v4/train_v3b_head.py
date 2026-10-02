#!/usr/bin/env python
"""v3b: the dPIC50 head, with the label leak removed and the head conditioned on (hit, analogue).

THE LEAK IN v3. The response text was
    Change: trim 7 heavy atoms
    Predicted dpIC50: +0.92        <-- the regression target, written in the response
    Analogue: C=CC(=O)...
and the head pooled hidden states over the WHOLE response span, so it read its own label as input --
in 100% of rows. Training reg-MSE fell to 0.0002 (RMSE 0.014 against a target with sd 0.54, R^2~0.999),
which is ~40x better than the theoretical ceiling for this task and is the signature of copying, not
prediction. Every v3 potency number is therefore void.

WHAT v3b DOES INSTEAD.
  1. The 'Predicted dpIC50:' line is STRIPPED from the response, so the number exists nowhere in the
     model's input. The head becomes the only predictor of the delta.
  2. The head reads exactly TWO spans and nothing else, located by construction, not by searching:
        hit_vec = mean hidden state over the HIT SMILES tokens only
        ana_vec = mean hidden state over the ANALOGUE SMILES tokens only
     No task text, no 'Measured: pIC50' line, no 'Change:' line reaches the head -- only the two
     molecules. The LM still sees the full prompt through attention; the HEAD's pooled input does not.
     and is fed [hit_vec, ana_vec, ana_vec - hit_vec]. The difference is the representation of the EDIT,
     which is what a delta measures, and it is what lets the head score different candidates of the same
     hit differently -- the thing a ranker must do.
  3. A startup assertion fails loudly if the target string ever appears in a response again.

WHY ONLY THE MOLECULES. The prompt also carries 'Measured: pIC50' for the hit, which is legitimate
input but is not a molecule; restricting the head to the two SMILES spans keeps its input to exactly
the pair whose difference it must score, and removes any route by which prompt phrasing could carry
information about the answer.

WHAT TO COMPARE AGAINST. Not predict-the-mean, which is trivially beatable. The same hit carries 28.8
analogues whose deltas span 0.82 log units, so the irreducible within-hit variance is the real floor.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--head-lr', type=float, default=1e-3)
    ap.add_argument('--reg-weight', type=float, default=0.5)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--maxlen', type=int, default=768)
    ap.add_argument('--ckpt-every', type=int, default=250)
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--auto-resume', action='store_true')
    a = ap.parse_args()

    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader

    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    free = torch.cuda.mem_get_info()[0] / 1048576 if torch.cuda.is_available() else 0
    print('GPU free %.0f MiB' % free, flush=True)
    if torch.cuda.is_available() and free < 30000:
        sys.exit('REFUSING TO START: only %.0f MiB free -- accelerate would place the model on CPU.' % free)

    MARK = 'Analogue: '
    def split_output(o):
        """-> (prefix_without_the_delta_line, analogue_smiles) or None"""
        o = str(o)
        if MARK not in o:
            return None
        pre, smi = o.split(MARK, 1)
        smi = smi.strip().split()[0] if smi.strip() else ''
        # DROP the leaking line entirely
        pre = '\n'.join(l for l in pre.split('\n') if 'dpIC50' not in l and 'dPIC50' not in l)
        pre = pre.strip()
        return ((pre + '\n') if pre else ''), smi

    class Rows(Dataset):
        def __init__(self, path):
            self.rows = []
            leaked = 0
            for l in open(path):
                r = json.loads(l)
                sp = split_output(r.get('output'))
                if sp is None:
                    self.rows.append((r, None, None)); continue
                pre, smi = sp
                if 'dpIC50' in pre or 'dpIC50' in smi:
                    leaked += 1
                self.rows.append((r, pre, smi))
            assert leaked == 0, 'FATAL: %d responses still contain the target string' % leaked
            # DROP ROWS THAT CANNOT TRAIN. If the prompt alone fills maxlen there is no room left for
            # the answer, so every label is -100 and the row contributes no gradient -- and worse,
            # HF's cross-entropy over an empty target set returns NaN, which the batch mean then
            # spreads to every other row. Measured on the v4 mix at maxlen 3072 this is ~2% of rows,
            # entirely rung3/contacts whose pocket block is long.
            #
            # THE FILTER TOKENISES RATHER THAN COUNTING CHARACTERS. A char-based budget is not just
            # imprecise here, it errs in the DANGEROUS direction: prose runs ~2.9 chars/token but the
            # pocket coordinate block runs ~1.25 (3,156 chars -> 2,529 tokens), because every digit
            # and decimal point is its own token. A 2.5x char budget would therefore admit ~5,900-token
            # rows at maxlen 3072 -- exactly the rows this filter exists to remove. One batched pass
            # with the fast tokeniser costs a couple of minutes and is exact.
            prompts = [t[0]['instruction'] +
                       (('\n' + t[0]['input']) if t[0].get('input')
                        and t[0]['input'] not in t[0]['instruction'] else '') + '\n'
                       for t in self.rows]
            plen = []
            B = 2000
            for i0 in range(0, len(prompts), B):
                plen.extend(len(x) for x in
                            tok(prompts[i0:i0 + B], add_special_tokens=False)['input_ids'])
            keep, dropped = [], 0
            for t, n_tok in zip(self.rows, plen):
                if n_tok + 8 >= a.maxlen:      # 8 tokens is the floor for any usable answer
                    dropped += 1
                else:
                    keep.append(t)
            if dropped:
                print('  dropped %d/%d rows (%.2f%%) whose prompt would fill maxlen=%d, leaving no '
                      'room for the answer (all labels -100 -> NaN loss)'
                      % (dropped, len(self.rows), 100.0 * dropped / len(self.rows), a.maxlen),
                      flush=True)
            self.rows = keep
            n = sum(1 for r, _, _ in self.rows if r.get('dpic50') is not None)
            print('%s: %d rows, %d with dpic50 (%.1f%%), leak check PASSED'
                  % (path, len(self.rows), n, 100.0 * n / max(len(self.rows), 1)), flush=True)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r, pre, smi = self.rows[i]
            p = r['instruction'] + (('\n' + r['input']) if r.get('input') and r['input'] not in r['instruction'] else '') + '\n'
            # split the prompt so the HIT SMILES has its own exact token span
            mh = re.search(r'(Hit:\s*)(\S+)', p)
            if mh is None:
                pi = tok(p, add_special_tokens=False)['input_ids']
                p1n = h_n = 0
            else:
                p1 = tok(p[:mh.end(1)], add_special_tokens=False)['input_ids']
                hh = tok(mh.group(2), add_special_tokens=False)['input_ids']
                p2 = tok(p[mh.end(2):], add_special_tokens=False)['input_ids']
                pi = p1 + hh + p2
                p1n, h_n = len(p1), len(hh)
            if smi is None:                       # non-affinity row: LM loss only, no head target
                ai = tok(str(r['output']) + tok.eos_token, add_special_tokens=False)['input_ids']
                ids = (pi + ai)[:a.maxlen]
                lab = ([-100] * len(pi) + ai)[:a.maxlen]
                return {'input_ids': ids, 'labels': lab, 'y': float('nan'),
                        'hs': 0, 'he': 0, 'as_': 0, 'ae': 0}   # no head target on non-affinity rows
            pre_i = tok(pre, add_special_tokens=False)['input_ids'] if pre else []
            mk_i = tok(MARK, add_special_tokens=False)['input_ids']
            smi_i = tok(smi + tok.eos_token, add_special_tokens=False)['input_ids']
            ids = (pi + pre_i + mk_i + smi_i)[:a.maxlen]
            lab = ([-100] * len(pi) + pre_i + mk_i + smi_i)[:a.maxlen]
            hs, he = min(p1n, len(ids)), min(p1n + h_n, len(ids))    # HIT SMILES tokens only
            as_ = min(len(pi) + len(pre_i) + len(mk_i), len(ids))     # analogue SMILES span
            ae = len(ids)
            y = r.get('dpic50')
            return {'input_ids': ids, 'labels': lab,
                    'y': float(y) if y is not None else float('nan'),
                    'hs': hs, 'he': he, 'as_': as_, 'ae': ae}

    def collate(b):
        L = max(len(x['input_ids']) for x in b)
        pad = tok.pad_token_id
        ii = torch.full((len(b), L), pad, dtype=torch.long)
        ll = torch.full((len(b), L), -100, dtype=torch.long)
        am = torch.zeros((len(b), L), dtype=torch.long)
        hm = torch.zeros((len(b), L), dtype=torch.float)
        anm = torch.zeros((len(b), L), dtype=torch.float)
        for k, x in enumerate(b):
            n = len(x['input_ids'])
            ii[k, :n] = torch.tensor(x['input_ids']); ll[k, :n] = torch.tensor(x['labels'])
            am[k, :n] = 1
            hm[k, x['hs']:x['he']] = 1.0
            if x['ae'] > x['as_']:
                anm[k, x['as_']:x['ae']] = 1.0
        return {'input_ids': ii, 'labels': ll, 'attention_mask': am,
                'y': torch.tensor([x['y'] for x in b], dtype=torch.float32),
                'hitmask': hm, 'anamask': anm}

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map='auto', trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias='none',
                     task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                     'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
    # ---- CAPTURE THE FINAL HIDDEN STATE VIA A HOOK, NOT output_hidden_states=True ----
    # The head reads exactly ONE tensor: the last layer's hidden state. Asking the model for
    # output_hidden_states=True instead returns and RETAINS all 29 of them in the autograd graph.
    # Measured cost at bs=2/maxlen=3072: 36.7 of 41 GB resident, which forced bs=2, which held
    # throughput to >=11.7 s/step -- a 56 h epoch. A forward hook on the decoder's final norm gives
    # the identical tensor (HF appends hidden_states[-1] AFTER self.norm, so norm's output IS that
    # element) without keeping the other 28.
    #
    # THE EQUIVALENCE IS ASSERTED AT RUNTIME, not assumed: the first micro-batch is run both ways and
    # compared. If any transformers version ever changes where hidden_states[-1] is taken from, this
    # fails loudly on batch 1 instead of silently training the head on the wrong tensor.
    _dec = model.base_model.model.get_decoder() if hasattr(model, 'base_model') else model.get_decoder()
    _norm = getattr(_dec, 'norm', None)
    assert _norm is not None, 'could not locate the decoder final norm for the hidden-state hook'
    _cap = {}

    def _grab(_m, _i, o):
        _cap['h'] = o[0] if isinstance(o, tuple) else o
    _norm.register_forward_hook(_grab)
    print('final-hidden-state hook attached to %s' % type(_norm).__name__, flush=True)

    hid = model.config.hidden_size
    dev = next(model.parameters()).device
    head = nn.Sequential(nn.LayerNorm(3 * hid), nn.Linear(3 * hid, 256), nn.GELU(),
                         nn.Linear(256, 1)).to(dev).to(torch.float32)
    print('head params %d (input 3*%d = %d)' % (sum(p.numel() for p in head.parameters()), hid, 3 * hid), flush=True)

    class LenGrouped(torch.utils.data.Sampler):
        """Yield BATCHES of similar length so a 50-token row is never padded up to a 3,072-token one.

        WHY THIS EXISTS, MEASURED. v4's corpus is bimodal: rung1/rung2/affinity run 50-300 tokens,
        while rung3 and contacts carry a pocket coordinate block and run ~2,500 (p50). collate() pads
        every micro-batch to its longest member, so with random order at bs=2 roughly half the
        micro-batches pair a short row with a long one and pad the short one by up to 50x. Launched
        that way the run did ZERO optimizer steps in 745 s of 100%-GPU time -- >15 s/step, i.e. 65 h
        for one epoch against a 10.5 h estimate made from v3's uniformly short rows.

        Sorting within a shuffled megabatch keeps batch composition close to random (so the mixture
        the model sees per step is still representative) while making padding nearly free. No row is
        discarded and no hyperparameter changes; the effective batch is still bs x accum."""

        def __init__(self, lengths, bs, seed, mega=64):
            self.lengths, self.bs, self.seed = lengths, bs, seed
            self.mega = mega * bs

        def __iter__(self):
            g = random.Random(self.seed)
            idx = list(range(len(self.lengths)))
            g.shuffle(idx)
            batches = []
            for i in range(0, len(idx), self.mega):
                chunk = sorted(idx[i:i + self.mega], key=lambda j: self.lengths[j])
                batches += [chunk[k:k + self.bs] for k in range(0, len(chunk), self.bs)]
            g.shuffle(batches)
            return iter(batches)

        def __len__(self):
            return (len(self.lengths) + self.bs - 1) // self.bs

    _ds = Rows(a.train)
    # character count as the length key: it is ~monotone in token count for this corpus and costs
    # one pass instead of tokenising 500k rows twice.
    _len = [len(r['instruction']) + len(str(r.get('output') or '')) for r, _, _ in _ds.rows]
    print('length-grouped batching: p50 %d chars, p95 %d, max %d'
          % tuple(sorted(_len)[int(q * len(_len))] for q in (0.5, 0.95, 0.999)), flush=True)
    dtr = DataLoader(_ds, batch_sampler=LenGrouped(_len, a.bs, a.seed), collate_fn=collate,
                     num_workers=12, pin_memory=True, persistent_workers=True,
                     prefetch_factor=4)
    total = int(len(dtr) * a.epochs / a.accum)
    opt = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': a.lr},
        {'params': list(head.parameters()), 'lr': a.head_lr}])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr],
                                              total_steps=max(total, 1), pct_start=0.03)
    reg_fn = nn.MSELoss()
    os.makedirs(a.out, exist_ok=True)
    step = 0
    prog = os.path.join(a.out, 'progress.json')
    if a.auto_resume and os.path.exists(prog):
        try:
            st = json.load(open(prog)); step = int(st.get('step', 0))
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(os.path.join(a.out, 'adapter_model.safetensors')))
            hp = os.path.join(a.out, 'reg_head.pt')
            if os.path.exists(hp):
                head.load_state_dict(torch.load(hp, map_location=dev))
            for _ in range(min(step, max(total - 1, 1))):
                sch.step()
            print('RESUMED step %d/%d lr %.3e' % (step, total, opt.param_groups[0]['lr']), flush=True)
        except Exception as e:
            print('resume failed (%s) -- from scratch' % type(e).__name__, flush=True); step = 0

    def save(cur):
        model.save_pretrained(a.out)
        torch.save(head.state_dict(), os.path.join(a.out, 'reg_head.pt'))
        json.dump({'step': cur, 'total': total}, open(prog, 'w'))

    def pooled(h, m):
        m = m.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    model.train(); head.train()
    runL = runR = 0.0; nR = 0; n_nan = 0
    print('total steps %d' % total, flush=True)
    for ep in range(math.ceil(a.epochs)):
        for i, b in enumerate(dtr):
            y = b.pop('y').to(dev)
            hm = b.pop('hitmask').to(dev); anm = b.pop('anamask').to(dev)
            b = {k: v.to(dev) for k, v in b.items()}
            # HOOK EQUIVALENCE CHECK, UNDER no_grad AND ON ONE SEQUENCE ONLY.
            # The first version of this check ran the real training forward with
            # output_hidden_states=True, which retained all 29 states AND built a graph over them --
            # on the very batch where memory peaks. It OOM'd at bs=4 trying to allocate 5.60 GiB,
            # i.e. the check defeated the optimisation it was there to validate. Running it as a
            # separate no_grad forward over a single sequence costs one cheap pass and no graph.
            if i == 0 and step == 0:
                with torch.no_grad():
                    b1 = {k: v[:1] for k, v in b.items()}
                    r1 = model(**b1, output_hidden_states=True)
                    ref, got = r1.hidden_states[-1], _cap.get('h')
                    assert got is not None, 'the final-norm hook did not fire'
                    assert got.shape == ref.shape and torch.allclose(got, ref, atol=1e-3), (
                        'HOOK MISMATCH: final-norm output != hidden_states[-1] (%s vs %s) -- '
                        'refusing to train the head on the wrong tensor'
                        % (tuple(got.shape), tuple(ref.shape)))
                    print('hook verified == output_hidden_states[-1] on 1 seq, shape %s'
                          % (tuple(ref.shape),), flush=True)
                    del r1, ref, got, b1
                _cap.clear()
                torch.cuda.empty_cache()
            out = model(**b)
            lm = out.loss
            h = _cap.get('h')
            assert h is not None, 'the final-norm hook did not fire -- head would train on nothing'
            hv, av = pooled(h, hm), pooled(h, anm)
            pred = head(torch.cat([hv, av, av - hv], -1).float()).squeeze(-1)
            m = (~torch.isnan(y)) & (anm.sum(1) > 0) & (hm.sum(1) > 0)   # need BOTH molecule spans
            reg = reg_fn(pred[m], y[m]) if m.any() else torch.zeros((), device=dev)
            # NaN GUARD -- NOT DEFENSIVE PADDING, THIS FIRED IN PRODUCTION. The first v4 launch
            # reported `lm nan` at step 50: a row whose PROMPT alone exceeds maxlen has every label
            # set to -100, so HF's cross-entropy averages over an empty target set and returns NaN.
            # One such row NaNs the batch mean, and .backward() then writes NaN into every LoRA
            # gradient -- 14 minutes of 100%-GPU training that learned nothing and silently corrupted
            # the adapter. The dataset filter below removes the cause; this is the backstop, because
            # a loss that can silently become NaN must never be allowed to reach the optimizer.
            if not torch.isfinite(lm):
                n_nan += 1
                lm = torch.zeros((), device=dev, requires_grad=True)
                if n_nan <= 5 or n_nan % 500 == 0:
                    print('  WARNING: non-finite lm loss on micro-batch %d (%d so far) -- skipped'
                          % (i, n_nan), flush=True)
            loss = (lm + a.reg_weight * reg) / a.accum
            loss.backward()
            runL += float(lm.detach())
            if m.any():
                runR += float(reg.detach()); nR += 1
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
                opt.step(); sch.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    print('  step %d/%d  lm %.4f  reg(mse) %.4f  n_reg %d'
                          % (step, total, runL / (50 * a.accum), runR / max(nR, 1), nR), flush=True)
                    runL = runR = 0.0; nR = 0
                if a.ckpt_every and step % a.ckpt_every == 0:
                    save(step); print('  [ckpt @ %d]' % step, flush=True)
                if step >= total:
                    break
        if step >= total:
            break
    save(step)
    json.dump({'step': total, 'total': total, 'final_save': True}, open(prog, 'w'))
    print('saved -> %s' % a.out, flush=True)

if __name__ == '__main__':
    main()
