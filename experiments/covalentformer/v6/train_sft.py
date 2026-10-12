#!/usr/bin/env python3
"""LoRA SFT of SmileyLlama on the chair-adjudicated pairs, one arm per invocation.

    python train_sft.py --arm pocket --out sft_runs/pocket
    python train_sft.py --arm blind  --out sft_runs/blind

Both arms read the JSONL that `sft_data.py` emitted, so they differ only in the user turn. Every
other knob is fixed in ONE place (`CFG` below) and dumped verbatim into the run directory as
`train_config.json`, because an ablation whose two halves were trained under hyperparameters that
drifted apart proves nothing about the thing being ablated.

WHY THE LOSS IS MASKED, AND WHY BY HAND
---------------------------------------
Loss is taken on the assistant turn only. The prompt is a long contact table: measured with this
tokenizer, the pocket arm's prompt runs a median of 960 tokens against 101 for the answer, so an
unmasked objective would spend ~91% of its gradient teaching the model to regenerate a table it will
always be given, and the sentence we actually want would be a rounding error in the loss.

`max_len` is 2304, above the longest full sequence in either arm (2148 tokens), so NOTHING is
truncated -- confirmed by `n_truncated: 0` in both run configs. That matters more than it looks:
truncation here cuts from the left, i.e. into the contact table, which would quietly delete pocket
evidence from the longest-pocket pairs and weaken exactly the arm the ablation is meant to favour.

The masking is done here rather than with trl's `DataCollatorForCompletionOnlyLM`. trl 1.14.2 IS
installed on the box, so this is a choice, not a workaround: the explicit version tokenises the
prompt with `add_generation_prompt=True`, tokenises prompt+answer, and sets the first `len(prompt)`
labels to -100. That is what the trl collator does, in six lines, with no coupling to a trl release
and no string-matching on a response template -- and critically it ASSERTS the prompt is a true
prefix of the full sequence, which catches a chat-template change silently shifting the boundary.
`n_label_tokens` is logged per run so the mask can be verified from the outside rather than trusted,
and it is the one number that would expose a mask that had quietly inverted or slipped.

SCALE
-----
731 training examples. Three epochs at an effective batch of 8 is 276 steps. This run is a pipeline
proof and the baseline half of an ablation, not a result: an 8B model with a 42M-parameter adapter
will overfit 731 examples, and the held-out numbers are there to be compared between the two arms,
not celebrated. Nothing here is tuned, deliberately -- tuning one arm harder than the other is the
easiest way to fake pocket dependence.

PREEMPTION
----------
The VM is preemptible. `save_strategy="epoch"` plus `--resume auto` means a killed run restarts from
the last completed epoch instead of from scratch; `run_state.json` records which epochs finished, so
a resumed run reports where it stopped rather than quietly pretending it ran straight through.
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_data import chat_ids  # noqa: E402  (one tokenisation path, shared with the data builder)

CFG = dict(
    model_id='THGLab/Llama-3.1-8B-SmileyLlama-1.1',
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lr=1e-4,
    lr_scheduler='cosine',
    warmup_ratio=0.03,
    epochs=3,
    per_device_batch=1,
    grad_accum=8,
    max_len=2304,
    bf16=True,
    gradient_checkpointing=True,
    seed=0,
)


class ChatSFT(Dataset):
    """Chat rows -> (input_ids, labels) with the prompt masked to -100."""

    def __init__(self, rows, tok, max_len):
        self.ex = []
        self.n_truncated = 0
        self.n_label_tokens = 0
        for r in rows:
            m = r['messages']
            prompt_ids = chat_ids(tok, m[:2], add_generation_prompt=True)
            full_ids = chat_ids(tok, m)
            # A chat-template change that stopped the prompt being a prefix of the full sequence
            # would silently shift the mask and train on the wrong tokens.
            assert full_ids[:len(prompt_ids)] == prompt_ids, \
                'prompt is not a prefix of the full sequence for %s' % r['pair']
            if len(full_ids) > max_len:
                self.n_truncated += 1
                # Truncate from the LEFT, inside the table: the answer must survive intact or the
                # example teaches a truncated sentence. The prompt boundary moves with it.
                cut = len(full_ids) - max_len
                full_ids = full_ids[:1] + full_ids[1 + cut:]
                prompt_len = max(1, len(prompt_ids) - cut)
            else:
                prompt_len = len(prompt_ids)
            labels = list(full_ids)
            labels[:prompt_len] = [-100] * prompt_len
            self.n_label_tokens += sum(1 for x in labels if x != -100)
            self.ex.append({'input_ids': full_ids, 'labels': labels, 'pair': r['pair']})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        return {'input_ids': e['input_ids'], 'labels': e['labels']}


def collate(batch, pad_id):
    n = max(len(b['input_ids']) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b['input_ids'])
        ids.append(b['input_ids'] + [pad_id] * k)
        lab.append(b['labels'] + [-100] * k)
        att.append([1] * len(b['input_ids']) + [0] * k)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(lab),
            'attention_mask': torch.tensor(att)}


def load_rows(path, arm):
    rows = [json.loads(l) for l in open(path)]
    assert all(r['arm'] == arm for r in rows), 'arm mismatch in %s' % path
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    ex = [r for r in rows if r['split'] == 'excluded']
    assert ex and all(r['verdict'] == 'REJECT' for r in ex)
    assert all(r['messages'][2]['content'] for r in tr + va), 'empty assistant turn in train/val'
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['pocket', 'blind'], required=True)
    # The second base checkpoint is the other half of the 2x2: chemistry-tuned start vs plain
    # instruct start. `NousResearch/...` rather than `meta-llama/...` because the Meta repo is gated
    # ("gated": "manual") and the box has no token; the mirror's shards are byte-identical and the
    # vocabularies were verified equal (128256, 256 added tokens, same embedding shapes), so one
    # data pipeline serves both arms. Pad is forced to 128004 below because the base ships none.
    ap.add_argument('--model', default=CFG['model_id'],
                    help='base checkpoint; the Llama arm uses '
                         'NousResearch/Meta-Llama-3.1-8B-Instruct')
    ap.add_argument('--seed', type=int, default=CFG['seed'])
    ap.add_argument('--data-dir', default='sft_out')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', default='auto', choices=['auto', 'no'])
    ap.add_argument('--max-len', type=int, default=CFG['max_len'])
    a = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    from peft import LoraConfig, get_peft_model

    CFG['model_id'] = a.model
    CFG['seed'] = a.seed
    set_seed(CFG['seed'])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.data_dir, 'sft_%s.jsonl' % a.arm)
    tr_rows, va_rows = load_rows(path, a.arm)

    tok = AutoTokenizer.from_pretrained(CFG['model_id'])
    if tok.pad_token is None:
        # Base Llama-3.1-8B-Instruct ships no pad token; SmileyLlama ships <|finetune_right_pad_id|>
        # (128004). Pin both arms to 128004 so they are identical here rather than one using eos.
        # It makes no numerical difference as things stand -- `collate` pads by POSITION, filling
        # labels with -100 and attention_mask with 0, so a pad id colliding with eos is never read
        # as end-of-sequence, and per_device_batch=1 means padding hardly occurs. It is pinned
        # anyway because that reasoning is a property of the current collator, not of the config,
        # and the next person to raise the batch size should not have to rediscover it.
        tok.pad_token = tok.convert_ids_to_tokens(128004)
        assert tok.pad_token_id == 128004, tok.pad_token_id

    train_ds = ChatSFT(tr_rows, tok, a.max_len)
    val_ds = ChatSFT(va_rows, tok, a.max_len)
    print('[%s] train %d  val %d  truncated %d/%d  label tokens train=%d val=%d'
          % (a.arm, len(train_ds), len(val_ds), train_ds.n_truncated + val_ds.n_truncated,
             len(train_ds) + len(val_ds), train_ds.n_label_tokens, val_ds.n_label_tokens))

    # transformers renamed torch_dtype -> dtype in 5.x; support both so this runs either side.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CFG['model_id'], dtype=torch.bfloat16, attn_implementation='sdpa')
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            CFG['model_id'], torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    if CFG['gradient_checkpointing']:
        model.enable_input_require_grads()

    peft_cfg = LoraConfig(r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'],
                          lora_dropout=CFG['lora_dropout'], bias='none',
                          task_type='CAUSAL_LM', target_modules=CFG['target_modules'])
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print('[%s] trainable %d / %d (%.3f%%)' % (a.arm, trainable, total, 100 * trainable / total))

    targs = dict(
        output_dir=a.out,
        num_train_epochs=CFG['epochs'],
        per_device_train_batch_size=CFG['per_device_batch'],
        per_device_eval_batch_size=CFG['per_device_batch'],
        gradient_accumulation_steps=CFG['grad_accum'],
        learning_rate=CFG['lr'],
        lr_scheduler_type=CFG['lr_scheduler'],
        warmup_ratio=CFG['warmup_ratio'],
        bf16=CFG['bf16'],
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=CFG['epochs'],
        report_to=[],
        gradient_checkpointing=CFG['gradient_checkpointing'],
        seed=CFG['seed'],
        remove_unused_columns=False,
    )
    try:
        args = TrainingArguments(eval_strategy='epoch', **targs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy='epoch', **targs)

    tr_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                     data_collator=lambda b: collate(b, tok.pad_token_id))
    try:
        trainer = Trainer(processing_class=tok, **tr_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **tr_kwargs)

    ckpts = [d for d in os.listdir(a.out) if d.startswith('checkpoint-')] if os.path.isdir(a.out) else []
    resume = bool(ckpts) and a.resume == 'auto'
    json.dump({'arm': a.arm, 'cfg': CFG, 'max_len': a.max_len, 'data': path,
               'n_train': len(train_ds), 'n_val': len(val_ds),
               'n_label_tokens_train': train_ds.n_label_tokens,
               'n_label_tokens_val': val_ds.n_label_tokens,
               'n_truncated': train_ds.n_truncated + val_ds.n_truncated,
               'trainable_params': trainable, 'total_params': total,
               'resumed_from_checkpoint': resume, 'checkpoints_found': sorted(ckpts)},
              open(os.path.join(a.out, 'train_config.json'), 'w'), indent=2)

    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=resume or None)
    dt = time.time() - t0

    final_eval = trainer.evaluate()
    model.save_pretrained(os.path.join(a.out, 'adapter'))
    tok.save_pretrained(os.path.join(a.out, 'adapter'))
    hist = [h for h in trainer.state.log_history]
    json.dump({'arm': a.arm, 'train_runtime_s': round(dt, 1),
               'train_loss': out.metrics.get('train_loss'),
               'final_eval': final_eval, 'log_history': hist,
               'epochs_completed': trainer.state.epoch,
               'global_step': trainer.state.global_step,
               'resumed_from_checkpoint': resume},
              open(os.path.join(a.out, 'run_state.json'), 'w'), indent=2)
    print('[%s] done in %.1f min  final_eval=%s' % (a.arm, dt / 60, final_eval))


if __name__ == '__main__':
    main()
