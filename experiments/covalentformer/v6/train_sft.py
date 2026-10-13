#!/usr/bin/env python3
"""LoRA SFT for one cell of the 2x2 pocket-dependence matrix, one seed per invocation.

    python train_sft.py --model smiley --arm pocket --seed 0 --out sft_runs/smiley_pocket_s0

                              pocket-aware prompt   pocket-blind prompt
      SmileyLlama-1.1               run 1                 run 2
      Llama-3.1-8B-Instruct         run 3                 run 4

Rows isolate what the ChEMBL SFT+DPO pretraining buys. Columns isolate whether the pocket
contributes anything at all, which is the primary question and the only one that can invalidate the
whole experiment: if the blind column matches the pocket column, the contact table carried no signal
and nothing has been demonstrated. Every hyperparameter below is shared by all four cells, which is
why they live in one `CFG` dict that is dumped verbatim into each run directory -- an ablation whose
halves were trained under hyperparameters that drifted apart proves nothing about the thing ablated.

Both checkpoints are ordinary merged full weights. SmileyLlama 1.1 is NOT an adapter, so this is
plain LoRA on a checkpoint, never PEFT-on-PEFT.

ALPACA FORMAT FOR ALL FOUR CELLS -- see sft_data.py. Not per-model chat templates: that would
confound template with checkpoint and make the row comparison unreadable.

WHY THE LOSS IS MASKED, AND WHY BY HAND
---------------------------------------
Loss is taken on the response only. The prompt is a long contact table -- measured with this
tokenizer the pocket arm's prompt runs a median of ~960 tokens against ~100 for the answer -- so an
unmasked objective would spend ~90% of its gradient teaching the model to regenerate a table it will
always be given, and the sentence we actually want would be a rounding error.

trl 1.14.2 IS installed, so the hand-written mask is a choice, not a workaround. It tokenises the
prompt, tokenises the completion with `add_special_tokens=False`, concatenates, appends the pinned
EOS, and masks the prompt span to -100. Because the full sequence is BUILT from the prompt ids
rather than re-tokenised from concatenated text, the prompt is a prefix by construction -- there is
no tokeniser-boundary drift to assert around, which is the failure mode a response-template string
match is exposed to. `n_label_tokens` is logged per run so the mask is checkable from outside.

THREE CHECKPOINT DIFFERENCES THAT ARE PINNED HERE, NOT INHERITED
----------------------------------------------------------------
The two checkpoints share a vocabulary exactly (vocab_size 128256, identical vocab dicts, identical
ids for the same SMILES), so one data pipeline and one tokenizer setup is safe. But three config
fields differ, and leaving any of them to its default would make the rows incomparable:

1. `pad_token` is None on base Llama and `<|finetune_right_pad_id|>` (128004) on SmileyLlama. It is
   pinned to 128004 in BOTH arms -- the id exists in both vocabularies. Letting it fall back to
   eos-as-pad would mask padding as end-of-sequence and corrupt the loss in a way that is very hard
   to see from the curves.
2. `eos_token_id` is the list [128001, 128008, 128009] on base and the scalar 128009 on SmileyLlama.
   Pinned to 128009 in both, so training appends the same terminator and generation stops the same
   way; otherwise the generation metrics would be comparing stopping behaviour, not models.
3. `rope_scaling` factor is 8.0 on base and 1.0 (identity) on SmileyLlama, so SmileyLlama's usable
   context is 8192 whatever its config advertises. Nothing here approaches that -- the longest
   sequence in either arm is ~2.1k tokens -- but it means the two checkpoints are not drop-in
   identical, and the row comparison is honestly "chemistry-tuned checkpoint as published" vs
   "plain instruct", not a perfectly controlled single-variable contrast. It belongs in the caveats.

EARLY STOPPING ON VAL, AND WHAT THE RESULTING NUMBER IS
-------------------------------------------------------
Up to 10 epochs, early stop on val loss with patience 2, best checkpoint kept. On 731 training
examples expect it to stop at 1-3 epochs; the stopping epoch is reported per arm.

Because the stopping epoch and the checkpoint are both selected on val, every val number from this
run is a MODEL-SELECTION SCORE, NOT A HELD-OUT TEST SCORE, and is optimistically biased by an
unknown amount. This probe is deliberately two-way (train/val only); the clean train/val/test
discipline begins with the 7k run. Selecting a stopping epoch on the very metric one then reports
would bias all four cells by amounts that need not be equal, which is why test is not used here.

PREEMPTION
----------
The VM is preemptible. `save_strategy="epoch"` plus `--resume auto` restarts from the last completed
epoch rather than from scratch, and `run_state.json` records the epochs that finished, so a resumed
run reports where it stopped instead of quietly appearing to have run straight through.
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_data import load_rows_jsonl  # noqa: E402

# Local snapshots. Both were fetched by the gpu-setup agent; nothing here downloads anything.
# The base arm uses the UNGATED MIRROR NousResearch/Meta-Llama-3.1-8B-Instruct, because
# meta-llama/Llama-3.1-8B-Instruct is gated and the box has no HF token. Shard sizes are
# byte-identical to the official repo. The SmileyLlama repo shipped no safetensors index; gpu-setup
# synthesized one into that snapshot, so a missing-index error there means the snapshot was
# replaced, NOT that anything needs re-downloading.
HUB = os.path.expanduser('~/.cache/huggingface/hub')
MODELS = {
    'smiley': os.path.join(HUB, 'models--THGLab--Llama-3.1-8B-SmileyLlama-1.1',
                           'snapshots', '6179dd73b34f055e76ba09d6748baf6a80dcec55'),
    'base': os.path.join(HUB, 'models--NousResearch--Meta-Llama-3.1-8B-Instruct',
                         'snapshots', 'd10aef7999a2b5ba950ab3974312feeedbfe0b77'),
}

PAD_ID = 128004      # <|finetune_right_pad_id|>, present in both vocabularies
EOS_ID = 128009      # <|eot_id|>, pinned identically in both arms

CFG = dict(
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lr=1e-4,
    lr_scheduler='cosine',
    warmup_ratio=0.03,
    max_epochs=10,
    early_stopping_patience=2,
    per_device_batch=1,
    grad_accum=8,
    # Re-measured on the 1,714-usable g1 set rather than inherited: pocket full-sequence p95/p99/max
    # = 1500/1652/2128, blind max 503. 2304 clears the longest sequence with ~8% margin and stays
    # far under SmileyLlama's effective 8192 (its rope_scaling factor is 1.0, the identity).
    # This is a REFUSAL THRESHOLD, not a padding target -- the collator pads to each batch's own
    # longest sequence -- so a tighter value costs nothing and AlpacaSFT raises rather than
    # truncating, because truncating cuts into the contact table and would delete pocket evidence
    # from exactly the arm the ablation is meant to favour.
    max_length=2304,
    bf16=True,
    gradient_checkpointing=True,
)


class AlpacaSFT(Dataset):
    """Alpaca rows -> (input_ids, labels) with the prompt span masked to -100."""

    def __init__(self, rows, tok, max_length):
        self.ex = []
        self.n_label_tokens = 0
        for r in rows:
            p_ids = list(tok(r['prompt'], add_special_tokens=True)['input_ids'])
            c_ids = list(tok(r['completion'], add_special_tokens=False)['input_ids']) + [EOS_ID]
            ids = p_ids + c_ids
            if len(ids) > max_length:
                # Loud, not silent: truncating cuts into the contact table and would delete pocket
                # evidence from exactly the arm this ablation is meant to favour.
                raise ValueError(
                    'pair %s is %d tokens, over max_length=%d. Raise max_length; do not truncate.'
                    % (r['pair'], len(ids), max_length))
            self.n_label_tokens += len(c_ids)
            self.ex.append({'input_ids': ids,
                            'labels': [-100] * len(p_ids) + c_ids,
                            'pair': r['pair']})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        return {'input_ids': e['input_ids'], 'labels': e['labels']}


def collate(batch, pad_id=PAD_ID):
    n = max(len(b['input_ids']) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b['input_ids'])
        ids.append(b['input_ids'] + [pad_id] * k)
        lab.append(b['labels'] + [-100] * k)
        att.append([1] * len(b['input_ids']) + [0] * k)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(lab),
            'attention_mask': torch.tensor(att)}


def build_tokenizer(model_path):
    """One tokenizer setup for all four arms, with pad and eos pinned rather than inherited."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.convert_ids_to_tokens(PAD_ID)
    tok.pad_token_id = PAD_ID
    tok.eos_token = tok.convert_ids_to_tokens(EOS_ID)
    tok.eos_token_id = EOS_ID
    assert tok.pad_token_id != tok.eos_token_id, 'pad must not alias eos'
    return tok


def load_model(model_path, bf16=True, device_map=None):
    from transformers import AutoModelForCausalLM
    kw = dict(attn_implementation='sdpa')
    if device_map:
        kw['device_map'] = device_map
    dt = torch.bfloat16 if bf16 else None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt, **kw)
    # Pin padding/termination identically across checkpoints.
    model.config.pad_token_id = PAD_ID
    model.config.eos_token_id = EOS_ID
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.pad_token_id = PAD_ID
        model.generation_config.eos_token_id = EOS_ID
    return model


def usable_checkpoints(out_dir):
    """Return (usable, discarded) checkpoint dirs, quarantining any that are corrupt.

    A preemption can kill the process midway through writing a checkpoint. That is not a
    hypothetical: the 2026-10-12 preemption landed inside the write of `checkpoint-92`, leaving a
    truncated `adapter_model.safetensors`, and because `--resume auto` trusted any directory whose
    name began with `checkpoint-`, the resume died with

        SafetensorError: Error while deserializing header: incomplete metadata

    94 seconds in -- and would have died identically on every future attempt, so the cell could
    never make progress again. A half-written checkpoint is worse than no checkpoint: it silently
    converts a recoverable interruption into a permanently stuck run.

    Every shard header is therefore opened before the checkpoint is trusted. Reading the header is
    cheap (no tensor data is materialised) and it is exactly the step that failed. A bad checkpoint
    is renamed to `<name>.corrupt` rather than deleted, so the evidence survives for inspection,
    and is reported in train_config.json instead of vanishing.
    """
    if not os.path.isdir(out_dir):
        return [], []
    good, bad = [], []
    for d in sorted(os.listdir(out_dir)):
        if not d.startswith('checkpoint-'):
            continue
        path = os.path.join(out_dir, d)
        shards = [f for f in os.listdir(path) if f.endswith('.safetensors')]
        if not shards:
            bad.append(d)
        else:
            try:
                from safetensors import safe_open
                for f in shards:
                    with safe_open(os.path.join(path, f), framework='pt') as fh:
                        fh.keys()
                good.append(d)
            except Exception as e:
                print('[checkpoint] discarding %s as unreadable: %s' % (d, e))
                bad.append(d)
    for d in bad:
        os.rename(os.path.join(out_dir, d), os.path.join(out_dir, d + '.corrupt'))
    return good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--arm', choices=['pocket', 'blind'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default='sft_out')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', default='auto', choices=['auto', 'no'])
    a = ap.parse_args()

    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model_path = MODELS[a.model]
    path = os.path.join(a.data_dir, 'sft_%s.jsonl' % a.arm)
    tr_rows, va_rows = load_rows_jsonl(path, a.arm)

    tok = build_tokenizer(model_path)
    train_ds = AlpacaSFT(tr_rows, tok, CFG['max_length'])
    val_ds = AlpacaSFT(va_rows, tok, CFG['max_length'])
    tag = '%s/%s/s%d' % (a.model, a.arm, a.seed)
    print('[%s] train %d  val %d  label tokens train=%d val=%d  (0 truncated by construction)'
          % (tag, len(train_ds), len(val_ds), train_ds.n_label_tokens, val_ds.n_label_tokens))

    model = load_model(model_path, CFG['bf16'])
    model.config.use_cache = False
    if CFG['gradient_checkpointing']:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'], lora_dropout=CFG['lora_dropout'],
        bias='none', task_type='CAUSAL_LM', target_modules=CFG['target_modules']))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print('[%s] trainable %d / %d (%.3f%%)' % (tag, trainable, total, 100 * trainable / total))

    targs = dict(
        output_dir=a.out,
        num_train_epochs=CFG['max_epochs'],
        per_device_train_batch_size=CFG['per_device_batch'],
        per_device_eval_batch_size=CFG['per_device_batch'],
        gradient_accumulation_steps=CFG['grad_accum'],
        learning_rate=CFG['lr'],
        lr_scheduler_type=CFG['lr_scheduler'],
        warmup_ratio=CFG['warmup_ratio'],
        bf16=CFG['bf16'],
        logging_steps=20,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=[],
        gradient_checkpointing=CFG['gradient_checkpointing'],
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
    )
    try:
        args = TrainingArguments(eval_strategy='epoch', **targs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy='epoch', **targs)

    tr_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                     data_collator=collate,
                     callbacks=[EarlyStoppingCallback(
                         early_stopping_patience=CFG['early_stopping_patience'])])
    try:
        trainer = Trainer(processing_class=tok, **tr_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **tr_kwargs)

    ckpts, bad_ckpts = usable_checkpoints(a.out)
    resume = bool(ckpts) and a.resume == 'auto'
    json.dump({'model': a.model, 'model_path': model_path, 'arm': a.arm, 'seed': a.seed,
               'cfg': CFG, 'pad_token_id': PAD_ID, 'eos_token_id': EOS_ID, 'data': path,
               'n_train': len(train_ds), 'n_val': len(val_ds),
               'n_label_tokens_train': train_ds.n_label_tokens,
               'n_label_tokens_val': val_ds.n_label_tokens, 'n_truncated': 0,
               'trainable_params': trainable, 'total_params': total,
               'resumed_from_checkpoint': resume, 'checkpoints_found': sorted(ckpts),
               'checkpoints_discarded_as_corrupt': bad_ckpts,
               'val_is_model_selection_not_test': True},
              open(os.path.join(a.out, 'train_config.json'), 'w'), indent=2)

    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=resume or None)
    dt = time.time() - t0
    final_eval = trainer.evaluate()

    # Which epoch actually won, and what every epoch scored. The stopping epoch is a REPORTED
    # quantity, not an implementation detail, precisely because it was selected on val.
    # Coerce to float before comparing. transformers 5.x renders these metrics as STRINGS in its
    # log output, and `min()` over strings is a lexicographic compare: '10.5' sorts below '9.8', so
    # a run whose loss crossed a digit boundary would report the wrong stopping epoch -- and the
    # stopping epoch is a reported quantity here, not a detail, because it was selected on val.
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    per_epoch = [{'epoch': _f(h.get('epoch')), 'eval_loss': _f(h.get('eval_loss'))}
                 for h in trainer.state.log_history if 'eval_loss' in h]
    per_epoch = [h for h in per_epoch if h['eval_loss'] is not None]
    best = min(per_epoch, key=lambda h: h['eval_loss']) if per_epoch else None

    model.save_pretrained(os.path.join(a.out, 'adapter'))
    tok.save_pretrained(os.path.join(a.out, 'adapter'))
    json.dump({'model': a.model, 'arm': a.arm, 'seed': a.seed,
               'train_runtime_s': round(dt, 1),
               'train_loss': out.metrics.get('train_loss'),
               'best_val_loss': best['eval_loss'] if best else None,
               'stopping_epoch': best['epoch'] if best else None,
               'epochs_ran': trainer.state.epoch,
               'early_stopped': (trainer.state.epoch or 0) < CFG['max_epochs'],
               'per_epoch_val_loss': per_epoch,
               'final_eval': final_eval,
               'global_step': trainer.state.global_step,
               'resumed_from_checkpoint': resume,
               'val_is_model_selection_not_test': True},
              open(os.path.join(a.out, 'run_state.json'), 'w'), indent=2)
    print('[%s] done in %.1f min  best_val_loss=%s at epoch %s  (ran %.2f epochs)'
          % (tag, dt / 60, best and round(best['eval_loss'], 4), best and best['epoch'],
             trainer.state.epoch or 0))


if __name__ == '__main__':
    main()
