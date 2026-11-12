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
    # The arm every section-3 ChemDFM row was trained from. It was passed in ad hoc for those runs
    # and so was never written down here, which meant the sarlitA arm could not be trained with one
    # command on a fresh machine. Pinned to the snapshot, like the other two.
    'chemdfm': os.path.join(HUB, 'models--OpenDFM--ChemDFM-v2.0-14B',
                            'snapshots', 'b3b1d2143bb3e307e6b6dae794f6e9b0a83a45ef'),
    # DESRO, the only strong competitor in this family whose weights are both released and
    # commercially usable (DrugR's are not actually released; GeLLM3O is CC BY-NC). It is NOT in
    # the HF hub cache -- it arrived as an archive and lives wherever it was extracted -- so the
    # path is an env var with the box-A default rather than a pinned snapshot hash. Keeping the
    # default here instead of requiring the variable means a run on a differently laid-out machine
    # fails on a missing directory, which is loud, rather than silently picking another model.
    #
    # Config read from the extracted checkpoint: LlamaForCausalLM, 32 layers, hidden 4096,
    # vocab_size 130304, max_position_embeddings 8192 -- an 8B Llama. Its special ids are NOT the
    # Llama-3.1 ones; see the SPECIAL_IDS entry, which was resolved from its tokenizer.
    # `check_special_ids` re-verifies at load and stops the run on a mismatch.
    #
    # The honest caveat for the paper: DESRO is 8B and ChemDFM-v2.0 is 14B, so "our adapter on
    # DESRO vs our adapter on ChemDFM" is confounded with base size. The clean comparison is each
    # base against ITSELF with and without our adapter.
    'desro': os.environ.get('DESRO_PATH', '/mnt/aigpu/desro_extract/checkpoint'),
}

# Llama-3 defaults. These are correct for smiley and base and WRONG for ChemDFM, which is built on
# Qwen: its vocabulary is 151643 tokens and id 128009 is not <|eot_id|> there but the ordinary token
# 'ナル'. Because an ordinary token is not special, `skip_special_tokens=True` cannot strip it, so
# every ChemDFM generation ended `SMILES: <smiles>ナル`, the SMILES failed to parse, and the eval
# reported format_compliance 0.000 for a model that was in fact emitting a well-formed SMILES line
# on all 262 rows. Re-scoring with the token removed gave validity 0.966. Resolve per model instead
# of pinning one vocabulary's ids and asserting they hold for every arm.
SPECIAL_IDS = {
    'smiley':  (128004, 128009),    # <|finetune_right_pad_id|>, <|eot_id|>
    'base':    (128004, 128009),
    'chemdfm': (151643, 151645),    # <|endoftext|>, <|im_end|>
    # DESRO. Verified against its own tokenizer rather than assumed from the Llama family, and the
    # assumption would have been wrong twice over:
    #   * its EOS is <|end_of_text|> = 128001, NOT <|eot_id|> = 128009. Its generation_config
    #     agrees (eos_token_id 128001). Pinning 128009 would have trained and generated to
    #     completion while the model never learned to stop where we score it -- the same silent
    #     class of failure as the ChemDFM 'ナル' corruption described above.
    #   * it defines NO pad token, and <|finetune_right_pad_id|> does not exist in its vocabulary
    #     (its specials are Llama-3.0-style). `convert_ids_to_tokens(128004)` there returns
    #     <|reserved_special_token_2|>, a real token the model never emits, which is what a pad
    #     must be. EOS must NOT be reused as pad here: pads are masked out of the labels, so
    #     pad == eos would mask the genuine end-of-sequence and the model would not learn to stop.
    # Note also that the checkpoint's config declares vocab_size 130304 while the tokenizer holds
    # 128256 -- the embedding matrix is padded past the vocabulary. Harmless as long as nothing
    # emits an id at or above 128256, which nothing can, since the tokenizer cannot produce one.
    'desro':   (128004, 128001),    # <|reserved_special_token_2|>, <|end_of_text|>
}

PAD_ID, EOS_ID = SPECIAL_IDS['smiley']


def set_special_ids(model_name):
    """Point PAD_ID/EOS_ID at the vocabulary of the model actually being run."""
    global PAD_ID, EOS_ID
    PAD_ID, EOS_ID = SPECIAL_IDS[model_name]
    return PAD_ID, EOS_ID


def check_special_ids(tok, model_name):
    """Fail loudly if the pinned ids are not the tokenizer's own specials.

    The bug this guards against was silent: training and generation both ran to completion and
    only the scored output was wrong. A mismatch must therefore stop the run, not warn.
    """
    want_pad, want_eos = SPECIAL_IDS[model_name]
    for what, pinned, actual in (('eos', want_eos, tok.eos_token_id),
                                 ('pad', want_pad, tok.pad_token_id)):
        if actual is not None and pinned != actual:
            raise ValueError(
                '%s id %d pinned for %r but its tokenizer reports %d (%r). Pinning one '
                'vocabulary\'s ids across models is what produced the ChemDFM "ナル" corruption.'
                % (what, pinned, model_name, actual, tok.convert_ids_to_tokens(actual)))

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


def collate(batch, pad_id=None):
    # Read PAD_ID at call time. A default argument would bind the Llama value at import, before
    # set_special_ids() has had a chance to switch vocabularies.
    pad_id = PAD_ID if pad_id is None else pad_id
    n = max(len(b['input_ids']) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b['input_ids'])
        ids.append(b['input_ids'] + [pad_id] * k)
        lab.append(b['labels'] + [-100] * k)
        att.append([1] * len(b['input_ids']) + [0] * k)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(lab),
            'attention_mask': torch.tensor(att)}


def build_tokenizer(model_path, model_name=None):
    """One tokenizer setup for all four arms, with pad and eos pinned rather than inherited.

    `model_name` is optional only so existing callers keep working; pass it, because it is what
    lets the pinned ids be checked against the vocabulary they are being applied to.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if model_name is not None:
        # Before the pins overwrite them -- afterwards the check compares the pins with themselves.
        check_special_ids(tok, model_name)
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
        # trainer_state.json is as load-bearing as the weights: Trainer reads global_step from
        # it, so a checkpoint without one cannot be resumed from at all. A full disk produces
        # exactly that shape -- B filled to 340G/340G at step 1995 and the write died AFTER the
        # adapter and BEFORE trainer_state.json, leaving a directory whose safetensors header
        # opens cleanly and which would therefore have been trusted, then failed on resume.
        if not shards or not os.path.isfile(os.path.join(path, 'trainer_state.json')):
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
    ap.add_argument('--epochs', type=int, default=None,
                    help='override CFG max_epochs. With --select keep_all this bounds the run '
                         'directly, since nothing stops it early: at 8,348 rows an epoch is ~2h, '
                         'so 10 epochs is a 21-hour run to produce ten checkpoints of which the '
                         'last few are deep into memorisation (train_loss hit 0.278 by epoch 10 '
                         'on the 656-row arm).')
    ap.add_argument('--select', default='val_loss', choices=['val_loss', 'keep_all'],
                    help='val_loss = early-stop on eval_loss and keep the best checkpoint (the '
                         'default every published arm was trained with). keep_all = run all '
                         'epochs, keep every epoch checkpoint, select nothing. Use keep_all on '
                         'low-data arms: on sarlitA, eval_loss was LOWEST at epoch 1 (1.909) and '
                         'that checkpoint produced one collapsed template with 45%% of its claims '
                         'correct, while epoch 3 had a worse eval_loss (2.128) and 77%% correct. '
                         'Early stopping with patience 2 therefore selected the worst checkpoint '
                         'the run produced, and the number it selected on never showed it.')
    # The 2304 ceiling in CFG was MEASURED on the Llama vocabulary that smiley and base share.
    # ChemDFM-v2.0-14B is a different tokenizer, so the same prompts come out longer -- the g2
    # benchmark's longest val pair is 2412 ChemDFM tokens against 2304 Llama ones. That is a
    # property of the tokeniser, not a hyperparameter being drifted, and the alternative the code
    # already refuses is truncating the contact table. Overridden per model, never silently.
    ap.add_argument('--max-length', dest='max_length', type=int, default=None,
                    help='override CFG max_length (required for a non-Llama vocabulary)')
    ap.add_argument('--patience', type=int, default=None,
                    help='early-stopping patience in EVALUATIONS, not epochs. With the default '
                         'per-epoch cadence the two are the same thing; with --eval-every 0.25 a '
                         'patience of 4 means one epoch of no improvement. Counted in evaluations '
                         'because that is what EarlyStoppingCallback counts, and reading it as '
                         'epochs is how a run gets stopped four times sooner than intended.')
    ap.add_argument('--val-limit', dest='val_limit', type=int, default=0,
                    help='evaluate on at most this many val rows (0 = all). MODEL SELECTION '
                         'ONLY: it makes a frequent eval affordable so a run on a machine that '
                         'is being preempted every ~30 minutes can checkpoint often enough to '
                         'make progress at all. A subsampled val loss is NOT comparable with '
                         'another arm scored on the full split, and run_state records the '
                         'number used so that cannot be forgotten.')
    ap.add_argument('--eval-every', dest='eval_every', type=float, default=1.0,
                    help='evaluate and checkpoint every N epochs; 0.25 gives four readings per '
                         'epoch. Below 1.0 the trainer switches to step-based strategies, with '
                         'save_steps == eval_steps because load_best_model_at_end requires the '
                         'save points to be a superset of the eval points.')
    a = ap.parse_args()

    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model_path = MODELS[a.model]
    set_special_ids(a.model)      # before anything tokenises or pads
    path = os.path.join(a.data_dir, 'sft_%s.jsonl' % a.arm)
    tr_rows, va_rows = load_rows_jsonl(path, a.arm)

    tok = build_tokenizer(model_path, a.model)
    max_len = a.max_length or CFG['max_length']
    if max_len != CFG['max_length']:
        print('max_length overridden: %d (CFG default %d)' % (max_len, CFG['max_length']))
        CFG['max_length'] = max_len          # so the dumped CFG records what actually ran
    train_ds = AlpacaSFT(tr_rows, tok, max_len)
    if a.val_limit and len(va_rows) > a.val_limit:
        # deterministic stride, not the head: val is ordered by target, so the first N rows are
        # a handful of targets and the loss would track those rather than the split
        step = len(va_rows) / float(a.val_limit)
        va_rows = [va_rows[int(i * step)] for i in range(a.val_limit)]
        # `tag` is built later in main(); printing it here crashed every launch
        print('val subsampled to %d rows for model selection' % len(va_rows))
    val_ds = AlpacaSFT(va_rows, tok, max_len)
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

    # Evaluation cadence. `eval_every` is in epochs; the trainer counts OPTIMIZER steps, so the
    # conversion goes through grad_accum. A quarter-epoch cadence on 9,453 rows at an effective
    # batch of 8 is every 295 optimizer steps.
    steps_per_epoch = max(1, len(train_ds) // (CFG['per_device_batch'] * CFG['grad_accum']))
    stepwise = a.eval_every < 1.0
    eval_steps = max(1, int(round(a.eval_every * steps_per_epoch))) if stepwise else None
    patience = a.patience if a.patience is not None else CFG['early_stopping_patience']
    if stepwise:
        print('[%s] eval+save every %d optimizer steps (%.2f epoch, %d per epoch); patience %d '
              'evaluations = %.2f epochs'
              % (tag, eval_steps, a.eval_every, steps_per_epoch, patience,
                 patience * a.eval_every))

    targs = dict(
        output_dir=a.out,
        num_train_epochs=a.epochs or CFG['max_epochs'],
        per_device_train_batch_size=CFG['per_device_batch'],
        per_device_eval_batch_size=CFG['per_device_batch'],
        gradient_accumulation_steps=CFG['grad_accum'],
        learning_rate=CFG['lr'],
        lr_scheduler_type=CFG['lr_scheduler'],
        warmup_ratio=CFG['warmup_ratio'],
        bf16=CFG['bf16'],
        logging_steps=20,
        save_strategy='steps' if stepwise else 'epoch',
        # Every checkpoint is kept when the cadence is sub-epoch. Not thrift: this project has one
        # measured case -- the sarlitA arm in the --select docstring above -- where the LOWEST
        # eval_loss checkpoint was the worst checkpoint the run produced, 45% of its claims correct
        # against 77% for a checkpoint with a worse loss. Four readings an epoch make that trap
        # finer-grained, not rarer, so the checkpoints have to survive long enough to be scored by
        # generation rather than by loss. The supervisor strips optimizer state after scoring.
        save_total_limit=None if (stepwise or a.select != 'val_loss') else 2,
        load_best_model_at_end=a.select == 'val_loss',
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=[],
        gradient_checkpointing=CFG['gradient_checkpointing'],
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
    )
    if stepwise:
        targs['save_steps'] = eval_steps
        targs['eval_steps'] = eval_steps
    strategy = 'steps' if stepwise else 'epoch'
    try:
        args = TrainingArguments(eval_strategy=strategy, **targs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy=strategy, **targs)

    tr_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                     data_collator=collate)
    if a.select == 'val_loss':
        tr_kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=patience)]
    try:
        trainer = Trainer(processing_class=tok, **tr_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **tr_kwargs)

    ckpts, bad_ckpts = usable_checkpoints(a.out)
    resume = bool(ckpts) and a.resume == 'auto'
    json.dump({'model': a.model, 'model_path': model_path, 'arm': a.arm, 'seed': a.seed,
               'cfg': CFG, 'eval_every_epochs': a.eval_every, 'eval_steps': eval_steps,
               'early_stopping_patience_evaluations': patience,
               'early_stopping_patience_epochs': patience * a.eval_every,
               'pad_token_id': PAD_ID, 'eos_token_id': EOS_ID, 'data': path,
               'n_train': len(train_ds), 'n_val': len(val_ds),
               'n_label_tokens_train': train_ds.n_label_tokens,
               'n_label_tokens_val': val_ds.n_label_tokens, 'n_truncated': 0,
               'trainable_params': trainable, 'total_params': total,
               'resumed_from_checkpoint': resume, 'checkpoints_found': sorted(ckpts),
               'checkpoints_discarded_as_corrupt': bad_ckpts,
               'val_is_model_selection_not_test': True,
               'val_rows_used': len(va_rows), 'val_limit_flag': a.val_limit},
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
               'select': a.select,
               'max_epochs_requested': a.epochs or CFG['max_epochs'],
               'early_stopped': (trainer.state.epoch or 0) < (a.epochs or CFG['max_epochs']),
               'per_epoch_val_loss': per_epoch,
               'final_eval': final_eval,
               'global_step': trainer.state.global_step,
               'resumed_from_checkpoint': resume,
               'val_is_model_selection_not_test': True,
               'val_rows_used': len(va_rows), 'val_limit_flag': a.val_limit},
              open(os.path.join(a.out, 'run_state.json'), 'w'), indent=2)
    print('[%s] done in %.1f min  best_val_loss=%s at epoch %s  (ran %.2f epochs)'
          % (tag, dt / 60, best and round(best['eval_loss'], 4), best and best['epoch'],
             trainer.state.epoch or 0))


if __name__ == '__main__':
    main()
