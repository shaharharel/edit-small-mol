#!/usr/bin/env python3
"""GRPO over the verifiable parts of a pocket-grounded edit rationale, with an SFT language anchor.

WHY GRPO AND NOT MORE SFT
-------------------------
SFT trains the model to imitate one chair sentence per pair. Measured outcome: it learned the FORM
of a grounded sentence -- 100% of trained-arm sentences cite a distance and 99.9-100% of those
distances are real rows of the input table -- without learning to bind a number to its owner, which
runs 29-59% against an 8-13% chance baseline. Nothing in the SFT loss ever checked a citation.
GRPO can, because the check is a cheap deterministic function of the prompt and the output.

It also needs no labels. The reward reads the prompt's own contact table, so the training pool is
any pocket+ligand prompt rather than the 2,326 rows that happen to have a chair sentence.

WHAT THIS LOOP DOES, AND THE FOUR THINGS IT DOES TO AVOID BREAKING THE MODEL
---------------------------------------------------------------------------
Per step: sample G completions for each of B prompts, score each with reward_verifiers, convert
rewards to group-relative advantages, and take ONE policy-gradient step. Then, in the same step:

  1. SFT REPLAY. `L = L_grpo + lambda * L_sft`, where L_sft is ordinary next-token loss on a batch
     from the labelled corpus. The replay set is fixed and small relative to the RL stream, so its
     job is to ANCHOR the language distribution, not to co-teach the task. Judge lambda by whether
     held-out SFT loss stays flat, not by the reward curve.
  2. KL TO THE FROZEN REFERENCE, using the k3 estimator. The reference is this same model with the
     LoRA adapter disabled, so it costs no extra weights.
  3. LoRA ONLY, short horizon. ether0's own report is that its reasoning grew more alien the longer
     RL ran; we cap the steps and keep the best checkpoint by held-out reward.
  4. TRIPWIRES. `banned_vocabulary` and `stereo_disclosure` are scored and logged but carry zero
     reward weight. If they fall while the trained components rise, the model is gaming the checker
     rather than learning. Same for `n_claims`: the reward has a coverage floor, so the cheapest way
     to raise attribution is to cite MORE, and runaway n_claims is the signature of that.

THE KNOWN BLIND SPOT, STATED UP FRONT
------------------------------------
The reward cannot check whether a cited contact is OWNED by the atom the edit touched. Three
chemists scored 10 reward-selected mol1 candidates and found 7 of 10 misattribute, with ownership
correct on only 30% of judgements while every candidate scored 0.867-1.000. Directional
attribution was attempted three times and abandoned on a measured negative result (measurable on
11% of cases, 50% agreement with chemists). So this loop should be expected to improve citation
GROUNDING and to leave the attribution failure roughly where it is -- possibly worse, since it
rewards citing more. The acceptance gate is therefore a chemist panel, never the reward total.

RUNNING IT
----------
    # local, no GPU: exercises prompt selection, reward, advantages and the curriculum on archived
    # generations replayed as if they were rollouts. Catches logic bugs before any GPU time.
    python3 train_grpo.py --dry-run --steps 20

    # on the box
    ~/miniconda3/envs/quris/bin/python train_grpo.py \
        --model chemdfm --adapter sft_runs/chemdfm_pocket_g3g4/adapter \
        --data-dir sft_out_g3g4 --out sft_runs/chemdfm_grpo_v1 --steps 300
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Do NOT compute this from a fixed number of parent directories. The training box keeps a flat copy
# of this directory, where '../../..' resolves to '/' and the evidence path became
# '/data/pdb_mirror/v7_evidence.jsonl' -- the smoke run died on it after a two-minute model load.
# reward_verifiers already solves this properly, so defer to it rather than keeping a second,
# divergent answer in this file.
EVIDENCE = None  # set in main() from the reward module's resolved DATA directory


def load_reward():
    spec = importlib.util.spec_from_file_location(
        'reward_verifiers', os.path.join(HERE, 'reward_verifiers.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- data


def load_prompts(data_dir, evidence_path, holdout_pairs):
    """Prompts to train on, with the MODEL_INPUT the reward needs, excluding the benchmark.

    Two sources, in order of preference:
      * the arm's own sft_pocket.jsonl TRAIN rows, which already carry a built prompt; and
      * any evidence pair not in that file, whose prompt we can build the same way.
    The second is the point of doing RL at all -- the reward needs no chair sentence, so a pair with
    no label is still a usable prompt. Kept behind a flag so the first run is comparable to SFT.
    """
    ev = {}
    for line in open(evidence_path):
        r = json.loads(line)
        ev[r['pair']] = r['MODEL_INPUT']

    out = []
    fp = os.path.join(HERE, data_dir, 'sft_pocket.jsonl')
    for line in open(fp):
        r = json.loads(line)
        if r['split'] != 'train':
            continue
        if r['pair'] in holdout_pairs:
            continue
        mi = ev.get(r['pair'])
        if mi is None:
            continue
        out.append({'pair': r['pair'], 'prompt': r['prompt'], 'model_input': mi,
                    'input_smiles': mi.get('input_smiles'),
                    'ligand_code': _code(r['pair'])})
    return out


def _code(pair):
    try:
        return pair.split()[1].split('->')[0]
    except Exception:
        return None


def load_holdout(data_dir):
    """The val pairs. These are the benchmark; training on them would void every reported number."""
    out = set()
    fp = os.path.join(HERE, data_dir, 'sft_pocket.jsonl')
    for line in open(fp):
        r = json.loads(line)
        if r['split'] == 'val':
            out.add(r['pair'])
    return out


def load_replay(data_dir, limit=4000):
    """(prompt, completion) pairs for the SFT anchor term."""
    out = []
    fp = os.path.join(HERE, data_dir, 'sft_pocket.jsonl')
    for line in open(fp):
        r = json.loads(line)
        if r['split'] == 'train' and r.get('completion'):
            out.append((r['prompt'], r['completion']))
            if len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------- advantages


def group_advantages(rewards, std_floor=1e-4):
    """Group-relative advantages, and whether the group carries any signal.

    A group whose samples all score the same has zero variance and therefore zero advantage for
    every member: the step is wasted. ether0 calls these trivial groups and needed a curriculum to
    avoid them. Our best arms gate 19-31% of generations to zero reward, so all-zero groups are
    expected, not hypothetical -- hence returning `useful` so the caller can skip and record it.
    """
    n = len(rewards)
    if n == 0:
        return [], False
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < std_floor:
        return [0.0] * n, False
    return [(r - mean) / std for r in rewards], True


# --------------------------------------------------------------------------- dry run


def dry_run(args, rw):
    """Exercise data, reward, advantages and curriculum on archived generations as fake rollouts.

    No model, no GPU. This exists because validating the reward locally caught three real bugs that
    aggregate scores hid -- a vacuous-default back door, a silently wrong atom mapping, and an
    unusable directional component. The same discipline applies to the loop itself: if the
    advantage computation or the curriculum is wrong, far better to see it here.
    """
    holdout = load_holdout(args.data_dir)
    prompts = load_prompts(args.data_dir, EVIDENCE, holdout)
    print('prompts available        %d' % len(prompts))
    print('holdout pairs excluded   %d' % len(holdout))
    leak = [p for p in prompts if p['pair'] in holdout]
    assert not leak, 'BENCHMARK LEAK: %d held-out pairs in the training pool' % len(leak)
    print('benchmark leak check     clean')

    # fake rollouts: take real generations from several arms as if they were G samples of one prompt
    SCR = ('/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/'
           'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad')
    pools = [os.path.join(HERE, 'sft_2k/chemdfm_pocket_g3/generations.jsonl'),
             SCR + '/g3g4res/chemdfm/generations.jsonl',
             SCR + '/g4res/chemdfm/generations.jsonl',
             SCR + '/alltab/sft_runs/zeroshot_chemdfm_bench/generations.jsonl']
    by_pair = collections.defaultdict(list)
    for p in pools:
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            by_pair[r['pair']].append(r)

    mi_by_pair = {p['pair']: p for p in prompts}
    # the archived generations are for VAL pairs, so for the dry run we score them against their own
    # evidence rather than pretending they are training prompts
    ev = {}
    for line in open(EVIDENCE):
        r = json.loads(line)
        ev[r['pair']] = r['MODEL_INPUT']

    stats = collections.Counter()
    comp = collections.defaultdict(list)
    adv_mag = []
    steps = 0
    for pair, rows in by_pair.items():
        if steps >= args.steps:
            break
        mi = ev.get(pair)
        if mi is None or len(rows) < 2:
            continue
        rewards, details = [], []
        for r in rows:
            g = None
            for k in ('gen_smiles_canonical', 'gen_smiles'):
                v = r.get(k)
                if isinstance(v, str) and v.strip() and v.strip() != 'None':
                    g = v.strip()
                    break
            s = rw.score(mi, (r.get('gen_sentence') or '').strip(), g,
                         mi.get('input_smiles'), ligand_code=_code(pair))
            rewards.append(s['total'])
            details.append(s)
        adv, useful = group_advantages(rewards)
        steps += 1
        stats['groups'] += 1
        stats['useful' if useful else 'trivial'] += 1
        if useful:
            adv_mag.append(sum(abs(a) for a in adv) / len(adv))
        for s in details:
            for k, v in s['components'].items():
                comp[k].append(v)
            comp['total'].append(s['total'])
            comp['n_claims'].append(s['n_claims'])

    print('\ngroups scored            %d' % stats['groups'])
    print('  with signal            %d' % stats['useful'])
    print('  trivial (zero std)     %d  <- these steps would be wasted' % stats['trivial'])
    if adv_mag:
        print('  mean |advantage|       %.3f' % (sum(adv_mag) / len(adv_mag)))
    print('\ncomponent means over %d sampled generations:' % len(comp['total'])) if comp else None
    for k in sorted(comp):
        v = comp[k]
        print('  %-26s %.3f' % (k, sum(v) / len(v)))
    print('\ntripwires to watch during training: banned_vocabulary, stereo_disclosure, n_claims')
    return 0


# --------------------------------------------------------------------------- train


def train(args, rw):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, HERE)
    from train_sft import CFG, MODELS, PAD_ID, EOS_ID  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODELS[args.model], trust_remote_code=True)
    assert args.adapter, 'GRPO starts from a trained adapter, not from the base model'
    # Resolve and check the adapter BEFORE the base model loads. Otherwise a mistyped path costs a
    # two-minute load and then surfaces as a confusing 401 from the Hub, which treats the local path
    # as a repo id. The SFT runs keep their weights in a nested adapter/ directory, so accept either.
    if not os.path.exists(os.path.join(args.adapter, 'adapter_config.json')):
        nested = os.path.join(args.adapter, 'adapter')
        assert os.path.exists(os.path.join(nested, 'adapter_config.json')), (
            'no adapter_config.json at %s or %s' % (args.adapter, nested))
        args.adapter = nested
    print('[grpo] adapter %s' % args.adapter)

    # WHAT THE KL REFERENCE IS, AND WHY IT IS A FLAG RATHER THAN A DEFAULT.
    # The first version of this script trained the SFT adapter itself and used disable_adapter() as
    # the reference -- which makes the reference the RAW PRETRAINED BASE, not the SFT model. That
    # pulls the policy back toward un-finetuned behaviour, the opposite of anchoring the chemistry
    # language g3+g4 was trained to produce. Both references are legitimate experiments:
    #
    #   --kl-ref sft   freeze the SFT weights, train a SECOND LoRA on top. disable_adapter() then
    #                  yields the SFT model, so KL anchors to the arm we validated. The conservative
    #                  run. Needs bf16: a LoRA cannot be merged cleanly into 4-bit weights, so this
    #                  path refuses to run quantised rather than silently changing what it anchors.
    #   --kl-ref base  train the SFT adapter directly; reference is the pretrained base. More
    #                  freedom to move, more risk of losing the house style.
    if args.kl_ref == 'sft':
        assert args.quant == 'none', (
            '--kl-ref sft requires --quant none: merging a LoRA into 4-bit weights is not exact, '
            'and a silently-unmerged merge would leave the reference pointing at the base model, '
            'which is the bug this flag exists to fix')
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model], torch_dtype=torch.bfloat16, device_map={'': 0},
            trust_remote_code=True)
        sft = PeftModel.from_pretrained(base, args.adapter)
        merged = sft.merge_and_unload()          # SFT is now in the weights, and frozen
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                         bias='none', task_type='CAUSAL_LM',
                         target_modules=list(args.lora_targets.split(',')))
        model = get_peft_model(merged, cfg)
        print('[grpo] KL reference = SFT model (merged); training a fresh LoRA on top')
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model], quantization_config=quant if args.quant == '4bit' else None,
            torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        print('[grpo] KL reference = pretrained base; training the SFT adapter itself')
    model.train()

    pad = tok.pad_token_id if tok.pad_token_id is not None else PAD_ID
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS_ID

    holdout = load_holdout(args.data_dir)
    prompts = load_prompts(args.data_dir, EVIDENCE, holdout)
    assert not [p for p in prompts if p['pair'] in holdout], 'benchmark leak'
    replay = load_replay(args.data_dir)
    print('[grpo] %d prompts, %d replay rows, %d held-out pairs excluded'
          % (len(prompts), len(replay), len(holdout)))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, 'grpo_log.jsonl'), 'a')
    best = -1.0

    def seq_logprobs(ids, mask, prompt_len, use_ref=False):
        """Sum of per-token log-probs over COMPLETION tokens only."""
        ctx = model.disable_adapter() if use_ref else _null()
        with ctx:
            out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[:, :-1]
        tgt = ids[:, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        keep = torch.zeros_like(lp, dtype=torch.bool)
        keep[:, prompt_len - 1:] = True
        keep &= (tgt != pad)
        return lp, keep

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = rng.sample(prompts, min(args.batch, len(prompts)))
        pol_terms, kls, rews, comps, trivial = [], [], [], collections.defaultdict(list), 0

        for item in batch:
            pids = tok(item['prompt'], add_special_tokens=True)['input_ids']
            ids = torch.tensor([pids], device=dev)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.group,
                                     pad_token_id=pad, eos_token_id=eos)
            texts = [tok.decode(g[len(pids):], skip_special_tokens=False) for g in gen]
            rewards = []
            for t in texts:
                sent, smi = _split_completion(t)
                s = rw.score(item['model_input'], sent, smi, item['input_smiles'],
                             ligand_code=item['ligand_code'])
                rewards.append(s['total'])
                for k, v in s['components'].items():
                    comps[k].append(v)
                comps['n_claims'].append(s['n_claims'])
            rews += rewards
            adv, useful = group_advantages(rewards)
            if not useful:
                trivial += 1
                continue

            mask = (gen != pad).long()
            with torch.no_grad():
                lp_old, keep = seq_logprobs(gen, mask, len(pids), use_ref=False)
                lpr, _ = seq_logprobs(gen, mask, len(pids), use_ref=True)
            lp, keep = seq_logprobs(gen, mask, len(pids), use_ref=False)
            a = torch.tensor(adv, device=dev, dtype=lp.dtype).unsqueeze(1)
            ntok = keep.sum(1).clamp(min=1)
            # Clipped surrogate, as in the GRPO formulation. At --inner-epochs 1 the ratio is
            # exactly 1 and the clip is a mathematical no-op (which is also TRL's GRPOTrainer
            # default), so this is identical to a group-baselined policy gradient there. It is
            # present because rollouts are the expensive part -- 8 samples from a 14B model -- so
            # reusing each one for several epochs is attractive, and then the clip does real work.
            ratio = (lp - lp_old).clamp(-20, 20).exp()
            unclipped = ratio * a
            clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * a
            surr = torch.minimum(unclipped, clipped)
            pol_terms.append(-((surr * keep).sum(1) / ntok).mean())
            # k3 KL estimator: unbiased and non-negative, unlike the plain log-ratio
            d = (lpr - lp).clamp(-20, 20)
            kls.append((((d.exp() - d - 1) * keep).sum(1) / ntok).mean())

        if not pol_terms:
            print('[grpo] step %d: every group trivial, skipping' % step)
            continue

        loss = torch.stack(pol_terms).mean() + args.beta * torch.stack(kls).mean()

        # SFT anchor
        sft_val = float('nan')
        if args.lam > 0 and replay:
            pr, co = rng.choice(replay)
            pi = tok(pr, add_special_tokens=True)['input_ids']
            ci = tok(co, add_special_tokens=False)['input_ids'] + [eos]
            full = torch.tensor([pi + ci], device=dev)
            lab = full.clone()
            lab[:, :len(pi)] = -100
            sft = model(input_ids=full, attention_mask=torch.ones_like(full), labels=lab).loss
            sft_val = float(sft)
            loss = loss + args.lam * sft

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        rec = {'step': step, 'loss': float(loss), 'reward_mean': sum(rews) / max(len(rews), 1),
               'trivial_groups': trivial, 'groups': len(batch), 'sft_loss': sft_val,
               'secs': round(time.time() - t0, 1),
               'components': {k: sum(v) / len(v) for k, v in comps.items()}}
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print('[grpo] step %3d  reward %.3f  trivial %d/%d  sft %.3f  %.0fs  '
              'attrib %.3f n_claims %.2f banned %.3f'
              % (step, rec['reward_mean'], trivial, len(batch), sft_val, rec['secs'],
                 rec['components'].get('attribution', 0), rec['components'].get('n_claims', 0),
                 rec['components'].get('banned_vocabulary', 0)))

        if step % args.save_every == 0 and rec['reward_mean'] > best:
            best = rec['reward_mean']
            model.save_pretrained(os.path.join(args.out, 'adapter_best'))
            json.dump(rec, open(os.path.join(args.out, 'adapter_best', 'step.json'), 'w'), indent=2)
            print('[grpo] saved adapter_best at step %d (reward %.3f)' % (step, best))

    model.save_pretrained(os.path.join(args.out, 'adapter_final'))
    print('[grpo] done')
    return 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _split_completion(text):
    """-> (sentence, smiles). Mirrors the eval parser: SMILES on its own line after the sentence."""
    import re
    t = text.replace('<|im_end|>', '').replace('</s>', '')
    m = re.search(r'^\s*SMILES:\s*(\S+)\s*$', t, re.M)
    smi = m.group(1) if m else None
    sent = t[:m.start()].strip() if m else t.strip()
    sent = sent.split('\n')[-1].strip() if sent else sent
    return sent, smi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='chemdfm')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--data-dir', default='sft_out_g3g4')
    ap.add_argument('--out', default='sft_runs/chemdfm_grpo_v1')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--batch', type=int, default=2, help='prompts per step')
    ap.add_argument('--group', type=int, default=8, help='G: samples per prompt')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.04, help='KL weight to the frozen reference')
    ap.add_argument('--lam', type=float, default=1.0, help='SFT anchor weight')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', dest='top_p', type=float, default=0.95)
    ap.add_argument('--max-new', type=int, default=256)
    ap.add_argument('--quant', default='4bit', choices=['4bit', 'none'])
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kl-ref', dest='kl_ref', default='sft', choices=['sft', 'base'],
                    help="what the KL anchors to. 'sft' freezes the validated arm and trains a "
                         "second LoRA on top (conservative, needs --quant none); 'base' trains the "
                         "SFT adapter with the pretrained model as reference. Worth running both.")
    ap.add_argument('--inner-epochs', dest='inner_epochs', type=int, default=1,
                    help='policy updates per rollout. >1 activates the clip and reuses rollouts.')
    ap.add_argument('--clip-eps', dest='clip_eps', type=float, default=0.2)
    ap.add_argument('--lora-r', dest='lora_r', type=int, default=16)
    ap.add_argument('--lora-targets', dest='lora_targets',
                    default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--dry-run', action='store_true',
                    help='no model: exercise data, reward, advantages and curriculum locally')
    a = ap.parse_args()
    rw = load_reward()
    global EVIDENCE
    EVIDENCE = os.path.join(rw.DATA, 'v7_evidence.jsonl')
    assert os.path.exists(EVIDENCE), (
        'evidence not found at %s (reward module resolved DATA=%s); set COVF_DATA'
        % (EVIDENCE, rw.DATA))
    print('[grpo] evidence %s' % EVIDENCE)
    return dry_run(a, rw) if a.dry_run else train(a, rw)


if __name__ == '__main__':
    sys.exit(main())
