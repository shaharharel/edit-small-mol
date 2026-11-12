#!/usr/bin/env python3
"""Score one arm -- trained or zero-shot -- on a named split. Runnable standalone for QA.

    # a trained cell of the 2x2
    python eval_sft.py --model smiley --arm pocket --adapter sft_runs/smiley_pocket_s0/adapter \
                       --split val --out sft_runs/smiley_pocket_s0/eval
    # a zero-shot baseline: no --adapter
    python eval_sft.py --model smiley --arm pocket --split val --out sft_runs/zeroshot_smiley

Everything needed to reproduce a number is on the command line: checkpoint, arm, split. Nothing is
read from a run directory except the adapter weights, so this can be pointed at any checkpoint.

FORMAT COMPLIANCE, SMILES VALIDITY AND NOVELTY ARE THREE NUMBERS, NEVER ONE
--------------------------------------------------------------------------
A combined "validity" rate conflates two completely different failures, and the zero-shot smoke test
showed exactly why. Base-Instruct scored 1/5 "valid" -- but it was writing prose with the SMILES in
a code fence ("Here is a SMILES string... `CC(=O)Nc1ccc(cc1)S(=O)(=O)N` ..."), and that string is
sulfanilamide, which parses fine. It failed because the parser was handed the whole completion
including the prose. A model that knows the chemistry but not our output format is a very different
problem from one that emits confident nonsense, so:

    format_compliance  did the output have the expected shape at all -- a sentence, then an
                       explicit `SMILES:` line that yields a token
    smiles_validity    of the SMILES actually EXTRACTED (robustly, see `extract_smiles`), what
                       fraction RDKit parses
    novelty            of the valid ones, what fraction differs from the input molecule

Validity is computed on what extraction recovered, not on the raw completion, so a format failure is
never scored as a chemistry failure. The extraction method used is recorded per row, so the share of
answers that needed rescuing from prose is visible rather than hidden.

This also means the zero-shot gap between the two checkpoints MUST NOT be read as a prediction of
the trained gap: base's zero-shot deficit is largely format, and LoRA SFT on 860 examples of our
format is precisely the thing that fixes format.

DISTINCT MOLECULES, BECAUSE VALIDITY IS GAMEABLE
-----------------------------------------------
`n_distinct_canonical_smiles` is reported alongside validity. In the smoke test 4 of base's 5
zero-shot samples were the same molecule. A model that emits one valid molecule every time scores
100% validity and is useless, and echo rate alone does not catch it -- the collapsed molecule need
not be the input.

GROUNDEDNESS IS THE HEADLINE, AND ITS CEILING IS ~0.90, NOT 1.0
--------------------------------------------------------------
Of the residues the generated sentence names, the fraction that appear in that pair's MODEL_INPUT
contact table. Ungrounded residue names are hallucinations, and this is the metric the whole run
exists to produce: the confident-misidentification failure the base checkpoint showed zero-shot
(naming one molecule three mutually contradictory things) surfaces here as ungrounded residue
claims.

It is reported against the reference ceiling, recomputed from the same rows. The chair's own
sentences are only ~0.90 grounded, because the analogue's new atoms reach residues the input
molecule never touched -- unknowable from MODEL_INPUT by construction. Reading 0.90 as "10%
hallucination" would be mistaking the ceiling for a failure.

The blind arm is scored against the same tables it was never shown. That is the point: a residue
named without having been shown the table is a hallucination even when it happens to be right.

WHAT THE VAL NUMBER IS
----------------------
A MODEL-SELECTION SCORE, NOT A HELD-OUT TEST SCORE -- the stopping epoch and checkpoint were both
chosen on this split. It is optimistically biased by an unknown amount and must be labelled so
wherever it is quoted. See sft_data.py and train_sft.py.
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
from sft_data import load_rows_jsonl, residue_tokens   # noqa: E402
import train_sft                                       # noqa: E402
from train_sft import (CFG, EOS_ID, MODELS, PAD_ID, AlpacaSFT,  # noqa: E402
                       build_tokenizer, load_model, set_special_ids)

_SMILES_CHARS = re.compile(r'^[A-Za-z0-9@+\-\[\]()=#$%/\\.*]+$')


def looks_like_smiles(tok):
    tok = tok.strip().strip('`').rstrip('.,;:')
    if len(tok) < 4 or ' ' in tok:
        return None
    if not _SMILES_CHARS.match(tok):
        return None
    if not re.search(r'[A-Za-z]', tok):
        return None
    # Prose words sneak through the character class ("molecule", "Pyrazinamide"), so require at
    # least one bracket, ring digit, bond symbol or lower-case aromatic atom -- something a word
    # does not have.
    if not re.search(r'[0-9\[\]()=#@/\\]', tok):
        return None
    return tok


def extract_smiles(text):
    """Recover a SMILES from a completion, in descending order of how much we trust the shape.

    Returns (smiles, method). Extraction is deliberately generous because scoring a prose-wrapped
    but chemically correct answer as invalid would conflate format failure with chemistry failure;
    `format_compliance` is what records whether the shape was right, and `method` says how far we
    had to reach.
    """
    t = (text or '').strip()
    # 1. the trained shape
    m = re.findall(r'(?im)^\s*SMILES:\s*(\S+)', t)
    if m:
        s = looks_like_smiles(m[-1])
        if s:
            return s, 'smiles_line'
    # 2. fenced blocks
    for block in reversed(re.findall(r'```[A-Za-z]*\s*(.*?)```', t, re.S)):
        for line in reversed([x for x in block.splitlines() if x.strip()]):
            s = looks_like_smiles(line)
            if s:
                return s, 'code_fence'
    # 3. backtick-quoted tokens: take the longest, which is the molecule rather than a fragment
    cands = [s for s in (looks_like_smiles(x) for x in re.findall(r'`([^`\n]+)`', t)) if s]
    if cands:
        return max(cands, key=len), 'backtick'
    # 4. last line that could be a molecule
    for line in reversed([x for x in t.splitlines() if x.strip()]):
        s = looks_like_smiles(line)
        if s:
            return s, 'last_line'
    return None, 'none'


def extract_sentence(text):
    """The prose part: everything before the SMILES line, with fences and fenced content dropped."""
    t = (text or '').strip()
    t = re.split(r'(?im)^\s*SMILES:', t)[0]
    t = re.sub(r'```[A-Za-z]*\s*.*?```', ' ', t, flags=re.S)
    t = t.replace('`', ' ')
    return ' '.join(t.split()).strip()


def is_format_compliant(text):
    """A sentence, then an explicit SMILES: line that yields a token. The shape we trained."""
    s, method = extract_smiles(text)
    return bool(s and method == 'smiles_line' and len(extract_sentence(text).split()) >= 5)


def canon(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


@torch.no_grad()
def response_loss(model, ds, device):
    """Mean per-token NLL over response tokens only. Batch 1, so padding cannot skew it."""
    tot, ntok = 0.0, 0
    for i in range(len(ds)):
        e = ds[i]
        ids = torch.tensor([e['input_ids']], device=device)
        lab = torch.tensor([e['labels']], device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        tot += torch.nn.functional.cross_entropy(logits[mask], tgt[mask], reduction='sum').item()
        ntok += int(mask.sum())
    loss = tot / ntok
    return {'response_loss': round(loss, 4),
            'response_perplexity': round(float(torch.exp(torch.tensor(loss))), 4),
            'n_response_tokens': ntok}


@torch.no_grad()
def generate(model, tok, rows, device, max_new=256):
    outs = []
    # PROGRESS, BECAUSE SILENCE AND A HANG LOOK IDENTICAL. Generation is unbatched and nothing
    # is written until scoring finishes, so a 400-row run on a 14B model is ~90 minutes of no
    # output at all. One such run died at ~95 minutes on a preemptible box and left no way to
    # tell how far it had got or whether it had stalled. The line is flushed and partial
    # completions are kept in `outs`, so a later failure is at least bounded.
    import time as _t
    t0 = _t.time()
    for i, r in enumerate(rows):
        ids = torch.tensor([tok(r['prompt'], add_special_tokens=True)['input_ids']], device=device)
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=PAD_ID, eos_token_id=EOS_ID)
        outs.append(tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            el = _t.time() - t0
            print('[gen] %d/%d  %.1fs/item  elapsed %.1f min  eta %.1f min'
                  % (i + 1, len(rows), el / (i + 1), el / 60,
                     (len(rows) - i - 1) * el / (i + 1) / 60), flush=True)
    return outs


# contact-table residues for one row, whatever the corpus calls them.
# THE BUG THIS REPLACES. score() read `table_residues` only. The ChEMBL potency corpus keeps
# its residues under `cited_residues` and its contact table inside the prompt text, so the
# lookup came back empty and -- because the naming counter was gated on it -- the summary
# reported `sentences_naming_a_residue: 0.0` for runs whose completions named a residue 190
# times out of 200. Naming and grounding are different questions: a sentence can be counted as
# naming a residue with no table in sight; only grounding needs something to check against.
def row_table_residues(r):
    # Order matters. `table_residues` is explicit eval metadata and is trusted first. Next comes
    # the prompt's own "MEASURED POCKET CONTACTS" block, because that is literally what the model
    # was shown. `cited_residues` is LAST and is a fallback only: on the ChEMBL potency corpus it
    # holds the residues the REFERENCE sentence cites (as dicts: {residue, atom, dist_A}), i.e. a
    # small subset of the table, so grounding a generation against it would score a correct
    # citation of any other contact as a hallucination.
    v = r.get('table_residues')
    if v:
        return residue_tokens(' '.join(v)) if not isinstance(v, str) else set(v)
    p = r.get('prompt') or ''
    if 'POCKET CONTACTS' in p:
        got = residue_tokens(p.split('POCKET CONTACTS', 1)[1])
        if got:
            return got
    cited = r.get('cited_residues') or []
    out = set()
    for c in cited:
        name = c.get('residue') if isinstance(c, dict) else c
        if name:
            out |= residue_tokens(str(name))
    return out


def score(rows, completions):
    recs = []
    g_hit = g_tot = n_named = n_fully = 0
    g_macro = []
    methods = {}
    for r, text in zip(rows, completions):
        sent = extract_sentence(text)
        smi, method = extract_smiles(text)
        methods[method] = methods.get(method, 0) + 1
        cs, ci = canon(smi), canon(r['input_smiles'])
        # A CORPUS WITHOUT A CONTACT TABLE IS NOT A SCORING ERROR. The crystal-pair corpora
        # carry `table_residues` (the pocket contacts a residue claim can be checked against);
        # the ChEMBL potency corpora do not, and the hard lookup threw KeyError after 30
        # minutes of generation had already been produced and then discarded. Missing means
        # "no table to check against": residue grounding is reported as None for those rows
        # rather than silently counted as ungrounded, which would read as invention.
        tbl = row_table_residues(r)
        has_tbl = bool(tbl)
        named = residue_tokens(sent)
        h = sum(1 for n in named if n in tbl)
        if named:
            n_named += 1          # NAMING does not depend on having a table to check against
        if named and has_tbl:
            g_hit += h
            g_tot += len(named)
            g_macro.append(h / len(named))
            n_fully += (h == len(named))
        recs.append({'pair': r['pair'], 'target': r['target'], 'split': r.get('split'),
                     'completion': text, 'gen_sentence': sent,
                     'gen_smiles': smi, 'extraction_method': method,
                     'gen_smiles_canonical': cs,
                     'format_compliant': is_format_compliant(text),
                     'valid': cs is not None,
                     'echo': bool(cs and ci and cs == ci),
                     'named_residues': sorted(named),
                     'grounded_residues': sorted(n for n in named if n in tbl) if has_tbl else None,
                     'ungrounded_residues': (sorted(n for n in named if n not in tbl)
                                             if has_tbl else None),
                     'groundedness': (h / len(named)) if (named and has_tbl) else None,
                     'reference_sentence': r.get('reference_sentence'),
                     'reference_smiles': r.get('analogue_smiles'),
                     'reference_residues': r.get('reference_residues'),
                     'n_table_residues': len(tbl)})
    n = len(recs)
    n_extracted = sum(1 for x in recs if x['gen_smiles'])
    n_valid = sum(1 for x in recs if x['valid'])
    distinct = {x['gen_smiles_canonical'] for x in recs if x['gen_smiles_canonical']}
    summary = {
        'n_rows': n,
        # 1. shape
        'format_compliance': round(sum(x['format_compliant'] for x in recs) / n, 4),
        'smiles_extracted_rate': round(n_extracted / n, 4),
        'extraction_methods': methods,
        # 2. chemistry, on what extraction recovered
        'smiles_validity_of_extracted': round(n_valid / n_extracted, 4) if n_extracted else None,
        'smiles_validity_of_all_rows': round(n_valid / n, 4),
        # 3. is it a different molecule
        'novelty_of_valid': (round(sum(1 for x in recs if x['valid'] and not x['echo']) / n_valid, 4)
                             if n_valid else None),
        'echo_rate_of_valid': (round(sum(1 for x in recs if x['echo']) / n_valid, 4)
                               if n_valid else None),
        'n_distinct_canonical_smiles': len(distinct),
        'distinct_fraction_of_valid': round(len(distinct) / n_valid, 4) if n_valid else None,
        # the headline
        'groundedness_micro': round(g_hit / g_tot, 4) if g_tot else None,
        'groundedness_macro': round(statistics.mean(g_macro), 4) if g_macro else None,
        'fully_grounded_frac': round(n_fully / n_named, 4) if n_named else None,
        'sentences_naming_a_residue': round(n_named / n, 4),
        'residues_named_per_sentence': round(g_tot / n_named, 2) if n_named else None,
    }
    return summary, recs


def reference_ceiling(rows):
    macro, hit, tot, full, named_n = [], 0, 0, 0, 0
    for r in rows:
        # same tolerance as score(): a corpus with no contact table has no ceiling to compute
        tbl = set(r.get('table_residues') or ())
        named = set(r.get('reference_residues') or ())
        if not tbl:
            continue
        if not named:
            continue
        named_n += 1
        h = sum(1 for n in named if n in tbl)
        hit += h
        tot += len(named)
        macro.append(h / len(named))
        full += (h == len(named))
    return {'groundedness_micro': round(hit / tot, 4) if tot else None,
            'groundedness_macro': round(statistics.mean(macro), 4) if macro else None,
            'fully_grounded_frac': round(full / named_n, 4) if named_n else None,
            'residues_named_per_sentence': round(tot / named_n, 2) if named_n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--arm', choices=['pocket', 'blind'], required=True)
    ap.add_argument('--adapter', default=None, help='omit for the zero-shot baseline')
    ap.add_argument('--base-adapter', dest='base_adapter', default=None,
                    help='adapter merged BEFORE --adapter. For a GRPO run trained with '
                         '--kl-ref sft, pass the SFT adapter here and the GRPO checkpoint as '
                         '--adapter, matching the stack it was trained in.')
    ap.add_argument('--split', default='val')
    ap.add_argument('--data-dir', default='sft_out')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0, help='recorded only; generation is greedy')
    ap.add_argument('--no-merge', dest='no_merge', action='store_true',
                    help='keep --adapter active instead of merging it into the base weights. '
                         'Matches how the adapter behaved during training; avoids the bf16 '
                         'rounding that merge_and_unload() applies to W + BA.')
    ap.add_argument('--max-length', dest='max_length', type=int, default=None,
                    help='token ceiling for the teacher-forced response_loss pass. Defaults to '
                         "train_sft CFG['max_length'] (2304), which was measured on the g1 corpus "
                         '(max 2128) and is stale for later ones: g3+g4 reaches 2413 on val and '
                         '2563 on train. The ceiling REFUSES rather than truncates, so an eval '
                         'over a longer corpus dies on it. Raising it here does not touch the '
                         'training threshold, and costs nothing at eval: the collator pads to '
                         "each batch's own longest sequence, not to this number.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.data_dir, 'sft_%s.jsonl' % a.arm)
    rows = load_rows_jsonl(path, a.arm, split=a.split)
    if a.limit:
        rows = rows[:a.limit]

    model_path = MODELS[a.model]
    # Switch to this model's vocabulary, then re-read the ids: `from train_sft import PAD_ID`
    # bound the Llama values at import time and would not see the change.
    set_special_ids(a.model)
    globals()['PAD_ID'], globals()['EOS_ID'] = train_sft.PAD_ID, train_sft.EOS_ID
    tok = build_tokenizer(model_path, a.model)
    model = load_model(model_path, CFG['bf16'], device_map='cuda')
    # --base-adapter is merged FIRST, then --adapter is applied on top of the merged weights.
    #
    # A GRPO run launched with --kl-ref sft merges the SFT adapter into the base and trains a
    # fresh LoRA over those merged weights (train_grpo.py, the args.kl_ref == 'sft' branch), so
    # its checkpoint is only meaningful in that same stack. Loading it onto the raw base instead
    # would evaluate GRPO-without-SFT -- a model that was never trained and never existed -- and
    # would do so silently, reporting a plausible-looking number.
    if a.base_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.base_adapter).merge_and_unload()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        # merge_and_unload() folds BA into W and rounds the sum to the base dtype. In bf16 that is
        # ~8 mantissa bits, so a small LoRA update can be partly rounded away -- and training never
        # sees that rounding, because it keeps the adapter active and computes W.x + BA.x. On v4's
        # checkpoint the two differ by 0.080 of reward over the same 32 prompts. --no-merge keeps
        # the adapter active, matching how the weights were trained and evaluated in-loop.
        if not a.no_merge:
            model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    max_len = a.max_length or CFG['max_length']
    out = response_loss(model, AlpacaSFT(rows, tok, max_len), device)
    summary, recs = score(rows, generate(model, tok, rows, device))
    summary.update(out)
    summary.update({'model': a.model, 'model_path': model_path, 'arm': a.arm, 'split': a.split,
                    'adapter': a.adapter, 'base_adapter': a.base_adapter, 'max_length': max_len, 'no_merge': a.no_merge,
                    'zero_shot': a.adapter is None, 'seed': a.seed,
                    'reference_ceiling': reference_ceiling(rows),
                    'val_is_model_selection_not_test': a.split == 'val'})

    with open(os.path.join(a.out, 'generations.jsonl'), 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    json.dump(summary, open(os.path.join(a.out, 'eval_summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print('\n' + '=' * 95)
    for r in recs[:5]:
        print('PAIR %s   target %s' % (r['pair'], r['target']))
        print('  GEN : %s' % (r['gen_sentence'] or '')[:320])
        print('  GEN SMILES: %s  [%s]  valid=%s echo=%s' % (r['gen_smiles'],
                                                            r['extraction_method'],
                                                            r['valid'], r['echo']))
        # the potency corpora carry no reference sentence: this dump is a courtesy, and it
        # threw TypeError AFTER eval_summary.json was written, so the step reported FAIL on a
        # run whose results were complete.
        print('  REF : %s' % (r.get('reference_sentence') or '')[:320])
        print('  REF SMILES: %s' % r['reference_smiles'])
        print('  grounded %s  ungrounded %s' % (r['grounded_residues'], r['ungrounded_residues']))
        print('-' * 95)


if __name__ == '__main__':
    main()
