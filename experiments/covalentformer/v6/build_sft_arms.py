#!/usr/bin/env python3
"""Build a training arm against a FIXED validation benchmark.

Why this exists. `sft_data.py` chooses its own 15% target-component holdout for whichever
generation it is given. That is correct for a single dataset, but it makes the question we now
want to answer unanswerable: "does training on g3 beat training on g2?" cannot be read off two
runs that were scored on two different val sets. Changing both the training set and the test set
at once measures nothing.

So this builder varies ONE thing. The validation rows are copied verbatim out of an existing
built arm -- by default `sft_out_g2/sft_pocket.jsonl`, the 262-pair g2 holdout every number in
sft_runs/ was scored on -- and the training rows are rebuilt from whichever label generations are
named. Every arm is therefore scored on the same 262 prompts with the same 262 reference
sentences, and the g2 numbers already on disk remain valid as the g2 row of the table.

The leakage guarantees are preserved, not assumed: a training pair is dropped if it shares a
target, a PDB (either side), or an input SMILES with any val pair, and the assertions at the end
re-measure all three on the rows actually written.

One caveat to state when reporting these runs. The benchmark sentences are g2-style, written
before the pocket envelope existed, so a g3-trained model that names an envelope residue can be
marked down on the reference-based axes (ref_residue_recall, selectivity). The axes computed
against the crystal evidence rather than the reference sentence -- element fidelity, direction
fidelity, groundedness -- are unaffected. Read both.

    python build_sft_arms.py --train-gens g3 --out-dir sft_out_g3
    python build_sft_arms.py --train-gens g2,g3 --out-dir sft_out_g2g3
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sft_data as S  # noqa: E402  -- reused so prompts are byte-identical to the g2 arm

GEN_ALIASES = {
    'g1': 'g1_v1_retrospective',
    'g2': 'g2_v2_input_anchored',
    'g3': 'g3_v2_input_anchored_envelope',
    # g4 is the independent SECOND pass over the full 7,069-pair corpus -- same pipeline, same
    # briefs, different labellers and chairs. Its records live in label_out_g4/, so a g4 build
    # must run with COVF_LABEL_DIR=label_out_g4.
    'g4': 'g4_v2_input_anchored_envelope_pass2',
}


def resolve_gens(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        out.append(GEN_ALIASES.get(part, part))
    return out


def pdbs_of(pid, rec):
    out = {pid.split()[0]}
    apdb = (rec.get('LABEL_ONLY') or {}).get('analogue_pdb')
    if apdb:
        out.add(apdb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-gens', required=True,
                    help='generations to TRAIN on, comma separated (g1/g2/g3 or full ids)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--val-from', default=os.path.join(HERE, 'sft_out_g2', 'sft_pocket.jsonl'),
                    help='built arm whose split=="val" rows become this arm\'s val set verbatim')
    # sft_data.load() asserts the loaded counts and has no safe default for a generation it was
    # not written for. Supplied per run rather than removed -- that assertion is the guard against
    # a dataset silently changing size. g2 2000/1747, g3 3069/2644, g2+g3 5069/4391.
    ap.add_argument('--n-pairs-expected', type=int, required=True,
                    help='assert the loaded pair count for --train-gens')
    ap.add_argument('--n-usable-expected', type=int, required=True,
                    help='assert the ADOPT+REWRITE count for --train-gens')
    a = ap.parse_args()

    gens = resolve_gens(a.train_gens)

    # ---- the fixed benchmark, taken verbatim ------------------------------------------------
    val_pocket, val_blind = [], []
    blind_path = a.val_from.replace('sft_pocket.jsonl', 'sft_blind.jsonl')
    for path, sink in ((a.val_from, val_pocket), (blind_path, val_blind)):
        for line in open(path):
            row = json.loads(line)
            if row.get('split') == 'val':
                sink.append(row)
    assert val_pocket, 'no val rows in %s' % a.val_from
    assert len(val_pocket) == len(val_blind), \
        'pocket/blind val sizes differ: %d vs %d' % (len(val_pocket), len(val_blind))
    val_pairs = {r['pair'] for r in val_pocket}
    val_targets = {r['target'] for r in val_pocket}
    val_smiles = {r['input_smiles'] for r in val_pocket}
    val_pdbs = set()
    for r in val_pocket:
        val_pdbs.add(r['pdb'])
    print('benchmark: %d val pairs, %d targets, from %s'
          % (len(val_pocket), len(val_targets), os.path.relpath(a.val_from, HERE)))

    # ---- the training rows, rebuilt -----------------------------------------------------------
    labels, recs = S.load(','.join(gens), n_pairs_expected=a.n_pairs_expected)
    trainable = sorted(k for k, v in labels.items() if v['verdict'] in ('ADOPT', 'REWRITE'))
    assert len(trainable) == a.n_usable_expected, \
        'expected %d usable (ADOPT+REWRITE), got %d' % (a.n_usable_expected, len(trainable))
    print('loaded %d labels over %s, %d usable' % (len(labels), '+'.join(gens), len(trainable)))

    drop = collections.Counter()
    train_pairs = []
    for pid in trainable:
        rec = recs[pid]
        if pid in val_pairs:
            drop['in the benchmark'] += 1
            continue
        if rec['target'] in val_targets:
            drop['shares a benchmark target'] += 1
            continue
        if pdbs_of(pid, rec) & val_pdbs:
            drop['shares a benchmark PDB'] += 1
            continue
        if rec['MODEL_INPUT']['input_smiles'] in val_smiles:
            drop['shares a benchmark input SMILES'] += 1
            continue
        train_pairs.append(pid)
    print('train %d pairs; dropped %d for benchmark overlap:' % (len(train_pairs), sum(drop.values())))
    for k, n in drop.most_common():
        print('    %-34s %d' % (k, n))

    rows_pocket, rows_blind = list(val_pocket), list(val_blind)
    for pid in train_pairs:
        rec, v = recs[pid], labels[pid]
        mi, target = rec['MODEL_INPUT'], rec['target']
        for pocket, sink in ((True, rows_pocket), (False, rows_blind)):
            input_block = S.render_prompt(target, mi, pocket=pocket)
            prompt = S.alpaca_prompt(input_block)
            S.assert_no_leak(prompt, rec)
            sink.append({
                'pair': pid,
                'target': target,
                'pdb': pid.split()[0],
                'split': 'train',
                'verdict': v['verdict'],
                'label_generation': v.get('label_generation'),
                'arm': 'pocket' if pocket else 'blind',
                'prompt': prompt,
                'completion': S.render_assistant(v['final'], rec['ANSWER']['analogue_smiles']),
                'reference_sentence': v['final'].strip(),
                'analogue_smiles': rec['ANSWER']['analogue_smiles'],
                'input_smiles': mi['input_smiles'],
                'n_atom_rows': len(mi['atoms']),
                'table_residues': sorted(S.table_residues(mi)),
                'reference_residues': sorted(S.residue_tokens(v['final'])),
            })

    # The REJECTs are kept on disk, carrying split "excluded" and an empty completion, exactly as
    # sft_data.py writes them. Two reasons, and load_rows_jsonl asserts the second: no pair is ever
    # dropped from the corpus, and every consumer filters on `split`, so a REJECT cannot reach
    # training or scoring by accident.
    rejected = sorted(k for k, v in labels.items() if v['verdict'] == 'REJECT')
    for pid in rejected:
        rec, v = recs[pid], labels[pid]
        mi, target = rec['MODEL_INPUT'], rec['target']
        for pocket, sink in ((True, rows_pocket), (False, rows_blind)):
            input_block = S.render_prompt(target, mi, pocket=pocket)
            prompt = S.alpaca_prompt(input_block)
            S.assert_no_leak(prompt, rec)
            sink.append({
                'pair': pid,
                'target': target,
                'pdb': pid.split()[0],
                'split': 'excluded',
                'verdict': 'REJECT',
                'label_generation': v.get('label_generation'),
                'arm': 'pocket' if pocket else 'blind',
                'prompt': prompt,
                'completion': '',
                'reference_sentence': '',
                'analogue_smiles': '',
                'input_smiles': mi['input_smiles'],
                'n_atom_rows': len(mi['atoms']),
                'table_residues': sorted(S.table_residues(mi)),
                'reference_residues': [],
            })
    print('kept %d REJECT rows as split="excluded"' % len(rejected))

    # ---- re-measure the guarantees on what was actually written -------------------------------
    for name, rows in (('pocket', rows_pocket), ('blind', rows_blind)):
        tr = [r for r in rows if r['split'] == 'train']
        va = [r for r in rows if r['split'] == 'val']
        t_tr, t_va = {r['target'] for r in tr}, {r['target'] for r in va}
        assert not (t_tr & t_va), '%s TARGET OVERLAP: %r' % (name, sorted(t_tr & t_va))
        p_tr, p_va = {r['pdb'] for r in tr}, {r['pdb'] for r in va}
        assert not (p_tr & p_va), '%s PDB OVERLAP: %r' % (name, sorted(p_tr & p_va))
        s_tr = {r['input_smiles'] for r in tr}
        s_va = {r['input_smiles'] for r in va}
        assert not (s_tr & s_va), '%s SMILES OVERLAP: %d' % (name, len(s_tr & s_va))
        assert not ({r['pair'] for r in tr} & {r['pair'] for r in va}), '%s PAIR OVERLAP' % name

    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (('sft_pocket.jsonl', rows_pocket), ('sft_blind.jsonl', rows_blind)):
        with open(os.path.join(a.out_dir, name), 'w') as fh:
            for row in rows:
                fh.write(json.dumps(row) + '\n')

    n_tr = sum(1 for r in rows_pocket if r['split'] == 'train')
    gen_counts = collections.Counter(r.get('label_generation') for r in rows_pocket
                                     if r['split'] == 'train')
    stats = {
        'builder': 'build_sft_arms.py',
        'train_generations': gens,
        'val_from': a.val_from,
        'val_is_verbatim_copy': True,
        'n_train': n_tr,
        'n_val': len(val_pocket),
        'train_generation_counts': dict(gen_counts),
        'train_targets': len({r['target'] for r in rows_pocket if r['split'] == 'train'}),
        'val_targets': len(val_targets),
        'dropped_for_benchmark_overlap': dict(drop),
        'target_overlap': 0,
        'pdb_overlap': 0,
        'input_smiles_overlap': 0,
    }
    json.dump(stats, open(os.path.join(a.out_dir, 'split_stats.json'), 'w'), indent=2)
    print('\nwrote %s: %d train + %d val = %d rows per arm'
          % (a.out_dir, n_tr, len(val_pocket), len(rows_pocket)))
    print('train by generation: %s' % dict(gen_counts))


if __name__ == '__main__':
    main()
