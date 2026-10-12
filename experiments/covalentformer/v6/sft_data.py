#!/usr/bin/env python3
"""Turn the 1,000 chair-adjudicated labels into chat-format SFT data, pocket-aware and pocket-blind.

Two JSONL files come out of one pass so the ablation is guaranteed to differ in exactly one thing:

    sft_pocket.jsonl   system + (target, input SMILES, measured contact table) -> sentence + SMILES
    sft_blind.jsonl    system + (target, input SMILES)                         -> sentence + SMILES

Same rows, same ids, same split assignment, same assistant turn, byte-identical system prompt. If
the blind run matches the pocket run then the contact table carried no signal and the experiment has
demonstrated nothing, which is the one result this pipeline exists to be able to detect. Anything
else that differed between the two files (a reshuffled split, a different seed, a stray extra field)
would confound that comparison, so both files are emitted together and never built separately.

THE MODEL_INPUT / LABEL_ONLY BOUNDARY
-------------------------------------
Each corpus record carries both the input ligand's own crystal (`MODEL_INPUT`) and the analogue's
own crystal (`LABEL_ONLY`). At inference the analogue does not exist yet, so its crystal cannot
exist either: every `LABEL_ONLY` field is counterfactual. Leaking one -- `analogue_contacts`,
`conserved_interactions`, `site_of_change`, `analogue_pdb` -- would let the model read off the answer
it is supposed to propose, and the measured numbers would make it look well-calibrated while
teaching it nothing.

The guard here is structural, not a convention anyone has to remember. `render_prompt()` takes
`(target, model_input)` as its only two arguments. It is never handed the record, so there is no
`LABEL_ONLY` in its scope to leak -- a future edit cannot reach the analogue's crystal without
changing the function's signature, which is a visible change. A belt-and-braces string check then
re-reads every finished prompt and asserts the analogue SMILES, the analogue PDB id and every
LABEL_ONLY-only residue token are absent; see `assert_no_leak()`.

`ANSWER.analogue_smiles` is the generation target and appears only in the assistant turn. The
record's `tanimoto`, `property_deltas`, `edit_type` and `QA` blocks are all computed *against the
analogue*, so despite sitting outside `LABEL_ONLY` they are just as leaky -- `property_deltas` is
literally [input_value, analogue_value] pairs. None of them enter a prompt. Only `MODEL_INPUT` and
`target` do.

WHY THE SPLIT IS BY TARGET COMPONENT AND NOT BY PAIR
----------------------------------------------------
The corpus has many pairs per crystal and many crystals per protein: 860 trainable pairs come from
699 input PDBs and only 235 distinct UniProt targets, and the largest single target supplies 55 of
them. A random pair-level split would put sibling pairs off the same crystal -- same pocket, same
residue numbering, often the same contact table to within an atom -- on both sides, and the held-out
number would mostly measure memorisation of pockets already seen. The whole claim being tested is
pocket dependence, so the split has to hold out whole pockets.

Holding out whole *targets* is still not quite enough. One input PDB (13CW) is annotated under two
different UniProt targets, so a target-level split alone can place the same crystal on both sides.
The split is therefore taken over connected components of the bipartite graph linking each target to
every PDB it touches (input PDBs, and analogue PDBs as metadata only -- never rendered). Components
are atomic, which makes zero target overlap and zero PDB overlap true by construction rather than by
inspection; both are asserted anyway. On this data the union merges exactly one target pair, 235
targets -> 234 components.

Validation draws whole components until ~15% of trainable pairs are held out, with one extra rule: a
single component may not exceed 20% of the validation budget. Without that cap a seeded draw can
land the 78-pair component and hand back a validation set that is 60% one protein, so a per-target
quirk would read as a model-level result. The cap costs nothing and buys a validation set spread
over many pockets.

NOTHING IS DROPPED
------------------
Training uses ADOPT + REWRITE (860 pairs). The 140 REJECTs are written into both files too, carrying
`split: "excluded"` and their verdict, because a REJECT is an informative label -- the chair's
judgement that the evidence supports no grounded sentence -- and a later run may want to train a
classifier on exactly that. They have an empty assistant turn and every loader here filters on
`split`, so they cannot leak into training by accident.

USAGE
    python sft_data.py                      # build both JSONL files + split_stats.json
    python sft_data.py --token-stats        # re-read the built files and report token lengths
                                            # (needs the SmileyLlama tokenizer; run it on the box)
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
LABEL_GLOB = os.path.join(HERE, 'label_out', 'batch_00*.json')
OUT_DIR = os.path.join(HERE, 'sft_out')
MODEL_ID = 'THGLab/Llama-3.1-8B-SmileyLlama-1.1'

VAL_FRACTION = 0.15
MAX_COMPONENT_SHARE_OF_VAL = 0.20
SEED = 0

_AA = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR '
                'VAL MSE SEC PYL HID HIE HIP'.split())
_RESTOK = re.compile(r'\b([A-Z]{2,4})(\d+)\b')


# --------------------------------------------------------------------------------------- prompts

SYSTEM = (
    "You are a medicinal chemist proposing a single-step analogue of a ligand whose crystal "
    "structure in its target pocket has been measured.\n"
    "You are given the ligand's SMILES and, where available, a table of its measured protein "
    "contacts: for each ligand heavy atom, the nearest protein atom it faces, the distance, the "
    "angle where one is defined, and the interaction typing.\n"
    "Propose one analogue and state the pocket consequence of the change in a single sentence.\n"
    "Name only residues that appear in the contact table you were given. Where a contact is "
    "flagged uncertain, do not cite it as established. Then give the analogue's SMILES on its own "
    "line.\n"
    "Answer with the sentence, then a line of the form 'SMILES: <smiles>'."
)


def render_prompt(target, model_input, pocket=True):
    """Render the user turn.

    Takes the target string and the record's MODEL_INPUT block, and NOTHING else -- in particular
    not the record, so the analogue's crystal is not in scope and cannot be leaked. See the module
    docstring. `pocket=False` renders the ablation: identity of the protein and the input molecule,
    no measured structure at all.
    """
    L = ['TARGET: %s' % target,
         'INPUT SMILES: %s' % model_input['input_smiles']]
    if not pocket:
        L.append('')
        L.append('No pocket structure is provided.')
        return '\n'.join(L)

    L += ['',
          'MEASURED POCKET CONTACTS of the input ligand, one row per heavy atom '
          '(nearest protein atom within 4.0 A):',
          '%-5s %-2s %-15s %-5s %5s %5s | %s'
          % ('atom', 'el', 'faces', 'chain', 'dist', 'ang', 'interaction')]

    # The interaction text is never truncated. render_v7.py's comments record what truncation cost
    # the labelling round: cutting this column at 60 chars hid the word `uncertain` in 1,116 of the
    # 1,116 rows that carry it, so the rule keyed on that word was unenforceable. The model is being
    # asked to respect the same caveat, so it has to be able to see it.
    no_contact = []
    for a in model_input['atoms']:
        it = a.get('interaction') or ''
        if it.startswith('no contact within') and '[' not in it:
            no_contact.append(a['atom'])
            continue
        L.append('%-5s %-2s %-15s %-5s %5.2f %5s | %s'
                 % (a['atom'], a['element'], a['faces'], a['chain'] or '-', a['distance_A'],
                    ('%.0f' % a['angle_deg']) if a.get('angle_deg') else '-', it))
        # The nearest protein N/O/S when that is not already `faces`. A polar atom whose nearest
        # neighbour is a carbon still has a hydrogen-bond partner; dropping this column deleted
        # typed H-bonds from over half of the labelling packets before it was caught.
        if a.get('polar_partner') and a['polar_partner'] != a['faces']:
            L.append('%-5s %-2s %-15s %-5s %5s %5s | polar partner: %s'
                     % ('', '', a['polar_partner'], '',
                        ('%.2f' % a['polar_distance_A']) if a.get('polar_distance_A') else '-',
                        ('%.0f' % a['polar_angle_deg']) if a.get('polar_angle_deg') else '-',
                        a.get('polar_interaction') or '-'))
    if no_contact:
        L.append('NOTHING WITHIN 4.0 A (listed, so the table is complete): %s'
                 % ', '.join(no_contact))

    # Caveats last, so they sit closest to the generation and are the last thing conditioned on.
    if model_input.get('input_atoms_listed'):
        L.append('COVERAGE: %s' % model_input['input_atoms_listed'])
    if model_input.get('atoms_not_modelled'):
        L.append('NOT MODELLED in this crystal (absent from the coordinates, so they appear in no '
                 'row above): %s' % ', '.join(model_input['atoms_not_modelled']))
    if model_input.get('atoms_dropped_as_clashes'):
        L.append('OMITTED AS CLASHES (present in the crystal but overlapping a protein atom, so '
                 'their rows are unreliable and absent above): %s%s'
                 % (', '.join(model_input['atoms_dropped_as_clashes']),
                    ' -- %s' % model_input['clash_note'] if model_input.get('clash_note') else ''))
    if model_input.get('suspect_residues'):
        L.append('SUSPECT RESIDUES in this crystal (each modelled implausibly close to two or more '
                 'ligand atoms, so its position is unreliable): %s'
                 % ', '.join(model_input['suspect_residues']))
    return '\n'.join(L)


def render_assistant(sentence, analogue_smiles):
    return '%s\nSMILES: %s' % (sentence.strip(), analogue_smiles)


# ------------------------------------------------------------------------- tokenisation helper

def chat_ids(tok, messages, add_generation_prompt=False):
    """Token ids for a conversation, as a flat list of ints.

    `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, which subclasses `UserDict` and
    is therefore NOT an instance of `dict`. An `isinstance(x, dict)` guard silently falls through,
    `len(x)` then counts the two KEYS, and every prompt measures 2 tokens long -- which is exactly
    how the first token-length report came back claiming a median prompt of 2. Duck-type on `keys`,
    and unwrap the batch dimension if one is present.
    """
    out = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=add_generation_prompt)
    if hasattr(out, 'keys'):
        out = out['input_ids']
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(x) for x in out]


# ------------------------------------------------------------------------------- groundedness

def residue_tokens(text):
    """Residue references in free text: a known amino-acid code followed by a number.

    Matching bare `[A-Z]{2,4}\\d+` instead picks up the ATOM-name suffix of every atom spec -- the
    `CD1` of `LEU168.CD1` -- and scores a sentence's own correct citations as hallucinations. That
    mistake put the reference corpus's groundedness at 0.66 instead of its true 0.89.
    """
    out = set()
    for m in _RESTOK.finditer(text or ''):
        if m.group(1) in _AA:
            out.add(m.group(1) + m.group(2))
    return out


def table_residues(model_input):
    """Every residue the model was (or, in the blind arm, would have been) shown for this pair.

    This is eval metadata, not prompt content: it is stored on the row, never inside `messages`. The
    blind arm is scored against the same set, because a residue named without having been shown the
    table is a hallucination even if it happens to be right.
    """
    blob = []
    for a in model_input['atoms']:
        blob += [a.get('faces') or '', a.get('polar_partner') or '',
                 a.get('interaction') or '', a.get('polar_interaction') or '']
    blob += list(model_input.get('suspect_residues') or [])
    return residue_tokens(' '.join(blob))


# ------------------------------------------------------------------------------- leak protection

def label_only_residue_tokens(record):
    """Residue tokens that occur in LABEL_ONLY and in no MODEL_INPUT field.

    Used only to assert they are absent from the prompts. A residue the analogue reaches that the
    input never touched is exactly the kind of token that would betray a leak.
    """
    def toks(blob):
        return {m.group(1) + m.group(2) for m in _RESTOK.finditer(blob) if m.group(1) in _AA}
    mi = json.dumps(record['MODEL_INPUT'])
    lo = json.dumps(record.get('LABEL_ONLY') or {})
    return toks(lo) - toks(mi)


def assert_no_leak(prompt, record):
    """Re-read a finished prompt and refuse anything that smells of the analogue's crystal."""
    pair = record['pair']
    ana = record['ANSWER']['analogue_smiles']
    # The input SMILES is masked out before the analogue is searched for. On 7 of 860 pairs the edit
    # shortens a chain (pentyl -> propyl), which makes the analogue's SMILES a literal substring of
    # the input's -- a coincidence of SMILES text, not a leak, since the input molecule is the one
    # thing the prompt is supposed to contain. Searching the raw prompt flags all 7.
    probe = prompt.replace(record['MODEL_INPUT']['input_smiles'], '<INPUT_SMILES>')
    assert ana not in probe, 'LEAK %s: analogue SMILES in prompt' % pair
    lo = record.get('LABEL_ONLY') or {}
    apdb = lo.get('analogue_pdb')
    if apdb:
        assert apdb not in prompt, 'LEAK %s: analogue PDB %s in prompt' % (pair, apdb)
    # Only LABEL_ONLY content that is ABSENT from MODEL_INPUT can constitute a leak. The two blocks
    # share a closed interaction vocabulary -- 'polar contact, geometry poor' is typed the same way
    # in either crystal -- so a shared string carries no analogue-specific information and flagging
    # it would make this check fire on every pair and get deleted. What matters is a value unique to
    # the analogue's crystal: a distance, a residue it alone reaches, an atom name it alone has.
    mi_blob = json.dumps(record['MODEL_INPUT'])
    for key in ('analogue_contacts', 'conserved_interactions', 'site_of_change'):
        for row in (lo.get(key) or []):
            if isinstance(row, dict):
                for v in row.values():
                    # Long, distinctive strings only: short tokens like a chain id collide with
                    # legitimate MODEL_INPUT content and would make this check meaningless.
                    if isinstance(v, str) and len(v) > 24 and v not in mi_blob:
                        assert v not in prompt, \
                            'LEAK %s: %s value %r in prompt' % (pair, key, v[:60])
    for tok in label_only_residue_tokens(record):
        assert not re.search(r'\b%s\b' % tok, prompt), \
            'LEAK %s: LABEL_ONLY-only residue %s in prompt' % (pair, tok)
    # The pair id encodes the analogue's ligand code ("7ENT J9L->5BX"); it is not rendered at all.
    assert pair not in prompt, 'LEAK %s: pair id in prompt' % pair


# ------------------------------------------------------------------------------------- the split

class _UF(dict):
    def find(self, x):
        self.setdefault(x, x)
        while self[x] != x:
            self[x] = self[self[x]]
            x = self[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self[ra] = rb


def build_components(pairs, recs):
    """Connected components over target <-> PDB. Components are the atomic unit of the split."""
    uf = _UF()
    for pid in pairs:
        r = recs[pid]
        t = 'T:' + r['target']
        uf.find(t)
        uf.union(t, 'P:' + pid.split()[0])
        # The analogue PDB is metadata here and is never rendered into a prompt; including it in the
        # graph only makes the holdout stricter (a crystal cannot sit on both sides in either role).
        apdb = (r.get('LABEL_ONLY') or {}).get('analogue_pdb')
        if apdb:
            uf.union(t, 'P:' + apdb)
    comp = collections.defaultdict(list)
    for pid in pairs:
        comp[uf.find('T:' + recs[pid]['target'])].append(pid)
    return comp


def choose_val(comp, n_pairs, seed=SEED):
    """Draw whole components for validation up to ~VAL_FRACTION, capping any single component."""
    import random
    rng = random.Random(seed)
    budget = VAL_FRACTION * n_pairs
    cap = MAX_COMPONENT_SHARE_OF_VAL * budget
    keys = sorted(comp)          # sorted first so the shuffle is reproducible across dict orders
    rng.shuffle(keys)
    val, total = [], 0
    for k in keys:
        size = len(comp[k])
        if size > cap:
            continue
        if total + size > budget:
            continue
        val.append(k)
        total += size
    return set(val), total


# ------------------------------------------------------------------------------------------ main

def load():
    labels = {}
    for f in sorted(glob.glob(LABEL_GLOB)):
        labels.update(json.load(open(f)))
    recs = {}
    with open(CORPUS) as fh:
        for line in fh:
            r = json.loads(line)
            if r['pair'] in labels:
                recs[r['pair']] = r
    missing = [k for k in labels if k not in recs]
    assert not missing, 'labels with no corpus record: %r' % missing[:5]
    return labels, recs


def build():
    labels, recs = load()
    trainable = sorted(k for k, v in labels.items() if v['verdict'] in ('ADOPT', 'REWRITE'))
    rejected = sorted(k for k, v in labels.items() if v['verdict'] == 'REJECT')
    assert len(trainable) + len(rejected) == len(labels)

    comp = build_components(trainable, recs)
    val_keys, n_val = choose_val(comp, len(trainable))
    comp_of = {pid: ck for ck, pids in comp.items() for pid in pids}
    split_of = {pid: ('val' if comp_of[pid] in val_keys else 'train') for pid in trainable}

    # --- assertions the whole experiment rests on -----------------------------------------
    tr = [p for p in trainable if split_of[p] == 'train']
    va = [p for p in trainable if split_of[p] == 'val']
    t_tr = {recs[p]['target'] for p in tr}
    t_va = {recs[p]['target'] for p in va}
    assert not (t_tr & t_va), 'TARGET OVERLAP: %r' % sorted(t_tr & t_va)

    def pdbs(ps):
        out = set()
        for p in ps:
            out.add(p.split()[0])
            apdb = (recs[p].get('LABEL_ONLY') or {}).get('analogue_pdb')
            if apdb:
                out.add(apdb)
        return out
    p_tr, p_va = pdbs(tr), pdbs(va)
    assert not (p_tr & p_va), 'PDB OVERLAP: %r' % sorted(p_tr & p_va)

    # Input molecules must not recur across the split either: the same ligand in a second crystal
    # would be a near-duplicate prompt. Not guaranteed by the component graph, so it is measured.
    s_tr = {recs[p]['MODEL_INPUT']['input_smiles'] for p in tr}
    s_va = {recs[p]['MODEL_INPUT']['input_smiles'] for p in va}
    smi_overlap = sorted(s_tr & s_va)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows_pocket, rows_blind = [], []
    for pid in trainable + rejected:
        r = recs[pid]
        v = labels[pid]
        is_rej = v['verdict'] == 'REJECT'
        split = 'excluded' if is_rej else split_of[pid]
        mi, target = r['MODEL_INPUT'], r['target']
        for pocket, sink in ((True, rows_pocket), (False, rows_blind)):
            user = render_prompt(target, mi, pocket=pocket)
            assert_no_leak(user, r)
            assistant = '' if is_rej else render_assistant(v['final'], r['ANSWER']['analogue_smiles'])
            sink.append({
                'pair': pid,
                'target': target,
                'pdb': pid.split()[0],
                'split': split,
                'verdict': v['verdict'],
                'arm': 'pocket' if pocket else 'blind',
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': user},
                             {'role': 'assistant', 'content': assistant}],
                'reference_sentence': '' if is_rej else v['final'].strip(),
                'analogue_smiles': '' if is_rej else r['ANSWER']['analogue_smiles'],
                'input_smiles': mi['input_smiles'],
                'n_atom_rows': len(mi['atoms']),
                # Eval metadata. Deliberately outside `messages`, so it cannot reach a prompt.
                'table_residues': sorted(table_residues(mi)),
                'reference_residues': sorted(residue_tokens(v['final'])) if not is_rej else [],
            })

    for name, rows in (('sft_pocket.jsonl', rows_pocket), ('sft_blind.jsonl', rows_blind)):
        path = os.path.join(OUT_DIR, name)
        with open(path, 'w') as fh:
            for row in rows:
                fh.write(json.dumps(row) + '\n')
        print('wrote %s  (%d rows)' % (path, len(rows)))

    # The two arms must differ in the user turn and in nothing else.
    for a, b in zip(rows_pocket, rows_blind):
        assert a['pair'] == b['pair'] and a['split'] == b['split']
        assert a['messages'][0]['content'] == b['messages'][0]['content']
        assert a['messages'][2]['content'] == b['messages'][2]['content']

    # The groundedness CEILING. The chair's sentences are the best achievable answers, and they are
    # only ~0.89 grounded: roughly one named residue in ten is one the ANALOGUE's crystal reaches
    # and the input's never did, which is unknowable from MODEL_INPUT by construction. A model
    # scoring 0.89 here is at the ceiling, not 11% hallucinatory, so the metric is meaningless
    # without this number printed next to it.
    def ceiling(pids):
        micro_h = micro_n = 0
        macro, full = [], 0
        for p in pids:
            tbl = table_residues(recs[p]['MODEL_INPUT'])
            named = residue_tokens(labels[p]['final'])
            if not named:
                continue
            h = sum(1 for n in named if n in tbl)
            micro_h += h
            micro_n += len(named)
            macro.append(h / len(named))
            full += (h == len(named))
        return {'n_sentences_naming_a_residue': len(macro),
                'micro': round(micro_h / micro_n, 4) if micro_n else None,
                'macro': round(statistics.mean(macro), 4) if macro else None,
                'fully_grounded_frac': round(full / len(macro), 4) if macro else None,
                'residues_named_per_sentence': round(micro_n / len(macro), 2) if macro else None}

    sizes = sorted((len(comp[k]) for k in val_keys), reverse=True)
    stats = {
        'reference_groundedness_ceiling': {'train': ceiling(tr), 'val': ceiling(va),
                                           'all_trainable': ceiling(trainable)},
        'n_labels': len(labels),
        'verdicts': dict(collections.Counter(v['verdict'] for v in labels.values())),
        'n_trainable': len(trainable),
        'n_excluded_reject': len(rejected),
        'n_components_trainable': len(comp),
        'n_components_val': len(val_keys),
        'train': {'pairs': len(tr), 'targets': len(t_tr), 'pdbs': len(p_tr)},
        'val': {'pairs': len(va), 'targets': len(t_va), 'pdbs': len(p_va),
                'frac_pairs': round(len(va) / len(trainable), 4),
                'component_sizes': sizes,
                'largest_component_share_of_val': round(sizes[0] / len(va), 4) if va else 0},
        'target_overlap': 0,
        'pdb_overlap': 0,
        'input_smiles_overlap': len(smi_overlap),
        'input_smiles_overlap_examples': smi_overlap[:5],
        'val_targets': sorted(t_va),
        'seed': SEED,
        'val_fraction_requested': VAL_FRACTION,
        'max_component_share_of_val': MAX_COMPONENT_SHARE_OF_VAL,
    }
    path = os.path.join(OUT_DIR, 'split_stats.json')
    json.dump(stats, open(path, 'w'), indent=2)
    print('wrote %s' % path)
    print(json.dumps({k: v for k, v in stats.items() if k != 'val_targets'}, indent=2))
    return stats


def token_stats():
    """Report rendered-prompt token lengths with the real tokenizer, so max_len is not a guess."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    out = {}
    for name in ('sft_pocket.jsonl', 'sft_blind.jsonl'):
        path = os.path.join(OUT_DIR, name)
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r['split'] in ('train', 'val')]
        per = []
        for r in rows:
            m = r['messages']
            prompt = chat_ids(tok, m[:2], add_generation_prompt=True)
            full = chat_ids(tok, m)
            per.append((len(prompt), len(full), max(0, len(full) - len(prompt))))
        p = sorted(x[0] for x in per)
        f = sorted(x[1] for x in per)
        c = sorted(x[2] for x in per)

        def q(v, frac):
            return v[min(len(v) - 1, int(frac * len(v)))]
        out[name] = {
            'n': len(per),
            'prompt_tokens': {'mean': round(statistics.mean(p), 1), 'median': q(p, .5),
                              'p95': q(p, .95), 'p99': q(p, .99), 'max': p[-1]},
            'full_tokens': {'mean': round(statistics.mean(f), 1), 'median': q(f, .5),
                            'p95': q(f, .95), 'p99': q(f, .99), 'max': f[-1]},
            'completion_tokens': {'mean': round(statistics.mean(c), 1), 'median': q(c, .5),
                                  'p95': q(c, .95), 'max': c[-1]},
        }
    path = os.path.join(OUT_DIR, 'token_stats.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out, indent=2))
    print('wrote %s' % path)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-stats', action='store_true',
                    help='re-read the built JSONL and report token lengths (needs the tokenizer)')
    a = ap.parse_args()
    if a.token_stats:
        token_stats()
    else:
        build()
