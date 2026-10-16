#!/usr/bin/env python3
"""Turn the 1,000 chair-adjudicated labels into Alpaca-format SFT data, pocket-aware and blind.

Two JSONL files come out of one pass so the ablation is guaranteed to differ in exactly one thing:

    sft_pocket.jsonl   instruction + (target, input SMILES, contact table) -> sentence + SMILES
    sft_blind.jsonl    instruction + (target, input SMILES)                -> sentence + SMILES

Same rows, same ids, same split assignment, same completion, byte-identical instruction. If the
blind run matches the pocket run then the contact table carried no signal and the experiment has
demonstrated nothing, which is the one result this pipeline exists to be able to detect. Anything
else that differed between the two files (a reshuffled split, a different seed, a stray extra field)
would confound that comparison, so both files are emitted together and never built separately.

ALPACA FORMAT, FOR ALL FOUR ARMS -- DO NOT "FIX" THIS TO PER-MODEL CHAT TEMPLATES
---------------------------------------------------------------------------------
SmileyLlama 1.1 was not tuned with the Llama-3.1 chat template; it was tuned with an Alpaca-style
prompt (`### Instruction:` / `### Input:` / `### Response:`), verified on the box, and under it the
checkpoint emits clean bare SMILES.

Every one of the four arms in the 2x2 -- {SmileyLlama, Llama-3.1-8B-Instruct} x {pocket, blind} --
uses this same Alpaca format, including the two plain-instruct arms. Giving each checkpoint its
native template would confound the template with the model, and the row comparison ("what does the
ChEMBL pretraining buy?") would no longer be a single-variable read. Llama-3.1-8B-Instruct handles
Alpaca prompts perfectly well. The format is held fixed on purpose; the only thing that varies
between columns is whether the contact table is present in the `### Input:` block, and the only
thing that varies between rows is which checkpoint the LoRA starts from.

The instruction line is adapted to this task rather than copied from the SmileyLlama paper, and is
identical across all four arms.

THE SPLIT IS TWO-WAY, AND THE NUMBER IT YIELDS IS A MODEL-SELECTION SCORE
------------------------------------------------------------------------
This 860-example run is a behaviour probe, not a result; the real run is the full ~7,000 labels.
So the split is two-way, train/val (~85/15) by protein target, and the stopping epoch and
checkpoint are selected on val.

That makes every val number reported from this run a MODEL-SELECTION SCORE, NOT A HELD-OUT TEST
SCORE. It is optimistically biased by an unknown amount, because the thing being reported is the
thing the checkpoint was chosen on. It must be labelled that way wherever it is quoted. The clean
train/val/test discipline starts with the 7k run: `split_pairs()` therefore takes a `mode` of
'two_way' or 'three_way' already, so nobody has to rediscover why it is needed.

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
# Loaded by GLOB, with the generation as an explicit filter. The 50 re-adjudicated pairs were
# merged back into batches 0001-0010 and the 9001/9002 files moved to label_out/archive_readjudicated/
# (outside this glob), so one record per pair is now the data model and a plain glob is correct.
#
# The generation filter is NOT redundant. Batches 0021-0040 are being labelled right now under the
# v2 input-anchored rules, and they will land in this same directory within hours. DATASET.md: g1
# and g2 obey DIFFERENT sentence rules, and a model trained across both learns to emit invented
# precision that no metric computed afterwards can separate back out. Filtering by generation means
# their arrival cannot change this dataset -- which a bare glob has silently done twice already
# (1,000 -> 1,800 pairs once, and 2,000 records that were really 1,950 unique pairs plus 50
# superseded readings the other time).
GENERATION = 'g1_v1_retrospective'
# g4 re-labels the SAME 7,069 pairs under the SAME batch numbers in a SEPARATE directory, so a
# glob spanning both label_out/ and label_out_g4/ would hand two records to one pair id and break
# the one-record-per-pair data model this file rests on. The directory is therefore selected, not
# merged: COVF_LABEL_DIR picks it, and the default is unchanged so every g1-g3 build stays
# byte-reproducible.
LABEL_GLOB = os.path.join(HERE, os.environ.get('COVF_LABEL_DIR', 'label_out'), 'batch_*.json')
N_PAIRS_EXPECTED = 2000
N_USABLE_EXPECTED = 1714

OUT_DIR = os.path.join(HERE, 'sft_out')
MODEL_ID = 'THGLab/Llama-3.1-8B-SmileyLlama-1.1'

VAL_FRACTION = 0.15
MAX_COMPONENT_SHARE_OF_VAL = 0.20
SEED = 0

_AA = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR '
                'VAL MSE SEC PYL HID HIE HIP'.split())
_RESTOK = re.compile(r'\b([A-Z]{2,4})(\d+)\b')


# --------------------------------------------------------------------------------------- prompts

# Identical across all four arms of the 2x2. Only the `### Input:` block differs between columns.
INSTRUCTION = (
    "You are a medicinal chemist proposing a single-step analogue of a ligand whose crystal "
    "structure in its target binding site has been measured.\n"
    "You are given the ligand's SMILES and, where available, a table of its measured protein "
    "contacts: for each ligand heavy atom, the nearest protein atom it faces, the distance, the "
    "angle where one is defined, and the interaction typing.\n"
    "Propose one analogue and state what the change does to that binding site, in a single "
    "sentence.\n"
    "Name only residues that appear in the contact table you were given. Where a contact is "
    "flagged uncertain, do not cite it as established. Then give the analogue's SMILES on its own "
    "line.\n"
    "Answer with the sentence, then a line of the form 'SMILES: <smiles>'."
)

ALPACA = '### Instruction:\n%s\n\n### Input:\n%s\n\n### Response:\n'


def alpaca_prompt(input_block):
    """The full prompt text. Training masks everything up to and including '### Response:\\n'."""
    return ALPACA % (INSTRUCTION, input_block)


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


def choose_val(comp, n_pairs, seed=SEED, mode='two_way', frac=VAL_FRACTION):
    """Draw whole components for validation up to ~frac, capping any single component.

    `mode` exists for the 7k run: 'two_way' returns (val, n) as here; 'three_way' is the discipline
    the real run needs, where val selects checkpoints and test is reported once at the end. It is
    deliberately left unimplemented rather than faked, so the 7k run has to make the choice
    explicitly instead of inheriting this probe's two-way shortcut by accident.
    """
    if mode == 'three_way':
        raise NotImplementedError(
            'three_way split is for the 7k run: draw val and test as disjoint component sets, '
            'assert target and PDB disjointness pairwise across all three, and report test once.')
    import random
    rng = random.Random(seed)
    budget = frac * n_pairs
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

def load(generation=GENERATION, n_pairs_expected=None, n_usable_expected=None):
    """Load the labels for one or more generations.

    `generation` may be a single generation id or a comma-separated list. A MIXED set is allowed
    only when it is explicitly requested, which is the point: DATASET.md's warning is that g1 and
    g2 obey different sentence rules and a model trained across both learns to emit invented
    precision no later metric can separate out. Accident is what the assertion below forbids;
    a deliberate, named request is a decision someone made in the open.
    """
    wanted = {g.strip() for g in generation.split(',')} if isinstance(generation, str) else set(generation)
    files = sorted(glob.glob(LABEL_GLOB))
    assert files, 'no label files matched %s' % LABEL_GLOB
    n_records = 0
    labels = {}
    for f in files:
        d = json.load(open(f))
        n_records += len(d)
        for pair, rec in d.items():
            if rec.get('label_generation') in wanted:
                labels[pair] = rec
    # ONE RECORD PER PAIR. This invariant was violated once -- 2,050 records for 2,000 pairs, where
    # a loader pinned to a batch range silently trained on the 50 SUPERSEDED readings, two of which
    # assert a covalent bond loss that is probably a bond-perception failure. The check is a line of
    # code; discovering it in a result costs a re-run of everything.
    gen_records = sum(1 for f in files for pair, rec in json.load(open(f)).items()
                      if rec.get('label_generation') in wanted)
    assert gen_records == len(labels), \
        'DUPLICATE PAIRS: %d records for %d unique pair ids in generation %s' % (
            gen_records, len(labels), sorted(wanted))
    exp = N_PAIRS_EXPECTED if n_pairs_expected is None else n_pairs_expected
    assert len(labels) == exp, \
        'expected %d pairs for %s, got %d (of %d records across %d files)' % (
            exp, sorted(wanted), len(labels), n_records, len(files))
    # One generation per training set, enforced rather than hoped for. DATASET.md: g1 and g2 obey
    # DIFFERENT sentence rules, and a model trained across both learns to emit invented precision
    # some fraction of the time, which no metric computed afterwards can separate back out. A mixed
    # set must therefore be a deliberate decision made somewhere visible, not an accident of which
    # batch numbers happened to be on disk.
    gens = collections.Counter(v.get('label_generation') for v in labels.values())
    assert set(gens) == wanted, (
        'loaded generations %r but %r were requested -- see DATASET.md; a mixed set must be '
        'asked for by name, never arrived at by accident' % (dict(gens), sorted(wanted)))
    if len(gens) > 1:
        print('MIXED GENERATIONS, requested explicitly: %s' % dict(gens))
    # `previous_adjudication` on the 50 re-adjudicated records is PROVENANCE, not an alternative
    # target: the superseded sentence. Training reads `final` only, and this asserts the merge
    # actually replaced the reading rather than leaving the old one in place (0 of 50 came back
    # identical upstream, so any match here means the merge did not take).
    n_readj = 0
    for pair, rec in labels.items():
        if 'previous_adjudication' in rec:
            n_readj += 1
            prev = rec['previous_adjudication']
            prev_sent = prev.get('final') if isinstance(prev, dict) else None
            assert not prev_sent or prev_sent.strip() != (rec.get('final') or '').strip(), \
                'pair %s still carries its superseded sentence as `final`' % pair
    print('loaded %d pairs for %s from %d files (%d records); %d re-adjudicated'
          % (len(labels), sorted(wanted), len(files), n_records, n_readj))
    recs = {}
    with open(CORPUS) as fh:
        for line in fh:
            r = json.loads(line)
            if r['pair'] in labels:
                recs[r['pair']] = r
    missing = [k for k in labels if k not in recs]
    assert not missing, 'labels with no corpus record: %r' % missing[:5]
    return labels, recs


def build(generation=GENERATION, out_dir=None, n_pairs_expected=None,
          n_usable_expected=None):
    out_dir = out_dir or OUT_DIR
    labels, recs = load(generation, n_pairs_expected=n_pairs_expected)
    trainable = sorted(k for k, v in labels.items() if v['verdict'] in ('ADOPT', 'REWRITE'))
    rejected = sorted(k for k, v in labels.items() if v['verdict'] == 'REJECT')
    assert len(trainable) + len(rejected) == len(labels)
    _exp_u = N_USABLE_EXPECTED if n_usable_expected is None else n_usable_expected
    assert len(trainable) == _exp_u, \
        'expected %d usable (ADOPT+REWRITE), got %d' % (_exp_u, len(trainable))

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

    os.makedirs(out_dir, exist_ok=True)
    rows_pocket, rows_blind = [], []
    for pid in trainable + rejected:
        r = recs[pid]
        v = labels[pid]
        is_rej = v['verdict'] == 'REJECT'
        split = 'excluded' if is_rej else split_of[pid]
        mi, target = r['MODEL_INPUT'], r['target']
        for pocket, sink in ((True, rows_pocket), (False, rows_blind)):
            input_block = render_prompt(target, mi, pocket=pocket)
            prompt = alpaca_prompt(input_block)
            assert_no_leak(prompt, r)
            completion = ('' if is_rej
                          else render_assistant(v['final'], r['ANSWER']['analogue_smiles']))
            sink.append({
                'pair': pid,
                'target': target,
                'pdb': pid.split()[0],
                'split': split,
                'verdict': v['verdict'],
                'label_generation': v.get('label_generation'),
                'arm': 'pocket' if pocket else 'blind',
                'prompt': prompt,
                'completion': completion,
                'reference_sentence': '' if is_rej else v['final'].strip(),
                'analogue_smiles': '' if is_rej else r['ANSWER']['analogue_smiles'],
                'input_smiles': mi['input_smiles'],
                'n_atom_rows': len(mi['atoms']),
                # Eval metadata. Deliberately outside `messages`, so it cannot reach a prompt.
                'table_residues': sorted(table_residues(mi)),
                'reference_residues': sorted(residue_tokens(v['final'])) if not is_rej else [],
            })

    for name, rows in (('sft_pocket.jsonl', rows_pocket), ('sft_blind.jsonl', rows_blind)):
        path = os.path.join(out_dir, name)
        with open(path, 'w') as fh:
            for row in rows:
                fh.write(json.dumps(row) + '\n')
        print('wrote %s  (%d rows)' % (path, len(rows)))

    # The two arms must differ in the Input block and in nothing else.
    for a, b in zip(rows_pocket, rows_blind):
        assert a['pair'] == b['pair'] and a['split'] == b['split']
        assert a['completion'] == b['completion'], 'completions differ between arms'
        assert a['prompt'].split('### Input:')[0] == b['prompt'].split('### Input:')[0], \
            'instruction differs between arms'
        assert a['prompt'].endswith('### Response:\n') and b['prompt'].endswith('### Response:\n')

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
    gen = next(iter({v.get('label_generation') for v in labels.values()}))
    stats = {
        'label_generation': gen,
        'label_files_globbed': sorted(os.path.basename(f) for f in glob.glob(LABEL_GLOB)),
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
    path = os.path.join(out_dir, 'split_stats.json')
    json.dump(stats, open(path, 'w'), indent=2)
    print('wrote %s' % path)
    print(json.dumps({k: v for k, v in stats.items() if k != 'val_targets'}, indent=2))
    return stats


def load_rows_jsonl(path, arm, split=None):
    """Read one arm's JSONL. Returns (train, val), or just the named split's rows.

    Shared by train_sft.py and eval_sft.py so that the two can never disagree about which rows are
    trainable. The REJECT rows carry split "excluded" and an empty completion; every consumer
    filters on `split`, so they cannot reach training or scoring by accident, and the assertion
    below would fire if one ever did.
    """
    rows = [json.loads(l) for l in open(path)]
    assert all(r['arm'] == arm for r in rows), 'arm mismatch in %s' % path
    ex = [r for r in rows if r['split'] == 'excluded']
    assert ex and all(r['verdict'] == 'REJECT' and not r['completion'] for r in ex), \
        'excluded rows must be the REJECTs, with empty completions'
    if split is not None:
        out = [r for r in rows if r['split'] == split]
        assert all(r['completion'] for r in out), 'empty completion in split %s' % split
        return out
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    assert all(r['completion'] for r in tr + va), 'empty completion in train/val'
    return tr, va


def token_stats(max_length=None, out_dir=None):
    """Report token lengths with the real tokenizer, so max_length is measured and not guessed.

    Both checkpoints share a vocabulary -- vocab_size 128256, len(tokenizer) 128256, 256 added
    tokens, identical vocab dicts, and the same SMILES tokenizes to the same ids on both -- so one
    measurement covers all four arms.

    If any example exceeds `max_length` this REFUSES rather than truncating. Truncation here cuts
    into the contact table, which would silently delete pocket evidence from the longest-pocket
    pairs and weaken precisely the arm the ablation is meant to favour; a corrupted comparison that
    still runs is worse than a loud failure. The ceiling also has to stay well under 8192, because
    SmileyLlama ships rope_scaling factor 1.0 (identity) against the base checkpoint's 8.0, so its
    usable context is 8192 regardless of the 131072 its config advertises.
    """
    from transformers import AutoTokenizer
    out_dir = out_dir or OUT_DIR
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    out = {}
    for name in ('sft_pocket.jsonl', 'sft_blind.jsonl'):
        path = os.path.join(out_dir, name)
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r['split'] in ('train', 'val')]
        per = []
        for r in rows:
            prompt = tok(r['prompt'], add_special_tokens=True)['input_ids']
            comp = tok(r['completion'], add_special_tokens=False)['input_ids']
            full = len(prompt) + len(comp) + 1          # + the pinned EOS appended in training
            per.append((len(prompt), full, len(comp) + 1))
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
        if max_length is not None:
            over = [x for x in f if x > max_length]
            out[name]['n_over_max_length'] = len(over)
            out[name]['max_length_checked'] = max_length
            assert not over, (
                'REFUSING to truncate: %d examples in %s exceed max_length=%d (longest %d). '
                'Raise max_length -- truncating would delete contact-table rows from the '
                'pocket-aware arm and corrupt the ablation.' % (len(over), name, max_length, f[-1]))
    path = os.path.join(out_dir, 'token_stats.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out, indent=2))
    print('wrote %s' % path)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-stats', action='store_true',
                    help='re-read the built JSONL and report token lengths (needs the tokenizer)')
    ap.add_argument('--max-length', type=int, default=None,
                    help='assert nothing exceeds this; refuses rather than truncating')
    ap.add_argument('--generation', default=GENERATION,
                    help='label generation(s) to train on, comma-separated. A MIXED set is allowed '
                         'only when named explicitly here -- see DATASET.md: g1 and g2 obey '
                         'different sentence rules, and mixing them by accident is the failure the '
                         'loader assertion exists to prevent.')
    ap.add_argument('--n-pairs-expected', type=int, default=None,
                    help='assert the loaded pair count. g1 2000, g2 2000, g1+g2 4000.')
    ap.add_argument('--n-usable-expected', type=int, default=None,
                    help='assert the ADOPT+REWRITE count. g1 1714, g2 1747, g1+g2 3461. These '
                         'assertions are the guard against a dataset silently changing size, so '
                         'they are supplied per run rather than removed.')
    ap.add_argument('--out-dir', default=None,
                    help='where to write; use a NEW dir rather than overwriting a set a run is '
                         'currently training on')
    a = ap.parse_args()
    if a.token_stats:
        token_stats(a.max_length, a.out_dir)
    else:
        build(a.generation, a.out_dir, a.n_pairs_expected, a.n_usable_expected)
