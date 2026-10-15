# CovalentFormer v7 — labelling task

Self-contained. Everything needed to label 1,249 crystallographic analogue pairs is in this folder.
No database, no external corpus, no network. **Read this file end to end before starting.**

---

## 1. What you are producing

For each analogue pair, **one sentence** (max 35 words) saying what the molecular edit does to the
protein binding site. These sentences train a pocket-aware generator to explain *why* an edit was
made, so a sentence that reads well and is chemically wrong is worse than useless — it teaches the
model to fabricate.

The corpus is 1,249 pairs in 13 batches of 100 (batch 13 has 49). Each batch is a self-contained
text file in `batches/`.

## 2. The pipeline, per batch

```
batch_NNNN.txt
      |
      +--> labeller A  ---+
      +--> labeller B  ---+--> CHAIR --> label_out/batch_NNNN.json   (the deliverable)
      +--> labeller C  ---+
           (independent)        (adjudicates)
```

**Three labellers, then one chair, per batch. 4 agents × 13 batches = 284 agents.**

### The three labellers

Each reads `briefs/REVIEWER.md` in full, then `batches/batch_NNNN.txt`, and writes one sentence per
pair into `label_out/raw/batch_NNNN_labA.json` (or `_labB`, `_labC`).

**They must be independent.** A labeller must not read the other two labellers' files, and must not
read anything in `label_out/`. Disagreement between the three is the signal the chair uses; if they
converge by peeking, that signal is destroyed and the chair has nothing to adjudicate. In our runs
the three named *different* residues on 43 of 50 pairs, and that disagreement is where the chair
found the errors.

### The chair

Reads `briefs/CHAIR_FULL.md` in full, then `briefs/REVIEWER.md` (it is held to the same rules), then
the batch evidence, then all three labeller files. It adjudicates **every** pair and writes
`label_out/batch_NNNN.json`.

The chair is not a vote-counter. Disagreement usually means one labeller saw something the other two
missed, so it checks who is right **against the evidence**, not against the majority.

## 3. Output format

### `label_out/raw/batch_NNNN_lab{A,B,C}.json`

```json
{
  "7BAA T5Z->T6N": {
    "sentence":   "<= 35 words",
    "basis":      "the specific rows used: WHICH TABLE each fact came from, atom, residue, distances",
    "confidence": "high|medium|low",
    "concern":    "only if something in the evidence is unsafe or contradictory"
  }
}
```

### `label_out/batch_NNNN.json` — the deliverable

```json
{
  "7BAA T5Z->T6N": {
    "verdict":      "ADOPT|REWRITE|REJECT",
    "final":        "<ALWAYS non-empty, <= 35 words>",
    "why":          "which reading you took and why, or what you repaired",
    "disagreement": "what the three differed on, and who was right"
  }
}
```

See `example_output/` for real files from an earlier round.

### THE KEY IS THE COMPLETE PAIR ID

Everything between `PAIR ` and the two spaces before `target`, **including the ligand codes**.

```
PAIR 7BAA T5Z->T6N   target P00918
          ^^^^^^^^^^^^^^^^^^
key = "7BAA T5Z->T6N"      NOT "7BAA"
```

A key of the PDB code alone cannot be joined to anything and loses the entire batch. This has
nearly happened to us; take it seriously.

## 4. Four rules that are not negotiable

**(a) Every pair gets a non-empty sentence — including REJECT.** A REJECT means "do not train on
this pair as a positive example", and the verdict is a separate column. An empty sentence destroys
information that cannot be cheaply recreated. On a REJECT, write the most defensible thing the
evidence still supports, however thin.

**(b) Nothing is dropped.** Do not delete a pair, do not skip a pair, do not omit a key. All 1,249
pairs must appear in the output. REJECTs are kept with their verdict.

**(c) The sentence is input-anchored.** This is the heart of the task and `REVIEWER.md` rules 1 and
1b state it precisely:

| half | side | status | contains |
|---|---|---|---|
| **ANCHOR** (required) | input | **MEASURED** | input-pocket residues and the input's own distances |
| **CONSEQUENCE** (required) | analogue | **QUALITATIVE** | direction, kind, effect — **and no numbers** |

Labellers *can see* the analogue's crystal and should use it to know what happened. But the sentence
must be one a chemist could have written **before** that structure existed, because at inference the
model has no analogue structure and cannot measure anything. Direction is predictable; magnitude is
not. **Never cite a measured analogue distance.**

Why this matters, measured: an earlier generation of labels did cite analogue distances, and a model
trained on them emitted `3.45 Å` in 16 of 40 sentences against 1 of 40 in the human labels — a
canned token dressed as a crystallographic measurement. Three medicinal chemists reviewing those
outputs judged 2 of 40 to describe the real edit. That failure is what rule 1b exists to prevent.

**(d) A wrong-shape sentence is a REWRITE, not a REJECT.** REJECT is about the *pair* — evidence that
cannot support any sentence. It is never about a sentence being in the wrong form. Repair it.

## 5. The pocket envelope — new, and read `REVIEWER.md` on it

Every packet carries a `pocket envelope` block: every residue with an atom within 8 Å of the input
ligand, in distance **bands**, whether or not it is any atom's nearest neighbour.

It exists because the atom table lists only each ligand atom's *nearest* protein atom, so a residue
4 Å away that is nobody's nearest neighbour appears nowhere in it. Measured: 24% of an earlier
round's sentences named a residue the input side could not express, and an 8 Å envelope recovers
94.7% of them.

- You **may name** any residue in the envelope. That is what it is for — saying where a new
  substituent grows toward.
- It is **not a contact list.** A residue at 7.5 Å touches nothing. Never assert contact, a hydrogen
  bond, or packing with a residue found only there.
- You **may not cite a band as a distance.** The bands exist so there is no decimal to copy.
- The flags are a crystal-packing disclosure: `(X)` another chain, `*` cofactor or ion, `~` chain
  terminus and therefore probably a lattice neighbour rather than a site residue.

Validated end to end on 10 pairs chosen so the envelope must matter: labellers used it on 8–9 of 10,
the chair adopted 9 of 10, no band was ever cited as a distance.

## 6. How to run it

Sequentially or in parallel, your choice, but **three labellers must finish a batch before its chair
starts**. Practical notes from running this at scale:

- **Batch files on disk are the checkpoint.** A labeller writes its file as the last step. If a run
  is interrupted, recompute which batches have three labeller files and no chair file, and run only
  those chairs. Never re-run a labeller whose file already exists.
- **Chairs are cheap, labellers are not.** If you must economise, never redo labeller work.
- Expect ~30–40 min per agent on a 100-pair batch.

A chairs-only pass, for recovering from an interruption:

```
for each batch NNNN in 0001..0013:
    if   label_out/raw/batch_NNNN_lab{A,B,C}.json  all exist
    and  label_out/batch_NNNN.json                 does not exist
    then run the chair for NNNN
```

## 7. Checking your output

```
python tools/check_label_rules.py            # rule compliance per label, as COLUMNS
python tools/verify_output.py                # completeness: 1,249 keys, no empties, valid JSON
```

`check_label_rules.py` **is not a gate.** It annotates, always exits 0, and never blocks a batch.
Compliance is a fact about a label, exactly as the REJECT verdict is. Do not sacrifice a defensible
sentence to satisfy it mechanically; if the honest sentence breaks a rule, write the honest sentence
and say so in `why`.

For calibration, the six hard rules on our last two generations of labels:

| rule | | g1 (old style) | g2 (input-anchored) |
|---|---|---|---|
| r1 | all named residues from the input side | 76.1% | **99.1%** |
| r2 | no analogue distance cited | 8.2% | **98.9%** |
| r2b | anchor at the site of change | 87.3% | **92.2%** |
| r3 | no analogue-provenance phrasing | 23.0% | **99.1%** |
| r5 | no banned claims | 100% | 100% |
| r6 | ≤ 35 words | 99.9% | 100% |
| | **all six** | 3.1% | **90.8%** |

**Aim for the g2 column.** It is achievable from the brief alone — our labellers hit it on the first
batch without special coaching, and it cost nothing in usable labels (87.3% usable vs 85.7%).

## 8. What "good" looks like

From a round scored against three chemists: mean 31.5 words, 100% name a residue, 100% cite a
distance, 99% cite both sides, 0 burial claims, 0 room claims, 0 potency claims, ~86% usable
(ADOPT+REWRITE).

```
Good:  "The methyl packing LEU91.CD2 at 3.39 A is replaced by a cyclohexyl ring, which should
        spread that contact over more ring carbons and relax it slightly, while the sulfonamide's
        1.92 A zinc coordination is untouched and should be retained."

Bad:   "...handing LEU91.CD2 from a methyl at 3.39 A to ring carbons at 3.61-3.69 A in the
        analogue's crystal."          <- an analogue measurement; 3.61-3.69 is unknowable
Bad:   "Replaces a methyl with a phenyl to increase lipophilicity."   <- chemical diff, no pocket
Bad:   "Extends into the 6 A channel lined by LEU37/PRO68."           <- room claim, banned
```

## 9. Folder contents

```
PIPELINE.md              this file
briefs/REVIEWER.md       the labeller brief -- rules, legend, worked examples. READ IN FULL.
briefs/CHAIR_FULL.md     the chair brief. READ IN FULL.
batches/batch_NNNN.txt   13 files, 1,249 pairs, all with the pocket envelope
batch_manifest.json      batch name -> list of pair ids, so coverage is checkable
example_output/          real labeller and chair files from an earlier round, for format
tools/                   check_label_rules.py, verify_output.py
label_out/               WRITE YOUR OUTPUT HERE (raw/ for labellers, top level for chairs)
```

## 10. Deliverable

```
label_out/batch_0001.json .. batch_0013.json        13 chair files, 1,249 keys total
label_out/raw/batch_NNNN_lab{A,B,C}.json            213 labeller files
```

`tools/verify_output.py` passing is the definition of done.
