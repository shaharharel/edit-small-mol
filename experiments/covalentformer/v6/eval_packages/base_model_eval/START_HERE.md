# Kick-off prompt — base-model evaluation

Paste everything below the line to the agent that will run this task.

---

You are running a **blind model-evaluation task**, not a labelling task. Everything you need is in
this folder. Nothing you write becomes training data — you are scoring untrained ("base") models
so a team can fill two columns of a results table.

## What this is

Three off-the-shelf models were each shown a ligand that was crystallised in a protein pocket,
together with that ligand's measured contact table, and asked to **propose an analogue and explain
in one sentence what the change does to the binding site**. None of them was fine-tuned for this.
Expect weaker output than you may have seen in other tasks here — that is the point; we need an
honest floor to compare trained models against.

**Each item shows ONE proposal.** You are not comparing models against each other. Score each item
on its own terms. Which model produced which item is hidden from you and we hold the key.

## What to do

1. Read this file, then work through the 3 files in `batches/` (16 items each, 48 total).
2. **Three reviewers independently per batch.** They must not read each other's output.
3. Each reviewer writes `score_out/batch_NN_rev{A,B,C}.json`.

That is 9 reviewer-runs, 48 items, 144 judgements.

## The question

**Not** "is this the analogue the crystallographer actually made" — many analogues are valid and
we are not testing that. The question is **self-consistency and groundedness**: does the rationale
truthfully describe the edit from the input to *this proposal's own molecule*, using only contacts
the input table supports?

Per item, score five booleans plus a one-line reason:

| field | true when |
|---|---|
| `edit_identified_correctly` | Compare the INPUT SMILES with the proposal's OWN molecule. Does the rationale describe the change that actually separates them? Describing the removal of a group present in both, or ignoring the group that really changed, is **false**. |
| `atom_attribution_ok` | Every distance or contact cited is a real row of the input contact table **AND** attached to the atom that owns it. |
| `invented_chemistry` | **true = bad.** It asserts a contact, bond or interaction the table does not support. |
| `chemically_sensible_edit` | The proposed molecule is a reasonable medicinal-chemistry analogue of the input — not a different scaffold, not nonsense, and it would survive in water. |
| `would_show_a_chemist` | You would put this molecule-plus-rationale in front of a medicinal chemistry audience. |

## What earlier panels found — this is the job

Across ~600 cited distances checked by six reviewers on *trained* models, almost none were
fabricated, but **about 28–30% were real table rows attached to the wrong atom** — usually a
contact belonging to a group the edit retained. A fluent sentence citing a genuine 3.45 Å contact
is still false if that contact is not owned by an atom the edit touched.

So: **look up every cited number in the table and check whose atom it is.** Recover atom numbering
where you can (bonded C–O, C–F, C–Cl pairs share a nearest protein atom, which pins assignments
down). Use RDKit for atom counts, ring sizes and stereocentres — several past failures described a
single-point change on a molecule that had also moved a halogen or changed a ring size.

Also watch for: claiming a contact is "retained" when the proposal deletes the atom that makes it;
a packing contact against a side-chain carbon (CB, CG, CD2, CE) called a hydrogen bond, which a
carbon cannot accept; protons that cannot exist, such as the NH of an N,N-dialkylamide; and
residues that appear nowhere in the table.

## Rules

1. **Score every item.** A skipped item is indistinguishable from one scored false.
2. **Never assert a number you did not verify.** If the table will not settle ownership, say so in
   `one_line_why` and score conservatively.
3. Three independent reviewers, no peeking at each other.
4. There is no answer key in this folder. Do not go looking for one.
5. Some of these rationales are long and rambling, and one model's outputs average far more words
   than the others. Length is not quality — judge the content against the table.

## Output

`score_out/batch_NN_rev{A,B,C}.json` — one JSON object keyed by `item_id` (the `item_id` on the
ITEM line), each value:

```json
{"edit_identified_correctly": false, "atom_attribution_ok": false,
 "invented_chemistry": true, "chemically_sensible_edit": true,
 "would_show_a_chemist": false, "one_line_why": "..."}
```

Verify each file parses before finishing the batch.
