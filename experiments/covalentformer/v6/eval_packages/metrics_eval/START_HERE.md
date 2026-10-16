# Kick-off prompt

Paste everything below the line to the agent that will run this task.

---

You are running a **blind model-evaluation task**, not a labelling task. Everything you need is in
this folder; there is no database, no external corpus and no network dependency. Nothing you write
becomes training data — you are scoring four trained models so a team can decide which to take
forward.

## What to do

**Read `README_METRICS_EVAL.md` in full first. It is the authority; this prompt is only the
orientation.**

Then, for each of the 7 batches in `batches/`:

1. Run **three reviewers independently** on `batches/batch_NN.txt`. Each reads
   `README_METRICS_EVAL.md` in full, then scores every case, and writes
   `score_out/batch_NN_rev{A,B,C}.json`.
   They must not read each other's output — their disagreement is signal the analysis uses.
2. Move to the next batch.

That is 3 agents per batch, 21 total, 165 cases, 660 proposals, 1,980 slot judgements.

## The task in one paragraph

Each case gives an input ligand, its measured contact table from the crystal, and **four proposed
analogues labelled A–D, each with its own one-sentence rationale**. For each of A, B, C and D,
decide whether the rationale truthfully describes the edit from the input to **that slot's own
molecule**, and whether every contact it cites is a real table row attached to the right atom.
Score five booleans per slot plus a one-line reason, and pick `best_overall` for the case.

The letters are reshuffled in every case. Do not try to infer which model is which.

## The one thing that matters most

Earlier panels checked ~600 cited distances across six reviewers. Almost none were invented —
**but about 28–30% were real numbers attached to the wrong atom**, nearly always a contact
belonging to a group the edit kept. A fluent sentence citing a genuine 3.45 Å contact is still
false if that contact is not owned by an atom the edit touched.

So: **look up every cited number in the table and check whose atom it is.** Recover the atom
numbering where you can (bonded C–O, C–F and C–Cl pairs share a nearest protein atom, which pins
assignments down), and say in `one_line_why` when that is what settled a call. Use RDKit to check
atom counts, ring sizes and stereocentres — several past failures were rationales describing a
single-point change on a molecule that had also silently moved a halogen or changed a ring.

## Hard rules

1. **Score every slot on every case.** A skipped slot is indistinguishable from a false one, and
   these become rates in a results table.
2. **Never guess a number you did not verify.** If the table does not let you settle ownership,
   say so in `one_line_why` and score conservatively.
3. **Three independent reviewers, no peeking.**
4. There is no answer key in this folder. Do not go looking for one.

## Output format

`score_out/batch_NN_rev{A,B,C}.json` — one JSON object keyed by the case id (the `id` on the CASE
line), each value:

```json
{
  "A": {"edit_identified_correctly": true, "atom_attribution_ok": false,
        "invented_chemistry": false, "chemically_sensible_edit": true,
        "would_show_a_chemist": false, "one_line_why": "..."},
  "B": {...}, "C": {...}, "D": {...},
  "best_overall": "C"
}
```

Verify each file parses before finishing the batch.
