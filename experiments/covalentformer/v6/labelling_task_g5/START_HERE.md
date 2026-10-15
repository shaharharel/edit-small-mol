# Kick-off prompt

Paste everything below the line to the agent that will run this task.

---

You are running a crystallographic labelling task. Everything you need is in this folder; there is
no database, no external corpus and no network dependency.

## Project context

We are building a training corpus for a **pocket-aware molecular generator with reasoning**. Each
example is a matched pair of X-ray crystal structures: the same protein with two analogous ligands,
where one ligand is a small chemical edit of the other. For each pair we need one sentence
explaining **what that edit does to the protein binding site** — the design rationale a medicinal
chemist would give for making the change.

These sentences are the training target. A model learns from them to propose an analogue *and*
explain why. So the failure that matters is not an ugly sentence, it is a **plausible sentence that
is chemically false** — that teaches the model to fabricate. We have measured this: an earlier
generation of labels cited distances from the product crystal, and a model trained on them emitted
the token "3.45 Å" in 16 of 40 generated rationales against 1 of 40 in the human labels, as a canned
placeholder dressed up as a measurement. Three medicinal chemists reviewing those outputs judged
2 of 40 to describe the real edit. The rules in this task exist to prevent exactly that, and the
most important of them is rule 1b.

## What to do

**Read `README_G5.md`, then `PIPELINE.md` in full first. It is the authority; this prompt is only the orientation.**

Then, for each of the 13 batches in `batches/`:

1. Run **three labellers independently** on `batches/batch_NNNN.txt`. Each reads
   `briefs/REVIEWER.md` in full and writes `label_out/raw/batch_NNNN_lab{A,B,C}.json`.
   They must not read each other's output or anything in `label_out/` — their disagreement is the
   signal the chair uses, and in our runs the three named *different* residues on 43 of 50 pairs.
2. When all three files exist, run **one chair** on that batch. It reads `briefs/CHAIR_FULL.md`,
   then `briefs/REVIEWER.md`, then the evidence, then all three readings, and writes
   `label_out/batch_NNNN.json`.

That is 4 agents per batch, 52 total, 1,249 pairs. Expect roughly 30–40 minutes per agent on a
100-pair batch. Run batches in parallel as far as your concurrency allows; the only ordering
constraint is that a batch's three labellers finish before its chair starts.

## The five things that actually go wrong

1. **The key must be the complete pair id.** Everything between `PAIR ` and the two spaces before
   `target`, including the ligand codes: `7BAA T5Z->T6N`, never `7BAA`. A bare PDB code joins to
   nothing and silently loses the whole batch.
2. **Every pair gets a non-empty sentence, including REJECT.** REJECT means "do not train on this
   as a positive example"; the verdict is a separate column. On a REJECT write the most defensible
   thing the evidence still supports, however thin.
3. **Nothing is dropped.** All 1,249 pairs must appear in the output. No skipping, no omitted keys.
4. **A wrong-shape sentence is a REWRITE, not a REJECT.** REJECT is about the *pair* — evidence that
   cannot support any sentence. Never about the form of a sentence.
5. **Never cite a measured distance from the product structure.** Labellers can see it and should
   use it to know what happened, but the sentence must be one a chemist could have written before
   that structure existed, because at inference the model has none. Direction is predictable;
   magnitude is not. `REVIEWER.md` rules 1 and 1b state this precisely.

## Checkpointing and interruption

**The files on disk are the checkpoint.** A labeller writes its file as its last action. If a run is
interrupted, do not restart from the beginning: recompute which batches have three labeller files
and no chair file, and run only those chairs. **Never re-run a labeller whose file already exists** —
labellers are the expensive part and chairs are cheap. `PIPELINE.md` §6 has the recovery loop.

## When you are done

```
python tools/verify_output.py        # completeness gate -- exit 0 is the definition of done
python tools/check_label_rules.py    # rule compliance, reported as columns, never a gate
```

`verify_output.py` checks all 1,249 keys are present exactly once, no key is a bare PDB code, every
`final` is non-empty and ≤35 words, and every batch has its three labeller files.

For calibration against our previous rounds: expect **85–87% usable** (ADOPT+REWRITE), mean ~32
words, and aim for the **g2 column** in the compliance table in `PIPELINE.md` §7 — 90%+ on all six
hard rules. That is achievable from the brief alone; our labellers hit it on the first batch.

## What not to do

- Do not modify `briefs/`, `batches/` or `batch_manifest.json`. They are the fixed inputs.
- Do not drop, filter or deduplicate pairs, however unpromising one looks. Flag it in the verdict.
- Do not try to satisfy `check_label_rules.py` mechanically at the cost of a defensible sentence.
  If the honest sentence breaks a rule, write the honest sentence and say so in `why`.
- Do not have labellers read one another's work.

Report back: the counts by verdict, `verify_output.py` output, the rule-compliance table, and any
systematic problem you hit in the evidence itself — the packets have known gaps and we want them.
