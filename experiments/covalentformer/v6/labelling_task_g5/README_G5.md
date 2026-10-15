# g5 — read this before START_HERE.md

Same task, same briefs, same pipeline as the 7,069-pair run you (or your predecessors) just
finished. This is an **extension**, not a re-run: **3,186 pairs in 32 batches**, none of which
appear in that corpus or in any earlier one — checked against both the 7,069 built pairs and the
full 8,359-pair selection list, by canonical SMILES pair and by ligand-code pair.

**Keep all output inside this `g5/` directory.** The batch numbers restart at 0001, and the parent
directory holds a different corpus under the same numbers. A file written one level up overwrites
real work.

## What is the same

- `briefs/REVIEWER.md`, `briefs/CHAIR_FULL.md` — the same two briefs, with the one addition below.
- `PIPELINE.md` — the authority. Read it in full.
- Three independent labellers per batch, then one chair. 4 agents x 32 batches = 128 agents.
- The same evidence packet: input atom table, conserved interactions, analogue contacts, the
  pocket envelope, the `***` pair flags and `!!` row markers.
- `tools/verify_output.py` — run it at the end; it now checks this corpus's 1,249 keys.

## What is new

**Two rules, 1c and 1d in the reviewer brief, with the chair's version in `CHAIR_FULL.md`.** Both
come from two blind medicinal-chemist panels over the previous pass, which agreed on exactly these
two failures and nothing else.

1. **A trailing "toward ⟨residue⟩" clause is only allowed if that residue contacts, in the INPUT
   atom table, an atom the edit actually changed.** This was the most common false statement in
   sentences that broke no other rule, and it was what decided which labelling pass the chemists
   preferred. Drop the clause rather than guess.
2. **Name every site that changed.** These are **same-series analogue pairs, not matched molecular
   pairs** — nothing constrains a pair to a single transformation, and some change two or three
   places at once. Read `site_of_change` before writing.

## What this corpus is — read this, it differs from the last run

Every pair is **two distinct crystal structures of the same protein**. Both depositions carry the
target's UniProt accession, which is checked and never bypassed, so no pair is two different
proteins.

What is **not** guaranteed is that the two structures come from one publication. The earlier
corpus required that; this one deliberately does not, because we are testing whether the
requirement can be replaced by chemical similarity. Most of these depositions carry no citation at
all, so for most pairs the question cannot be settled either way.

Rather than assert a provenance we cannot check, the selection threshold was **calibrated** on the
pairs where provenance IS verifiable — both ligands appearing in cited structures:

| P(same paper) | any scaffold | same Murcko scaffold |
|---|---|---|
| Tanimoto ≥ 0.60 | **~78%** ← this corpus | 81% |
| Tanimoto ≥ 0.65 | 79.7% | 83.9% |
| Tanimoto ≥ 0.70 | 81.4% | 85.0% |

So these pairs are drawn from a population that is one series roughly **four times in five**.
Expect **about one pair in five to be two separate experiments** rather than one series.

No scaffold filter was applied, deliberately: the corpus you already labelled shares an identical
Murcko scaffold on only **41.1%** of its pairs, and this one sits at **38.6%** — so the edit mix
here matches what you have already seen, rather than being skewed toward substituent-only changes.

**18.4% of pairs carry `PAIR FLAGGED UNLABELLABLE`** with a reason, up from 9% last run. That rise
is the cost of dropping the same-paper rule, and it is exactly the number this corpus exists to
measure. As always: write the most defensible sentence the evidence still supports and use the
verdict column to exclude — never an empty sentence.

**What that means when you write.** Treat a difference between the two crystals as possibly a
difference between two experiments, not necessarily a consequence of the edit. The packet's
caveats already carry this — `NO SHARED RESIDUE NUMBERING`, `HETERO PARTNER RENUMBERED`, the
conserved-row markers — and here they deserve more weight, not less. Where the two crystals
disagree in a way the edit cannot explain, **say so in `why`** and let the verdict column carry
it. That disagreement is a result we want, not a labelling failure: this corpus exists to measure
how often dropping the same-paper rule costs us a usable label.

`example_output/` is carried over unchanged and shows the file format using a batch number this
corpus does not have. It is a format example only.
