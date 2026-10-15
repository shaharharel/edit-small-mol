# g5 — read this before START_HERE.md

Same task, same briefs, same pipeline as the 7,069-pair run you (or your predecessors) just
finished. This is an **extension**, not a re-run: 1,249 new pairs that were not in that corpus and
are not in any earlier one. Nothing here re-labels anything already labelled.

**Keep all output inside this `g5/` directory.** The batch numbers restart at 0001, and the parent
directory holds a different corpus under the same numbers. A file written one level up overwrites
real work.

## What is the same

- `briefs/REVIEWER.md`, `briefs/CHAIR_FULL.md` — the same two briefs, with the one addition below.
- `PIPELINE.md` — the authority. Read it in full.
- Three independent labellers per batch, then one chair. 4 agents x 13 batches = 52 agents.
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

## What this corpus is

1,249 pairs, each two **distinct crystal structures of the same protein** — both depositions carry
the target's UniProt accession, which is checked and never bypassed — filtered to **Tanimoto ≥
0.70** between the two ligands with no per-target cap. Mean TC 0.815 against the earlier corpus's
0.653, mean heavy-atom change 2.2, so expect smaller and more local edits and fewer scaffold
rebuilds.

**Unlike the earlier corpus, most of these are not from one publication.** 226 of the 1,249 come
from a shared paper; the other 1,023 are same-protein pairs whose depositions carry no citation in
the PDB, so we cannot show they are one series. Same protein is guaranteed; same paper, same lab
and same assay are not. Practical consequence for you: treat a difference between the two crystals
as possibly a difference between two experiments. The packet's caveats already carry this —
`NO SHARED RESIDUE NUMBERING`, `HETERO PARTNER RENUMBERED`, the conserved-row markers — and on this
corpus they deserve more weight than they did last time, not less.

6.9% of pairs are flagged `PAIR FLAGGED UNLABELLABLE` with a reason. As before: write the most
defensible sentence the evidence still supports, and use the verdict column to exclude — never an
empty sentence.

`example_output/` is carried over unchanged and shows the file format using a batch number this
corpus does not have. It is a format example only.
