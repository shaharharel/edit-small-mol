# Metrics evaluation task — four models, 165 cases, blind

This is **not** a labelling task. Nothing you write here becomes training data. You are scoring
model output so we can decide which of four trained models to take forward.

Read this file, then `START_HERE.md`. Do not skip to the batches.

## What you are scoring

Four models were each asked the same thing: given a ligand that was crystallised in a protein
pocket, and a table of that ligand's measured contacts, **propose an analogue and explain in one
sentence what the change does to the binding site.**

Each case in `batches/` shows:

- the input ligand's SMILES,
- the input crystal's contact table — for each ligand atom, its nearest protein atom and the
  measured distance,
- **four proposals labelled A, B, C, D.** Each is one model's own proposed molecule together with
  its own rationale.

The four are the same four models in every case, but **the letters are reshuffled case by case**,
so A in case 1 is not A in case 2. You cannot and should not try to track "which model is which" —
that is the point, and we hold the key.

## The question you are answering

**Not** "is this the analogue the crystallographer actually made". Many analogues are valid and we
are not testing that. The question is **self-consistency and groundedness**: does the rationale
truthfully describe the edit from the input to *that proposal's own molecule*, using only contacts
the input table actually supports?

For each of A, B, C, D, score five booleans:

| field | true when |
|---|---|
| `edit_identified_correctly` | Compare the INPUT SMILES with **that slot's own molecule**. Does the rationale describe the change that actually separates those two structures? If it describes removing a group present in both, or ignores the group that really changed, this is **false**. |
| `atom_attribution_ok` | Every distance or contact the rationale cites is a real row of the input contact table **AND** attached to the atom that owns it. |
| `invented_chemistry` | **true = bad.** It asserts a contact, bond or interaction the table does not support. |
| `chemically_sensible_edit` | The proposed molecule is a reasonable medicinal-chemistry analogue of the input at all — not a different scaffold, not nonsense, and it would survive in water. |
| `would_show_a_chemist` | You would put this molecule-plus-rationale in front of a medicinal chemistry audience. |

Plus `one_line_why` per slot, and `best_overall` per case (A, B, C, D or none).

## What we have learned from three earlier, smaller panels — read this, it is the job

The failure that matters is **not invented numbers**. Across ~600 cited distances checked by six
reviewers, almost none were fabricated. **About 28–30% were real table rows attached to the wrong
atom.** Overwhelmingly: a contact belonging to a group the edit *retained*, credited to the group
that changed.

So the work is to **look up every cited number in the table and check whose atom it is.** A
sentence that reads perfectly and cites a real 3.45 Å contact is still false if that contact
belongs to an atom the edit never touched. This is invisible without the table in hand, which is
why we are asking people and not a script.

Specific recurring failures worth checking for:

1. **Contact theft from a retained group** — the dominant one, above.
2. **Claiming a contact is "retained" when the proposal deletes the atom that makes it.** In one
   earlier case four of five models deleted a hydroxyl and all still claimed its 2.71 Å hydrogen
   bond survived.
3. **Interaction-type upgrades** — a packing contact against a side-chain *carbon* (CB, CG, CD2,
   CE) described as a hydrogen bond. A carbon cannot accept one.
4. **Protons that cannot exist** — e.g. "the amide NH" of an N,N-dialkylamide.
5. **Silent extra edits** — the rationale describes one single-point change while the molecule also
   moved a halogen, inverted a stereocentre or changed a ring size. Counting atoms and rings with
   RDKit catches these.
6. **Residues that appear nowhere in the table.** Rare but real; three were found in 16 cases.

## Ground rules

- **Score every slot on every case.** A slot you skip is indistinguishable from one you scored
  false, and these are rates in a results table.
- **Three independent reviewers per batch.** Do not read another reviewer's output before writing
  your own — the disagreement is signal we use.
- Judge only what the table supports. Where the packet flags a row as uncertain or a residue as
  suspect, a rationale leaning on it without caveat is a `would_show_a_chemist` problem.
- There is no answer key in this folder. We hold it; it is applied after all scores are in.

## Why 165 and not 200

165 is every case in the benchmark where **all four** models produced both a sentence and a
parseable molecule. Judging all four on identical cases is a paired comparison and is
substantially more sensitive than scoring each model on 200 cases of its own. Going to 200 would
mean unmatched cases and a weaker result.

## Output

For each batch `batches/batch_NN.txt`, each reviewer writes
`score_out/batch_NN_rev{A,B,C}.json`: one JSON object keyed by the case id (the `id` on the CASE
line), each value an object with keys `A`, `B`, `C`, `D` — each holding the five booleans and
`one_line_why` — plus `best_overall`. Verify it parses before you finish.

7 batches × 25 cases (the last has 15) × 3 reviewers = 21 reviewer-runs, 1,980 slot judgements.
