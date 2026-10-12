# Chair brief — full run

You adjudicate ONE batch. Three labellers read the same pairs independently and never saw each
other; their readings are in `label_out/raw/batch_<NNNN>_labA.json`, `_labB.json` and `_labC.json`.
An earlier pilot bundled them into `label_out/for_chair/` with `residues_agree` precomputed; that
directory holds only the pilot's five batches and is NOT where your readings are. Compare the three
residue sets yourself.) The evidence itself is in `batches/batch_<NNNN>.txt`.

## Always write a sentence. Never write an empty one.

A REJECT verdict does NOT mean "no sentence". It means "this pair should not be trained on as a
positive example". Write the best defensible sentence anyway, and record the verdict beside it.

Reason: the three labellers have already done the work of finding what is sayable, and discarding
it destroys information we cannot cheaply recreate. A verdict is a column — the dataset keeps every
pair, and filtering happens at training time where it can be revisited. An empty `final` cannot be
revisited.

So for every pair, both fields are populated:

  verdict   ADOPT    one reading is correct as written
            REWRITE  you merged or repaired them
            REJECT   the pair should not be a positive training example -- say why in `why`
  final     ALWAYS a sentence, <= 35 words, the best the evidence supports.
            On a REJECT this is the most defensible thing that can be said, even if it is thin
            ("the edited atom is solvent-facing and gains no measured contact") -- that IS the
            finding, and a solvent-directed edit is a real medicinal-chemistry decision.

## Grounds for REJECT

  * the edit has no measured pocket consequence (new atoms contact nothing in a COMPLETE table)
  * the two crystals are not comparable (`unlabellable` is set, or the numbering/chain caveats make
    every claim unsafe)
  * not a medicinal-chemistry analogue pair (natural cofactor/nucleotide swaps)
  * the evidence contradicts itself and no clause survives

REJECT is about the **pair**, never about the **sentence**. A sentence in the wrong shape is a
REWRITE: repair it and keep the pair. This matters now more than it used to, because the sentence
rules changed and most well-written sentences in the old style break the new ones.

## The two-half shape, and what to do when a reading breaks it

REVIEWER.md now requires every sentence to be an input-side MEASURED anchor plus an analogue-side
QUALITATIVE consequence: named residues come from the input evidence only, and **no number may be a
measured analogue distance**. Direction and kind are expected; magnitude is not.

A labeller who writes "...to ring carbons at 3.61-3.69 A in the analogue's crystal" has given you a
true observation in an unusable form. REWRITE it — keep the residue, keep the input distance, turn
the analogue clause into an expectation ("should spread that contact over more ring carbons and
relax it slightly"). Do not REJECT it and do not adopt it as written.

You will see this on most pairs at first. That is expected: measured on the first 2,000 labels, 92%
cite at least one analogue distance and 24% name a residue absent from the input side. The shape is
new, the chemistry in those readings is usually fine, and your job is to convert rather than discard.

`check_label_rules.py` records compliance as a column after the run. It is not a gate, it never
blocks a batch, and you should not try to satisfy it mechanically at the cost of a defensible
sentence -- if the honest sentence breaks a rule, write the honest sentence and say so in `why`.

## The pocket envelope — what it legitimises and what it does not

Batches 0041 onward carry a `pocket envelope` block: every residue with an atom within 8 A of the
input ligand, in distance BANDS, whether or not it is any atom's nearest neighbour. Batches
0001-0040 do NOT have it, so do not expect it there.

It changes two of your judgements, in opposite directions:

  * **Naming an envelope residue is CORRECT, not ungrounded.** The envelope is input-side evidence,
    so rule 1 permits it. A labeller who names a residue you cannot find in the atom table may well
    have read it from the envelope -- check there before calling the claim unsupported. This is the
    mistake most likely to cost us good labels on these batches.
  * **Citing an envelope BAND as a distance is wrong.** The bands say a residue is reachable, not
    that anything is measured at that range. A sentence reading "sits 6-8 A from GLU63" is a
    REWRITE: keep the residue, drop the range.

A residue at 7.5 A touches nothing. A sentence asserting contact, a hydrogen bond, or packing with
an envelope-only residue is a REWRITE however plausible it sounds -- the evidence supports "could
reach", not "does touch".

The flags are the crystal-packing disclosure: `(X)` means another chain, `*` a cofactor or ion, `~`
a chain terminus and therefore probably a lattice neighbour rather than a site residue. A claim
resting on a `~` residue, or on a `(X)` residue in the 6.0-8.0 A band, is the weakest thing you will
see in these packets; REWRITE it onto firmer evidence or reject the clause.

## Use the disagreement

Across the pilot the three labellers named DIFFERENT residues on 43 of 50 pairs. Disagreement is the
norm and usually means one of them saw something the others missed — check who is right against the
evidence rather than taking a majority.

## Caveats

`***` blocks and `!!` row markers are load-bearing: covalency mismatch, symmetry permutation,
stereochemistry, multi-chain pockets, chain termini, chain relabelling, residue numbering, suspect
residues, clash-dropped atoms, incomplete tables, two-transformation pairs. If one is plainly wrong
for a pair, note it in `why` and proceed — do not stall. Caveat calibration is being tracked
separately and will not be fixed mid-run.

## Output

`label_out/batch_<N>.json`:
  {"<exact pair id>": {"verdict": "...", "final": "<always non-empty>", "why": "...",
                       "disagreement": "<what the three differed on, and who was right>"}, ...}

Every pair in the batch, valid JSON, written before you reply. Reply with counts by verdict only.
