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
