# Review 07 — adversarial ML read of the intent-conditioning results

Posture: hostile. I assume every number is real and attack what it can support. Two numbers below
I computed myself on `shared_pairs_deduped.csv` (25k pairs / 4k parents, ECFP4-2048, scratchpad
script, review-only); they decide §2 and they are the reason I reject TC→reference as primary.

---

## 1. The mismatched-intent control cannot separate instruction-following from distribution match

`build_rm_arms_21k.py` builds M's training prompt as `prompt + "DESIGN REASON (given):" +
sentence`, where `sentence` is **always the reference sentence**. So M was trained on exactly one
conditioning distribution. At eval, the oracle arm is **in-distribution**; R→M and mismatch are both
**out-of-distribution**, and mismatch is OOD along four axes at once (off-pocket, off-molecule,
absent residues at a 55.7% invented-residue rate on the sister corpus, and foreign surface form).
The +0.379 is therefore an upper bound on instruction-following contaminated by train/test prompt
match. The validity drop 0.963 → 0.915 under mismatch is a distribution-shift signature, not an
obedience one.

Cheapest decisive control, and it beats a same-pocket counterfactual on cost: a **content-ablation
ladder on the reference sentence**, same checkpoint, prompt edits only, 400 greedy generations per
rung (~90 min/rung at the `eval_sft.py` rate).
(a) **Paraphrase**: LLM-rewrite the reference sentence, content preserved, surface changed.
(b) **Group-swap minimal counterfactual**: keep the whole sentence, replace only the named chemical
group with a different real group ("replace the meta chlorine" → "replace the meta methoxy").
(c) **Content-deleted**: strip the group and site, keep length, syntax and residue citation.
Coherence is constant across all three. If (a) holds ~0.9 and (b)/(c) fall to the sibling band (§2),
the claim survives; if (a) collapses, 0.918 is surface memorisation. Gate the paper on (b); the
same-pocket swap is a nice-to-have afterwards.

## 2. TC→reference has ~0.07 of usable dynamic range, and the pipeline sits below a chemist

Measured on this project's own MMP corpus:

| operating point | mean TC | median |
|---|---|---|
| input vs its own analogue | 0.609 | 0.625 |
| **a chemist's *other* real analogue of the same input, vs the reference** | **0.607** | **0.623** |
| real pairs with \|Δheavy\|≤1 | 0.626 | — |
| real pairs with \|Δheavy\| 3–5 | 0.562 | — |

Read those rows against the paper's arms. A *correct, valid, different* design — another analogue a
chemist actually made from the same hit — scores **0.607**. R→M scores **0.570**, i.e. *below* a
chemistry-blind "emit any real analogue of this input" null. Mismatch scores 0.537. The whole band
the paper is discriminating in is ~0.54–0.61, **0.07 wide**, and the metric's own `none` arm —
which does nothing at all — scores **0.684**, beating the full pipeline by 0.114. A primary metric
that ranks *doing nothing* 0.114 above your system, and ranks a real chemist's alternative analogue
below your oracle by 0.31, is not measuring design.

Edit size is *not* the confound, and I say so against my own interest: the \|Δheavy\| sweep moves TC
by only 0.064 across the full 0.374→3.907 range the arms span, so size explains **~17%** of the
0.379 gap. The other ~82% is literal transcription — median TC 1.000 means the model *emits the
reference molecule* in >50% of rows, and §8's 86.1% dictation rate says the sentence told it which
one. RESULT 1's headline is a transcription benchmark with a known answer key.

Primary metric should have been **identifiability against a decoy pool**: rank the reference
analogue by model likelihood among *k* real sibling analogues of the same input (chance = 1/k,
"different but valid" is no longer penalised, the 0.07 band disappears). Secondary: **site accuracy**
— does the generated molecule change the same site as the reference (MCS map; note the stereo-blind
defect already logged) — because the site is the design decision. TC→reference becomes a
supplementary column.

## 3. Uniqueness inverts the reading, and the arithmetic does not close

The `none` arm reports TC→input = 1.000 on **100%** of rows and uniqueness **0.235**. If every
output equals its input, distinct outputs = distinct inputs, so 0.235 × ~400 ⇒ **~94 distinct input
molecules among 400 "paired prompts."** Two consequences: (i) the uniqueness column's ceiling is set
by input diversity, not by model entropy, so R→M's 0.238 ≈ none's 0.235 is "one deterministic output
per input," not demonstrated collapse; (ii) **n=400 is not 400 independent units** — effective n may
be ~94, and the paired Wilcoxon p-values are inflated accordingly. Report `len(set(input_smiles))`
and cluster-bootstrap or mixed-model by input molecule. One line of code; it moves p=0.014.

Mismatch's 0.937 is the paper's *strongest* evidence that M reads the sentence (diverse sentences →
diverse molecules) and it is filed under "worst arm." The joint model at 0.173 is the worst
offender on the only metric that catches canning.

**The degenerate strategy that beats this metric suite**: "echo the input with the corpus's modal
edit, and cite any residue from the table." It scores validity ≈1.0, groundedness **1.000**, names a
chemical group 100%, edit verb ~86%, TC→reference ≈0.61 (**above R→M's 0.570**). Only uniqueness
catches it, and R is already measurably near it: 97 distinct sentences of 349, one 6-word opening on
16% of rows. The suite as composed cannot distinguish the pipeline from a canned-edit policy.

## 4. Property buckets: circular, overlapping, uncorrected — two survive, and they are one effect

The label is read by regex from the **conditioning sentence** (`property_response.py`, `TARGETS`),
and that sentence was written by an agent **with the product in view**. So the bucket is a
description of the reference edit, and the oracle arm — which reproduces the reference in >50% of
rows — inherits the reference's property delta by construction. RESULT 4 is entailed by RESULT 1,
not independent of it. Mandatory check: **recompute every cell for the reference
analogues themselves** — if oracle ≈ reference, the experiment has zero marginal content.

Holm/Bonferroni across the 6 tests (0.00013, 0.00095, 0.044, 0.047, 0.37, 0.45): **only cLogP-down
(n=148, 0.00013×6=0.0008) and MW-down (n=78, 0.00095×5=0.0048) survive.** TPSA at p=0.047 — the
largest bucket, n=263 — dies at the second step. And the two survivors are one effect: "remove a
group" makes MW fall by arithmetic and cLogP follows it; MW/cLogP are near-deterministic in atom
composition. Two wrong-sign results and 645 bucket memberships from 349 rows (overlapping regexes;
`solubility` and `lipophilicity` both map to cLogP) complete the picture.

Non-circular version: label the direction from a source **the sentence cannot see** — the measured
assay/property delta of that pair, or a prospective instruction *we* write and hand to M for inputs
whose analogue we never show. Then the metric is "fraction of rows moving in the instructed
direction" vs a 50% chance line, one test per property, pre-registered.

## 5. Groundedness 1.000 is near-definitional; specificity 0.417 is not yet above chance

`row_table_residues()` parses the table **out of the prompt the model was just shown**, then
groundedness asks whether cited tokens appear in it. Copying is the trivially optimal policy, and the
model scores **1.000 against a ~0.90 human ceiling** — exceeding the reference is the signature of
metric gaming, not of better chemistry. Minimum informative replacement: **precision at a matched
naming budget** (residues/sentence held equal across arms), plus "is the cited residue within X Å of
the atom the sentence says changed" — the citation's *relevance*, which is the thing chemists reject.

Specificity in `big_table.py` is `|named ∩ ref| / |ref|` — **recall only**, monotone increasing in
how many residues you name. Random-pick scores 0.148, so a table carries ~7 residues; therefore
**"name the 3 most-contacted residues" scores ≈3×0.148 = 0.44 > the joint model's 0.417**, and "name
the whole table" scores 1.000. R names 1.17 residues/sentence; **residues/sentence for the joint
model is not reported**, and without it 0.417 is uninterpretable — if joint names ~2.8, it is exactly
chance. Report Jaccard or F1, or precision at matched budget, and add the two stronger
zero-parameter baselines (top-k contacts; most-frequently-cited residue for that target).

Two numbers must also be reconciled: `big_table.py`'s own docstring states the nearest-residue
baseline scores **0.5625**, while the result claims **0.121**. If 0.5625 is the right corpus, *both*
models lose to a distance sort. Also, `sel` falls back from `reference_residues` to `cited_residues`
depending on corpus, and `reference_ceiling()` silently **skips** rows without `table_residues` — so
joint-vs-R may be two different denominators. Until that is shown to be one code path on one corpus,
0.417 vs 0.190 is not a comparison.

## 6. On this evidence the factorisation is not defensible

Joint wins on molecules (0.679 vs 0.570, a 0.109 gap — larger than the 0.07 discrimination band in
§2) and on specificity (0.417 vs 0.190, caveats above). The pocket ablation then removes the stated
motivation: table shown vs withheld gives TC→reference **0.679 vs 0.680**, and a wrong pocket costs
**0.013**. The pocket is worth ~0.001–0.013 TC on the molecular output while the paper is framed as
pocket-conditioned design. The only thing factorisation buys that is actually demonstrated is
*intervenability* — and §1 says intervenability itself is not yet separated from distribution match.

The single deciding experiment: **matched-budget head-to-head where intervention is the task.** One
corpus, one split, equal adapter parameter count and equal token budget (joint's single adapter vs
R+M's two — currently unequal and unreported), scored on (i) identifiability vs decoys (§2), and
(ii) **steer accuracy under instructions we author** (§4), where joint cannot be instructed at all.
If factorisation does not win (ii) decisively, drop it.

## 7. Validation-as-test, single run, no seeds — what is safe and what is noise

Checkpoints were selected on val and every number is a val number; `eval_sft.py` flags this itself
(`val_is_model_selection_not_test`) and generation is greedy (`do_sample=False`), so decoding adds no
variance but **training-seed and checkpoint-selection variance are entirely unmeasured** and
optimistic by an unknown amount.

- **Safe:** oracle − mismatch +0.379; none − mismatch +0.147; dictation 86.1%; grounded 1.000 vs
  0.432. Large, mechanical, or both.
- **Will not survive 3 seeds:** **R→M − mismatch +0.033, p=0.014** — one third of the §2
  discrimination band, single seed, val-selected, and with effective n possibly ~94 (§3) the p is not
  0.014. This is the load-bearing number for "the pipeline works," and I do not believe it.
  Likewise the conditional split 0.654 (n=73) vs 0.546 (n=262) — post-hoc subgroup, n=73, no
  correction; rigidity p=0.044 and TPSA p=0.047.
- **Probably survives but confounded:** joint 0.679 vs R→M 0.570 (budget mismatch, §6).
- **Unresolvable as reported:** specificity 0.417 vs 0.190 (§5, denominators).

Minimum to proceed: 3 training seeds for R and M, a true test split never used for selection, and
every val number relabelled a model-selection score in the table itself, not in a footnote.
