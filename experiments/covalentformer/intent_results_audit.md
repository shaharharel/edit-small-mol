# Audit of the PocketPair-21k intent results (2026-11-12)

Written outside `manuscript/` deliberately — the manuscript is frozen until the PI says otherwise,
and several numbers in `manuscript/EVIDENCE.md` are superseded by what is below.

Everything here is measured on the same molecule-level holdout the arms were scored on
(`rm_eval_*/sft_pocket.jsonl`, split `val`), by `scratchpad/verify_attacks.py` and
`scratchpad/attack_checks.py`.

---

## 1. The effective sample size is 94, not 400

349 scored prompts contain **94 distinct input molecules** across **16 distinct targets**; median 3
prompts per input, max 10. The arms are paired on the *prompt*, so the same molecule appears up to
ten times and the paired tests treat correlated rows as independent.

Consequence, by size of effect:

| comparison | reported | survives an effective n of 94? |
|---|---|---|
| reference − mismatch, +0.379, p=5e-50 | huge | yes |
| none − mismatch, +0.147, p=5e-40 | large | yes |
| **R→M − mismatch, +0.033, p=0.014** | small | **almost certainly not** |
| polarity bucket, p=0.047 | marginal | no |
| rigidity bucket, p=0.044 | marginal | no |
| the 0.654-vs-0.546 subgroup split (n=73) | small | not established |

The load-bearing claim for the factorisation — that R's intents beat a wrong sentence — rests on
the one p-value this kills. It must be re-tested with a cluster-robust or per-molecule-aggregated
test before it is quoted again.

## 2. The right null is "any other real analogue", and it is 0.710

For each held-out input, take the OTHER analogues of that same input that chemists actually made
in this corpus, and score them against the reference analogue. That is what a chemist proposing a
different-but-real edit scores, with no model involved:

**mean TC(other real analogue → reference) = 0.710** (n=346 prompts that have a sibling)

Against that null:

| arm | TC→reference | vs the 0.710 null |
|---|---|---|
| reference (dictated) intent | 0.918 | **+0.208** |
| no intent (echoes the input) | 0.684 | −0.026 |
| **R→M (the deployable pipeline)** | **0.570** | **−0.140** |
| mismatched intent | 0.537 | −0.173 |

So the only arm that beats a chemistry-blind "any real analogue" baseline is the one with the
answer dictated into its prompt. **The pipeline loses to the null by 0.14.** Any framing of
R→M as a design result has to answer this first, and TC→reference cannot be the primary metric
while this is true.

## 3. Residue specificity is recall, so it must be read at matched residue count

`big_table.py` defines specificity as `|named ∩ reference| / |reference|` — recall, monotone in how
many residues a sentence names. Recomputed baselines on this corpus, naming the k nearest residues
in the table the model was shown:

| baseline | recall |
|---|---|
| nearest 1 residue | 0.121 |
| nearest 2 | 0.200 |
| nearest 3 | 0.240 |
| nearest 5 | 0.262 |
| one residue at random from the table | 0.148 |

Against that:

| arm | residues/sentence | specificity | read against the matched baseline |
|---|---|---|---|
| **joint ChemDFM-21k** | 1.59 | **0.417** | **well above k=2 (0.200) — real signal** |
| R (reason adapter) | 1.17 | 0.190 | **between k=1 and k=2 — at the distance-sort baseline** |
| blind model (no table) | — | 0.121 | **exactly k=1 — strip the pocket and it degenerates to a distance sort** |

Two conclusions. The joint model's intent specificity survives this correction and is the single
genuinely positive intent-quality result we hold. **R does not: it is no better than naming the
nearest residues in the table it was handed.** Groundedness of 1.000 remains near-definitional,
since the table is in the prompt.

Note for the paper: `big_table.py`'s docstring quotes a nearest-residue baseline of **0.5625**,
measured on the covalent corpus against `reference_residues`. On the 21k corpus against
`cited_residues` the same baseline is **0.121**. Both are correct for their own corpus and neither
may be quoted against the other arm's number.

## 4. The property buckets were a regex artefact; corrected, they say something different

The published buckets were assigned by regex over the intent sentence, and `/polar/` also matches
**"apolar"** and **"nonpolar"** -- stock phrases in a pocket description ("lies near the apolar
ILE189 side chain"). The word "polar" appears in **75% of these sentences**, "apolar" in 27% and
"nonpolar" in 8%, so the polarity bucket swallowed 263 of 349 rows and its "+3.63 TPSA" was close
to the corpus-average TPSA change. `/lipophilic/` matched a description of the pocket as often as a
request to lower lipophilicity (85 sentences contain the word; 20 actually ask for less of it), and
`/lock/` matched "block" and "unlock".

Bucket sizes, published vs corrected: polarity 263 -> **15** (94% noise), lipophilicity 148 -> **20**
(86% noise), rigidity 35 -> 26, size-up 90 -> 101, size-down 78 -> 78 (clean).

Rebuilt with direction-aware buckets, and with the two columns that were missing -- what the
analogue **chemists actually made** does to that property (the ceiling), and a permutation null that
re-pairs the same sentences with the same molecules:

| bucket | prop | n | chemists | reference intent | R->M | mismatch | perm null | p | Holm |
|---|---|---|---|---|---|---|---|---|---|
| lipophilicity down | cLogP | 25 | -1.18 | **-1.20** | -0.25 | +0.06 | -0.48 | 6.7e-5 | **survives** |
| size down | MW | 54 | -31.90 | **-31.24** | +1.02 | +1.17 | -4.37 | 0.0018 | **survives** |
| polarity up | TPSA | 8 | +17.34 | +16.99 | -1.65 | +1.20 | -13.32 | 0.078 | dies (n=8) |
| rigidity down | RotB | 11 | +1.09 | +1.18 | +0.27 | +0.45 | +0.19 | 0.188 | dies |
| basicity down | basicN | 24 | +0.04 | +0.04 | +0.00 | +0.09 | - | 1.0 | dies |
| size up | MW | 60 | +3.21 | +0.68 | -5.22 | +3.32 | -14.69 | 0.997 | dies |

Three readings, and the first two are better news than the published table:

1. **In the clean buckets the model reproduces the chemists' own property move almost exactly** --
   -1.20 against -1.18, -31.24 against -31.90, +16.99 against +17.34. The diluted buckets had been
   dragging these effects toward zero; the corrected effect sizes are roughly twice as large.
2. **The "wrong-signed" buckets are not model failures.** Where the model moves rigidity the wrong
   way (+1.18), the chemists' own analogues move it the same wrong way (+1.09); same for basicity
   (+0.04 vs +0.04). The sentence talks about rigidity while the realised edit adds flexibility --
   a corpus labelling problem, not a steering failure.
3. **R->M steers nothing.** -0.25 against the reference's -1.20 on lipophilicity, and +1.02 against
   -31.24 on size-down, where it moves mass in the wrong direction entirely.

Two of six survive Holm. The honest claim is narrow: conditioned on an unambiguous property
request, the molecule model reproduces the chemists' move on lipophilicity and molecular size --
which, given the 86.1% dictation rate, is still an obedience result rather than a design one.

## 5. What still stands

- **The molecule responds to the sentence it is handed.** +0.379 paired, and a wrong sentence is
  worse than no sentence (−0.147) while also costing validity (1.000 → 0.915). Both effects are far
  too large for the n=94 correction to touch.
- **Zero intent, zero edit.** With no sentence the model returns the input unchanged in 100% of
  rows, so the sentence is the sole driver of structural change, and `none` is a diagnostic rather
  than a baseline.
- **The reference arm measures transcription.** The reference intent names a group that actually
  changed in 86.1% of rows (2.09 of 2.71 changed groups), with heavy-atom Δ MAE 0.374 in that arm.
- **The pocket governs the prose and not the molecule.** Grounded 1.000 vs 0.432 and specificity
  0.388 vs 0.121 with the table vs without, while TC→reference is 0.679 vs 0.680 and a wrong pocket
  costs 0.013. The blind arm landing exactly on the k=1 distance-sort baseline is the cleanest
  reading of what the table buys.
- **The joint model beats the factorised pipeline on both sides**: molecules 0.679 vs 0.570, intent
  specificity 0.417 vs 0.190.

## 6. What to run next, in order

1. **Cluster-robust re-test** of every paired comparison, aggregating to one observation per input
   molecule (n=94) or per target (n=16). No GPU. Decides whether R→M beats a wrong sentence at all.
2. **Content-ablation ladder** on the reference sentence — paraphrase, minimal group-swap
   counterfactual, content-deleted — to split the 0.918 into obedience and dictation. Prompt edits
   on the existing checkpoint, ~90 min of GPU per rung.
3. **Decoy-pool identifiability** as the replacement primary metric: given k contrasting intents for
   one input, can a blind scorer match molecule to intent? Immune to the "different but valid
   analogue" problem that makes TC→reference unusable here.
4. **Site accuracy** (does the edit land on the atoms the cited residue contacts) rather than
   whole-molecule similarity.
5. **Joint vs factorised** at matched parameter and token budget, scored on 3 and 4 rather than on
   TC→reference.
