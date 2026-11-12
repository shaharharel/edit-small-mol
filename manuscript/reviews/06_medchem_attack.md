# 06 — Med-chem attack on the R/M intent results

Reviewer stance: senior med chemist, project-team perspective. Every number below is either from the
brief, from `manuscript/EVIDENCE.md`, or computed by me today on files in this repo.

---

## 0. The one arithmetic fact that reframes everything

`TC→reference` is scored against a scale whose floor is not zero. The `none` arm echoes the input and
scores **0.684** — that *is* TC(input, reference). So:

| arm | TC→ref | gap closed from the echo floor to 1.0 |
|---|---|---|
| reference-dictated intent | 0.918 | (0.918−0.684)/(1−0.684) = **74%** |
| R→M (the deployable pipeline) | 0.570 | **−36%** — *below the starting material* |
| mismatched intent | 0.541 | −45% |

**R→M's proposals are less similar to the compound the chemists actually made than the hit itself is.**
The pack defends this by saying `none` "declines to design", which is true and irrelevant: a chemist
does not care whether the tool declined, they care whether the proposal is closer to the right answer
than what they already had.

I verified the per-row version on the 14k arms on disk (`results/covf_rm_eval/gens/*.jsonl`, joined to
`data/pdb_mirror/v6_pairs.jsonl`, Morgan2/2048):

| arm | n | TC(gen,ref) | echo TC(input,ref) | **rows beating the echo** | exact hits |
|---|---|---|---|---|---|
| oracle | 124 | 0.732 | 0.639 | **66.9%** | 25.8% |
| pipeline (R→M) | 119 | 0.514 | 0.641 | **14.3%** | 0.8% |
| mismatch | 122 | 0.496 | 0.637 | **9.8%** | 0.0% |

R→M beats "hand the chemist back their own hit" in **14.3% of rows**, against 9.8% for a random
sentence. That 4.5-point margin is the entire deployable result. **Report win-rate-against-echo, not
mean TC, in the paper.** It is a one-line CPU computation and it is the number a project team would ask for.

Calibration: R's whole contribution over a random sentence is **+0.033 TC**. A *deliberately wrong
pocket* costs 0.013 TC. R's signal is 2.5 wrong-pocket units.

---

## 1. Ranking the four claims

**Claim 2 (dictation 86.1%) — believe it.** It is a SMARTS count over a corpus with no model in the
loop. It is the most load-bearing number in the pack because it invalidates the word "oracle". Two
corrections: 86.1% is *recall* under an OR over 40 classes; the precision is **2.09/4.91 = 42.6%**, and
the class list cannot see ring fusions, regiochemistry or stereochemistry (`site_of_change` is
stereo-blind), so 86.1% is a **lower bound**. *Lets you do:* correctly discount Claim 1. Nothing else.

**Claim 1 — believe the direction, not the headline.** The qualitative core is solid and I would not
argue with it: mismatch sits 0.147 *below* `none` with validity dropping 100%→91.5%, so M is genuinely
executing the sentence rather than decorating. That is real. The 0.918 is transcription (§5 below).
*Lets you do:* nothing on a live project. See §0.

**Claim 3 (intent quality) — weak.** §4.

**Claim 4 (property controllability) — do not believe it as presented.** §2. *Lets you do:* nothing;
two of six axes are wrong-signed and the bucket code has reproducible bugs.

---

## 2. The property table is a bucketing artefact

I ran the actual regexes in `experiments/covalentformer/v6/property_response.py`. They are
direction-blind and lack word boundaries. Confirmed misclassifications:

| sentence | bucket assigned |
|---|---|
| "replace the polar amide with a **nonpolar** methyl" | polarity ↑ |
| "the **nonpolar** contact with Val123 is retained" | polarity ↑ |
| "**block** the metabolic soft spot" / "**unlock** the amide" | rigidity ↓ |
| "**increase** logP to improve permeability" | lipophilicity ↓ |
| "**increase** basicity by adding a piperidine" | basicity ↓ |
| "**extended** conjugation lowers the HOMO" | size ↑ |

`/polar(?:ity)?/` matches *nonpolar* and *apolar*; `/lock/` matches *block*, *blocking*, *unlock*;
`/protonat/` matches *deprotonated*. The two wrong-signed buckets (size ↑ n=90, driven by "extend";
rigidity ↓ n=35, driven by "block") are exactly the two whose regexes are most contaminated. The
wrong signs are probably **not** a model failure at all — they are a labelling failure.

Then the statistics. 645 memberships over 349 rows = **1.85 buckets per row**; the six Wilcoxon tests
share rows and are not independent. Taking them at face value and applying Holm at α=0.05
(thresholds 0.0083, 0.010, 0.0125, …): cLogP (1.3e-4) and MW↓ (9.5e-4) pass; **polarity ↑ p=0.047 and
rigidity p=0.044 both fail and the procedure stops there.** The largest bucket in the table does not
survive multiplicity. And MW↓'s own regex (`truncat|trim|smaller`) is a literal description of removing
atoms, so "the sentence said remove atoms and 32 Da came off" is close to tautological. **One of six
axes (cLogP) is a non-trivial survivor** — and it is the axis whose regex most readily mislabels
"increase logP" as "decrease".

Two deeper confounds:

- **Circular labels.** Bucket membership is read from the *reference* sentence, which an agent wrote
  with the product in view and which names an actually-changed group 86.1% of the time. "Rows whose
  dictated sentence says polar" ≈ "rows where a polar group was added". The bucket is a leaky readout
  of the ground-truth delta, so the oracle arm is partly scored on a label derived from its own answer.
- **TPSA/MW/cLogP are not independent axes.** Adding one amide is +43 Å² TPSA, +43 Da MW and −1.0
  cLogP simultaneously. Almost any polar-group addition "passes" polarity ↑ and lipophilicity ↓ at once.
  The table has no mechanism to tell steering from heavy-atom drift.
- **>50% of oracle rows are exact copies** (median TC 1.000). On those rows the oracle's property delta
  *is* the reference analogue's delta. +3.63 Å² is largely a measurement of the corpus, not the model.

### The right experiments

**(a) Free, CPU-only, today — fix and re-run the existing table.** Word-boundary, direction-aware,
negation-aware patterns; single-membership assignment (drop multi-match rows rather than double-count);
bootstrap 95% CIs per cell; a **fifth column giving the reference analogue's own delta** (the achievable
ceiling — if the real analogues move TPSA +14 Å² and the oracle arm moves +3.6, the model recovers 26%
and that is the honest headline); and the whole table recomputed on the **TC<1.0 subset** so exact
copies cannot carry it.

**(b) The decisive experiment — direction-crossed minimal pairs.** Same input, same template, one word
flipped: *"replace the 4-chloro with a **more** polar group"* vs *"…a **less** polar group"*. 6 axes × 2
directions × 200 inputs = 2,400 prompts. Metric: sign-agreement rate and the paired difference between
the two directions, with bootstrap CIs. No regexes, no buckets, no overlap, no circularity — the only
variable is one word. Pre-register the bar: **sign agreement ≥70% on ≥4 of 6 axes**, or steerability is
not claimed.

---

## 3. Are +3.63 Å², −0.62 cLogP, −32 Da real moves?

Single-edit reference scale a chemist actually works in:

| edit | ΔTPSA | ΔcLogP (Hansch π) | ΔMW |
|---|---|---|---|
| aromatic CH → N (benzene→pyridine) | +12.9 | ≈ −1.2 | +1.0 |
| H → OH | +20.2 | −0.67 | +16 |
| H → CN | +23.8 | −0.57 | +25 |
| H → primary amide | +43.1 | −1.5 | +43 |
| H → COOH | +37.3 | −0.3 | +44 |
| H → Cl | 0 | +0.71 | +34.5 |
| H → CH₃ | 0 | +0.56 | +14 |
| OMe → OH | +9 | −0.65 | −14 |

- **TPSA +3.63 Å² is not a move.** It is 28% of the *smallest* polar edit available (an aromatic N) and
  ~9% of an amide. No filter band is 3.6 Å² wide: Veber is TPSA ≤140, the CNS-MPO desirable band is
  40–90. With a per-compound ΔTPSA spread of ~15–20 Å² in single-substituent sets, +3.6 is ≈0.2 SD —
  invisible on any one compound a chemist would actually make.
- **cLogP −0.62 is one real move, and the only defensible number in the table.** It is one
  de-methylation or one H→OH, and at constant potency it is +0.62 LLE/LipE, which a project would bank.
  Caveat: the usual hit-to-lead ask is **1.0–1.5 log units** (cLogP 4.5–5 hit → 3.0–3.5 lead), so this
  is ~40–60% of one campaign step. And Crippen's absolute RMSE vs measured logD is ~1.0–1.3 log units
  (MMP-style deltas are better, ~0.4–0.5), so −0.62 is near the predictor's own delta noise.
- **MW −32 Da is one substituent** (an OMe at −30, a Cl→H at −34.5), ~8% of a 400 Da hit. It moves
  nothing against MW ≤500 unless you are already sitting at 500–530. The hit-to-lead pressure is
  usually to hold MW flat while gaining 10–100× potency, not to shave 32 Da.

Verdict: one of three is chemically meaningful, and it is the one with the most suspect bucket.

---

## 4. Intent quality: 0.417 vs 0.148 is a weak result

It is 2.8× a strawman. The failure is the baseline set, not the arithmetic.

- **The reference is not ground truth.** Specificity measures agreement with an *agent that had the
  product in view*. Scoring 0.417 against a dictated reference is 0.417 agreement with a transcript.
- **Random = 0.148 implies only ~7 candidate residues per table.** At ~1.1 residues per sentence,
  overlap ≈ precision ≈ recall, so 0.417 means **the wrong residue roughly 6 times in 10**.
- **Missing baselines that would likely eat most of the margin:** (i) the **per-target modal cited
  residue** — kinase hinge/gatekeeper residues dominate these corpora and this alone could score 0.3+;
  (ii) top-k by contact count or burial from the table, not uniform-random; (iii) **1-nearest-neighbour
  retrieval** — copy the intent of the most similar *training input molecule*. R emits 97 distinct
  sentences over 349 rows with one 6-word opening on 16%; a 1-NN retriever is the honest null and may
  reproduce R's 0.190 overlap exactly. Until (i)–(iii) are run, 0.417 is uninterpretable.
- **Groundedness 1.000 is vacuous and should be deleted**, not reported. The table is in the prompt.

### Strongest non-panel, no-GPU measurement of intent quality: predictive sufficiency in bits

Fit a CPU-only model (bag-of-words / char n-grams → logistic or gradient boosting) from the **intent
sentence alone** to the **realised edit**: which of the 40 SMARTS classes appeared and disappeared, and
the sign of ΔTPSA/ΔcLogP/ΔMW/ΔRotB. Report macro-AUC and **bits of mutual information** for five
sentence sources: (1) reference intents, (2) R's intents, (3) token-shuffled R, (4) per-target modal
intent, (5) 1-NN retrieved intent. "Does the sentence tell you what to make?" becomes a number in bits
with no chemist, no model labeller and no GPU. Run the mirror image too — given the realised edit, rank
all 349 intents and report MRR/diagonal dominance (this is TBD-4, done as retrieval, free).

---

## 5. Attacking 0.918

The reference sentences are parts lists. From `oracle.jsonl`, verbatim:

> "Replacing the **pyridine** with a **methoxyphenyl** removes the ring nitrogen 3.53 Å from SER92.CA
> and grows a **methoxy** off carbons packing GLU132, while the 1.92 Å covalent bond to CYS95.SG is held."

Leaving group, arriving group, added substituent and retained warhead, all named. In that very row the
generation differs from the reference only in *methoxy regiochemistry* (para vs meta) — the residual is
precisely what the sentence underspecifies. Together with dictation recall 86.1%, precision 42.6% (a
~5-item list covering 2.71 real changes) and heavy-atom Δ MAE 0.374 (the atom count recovered to within
0.37 atoms), 0.918/median 1.000 measures **transcription fidelity of a dictated edit**. It is not
evidence about design, and the arm must be renamed "reference (dictated) intent" everywhere.

Two further holes. (i) **The p-values are inflated.** 400 rows over ~21–100 targets with repeated target
space; p=9.6e-50 assumes independent rows. Cluster at target level and the 0.379 gap survives easily but
R→M's **+0.033, p=0.014 will not**. Report target-clustered or target-block-bootstrap CIs. (ii)
**Uniqueness 0.626** means ~37% of 400 outputs duplicate each other, so the effective sample is smaller
again.

### The two cheap experiments that settle it

1. **Group-masking ablation (≈1 GPU-hour, 400 prompts).** Rewrite each reference sentence replacing
   every named chemical group with a placeholder — "replace the ⟨GROUP⟩ at the ⟨SITE⟩" — keeping all
   residues and distances. If TC collapses toward mismatch (≈0.55), 0.918 is transcription, full stop.
   If it holds near 0.75, M is genuinely inferring chemistry from the pocket and the claim partly stands.
2. **Rule-based transcriber, CPU only — the most important missing experiment in the pack.** A ~200-line
   program that parses "replace X with Y" from the sentence and applies it by SMARTS to the input. If it
   reaches TC 0.80–0.85 on the 86.1% of rows where the sentence names a changed group, then a 14B
   chemistry LLM is buying 0.07–0.12 Tanimoto over regex-and-RDKit, and the design framing is dead.
   **Until this baseline exists, 0.918 cannot go in a paper.**

### The one honest reading

There is exactly one: **0.918 is an upper bound on the factorisation, and it localises the loss in R.**
0.918 → 0.570 means ~92% of the deployed shortfall is R's intent, not M's chemistry. The site-hit split
(TC 0.654 when R names a reference residue, n=73, vs 0.546 when it does not, n=262) supports it. That is
a legitimate *engineering* claim about where to spend the next GPU-month. It is not a claim about design,
and it should be stated in those words.

---

## 6. Paper / SI / do not publish

**Paper (and the paper is a negative result — lead with it):**
- Dictation rate 86.1% / precision 42.6%, and the renaming of "oracle" to "reference (dictated)".
- Pocket ablation: grounded 1.000 vs 0.432, specificity 0.388 vs 0.121, **TC→reference 0.679 vs 0.680**.
  The pocket governs the prose and not the molecule; a wrong pocket costs 0.013 TC.
- Echo-normalised results: win-rate-against-echo 66.9% / 14.3% / 9.8%, and the plain statement that
  **R→M currently sits below the starting material** on TC→reference.
- A wrong sentence is worse than no sentence (−0.147, validity 100%→91.5%): M follows instructions.
- The joint model (0.679) beating the factorised pipeline (0.570), stated as a cost of factorisation,
  alongside the 21k joint model's uniqueness of 0.173.
- R's mode collapse: 97/349 distinct, one opening on 16%, residue overlap 0.190, exact-set 9.7%.

**SI:**
- The full four-arm table with target-clustered CIs and the group-masking ablation.
- The site-hit split (0.654 vs 0.546).
- The fixed property table with CIs, non-overlapping buckets and the reference-analogue ceiling column,
  reported as a **limitation**, not a capability.

**Do not publish:**
- The property-controllability table in its current form. The bucket code has three reproducible
  substring/direction bugs, rows are counted 1.85× each, four of six axes fail Holm, and two are
  wrong-signed for labelling reasons. Publishing it invites exactly this review in public.
- Residue specificity 0.417 as a quality claim, until the modal-residue, contact-ranked and 1-NN
  baselines exist.
- Groundedness 1.000 as evidence of anything.
- Any use of the word "oracle", and any use of 0.918 as a design result before the rule-based
  transcriber baseline is run.
- Anything resting on §5's 46.5% "sound reasoning" — it is a model-labeller panel on a different (9k)
  model, and the pack's own caveat 1 says panel drift exceeds the differences being measured.
