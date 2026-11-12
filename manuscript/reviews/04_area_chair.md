# Meta-review, written in advance

AI area chair / harshest reader. `manuscript/EVIDENCE.md` is the only admissible source of numbers.
Working memo.

## 1. Score as submitted today

**Reject (borderline). 4/10, confidence 4/5.**

Not lower: the pocket ablation (§3) and the oracle/mismatch/`none` contrast (§4b) are real, honestly
reported, and more interesting than most submissions here. Diagnosing that `none` echoes its input is
the work of people who checked.

Not higher: **the title claim has no model behind it.** Nothing in the pack is trained on the
literature corpus (TBD-2, gated on cofolds that started the day the pack was written). A reviewer reads
§2 as a well-QA'd dataset and §3–4 as a different paper about instruction-conditioned analogue
generation. Also: zero seed replication, no non-ablation baseline, and the central design decision
(factorising R/M) is never compared against the joint model it claims to beat.

**Highest-leverage addition: an intent-leakage decomposition of the oracle arm.** The one large effect
(oracle − mismatch +0.379, p = 9.6×10⁻⁵⁰; oracle median TC→reference 1.000; heavy-atom Δ MAE 0.374 vs
2.622) is the most attackable thing in the paper. §2 says the 14k/21k intents are **agent-written from
the pair** — with the product in view — while §4b calls them "the chemists' own." That is a
label-provenance leak plus an internal contradiction. If the sentence names the transformation, M is
paraphrasing, not reasoning. The fix needs no cofolds and no training: re-run M inference on (i) oracle
sentences with the chemical content redacted but the pocket hypothesis kept, and (ii) a **same-pocket
counterfactual intent** replacing the foreign-pair sentence. That turns the headline into a measured
decomposition and fixes a confounded control. Runner-up, nearly free: a frontier LLM zero-shot on the
same contact table.

## 2. Claim / evidence audit

| # | Claim | Evidence | Supported? |
|---|---|---|---|
| C1 | Finetuning makes a chemistry LLM usable | validity 0.118→0.973; sound reasoning 3.9%→46.5%; invented 67.7%→42.0% | **Partly.** Validity yes. Reasoning is a model-labeller panel, and §6.1 says panel drift exceeds the between-arm differences — the column cannot carry a trend without same-panel seeded controls. No reviewer grants "usable" at 46.5% sound / 42% invented. Single-run. |
| C2 | The model is pocket-aware | grounded 1.000 vs 0.432, selectivity 0.388 vs 0.121, TC→ref 0.679 vs 0.680, wrong pocket 0.013 Tc, distances changed nothing | **No, as stated.** It supports only: the pocket decides which residues the prose may cite. Worse, groundedness is near-definitional — the table is in the prompt, so one arm copies and the other must invent; 1.000 vs 0.432 measures copy-from-context. The only non-trivial number, selectivity 0.388, is low. Retitle to **grounded attribution** and put 0.680 / 0.013 in the same sentence. Missing control: pocket-*shuffled* groundedness (shuffled tables still score 1.000 by construction). |
| C3 | The model follows design intent | +0.379 p=9.6×10⁻⁵⁰; mismatch 0.147 below `none` p=6.9×10⁻⁴⁰, validity 1.000→0.915; §4a +0.180 | **Instruction-sensitivity, yes. Design-intent-following, no.** R→M retains **8.7%** of the oracle-over-mismatch gap ((0.574−0.541)/(0.919−0.541)); report that yourselves. And `mismatch` is confounded — a foreign sentence names absent residues, so the drop may be prompt-incoherence; its 55.7% invented-residue rate (§4a) is evidence for the confound. Separately, §4a and §4b disagree about the same system (R alone groundedness 0.869 / invented 13.1% vs 1.000; selectivity 0.421 vs 0.214; mismatch ≈ R→M vs mismatch ≪ R→M). Reconcile or hard-separate, or it reads as metric shopping. |
| C4 | We learn from chemists' intent | 1,622 papers, 17,601 pairs, v5.3 gates 100%, leakage 0.00%; **60.7% restate an outcome**; no model trained on it | **Unsupported today.** Defensible version: a corpus of 17,601 pairs with quoted rationales, 39.3% of which state an instruction. Worse than missing — training on outcome-restatements teaches post-hoc narration, the exact failure measured at explanation-sound 0.08. Filter to the instruction subset and report its n, or report both and show which trains better. |
| C5 | Retrospective recovery works | 0.188 / 0.703 are recovery-NEAR, not exact; date-cutoff run PENDING | **No.** Cannot appear as a result. |
| C6 | Our system optimises hits (ZAP70) | MAE 0.489 / Spearman 0.792; docking+pretraining additive | **CONFIRMED but off-thesis** — a FiLMDelta regressor, not R/M. Appendix. Name the withdrawn 0.469 as withdrawn. |
| C7 | R/M factorisation beats joint LLM training | **none**; §5 says R is the bottleneck 18.6× and the split costs ~2× anyway | **The thesis is untested and the pack leans against it.** Must-have. |
| C8 | Property steerability | TPSA +3.62 (259), cLogP −0.63 (146), MW −31.2 Da (75); basicity flat; size↑/rigidity↓ wrong sign | **Yes as stated** ("real but uneven"), if the wrong-signed buckets ship with counts (TBD-5) and CIs. Buckets are read from the reason text, so labels are partly circular with the conditioning — say so. |

## 3. Experiments reviewers will demand and we lack

**Must-have.** (1) **Joint vs factorised head-to-head** — one corpus, one molecule-level split, one
metric set; without it there is no paper-shaped claim. (2) **Leakage decomposition + same-pocket
counterfactual** (§1). (3) **Seeds**: ≥3 with CIs on the PocketPair-9k row, the pocket/blind ablation,
and the four §4b arms — everything is single-run while §6.1 says drift exceeds the effects.
(4) **Frontier LLM zero-/few-shot on the identical contact table**; in 2026 its absence reads as
avoidance, and a split result (wins reasoning, loses validity) is a finding, not a loss.
(5) **Intent recovery / identifiability** (TBD-4, automated k=3 diagonal dominance with CIs) — without
it, "the sentence carries information about the molecule" has no non-circular measurement.
(6) **Blinded chemist panel, rebuttal-sized**: ≥100 items over oracle / R→M / mismatch, ≥2 chemists, κ
reported. The claim is usefulness to chemists; model labellers cannot establish it, and ours already sit
at would_make 0.61 / explanation-sound 0.08. Reduced scope is fine; zero is not.
(7) **Contamination control**: the base models were pretrained on literature and ChEMBL plausibly
containing these pairs and papers, so the date cutoff is needed as a contamination control, not only for
the retrospective claim.

**Rebuttal-able.** (8) **Strong pocket-conditioned generator** (Pocket2Mol / TargetDiff / DiffSBDD):
state the task mismatch — they do de novo, we do analogue optimisation — and run the fair in-house
substitute you already have, REINVENT4 Mol2Mol at matched budget. (9) **Contact-table stability**
(TBD-6); cheap, and reviewers distrust cofolded pockets. (10) **Prospective** (TBD-7): nice-to-have at an
ML venue, must-have at a med-chem journal; promise it, do not gate on it, and never hint at unpublished
wet-lab data.

## 4. Reproducibility and ethics

- **Literature TDM**: name the legal basis (EU DSM Art. 3 / UK TDM / US fair use) and licensed access.
  **Do not redistribute text** — release DOI + ChEMBL `compound_key` + offsets, quotes only at fair-use
  length, and no artefact carrying publisher watermarks (our QA found them on ~90% of packets).
- **ChEMBL** CC-BY-SA 3.0: attribute, share alike on derived tables, pin the release version.
  **PDB**: list all IDs; state crystal pairs carry no potency.
- **Agent supervision**: release labeller model + version + date, the v5.3 brief verbatim, and the labels
  themselves — labellers drift, so the recipe alone is not reproducible.
- **Model-as-judge**: judge, prompt, temperature, seeds; state explicitly that panel numbers are not
  comparable across panels (§6.1); report κ wherever a panel number appears.
- **Splits**: molecule-level holdout is the floor; state that random pair splits leak (71% duplicates in
  the sibling project) and that we do not use them.
- **Compute**: LoRA config, seeds, GPU-hours, the 2×A100 cofolding cost.
- **Dual use**: a generator of bioactive and covalent molecules — state output screening, no
  toxin/select-agent targets, the weight-release decision and its gating, and that every molecule is an
  unsynthesised, unassayed prediction.
- **Human panel**: recruitment, compensation, consent, IRB exemption determination.
- **Retraction hygiene**: name the v2-cond non-reproduction and the withdrawn ZAP70 0.469. Naming your
  own retractions is the cheapest credibility in the paper.

## 5. Ten pages

| § | Content | pp |
|---|---|---|
| 1 | Intro; contributions stated as *measured*, not aspired | 1.00 |
| 2 | Task, complexes-not-pockets, corpora, **with the literature defects in the same section** | 1.50 |
| 3 | Method: R / M / R→M, and why factorise (answer C7 here) | 1.00 |
| 4 | Finetuning lifts a chemistry LLM: main table, seeds, frontier-LLM baseline | 1.25 |
| 5 | **What the pocket governs**: lead 1.000 vs 0.432, land on 0.680 / 0.013 Tc | 1.25 |
| 6 | **Does the intent decide the molecule**: oracle / R→M / counterfactual / none, leakage decomposition, identifiability, property response | 1.75 |
| 7 | Retrospective hit-to-lead, publication-date cutoff | 0.75 |
| 8 | Related work | 0.50 |
| 9 | Limitations + ethics, not a footnote | 0.75 |
|  | | **9.75** |

§5 and §6 are the paper. Cut order if over: ZAP70 → appendix; LLM/base-model ablation → appendix with one
sentence in §4; agentic-vs-extracted reasoning → appendix, keeping only explanation-sound 0.08 in §9;
compress §4c to three rows. Never cut: the pocket null, mismatch < `none`, the 60.7% caveat, §9.

## 6. The abstract I would accept (197 words)

> Medicinal chemists optimise hits by acting on opportunities in the binding pocket, and every analogue
> encodes a design hypothesis. We ask whether a language model can be taught that hypothesis and then act
> on it. We factorise hit-to-lead into a reason model R (complex → design intent) and a molecule model M
> (complex + intent → analogue), trained on crystal-derived pairs with agent-written intents and on design
> intents mined from 1,622 published SAR papers (17,601 pairs; 39.3% of extracted rationales state an
> instruction rather than restate an outcome). Finetuning raises a chemistry LLM from 0.118 to 0.973
> validity and from 3.9% to 46.5% sound reasoning. Conditioning on the chemists' own intent recovers the
> published analogue at 0.919 mean Tanimoto against 0.541 under a mismatched intent (+0.379 paired,
> p = 9.6×10⁻⁵⁰); the deployable R→M pipeline reaches 0.574, so most of the gain is still inaccessible
> without a better reason model. Withholding the pocket collapses groundedness from 1.000 to 0.432 while
> leaving similarity to the reference unchanged (0.679 vs 0.680): the pocket governs what the model may
> claim, not yet what it makes. On a publication-date retrospective R→M recovers [TBD-3]; a blinded
> chemist panel rates R's intents at [TBD-1].

Every number is from the pack; 39.3% and the 0.574 reading are arithmetic on pack numbers.

## 7. Limitations, with no hole left unnamed

> Our strongest conditioning result is also our most fragile. The oracle intents for the crystal and
> potency corpora were written by an agent with the product molecule in view, so part of the 0.919
> Tanimoto may reflect a transformation named in the sentence rather than a hypothesis executed by the
> model; we decompose this with redacted and counterfactual intents, and the residual, not the raw oracle,
> is our claim. The mismatch control is imperfect for the same reason: a foreign pair's sentence is also
> an off-pocket prompt. The pocket result is narrow — withholding the contact table changes what the prose
> may cite (1.000 vs 0.432) but not the molecule (0.679 vs 0.680), a wrong pocket costs 0.013 Tanimoto at
> cohort scale, and distance annotations changed nothing; we therefore claim grounded attribution, not
> pocket-aware generation. The deployable pipeline recovers only a small fraction of the
> oracle-over-mismatch gap, and R's intents are repetitive (uniqueness 0.238); the potency-trained model's
> uniqueness falls to 0.173. `none` is not a neutral baseline — with no sentence the model echoes its
> input in 100% of rows. Steerability is uneven, and two buckets (size ↑, rigidity ↓) move the wrong way.
> Our literature supervision is SAR pairs, not matched molecular pairs, and 60.7% of extracted rationales
> restate an outcome rather than state an instruction, so some of what we learn is post-hoc narration.
> Most quality judgements come from model labellers whose panel-to-panel drift exceeds the differences we
> measure, our chemist panel is small, and we make no cross-panel comparisons. All results are single-run
> unless a confidence interval is given. Our base models were pretrained on literature and ChEMBL that may
> include our test pairs, which the date-cutoff retrospective controls only partially. No molecule here has
> been synthesised or assayed, and a related claim from an earlier system in this programme failed to
> reproduce and is treated as unproven.
