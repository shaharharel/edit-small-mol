# Evidence pack — every number the manuscript is allowed to use

**Rule for every contributor: if a number is not in this file, it does not go in the paper.**
Write `[TBD-n]` instead and add it to the placeholder register at the bottom. Several results in
this programme were retracted after being quoted once; the register exists so that a retraction
costs a placeholder and not a claim.

Status key: **CONFIRMED** = computed and re-checked; **SINGLE-RUN** = computed once, no seed
replication; **PANEL** = model-labeller or chemist panel, with its own known biases; **PENDING** =
experiment running or not started.

---

## 1. The task

Given (a) a hit molecule, (b) a per-atom contact table for that molecule in its target's pocket
(residue, ligand atom, distance), the model emits (i) an analogue and (ii) a sentence stating the
design intent, grounded in named pocket residues.

Two models, trained separately on the same corpus:

- **R** (reason model): pocket + input molecule → design intent sentence.
- **M** (molecule model): pocket + input molecule + intent sentence → analogue.
- **R→M** (joint): R's own sampled intent is fed to M. This is the deployable pipeline.

The contrast is against **direct LLM training** — one model emitting molecule and rationale
together — which is what every row labelled "ChemDFM-14B / ether0 / Llama / SmileyLlama" in the
big table is.

A **complex**, not a pocket, is the unit of structural input: the contact table is ligand-specific
and per-atom, so a target we hold a structure for buys nothing for a ligand we have never folded.

---

## 2. Datasets

| corpus | what it is | size | status |
|---|---|---|---|
| PocketPair-9k | PDB-derived crystal pairs + pockets, agent-labelled intents | 7,692 train rows, 2,954 distinct PDBs | CONFIRMED |
| PocketPair-10k | as above, extended | 9,453 train rows, 3,272 PDBs | CONFIRMED |
| PocketPair-14k | as above, extended; the R/M split corpus | 12,392 train rows, 2,827 PDBs | CONFIRMED |
| PocketPair-21k | ChEMBL within-assay SAR pairs with cofolded complexes and agent-written intents carrying measured potency deltas | 3,992 distinct (molecule, target) complexes | CONFIRMED |
| PocketPair-Literature-1.6k (new) | design intents extracted from 1,622 papers under brief v5.3 | 17,601 resolved pairs; 32,758 doubles (molecule + reason) | CONFIRMED |

**PocketPair-Literature provenance.** Molecules are ChEMBL's own curation under the paper's own
compound labels (retrieved by label, not substructure); reasons are quoted from the paper text;
potency Δ only between two molecules measured in the **same assay** of that paper. 2,000 papers in,
reviews dropped, 1,622 papers out.

**v5.3 brief gates (CONFIRMED, measured on all 1,622 papers):** endpoint stated 100%,
reference_source stated 100%, rationale present 100%, evidence quote present 100%, extraction-meta
leakage 0.00%, repeated `what_changed` 0.0%. Multi-site edits fell 44.8% → 5.5% between v4 and v5.3.

**Known defects of the literature corpus (must appear in the paper):**
- These are **SAR pairs, not matched molecular pairs** — no MMP transformation constraint. The gate
  is Tanimoto + a same-assay potency delta; the extracted `transformation` field is descriptive.
- Explained edits have **small** potency deltas, so delta-ranked pair selection systematically
  excludes them (CONFIRMED, this is why intent mining must precede pair selection).
- Author-label retrieval finds the author's actual pair 32% of the time by SMARTS; ChEMBL's
  `compound_key` is the reliable route (CONFIRMED).
- 100-paper QA: molecules 3.3% contradicted by the text; reasons 90% honest but **60.7% restate an
  outcome rather than an instruction** (PANEL). The v5 brief revision cut overstatement 44% → 15%.

**Complex coverage for the literature corpus (CONFIRMED today):** 17,601 pairs need 6,590 distinct
(input molecule, target) complexes over 658 targets. 27 are already held. 552 of 658 targets have a
ChEMBL sequence; 106 do not (1,995 complexes, 4,967 pairs unreachable without another sequence
source). 369 complexes exceed the 1,187-residue cofold cap and are recorded separately rather than
dropped — an earlier run lost all of EGFR to exactly this cap. Queue: 4,199 jobs ranked by pairs
unlocked; top 1,000 → 43.2% of needed pairs, top 3,000 → 58.6%. **PENDING**: cofolding started
2026-11-12 on two A100s.

---

## 3. Main table (direct LLM training, section 3 of the internal report)

All on the covalent PocketPair corpora. `sensible` / `reasoning` are blind chemist or
model-labeller panels; `invented chem` counts sentences asserting a contact that is not in that
pair's own table.

| arm | data | validity | uniq | echo ↓ | TC→input | scaffold ret. | sensible | reasoning | invented ↓ |
|---|---|---|---|---|---|---|---|---|---|
| ChemDFM-14B base | — | 0.118 | 0.871 | 0.161 | 0.793 | 0.104 | 10.6% | 3.9% | 67.7% |
| ether0-24B base | — | 0.225 | 0.847 | 0.678 | 0.884 | 0.247 | 12.7% | 0.0% | 37.3% |
| Llama-3.1-8B-Inst base | — | 0.519 | 0.750 | 0.059 | 0.737 | 0.761 | 32.4% | 6.5% | 85.3% |
| SmileyLlama-8B base | — | 0.882 | 0.753 | 0.069 | 0.582 | 0.277 | 30.6% | 0.0% | 0.0% |
| ChemDFM-14B | g2 (1,485) | 0.893 | 0.752 | 0.021 | 0.658 | 0.523 | 72.4% | 22.9% | — |
| ChemDFM-14B | g3 (2,326) | 0.958 | 0.777 | 0.032 | 0.683 | 0.482 | 95.8% | 29.9% | 63.2% |
| ChemDFM-14B | g4 (5,366) | 0.966 | 0.751 | 0.004 | 0.644 | 0.462 | 89.2% | 37.6% | 51.5% |
| **ChemDFM-14B** | **PocketPair-9k (7,692)** | **0.973** | 0.757 | 0.000 | 0.658 | 0.455 | **92.4%** | **46.5%** | **42.0%** |
| ChemDFM-14B | PocketPair-10k (9,453) | 0.969 | 0.760 | 0.004 | 0.647 | 0.472 | TBD | TBD | TBD |
| **ChemDFM-14B** | **PocketPair-21k, on 21k val** | **0.985** | 0.173 | — | 0.795 | — | TBD | TBD | — |
| DESRO-8B | PocketPair-14k (12,392) | 0.954 | 0.756 | 0.011 | 0.662 | 0.480 | 90.0% | 33.3% | — |
| ether0-24B schema B | PocketPair-9k | 0.855 | 0.768 | 0.018 | 0.639 | 0.487 | 79.6% | 16.6% | 45.6% |
| Llama-3.1-8B-Inst | g3 | 0.771 | 0.743 | 0.020 | 0.594 | 0.322 | 57.8% | 3.2% | 63.6% |
| SmileyLlama-8B | g3 | 0.908 | 0.761 | 0.055 | 0.588 | 0.332 | 51.1% | 3.8% | 56.0% |

Status: all CONFIRMED except cells marked TBD. The trend the paper can claim: **finetuning on
pocket-conditioned pairs lifts a chemistry LLM from unusable (11.8% validity, 3.9% sound reasoning)
to usable (97.3% validity, 46.5% sound reasoning), and reasoning quality keeps climbing with corpus
size while validity saturates.** The low `uniqueness` of the 21k-trained model (0.173) is a real
cost of the potency corpus and must be reported, not buried.

**Pocket ablation at the model level (CONFIRMED, 21k val, n=200 each):**

| trained on | grounded | selectivity | validity | TC→reference |
|---|---|---|---|---|
| pocket-21k (contact table shown) | **1.000** | **0.388** | 0.985 | 0.679 |
| blind-21k (table withheld) | 0.432 | 0.121 | 1.000 | 0.680 |

So the pocket decides **what the sentence may cite** (grounded 1.000 vs 0.432, selectivity 0.388 vs
0.121) and **not** the molecule (TC→reference 0.679 vs 0.680). This is the single most important
honest result in the programme and the paper must lead the pocket section with it, not against it.

Supporting negative result (CONFIRMED): at cohort scale, swapping in the **wrong pocket** costs
only 0.013 Tanimoto against the true-analogue noise floor. Prose is grounded in the pocket;
molecules largely are not. Distance annotations added to the prompt changed nothing.

---

## 4. Does the stated reason decide the molecule? (R/M arms)

### 4a. On PocketPair-14k, R-split, 262 pairs (CONFIRMED)

| what M was given | median TC to the crystallised analogue | paired Δ vs R's own sentence | sentences inventing a residue ↓ |
|---|---|---|---|
| the chemists' sentence (oracle) | 0.698 | +0.180, p = 3×10⁻²⁷ | 0.0% |
| R's own sampled sentence (R→M) | 0.518 | — | 27.1% |
| a sentence written for another pair | 0.503 | −0.007, p = 0.21 (ns) | 55.7% |

R alone, 262 rows: names a residue in 100% of sentences, 2.24 residues per sentence, groundedness
0.869 micro, 0.729 fully grounded, selectivity 0.421, invented chemistry 13.1%. The chemists' own
sentences score 1.000 / 1.000 / 2.42 / 0.0% — that is the ceiling, and it is a real ceiling.

### 4b. On PocketPair-21k, molecule-level held-out, 400 prompts per arm (CONFIRMED today)

Held out at the **molecule** level: neither the input nor the reference analogue appears anywhere in
training. Target space repeats across papers, so unseen-target holdout leaves 21 targets; unseen
molecules leave 1,469 rows (553 on never-trained targets, 916 on trained targets with new
molecules). Only the sentence handed to M differs between arms.

| arm | sentence given to M | validity | uniqueness | TC→input | **TC→reference** | heavy-atom Δ MAE |
|---|---|---|---|---|---|---|
| M oracle | the chemists' own | 0.963 | 0.626 | 0.677 | **0.919** | 0.374 |
| M ← R (the pipeline) | R's generated intent | 0.965 | 0.238 | 0.746 | **0.574** | 2.622 |
| M mismatch | another pair's sentence | 0.915 | 0.937 | 0.718 | **0.541** | 3.907 |
| M none | nothing | 1.000 | 0.235 | **1.000** | 0.684 | 2.445 |
| R alone | — (emits an intent) | — | — | — | — | groundedness 1.000, selectivity 0.214 |

Readings the paper may make:
- **Oracle 0.919 mean TC→reference, median 1.000**: given the chemists' own sentence, M reproduces
  the exact analogue they made in more than half of held-out cases. Oracle − mismatch = **+0.379
  paired**, p = 9.6×10⁻⁵⁰ (Wilcoxon).
- **A wrong sentence is worse than no sentence**: mismatch sits 0.147 below `none`, p = 6.9×10⁻⁴⁰,
  and validity falls 100% → 91.5%. The model is following the instruction, not decorating.
- **`none` is not a baseline.** With no sentence M **echoes the input** (TC→input = 1.000, 100% of
  rows): it declines to design. Its 0.684 is the input's own similarity to the reference, and every
  property delta under that arm is exactly zero. The honest control is mismatch.
- **R→M (0.574) beats mismatch (0.541) and is far below oracle (0.919).** R's intents carry real but
  partial signal. Its low uniqueness (0.238) says R's sentences are repetitive — a stated limitation,
  and the first target of the next round.

### 4c. Property-level controllability (CONFIRMED, same arms)

Each row classified by what its reason **targets**, read from the reason text rather than the assay;
signed change (generated − input) compared across arms on exactly those rows, paired by input.

| the reason targets | property | rows | oracle | verdict |
|---|---|---|---|---|
| polarity ↑ | TPSA | 259 | +3.62 | moves as asked |
| lipophilicity ↓ | cLogP | 146 | −0.63 | moves as asked |
| size ↓ | MW | 75 | −31.2 Da | moves as asked |
| basicity ↓ | basic N | 31 | −0.03 | flat |
| size ↑ | MW | TBD | wrong sign | fails |
| rigidity ↓ | RotB | TBD | wrong sign | fails |

Claim: steerability is real but uneven, and strongest for properties a single functional-group swap
can move. Under a mismatched reason the targeted property barely budges.

---

## 5. Retrospective and prospective hit-to-lead

### ZAP70 (CHEMBL2803), covalent, Cys346 (CONFIRMED)
- Honest rebuilt baseline MAE **0.723**. The previously circulated 0.469 came from one-leg pair
  leakage worth 0.163 MAE and **must never be quoted again**.
- v3 MAE 0.555; v6 best MAE **0.554**, Spearman **0.726**.
- With Vina docking on 4K2R and pretraining: absolute MAE **0.489**, Spearman **0.792**. Docking and
  pretraining are additive. At 280 molecules every more complex architecture hurt.
- REINVENT4 integration with FiLMDelta anchor scoring: 38K de novo molecules, max predicted pIC50
  8.15; Mol2Mol 6K molecules, 29.6% potent.

### Retrospective recovery (PENDING / PARTIAL)
- R/M split study: R is the bottleneck by **18.6×**; splitting costs about 2× regardless; the
  reported 0.188 / 0.703 are **recovery-NEAR, not exact recovery** (CONFIRMED caveat).
- Publication-date-cutoff retrospective on the literature corpus: **PENDING**, waiting on cofolds.

### Agentic vs extracted reasoning (PANEL, partial)
- Cohort judgement round, 613 candidates over 5 targets: `would_make` 0.61 but explanation sound
  only **0.08**. The labellers are models, not chemists — treat as a model-labeller result.
- 26–40% of "covalent anchor" clauses cite a 2.7–3.8 Å **non-bonded** contact as if it were the bond.
- Residue existence checks pass 92.5%; the pose-distance version of that check is circular.
- Ownership: panel says 6/6, a rigorous atom-ownership check says 2/6; distance-free clauses go
  unscored.
- The `intent` column labelled "specific" measures the template, not the content (p = 1.000 on
  usefulness).

### Human panel on the literature intents (PENDING)
Web tool live; 101 v5.3 cards served; verdicts being collected. No numbers yet.

---

## 6. Hard caveats that must survive review

1. Model-labeller panels and chemist panels disagree, and **panel-to-panel drift exceeds the
   between-arm differences** we are measuring. Every panel number needs its seeded controls read
   first, and comparisons across panels are not permitted.
2. `none` echoes the input; it is not a neutral baseline (§4b).
3. The pocket governs the prose, not the molecule (§3).
4. Random splits on pair data leak: 71% exact duplicate pairs in the sibling project. Molecule-level
   holdout is the minimum acceptable split here.
5. A previous paper claim in this programme (v2-cond, Tables 5/6) did not reproduce post-publication
   and is treated as UNPROVEN. Nothing in this manuscript may rest on it.
6. `site_of_change` is stereo-blind (MCS graph map), so stereo-only pairs get no site rows.
7. The covalent programme's "g7" is base ChemDFM, not our adapter — do not attribute its behaviour
   to a trained model.

---

## 7. Placeholder register

Each `[TBD-n]` in the manuscript gets a line here: what is missing, which experiment supplies it,
and what the hypothesis is. **State the expected direction** so a failed experiment costs us a
revert and not a rewrite.

| id | what | supplied by | hypothesis |
|---|---|---|---|
| TBD-1 | chemist panel on R's literature intents vs the papers' own sentences | panel running on the 101-card tool, then a 400-card round | R scores below the paper ceiling on usefulness but at parity on groundedness |
| TBD-2 | M/R/R→M trained on PocketPair-Literature triplets | training after cofolds land | literature-trained R improves intent usefulness over agent-labelled R; molecule metrics roughly unchanged |
| TBD-3 | retrospective hit-to-lead with a publication-date cutoff | literature corpus + cofolds | R→M recovers subsequently published analogues above a pocket-blind baseline, but below oracle |
| TBD-4 | intent-recovery matrix (k=3 contrasting intents, blind molecule→intent matching) | automated diagonal-dominance version first | diagonal dominance significant but well under 1.0 |
| TBD-5 | size ↑ and rigidity ↓ property buckets, row counts and deltas | rerun of property_response.py with per-arm CIs | both stay wrong-signed; reported as a limitation |
| TBD-6 | 5-sample diffusion re-run on a cofold subset to show contact tables are stable | boltz re-run at `--diffusion_samples 5` | contact lists stable; conclusions unchanged |
| TBD-7 | prospective round | not scheduled | — |

---

## 8. R error analysis and the dictation rate (CONFIRMED 2026-11-12, laptop-only, no GPU)

**There is no oracle in the 14k/21k corpora.** The reference intents were written by an agent with
the analogue in view. Measured on 1,260 val rows by checking whether the sentence names a chemical
group that actually changed between input and analogue (SMARTS on 40 group classes):

- names at least one group: **99.9%**
- names a group that **actually changed**: **86.1%** ← the dictation rate
- per pair: 2.71 groups change, the sentence names 4.91, of which 2.09 are the right ones

So the 0.918 / median 1.000 arm measures **transcription fidelity**, not design. Rename it
"reference (dictated) intent" everywhere. The only genuine oracle is a paper author's stated
intent, i.e. PocketPair-Literature, which is gated on the cofolds.

**Why R→M is weak — four measured failure modes** (349 paired val rows, R's generated intents
against the reference intent for the same pair):

| finding | R | reference |
|---|---|---|
| distinct sentences | **97 / 349 = 0.278** | 349 / 349 = 1.000 |
| most common 6-word opening | **57 rows (16%)**, "replace the meta chlorine with a" | — |
| names a chemical group | 100% | 100% |
| uses an edit verb | 85.7% | 88.0% |
| sentences naming a residue | 94.0% | 82.8% |
| residues per sentence | 1.17 | 1.09 |
| residue in that pair's own table (groundedness) | **1.000** | — |
| overlap with the reference's residues | **0.190** (exact same set **9.7%**) | — |

1. **Mode collapse.** R has a few favourite edits; one opening covers 16% of rows.
2. **Fluency is not the problem.** R names chemistry and uses an edit verb at reference rates.
3. **Right table, wrong residue.** Groundedness 1.000 is near-vacuous — R picks *a* residue from
   the prompt, not *the* relevant one (overlap 0.190, exact match 9.7%).
4. **M is not the bottleneck.** Splitting R's rows by whether its intent hits a reference residue:
   **TC 0.654 (n=73) vs 0.546 (n=262), gap +0.108.** When R picks the right site, M delivers.
5. **Meta-commentary instead of an instruction.** R emits things like "the reported inversion
   therefore cannot be verified" — and repeats it verbatim across different pairs. Inherited from
   the corpus's stereo-pair defect (site_of_change is stereo-blind, see §6.6).

**Caveat on the metric itself.** TC$\rightarrow$reference penalises R for proposing a *different
valid* edit, so 0.570 conflates "R was wrong" with "R was different". The site-hit split above is
the cleaner read, and an identifiability measurement (TBD-4) is the proper one.

**The joint model is currently ahead of the split.** ChemDFM-14B trained jointly on PocketPair-21k
reaches TC$\rightarrow$reference **0.679** on 21k val, against **0.570** for R$\rightarrow$M. The
factorisation is not yet earning its keep on molecular metrics; what it earns is the ability to
intervene on the intent at all.
