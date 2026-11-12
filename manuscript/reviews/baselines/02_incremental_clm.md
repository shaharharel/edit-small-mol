# Baseline assessment 02 — Incremental fine-tuning of chemical language models (IFT / "incremental-clm")

**Paper.** Hörmann T, Mayer D, Lewandowski M, Hunklinger A, Wein T, Merk D.
*Structural optimization of drug molecules with incrementally trained language models.*
**Nat Commun 17, 3456 (2026)**, DOI 10.1038/s41467-026-71591-w. Received 16 Sep 2025, accepted 24 Mar 2026, published 11 Apr 2026. Open Access (CC BY 4.0).
Full text **reached** (nature.com HTML, open access; read in two passes). Code repo reached and its file tree enumerated via the GitHub API. Supplementary Information PDF, Supplementary Data 1 (XLSX) and the Source Data XLSX were **not** opened — numbers that live only there are flagged below.

**Verdict in one line: not a baseline for us. Different task shape, no structure, no rationale, and it cannot be given our input. It is a RELATED-WORK citation and, at best, the source of one borrowed metric (potency–perplexity correlation). Do not promise reviewers a head-to-head.**

---

## 1. What the paper actually does

- **Task.** Ligand-based *structural optimisation* of a chemotype: take a published SAR series for one target, bias a SMILES generator toward the potent end of that series, and sample new analogues. No oracle, no docking, no QSAR scoring in the loop — the authors stress the method is "external-scoring-free".
- **Input.** A **whole SAR series**: a CSV of `identifier, smiles, value` (value = pEC50, ascending) with 25–132 molecules per series. Nothing else. **No protein structure, no pocket, no sequence, no contacts anywhere in the pipeline.**
- **Output.** SMILES strings only, ranked by model perplexity. **No text, no rationale, no residue claims.** The medicinal-chemistry interpretations in the paper (TZD bioisostere, regiochemistry) are the authors' own post-hoc analysis, not model output.
- **Method.** LSTM chemical language model pretrained on ChEMBL, then **fine-tuned in 4–5 successive steps on potency-ordered "boxes"** of the series (weakest box first, most potent box last), 100 epochs per step, LR 1e-4. Sampling is multinomial at T=1.0, then perplexity ranking plus Tanimoto/known-compound filters.
- **Architecture.** PPARγ: Embedding(64) → BN → LSTM(1024) → LSTM(256) → BN → Dense(64), **5,789,297 params**. RORγ: 3 × LSTM(1024). TensorFlow 2.11, Python 3.10.7, RDKit 2023.05.
- **Training data.** Pretraining ChEMBL v32: 496,056 molecules → 448,265 retained after PPAR-activity/similarity filtering (177,101 unique scaffolds; 47,791 excluded). Fine-tuning series from ChEMBL + BindingDB v2023.06: 27 PPARγ series of 25–79 molecules (intra-series Tanimoto 0.30–0.77, spanning 1.3–4.3 log units of potency), and 10 RORγ clusters of 78–132 molecules (internal Tanimoto 0.25–0.47).
- **Headline metrics.**
  - **Rediscovery score** (retrospective, holdout = most potent 25 % of each series). Three components: rediscovered holdout molecules in the top-30 designs by perplexity (N_R/30); their sampling frequency (ΣF_R/ΣF_30); and holdout coverage (N_R/N_hold). Incremental FT beat naive CLM, one-shot FT on all templates, and FT on the most-potent subset in *almost all* 27 series, **p < 0.001** (linear mixed model, dataset as random effect). Optimum = 4–5 boxes. **The combining formula and the per-method scalar scores are not in the HTML main text — they are in the Methods equations/Source Data, which I did not open. Treat "how much better" as unquantified.**
  - **Holdout perplexity** lower than all-template FT (p < 0.001) and than best-subset FT (p < 1e-5).
  - **Generation statistics**, prospective PPARγ: 5 runs × 10k = **50,000 designs, 99 % valid, 28 % unique**; of the top-1,024 valid unique designs, **799 (78 %) on the intended scaffold**. Novelty % not reported in the main text.
  - **Prospective wet-lab result (the real headline).** 9 PPARγ designs synthesised and assayed: compounds 2, 5, 6, 8, 9 at **EC50 0.6–3.1 nM vs 37 nM for reference A (12–62×)**; 1, 3, 7, 10 up to **6.5× higher efficacy**. RORγ: 2 designs; compound **27 ≈ 20× more potent** than its nearest template B, Tanimoto 0.302 to the most potent template C; compound 28 markedly less active. **11 CLM designs synthesised in total.**

## 2. Is it a real comparison for us? No — four independent mismatches

| Axis | IFT (this paper) | Ours |
|---|---|---|
| Task unit | one SAR **series** (25–132 potency-ranked molecules) → a *pool* of analogues | one **hit molecule** → one analogue |
| Conditioning | ligand SMILES + ordinal potency only | SMILES + 16-field per-atom contact table for the ligand in its pocket |
| Structure | **none** (explicitly ligand-based) | crystal or Boltz cofold, required |
| Rationale | **none**; SMILES only | NL design intent naming residues, emitted *before* the SMILES, teacher-forceable |
| Target transfer | per-target: a new target wants its own pretrain (actives filtered out) + its own series | one model, 32 held-out targets |

- **Could it run on our inputs at all? No.** Our unit of input is a single molecule plus a contact table. IFT has nowhere to put a contact table, and it cannot start from one molecule — the potency *ordering across ≥25 molecules* is the entire training signal. Feeding it our val rows one at a time degenerates to sampling from the pretrained prior.
- **It also cannot be scored on our metrics.** Residue specificity and residue groundedness are undefined for a model that emits no residues, so our two headline differentiators (0.417 vs 0.200 baseline; 1.000 → 0.432 under pocket ablation) have no IFT counterpart. Our pocket-ablation argument has no analogue either: there is no pocket to ablate.
- **Converse check (is their evaluation runnable by us?)** Partly — see §6. Their retrospective rediscovery protocol is the one place the two systems touch.

## 3. Availability — genuinely good, and that is the one thing in their favour

- **Repo:** https://github.com/AkMerk/incremental-clm — **MIT licence**, archived at Zenodo DOI 10.5281/zenodo.18773078. 19 stars, last push 2026-04-09, ~249 MB checkout.
- **Pretrained checkpoints ARE shipped**, not just training code: `example_workspace/pparg_2l_rnd.h5` (23.2 MB, the PPAR-filtered 2-layer randomized-SMILES pretrained CLM), plus the five prospective incremental checkpoints `runs/prospective/runs/{0..4}/models/4_7.24-8.36_(25).h5` and the five one-shot baselines `runs/prospective_all/runs/{0..4}/models/all.h5`. Training/val/test SMILES for the PPARγ pretrain are in the repo (`data/training/pparg/{train,val,test}.txt`, ~25 MB total), as is the prepared SAR CSV and the prospective design output (`finals/prospective/top_all.smi`, `variants.csv.zst`).
- **No RORγ checkpoint** in the tree (3-layer model); only PPARγ.
- **Requirements:** Python 3.10, TF 2.11 (Linux) / 2.13 (macOS), ~16 GB VRAM, sqlite3/pigz/mawk/pv. CLI `ift`. Demo run ~20 min; full pipeline > 200 min. Pretraining from scratch: 2-layer randomized = 120 epochs × ~3 min/epoch on CUDA (~6 h) or ~10 min/epoch on Metal (~20 h); the 10× augmented 3-layer config is ~80 min/epoch on CUDA, i.e. days.
- **Caveat:** the README cites a `requirements.txt` that is not in the tree (only `pyproject.toml`); `ift final` is implemented for randomized-SMILES mode only. Reproducing *their* result is well supported; repurposing it is not.

## 4. How we differ — the one or two sentences for the paper

> IFT conditions a SMILES language model on the *ordinal potency ranking of an entire published SAR series* and emits molecules only; our model conditions on a *single* hit plus its measured per-atom pocket contacts and emits a residue-level design intent before the analogue, so the proposal is both structure-grounded and auditable. The two therefore address different stages: IFT exploits a series that already exists, whereas we address the point where only one hit and one structure do.

(Reviewer-proof corollary worth stating explicitly: IFT's signal *requires* ≥25 congeneric molecules with assay-matched potencies, which is precisely the data a hit-to-lead starting point does not have.)

## 5. Who wins on what

**Where we lose, and should concede it in print:**
- **Prospective validation.** They have 11 synthesised compounds with sub-nanomolar EC50s and a 12–62× potency win over the reference. We have zero wet-lab molecules. This is the gap a medicinal-chemistry reviewer will lead with, and no retrospective metric answers it.
- **Uniqueness/diversity of output.** 50k samples at 99 % validity / 28 % unique gives them ~14k distinct candidates per campaign; greedy decoding gives us exactly one analogue per input. Their 0.99 validity also edges our 0.985. If a reviewer asks for enrichment-style metrics, we currently cannot play — sampling (nucleus/temperature + dedup) is the cheapest fix and we should do it before submission.
- **Potency-directedness.** Every IFT training signal is a measured potency ordering. Our PocketPair-21k pairs are oriented weak→strong, but our reported metrics are similarity/groundedness, not realised potency. They can say "our molecules are more potent"; we currently cannot.

**Where we win:**
- **Applicability regime.** 1 molecule + 1 structure vs ≥25 potency-ranked congeners. On our 32 val targets, only 21 have ≥25 unique molecules even pooled across chemotypes (min 7, max 116 — measured from `data/sft_out_potency_21k/sft_pocket.jsonl`), and the paper's series additionally demand intra-series Tanimoto ≥0.30 from TMAP clustering. IFT is simply inapplicable to roughly a third of our evaluation set and to any genuinely new chemotype.
- **Explanation.** Residue specificity 0.417 vs a matched-count distance-sort baseline of 0.200, and groundedness 1.000 collapsing to 0.432 when the contact table is withheld, are claims IFT cannot make at all. This is our differentiator and it is uncontested.
- **Structure use.** Demonstrated causally on our side (the ablation). IFT has no structural channel, so it cannot transfer to a target with structure but no series — e.g. a fresh cofold.
- **Scaffold adherence under weak data.** Their 78 % on-scaffold rate is achieved *given* a 25–79-molecule series; our TC→reference 0.679 is achieved from one molecule. Note honestly that 0.679 sits *below* our own "any other real analogue" null of 0.700, so this is a parity claim, not a win — do not oversell it.
- **Cost.** One LoRA over one base model covers 32 targets; IFT re-pretrains and re-fine-tunes per target/series.

## 6. Implementation cost, and the nearest fair comparison

**Running IFT on our 21k inputs: not possible.** (Note also that the 21,279-row split is *train*; the held-out split is **2,795 rows over 32 targets**, so "21k val" is a misnomer we should fix in our own text.) There is no code path that accepts a single molecule or a contact table, and nothing to compare on residue metrics. Writing an adapter would mean inventing the series, i.e. inventing the baseline — do not do it.

**Nearest fair comparison (recommended, ~1.5–2 days of work, 1× 16 GB GPU):** run *their* retrospective rediscovery protocol on *their* terms and insert our model into it.
1. Pick the subset of our val targets where an IFT-legal series exists: group val molecules by target, cluster by scaffold/Tanimoto to intra-series ≥0.30, keep clusters ≥25. From the counts above this yields at most 21 of 32 targets and fewer after the similarity constraint — **state the shrinkage, it is itself a result**. Absolute potencies are not in `sft_pocket.jsonl` (only `potency_gain_log`); they must be pulled back from the ChEMBL assays PocketPair was built from (~2–3 h with the existing extraction code).
2. Hold out the most potent 25 % per series, run `ift setup`/`ift run all` from the shipped `pparg_2l_rnd.h5` (~10–30 min per series on GPU; a fresh kinase-filtered pretrain would be ~6 h CUDA if a reviewer demands no PPAR-leakage, and is probably worth doing for one target only).
3. Score both systems by **holdout rediscovery and by Tanimoto-to-holdout**, the only metrics both can produce. Expectation to pre-register: **IFT wins where a dense series exists; we win on target coverage and are the only system with a residue-level rationale.** Do not expect to beat it inside its own regime, and say so.

**Blockers:** absolute potency values must be recovered; series construction is a judgement call that will be attacked unless the clustering is scripted and reported; the paper's own rediscovery-score scalars are in Supplementary/Source Data, so our reimplementation of the metric will not be bit-comparable to their figures and we should compare methods *within* our run only.

## 7. Flagged / unverified

- Rediscovery-score combining formula and all per-method scalar values — in Methods equations / Source Data XLSX, **not read**. Only the significance levels (p<0.001; p<1e-5) are verified from the main text.
- Novelty percentage of generated designs — not found in the main text.
- Exact sizes of the two *prospective* series (benzimidazole PPARγ, arylsulfonamide RORγ) — Supplementary Tables 1 and 3, **not read**.
- Compound 28's EC50 — described only as "markedly less active" in the text I read.
- Whether `example_workspace/pparg_2l_rnd.h5` is literally the checkpoint used for the published prospective run is inferred from the demo invocation in the README, not stated.
- Article volume/page (17, 3456) came from the page's citation block, not Crossref, which returned title/authors/date only.
