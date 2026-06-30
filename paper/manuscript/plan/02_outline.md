# Paper outline

**Title**: *RL-tuned chemical language models for covalent ZAP70 lead optimization*

**Estimated length** (full version, pre-trim): ~14 pages excluding figures + supplement.

## Section-by-section claims

### 1. Introduction (~1.5 pages)

* **Bullet 1**: Covalent kinase inhibitors are clinically important; ZAP70 is a tractable T-cell-receptor-signaling target with a hinge-adjacent reactive cysteine (Cys346) and no approved selective covalent inhibitor.
* **Bullet 2**: We have an in-house covalent ZAP70 lead, **Mol1** (measured pIC50 = 6.59, acrylamide warhead, novel chemotype with a THIQ-acrylamide–isoindolinone body and a 2-aminoimidazole hinge head).
* **Bullet 3**: Lead optimization of a covalent inhibitor presents three intertwined design constraints — (i) non-covalent recognition affinity, (ii) covalent warhead reactivity and selectivity, (iii) pre-reactive pose geometry relative to the target cysteine. Existing generative tools optimize one at a time.
* **Bullet 4**: We assemble a multi-objective RL pipeline over a covalent-pretrained chemical language model (REINVENT4 mol2mol covalent FT) with rewards that bind together a pair-trained ΔpIC50 predictor (FiLMDelta, prior work), a warhead-substructure indicator, and drug-likeness; downstream Boltz-2 cofold with explicit covalent constraint screens for pre-reactive pose; a multi-disciplinary medchem rubric narrows to a wet-lab-ready cohort.
* **Bullet 5**: Contributions (numbered):
  1. End-to-end covalent-aware lead-optimization pipeline with full code, configs, and 1.18M scored generated molecules released as a benchmark.
  2. Comparative RL matrix (11 cohorts × 2 backbones × 3 target panels) quantifying how reward target and seed pool shape generative output.
  3. A small distilled validator (XGBoost on Boltz-2 cofold metrics) that enables fast pose-quality rewards inside RL.
  4. Empirical evidence of reward-driven motif recapitulation: known kinase recognition substructures appear preferentially in the output when the FiLM reward is target-aligned.
  5. Prospective IC50 screening of the final candidate set at the partner site.
* **Bullet 6**: Outline / roadmap of the paper.

### 2. Methods (~6 pages) — the bulk; mathematical formalism

* **2.1 Notation and problem statement** — define the chemical-language model, the molecule-space SMILES distribution, and the lead-optimization objective formally.
* **2.2 Covalent-pretrained chemical language model** — architecture (Transformer encoder–decoder, mol2mol), covalent fine-tuning corpus and protocol, sampling distribution.
* **2.3 Pair-trained ΔpIC50 reward (FiLMDelta)** — brief recap of prior work; equations for the FiLM-conditioned delta predictor; cite our prior paper for full derivation.
* **2.4 Multi-objective reward design** — define each component formally (FiLMDelta sigmoid transform, SMARTS retention indicator, QED). Derive the **geometric-mean reward** and prove it implements an implicit hard constraint on any indicator-valued component (one-line proof: any zero in any factor zeros the score → optimal policies have non-zero indicator). This is the only modest novelty bullet on the methods side; lean into it.
* **2.5 RL fine-tuning** — REINVENT4 DAP / DS staged learning, formal objective, hyperparameters table.
* **2.6 Seed selection strategy** — 280 ChEMBL ZAP70 binders (90% non-covalent) + Mol1; justify the non-covalent-seed choice on chemistry grounds (decouples recognition motif learning from warhead chemistry).
* **2.7 Boltz-2 covalent cofold protocol** — explicit covalent SG–Cβ bond constraint specification, sampling settings, confidence metrics extracted (iptm, complex pLDDT, mPAE, ligand_iptm).
* **2.8 Pose-grounded covalent geometry metrics** — formal definitions of d_SG, Bürgi–Dunitz angle deviation, pKa(Cys346) via PROPKA3.
* **2.9 Multi-tier filter cascade** — formalize as composition of predicates over the molecule × pose space; per-tier biophysical rationale.
* **2.10 Distilled pose-quality validator** — XGBoost regression on ECFP4 → Boltz iptm / mPAE; cross-validation protocol; deployment as an RL reward term.
* **2.11 Composite scoring vs strict filters** — design choice for the medchem rubric stage; the composite k_inact_proxy is used as a soft gate only (explicitly flag its 4-log spread as formula artifact, not biology).
* **2.12 Multi-disciplinary expert rubric** — 3-lens review (covalent kinase / kinase SAR / DMPK), nomination protocol, planned inter-rater agreement reporting.
* **2.13 Statistical analysis** — Pearson/Spearman correlations, bootstrap CIs, multi-comparison corrections.
* **2.14 Code, data, and reproducibility** — REINVENT4 TOMLs, scoring server code, distilled validator pickle, full cohort CSVs released; git tag for the paper version.

### 3. Results (scaffold for now; populated as ablation experiments + wet-lab data return)

* **3.1** Pipeline characterization: yield per filter tier, predicted pIC50 distribution shift, novelty Tc.
* **3.2** Ablation matrix: reward-component contributions; cohort-vs-cohort comparison (Murcko-RL × ZAP70-only vs THIQ-RL × ZAP70-only vs kinase-broad).
* **3.3** Reward-driven motif recapitulation: quantitative 2-aminopyridine and related substructure enrichment dynamics; controlled comparison ZAP70-trained vs kinase-broad vs Mol1-only.
* **3.4** Distilled validator demonstration: held-out CV correlations; downstream RL cohort comparison (with/without the distilled validator reward — uses the V100 cohort kicked off in parallel).
* **3.5** Final candidate set: 47 mol shortlist, multi-disciplinary triage results, top picks (T13, M5 currently), inter-rater agreement.
* **3.6** [PLACEHOLDER] Prospective IC50 screening at collaborator site.

### 4. Discussion (~1 page)

* What the pipeline does well; honest limitations (single-snapshot Boltz, predictor circularity, no architectural novelty); what wet-lab data does and doesn't change.

### 5. Limitations (~half page, explicit)

* Predictor circularity; single-seed RL for some cohorts; FiLMDelta extrapolation regime for scaffold-hopped chemistry; k_inact_proxy known artifact; Boltz pose noise.

### 6. Conclusions

* Brief, restate contributions, point at next-paper directions (geometry-constrained generative architecture for paper 2).
