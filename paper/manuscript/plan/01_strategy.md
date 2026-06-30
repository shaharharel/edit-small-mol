# Paper strategy

**Title** (locked 2026-06-23): *"RL-tuned chemical language models for covalent ZAP70 lead optimization"*.

## What we are claiming

**Core claim**: we demonstrate end-to-end multi-objective reinforcement learning on a covalent-pretrained chemical language model for the lead optimization of an in-house ZAP70 covalent inhibitor (Mol1, measured pIC50 = 6.59). The pipeline produces a tractable set of synthesizable, geometrically-plausible covalent candidate molecules that pass multi-disciplinary medchem triage and are scheduled for prospective wet-lab validation.

**Sub-claims** (each backed by results section):

1. **Reward design**: a weighted geometric-mean reward over a pair-trained ΔpIC50 predictor, a substructure-retention indicator, and a drug-likeness term yields high warhead-retention rates without hard symbolic constraints (the geometric mean makes warhead presence implicitly mandatory; we derive the formal consequence).
2. **Comparative experimental matrix**: 11 RL cohorts spanning 2 backbone strategies × 3 target panels × 2 base priors give a controlled study of how reward shaping interacts with seed choice. We report quantitative chemistry-output shifts (e.g., 2-aminopyridine substructure: 25% in seeds → 63.8% in ZAP70-trained output vs 1.9% in kinase-broad output → reward target controls motif retention).
3. **Cofold-validated downstream funnel**: 1,684 candidates go through Boltz-2 covalent cofolding with explicit SG–Cβ constraint; ~half satisfy strict covalent geometry filters (d_SG ≤ 2 Å, Bürgi–Dunitz dev ≤ 30°).
4. **Distilled validator (small methods addition)**: an XGBoost model trained on (ECFP4, Boltz-2 cofold metrics) learns to predict iptm (CV r = 0.67) and London mPAE (r = 0.69) from 2D fingerprints alone. This enables a fast pose-quality reward inside RL — demonstrated by a follow-up RL cohort with the distilled validator as a 4th reward term.
5. **Multi-disciplinary triage**: a structured rubric over 3 medchem expert lenses (covalent kinase, kinase SAR, DMPK) narrows the 47-mol shortlist to wet-lab nominees with documented rationale and inter-rater agreement.
6. **Wet-lab validation**: [PLACEHOLDER, scope set by collaborators at London] IC50 screening on a subset of nominated candidates is planned. The exact assay panel (occupancy, kinact/Ki, selectivity, cellular) is to be specified by the collaborators based on the candidate set we deliver. Results section has placeholder scaffolds that will be populated as data returns.

## What we are NOT claiming

* NOT "de novo discovery" of new chemistry — this is **lead optimization** anchored on a measured lead with documented chemistry.
* NOT "rediscovery of the kinase pharmacophore" — the 2-aminopyridine motif enrichment is reward-driven recapitulation of substructure present in 25% of the seed pool, not de novo emergence.
* NOT "warhead grafting" of literature non-covalent leads — Tc-NN analysis shows only ~2.6% of the cohort has Tc ≥ 0.25 to any non-covalent ChEMBL ZAP70 seed; 96% are novel relative to any single ChEMBL seed.
* NOT new architectural contributions — the chemical language model architecture, RL algorithm (REINVENT4 DAP), and Boltz-2 cofold tool are prior published work. Our contribution is the **integrated pipeline** + the **reward design + emergent findings + wet-lab validation**.

## Framing decisions (locked by user)

* **Ambition**: aim higher than pure-methods; combined methods + case study + wet-lab demonstration.
* **Style**: thorough methods with mathematical formalism (advisor is a top ML scientist). Full version now; trim for venue at submission.
* **Audience**: ML-aware chemistry/CADD reviewers + the user's ML advisor.
* **Venue**: TBD; provisional JCIM with stretch to JMC if wet-lab is solid; stretch to Nat Comms if wet-lab includes cellular activity + selectivity panel.
* **Wet-lab planned in parallel**; intro contains a "scheduled wet-lab validation" paragraph; results section has placeholder scaffolds for the assays.

## Reviewer-2 risks (pre-empt explicitly in text)

1. **Predictor circularity** — reward used FiLMDelta, then we report FiLMDelta scores. Mitigation: (a) calibration on held-out ChEMBL ZAP70 binders (planned), (b) report orthogonal Vina/MM-GBSA scores on final picks, (c) wet-lab validation as the ultimate orthogonal check.
2. **k_inact_proxy 4-log spread is largely formula artifact** (the BD-angle term in the proxy is over-penalizing per our QA + expert analysis). Mitigation: state this explicitly in methods, use k_inact_proxy only as a soft gate not a relative-reactivity score.
3. **No baselines / ablations** — needs: pre-RL prior-only sampling, reward-component ablations (drop FiLMDelta term / drop SMARTS / drop QED), ChEMBL ZAP70 baseline comparison, multi-seed RL for headline cohort. Some of these are runnable in days.
4. **Single-snapshot Boltz pose** under cofold constraint — pose noise may inflate the 4-log proxy spread. Mitigation: report sensitivity, discuss as limitation.

## Data preservation note

All cohort CSVs, scored molecules, Boltz cofold CIFs, and the multi-tier-filter dashboard contents are existing experiments to be referenced in the Results section. Do NOT delete anything from `data/tier4_scored/`, `data/boltz_*`, `data/mol1_rl_seeds/`, `models/rl_checkpoints_b/`, or the `/light` deployed report. These are the paper's empirical foundation.
