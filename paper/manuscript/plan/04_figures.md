# Figure plan

Target: 4–5 main figures + supplement. Each figure tells one claim; the caption is the one-line scientific point of the section.

## Figure 1 — Pipeline and yield funnel (the engineering)

**Claim**: End-to-end design pipeline produces a tractable shortlist from millions of generated molecules.

**Panels**:
- (a) Block diagram: covalent-pretrained CLM → RL fine-tuning (per cohort) → multi-tier filter cascade → Boltz-2 covalent cofold → multi-disciplinary triage → final candidate set → wet-lab screening (placeholder)
- (b) Yield funnel for the headline cohort (e.g., thiq_rl_exp2_zap70): 50K generated → L0 valid → L1 drug-like → L2 similarity-gated → L3-L4 cofold-geometry-valid → L5 ranked → triaged → wet-lab nominees. Numbers labeled at each step.
- (c) Per-cohort yield comparison: 11 cohorts × pass-rate at each filter level (heatmap or grouped bars).

Where data lives: `data/tier4_scored/F4_boltz_full.csv` + per-cohort scored CSVs + `paper/manuscript/plan/03_data_inventory.md`.

---

## Figure 2 — Reward design and the implicit warhead constraint (the modest methods novelty)

**Claim**: A weighted geometric-mean reward over a soft pIC50 term, a binary substructure indicator, and a drug-likeness term turns the indicator into an implicit hard constraint, yielding effectively 100% warhead retention without any rule-based filter inside the generator.

**Panels**:
- (a) Reward function diagram with explicit formula: `R = (FiLMpIC50_σ)^0.5 · (𝟙[SMARTS match])^0.4 · (QED)^0.1`. Annotate that the geometric mean → any-zero-zeros-all → policy must learn to produce non-zero indicator (i.e., warhead present).
- (b) Empirical confirmation: warhead-retention rate across RL training steps for cohorts with vs. without the SMARTS term in the reward (will require the ablation RL run from §3.2 — placeholder for now).
- (c) ΔpIC50 distribution shift seed → output, per cohort.

---

## Figure 3 — Reward-driven motif recapitulation (the emergent finding)

**Claim**: The output of target-specific RL preferentially preserves chemical substructures that the FiLMDelta predictor has learned to associate with the target — illustrated by 2-aminopyridine enrichment specifically in ZAP70-trained cohorts (vs. broad-kinase or Mol1-only controls).

**Panels**:
- (a) Stacked bar: 2-aminopyridine fraction in (seeds, ZAP70-trained output, kinase-broad output, Mol1-only output). 25% / 63.8% / 1.9% / 0%.
- (b) Per-cohort breakdown for substructure family (anilinopyridine_2/3/4, aniline_pyrimidine, etc.) showing the motifs are concentrated in ZAP70-trained THIQ cohorts.
- (c) Optional: 2-aminopyridine frequency as a function of RL training step (would require checkpoint-level sampling — schedule as ablation).

---

## Figure 4 — Distilled validator and downstream cohort

**Claim**: A small XGBoost model learns to predict Boltz-2 cofold confidence (iptm) and London mPAE from 2D ECFP4 fingerprints (CV r > 0.65), enabling fast pose-quality rewards inside RL. A follow-up RL cohort using this as a 4th reward term shifts output toward better predicted cofold geometry without sacrificing potency.

**Panels**:
- (a) Predicted vs measured scatter on held-out CV: iptm (r=0.671), mPAE_london (r=0.692).
- (b) RL cohort comparison: baseline (3-term reward) vs new (4-term with distilled validator) — distributions of generated mols' predicted iptm and FiLM pIC50.
- (c) Final-cohort survivor count comparison: baseline 47 vs new cohort survivors.

(b)-(c) placeholders until V100 cohort returns.

---

## Figure 5 — Final candidate set and multi-disciplinary triage

**Claim**: A documented multi-disciplinary review process (3 medchem expert lenses) narrows the 916 visible survivors to 47 cohort candidates and 2–6 wet-lab nominees with explicit per-pick rationale.

**Panels**:
- (a) Top picks (T13, M5, +4 backups): 2D structure + Boltz cofold pose overlay on ZAP70 Cys346 + key feature table (pIC50, d_SG, BD, LE, LLE, P_kinase).
- (b) Expert rubric agreement matrix: heatmap of (mol × expert) verdicts (ADVANCE / CONDITIONAL / REJECT) with inter-rater agreement (Cohen's κ, placeholder until re-rated blindly).
- (c) Mol1 vs top-pick Boltz pose overlay (zoomed at Cys346 pocket) — illustrates lead-optimization geometry preservation.

---

## Supplementary figures

- S1: Reward weight sensitivity (RL outcome vs FiLM/SMARTS/QED weights).
- S2: ChEMBL Tc-NN distribution showing the cohort is 96% novel (Tc < 0.25) — supports "not warhead grafting" framing.
- S3: PrexSyn synthesis route examples for top 3 picks.
- S4: k_inact_proxy formula sensitivity analysis — show the 4-log spread is dominated by the BD term, motivating its use as soft gate only.
- S5: Per-cohort distributions of every multi-tier filter metric.
- S6: Boltz cofold confidence stratified by whether covalent constraint was satisfied (d_SG ≤ 2 Å).
- S7: Wet-lab placeholder figures (occupancy curves, kinact/Ki fits, kinome scan) — populated when data returns.
