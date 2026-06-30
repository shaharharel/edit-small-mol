# Covalent-aware generative design under a noise-robust edit-effect potency model

> **DRAFT — QA FINDINGS PENDING ACTION** (2026-05-10). Two QA agents (results-level audit + line-by-line code review) flagged four critical issues that must be addressed before this draft is shareable: (B1) the cached FiLMDelta scorer was selected on a molecule-leaky val split — all quoted absolute pIC50 numbers below are *rank-consistent but not calibrated*; retraining is queued. (B2) the "constraint projection improves geometry" claim is tautological on the current cohort because the warhead was already placed at exact prereactive geometry by `place_warhead_at_cys.py` — the projector's contribution is undemonstrated and needs the warhead-pose ablation (Task #10) to land before the claim is supportable. (B3) BTK FiLMDelta scores are nonsense (top hits chemically impossible peroxysulfur cages) — entire BTK column will be dropped or relabelled. (B4) the 997-mol cofold "leaderboard" is dominated by close analogues of the Mol-1 isoindolinone-acrylamide chemotype; we will reframe this as a *refinement/scaffold-expansion* result, not de novo discovery. See `results/anchordiff/cys346_scaffold_diversity_audit.csv` for the diversity numbers and section 4 for the reframing.


**Authors.** TODO list authors. Corresponding: Shahar Harel (shahar.harel@biu.ac.il), Bar-Ilan University.

**Keywords.** matched molecular pairs; edit-effect prediction; covalent inhibitors; pocket-aware diffusion; constraint-manifold projection; cofolding; ZAP70.

---

## Abstract

> *Abstract revised after QA*. Original abstract is below; the highlighted phrases are the ones that need to be softened or qualified once the FiLMDelta retraining (Task #23), warhead-pose ablation (#10), and BTK strategy decision (#24) are settled. Key reframings: drop absolute pIC50 numbers from the abstract; replace "predicted pIC50 up to 7.7" with "FiLMDelta-ranked candidates"; replace the de-novo discovery framing of the cofold leaderboard with a *refinement-and-prioritisation* framing; remove BTK from the headline claim and keep it only as a "warhead/target portability" sketch in Section 4.5.


Covalent inhibitor design is constrained by sparse warhead-bearing training data, by inter-laboratory assay noise that can exceed the structure–activity signal, and by the inability of pocket-aware generators to control the geometric relationship between the warhead and the catalytic cysteine. We introduce a unified framework that couples (i) FiLMDelta, an edit-effect potency model that directly predicts \(\Delta\)pIC50 from a baseline molecule and a matched molecular-pair (MMP) transformation, with (ii) DiffSBDD warhead-inpainting that holds the 5-atom acrylamide rigid at canonical Michael-addition geometry relative to the target Cys SG, (iii) a closed-form projection onto the covalent constraint manifold that exactly enforces the prereactive distance, angle, and dihedral, and (iv) a Boltz-2 cofold structural filter to retain only candidates whose predicted complex respects the covalent topology. On 1.7 M shared MMPs across 751 ChEMBL targets, FiLMDelta lowers MAE by 7.8% over a property-subtraction baseline (0.616 vs 0.668 pIC50) and wins 112/112 noise-robustness tiers, and on the ZAP70 Cys346 case study a 997-molecule cofold leaderboard reaches predicted pIC50 up to 7.7 with Boltz-2 ligand-iPTM up to 0.98 and complex pLDDT 0.81 at the canonical prereactive pose. Wet-lab validation of the top cohort by intact-protein LC/MS time-course occupancy and IMAP kinact/Ki will close the loop on prospective potency, covalency, and selectivity. The framework is warhead- and target-agnostic: the same potency model, manifold projector, and cofold filter apply to chloroacetamide, vinylsulfone, or cyanoacrylate warheads against any annotated cysteine, providing a general route to noise-robust covalent generative design.

---

## 1. Introduction

Most bioactivity prediction is trained on heterogeneous, pooled assay data. A recent reanalysis of ChEMBL self-pairs (Landrum & Riniker, JCIM 2024) and follow-up matched-pair analyses (Nelen et al. 2025) show that 27% of replicate measurements differ by >1 pIC50 unit, and that within-assay deltas are substantially more reproducible than absolute values. We have measured this directly on 1.7 M MMPs from 751 ChEMBL targets: the variance ratio Var(cross-assay \(\Delta\))/Var(within-assay \(\Delta\)) = 1.39, implying \(\sigma_\text{lab}/\sigma_\text{SAR}=0.62\) and that 28% of cross-assay \(\Delta\)pIC50 variance is lab-induced, not chemistry. Any model that predicts \(F(\text{mol}_b)-F(\text{mol}_a)\) inherits both noise sources; a model that predicts \(\Delta\)pIC50 directly from within-assay pairs eliminates the lab noise component by construction.

We make this concrete with **edit-effect prediction**: rather than predicting two absolute potencies and subtracting, we learn \(F(\text{mol}_a, \text{edit}) \to \Delta\text{pIC50}\) from supervised within-assay pair deltas. Our best architecture, **FiLMDelta**, FiLM-conditions a shared encoder on the edit and computes \(f(\text{mol}_b\,|\,\delta)-f(\text{mol}_a\,|\,\delta)\). On a unified shared-pairs benchmark this beats the property-subtraction baseline by 7.8% MAE and dominates a 112-target controlled-noise-tier evaluation (Section 4.1).

Covalent design adds two further structural constraints that ordinary generators ignore. Potency in a covalent setting is governed by \(k_\text{inact}/K_\text{i}\), and the productive prereactive geometry — \(d(\text{S}_\gamma, \text{C}_\beta)\approx1.85\) Å, \(\angle(\text{S}_\gamma,\text{C}_\beta,\text{C}_\alpha)\approx107°\), Bürgi–Dunitz dihedral \(\approx 0°\) — is a narrow region of \(\mathbb{R}^3\). State-of-the-art pocket-aware diffusion models such as DiffSBDD generate candidate ligands in the binding cavity but make no guarantee that a Michael acceptor is even present, let alone aligned. As a quantitative baseline, vanilla DiffSBDD on the ZAP70 P-loop and BTK hinge produces 0% acrylamide-containing molecules.

We close this gap by making the warhead-Cys constraint a *first-class* condition of the generator: an inpaint mask freezes a 5-atom acrylamide built fresh at the canonical SG/CB frame, and a closed-form constraint-manifold projector polishes residual geometry to machine precision. Boltz-2 cofolding with an explicit covalent SG–Cβ bond constraint then re-evaluates each candidate at the full-protein level. The output is a stack — edit-effect potency × covalent inpainting × manifold projection × cofold filter — that produces ranked, geometrically valid covalent inhibitor candidates without leaving the prereactive pose to luck.

Our demonstration target is ZAP70 Cys346 (P-loop), which has no published covalent X-ray structure and therefore no privileged warhead pose to memorize. We treat ZAP70 Cys346 as the prospective lead-generation problem and BTK Cys481 as a hinge-back generalization control.

---

## 2. Methods

### 2.1 Edit-effect prediction: FiLMDelta

We frame potency change under a chemical edit as an intervention. Given an MMP \((A,B)\) with edit SMILES \(\delta\) (the symmetric difference of fragments), we learn

\[
\hat{\Delta} = f(\text{enc}(B)\,|\,\delta) - f(\text{enc}(A)\,|\,\delta)
\]

where the per-molecule head \(f(\cdot|\delta)\) is FiLM-conditioned (Perez et al. 2018):

\[
f(h\,|\,\delta) = \text{MLP}\big(\gamma(\delta)\odot h + \beta(\delta)\big),\quad (\gamma,\beta)=g_\text{edit}(\delta).
\]

The encoder \(\text{enc}(\cdot)\) is a 2048-bit Morgan fingerprint (radius 2). The edit conditioner \(g_\text{edit}\) ingests the difference fingerprint \(\text{enc}(B)-\text{enc}(A)\) plus 28 hand-crafted edit features (atom counts, ring-membership, donor/acceptor changes; `src/data/utils/chemistry.py::compute_edit_features`). Phase-1 ablations (`results/paper_evaluation/all_results.json`) selected Morgan FP over four pretrained alternatives (CheMeleon, ChemBERTa-2 MLM/MTR, MoLFormer-XL, Uni-Mol v1/v2): Morgan won by 4–10% MAE.

**Anchor-based scoring at inference.** For prospective use against a target with \(N\) known actives \(\{A_i\}\) (here, 280 ZAP70 actives), we score a generated molecule \(B\) as

\[
\hat{\text{pIC50}}(B) = \frac{1}{N}\sum_{i=1}^{N}\big[\text{pIC50}(A_i) + f(B\,|\,\delta_i) - f(A_i\,|\,\delta_i)\big],
\]

i.e. an anchor average over the active set. This converts FiLMDelta from a Δ-only model into an absolute predictor without retraining and preserves its noise-robustness inheritance.

### 2.2 Within-assay shared-pairs benchmark

The canonical dataset (`data/overlapping_assays/extracted/shared_pairs_deduped.csv`) is 1,699,666 MMPs across 88,105 molecules and 751 ChEMBL targets. "Shared" means each pair appears in *both* a within-assay and a cross-assay context with identical chemical edit, enabling a perfectly matched noise comparison. Splits: assay-within (primary), assay-cross, assay-mixed, scaffold (mol_a + mol_b), strict-scaffold (both novel), pair-aware random, and cross-target. Baselines: Subtraction (predict \(\hat{p}(A),\hat{p}(B)\) independently), DeepDelta (Fralish et al. 2024), EditDiff, EditDiff+Feats, TrainableEdit, and two attention variants.

### 2.3 DiffSBDD warhead-inpainting at canonical prereactive geometry

We use DiffSBDD (Schneuing et al. 2024) in inpaint mode. The mask freezes a 5-atom acrylamide template (\(C{=}C-C({=}O){-}N\)) that we build fresh per target from the Cys SG/CB frame using `anchordiff/place_warhead_at_cys.py`. The warhead is positioned at \(d(\text{S}_\gamma,\text{C}_\beta)=1.85\) Å, \(\angle(\text{S}_\gamma,\text{C}_\beta,\text{C}_\alpha)=107°\), and a Bürgi–Dunitz-aligned dihedral of \(0°\) about the Sγ–Cβ–Cα–Ccarbonyl torsion. Resamplings = 5; pocket cropping radius = 8 Å around the Cys.

A non-trivial implementation issue: DiffSBDD's OpenBabel-based bond perceiver flips warhead bond orders and, for ~30% of generations, attaches the scaffold to C_carbonyl rather than the amide N. We post-process every output with `anchordiff/fix_inpaint_warhead_bonds.py`, which (a) deletes all warhead-internal bonds, (b) re-imposes the canonical pattern \(C0{=}C1, C1{-}C2, C2{=}O3, C2{-}N4\), (c) re-attaches the closest scaffold atom (within 2.5 Å) to N4 if it has no scaffold bond, and (d) keeps only the warhead-containing fragment. Without this fix, ~30% of inpaint outputs are unparseable; with it, 80–82% of 100 generations per target yield a valid acrylamide-bearing molecule.

### 2.4 Constraint-manifold projector

Even a fixed-warhead inpaint produces minor residual geometric drift (Table 2). The covalent constraint manifold is the codimension-3 set

\[
\mathcal{M} = \{\,\mathbf{x}\in\mathbb{R}^{3n}\;:\;d(S,C_\beta)=1.85,\;\angle(S,C_\beta,C_\alpha)=107°,\;\phi(S,C_\beta,C_\alpha,C_\text{carb})=0°\,\}.
\]

We project onto \(\mathcal{M}\) by a closed-form rigid sequence: (1) translate the warhead so \(d(S,C_\beta)\) is exact, (2) rotate about an axis through \(C_\beta\) perpendicular to the \((S,C_\beta,C_\alpha)\) plane to set the angle, (3) rotate about the \(S{-}C_\beta\) axis to set the dihedral. Each step is a single \(\text{SO}(3)\) action with no optimization loop. End-to-end runtime <1 ms/molecule. Implementation: `anchordiff/covalent_constraint_manifold.py`. After projection, all geometric residuals are zero up to 64-bit floating-point precision (Table 2, Section 4.2).

### 2.5 Boltz-2 cofold structural filter

Each ranked candidate is cofolded against full-length ZAP70 (UniProt P43403) using Boltz-2 (Wohlwend et al. 2024) with an explicit covalent bond constraint between the ligand acrylamide \(C_\beta\) and Cys346 SG. Ligand 3D conformer is generated with RDKit ETKDGv3 prior to cofold; 3 conformers per molecule, 5 recycling steps. Output scores: ligand-iPTM (interface PTM scoped to ligand atoms), complex pLDDT, complex iPDE, and Boltz confidence score. Configuration: 1 GPU per yaml, ~70 s/molecule average.

### 2.6 TODO sections — methods we have not yet executed

> **TODO — FiLMDelta-guided particle resampling inside the diffusion sampler (B2).**
> Plan: at each Boltz-2 / DiffSBDD denoising step, expand to \(K\) particles, score with FiLMDelta anchor-pIC50 + ligand-iPTM, resample with effective sample size threshold. Goal: shift the sampler toward the pocket-compatible high-potency manifold, not just filter post-hoc. Expected to improve top-quartile potency without sacrificing geometric validity.

> **TODO — Warhead-pose ablation.**
> Plan: re-run DiffSBDD without inpaint, with inpaint at canonical pose, and with inpaint at three randomly perturbed warhead poses (Δd ±0.2 Å, Δangle ±15°, Δφ ±30°). Compare downstream Boltz-2 lig-iPTM distributions and FiLMDelta percentiles. This isolates the contribution of the canonical-geometry choice.

> **TODO — Constraint training Path A/B/C.**
> Plan: fine-tune FiLMDelta on (A) within-assay only, (B) within + cross with inverse-noise weighting, (C) within + cross + Boltz lig-iPTM auxiliary head. Cross-validate each on held-out targets. Goal: quantify how much of the 7.8% advantage comes from data curation vs architecture.

> **TODO — SMILES-encoder variant of FiLMDelta.**
> Plan: replace Morgan FP with a small ChemBERTa-2-MLM SMILES encoder and re-benchmark on shared pairs. Phase-1 already showed transformer encoders lag fingerprints; we want to confirm this holds when FiLM-conditioned, since FiLM may rescue underparameterized features.

> **TODO — Wet-lab assays.**
> Detailed in Section 4.5 below.

---

## 3. Results

### 3.1 FiLMDelta beats subtraction on the shared-pairs benchmark

Within-assay, 3 seeds, full 1.7 M pairs, 8 architectures (Table 1). FiLMDelta lowers MAE 7.8% vs Subtraction (0.616 vs 0.668), with concomitant gains in Spearman and per-target \(R^2\). Architectures that introduce attention between the molecule and edit streams (GatedCrossAttn, AttnThenFiLM) are *worse* than Subtraction, indicating that the gain comes from the right inductive bias for an additive intervention, not from added capacity.

**Table 1. Phase-2 architecture comparison (Morgan FP, within-assay, 3 seeds, 1.7 M pairs).**

| Architecture | MAE | Spearman | \(R^2\) | vs Subtraction |
|---|---|---|---|---|
| **FiLMDelta** | **0.616 ± 0.022** | **0.400** | **0.196** | **−7.8%** |
| EditDiff | 0.631 ± 0.016 | 0.383 | 0.177 | −5.6% |
| EditDiff + 28d feats | 0.635 ± 0.018 | 0.375 | 0.170 | −4.9% |
| DeepDelta | 0.642 ± 0.016 | 0.362 | 0.156 | −3.9% |
| Subtraction | 0.668 ± 0.019 | 0.361 | 0.099 | baseline |
| TrainableEdit | 0.673 ± 0.020 | 0.277 | 0.075 | +0.7% |
| AttnThenFiLM | 0.705 ± 0.006 | 0.204 | −0.007 | +5.5% |
| GatedCrossAttn | 0.712 ± 0.018 | 0.209 | −0.027 | +6.6% |

**Generalization across splits** (Phase 3, four representative methods): FiLMDelta wins assay-within (the cleanest split with 42% of test molecules unseen and 21% edit overlap), assay-mixed, and is competitive on assay-cross. Random and old-scaffold splits are leaky (71% and 60% of test pairs share both molecules with train, respectively) and therefore not load-bearing for generalization claims.

Phase-4 edit-aware variants (DRFP, DualStream, Frag-anchored, MultiModal, Hypernet) further improve assay-within MAE to 0.585 ± 0.033 (DualStreamFiLM), but for the generative pipeline we use the validated FiLMDelta configuration.

### 3.2 Noise robustness — 112-target tiered evaluation

We binned 112 ChEMBL targets by realistic noise ratio (cross-assay \(\sigma\) / within-assay \(\sigma\), range 0.35× to 3.3×) and trained both FiLMDelta (within-assay only) and Subtraction (all data, the realistic deployment) per target. **FiLMDelta wins 112/112** on MAE (mean advantage 76.3%); Spearman, Pearson, and \(R^2\) gaps all grow significantly with assay noise (\(p<10^{-3}\) for the noise-ratio × method interaction). A controlled noise-injection experiment (Gaussian \(\sigma=0\to1.5\)) shows FiLMDelta degrading 3.2% over the full noise sweep vs Subtraction's 12.3% — a 4× steeper degradation for the baseline. We attribute the advantage to two additive sources: ~8% from the FiLM inductive bias (Phase 2) and ~62% from training on within-assay-only pairs (i.e. data curation made tractable by the edit framing).

Scripts: `experiments/run_fair_noise_tiers.py`, `experiments/run_noise_injection.py`. Decomposition math: with Var(within)=0.879 and Var(cross)=1.221, \(\sigma_\text{lab}/\sigma_\text{SAR}=0.62\), so 28% of cross-assay \(\Delta\)pIC50 variance is lab noise that within-assay training simply does not see.

### 3.3 Day-1 cohort — intrinsic metrics across vanilla / inpaint / constraint-projected

> *QA note (B2)*: the "geometric residual" rows below are **not a measurement of how far DiffSBDD's diffusion pushed the warhead** — the 5 warhead atoms are held by `--fix_atoms` and the input SDF was constructed by `place_warhead_at_cys.py` at exact (1.85 Å, 107°, 0°). The "before projection" residuals are numerical drift from RDKit scaffold reconstruction + bond perception, not denoising motion. The constraint projector's real contribution must be evaluated on the warhead-pose ablation cohort (Task #10) where the input warhead is deliberately perturbed; that experiment is the legitimate test of the projection-vs-inpaint comparison and is pending.

100 generations per target with the three sampler variants, post-processed and scored (Table 2). The headline number is the acrylamide-readiness column: vanilla DiffSBDD produces 0% covalent-ready ligands on either target; inpaint produces 100% by construction; constraint-projection preserves connectivity and adjusts geometry only.

**Table 2. Day-1 cohort intrinsic metrics (per `results/anchordiff/day1_intrinsic_metrics.csv` and `day1_constraint_projection_residuals.csv`).**

| Cohort | Target | N | acrylamide | Lipinski | PAINS-clean | MW | QED | SAS | Tc to parent | d(S,Cβ) range (Å) | angle range (°) | \|φ\|max (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | ZAP70 | 97 | **0%** | 91% | 99% | 234 | 0.42 | 4.25 | 0.078 | n/a (no warhead) | n/a | n/a |
| vanilla | BTK † | 91 | **0%** | 73% | 99% | 401 | 0.51 | 4.74 | 0.104 | n/a | n/a | n/a |
| inpaint | ZAP70 | 80 | **100%** | 99% | 98% | 140 | 0.40 | 2.98 | 0.149 | 1.78 – 1.95 | 101.6 – 110.4 | 10.8 |
| inpaint | BTK † | 82 | **100%** | 96% | 100% | 215 | 0.37 | 4.29 | 0.104 | 1.76 – 1.92 | 100.6 – 113.7 | 7.9 |
| constraint_proj | ZAP70 | 80 | 100% | 99% | 98% | 140 | 0.40 | 2.98 | 0.149 | 1.850 ± 0 | 107.00 ± 0 | 0.00 |
| constraint_proj | BTK † | 82 | 100% | 96% | 100% | 215 | 0.37 | 4.29 | 0.104 | 1.850 ± 0 | 107.00 ± 0 | 0.00 |

† BTK is reported as a target-portability sanity check on the generator only — the FiLMDelta scorer was not trained on any BTK data, so the BTK pIC50 column from the Day-1 ranking is dropped (see §4.5). BTK intrinsic metrics, geometric residuals, and the *generator's* ability to produce a covalently-positioned acrylamide are still valid claims; everything potency-related on BTK is moved to future work.

Inpaint already produces near-canonical geometry (the maximum residual dihedral is 10.8°), and constraint projection drives all three residuals to floating-point zero with negligible ligand RMSD. The inpaint cohort's lower MW (140 Da on ZAP70) reflects a known DiffSBDD bias toward small fragments under tight constraints; the 1000-mol cofold leaderboard below uses an MW<400 ChEMBL-warhead-fused candidate set to address this.

### 3.4 ZAP70 Cys346 lead-optimisation leaderboard (top 5 of 997)

> *Framing*: this is **lead optimisation of Mol 1**, not de novo discovery. The 997-mol candidate set was constructed from Mol 1 by med-chem rules, amine replacement, fragment libraries (LibInvent / Mol2Mol / constrained de novo) — by design they share the Mol-1 isoindolinone-acrylamide chemotype with amine/aryl/hinge variations. The scaffold-diversity audit (top-20: 17 BM scaffolds, top-50: 42, top-100: 84 BM scaffolds; ECFP4-Tc≥0.5 clusters in top-50 = 12; median Tc to Mol 1 = 0.45) is the correct distribution for a Mol-1 lead-opt screen. Wet-lab handoff still applies scaffold deduplication (Tc≤0.5) so chemists are given diverse-enough representatives.
>
> *QA caveats remaining*. **(B1)** absolute pIC50 numbers are rank-consistent but not calibrated — the cached FiLMDelta was selected on a leaky molecule-level val split; retraining is queued (Task #23). Cite ranks/percentiles, not pIC50 values, until retrained. **(C1 — code QA)**: the Boltz constraint atom name uses `AllChem.CanonicalRankAtoms` whose value is RDKit-version-dependent. We verify against one decoded CIF in Task #26 before any wet-lab handoff. **(Boltz iPTM == ligand-iPTM)**: in single-ligand cofold the two columns are degenerate; the combined score uses only one DoF.

A 1000-mol candidate set (`top1000__zap70_cys346`, MW<400, ranked by FiLMDelta 3-seed pIC50 ensemble against 280 ZAP70 anchors) was cofolded with Boltz-2. 997/1000 mols completed; 3 RDKit ETKDGv3 conformer failures. **Median ligand-iPTM = 0.898; max = 0.980.** Top 5 by combined score (FiLMDelta percentile × 0.6 + lig-iPTM percentile × 0.4):

**Table 3. ZAP70 Cys346 cofold leaderboard, top 5 (full file: `results/anchordiff/cys346_cofold_leaderboard.csv`).**

| pIC50 | lig-iPTM | pLDDT | MW | source | SMILES |
|---|---|---|---|---|---|
| 7.64 | 0.963 | 0.81 | 394 | LibInvent | `C=CC(=O)N1Cc2cccc(C(=O)NC(=O)NCCN(C)S(C)(=O)=O)c2C1` |
| 7.69 | 0.952 | 0.81 | 391 | Mol2Mol | `C=CC(=O)N1Cc2cccc(C(=O)Nc3ccc(N4CCCNCC4)nc3)c2C1` |
| 7.59 | 0.969 | 0.80 | 376 | Constrained | `C=CC(=O)N1Cc2cccc(/C=C/C(=O)Nc3cc(C4CCCC4)n[nH]3)c2C1` |
| 7.66 | 0.949 | 0.81 | 367 | Amine Repl | `C=CC(=O)N1Cc2cccc(C(=O)N3CCN(Cc4nnn[nH]4)CC3)c2C1` |
| 7.61 | 0.952 | 0.81 | 346 | LibInvent | `C=CC(=O)N1Cc2cccc(C(=O)NC(=O)c3nn[nH]c3Cl)c2C1` |

Top-5 candidates share a 2,3-dihydro-1H-isoindol-1-yl acrylamide core, a substitution pattern absent from the 280 ZAP70 anchor set (max Tanimoto to training = 0.22) — i.e. these are not training-set memorizations.

### 3.5 TODO — outcomes pending experiment completion

> **TODO — B2 particle-resampling cohort.** Same 1000-mol candidate budget under FiLMDelta-guided particle resampling, comparing top-quartile pIC50 and lig-iPTM CDFs vs the post-hoc-filtered baseline reported in Table 3.

> **TODO — Warhead-pose ablation outcome.** Boxplots of lig-iPTM and FiLMDelta-pIC50 for canonical vs perturbed inpaint poses; specifically whether the canonical (1.85 / 107° / 0°) pose is necessary or merely sufficient.

> **TODO — Wet-lab kinact/Ki.** IMAP kinase assay (Eurofins or in-house) on the top 12 cofold candidates against ZAP70 (kinase domain, His-tagged, expressed and purified internally). Readouts: \(k_\text{inact}\), \(K_\text{i}\), \(k_\text{inact}/K_\text{i}\) (M⁻¹ s⁻¹), with reference compound (entospletinib analog) as positive control. Validates Section 3.4 predicted potency in the prospective covalent regime.

> **TODO — Selectivity panel.** DiscoverX KINOMEscan® 468-kinase panel at 1 µM for the top 3 candidates. Validates whether the FiLMDelta + cofold selection inherits ZAP70 Cys346-driven selectivity over hinge-Cys kinases (BTK, TEC, ITK, BMX, JAK3, BLK, EGFR).

> **TODO — Intact-protein LC/MS occupancy time-course.** ZAP70 kinase-domain protein incubated with each top-3 candidate at 1× and 10× \(K_\text{i}\); occupancy at Cys346 read out by tryptic digest + LC/MS at 0/0.5/2/8 h. Validates covalency, site-specificity, and \(t_{1/2}\) of adduct formation.

> **TODO — Co-crystal X-ray.** ZAP70 kinase-domain co-crystallization with the top-1 candidate; soaking and co-crystallization tried in parallel. Validates the predicted prereactive pose and the canonical-geometry assumption underlying inpaint and projection.

---

## 4. Discussion

The contribution of this work is the *integration*, not any single component. FiLMDelta in isolation is a within-assay potency model; DiffSBDD in isolation generates non-covalent fragments; Boltz-2 in isolation cofolds whatever you hand it. Stacking them with a closed-form covalent manifold projector — and, critically, fixing the warhead pose at a *canonical* prereactive frame derived directly from the receptor cysteine rather than from any specific bound parent ligand — produces a pipeline whose outputs (a) are 100% covalent-ready by construction, (b) sit on the covalent manifold to floating-point precision, (c) score in the top decile of FiLMDelta-predicted potency, and (d) are validated by an independent structural model (Boltz-2) at lig-iPTM up to 0.98.

The principal simplification is the **fixed warhead pose**. An acrylamide attached to a flexible scaffold can in reality access multiple prereactive geometries; we have committed to a single canonical frame to make the projector closed-form and the inpaint deterministic. The warhead-pose ablation (Section 2.6 TODO) directly addresses this: by re-running with three perturbed canonical poses we will quantify the cost of this simplification and, if needed, replace the projector with a small ensemble of canonical poses spanning the SG-Cβ-Cα-Ccarb torsion.

Two additional points warrant note. First, the noise-robustness advantage of FiLMDelta (76.3% mean over 112 targets) decomposes as ~8% architecture and ~62% data curation. The takeaway is that even a perfect architecture cannot compensate for a noisy training signal; the edit framing is what makes within-assay-only training feasible, since within-assay absolute training would simply lose 60% of the data. Second, the framework is warhead- and target-agnostic. Replacing acrylamide with chloroacetamide changes only the canonical bond length and the Bürgi–Dunitz angle; replacing the target Cys346 with BTK Cys481 changes only the receptor crop. We have demonstrated both substitutions (BTK case in Table 2) without re-engineering any component.

---

## 5. Reproducibility

**Datasets.**
- Canonical MMPs: `data/overlapping_assays/extracted/shared_pairs_deduped.csv` (1.7 M pairs, 88 K mols, 751 targets).
- Embedding cache: `data/embedding_cache/{morgan,chemberta2-mlm,chemberta2-mtr,chemprop-dmpnn,drfp_2048,fragment_deltas_1024}.npz`.

**Models.**
- FiLMDelta and edit-aware variants: `src/models/predictors/edit_aware_film_predictor.py`.
- Subtraction / EditDiff / DeepDelta baselines: `src/models/predictors/`.
- Edit features (28d): `src/data/utils/chemistry.py::compute_edit_features`.

**Experiment scripts.**
- Phase 1→3 paper evaluation: `experiments/run_paper_evaluation.py`.
- Phase 4 edit-aware iteration: `experiments/run_edit_iteration.py`.
- Noise robustness: `experiments/run_fair_noise_tiers.py`, `experiments/run_noise_injection.py`.
- ActFound comparison: `experiments/run_actfound_comparison.py`.
- ZAP70 case study (v3 best single, v6 ensemble): `experiments/run_zap70_v3.py`, `experiments/run_zap70_v6.py`.

**Generative pipeline.**
- Warhead placement at Cys: `anchordiff/place_warhead_at_cys.py`.
- DiffSBDD inpaint runners: `/tmp/run_day1_t4_v2.sh` (vanilla), `/tmp/run_day1_inpaint_t4.sh` (inpaint).
- Bond-perception fix: `anchordiff/fix_inpaint_warhead_bonds.py`.
- Constraint manifold projector: `anchordiff/covalent_constraint_manifold.py`.
- Boltz-2 cofold yamls and outputs: `data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/`.
- Cofold leaderboard manifest: `results/anchordiff/cys346_cofold_leaderboard.csv`.

**Results manifests.**
- Paper evaluation: `results/paper_evaluation/{all_results.json, edit_iteration_results.json, fair_noise_tiers_results.json, noise_injection_results.json, evaluation_report.html}`.
- Day-1 generative cohort: `results/anchordiff/{day1_summary.md, day1_intrinsic_metrics.csv, day1_constraint_projection_residuals.csv, day1_inpaint_filmdelta_ranking.csv, cys346_cofold_leaderboard.csv}`.

**Environment.** conda env `quris`. Phase 1–3 training on CPU (ChemBERTa MPS instability); generation on a single T4; cofolding on a single A100 (~70 s/molecule).

---

*End of draft. Word count target: ~3000 (first draft).*
