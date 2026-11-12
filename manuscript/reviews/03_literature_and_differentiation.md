# Review 03 — Literature and differentiation

Working memo. Reviewer: literature + differentiation claims. 2026-11-12.
**VERIFIED** = I opened the arXiv/DOI/journal record and the title, ID and (where stated) authors matched.
**PARTIAL** = the work exists and the ID resolves, but a field (authors, venue, journal version) rests on a
secondary source. **UNVERIFIED** = could not confirm; do not put in the bibliography as written.

---

## 1. Reference list

### A. Pocket-conditioned / structure-based generative models
- **Pocket2Mol** — Peng, Luo, Guan et al., *Efficient Molecular Sampling Based on 3D Protein Pockets*, arXiv:2205.07249. **VERIFIED** (arXiv). ICML 2022 venue **PARTIAL**.
- **TargetDiff** — Guan, Qian, Peng, Su, Peng, Ma, *3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction*, ICLR 2023, arXiv:2303.03543. **VERIFIED** (arXiv comment states ICLR 2023).
- **DiffSBDD** — Schneuing et al., arXiv:2210.13695. **VERIFIED** (arXiv). Journal version *Nature Computational Science* (DOI 10.1038/s43588-024-00737-x) **PARTIAL**.
- **DrugFlow** — Schneuing et al., *Multi-domain Distribution Learning for De Novo Drug Design*, ICLR 2025, arXiv:2508.17815. **VERIFIED** (ICLR 2025 proceedings page + arXiv). Note: the name "DrugFlow" is the model, the paper title is the distribution-learning one.
- **ResGen** — *Nature Machine Intelligence* 2023, DOI 10.1038/s42256-023-00712-7. DOI/title **VERIFIED**; author list **UNVERIFIED**.
- **Lingo3DMol** — *Generation of 3D molecules in pockets via a language model*, *Nat. Mach. Intell.* **6**, 62–73 (2024), arXiv:2305.10133. Venue/pages **VERIFIED**; author list **UNVERIFIED**.
- **PocketFlow** — *Nature Machine Intelligence* 2024, DOI 10.1038/s42256-024-00808-8. DOI **VERIFIED**; authors **UNVERIFIED**.
- **2026 successor, directly relevant to us** — MacDougall, Kuznetsov, Schutski, Shayakhmetov, Malkov, Aladinskiy, Aliper, Zhavoronkov, *Do Language Models Dream of Binding Molecules? Benchmarking LLMs under Spatial Constraints*, arXiv:2607.18144 (20 Jul 2026). **VERIFIED** (metadata). LLMs generating pocket-conditioned 3D ligands under anchor/pharmacophore/**required residue-contact** constraints.
- CAGenMol arXiv:2604.11483; LaMGen (*Nat. Commun.* 2026); NEAT-POCKET arXiv:2609.05097 — **UNVERIFIED**, cite only if someone opens them.

### B. Chemistry / molecular LLMs and reasoning models
- **ChemDFM** — arXiv:2401.14818 (LLaMA-13B; domain pretraining + instruction tuning). arXiv ID/title **VERIFIED**; author list **UNVERIFIED**. Latest version is retitled *Developing ChemDFM as a Large Language Foundation Model for Chemistry* — cite that title.
- **ether0** — Narayanan et al. (FutureHouse), *Training a Scientific Reasoning Model for Chemistry*, arXiv:2506.17238. 24B, RL on 640,730 problems / 375 tasks. **VERIFIED**.
- **SmileyLlama** — Cavanagh et al., *Modifying Large Language Models for Directed Chemical Space Exploration*, arXiv:2409.02231 (SFT + DPO on ChEMBL). **VERIFIED**.
- **Mol-Instructions** — arXiv:2306.08018. **VERIFIED** (ID/title); ICLR 2024 venue **UNVERIFIED**.
- **ChemCrow** — Bran et al., arXiv:2304.05376 (18 expert tools). arXiv **VERIFIED**; *Nat. Mach. Intell.* version **UNVERIFIED**.
- **ChemLLM** — Zhang et al., arXiv:2402.06852 (+ChemData, ChemBench). **VERIFIED**. **Do not write "ICLR"** — not confirmed.
- **nach0** — *Chemical Science* 2024, DOI 10.1039/D4SC00966E. DOI **VERIFIED**; authors **UNVERIFIED**.
- **LlaSMol** — arXiv:2402.09391. **VERIFIED** (ID).
- **DESRO** — Liu, Wu, Xie, Guo, Zhang, Qin, Liu, Xia, *Deciphering Scientific Reasoning Steps from Outcome Data for Molecule Optimization*, arXiv:2603.20262 (13 Mar 2026). **VERIFIED**. *This is the base model of our DESRO-8B row and the closest prior art to our factorisation — it must be cited in the intro, not the related-work tail.*
- **DrugR** — Liu, Zeng, Yan, Chen, Xiao, *Optimizing Molecular Drugs through LLM-based Explicit Reasoning*, arXiv:2602.08213 (9 Feb 2026). **VERIFIED** (metadata). Rationales built by "reverse data engineering" from verified ADMET outcomes; rationale and SMILES generated **jointly**.
- **Logos** — Wen, Zhao, Wang, Xu, Zhang, Yang, Wei, *An evolvable reasoning engine for rational molecular design*, arXiv:2603.09268 (10 Mar 2026). **VERIFIED** (metadata). Reason-block-then-SMILES in one model; no pocket input mentioned.
- **MolecularCanvas** — Dong, Sheng, Zhang, Sun, Wu, Chao, Qu, Wu, Lin, arXiv:2608.00393 (1 Aug 2026). **VERIFIED** (metadata). LLM emits an explicit edit plan, RDKit executes it; "structure-guided" here means *molecular* structure, not pocket.
- **RLVR for molecules** — LLMol arXiv:2607.19044 **PARTIAL**; RePO (ICLR 2026) **PARTIAL**; Round-trip RL arXiv:2510.01527 **PARTIAL**; MolReasoner arXiv:2508.02066 **PARTIAL**.
- **Text-guided editing** — MolT5 arXiv:2204.11817 **PARTIAL**; ChatDrug arXiv:2305.18090 (ICLR 2024) **PARTIAL**; UTGDiff arXiv:2408.09896 **VERIFIED** (ID) — see §3(d).

### C. Lead optimisation, matched pairs, delta learning
- **DeepDelta** — Fralish, Chen, Skaluba, Reker, *J. Cheminform.* 15 (2023), DOI 10.1186/s13321-023-00769-x. **VERIFIED**.
- **ActFound** — Feng et al., *A foundation model for bioactivity prediction using pairwise meta-learning*, *Nat. Mach. Intell.* 2024; preprint DOI 10.1101/2023.10.30.564861. Preprint **VERIFIED**; journal DOI **UNVERIFIED**.
- **REINVENT 4** — Loeffler et al., *J. Cheminform.* **16**:20 (2024), DOI 10.1186/s13321-024-00812-5. **VERIFIED**. Contains the functionality comparison against **DrugEx v3** (cite DrugEx only once someone opens its own paper — currently **UNVERIFIED**).
- **Matched molecular series** — *What compound should I make next?*, *J. Cheminform.* 2014, PMC3980175. **VERIFIED**.
- MMP-as-translation / similar-pair transformers: *J. Cheminform.* 16:95 (2024), DOI 10.1186/s13321-024-00887-0 — DOI **VERIFIED**, title **UNVERIFIED**.
- **Landrum & Riniker**, JCIM 2024, bioactivity reproducibility — **UNVERIFIED by me**; it is in our project memory and is the right citation for the assay-noise argument, but someone must confirm volume/pages.

### D. LLMs extracting structured data from the chemistry literature
- **ChemMiner** — arXiv:2402.12993, multi-agent reaction mining. **PARTIAL**.
- CIKM '24, *Application of Large Language Models in Chemistry Reaction Data Extraction and Cleaning*, ACM DOI 10.1145/3627673.3679874. **PARTIAL**.
- **ChemActor** — ACL 2025 Long (aclanthology 2025.acl-long.1183). **PARTIAL**.
- Materials-science IE at scale — *Commun. Mater.* 2025, DOI 10.1038/s43246-025-01043-3. **VERIFIED**.
- **Finding to state in the paper:** two extended searches returned **no prior work extracting the stated design rationale of a medicinal-chemistry modification from paper text**. Everything in this group extracts *reactions*, *procedures* or *entities*. This is the cleanest novelty we have; it is also the one most likely to be contested by a reviewer who knows an industry-internal system, so phrase it as "to our knowledge".

### E. Structure prediction / cofolding as design input
- **AlphaFold 3** — Abramson et al., *Nature* 2024. **PARTIAL** (cited only secondhand here; confirm DOI 10.1038/s41586-024-07487-9).
- **Boltz-1** — Wohlwend et al., bioRxiv DOI 10.1101/2024.11.19.624167. **VERIFIED** (DOI).
- **Boltz-2** — *Towards Accurate and Efficient Binding Affinity Prediction*, bioRxiv DOI 10.1101/2025.06.14.659707. DOI/title **VERIFIED**; **author order conflicts across sources** (Passaro/Corso/Wohlwend vs Roy first) — open the bioRxiv page before typesetting.
- **Chai-1** — Chai Discovery, *Decoding the molecular interactions of life*, bioRxiv DOI 10.1101/2024.10.10.615955. **VERIFIED** (DOI).
- **Cofolding caveat citation we need** — eLife reviewed preprint 110475, prospective cofolding across hundreds of Mac1–ligand complexes plus three virtual screens. **PARTIAL**. Cite it next to our cofold-derived contact tables; it is the honest hedge and pre-empts the obvious referee question.

### F. Benchmarks and their criticisms
- **PoseBusters** — Buttenschoen, Morris, Deane, *Chem. Sci.* 2024, arXiv:2308.05777. **VERIFIED**.
- **PoseCheck** — Harris et al., *Benchmarking Generated Poses: How Rational is Structure-based Drug Design with Generative Models?*, arXiv:2308.07413 (MLSB 2023). **VERIFIED**.
- **Docking-score critique** — Ciepliński et al., *Generative Models Should at Least Be Able to Design Molecules That Dock Well*, JCIM 63(11):3238 (2023), arXiv:2006.16955. **VERIFIED**.
- **Delta Score / specificity** — arXiv:2403.12987. **VERIFIED** (ID).
- **GenBench3D** — arXiv:2407.04424. **VERIFIED** (ID).
- *Nat. Commun.* 2026, *Evaluating generative models for structure-based drug discovery from de novo design to lead optimization*, s41467-026-78105-8 — **PARTIAL**; its point that QED/Vina correlate with atom count is exactly the argument for not using docking as supervision.

---

## 2. What is genuinely new — verdict on each candidate claim

**(a) "Pocket generators don't say why; reasoning LLMs don't see the pocket; we do both." — PARTIALLY NEW; the second half is now false. Reject as written.**
Lingo3DMol already trains a separate non-covalent-interaction/anchor predictor, so pocket generators are interaction-aware even when mute. More damaging: arXiv:2607.18144 (Jul 2026) puts pockets *and required named-residue contacts* into LLM prompts. We cannot say reasoning LLMs do not see the pocket. **Defensible version:** no prior system emits a natural-language design intent whose citations are *checkable against the ligand's own per-atom contact table*, and none reports groundedness/invented-contact rates as metrics. The novelty is the **auditable grounding of the prose**, not pocket-awareness.

**(b) "Supervision is chemists' stated intent mined from papers, not docking scores." — NEW. Keep, with one honest hedge.**
DESRO derives reasoning steps *from outcome data* by LLM analysis of fragment groups; DrugR derives rationales *backwards from computed ADMET outcomes*. Both manufacture the rationale from the result. We quote the author's sentence. Nothing in group D does this. The hedge: our own 100-paper QA says **60.7% of extracted reasons restate an outcome rather than an instruction**, which puts that majority in the same category as DESRO/DrugR. Say so in the same paragraph as the claim, and the claim survives; bury it and a referee finds it.

**(c) "Factorising into R (intent) and M (molecule | intent) makes the rationale evaluable and human-substitutable." — PARTIALLY NEW. Sharpen.**
Prior art: Jin et al., *Multi-Objective Molecule Generation using Interpretable Substructures* (ICML 2020, arXiv:2002.03244) factorises rationale→molecule, but the rationale is a substructure, not a sentence. DESRO, DrugR and Logos all do reason-then-molecule *inside one model and one trace*, so the reason cannot be swapped. MolecularCanvas separates an LLM plan from an RDKit executor, but it is an interactive tool with no held-out oracle/mismatch measurement. **What is ours:** two separately trained models with a **natural-language sentence as the interface**, so the intent is (i) replaceable at inference by a human or by the chemists' own sentence, and (ii) quantified — oracle 0.919 vs R→M 0.574 vs mismatch 0.541. The measured oracle–pipeline gap, not the factorisation itself, is the contribution.

**(d) "We evaluate instruction-following with a mismatched-intent control." — PARTIALLY NEW. Keep, credit the antecedent.**
Pocket-shuffled controls are standard in SBDD. UTGDiff (arXiv:2408.09896) already argues that distributional metrics cannot detect cross-matched molecule/instruction pairs — cite it as the antecedent. NLP has instruction-faithfulness work (*Dancing in Chains*, EMNLP 2024). **What is ours and I found nowhere else:** the three-arm decomposition in which **mismatch (0.541) falls 0.147 *below* no-instruction (0.684), p = 6.9×10⁻⁴⁰, with validity dropping 100% → 91.5%**, together with the finding that the no-instruction arm is not a baseline at all because it echoes the input (TC→input = 1.000, every property delta exactly zero). That is a genuinely new negative control design and it is the most quotable methodological result in the paper.

---

## 3. Closest prior art to (c) and (d) — must be cited and positioned

Ranked: **DESRO (arXiv:2603.20262)** → **DrugR (arXiv:2602.08213)** → **Logos (arXiv:2603.09268)** → **RationaleRL (arXiv:2002.03244)** → **MolecularCanvas (arXiv:2608.00393)** for (c); **UTGDiff (arXiv:2408.09896)** for (d). None of the six uses pocket structure as conditioning, and none separates the reason into an independently trained, independently substitutable model. Position with one sentence each in related work and one sentence in the intro — do not pretend the reason-then-molecule idea originates here; claim the **separation, the structural grounding, and the control arms**.

---

## 4. Three sentences we must not write

1. **MUST NOT:** "Our model is pocket-aware: conditioning on the binding pocket yields better analogues."
 **CAN:** "The contact table determines what the explanation may cite — groundedness 1.000 vs 0.432, selectivity 0.388 vs 0.121 — and leaves the molecule essentially unchanged (TC→reference 0.679 vs 0.680); at cohort scale a *wrong* pocket costs only 0.013 Tanimoto. The pocket governs the prose, not the molecule."

2. **MUST NOT:** "Factorising design into intent and molecule improves hit-to-lead optimisation."
 **CAN:** "A correct intent is worth +0.379 TC→reference over a mismatched one (paired, p = 9.6×10⁻⁵⁰), establishing that M follows the instruction; our own intent model closes only 0.574 of the 0.919 oracle, so we report the factorisation as a demonstrated *upper bound* with R as the measured bottleneck (18.6×), not as an end-to-end gain. The no-instruction arm is excluded as a baseline because it echoes the input (TC→input = 1.000)."

3. **MUST NOT:** "Blinded expert assessment shows our rationales are chemically sound (92.4% sensible, 46.5% sound reasoning)."
 **CAN:** "Under a single model-labeller panel with fixed rubric and seed, finetuning raises *sound reasoning* from 3.9% to 46.5% and *sensible* from 10.6% to 92.4%; panel-to-panel drift exceeds our between-arm differences, so no comparison is made across panels, an independent cohort round scored explanation soundness at only 0.08, and a human chemist panel is pending [TBD-1]."

---

## 5. Title and contribution statement

**Recommended title:** *Design Intent as an Evaluable Output: Pocket-Grounded Hit-to-Lead Optimisation with Language Models*
(Alternative, closer to the PI's: *Learning Medicinal-Chemistry Design Intent: Pocket-Grounded Hit-to-Lead Optimisation with Language Models* — swap "Pocket-Aware" for "Pocket-Grounded" either way; "aware" promises molecule-level effects our ablation does not support.)

**Contribution statement:** We mine chemists' stated design intent from 1,622 SAR papers, factorise hit-to-lead design into an intent model and an intent-conditioned molecule model whose shared structural input is a per-atom pocket contact table, and show with oracle, mismatched and no-intent controls that a correct intent determines the analogue (+0.379 TC, p = 9.6×10⁻⁵⁰) while the contact table determines what the stated reason may cite rather than which molecule is made.
