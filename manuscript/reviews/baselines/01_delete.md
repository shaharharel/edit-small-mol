# Baseline assessment 01 — Delete / "Causal-Delete" (pocket-enveloped 3D lead optimisation)

**Paper.** Chen S-C, Zhang O (Haotian), Jiang C, Zhao H-F, Zhang X-J, Chen M-T, Liu Y, Su Q, Wu Z-X, Wang X-Y, … Kang Y, Hsieh C-Y, Pan P, Xie C, Huang Y, Hou T.
*Deep lead optimization enveloped in protein pocket and its application in designing potent and selective ligands targeting LTK protein.*
**Nat Mach Intell 7, 448–458 (2025)**, DOI 10.1038/s42256-025-00997-w. Accepted 16 Jan 2025, published 20 Feb 2025. **Closed access.**

**Full-text access.** The journal main text is **paywalled** ("This is a preview of subscription content") — abstract, figure titles, references, data/code statements only. I worked instead from:
- the **arXiv preprint in full** (arXiv:2308.02172, 35 pp., `Delete: Deep Lead Optimization Enveloped in Protein Pocket through Unified Deleting Strategies and a Structure-aware Network`) — this is the method paper and carries main Tables 1–3;
- the **journal Supplementary Information PDF in full** (41 pp., freely served from static-content.springer.com) — carries SI Tables 1–3, the extended retrospective studies, the full LTK campaign narrative, and the Supplementary Methods;
- the GitHub repo, the `delete-protocol` repo, and the four Zenodo records.
Numbers that live **only** in the paywalled main text are flagged `[UNVERIFIED]` throughout. Supplementary Data (Source Data XLSX, 0.9 MB) was downloaded but not parsed; the peer-review file (ESM2) is image-only.

**Verdict in one line: a real baseline for the molecule, not for the task. Same conditioning side of the line (3D complex) and genuinely available under MIT, but it is unsupervised Vina-shaped 3D completion of a region *you* delete, it emits no rationale, and it cannot be given our input without us choosing the edit site for it — which hands it the hardest half of our problem for free. Run it as a handicapped single-subtask comparator on a crystal-only MMP subset; do not promise reviewers a head-to-head on our full val set.**

---

## 1. What the paper actually does

- **Task.** Structure-based **lead optimisation** as *region completion*: four subtasks under one model — linker design, fragment elaboration/growing, scaffold hopping (Bemis–Murcko), side-chain decoration. Explicitly framed as me-too/me-better and hit-to-lead, not de novo.
- **Input.** (i) the **protein pocket as a triangulated molecular surface** (`.ply`, built by MaSIF-style surface generation + APBS electrostatics from a `.pdb`), and (ii) a **3D lead fragment** (`.sdf`, with coordinates) that is the *input molecule with some atoms already deleted*. The deletion mask is supplied by the user. No sequence, no potency label, no text.
- **Output.** A completed molecule generated **atom by atom**: coordinate (relative to the attachment point), element, and bonds predicted directly by the network (not by OpenBabel rules). So the output is a 2D graph **plus a bound 3D pose**. **No text, no rationale, no residue claims of any kind.**
- **Architecture.** GVP scalar/vector layers + geometric message passing on a joint pocket-surface/ligand graph; attachment-point enumeration head → atom placement head → bond head with a triangular-attention term. SE(3)-equivariant. Checkpoint is **44.9 MB** — a small model. SI calls the section "Architecture of **Casual**-Delete model" (sic); the main-text Fig. 4 title says "Causal-Delete".
- **Training data.** **No potency labels anywhere.** Self-supervised masking on two pocket–ligand corpora: **CrossDock = 100,000 training pocket–ligand pairs** (cross-docked poses) and **BindingMOAD = 35,516 training pairs** (crystal only). Three variants: Delete-C (CrossDock), Delete-M (MOAD), Delete-CM (CrossDock-pretrained → MOAD-finetuned). Seven maskers: random / spatial / topological (augmentation) + linker / fragment / scaffold / side-chain (task-specific). **A separate parameter set is trained per subtask** ("to train a target-specific model, we combine three enhanced maskers with each task-specific masker").
- **Headline metrics** (main Table 1, arXiv version; **means over the Top-5 molecules per target** — see §5 fairness flag). Docking Energy / Scoring Energy = AutoDock-Vina with / without conformational search; Hit Pocket % = fraction of targets for which a generated molecule beats the original ligand on affinity.

| Subtask | Test-set ref | Delete-C | Delete-CM | Hit Pocket (Delete-C / -CM) | Competitor |
|---|---|---|---|---|---|
| Linker design | −8.888 | **−10.235** | −9.454 | 96.44% / 92.50% | DiffLinker −7.992, **Hit Pocket 51.28%** |
| Scaffold hopping | −7.896 | **−9.357** | −8.871 | 92.72% / 91.95% | none of this kind |
| Side-chain decoration | −7.605 | **−8.845** | −9.118 | 91.17% / 86.77% | none of this kind |
| Fragment elaboration | −8.781 | **−9.815** | −9.889 | 94.28% / 91.61% | none of this kind |

  QED 0.31–0.50, SA 0.58–0.74, Lipinski 2.6–4.5 — the authors argue only that these must stay "within reasonable intervals", not improve.
- **Journal version adds baselines** DeLinker, DeepFrag, FFLOM, Fragmenstein, ScaffoldGVAE, SCMG. **SI Table 1 gives only their standard deviations; the means are in the paywalled main table.** `[UNVERIFIED: the actual margin over these six baselines.]` The claim "Delete Outperformed Baselines across Four Tasks" is the SI section title, so the direction is stated by the authors but I could not read the numbers.
- **Conformation quality** (Table 3): docked-vs-generated RMSD 1.14–1.79 Å, statistically indistinguishable from crystal-vs-redocked RMSD for the MOAD reference ligands (1.39–1.62 Å). This is their strongest technical result.
- **Retrospective case studies (n = 5, Table 2 / SI Table 3).** Mean 2D Tanimoto to the real optimised compound vs mean 3D shape similarity: Adrb1 **0.347 / 0.813**, IMPDH **0.501 / 0.928**, SMARCA2 PROTAC **0.844 / 0.893**, DRD2 **0.458 / 0.742**, COX2 **0.371 / 0.889**. Recoveries: the exact structure of cyanopindolol *and* carazolol from 117 Adrb1 generations; the Trapero IMPDH linked compound within 76 generations. **Low 2D similarity is reported as a virtue (novelty/scaffold hop), not as a failure** — the opposite sign convention to our TC→reference.
- **Prospective LTK campaign (the real headline).** No LTK crystal, so a **homology model from ALK**; general Delete **finetuned on ALK inhibitors** (transfer learning) with an in-house QSAR filter; starting fragment = their own prior ALK compound **001-017** with its hydrophobic-back-pocket portion deleted. **227 candidates in ~30 min** → ranked on Vina + ADMETLab bioavailability/half-life + chemist-judged synthetic complexity, with a deliberate *distribution* (not top-N) selection → **13 selected → 8 synthesised → 6/8 below 1 µM**. **CA-B-1: cellular IC50 6.60 nM** in Ba/F3-CLIP1-LTK (SI Fig. 7); abstract reports **1.36 nM** potency and selectivity in vitro/in vivo. `[UNVERIFIED: the assay behind 1.36 nM — the SI only gives the 6.60 nM cell number; the two are different assays and the main text is paywalled.]`

## 2. Is it a real comparison for us?

| Axis | Delete | Ours |
|---|---|---|
| Task | analogue optimisation in-pocket — **same side of the line**, not de novo | analogue optimisation in-pocket |
| Conditioning | **3D complex**: pocket surface mesh + bound fragment pose | **3D complex**, abstracted to a 16-field per-atom contact table |
| Edit site | **supplied by the user** as a deletion mask | **decided by the model** |
| Supervision | none on potency; Vina is the only objective | measured within-assay potency deltas, pairs oriented weak→strong |
| Rationale | **none** | NL design intent naming residues, emitted before the SMILES, teacher-forceable |
| Output geometry | 3D pose + graph | SMILES only |

- **Same task? Yes at the level a reviewer cares about** — one hit molecule in a pocket → analogues of it. This is the closest published baseline to us on task shape, and much closer than the ligand-only CLM work in `02_incremental_clm.md`.
- **Same conditioning? Yes in substance, no in encoding.** A surface mesh + APBS electrostatics and our per-atom contact table are two lossy views of the same 3D complex. That makes it a legitimate test of *whether our contact-table encoding earns its place*, which is exactly the ablation we already run internally (0.679 vs 0.680 TC, groundedness 1.000 → 0.432).
- **Rationale? None.** Residue specificity (0.417 vs 0.200) and residue groundedness have **no Delete counterpart** — undefined, not zero. Our pocket-ablation argument has an analogue here (you can feed it the wrong surface), and that is worth running: it is the one place their model and ours can be stressed identically.
- **Could it run on our inputs? Only after we change the task.** Our input is an *unmasked* hit plus a contact table. Delete requires a deletion mask, and that choice dominates what comes out — our own L2 anchor-depth work says the same thing (`l2_scaffold_anchor_breakthrough`, `anchor_depth_tradeoff_curve`: clamping more of the molecule drove CHR 0% → 44%, and H2's 97% was effectively R-group enumeration). **Giving Delete the edit site is giving it the answer to half our problem.** It is therefore a baseline for "molecule quality given a hand-chosen site", not for "decide what to change and why".
- **Second structural mismatch:** Delete assumes **rigid retention** — kept atoms do not move. The paper states this as its central limitation. Our complexes are crystal where available and otherwise **single-sample Boltz cofolds**; under rigid retention, cofold pose error propagates straight into the generated geometry with no recovery mechanism. Our contact table is a coarser abstraction and is likely more pose-error tolerant, but that is a hypothesis, not a result we have.

## 3. Availability — good, with a real operational tax

- **Code:** https://github.com/HaotianZhangAI4Science/Delete (= `OdinZhang/Delete`), **MIT licence**, ~80 stars, 5 commits. Protocol repo: https://github.com/HaotianZhangAI4Science/delete-protocol.
- **Pretrained checkpoints: yes.** `delete.pt`, **44.9 MB**, Zenodo 10.5281/zenodo.7985201 (CC-BY-4.0). More checkpoints via delete-protocol (Zenodo 10.5281/zenodo.8104141). **Note:** training is per-subtask, so covering fragment/linker/scaffold/side-chain needs four parameter sets; `[UNVERIFIED: which masker the single `delete.pt` in the main repo corresponds to.]`
- **Code zip / "data":** Zenodo 10.5281/zenodo.14586176, CC-BY-4.0 — titled **"Delete Code"**, 6.4 MB. The paper's *Data availability* statement points at this record, i.e. **at a code zip, not at datasets.** Actual training LMDBs are elsewhere: CrossDock 10.5281/zenodo.7980002, BindingMOAD 10.5281/zenodo.7980026. Prebuilt conda env tarball: 10.5281/zenodo.7980554 (3.1 GB).
- **Inference requirements:** one GPU. CUDA 11.3 via the shipped YAML, or a manual CUDA 12.1 / PyTorch 2.2.2 / PyG path for 40-series cards (requires hand-editing `./utils/transform.py`). Author's test box: RTX 4080, Linux. Model is tiny; VRAM is a non-issue.
- **The actual blocker is input preparation, not the net.** Pocket surfaces need a *second* environment (`pymesh2`, biopython, plyfile) plus **APBS-3.0.0 and pdb2pqr-2.1.1**. The README itself documents the failure modes: `libTABIPBlib.so` not found (needs `LD_LIBRARY_PATH`), and `libreadline.so.7` on Ubuntu 22 with **"no simple fix"** — the author's workaround is hand-wired APBS paths in `./utils/masif/generate_prot_ply.py`. English tutorial "coming soon"; only a Chinese video exists.

## 4. How we differ — the two sentences for a reviewer

Delete is an **unsupervised 3D completion model**: a user deletes a region of a posed ligand and the network refills it with atoms that fit the pocket surface, optimising a docking score, emitting no statement of what it was trying to achieve. Our system receives the hit **unmasked**, decides the edit itself, **states a residue-level design intent in language before committing to a structure**, and is trained on **measured within-assay potency deltas oriented weak→strong** — so we are supervised on the direction of potency and auditable on the reason, where Delete is neither.

## 5. Who wins on what

**Where we expect to lose:**
- **3D pose and shape fidelity.** They emit a bound conformation at docked-RMSD 1.14–1.79 Å and 3D shape similarity 0.742–0.928. We emit SMILES only; these metrics are **N/A for us**, and a reviewer will read N/A as a gap, not a neutral.
- **Docking-score improvement.** Hit Pocket 86.8–96.4% on Vina across four subtasks, against DiffLinker's 51.3%. We do not optimise Vina and have no comparable number outside the ZAP70 case study. If a reviewer asks "do your analogues dock better?", we currently cannot answer at val scale.
- **Prospective validation.** They have a synthesised, selective, nanomolar compound with in vivo efficacy (8 synthesised, 6/8 < 1 µM). We have zero wet-lab compounds. This is the single most damaging asymmetry and no metric of ours offsets it.
- **Subtask coverage.** One framework covering linker/fragment/scaffold/side-chain with published per-subtask numbers; our edit types are not decomposed that way in the manuscript.

**Where we expect to win:**
- **Rationale, outright.** Residue specificity 0.417 vs a matched-count distance-sort baseline of 0.200, and groundedness collapsing 1.000 → 0.432 when the contact table is withheld. Delete has **no analogue at all** — not a worse number, an absent axis. This is the whole differentiation and it is uncontested.
- **Potency direction.** They have no potency supervision; Vina is the proxy and its rank correlation with measured potency is weak (our own docking work: `docking_integration_results`). Our PocketPair-21k pairs carry measured deltas. A reviewer will accept "their objective is a scoring function, ours is data".
- **Edit-site decision.** They require the mask; we don't. State this as a capability difference, and state plainly that when we benchmark them we *give* them the mask.
- **Sampling budget.** Their affinity table is **Top-5-per-target** and the case studies are best-of-76/106/110/117/131 generations. We report **greedy, single sample**. Any head-to-head must equalise N, and at equal N their numbers will move.
- **SMILES validity.** Ours 0.985. They report **no validity / uniqueness / novelty / diversity anywhere I could reach** — a real reporting gap for a generative model that predicts its own bonds. `[UNVERIFIED: whether the paywalled main text reports validity.]`

**Where the apparent win is NOT real, and we must not claim it:**
- Our **TC→reference 0.679** vs their case-study 2D similarity **0.347–0.501** looks like a win and is not one. Different inputs, different targets, 5 cases vs a val set, and **opposite sign conventions**: they *want* low 2D similarity (novelty) at high 3D similarity, so 0.347 is reported as success. Their 0.844 (SMARCA2) is high only because most of the PROTAC was retained. And our own 0.679 sits **below** our "any other real analogue" null of 0.700 — our TC advantage is not established either. The only legitimate version of this comparison is re-run head-to-head on the same inputs against the same reference.

## 6. Implementation cost, and the comparison I actually recommend

**It cannot be run on our 21k val inputs as they stand** — our rows carry no deletion mask, and inventing one changes the task. Scoping honestly:

- **Environment + surface pipeline: 1–2 days.** The APBS-3.0.0 / pdb2pqr-2.1.1 / pymesh2 stack is the documented pain point (two named breakages with one declared "no simple fix"). The prebuilt 3.1 GB conda env helps for the net but not for surfaces. Then ~0.5 day wiring our complexes → `.ply` + masked `.sdf`.
- **Throughput:** surface generation (APBS electrostatics per pocket) is the bottleneck, not inference. Scope to the **3,992 PocketPair complexes**, not 21k rows — most val rows share complexes. Budget low-thousands of CPU-hours for surfaces, parallelisable.
- **Generation:** 227 molecules in ~30 min on one consumer GPU ≈ 8 s/molecule. **~9 GPU-hours** for 1 sample per complex over ~4k complexes; **~180 GPU-hours** at 20 samples each. One GPU, any card; a 45 MB model.
- **Blockers, in order:** (a) no deletion mask in our data; (b) rigid-retention assumption vs single-sample Boltz cofold poses; (c) per-subtask checkpoints — must confirm which masker ships in `delete.pt`; (d) **half our metric panel is undefined** for a model with no text output.

**Nearest fair comparison — recommended protocol:**
1. **Subset:** val rows where (i) a **crystal** complex exists (drop Boltz cofolds, so their rigid assumption is not the thing under test) and (ii) hit → reference analogue differs by a **single contiguous acyclic fragment**, so an unambiguous mask exists.
2. **Hand Delete the mask** computed from the hit→reference MMP, i.e. delete exactly the atoms that change. Run the fragment-elaboration checkpoint. **State in the paper that Delete is given the edit site and our model is not** — that is a handicap in their favour and it strengthens, not weakens, whatever we report.
3. **Compare only TC→reference** at **matched sample budget** (their greedy/top-1 against our greedy), plus Vina as a secondary where we have poses. Report their 3D-pose capability as a capability we lack rather than pretending the axis does not exist.
4. **Report separately, not as a metric:** Delete produces no intent, so residue specificity/groundedness are N/A. Say "undefined", never "0".
5. **Two cheap extras worth the GPU time.** (i) **Wrong-surface ablation** — feed Delete a mismatched pocket and measure the TC cost, directly comparable to our 0.013 Tanimoto result; this is the cleanest apples-to-apples claim available between the two systems. (ii) **Random-mask control** — delete a random acyclic fragment instead of the true one, to test whether pocket conditioning alone recovers the author's edit site. Their Adrb1 cyanopindolol/carazolol recovery suggests it sometimes can, at n = 1.

**Do not** attempt their LTK-style prospective comparison, and do not cite their 1.36 nM against anything of ours — it is one compound from a chemist-filtered 227 → 13 → 8 funnel with an in-house QSAR model and expert synthetic triage in the loop, i.e. a campaign result, not a model metric.

---

### Flagged unverifiable from accessible text
1. Journal main Table 1 **means** for the six added baselines (DeLinker, DeepFrag, FFLOM, Fragmenstein, ScaffoldGVAE, SCMG) — SI gives only their standard deviations. The direction of the claim is the authors'; the margin is unread.
2. The assay behind **1.36 nM** (abstract). SI reports **6.60 nM** cellular IC50 for CA-B-1 — different assay, and I cannot reconcile them without the main text.
3. **"Causal-Delete"**: appears in the main-text Fig. 4 title and, spelled **"Casual-Delete"**, as the SI architecture heading — where the content is the ordinary GVP/geometric-message-passing generator. **No causal-inference content is visible anywhere.** Treat "causal" as meaning autoregressive decoding until the main text proves otherwise, and do not let it be mistaken for a causal-effect method adjacent to our edit-effect framing.
4. The **ALK transfer-learning** set size, the in-house QSAR model, and the finetuning schedule — described only qualitatively.
5. Any **validity / uniqueness / novelty / diversity** statistic for the generated molecules.
6. Whether `delete.pt` is the fragment, linker, scaffold or side-chain parameter set.
