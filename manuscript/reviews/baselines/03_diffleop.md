# Baseline assessment: Diffleop

**Paper.** Qiao, Chen, Xie, Huang, Zhang, Deng, Rao, Deng, Meng, Wang, Xu, Chen, Xie, Zheng, Yang, Li, Lei.
"A 3D pocket-aware lead optimization model with knowledge guidance and its application for discovery of
new glutaminyl cyclase inhibitors." *Briefings in Bioinformatics* 26(4), bbaf345, July 2025. CC BY-NC 4.0.
Preprint (different title, same model): arXiv:2504.21065, "A 3D pocket-aware and affinity-guided diffusion
model for lead optimization."

**Sources read.** OUP full text (read in two passes, ~126 kB of page text; the abstract could not be
reproduced verbatim because of the fetcher's quote limit), arXiv HTML v1 (all three main tables),
GitHub `biomed-AI/Diffleop` README, Zenodo record 14210941 (checkpoints). **Not reached:** the PDF
supplement (Tables S1–S6), which is where the paper says validity / similarity / diversity /
connectivity numbers live. There is no PMC version.

---

## 1. What the paper does

- **Task.** Structure-based lead optimization by *3D fragment growth*: scaffold decoration (grow a
  substituent off a retained scaffold at a given anchor atom), fragment linking (build a linker between
  two retained terminal fragments), and scaffold hopping. Not de novo, not full-molecule generation.
- **Inputs.** The protein pocket as 3D atom coordinates + the retained ligand substructure (scaffold or
  terminal fragments), both held fixed as conditions, plus an anchor atom index marking where to grow.
  In the released code: a pocket `.pdb`, a `retain.sdf`, and `--anchor_id_given_1`.
- **Outputs.** The missing fragment as 3D atom coordinates, atom types and **bond types** (the bond
  diffusion over a fully connected graph is one of the two contributions). Written as `.sdf`. **No text,
  no rationale, no residue naming, no stated reason for the edit.**
- **Method.** An E(3)-equivariant diffusion model; a separately trained EGNN affinity predictor whose
  *gradients* steer denoising ("affinity guidance"); bond diffusion on fully connected graphs.
- **Training data.** A newly assembled set of 168,146 complexes / 8,806 proteins: PDBbind co-crystals,
  plus ChEMBL actives aligned and redocked into those same pockets, filtered to Morgan Tc > 0.7 against
  the co-crystal ligand and screened for clashes. 50k complexes used for the benchmark model, 100k for
  the QC application. **The split procedure is never described** — no sequence-identity or scaffold
  criterion is stated; only "100 test protein targets."
- **Headline metrics** (100 test targets, 100 molecules generated per pocket/fragment pair, **top 5 by
  predicted affinity scored**):

  | Scaffold decoration | Affinity (TANKBind, −logKd) | High Affinity % | QED | SA | LogP | Lipinski |
  |---|---|---|---|---|---|---|
  | Reference ligand | 6.69 | – | 0.642 | 0.770 | 3.085 | 4.92 |
  | GraphBP | 5.47 | 17.80 | 0.605 | 0.316 | 0.567 | 4.82 |
  | AR | 5.93 | 18.60 | 0.660 | 0.710 | 1.855 | 4.91 |
  | Pocket2Mol | 5.78 | 21.80 | 0.704 | 0.773 | 1.904 | 4.96 |
  | **Diffleop** | **6.71** | **52.60** | 0.654 | 0.706 | 2.772 | 4.93 |

  | Linker design | Affinity | High Affinity % | QED | SA | LogP | Lipinski |
  |---|---|---|---|---|---|---|
  | Reference | 6.73 | – | 0.601 | 0.770 | 2.867 | 4.84 |
  | DiffLinker | 6.09 | 26.87 | 0.625 | 0.321 | 0.893 | 4.91 |
  | DiffLinker* | 6.14 | 28.89 | 0.635 | 0.321 | 0.948 | 4.93 |
  | **Diffleop** | **6.74** | **54.60** | 0.604 | 0.716 | 2.865 | 4.85 |

- **Wet-lab application.** Their own QC hit A0 (5-methyl-1H-imidazole series, sQC IC50 1.82 µM) was
  decorated at ortho/meta/para of an outer phenyl (500 molecules per position), filtered by predicted
  affinity ≥ 6.4 and SA ≥ 0.88 and by **manual visual inspection of binding mode**, giving A1–A3.
  A1: sQC IC50 **0.008 µM** (226× A0, 3.5× PQ912's 0.036 µM), gQC 69 nM. This is a genuine, strong result.

**Two numbers I would not let past review.** (a) "High Affinity %" is computed on the **top 5 of 100**
generated molecules, i.e. best-of-20 oracle selection, and the oracle (TANKBind) is a *predicted*
affinity — there is no measured-potency evaluation anywhere in the benchmark. (b) Affinity guidance is
*itself* driven by an affinity predictor and then evaluated with a predicted-affinity scorer; the
ablation shows exactly the expected circularity (removing affinity guidance drops High Affinity 52.6 →
40.4 while *improving* QED 0.654 → 0.697). Also note Diffleop's mean affinity (6.71) is statistically
indistinguishable from the reference ligand's (6.69); the entire claim rests on the best-of-100 rate.

---

## 2. Is it a real comparison for us? **No — adjacent task, incompatible interface.**

| | Diffleop | Ours |
|---|---|---|
| Task | grow a fragment at a **given anchor** on a retained scaffold | propose a **whole analogue** of a hit, edit site unconstrained |
| Conditioning | 3D pocket atom coordinates + retained 3D substructure + anchor index | per-atom **contact table** (16 fields/heavy atom) derived from the complex |
| Output | 3D coordinates + bond types (`.sdf`) | design **intent naming residues**, then analogue SMILES |
| Rationale | none | the point of the model |
| Objective | maximise *predicted* affinity | reproduce the medicinal chemist's actual next analogue |

Three things make it not a like-for-like baseline:

1. **It requires an anchor.** The user must pick the growth atom. We do not supply one and do not want to
   — choosing *where* to edit is part of what we are evaluating. Running Diffleop on our inputs means
   either hand-picking anchors (leaking the answer: our reference analogue tells you the site) or
   enumerating all anchors and taking a best-of, which is not the same decision problem.
2. **It cannot produce a rationale.** Our headline claims — residue specificity 0.417 vs 0.200 baseline,
   groundedness collapsing 1.000 → 0.432 when the contact table is withheld — have no Diffleop analogue.
   It is not that Diffleop scores badly on them; the metrics are undefined for it.
3. **Its evaluation axis is different.** It never measures similarity to a reference analogue, which is
   our primary metric (TC→reference 0.679). We never measure docked/predicted affinity of our outputs.
   There is no shared number in the two papers.

It *could* be run on our inputs with work (§6), because our complexes are exactly its input type
(crystal structures where available, Boltz cofolds otherwise) — but only after we invent an anchor
policy, and only against a re-derived metric.

---

## 3. Availability: good, among the better-released models in this space

- **Code:** https://github.com/biomed-AI/Diffleop — MIT licence, `environment.yaml`, `scripts/train.py`,
  `scripts/sample.py`, `scripts/sample_for_case.py` (custom pocket + fragments), 500-point CrossDocked demo.
- **Weights:** https://zenodo.org/records/14210941 — "Models of Diffleop", `diffleop_dec.pt` and
  `diffleop_linker.pt`, 60.6 MB each, CC BY 4.0. **No retraining needed.**
- **Requirements:** CUDA GPU (`--device cuda:0`). Affinity evaluation additionally needs a TankBind
  checkout with their `cal_affinity.py` dropped into `TankBind/examples` — an extra env, and TankBind is
  itself a research repo with its own pinned deps. Model size and sampling throughput are **not stated
  anywhere I could reach**; from the 60 MB checkpoint it is a small EGNN (order 10M params), and
  diffusion sampling at 100 molecules per pocket is the cost driver, not the model.

---

## 4. How we differ (reviewer-acceptable phrasing)

> Diffleop optimizes a lead by diffusing new 3D atoms into a pocket at a user-specified anchor, scored by
> a learned affinity oracle; it emits coordinates and no explanation, and its benchmark reports predicted
> affinity rather than agreement with the analogue a chemist actually made. We instead condition a
> language model on a per-atom interaction summary of the same complex and require it to state a
> residue-level design intent before emitting the analogue, so the proposal is auditable and the edit
> site is the model's decision, not an input.

---

## 5. Who wins on what

**They win:**
- *Any geometry-dependent claim.* They output a pose; we output SMILES. Predicted binding affinity,
  pocket shape complementarity, interaction-count analyses — we cannot compete and should not try.
- *Prospective wet-lab validation.* A1 at 8 nM against a clinical candidate is a stronger single piece of
  evidence than anything in our paper. Expect a reviewer to say so. Our answer is scope (they report one
  target and three compounds, hand-filtered; we report cohort-scale retrospective agreement) — not
  a counter-claim.
- *QED/SA/Lipinski of outputs* are reasonable (QED 0.654, SA 0.706) and they beat the 3D de novo
  baselines decisively on SA, where GraphBP/DiffLinker sit at 0.32.

**We win:**
- *Any claim about rationale.* Residue specificity 0.417 vs a 0.200 distance-sort baseline, and the
  ablation where removing the contact table leaves TC→reference flat (0.679 vs 0.680) but collapses
  groundedness (1.000 → 0.432), have no counterpart. Diffleop scores 0 by construction.
- *Agreement with real chemistry.* Our TC→reference 0.679 against a 0.700 "any other real analogue"
  null is a measured, honest comparison to what chemists made. Their benchmark has no such metric; their
  only reference comparison is 6.71 vs 6.69 predicted affinity, which is a tie.
- *Honest conditioning ablations.* We report that a deliberately wrong pocket costs only 0.013 Tc — an
  uncomfortable number we publish. Diffleop reports no pocket-shuffle control at all, so the claim that
  its gains are pocket-driven rather than affinity-oracle-driven is untested; its own ablation ("without
  affinity guidance": 52.6 → 40.4) suggests the oracle is doing most of the work.
- *Edit-site selection.* We choose where to edit; they are told. A reviewer who notices this will treat
  our task as strictly harder.
- *Scale and chemistry grounding of supervision.* Our PocketPair-21k is oriented by **measured** within-
  assay potency deltas; their 168k complexes are Tc>0.7 redocks with (as far as I can verify) no potency
  supervision at all — the paper never states the affinity predictor's label source.

**Honest draw / risk:** validity. We report 0.985 SMILES validity. Their validity/connectivity numbers
are in the unreachable supplement; 3D diffusion models typically report high validity after
post-processing, so do not claim a win here without the supplement.

---

## 6. Implementation cost, and what I recommend instead

**Running Diffleop on our 21k val inputs: not a fair comparison, and I would not do it.** The blocker is
structural, not engineering: Diffleop needs an anchor atom per input, and supplying one from the
reference analogue leaks the answer to the question we are scoring.

If the team wants it anyway, the mechanical cost is modest:

- **Work:** ~2–3 days. 0.5 d to stand up `environment.yaml` + checkpoint; 0.5 d to convert our complexes
  (we already have pocket PDBs and ligand SDFs, so this is bookkeeping, not new science); 1 d to write an
  anchor policy and an evaluation that computes TC→reference on their `.sdf` output; 0.5 d for the
  separate TankBind env if we also want their native metric.
- **GPU:** one CUDA GPU. Diffusion sampling is the cost. At their protocol (100 molecules/input) 21k
  inputs is ~2.1M samples — that is a multi-week single-GPU job. A 1–2k input subsample at 20
  molecules/input is the only sane scale, ~1–2 GPU-days.
- **Blockers:** (a) anchor selection, as above; (b) TankBind as a second environment; (c) their training
  set is PDBbind + ChEMBL redocks, which almost certainly overlaps our PDB-derived pairs — we would have
  to deduplicate by protein before quoting any number; (d) sampling throughput is unreported, so the GPU
  estimate above is mine, not theirs.

**Nearest fair comparison I would actually run.** Two options, cheapest first:

1. **Cite, don't run.** Place Diffleop in related work as 3D-generative lead optimization and state the
   two axes it cannot be scored on (no rationale, anchor supplied). This is defensible and costs an hour.
2. **A site-matched, pose-free head-to-head on a few hundred inputs.** Give Diffleop the *same* edit site
   our model chose (not the reference's), 20 samples per input, and compare TC→reference of its best
   sample against our greedy single sample on the identical input set, protein-deduplicated against its
   training set. Report it as an anchor-advantaged upper bound for Diffleop, since it gets a site and
   best-of-20 while we get neither. ~3 days plus ~1 GPU-day. This is the only version where a loss would
   mean something and a win would be unimpeachable.

A third framing, if a reviewer presses on affinity: dock our analogues and theirs with the same scorer
and report both. Cheap, but it moves us onto their axis and invites all the usual objections to docking
scores, so I would only do it under duress.

---

## Unverified / flagged

- **Full abstract** could not be reproduced verbatim (fetcher quote limit); content above is from the
  body text and the arXiv version.
- **Supplement (Tables S1–S6)** not reached: validity, novelty, diversity, connectivity, similarity-to-
  reference, and Vina scores are all there per the body text, and are therefore **unverified**.
- **Affinity-predictor label source** (PDBbind −logKd? ChEMBL pChEMBL? docking score?) is not stated in
  either the OUP body text or the arXiv HTML. This matters: if the guide signal is a docking score, the
  whole affinity claim is a self-consistency result.
- **Train/test split criterion** is not described. With PDBbind + ChEMBL redocks into the *same* pockets,
  pocket-level leakage between the 168k complexes and the 100 test targets is plausible and untested.
- **Training hyperparameters, parameter count, training wall-clock, sampling time** are reported nowhere
  I could reach.
- The Briefings title and the arXiv title differ; treat them as the same model (same first author, same
  method, same contributions) but cite the journal version.
