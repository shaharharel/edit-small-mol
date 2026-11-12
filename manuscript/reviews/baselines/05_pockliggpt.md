# Baseline assessment: PockLigGPT

**Paper.** Varas Pardo, Marcos-Ayuso, Ulzurrun, Ríos Insua, Campillo. "PockLigGPT:
Pocket-Sequence-Conditioned Molecular Generation with GPTs and RL." *J. Chem. Inf. Model.* 66(17),
10847–10861, 17 Aug 2026. DOI 10.1021/acs.jcim.6c01980. **CC BY 4.0.**

**Sources read.** Full text read in full via the CC-BY deposit in Europe PMC / PMC13580112
(JATS full-text XML, ~62 kB of body text, both main tables complete). The ACS landing page, the ACS
PDF and the ACS SI are all behind a Cloudflare challenge and returned HTTP 403 to every fetch — but
the ACS copy was not needed, because the PMC deposit is the published version of record. Also read:
the GitHub repo `pvaras8/PockLigGPT_official` (full file tree + the RL config, prompt builder, model
adapter, model forward/generate, docking config — the repo answers the feasibility questions better
than the SI would have) and the HuggingFace model repo `pablovp8/PockLigGPT`.

**Not reached:** the Supporting Information PDF (S1–S9) and Supplementary Data. PMC hosts no
supplement for this article and ACS 403s. Everything that lives only in the SI is therefore
unverified — see §7. There is no arXiv or ChemRxiv preprint; a submitted version exists in the CSIC
repository (`hdl.handle.net/10261/445405`) but was not needed.

---

## 1. What the paper does

- **Task.** **De novo** ligand generation conditioned on a binding pocket. There is no input
  molecule, no scaffold to retain, no edit, no matched pair. The paper's own framing: "De novo drug
  design aims to generate molecules targeting specific protein pockets."
- **Inputs.** (i) The pocket as a **one-letter amino-acid sequence** — residues with ≥1 heavy atom
  within 6.0 Å of any ligand heavy atom, ordered by chain ID then residue number; (ii) **ProtT5-XL-
  UniRef50 per-residue embeddings** (1024-d) for those residues. No coordinates, no distances, no
  interaction types, no burial. 3D geometry enters the system **only** through the RL reward (a
  receptor `.pdbqt` + a docking box centre), never through the model's input.
- **Outputs.** A **SELFIES string** → one molecule. **No text, no rationale, no residue naming, no
  stated reason for the design.** (Their "Explainability" section is corpus-level SELFIES
  token-frequency shifts with BH-FDR correction; the residue-level interaction talk — Glu239/Leu241/
  Lys188 for DYRK1A etc. — is *post-hoc analysis performed by a human co-author* (E.U. "conducted the
  docking studies"), not model output.)
- **Architecture.** GPT-2 Medium, 350M params, decoder-only, 24L/16H/1024d, SELFIES vocab of 469
  tokens, block size **156 tokens total** (pocket + ligand share that budget). Pocket conditioning is
  an **additive** injection: `tok_emb = tok_emb + LayerNorm(pocket_emb)` at the AA token positions.
  Prompt format is literally
  `<SOS> <POCKET> <AA_SER> <AA_LEU> … </POCKET> <LIGAND> [tok]…`.
- **Training data (4 stages).** (1) ZINC20, ~750M filtered molecules (logP 1–4.5, MW 250–500), 10k
  iters at ~0.5M tokens/batch — ~80M distinct molecules actually seen, *not* a full epoch;
  (2) ChEMBL, 1.1M Lipinski-compliant, full-parameter FT; (3) pocket–ligand FT on **226K pairs**
  (CrossDocked + a Jglaser-derived affinity set filtered to pChEMBL > 6.5); (4) **per-pocket PPO**
  with reward `R = (1 − σ(vina_rescaled))^1.3 × exp(−(logP−2.5)²/2)`.
- **Headline metrics.** Table 2, 10 pockets from the TargetDiff test set, held out of stage-3 FT:

  | Method | Vina (kcal/mol ↓) | QED ↑ | logP | SAS ↓ | Diversity ↑ | Valid % | Unique % | Novel % |
  |---|---|---|---|---|---|---|---|---|
  | Reference (crystal) | −7.96 ± 2.64 | 0.43 ± 0.21 | −0.39 ± 4.09 | 3.39 ± 1.03 | – | 100 | 100 | – |
  | AR | −7.08 ± 2.07 | 0.42 ± 0.10 | 0.03 ± 1.64 | 4.63 ± 0.94 | 0.83 | 100 | 100 | 100 |
  | Pocket2Mol | −8.21 ± 3.25 | 0.55 ± 0.12 | 1.62 ± 1.50 | 3.27 ± 0.88 | 0.86 | 100 | 100 | 100 |
  | TargetDiff | −8.19 ± 2.22 | 0.38 ± 0.13 | 1.08 ± 1.95 | 4.74 ± 0.80 | 0.87 | 100 | 98.07 | 100 |
  | **PockLigGPT** | **−8.38 ± 0.98** | **0.75 ± 0.03** | 2.89 ± 0.35 | **2.92 ± 0.37** | 0.84 | 100 | 95.07 | 100 |

  Ablation (Table 1, 3 Alzheimer's pockets — DYRK1A 6EIF, BuChE 7Q1P, BACE-1 5TOL):

  | Stage | Vina ↓ | QED ↑ | logP | SAS ↓ | Unique % |
  |---|---|---|---|---|---|
  | ZINC20 pretrain | −7.90 | 0.76 | 2.44 | 3.12 | 100 |
  | + ChEMBL (ligand-only) FT | −8.08 | 0.63 | 3.27 | 3.11 | 100 |
  | + pocket–ligand FT | −8.45 | 0.52 | 3.72 | 3.70 | 100 |
  | ligand-only FT **+ RL** (no pocket cond.) | −9.26 | 0.74 | 2.55 | 2.27 | 89.19 |
  | pocket–ligand FT **+ RL** | **−9.90** | 0.73 | 2.95 | 3.53 | 87.76 |

  No experimental validation. Cross-check docking with Glide XP on the 3 AD targets, 256 molecules
  per target, top-10 inspected by hand.

---

## 2. Is this a real comparison for us? **No.**

Four independent mismatches, any one of which would disqualify it as a head-to-head baseline:

1. **Task.** De novo vs our analogue optimisation. It takes no input hit. There is no scaffold
   retention, no anchor atom, no edit operator — unlike Diffleop (§03), which at least does fragment
   growth off a retained scaffold. The only hook for an input molecule is incidental: the RL prompt
   builder (`pockliggpt/rl/prompts.py`) SELFIES-encodes a seed SMILES from ZINC250K and keeps its
   **first `initial_ligand_tokens` = 7 tokens** as the generation prefix. That is a diversity-seeding
   device, not analogue conditioning, and a left-to-right SELFIES prefix cannot express "keep this
   scaffold, change this R-group" — the edit site would have to fall at the *end* of the string.
2. **Conditioning.** Pocket **residue identity only** (+ a sequence-derived ProtT5 embedding). Our
   16-field per-ligand-atom contact table has no counterpart, and the two are not even the same *kind*
   of object: their pocket representation is **ligand-independent** (a property of the protein),
   whereas ours is a per-atom contact table that presupposes a pose. They cannot consume our input;
   we cannot consume theirs without discarding the hit.
3. **No rationale.** Our intent-then-SMILES design, residue specificity (0.417 vs 0.200 baseline),
   and the teacher-forcing hook have *no counterpart whatsoever*. They cannot be scored on it.
4. **Metrics.** They report Vina/QED/SAS/logP/diversity/validity/uniqueness/novelty. We report
   TC→reference, residue specificity, validity. The only shared axis is validity, and theirs is
   100% **by construction** because SELFIES decodes to a valid molecule by definition — it is a
   representation artefact, not a modelling result.

**Could it run on our inputs?** Not as posed. We could extract pocket sequences from our 3,992
complexes and run it, but it would ignore the hit entirely and emit an unrelated de novo molecule, so
TC→reference would collapse and residue specificity would be undefined. See §6 for the nearest fair
construction.

---

## 3. Availability: good on paper, unexercised in practice

- **Code.** `github.com/pvaras8/PockLigGPT_official`, **MIT**. Complete and reasonably well
  organised: full 4-stage training pipeline, PPO trainer, Meeko/Vina reward, tokenizer pickle, 7
  tests, 3 example receptor PDBQTs, two ProtT5 notebooks. Created 2026-05-21, last push 2026-10-19.
- **Weights.** `huggingface.co/pablovp8/PockLigGPT` — three checkpoints, **3.63 GB each** (~10.9 GB
  total): `ckpt_zinc_M.pt` (ZINC20), `ckpt_zinc_chembl.pt`, and
  `ckpt_zinc_chembl_cross_sequence_add.pt` (the pocket-conditioned one — the one we'd want). Licence
  on the HF card is MIT, but the card also says "research use… industrial or large-scale work
  requires contacting the developers," which contradicts MIT; worth a line in any usage statement.
- **Inference requirements.** 350M params → trivial, <2 GB in fp16 on one GPU. RL needs 2 GPUs
  (`train_device: cuda:0`, `ref_device: cuda:1`, bfloat16) + Vina/Meeko + 60 CPU cores.
- **Real friction.** (a) **No standalone inference/sampling script** — the README routes all
  generation through `scripts/run_rl_pipeline.py`, i.e. through RL. `model.generate(…, pocket_emb=…)`
  exists and is unit-tested, so sampling is reachable, but we'd write the harness.
  (b) The `SequenceAddAdapter` builds **one** pocket embedding template and `expand`s it across the
  batch — **one pocket per batch**; batching across 3,992 different pockets needs a rewrite.
  (c) The embedding pack ships as a **split archive** needing `assemble_embedding_pack.py` +
  `convert_embedding_pack.py`. (d) ZINC20 is not provided, must be sourced manually.
  (e) Checkpoint filenames **disagree**: the RL config points at `ckpt_zink_chembl_cross_sequence_add.pt`
  ("zink"), HF ships `ckpt_zinc_…` ("zinc").
- **Adoption signal — weak.** **0 GitHub stars, 1 fork, 0 HF downloads, 0 Crossref citations** as of
  2026-11-13. HF card citation says "Coming soon." Nobody has run this but the authors.

---

## 4. How we differ (reviewer-acceptable phrasing)

> PockLigGPT generates a molecule de novo from a pocket's amino-acid sequence and optimises it against
> a docking oracle; it takes no input ligand and emits no reason for its choices. We optimise a *given*
> hit, condition on the hit's per-atom contact geometry rather than on residue identity alone, and emit
> a residue-naming design intent before the molecule, so the structure can be read and the reason
> teacher-forced.

---

## 5. Who wins on what

**Where we lose (and how much of it is real):**

| Axis | Them | Us | Honest read |
|---|---|---|---|
| Validity | 100% | 0.985 | **Not a real loss.** SELFIES is valid by construction; we emit SMILES. Do not concede this as a modelling gap. |
| Vina docking score | −8.38 (RL'd) / −9.90 (3 AD targets) | we never optimise docking | **Real loss, but circular** — see below. They train on the metric; we don't train on it at all. |
| QED / SAS | 0.75 / 2.92 | inherited from the hit | Real loss, and uninteresting: they *reward* logP≈2.5 and have a 750M-molecule drug-like prior. Note the crystal ligands score QED **0.43** — these metrics don't track real ligands. |
| Novelty / diversity | 100% / 0.84 | low by design | A de novo metric. We **want** close analogues; scoring 0 on novelty is the task, not a failure. |

**Where we win:**

- **TC→reference.** They have no mechanism to recover a known analogue of a given hit; a de novo
  generator's ECFP4 TC to one specific held-out ligand is not comparable to our 0.679. Caveat for our
  own write-up: 0.679 sits *below* our own "any other real analogue" null of 0.700, so the honest
  claim is "matches the null," and a win here is a **task-mismatch win, not a modelling win**.
- **Rationale / residue grounding.** 0.417 vs 0.200 matched-count baseline, against *nothing*. No
  contest; they produce no per-molecule text.
- **Ablation rigour.** They have **no wrong-pocket / shuffled-sequence control anywhere in the paper.**
  Our deliberately-wrong-pocket control (0.013 Tanimoto) and our contact-table-withheld ablation are
  methodologically ahead of them. This is a point we can make in review.

**The Vina loss is circular and should be said out loud.** Table 2's PockLigGPT row is **per-pocket
PPO'd against Vina and then evaluated by Vina**, while AR/Pocket2Mol/TargetDiff "were generated
according to their original training objectives and not directly optimized against Vina" (the authors'
own sentence). After directly optimising the evaluation metric with a dedicated RL run per pocket, the
margin over Pocket2Mol is **0.17 kcal/mol** (−8.38 vs −8.21) and over the crystal ligands 0.42. The
authors also concede the RL'd model "is no longer considered zero-shot for that pocket." The defensible
reading of Table 2 is *ties 3D baselines on the metric it was trained on*, not "best."

---

## 6. Implementation cost, and the only fair comparison

**The headline configuration is out of reach at our scale.** From `config/rl/sequence_add.yaml`:
`epochs: 30 × num_rollouts: 256` ⇒ **~7,700 Vina dockings per pocket** (exhaustiveness 8, 5 poses),
one PPO run per pocket, 2 GPUs. Across 3,992 complexes that is ~30M dockings. Not happening — and note
RL is also what produces nearly all of their reported gain (−8.45 → −9.90).

**So any run of ours is restricted to the zero-shot checkpoint** `ckpt_zinc_chembl_cross_sequence_add.pt`
— which is, by their own Table 1, their **weakest** pocket-conditioned configuration (QED 0.52, SAS
3.70, worse than their own ZINC20 baseline on both). A head-to-head there would flatter us and
misrepresent their method; we should present it as an ablation probe, not as "we beat PockLigGPT."

**Nearest fair comparison — recommended.** Use it as a *pocket-conditioner ablation*, not a generator
baseline. Seed generation with our hit's SELFIES prefix (raise `initial_ligand_tokens` from 7 to a
longer prefix — a one-line config change the `PromptBuilder` already supports), condition on the pocket
sequence extracted from the same complex, and measure (a) TC→reference and (b) the **wrong-pocket
delta**. That tests exactly the question we care about: does a sequence-only pocket conditioner move
the *molecule* more or less than our contact table does?

**Cost: ~1.5–3 days of work, modest GPU.**
- Pocket sequences at 6.0 Å from our complexes: trivial, we already hold the structures. (½ day incl.
  ordering/format checks.)
- ProtT5-XL-UniRef50 per-residue embeddings for 3,992 pockets: notebook provided; ProtT5-XL is ~3B
  params/11 GB but pockets are short. A few GPU-hours. (½ day.)
- Batched sampler + per-example pocket-embedding padding (rewrite of `SequenceAddAdapter`'s
  one-pocket-per-batch `expand`): ~0.5–1 day. Inference itself is a few GPU-hours for 400k short
  sequences on one GPU.

**Blockers to flag before committing:**
1. **The 156-token budget is shared between pocket and ligand.** Their bundled example pocket is 34
   residues; a 6.0 Å pocket on a larger site can be 60–100+, which squeezes or eliminates the ligand
   budget. We must histogram our pocket lengths first — this could silently truncate a chunk of our
   3,992 complexes. (Compare: our own cofold queue already lost all 82 covalent anchors to a 1,187-aa
   length cap. Same failure mode, different cap.)
2. No rationale output ⇒ residue specificity is simply **not comparable**. Don't promise that column.
3. 0 external users; expect undocumented breakage (filename mismatch, split archives, Spanish-language
   error strings).

---

## 7. Numbers I could **not** verify

Everything above from Tables 1–2, the reward equations, the dataset sizes and the architecture is from
the published full text and is solid. The following are **not verified**:

- **All SI content** (S1 tokenizer; **S2 computational cost** — the one I most wanted; S3 reward
  sensitivity; S4 PPO hyperparameters/loss curves/RL cost; S5 conditioning-strategy variants; S6
  docking-formula correlations; **S7 per-pocket benchmark breakdown**; S8 top-ranked molecules; S9 the
  mathematical formulation of pocket conditioning) and **Table S1** (the Glide XP docking scores) and
  **Table S5**. ACS 403s on the SI and PMC hosts no supplement.
- The **Lingo3DMol** comparison, which the paper says is in SI-S7 and excludes from Table 2 because
  generation failed on some of the 10 pockets. Unseen.
- **My ~7,700-dockings-per-pocket figure** is derived by me from `config/rl/sequence_add.yaml`
  (`epochs × num_rollouts`), **not stated in the paper**. Treat as an order-of-magnitude estimate; the
  true figure is in SI-S2.
- **My inference** that the 3.63 GB checkpoints embed optimizer state (350M params × 4 B ≈ 1.4 GB, so
  3.63 GB ≈ params + Adam moments). Not stated anywhere.
- Whether Table 2's PockLigGPT row is RL'd. The text strongly implies yes ("when RL is subsequently
  applied to a fixed pocket… no longer zero-shot"; "PockLigGPT includes a pocket-specific RL stage"),
  but it is never stated in the table or caption. Load-bearing for the circularity critique in §5, so
  worth one careful re-read of §"Benchmarking" if we cite it.

---

## 8. Verdict and the one thing worth taking from it

**Not a fair head-to-head baseline. Do not run it as one.** Different task (de novo, no input hit),
incompatible conditioning (ligand-independent residue identity vs our per-atom contact table), no
rationale output, and a headline that requires a per-pocket RL run we cannot afford 3,992 times.

**But it is the most useful paper in this batch for our own ablation story**, because it is a clean
natural experiment on exactly the question the team lead raised — *does a sequence-only pocket
conditioner buy anything for the molecule?* Their answer, from their own data, is **barely**:

- Pocket-sequence FT moves median Vina by **0.37 kcal/mol** (−8.08 → −8.45) and *degrades* QED
  0.63 → 0.52 and SAS 3.11 → 3.70, while logP rises 3.27 → 3.72. Given that the authors themselves
  introduce logP regularisation *specifically because* "the hydrophobic contribution can favor
  molecules with increased global hydrophobicity," that 0.37 kcal/mol is **not established as
  pocket-specific at all** — it is consistent with "bigger, greasier molecules dock better," a Vina
  artefact they document in their own methods.
- Their words, twice over: pocket-sequence conditioning "alone produces only a limited pocket-specific
  shift," and "provides a contextual prior… but does not by itself enforce strong pocket-specific
  binding strategies."
- The *only* number arguing pocket sequence adds something beyond the docking oracle is −9.26
  (RL, no pocket cond.) vs −9.90 (RL + pocket cond.), i.e. **0.64 kcal/mol on n = 3 pockets**, medians,
  no seed replicates, no per-pocket breakdown in the main text, **and no wrong-pocket control**.
- Their "pocket specificity" evidence is corpus-level SELFIES token-frequency shifts between targets —
  a *distributional* claim ("DYRK1A generations are enriched in [N]"), not per-pocket correctness. They
  even concede RL drives "partially convergent chemical trends" across all three targets, i.e. the
  optimiser is learning generic Vina-pleasing heuristics (rigidity, heteroatom density, less branching)
  rather than pocket-specific design.

**This corroborates our ablation rather than threatening it.** Our finding — contact table governs the
prose (groundedness 1.000 → 0.432 when withheld) and not the molecule (TC 0.679 → 0.680) — is the same
phenomenon measured more rigorously: we ran the wrong-pocket control they didn't. The honest joint
statement for the manuscript is that **sequence-level and contact-level pocket conditioning both move
the molecule weakly, and that published pocket-conditioned generators have not demonstrated otherwise
either.** Cite PockLigGPT in the discussion as independent support for the negative half of our
ablation, and as the literature's own admission; do not cite it as a baseline we beat.
