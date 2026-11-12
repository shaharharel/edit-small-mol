# Baseline assessment: PROBE (arXiv 2606.00555)

**Verdict in one line: NOT a drop-in baseline for our headline claim. It is a training-free agentic
search with a docking oracle in the loop, optimising in-silico proxies (Vina/QED/SA) with no measured
bioactivity and no analogue constraint. It IS the best available comparator for one axis — pocket-grounded
rationale text — and should be run on that axis only.**

- **Paper**: "Probe Before You Edit: Probing-Guided Molecular Optimization for LLM Agents in
  Structure-Based Drug Design". Zaifei Yang, Weiyu Chen, Yaqing Wang, James Kwok (HKUST / CityU HKIAIS /
  BIMSA). arXiv:2606.00555v3, submitted 30 May 2026, v3 23 Oct 2026. **Accepted at NeurIPS 2026.**
  Licence CC BY 4.0. The arXiv id resolves and the paper is what the URL suggests.
- **Code**: https://github.com/yzf-code/PROBE (MIT). Read in full (PDF text extraction of all 10 body
  pages + Appendices A–I) plus the repo tree and README.

---

## 1. What the paper actually does

**Task.** Post-hoc, iterative optimisation of a *given* pocket–ligand complex. Not de novo generation —
they explicitly position themselves as "what should happen after generation". Objective is joint
improvement of binding affinity (A = −Vina Dock) and druggability (D = QED + SA).

**Inputs.** A receptor pocket PDB plus a 3D ligand pose. In Table 4 the pose comes from a
CrossDocked2020-trained 3D generator (TAGMol / MolCRAFT / MolJO, 10 molecules per pocket, 100 test
pockets); in Section 5.2 it is the Binding MOAD crystal reference ligand. **Every candidate is re-docked
with AutoDock Vina.**

**Pipeline (no training at all — all agents are prompted GPT-4o-mini).**
1. *Site Map* (§4.1, App. B): BRICS-decompose the ligand; aggregate three signal streams —
   **interaction** (PLIP: H-bonds, hydrophobic, π-stacking, salt bridges, halogen bonds, water bridges,
   each with ligand atom indices + protein residue label), **geometric** (per-atom VDW-scaled steric
   clash, solvent exposure by nearest-receptor-atom distance, electrostatic repulsion from Gasteiger
   charges, buried-unsatisfied polar N/O split into hydrophobic burial vs geometry mismatch), and
   **property** (MW, logP, TPSA, QED, ligand efficiency, chiral centres, spiro/bridgehead/macrocycle
   alerts, Bertz complexity, ring-hybridisation rules, PAINS/toxicophore SMARTS). A diagnosis LLM then
   labels fragments as SYNERGY / TENSION / LIABILITY with free-text `local_symptoms`.
2. *EditManual* (§4.2): 12 controlled probe edits per input molecule — 3 strategies (affinity-first,
   druggability-first, co-optimisation) × 4 magnitudes (high / medium / low / **counterfactual control**).
   All 12 are docked and scored; response shapes (monotone / cliff / saturation / flat) are distilled into
   per-site allowed/forbidden actions, quantitative property constraints, a structural blacklist and
   cross-site rules.
3. *DMTA loop* (§4.3): 5 rounds. Affinity agent + druggability agent each draft one localised edit,
   cross-review each other, a co-optimisation agent reconciles into a third. All three are instantiated by
   a **fragment-assembly engine** (RDKit; 4 actions: DELETE_GROUP / GROW / REPLACE_TERMINAL_GROUP /
   REPLACE_SIDECHAIN_OR_RING) retrieving from an 8,505-fragment BRICS library built from ChEMBL Phase-1
   compounds, ranked by 0.5·QED + 0.5·SA − 0.15·N_chiral. Candidates selected by a hypervolume score
   S_bal that is **zero unless both objectives improve** over the initial ligand. Total 27 Vina calls per
   molecule (12 probes + 15 loop candidates).

**Headline numbers (all from their tables, as reported, not independently verified).**
| | Vina↓ | QED↑ | SA↑ | Success↑ | MRR↑ | QikProp↑ |
|---|---|---|---|---|---|---|
| MolCRAFT seeds (random-10) | −7.679 | 0.503 | 0.684 | 24.8% | 63.5% | 22.2% |
| MolCRAFT + CIDD (best prior agent) | −7.946 | 0.598 | 0.739 | 31.40% | 70.60% | 37.10% |
| **MolCRAFT + PROBE** | **−9.203** | **0.689** | **0.766** | **63.76%** | **79.58%** | **52.61%** |
| MolJO + PROBE (overall rank 1) | −9.626 | 0.664 | 0.782 | 67.67% | 66.94% | 34.69% |
| MolCRAFT best@27 + PROBE | −9.574 | 0.735 | 0.796 | 72.0% | 84.9% | 59.9% |

"Success" = Vina < −8.18 **and** QED > 0.25 **and** SA > 0.59 (TargetDiff convention).
Diagnostics: joint-improvement per edit 52.8% for PROBE vs 16.6/28.4/25.1/18.6% for
MoLLM/CIDD/CIDD+MOO/LIDDIA; objective interference 25–30% vs 61–88%. External Binding MOAD: PROBE
Success 56.35% vs the crystal reference ligand's own 27.8% and CIDD's 22.2% (CIDD *lowers* affinity and
Success below the ligands it starts from). Three independent runs: Success 63.08% ± 0.63%. Backbone
swap (GPT-4o, Gemini-3-flash) moves Success only 63.76 → 64.68 → 65.82%.

---

## 2. Is it a real comparison for us?

**Same task family: yes, genuinely.** This is analogue/ligand optimisation of a given molecule in a given
pocket, not de novo. Of the SBDD literature this is about as close as the task gets to ours.

**Same conditioning: yes, and uncomfortably similar in spirit.** Their diagnosis signals are essentially a
hand-built version of our contact table — PLIP typed interactions with residue labels and ligand atom
indices, per-atom nearest-receptor distances, solvent exposure, burial, clash. We have 16 fields per heavy
atom learned into the model; they have the same information categories serialised into a prompt. A
reviewer who knows this paper will ask why our contact table is a contribution and their PLIP+geometry
report is not. Our answer has to be the *learned, teacher-forceable* intent and the measured-potency
supervision, not the fields themselves.

**Does it produce a rationale: yes — and in the same order as ours.** The site map carries free-text
`local_symptoms` naming specific residues (e.g. "interactions with HIS 311 and LYS 438"), the EditManual
carries justified allowed/forbidden actions, and each design agent emits an `"analysis"` string **before**
the edit. So this is a real rationale comparator, not a null baseline. Note their rationale is only
*moderately* reproducible across runs: site-type agreement 83%, fragment-name agreement 69%,
`local_symptoms` TF-IDF cosine 0.53, EditManual pharmacophore-role agreement 61% (chance levels 31%, 7%,
0.15, 9% respectively).

**Could it be run on our inputs: in principle yes, at scale no.** `probe run` wants exactly what we have
— a receptor PDB and a 3D ligand pose (crystal or Boltz cofold both qualify). But (a) 21k inputs × 27 Vina
docks ≈ 567k docking runs, and (b) **PROBE has no target our metrics can score.** It optimises Vina/QED/SA;
it never sees a measured potency delta, and it is explicitly permitted to replace the core scaffold
(high-intensity co-optimisation = "Replace core scaffold"), so its output is not constrained to be an
analogue at all. Running it unchanged produces molecules that cannot be scored on TC→reference or on
measured ΔpIC50 in any meaningful way.

**Where it is NOT a fair comparison, plainly:** it has an oracle in the loop and we do not. It gets 27
docked evaluations per molecule and keeps the best; we emit one molecule greedily with no scoring pass.
Any direct Vina/Success comparison measures the oracle, not the model.

---

## 3. Availability

- **Repo**: https://github.com/yzf-code/PROBE — MIT licence, created 2026-10-25, last push 2026-11-08,
  **2 commits**, 65 files, ~95 KB, 2 stars. Real and complete-looking: 5 agent modules, docking/signals/
  fragments chemistry layer, 17 prompt templates, full evaluation suite (Vina, QED, SA, MRR, QikProp,
  comparability/ranking).
- **No weights, and none needed** — there is no trained model. It is prompts over an OpenAI-compatible
  endpoint (`OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`). Paper used GPT-4o-mini.
- **Not bundled**: CrossDocked2020 inputs (prepare per CIDD's instructions), and the 8,505-fragment BRICS
  library — build it with `scripts/download_chembl.py` + `scripts/build_fragment_library.py`.
- **Requirements**: Python ≥3.10, RDKit, Open Babel, AutoDock Vina, PLIP; **Linux recommended** for
  docking + PLIP. **CPU only, no GPU.** Their runs: 64 parallel workers on a 96-core Xeon Gold 6342.
- Maturity caveat: 2 commits, three weeks old, 2 stars. Expect to debug it ourselves; nobody else has.

---

## 4. How we differ (reviewer-acceptable phrasing)

> PROBE is a training-free agentic search: a frozen general-purpose LLM reads a PLIP-and-geometry report,
> then explores 27 docked candidates per molecule and keeps the best under a Vina/QED/SA hypervolume. Our
> system is a finetuned conditional generator that, in a single forward pass and with no oracle at
> inference, emits a residue-grounded design intent and then one analogue conditioned on it — trained on,
> and evaluated against, *measured* within-assay potency deltas and real published analogues rather than
> docking proxies, and constrained to the analogue regime rather than free to replace the scaffold.

---

## 5. Who wins on what

**We expect to lose:**
- **Vina / Success ratio / any docking metric.** They optimise Vina directly with 27 evaluations per
  molecule; we never dock at inference. Their Success 63.8–67.7% vs seeds' 24.8% is mostly the oracle. We
  should not enter this comparison without a budget-matched arm, and even then we lose.
- **QED / SA / druggability.** They have a dedicated druggability agent *and* a fragment library pre-ranked
  by 0.5·QED+0.5·SA. We have no druggability term anywhere. Expect a reviewer to ask; we should at minimum
  *report* QED/SA/Lipinski on our outputs even if we do not optimise them.
- **Validity.** Their edits are constructed by RDKit fragment assembly, so validity is ~100% by
  construction vs our 0.985 from a language model. (Flagged: they never report a validity number — this is
  my inference from the fragment-assembly design, not a reported figure.)
- **Possibly pocket-sensitivity rhetoric.** Our wrong-pocket penalty is 0.013 Tanimoto, which is weak.
  PROBE's candidates are re-docked into the actual pocket, so the oracle enforces pocket dependence even
  where the LLM text is generic. A hostile reviewer could cite PROBE to argue our pocket conditioning is
  decorative. Pre-empt this by leaning on the groundedness ablation (1.000 → 0.432), which is the strong
  result, not on the Tanimoto number.

**We expect to win:**
- **Analogue faithfulness.** They report *no* similarity-to-start metric at all, and their highest-intensity
  co-optimisation move is literally "replace core scaffold". Our TC→reference 0.679 against a
  real-analogue null of 0.700 has no counterpart; measured on their outputs it would almost certainly be
  far lower. This is our cleanest structural win and it is a *task-definition* win, not a tuning win.
- **Grounding in measured data.** Their Appendix H concedes it: "improvements in these scores do not
  establish experimental binding affinity, safety, or synthetic feasibility". Nothing in PROBE touches a
  measured potency value. Our supervision is within-assay ChEMBL deltas (every pair oriented weak→strong)
  plus intents mined from 1,622 SAR papers.
- **Rationale specificity and reproducibility.** Our residue specificity 0.417 vs a matched-count
  distance-sort null of 0.200 is a null-controlled number; they run no null for residue grounding at all
  (their ablations are end-task only). And their rationale is only moderately stable across runs
  (`local_symptoms` TF-IDF 0.53, role agreement 61%), whereas ours is a single deterministic greedy decode.
  Our groundedness ablation — withholding the contact table leaves TC→reference flat (0.679 vs 0.680) while
  collapsing residue groundedness 1.000 → 0.432 — is a cleaner causal demonstration than anything in their
  paper.
- **Inference cost, by orders of magnitude.** One forward pass vs 27 Vina docks + ~30 LLM calls per
  molecule.

---

## 6. Implementation cost, and what to run instead

**It cannot be run on our 21k val inputs. Say so plainly.** 21k × 27 docks ≈ 567k AutoDock Vina runs
(≈5–10k core-hours; ~4–8 days even on their 64-worker configuration) plus ~630k GPT calls — and the output
still would not be scorable on our metrics, because PROBE optimises a different objective and is not
constrained to produce analogues.

**Nearest fair comparison — two arms, run both:**

**(a) Rationale-only head-to-head (cheap, high value, do this first).** Run *only* PROBE's diagnosis +
site-map stage (§4.1 / App. B — no probing, no optimisation loop, no extra docking since we already have
poses) on 200–300 of our held-out complexes. Score its residue mentions with *our* residue-specificity
metric and against our distance-sort null, head-to-head with our emitted intents. This isolates the single
axis that genuinely transfers and needs no docking budget. ~1–2 days engineering, no GPU, trivial API spend.

**(b) Full pipeline on a matched subset (the arm a reviewer will accept).** 100–200 complexes from our
held-out PocketPair set where the measured strong analogue is known. Run full PROBE (27 docks each ≈
2.7–5.4k docks, ~1–2 days on a 64-core Linux box). Then score **both** systems on **both** metric sets:
ours (TC→the real published analogue, residue specificity, sign of measured ΔpIC50 where the proposed pair
exists in ChEMBL) and theirs (Vina, QED, SA, Success), stating openly that we do not optimise Vina/QED/SA
and that PROBE gets 27 oracle calls to our one. The honest headline we can defend: *PROBE wins the
docking-proxy objective it optimises; we win analogue faithfulness and measured-potency agreement at 1/27th
the evaluation budget.*

**Blockers:** Linux-only docking/PLIP stack (our ZAP70/Boltz tooling is on macOS — this has to go on the
Linux box); build the ChEMBL BRICS fragment library first; the repo is 2 commits old and effectively
unexercised, so budget debugging time; needs an OpenAI-compatible key.

**Also worth doing regardless of PROBE:** add a Vina/QED/SA/Lipinski report over our own outputs. PROBE's
existence makes "you never checked druggability or docking" a near-certain reviewer complaint.

---

## Numbers I could not verify from the paper

- **Figure 2 compute-budget values** (per-molecule wall-clock and token counts for PROBE vs CIDD vs
  CIDD-27-rounds) exist only inside the figure image; no numeric values appear in the text. The *claims*
  ("CIDD-27 spends comparable Vina calls and more LLM tokens yet yields very small improvements") are
  textual and unquantified.
- **SMILES validity** for PROBE: never reported.
- **Similarity-to-starting-ligand / novelty / diversity**: never reported anywhere in the paper.
- All baseline rows (MoLLM, LIDDIA, CIDD, and the 15 de novo generators) are taken as the authors report
  them; not independently reproduced. The MolJO 100-sample figures (Vina −9.05, QED 0.56, SA 0.78,
  Success 51.3%) are the authors' own re-run of the official checkpoint, per Appendix E.
