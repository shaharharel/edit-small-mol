# Experiment Queue — single source of truth (2026-05-13)

This document supersedes `results/anchordiff/24h_burn_plan.md` and the slash-task list.
Every experiment has a **Why** so we know months later what we wanted to learn.

---

## Hot-fix queue (must close before next training round)

| ID | What | Why | Owner | Status | Output |
|---|---|---|---|---|---|
| HF1 | `dataset.py` full-atom pocket encoding | The pre-trained DiffSBDD `crossdocked_fullatom_cond` ckpt expects every heavy atom of every pocket residue encoded by element. Prior code passed only Cα atoms encoded as 'C' — ~22 vs ~150 atoms, all collapsed to one element. The 30-epoch fine-tune was on malformed input. | local | ✅ committed | `anchordiff/covind/dataset.py` |
| HF2 | `MAX_RESOLUTION` 3.5→2.5 | At 3.5 Å, crystal SG→anchor distances are noisy by ±0.3 Å — same order as the geometric signal we train on. Tightening to 2.5 Å (paper recommends ≤2.0) trades 15% data loss for trustworthy bond geometry. | local | ✅ done | `anchordiff/covind/curate.py:MAX_RESOLUTION` |
| HF3 | Acrylamide labeling audit | QA agent suspected silent class-drop. Verified: CovInDB2 files acrylamides under "Michael Acceptor" (491 CYS-targeting entries). No bug, just confirmed. | local | ✅ resolved (no bug) | inline audit |
| HF4 | Expand `WARHEAD_VOCAB` 12→18 classes | The original 12 covered 84.9% of CYS-targeting CovInDB2 entries. Adding Sulfonic acid (56), Thiol (23), Ester (22), Diazomethyl Carbonyl (15), Thiosulfonate (10), Aziridine (3) captures the missing 230 entries (~15% more training data) without changing any of the original class semantics. | local | ✅ done | `covalent_token.py`, `curate.py` |
| HF5 | Add per-row mechanism axis to covalent token | Two Michael Acceptors with different substituents can react by different mechanisms (classical Michael vs aza-Michael vs Boronic Acid addition, etc.). The per-warhead-class hard-coded mechanism in `WARHEAD_GEOM` conflated these. Adding a 12-d mechanism one-hot from CovInDB2's `Reaction` column gives the model finer-grained reactivity context. | local | ✅ done | TOKEN_DIM 37→52 in `covalent_token.py` |
| HF6 | `WARHEAD_GEOM` parity between `curate.py` and `sample_dc.py` | `sample_dc.py` had its own (incomplete) `WARHEAD_GEOM` dict — missing classes silently fell back to Michael Acceptor at inference. Now sample_dc imports from curate. | local | ✅ done | `sample_dc.py` |
| HF7 | Inject `_struct_conn` covalent bond record into Boltz CIF | Boltz-2's CIF output omits cross-chain bonds from `_struct_conn` even when bond was specified in input YAML. Without it, 3Dmol.js viewer auto-perceives only intra-residue bonds — the Cys–ligand bond is invisible in the report. | local | ✅ done | `anchordiff/inject_struct_conn.py` |
| HF8 | Report viewer "Cys560" → "Cys346" UI cleanup | Report HTML/JS had Cys560 baked in as a hard-coded label. Cys560 was deprecated May 2026; Cys346 is the canonical target. | personal-clinician | ✅ done | `experiments/server/report.html` |
| HF9 | Rebuild `top1000_manifest__zap70_cys346.json` with **clean+KP FiLMDelta** + **mPAE** | Current `combined_score` uses leaky-split FiLMDelta and lig-iPTM. London's COValid paper showed mPAE > iPTM for enrichment; our 20-seed kinase-pretrain ensemble is the better FiLMDelta. Refresh the manifest after B5 finishes. | local | waiting on B5 | `data/boltz_poses/top1000_manifest__zap70_cys346.json` |

---

## A — Paper experiments (Edit Effect Framework, multi-target, 1.7M pairs)

| ID | What | Why | Status | Output |
|---|---|---|---|---|
| A1 | Phase 1 — embedder selection | Identify the best molecule encoder for FiLMDelta. Tests 6 embedders (Morgan, ChemProp, ChemBERTa, MoLFormer, CheMeleon, Uni-Mol). | ✅ Morgan FP wins on 1.7M pairs (MAE=0.631) | `all_results.json` |
| A2 | Phase 2 — architecture comparison | Test 8 architectures (FiLMDelta, EditDiff, DeepDelta, Subtraction, ...). Confirms whether explicit edit conditioning beats subtraction baseline. | ✅ FiLMDelta wins 7.8% over Subtraction | same |
| A3 | Phase 3 — generalization across 7 splits | Show the edit framework generalizes — including hard scaffold and assay-cross splits. | ✅ FiLMDelta wins on within-assay, EditDiff on cross-assay | same |
| A4 | Noise robustness | Demonstrate the framework's main practical advantage: within-assay pairs reduce noise vs cross-assay aggregation. | ✅ FiLMDelta wins 112/112 targets, avg +76.3% | `fair_noise_tiers_results.json` |
| A5 | Phase 4 — edit-aware architectures | Test richer edit representations (DRFP / DualStream / Hypernet) — does reaction fingerprint beat Morgan diff? | ✅ DualStream wins MAE=0.585 | `edit_iteration_results.json` |
| A6 | ActFound comparison | Compare against the published MAML+linear baseline (Nature MI 2024) — does our architecture beat the SOTA pairwise model? | Strategy A done; B needs ActFound ckpt | `actfound_comparison_results.json` |

---

## B — ZAP70 case study (19-mol REINVENT4 cohort)

| ID | What | Why | Status |
|---|---|---|---|
| B1 | Clean FiLMDelta retrain (mol-disjoint val) | The cached `reinvent4_film_model.pt` used pair-row-level val split, which leaked the target molecule across train/val. Need a true molecule-disjoint baseline to validate the wet-lab nomination. | ✅ done |
| B2 | 30-Trial Stability — clean | Confirm whether the 19-mol ranking holds under mol-disjoint val. If Spearman(leaky, clean) < 0.8, the leaderboard is invalid. | ✅ Spearman 0.981, ranking validated |
| B3 | 20-Seed Uncertainty — clean | Per-candidate pIC50 uncertainty bands under mol-disjoint val. | ✅ Spearman 0.954 vs leaky |
| B4 | 30-Trial Stability — clean **+ kinase pretrain** | Test whether 32K-kinase MMP pretrain stabilizes ranking. Pretrain may rescue rare-warhead chemistries and shift top mol — currently Mol-7 dominating in early seeds with KP, was Mol-18 without. | running locally |
| B5 | 20-Seed Uncertainty — clean **+ kinase pretrain** | Same as B4 but with the full mol-disjoint val (28 held-out). Will refresh `top1000_manifest` combined_score. | running locally |
| B6 | Simple-classifier baseline | Critical sanity check: does pairs training beat a trivial Ridge regression on Morgan FP? **Finding**: Ridge ρ=0.90 vs FiLMDelta on 19 mols — pairs training adds little for single-target small-N. Validates the paper's multi-target framing. | ✅ done |
| B7 | Stability comparison (leaky vs clean) | Documented headline + the Mol-1/Mol-4/Mol-7 rank flip. | ✅ md/csv |
| B8 | Stability comparison (clean vs clean+KP) | What does kinase pretrain change? Hypothesis: rescues mols whose warhead is rare in ZAP70 alone but common in the kinase corpus. | pending B4/B5 |

---

## C — Covalent generative track (C+D flagship)

| ID | What | Why | Status |
|---|---|---|---|
| C1-C3 | Day-1 baselines (vanilla / inpaint / constraint-projected) | Establish the 0% → 100% covalent-readiness curve and the geometric residuals for each strategy. | ✅ all 3 cohorts saved |
| C4 | C+D fine-tune v1 (Cα pocket — INVALID) | Was supposed to be the flagship; turned out to be on malformed input (HF1). Scrapped. | ❌ invalid |
| **C5** | **C+D fine-tune v2 on corrected dataset** | The real flagship: full-atom pocket (HF1) + tight resolution (HF2) + per-row mechanism (HF5) + 18-class vocab (HF4). 30 epochs. | running on T4 (will move to A100 once env ready) |
| C6 | C+D sampling with `--reference_ligand_sdf` | Inference-time pocket has to match training-time pocket size (22-58 residues, not 9). The reference ligand fix builds a correctly-sized pocket. | pending C5 |
| C7 | C-arm ablation: `--no_carm` (D-only) | Counterfactual for the covalent-token contribution. If C7 ≈ C5, the C-arm adapter (~380 params) adds nothing and we drop it. | pending C5 |
| C8 | Bias-context adapter ablations | The current 37-d (now 52-d) token has weak inductive priors. Test alternatives to identify which conditioning matters: (1) 16-d learned warhead embedding, (2) per-residue (distance, type) for top-K, (3) ESM2 mean-pooled pocket embedding, (4) deeper adapter MLP. | NOT STARTED — queued |
| C9 | TargetDiff baseline (out-of-the-box sampling) | Direct comparison: does our C+D on DiffSBDD beat the SOTA ICLR 2023 baseline on the same pockets, OUT OF THE BOX (no covalent fine-tune)? Tests whether our covalent inductive bias is meaningful at all. | env install in progress on A100/ai-gpu2 |
| C10 | TargetDiff + D+C fine-tune | More important than C9: graft our D-arm (local frame) + C-arm (covalent token adapter) onto TargetDiff. Does the D+C framework generalize beyond DiffSBDD's specific architecture? If yes, this is a portable framework, not a DiffSBDD-specific trick. | NOT STARTED — depends on C9 |
| C11 | Wet-lab handoff package | 24 scaffold-deduped + controls for in-house assay. Currently based on pre-fix data; needs refresh after C5. | needs refresh post-C5 |

---

## D — COValid evaluation (London JACS 2026)

Goal: assess our covalent stack on the first published covalent virtual-screening benchmark. The paper reports AF3 + mPAE at avg adj LogAUC 71.8% across 9 kinase targets; the SI mentions Boltz-2 + mPAE matches.

| ID | What | Why | Status |
|---|---|---|---|
| D0 | Download COValid SI XLSX | Cloudflare blocks scripted access — user has to download from JACS. | pending user |
| D1 | FiLMDelta per-target ranking on COValid | Test the **2D pairwise model** as a covalent enrichment ranker. Hypothesis: pairs activity signal alone gives ~20-40% adj LogAUC (better than docking, worse than mPAE). If yes, FiLMDelta is a cheap pre-filter before expensive cofold. | not started |
| D2 | Boltz-2 cofold on **top-100 actives × decoys per target** (~5K cofolds) | Confirm our Boltz-2 pipeline gives same enrichment as London's AF3+mPAE on the same compounds. If our numbers diverge significantly, our cofold setup has a bug. ai-gpu2 has the GPU; ~5 GPU-hours. | env install in progress |
| D3 | Combined FiLMDelta + mPAE ranking | Test the **orthogonality hypothesis**: 2D activity signal (FiLMDelta) is independent of 3D structural confidence (mPAE), so combining them beats either alone. If combined adj LogAUC > 75%, this is a publishable claim — beats London's pure-mPAE 71.8%. | pending D1/D2 |
| D4 | Generator-side recall on COValid actives | Different question: does our covalent **generator** know what active ligands look like? For each target Cys, generate N mols with C+D, measure Tc ≥ 0.4 recall against COValid actives. Baselines: REINVENT4, DiffSBDD vanilla, TargetDiff. | pending C5 + D0 |

---

## E — Data fixes / re-curations (must close before C5 finalizes)

| ID | What | Why | Status |
|---|---|---|---|
| E1 | Acrylamide labeling audit | See HF3. | ✅ no bug found |
| E2 | Tighten MAX_RESOLUTION | See HF2. | ✅ done |
| E3 | **Derive pre-reaction structures** | CovInDB2 stores post-bond covalent adducts. The pre-reaction (van-der-Waals, prereactive) geometry is what determines kinact/Ki (the kinetic step). Deriving pre-reaction structures by computationally breaking the SG-Cβ bond, restoring the Michael double bond, and translating outward 1.5-2 Å gives us the *reactive geometry* target — a more physically meaningful supervision signal. Major architectural change; queued as **C+D v3**. | LATER — not blocking C5 |
| E4 | Per-row Reaction mechanism in covalent token | See HF5. | ✅ done |
| E5 | Temporal hold-out by year > 2023 | Boltz-2 training cutoff is Sep 2023. To honestly evaluate "we beat London with our pipeline", we need to exclude post-Sep-2023 PDB structures from CovInDB2 train — otherwise Boltz could have seen the answer. Currently year is captured but not used for split. | pending — small CLI flag in `dataset.py` |
| E6 | Boltz YAML constraint with explicit bond length | All 3 Mol-1 cofold models gave unphysical SG-C distances (1.28-2.24 Å vs 1.81 expected). Boltz interprets `bond:` as "bond exists" without length enforcement. If Boltz-2 schema accepts `length:` in `bond:`, use it. Else escalate to upstream as a quirk. | open question |

---

## F — Compute routing (tonight, 2026-05-13)

| Machine | What's running | Owner |
|---|---|---|
| **local Mac (MPS)** | B4 30-trial pairwise + KP (~6h ETA) and B5 20-seed + KP (~3h ETA) | continues |
| **T4 (ai-gpu, us-east1-c)** | **C5 C+D v2 fine-tune** with full-atom + 18-class vocab + per-row mechanism + res≤2.5 — 30 epochs | NEW |
| **A100 (ai-gpu-a100, us-central1-b)** | `diffsbdd` conda env installing (~15min); next: **C9 TargetDiff sampling** on 5 covalent pockets | NEW |
| **ai-gpu2 (us-central1-a)** | `quris` env + Boltz install (~10min); next: **D2 Boltz cofold top-100 × decoys** for COValid alignment OR **HF9 top-200 cofold refresh** | NEW |
| **personal-clinician (us-central1-b)** | Flask report server | unchanged |

---

## G — QA agents (one per running experiment)

Each running experiment has a dedicated QA agent watching for:
- **Code bugs**: shape mismatches, off-by-one, wrong dimension order
- **Data bugs**: missing rows, wrong columns, NaN propagation, leakage between splits
- **Thinking bugs**: experimental design flaws (e.g., wrong baseline, wrong metric)
- **Runtime bugs**: silent failures, OOM, NaN losses, stuck dataloaders

| Experiment | QA agent | Status |
|---|---|---|
| B4 (30-trial stability + KP) | not yet dispatched | TODO |
| B5 (20-seed uncertainty + KP) | not yet dispatched | TODO |
| C5 (C+D v2 fine-tune) | not yet dispatched | TODO |
| C9 (TargetDiff baseline) | not yet dispatched | TODO |
| D2 (Boltz cofold backlog) | not yet dispatched | TODO |

QA dispatch happening next.

---

## H — Decision points for the user

1. **C10 TargetDiff D+C fine-tune scope**: graft full D+C (Cβ-anchored frame + cov adapter) or just the cov adapter on TargetDiff's existing data preprocessing? Full graft = ~3 days dev; adapter-only = ~1 day.
2. **D0 COValid SI download** — needed to start D1/D2/D3/D4.
3. **E5 temporal hold-out**: implement now (1 hour) or after C5 finishes (so we have a fair vs-Boltz-2 eval ready in parallel)?
4. **Should I move C5 from T4 → A100** once A100 env is ready, or run both T4 (current seed) and A100 (different seed) for ensemble?

