# CovalentFormer — HANDOFF (2026-09-16)

**Goal: a generative editor that follows a covalent-medchem steering instruction.**
Not "invent sophisticated covalent descriptors". Every param below is judged against
"would a medicinal chemist issue this instruction during hit-to-lead, and does the model
obey it in GENERATED molecules".

Supersedes `HANDOFF_prev.md` (2026-09-15).

---

## 1. STATUS PER STEERING PARAM

| param | train rows | verdict | evidence |
|---|---|---|---|
| **role** (WARHEAD/LINKER/SCAFFOLD/DECORATION) | large | **WORKS** | 82.0%/77.1% obedience on a contradicted token vs ~27% ignore-floor. **NOT yet replicated on the clean v2 corpus — do this FIRST, it is the anchor result** |
| **acyl_N_motif** | 180,000 | **WORKS** | BENEFIT +0.0077 (MPS) / +0.0068 (V100), GAP_perm +0.006, GAP_flip +0.005. Replicated on two devices. Labels NOT affected by the ring bug |
| **linker_atom_count** | 113,181 | **MUST RE-EARN** | +0.0103/+0.0111 was real but measured on BROKEN labels (§3). Relabelled; retrain pending |
| **warhead_planarity** | 216,550 | **NULL twice, CAUSE DIAGNOSED** | +0.0026 vs ~0.002 noise floor even on large-effect-only data. Cause is NOT saturation (§4) |
| **theta_bd** | 2,296 | undecided | survived a pre-registered falsifier; arm underpowered. v7 running |
| **buried_sasa** | 1,900 | undecided | 100% coverage, best-covered pocket param. v7 running |
| **pocket_occupancy** | 1,374 | undecided | noisiest denominator (cavity volume). v7 running |
| **d_cys_scaffold** | 1,251 | undecided | ring bug fixed, labels 4,516 → 1,469. v7 running |
| **polar_contacts** | ~5,700 | BUILT, NEVER TRAINED | 100% coverage, 69% moved, balance 0.77 |
| **buried_sasa_per_ha** | ~5,700 | BUILT, NEVER TRAINED | 100% coverage, 80% moved, size-normalised burial |

**DEAD, do not rebuild:** `warhead_span`, `warhead_linker_flex`, `d_cys` (electrophile→Cys IS
the formed bond, 1.797 A sd 0.106), `extent`/warhead_reach_3d, `michael_subst_class`
(99.3% of mass in two values — keep as a pair PRECONDITION only).

**Target is role + 2-3 ligand + 1-2 complex. Currently 1 proven, 1 solid, 1 recoverable,
1 diagnosed. THE GAP IS ON THE COMPLEX SIDE.**

---

## 2. WHAT THE TWO WORKING PARAMS MEAN

- **`linker_atom_count`** — atoms strictly between the warhead's electrophilic carbon and the
  first Murcko scaffold ring, walking TOWARD the scaffold. "How far the electrophile is held
  off the recognition core." Instruction is UP/DOWN, **direction only, magnitude unspecified**.
- **`acyl_N_motif`** — 9 classes for what hangs off the warhead's amide N. In one sentence:
  *take the N-H dangling off the warhead and tuck that nitrogen into a ring* — which stiffens
  the vector the warhead points along and makes the acrylamide more electrophilic.

---

## 3. THE BUG THAT INVALIDATES `linker_atom_count`'s NUMBER

A CYCLIC warhead **is** a ring, so the BFS from the electrophile terminated on the warhead's
own ring at distance 0. Measured per warhead class on the v2 labels:

    beta_lactam   814/814 = 100.0% zero, ONE distinct value
    epoxide       266/266 = 100.0% zero, ONE distinct value
    nitrile_act.  59.2% zero   ketoamide 52.6%   sulfonyl_fluoride 74.6%
    POOLED 25.6% of the corpus pinned at zero

1,140 molecules carried a CONSTANT label; those rows can only produce ties.

**FIXED** in `label_molecule_params.linker_atom_count` — excludes the ring system containing
or fused to the electrophile. Verified on 3,000 molecules: beta_lactam 100% → **0.0%** zero
with 7 distinct values; every class 0.0%. Full relabel **DONE** →
`data/labels/molecule_params_v3.csv`.

**HOW IT WAS MISSED:** QA printed POOLED distributions ("15 levels, 84% in {2,3,0}") which
looks healthy. The degeneracy exists only PER WARHEAD CLASS. **Always split the histogram by
class.** Same defect was found and fixed in `d_cys_scaffold` an hour earlier and not carried
across.

---

## 4. WHY PLANARITY DOES NOT STEER (diagnosed; fix designed, NOT run)

Headroom is NOT the problem — corpus planar_dev median 24.89 deg, p95 66.0, **41.9% over 30
deg**, only 21.4% under 5.

**UP and DOWN are applied to disjoint populations:**

    UP   (-> more twisted)  n=65,729   anchor planarity mean 13.90  median  8.03
    DOWN (-> more planar)   n=60,001   anchor planarity mean 44.02  median 36.90

An 8-deg anchor can only go UP; a 37-deg anchor can only go DOWN. **The token is REDUNDANT
given the anchor, not ignored.** That is why GAP ~ 0.

**FIX: anchor-stratified pairs** — within each bin of anchor planarity include both UP and
DOWN, so the token carries information the anchor does not.

---

## 5. RUNNING AT HANDOFF TIME

- **ai-gpu (34.57.253.170)** — `stage0/run_v7.sh`, 7/16 arms done, ETA ~22 min.
  Two rescues per pocket param: **matched warm-start** (BOTH arms init from the trained linker
  arm — v6 warm-started only `instr`, which is why it must be re-run) and **oversampling**
  (12 ep, bs16, lr5e-5 — is the null undertraining?). Results land in `ckpt_v7/*_history.json`.
- **ai-gpu2 (34.28.139.166)** — being provisioned, then planarity m25 arms → `ckpt_g2/`.
  Repo did not exist there; `xxhash` was missing (same as ai-gpu).
- **local** — idle; relabel finished.

**GPU IPs ARE EPHEMERAL AND `~/.ssh/config` IS STALE.** Get them from
`gcloud compute instances list`; connect by IP with `-i ~/.ssh/google_compute_engine`.

---

## 6. WITHDRAWN TODAY — DO NOT QUOTE

- **v6 pocket "BENEFIT +0.03..+0.08"** — the launch script warm-started ONLY the `instr` arm,
  so BENEFIT measured the WARM START. GAP_perm was ~0 in all four arms, i.e. the instruction
  was ignored. Unmatched control; best-looking worthless number of the run.
- **"generated MW 261 vs corpus 482, model is truncating"** — false twice over. The valid CSV
  is in MW-ASCENDING corpus order and `--n 60` took the 60 SMALLEST rows. Sampled correctly,
  train/valid/corpus are all ~480-490 Da. **There is no MW bug.**
- **"34/37 ties at generation"** — same unrepresentative head sample.

---

## 7. TRAPS (each cost real time today)

- **A full disk reports as file corruption.** ai-gpu hit 100% of 291 GB; torch.save died with
  `unexpected pos 128 vs 0`, killing 13 of 16 arms. Trainer now saves BEST-ONLY. Box is still
  at 98% and the 283 GB has NOT been traced.
- **`ssh host "job & sleep 50; check"` hangs the ssh session** and dies on the tool timeout
  even though the job launched. Fire-and-forget, poll separately.
- **zsh does not word-split unquoted expansions** — `set -- $pair` gives `$1`=whole string.
- **`scp host:dir/*.json` is glob-expanded LOCALLY by zsh** — quote the remote path.
- **REMARK 2:** `line.split()` grabs the `2` of "REMARK   2" before the resolution. Every
  structure came back 2.0 A. No real PDB set has one resolution — that is the tell.
- **A leak check does not check usability.** The first molecule-disjoint split asserted ZERO
  leak with **248 train rows against 575,777 valid**. Both assertions now exist.
- **A molecule-disjoint split is IMPOSSIBLE here** — all-vs-all puts 99.9% of pairs in ONE
  connected component. Scaffold-disjoint is used and is stricter.
- **Decoder:** EOS `$`=2, PAD `*`=0, BOS `^`=1. `vocab.decode` returns a token LIST; calling
  `.replace()` raised and a bare except made every molecule None — reported as 0.0% validity.
- **VALID CSVs ARE IN MW-ASCENDING ORDER.** Reshuffle them. This produced three separate
  unrepresentative-sample errors today.

---

## 8. WHAT THE METRICS MEAN

- **GAP_perm** — same model, instructions permuted within batch. DEPENDENCE: does the model
  use the token? The arm's own null. Noise floor ~0.002.
- **GAP_flip** — same model, instruction replaced by its opposite. DIRECTIONALITY.
  **INVALID WHEN `SAME` DOMINATES** — flip leaves SAME rows unchanged, so at 65% SAME
  (d_cys_scaffold) and 66% (pocket_occupancy) GAP_flip collapses onto GAP_perm. They came back
  equal to 4 dp, which exposed it. v6 caps SAME at 20% of train AND valid.
- **BENEFIT** — `none`-arm valid loss minus `instr`-arm valid loss, both at their own best
  epoch on the same rows. Between models. **IT IS A PROXY AND IT MISLED HERE**: linker showed
  +0.0103 BENEFIT while its cohorts tied. Loss asks "does the token help predict the reference
  product", never "do UP and DOWN produce different molecules".
- **OBEDIENCE (generation)** — fraction of anchors where param(B_up) > param(B_down).
  **NULL IS 0.5, NOT 0.** This is the metric that decides, and it is barely exercised.

---

## 9. ORDER OF OPERATIONS FOR THE NEXT SESSION

1. **Rebuild instructions + splits from `molecule_params_v3.csv`** and **RESHUFFLE every valid
   CSV** (§7).
2. **Replicate ROLE steering on the clean corpus** — the anchor result and the only proven one.
   `build_roles.py` + `train_phaseA.py --mode role`.
3. **Retrain `linker_atom_count`** on corrected labels.
4. **Read v7** — does any pocket param show GAP > noise under a MATCHED control?
5. **Build anchor-stratified planarity** (§4) and retrain.
6. **Train `polar_contacts` + `buried_sasa_per_ha`** — built, 100% coverage, never trained.
7. **5k cohorts** for survivors: obedience vs 0.5, tie rate, **steered-vs-unsteered cohort
   shift** (the comparison that answers "did steering help"), plus validity/uniqueness/
   novelty/QED and the manuscript panel. Also **base-mol2mol vs covalent-FT vs steered** on
   planarity — never run, and it is the comparison that matches the manuscript claim.
8. **FULL EVALUATION SWEEP — `bash stage0/run_full_eval.sh`** (PY=... N=5000). One command,
   one table for every steering param. It generates 5k cohorts per arm and scores:
     - COHORT SHIFT vs the unsteered `none` arm  <- THE PRODUCT CLAIM
     - COHORT SHIFT vs the raw mol2mol PRIOR      <- separates steering from covalent FT
     - generation panel (validity / uniqueness / novelty / QED)
     - manuscript panel (warhead retention, planarity)
   THE PRIOR COHORT IS GENERATED ONCE and reused for every param: it does not depend on
   which param is steered, so a per-param baseline would be 8x the compute for identical
   numbers AND would make cross-param comparison a SEED contrast rather than a PARAM one.
   ROLE RUNS FIRST because it is the only param with a known answer (82.0%/77.1% vs a ~27%
   floor). A harness that cannot reproduce role is not trustworthy on anything else.
   Summarise with `python3 stage0/summarise_eval.py results/full_eval`.
   READ IT AS: `UP_vs_none` / `DOWN_vs_none` are the product claim. `UP_vs_DOWN` is easier
   to move and is NOT the same statement -- a model can separate its own two instructions
   while neither cohort differs from an unsteered baseline. `none_vs_prior` tells you how
   much is the covalent fine-tuning rather than the instruction.
9. Only then: joint / alternating multi-instruction training.

## 9d. PLANARITY FIX IS BUILT AND READY TO TRAIN

`data/steer_v6_strat/` — 17,403 train / 2,813 valid, ANCHOR-STRATIFIED over 8 bins
(`build_steer_v6.py --anchor-stratify warhead_planarity`). Within each bin of ANCHOR
planarity, UP and DOWN are equalised, so the direction is no longer inferable from the input
and the token must be read. Cost: 221,631 one-sided rows dropped to keep 25,198 balanced --
and that cost IS the fix, since those rows were teaching the model to ignore the instruction.

CAVEAT TO STATE UP FRONT: 17k rows is an order of magnitude below the 216k that produced the
null, so a failure here is ambiguous between "the token still does not help" and "too few
rows". Decide which by comparing GAP against the m25 arm at MATCHED row count.

---

## 9b. ROLE STEERING — WEIGHTS EXIST, TWO GOTCHAS

Weights are `ckpt_A_role/`, `ckpt_A_role_strat/`, `ckpt_A_role_conv/` (13 epochs) plus the
MATCHED CONTROL `ckpt_A0_strat/`. Role is the only param that already has its unconditioned
control trained.

  * THE FILES ARE `.ckpt`, NOT `.pt`. A `ls ckpt_A_role/*.pt` glob returns nothing and reads
    like the weights are gone. They are not.
  * THE KEY IS `model_state`, NOT `network_state`, and the conditioning table is `role_emb`
    (PhaseA), not `instr_emb` (SteerNet). Both loaders use strict=False, so feeding one to
    the other SUCCEEDS SILENTLY with a RANDOM embedding -- valid molecules, meaningless
    numbers. `cohort_shift.load()` now detects the type from the weights and ASSERTS the
    conditioning tensor came from the file.

  ROLE COHORT SHIFT MUST BE REPORTED PER ROLE. train_strat is
  WARHEAD 149,953 / DECORATION 73,967 / SCAFFOLD 35,993 / LINKER 3,335 (45x spread) and
  valid_strat is deliberately REBALANCED (LINKER ~10x enriched). Any pooled role number is a
  MIX ARTIFACT. Data: `data/roles/train_strat.csv` 263,248 rows, `valid_strat.csv` 3,169.

## 9c. FIRST COHORT-SHIFT RESULT (linker_atom_count, 120 anchors, OLD broken labels)

    cohort        mean   median
    instr_UP     3.635    3.0
    none         3.135    3.0     <- unsteered baseline
    instr_DOWN   2.711    2.0

    UP vs DOWN    mean +0.92   KS p 6.2e-13
    UP vs none    mean +0.50   KS p 0.0008
    DOWN vs none  mean -0.42   KS p 0.003

The steered cohorts BRACKET the baseline in both directions, near-symmetrically. This is the
strongest steering evidence in the project and it was produced on the BROKEN labels, so it
should improve after the relabel. Redo at n=5000 on `molecule_params_v3.csv`.

## 10. KEY FILES

    stage0/covalent_filter.py        warhead panel, 9/9 on named drugs
    stage0/build_covalent_union.py   BOTH branches gated (CovInDB was unfiltered: 50.8% dropped)
    stage0/label_molecule_params.py  ligand params — RING FIX HERE
    stage0/label_pocket_pairs.py     pocket params — RING FIX HERE TOO
    stage0/label_pocket_extra.py     polar_contacts, buried_sasa_per_ha
    stage0/build_steer_v5.py         scaffold-disjoint + random splits
    stage0/build_steer_v6.py         large-effect-only, SAME capped at 20%
    stage0/train_steer_v5.py         one arm per param, fine-tunes the mol2mol prior
    stage0/generate_and_score.py     cohort generation + both scoring panels
    stage0/theta_bd_falsifier.py     the pre-registered test theta_bd survived
    stage0/cohort_shift.py           COHORT-LEVEL eval: steered vs unsteered distributions,
                                     KS test, per-role mode, conditioning-load guard
    stage0/run_v7.sh                 matched warm-start + oversample pocket rescue
    data/covalent_final_v2/          1,101,089 pairs / 19,538 molecules, 0.00% contaminant
    data/labels/molecule_params_v3.csv   RELABELLED — USE THIS ONE

---

## 11. SETTLED NUMBERS

- Corpus v2: **1,101,089 pairs / 19,538 molecules**; contaminant match **37.69% → 0.00%**;
  terminal acrylamide **15.16% → 43.22%**; MW median 482.
- theta_bd survives its falsifier: sd **10.379 deg at <=1.8 A** vs 11.228 pooled, FLAT across
  four resolution bins. Not restraint slop.
- Pocket instruction yields (5,689 same-protein pairs): theta_bd 67.6% moved / balance 0.98 ·
  buried_sasa 66.5% / 0.94 · pocket_occupancy 37.7% / 0.96 · d_cys_scaffold 31.1% / 0.64.
- **280 proteins**, but **P0DTD1 is 38.5% of pairs**; 45 have >=20 pairs, 8 have >=100.
  (Bears on any per-protein or MAML plan.)
- Generation panel (linker arm, small sample): validity 96.7%, uniqueness 94.0%,
  novelty 94.6%, QED 0.758, warhead retention 75%.
