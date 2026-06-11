# RLHF Preference-Collection Demo — Design Spec

**Date:** 2026-06-11
**Branch:** `rlhf-system`
**Status:** Approved direction (user green-lit during brainstorming)

## Goal

The next phase of edit-small-mol is **RLHF from medicinal chemists** to train the system
for lead optimization. This spec covers the **first sub-project**: a simple, intuitive,
fast web interface for *collecting* pairwise preference data. A chemist logs in, is shown
two similar molecules from the same target program (one higher / one lower potency,
ideally from the same lab), can inspect each in 2D and in 3D (Boltz pocket pose), picks the
one they prefer, and immediately gets the next pair. Every judgment is attributed to a
named chemist.

This demo runs on the **ZAP70 (CHEMBL2803)** dataset. Once validated, the dataset for full
labeling is decided in a later sub-project (out of scope here).

## Non-Goals (YAGNI)

- Reward-model training / preference learning.
- Analytics or inter-annotator-agreement dashboards.
- Multi-target / multi-program support.
- Confidence scores or rationale capture — **binary choice only** (user decision).

## Data Realities (verified 2026-06-11)

- **Source molecules:** `data/overlapping_assays/molecule_pIC50_minimal.csv`, filtered to
  `target_chembl_id == 'CHEMBL2803'` → **280 molecules, 305 rows, 54 assays = 53 docs**
  (each ZAP70 assay maps 1:1 to one publication, so `assay_chembl_id` ≡ `doc_id` = "same lab").
- **Pair supply is abundant.** Same-lab, similar, one-higher/one-lower-pIC50 pairs:
  | Tanimoto | gap ≥0.5 | gap ≥1.0 |
  |---|---|---|
  | ≥0.6 | **755 pairs / 204 mols** | 342 / 157 |
  | ≥0.7 (MMP-like) | 314 / 159 | 123 / 100 |
- **Boltz poses already exist.** 253 of 280 ZAP70 anchor cofold poses were already computed
  on `ai-gpu-a100` and are now downloaded to `data/rlhf_demo/boltz_poses/<NN_CHEMBLID>/`
  (`*_model_0.cif` + `confidence_*.json`). This covers essentially all 204 pair-molecules.
  The remaining 27 can be folded later if a chosen pair needs one (A100, cached
  `~/zap70_msa.csv`). The 280 anchor YAMLs live at `experiments/boltz_inputs/zap70_anchors/`.

## Data Safety

**No existing data is moved or deleted.** All new artifacts live under the new
`data/rlhf_demo/` directory. Source CSVs and pose dirs are read only; pair/molecule data is
*duplicated* into `data/rlhf_demo/`.

## Architecture

A **new, standalone, lightweight Flask app** at `experiments/rlhf_server/`, deliberately
separate from the 756K-mol `experiments/server/backend.py`. It reuses two proven patterns
from that server: RDKit→SVG for 2D structures and **3Dmol.js + CIF** for the 3D viewer.

### Components

1. **Offline pair builder** — `experiments/rlhf_server/build_pairs.py`
   - Reads the ZAP70 slice of `molecule_pIC50_minimal.csv`.
   - Collapses to one pIC50 per (molecule, assay) = same-lab measurement.
   - Within each assay/doc cohort, forms candidate pairs where Tanimoto ≥ 0.6 (Morgan r2,
     2048-bit) and |Δ pIC50| ≥ 0.5. Prefers same-lab; falls back to same-target cross-lab
     only if needed (not needed for ZAP70).
   - Keeps only pairs where **both** molecules have a Boltz pose on disk.
   - Writes `data/rlhf_demo/molecules.json` (id → smiles, pIC50, assay, doc, pose path) and
     `data/rlhf_demo/pairs.json` (pair_id, mol_a_id, mol_b_id, tanimoto, delta, cohort).

2. **Backend** — `experiments/rlhf_server/app.py` (Flask + SQLite)
   - **Auth:** simple login — name + email → upsert `users` row → identity in Flask session
     cookie. Internal demo; lightweight, no password ceremony, every judgment attributed.
   - **DB** `data/rlhf_demo/rlhf.db`:
     - `users(id, name, email, created_at)`
     - `judgments(id, user_id, pair_id, mol_a_id, mol_b_id, chosen_id, shown_left_id, ts)`
       — `shown_left_id` records which molecule was rendered on the left, to audit/correct
       left/right position bias.
   - **Routes:** `/login`, `/logout`, `/` (comparison page), `/api/next_pair`
     (next not-yet-judged pair for this user, **A/B side randomized**), `/api/svg/<mol_id>`
     (RDKit SVG), `/api/pose/<mol_id>` (CIF text, or `{pending:true}` if no pose),
     `/api/submit`, `/api/progress`.

3. **Frontend** — `experiments/rlhf_server/templates/` + `static/`
   - One comparison screen: two molecule cards side by side (approved mockup, **no
     confidence row**). Click a card → it highlights → **Submit** advances with a smooth
     transition. **🔍 View 3D** opens a 3Dmol.js modal of the Boltz pocket pose.
   - Fast & fun: progress bar (`pair n/N`), keyboard shortcuts (← prefer A, → prefer B,
     `3` view 3D, Enter submit), running count, prefetch next pair's SVGs. Final visual
     polish via `frontend-design-skill`.

## Data Flow

login → server picks next unjudged pair for user (randomizes side) → client renders 2D
(SVG) + optional 3D (CIF) → chemist clicks preferred card → submit → row written to
`judgments` → next pair. Export: `judgments` joined with `pairs`/`molecules` → labeled
preference dataset for the later modeling sub-project.

## Build Approach

Implementation driven by a **multi-disciplinary agent team** (product/UX, frontend dev,
backend dev, and a scientist agent validating that selected pairs are genuinely same-lab,
similar, and have a meaningful potency gap), per the user's request.

## Success Criteria

- Chemist can log in, judge a stream of valid ZAP70 same-lab pairs, and inspect 2D + 3D.
- Each judgment is persisted with user attribution and side-shown for bias auditing.
- The pair set is real: same lab, Tanimoto ≥ 0.6, |Δ pIC50| ≥ 0.5, both poses present.
- No existing project data is moved or deleted.
