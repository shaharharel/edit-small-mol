# RLHF Preference-Collection Demo (ZAP70)

A lightweight web tool for collecting medicinal-chemist pairwise preferences to drive
RLHF for lead optimization. A chemist logs in, is shown two similar **same-lab** ZAP70
(CHEMBL2803) inhibitors — one more potent, one less — inspects each in 2D and in 3D
(Boltz cofold pose), and picks the one they would advance. Potency is **hidden** during
the choice so we capture genuine medchem judgment, not number-reading.

Spec: `docs/superpowers/specs/2026-06-11-rlhf-preference-collection-design.md`

## Run

```bash
# 1. (one-time / to rebuild the pair set) build the curated pairs + molecule index
conda run -n quris python experiments/rlhf_server/build_pairs.py

# 2. start the server
conda run -n quris python experiments/rlhf_server/app.py
# open http://localhost:5055
```

## What the data is

- **216 pairs / 135 molecules / 11 same-lab cohorts** (current build).
- Each pair: same `assay_chembl_id` (= same publication/lab for ZAP70), Tanimoto ≥ 0.6
  (Morgan r2), |Δ pIC50| ≥ 0.5, both molecules with a Boltz pose passing the confidence
  floor (ligand-ipTM ≥ 0.80 **and** complex-pLDDT ≥ 0.78). Stereo-only near-duplicate
  pairs (Tc ≈ 1.0) are dropped; each cohort is capped at 40 pairs so no series dominates.
- Filters and thresholds are constants at the top of `build_pairs.py`.

## Files

| File | Role |
|---|---|
| `build_pairs.py` | Offline: source CSV + Boltz poses → `data/rlhf_demo/{molecules,pairs}.json` |
| `app.py` | Flask server: login, pair serving, 2D SVG, 3D pose (CIF), judgment recording (SQLite) |
| `templates/login.html` | Sign-in (name + email; every judgment is attributed) |
| `templates/compare.html` | Side-by-side comparison screen + 3D modal |
| `static/style.css` | Chemistry-journal theme |
| `static/app.js` | Comparison flow, keyboard shortcuts, 3Dmol.js viewer |

## Data locations (nothing existing is moved or deleted)

- Source potency/lab data (read-only): `data/overlapping_assays/molecule_pIC50_minimal.csv`
- Boltz poses (downloaded from A100): `data/rlhf_demo/boltz_poses/<NN_CHEMBLID>/*_model_0.cif`
- Generated pair set: `data/rlhf_demo/molecules.json`, `data/rlhf_demo/pairs.json`
- Collected labels: `data/rlhf_demo/rlhf.db` (SQLite; `users`, `judgments`)

## Exporting collected preferences

```bash
sqlite3 data/rlhf_demo/rlhf.db \
  "SELECT u.name, j.pair_id, j.mol_high_id, j.mol_low_id, j.chosen_id, j.shown_left_id, j.action, j.ts \
   FROM judgments j JOIN users u ON u.id=j.user_id;"
```

`chosen_id` is the molecule the chemist would advance (NULL for skip). `shown_left_id`
records which molecule was on the left, to audit/correct left-right position bias. Join
`mol_high_id`/`mol_low_id` against `pairs.json` to recover the held-out pIC50 delta and
compute a chemist's potency-agreement rate.

## Keyboard

`←` pick A · `→` pick B · `3` view 3D · `Enter` confirm · `S` skip
