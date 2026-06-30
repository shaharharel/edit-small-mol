# Data + experiment inventory

Existing assets (do not delete) that map to results sections.

## Core generated/scored cohorts

| Asset | Path | Size | Use in paper |
|---|---|---|---|
| Full F4 cohort scored | `data/tier4_scored/F4_boltz_full.csv` | 2,221 × 106 | §3.1 yield-per-tier, §3.2 ablation, §3.4 distilled-validator training pool |
| 838 visible survivor SMILES | `data/tier4_scored/visible_838_smiles.json` | 838 canonical SMILES | §3.1 funnel endpoint of main pipeline |
| Rescue-78 (stricter Tc/shape, no Vina gate) | `data/tier4_scored/rescue_78_full.csv` | 78 × 107 | §3.5 final candidate set (combined with main into 916) |
| PrexSyn synthesizability for 916 | `data/tier4_scored/prexsyn_916.csv` | 916 × 11 | §3.5 synthesizability column for shortlist |
| Multi-tier filter dashboard | `experiments/server/report_light.html` + slim backend | Live at `https://zap70.pk-labs.net/light` | §2.12 multi-disciplinary triage interface; §3.5 final candidate selection |

## RL training artifacts (11 cohorts)

| Cohort | TOML config | Checkpoint | Backbone family | Target panel | Base prior |
|---|---|---|---|---|---|
| thiq_rl_exp2_zap70 | `experiments/thiq_rl_tomls/thiq_rl_exp2_zap70.toml` | `models/rl_checkpoints_b/thiq_rl_exp2_zap70_stage1.chkpt` | THIQ-acryl reward | ZAP70 | covalent_ft (EXP2) |
| thiq_rl_exp2_kinase | `experiments/thiq_rl_tomls/thiq_rl_exp2_kinase.toml` | local backup | THIQ-acryl reward | kinase-broad | covalent_ft (EXP2) |
| thiq_rl_exp2_mol1only | `experiments/thiq_rl_tomls/thiq_rl_exp2_mol1only.toml` | local backup | THIQ-acryl reward | Mol1-only | covalent_ft (EXP2) |
| thiq_rl_zap70 | `experiments/thiq_rl_tomls/thiq_rl_zap70.toml` | local backup | THIQ-acryl reward | ZAP70 | warhead_tokens |
| thiq_rl_kinase | `experiments/thiq_rl_tomls/thiq_rl_kinase.toml` | local backup | THIQ-acryl reward | kinase-broad | warhead_tokens |
| thiq_rl_mol1only | `experiments/thiq_rl_tomls/thiq_rl_mol1only.toml` | local backup | THIQ-acryl reward | Mol1-only | warhead_tokens |
| murcko_rl_exp2_zap70 | `experiments/murcko_rl_tomls/murcko_rl_exp2_zap70.toml` | `models/rl_checkpoints_b/murcko_rl_exp2_zap70_stage1.chkpt` | Murcko-preserving | ZAP70 | covalent_ft (EXP2) |
| murcko_rl_exp2_kinase | local | local | Murcko | kinase-broad | covalent_ft (EXP2) |
| murcko_rl_zap70 | local | local | Murcko | ZAP70 | warhead_tokens |
| murcko_rl_kinase | local | local | Murcko | kinase-broad | warhead_tokens |
| mol1RL_v5 | older | older | Mol1-anchored | Mol1-only | various |

Sampled output per cohort: ~50K mols × 11 cohorts = ~1.18M scored generated molecules total.

## RL seed files

| Seed | Path | n | Composition |
|---|---|---|---|
| ZAP70 + Mol1, cleaned | `data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi` | 220 | 280 ChEMBL ZAP70 (90% non-covalent, 10% acrylamide) + Mol1, deduped |
| Mol1 only | `data/mol1_rl_seeds/seed_mol1_only.smi` | 1 | Mol1 canonical SMILES |
| Kinase panel ×5 + Mol1 ×20 | `data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20.smi` | ~thousands | Broad kinase panel + ZAP70 oversampled + Mol1 oversampled |
| Mol1-heavy | `data/mol1_rl_seeds/seed_zap70_mol1heavy.smi` | varies | Mol1-anchored variants |

Source for ChEMBL ZAP70: `data/docking_chembl_zap70/docking_results.csv` (280 mols, pIC50_exp range 2.7–9.0).

## Boltz-2 cofold artifacts

| Asset | Path | Size | Use |
|---|---|---|---|
| Cofold pool v2 with metrics | `data/tier4_scored/F4_boltz_pool_v2_with_boltz.csv` | 1,546 × 92 | Per-mol iptm/mPAE/plddt/etc. |
| Extra cofold batch | `data/tier4_scored/F4_boltz_extra_v2_with_boltz.csv` | 675 × 128 | Extension cohort cofolds |
| 838 visible CIFs bundle | `data/tier4_scored/visible838_cifs_v2.json.gz` | 99 MB, 838 entries | Serves 3D pose modal in /light report |
| Rescue-78 CIFs | `data/boltz_rescue_78/R0000..R0077/` | 78 dirs | Rescue cohort cofolds (3D pose) |
| Combined Boltz metrics on F4 | merged into `F4_boltz_full.csv` cols | — | §3.4 distilled validator training |

## Distilled validator (new — 2026-06-23)

| Asset | Path | Notes |
|---|---|---|
| Trained model bundle | `models/boltz_distilled/boltz_distilled_v1.pkl` | XGBoost regressors for boltz_iptm + mPAE_london, ECFP4 → metric. CV r = 0.671 (iptm), 0.692 (mPAE). 2,221 training mols. |
| Training script (inline above) | embedded in `paper/manuscript/plan/03_data_inventory.md` history | Will be moved to `experiments/train_distilled_validator.py` and the model loaded in REST server for RL deployment. |

## Computed downstream columns (used in expert triage, §3.5)

* **Drug-likeness**: MW, LogP, TPSA, HBD, HBA, RotBonds, fsp3, QED, SAScore
* **Ligand efficiency**: LE, LLE, BEI
* **Tox alerts** (29-pattern): tox_alerts_count, tox_alert_names; strict-count variant excluding 8 kinase-precedented patterns (anilinopyridine_2/3/4, primary/secondary aniline, aniline_pyrimidine, aniline_pyridine_link, cyclic_sulfamide) — derived client-side in the /light report
* **Covalent productivity proxy**: k_inact_proxy = thiolate_fraction × exp(-((d_SG-1.8)² + (BD/10)²)) — known formula artifact in BD term, used as soft gate only
* **MM-GBSA single-frame**: dG_recognition_md_kcalmol on Boltz pose (capped-analog protocol)
* **PROPKA3**: pKa_Cys346 per pose
* **RDKit strain**: rdkit_strain_kcal_mol (Boltz pose), rdkit_strain_posefree_kcal_mol (free-state)
* **Docking rescore**: vanilla_vina_kcalmol, adcov_local_kcalmol, covvina_rescore_*
* **Kinase classifier**: P_kinase, P_Tec_family (3-head FFN trained on 232K mols, scaffold-split AUROC 0.98)
* **Pharmacophore tags**: mol1_murcko_smarts_match, thiq_core, acryl_match
* **Charge state**: net_charge_pH74 (Dimorphite-DL), frac_charge_pH74 (MolGpKa fractional), net_charge_pH74_mg, pKa_basic_max
* **Pose geometry**: d_SG, burgi_dunitz_dev_deg, warhead_dev_deg (ligand-only — empirically uncorrelated with pose metrics, will be discussed as negative control)
* **Cofold contacts**: n_h_bonds, n_stabilizing_contacts (strict PLIP), n_contacts_total, n_salt_bridges, atp_pocket_fraction
* **Reactivity proxies**: pred_log_k2_GSH

## Mol1 reference data

* Mol1 SMILES + measured pIC50 = 6.59 (in-house ZAP70 assay)
* Mol1 Boltz cofold pose: locally generated, served in /light modal
* Mol1 full feature row: `results/paper_evaluation/mol1_full_features.json`

## Expert triage records

* 3 specialist reviews of the final 47 (covalent kinase, kinase SAR, DMPK) generated by scientific-analyst agent, saved in conversation transcript (extractable from chat log). Will be reorganized into a structured rubric file `paper/manuscript/supp/expert_rubric.md` with top picks (T13, M5) and rejection rationale.

## Code + reproducibility surface

* REINVENT4 mol2mol configs: `experiments/{thiq,murcko}_rl_tomls/`
* FiLMDelta scoring server: `experiments/reinvent4_film_rest_server.py`
* Multi-tier filter + report: `experiments/server/backend_light.py`, `experiments/server/report_light.html`
* Boltz-2 cofold runner: `experiments/compute_full_boltz_metrics.py`, `experiments/compute_pose_extras_for_f4.py`
* Tox + k_inact: `experiments/compute_tox_and_kinact.py`
* PrexSyn: external V100 deployment (separate repo)
* Distilled validator: `models/boltz_distilled/boltz_distilled_v1.pkl` + training script (to be added)

All RL TOMLs document hyperparameters (sigma, learning rate, max_steps, reward weights, transform parameters).

## Wet-lab returns (placeholder, populated as data arrives)

* `paper/manuscript/results/wetlab/` — directory will hold collaborator IC50 data, occupancy traces, kinact/Ki fits, kinome-scan tables. To be created when data lands.
