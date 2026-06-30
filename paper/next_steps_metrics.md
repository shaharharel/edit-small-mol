# Next-Steps Metrics — Expert Panel Recommendations

After the 4-panel review (covalent SAR / DMPK / Tox / Kinome) of the 58-mol
top-cohort, these are the metrics the panels identified as missing. Priority 1
(tox SMARTS + k_inact proxy) is now shipped — this file tracks the remaining
work, sorted by info/$ ratio.

---

## Priority 2 — DMPK gates (fast, free, would block half the impermeable kills)

| Metric | Tool | Cost | Status |
|---|---|---|---|
| **cLogD₇.₄** | SwissADME (free batch) OR RDKit Crippen LogP + Henderson-Hasselbalch | seconds/mol, free | TODO |
| **Caco-2 Papp** | ADMETlab2 (free web/API) OR train QSAR on Wang 2016 Caco-2 dataset | seconds/mol, free | TODO |
| **hERG QSAR** | Pred-hERG (free) / ADMETlab2 / BMC AILab | seconds/mol, free | TODO |
| **CYP3A4 / 2D6 / 2C9 inhibition** | ADMETlab2 | seconds/mol, free | TODO |
| **Plasma protein binding** | ADMETlab2 + LogP regression | seconds/mol, free | TODO |
| **Aqueous solubility (FaSSIF)** | ADMETlab2, ESOL (RDKit) | seconds/mol, free | TODO |

**Decision required**: which tool stack (ADMETlab2 only / SwissADME / mix)?

---

## Priority 3 — Covalent productivity refinement (already have proxy; refine)

| Metric | Tool | Cost | Status |
|---|---|---|---|
| **DFT LUMO of acrylamide** | B3LYP/6-31G* on Mol1 warhead | 5 min CPU; identical warhead → 1 calc total | OMITTED (constant across cohort) |
| **Reversible K_I prediction** | Retrain FiLMDelta without covalent label | ~1 day V100 | TODO |
| **Bürgi-Dunitz angle from Boltz pose** | already in `burgi_dunitz_dev_deg` column | done | ✅ |

---

## Priority 4 — Selectivity (single highest-info experiment is Lck)

| Metric | Tool | Cost | Status |
|---|---|---|---|
| **Lck FiLMDelta IC50 prediction** | Retrain FiLMDelta on ChEMBL Lck (~3,000 mols) | ~1 day V100 | TODO |
| **BTK / ITK FiLMDelta** | Same recipe | ~1 day V100 each | TODO |
| **Off-Cys covalent docking** (EGFR-T790M / JAK3 / HER2) | DOCKovalent or CovDock (commercial) | hours/mol, GPU | TODO |
| **Predicted KINOMEscan vector** | KinHub model on 442 kinases | hours, V100 | TODO |

**Wet-lab equivalent**: Lck + BTK + JAK3 + EGFR-T790M biochemical IC50 panel on top 10 = $4K / 1 week. **Single highest selectivity-information experiment** per kinome panel review.

---

## Priority 5 — Pose ensemble / model overconfidence

| Metric | Tool | Cost | Status |
|---|---|---|---|
| **3-replica Boltz pose ensemble** | Re-run Boltz seeds 0,1,2 on top 10 | 2.5 hr A100 | TODO |
| **100ns covalent-adduct MD** | OpenMM | ~6 hr/mol A100 on top 5 | TODO |
| **Photo-tox UV-Vis (TD-DFT)** | Gaussian / ORCA on chromophore-extended candidates only | hours/mol | TODO |

---

## Priority 6 — Lhasa commercial gates (premium)

| Metric | Tool | Cost |
|---|---|---|
| Derek Nexus expert-system tox alerts | Lhasa Derek | ~$25K/year academic |
| Sarah Nexus Ames QSAR | Lhasa Sarah | bundled |

Free alternatives already in plan:
- VEGA HCB (Italian Min. Health) — Ames CAESAR, carcinogenicity, mutagenicity
- EPA CompTox / OPERA — broad QSAR endpoints
- eMolTox (Wuhan U) — 13 tox endpoints batch

---

## Cohort filter recommendations (next RL round)

Apply to RL reward at sampling time:

1. **Hard mask** any molecule with `tox_alerts_count ≥ 1` (extended SMARTS catalog already in `experiments/compute_tox_and_kinact.py`)
2. **Hard mask** `warhead_dev_deg > 20` (Boltz pose failure mode — bimodal distribution at 0° vs 28°)
3. **Hard mask** `pKa_Cys346 > 9.5` (Cys protonated → no covalent reaction; SAR panel rec)
4. **Lead-like MW cap**: target MW ≤ 380 (Mol1 + 56 Da budget). The current 58-pool sits at MW 405 median which exhausts late-stage optimization room.
5. **Charge target**: ideal `net_charge_pH74_mg = 0`; allow +1 only if (TPSA < 90 AND HBD ≤ 2)
6. **HBD ≤ 2** for cytoplasmic kinase permeability
7. **Sort tie-breaker**: `k_inact_proxy` (covalent productivity) over `desirability_score` (reversible affinity proxy)

---

## Program-level concerns (not per-molecule)

- **2-amino-N-alkyl-imidazole scaffold** is in Pfizer's Stepan 2011 reactive-metabolite database (CYP1A2 bioactivation hotspot). Every cohort molecule inherits this from Mol1. **Recommendation: start backup chemotype series on 1,2,4-triazole or pyrazole core in parallel.**
- **Combined Score is potency-/pose-weighted with weak developability terms.** The cs #1 pick (row_id 40886) has a cyanocyclopropane substituent that the tox panel flagged as a CN-release hazard. Adding the tox_alerts_count column to the score with a hard negative weight (or as a categorical gate) will prevent this.
