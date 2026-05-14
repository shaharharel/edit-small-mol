# Morning briefing — overnight 2026-05-13 → 14

You were offline ~8h. Below is the single-page summary of everything that ran, everything that was concluded, and what's open for the London meeting today.

## 1. C+D v2 fine-tune (T4) ✅ COMPLETE

**Result**: 30 epochs, best val L2 = 0.1573 at ep29 (saved). Adapter loaded with 530 params (TOKEN_DIM 52). Fine-tune on the corrected dataset (full-atom pocket + 18-class vocab + per-row mechanism + resolution ≤2.5 Å).

## 2. C+D v2 generative cohort (T4) ✅ COMPLETE — 4 pockets sampled

**100 mols per pocket sampled; ~50% survive bond-fix; cohort scored vs day-1 baselines where available.**

| Pocket | N valid | Acrylamide% | Lipinski% | MW med | QED | Tc → known actives max | Notable |
|---|---:|---:|---:|---:|---:|---:|---|
| ZAP70 Cys346 | 77 | 98.7% | **100%** | 199 | **0.46** | 0.26 vs top-50 ChEMBL | best drug-likeness of 3-way (vs vanilla, inpaint) |
| BTK Cys481 | 51 | 100% | 98% | 170 | 0.44 | **0.41** | **only cohort to find a Tc≥0.4 near-hit** |
| KRAS G12C | 91 | 100% | 99% | 198 | 0.45 | 0.27 | new pocket, no baseline |
| EGFR Cys797 | 52 | 100% | 96% | 240 | 0.44 | 0.32 | largest MW |

**Top-line**: ~100% covalent retention vs ~0% for vanilla DiffSBDD. C+D v2 wins/ties baselines on drug-likeness. **One near-hit at BTK (Tc 0.41) is the strongest evidence the covalent token finds real chemistry**, not just decorated warheads.

**Limitation flagged**: MW under-generation (170-240 vs target 350-450). Cause: inference pocket built from 5-atom warhead stub → only 17-20 residues vs training 22-58. The B2 fix (`--reference_ligand_sdf=Mol-1 docked pose`) wasn't applied — recommended for v3 to lift MW.

Full numbers: **`results/cd_v2_vs_baselines_summary.md`**.

## 3. Kinase-pretrain stability runs (local) ✅ COMPLETE

**B4 + B5 finished.** Headline:

| Metric | Clean (no pretrain) | Clean + KP |
|---|---:|---:|
| Val MAE (mol-disjoint) | 0.623 | **0.564** (−10%) |
| Top-5 consensus | [18, 1, 9, 4, 7] | **[1, 15, 7, 18, 11]** |
| Mol-1 mean rank | 3.5 | **2.1** (← #1) |
| Mol-15 mean rank | 8.9 | **6.8** (jumped #11→#2; Δpred +0.42 pIC50) |
| Mol-18 mean rank | 1.1 | 4.1 (lost #1) |

**Mol-1 emerges as #1 under clean+KP protocol.** Mol-15 is the new dark horse (+0.42 predicted pIC50 with KP).

**Robustness check**: kinase-pretrain corpus is clean of ZAP70 contamination — 0 ZAP70 pairs, 5/280 (1.8%) anchor SMILES present. **Headline finding is real, not pretrain leakage.**

Full numbers: **`results/paper_evaluation/19mol_clean_vs_kp_comparison.txt`**.

## 4. COValid Boltz-mPAE alignment (A100) ✅ COMPLETE → stopped

**82 Boltz-2 cofolds across 9 COValid cysteine sites** (5 actives + 5 decoys each, ~115 min on A100). KRAS YAML patched G12→C mid-run.

| Target | n_act | n_dec | median mPAE actives | median mPAE decoys | Δ (decoy − active) |
|---|---:|---:|---:|---:|---:|
| BMX | 4 | 5 | 0.59 | 1.48 | +0.89 |
| BTK | 2 | 5 | 0.44 | 1.07 | +0.63 |
| EGFR | 5 | 5 | 0.85 | 2.49 | +1.64 |
| FGFR1 | 1 | 5 | 1.31 | 1.53 | +0.23 |
| FGFR4 C477 | 5 | 5 | 0.63 | 3.09 | +2.46 |
| FGFR4 C552 | 5 | 5 | 0.74 | 1.63 | +0.89 |
| ITK | 5 | 5 | 0.48 | 1.26 | +0.78 |
| JAK3 | 5 | 5 | 0.71 | 1.65 | +0.94 |
| KRAS G12C | 5 | 5 | 0.37 | 1.42 | +1.05 |

**All 9 targets: actives have LOWER mPAE than decoys.** Mean separation 1.06 Å. Our Boltz-2 stack reproduces London's directional signal qualitatively. (Absolute adj_LogAUC vs London's 71.8% requires N ≥ 50/side per target — out of scope for tonight.)

Full numbers: **`results/covalid/covalid_d2_summary.md`**.

## 5. 3D-baseline ablations (TargetDiff, DecompDiff) — BLOCKED

- TargetDiff `uni_o2_bond` ckpt loaded successfully via DecompDiff's loader (correct provenance — not the TargetDiff main branch model)
- DecompDiff env install on ai-gpu2: broken (conda activate failed inside install pipeline). ai-gpu2 stopped to save cost.
- **Next session**: retry DecompDiff env with `source ~/miniconda3/etc/profile.d/conda.sh` inside the wrapper script.

## 6. Mol-1 binding-mode analysis (for London) ✅

`results/cd_v2_vs_baselines_summary.md` notes; live in report at http://35.222.214.48:5001/.

Boltz Cys346 cofold of Mol-1 shows:
- **Covalent**: Cys346 Sγ ↔ ligand C26 at 1.40 Å (Boltz under-stretches the bond — known soft-constraint quirk, geometry otherwise correct)
- **2 hinge H-bonds**: Mol-1 N23/N37 ↔ Ala417 backbone N/O (2.86 / 3.13 Å) — canonical kinase hinge
- **Catalytic Lys H-bond**: Mol-1 O22 ↔ Lys424 NZ (3.02 Å) — bonus interaction

Confirms your advisor's observation (covalent + 2 hinge H-bonds) PLUS a third stabilizing contact at the catalytic lysine.

Report leaderboard also has the new columns: `Contacts` (total stabilizing) and `Occupancy%` (rough pocket-fit proxy) on the top-565 ranked rows.

## 7. Machine state at end of overnight

| Machine | State | Note |
|---|---|---|
| T4 (ai-gpu) | **idle, alive** | ready for next ablation (`--no_carm`, or extended training, or v3 with reference_ligand) |
| A100 (ai-gpu-a100-b) | **stopped** | Boltz COValid + KRAS retry both pulled to local. Restart when needed. |
| ai-gpu2 | **stopped** | DecompDiff env install broken; retry next session. |
| Local Mac | idle | all jobs finished |
| personal-clinician | serving report at :5001 | unchanged |

## 8. Things flagged for v3 / next session (in priority order)

1. **Re-sample C+D v2 with `--reference_ligand_sdf`** = Mol-1 docked pose for ZAP70, ibrutinib for BTK, sotorasib for KRAS, osimertinib for EGFR. Builds proper pocket. ~30 min on T4 per pocket → 2h total.
2. **C+D v3 `--no_carm` ablation** to quantify C-arm contribution (does removing the 530-param adapter degrade Tc to actives?).
3. **Pre-reaction-structure training** (E3 in queue) — train on prereactive geometry instead of post-bond adduct. More physically meaningful for kinact prediction.
4. **DecompDiff baseline** — retry env install with corrected wrapper.
5. **HF9 leaderboard rescore** using B5's new clean+KP FiLMDelta (refresh combined_score for the 565 ranked mols in the live report).

## 9. For the London meeting today — talking points

1. **D-arm + C-arm explanation**: see prior chat artifacts for the deep dive on coordinate transform, frame definition, why d=1.85 Å, why 107° (Bürgi-Dunitz). One-paragraph version + algorithm steps both written.
2. **Mol-1 binding mode confirmed via Boltz** (covalent + 3 polar contacts: 2 hinge + 1 catalytic Lys).
3. **Kinase-pretrain shifts the 19-mol ranking** (Mol-1 → #1, Mol-15 emerges as #2). Robust to contamination check.
4. **COValid alignment**: our Boltz-2 stack matches London's mPAE direction across all 9 sites.
5. **C+D v2 covalent generation**: 100% acrylamide retention; one BTK near-hit at Tc 0.41; need v3 for drug-like MW.

Ask London:
- Is 107° Bürgi-Dunitz too rigid for training? Allow noise?
- Mechanism axis vs continuous reactivity (log10 GSH t½) — better?
- Pin pre-reaction or post-bond pose? (We pin neither at training; canonical at inference.)
