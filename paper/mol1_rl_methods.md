# Mol1-Anchored RL Methods — Master Index

**Goal:** lead-optimize ZAP70 covalent binder Mol1 (`C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`) by generating diverse but Mol1-faithful candidates.

**Last updated:** 2026-06-08

## Mol1 reference

- **Canonical SMILES:** `C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`
- **Murcko scaffold:** `O=C(Nc1c[nH]cn1)c1cccc2c1CNC2`
- **THIQ-acryl core SMARTS:** `C=CC(=O)N1Cc2ccccc2C1` (warhead-bearing pharmacophore)
- **FiLM-predicted pIC50:** 6.59

## Generation method ledger

### Phase 1: Pre-RL Mol1-anchored sampling (no RL, just FT priors)

| ID | Status | Base prior | Anchor | N output | THIQ% | Murcko=Mol1% |
|---|---|---|---|---|---|---|
| EXP2 legacy | done 2026-06 | covalent_ft | Mol1 | 500 | 98.6% | 16.0% |
| EXP6 legacy | done 2026-06 | warhead_tokens | Mol1 | 500 | 47.4% | 0% |
| exp2_preRL_mol1_50K | done 2026-06-07 | covalent_ft | Mol1 | 10,647 | _placeholder_ | _placeholder_ |
| exp6_preRL_mol1_50K | done 2026-06-07 | warhead_tokens | Mol1 | 41,091 | _placeholder_ | _placeholder_ |
| exp6v3_preRL_mol1_50K | done 2026-06-07 | warhead_tokens_v2 | Mol1 | 2,669 | _placeholder_ | _placeholder_ |

### Phase 2: Existing RL'd chkpts × Mol1-anchored sampling (Track A)

These re-sample from RL'd agents (trained on 21–43-seed pools) using Mol1 as input.

| ID | Base prior | RL seeds | N output | Acryl% | **THIQ%** | Murcko=Mol1% |
|---|---|---|---|---|---|---|
| exp2_v2_rl_postRL_mol1_50K | covalent_ft | 43 actives | 15,163 (retry) | _placeholder_ | _placeholder_ | _placeholder_ |
| **exp2_v2_rl_v2_postRL_mol1_50K** | covalent_ft | 21 ZAP70 acryl | 20,019 | 98.1% | **98.4%** ⭐ | 2.5% |
| **exp6_v3_postRL_mol1_50K** | warhead_tokens_v2 | 43 actives | 6,690 | 99.7% | **96.8%** ⭐ | 2.1% |
| exp6_v4_postRL_mol1_50K | covalent_ft | 21 ZAP70 + Vina-in-loop | 18,124 | 6.0% | 8.1% | 3.9% |
| exp6_v5_postRL_mol1_50K | warhead_tokens | 21 ZAP70 acryl | 37,082 | 98.1% | 15.2% | 0% |

**Key finding:** anchoring on Mol1 partially helps but base prior dominates. `covalent_ft` priors give 98%+ THIQ on Mol1 anchor; `warhead_tokens` gives 15%; `warhead_tokens_v2` gives 97%. **Vina-in-loop RL (exp6_v4) destroys the warhead** — only 6% acrylamide retained.

### Phase 3: Track B' — RL with Mol1 in training (warhead_tokens base, GENERIC acrylamide reward)

| ID | RL seeds | Reward | N output | Acryl% | **THIQ%** | Murcko=Mol1% |
|---|---|---|---|---|---|---|
| **B'-1 (mol1RL_v5_seed_mol1_only)** | 1 (Mol1) | acryl SMARTS | 1,480 (smoke) | 99.7% | **99.5%** ⭐ | 0% |
| B'-2 (mol1RL_v5_seed_zap70_all_plus_mol1) | 221 (ZAP70+Mol1) | acryl SMARTS | FAILED | — | — | — |
| B'-3 (mol1RL_v5_seed_kinase_zap70x5_mol1x20) | 34,314 (kinase+ZAP70x5+Mol1x20) | acryl SMARTS | FAILED | — | — | — |

**Key finding:** Mol1-only single-seed RL gives **99.5% THIQ** — but partly because Mol1 is the only seed, not because of reward design.

### Phase 4: THIQ-reward RL on warhead_tokens prior (in progress 2026-06-08)

Same as Phase 3 but reward = THIQ-acryl SMARTS `C=CC(=O)N1Cc2ccccc2C1` instead of generic acrylamide.

| ID | VM | tmux | Seeds | Batch | Status |
|---|---|---|---|---|---|
| thiq_rl_mol1only | ai-gpu2 | thiq_mol1 | 1 | 16 | running |
| thiq_rl_zap70 | ai-gpu | thiq_zap70 | 221 | 8 | running |
| thiq_rl_kinase | ai-gpu-a100 | thiq_kinase | 34,314 | 16 | running |

**Pending columns** (to fill after sampling 1-10K from each chkpt on Mol1):
- N output | Acryl% | THIQ% | Murcko=Mol1% | Tc-to-Mol1 distribution

### Phase 5: THIQ-reward RL on covalent_ft prior (pending — to launch after Phase 4)

EXP2-style base (covalent_ft) is already known to be Mol1-faithful when anchored on Mol1 (98.4% THIQ in `exp2_v2_rl_v2_postRL_mol1_50K`). The hypothesis: THIQ-reward RL on top should push it from 98.4% → 99%+ AND tighten the pIC50 distribution upward.

| ID | VM | tmux | Seeds | Batch | Status |
|---|---|---|---|---|---|
| thiq_rl_exp2_mol1only | TBD | TBD | 1 | 16 | queued |
| thiq_rl_exp2_zap70 | TBD | TBD | 221 | 8 | queued |
| thiq_rl_exp2_kinase | TBD | TBD | 34,314 | 16 | queued |

## Checkpoint storage policy

**All RL'd checkpoints must be backed up to local `models/rl_checkpoints/` (and `models/rl_checkpoints_b/` for Phase 3+) — never relied upon cloud VMs alone for model weights.** Verified policy as of 2026-06-08.

Current local backups:
- `models/reinvent4_mol2mol_covalent_ft.prior` (covalent_ft base)
- `models/reinvent4_mol2mol_warhead_tokens.prior` (warhead_tokens base)
- `models/reinvent4_mol2mol_warhead_tokens_v2.prior` (warhead_tokens_v2 base)
- `models/rl_checkpoints/exp{2_v2_rl, 2_v2_rl_v2, 6_v3, 6_v4, 6_v5}_stage1.chkpt`
- `models/rl_checkpoints_b/mol1RL_v5_seed_mol1_only_stage1.chkpt`
- _placeholder_ — Phase 4 thiq_rl_* chkpts to be saved as they finish

## Lessons learned

1. **Generic acrylamide reward isn't enough** when seed pool is large — RL learns to drop the THIQ scaffold and keep just the warhead. Use the full THIQ-acrylamide SMARTS as reward.
2. **Vina-in-loop RL destroys covalent pharmacophores** — Vina doesn't know about reactive warheads, so the RL drops them (V4: 6% acrylamide).
3. **Base prior matters as much as RL** — `covalent_ft` and `warhead_tokens_v2` priors are already Mol1-anchoring-friendly; `warhead_tokens` isn't.
4. **Single-seed RL specializes** — B'-1 (Mol1-only) gives 99.5% THIQ but very low diversity. Useful for lead-opt; useless for scaffold hopping.
5. **Multi-seed RL diversifies but loses Mol1 fidelity** — needs explicit pharmacophore reward.
