# PPO for REINVENT4 mol2mol — minimal loop

A drop-in alternative to REINVENT4's native DAP staged learning, intended for
direct head-to-head comparison on the same cohort (`thiq_rl_exp2_zap70`).

## Files

| file | role |
|---|---|
| `ppo_mol2mol.py` | self-contained PPO trainer (CLI driver) |
| `test_ppo_small.py` | 5-step CPU smoke test (in-process dummy scorer) |
| `README_PPO.md` | this file |

## (a) PPO vs DAP — intuitive difference

REINVENT4's **DAP** loss for a sampled SMILES `s` is
`L_DAP = (log p_prior(s) + sigma · R(s) − log p_agent(s))^2`
i.e. it asks the agent to match an *augmented* prior whose likelihood has been
shifted upward by the (per-sample, scalar) reward. The agent is a regression
target ("be the prior shifted by sigma·R"); there is no concept of a "step
size" in policy space. Every batch is one gradient step.

**PPO** treats the agent as a stochastic policy and the SMILES as actions. It
re-uses each sampled batch for K inner gradient steps with a clipped surrogate
that bounds how far the new policy can move:
`L_CLIP = −E[min(r·A, clip(r, 1−ε, 1+ε)·A)]`, with
`r = π_θ(s) / π_θ_old(s)`, `A = R(s) − baseline`.
Plus a KL anchor against the prior (`+β·KL(π_θ‖π_prior)`) and a small entropy
bonus.

What this changes in practice:
* **K=4 PPO epochs per batch** ≈ 4× more gradient steps per scoring call
  (scoring is expensive, sampling is cheap, gradients are cheaper than both —
  this is the whole point of PPO for off-the-shelf RLHF).
* **Trust-region clipping** prevents the agent from drifting catastrophically
  on a single noisy reward batch; in DAP a high-reward outlier directly
  pushes the regression target.
* **Explicit prior KL anchor** instead of DAP's implicit `prior_lls` in the
  augmented likelihood. This decouples "stay near a valid prior" (KL) from
  "do what the reward says" (advantage), so the two can be tuned
  independently.

## (b) Simplifications in this implementation

| simplification | what it costs | why we accepted it |
|---|---|---|
| **Moving-average baseline** (no value head) | higher gradient variance | adding a value net = another model to train, tune, and checkpoint. For ~50 outer batches × 64 trajectories the MA baseline converges fine in our smoke runs. |
| **Trajectory-level reward** (single scalar per full SMILES) | no token-level credit assignment — the model can't tell *which* token of the output was the good one | this is the standard RLHF formulation for sequence generators with a final-state reward; token-level reward would require a process reward model. |
| **KL estimated on π_θ_old samples** (no importance correction) | biased KL estimate when π_θ moves far from π_θ_old | β_KL is small (0.01) and the clip range keeps π_θ close to π_θ_old; bias is dominated by clip-region error. |
| **Entropy ≈ −E[log π_θ(s)]** (1-sample MC) | high-variance entropy estimate | α_ent is tiny (0.001) and entropy is here only as a soft regularizer, not as a primary signal. |
| **Per-trajectory, not per-token, log-probs** | larger effective batch needed for low-variance ratios | matches REINVENT4's `likelihood_smiles` interface — no need to fork the model API. |

## (c) Running a real experiment tomorrow (A100)

The setup mirrors `experiments/thiq_rl_tomls/thiq_rl_exp2_zap70.toml`: same
prior, same seeds, same FiLM REST scorer, same reward composition. The only
difference is the optimizer (PPO vs DAP).

### 1. Launch the FiLM REST scorer (separate shell on the A100)

```bash
cd ~/edit-small-mol
source /opt/miniconda3/envs/quris/bin/activate
python experiments/reinvent4_film_rest_server.py --host 127.0.0.1 --port 8088
# wait for "[server] Model loaded: N anchors, ..." line
```

### 2. Run the PPO loop

```bash
cd ~/edit-small-mol
python experiments/exp_ppo/ppo_mol2mol.py \
    --prior   models/reinvent4_mol2mol_covalent_ft.prior \
    --smiles_file data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi \
    --reward_url  http://127.0.0.1:8088/score \
    --output      results/ppo_thiq_rl_exp2_zap70.csv \
    --steps       50 \
    --batch_size  64 \
    --ppo_epochs  4 \
    --lr          1e-5 \
    --clip_eps    0.2 \
    --beta_kl     0.01 \
    --alpha_ent   0.001 \
    --device      cuda \
    --reward_tag  thiq_rl_exp2_zap70
# trained checkpoint will be saved to
#   models/ppo_reinvent4_mol2mol_covalent_ft_thiq_rl_exp2_zap70.prior
```

### 3. Direct comparison to the DAP cohort

The DAP-trained checkpoint
`models/rl_checkpoints_b/thiq_rl_exp2_zap70_stage1.chkpt` was produced from
the same prior with `batch_size=8, max_steps=50, sigma=128, rate=1e-4`. The
PPO config above scales the effective work to a similar wall-clock budget
(`batch_size=64, ppo_epochs=4` → 64×4 = 256 grad-aware passes per outer
batch vs DAP's 8×1 = 8; comparable on an A100 because the dominant cost is
the FiLM REST call, not the model forward).

For an apples-to-apples reward comparison, generate molecules from both
checkpoints with `experiments/run_reinvent4_generation.py --prior <ckpt>`
and score them with the same FiLM model.

### 4. What to watch in the log CSV

* `mean_reward`: should trend upward over the 50 steps.
* `mean_kl_to_prior`: starts at ~0, grows as the agent moves away. If it
  blows past ~5, the agent is going off-policy; raise `--beta_kl` to 0.05.
* `clip_fraction`: ratio of trajectories that hit the ±0.2 clip boundary.
  Healthy range is 0.1–0.3. If consistently > 0.5, lower `--lr` to 3e-6.
* `mean_entropy`: should drop slightly as the policy sharpens. A sudden
  collapse to near-zero = mode collapse; raise `--alpha_ent` to 0.005.

## Smoke test

```bash
cd experiments/exp_ppo
python test_ppo_small.py
# expected: "[smoke] PASS" in ~3-5 minutes on a Mac laptop CPU
```
