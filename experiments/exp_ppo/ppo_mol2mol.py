"""Minimal PPO (Proximal Policy Optimization) loop for REINVENT4 Mol2Mol.

This is a thin, self-contained alternative to REINVENT4's native DAP (Difference-in-
Augmented-Probability) staged learning loop. Same data flow (sample → score →
gradient step) but with a clipped surrogate PPO objective and K inner update epochs
per sampled batch.

Key simplifications (intentional, documented):
  * Reward is a single scalar per full trajectory (per generated SMILES). We do
    NOT distribute reward token-by-token; we attribute the whole reward to the
    sequence-level log-probability. This is the "REINFORCE-with-clipping"
    formulation of PPO and is the standard simplification for sequence
    generators with a final-state scalar reward.
  * The advantage baseline is a moving average of recent rewards (deque-based).
    A learned value head would lower variance but adds another network to
    train/tune; for the scale of cohorts we run (~50 outer batches × 64
    trajectories), the moving-average baseline is fine and converges in our
    smoke tests.
  * The KL penalty is against the immutable PRIOR (CovInDB-finetuned anchor),
    not against pi_theta_old. This is the "RLHF-style" KL anchoring that keeps
    the agent close to chemically valid space (same role as DAP's prior_lls
    term). PPO's own ratio-clipping handles the pi_theta vs pi_theta_old
    deviation.

Outer loop (per "outer batch"):
  1. Sample B trajectories under pi_theta_old (no_grad, store old log_probs)
  2. Score via REST endpoint (REINVENT4 contract)
  3. Compute advantage = reward - moving_avg_baseline
  4. For K inner PPO epochs:
       - Forward pi_theta on (input, output) → new log_probs (with grad)
       - ratio = exp(new_lp - old_lp)
       - L_clip = -E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
       - L_kl   = +beta_kl * KL(pi_theta || pi_prior)
       - L_ent  = -alpha_ent * H_hat(pi_theta)
       - backward; Adam step
  5. pi_theta_old <- pi_theta (copy state_dict)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as tud

# ----- REINVENT4 imports -----
# Make sure REINVENT4 is importable; the conda quris env should already have it.
try:
    from reinvent.runmodes.create_adapter import create_adapter
    from reinvent.models.transformer.core.dataset.dataset import Dataset
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
    from reinvent.chemistry import conversions
except Exception as e:
    sys.stderr.write(
        f"Could not import REINVENT4 — make sure the quris conda env is active.\n{e}\n"
    )
    raise

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppo_mol2mol")


# =============================================================================
#                          Scoring backends
# =============================================================================


class RESTScorer:
    """REINVENT4-compatible REST scorer.

    Mirrors the request/response contract used by REINVENT4's comp_generic_rest
    plugin (see experiments/reinvent4_film_rest_server.py /score endpoint).
    """

    def __init__(
        self,
        url: str,
        predictor_id: str = "film_delta",
        predictor_version: str = "v1",
        inp_fmt: str = "smiles",
        timeout: float = 600.0,
    ):
        import requests  # local import; only needed when using REST scorer

        self._requests = requests
        self.url = url
        self.params = dict(
            predictor_id=predictor_id,
            predictor_version=predictor_version,
            inp_fmt=inp_fmt,
        )
        self.timeout = timeout

    def __call__(self, smiles: List[str]) -> np.ndarray:
        body = [
            {"input_string": s if s else "", "query_id": str(i)}
            for i, s in enumerate(smiles)
        ]
        r = self._requests.post(self.url, params=self.params, json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()["output"]["successes_list"]
        # Map back by query_id (server may reorder)
        scores = np.zeros(len(smiles), dtype=np.float32)
        for entry in out:
            qid = int(entry["query_id"])
            scores[qid] = float(entry["output_value"])
        return scores


class CallableScorer:
    """Thin wrapper around any callable that takes List[str] and returns np.ndarray."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, smiles: List[str]) -> np.ndarray:
        return np.asarray(self.fn(smiles), dtype=np.float32)


# =============================================================================
#                          Sampling helpers
# =============================================================================


def _standardize_smiles_list(smilies: List[str], randomize: bool = True, isomeric: bool = True) -> List[str]:
    """Same preprocessing the Mol2Mol sampler applies before tokenization."""
    out = []
    for s in smilies:
        try:
            s_std = conversions.convert_to_standardized_smiles(s)
        except Exception:
            s_std = s
        if randomize:
            try:
                mol = conversions.smile_to_mol(s_std)
                if mol is not None:
                    s_std = conversions.mol_to_random_smiles(mol, isomericSmiles=isomeric)
            except Exception:
                pass
        out.append(s_std)
    return out


def _validate_smiles(out_smiles_list: List[str]) -> List[bool]:
    """Return a boolean mask of which output SMILES parse with RDKit."""
    from rdkit import Chem
    mask = []
    for s in out_smiles_list:
        if not s:
            mask.append(False)
            continue
        m = Chem.MolFromSmiles(s)
        mask.append(m is not None)
    return mask


@dataclass
class SampledBatch:
    inputs: List[str]   # SMILES that were fed in (post-randomization)
    outputs: List[str]  # SMILES the model produced
    old_log_probs: torch.Tensor  # shape (B,) — log pi_theta_old(out|in)
    rewards: np.ndarray          # shape (B,)
    valid_mask: np.ndarray       # shape (B,) bool — RDKit-parseable outputs


def sample_batch(
    agent_old,
    input_pool: List[str],
    batch_size: int,
    device: torch.device,
    randomize: bool = True,
) -> Tuple[List[str], List[str], torch.Tensor]:
    """Sample `batch_size` trajectories.

    Implementation mirrors reinvent.runmodes.samplers.mol2mol.Mol2MolSampler.sample
    but stripped down: we draw `batch_size` SMILES per call, picking seeds
    uniformly at random from `input_pool`. Returns (inputs, outputs, old_nlls).
    `old_nlls` is the NLL under the OLD policy (this is what pi_theta_old gave us
    when it sampled; PPO will treat -old_nll as old_log_prob).
    """
    seed_smilies = list(np.random.choice(input_pool, size=batch_size, replace=True))
    proc_smilies = _standardize_smiles_list(seed_smilies, randomize=randomize)

    tokenizer = SMILESTokenizer()
    dataset = Dataset(proc_smilies, agent_old.get_vocabulary(), tokenizer)
    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate_fn,
    )

    inputs_all, outputs_all, nlls_all = [], [], []
    agent_old.set_mode("inference")  # eval mode for sampling (no dropout)
    for src, src_mask in loader:
        src = src.to(device)
        src_mask = src_mask.to(device)
        # adapter.sample returns a SampleBatch(items1=inputs, items2=outputs, nlls=...)
        sb = agent_old.sample(src, src_mask, "multinomial")
        inputs_all.extend(list(sb.input))
        outputs_all.extend(list(sb.output))
        # sb.nlls may be a numpy array of floats
        nlls = sb.nlls
        if not isinstance(nlls, torch.Tensor):
            nlls = torch.tensor(nlls, dtype=torch.float32)
        nlls_all.append(nlls.float())
    old_nlls = torch.cat(nlls_all, dim=0).to(device)
    return inputs_all, outputs_all, old_nlls


def compute_new_log_probs(
    agent,
    inputs: List[str],
    outputs: List[str],
    device: torch.device,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Compute log pi(out|in) under the given agent, batching internally."""
    # PairedDataset filters invalid tokens silently; we need stable alignment, so
    # we tokenize one pair at a time and pad in fixed mini-batches.
    tokenizer = SMILESTokenizer()
    vocab = agent.get_vocabulary()

    # Filter/encode pairs ourselves so we keep the original index alignment.
    encoded_pairs = []
    keep_idx = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        try:
            ei = vocab.encode(tokenizer.tokenize(inp))
            eo = vocab.encode(tokenizer.tokenize(out))
        except KeyError:
            continue
        encoded_pairs.append(
            (
                torch.tensor(ei).long(),
                torch.tensor(eo).long(),
                torch.tensor([0.0]).float(),
            )
        )
        keep_idx.append(i)

    if not encoded_pairs:
        return torch.full((len(inputs),), float("nan"), device=device)

    nll_chunks: List[torch.Tensor] = []
    MB = 16  # likelihood mini-batch (memory-bounded)
    for start in range(0, len(encoded_pairs), MB):
        chunk = encoded_pairs[start : start + MB]
        dto = PairedDataset.collate_fn(chunk)
        src = dto.input.to(device)
        src_mask = dto.input_mask.to(device)
        trg = dto.output.to(device)
        trg_mask = dto.output_mask.to(device)
        if requires_grad:
            agent.set_mode("training")
            nll = agent.likelihood(src, src_mask, trg, trg_mask)
        else:
            agent.set_mode("inference")
            with torch.no_grad():
                nll = agent.likelihood(src, src_mask, trg, trg_mask)
        nll_chunks.append(nll)

    nll_all_kept = torch.cat(nll_chunks, dim=0)  # length == len(keep_idx)
    # Rebuild full-length tensor, NaN for filtered (token-error) rows.
    out_lp = torch.full((len(inputs),), float("nan"), device=device)
    out_lp[keep_idx] = -nll_all_kept  # log_prob = -nll
    return out_lp


# =============================================================================
#                          PPO update
# =============================================================================


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    alpha_ent: float = 0.001
    ppo_epochs: int = 4
    lr: float = 1e-5
    batch_size: int = 64
    outer_steps: int = 50
    baseline_window: int = 256


@dataclass
class PPOStepLog:
    step: int
    n_valid: int
    mean_reward: float
    mean_advantage: float
    mean_kl_to_prior: float
    mean_entropy: float
    clip_fraction: float
    loss: float


class PPOTrainer:
    def __init__(
        self,
        agent,
        agent_old,
        prior,
        scorer,
        input_pool: List[str],
        device: torch.device,
        cfg: PPOConfig,
    ):
        self.agent = agent
        self.agent_old = agent_old
        self.prior = prior
        self.scorer = scorer
        self.input_pool = input_pool
        self.device = device
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(
            self.agent.get_network_parameters(), lr=cfg.lr
        )
        self.baseline = deque(maxlen=cfg.baseline_window)

    # --- helpers ---

    def _swap_old_to_new(self):
        """Copy agent.state_dict -> agent_old.state_dict (in place)."""
        self.agent_old.network.load_state_dict(self.agent.network.state_dict())
        for p in self.agent_old.get_network_parameters():
            p.requires_grad = False

    def _clear_attn_refs(self):
        """Clear MultiHeadedAttention.attn cached refs to keep GPU memory flat.

        Same trick the staged_learning learning.py uses — without it, every PPO
        inner step retains the softmax tensor on the model module attribute,
        and the allocator can't free it across the outer loop on long runs.
        """
        for model in (self.agent, self.agent_old, self.prior):
            net = getattr(model, "network", None)
            if net is None:
                continue
            for m in net.modules():
                if hasattr(m, "attn"):
                    m.attn = None

    # --- the core PPO step ---

    def step(self, step_idx: int) -> PPOStepLog:
        # 1) sample under pi_theta_old
        inputs, outputs, old_nlls = sample_batch(
            self.agent_old,
            self.input_pool,
            self.cfg.batch_size,
            self.device,
            randomize=True,
        )
        old_log_probs = (-old_nlls).detach()  # shape (B,)

        # 2) RDKit-validate outputs and score only valid ones
        valid_mask_list = _validate_smiles(outputs)
        valid_mask = np.array(valid_mask_list, dtype=bool)
        n_valid = int(valid_mask.sum())

        rewards = np.zeros(len(outputs), dtype=np.float32)
        if n_valid > 0:
            valid_smiles = [outputs[i] for i, ok in enumerate(valid_mask_list) if ok]
            valid_scores = self.scorer(valid_smiles)
            j = 0
            for i, ok in enumerate(valid_mask_list):
                if ok:
                    rewards[i] = valid_scores[j]
                    j += 1
            # Invalid SMILES get reward 0 (consistent with REINVENT's invalid_mask)

        # 3) advantage = reward - moving_avg_baseline
        if len(self.baseline) > 0:
            base = float(np.mean(self.baseline))
        else:
            base = float(np.mean(rewards)) if len(rewards) else 0.0
        advantages = rewards - base
        # Update baseline AFTER computing advantage (don't peek)
        for r in rewards:
            self.baseline.append(float(r))

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # 4) prior log-probs (frozen, no_grad) — used for KL penalty.
        with torch.no_grad():
            prior_lp = compute_new_log_probs(
                self.prior, inputs, outputs, self.device, requires_grad=False
            )

        # 5) K inner PPO epochs
        last_loss = float("nan")
        clip_frac_acc = 0.0
        kl_acc = 0.0
        ent_acc = 0.0
        n_epochs_run = 0

        for k in range(self.cfg.ppo_epochs):
            self.optimizer.zero_grad()
            new_lp = compute_new_log_probs(
                self.agent, inputs, outputs, self.device, requires_grad=True
            )

            # Mask: drop NaN log-probs (token-encoding failures) and invalid SMILES
            # (we still want gradient signal from invalid samples via advantage=0?
            # No — invalid samples have reward 0 by definition; their advantage is
            # then -baseline, which would actively push the model AWAY from
            # invalids. That's the right signal — keep them.)
            finite_mask = torch.isfinite(new_lp) & torch.isfinite(old_log_probs)
            if finite_mask.sum().item() == 0:
                logger.warning(f"[step {step_idx} epoch {k}] no finite log-probs; skipping epoch")
                continue

            new_lp_f = new_lp[finite_mask]
            old_lp_f = old_log_probs[finite_mask]
            adv_f = adv_t[finite_mask]
            prior_lp_f = prior_lp[finite_mask]

            # PPO clipped surrogate (we MAXIMIZE this, so the LOSS is its negation)
            ratio = torch.exp(new_lp_f - old_lp_f)
            unclipped = ratio * adv_f
            clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_f
            l_clip = -torch.mean(torch.min(unclipped, clipped))

            # KL(pi_theta || pi_prior) ≈ E_{s~pi_theta_old}[log pi_theta(s) - log pi_prior(s)]
            # We use the samples from pi_theta_old as a Monte Carlo proxy. With
            # the importance-ratio correction this would be unbiased, but for a
            # small per-step KL coefficient the un-corrected estimate is the
            # standard RLHF approximation and works fine in practice.
            kl = torch.mean(new_lp_f - prior_lp_f)

            # Entropy estimate: H ≈ -E[log pi_theta(s)] = -mean(new_lp)
            ent = -torch.mean(new_lp_f)

            loss = l_clip + self.cfg.beta_kl * kl - self.cfg.alpha_ent * ent
            loss.backward()
            # Optional gradient clipping — small but helps stability for transformers
            torch.nn.utils.clip_grad_norm_(self.agent.get_network_parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                # clip fraction = fraction of ratios that hit the clip boundary
                clipped_mask = (ratio < 1.0 - self.cfg.clip_eps) | (ratio > 1.0 + self.cfg.clip_eps)
                clip_frac_acc += float(clipped_mask.float().mean().item())
                kl_acc += float(kl.item())
                ent_acc += float(ent.item())
            last_loss = float(loss.item())
            n_epochs_run += 1

        # 6) swap pi_theta_old <- pi_theta
        self._swap_old_to_new()
        self._clear_attn_refs()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # mean across epochs
        denom = max(1, n_epochs_run)
        return PPOStepLog(
            step=step_idx,
            n_valid=n_valid,
            mean_reward=float(np.mean(rewards)) if len(rewards) else 0.0,
            mean_advantage=float(np.mean(advantages)) if len(advantages) else 0.0,
            mean_kl_to_prior=kl_acc / denom,
            mean_entropy=ent_acc / denom,
            clip_fraction=clip_frac_acc / denom,
            loss=last_loss,
        )

    def train(self, csv_log_path: Path) -> List[PPOStepLog]:
        logs: List[PPOStepLog] = []
        with open(csv_log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "step",
                    "n_valid",
                    "mean_reward",
                    "mean_advantage",
                    "mean_kl_to_prior",
                    "mean_entropy",
                    "clip_fraction",
                    "loss",
                    "wall_s",
                ]
            )
            t0 = time.time()
            for step_idx in range(self.cfg.outer_steps):
                step_t0 = time.time()
                log = self.step(step_idx)
                dt = time.time() - step_t0
                logger.info(
                    f"[step {log.step:03d}] reward={log.mean_reward:+.4f} "
                    f"adv={log.mean_advantage:+.4f} kl={log.mean_kl_to_prior:+.4f} "
                    f"ent={log.mean_entropy:.4f} clipf={log.clip_fraction:.3f} "
                    f"loss={log.loss:+.4f} valid={log.n_valid}/{self.cfg.batch_size} "
                    f"dt={dt:.1f}s"
                )
                writer.writerow(
                    [
                        log.step,
                        log.n_valid,
                        f"{log.mean_reward:.6f}",
                        f"{log.mean_advantage:.6f}",
                        f"{log.mean_kl_to_prior:.6f}",
                        f"{log.mean_entropy:.6f}",
                        f"{log.clip_fraction:.6f}",
                        f"{log.loss:.6f}",
                        f"{time.time() - t0:.1f}",
                    ]
                )
                fh.flush()
                logs.append(log)
        return logs


# =============================================================================
#                          Driver
# =============================================================================


def load_seed_smiles(path: Path) -> List[str]:
    """One SMILES per line (column 0). Skip blank/comment lines."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Tolerate space- or tab-separated supplemental columns
            sm = line.split()[0]
            out.append(sm)
    return out


def build_scorer(reward_url: Optional[str], inproc_fn=None):
    if inproc_fn is not None:
        return CallableScorer(inproc_fn)
    if not reward_url:
        raise ValueError("Either --reward_url or an in-process callable must be provided")
    return RESTScorer(reward_url)


def main():
    p = argparse.ArgumentParser(description="Minimal PPO loop on REINVENT4 mol2mol")
    p.add_argument("--prior", required=True, type=str, help="path to .prior checkpoint")
    p.add_argument("--smiles_file", required=True, type=str, help="seed SMILES (one per line)")
    p.add_argument(
        "--reward_url",
        required=True,
        type=str,
        help="REST scorer endpoint, e.g. http://127.0.0.1:8088/score",
    )
    p.add_argument("--output", required=True, type=str, help="CSV log path")
    p.add_argument("--steps", type=int, default=50, help="number of outer PPO batches")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--alpha_ent", type=float, default=0.001)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint_out",
        type=str,
        default=None,
        help="path for trained checkpoint (default: models/ppo_<prior>_<tag>.prior)",
    )
    p.add_argument("--reward_tag", type=str, default="film", help="tag in default checkpoint name")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        device = torch.device("cpu")

    logger.info(f"Loading prior + agent from {args.prior} on {device}")
    prior, _, model_type = create_adapter(args.prior, "inference", device)
    agent, _, agent_type = create_adapter(args.prior, "inference", device)
    agent_old, _, _ = create_adapter(args.prior, "inference", device)
    assert model_type == "Mol2Mol", f"This script targets Mol2Mol; got {model_type}"

    # Freeze the prior and pi_theta_old completely.
    for net in (prior, agent_old):
        for p_ in net.get_network_parameters():
            p_.requires_grad = False

    # Re-enable training on the agent (the trainable copy).
    for p_ in agent.get_network_parameters():
        p_.requires_grad = True

    smilies = load_seed_smiles(Path(args.smiles_file))
    logger.info(f"Loaded {len(smilies)} seed SMILES from {args.smiles_file}")

    scorer = build_scorer(args.reward_url)

    cfg = PPOConfig(
        clip_eps=args.clip_eps,
        beta_kl=args.beta_kl,
        alpha_ent=args.alpha_ent,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        outer_steps=args.steps,
    )

    trainer = PPOTrainer(agent, agent_old, prior, scorer, smilies, device, cfg)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"PPO config: {cfg}")
    logger.info(f"Logging metrics to {output_path}")

    trainer.train(output_path)

    # Save the trained checkpoint
    if args.checkpoint_out:
        ckpt_path = Path(args.checkpoint_out)
    else:
        prior_name = Path(args.prior).stem
        ckpt_dir = Path(__file__).resolve().parents[2] / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ppo_{prior_name}_{args.reward_tag}.prior"
    logger.info(f"Saving trained checkpoint to {ckpt_path}")
    agent.save_to_file(str(ckpt_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
