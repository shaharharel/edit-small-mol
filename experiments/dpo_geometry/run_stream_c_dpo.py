#!/usr/bin/env python3
"""Stream C: DPO covFT+RL mol2mol with GEOMETRY-ONLY preference pairs.

Reuses the DPO infrastructure from experiments/run_dpo_generation.py:
  - ReinventMol2MolBackend (wraps REINVENT4 Mol2Mol)
  - dpo_train() training loop

But:
  - Loads preference pairs from data/dpo_pairs/geometry_only.parquet (offline)
  - Loads base policy from models/rl_checkpoints/exp2_v2_rl_v2_stage1.chkpt
  - Samples 10K cohort from trained policy anchored on Mol1

Three phases (each can be skipped via flags):
  --train   : DPO fine-tune
  --sample  : Sample N from the trained policy (anchored on Mol1)
  --status  : Print current state of checkpoints/cohort
"""
from __future__ import annotations
import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
# Stream C uses GPU. run_dpo_generation.py forces CUDA_VISIBLE_DEVICES="" at
# import. Pre-set a sentinel so we can restore visibility right after the
# import (torch reads CUDA_VISIBLE_DEVICES lazily on first cuda call, so this
# is safe as long as we restore before any torch.cuda usage).
_USER_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "REINVENT4"))

# Reuse the existing DPO building blocks
from experiments.run_dpo_generation import (
    DPOConfig,
    ReinventMol2MolBackend,
    PreferencePair,
    setup_logging,
)

# Restore CUDA visibility before any torch.cuda call.
os.environ["CUDA_VISIBLE_DEVICES"] = _USER_CUDA

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Patch ReinventMol2MolBackend.compute_log_probs to PRESERVE GRAD
# ---------------------------------------------------------------------------
# Upstream call in run_dpo_generation.py:298 does `nll.detach()`, which means
# the existing DPO trainer for the REINVENT4 backend cannot compute gradients.
# We monkey-patch to use a grad-preserving variant. (For pi_ref calls, we still
# wrap in torch.no_grad() at the call site.)

def _grad_compute_log_probs(self, source_smiles, target_smiles, batch_size=32):
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from torch.utils.data import DataLoader
    dataset = PairedDataset(
        source_smiles, target_smiles,
        vocabulary=self.model.vocabulary,
        tokenizer=self.model.tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=PairedDataset.collate_fn, drop_last=False,
    )
    nlls = []
    for batch in loader:
        nll = self.model.likelihood(
            batch.input, batch.input_mask, batch.output, batch.output_mask
        )
        nlls.append(nll)  # NO .detach() — preserve grad
    return -torch.cat(nlls, dim=0)


ReinventMol2MolBackend.compute_log_probs = _grad_compute_log_probs


# ---------------------------------------------------------------------------
# Stream C config
# ---------------------------------------------------------------------------

PAIRS_PATH = PROJECT_ROOT / "data" / "dpo_pairs" / "geometry_only.parquet"
BASE_CHKPT = PROJECT_ROOT / "models" / "rl_checkpoints" / "exp2_v2_rl_v2_stage1.chkpt"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "dpo_geometry"
COHORT_DIR = PROJECT_ROOT / "data" / "dpo_geometry_cohort"
ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COHORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pair loader
# ---------------------------------------------------------------------------

def load_pairs(path: Path, split: str = "train") -> list[PreferencePair]:
    df = pd.read_parquet(path)
    df = df[df["split"] == split].copy()
    pairs = [
        PreferencePair(
            source_smi=row["source_smiles"],
            win_smi=row["chosen_smiles"],
            lose_smi=row["rejected_smiles"],
            win_score=float(row["q_chosen"]),
            lose_score=float(row["q_rejected"]),
        )
        for _, row in df.iterrows()
    ]
    return pairs


# ---------------------------------------------------------------------------
# DPO training (adapted from run_dpo_generation.dpo_train)
# ---------------------------------------------------------------------------

def stream_c_dpo_train(
    cfg: DPOConfig,
    pi_theta: ReinventMol2MolBackend,
    pi_ref: ReinventMol2MolBackend,
    train_pairs: list[PreferencePair],
    val_pairs: list[PreferencePair],
    logger: logging.Logger,
    checkpoint_path: Path,
    checkpoint_every_steps: int = 500,
    warmup_steps: int = 500,
) -> dict:
    logger.info(
        f"DPO: {len(train_pairs)} train, {len(val_pairs)} val pairs, "
        f"{cfg.n_epochs} epochs, batch={cfg.train_batch_size}, "
        f"beta={cfg.beta_dpo}, lr={cfg.lr}"
    )

    pi_theta.set_mode("training")
    pi_ref.set_mode("inference")

    optimizer = torch.optim.AdamW(
        pi_theta.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, (len(train_pairs) // cfg.train_batch_size) * cfg.n_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay over remaining
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "step": [], "epoch": [], "train_loss": [],
        "train_acc": [], "lr": [], "reward_gap": [],
    }
    val_history = {
        "epoch": [], "val_loss": [], "val_acc": [], "val_reward_gap": [], "kl_div": [],
    }

    global_step = 0
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        np.random.shuffle(train_pairs)
        pi_theta.set_mode("training")

        for batch_start in range(0, len(train_pairs), cfg.train_batch_size):
            batch = train_pairs[batch_start : batch_start + cfg.train_batch_size]
            if len(batch) == 0:
                continue

            src_list = [p.source_smi for p in batch]
            win_list = [p.win_smi for p in batch]
            lose_list = [p.lose_smi for p in batch]

            lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            with torch.no_grad():
                lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
            log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)

            logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
            loss = -F.logsigmoid(logits).mean()
            reward_gap = (cfg.beta_dpo * (log_ratio_win - log_ratio_lose)).detach().mean().item()
            acc = (logits.detach() > 0).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            history["step"].append(global_step)
            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["train_acc"].append(acc)
            history["lr"].append(scheduler.get_last_lr()[0])
            history["reward_gap"].append(reward_gap)

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  ep{epoch} step{global_step}/{total_steps} "
                    f"loss={loss.item():.4f} acc={acc:.3f} gap={reward_gap:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.0f}s"
                )

            # Periodic checkpoint
            if global_step % checkpoint_every_steps == 0:
                pi_theta.save(str(checkpoint_path))
                logger.info(f"  [ckpt] saved → {checkpoint_path} @ step {global_step}")
                _write_history(history, val_history, checkpoint_path.parent)

        # End of epoch validation
        val_loss, val_acc, val_gap, kl = _validate(pi_theta, pi_ref, val_pairs, cfg)
        val_history["epoch"].append(epoch)
        val_history["val_loss"].append(val_loss)
        val_history["val_acc"].append(val_acc)
        val_history["val_reward_gap"].append(val_gap)
        val_history["kl_div"].append(kl)
        logger.info(
            f"=== Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"val_gap={val_gap:.3f} KL={kl:.4f} ==="
        )
        # save after every epoch
        pi_theta.save(str(checkpoint_path))
        logger.info(f"  [ckpt] end-of-epoch saved → {checkpoint_path}")
        _write_history(history, val_history, checkpoint_path.parent)

    return {"train": history, "val": val_history}


def _validate(
    pi_theta: ReinventMol2MolBackend,
    pi_ref: ReinventMol2MolBackend,
    val_pairs: list[PreferencePair],
    cfg: DPOConfig,
) -> tuple[float, float, float, float]:
    pi_theta.set_mode("inference")
    losses, gaps = [], []
    correct, total = 0, 0
    kl_acc = []
    with torch.no_grad():
        for batch_start in range(0, len(val_pairs), cfg.train_batch_size):
            batch = val_pairs[batch_start : batch_start + cfg.train_batch_size]
            if not batch:
                continue
            src_list = [p.source_smi for p in batch]
            win_list = [p.win_smi for p in batch]
            lose_list = [p.lose_smi for p in batch]

            lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))
            lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
            log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)

            logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
            loss = -F.logsigmoid(logits).mean()
            losses.append(loss.item())
            gaps.append(logits.mean().item())
            correct += (logits > 0).sum().item()
            total += len(batch)
            kl_acc.append((lp_theta_win - lp_ref_win.to(lp_theta_win.device)).mean().item())

    return (
        float(np.mean(losses)) if losses else float("nan"),
        correct / max(total, 1),
        float(np.mean(gaps)) if gaps else float("nan"),
        float(np.mean(kl_acc)) if kl_acc else float("nan"),
    )


def _write_history(history: dict, val_history: dict, out_dir: Path) -> None:
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(val_history).to_csv(out_dir / "validation_history.csv", index=False)


# ---------------------------------------------------------------------------
# Sampling from trained policy
# ---------------------------------------------------------------------------

def sample_cohort(
    backend: ReinventMol2MolBackend,
    source_smi: str,
    n_samples: int,
    batch_size: int,
    out_csv: Path,
    logger: logging.Logger,
    temperature: float = 1.0,
) -> None:
    backend.set_mode("inference")
    logger.info(f"sampling {n_samples} from anchor={source_smi}")

    all_samples = []
    n_remaining = n_samples
    t0 = time.time()
    while n_remaining > 0:
        bsz = min(n_remaining, batch_size)
        gens = backend.sample([source_smi], n_samples_per_source=bsz,
                              temperature=temperature, batch_size=bsz)
        all_samples.extend(gens[0])
        n_remaining -= bsz
        if (n_samples - n_remaining) % 1000 == 0 or n_remaining <= 0:
            logger.info(
                f"  sampled {n_samples - n_remaining}/{n_samples} "
                f"({(n_samples - n_remaining) / max(time.time()-t0, 1):.1f}/s)"
            )

    # Validate + dedupe (keep one canonical record per unique canonical SMILES)
    canon_set = {}
    n_valid = 0
    for raw in all_samples:
        if not raw:
            continue
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            continue
        n_valid += 1
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon not in canon_set:
            canon_set[canon] = raw

    rows = []
    for canon, raw in canon_set.items():
        rows.append({
            "SMILES": canon,
            "SMILES_state": 1,
            "Input_SMILES": source_smi,
            "raw_smiles": raw,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(
        f"sampled: {len(all_samples)} raw, {n_valid} valid, "
        f"{len(df)} unique canonical → {out_csv}"
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="run DPO training")
    p.add_argument("--sample", action="store_true", help="sample cohort from trained policy")
    p.add_argument("--status", action="store_true", help="print status")
    p.add_argument("--pairs", type=str, default=str(PAIRS_PATH))
    p.add_argument("--base-ckpt", type=str, default=str(BASE_CHKPT))
    p.add_argument("--trained-ckpt", type=str,
                   default=str(RESULTS_DIR / "stream_c_dpo.chkpt"))
    p.add_argument("--cohort", type=str,
                   default=str(COHORT_DIR / "cohort_mol1.csv"))
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--sample-batch", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-pairs", type=int, default=0,
                   help="cap training pairs (0 = use all)")
    p.add_argument("--max-val-pairs", type=int, default=800,
                   help="cap validation pairs per epoch (0 = use all)")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(RESULTS_DIR)
    logger.info(f"Stream C DPO driver | device={args.device}")

    if args.status:
        print(json.dumps({
            "base_ckpt": str(BASE_CHKPT),
            "base_exists": BASE_CHKPT.exists(),
            "trained_ckpt": args.trained_ckpt,
            "trained_exists": Path(args.trained_ckpt).exists(),
            "pairs": args.pairs,
            "pairs_exists": Path(args.pairs).exists(),
            "cohort": args.cohort,
            "cohort_exists": Path(args.cohort).exists(),
        }, indent=2))
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"using device {device}")

    cfg = DPOConfig(
        prior_path=args.base_ckpt,
        results_dir=str(RESULTS_DIR),
        model_backend="reinvent_mol2mol",
        device=str(device),
        beta_dpo=args.beta,
        lr=args.lr,
        weight_decay=args.wd,
        n_epochs=args.epochs,
        train_batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        seed=args.seed,
    )

    # Load pairs
    train_pairs = load_pairs(Path(args.pairs), split="train")
    val_pairs = load_pairs(Path(args.pairs), split="val")
    logger.info(f"loaded {len(train_pairs)} train, {len(val_pairs)} val pairs")
    if args.max_train_pairs and args.max_train_pairs < len(train_pairs):
        train_pairs = train_pairs[: args.max_train_pairs]
        logger.info(f"capped train to {len(train_pairs)} pairs")
    if args.max_val_pairs and args.max_val_pairs < len(val_pairs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_pairs), size=args.max_val_pairs, replace=False)
        val_pairs = [val_pairs[i] for i in idx]
        logger.info(f"subsampled val to {len(val_pairs)} pairs (deterministic seed={args.seed})")

    if args.train:
        # Two backends from the SAME base checkpoint
        logger.info(f"loading pi_theta from {args.base_ckpt}")
        pi_theta = ReinventMol2MolBackend(
            prior_path=args.base_ckpt, device=device, mode="training"
        )
        logger.info(f"loading pi_ref from {args.base_ckpt}")
        pi_ref = ReinventMol2MolBackend(
            prior_path=args.base_ckpt, device=device, mode="inference"
        )
        # Freeze pi_ref parameters
        for p in pi_ref.model.get_network_parameters():
            p.requires_grad_(False)

        history = stream_c_dpo_train(
            cfg, pi_theta, pi_ref, train_pairs, val_pairs,
            logger,
            checkpoint_path=Path(args.trained_ckpt),
            checkpoint_every_steps=args.checkpoint_every,
            warmup_steps=args.warmup_steps,
        )

        # Final save
        pi_theta.save(args.trained_ckpt)
        logger.info(f"FINAL ckpt → {args.trained_ckpt}")

        # Save history JSON
        (RESULTS_DIR / "training_summary.json").write_text(json.dumps({
            "config": asdict(cfg),
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "final_train_loss": history["train"]["train_loss"][-1] if history["train"]["train_loss"] else None,
            "final_val_loss": history["val"]["val_loss"][-1] if history["val"]["val_loss"] else None,
            "final_val_acc": history["val"]["val_acc"][-1] if history["val"]["val_acc"] else None,
            "final_kl": history["val"]["kl_div"][-1] if history["val"]["kl_div"] else None,
        }, indent=2))

        # Free GPU memory before sampling
        del pi_ref
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if args.sample:
            sample_cohort(
                pi_theta, ANCHOR_SMI, args.n_samples, args.sample_batch,
                Path(args.cohort), logger,
            )
        return

    if args.sample:
        logger.info(f"loading trained policy from {args.trained_ckpt}")
        backend = ReinventMol2MolBackend(
            prior_path=args.trained_ckpt, device=device, mode="inference"
        )
        sample_cohort(
            backend, ANCHOR_SMI, args.n_samples, args.sample_batch,
            Path(args.cohort), logger,
        )


if __name__ == "__main__":
    main()
