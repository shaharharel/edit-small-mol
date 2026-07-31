"""Offline DPO training on precomputed (prompt, chosen, rejected) preference pairs.

Loss:
  L_DPO = -log σ(β · [logπ_θ(chosen) - logπ_ref(chosen)
                     - logπ_θ(rejected) + logπ_ref(rejected)])

Base policy:    models/rl_checkpoints/exp2_v2_rl_v2_stage1.chkpt (covFT+RL'd Mol2Mol)
Reference:      same checkpoint, frozen
Optimizer:      AdamW lr=5e-5, weight_decay=1e-2
Schedule:       linear warmup 500 steps + cosine decay over total_steps
Batch:          8 pairs (per gradient step)
Epochs:         configurable (default 5)
Checkpointing:  every 500 steps to <out_ckpt_dir>/dpo_step<NNN>.chkpt + dpo_latest.chkpt
Auto-resume:    if dpo_latest.chkpt exists in out_ckpt_dir.
Validation:     10% held out; report DPO loss + reward gap per epoch.

Reuses likelihood-compute pattern from experiments/exp_ppo/dpo_v1.py.

Usage:
    conda run --no-capture-output -n quris python experiments/dpo_composite_offline.py \\
        --prior models/rl_checkpoints/exp2_v2_rl_v2_stage1.chkpt \\
        --pairs data/dpo_pairs/composite_quality.parquet \\
        --out_ckpt_dir models/dpo_checkpoints/composite_v1 \\
        --log_csv data/dpo_pairs/composite_v1_train_log.csv \\
        --epochs 5 --batch_size 8 --lr 5e-5 --beta 0.1
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as tud

try:
    from reinvent.runmodes.create_adapter import create_adapter
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
except Exception as e:
    sys.stderr.write(f"REINVENT4 import failed: {e}\n")
    raise

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dpo_composite_offline")


# --------------------------------------------------------------------------------------
# log-prob computation (reused pattern from exp_ppo/dpo_v1.py)
# --------------------------------------------------------------------------------------

def _encode_batch(inputs, outputs, vocab, tokenizer):
    """Encode (input, output) pairs. Returns list of triples + kept indices."""
    encoded = []
    keep = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        try:
            ei = vocab.encode(tokenizer.tokenize(inp))
            eo = vocab.encode(tokenizer.tokenize(out))
        except KeyError:
            continue
        encoded.append(
            (
                torch.tensor(ei).long(),
                torch.tensor(eo).long(),
                torch.tensor([0.0]).float(),
            )
        )
        keep.append(i)
    return encoded, keep


def compute_log_probs(agent, inputs, outputs, device, requires_grad: bool):
    """Compute logπ(out | in) under the policy adapter. NaN for failures."""
    tokenizer = SMILESTokenizer()
    vocab = agent.get_vocabulary()
    encoded, keep_idx = _encode_batch(inputs, outputs, vocab, tokenizer)
    if not encoded:
        return torch.full((len(inputs),), float("nan"), device=device)
    dto = PairedDataset.collate_fn(encoded)
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
    out = torch.full((len(inputs),), float("nan"), device=device)
    out[keep_idx] = -nll
    return out


def _clear_attn_refs(*models):
    """REINVENT4 caches attention; clear to avoid leaks between steps."""
    for model in models:
        net = getattr(model, "network", None)
        if net is None:
            continue
        for m in net.modules():
            if hasattr(m, "attn"):
                m.attn = None


# --------------------------------------------------------------------------------------
# Dataset + collate
# --------------------------------------------------------------------------------------

class PrefDataset(tud.Dataset):
    def __init__(self, df):
        self.prompts = df["prompt"].tolist()
        self.chosen = df["chosen"].tolist()
        self.rejected = df["rejected"].tolist()
        self.q_chosen = df["q_chosen"].astype(float).tolist()
        self.q_rejected = df["q_rejected"].astype(float).tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return dict(
            prompt=self.prompts[i],
            chosen=self.chosen[i],
            rejected=self.rejected[i],
            q_chosen=self.q_chosen[i],
            q_rejected=self.q_rejected[i],
        )


def collate(batch):
    return dict(
        prompts=[b["prompt"] for b in batch],
        chosen=[b["chosen"] for b in batch],
        rejected=[b["rejected"] for b in batch],
        q_chosen=np.asarray([b["q_chosen"] for b in batch], dtype=np.float32),
        q_rejected=np.asarray([b["q_rejected"] for b in batch], dtype=np.float32),
    )


# --------------------------------------------------------------------------------------
# Optimizer schedule
# --------------------------------------------------------------------------------------

def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    # cosine decay 1.0 -> 0
    progress = (step - warmup) / float(max(1, total - warmup))
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------
# DPO loss
# --------------------------------------------------------------------------------------

def dpo_step(
    agent,
    prior,
    batch,
    beta: float,
    device,
):
    """Returns dict with loss tensor + scalar diagnostics."""
    lp_c_theta = compute_log_probs(agent, batch["prompts"], batch["chosen"], device, requires_grad=True)
    lp_r_theta = compute_log_probs(agent, batch["prompts"], batch["rejected"], device, requires_grad=True)
    with torch.no_grad():
        lp_c_ref = compute_log_probs(prior, batch["prompts"], batch["chosen"], device, requires_grad=False)
        lp_r_ref = compute_log_probs(prior, batch["prompts"], batch["rejected"], device, requires_grad=False)

    finite = (
        torch.isfinite(lp_c_theta)
        & torch.isfinite(lp_r_theta)
        & torch.isfinite(lp_c_ref)
        & torch.isfinite(lp_r_ref)
    )
    n_finite = int(finite.sum().item())
    if n_finite == 0:
        return None

    lp_c_t = lp_c_theta[finite]
    lp_r_t = lp_r_theta[finite]
    lp_c_r = lp_c_ref[finite]
    lp_r_r = lp_r_ref[finite]

    logits = beta * ((lp_c_t - lp_c_r) - (lp_r_t - lp_r_r))
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        # implicit reward = beta * (lp_theta - lp_ref) per Rafailov 2023
        r_chosen = (beta * (lp_c_t - lp_c_r)).mean().item()
        r_rejected = (beta * (lp_r_t - lp_r_r)).mean().item()
        reward_gap = r_chosen - r_rejected
        # Reward-accuracy: fraction of pairs where r_chosen > r_rejected
        acc = (logits > 0).float().mean().item()
    return dict(loss=loss, n=n_finite, r_chosen=r_chosen, r_rejected=r_rejected, gap=reward_gap, acc=acc)


# --------------------------------------------------------------------------------------
# Validation pass (no grad)
# --------------------------------------------------------------------------------------

def validate(agent, prior, val_loader, beta, device, max_batches=None):
    losses, gaps, accs, ns = [], [], [], []
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        lp_c_theta = compute_log_probs(agent, batch["prompts"], batch["chosen"], device, requires_grad=False)
        lp_r_theta = compute_log_probs(agent, batch["prompts"], batch["rejected"], device, requires_grad=False)
        lp_c_ref = compute_log_probs(prior, batch["prompts"], batch["chosen"], device, requires_grad=False)
        lp_r_ref = compute_log_probs(prior, batch["prompts"], batch["rejected"], device, requires_grad=False)
        finite = (
            torch.isfinite(lp_c_theta) & torch.isfinite(lp_r_theta)
            & torch.isfinite(lp_c_ref) & torch.isfinite(lp_r_ref)
        )
        if int(finite.sum()) == 0:
            continue
        lp_c_t = lp_c_theta[finite]; lp_r_t = lp_r_theta[finite]
        lp_c_r = lp_c_ref[finite]; lp_r_r = lp_r_ref[finite]
        logits = beta * ((lp_c_t - lp_c_r) - (lp_r_t - lp_r_r))
        loss = -F.logsigmoid(logits).mean()
        losses.append(float(loss.item()))
        gaps.append(float((logits / max(beta, 1e-8)).mean().item()))  # raw logodds gap
        accs.append(float((logits > 0).float().mean().item()))
        ns.append(int(finite.sum().item()))
        _clear_attn_refs(agent, prior)
    if not losses:
        return dict(val_loss=float("nan"), val_gap=0.0, val_acc=0.0, val_n=0)
    return dict(val_loss=float(np.mean(losses)), val_gap=float(np.mean(gaps)),
                val_acc=float(np.mean(accs)), val_n=int(sum(ns)))


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------

def save_checkpoint(agent, optimizer, scheduler, step: int, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / f"dpo_step{step:06d}.chkpt"
    latest_path = ckpt_dir / "dpo_latest.chkpt"
    agent.save_to_file(str(model_path))
    agent.save_to_file(str(latest_path))
    meta = dict(step=step, model_path=str(model_path))
    optim_path = ckpt_dir / "dpo_latest_optim.pt"
    torch.save(
        dict(
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            meta=meta,
        ),
        optim_path,
    )
    with open(ckpt_dir / "dpo_latest_meta.json", "w") as fh:
        json.dump(meta, fh)
    return model_path


def maybe_resume(args, agent, optimizer, scheduler, device):
    ckpt_dir = Path(args.out_ckpt_dir)
    latest = ckpt_dir / "dpo_latest.chkpt"
    optim_state = ckpt_dir / "dpo_latest_optim.pt"
    meta_path = ckpt_dir / "dpo_latest_meta.json"
    if not (latest.exists() and optim_state.exists() and meta_path.exists()):
        return 0
    logger.info(f"Resuming from {latest}")
    # reload weights into agent
    new_agent, _, _ = create_adapter(str(latest), "inference", device)
    agent.network.load_state_dict(new_agent.network.state_dict())
    del new_agent
    state = torch.load(optim_state, map_location=device)
    try:
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
    except Exception as e:
        logger.warning(f"Could not restore optimizer/scheduler state ({e}); continuing fresh.")
    with open(meta_path) as fh:
        meta = json.load(fh)
    return int(meta["step"])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out_ckpt_dir", required=True)
    p.add_argument("--log_csv", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_max_batches", type=int, default=80)
    p.add_argument("--val_pct", type=float, default=0.10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=0, help="0 = unlimited; cap total grad steps")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    # Load pairs
    df = pd.read_parquet(args.pairs)
    logger.info(f"Loaded {len(df)} pairs from {args.pairs}")
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(df) * args.val_pct)
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    logger.info(f"Train pairs: {len(train_df)}  Val pairs: {len(val_df)}")

    train_ds = PrefDataset(train_df)
    val_ds = PrefDataset(val_df)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = tud.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    logger.info(f"Steps/epoch: {steps_per_epoch}  Total steps: {total_steps}")

    # Load policies
    logger.info(f"Loading prior from {args.prior}")
    prior, _, mt = create_adapter(args.prior, "inference", device)
    agent, _, _ = create_adapter(args.prior, "inference", device)
    assert mt == "Mol2Mol", f"Expected Mol2Mol checkpoint, got {mt}"

    for pm in prior.get_network_parameters():
        pm.requires_grad = False
    for pm in agent.get_network_parameters():
        pm.requires_grad = True

    optimizer = torch.optim.AdamW(
        agent.get_network_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, args.warmup_steps, total_steps)
    )

    resume_step = maybe_resume(args, agent, optimizer, scheduler, device)
    if resume_step > 0:
        logger.info(f"Resuming at step {resume_step}")

    # CSV log
    log_path = Path(args.log_csv); log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_mode = "a" if resume_step > 0 and log_path.exists() else "w"
    fh = open(log_path, csv_mode, newline="")
    w = csv.writer(fh)
    if csv_mode == "w":
        w.writerow([
            "step", "epoch", "lr", "loss", "n", "r_chosen", "r_rejected",
            "reward_gap", "acc", "wall_s", "val_loss", "val_gap", "val_acc", "val_n",
        ])
        fh.flush()

    t_start = time.time()
    step = resume_step
    done = False
    last_val = dict(val_loss=float("nan"), val_gap=0.0, val_acc=0.0, val_n=0)
    try:
        for epoch in range(args.epochs):
            if done:
                break
            for batch in train_loader:
                if step >= total_steps:
                    done = True
                    break
                t_step = time.time()
                optimizer.zero_grad()
                result = dpo_step(agent, prior, batch, args.beta, device)
                if result is None:
                    # Skip non-encodable batch
                    step += 1
                    continue
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(agent.get_network_parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                _clear_attn_refs(agent, prior)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                lr_now = optimizer.param_groups[0]["lr"]
                step += 1
                if step % 25 == 0 or step <= 5:
                    logger.info(
                        f"step={step}/{total_steps} ep={epoch} lr={lr_now:.2e} "
                        f"loss={result['loss'].item():+.4f} gap={result['gap']:+.4f} "
                        f"acc={result['acc']:.3f} dt={time.time()-t_step:.2f}s"
                    )

                if step % args.val_every == 0:
                    logger.info(f"Validating @ step {step} ...")
                    last_val = validate(agent, prior, val_loader, args.beta, device, max_batches=args.val_max_batches)
                    logger.info(
                        f"VAL step={step} loss={last_val['val_loss']:.4f} "
                        f"gap={last_val['val_gap']:.4f} acc={last_val['val_acc']:.3f} n={last_val['val_n']}"
                    )

                w.writerow([
                    step, epoch, f"{lr_now:.6e}", f"{result['loss'].item():.6f}", result["n"],
                    f"{result['r_chosen']:.6f}", f"{result['r_rejected']:.6f}",
                    f"{result['gap']:.6f}", f"{result['acc']:.6f}",
                    f"{time.time()-t_start:.1f}",
                    f"{last_val['val_loss']:.6f}", f"{last_val['val_gap']:.6f}",
                    f"{last_val['val_acc']:.6f}", last_val["val_n"],
                ])
                if step % 5 == 0:
                    fh.flush()

                if step % args.ckpt_every == 0:
                    ck = save_checkpoint(agent, optimizer, scheduler, step, Path(args.out_ckpt_dir))
                    logger.info(f"Saved checkpoint: {ck}")
        # Final
        logger.info("Final validation...")
        last_val = validate(agent, prior, val_loader, args.beta, device, max_batches=None)
        logger.info(
            f"FINAL VAL loss={last_val['val_loss']:.4f} "
            f"gap={last_val['val_gap']:.4f} acc={last_val['val_acc']:.3f} n={last_val['val_n']}"
        )
        ck = save_checkpoint(agent, optimizer, scheduler, step, Path(args.out_ckpt_dir))
        logger.info(f"Saved FINAL checkpoint: {ck}")
        # Mark done
        done_marker = Path(args.out_ckpt_dir) / "dpo_done.json"
        with open(done_marker, "w") as fh2:
            json.dump(
                dict(
                    total_steps=step,
                    epochs_completed=args.epochs,
                    final_val=last_val,
                    wall_s=time.time() - t_start,
                ),
                fh2,
                indent=2,
            )
        logger.info(f"Wrote {done_marker}")
    finally:
        fh.close()


if __name__ == "__main__":
    main()
