"""Same as train_m1a_v2_cfg.py but ALSO unfreezes the base mol2mol decoder.

Rationale: with the base decoder frozen (v2 default), the null_emb path
cannot produce a sensible unconditional distribution because the frozen
decoder was never trained to accept the null token as input.  This causes
CFG at s>1 to amplify a garbage uncond distribution.

Unfreezing the base decoder lets it adapt to accepting the null token as an
alternative conditioning, at the cost of some covFT drift.  We use a MUCH
lower LR for the base (1e-5) than the new/null modules (5e-5) so the covFT
prior isn't destroyed.

This is a DIAGNOSTIC pass — we compare the two trained CFG checkpoints in the
final report.  Whichever wins the "CFG at s>=2 keeps validity" test is used
for downstream sampling.

Outputs:
  models/cfg_retrieval_unfrozen/cfg_final.ckpt
  data/paper_pair_training/cfg_retrieval/train_log_unfrozen.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/cfg_retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from m1a_v2_cfg_model import load_cfg_model, save_cfg_model, M1aV2CfgModel  # noqa: E402
from train_m1a_v2_cfg import (M1aV2Dataset, collate_factory, make_std_mask,  # noqa: E402
                                    cosine_warmup_lr, evaluate, append_csv,
                                    write_progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--warm_start_v2", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "models/cfg_retrieval_unfrozen"))
    ap.add_argument("--log_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval"))
    ap.add_argument("--log_csv_name", default="train_log_unfrozen.csv")
    ap.add_argument("--p_drop", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_new", type=float, default=5e-5,
                    help="LR for null_emb + pocket_enc + pose_enc.")
    ap.add_argument("--lr_base", type=float, default=1e-5,
                    help="LR for base mol2mol decoder (kept low to preserve covFT).")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=15,
                    help="Fewer epochs; base decoder is far more expressive.")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--ckpt_interval", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / "progress_unfrozen.json"
    train_csv = log_dir / args.log_csv_name
    header_written = [False]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]
    print(f"Pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)

    print(f"Loading CFG model warm-started from {args.warm_start_v2}", flush=True)
    model = load_cfg_model(args.prior, device,
                             pose_mean=pose_mean, pose_std=pose_std,
                             init_from_v2_ckpt=args.warm_start_v2)

    # Split params into two groups.
    base_net = model.base.network
    base_ids = set(id(p) for p in base_net.parameters())
    new_params = [p for p in model.parameters()
                    if id(p) not in base_ids and p.requires_grad]
    base_params = [p for p in base_net.parameters() if p.requires_grad]

    # Also add null_emb to new_params (it's a Parameter of the wrapper).
    # Sanity: null_emb should NOT be in base_net.parameters().
    n_new = sum(p.numel() for p in new_params)
    n_base = sum(p.numel() for p in base_params)
    print(f"Trainable params:  new={n_new/1e6:.2f}M  base={n_base/1e6:.2f}M",
           flush=True)

    optim = torch.optim.AdamW(
        [{"params": new_params, "lr": args.lr_new, "weight_decay": args.weight_decay},
         {"params": base_params, "lr": args.lr_base, "weight_decay": args.weight_decay}])

    ds = M1aV2Dataset(Path(args.cache), model.base.vocabulary,
                        model.base.tokenizer, max_len=128)
    N = len(ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"Train={len(train_idx)} Val={len(val_idx)}", flush=True)
    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate, num_workers=0)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}", flush=True)

    start_step = 0
    if args.resume:
        ckpts = sorted(out_dir.glob("cfg_step*.ckpt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from {latest}", flush=True)
            sd = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(sd["model_state"])
            if "optim_state" in sd:
                optim.load_state_dict(sd["optim_state"])
            start_step = sd.get("step", 0)
            print(f"Resumed at step {start_step}", flush=True)
            if train_csv.exists() and train_csv.stat().st_size > 0:
                header_written[0] = True

    model.train()
    step = start_step
    t_start = time.time()
    accum_loss = 0.0; accum_n = 0
    epoch = 0
    keep_going = True
    write_progress(progress_path, phase="training_unfrozen",
                     epoch=epoch, step=step, total_steps=total_steps)

    if start_step == 0:
        val = evaluate(model, val_loader, device)
        print(f"[baseline step 0] val={val}", flush=True)
        append_csv(train_csv, {"step": 0, "epoch": 0, "loss": float("nan"),
                                 **val, "lr_new": 0.0, "lr_base": 0.0,
                                 "wallclock_s": 0.0,
                                 "timestamp_iso":
                                     time.strftime("%Y-%m-%dT%H:%M:%S")},
                     header_written)

    while keep_going and step < total_steps:
        for batch in train_loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose = batch
            B = src.shape[0]
            lr_new = cosine_warmup_lr(step + 1, args.warmup_steps, total_steps,
                                         args.lr_new)
            lr_base = cosine_warmup_lr(step + 1, args.warmup_steps, total_steps,
                                          args.lr_base)
            optim.param_groups[0]["lr"] = lr_new
            optim.param_groups[1]["lr"] = lr_base

            drop_cond = (torch.rand(B, device=device) < args.p_drop)
            nll = model.likelihood(src, src_mask, trg, trg_mask,
                                     res_emb, res_mask, pose,
                                     drop_cond=drop_cond)
            loss = nll.mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            accum_loss += float(loss.item()) * B
            accum_n += B
            step += 1
            if step % 25 == 0:
                avg = accum_loss / accum_n
                elapsed = time.time() - t_start
                ips = (step - start_step) / max(elapsed, 1e-6)
                print(f"step={step}  epoch={epoch}  loss={avg:.4f}  "
                       f"lr_new={lr_new:.2e}  lr_base={lr_base:.2e}  "
                       f"steps/s={ips:.2f}", flush=True)
                accum_loss = 0.0; accum_n = 0
            if step % args.ckpt_interval == 0 or step == total_steps:
                val = evaluate(model, val_loader, device)
                ck_path = out_dir / f"cfg_step{step:06d}.ckpt"
                torch.save({"model_state": model.state_dict(),
                             "optim_state": optim.state_dict(),
                             "step": step, "epoch": epoch,
                             "val_loss": val["val_loss"]},
                            ck_path)
                ckpts = sorted(out_dir.glob("cfg_step*.ckpt"))
                for old in ckpts[:-2]:
                    if int(old.stem.split("step")[-1]) % 5000 != 0:
                        old.unlink(missing_ok=True)
                append_csv(train_csv, {"step": step, "epoch": epoch,
                                         "loss": float(loss.item()), **val,
                                         "lr_new": lr_new, "lr_base": lr_base,
                                         "wallclock_s": time.time() - t_start,
                                         "timestamp_iso":
                                             time.strftime("%Y-%m-%dT%H:%M:%S")},
                             header_written)
                write_progress(progress_path, phase="training_unfrozen_ckpt",
                                 epoch=epoch, step=step,
                                 val_loss=val["val_loss"],
                                 val_cond=val["val_cond_loss"],
                                 val_uncond=val["val_uncond_loss"],
                                 total_steps=total_steps,
                                 wallclock_s=time.time() - t_start)
                print(f"[ckpt] step={step}  val={val}  -> {ck_path}", flush=True)
            if step >= total_steps:
                keep_going = False; break
        epoch += 1

    final_path = out_dir / "cfg_final.ckpt"
    save_cfg_model(model, str(final_path), extra={"step": step, "epoch": epoch})
    print(f"FINAL: {final_path}", flush=True)
    write_progress(progress_path, phase="training_unfrozen_done",
                     final_step=step, final_epoch=epoch)


if __name__ == "__main__":
    main()
