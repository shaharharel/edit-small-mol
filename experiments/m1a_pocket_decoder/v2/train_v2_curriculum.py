"""Train the v2-curriculum model.

Reads pair NPZs produced by
`experiments/m1a_pocket_decoder/v2/fixes/build_pairs_v2_curriculum.py`
and fine-tunes the v2 pocket+pose-conditioned decoder so that the [POSE]
token becomes information-necessary (target != anchor).

Differs from `train_m1a_pairs.py`:
  * per-step checkpointing (safe under SPOT preemption)
  * separate val NPZ (not a random subset of train — pockets are held out)
  * "pose-response test": every val step, sample the same anchors with two
    different pose conditions and measure per-token logit KL
  * writes training_curve.json for the report
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT
LOCAL_REINVENT = Path("/Users/shaharharel/Documents/github/REINVENT4")
A100_REINVENT = Path("/home/shaharh_quris_ai/REINVENT4")
REINVENT4_ROOT = LOCAL_REINVENT if LOCAL_REINVENT.exists() else A100_REINVENT
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

FIXES_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
if str(FIXES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXES_DIR))

from m1a_v2_model import load_m1a_v2, save_m1a_v2  # noqa: E402
from train_m1a_pairs import PairDataset, collate_factory, cosine_warmup_lr  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def pose_response_probe(model, val_loader, n_anchors: int = 8) -> dict:
    """Measure how the decoder's next-token logits change when the [POSE]
    input is swapped between two very different values.

    Signal = mean per-token KL divergence of logits(pose=A) vs logits(pose=B).
    A robust pose-listening model should return KL >> 0; a pose-ignoring
    model returns KL ~ 0.
    """
    model.eval()
    device = next(model.parameters()).device
    it = iter(val_loader)
    try:
        batch = next(it)
    except StopIteration:
        return {"n_anchors": 0, "mean_kl": 0.0, "note": "empty val_loader"}
    src, src_mask, trg, trg_mask, res_emb, res_mask, pose = batch
    # Take first n_anchors samples
    B = min(n_anchors, src.shape[0])
    src, src_mask = src[:B], src_mask[:B]
    trg, trg_mask = trg[:B], trg_mask[:B]
    res_emb, res_mask, pose = res_emb[:B], res_mask[:B], pose[:B]

    # Pose A = original z-scored pose in the batch
    # Pose B = shift theta_z by ±1.5 stds (large clamp, definitely different)
    pose_a = pose.clone()
    pose_b = pose.clone()
    pose_b[:, 1] = pose_b[:, 1] + 1.5   # push theta above baseline
    pose_c = pose.clone()
    pose_c[:, 1] = pose_c[:, 1] - 1.5   # push theta below baseline

    def _logits_of(p):
        # We mirror model.likelihood but grab the raw logits
        memory_ext, src_mask_ext = model._build_conditioned_memory(
            src, src_mask, res_emb, res_mask, p)
        trg_in = trg[:, :-1]
        out = model.base.network.decoder(
            model.base.network.tgt_embed(trg_in), memory_ext,
            src_mask_ext, trg_mask)
        log_prob = model.base.network.generator(out, model.base.temperature)
        return log_prob  # (B, T, V) log-probs

    lp_a = _logits_of(pose_a)
    lp_b = _logits_of(pose_b)
    lp_c = _logits_of(pose_c)
    # KL(A || B) per token per anchor
    def _kl(lp_p, lp_q):
        p = lp_p.exp()
        return (p * (lp_p - lp_q)).sum(-1).mean().item()
    kl_ab = _kl(lp_a, lp_b)
    kl_ac = _kl(lp_a, lp_c)
    kl_aa = _kl(lp_a, lp_a)   # sanity ~ 0
    model.train()
    return {"n_anchors": B, "kl_A_vs_B_theta+1.5": kl_ab,
            "kl_A_vs_C_theta-1.5": kl_ac, "kl_A_vs_A_sanity": kl_aa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/v2_curriculum/pairs_train.npz"))
    ap.add_argument("--val_npz", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/v2_curriculum/pairs_val.npz"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                    "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--warm_start", default=str(PROJECT_ROOT /
                    "models/m1a_v2.ckpt"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                    "models/v2_curriculum"))
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--ckpt_interval", type=int, default=1000)
    ap.add_argument("--val_interval", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--curve_path", default=None)
    args = ap.parse_args()

    if args.curve_path is None:
        args.curve_path = str(PROJECT_ROOT /
                              "data/paper_pair_training/v2_curriculum/training_curve.json")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = Path(args.curve_path); curve_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_v2_curriculum] device={device}", flush=True)
    if torch.cuda.is_available():
        print(f"[train_v2_curriculum] gpu={torch.cuda.get_device_name(0)}", flush=True)

    # ---- Pose normalizer from cache ----
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]
    print(f"[train_v2_curriculum] pose_mean={pose_mean} pose_std={pose_std}", flush=True)

    # ---- Load base + warm-start ----
    print("[train_v2_curriculum] loading base model from prior...", flush=True)
    model = load_m1a_v2(args.prior, device, pose_mean=pose_mean, pose_std=pose_std)
    if args.warm_start and Path(args.warm_start).exists():
        print(f"[train_v2_curriculum] warm-starting from {args.warm_start}", flush=True)
        sd = torch.load(args.warm_start, map_location=device, weights_only=False)
        state = sd.get("model_state", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"    missing={len(missing)}  unexpected={len(unexpected)}", flush=True)

    # ---- Datasets ----
    train_ds = PairDataset(Path(args.train_npz), Path(args.cache),
                           model.base.vocabulary, model.base.tokenizer, max_len=128)
    val_ds = PairDataset(Path(args.val_npz), Path(args.cache),
                         model.base.vocabulary, model.base.tokenizer, max_len=128)
    print(f"[train_v2_curriculum] train={len(train_ds)} val={len(val_ds)}", flush=True)
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            collate_fn=collate, num_workers=0)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.max_steps
    print(f"[train_v2_curriculum] total_steps={total_steps} lr_peak={args.lr} "
           f"warmup={args.warmup_steps}", flush=True)

    start_step = 0
    curve = {"steps": [], "train_loss": [], "val_loss": [], "pose_response": [],
             "wallclock_s": [], "best_val": None, "best_step": None,
             "config": vars(args)}
    if args.resume:
        # Find latest step_XXXX.ckpt
        ckpts = sorted(out_dir.glob("step_*.ckpt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"[train_v2_curriculum] resuming from {latest}", flush=True)
            sd = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(sd["model_state"])
            optim.load_state_dict(sd["optim_state"])
            start_step = sd["step"]
            print(f"[train_v2_curriculum] resumed at step {start_step}", flush=True)
            if curve_path.exists():
                curve = json.loads(curve_path.read_text())
                # Truncate history to resume point
                while curve["steps"] and curve["steps"][-1] > start_step:
                    for k in ["steps", "train_loss", "val_loss",
                              "pose_response", "wallclock_s"]:
                        curve[k].pop()

    log_path = out_dir / "train_v2_curriculum.jsonl"
    log_fp = open(log_path, "a")
    model.train()
    step = start_step
    t_start = time.time()
    accum_loss = 0.0; accum_n = 0
    best_val = curve.get("best_val")
    best_step = curve.get("best_step")

    keep_going = True
    epoch = 0
    while keep_going and step < total_steps:
        for batch in train_loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose = batch
            lr = cosine_warmup_lr(step, args.warmup_steps, total_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            nll = model.likelihood(src, src_mask, trg, trg_mask,
                                    res_emb, res_mask, pose)
            loss = nll.mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            accum_loss += float(loss.item()) * src.shape[0]
            accum_n += src.shape[0]
            step += 1
            if step % args.log_every == 0:
                avg = accum_loss / max(1, accum_n)
                elapsed = time.time() - t_start
                print(f"[train] step={step}/{total_steps} epoch={epoch} "
                       f"loss={avg:.4f} lr={lr:.2e} "
                       f"steps/s={(step-start_step)/max(0.1, elapsed):.2f}", flush=True)
                log_fp.write(json.dumps({"step": step, "epoch": epoch,
                                          "train_loss": avg, "lr": lr,
                                          "wallclock_s": elapsed}) + "\n")
                log_fp.flush()
                accum_loss = 0.0; accum_n = 0
            if step % args.val_interval == 0 or step == total_steps:
                model.eval()
                v_loss = 0.0; v_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        s, sm, t, tm, re, rm, po = vb
                        v_nll = model.likelihood(s, sm, t, tm, re, rm, po).mean()
                        v_loss += float(v_nll.item()) * s.shape[0]
                        v_n += s.shape[0]
                val_avg = v_loss / max(1, v_n)
                probe = pose_response_probe(model, val_loader, n_anchors=8)
                model.train()
                elapsed = time.time() - t_start
                curve["steps"].append(step)
                curve["train_loss"].append(avg if accum_n == 0 else accum_loss / max(1, accum_n))
                curve["val_loss"].append(val_avg)
                curve["pose_response"].append(probe)
                curve["wallclock_s"].append(elapsed)
                if best_val is None or val_avg < best_val:
                    best_val = val_avg; best_step = step
                    curve["best_val"] = best_val; curve["best_step"] = best_step
                    # Save best.chkpt eagerly
                    best_path = out_dir / "best.chkpt"
                    save_m1a_v2(model, str(best_path),
                                  extra={"step": step, "val_loss": val_avg,
                                          "pose_response": probe})
                    print(f"[val] step={step} val_loss={val_avg:.4f} "
                           f"(new best; saved {best_path})", flush=True)
                else:
                    print(f"[val] step={step} val_loss={val_avg:.4f} "
                           f"(best {best_val:.4f} @ {best_step})", flush=True)
                print(f"[pose_probe] step={step} kl_A_vs_B={probe['kl_A_vs_B_theta+1.5']:.4f} "
                       f"kl_A_vs_C={probe['kl_A_vs_C_theta-1.5']:.4f} "
                       f"kl_sanity={probe['kl_A_vs_A_sanity']:.6f}", flush=True)
                log_fp.write(json.dumps({"step": step, "val_loss": val_avg,
                                          "pose_response": probe,
                                          "wallclock_s": elapsed}) + "\n")
                log_fp.flush()
                curve_path.write_text(json.dumps(curve, indent=2))
            if step % args.ckpt_interval == 0:
                ck_path = out_dir / f"step_{step:06d}.ckpt"
                torch.save({"model_state": model.state_dict(),
                             "optim_state": optim.state_dict(),
                             "step": step, "epoch": epoch}, ck_path)
                print(f"[ckpt] step={step} -> {ck_path}", flush=True)
                # Retain only latest 3 step_ ckpts (plus milestone every 5000)
                ck_list = sorted(out_dir.glob("step_*.ckpt"))
                for old in ck_list[:-3]:
                    if int(old.stem.split("_")[-1]) % 5000 != 0:
                        old.unlink(missing_ok=True)
            if step >= total_steps:
                keep_going = False; break
        epoch += 1

    final_path = out_dir / "final.ckpt"
    save_m1a_v2(model, str(final_path),
                  extra={"step": step, "epoch": epoch})
    bv = f"{best_val:.4f}" if best_val is not None else "n/a"
    print(f"[train_v2_curriculum] DONE  step={step}  final={final_path}  "
           f"best={best_step}  best_val={bv}", flush=True)
    log_fp.close()


if __name__ == "__main__":
    main()
