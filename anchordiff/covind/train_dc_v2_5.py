"""C+D v2.5 fine-tune — same loop as v2 but with v2.5 C-arm components.

Differences from train_dc.py:
  - token_variant="v2_5"  (290-d warhead-Morgan-FP token)
  - CovalentConditioningAdapterV25 (learnable scale + 2-layer MLP + token dropout)

Everything else (data, D-arm local frame, training loop, optimizer, checkpointing,
val-loss tracking) is identical to v2.

CLI:
  python -m anchordiff.covind.train_dc_v2_5 --smoke
  python -m anchordiff.covind.train_dc_v2_5 --epochs 30 --batch_size 4 --lr 1e-5
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# DiffSBDD import path
for cand in [
    Path.home() / "DiffSBDD",
    Path("/home/shaharh_quris_ai/DiffSBDD"),
]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DIFFSBDD_ROOT = cand
        break
else:
    DIFFSBDD_ROOT = None

import torch
import torch.nn as nn
import numpy as np

from anchordiff.covind.dataset import CovIndDataset, F_A
from anchordiff.covind.covalent_token_v2_5 import TOKEN_DIM_V2_5
from anchordiff.covind.cov_adapter_v2_5 import (
    CovalentConditioningAdapterV25, inject_into_pocket_oh_v25,
)


def set_seed(seed: int):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_one(model, adapter, ex, device):
    """Single-example forward — same as v2 but uses v2.5 adapter."""
    try:
        n_lig = ex["lig_pos"].shape[0]
        n_pkt = ex["pkt_pos"].shape[0]
        cov_token = ex["cov_token"].to(device).float()
        pkt_oh = ex["pkt_oh"].to(device).float()
        pkt_oh_aug = inject_into_pocket_oh_v25(pkt_oh, cov_token, adapter)
        data = {
            "lig_coords":     ex["lig_pos"].to(device).float(),
            "lig_one_hot":    ex["lig_oh"].to(device).float(),
            "num_lig_atoms":  torch.tensor([n_lig], device=device, dtype=torch.long),
            "lig_mask":       torch.zeros(n_lig, device=device, dtype=torch.long),
            "pocket_coords":  ex["pkt_pos"].to(device).float(),
            "pocket_one_hot": pkt_oh_aug,
            "num_pocket_nodes": torch.tensor([n_pkt], device=device, dtype=torch.long),
            "pocket_mask":    torch.zeros(n_pkt, device=device, dtype=torch.long),
        }
        out = model(data)
        loss = out[0] if isinstance(out, tuple) else out
        loss_mean = loss.mean()
        if not torch.isfinite(loss_mean):
            return None, "non_finite_loss"
        return loss_mean, None
    except RuntimeError as e:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None, type(e).__name__
    except Exception as e:
        return None, type(e).__name__


def smoke_test(args):
    """Verify v2.5 token + adapter wire end-to-end through DiffSBDD."""
    if DIFFSBDD_ROOT is None:
        print("DiffSBDD repo not found; smoke test cannot run.")
        return False
    from lightning_modules import LigandPocketDDPM

    print(f"Loading DiffSBDD from {DIFFSBDD_ROOT}")
    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt_path), map_location=device).to(device)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")

    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=F_A,
        token_dropout_p=args.token_dropout_p, init_scale=args.init_scale,
    ).to(device)
    print(f"  v2.5 adapter: {sum(p.numel() for p in adapter.parameters()):,} params")
    print(f"    learnable scale (init): {float(adapter.scale):.3f}")
    print(f"    token dropout p: {adapter.token_dropout_p}")

    ds = CovIndDataset(args.csv, split="train", token_variant="v2_5")
    ex = None
    for i in range(20):
        ex = ds[i]
        if ex is not None: break
    if ex is None:
        print("  No valid example available")
        return False
    print(f"  example: {ex['record_id']}  warhead={ex['warhead_class']}  "
          f"token_dim={ex['cov_token'].shape[0]}")
    assert ex["cov_token"].shape[0] == TOKEN_DIM_V2_5, (
        f"Token dim mismatch: dataset gave {ex['cov_token'].shape[0]}, expected {TOKEN_DIM_V2_5}")

    # Forward without grad
    model.eval(); adapter.eval()
    with torch.no_grad():
        loss, err = _forward_one(model, adapter, ex, device)
    print(f"  eval forward loss: {float(loss):+.4f}" if loss is not None else f"  EVAL FAILED: {err}")

    # Forward with grad — check end-to-end gradient flow.
    # Use adapter.eval() for the GRAD CHECK (no token dropout), otherwise
    # there's a token_dropout_p chance the token gets zeroed → adapter output
    # is exactly 0 (zero-init biases) → zero gradient by design. That's correct
    # behavior under dropout but useless for the wire-up sanity check.
    model.train(); adapter.eval()
    for p in adapter.parameters(): p.grad = None
    for p in model.parameters(): p.grad = None
    loss, err = _forward_one(model, adapter, ex, device)
    if loss is None:
        print(f"  TRAIN FAILED: {err}")
        return False
    loss.backward()
    adapter_grad = sum(p.grad.norm().item() ** 2 for p in adapter.parameters() if p.grad is not None) ** 0.5
    log_scale_grad = float(adapter.log_scale.grad) if adapter.log_scale.grad is not None else 0.0
    model_param = next((p for p in model.parameters() if p.grad is not None), None)
    model_grad = float(model_param.grad.norm()) if model_param is not None else 0.0
    print(f"  grad-check loss (adapter.eval): {float(loss):+.4f}")
    print(f"    adapter grad ‖.‖={adapter_grad:.3e}  log_scale grad={log_scale_grad:.3e}")
    print(f"    model   grad ‖.‖={model_grad:.3e}")
    if adapter_grad > 1e-12 and model_grad > 1e-8:
        print("  END-TO-END OK — v2.5 C-arm wires through DiffSBDD.")
        # Also do a sanity check that dropout works in train mode
        adapter.train()
        n_drops = 0
        for _ in range(50):
            with torch.no_grad():
                b = adapter(ex["cov_token"].to(device).float())
            if float(b.abs().sum()) < 1e-9:
                n_drops += 1
        print(f"  token-dropout sanity: {n_drops}/50 zero outputs in train mode "
              f"(expect ~{int(50*adapter.token_dropout_p)})")
        return True
    print("  GRAD CHECK FAILED — dead gradient on adapter or model.")
    return False


def run_fine_tune(args):
    if DIFFSBDD_ROOT is None:
        raise RuntimeError("DiffSBDD repo not found; run on a machine with DiffSBDD installed.")
    from lightning_modules import LigandPocketDDPM

    out_dir = PROJECT_ROOT / "results" / "covind_v2_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets (v2_5 token)…")
    train_ds = CovIndDataset(args.csv, split="train", token_variant="v2_5")
    val_ds   = CovIndDataset(args.csv, split="val",   token_variant="v2_5")
    print(f"  train: {len(train_ds)}   val: {len(val_ds)}")

    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    print(f"Loading DiffSBDD from {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt_path), map_location=device).to(device)
    model.train()

    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=F_A,
        token_dropout_p=args.token_dropout_p, init_scale=args.init_scale,
    ).to(device)
    adapter.train()

    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  v2.5 adapter: {sum(p.numel() for p in adapter.parameters()):,} params  "
          f"(token_dropout_p={adapter.token_dropout_p})")

    adapter_lr = args.lr * args.adapter_lr_mult
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(),   "lr": args.lr},
            {"params": adapter.parameters(), "lr": adapter_lr},
        ],
        weight_decay=1e-5,
    )
    print(f"  optimizer: AdamW  model_lr={args.lr:.1e}  adapter_lr={adapter_lr:.1e}")

    history = {"epoch": [], "train_loss": [], "val_loss": [], "fail_rate": [], "scale": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); adapter.train()
        running_loss = 0.0; opt.zero_grad()
        t_start = time.time()
        n_seen = 0; n_used = 0; n_fail = 0; accum_count = 0
        for idx in range(len(train_ds)):
            ex = train_ds[idx]
            if ex is None: continue
            n_seen += 1
            loss, _ = _forward_one(model, adapter, ex, device)
            if loss is None:
                n_fail += 1; continue
            accum_count += 1
            (loss / args.batch_size).backward()
            n_used += 1
            running_loss += float(loss.detach())
            if accum_count >= args.batch_size:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(adapter.parameters()), max_norm=10.0)
                opt.step(); opt.zero_grad(); accum_count = 0
        if accum_count > 0:
            for p in list(model.parameters()) + list(adapter.parameters()):
                if p.grad is not None:
                    p.grad.mul_(args.batch_size / accum_count)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), max_norm=10.0)
            opt.step(); opt.zero_grad()

        train_loss = running_loss / max(n_used, 1)
        fail_rate = n_fail / max(n_seen + n_fail, 1)
        if fail_rate > 0.01:
            print(f"  ABORT: fail_rate={fail_rate:.2%} > 1%")
            raise RuntimeError(f"fail_rate {fail_rate:.2%} > 1%")

        with torch.no_grad():
            v_loss = 0.0; v_n = 0
            for idx in range(len(val_ds)):
                ex = val_ds[idx]
                if ex is None: continue
                loss, _ = _forward_one(model, adapter, ex, device)
                if loss is None: continue
                v_loss += float(loss); v_n += 1
        val_loss = v_loss / max(v_n, 1)
        scale_val = float(adapter.scale)
        dt = time.time() - t_start
        print(f"  epoch {epoch:3d}/{args.epochs}  train_L2={train_loss:+.4f} (n={n_used}/{n_seen})  "
              f"val_L2={val_loss:+.4f} (n={v_n})  scale={scale_val:.3f}  fail={fail_rate:.2%}  {dt:.0f}s")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["fail_rate"].append(fail_rate)
        history["scale"].append(scale_val)

        is_best = val_loss < best_val
        if is_best: best_val = val_loss
        keep_opt = (epoch == args.epochs) or is_best
        ckpt_out = out_dir / f"dc_v25_ep{epoch}.pt"
        payload = {
            "epoch": epoch,
            "model_state":   model.state_dict(),
            "adapter_state": adapter.state_dict(),
            "args": vars(args),
            "history": history,
            "token_variant": "v2_5",
        }
        if keep_opt: payload["optimizer_state"] = opt.state_dict()
        torch.save(payload, ckpt_out)
        print(f"    saved {ckpt_out}{' (best)' if is_best else ''}{' [+opt]' if keep_opt else ''}")
    print(f"\nDone. Drop counts (train): {train_ds.drop_counts}")
    print(f"Drop counts (val):   {val_ds.drop_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default=str(PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"))
    ap.add_argument("--smoke", action="store_true", help="run smoke test only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter_lr_mult", type=float, default=10.0)
    ap.add_argument("--token_dropout_p", type=float, default=0.1)
    ap.add_argument("--init_scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    print(f"=== C+D v2.5 (seed {args.seed}, token_dropout {args.token_dropout_p}, init_scale {args.init_scale}) ===")

    if args.smoke or args.epochs == 0:
        ok = smoke_test(args)
        print(f"\nSMOKE STATUS: {'OK' if ok else 'FAIL'}")
        return

    run_fine_tune(args)


if __name__ == "__main__":
    main()
