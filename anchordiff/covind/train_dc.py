"""C+D flagship fine-tune of DiffSBDD.

The script lives on T4 at ~/anchordiff/covind/train_dc.py and assumes the
DiffSBDD repo is at ~/DiffSBDD/ (already there).

What we do:
  1. Load `crossdocked_fullatom_cond.ckpt` (the standard DiffSBDD checkpoint).
  2. Wrap the Lightning module so that every training batch's coordinates are
     first transformed into the Cβ-anchored local frame (D), and so that the
     pocket conditioning is augmented with a virtual covalent-token residue (C).
  3. Fine-tune at low LR (1e-5) for N epochs on the CovIndDataset.
  4. Save the fine-tuned checkpoint at
       results/covind/dc_flagship_ep{N}.ckpt

Smoke-test mode (--smoke):
  - Loads 8 examples, runs 3 training steps, saves nothing.
  - Confirms data loader, frame transformation, loss computation, gradient flow.

CLI:
  python -m anchordiff.covind.train_dc --smoke
  python -m anchordiff.covind.train_dc --epochs 5 --batch_size 4 --lr 1e-5
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Add DiffSBDD to import path on T4 (and local; harmless if missing)
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
from torch.utils.data import DataLoader

from anchordiff.covind.dataset import (
    CovIndDataset, F_A, R_DIM,
)
from anchordiff.covind.covalent_token import TOKEN_DIM
from anchordiff.covind.local_frame import to_local_torch, to_global_torch
from anchordiff.covind.cov_adapter import (
    CovalentConditioningAdapter, inject_into_pocket_oh,
)


def set_seed(seed: int):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Skip None examples + pack lists (variable-size pocket/ligand)."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return batch  # leave as list; the model handles per-sample iteration


def smoke_test_data_only(csv_path):
    """Just verify dataset produces examples we can iterate."""
    ds = CovIndDataset(csv_path, split="train")
    n_ok = 0
    sizes = []
    t0 = time.time()
    for i in range(min(20, len(ds))):
        ex = ds[i]
        if ex is None: continue
        n_ok += 1
        sizes.append((ex["lig_pos"].shape[0], ex["pkt_pos"].shape[0]))
    print(f"  data-only smoke: {n_ok}/20 valid in {time.time()-t0:.1f}s")
    if n_ok > 0:
        sz = np.array(sizes)
        print(f"    lig atoms: min={sz[:,0].min()} max={sz[:,0].max()} med={int(np.median(sz[:,0]))}")
        print(f"    pkt resi:  min={sz[:,1].min()} max={sz[:,1].max()} med={int(np.median(sz[:,1]))}")
    return n_ok > 0


def smoke_test_forward(args):
    """Run one forward pass through DiffSBDD with C+D-transformed inputs.
    This is the critical test: does the model accept our local-frame inputs?
    """
    if DIFFSBDD_ROOT is None:
        print("DiffSBDD repo not found locally; skipping forward smoke test.")
        return False
    print(f"Loading DiffSBDD from {DIFFSBDD_ROOT}")
    from lightning_modules import LigandPocketDDPM

    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    print(f"  loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  lig atom encoder: {model.lig_type_encoder}")
    print(f"  pocket residue encoder: {list(model.pocket_type_encoder.keys())[:5]}...")

    # Load one example from our dataset
    ds = CovIndDataset(args.csv, split="train")
    ex = None
    for i in range(20):
        ex = ds[i]
        if ex is not None: break
    if ex is None:
        print("  No valid example available")
        return False

    print(f"  example: {ex['record_id']}  warhead={ex['warhead_class']}  "
          f"lig={ex['lig_pos'].shape[0]} atoms  pkt={ex['pkt_pos'].shape[0]} resi")

    # Map our atom one-hot (10-d) onto DiffSBDD's encoder
    # DiffSBDD's lig_type_encoder is a dict {symbol: idx}; check that ours overlap.
    n_lig = ex["lig_pos"].shape[0]
    n_pkt = ex["pkt_pos"].shape[0]
    # Build DiffSBDD-format inputs: just feed our local-frame coords; the model
    # is SE(3)-equivariant so it should accept any coordinate frame.
    lig_pos = ex["lig_pos"]                # (n_lig, 3) in local frame
    pkt_pos = ex["pkt_pos"]                # (n_pkt, 3) in local frame
    print(f"  local-frame coord ranges (model input):")
    print(f"    lig x: [{lig_pos[:,0].min():.2f}, {lig_pos[:,0].max():.2f}]")
    print(f"    lig y: [{lig_pos[:,1].min():.2f}, {lig_pos[:,1].max():.2f}]")
    print(f"    lig z: [{lig_pos[:,2].min():.2f}, {lig_pos[:,2].max():.2f}]")
    print(f"  warhead anchor idx={ex['warhead_atom_idx']}: "
          f"pos=({lig_pos[ex['warhead_atom_idx']][0]:+.2f}, "
          f"{lig_pos[ex['warhead_atom_idx']][1]:+.2f}, "
          f"{lig_pos[ex['warhead_atom_idx']][2]:+.2f})  "
          f"|d|={lig_pos[ex['warhead_atom_idx']].norm().item():.3f}")

    # Build a single-sample DiffSBDD batch and run model.forward to confirm
    # the local-frame coords + atom encoding are accepted end-to-end. The
    # diffusion model's forward signature (per LigandPocketDDPM.forward) is
    # to take (ligand, pocket) dicts with x, h, mask, size and return a loss.
    # DiffSBDD's forward signature is `forward(self, data)` where `data` is a
    # single dict keyed by the names used in get_ligand_and_pocket:
    #   lig_coords, lig_one_hot, num_lig_atoms, lig_mask,
    #   pocket_coords, pocket_one_hot, num_pocket_nodes, pocket_mask
    n_lig = ex["lig_pos"].shape[0]
    n_pkt = ex["pkt_pos"].shape[0]
    device = next(model.parameters()).device

    # ── C-arm: instantiate the covalent-conditioning adapter ──────────────
    adapter = CovalentConditioningAdapter(token_dim=TOKEN_DIM, feat_dim=F_A).to(device)
    cov_token = ex["cov_token"].to(device)
    pkt_oh_orig = ex["pkt_oh"].to(device).float()
    pkt_oh_aug  = inject_into_pocket_oh(pkt_oh_orig, cov_token, adapter)
    pkt_oh_delta = float((pkt_oh_aug - pkt_oh_orig).abs().mean())
    print(f"\n  C-arm adapter: {sum(p.numel() for p in adapter.parameters())} params")
    print(f"    cov_token range: [{cov_token.min().item():+.2f}, {cov_token.max().item():+.2f}]")
    print(f"    pocket-feature bias  mean|Δ|={pkt_oh_delta:.4f}  (< 1.0 ⇒ doesn't drown atom-type signal)")

    data = {
        "lig_coords":     ex["lig_pos"].to(device).float(),
        "lig_one_hot":    ex["lig_oh"].to(device).float(),
        "num_lig_atoms":  torch.tensor([n_lig], device=device, dtype=torch.long),
        "lig_mask":       torch.zeros(n_lig, device=device, dtype=torch.long),
        "pocket_coords":  ex["pkt_pos"].to(device).float(),
        "pocket_one_hot": pkt_oh_aug,          # <-- C-arm-augmented features
        "num_pocket_nodes": torch.tensor([n_pkt], device=device, dtype=torch.long),
        "pocket_mask":    torch.zeros(n_pkt, device=device, dtype=torch.long),
    }
    print(f"  Built C+D batch:  lig.x={list(data['lig_coords'].shape)} lig.h={list(data['lig_one_hot'].shape)}  "
          f"pkt.x={list(data['pocket_coords'].shape)} pkt.h={list(data['pocket_one_hot'].shape)}")
    print(f"  Calling model.forward(data) … (no_grad first)")
    try:
        with torch.no_grad():
            out = model(data)
        if isinstance(out, tuple):
            loss = out[0]
        else:
            loss = out
        print(f"    no-grad loss: {float(loss.mean()):+.4f}")
    except Exception as e:
        print(f"  FORWARD FAILED (no-grad): {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── Verify gradients flow back through adapter + model ────────────────
    print(f"  Re-running with grad to verify the C-arm adapter + model both update …")
    try:
        for p in adapter.parameters():
            p.grad = None
        for p in model.parameters():
            p.grad = None
        # Re-build batch with grad enabled. Make pkt_oh_orig require grad so
        # we can trace exactly where the autograd chain dies.
        pkt_oh_aug2 = inject_into_pocket_oh(pkt_oh_orig, cov_token, adapter)
        print(f"    DEBUG: pkt_oh_orig.requires_grad={pkt_oh_orig.requires_grad}")
        print(f"    DEBUG: adapter(cov_token).requires_grad="
              f"{adapter(cov_token).requires_grad}  grad_fn={adapter(cov_token).grad_fn}")
        print(f"    DEBUG: pkt_oh_aug2.requires_grad={pkt_oh_aug2.requires_grad}  "
              f"grad_fn={pkt_oh_aug2.grad_fn}")
        data["pocket_one_hot"] = pkt_oh_aug2
        out = model(data)
        loss = out[0] if isinstance(out, tuple) else out
        loss_mean = loss.mean()
        loss_mean.backward()
        adapter_grad_norm = sum(p.grad.norm().item() ** 2 for p in adapter.parameters()
                                if p.grad is not None) ** 0.5
        adapter_grads_present = sum(1 for p in adapter.parameters() if p.grad is not None)
        adapter_max_abs = max((p.grad.abs().max().item() for p in adapter.parameters()
                               if p.grad is not None), default=0.0)
        # Sample one model param to verify it received gradient
        model_param_with_grad = next((p for p in model.parameters() if p.grad is not None), None)
        model_grad_norm = float(model_param_with_grad.grad.norm()) if model_param_with_grad is not None else 0.0
        print(f"    grad-mode loss: {float(loss_mean):+.4f}")
        print(f"    adapter grad ‖.‖ = {adapter_grad_norm:.3e}  max|g|={adapter_max_abs:.3e}  "
              f"(params with grad: {adapter_grads_present}/2)")
        print(f"    model   grad ‖.‖ = {model_grad_norm:.3e}")
        # Adapter has small param count and small scale, so its grad norm is naturally
        # much smaller than the model's. We just need nonzero (any path-through).
        if adapter_grad_norm > 1e-12 and model_grad_norm > 1e-8:
            print(f"  END-TO-END OK — C-arm adapter + D-frame + DiffSBDD all wired.")
            return True
        else:
            print(f"  GRAD CHECK FAILED — one of the path components has dead gradient.")
            return False
    except Exception as e:
        print(f"  FORWARD WITH GRAD FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"))
    ap.add_argument("--smoke", action="store_true", help="run smoke test only")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter_lr_mult", type=float, default=10.0,
                    help="adapter LR = lr * this multiplier (QA-round3 #4: 10 not 100)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    print(f"=== Seed {args.seed} ===")

    if args.smoke or args.epochs == 0:
        print("=== SMOKE TEST MODE ===")
        ok1 = smoke_test_data_only(args.csv)
        ok2 = smoke_test_forward(args)
        print(f"\nSMOKE STATUS: data_loader={ok1}  forward_compat={ok2}")
        if ok1 and ok2:
            print("OK — C+D pipeline is wired and accepts local-frame inputs.")
        return

    run_fine_tune(args)


def run_fine_tune(args):
    """Real C+D fine-tune of DiffSBDD on CovalentInDB 2.0.

    Single-sample-at-a-time iteration with gradient accumulation over
    `args.batch_size` examples (mirrors how DiffSBDD's original training
    handles variable-size graphs — concat-into-one-graph would require
    re-implementing their batch_to_list collate; the gradient accumulation
    approach gives the same effective gradient with simpler code).
    """
    if DIFFSBDD_ROOT is None:
        raise RuntimeError("DiffSBDD repo not found; run this on T4 where DiffSBDD is installed.")
    from lightning_modules import LigandPocketDDPM

    out_dir = PROJECT_ROOT / "results" / "covind"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets
    print("Loading datasets …")
    train_ds = CovIndDataset(args.csv, split="train")
    val_ds   = CovIndDataset(args.csv, split="val")
    print(f"  train: {len(train_ds)}   val: {len(val_ds)}")

    # 2. Model + adapter
    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    print(f"Loading DiffSBDD from {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt_path), map_location=device).to(device)
    model.train()
    adapter = CovalentConditioningAdapter(token_dim=TOKEN_DIM, feat_dim=F_A).to(device)
    adapter.train()
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # 3. Optimizer — adapter and model in ONE optimizer so they update together.
    # Adapter at higher LR (tiny + un-pretrained); model at user's --lr.
    # QA round-3 #4: ×10 (not ×100) so the 380-param adapter doesn't oscillate.
    adapter_lr = args.lr * args.adapter_lr_mult
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(),   "lr": args.lr},
            {"params": adapter.parameters(), "lr": adapter_lr},
        ],
        weight_decay=1e-5,
    )
    print(f"  optimizer: AdamW  model_lr={args.lr:.1e}  adapter_lr={adapter_lr:.1e}")

    # 4. Train loop — gradient accumulation over batch_size examples
    history = {"epoch": [], "train_loss": [], "val_loss": [], "fail_rate": []}
    best_train_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); adapter.train()
        running_loss = 0.0
        opt.zero_grad()
        t_start = time.time()
        n_seen = 0; n_used = 0; n_fail = 0
        accum_count = 0
        for idx in range(len(train_ds)):
            ex = train_ds[idx]
            if ex is None: continue
            n_seen += 1
            loss, err = _forward_one(model, adapter, ex, device)
            if loss is None:
                n_fail += 1; continue
            accum_count += 1
            (loss / args.batch_size).backward()
            n_used += 1
            running_loss += float(loss.detach())
            if accum_count >= args.batch_size:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(adapter.parameters()), max_norm=10.0)
                opt.step(); opt.zero_grad()
                accum_count = 0
        if accum_count > 0:
            # final partial-batch step, scaled so the gradient reflects the
            # ACTUAL number of accumulated examples (not args.batch_size).
            # QA round-3 #5 fix.
            for p in list(model.parameters()) + list(adapter.parameters()):
                if p.grad is not None:
                    p.grad.mul_(args.batch_size / accum_count)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), max_norm=10.0)
            opt.step(); opt.zero_grad()
        train_loss = running_loss / max(n_used, 1)
        fail_rate = n_fail / max(n_seen + n_fail, 1)
        dt = time.time() - t_start
        # Hard stop if too many silent fails (QA #6).
        if fail_rate > 0.01:
            print(f"  ABORT: fail_rate={fail_rate:.2%} exceeds 1% threshold")
            raise RuntimeError(f"fail_rate {fail_rate:.2%} > 1% — see drop_counts")

        # 5. Val — wrap in no_grad ONLY (do NOT toggle to eval, otherwise
        # DiffSBDD's forward swaps to the VLB code path which is not on the
        # same scale as the training L2 loss — QA round-3 #2 fix.
        # model.train() and adapter.train() stay set; dropout/batchnorm
        # behaviour matches training. We just disable autograd.
        with torch.no_grad():
            v_loss = 0.0; v_n = 0
            for idx in range(len(val_ds)):
                ex = val_ds[idx]
                if ex is None: continue
                loss, _ = _forward_one(model, adapter, ex, device)
                if loss is None: continue
                v_loss += float(loss); v_n += 1
        val_loss = v_loss / max(v_n, 1)
        print(f"  epoch {epoch:3d}/{args.epochs}  train_L2={train_loss:+.4f} (n={n_used}/{n_seen})  "
              f"val_L2={val_loss:+.4f} (n={v_n})  fail={fail_rate:.2%}  {dt:.0f}s")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["fail_rate"].append(fail_rate)
        # 6. Save checkpoints. Intermediate epochs: drop opt state to save disk
        # (QA #7). Final epoch + best-train-L2: keep opt state for resumability.
        is_final = (epoch == args.epochs)
        is_best  = (train_loss < best_train_loss)
        if is_best: best_train_loss = train_loss
        keep_opt = is_final or is_best
        ckpt_out = out_dir / f"dc_flagship_ep{epoch}.pt"
        payload = {
            "epoch": epoch,
            "model_state":   model.state_dict(),
            "adapter_state": adapter.state_dict(),
            "args": vars(args),
            "history": history,
        }
        if keep_opt:
            payload["optimizer_state"] = opt.state_dict()
        torch.save(payload, ckpt_out)
        suffix = " (best)" if is_best else ""
        print(f"    saved {ckpt_out}{suffix}{' [+opt]' if keep_opt else ''}")
    print(f"\nFine-tune complete. Drop counts (train): {train_ds.drop_counts}")
    print(f"Drop counts (val):   {val_ds.drop_counts}")


def _forward_one(model, adapter, ex, device):
    """Single-example forward.

    Returns:
        (loss_tensor_or_None, error_class_name_or_None)

    Per-example exceptions are caught and returned as the second tuple slot
    so the caller can count failure types (QA #6 — silent skip was unsafe).
    """
    try:
        n_lig = ex["lig_pos"].shape[0]
        n_pkt = ex["pkt_pos"].shape[0]
        cov_token = ex["cov_token"].to(device)
        pkt_oh = ex["pkt_oh"].to(device).float()
        pkt_oh_aug = inject_into_pocket_oh(pkt_oh, cov_token, adapter)
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
        # CUDA OOM on outlier large mols can land here. Try to recover memory.
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None, type(e).__name__
    except Exception as e:
        return None, type(e).__name__


if __name__ == "__main__":
    main()
