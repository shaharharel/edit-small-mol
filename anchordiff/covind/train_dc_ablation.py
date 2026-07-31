"""C+D unified ablation training — runs J3/J4/J5 variants from a single script.

Variants (chosen via --variant flag, which sets defaults; CLI overrides win):
  j4a: scale-only ablation — v2 token + scale=1.0 (fixed). Tests if bumping
       the 0.25 scale ceiling alone fixes the v2 no-op problem.
  j4b: warhead-FP only — v2.5 token (290-d Morgan FP) + v2-arch adapter
       (single Linear, scale=0.25). Tests if warhead FP alone helps.
  j4c: token-dropout only — v2 token + v2 adapter + token_dropout_p=0.1.
       Tests if dropout regularization alone helps.
  j3:  D-arm jitter — v2 token + v2 adapter, dataset adds N(0, 0.1Å) noise
       to first 5 atom positions in local frame. Tests if hard canonical
       anchor pinning hurts vs soft positioning.
  j5:  spatially-resolved adapter — v2.5 token + 2-layer MLP + dropout +
       spatial_decay_lambda=4.0 (bias to pocket atom decays as exp(-d/4)
       where d = ||pkt_pos|| from Sγ origin). Tests if local C-arm
       conditioning beats uniform broadcast.

CLI:
  python -m anchordiff.covind.train_dc_ablation --variant j4a --epochs 30
  python -m anchordiff.covind.train_dc_ablation --smoke --variant j5
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for cand in [Path.home() / "DiffSBDD", Path("/home/shaharh_quris_ai/DiffSBDD")]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DIFFSBDD_ROOT = cand
        break
else:
    DIFFSBDD_ROOT = None

import torch
import numpy as np

from anchordiff.covind.dataset import CovIndDataset, F_A
from anchordiff.covind.covalent_token import TOKEN_DIM as TOKEN_DIM_V2
from anchordiff.covind.covalent_token_v2_5 import TOKEN_DIM_V2_5
from anchordiff.covind.cov_adapter_ablation import (
    AblationAdapter, inject_into_pocket_oh, inject_into_pocket_oh_spatial,
)


VARIANT_DEFAULTS = {
    "j4a": dict(token_variant="v2",   scale_init=1.0,  scale_learnable=False,
                hidden_dim=0, token_dropout_p=0.0, spatial_decay_lambda=0.0,
                darm_jitter_sigma_pos=0.0),
    "j4b": dict(token_variant="v2_5", scale_init=0.25, scale_learnable=False,
                hidden_dim=0, token_dropout_p=0.0, spatial_decay_lambda=0.0,
                darm_jitter_sigma_pos=0.0),
    "j4c": dict(token_variant="v2",   scale_init=0.25, scale_learnable=False,
                hidden_dim=0, token_dropout_p=0.1, spatial_decay_lambda=0.0,
                darm_jitter_sigma_pos=0.0),
    "j3":  dict(token_variant="v2",   scale_init=0.25, scale_learnable=False,
                hidden_dim=0, token_dropout_p=0.0, spatial_decay_lambda=0.0,
                darm_jitter_sigma_pos=0.1),
    "j5":  dict(token_variant="v2_5", scale_init=0.5,  scale_learnable=True,
                hidden_dim=64, token_dropout_p=0.1, spatial_decay_lambda=4.0,
                darm_jitter_sigma_pos=0.0),
}


def set_seed(seed: int):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_one(model, adapter, ex, device, use_spatial_injection: bool):
    try:
        n_lig = ex["lig_pos"].shape[0]; n_pkt = ex["pkt_pos"].shape[0]
        cov_token = ex["cov_token"].to(device).float()
        pkt_oh = ex["pkt_oh"].to(device).float()
        pkt_pos = ex["pkt_pos"].to(device).float()
        if use_spatial_injection:
            pkt_oh_aug = inject_into_pocket_oh_spatial(pkt_oh, cov_token, adapter, pkt_pos)
        else:
            pkt_oh_aug = inject_into_pocket_oh(pkt_oh, cov_token, adapter)
        data = {
            "lig_coords":     ex["lig_pos"].to(device).float(),
            "lig_one_hot":    ex["lig_oh"].to(device).float(),
            "num_lig_atoms":  torch.tensor([n_lig], device=device, dtype=torch.long),
            "lig_mask":       torch.zeros(n_lig, device=device, dtype=torch.long),
            "pocket_coords":  pkt_pos,
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


def run_fine_tune(args):
    if DIFFSBDD_ROOT is None:
        raise RuntimeError("DiffSBDD repo not found")
    from lightning_modules import LigandPocketDDPM

    out_dir = PROJECT_ROOT / "results" / f"covind_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    token_dim = TOKEN_DIM_V2_5 if args.token_variant == "v2_5" else TOKEN_DIM_V2
    use_spatial = args.spatial_decay_lambda > 0
    print(f"=== variant={args.variant}  token_variant={args.token_variant} (dim={token_dim})  "
          f"scale_init={args.scale_init}  learnable={args.scale_learnable}  "
          f"hidden={args.hidden_dim}  dropout={args.token_dropout_p}  "
          f"spatial_lambda={args.spatial_decay_lambda}  "
          f"jitter_sigma={args.darm_jitter_sigma_pos} ===")

    train_ds = CovIndDataset(args.csv, split="train",
                              token_variant=args.token_variant,
                              darm_jitter_sigma_pos=args.darm_jitter_sigma_pos)
    # NB: val never jitters (we want clean eval)
    val_ds = CovIndDataset(args.csv, split="val",
                            token_variant=args.token_variant,
                            darm_jitter_sigma_pos=0.0)
    print(f"  train: {len(train_ds)}   val: {len(val_ds)}")

    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt_path), map_location=device).to(device)
    model.train()

    adapter = AblationAdapter(
        token_dim=token_dim, feat_dim=F_A,
        scale_init=args.scale_init,
        scale_learnable=args.scale_learnable,
        hidden_dim=args.hidden_dim,
        token_dropout_p=args.token_dropout_p,
        spatial_decay_lambda=args.spatial_decay_lambda,
    ).to(device)
    adapter.train()

    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  adapter: {sum(p.numel() for p in adapter.parameters()):,} params  "
          f"(scale={float(adapter.scale):.3f})")

    adapter_lr = args.lr * args.adapter_lr_mult
    opt = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr},
         {"params": adapter.parameters(), "lr": adapter_lr}],
        weight_decay=1e-5,
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "scale": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); adapter.train()
        opt.zero_grad()
        running_loss = 0.0; n_used = n_seen = n_fail = 0; accum_count = 0
        t_start = time.time()
        for idx in range(len(train_ds)):
            ex = train_ds[idx]
            if ex is None: continue
            n_seen += 1
            loss, _ = _forward_one(model, adapter, ex, device, use_spatial)
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
            raise RuntimeError(f"fail_rate {fail_rate:.2%} > 1%")

        with torch.no_grad():
            v_loss = 0.0; v_n = 0
            for idx in range(len(val_ds)):
                ex = val_ds[idx]
                if ex is None: continue
                loss, _ = _forward_one(model, adapter, ex, device, use_spatial)
                if loss is None: continue
                v_loss += float(loss); v_n += 1
        val_loss = v_loss / max(v_n, 1)
        scale_v = float(adapter.scale)
        dt = time.time() - t_start
        print(f"  epoch {epoch:3d}/{args.epochs}  train_L2={train_loss:+.4f} "
              f"val_L2={val_loss:+.4f}  scale={scale_v:.3f}  {dt:.0f}s")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["scale"].append(scale_v)

        is_best = val_loss < best_val
        if is_best: best_val = val_loss
        if epoch == args.epochs or is_best:
            ckpt_out = out_dir / f"dc_{args.variant}_ep{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "adapter_state": adapter.state_dict(),
                "args": vars(args),
                "history": history,
                "variant": args.variant,
            }, ckpt_out)
            print(f"    saved {ckpt_out}{' (best)' if is_best else ''}")
    print(f"\nDone. Drop counts (train): {train_ds.drop_counts}")
    print(f"Drop counts (val):   {val_ds.drop_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_DEFAULTS.keys()))
    ap.add_argument("--csv", type=str,
                    default=str(PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter_lr_mult", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    # Overridable variant defaults
    ap.add_argument("--token_variant", default=None)
    ap.add_argument("--scale_init", type=float, default=None)
    ap.add_argument("--scale_learnable", action="store_true", default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--token_dropout_p", type=float, default=None)
    ap.add_argument("--spatial_decay_lambda", type=float, default=None)
    ap.add_argument("--darm_jitter_sigma_pos", type=float, default=None)
    args = ap.parse_args()

    # Apply variant defaults for any unset args
    v = VARIANT_DEFAULTS[args.variant]
    for k, default in v.items():
        if getattr(args, k) is None:
            setattr(args, k, default)
    if args.scale_learnable is None:
        args.scale_learnable = v["scale_learnable"]

    set_seed(args.seed)
    if args.smoke:
        args.epochs = 1
        print(f"=== SMOKE: 1 epoch  variant={args.variant} ===")
    run_fine_tune(args)


if __name__ == "__main__":
    main()
