"""M3 family training — soft D-arm finetune of DiffSBDD on CovBinder.

Starts from pretrained `crossdocked_fullatom_cond.ckpt` and adds:
  - (J3) Gaussian jitter on warhead-atom target position (sigma_d / sigma_theta)
  - (B)  optional λ·||predicted_warhead_pos − canonical||² loss penalty
         (DEFERRED — requires invasive change to DiffSBDD denoiser to expose
          predicted positions; skipped for v1)
  - NO C-arm (per 2026-05-24 directive: start clean from pretrained DiffSBDD)

CLI flags (so the same script runs 3 ablation variants across T4/V100/A100):
  --variant {baseline, jitter_small, jitter_med, jitter_large}
  --jitter_sigma_d FLOAT (Å)         default 0.1
  --jitter_sigma_theta FLOAT (deg)   default 5.0
  --epochs INT                       default 10
  --batch_size INT                   default 4
  --lr FLOAT                         default 1e-5
  --csv PATH                         default data/covbinder/covind_training_set.csv
  --out_dir PATH                     default ~/runs/m3_<variant>
"""
from __future__ import annotations
import argparse
import sys
import time
import math
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from anchordiff.covind.dataset import CovIndDataset

# DiffSBDD repo path
DIFFSBDD_ROOT = None
for p in [Path.home() / "DiffSBDD", PROJECT_ROOT.parent / "DiffSBDD"]:
    if p.exists():
        DIFFSBDD_ROOT = p; break
assert DIFFSBDD_ROOT is not None, "DiffSBDD repo not found"
sys.path.insert(0, str(DIFFSBDD_ROOT))

from lightning_modules import LigandPocketDDPM

VARIANTS = {
    "baseline":     (0.0, 0.0),       # finetune CovBinder, no jitter — ablation
    "jitter_small": (0.05, 2.5),      # tight tolerance
    "jitter_med":   (0.10, 5.0),      # the headline variant
    "jitter_large": (0.20, 10.0),     # wide tolerance
}


def apply_warhead_jitter(lig_pos: torch.Tensor, warhead_idx: int,
                         sigma_d: float, sigma_theta_deg: float, rng) -> torch.Tensor:
    """Jitter the warhead atom's position by Gaussian noise.

    Direction-decomposed: sigma_d in the bond-direction (radial), sigma_theta_deg
    in the perpendicular plane (angular wobble around the canonical 1.81 Å direction).

    Simpler isotropic version: sigma_d controls radial; tangential noise modeled
    as sigma_d_perp ≈ d * tan(sigma_theta) ≈ 1.81 * tan(5°) = 0.16 Å for sigma_theta=5°.
    For v1 we use a single isotropic 3D Gaussian with combined sigma:
        sigma_total = sqrt(sigma_d² + (1.81·tan(sigma_theta·π/180))²)
    """
    if sigma_d == 0 and sigma_theta_deg == 0:
        return lig_pos
    sigma_perp = 1.81 * math.tan(math.radians(sigma_theta_deg))
    sigma_total = math.sqrt(sigma_d ** 2 + sigma_perp ** 2)
    noise = torch.from_numpy(rng.normal(0, sigma_total, size=(3,))).float().to(lig_pos.device)
    new_pos = lig_pos.clone()
    new_pos[warhead_idx] = lig_pos[warhead_idx] + noise
    return new_pos


def collate_batch(samples: list) -> dict:
    """Variable-size batching: pack ligands and pockets with masks."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    device = samples[0]["lig_pos"].device
    lig_coords = []
    lig_oh = []
    pkt_coords = []
    pkt_oh = []
    lig_mask = []
    pkt_mask = []
    num_lig = []
    num_pkt = []
    for i, s in enumerate(samples):
        n_l = s["lig_pos"].shape[0]
        n_p = s["pkt_pos"].shape[0]
        lig_coords.append(s["lig_pos"])
        lig_oh.append(s["lig_oh"].float())
        pkt_coords.append(s["pkt_pos"])
        pkt_oh.append(s["pkt_oh"].float())
        lig_mask.append(torch.full((n_l,), i, dtype=torch.long, device=device))
        pkt_mask.append(torch.full((n_p,), i, dtype=torch.long, device=device))
        num_lig.append(n_l)
        num_pkt.append(n_p)
    return {
        "lig_coords":       torch.cat(lig_coords, dim=0).float(),
        "lig_one_hot":      torch.cat(lig_oh, dim=0),
        "num_lig_atoms":    torch.tensor(num_lig, dtype=torch.long, device=device),
        "lig_mask":         torch.cat(lig_mask, dim=0),
        "pocket_coords":    torch.cat(pkt_coords, dim=0).float(),
        "pocket_one_hot":   torch.cat(pkt_oh, dim=0),
        "num_pocket_nodes": torch.tensor(num_pkt, dtype=torch.long, device=device),
        "pocket_mask":      torch.cat(pkt_mask, dim=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), default="jitter_med")
    ap.add_argument("--jitter_sigma_d", type=float, default=None,
                    help="override variant's sigma_d (Å)")
    ap.add_argument("--jitter_sigma_theta", type=float, default=None,
                    help="override variant's sigma_theta (deg)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--csv", type=str,
                    default=str(PROJECT_ROOT / "data/covbinder/covind_training_set.csv"))
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--max_steps_per_epoch", type=int, default=None,
                    help="cap steps/epoch for shorter runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve jitter sigmas
    v_d, v_t = VARIANTS[args.variant]
    sigma_d = args.jitter_sigma_d if args.jitter_sigma_d is not None else v_d
    sigma_theta = args.jitter_sigma_theta if args.jitter_sigma_theta is not None else v_t

    # Output dir
    out_dir = Path(args.out_dir) if args.out_dir else (Path.home() / f"runs/m3_{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== M3 train_d_soft  variant={args.variant}  sigma_d={sigma_d}  sigma_theta={sigma_theta} ===")
    print(f"  out_dir: {out_dir}")
    print(f"  csv:     {args.csv}")

    # Seed
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load DiffSBDD pretrained
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    print(f"  loading DiffSBDD pretrained: {ckpt}")
    model = LigandPocketDDPM.load_from_checkpoint(str(ckpt), map_location=device)
    model.to(device).train()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    ds = CovIndDataset(args.csv, split="train")
    print(f"  dataset: {len(ds)} examples")

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    log_path = out_dir / "train.log"
    flog = open(log_path, "a", buffering=1)  # line-buffered

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        flog.write(line + "\n")

    log(f"START variant={args.variant}  sigma_d={sigma_d}  sigma_theta={sigma_theta}")
    log(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        n_ok = 0; n_skip = 0; ema_loss = None
        for i in range(len(ds)):
            ex = ds[i]
            if ex is None:
                n_skip += 1; continue
            # Apply warhead jitter (data augmentation; the soft D-arm)
            ex_lig_pos = apply_warhead_jitter(
                ex["lig_pos"], ex["warhead_atom_idx"], sigma_d, sigma_theta, rng
            )
            # Build single-sample batch (variable size; batch_size=1 for simplicity)
            sample = {"lig_pos": ex_lig_pos, "lig_oh": ex["lig_oh"],
                      "pkt_pos": ex["pkt_pos"], "pkt_oh": ex["pkt_oh"]}
            for k in sample:
                sample[k] = sample[k].to(device)
            batch = collate_batch([sample])
            if batch is None:
                n_skip += 1; continue

            try:
                out = model(batch)
                loss = out[0] if isinstance(out, tuple) else out
                loss_mean = loss.mean()
                optim.zero_grad()
                loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                lv = float(loss_mean.detach())
                ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
                n_ok += 1; step += 1

                if step % 50 == 0:
                    elapsed = time.time() - t0
                    log(f"epoch {epoch+1}/{args.epochs}  step {step}  loss_ema={ema_loss:.4f}  rate={step/elapsed:.1f} steps/s")
                if step % args.save_every_steps == 0:
                    ck = out_dir / f"step_{step:06d}.ckpt"
                    torch.save({"model": model.state_dict(), "step": step,
                                "args": vars(args), "ema_loss": ema_loss}, ck)
                    log(f"  saved {ck.name}")
                if args.max_steps_per_epoch and (step % args.max_steps_per_epoch == 0):
                    break
            except Exception as e:
                n_skip += 1
                if n_skip < 5:
                    log(f"  step error ({type(e).__name__}): {str(e)[:200]}")

        log(f"epoch {epoch+1} done — ok={n_ok} skip={n_skip} ema_loss={ema_loss}")
        ck = out_dir / f"epoch_{epoch+1:02d}.ckpt"
        torch.save({"model": model.state_dict(), "epoch": epoch+1,
                    "args": vars(args), "ema_loss": ema_loss}, ck)
        log(f"  saved {ck.name}")

    final = out_dir / "final.ckpt"
    torch.save({"model": model.state_dict(), "args": vars(args),
                "ema_loss": ema_loss, "step": step}, final)
    log(f"DONE — final ckpt: {final}  total time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
