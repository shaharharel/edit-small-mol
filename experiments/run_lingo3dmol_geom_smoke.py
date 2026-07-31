"""L1 Phase 4-geom smoke test — L_anchor_geometry loss.

Runs a tiny CPU overfit smoke on N=4 covalent cocrystals from
`covind_smoke_subset.csv`, using the CovalentInDB2Dataset (with the new
bd_target_coord / warhead_atom_idx / warhead_valid fields wired through),
and the model.forward_train with `anchor_geom_ce_w` / `anchor_geom_weight`
hyperparameters.

Two configurations, both 100 iters on the SAME 4-mol batch:

  Run A — base 4 losses (token + X + Y + Z). geometry loss DISABLED
          via anchor_geom_ce_w=0, anchor_geom_weight=0.
  Run B — base 4 losses + L_anchor_geometry. Default hyperparams
          (anchor_geom_ce_w=10, anchor_geom_weight=1).

Verification:
  1. Both runs' token/X/Y/Z losses decrease.
  2. Run B's loss_anchor_geom (soft-argmax MSE on the warhead position)
     decreases over 100 iters.
  3. Run B's warhead_hard_dist (argmax-to-BD-target Euclidean voxel
     distance) drops to within ~3 grid bins (= 0.3 Å) by step ~100.
  4. Run B's token loss decreases at a rate ~ comparable to Run A's,
     i.e. the geometry loss does NOT overwhelm the other heads.

Output: writes a per-run CSV and a JSON summary to `out_dir`.

CPU only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# CPU shim must be installed before any Lingo3DMol import — match the L1
# train script's exact path setup.
ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))

import lingo3dmol_cpu_shim  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

from model.transformer_v1_res_fac2 import TransformerModel


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    for base in (ORIG_CWD, ROOT):
        cand = (base / p).resolve()
        if cand.exists():
            return str(cand)
    return str((ORIG_CWD / p).resolve())


def freeze_encoder_layers(model: nn.Module, n_freeze: int):
    prefixes = tuple(f"encoder.layers.{i}." for i in range(n_freeze))
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = False
            frozen += p.numel()
        else:
            trainable += p.numel()
    print(f"[freeze] frozen={frozen/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
    return frozen, trainable


def load_pretrained(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {ckpt_path}: missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")


def build_batch(complex_csv: str, pdb_dir: str, cache_dir: str | None,
                T: int, batch_size: int):
    """Build one frozen batch from N covalent cocrystals."""
    from lingo3dmol_l1_dataloader import (
        CovalentInDB2Dataset, collate_fn,
    )
    ds = CovalentInDB2Dataset(
        complex_csv_path=complex_csv,
        pdb_dir=pdb_dir,
        T=T,
        max_complexes=batch_size * 6,   # overshoot in case some rows skip
        cache_dir=cache_dir,
        pocket_radius=15.0,
        verbose=True,
    )
    # Pull `batch_size` valid samples — the dataset walks forward on failure.
    samples = []
    seen_idx = set()
    i = 0
    while len(samples) < batch_size and i < len(ds):
        try:
            s = ds[i]
        except Exception as e:
            print(f"[smoke-build] sample {i} failed: {e}")
            i += 1
            continue
        # Skip rows that have warhead_valid=0 — they don't exercise the
        # geometry loss. Prefer rows with a valid BD target.
        wv = float(s.get("warhead_valid", torch.tensor(0.0)).item())
        if wv > 0.5 and i not in seen_idx:
            samples.append(s)
            seen_idx.add(i)
            print(f"[smoke-build] accepted sample {i} "
                  f"(warhead_atom_idx={int(s['warhead_atom_idx'].item())}, "
                  f"bd_target={s['bd_target_coord'].tolist()})")
        else:
            print(f"[smoke-build] sample {i}: warhead_valid={wv}, skipping")
        i += 1
    if len(samples) < batch_size:
        print(f"[smoke-build] WARNING: only got {len(samples)} valid samples "
              f"(needed {batch_size}). Returning what we have.")
    batch = collate_fn(samples)
    return batch, ds


def _step(model, batch, optimizer, anchor_ce_w, anchor_w):
    """One step. Returns (loss_dict_floats)."""
    optimizer.zero_grad()
    # Inject the per-step hyperparameter values.
    batch_with_hp = dict(batch)
    batch_with_hp["anchor_geom_ce_w"] = anchor_ce_w
    batch_with_hp["anchor_geom_weight"] = anchor_w
    out = model.forward_train(**batch_with_hp)
    loss = out["loss_total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), max_norm=5.0)
    optimizer.step()
    rec = {}
    for k, v in out.items():
        if k == "loss_total":
            continue
        if torch.is_tensor(v):
            if v.dim() == 0:
                rec[k] = float(v.detach().item())
            else:
                # Reduce to mean for higher-dim debug metrics.
                rec[k] = float(v.detach().float().mean().item())
        else:
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                pass
    rec["loss_total"] = float(out["loss_total"].detach().item())
    return rec


def run_smoke(name: str, batch, ckpt_path: str, anchor_ce_w: float,
              anchor_w: float, n_iter: int, lr: float, out_dir: Path,
              freeze_n: int) -> dict:
    print(f"\n========== {name} "
          f"(anchor_ce_w={anchor_ce_w}, anchor_w={anchor_w}, "
          f"n_iter={n_iter}, lr={lr}) ==========")

    model = TransformerModel()
    load_pretrained(model, ckpt_path)
    freeze_encoder_layers(model, freeze_n)
    model = model.to("cpu")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    csv_path = out_dir / f"smoke_{name}.csv"
    cols = ["step", "wall_sec", "loss_total", "loss_token",
            "loss_x", "loss_y", "loss_z",
            "loss_anchor_geom", "loss_warhead_ce",
            "warhead_hard_dist", "warhead_n_valid"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    model.train()
    t0 = time.time()
    history = []
    for step in range(n_iter):
        m = _step(model, batch, opt, anchor_ce_w, anchor_w)
        wall = time.time() - t0
        row = [
            step, wall,
            m.get("loss_total", float("nan")),
            m.get("loss_token", float("nan")),
            m.get("loss_x", float("nan")),
            m.get("loss_y", float("nan")),
            m.get("loss_z", float("nan")),
            m.get("loss_anchor_geom", float("nan")),
            m.get("loss_warhead_ce", float("nan")),
            m.get("warhead_hard_dist", float("nan")),
            m.get("warhead_n_valid", float("nan")),
        ]
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if step % max(1, n_iter // 10) == 0 or step == n_iter - 1:
            geo_str = ""
            if "loss_anchor_geom" in m:
                geo_str = (f"  geom={m['loss_anchor_geom']:.1f} "
                           f"wh_ce={m.get('loss_warhead_ce', float('nan')):.3f} "
                           f"hd={m.get('warhead_hard_dist', float('nan')):.2f}")
            print(f"  step={step:3d} t={wall:6.1f}s "
                  f"L={m['loss_total']:.4f} (tok={m['loss_token']:.3f} "
                  f"x={m['loss_x']:.3f} y={m['loss_y']:.3f} "
                  f"z={m['loss_z']:.3f}){geo_str}")

    summary = {
        "name": name,
        "anchor_ce_w": anchor_ce_w,
        "anchor_w": anchor_w,
        "n_iter": n_iter,
        "lr": lr,
        "wall_total_sec": history[-1][1] if history else 0.0,
        "wall_per_step_sec": (history[-1][1] / n_iter) if history else 0.0,
        "initial": {c: history[0][i] for i, c in enumerate(cols)},
        "final": {c: history[-1][i] for i, c in enumerate(cols)},
        "csv_path": str(csv_path),
    }
    # Also save best for each non-NaN loss
    arr = np.asarray([[h[i] for i in range(2, len(cols))] for h in history],
                     dtype=np.float64)
    best = {}
    for j, c in enumerate(cols[2:]):
        col = arr[:, j]
        if np.isfinite(col).any():
            best[c] = float(np.nanmin(col))
    summary["best"] = best
    return summary


def main(args):
    args.complex_csv = _abs(args.complex_csv)
    args.pdb_dir = _abs(args.pdb_dir)
    args.cache_dir = _abs(args.cache_dir) if args.cache_dir else None
    args.pretrained_ckpt = _abs(args.pretrained_ckpt)
    args.out_dir = _abs(args.out_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] complex_csv={args.complex_csv}")
    print(f"[smoke] pdb_dir={args.pdb_dir}")
    print(f"[smoke] cache_dir={args.cache_dir}")
    print(f"[smoke] out_dir={args.out_dir}")
    print(f"[smoke] n_iter={args.n_iter} batch_size={args.batch_size} lr={args.lr}")

    # Build a single batch once and freeze it for the entire smoke.
    print("[smoke] building batch...")
    batch, ds = build_batch(args.complex_csv, args.pdb_dir, args.cache_dir,
                            T=args.seq_len, batch_size=args.batch_size)
    print(f"[smoke] batch shapes: target_token={tuple(batch['target_token'].shape)} "
          f"warhead_atom_idx={batch['warhead_atom_idx'].tolist()} "
          f"warhead_valid={batch['warhead_valid'].tolist()}")
    print(f"[smoke] bd_target_coord={batch['bd_target_coord'].tolist()}")

    # Move to CPU explicitly (paranoid).
    for k, v in batch.items():
        if hasattr(v, "to"):
            batch[k] = v.to("cpu")

    results = {}
    if args.run == "both" or args.run == "A":
        summary_A = run_smoke("A_baseline", batch, args.pretrained_ckpt,
                              anchor_ce_w=0.0, anchor_w=0.0,
                              n_iter=args.n_iter, lr=args.lr,
                              out_dir=out_dir,
                              freeze_n=args.freeze_encoder_layers)
        results["A"] = summary_A
    if args.run == "both" or args.run == "B":
        summary_B = run_smoke("B_with_geom", batch, args.pretrained_ckpt,
                              anchor_ce_w=args.anchor_ce_w,
                              anchor_w=args.anchor_w,
                              n_iter=args.n_iter, lr=args.lr,
                              out_dir=out_dir,
                              freeze_n=args.freeze_encoder_layers)
        results["B"] = summary_B

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[smoke] summary written to {out_dir/'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--complex_csv",
                   default="data/covbinder/covind_smoke_subset.csv")
    p.add_argument("--pdb_dir",
                   default="data/covbinder/raw_covindb2/PDB")
    p.add_argument("--cache_dir",
                   default="data/lingo3dmol_geom_smoke_cache")
    p.add_argument("--pretrained_ckpt",
                   default="external/Lingo3DMol/checkpoint/gen_mol.pkl")
    p.add_argument("--out_dir",
                   default="results/lingo3dmol_geom_smoke")
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze_encoder_layers", type=int, default=3)
    p.add_argument("--anchor_ce_w", type=float, default=10.0)
    p.add_argument("--anchor_w", type=float, default=1.0)
    p.add_argument("--run", choices=["A", "B", "both"], default="both")
    args = p.parse_args()
    main(args)
