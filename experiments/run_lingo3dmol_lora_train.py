"""LoRA fine-tuning for Lingo3DMol — last attempt to make L1 FT produce
valid samples.

Three full-FT recipes already failed (CovalentLingo Full LR=5e-6, L1v2-A
LR=1e-6, L1v2-B frozen-coord-heads). All converged on training loss but
produced 0 valid samples — the coord distribution drifted off the
pretrained manifold.

Strategy: freeze the entire pretrained model, attach LoRA low-rank
adapters to attention q/k/v/o projections in encoder + decoder layers.
This preserves the pretrained weights UNCHANGED and only learns small
deltas to the attention maps. If even LoRA produces 0 mols, FT is
architecturally incompatible with this model.

The attention modules in `model/Module.py` are:
  - MultiHeadedAttention            : self.linears = ModuleList([q, k, v, o])
  - MultiHeadedAttention_att        : self.linears = ModuleList([q, k, v, o])
  - MultiHeadedAttentionBias        : self.linears = ModuleList([v, o]) (only 2)

We wrap every nn.Linear in these `linears` ModuleLists with a LoRALinear
adapter. The base weight stays frozen; only A and B (rank-r) and the
scalar α/r mix are trained.

Default config:  rank=8, alpha=16, dropout=0.05, lr=5e-4
"""
from __future__ import annotations
import os, sys, time, argparse, csv, json
from pathlib import Path

# CPU shim must be installed before any Lingo3DMol import.
ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))

# Decide device from argv early (matches run_lingo3dmol_l1_train).
_DEVICE_FROM_ARGV = "cpu"
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _DEVICE_FROM_ARGV = sys.argv[_i + 1]
        break
    if _a.startswith("--device="):
        _DEVICE_FROM_ARGV = _a.split("=", 1)[1]
        break

if _DEVICE_FROM_ARGV != "cuda":
    import lingo3dmol_cpu_shim  # noqa: F401
    # Extra shims the base file missed but PyTorch >= 2.4 needs for AdamW.
    # The optimizer's _accelerator_graph_capture_health_check calls
    # torch.cuda.is_current_stream_capturing() — needs to be faked to False.
    import torch as _torch
    _torch.cuda.is_current_stream_capturing = lambda: False
else:
    import numpy as _np_aliases
    for _name, _builtin in [("float", float), ("int", int), ("bool", bool),
                            ("long", int), ("object", object), ("str", str)]:
        if not hasattr(_np_aliases, _name):
            setattr(_np_aliases, _name, _builtin)
    print(f"[shim] device=cuda → skipping CPU shim, applied numpy aliases only")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.transformer_v1_res_fac2 import TransformerModel
from model.Module import (MultiHeadedAttention, MultiHeadedAttention_att,
                          MultiHeadedAttentionBias)


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    for base in (ORIG_CWD, ROOT):
        cand = (base / p).resolve()
        if cand.exists():
            return str(cand)
    return str((ORIG_CWD / p).resolve())


# ---------------------------------------------------------------------------
# LoRA implementation (inline — peft not installed in lingo3dmol env)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A LoRA-wrapped nn.Linear.

    Forward: y = base(x) + scale * dropout(x) @ A^T @ B^T
    The base linear is frozen (requires_grad=False). Only A, B are trained.
    A is initialized with Kaiming-uniform; B is initialized to 0 so that the
    adapter starts as a no-op (the model is identical to the pretrained one
    at step 0).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.05):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        self.base = base
        # Freeze the base weight + bias.
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)

        # Two small matrices: A (rank x in), B (out x rank). dtype follows base.
        wdtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, dtype=wdtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=wdtype))

        # Init A with Kaiming uniform (PEFT default), B with zeros so init delta=0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        # x: (..., in_features)
        h = self.dropout(x)
        # (..., in) @ (rank, in)^T -> (..., rank)
        h = F.linear(h, self.lora_A)
        # (..., rank) @ (out, rank)^T -> (..., out)
        h = F.linear(h, self.lora_B)
        return base_out + self.scale * h


def _wrap_linears_modulelist(linears: nn.ModuleList, rank: int, alpha: int,
                              dropout: float, targets_per_mod: int) -> int:
    """Wrap the first `targets_per_mod` nn.Linear children of `linears` with
    LoRALinear. Returns the count of wrapped linears.

    For MultiHeadedAttention (4 linears): wrap all 4 (q, k, v, o).
    For MultiHeadedAttentionBias (2 linears): wrap both (v, o).
    """
    n = 0
    for i in range(min(targets_per_mod, len(linears))):
        child = linears[i]
        if not isinstance(child, nn.Linear):
            continue
        linears[i] = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        n += 1
    return n


def apply_lora_to_attention(model: nn.Module, *, rank: int = 8, alpha: int = 16,
                            dropout: float = 0.05) -> dict:
    """Walk `model` and wrap every attention `linears` ModuleList with LoRA.

    Targets:
      - MultiHeadedAttention.linears        — q/k/v/o (all 4)
      - MultiHeadedAttention_att.linears    — q/k/v/o (all 4)
      - MultiHeadedAttentionBias.linears    — v/o (both)

    Returns a stats dict.
    """
    stats = {"mha": 0, "mha_att": 0, "mha_bias": 0, "linears_wrapped": 0}
    for mod in model.modules():
        if isinstance(mod, MultiHeadedAttention) and not isinstance(mod, MultiHeadedAttention_att):
            stats["mha"] += 1
            stats["linears_wrapped"] += _wrap_linears_modulelist(
                mod.linears, rank, alpha, dropout, targets_per_mod=4)
        elif isinstance(mod, MultiHeadedAttention_att):
            stats["mha_att"] += 1
            stats["linears_wrapped"] += _wrap_linears_modulelist(
                mod.linears, rank, alpha, dropout, targets_per_mod=4)
        elif isinstance(mod, MultiHeadedAttentionBias):
            stats["mha_bias"] += 1
            stats["linears_wrapped"] += _wrap_linears_modulelist(
                mod.linears, rank, alpha, dropout, targets_per_mod=2)
    return stats


def freeze_all_except_lora(model: nn.Module) -> dict:
    """Set requires_grad=False on every parameter that is NOT a LoRA adapter
    parameter. LoRA params are named '...lora_A' and '...lora_B'.
    """
    n_total = 0
    n_train = 0
    train_param_names = []
    for name, p in model.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        p.requires_grad = bool(is_lora)
        n_total += p.numel()
        if is_lora:
            n_train += p.numel()
            train_param_names.append(name)
    return {
        "n_total_params": n_total,
        "n_trainable_params": n_train,
        "trainable_pct": 100.0 * n_train / max(n_total, 1),
        "n_lora_params": len(train_param_names),
    }


def load_pretrained(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {ckpt_path}: missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")
    if info.missing_keys:
        print(f"[ckpt] first missing: {info.missing_keys[:3]}")
    if info.unexpected_keys:
        print(f"[ckpt] first unexpected: {info.unexpected_keys[:3]}")
    return info


def _do_step(model, batch, optimizer, *, anchor_kwargs=None):
    """One optimizer step (identical to run_lingo3dmol_l1_train._do_step)."""
    optimizer.zero_grad()
    fwd_batch = dict(batch)
    if anchor_kwargs:
        fwd_batch.update(anchor_kwargs)
    else:
        fwd_batch["anchor_geom_weight"] = 0.0
        fwd_batch["anchor_geom_ce_w"] = 0.0
    out = model.forward_train(**fwd_batch)
    loss = out["loss_total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), max_norm=5.0)
    optimizer.step()
    rec = {
        "loss_total": float(out["loss_total"].detach()),
        "loss_token": float(out["loss_token"]),
        "loss_x":     float(out["loss_x"]),
        "loss_y":     float(out["loss_y"]),
        "loss_z":     float(out["loss_z"]),
    }
    for k in ("loss_r", "loss_theta", "loss_phi"):
        if k in out:
            rec[k] = float(out[k])
    return rec


def train_loop_streaming(model, loader, optimizer, args, out_dir):
    csv_path = Path(out_dir) / "train_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "wall_sec", "loss_total", "loss_token",
                    "loss_x", "loss_y", "loss_z", "loss_r", "loss_theta",
                    "loss_phi", "complexes_per_sec"])

    model.train()
    best_loss = float("inf")
    history = []
    global_step = 0
    t_start = time.time()
    device = next(model.parameters()).device

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_steps = 0
        for batch in loader:
            t_step = time.time()
            B = batch["target_token"].shape[0]
            for k, v in batch.items():
                if hasattr(v, "to"):
                    batch[k] = v.to(device)
            m = _do_step(model, batch, optimizer)
            dt_step = time.time() - t_step
            cps = B / max(dt_step, 1e-6)
            row = [epoch, global_step, time.time() - t_start,
                   m["loss_total"], m["loss_token"], m["loss_x"],
                   m["loss_y"], m["loss_z"],
                   m.get("loss_r", float("nan")),
                   m.get("loss_theta", float("nan")),
                   m.get("loss_phi", float("nan")),
                   cps]
            history.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if global_step % 5 == 0:
                rtp_str = ""
                if "loss_r" in m:
                    rtp_str = (f" r={m['loss_r']:.3f} "
                               f"t={m['loss_theta']:.3f} p={m['loss_phi']:.3f}")
                print(f"  ep={epoch} step={global_step:4d} t={row[2]:6.1f}s "
                      f"loss={m['loss_total']:.4f} (tok={m['loss_token']:.3f} "
                      f"x={m['loss_x']:.3f} y={m['loss_y']:.3f} "
                      f"z={m['loss_z']:.3f}{rtp_str}) bs={B} cps={cps:.2f}")
            if m["loss_total"] < best_loss:
                best_loss = m["loss_total"]
            global_step += 1
            epoch_steps += 1
        dt_epoch = time.time() - epoch_t0
        print(f"[epoch {epoch}] {epoch_steps} steps in {dt_epoch:.1f}s "
              f"({epoch_steps / max(dt_epoch, 1e-6):.2f} step/s)")
        # Save full state_dict for compatibility with run_lingo3dmol_l1_ft_sample.
        ep_ckpt = Path(out_dir) / f"ckpt_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_loss": best_loss}, ep_ckpt)
        print(f"[epoch {epoch}] saved {ep_ckpt}")

    final = Path(out_dir) / "ckpt_final.pt"
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "history_tail": history[-200:],
                "best_loss": best_loss}, final)
    print(f"[done] best loss={best_loss:.4f}  ckpt={final}")
    return best_loss, history


def main(args):
    args.pretrained_ckpt = _abs(args.pretrained_ckpt)
    args.out_dir = _abs(args.out_dir)
    if args.complex_csv:
        args.complex_csv = _abs(args.complex_csv)
    if args.pocket_pdb_dir:
        args.pocket_pdb_dir = _abs(args.pocket_pdb_dir)
    if args.cache_dir:
        args.cache_dir = _abs(args.cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[LoRA train] device={args.device}")
    print(f"[LoRA train] pretrained={args.pretrained_ckpt}")
    print(f"[LoRA train] out_dir={args.out_dir}")
    print(f"[LoRA train] rank={args.lora_rank} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout} lr={args.lr}")

    # 1) Build model + load pretrained ckpt FIRST (before LoRA wraps).
    model = TransformerModel()
    load_pretrained(model, args.pretrained_ckpt)

    # 2) Apply LoRA wrappers to attention projections.
    lora_stats = apply_lora_to_attention(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    print(f"[LoRA] wrap stats: {lora_stats}")

    # 3) Freeze everything except LoRA params.
    freeze_stats = freeze_all_except_lora(model)
    print(f"[LoRA] total params: {freeze_stats['n_total_params']/1e6:.2f}M  "
          f"trainable: {freeze_stats['n_trainable_params']/1e6:.4f}M  "
          f"({freeze_stats['trainable_pct']:.3f}%)")

    # Persist setup stats early — useful even if training crashes.
    setup_path = Path(args.out_dir) / "lora_setup.json"
    with open(setup_path, "w") as f:
        json.dump({
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "wrap_stats": lora_stats,
            "freeze_stats": freeze_stats,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }, f, indent=2)

    # 4) Device transfer.
    target_device = torch.device(args.device)
    try:
        model = model.to(target_device)
        print(f"[device] model moved to {target_device}")
    except Exception as e:
        print(f"[device] {target_device} failed: {e}; falling back to CPU")
        target_device = torch.device("cpu")
        model = model.to(target_device)
    args.device = str(target_device)

    # 5) Optimizer — ONLY over LoRA params.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable params after LoRA wrap+freeze — bug.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # 6) Build dataloader (streaming CovInDB 2.0). Smoke override uses
    #    --max_complexes 20.
    if not args.complex_csv or not args.pocket_pdb_dir:
        raise ValueError("Need --complex_csv and --pocket_pdb_dir.")

    from lingo3dmol_l1_dataloader import (
        CovalentInDB2Dataset, collate_fn,
    )
    from torch.utils.data import DataLoader

    cache_dir = args.cache_dir or str(Path(args.out_dir) / "cache")
    dataset = CovalentInDB2Dataset(
        complex_csv_path=args.complex_csv,
        pdb_dir=args.pocket_pdb_dir,
        T=args.seq_len,
        max_complexes=args.max_complexes,
        cache_dir=cache_dir,
        pocket_radius=args.pocket_radius,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after precheck.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    best_loss, history = train_loop_streaming(
        model, loader, optimizer, args, args.out_dir)

    # Write summary.
    summary = {
        "best_loss": best_loss,
        "n_steps": len(history),
        "elapsed_sec": history[-1][2] if history else 0.0,
        "initial_loss": history[0][3] if history else None,
        "final_loss": history[-1][3] if history else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_complexes": args.max_complexes,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "freeze_stats": freeze_stats,
        "wrap_stats": lora_stats,
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary = {summary}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_ckpt", default="checkpoint/gen_mol.pkl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--complex_csv", required=True)
    p.add_argument("--pocket_pdb_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_complexes", type=int, default=None,
                   help="Cap dataset size (use 20 for local smoke).")
    p.add_argument("--pocket_radius", type=float, default=15.0)
    # LoRA hyperparameters.
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    args = p.parse_args()
    main(args)
