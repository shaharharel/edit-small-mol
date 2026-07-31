"""L1 Phase 1 — Lingo3DMol covalent fine-tune scaffold (CPU only).

Loads `checkpoint/gen_mol.pkl`, freezes the encoder layers per the paper's
fine-tune recipe, and runs AdamW over the remaining parameters using a
new `TransformerModel.forward_train()` method. Phase 1 covers 4 of the 8
paper losses (token + X + Y + Z); r/theta/phi + aux heads are TODO.

Use the companion `run_lingo3dmol_l1_smoke.py` to drive a one-batch
overfit test on 4 acrylamide ligands.
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

# The CPU shim makes .cuda() a no-op so the legacy Lingo3DMol model code
# runs on Mac CPU. On a real CUDA box that shim would leave .cuda()-created
# tensors on CPU -> device mismatch with the model on GPU. Inspect argv
# BEFORE any torch use so we can skip the shim on cuda runs.
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
else:
    # Still need the numpy-alias half of the shim for legacy code (np.float etc.).
    import numpy as _np_aliases
    for _name, _builtin in [("float", float), ("int", int), ("bool", bool),
                            ("long", int), ("object", object), ("str", str)]:
        if not hasattr(_np_aliases, _name):
            setattr(_np_aliases, _name, _builtin)
    print(f"[shim] device=cuda → skipping CPU shim, applied numpy aliases only")

import torch
import torch.nn as nn
import numpy as np

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
    """Freeze encoder.layers.{0..n_freeze-1}.* and report counts.

    The encoder stack is `model.encoder.layers` (3 layers, see TransformerModel
    __init__). With n_freeze=3 the entire encoder stack is frozen (matches
    paper's fine-tune recipe of freezing the first 3 layers).
    """
    prefixes = tuple(f"encoder.layers.{i}." for i in range(n_freeze))
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = False
            frozen += p.numel()
        else:
            trainable += p.numel()
    total = frozen + trainable
    print(f"[freeze] prefixes={prefixes}")
    print(f"[freeze] total={total/1e6:.2f}M  frozen={frozen/1e6:.2f}M  "
          f"trainable={trainable/1e6:.2f}M")
    return total, frozen, trainable


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) -- Exp 3
# ---------------------------------------------------------------------------
# Wraps an existing nn.Linear with a frozen base + trainable rank-r update:
#     y = base(x) + (x @ A^T) @ B^T * (alpha / r)
# where A is (r, in_features), B is (out_features, r). The base linear's
# weights are frozen; only A and B receive gradient. Initialised so that
# the LoRA delta is zero at init (B = 0).
class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        out = out + (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return out


def _replace_linear_with_lora(parent: nn.Module, name: str, rank: int,
                              alpha: float):
    """Replace `parent.<name>` (an nn.Linear) with a LoraLinear wrapper."""
    orig = getattr(parent, name)
    if not isinstance(orig, nn.Linear):
        return False
    setattr(parent, name, LoraLinear(orig, rank=rank, alpha=alpha))
    return True


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Apply LoRA to all attention nn.Linear layers in encoder + decoders.

    Targets:
      - encoder.layers[i].self_attn.linears  (MultiHeadedAttention, 4 linears)
      - decoder.layers[i].self_attn.linears  (MultiHeadedAttentionBias, 2)
      - decoder.layers[i].src_attn.linears   (MultiHeadedAttention_att, 4)
      - decoder_relative.layers[i].self_attn.linears (2)
      - decoder_relative.layers[i].src_attn.linears  (4)

    The `linears` attribute is an nn.ModuleList (`clones(...)`), so we
    replace each element in-place. Then freeze everything outside the
    LoRA params + the task heads (reactivity_head, chassis_head if any).
    """
    n_replaced = 0

    def patch_modulelist(attn_module):
        nonlocal n_replaced
        if not hasattr(attn_module, "linears"):
            return
        for i, lin in enumerate(attn_module.linears):
            if isinstance(lin, nn.Linear):
                attn_module.linears[i] = LoraLinear(lin, rank=rank, alpha=alpha)
                n_replaced += 1

    # Encoder
    for layer in getattr(model.encoder, "layers", []):
        if hasattr(layer, "self_attn"):
            patch_modulelist(layer.self_attn)

    # Decoders
    for dec_name in ("decoder", "decoder_relative"):
        dec = getattr(model, dec_name, None)
        if dec is None:
            continue
        for layer in getattr(dec, "layers", []):
            if hasattr(layer, "self_attn"):
                patch_modulelist(layer.self_attn)
            if hasattr(layer, "src_attn"):
                patch_modulelist(layer.src_attn)

    print(f"[lora] replaced {n_replaced} attention nn.Linear layers "
          f"(rank={rank}, alpha={alpha})")

    # Freeze everything except LoRA params (lora_A, lora_B) and the new
    # task heads (reactivity_head, chassis_head). Task heads must remain
    # trainable because they are FRESH (missing from pretrained ckpt).
    keep_trainable_prefixes = (
        "reactivity_head.",
        "chassis_head.",
    )
    frozen, trainable_lora, trainable_head = 0, 0, 0
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad = True
            trainable_lora += p.numel()
        elif name.startswith(keep_trainable_prefixes):
            p.requires_grad = True
            trainable_head += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    total = frozen + trainable_lora + trainable_head
    print(f"[lora] total={total/1e6:.2f}M  frozen={frozen/1e6:.2f}M  "
          f"lora={trainable_lora/1e6:.3f}M  heads={trainable_head/1e6:.3f}M")
    return n_replaced, trainable_lora, trainable_head


def load_pretrained(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = model.load_state_dict(sd, strict=False)
    n_missing = len(info.missing_keys)
    n_unexp = len(info.unexpected_keys)
    print(f"[ckpt] loaded {ckpt_path}: missing={n_missing} unexpected={n_unexp}")
    if n_missing:
        print(f"[ckpt] first missing: {info.missing_keys[:3]}")
    if n_unexp:
        print(f"[ckpt] first unexpected: {info.unexpected_keys[:3]}")
    return info


def _do_step(model, batch, optimizer, *, anchor_kwargs=None):
    """One optimizer step. Returns the loss dict (detached, scalars).

    `anchor_kwargs` is an optional dict of {anchor_geom_weight, anchor_geom_ce_w}
    that, when passed, enables the L_anchor_geometry contribution to
    loss_total. (Bookkeeping fields bd_target_coord / warhead_atom_idx /
    warhead_valid already arrive in `batch` from the dataloader.) When None
    or empty, forward_train still computes the geometry diagnostics but the
    loss contribution is gated by `anchor_geom_ce_w=0, anchor_geom_weight=0`.
    """
    optimizer.zero_grad()
    fwd_batch = dict(batch)
    if anchor_kwargs:
        fwd_batch.update(anchor_kwargs)
    # Defaults for anchor-geom args (off when not enabled).
    fwd_batch.setdefault("anchor_geom_weight", 0.0)
    fwd_batch.setdefault("anchor_geom_ce_w", 0.0)
    # Exp 5 — pose-aware token reweighting defaults (off when not enabled).
    fwd_batch.setdefault("pose_aware_token_weight", 1.0)
    fwd_batch.setdefault("pose_aware_n_tokens", 8)
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
    for k in ("loss_anchor_geom", "loss_warhead_ce",
              "warhead_hard_dist", "warhead_n_valid"):
        if k in out:
            rec[k] = float(out[k])
    return rec


def _build_anchor_kwargs(args):
    """Return kwargs dict for L_anchor_geometry + pose-aware token weight.

    Always returns a dict (possibly empty) so we can carry the pose-aware
    weight even when anchor-geometry is off.
    """
    out: dict = {}
    if getattr(args, "enable_anchor_geometry_loss", False):
        out["anchor_geom_weight"] = float(args.anchor_geom_weight)
        out["anchor_geom_ce_w"]   = float(args.anchor_geom_ce_w)
    # Pose-aware token weight is always forwarded (default 1.0 = no-op).
    paw = float(getattr(args, "pose_aware_token_weight", 1.0))
    pan = int(getattr(args, "pose_aware_n_tokens", 8))
    if paw != 1.0:
        out["pose_aware_token_weight"] = paw
        out["pose_aware_n_tokens"]      = pan
    return out if out else None


def train_loop(model, batch, optimizer, args, out_dir):
    """SMOKE-MODE training: repeats the SAME batch for args.epochs iterations
    to verify overfit + gradient flow. Used only when --smoke is set; the
    Phase 2 streaming path is in `train_loop_streaming`."""
    csv_path = Path(out_dir) / "train_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_sec", "loss_total", "loss_token",
                    "loss_x", "loss_y", "loss_z",
                    "loss_anchor_geom", "loss_warhead_ce",
                    "warhead_hard_dist", "warhead_n_valid"])

    n_iter = args.epochs  # in smoke mode this is the number of overfit iters
    t0 = time.time()
    model.train()
    best_loss = float("inf")
    anchor_kwargs = _build_anchor_kwargs(args)

    history = []
    for step in range(n_iter):
        m = _do_step(model, batch, optimizer, anchor_kwargs=anchor_kwargs)
        row = [step, time.time() - t0,
               m["loss_total"], m["loss_token"],
               m["loss_x"], m["loss_y"], m["loss_z"],
               m.get("loss_anchor_geom", float("nan")),
               m.get("loss_warhead_ce", float("nan")),
               m.get("warhead_hard_dist", float("nan")),
               m.get("warhead_n_valid", float("nan"))]
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if step % max(1, n_iter // 20) == 0 or step == n_iter - 1:
            geo_str = ""
            if "loss_anchor_geom" in m:
                geo_str = (f"  geom={m['loss_anchor_geom']:.1f} "
                           f"wh_ce={m.get('loss_warhead_ce', float('nan')):.2f} "
                           f"hd={m.get('warhead_hard_dist', float('nan')):.1f} "
                           f"nv={int(m.get('warhead_n_valid', 0))}")
            print(f"  step={step:3d} t={row[1]:6.1f}s  loss_total={row[2]:.4f} "
                  f"(tok={row[3]:.3f} x={row[4]:.3f} y={row[5]:.3f} z={row[6]:.3f})"
                  f"{geo_str}")

        if m["loss_total"] < best_loss:
            best_loss = m["loss_total"]

    # Save final checkpoint (smoke mode = final overfit)
    ckpt_out = Path(out_dir) / "ckpt_best.pt"
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "history": history,
                "best_loss": best_loss}, ckpt_out)
    print(f"[done] best loss={best_loss:.4f}  ckpt={ckpt_out}  log={csv_path}")
    return best_loss, history


def train_loop_streaming(model, loader, optimizer, args, out_dir):
    """PHASE 2 streaming training: epoch-loop over a DataLoader yielding
    real (pocket, ligand) batches from CovalentInDB 2.0. Logs per-step
    loss + per-epoch throughput, projects T4 wall time at scale."""
    csv_path = Path(out_dir) / "train_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "wall_sec", "loss_total", "loss_token",
                    "loss_x", "loss_y", "loss_z", "loss_r", "loss_theta",
                    "loss_phi", "loss_anchor_geom", "loss_warhead_ce",
                    "warhead_hard_dist", "warhead_n_valid", "complexes_per_sec"])

    model.train()
    best_loss = float("inf")
    history = []
    global_step = 0
    t_start = time.time()
    device = next(model.parameters()).device
    anchor_kwargs = _build_anchor_kwargs(args)
    if anchor_kwargs:
        if "anchor_geom_weight" in anchor_kwargs:
            print(f"[geom] L_anchor_geometry enabled: weight={anchor_kwargs['anchor_geom_weight']} "
                  f"ce_w={anchor_kwargs['anchor_geom_ce_w']}")
        if "pose_aware_token_weight" in anchor_kwargs:
            print(f"[pose] Exp 5 pose-aware token weight enabled: "
                  f"weight={anchor_kwargs['pose_aware_token_weight']} "
                  f"n_tokens={anchor_kwargs['pose_aware_n_tokens']}")

    # Throughput tracking across all steps (excluding the first to avoid cold cache).
    per_step_secs: list[float] = []
    per_step_bs: list[int] = []

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_steps = 0
        for batch in loader:
            t_step = time.time()
            B = batch["target_token"].shape[0]
            # Move tensors to device
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(device)
            m = _do_step(model, batch, optimizer, anchor_kwargs=anchor_kwargs)
            dt_step = time.time() - t_step
            per_step_secs.append(dt_step)
            per_step_bs.append(B)
            cps = B / max(dt_step, 1e-6)

            row = [epoch, global_step, time.time() - t_start,
                   m["loss_total"], m["loss_token"], m["loss_x"],
                   m["loss_y"], m["loss_z"],
                   m.get("loss_r", float("nan")),
                   m.get("loss_theta", float("nan")),
                   m.get("loss_phi", float("nan")),
                   m.get("loss_anchor_geom", float("nan")),
                   m.get("loss_warhead_ce", float("nan")),
                   m.get("warhead_hard_dist", float("nan")),
                   m.get("warhead_n_valid", float("nan")),
                   cps]
            history.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if global_step % 5 == 0:
                rtp_str = ""
                if "loss_r" in m:
                    rtp_str = (f" r={m['loss_r']:.3f} "
                               f"t={m['loss_theta']:.3f} p={m['loss_phi']:.3f}")
                geo_str = ""
                if "loss_anchor_geom" in m:
                    geo_str = (f" geom={m['loss_anchor_geom']:.1f} "
                               f"wh_ce={m.get('loss_warhead_ce', float('nan')):.2f} "
                               f"hd={m.get('warhead_hard_dist', float('nan')):.1f} "
                               f"nv={int(m.get('warhead_n_valid', 0))}")
                print(f"  ep={epoch} step={global_step:4d} t={row[2]:6.1f}s "
                      f"loss={m['loss_total']:.4f} (tok={m['loss_token']:.3f} "
                      f"x={m['loss_x']:.3f} y={m['loss_y']:.3f} "
                      f"z={m['loss_z']:.3f}{rtp_str}{geo_str}) bs={B} cps={cps:.2f}")
            if m["loss_total"] < best_loss:
                best_loss = m["loss_total"]
            global_step += 1
            epoch_steps += 1
        dt_epoch = time.time() - epoch_t0
        print(f"[epoch {epoch}] {epoch_steps} steps in {dt_epoch:.1f}s "
              f"({epoch_steps / max(dt_epoch, 1e-6):.2f} step/s)")
        # Per-epoch checkpoint
        ep_ckpt = Path(out_dir) / f"ckpt_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_loss": best_loss}, ep_ckpt)
        print(f"[epoch {epoch}] saved {ep_ckpt}")

    # T4 wall-time projection
    if len(per_step_secs) > 1:
        # drop first step (warmup)
        secs_arr = np.asarray(per_step_secs[1:])
        bs_arr = np.asarray(per_step_bs[1:])
        complexes_per_sec_cpu = (bs_arr.sum()) / max(secs_arr.sum(), 1e-6)
        # T4 speedup vs Mac CPU for FW+BW of a 47M-param transformer is
        # empirically 5x-15x (depends on op mix; FlashAttention bumps it).
        # We DO NOT scale further by batch — GPU saturation at bs=8-16 has
        # only a modest effect on per-sample throughput for this model size.
        t4_low, t4_high = 5.0, 15.0
        scale_pairs = 3400 * 3   # 3.4K complexes × 3 epochs
        avg_bs_dev = float(bs_arr.mean())
        # WORST-CASE (conservative) and BEST-CASE projections.
        wall_worst = scale_pairs / (complexes_per_sec_cpu * t4_low) / 3600.0
        wall_best  = scale_pairs / (complexes_per_sec_cpu * t4_high) / 3600.0
        proj = {
            "dev_cps_cpu": float(complexes_per_sec_cpu),
            "dev_avg_bs": avg_bs_dev,
            "t4_speedup_low": t4_low,
            "t4_speedup_high": t4_high,
            "scale_complexes": scale_pairs,
            "t4_bs_target": 16,
            "wall_hours_worst_case": float(wall_worst),
            "wall_hours_best_case":  float(wall_best),
        }
        print(f"[proj] dev: {complexes_per_sec_cpu:.3f} cplx/s on CPU @ "
              f"avg_bs={avg_bs_dev:.1f}")
        print(f"[proj] T4 wall (3.4K cplx × 3 epochs, 5-15x speedup): "
              f"{wall_best:.2f}-{wall_worst:.2f} hours")
        if wall_worst > 24.0:
            print(f"[proj] !!! WORST-CASE > 24h — recommend A100 over T4")
    else:
        proj = {}

    ckpt_out = Path(out_dir) / "ckpt_phase2_dev.pt"
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "history_tail": history[-200:],
                "best_loss": best_loss,
                "projection": proj}, ckpt_out)
    print(f"[done] best loss={best_loss:.4f}  ckpt={ckpt_out}  log={csv_path}")
    return best_loss, history, proj


def main(args):
    args.pretrained_ckpt = _abs(args.pretrained_ckpt)
    if args.data_csv:
        args.data_csv = _abs(args.data_csv)
    args.out_dir = _abs(args.out_dir)
    if args.pocket_pdb:
        args.pocket_pdb = _abs(args.pocket_pdb)
    if args.complex_csv:
        args.complex_csv = _abs(args.complex_csv)
    if args.pocket_pdb_dir:
        args.pocket_pdb_dir = _abs(args.pocket_pdb_dir)
    if args.cache_dir:
        args.cache_dir = _abs(args.cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[L1 train] device={args.device} smoke={args.smoke}")
    print(f"[L1 train] pretrained={args.pretrained_ckpt}")
    print(f"[L1 train] out_dir={args.out_dir}")

    # -- Build model + load pretrained
    model = TransformerModel()
    load_pretrained(model, args.pretrained_ckpt)
    if int(getattr(args, "lora_rank", 0)) > 0:
        # LoRA mode: freeze everything, apply low-rank adapters to attention
        # nn.Linear layers + keep new task heads trainable.
        apply_lora(model, rank=int(args.lora_rank),
                   alpha=float(getattr(args, "lora_alpha", 16.0)))
    else:
        freeze_encoder_layers(model, args.freeze_encoder_layers)
    # Device transfer — MPS supported (with silent CPU fallback on op failure).
    target_device = torch.device(args.device)
    try:
        model = model.to(target_device)
        print(f"[device] model moved to {target_device}")
    except Exception as e:
        print(f"[device] {target_device} transfer failed: {e}; falling back to CPU")
        target_device = torch.device("cpu")
        model = model.to(target_device)
    args.device = str(target_device)

    # -- Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # -- Build training batch / loader
    if args.smoke:
        from run_lingo3dmol_l1_smoke import build_smoke_batch
        batch = build_smoke_batch(args.data_csv, args.batch_size,
                                  pocket_pdb=args.pocket_pdb,
                                  T=args.seq_len)
        # Device transfer for smoke batch
        for k, v in batch.items():
            if hasattr(v, 'to'):
                batch[k] = v.to(target_device)
        best_loss, history = train_loop(model, batch, optimizer, args, args.out_dir)
        proj = {}
    else:
        # PHASE 2: real CovInDB streaming.
        if not args.complex_csv or not args.pocket_pdb_dir:
            raise ValueError(
                "Phase 2 streaming requires --complex_csv and --pocket_pdb_dir.")
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
            raise RuntimeError("Dataset is empty after precheck — see logs.")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        )
        best_loss, history, proj = train_loop_streaming(
            model, loader, optimizer, args, args.out_dir)

        # Persist dataset skip stats.
        with open(Path(args.out_dir) / "dataset_stats.json", "w") as f:
            json.dump({
                "complex_csv": args.complex_csv,
                "pdb_dir": args.pocket_pdb_dir,
                "precheck_skips": dict(dataset.precheck_skips),
                "getitem_skips": dict(dataset.getitem_skips),
                "n_usable_precheck": len(dataset),
                "max_complexes": args.max_complexes,
                "pocket_radius": args.pocket_radius,
                "T": args.seq_len,
            }, f, indent=2)

    # Summary JSON
    if args.smoke:
        summary = {
            "best_loss": best_loss,
            "n_steps": len(history),
            "elapsed_sec": history[-1][1] if history else 0.0,
            "initial_loss": history[0][2] if history else None,
            "final_loss": history[-1][2] if history else None,
            "smoke": args.smoke,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "freeze_encoder_layers": args.freeze_encoder_layers,
        }
    else:
        summary = {
            "best_loss": best_loss,
            "n_steps": len(history),
            "elapsed_sec": history[-1][2] if history else 0.0,
            "initial_loss": history[0][3] if history else None,
            "final_loss": history[-1][3] if history else None,
            "smoke": args.smoke,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "freeze_encoder_layers": args.freeze_encoder_layers,
            "max_complexes": args.max_complexes,
            "pocket_radius": args.pocket_radius,
            "lora_rank": int(getattr(args, "lora_rank", 0)),
            "lora_alpha": float(getattr(args, "lora_alpha", 16.0)),
            "projection": proj,
        }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", default=None,
                   help="CSV with `smiles` column (only smiles used in smoke). "
                   "Optional in streaming mode.")
    p.add_argument("--pretrained_ckpt", default="checkpoint/gen_mol.pkl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=100,
                   help="In smoke mode: number of overfit iterations.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--freeze_encoder_layers", type=int, default=3)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="Overfit a single 4-mol batch for `epochs` iterations.")
    p.add_argument("--pocket_pdb",
                   default="data/lingo3dmol_smoke/zap70_pocket_cys346.pdb",
                   help="Stub pocket PDB used for the smoke batch.")
    p.add_argument("--seq_len", type=int, default=60,
                   help="Padded FSMILES sequence length T for the smoke batch.")
    # Phase 2 streaming dataloader args
    p.add_argument("--complex_csv", default=None,
                   help="Phase 2: CovInDB 2.0 Covalent_Complex_Records.csv.")
    p.add_argument("--pocket_pdb_dir", default=None,
                   help="Phase 2: directory of cocrystal PDB files "
                   "(file naming: <PDB_ID>.pdb).")
    p.add_argument("--cache_dir", default=None,
                   help="Phase 2: directory for cached preprocessed samples.")
    p.add_argument("--max_complexes", type=int, default=None,
                   help="Phase 2: cap dataset size for dev runs "
                   "(e.g. 20 for the throughput probe).")
    p.add_argument("--pocket_radius", type=float, default=15.0,
                   help="Phase 2: pocket crop radius (Å) around ligand atoms.")
    # L_anchor_geometry training loss (CovalentLingo Full v1)
    p.add_argument("--enable_anchor_geometry_loss", action="store_true",
                   help="If set, L_anchor_geometry contributes to loss_total.")
    p.add_argument("--anchor_geom_weight", type=float, default=0.1,
                   help="MSE weight (ml-architect rec: <=0.1 to prevent "
                        "encoder reshape). Scaled internally by 1/240^2.")
    p.add_argument("--anchor_geom_ce_w", type=float, default=0.1,
                   help="CE upweight at the warhead position (X/Y/Z heads). "
                        "Default 0.1 — loss_wh_ce is a per-warhead-averaged "
                        "X+Y+Z CE (~10-15 per term, ~30-45 total), so ce_w=0.1 "
                        "yields ~3-5 contribution, commensurate with loss_x. "
                        "Higher (e.g. 1.0+) overwhelms the topology losses.")
    # Exp 5 — Pose-aware token reweighting
    p.add_argument("--pose_aware_token_weight", type=float, default=1.0,
                   help="Multiplier on warhead-prefix tokens in loss_token. "
                        "1.0 = off (default). Set to e.g. 5.0 to upweight "
                        "the first --pose_aware_n_tokens GT positions "
                        "(covers the C=CC(=O)N pattern + start/sep).")
    p.add_argument("--pose_aware_n_tokens", type=int, default=8,
                   help="How many prefix GT positions are upweighted by "
                        "--pose_aware_token_weight. Default 8.")
    # Exp 3 — LoRA low-rank adaptation
    p.add_argument("--lora_rank", type=int, default=0,
                   help="If >0, freeze the pretrained model and add LoRA "
                        "adapters of this rank to all attention nn.Linear "
                        "layers (q/k/v/out in encoder + decoder self/cross "
                        "attention). New task heads remain trainable. "
                        "Overrides --freeze_encoder_layers when active.")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA scaling alpha (default 16). Effective scaling "
                        "is alpha / rank.")
    args = p.parse_args()

    main(args)
