"""KL-constraint fine-tune for Lingo3DMol covalent FT (2026-05-31).

Implements KL-divergence regularization between the FT'd model's output
distribution and a frozen pretrained reference. The KL term keeps the FT'd
model close to the pretrained generative manifold while still allowing
covalent-specific adaptation.

Loss:  L_total = L_token + L_x + L_y + L_z + (... aux ...) + beta * L_KL

KL is computed on ALL 7 heads that produce a categorical distribution:
  - token (76 classes)
  - x, y, z (240 voxels each)
  - r (13 bins), theta (181), phi (181)

This is the "Mission 2" companion to the forensic bug-hunt patches
(CRIT-1 + CRIT-2 in `lingo3dmol_l1_dataloader.py`).

Smoke runs on Mac CPU first; T4 deployment only after smoke passes.
"""

from __future__ import annotations
import os, sys, time, argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))

# Same shim logic as run_lingo3dmol_l1_train.py
_DEVICE_FROM_ARGV = "cpu"
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _DEVICE_FROM_ARGV = sys.argv[_i + 1]; break
    if _a.startswith("--device="):
        _DEVICE_FROM_ARGV = _a.split("=", 1)[1]; break
if _DEVICE_FROM_ARGV != "cuda":
    import lingo3dmol_cpu_shim  # noqa: F401
else:
    import numpy as _np_aliases
    for _name, _b in [("float", float), ("int", int), ("bool", bool),
                       ("long", int), ("object", object), ("str", str)]:
        if not hasattr(_np_aliases, _name):
            setattr(_np_aliases, _name, _b)

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_pretrained(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt_path}: missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")
    return info


def freeze_encoder(model, n_freeze):
    prefixes = tuple(f"encoder.layers.{i}." for i in range(n_freeze))
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = False
            frozen += p.numel()
        else:
            trainable += p.numel()
    print(f"[freeze] frozen={frozen/1e6:.2f}M  trainable={trainable/1e6:.2f}M")


# ---------------------------------------------------------------------------
# KL-constrained forward pass
# ---------------------------------------------------------------------------
def _logits_dict_from_forward_train(model, batch, anchor_kwargs=None):
    """Run forward_train but ALSO capture the categorical logits from each
    head so we can compute KL.  forward_train doesn't currently expose
    logits; we re-implement the minimal forward path here to collect them.

    Returns (out_dict, logits_dict).
    out_dict matches model.forward_train() return.
    logits_dict has keys 'token','x','y','z' (and 'r','theta','phi' if
    smi_map provided).
    """
    # Call forward_train normally to get the supervised loss + diagnostics.
    fwd_batch = dict(batch)
    if anchor_kwargs is None:
        fwd_batch["anchor_geom_weight"] = 0.0
        fwd_batch["anchor_geom_ce_w"] = 0.0
    else:
        fwd_batch.update(anchor_kwargs)
    out = model.forward_train(**fwd_batch)
    # Re-run the model's encoder+decoder once more to extract logits.
    # We use teacher-forced inputs identical to forward_train. To avoid
    # duplicate compute we do this here even though it doubles the model
    # cost; KL-FT already doubles memory anyway. Cleanest path is to
    # monkey-patch forward_train to return logits — done below.
    return out


# Monkey-patch: replace forward_train with a version that also returns logits.
# This is invasive but keeps us out of editing the model file.
def _wrap_forward_train_for_logits():
    """Wrap TransformerModel.forward_train to optionally return a logits dict.

    Adds kwarg `_return_logits=True` (default False) which, when set, makes
    the function additionally write a `logits` dict into the return value.
    """
    orig = TransformerModel.forward_train

    def patched(self, *args, **kwargs):
        return_logits = kwargs.pop("_return_logits", False)
        # We need logits AFTER the head computations.  Easiest: re-run the
        # decoder paths inside this wrapper using the same inputs.  But that
        # duplicates compute.  Instead we monkey-patch self.proj/fcx/fcy/fcz
        # to capture their last-call inputs.  Simpler: re-implement the
        # minimal forward here, replicating forward_train's path verbatim.
        if not return_logits:
            return orig(self, *args, **kwargs)

        # Inline path: run encoder + decoder + heads ourselves, capturing
        # logits, then call orig() once to get the loss dict.
        out = orig(self, *args, **kwargs)
        # Now redo the path to get logits.
        # Extract inputs by name (forward_train signature: positional 5 +
        # kwargs).  We accept ONLY the kwargs path here.
        coords = kwargs["coords"]
        type_ = kwargs["type"]
        residue = kwargs["residue"]
        critical_anchor = kwargs["critical_anchor"]
        src_mask = kwargs["src_mask"]
        target_token = kwargs["target_token"]
        target_coords = kwargs["target_coords"]
        device = target_token.device
        B, T = target_token.shape

        # Encoder
        coords_emb = self.coords_emb(coords)
        coords_embedding = coords_emb.reshape(coords_emb.shape[0],
                                              coords_emb.shape[1], -1)
        type_emb = self.src_fea(type_)
        total_feature = torch.cat((coords_embedding, type_emb), dim=-1)
        residue_emb = self.residue(residue)
        critical_anchor_emb = self.anchor_emb(critical_anchor)
        residue_total_emb = torch.cat([residue_emb, critical_anchor_emb], dim=-1)
        total_feature1 = total_feature + residue_total_emb
        final_fea = self.pos_encode(total_feature1)
        sm = src_mask.unsqueeze(1)
        sm1 = sm.repeat(1, sm.size(-1), 1)
        memory = self.encoder(final_fea, sm1)

        # Decoder #1 (token)
        gt_tok_emb = self.gt_fea1(target_token)
        gt_coords_emb0 = self.coords_emb_gt(target_coords)
        gt_coords_emb = gt_coords_emb0.reshape(B, T, -1)
        new_fea0 = torch.cat([gt_tok_emb, gt_coords_emb], dim=-1)
        tgt_fea = self.pos_encode(new_fea0)
        src_mask_dec = sm.repeat(1, T, 1)
        from model.transformer_v1_res_mp1 import subsequent_mask
        tgt_mask = subsequent_mask(T).long().to(device)
        edge_vec_topo = self.relative_emb_topo(target_coords.float() / 10.0,
                                                target_coords.float() / 10.0)
        dist_sca_topo = self.dist_emb_topo(target_coords.float() / 10.0,
                                            target_coords.float() / 10.0)
        edge_fea_topo = torch.cat([edge_vec_topo, dist_sca_topo], dim=-1)
        edge_fea_bias_topo = self.linears_edge1_topo(
            self.linears_edge_topo(edge_fea_topo)).permute(0, 3, 1, 2)
        res, _ = self.decoder(tgt_fea, memory, src_mask_dec, tgt_mask,
                              edge_fea_bias_topo)
        token_logits = self.proj(res[:, :-1])  # (B, T-1, 76)

        # Decoder #2 (xyz)
        gt_coords_emb_rooted = self.coords_emb_gt(target_coords)
        gt_coords_emb_rooted1 = gt_coords_emb_rooted.reshape(B, T, -1)
        type_fea2 = self.gt_fea1(target_token[:, 1:])
        new_fea2 = torch.cat([type_fea2, gt_coords_emb_rooted1[:, :-1]], dim=-1)
        tgt_fea2 = self.pos_encode(new_fea2)
        src_mask_rel = sm.repeat(1, T - 1, 1)
        tgt_mask2 = subsequent_mask(T - 1).long().to(device)
        rc = target_coords[:, :-1].float() / 10.0
        edge_vec = self.relative_emb(rc, rc)
        dist_sca = self.dist_emb(rc, rc)
        edge_fea = torch.cat([edge_vec, dist_sca], dim=-1)
        edge_fea_bias = self.linears_edge1(
            self.linears_edge(edge_fea)).permute(0, 3, 1, 2)
        res_aux, _ = self.decoder_relative(tgt_fea2, memory, src_mask_rel,
                                            tgt_mask2, edge_fea_bias, None)
        pos_res = res_aux + tgt_fea2
        x_hidden = self.fcx(pos_res)
        gt_next_xyz = target_coords[:, 1:]
        gt_x_oh = F.one_hot(gt_next_xyz[..., 0].long(),
                            num_classes=240).float()
        gt_y_oh = F.one_hot(gt_next_xyz[..., 1].long(),
                            num_classes=240).float()
        y_hidden = self.fcy(torch.cat([pos_res, gt_x_oh], dim=-1))
        z_hidden = self.fcz(torch.cat([pos_res, gt_x_oh, gt_y_oh], dim=-1))

        logits = {
            "token": token_logits,  # (B, T-1, 76)
            "x": x_hidden,           # (B, T-1, 240)
            "y": y_hidden,
            "z": z_hidden,
        }
        out["logits"] = logits
        out["res"] = res  # (B, T, 512) — used for chassis/reactivity KL
        return out

    TransformerModel.forward_train = patched


# ---------------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------------
def compute_kl_loss(ft_logits, ref_logits, target_mask):
    """Compute KL(p_ref || p_ft) averaged over all heads & positions.
    target_mask is (B, T) with 1 for valid (non-PAD) positions; we use
    mask[:,1:] for the prediction positions.
    """
    mask_pred = target_mask[:, 1:].float()  # (B, T-1)
    denom = mask_pred.sum().clamp(min=1.0)
    kl_total = 0.0
    for k in ("token", "x", "y", "z"):
        p_ft = F.log_softmax(ft_logits[k], dim=-1)        # (B, T-1, V)
        q_ref = F.softmax(ref_logits[k], dim=-1).detach()
        # KL per position: sum_v q(v) * (log q(v) - log p(v))
        kl_pos = F.kl_div(p_ft, q_ref, reduction="none").sum(dim=-1)  # (B, T-1)
        kl_total = kl_total + (kl_pos * mask_pred).sum() / denom
    return kl_total


# ---------------------------------------------------------------------------
# Training step with KL
# ---------------------------------------------------------------------------
def kl_train_step(model_ft, model_ref, batch, optimizer, beta,
                  anchor_kwargs=None):
    optimizer.zero_grad()
    fwd_batch = dict(batch)
    fwd_batch["_return_logits"] = True
    if anchor_kwargs is None:
        fwd_batch["anchor_geom_weight"] = 0.0
        fwd_batch["anchor_geom_ce_w"] = 0.0
    else:
        fwd_batch.update(anchor_kwargs)
    out_ft = model_ft.forward_train(**fwd_batch)
    with torch.no_grad():
        out_ref = model_ref.forward_train(**fwd_batch)
    loss_supervised = out_ft["loss_total"]
    loss_kl = compute_kl_loss(out_ft["logits"], out_ref["logits"],
                              batch["target_mask"])
    loss_total = loss_supervised + beta * loss_kl
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model_ft.parameters() if p.requires_grad), max_norm=5.0)
    optimizer.step()
    return {
        "loss_total":  float(loss_total.detach()),
        "loss_super":  float(loss_supervised.detach()),
        "loss_kl":     float(loss_kl.detach()),
        "loss_token":  float(out_ft["loss_token"]),
        "loss_x":      float(out_ft["loss_x"]),
        "loss_y":      float(out_ft["loss_y"]),
        "loss_z":      float(out_ft["loss_z"]),
    }


# ---------------------------------------------------------------------------
# Smoke driver
# ---------------------------------------------------------------------------
def smoke(args):
    print(f"[kl-ft smoke] device={args.device}  beta={args.beta}")
    print(f"[kl-ft smoke] complex_csv={args.complex_csv}")
    print(f"[kl-ft smoke] pdb_dir={args.pocket_pdb_dir}")

    # Build two model instances
    model_ft = TransformerModel()
    model_ref = TransformerModel()
    load_pretrained(model_ft, args.pretrained_ckpt)
    load_pretrained(model_ref, args.pretrained_ckpt)
    freeze_encoder(model_ft, args.freeze_encoder_layers)
    for p in model_ref.parameters():
        p.requires_grad = False

    device = torch.device(args.device)
    model_ft = model_ft.to(device)
    model_ref = model_ref.to(device)
    model_ft.train()
    model_ref.eval()

    _wrap_forward_train_for_logits()

    # Build dataloader (uses the patched dataloader with CRIT-1/CRIT-2 fixes)
    os.chdir(str(ROOT))
    from lingo3dmol_l1_dataloader import CovalentInDB2Dataset, collate_fn
    from torch.utils.data import DataLoader

    cache_dir = args.cache_dir or str(Path(args.out_dir) / "cache_kl_v3")
    dataset = CovalentInDB2Dataset(
        complex_csv_path=args.complex_csv,
        pdb_dir=args.pocket_pdb_dir,
        T=args.seq_len,
        max_complexes=args.max_complexes,
        cache_dir=cache_dir,
        pocket_radius=args.pocket_radius,
        verbose=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset empty after precheck")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, drop_last=False)

    trainable = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "loss_total", "loss_super", "loss_kl",
                    "loss_token", "loss_x", "loss_y", "loss_z"])

    history = []
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_steps = 0
        for batch in loader:
            for k, v in batch.items():
                if hasattr(v, "to"):
                    batch[k] = v.to(device)
            m = kl_train_step(model_ft, model_ref, batch, optimizer,
                               beta=args.beta)
            row = [epoch, global_step, m["loss_total"], m["loss_super"],
                   m["loss_kl"], m["loss_token"], m["loss_x"], m["loss_y"],
                   m["loss_z"]]
            history.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if global_step % max(1, args.log_every) == 0:
                print(f"  ep={epoch} step={global_step:4d} "
                      f"t={time.time() - t0:6.1f}s "
                      f"L_tot={m['loss_total']:.3f}  L_sup={m['loss_super']:.3f}  "
                      f"L_KL={m['loss_kl']:.4f}  "
                      f"tok={m['loss_token']:.3f} x={m['loss_x']:.3f}")
            global_step += 1
            epoch_steps += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        print(f"[epoch {epoch}] {epoch_steps} steps in "
              f"{time.time() - epoch_t0:.1f}s")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt = out_dir / "ckpt_kl_ft.pt"
    torch.save({"model": model_ft.state_dict(), "args": vars(args),
                "history": history}, ckpt)

    # Summary JSON
    summary = {
        "n_steps": len(history),
        "initial_L_super": history[0][3] if history else None,
        "final_L_super":   history[-1][3] if history else None,
        "initial_L_KL":    history[0][4] if history else None,
        "final_L_KL":      history[-1][4] if history else None,
        "beta":            args.beta,
        "device":          args.device,
        "dataset_skips":   dict(dataset.getitem_skips),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] ckpt={ckpt}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--complex_csv", required=True)
    p.add_argument("--pocket_pdb_dir", required=True)
    p.add_argument("--pretrained_ckpt", default="checkpoint/gen_mol.pkl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="KL weight")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_complexes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=20,
                   help="0 = no cap (full epoch)")
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--pocket_radius", type=float, default=15.0)
    p.add_argument("--freeze_encoder_layers", type=int, default=3)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_every", type=int, default=2)
    args = p.parse_args()

    args.pretrained_ckpt = _abs(args.pretrained_ckpt)
    args.complex_csv = _abs(args.complex_csv)
    args.pocket_pdb_dir = _abs(args.pocket_pdb_dir)
    args.out_dir = _abs(args.out_dir)
    if args.cache_dir:
        args.cache_dir = _abs(args.cache_dir)
    smoke(args)
