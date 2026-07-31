"""Train the CFG-augmented M1a v2 pocket+pose-conditioned mol2mol decoder.

Warm-starts from an existing v2 checkpoint (the covFT->v2-pocket/pose model,
default `models/m1a_v2.ckpt`) and adds a learned null token for
Classifier-Free Guidance.  With probability `--p_drop` (default 0.15) a
training sample's [POCKET]/[POSE] tokens are replaced together by the null
token.  All other training details match `train_m1a_v2.py`.

We save checkpoints every `--ckpt_interval` steps (default 500 for spot
resiliency) into `models/cfg_retrieval/`, and write a per-step CSV log:
    data/paper_pair_training/cfg_retrieval/train_log.csv

with columns: step, epoch, loss, val_loss, val_cond_loss, val_uncond_loss,
                lr, wallclock_s, timestamp_iso.
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/cfg_retrieval"))

from m1a_v2_cfg_model import load_cfg_model, save_cfg_model, M1aV2CfgModel  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


class M1aV2Dataset(Dataset):
    def __init__(self, npz_path: Path, vocabulary, tokenizer, max_len: int = 128):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.residues_emb = d["residues_emb"]
        self.residues_mask = d["residues_mask"]
        self.row_seq_idx = d["row_seq_idx"]
        self.poses = d["poses"]
        self.smiles = d["smiles"]
        self.sources = d["sources"]
        self.struct_ids = d["struct_ids"]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len
        ok = []
        for i, smi in enumerate(self.smiles):
            try:
                tokens = self.tokenizer.tokenize(smi)
                if len(tokens) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok)
        print(f"[Dataset v2] {len(self.idx)} / {len(self.smiles)} usable rows "
               f"(U={self.residues_emb.shape[0]} unique pockets)", flush=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = self.idx[k]
        canon_smi = self.smiles[i]
        anchor_smi = randomize_smi(canon_smi)
        seq_idx = self.row_seq_idx[i]
        return {
            "src_smi": anchor_smi,
            "tgt_smi": canon_smi,
            "residues_emb": self.residues_emb[seq_idx],
            "residues_mask": self.residues_mask[seq_idx],
            "pose": self.poses[i],
        }


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask & sub_mask


def collate_factory(vocabulary, tokenizer, device):
    def collate(batch):
        B = len(batch)
        src_seqs, tgt_seqs = [], []
        for b in batch:
            src = tokenizer.tokenize(b["src_smi"])
            tgt = tokenizer.tokenize(b["tgt_smi"])
            src_seqs.append(np.array(vocabulary.encode(src), dtype=np.int64))
            tgt_seqs.append(np.array(vocabulary.encode(tgt), dtype=np.int64))
        L_src = max(len(s) for s in src_seqs)
        L_tgt = max(len(s) for s in tgt_seqs)
        src = np.zeros((B, L_src), dtype=np.int64)
        trg = np.zeros((B, L_tgt), dtype=np.int64)
        for j, s in enumerate(src_seqs):
            src[j, :len(s)] = s
        for j, s in enumerate(tgt_seqs):
            trg[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        trg_t = torch.from_numpy(trg).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        trg_mask = make_std_mask(trg_t[:, :-1], 0)
        res_emb = torch.from_numpy(np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(np.stack([b["pose"] for b in batch])).to(device)
        return src_t, src_mask, trg_t, trg_mask, res_emb, res_mask, pose
    return collate


def cosine_warmup_lr(step: int, warmup: int, total: int, peak: float,
                       min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cos)


def write_progress(progress_path: Path, **fields):
    rec = {"timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    progress_path.write_text(json.dumps(rec, indent=2))


def append_csv(csv_path: Path, row: dict, header_written_ref: list):
    write_header = not header_written_ref[0] and not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    header_written_ref[0] = True


def evaluate(model: M1aV2CfgModel, loader, device) -> dict:
    """Report conditional NLL, unconditional NLL and dropout-average NLL on
    the val split.  The dropout-average is what the CFG training loss
    minimises."""
    model.eval()
    n = 0
    s_cond = 0.0; s_unc = 0.0; s_all = 0.0
    with torch.no_grad():
        for vb in loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose = vb
            B = src.shape[0]
            drop_none = torch.zeros(B, dtype=torch.bool, device=device)
            drop_all = torch.ones(B, dtype=torch.bool, device=device)
            nll_c = model.likelihood(src, src_mask, trg, trg_mask,
                                       res_emb, res_mask, pose,
                                       drop_cond=drop_none).mean()
            nll_u = model.likelihood(src, src_mask, trg, trg_mask,
                                       res_emb, res_mask, pose,
                                       drop_cond=drop_all).mean()
            s_cond += float(nll_c.item()) * B
            s_unc += float(nll_u.item()) * B
            s_all += 0.5 * (float(nll_c.item()) + float(nll_u.item())) * B
            n += B
    model.train()
    return {"val_cond_loss": s_cond / max(1, n),
             "val_uncond_loss": s_unc / max(1, n),
             "val_loss": s_all / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--warm_start_v2", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"),
                    help="Warm-start CFG training from this v2 checkpoint.")
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "models/cfg_retrieval"))
    ap.add_argument("--log_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval"))
    ap.add_argument("--p_drop", type=float, default=0.15,
                    help="CFG conditioning dropout probability.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="Lower LR since we warm-start from a tuned model.")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--ckpt_interval", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / "progress.json"
    train_csv = log_dir / "train_log.csv"
    header_written = [False]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Pose normalizer from cache.
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]
    print(f"Pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)

    # Load CFG wrapper warm-started from v2 checkpoint.
    print(f"Loading CFG model warm-started from {args.warm_start_v2}", flush=True)
    model = load_cfg_model(args.prior, device,
                             pose_mean=pose_mean, pose_std=pose_std,
                             init_from_v2_ckpt=args.warm_start_v2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params/1e6:.1f}M", flush=True)

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

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
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
            # Detect if csv exists so we don't rewrite the header.
            if train_csv.exists() and train_csv.stat().st_size > 0:
                header_written[0] = True

    model.train()
    step = start_step
    t_start = time.time()
    accum_loss = 0.0; accum_n = 0
    epoch = 0
    keep_going = True
    write_progress(progress_path, phase="training", epoch=epoch, step=step,
                     total_steps=total_steps)

    # Baseline eval at step 0 for reference.
    if start_step == 0:
        val = evaluate(model, val_loader, device)
        print(f"[baseline step 0] val={val}", flush=True)
        append_csv(train_csv, {"step": 0, "epoch": 0, "loss": float("nan"),
                                 **val, "lr": 0.0, "wallclock_s": 0.0,
                                 "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S")},
                     header_written)

    while keep_going and step < total_steps:
        for batch in train_loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose = batch
            B = src.shape[0]
            lr = cosine_warmup_lr(step + 1, args.warmup_steps, total_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            # Per-sample independent drop.
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
                print(f"step={step}  epoch={epoch}  loss={avg:.4f}  lr={lr:.2e}  "
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
                # Keep only latest 2 ckpts + every 5000th.
                ckpts = sorted(out_dir.glob("cfg_step*.ckpt"))
                for old in ckpts[:-2]:
                    if int(old.stem.split("step")[-1]) % 5000 != 0:
                        old.unlink(missing_ok=True)
                append_csv(train_csv, {"step": step, "epoch": epoch,
                                         "loss": float(loss.item()),
                                         **val, "lr": lr,
                                         "wallclock_s": time.time() - t_start,
                                         "timestamp_iso":
                                             time.strftime("%Y-%m-%dT%H:%M:%S")},
                             header_written)
                write_progress(progress_path, phase="training_ckpt",
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
    save_cfg_model(model, str(final_path),
                     extra={"step": step, "epoch": epoch})
    print(f"FINAL: {final_path}  (step={step})", flush=True)
    write_progress(progress_path, phase="training_done",
                     final_step=step, final_epoch=epoch)


if __name__ == "__main__":
    main()
