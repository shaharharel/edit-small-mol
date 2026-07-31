"""Train the M1a pocket+warhead-pose conditioned mol2mol decoder.

Loss: conditional NLL  L = -log P(SMILES | pocket, warhead_pose)
Source SMILES: we use the *anchor* SMILES = a randomized version of the target
SMILES (this is the standard mol2mol training scheme — the model learns to map
a noisy anchor to its canonical SAR-relevant target). The conditioning gives
the model the spatial context.

Save checkpoint every CKPT_INTERVAL steps.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder"))

from m1a_model import load_m1a, save_m1a, M1aConditionedModel  # noqa: E402
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def randomize_smi(smi: str) -> str:
    """Return a random SMILES of the same molecule (mol2mol-style noise)."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


class M1aDataset(Dataset):
    def __init__(self, npz_path: Path, vocabulary, tokenizer, max_len: int = 128):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.residues_emb = d["residues_emb"]    # (N, R_max, 320)
        self.residues_mask = d["residues_mask"]  # (N, R_max)
        self.poses = d["poses"]                   # (N, 6)
        self.smiles = d["smiles"]                 # (N,)
        self.sources = d["sources"]
        self.struct_ids = d["struct_ids"]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Filter out rows where canonical smiles tokenizes too long
        ok = []
        for i, smi in enumerate(self.smiles):
            try:
                tokens = self.tokenizer.tokenize(smi)
                if len(tokens) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok)
        print(f"[Dataset] {len(self.idx)} / {len(self.smiles)} usable rows",
               flush=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = self.idx[k]
        canon_smi = self.smiles[i]
        anchor_smi = randomize_smi(canon_smi)
        return {
            "src_smi": anchor_smi,
            "tgt_smi": canon_smi,
            "residues_emb": self.residues_emb[i],
            "residues_mask": self.residues_mask[i],
            "pose": self.poses[i],
        }


def collate_factory(vocabulary, tokenizer, device, max_len: int = 128):
    def collate(batch):
        B = len(batch)
        # Tokenize
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
        src_mask = (src_t != 0).unsqueeze(-2).long()  # (B, 1, L_src)
        trg_mask = make_std_mask(trg_t[:, :-1], 0)    # (B, 1, L_tgt-1)
        # Conditioning
        res_emb = torch.from_numpy(np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(np.stack([b["pose"] for b in batch])).to(device)
        return src_t, src_mask, trg_t, trg_mask, res_emb, res_mask, pose
    return collate


def make_std_mask(tgt, pad):
    """Standard mol2mol target mask: (B,1,L) padding * (1,L,L) subsequent."""
    tgt_mask = (tgt != pad).unsqueeze(-2)  # (B,1,L)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)  # (1,L,L)
    return tgt_mask & sub_mask


def cosine_warmup_lr(step: int, warmup: int, total: int, peak: float,
                       min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_emb", default=str(PROJECT_ROOT / "data/m1a_triples/pocket_embeddings.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "models/m1a_checkpoints"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--ckpt_interval", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- Load model ----
    print("Loading M1a model from prior...", flush=True)
    model = load_m1a(args.prior, device)
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
           f" (base + pocket + pose)", flush=True)
    print(f"Pocket+pose extra: {sum(p.numel() for p in model.pocket_enc.parameters()) / 1e6:.2f}M "
           f"+ {sum(p.numel() for p in model.pose_enc.parameters()) / 1e3:.1f}K",
           flush=True)

    # ---- Dataset ----
    ds = M1aDataset(Path(args.triples_emb), model.base.vocabulary,
                     model.base.tokenizer, max_len=128)
    N = len(ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Train={len(train_idx)} Val={len(val_idx)}", flush=True)
    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate, num_workers=0)

    # ---- Optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}",
           flush=True)

    # ---- Resume ----
    start_step = 0
    if args.resume:
        ckpts = sorted(out_dir.glob("m1a_step*.ckpt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from {latest}", flush=True)
            sd = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(sd["model_state"])
            optim.load_state_dict(sd["optim_state"])
            start_step = sd["step"]
            print(f"Resumed at step {start_step}", flush=True)

    # ---- Training loop ----
    log_path = out_dir / "train_log.jsonl"
    log_fp = open(log_path, "a")
    model.train()
    step = start_step
    t_start = time.time()
    accum_loss = 0.0
    accum_n = 0
    epoch = 0
    keep_going = True
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
            if step % 50 == 0:
                avg = accum_loss / accum_n
                elapsed = time.time() - t_start
                ips = step / elapsed
                print(f"step={step}  epoch={epoch}  loss={avg:.4f}  lr={lr:.2e}  "
                       f"steps/s={ips:.2f}", flush=True)
                accum_loss = 0.0; accum_n = 0
            if step % args.ckpt_interval == 0:
                # Validation
                model.eval()
                v_loss = 0.0; v_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        s, sm, t, tm, re, rm, po = vb
                        v_nll = model.likelihood(s, sm, t, tm, re, rm, po).mean()
                        v_loss += float(v_nll.item()) * s.shape[0]
                        v_n += s.shape[0]
                val_avg = v_loss / max(1, v_n)
                model.train()
                ck_path = out_dir / f"m1a_step{step:06d}.ckpt"
                torch.save({"model_state": model.state_dict(),
                             "optim_state": optim.state_dict(),
                             "step": step, "epoch": epoch,
                             "val_loss": val_avg},
                            ck_path)
                rec = {"step": step, "epoch": epoch, "val_loss": val_avg,
                        "lr": lr, "wallclock_s": time.time() - t_start}
                log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
                print(f"[ckpt] step={step}  val_loss={val_avg:.4f}  -> {ck_path}",
                       flush=True)
                # Keep last 3 checkpoints + every 5000th
                ckpts = sorted(out_dir.glob("m1a_step*.ckpt"))
                for old in ckpts[:-3]:
                    if int(old.stem.split("step")[-1]) % 5000 != 0:
                        old.unlink(missing_ok=True)
            if step >= total_steps:
                keep_going = False; break
        epoch += 1

    # Final checkpoint
    final_path = out_dir / "m1a_final.ckpt"
    torch.save({"model_state": model.state_dict(),
                 "optim_state": optim.state_dict(),
                 "step": step, "epoch": epoch}, final_path)
    print(f"FINAL: {final_path}  (step={step})", flush=True)
    log_fp.close()


if __name__ == "__main__":
    main()
