"""Fine-tune the M1a v2-cond decoder on an offline pair NPZ (Scheme A or B).

This is a MINIMAL alternative to `train_m1a_v2.py`:
  * Reads a pairs NPZ built by build_pairs_scheme_{A,B}.py.
  * Warm-starts weights from an existing v2 checkpoint (typically
    `models/m1a_v2.ckpt` — the self-reconstruction autoencoder).
  * Uses the SAME optimizer, loss, and forward pass as train_m1a_v2.py.
  * Only difference: the dataset now yields (src, tgt) where src is a
    DIFFERENT molecule than tgt (so the model must actually consume the
    pocket + pose tokens).

CLI: --pairs_npz PATH  --out_ckpt PATH  --epochs N  --lr 1e-4  --bs 16
     --limit N (unit-test convenience)

The pose is already z-scored in the cache and copied verbatim into the pair
NPZ; the encoder pass does not re-normalize (this matches train_m1a_v2.py).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---- Path shims (local mac vs a100) ----
LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT
LOCAL_REINVENT = Path("/Users/shaharharel/Documents/github/REINVENT4")
A100_REINVENT = Path("/home/shaharh_quris_ai/REINVENT4")
REINVENT4_ROOT = LOCAL_REINVENT if LOCAL_REINVENT.exists() else A100_REINVENT
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

FIXES_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
if str(FIXES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXES_DIR))

from m1a_v2_model import load_m1a_v2, save_m1a_v2  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import (  # noqa: E402
    subsequent_mask,
)
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MIN_PAIRS_TO_TRAIN = 500


def randomize_smi(smi: str) -> str:
    """Re-randomize per epoch. Never touch tgt SMILES with this."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


class PairDataset(Dataset):
    """Backed by a pairs NPZ and the ESM pocket cache.

    The pair NPZ stores an index into the pocket cache to avoid duplicating
    the 3155*18*320 residues_emb tensor per pair.
    """

    def __init__(self, pairs_npz: Path, cache_npz: Path, vocabulary,
                 tokenizer, max_len: int = 128, rerandomize_src: bool = True):
        super().__init__()
        p = np.load(pairs_npz, allow_pickle=True)
        c = np.load(cache_npz, allow_pickle=True)
        self.src_smi = p["src_smi"]
        self.tgt_smi = p["tgt_smi"]
        self.res_idx = p["residues_emb_idx"].astype(np.int64)
        self.mask_idx = p["residues_mask_idx"].astype(np.int64)
        self.pose = p["pose"].astype(np.float32)
        self.residues_emb = c["residues_emb"]
        self.residues_mask = c["residues_mask"]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rerandomize_src = rerandomize_src

        # Drop pairs whose tgt won't fit in max_len (src is regenerated each
        # epoch so we can't check it once; we check the canonical form of tgt).
        ok = []
        for i in range(len(self.src_smi)):
            try:
                t_toks = self.tokenizer.tokenize(str(self.tgt_smi[i]))
                # We ALSO check the initial src (the anchor stored in NPZ)
                s_toks = self.tokenizer.tokenize(str(self.src_smi[i]))
                if len(t_toks) < max_len - 2 and len(s_toks) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok, dtype=np.int64)
        print(f"[PairDataset] {len(self.idx)} / {len(self.src_smi)} usable "
               f"pairs; cache_pockets={self.residues_emb.shape[0]}",
               flush=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        src_smi = str(self.src_smi[i])
        tgt_smi = str(self.tgt_smi[i])  # already canonical; DO NOT randomize
        if self.rerandomize_src:
            src_smi = randomize_smi(src_smi)
        seq_idx = int(self.res_idx[i])
        return {
            "src_smi": src_smi,
            "tgt_smi": tgt_smi,
            "residues_emb": self.residues_emb[seq_idx],
            "residues_mask": self.residues_mask[seq_idx],
            "pose": self.pose[i],
        }


def collate_factory(vocabulary, tokenizer, device, max_len: int = 128):
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
            src[j, : len(s)] = s
        for j, s in enumerate(tgt_seqs):
            trg[j, : len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        trg_t = torch.from_numpy(trg).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        trg_mask = make_std_mask(trg_t[:, :-1], 0)
        res_emb = torch.from_numpy(
            np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(
            np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(
            np.stack([b["pose"] for b in batch])).to(device)
        return src_t, src_mask, trg_t, trg_mask, res_emb, res_mask, pose

    return collate


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
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
    ap.add_argument("--pairs_npz", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                    "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--warm_start", default=str(PROJECT_ROOT /
                    "models/m1a_v2.ckpt"),
                    help="Warm-start weights (v2 self-reconstruction ckpt). "
                         "Set to '' to skip warm-start.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first --limit pairs. For dry-runs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=MIN_PAIRS_TO_TRAIN,
                    help="Refuse to train if fewer than this many pairs. "
                         "Override for smoke tests.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_m1a_pairs] Device: {device}", flush=True)

    # ---- Guardrail: pair count ----
    p = np.load(args.pairs_npz, allow_pickle=True)
    n_total = int(len(p["src_smi"]))
    print(f"[train_m1a_pairs] pair NPZ has {n_total} pairs (scheme="
           f"{str(p['scheme']) if 'scheme' in p.files else '?'})", flush=True)
    if n_total < args.min_pairs:
        print(f"[train_m1a_pairs] REFUSING to train: {n_total} < "
               f"--min_pairs {args.min_pairs}. Something went wrong upstream.",
               flush=True)
        sys.exit(2)

    # ---- Pose normalizer from cache (buffers are persisted with the model) ----
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]
    pose_std = d_in["pose_std"]
    print(f"[train_m1a_pairs] pose_mean={pose_mean} pose_std={pose_std}",
           flush=True)

    print("[train_m1a_pairs] Loading base model from prior...", flush=True)
    model = load_m1a_v2(args.prior, device,
                        pose_mean=pose_mean, pose_std=pose_std)

    if args.warm_start:
        ws_path = Path(args.warm_start)
        if ws_path.exists():
            print(f"[train_m1a_pairs] Warm-starting from {ws_path}",
                   flush=True)
            sd = torch.load(ws_path, map_location=device, weights_only=False)
            state = sd["model_state"] if "model_state" in sd else sd
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"    missing={len(missing)}  unexpected={len(unexpected)}",
                   flush=True)
            if unexpected:
                print(f"    (first 5 unexpected: {unexpected[:5]})", flush=True)
        else:
            print(f"[train_m1a_pairs] WARN: warm_start path does not exist: "
                   f"{ws_path}. Training from prior only.", flush=True)

    # ---- Dataset ----
    ds = PairDataset(Path(args.pairs_npz), Path(args.cache),
                     model.base.vocabulary, model.base.tokenizer, max_len=128)
    N = len(ds)
    if args.limit is not None:
        N = min(N, args.limit)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))[:N]
    n_val = max(1, int(N * args.val_frac)) if N >= 10 else 0
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"[train_m1a_pairs] N={N}  train={len(train_idx)}  val={len(val_idx)}",
           flush=True)
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist()) if n_val else None
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = (DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              collate_fn=collate, num_workers=0)
                   if val_ds is not None else None)

    optim = torch.optim.AdamW(
        [pp for pp in model.parameters() if pp.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    print(f"[train_m1a_pairs] steps_per_epoch={steps_per_epoch} "
           f"total_steps={total_steps}", flush=True)

    out_ckpt_path = Path(args.out_ckpt)
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_ckpt_path.parent / (out_ckpt_path.stem + "_log.jsonl")
    log_fp = open(log_path, "a")

    model.train()
    step = 0
    t_start = time.time()
    loss_hist: list[float] = []
    for epoch in range(args.epochs):
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
                [pp for pp in model.parameters() if pp.requires_grad], 1.0)
            optim.step()
            step += 1
            loss_hist.append(float(loss.item()))
            if step % args.log_every == 0 or step <= 5:
                avg = sum(loss_hist[-args.log_every:]) / min(len(loss_hist), args.log_every)
                elapsed = time.time() - t_start
                print(f"[train_m1a_pairs] epoch={epoch} step={step} "
                       f"loss={loss.item():.4f} avg{args.log_every}={avg:.4f} "
                       f"lr={lr:.2e} elapsed={elapsed:.1f}s", flush=True)
                log_fp.write(json.dumps({
                    "step": step, "epoch": epoch,
                    "loss": float(loss.item()), "avg_loss": avg,
                    "lr": lr, "wallclock_s": elapsed}) + "\n")
                log_fp.flush()

        # ---- End-of-epoch val ----
        if val_loader is not None:
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
            print(f"[train_m1a_pairs] epoch={epoch} val_loss={val_avg:.4f}",
                   flush=True)
            log_fp.write(json.dumps({
                "epoch": epoch, "val_loss": val_avg,
                "wallclock_s": time.time() - t_start}) + "\n")
            log_fp.flush()

    # ---- Save ----
    save_m1a_v2(model, str(out_ckpt_path),
                extra={"step": step, "epoch": args.epochs,
                        "pairs_npz": args.pairs_npz})
    print(f"[train_m1a_pairs] wrote {out_ckpt_path}", flush=True)
    log_fp.close()

    # ---- Report loss trajectory summary ----
    if loss_hist:
        first = sum(loss_hist[:5]) / min(5, len(loss_hist))
        last = sum(loss_hist[-5:]) / min(5, len(loss_hist))
        print(f"[train_m1a_pairs] loss trajectory: "
               f"first5_avg={first:.4f}  last5_avg={last:.4f}  "
               f"delta={first-last:+.4f}", flush=True)


if __name__ == "__main__":
    main()
