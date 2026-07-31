"""Train the M1a v2 pocket+pose-conditioned mol2mol decoder.

Same architecture/optimizer/loss as v1 (train_m1a.py) but:
  - Loads esm2_cache_posefix.npz which contains z-scored poses (Fix 3).
  - Reads the row->seq dedupe index (row_seq_idx) to avoid blowing up memory
    on the residue-embedding tensor.
  - Persists pose_mean/std as model buffers (via M1aV2ConditionedModel).
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

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from m1a_v2_model import load_m1a_v2, save_m1a_v2, M1aV2ConditionedModel  # noqa: E402
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
    """Dataset backed by a deduplicated ESM cache.

    The cache layout (from enrich_and_recompute_poses.py):
      residues_emb : (U, R_max, 320) float32  - U unique sequences
      residues_mask: (U, R_max) bool
      row_seq_idx  : (N,) int32                - per-row index into U
      poses        : (N, 3) float32            - ALREADY z-scored (v3 invariant pose)
      smiles       : (N,) object               - canonical SMILES per row
    """

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
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "models/m1a_v2_checkpoints"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--ckpt_interval", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/m1a_v2_progress.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(phase: str, **extra):
        rec = {"phase": phase, "timestamp": time.time(),
               "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        Path(args.progress_path).write_text(json.dumps(rec, indent=2))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- Pose normalizer from the cache ----
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]
    pose_std = d_in["pose_std"]
    print(f"Pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)

    # ---- Load model with persisted normalizer ----
    print("Loading M1a v2 model from prior...", flush=True)
    model = load_m1a_v2(args.prior, device, pose_mean=pose_mean, pose_std=pose_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params/1e6:.1f}M", flush=True)

    # ---- Dataset ----
    ds = M1aV2Dataset(Path(args.cache), model.base.vocabulary, model.base.tokenizer,
                       max_len=128)
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
        ckpts = sorted(out_dir.glob("m1a_v2_step*.ckpt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from {latest}", flush=True)
            sd = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(sd["model_state"])
            optim.load_state_dict(sd["optim_state"])
            start_step = sd["step"]
            print(f"Resumed at step {start_step}", flush=True)

    log_path = out_dir / "train_log_v2.jsonl"
    log_fp = open(log_path, "a")
    model.train()
    step = start_step
    t_start = time.time()
    accum_loss = 0.0; accum_n = 0
    epoch = 0
    keep_going = True
    write_progress("phase1_training", epoch=epoch, step=step,
                   total_steps=total_steps)
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
                ck_path = out_dir / f"m1a_v2_step{step:06d}.ckpt"
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
                # Keep last 3 ckpts + every 5000th
                ckpts = sorted(out_dir.glob("m1a_v2_step*.ckpt"))
                for old in ckpts[:-3]:
                    if int(old.stem.split("step")[-1]) % 5000 != 0:
                        old.unlink(missing_ok=True)
                write_progress("phase1_training_ckpt",
                               epoch=epoch, step=step, val_loss=val_avg,
                               total_steps=total_steps,
                               wallclock_s=time.time() - t_start)
            if step >= total_steps:
                keep_going = False; break
        epoch += 1

    final_path = out_dir / "m1a_v2_final.ckpt"
    save_m1a_v2(model, str(final_path),
                  extra={"step": step, "epoch": epoch,
                          "optim_state": optim.state_dict()})
    # Also save as ../m1a_v2.ckpt for the spec deliverable name
    spec_path = Path(str(out_dir).rstrip("/")) / ".." / "m1a_v2.ckpt"
    spec_path = spec_path.resolve()
    save_m1a_v2(model, str(spec_path),
                  extra={"step": step, "epoch": epoch})
    print(f"FINAL: {final_path}  (step={step})", flush=True)
    print(f"SPEC-NAMED: {spec_path}", flush=True)
    log_fp.close()
    write_progress("phase1_done", final_step=step, final_epoch=epoch)


if __name__ == "__main__":
    main()
