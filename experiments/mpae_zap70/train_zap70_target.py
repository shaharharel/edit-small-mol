#!/usr/bin/env python3
"""Train one (target, head_variant) model in the ZAP70 10-way factorial.

For each of 5 targets × 2 head variants (V3=InfoNCE-only, V4=InfoNCE+regression):
  - Warm-start models/m1a_v2.ckpt (ORIGINAL autoencoder)
  - Add structural fixes:
      separate ContrastivePoseEncoder (not shared with h_[POSE] injection)
      ProjectionHead g(·) on decoder h_pooled
  - InfoNCE: K=4 negatives per anchor
      2 real from OPPOSITE target-quartile rows in the batch (target-quartile
        computed from `target_<name>` — lower=better)
      2 synthetic ±3σ pose perturbations + φ-flip 180°
  - V4 also has RegressionHead predicting -target from h_pooled (MSE)

12 epochs, batch 32, LR 1e-4, warmup 500.
Curriculum: λ_reg 0→0.5 over ep 0-3, λ_nce 0→0.3 over ep 0-5.

Fail-fast:
  val_infonce > 0.9*log(K+1)=1.45 after ep 5   → stop that variant
  val_ce > warm_start_val_ce + 2               → stop that variant
  NaN loss                                     → stop that variant
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Path shim
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

from m1a_v2_model import load_m1a_v2, M1aV2ConditionedModel  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import (  # noqa: E402
    subsequent_mask,
)
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

D_MODEL = 256
POSE_DIM = 3
K_NEGS = 4
TARGETS = ["d", "theta", "phi", "mpae", "composite"]


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


# ============================================================================
# Dataset
# ============================================================================
class ZAP70Dataset(Dataset):
    def __init__(self, labeled_npz: Path, esm_npz: Path, vocabulary, tokenizer,
                  target_name: str, max_len: int = 128):
        super().__init__()
        L = np.load(labeled_npz, allow_pickle=True)
        E = np.load(esm_npz, allow_pickle=True)
        self.smiles = L["smiles"]
        self.pose_boltz = L["pose_boltz"].astype(np.float32)  # UNNORMALIZED, (N,3)
        self.pose_norm = L["pose_norm"].astype(np.float32)     # NORMALIZED, (N,3)
        self.row_seq_idx = L["row_seq_idx"].astype(np.int64)   # (N,) idx into esm bank
        # Target column
        key = f"target_{target_name}"
        assert key in L.files, f"target key {key!r} missing (available: {L.files})"
        self.target = L[key].astype(np.float32)
        # z-normalize for the regression loss (targets like theta/phi/composite have
        # large raw magnitudes; unnormalized MSE would swamp CE). We keep the raw
        # values for quartile-binning and reg_head_prediction reporting.
        self.target_mean = float(self.target.mean())
        self.target_std = float(self.target.std() + 1e-6)
        self.target_z = ((self.target - self.target_mean) / self.target_std).astype(np.float32)
        print(f"[Dataset] target={target_name} z-norm: mean={self.target_mean:.3f} "
               f"std={self.target_std:.3f}", flush=True)
        self.residues_emb = E["residues_emb"].astype(np.float32)
        self.residues_mask = E["residues_mask"].astype(bool)
        self.vocabulary = vocabulary; self.tokenizer = tokenizer; self.max_len = max_len

        # Filter tokenizable
        ok = []
        for i, smi in enumerate(self.smiles):
            try:
                toks = tokenizer.tokenize(smi)
                if len(toks) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok, dtype=np.int64)
        print(f"[Dataset] target={target_name}  {len(self.idx)}/{len(self.smiles)} "
               f"usable, pockets={self.residues_emb.shape[0]}", flush=True)

        # Precompute per-anchor target quartile (bottom-quartile = "best" since lower=better,
        # top-quartile = "worst").
        vals = self.target[self.idx]
        self.q25 = float(np.quantile(vals, 0.25))
        self.q75 = float(np.quantile(vals, 0.75))
        # For anchor sampling, we want opposite-target quartile negatives.
        # Bin each row: 0=best (below q25), 2=worst (above q75), 1=middle.
        self.bin = np.full(len(self.smiles), 1, dtype=np.int64)
        self.bin[self.target <= self.q25] = 0
        self.bin[self.target >= self.q75] = 2
        # Row-pool per bin (restricted to usable idx)
        usable_bin = self.bin[self.idx]
        self.pool_best = self.idx[usable_bin == 0]
        self.pool_worst = self.idx[usable_bin == 2]
        self.pool_mid = self.idx[usable_bin == 1]
        print(f"[Dataset] target quartile pools: best(≤q25={self.q25:.3f}):"
               f"{len(self.pool_best)} worst(≥q75={self.q75:.3f}):{len(self.pool_worst)} "
               f"mid:{len(self.pool_mid)}", flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        canon = str(self.smiles[i])
        src_smi = randomize_smi(canon)
        return {
            "src_smi": src_smi,
            "tgt_smi": canon,
            "residues_emb": self.residues_emb[self.row_seq_idx[i]],
            "residues_mask": self.residues_mask[self.row_seq_idx[i]],
            "pose": self.pose_norm[i].astype(np.float32),
            "pose_unnorm": self.pose_boltz[i].astype(np.float32),
            "target_val": float(self.target[i]),
            "target_val_z": float(self.target_z[i]),
            "bin": int(self.bin[i]),
            "row": i,
        }


def build_negatives_batch(batch, ds: ZAP70Dataset,
                            pose_mean: np.ndarray, pose_std: np.ndarray) -> np.ndarray:
    """(B, K, 3) NORMALIZED pose negatives. K=4:
        2 real from OPPOSITE target-quartile in the training pool
        2 synthetic (θ±3σ, φ-flip 180°)
    Cross-quartile logic: for anchor in bin 0 (best), sample from pool_worst.
                          for anchor in bin 2 (worst), sample from pool_best.
                          for anchor in bin 1 (mid), sample from pool_best ∪ pool_worst.
    """
    B = len(batch)
    negs = np.zeros((B, K_NEGS, 3), dtype=np.float32)
    rng = np.random.default_rng()
    for b_i, b in enumerate(batch):
        bin_a = b["bin"]
        if bin_a == 0:
            pool = ds.pool_worst
        elif bin_a == 2:
            pool = ds.pool_best
        else:
            pool = np.concatenate([ds.pool_best, ds.pool_worst])
        pool = pool[pool != b["row"]]
        if len(pool) >= 2:
            picks = rng.choice(pool, size=2, replace=False)
        elif len(pool) == 1:
            picks = np.array([pool[0], pool[0]])
        else:
            picks = rng.choice(ds.idx, size=2, replace=False)
        for k_i, pi in enumerate(picks):
            negs[b_i, k_i] = ds.pose_norm[int(pi)]
        # Synthetic negatives: perturb θ by ±3σ_θ + flip φ 180°
        pos = b["pose_unnorm"].copy()
        sig_th = float(pose_std[1])
        neg_a = pos.copy(); neg_a[1] += 3.0 * sig_th
        neg_b = pos.copy(); neg_b[1] -= 3.0 * sig_th
        neg_b[2] = ((neg_b[2] + 180.0 + 180.0) % 360.0) - 180.0
        negs[b_i, 2] = (neg_a - pose_mean) / np.clip(pose_std, 1e-6, None)
        negs[b_i, 3] = (neg_b - pose_mean) / np.clip(pose_std, 1e-6, None)
    return negs


def collate_factory(vocabulary, tokenizer, device, ds: ZAP70Dataset,
                     pose_mean: np.ndarray, pose_std: np.ndarray):
    def collate(batch):
        B = len(batch)
        src_seqs, tgt_seqs = [], []
        for b in batch:
            src_seqs.append(np.array(vocabulary.encode(tokenizer.tokenize(b["src_smi"])),
                                       dtype=np.int64))
            tgt_seqs.append(np.array(vocabulary.encode(tokenizer.tokenize(b["tgt_smi"])),
                                       dtype=np.int64))
        L_src = max(len(s) for s in src_seqs)
        L_tgt = max(len(s) for s in tgt_seqs)
        src = np.zeros((B, L_src), dtype=np.int64)
        trg = np.zeros((B, L_tgt), dtype=np.int64)
        for j, s in enumerate(src_seqs): src[j, :len(s)] = s
        for j, s in enumerate(tgt_seqs): trg[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        trg_t = torch.from_numpy(trg).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        trg_mask = _make_std_mask(trg_t[:, :-1], 0)
        res_emb = torch.from_numpy(np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(np.stack([b["pose"] for b in batch])).to(device)
        target_val = torch.from_numpy(np.array([b["target_val"] for b in batch],
                                                  dtype=np.float32)).to(device)
        target_val_z = torch.from_numpy(np.array([b["target_val_z"] for b in batch],
                                                    dtype=np.float32)).to(device)
        negs_np = build_negatives_batch(batch, ds, pose_mean, pose_std)
        negs = torch.from_numpy(negs_np).to(device)
        return (src_t, src_mask, trg_t, trg_mask, res_emb, res_mask, pose,
                 target_val, target_val_z, negs)
    return collate


def _make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask & sub_mask


# ============================================================================
# Heads
# ============================================================================
class RegressionHead(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class ContrastivePoseEncoder(nn.Module):
    def __init__(self, d_in: int = POSE_DIM, d_hidden: int = 64, d_out: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden),
                                    nn.GELU(), nn.Linear(d_hidden, d_out))
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.out_norm(self.net(x))


class ProjectionHead(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, d))

    def forward(self, x):
        return self.net(x)


def cosine_warmup_lr(step, warmup, total, peak, min_frac=0.1):
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def curriculum_lambda(epoch: int, peak: float, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return peak
    return peak * min(1.0, (epoch + 1) / float(ramp_epochs))


def get_h_pooled(model: M1aV2ConditionedModel, src, src_mask, trg, trg_mask,
                  res_emb, res_mask, pose_norm):
    memory_ext, src_mask_ext = model._build_conditioned_memory(
        src, src_mask, res_emb, res_mask, pose_norm)
    trg_in = trg[:, :-1]
    dec_out = model.base.network.decoder(
        model.base.network.tgt_embed(trg_in), memory_ext, src_mask_ext, trg_mask)
    pad_mask = (trg_in != 0).float().unsqueeze(-1)
    denom = pad_mask.sum(dim=1).clamp_min(1e-6)
    pooled = (dec_out * pad_mask).sum(dim=1) / denom
    return pooled  # (B, D)


def infonce_loss(h_pooled, pose_norm_pos, negs, cpe, proj, temperature: float = 0.1):
    B = h_pooled.size(0)
    K = negs.size(1)
    z_mol = F.normalize(proj(h_pooled), dim=-1)
    z_pos = F.normalize(cpe(pose_norm_pos), dim=-1)
    z_neg = F.normalize(cpe(negs.reshape(-1, POSE_DIM)), dim=-1).view(B, K, -1)
    sim_pos = (z_mol * z_pos).sum(-1) / temperature
    sim_neg = torch.einsum("bd,bkd->bk", z_mol, z_neg) / temperature
    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=h_pooled.device)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    with torch.no_grad():
        pos_win = (logits.argmax(dim=1) == 0).float().mean().item()
    return loss, pos_win


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r) if np.isfinite(r) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--variant", type=int, required=True, choices=[3, 4])
    ap.add_argument("--labeled_npz",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/labeled_mols_zap70.npz"))
    ap.add_argument("--esm_cache",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/esm_cache_zap70.npz"))
    ap.add_argument("--pose_stats",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/pose_stats_zap70.json"))
    ap.add_argument("--warm_start",
                    default=str(PROJECT_ROOT / "models/m1a_v2.ckpt"))
    ap.add_argument("--prior",
                    default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--out_ckpt", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_wall_seconds", type=int, default=45 * 60)
    args = ap.parse_args()

    reg_on = args.variant == 4
    nce_on = True  # both V3 and V4 have InfoNCE
    print(f"[cfg] target={args.target}  variant=V{args.variant}  "
           f"reg_on={reg_on}  nce_on={nce_on}", flush=True)

    if args.out_ckpt is None:
        args.out_ckpt = str(PROJECT_ROOT / f"models/m1a_v2_v{args.variant}_{args.target}.ckpt")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev] {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[dev] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ps = json.loads(Path(args.pose_stats).read_text())
    pose_mean = np.array(ps["pose_mean"], dtype=np.float32)
    pose_std = np.array(ps["pose_std"], dtype=np.float32)
    print(f"[pose_norm] mean={pose_mean}  std={pose_std}", flush=True)

    print("[load] warm-start m1a_v2 model...", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.warm_start,
                         pose_mean=pose_mean, pose_std=pose_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] total params: {n_params/1e6:.1f}M", flush=True)

    reg_head = RegressionHead(D_MODEL).to(device) if reg_on else None
    cpe = ContrastivePoseEncoder().to(device) if nce_on else None
    proj = ProjectionHead(D_MODEL).to(device) if nce_on else None

    ds = ZAP70Dataset(Path(args.labeled_npz), Path(args.esm_cache),
                        model.base.vocabulary, model.base.tokenizer,
                        target_name=args.target, max_len=128)
    N = len(ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"[split] Train={len(train_idx)}  Val={len(val_idx)}", flush=True)
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device, ds,
                                pose_mean, pose_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    if reg_head is not None: params += list(reg_head.parameters())
    if cpe is not None: params += list(cpe.parameters()) + list(proj.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"[train] steps/epoch={steps_per_epoch}  total_steps={total_steps}", flush=True)

    # ---- Warm-start val CE (for guardrail) ----
    def eval_val():
        model.eval()
        if reg_head is not None: reg_head.eval()
        if cpe is not None:
            cpe.eval(); proj.eval()
        v_ce = 0.0; v_reg = 0.0; v_nce = 0.0; v_pos = 0.0; v_n = 0
        preds, targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                s, sm, t, tm, re, rm, po, tgt_v, tgt_v_z, negs = batch
                nll = model.likelihood(s, sm, t, tm, re, rm, po)
                ce = nll.mean()
                v_ce += float(ce.item()) * s.shape[0]
                if reg_on or nce_on:
                    h = get_h_pooled(model, s, sm, t, tm, re, rm, po)
                if reg_on:
                    pred = reg_head(h)
                    # MSE on z-normalized target; report on same scale as train
                    v_reg += float(F.mse_loss(pred, -tgt_v_z).item()) * s.shape[0]
                    preds.append(pred.cpu().numpy())
                    targs.append((-tgt_v_z).cpu().numpy())
                if nce_on:
                    nce_loss, pos_win = infonce_loss(h, po, negs, cpe, proj)
                    v_nce += float(nce_loss.item()) * s.shape[0]
                    v_pos += pos_win * s.shape[0]
                v_n += s.shape[0]
        v_ce /= max(1, v_n); v_reg /= max(1, v_n); v_nce /= max(1, v_n); v_pos /= max(1, v_n)
        spr = float("nan")
        if reg_on and preds:
            p = np.concatenate(preds); t_ = np.concatenate(targs)
            spr = spearman_rank_corr(p, t_)
        model.train()
        return v_ce, v_reg, v_nce, v_pos, spr

    ws_ce, _, ws_nce, ws_wins, _ = eval_val()
    print(f"[ws] val_ce={ws_ce:.3f}  val_infonce={ws_nce:.3f}  val_pos_wins={ws_wins:.3f}",
           flush=True)

    LOG_K_1 = math.log(K_NEGS + 1)
    NCE_COLLAPSE_THRESH = 0.9 * LOG_K_1
    out_dir = Path(args.out_ckpt).parent; out_dir.mkdir(parents=True, exist_ok=True)
    log_out = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70" / f"train_v{args.variant}_{args.target}.jsonl"
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_out, "a")

    def close_and_stop(status: str, epoch: int, last_metrics: dict):
        print(f"[FAIL-FAST] {status} at epoch {epoch}: {last_metrics}", flush=True)
        log_fp.write(json.dumps({"failed": True, "status": status,
                                    "epoch": epoch, **last_metrics}) + "\n")
        log_fp.flush(); log_fp.close()
        return

    t0 = time.time()
    step = 0
    val_history = []
    reg_mse_prev = float("inf")
    reg_mse_prev2 = float("inf")

    for epoch in range(args.epochs):
        lam_reg = curriculum_lambda(epoch, 0.5, 4) if reg_on else 0.0
        lam_nce = curriculum_lambda(epoch, 0.3, 6) if nce_on else 0.0
        model.train()
        if reg_head is not None: reg_head.train()
        if cpe is not None:
            cpe.train(); proj.train()
        tr_ce = 0.0; tr_reg = 0.0; tr_nce = 0.0; tr_n = 0
        for batch in train_loader:
            s, sm, t, tm, re, rm, po, tgt_v, tgt_v_z, negs = batch
            lr = cosine_warmup_lr(step, args.warmup, total_steps, args.lr)
            for g in optim.param_groups: g["lr"] = lr

            nll = model.likelihood(s, sm, t, tm, re, rm, po)
            ce_loss = nll.mean()

            if not torch.isfinite(ce_loss):
                close_and_stop("nan_ce", epoch, {"step": step, "ce": float(ce_loss.item() if torch.isfinite(ce_loss) else float('nan'))})
                return

            loss = ce_loss
            reg_val = torch.tensor(0.0, device=device)
            nce_val = torch.tensor(0.0, device=device)

            if reg_on or nce_on:
                h = get_h_pooled(model, s, sm, t, tm, re, rm, po)
            if reg_on:
                pred = reg_head(h)
                # MSE on z-normalized target so its magnitude is O(1)
                reg_val = F.mse_loss(pred, -tgt_v_z)
                loss = loss + lam_reg * reg_val
            if nce_on:
                nce_val, _ = infonce_loss(h, po, negs, cpe, proj)
                loss = loss + lam_nce * nce_val

            if not torch.isfinite(loss):
                close_and_stop("nan_total_loss", epoch,
                                {"step": step, "ce": float(ce_loss.item()),
                                 "reg": float(reg_val.item()), "nce": float(nce_val.item())})
                return

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            B = s.shape[0]
            tr_ce += float(ce_loss.item()) * B
            tr_reg += float(reg_val.item()) * B
            tr_nce += float(nce_val.item()) * B
            tr_n += B
            step += 1

            if time.time() - t0 > args.max_wall_seconds:
                close_and_stop("wall_budget_exceeded", epoch, {"step": step,
                                "wall_s": time.time() - t0})
                return

        v_ce, v_reg, v_nce, v_pos, v_spr = eval_val()
        wall = time.time() - t0
        rec = {"epoch": epoch, "step": step, "wall_s": wall,
                "train_ce": tr_ce / max(1, tr_n),
                "train_reg": tr_reg / max(1, tr_n),
                "train_nce": tr_nce / max(1, tr_n),
                "val_ce": v_ce, "val_regression_mse": v_reg,
                "val_infonce": v_nce, "val_pos_wins": v_pos,
                "val_regression_spearman": v_spr,
                "lam_reg": lam_reg, "lam_nce": lam_nce, "lr": lr}
        val_history.append(rec)
        log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
        print(f"[V{args.variant}_{args.target} ep{epoch}] "
               f"ce={rec['train_ce']:.3f}/{v_ce:.3f}  "
               f"reg={rec['train_reg']:.4f}/{v_reg:.4f}  "
               f"nce={rec['train_nce']:.3f}/{v_nce:.3f}  "
               f"pos_win={v_pos:.3f}  spr={v_spr:.3f}  "
               f"λr={lam_reg:.2f} λn={lam_nce:.2f}  wall={wall/60:.1f}min",
               flush=True)

        # Fail-fast
        if v_ce > ws_ce + 2.0:
            close_and_stop("val_ce_ballooned", epoch,
                            {"val_ce": v_ce, "ws_ce": ws_ce})
            # Save partial ckpt anyway
            _save_ckpt(model, reg_head, cpe, proj, args, val_history, pose_mean,
                          pose_std, ds.target_mean, ds.target_std,
                          out_ckpt=args.out_ckpt + ".partial")
            return
        if nce_on and epoch >= 5 and v_nce > NCE_COLLAPSE_THRESH:
            close_and_stop("infonce_collapse", epoch,
                            {"val_nce": v_nce, "threshold": NCE_COLLAPSE_THRESH})
            _save_ckpt(model, reg_head, cpe, proj, args, val_history, pose_mean,
                          pose_std, ds.target_mean, ds.target_std,
                          out_ckpt=args.out_ckpt + ".partial")
            return
        if reg_on and epoch >= 2:
            if v_reg > reg_mse_prev > reg_mse_prev2:
                close_and_stop("regression_worsening_2_epochs", epoch,
                                {"prev2": reg_mse_prev2, "prev": reg_mse_prev,
                                 "cur": v_reg})
                _save_ckpt(model, reg_head, cpe, proj, args, val_history, pose_mean,
                              pose_std, out_ckpt=args.out_ckpt + ".partial")
                return
        reg_mse_prev2, reg_mse_prev = reg_mse_prev, v_reg

    _save_ckpt(model, reg_head, cpe, proj, args, val_history, pose_mean, pose_std,
                  ds.target_mean, ds.target_std, out_ckpt=args.out_ckpt)
    log_fp.close()


def _save_ckpt(model, reg_head, cpe, proj, args, val_history, pose_mean, pose_std,
                target_mean: float, target_std: float,
                out_ckpt: str):
    payload = {"model_state": model.state_dict(),
                "variant": args.variant, "target": args.target,
                "reg_on": args.variant == 4, "nce_on": True,
                "pose_mean": pose_mean.tolist(), "pose_std": pose_std.tolist(),
                "target_mean": float(target_mean),
                "target_std": float(target_std),
                "val_history": val_history}
    if reg_head is not None:
        payload["reg_head_state"] = reg_head.state_dict()
    if cpe is not None:
        payload["cpe_state"] = cpe.state_dict()
        payload["proj_state"] = proj.state_dict()
    torch.save(payload, out_ckpt)
    print(f"[save] {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
