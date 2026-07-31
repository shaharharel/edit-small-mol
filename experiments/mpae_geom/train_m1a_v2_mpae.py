#!/usr/bin/env python3
"""4-variant m1a v2 training for the mpae-geometry experiment.

Warm-starts models/m1a_v2.ckpt, applies two auxiliary heads:
  - Regression head g_reg(h_pooled) → −mpae_warhead_cys (MSE)
  - Fixed InfoNCE: separate contrastive_pose_encoder + projection head g_proj(h_pooled)
    K=4 negs (2 real chem-matched + 2 synthetic ±3σ θ + φ-flip 180°)

Variants (factorial):
  V1: OFF/OFF (CE only baseline retrain)
  V2: ON /OFF (regression only)
  V3: OFF/ON  (InfoNCE only, fixed)
  V4: ON /ON  (both)

Data: data/paper_pair_training/mpae_geom/{labeled_mols,esm_cache,pose_stats}.
Ckpt output: models/m1a_v2_mpae_V{1,2,3,4}.ckpt

10 epochs, batch 16, LR 1e-4, warmup 500. Curriculum:
  λ_reg 0→0.5 over ep 0-3 (when reg=ON)
  λ_nce 0→0.3 over ep 0-5 (when nce=ON)

Guardrails:
  Val CE ≤ base+2 (~19)
  Val regression MSE < var(mpae) by ep 5
  Val InfoNCE < 0.9*log(K+1)=1.45 by ep 5
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
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/mpae_geom"))

from m1a_v2_model import load_m1a_v2, save_m1a_v2, M1aV2ConditionedModel  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

D_MODEL = 256
POSE_DIM = 3
K_NEGS = 4


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


class MpaeGeomDataset(Dataset):
    def __init__(self, labeled_npz: Path, esm_npz: Path, vocabulary, tokenizer,
                  pose_mean: np.ndarray, pose_std: np.ndarray, max_len: int = 128):
        super().__init__()
        L = np.load(labeled_npz, allow_pickle=True)
        E = np.load(esm_npz, allow_pickle=True)
        self.smiles = L["smiles"]
        self.pose_boltz = L["pose_boltz"].astype(np.float32)  # (N,3) UNNORMALIZED
        self.mpae = L["mpae_warhead_cys"].astype(np.float32)
        self.q_bin = L["q_bin"]
        self.target = L["target"]
        self.residues_emb = E["residues_emb"]
        self.residues_mask = E["residues_mask"]
        self.row_seq_idx = E["row_seq_idx"]
        self.target_names = E["target_names"]
        self.pose_mean = pose_mean.astype(np.float32)
        self.pose_std = pose_std.astype(np.float32)
        # Filter tokenizable
        self.vocabulary = vocabulary; self.tokenizer = tokenizer; self.max_len = max_len
        ok = []
        for i, smi in enumerate(self.smiles):
            try:
                toks = tokenizer.tokenize(smi)
                if len(toks) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok, dtype=np.int64)
        print(f"[Dataset] {len(self.idx)}/{len(self.smiles)} usable rows, "
               f"pockets={self.residues_emb.shape[0]}", flush=True)

    def __len__(self) -> int:
        return len(self.idx)

    def normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        return (pose - self.pose_mean) / np.clip(self.pose_std, 1e-6, None)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        canon = str(self.smiles[i])
        src_smi = randomize_smi(canon)
        pose_norm = self.normalize_pose(self.pose_boltz[i])
        return {
            "src_smi": src_smi,
            "tgt_smi": canon,
            "residues_emb": self.residues_emb[self.row_seq_idx[i]],
            "residues_mask": self.residues_mask[self.row_seq_idx[i]],
            "pose": pose_norm.astype(np.float32),
            "pose_raw": self.pose_boltz[i].astype(np.float32),
            "mpae": self.mpae[i],
            "target": str(self.target[i]),
            "row": i,
        }


def _build_negatives(batch, ds: MpaeGeomDataset) -> np.ndarray:
    """Return (B, K, 3) NORMALIZED pose negatives.
    2 real chem-matched (different mol on same target, sampled from full ds)
    + 2 synthetic (θ±3σ, φ-flip 180°).
    """
    B = len(batch)
    negs = np.zeros((B, K_NEGS, 3), dtype=np.float32)
    tgt_pool: dict[str, list[int]] = {}
    for i in range(len(ds.idx)):
        row = int(ds.idx[i])
        t = str(ds.target[row])
        tgt_pool.setdefault(t, []).append(row)
    rng = np.random.default_rng()
    for b_i, b in enumerate(batch):
        tgt = b["target"]
        pool = tgt_pool.get(tgt, list(range(len(ds.smiles))))
        pool = [p for p in pool if p != b["row"]]
        if len(pool) >= 2:
            picks = rng.choice(len(pool), size=2, replace=False)
            for k_i, pi in enumerate(picks):
                row = pool[int(pi)]
                negs[b_i, k_i] = ds.normalize_pose(ds.pose_boltz[row])
        # Two synthetic negs: perturb θ by ±3σ_θ + flip φ 180°
        pos = b["pose_raw"].copy()
        sig_th = float(ds.pose_std[1])
        neg_a = pos.copy(); neg_a[1] += 3.0 * sig_th
        neg_b = pos.copy(); neg_b[1] -= 3.0 * sig_th; neg_b[2] = ((neg_b[2] + 180.0 + 180.0) % 360.0) - 180.0
        negs[b_i, 2] = ds.normalize_pose(neg_a)
        negs[b_i, 3] = ds.normalize_pose(neg_b)
    return negs


def collate_factory(vocabulary, tokenizer, device, ds: MpaeGeomDataset, max_len: int = 128):
    def collate(batch):
        B = len(batch)
        src_seqs, tgt_seqs = [], []
        for b in batch:
            src_seqs.append(np.array(vocabulary.encode(tokenizer.tokenize(b["src_smi"])), dtype=np.int64))
            tgt_seqs.append(np.array(vocabulary.encode(tokenizer.tokenize(b["tgt_smi"])), dtype=np.int64))
        L_src = max(len(s) for s in src_seqs); L_tgt = max(len(s) for s in tgt_seqs)
        src = np.zeros((B, L_src), dtype=np.int64); trg = np.zeros((B, L_tgt), dtype=np.int64)
        for j, s in enumerate(src_seqs): src[j, :len(s)] = s
        for j, s in enumerate(tgt_seqs): trg[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        trg_t = torch.from_numpy(trg).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        trg_mask = _make_std_mask(trg_t[:, :-1], 0)
        res_emb = torch.from_numpy(np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(np.stack([b["pose"] for b in batch])).to(device)
        mpae = torch.from_numpy(np.stack([b["mpae"] for b in batch])).float().to(device)
        negs_np = _build_negatives(batch, ds)
        negs = torch.from_numpy(negs_np).to(device)  # (B,K,3) normalized
        return src_t, src_mask, trg_t, trg_mask, res_emb, res_mask, pose, mpae, negs
    return collate


def _make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask & sub_mask


class RegressionHead(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, h_pooled):
        return self.net(h_pooled).squeeze(-1)


class ContrastivePoseEncoder(nn.Module):
    """Separate pose encoder (NOT shared with h_[POSE] injection).

    3 → 64 → 256, LayerNorm at output.
    """
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
    return peak * min(1.0, (epoch + 1) / ramp_epochs)


def get_h_pooled(model: M1aV2ConditionedModel, src, src_mask, trg, trg_mask,
                  res_emb, res_mask, pose_norm):
    """Return decoder final-layer h_pooled (mean over target positions).

    Reproduces model.base.network forward path up to decoder outputs, mean-pooled.
    """
    memory_ext, src_mask_ext = model._build_conditioned_memory(src, src_mask, res_emb, res_mask, pose_norm)
    trg_in = trg[:, :-1]
    dec_out = model.base.network.decoder(model.base.network.tgt_embed(trg_in), memory_ext, src_mask_ext, trg_mask)
    # dec_out: (B, L, D). Mean over valid target positions using trg_mask.
    # trg_mask is (B, 1, L, L) after subsequent_mask expansion; take dim -1 diag as pad
    pad_mask = (trg_in != 0).float().unsqueeze(-1)  # (B, L, 1)
    denom = pad_mask.sum(dim=1).clamp_min(1e-6)
    pooled = (dec_out * pad_mask).sum(dim=1) / denom
    return pooled  # (B, D)


def infonce_loss(h_pooled, pose_norm_pos, negs, cpe: ContrastivePoseEncoder, proj: ProjectionHead, temperature: float = 0.1):
    """Symmetric-like InfoNCE: mol → pose in {pos, K negs}. K=4.

    Returns (loss, mean_pos_rank_prob).
    """
    B = h_pooled.size(0)
    K = negs.size(1)
    z_mol = torch.nn.functional.normalize(proj(h_pooled), dim=-1)          # (B, D)
    z_pos = torch.nn.functional.normalize(cpe(pose_norm_pos), dim=-1)      # (B, D)
    z_neg = torch.nn.functional.normalize(cpe(negs.reshape(-1, POSE_DIM)), dim=-1).view(B, K, -1)  # (B, K, D)
    sim_pos = (z_mol * z_pos).sum(-1) / temperature                        # (B,)
    sim_neg = torch.einsum("bd,bkd->bk", z_mol, z_neg) / temperature       # (B, K)
    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)             # (B, K+1)
    labels = torch.zeros(B, dtype=torch.long, device=h_pooled.device)
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
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
    ap.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--labeled_npz", default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/labeled_mols.npz"))
    ap.add_argument("--esm_cache",   default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/esm_cache.npz"))
    ap.add_argument("--pose_stats",  default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/pose_stats.json"))
    ap.add_argument("--warm_start",  default=str(PROJECT_ROOT / "models/m1a_v2.ckpt"))
    ap.add_argument("--prior",       default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--out_ckpt",    default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--warmup",      type=int, default=500)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--val_frac",    type=float, default=0.10)
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    reg_on = args.variant in (2, 4)
    nce_on = args.variant in (3, 4)
    print(f"Variant V{args.variant}: reg_on={reg_on}  nce_on={nce_on}", flush=True)

    if args.out_ckpt is None:
        args.out_ckpt = str(PROJECT_ROOT / f"models/m1a_v2_mpae_V{args.variant}.ckpt")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ps = json.loads(Path(args.pose_stats).read_text())
    pose_mean = np.array(ps["pose_mean"], dtype=np.float32)
    pose_std  = np.array(ps["pose_std"], dtype=np.float32)
    print(f"Pose normalizer: mean={pose_mean}  std={pose_std}", flush=True)

    # Load M1a v2 model warm-start
    print("Loading warm-start M1a v2 model...", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.warm_start,
                         pose_mean=pose_mean, pose_std=pose_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params/1e6:.1f}M", flush=True)

    # Aux heads
    reg_head = RegressionHead(D_MODEL).to(device) if reg_on else None
    cpe = ContrastivePoseEncoder().to(device) if nce_on else None
    proj = ProjectionHead(D_MODEL).to(device) if nce_on else None

    # Data
    ds = MpaeGeomDataset(Path(args.labeled_npz), Path(args.esm_cache),
                          model.base.vocabulary, model.base.tokenizer,
                          pose_mean, pose_std, max_len=128)
    N = len(ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"Train={len(train_idx)} Val={len(val_idx)}", flush=True)
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_factory(model.base.vocabulary, model.base.tokenizer, device, ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    if reg_head is not None:
        params += list(reg_head.parameters())
    if cpe is not None:
        params += list(cpe.parameters()) + list(proj.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps/epoch={steps_per_epoch}  total={total_steps}", flush=True)

    # Trivial baseline for regression: predict mean(mpae) on val
    mpae_all = ds.mpae
    val_mpae_mean = float(np.mean(mpae_all))
    trivial_mse = float(np.var(mpae_all))
    print(f"Trivial predict-mean MSE = var(mpae) = {trivial_mse:.4f}", flush=True)

    out_dir = Path(args.out_ckpt).parent; out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"m1a_v2_mpae_V{args.variant}_log.jsonl"
    log_fp = open(log_path, "a")

    t0 = time.time()
    step = 0
    LOG_K_1 = math.log(K_NEGS + 1)
    for epoch in range(args.epochs):
        lam_reg = curriculum_lambda(epoch, 0.5, 4) if reg_on else 0.0
        lam_nce = curriculum_lambda(epoch, 0.3, 6) if nce_on else 0.0
        model.train()
        if reg_head is not None: reg_head.train()
        if cpe is not None: cpe.train(); proj.train()
        tr_ce = 0.0; tr_reg = 0.0; tr_nce = 0.0; tr_n = 0
        for batch in train_loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose, mpae, negs = batch
            lr = cosine_warmup_lr(step, args.warmup, total_steps, args.lr)
            for g in optim.param_groups: g["lr"] = lr

            # CE loss (base autoencoder)
            nll = model.likelihood(src, src_mask, trg, trg_mask, res_emb, res_mask, pose)
            ce_loss = nll.mean()

            loss = ce_loss
            reg_val = torch.tensor(0.0, device=device)
            nce_val = torch.tensor(0.0, device=device)

            # Aux heads use h_pooled on the SAME forward's decoder output
            if reg_on or nce_on:
                h_pooled = get_h_pooled(model, src, src_mask, trg, trg_mask,
                                          res_emb, res_mask, pose)
            if reg_on:
                pred = reg_head(h_pooled)
                reg_val = torch.nn.functional.mse_loss(pred, -mpae)
                loss = loss + lam_reg * reg_val
            if nce_on:
                nce_val, _ = infonce_loss(h_pooled, pose, negs, cpe, proj)
                loss = loss + lam_nce * nce_val

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            B = src.shape[0]
            tr_ce += float(ce_loss.item()) * B
            tr_reg += float(reg_val.item()) * B
            tr_nce += float(nce_val.item()) * B
            tr_n += B
            step += 1

        # Validation
        model.eval()
        if reg_head is not None: reg_head.eval()
        if cpe is not None: cpe.eval(); proj.eval()
        v_ce = 0.0; v_reg = 0.0; v_nce = 0.0; v_pos = 0.0; v_n = 0
        preds_all = []; targs_all = []
        with torch.no_grad():
            for batch in val_loader:
                src, src_mask, trg, trg_mask, res_emb, res_mask, pose, mpae, negs = batch
                nll = model.likelihood(src, src_mask, trg, trg_mask, res_emb, res_mask, pose)
                ce = nll.mean()
                v_ce += float(ce.item()) * src.shape[0]
                if reg_on or nce_on:
                    h_pooled = get_h_pooled(model, src, src_mask, trg, trg_mask,
                                              res_emb, res_mask, pose)
                if reg_on:
                    pred = reg_head(h_pooled)
                    v_reg += float(torch.nn.functional.mse_loss(pred, -mpae).item()) * src.shape[0]
                    preds_all.append(pred.cpu().numpy())
                    targs_all.append((-mpae).cpu().numpy())
                if nce_on:
                    nce_loss, pos_win = infonce_loss(h_pooled, pose, negs, cpe, proj)
                    v_nce += float(nce_loss.item()) * src.shape[0]
                    v_pos += pos_win * src.shape[0]
                v_n += src.shape[0]
        v_ce /= max(1, v_n); v_reg /= max(1, v_n); v_nce /= max(1, v_n); v_pos /= max(1, v_n)
        spr = float("nan")
        if reg_on and preds_all:
            p = np.concatenate(preds_all); t = np.concatenate(targs_all)
            spr = spearman_rank_corr(p, t)

        wall = time.time() - t0
        rec = {"epoch": epoch, "step": step, "wall_s": wall,
                "train_ce": tr_ce / max(1, tr_n),
                "train_reg": tr_reg / max(1, tr_n),
                "train_nce": tr_nce / max(1, tr_n),
                "val_ce": v_ce, "val_reg_mse": v_reg,
                "val_nce": v_nce, "val_pos_wins": v_pos, "val_reg_spearman": spr,
                "lam_reg": lam_reg, "lam_nce": lam_nce, "lr": lr}
        log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
        print(f"[V{args.variant} ep{epoch}] ce={rec['train_ce']:.3f}/{v_ce:.3f}  "
               f"reg={rec['train_reg']:.4f}/{v_reg:.4f}  nce={rec['train_nce']:.3f}/{v_nce:.3f}  "
               f"pos_win={v_pos:.3f}  spr={spr:.3f}  λr={lam_reg:.2f} λn={lam_nce:.2f}  wall={wall/60:.1f}min",
               flush=True)

        # Guardrails
        if v_ce > 19.5:
            print(f"[WARN] val CE {v_ce:.3f} exceeds guardrail (~19).", flush=True)
        if reg_on and epoch >= 5 and v_reg >= trivial_mse:
            print(f"[WARN] regression MSE {v_reg:.4f} still ≥ trivial {trivial_mse:.4f} by ep 5.", flush=True)
        if nce_on and epoch >= 5 and v_nce > 0.9 * LOG_K_1:
            print(f"[WARN] InfoNCE {v_nce:.3f} > 0.9*log({K_NEGS+1}) — possible collapse.", flush=True)

    # Save ckpt (base + aux heads)
    payload = {"model_state": model.state_dict(),
                "variant": args.variant,
                "reg_on": reg_on, "nce_on": nce_on,
                "pose_mean": pose_mean.tolist(),
                "pose_std":  pose_std.tolist()}
    if reg_head is not None:
        payload["reg_head_state"] = reg_head.state_dict()
    if cpe is not None:
        payload["cpe_state"] = cpe.state_dict()
        payload["proj_state"] = proj.state_dict()
    torch.save(payload, args.out_ckpt)
    print(f"FINAL: {args.out_ckpt}", flush=True)
    log_fp.close()


if __name__ == "__main__":
    main()
