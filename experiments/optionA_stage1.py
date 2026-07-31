"""OptionA Stage 1 — Coord regression head.

Input: pocket ESM-2 embeddings (320-d per residue, up to 18 residues from cache)
       + nucleophile identity token.
Output: 5-vector (bc_local_x, bc_local_y, bc_local_z, bd_angle_deg, phi_planar_rad).

Reuses cached ESM-2 embeddings from data/m1a_triples_v2/esm2_cache_posefix_v3.npz.

Architecture:
    ESM-2 tokens (320-d) → linear proj (256) →
    prepend [READOUT] + nucleophile-id learned tokens →
    3-layer TransformerEncoder (d=256, heads=4) →
    take [READOUT] hidden → MLP → 5-dim (z-scored) →
    de-normalize with train stats.

Loss: MSE on z-scored targets (each dim variance-normalized).

Success: β-C RMSD < 3 Å on held-out test; BD angle MAE < 30°.

Outputs:
    models/optionA_stage1.pt      (state_dict + norm stats)
    data/optionA/stage1_metrics.json
    data/optionA/stage1_test_predictions.parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
OPT = REPO / "data" / "optionA"
CACHE_NPZ = REPO / "data" / "m1a_triples_v2" / "esm2_cache_posefix_v3.npz"
MODELS = REPO / "models"
MODELS.mkdir(parents=True, exist_ok=True)

NUC_VOCAB = ["CYS", "SER", "LYS", "TYR", "THR", "HIS", "ASN", "GLN", "ASP", "GLU", "MET", "TRP", "ARG", "XXX"]
NUC2ID = {n: i for i, n in enumerate(NUC_VOCAB)}
D = 256
MAX_POCKET = 18


def load_cache():
    z = np.load(CACHE_NPZ, allow_pickle=True)
    return {
        "struct_ids": z["struct_ids"].tolist(),
        "row_seq_idx": z["row_seq_idx"],
        "residues_emb": z["residues_emb"],  # (n_seq, MAX_POCKET, 320)
        "residues_mask": z["residues_mask"],  # (n_seq, MAX_POCKET) bool
    }


class PocketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache: dict, target_mean: np.ndarray, target_std: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.sid_to_idx = {s: i for i, s in enumerate(cache["struct_ids"])}
        self.cache = cache
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = np.where(target_std > 1e-6, target_std, 1.0).astype(np.float32)
        # filter to rows with cache entries and valid targets (drop NaN)
        keep = []
        for i, row in self.df.iterrows():
            sid = row["struct_id"]
            if sid not in self.sid_to_idx:
                continue
            bd = row.get("bd_angle"); ph = row.get("phi_planar")
            if bd is None or ph is None:
                continue
            try:
                if math.isnan(float(bd)) or math.isnan(float(ph)):
                    continue
            except Exception:
                continue
            bc = row.get("bc_local_xyz")
            if bc is None or (isinstance(bc, float) and math.isnan(bc)):
                continue
            keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["struct_id"]
        cache_i = self.sid_to_idx[sid]
        seq_i = int(self.cache["row_seq_idx"][cache_i])
        emb = self.cache["residues_emb"][seq_i]  # (18, 320)
        mask = self.cache["residues_mask"][seq_i]  # (18,) bool
        nuc = row["nucleophile_resname"] if row["nucleophile_resname"] in NUC2ID else "XXX"
        nuc_id = NUC2ID[nuc]
        bc = np.array(row["bc_local_xyz"], dtype=np.float32)
        y_raw = np.array([bc[0], bc[1], bc[2], float(row["bd_angle"]), float(row["phi_planar"])], dtype=np.float32)
        y_norm = (y_raw - self.target_mean) / self.target_std
        return {
            "emb": torch.from_numpy(emb.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.bool_)),
            "nuc_id": torch.tensor(nuc_id, dtype=torch.long),
            "y_norm": torch.from_numpy(y_norm),
            "y_raw": torch.from_numpy(y_raw),
        }


def collate(batch):
    return {
        "emb": torch.stack([b["emb"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "nuc_id": torch.stack([b["nuc_id"] for b in batch]),
        "y_norm": torch.stack([b["y_norm"] for b in batch]),
        "y_raw": torch.stack([b["y_raw"] for b in batch]),
    }


class Stage1Model(nn.Module):
    def __init__(self, esm_dim=320, d=D, n_layers=3, n_heads=4, out_dim=5):
        super().__init__()
        self.proj = nn.Linear(esm_dim, d)
        self.readout = nn.Parameter(torch.randn(d) * 0.02)
        self.nuc_embed = nn.Embedding(len(NUC_VOCAB), d)
        enc_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))

    def forward(self, emb, mask, nuc_id):
        # emb: (B, K, 320)  mask: (B, K) True for valid
        B, K, _ = emb.shape
        x = self.proj(emb)  # (B, K, d)
        readout = self.readout.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        nuc_tok = self.nuc_embed(nuc_id).unsqueeze(1)  # (B, 1, d)
        x = torch.cat([readout, nuc_tok, x], dim=1)  # (B, K+2, d)
        # attention mask: True where PADDING for nn.Transformer key_padding_mask
        pad = torch.zeros(B, K + 2, dtype=torch.bool, device=emb.device)
        pad[:, 2:] = ~mask  # invert; pad positions get True
        h = self.encoder(x, src_key_padding_mask=pad)
        readout_h = h[:, 0]  # (B, d)
        y = self.head(readout_h)  # (B, out_dim)
        return y


def compute_metrics(y_pred_raw: np.ndarray, y_true_raw: np.ndarray):
    # (N, 5) — bc_x, bc_y, bc_z, bd_deg, phi_rad
    d_pred = y_pred_raw[:, :3]
    d_true = y_true_raw[:, :3]
    rmsd = float(np.sqrt(np.mean(np.sum((d_pred - d_true) ** 2, axis=1))))
    med_err = float(np.median(np.linalg.norm(d_pred - d_true, axis=1)))
    bd_mae = float(np.mean(np.abs(y_pred_raw[:, 3] - y_true_raw[:, 3])))
    # phi is periodic (rad); wrap to [-pi, pi]
    dp = y_pred_raw[:, 4] - y_true_raw[:, 4]
    dp = np.arctan2(np.sin(dp), np.cos(dp))
    phi_mae = float(np.mean(np.abs(dp)))
    return {
        "bc_rmsd_A": rmsd,
        "bc_median_err_A": med_err,
        "bd_angle_mae_deg": bd_mae,
        "phi_mae_rad": phi_mae,
        "phi_mae_deg": float(phi_mae * 180 / math.pi),
    }


def compute_target_stats(df: pd.DataFrame):
    ys = []
    for _, row in df.iterrows():
        bd = row.get("bd_angle"); ph = row.get("phi_planar")
        if bd is None or ph is None:
            continue
        try:
            bdf = float(bd); phf = float(ph)
            if math.isnan(bdf) or math.isnan(phf):
                continue
        except Exception:
            continue
        bc = row.get("bc_local_xyz")
        if bc is None:
            continue
        ys.append([bc[0], bc[1], bc[2], bdf, phf])
    ys = np.array(ys, dtype=np.float64)
    return ys.mean(axis=0), ys.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out-ckpt", default=str(MODELS / "optionA_stage1.pt"))
    ap.add_argument("--out-metrics", default=str(OPT / "stage1_metrics.json"))
    ap.add_argument("--out-preds", default=str(OPT / "stage1_test_predictions.parquet"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[stage1] device={device}", flush=True)

    print("[stage1] loading data...", flush=True)
    tr = pd.read_parquet(OPT / "train.parquet")
    va = pd.read_parquet(OPT / "val.parquet")
    te = pd.read_parquet(OPT / "test.parquet")
    cache = load_cache()

    y_mean, y_std = compute_target_stats(tr)
    print(f"[stage1] y_mean={y_mean.round(3).tolist()} y_std={y_std.round(3).tolist()}", flush=True)

    ds_tr = PocketDataset(tr, cache, y_mean, y_std)
    ds_va = PocketDataset(va, cache, y_mean, y_std)
    ds_te = PocketDataset(te, cache, y_mean, y_std)
    print(f"[stage1] rows kept: train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, collate_fn=collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, collate_fn=collate, num_workers=0)

    model = Stage1Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)

    best_val = float("inf")
    best_state = None
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for batch in dl_tr:
            emb = batch["emb"].to(device)
            mask = batch["mask"].to(device)
            nuc = batch["nuc_id"].to(device)
            y_norm = batch["y_norm"].to(device)
            pred_norm = model(emb, mask, nuc)
            loss = F.mse_loss(pred_norm, y_norm)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * emb.shape[0]
            n += emb.shape[0]
        sched.step()
        tr_loss /= max(n, 1)

        # val
        model.eval()
        with torch.no_grad():
            v_pred_raw, v_true_raw = [], []
            for batch in dl_va:
                emb = batch["emb"].to(device)
                mask = batch["mask"].to(device)
                nuc = batch["nuc_id"].to(device)
                pred_norm = model(emb, mask, nuc)
                pred_raw = pred_norm * y_std_t + y_mean_t
                v_pred_raw.append(pred_raw.cpu().numpy())
                v_true_raw.append(batch["y_raw"].numpy())
            v_pred_raw = np.concatenate(v_pred_raw)
            v_true_raw = np.concatenate(v_true_raw)
            vm = compute_metrics(v_pred_raw, v_true_raw)
        val_score = vm["bc_rmsd_A"]
        log.append({"ep": ep, "tr_loss": tr_loss, "val": vm})
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elapsed = time.time() - t0
        print(f"[stage1] ep={ep:03d} tr_loss={tr_loss:.4f} val_rmsd={vm['bc_rmsd_A']:.3f}Å bd_mae={vm['bd_angle_mae_deg']:.1f}° phi_mae={vm['phi_mae_deg']:.1f}° [{elapsed:.0f}s]", flush=True)

        # progress log every ~10 epochs
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            (OPT / "progress.md").write_text(
                f"# OptionA Stage 1 progress\n\nepoch {ep+1}/{args.epochs}\nlast val rmsd: {vm['bc_rmsd_A']:.3f} Å\nbest val rmsd: {best_val:.3f} Å\nelapsed: {elapsed:.0f}s\n"
            )

    # Load best, evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t_pred_raw, t_true_raw = [], []
        sids = []
        for i in range(len(ds_te)):
            item = ds_te[i]
            sids.append(ds_te.df.iloc[i]["struct_id"])
        for batch in dl_te:
            emb = batch["emb"].to(device)
            mask = batch["mask"].to(device)
            nuc = batch["nuc_id"].to(device)
            pred_norm = model(emb, mask, nuc)
            pred_raw = pred_norm * y_std_t + y_mean_t
            t_pred_raw.append(pred_raw.cpu().numpy())
            t_true_raw.append(batch["y_raw"].numpy())
        t_pred_raw = np.concatenate(t_pred_raw)
        t_true_raw = np.concatenate(t_true_raw)
    tm = compute_metrics(t_pred_raw, t_true_raw)
    print(f"[stage1] TEST: rmsd={tm['bc_rmsd_A']:.3f}Å median={tm['bc_median_err_A']:.3f}Å bd_mae={tm['bd_angle_mae_deg']:.1f}° phi_mae={tm['phi_mae_deg']:.1f}°", flush=True)

    # Save ckpt
    torch.save({
        "state_dict": best_state,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "args": vars(args),
        "test_metrics": tm,
    }, args.out_ckpt)
    (OPT / "stage1_metrics.json").write_text(json.dumps({
        "val_best_rmsd": best_val,
        "test": tm,
        "log": log,
    }, indent=2))
    preds_df = pd.DataFrame({
        "struct_id": sids,
        "pred_bc_x": t_pred_raw[:, 0],
        "pred_bc_y": t_pred_raw[:, 1],
        "pred_bc_z": t_pred_raw[:, 2],
        "pred_bd": t_pred_raw[:, 3],
        "pred_phi": t_pred_raw[:, 4],
        "true_bc_x": t_true_raw[:, 0],
        "true_bc_y": t_true_raw[:, 1],
        "true_bc_z": t_true_raw[:, 2],
        "true_bd": t_true_raw[:, 3],
        "true_phi": t_true_raw[:, 4],
    })
    preds_df.to_parquet(args.out_preds)
    print(f"[stage1] wrote {args.out_ckpt}, {args.out_metrics}, {args.out_preds}", flush=True)
    (OPT / "progress.md").write_text(
        f"# OptionA Stage 1 progress\n\nDONE\ntest rmsd: {tm['bc_rmsd_A']:.3f} Å (target < 3 Å)\ntest bd mae: {tm['bd_angle_mae_deg']:.1f}° (target < 30°)\ntest phi mae: {tm['phi_mae_deg']:.1f}°\n"
    )


if __name__ == "__main__":
    main()
