"""OptionA Stage 2 — Coord-conditioned SMILES decoder trainer.

Training data: RANDOM OTHER covalent SMILES from training set as SRC, target canonical SMILES as TGT,
    conditioned on 5-vector (bc_local_xyz, bd_angle_deg, phi_planar_rad).

CRITICAL DESIGN NOTE: previous version used src == tgt (autoencoder reconstruction).
That task is trivially solvable by copying input; the model learned to ignore coord
conditioning entirely (v2-cond failure reproduced — QA `dynamic_diff_across_coords: 0`).

Fix: src is randomly resampled per epoch from the training pool. Output CANNOT be
predicted from src alone; the coord vector is the only signal that specifies which
tgt to produce for a given src. Aux loss weight raised 0.1 → 1.0 for good measure.

Loss = SMILES CE + λ * MSE_aux (λ=1.0). Aux head reconstructs the 5-vector from
mean-pooled decoder hidden state, keeping coord info alive through the decoder stack.

Init: covFT prior weights (frozen 1 epoch, then unfrozen with LR 5e-6).
Coord conditioning MLP + aux head are always trainable.

Outputs:
    models/optionA_stage2.pt (state_dict of CoordConditionedMol2Mol + vocab)
    data/optionA/stage2_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(REPO / "experiments"))
from optionA_mol2mol import (
    CoordConditionedMol2Mol,
    load_mol2mol_prior,
    subsequent_mask,
    tokenize_smiles,
    detokenize,
)

OPT = REPO / "data" / "optionA"
MODELS = REPO / "models"
PRIOR = MODELS / "reinvent4_mol2mol_covalent_ft.prior"


class Stage2Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokens: dict, bos: int, eos: int, pad: int, max_len: int, y_mean: np.ndarray, y_std: np.ndarray, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.tokens = tokens
        self.bos, self.eos, self.pad = bos, eos, pad
        self.max_len = max_len
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = np.where(y_std > 1e-6, y_std, 1.0).astype(np.float32)
        # Filter rows with valid SMILES and coord
        keep = []
        for i, row in self.df.iterrows():
            smi = row.get("canon_smi")
            if not isinstance(smi, str) or not smi:
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
            if bc is None:
                continue
            keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)
        self._rng = np.random.default_rng(seed)
        # Pre-tokenize all SMILES once (fast lookup later)
        self._tok_cache = [
            np.array(tokenize_smiles(s, self.tokens, self.bos, self.eos, self.pad, self.max_len), dtype=np.int64)
            for s in self.df["canon_smi"].tolist()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # TGT: current row's SMILES
        toks_tgt = self._tok_cache[idx]
        # SRC: a RANDOM OTHER SMILES from the training pool.
        # This forces the model to use the coord vector to know what to output —
        # src alone does not determine tgt (the row's canonical SMILES).
        # This is the FIX for the v2-cond-style failure where src == tgt makes
        # coord conditioning ignorable.
        src_idx = int(self._rng.integers(0, len(self._tok_cache)))
        # Avoid the trivial identical case (occasionally will still coincide, that's fine)
        if src_idx == idx and len(self._tok_cache) > 1:
            src_idx = (src_idx + 1) % len(self._tok_cache)
        toks_src = self._tok_cache[src_idx]
        bc = row["bc_local_xyz"]
        y_raw = np.array([bc[0], bc[1], bc[2], float(row["bd_angle"]), float(row["phi_planar"])], dtype=np.float32)
        y_norm = (y_raw - self.y_mean) / self.y_std
        return {
            "src": torch.from_numpy(toks_src.copy()),
            "tgt_in": torch.from_numpy(toks_tgt[:-1].copy()),
            "tgt_out": torch.from_numpy(toks_tgt[1:].copy()),
            "y_norm": torch.from_numpy(y_norm),
            "y_raw": torch.from_numpy(y_raw),
        }


def collate(batch):
    return {
        "src": torch.stack([b["src"] for b in batch]),
        "tgt_in": torch.stack([b["tgt_in"] for b in batch]),
        "tgt_out": torch.stack([b["tgt_out"] for b in batch]),
        "y_norm": torch.stack([b["y_norm"] for b in batch]),
        "y_raw": torch.stack([b["y_raw"] for b in batch]),
    }


def make_masks(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(1)  # (B, 1, S)
    tgt_mask = (tgt != pad).unsqueeze(1)
    T = tgt.size(1)
    sub = subsequent_mask(T).to(tgt.device)
    tgt_mask = tgt_mask & sub
    return src_mask, tgt_mask


def train_epoch(model, dl, opt, pad, aux_weight, device):
    model.train()
    total_ce, total_aux, n = 0.0, 0.0, 0
    for batch in dl:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        y_norm = batch["y_norm"].to(device)
        src_mask, tgt_mask = make_masks(src, tgt_in, pad)
        logp, aux_pred, _h = model(src, tgt_in, src_mask, tgt_mask, y_norm)
        # CE
        ce = F.nll_loss(logp.reshape(-1, logp.size(-1)), tgt_out.reshape(-1), ignore_index=pad)
        aux = F.mse_loss(aux_pred, y_norm)
        loss = ce + aux_weight * aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        B = src.size(0)
        total_ce += float(ce.item()) * B
        total_aux += float(aux.item()) * B
        n += B
    return total_ce / max(n, 1), total_aux / max(n, 1)


def eval_epoch(model, dl, pad, aux_weight, device):
    model.eval()
    total_ce, total_aux, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in dl:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            y_norm = batch["y_norm"].to(device)
            src_mask, tgt_mask = make_masks(src, tgt_in, pad)
            logp, aux_pred, _h = model(src, tgt_in, src_mask, tgt_mask, y_norm)
            ce = F.nll_loss(logp.reshape(-1, logp.size(-1)), tgt_out.reshape(-1), ignore_index=pad)
            aux = F.mse_loss(aux_pred, y_norm)
            B = src.size(0)
            total_ce += float(ce.item()) * B
            total_aux += float(aux.item()) * B
            n += B
    return total_ce / max(n, 1), total_aux / max(n, 1)


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
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--frozen-epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-unfrozen", type=float, default=5e-6)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--prior", default=str(PRIOR))
    ap.add_argument("--out-ckpt", default=str(MODELS / "optionA_stage2.pt"))
    ap.add_argument("--out-metrics", default=str(OPT / "stage2_metrics.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[stage2] device={device}", flush=True)

    print("[stage2] loading prior...", flush=True)
    base, vocab, prior_max_len, missing, unexpected = load_mol2mol_prior(args.prior, device=device)
    tokens = vocab["tokens"]
    pad = vocab["pad_token"]; bos = vocab["bos_token"]; eos = vocab["eos_token"]
    print(f"[stage2] prior loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    print("[stage2] loading data...", flush=True)
    tr = pd.read_parquet(OPT / "train.parquet")
    va = pd.read_parquet(OPT / "val.parquet")
    y_mean, y_std = compute_target_stats(tr)
    print(f"[stage2] y_mean={y_mean.round(3).tolist()} y_std={y_std.round(3).tolist()}", flush=True)

    ds_tr = Stage2Dataset(tr, tokens, bos, eos, pad, args.max_len, y_mean, y_std)
    ds_va = Stage2Dataset(va, tokens, bos, eos, pad, args.max_len, y_mean, y_std)
    print(f"[stage2] train={len(ds_tr)} val={len(ds_va)}", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, collate_fn=collate, num_workers=0)

    model = CoordConditionedMol2Mol(base).to(device)
    # Verify aux head + coord_mlp are on device
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_new = sum(p.numel() for p in list(model.coord_mlp.parameters()) + list(model.aux_head.parameters()))
    print(f"[stage2] total params={n_params_total/1e6:.2f}M   new (coord_mlp+aux)={n_params_new/1e3:.1f}k", flush=True)

    # Freeze base for first `frozen_epochs`
    for p in model.base.parameters():
        p.requires_grad = False
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[stage2] frozen-phase trainable={sum(p.numel() for p in train_params)/1e3:.1f}k", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    log = []
    t0 = time.time()
    best_val = float("inf"); best_state = None
    for ep in range(args.epochs):
        if ep == args.frozen_epochs:
            # unfreeze base
            for p in model.base.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr_unfrozen, weight_decay=1e-4)
            print(f"[stage2] ep={ep} UNFREEZE base, lr={args.lr_unfrozen}", flush=True)
        ce, aux = train_epoch(model, dl_tr, opt, pad, args.aux_weight, device)
        vce, vaux = eval_epoch(model, dl_va, pad, args.aux_weight, device)
        elapsed = time.time() - t0
        log.append({"ep": ep, "tr_ce": ce, "tr_aux": aux, "val_ce": vce, "val_aux": vaux, "t": elapsed})
        vs = vce + args.aux_weight * vaux
        if vs < best_val:
            best_val = vs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[stage2] ep={ep:02d} tr_ce={ce:.4f} tr_aux={aux:.4f}  val_ce={vce:.4f} val_aux={vaux:.4f}  [{elapsed:.0f}s]", flush=True)

        (OPT / "progress.md").write_text(
            f"# OptionA Stage 2 progress\n\nep {ep+1}/{args.epochs}\ntr_ce {ce:.4f} val_ce {vce:.4f}\nbest val {best_val:.4f}\nelapsed {elapsed:.0f}s\n"
        )

    # Save best
    torch.save({
        "state_dict": best_state,
        "vocab": vocab,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "args": vars(args),
    }, args.out_ckpt)
    (OPT / "stage2_metrics.json").write_text(json.dumps({
        "best_val": best_val,
        "log": log,
    }, indent=2))

    # ----- Conditioning sensitivity check (N=20 samples per coord for statistics) -----
    model.load_state_dict(best_state)
    model.eval()
    # Use a fixed anchor SMILES (first val row)
    anchor_smi = va.iloc[0]["canon_smi"]
    ids = tokenize_smiles(anchor_smi, tokens, bos, eos, pad, args.max_len)
    src_single = torch.tensor([ids], dtype=torch.long, device=device)
    # Two very different coord vectors (opposite octants + very different BD angle + phi)
    c1_raw = np.array([1.5, -0.5, 0.5, 105.0, 3.10], dtype=np.float32)
    c2_raw = np.array([-2.5, 1.5, -0.7, 60.0, 0.5], dtype=np.float32)
    c1 = torch.tensor([(c1_raw - y_mean) / np.where(y_std > 1e-6, y_std, 1.0)], dtype=torch.float32, device=device)
    c2 = torch.tensor([(c2_raw - y_mean) / np.where(y_std > 1e-6, y_std, 1.0)], dtype=torch.float32, device=device)
    N_SAMPLES = 20
    from collections import Counter
    def sample_smis(coord, n):
        smis = []
        for k in range(n):
            torch.manual_seed(1000 + k)
            src = src_single.repeat(1, 1)
            src_mask = (src != pad).unsqueeze(1)
            out = model.sample(src, src_mask, coord, bos, eos, max_len=args.max_len, temperature=1.0)
            smi = detokenize(out[0].tolist(), {v: k for k, v in tokens.items()}, bos, eos, pad)
            smis.append(smi)
        return smis
    s1 = sample_smis(c1, N_SAMPLES)
    s2 = sample_smis(c2, N_SAMPLES)
    # Compare: fraction of matched-seed pairs where s1[k] != s2[k]
    diff_frac = sum(1 for a, b in zip(s1, s2) if a != b) / max(N_SAMPLES, 1)
    uniq1 = len(set(s1)); uniq2 = len(set(s2))
    overlap = len(set(s1) & set(s2))
    print(f"[stage2] SENSITIVITY: matched-seed diff_frac = {diff_frac:.2f}  uniq(c1)={uniq1}  uniq(c2)={uniq2}  overlap={overlap}", flush=True)
    print(f"[stage2] sample s1[0]={s1[0][:60]}", flush=True)
    print(f"[stage2] sample s2[0]={s2[0][:60]}", flush=True)
    with open(OPT / "stage2_sensitivity.json", "w") as f:
        json.dump({"diff_frac": diff_frac, "uniq_c1": uniq1, "uniq_c2": uniq2, "overlap": overlap, "sample_c1": s1, "sample_c2": s2}, f, indent=2)
    print(f"[stage2] done  (PASS if diff_frac >= 0.5 and overlap < 0.5*min(uniq1,uniq2))", flush=True)


if __name__ == "__main__":
    main()
