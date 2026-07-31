"""Clean control: fine-tune v1_covaFT (mol2mol + covalent FT) on v2_cond's EXACT training pairs
(1,286 src->tgt pairs from pairs_scheme_A.npz), WITHOUT any pocket/pose conditioning.

Question: is v2_cond's planar_dihedral 2.6° due to ARCHITECTURE or DATA?

If this control reaches planar ~2-3°, arch is a confound and the win is in the training data.
If planar stays at ~62° (v1_covaFT baseline), arch is doing real work.

Usage on ai-gpu:
  python experiments/v1_ft_on_v2_data.py --epochs 20 --lr 5e-5 --bs 32
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
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from optionA_mol2mol import (
    Mol2MolTransformer,
    load_mol2mol_prior,
    subsequent_mask,
    tokenize_smiles,
)

PRIOR = REPO / "models" / "reinvent4_mol2mol_covalent_ft.prior"
PAIRS_CSV = REPO / "data" / "optionA" / "v2_pairs_scheme_A_1286.csv"
OUT_DIR = REPO / "data" / "v1_ft_on_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokens: dict, bos: int, eos: int, pad: int, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokens = tokens
        self.bos, self.eos, self.pad = bos, eos, pad
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = np.array(tokenize_smiles(str(row["src"]), self.tokens, self.bos, self.eos, self.pad, self.max_len), dtype=np.int64)
        tgt = np.array(tokenize_smiles(str(row["tgt"]), self.tokens, self.bos, self.eos, self.pad, self.max_len), dtype=np.int64)
        return {
            "src": torch.from_numpy(src),
            "tgt_in": torch.from_numpy(tgt[:-1].copy()),
            "tgt_out": torch.from_numpy(tgt[1:].copy()),
        }


def collate(batch):
    return {
        "src": torch.stack([b["src"] for b in batch]),
        "tgt_in": torch.stack([b["tgt_in"] for b in batch]),
        "tgt_out": torch.stack([b["tgt_out"] for b in batch]),
    }


def make_masks(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(1)
    tgt_mask = (tgt != pad).unsqueeze(1)
    T = tgt.size(1)
    sub = subsequent_mask(T).to(tgt.device)
    tgt_mask = tgt_mask & sub
    return src_mask, tgt_mask


def train_epoch(model, dl, opt, pad, device):
    model.train()
    total, n = 0.0, 0
    for batch in dl:
        src = batch["src"].to(device); tgt_in = batch["tgt_in"].to(device); tgt_out = batch["tgt_out"].to(device)
        src_mask, tgt_mask = make_masks(src, tgt_in, pad)
        logp, _ = model(src, tgt_in, src_mask, tgt_mask)
        loss = F.nll_loss(logp.reshape(-1, logp.size(-1)), tgt_out.reshape(-1), ignore_index=pad)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        B = src.size(0); total += float(loss.item()) * B; n += B
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--out_ckpt", default=str(OUT_DIR / "v1_ft_on_v2_pairs.pt"))
    ap.add_argument("--out_metrics", default=str(OUT_DIR / "train_metrics.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[v1_ft_on_v2] device={device}", flush=True)

    print("[v1_ft_on_v2] loading v1_covaFT prior + vocab...", flush=True)
    model, vocab, prior_max_len, missing, unexpected = load_mol2mol_prior(str(PRIOR), device=device)
    tokens = vocab["tokens"]
    pad = vocab["pad_token"]; bos = vocab["bos_token"]; eos = vocab["eos_token"]
    print(f"[v1_ft_on_v2] loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    model = model.to(device)

    df = pd.read_csv(PAIRS_CSV)
    print(f"[v1_ft_on_v2] pairs: n={len(df)} (unique src={df.src.nunique()}, tgt={df.tgt.nunique()})", flush=True)
    ds = PairsDataset(df, tokens, bos, eos, pad, args.max_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        loss = train_epoch(model, dl, opt, pad, device)
        el = time.time() - t0
        log.append({"ep": ep, "loss": loss, "t": el})
        print(f"[v1_ft_on_v2] ep={ep:02d} loss={loss:.4f}  [{el:.0f}s]", flush=True)

    ckpt = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "vocab": vocab,
        "args": vars(args),
        "log": log,
    }
    torch.save(ckpt, args.out_ckpt)
    open(args.out_metrics, "w").write(json.dumps({"final_loss": log[-1]["loss"], "log": log}, indent=2))
    print(f"[v1_ft_on_v2] saved ckpt to {args.out_ckpt}", flush=True)
    print(f"[v1_ft_on_v2] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
