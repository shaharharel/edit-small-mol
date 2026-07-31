"""Arch verification runner. For a single recipe:

  1. Build training pairs CSV (or reuse existing).
  2. Fine-tune from a prior (v1_covaFT or base mol2mol_medium_similarity).
  3. Sample N=500 with Mol1 anchor.
  4. Compute Mol1-anchored planar_dev median (ETKDGv3+MMFF, seed=42, acrylamide dihedral).
  5. Append result row to data/arch_verification/results.csv.

Recipe is described by CLI flags so all recipes share this runner.

Usage:
  python experiments/arch_verification_runner.py \
      --recipe v2_pairs_only \
      --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
      --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
      --epochs 20 --lr 5e-5 --bs 32 \
      --n_samples 500 --temperature 1.0
"""
from __future__ import annotations
import argparse
import csv
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
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from optionA_mol2mol import (
    Mol2MolTransformer,
    load_mol2mol_prior,
    subsequent_mask,
    tokenize_smiles,
    detokenize,
)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

RDLogger.DisableLog("rdApp.*")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
OUT_DIR = REPO / "data" / "arch_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def normalize_vocab(vocab_obj):
    """Convert v1_covaFT-style dict OR REINVENT4 Vocabulary object into
    a standard dict:  {tokens: {str->int}, pad_token, bos_token, eos_token}.
    """
    if isinstance(vocab_obj, dict) and "tokens" in vocab_obj:
        # Already the covaFT-adapted dict form.
        return vocab_obj
    # REINVENT4 Vocabulary object: has _tokens (bidirectional), pad_token, bos_token, eos_token
    if hasattr(vocab_obj, "_tokens"):
        raw = vocab_obj._tokens
        # bidirectional dict: keep only str->int entries
        str_to_int = {k: v for k, v in raw.items() if isinstance(k, str)}
        return {
            "tokens": str_to_int,
            "pad_token": int(getattr(vocab_obj, "pad_token", 0)),
            "bos_token": int(getattr(vocab_obj, "bos_token", 1)),
            "eos_token": int(getattr(vocab_obj, "eos_token", 2)),
        }
    raise ValueError(f"Unknown vocab format: {type(vocab_obj)}")

RESULT_FIELDS = [
    "recipe_name", "prior", "n_train_pairs", "epochs", "lr", "bs",
    "final_loss", "n_valid_samples", "n_acryl_samples",
    "planar_dev_median_deg", "planar_dev_iqr_deg",
    "n_samples", "temperature", "elapsed_s", "notes",
]


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

class PairsDataset(Dataset):
    def __init__(self, df, tokens, bos, eos, pad, max_len):
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
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        src_mask, tgt_mask = make_masks(src, tgt_in, pad)
        logp, _ = model(src, tgt_in, src_mask, tgt_mask)
        loss = F.nll_loss(logp.reshape(-1, logp.size(-1)), tgt_out.reshape(-1), ignore_index=pad)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        B = src.size(0)
        total += float(loss.item()) * B
        n += B
    return total / max(n, 1)


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------

def sample_from_model(model, vocab, anchor, n, max_len, batch_size, device, temperature=1.0, seed=0):
    tokens = vocab["tokens"]
    bos, eos, pad = vocab["bos_token"], vocab["eos_token"], vocab["pad_token"]
    inv = {v: k for k, v in tokens.items()}
    torch.manual_seed(seed)
    ids = tokenize_smiles(anchor, tokens, bos, eos, pad, max_len)
    src_single = torch.tensor([ids], dtype=torch.long, device=device)
    out_smis = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = min(batch_size, n - i)
            src = src_single.repeat(b, 1)
            src_mask = (src != pad).unsqueeze(1)
            memory = model.encode(src, src_mask)
            ys = torch.full((b, 1), bos, dtype=torch.long, device=device)
            done = torch.zeros(b, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                tm = subsequent_mask(ys.size(1)).to(device)
                h = model.decode(memory, src_mask, ys, tm)
                logp = model.generator(h[:, -1]) / max(temperature, 1e-6)
                probs = logp.exp()
                nxt = torch.multinomial(probs, 1)
                nxt = torch.where(done.unsqueeze(-1), torch.full_like(nxt, eos), nxt)
                ys = torch.cat([ys, nxt], dim=1)
                done = done | (nxt.squeeze(-1) == eos)
                if done.all():
                    break
            for r in range(b):
                out_smis.append(detokenize(ys[r].tolist(), inv, bos, eos, pad))
    return out_smis


# ------------------------------------------------------------------
# Planar dev computation (matches covft_geometric_comparison.py)
# ------------------------------------------------------------------

def _planar_dev(d_deg):
    d = abs(d_deg)
    return min(d, abs(180.0 - d))


def compute_planar_dev(smi, seed=42):
    """Return (parsed_ok, acryl_ok, embed_ok, planar_dev_deg or None)."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return (False, False, False, None)
        patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return (True, False, False, None)
        m = matches[0]
        b_idx, a_idx, c_idx, n_idx = int(m[0]), int(m[1]), int(m[2]), int(m[4])
        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                return (True, True, False, None)
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        conf = mol_h.GetConformer()
        d = GetDihedralDeg(conf, b_idx, a_idx, c_idx, n_idx)
        return (True, True, True, _planar_dev(d))
    except Exception:
        return (False, False, False, None)


def analyze_samples(smi_list):
    rows = []
    for smi in smi_list:
        parsed, acryl, embed, dev = compute_planar_dev(smi)
        rows.append({
            "smi": smi,
            "parsed": parsed,
            "acryl": acryl,
            "embed": embed,
            "planar_dev_deg": dev,
        })
    df = pd.DataFrame(rows)
    valid = df[df["parsed"]]
    with_acryl = df[df["acryl"]]
    devs = df.loc[df["planar_dev_deg"].notna(), "planar_dev_deg"].astype(float).values
    if len(devs):
        med = float(np.median(devs))
        q1 = float(np.percentile(devs, 25))
        q3 = float(np.percentile(devs, 75))
        iqr = q3 - q1
    else:
        med = float("nan")
        iqr = float("nan")
    return {
        "df": df,
        "n_valid": int(len(valid)),
        "n_acryl": int(len(with_acryl)),
        "planar_med": med,
        "planar_iqr": iqr,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def append_result(row):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--prior_ckpt", required=True)
    ap.add_argument("--train_csv", required=True, help="CSV with columns src,tgt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample_bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    recipe_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / args.recipe)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    t0_total = time.time()

    prior_abs = str(REPO / args.prior_ckpt) if not os.path.isabs(args.prior_ckpt) else args.prior_ckpt
    train_csv_abs = str(REPO / args.train_csv) if not os.path.isabs(args.train_csv) else args.train_csv

    print(f"[{args.recipe}] device={device}", flush=True)
    print(f"[{args.recipe}] prior={prior_abs}", flush=True)
    print(f"[{args.recipe}] train_csv={train_csv_abs}", flush=True)

    print(f"[{args.recipe}] loading prior + vocab...", flush=True)
    model, vocab_raw, prior_max_len, missing, unexpected = load_mol2mol_prior(prior_abs, device=device)
    print(f"[{args.recipe}] loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    vocab = normalize_vocab(vocab_raw)
    print(f"[{args.recipe}] vocab size={len(vocab['tokens'])} pad={vocab['pad_token']} "
          f"bos={vocab['bos_token']} eos={vocab['eos_token']}", flush=True)
    tokens = vocab["tokens"]
    pad = vocab["pad_token"]; bos = vocab["bos_token"]; eos = vocab["eos_token"]
    model = model.to(device)

    df = pd.read_csv(train_csv_abs)
    df = df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    print(f"[{args.recipe}] pairs: n={len(df)} unique_src={df.src.nunique()} unique_tgt={df.tgt.nunique()}", flush=True)

    ds = PairsDataset(df, tokens, bos, eos, pad, args.max_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        loss = train_epoch(model, dl, opt, pad, device)
        el = time.time() - t0
        log.append({"ep": ep, "loss": loss, "t": el})
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            print(f"[{args.recipe}] ep={ep:03d} loss={loss:.4f}  [{el:.0f}s]", flush=True)

    ckpt_path = recipe_dir / "ckpt.pt"
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "vocab": vocab, "args": vars(args), "log": log}, ckpt_path)
    (recipe_dir / "train_metrics.json").write_text(json.dumps({
        "final_loss": log[-1]["loss"], "log": log}, indent=2))
    print(f"[{args.recipe}] ckpt saved -> {ckpt_path}  train_time={time.time()-t0:.0f}s", flush=True)

    print(f"[{args.recipe}] sampling n={args.n_samples} T={args.temperature}...", flush=True)
    t0s = time.time()
    smis = sample_from_model(model, vocab, MOL1_SMI, args.n_samples, args.max_len,
                             args.sample_bs, device, temperature=args.temperature, seed=args.seed)
    print(f"[{args.recipe}] sampled {len(smis)} in {time.time()-t0s:.0f}s", flush=True)
    pd.DataFrame({"SMILES": smis, "anchor": MOL1_SMI}).to_csv(recipe_dir / "samples.csv", index=False)

    print(f"[{args.recipe}] computing planar dev...", flush=True)
    t0p = time.time()
    analysis = analyze_samples(smis)
    analysis["df"].to_parquet(recipe_dir / "planar_per_sample.parquet")
    print(f"[{args.recipe}] planar: n_valid={analysis['n_valid']} n_acryl={analysis['n_acryl']} "
          f"median={analysis['planar_med']:.2f} iqr={analysis['planar_iqr']:.2f} "
          f"[{time.time()-t0p:.0f}s]", flush=True)

    elapsed_total = time.time() - t0_total
    row = {
        "recipe_name": args.recipe,
        "prior": Path(prior_abs).name,
        "n_train_pairs": int(len(df)),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "bs": int(args.bs),
        "final_loss": float(log[-1]["loss"]),
        "n_valid_samples": int(analysis["n_valid"]),
        "n_acryl_samples": int(analysis["n_acryl"]),
        "planar_dev_median_deg": float(analysis["planar_med"]) if not math.isnan(analysis["planar_med"]) else "",
        "planar_dev_iqr_deg": float(analysis["planar_iqr"]) if not math.isnan(analysis["planar_iqr"]) else "",
        "n_samples": int(args.n_samples),
        "temperature": float(args.temperature),
        "elapsed_s": round(elapsed_total, 1),
        "notes": args.notes,
    }
    append_result(row)
    print(f"[{args.recipe}] DONE elapsed={elapsed_total:.0f}s  "
          f"planar_median={row['planar_dev_median_deg']} deg  "
          f"n_acryl={row['n_acryl_samples']}/{args.n_samples}", flush=True)
    (recipe_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
