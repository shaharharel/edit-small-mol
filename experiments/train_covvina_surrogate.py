"""Train a Morgan FP -> cov-Vina affinity surrogate (3-layer MLP).

Pools all covft_geometric_covvina_*.csv from prior cohorts (~2K scored mols).
Targets transformed by y_log = log1p(vina_affinity) for stable regression
(raw cov-Vina is a positive repulsion-style score; lower = better binding).

Outputs:
  models/covvina_surrogate.pt       — MLP state dict + scaler stats + meta
  models/covvina_surrogate_metrics.json — train/val MAE, RMSE, Spearman
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RESULT_GLOB = "results/paper_evaluation/covft_geometric_covvina_*.csv"


def smi_to_fp(smi: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden=(512, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / RESULT_GLOB))
    ap.add_argument("--out_model", default=str(ROOT / "models/covvina_surrogate.pt"))
    ap.add_argument("--out_metrics", default=str(ROOT / "models/covvina_surrogate_metrics.json"))
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load + pool
    import glob as _glob
    files = sorted(_glob.glob(args.glob))
    print(f"Found {len(files)} cov-vina CSVs")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "vina_ok" in df.columns:
            df = df[df.vina_ok.astype(bool) & df.vina_affinity.notna()]
        rows.append(df[["smi", "vina_affinity", "cohort"]])
    pool = pd.concat(rows, ignore_index=True)
    # Dedup: keep first by SMILES
    pool = pool.drop_duplicates(subset=["smi"]).reset_index(drop=True)
    print(f"Pooled scored mols (dedup): {len(pool)}")

    # Featurize
    fps = []
    keep_idx = []
    for i, smi in enumerate(pool["smi"].tolist()):
        fp = smi_to_fp(smi, args.n_bits)
        if fp is None:
            continue
        fps.append(fp)
        keep_idx.append(i)
    X = np.array(fps, dtype=np.float32)
    y_raw = pool.iloc[keep_idx]["vina_affinity"].values.astype(np.float32)
    y = np.log1p(np.clip(y_raw, 0.0, None)).astype(np.float32)
    print(f"Featurized: X.shape={X.shape}, y (log1p) median={np.median(y):.3f} std={y.std():.3f}")

    Xtr, Xva, ytr, yva, ytr_raw, yva_raw = train_test_split(
        X, y, y_raw, test_size=args.val_frac, random_state=args.seed
    )
    print(f"Train n={len(Xtr)}  Val n={len(Xva)}")

    # Standardize y for stable training; un-standardize for reporting
    y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-9)
    ytr_z = (ytr - y_mean) / y_std
    yva_z = (yva - y_mean) / y_std

    Xtr_t = torch.from_numpy(Xtr)
    Xva_t = torch.from_numpy(Xva)
    ytr_t = torch.from_numpy(ytr_z)
    yva_t = torch.from_numpy(yva_z)

    model = MLP(in_dim=args.n_bits)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.MSELoss()

    best_val = float("inf"); best_state = None; wait = 0
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        losses = []
        for s in range(0, len(perm), args.batch_size):
            bi = perm[s:s + args.batch_size]
            opt.zero_grad()
            p = model(Xtr_t[bi])
            loss = crit(p, ytr_t[bi])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pva = model(Xva_t)
            val_loss = float(crit(pva, yva_t).item())
            pva_orig = pva.numpy() * y_std + y_mean
            yva_orig = yva
            val_mae = float(np.mean(np.abs(pva_orig - yva_orig)))
            val_rmse = float(np.sqrt(np.mean((pva_orig - yva_orig) ** 2)))
            val_sp = float(spearmanr(pva_orig, yva_orig).statistic) if len(set(yva_orig)) > 1 else 0.0
        sched.step(val_loss)
        history.append(dict(epoch=ep, train_loss=float(np.mean(losses)),
                            val_loss=val_loss, val_mae_log1p=val_mae,
                            val_rmse_log1p=val_rmse, val_spearman=val_sp))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at ep {ep+1}")
                break
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}  trL={np.mean(losses):.4f}  vL={val_loss:.4f}  vMAE(log1p)={val_mae:.3f}  vSp={val_sp:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pva_orig = (model(Xva_t).numpy() * y_std + y_mean)
        yva_orig = yva
        final_mae = float(np.mean(np.abs(pva_orig - yva_orig)))
        final_rmse = float(np.sqrt(np.mean((pva_orig - yva_orig) ** 2)))
        final_sp = float(spearmanr(pva_orig, yva_orig).statistic) if len(set(yva_orig)) > 1 else 0.0
        # Compare in raw vina_affinity space too
        pva_raw = np.expm1(pva_orig)
        raw_mae = float(np.mean(np.abs(pva_raw - yva_raw)))
        raw_sp = float(spearmanr(pva_raw, yva_raw).statistic) if len(set(yva_raw)) > 1 else 0.0

    metrics = dict(
        n_train=len(Xtr_t), n_val=len(Xva_t), n_total=len(X),
        y_mean=y_mean, y_std=y_std,
        final_val_mae_log1p=final_mae,
        final_val_rmse_log1p=final_rmse,
        final_val_spearman=final_sp,
        final_val_mae_raw=raw_mae,
        final_val_spearman_raw=raw_sp,
        train_seconds=round(time.time() - t0, 1),
        history=history,
    )
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out_metrics}")
    print(f"Final val: MAE(log1p)={final_mae:.3f}  Sp={final_sp:.3f}  MAE(raw)={raw_mae:.2f}  Sp(raw)={raw_sp:.3f}")

    torch.save(dict(
        state_dict=model.state_dict(),
        y_mean=y_mean,
        y_std=y_std,
        n_bits=args.n_bits,
        hidden=(512, 256, 128),
        dropout=0.2,
    ), args.out_model)
    print(f"Wrote {args.out_model}")


if __name__ == "__main__":
    main()
