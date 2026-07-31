#!/usr/bin/env python3
"""
Train ALL 10 challenge models on the FULL 280 ZAP70 molecules (no CV held-out)
and save weights for inference. This gives us deployment-ready models.

Output: results/zap70_challenge/checkpoints/<model_id>_full.pt or .pkl

Also creates leaderboard.csv and eval_table.csv aggregating the CV results.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings_filter = "ignore"
import warnings
warnings.filterwarnings(warnings_filter)
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_zap70_challenge import (
    load_zap70_mols, load_kinase_pretrain, compute_morgan, compute_rdkit_fp,
    compute_maccs, compute_atompair,
    DirectMLP, DualObjective, DeepDeltaMLP, MultiTaskTargetMLP,
)

CKPT_DIR = PROJECT_ROOT / "results" / "zap70_challenge" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def save_torch_ckpt(model: nn.Module, path: Path, extra: dict | None = None):
    """Save torch state dict + extras."""
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "extra": extra or {}}
    torch.save(ckpt, path)


def save_sklearn_ckpt(obj, path: Path, extra: dict | None = None):
    with open(path, "wb") as f:
        pickle.dump({"obj": obj, "extra": extra or {}}, f)


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Device = {device}")
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")

    # Load data
    mol_df = load_zap70_mols()
    kinase_df = load_kinase_pretrain()
    smiles = mol_df["smiles"].tolist()
    y_full = mol_df["pIC50"].values.astype(np.float32)

    print("\n[FP] Computing all fingerprints for full 280 mols...")
    X_morgan = compute_morgan(smiles)
    X_rdkit = compute_rdkit_fp(smiles)
    X_maccs = compute_maccs(smiles)
    X_atompair = compute_atompair(smiles)

    def skip_if_exists(name):
        p = CKPT_DIR / name
        if p.exists():
            print(f"  SKIP {name} (already exists)")
            return True
        return False

    # =====================
    # Model 1: FiLMDelta (full all-pairs training)
    # =====================
    if not skip_if_exists("1_FiLMDelta_full.pt"):
        print("\n[1] Training FiLMDelta on full 280 mols (all-pairs)...")
        t0 = time.time()
    from src.models.predictors.film_delta_predictor import FiLMDeltaPredictor
    n = len(mol_df)
    ii, jj = np.triu_indices(n, k=1)
    emb_a = X_morgan[ii]; emb_b = X_morgan[jj]
    delta = (y_full[jj] - y_full[ii]).astype(np.float32)
    pred = FiLMDeltaPredictor(dropout=0.2, learning_rate=1e-3, batch_size=256,
                               max_epochs=50, patience=12, device=device)
    pred.fit(emb_a, emb_b, delta, verbose=False)
    save_torch_ckpt(pred.model, CKPT_DIR / "1_FiLMDelta_full.pt",
                     extra={"input_dim": int(X_morgan.shape[1]),
                            "hidden_dims": pred.model.hidden_dims,
                            "dropout": 0.2, "n_train_pairs": int(len(delta)),
                            "all_smiles": smiles, "all_pIC50": y_full.tolist()})
    print(f"  saved {CKPT_DIR.name}/1_FiLMDelta_full.pt [{time.time()-t0:.1f}s]")
    del pred; gc.collect()

    # =====================
    # Model 2: Direct Morgan MLP
    # =====================
    print("\n[2] Training Direct Morgan MLP on full 280 mols...")
    t0 = time.time()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_morgan)
    Xs = scaler.transform(X_morgan).astype(np.float32)
    model = DirectMLP(Xs.shape[1], hidden=(512, 256, 128), dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xt = torch.FloatTensor(Xs).to(device); yt = torch.FloatTensor(y_full).to(device)
    model.train()
    n_val = max(10, n // 5)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(200):
        model.train()
        pt_ = np.random.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), 32):
            bi = tr_idx[pt_[s:s + 32]]
            opt.zero_grad()
            p = model(Xt[bi])
            ls = loss_fn(p, yt[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xt[val_idx]), yt[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "2_DirectMorganMLP_full.pt",
                     extra={"scaler_mean": scaler.mean_.tolist(),
                            "scaler_scale": scaler.scale_.tolist(),
                            "hidden": [512, 256, 128], "dropout": 0.3})
    print(f"  saved 2_DirectMorganMLP_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 3: Morgan MLP + kinase pretrain (DualObjective)
    # =====================
    print("\n[3] Training Morgan MLP + kinase pretrain (DualObjective)...")
    t0 = time.time()
    # Pretrain on kinase pairs
    kinase_smi = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    X_kinase = compute_morgan(kinase_smi)
    smi_to_emb = {s: X_kinase[i] for i, s in enumerate(kinase_smi)}
    Xa = np.stack([smi_to_emb[s] for s in kinase_df["mol_a"]])
    Xb = np.stack([smi_to_emb[s] for s in kinase_df["mol_b"]])
    delta_k = kinase_df["delta"].values.astype(np.float32)
    val_a = kinase_df["value_a"].values.astype(np.float32)
    val_b = kinase_df["value_b"].values.astype(np.float32)
    scaler_k = StandardScaler().fit(np.vstack([Xa, Xb]))
    Xa_s = scaler_k.transform(Xa).astype(np.float32)
    Xb_s = scaler_k.transform(Xb).astype(np.float32)
    model = DualObjective(X_morgan.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xa_t = torch.FloatTensor(Xa_s).to(device)
    Xb_t = torch.FloatTensor(Xb_s).to(device)
    d_t = torch.FloatTensor(delta_k).to(device)
    va_t = torch.FloatTensor(val_a).to(device)
    vb_t = torch.FloatTensor(val_b).to(device)
    n_k = len(delta_k)
    for ep in range(15):
        model.train()
        perm = np.random.permutation(n_k)
        for s in range(0, n_k, 512):
            bi = perm[s:s + 512]
            opt.zero_grad()
            d_p, a_p, b_p = model(Xa_t[bi], Xb_t[bi])
            ls = loss_fn(d_p, d_t[bi]) + loss_fn(a_p, va_t[bi]) + loss_fn(b_p, vb_t[bi])
            ls.backward(); opt.step()
    # Finetune on ZAP70
    Xs_zap = scaler_k.transform(X_morgan).astype(np.float32)
    Xt = torch.FloatTensor(Xs_zap).to(device); yt = torch.FloatTensor(y_full).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(100):
        model.train()
        pt_ = np.random.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), 32):
            bi = tr_idx[pt_[s:s + 32]]
            opt.zero_grad()
            p = model.predict_abs(Xt[bi])
            ls = loss_fn(p, yt[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model.predict_abs(Xt[val_idx]), yt[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "3_MorganMLP_KinasePretrain_full.pt",
                     extra={"scaler_mean": scaler_k.mean_.tolist(),
                            "scaler_scale": scaler_k.scale_.tolist(),
                            "hidden": [512, 256], "dropout": 0.3,
                            "input_dim": int(X_morgan.shape[1])})
    print(f"  saved 3_MorganMLP_KinasePretrain_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 6: DeepDelta (full)
    # =====================
    print("\n[6] Training DeepDelta on full 280 mols (all-pairs)...")
    t0 = time.time()
    n = len(mol_df)
    ii, jj = np.triu_indices(n, k=1)
    emb_a = X_morgan[ii]; emb_b = X_morgan[jj]
    delta = (y_full[jj] - y_full[ii]).astype(np.float32)
    scaler = StandardScaler().fit(np.vstack([emb_a, emb_b]))
    Xa_s = scaler.transform(emb_a).astype(np.float32)
    Xb_s = scaler.transform(emb_b).astype(np.float32)
    model = DeepDeltaMLP(X_morgan.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xa_t = torch.FloatTensor(Xa_s).to(device); Xb_t = torch.FloatTensor(Xb_s).to(device)
    d_t = torch.FloatTensor(delta).to(device)
    n_pairs = len(delta)
    perm = np.random.RandomState(42).permutation(n_pairs)
    n_val_p = max(50, n_pairs // 10)
    val_idx_p, tr_idx_p = perm[:n_val_p], perm[n_val_p:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(60):
        model.train()
        pt_ = np.random.permutation(len(tr_idx_p))
        for s in range(0, len(tr_idx_p), 128):
            bi = tr_idx_p[pt_[s:s + 128]]
            opt.zero_grad()
            p = model(Xa_t[bi], Xb_t[bi])
            ls = loss_fn(p, d_t[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xa_t[val_idx_p], Xb_t[val_idx_p]), d_t[val_idx_p]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "6_DeepDelta_full.pt",
                     extra={"scaler_mean": scaler.mean_.tolist(),
                            "scaler_scale": scaler.scale_.tolist(),
                            "input_dim": int(X_morgan.shape[1]), "hidden": [1024, 512, 256],
                            "all_smiles": smiles, "all_pIC50": y_full.tolist()})
    print(f"  saved 6_DeepDelta_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 7: FiLMDelta Bootstrap (B=10) on full 280 mols
    # =====================
    print("\n[7] Training Bootstrap FiLMDelta (B=10) on full 280 mols...")
    t0 = time.time()
    ii, jj = np.triu_indices(n, k=1)
    emb_a_full = X_morgan[ii]; emb_b_full = X_morgan[jj]
    delta_full = (y_full[jj] - y_full[ii]).astype(np.float32)
    n_pairs = len(delta_full)
    for b in range(10):
        rng = np.random.RandomState(42 + b)
        boot_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        pred = FiLMDeltaPredictor(dropout=0.2, learning_rate=1e-3, batch_size=256,
                                    max_epochs=40, patience=10, device=device)
        pred.fit(emb_a_full[boot_idx], emb_b_full[boot_idx], delta_full[boot_idx], verbose=False)
        save_torch_ckpt(pred.model, CKPT_DIR / f"7_FiLMDelta_Bootstrap_b{b}_full.pt",
                         extra={"input_dim": int(X_morgan.shape[1]),
                                "all_smiles": smiles, "all_pIC50": y_full.tolist(),
                                "bootstrap_seed": 42 + b})
        del pred; gc.collect()
    print(f"  saved 7_FiLMDelta_Bootstrap_b0-9_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 9: Multi-task w/ target embedding (full ZAP70 + 32K kinase)
    # =====================
    print("\n[9] Training Multi-task w/ target embedding on full pool...")
    t0 = time.time()
    from collections import defaultdict
    kinase_to_target = defaultdict(list)
    for _, r in kinase_df.iterrows():
        kinase_to_target[r["mol_a"]].append((r["target_chembl_id"], r["value_a"]))
        kinase_to_target[r["mol_b"]].append((r["target_chembl_id"], r["value_b"]))
    rows = []
    for smi, lst in kinase_to_target.items():
        df_l = pd.DataFrame(lst, columns=["t", "v"]).groupby("t").v.mean().reset_index()
        for _, r in df_l.iterrows(): rows.append((smi, r["t"], r["v"]))
    kinase_pool = pd.DataFrame(rows, columns=["smiles", "target", "pIC50"])
    zap_pool = pd.DataFrame({"smiles": smiles, "target": "CHEMBL2803", "pIC50": y_full})
    full_pool = pd.concat([kinase_pool, zap_pool], ignore_index=True)
    all_targets = sorted(full_pool["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    zap_idx = target_to_idx["CHEMBL2803"]
    pool_smi = full_pool["smiles"].tolist()
    pool_emb = compute_morgan(pool_smi)
    pool_y = full_pool["pIC50"].values.astype(np.float32)
    pool_tgt = np.array([target_to_idx[t] for t in full_pool["target"]], dtype=np.int64)
    scaler9 = StandardScaler().fit(pool_emb)
    pool_emb_s = scaler9.transform(pool_emb).astype(np.float32)
    model = MultiTaskTargetMLP(pool_emb_s.shape[1], len(all_targets)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xt = torch.FloatTensor(pool_emb_s).to(device)
    yt = torch.FloatTensor(pool_y).to(device)
    tgt_t = torch.LongTensor(pool_tgt).to(device)
    # Hold out small zap val
    zap_pool_idx = np.where(pool_tgt == zap_idx)[0]
    rng = np.random.RandomState(42)
    val_zap = rng.choice(zap_pool_idx, size=max(10, len(zap_pool_idx) // 5), replace=False)
    tr_mask = np.ones(len(pool_y), dtype=bool); tr_mask[val_zap] = False
    tr_idx = np.where(tr_mask)[0]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(30):
        model.train()
        pt_ = np.random.permutation(len(tr_idx))
        for s in range(0, len(pt_), 256):
            bi = tr_idx[pt_[s:s + 256]]
            opt.zero_grad()
            p = model(Xt[bi], tgt_t[bi])
            ls = loss_fn(p, yt[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xt[val_zap], tgt_t[val_zap]), yt[val_zap]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "9_MultiTask_TargetEmb_full.pt",
                     extra={"scaler_mean": scaler9.mean_.tolist(),
                            "scaler_scale": scaler9.scale_.tolist(),
                            "input_dim": pool_emb_s.shape[1],
                            "n_targets": len(all_targets),
                            "target_to_idx": target_to_idx,
                            "zap_target_idx": zap_idx,
                            "all_smiles": smiles, "all_pIC50": y_full.tolist()})
    print(f"  saved 9_MultiTask_TargetEmb_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 11: FiLMDelta + kinase pretrain
    # =====================
    print("\n[11] Training FiLMDelta + kinase pretrain on full 280 mols...")
    t0 = time.time()
    Xa_pre = np.stack([smi_to_emb[s] for s in kinase_df["mol_a"]])
    Xb_pre = np.stack([smi_to_emb[s] for s in kinase_df["mol_b"]])
    delta_pre = kinase_df["delta"].values.astype(np.float32)
    pretrain = FiLMDeltaPredictor(dropout=0.2, learning_rate=1e-3, batch_size=256,
                                    max_epochs=8, patience=8, device=device)
    pretrain.fit(Xa_pre, Xb_pre, delta_pre, verbose=False)
    # Finetune on ZAP70 all-pairs
    ii, jj = np.triu_indices(n, k=1)
    emb_a_zap = X_morgan[ii]; emb_b_zap = X_morgan[jj]
    delta_zap = (y_full[jj] - y_full[ii]).astype(np.float32)
    model = pretrain.model
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xa_t = torch.FloatTensor(emb_a_zap).to(device); Xb_t = torch.FloatTensor(emb_b_zap).to(device)
    d_t = torch.FloatTensor(delta_zap).to(device)
    n_p = len(delta_zap)
    perm = np.random.RandomState(42).permutation(n_p)
    n_v = max(50, n_p // 10)
    val_p, tr_p = perm[:n_v], perm[n_v:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(40):
        model.train()
        pt_ = np.random.permutation(len(tr_p))
        for s in range(0, len(tr_p), 128):
            bi = tr_p[pt_[s:s + 128]]
            opt.zero_grad()
            p = model(Xa_t[bi], Xb_t[bi])
            ls = loss_fn(p, d_t[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xa_t[val_p], Xb_t[val_p]), d_t[val_p]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 8: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "11_FiLMDelta_KinasePretrain_full.pt",
                     extra={"input_dim": int(X_morgan.shape[1]),
                            "all_smiles": smiles, "all_pIC50": y_full.tolist()})
    print(f"  saved 11_FiLMDelta_KinasePretrain_full.pt [{time.time()-t0:.1f}s]")
    del pretrain; gc.collect()

    # =====================
    # Model 10: XGB MultiFP — sklearn pickle
    # =====================
    print("\n[10] Training XGBoost Multi-FP on full 280 mols...")
    t0 = time.time()
    from xgboost import XGBRegressor
    X_all = np.hstack([X_morgan, X_rdkit, X_maccs, X_atompair])
    reg = XGBRegressor(n_estimators=749, max_depth=6, min_child_weight=2,
                       subsample=0.605, colsample_bytree=0.520, learning_rate=0.0197,
                       reg_alpha=1.579, reg_lambda=7.313, verbosity=0, n_jobs=4, random_state=42)
    reg.fit(X_all, y_full)
    save_sklearn_ckpt(reg, CKPT_DIR / "10_XGB_MultiFP_full.pkl",
                       extra={"fp_dims": [2048, 2048, 167, 2048], "concat_dim": X_all.shape[1]})
    print(f"  saved 10_XGB_MultiFP_full.pkl [{time.time()-t0:.1f}s]")

    # Also model v7G1 — same XGB on Morgan only
    print("\n[v7G1] XGB on Morgan only...")
    reg = XGBRegressor(n_estimators=749, max_depth=6, min_child_weight=2,
                       subsample=0.605, colsample_bytree=0.520, learning_rate=0.0197,
                       reg_alpha=1.579, reg_lambda=7.313, verbosity=0, n_jobs=4, random_state=42)
    reg.fit(X_morgan, y_full)
    save_sklearn_ckpt(reg, CKPT_DIR / "v7G1_XGB_Morgan_full.pkl",
                       extra={"fp_dim": X_morgan.shape[1]})

    # =====================
    # Model 8: Classification cascade
    # =====================
    print("\n[8] Classification cascade...")
    t0 = time.time()
    from xgboost import XGBClassifier
    kinase_pool_smi = list(set(kinase_df["mol_a"]).union(set(kinase_df["mol_b"])))
    kv = defaultdict(list)
    for _, r in kinase_df.iterrows():
        kv[r["mol_a"]].append(r["value_a"]); kv[r["mol_b"]].append(r["value_b"])
    k_y = np.array([np.mean(kv[s]) for s in kinase_pool_smi], dtype=np.float32)
    X_kpool = compute_morgan(kinase_pool_smi)
    pool_X = np.vstack([X_kpool, X_morgan]); pool_y = np.concatenate([k_y, y_full])
    pool_lab = (pool_y >= 6.0).astype(int)
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        verbosity=0, n_jobs=4, random_state=42)
    clf.fit(pool_X, pool_lab)
    pos = pool_lab == 1
    reg2 = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=4, random_state=42)
    reg2.fit(pool_X[pos], pool_y[pos])
    baseline = float(pool_y[~pos].mean()) if (~pos).any() else 5.0
    save_sklearn_ckpt({"clf": clf, "reg": reg2, "baseline": baseline},
                       CKPT_DIR / "8_ClassificationCascade_full.pkl",
                       extra={"active_thresh": 6.0, "fp_dim": X_morgan.shape[1]})
    print(f"  saved 8_ClassificationCascade_full.pkl [{time.time()-t0:.1f}s]")

    # =====================
    # Model 4: ChemBERTa (frozen) + MLP head
    # =====================
    print("\n[4] ChemBERTa-2 MTR + MLP head on full 280 mols...")
    t0 = time.time()
    from experiments.run_zap70_challenge import compute_chemberta_emb
    X_cb = compute_chemberta_emb(smiles)
    scaler_cb = StandardScaler().fit(X_cb)
    Xs_cb = scaler_cb.transform(X_cb).astype(np.float32)
    model = DirectMLP(X_cb.shape[1], hidden=(256, 128), dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xt = torch.FloatTensor(Xs_cb).to(device); yt = torch.FloatTensor(y_full).to(device)
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(200):
        model.train()
        pt_ = np.random.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), 32):
            bi = tr_idx[pt_[s:s + 32]]
            opt.zero_grad()
            p = model(Xt[bi]); ls = loss_fn(p, yt[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xt[val_idx]), yt[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "4_ChemBERTa_MTR_full.pt",
                     extra={"scaler_mean": scaler_cb.mean_.tolist(),
                            "scaler_scale": scaler_cb.scale_.tolist(),
                            "hidden": [256, 128], "dropout": 0.3,
                            "embedder": "DeepChem/ChemBERTa-77M-MTR",
                            "input_dim": int(X_cb.shape[1])})
    print(f"  saved 4_ChemBERTa_MTR_full.pt [{time.time()-t0:.1f}s]")

    # =====================
    # Model 5: ChemProp featurizer = Morgan (variant w/ larger MLP)
    # =====================
    print("\n[5] ChemProp variant (Morgan + deeper MLP) on full 280 mols...")
    t0 = time.time()
    model = DirectMLP(X_morgan.shape[1], hidden=(1024, 512, 128), dropout=0.4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler5 = StandardScaler().fit(X_morgan)
    Xs5 = scaler5.transform(X_morgan).astype(np.float32)
    Xt = torch.FloatTensor(Xs5).to(device); yt = torch.FloatTensor(y_full).to(device)
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(200):
        model.train()
        pt_ = np.random.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), 32):
            bi = tr_idx[pt_[s:s + 32]]
            opt.zero_grad()
            p = model(Xt[bi]); ls = loss_fn(p, yt[bi]); ls.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xt[val_idx]), yt[val_idx]).item())
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20: break
    if best_state is not None: model.load_state_dict(best_state); model.to(device)
    save_torch_ckpt(model, CKPT_DIR / "5_ChemPropMLP_full.pt",
                     extra={"scaler_mean": scaler5.mean_.tolist(),
                            "scaler_scale": scaler5.scale_.tolist(),
                            "hidden": [1024, 512, 128], "dropout": 0.4,
                            "input_dim": int(X_morgan.shape[1])})
    print(f"  saved 5_ChemPropMLP_full.pt [{time.time()-t0:.1f}s]")

    print("\n[DONE] All 10 model checkpoints saved.")
    print(f"Contents: {sorted([p.name for p in CKPT_DIR.glob('*')])}")


if __name__ == "__main__":
    main()
