#!/usr/bin/env python3
"""Retrain FiLMDelta on A100 GPU (kinase pretrain -> ZAP70 fine-tune).

Matches the CLEAN training protocol used to produce reinvent4_film_model_clean.pt:
  - Pretrain: 100K kinase pairs subsample, LR=2e-4, L1 loss, 30 ep max, patience=5
  - Fine-tune: ZAP70 all-pairs (mol-disjoint train/val), LR=1e-4, MSE, grad-clip=1.0,
               batch=256, 60 ep max, patience=10
Held-out mol-disjoint val split reuses the same seed/policy as the clean model.
"""
from __future__ import annotations
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
ZAP70_FILE = PROJECT_ROOT / "data" / "zap70_minimal.csv"
KINASE_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_retrain_2026_07_02.pt"

ZAP70_ID = "CHEMBL2803"
SEED = 42
N_KINASE_PAIRS = 100_000
BATCH_SIZE = 256
PRETRAIN_LR = 2e-4
PRETRAIN_EPOCHS = 30
PRETRAIN_PATIENCE = 5
FINETUNE_LR = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10
HOLDOUT_FRAC = 0.10
CLIP = 1.0


def load_zap70() -> pd.DataFrame:
    raw = pd.read_csv(ZAP70_FILE)
    if "target_chembl_id" in raw.columns:
        raw = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol = raw.groupby("molecule_chembl_id").agg({"smiles": "first", "pIC50": "mean"}).reset_index()
    print(f"  ZAP70: {len(mol)} unique molecules, pIC50 {mol['pIC50'].min():.2f}-{mol['pIC50'].max():.2f}",
          flush=True)
    return mol


def compute_fps(smiles_list):
    fps = np.zeros((len(smiles_list), 2048), dtype=np.float32)
    keep = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
        keep.append(i)
    return fps, keep


def kinase_pretrain(fp_cache, scaler, device):
    print(f"[retrain] Loading kinase pairs from {KINASE_FILE}", flush=True)
    kp = pd.read_csv(KINASE_FILE, usecols=["mol_a", "mol_b", "delta"])
    mask = kp["mol_a"].apply(lambda s: s in fp_cache) & kp["mol_b"].apply(lambda s: s in fp_cache)
    kp = kp[mask].reset_index(drop=True)
    if len(kp) > N_KINASE_PAIRS:
        kp = kp.sample(n=N_KINASE_PAIRS, random_state=SEED).reset_index(drop=True)
    print(f"[retrain] Kinase pairs (subsampled): {len(kp):,}", flush=True)

    ea = np.array([fp_cache[s] for s in kp["mol_a"]])
    eb = np.array([fp_cache[s] for s in kp["mol_b"]])
    d  = kp["delta"].values.astype(np.float32)
    Xa = torch.FloatTensor(scaler.transform(ea)).to(device)
    Xb = torch.FloatTensor(scaler.transform(eb)).to(device)
    yd = torch.FloatTensor(d).to(device)
    del ea, eb, d, kp; gc.collect()

    n_val = len(Xa) // 10
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    crit = nn.L1Loss()
    best, best_st, w = float('inf'), None, 0
    t0 = time.time()
    print(f"[retrain] ===== PRETRAIN (100K kinase, L1, LR={PRETRAIN_LR}) =====", flush=True)
    for ep in range(PRETRAIN_EPOCHS):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        losses = []
        for s in range(0, len(perm), BATCH_SIZE):
            bi = perm[s:s+BATCH_SIZE]
            bi_t = torch.LongTensor(bi).to(device)
            opt.zero_grad()
            loss = crit(model(Xa[bi_t], Xb[bi_t]), yd[bi_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vp = model(Xa[:n_val], Xb[:n_val])
            vl = crit(vp, yd[:n_val]).item()
        tl = float(np.mean(losses))
        print(f"[retrain] pretrain ep {ep+1:3d} train_mae={tl:.4f} val_mae={vl:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        if vl < best:
            best, best_st, w = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= PRETRAIN_PATIENCE:
                print(f"[retrain] pretrain early stop ep {ep+1}, best val_mae={best:.4f}", flush=True)
                break
    print(f"[retrain] Kinase pretrain final val_mae={best:.4f}", flush=True)
    del Xa, Xb, yd; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return best_st


def train(device: torch.device, out_path: Path):
    np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == 'cuda': torch.cuda.manual_seed_all(SEED)

    zap = load_zap70()
    smiles_list = zap["smiles"].tolist()
    pIC50_arr = zap["pIC50"].values.astype(np.float64)
    n = len(smiles_list)

    fp_cache: dict[str, np.ndarray] = {}
    fps, keep = compute_fps(smiles_list)
    for i in keep: fp_cache[smiles_list[i]] = fps[i]

    kp_all_smi = list(set(pd.read_csv(KINASE_FILE, usecols=["mol_a", "mol_b"]).stack().unique().tolist()))
    extra = [s for s in kp_all_smi if s not in fp_cache]
    if extra:
        efps, ekeep = compute_fps(extra)
        for i in ekeep: fp_cache[extra[i]] = efps[i]
    print(f"[retrain] Total fingerprint cache: {len(fp_cache):,} molecules", flush=True)

    # Fit scaler on ZAP70 anchors (matches clean protocol)
    zap_fps = np.array([fp_cache[s] for s in smiles_list])
    scaler = StandardScaler()
    scaler.fit(zap_fps)

    # ---- Kinase pretrain ----
    pretrain_state = kinase_pretrain(fp_cache, scaler, device)

    # ---- Mol-disjoint split ----
    np.random.seed(SEED)
    perm_mols = np.random.permutation(n)
    n_val_mols = max(1, int(round(n * HOLDOUT_FRAC)))
    held = set(perm_mols[:n_val_mols].tolist())
    train_mol_ids = [i for i in range(n) if i not in held]
    val_mol_ids   = sorted(held)
    print(f"[retrain] Mol-disjoint split: train_mols={len(train_mol_ids)} val_mols={len(val_mol_ids)}",
          flush=True)

    # Build mol-disjoint pair lists
    pa_tr, pb_tr, d_tr = [], [], []
    pa_va, pb_va, d_va = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            a = fp_cache[smiles_list[i]]
            b = fp_cache[smiles_list[j]]
            delta = float(pIC50_arr[j] - pIC50_arr[i])
            if (i in held) or (j in held):
                pa_va.append(a); pb_va.append(b); d_va.append(delta)
            else:
                pa_tr.append(a); pb_tr.append(b); d_tr.append(delta)
    print(f"[retrain] Train pairs: {len(pa_tr):,}  Val pairs: {len(pa_va):,}", flush=True)

    Xa_tr = torch.FloatTensor(scaler.transform(np.array(pa_tr))).to(device)
    Xb_tr = torch.FloatTensor(scaler.transform(np.array(pb_tr))).to(device)
    yd_tr = torch.FloatTensor(np.array(d_tr, dtype=np.float32)).to(device)
    Xa_va = torch.FloatTensor(scaler.transform(np.array(pa_va))).to(device)
    Xb_va = torch.FloatTensor(scaler.transform(np.array(pb_va))).to(device)
    yd_va = torch.FloatTensor(np.array(d_va, dtype=np.float32)).to(device)
    del pa_tr, pb_tr, d_tr, pa_va, pb_va, d_va; gc.collect()

    # ---- Fine-tune ----
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    model.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()})
    opt2 = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit = nn.MSELoss()
    best, best_st, pat = float('inf'), None, 0
    t1 = time.time()
    print(f"[retrain] ===== FINE-TUNE (mol-disjoint all-pairs, MSE, LR={FINETUNE_LR}, clip={CLIP}) =====",
          flush=True)
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = np.random.permutation(len(Xa_tr))
        losses = []
        for s in range(0, len(perm), BATCH_SIZE):
            bi = perm[s:s+BATCH_SIZE]
            bi_t = torch.LongTensor(bi).to(device)
            opt2.zero_grad()
            loss = crit(model(Xa_tr[bi_t], Xb_tr[bi_t]), yd_tr[bi_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt2.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vp = model(Xa_va, Xb_va)
            val_mse = crit(vp, yd_va).item()
            val_mae = float(torch.mean(torch.abs(vp - yd_va)).item())
        tl = float(np.mean(losses))
        print(f"[retrain] finetune ep {ep+1:3d} train_mse={tl:.4f} val_mse={val_mse:.4f} "
              f"val_mae_delta={val_mae:.4f} elapsed={time.time()-t1:.0f}s", flush=True)
        if val_mse < best:
            best, best_st, pat = val_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"[retrain] finetune early stop ep {ep+1}, best val_mse={best:.4f}", flush=True)
                break
    if best_st is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    model = model.to('cpu').eval()

    # ---- Anchor-based mol-disjoint absolute pIC50 val ----
    print(f"[retrain] ===== MOL-DISJOINT ABSOLUTE pIC50 VAL =====", flush=True)
    anchor_train_fps = np.array([fp_cache[smiles_list[i]] for i in train_mol_ids])
    anchor_train_embs = torch.FloatTensor(scaler.transform(anchor_train_fps))
    train_pIC50 = np.array([pIC50_arr[i] for i in train_mol_ids])
    val_fps = np.array([fp_cache[smiles_list[i]] for i in val_mol_ids])
    val_embs = torch.FloatTensor(scaler.transform(val_fps))
    val_pIC50 = np.array([pIC50_arr[i] for i in val_mol_ids])

    n_anchors_tr = len(train_mol_ids)
    preds = np.zeros(len(val_mol_ids), dtype=np.float64)
    with torch.no_grad():
        for k in range(len(val_mol_ids)):
            tgt = val_embs[k:k+1].expand(n_anchors_tr, -1)
            deltas = model(anchor_train_embs, tgt).numpy().flatten()
            preds[k] = float(np.mean(train_pIC50 + deltas))
    val_mae_mol_disjoint = float(np.mean(np.abs(preds - val_pIC50)))
    val_spr = float(pd.Series(preds).corr(pd.Series(val_pIC50), method='spearman'))
    val_pearson = float(np.corrcoef(preds, val_pIC50)[0, 1])
    print(f"[retrain] val_mae_mol_disjoint={val_mae_mol_disjoint:.4f} "
          f"val_spr={val_spr:.4f} val_pearson={val_pearson:.4f} n_val={len(val_mol_ids)}",
          flush=True)

    # Final anchor set: ALL ZAP70 mols (matches inference behavior of clean model)
    anchor_fps_all = np.array([fp_cache[s] for s in smiles_list])
    anchor_embs_all = torch.FloatTensor(scaler.transform(anchor_fps_all))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "anchor_embs": anchor_embs_all,
        "anchor_pIC50": pIC50_arr.copy(),
        "val_mae_mol_disjoint": val_mae_mol_disjoint,
        "val_spr_mol_disjoint": val_spr,
        "val_pearson_mol_disjoint": val_pearson,
        "heldout_idx": val_mol_ids,
        "seed": SEED,
        "training_ts": "2026-07-02",
        "protocol": "kinase_pretrain_L1_2e-4 -> zap70_allpairs_MSE_1e-4_clip1.0",
    }, out_path)
    print(f"[retrain] Saved to {out_path}", flush=True)
    return val_mae_mol_disjoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    print(f"[retrain] device={device}", flush=True)
    val_mae = train(device, args.out)
    print(f"[retrain] Final val_mae_mol_disjoint={val_mae:.4f}", flush=True)


if __name__ == "__main__":
    main()
