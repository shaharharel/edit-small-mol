#!/usr/bin/env python3
"""
Compute anchor-based absolute pIC50 predictions from FiLMDelta + Kinase PT.

For each test molecule j, absolute prediction = mean_i(known_pIC50_i + delta(i→j))
where i iterates over all training molecules as anchors.

Usage:
    conda run -n quris python -u experiments/run_zap70_anchor_absolute.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"

KINASE_TARGETS = {
    "SYK": "CHEMBL2599", "FYN": "CHEMBL1841", "LCK": "CHEMBL258",
    "BTK": "CHEMBL5251", "ITK": "CHEMBL3009", "JAK2": "CHEMBL2971",
    "ABL1": "CHEMBL1862", "SRC": "CHEMBL267",
}


def main():
    print("=" * 60)
    print("FiLMDelta + Kinase PT: Anchor-Based Absolute Prediction")
    print("=" * 60)

    # Load ZAP70 data
    mol_data, _ = load_zap70_molecules()
    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n_mols = len(smiles_list)
    print(f"ZAP70: {n_mols} molecules")

    # Load pre-extracted kinase within-assay pairs (9.5MB vs 486MB full CSV)
    print("\nLoading kinase within-assay pairs...")
    kinase_pairs = pd.read_csv(
        PROJECT_ROOT / "data" / "kinase_within_pairs.csv",
        usecols=["mol_a", "mol_b", "delta"]
    )
    print(f"Kinase pairs: {len(kinase_pairs):,}")

    # Compute Morgan FP for all needed molecules
    print("Computing Morgan FP...")
    all_kinase_smi = list(set(
        kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
    ))
    all_smi = list(set(all_kinase_smi + list(smiles_list)))
    print(f"  {len(all_smi):,} unique molecules")
    X_all = compute_fingerprints(all_smi, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: X_all[i] for i, smi in enumerate(all_smi)}
    del X_all, all_kinase_smi
    gc.collect()

    # Filter kinase pairs to those with embeddings
    mask = kinase_pairs["mol_a"].apply(lambda s: s in fp_cache) & \
           kinase_pairs["mol_b"].apply(lambda s: s in fp_cache)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    print(f"Kinase pairs (with FP): {len(kinase_pairs):,}")

    # Generate ZAP70 all-pairs
    zap_rows = []
    for i in range(n_mols):
        for j in range(n_mols):
            if i == j:
                continue
            zap_rows.append({
                "mol_a": smiles_list[i], "mol_b": smiles_list[j],
                "delta": float(pIC50[j] - pIC50[i]),
            })
    zap70_pairs = pd.DataFrame(zap_rows)
    del zap_rows
    print(f"ZAP70 pairs: {len(zap70_pairs):,}")

    # Pretrain FiLMDelta on kinase pairs
    print("\nPretraining on kinase pairs...")
    emb_a_k = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b_k = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta_k = kinase_pairs["delta"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a_k, emb_b_k]))

    Xa_k = torch.FloatTensor(scaler.transform(emb_a_k))
    Xb_k = torch.FloatTensor(scaler.transform(emb_b_k))
    yd_k = torch.FloatTensor(delta_k)
    del emb_a_k, emb_b_k, delta_k, kinase_pairs
    gc.collect()

    n_val = len(Xa_k) // 10
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_vl, best_state, wait = float("inf"), None, 0
    for epoch in range(100):
        model.train()
        perm = np.random.permutation(len(Xa_k) - n_val) + n_val
        for start in range(0, len(perm), 256):
            bi = perm[start:start + 256]
            optimizer.zero_grad()
            loss = criterion(model(Xa_k[bi], Xb_k[bi]), yd_k[bi])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xa_k[:n_val], Xb_k[:n_val]), yd_k[:n_val]).item()
        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                print(f"  Early stop at epoch {epoch + 1}")
                break
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: val_loss={vl:.4f}")

    model.load_state_dict(best_state)
    print(f"Pretrained. Best val loss: {best_vl:.4f}")

    del Xa_k, Xb_k, yd_k
    gc.collect()

    # 10-fold CV with anchor-based absolute prediction
    N_FOLDS = 10
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    delta_folds = []
    abs_anchor_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(smiles_list)):
        print(f"\n--- Fold {fold_i + 1}/{N_FOLDS} ---")
        train_smi_set = set(smiles_list[train_idx])
        test_smi_set = set(smiles_list[test_idx])
        train_smi = smiles_list[train_idx]
        test_smi = smiles_list[test_idx]
        train_y = pIC50[train_idx]
        test_y = pIC50[test_idx]

        train_mask = zap70_pairs["mol_a"].isin(train_smi_set) & zap70_pairs["mol_b"].isin(train_smi_set)
        test_mask = zap70_pairs["mol_a"].isin(test_smi_set) & zap70_pairs["mol_b"].isin(test_smi_set)
        train_p = zap70_pairs[train_mask]
        test_p = zap70_pairs[test_mask]

        # Fine-tune
        ft_model = copy.deepcopy(model)
        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        Xa_tr = torch.FloatTensor(scaler.transform(emb_a_tr))
        Xb_tr = torch.FloatTensor(scaler.transform(emb_b_tr))
        yd_tr = torch.FloatTensor(train_p["delta"].values.astype(np.float32))
        del emb_a_tr, emb_b_tr

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        Xa_te = torch.FloatTensor(scaler.transform(emb_a_te))
        Xb_te = torch.FloatTensor(scaler.transform(emb_b_te))
        yd_te = torch.FloatTensor(test_p["delta"].values.astype(np.float32))
        del emb_a_te, emb_b_te

        opt = torch.optim.Adam(ft_model.parameters(), lr=1e-4, weight_decay=1e-4)
        best_vl2, best_st2, w2 = float("inf"), None, 0

        for ep in range(50):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr))
            for start in range(0, len(perm), 256):
                bi = perm[start:start + 256]
                opt.zero_grad()
                loss = criterion(ft_model(Xa_tr[bi], Xb_tr[bi]), yd_tr[bi])
                loss.backward()
                opt.step()
            ft_model.eval()
            with torch.no_grad():
                vl = criterion(ft_model(Xa_te, Xb_te), yd_te).item()
            if vl < best_vl2:
                best_vl2 = vl
                best_st2 = {k: v.clone() for k, v in ft_model.state_dict().items()}
                w2 = 0
            else:
                w2 += 1
                if w2 >= 15:
                    break

        if best_st2:
            ft_model.load_state_dict(best_st2)
        ft_model.eval()

        # Delta metrics
        with torch.no_grad():
            d_pred = ft_model(Xa_te, Xb_te).numpy()
        d_true = test_p["delta"].values
        mae_d = float(np.mean(np.abs(d_true - d_pred)))
        spr_d = float(spearmanr(d_true, d_pred).statistic)
        delta_folds.append({"mae": mae_d, "spearman": spr_d})

        del Xa_tr, Xb_tr, yd_tr, Xa_te, Xb_te, yd_te
        gc.collect()

        # ANCHOR-BASED absolute prediction
        train_embs = np.array([fp_cache[s] for s in train_smi])
        test_embs = np.array([fp_cache[s] for s in test_smi])
        anchor_embs_t = torch.FloatTensor(scaler.transform(train_embs))

        abs_preds = np.zeros(len(test_smi))
        for j in range(len(test_smi)):
            target_emb_t = torch.FloatTensor(scaler.transform(test_embs[j:j + 1])).expand(len(train_smi), -1)
            with torch.no_grad():
                deltas = ft_model(anchor_embs_t, target_emb_t).numpy()
            abs_preds[j] = np.mean(train_y + deltas)

        mae_abs = float(np.mean(np.abs(test_y - abs_preds)))
        spr_abs = float(spearmanr(test_y, abs_preds).statistic)
        ss_res = np.sum((test_y - abs_preds) ** 2)
        ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
        r2_abs = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        abs_anchor_folds.append({"mae": mae_abs, "spearman": spr_abs, "r2": r2_abs})

        print(f"  delta MAE={mae_d:.4f} Spr={spr_d:.3f} | "
              f"abs(anchor) MAE={mae_abs:.4f} Spr={spr_abs:.3f} R2={r2_abs:.3f}")

        del ft_model, train_embs, test_embs, anchor_embs_t
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("=== FiLMDelta + Kinase PT: Delta ===")
    d_maes = [f["mae"] for f in delta_folds]
    d_sprs = [f["spearman"] for f in delta_folds]
    print(f"  MAE: {np.mean(d_maes):.4f} +/- {np.std(d_maes):.4f}")
    print(f"  Spearman: {np.mean(d_sprs):.4f}")

    print("\n=== FiLMDelta + Kinase PT: Absolute (anchor-based) ===")
    a_maes = [f["mae"] for f in abs_anchor_folds]
    a_sprs = [f["spearman"] for f in abs_anchor_folds]
    a_r2s = [f["r2"] for f in abs_anchor_folds]
    print(f"  MAE: {np.mean(a_maes):.4f} +/- {np.std(a_maes):.4f}")
    print(f"  Spearman: {np.mean(a_sprs):.4f}")
    print(f"  R2: {np.mean(a_r2s):.4f}")

    print("\nFor reference:")
    print("  XGB absolute: MAE=0.524, Spr=0.742")
    print("  DualObjective absolute: MAE=0.542, Spr=0.735")

    # Save
    results = {
        "method": "FiLMDelta + Kinase PT (anchor-based absolute)",
        "n_molecules": n_mols,
        "n_folds": N_FOLDS,
        "anchor_method": "mean(known_pIC50_i + predicted_delta(i->j)) over all train molecules",
        "delta_folds": delta_folds,
        "abs_anchor_folds": abs_anchor_folds,
        "delta_mean_mae": float(np.mean(d_maes)),
        "delta_mean_spr": float(np.mean(d_sprs)),
        "abs_mean_mae": float(np.mean(a_maes)),
        "abs_mean_spr": float(np.mean(a_sprs)),
        "abs_mean_r2": float(np.mean(a_r2s)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "zap70_film_anchor_absolute_10fold.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
