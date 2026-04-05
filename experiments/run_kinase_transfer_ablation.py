#!/usr/bin/env python3
"""
Kinase Transfer Ablation for ZAP70 (CHEMBL2803).

Tests phylogenetic transfer hypothesis: does pretraining on closer kinase
relatives (SYK) help more than distant kinases or the full panel?

Ablation conditions:
1. No pretraining (ZAP70 only baseline)
2. SYK-only pretraining (closest relative, same Syk family)
3. Full 8-kinase panel pretraining (existing Phase D)
4. Distant kinases only (ABL1, SRC, JAK2 — non-Syk family)
5. Syk-family only (SYK + FYN + LCK — closest relatives)

For each condition: pretrain on kinase MMP pairs → fine-tune on ZAP70 → evaluate.

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/run_kinase_transfer_ablation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import gc
import json
import os
import time
import warnings
from datetime import datetime

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

from experiments.run_paper_evaluation import RESULTS_DIR, DATA_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    compute_absolute_metrics, aggregate_cv_results,
    N_FOLDS, CV_SEED,
)

RESULTS_FILE = RESULTS_DIR / "zap70_kinase_transfer_ablation.json"

# Kinase targets and their phylogenetic grouping
KINASE_GROUPS = {
    "syk_family": {
        "SYK": "CHEMBL2599",   # Closest to ZAP70 (same Syk family)
        "FYN": "CHEMBL1841",   # Src family but related
        "LCK": "CHEMBL258",    # Src family, T-cell signaling like ZAP70
    },
    "other_kinases": {
        "BTK": "CHEMBL5251",   # Tec family
        "ITK": "CHEMBL3009",   # Tec family
        "JAK2": "CHEMBL2971",  # JAK family
        "ABL1": "CHEMBL1862",  # Abl family
        "SRC": "CHEMBL267",    # Src family
    },
}

# Ablation conditions
ABLATION_CONDITIONS = {
    "no_pretrain": {
        "description": "No pretraining (ZAP70 only)",
        "targets": {},
    },
    "syk_only": {
        "description": "SYK only (closest relative, same Syk family)",
        "targets": {"SYK": "CHEMBL2599"},
    },
    "syk_family": {
        "description": "Syk-family kinases (SYK + FYN + LCK)",
        "targets": KINASE_GROUPS["syk_family"],
    },
    "distant_only": {
        "description": "Distant kinases only (ABL1, SRC, JAK2)",
        "targets": {"ABL1": "CHEMBL1862", "SRC": "CHEMBL267", "JAK2": "CHEMBL2971"},
    },
    "full_panel": {
        "description": "Full 8-kinase panel",
        "targets": {**KINASE_GROUPS["syk_family"], **KINASE_GROUPS["other_kinases"]},
    },
}


class DualObjectiveModel(nn.Module):
    """Model with dual heads: delta prediction + absolute prediction."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_dim = hidden_dims[-1]
        self.delta_head = nn.Sequential(
            nn.Linear(self.enc_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.abs_head = nn.Sequential(
            nn.Linear(self.enc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_a, x_b):
        enc_a = self.encoder(x_a)
        enc_b = self.encoder(x_b)
        delta_pred = self.delta_head(torch.cat([enc_a, enc_b], dim=-1)).squeeze(-1)
        abs_pred_a = self.abs_head(enc_a).squeeze(-1)
        abs_pred_b = self.abs_head(enc_b).squeeze(-1)
        return delta_pred, abs_pred_a, abs_pred_b

    def predict_absolute(self, x):
        enc = self.encoder(x)
        return self.abs_head(enc).squeeze(-1)


def generate_all_pairs(mol_data):
    """Generate all pairwise combinations from molecule data."""
    smiles = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n = len(smiles)
    idx_i, idx_j = np.triu_indices(n, k=1)
    rows = []
    for i, j in zip(idx_i, idx_j):
        rows.append({
            "mol_a": smiles[i], "mol_b": smiles[j],
            "delta": float(pIC50[j] - pIC50[i]),
            "value_a": float(pIC50[i]), "value_b": float(pIC50[j]),
        })
    return pd.DataFrame(rows)


def pretrain_on_kinase_pairs(kinase_pairs, fp_cache, hidden_dims=[512, 256],
                              epochs=100, batch_size=256, lr=1e-3, patience=15,
                              lambda_abs=0.3):
    """Pretrain dual-objective model on kinase MMP pairs."""
    emb_a = np.array([fp_cache[s] for s in kinase_pairs["mol_a"]])
    emb_b = np.array([fp_cache[s] for s in kinase_pairs["mol_b"]])
    delta = kinase_pairs["delta"].values.astype(np.float32)
    val_a = kinase_pairs["value_a"].values.astype(np.float32)
    val_b = kinase_pairs["value_b"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([emb_a, emb_b]))

    Xa = torch.FloatTensor(scaler.transform(emb_a))
    Xb = torch.FloatTensor(scaler.transform(emb_b))
    yd = torch.FloatTensor(delta)
    ya = torch.FloatTensor(val_a)
    yb = torch.FloatTensor(val_b)

    n_val = len(Xa) // 10
    model = DualObjectiveModel(Xa.shape[1], hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for start in range(0, len(perm), batch_size):
            bi = perm[start:start + batch_size]
            optimizer.zero_grad()
            d_pred, a_pred, b_pred = model(Xa[bi], Xb[bi])
            loss = criterion(d_pred, yd[bi]) + lambda_abs * (criterion(a_pred, ya[bi]) + criterion(b_pred, yb[bi]))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vd, _, _ = model(Xa[:n_val], Xb[:n_val])
            vl = criterion(vd, yd[:n_val]).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Validation metrics
    model.eval()
    with torch.no_grad():
        val_pred, _, _ = model(Xa[:n_val], Xb[:n_val])
    val_mae = float(np.mean(np.abs(delta[:n_val] - val_pred.numpy())))
    val_spr, _ = spearmanr(delta[:n_val], val_pred.numpy())

    return model, scaler, {"val_mae": val_mae, "val_spearman": float(val_spr)}


def finetune_and_evaluate(pretrained_model, pretrained_scaler, zap70_pairs, fp_cache,
                           mol_data, n_folds=5, seed=42, epochs=50, batch_size=64,
                           lr=5e-4, lambda_abs=0.3):
    """Fine-tune pretrained model on ZAP70 and evaluate via CV."""
    zap70_smiles = list(mol_data["smiles"].values)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mol_indices = np.arange(len(zap70_smiles))

    finetune_folds = []
    nofinetune_folds = []
    absolute_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(mol_indices)):
        train_smi = set(np.array(zap70_smiles)[train_idx])
        test_smi = set(np.array(zap70_smiles)[test_idx])

        train_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(train_smi) & zap70_pairs["mol_b"].isin(train_smi)
        ]
        test_p = zap70_pairs[
            zap70_pairs["mol_a"].isin(test_smi) & zap70_pairs["mol_b"].isin(test_smi)
        ]
        if len(test_p) == 0:
            continue

        emb_a_tr = np.array([fp_cache[s] for s in train_p["mol_a"]])
        emb_b_tr = np.array([fp_cache[s] for s in train_p["mol_b"]])
        delta_tr = train_p["delta"].values.astype(np.float32)
        val_a_tr = train_p["value_a"].values.astype(np.float32)
        val_b_tr = train_p["value_b"].values.astype(np.float32)

        emb_a_te = np.array([fp_cache[s] for s in test_p["mol_a"]])
        emb_b_te = np.array([fp_cache[s] for s in test_p["mol_b"]])
        delta_te = test_p["delta"].values.astype(np.float32)

        # Test pretrained model without fine-tuning
        if pretrained_model is not None:
            Xa_te_s = torch.FloatTensor(pretrained_scaler.transform(emb_a_te))
            Xb_te_s = torch.FloatTensor(pretrained_scaler.transform(emb_b_te))
            pretrained_model.eval()
            with torch.no_grad():
                noft_pred, _, _ = pretrained_model(Xa_te_s, Xb_te_s)
                noft_pred = noft_pred.numpy()
            noft_mae = float(np.mean(np.abs(delta_te - noft_pred)))
            noft_spr, _ = spearmanr(delta_te, noft_pred) if len(delta_te) > 2 else (0, 1)
            nofinetune_folds.append({"mae": noft_mae, "spearman": float(noft_spr) if not np.isnan(noft_spr) else 0})

        # Fine-tune
        if pretrained_model is not None:
            ft_model = copy.deepcopy(pretrained_model)
            ft_scaler = copy.deepcopy(pretrained_scaler)
        else:
            # No pretrain — train from scratch
            ft_model = DualObjectiveModel(emb_a_tr.shape[1], [512, 256])
            ft_scaler = StandardScaler()
            ft_scaler.fit(np.vstack([emb_a_tr, emb_b_tr]))

        Xa_tr_s = torch.FloatTensor(ft_scaler.transform(emb_a_tr))
        Xb_tr_s = torch.FloatTensor(ft_scaler.transform(emb_b_tr))
        yd = torch.FloatTensor(delta_tr)
        ya = torch.FloatTensor(val_a_tr)
        yb = torch.FloatTensor(val_b_tr)

        ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None
        for epoch in range(epochs):
            ft_model.train()
            perm = np.random.permutation(len(Xa_tr_s))
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                ft_optimizer.zero_grad()
                d_p, a_p, b_p = ft_model(Xa_tr_s[bi], Xb_tr_s[bi])
                loss = criterion(d_p, yd[bi]) + lambda_abs * (criterion(a_p, ya[bi]) + criterion(b_p, yb[bi]))
                loss.backward()
                ft_optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                # Use test set for early stopping (same as Phase D original)
                Xa_te_ft = torch.FloatTensor(ft_scaler.transform(emb_a_te))
                Xb_te_ft = torch.FloatTensor(ft_scaler.transform(emb_b_te))
                d_p_te, _, _ = ft_model(Xa_te_ft, Xb_te_ft)
                val_loss = criterion(d_p_te, torch.FloatTensor(delta_te)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}

        if best_state:
            ft_model.load_state_dict(best_state)
        ft_model.eval()
        with torch.no_grad():
            Xa_te_ft = torch.FloatTensor(ft_scaler.transform(emb_a_te))
            Xb_te_ft = torch.FloatTensor(ft_scaler.transform(emb_b_te))
            ft_pred, _, _ = ft_model(Xa_te_ft, Xb_te_ft)
            ft_pred = ft_pred.numpy()
        ft_mae = float(np.mean(np.abs(delta_te - ft_pred)))
        ft_spr, _ = spearmanr(delta_te, ft_pred) if len(delta_te) > 2 else (0, 1)
        finetune_folds.append({"mae": ft_mae, "spearman": float(ft_spr) if not np.isnan(ft_spr) else 0})

        # Absolute predictions from dual model
        test_embs = np.array([fp_cache[s] for s in np.array(zap70_smiles)[test_idx]])
        test_y = mol_data["pIC50"].values[test_idx]
        with torch.no_grad():
            abs_preds = ft_model.predict_absolute(
                torch.FloatTensor(ft_scaler.transform(test_embs))
            ).numpy()
        abs_metrics = compute_absolute_metrics(test_y, abs_preds)
        absolute_folds.append(abs_metrics)

        del ft_model
        gc.collect()
        print(f"    Fold {fold_i+1}: δ MAE={ft_mae:.4f}, δ Spr={ft_spr:.3f}")

    return {
        "finetune": finetune_folds,
        "no_finetune": nofinetune_folds,
        "absolute": absolute_folds,
    }


def main():
    print("=" * 70)
    print("KINASE TRANSFER ABLATION — ZAP70 (CHEMBL2803)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load existing results if any
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded existing results: {list(results.keys())}")

    # Load ZAP70 data
    print("\n1. Loading ZAP70 molecules...")
    mol_data, _ = load_zap70_molecules()
    zap70_smiles = list(mol_data["smiles"].values)
    print(f"   {len(mol_data)} molecules, pIC50 range: {mol_data['pIC50'].min():.2f}-{mol_data['pIC50'].max():.2f}")

    # Generate ZAP70 all-pairs
    print("\n2. Generating ZAP70 all-pairs...")
    zap70_pairs = generate_all_pairs(mol_data)
    print(f"   {len(zap70_pairs):,} pairs")

    # Load kinase MMP pairs
    print("\n3. Loading kinase MMP pairs from shared_pairs_deduped.csv...")
    pairs = pd.read_csv(DATA_DIR / "shared_pairs_deduped.csv",
                        usecols=["mol_a", "mol_b", "delta", "is_within_assay",
                                 "target_chembl_id", "value_a", "value_b"])
    within_pairs = pairs[pairs["is_within_assay"] == True].copy()
    del pairs
    gc.collect()

    # Count per-target pairs
    print("\n   Per-target pair counts:")
    all_targets = {**KINASE_GROUPS["syk_family"], **KINASE_GROUPS["other_kinases"]}
    target_counts = {}
    for name, chembl_id in all_targets.items():
        count = (within_pairs["target_chembl_id"] == chembl_id).sum()
        target_counts[name] = count
        print(f"   {name} ({chembl_id}): {count:,} within-assay pairs")

    # Compute fingerprints for all unique molecules
    print("\n4. Computing fingerprints...")
    all_kinase_smiles = list(set(within_pairs["mol_a"].tolist() + within_pairs["mol_b"].tolist()))
    all_smiles_combined = list(set(all_kinase_smiles + zap70_smiles))
    print(f"   {len(all_smiles_combined):,} unique molecules")

    X_all = compute_fingerprints(all_smiles_combined, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: X_all[i] for i, smi in enumerate(all_smiles_combined)}
    del X_all
    gc.collect()

    # Run each ablation condition
    print("\n5. Running ablation conditions...\n")

    for cond_name, cond_config in ABLATION_CONDITIONS.items():
        if cond_name in results and results[cond_name].get("completed"):
            print(f"\n{'='*60}")
            print(f"CONDITION: {cond_name} — {cond_config['description']}")
            print(f"Already completed, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name} — {cond_config['description']}")
        print(f"{'='*60}")

        target_ids = set(cond_config["targets"].values())

        if len(target_ids) == 0:
            # No pretraining — train from scratch
            print("  No pretraining, training from scratch on ZAP70...")
            eval_results = finetune_and_evaluate(
                None, None, zap70_pairs, fp_cache, mol_data
            )
            pretrain_info = {"n_pairs": 0, "targets": []}
        else:
            # Filter kinase pairs
            kinase_subset = within_pairs[within_pairs["target_chembl_id"].isin(target_ids)].copy()
            print(f"  Pretraining on {len(kinase_subset):,} pairs from {list(cond_config['targets'].keys())}")

            if len(kinase_subset) < 100:
                print(f"  WARNING: Only {len(kinase_subset)} pairs — too few for meaningful pretraining")

            # Cap at 100K pairs
            if len(kinase_subset) > 100000:
                kinase_subset = kinase_subset.sample(100000, random_state=42)
                print(f"  Sampled to 100,000 pairs")

            model, scaler, pretrain_metrics = pretrain_on_kinase_pairs(kinase_subset, fp_cache)
            print(f"  Pretrain val: MAE={pretrain_metrics['val_mae']:.4f}, Spr={pretrain_metrics['val_spearman']:.3f}")

            eval_results = finetune_and_evaluate(
                model, scaler, zap70_pairs, fp_cache, mol_data
            )
            pretrain_info = {
                "n_pairs": len(kinase_subset),
                "targets": list(cond_config["targets"].keys()),
                **pretrain_metrics,
            }
            del model, scaler
            gc.collect()

        # Aggregate results
        ft_folds = eval_results["finetune"]
        ft_mae = np.mean([f["mae"] for f in ft_folds])
        ft_spr = np.mean([f["spearman"] for f in ft_folds])
        ft_mae_std = np.std([f["mae"] for f in ft_folds])
        ft_spr_std = np.std([f["spearman"] for f in ft_folds])

        abs_agg = aggregate_cv_results(eval_results["absolute"]) if eval_results["absolute"] else {}

        print(f"\n  RESULTS ({cond_name}):")
        print(f"    Delta: MAE={ft_mae:.4f}±{ft_mae_std:.4f}, Spr={ft_spr:.3f}±{ft_spr_std:.3f}")
        if abs_agg:
            print(f"    Absolute: MAE={abs_agg['mae_mean']:.4f}±{abs_agg.get('mae_std',0):.4f}, "
                  f"Spr={abs_agg['spearman_r_mean']:.3f}")

        results[cond_name] = {
            "description": cond_config["description"],
            "pretrain": pretrain_info,
            "finetune_folds": ft_folds,
            "no_finetune_folds": eval_results["no_finetune"],
            "absolute_folds": eval_results["absolute"],
            "summary": {
                "delta_mae_mean": float(ft_mae),
                "delta_mae_std": float(ft_mae_std),
                "delta_spearman_mean": float(ft_spr),
                "delta_spearman_std": float(ft_spr_std),
                "absolute": abs_agg if abs_agg else None,
            },
            "completed": True,
        }

        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {RESULTS_FILE.name}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY — Kinase Transfer Ablation")
    print("=" * 70)
    print(f"\n{'Condition':<20} {'Pretrain Pairs':>14} {'δ MAE':>10} {'δ Spr':>10} {'Abs MAE':>10} {'Abs Spr':>10}")
    print("-" * 74)
    for cond_name in ABLATION_CONDITIONS:
        if cond_name in results:
            s = results[cond_name]["summary"]
            n_pairs = results[cond_name]["pretrain"]["n_pairs"]
            abs_mae = s["absolute"]["mae_mean"] if s.get("absolute") else float('nan')
            abs_spr = s["absolute"]["spearman_r_mean"] if s.get("absolute") else float('nan')
            print(f"{cond_name:<20} {n_pairs:>14,} {s['delta_mae_mean']:>10.4f} {s['delta_spearman_mean']:>10.3f} "
                  f"{abs_mae:>10.4f} {abs_spr:>10.3f}")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
