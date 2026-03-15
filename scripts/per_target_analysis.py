#!/usr/bin/env python3
"""
Per-target analysis: train EditDiff + Subtraction, show top problematic targets.
Lightweight one-off script for reporting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False

from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

DEVICE = "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
DROPOUT = 0.2
SEED = 42

CACHE_DIR = Path("data/embedding_cache")
DATA_FILE = Path("data/overlapping_assays/extracted/shared_pairs_deduped.csv")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_metrics(y_true, y_pred):
    if len(y_true) < 5:
        return None
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    try:
        pr, _ = pearsonr(y_true, y_pred)
    except:
        pr = float('nan')
    try:
        sr, _ = spearmanr(y_true, y_pred)
    except:
        sr = float('nan')
    return {"n": len(y_true), "mae": mae, "r2": r2, "pearson": pr, "spearman": sr}


def train_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                val_losses.append(criterion(model(x.to(DEVICE)), y.to(DEVICE)).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model.to(DEVICE)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x.to(DEVICE)).cpu().numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    within = df[df["is_within_assay"] == True].copy()
    print(f"  Within-assay pairs: {len(within):,}")

    # Load embeddings
    print("Loading chemprop-dmpnn embeddings...")
    data = np.load(CACHE_DIR / "chemprop-dmpnn.npz", allow_pickle=True)
    emb_dict = {s: data['embeddings'][i] for i, s in enumerate(data['smiles'].tolist())}
    emb_dim = int(data['emb_dim'])

    # Split (assay-within)
    from src.utils.splits import get_splitter
    splitter = get_splitter("assay", random_state=SEED, scenario="within_assay")
    train_df, val_df, test_df = splitter.split(within)
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    zero = np.zeros(emb_dim)

    def get_embs(df_):
        ea = np.array([emb_dict.get(s, zero) for s in df_["mol_a"]])
        eb = np.array([emb_dict.get(s, zero) for s in df_["mol_b"]])
        d = df_["delta"].values.astype(np.float32)
        return torch.from_numpy(ea).float(), torch.from_numpy(eb).float(), torch.from_numpy(d).float()

    # --- EditDiff ---
    print("\nTraining EditDiff...")
    ea_tr, eb_tr, d_tr = get_embs(train_df)
    x_tr = torch.cat([ea_tr, eb_tr - ea_tr], dim=-1)
    ea_v, eb_v, d_v = get_embs(val_df)
    x_v = torch.cat([ea_v, eb_v - ea_v], dim=-1)

    model_ed = MLP(emb_dim * 2, [512, 256, 128], DROPOUT)
    model_ed = train_model(
        model_ed,
        DataLoader(TensorDataset(x_tr, d_tr), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(x_v, d_v), batch_size=BATCH_SIZE, shuffle=False),
    )

    ea_te, eb_te, d_te = get_embs(test_df)
    x_te = torch.cat([ea_te, eb_te - ea_te], dim=-1)
    pred_edit = predict(model_ed, x_te)

    # --- Subtraction ---
    print("Training Subtraction baseline...")
    mol_vals = {}
    for _, r in train_df.iterrows():
        mol_vals[r["mol_a"]] = r["value_a"]
        mol_vals[r["mol_b"]] = r["value_b"]

    smiles_list = list(mol_vals.keys())
    y_abs = np.array([mol_vals[s] for s in smiles_list], dtype=np.float32)
    X_abs = np.array([emb_dict.get(s, zero) for s in smiles_list], dtype=np.float32)

    val_vals = {}
    for _, r in val_df.iterrows():
        val_vals[r["mol_a"]] = r["value_a"]
        val_vals[r["mol_b"]] = r["value_b"]
    val_smi = list(val_vals.keys())
    val_y = np.array([val_vals[s] for s in val_smi], dtype=np.float32)
    val_X = np.array([emb_dict.get(s, zero) for s in val_smi], dtype=np.float32)

    model_sub = MLP(emb_dim, [512, 256, 128], DROPOUT)
    model_sub = train_model(
        model_sub,
        DataLoader(TensorDataset(torch.from_numpy(X_abs).float(), torch.from_numpy(y_abs).float()),
                   batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float()),
                   batch_size=BATCH_SIZE, shuffle=False),
    )

    pred_sub = predict(model_sub, eb_te) - predict(model_sub, ea_te)

    # --- Per-target analysis ---
    print("\n\nPer-target analysis...")
    y_true = test_df["delta"].values
    targets = test_df["target_chembl_id"].values

    rows = []
    for t in np.unique(targets):
        mask = targets == t
        n = mask.sum()
        if n < 20:
            continue
        m_edit = compute_metrics(y_true[mask], pred_edit[mask])
        m_sub = compute_metrics(y_true[mask], pred_sub[mask])
        if m_edit and m_sub:
            rows.append({
                "target": t, "n_pairs": n,
                "delta_std": np.std(y_true[mask]),
                "edit_mae": m_edit["mae"], "sub_mae": m_sub["mae"],
                "edit_r2": m_edit["r2"], "sub_r2": m_sub["r2"],
                "edit_spearman": m_edit["spearman"], "sub_spearman": m_sub["spearman"],
                "edit_pearson": m_edit["pearson"], "sub_pearson": m_sub["pearson"],
                "mae_improvement": (m_sub["mae"] - m_edit["mae"]) / m_sub["mae"] * 100,
            })

    results = pd.DataFrame(rows)
    results = results.sort_values("mae_improvement", ascending=True)

    print(f"\nTotal targets with >=20 test pairs: {len(results)}")
    print(f"EditDiff wins (lower MAE): {(results['mae_improvement'] > 0).sum()}/{len(results)}")
    print(f"Mean MAE improvement: {results['mae_improvement'].mean():.1f}%")

    # Top 10 worst (where subtraction beats edit)
    print("\n" + "=" * 100)
    print("TOP 10 TARGETS WHERE SUBTRACTION BEATS EDITDIFF (worst for edit effect)")
    print("=" * 100)
    cols = ["target", "n_pairs", "delta_std", "edit_mae", "sub_mae", "mae_improvement",
            "edit_spearman", "sub_spearman", "edit_r2", "sub_r2"]
    worst = results.head(10)[cols]
    for _, r in worst.iterrows():
        print(f"  {r['target']:<20} n={int(r['n_pairs']):>5}  δ_std={r['delta_std']:.3f}"
              f"  MAE: edit={r['edit_mae']:.3f} sub={r['sub_mae']:.3f} ({r['mae_improvement']:+.1f}%)"
              f"  ρ: edit={r['edit_spearman']:.3f} sub={r['sub_spearman']:.3f}"
              f"  R²: edit={r['edit_r2']:.3f} sub={r['sub_r2']:.3f}")

    # Top 10 best (where edit effect shines)
    print("\n" + "=" * 100)
    print("TOP 10 TARGETS WHERE EDITDIFF BEATS SUBTRACTION (best for edit effect)")
    print("=" * 100)
    best = results.tail(10).iloc[::-1][cols]
    for _, r in best.iterrows():
        print(f"  {r['target']:<20} n={int(r['n_pairs']):>5}  δ_std={r['delta_std']:.3f}"
              f"  MAE: edit={r['edit_mae']:.3f} sub={r['sub_mae']:.3f} ({r['mae_improvement']:+.1f}%)"
              f"  ρ: edit={r['edit_spearman']:.3f} sub={r['sub_spearman']:.3f}"
              f"  R²: edit={r['edit_r2']:.3f} sub={r['sub_r2']:.3f}")

    # Targets with highest variance (most challenging)
    print("\n" + "=" * 100)
    print("TOP 10 HIGHEST-VARIANCE TARGETS (most challenging)")
    print("=" * 100)
    high_var = results.nlargest(10, "delta_std")[cols]
    for _, r in high_var.iterrows():
        print(f"  {r['target']:<20} n={int(r['n_pairs']):>5}  δ_std={r['delta_std']:.3f}"
              f"  MAE: edit={r['edit_mae']:.3f} sub={r['sub_mae']:.3f} ({r['mae_improvement']:+.1f}%)"
              f"  ρ: edit={r['edit_spearman']:.3f} sub={r['sub_spearman']:.3f}"
              f"  R²: edit={r['edit_r2']:.3f} sub={r['sub_r2']:.3f}")

    # Summary stats
    print("\n" + "=" * 100)
    print("OVERALL DISTRIBUTION")
    print("=" * 100)
    for label, subset in [("All", results),
                          ("n>=100", results[results["n_pairs"] >= 100]),
                          ("n>=500", results[results["n_pairs"] >= 500])]:
        n = len(subset)
        wins = (subset["mae_improvement"] > 0).sum()
        print(f"  {label:<10} {n:>4} targets, EditDiff wins {wins}/{n} ({100*wins/n:.0f}%)"
              f"  mean MAE imp={subset['mae_improvement'].mean():+.1f}%"
              f"  median={subset['mae_improvement'].median():+.1f}%")

    # Save for later use
    results.to_csv("results/paper_evaluation/per_target_analysis.csv", index=False)
    print(f"\nSaved to results/paper_evaluation/per_target_analysis.csv")


if __name__ == "__main__":
    main()
