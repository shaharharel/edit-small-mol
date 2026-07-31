#!/usr/bin/env python3
"""EXP7 LO eval — Phase 1 (LOCAL, ~10 min per target).

For each of 5 covalent kinase targets, build:
  - target_pairs.csv (all within-assay pairs for the target's ChEMBL IDs)
  - exclude_set.csv (EXP7 policy: drug SMILES + named clinical successors ONLY;
                     NO Tc>=0.6 filter; canonicalized via RDKit)
  - train_pairs.csv (target_pairs minus pairs touching exclude_set by SMILES)
  - filmdelta.pt (kinase-pretrained -> target-finetuned)
  - directabs.pt (kinase-pretrained -> target-finetuned)
  - warhead_smarts.json (acrylamide-on-* SMARTS per spec)
  - prep_summary.json (sanity numbers + held-out MAE)

Reuses kinase pretrain pool logic from exp6.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False  # CPU only per CLAUDE.md
torch.set_num_threads(4)

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa: E402

from experiments.exp7_pair_loader import (  # noqa: E402
    load_all_pairs,
    NAMED_SUCCESSORS,
    TARGET_CHEMBL,
)

# Config
SHARED_PAIRS_CSV = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
OUT_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_phase1"
N_BITS = 2048
RADIUS = 2
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.2
PRETRAIN_LR = 5e-4
FINETUNE_LR = 5e-4
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 5
PRETRAIN_EPOCHS = 3
MAX_KINASE_PAIRS = 30000
HELDOUT_FRAC = 0.10
SEED = 42

# Warhead SMARTS per target (covalent kinase irreversible warheads)
WARHEAD_SMARTS = {
    "egfr_t790m": {"strict": "[NH]([c])C(=O)C=C", "generic": "C=CC(=O)N"},
    "btk":        {"strict": "C=CC(=O)N1CCCCC1", "generic": "C=CC(=O)N"},
    "jak3":       {"strict": "C=CC(=O)N1CC[CH]([CH])[CH]C1", "generic": "C=CC(=O)N"},
    "her2":       {"strict": "C(=O)/C=C/CN", "generic": "C(=O)C=C"},
    "fgfr":       {"strict": "C=CC(=O)N", "generic": "C=CC(=O)N"},
}


def smi_to_morgan(smi: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(N_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


class DirectAbsoluteMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_loop(model, train_loader, val_loader, lr, max_epochs, patience, fwd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            pred, y = fwd(model, batch)
            opt.zero_grad()
            loss_fn(pred, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vl = [loss_fn(*fwd(model, b)).item() for b in val_loader]
            v = float(np.mean(vl))
            sched.step(v)
            if v < best - 1e-5:
                best, best_state, bad = v, {k: vv.cpu().clone() for k, vv in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_kinase_pool(all_pairs, exclude_targets, max_pairs):
    pool = all_pairs[(all_pairs["is_within_assay"] == True)
                     & (~all_pairs["target_chembl_id"].isin(exclude_targets))]
    kinase_targets = {"CHEMBL1862","CHEMBL258","CHEMBL2599","CHEMBL267","CHEMBL2971","CHEMBL5251",
                      "CHEMBL203","CHEMBL5145","CHEMBL2803","CHEMBL2189121","CHEMBL5658","CHEMBL4296",
                      "CHEMBL3717","CHEMBL279","CHEMBL4005","CHEMBL2842","CHEMBL5407","CHEMBL2492",
                      "CHEMBL1075091","CHEMBL2148","CHEMBL1824","CHEMBL2742","CHEMBL3650"}
    kinase_targets -= set(exclude_targets)
    pool = pool[pool["target_chembl_id"].isin(kinase_targets)]
    if len(pool) > max_pairs:
        pool = pool.sample(n=max_pairs, random_state=SEED).reset_index(drop=True)
    return pool


def pretrain_film(kinase_df, fp_cache, epochs):
    A = np.stack([fp_cache[s] for s in kinase_df["mol_a"]]).astype(np.float32)
    B = np.stack([fp_cache[s] for s in kinase_df["mol_b"]]).astype(np.float32)
    d = kinase_df["delta"].values.astype(np.float32)
    n_val = max(500, int(0.1 * len(d)))
    A_tr, A_v = A[:-n_val], A[-n_val:]
    B_tr, B_v = B[:-n_val], B[-n_val:]
    d_tr, d_v = d[:-n_val], d[-n_val:]
    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(A_tr), torch.from_numpy(B_tr), torch.from_numpy(d_tr)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(A_v), torch.from_numpy(B_v), torch.from_numpy(d_v)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    m = train_loop(m, train_loader, val_loader, PRETRAIN_LR, epochs, patience=3, fwd=fwd)
    return {k: v.cpu().clone() for k, v in m.state_dict().items()}


def pretrain_dabs(kinase_df, fp_cache, epochs):
    rows = []
    rows.extend(zip(kinase_df["mol_a_id"], kinase_df["mol_a"], kinase_df["value_a"]))
    rows.extend(zip(kinase_df["mol_b_id"], kinase_df["mol_b"], kinase_df["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id","smiles","value"]).groupby("mol_id", as_index=False).agg({"smiles":"first","value":"mean"})
    X = np.stack([fp_cache[s] for s in df["smiles"]]).astype(np.float32)
    y = df["value"].values.astype(np.float32)
    n_v = max(200, int(0.1 * len(y)))
    X_tr, X_v = X[:-n_v], X[-n_v:]
    y_tr, y_v = y[:-n_v], y[-n_v:]
    m = DirectAbsoluteMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        x, y = [t.float() for t in batch]
        return model(x), y
    m = train_loop(m, train_loader, val_loader, PRETRAIN_LR, epochs, patience=3, fwd=fwd)
    return {k: v.cpu().clone() for k, v in m.state_dict().items()}


def finetune_film(train_pairs, fp_cache, pre_state, save_path):
    mols = sorted(set(train_pairs["mol_a_id"]).union(train_pairs["mol_b_id"]))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(mols))
    n_test_mols = max(1, int(HELDOUT_FRAC * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test_mols])
    test_mask = train_pairs["mol_a_id"].isin(test_mols) & train_pairs["mol_b_id"].isin(test_mols)
    train_mask = ~train_pairs["mol_a_id"].isin(test_mols) & ~train_pairs["mol_b_id"].isin(test_mols)
    train_split = train_pairs[train_mask].reset_index(drop=True)
    test_split = train_pairs[test_mask].reset_index(drop=True)
    if len(train_split) < 20 or len(test_split) < 5:
        return {"n_train": len(train_split), "n_test": len(test_split), "delta_mae": float("nan"),
                "delta_spearman": float("nan"), "note": "insufficient_data"}
    A_tr = np.stack([fp_cache[s] for s in train_split["mol_a"]]).astype(np.float32)
    B_tr = np.stack([fp_cache[s] for s in train_split["mol_b"]]).astype(np.float32)
    d_tr = train_split["delta"].values.astype(np.float32)
    A_te = np.stack([fp_cache[s] for s in test_split["mol_a"]]).astype(np.float32)
    B_te = np.stack([fp_cache[s] for s in test_split["mol_b"]]).astype(np.float32)
    d_te = test_split["delta"].values.astype(np.float32)
    n_val = max(20, int(0.1 * len(train_split)))
    rng2 = np.random.RandomState(SEED + 1)
    perm2 = rng2.permutation(len(train_split))
    val_idx = perm2[:n_val]; fit_idx = perm2[n_val:]
    A_fit, B_fit, d_fit = A_tr[fit_idx], B_tr[fit_idx], d_tr[fit_idx]
    A_val, B_val, d_val = A_tr[val_idx], B_tr[val_idx], d_tr[val_idx]
    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if pre_state is not None:
        m.load_state_dict(pre_state)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(A_fit), torch.from_numpy(B_fit), torch.from_numpy(d_fit)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(A_val), torch.from_numpy(B_val), torch.from_numpy(d_val)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    m = train_loop(m, train_loader, val_loader, FINETUNE_LR, MAX_EPOCHS, PATIENCE, fwd)
    m.eval()
    with torch.no_grad():
        d_pred = m(torch.from_numpy(A_te), torch.from_numpy(B_te)).cpu().numpy()
    mae = float(np.mean(np.abs(d_pred - d_te)))
    sr = float(scipy_stats.spearmanr(d_pred, d_te)[0]) if np.std(d_pred) > 1e-9 else 0.0
    torch.save({
        "model_state_dict": m.state_dict(),
        "hyperparameters": {"input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
                            "spectral": False, "modulation_strength": 1.0,
                            "learning_rate": FINETUNE_LR},
        "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
    }, save_path)
    return {"n_train_pairs": len(fit_idx), "n_val_pairs": len(val_idx),
            "n_test_pairs": len(test_split), "n_test_mols": n_test_mols,
            "delta_mae": mae, "delta_spearman": sr,
            "model_path": str(save_path)}


def finetune_dabs(train_pairs, fp_cache, pre_state, save_path):
    rows = []
    rows.extend(zip(train_pairs["mol_a_id"], train_pairs["mol_a"], train_pairs["value_a"]))
    rows.extend(zip(train_pairs["mol_b_id"], train_pairs["mol_b"], train_pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id","smiles","value"]).groupby("mol_id", as_index=False).agg({"smiles":"first","value":"mean"})
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_test = max(5, int(HELDOUT_FRAC * len(df)))
    test_df = df.iloc[perm[:n_test]].reset_index(drop=True)
    train_df = df.iloc[perm[n_test:]].reset_index(drop=True)
    X_tr = np.stack([fp_cache[s] for s in train_df["smiles"]]).astype(np.float32)
    y_tr = train_df["value"].values.astype(np.float32)
    X_te = np.stack([fp_cache[s] for s in test_df["smiles"]]).astype(np.float32)
    y_te = test_df["value"].values.astype(np.float32)
    n_val = max(20, int(0.1 * len(train_df)))
    rng2 = np.random.RandomState(SEED + 1)
    perm2 = rng2.permutation(len(train_df))
    val_idx, fit_idx = perm2[:n_val], perm2[n_val:]
    X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    m = DirectAbsoluteMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
    if pre_state is not None:
        m.load_state_dict(pre_state)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        x, y = [t.float() for t in batch]
        return model(x), y
    m = train_loop(m, train_loader, val_loader, FINETUNE_LR, MAX_EPOCHS, PATIENCE, fwd)
    m.eval()
    with torch.no_grad():
        y_pred = m(torch.from_numpy(X_te)).cpu().numpy()
    mae = float(np.mean(np.abs(y_pred - y_te)))
    sr = float(scipy_stats.spearmanr(y_pred, y_te)[0]) if np.std(y_pred) > 1e-9 else 0.0
    torch.save({
        "model_state_dict": m.state_dict(),
        "hyperparameters": {"input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT},
        "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
    }, save_path)
    return {"n_train_mols": len(fit_idx), "n_val_mols": len(val_idx),
            "n_test_mols": len(test_df), "abs_mae": mae, "abs_spearman": sr,
            "model_path": str(save_path)}


def build_exclude_canonset(pairs_for_target: list[dict], target_key: str) -> dict:
    """EXP7 exclude: drug + named successors only (canonical SMILES set)."""
    cs = set()
    items = {}
    for pair in pairs_for_target:
        # add drug
        dc = _canon(pair["drug_smiles"])
        if dc:
            cs.add(dc)
            items[dc] = f"drug:{pair['drug_name']}"
    # add named successors per target
    for name, smi in NAMED_SUCCESSORS.get(target_key, {}).items():
        c = _canon(smi)
        if c:
            cs.add(c)
            items[c] = f"successor:{name}"
    return {"canon_set": cs, "items": items}


def process_target(tk: str, target_chembl_ids: list[str], target_label: str,
                   pairs_for_target: list[dict], all_pairs: pd.DataFrame) -> dict:
    out_dir = OUT_BASE / tk
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    def L(s):
        msg = f"[{tk}] {s}"
        print(msg, flush=True)
        log.append(msg)

    t0 = time.time()
    L(f"=== {target_label} ===")

    # 1) Target pairs
    tp = all_pairs[(all_pairs["is_within_assay"] == True) & (all_pairs["target_chembl_id"].isin(target_chembl_ids))].copy()
    tp["assay_id"] = tp["assay_id_a"]
    tp = tp[["assay_id","mol_a","mol_b","mol_a_id","mol_b_id","value_a","value_b","delta","target_chembl_id"]].reset_index(drop=True)
    n_mols = len(set(tp["mol_a_id"]).union(tp["mol_b_id"]))
    L(f"  target_pairs: {len(tp)}, mols={n_mols}, targets={tp['target_chembl_id'].nunique()}")
    tp.to_csv(out_dir / "target_pairs.csv", index=False)

    # 2) Exclude set
    excl = build_exclude_canonset(pairs_for_target, tk)
    # Build CHEMBL_ID set for filtering by canonical SMILES match
    excl_pair_smiles = excl["canon_set"]
    L(f"  exclude_set (drug + named successors): {len(excl_pair_smiles)} canonical SMILES")
    excl_df = pd.DataFrame([{"canon_smiles": cs, "kind": kind} for cs, kind in excl["items"].items()])
    excl_df.to_csv(out_dir / "exclude_set.csv", index=False)

    # 3) Drop train pairs whose mol_a or mol_b canonical match exclude
    L("  canonicalizing target pair SMILES (slow)...")
    can_cache = {}
    def canon(s):
        if s not in can_cache:
            can_cache[s] = _canon(s)
        return can_cache[s]
    keep = []
    for i, row in tp.iterrows():
        ca, cb = canon(row["mol_a"]), canon(row["mol_b"])
        if ca in excl_pair_smiles or cb in excl_pair_smiles:
            continue
        keep.append(i)
    train_pairs = tp.loc[keep].reset_index(drop=True)
    train_pairs.to_csv(out_dir / "train_pairs.csv", index=False)
    pct_lost = 100.0 * (1 - len(train_pairs) / max(1, len(tp)))
    L(f"  train_pairs after exclude: {len(train_pairs)} (lost {pct_lost:.2f}%)")

    if len(train_pairs) < 50:
        L(f"  CRITICAL: <50 train pairs — Phase 1 NO-GO")
        return {"target_key": tk, "go_no_go": "NO-GO", "n_train_pairs": len(train_pairs)}

    # 4) Warhead SMARTS
    wh = WARHEAD_SMARTS.get(tk, {"strict": "C=CC(=O)N", "generic": "C=CC(=O)N"})
    wh_json = {"target_label": target_label, "smarts_strict": wh["strict"],
               "smarts_generic": wh["generic"]}
    (out_dir / "warhead_smarts.json").write_text(json.dumps(wh_json, indent=2))

    # 5) Build FP cache
    L("  building Morgan FP cache + kinase pretrain pool...")
    kinase_pool = build_kinase_pool(all_pairs, target_chembl_ids, MAX_KINASE_PAIRS)
    L(f"  kinase_pretrain_pool: {len(kinase_pool)} pairs across {kinase_pool['target_chembl_id'].nunique()} targets")
    smis = set()
    for df in [train_pairs, kinase_pool]:
        smis.update(df["mol_a"].tolist())
        smis.update(df["mol_b"].tolist())
    fp_cache = {s: smi_to_morgan(s) for s in smis}

    # 6) Pretrain + finetune FiLMDelta
    L("  pretraining FiLMDelta...")
    film_pre = pretrain_film(kinase_pool, fp_cache, PRETRAIN_EPOCHS)
    L("  pretraining DirectAbs...")
    dabs_pre = pretrain_dabs(kinase_pool, fp_cache, PRETRAIN_EPOCHS)

    L("  finetuning FiLMDelta on target...")
    film_path = out_dir / "filmdelta.pt"
    film_res = finetune_film(train_pairs, fp_cache, film_pre, film_path)
    L(f"    FiLMDelta held-out MAE={film_res.get('delta_mae', float('nan')):.3f} Spr={film_res.get('delta_spearman', float('nan')):.3f}")

    L("  finetuning DirectAbs on target...")
    dabs_path = out_dir / "directabs.pt"
    dabs_res = finetune_dabs(train_pairs, fp_cache, dabs_pre, dabs_path)
    L(f"    DirectAbs held-out MAE={dabs_res.get('abs_mae', float('nan')):.3f} Spr={dabs_res.get('abs_spearman', float('nan')):.3f}")

    elapsed = time.time() - t0
    summary = {
        "target_key": tk, "target_label": target_label,
        "target_chembl_ids": target_chembl_ids,
        "n_target_pairs": len(tp), "n_unique_mols": n_mols,
        "n_exclude": len(excl_pair_smiles), "n_train_pairs": len(train_pairs),
        "pct_lost": pct_lost,
        "exp7_exclude_policy": "drug + named successors only (NO Tc>=0.6)",
        "warhead": wh_json,
        "filmdelta_eval": film_res,
        "directabs_eval": dabs_res,
        "kinase_pretrain_n_pairs": len(kinase_pool),
        "elapsed_sec": round(elapsed, 1),
        "go_no_go": "GO" if len(train_pairs) >= 500 else ("CONDITIONAL" if len(train_pairs) >= 100 else "NO-GO"),
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    L(f"  DONE in {elapsed/60:.1f} min — verdict: {summary['go_no_go']}")

    del fp_cache, kinase_pool
    gc.collect()
    return summary


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("Loading shared_pairs (1.7M)...", flush=True)
    all_pairs = pd.read_csv(SHARED_PAIRS_CSV)
    print(f"  loaded {len(all_pairs)} rows", flush=True)

    all_curated = load_all_pairs()
    by_target = {}
    for p in all_curated:
        by_target.setdefault(p["target_key"], []).append(p)

    summaries = {}
    for tk, pairs_for_target in by_target.items():
        target_chembl_ids, target_label = TARGET_CHEMBL[tk]
        s = process_target(tk, target_chembl_ids, target_label, pairs_for_target, all_pairs)
        summaries[tk] = s

    (OUT_BASE / "phase1_all_targets_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n=== Phase 1 Summary ===")
    for tk, s in summaries.items():
        if "filmdelta_eval" in s:
            fe = s["filmdelta_eval"]
            print(f"  {tk}: n_train={s['n_train_pairs']}, FiLM MAE={fe.get('delta_mae','?'):.3f}, "
                  f"Spr={fe.get('delta_spearman','?'):.3f}, verdict={s['go_no_go']}")
        else:
            print(f"  {tk}: {s.get('go_no_go','?')} ({s.get('n_train_pairs','?')} train pairs)")


if __name__ == "__main__":
    main()
