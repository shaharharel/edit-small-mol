#!/usr/bin/env python3
"""EXP7-v2 Phase 1 prep for the 54-pair benchmark.

For each of 9 v2 targets, builds a target-specific FiLMDelta scorer:
  - Uses shared_pairs_deduped.csv (1.7M within+cross-assay pairs)
  - Kinase-pretrains on a 30K pair pool (excluding the target)
  - Fine-tunes on target-specific within-assay pairs (if available)
  - For targets with zero data (SOS1, CDK7): saves the kinase-pretrained model directly

Outputs per-target dir at data/exp7_v2_benchmark/_phase1/<target_key>/:
  filmdelta.pt                 -- scorer checkpoint
  warhead_smarts.json          -- per-target SMARTS by warhead_class
  anchor_pool_strategy_b.csv   -- 100 anchors (random target mols, used by REST FiLM ensemble)
  prep_summary.json
"""
from __future__ import annotations
import gc, json, os, sys, time, warnings, inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(4)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

SHARED_PAIRS_CSV = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
V2_BENCH = PROJECT_ROOT / "data" / "exp7_v2_benchmark"
OUT_BASE = V2_BENCH / "_phase1"

N_BITS, RADIUS = 2048, 2
HIDDEN_DIMS, DROPOUT = [512, 256, 128], 0.2
PRETRAIN_LR, FINETUNE_LR = 5e-4, 5e-4
BATCH_SIZE = 256
MAX_EPOCHS, PATIENCE = 20, 5
PRETRAIN_EPOCHS = 3
MAX_KINASE_PAIRS = 30000
HELDOUT_FRAC = 0.10
SEED = 42

# v2 target -> ChEMBL ID
TARGET_CHEMBL_MAP = {
    "SOS1":       "CHEMBL2079846",
    "KRAS_G12D":  "CHEMBL2189121",
    "KRAS_G12C":  "CHEMBL2189121",
    "CDK7":       "CHEMBL3038473",
    "BCL2":       "CHEMBL4860",
    "FGFR1":      "CHEMBL3650",
    "EGFR_T790M": "CHEMBL203",
    "BTK":        "CHEMBL5251",
    "BTK_Cys481": "CHEMBL5251",
}

# Per warhead_class SMARTS (used for warhead_retention metric)
# noncov -> no SMARTS expected (warhead_retention not meaningful, scoring uses neutral 0.5)
WARHEAD_SMARTS = {
    "acrylamide":  {"strict": "C=CC(=O)N",       "generic": "C=CC(=O)N"},
    "chloroacet":  {"strict": "ClCC(=O)N",       "generic": "ClCC(=O)N"},
    "vinylsulfon": {"strict": "C=CS(=O)(=O)",    "generic": "C=CS(=O)(=O)"},
    "epoxide":     {"strict": "C1CO1",           "generic": "C1CO1"},
    # reversible / noncov: any-atom (always matches) — turns off warhead bonus during scoring
    # but still allows pattern logic to run. We'll use a permissive match.
    "reversible":  {"strict": "[*]",             "generic": "[*]"},
    "noncov":      {"strict": "[*]",             "generic": "[*]"},
}

# Kinase + cysteine-target ChEMBL pool for pretrain (added KRAS, BCL2, SOS1 for relevance)
KINASE_POOL_CHEMBL = {
    "CHEMBL1862","CHEMBL258","CHEMBL2599","CHEMBL267","CHEMBL2971","CHEMBL5251",
    "CHEMBL203","CHEMBL5145","CHEMBL2803","CHEMBL5658","CHEMBL4296",
    "CHEMBL3717","CHEMBL279","CHEMBL4005","CHEMBL2842","CHEMBL5407","CHEMBL2492",
    "CHEMBL1075091","CHEMBL2148","CHEMBL1824","CHEMBL2742","CHEMBL3650",
    "CHEMBL4860","CHEMBL2189121","CHEMBL3038473","CHEMBL2079846",
}


def smi_to_morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(N_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


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


def pretrain_film(kinase_df, fp_cache, epochs):
    A = np.stack([fp_cache[s] for s in kinase_df["mol_a"]]).astype(np.float32)
    B = np.stack([fp_cache[s] for s in kinase_df["mol_b"]]).astype(np.float32)
    d = kinase_df["delta"].values.astype(np.float32)
    n_val = max(500, int(0.1 * len(d)))
    A_tr, A_v = A[:-n_val], A[-n_val:]
    B_tr, B_v = B[:-n_val], B[-n_val:]
    d_tr, d_v = d[:-n_val], d[-n_val:]
    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    tl = DataLoader(TensorDataset(torch.from_numpy(A_tr), torch.from_numpy(B_tr), torch.from_numpy(d_tr)), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(A_v), torch.from_numpy(B_v), torch.from_numpy(d_v)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    m = train_loop(m, tl, vl, PRETRAIN_LR, epochs, patience=3, fwd=fwd)
    return {k: v.cpu().clone() for k, v in m.state_dict().items()}


def finetune_film(train_pairs, fp_cache, pre_state, save_path):
    if len(train_pairs) < 20:
        # No data — save the pretrained-only model
        m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        if pre_state is not None:
            m.load_state_dict(pre_state)
        torch.save({
            "model_state_dict": m.state_dict(),
            "hyperparameters": {"input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
                                "spectral": False, "modulation_strength": 1.0, "learning_rate": FINETUNE_LR},
            "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
        }, save_path)
        return {"n_train": 0, "n_test": 0, "delta_mae": float("nan"),
                "delta_spearman": float("nan"), "note": "pretrained_only_no_target_data",
                "model_path": str(save_path)}
    mols = sorted(set(train_pairs["mol_a_id"]).union(train_pairs["mol_b_id"]))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(mols))
    n_test_mols = max(1, int(HELDOUT_FRAC * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test_mols])
    test_mask = train_pairs["mol_a_id"].isin(test_mols) & train_pairs["mol_b_id"].isin(test_mols)
    train_mask = ~train_pairs["mol_a_id"].isin(test_mols) & ~train_pairs["mol_b_id"].isin(test_mols)
    train_split = train_pairs[train_mask].reset_index(drop=True)
    test_split = train_pairs[test_mask].reset_index(drop=True)
    if len(train_split) < 20:
        # rebuild without holding out
        train_split = train_pairs.reset_index(drop=True)
        test_split = train_pairs.iloc[:5].reset_index(drop=True)
    A_tr = np.stack([fp_cache[s] for s in train_split["mol_a"]]).astype(np.float32)
    B_tr = np.stack([fp_cache[s] for s in train_split["mol_b"]]).astype(np.float32)
    d_tr = train_split["delta"].values.astype(np.float32)
    A_te = np.stack([fp_cache[s] for s in test_split["mol_a"]]).astype(np.float32) if len(test_split) else None
    B_te = np.stack([fp_cache[s] for s in test_split["mol_b"]]).astype(np.float32) if len(test_split) else None
    d_te = test_split["delta"].values.astype(np.float32) if len(test_split) else None
    n_val = max(10, int(0.1 * len(train_split)))
    rng2 = np.random.RandomState(SEED + 1)
    perm2 = rng2.permutation(len(train_split))
    val_idx, fit_idx = perm2[:n_val], perm2[n_val:]
    A_fit, B_fit, d_fit = A_tr[fit_idx], B_tr[fit_idx], d_tr[fit_idx]
    A_val, B_val, d_val = A_tr[val_idx], B_tr[val_idx], d_tr[val_idx]
    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if pre_state is not None:
        m.load_state_dict(pre_state)
    tl = DataLoader(TensorDataset(torch.from_numpy(A_fit), torch.from_numpy(B_fit), torch.from_numpy(d_fit)), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(A_val), torch.from_numpy(B_val), torch.from_numpy(d_val)), batch_size=BATCH_SIZE, shuffle=False)
    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    m = train_loop(m, tl, vl, FINETUNE_LR, MAX_EPOCHS, PATIENCE, fwd)
    m.eval()
    mae, spr = float("nan"), float("nan")
    if A_te is not None and len(A_te) >= 5:
        from scipy.stats import spearmanr
        with torch.no_grad():
            d_pred = m(torch.from_numpy(A_te), torch.from_numpy(B_te)).cpu().numpy()
        mae = float(np.mean(np.abs(d_pred - d_te)))
        spr = float(spearmanr(d_pred, d_te)[0]) if np.std(d_pred) > 1e-9 else 0.0
    torch.save({
        "model_state_dict": m.state_dict(),
        "hyperparameters": {"input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
                            "spectral": False, "modulation_strength": 1.0, "learning_rate": FINETUNE_LR},
        "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
    }, save_path)
    return {"n_train": len(fit_idx), "n_val": len(val_idx), "n_test": len(test_split),
            "delta_mae": mae, "delta_spearman": spr, "model_path": str(save_path)}


def build_anchor_pool(target_pairs, anchor_pic_default=6.5, max_anchors=100):
    """Return DataFrame with columns smiles, pIC50 — used by REST FiLM ensemble."""
    rows = []
    rows.extend(zip(target_pairs["mol_a_id"], target_pairs["mol_a"], target_pairs["value_a"]))
    rows.extend(zip(target_pairs["mol_b_id"], target_pairs["mol_b"], target_pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id","smiles","pIC50"]).groupby("mol_id", as_index=False).agg({"smiles":"first","pIC50":"mean"})
    df = df.dropna(subset=["pIC50"])
    if len(df) > max_anchors:
        df = df.sample(n=max_anchors, random_state=SEED).reset_index(drop=True)
    return df[["smiles","pIC50"]]


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"[v2-phase1] Loading {SHARED_PAIRS_CSV.name}...", flush=True)
    all_pairs = pd.read_csv(SHARED_PAIRS_CSV)
    print(f"  loaded {len(all_pairs)} rows", flush=True)

    # 1) Build a shared kinase-pretrain pool ONCE (exclude all v2 target IDs)
    v2_targets_set = set(TARGET_CHEMBL_MAP.values())
    print(f"[v2-phase1] Building kinase pretrain pool (excluding v2 targets: {len(v2_targets_set)})...", flush=True)
    pool = all_pairs[(all_pairs["is_within_assay"] == True)
                     & all_pairs["target_chembl_id"].isin(KINASE_POOL_CHEMBL - v2_targets_set)]
    if len(pool) > MAX_KINASE_PAIRS:
        pool = pool.sample(n=MAX_KINASE_PAIRS, random_state=SEED).reset_index(drop=True)
    print(f"  kinase pool: {len(pool)} pairs across {pool['target_chembl_id'].nunique()} targets", flush=True)

    # Cache FPs for kinase pool (shared)
    kinase_smis = set(pool["mol_a"]).union(pool["mol_b"])
    print(f"[v2-phase1] FP cache for kinase pool ({len(kinase_smis)} smiles)...", flush=True)
    fp_cache = {s: smi_to_morgan(s) for s in kinase_smis}

    # Pretrain ONCE
    print(f"[v2-phase1] Pretraining shared FiLMDelta for {PRETRAIN_EPOCHS} epochs...", flush=True)
    t0 = time.time()
    shared_pretrain_state = pretrain_film(pool, fp_cache, PRETRAIN_EPOCHS)
    print(f"  pretrain done in {(time.time()-t0)/60:.1f} min", flush=True)

    summaries = {}
    for target_key, chembl_id in TARGET_CHEMBL_MAP.items():
        t0 = time.time()
        out_dir = OUT_BASE / target_key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[v2-phase1] === {target_key} (ChEMBL: {chembl_id}) ===", flush=True)

        tp = all_pairs[(all_pairs["is_within_assay"] == True) & (all_pairs["target_chembl_id"] == chembl_id)].copy()
        tp = tp[["assay_id_a","mol_a","mol_b","mol_a_id","mol_b_id","value_a","value_b","delta","target_chembl_id"]].rename(columns={"assay_id_a":"assay_id"}).reset_index(drop=True)
        n_pairs = len(tp)
        n_mols = len(set(tp["mol_a_id"]).union(tp["mol_b_id"])) if n_pairs else 0
        print(f"  target within-assay pairs: {n_pairs}, mols: {n_mols}", flush=True)
        tp.to_csv(out_dir / "target_pairs.csv", index=False)

        # Build FP cache for target mols (add to shared)
        if n_pairs:
            target_smis = set(tp["mol_a"]).union(tp["mol_b"])
            new_smis = [s for s in target_smis if s not in fp_cache]
            for s in new_smis:
                fp_cache[s] = smi_to_morgan(s)
            print(f"  added {len(new_smis)} new FPs", flush=True)

        # Fine-tune (or save pretrained-only if no data)
        film_path = out_dir / "filmdelta.pt"
        if n_pairs > 0:
            print(f"  finetuning FiLMDelta on {n_pairs} pairs...", flush=True)
        else:
            print(f"  NO target data — saving kinase-pretrained model as-is", flush=True)
        film_res = finetune_film(tp, fp_cache, shared_pretrain_state, film_path)
        print(f"  -> MAE={film_res['delta_mae']:.3f} Spr={film_res['delta_spearman']:.3f} (note: {film_res.get('note','')})", flush=True)

        # Build anchor pool (for REST FiLM ensemble)
        if n_pairs > 0:
            ap = build_anchor_pool(tp)
            ap.to_csv(out_dir / "anchor_pool_strategy_b.csv", index=False)
            n_anchors = len(ap)
        else:
            # Use a tiny default: just the 9 anchors for that target from v2 pairs
            with open(V2_BENCH / "clean_pairs.json") as fh:
                pj = json.load(fh)
            anchors = [(p["anchor"]["smiles"], p["anchor"]["pic50"])
                       for p in pj["pairs"] if p["target"] == target_key]
            ap = pd.DataFrame(anchors, columns=["smiles","pIC50"])
            ap.to_csv(out_dir / "anchor_pool_strategy_b.csv", index=False)
            n_anchors = len(ap)
        print(f"  anchor_pool: {n_anchors} mols", flush=True)

        # Warhead SMARTS: pick per target's dominant warhead class
        with open(V2_BENCH / "clean_pairs.json") as fh:
            pj = json.load(fh)
        target_pairs_v2 = [p for p in pj["pairs"] if p["target"] == target_key]
        wh_class = target_pairs_v2[0]["warhead_class"] if target_pairs_v2 else "noncov"
        wh = WARHEAD_SMARTS.get(wh_class, WARHEAD_SMARTS["noncov"])
        wh_spec = {"target_label": target_key, "warhead_class": wh_class,
                   "smarts_strict": wh["strict"], "smarts_generic": wh["generic"]}
        (out_dir / "warhead_smarts.json").write_text(json.dumps(wh_spec, indent=2))

        summaries[target_key] = {
            "target_key": target_key, "chembl_id": chembl_id,
            "warhead_class": wh_class,
            "n_target_pairs": n_pairs, "n_unique_mols": n_mols,
            "filmdelta": film_res,
            "n_anchors": n_anchors,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        gc.collect()

    (OUT_BASE / "phase1_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n=== v2 Phase 1 Summary ===")
    for tk, s in summaries.items():
        fe = s.get("filmdelta", {})
        print(f"  {tk:12s} n_pairs={s['n_target_pairs']:5d} MAE={fe.get('delta_mae', float('nan')):.3f} ({fe.get('note','')})")


if __name__ == "__main__":
    main()
