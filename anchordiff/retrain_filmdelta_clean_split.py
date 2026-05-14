"""B1 fix: re-train the cached FiLMDelta scorer with a *molecule-disjoint*
held-out validation set, addressing the QA-flagged leak where pair-row-order
splitting let every target molecule appear in both train and val.

Pipeline:
  1. Load 280 ZAP70 anchors via `experiments.run_zap70_v3.load_zap70_molecules`.
  2. Randomly hold out 28 (10%) molecules. ALL pairs touching any held-out mol
     go to val; remaining (252×251 = 63,252) pairs are train.
  3. Pretrain on 32K kinase pairs (cached), then fine-tune on the 252×251
     train pairs.
  4. Early-stop on the molecule-disjoint val MAE.
  5. Save to `results/paper_evaluation/reinvent4_film_model_clean.pt`.
  6. Rescore Day-1 inpaint + cofold leaderboard, write a side-by-side comparison
     against the original ranking.
"""
from __future__ import annotations
import sys, os, warnings, gc, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")
from scipy.stats import spearmanr

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints

DEVICE = torch.device("cpu")
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
OUT_PT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"
SEED = 42


def smi_to_fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a


def main():
    rng = np.random.default_rng(SEED)
    print("Loading 280 ZAP70 anchors")
    smiles_df, _ = load_zap70_molecules()
    smiles = smiles_df["smiles"].tolist()
    pIC50 = smiles_df["pIC50"].values.astype(np.float64)
    n = len(smiles)
    print(f"  {n} mols, pIC50 [{pIC50.min():.2f}, {pIC50.max():.2f}]")
    fps_all = compute_fingerprints(smiles, "morgan", radius=2, n_bits=2048)
    fp_cache = {s: fps_all[i] for i, s in enumerate(smiles)}
    n_hold = max(1, n // 10)
    held = set(rng.choice(n, n_hold, replace=False).tolist())
    print(f"  holding out {n_hold} molecules (indices: {sorted(list(held))[:5]}…)")

    # --- Pretrain on kinase pairs (skip if already cached; same as scorer) ---
    print("Pretraining on kinase pairs…")
    kp = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"]).sample(
        n=min(100_000, sum(1 for _ in open(KINASE_PAIRS_FILE)) - 1), random_state=SEED
    )
    extra = list({s for s in kp.mol_a.tolist() + kp.mol_b.tolist() if s not in fp_cache})
    if extra:
        ex_fps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
        for i, s in enumerate(extra):
            fp_cache[s] = ex_fps[i]
    # build kinase pair tensors
    Xa, Xb, yd = [], [], []
    for _, r in kp.iterrows():
        fa, fb = fp_cache.get(r.mol_a), fp_cache.get(r.mol_b)
        if fa is None or fb is None: continue
        Xa.append(fa); Xb.append(fb); yd.append(float(r.delta))
    Xa, Xb, yd = np.array(Xa), np.array(Xb), np.array(yd, dtype=np.float32)
    scaler = StandardScaler().fit(np.vstack([Xa, Xb]))
    Xa = torch.FloatTensor(scaler.transform(Xa))
    Xb = torch.FloatTensor(scaler.transform(Xb))
    yd = torch.FloatTensor(yd)
    print(f"  {len(yd):,} kinase pretrain pairs")
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    crit = nn.L1Loss()
    n_val = len(Xa) // 10
    best, best_st, w = float("inf"), None, 0
    for ep in range(30):
        model.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), 256):
            bi = perm[s:s+256]
            opt.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
        if vl < best:
            best, best_st, w = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= 5: break
    if best_st: model.load_state_dict(best_st)
    print(f"  pretrain best val MAE: {best:.4f}")
    del Xa, Xb, yd; gc.collect()

    # --- Fine-tune with MOLECULE-DISJOINT split ---
    print("Building train / val pairs (molecule-disjoint)…")
    pairs_train_a, pairs_train_b, pairs_train_d = [], [], []
    pairs_val_a,   pairs_val_b,   pairs_val_d   = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = float(pIC50[j] - pIC50[i])
            if i in held or j in held:
                pairs_val_a.append(fp_cache[smiles[i]]); pairs_val_b.append(fp_cache[smiles[j]]); pairs_val_d.append(d)
            else:
                pairs_train_a.append(fp_cache[smiles[i]]); pairs_train_b.append(fp_cache[smiles[j]]); pairs_train_d.append(d)
    print(f"  train pairs: {len(pairs_train_a):,}    val pairs: {len(pairs_val_a):,}")

    Xa = torch.FloatTensor(scaler.transform(np.array(pairs_train_a)))
    Xb = torch.FloatTensor(scaler.transform(np.array(pairs_train_b)))
    yd = torch.FloatTensor(np.array(pairs_train_d, dtype=np.float32))
    Xa_v = torch.FloatTensor(scaler.transform(np.array(pairs_val_a)))
    Xb_v = torch.FloatTensor(scaler.transform(np.array(pairs_val_b)))
    yd_v = torch.FloatTensor(np.array(pairs_val_d, dtype=np.float32))
    del pairs_train_a, pairs_train_b, pairs_train_d
    del pairs_val_a, pairs_val_b, pairs_val_d
    gc.collect()

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best, best_st, w = float("inf"), None, 0
    for ep in range(60):
        model.train()
        perm = np.random.permutation(len(Xa))
        for s in range(0, len(perm), 256):
            bi = perm[s:s+256]
            opt2.zero_grad()
            crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
            opt2.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xa_v, Xb_v), yd_v).item()
        if ep % 5 == 0:
            print(f"  ep {ep:3d}  train_size={len(Xa):,}  val MAE={vl:.4f}")
        if vl < best:
            best, best_st, w = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= 10: break
    if best_st: model.load_state_dict(best_st)
    print(f"  finetune best mol-disjoint val MAE: {best:.4f}")

    anchor_embs = torch.FloatTensor(scaler.transform(np.array([fp_cache[s] for s in smiles])))
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "anchor_embs": anchor_embs,
        "anchor_pIC50": pIC50.copy(),
        "heldout_idx": sorted(list(held)),
        "val_mae_mol_disjoint": best,
    }, OUT_PT)
    print(f"Saved {OUT_PT}")

    # --- Rank-stability check: rescore Day-1 inpaint + cofold leaderboard ---
    from rdkit import Chem as Ch
    print("\nRank-stability check vs old cached scorer…")
    new_path = OUT_PT
    old_path = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if not old_path.exists():
        print(f"  no old scorer at {old_path}, skipping comparison")
        return

    def score_smi(smi_list, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        m.load_state_dict(ck["model_state"]); m.eval()
        sc = StandardScaler(); sc.mean_=ck["scaler_mean"]; sc.scale_=ck["scaler_scale"]
        sc.var_=sc.scale_**2; sc.n_features_in_=len(sc.mean_)
        ae = ck["anchor_embs"]; ap = np.asarray(ck["anchor_pIC50"])
        out = np.full(len(smi_list), np.nan)
        for i, s in enumerate(smi_list):
            a = smi_to_fp(s)
            if a is None: continue
            e = torch.FloatTensor(sc.transform(a[None,:]))
            with torch.no_grad():
                d = m(ae, e.expand(len(ap),-1)).numpy().flatten()
            out[i] = float(np.mean(ap + d))
        return out

    inp_csv = PROJECT_ROOT / "results" / "anchordiff" / "day1_inpaint_filmdelta_ranking.csv"
    df = pd.read_csv(inp_csv)
    smi_list = df.smiles.tolist()
    df["pIC50_old"] = df["pIC50_filmdelta"]
    df["pIC50_clean"] = score_smi(smi_list, new_path)
    rho, p = spearmanr(df["pIC50_old"], df["pIC50_clean"], nan_policy="omit")
    df.to_csv(inp_csv.with_name("day1_inpaint_filmdelta_ranking_clean.csv"), index=False)
    print(f"  Day-1 inpaint Spearman(old, clean) = {rho:.3f} (n={len(df.dropna(subset=['pIC50_old','pIC50_clean']))})")

    lb_csv = PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv"
    if lb_csv.exists():
        df2 = pd.read_csv(lb_csv)
        df2["pIC50_old"] = df2["rank_score"]
        df2["pIC50_clean"] = score_smi(df2.smiles.tolist(), new_path)
        rho2, _ = spearmanr(df2["pIC50_old"], df2["pIC50_clean"], nan_policy="omit")
        df2.to_csv(lb_csv.with_name("cys346_cofold_leaderboard_clean.csv"), index=False)
        print(f"  997-mol leaderboard Spearman(old, clean) = {rho2:.3f}")
    summary = {
        "val_mae_old_leaky_approx": "see results/paper_evaluation/reinvent4_film_model.pt loss",
        "val_mae_clean_mol_disjoint": best,
        "heldout_idx_count": len(held),
        "rank_stability_day1": float(rho),
    }
    json.dump(summary, open(PROJECT_ROOT/"results"/"anchordiff"/"filmdelta_retrain_summary.json","w"), indent=2)


if __name__ == "__main__":
    main()
