"""Re-rank the 19-molecule competition cohort with the clean FiLMDelta.
Compare against the original (leaky) FiLMDelta_anchor ranking. Don't update
the live report — just write a side-by-side comparison CSV + Spearman.
"""
from __future__ import annotations
import sys, json, warnings, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")
from scipy.stats import spearmanr, pearsonr
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

SRC = PROJECT_ROOT / "results" / "paper_evaluation" / "19_molecules_scoring.json"
OLD_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
NEW_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"
OUT_CSV  = PROJECT_ROOT / "results" / "paper_evaluation" / "19mol_clean_vs_leaky.csv"


def load_scorer(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck["anchor_embs"], np.asarray(ck["anchor_pIC50"])


def fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), a)
    return a


def score(smi, model, scaler, ae, ap):
    f = fp(smi)
    if f is None: return float("nan")
    e = torch.FloatTensor(scaler.transform(f[None, :]))
    with torch.no_grad():
        d = model(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def main():
    data = json.loads(SRC.read_text())
    items = data["results"]
    print(f"loaded {len(items)} mols from {SRC.name}")
    print(f"loading OLD scorer: {OLD_CKPT.name}")
    m_old, s_old, ae_old, ap_old = load_scorer(OLD_CKPT)
    print(f"loading NEW scorer: {NEW_CKPT.name}")
    m_new, s_new, ae_new, ap_new = load_scorer(NEW_CKPT)
    print(f"  anchors (both): {len(ap_old)}")

    rows = []
    for r in items:
        smi = r.get("smiles_clean") or r.get("smiles_raw")
        old_orig = float(r.get("film_delta", float("nan")))
        new = score(smi, m_new, s_new, ae_new, ap_new)
        old_rerun = score(smi, m_old, s_old, ae_old, ap_old)
        rows.append({
            "idx": r["idx"], "smiles": smi,
            "filmdelta_leaky_published": old_orig,
            "filmdelta_leaky_rerun":     old_rerun,
            "filmdelta_clean":           new,
            "MW": r.get("MW"), "QED": r.get("QED"),
            "tc_to_train_max": r.get("max_tanimoto"),
        })
    df = pd.DataFrame(rows)
    df["rank_leaky"] = df["filmdelta_leaky_rerun"].rank(ascending=False, method="min").astype(int)
    df["rank_clean"] = df["filmdelta_clean"].rank(ascending=False, method="min").astype(int)
    df["rank_delta"] = df["rank_leaky"] - df["rank_clean"]

    sp_lk_pub, _ = spearmanr(df["filmdelta_leaky_published"], df["filmdelta_leaky_rerun"])
    sp_lk_cl,  _ = spearmanr(df["filmdelta_leaky_rerun"], df["filmdelta_clean"])
    pr_lk_cl,  _ = pearsonr(df["filmdelta_leaky_rerun"], df["filmdelta_clean"])

    print(f"\nSpearman(leaky_published, leaky_rerun) = {sp_lk_pub:.3f}  (sanity check: should be ~1.0)")
    print(f"Spearman(leaky_rerun,    clean)        = {sp_lk_cl:.3f}")
    print(f"Pearson (leaky_rerun,    clean)        = {pr_lk_cl:.3f}")
    print(f"\nMolecules with largest rank change (clean vs leaky):")
    df_sorted = df.assign(abs_rd=df["rank_delta"].abs()).sort_values("abs_rd", ascending=False)
    for _, r in df_sorted.head(10).iterrows():
        print(f"  idx{int(r.idx):2d}  rank: leaky #{int(r.rank_leaky):2d} → clean #{int(r.rank_clean):2d}  Δ={int(r.rank_delta):+d}  "
              f"leaky pIC50={r.filmdelta_leaky_rerun:.2f}  clean={r.filmdelta_clean:.2f}  "
              f"QED={r.QED:.2f}  smi={r.smiles[:50]}")
    print(f"\nFull table:")
    cols = ["idx", "rank_leaky", "rank_clean", "rank_delta",
            "filmdelta_leaky_rerun", "filmdelta_clean", "QED", "tc_to_train_max"]
    print(df.sort_values("rank_clean")[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
