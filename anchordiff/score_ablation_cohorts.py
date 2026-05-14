"""Score the 6 warhead-pose ablation cohorts (Day 3 / Task #10).

Loads each SDF in `anchordiff_results/24h/ablation/`, scores every molecule
with the CLEAN FiLMDelta scorer (`reinvent4_film_model_clean.pt`, the
molecule-disjoint val retrain — the leaky one's rankings shuffle a lot on
new chemotypes).

Reports per-cohort:
  - n_valid, %acrylamide-or-chloroacetamide retention
  - FiLMDelta pIC50 distribution (min/median/max)
  - Tc to Mol 1
  - Top-5 candidates per cohort

Writes `results/anchordiff/ablation_scored.csv` with all rows.

This is the data backing two paper claims:
  1. Projector adds value (perturbed → projected scores ≥ baseline scores).
  2. Pre-registered kill-shot: warhead_chloroacetamide vs baseline FiLMDelta
     distribution (this is the in-silico version of the kinact A/B test).
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
RDLogger.DisableLog("rdApp.*")

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL  = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CHLORO = Chem.MolFromSmarts("[Cl][CH2]C(=O)N")
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"


def load_clean_scorer():
    ck = torch.load(CLEAN_CKPT, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    ae = ck["anchor_embs"]; ap = np.asarray(ck["anchor_pIC50"])
    return m, sc, ae, ap


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
    return a


def tc(smi_a, smi_b):
    a, b = fp(smi_a), fp(smi_b)
    if a is None or b is None: return float("nan")
    return DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(Chem.MolFromSmiles(smi_a)),
        Chem.RDKFingerprint(Chem.MolFromSmiles(smi_b)),
    ) if False else DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi_a), 2, nBits=2048),
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi_b), 2, nBits=2048),
    )


def score_one(smi, model, scaler, ae, ap):
    a = fp(smi)
    if a is None: return float("nan")
    e = torch.FloatTensor(scaler.transform(a[None, :]))
    with torch.no_grad():
        d = model(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def main():
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    out_dir.mkdir(exist_ok=True, parents=True)
    model, scaler, ae, ap = load_clean_scorer()
    print(f"loaded clean FiLMDelta scorer ({len(ap)} anchors)")

    ablation_dir = PROJECT_ROOT / "anchordiff_results" / "24h" / "ablation"
    rows = []
    summary = []
    for sdf in sorted(ablation_dir.glob("*.sdf")):
        cohort = sdf.stem
        suppl = Chem.SDMolSupplier(str(sdf), sanitize=False)
        n_total = 0; n_valid = 0; n_acryl = 0; n_chloro = 0
        cohort_rows = []
        for m in suppl:
            if m is None: continue
            n_total += 1
            try: Chem.SanitizeMol(m)
            except Exception: continue
            smi = Chem.MolToSmiles(m)
            if "." in smi: continue
            n_valid += 1
            has_acryl = m.HasSubstructMatch(ACRYL)
            has_chloro = m.HasSubstructMatch(CHLORO)
            if has_acryl: n_acryl += 1
            if has_chloro: n_chloro += 1
            pIC50 = score_one(smi, model, scaler, ae, ap)
            cohort_rows.append({
                "cohort": cohort, "smiles": smi,
                "MW": Descriptors.MolWt(m),
                "QED": float(Descriptors.qed(m)),
                "has_acrylamide": int(has_acryl),
                "has_chloroacetamide": int(has_chloro),
                "pIC50_clean": pIC50,
                "tc_to_mol1": tc(smi, MOL1),
            })
        df = pd.DataFrame(cohort_rows)
        if len(df):
            print(f"\n=== {cohort} ===")
            print(f"  n_read={n_total}, n_valid={n_valid}, "
                  f"acrylamide={n_acryl}/{n_valid} ({100*n_acryl/max(n_valid,1):.0f}%), "
                  f"chloroacetamide={n_chloro}/{n_valid} ({100*n_chloro/max(n_valid,1):.0f}%)")
            print(f"  pIC50 (clean):  min={df['pIC50_clean'].min():.2f}  "
                  f"median={df['pIC50_clean'].median():.2f}  "
                  f"max={df['pIC50_clean'].max():.2f}")
            print(f"  Tc to Mol1:  median={df['tc_to_mol1'].median():.2f}  "
                  f"max={df['tc_to_mol1'].max():.2f}")
            print(f"  Top 5:")
            for _, r in df.nlargest(5, "pIC50_clean").iterrows():
                print(f"    pIC50={r['pIC50_clean']:.3f}  MW={r['MW']:.0f}  "
                      f"QED={r['QED']:.2f}  Tc={r['tc_to_mol1']:.2f}  {r['smiles'][:60]}")
            rows.extend(cohort_rows)
            summary.append({
                "cohort": cohort, "n_total": n_total, "n_valid": n_valid,
                "frac_acryl": n_acryl / max(n_valid, 1),
                "frac_chloro": n_chloro / max(n_valid, 1),
                "pIC50_min": df["pIC50_clean"].min(),
                "pIC50_med": df["pIC50_clean"].median(),
                "pIC50_max": df["pIC50_clean"].max(),
                "tc_med": df["tc_to_mol1"].median(),
                "tc_max": df["tc_to_mol1"].max(),
            })

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "ablation_scored.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"\nwrote {out_dir/'ablation_scored.csv'}")
    print(f"wrote {out_dir/'ablation_summary.csv'}")

    # Headline comparison: warhead_chloroacetamide vs baseline FiLMDelta distribution
    print("\n=== KILL-SHOT: chloroacetamide vs baseline acrylamide ===")
    if {"warhead_chloroacetamide", "baseline"} <= set(full["cohort"].unique()):
        a = full[full["cohort"] == "baseline"]["pIC50_clean"].dropna()
        b = full[full["cohort"] == "warhead_chloroacetamide"]["pIC50_clean"].dropna()
        from scipy.stats import ks_2samp, mannwhitneyu
        ks_stat, ks_p = ks_2samp(a, b)
        mwu_stat, mwu_p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"  baseline:        n={len(a)}, med pIC50={a.median():.2f}")
        print(f"  chloroacetamide: n={len(b)}, med pIC50={b.median():.2f}")
        print(f"  KS p={ks_p:.4f}  MWU p={mwu_p:.4f}  ΔMed={b.median()-a.median():+.3f}")
        verdict = "WARHEAD-AGNOSTIC" if ks_p > 0.05 else ("BASELINE ahead" if a.median() > b.median() else "CHLORO ahead")
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
