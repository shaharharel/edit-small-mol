"""Head-to-head audit: vanilla DiffSBDD vs fine-tuned C+D on ZAP70 + portability cohorts.

Reads:
  results/9h_run/phase2/zap70_vanilla.sdf      (196 mols)
  results/9h_run/phase3/btk_vanilla.sdf        ( 90 mols)
  results/9h_run/phase4/zap70_finetuned.sdf    (197 mols)
  results/9h_run/phase5/egfr_finetuned.sdf     ( 97 mols)
  results/9h_run/phase6/kras_finetuned.sdf     ( 97 mols)

Per cohort, computes:
  - n_valid (SMILES parse + no '.')
  - %acrylamide  (SMARTS retained)
  - drug-likeness: median MW, QED, SAS, %Lipinski
  - Tc to Mol 1 (just for ZAP70 cohorts)
  - top-5 by clean FiLMDelta pIC50

Prints a side-by-side table (vanilla vs fine-tuned on ZAP70) + portability summary.
"""
from __future__ import annotations
import sys, warnings, os
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
from rdkit.Chem import AllChem, Descriptors, DataStructs, Crippen
RDLogger.DisableLog("rdApp.*")
from scipy.stats import mannwhitneyu, ks_2samp
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"


def calc_sas(mol):
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def load_scorer(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck["anchor_embs"], np.asarray(ck["anchor_pIC50"])


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
    return a


def score_smi(smi, model, scaler, ae, ap):
    a = fp(smi)
    if a is None: return float("nan")
    e = torch.FloatTensor(scaler.transform(a[None, :]))
    with torch.no_grad():
        d = model(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def parent_tc(smi, parent_fp):
    a = fp(smi)
    if a is None: return float("nan")
    # Tc with the parent's full Morgan FP
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048),
        parent_fp,
    )


def audit_cohort(sdf_path, name, scorer):
    print(f"\n=== {name} ===")
    print(f"  src: {sdf_path}")
    if not sdf_path.exists():
        print(f"  MISSING"); return None
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    rows = []
    for i, m in enumerate(suppl):
        if m is None: continue
        try: Chem.SanitizeMol(m)
        except Exception: continue
        smi = Chem.MolToSmiles(m)
        if "." in smi: continue
        try:
            mw = Descriptors.MolWt(m); logp = Crippen.MolLogP(m); qed = float(Descriptors.qed(m))
            hba = Descriptors.NumHAcceptors(m); hbd = Descriptors.NumHDonors(m)
            rotb = Descriptors.NumRotatableBonds(m)
            sas = calc_sas(m)
            acryl = int(m.HasSubstructMatch(ACRYL))
            lipinski = int(mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)
            rows.append({
                "smi": smi, "MW": mw, "logP": logp, "QED": qed, "SAS": sas,
                "acryl": acryl, "lipinski": lipinski, "rotbonds": rotb,
            })
        except Exception: continue
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  0 valid"); return None
    print(f"  valid: {len(df)}  acrylamide: {df.acryl.sum()}/{len(df)} ({100*df.acryl.mean():.0f}%)  "
          f"lipinski: {df.lipinski.sum()}/{len(df)} ({100*df.lipinski.mean():.0f}%)")
    print(f"  MW: med={df.MW.median():.0f}  IQR=[{df.MW.quantile(0.25):.0f},{df.MW.quantile(0.75):.0f}]  "
          f"range=[{df.MW.min():.0f},{df.MW.max():.0f}]")
    print(f"  QED: med={df.QED.median():.2f}  IQR=[{df.QED.quantile(0.25):.2f},{df.QED.quantile(0.75):.2f}]")
    print(f"  SAS: med={df.SAS.median():.2f}")
    # FiLMDelta scoring + Tc to Mol 1
    if scorer is not None:
        m_, s_, ae, ap = scorer
        parent_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(MOL1), 2, nBits=2048)
        df["pIC50_clean"] = df.smi.apply(lambda s: score_smi(s, m_, s_, ae, ap))
        df["tc_to_mol1"] = df.smi.apply(lambda s: parent_tc(s, parent_fp))
        print(f"  pIC50 (clean FiLMDelta): med={df.pIC50_clean.median():.2f}  max={df.pIC50_clean.max():.2f}")
        print(f"  Tc to Mol1: med={df.tc_to_mol1.median():.3f}  max={df.tc_to_mol1.max():.3f}")
        print(f"  Top 5 by pIC50:")
        for _, r in df.nlargest(5, "pIC50_clean").iterrows():
            print(f"    pIC50={r.pIC50_clean:.2f}  MW={r.MW:.0f}  QED={r.QED:.2f}  "
                  f"Tc={r.tc_to_mol1:.2f}  {r.smi[:60]}")
    df["cohort"] = name
    return df


def main():
    print(f"Loading clean FiLMDelta scorer: {CLEAN_CKPT.name}")
    scorer = load_scorer(CLEAN_CKPT)
    base = PROJECT_ROOT / "results" / "9h_run"
    cohorts = {
        "ZAP70 vanilla":    base / "phase2" / "zap70_vanilla.sdf",
        "ZAP70 fine-tuned": base / "phase4" / "zap70_finetuned.sdf",
        "BTK vanilla":      base / "phase3" / "btk_vanilla.sdf",
        "EGFR fine-tuned":  base / "phase5" / "egfr_finetuned.sdf",
        "KRAS fine-tuned":  base / "phase6" / "kras_finetuned.sdf",
    }
    dfs = {}
    for name, p in cohorts.items():
        dfs[name] = audit_cohort(p, name, scorer)

    # Head-to-head: ZAP70 vanilla vs ZAP70 fine-tuned
    if dfs["ZAP70 vanilla"] is not None and dfs["ZAP70 fine-tuned"] is not None:
        a = dfs["ZAP70 vanilla"]; b = dfs["ZAP70 fine-tuned"]
        print("\n" + "=" * 70)
        print("HEAD-TO-HEAD: ZAP70 vanilla DiffSBDD vs ZAP70 fine-tuned D+C")
        print("=" * 70)
        for metric in ["acryl", "lipinski", "MW", "QED", "SAS", "pIC50_clean", "tc_to_mol1"]:
            av = a[metric].astype(float); bv = b[metric].astype(float)
            if metric in {"acryl", "lipinski"}:
                print(f"  {metric:18s}  vanilla {av.mean()*100:5.1f}%  vs  fine-tuned {bv.mean()*100:5.1f}%")
            else:
                u, p = mannwhitneyu(av.dropna(), bv.dropna(), alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
                print(f"  {metric:18s}  vanilla med={av.median():.3f}  fine-tuned med={bv.median():.3f}  "
                      f"ΔMed={bv.median()-av.median():+.3f}  MWU p={p:.4f} {sig}")

    out_dir = PROJECT_ROOT / "results" / "9h_run"
    full = pd.concat([df for df in dfs.values() if df is not None], ignore_index=True)
    full.to_csv(out_dir / "all_cohorts_scored.csv", index=False)
    print(f"\nwrote {out_dir/'all_cohorts_scored.csv'}")


if __name__ == "__main__":
    main()
