"""Day 2: FiLMDelta-rank Day 1 inpaint candidates.

For each target (ZAP70 + BTK), load the inpaint_fixed.sdf, score each molecule
with the cached FiLMDelta anchor-based predictor, and write a ranked CSV with
the top 20 candidates per target.

For BTK we use the same FiLMDelta model (kinase pretrain → ZAP70 fine-tune)
since that's what we have cached. Treating it as a "kinase potency proxy" —
absolute pIC50 numbers should be interpreted with caution for BTK (the model
wasn't specifically fine-tuned on BTK), but RANKING within each cohort is
informative.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
torch.backends.mps.is_available = lambda: False  # CPU only

from sklearn.preprocessing import StandardScaler
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from rdkit.Chem import AllChem, DataStructs

CACHE = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"


def load_model():
    ckpt = torch.load(CACHE, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return model, scaler, ckpt["anchor_embs"], ckpt["anchor_pIC50"]


def smi_to_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def score_smiles_list(smiles_list, model, scaler, anchor_embs, anchor_pIC50):
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        a = smi_to_fp(smi)
        if a is not None:
            fps.append(a)
            valid_idx.append(i)
    fps = np.array(fps)
    embs = torch.FloatTensor(scaler.transform(fps))
    n_anchors = len(anchor_pIC50)
    out = np.full(len(smiles_list), np.nan, dtype=float)
    with torch.no_grad():
        for k, orig_i in enumerate(valid_idx):
            target = embs[k:k+1].expand(n_anchors, -1)
            deltas = model(anchor_embs, target).numpy().flatten()
            out[orig_i] = float(np.mean(anchor_pIC50 + deltas))
    return out


def parent_tc(smi: str, parent_smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    p = Chem.MolFromSmiles(parent_smi)
    if m is None or p is None:
        return float("nan")
    fp_m = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    fp_p = AllChem.GetMorganFingerprintAsBitVect(p, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_m, fp_p))


def score_cohort(sdf_path: Path, target: str, parent: str,
                 model, scaler, anchor_embs, anchor_pIC50) -> pd.DataFrame:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True)
    rows = []
    for i, m in enumerate(suppl):
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        rows.append({"mol_idx": i, "smiles": smi, "target": target})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pIC50 = score_smiles_list(df["smiles"].tolist(), model, scaler, anchor_embs, anchor_pIC50)
    df["pIC50_filmdelta"] = pIC50
    df["mw"] = df["smiles"].apply(lambda s: Chem.Descriptors.MolWt(Chem.MolFromSmiles(s)))
    df["qed"] = df["smiles"].apply(lambda s: float(Chem.Descriptors.qed(Chem.MolFromSmiles(s))))
    df["tc_to_parent"] = df["smiles"].apply(lambda s: parent_tc(s, parent))
    return df


def main():
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading FiLMDelta model from {CACHE}")
    model, scaler, anchor_embs, anchor_pIC50 = load_model()
    print(f"  anchors: {len(anchor_pIC50)}, anchor pIC50 ∈ [{anchor_pIC50.min():.2f}, {anchor_pIC50.max():.2f}]")

    MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
    IBRUTINIB = "C=CC(=O)N1CCC[C@@H](C1)n1nc(c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21"

    base = PROJECT_ROOT / "anchordiff_results" / "day1"
    cohorts = [
        ("zap70_cys346", MOL1,
         base / "zap70_cys346" / "inpaint_fixed.sdf"),
        ("btk_cys481", IBRUTINIB,
         base / "btk_cys481" / "inpaint_fixed.sdf"),
    ]
    all_dfs = []
    for tgt, parent, sdf in cohorts:
        if not sdf.exists():
            print(f"  {tgt}: {sdf} not found")
            continue
        print(f"\n=== {tgt} ===")
        df = score_cohort(sdf, tgt, parent, model, scaler, anchor_embs, anchor_pIC50)
        if df.empty:
            print("  no valid mols")
            continue
        df = df.sort_values("pIC50_filmdelta", ascending=False).reset_index(drop=True)
        print(f"  scored {len(df)} mols")
        print(f"  pIC50 ∈ [{df['pIC50_filmdelta'].min():.2f}, {df['pIC50_filmdelta'].max():.2f}], "
              f"median {df['pIC50_filmdelta'].median():.2f}")
        print(f"  Top 10:")
        for _, r in df.head(10).iterrows():
            print(f"    pIC50={r['pIC50_filmdelta']:.3f}  MW={r['mw']:.0f}  QED={r['qed']:.2f}  Tc={r['tc_to_parent']:.2f}  {r['smiles'][:60]}")
        all_dfs.append(df)

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        out_csv = out_dir / "day1_inpaint_filmdelta_ranking.csv"
        full.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv} ({len(full)} mols)")


if __name__ == "__main__":
    main()
