#!/usr/bin/env python3
"""Simple classifier baseline for 19-mol ranking: train on 280 ZAP70 mols
(Morgan FP → pIC50) WITHOUT pairs training, predict for 19 candidates.

Three baselines:
  1. RandomForest ensemble
  2. XGBoost ensemble
  3. Ridge regression (linear baseline)

For each: 20 seeds (different bootstrap subsamples), mean prediction.

Comparison vs FiLMDelta 20-seed (pairs) prediction.

Output: results/paper_evaluation/19mol_simple_classifier_baseline.json + console.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_ROOT = Path(__file__).parent.parent
RD = PROJECT_ROOT / "results" / "paper_evaluation"
RAW = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
N_SEEDS = 20

SMILES_19 = [
    'C=CC(=O)N1CC=2C=CC=C(C(=O)NC3=CN(C=N3)C(C)C)C2C1',
    'C=CC(=O)N1CC2(CCC(=O)NC3=CNN=C3C(=O)NCC(C)O)CCC1CC2',
    'C=CC(=O)NC[C@H]1C[C@H]2C[C@@H]1CN2C=3N=CN=C(N)C3Cl',
    'C=CC(=O)N1CCCC1(C(=O)NC2=CNN=C2OCC(F)F)C=3C=CC=CC3',
    'C=CC(=O)N1CCC(CC(=O)NC2=CNN=C2C=3C=NC=CN3)CC41CC4',
    'C=CC(=O)N1CC2(CCC2)CC1CC(=O)NC3=CNN=C3C(=O)NCC(C)O',
    'C=CC(=O)N1C[C@H](CC(C)(C)C)[C@H](C1)C(=O)NC2=CNN=C2C(=O)NCC(C)O',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C3=CNC=4C=C(F)C(Cl)=CC34)S2',
    'C=CC(=O)NC1C2C3C[C@@H]1[C@H](C(=O)NC4=CNN=C4C=5C=NC=CN5)C32',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C=3C=CC(F)=C(C3)S(=O)(=O)N(C)C)S2',
    'C=CC(=O)NCC1=CN(N=N1)[C@@H]2C[C@H](C2)C(=O)NC=3C=CC=NC3NC(C)=O',
    'C=CC(=O)N1CC2(CC1CCC2)NC=3N=CC=C(N3)OC=4C=CC=C(C#N)C4',
    'C=CC(=O)NC(C)C=1N=CC(=CN1)NC(=O)C=2N=CN=C3NC=C(C)C23',
    'C=CC(=O)N(C)C1(CNC=2N=CN=C(N)C2C(=O)OCC)CCC1',
    'C=CC(=O)N1CCC[C@H]1C(C)NC(=O)C=2C=NNC2C=3C=CN(C)N3',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CC(CCNC=2C=C(N=CN2)NC=3C=CC=CC3)(C1)N(C)C',
    'O=C(O)C(F)(F)F.C=CC(=O)N(C)C1(CNC=2N=CN=C3NC=C(C(N)=O)C23)CCC1',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CCN(CC1)C=2N=CN=C(N2)NC3(CC)CCNCC3',
    'C=CC(=O)N1CCCC(NC(=O)C2=CNN=C2C3=CC=4C=CC=CC4O3)C51CCC5',
]


def clean_smi(s):
    s = s.split(' |')[0] if ' |' in s else s
    m = Chem.MolFromSmiles(s)
    if m is None: return s
    return Chem.MolToSmiles(SaltRemover().StripMol(m))


def morgan_fp(smi, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    print("=" * 70)
    print("SIMPLE CLASSIFIER BASELINE — train on 280 mols, predict 19 candidates")
    print("=" * 70)

    raw = pd.read_csv(RAW)
    zap = raw[raw['target_chembl_id'] == 'CHEMBL2803'].copy()
    agg = zap.groupby('molecule_chembl_id').agg({'smiles':'first','pIC50':'mean'}).reset_index()
    train_smi = agg['smiles'].tolist()
    train_y = agg['pIC50'].values.astype(np.float32)
    print(f"Train: {len(train_smi)} ZAP70 mols, pIC50 [{train_y.min():.2f}, {train_y.max():.2f}], "
          f"mean={train_y.mean():.2f}")

    cand_smi = [clean_smi(s) for s in SMILES_19]
    print(f"Candidates: {len(cand_smi)}")

    X_train = np.array([morgan_fp(s) for s in train_smi])
    X_cand = np.array([morgan_fp(s) for s in cand_smi])
    print(f"FP shapes: train={X_train.shape}, cand={X_cand.shape}")

    # Standardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_cand_s = scaler.transform(X_cand).astype(np.float32)

    methods = {
        'RandomForest': RandomForestRegressor,
        'Ridge': Ridge,
    }
    if HAS_XGB:
        methods['XGBoost'] = XGBRegressor
    print(f"Methods: {list(methods.keys())}")

    all_preds = {}  # method -> (N_SEEDS, 19) predictions
    for method_name, ModelClass in methods.items():
        print(f"\n=== {method_name} ({N_SEEDS} seeds) ===")
        preds = np.zeros((N_SEEDS, 19))
        for s in range(N_SEEDS):
            seed = s * 17 + 5
            np.random.seed(seed)
            # Bootstrap subsample (with replacement, n=280)
            idx = np.random.choice(len(train_smi), size=len(train_smi), replace=True)
            X_boot = X_train_s[idx]
            y_boot = train_y[idx]
            if method_name == 'RandomForest':
                model = ModelClass(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
            elif method_name == 'Ridge':
                model = ModelClass(alpha=1.0, random_state=seed)
            else:  # XGBoost
                model = ModelClass(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   random_state=seed, n_jobs=-1, verbosity=0)
            model.fit(X_boot, y_boot)
            preds[s] = model.predict(X_cand_s)
            if s == 0:
                print(f"  Seed {s+1}: top-5 = {[int(i+1) for i in np.argsort(-preds[s])[:5]]} "
                      f"(pIC50 [{preds[s].min():.2f}, {preds[s].max():.2f}])")
        all_preds[method_name] = preds
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        rank = np.argsort(-mean)
        print(f"  ENSEMBLE top-5: {[int(rank[i]+1) for i in range(5)]} "
              f"(pIC50 [{mean.min():.2f}, {mean.max():.2f}])")

    # === Compare to FiLMDelta 20-seed clean ===
    se = json.load(open(RD / "19_molecules_20seed_uncertainty_clean.json"))
    film_preds = {r['mol_idx']: r['mean_pIC50'] for r in se['mol_results']}

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD: FiLMDelta-pairs vs Simple-classifier baselines (mean ensemble preds)")
    print("=" * 90)
    cols = list(methods.keys())
    print(f"  {'Mol':>4s}  {'FiLMDelta':>10s}  " + "  ".join(f"{m:>10s}" for m in cols) + "  rank_film  rank_other")
    print("-" * 100)

    film_pred_arr = np.array([film_preds[i+1] for i in range(19)])
    rank_film = np.argsort(-film_pred_arr).argsort() + 1
    rank_other = {m: np.argsort(-all_preds[m].mean(axis=0)).argsort() + 1 for m in cols}

    for i in range(19):
        line = f"  {i+1:2d}    {film_pred_arr[i]:8.3f}  "
        for m in cols:
            line += f"  {all_preds[m].mean(axis=0)[i]:8.3f}"
        line += f"      {rank_film[i]:2d}        " + " ".join(f"{rank_other[m][i]:2d}" for m in cols)
        print(line)

    # Spearman correlations vs FiLMDelta
    print(f"\n  Spearman(predictions, FiLMDelta clean 20-seed):")
    for m in cols:
        rho, _ = spearmanr(film_pred_arr, all_preds[m].mean(axis=0))
        print(f"    {m:>15s}: ρ = {rho:.3f}")

    # Save
    out = {
        'n_seeds': N_SEEDS, 'n_train': len(train_smi),
        'film_delta_clean_pIC50': {str(i+1): float(film_pred_arr[i]) for i in range(19)},
    }
    for m in cols:
        out[f'{m}_mean'] = {str(i+1): float(all_preds[m].mean(axis=0)[i]) for i in range(19)}
        out[f'{m}_std'] = {str(i+1): float(all_preds[m].std(axis=0)[i]) for i in range(19)}
        out[f'{m}_rank'] = {str(i+1): int(rank_other[m][i]) for i in range(19)}
        rho, _ = spearmanr(film_pred_arr, all_preds[m].mean(axis=0))
        out[f'{m}_spearman_vs_film'] = float(rho)
    out['film_rank'] = {str(i+1): int(rank_film[i]) for i in range(19)}

    out_path = RD / "19mol_simple_classifier_baseline.json"
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
