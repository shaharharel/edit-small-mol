#!/usr/bin/env python3
"""COValid Ridge regression baseline.

For each target, train Ridge on (Morgan FP → pIC50) using ChEMBL actives
for that target, score COValid actives + decoys, compute adj LogAUC.

This is the ABSOLUTE-prediction-without-pairs baseline. If FiLMDelta-pairs
doesn't beat it on COValid, the pairs framework doesn't help here.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
PIC50_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
COVALID = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_004.xlsx"
OUT = PROJECT_ROOT / "results" / "covalid" / "covalid_ridge_baseline.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "BMX":   "CHEMBL2581", "BTK":   "CHEMBL5251", "FGFR1": "CHEMBL3650",
    "JAK3":  "CHEMBL2148", "KRAS":  "CHEMBL2189121", "EGFR":  "CHEMBL203",
    "FGFR4_477": "CHEMBL3973", "FGFR4_552": "CHEMBL3973",
}


def morgan_fp(smi, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(n, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n)
    a = np.zeros(n, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, a); return a


def adj_log_auc(labels, scores, lam=10):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    n_act = int(y.sum()); n_dec = len(y) - n_act
    if n_act == 0 or n_dec == 0: return float('nan'), float('nan')
    cum_act = np.cumsum(y) / n_act
    cum_dec = np.cumsum(1 - y) / n_dec
    eps = 1e-12
    mask = cum_dec >= 1 / lam
    if not mask.any(): return 0.0, 0.0
    log_x = np.log10(np.clip(cum_dec[mask], eps, 1.0))
    auc = float(np.trapz(cum_act[mask], log_x))
    raw = float(np.trapz(cum_act, cum_dec))
    return (auc / np.log10(lam)) * 100, raw


def main():
    pic50 = pd.read_csv(PIC50_FILE)
    xl = pd.ExcelFile(COVALID)
    print(f"Ridge baseline on COValid")
    results = {}
    for label, chembl in TARGETS.items():
        print(f"\n=== {label} ({chembl}) ===")
        # Train data
        tgt = pic50[pic50.target_chembl_id == chembl].copy()
        if len(tgt) < 30:
            print(f"    <30 anchors, skipping")
            continue
        tgt = tgt.groupby('molecule_chembl_id').agg({'smiles': 'first', 'pIC50': 'mean'}).reset_index()
        X = np.array([morgan_fp(s) for s in tgt['smiles']])
        y = tgt['pIC50'].values
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X).astype(np.float32)
        # Ensemble: 5 seeds × Ridge with different alpha
        preds_all = []
        # COValid protomers
        if f"{label}_actives" not in xl.sheet_names: print("    skip — missing sheet"); continue
        actives = pd.read_excel(COVALID, sheet_name=f"{label}_actives")
        decoys = pd.read_excel(COVALID, sheet_name=f"{label}_decoys")
        sm_a = next(c for c in actives.columns if 'smiles' in c.lower())
        sm_d = next(c for c in decoys.columns if 'smiles' in c.lower())
        protos = pd.concat([
            actives.assign(label=1)[[sm_a, 'label']].rename(columns={sm_a: 'smi'}),
            decoys.assign(label=0)[[sm_d, 'label']].rename(columns={sm_d: 'smi'}),
        ], ignore_index=True)
        X_p = np.array([morgan_fp(s) for s in protos.smi])
        X_p_s = scaler.transform(X_p).astype(np.float32)
        for s in range(5):
            np.random.seed(s)
            ridge = Ridge(alpha=1.0, random_state=s)
            ridge.fit(X_s, y)
            preds_all.append(ridge.predict(X_p_s))
        preds = np.mean(preds_all, axis=0)
        adj_pct, raw = adj_log_auc(protos.label.values, preds)
        n_a = int((protos.label == 1).sum()); n_d = int((protos.label == 0).sum())
        print(f"    n_train={len(tgt)}  n_act={n_a}  n_dec={n_d}  adj_LogAUC={adj_pct:.1f}%  raw_AUC={raw:.3f}")
        results[label] = {
            'chembl_id': chembl, 'n_anchors': len(tgt),
            'n_actives': n_a, 'n_decoys': n_d,
            'adj_logAUC_pct': float(adj_pct), 'raw_AUC': float(raw),
        }
    out = {
        'method': 'Ridge regression on Morgan FP → pIC50 (5-seed ensemble), trained per-target',
        'per_target': results,
        'avg_adj_logAUC_pct': float(np.mean([v['adj_logAUC_pct'] for v in results.values()])) if results else 0,
        'london_avg_adj_logAUC_pct': 71.8,
    }
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\nAvg adj_LogAUC: {out['avg_adj_logAUC_pct']:.1f}% (London: 71.8%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
