"""Add PubTc panel columns to Mol1-anchored cohort scored CSVs.

Computes max_pubTc, mean_pubTc, median_pubTc, top10_mean_pubTc, closest_lead
against the 300-lead v3 panel for each cohort.

Run: python experiments/add_pubtc_to_mol1_cohorts.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Load 300-lead panel
from experiments.pubtc_panel_v3 import PANEL_V3  # noqa

panel_smiles = [(p["drug_name"], p["smiles"]) for p in PANEL_V3]
print(f"Loaded {len(panel_smiles)} leads")

# Build Morgan FPs for panel
panel_fps = []
panel_names = []
for name, smi in panel_smiles:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        print(f"  WARN: panel parse fail {name}")
        continue
    panel_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    panel_names.append(name)
print(f"Built {len(panel_fps)} panel FPs")

# Compute Tc for one mol vs all leads
def compute_pubtc(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return {"max_pubTc": np.nan, "mean_pubTc": np.nan, "median_pubTc": np.nan,
                "top10_mean_pubTc": np.nan, "closest_lead": ""}
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    tcs = np.array(DataStructs.BulkTanimotoSimilarity(fp, panel_fps))
    idx_max = int(np.argmax(tcs))
    top10 = np.sort(tcs)[-10:]
    return {
        "max_pubTc": round(float(tcs.max()), 4),
        "mean_pubTc": round(float(tcs.mean()), 4),
        "median_pubTc": round(float(np.median(tcs)), 4),
        "top10_mean_pubTc": round(float(top10.mean()), 4),
        "closest_lead": panel_names[idx_max],
    }


def main():
    cohorts = [
        "mol1RL_v5_seed_mol1_only",
        "thiq_rl_zap70",
        "thiq_rl_kinase",
        "thiq_rl_exp2_zap70",
        "thiq_rl_exp2_kinase",
        "murcko_rl_zap70",
        "murcko_rl_exp2_zap70",
        "murcko_rl_exp2_kinase",
        "thiq_rl_mol1only",
        "thiq_rl_exp2_mol1only",
    ]
    for c in cohorts:
        csv = PROJECT / "data/tier4_scored" / f"{c}_scored.csv"
        if not csv.exists():
            print(f"SKIP {c}: missing")
            continue
        df = pd.read_csv(csv)
        if "max_pubTc" in df.columns and df["max_pubTc"].notna().mean() > 0.9:
            print(f"SKIP {c}: already enriched")
            continue
        print(f">>> {c}: {len(df):,} mols")
        # Parallel
        with mp.Pool(8) as pool:
            results = pool.map(compute_pubtc, df["smiles"].tolist(), chunksize=200)
        df["max_pubTc"] = [r["max_pubTc"] for r in results]
        df["mean_pubTc"] = [r["mean_pubTc"] for r in results]
        df["median_pubTc"] = [r["median_pubTc"] for r in results]
        df["top10_mean_pubTc"] = [r["top10_mean_pubTc"] for r in results]
        df["closest_lead"] = [r["closest_lead"] for r in results]
        df.to_csv(csv, index=False)
        print(f"  med max_pubTc: {df['max_pubTc'].median():.3f}, "
              f"max: {df['max_pubTc'].max():.3f}, "
              f"closest_lead mode: {df['closest_lead'].mode()[0] if len(df) else 'n/a'}")


if __name__ == "__main__":
    main()
