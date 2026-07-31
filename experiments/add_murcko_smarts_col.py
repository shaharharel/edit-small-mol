"""Add mol1_murcko_smarts_match column (lenient match used by the training reward)
to all Mol1-anchored scored CSVs. Uses the same SMARTS as the murcko_rl_*.toml:
  O=C(Nc1cncn1)c1cccc2c1CNC2
"""
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

MURCKO_SMARTS = Chem.MolFromSmarts("O=C(Nc1cncn1)c1cccc2c1CNC2")
COHORTS = [
    "mol1RL_v5_seed_mol1_only", "thiq_rl_mol1only",
    "thiq_rl_zap70", "thiq_rl_kinase",
    "thiq_rl_exp2_zap70", "thiq_rl_exp2_kinase",
    "murcko_rl_zap70", "murcko_rl_exp2_zap70",
    "murcko_rl_exp2_kinase",
    "thiq_rl_exp2_mol1only",
]
for c in COHORTS:
    p = Path("data/tier4_scored") / f"{c}_scored.csv"
    if not p.exists():
        print(f"SKIP {c}: missing")
        continue
    df = pd.read_csv(p)
    if "mol1_murcko_smarts_match" in df.columns:
        print(f"SKIP {c}: already has col")
        continue
    matches = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(str(s))
        matches.append(bool(m and m.HasSubstructMatch(MURCKO_SMARTS)))
    df["mol1_murcko_smarts_match"] = matches
    pct = sum(matches) / len(matches) * 100
    df.to_csv(p, index=False)
    strict = df["mol1_murcko_match"].mean() * 100 if "mol1_murcko_match" in df.columns else None
    print(f"  {c}: N={len(df):,}, strict_murcko={strict:.2f}%  smarts_murcko=**{pct:.2f}%**")
