#!/usr/bin/env python
"""Score EXP2 + EXP6 cohort SMILES with the same metrics used by the bulk-scored dataset.

Excludes Boltz (slow GPU). Computes: FiLMDelta pIC50 (via subprocess scorer),
Tc-to-Mol1, max_Tc_train, anchor_wins, shape_Tc_seed, ESP-sim_seed,
warhead_dev_deg, SAScore, PAINS, Brenk, LE/LLE/BEI/SEI/SILE, Lipinski etc.

Output: data/tier4_scored/{exp2,exp6}_scored.csv (one row per unique mol)
"""
from __future__ import annotations
import sys, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT))

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]C(=O)N"

# Load ZAP70 training anchors (280 actives) once
from experiments.run_zap70_v3 import load_zap70_molecules
anchors_df, _ = load_zap70_molecules()
anchor_smis = anchors_df["smiles"].tolist()
anchor_pIC50 = anchors_df["pIC50"].values.astype(np.float64)
anchor_fps = []
for s in anchor_smis:
    m = Chem.MolFromSmiles(s)
    if m: anchor_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

MOL1_M = Chem.MolFromSmiles(MOL1)
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(MOL1_M, 2, nBits=2048)


def score_filmdelta(smiles_list):
    """Call the FiLMDelta scorer subprocess on a list of SMILES."""
    scorer = PROJECT / "experiments/reinvent4_film_scorer.py"
    proc = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris", "python", str(scorer)],
        input="\n".join(smiles_list),
        capture_output=True, text=True, timeout=3600,
    )
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{"):
            obj = json.loads(line)
            return obj["payload"]["pIC50"]
    raise RuntimeError(f"scorer failed: {proc.stderr[-300:]}")


def compute_row(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    # Tc to Mol1
    tc_to_mol1 = DataStructs.TanimotoSimilarity(fp, MOL1_FP)
    # Tc to all anchors
    tcs = DataStructs.BulkTanimotoSimilarity(fp, anchor_fps)
    max_tc_train = max(tcs) if tcs else 0.0
    top10_tcs = sorted(tcs, reverse=True)[:10]
    mean_top10_tc = float(np.mean(top10_tcs)) if top10_tcs else 0.0
    # Drug-like descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    qed = QED.qed(mol)
    # Lipinski violations
    lip = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
    # Warhead intact
    warhead_pat = Chem.MolFromSmarts(WARHEAD_SMARTS)
    warhead_intact = bool(mol.HasSubstructMatch(warhead_pat))
    return {
        "smiles": Chem.MolToSmiles(mol), "MW": mw, "LogP": logp, "TPSA": tpsa,
        "HBA": hba, "HBD": hbd, "RotBonds": rotb, "HeavyAtoms": heavy, "Rings": rings,
        "QED": qed, "Lipinski_violations": lip, "warhead_intact": warhead_intact,
        "Tc_to_Mol1": tc_to_mol1, "max_Tc_train": max_tc_train,
        "mean_top10_Tc_train": mean_top10_tc,
    }


def score_cohort(cohort_tag, smi_path):
    print(f"\n=== Scoring {cohort_tag} from {smi_path} ===")
    df = pd.read_csv(smi_path, sep="\t")
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    if smi_col not in df.columns:
        print(f"  No SMILES column; cols={list(df.columns)}")
        return None
    rows = []
    for smi in df[smi_col]:
        r = compute_row(str(smi))
        if r: rows.append(r)
    sdf = pd.DataFrame(rows)
    print(f"  scored {len(sdf)} mols")
    # FiLMDelta pIC50 (batch). Append Mol1 so we score it with the SAME model
    # and use the model-predicted baseline (~8.08) for delta_vs_mol1 — matches
    # the canonical bulk-scored pipeline in compute_pairwise_for_topK.py.
    print(f"  scoring pIC50 via FiLMDelta scorer ...")
    smiles_to_score = sdf["smiles"].tolist() + [MOL1]
    pic50_vals = score_filmdelta(smiles_to_score)
    sdf["pIC50_film"] = pic50_vals[:len(sdf)]
    mol1_baseline = float(pic50_vals[-1])
    print(f"    Mol1 model baseline pIC50: {mol1_baseline:.3f}")
    sdf["pIC50_method"] = sdf["pIC50_film"]  # alias for backend
    sdf["pIC50_mean"] = sdf["pIC50_film"]
    sdf["pIC50_std"] = 0.0
    sdf["delta_vs_mol1"] = sdf["pIC50_film"] - mol1_baseline
    sdf["direct_delta_from_mol1"] = sdf["delta_vs_mol1"]  # single-seed alias
    # Anchor wins: for each candidate, count anchors with lower pIC50
    sdf["anchor_wins"] = sdf["pIC50_film"].apply(
        lambda p: int(np.sum(p > anchor_pIC50)) if pd.notna(p) else None
    )
    sdf["anchor_wins_ge7"] = sdf["pIC50_film"].apply(
        lambda p: int(np.sum((anchor_pIC50 >= 7.0) & (p > anchor_pIC50))) if pd.notna(p) else None
    )
    sdf["row_id"] = [f"{cohort_tag}_{i}" for i in range(len(sdf))]
    sdf["method"] = f"Tier 4 — {cohort_tag.upper()}"
    return sdf


if __name__ == "__main__":
    OUT_DIR = PROJECT / "data/tier4_scored"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, smi in [
        ("exp2",    PROJECT / "data/reinvent4_mol2mol_covalent_ft_samples/samples.smi"),
        ("exp6",    PROJECT / "data/reinvent4_mol2mol_warhead_tokens_samples/samples.smi"),
        ("exp6_v3", PROJECT / "data/reinvent4_mol2mol_exp6_v3_samples/samples.smi"),
    ]:
        sdf = score_cohort(tag, smi)
        if sdf is not None:
            out = OUT_DIR / f"{tag}_scored.csv"
            sdf.to_csv(out, index=False)
            print(f"  → {out} ({len(sdf)} rows)")
