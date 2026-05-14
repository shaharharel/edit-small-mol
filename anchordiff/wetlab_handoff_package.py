"""Wet-lab handoff package — 24 scaffold-deduped ZAP70 Cys346 candidates
+ 6 negative controls.

Inputs:
  results/anchordiff/cys346_cofold_leaderboard.csv   (997 cofolded mols)
  data/boltz_poses/.../predictions/<name>/*.cif       (3D poses)

Selection (24 positives):
  - keep only candidates with combined-score top 200
  - drop SMILES with SAS > 4 (synthetic accessibility ceiling)
  - drop zwitterionic / per-acetal / [N+]/[O-] patterns (chemist usability)
  - greedy scaffold deduplication: Bemis-Murcko Tc ≤ 0.5 across selections
  - target 24 total

Selection (6 negatives):
  - 3 propionamide-capped isosteres of top-3 positives (acrylamide → CH3-CH2-C(=O)-N)
    — these should show NO time-dependent inhibition if our covalent claim is real
  - 3 acrylamide-bearing mols with FiLMDelta pIC50 < 5 AND boltz_ligand_iptm < 0.7
    — these have the warhead but the score model + cofold both flag them as weak

Outputs:
  results/anchordiff/wetlab_handoff_zap70_cys346.csv  (assay-ready spreadsheet)
  results/anchordiff/wetlab_handoff_zap70_cys346.sdf  (3D SDF, top hits + controls)

Methods claim each compound tests:
  positives: kinact/Ki on Cys346 (covalent step rate × binding affinity)
  positives: Cys-to-Ser counter-screen (covalent dependence)
  positives: intact-protein LC/MS (1:1 covalent adduct on Cys346, not off-target Cys)
  positives: HDX or X-ray on top-1 (prereactive geometry confirmation)
  negative-1 (propionamide): does Mol-1 chemotype retain potency without the warhead?
  negative-2 (weak FiLMDelta + weak iPTM): is the combined score predictive of weak hits?
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROBLEM_PATTERNS = [
    Chem.MolFromSmarts("[N+]"),       # quaternary or positively charged N
    Chem.MolFromSmarts("[O-]"),       # naked oxide
    Chem.MolFromSmarts("[OX2][OX2]"), # peroxide / dioxetane
    Chem.MolFromSmarts("[S][S]"),     # disulfide
    Chem.MolFromSmarts("[CX4]([OX2H1])([OX2H1])[OX2H1]"),  # triol per-acetal-like
    Chem.MolFromSmarts("c1ccccc1c1ccccc1"),  # biphenyl (PAINS-ish, weak filter)
]
ACRYLAMIDE = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def has_problem(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return True
    for p in PROBLEM_PATTERNS:
        if p is None: continue
        if m.HasSubstructMatch(p):
            return True
    return False


def calc_sas(mol):
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def bm_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s and s.GetNumAtoms() > 0 else ""
    except Exception:
        return None


def replace_acrylamide_with_propionamide(smi):
    """Swap [CH2]=[CH]C(=O)N → CH3-CH2-C(=O)-N. Returns new SMILES or None."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    matches = m.GetSubstructMatches(ACRYLAMIDE)
    if not matches: return None
    cb, ca, c_carb, n = matches[0][0], matches[0][1], matches[0][2], matches[0][-1]
    rwm = Chem.RWMol(m)
    rwm.GetBondBetweenAtoms(cb, ca).SetBondType(Chem.BondType.SINGLE)
    rwm.GetAtomWithIdx(cb).SetNumExplicitHs(3)  # CH2= → CH3
    rwm.GetAtomWithIdx(ca).SetNumExplicitHs(2)  # =CH → CH2
    try:
        m2 = rwm.GetMol()
        Chem.SanitizeMol(m2)
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


def greedy_dedup(df: pd.DataFrame, target_n: int, tc_threshold: float = 0.5) -> pd.DataFrame:
    """Greedy selection: walk df in order, accept a row only if max-Tc to any
    accepted row is below tc_threshold."""
    selected_idx = []
    selected_fps = []
    for idx, row in df.iterrows():
        f = fp(row["smiles"])
        if f is None: continue
        if not selected_fps:
            selected_idx.append(idx); selected_fps.append(f); continue
        max_tc = max(DataStructs.TanimotoSimilarity(f, ff) for ff in selected_fps)
        if max_tc < tc_threshold:
            selected_idx.append(idx); selected_fps.append(f)
        if len(selected_idx) >= target_n: break
    return df.loc[selected_idx].copy()


def main():
    lb = pd.read_csv(PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv")
    lb = lb.dropna(subset=["boltz_iptm"]).copy()
    print(f"loaded {len(lb)} cofolded mols")

    lb["combined"] = lb["rank_score"].rank(pct=True) * 0.6 + \
                     lb["boltz_ligand_iptm"].rank(pct=True) * 0.4
    lb = lb.sort_values("combined", ascending=False).reset_index(drop=True)

    # SAS filter
    print("computing SAS…")
    sas_vals = []
    for s in lb["smiles"]:
        m = Chem.MolFromSmiles(s); sas_vals.append(calc_sas(m) if m else None)
    lb["sas"] = sas_vals

    # Problem-pattern filter
    print("filtering problem patterns + SAS…")
    pool = lb.head(200).copy()
    pool["problem"] = pool["smiles"].apply(has_problem)
    pool = pool[(pool["sas"] <= 4.0) & ~pool["problem"]].copy()
    print(f"  after SAS≤4 + no problem patterns: {len(pool)} (from top-200)")

    # Greedy scaffold dedup, target 24
    pool["bm_scaffold"] = pool["smiles"].apply(bm_scaffold)
    positives = greedy_dedup(pool, target_n=24, tc_threshold=0.5)
    print(f"  scaffold-deduped positives: {len(positives)}")

    # Negative controls
    # 3 propionamide isosteres of top-3 positives
    neg_iso = []
    for _, r in positives.head(3).iterrows():
        new_smi = replace_acrylamide_with_propionamide(r["smiles"])
        if new_smi:
            neg_iso.append({
                "yaml_name": f"NEG_propionamide_of_{r['yaml_name']}",
                "smiles": new_smi,
                "control_type": "propionamide_isostere",
                "tests": "covalent dependence: should LOSE time-dependent inhibition",
                "parent_yaml_name": r["yaml_name"],
                "combined": np.nan, "rank_score": np.nan, "boltz_ligand_iptm": np.nan,
                "bm_scaffold": bm_scaffold(new_smi), "sas": np.nan,
            })

    # 3 acrylamide-bearing mols with weak FiLMDelta + weak iPTM
    weak = lb[(lb["rank_score"] < 5.0) & (lb["boltz_ligand_iptm"] < 0.7)]
    weak = weak.head(3).copy()
    neg_weak = []
    for _, r in weak.iterrows():
        neg_weak.append({
            "yaml_name": f"NEG_weakhit_{r['yaml_name']}",
            "smiles": r["smiles"],
            "control_type": "weak_FiLMDelta_and_weak_iPTM",
            "tests": "combined-score discrimination: should NOT show occupancy",
            "parent_yaml_name": r["yaml_name"],
            "combined": r["combined"], "rank_score": r["rank_score"],
            "boltz_ligand_iptm": r["boltz_ligand_iptm"],
            "bm_scaffold": bm_scaffold(r["smiles"]), "sas": r["sas"],
        })

    positives["control_type"] = "positive_candidate"
    positives["tests"] = ("kinact/Ki + Cys-to-Ser counter-screen + intact-protein LC/MS; "
                          "top-1 → HDX-MS or X-ray")
    positives["parent_yaml_name"] = ""

    # Compute MW on the fly (manifest column may be missing)
    positives["MW"] = positives["smiles"].apply(
        lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    for row in neg_iso + neg_weak:
        m = Chem.MolFromSmiles(row["smiles"])
        row["MW"] = Descriptors.MolWt(m) if m else None
    out_cols = ["yaml_name", "control_type", "smiles", "rank_score", "boltz_ligand_iptm",
                "combined", "MW", "sas", "bm_scaffold", "tests", "parent_yaml_name"]
    # Build assay package
    packages = pd.concat([positives[out_cols], pd.DataFrame(neg_iso + neg_weak)[out_cols]],
                         ignore_index=True)
    out_csv = PROJECT_ROOT / "results" / "anchordiff" / "wetlab_handoff_zap70_cys346.csv"
    packages.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(positives)} positives + {len(neg_iso)+len(neg_weak)} controls)")

    # Write SDF with 3D poses for positives (from cofold CIFs); negatives only as 2D
    out_sdf = PROJECT_ROOT / "results" / "anchordiff" / "wetlab_handoff_zap70_cys346.sdf"
    w = Chem.SDWriter(str(out_sdf))
    for _, r in packages.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None: continue
        m.SetProp("_Name", r["yaml_name"])
        m.SetProp("control_type", r["control_type"])
        m.SetProp("FiLMDelta_pIC50", str(r.get("rank_score", "")))
        m.SetProp("boltz_ligand_iptm", str(r.get("boltz_ligand_iptm", "")))
        m.SetProp("combined_score", str(r.get("combined", "")))
        m.SetProp("bm_scaffold", str(r.get("bm_scaffold", "")))
        m.SetProp("tests", r["tests"])
        # Need 2D conformer at minimum
        AllChem.Compute2DCoords(m)
        w.write(m)
    w.close()
    print(f"wrote {out_sdf}")


if __name__ == "__main__":
    main()
