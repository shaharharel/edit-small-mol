#!/usr/bin/env python3
"""Distributional audit across 6 emitted cohorts + reference sets.

Local + CPU only. No training. No GPU.

Cohorts audited:
  1. scheme_A  (paper_pair_training/scheme_A/samples_scheme_A_10k.csv)
  2. scheme_B  (paper_pair_training/scheme_B/samples_scheme_B_10k.csv)
  3. scheme_C  (paper_pair_training/scheme_C/eval/samples_scheme_B_10k.csv — misnamed, is scheme C)
  4. infonce   (paper_pair_training/infonce/samples_infonce_10k.csv)
  5. v2cond    (paper_dap_repro/samples_v2cond_QA_10k_scored.csv — canonical baseline)

Reference sets:
  - FDA covalent drugs (curated ~16 mols)
  - ZAP70 PDB co-crystals + boltz_zap70 (train pool proxy)
  - Random ChEMBL kinase actives (from lingo3dmol_kinase_FT/kinase_covindb_complexes.csv, ~200)

Panel: RDKit chemistry + PAINS/Brenk + geometric proxies + xTB label join + Boltz label join.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# SA score
from rdkit.Chem import RDConfig
SA_PATH = Path(RDConfig.RDContribDir) / "SA_Score"
sys.path.insert(0, str(SA_PATH))
import sascorer

from scipy import stats as sstats

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data" / "paper_pair_training" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "scheme_A": ROOT / "data/paper_pair_training/scheme_A/samples_scheme_A_10k.csv",
    "scheme_B": ROOT / "data/paper_pair_training/scheme_B/samples_scheme_B_10k.csv",
    "scheme_C": ROOT / "data/paper_pair_training/scheme_C/eval/samples_scheme_B_10k.csv",
    "infonce":  ROOT / "data/paper_pair_training/infonce/samples_infonce_10k.csv",
    "v2cond":   ROOT / "data/paper_dap_repro/samples_v2cond_QA_10k_scored.csv",
}
# Note: 6th cohort (ablation3 / B-max) not yet emitted locally.
MISSING_COHORTS = ["ablation3_data_only", "B_max"]

M1A_NPZ = ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"
# ZAP70-specific Boltz co-fold energies (has SMILES)
BOLTZ_ZAP70_CSV = ROOT / "data/boltz_poses/zap70_cys346_energy_scores_v2.csv"
# ZAP70 warhead reactivity (xTB-labeled) — already computed for many mols
BOLTZ_WH_REACTIVITY_CSV = ROOT / "data/boltz_poses/zap70_cys346_warhead_reactivity.csv"
# generic cross-target Boltz metrics (kept for optional off-target join)
BOLTZ_METRICS_CSV = ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"
XTB_LABELED_CSV = ROOT / "data/tier4_scored/F4_boltz_pool_scored.csv"
KINASE_REF_CSV = ROOT / "data/lingo3dmol_kinase_FT/kinase_covindb_complexes.csv"

FDA_COVALENT_SMILES = {
    "ibrutinib":     "C=CC(=O)N1CCC[C@H]1CN2C3=CC=CC=C3C4=C2C(=NC=N4)N5CCC(CC5)OC6=CC=CC=C6",
    "acalabrutinib": "CC#CC(=O)N1CCC[C@H]1C2=NC(=C3N2C=CN=C3N)C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5",
    "afatinib":      "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
    "osimertinib":   "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    "neratinib":     "CCOC1=CC2=C(C=C1NC(=O)/C=C/CN(C)C)N=CN=C2NC3=CC(=C(C=C3)OCC4=CC=CC=N4)Cl",
    "dacomitinib":   "CN1CCC(CC1)/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "pyrotinib":     "CN1CCC(CC1)COc1cc2c(Nc3ccc(OCc4ccccn4)c(Cl)c3)ncnc2cc1NC(=O)/C=C/CN1CCCCC1",
    "olmutinib":     "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1ncc(Cl)c(Oc2cccc(NC(=O)C=C)c2)n1",
    "sotorasib":     "CC1=C(c2c(F)cccc2F)C(=O)N(c2cc(N3CCN(C(=O)C=C)C[C@@H]3C)nc(N)c2C#N)N1c1c(C)cnc(C(C)C)c1C",
    "adagrasib":     "Cn1c(=O)c2cn(-c3c(Cl)cc(N4CCC(C#N)(C(=O)C=C)C4)cc3F)c3cccnc3c2n(C)c1=O",
    "evobrutinib":   "C=CC(=O)N1CCC(Oc2ncnc3[nH]c(-c4ccc(OCc5cccc(N6CCOCC6)c5)cc4)cc23)CC1",
    "tolebrutinib":  "CCOc1ccc(-c2cc3c(N4C[C@@H](N(C)C(=O)C=C)C4)ncnc3[nH]2)cc1",
    "orelabrutinib": "C=CC(=O)N1CCC[C@H]1CN2C3=C(C(=NC=N3)N)C(=C2)C4=CC=C(C=C4)OC5=CC=CC=C5",
    "zanubrutinib":  "C=CC(=O)N1CCC[C@H]1C1=NC(=NC2=C1C=CN2C1=CC=C(C(=O)N)C=C1)N1CCOCC1",
    "spebrutinib":   "C=CC(=O)Nc1cccc(Nc2ncnc3ccc(Oc4ccccc4)cc23)c1",
    "poseltinib":    "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1ncc(Cl)c(Nc2ccccc2P(C)(C)=O)n1",
}

MOL1_SMI = "CC(C)n1cnc(NC(=O)c2cccc3c2CN(C(=O)C=C)C3)c1"  # ZAP70 Mol1 anchor

ACRYL_SMARTS = "[CX3;H2]=[CX3;H1][CX3](=[OX1])[NX3]"  # acrylamide
MICHAEL_SMARTS_LIST = [
    ("acrylamide", "[CX3;H2]=[CX3;H1][CX3](=[OX1])[NX3]"),
    ("acrylate",   "[CX3;H2]=[CX3;H1][CX3](=[OX1])[OX2]"),
    ("vinylsulfone", "[CX3;H2]=[CX3;H1][SX4](=[OX1])(=[OX1])"),
    ("propiolamide", "[CX2]#[CX2][CX3](=[OX1])[NX3]"),
    ("chloroacetamide", "[Cl][CX4;H2][CX3](=[OX1])[NX3]"),
    ("cyanoacrylamide", "N#C/C=C/C(=O)N"),
]

# Hammett-σ_p for β-substituents (values from Hansch-Leo tables, common ones)
HAMMETT_SIGMA = {
    "H": 0.00, "F": 0.06, "Cl": 0.23, "Br": 0.23, "CN": 0.66,
    "CF3": 0.54, "CH3": -0.17, "OCH3": -0.27, "OMe": -0.27,
    "COOMe": 0.45, "COOEt": 0.45, "NO2": 0.78, "NH2": -0.66,
    "OH": -0.37, "COOH": 0.45, "SO2Me": 0.72,
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
_pains_catalog = None
_brenk_catalog = None


def get_pains() -> FilterCatalog:
    global _pains_catalog
    if _pains_catalog is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog(p)
    return _pains_catalog


def get_brenk() -> FilterCatalog:
    global _brenk_catalog
    if _brenk_catalog is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog(p)
    return _brenk_catalog


def canon(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi:
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def largest_frag(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    biggest = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(biggest)


def has_smarts(smi: str, patt: Chem.Mol) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return m.HasSubstructMatch(patt)


def morgan_fp(smi: str, n_bits: int = 2048, radius: int = 2):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def compute_row(smi: str, patts: Dict) -> Optional[Dict]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        canonical = Chem.MolToSmiles(m)
        # largest frag
        frags = Chem.GetMolFrags(m, asMols=True)
        biggest = max(frags, key=lambda x: x.GetNumHeavyAtoms()) if frags else m
        # RDKit descriptors
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        tpsa = Descriptors.TPSA(m)
        hba = Lipinski.NumHAcceptors(m)
        hbd = Lipinski.NumHDonors(m)
        rotb = Lipinski.NumRotatableBonds(m)
        qed = Descriptors.qed(m)
        try:
            sa = sascorer.calculateScore(m)
        except Exception:
            sa = float("nan")
        fsp3 = rdMolDescriptors.CalcFractionCSP3(m)
        nrings = rdMolDescriptors.CalcNumRings(m)
        naromatic = rdMolDescriptors.CalcNumAromaticRings(m)
        # Lipinski violations
        lipv = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
        # warhead retention on largest fragment
        acryl_largest = biggest.HasSubstructMatch(patts["acryl"])
        acryl_any = m.HasSubstructMatch(patts["acryl"])
        # michael-acceptor families
        michael_hits = {}
        for name, patt in patts["michael"].items():
            michael_hits[f"has_{name}"] = m.HasSubstructMatch(patt)
        michael_any = any(michael_hits.values())
        # filters
        pains = get_pains().HasMatch(m)
        brenk = get_brenk().HasMatch(m)
        row = {
            "canon_smi": canonical,
            "MW": mw, "LogP": logp, "TPSA": tpsa, "HBA": hba, "HBD": hbd,
            "RotBonds": rotb, "QED": qed, "SA": sa, "fsp3": fsp3,
            "NumRings": nrings, "NumAromRings": naromatic,
            "Lipinski_violations": lipv,
            "acryl_largest": bool(acryl_largest),
            "acryl_any": bool(acryl_any),
            "michael_any": bool(michael_any),
            "PAINS": bool(pains),
            "Brenk": bool(brenk),
        }
        row.update(michael_hits)
        return row
    except Exception:
        return None


def build_patts() -> Dict:
    patts = {
        "acryl": Chem.MolFromSmarts(ACRYL_SMARTS),
        "michael": {name: Chem.MolFromSmarts(sm) for name, sm in MICHAEL_SMARTS_LIST},
    }
    return patts


# ----------------------------------------------------------------------------
# Similarity
# ----------------------------------------------------------------------------
def build_fp_list(smiles: List[str], n_bits: int = 2048):
    fps = []
    for s in smiles:
        fp = morgan_fp(s, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
    return fps


def max_tc(fp_query, ref_fps) -> float:
    if not ref_fps or fp_query is None:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, ref_fps)
    return float(max(sims))


def median_tc(fp_query, ref_fps) -> float:
    if not ref_fps or fp_query is None:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, ref_fps)
    return float(np.median(sims))


# ----------------------------------------------------------------------------
# Geometric proxy
# ----------------------------------------------------------------------------
def planar_dihedral(smi: str) -> float:
    """Dihedral around the vinyl-amide plane, computed on 1 ETKDG conformer.
    Returns absolute deg (0 = planar)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return float("nan")
    patt = Chem.MolFromSmarts("[CH2]=[CH]-[CX3](=[OX1])-[NX3]")
    match = m.GetSubstructMatch(patt)
    if not match:
        return float("nan")
    try:
        mh = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mh, params) != 0:
            return float("nan")
        mh_match = mh.GetSubstructMatch(patt)
        if not mh_match:
            return float("nan")
        from rdkit.Chem import rdMolTransforms
        # C=C-C(=O)-N dihedral. Absolute deviation from planar (0 or 180).
        dih = rdMolTransforms.GetDihedralDeg(
            mh.GetConformer(), mh_match[0], mh_match[1], mh_match[2], mh_match[4])
        # planarity: 0 or 180 = planar; measure absolute deviation
        d = abs(dih)
        # wrap to distance from planar (0 or 180)
        dev = min(d, abs(180.0 - d))
        return float(dev)
    except Exception:
        return float("nan")


def hammett_sigma_from_smi(smi: str) -> float:
    """Sum σ_p over recognizable beta-substituents on the vinyl-C.
    Returns NaN if no acrylamide pattern found."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return float("nan")
    # Find acrylamide β-C (the terminal CH2)
    patt = Chem.MolFromSmarts("[CH2:1]=[CH:2]-[CX3](=[OX1])-[NX3]")
    match = m.GetSubstructMatch(patt)
    if not match:
        return 0.0
    beta_c_idx = match[0]  # =CH2
    beta_c = m.GetAtomWithIdx(beta_c_idx)
    # substituents on β-C (excluding the vinyl neighbor)
    vinyl_neighbor = match[1]
    sigma_sum = 0.0
    # For H2C= where both H's are on β-C, no non-H subs; sigma_sum = 0 (parent acrylamide)
    for nbr in beta_c.GetNeighbors():
        if nbr.GetIdx() == vinyl_neighbor:
            continue
        # skip H atoms (implicit)
        sym = nbr.GetSymbol()
        key = sym
        sigma_sum += HAMMETT_SIGMA.get(key, 0.0)
    return sigma_sum


# ----------------------------------------------------------------------------
# Cohort loader
# ----------------------------------------------------------------------------
def load_cohort(name: str, path: Path, n_max: int = 3000) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SMILES" not in df.columns:
        raise ValueError(f"{path} has no SMILES column")
    df = df.head(n_max).copy()
    df["cohort"] = name
    return df


def load_reference_fda() -> pd.DataFrame:
    rows = []
    for name, smi in FDA_COVALENT_SMILES.items():
        c = canon(smi)
        if c is None:
            print(f"[FDA] dropping {name}: invalid SMILES")
            continue
        rows.append({"name": name, "SMILES": c})
    df = pd.DataFrame(rows)
    df["cohort"] = "ref_FDA"
    return df


def load_reference_zap70() -> pd.DataFrame:
    d = np.load(M1A_NPZ, allow_pickle=True)
    sources = d["sources"]
    struct_ids = d["struct_ids"]
    smiles = d["smiles"]
    # Direct PDB co-crystals + boltz-cofold ZAP70 (real ZAP70 ligand set)
    mask_pdb = np.array([("zap" in str(s).lower() or "ZAP" in str(s)) for s in struct_ids])
    mask_boltz = sources == "boltz_zap70"
    mask = mask_pdb | mask_boltz
    smi_list = smiles[mask]
    canon_list = []
    for s in smi_list:
        c = canon(str(s))
        if c is not None:
            canon_list.append(c)
    canon_list = sorted(set(canon_list))
    df = pd.DataFrame({"SMILES": canon_list})
    df["cohort"] = "ref_ZAP70"
    return df


def load_reference_kinase(n_max: int = 500) -> pd.DataFrame:
    if not KINASE_REF_CSV.exists():
        return pd.DataFrame(columns=["SMILES", "cohort"])
    df = pd.read_csv(KINASE_REF_CSV)
    if "SMILES" not in df.columns:
        return pd.DataFrame(columns=["SMILES", "cohort"])
    smi_list = df["SMILES"].dropna().astype(str).tolist()
    canon_list = []
    for s in smi_list:
        c = canon(s)
        if c is not None:
            canon_list.append(c)
    canon_list = sorted(set(canon_list))
    if len(canon_list) > n_max:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(canon_list), size=n_max, replace=False)
        canon_list = [canon_list[i] for i in idx]
    out = pd.DataFrame({"SMILES": canon_list})
    out["cohort"] = "ref_kinase"
    return out


# ----------------------------------------------------------------------------
# Master audit
# ----------------------------------------------------------------------------
def audit_dataframe(df: pd.DataFrame, patts: Dict,
                    train_pool_fps=None,
                    kinase_ref_fps=None,
                    mol1_fp=None,
                    do_planar: bool = True,
                    planar_sample: int = 200,
                    hammett: bool = True) -> pd.DataFrame:
    rows = []
    smiles = df["SMILES"].astype(str).tolist()
    fps = []
    # basic descriptors
    for i, smi in enumerate(smiles):
        rec = compute_row(smi, patts)
        if rec is None:
            continue
        rec["cohort"] = df["cohort"].iloc[i]
        rec["raw_smi"] = smi
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # dedup by canonical
    out = out.drop_duplicates(subset=["canon_smi"]).reset_index(drop=True)

    # fingerprints for similarity
    fps = [morgan_fp(s) for s in out["canon_smi"]]

    # Tc to Mol1
    if mol1_fp is not None:
        out["Tc_to_Mol1"] = [DataStructs.TanimotoSimilarity(fp, mol1_fp) if fp is not None else np.nan for fp in fps]

    # Max Tc to training pool (novelty axis)
    if train_pool_fps:
        out["MaxTc_to_train"] = [max_tc(fp, train_pool_fps) for fp in fps]

    # Median Tc to random kinase actives
    if kinase_ref_fps:
        out["MedTc_to_kinase"] = [median_tc(fp, kinase_ref_fps) for fp in fps]

    # planar dihedral (subsample)
    if do_planar:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(out), size=min(planar_sample, len(out)), replace=False)
        pl = [np.nan] * len(out)
        for i in idx:
            pl[i] = planar_dihedral(out["canon_smi"].iloc[i])
        out["planar_dihedral_deg"] = pl

    # Hammett-σ
    if hammett:
        out["hammett_sigma"] = [hammett_sigma_from_smi(s) for s in out["canon_smi"]]

    return out


# ----------------------------------------------------------------------------
# Label joins
# ----------------------------------------------------------------------------
def join_xtb_labels(master: pd.DataFrame) -> pd.DataFrame:
    if not XTB_LABELED_CSV.exists():
        print("[join_xtb] file missing")
        return master
    xtb = pd.read_csv(XTB_LABELED_CSV, usecols=lambda c: c in [
        "smiles", "omega_eV", "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH"
    ])
    xtb["canon_smi"] = xtb["smiles"].apply(canon)
    xtb = xtb.dropna(subset=["canon_smi"]).drop_duplicates(subset=["canon_smi"])
    xtb = xtb.rename(columns={
        "omega_eV": "xtb_omega_eV",
        "q_Cb": "xtb_q_Cb",
        "fukui_plus_Cb": "xtb_fukui_plus_Cb",
        "pred_log_k2_GSH": "xtb_pred_log_k2_GSH",
    })
    keep = ["canon_smi", "xtb_omega_eV", "xtb_q_Cb", "xtb_fukui_plus_Cb", "xtb_pred_log_k2_GSH"]
    return master.merge(xtb[keep], on="canon_smi", how="left")


def join_boltz_labels(master: pd.DataFrame) -> pd.DataFrame:
    """Join ZAP70-cofold Boltz-derived labels (dG_bind, ligand_strain, sg_cb_dist,
    combined_score) where SMILES matches. This is the ZAP70-specific table."""
    if not BOLTZ_ZAP70_CSV.exists():
        print("[join_boltz] ZAP70 file missing")
        return master
    boltz = pd.read_csv(BOLTZ_ZAP70_CSV)
    if "smiles" not in boltz.columns:
        print("[join_boltz] no smiles col")
        return master
    boltz = boltz[boltz.get("success_flag", 1) == 1].copy() if "success_flag" in boltz.columns else boltz
    boltz["canon_smi"] = boltz["smiles"].apply(canon)
    boltz = boltz.dropna(subset=["canon_smi"]).drop_duplicates(subset=["canon_smi"])
    keep_cols = ["canon_smi"] + [c for c in [
        "dG_bind_kcalmol", "ligand_strain_kcalmol", "rmsd_min_A",
        "sg_cb_dist_A", "combined_score",
    ] if c in boltz.columns]
    boltz = boltz[keep_cols].rename(columns={
        "dG_bind_kcalmol": "boltz_dG_bind_kcalmol",
        "ligand_strain_kcalmol": "boltz_ligand_strain_kcalmol",
        "rmsd_min_A": "boltz_rmsd_min_A",
        "sg_cb_dist_A": "boltz_sg_cb_dist_A",
        "combined_score": "boltz_combined_score",
    })
    return master.merge(boltz, on="canon_smi", how="left")


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
NUMERIC_PROXIES = [
    "MW", "LogP", "TPSA", "HBA", "HBD", "RotBonds", "QED", "SA", "fsp3",
    "NumRings", "NumAromRings", "Lipinski_violations",
    "Tc_to_Mol1", "MaxTc_to_train", "MedTc_to_kinase",
    "planar_dihedral_deg", "hammett_sigma",
    "xtb_omega_eV", "xtb_q_Cb", "xtb_fukui_plus_Cb", "xtb_pred_log_k2_GSH",
    "boltz_dG_bind_kcalmol", "boltz_ligand_strain_kcalmol",
    "boltz_sg_cb_dist_A", "boltz_rmsd_min_A", "boltz_combined_score",
]
BOOL_PROXIES = ["acryl_largest", "acryl_any", "michael_any", "PAINS", "Brenk"]


def summarize(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in master.groupby("cohort"):
        row = {"cohort": cohort, "n": len(sub)}
        for p in NUMERIC_PROXIES:
            if p not in sub.columns:
                continue
            vals = pd.to_numeric(sub[p], errors="coerce").dropna()
            row[f"{p}_n"] = len(vals)
            if len(vals) == 0:
                continue
            row[f"{p}_median"] = float(vals.median())
            row[f"{p}_mean"] = float(vals.mean())
            row[f"{p}_std"] = float(vals.std())
            row[f"{p}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25))
            row[f"{p}_min"] = float(vals.min())
            row[f"{p}_max"] = float(vals.max())
        for p in BOOL_PROXIES:
            if p not in sub.columns:
                continue
            vals = sub[p].astype(float).dropna()
            if len(vals) > 0:
                row[f"{p}_frac"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)


def compute_wasserstein_to_ref(master: pd.DataFrame, ref_cohort: str) -> pd.DataFrame:
    ref = master[master["cohort"] == ref_cohort]
    rows = []
    for cohort, sub in master.groupby("cohort"):
        if cohort == ref_cohort:
            continue
        row = {"cohort": cohort}
        for p in NUMERIC_PROXIES:
            if p not in sub.columns or p not in ref.columns:
                continue
            r_vals = pd.to_numeric(ref[p], errors="coerce").dropna().values
            s_vals = pd.to_numeric(sub[p], errors="coerce").dropna().values
            if len(r_vals) < 3 or len(s_vals) < 3:
                continue
            try:
                w = sstats.wasserstein_distance(r_vals, s_vals)
                row[f"W_{p}"] = float(w)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


GEN_COHORTS = ["scheme_A", "scheme_B", "scheme_C", "infonce", "v2cond"]


def flag_saturation(summary: pd.DataFrame, ref_cohort_summary: pd.Series, all_master: pd.DataFrame,
                    exclude_ref: bool = True) -> pd.DataFrame:
    """For each numeric proxy, decide:
       - controllable if variance-across-generative-cohort-medians > 0.5 * median within-cohort std
       - gap if |median_of_gen_medians - FDA_median| > 0.5 * FDA_std
    Only compares across the 5 generative cohorts against FDA drug reference."""
    gen = summary[summary["cohort"].isin(GEN_COHORTS)] if exclude_ref else summary
    ref = summary[summary["cohort"] == "ref_FDA"]
    rows = []
    for p in NUMERIC_PROXIES:
        med_col = f"{p}_median"
        std_col = f"{p}_std"
        if med_col not in gen.columns:
            continue
        gen_meds = pd.to_numeric(gen[med_col], errors="coerce").dropna()
        if len(gen_meds) < 2:
            continue
        gen_med_std = float(gen_meds.std())
        # median within-cohort std
        within_stds = pd.to_numeric(gen[std_col], errors="coerce").dropna()
        med_within_std = float(within_stds.median()) if len(within_stds) else float("nan")
        # reliability: require ≥30 samples per cohort in ≥3 of 5 generative cohorts
        n_col = f"{p}_n"
        n_reliable = 0
        if n_col in gen.columns:
            for _, r in gen.iterrows():
                if pd.notna(r.get(n_col)) and r.get(n_col, 0) >= 30:
                    n_reliable += 1
        reliable = n_reliable >= 3
        # controllable if variance across cohort medians > 0.25 * median within-cohort std
        # high_leverage: > 0.5x (stronger signal)
        if med_within_std > 0:
            controllable = reliable and gen_med_std > 0.25 * med_within_std
            high_leverage = reliable and gen_med_std > 0.5 * med_within_std
        else:
            controllable = False
            high_leverage = False
        # gap vs FDA
        gap = float("nan")
        gap_flag = False
        if len(ref) > 0 and med_col in ref.columns and std_col in ref.columns:
            ref_med = float(pd.to_numeric(ref[med_col], errors="coerce").iloc[0])
            ref_std = float(pd.to_numeric(ref[std_col], errors="coerce").iloc[0])
            gen_med_of_meds = float(gen_meds.median())
            if not np.isnan(ref_std) and ref_std > 0:
                gap = (gen_med_of_meds - ref_med) / ref_std
                gap_flag = abs(gap) > 0.5
        rows.append({
            "proxy": p,
            "gen_median_of_medians": float(gen_meds.median()),
            "gen_med_std": gen_med_std,
            "med_within_std": med_within_std,
            "reliable": reliable,
            "n_reliable_cohorts": n_reliable,
            "controllable": controllable,
            "high_leverage": high_leverage,
            "gap_vs_FDA_zscore": gap,
            "gap_flag": gap_flag,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("COHORT DISTRIBUTIONAL AUDIT")
    print("=" * 70)

    patts = build_patts()

    # ----- Load cohorts -----
    cohort_dfs = []
    for name, path in COHORTS.items():
        if not path.exists():
            print(f"  [MISS] cohort {name} @ {path}")
            continue
        df = load_cohort(name, path, n_max=3000)
        print(f"  [OK] {name}: {len(df)} rows")
        cohort_dfs.append(df)

    # ----- Load references -----
    ref_fda = load_reference_fda()
    print(f"  [REF] FDA covalent drugs: {len(ref_fda)} mols")
    ref_zap = load_reference_zap70()
    print(f"  [REF] ZAP70 (PDB+boltz): {len(ref_zap)} mols")
    ref_kin = load_reference_kinase()
    print(f"  [REF] Kinase controls: {len(ref_kin)} mols")

    all_input = pd.concat(cohort_dfs + [ref_fda, ref_zap, ref_kin], ignore_index=True)
    print(f"  Total input rows: {len(all_input)}")

    # ----- Build training-pool FPs (proxy = ZAP70 real ligand set) -----
    print("\n[FP] Building reference FPs...")
    train_pool_smi = ref_zap["SMILES"].tolist()
    train_pool_fps = build_fp_list(train_pool_smi)
    kinase_ref_fps = build_fp_list(ref_kin["SMILES"].tolist())
    mol1_fp = morgan_fp(MOL1_SMI)
    print(f"  train_pool_fps={len(train_pool_fps)}, kinase_ref_fps={len(kinase_ref_fps)}")

    # ----- Compute per-mol proxy panel -----
    print("\n[PROXY] Computing per-mol chemistry panel...")
    master = audit_dataframe(
        all_input, patts,
        train_pool_fps=train_pool_fps,
        kinase_ref_fps=kinase_ref_fps,
        mol1_fp=mol1_fp,
        do_planar=True,
        planar_sample=200,
        hammett=True,
    )
    print(f"  master rows: {len(master)}")

    # ----- Join xTB + Boltz labels -----
    print("\n[JOIN] xTB labels...")
    master = join_xtb_labels(master)
    n_xtb = master["xtb_omega_eV"].notna().sum() if "xtb_omega_eV" in master.columns else 0
    print(f"  xTB label overlap: {n_xtb} rows")

    print("\n[JOIN] Boltz labels...")
    master = join_boltz_labels(master)
    n_boltz = master["boltz_mpae_warhead_cys"].notna().sum() if "boltz_mpae_warhead_cys" in master.columns else 0
    print(f"  Boltz label overlap: {n_boltz} rows")

    # ----- Save master -----
    master_path = OUT_DIR / "audit_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\n[SAVE] master -> {master_path}")

    # ----- Summarize -----
    print("\n[SUM] Building per-cohort summary...")
    summary = summarize(master)
    summary_path = OUT_DIR / "audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] summary -> {summary_path}")

    # ----- Wasserstein distance to FDA -----
    print("\n[WASS] Wasserstein vs FDA...")
    wass = compute_wasserstein_to_ref(master, "ref_FDA")
    wass_path = OUT_DIR / "audit_wasserstein_vs_FDA.csv"
    wass.to_csv(wass_path, index=False)
    print(f"[SAVE] wasserstein -> {wass_path}")

    # ----- Saturation / gap flags -----
    print("\n[FLAG] Computing saturation + gap flags...")
    fda_series = summary[summary["cohort"] == "ref_FDA"].iloc[0] if (summary["cohort"] == "ref_FDA").any() else pd.Series()
    flags = flag_saturation(summary, fda_series, master)
    flags_path = OUT_DIR / "audit_axis_flags.csv"
    flags.to_csv(flags_path, index=False)
    print(f"[SAVE] flags -> {flags_path}")

    # ----- Boltz overlap sanity -----
    print("\n[SANITY] Boltz coverage per cohort:")
    if "boltz_dG_bind_kcalmol" in master.columns:
        for cohort, sub in master.groupby("cohort"):
            overlap = sub["boltz_dG_bind_kcalmol"].notna().sum()
            frac = overlap / max(1, len(sub))
            flag = " (LOW!)" if frac < 0.20 else ""
            print(f"  {cohort:20s}  {overlap:5d}/{len(sub):5d}  {frac:.1%}{flag}")

    # ----- Write markdown reports -----
    write_report_md(master, summary, wass, flags, MISSING_COHORTS)
    write_findings_md(summary, wass, flags)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print(f"Outputs in {OUT_DIR}/")
    print("=" * 70)


def _fmt(x, digits=3):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def write_report_md(master: pd.DataFrame, summary: pd.DataFrame,
                    wass: pd.DataFrame, flags: pd.DataFrame,
                    missing: List[str]):
    p = OUT_DIR / "audit_report.md"
    lines = ["# Distributional Audit Report",
             "",
             f"Cohorts audited: {sorted(master['cohort'].unique().tolist())}",
             f"Missing cohorts (not emitted locally): {missing}",
             "",
             "## 1. Per-cohort n and headline metrics",
             ""]
    hdr = ["cohort", "n", "acryl_largest_frac", "michael_any_frac", "PAINS_frac", "Brenk_frac",
           "QED_med", "SA_med", "MW_med", "LogP_med", "Tc_to_Mol1_med", "MaxTc_train_med",
           "hammett_med", "planar_dih_med"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for _, r in summary.iterrows():
        row = [r.get("cohort", "-"), int(r.get("n", 0)),
               _fmt(r.get("acryl_largest_frac"), 3),
               _fmt(r.get("michael_any_frac"), 3),
               _fmt(r.get("PAINS_frac"), 3),
               _fmt(r.get("Brenk_frac"), 3),
               _fmt(r.get("QED_median"), 3),
               _fmt(r.get("SA_median"), 2),
               _fmt(r.get("MW_median"), 1),
               _fmt(r.get("LogP_median"), 2),
               _fmt(r.get("Tc_to_Mol1_median"), 3),
               _fmt(r.get("MaxTc_to_train_median"), 3),
               _fmt(r.get("hammett_sigma_median"), 3),
               _fmt(r.get("planar_dihedral_deg_median"), 2)]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")

    lines.append("## 2. xTB / Boltz label coverage")
    lines.append("")
    lines.append("| cohort | n | xTB_overlap | boltz_overlap |")
    lines.append("|---|---|---|---|")
    for cohort, sub in master.groupby("cohort"):
        x_n = sub["xtb_omega_eV"].notna().sum() if "xtb_omega_eV" in sub.columns else 0
        b_n = sub["boltz_dG_bind_kcalmol"].notna().sum() if "boltz_dG_bind_kcalmol" in sub.columns else 0
        lines.append(f"| {cohort} | {len(sub)} | {x_n} ({x_n/max(1,len(sub)):.1%}) | {b_n} ({b_n/max(1,len(sub)):.1%}) |")
    lines.append("")

    lines.append("## 3. Saturation + Gap-to-FDA flags")
    lines.append("")
    lines.append("- **controllable**: variance across cohort medians > 0.5 * median within-cohort std")
    lines.append("- **gap_flag**: |gen_median - FDA_median| > 0.5 * FDA_std")
    lines.append("")
    lines.append("| proxy | gen_med_of_meds | gen_med_std | med_within_std | reliable | controllable | high_leverage | gap_vs_FDA (z) | gap_flag |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    flags_sorted = flags.copy()
    flags_sorted["combined"] = flags_sorted["gen_med_std"].abs().fillna(0) * flags_sorted["gap_vs_FDA_zscore"].abs().fillna(0)
    flags_sorted = flags_sorted.sort_values("combined", ascending=False)
    for _, f in flags_sorted.iterrows():
        lines.append(f"| {f['proxy']} | {_fmt(f['gen_median_of_medians'],3)} | {_fmt(f['gen_med_std'],3)} | "
                     f"{_fmt(f['med_within_std'],3)} | {f.get('reliable', True)} | "
                     f"{f['controllable']} | {f.get('high_leverage', False)} | "
                     f"{_fmt(f['gap_vs_FDA_zscore'],2)} | {f['gap_flag']} |")
    lines.append("")

    lines.append("## 4. Wasserstein distance to FDA (top 12 axes, ranked)")
    lines.append("")
    if len(wass) > 0:
        w = wass.copy()
        num_cols = [c for c in w.columns if c.startswith("W_")]
        # rank by median W across generative cohorts
        gen_wass = w[~w["cohort"].isin(["ref_ZAP70", "ref_kinase"])]
        med_w = {c: float(pd.to_numeric(gen_wass[c], errors="coerce").median()) for c in num_cols}
        med_w = {k: v for k, v in med_w.items() if not np.isnan(v)}
        top = sorted(med_w.items(), key=lambda kv: -kv[1])[:12]
        lines.append("| proxy | " + " | ".join(w["cohort"].astype(str).tolist()) + " |")
        lines.append("|---|" + "|".join(["---"] * len(w)) + "|")
        for prop, _ in top:
            row = [prop.replace("W_", "")]
            for _, r in w.iterrows():
                row.append(_fmt(r[prop], 3))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 5. Ranked top 5 controllable axes with real gap (reliable proxies only)")
    lines.append("")
    reliable_only = flags[flags.get("reliable", True)] if "reliable" in flags.columns else flags
    rank_df = reliable_only[reliable_only["controllable"] & reliable_only["gap_flag"]].copy()
    if len(rank_df):
        # normalize each axis by median-within-std to make the score dimensionless
        rank_df["ctrl_z"] = rank_df["gen_med_std"] / rank_df["med_within_std"].replace(0, np.nan)
        rank_df["score"] = rank_df["ctrl_z"].abs().fillna(0) * rank_df["gap_vs_FDA_zscore"].abs().fillna(0)
        rank_df = rank_df.sort_values("score", ascending=False).head(5)
        for _, r in rank_df.iterrows():
            direction = "generated > FDA" if r["gap_vs_FDA_zscore"] > 0 else "generated < FDA"
            lines.append(f"- **{r['proxy']}**: ctrl_z={_fmt(r['ctrl_z'],2)}, "
                         f"gap z={_fmt(r['gap_vs_FDA_zscore'],2)} ({direction}); "
                         f"combined score={_fmt(r['score'],3)}")
    else:
        lines.append("(no reliable proxy meets both controllable and gap_flag criteria)")

    p.write_text("\n".join(lines))
    print(f"[SAVE] report -> {p}")


def write_findings_md(summary: pd.DataFrame, wass: pd.DataFrame, flags: pd.DataFrame):
    p = OUT_DIR / "audit_key_findings.md"
    reliable_flags = flags[flags.get("reliable", True)] if "reliable" in flags.columns else flags
    saturated = reliable_flags[~reliable_flags["controllable"]]["proxy"].tolist()
    controllable = reliable_flags[reliable_flags["controllable"]]["proxy"].tolist()
    gap_flagged = reliable_flags[reliable_flags["gap_flag"]]["proxy"].tolist()
    controllable_and_gap = reliable_flags[reliable_flags["controllable"] & reliable_flags["gap_flag"]].copy()
    if len(controllable_and_gap):
        controllable_and_gap["ctrl_z"] = controllable_and_gap["gen_med_std"] / \
                                          controllable_and_gap["med_within_std"].replace(0, np.nan)
        controllable_and_gap["score"] = controllable_and_gap["ctrl_z"].abs().fillna(0) * \
                                        controllable_and_gap["gap_vs_FDA_zscore"].abs().fillna(0)
        controllable_and_gap = controllable_and_gap.sort_values("score", ascending=False)
    lines = ["# Distributional Audit — Key Findings (TL;DR)",
             "",
             f"- **Saturated axes (no leverage across cohorts)**: {', '.join(saturated) if saturated else '(none)'}",
             f"- **Controllable axes (real variance across cohorts)**: {', '.join(controllable) if controllable else '(none)'}",
             f"- **Gap vs FDA drugs (|z|>0.5)**: {', '.join(gap_flagged) if gap_flagged else '(none)'}",
             ""]
    if len(controllable_and_gap):
        top = controllable_and_gap.head(3)
        axis_list = [f"{r['proxy']} (z={_fmt(r['gap_vs_FDA_zscore'],2)}, score={_fmt(r['score'],3)})"
                     for _, r in top.iterrows()]
        lines.append(f"- **Verdict**: significant gap remains on: {', '.join(axis_list)}")
        lines.append(f"- **Recommended next training target**: {top.iloc[0]['proxy']} "
                     f"(largest controllable × gap combined score)")
    else:
        lines.append("- **Verdict**: our generated cohorts are already good enough on all evaluated axes")
        lines.append("- **Recommended**: we're saturated — publish negative result on control experiments")
    p.write_text("\n".join(lines))
    print(f"[SAVE] key findings -> {p}")


if __name__ == "__main__":
    main()
