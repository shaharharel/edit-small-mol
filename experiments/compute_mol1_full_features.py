"""Compute the FULL feature dict for Mol1 — mirroring every column the
838 survivors table renders.

Strategy:
  1. Reuse existing `mol1_anchor_enriched.json` for the slow-to-compute
     per-pose cofold metrics (d_SG, h-bonds, contacts, occupancy, Vina rescore,
     PROPKA, xTB, RDKit strain, 3D-seed self-similarity). Already computed
     2026-05-27 against the Mol1__zap70_cys346 cofold.
  2. Add the full Boltz confidence panel from the Mol1 cofold confidence JSON
     + parse the PAE .npz for mPAE_paper / mPAE_london.
  3. Add identity / drug-likeness from RDKit (MW, LogP, TPSA, ..., SAScore,
     PAINS_alerts, Brenk_alerts, fsp3).
  4. Add Mol1's FiLMDelta ensemble pIC50 + anchor wins via the same scorer
     used in `recompute_anchor_wins_for_838.py`. Pass `[MOL1_SMI]`, take the
     single result. Delta vs Mol1 = 0 by definition.
  5. Add kinase-classifier P_kinase / P_Tec_family / pIC50_kinase_aux —
     featurize Morgan + ChemBERTa MTR, score against `trunk.pt` + Platt.
  6. Add cohort-percentile-ranked Combined Score. Percentile of Mol1 WITHIN
     the survivors cohort (F4_boltz_full_filtered.csv) for each of the 8
     desirability components, then apply the same 8-component geometric
     mean formula used in backend.py (`(P_pot)^0.22 × (P_pose)^0.18 × ...`).

Output: results/paper_evaluation/mol1_full_features.json
        — single dict with every column → value the survivors table renders.

Wall-clock target: ~3 minutes (the slow pieces are cached from May 27).
Run: conda run -n quris python experiments/compute_mol1_full_features.py
"""
from __future__ import annotations

import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
COFOLD_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "mol1__zap70_cys346" / "mol1"
CONFIDENCE_JSON = COFOLD_DIR / "confidence_mol1_model_0.json"
PAE_NPZ = COFOLD_DIR / "pae_mol1_model_0.npz"
CIF_PATH = COFOLD_DIR / "mol1_model_0.cif"
ENRICHED_JSON = PROJECT_ROOT / "data" / "boltz_poses" / "mol1_anchor_enriched.json"
COHORT_CSV = PROJECT_ROOT / "data" / "tier4_scored" / "F4_boltz_full_filtered.csv"
OUT_JSON = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_full_features.json"


def log(msg: str) -> None:
    print(f"[mol1_full] {msg}", flush=True)


# ── 1. Identity / drug-likeness via RDKit ─────────────────────────────────

def compute_identity_druglike() -> dict:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, Crippen, Lipinski, QED, rdMolDescriptors,
        FilterCatalog, RDConfig, AllChem, DataStructs,
    )

    mol = Chem.MolFromSmiles(MOL1_SMILES)
    out = {
        "smiles": MOL1_SMILES,
        "method": "Seed (Mol 1) - Design Anchor",
        "methods_list": "Seed (Mol 1) - Design Anchor",
        "yaml_name": "mol1",
        "row_id": "M1",
    }
    out["MW"] = float(Descriptors.MolWt(mol))
    out["LogP"] = float(Crippen.MolLogP(mol))
    out["TPSA"] = float(Descriptors.TPSA(mol))
    out["HBA"] = int(Lipinski.NumHAcceptors(mol))
    out["HBD"] = int(Lipinski.NumHDonors(mol))
    out["RotBonds"] = int(Lipinski.NumRotatableBonds(mol))
    out["HeavyAtoms"] = int(mol.GetNumHeavyAtoms())
    out["Rings"] = int(rdMolDescriptors.CalcNumRings(mol))
    out["fsp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    out["QED"] = float(QED.qed(mol))
    out["Lipinski_violations"] = int(
        (out["MW"] > 500) + (out["LogP"] > 5) + (out["HBA"] > 10) + (out["HBD"] > 5)
    )

    # PAINS
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog.FilterCatalog(params)
        out["PAINS_alerts"] = int(len(cat.GetMatches(mol)))
    except Exception as e:
        log(f"  PAINS failed: {e}")
        out["PAINS_alerts"] = None

    # Brenk
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        cat = FilterCatalog.FilterCatalog(params)
        out["Brenk_alerts"] = int(len(cat.GetMatches(mol)))
    except Exception as e:
        log(f"  Brenk failed: {e}")
        out["Brenk_alerts"] = None

    # SAScore
    try:
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_path) not in sys.path:
            sys.path.append(str(sa_path))
        import sascorer  # type: ignore
        out["SAScore"] = float(sascorer.calculateScore(mol))
    except Exception as e:
        log(f"  SAScore failed: {e}")
        out["SAScore"] = None

    # Substructure / warhead flags
    out["warhead_intact"] = True
    out["acryl_match"] = True
    out["mol1_murcko_match"] = True
    out["mol1_murcko_smarts_match"] = True
    out["thiq_core"] = False  # Mol1 itself is not the THIQ derivative
    out["novel_vs_chembl_zap70"] = False  # Mol1 IS the anchor; not novel

    # 2D Tc
    out["Tc_to_Mol1"] = 1.0

    # Tc to training set (mean top-10 + max)
    try:
        from experiments.run_zap70_v3 import load_zap70_molecules
        train_df, _ = load_zap70_molecules()
        mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        train_fps = []
        for s in train_df["smiles"].tolist():
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        sims = np.array(DataStructs.BulkTanimotoSimilarity(mol1_fp, train_fps))
        out["max_Tc_train"] = float(sims.max())
        top10 = np.partition(sims, -10)[-10:]
        out["mean_top10_Tc_train"] = float(top10.mean())
    except Exception as e:
        log(f"  Tc-train failed: {e}")
        out["max_Tc_train"] = None
        out["mean_top10_Tc_train"] = None

    # pubTc v3 panel
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from pubtc_panel_v3 import PANEL_SMILES  # type: ignore
        panel_fps, panel_names = [], []
        for name, ps in PANEL_SMILES.items():
            pm = Chem.MolFromSmiles(ps)
            if pm is None:
                continue
            panel_fps.append(AllChem.GetMorganFingerprintAsBitVect(pm, 2, nBits=2048))
            panel_names.append(name)
        mol1_fp_v3 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(mol1_fp_v3, panel_fps))
        out["max_pubTc"] = float(sims.max())
        out["median_pubTc"] = float(np.median(sims))
        out["mean_pubTc"] = float(sims.mean())
        k = min(10, len(sims))
        top10 = np.partition(sims, -k)[-k:]
        out["top10_mean_pubTc"] = float(top10.mean())
        out["closest_lead"] = panel_names[int(np.argmax(sims))]
    except Exception as e:
        log(f"  pubTc v3 panel failed: {e}")
        for k in ("max_pubTc", "median_pubTc", "mean_pubTc", "top10_mean_pubTc"):
            out[k] = None
        out["closest_lead"] = None

    # By-definition self-similarity
    out["shape_Tc_mol1"] = 1.0
    out["o3a_score"] = None  # n/a vs self

    return out


# ── 2. Boltz confidence panel + mPAE_paper / mPAE_london ────────────────

def compute_boltz_confidence() -> dict:
    out = {}
    if not CONFIDENCE_JSON.exists():
        log(f"  Boltz confidence JSON missing: {CONFIDENCE_JSON}")
        return out
    conf = json.loads(CONFIDENCE_JSON.read_text())
    # Map confidence.json keys -> survivor column names
    out["boltz_confidence_score"] = conf.get("confidence_score")
    out["boltz_ptm"] = conf.get("ptm")
    out["boltz_iptm"] = conf.get("iptm")
    out["boltz_ligand_iptm"] = conf.get("ligand_iptm")
    out["boltz_protein_iptm"] = conf.get("protein_iptm")
    out["boltz_complex_plddt"] = conf.get("complex_plddt")
    out["boltz_complex_iplddt"] = conf.get("complex_iplddt")
    out["boltz_complex_pde"] = conf.get("complex_pde")
    out["boltz_complex_ipde"] = conf.get("complex_ipde")
    # Legacy aliases (the UI sometimes reads these directly)
    out["iptm"] = out["boltz_iptm"]
    out["ligand_iptm"] = out["boltz_ligand_iptm"]
    out["boltz_plddt"] = out["boltz_complex_plddt"]
    out["complex_plddt"] = out["boltz_complex_plddt"]
    out["boltz_pde"] = out["boltz_complex_pde"]
    out["complex_pde"] = out["boltz_complex_pde"]
    out["boltz_confidence"] = out["boltz_confidence_score"]

    # mPAE_paper: PROXY = scalar from confidence.json.complex_pde (matches the
    # convention used elsewhere in the codebase — `out["mPAE_paper"] =
    # out.get("complex_pde")` in compute_full_boltz_metrics.py).
    out["mPAE_paper"] = out["boltz_complex_pde"]
    out["mPAE"] = out["mPAE_paper"]  # alias

    # mPAE_london: TRUE = min over protein × ligand block of raw PAE matrix.
    if PAE_NPZ.exists():
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(MOL1_SMILES)
            mol_h = Chem.AddHs(mol)
            n_lig_atoms_with_h = mol_h.GetNumAtoms()
            d = np.load(PAE_NPZ)
            pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
            if pae.ndim == 2:
                N = pae.shape[0]
                # The PAE matrix has heavy-atom granularity for the ligand in
                # Boltz; try both heavy-atom count and full-H count, take whichever
                # leaves a sane protein block (>50 rows).
                for n_lig in (mol.GetNumHeavyAtoms(), n_lig_atoms_with_h):
                    lig_lo = N - n_lig
                    if lig_lo > 50 and lig_lo < N:
                        cross = pae[:lig_lo, lig_lo:]
                        out["mPAE_london"] = float(np.min(cross))
                        break
                else:
                    out["mPAE_london"] = None
            else:
                out["mPAE_london"] = None
        except Exception as e:
            log(f"  mPAE_london compute failed: {e}")
            out["mPAE_london"] = None
    else:
        out["mPAE_london"] = None
    return out


# ── 3. FiLMDelta pIC50 + anchor wins (Mol1 vs 280 anchors) ──────────────

def compute_pic50_anchor_wins() -> dict:
    """Score Mol1 with the 3-seed FiLMDelta ensemble (same logic as
    `recompute_anchor_wins_for_838.py`). Returns pIC50_mean, pIC50_std,
    anchor_wins, anchor_wins_ge7. delta_vs_mol1 / direct_delta_from_mol1 = 0
    by definition.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    ENS_DIR = PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_ensemble"
    out = {
        "pIC50_method": "FiLMDelta",
        "pIC50_mean": None,
        "pIC50_std": None,
        "pIC50_film": None,
        "anchor_wins": None,
        "anchor_wins_ge7": None,
        "delta_vs_mol1": 0.0,
        "direct_delta_from_mol1": 0.0,
    }

    mol = Chem.MolFromSmiles(MOL1_SMILES)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    fp = np.array(bv, dtype=np.float32)

    ensemble = []
    for k in range(3):
        path = ENS_DIR / f"film_seed{k}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256])
        model.load_state_dict(ck["model_state"])
        model.eval()
        ensemble.append({
            "model": model,
            "scaler_mean": np.asarray(ck["scaler_mean"], dtype=np.float32),
            "scaler_scale": np.asarray(ck["scaler_scale"], dtype=np.float32),
            "anchor_embs": ck["anchor_embs"].float(),
            "anchor_pIC50": np.asarray(ck["anchor_pIC50"], dtype=np.float64),
        })

    n_anchors = len(ensemble[0]["anchor_pIC50"])
    high_potency_mask = ensemble[0]["anchor_pIC50"] >= 7.0
    log(f"  ensemble loaded: {n_anchors} anchors, {high_potency_mask.sum()} with pIC50>=7")

    seed_means, seed_wins, seed_wins_ge7 = [], [], []
    with torch.no_grad():
        for ens in ensemble:
            z = (fp - ens["scaler_mean"]) / ens["scaler_scale"]
            cand_e = torch.FloatTensor(z.reshape(1, -1))
            cand_t = cand_e.expand(n_anchors, -1)
            deltas = ens["model"](ens["anchor_embs"], cand_t).numpy()  # (280,)
            seed_means.append(float((ens["anchor_pIC50"] + deltas).mean()))
            seed_wins.append(int((deltas > 0).sum()))
            seed_wins_ge7.append(int(((deltas > 0) & high_potency_mask).sum()))
    out["pIC50_mean"] = float(np.mean(seed_means))
    out["pIC50_std"] = float(np.std(seed_means))
    out["pIC50_film"] = float(np.mean(seed_means))
    out["anchor_wins"] = float(np.mean(seed_wins))
    out["anchor_wins_ge7"] = float(np.mean(seed_wins_ge7))
    log(f"  Mol1 pIC50_mean = {out['pIC50_mean']:.3f} ± {out['pIC50_std']:.3f}  wins={out['anchor_wins']:.0f}/280  wins_ge7={out['anchor_wins_ge7']:.0f}/57")
    return out


# ── 4. Kinase classifier (P_kinase, P_Tec_family, pIC50_kinase_aux) ─────

def compute_kinase_classifier() -> dict:
    import torch
    from experiments.kinase_clf.featurize_and_train import (
        ThreeHeadFFN, featurize_morgan, featurize_chemberta,
        load_cache_dict, EMB_CACHE_MORGAN,
    )

    MODEL_DIR = PROJECT_ROOT / "models" / "kinase_clf"

    morgan_lookup = load_cache_dict(EMB_CACHE_MORGAN, expected_dim=2048)
    X_morgan = featurize_morgan([MOL1_SMILES], morgan_lookup)
    X_cb = featurize_chemberta([MOL1_SMILES], batch_size=1)
    X = np.concatenate([X_morgan, X_cb], axis=1).astype(np.float32)

    ckpt = torch.load(MODEL_DIR / "trunk.pt", map_location="cpu")
    pic_mu = ckpt["pic_mu"]; pic_sd = ckpt["pic_sd"]
    model = ThreeHeadFFN(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(MODEL_DIR / "calibrators.pkl", "rb") as fh:
        cal = pickle.load(fh)
    platt_kin = cal["platt_kin"]
    platt_tec = cal["platt_tec"]

    with torch.no_grad():
        xb = torch.from_numpy(X)
        lk, lt, lp = model(xb)
        lk = lk.numpy(); lt = lt.numpy(); lp = lp.numpy()

    p_kin = platt_kin.predict_proba(lk.reshape(-1, 1))[:, 1]
    if platt_tec is not None:
        p_tec = platt_tec.predict_proba(lt.reshape(-1, 1))[:, 1]
    else:
        p_tec = 1 / (1 + np.exp(-lt))
    pic50_pred = lp * pic_sd + pic_mu

    out = {
        "P_kinase": float(p_kin[0]),
        "P_Tec_family": float(p_tec[0]),
        "pIC50_kinase_aux": float(pic50_pred[0]),
    }
    log(f"  P_kinase={out['P_kinase']:.3f}  P_Tec={out['P_Tec_family']:.3f}  pIC50_aux={out['pIC50_kinase_aux']:.3f}")
    return out


# ── 5. Cohort percentile + Combined Score (Mol1 inserted into 838) ──────

def compute_combined_score(mol1_dict: dict) -> dict:
    """Insert Mol1's values into the survivor cohort (F4_boltz_full_filtered),
    rank each desirability component as a cohort percentile, then apply the
    same 8-component geometric mean used in backend.py.

    Returns the per-component P_x percentiles + the final desirability_score.
    """
    if not COHORT_CSV.exists():
        log(f"  cohort CSV missing: {COHORT_CSV}")
        return {}
    df = pd.read_csv(COHORT_CSV, low_memory=False)
    log(f"  cohort loaded: {len(df):,} rows")

    # Pull Mol1's values for each desirability component.
    mol1_row = {
        "anchor_wins_ge7": mol1_dict.get("anchor_wins_ge7"),
        "mPAE_london": mol1_dict.get("mPAE_london"),
        "boltz_ligand_iptm": mol1_dict.get("boltz_ligand_iptm"),
        "shape_Tc_mol1": mol1_dict.get("shape_Tc_mol1"),
        "d_SG": mol1_dict.get("d_SG"),
        "burgi_dunitz_dev_deg": mol1_dict.get("burgi_dunitz_dev_deg"),
        "pred_log_k2_GSH": mol1_dict.get("pred_log_k2_GSH"),
        "P_Tec_family": mol1_dict.get("P_Tec_family"),
        "P_kinase": mol1_dict.get("P_kinase"),
        "pIC50_kinase_aux": mol1_dict.get("pIC50_kinase_aux"),
    }

    # Append Mol1 as a single row, then rank.
    cohort_plus_mol1 = pd.concat(
        [df, pd.DataFrame([mol1_row])], ignore_index=True
    )

    def _pctl(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if not higher_is_better:
            s = -s
        return s.rank(pct=True, method="average").fillna(0.0)

    P_pot   = _pctl(cohort_plus_mol1["anchor_wins_ge7"], True)
    P_pose_raw = _pctl(cohort_plus_mol1["mPAE_london"], False)
    iptm_gate = (pd.to_numeric(cohort_plus_mol1.get("boltz_ligand_iptm"),
                               errors="coerce") >= 0.92).fillna(False)
    P_pose = P_pose_raw.where(iptm_gate, P_pose_raw * 0.5)
    P_shape = _pctl(cohort_plus_mol1["shape_Tc_mol1"], True)
    P_dsg = _pctl(cohort_plus_mol1["d_SG"], False)
    P_bd  = _pctl(cohort_plus_mol1["burgi_dunitz_dev_deg"], False)
    P_geom = pd.concat([P_dsg, P_bd], axis=1).min(axis=1)
    P_react = _pctl(cohort_plus_mol1["pred_log_k2_GSH"], False)
    P_tec = _pctl(cohort_plus_mol1["P_Tec_family"], True)
    P_kin = _pctl(cohort_plus_mol1["P_kinase"], True)
    P_pic_aux = _pctl(cohort_plus_mol1["pIC50_kinase_aux"], True)

    EPS = 1e-3
    desirability = (
          (P_pot     + EPS) ** 0.22
        * (P_pose    + EPS) ** 0.18
        * (P_shape   + EPS) ** 0.20
        * (P_geom    + EPS) ** 0.10
        * (P_react   + EPS) ** 0.10
        * (P_tec     + EPS) ** 0.10
        * (P_kin     + EPS) ** 0.05
        * (P_pic_aux + EPS) ** 0.05
    )

    last = len(cohort_plus_mol1) - 1  # Mol1 row index
    out = {
        "P_potency":          float(P_pot.iloc[last]),
        "P_pose":             float(P_pose.iloc[last]),
        "P_shape_to_mol1":    float(P_shape.iloc[last]),
        "P_covgeom":          float(P_geom.iloc[last]),
        "P_clean_reactivity": float(P_react.iloc[last]),
        "P_Tec_family_pctl":  float(P_tec.iloc[last]),
        "P_kinase_pctl":      float(P_kin.iloc[last]),
        "P_pIC50_kinase_aux_pctl": float(P_pic_aux.iloc[last]),
        "desirability_score": float(desirability.iloc[last]),
    }
    log(f"  Mol1 Combined Score = {out['desirability_score']:.3f}")
    log(f"    components: P_pot={out['P_potency']:.2f}  P_pose={out['P_pose']:.2f}  "
        f"P_shape={out['P_shape_to_mol1']:.2f}  P_geom={out['P_covgeom']:.2f}  "
        f"P_react={out['P_clean_reactivity']:.2f}  P_Tec={out['P_Tec_family_pctl']:.2f}  "
        f"P_kin={out['P_kinase_pctl']:.2f}  P_pic_aux={out['P_pIC50_kinase_aux_pctl']:.2f}")
    return out


# ── 6. Efficiency metrics (LE / LLE / BEI / SEI / SILE) ─────────────────

def compute_efficiency(mol1_dict: dict) -> dict:
    """LE, LLE, BEI, SEI, SILE — defined wrt pIC50_mean."""
    pIC50 = mol1_dict.get("pIC50_mean")
    HA = mol1_dict.get("HeavyAtoms")
    MW = mol1_dict.get("MW")
    LogP = mol1_dict.get("LogP")
    TPSA = mol1_dict.get("TPSA")
    out = {"LE": None, "LLE": None, "BEI": None, "SEI": None, "SILE": None}
    if pIC50 is None or HA is None or HA == 0:
        return out
    out["LE"] = round(1.4 * pIC50 / HA, 3)
    if LogP is not None:
        out["LLE"] = round(pIC50 - LogP, 3)
    if MW is not None and MW > 0:
        out["BEI"] = round(pIC50 / (MW / 1000.0), 2)
    if TPSA is not None and TPSA > 0:
        out["SEI"] = round(pIC50 / (TPSA / 100.0), 2)
    out["SILE"] = round(pIC50 / (HA ** 0.3), 3)
    return out


# ── 7. Re-export legacy/extra columns from enriched cache ──────────────

def load_enriched_cache() -> dict:
    if not ENRICHED_JSON.exists():
        log(f"  ERROR: enriched cache missing at {ENRICHED_JSON}")
        return {}
    enr = json.loads(ENRICHED_JSON.read_text())

    # Renames to match the survivor column names exactly.
    out = dict(enr)
    # vina_kcalmol -> vina_rescore_affinity_kcalmol (survivor name)
    if enr.get("vina_kcalmol") is not None:
        out["vina_rescore_affinity_kcalmol"] = enr["vina_kcalmol"]
    if enr.get("vina_intra_kcalmol") is not None:
        out["vina_rescore_intra_kcalmol"] = enr["vina_intra_kcalmol"]
    if enr.get("vina_inter_kcalmol") is not None:
        out["vina_rescore_inter_kcalmol"] = enr["vina_inter_kcalmol"]
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()

    if not CONFIDENCE_JSON.exists():
        log(f"FATAL: Mol1 cofold not found at {CONFIDENCE_JSON}")
        return 2

    out: dict = {}

    log("1. Identity + drug-likeness (RDKit)...")
    out.update(compute_identity_druglike())

    log("2. Boltz confidence panel + mPAE_paper / mPAE_london ...")
    out.update(compute_boltz_confidence())

    log("3. Cofold pose enriched cache (Vina, PROPKA, xTB, strain, geometry)...")
    enr = load_enriched_cache()
    # Only fill in keys we don't already have (preserve identity / boltz values)
    for k, v in enr.items():
        if k not in out or out.get(k) in (None, "", float('nan')):
            out[k] = v

    log("4. FiLMDelta pIC50 + anchor wins (3-seed ensemble)...")
    out.update(compute_pic50_anchor_wins())

    log("5. Efficiency metrics (LE / LLE / BEI / SEI / SILE)...")
    out.update(compute_efficiency(out))

    log("6. Kinase classifier (P_kinase / P_Tec_family / pIC50_kinase_aux)...")
    try:
        out.update(compute_kinase_classifier())
    except Exception as e:
        log(f"  kinase classifier failed: {e}")
        out["P_kinase"] = None
        out["P_Tec_family"] = None
        out["pIC50_kinase_aux"] = None

    log("7. Cohort-percentile Combined Score (Mol1 inserted into F4_boltz_full_filtered)...")
    out.update(compute_combined_score(out))

    # Misc fields the table renders
    out["_score"] = None
    out["_tier"] = "seed"
    out["pIC50_xgb_multifp"] = None
    out["vanilla_vina_kcalmol"] = None
    out["adcov_local_kcalmol"] = None
    out["xtb_status"] = "ok" if out.get("LUMO_eV") is not None else None
    out["pharmacophore"] = "Mol1 Murcko / Murcko+acryl"

    # Clean numpy types -> Python primitives for JSON serialization
    clean: dict = {}
    for k, v in out.items():
        if isinstance(v, (np.floating,)):
            clean[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        elif isinstance(v, float) and not np.isfinite(v):
            clean[k] = None
        else:
            clean[k] = v

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(clean, indent=2, default=str))
    log(f"\nWROTE {OUT_JSON}")
    log(f"  total keys: {len(clean)}")
    log(f"  populated: {sum(1 for v in clean.values() if v not in (None, ''))}")
    nulls = sorted([k for k, v in clean.items() if v in (None, "")])
    if nulls:
        log(f"  null keys ({len(nulls)}): {nulls}")
    log(f"  wall-clock: {time.time() - t0:.1f}s")

    # Banner of headline values for the user
    log("")
    log("=== HEADLINE VALUES ===")
    log(f"  vina_rescore_affinity_kcalmol = {clean.get('vina_rescore_affinity_kcalmol')}")
    log(f"  vina_rescore_intra_kcalmol    = {clean.get('vina_rescore_intra_kcalmol')}")
    log(f"  pIC50_mean (FiLMDelta 3-seed) = {clean.get('pIC50_mean')}")
    log(f"  anchor_wins                    = {clean.get('anchor_wins')}")
    log(f"  anchor_wins_ge7                = {clean.get('anchor_wins_ge7')}")
    log(f"  P_kinase                       = {clean.get('P_kinase')}")
    log(f"  P_Tec_family                   = {clean.get('P_Tec_family')}")
    log(f"  pIC50_kinase_aux               = {clean.get('pIC50_kinase_aux')}")
    log(f"  mPAE_paper                     = {clean.get('mPAE_paper')}")
    log(f"  mPAE_london                    = {clean.get('mPAE_london')}")
    log(f"  d_SG                           = {clean.get('d_SG')}")
    log(f"  desirability_score             = {clean.get('desirability_score')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
