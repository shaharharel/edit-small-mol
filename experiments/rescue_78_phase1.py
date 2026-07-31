"""Rescue-78 Phase 1 driver.

Computes Phase 1 columns for the 78 rescued molecules:
  1.1 tox_alerts_count, tox_alert_names, k_inact_proxy (placeholder)
  1.2 formal_charge_drawn, net_charge_pH74 (Dimorphite), frac_charge_pH74, net_charge_pH74_mg, pKa_basic_max (MolGpKa)
  1.3 anchor_wins, anchor_wins_ge7, pIC50_std, delta_vs_mol1, direct_delta_from_mol1, pIC50_mean (recomputed)
  1.4 P_kinase, P_Tec_family, pIC50_kinase_aux (+ percentiles)
  1.5 shape_Tc_seed, esp_sim_seed, warhead_dev_deg

Input  : data/tier4_scored/rescue_78_input.csv  (78 rows, 61 cols)
Output : data/tier4_scored/rescue_78_working.csv (in place after each sub-phase)
State  : data/tier4_scored/rescue_78_state.json updated after each sub-phase.

Reads F4_boltz_full.csv for read-only reference (pIC50_kinase_aux percentile baseline).
Never writes to F4_boltz_full.csv.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

INPUT = ROOT / "data/tier4_scored/rescue_78_input.csv"
WORKING = ROOT / "data/tier4_scored/rescue_78_working.csv"
STATE = ROOT / "data/tier4_scored/rescue_78_state.json"
LOG = ROOT / "data/tier4_scored/rescue_78_progress.log"
F4 = ROOT / "data/tier4_scored/F4_boltz_full.csv"


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{now_z()}] {msg}\n")


def update_state(phase: str, status: str, **extra) -> None:
    s = json.loads(STATE.read_text())
    s["phases"][phase] = {"status": status, "ts": now_z(), **extra}
    STATE.write_text(json.dumps(s, indent=2))


def load_working() -> pd.DataFrame:
    if WORKING.exists():
        return pd.read_csv(WORKING)
    df = pd.read_csv(INPUT)
    df.to_csv(WORKING, index=False)
    return df


def save_working(df: pd.DataFrame) -> None:
    df.to_csv(WORKING, index=False)


# ─────────────────────────── 1.1 tox + k_inact placeholder ───────────────────────

ALERTS = [
    ("primary_aniline",         "[NX3H2]c1ccccc1",                  "major"),
    ("secondary_aniline",       "[NX3H1]([#6;!H0])c1ccccc1",        "major"),
    ("ortho_F_aniline",         "Nc1cccc(F)c1",                     "major"),
    ("anilinopyridine_2",       "Nc1ccccn1",                        "major"),
    ("anilinopyridine_3",       "Nc1cccnc1",                        "major"),
    ("anilinopyridine_4",       "Nc1ccncc1",                        "major"),
    ("aniline_pyridine_link",   "c1ccc(N[#6]c2ccncc2)cc1",          "major"),
    ("aniline_pyrimidine",      "Nc1ncccc1[NX3]",                   "major"),
    ("vinyl_aniline",           "C=Cc1ccc(N)cc1",                   "major"),
    ("vinyl_aniline_alt",       "C=Cc1cccc(N)c1",                   "major"),
    ("peroxide",                "[OX2][OX2]",                       "major"),
    ("acyl_hydrazide",          "[CX3](=O)[NX3H1][NX3H2]",          "major"),
    ("acyl_hydrazide_subst",    "[CX3](=O)[NX3H1][NX3H1]",          "major"),
    ("cyclic_sulfamide",        "[NX3R][SX4](=O)(=O)[NX3R]",        "major"),
    ("vinyl_ether_aliphatic",   "[#6]=[#6][OX2][#6]",               "major"),
    ("dihydrofuran",            "C1=CCO[CH2]1",                     "major"),
    ("enamine",                 "[NX3]([!#1])([!#1])C=C",           "major"),
    ("tetrahydropyridine",      "[NX3R]1[CH2][CH2]C=C[CH2]1",       "major"),
    ("cyclopropyl_CN",          "[#6;R3](C#N)",                     "major"),
    ("cyclobutyl_CN",           "[#6;R4](C#N)",                     "major"),
    ("alpha_dicyano",           "[CX4](C#N)(C#N)",                  "major"),
    ("N_F_bond",                "[NX3][F]",                         "major"),
    ("alpha_halo_carbonyl",     "[F,Cl,Br,I][CX4][CX3](=O)",        "major"),
    ("isocyanate",              "[NX2]=[CX2]=[OX1]",                "major"),
    ("thiocyanate",             "[#6][SX2][CX2]#[NX1]",             "major"),
    ("sulfonyl_chloride",       "[SX4](=O)(=O)[Cl]",                "major"),
    ("nitroalkene",             "[CX3]=[CX3][NX3](=O)=O",           "major"),
    ("second_michael_enone",    "[CX3]=[CX3][CX3](=O)[#6;!$([NX3])]","major"),
    ("second_michael_ester",    "[CX3]=[CX3][CX3](=O)[OX2][#6]",     "major"),
    ("aromatic_nitro",          "[c][NX3](=O)=O",                   "major"),
    ("epoxide",                 "C1OC1",                            "major"),
    ("imidoyl_chloride",        "[CX3](=[NX2])[Cl]",                "major"),
]


def count_alerts(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    hits = []
    for name, smarts, _ in ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and m.HasSubstructMatch(patt):
            hits.append(name)
    return len(hits), (",".join(hits) if hits else "")


def k_inact_proxy(pKa_Cys, d_SG, bd_dev):
    if any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
           for v in (pKa_Cys, d_SG)):
        return None
    if bd_dev is None or (isinstance(bd_dev, float) and (math.isnan(bd_dev) or math.isinf(bd_dev))):
        bd_dev = 30.0
    try:
        thiolate = 1.0 / (1.0 + 10 ** (float(pKa_Cys) - 7.4))
        d_term = (float(d_SG) - 1.8) ** 2
        bd_term = (float(bd_dev) / 10.0) ** 2
        geom = math.exp(-(d_term + bd_term))
        return thiolate * geom
    except (TypeError, ValueError, OverflowError):
        return None


def phase_1_1():
    df = load_working()
    t0 = time.time()
    counts, names = [], []
    for smi in df["smiles"]:
        c, n = count_alerts(smi)
        counts.append(c); names.append(n)
    df["tox_alerts_count"] = counts
    df["tox_alert_names"] = names
    # k_inact_proxy is computed later (Phase 3.8) once we have real Boltz pKa / d_SG / BD;
    # for now leave a NaN column so downstream merging is simpler.
    if "k_inact_proxy" not in df.columns:
        df["k_inact_proxy"] = np.nan
    save_working(df)
    update_state("1.1_tox_kinact", "done",
                 rows=len(df),
                 with_alert=int((df["tox_alerts_count"].fillna(0) >= 1).sum()))
    append_log(f"1.1 tox+kinact done ({time.time()-t0:.1f}s)")


# ─────────────────────────── 1.2 charges ──────────────────────────────────────

def compute_charges_dim(smi: str):
    from dimorphite_dl import protonate_smiles
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    fc = Chem.GetFormalCharge(m)
    try:
        out = protonate_smiles(smi, ph_min=7.4, ph_max=7.4, precision=0.5)
    except Exception:
        return fc, None
    if not out:
        return fc, None
    m2 = Chem.MolFromSmiles(out[0])
    if m2 is None:
        return fc, None
    return fc, Chem.GetFormalCharge(m2)


def compute_charges_mg(smi: str, predict_fn):
    PH = 7.4
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    try:
        base_dict, acid_dict = predict_fn(m)
    except Exception:
        return None, None, None
    frac_b = sum(1.0 / (1 + 10 ** (PH - pka)) for pka in base_dict.values())
    frac_a = sum(1.0 / (1 + 10 ** (pka - PH)) for pka in acid_dict.values())
    frac = frac_b - frac_a
    pka_basic_max = max(base_dict.values()) if base_dict else None
    return frac, round(frac), pka_basic_max


def phase_1_2():
    df = load_working()
    t0 = time.time()
    # Dimorphite
    drawn, ph74 = [], []
    for smi in df["smiles"]:
        d, p = compute_charges_dim(smi)
        drawn.append(d); ph74.append(p)
    df["formal_charge_drawn"] = drawn
    df["net_charge_pH74"] = ph74
    # MolGpKa
    MOLGPKA_SRC = Path("/tmp/MolGpKa/src")
    sys.path.insert(0, str(MOLGPKA_SRC))
    prev_cwd = os.getcwd()
    os.chdir(MOLGPKA_SRC)
    try:
        from predict_pka import predict as predict_fn
        frac, integer, pka_b = [], [], []
        for smi in df["smiles"]:
            f, n, p = compute_charges_mg(smi, predict_fn)
            frac.append(f); integer.append(n); pka_b.append(p)
        df["frac_charge_pH74"] = frac
        df["net_charge_pH74_mg"] = integer
        df["pKa_basic_max"] = pka_b
    finally:
        os.chdir(prev_cwd)
    save_working(df)
    update_state("1.2_charges", "done",
                 dim_nan=int(df["net_charge_pH74"].isna().sum()),
                 mg_nan=int(df["frac_charge_pH74"].isna().sum()))
    append_log(f"1.2 charges done ({time.time()-t0:.1f}s)")


# ─────────────────────────── 1.3 anchor wins ──────────────────────────────────

def phase_1_3():
    df = load_working()
    t0 = time.time()
    import torch
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    ENSEMBLE_DIR = ROOT / "results/paper_evaluation/reinvent4_film_ensemble"
    MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

    def morgan_fp(smi, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        return np.array(bv, dtype=np.float32)

    ensemble = []
    for k in range(3):
        ck = torch.load(ENSEMBLE_DIR / f"film_seed{k}.pt", map_location="cpu", weights_only=False)
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
    high_mask = ensemble[0]["anchor_pIC50"] >= 7.0
    append_log(f"1.3 ensemble loaded: {n_anchors} anchors, {int(high_mask.sum())} ≥7")

    def standardize(fps, ens):
        z = (fps - ens["scaler_mean"]) / ens["scaler_scale"]
        return torch.FloatTensor(z)

    # Mol1 baseline
    mol1_fp = morgan_fp(MOL1_SMI)
    mol1_baselines = []
    mol1_emb_per_seed = []
    for ens in ensemble:
        m1_e = standardize(mol1_fp.reshape(1, -1), ens)
        mol1_emb_per_seed.append(m1_e)
        with torch.no_grad():
            cand_t = m1_e.expand(n_anchors, -1)
            d = ens["model"](ens["anchor_embs"], cand_t).numpy()
            mol1_baselines.append(float((ens["anchor_pIC50"] + d).mean()))
    mol1_baseline = float(np.mean(mol1_baselines))

    smis = df["smiles"].tolist()
    fps_raw = np.stack([morgan_fp(s) for s in smis])
    cand_embs_per_seed = [standardize(fps_raw, ens) for ens in ensemble]
    n = len(smis)

    pIC50 = np.zeros(n); pIC50_std = np.zeros(n)
    wins = np.zeros(n); wins7 = np.zeros(n)
    direct = np.zeros(n); delta_vs = np.zeros(n)

    CHUNK = 32
    with torch.no_grad():
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            k = hi - lo
            ps_means, ps_deltas, ps_direct = [], [], []
            for si, ens in enumerate(ensemble):
                chunk_e = cand_embs_per_seed[si][lo:hi]
                a_tile = ens["anchor_embs"].unsqueeze(0).expand(k, -1, -1).reshape(k * n_anchors, -1)
                c_tile = chunk_e.unsqueeze(1).expand(-1, n_anchors, -1).reshape(k * n_anchors, -1)
                deltas = ens["model"](a_tile, c_tile).numpy().reshape(k, n_anchors)
                ps_deltas.append(deltas)
                ps_means.append((ens["anchor_pIC50"][None, :] + deltas).mean(axis=1))
                m1_a = mol1_emb_per_seed[si].expand(k, -1)
                d_direct = ens["model"](m1_a, chunk_e).numpy().flatten()
                ps_direct.append(d_direct)
            seed_means = np.array(ps_means)
            all_deltas = np.array(ps_deltas)
            seed_directs = np.array(ps_direct)
            mean = seed_means.mean(axis=0)
            std = seed_means.std(axis=0)
            w = (all_deltas > 0).sum(axis=2).mean(axis=0)
            w7 = ((all_deltas > 0) & high_mask[None, None, :]).sum(axis=2).mean(axis=0)
            d_mean = seed_directs.mean(axis=0)
            d_vs = mean - mol1_baseline
            for j in range(k):
                pIC50[lo + j] = mean[j]; pIC50_std[lo + j] = std[j]
                wins[lo + j] = w[j]; wins7[lo + j] = w7[j]
                direct[lo + j] = d_mean[j]; delta_vs[lo + j] = d_vs[j]

    # The input already has pIC50_mean, anchor_wins, anchor_wins_ge7 etc. — but with
    # the OLD scorer (possibly per-cohort). We OVERWRITE with the corrected scorer
    # output so the 78 numbers align with the 838's anchor_wins semantics.
    df["pIC50_mean"] = pIC50
    df["pIC50_std"] = pIC50_std
    df["pIC50_method"] = "FiLMDelta_ensemble3_anchor280"
    df["anchor_wins"] = wins
    df["anchor_wins_ge7"] = wins7
    df["delta_vs_mol1"] = delta_vs
    df["direct_delta_from_mol1"] = direct
    save_working(df)
    update_state("1.3_anchor_wins", "done",
                 mol1_baseline=mol1_baseline,
                 mean_pIC50=float(np.mean(pIC50)),
                 max_wins=int(wins.max()),
                 max_wins7=int(wins7.max()))
    append_log(f"1.3 anchor wins done ({time.time()-t0:.1f}s, mean pIC50 {np.mean(pIC50):.3f})")


# ─────────────────────────── 1.4 kinase classifier ─────────────────────────────

def phase_1_4():
    df = load_working()
    t0 = time.time()
    import torch
    from experiments.kinase_clf.featurize_and_train import (
        ThreeHeadFFN, featurize_morgan, featurize_chemberta, load_cache_dict,
        EMB_CACHE_MORGAN
    )
    MODEL_DIR = ROOT / "models" / "kinase_clf"
    ckpt = torch.load(MODEL_DIR / "trunk.pt", map_location="cpu")
    in_dim = ckpt["in_dim"]; pic_mu = ckpt["pic_mu"]; pic_sd = ckpt["pic_sd"]
    model = ThreeHeadFFN(in_dim=in_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(MODEL_DIR / "calibrators.pkl", "rb") as fh:
        cal = pickle.load(fh)
    platt_kin = cal["platt_kin"]; platt_tec = cal["platt_tec"]

    smiles = df["smiles"].tolist()
    morgan_lookup = load_cache_dict(EMB_CACHE_MORGAN, expected_dim=2048)
    X_morgan = featurize_morgan(smiles, morgan_lookup)
    X_cb = featurize_chemberta(smiles, batch_size=32)
    X = np.concatenate([X_morgan, X_cb], axis=1).astype(np.float32)

    with torch.no_grad():
        lk_all, lt_all, lp_all = [], [], []
        for b in range(0, len(X), 4096):
            xb = torch.from_numpy(X[b:b+4096])
            lk, lt, lp = model(xb)
            lk_all.append(lk.numpy()); lt_all.append(lt.numpy()); lp_all.append(lp.numpy())
        lk_all = np.concatenate(lk_all)
        lt_all = np.concatenate(lt_all)
        lp_all = np.concatenate(lp_all)

    p_kin = platt_kin.predict_proba(lk_all.reshape(-1, 1))[:, 1]
    if platt_tec is not None:
        p_tec = platt_tec.predict_proba(lt_all.reshape(-1, 1))[:, 1]
    else:
        p_tec = 1 / (1 + np.exp(-lt_all))
    pic50_pred = lp_all * pic_sd + pic_mu

    df["P_kinase"] = p_kin.astype(np.float32)
    df["P_Tec_family"] = p_tec.astype(np.float32)
    df["pIC50_kinase_aux"] = pic50_pred.astype(np.float32)

    # Percentile within the 78 cohort itself (rescue-78 lives in its own table)
    for col in ["P_kinase", "P_Tec_family", "pIC50_kinase_aux"]:
        df[f"{col}_pctl"] = df[col].rank(pct=True)

    save_working(df)
    update_state("1.4_kinase_clf", "done",
                 mean_P_kinase=float(np.mean(p_kin)),
                 mean_P_Tec=float(np.mean(p_tec)),
                 mean_pIC50_aux=float(np.mean(pic50_pred)))
    append_log(f"1.4 kinase clf done ({time.time()-t0:.1f}s)")


# ─────────────────────────── 1.5 3D shape ─────────────────────────────────────

def phase_1_5():
    df = load_working()
    t0 = time.time()
    from src.utils.mol1_scoring import (
        shape_tanimoto_seed, esp_sim_seed as esp_sim_fn, warhead_vector_deviation
    )
    sh, esp, wh = [], [], []
    for i, smi in enumerate(df["smiles"]):
        try:
            sh.append(shape_tanimoto_seed(smi))
        except Exception:
            sh.append(np.nan)
        try:
            esp.append(esp_sim_fn(smi))
        except Exception:
            esp.append(np.nan)
        try:
            wh.append(warhead_vector_deviation(smi))
        except Exception:
            wh.append(np.nan)
        if (i + 1) % 10 == 0:
            append_log(f"  1.5 progress {i+1}/{len(df)}")
    df["shape_Tc_seed"] = sh
    df["esp_sim_seed"] = esp
    df["warhead_dev_deg"] = wh
    save_working(df)
    update_state("1.5_3d_shape", "done",
                 nan_shape=int(pd.Series(sh).isna().sum()),
                 nan_esp=int(pd.Series(esp).isna().sum()),
                 nan_wh=int(pd.Series(wh).isna().sum()))
    append_log(f"1.5 3D shape done ({time.time()-t0:.1f}s)")


PHASES = {
    "1.1": phase_1_1,
    "1.2": phase_1_2,
    "1.3": phase_1_3,
    "1.4": phase_1_4,
    "1.5": phase_1_5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES) + ["all"])
    args = ap.parse_args()
    if args.phase == "all":
        for ph in PHASES:
            print(f"\n=== running phase {ph} ===")
            PHASES[ph]()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
