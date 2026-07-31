#!/usr/bin/env python3
"""Full metric panel scorer for v2-cond RL cohorts using CLEAN FiLM predictor.

Computes:
  - validity, dedup, scaffold uniqueness, top-scaffold share
  - QED mean
  - pIC50 mean/median/p95/top-10-mean/frac>=7 (from reinvent4_film_model_clean.pt)
  - acryl SMARTS on LARGEST fragment (headline metric)
  - planar dihedral (RDKit ETKDG, sample-based) — median + frac<=30deg
  - pre-reactivity (frac dihedral <=30 in vacuum)
  - Tc to Mol1 median

Usage:
  python paper_dap_metric_panel_clean.py \
    --tag v2cond_E1 \
    --samples data/paper_dap_repro/samples_v2cond_E1_10k.csv \
    --report data/paper_dap_repro/report_v2cond_E1.json \
    --scored-csv data/paper_dap_repro/samples_v2cond_E1_10k_scored.csv \
    --planar-sample 1000
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, "/Users/shaharharel/Documents/github/edit-small-mol")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

_CANDIDATES_CLEAN = [
    Path("/home/shaharh_quris_ai/edit-small-mol/results/paper_evaluation/reinvent4_film_model_clean.pt"),
    Path("/Users/shaharharel/Documents/github/edit-small-mol/results/paper_evaluation/reinvent4_film_model_clean.pt"),
]
MODEL_CACHE_CLEAN = next((p for p in _CANDIDATES_CLEAN if p.exists()), _CANDIDATES_CLEAN[0])
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def load_clean_model():
    """Load reinvent4_film_model_clean.pt.

    The 'clean' variant is a multi-task FiLM predictor trained on 280 anchors.
    Same interface as reinvent4_film_model.pt.
    """
    print(f"[scorer] loading {MODEL_CACHE_CLEAN}", file=sys.stderr, flush=True)
    ckpt = torch.load(MODEL_CACHE_CLEAN, map_location="cpu", weights_only=False)
    # Try common architectures; the clean model is FiLMDeltaMLP based on the naming convention.
    input_dim = 2048
    hidden_dims = [1024, 512, 256]
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
        input_dim = cfg.get("input_dim", input_dim)
        hidden_dims = cfg.get("hidden_dims", hidden_dims)
    model = FiLMDeltaMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    anchor_embs = ckpt["anchor_embs"]
    anchor_pIC50 = ckpt["anchor_pIC50"]
    print(f"[scorer] {len(anchor_pIC50)} anchors", file=sys.stderr, flush=True)
    return model, scaler, anchor_embs, anchor_pIC50


def canon(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def largest_fragment_smi(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    largest = max(frags, key=lambda mm: mm.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


def acryl_on_largest(smi):
    lf = largest_fragment_smi(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return bool(m.GetSubstructMatch(ACRYL_SMARTS))


def scaffold(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def compute_fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fp_bv(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def batched_score(smiles, model, scaler, anchor_embs, anchor_pIC50, target_batch=64):
    n = len(smiles)
    fps = []
    valid_indices = []
    for i, s in enumerate(smiles):
        fp = compute_fp(s)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    scores = [float("nan")] * n
    if not fps:
        return scores
    fps_arr = np.stack(fps)
    embs = torch.from_numpy(scaler.transform(fps_arr).astype(np.float32))
    n_anchors = len(anchor_pIC50)
    anchor_embs_t = torch.as_tensor(anchor_embs, dtype=torch.float32)
    anchor_pIC50_t = torch.as_tensor(anchor_pIC50, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, embs.shape[0], target_batch):
            batch = embs[start : start + target_batch]
            B = batch.shape[0]
            targets = batch.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            anchors = anchor_embs_t.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            deltas = model(anchors, targets).reshape(B, n_anchors)
            abs_preds = anchor_pIC50_t.unsqueeze(0) + deltas
            means = abs_preds.mean(dim=1).numpy()
            for j, mv in enumerate(means):
                scores[valid_indices[start + j]] = float(mv)
            if (start + B) % (target_batch * 20) == 0:
                print(f"[scorer] {start+B}/{len(fps)} valid scored", file=sys.stderr, flush=True)
    return scores


def compute_planar_dihedral_and_prereact(smi_list, seed=42):
    """For each SMILES compute the C=C-C(=O)-N torsion using ETKDG single conformer.

    Returns arrays: dihedrals_deg (NaN for failure), prereact_bool (True if |phi|<=30).
    """
    n = len(smi_list)
    dihs = np.full(n, np.nan, dtype=np.float64)
    for i, smi in enumerate(smi_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            continue
        a1, a2, a3, _o, a4 = matches[0]
        try:
            mH = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            cid = AllChem.EmbedMolecule(mH, params)
            if cid < 0:
                continue
            conf = mH.GetConformer(cid)
            phi = AllChem.GetDihedralDeg(conf, a1, a2, a3, a4)
            # Wrap and reflect: min(|phi|, |180-|phi||)
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            d = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
            dihs[i] = d
        except Exception:
            continue
    prereact = np.where(np.isfinite(dihs), dihs <= 30.0, False)
    return dihs, prereact


def tc_to_mol1(smi_list):
    mol1 = Chem.MolFromSmiles(MOL1_SMI)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    tcs = np.full(len(smi_list), np.nan, dtype=np.float64)
    for i, s in enumerate(smi_list):
        fp = compute_fp_bv(s)
        if fp is None:
            continue
        tcs[i] = DataStructs.TanimotoSimilarity(mol1_fp, fp)
    return tcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scored-csv", required=True)
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--target-batch", type=int, default=64)
    ap.add_argument("--planar-sample", type=int, default=1000,
                    help="Subsample size for planar dihedral (ETKDG is slow).")
    args = ap.parse_args()

    df = pd.read_csv(args.samples)
    smi_col = None
    for c in ["SMILES", "smiles", "Smiles"]:
        if c in df.columns:
            smi_col = c
            break
    if smi_col is None:
        smi_col = df.columns[0]
    print(f"[load] {args.samples}: {len(df)} rows, using '{smi_col}'", file=sys.stderr, flush=True)

    n_raw = len(df)
    df["canon"] = df[smi_col].map(canon)
    n_valid_raw = df["canon"].notnull().sum()
    df = df[df["canon"].notnull()].drop_duplicates(subset=["canon"]).reset_index(drop=True)
    if len(df) > args.max:
        df = df.iloc[: args.max].reset_index(drop=True)
    n = len(df)
    print(f"[dedup] {n} unique canonical (of {n_valid_raw} valid / {n_raw} raw)", file=sys.stderr, flush=True)

    model, scaler, anchor_embs, anchor_pIC50 = load_clean_model()
    scores = batched_score(df["canon"].tolist(), model, scaler, anchor_embs, anchor_pIC50, args.target_batch)
    df["pIC50"] = scores

    print("[qed]", file=sys.stderr, flush=True)
    df["qed"] = df["canon"].map(lambda s: QED.qed(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else float("nan"))
    df["has_acrylamide"] = df["canon"].map(
        lambda s: bool(Chem.MolFromSmiles(s).HasSubstructMatch(ACRYL_SMARTS)) if Chem.MolFromSmiles(s) else False
    )
    df["acryl_largest"] = df["canon"].map(acryl_on_largest)
    df["scaffold"] = df["canon"].map(scaffold)

    print(f"[tc-mol1] computing Tc to Mol1 for {n}...", file=sys.stderr, flush=True)
    df["tc_to_mol1"] = tc_to_mol1(df["canon"].tolist())

    # Planar dihedral (subsample if slow)
    plan_n = min(args.planar_sample, n)
    plan_idx = np.random.default_rng(0).choice(n, size=plan_n, replace=False)
    plan_smi = df["canon"].iloc[plan_idx].tolist()
    print(f"[planar] ETKDG on {plan_n} mols...", file=sys.stderr, flush=True)
    plan_dihs, plan_prereact = compute_planar_dihedral_and_prereact(plan_smi)

    df.to_csv(args.scored_csv, index=False)

    p = np.asarray(df["pIC50"].values, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    top10 = np.sort(pv)[-10:]
    scaf_counts = df["scaffold"].value_counts(dropna=True)
    top_share = float(scaf_counts.iloc[0] / n) if len(scaf_counts) else 0.0

    plan_finite = plan_dihs[np.isfinite(plan_dihs)]
    tc_arr = df["tc_to_mol1"].dropna().values

    report = {
        "tag": args.tag,
        "n_raw": int(n_raw),
        "n_valid_raw": int(n_valid_raw),
        "n_cohort": int(n),
        "validity_frac": float(n_valid_raw / max(n_raw, 1)),
        "n_valid_scored": int(valid.sum()),
        # pIC50
        "pIC50_mean": float(np.mean(pv)),
        "pIC50_median": float(np.median(pv)),
        "pIC50_p95": float(np.percentile(pv, 95)),
        "pIC50_std": float(np.std(pv)),
        "pIC50_min": float(np.min(pv)),
        "pIC50_max": float(np.max(pv)),
        "frac_ge_7": float(np.mean(pv >= 7.0)),
        "frac_ge_6_5": float(np.mean(pv >= 6.5)),
        "top10_mean_pIC50": float(np.mean(top10)),
        # Warhead
        "acrylamide_retention_frac": float(df["has_acrylamide"].mean()),
        "acryl_largest_frag_pct": float(df["acryl_largest"].mean()),
        # QED
        "qed_mean": float(np.nanmean(df["qed"].values)),
        # Scaffold
        "top_scaffold_share": top_share,
        "n_unique_scaffolds": int(scaf_counts.shape[0]),
        "scaffold_uniqueness": float(scaf_counts.shape[0] / max(n, 1)),
        # Planar / pre-reactivity
        "planar_dihedral_n_computed": int(len(plan_finite)),
        "planar_dihedral_median_deg": float(np.median(plan_finite)) if len(plan_finite) else None,
        "planar_dihedral_mean_deg": float(np.mean(plan_finite)) if len(plan_finite) else None,
        "prereact_frac_le_30deg": float(np.mean(plan_prereact)) if plan_n > 0 else None,
        "planar_sample_size": plan_n,
        # Tc to Mol1
        "tc_to_mol1_median": float(np.median(tc_arr)) if len(tc_arr) else None,
        "tc_to_mol1_mean": float(np.mean(tc_arr)) if len(tc_arr) else None,
        # Hard fail marker
        "hard_fail_acryl_lt_70pct": bool(float(df["acryl_largest"].mean()) < 0.70),
        "scored_csv": args.scored_csv,
    }
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
