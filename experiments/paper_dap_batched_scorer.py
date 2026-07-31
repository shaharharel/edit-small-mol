#!/usr/bin/env python3
"""Fast batched FiLM scoring of a CSV of SMILES. Loads model ONCE, batches all inference.

Usage:
    python paper_dap_batched_scorer.py --input samples.csv --output scored.csv [--column SMILES]

Writes scored_<tag>.csv with pIC50, qed, has_acrylamide, scaffold columns.
Emits a JSON report to --report <path>.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (same as reinvent4_film_scorer)
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/shaharh_quris_ai/edit-small-mol")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Reuse model definition
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

MODEL_CACHE = Path("/home/shaharh_quris_ai/edit-small-mol/results/paper_evaluation/reinvent4_film_model.pt")
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def load_model():
    print(f"[scorer] loading {MODEL_CACHE}", file=sys.stderr, flush=True)
    ckpt = torch.load(MODEL_CACHE, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
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


def batched_score(smiles, model, scaler, anchor_embs, anchor_pIC50, target_batch=64):
    """Batched FiLM anchor scoring. Returns list of floats aligned to smiles."""
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
            batch = embs[start : start + target_batch]  # (B, D)
            B = batch.shape[0]
            # Expand: (B, A, D)
            targets = batch.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            anchors = anchor_embs_t.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            deltas = model(anchors, targets).reshape(B, n_anchors)  # (B, A)
            abs_preds = anchor_pIC50_t.unsqueeze(0) + deltas  # broadcast
            means = abs_preds.mean(dim=1).numpy()  # (B,)
            for j, m in enumerate(means):
                scores[valid_indices[start + j]] = float(m)
            if (start + B) % (target_batch * 5) == 0:
                print(f"[scorer] {start+B}/{len(fps)} valid scored", file=sys.stderr, flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scored-csv", required=True)
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--target-batch", type=int, default=64)
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

    df["canon"] = df[smi_col].map(canon)
    df = df[df["canon"].notnull()].drop_duplicates(subset=["canon"]).reset_index(drop=True)
    if len(df) > args.max:
        df = df.iloc[: args.max].reset_index(drop=True)
    n = len(df)
    print(f"[dedup] {n} unique canonical", file=sys.stderr, flush=True)

    model, scaler, anchor_embs, anchor_pIC50 = load_model()
    scores = batched_score(df["canon"].tolist(), model, scaler, anchor_embs, anchor_pIC50, args.target_batch)
    df["pIC50"] = scores

    print("[qed]", file=sys.stderr, flush=True)
    df["qed"] = df["canon"].map(lambda s: QED.qed(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else float("nan"))
    df["has_acrylamide"] = df["canon"].map(
        lambda s: bool(Chem.MolFromSmiles(s).HasSubstructMatch(ACRYL_SMARTS)) if Chem.MolFromSmiles(s) else False
    )
    df["scaffold"] = df["canon"].map(scaffold)

    df.to_csv(args.scored_csv, index=False)

    p = np.asarray(df["pIC50"].values, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    top10 = np.sort(pv)[-10:]
    scaf_counts = df["scaffold"].value_counts(dropna=True)
    top_share = float(scaf_counts.iloc[0] / n) if len(scaf_counts) else 0.0

    report = {
        "tag": args.tag,
        "n_cohort": int(n),
        "n_valid_scored": int(valid.sum()),
        "pIC50_mean": float(np.mean(pv)),
        "pIC50_median": float(np.median(pv)),
        "pIC50_std": float(np.std(pv)),
        "pIC50_min": float(np.min(pv)),
        "pIC50_max": float(np.max(pv)),
        "frac_ge_7": float(np.mean(pv >= 7.0)),
        "frac_ge_6_5": float(np.mean(pv >= 6.5)),
        "top10_mean_pIC50": float(np.mean(top10)),
        "acrylamide_retention_frac": float(df["has_acrylamide"].mean()),
        "qed_mean": float(np.nanmean(df["qed"].values)),
        "top_scaffold_share": top_share,
        "n_unique_scaffolds": int(scaf_counts.shape[0]),
        "scored_csv": args.scored_csv,
    }
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
