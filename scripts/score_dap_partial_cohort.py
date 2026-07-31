"""Score the partial DAP same-infra cohort using identical scoring logic as
other PPO/DPO cohorts (ladder warhead + thiq_exact + FiLM_pIC50 + QED + composite_v2).

Ports the relevant pieces from data/exp_ppo_v2plus/film_rest_server_v2.py
(SMARTS_LADDER, _ladder_score, _thiq_match, composite_v2) into a one-shot
batch scorer over a CSV with a SMILES column.

Writes:
  - <out_csv> with columns: SMILES, Input_SMILES, valid, ladder, thiq_exact,
    warhead_any, film_pIC50, QED, composite_v2
  - <out_csv>.summary.json  matching the schema in cohort_*.summary.json

Usage:
    python scripts/score_dap_partial_cohort.py \
        --in_csv  data/exp_ppo_v2plus/cohort_dap_same_infra_partial.csv \
        --out_csv data/exp_ppo_v2plus/cohort_dap_same_infra_partial.csv \
        --model_pt results/paper_evaluation/reinvent4_film_model.pt \
        --src_root .
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

# -- DAP-matched composite parameters (mirrors film_rest_server_v2.py) ---------
SIGMOID_LOW = 5.5
SIGMOID_HIGH = 7.5
SIGMOID_K = 0.5
W_FILM = 0.50
W_THIQ = 0.40
W_QED = 0.10

# SMARTS ladder used by composite_v2 / "ladder" predictor_id on the remote v2 server
# (extracted from /health endpoint of film_rest_server_v2.py running on a100-b)
SMARTS_LADDER = [
    ("C=CC(=O)N1Cc2ccccc2C1", 1.0),       # THIQ-acrylamide (exact match)
    ("C(=O)C=C[#7]", 0.75),                # acrylamide bonded to N
    ("C=CC(=O)N", 0.50),                   # generic acrylamide
    ("[CX3]=[CX3][CX3]=[OX1]", 0.25),      # any Michael acceptor
]
_LADDER_PATTERNS = [(Chem.MolFromSmarts(s), v) for s, v in SMARTS_LADDER]
_THIQ_PATTERN = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
TAIL_WEIGHT = 0.15  # ladder mode: composite = ladder + tail_weight * (sat tail)


def _double_sigmoid(x: float, low=SIGMOID_LOW, high=SIGMOID_HIGH, k=SIGMOID_K) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    mid = 0.5 * (low + high)
    span = (high - low) if high > low else 1.0
    z = (x - mid) / (span / 4.0)
    return float(1.0 / (1.0 + math.exp(-z * 4.0 * k)))


def _qed_score(mol) -> float:
    try:
        return float(QED.qed(mol)) if mol is not None else 0.0
    except Exception:
        return 0.0


def _thiq_match(mol) -> float:
    if mol is None or _THIQ_PATTERN is None:
        return 0.0
    return 1.0 if mol.HasSubstructMatch(_THIQ_PATTERN) else 0.0


def _ladder_score(mol) -> float:
    if mol is None:
        return 0.0
    for pat, val in _LADDER_PATTERNS:
        if pat is not None and mol.HasSubstructMatch(pat):
            return float(val)
    return 0.0


def _warhead_any(mol) -> int:
    """1 if ladder hit at any level (i.e. ladder > 0)."""
    return 1 if _ladder_score(mol) > 0.0 else 0


def _geom_mean(values, weights) -> float:
    eps = 1e-9
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0
    log_acc = 0.0
    for v, w in zip(values, weights):
        if v <= 0.0:
            return 0.0
        log_acc += (w / w_sum) * float(np.log(max(v, eps)))
    return float(np.exp(log_acc))


def _composite_v2_ladder(film_pic50: float, mol) -> float:
    """v2 composite (ladder mode): geometric_mean(film_sigmoid, ladder_smarts, QED)."""
    if mol is None:
        return 0.0
    film_t = _double_sigmoid(film_pic50)
    war_t = _ladder_score(mol)
    qed_t = _qed_score(mol)
    return _geom_mean([film_t, war_t, qed_t], [W_FILM, W_THIQ, W_QED])


# -- FiLM scoring (ported from film_rest_server_slim.py) -----------------------
def load_film_model(model_pt: str, src_root: str):
    sys.path.insert(0, src_root)
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    anchor_embs = ckpt["anchor_embs"]
    anchor_pic50 = ckpt["anchor_pIC50"]
    print(f"[score] FiLM loaded: {len(anchor_pic50)} anchors", flush=True)
    return model, scaler, anchor_embs, anchor_pic50


def score_film_batch(smiles_list, model, scaler, anchor_embs, anchor_pic50, batch_log=500):
    """Return list of pIC50 floats (NaN for invalid). Anchor-mean prediction."""
    scores = [float("nan")] * len(smiles_list)
    valid_indices = []
    valid_fps = []

    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            valid_fps.append(arr)
            valid_indices.append(i)
        except Exception:
            continue

    if not valid_fps:
        return scores

    fps_arr = np.asarray(valid_fps)
    batch_embs = torch.FloatTensor(scaler.transform(fps_arr))
    n_anchors = len(anchor_pic50)
    anchor_pic50_arr = np.asarray(anchor_pic50)

    with torch.no_grad():
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            target_emb = batch_embs[idx_in_valid:idx_in_valid + 1].expand(n_anchors, -1)
            deltas = model(anchor_embs, target_emb).numpy().flatten()
            abs_preds = anchor_pic50_arr + deltas
            scores[orig_idx] = float(np.mean(abs_preds))
            if (idx_in_valid + 1) % batch_log == 0:
                print(f"  FiLM scored {idx_in_valid + 1}/{len(valid_indices)}", flush=True)

    return scores


# -- Scaffolds (for parity with cohort_*.summary.json) -------------------------
def _scaffold_smiles(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff)
    except Exception:
        return ""


def _internal_mean_tc(smiles_list, n=500, seed=0):
    """Mean pairwise Tanimoto of first `n` valid SMILES (matches other scripts)."""
    rng = np.random.default_rng(seed)
    fps = []
    for smi in smiles_list:
        if not isinstance(smi, str):
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        if len(fps) >= n:
            break
    if len(fps) < 2:
        return 0.0
    tcs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            tcs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(tcs)) if tcs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--input_smiles_col", default="Input_SMILES")
    args = ap.parse_args()

    print(f"[score] reading {args.in_csv}", flush=True)
    df = pd.read_csv(args.in_csv)
    print(f"[score]   {len(df)} rows, columns={list(df.columns)}", flush=True)

    smiles = df[args.smiles_col].astype(str).fillna("").tolist()
    n_total = len(smiles)

    # Validity + per-mol substructure scores
    print("[score] computing validity / ladder / thiq / QED ...", flush=True)
    valids = []
    ladders = []
    thiqs = []
    warhead_anys = []
    qeds = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if smi else None
        ok = 1 if m is not None else 0
        valids.append(ok)
        ladders.append(_ladder_score(m) if ok else 0.0)
        thiqs.append(int(_thiq_match(m)) if ok else 0)
        warhead_anys.append(_warhead_any(m) if ok else 0)
        qeds.append(_qed_score(m) if ok else 0.0)

    valid_mask = np.array(valids, dtype=bool)
    n_valid = int(valid_mask.sum())
    print(f"[score]   validity {n_valid}/{n_total} ({n_valid/n_total:.1%})", flush=True)

    # FiLM scoring
    model, scaler, anchor_embs, anchor_pic50 = load_film_model(args.model_pt, args.src_root)
    print(f"[score] scoring FiLM pIC50 over {n_total} mols ...", flush=True)
    film_pic50 = score_film_batch(smiles, model, scaler, anchor_embs, anchor_pic50)

    # Composite v2 (ladder mode, same as v2 server's `predictor_id=ladder`)
    composites = []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi) if smi else None
        fp = film_pic50[i] if film_pic50[i] is not None and np.isfinite(film_pic50[i]) else 0.0
        composites.append(_composite_v2_ladder(fp, m))

    # Build canonical cohort dataframe
    out_cols = {
        "SMILES": df[args.smiles_col],
        "valid": valids,
        "ladder": ladders,
        "thiq_exact": thiqs,
        "warhead_any": warhead_anys,
        "film_pIC50": film_pic50,
        "QED": qeds,
        "composite_v2": composites,
    }
    if args.input_smiles_col in df.columns:
        out_cols["Input_SMILES"] = df[args.input_smiles_col]
    # Pass through DAP-specific columns for traceability
    for c in ("Agent", "Prior", "Target", "Score", "Scaffold", "step",
              "FiLM_composite_ladder", "FiLM_composite_ladder (raw)"):
        if c in df.columns:
            out_cols[c] = df[c]
    out_df = pd.DataFrame(out_cols)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[score] wrote {args.out_csv}", flush=True)

    # Summary JSON matching schema used by other cohort_*.summary.json files
    valid_film = np.array([f for f in film_pic50 if f is not None and np.isfinite(f)],
                          dtype=np.float64)
    valid_qed = np.array(qeds, dtype=np.float64)[valid_mask]
    valid_composite = np.array(composites, dtype=np.float64)[valid_mask]
    valid_ladder = np.array(ladders, dtype=np.float64)[valid_mask]
    valid_warhead = np.array(warhead_anys, dtype=np.int64)[valid_mask]

    valid_smiles = [smiles[i] for i in range(n_total) if valid_mask[i]]
    print("[score] computing scaffolds ...", flush=True)
    scaffs = [_scaffold_smiles(s) for s in valid_smiles]
    uniq_scaffs = set(s for s in scaffs if s)
    n_uniq_scaff = len(uniq_scaffs)
    scaff_per_mol = (n_uniq_scaff / max(n_valid, 1))

    print("[score] computing internal_mean_tc_sub500 ...", flush=True)
    internal_mean_tc = _internal_mean_tc(valid_smiles, n=500)

    summary = {
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else 0.0,
        "thiq_exact_rate": float(np.mean(np.array(thiqs)[valid_mask])) if n_valid else 0.0,
        "warhead_any_rate": float(np.mean(valid_warhead)) if n_valid else 0.0,
        "mean_ladder": float(np.mean(valid_ladder)) if n_valid else 0.0,
        "mean_film_pIC50": float(np.mean(valid_film)) if len(valid_film) else 0.0,
        "median_film_pIC50": float(np.median(valid_film)) if len(valid_film) else 0.0,
        "max_film_pIC50": float(np.max(valid_film)) if len(valid_film) else 0.0,
        "frac_film_ge_7": float(np.mean(valid_film >= 7.0)) if len(valid_film) else 0.0,
        "frac_film_ge_8": float(np.mean(valid_film >= 8.0)) if len(valid_film) else 0.0,
        "mean_QED": float(np.mean(valid_qed)) if n_valid else 0.0,
        "mean_composite_v2": float(np.mean(valid_composite)) if n_valid else 0.0,
        "n_unique_scaffolds": int(n_uniq_scaff),
        "scaffolds_per_mol_valid": float(scaff_per_mol),
        "internal_mean_tc_sub500": float(internal_mean_tc),
        "ckpt": "models/rl_checkpoints_b/dap_zap70_partial (OOM @ step 12/50)",
        "seeds": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi",
        "note": ("Partial DAP same-infra cohort. OOM @ step 12 on A100-40GB "
                 "(batch=4 after retries from 16->8->4). Rows are DURING-RL sampled "
                 "mols across steps 1-12, NOT post-RL sampling like other cohorts. "
                 "Not directly comparable to fully-trained checkpoints sampled with "
                 "sample_and_analyze.py."),
    }
    if int(np.sum(valid_warhead)) > 0:
        wf = valid_film[valid_warhead[: len(valid_film)].astype(bool)] \
            if len(valid_film) == len(valid_warhead) else np.array([])
        # Robust fallback: compute warhead-positive metrics over valid rows only
        wmask = valid_warhead.astype(bool)
        wf = np.array([film_pic50[i] for i, v in enumerate(valid_mask) if v],
                      dtype=np.float64)[wmask]
        wf = wf[np.isfinite(wf)]
        wq = valid_qed[wmask]
        summary["warhead_positive"] = {
            "n": int(wmask.sum()),
            "mean_film_pIC50": float(np.mean(wf)) if len(wf) else 0.0,
            "mean_QED": float(np.mean(wq)) if len(wq) else 0.0,
            "frac_film_ge_7": float(np.mean(wf >= 7.0)) if len(wf) else 0.0,
        }

    out_json = args.out_csv + ".summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[score] wrote {out_json}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
