#!/usr/bin/env python3
"""FiLM Tail-Lift Analysis: fraction of samples with predicted pIC50 >= 7.0.

Scores 4 cohorts (base, covft, v1-DAP, v2-cond) with the OFFICIAL FiLM scorer
(reinvent4_film_model_clean.pt) and reports median/mean/frac(>=7.0)/frac(>=7.5)/N_valid.

Optionally accepts a second checkpoint path via --retrain_ckpt to compare a retrained
model side-by-side (Part 2 of the workflow).

Output: results/paper_evaluation/film_tail_analysis.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"

COHORTS = {
    "base":         PROJECT_ROOT / "experiments" / "exp_covft_value" / "samples_base.csv",
    "covft":        PROJECT_ROOT / "experiments" / "exp_covft_value" / "samples_covft.csv",
    "v1_DAP":       PROJECT_ROOT / "experiments" / "exp_geom_bc"     / "samples_rl.csv",
    "v2_cond":      PROJECT_ROOT / "data" / "m1a_v2_ablation"        / "cohort_A_10k.csv",
    "v2_cond_DAP":  PROJECT_ROOT / "data" / "paper_dap_repro"        / "samples_v2cond_E2_10k.csv",
}


def load_ckpt(path: Path, device):
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck['scaler_mean']; sc.scale_ = ck['scaler_scale']
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    ae = ck['anchor_embs']
    if not isinstance(ae, torch.Tensor):
        ae = torch.FloatTensor(ae)
    ae = ae.to(device)
    ap = np.asarray(ck['anchor_pIC50'])
    val_mae = ck.get('val_mae_mol_disjoint')
    return m, sc, ae, ap, val_mae


def compute_fp(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def score_batch(smiles_list, model, scaler, anchor_embs, anchor_pIC50, device, batch=64):
    """Vectorized anchor-mean scoring: pred(j) = mean_i(pIC50(i) + delta(i->j))."""
    n_anchors = len(anchor_pIC50)
    preds = np.full(len(smiles_list), np.nan, dtype=np.float64)

    # Compute fingerprints
    valid_idx, valid_fps = [], []
    for i, smi in enumerate(smiles_list):
        fp = compute_fp(smi)
        if fp is not None:
            valid_idx.append(i)
            valid_fps.append(fp)
    if not valid_fps:
        return preds

    fps_arr = np.array(valid_fps)
    X = torch.FloatTensor(scaler.transform(fps_arr)).to(device)

    ap_t = torch.FloatTensor(anchor_pIC50).to(device)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch]                       # (B, 2048)
            B = xb.shape[0]
            # broadcast to (B*n_anchors, 2048)
            a_rep = anchor_embs.unsqueeze(0).expand(B, -1, -1).reshape(B * n_anchors, -1)
            b_rep = xb.unsqueeze(1).expand(-1, n_anchors, -1).reshape(B * n_anchors, -1)
            deltas = model(a_rep, b_rep).view(B, n_anchors)
            abs_preds = ap_t.unsqueeze(0) + deltas   # (B, n_anchors)
            mean_preds = abs_preds.mean(dim=1).cpu().numpy()
            for k, orig_i in enumerate(valid_idx[s:s+batch]):
                preds[orig_i] = float(mean_preds[k])
    return preds


def summarize(preds: np.ndarray) -> dict:
    valid = preds[~np.isnan(preds)]
    if len(valid) == 0:
        return {"n": int(len(preds)), "n_valid": 0}
    return {
        "n": int(len(preds)),
        "n_valid": int(len(valid)),
        "median": float(np.median(valid)),
        "mean":   float(np.mean(valid)),
        "p25":    float(np.percentile(valid, 25)),
        "p75":    float(np.percentile(valid, 75)),
        "max":    float(np.max(valid)),
        "frac_ge_7":  float(np.mean(valid >= 7.0)),
        "frac_ge_75": float(np.mean(valid >= 7.5)),
        "frac_ge_8":  float(np.mean(valid >= 8.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="Path to FiLM checkpoint (default: clean).")
    ap.add_argument("--label", type=str, default="current",
                    help="Label for this run in the output JSON.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "results" / "paper_evaluation" / "film_tail_analysis.json")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    print(f"[tail] device={device}", file=sys.stderr, flush=True)
    print(f"[tail] Loading checkpoint: {args.ckpt}", file=sys.stderr, flush=True)
    m, sc, ae, ap_arr, val_mae = load_ckpt(args.ckpt, device)
    print(f"[tail] Anchors: {len(ap_arr)}, val_mae_mol_disjoint: {val_mae}", file=sys.stderr, flush=True)

    results = {
        "ckpt": str(args.ckpt),
        "val_mae_mol_disjoint": val_mae,
        "n_anchors": int(len(ap_arr)),
        "cohorts": {},
    }
    for name, csv_path in COHORTS.items():
        t0 = time.time()
        df = pd.read_csv(csv_path)
        smiles = df["SMILES"].tolist()
        print(f"[tail] {name}: scoring {len(smiles):,} SMILES from {csv_path.name}...",
              file=sys.stderr, flush=True)
        preds = score_batch(smiles, m, sc, ae, ap_arr, device, batch=args.batch)
        stats = summarize(preds)
        stats["elapsed_s"] = time.time() - t0
        results["cohorts"][name] = stats
        print(f"[tail] {name}: n_valid={stats.get('n_valid','?')} "
              f"median={stats.get('median','?'):.3f} "
              f"frac>=7={stats.get('frac_ge_7','?'):.3%} "
              f"frac>=7.5={stats.get('frac_ge_75','?'):.3%} "
              f"({stats['elapsed_s']:.1f}s)",
              file=sys.stderr, flush=True)

    # Merge into existing JSON if present, keyed by label
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            all_results = json.loads(out.read_text())
        except Exception:
            all_results = {}
    else:
        all_results = {}
    all_results[args.label] = results
    out.write_text(json.dumps(all_results, indent=2))
    print(f"[tail] Saved to {out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
