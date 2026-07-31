#!/usr/bin/env python3
"""Post-hoc ensemble analysis: combine top-2 models per split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "zap70_challenge"
PREDS_DIR = RESULTS_DIR / "per_model_logs"


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": float(sr), "pearson": float(pr)}


def load_preds(model_name: str) -> dict:
    f = PREDS_DIR / f"{model_name}_preds.json"
    if not f.exists():
        return {}
    with open(f) as fp:
        return json.load(fp)


def ensemble_two(model_a: str, model_b: str, weights=(0.5, 0.5)) -> dict:
    """Average predictions per fold."""
    a = load_preds(model_a); b = load_preds(model_b)
    if not a or not b: return {}
    out = {"random": [], "butina": []}
    for split in ("random", "butina"):
        for fi in range(5):
            try:
                pa = a["predictions"][split][fi]; pb = b["predictions"][split][fi]
                assert pa["test_idx"] == pb["test_idx"]
                y = np.array(pa["y_true"])
                preds = weights[0] * np.array(pa["preds"]) + weights[1] * np.array(pb["preds"])
                m = compute_metrics(y, preds); m["fold"] = fi
                out[split].append(m)
            except Exception as e:
                print(f"  skip {split} fold {fi}: {e}")
    return out


def agg(folds):
    if not folds: return {}
    keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float)) and k != "fold"]
    out = {}
    for k in keys:
        vals = [m[k] for m in folds]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def ensemble_n(models: list, weights: list) -> dict:
    preds_per_model = [load_preds(m) for m in models]
    if any(not p for p in preds_per_model): return {}
    out = {"random": [], "butina": []}
    for split in ("random", "butina"):
        for fi in range(5):
            try:
                fold_data = [p["predictions"][split][fi] for p in preds_per_model]
                idx = fold_data[0]["test_idx"]
                assert all(fd["test_idx"] == idx for fd in fold_data)
                y = np.array(fold_data[0]["y_true"])
                preds = np.zeros(len(y))
                w_sum = sum(weights)
                for w, fd in zip(weights, fold_data):
                    preds += (w / w_sum) * np.array(fd["preds"])
                m = compute_metrics(y, preds); m["fold"] = fi
                out[split].append(m)
            except Exception as e:
                print(f"  skip {split} fold {fi}: {e}")
    return out


def main():
    pairs = [
        ("11_FiLMDelta_KinasePretrain", "10_XGB_MultiFP"),
        ("1_FiLMDelta", "10_XGB_MultiFP"),
        ("11_FiLMDelta_KinasePretrain", "1_FiLMDelta"),
        ("1_FiLMDelta", "v7G1_XGB_Morgan_repro"),
        ("6_DeepDelta", "10_XGB_MultiFP"),
        ("1_FiLMDelta", "6_DeepDelta"),
        ("1_FiLMDelta", "v7C_XGB_Interpretable_repro"),
        ("11_FiLMDelta_KinasePretrain", "v7C_XGB_Interpretable_repro"),
    ]
    triples = [
        ("11_FiLMDelta_KinasePretrain", "10_XGB_MultiFP", "v7G1_XGB_Morgan_repro"),
        ("1_FiLMDelta", "10_XGB_MultiFP", "v7G1_XGB_Morgan_repro"),
        ("1_FiLMDelta", "6_DeepDelta", "10_XGB_MultiFP"),
        ("11_FiLMDelta_KinasePretrain", "1_FiLMDelta", "10_XGB_MultiFP"),
        ("11_FiLMDelta_KinasePretrain", "6_DeepDelta", "10_XGB_MultiFP"),
        ("1_FiLMDelta", "10_XGB_MultiFP", "v7C_XGB_Interpretable_repro"),
    ]
    results = {}
    for a, b in pairs:
        name = f"ENSEMBLE({a}+{b}, 0.5/0.5)"
        print(f"\n{name}")
        out = ensemble_two(a, b)
        if not out: print("  ⚠ missing preds"); continue
        for split, folds in out.items():
            a_agg = agg(folds)
            print(f"  [{split}] MAE={a_agg.get('mae_mean','?'):.4f}±{a_agg.get('mae_std','?'):.4f}, "
                  f"Spr={a_agg.get('spearman_mean','?'):.3f}, R²={a_agg.get('r2_mean','?'):.3f}")
        results[name] = {split: agg(folds) for split, folds in out.items()}

    for trio in triples:
        name = f"ENSEMBLE({'+'.join(trio)}, equal)"
        print(f"\n{name}")
        out = ensemble_n(list(trio), [1, 1, 1])
        if not out: print("  ⚠ missing preds"); continue
        for split, folds in out.items():
            a_agg = agg(folds)
            print(f"  [{split}] MAE={a_agg.get('mae_mean','?'):.4f}±{a_agg.get('mae_std','?'):.4f}, "
                  f"Spr={a_agg.get('spearman_mean','?'):.3f}, R²={a_agg.get('r2_mean','?'):.3f}")
        results[name] = {split: agg(folds) for split, folds in out.items()}

    # Also try 0.4/0.6 and 0.6/0.4 weights for the top pair
    print("\n--- Weight sweep for FiLMDelta(+pretrain) + XGB MultiFP ---")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        out = ensemble_two("11_FiLMDelta_KinasePretrain", "10_XGB_MultiFP", weights=(w, 1 - w))
        for split, folds in out.items():
            a_agg = agg(folds)
            print(f"  w_film11={w}, [{split}]: MAE={a_agg.get('mae_mean','?'):.4f}, Spr={a_agg.get('spearman_mean','?'):.3f}")
        results[f"FiLM11_w{w}_XGB_w{1-w:.1f}"] = {split: agg(folds) for split, folds in out.items()}

    out_f = RESULTS_DIR / "ensemble_analysis.json"
    out_f.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVE] {out_f}")


if __name__ == "__main__":
    main()
