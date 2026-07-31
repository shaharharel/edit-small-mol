#!/usr/bin/env python3
"""Build eval_table.csv (per-fold) and leaderboard.csv (per-model) from all_results.json."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "zap70_challenge"
RESULTS_JSON = RESULTS_DIR / "all_results.json"
CKPT_DIR = RESULTS_DIR / "checkpoints"


def main():
    if not RESULTS_JSON.exists():
        raise SystemExit("No results yet.")
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    # Load fold definitions to compute n_train per fold
    fold_def_path = RESULTS_DIR / "fold_definitions.json"
    n_train_by_split_fold = {}
    if fold_def_path.exists():
        fd = json.loads(fold_def_path.read_text())
        for split_key, fold_list in fd.items():
            for f in fold_list:
                n_train_by_split_fold[(split_key, f["fold"])] = len(f["train_idx"])

    # Eval table: one row per (model, split_type, fold)
    rows = []
    for model, info in results.items():
        if not isinstance(info, dict) or "per_fold" not in info:
            continue
        for split_name, fold_list in info["per_fold"].items():
            split_type = "random_kfold" if split_name == "random" else "butina_groupkfold"
            for fm in fold_list:
                if "mae" not in fm:
                    continue
                fold_id = fm.get("fold", -1)
                n_train = n_train_by_split_fold.get((split_name, fold_id), fm.get("n_train", -1))
                rows.append({
                    "model": model, "split_type": split_type, "fold": fold_id,
                    "mae": fm.get("mae"), "rmse": fm.get("rmse"), "r2": fm.get("r2"),
                    "spearman": fm.get("spearman_r"), "pearson": fm.get("pearson_r"),
                    "n_train": n_train, "n_test": fm.get("n", -1),
                })
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(RESULTS_DIR / "eval_table.csv", index=False)
    print(f"[SAVE] eval_table.csv ({len(eval_df)} rows)")

    # Leaderboard: per model
    lb_rows = []
    for model, info in results.items():
        if not isinstance(info, dict) or "aggregated" not in info:
            continue
        agg = info["aggregated"]
        # Use butina if available, else random
        for split_name in ("butina", "random"):
            if split_name not in agg:
                continue
            a = agg[split_name]
            # Find weights file (full-trained ones)
            ckpt_candidates = []
            for ext in (".pt", ".pkl"):
                p = CKPT_DIR / f"{model.split('_repro')[0]}_full{ext}"
                if p.exists():
                    ckpt_candidates.append(str(p.relative_to(ROOT)))
            for ext in (".pt", ".pkl"):
                p = CKPT_DIR / f"{model}_full{ext}"
                if p.exists():
                    ckpt_candidates.append(str(p.relative_to(ROOT)))
            # Special case for Bootstrap (multiple files)
            if model == "7_FiLMDelta_Bootstrap":
                boot_files = sorted(CKPT_DIR.glob("7_FiLMDelta_Bootstrap_b*_full.pt"))
                if boot_files:
                    ckpt_candidates.append("results/zap70_challenge/checkpoints/7_FiLMDelta_Bootstrap_b{0-9}_full.pt")
            # v7 reproductions map to model number duplicates
            if model.startswith("v7A_"):
                p = CKPT_DIR / "1_FiLMDelta_full.pt"
                if p.exists(): ckpt_candidates.append(str(p.relative_to(ROOT)) + " (= 1_FiLMDelta)")
            if model.startswith("v7B_"):
                p = CKPT_DIR / "2_DirectMorganMLP_full.pt"
                if p.exists(): ckpt_candidates.append(str(p.relative_to(ROOT)) + " (= 2_DirectMorganMLP)")
            if model.startswith("v7D_"):
                p = CKPT_DIR / "3_MorganMLP_KinasePretrain_full.pt"
                if p.exists(): ckpt_candidates.append(str(p.relative_to(ROOT)) + " (= 3_MorganMLP_KinasePretrain)")
            if model.startswith("v7C_"):
                ckpt_candidates.append("(re-train XGB on interpretable features)")
            if model.startswith("v7H1"):
                ckpt_candidates.append("(non-parametric kNN; no weights to save)")
            weights_path = ckpt_candidates[0] if ckpt_candidates else ""
            lb_rows.append({
                "model": model, "split_type": "random_kfold" if split_name=="random" else "butina_groupkfold",
                "abs_mae_mean": a.get("mae_mean"), "abs_mae_std": a.get("mae_std"),
                "rmse_mean": a.get("rmse_mean"), "rmse_std": a.get("rmse_std"),
                "r2_mean": a.get("r2_mean"), "r2_std": a.get("r2_std"),
                "spearman_mean": a.get("spearman_r_mean"), "spearman_std": a.get("spearman_r_std"),
                "pearson_mean": a.get("pearson_r_mean"), "pearson_std": a.get("pearson_r_std"),
                "n_folds": a.get("n_folds"),
                "weights_path": weights_path,
            })
    lb_df = pd.DataFrame(lb_rows)
    # Sort by butina MAE primary
    butina = lb_df[lb_df["split_type"] == "butina_groupkfold"].sort_values("abs_mae_mean")
    random = lb_df[lb_df["split_type"] == "random_kfold"].sort_values("abs_mae_mean")
    lb_sorted = pd.concat([butina, random], ignore_index=True)
    lb_sorted.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    print(f"[SAVE] leaderboard.csv ({len(lb_sorted)} rows)")

    # Print short summary
    print("\n=== Top 5 by Butina MAE ===")
    print(butina.head(5).to_string(index=False))
    print("\n=== Top 5 by Random MAE ===")
    print(random.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
