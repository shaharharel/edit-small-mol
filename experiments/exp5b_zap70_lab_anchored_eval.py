#!/usr/bin/env python3
"""
EXP5b ZAP70 LAB-ANCHORED EVAL — does FiLMDelta's "lab-bias-clean parameters"
property pay off when the chemist has only 3 in-house anchor measurements?

Setup (lab_disjoint only — the realistic regime):
  Hold out lab E's pairs as test.
  Train 3 predictors x 2 pretrain regimes on remaining labs.
  From lab E's UNIQUE test molecules, pick 3 random "in-house anchors".
  Predict abs pIC50 of remaining lab E test molecules using ONLY the 3 anchors.

Predictors:
  - FiLMDelta:            pred(test) = mean over 3 anchors of [v_a + Δpred(a, test)]
  - DirectDeltaMLP:       same recipe with its own Δ predictor
  - DirectAbsoluteMLP:    two conditions reported:
      (a) "no_anchor":          MLP(test)                  (training-lab biased)
      (b) "anchor_recentered":  MLP(test) - bias_offset
          where bias_offset = mean_3anchors(MLP(anchor) - true_anchor_value).

Cells: 5 folds × 3 predictors × 2 pretrains × 3 anchor seeds = 90.

Reuses the exact FiLMDelta / DirectDeltaMLP / DirectAbsoluteMLP wrappers and
the kinase pretrain checkpoints from `exp5b_zap70_full_eval.py`.

Outputs:
  results/paper_evaluation/exp5b_zap70_lab_anchored.json
  results/paper_evaluation/exp5b_zap70_lab_anchored_summary.md
  results/paper_evaluation/exp5b_zap70_lab_anchored.png
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False  # CPU only

from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse infra from the full eval script (predictors, FP cache, splits, metrics).
from experiments.exp5b_zap70_full_eval import (  # noqa: E402
    DirectAbsoluteWrapper,
    DirectDeltaWrapper,
    FiLMDeltaWrapper,
    KINASE_PAIRS,
    MAX_KINASE_PAIRS,
    N_BITS,
    PRETRAIN_EPOCHS,
    ZAP70_PAIRS,
    collect_endpoint_table,
    kinase_pretrain_state,
    lab_disjoint_folds,
    regression_metrics,
    smi_to_morgan,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_JSON = RESULTS_DIR / "exp5b_zap70_lab_anchored.json"
OUT_MD = RESULTS_DIR / "exp5b_zap70_lab_anchored_summary.md"
OUT_PNG = RESULTS_DIR / "exp5b_zap70_lab_anchored.png"

N_LAB_FOLDS = 5
N_ANCHORS = 3
ANCHOR_SEEDS = [11, 23, 47]  # 3 distinct random selections of in-house anchors
MIN_REMAINING_TEST_MOLS = 10  # skip fold if fewer test mols remain post-anchor


# ---------------------------------------------------------------------------
# Lab-anchored absolute prediction
# ---------------------------------------------------------------------------
def lab_anchored_delta_predictions(
    predict_delta_fn,
    anchors: pd.DataFrame,           # cols: mol_id, smiles, value
    target_mols: pd.DataFrame,       # cols: mol_id, smiles, true_value
    fp_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """For each target mol, predict abs pIC50 as mean over anchors of
       (anchor_value + predicted_delta(anchor -> target_mol)).
       Returns (y_true, y_pred, mol_ids)."""
    if len(anchors) == 0 or len(target_mols) == 0:
        return np.array([]), np.array([]), []

    A_fp = np.stack([fp_cache[s] for s in anchors["smiles"]]).astype(np.float32)
    anchor_vals = anchors["value"].values.astype(np.float32)
    n_anchors = len(anchors)

    y_true_list, y_pred_list, ids = [], [], []
    for _, row in target_mols.iterrows():
        smi = row["smiles"]
        if smi not in fp_cache:
            continue
        b_fp = fp_cache[smi].astype(np.float32)
        B_fp = np.tile(b_fp, (n_anchors, 1))
        delta_pred = predict_delta_fn(A_fp, B_fp)
        implied_abs = anchor_vals + delta_pred
        y_pred_list.append(float(np.mean(implied_abs)))
        y_true_list.append(float(row["true_value"]))
        ids.append(row["mol_id"])
    return (np.array(y_true_list, dtype=np.float32),
            np.array(y_pred_list, dtype=np.float32), ids)


def direct_absolute_predictions(
    wrapper: DirectAbsoluteWrapper,
    target_mols: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Direct absolute prediction: out = MLP(target_smi).
       Returns (y_true, y_pred_raw, mol_ids). Caller applies optional recenter."""
    if len(target_mols) == 0:
        return np.array([]), np.array([]), []
    ids, y_true, X_rows = [], [], []
    for _, row in target_mols.iterrows():
        smi = row["smiles"]
        if smi not in fp_cache:
            continue
        ids.append(row["mol_id"])
        y_true.append(float(row["true_value"]))
        X_rows.append(fp_cache[smi].astype(np.float32))
    if not X_rows:
        return np.array([]), np.array([]), []
    X = np.stack(X_rows).astype(np.float32)
    y_pred = wrapper.predict_abs(X)
    return np.array(y_true, dtype=np.float32), y_pred.astype(np.float32), ids


# ---------------------------------------------------------------------------
# Single fold runner (fits each predictor once; then loops over anchor seeds)
# ---------------------------------------------------------------------------
def run_lab_anchored_fold(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    fp_cache: Dict[str, np.ndarray],
    pretrain_states: Dict[str, dict],
    fold_id: int,
) -> List[Dict]:
    """For one lab_disjoint fold, train all 3 predictors x 2 pretrain regimes
       once, then evaluate over 3 anchor seeds.
       Returns list of per-cell records (predictor x pretrain x anchor_seed)."""
    # Internal val split (10% of train_pairs) for early stopping
    n_val = max(20, int(0.1 * len(train_pairs)))
    rng_val = np.random.RandomState(fold_id * 7 + 3)
    perm = rng_val.permutation(len(train_pairs))
    val_pairs = train_pairs.iloc[perm[:n_val]].reset_index(drop=True)
    fit_pairs = train_pairs.iloc[perm[n_val:]].reset_index(drop=True)

    # Stack FPs for delta predictor fitting
    def stack(pairs):
        A = np.stack([fp_cache[s] for s in pairs["mol_a"]]).astype(np.float32)
        B = np.stack([fp_cache[s] for s in pairs["mol_b"]]).astype(np.float32)
        d = pairs["delta"].values.astype(np.float32)
        return A, B, d

    A_fit, B_fit, d_fit = stack(fit_pairs)
    A_val, B_val, d_val = stack(val_pairs)

    # Test lab endpoint table (unique mols + their true pIC50 in lab E).
    # Keep both `value` (used as anchor's known pIC50) and `true_value`
    # (used as ground truth when the mol is a held-out target) — they're
    # the same number; the alias keeps anchor and target code paths clean.
    test_endpoints = collect_endpoint_table(test_pairs)
    test_endpoints["true_value"] = test_endpoints["value"]
    n_test_mols = len(test_endpoints)
    if n_test_mols < N_ANCHORS + MIN_REMAINING_TEST_MOLS:
        print(f"  fold {fold_id}: SKIP — only {n_test_mols} test mols "
              f"(need {N_ANCHORS + MIN_REMAINING_TEST_MOLS})", flush=True)
        return []

    # Training endpoint table (for the "no_anchor" DirectAbs baseline reference
    # — none needed here, but kept for clarity/diagnostics).
    train_endpoints = collect_endpoint_table(fit_pairs)

    # --- Train all 6 (predictor x pretrain) wrappers up front
    wrappers: Dict[Tuple[str, str], object] = {}
    predictor_classes = [FiLMDeltaWrapper, DirectDeltaWrapper, DirectAbsoluteWrapper]
    pretrain_regimes = ["none", "kinase"]

    for pcls in predictor_classes:
        for pretrain in pretrain_regimes:
            t0 = time.time()
            wrapper = pcls()
            pretrained = pretrain_states[pcls.__name__] if pretrain == "kinase" else None
            if pcls is DirectAbsoluteWrapper:
                wrapper.fit_from_pairs(fit_pairs, val_pairs, fp_cache,
                                       pretrained_state=pretrained)
            else:
                wrapper.fit(A_fit, B_fit, d_fit, A_val, B_val, d_val,
                            pretrained_state=pretrained)
            wrappers[(pcls.NAME, pretrain)] = wrapper
            print(f"    fold {fold_id} trained {pcls.NAME}/{pretrain} "
                  f"in {time.time()-t0:.1f}s", flush=True)

    # --- Evaluate per anchor seed
    records: List[Dict] = []
    for anchor_seed in ANCHOR_SEEDS:
        rng_a = np.random.RandomState(anchor_seed * 100 + fold_id)
        anchor_idx = rng_a.choice(n_test_mols, size=N_ANCHORS, replace=False)
        anchor_idx_set = set(int(i) for i in anchor_idx)
        anchors = test_endpoints.iloc[list(anchor_idx_set)].reset_index(drop=True)
        remaining = test_endpoints.iloc[
            [i for i in range(n_test_mols) if i not in anchor_idx_set]
        ].reset_index(drop=True)
        anchor_mol_ids = sorted(anchors["mol_id"].tolist())

        for pcls in predictor_classes:
            for pretrain in pretrain_regimes:
                wrapper = wrappers[(pcls.NAME, pretrain)]
                if pcls is DirectAbsoluteWrapper:
                    # (a) no_anchor: raw MLP(test) prediction
                    y_true, y_pred_raw, ids = direct_absolute_predictions(
                        wrapper, remaining, fp_cache
                    )
                    m_no_anchor = regression_metrics(y_true, y_pred_raw)

                    # (b) anchor_recentered: subtract mean bias
                    # bias = mean over anchors of (MLP(anchor) - true_anchor)
                    if len(anchors) > 0:
                        A_fp = np.stack([fp_cache[s] for s in anchors["smiles"]]).astype(np.float32)
                        anchor_preds = wrapper.predict_abs(A_fp)
                        bias = float(np.mean(anchor_preds - anchors["value"].values.astype(np.float32)))
                    else:
                        bias = 0.0
                    y_pred_rec = y_pred_raw - bias
                    m_recentered = regression_metrics(y_true, y_pred_rec)

                    records.append({
                        "fold": int(fold_id),
                        "predictor": pcls.NAME,
                        "pretrain": pretrain,
                        "anchor_seed": int(anchor_seed),
                        "mode": "no_anchor",
                        "n_anchors": int(N_ANCHORS),
                        "anchor_mol_ids": anchor_mol_ids,
                        "n_test_mols": int(len(ids)),
                        "n_train_pairs": int(len(fit_pairs)),
                        "n_test_pairs": int(len(test_pairs)),
                        "mae": m_no_anchor["mae"],
                        "pearson": m_no_anchor["pearson"],
                        "spearman": m_no_anchor["spearman"],
                        "bias_offset": 0.0,
                    })
                    records.append({
                        "fold": int(fold_id),
                        "predictor": pcls.NAME,
                        "pretrain": pretrain,
                        "anchor_seed": int(anchor_seed),
                        "mode": "anchor_recentered",
                        "n_anchors": int(N_ANCHORS),
                        "anchor_mol_ids": anchor_mol_ids,
                        "n_test_mols": int(len(ids)),
                        "n_train_pairs": int(len(fit_pairs)),
                        "n_test_pairs": int(len(test_pairs)),
                        "mae": m_recentered["mae"],
                        "pearson": m_recentered["pearson"],
                        "spearman": m_recentered["spearman"],
                        "bias_offset": bias,
                    })
                else:
                    y_true, y_pred, ids = lab_anchored_delta_predictions(
                        wrapper.predict_delta, anchors, remaining, fp_cache
                    )
                    m = regression_metrics(y_true, y_pred)
                    records.append({
                        "fold": int(fold_id),
                        "predictor": pcls.NAME,
                        "pretrain": pretrain,
                        "anchor_seed": int(anchor_seed),
                        "mode": "lab_anchored",
                        "n_anchors": int(N_ANCHORS),
                        "anchor_mol_ids": anchor_mol_ids,
                        "n_test_mols": int(len(ids)),
                        "n_train_pairs": int(len(fit_pairs)),
                        "n_test_pairs": int(len(test_pairs)),
                        "mae": m["mae"],
                        "pearson": m["pearson"],
                        "spearman": m["spearman"],
                        "bias_offset": 0.0,
                    })
    # Cleanup
    del wrappers, A_fit, B_fit, A_val, B_val
    gc.collect()
    return records


# ---------------------------------------------------------------------------
# Summarize & write outputs
# ---------------------------------------------------------------------------
# (predictor, pretrain, mode) -> display label
DISPLAY_LABELS = {
    ("FiLMDelta", "none", "lab_anchored"): "FiLMDelta (3-anchor, no pretrain)",
    ("FiLMDelta", "kinase", "lab_anchored"): "FiLMDelta (3-anchor, kinase pre)",
    ("DirectDeltaMLP", "none", "lab_anchored"): "DirectDeltaMLP (3-anchor, no pretrain)",
    ("DirectDeltaMLP", "kinase", "lab_anchored"): "DirectDeltaMLP (3-anchor, kinase pre)",
    ("DirectAbsoluteMLP", "none", "no_anchor"): "DirectAbs (no anchor info, no pre)",
    ("DirectAbsoluteMLP", "kinase", "no_anchor"): "DirectAbs (no anchor info, kinase pre)",
    ("DirectAbsoluteMLP", "none", "anchor_recentered"): "DirectAbs (3-anchor recentered, no pre)",
    ("DirectAbsoluteMLP", "kinase", "anchor_recentered"): "DirectAbs (3-anchor recentered, kinase pre)",
}
ROW_ORDER = [
    ("FiLMDelta", "none", "lab_anchored"),
    ("FiLMDelta", "kinase", "lab_anchored"),
    ("DirectDeltaMLP", "none", "lab_anchored"),
    ("DirectDeltaMLP", "kinase", "lab_anchored"),
    ("DirectAbsoluteMLP", "none", "anchor_recentered"),
    ("DirectAbsoluteMLP", "kinase", "anchor_recentered"),
    ("DirectAbsoluteMLP", "none", "no_anchor"),
    ("DirectAbsoluteMLP", "kinase", "no_anchor"),
]


def summarize_records(records: List[dict]) -> List[dict]:
    if not records:
        return []
    df = pd.DataFrame(records)
    # Bucket key includes mode so we keep no_anchor and anchor_recentered apart.
    bucket_cols = ["predictor", "pretrain", "mode"]
    metric_cols = ["mae", "pearson", "spearman"]
    out = []
    for keys, grp in df.groupby(bucket_cols):
        rec = dict(zip(bucket_cols, keys))
        rec["n_cells"] = int(len(grp))
        rec["n_folds_unique"] = int(grp["fold"].nunique())
        rec["n_anchor_seeds"] = int(grp["anchor_seed"].nunique())
        rec["n_test_mols_mean"] = float(grp["n_test_mols"].mean())
        for m in metric_cols:
            rec[f"{m}_mean"] = float(grp[m].mean())
            rec[f"{m}_std"] = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
        out.append(rec)
    return out


def write_md(summary: List[dict], cfg: dict, records: List[dict], path: Path):
    lines: List[str] = []
    lines.append("# EXP5b ZAP70 LAB-ANCHORED EVAL — does FiLM's lab-bias-clean property pay off?\n\n")
    lines.append(f"- Target: {cfg['target']}\n")
    lines.append(f"- ZAP70 pairs: {cfg['n_pairs_zap70']} ({cfg['n_mols_zap70']} mols, "
                 f"{cfg['n_assays_zap70']} assays)\n")
    lines.append(f"- Kinase pretrain pool: {cfg['n_kinase_pairs_used']} pairs across "
                 f"{len(cfg['kinase_targets'])} targets {cfg['kinase_targets']}\n")
    lines.append(f"- Lab-disjoint folds: {cfg['n_lab_folds']} (test = one held-out lab E)\n")
    lines.append(f"- In-house anchors per fold: {cfg['n_anchors']} mols drawn from test lab E\n")
    lines.append(f"- Anchor seeds: {cfg['anchor_seeds']} ({len(cfg['anchor_seeds'])} distinct random anchor draws per fold)\n")
    n_unique_train_cells = cfg['n_lab_folds'] * 3 * 2
    n_eval_cells = n_unique_train_cells * len(cfg['anchor_seeds'])
    lines.append(f"- Cells: {cfg['n_lab_folds']} folds × 3 predictors × 2 pretrains × "
                 f"{len(cfg['anchor_seeds'])} anchor seeds = {n_eval_cells} evaluations "
                 f"({n_unique_train_cells} unique train runs; DirectAbs reported in 2 modes)\n\n")

    lines.append("## Lab-anchored absolute prediction (lab_disjoint only)\n\n")
    lines.append("Each test molecule's pIC50 is predicted using only 3 lab-E "
                 "in-house anchor measurements. Numbers aggregate over "
                 f"{cfg['n_lab_folds']} folds × {len(cfg['anchor_seeds'])} anchor seeds "
                 f"= up to {cfg['n_lab_folds']*len(cfg['anchor_seeds'])} cells.\n\n")
    lines.append("| Predictor / mode | n_cells | n_test_mols | MAE | Pearson | Spearman |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for key in ROW_ORDER:
        rec = next((r for r in summary
                    if (r["predictor"], r["pretrain"], r["mode"]) == key), None)
        if rec is None:
            continue
        label = DISPLAY_LABELS[key]
        lines.append(
            f"| {label} | {rec['n_cells']} | {rec['n_test_mols_mean']:.0f} | "
            f"{rec['mae_mean']:.3f}±{rec['mae_std']:.3f} | "
            f"{rec['pearson_mean']:.3f}±{rec['pearson_std']:.3f} | "
            f"{rec['spearman_mean']:.3f}±{rec['spearman_std']:.3f} |\n"
        )
    lines.append("\n")

    # Per-fold spread tables — MAE only, FiLM and DirectAbs-recentered side by side.
    lines.append("## Per-fold spread (MAE, mean over 3 anchor seeds)\n\n")
    df = pd.DataFrame(records)
    if not df.empty:
        folds_present = sorted(df["fold"].unique())
        keys_to_show = [
            ("FiLMDelta", "kinase", "lab_anchored", "FiLM-kp"),
            ("FiLMDelta", "none", "lab_anchored", "FiLM-no"),
            ("DirectDeltaMLP", "kinase", "lab_anchored", "DDelta-kp"),
            ("DirectAbsoluteMLP", "kinase", "anchor_recentered", "DAbs-rec-kp"),
            ("DirectAbsoluteMLP", "none", "anchor_recentered", "DAbs-rec-no"),
            ("DirectAbsoluteMLP", "kinase", "no_anchor", "DAbs-raw-kp"),
            ("DirectAbsoluteMLP", "none", "no_anchor", "DAbs-raw-no"),
        ]
        header = "| Fold | " + " | ".join(label for _, _, _, label in keys_to_show) + " |\n"
        sep = "|---|" + "---|" * len(keys_to_show) + "\n"
        lines.append(header)
        lines.append(sep)
        for fi in folds_present:
            cells = []
            for p, pt, mode, _ in keys_to_show:
                sub = df[(df["fold"] == fi) & (df["predictor"] == p)
                         & (df["pretrain"] == pt) & (df["mode"] == mode)]
                if sub.empty:
                    cells.append("—")
                else:
                    cells.append(f"{sub['mae'].mean():.3f}")
            lines.append(f"| {fi} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Headline / verdict
    lines.append("## Verdict — does FiLM's lab-bias-clean property show up?\n\n")
    by_key = {(r["predictor"], r["pretrain"], r["mode"]): r for r in summary}
    film_kp = by_key.get(("FiLMDelta", "kinase", "lab_anchored"))
    film_no = by_key.get(("FiLMDelta", "none", "lab_anchored"))
    dd_kp = by_key.get(("DirectDeltaMLP", "kinase", "lab_anchored"))
    dabs_rec_kp = by_key.get(("DirectAbsoluteMLP", "kinase", "anchor_recentered"))
    dabs_rec_no = by_key.get(("DirectAbsoluteMLP", "none", "anchor_recentered"))
    dabs_raw_kp = by_key.get(("DirectAbsoluteMLP", "kinase", "no_anchor"))
    dabs_raw_no = by_key.get(("DirectAbsoluteMLP", "none", "no_anchor"))

    if film_kp and dabs_rec_kp and dabs_raw_kp:
        delta_vs_recentered = dabs_rec_kp["mae_mean"] - film_kp["mae_mean"]
        delta_vs_raw = dabs_raw_kp["mae_mean"] - film_kp["mae_mean"]
        lines.append(
            f"- FiLMDelta (3-anchor, kinase pre) MAE = {film_kp['mae_mean']:.3f}±"
            f"{film_kp['mae_std']:.3f}.\n"
            f"- DirectAbs (3-anchor recentered, kinase pre) MAE = "
            f"{dabs_rec_kp['mae_mean']:.3f}±{dabs_rec_kp['mae_std']:.3f}.\n"
            f"- DirectAbs (no anchor info, kinase pre) MAE = "
            f"{dabs_raw_kp['mae_mean']:.3f}±{dabs_raw_kp['mae_std']:.3f}.\n"
            f"- Gap FiLM vs DirectAbs-recentered: {delta_vs_recentered:+.3f} pIC50 "
            f"({'FiLM better' if delta_vs_recentered > 0 else 'DirectAbs better'}).\n"
            f"- Gap FiLM vs DirectAbs-no-anchor: {delta_vs_raw:+.3f} pIC50.\n"
        )
        if delta_vs_recentered > 0.05:
            verdict = ("YES — FiLM beats DirectAbs (3-anchor recentered) by a meaningful "
                       "margin; the lab-bias-clean Δ property pays off when anchors come from "
                       "the test lab.")
        elif delta_vs_recentered > 0.0:
            verdict = ("MIXED — FiLM beats DirectAbs (recentered) but the margin is small. "
                       "Simple lab-mean recentering already closes most of the gap.")
        else:
            verdict = ("NO — DirectAbs with 3-anchor recentering matches or beats FiLM. "
                       "The lab-bias argument is largely captured by a one-parameter offset; "
                       "FiLM's per-pair anchoring buys little additional accuracy here.")
        lines.append(f"\n**Verdict: {verdict}**\n")

    # Constant-offset hypothesis check
    if dabs_raw_kp and dabs_rec_kp:
        gap = dabs_raw_kp["mae_mean"] - dabs_rec_kp["mae_mean"]
        lines.append(
            f"\n*Lab-bias offset diagnostic:* DirectAbs MAE drops by {gap:+.3f} pIC50 "
            f"({dabs_raw_kp['mae_mean']:.3f} -> {dabs_rec_kp['mae_mean']:.3f}) when a "
            f"single 3-anchor mean-bias offset is subtracted. This quantifies the "
            f"training-lab bias DirectAbs carries into the test lab.\n"
        )

    with open(path, "w") as f:
        f.writelines(lines)


def make_plot(summary: List[dict], records: List[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bar chart: MAE / Pearson / Spearman for each (predictor, pretrain, mode).
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("mae", "MAE (lower better)"),
               ("pearson", "Pearson r"),
               ("spearman", "Spearman rho")]

    labels = [DISPLAY_LABELS[k] for k in ROW_ORDER
              if any((r["predictor"], r["pretrain"], r["mode"]) == k for r in summary)]
    keys_present = [k for k in ROW_ORDER
                    if any((r["predictor"], r["pretrain"], r["mode"]) == k for r in summary)]

    palette = {
        ("FiLMDelta", "none", "lab_anchored"): "#5fa8d3",
        ("FiLMDelta", "kinase", "lab_anchored"): "#08306b",
        ("DirectDeltaMLP", "none", "lab_anchored"): "#ec7063",
        ("DirectDeltaMLP", "kinase", "lab_anchored"): "#922b21",
        ("DirectAbsoluteMLP", "none", "anchor_recentered"): "#82e0aa",
        ("DirectAbsoluteMLP", "kinase", "anchor_recentered"): "#186a3b",
        ("DirectAbsoluteMLP", "none", "no_anchor"): "#d7dbdd",
        ("DirectAbsoluteMLP", "kinase", "no_anchor"): "#566573",
    }

    by_key = {(r["predictor"], r["pretrain"], r["mode"]): r for r in summary}
    x = np.arange(len(keys_present))
    for i, (m, ylabel) in enumerate(metrics):
        ax = axes[i]
        ys = [by_key[k][f"{m}_mean"] for k in keys_present]
        errs = [by_key[k][f"{m}_std"] for k in keys_present]
        colors = [palette.get(k, "#808080") for k in keys_present]
        ax.bar(x, ys, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_title(ylabel)

    fig.suptitle("ZAP70 lab-anchored eval (lab_disjoint): predict test-lab pIC50 from "
                 f"{N_ANCHORS} in-house anchor measurements",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: 2 folds × 2 anchor seeds, 1 pretrain epoch.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    # Derive sibling MD + PNG paths from the JSON output stem so smoke runs
    # don't clobber production summaries.
    out_md = out_path.parent / (out_path.stem + "_summary.md")
    out_png = out_path.parent / (out_path.stem + ".png")
    t_global = time.time()

    # --- Load ZAP70 pairs
    pairs = pd.read_csv(ZAP70_PAIRS)
    n_mols = len(set(pairs["mol_a_id"]).union(pairs["mol_b_id"]))
    n_assays = pairs["assay_id"].nunique()
    print(f"PROGRESS: ZAP70: {len(pairs)} pairs, {n_mols} mols, "
          f"{n_assays} assays", flush=True)

    # --- Load kinase pretrain pairs
    kinase_df = pd.read_csv(KINASE_PAIRS)
    print(f"PROGRESS: kinase: {len(kinase_df)} total pairs, "
          f"targets={sorted(kinase_df['target_chembl_id'].unique())}", flush=True)
    if len(kinase_df) > MAX_KINASE_PAIRS:
        kinase_df = kinase_df.sample(n=MAX_KINASE_PAIRS, random_state=42).reset_index(drop=True)
        print(f"PROGRESS: kinase subsampled to {len(kinase_df)}", flush=True)

    # --- Build Morgan FP cache
    t0 = time.time()
    smis = list(
        set(pairs["mol_a"]).union(pairs["mol_b"])
        | set(kinase_df["mol_a"]).union(kinase_df["mol_b"])
    )
    fp_cache = {s: smi_to_morgan(s) for s in smis}
    print(f"PROGRESS: FP cache built for {len(fp_cache)} unique mols in "
          f"{time.time()-t0:.1f}s", flush=True)

    # --- Settings
    global ANCHOR_SEEDS  # may be overridden in quick mode
    if args.quick:
        n_lab_folds = 2
        anchor_seeds = ANCHOR_SEEDS[:2]
        pretrain_epochs = 1
    else:
        n_lab_folds = N_LAB_FOLDS
        anchor_seeds = ANCHOR_SEEDS
        pretrain_epochs = PRETRAIN_EPOCHS
    ANCHOR_SEEDS = anchor_seeds

    # --- Build folds (same seed=0 as full eval so results align directly)
    lab_folds = lab_disjoint_folds(pairs, n_folds=n_lab_folds, seed=0)
    for fi, (tr, te) in enumerate(lab_folds):
        train_assays = sorted(tr["assay_id"].unique())
        test_assays = sorted(te["assay_id"].unique())
        n_test_mols_full = len(set(te["mol_a_id"]).union(te["mol_b_id"]))
        print(f"  lab fold {fi}: train_pairs={len(tr)} test_pairs={len(te)} "
              f"test_mols={n_test_mols_full} "
              f"test_assays={test_assays[:3]}...({len(test_assays)})", flush=True)

    # --- Kinase pretrain checkpoints (one per predictor type, reused across folds)
    print("PROGRESS: kinase pretraining (3 predictor types)...", flush=True)
    pretrain_states: Dict[str, dict] = {}
    for pcls in (FiLMDeltaWrapper, DirectDeltaWrapper, DirectAbsoluteWrapper):
        st = kinase_pretrain_state(pcls, kinase_df, fp_cache,
                                   epochs=pretrain_epochs, verbose=True)
        pretrain_states[pcls.__name__] = st

    # --- Resume support
    results = {"runs": []}
    if args.resume and out_path.exists():
        try:
            prior = json.load(open(out_path))
            results["runs"] = prior.get("runs", [])
            print(f"PROGRESS: resume — loaded {len(results['runs'])} prior runs",
                  flush=True)
        except Exception as e:
            print(f"PROGRESS: resume parse failed: {e}", flush=True)

    done_folds = set(r["fold"] for r in results["runs"])

    config = {
        "target": "ZAP70 (CHEMBL2803)",
        "zap70_pairs": str(ZAP70_PAIRS),
        "kinase_pairs": str(KINASE_PAIRS),
        "n_pairs_zap70": int(len(pairs)),
        "n_mols_zap70": int(n_mols),
        "n_assays_zap70": int(n_assays),
        "n_kinase_pairs_used": int(len(kinase_df)),
        "kinase_targets": sorted(kinase_df["target_chembl_id"].unique().tolist()),
        "n_lab_folds": int(n_lab_folds),
        "n_anchors": int(N_ANCHORS),
        "anchor_seeds": list(anchor_seeds),
        "min_remaining_test_mols": int(MIN_REMAINING_TEST_MOLS),
        "pretrain_epochs": int(pretrain_epochs),
        "morgan_n_bits": int(N_BITS),
    }
    results["config"] = config

    # --- Main loop (one entry per fold; each adds 3 anchor_seeds × 8 records)
    for fi, (train_pairs, test_pairs) in enumerate(lab_folds):
        if fi in done_folds:
            print(f"PROGRESS: skip fold {fi} (already done)", flush=True)
            continue
        t_fold = time.time()
        print(f"\nPROGRESS: fold {fi} start", flush=True)
        try:
            fold_records = run_lab_anchored_fold(
                train_pairs, test_pairs, fp_cache, pretrain_states, fold_id=fi
            )
        except Exception as e:
            import traceback
            print(f"PROGRESS: ERROR fold {fi}: {e}\n{traceback.format_exc()}",
                  flush=True)
            fold_records = []

        results["runs"].extend(fold_records)
        # Incremental save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"PROGRESS: fold {fi} done in {time.time()-t_fold:.1f}s "
              f"({len(fold_records)} records added)", flush=True)

    # --- Summarize
    summary = summarize_records(results["runs"])
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPROGRESS: wrote {out_path}", flush=True)

    write_md(summary, config, results["runs"], out_md)
    print(f"PROGRESS: wrote {out_md}", flush=True)

    try:
        make_plot(summary, results["runs"], out_png)
        print(f"PROGRESS: wrote {out_png}", flush=True)
    except Exception as e:
        print(f"PROGRESS: plot failed: {e}", flush=True)

    print(f"\nPROGRESS: TOTAL TIME {(time.time()-t_global)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
