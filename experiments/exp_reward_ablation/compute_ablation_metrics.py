#!/usr/bin/env python3
"""
Reward Ablation Analysis (EXP3): per-cohort metric distributions
=================================================================

Computes headline metrics across 5 cohorts:
  - prior            : prior-only baseline (samples_covft.csv)
  - drop_film        : RL with FiLM removed (kept SMARTS+QED)
  - drop_smarts      : RL with SMARTS removed (kept FiLM+QED)
  - drop_qed         : RL with QED removed (kept FiLM+SMARTS)
  - full_dap         : full 3-component DAP baseline

Outputs:
  - results/paper_evaluation/exp3_ablation_analysis.json
  - results/paper_evaluation/exp3_ablation_analysis.png
  - results/paper_evaluation/exp3_ablation_summary.md

Local CPU, RDKit only. FiLM re-scoring on drop_film + prior uses the
cached REINVENT4 FiLM model (anchor-based pIC50).
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THIQ_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"
THIQ_PATTERN = Chem.MolFromSmarts(THIQ_SMARTS)

RNG = np.random.default_rng(20260624)
MATCHED_N = 8000          # for drop_film matched-N subsample
TANIMOTO_PAIRS = 500      # for internal diversity estimate


# ------------------------------- I/O helpers --------------------------------

COHORT_FILES = {
    "prior": {
        "path": PROJECT_ROOT / "experiments" / "exp_covft_value" / "samples_covft.csv",
        "smiles_col": "SMILES",
        "qed_col": None,
        "smarts_col": None,
        "film_col": None,
    },
    "drop_film": {
        "path": PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_film" / "thiq_rl_ablation_drop_film_1.csv",
        "smiles_col": "SMILES",
        "qed_col": "QED (raw)",
        "smarts_col": "THIQ-acrylamide core retained (raw)",
        "film_col": None,           # no native FiLM, will re-score
    },
    "drop_smarts": {
        "path": PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_smarts" / "thiq_rl_ablation_drop_smarts_1.csv",
        "smiles_col": "SMILES",
        "qed_col": "QED (raw)",
        "smarts_col": None,         # no native SMARTS score, compute from SMILES
        "film_col": "FiLMDelta pIC50 (raw)",
    },
    "drop_qed": {
        "path": PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_qed" / "thiq_rl_ablation_drop_qed_1.csv",
        "smiles_col": "SMILES",
        "qed_col": None,            # no native QED score, compute from SMILES
        "smarts_col": "THIQ-acrylamide core retained (raw)",
        "film_col": "FiLMDelta pIC50 (raw)",
    },
    "full_dap": {
        "path": PROJECT_ROOT / "data" / "tier4_scored" / "thiq_rl_exp2_zap70_scored.csv",
        "smiles_col": "smiles",
        "qed_col": "QED",
        "smarts_col": "thiq_core",       # bool True/False
        "film_col": "pIC50_film",
    },
}

COHORT_ORDER = ["prior", "drop_film", "drop_smarts", "drop_qed", "full_dap"]


# ------------------------------- RDKit utils --------------------------------

def safe_mol(smi):
    if not isinstance(smi, str) or not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def is_valid(smi):
    return safe_mol(smi) is not None


def thiq_match(smi):
    mol = safe_mol(smi)
    if mol is None:
        return 0.0
    return 1.0 if mol.HasSubstructMatch(THIQ_PATTERN) else 0.0


def qed_value(smi):
    mol = safe_mol(smi)
    if mol is None:
        return float("nan")
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def murcko_scaffold(smi):
    mol = safe_mol(smi)
    if mol is None:
        return None
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff, canonical=True)
    except Exception:
        return None


def morgan_fp(smi, radius=2, nbits=2048):
    mol = safe_mol(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


# ----------------------- Tanimoto diversity sampling ------------------------

def internal_tanimoto_diversity(smiles_list, n_pairs=TANIMOTO_PAIRS, rng=RNG):
    """Mean Tanimoto similarity over n_pairs random pairs; (1 - mean) is diversity."""
    fps = []
    for s in smiles_list:
        fp = morgan_fp(s)
        if fp is not None:
            fps.append(fp)
        if len(fps) >= 5000:           # cap to keep CPU cost bounded
            break
    if len(fps) < 2:
        return float("nan"), float("nan")
    pairs = min(n_pairs, len(fps) * (len(fps) - 1) // 2)
    sims = np.empty(pairs, dtype=np.float64)
    for k in range(pairs):
        i, j = rng.integers(0, len(fps), size=2)
        while i == j:
            j = int(rng.integers(0, len(fps)))
        sims[k] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return float(np.mean(sims)), float(1.0 - np.mean(sims))


# ---------------------------- FiLM re-scoring -------------------------------

def load_film_scorer():
    """Loads cached REINVENT4 FiLM model. Returns (model, scaler, anchor_embs, anchor_pIC50) or None."""
    cache_path = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if not cache_path.exists():
        print(f"[film] No cached model at {cache_path}; skipping re-scoring", file=sys.stderr)
        return None
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
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
        print(f"[film] Loaded cached model with {len(anchor_pIC50)} anchors", file=sys.stderr)
        return model, scaler, anchor_embs, anchor_pIC50
    except Exception as exc:
        print(f"[film] Failed to load cached model: {exc}", file=sys.stderr)
        return None


def film_score(smiles_list, scorer, batch_targets=32):
    """Anchor-based pIC50 prediction. Returns np.array (NaN for invalid).

    Batched: scores `batch_targets` molecules x 280 anchors per forward pass.
    """
    import torch
    model, scaler, anchor_embs, anchor_pIC50 = scorer
    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
    valid_idx, valid_fps = [], []
    for i, smi in enumerate(smiles_list):
        mol = safe_mol(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_fps.append(arr)
        valid_idx.append(i)
    if not valid_fps:
        return scores

    n_anchors = len(anchor_pIC50)
    anchor_pIC50_t = torch.as_tensor(anchor_pIC50, dtype=torch.float32)
    if not torch.is_tensor(anchor_embs):
        anchor_embs = torch.as_tensor(anchor_embs, dtype=torch.float32)

    target_embs_np = scaler.transform(np.asarray(valid_fps, dtype=np.float32))
    target_embs = torch.from_numpy(target_embs_np)

    with torch.no_grad():
        for start in range(0, len(valid_idx), batch_targets):
            end = min(start + batch_targets, len(valid_idx))
            B = end - start
            tgt = target_embs[start:end]                                  # [B, D]
            # Broadcast anchors x targets: (B, n_anchors, D)
            anchors_b = anchor_embs.unsqueeze(0).expand(B, n_anchors, -1) # [B, A, D]
            tgts_b = tgt.unsqueeze(1).expand(B, n_anchors, -1)            # [B, A, D]
            flat_anchors = anchors_b.reshape(B * n_anchors, -1)
            flat_tgts = tgts_b.reshape(B * n_anchors, -1)
            deltas = model(flat_anchors, flat_tgts).view(B, n_anchors)
            abs_preds = deltas + anchor_pIC50_t.unsqueeze(0)
            means = abs_preds.mean(dim=1).cpu().numpy()
            for k, m in enumerate(means):
                scores[valid_idx[start + k]] = float(m)
            if (start // batch_targets) % 20 == 0:
                print(f"[film]   scored {end:,}/{len(valid_idx):,}", file=sys.stderr)
    return scores


# ------------------------------ Per-cohort ----------------------------------

def coerce_smarts_col(series):
    """Coerce 'thiq_core' / 'THIQ-acrylamide core retained (raw)' to {0,1}."""
    if series.dtype == bool:
        return series.astype(float).values
    if series.dtype == object:
        return series.map(lambda v: 1.0 if str(v).lower() in ("true", "1", "1.0") else 0.0).values
    # numeric: 0.0/0.5/1.0 in REINVENT4 outputs — treat >=0.5 as "match"
    arr = series.astype(float).values
    return (arr >= 0.5).astype(float)


def compute_cohort_metrics(name, cfg, film_scorer=None, subsample=None, label_suffix=""):
    path = cfg["path"]
    print(f"\n[{name}{label_suffix}] Loading {path}", file=sys.stderr)
    df = pd.read_csv(path)
    n_raw = len(df)

    if subsample is not None and n_raw > subsample:
        df = df.sample(n=subsample, random_state=20260624).reset_index(drop=True)
        sub_note = f" subsampled to {len(df):,}"
    else:
        sub_note = ""
    n_total = len(df)

    smi_col = cfg["smiles_col"]
    smiles = df[smi_col].astype(str).tolist()

    # validity & uniqueness
    valid_mask = np.array([is_valid(s) for s in smiles])
    n_valid = int(valid_mask.sum())
    valid_smiles = [s for s, m in zip(smiles, valid_mask) if m]
    n_unique = int(pd.Series(valid_smiles).nunique())
    validity_pct = 100.0 * n_valid / n_total if n_total > 0 else float("nan")
    uniqueness_pct = 100.0 * n_unique / n_valid if n_valid > 0 else float("nan")

    print(f"[{name}{label_suffix}] N={n_total:,}{sub_note}, valid={n_valid:,} ({validity_pct:.1f}%), unique={n_unique:,}", file=sys.stderr)

    # SMARTS retention
    if cfg["smarts_col"] is not None and cfg["smarts_col"] in df.columns:
        smarts_arr = coerce_smarts_col(df[cfg["smarts_col"]])
    else:
        smarts_arr = np.array([thiq_match(s) for s in smiles])
    smarts_valid = smarts_arr[valid_mask]
    smarts_mean = float(np.nanmean(smarts_valid)) if len(smarts_valid) else float("nan")

    # QED
    if cfg["qed_col"] is not None and cfg["qed_col"] in df.columns:
        qed_arr = df[cfg["qed_col"]].astype(float).values
        # Some rows from REINVENT have QED=0 for invalid; mask via validity
        qed_arr = np.where(valid_mask & (qed_arr > 0), qed_arr, np.nan)
    else:
        qed_arr = np.array([qed_value(s) for s in smiles])
    qed_valid = qed_arr[~np.isnan(qed_arr)]
    qed_mean = float(np.mean(qed_valid)) if len(qed_valid) else float("nan")
    qed_median = float(np.median(qed_valid)) if len(qed_valid) else float("nan")

    # FiLM pIC50
    film_source = "native"
    if cfg["film_col"] is not None and cfg["film_col"] in df.columns:
        film_arr = pd.to_numeric(df[cfg["film_col"]], errors="coerce").values.astype(float)
        # Mask non-matches (REINVENT outputs 0 for invalid; full_dap may have NaN already)
        film_arr = np.where(valid_mask & (film_arr > 1.0), film_arr, np.nan)
    elif film_scorer is not None:
        print(f"[{name}{label_suffix}] Re-scoring {len(smiles):,} SMILES with cached FiLM model...", file=sys.stderr)
        film_arr = film_score(smiles, film_scorer)
        film_source = "rescored"
    else:
        film_arr = np.full(len(smiles), np.nan)
        film_source = "unavailable"
    film_valid = film_arr[~np.isnan(film_arr)]
    film_mean = float(np.mean(film_valid)) if len(film_valid) else float("nan")
    film_median = float(np.median(film_valid)) if len(film_valid) else float("nan")
    film_p90 = float(np.percentile(film_valid, 90)) if len(film_valid) else float("nan")
    film_p99 = float(np.percentile(film_valid, 99)) if len(film_valid) else float("nan")

    # Scaffold uniqueness
    scaffolds = []
    for s, ok in zip(smiles, valid_mask):
        if not ok:
            continue
        sc = murcko_scaffold(s)
        if sc:
            scaffolds.append(sc)
    n_scaffolds = int(pd.Series(scaffolds).nunique()) if scaffolds else 0
    scaffold_uniqueness = n_scaffolds / n_valid if n_valid > 0 else float("nan")

    # Internal Tanimoto diversity
    sim_mean, div = internal_tanimoto_diversity(valid_smiles)

    return {
        "cohort": f"{name}{label_suffix}",
        "n_raw": n_raw,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "validity_pct": validity_pct,
        "uniqueness_pct": uniqueness_pct,
        "smarts_mean_retention": smarts_mean,
        "qed_mean": qed_mean,
        "qed_median": qed_median,
        "film_mean": film_mean,
        "film_median": film_median,
        "film_p90": film_p90,
        "film_p99": film_p99,
        "film_source": film_source,
        "n_scaffolds": n_scaffolds,
        "scaffold_uniqueness": scaffold_uniqueness,
        "internal_tanimoto_mean": sim_mean,
        "internal_diversity": div,
    }


# --------------------------------- Plot -------------------------------------

def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cohorts = [r["cohort"] for r in rows]

    metrics = [
        ("FiLM-pred pIC50 (mean)", "film_mean", "#3b82f6"),
        ("Warhead retention (%)", "smarts_mean_retention", "#ef4444"),
        ("QED (mean)", "qed_mean", "#10b981"),
        ("Scaffold uniqueness", "scaffold_uniqueness", "#f59e0b"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (title, key, color) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(cohorts, vals, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        for b, v in zip(bars, vals):
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3f}" if v < 10 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Reward Ablation (EXP3): per-cohort metric distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote {path}", file=sys.stderr)


# --------------------------- Markdown summary -------------------------------

def make_markdown(rows, path):
    def fmt(v, dp=3):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.{dp}f}"

    # Pivot rows by cohort
    by_cohort = {r["cohort"]: r for r in rows}

    cols = COHORT_ORDER + (["drop_film_full"] if "drop_film_full" in by_cohort else [])
    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"

    lines = [
        "# EXP3 Reward Ablation Summary",
        "",
        "Per-cohort metric distributions for the 3-reward DAP system on ZAP70 (THIQ-acrylamide chassis).",
        "All RL cohorts share the same prior (`covalent_ft`). Reward weights:",
        "",
        "- **full_dap**: FiLM=0.56, SMARTS=0.27, QED=0.17",
        "- **drop_film**: SMARTS=0.80, QED=0.20 (FiLM removed)",
        "- **drop_smarts**: FiLM=0.83, QED=0.17 (SMARTS removed)",
        "- **drop_qed**: FiLM=0.56, SMARTS=0.44 (QED removed)",
        "",
        "## Headline table",
        "",
        header,
        sep,
    ]

    def row_for(label, key, dp=3):
        cells = [label]
        for c in cols:
            v = by_cohort.get(c, {}).get(key)
            cells.append(fmt(v, dp))
        return "| " + " | ".join(cells) + " |"

    def int_row(label, key):
        cells = [label]
        for c in cols:
            v = by_cohort.get(c, {}).get(key)
            cells.append(f"{int(v):,}" if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else "n/a")
        return "| " + " | ".join(cells) + " |"

    lines += [
        int_row("N (raw)", "n_raw"),
        int_row("N (analyzed)", "n_total"),
        int_row("N valid", "n_valid"),
        int_row("N unique", "n_unique"),
        row_for("Validity (%)", "validity_pct", 1),
        row_for("Uniqueness (%)", "uniqueness_pct", 1),
        row_for("Warhead retention (mean)", "smarts_mean_retention"),
        row_for("QED (mean)", "qed_mean"),
        row_for("QED (median)", "qed_median"),
        row_for("FiLM pIC50 (mean)", "film_mean"),
        row_for("FiLM pIC50 (median)", "film_median"),
        row_for("FiLM pIC50 (P90)", "film_p90"),
        row_for("FiLM pIC50 (P99)", "film_p99"),
        row_for("Scaffold uniqueness (n_uniq/n_valid)", "scaffold_uniqueness"),
        row_for("Internal Tanimoto (sample mean)", "internal_tanimoto_mean"),
        row_for("Internal diversity (1 - mean Tc)", "internal_diversity"),
    ]

    # FiLM source row
    src_cells = ["FiLM score source"]
    for c in cols:
        src_cells.append(by_cohort.get(c, {}).get("film_source", "n/a"))
    lines += ["| " + " | ".join(src_cells) + " |"]

    # Per-component contribution interpretation
    prior_r = by_cohort["prior"]
    df_r = by_cohort["drop_film"]
    ds_r = by_cohort["drop_smarts"]
    dq_r = by_cohort["drop_qed"]
    fd_r = by_cohort["full_dap"]

    lines += [
        "",
        "## Per-component contribution (full_dap minus drop_X)",
        "",
        "| Reward removed | Drop in mean pIC50 | Drop in warhead retention | Drop in QED |",
        "|---|---|---|---|",
        f"| FiLM   | {fmt(fd_r['film_mean'] - df_r['film_mean'])} | {fmt(fd_r['smarts_mean_retention'] - df_r['smarts_mean_retention'])} | {fmt(fd_r['qed_mean'] - df_r['qed_mean'])} |",
        f"| SMARTS | {fmt(fd_r['film_mean'] - ds_r['film_mean'])} | {fmt(fd_r['smarts_mean_retention'] - ds_r['smarts_mean_retention'])} | {fmt(fd_r['qed_mean'] - ds_r['qed_mean'])} |",
        f"| QED    | {fmt(fd_r['film_mean'] - dq_r['film_mean'])} | {fmt(fd_r['smarts_mean_retention'] - dq_r['smarts_mean_retention'])} | {fmt(fd_r['qed_mean'] - dq_r['qed_mean'])} |",
        "",
        "## Interpretation",
        "",
        f"Compared to the full 3-reward DAP baseline (mean pIC50={fmt(fd_r['film_mean'])}, "
        f"warhead retention={fmt(fd_r['smarts_mean_retention'])}, QED={fmt(fd_r['qed_mean'])}), each ablation "
        f"selectively degrades the metric tied to the removed reward. "
        f"Removing FiLM lowers mean predicted pIC50 from {fmt(fd_r['film_mean'])} -> {fmt(df_r['film_mean'])} "
        f"(Delta={fmt(fd_r['film_mean']-df_r['film_mean'])}); notably, the drop_film agent ends up "
        f"BELOW the prior-only baseline ({fmt(prior_r['film_mean'])}), so without FiLM the policy drifts away "
        f"from potent chemistry while still satisfying the SMARTS gate. "
        f"Warhead retention and QED stay high in drop_film because their gates remain active. "
        f"Removing SMARTS drops warhead retention from {fmt(fd_r['smarts_mean_retention'])} -> {fmt(ds_r['smarts_mean_retention'])} "
        f"(Delta={fmt(fd_r['smarts_mean_retention']-ds_r['smarts_mean_retention'])}) — a near-total collapse, "
        f"confirming the SMARTS term is what enforces THIQ-acrylamide preservation; the agent immediately diversifies away "
        f"from the chassis once unconstrained, taking QED with it ({fmt(ds_r['qed_mean'])}). "
        f"Removing QED lowers mean QED from {fmt(fd_r['qed_mean'])} -> {fmt(dq_r['qed_mean'])} "
        f"(Delta={fmt(fd_r['qed_mean']-dq_r['qed_mean'])}); pIC50 stays close to baseline ({fmt(dq_r['film_mean'])} vs {fmt(fd_r['film_mean'])}), "
        f"so QED behaves as a near-orthogonal drug-likeness anchor. "
        f"The prior already sits at high warhead retention ({fmt(prior_r['smarts_mean_retention'])}) and decent FiLM ({fmt(prior_r['film_mean'])}) "
        f"because the covalent_ft prior is itself THIQ-biased; RL's job is to push pIC50 the remaining ~0.1 units while keeping QED reasonable, "
        f"and FiLM is the only reward that achieves that lift.",
        "",
        "## Headline number for paper",
        "",
        f"**Removing FiLM costs Delta pIC50 = {fmt(fd_r['film_mean']-df_r['film_mean'])} units** "
        f"(full_dap {fmt(fd_r['film_mean'])} -> drop_film {fmt(df_r['film_mean'])}, "
        f"falling below prior {fmt(prior_r['film_mean'])}); "
        f"**Removing SMARTS costs Delta warhead retention = {fmt(fd_r['smarts_mean_retention']-ds_r['smarts_mean_retention'])}** "
        f"(full_dap {fmt(fd_r['smarts_mean_retention'])} -> drop_smarts {fmt(ds_r['smarts_mean_retention'])}); "
        f"**Removing QED costs Delta QED = {fmt(fd_r['qed_mean']-dq_r['qed_mean'])}** "
        f"(full_dap {fmt(fd_r['qed_mean'])} -> drop_qed {fmt(dq_r['qed_mean'])}). "
        f"Each reward term controls its own metric with minimal cross-interference; the only cross-coupling is that "
        f"SMARTS-removal also drags down QED, suggesting the THIQ-acrylamide chassis is what kept the prior in a drug-like region.",
    ]

    path.write_text("\n".join(lines))
    print(f"[md] Wrote {path}", file=sys.stderr)


# ---------------------------------- main ------------------------------------

def main():
    film_scorer = load_film_scorer()

    rows = []
    for name in COHORT_ORDER:
        cfg = COHORT_FILES[name]
        # For drop_film: subsample to MATCHED_N for the primary row (fair comparison)
        sub = MATCHED_N if name == "drop_film" else None
        rows.append(compute_cohort_metrics(name, cfg, film_scorer=film_scorer, subsample=sub))

    # Also report the FULL drop_film (84k) with cheap metrics only (no FiLM rescoring)
    rows.append(
        compute_cohort_metrics(
            "drop_film", COHORT_FILES["drop_film"],
            film_scorer=None,  # skip FiLM rescoring for 84k row
            subsample=None, label_suffix="_full",
        )
    )

    out_json = RESULTS_DIR / "exp3_ablation_analysis.json"
    out_json.write_text(json.dumps(rows, indent=2, default=float))
    print(f"[json] Wrote {out_json}", file=sys.stderr)

    out_png = RESULTS_DIR / "exp3_ablation_analysis.png"
    make_plot([r for r in rows if r["cohort"] in COHORT_ORDER], out_png)

    out_md = RESULTS_DIR / "exp3_ablation_summary.md"
    make_markdown(rows, out_md)

    print("\n=== SUMMARY ===", file=sys.stderr)
    for r in rows:
        print(
            f"  {r['cohort']:>20s}  N={r['n_total']:>6d}  valid={r['validity_pct']:>5.1f}%  "
            f"warhead={r['smarts_mean_retention']:.3f}  QED={r['qed_mean']:.3f}  "
            f"pIC50={r['film_mean']:.3f}  scaffold_uniq={r['scaffold_uniqueness']:.3f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
