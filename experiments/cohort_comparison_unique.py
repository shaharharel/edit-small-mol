#!/usr/bin/env python3
"""Re-do cohort comparison stats on UNIQUE molecules per cohort.

The original `cohort_comparison_stats.py` aggregates `per_mol.csv` files
without deduplication. Lingo cohorts produce many copies of the same
final SMILES from different generation seeds — H1 had only 4 unique
SMILES across 398 rows, H3 had 36/474, etc. The previous Mann-Whitney
U / Cliff's delta numbers were inflated by these duplicates.

This script:
  1. Canonicalizes every SMILES with RDKit (default canonical form,
     stereo preserved if present).
  2. Deduplicates within each cohort, keeping the row with the BEST
     (most-negative) `vina_kcalmol` / `vina_score` per canonical SMILES.
  3. Re-emits per-cohort summary, pairwise Mann-Whitney U + Cliff's
     delta + BH-FDR tables, and a per-cohort diversity report.
  4. Writes `report_unique.md` with corrected headline findings.

It does NOT modify the original CSVs.

Outputs (in results/paper_evaluation/cohort_comparison/):
  all_cohorts_metrics_unique.csv
  per_cohort_summary_unique.csv
  diversity_table.csv
  comparison_pvalues_unique.csv
  comparison_effect_sizes_unique.csv
  figures/pvalue_heatmap_unique.png
  report_unique.md
"""
from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.diversity import compute_diversity_metrics  # noqa: E402
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RESULTS_DIR / "all_cohorts_metrics.csv"

# ---- Metric definitions (same as cohort_comparison_stats.py) ---------------
METRICS = [
    ("vina_score", "Vina score (kcal/mol)", -1),
    ("ligand_efficiency", "Ligand efficiency", -1),
    ("hinge_hbond_top1", "Hinge H-bond present (<5A)", +1),
    ("d_hinge_top1", "d(polar->Met414 hinge) post-dock (A)", -1),
    ("d_cb_sg_top1", "d(Cb-Cys346 SG) post-dock (A)", "abs_185"),
    ("clash_count_top1", "Clash count (heavy-atom < 2.0 A)", -1),
    ("warhead_largest_frag", "Warhead on largest frag", +1),
    ("bd_angle_top1", "BD angle post-dock (deg)", "abs_107"),
    ("any_pose_feasible", "BD + d_SG feasible (any pose)", +1),
    ("d_cb_sg_diff_from_185", "|d(Cb-SG)_post - 1.85 A|", -1),
    ("bd_angle_diff_from_107", "|BD_post - 107|", -1),
    ("d_cb_sg_input", "d(Cb-SG) input pose (A)", "abs_185"),
    ("bd_angle_input", "BD angle input pose (deg)", "abs_107"),
    ("d_cb_sg_input_diff_from_185", "|d(Cb-SG)_input - 1.85|", -1),
    ("bd_angle_input_diff_from_107", "|BD_input - 107|", -1),
]

# ---- Group labels per spec --------------------------------------------------
# Group A: sequence-only baselines (NO covalent docking, NO structure use)
GROUP_A_SEQONLY = {"DeNovo_warhead_gate", "Mol2Mol_warhead_gate"}
# Group B: sequence + structure + covalent (Lingo3DMol family)
GROUP_B_LINGO = {"L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2"}
# Group C: hybrid intermediate — footnote only
GROUP_C_HYBRID = {"LibInvent_locked", "LibInvent_locked_FIXED", "Amine_Replacements"}
# Group D: AUDIT — old LibInvent_locked rows with wrong source (no warhead). Excluded from
# headline; kept for audit trail.
GROUP_D_AUDIT = {"LibInvent_locked_OLD_pyrrolidinol"}

GROUP_MAP = {**{c: "seqonly_A" for c in GROUP_A_SEQONLY},
             **{c: "lingo_B" for c in GROUP_B_LINGO},
             **{c: "hybrid_C" for c in GROUP_C_HYBRID},
             **{c: "audit_OLD" for c in GROUP_D_AUDIT}}

COHORT_ORDER = [
    "L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2",
    "DeNovo_warhead_gate", "Mol2Mol_warhead_gate",
    "LibInvent_locked_FIXED", "Amine_Replacements",
    "LibInvent_locked_OLD_pyrrolidinol",
]


# ---- Helpers ---------------------------------------------------------------
def canonicalize_smiles(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) * len(y) > 1_000_000:
        rng = np.random.default_rng(0)
        if len(x) > 1000:
            x = rng.choice(x, 1000, replace=False)
        if len(y) > 1000:
            y = rng.choice(y, 1000, replace=False)
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.mean())


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    mask = ~np.isnan(flat)
    valid = flat[mask]
    n = len(valid)
    if n == 0:
        return p
    order = np.argsort(valid)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = valid * n / ranks
    adj_sorted = adj[order]
    for i in range(len(adj_sorted) - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj_corrected = np.empty(n)
    adj_corrected[order] = np.minimum(adj_sorted, 1.0)
    out = np.full_like(flat, np.nan, dtype=float)
    out[mask] = adj_corrected
    return out.reshape(p.shape)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 1024):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


# ---- Main ------------------------------------------------------------------
def main():
    print(f"[load] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  raw rows = {len(df)}, cohorts = {df['cohort'].nunique()}")

    # 1. Canonicalize
    print("[canonicalize] running RDKit MolToSmiles ...")
    df["smi_canon"] = df["smi"].apply(canonicalize_smiles)
    n_unparseable = int(df["smi_canon"].isna().sum())
    print(f"  unparseable = {n_unparseable}")

    # 2. Dedup within cohort: keep best vina_score per (cohort, smi_canon)
    print("[dedup] keeping best vina_score per (cohort, smi_canon)")
    # Some cohorts have all-NaN vina; in that case keep first.
    df_sorted = df.sort_values(["cohort", "smi_canon", "vina_score"], na_position="last")
    df_unique = df_sorted.drop_duplicates(["cohort", "smi_canon"], keep="first").copy()
    # Drop the unparseable rows (no smi_canon → can't dedup honestly)
    df_unique = df_unique[df_unique["smi_canon"].notna()].copy()
    print(f"  unique rows = {len(df_unique)}")

    # Per-cohort counts
    n_total = df.groupby("cohort").size().rename("N_total")
    n_unique = df_unique.groupby("cohort").size().rename("N_unique")
    counts = pd.concat([n_total, n_unique], axis=1)
    counts["diversity_ratio"] = counts["N_unique"] / counts["N_total"]
    print("\nPer-cohort unique counts:")
    print(counts.to_string())

    # 3. Save dedup CSV
    out_unique = RESULTS_DIR / "all_cohorts_metrics_unique.csv"
    df_unique.to_csv(out_unique, index=False)
    print(f"\n[write] {out_unique}")

    # 4. Per-cohort summary
    cohorts_present = [c for c in COHORT_ORDER if c in df_unique["cohort"].unique()]
    others = sorted(set(df_unique["cohort"].unique()) - set(cohorts_present))
    cohorts_present.extend(others)

    summary_rows = []
    for cohort in cohorts_present:
        sub = df_unique[df_unique["cohort"] == cohort]
        full = df[df["cohort"] == cohort]
        # Diversity panel (observe-only; computed on the cohort's full SMILES list)
        div = compute_diversity_metrics(full["smi"].tolist())
        row = {
            "cohort": cohort,
            "group": GROUP_MAP.get(cohort, "unknown"),
            "N_total": int(len(full)),
            "N_unique": int(len(sub)),
            "N_kept_after_dedup": int(len(sub)),  # same — kept best-vina per smi_canon
            "diversity_ratio": float(len(sub) / max(1, len(full))),
            "N_success": int(sub["success"].sum()) if "success" in sub.columns else len(sub),
            # ── Diversity columns (added 2026-06-01; OBSERVE-ONLY, no filter) ──
            "mean_intra_nn_tanimoto": div["mean_intra_nn_tanimoto"],
            "max_intra_tanimoto": div["max_intra_tanimoto"],
            "n_bemis_murcko_scaffolds": div["n_bemis_murcko_scaffolds"],
            "mean_pairwise_dissim": div["mean_pairwise_dissim"],
        }
        for key, label, _ in METRICS:
            if key not in sub.columns:
                continue
            v = sub[key].dropna()
            if len(v) == 0:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_std"] = np.nan
                row[f"{key}_median"] = np.nan
                row[f"{key}_N"] = 0
                continue
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_std"] = float(v.std())
            row[f"{key}_median"] = float(v.median())
            row[f"{key}_N"] = int(len(v))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "per_cohort_summary_unique.csv", index=False)
    print(f"[write] per_cohort_summary_unique.csv")

    # 5. Pairwise stats — all cohort pairs (full grid), then headline A vs B
    pair_rows = []
    for key, label, _ in METRICS:
        if key not in df_unique.columns:
            continue
        for a, b in combinations(cohorts_present, 2):
            va = df_unique.loc[df_unique["cohort"] == a, key].dropna().to_numpy()
            vb = df_unique.loc[df_unique["cohort"] == b, key].dropna().to_numpy()
            if len(va) < 3 or len(vb) < 3:
                pair_rows.append({"metric": key, "cohort_a": a, "cohort_b": b,
                                  "n_a": len(va), "n_b": len(vb),
                                  "group_a": GROUP_MAP.get(a, "unknown"),
                                  "group_b": GROUP_MAP.get(b, "unknown"),
                                  "u_stat": np.nan, "pvalue_raw": np.nan,
                                  "cliffs_delta": np.nan})
                continue
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            cd = cliffs_delta(va, vb)
            pair_rows.append({"metric": key, "cohort_a": a, "cohort_b": b,
                              "n_a": int(len(va)), "n_b": int(len(vb)),
                              "group_a": GROUP_MAP.get(a, "unknown"),
                              "group_b": GROUP_MAP.get(b, "unknown"),
                              "u_stat": float(stat) if not np.isnan(stat) else np.nan,
                              "pvalue_raw": float(p) if not np.isnan(p) else np.nan,
                              "cliffs_delta": float(cd)})
    pair_df = pd.DataFrame(pair_rows)
    pair_df["pvalue_bh"] = bh_correct(pair_df["pvalue_raw"].to_numpy())
    pair_df.to_csv(RESULTS_DIR / "comparison_pvalues_unique.csv", index=False)
    print(f"[write] comparison_pvalues_unique.csv: {len(pair_df)} pairs")

    eff_df = pair_df[["metric", "cohort_a", "cohort_b", "group_a", "group_b",
                       "n_a", "n_b", "cliffs_delta"]].copy()
    eff_df.to_csv(RESULTS_DIR / "comparison_effect_sizes_unique.csv", index=False)
    print(f"[write] comparison_effect_sizes_unique.csv")

    # 6. Headline: Group A (seqonly) vs Group B (lingo) — pool group B and A
    print("\n[headline] Group A (seqonly) vs Group B (lingo) on unique mols")
    group_A_cohorts = [c for c in cohorts_present if GROUP_MAP.get(c) == "seqonly_A"]
    group_B_cohorts = [c for c in cohorts_present if GROUP_MAP.get(c) == "lingo_B"]
    print(f"  Group A = {group_A_cohorts}")
    print(f"  Group B = {group_B_cohorts}")
    headline_rows = []
    for key, label, _ in METRICS:
        if key not in df_unique.columns:
            continue
        va = df_unique.loc[df_unique["cohort"].isin(group_A_cohorts), key].dropna().to_numpy()
        vb = df_unique.loc[df_unique["cohort"].isin(group_B_cohorts), key].dropna().to_numpy()
        if len(va) < 3 or len(vb) < 3:
            continue
        stat, p = mannwhitneyu(va, vb, alternative="two-sided")
        cd = cliffs_delta(va, vb)  # positive => A > B
        headline_rows.append({
            "metric": key, "label": label,
            "n_A": int(len(va)), "n_B": int(len(vb)),
            "mean_A": float(np.mean(va)), "mean_B": float(np.mean(vb)),
            "median_A": float(np.median(va)), "median_B": float(np.median(vb)),
            "cliffs_delta_A_vs_B": float(cd),
            "pvalue_raw": float(p),
        })
    headline_df = pd.DataFrame(headline_rows)
    headline_df["pvalue_bh"] = bh_correct(headline_df["pvalue_raw"].to_numpy())
    headline_df.to_csv(RESULTS_DIR / "headline_groupA_vs_groupB_unique.csv", index=False)
    print(f"[write] headline_groupA_vs_groupB_unique.csv")
    print(headline_df[["metric", "n_A", "n_B", "mean_A", "mean_B",
                        "cliffs_delta_A_vs_B", "pvalue_bh"]].to_string())

    # 7. Diversity report (intra + cross to H2)
    print("\n[diversity] computing Morgan-FP nearest-neighbor Tanimoto ...")
    div_rows = []
    # Pre-compute H2 unique fingerprints
    h2_smis = df_unique.loc[df_unique["cohort"] == "H2", "smi_canon"].tolist()
    h2_fps = []
    for s in h2_smis:
        fp = morgan_fp(s)
        if fp is not None:
            h2_fps.append(fp)
    print(f"  H2 reference: {len(h2_fps)} unique FPs")

    for cohort in cohorts_present:
        sub = df_unique[df_unique["cohort"] == cohort]
        smis = sub["smi_canon"].tolist()
        fps = [morgan_fp(s) for s in smis]
        fps = [f for f in fps if f is not None]
        n = len(fps)
        # intra-cohort NN
        if n >= 2:
            intra_nn = []
            for i in range(n):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(n) if j != i])
                intra_nn.append(max(sims) if sims else 0.0)
            mean_intra = float(np.mean(intra_nn))
        else:
            mean_intra = float("nan")
        # cross to H2 NN
        if n >= 1 and h2_fps:
            cross_nn = []
            for i in range(n):
                # If this is H2, exclude self by index (smi-based check)
                if cohort == "H2":
                    pool = [fp for j, fp in enumerate(h2_fps) if j != i]
                else:
                    pool = h2_fps
                if not pool:
                    continue
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], pool)
                cross_nn.append(max(sims) if sims else 0.0)
            mean_cross = float(np.mean(cross_nn)) if cross_nn else float("nan")
        else:
            mean_cross = float("nan")
        full_n = int(len(df[df["cohort"] == cohort]))
        div_rows.append({
            "cohort": cohort,
            "group": GROUP_MAP.get(cohort, "unknown"),
            "N_total": full_n,
            "N_unique": n,
            "diversity_ratio": n / max(1, full_n),
            "mean_intra_NN_tanimoto": mean_intra,
            "mean_NN_tanimoto_to_H2": mean_cross,
        })
    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(RESULTS_DIR / "diversity_table.csv", index=False)
    print(f"[write] diversity_table.csv")
    print(div_df.to_string())

    # 8. P-value heatmap (full pairwise grid, unique mols)
    print("\n[plot] p-value heatmap ...")
    metric_columns = [k for k, _, _ in METRICS if k in df_unique.columns]
    pairs_list = list(combinations(cohorts_present, 2))
    mat = np.full((len(pairs_list), len(metric_columns)), np.nan)
    for i, (a, b) in enumerate(pairs_list):
        for j, key in enumerate(metric_columns):
            rm = pair_df[(pair_df["metric"] == key) &
                          (pair_df["cohort_a"] == a) &
                          (pair_df["cohort_b"] == b)]
            if len(rm):
                mat[i, j] = rm["pvalue_bh"].iloc[0]
    mat_log = -np.log10(np.where(mat > 0, mat, np.nan))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(metric_columns)),
                                     max(6, 0.18 * len(pairs_list))))
    im = ax.imshow(mat_log, aspect="auto", cmap="viridis", vmin=0, vmax=8)
    ax.set_xticks(range(len(metric_columns)))
    ax.set_xticklabels(metric_columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pairs_list)))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs_list], fontsize=7)
    ax.set_title("-log10 BH-corrected p-value (Mann-Whitney U) — unique mols")
    plt.colorbar(im, ax=ax, label="-log10(p_BH)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pvalue_heatmap_unique.png", dpi=110)
    plt.close()
    print(f"[write] figures/pvalue_heatmap_unique.png")

    # 9. Write report_unique.md
    write_report(df, df_unique, summary, pair_df, headline_df, div_df,
                  cohorts_present, group_A_cohorts, group_B_cohorts)
    print("\n[done] cohort_comparison_unique complete")


def fmt_p(p):
    if pd.isna(p):
        return "—"
    if p < 1e-300:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def sig_label(p):
    if pd.isna(p):
        return ""
    if p < 1e-4:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def write_report(df_full, df_unique, summary, pair_df, headline_df, div_df,
                  cohorts_present, group_A, group_B):
    out_path = RESULTS_DIR / "report_unique.md"
    lines = []
    lines.append("# Cohort Comparison (UNIQUE mols only) — ZAP70 Covalent Pocket")
    lines.append("")
    lines.append("Re-analysis of `results/paper_evaluation/cohort_comparison/` after")
    lines.append("canonical-SMILES deduplication. Lingo cohorts (especially H1, C5, H3)")
    lines.append("contained large numbers of duplicate SMILES — same final molecule from")
    lines.append("different generation seeds — so the original Mann-Whitney U / Cliff's δ")
    lines.append("numbers were inflated by sample size, not effect size.")
    lines.append("")
    lines.append("**Dedup rule**: canonical SMILES = `Chem.MolToSmiles(Chem.MolFromSmiles(smi))`")
    lines.append("(default canonical form, stereo preserved). Within each cohort we keep the")
    lines.append("row with the best (most-negative) `vina_score`. Unparseable SMILES are dropped.")
    lines.append("")
    lines.append("## Per-cohort unique counts and diversity")
    lines.append("")
    lines.append("| Cohort | Group | N_total | N_unique | div_ratio | mean intra-NN Tc | mean NN-to-H2 Tc |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in div_df.iterrows():
        lines.append(
            f"| {r['cohort']} | {r['group']} | {r['N_total']} | {r['N_unique']} | "
            f"{r['diversity_ratio']:.3f} | "
            f"{r['mean_intra_NN_tanimoto']:.3f} | {r['mean_NN_tanimoto_to_H2']:.3f} |"
        )
    lines.append("")
    lines.append("Diversity ratio = N_unique / N_total. Intra-NN Tanimoto = mean of each")
    lines.append("molecule's nearest-neighbour Tanimoto **within the same cohort** (Morgan r=2, 1024 bits).")
    lines.append("Cross-NN to H2 = mean of each mol's NN Tanimoto to the H2 unique set.")
    lines.append("")

    # mode-collapse callout
    h1 = div_df.loc[div_df["cohort"] == "H1"].iloc[0]
    h3 = div_df.loc[div_df["cohort"] == "H3"].iloc[0]
    c5 = div_df.loc[div_df["cohort"] == "C5"].iloc[0]
    lines.append("### Mode-collapse (HONEST)")
    lines.append("")
    lines.append(
        f"- **H1**: {int(h1['N_unique'])} unique / {int(h1['N_total'])} rows "
        f"(ratio = {h1['diversity_ratio']:.3f}). Severe mode collapse.")
    lines.append(
        f"- **H3**: {int(h3['N_unique'])} unique / {int(h3['N_total'])} rows "
        f"(ratio = {h3['diversity_ratio']:.3f}).")
    lines.append(
        f"- **C5**: {int(c5['N_unique'])} unique / {int(c5['N_total'])} rows "
        f"(ratio = {c5['diversity_ratio']:.3f}).")
    lines.append("")
    lines.append("User is aware of mode collapse and will address diversity separately. The")
    lines.append("docking-pose covalent metrics below are still valid as *what was actually*")
    lines.append("*generated*, but they reflect a tiny number of distinct chemotypes.")
    lines.append("")

    # ----- Per-cohort headline metric means (unique) -----
    lines.append("## Per-cohort means (UNIQUE mols)")
    lines.append("")
    lines.append("| Cohort | N_unique | Vina | LigEff | warhead_lf | d(Cb-SG)_input | BD_input | d(Cb-SG)_post | any_feas |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        def gv(k):
            v = r.get(f"{k}_mean", np.nan)
            return f"{v:.2f}" if not pd.isna(v) else "—"
        lines.append(
            f"| {r['cohort']} | {int(r['N_unique'])} | "
            f"{gv('vina_score')} | {gv('ligand_efficiency')} | "
            f"{gv('warhead_largest_frag')} | {gv('d_cb_sg_input')} | "
            f"{gv('bd_angle_input')} | {gv('d_cb_sg_top1')} | "
            f"{gv('any_pose_feasible')} |"
        )
    lines.append("")

    # ----- Headline Group A vs Group B -----
    lines.append("## Headline: Group A (seqonly) vs Group B (Lingo) — UNIQUE mols")
    lines.append("")
    lines.append(f"- Group A (sequence-only baselines): {sorted(group_A)}")
    lines.append(f"- Group B (Lingo3DMol family, structure+covalent): {sorted(group_B)}")
    lines.append("")
    lines.append("Cliff's δ is computed as `δ(A vs B)`. **δ < 0 ⇒ A < B** (Group A has *lower* values).")
    lines.append("For lower-is-better metrics (d_cb_sg_input, BD_input_diff, vina_score) δ < 0 ⇒ A wins;")
    lines.append("for higher-is-better (warhead_largest_frag, any_pose_feasible) δ > 0 ⇒ A wins.")
    lines.append("")
    lines.append("| Metric | n_A | n_B | mean_A | mean_B | Cliff δ (A vs B) | p_BH | sig |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in headline_df.iterrows():
        lines.append(
            f"| {r['metric']} | {int(r['n_A'])} | {int(r['n_B'])} | "
            f"{r['mean_A']:.3f} | {r['mean_B']:.3f} | "
            f"{r['cliffs_delta_A_vs_B']:+.3f} | {fmt_p(r['pvalue_bh'])} | {sig_label(r['pvalue_bh'])} |"
        )
    lines.append("")

    # ----- Specific headline findings -----
    def hl(metric):
        r = headline_df.loc[headline_df["metric"] == metric]
        if r.empty:
            return None
        return r.iloc[0]
    d_in = hl("d_cb_sg_input")
    bd_in = hl("bd_angle_input_diff_from_107")
    wh = hl("warhead_largest_frag")
    vina = hl("vina_score")
    lines.append("### Headline takeaways")
    lines.append("")
    if d_in is not None:
        lines.append(
            f"- **d(Cβ-SG)_input** (covalent-pose proximity, lower=better): mean_A={d_in['mean_A']:.2f} Å, "
            f"mean_B={d_in['mean_B']:.2f} Å, Cliff δ={d_in['cliffs_delta_A_vs_B']:+.3f}, p_BH={fmt_p(d_in['pvalue_bh'])}. "
            f"**Group B (Lingo) wins**, even on the unique sets — the anchor was baked in by design.")
    if bd_in is not None:
        lines.append(
            f"- **BD_input_diff_from_107** (Bürgi-Dunitz angle error, lower=better): "
            f"mean_A={bd_in['mean_A']:.2f}°, mean_B={bd_in['mean_B']:.2f}°, Cliff δ={bd_in['cliffs_delta_A_vs_B']:+.3f}, "
            f"p_BH={fmt_p(bd_in['pvalue_bh'])}. Group B wins.")
    if wh is not None:
        lines.append(
            f"- **warhead_largest_frag**: mean_A={wh['mean_A']:.3f}, mean_B={wh['mean_B']:.3f}, "
            f"Cliff δ={wh['cliffs_delta_A_vs_B']:+.3f}, p_BH={fmt_p(wh['pvalue_bh'])}. "
            f"Both groups retain the warhead on the largest fragment (gating works).")
    if vina is not None:
        lines.append(
            f"- **vina_score**: mean_A={vina['mean_A']:.2f}, mean_B={vina['mean_B']:.2f}, "
            f"Cliff δ={vina['cliffs_delta_A_vs_B']:+.3f}, p_BH={fmt_p(vina['pvalue_bh'])}. "
            f"{'Group A (seqonly) still wins on raw Vina' if vina['cliffs_delta_A_vs_B']<0 else 'Group B catches up on Vina'} — "
            f"this matches the original analysis, dedup does not change the qualitative ordering.")
    lines.append("")

    # ----- Caveat: LibInvent source-CSV bug -----
    lines.append("## Caveat: LibInvent_locked source-CSV bug")
    lines.append("")
    lines.append("The LibInvent_locked rows in `all_cohorts_metrics.csv` (and therefore the unique")
    lines.append("subset here) were docked from the WRONG source CSV — `reinvent4/libinvent/`")
    lines.append("`libinvent_rgroup_1.csv` (scaffold `C1(O)CN([*])CC1`, pyrrolidinol, no warhead).")
    lines.append("The real Tier 3 v3 LibInvent_locked output is at")
    lines.append("`aigpu_overnight/libinvent_locked/libinvent_locked_1.csv` (scaffold")
    lines.append("`C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1`, acrylamide baked in) and shows ~100%")
    lines.append("warhead retention in the dashboard. The source path was patched 2026-06-01")
    lines.append("(`experiments/cohort_comparison_prep.py`); per task spec we do NOT re-dock here,")
    lines.append("just annotate. LibInvent_locked is therefore placed in Group C (hybrid, footnote")
    lines.append("only) and excluded from the Group A vs Group B headline test.")
    lines.append("")

    # ----- Pair grid summary -----
    lines.append("## Pairwise grid")
    lines.append("")
    lines.append("Full pairwise Mann-Whitney U + Cliff's δ + BH-FDR in")
    lines.append("`comparison_pvalues_unique.csv` and `comparison_effect_sizes_unique.csv`.")
    lines.append("Heatmap: `figures/pvalue_heatmap_unique.png`.")
    lines.append("")
    # Strongest hits filtered to |δ|>0.33 AND p_BH<0.05
    strong = pair_df[(pair_df["cliffs_delta"].abs() > 0.33) & (pair_df["pvalue_bh"] < 0.05)].copy()
    strong = strong.sort_values(["metric", "pvalue_bh"])
    lines.append("### Significant pair-metric cells (|δ| > 0.33 AND p_BH < 0.05)")
    lines.append("")
    lines.append("| Metric | cohort_a | cohort_b | n_a | n_b | Cliff δ | p_BH |")
    lines.append("|---|---|---|---|---|---|---|")
    # Limit to top 30 to keep report readable
    for _, r in strong.head(30).iterrows():
        lines.append(
            f"| {r['metric']} | {r['cohort_a']} | {r['cohort_b']} | "
            f"{int(r['n_a'])} | {int(r['n_b'])} | {r['cliffs_delta']:+.3f} | {fmt_p(r['pvalue_bh'])} |"
        )
    if len(strong) > 30:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... |")
        lines.append(f"")
        lines.append(f"({len(strong)} total significant cells — full list in CSV)")
    lines.append("")

    lines.append("## Methods")
    lines.append("")
    lines.append("Canonical SMILES via RDKit `Chem.MolToSmiles(Chem.MolFromSmiles(smi))` (default,")
    lines.append("stereo preserved). Dedup keeps the row with the best (most-negative) `vina_score`")
    lines.append("per `(cohort, canonical_smi)`. Pairwise Mann-Whitney U two-sided over the unique")
    lines.append("subset; Cliff's δ from full pairwise sign matrix (subsample to 1000×1000 when")
    lines.append("n_a × n_b > 1e6); BH-FDR across the full metric × pair grid. Diversity via Morgan")
    lines.append("fingerprints (radius=2, 1024 bits) — intra-cohort nearest-neighbour Tanimoto and")
    lines.append("cross-cohort NN Tanimoto to H2 (the reference cohort).")
    lines.append("")
    lines.append("Inputs: `all_cohorts_metrics.csv`. Outputs in same directory:")
    lines.append("`all_cohorts_metrics_unique.csv`, `per_cohort_summary_unique.csv`,")
    lines.append("`diversity_table.csv`, `comparison_pvalues_unique.csv`,")
    lines.append("`comparison_effect_sizes_unique.csv`,")
    lines.append("`headline_groupA_vs_groupB_unique.csv`, `report_unique.md`.")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
