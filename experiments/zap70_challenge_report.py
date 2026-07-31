#!/usr/bin/env python3
"""Generate ZAP70 Challenge leaderboard report (Markdown + HTML)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "zap70_challenge"
RESULTS_JSON = RESULTS_DIR / "all_results.json"
RESULTS_CSV = RESULTS_DIR / "all_results.csv"


def load() -> dict:
    if not RESULTS_JSON.exists():
        raise SystemExit(f"No results at {RESULTS_JSON}")
    with open(RESULTS_JSON) as f:
        return json.load(f)


def build_leaderboard(results: dict) -> pd.DataFrame:
    rows = []
    for model, info in results.items():
        if not isinstance(info, dict) or "aggregated" not in info:
            continue
        for split_name, agg in info["aggregated"].items():
            rows.append({
                "model": model, "split": split_name,
                "MAE_mean": agg.get("mae_mean"), "MAE_std": agg.get("mae_std"),
                "RMSE_mean": agg.get("rmse_mean"), "RMSE_std": agg.get("rmse_std"),
                "R2_mean": agg.get("r2_mean"), "R2_std": agg.get("r2_std"),
                "Spearman_mean": agg.get("spearman_r_mean"), "Spearman_std": agg.get("spearman_r_std"),
                "Pearson_mean": agg.get("pearson_r_mean"), "Pearson_std": agg.get("pearson_r_std"),
                "n_folds": agg.get("n_folds"),
            })
    df = pd.DataFrame(rows)
    return df


def fmt(x, std=None, fmt_str="{:.4f}", std_fmt="{:.4f}"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    base = fmt_str.format(x)
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        base += " ± " + std_fmt.format(std)
    return base


def render_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "# ZAP70 Challenge — no results yet"
    out = ["# ZAP70 Challenge — Leaderboard", ""]
    out.append("Five metrics per fold (MAE, RMSE, R², Spearman, Pearson). Aggregated over 5 folds each split.")
    out.append("")

    for split in ("butina", "random"):
        sub = df[df["split"] == split].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("MAE_mean")
        out.append(f"## Sorted by **{split}** split — MAE (primary)")
        out.append("")
        out.append("| Rank | Model | MAE | RMSE | R² | Spearman | Pearson |")
        out.append("|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(sub.iterrows()):
            out.append(
                f"| {i+1} | **{r['model']}** | {fmt(r['MAE_mean'], r['MAE_std'])} "
                f"| {fmt(r['RMSE_mean'], r['RMSE_std'])} "
                f"| {fmt(r['R2_mean'], r['R2_std'], fmt_str='{:.3f}', std_fmt='{:.3f}')} "
                f"| {fmt(r['Spearman_mean'], r['Spearman_std'], fmt_str='{:.3f}', std_fmt='{:.3f}')} "
                f"| {fmt(r['Pearson_mean'], r['Pearson_std'], fmt_str='{:.3f}', std_fmt='{:.3f}')} |"
            )
        out.append("")

    # v7 reproduction status
    out.append("## v7 Reproduction Status")
    out.append("")
    out.append("| Phase | v7 Target MAE (random) | Current MAE (random) | Within ±0.02? |")
    out.append("|---|---|---|---|")
    targets = {
        "v7A_FiLMDelta_repro": (None, "0.877 (delta MAE)"),
        "v7B_MLP_Morgan_repro": (0.586, "0.586 ± 0.056"),
        "v7C_XGB_Interpretable_repro": (0.562, "0.562 ± 0.053"),
        "v7D_DualObj_Pretrain_repro": (0.470, "0.470 ± 0.041"),
        "v7G1_XGB_Morgan_repro": (0.508, "0.508 ± 0.060"),
        "v7H1_Morgan_KNN10_repro": (0.504, "0.504 ± 0.062 (bias-corr variant)"),
    }
    for model, (target_mae, target_str) in targets.items():
        sub_r = df[(df["model"] == model) & (df["split"] == "random")]
        if sub_r.empty:
            curr_str = "(not run)"
            status = "—"
        else:
            mae = sub_r.iloc[0]["MAE_mean"]
            std = sub_r.iloc[0]["MAE_std"]
            curr_str = f"{mae:.4f} ± {std:.4f}"
            if target_mae is not None:
                status = "PASS" if abs(mae - target_mae) <= 0.02 else f"DIFF={mae-target_mae:+.3f}"
            else:
                status = "(no num target)"
        out.append(f"| {model.replace('v7', '').replace('_repro', '')} | {target_str} | {curr_str} | {status} |")

    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ZAP70 Challenge Leaderboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 30px; max-width: 1400px; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90e2; padding-bottom: 8px; }}
h2 {{ color: #2c3e50; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 24px; font-size: 14px; }}
th {{ background: #4a90e2; color: white; padding: 10px; text-align: left; }}
td {{ padding: 8px 10px; border-bottom: 1px solid #e0e0e0; }}
tr:nth-child(even) {{ background: #f9fafb; }}
tr:hover {{ background: #eef5fb; }}
.top1 {{ background: #d4edda !important; font-weight: bold; }}
.top2 {{ background: #fff3cd !important; }}
.top3 {{ background: #e7f3ff !important; }}
.metric-num {{ font-family: 'SF Mono', Consolas, monospace; text-align: right; }}
.pass {{ color: #28a745; font-weight: bold; }}
.fail {{ color: #dc3545; font-weight: bold; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 4px; }}
.tag-v7 {{ background: #6c757d; color: white; }}
.tag-new {{ background: #28a745; color: white; }}
.notes {{ background: #fff8e1; padding: 14px; border-left: 4px solid #ffc107; margin: 10px 0; }}
</style></head><body>
{body}
</body></html>
"""


def render_html(df: pd.DataFrame, md_body: str) -> str:
    if df.empty:
        body = "<h1>ZAP70 Challenge — no results yet</h1>"
        return HTML_TEMPLATE.format(body=body)

    body_parts = ["<h1>ZAP70 Challenge — Leaderboard</h1>"]
    body_parts.append('<div class="notes"><b>Setup</b>: 280 unique ZAP70 ChEMBL molecules (CHEMBL2803), 5-fold CV ' +
                       '(Random KFold seed=42 + Butina cluster GroupKFold, cutoff=0.35). All models trained with identical splits. ' +
                       'Delta-based models use median-anchor reconstruction to absolute pIC50. Per-fold + aggregated mean ± std reported.</div>')

    for split in ("butina", "random"):
        sub = df[df["split"] == split].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("MAE_mean").reset_index(drop=True)
        body_parts.append(f"<h2>Sorted by <b>{split}</b> split — MAE (primary)</h2>")
        rows_html = ["<table>",
                     "<tr><th>Rank</th><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Spearman</th><th>Pearson</th><th>N folds</th></tr>"]
        for i, r in sub.iterrows():
            cls = ""
            if i == 0: cls = "top1"
            elif i == 1: cls = "top2"
            elif i == 2: cls = "top3"
            tag = '<span class="tag tag-v7">v7 repro</span>' if r["model"].startswith("v7") else '<span class="tag tag-new">new</span>'
            rows_html.append(
                f'<tr class="{cls}"><td>{i+1}</td><td>{r["model"]} {tag}</td>'
                f'<td class="metric-num">{fmt(r["MAE_mean"], r["MAE_std"])}</td>'
                f'<td class="metric-num">{fmt(r["RMSE_mean"], r["RMSE_std"])}</td>'
                f'<td class="metric-num">{fmt(r["R2_mean"], r["R2_std"], fmt_str="{:.3f}", std_fmt="{:.3f}")}</td>'
                f'<td class="metric-num">{fmt(r["Spearman_mean"], r["Spearman_std"], fmt_str="{:.3f}", std_fmt="{:.3f}")}</td>'
                f'<td class="metric-num">{fmt(r["Pearson_mean"], r["Pearson_std"], fmt_str="{:.3f}", std_fmt="{:.3f}")}</td>'
                f'<td>{r["n_folds"]}</td></tr>'
            )
        rows_html.append("</table>")
        body_parts.extend(rows_html)

    # v7 reproduction status section
    body_parts.append("<h2>v7 Reproduction Verification</h2>")
    body_parts.append("<table>")
    body_parts.append("<tr><th>Phase</th><th>v7 Target MAE</th><th>Reproduced MAE</th><th>Gate</th></tr>")
    targets = {
        "v7G1_XGB_Morgan_repro": ("G1 (XGB Morgan binary)", 0.508),
        "v7H1_Morgan_KNN10_repro": ("H1 (Morgan kNN-10)", 0.504),
        "v7B_MLP_Morgan_repro": ("B (MLP Morgan)", 0.586),
        "v7C_XGB_Interpretable_repro": ("C (XGB Interpretable)", 0.562),
        "v7A_FiLMDelta_repro": ("A (FiLMDelta abs)", None),
        "v7D_DualObj_Pretrain_repro": ("D (Dual-obj + pretrain)", 0.470),
    }
    for model, (label, target_mae) in targets.items():
        sub_r = df[(df["model"] == model) & (df["split"] == "random")]
        if sub_r.empty:
            curr_str = "(pending)"; status = "—"; cls = ""
        else:
            mae = sub_r.iloc[0]["MAE_mean"]; std = sub_r.iloc[0]["MAE_std"]
            curr_str = f"{mae:.4f} ± {std:.4f}"
            if target_mae is not None:
                if abs(mae - target_mae) <= 0.02:
                    status = '<span class="pass">PASS</span>'; cls = ""
                else:
                    status = f'<span class="fail">Δ={mae-target_mae:+.3f}</span>'; cls = ""
            else:
                status = "(abs MAE via anchor recon, no v7 num)"
                cls = ""
        target_str = f"{target_mae:.3f}" if target_mae is not None else "— (delta MAE only in v7)"
        body_parts.append(f'<tr class="{cls}"><td>{label}</td><td>{target_str}</td><td>{curr_str}</td><td>{status}</td></tr>')
    body_parts.append("</table>")

    body_parts.append("<h2>Markdown source</h2><pre style='background:#f4f6f8;padding:14px;overflow:auto'>" + md_body.replace("<", "&lt;").replace(">", "&gt;") + "</pre>")

    return HTML_TEMPLATE.format(body="\n".join(body_parts))


def main():
    results = load()
    df = build_leaderboard(results)
    if df.empty:
        print("No aggregated results yet.")
        return
    md = render_md(df)
    out_md = RESULTS_DIR / "report.md"
    out_md.write_text(md)
    print(f"[SAVE] {out_md}")

    html = render_html(df, md)
    out_html = RESULTS_DIR / "report.html"
    out_html.write_text(html)
    print(f"[SAVE] {out_html}")

    # Also save a tabular CSV for easy ingest
    out_summary = RESULTS_DIR / "summary_by_model_split.csv"
    df.to_csv(out_summary, index=False)
    print(f"[SAVE] {out_summary}")

    # Print top of leaderboard
    print("\n" + md.split("\n## v7")[0])


if __name__ == "__main__":
    main()
