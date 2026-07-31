"""EXP7 4-way comparison HTML report.

Reads `exp7_all_cohorts_scored.csv` and produces a sortable HTML table.
Conditions to compare: mol2mol_baseline, mol2mol_RL, baseline_prior_anchor (covFT-baseline),
A (covFT-RL).

Per-pair row columns:
  target | pair_id | anchor | drug | Tc(a,d) | dpIC50 |
  for each of 4 conditions: max_Tc, hit_rate@Tc05, mean pred_pIC50,
                            best mol (PNG by Tc-to-drug, by pred_pIC50)

Also: aggregate stats by condition + by contamination subset (loose, strict).
"""
from __future__ import annotations
import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs

SCORED_CSV = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_all_cohorts_scored.csv"
CONTAM_JSON = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_contamination.json"
OUT_HTML = PROJECT_ROOT / "results" / "paper_evaluation" / "exp7_4way_report.html"
IMG_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "exp7_imgs"

# Mapping: nice condition name -> strategy as stored in CSV
COND_MAP = {
    "mol2mol_baseline": "mol2mol_baseline",
    "mol2mol_RL":       "mol2mol_RL",
    "covFT_baseline":   "baseline_prior_anchor",
    "covFT_RL":         "A",
}
COND_ORDER = ["mol2mol_baseline", "mol2mol_RL", "covFT_baseline", "covFT_RL"]


def smi_to_png_b64(smi, size=160) -> str:
    """Render SMILES as base64-encoded PNG data URI."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        img = Draw.MolToImage(m, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{digits}f}"


def main():
    if not SCORED_CSV.exists():
        print(f"ERROR: {SCORED_CSV} missing — run phase3 scoring first")
        return
    df = pd.read_csv(SCORED_CSV)
    print(f"Scored cohorts: {len(df)} ({df['strategy'].value_counts().to_dict()})")
    pairs = load_all_pairs()
    pair_lookup = {p["pair_id"]: p for p in pairs}

    contam = {"pairs": []}
    if CONTAM_JSON.exists():
        contam = json.loads(CONTAM_JSON.read_text())
    contam_lookup = {p["pair_id"]: p for p in contam.get("pairs", [])}

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-pair rows
    rows = []
    for pair in pairs:
        pid = pair["pair_id"]
        cell_row = {
            "pair_id": pid,
            "target": pair["target_key"],
            "anchor_name": pair["anchor_name"],
            "anchor_smi": pair["anchor_smiles"],
            "drug_name": pair["drug_name"],
            "drug_smi": pair["drug_smiles"],
            "tc_ad": pair.get("tc_anchor_drug", float("nan")),
            "dpic50": pair.get("delta_pIC50", float("nan")),
            "anchor_png": smi_to_png_b64(pair["anchor_smiles"]),
            "drug_png": smi_to_png_b64(pair["drug_smiles"]),
            "drug_in_db": contam_lookup.get(pid, {}).get("drug_in_covindb", False),
            "drug_year": contam_lookup.get(pid, {}).get("drug_year"),
        }
        for nice_cond, strategy in COND_MAP.items():
            mask = (df["pair_id"] == pid) & (df["strategy"] == strategy)
            sub = df[mask]
            if len(sub) == 0:
                cell_row[f"{nice_cond}_max_tc"] = float("nan")
                cell_row[f"{nice_cond}_n_tc05"] = float("nan")
                cell_row[f"{nice_cond}_pred_pic_med"] = float("nan")
                cell_row[f"{nice_cond}_pred_pic_max"] = float("nan")
                cell_row[f"{nice_cond}_warhead_pct"] = float("nan")
                cell_row[f"{nice_cond}_qed_med"] = float("nan")
                cell_row[f"{nice_cond}_n_valid"] = 0
                cell_row[f"{nice_cond}_best_tc_smi"] = ""
                cell_row[f"{nice_cond}_best_pic_smi"] = ""
                cell_row[f"{nice_cond}_best_tc_png"] = ""
                cell_row[f"{nice_cond}_best_pic_png"] = ""
                continue
            r = sub.iloc[0]
            cell_row[f"{nice_cond}_max_tc"] = float(r["max_tc_drug"])
            cell_row[f"{nice_cond}_n_tc05"] = int(r["n_tc05"])
            cell_row[f"{nice_cond}_pred_pic_med"] = float(r["pred_pic50_median"])
            cell_row[f"{nice_cond}_pred_pic_max"] = float(r["pred_pic50_max"])
            cell_row[f"{nice_cond}_warhead_pct"] = float(r["warhead_broad_pct"])
            cell_row[f"{nice_cond}_qed_med"] = float(r["qed_median"])
            cell_row[f"{nice_cond}_n_valid"] = int(r["n_cohort_valid"])
            best_tc_smi = str(r.get("best_tc_drug_smiles", ""))
            best_pic_smi = str(r.get("best_pred_pic_smiles", ""))
            cell_row[f"{nice_cond}_best_tc_smi"] = best_tc_smi
            cell_row[f"{nice_cond}_best_pic_smi"] = best_pic_smi
            cell_row[f"{nice_cond}_best_tc_png"] = smi_to_png_b64(best_tc_smi)
            cell_row[f"{nice_cond}_best_pic_png"] = smi_to_png_b64(best_pic_smi)
        rows.append(cell_row)

    # ---- Aggregate stats ----
    def agg_stats(rows_subset):
        st = {}
        for cond in COND_ORDER:
            tcs = [r[f"{cond}_max_tc"] for r in rows_subset if not np.isnan(r.get(f"{cond}_max_tc", float("nan")))]
            n05 = [r[f"{cond}_n_tc05"] for r in rows_subset if not np.isnan(r.get(f"{cond}_n_tc05", float("nan")))]
            pic = [r[f"{cond}_pred_pic_med"] for r in rows_subset if not np.isnan(r.get(f"{cond}_pred_pic_med", float("nan")))]
            war = [r[f"{cond}_warhead_pct"] for r in rows_subset if not np.isnan(r.get(f"{cond}_warhead_pct", float("nan")))]
            st[cond] = {
                "n_pairs": len(tcs),
                "mean_max_tc": float(np.mean(tcs)) if tcs else float("nan"),
                "median_max_tc": float(np.median(tcs)) if tcs else float("nan"),
                "mean_n_tc05": float(np.mean(n05)) if n05 else float("nan"),
                "mean_pred_pic_med": float(np.mean(pic)) if pic else float("nan"),
                "mean_warhead_pct": float(np.mean(war)) if war else float("nan"),
            }
        return st

    all_stats = agg_stats(rows)
    loose_rows = [r for r in rows if not r["drug_in_db"]]
    loose_stats = agg_stats(loose_rows)
    strict_rows = [r for r in rows if (r.get("drug_year") is not None and r["drug_year"] > 2022)]
    strict_stats = agg_stats(strict_rows)

    # ---- HTML ----
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'><title>EXP7 4-way comparison</title>")
    html.append("""<style>
        body { font-family: -apple-system, sans-serif; margin: 20px; color: #222; }
        h1 { font-size: 24px; }
        h2 { margin-top: 36px; font-size: 18px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
        table { border-collapse: collapse; font-size: 11px; }
        th, td { padding: 4px 6px; border: 1px solid #ddd; text-align: left; vertical-align: top; }
        th { background: #f5f5f5; cursor: pointer; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        img { display: block; }
        .cond-mol2mol-baseline { background: #fffde7; }
        .cond-mol2mol-RL       { background: #fff9c4; }
        .cond-covFT-baseline   { background: #e1f5fe; }
        .cond-covFT-RL         { background: #b3e5fc; }
        .header2 { background: #f0f0f0; font-weight: bold; }
        .contam { color: #c62828; font-weight: bold; }
        .strict { color: #2e7d32; font-weight: bold; }
        .stat-table th, .stat-table td { padding: 6px 10px; font-size: 13px; }
        .caveat { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
        .legend { font-size: 11px; color: #555; margin: 8px 0; }
    </style>""")
    html.append("""<script>
    function sortTable(tableId, colIdx, numeric) {
        const tbl = document.getElementById(tableId);
        const tbody = tbl.tBodies[0];
        const rows = Array.from(tbody.rows);
        const asc = !tbl.dataset.sortAsc || tbl.dataset.sortCol !== String(colIdx) ? true : tbl.dataset.sortAsc === 'false';
        rows.sort((a, b) => {
            let va = a.cells[colIdx].dataset.sort || a.cells[colIdx].innerText;
            let vb = b.cells[colIdx].dataset.sort || b.cells[colIdx].innerText;
            if (numeric) { va = parseFloat(va) || -Infinity; vb = parseFloat(vb) || -Infinity; }
            return asc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
        });
        rows.forEach(r => tbody.appendChild(r));
        tbl.dataset.sortAsc = String(asc);
        tbl.dataset.sortCol = String(colIdx);
    }
    </script>""")
    html.append("</head><body>")

    html.append("<h1>EXP7 LO benchmark — 4-condition comparison</h1>")
    html.append(f"""<div class='caveat'>
    <b>Experiment description:</b> 50 (anchor, drug) pairs across 5 covalent kinase targets
    (EGFR-T790M, BTK, JAK3, HER2/pan-ErbB, FGFR). Four generation conditions:
    <ul>
      <li><b>mol2mol_baseline</b> — vanilla REINVENT4 mol2mol prior, no RL, anchor as seed (5500 mols)</li>
      <li><b>mol2mol_RL</b> — vanilla mol2mol + composite RL reward (FiLM+SMARTS+QED), 50 steps, then sample 5500</li>
      <li><b>covFT_baseline</b> — covalent fine-tuned mol2mol prior, no RL, anchor as seed (5500 mols)</li>
      <li><b>covFT_RL</b> — covalent FT prior + composite RL reward, 50 steps, then sample 5500</li>
    </ul>
    <b>Contamination caveat:</b> {contam.get('n_drug_in_covindb', '?')}/{contam.get('n_pairs', 50)} drugs are in CovInDB v2 (FT corpus).
    All drugs in this benchmark predate the mol2mol 2022 cutoff. <b>Loose subset</b> = drug not in CovInDB v2;
    <b>Strict subset</b> = drug first published after 2022 (effectively empty for this benchmark).
    </div>""")

    # --- Aggregate stats tables ---
    def stat_block(name, st, n):
        h = [f"<h2>Aggregate stats — {name} (n={n} pairs)</h2>"]
        h.append("<table class='stat-table'><tr><th>condition</th><th>mean max_Tc</th><th>median max_Tc</th>"
                 "<th>mean n@Tc>=0.5</th><th>mean pred pIC50 median</th><th>mean warhead %</th><th>n with data</th></tr>")
        for cond in COND_ORDER:
            v = st[cond]
            h.append(f"<tr><td>{cond}</td><td class='num'>{fmt(v['mean_max_tc'])}</td>"
                     f"<td class='num'>{fmt(v['median_max_tc'])}</td>"
                     f"<td class='num'>{fmt(v['mean_n_tc05'],1)}</td>"
                     f"<td class='num'>{fmt(v['mean_pred_pic_med'],2)}</td>"
                     f"<td class='num'>{fmt(v['mean_warhead_pct'],1)}</td>"
                     f"<td class='num'>{v['n_pairs']}</td></tr>")
        h.append("</table>")
        return "".join(h)

    html.append(stat_block("All pairs", all_stats, len(rows)))
    html.append(stat_block("Loose subset (drug not in CovInDB v2)", loose_stats, len(loose_rows)))
    html.append(stat_block("Strict subset (drug post-2022)", strict_stats, len(strict_rows)))

    # --- Per-pair table ---
    html.append("<h2>Per-pair detail (click headers to sort)</h2>")
    html.append("<div class='legend'>For each condition: max_Tc to drug | n at Tc>=0.5 | mean pred pIC50 | best mol by Tc-to-drug | best by pred pIC50</div>")
    html.append("<table id='mainTable'>")
    # Header row 1: condition groupings
    html.append("<thead><tr><th colspan='8' class='header2'>PAIR</th>")
    for cond in COND_ORDER:
        cond_dash = cond.replace("_", "-")
        html.append(f"<th colspan='5' class='header2 cond-{cond_dash}'>{cond}</th>")
    html.append("</tr>")
    # Header row 2: column names
    headers = ["#", "target", "pair_id", "anchor", "drug", "Tc(a,d)", "Δ pIC50", "FT-contam"]
    col_idx = 0
    html.append("<tr>")
    for h in headers:
        html.append(f"<th onclick='sortTable(\"mainTable\",{col_idx},true)'>{h}</th>")
        col_idx += 1
    for cond in COND_ORDER:
        cond_dash = cond.replace("_", "-")
        for sub in ["max_Tc", "n@Tc05", "pred_pic", "best Tc mol", "best pic mol"]:
            html.append(f"<th onclick='sortTable(\"mainTable\",{col_idx},true)' class='cond-{cond_dash}'>{sub}</th>")
            col_idx += 1
    html.append("</tr></thead>")
    html.append("<tbody>")
    for i, r in enumerate(rows):
        contam_marker = "<span class='contam'>YES</span>" if r["drug_in_db"] else "<span class='strict'>no</span>"
        html.append(f"<tr>")
        html.append(f"<td>{i+1}</td>")
        html.append(f"<td>{r['target']}</td>")
        html.append(f"<td>{r['pair_id']}</td>")
        html.append(f"<td>{r['anchor_name']}<br><img src='{r['anchor_png']}' width=120></td>")
        html.append(f"<td>{r['drug_name']}<br><img src='{r['drug_png']}' width=120></td>")
        html.append(f"<td class='num' data-sort='{r['tc_ad']}'>{fmt(r['tc_ad'])}</td>")
        html.append(f"<td class='num' data-sort='{r['dpic50']}'>{fmt(r['dpic50'])}</td>")
        html.append(f"<td>{contam_marker}<br>yr={r.get('drug_year','?')}</td>")
        for cond in COND_ORDER:
            cond_dash = cond.replace("_", "-")
            tc = r[f"{cond}_max_tc"]
            n05 = r[f"{cond}_n_tc05"]
            pic = r[f"{cond}_pred_pic_med"]
            tc_png = r[f"{cond}_best_tc_png"]
            tc_smi = r[f"{cond}_best_tc_smi"]
            pic_png = r[f"{cond}_best_pic_png"]
            pic_smi = r[f"{cond}_best_pic_smi"]
            cls = f"cond-{cond_dash}"
            html.append(f"<td class='num {cls}' data-sort='{tc}'>{fmt(tc)}</td>")
            html.append(f"<td class='num {cls}' data-sort='{n05}'>{fmt(n05,0)}</td>")
            html.append(f"<td class='num {cls}' data-sort='{pic}'>{fmt(pic,2)}</td>")
            png_html_t = f"<img src='{tc_png}' width=100 title='{tc_smi}'>" if tc_png else "—"
            png_html_p = f"<img src='{pic_png}' width=100 title='{pic_smi}'>" if pic_png else "—"
            html.append(f"<td class='{cls}'>{png_html_t}</td>")
            html.append(f"<td class='{cls}'>{png_html_p}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    html.append("</body></html>")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(html))
    print(f"Wrote {OUT_HTML}")

    # Also write a small JSON summary
    SUMMARY_JSON = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_4way_summary.json"
    SUMMARY_JSON.write_text(json.dumps({
        "n_pairs": len(rows),
        "n_loose": len(loose_rows),
        "n_strict": len(strict_rows),
        "all": all_stats,
        "loose": loose_stats,
        "strict": strict_stats,
    }, indent=2))
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
