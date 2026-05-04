#!/usr/bin/env python3
"""
Unified report for Mol 1 Tier 1 / Tier 2 / Tier 3 generation runs.

Loads:
  - tier1_candidates.csv (rule-based med-chem playbook)
  - tier2_candidates.csv (fragment replacement)
  - tier3_candidates.csv (constrained generative, warhead-filtered)

Produces a single HTML report comparing all three approaches.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tiers_report.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
RES = PROJECT_ROOT / "results" / "paper_evaluation"
T1 = RES / "mol1_tier1_rules" / "tier1_candidates.csv"
T2 = RES / "mol1_tier2_fragreplace" / "tier2_candidates.csv"
T3 = RES / "mol1_tier3_constrained" / "tier3_candidates.csv"
T3_JSON = RES / "mol1_tier3_constrained" / "tier3_results.json"
OUT_HTML = RES / "mol1_expansion" / "tiers_unified_report.html"
OUT_JSON = RES / "mol1_expansion" / "tiers_unified_summary.json"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def svg_of(smi, w=180, h=130):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    AllChem.Compute2DCoords(m)
    d = Draw.MolDraw2DSVG(w, h)
    d.drawOptions().bondLineWidth = 1.2
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def load_tier(path, tier_label):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['tier'] = tier_label
    return df


def fmt_num(v, nd=3):
    if pd.isna(v):
        return "—"
    return f"{v:.{nd}f}"


def render_top_table(df, top_n=20, extra_cols=None):
    extra_cols = extra_cols or []
    rows = []
    for i, r in df.head(top_n).iterrows():
        extras = "".join(f"<td>{r.get(c, '—')}</td>" for c in extra_cols)
        sas = fmt_num(r.get('SAScore', float('nan')), 2)
        pains = int(r.get('PAINS_alerts', 0)) if not pd.isna(r.get('PAINS_alerts', float('nan'))) else "—"
        rows.append(f"""
        <tr>
          <td>{i+1}</td>
          <td class="struct">{svg_of(r['smiles'])}</td>
          <td class="smiles">{r['smiles']}</td>
          <td><b>{fmt_num(r['pIC50'])}</b></td>
          <td>{sas}</td>
          <td>{pains}</td>
          <td>{fmt_num(r.get('shape_Tc_seed'))}</td>
          <td>{fmt_num(r.get('warhead_dev_deg'), 1)}</td>
          <td>{fmt_num(r.get('Tc_to_Mol1'))}</td>
          <td>{fmt_num(r.get('max_Tc_train'))}</td>
          <td>{fmt_num(r.get('mean_top10_Tc_train'))}</td>
          <td>{fmt_num(r.get('MW'), 0)}</td>
          <td>{fmt_num(r.get('QED'), 2)}</td>
          {extras}
        </tr>""")
    headers = (
        "<tr><th>#</th><th>Structure</th><th>SMILES</th><th>pIC50</th>"
        "<th>SAScore</th><th>PAINS</th>"
        "<th>shape Tc</th><th>warhead Δ°</th>"
        "<th>Tc→Mol1</th><th>max Tc→train</th><th>mean Tc→top10 train</th>"
        "<th>MW</th><th>QED</th>"
    )
    for c in extra_cols:
        headers += f"<th>{c}</th>"
    headers += "</tr>"
    return f'<table>{headers}{"".join(rows)}</table>'


def histogram(values, bins=20):
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return None
    h, edges = np.histogram(vals, bins=bins)
    return list(zip(edges[:-1].tolist(), edges[1:].tolist(), h.tolist()))


def main():
    print(f"[+] Loading tier results...")
    t1 = load_tier(T1, "Tier 1 — Rule-based")
    t2 = load_tier(T2, "Tier 2 — Frag-replace")
    t3 = load_tier(T3, "Tier 3 — Constrained generative")
    print(f"  Tier 1: {len(t1)} | Tier 2: {len(t2)} | Tier 3: {len(t3)}")

    # Per-tier summary
    summary = {}
    for label, df in [("tier1", t1), ("tier2", t2), ("tier3", t3)]:
        if df.empty:
            continue
        summary[label] = {
            "n": len(df),
            "pIC50_mean": float(df['pIC50'].mean()),
            "pIC50_max": float(df['pIC50'].max()),
            "pIC50_median": float(df['pIC50'].median()),
            "n_potent_7": int((df['pIC50'] >= 7.0).sum()),
            "n_potent_8": int((df['pIC50'] >= 8.0).sum()),
            "SAScore_mean": float(df['SAScore'].mean()) if 'SAScore' in df else None,
            "SAScore_median": float(df['SAScore'].median()) if 'SAScore' in df else None,
            "shape_Tc_mean": float(df['shape_Tc_seed'].mean()) if 'shape_Tc_seed' in df else None,
            "warhead_dev_median": float(df['warhead_dev_deg'].median()) if 'warhead_dev_deg' in df else None,
            "max_Tc_train_mean": float(df['max_Tc_train'].mean()) if 'max_Tc_train' in df else None,
            "MW_mean": float(df['MW'].mean()) if 'MW' in df else None,
            "QED_mean": float(df['QED'].mean()) if 'QED' in df else None,
        }

    # Tier 3 retention info
    retention = None
    if T3_JSON.exists():
        retention = json.loads(T3_JSON.read_text()).get("retention_per_method", {})

    # Combined ranking
    combined = pd.concat([t1, t2, t3], ignore_index=True) if not (t1.empty and t2.empty and t3.empty) else pd.DataFrame()
    combined = combined.sort_values('pIC50', ascending=False).reset_index(drop=True) if not combined.empty else combined

    # ── Render HTML ────────────────────────────────────────────────────────────
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; max-width: 1700px; color: #222; }
    h1 { color: #111; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
    h2 { color: #1f2937; margin-top: 36px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
    h3 { color: #374151; margin-top: 24px; }
    table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 9px; text-align: right; }
    th { background: #f3f4f6; font-weight: 600; }
    td.smiles { text-align: left; max-width: 320px; word-break: break-all; font-family: monospace; font-size: 11px; }
    td.struct { padding: 2px; }
    tr:nth-child(even) td { background: #fafafa; }
    .seed-card { border: 2px solid #2563eb; border-radius: 8px; padding: 12px; max-width: 360px; }
    .tier-card { display: inline-block; vertical-align: top; margin: 8px; padding: 12px 18px;
                  border: 1px solid #d1d5db; border-radius: 8px; min-width: 240px; background: #f9fafb; }
    .tier-card h4 { margin-top: 0; color: #1f2937; font-size: 15px; }
    .tier-card .big { font-size: 28px; font-weight: 600; color: #2563eb; }
    .tier-card .lbl { font-size: 12px; color: #6b7280; }
    .footnote { font-size: 12px; color: #555; max-width: 1200px; margin: 12px 0; }
    .pill-tier1 { background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:10px;font-size:11px; }
    .pill-tier2 { background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;font-size:11px; }
    .pill-tier3 { background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:10px;font-size:11px; }
    .warning { background:#fef2f2;border-left:4px solid #dc2626;padding:10px 14px;font-size:13px;color:#7f1d1d; }
    """

    # Summary cards
    cards_html = []
    for label, key, color in [("Tier 1 — Rule-based", "tier1", "#1e40af"),
                              ("Tier 2 — Frag-replace", "tier2", "#166534"),
                              ("Tier 3 — Constrained generative", "tier3", "#92400e")]:
        s = summary.get(key, {})
        if not s:
            continue
        cards_html.append(f"""
        <div class="tier-card" style="border-color:{color};">
          <h4 style="color:{color};">{label}</h4>
          <div><span class="big">{s['n']}</span> <span class="lbl">candidates</span></div>
          <div><span class="big" style="font-size:20px;">{s['pIC50_max']:.2f}</span> <span class="lbl">max pIC50</span>
               &nbsp;<span class="lbl">(median {s['pIC50_median']:.2f})</span></div>
          <div><span class="lbl">≥7.0:</span> {s['n_potent_7']} &nbsp; <span class="lbl">≥8.0:</span> {s['n_potent_8']}</div>
          <div><span class="lbl">SAScore median:</span> {fmt_num(s.get('SAScore_median'), 2)}</div>
          <div><span class="lbl">shape Tc mean:</span> {fmt_num(s.get('shape_Tc_mean'), 3)}</div>
          <div><span class="lbl">warhead dev median:</span> {fmt_num(s.get('warhead_dev_median'), 1)}°</div>
          <div><span class="lbl">max Tc→train mean:</span> {fmt_num(s.get('max_Tc_train_mean'), 3)}</div>
          <div><span class="lbl">MW mean:</span> {fmt_num(s.get('MW_mean'), 0)} &nbsp;
               <span class="lbl">QED mean:</span> {fmt_num(s.get('QED_mean'), 2)}</div>
        </div>
        """)

    # Tier 3 retention diagnostic table
    retention_html = ""
    if retention:
        items = sorted(retention.items(), key=lambda x: -x[1]["retention_pct"])
        rows = "".join(
            f"<tr><td>{m}</td><td>{r['warhead_intact']}</td><td>{r['total']}</td><td>{r['retention_pct']:.1f}%</td></tr>"
            for m, r in items
        )
        retention_html = f"""
        <h3>Diagnostic: warhead retention rate per generator (Phase 1+2 raw outputs)</h3>
        <p class="footnote">Out of the original 39,203 candidates from CReM/BRICS/MMP/Mol2Mol/De Novo/LibInvent,
        what fraction kept the acrylamide intact under SMARTS check <code>[CH2]=[CH]C(=O)[N;!H2]</code>?</p>
        <table>
          <tr><th>Generator</th><th>Warhead intact</th><th>Total</th><th>Retention</th></tr>
          {rows}
        </table>
        """

    # Top-20 across all tiers
    top_combined_html = ""
    if not combined.empty:
        # add tier label to table
        combined_display = combined.head(30).copy()
        combined_display['tier_label'] = combined_display['tier'].apply(
            lambda t: f'<span class="pill-tier{t.split()[1]}">' + t + '</span>'
        )
        top_combined_html = render_top_table(combined_display, top_n=30, extra_cols=['tier_label'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mol 1 Tiered Generation — Unified Report</title>
<style>{css}</style>
</head>
<body>
<h1>Mol 1 Tiered Generation — Unified Report</h1>
<p class="footnote">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
   Seed: <code>{MOL1_SMILES}</code></p>

<h2>1. Seed</h2>
<div class="seed-card">{svg_of(MOL1_SMILES, 320, 220)}</div>

<h2>2. Tier Summary</h2>
<p class="footnote">
  <b>Tier 1 (Rule-based)</b> — deterministic SMARTS transforms (methyl scan, halogen scan, bioisosteric ring
  swaps, N-walk, isopropyl variants, linker tweaks). Every output is human-interpretable; warhead atoms are
  protected by construction.
  <br><b>Tier 2 (Fragment-replace)</b> — Fragment A (acrylamide-isoindoline carboxylic acid) is held fixed;
  Fragment B (the amine partner) is replaced from a curated library (hand-picked + ChEMBL kinase amines +
  BRICS-derived fragments). Single amide-coupling synthesis route.
  <br><b>Tier 3 (Constrained generative)</b> — top-N per generator from the original Phase 1+2 expansion
  (BRICS/CReM/MMP/Mol2Mol/De Novo/LibInvent), filtered to warhead-intact only, then re-scored with the same
  full suite.
</p>
<div>{''.join(cards_html)}</div>

<h2>3. Common Scoring Columns</h2>
<p class="footnote">Every candidate gets the same scoring suite. <b>No hard filters</b> beyond the warhead
  presence — SAScore, MW, PAINS, etc. are reported for analysis only.</p>
<ul style="font-size:13px;color:#374151;max-width:1100px;">
  <li><b>pIC50</b> — FiLMDelta anchor-mean prediction (model trained on 280 ZAP70 mols, kinase pre-trained).
      Same model used for the Phase 1+2 scoring (<code>reinvent4_film_model.pt</code>).</li>
  <li><b>SAScore</b> — Ertl &amp; Schuffenhauer 2009 synthetic accessibility, 1 (easy) → 10 (hard).</li>
  <li><b>PAINS</b> — count of PAINS alerts (RDKit FilterCatalog). 0 means clean.</li>
  <li><b>shape Tc</b> — RDKit O3A-aligned Gaussian-volume Tanimoto vs Mol 1's 3D conformer (1 = identical
      shape, 0 = unrelated).</li>
  <li><b>warhead Δ°</b> — angle between the C=C–C(=O)–N vector in candidate vs seed after MCS-based 3D
      alignment. Direct proxy for "does the warhead still point in the same direction".</li>
  <li><b>Tc→Mol1, max Tc→train, mean Tc→top10 train</b> — standard 2D Morgan Tanimoto novelty measures.</li>
</ul>

<h2>4. Tier 1 — Med-Chem Playbook (top {min(20, len(t1))})</h2>
{render_top_table(t1, 20, extra_cols=['method', 'note'])}

<h2>5. Tier 2 — Fragment Replacement (top {min(20, len(t2))})</h2>
{render_top_table(t2, 20, extra_cols=['amine_source', 'note'])}

<h2>6. Tier 3 — Constrained Generative (top {min(20, len(t3))})</h2>
{render_top_table(t3, 20, extra_cols=['method'])}

{retention_html}

<h2>7. Top-30 Across All Tiers (by FiLMDelta pIC50)</h2>
{top_combined_html}

<h2>Caveats</h2>
<div class="warning">
  All candidates sit in the Tc&lt;0.3 extrapolation regime; FiLMDelta MAE on this regime is ~0.86 pIC50
  (extrapolation_test_results.json). Absolute pIC50 numbers should be read as <i>model-ranked top of library</i>,
  not absolute potency claims. Use within-tier and within-cohort rankings for prioritization, and validate top
  picks experimentally.
</div>

</body>
</html>
"""
    OUT_HTML.write_text(html)
    OUT_JSON.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "seed": MOL1_SMILES,
        "summary": summary,
        "retention_per_method": retention,
    }, indent=2))
    print(f"[+] Saved → {OUT_HTML}")
    print(f"[+] Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
