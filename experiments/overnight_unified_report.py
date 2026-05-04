#!/usr/bin/env python3
"""
Unified all-tier overnight report.

Loads every tier output (Tier 1, 1.5, 2 original, 2 SCALED, 2.5, 3 v2,
ai-gpu Tier 3 v3 + Tier 4 + Methods A/B), normalises into a single dataframe,
generates HTML.

Output: results/paper_evaluation/overnight_unified_report.html
"""

import sys
import json
import os
import warnings
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog('rdApp.*')

RES = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_HTML = RES / "overnight_unified_report.html"
OUT_JSON = RES / "overnight_unified_summary.json"

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(values):
    """Return summary stats for a list of pIC50 scores (filter NaNs)."""
    arr = np.array([v for v in values if pd.notna(v) and np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return None
    return {
        "n": len(arr),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "n_potent_7": int((arr >= 7.0).sum()),
        "n_potent_8": int((arr >= 8.0).sum()),
    }


def load_existing_tier(name, csv_path, smiles_col="smiles", pic50_col="pIC50"):
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        for alt in ["SMILES", "Smiles", "smiles_clean", "canonical_smiles"]:
            if alt in df.columns:
                smiles_col = alt
                break
    if pic50_col not in df.columns:
        for alt in ["pIC50_film", "FiLMDelta pIC50 (raw)", "FiLMDelta pIC50"]:
            if alt in df.columns:
                pic50_col = alt
                break
    if smiles_col not in df.columns or pic50_col not in df.columns:
        return None, None
    sub = df[[smiles_col, pic50_col]].copy().dropna()
    sub.columns = ["smiles", "pIC50"]
    sub["tier"] = name
    return sub, stats(sub["pIC50"])


def load_aigpu_job(job_name, csv_path):
    """Load a REINVENT4 output CSV. Uses 'FiLMDelta pIC50 (raw)' column."""
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    pic_col = "FiLMDelta pIC50 (raw)"
    smi_col = "SMILES"
    if pic_col not in df.columns or smi_col not in df.columns:
        return None, None
    sub = df[[smi_col, pic_col]].copy().dropna()
    sub.columns = ["smiles", "pIC50"]
    sub["tier"] = job_name
    return sub, stats(sub["pIC50"])


def main():
    log("=" * 70)
    log("UNIFIED OVERNIGHT REPORT")
    log("=" * 70)

    all_dfs = []
    all_stats = {}

    # ── Existing local tiers ──
    sources_local = [
        ("Tier 1 — Rule-based",         RES / "mol1_tier1_rules" / "tier1_candidates.csv",      "smiles", "pIC50"),
        ("Tier 1.5 — Warhead/MedChem",  RES / "mol1_tier1_5_warhead_panel" / "tier1_5_candidates.csv", "smiles", "pIC50"),
        ("Tier 2 — Frag-replace (orig)", RES / "mol1_tier2_fragreplace" / "tier2_candidates.csv", "smiles", "pIC50"),
        ("Tier 2.5 — Post-filter",       RES / "mol1_tier2_5_postfilter" / "tier2_5_candidates.csv", "smiles", "pIC50"),
        ("Tier 3 v2 — Constrained (old)", RES / "mol1_tier3_constrained" / "tier3_candidates.csv", "smiles", "pIC50"),
    ]
    for name, path, sc, pc in sources_local:
        df, st = load_existing_tier(name, path, sc, pc)
        if df is not None:
            all_dfs.append(df)
            all_stats[name] = st
            log(f"  ✓ {name}: {st['n']} mols, max {st['max']:.2f}")
        else:
            log(f"  ✗ {name}: failed to load {path}")

    # ── Tier 2 SCALED (ai-chem) ──
    aichem_csv = RES / "aichem_tier2_scaled" / "products_scored.csv"
    df, st = load_existing_tier("Tier 2 SCALED — Frag-replace 498K", aichem_csv,
                                  smiles_col="smiles", pic50_col="pIC50_film")
    if df is not None:
        all_dfs.append(df)
        all_stats["Tier 2 SCALED — Frag-replace 498K"] = st
        log(f"  ✓ Tier 2 SCALED: {st['n']:,} mols, max {st['max']:.2f}, ≥8: {st['n_potent_8']}")
    else:
        log(f"  ✗ Tier 2 SCALED: failed")

    # ── ai-gpu Tier 3 v3 + Tier 4 + Methods A/B ──
    aigpu_root = RES / "aigpu_overnight"
    job_meta = [
        ("Tier 3 v3 — LibInvent locked", "libinvent_locked", "libinvent_locked_1.csv"),
        ("Tier 3 v3 — Mol2Mol+warhead",  "mol2mol_warhead",  "mol2mol_warhead_1.csv"),
        ("Tier 3 v3 — De Novo+warhead",  "denovo_warhead",   "denovo_warhead_1.csv"),
        ("Tier 4 — De Novo unconstrained", "tier4_denovo",   "tier4_denovo_1.csv"),
        ("Tier 4 — Mol2Mol unconstrained", "tier4_mol2mol",  "tier4_mol2mol_1.csv"),
        ("Method A — De Novo FiLMDelta-driven", "method_a", "method_a_filmdriven_denovo_1.csv"),
        ("Method B — Mol2Mol FiLMDelta-driven", "method_b", "method_b_filmdriven_mol2mol_1.csv"),
    ]
    for label, dirname, csv in job_meta:
        csv_path = aigpu_root / dirname / csv
        df, st = load_aigpu_job(label, csv_path)
        if df is not None:
            all_dfs.append(df)
            all_stats[label] = st
            log(f"  ✓ {label}: {st['n']:,} mols, max {st['max']:.2f}, ≥8: {st['n_potent_8']}")
        else:
            log(f"  ✗ {label}: failed (csv: {csv_path})")

    if not all_dfs:
        log("ERROR: no data loaded")
        return

    big = pd.concat(all_dfs, ignore_index=True)
    log(f"\nTotal candidates across all tiers: {len(big):,}")

    # Save aggregate
    big.to_csv(RES / "overnight_unified_candidates.csv", index=False)

    # Top 100 across all
    top100 = big.sort_values("pIC50", ascending=False).head(100)
    top100.to_csv(RES / "overnight_unified_top100.csv", index=False)
    log(f"Top 100 saved")

    # Build HTML report
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;max-width:1500px;color:#222;}
    h1{color:#111;border-bottom:3px solid #2563eb;padding-bottom:8px;}
    h2{color:#1f2937;margin-top:32px;}
    table{border-collapse:collapse;font-size:13px;margin:8px 0;}
    th,td{border:1px solid #d1d5db;padding:6px 10px;}
    th{background:#f3f4f6;font-weight:600;}
    td.smiles{font-family:monospace;font-size:11px;max-width:380px;word-break:break-all;}
    tr:nth-child(even) td{background:#fafafa;}
    .num{text-align:right;font-variant-numeric:tabular-nums;}
    .top{background:#dbeafe!important;}
    """

    rows = []
    # Order: tier 1, 1.5, 2, 2 SCALED, 2.5, 3 v2, 3 v3 ×3, 4 ×2, methods A/B
    order = list(all_stats.keys())
    for tier in order:
        s = all_stats[tier]
        if s is None:
            continue
        rows.append(f"<tr><td>{tier}</td><td class='num'>{s['n']:,}</td>"
                    f"<td class='num'>{s['max']:.2f}</td>"
                    f"<td class='num'>{s['median']:.2f}</td>"
                    f"<td class='num'>{s['mean']:.2f}</td>"
                    f"<td class='num'>{s['n_potent_7']:,}</td>"
                    f"<td class='num'>{s['n_potent_8']:,}</td></tr>")
    summary_html = (
        "<table><tr><th>Tier / Method</th><th>n</th>"
        "<th>max pIC50</th><th>median</th><th>mean</th>"
        "<th>≥7.0</th><th>≥8.0</th></tr>" + "".join(rows) + "</table>"
    )

    # Top-50 cross-tier table
    top50 = big.sort_values("pIC50", ascending=False).head(50)
    top_rows = []
    for i, r in top50.reset_index(drop=True).iterrows():
        # Compact mol image
        m = Chem.MolFromSmiles(r['smiles'])
        if m is not None:
            try:
                AllChem.Compute2DCoords(m)
                d = Draw.MolDraw2DSVG(140, 100)
                d.drawOptions().bondLineWidth = 1.0
                d.DrawMolecule(m)
                d.FinishDrawing()
                svg = d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
            except Exception:
                svg = ""
        else:
            svg = ""
        top_rows.append(
            f"<tr><td class='num'>{i+1}</td><td>{svg}</td>"
            f"<td class='smiles'>{r['smiles']}</td>"
            f"<td class='num'>{r['pIC50']:.3f}</td>"
            f"<td>{r['tier']}</td></tr>"
        )
    top_html = (
        "<table><tr><th>#</th><th>Structure</th><th>SMILES</th>"
        "<th>pIC50</th><th>Source tier/method</th></tr>" + "".join(top_rows) + "</table>"
    )

    # Method counts in top-50 + top-100
    top50_counts = top50["tier"].value_counts().to_dict()
    top100_counts = top100["tier"].value_counts().to_dict()
    method_counts_html = "<table><tr><th>Tier / Method</th><th>in top-50</th><th>in top-100</th></tr>"
    for t in order:
        method_counts_html += (f"<tr><td>{t}</td>"
                              f"<td class='num'>{top50_counts.get(t,0)}</td>"
                              f"<td class='num'>{top100_counts.get(t,0)}</td></tr>")
    method_counts_html += "</table>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Mol 1 — Overnight Unified Report</title>
<style>{css}</style></head><body>
<h1>Mol 1 — Overnight Unified Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Seed: <code>{MOL1}</code></p>

<h2>1. Per-Tier Summary</h2>
{summary_html}
<p style="font-size:13px;color:#555;max-width:1100px;">
All pIC50 values are FiLMDelta predictions. Tier 3 v3 used the 3-seed
uncertainty-aware ensemble; Tiers 1, 1.5, 2/2.5, 3v2, Tier 4, Methods A/B used
single-seed FiLMDelta. Ensemble vs single-seed differences are not strictly
comparable.</p>

<h2>2. Method Composition of Top Candidates</h2>
<p style="font-size:13px;color:#555;max-width:900px;">Of the top-50 / top-100 by predicted pIC50,
how many came from each method? Methods generating richer top-of-distribution win here.</p>
{method_counts_html}

<h2>3. Top-50 Across All Tiers (by FiLMDelta pIC50)</h2>
<p style="font-size:13px;color:#555;max-width:1100px;"><strong>⚠ caveat</strong>: All candidates
sit in the Tc&lt;0.3 OOD regime where FiLMDelta MAE ≈ 0.86 pIC50. Read pIC50 values as
"model-ranked top of library", not absolute potency. Within-method ranking is more
reliable than cross-method when underlying scoring differs (uncertainty vs single-seed).</p>
{top_html}

<h2>4. Files Available</h2>
<ul style="font-size:13px;">
<li><code>results/paper_evaluation/overnight_unified_candidates.csv</code> — all candidates merged ({len(big):,} mols)</li>
<li><code>results/paper_evaluation/overnight_unified_top100.csv</code> — top 100 ranked</li>
<li><code>results/paper_evaluation/aigpu_overnight/*/{{job_name}}_1.csv</code> — raw REINVENT4 outputs (7 jobs)</li>
<li><code>results/paper_evaluation/aichem_tier2_scaled/products_scored.csv</code> — 498K coupled products with FiLMDelta scores</li>
</ul>

</body></html>
"""
    OUT_HTML.write_text(html)
    OUT_JSON.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "n_total": len(big),
        "stats": all_stats,
        "top50_method_counts": top50_counts,
        "top100_method_counts": top100_counts,
    }, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o))
    log(f"\nReport → {OUT_HTML}")
    log(f"Summary JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
