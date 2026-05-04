#!/usr/bin/env python3
"""
Method-by-method overnight report.

For each tier/method:
  1. Two-sentence description
  2. Top-20 candidates with full scoring suite (FiLMDelta pIC50, SAScore, MW,
     QED, shape Tc vs seed, ESP-Sim, warhead Δ°, Tc→Mol1, max Tc→train, warhead intact)

Then a global top-50 across all tiers.

3D scoring is restricted to the top-20 of each method to keep runtime manageable
(~12 methods × 20 mols × ~1s each = ~4 min).
"""

import sys
import json
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

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_zap70_train_smiles,
)

RES = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_HTML = RES / "overnight_method_by_method.html"
OUT_JSON = RES / "overnight_method_by_method.json"

TOP_N_PER_METHOD = 20
TOP_N_GLOBAL = 50


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Method descriptions ──────────────────────────────────────────────────────

METHODS = [
    {
        "name": "Tier 1 — Med-Chem Playbook (rule-based)",
        "csv": RES / "mol1_tier1_rules" / "tier1_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Deterministic SMARTS-based med-chem playbook applied only to non-warhead atoms. "
            "Includes aryl C-H/halogen/CN/CF3/OMe scans, imidazole C2 and N1 substituent variants, "
            "isoindoline ring modifications (gem-dimethyl, gem-difluoro, ring expansion to THIQ), "
            "and amide-linker bioisosteres (sulfonamide, urea, oxadiazole, +CH2 spacers)."
        ),
    },
    {
        "name": "Tier 1.5 — Warhead Controls + Med-Chem Tricks",
        "csv": RES / "mol1_tier1_5_warhead_panel" / "tier1_5_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Two subsections. <b>1.5a:</b> warhead-control panel — α-Me/-F/-CN/-CF3 acrylamide, β-Me/β,β-diMe/β-NMe₂ "
            "(afatinib trick), propiolamide alkyne, 2-butynamide, propanamide saturated null. "
            "<b>1.5b:</b> med-chem tricks the agent panel flagged — imidazole C5-scan, ortho-to-amide substituents "
            "(BTK-vs-TEC selectivity lever), aza-isoindoline fused-N variants, 1-substituted isoindoline (introduces stereocenter)."
        ),
    },
    {
        "name": "Tier 2 — Fragment Replacement (curated 204)",
        "csv": RES / "mol1_tier2_fragreplace" / "tier2_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Fragment A (acrylamide-isoindoline-COOH) held fixed; amine partner replaced via amide coupling. "
            "Uses 34 hand-curated kinase-relevant amines + 39 ChEMBL whole-mol amines + 154 BRICS-decomposed amine fragments → 204 unique amide-coupling products. "
            "Synthesis route is unambiguous: single HATU/T3P amide coupling step."
        ),
    },
    {
        "name": "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)",
        "csv": RES / "aichem_tier2_scaled" / "products_scored.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50_film",
        "description": (
            "Same fragment-replacement strategy as Tier 2, but the amine library was scaled by mining ChEMBL 35 "
            "(2.5M compounds) for primary/secondary amines under MW ≤ 500. "
            "Generated 498,992 unique amide-coupling products and scored all with single-seed FiLMDelta on ai-chem (16-CPU, ~3.5 hr CPU-tiled batched inference)."
        ),
    },
    {
        "name": "Tier 3 v2 — Constrained Generative (single-seed, old)",
        "csv": RES / "mol1_tier3_constrained" / "tier3_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "REINVENT4 with warhead-locked LibInvent scaffold (warhead baked in, only [*:1] varies) + "
            "MatchingSubstructure warhead gate on Mol2Mol/De Novo + protected_ids CReM + warhead-aware BRICS. "
            "Uses the single-seed cached FiLMDelta as scoring component (no uncertainty penalty)."
        ),
    },
    {
        "name": "Tier 3 v3 — LibInvent locked (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "libinvent_locked" / "libinvent_locked_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "Same warhead-locked scaffold as v2, but the scoring component is now the 3-seed FiLMDelta ensemble returning "
            "mean − 0.5·std. The agent is rewarded for high-confidence high-pIC50 picks rather than just high-mean picks, "
            "penalising candidates the model is uncertain about."
        ),
    },
    {
        "name": "Tier 3 v3 — Mol2Mol + warhead gate (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "mol2mol_warhead" / "mol2mol_warhead_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 Mol2Mol seeded on Mol 1 with hard warhead gate (MatchingSubstructure component, weight 1.0 → "
            "geometric-mean → 0 if warhead missing) + uncertainty-aware FiLMDelta reward. "
            "Local SAR exploration around Mol 1 with the warhead preserved by both gating and post-filter."
        ),
    },
    {
        "name": "Tier 3 v3 — De Novo + warhead gate (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "denovo_warhead" / "denovo_warhead_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "From-scratch REINVENT4 De Novo generation with hard warhead gate + Tanimoto-to-Mol-1 reward + "
            "uncertainty-aware FiLMDelta. "
            "Most exploratory of the warhead-preserving methods — generates novel scaffolds that retain the acrylamide."
        ),
    },
    {
        "name": "Tier 4 — De Novo unconstrained",
        "csv": RES / "aigpu_overnight" / "tier4_denovo" / "tier4_denovo_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 De Novo with FiLMDelta-uncertainty (0.7) + QED (0.2) + alerts (0.15) only. NO warhead constraint at generation — "
            "warhead presence is checked only as a post-filter after RL completes. "
            "Tests whether the FiLMDelta reward alone steers the policy toward acrylamide-bearing molecules organically."
        ),
    },
    {
        "name": "Tier 4 — Mol2Mol unconstrained",
        "csv": RES / "aigpu_overnight" / "tier4_mol2mol" / "tier4_mol2mol_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 Mol2Mol seeded on Mol 1 with FiLMDelta-uncertainty + QED reward, no warhead gate, post-filter only. "
            "Compared to Tier 3 v3 Mol2Mol it lets the warhead drift but should still bias toward Mol-1-similar candidates "
            "via the Mol2Mol prior's training distribution."
        ),
    },
    {
        "name": "Method A — De Novo FiLMDelta-driven",
        "csv": RES / "aigpu_overnight" / "method_a" / "method_a_filmdriven_denovo_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "De Novo with FiLMDelta-uncertainty weighted heavily as the PRIMARY reward (weight 0.85), QED 0.15, no warhead constraint. "
            "Pure 'what does the model think is potent?' exploration, no chemistry priors injected via Tc or warhead constraints."
        ),
    },
    {
        "name": "Method B — Mol2Mol FiLMDelta-driven",
        "csv": RES / "aigpu_overnight" / "method_b" / "method_b_filmdriven_mol2mol_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "Mol2Mol seeded on Mol 1 with FiLMDelta-uncertainty (0.7) as primary reward + Tc-to-Mol-1 (0.3) for some chemical anchoring + QED, "
            "no warhead gate. SAR exploration around Mol 1 prioritising potency over warhead preservation."
        ),
    },
]


def svg_of(smi, w=140, h=100):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        AllChem.Compute2DCoords(m)
        d = Draw.MolDraw2DSVG(w, h)
        d.drawOptions().bondLineWidth = 1.0
        d.DrawMolecule(m)
        d.FinishDrawing()
        return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    except Exception:
        return ""


def fmt(v, nd=3, na="—"):
    if pd.isna(v) or v is None:
        return na
    if isinstance(v, bool) or v is True or v is False:
        return "✓" if v else "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def load_method(meta):
    if not meta["csv"].exists():
        return None
    df = pd.read_csv(meta["csv"])
    smi_col, pic_col = meta["smiles_col"], meta["pic50_col"]
    if smi_col not in df.columns or pic_col not in df.columns:
        return None
    sub = df[[smi_col, pic_col]].copy().dropna()
    sub.columns = ["smiles", "pIC50"]
    sub = sub.sort_values("pIC50", ascending=False).drop_duplicates(subset="smiles").reset_index(drop=True)
    sub["method"] = meta["name"]
    return sub


def render_top_table(df_top, n_anchors=None):
    rows = []
    for i, r in df_top.head(TOP_N_PER_METHOD).reset_index(drop=True).iterrows():
        rows.append(f"""
        <tr>
          <td class="num">{i+1}</td>
          <td class="struct">{svg_of(r['smiles'])}</td>
          <td class="smiles">{r['smiles']}</td>
          <td class="num"><b>{fmt(r['pIC50'], 3)}</b></td>
          <td class="num">{fmt(r.get('SAScore'), 2)}</td>
          <td class="num">{fmt(r.get('MW'), 0)}</td>
          <td class="num">{fmt(r.get('QED'), 2)}</td>
          <td class="num">{fmt(r.get('shape_Tc_seed'), 3)}</td>
          <td class="num">{fmt(r.get('esp_sim_seed'), 3)}</td>
          <td class="num">{fmt(r.get('warhead_dev_deg'), 1)}</td>
          <td class="num">{fmt(r.get('Tc_to_Mol1'), 3)}</td>
          <td class="num">{fmt(r.get('max_Tc_train'), 3)}</td>
          <td>{fmt(r.get('warhead_intact'), na='—')}</td>
        </tr>""")
    headers = ("<tr><th>#</th><th>Structure</th><th class='smiles'>SMILES</th>"
               "<th>pIC50</th><th>SAS</th><th>MW</th><th>QED</th>"
               "<th>shape Tc<br/>vs seed</th><th>ESP-Sim<br/>vs seed</th><th>warhead Δ°</th>"
               "<th>Tc→Mol1</th><th>max Tc→<br/>train</th><th>WH intact</th></tr>")
    return f"<table>{headers}{''.join(rows)}</table>"


def main():
    log("=" * 70)
    log("METHOD-BY-METHOD REPORT")
    log("=" * 70)

    train_smiles = load_zap70_train_smiles()
    log(f"Loaded {len(train_smiles)} training SMILES for max-Tc-train scoring")

    method_data = {}
    all_top_for_global = []

    for meta in METHODS:
        df = load_method(meta)
        if df is None or len(df) == 0:
            log(f"  ✗ {meta['name']}: no data")
            continue
        log(f"  ✓ {meta['name']}: {len(df):,} candidates")

        # Take top-20 + score with full suite (3D etc.)
        top = df.head(TOP_N_PER_METHOD).copy()
        log(f"    Scoring top {len(top)} with full suite (3D + ESP-Sim)...")
        top_scored = score_dataframe(
            top.drop(columns=["pIC50"]).rename(columns={}),
            smiles_col="smiles",
            train_smiles=train_smiles,
            compute_3d=True,
            pIC50_predictor=None,  # keep the FiLMDelta pIC50 from the source
        )
        # Restore the pIC50 from source (the FiLMDelta from each method's own scorer)
        top_scored["pIC50"] = top["pIC50"].values
        top_scored["method"] = meta["name"]

        method_data[meta["name"]] = {
            "description": meta["description"],
            "n_total": int(len(df)),
            "max_pIC50": float(df["pIC50"].max()),
            "median_pIC50": float(df["pIC50"].median()),
            "top": top_scored,
        }
        all_top_for_global.append(top_scored)

    if not method_data:
        log("ERROR: no data loaded")
        return

    # ── Global top-50 ────────────────────────────────────────────────────────
    big = pd.concat(all_top_for_global, ignore_index=True).sort_values("pIC50", ascending=False).reset_index(drop=True)
    global_top = big.head(TOP_N_GLOBAL)
    log(f"\nGlobal top-{TOP_N_GLOBAL} built")

    # ── Render HTML ──────────────────────────────────────────────────────────
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;max-width:1850px;color:#222;line-height:1.45;}
    h1{color:#111;border-bottom:3px solid #2563eb;padding-bottom:8px;}
    h2{color:#1f2937;margin-top:48px;border-bottom:1px solid #d1d5db;padding-bottom:4px;}
    h3{color:#374151;margin-top:24px;}
    table{border-collapse:collapse;font-size:12px;margin:8px 0;}
    th,td{border:1px solid #d1d5db;padding:5px 8px;}
    th{background:#f3f4f6;font-weight:600;text-align:center;}
    td.smiles{font-family:monospace;font-size:10px;max-width:300px;word-break:break-all;}
    td.struct{padding:2px;}
    td.num{text-align:right;font-variant-numeric:tabular-nums;}
    tr:nth-child(even) td{background:#fafafa;}
    .desc{font-size:13px;color:#444;max-width:1100px;background:#f9fafb;padding:10px 14px;border-left:3px solid #2563eb;margin:6px 0;}
    .stats{font-size:13px;color:#666;margin:6px 0;}
    .footnote{font-size:12px;color:#555;max-width:1200px;margin:8px 0;}
    .seed-box{display:inline-block;border:2px solid #2563eb;border-radius:6px;padding:8px;}
    """

    sections = []
    for meta in METHODS:
        if meta["name"] not in method_data:
            continue
        md = method_data[meta["name"]]
        sections.append(f"""
        <h2>{meta['name']}</h2>
        <div class="desc">{md['description']}</div>
        <div class="stats">
          <b>Total candidates:</b> {md['n_total']:,} &nbsp; · &nbsp;
          <b>max pIC50:</b> {md['max_pIC50']:.2f} &nbsp; · &nbsp;
          <b>median pIC50:</b> {md['median_pIC50']:.2f}
        </div>
        <h3>Top {TOP_N_PER_METHOD} by FiLMDelta pIC50</h3>
        {render_top_table(md['top'])}
        """)

    # Global top 50
    sections.append(f"""
    <h2>Global Top-{TOP_N_GLOBAL} Across All Methods</h2>
    <p class="footnote">Pooled top candidates from each method's top-{TOP_N_PER_METHOD}, ranked by FiLMDelta pIC50.
    <strong>Caveat</strong>: Tier 3 v3 uses uncertainty-aware reward (mean − 0.5·std) which artificially caps scores
    relative to single-seed methods (Tiers 1, 1.5, 2, 2 SCALED, 2.5, 3 v2, Tier 4, Methods A/B). Cross-method comparison
    is therefore not fully apples-to-apples. Within-method rankings are reliable.</p>
    {render_top_table(global_top, n_anchors=None).replace(f"<th>#</th>", "<th>#</th><th>Method</th>")}
    """)

    # Replace the global table to actually include method column
    rows = []
    for i, r in global_top.head(TOP_N_GLOBAL).reset_index(drop=True).iterrows():
        rows.append(f"""
        <tr>
          <td class="num">{i+1}</td>
          <td>{r['method'].split('—')[0].strip() if '—' in r['method'] else r['method']}</td>
          <td class="struct">{svg_of(r['smiles'])}</td>
          <td class="smiles">{r['smiles']}</td>
          <td class="num"><b>{fmt(r['pIC50'], 3)}</b></td>
          <td class="num">{fmt(r.get('SAScore'), 2)}</td>
          <td class="num">{fmt(r.get('MW'), 0)}</td>
          <td class="num">{fmt(r.get('QED'), 2)}</td>
          <td class="num">{fmt(r.get('shape_Tc_seed'), 3)}</td>
          <td class="num">{fmt(r.get('esp_sim_seed'), 3)}</td>
          <td class="num">{fmt(r.get('warhead_dev_deg'), 1)}</td>
          <td class="num">{fmt(r.get('Tc_to_Mol1'), 3)}</td>
          <td class="num">{fmt(r.get('max_Tc_train'), 3)}</td>
          <td>{fmt(r.get('warhead_intact'), na='—')}</td>
        </tr>""")
    global_table = ("<table>"
        "<tr><th>#</th><th>Method</th><th>Structure</th><th class='smiles'>SMILES</th>"
        "<th>pIC50</th><th>SAS</th><th>MW</th><th>QED</th>"
        "<th>shape Tc<br/>vs seed</th><th>ESP-Sim<br/>vs seed</th><th>warhead Δ°</th>"
        "<th>Tc→Mol1</th><th>max Tc→<br/>train</th><th>WH intact</th></tr>"
        + "".join(rows) + "</table>")

    # Replace placeholder
    sections[-1] = f"""
    <h2>Global Top-{TOP_N_GLOBAL} Across All Methods</h2>
    <p class="footnote">Pooled top candidates from each method's top-{TOP_N_PER_METHOD}, ranked by FiLMDelta pIC50.
    <strong>Caveat</strong>: Tier 3 v3 uses uncertainty-aware reward (mean − 0.5·std) which artificially caps scores
    relative to single-seed methods. Cross-method comparison is not fully apples-to-apples; within-method rankings are reliable.</p>
    {global_table}
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Mol 1 — Method-by-Method Overnight Report</title>
<style>{css}</style></head><body>
<h1>Mol 1 — Method-by-Method Overnight Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Seed: <code>{MOL1_SMILES}</code></p>

<div class="seed-box">{svg_of(MOL1_SMILES, 320, 220)}</div>

<p class="footnote"><b>Column legend:</b>
<b>pIC50</b> = FiLMDelta predicted pIC50; for Tier 3 v3 this is the uncertainty-aware mean−0.5·std reward.
<b>SAS</b> = Ertl-Schuffenhauer synthetic accessibility (1 easy → 10 hard).
<b>shape Tc vs seed</b> = RDKit O3A-aligned Gaussian-volume Tanimoto vs Mol 1 (conformer-ensemble best, 0–1).
<b>ESP-Sim vs seed</b> = Espsim Gasteiger-charge electrostatic Tanimoto (-1 to 1).
<b>warhead Δ°</b> = MIN angle (degrees, conformer ensemble) between seed's C=C–C(=O)–N vector and candidate's after MCS alignment.
<b>Tc→Mol1</b> = Morgan FP r=2/2048 Tanimoto to Mol 1.
<b>max Tc→train</b> = Morgan Tc to nearest of 280 ZAP70 training mols.
<b>WH intact</b> = SMARTS warhead-presence check.</p>

{''.join(sections)}

</body></html>
"""
    OUT_HTML.write_text(html)
    log(f"\nReport → {OUT_HTML}")

    # Save JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "methods": {
            name: {
                "description": d["description"],
                "n_total": d["n_total"],
                "max_pIC50": d["max_pIC50"],
                "median_pIC50": d["median_pIC50"],
                "top_n_smiles": d["top"]["smiles"].tolist(),
            }
            for name, d in method_data.items()
        },
        "global_top": global_top.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(summary, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2))
    log(f"JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
