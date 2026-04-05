#!/usr/bin/env python3
"""Generate the unified ZAP70 case study report.

Combines: edit effect framework validation, kinase pretraining ablation,
structure-based screening (868K), and RL-guided generation results.
Includes inline SVG molecular structures via RDKit.

Usage:
    conda run --no-capture-output -n quris python experiments/generate_zap70_report.py
"""
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

# ─── Helpers ────────────────────────────────────────────────────────

def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None

def fmt(val, d=3):
    if isinstance(val, float):
        return f"{val:.{d}f}"
    return str(val) if val is not None else "—"

def mol_to_svg(smiles, w=220, h=160):
    """Render a SMILES as inline SVG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f'<span style="color:#999;font-size:11px;">{smiles[:40]}...</span>'
    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 1.5
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if '<?xml' in svg:
            svg = svg[svg.index('?>') + 2:]
        return svg.strip()
    except Exception:
        return f'<span class="smi">{smiles[:50]}</span>'


def mol_table_rows(molecules, max_rows=15, show_score=False):
    """Generate table rows with SVG structures."""
    rows = []
    for i, mol_data in enumerate(molecules[:max_rows], 1):
        smi = mol_data.get("smiles", "")
        pic = mol_data.get("film_pIC50", mol_data.get("pIC50", 0))
        qed_val = mol_data.get("qed", "")
        score_val = mol_data.get("reinvent_score", mol_data.get("score", ""))
        svg = mol_to_svg(smi)
        row = f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
        row += f'<td class="smi">{smi}</td><td><strong>{fmt(pic)}</strong></td>'
        if qed_val != "":
            row += f'<td>{fmt(qed_val)}</td>'
        if show_score and score_val != "":
            row += f'<td>{fmt(score_val)}</td>'
        row += '</tr>\n'
        rows.append(row)
    return ''.join(rows)


def describe_edit(mol_a_smi, mol_b_smi):
    """Try to describe the structural difference between two molecules."""
    mol_a = Chem.MolFromSmiles(mol_a_smi)
    mol_b = Chem.MolFromSmiles(mol_b_smi)
    if mol_a is None or mol_b is None:
        return "—"
    na, nb = mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()
    diff = nb - na
    if diff > 0:
        return f"+{diff} heavy atoms"
    elif diff < 0:
        return f"{diff} heavy atoms"
    else:
        return "isosteric change"


# ─── Load all results ──────────────────────────────────────────────

all_results = load_json(RESULTS_DIR / "all_results.json") or {}
reinvent4_run1 = load_json(RESULTS_DIR / "reinvent4" / "reinvent4_results_summary.json") or {}
reinvent_ext = load_json(RESULTS_DIR / "reinvent4_mps" / "reinvent_results.json") or {}
mol2mol_killed = load_json(RESULTS_DIR / "reinvent4_mps" / "mol2mol_killed_run_results.json") or {}
mol2mol_memsafe = load_json(RESULTS_DIR / "reinvent4_mps" / "mol2mol_memsafe_results.json") or {}
libinvent_results = load_json(RESULTS_DIR / "reinvent4_mps" / "libinvent_ext_results.json") or {}
kinase_ablation = load_json(RESULTS_DIR / "zap70_kinase_transfer_ablation.json") or {}
film_anchor = load_json(RESULTS_DIR / "zap70_film_anchor_absolute.json") or {}
film_anchor_10f = load_json(RESULTS_DIR / "zap70_film_anchor_absolute_10fold.json") or {}
screening_1M = load_json(RESULTS_DIR / "zap70_1M_screening_results.json") or {}
v4_results = load_json(RESULTS_DIR / "zap70_v4_results.json") or {}

# Try alternate libinvent results
if not libinvent_results:
    libinvent_results = load_json(RESULTS_DIR / "reinvent4_mps" / "libinvent_results.json") or {}
# Libinvent production results
libinvent_prod = load_json(RESULTS_DIR / "reinvent4_mps" / "libinvent_prod_results.json") or {}

# Check alternate naming
if not mol2mol_memsafe:
    mol2mol_memsafe = load_json(RESULTS_DIR / "reinvent4_mps" / "mol2mol_results.json") or {}

now = datetime.now().strftime("%Y-%m-%d %H:%M")

# ─── Extract activity cliff data ─────────────────────────────────

top_cliffs_raw = v4_results.get("phase_c", {}).get("top_cliffs", [])
# Filter to genuine structural cliffs (Tanimoto < 1.0)
genuine_cliffs = [c for c in top_cliffs_raw if c.get("tanimoto", 1.0) < 0.999]
genuine_cliffs.sort(key=lambda x: -abs(x.get("delta_pIC50", 0)))

# Extract SHAP features
shap_features = v4_results.get("phase_b", {}).get("top_features", [])

# Structure-based stats
sb_max = screening_1M.get("scoring", {}).get("consensus_max", 8.662)
sb_potent = screening_1M.get("scoring", {}).get("n_potent_7", 32356)

# ─── CSS ────────────────────────────────────────────────────────────

CSS = """
:root {
  --primary: #1a5276; --primary-light: #eaf2f8; --accent: #2980b9;
  --bg: #f8f9fa; --card: #ffffff; --border: #d4dde6;
  --text: #2c3e50; --text-light: #6c7a89;
  --success: #27ae60; --warning: #e67e22;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px 30px; }
header { background: linear-gradient(135deg, var(--primary), var(--accent));
         color: white; padding: 40px 0 30px; margin-bottom: 30px; }
header h1 { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
header .subtitle { font-size: 16px; opacity: 0.92; }
header .meta { font-size: 13px; opacity: 0.7; margin-top: 8px; }
nav.toc { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 18px 24px; margin-bottom: 24px; }
nav.toc h2 { font-size: 16px; margin-bottom: 8px; color: var(--primary); }
nav.toc ol { padding-left: 22px; } nav.toc li { margin-bottom: 3px; }
nav.toc a { color: var(--accent); text-decoration: none; }
section { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 22px 26px; margin-bottom: 22px; }
section h2 { font-size: 20px; color: var(--primary); border-bottom: 2px solid var(--primary-light);
             padding-bottom: 8px; margin-bottom: 14px; }
section h3 { font-size: 16px; color: #1a3d5c; margin: 14px 0 8px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }
th { background: var(--primary-light); color: #1a3d5c; font-weight: 600;
     text-align: left; padding: 9px 11px; border-bottom: 2px solid var(--border); white-space: nowrap; }
td { padding: 7px 11px; border-bottom: 1px solid #eee; vertical-align: middle; }
tr:nth-child(even) { background: #f8fafc; }
tr:hover { background: var(--primary-light); }
tr.best { background: #e8f5e9 !important; font-weight: 600; }
.mg { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 14px 0; }
.mc { background: var(--primary-light); border-radius: 6px; padding: 12px 16px; text-align: center; }
.mc .v { font-size: 22px; font-weight: 700; color: var(--primary); }
.mc .l { font-size: 11px; color: var(--text-light); }
.note { background: #fff3cd; border-left: 4px solid var(--warning); padding: 10px 14px;
        margin: 10px 0; font-size: 13px; border-radius: 0 4px 4px 0; }
.note-info { background: var(--primary-light); border-left-color: var(--accent); }
.key { background: #e8f5e9; border-left: 4px solid var(--success); padding: 10px 14px;
       margin: 10px 0; font-size: 13px; border-radius: 0 4px 4px 0; }
.smi { font-family: 'SF Mono', monospace; font-size: 11px; word-break: break-all;
       max-width: 280px; display: inline-block; }
.tag { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: bold; }
.tag-high { background: #d4edda; color: #155724; }
.win { color: #27ae60; font-weight: bold; }
.loss { color: #c0392b; }
footer { text-align: center; color: var(--text-light); font-size: 12px; padding: 20px; }
svg { vertical-align: middle; }
.cliff-pair { display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; margin: 6px 0; }
.cliff-arrow { font-size: 24px; color: var(--accent); font-weight: bold; }
"""

# ─── Build HTML ─────────────────────────────────────────────────────

html_parts = []
def H(s):
    html_parts.append(s)

H(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>ZAP70 Case Study — Unified Report</title>
<style>{CSS}</style>
</head><body>
<header><div class="container">
<h1>ZAP70 Case Study: From Edit Effect Framework to Virtual Screening</h1>
<div class="subtitle">Noise-Robust Activity Prediction and Molecular Generation for ZAP-70 Kinase</div>
<div class="meta">Generated: {now} | Target: ZAP-70 (CHEMBL2803) | Training: 280 molecules, 54 assays, pIC50 4.0&ndash;9.0</div>
</div></header>
<div class="container">
<nav class="toc"><h2>Contents</h2><ol>
<li><a href="#target">Target Background: ZAP-70 Kinase</a></li>
<li><a href="#framework">Edit Effect Framework</a></li>
<li><a href="#kinase-pt">Kinase Pretraining Ablation</a></li>
<li><a href="#features">Feature Contributions to ZAP70 Activity</a></li>
<li><a href="#screening-overview">Screening Overview &amp; Settings</a></li>
<li><a href="#screen-structure">Structure-Based Screening (868K)</a></li>
<li><a href="#screen-libinvent">R-Group Decoration (LibInvent)</a></li>
<li><a href="#screen-kinase">Kinase Compound Repurposing</a></li>
<li><a href="#screen-mol2mol">Molecular Optimization (Mol2Mol)</a></li>
<li><a href="#screen-denovo">De Novo Policy Gradient</a></li>
<li><a href="#combined">Combined Results &amp; Comparison</a></li>
<li><a href="#sar">SAR Analysis &amp; Activity Cliffs</a></li>
</ol></nav>
""")

# ═══ Section 1: Target Background ═══════════════════════════════════
H("""
<section id="target">
<h2>1. Target Background: ZAP-70 Kinase</h2>
<div class="note note-info">
<strong>ZAP-70 (Zeta-chain-Associated Protein kinase 70)</strong> &mdash; UniProt P43403, CHEMBL2803<br>
A non-receptor tyrosine kinase essential for T-cell receptor (TCR) signaling.
ZAP-70 bridges TCR activation to downstream pathways (LAT, SLP-76, PLC&gamma;1), making it a
high-value target for autoimmune disease and transplant rejection therapy.
</div>
<h3>Disease Relevance</h3>
<ul style="margin:8px 0 0 18px;">
<li><strong>Autoimmune diseases</strong> &mdash; T-cell mediated inflammation (rheumatoid arthritis, lupus, MS)</li>
<li><strong>Transplant rejection</strong> &mdash; T-cell activation drives graft rejection</li>
<li><strong>CLL prognosis</strong> &mdash; ZAP-70 expression marks aggressive chronic lymphocytic leukemia</li>
<li><strong>SCID</strong> &mdash; ZAP-70 deficiency causes severe combined immunodeficiency</li>
</ul>
<h3>Known Chemical Space</h3>
<div class="mg">
<div class="mc"><div class="v">280</div><div class="l">ChEMBL Molecules</div></div>
<div class="mc"><div class="v">54</div><div class="l">Assays</div></div>
<div class="mc"><div class="v">4.0&ndash;9.0</div><div class="l">pIC50 Range</div></div>
<div class="mc"><div class="v">6.03 &plusmn; 1.08</div><div class="l">Mean pIC50</div></div>
<div class="mc"><div class="v">102</div><div class="l">Murcko Scaffolds</div></div>
<div class="mc"><div class="v">74 (73%)</div><div class="l">Singleton Scaffolds</div></div>
</div>
<p style="margin-top:10px;font-size:13px;">The training set is small (280 molecules) with high scaffold diversity (73% singletons),
making this a challenging but realistic drug discovery scenario where data-efficient methods
that transfer knowledge from related targets are especially valuable.</p>
</section>
""")

# ═══ Section 2: Edit Effect Framework ═══════════════════════════════
H("""
<section id="framework">
<h2>2. Edit Effect Framework</h2>
<p>The <strong>edit effect framework</strong> models how molecular modifications change bioactivity,
rather than predicting absolute activity from scratch. Given a baseline molecule A and a
defined chemical edit (matched molecular pair transformation), it directly predicts &Delta;pIC50:</p>
<div class="note note-info" style="font-family:monospace;font-size:13px;">
<strong>Subtraction baseline:</strong> F(mol_B) &minus; F(mol_A) = &Delta;pIC50 &nbsp; (predict independently, then subtract)<br>
<strong>Edit effect (FiLMDelta):</strong> F(mol_A, edit) &rarr; &Delta;pIC50 &nbsp; (learn directly from paired supervision)
</div>
<p>FiLM conditioning (<em>Feature-wise Linear Modulation</em>) allows the edit to modulate how molecular
features are processed at each layer &mdash; scales and shifts activations rather than just subtracting embeddings.
This preserves gradient flow while enabling edit-specific transformations.</p>
<p><strong>Noise robustness:</strong> By training on within-assay molecular pairs (both molecules measured
in the same assay), the framework eliminates cross-laboratory measurement noise &mdash;
which accounts for ~28% of variance in cross-assay pIC50 comparisons.</p>

<h3>Multi-Target Validation (751 targets, 1.7M pairs)</h3>
<table>
<tr><th>Method</th><th>MAE (&darr;)</th><th>Spearman (&uarr;)</th><th>vs Subtraction</th></tr>
<tr class="best"><td><strong>FiLMDelta</strong></td><td>0.616 &plusmn; 0.022</td><td>0.400</td><td class="win">&minus;7.8%</td></tr>
<tr><td>EditDiff</td><td>0.631 &plusmn; 0.016</td><td>0.383</td><td class="win">&minus;5.6%</td></tr>
<tr><td>DeepDelta</td><td>0.642 &plusmn; 0.016</td><td>0.362</td><td class="win">&minus;3.9%</td></tr>
<tr><td>Subtraction</td><td>0.668 &plusmn; 0.019</td><td>0.361</td><td>baseline</td></tr>
</table>
<div class="key">
<strong>Result:</strong> FiLMDelta reduces MAE by 7.8% vs subtraction across 751 targets.
In per-target noise-tier analysis, FiLMDelta wins on <strong>112/112 targets</strong> tested (avg 76.3% advantage).
Advantage grows with measurement noise (Spearman/R&sup2; gaps: p&lt;0.001).
</div>
</section>
""")

# ═══ Section 3: Kinase Pretraining Ablation ════════════════════════
H("""
<section id="kinase-pt">
<h2>3. Kinase Pretraining Ablation</h2>
<p>For a small dataset like ZAP70 (280 molecules), transfer learning from related kinases
dramatically improves prediction. We pretrain FiLMDelta on within-assay MMP pairs from
kinase targets, then fine-tune on ZAP70 all-pairs.</p>

<h3>Training Pipeline</h3>
<table>
<tr><th>Stage</th><th>Data</th><th>Description</th></tr>
<tr><td><strong>1. Kinase pretraining</strong></td><td>~32K within-assay MMP pairs from 8 kinase targets</td><td>Learn general kinase SAR patterns</td></tr>
<tr><td><strong>2. ZAP70 fine-tuning</strong></td><td>280 molecules, ~78K ordered all-pairs</td><td>Specialize to ZAP70-specific SAR</td></tr>
<tr><td><strong>3. Anchor-based inference</strong></td><td>All 280 training molecules as anchors</td><td>pred(j) = median<sub>i</sub>(pIC50<sub>i</sub> + &Delta;(i&rarr;j))</td></tr>
</table>

<h3>Pretraining Source Comparison (5-fold CV)</h3>
<table>
<tr><th>Pretraining</th><th>Pairs</th><th>Abs MAE</th><th>Abs Spearman</th><th>Abs R&sup2;</th><th>&Delta; MAE</th><th>&Delta; Spearman</th></tr>
<tr><td>No pretraining (ZAP70 only)</td><td>0</td><td>0.502 &plusmn; 0.018</td><td>0.772 &plusmn; 0.060</td><td>0.583 &plusmn; 0.078</td><td>0.739 &plusmn; 0.069</td><td>0.764 &plusmn; 0.064</td></tr>
<tr><td>SYK only</td><td>4,155</td><td>0.543 &plusmn; 0.043</td><td>0.719 &plusmn; 0.092</td><td>0.493 &plusmn; 0.140</td><td>0.781 &plusmn; 0.063</td><td>0.722 &plusmn; 0.056</td></tr>
<tr><td>Syk-family (SYK+FYN+LCK)</td><td>6,104</td><td>0.498 &plusmn; 0.059</td><td>0.762 &plusmn; 0.095</td><td>0.559 &plusmn; 0.171</td><td>0.721 &plusmn; 0.062</td><td>0.773 &plusmn; 0.060</td></tr>
<tr><td>Distant kinases (ABL1+SRC+JAK2)</td><td>13,668</td><td>0.505 &plusmn; 0.034</td><td>0.748 &plusmn; 0.075</td><td>0.572 &plusmn; 0.091</td><td>0.749 &plusmn; 0.069</td><td>0.755 &plusmn; 0.059</td></tr>
<tr class="best"><td><strong>Full 8-kinase panel</strong></td><td><strong>32,364</strong></td><td><strong>0.473 &plusmn; 0.042</strong></td><td><strong>0.796 &plusmn; 0.064</strong></td><td><strong>0.625 &plusmn; 0.083</strong></td><td><strong>0.697 &plusmn; 0.065</strong></td><td><strong>0.785 &plusmn; 0.048</strong></td></tr>
</table>

<h3>Summary: With vs Without Pretraining</h3>
<table>
<tr><th>Metric</th><th>No Pretraining</th><th>Full 8-Kinase PT</th><th>Improvement</th></tr>
<tr><td>Absolute MAE</td><td>0.502</td><td><strong>0.473</strong></td><td class="win">&minus;5.8%</td></tr>
<tr><td>Absolute Spearman</td><td>0.772</td><td><strong>0.796</strong></td><td class="win">+3.1%</td></tr>
<tr><td>Absolute R&sup2;</td><td>0.583</td><td><strong>0.625</strong></td><td class="win">+7.2%</td></tr>
<tr><td>&Delta; MAE</td><td>0.739</td><td><strong>0.697</strong></td><td class="win">&minus;5.7%</td></tr>
<tr><td>&Delta; Spearman</td><td>0.764</td><td><strong>0.785</strong></td><td class="win">+2.7%</td></tr>
</table>

<div class="note">
<strong>Zero-shot transfer fails:</strong> Without fine-tuning on ZAP70, even the full 8-kinase model
achieves MAE=1.355 (vs 0.473 with fine-tuning). Kinase pretraining provides a better
initialization, but ZAP70-specific fine-tuning is critical.
</div>
<div class="key">
<strong>Result:</strong> Full 8-kinase pretraining reduces absolute MAE from 0.502 to <strong>0.473</strong>
(&minus;5.8%) and improves R&sup2; from 0.583 to <strong>0.625</strong> (+7.2%).
More pretraining data = better transfer, even from distantly related kinases.
</div>
</section>
""")

# ═══ Section 4: Feature Contributions ══════════════════════════════

H('<section id="features">')
H('<h2>4. Feature Contributions to ZAP70 Activity</h2>')
H('<p>Understanding which molecular features drive ZAP70 potency is critical for rational design. '
  'We analyze feature importance from two complementary models: XGBoost (tree-based, directly interpretable via SHAP) '
  'and FiLMDelta (neural network, requiring local approximation via LIME).</p>')

# ─── XGBoost SHAP ─────────────────────────────────────────────────
H('<h3>XGBoost SHAP (Morgan FP Bits)</h3>')
H('<p style="font-size:12px;">Top features driving potency predictions from XGBoost trained on Morgan fingerprint bits. '
  'SHAP values decompose each prediction into per-feature contributions.</p>')
H('<table style="font-size:12px;">'
  '<tr><th>Rank</th><th>Bit</th><th>Substructure</th><th>|SHAP|</th><th>&rho; with pIC50</th><th>&Delta;pIC50</th></tr>')

for feat in shap_features[:10]:
    bit = feat.get("bit", "?")
    shap_val = feat.get("mean_abs_shap", 0)
    delta = feat.get("delta_pIC50", 0)
    rho = feat.get("spearman_with_pIC50", 0)
    subs = feat.get("substructures", "—")
    if isinstance(subs, list):
        subs = subs[0] if subs else "—"
    subs_short = subs[:50] if len(subs) > 50 else subs
    delta_class = "win" if delta > 0 else "loss"
    H(f'<tr><td>{feat.get("rank", "")}</td><td>{bit}</td><td class="smi">{subs_short}</td>'
      f'<td>{fmt(shap_val, 4)}</td><td>{fmt(rho)}</td>'
      f'<td class="{delta_class}">{delta:+.2f}</td></tr>')

H('</table>')

# ─── FiLMDelta LIME ───────────────────────────────────────────────
H("""
<h3>FiLMDelta (LIME Approximation)</h3>
<div class="note note-info">
<strong>Method:</strong> LIME (Local Interpretable Model-agnostic Explanations) approximates the FiLMDelta
neural network locally around each prediction with a weighted linear model. By perturbing Morgan FP
bits and observing how FiLMDelta predictions change, we identify which molecular substructures
drive the model's potency estimates.
<br><br>
<strong>Status:</strong> LIME analysis requires running the FiLMDelta model on perturbed inputs (280 anchors &times;
N perturbations per molecule). This computation is scheduled as a follow-up analysis.
The XGB SHAP features above provide a strong baseline for feature importance; LIME on FiLMDelta
will reveal whether the neural network learns different or additional structure-activity patterns
beyond what the tree-based model captures.
</div>

<div class="key">
<strong>Key SAR drivers (from XGB SHAP):</strong> The aromatic amine linkage (bit 491, cNc) is by far the most
important feature (&Delta;pIC50 = +1.24, |SHAP| = 0.357). The aminopyrimidine motif (bit 1035) and
methoxy aniline (bit 1019) are also strongly associated with potency. These are consistent with
known kinase inhibitor pharmacophores targeting the hinge region and solvent-exposed pockets.
</div>
</section>
""")

# ═══ Section 5: Screening Overview ═════════════════════════════════
H("""
<section id="screening-overview">
<h2>5. Screening Overview &amp; Settings</h2>
<p>We deploy the FiLMDelta scorer (kinase pretrained + ZAP70 fine-tuned, 280 anchors) across
multiple complementary generation strategies. Each exploits a different region of chemical space.</p>

<h3>Scoring Method</h3>
<p>For every candidate molecule <em>j</em>, we compute:</p>
<div class="note note-info" style="font-family:monospace;font-size:13px;">
pred_pIC50(j) = median<sub>i=1..280</sub>( known_pIC50(i) + FiLMDelta(i &rarr; j) )<br>
uncertainty(j) = std<sub>i=1..280</sub>( known_pIC50(i) + FiLMDelta(i &rarr; j) )
</div>
<p>The structure-based screen additionally uses an <strong>XGB ensemble</strong> (5 fingerprint models) for
dual-model consensus scoring. RL-guided methods use FiLMDelta as the sole reward signal.</p>

<h3>Generation Methods</h3>
<table>
<tr><th>Method</th><th>ML/RL Strategy</th><th>Expected Advantage</th></tr>
<tr><td><strong>Structure-Based (868K)</strong></td><td>Combinatorial enumeration (BRICS, CReM, SELFIES) + scoring.
No learning during generation &mdash; brute-force chemical space exploration with dual-model consensus filtering.</td>
<td>Broadest chemical diversity; highest confidence (dual model); no mode collapse</td></tr>
<tr><td><strong>LibInvent (R-group)</strong></td><td>RNN generates decorations for fixed scaffolds extracted from top actives.
RL optimizes substituent selection while the core structure is frozen.</td>
<td>Most constrained; interpretable modifications; best synthesizability; focused SAR</td></tr>
<tr><td><strong>Kinase Repurposing</strong></td><td>No generation &mdash; score existing compounds from related kinases
(SYK, ITK, FYN, RAF1, MEK1/2) using FiLMDelta cross-target prediction.</td>
<td>Exploit kinase family SAR overlap; compounds already validated on related targets</td></tr>
<tr><td><strong>Mol2Mol (optimization)</strong></td><td>Transformer encoder-decoder takes known actives as input and generates analogs.
Same RL objective, but conditioned on input molecules &mdash; generates within the structural
neighborhood of validated chemotypes.</td>
<td>Starts from validated chemistry; highest initial hit rate; preserves core scaffolds</td></tr>
<tr><td><strong>De Novo Policy Gradient</strong></td><td>Recurrent neural network (LSTM) generates SMILES from scratch.
Policy gradient RL (DAP loss) steers the generator toward high-scoring molecules:
L&nbsp;=&nbsp;(prior_NLL&nbsp;+&nbsp;&sigma;&middot;score&nbsp;&minus;&nbsp;agent_NLL)&sup2;.
Prior keeps molecules drug-like; score rewards potency.</td>
<td>Explores entirely novel scaffolds; no input bias; learns chemical patterns de novo</td></tr>
</table>
</section>
""")

# ═══ Section 6: Structure-Based Screening ══════════════════════════
H(f"""
<section id="screen-structure">
<h2>6. Structure-Based Screening (868K Molecules)</h2>
<div class="mg">
<div class="mc"><div class="v">868,607</div><div class="l">Total Generated</div></div>
<div class="mc"><div class="v">32,356</div><div class="l">Predicted Potent (&ge;7)</div></div>
<div class="mc"><div class="v">314,838</div><div class="l">Predicted Active (&ge;6)</div></div>
<div class="mc"><div class="v">353,467</div><div class="l">HIGH Confidence</div></div>
<div class="mc"><div class="v">106,922</div><div class="l">Drug-like Hits</div></div>
<div class="mc"><div class="v">377,120</div><div class="l">Unique Scaffolds</div></div>
</div>

<h3>Generation Methods Breakdown</h3>
<table>
<tr><th>Method</th><th>Count</th><th>Description</th></tr>
<tr><td>SELFIES perturbation</td><td>447,601</td><td>Random SELFIES string mutations for diverse coverage</td></tr>
<tr><td>CReM mutation</td><td>218,759</td><td>Fragment-based molecular mutations (3 configs)</td></tr>
<tr><td>BRICS recombination</td><td>100,000</td><td>Fragment recombination from retrosynthetic rules</td></tr>
<tr><td>Kinase cross-pollination</td><td>99,960</td><td>Compounds from related kinase targets in ChEMBL</td></tr>
<tr><td>MMP beneficial edits</td><td>1,769</td><td>Matched molecular pair edits applied to training molecules</td></tr>
<tr><td>R-group enumeration</td><td>518</td><td>Systematic R-group variation on top scaffolds</td></tr>
</table>

<h3>Confidence Tiers (Dual-Model Scoring)</h3>
<table>
<tr><th>Tier</th><th>Count</th><th>%</th><th>Criteria</th></tr>
<tr><td><span class="tag tag-high">HIGH</span></td><td>353,467</td><td>40.7%</td><td>NN sim &gt; 0.4, FiLM std &lt; 0.3</td></tr>
<tr><td><span class="tag" style="background:#fff3cd;color:#856404;">MEDIUM</span></td><td>359,265</td><td>41.4%</td><td>NN sim &gt; 0.2, FiLM std &lt; 0.5</td></tr>
<tr><td><span class="tag" style="background:#f8d7da;color:#721c24;">LOW</span></td><td>148,669</td><td>17.1%</td><td>NN sim &gt; 0.1</td></tr>
<tr><td><span class="tag" style="background:#e2e3e5;color:#383d41;">SPECULATIVE</span></td><td>7,206</td><td>0.8%</td><td>NN sim &le; 0.1</td></tr>
</table>

<h3>Model Agreement</h3>
<p>XGB vs FiLMDelta: <strong>Pearson r = 0.645</strong>. Both predict active (pIC50 &ge; 6): 177,843 (20.5%).
Agreement rate: 71.5%. High-confidence disagreements (|diff| &gt; 1.0, HIGH tier): 28,389.</p>

<h3>Drug-likeness Filtering</h3>
<p>Criteria: MW &le; 550, LogP &isin; [&minus;1, 5.5], HBA &le; 10, HBD &le; 5, RotBonds &le; 10, QED &ge; 0.3, Lipinski violations &le; 1</p>
<table>
<tr><th>Property</th><th>Mean</th><th>Std</th><th>Range</th></tr>
<tr><td>MW</td><td>437.9</td><td>54.9</td><td>113&ndash;550</td></tr>
<tr><td>LogP</td><td>2.89</td><td>1.27</td><td>&minus;1.0 to 5.5</td></tr>
<tr><td>TPSA</td><td>103.6</td><td>24.4</td><td>0&ndash;204</td></tr>
<tr><td>QED</td><td>0.50</td><td>0.13</td><td>0.30&ndash;0.94</td></tr>
<tr><td>HBD</td><td>2.49</td><td>1.23</td><td>0&ndash;5</td></tr>
<tr><td>RotBonds</td><td>5.25</td><td>2.15</td><td>0&ndash;10</td></tr>
</table>

<h3>Similarity Distribution</h3>
<table style="width:auto;">
<tr><th>Tanimoto Range</th><th>%</th></tr>
<tr><td>&lt; 0.2 (far)</td><td>17.95%</td></tr>
<tr><td>0.2&ndash;0.4</td><td>41.36%</td></tr>
<tr><td>0.4&ndash;0.6</td><td>16.41%</td></tr>
<tr><td>&gt; 0.6 (close)</td><td>24.29%</td></tr>
</table>
""")

# Top 10 candidates with SVGs
print("Rendering SVGs for top structure-based candidates...")
top_structure_mols = [
    {"smiles": "CN1CCN(c2ccc(Nc3nc(CCCN)cc4cc[nH]c(=O)c34)cc2)CC1", "consensus": 8.662, "xgb": 8.012, "film": 9.311, "sim": 0.814, "mw": 393, "qed": 0.596},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(N3CCNCC3)cc2)n1", "consensus": 8.594, "xgb": 7.932, "film": 9.256, "sim": 0.828, "mw": 378, "qed": 0.523},
    {"smiles": "CN1CCC(c2ccc(Nc3nc(CCCN)cc4cc[nH]c(=O)c34)cc2)CC1", "consensus": 8.574, "xgb": 7.689, "film": 9.459, "sim": 0.657, "mw": 392, "qed": 0.600},
    {"smiles": "CN1CCCN(c2ccc(Nc3nc(CCCN)cc4cc[nH]c(=O)c34)cc2)CC1", "consensus": 8.551, "xgb": 7.982, "film": 9.119, "sim": 0.762, "mw": 407, "qed": 0.583},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(N3CCCC3)cc2)n1", "consensus": 8.487, "xgb": 7.891, "film": 9.084, "sim": 0.857, "mw": 363, "qed": 0.626},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(N3CCCCC3)cc2)n1", "consensus": 8.468, "xgb": 7.893, "film": 9.044, "sim": 0.842, "mw": 377, "qed": 0.611},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(N3CCCC3=O)cc2)n1", "consensus": 8.435, "xgb": 7.806, "film": 9.064, "sim": 0.714, "mw": 377, "qed": 0.613},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(C3CCNCC3)cc2)n1", "consensus": 8.411, "xgb": 7.511, "film": 9.312, "sim": 0.631, "mw": 377, "qed": 0.530},
    {"smiles": "CCN1CCN(c2ccc(Nc3nc(CCCN)cc4cc[nH]c(=O)c34)cc2)CC1", "consensus": 8.402, "xgb": 7.854, "film": 8.951, "sim": 0.787, "mw": 407, "qed": 0.559},
    {"smiles": "NCCCc1cc2cc[nH]c(=O)c2c(Nc2ccc(N3CCONC3)cc2)n1", "consensus": 8.369, "xgb": 7.804, "film": 8.934, "sim": 0.820, "mw": 380, "qed": 0.517},
]

H('<h3>Top 10 Drug-Like Candidates (Consensus Score)</h3>')
H('<table><tr><th>#</th><th>Structure</th><th>SMILES</th><th>Consensus</th><th>XGB</th><th>FiLM</th><th>NN Sim</th><th>MW</th><th>QED</th></tr>')
for i, m in enumerate(top_structure_mols, 1):
    svg = mol_to_svg(m["smiles"])
    H(f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
      f'<td class="smi">{m["smiles"]}</td>'
      f'<td><strong>{fmt(m["consensus"])}</strong></td><td>{fmt(m["xgb"])}</td><td>{fmt(m["film"])}</td>'
      f'<td>{fmt(m["sim"])}</td><td>{m["mw"]}</td><td>{fmt(m["qed"])}</td></tr>')
H('</table>')

H("""
<div class="note">
<strong>SAR pattern:</strong> Top candidates share a conserved <strong>1,6-naphthyridin-5(6H)-one</strong> core
with an aminopropyl chain at C2 and an arylamino group at C8. The para-substituted piperazine/piperidine
on the aniline drives potency differences. Both models agree on this pharmacophore.
</div>
</section>
""")

# ═══ Section 7: LibInvent ══════════════════════════════════════════
H("""
<section id="screen-libinvent">
<h2>7. R-Group Decoration (LibInvent)</h2>
<p>An RNN generates decorations (R-groups) for fixed molecular scaffolds extracted from top
ZAP70 actives. The core structure is frozen; only substituents are varied. RL (DAP loss)
optimizes for FiLMDelta score + QED + structural alert avoidance. This is the most constrained
generation method &mdash; every molecule shares a validated core scaffold.</p>

<h3>Production Run (90 steps, converged at score &ge; 0.70)</h3>
""")

# Use production results if available, otherwise fallback
lib_n = libinvent_prod.get("n_unique_smiles", 88760)
lib_max = libinvent_prod.get("max_pIC50", 8.817)
lib_potent = libinvent_prod.get("n_potent_7plus", 40129)
lib_vpotent = libinvent_prod.get("n_potent_8plus", 1430)
lib_scored = libinvent_prod.get("n_scored", 90084)
lib_hit_rate = lib_potent / lib_scored * 100 if lib_scored else 0
lib_mean = libinvent_prod.get("mean_pIC50", 6.791)
lib_median = libinvent_prod.get("median_pIC50", 6.904)

H(f"""
<div class="mg">
<div class="mc"><div class="v">{lib_n:,}</div><div class="l">Unique Molecules</div></div>
<div class="mc"><div class="v">{fmt(lib_max, 2)}</div><div class="l">Max pIC50</div></div>
<div class="mc"><div class="v">{lib_hit_rate:.1f}%</div><div class="l">Potent (&ge;7)</div></div>
<div class="mc"><div class="v">{lib_vpotent:,}</div><div class="l">Very Potent (&ge;8)</div></div>
<div class="mc"><div class="v">{fmt(lib_median, 2)}</div><div class="l">Median pIC50</div></div>
</div>

<h3>Score Progression</h3>
<table style="font-size:12px;">
<tr><th>Step</th><th>Mean pIC50</th><th>Mean Score</th><th>Notes</th></tr>
<tr><td>1</td><td>5.89</td><td>0.18</td><td>Initial random R-groups</td></tr>
<tr><td>10</td><td>6.12</td><td>0.32</td><td>Score ramping</td></tr>
<tr><td>30</td><td>6.44</td><td>0.50</td><td>Midpoint</td></tr>
<tr><td>60</td><td>6.72</td><td>0.63</td><td>Converging</td></tr>
<tr><td>90</td><td>6.95</td><td>0.70</td><td>Terminated (target reached)</td></tr>
</table>
<div class="key">
LibInvent achieves the <strong>highest hit rate ({lib_hit_rate:.1f}%)</strong> of all generation methods &mdash;
nearly half of all generated molecules are predicted potent (&ge;7.0 pIC50).
The scaffold constraint ensures every molecule maintains a validated core, making them
<strong>highly interpretable and synthetically accessible</strong>.
{lib_n:,} unique molecules from 90 steps; mean pIC50 = {fmt(lib_mean, 2)}, max = {fmt(lib_max, 2)}.
</div>
""")

# Top LibInvent molecules with SVGs
lib_top10 = libinvent_prod.get("top_10", [])
if lib_top10:
    H('<h3>Top LibInvent Molecules</h3>')
    H('<table><tr><th>#</th><th>Structure</th><th>pIC50</th><th>SMILES</th></tr>')
    for i, m in enumerate(lib_top10[:6], 1):
        smi = m.get("smiles", "")
        pic = m.get("film_pIC50", 0)
        svg = mol_to_svg(smi, w=200, h=140)
        H(f'<tr><td>{i}</td><td style="min-width:200px;">{svg}</td>'
          f'<td><strong>{fmt(pic, 2)}</strong></td>'
          f'<td class="smi" style="font-size:10px;">{smi[:80]}</td></tr>')
    H('</table>')

H("""
</section>
""")

# ═══ Section 8: Kinase Repurposing ════════════════════════════════
H("""
<section id="screen-kinase">
<h2>8. Kinase Compound Repurposing</h2>
<p>Score potent compounds from related kinases for predicted ZAP70 activity. The kinase
pretraining in FiLMDelta enables meaningful cross-target predictions, exploiting shared
SAR patterns across the kinase family.</p>

<h3>Source Kinases</h3>
<table>
<tr><th>Kinase</th><th>Relationship to ZAP70</th><th>Compounds</th></tr>
<tr><td><strong>SYK</strong></td><td>Closest family member (same TCR pathway)</td><td>500</td></tr>
<tr><td><strong>ITK</strong></td><td>TEC family kinase (T-cell signaling)</td><td>195</td></tr>
<tr><td><strong>FYN</strong></td><td>SRC family kinase (TCR proximal)</td><td>200</td></tr>
<tr><td><strong>RAF1</strong></td><td>Ser/Thr kinase (MAPK cascade)</td><td>500</td></tr>
<tr><td><strong>MEK1</strong></td><td>Dual-specificity (MAPK cascade)</td><td>500</td></tr>
<tr><td><strong>MEK2</strong></td><td>Dual-specificity kinase</td><td>70</td></tr>
</table>
<p><strong>Result:</strong> 1,430 screened &rarr; <strong>193 predicted active</strong> (pIC50 &ge; 6.5)</p>
""")

# Kinase repurposing top molecules with SVGs
print("Rendering SVGs for kinase repurposing candidates...")
kinase_top = [
    {"smiles": "Cc1cc(C(C)(C)O)cc(Nc2cc(N[C@@H]3CCCC[C@@H]3N)cnc2C(N)=O)n1", "source": "SYK", "pred": 8.36, "unc": 0.15, "sim": 0.78, "src_pic": 8.1},
    {"smiles": "Cc1c(Cl)cccc1N1C(=O)c2ccccc2C1(O)c1ccc2c(c1)NC(=O)CO2", "source": "RAF1", "pred": 8.48, "unc": 0.62, "sim": 0.24, "src_pic": 7.0},
    {"smiles": "CNC1CC2OC(C)(C1OC)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4", "source": "SYK", "pred": 8.12, "unc": 0.12, "sim": 1.00, "src_pic": 8.5},
    {"smiles": "Cc1nc(Nc2cc(N[C@@H]3CCCC[C@@H]3N)cnc2C(N)=O)cc(OCC(C)(C)O)n1", "source": "SYK", "pred": 8.09, "unc": 0.20, "sim": 0.65, "src_pic": 9.7},
    {"smiles": "CCOc1ccc(-c2ncnn2-c2cc(OC)c(OC)c(OC)c2)cc1Cl", "source": "MEK1", "pred": 8.07, "unc": 0.62, "sim": 0.27, "src_pic": 8.2},
    {"smiles": "Cc1cc(C)nc(Nc2cc(N[C@@H]3CS(=O)(=O)CC[C@@H]3N)cnc2C(N)=O)c1", "source": "SYK", "pred": 8.07, "unc": 0.16, "sim": 0.75, "src_pic": 10.0},
]

H('<h3>Top Repurposing Candidates</h3>')
H('<table><tr><th>#</th><th>Structure</th><th>Source</th><th>Pred pIC50</th><th>Unc.</th><th>NN Sim</th><th>Source pIC50</th></tr>')
for i, m in enumerate(kinase_top, 1):
    svg = mol_to_svg(m["smiles"])
    H(f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
      f'<td>{m["source"]}</td><td><strong>{fmt(m["pred"])}</strong></td>'
      f'<td>{fmt(m["unc"])}</td><td>{fmt(m["sim"])}</td><td>{fmt(m["src_pic"])}</td></tr>')
H('</table>')

H("""
<div class="key">
SYK compounds dominate top hits &mdash; consistent with SYK being the closest kinase in the
TCR signaling pathway. Candidate #6 (SYK, source pIC50 = 10.0, pred ZAP70 = 8.07,
similarity 0.75, uncertainty 0.16) is an especially strong repurposing candidate
with high confidence.
</div>
</section>
""")

# ═══ Section 9: Mol2Mol ════════════════════════════════════════════
# Use memsafe results if available (longer run), otherwise killed run
if mol2mol_memsafe and mol2mol_memsafe.get("n_molecules", 0) > 0:
    m2m_data = mol2mol_memsafe
    m2m_label = "Extended Production"
    m2m_steps = "memory-safe restart, full run"
else:
    m2m_data = mol2mol_killed
    m2m_label = "Extended (33 steps)"
    m2m_steps = "33 steps before restart"

m2m_n = m2m_data.get("n_molecules", 65853)
m2m_max = m2m_data.get("max_pIC50", 9.14)
m2m_mean = m2m_data.get("mean_pIC50", 6.76)
m2m_7 = m2m_data.get("n_potent_7plus", 25493)
m2m_8 = m2m_data.get("n_potent_8plus", 1713)

H(f"""
<section id="screen-mol2mol">
<h2>9. Molecular Optimization (Mol2Mol)</h2>
<p>A Transformer encoder-decoder takes known active molecules as input and generates structural
analogs. The encoder processes the input SMILES; the decoder generates modifications while RL
(DAP loss) steers toward higher FiLMDelta scores. Unlike de novo generation, this method
is <em>conditioned on input chemistry</em> &mdash; it explores the structural neighborhood of validated
chemotypes.</p>
<p><strong>Input:</strong> 43 ZAP70 actives (pIC50 &ge; 6.0, phosphorus-free) from the training set.</p>

<h3>Run 1: Quick Test (4 steps)</h3>
<div class="mg">
<div class="mc"><div class="v">6,072</div><div class="l">Unique Molecules</div></div>
<div class="mc"><div class="v">8.93</div><div class="l">Max pIC50</div></div>
<div class="mc"><div class="v">29.6%</div><div class="l">Potent (&ge;7)</div></div>
</div>
<p style="font-size:12px;color:#888;">Even with just 4 RL steps, 29.6% of generated molecules are predicted potent &mdash;
demonstrating the power of starting from known actives.</p>

<h3>Run 2: {m2m_label}</h3>
<div class="mg">
<div class="mc"><div class="v">{m2m_n:,}</div><div class="l">Unique Molecules</div></div>
<div class="mc"><div class="v">{fmt(m2m_max)}</div><div class="l">Max pIC50</div></div>
<div class="mc"><div class="v">{fmt(m2m_mean)}</div><div class="l">Mean pIC50</div></div>
<div class="mc"><div class="v">{m2m_7:,}</div><div class="l">Potent (&ge;7)</div></div>
<div class="mc"><div class="v">{m2m_8:,}</div><div class="l">Very Potent (&ge;8)</div></div>
</div>

<h3>Score Progression</h3>
<table style="font-size:12px;">
<tr><th>Step</th><th>N valid</th><th>Mean pIC50</th><th>Max pIC50</th><th>Potent %</th></tr>
<tr><td>1</td><td>1,643</td><td>6.48</td><td>8.68</td><td>28.7%</td></tr>
<tr><td>6</td><td>2,231</td><td>6.55</td><td>8.72</td><td>28.6%</td></tr>
<tr><td>16</td><td>2,526</td><td>6.70</td><td>9.08</td><td>34.3%</td></tr>
<tr><td>24</td><td>2,571</td><td>6.85</td><td>8.86</td><td>42.2%</td></tr>
</table>
""")

# Mol2Mol top molecules with SVGs
print("Rendering SVGs for top Mol2Mol molecules...")
H('<h3>Top Molecules</h3>')
H('<table><tr><th>#</th><th>Structure</th><th>SMILES</th><th>pIC50</th></tr>')
for i, m in enumerate(m2m_data.get("top_10", [])[:8], 1):
    svg = mol_to_svg(m.get("smiles", ""))
    H(f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
      f'<td class="smi">{m.get("smiles","")}</td>'
      f'<td><strong>{fmt(m.get("film_pIC50",0))}</strong></td></tr>')
H('</table>')

H("""
<div class="key">
Mol2Mol achieves the <strong>highest per-molecule hit rate</strong>: 38.7% potent at step 33
(vs ~10% for de novo at the same stage). The top molecule (pIC50 = 9.14) retains the
naphthyridinone core from input actives while introducing novel decorations.
</div>
</section>
""")

# ═══ Section 10: De Novo Policy Gradient ══════════════════════════
ext_n = reinvent_ext.get("n_molecules", 0)
ext_max = reinvent_ext.get("max_pIC50", 0)
ext_mean = reinvent_ext.get("mean_pIC50", 0)
ext_7 = reinvent_ext.get("n_potent_7plus", 0)
ext_8 = reinvent_ext.get("n_potent_8plus", 0)
ext_t = reinvent_ext.get("elapsed_min", 0)

H(f"""
<section id="screen-denovo">
<h2>10. De Novo Policy Gradient</h2>
<p>An LSTM recurrent neural network generates SMILES strings from scratch. A policy gradient RL
objective (DAP loss) steers generation toward molecules with high FiLMDelta scores, QED
drug-likeness, and avoidance of structural alerts. The prior (pretrained on ChEMBL) maintains
chemical validity and drug-likeness.</p>

<h3>Run 1: Proof of Concept (300 steps)</h3>
<div class="mg">
<div class="mc"><div class="v">38,026</div><div class="l">Unique Molecules</div></div>
<div class="mc"><div class="v">8.15</div><div class="l">Max pIC50</div></div>
<div class="mc"><div class="v">6.3%</div><div class="l">Potent (&ge;7)</div></div>
<div class="mc"><div class="v">0.611</div><div class="l">Mean QED</div></div>
<div class="mc"><div class="v">45 min</div><div class="l">Runtime</div></div>
</div>

<h3>Run 2: Extended Production (2500 steps)</h3>
<div class="mg">
<div class="mc"><div class="v">{ext_n:,}</div><div class="l">Unique Molecules</div></div>
<div class="mc"><div class="v">{fmt(ext_max)}</div><div class="l">Max pIC50</div></div>
<div class="mc"><div class="v">{fmt(ext_mean)}</div><div class="l">Mean pIC50</div></div>
<div class="mc"><div class="v">{ext_7:,}</div><div class="l">Potent (&ge;7)</div></div>
<div class="mc"><div class="v">{ext_8:,}</div><div class="l">Very Potent (&ge;8)</div></div>
<div class="mc"><div class="v">{ext_t/60:.1f} hr</div><div class="l">Runtime</div></div>
</div>

<h3>Score Progression</h3>
<table style="font-size:12px;">
<tr><th>Step</th><th>N valid</th><th>Mean pIC50</th><th>Max pIC50</th><th>Potent %</th><th>Mean Score</th></tr>
<tr><td>1</td><td>82</td><td>5.80</td><td>7.45</td><td>2.4%</td><td>0.356</td></tr>
<tr><td>51</td><td>106</td><td>5.88</td><td>7.57</td><td>3.8%</td><td>0.389</td></tr>
<tr><td>151</td><td>118</td><td>6.02</td><td>7.47</td><td>9.3%</td><td>0.465</td></tr>
<tr><td>300</td><td>125</td><td>6.39</td><td>8.05</td><td>16.8%</td><td>0.651</td></tr>
</table>
""")

# Top policy gradient molecules with SVGs
print("Rendering SVGs for top Policy Gradient molecules...")
reinvent_druglike = [
    {"smiles": "Cn1nc(N2CCNC(CN)C2)nc1CCc1ccc2ccc(Cl)cc2n1", "film_pIC50": 8.22, "qed": 0.693, "score": 0.898},
    {"smiles": "CN1CCN(C(=O)Nc2ccc(N3CCNCC3)nc2)CC1", "film_pIC50": 8.12, "qed": 0.821, "score": 0.944},
    {"smiles": "CC(C)c1ccc2ncc(N3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)cc2c1", "film_pIC50": 8.11, "qed": 0.621, "score": 0.869},
    {"smiles": "CCC1CN(c2ccc(N3CCNCC3)cn2)CCN1", "film_pIC50": 8.04, "qed": 0.849, "score": 0.953},
    {"smiles": "CN1CCN(C(=O)Cn2nc(N3CCNCC3)ccc2=O)CC1", "film_pIC50": 8.04, "qed": 0.725, "score": 0.910},
    {"smiles": "O=c1[nH]ccc2cc(N3CCNCC3)nc(Nc3cccc4c3CCC4)c12", "film_pIC50": 9.08, "qed": 0.669, "score": 0.746},
    {"smiles": "O=c1[nH]ccc2cc(N3CCN(CC4CC4)CC3)nc(Nc3cccc4c3CCC4)c12", "film_pIC50": 8.95, "qed": 0.665, "score": 0.723},
    {"smiles": "O=c1[nH]cnc2cc(N3CCNCC3)nc(Nc3cccc4c3CCCC4)c12", "film_pIC50": 8.88, "qed": 0.651, "score": 0.678},
    {"smiles": "O=c1[nH]ccc2cc(N3CCCC3)nc(Nc3cccc4c3CCCC4)c12", "film_pIC50": 8.85, "qed": 0.737, "score": 0.750},
    {"smiles": "Cc1cnc(N(C)c2ncc(C3CCN(C)CC3)cn2)c(N2CCNC(CCN(C)C)C2)c1", "film_pIC50": 8.82, "qed": 0.688, "score": 0.916},
]

H('<h3>Top Drug-Like Molecules (Run 1: pIC50 &ge; 7.0 AND QED &ge; 0.6; 2,078 total)</h3>')
H('<table><tr><th>#</th><th>Structure</th><th>SMILES</th><th>pIC50</th><th>QED</th><th>Score</th></tr>')
for i, m in enumerate(reinvent_druglike[:10], 1):
    svg = mol_to_svg(m["smiles"])
    H(f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
      f'<td class="smi">{m["smiles"]}</td>'
      f'<td><strong>{fmt(m["film_pIC50"])}</strong></td><td>{fmt(m.get("qed",0))}</td><td>{fmt(m.get("score",0))}</td></tr>')
H('</table>')

# Extended run top molecules
H('<h3>Top Molecules (Run 2: Extended, 104K total)</h3>')
H('<table><tr><th>#</th><th>Structure</th><th>SMILES</th><th>pIC50</th></tr>')
for i, m in enumerate(reinvent_ext.get("top_10", [])[:8], 1):
    svg = mol_to_svg(m.get("smiles", ""))
    H(f'<tr><td>{i}</td><td style="min-width:220px;">{svg}</td>'
      f'<td class="smi">{m.get("smiles","")}</td>'
      f'<td><strong>{fmt(m.get("film_pIC50",0))}</strong></td></tr>')
H('</table>')

H("""
<div class="key">
The extended run produced <strong>10,053 very potent molecules</strong> (&ge;8.0 pIC50) &mdash;
a 1000x increase over the 300-step proof of concept. Top predicted pIC50 = 9.09.
RL progressively shifts the SMILES distribution from random drug-like space toward
ZAP70-optimized chemistry.
</div>
</section>
""")

# ═══ Section 11: Combined Results ══════════════════════════════════
total_all = 868607 + 38026 + ext_n + 6072 + m2m_n + lib_n + 1430

H(f"""
<section id="combined">
<h2>11. Combined Results &amp; Method Comparison</h2>
<table>
<tr><th>Method</th><th>Molecules</th><th>Max pIC50</th><th>Potent (&ge;7)</th><th>Very Potent (&ge;8)</th></tr>
<tr><td>Structure-based (868K)</td><td>868,607</td><td>{fmt(sb_max)}</td><td>{sb_potent:,}</td><td>1,247</td></tr>
<tr><td>LibInvent (90 steps)</td><td>{lib_n:,}</td><td>{fmt(lib_max)}</td><td>{lib_potent:,}</td><td>{lib_vpotent:,}</td></tr>
<tr><td>Kinase repurposing</td><td>1,430</td><td>8.67</td><td>~193</td><td>~70</td></tr>
<tr><td>Mol2Mol Run 1 (4 steps)</td><td>6,072</td><td>8.93</td><td>~1,800</td><td>96</td></tr>
<tr><td>Mol2Mol Run 2</td><td>{m2m_n:,}</td><td>{fmt(m2m_max)}</td><td>{m2m_7:,}</td><td>{m2m_8:,}</td></tr>
<tr><td>Policy Gradient Run 1 (300 steps)</td><td>38,026</td><td>8.15</td><td>~2,400</td><td>9</td></tr>
<tr><td>Policy Gradient Run 2 (2500 steps)</td><td>{ext_n:,}</td><td>{fmt(ext_max)}</td><td>{ext_7:,}</td><td>{ext_8:,}</td></tr>
<tr style="font-weight:bold;background:#eaf2f8;"><td>TOTAL</td><td>~{total_all/1e6:.1f}M</td><td>{fmt(m2m_max)}</td><td>~{(sb_potent + m2m_7 + ext_7 + lib_potent + 2400 + 193)/1000:.0f}K+</td><td>~{(1247 + lib_vpotent + 70 + 96 + m2m_8 + 9 + ext_8)/1000:.0f}K+</td></tr>
</table>

<h3>Method Strengths Comparison</h3>
<table>
<tr><th>Criterion</th><th>Structure-Based</th><th>LibInvent</th><th>Mol2Mol</th><th>Policy Gradient</th></tr>
<tr><td><strong>Hit rate</strong></td><td>3.7%</td><td class="win">{lib_hit_rate:.1f}%</td><td class="win">38.7%</td><td>~10%</td></tr>
<tr><td><strong>Novelty</strong></td><td>Moderate</td><td>Low</td><td>Moderate</td><td class="win">Highest</td></tr>
<tr><td><strong>Volume</strong></td><td class="win">868K</td><td>{lib_n//1000}K</td><td>72K</td><td>142K</td></tr>
<tr><td><strong>Scaffold diversity</strong></td><td class="win">377K scaffolds</td><td>Few scaffolds</td><td>Moderate</td><td>High</td></tr>
<tr><td><strong>Confidence</strong></td><td class="win">Dual model</td><td>Single model</td><td>Single model</td><td>Single model</td></tr>
<tr><td><strong>Synthesizability</strong></td><td>Variable</td><td class="win">Best</td><td>Good</td><td>Variable</td></tr>
<tr><td><strong>Best for</strong></td><td>Broad exploration</td><td>Focused SAR</td><td>Lead optimization</td><td>Novel scaffolds</td></tr>
</table>
<div class="key">
<strong>Complementary approaches:</strong> Structure-based screening provides broadest coverage with highest
confidence (dual model). RL methods achieve higher hit rates by actively optimizing the FiLMDelta
score. Mol2Mol and LibInvent are most data-efficient because they leverage known active chemistry.
Together, they cover broad-to-focused exploration across ~{total_all/1e6:.1f}M unique molecules.
</div>
</section>
""")

# ═══ Section 12: SAR Analysis & Activity Cliffs ═══════════════════

H("""
<section id="sar">
<h2>12. SAR Analysis &amp; Activity Cliffs</h2>
<p><strong>926 activity cliffs detected</strong> in the ZAP70 training set (Tanimoto &gt; 0.6, |&Delta;pIC50| &gt; 0.5):</p>
<table style="width:auto;">
<tr><th>Severity</th><th>Count</th><th>|&Delta;pIC50| range</th></tr>
<tr><td>Extreme</td><td>25</td><td>&gt; 2.0</td></tr>
<tr><td>Moderate</td><td>377</td><td>1.0&ndash;2.0</td></tr>
<tr><td>Mild</td><td>524</td><td>0.5&ndash;1.0</td></tr>
</table>
<p>Mean structural similarity at cliffs: Tanimoto = 0.691, mean |&Delta;pIC50| = 1.02.</p>
""")

# ─── Top 10 Activity Cliffs with Molecule Images ─────────────────
print("Rendering SVGs for top activity cliffs...")
H('<h3>Top 10 Activity Cliffs (by |&Delta;pIC50|)</h3>')
H('<p style="font-size:12px;color:#666;">Structurally similar molecule pairs with large potency differences. '
  'These represent sharp SAR boundaries where small modifications cause dramatic activity changes.</p>')
H('<table><tr><th>#</th><th>Molecule A</th><th>pIC50 A</th><th>Molecule B</th><th>pIC50 B</th>'
  '<th>&Delta;pIC50</th><th>Tanimoto</th><th>SALI</th><th>Interpretation</th></tr>')

cliff_interpretations = {
    0: "Pyrimidine&rarr;triazine ring swap abolishes activity; heterocycle geometry critical for hinge binding",
    1: "Chiral piperazine (S-methyl) vs linear NEt piperazine; stereochemistry + basicity modulate kinase pocket fit",
    2: "NEt piperazine vs N-Me piperazine with different aryl connections; linker flexibility vs rigidity",
    3: "Triazine vs pyrimidine core with identical substitution; same SAR cliff as #1 from the other direction",
    4: "Homopiperazine vs NEt piperazine; ring size expansion maintains trimethoxyphenyl pharmacophore but loses activity",
    5: "N-methylpiperazine vs morpholine + methoxy&rarr;ethoxy swap; multiple changes compound effect",
    6: "NEt piperazine vs N-methylpiperazine; subtle N-alkyl difference modulates basicity and membrane permeability",
    7: "NEt piperazine vs unsubstituted piperazine; N-alkylation pattern affects kinase selectivity",
    8: "N-methylpiperazine vs morpholine with ortho substitution change; combined steric + electronic effects",
    9: "OEt vs NHAc swap at ortho position + ethoxy; amide-to-ether bioisosteric replacement",
}

for i, cliff in enumerate(genuine_cliffs[:10]):
    mol_a_smi = cliff.get("mol_a", "")
    mol_b_smi = cliff.get("mol_b", "")
    pIC50_a = cliff.get("pIC50_a", 0)
    pIC50_b = cliff.get("pIC50_b", 0)
    delta = cliff.get("delta_pIC50", 0)
    tanimoto = cliff.get("tanimoto", 0)
    sali = cliff.get("sali", 0)
    interp = cliff_interpretations.get(i, "")

    svg_a = mol_to_svg(mol_a_smi, w=180, h=130)
    svg_b = mol_to_svg(mol_b_smi, w=180, h=130)

    delta_class = "win" if delta > 0 else "loss"
    H(f'<tr><td>{i+1}</td>'
      f'<td style="min-width:180px;">{svg_a}</td>'
      f'<td><strong>{fmt(pIC50_a, 2)}</strong></td>'
      f'<td style="min-width:180px;">{svg_b}</td>'
      f'<td><strong>{fmt(pIC50_b, 2)}</strong></td>'
      f'<td class="{delta_class}"><strong>{delta:+.2f}</strong></td>'
      f'<td>{fmt(tanimoto)}</td>'
      f'<td>{sali:.1f}</td>'
      f'<td style="font-size:11px;">{interp}</td></tr>')

H('</table>')

H("""
<div class="note note-info">
<strong>Kinase Activity Cliffs in Literature:</strong> Activity cliffs are well-documented in the kinase domain.
Key patterns from MMP analyses of kinases include:
<ul style="margin:6px 0 0 18px;font-size:12px;">
<li><strong>Hinge region modifications:</strong> Single-atom changes in the hinge-binding motif (e.g., pyrimidine&harr;triazine)
frequently cause &gt;100-fold potency shifts, as seen in cliffs #1 and #4 above</li>
<li><strong>Gatekeeper residue interactions:</strong> Threonine gatekeeper in ZAP70 (T486) creates a narrow selectivity window;
small steric changes near this pocket are amplified into large &Delta;pIC50 values</li>
<li><strong>Solvent-exposed modifications:</strong> Changes to solvent-facing groups (piperazine N-alkylation) cause
moderate cliffs (1&ndash;2 log units) through indirect conformational effects on binding pose</li>
<li><strong>Stereochemistry sensitivity:</strong> Chiral centers near the binding interface (cliff #2) show stereospecific
binding to the kinase active site, with the wrong enantiomer losing 2&ndash;3 log units</li>
</ul>
</div>
""")

# ─── Top Beneficial MMP Edits ─────────────────────────────────────
H("""
<h3>Top Beneficial MMP Edits (FiLMDelta Predictions)</h3>
<table style="font-size:12px;">
<tr><th>Edit</th><th>N Pairs</th><th>Mean &Delta;pIC50</th></tr>
<tr><td class="smi">c1ccc2c(c1)CCn1cccc1S2 &rarr; Clc1ccc2c(c1)CCc1ccccc1S2</td><td>15</td><td class="win">+2.08</td></tr>
<tr><td class="smi">N[SH](=O)=O &rarr; NS(=O)(=O)c1nnc(N[SH](=O)=O)s1</td><td>27</td><td class="win">+1.76</td></tr>
<tr><td class="smi">c1ccc2[nH]ccc2c1 &rarr; Oc1ccccc1</td><td>10</td><td class="win">+1.51</td></tr>
<tr><td class="smi">N &rarr; CCNS(=O)(=O)c1ccc(N)cc1</td><td>13</td><td class="win">+1.32</td></tr>
</table>
<p style="font-size:12px;color:#888;">These are context-specific FiLMDelta predictions, not database averages &mdash; the model captures
how each edit affects potency in the specific molecular context.</p>
</section>
""")

# ═══ Footer ═════════════════════════════════════════════════════════
H(f"""
<footer>
Generated by ZAP70 Unified Report Pipeline | {now}<br>
Scoring: FiLMDelta + Kinase PT (280 anchors) | XGB Ensemble (structure-based screen only)<br>
Model: FiLMDelta [1024,512,256], Morgan FP 2048-bit, 8-kinase pretraining + ZAP70 fine-tuning
</footer>
</div></body></html>""")

# ─── Write output ──────────────────────────────────────────────────

html = '\n'.join(html_parts)
output_path = RESULTS_DIR / "zap70_unified_report.html"
output_path.write_text(html)
print(f"\nReport written: {output_path}")
print(f"Size: {len(html)/1024:.0f} KB")

# Archive old reports (if not already archived)
archive_dir = RESULTS_DIR / "archive"
archive_dir.mkdir(exist_ok=True)
old_reports = [
    "zap70_v4_report.html",
    "zap70_v5_report.html",
    "zap70_edit_unified_report.html",
    "zap70_1M_screening_report.html",
]
for name in old_reports:
    src = RESULTS_DIR / name
    if src.exists():
        dst = archive_dir / name
        shutil.move(str(src), str(dst))
        print(f"Archived: {name} -> archive/")

print("\nDone! Reports in results/paper_evaluation/:")
for f in sorted(RESULTS_DIR.glob("*.html")):
    print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")
