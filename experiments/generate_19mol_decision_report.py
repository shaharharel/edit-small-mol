#!/usr/bin/env python3
"""
Generate decision-focused HTML report for 19 ZAP70 candidate molecules.

Integrates ALL evidence sources into a unified ranking and recommendation.
Renders molecule structures as inline SVG images.

Usage:
    conda run -n quris python experiments/generate_19mol_decision_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import base64
import numpy as np
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

# ============================================================
# LOAD ALL DATA
# ============================================================

def load_json(name):
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

scoring = load_json("19_molecules_scoring.json")
pairwise = load_json("19_molecules_pairwise.json")
docking_data = load_json("19_molecules_docking.json")
kinase_sim = load_json("19_molecules_kinase_similarity.json")
cross_kinase = load_json("19_molecules_cross_kinase.json")
embeddings = load_json("19_molecules_embeddings.json")
binary_v2 = load_json("19_molecules_binary_v2.json")
dualstream = load_json("19_molecules_dualstream.json")
full_cv = load_json("19_molecules_full_cv.json")
stability = load_json("19_molecules_pairwise_stability.json")
uncertainty = load_json("19_molecules_20seed_uncertainty.json")

results = scoring["results"]

# ============================================================
# SMILES & MOLECULE IMAGES
# ============================================================

SMILES_RAW = [r["smiles_raw"] for r in results]

def clean_smiles(smi):
    smi_clean = smi.split(' |')[0] if ' |' in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return smi_clean
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)

SMILES_CLEAN = [clean_smiles(s) for s in SMILES_RAW]

def mol_to_svg(smi, size=(280, 200)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "<p>Could not render</p>"
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 1.5
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def mol_to_png_base64(smi, size=(300, 220)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# BUILD LOOKUP TABLES
# ============================================================

# Docking
dock_scores = {}
if docking_data:
    for d in docking_data:
        dock_scores[d["idx"]] = d.get("vina_score")

# Cross-kinase
btk_preds = {}
syk_preds = {}
p_binder_v1 = {}
if cross_kinase:
    for c in cross_kinase.get("BTK", {}).get("candidates", []):
        btk_preds[c["idx"]] = c["pred_mean"]
    for c in cross_kinase.get("SYK", {}).get("candidates", []):
        syk_preds[c["idx"]] = c["pred_mean"]
    for c in cross_kinase.get("binary_classifier", {}).get("candidates", []):
        p_binder_v1[c["idx"]] = c["p_binder"]

# Binary v2
p_binder_v2 = {}
if binary_v2:
    for c in binary_v2.get("v2_classifiers", {}).get("RF", {}).get("candidates", []):
        p_binder_v2[c["idx"]] = c["p_binder"]

# DualStream / FiLMDelta
ds_preds = {}
fd_preds = {}
if dualstream and "candidates" in dualstream:
    for c in dualstream["candidates"]:
        ds_preds[c["idx"]] = (c["dualstream_mean"], c["dualstream_std"])
        fd_preds[c["idx"]] = (c["filmdelta_mean"], c["filmdelta_std"])

# Stability
stab_data = {}
if stability:
    for s in stability["mol_stats"]:
        stab_data[s["mol_idx"]] = s

# 20-seed uncertainty
unc_data = {}
if uncertainty:
    for m in uncertainty["mol_results"]:
        unc_data[m["mol_idx"]] = m

# ChemBERTa embeddings
chemberta_mlm = {}
chemberta_mtr = {}
if embeddings:
    if "chemberta2-mlm" in embeddings:
        for c in embeddings["chemberta2-mlm"]["candidates"]:
            chemberta_mlm[c["idx"]] = (c["pred_mean"], c["pred_std"])
    if "chemberta2-mtr" in embeddings:
        for c in embeddings["chemberta2-mtr"]["candidates"]:
            chemberta_mtr[c["idx"]] = (c["pred_mean"], c["pred_std"])

# Kinase similarity
kinase_tc = {}
if kinase_sim:
    for c in kinase_sim["candidates"]:
        kinase_tc[c["candidate_idx"] + 1] = {
            "best_kinase": c["most_similar_kinase"],
            "best_tc": c["most_similar_score"],
            "sims": c["max_tanimoto"]
        }

# Descriptions
MOL_DESCRIPTIONS = {
    1: "Isoindoline-imidazole",
    2: "Spirocyclic-pyrazole-aminoalcohol",
    3: "Norbornane-aminopyrimidine-Cl",
    4: "Pyrrolidine-phenyl-pyrazole-difluoro",
    5: "Spirocyclic-pyrazole-pyrimidine",
    6: "Cyclobutane-pyrazole-aminoalcohol",
    7: "Pyrrolidine-tBu-pyrazole-aminoalcohol",
    8: "Azetidine-thiazole-fluorochloroindole",
    9: "Norbornane-pyrazole-pyrimidine",
    10: "Azetidine-thiazole-sulfonamide",
    11: "Triazole-cyclobutane-pyridine-acetamide",
    12: "Bicyclic-aminopyrimidine-cyanophenoxy",
    13: "Purine-pyrimidine",
    14: "Pyrrolidine-aminopyrimidine-ester",
    15: "Pyrrolidine-pyrazole-pyrazole",
    16: "Azetidine-diaminopyrimidine-aniline",
    17: "Pyrrolidine-purine-carboxamide",
    18: "Piperazine-pyrimidine-aminocyclohexane",
    19: "Spiro-pyrrolidine-pyrazole-benzofuran",
}


# ============================================================
# CSS
# ============================================================

CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; max-width: 1400px; margin: 0 auto;
       padding: 20px; color: #333; line-height: 1.5; background: #fafafa; }
h1 { color: #0d47a1; border-bottom: 3px solid #0d47a1; padding-bottom: 10px; }
h2 { color: #1565c0; border-bottom: 2px solid #1565c0; padding-bottom: 8px; margin-top: 40px; }
h3 { color: #2e7d32; margin-top: 30px; }
h4 { color: #455a64; }
.info-box { background: #e3f2fd; border-left: 4px solid #1976d2; padding: 15px; margin: 15px 0; border-radius: 4px; }
.warn-box { background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 4px; }
.success-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 4px; }
.danger-box { background: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 15px 0; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; }
th { background: #f5f5f5; font-weight: bold; position: sticky; top: 0; }
tr:nth-child(even) { background: #fafafa; }
tr:hover { background: #e3f2fd; }
.tier1 { background: #c8e6c9 !important; font-weight: bold; }
.tier2 { background: #e8f5e9 !important; }
.tier3 { background: #fff3e0 !important; }
.bottom { background: #ffebee !important; color: #999; }
.mol-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin: 20px 0; }
.mol-card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.mol-card.winner { border: 3px solid #4caf50; box-shadow: 0 4px 8px rgba(76,175,80,0.3); }
.mol-card.strong { border: 2px solid #2196f3; }
.mol-card h4 { margin: 5px 0; color: #1565c0; }
.mol-card .smiles { font-family: monospace; font-size: 10px; color: #666; word-break: break-all;
                    background: #f5f5f5; padding: 4px 6px; border-radius: 3px; margin: 5px 0; }
.mol-card .props { font-size: 11px; color: #555; }
.mol-card .mol-img { text-align: center; margin: 5px 0; }
.rank-badge { display: inline-block; background: #1565c0; color: white; border-radius: 50%;
              width: 28px; height: 28px; line-height: 28px; text-align: center; font-weight: bold; font-size: 14px; }
.rank-badge.gold { background: #ffc107; color: #333; }
.rank-badge.silver { background: #90a4ae; color: white; }
.rank-badge.bronze { background: #8d6e63; color: white; }
.evidence-bar { height: 8px; border-radius: 4px; display: inline-block; }
.bar-green { background: #4caf50; }
.bar-yellow { background: #ff9800; }
.bar-red { background: #f44336; }
.bar-gray { background: #bdbdbd; }
code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
.toc { background: white; border: 1px solid #ddd; padding: 15px 25px; border-radius: 8px; margin: 20px 0; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { padding: 3px 0; }
.toc a { text-decoration: none; color: #1565c0; }
.toc a:hover { text-decoration: underline; }
.split-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .split-panel { grid-template-columns: 1fr; } .mol-grid { grid-template-columns: 1fr; } }
</style>
"""


# ============================================================
# HTML GENERATION
# ============================================================

html = []

# ---- HEADER ----
html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ZAP70 Inhibitor Candidate Selection &mdash; Decision Report</title>
{CSS}
</head>
<body>

<h1>ZAP70 Inhibitor Candidate Selection</h1>
<p><em>Decision Report &mdash; Generated 2026-04-18 | 19 Candidates | 280 ZAP70 Training Molecules (ChEMBL) |
12 Independent Analyses</em></p>

<div class="info-box">
<p><strong>Objective:</strong> Identify which of 19 candidate covalent kinase inhibitors (all acrylamide warheads,
all AlphaFold2-validated to fit the ZAP70 pocket) has the highest ZAP70 inhibitory potency.</p>
<p><strong>Assumption:</strong> We expect at most a few genuine ZAP70 inhibitors among the 19. Ranking by
predicted ZAP70 pIC50 is the primary criterion.</p>
</div>
""")

# ---- TABLE OF CONTENTS ----
html.append("""
<div class="toc">
<h3>Contents</h3>
<ul>
<li><a href="#molecules">1. The 19 Candidate Molecules</a></li>
<li><a href="#methods">2. Methods Overview (12 Evidence Sources)</a></li>
<li><a href="#cv">3. Model Reliability &mdash; Cross-Validation</a></li>
<li><a href="#master">4. Master Evidence Table</a></li>
<li><a href="#stability">5. Ranking Stability (30-Trial Stress Test)</a></li>
<li><a href="#uncertainty">6. Prediction Uncertainty (20-Seed Estimation)</a></li>
<li><a href="#disagreements">7. Model Disagreements &amp; Resolution</a></li>
<li><a href="#cross-kinase">8. Cross-Kinase Evidence</a></li>
<li><a href="#verdict">9. Expert Panel Verdict &amp; Recommendation</a></li>
</ul>
</div>
""")

# ============================================================
# SECTION 1: THE 19 MOLECULES
# ============================================================
html.append("""
<h2 id="molecules">1. The 19 Candidate Molecules</h2>
<p>All candidates share a covalent acrylamide warhead and were designed as kinase inhibitors.
Structural diversity lies in the scaffold, hinge binder, and back-pocket substituents.
Max Tanimoto similarity to the 280-molecule ZAP70 training set: 0.177&ndash;0.280 (significant extrapolation).</p>
<div class="mol-grid">
""")

for idx in range(1, 20):
    smi = SMILES_CLEAN[idx - 1]
    r = results[idx - 1]
    desc = MOL_DESCRIPTIONS.get(idx, "")

    # Determine card class
    card_class = ""
    if unc_data.get(idx, {}).get("mean_rank", 99) <= 2:
        card_class = "winner"
    elif unc_data.get(idx, {}).get("mean_rank", 99) <= 5:
        card_class = "strong"

    svg = mol_to_svg(smi, size=(280, 190))
    has_salt = " [TFA salt]" if "." in SMILES_RAW[idx - 1].split(" |")[0] else ""

    u = unc_data.get(idx, {})
    pred_str = f"20-seed: {u.get('mean_pIC50', 0):.2f}&plusmn;{u.get('std_pIC50', 0):.2f}" if u else ""

    html.append(f"""
    <div class="mol-card {card_class}">
        <h4><span class="rank-badge">{idx}</span> Mol {idx}: {desc}{has_salt}</h4>
        <div class="mol-img">{svg}</div>
        <div class="smiles">{smi}</div>
        <div class="props">
            MW={r['MW']:.0f} | LogP={r['LogP']:.1f} | TPSA={r['TPSA']:.0f} | QED={r['QED']:.2f} |
            MaxTc={r['max_tanimoto']:.3f}<br>
            {pred_str}
        </div>
    </div>
    """)

html.append("</div>")  # close mol-grid


# ============================================================
# SECTION 2: METHODS OVERVIEW
# ============================================================
html.append("""
<h2 id="methods">2. Methods Overview (12 Evidence Sources)</h2>
<table>
<thead>
<tr><th>#</th><th>Method</th><th>Type</th><th>Embedding</th><th>What It Measures</th><th>Reliability*</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>FiLMDelta (20-seed)</td><td>Regression</td><td>Morgan FP 2048d</td>
    <td>pIC50 via anchor-based prediction, 20 different seeds</td><td class="tier1">HIGH</td></tr>
<tr><td>2</td><td>30-Trial Stability</td><td>Rank stability</td><td>Morgan FP 2048d</td>
    <td>Rank robustness across 30 different 80% subsamples</td><td class="tier1">HIGH</td></tr>
<tr><td>3</td><td>FiLMDelta (anchor-based)</td><td>Regression</td><td>Morgan FP 2048d</td>
    <td>pIC50 from median of 280 anchor predictions (5 seeds)</td><td class="tier2">MED-HIGH</td></tr>
<tr><td>4</td><td>DualStreamFiLM (anchor)</td><td>Regression</td><td>Morgan FP + DRFP 2048d</td>
    <td>pIC50 using gated reaction FP + Morgan diff (3 seeds)</td><td class="tier2">MED-HIGH</td></tr>
<tr><td>5</td><td>XGB/RF Ensemble (5 models)</td><td>Regression</td><td>4 FP types</td>
    <td>Direct absolute pIC50 prediction</td><td class="tier3">MEDIUM</td></tr>
<tr><td>6</td><td>DualObjective (kinase PT)</td><td>Regression</td><td>Morgan FP 2048d</td>
    <td>32K kinase pretrained &rarr; ZAP70 fine-tuned</td><td class="tier3">MEDIUM</td></tr>
<tr><td>7</td><td>AutoDock Vina</td><td>Docking</td><td>3D structure</td>
    <td>Binding pose energy (PDB 4K2R)</td><td class="tier3">LOW-MED</td></tr>
<tr><td>8</td><td>BTK cross-kinase</td><td>Regression</td><td>Morgan FP 2048d</td>
    <td>BTK FiLMDelta anchor prediction (proxy kinase)</td><td class="tier3">LOW-MED</td></tr>
<tr><td>9</td><td>SYK cross-kinase</td><td>Regression</td><td>Morgan FP 2048d</td>
    <td>SYK FiLMDelta anchor prediction (proxy kinase)</td><td class="tier3">LOW</td></tr>
<tr><td>10</td><td>Binary classifier (v2)</td><td>Classification</td><td>Morgan FP 2048d</td>
    <td>P(ZAP70 binder) with hard negatives</td><td class="tier3">LOW</td></tr>
<tr><td>11</td><td>ChemBERTa-MLM FiLMDelta</td><td>Regression</td><td>ChemBERTa 384d</td>
    <td>SMILES-based learned embeddings</td><td class="bottom">UNRELIABLE</td></tr>
<tr><td>12</td><td>ChemBERTa-MTR FiLMDelta</td><td>Regression</td><td>ChemBERTa 384d</td>
    <td>SMILES-based learned embeddings (MTR variant)</td><td class="bottom">UNRELIABLE</td></tr>
</tbody>
</table>
<p><em>*Reliability assessed based on: (a) CV performance on ZAP70 data, (b) internal consistency across seeds/trials,
(c) agreement between MLM and MTR variants (ChemBERTa fails this). See Section 3 for details.</em></p>
""")


# ============================================================
# SECTION 3: CV PERFORMANCE
# ============================================================
html.append("""
<h2 id="cv">3. Model Reliability &mdash; Cross-Validation on ZAP70</h2>
<p>5-fold CV on 280 ZAP70 molecules, 3 seeds (15 folds total). Three split types test different
generalization scenarios. <strong>Scaffold and distant splits are most relevant</strong> to our
extrapolation task (max Tc &le; 0.28).</p>
""")

if full_cv:
    method_order = ["Subtraction", "FiLMDelta", "FiLMDelta+KinasePT", "DualStreamFiLM", "DualStreamFiLM+KinasePT"]
    for split_name in ["random", "scaffold", "distant"]:
        if split_name not in full_cv:
            continue
        best_mae = min(full_cv[split_name][m].get('mae_mean', 99) for m in method_order if m in full_cv[split_name])
        html.append(f"""<h4>{split_name.title()} Split</h4>
<table><thead><tr><th>Method</th><th>MAE &darr;</th><th>Spearman &uarr;</th><th>Pearson</th><th>R&sup2;</th></tr></thead><tbody>""")
        for method in method_order:
            if method not in full_cv[split_name]:
                continue
            m = full_cv[split_name][method]
            mae = m.get('mae_mean', 0)
            is_best = abs(mae - best_mae) < 0.001
            cls = ' class="tier1"' if is_best else ''
            html.append(f"""<tr{cls}>
<td>{method}</td>
<td>{mae:.3f} &pm; {m.get('mae_std',0):.3f}</td>
<td>{m.get('spearman_r_mean',0):.3f} &pm; {m.get('spearman_r_std',0):.3f}</td>
<td>{m.get('pearson_r_mean',0):.3f} &pm; {m.get('pearson_r_std',0):.3f}</td>
<td>{m.get('r2_mean',0):.3f} &pm; {m.get('r2_std',0):.3f}</td>
</tr>""")
        html.append("</tbody></table>")

    # ChemBERTa CV
    if embeddings:
        html.append("""<h4>ChemBERTa Embeddings (for comparison)</h4>
<table><thead><tr><th>Embedder</th><th>MAE &darr;</th><th>Spearman &uarr;</th><th>Pearson</th><th>R&sup2;</th></tr></thead><tbody>""")
        for name in ["chemberta2-mlm", "chemberta2-mtr"]:
            if name in embeddings:
                cv = embeddings[name]["cv_metrics"]
                html.append(f"""<tr>
<td>{name}</td>
<td>{cv['mae_mean']:.3f} &pm; {cv['mae_std']:.3f}</td>
<td>{cv['spearman_r_mean']:.3f} &pm; {cv['spearman_r_std']:.3f}</td>
<td>{cv['pearson_r_mean']:.3f} &pm; {cv['pearson_r_std']:.3f}</td>
<td>{cv['r2_mean']:.3f} &pm; {cv['r2_std']:.3f}</td>
</tr>""")
        html.append("</tbody></table>")

    html.append("""
<div class="warn-box">
<strong>Key findings:</strong>
<ul>
<li><strong>Scaffold split</strong> (most relevant): FiLMDelta wins (MAE=0.700). Kinase pretraining <em>hurts</em>.</li>
<li><strong>Distant split:</strong> DualStreamFiLM wins (MAE=0.645). Edit-aware architectures outperform Subtraction by 10%.</li>
<li><strong>ChemBERTa CV performance is competitive</strong> (MLM MAE=0.555) but MLM and MTR produce
<em>anti-correlated</em> candidate rankings &mdash; unreliable for extrapolation.</li>
<li>Subtraction baseline is worst on all splits &mdash; edit-aware models consistently superior.</li>
</ul>
</div>
""")


# ============================================================
# SECTION 4: MASTER EVIDENCE TABLE
# ============================================================
html.append("""
<h2 id="master">4. Master Evidence Table</h2>
<p>All 12 evidence sources for each molecule. Sorted by 20-seed predicted pIC50 (our most reliable single metric).
Color coding: <span class="tier1" style="padding:2px 6px">Tier 1</span>
<span class="tier2" style="padding:2px 6px">Tier 2</span>
<span class="tier3" style="padding:2px 6px">Tier 3</span>
<span class="bottom" style="padding:2px 6px">Bottom</span></p>
""")

# Sort molecules by 20-seed mean pIC50
mol_order = sorted(range(1, 20), key=lambda idx: -unc_data.get(idx, {}).get("mean_pIC50", 0))

html.append("""<div style="overflow-x:auto;">
<table>
<thead>
<tr>
<th rowspan="2">Rank</th><th rowspan="2">Mol</th><th rowspan="2">SMILES</th>
<th colspan="3">PRIMARY (Morgan FP FiLMDelta)</th>
<th colspan="2">DualStream</th>
<th>XGB</th><th>DualObj</th>
<th>Docking</th>
<th colspan="2">Cross-Kinase</th>
<th>Binary</th>
<th colspan="2">ChemBERTa</th>
</tr>
<tr>
<th>20-seed pIC50</th><th>&sigma;</th><th>95% CI</th>
<th>DS pred</th><th>DS &sigma;</th>
<th>Ensemble</th><th>KinPT</th>
<th>Vina</th>
<th>BTK</th><th>SYK</th>
<th>P(bind)</th>
<th>MLM</th><th>MTR</th>
</tr>
</thead>
<tbody>
""")

for rank, idx in enumerate(mol_order):
    u = unc_data.get(idx, {})
    r = results[idx - 1]
    ds = ds_preds.get(idx, (0, 0))
    fd = fd_preds.get(idx, (0, 0))
    vs = dock_scores.get(idx)
    btk = btk_preds.get(idx, 0)
    syk = syk_preds.get(idx, 0)
    pb = p_binder_v2.get(idx, 0)
    mlm = chemberta_mlm.get(idx, (0, 0))
    mtr = chemberta_mtr.get(idx, (0, 0))
    st = stab_data.get(idx, {})

    # Tier class
    if rank < 2:
        cls = ' class="tier1"'
    elif rank < 5:
        cls = ' class="tier2"'
    elif rank < 10:
        cls = ''
    else:
        cls = ' class="bottom"'

    smi_short = SMILES_CLEAN[idx - 1]
    if len(smi_short) > 50:
        smi_short = smi_short[:47] + "..."

    html.append(f"""<tr{cls}>
<td>{rank + 1}</td><td><strong>{idx}</strong></td>
<td style="font-family:monospace;font-size:10px;text-align:left;max-width:180px;word-break:break-all">{smi_short}</td>
<td><strong>{u.get('mean_pIC50',0):.3f}</strong></td>
<td>{u.get('std_pIC50',0):.3f}</td>
<td>[{u.get('ci_2.5',0):.2f}, {u.get('ci_97.5',0):.2f}]</td>
<td>{ds[0]:.3f}</td><td>{ds[1]:.3f}</td>
<td>{r['xgb_ensemble']:.3f}</td>
<td>{r['dual_objective']:.3f}</td>
<td>{f"{vs:.1f}" if vs else "&mdash;"}</td>
<td>{btk:.2f}</td><td>{syk:.2f}</td>
<td>{pb:.3f}</td>
<td>{mlm[0]:.2f}</td><td>{mtr[0]:.2f}</td>
</tr>""")

html.append("</tbody></table></div>")


# ============================================================
# SECTION 5: STABILITY
# ============================================================
html.append("""
<h2 id="stability">5. Ranking Stability (30-Trial Stress Test)</h2>
<p>FiLMDelta trained 30 times from different 80% subsamples of the 280 training molecules.
Measures how robust each molecule's ranking is to training data variation.</p>
""")

if stability:
    sorted_stab = sorted(stability["mol_stats"], key=lambda x: x["mean_rank"])
    spr = stability["inter_trial_spearman_mean"]
    html.append(f"""<p>Inter-trial Spearman: <strong>{spr:.3f}</strong> &pm; {stability['inter_trial_spearman_std']:.3f}
(rankings are moderately consistent across trials)</p>
<table>
<thead>
<tr><th>Stab. Rank</th><th>Mol</th><th>Mean Rank</th><th>&sigma;</th><th>Median</th><th>Range</th>
<th>Top-3 %</th><th>Top-5 %</th><th>Mean Wins (/18)</th><th>Mean pIC50</th></tr>
</thead><tbody>""")

    for i, s in enumerate(sorted_stab):
        idx = s["mol_idx"]
        cls = ""
        if s["top3_pct"] > 50:
            cls = ' class="tier1"'
        elif s["top5_pct"] > 50:
            cls = ' class="tier2"'
        elif s["top5_pct"] > 0:
            cls = ''
        else:
            cls = ' class="bottom"'

        html.append(f"""<tr{cls}>
<td>{i+1}</td><td><strong>{idx}</strong></td>
<td>{s['mean_rank']:.1f}</td><td>{s['std_rank']:.1f}</td><td>{s['median_rank']:.0f}</td>
<td>{s['min_rank']}&ndash;{s['max_rank']}</td>
<td><strong>{s['top3_pct']:.0f}%</strong></td><td>{s['top5_pct']:.0f}%</td>
<td>{s['mean_wins']:.1f}</td><td>{s['mean_pred']:.3f}</td>
</tr>""")
    html.append("</tbody></table>")

    html.append("""
<div class="success-box">
<strong>Stability tiers:</strong>
<ul>
<li><strong>Dominant (#1):</strong> Mol 18 &mdash; rank 1.6&pm;0.8, top-3 in 97%, top-5 in 100%</li>
<li><strong>Strong top-5:</strong> Mols 1, 4, 7, 9 &mdash; all reach top-5 in &ge;70% of trials</li>
<li><strong>Fringe (sometimes top-5):</strong> Mols 15, 19, 5, 11</li>
<li><strong>Never top-5:</strong> Mols 10, 13, 14, 17 (despite mol 17 being in the original top-3)</li>
</ul>
</div>
""")


# ============================================================
# SECTION 6: UNCERTAINTY
# ============================================================
html.append("""
<h2 id="uncertainty">6. Prediction Uncertainty (20-Seed Estimation)</h2>
<p>FiLMDelta trained 20 times with different random seeds on the <em>same full training data</em>.
Provides confidence intervals for each candidate's predicted pIC50.</p>
""")

if uncertainty:
    sorted_unc = sorted(uncertainty["mol_results"], key=lambda x: -x["mean_pIC50"])
    html.append("""<table>
<thead>
<tr><th>Rank</th><th>Mol</th><th>Mean pIC50</th><th>&sigma;</th><th>95% CI</th><th>Range</th><th>Mean Rank</th></tr>
</thead><tbody>""")

    for i, m in enumerate(sorted_unc):
        idx = m["mol_idx"]
        cls = ""
        if i < 2:
            cls = ' class="tier1"'
        elif i < 5:
            cls = ' class="tier2"'
        elif i >= 14:
            cls = ' class="bottom"'

        html.append(f"""<tr{cls}>
<td>{i+1}</td><td><strong>{idx}</strong></td>
<td><strong>{m['mean_pIC50']:.3f}</strong></td><td>{m['std_pIC50']:.3f}</td>
<td>[{m['ci_2.5']:.3f}, {m['ci_97.5']:.3f}]</td>
<td>[{m['min_pIC50']:.3f}, {m['max_pIC50']:.3f}]</td>
<td>{m['mean_rank']:.1f}</td>
</tr>""")
    html.append("</tbody></table>")

    html.append("""
<div class="info-box">
<strong>Separation analysis:</strong> The 95% CIs of the top-2 (mols 18, 1) do NOT overlap with
molecules ranked 6+. The top-5 are separated from the bottom half by &ge;0.15 pIC50.
Mol 18 has the tightest CI (&sigma;=0.084) &mdash; the model is most confident about this molecule.
</div>
""")


# ============================================================
# SECTION 7: MODEL DISAGREEMENTS
# ============================================================
html.append("""
<h2 id="disagreements">7. Model Disagreements &amp; Resolution</h2>
<p>When models disagree, we must decide which to trust. Key disagreements:</p>
""")

# Build disagreement table for top molecules
html.append("""
<h3>7a. Ranking Comparison Across Methods (Top Molecules)</h3>
<table>
<thead>
<tr><th>Mol</th>
<th>FiLMDelta (anchor)</th><th>DualStream</th><th>20-Seed</th><th>Stability</th>
<th>Docking</th><th>BTK</th><th>Binary v2</th>
<th>ChemBERTa MLM</th><th>ChemBERTa MTR</th>
<th>Agreement?</th></tr>
</thead><tbody>
""")

# Rankings for each method
fd_ranking = dualstream.get("filmdelta_ranking", []) if dualstream else []
ds_ranking = dualstream.get("dualstream_ranking", []) if dualstream else []
stab_ranking = stability.get("consensus_ranking", []) if stability else []
unc_ranking = [m["mol_idx"] for m in sorted(uncertainty["mol_results"], key=lambda x: -x["mean_pIC50"])] if uncertainty else []

def get_rank(ranking_list, mol_idx):
    try:
        return ranking_list.index(mol_idx) + 1
    except ValueError:
        return "?"

# Docking ranking
dock_sorted = sorted(docking_data, key=lambda x: x.get("vina_score") or 0) if docking_data else []
dock_ranking = [d["idx"] for d in dock_sorted]

# BTK ranking
btk_ranking = cross_kinase["BTK"]["ranking"] if cross_kinase else []

# Binary v2 ranking
bin_ranking = binary_v2.get("v2_classifiers", {}).get("RF", {}).get("ranking", []) if binary_v2 else []

# ChemBERTa rankings
mlm_ranking_list = []
mtr_ranking_list = []
if embeddings:
    if "chemberta2-mlm" in embeddings:
        mlm_sorted = sorted(embeddings["chemberta2-mlm"]["candidates"], key=lambda x: -x["pred_mean"])
        mlm_ranking_list = [c["idx"] for c in mlm_sorted]
    if "chemberta2-mtr" in embeddings:
        mtr_sorted = sorted(embeddings["chemberta2-mtr"]["candidates"], key=lambda x: -x["pred_mean"])
        mtr_ranking_list = [c["idx"] for c in mtr_sorted]

focus_mols = [18, 1, 4, 7, 9, 11, 8, 13, 10]
agreements = {
    18: "Strong ML consensus",
    1: "Strong ML consensus",
    4: "SPLIT: DualStream vs Docking",
    7: "ML agrees, others disagree",
    9: "Good ML + docking + binary",
    11: "Binary high, ML moderate",
    8: "Best docking, weak ML",
    13: "Binary high, ML bottom",
    10: "MLM #1, everything else bottom",
}

for idx in focus_mols:
    cls = ""
    if idx in [18, 1]:
        cls = ' class="tier1"'
    elif idx in [4, 7, 9]:
        cls = ' class="tier2"'

    html.append(f"""<tr{cls}>
<td><strong>{idx}</strong></td>
<td>#{get_rank(fd_ranking, idx)}</td>
<td>#{get_rank(ds_ranking, idx)}</td>
<td>#{get_rank(unc_ranking, idx)}</td>
<td>#{get_rank(stab_ranking, idx)}</td>
<td>#{get_rank(dock_ranking, idx)}</td>
<td>#{get_rank(btk_ranking, idx)}</td>
<td>#{get_rank(bin_ranking, idx)}</td>
<td>#{get_rank(mlm_ranking_list, idx)}</td>
<td>#{get_rank(mtr_ranking_list, idx)}</td>
<td style="text-align:left;font-size:11px">{agreements.get(idx, "")}</td>
</tr>""")

html.append("</tbody></table>")

# ChemBERTa discrepancy
html.append("""
<h3>7b. ChemBERTa &mdash; Why It's Unreliable Here</h3>
<div class="danger-box">
<p><strong>ChemBERTa MLM vs MTR produce anti-correlated rankings for these candidates:</strong></p>
<ul>
<li>Mol 10: MLM rank <strong>#1</strong>, MTR rank <strong>#19</strong> (dead last)</li>
<li>Mol 15: MLM rank <strong>#18</strong>, MTR rank <strong>#1</strong></li>
<li>Mol 4: MLM rank <strong>#19</strong>, MTR rank #16</li>
</ul>
<p>When two variants of the same model produce opposite rankings, neither can be trusted for candidate selection.
ChemBERTa CV performance is good on in-distribution data (MAE=0.555), but the 384-dim learned representations
fail catastrophically in the extrapolation regime (Tc &le; 0.28). <strong>ChemBERTa rankings are discarded.</strong></p>
</div>

<h3>7c. DualStream Anomaly for Mol 4</h3>
<div class="warn-box">
<p>DualStreamFiLM predicts mol 4 at <strong>7.491</strong> (rank #1) &mdash; a full 1.15 pIC50 above FiLMDelta's 6.341.
No other molecule has a &gt;0.5 gap between these models. This outlier likely reflects the DRFP reaction
fingerprint over-weighting BTK-like transformation patterns (mol 4's BTK prediction is also #1 at 8.070).
Docking score is the worst of the top-5 (-6.784), suggesting the phenyl group doesn't fit ZAP70's pocket as well
as BTK's. <strong>DualStream's mol 4 prediction is likely a BTK artifact, not a ZAP70 signal.</strong></p>
</div>
""")


# ============================================================
# SECTION 8: CROSS-KINASE
# ============================================================
html.append("""
<h2 id="cross-kinase">8. Cross-Kinase Evidence</h2>
""")

if cross_kinase:
    html.append("""
<p><strong>Experimental basis:</strong> In ChEMBL, BTK&harr;ZAP70 Spearman = 0.823 (16 overlap molecules, p=9&times;10<sup>-5</sup>).
SYK&harr;ZAP70 Spearman = 0.312 (100 overlap molecules). BTK is the most informative proxy kinase.</p>
<p><strong>However:</strong> Model-predicted BTK vs ZAP70 rankings show Spearman = 0.037 (near zero).
This means our models capture <em>different SAR patterns</em> for BTK vs ZAP70, even though experimental
activities are correlated. Cross-kinase predictions provide weak, corroborating evidence only.</p>

<table>
<thead>
<tr><th>Mol</th><th>BTK Pred</th><th>BTK Rank</th><th>SYK Pred</th><th>SYK Rank</th>
<th>Binary v2 P(bind)</th><th>ZAP70 20-Seed</th></tr>
</thead><tbody>
""")
    for idx in mol_order[:10]:
        u = unc_data.get(idx, {})
        btk = btk_preds.get(idx, 0)
        syk = syk_preds.get(idx, 0)
        pb = p_binder_v2.get(idx, 0)
        btk_r = get_rank(btk_ranking, idx)
        syk_r = get_rank(cross_kinase["SYK"]["ranking"], idx) if cross_kinase else "?"
        cls = ' class="tier1"' if u.get("mean_rank", 99) <= 2 else (' class="tier2"' if u.get("mean_rank", 99) <= 5 else '')
        html.append(f"""<tr{cls}>
<td><strong>{idx}</strong></td>
<td>{btk:.2f}</td><td>#{btk_r}</td>
<td>{syk:.2f}</td><td>#{syk_r}</td>
<td>{pb:.3f}</td>
<td>{u.get('mean_pIC50',0):.3f}</td>
</tr>""")
    html.append("</tbody></table>")

    html.append("""
<div class="info-box">
<strong>Notable:</strong> Mol 4 is BTK #1 (8.070) but ZAP70 #4. Mol 9 has the highest binary classifier
score among top-5 (p=0.180) &mdash; it <em>looks like</em> a ZAP70 binder structurally.
Mol 18 has very low binary classifier score (0.040) despite being the #1 regression prediction &mdash;
it doesn't <em>look like</em> known binders but the model predicts it would <em>act like</em> one.
</div>
""")


# ============================================================
# SECTION 9: VERDICT
# ============================================================
html.append("""
<h2 id="verdict">9. Expert Panel Verdict &amp; Recommendation</h2>
""")

# Final top-5 cards with structures
html.append("""
<h3>Recommended Testing Order</h3>
<div class="mol-grid">
""")

tier_info = {
    18: ("TIER 1 &mdash; TEST FIRST", "winner",
         "Highest convergent evidence. #1 by 20-seed (6.632&pm;0.084), #1 by stability (97% top-3, 100% top-5), "
         "tightest CI, good docking (-8.17), validated piperazine-pyrimidine pharmacophore. "
         "Excellent drug properties (MW=345, LogP=0.65, QED=0.76)."),
    9: ("TIER 1 &mdash; SAFETY PICK", "winner",
        "Best structure-evidence harmony. #2 docking (-8.44), binary classifier #2 (p=0.180), "
        "regression top-5 (73% of trials). Ultra-rigid norbornane scaffold minimizes entropic penalty. "
        "Most similar to known ZAP70 actives (Tc=0.217). MW=350, LogP=0.99."),
    1: ("TIER 2 &mdash; DRUG-LIKE PICK", "strong",
        "Most classically drug-like candidate. #2 by 20-seed (6.588), highest QED (0.879), "
        "smallest MW (324), #1 by FiLMDelta anchor. Strong BTK cross-prediction (8.024 = #2). "
        "Risk: higher LogP (2.75), moderate docking (-7.76)."),
    4: ("TIER 2 &mdash; BTK-TYPE", "strong",
        "DualStream #1 (7.491) and BTK #1 (8.070), but worst docking of top-5 (-6.78). "
        "The 1.15 pIC50 gap between DualStream and FiLMDelta is a red flag. "
        "Likely a better BTK inhibitor than ZAP70 inhibitor. Test only if BTK activity also valuable."),
    7: ("TIER 3 &mdash; HIGH UNCERTAINTY", "",
        "Strong pairwise dominance (13/18 wins) but widest prediction interval (&sigma;=0.175). "
        "Most novel structure (Tc=0.177), unusual tert-butyl group, lowest QED (0.506). "
        "Could be as low as rank 7 if uncertainty resolves low. Deprioritize."),
}

for idx in [18, 9, 1, 4, 7]:
    tier_label, card_cls, rationale = tier_info[idx]
    smi = SMILES_CLEAN[idx - 1]
    r = results[idx - 1]
    u = unc_data.get(idx, {})
    s = stab_data.get(idx, {})
    vs = dock_scores.get(idx)
    btk = btk_preds.get(idx, 0)
    pb = p_binder_v2.get(idx, 0)

    svg = mol_to_svg(smi, size=(300, 210))

    html.append(f"""
    <div class="mol-card {card_cls}">
        <h4>{tier_label}</h4>
        <h4><span class="rank-badge {'gold' if idx==18 else 'silver' if idx==9 else ''}">{idx}</span>
            Mol {idx}: {MOL_DESCRIPTIONS[idx]}</h4>
        <div class="mol-img">{svg}</div>
        <div class="smiles">{smi}</div>
        <div class="props">
        <table style="font-size:11px;border:none;width:100%">
        <tr><td style="border:none;text-align:left">20-seed pIC50:</td>
            <td style="border:none;text-align:left"><strong>{u.get('mean_pIC50',0):.3f} &pm; {u.get('std_pIC50',0):.3f}</strong></td></tr>
        <tr><td style="border:none;text-align:left">95% CI:</td>
            <td style="border:none;text-align:left">[{u.get('ci_2.5',0):.3f}, {u.get('ci_97.5',0):.3f}]</td></tr>
        <tr><td style="border:none;text-align:left">Stability top-3/top-5:</td>
            <td style="border:none;text-align:left">{s.get('top3_pct',0):.0f}% / {s.get('top5_pct',0):.0f}%</td></tr>
        <tr><td style="border:none;text-align:left">Docking (Vina):</td>
            <td style="border:none;text-align:left">{f"{vs:.1f}" if vs else "&mdash;"} kcal/mol</td></tr>
        <tr><td style="border:none;text-align:left">BTK prediction:</td>
            <td style="border:none;text-align:left">{btk:.2f}</td></tr>
        <tr><td style="border:none;text-align:left">Binary P(binder):</td>
            <td style="border:none;text-align:left">{pb:.3f}</td></tr>
        <tr><td style="border:none;text-align:left">MW / LogP / QED:</td>
            <td style="border:none;text-align:left">{r['MW']:.0f} / {r['LogP']:.1f} / {r['QED']:.2f}</td></tr>
        </table>
        </div>
        <p style="font-size:12px;margin-top:8px">{rationale}</p>
    </div>
    """)

html.append("</div>")  # close mol-grid

# Summary box
html.append("""
<div class="success-box">
<h3>Final Recommendation</h3>
<table>
<thead>
<tr><th>Priority</th><th>Mol</th><th>Rationale</th><th>Risk</th></tr>
</thead>
<tbody>
<tr class="tier1"><td><strong>1st</strong></td><td><strong>Mol 18</strong></td>
    <td>Highest convergent evidence across all reliable models. Tightest prediction uncertainty.
    Validated kinase pharmacophore. Excellent drug properties.</td>
    <td>Binary classifier ranks it #18 (structural novelty, not a concern for regression models)</td></tr>
<tr class="tier1"><td><strong>2nd</strong></td><td><strong>Mol 9</strong></td>
    <td>Best structure-evidence harmony: strong docking + strong binary classifier + strong regression.
    Ultra-rigid scaffold = predictable binding mode.</td>
    <td>Slightly lower absolute prediction (6.375 vs 6.632)</td></tr>
<tr class="tier2"><td>3rd</td><td>Mol 1</td>
    <td>Most drug-like (highest QED). Classic kinase inhibitor design.
    Strong BTK cross-evidence.</td>
    <td>Higher LogP (2.75), moderate docking</td></tr>
<tr class="tier2"><td>4th</td><td>Mol 4</td>
    <td>Highest DualStream and BTK predictions.</td>
    <td>Worst docking of top-5 &mdash; likely better BTK than ZAP70 inhibitor</td></tr>
<tr class="tier3"><td>5th</td><td>Mol 7</td>
    <td>Strong pairwise dominance but highest uncertainty.</td>
    <td>Most novel structure, low drug-likeness (QED=0.506)</td></tr>
</tbody>
</table>

<p style="margin-top:15px"><strong>If you can test only ONE molecule: Mol 18.</strong><br>
If you can test TWO: Mol 18 + Mol 9 (complementary evidence profiles).<br>
If you can test THREE: Add Mol 1 (best drug-likeness, different scaffold).</p>
</div>
""")

# ============================================================
# FOOTER
# ============================================================
html.append("""
<hr>
<p style="font-size:11px;color:#999">
<strong>Data sources:</strong> 280 ZAP70 molecules (ChEMBL, CHEMBL2803) | 32K kinase within-assay pairs (pretraining) |
AutoDock Vina docking (PDB 4K2R) | Morgan FP 2048d + DRFP 2048d embeddings<br>
<strong>Models:</strong> FiLMDelta (f(B|&delta;)&minus;f(A|&delta;) with FiLM conditioning) |
DualStreamFiLM (gated DRFP + Morgan diff fusion) | XGB/RF ensemble | GradientBoosting subtraction baseline<br>
<strong>Validation:</strong> 5-fold CV &times; 3 seeds &times; 3 splits (random, scaffold, distant) |
30-trial stability (80% subsampling) | 20-seed uncertainty estimation<br>
<strong>Generated:</strong> 2026-04-18 by edit-small-mol framework
</p>
</body>
</html>
""")


# ============================================================
# WRITE OUTPUT
# ============================================================
full_html = "\n".join(html)
out_path = RESULTS_DIR / "19_molecules_report.html"
with open(out_path, "w") as f:
    f.write(full_html)

print(f"Report saved to {out_path}")
print(f"Size: {len(full_html):,} bytes")
print(f"Molecules rendered: 19 + 5 (verdict cards)")
