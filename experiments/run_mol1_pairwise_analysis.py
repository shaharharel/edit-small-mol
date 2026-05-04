#!/usr/bin/env python3
"""
Mol 1 Pairwise Analysis — Augment expansion report with novelty + pairwise metrics.

Loads the cached FiLMDelta model (reinvent4_film_model.pt) and the top-50 candidates
from the Mol 1 expansion run. Computes:

  1. Tc to Mol 1 (already in CSV — keep as 'Tc_to_Mol1')
  2. max_Tc_train      — Tc to nearest training neighbour
  3. mean_top10_Tc_train — mean Tc to 10 nearest training neighbours
  4. train_wins (out of 280) — # of training anchors where Δ(anchor → candidate) > 0
  5. Top-20 × Top-20 pairwise Δ matrix among generated candidates (with Tc cells)

Outputs:
  - results/paper_evaluation/mol1_expansion/pairwise_analysis.json
  - results/paper_evaluation/mol1_expansion/pairwise_analysis_report.html
  - augments existing expansion_report.html with new columns + footer link

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_pairwise_analysis.py
"""

import sys
import os
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_expansion"
MODEL_CACHE = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
SCORED_CSV = RESULTS_DIR / "all_scored_candidates.csv"
EXPANSION_HTML = RESULTS_DIR / "expansion_report.html"
OUT_JSON = RESULTS_DIR / "pairwise_analysis.json"
OUT_HTML = RESULTS_DIR / "pairwise_analysis_report.html"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
TOP_N_TABLE = 50      # how many candidates get the new property columns
TOP_N_PAIRWISE = 20   # how many candidates get full pairwise matrix


def fp_array(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fp_bitvec(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_film():
    print(f"[+] Loading FiLMDelta checkpoint: {MODEL_CACHE}")
    ckpt = torch.load(MODEL_CACHE, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    anchor_embs = ckpt["anchor_embs"]            # FloatTensor [280, 2048]
    anchor_pIC50 = np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)
    print(f"    Loaded model with {len(anchor_pIC50)} ZAP70 anchors")
    return model, scaler, anchor_embs, anchor_pIC50


def get_zap70_smiles():
    """Re-load the ZAP70 training SMILES (same source as the model checkpoint)."""
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    return smiles_df['smiles'].tolist(), smiles_df['pIC50'].values.astype(np.float64)


def main():
    print("=" * 70)
    print("MOL 1 PAIRWISE ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load model + training data ─────────────────────────────────────────────
    model, scaler, anchor_embs, anchor_pIC50 = load_film()
    train_smiles, train_pIC50 = get_zap70_smiles()
    assert np.allclose(train_pIC50, anchor_pIC50, atol=1e-6), \
        f"Anchor pIC50 mismatch ({len(train_pIC50)} vs {len(anchor_pIC50)})"

    # ── Load top candidates from expansion CSV ────────────────────────────────
    print(f"[+] Loading scored candidates from {SCORED_CSV}")
    df = pd.read_csv(SCORED_CSV)
    df = df.dropna(subset=['pIC50']).copy()
    df = df.sort_values('pIC50', ascending=False).head(TOP_N_TABLE).reset_index(drop=True)
    print(f"    Loaded top {len(df)} by pIC50 (range {df['pIC50'].min():.3f}–{df['pIC50'].max():.3f})")

    # ── Compute fingerprints for candidates ───────────────────────────────────
    print(f"[+] Computing fingerprints for {len(df)} candidates and {len(train_smiles)} training mols")
    cand_smiles = df['smiles'].tolist()
    cand_fps = [fp_array(s) for s in cand_smiles]
    valid_idx = [i for i, fp in enumerate(cand_fps) if fp is not None]
    df = df.iloc[valid_idx].reset_index(drop=True)
    cand_fps = np.array([cand_fps[i] for i in valid_idx], dtype=np.float32)
    cand_bv = [fp_bitvec(s) for s in df['smiles']]

    train_fps = np.array([fp_array(s) for s in train_smiles], dtype=np.float32)
    train_bv = [fp_bitvec(s) for s in train_smiles]

    # ── 1) Tc to Mol 1 (already in CSV as Tc_to_Mol18 — alias to Tc_to_Mol1)
    if 'Tc_to_Mol1' not in df.columns and 'Tc_to_Mol18' in df.columns:
        df['Tc_to_Mol1'] = df['Tc_to_Mol18']

    # ── 2/3) Tc to training (max + mean of top-10 closest)
    print(f"[+] Computing Tc to training set (max + mean-top10)")
    max_tc, mean_top10 = [], []
    for bv in cand_bv:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(bv, train_bv))
        max_tc.append(float(sims.max()))
        top10 = np.sort(sims)[-10:]
        mean_top10.append(float(top10.mean()))
    df['max_Tc_train'] = max_tc
    df['mean_top10_Tc_train'] = mean_top10
    print(f"    max_Tc_train range: {min(max_tc):.3f}–{max(max_tc):.3f}")
    print(f"    mean_top10_Tc_train range: {min(mean_top10):.3f}–{max(mean_top10):.3f}")

    # ── 4) Pairwise wins vs training (count of anchors with Δ(anchor → cand) > 0)
    print(f"[+] Predicting Δ(anchor → candidate) for {len(df)} × {len(train_smiles)} pairs")
    cand_embs = torch.FloatTensor(scaler.transform(cand_fps))
    n_anchors = anchor_embs.shape[0]

    train_wins = np.zeros(len(df), dtype=int)
    delta_matrix = np.zeros((len(df), n_anchors), dtype=np.float32)
    pred_pIC50_anchor = np.zeros((len(df), n_anchors), dtype=np.float32)

    with torch.no_grad():
        for j in range(len(df)):
            tgt = cand_embs[j:j+1].expand(n_anchors, -1)
            deltas = model(anchor_embs, tgt).numpy().flatten()
            delta_matrix[j] = deltas
            pred_pIC50_anchor[j] = anchor_pIC50 + deltas
            train_wins[j] = int((deltas > 0).sum())

    df['train_wins'] = train_wins
    df['anchor_pIC50_mean'] = pred_pIC50_anchor.mean(axis=1)
    df['anchor_pIC50_std'] = pred_pIC50_anchor.std(axis=1)
    print(f"    train_wins range: {train_wins.min()}–{train_wins.max()} (out of {n_anchors})")

    # ── 5) Top-N pairwise within generated ────────────────────────────────────
    n_pw = min(TOP_N_PAIRWISE, len(df))
    print(f"[+] Computing top-{n_pw} pairwise Δ matrix")
    pw_embs = cand_embs[:n_pw]
    pw_smiles = df['smiles'].tolist()[:n_pw]
    pw_bv = cand_bv[:n_pw]

    delta_pw = np.zeros((n_pw, n_pw), dtype=np.float32)
    with torch.no_grad():
        # delta_pw[i,j] = predicted Δ when going from anchor=i to target=j
        for i in range(n_pw):
            anchor = pw_embs[i:i+1].expand(n_pw, -1)
            delta_pw[i] = model(anchor, pw_embs).numpy().flatten()

    # Tc matrix among top-N
    tc_pw = np.zeros((n_pw, n_pw), dtype=np.float32)
    for i in range(n_pw):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(pw_bv[i], pw_bv))
        tc_pw[i] = sims

    # Pairwise wins among top-N: candidate j wins over i if delta_pw[i, j] > 0
    pw_wins = (delta_pw > 0).sum(axis=0) - np.diag(delta_pw > 0).astype(int)
    df.loc[:n_pw - 1, 'pw_wins_top'] = pw_wins.astype(int)
    avg_delta_as_anchor = delta_pw.mean(axis=1)  # average Δ when this mol is the anchor

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out = {
        "seed": MOL1_SMILES,
        "timestamp": datetime.now().isoformat(),
        "n_candidates_table": len(df),
        "n_pairwise": n_pw,
        "n_anchors": int(n_anchors),
        "candidates": df.to_dict(orient="records"),
        "delta_pw": delta_pw.tolist(),
        "tc_pw": tc_pw.tolist(),
        "pw_smiles": pw_smiles,
        "pw_wins": pw_wins.tolist(),
        "avg_delta_as_anchor": avg_delta_as_anchor.tolist(),
    }
    OUT_JSON.write_text(json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2))
    print(f"[+] Saved JSON → {OUT_JSON}")

    # ── Render HTML report ─────────────────────────────────────────────────────
    print(f"[+] Rendering HTML")
    render_html(df, delta_pw, tc_pw, pw_smiles, pw_wins, avg_delta_as_anchor, n_anchors)
    print(f"[+] Saved HTML → {OUT_HTML}")

    # ── Patch expansion report (add link + summary banner) ────────────────────
    if EXPANSION_HTML.exists():
        patch_expansion_report()
        print(f"[+] Patched {EXPANSION_HTML}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_html(df, delta_pw, tc_pw, pw_smiles, pw_wins, avg_delta_as_anchor, n_anchors):
    n_pw = len(pw_smiles)

    # Render mini structures for top-N
    def svg_of(smi, w=160, h=120):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        AllChem.Compute2DCoords(m)
        d = Draw.MolDraw2DSVG(w, h)
        d.drawOptions().bondLineWidth = 1.2
        d.DrawMolecule(m)
        d.FinishDrawing()
        return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; max-width: 1600px; color: #222; }
    h1 { color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
    h2 { color: #1f2937; margin-top: 32px; }
    table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }
    th { background: #f3f4f6; font-weight: 600; }
    td.smiles, th.smiles { text-align: left; max-width: 360px; word-break: break-all; font-family: monospace; font-size: 11px; }
    tr:nth-child(even) td { background: #fafafa; }
    .heatmap td { font-size: 10px; padding: 3px 5px; min-width: 40px; }
    .pos { background: #d1fae5 !important; }
    .neg { background: #fee2e2 !important; }
    .strong-pos { background: #34d399 !important; color: white; }
    .strong-neg { background: #f87171 !important; color: white; }
    .footnote { font-size: 12px; color: #555; max-width: 1100px; margin: 16px 0; }
    .struct { display: inline-block; vertical-align: middle; }
    """

    # Property table
    cols_show = ['pIC50', 'method', 'Tc_to_Mol1', 'max_Tc_train', 'mean_top10_Tc_train',
                 'train_wins', 'pw_wins_top', 'QED', 'MW', 'LogP', 'TPSA']

    rows = []
    for i, r in df.iterrows():
        pwwins = r.get('pw_wins_top', '')
        pwwins_str = f"{int(pwwins)}/{n_pw - 1}" if isinstance(pwwins, (int, float, np.integer, np.floating)) and not pd.isna(pwwins) else ""
        rows.append(f"""
        <tr>
          <td>{i+1}</td>
          <td class="struct">{svg_of(r['smiles'], 160, 110)}</td>
          <td class="smiles">{r['smiles']}</td>
          <td>{r['method']}</td>
          <td><b>{r['pIC50']:.3f}</b></td>
          <td>{r['Tc_to_Mol1']:.3f}</td>
          <td>{r['max_Tc_train']:.3f}</td>
          <td>{r['mean_top10_Tc_train']:.3f}</td>
          <td>{int(r['train_wins'])}/{n_anchors}</td>
          <td>{pwwins_str}</td>
          <td>{r['QED']:.2f}</td>
          <td>{r['MW']:.0f}</td>
          <td>{r['LogP']:.2f}</td>
          <td>{r['TPSA']:.1f}</td>
        </tr>""")
    prop_table = f"""
    <table>
      <tr>
        <th>#</th><th>Structure</th><th class="smiles">SMILES</th><th>Method</th>
        <th>pIC50</th>
        <th>Tc→Mol1</th>
        <th>max Tc→train</th>
        <th>mean Tc→top10 train</th>
        <th>train wins<br>(of {n_anchors})</th>
        <th>pw wins<br>(of {n_pw - 1})</th>
        <th>QED</th><th>MW</th><th>LogP</th><th>TPSA</th>
      </tr>
      {''.join(rows)}
    </table>
    """

    # Pairwise Δ heatmap (anchor=row, target=col)
    def cell_class(d):
        if d > 0.5: return "strong-pos"
        if d > 0.05: return "pos"
        if d < -0.5: return "strong-neg"
        if d < -0.05: return "neg"
        return ""

    head_cells = "".join(f"<th>C{j+1}</th>" for j in range(n_pw))
    body_rows = []
    for i in range(n_pw):
        cells = []
        for j in range(n_pw):
            if i == j:
                cells.append('<td style="background:#9ca3af;color:white;">—</td>')
            else:
                d = float(delta_pw[i, j])
                tc = float(tc_pw[i, j])
                cells.append(
                    f'<td class="{cell_class(d)}" title="C{i+1}→C{j+1}: Δ={d:+.2f}, Tc={tc:.2f}">'
                    f'{d:+.2f}<br><span style="font-size:9px;color:#666;">Tc {tc:.2f}</span></td>'
                )
        body_rows.append(f"<tr><th>C{i+1}</th>{''.join(cells)}</tr>")
    pairwise_table = f"""
    <table class="heatmap">
      <tr><th>anchor ↓ / target →</th>{head_cells}</tr>
      {''.join(body_rows)}
    </table>
    """

    # Pairwise summary table (sorted by wins)
    sum_rows = []
    order = np.argsort(-pw_wins)
    for rank, k in enumerate(order, 1):
        sum_rows.append(f"""
        <tr>
          <td>{rank}</td>
          <td>C{k+1}</td>
          <td class="struct">{svg_of(pw_smiles[k], 140, 100)}</td>
          <td class="smiles">{pw_smiles[k]}</td>
          <td>{int(pw_wins[k])}/{n_pw - 1}</td>
          <td>{avg_delta_as_anchor[k]:+.3f}</td>
          <td>{df.iloc[k]['pIC50']:.3f}</td>
          <td>{df.iloc[k]['method']}</td>
        </tr>""")
    summary_table = f"""
    <table>
      <tr>
        <th>Rank</th><th>ID</th><th>Structure</th><th class="smiles">SMILES</th>
        <th>Pairwise wins</th><th>Avg Δ as anchor<br>(over top-{n_pw})</th>
        <th>pIC50</th><th>Method</th>
      </tr>
      {''.join(sum_rows)}
    </table>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mol 1 Pairwise Analysis</title>
<style>{css}</style>
</head>
<body>
<h1>Mol 1 Expansion — Pairwise &amp; Novelty Analysis</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
   Seed: <code>{MOL1_SMILES}</code></p>

<h2>1. Top {len(df)} Candidates — Property Table</h2>
<p class="footnote">
  <b>Tc→Mol1</b>: Tanimoto (Morgan r=2 / 2048) to seed Mol 1.
  <b>max Tc→train</b>: Tc to nearest of {n_anchors} ZAP70 training mols (used in 19-mol report).
  <b>mean Tc→top10 train</b>: average over 10 nearest training neighbours (smoother novelty signal).
  <b>train wins</b>: # of training anchors where the model predicts Δ(anchor → candidate) > 0
  out of {n_anchors} (i.e., # training mols this candidate is predicted to beat).
  <b>pw wins</b>: head-to-head wins inside the top-{n_pw} cohort.
</p>
{prop_table}

<h2>2. Top-{n_pw} Pairwise Δ Matrix (predicted pIC50 shift)</h2>
<p class="footnote">
  Cell [row, col] = predicted Δ pIC50 when going from anchor=row to target=col.
  Green/positive → target predicted more potent than anchor; red/negative → less potent.
  Small grey number is Tanimoto similarity between the two candidates.
  Row-sums close to 0 mean the model can't separate that anchor from the cohort;
  asymmetry (Δ(i→j) ≠ −Δ(j→i)) signals prediction noise.
</p>
{pairwise_table}

<h2>3. Top-{n_pw} Pairwise Ranking</h2>
<p class="footnote">
  Sorted by within-cohort head-to-head wins. <b>Avg Δ as anchor</b> is the mean
  predicted Δ when this molecule is treated as the reference — a candidate with
  consistently positive avg-Δ-as-anchor "loses" to the others.
</p>
{summary_table}

<p class="footnote">⚠ Predictions inherit the same Tc&lt;0.3 extrapolation regime as the 19-candidate
   evaluation: per-candidate MAE ≈ 0.86. Within-cohort rankings are more reliable
   than absolute pIC50 values.</p>
</body>
</html>
"""
    OUT_HTML.write_text(html)


def patch_expansion_report():
    """Add a link to the pairwise report at the top of expansion_report.html."""
    html = EXPANSION_HTML.read_text()
    banner = (
        f'<div style="background:#fef3c7;border-left:4px solid #f59e0b;'
        f'padding:10px 16px;margin:8px 0 24px;font-family:-apple-system,sans-serif;font-size:14px;">'
        f'See also: <a href="pairwise_analysis_report.html"><b>Pairwise &amp; Novelty Analysis</b></a> '
        f'(Tc to training, train-wins, top-20 pairwise Δ matrix).</div>'
    )
    if "pairwise_analysis_report.html" in html:
        return  # already patched
    # Insert right after <h1>...</h1>
    import re
    html = re.sub(r'(</h1>)', r'\1' + banner, html, count=1)
    EXPANSION_HTML.write_text(html)


if __name__ == "__main__":
    main()
