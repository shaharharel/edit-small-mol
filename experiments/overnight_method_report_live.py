#!/usr/bin/env python3
"""
Live, sortable, filterable method-by-method overnight report.

Top-50 per method with full scoring suite + pairwise anchor metrics:
  - pIC50 (mean over 280 anchors)
  - pIC50 std (anchor spread = uncertainty proxy)
  - anchor_wins (count of 280 where δ > 0; high = beats most of training)
  - anchor_wins_ge7 (beats high-potency anchors specifically; max ~30)
  - direct_delta_from_mol1 (single FiLMDelta(Mol1, candidate))
  - delta_vs_mol1 (anchor-mean pIC50 minus Mol 1's anchor-mean pIC50)
  - SAScore, MW, QED, shape Tc, ESP-Sim, warhead Δ°, Tc→Mol1, max Tc→train

Uses DataTables.js for client-side sorting/filtering. Each method table starts at
top-20 visible; users can expand to show all 50, with column filters that maintain
the top-K-matching-filter view via DataTables' built-in ordered filter behavior.
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
import torch

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_zap70_train_smiles,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

RES = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_HTML = RES / "overnight_method_report_live.html"

TOP_N_PER_METHOD = 50      # cap for large methods (>100 candidates)
SHOW_ALL_THRESHOLD = 100   # if a method has ≤ this many candidates, show all
TOP_N_GLOBAL = 100
MOL1_BASELINE_PIC50 = 6.768  # FiLMDelta single-seed prediction for Mol 1


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Load FiLMDelta 3-seed ensemble for uniform metrics ──────────────────────

def load_filmdelta_ensemble():
    """Load all 3 seeds of FiLMDelta. Returns list of (model, scaler, anchor_embs, anchor_pIC50)."""
    ensemble_dir = RES / "reinvent4_film_ensemble"
    ensemble = []
    for k in range(3):
        path = ensemble_dir / f"film_seed{k}.pt"
        if not path.exists():
            log(f"  WARN: missing {path}; falling back to single-seed cached model")
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        sc = StandardScaler()
        sc.mean_ = ckpt["scaler_mean"]
        sc.scale_ = ckpt["scaler_scale"]
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = len(sc.mean_)
        ensemble.append({
            "model": m, "scaler": sc,
            "anchor_embs": ckpt["anchor_embs"],
            "anchor_pIC50": np.asarray(ckpt["anchor_pIC50"]).astype(np.float64),
        })
    return ensemble


def fp_array(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_pairwise_metrics(smiles_list, ensemble, mol1_smi):
    """For each SMILES, compute uniform pairwise metrics using the 3-seed ensemble.

    Returns dict per candidate with:
      - pIC50_mean: mean of 3 seed-anchor-means (NO penalty applied)
      - pIC50_std: std across the 3 seed-anchor-means (real ensemble disagreement)
      - anchor_mean_pIC50: same as pIC50_mean (alias for back-compat)
      - anchor_std_pIC50: anchor-spread within the FIRST seed (different from pIC50_std)
      - anchor_wins: count of 280 anchors where δ > 0 (averaged across 3 seeds, rounded)
      - anchor_wins_ge7: count of high-potency (≥7) anchors beaten (averaged across 3 seeds)
      - direct_delta_from_mol1: Mol1→cand single delta, averaged across 3 seeds
      - delta_vs_mol1: pIC50_mean(candidate) - pIC50_mean(Mol 1)
    """
    # First compute Mol 1's ensemble pIC50 for the Δ-vs-Mol1 baseline
    mol1_fp = fp_array(mol1_smi)
    mol1_seed_means = []
    for ens in ensemble:
        emb = torch.FloatTensor(ens["scaler"].transform(mol1_fp.reshape(1, -1)))
        n_anchors = ens["anchor_embs"].shape[0]
        target = emb.expand(n_anchors, -1)
        with torch.no_grad():
            deltas = ens["model"](ens["anchor_embs"], target).numpy().flatten()
        mol1_seed_means.append(float((ens["anchor_pIC50"] + deltas).mean()))
    mol1_baseline = float(np.mean(mol1_seed_means))

    results = []
    for smi in smiles_list:
        fp = fp_array(smi)
        if fp is None:
            results.append({})
            continue

        # Each seed produces an anchor-mean pIC50 + per-anchor delta vector
        seed_means = []
        per_seed_deltas = []
        per_seed_anchor_pIC50 = ensemble[0]["anchor_pIC50"]   # all seeds use same training set
        direct_from_mol1_per_seed = []

        for ens in ensemble:
            emb = torch.FloatTensor(ens["scaler"].transform(fp.reshape(1, -1)))
            n_anchors = ens["anchor_embs"].shape[0]
            target = emb.expand(n_anchors, -1)
            with torch.no_grad():
                deltas = ens["model"](ens["anchor_embs"], target).numpy().flatten()
                # Mol1 → cand direct delta
                mol1_emb_seed = torch.FloatTensor(ens["scaler"].transform(mol1_fp.reshape(1, -1)))
                direct = float(ens["model"](mol1_emb_seed, emb).numpy().flatten()[0])
            seed_means.append(float((ens["anchor_pIC50"] + deltas).mean()))
            per_seed_deltas.append(deltas)
            direct_from_mol1_per_seed.append(direct)

        seed_means = np.array(seed_means)
        all_deltas = np.array(per_seed_deltas)        # (3, 280)

        pIC50_mean = float(seed_means.mean())
        pIC50_std = float(seed_means.std())

        # Wins: average across seeds (per-anchor agreement)
        wins_per_seed = (all_deltas > 0).sum(axis=1)  # (3,)
        wins = int(wins_per_seed.mean())              # average wins across seeds, integer-ish

        high_potency_mask = per_seed_anchor_pIC50 >= 7.0
        n_high = int(high_potency_mask.sum())
        wins_high_per_seed = ((all_deltas > 0) & high_potency_mask[None, :]).sum(axis=1)  # (3,)
        wins_high = int(wins_high_per_seed.mean())

        # Anchor-spread within seed 0 (kept for completeness; not the main uncertainty)
        anchor_std_seed0 = float((per_seed_anchor_pIC50 + all_deltas[0]).std())

        results.append({
            "pIC50_mean": pIC50_mean,
            "pIC50_std": pIC50_std,
            "anchor_mean_pIC50": pIC50_mean,                   # alias
            "anchor_std_pIC50": anchor_std_seed0,
            "anchor_wins": wins,
            "anchor_wins_ge7": wins_high,
            "n_high_potency_anchors": n_high,
            "direct_delta_from_mol1": float(np.mean(direct_from_mol1_per_seed)),
            "delta_vs_mol1": pIC50_mean - mol1_baseline,
            "_mol1_baseline": mol1_baseline,
        })
    return results, mol1_baseline


# ── Method descriptions (re-used from previous report) ──────────────────────

METHODS = [
    {
        "name": "Tier 1 — Med-Chem Playbook (rule-based)",
        "csv": RES / "mol1_tier1_rules" / "tier1_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Deterministic SMARTS-based med-chem playbook applied only to non-warhead atoms. "
            "Includes aryl C-H/halogen/CN/CF3/OMe scans, imidazole C2 and N1 substituent variants, "
            "isoindoline ring modifications, and amide-linker bioisosteres. All 73 outputs preserve the warhead by construction."
        ),
    },
    {
        "name": "Tier 1.5 — Warhead Controls + Med-Chem Tricks",
        "csv": RES / "mol1_tier1_5_warhead_panel" / "tier1_5_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Two subsections. <b>1.5a:</b> warhead-control panel — α-Me/-F/-CN/-CF3 acrylamide, β-Me/β,β-diMe/β-NMe₂ "
            "(afatinib trick), propiolamide alkyne, propanamide null. <b>1.5b:</b> med-chem tricks — imidazole C5-scan, "
            "ortho-to-amide, aza-isoindoline, 1-substituted isoindoline."
        ),
    },
    {
        "name": "Tier 2 — Fragment Replacement (curated 204)",
        "csv": RES / "mol1_tier2_fragreplace" / "tier2_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "Fragment A (acrylamide-isoindoline-COOH) held fixed; amine partner replaced via amide coupling. "
            "Curated library: 34 hand-picked + 39 ChEMBL whole-mol + 154 BRICS-decomposed = 204 unique products. "
            "Synthesis = single HATU/T3P amide coupling step."
        ),
    },
    {
        "name": "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)",
        "csv": RES / "aichem_tier2_scaled" / "products_scored.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50_film",
        "description": (
            "Same fragment-replacement strategy as Tier 2, scaled by mining ChEMBL 35 (2.5M compounds) for "
            "primary/secondary amines under MW ≤ 500. Generated 498,992 unique amide-coupling products and "
            "scored all on ai-chem with single-seed FiLMDelta + tiled-batch CPU inference."
        ),
    },
    {
        "name": "Tier 3 v2 — Constrained Generative (single-seed, old)",
        "csv": RES / "mol1_tier3_constrained" / "tier3_candidates.csv",
        "smiles_col": "smiles", "pic50_col": "pIC50",
        "description": (
            "REINVENT4 with warhead-locked LibInvent scaffold + MatchingSubstructure warhead gate on Mol2Mol/De Novo + "
            "protected_ids CReM + warhead-aware BRICS. Uses single-seed cached FiLMDelta as scoring component."
        ),
    },
    {
        "name": "Tier 3 v3 — LibInvent locked (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "libinvent_locked" / "libinvent_locked_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "Same warhead-locked scaffold as v2, but the scoring component is now the 3-seed FiLMDelta ensemble returning "
            "mean − 0.5·std. The agent is rewarded for high-confidence high-pIC50 picks rather than just high-mean picks."
        ),
    },
    {
        "name": "Tier 3 v3 — Mol2Mol + warhead gate (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "mol2mol_warhead" / "mol2mol_warhead_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 Mol2Mol seeded on Mol 1 with hard warhead gate (MatchingSubstructure component, weight 1.0 → "
            "geometric-mean → 0 if warhead missing) + uncertainty-aware FiLMDelta reward. Local SAR with warhead preserved."
        ),
    },
    {
        "name": "Tier 3 v3 — De Novo + warhead gate (uncertainty-aware)",
        "csv": RES / "aigpu_overnight" / "denovo_warhead" / "denovo_warhead_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "From-scratch REINVENT4 De Novo with hard warhead gate + Tanimoto-to-Mol-1 reward + uncertainty-aware FiLMDelta. "
            "Most exploratory of the warhead-preserving methods — generates novel scaffolds that retain the acrylamide."
        ),
    },
    {
        "name": "Tier 4 — De Novo unconstrained",
        "csv": RES / "aigpu_overnight" / "tier4_denovo" / "tier4_denovo_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 De Novo with FiLMDelta-uncertainty (0.7) + QED + alerts only. NO warhead constraint at generation; "
            "checked only as post-filter. Tests whether FiLMDelta alone steers toward acrylamide-bearing molecules organically."
        ),
    },
    {
        "name": "Tier 4 — Mol2Mol unconstrained",
        "csv": RES / "aigpu_overnight" / "tier4_mol2mol" / "tier4_mol2mol_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "REINVENT4 Mol2Mol seeded on Mol 1 with FiLMDelta-uncertainty + QED reward, no warhead gate, post-filter only. "
            "Lets warhead drift but Mol2Mol prior biases toward Mol-1-similar candidates."
        ),
    },
    {
        "name": "Method A — De Novo FiLMDelta-driven",
        "csv": RES / "aigpu_overnight" / "method_a" / "method_a_filmdriven_denovo_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "De Novo with FiLMDelta-uncertainty weighted heavily as the PRIMARY reward (weight 0.85), QED 0.15, no warhead constraint. "
            "Pure 'what does the model think is potent?' exploration."
        ),
    },
    {
        "name": "Method B — Mol2Mol FiLMDelta-driven",
        "csv": RES / "aigpu_overnight" / "method_b" / "method_b_filmdriven_mol2mol_1.csv",
        "smiles_col": "SMILES", "pic50_col": "FiLMDelta pIC50 (raw)",
        "description": (
            "Mol2Mol seeded on Mol 1 with FiLMDelta-uncertainty (0.7) + Tc-to-Mol-1 (0.3) + QED, no warhead gate. "
            "SAR exploration around Mol 1 prioritising potency over warhead preservation."
        ),
    },
]


def svg_of(smi, w=140, h=100, clickable=True):
    """Render molecule SVG. If clickable=True, wrap in a clickable div that triggers a modal expand."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        AllChem.Compute2DCoords(m)
        d = Draw.MolDraw2DSVG(w, h)
        d.drawOptions().bondLineWidth = 1.0
        d.DrawMolecule(m)
        d.FinishDrawing()
        svg = d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
        if clickable:
            # Encode the SMILES in a data attribute (HTML-escaped)
            esc = smi.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
            return (f'<div class="mol-thumb" data-smiles="{esc}" '
                    f'onclick="showMolModal(this)" title="Click to enlarge">{svg}</div>')
        return svg
    except Exception:
        return ""


def fmt(v, nd=3, na="—"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
        return na
    if isinstance(v, (bool, np.bool_)):
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
    sub.columns = ["smiles", "pIC50_method"]
    sub = sub.sort_values("pIC50_method", ascending=False).drop_duplicates(subset="smiles").reset_index(drop=True)
    sub["method"] = meta["name"]
    return sub


def render_method_table(df, table_id):
    """Build HTML <table> with id + classes for DataTables.js. Uses ensemble mean ± std as pIC50."""
    head = ("<thead><tr>"
            "<th>#</th><th>Structure</th><th class='smiles'>SMILES</th>"
            "<th>pIC50<br/>(mean ± std)</th>"
            "<th>Δ vs Mol1</th><th>direct δ<br/>from Mol1</th>"
            "<th>wins<br/>/280</th><th>wins ≥7<br/>(of pots.)</th>"
            "<th>SAS</th><th>MW</th><th>QED</th>"
            "<th>shape Tc<br/>vs seed</th><th>ESP-Sim<br/>vs seed</th><th>warhead Δ°</th>"
            "<th>Tc→Mol1</th><th>max Tc<br/>→train</th><th>WH</th>"
            "</tr></thead>")
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        m, s = r.get('pIC50_mean'), r.get('pIC50_std')
        if pd.notna(m) and pd.notna(s):
            pic_disp = f"<b>{m:.3f}</b> ± {s:.3f}"
            pic_sort = float(m)
        else:
            pic_disp = "—"
            pic_sort = 0.0
        rows.append(f"""
        <tr>
          <td data-order="{i+1}">{i+1}</td>
          <td>{svg_of(r['smiles'])}</td>
          <td class="smiles">{r['smiles']}</td>
          <td data-order="{pic_sort}">{pic_disp}</td>
          <td data-order="{r.get('delta_vs_mol1', 0)}">{fmt(r.get('delta_vs_mol1'), 3)}</td>
          <td data-order="{r.get('direct_delta_from_mol1', 0)}">{fmt(r.get('direct_delta_from_mol1'), 3)}</td>
          <td data-order="{r.get('anchor_wins', 0)}">{int(r.get('anchor_wins', 0))}</td>
          <td data-order="{r.get('anchor_wins_ge7', 0)}">{int(r.get('anchor_wins_ge7', 0))}/{int(r.get('n_high_potency_anchors', 0))}</td>
          <td data-order="{r.get('SAScore', 99) or 99}">{fmt(r.get('SAScore'), 2)}</td>
          <td data-order="{r.get('MW', 0)}">{fmt(r.get('MW'), 0)}</td>
          <td data-order="{r.get('QED', 0)}">{fmt(r.get('QED'), 2)}</td>
          <td data-order="{r.get('shape_Tc_seed', 0) or 0}">{fmt(r.get('shape_Tc_seed'), 3)}</td>
          <td data-order="{r.get('esp_sim_seed', 0) or 0}">{fmt(r.get('esp_sim_seed'), 3)}</td>
          <td data-order="{r.get('warhead_dev_deg', 999) or 999}">{fmt(r.get('warhead_dev_deg'), 1)}</td>
          <td data-order="{r.get('Tc_to_Mol1', 0)}">{fmt(r.get('Tc_to_Mol1'), 3)}</td>
          <td data-order="{r.get('max_Tc_train', 0)}">{fmt(r.get('max_Tc_train'), 3)}</td>
          <td>{fmt(r.get('warhead_intact'), na='—')}</td>
        </tr>""")
    return f"<table id='{table_id}' class='dt-table display compact'>{head}<tbody>{''.join(rows)}</tbody></table>"


def main():
    log("=" * 70)
    log("LIVE METHOD-BY-METHOD REPORT")
    log("=" * 70)

    train_smiles = load_zap70_train_smiles()
    log(f"Loaded {len(train_smiles)} ZAP70 training SMILES")

    log("Loading 3-seed FiLMDelta ensemble for uniform pairwise metrics...")
    ensemble = load_filmdelta_ensemble()
    if ensemble is None:
        log("ERROR: ensemble not available")
        return
    a_p = ensemble[0]["anchor_pIC50"]
    log(f"  Ensemble: 3 seeds, {len(a_p)} anchors, mean pIC50 {a_p.mean():.2f}, "
        f"max {a_p.max():.2f}, # ≥7.0: {(a_p >= 7.0).sum()}")

    method_top_data = {}
    for meta in METHODS:
        df = load_method(meta)
        if df is None or len(df) == 0:
            log(f"  ✗ {meta['name']}: no data"); continue
        log(f"  ✓ {meta['name']}: {len(df):,} candidates")

        # Show all if small, cap if large
        n_show = len(df) if len(df) <= SHOW_ALL_THRESHOLD else TOP_N_PER_METHOD
        top = df.head(n_show).copy().reset_index(drop=True)
        log(f"    Showing {n_show} (cap={SHOW_ALL_THRESHOLD if len(df) > SHOW_ALL_THRESHOLD else 'all'})")

        # 3D + descriptor scoring
        log(f"    Scoring {len(top)} with 3D suite...")
        top_scored = score_dataframe(
            top.drop(columns=["pIC50_method", "method"]),
            smiles_col="smiles",
            train_smiles=train_smiles,
            compute_3d=True,
            pIC50_predictor=None,
        )
        top_scored["pIC50_method"] = top["pIC50_method"].values
        top_scored["method"] = meta["name"]

        # Pairwise anchor metrics from 3-seed ensemble (uniform across all methods)
        log(f"    Ensemble pairwise metrics on {len(top)}...")
        pw_metrics, mol1_baseline = compute_pairwise_metrics(
            top_scored["smiles"].tolist(), ensemble, MOL1_SMILES,
        )
        for col in ["pIC50_mean", "pIC50_std", "anchor_mean_pIC50", "anchor_std_pIC50",
                    "anchor_wins", "anchor_wins_ge7", "n_high_potency_anchors",
                    "direct_delta_from_mol1", "delta_vs_mol1"]:
            top_scored[col] = [m.get(col, np.nan) for m in pw_metrics]

        method_top_data[meta["name"]] = {
            "description": meta["description"],
            "n_total": int(len(df)),
            "max_pIC50": float(df["pIC50_method"].max()),
            "median_pIC50": float(df["pIC50_method"].median()),
            "df": top_scored,
        }

    if not method_top_data:
        log("ERROR: no data"); return

    # Build global top — sort by ensemble mean (uniform metric across all methods)
    big = pd.concat([d["df"] for d in method_top_data.values()], ignore_index=True)
    big = big.sort_values("pIC50_mean", ascending=False).reset_index(drop=True)
    # Also sort each method's top-50 by ensemble mean for consistency
    for name in method_top_data:
        method_top_data[name]["df"] = method_top_data[name]["df"].sort_values(
            "pIC50_mean", ascending=False, na_position="last"
        ).reset_index(drop=True)
    global_top = big.head(TOP_N_GLOBAL).copy().reset_index(drop=True)
    log(f"Global top-{TOP_N_GLOBAL} pooled")

    # ── HTML ────────────────────────────────────────────────────────────────
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;max-width:1900px;color:#222;line-height:1.45;}
    .mol-thumb{cursor:pointer;display:inline-block;border-radius:4px;}
    .mol-thumb:hover{outline:2px solid #2563eb;outline-offset:2px;}
    /* Modal */
    .mol-modal-bg{display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;
        background:rgba(0,0,0,0.65);z-index:9999;justify-content:center;align-items:center;}
    .mol-modal-bg.show{display:flex;}
    .mol-modal{background:white;border-radius:8px;padding:20px 28px;
        max-width:90vw;max-height:90vh;overflow:auto;position:relative;box-shadow:0 10px 50px rgba(0,0,0,0.3);}
    .mol-modal svg{display:block;width:600px;height:500px;}
    .mol-modal .close{position:absolute;top:6px;right:12px;font-size:24px;
        background:none;border:none;cursor:pointer;color:#444;line-height:1;padding:4px 8px;}
    .mol-modal .close:hover{color:#dc2626;}
    .mol-modal .smiles-box{font-family:monospace;font-size:12px;
        background:#f3f4f6;padding:8px 12px;border-radius:4px;margin-top:10px;
        word-break:break-all;max-width:600px;}
    h1{color:#111;border-bottom:3px solid #2563eb;padding-bottom:8px;}
    h2{color:#1f2937;margin-top:48px;border-bottom:1px solid #d1d5db;padding-bottom:4px;}
    h3{color:#374151;margin-top:24px;}
    .desc{font-size:13px;color:#444;max-width:1100px;background:#f9fafb;padding:10px 14px;border-left:3px solid #2563eb;margin:6px 0;}
    .stats{font-size:13px;color:#666;margin:6px 0;}
    .footnote{font-size:12px;color:#555;max-width:1300px;margin:8px 0;line-height:1.5;}
    .seed-box{display:inline-block;border:2px solid #2563eb;border-radius:6px;padding:8px;}
    table.dt-table{border-collapse:collapse;font-size:11px;}
    table.dt-table th,table.dt-table td{border:1px solid #d1d5db;padding:4px 6px;}
    table.dt-table th{background:#f3f4f6;font-weight:600;text-align:center;font-size:10px;line-height:1.2;}
    table.dt-table td{text-align:right;font-variant-numeric:tabular-nums;}
    table.dt-table td.smiles{font-family:monospace;font-size:9px;max-width:250px;word-break:break-all;text-align:left;}
    table.dt-table td:nth-child(2){text-align:center;padding:1px;}
    """

    sections = []
    table_inits = []
    for idx, meta in enumerate(METHODS):
        if meta["name"] not in method_top_data: continue
        md = method_top_data[meta["name"]]
        table_id = f"tbl_{idx}"
        sections.append(f"""
        <h2>{meta['name']}</h2>
        <div class="desc">{md['description']}</div>
        <div class="stats">
          <b>Total candidates:</b> {md['n_total']:,} &nbsp; · &nbsp;
          <b>max pIC50:</b> {md['max_pIC50']:.2f} &nbsp; · &nbsp;
          <b>median pIC50:</b> {md['median_pIC50']:.2f} &nbsp; · &nbsp;
          <b>showing:</b> top {TOP_N_PER_METHOD} (sortable, filterable)
        </div>
        {render_method_table(md['df'], table_id)}
        """)
        table_inits.append(table_id)

    # Global top with method column
    def render_global_table(df, table_id):
        head = ("<thead><tr>"
                "<th>#</th><th>Method</th><th>Structure</th><th class='smiles'>SMILES</th>"
                "<th>pIC50<br/>(mean ± std)</th>"
                "<th>Δ vs Mol1</th><th>direct δ<br/>from Mol1</th>"
                "<th>wins<br/>/280</th><th>wins ≥7</th>"
                "<th>SAS</th><th>MW</th><th>QED</th>"
                "<th>shape Tc</th><th>ESP-Sim</th><th>WH Δ°</th>"
                "<th>Tc→Mol1</th><th>max Tc<br/>→train</th><th>WH</th>"
                "</tr></thead>")
        rows = []
        for i, r in df.reset_index(drop=True).iterrows():
            method_short = r['method'].split('—')[0].strip() if '—' in r['method'] else r['method']
            m, s = r.get('pIC50_mean'), r.get('pIC50_std')
            if pd.notna(m) and pd.notna(s):
                pic_disp = f"<b>{m:.3f}</b> ± {s:.3f}"
                pic_sort = float(m)
            else:
                pic_disp = "—"
                pic_sort = 0.0
            rows.append(f"""
            <tr>
              <td data-order="{i+1}">{i+1}</td>
              <td>{method_short}</td>
              <td>{svg_of(r['smiles'])}</td>
              <td class="smiles">{r['smiles']}</td>
              <td data-order="{pic_sort}">{pic_disp}</td>
              <td data-order="{r.get('delta_vs_mol1', 0)}">{fmt(r.get('delta_vs_mol1'), 3)}</td>
              <td data-order="{r.get('direct_delta_from_mol1', 0)}">{fmt(r.get('direct_delta_from_mol1'), 3)}</td>
              <td data-order="{r.get('anchor_wins', 0)}">{int(r.get('anchor_wins', 0))}</td>
              <td data-order="{r.get('anchor_wins_ge7', 0)}">{int(r.get('anchor_wins_ge7', 0))}</td>
              <td data-order="{r.get('SAScore', 99) or 99}">{fmt(r.get('SAScore'), 2)}</td>
              <td data-order="{r.get('MW', 0)}">{fmt(r.get('MW'), 0)}</td>
              <td data-order="{r.get('QED', 0)}">{fmt(r.get('QED'), 2)}</td>
              <td data-order="{r.get('shape_Tc_seed', 0) or 0}">{fmt(r.get('shape_Tc_seed'), 3)}</td>
              <td data-order="{r.get('esp_sim_seed', 0) or 0}">{fmt(r.get('esp_sim_seed'), 3)}</td>
              <td data-order="{r.get('warhead_dev_deg', 999) or 999}">{fmt(r.get('warhead_dev_deg'), 1)}</td>
              <td data-order="{r.get('Tc_to_Mol1', 0)}">{fmt(r.get('Tc_to_Mol1'), 3)}</td>
              <td data-order="{r.get('max_Tc_train', 0)}">{fmt(r.get('max_Tc_train'), 3)}</td>
              <td>{fmt(r.get('warhead_intact'), na='—')}</td>
            </tr>""")
        return f"<table id='{table_id}' class='dt-table display compact'>{head}<tbody>{''.join(rows)}</tbody></table>"

    sections.append(f"""
    <h2>Global Top-{TOP_N_GLOBAL} Across All Methods</h2>
    <p class="footnote">Pooled top-{TOP_N_PER_METHOD} from each method, sorted by ensemble pIC50 mean.
    All pIC50 values are computed identically with the 3-seed FiLMDelta ensemble — uniform across all methods,
    no penalty applied. The ± value is the std across the 3 seed-anchor-means for that candidate (lower = ensemble more confident).</p>
    {render_global_table(global_top, 'tbl_global')}
    """)
    table_inits.append("tbl_global")

    # DataTables init JS
    init_js = "\n".join([
        f"$('#{tid}').DataTable({{"
        f"  pageLength: 20,"
        f"  lengthMenu: [[10, 20, 50, 100, -1], [10, 20, 50, 100, 'All']],"
        f"  order: [[3, 'desc']]," if not tid.endswith("global") else
        f"$('#{tid}').DataTable({{"
        f"  pageLength: 20,"
        f"  lengthMenu: [[10, 20, 50, 100, -1], [10, 20, 50, 100, 'All']],"
        f"  order: [[4, 'desc']],"
        for tid in table_inits
    ])
    # Above expression has a tricky construction — simplify
    init_js_lines = []
    for tid in table_inits:
        # global table has Method col at idx 1, so pIC50 is idx 4; method tables: pIC50 at idx 3
        order_col = 4 if tid == "tbl_global" else 3
        init_js_lines.append(
            f"$('#{tid}').DataTable({{ "
            f"dom: 'Qlfrtip', "
            f"pageLength: 20, "
            f"lengthMenu: [[10, 20, 50, 100, -1], [10, 20, 50, 100, 'All']], "
            f"order: [[{order_col}, 'desc']], "
            f"orderCellsTop: true, "
            f"scrollX: true, "
            f"scrollY: '360px', "
            f"scrollCollapse: true, "
            f"searchBuilder: {{ depthLimit: 2 }} "
            f"}});"
        )
    init_js = "\n".join(init_js_lines)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Mol 1 — Live Method-by-Method Report</title>
<link rel="stylesheet" href="https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/4.0.1/css/fixedHeader.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/searchbuilder/1.7.1/css/searchBuilder.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/datetime/1.5.3/css/dataTables.dateTime.min.css">
<style>{css}</style>
</head><body>
<h1>Mol 1 — Live Method-by-Method Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Seed: <code>{MOL1_SMILES}</code></p>
<p style="font-size:13px;"><b>Note</b>: every pIC50 reported below is the 3-seed FiLMDelta ensemble <b>mean ± std</b>,
computed identically for every candidate. <strong>No penalty is applied to any prediction.</strong> The ± term is
just the seed disagreement so you can see prediction stability per candidate. Rankings are by the ensemble mean.</p>

<div class="seed-box">{svg_of(MOL1_SMILES, 320, 220)}</div>

<h2>Column Legend</h2>
<p class="footnote">Every pIC50 value below is computed identically: each candidate is scored with all 3 FiLMDelta ensemble seeds,
each seed is anchor-averaged over the 280 ZAP70 training molecules to give a single pIC50, and the table reports
<code>mean ± std</code> across those 3 seed-anchor-means. <strong>No penalty is applied to any prediction</strong> —
the ± term is just the model's own ensemble disagreement so you can see how much the seeds vary on a per-candidate basis.</p>

<table id="legend-table" class="dt-table" style="font-size:12px;">
<thead><tr>
<th style="text-align:left;width:170px;">Column</th>
<th style="text-align:left;">What it is</th>
<th style="text-align:left;width:220px;">Range / interpretation</th>
</tr></thead>
<tbody>
<tr><td><b>pIC50 (mean ± std)</b></td>
    <td>For each candidate: compute three anchor-mean pIC50s (one per ensemble seed: <code>(1/280) Σ_i [P_i + FiLMDelta_seed_k(A_i, candidate)]</code> for k = 0, 1, 2).
        Report <b>mean</b> and <b>std</b> across those three values. Identical computation for every candidate; nothing method-specific.</td>
    <td><b>mean</b>: higher = more potent. <b>±std</b>: lower = ensemble agrees more.
        Sort table by mean.</td></tr>
<tr><td><b>Δ vs Mol1</b></td>
    <td>Candidate's ensemble mean pIC50 minus Mol 1's ensemble mean pIC50 (baseline computed at the start of the report from the same 3-seed ensemble).
        Positive = candidate predicted more potent than Mol 1.</td>
    <td>Typical range −2 to +2 in our libraries.</td></tr>
<tr><td><b>direct δ from Mol1</b></td>
    <td>Direct prediction averaged across 3 seeds: <code>mean_k FiLMDelta_seed_k(Mol 1, candidate)</code>. Single anchor (Mol 1 itself), not anchor-averaged.</td>
    <td>Positive = candidate &gt; Mol 1. Noisier than Δ-vs-Mol1 because of single-anchor estimation.</td></tr>
<tr><td><b>wins /280</b></td>
    <td>Average across 3 seeds of: count of training anchors where the candidate is predicted more potent than the anchor (i.e. <code>δ &gt; 0</code>).</td>
    <td>0–280. Higher = better than more of training. ≥260 suggests broad superiority claim.</td></tr>
<tr><td><b>wins ≥7</b></td>
    <td>Average across 3 seeds restricted to HIGH-POTENCY anchors only (measured pIC50 ≥ 7.0) — the harder bar. Reported as <code>n_won / n_high_potency_anchors</code>.</td>
    <td>For ZAP70 there are ~30 anchors at pIC50 ≥ 7. High count = candidate predicted to outperform actual potent compounds.</td></tr>
<tr><td><b>SAS</b></td>
    <td>Ertl &amp; Schuffenhauer synthetic accessibility score.</td>
    <td>1 (very easy) → 10 (impossible). Most drug-like candidates land at 2–4.</td></tr>
<tr><td><b>MW</b></td>
    <td>Molecular weight (Da).</td>
    <td>Drug-like &lt; 500. We allowed up to ~700 in Tier 2 SCALED.</td></tr>
<tr><td><b>QED</b></td>
    <td>Quantitative Estimate of Drug-likeness (Bickerton 2012).</td>
    <td>0 (not drug-like) → 1 (very drug-like). Useful filter; 0.5+ generally good.</td></tr>
<tr><td><b>shape Tc vs seed</b></td>
    <td>RDKit O3A-aligned Gaussian-volume shape Tanimoto vs Mol 1, computed over a 4-conformer ensemble (best alignment kept).</td>
    <td>0 (different shape) → 1 (identical). Above 0.7 = strong shape preservation.</td></tr>
<tr><td><b>ESP-Sim vs seed</b></td>
    <td>Espsim Gasteiger-charge electrostatic Tanimoto vs Mol 1.</td>
    <td>−1 to +1. Captures whether polar regions / charge distribution match.</td></tr>
<tr><td><b>warhead Δ°</b></td>
    <td>MIN angle (degrees, over the conformer ensemble after MCS-aligning) between the seed's C=C–C(=O)–N warhead vector and the candidate's.</td>
    <td>0° = warhead points the same way as in seed (great for covalent reach to Cys).
        &gt;90° = warhead pointing away. NaN if warhead missing.</td></tr>
<tr><td><b>Tc→Mol1</b></td>
    <td>Morgan FP (r=2, 2048-bit) Tanimoto similarity to Mol 1.</td>
    <td>0 (unrelated) → 1 (identical). 0.4+ = clear analog.</td></tr>
<tr><td><b>max Tc→train</b></td>
    <td>Morgan FP Tanimoto to the closest of the 280 ZAP70 training molecules. Standard out-of-distribution / novelty measure.</td>
    <td>≥ 0.5 = within model's training distribution; &lt; 0.3 = OOD extrapolation regime where FiLMDelta MAE ~0.86.</td></tr>
<tr><td><b>WH</b></td>
    <td>Boolean — does the candidate contain the acrylamide warhead substructure (SMARTS <code>[CH2]=[CH]C(=O)[N;!H2]</code>)?</td>
    <td>✓ = warhead intact. — = warhead missing or modified.</td></tr>
</tbody></table>

<p class="footnote" style="background:#fef3c7;padding:8px 12px;border-left:3px solid #f59e0b;">
<b>How to use the tables</b>:
<br>· <b>Sort</b>: click any column header.
<br>· <b>Numeric/value filter</b>: click the <b>"Add condition"</b> button (top-left of each table; SearchBuilder UI). Pick a column → operator (<code>&lt;</code>, <code>&gt;</code>, <code>=</code>, <code>≠</code>, <code>between</code>) → value. Example: <code>MW &lt; 400</code>, or <code>SAS &lt; 4 AND wins ≥7 &gt; 10</code>. Multiple conditions can be combined with AND/OR.
<br>· <b>Free-text search</b>: top-right search box (matches any cell text, e.g. SMILES substring).
<br>· <b>Per-table scroll</b>: each table has its own vertical scrollbar (600px tall). Headers stay visible while you scroll.
<br>· <b>Page size</b>: top-left dropdown (10/20/50/100/All). Default 20.
<br>· When you filter, the visible top-K becomes the top-K matching the filter (DataTables re-paginates filtered results sorted by current sort column).
</p>

<p class="footnote" style="background:#fef3c7;padding:8px 12px;border-left:3px solid #f59e0b;">
<b>Sortable / filterable</b>: click any column header to sort. Use the search box at top-right of each table to filter (matches across all columns).
Use the column visibility / page-length dropdowns to expand/collapse. The default page is top-20; click "All" in the dropdown to see all 50 (or 100 for global) with scrolling.
</p>

{''.join(sections)}

<!-- Modal for enlarged molecule view -->
<div id="mol-modal" class="mol-modal-bg" onclick="if(event.target===this) closeMolModal();">
  <div class="mol-modal">
    <button class="close" onclick="closeMolModal()" title="Close">&times;</button>
    <div id="mol-modal-svg"></div>
    <div id="mol-modal-smiles" class="smiles-box"></div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.1.8/js/dataTables.min.js"></script>
<script src="https://cdn.datatables.net/fixedheader/4.0.1/js/dataTables.fixedHeader.min.js"></script>
<script src="https://cdn.datatables.net/datetime/1.5.3/js/dataTables.dateTime.min.js"></script>
<script src="https://cdn.datatables.net/searchbuilder/1.7.1/js/dataTables.searchBuilder.min.js"></script>
<script>
function showMolModal(thumb) {{
  // Clone the thumb's SVG into the modal at a larger size (SVG is vector → scales cleanly)
  var svgEl = thumb.querySelector('svg');
  if (!svgEl) return;
  var clone = svgEl.cloneNode(true);
  // Strip explicit width/height so CSS .mol-modal svg can size it
  clone.removeAttribute('width');
  clone.removeAttribute('height');
  document.getElementById('mol-modal-svg').innerHTML = '';
  document.getElementById('mol-modal-svg').appendChild(clone);
  document.getElementById('mol-modal-smiles').textContent = thumb.getAttribute('data-smiles') || '';
  document.getElementById('mol-modal').classList.add('show');
}}
function closeMolModal() {{
  document.getElementById('mol-modal').classList.remove('show');
}}
// Close on Escape
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeMolModal();
}});
$(document).ready(function() {{
{init_js}
}});
</script>
</body></html>
"""
    OUT_HTML.write_text(html)
    log(f"\nReport → {OUT_HTML}")
    log(f"Size: {OUT_HTML.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
