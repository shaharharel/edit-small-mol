#!/usr/bin/env python3
"""
Flask backend for live filtering of 498K Tier 2 SCALED candidates.

Loads products_scored_full.csv into memory at startup, serves DataTables
server-side AJAX requests with SearchBuilder filter conditions parsed into
pandas queries. Plus quick chip presets, structure-on-demand SVG rendering.

Run:
    cd ~/Documents/github/edit-small-mol
    conda run -n quris python experiments/server/backend.py
    # listens on http://localhost:5000

Frontend HTML lives at: results/paper_evaluation/overnight_method_report_live.html
                        (Tier 2 SCALED section uses serverSide: true with this backend)
"""

import json
import sys
from pathlib import Path
from io import BytesIO
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog('rdApp.*')

DATA_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v3.csv"
LEGACY_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "aichem_tier2_scaled" / "products_scored_full.csv"

app = Flask(__name__)
CORS(app)

print(f"Loading {DATA_FILE} ...")
if DATA_FILE.exists():
    DF = pd.read_csv(DATA_FILE)
else:
    print(f"  Falling back to {LEGACY_FILE}")
    DF = pd.read_csv(LEGACY_FILE)
    DF["method"] = "Tier 2 SCALED — Fragment Replacement (498K)"
    if "pIC50_film" in DF.columns and "pIC50_method" not in DF.columns:
        DF["pIC50_method"] = DF["pIC50_film"]
print(f"Loaded {len(DF):,} candidates with columns: {list(DF.columns)}")
if "method" in DF.columns:
    print(f"Methods: {DF['method'].value_counts().to_dict()}")
DF = DF.reset_index(drop=True)
DF["row_id"] = DF.index
# ── Method unification + tier grouping (per user 2026-05-04 spec) ──
#  Tier 1: "Medchem Rules"  ← Tier 1 + Tier 1.5 unified
#  Tier 2: "Amine Replacements"  ← Tier 2 + Tier 2 SCALED unified
#  Tier 3: "Constrained Generative"  ← LibInvent locked / Mol2Mol+wh / De Novo+wh / v2 (kept as sub-methods)
#  Tier 4: removed for now (Tier 4 De Novo / Tier 4 Mol2Mol / Method A / Method B filtered out)
if "method" in DF.columns:
    DF["method"] = DF["method"].replace({
        "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
        "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
        "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
        # Tier 3 v2 + Tier 3 v3 sub-methods kept as-is, will group in frontend
    })
    # Drop Tier 4 + Methods A/B (generate-and-filter family) for now
    DROPPED = [
        "Tier 4 — De Novo unconstrained",
        "Tier 4 — Mol2Mol unconstrained",
        "Method A — De Novo FiLMDelta-driven",
        "Method B — Mol2Mol FiLMDelta-driven",
    ]
    pre = len(DF)
    DF = DF[~DF["method"].isin(DROPPED)].reset_index(drop=True)
    print(f"  Filtered out Tier 4 / Methods A/B: {pre:,} -> {len(DF):,}")
    # Drop warhead-modifying controls — every reported candidate must have warhead intact
    if "warhead_intact" in DF.columns:
        pre = len(DF)
        DF = DF[DF["warhead_intact"] == True].reset_index(drop=True)
        print(f"  Filtered out warhead-MODIFIED candidates: {pre:,} -> {len(DF):,}")
    # Drop disconnected-fragment SMILES (Tier 1 ReplaceSubstructs sometimes produces
    # mol1.fragment2 SMILES where a bond was inadvertently broken; these pass the warhead
    # SMARTS via the first fragment but aren't real candidates).
    if "smiles" in DF.columns:
        pre = len(DF)
        DF = DF[~DF["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
        print(f"  Filtered out disconnected (multi-fragment) SMILES: {pre:,} -> {len(DF):,}")
    DF["row_id"] = DF.index
    print(f"  Methods after unification: {sorted(DF['method'].unique().tolist())}")
# Use pIC50_method as the primary pIC50 (uniform across methods)
if "pIC50_method" in DF.columns and "pIC50_film" not in DF.columns:
    DF["pIC50_film"] = DF["pIC50_method"]
elif "pIC50_film" not in DF.columns and "pIC50_method" in DF.columns:
    DF["pIC50_film"] = DF["pIC50_method"]

NUMERIC_COLS = [c for c in DF.columns if pd.api.types.is_numeric_dtype(DF[c])]
print(f"Numeric columns: {NUMERIC_COLS}")


# ── SearchBuilder condition parser ───────────────────────────────────────────

def apply_searchbuilder(df: pd.DataFrame, sb_payload: dict) -> pd.DataFrame:
    """Apply SearchBuilder JSON conditions to a pandas DataFrame.

    Payload structure (from DataTables SearchBuilder):
      {
        "criteria": [
          {"data": "MW", "condition": "<", "value": ["400"]},
          {"data": "QED", "condition": ">", "value": ["0.5"]},
          ...
        ],
        "logic": "AND"
      }
    """
    if not sb_payload:
        return df
    criteria = sb_payload.get("criteria", [])
    logic = sb_payload.get("logic", "AND").upper()
    if not criteria:
        return df

    masks = []
    for c in criteria:
        col = c.get("data") or c.get("origData")
        cond = c.get("condition")
        vals = c.get("value", [])
        if col not in df.columns:
            continue
        try:
            if cond in ("<", "<="):
                mask = df[col] <= float(vals[0]) if cond == "<=" else df[col] < float(vals[0])
            elif cond in (">", ">="):
                mask = df[col] >= float(vals[0]) if cond == ">=" else df[col] > float(vals[0])
            elif cond in ("=", "==", "equals"):
                # Try numeric first, fall back to string
                try:
                    mask = df[col] == float(vals[0])
                except (ValueError, TypeError):
                    mask = df[col].astype(str) == str(vals[0])
            elif cond in ("!=", "≠", "not"):
                try:
                    mask = df[col] != float(vals[0])
                except (ValueError, TypeError):
                    mask = df[col].astype(str) != str(vals[0])
            elif cond == "between":
                lo, hi = float(vals[0]), float(vals[1])
                mask = (df[col] >= lo) & (df[col] <= hi)
            elif cond in ("starts", "starts with"):
                mask = df[col].astype(str).str.startswith(str(vals[0]))
            elif cond in ("ends", "ends with"):
                mask = df[col].astype(str).str.endswith(str(vals[0]))
            elif cond in ("contains",):
                mask = df[col].astype(str).str.contains(str(vals[0]), na=False, regex=False)
            elif cond in ("null", "isnull"):
                mask = df[col].isna()
            elif cond in ("notnull",):
                mask = df[col].notna()
            else:
                continue
        except Exception as e:
            print(f"  SB error on {col}/{cond}/{vals}: {e}")
            continue
        masks.append(mask)

    if not masks:
        return df
    if logic == "AND":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
    return df[combined]


# ── Quick filter chips ──────────────────────────────────────────────────────

CHIP_PRESETS = {
    "lipinski": lambda d: d[(d["MW"] < 500) & (d["LogP"] < 5) & (d["HBD"] <= 5) & (d["HBA"] <= 10)],
    "leadlike": lambda d: d[(d["MW"] < 350) & (d["SAScore"] < 4) & (d["QED"] > 0.5)],
    "potent": lambda d: d[d["pIC50_film"] >= 7.0],
    "ultra_potent": lambda d: d[d["pIC50_film"] >= 8.0],
    "in_distribution": lambda d: d[d["max_Tc_train"] >= 0.3],
    "warhead_intact": lambda d: d[d["warhead_intact"] == True],
    "pains_clean": lambda d: d[d["PAINS_alerts"] == 0],
    "synthesizable": lambda d: d[d["SAScore"] < 4.0],
    "drug_like": lambda d: d[d["QED"] >= 0.5],
}


@app.route("/api/data", methods=["POST", "GET"])
def api_data():
    """DataTables server-side endpoint."""
    if request.method == "POST":
        payload = request.get_json() or request.form.to_dict() or {}
    else:
        payload = request.args.to_dict()
    # Standard DataTables params
    draw = int(payload.get("draw", 1))
    start = int(payload.get("start", 0))
    length = int(payload.get("length", 20))
    if length < 0: length = len(DF)

    df = DF
    # Method filter (selects one method)
    method_filter = payload.get("method", "")
    if method_filter and method_filter != "_all" and "method" in df.columns:
        df = df[df["method"] == method_filter]
    n_total = len(df)

    # Apply quick chips
    chips_str = payload.get("chips", "")
    if chips_str:
        chips = [c.strip() for c in chips_str.split(",") if c.strip()]
        for chip in chips:
            if chip in CHIP_PRESETS:
                df = CHIP_PRESETS[chip](df)

    # Apply SearchBuilder
    sb_str = payload.get("searchBuilder", "") or payload.get("sb", "")
    if sb_str:
        try:
            sb = json.loads(sb_str) if isinstance(sb_str, str) else sb_str
            df = apply_searchbuilder(df, sb)
        except Exception as e:
            print(f"SB parse error: {e}")

    # Free-text search across SMILES
    text = payload.get("search", "") or payload.get("q", "")
    if text:
        df = df[df["smiles"].astype(str).str.contains(text, case=False, na=False)]

    n_filtered = len(df)

    # Sorting
    order_col = payload.get("order_col", "pIC50_film")
    order_dir = payload.get("order_dir", "desc")
    if order_col in df.columns:
        df = df.sort_values(order_col, ascending=(order_dir == "asc"), na_position="last")

    # Pagination
    df_page = df.iloc[start:start + length]

    # Build response
    rows = df_page.to_dict(orient="records")
    return jsonify({
        "draw": draw,
        "recordsTotal": n_total,
        "recordsFiltered": n_filtered,
        "data": rows,
    })


@app.route("/api/svg/<int:row_id>")
def api_svg(row_id: int):
    """Render a small SVG for one row's molecule."""
    if row_id < 0 or row_id >= len(DF):
        return "", 404
    smi = DF.iloc[row_id]["smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "", 404
    AllChem.Compute2DCoords(mol)
    w = int(request.args.get("w", 140))
    h = int(request.args.get("h", 100))
    drawer = Draw.MolDraw2DSVG(w, h)
    drawer.drawOptions().bondLineWidth = 1.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    return svg, 200, {"Content-Type": "image/svg+xml"}


MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

@app.route("/api/seed_svg")
def api_seed_svg():
    w = int(request.args.get("w", 200))
    h = int(request.args.get("h", 140))
    mol = Chem.MolFromSmiles(MOL1_SMILES)
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(w, h)
    drawer.drawOptions().bondLineWidth = 1.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    return svg, 200, {"Content-Type": "image/svg+xml"}


@app.route("/api/methods")
def api_methods():
    """List methods + their candidate counts + per-method stats."""
    if "method" not in DF.columns:
        return jsonify({"methods": [{"name": "All", "n": len(DF)}]})
    rows = []
    for m, sub in DF.groupby("method"):
        pic = sub["pIC50_film"].dropna() if "pIC50_film" in sub.columns else pd.Series(dtype=float)
        rows.append({
            "name": m,
            "n": int(len(sub)),
            "max_pIC50": float(pic.max()) if len(pic) else None,
            "median_pIC50": float(pic.median()) if len(pic) else None,
            "n_potent_7": int((pic >= 7).sum()),
            "n_potent_8": int((pic >= 8).sum()),
            "n_warhead": int(sub["warhead_intact"].sum()) if "warhead_intact" in sub.columns else None,
        })
    return jsonify({"methods": rows, "total": len(DF)})


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "n_rows": len(DF), "columns": list(DF.columns),
                    "methods": list(DF["method"].unique()) if "method" in DF.columns else []})


@app.route("/")
def index():
    """Serve the report HTML directly so a single Flask process handles everything."""
    html_path = Path(__file__).parent / "report.html"
    if html_path.exists():
        return send_file(html_path)
    return jsonify({"endpoints": ["/api/data", "/api/svg/<row_id>", "/api/health"]})


if __name__ == "__main__":
    import os
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", "5001"))
    print(f"Listening on {host}:{port}")
    app.run(host=host, port=port, debug=False)
