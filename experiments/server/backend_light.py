#!/usr/bin/env python3
"""Slim Flask backend serving ONLY the /light view + its endpoints.

Why: the heavy backend (port 5001) loads the full 1M-row cohort + serves all
report-page routes, which under SVG-flood load can exceed 16 GB RSS and trigger
the kernel OOM killer. This slim backend loads ONLY:
  - the 838 visible-survivor rows (from a pre-baked JSON snapshot)
  - the 78 rescue rows (from rescue_78_full.csv)
  - Mol1 anchor features
  - the 838-CIF bundle for 3D poses
  - the Mol1 seed CIF for the modal 3D tab

Routes exposed:
  /light                — serves report_light.html
  /api/filter           — returns the 838 default-precompute response
  /api/mol1_anchor      — Mol1 reference row
  /api/svg/<row_id>     — RDKit-rendered SVG (with explicit LRU cache to bound mem)
  /api/seed_svg         — Mol1 seed SVG
  /api/pose/<row_id>    — Boltz CIF + metadata for visible-838 row
  /api/pose_seed        — Mol1 seed pose

Designed to stay under 2 GB RSS even under load.

Run: python experiments/server/backend_light.py --port 5002
"""

import argparse
import gzip
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")
try:
    from flask_compress import Compress
    HAS_COMPRESS = True
except ImportError:
    HAS_COMPRESS = False

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "tier4_scored"
F4_FULL   = DATA_DIR / "F4_boltz_full.csv"
RESCUE_78 = DATA_DIR / "rescue_78_full.csv"
PREXSYN_CSV = DATA_DIR / "prexsyn_916.csv"
CIF_BUNDLE = DATA_DIR / "visible838_cifs_v2.json.gz"  # SMILES-keyed (838/838 coverage)
CIF_BUNDLE_LEGACY = DATA_DIR / "visible838_cifs.json.gz"  # old row_id-keyed fallback
MOL1_FEATURES = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_full_features.json"

REPORT_LIGHT_HTML = PROJECT_ROOT / "experiments" / "server" / "report_light.html"

# Pre-computed default visible-838 snapshot (built at startup from the same
# logic the heavy backend uses, but evaluated here without the 1M-row DF).
# We just re-use the heavy backend's already-baked F4_boltz_full.csv as the
# source of truth for the 838 rows (those are the only rows the snapshot
# actually contains anyway after the filter cascade resolves).

# ── App init ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
if HAS_COMPRESS:
    app.config["COMPRESS_MIMETYPES"] = ["application/json", "text/html", "text/css", "application/javascript"]
    app.config["COMPRESS_LEVEL"] = 6
    app.config["COMPRESS_MIN_SIZE"] = 500
    Compress(app)
    print("Flask-Compress enabled")

# ── Load data once at startup ──────────────────────────────────────────────
print(f"[init] loading F4_boltz_full from {F4_FULL.name}")
f4 = pd.read_csv(F4_FULL, low_memory=False)
print(f"  loaded {len(f4):,} rows × {len(f4.columns)} cols")

# Replicate the strict filter pipeline that yields the 838 visible survivors.
# We import the heavy-backend's RESCUE_CRITERIA / cascade logic to stay
# identical — but only run it on F4 (small enough to be cheap in slim backend).
# Simpler: pull row_ids from the heavy backend's snapshot if available;
# otherwise just expose all F4 rows. We pick the latter — keeps slim backend
# truly independent.

# Build per-row dicts for fast lookup
print(f"[init] building lookups for SVG / pose endpoints")
# F4_BY_ROWID + SMILES_BY_ROWID populated below (after we subset to the 838).

# Mol1 anchor features
MOL1_ROW = {}
if MOL1_FEATURES.exists():
    try:
        MOL1_ROW = json.loads(MOL1_FEATURES.read_text())
        print(f"  mol1 features: {len(MOL1_ROW)} keys")
    except Exception as e:
        print(f"  [warn] mol1 features load failed: {e}")

# 838 visible CIF bundle (for 3D pose tab)
VISIBLE838_CIFS = {}
if CIF_BUNDLE.exists():
    try:
        with gzip.open(CIF_BUNDLE, "rt") as f:
            VISIBLE838_CIFS = json.load(f)
        print(f"  visible-838 CIFs: {len(VISIBLE838_CIFS):,} row_id → CIF text")
    except Exception as e:
        print(f"  [warn] CIF bundle load failed: {e}")

# Rescue 78 (optional — for column completeness if /light wants both cohorts later)
RESCUE_BY_ROWID = {}
RESCUE_ROWS_FOR_FILTER = []  # rescue rows in same dict-shape as ROWS_FOR_FILTER
if RESCUE_78.exists():
    try:
        r78 = pd.read_csv(RESCUE_78, low_memory=False)
        RESCUE_BY_ROWID = {r["rescue_row_id"]: r for _, r in r78.iterrows()}
        print(f"  rescue 78: {len(RESCUE_BY_ROWID)} rows")
    except Exception as e:
        print(f"  [warn] rescue load failed: {e}")

# ── Pre-bake the default-filter response so /api/filter is instant ─────────
print(f"[init] baking default-filter response")
# Drop NaN/Inf for JSON safety + convert numpy types
def _clean(v):
    if v is None: return None
    if isinstance(v, float):
        if pd.isna(v) or v != v: return None
        if v == float('inf') or v == float('-inf'): return None
        return float(v)
    if isinstance(v, (int, bool, str)): return v
    try: return v.item()  # numpy scalar
    except: return str(v)

# For the slim backend, the "default filter" returns all F4 rows that pass the
# Subset F4 to ONLY the 838 visible mols (canonical cascade output).
# Keyed by SMILES because F4 row_ids are cohort strings (e.g. "thiq_rl_kinase_27239")
# while the heavy backend assigns integer row_ids at serve time. SMILES is the universal key.
VISIBLE_838_SMIS_FILE = DATA_DIR / "visible_838_smiles.json"
if VISIBLE_838_SMIS_FILE.exists():
    VISIBLE_838_SMIS = set(json.loads(VISIBLE_838_SMIS_FILE.read_text()))
    print(f"  loaded {len(VISIBLE_838_SMIS)} canonical visible SMILES")
else:
    VISIBLE_838_SMIS = None
    print("  [warn] no visible_838_smiles.json — falling back to full F4 (2,221)")

# ── PrexSyn synthesizability lookup (keyed by canonical SMILES) ────────────
PREXSYN_FIELDS = [
    "prexsyn_route_found", "prexsyn_n_steps", "prexsyn_n_blocks",
    "prexsyn_score", "prexsyn_best_route_str",
]
PREXSYN_BY_CANON = {}

def _canon(smi):
    if not smi: return None
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None

if PREXSYN_CSV.exists():
    try:
        ps = pd.read_csv(PREXSYN_CSV, low_memory=False)
        for _, r in ps.iterrows():
            cs = _canon(r.get("smiles"))
            if not cs: continue
            PREXSYN_BY_CANON[cs] = {k: _clean(r.get(k)) for k in PREXSYN_FIELDS}
        print(f"  prexsyn: {len(PREXSYN_BY_CANON):,} canonical SMILES with routes")
    except Exception as e:
        print(f"  [warn] prexsyn load failed: {e}")
else:
    print(f"  [warn] {PREXSYN_CSV.name} not found")

def _attach_prexsyn(row_dict):
    cs = _canon(row_dict.get("smiles"))
    fields = PREXSYN_BY_CANON.get(cs) if cs else None
    for k in PREXSYN_FIELDS:
        row_dict[k] = (fields.get(k) if fields else None)
    return row_dict

# Short-name aliases the frontend uses (report_light.html COLS keys) → actual
# column names in F4_boltz_full.csv / rescue_78_full.csv. Filled at serve-time
# so both main 838 and rescue 78 surface these cofold/Boltz cells.
BOLTZ_ALIASES = {
    "iptm":             "boltz_iptm",
    "ligand_iptm":      "boltz_ligand_iptm",
    "boltz_plddt":      "boltz_complex_plddt",
    "boltz_confidence": "boltz_confidence_score",
    "mPAE":             "mPAE_paper",
}

def _attach_boltz_aliases(row_dict):
    for short, source in BOLTZ_ALIASES.items():
        if row_dict.get(short) in (None, "") and row_dict.get(source) not in (None, ""):
            row_dict[short] = row_dict[source]
    return row_dict

# Build a SMILES → integer-row_id map to mirror heavy backend's serving format
ROWS_FOR_FILTER = []
for serving_idx, (_, r) in enumerate(f4.iterrows()):
    smi = r.get("smiles")
    if VISIBLE_838_SMIS is not None and smi not in VISIBLE_838_SMIS:
        continue
    d = {k: _clean(v) for k, v in r.to_dict().items()}
    # Heavy backend uses integer row_ids in its served response; we'll mirror that
    # by overriding row_id with a stable integer derived from the SMILES position.
    d["_orig_row_id"] = d.get("row_id")
    d["row_id"] = serving_idx  # integer for SVG/pose endpoint compatibility
    _attach_prexsyn(d)
    _attach_boltz_aliases(d)
    ROWS_FOR_FILTER.append(d)

# Build the lookup tables now that row_ids are finalized
F4_BY_ROWID = {str(r["row_id"]): r for r in ROWS_FOR_FILTER}
SMILES_BY_ROWID = {str(r["row_id"]): r["smiles"] for r in ROWS_FOR_FILTER}
# Serving-int row_id (0..837) → ORIGINAL F4 row_id (e.g. 'thiq_rl_exp2_kinase_34597' or '40886')
# Needed for CIF bundle lookups (bundle keyed by original IDs).
ORIG_BY_SERVING_RID = {str(r["row_id"]): str(r.get("_orig_row_id")) for r in ROWS_FOR_FILTER}

# Build parallel rescue rows in same dict shape (row_id = rescue_row_id string like "R0000")
if RESCUE_BY_ROWID:
    for rid, r in RESCUE_BY_ROWID.items():
        d = {k: _clean(v) for k, v in r.to_dict().items()}
        d["row_id"] = rid  # keep rescue identifier as the row_id (string like "R0000")
        _attach_prexsyn(d)
        _attach_boltz_aliases(d)
        RESCUE_ROWS_FOR_FILTER.append(d)
    print(f"  rescue rows shaped for filter: {len(RESCUE_ROWS_FOR_FILTER)}")

# SMILES map for rescue (so /api/svg/R0000 works)
SMILES_BY_RESCUE_RID = {r["row_id"]: r["smiles"] for r in RESCUE_ROWS_FOR_FILTER}

DEFAULT_FILTER_PAYLOAD = {
    "rows": ROWS_FOR_FILTER,
    "rows_rescue": RESCUE_ROWS_FOR_FILTER,
    "n_visible": len(ROWS_FOR_FILTER),
    "n_rescue":  len(RESCUE_ROWS_FOR_FILTER),
    "counts": {"visible": len(ROWS_FOR_FILTER), "rescue": len(RESCUE_ROWS_FOR_FILTER), "total_unfiltered": len(ROWS_FOR_FILTER) + len(RESCUE_ROWS_FOR_FILTER)},
    "sort": "desirability_score:desc",
    "composite": True,
    "start": 0,
    "length": len(ROWS_FOR_FILTER),
}
print(f"  baked: {len(ROWS_FOR_FILTER)} main + {len(RESCUE_ROWS_FOR_FILTER)} rescue rows")

# ── Bounded LRU cache for SVG renders (caps memory growth under flood) ────
@lru_cache(maxsize=4096)
def _render_svg(smi: str, w: int, h: int) -> str:
    """RDKit-rendered SVG for a single SMILES. LRU-capped to bound memory."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="5" y="20">?</text></svg>'
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/light")
def light_page():
    if REPORT_LIGHT_HTML.exists():
        return send_file(REPORT_LIGHT_HTML)
    return jsonify({"error": "report_light.html not found"}), 404

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "backend_light", "rows": len(ROWS_FOR_FILTER)})

@app.route("/api/filter", methods=["GET", "POST"])
def api_filter():
    # Slim backend ignores filter params — always returns the baked default.
    # The /light frontend applies all filters client-side anyway.
    return jsonify(DEFAULT_FILTER_PAYLOAD)

@app.route("/api/spec")
def api_spec():
    # Minimal spec — frontend uses this to detect backend availability.
    return jsonify({"layers": {}, "backend": "light"})

@app.route("/api/methods")
def api_methods():
    # /light doesn't actually need methods, but the heavy report does.
    # Return empty to avoid 404s from any frontend probe.
    return jsonify({"methods": []})

@app.route("/api/mol1_anchor")
def api_mol1_anchor():
    return jsonify({"available": bool(MOL1_ROW), "row": MOL1_ROW})

@app.route("/api/svg/<row_id>")
def api_svg(row_id):
    from flask import request
    w = int(request.args.get("w", 110))
    h = int(request.args.get("h", 80))
    # Bound dimensions to prevent OOM-by-huge-image
    w = min(max(w, 20), 600); h = min(max(h, 20), 400)
    smi = None
    if row_id == "M1":
        smi = MOL1_ROW.get("smiles") if MOL1_ROW else None
    else:
        # Try main 838 first (integer-string row_ids)
        smi = SMILES_BY_ROWID.get(str(row_id))
        if smi is None:
            # Try rescue (rescue_row_id like 'R0000')
            smi = SMILES_BY_RESCUE_RID.get(str(row_id))
        if smi is None:
            # Last-resort fallback to raw rescue dict
            try:
                r = RESCUE_BY_ROWID.get(row_id, {}) if RESCUE_BY_ROWID else {}
                smi = r.get("smiles") if hasattr(r, "get") else None
            except Exception:
                smi = None
    if not smi:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="5" y="20">?</text></svg>', 200, {"Content-Type": "image/svg+xml"}
    svg = _render_svg(smi, w, h)
    return svg, 200, {"Content-Type": "image/svg+xml"}

@app.route("/api/svg_smi")
def api_svg_smi():
    from flask import request
    smi = request.args.get("smi", "").strip()
    w = int(request.args.get("w", 160)); h = int(request.args.get("h", 110))
    w = min(max(w, 20), 600); h = min(max(h, 20), 400)
    if not smi:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>', 200, {"Content-Type": "image/svg+xml"}
    return _render_svg(smi, w, h), 200, {"Content-Type": "image/svg+xml"}

@app.route("/api/seed_svg")
def api_seed_svg():
    from flask import request
    w = int(request.args.get("w", 480))
    h = int(request.args.get("h", 380))
    w = min(max(w, 20), 800); h = min(max(h, 20), 600)
    smi = MOL1_ROW.get("smiles") if MOL1_ROW else "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
    return _render_svg(smi, w, h), 200, {"Content-Type": "image/svg+xml"}

@app.route("/api/pose/<row_id>")
def api_pose(row_id):
    key = str(row_id)
    # Main 838: bundle is keyed by canonical SMILES (v2). Resolve served-int → SMILES → canonical.
    smi = SMILES_BY_ROWID.get(key)
    if smi:
        cs = _canon(smi)
        if cs and cs in VISIBLE838_CIFS:
            e = VISIBLE838_CIFS[cs]
            return jsonify({"available": True, "row_id": row_id, "cif": e["cif"],
                            "metadata": dict(e.get("metadata", {}), pose_source="visible838_v2")})
    # Rescue: read CIF from data/boltz_rescue_78/<RID>/<RID>_model_0.cif on demand
    if key.startswith("R") and key in RESCUE_BY_ROWID:
        cif_path = PROJECT_ROOT / "data" / "boltz_rescue_78" / key / f"{key}_model_0.cif"
        if cif_path.exists():
            return jsonify({"available": True, "row_id": row_id, "cif": cif_path.read_text(),
                            "metadata": {"pose_source": "boltz_rescue_78"}})
    return jsonify({"available": False, "row_id": row_id})

@app.route("/api/pose_seed")
def api_pose_seed():
    # Mol1 seed pose — look up by Mol1's row_id if present in bundle (it's row M1
    # in mol1_full_features but the bundle is keyed by visible-838 row_ids only).
    # The seed pose lives in the heavy backend's filesystem; slim backend doesn't
    # ship it. Return unavailable for now (modal will hide 3D tab for Mol1).
    return jsonify({"available": False})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()
    print(f"\n[backend_light] listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
