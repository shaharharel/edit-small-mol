#!/usr/bin/env python3
"""RLHF preference-collection server (ZAP70 demo).

A lightweight Flask app where a logged-in medicinal chemist is shown two similar
same-lab ZAP70 molecules and picks the one they prefer for lead optimization.
Every judgment is attributed to a named chemist and stored in SQLite.

Reuses the project's proven viz patterns: RDKit -> SVG for 2D, 3Dmol.js + Boltz
CIF for the 3D pocket pose.

Run:  conda run -n quris python experiments/rlhf_server/app.py
      open http://localhost:5055
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from functools import wraps

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdFMCS

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data/rlhf_demo"
DB_PATH = DATA_DIR / "rlhf.db"

# Hide numeric pIC50 during the choice so we capture genuine medchem preference
# rather than "pick the bigger number". The values are still stored with every
# pair for later analysis.
SHOW_PIC50 = False

app = Flask(__name__)
app.secret_key = "rlhf-zap70-demo-secret"  # internal demo only

# ---- in-memory dataset -----------------------------------------------------
MOLECULES: dict[str, dict] = json.loads((DATA_DIR / "molecules.json").read_text())
PAIRS: list[dict] = json.loads((DATA_DIR / "pairs.json").read_text())
PAIRS.sort(key=lambda p: p["order"])
PAIR_BY_ID = {p["pair_id"]: p for p in PAIRS}
N_PAIRS = len(PAIRS)


# ---- database --------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        # timeout lets concurrent writers wait instead of erroring with "database is locked"
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute("PRAGMA journal_mode=WAL")  # concurrent readers + a single writer
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                email      TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS judgments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                pair_id       TEXT NOT NULL,
                mol_high_id   TEXT NOT NULL,
                mol_low_id    TEXT NOT NULL,
                chosen_id     TEXT,           -- NULL = explicit "no preference"/skip-with-reason
                shown_left_id TEXT NOT NULL,
                action        TEXT NOT NULL DEFAULT 'choice',  -- 'choice' | 'skip'
                ts            TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, pair_id)
            );
            """
        )
        db.commit()


# ---- auth ------------------------------------------------------------------
def current_user() -> sqlite3.Row | None:
    uid = session.get("uid")
    if uid is None:
        return None
    return get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            return jsonify({"error": "not_logged_in"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        if not name or not email:
            return render_template("login.html", error="Name and email are required.")
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row is None:
            cur = db.execute(
                "INSERT INTO users(name, email) VALUES(?, ?)", (name, email)
            )
            db.commit()
            uid = cur.lastrowid
        else:
            uid = row["id"]
        session["uid"] = uid
        session["name"] = name
        return redirect(url_for("index"))
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    return render_template("compare.html", name=user["name"], total=N_PAIRS)


# ---- pair serving ----------------------------------------------------------
def _judged_pair_ids(uid: int) -> set[str]:
    rows = get_db().execute(
        "SELECT pair_id FROM judgments WHERE user_id=?", (uid,)
    ).fetchall()
    return {r["pair_id"] for r in rows}


def _side_for(uid: int, pair_id: str) -> str:
    """Deterministically randomize left/right per (user, pair) so a reload is stable
    but position bias is balanced across pairs."""
    h = hashlib.md5(f"{uid}:{pair_id}".encode()).hexdigest()
    return "high" if int(h, 16) % 2 == 0 else "low"


def _mol_public(mol_id: str, pair: dict, which: str) -> dict:
    m = MOLECULES[mol_id]
    out = {
        "id": mol_id,
        "chembl_id": m["chembl_id"],
        "has_pose": bool(m.get("pose")),
        "conf": m.get("conf", {}),
    }
    if SHOW_PIC50:
        out["pIC50"] = pair["pIC50_high"] if which == "high" else pair["pIC50_low"]
    return out


@app.route("/api/next_pair")
def api_next_pair():
    user = current_user()
    if user is None:
        return jsonify({"error": "not_logged_in"}), 401
    uid = user["id"]
    judged = _judged_pair_ids(uid)
    nxt = next((p for p in PAIRS if p["pair_id"] not in judged), None)
    done = len(judged)
    if nxt is None:
        return jsonify({"done": True, "judged": done, "total": N_PAIRS})

    left_which = _side_for(uid, nxt["pair_id"])
    high_id, low_id = nxt["mol_high_id"], nxt["mol_low_id"]
    left_id = high_id if left_which == "high" else low_id
    right_id = low_id if left_which == "high" else high_id
    return jsonify(
        {
            "done": False,
            "judged": done,
            "total": N_PAIRS,
            "pair_id": nxt["pair_id"],
            "cohort": nxt["assay_chembl_id"],
            "shown_left_id": left_id,
            "left": _mol_public(left_id, nxt, left_which),
            "right": _mol_public(
                right_id, nxt, "low" if left_which == "high" else "high"
            ),
        }
    )


_svg_cache: dict = {}


def _clamp_dim(v, default):
    try:
        return max(80, min(1200, int(v)))
    except (TypeError, ValueError):
        return default


def _aligned_coords(mol, ref_mol) -> None:
    """Orient `mol` so its maximum common substructure with `ref_mol` shares the
    same 2D layout — makes the matched pair visually comparable side by side."""
    try:
        res = rdFMCS.FindMCS(
            [mol, ref_mol], timeout=2, completeRingsOnly=True,
            ringMatchesRingOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        )
        patt = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
        if patt is None:
            raise ValueError("no MCS")
        ref_match = ref_mol.GetSubstructMatch(patt)
        mol_match = mol.GetSubstructMatch(patt)
        if not ref_match or not mol_match:
            raise ValueError("no match")
        conf = ref_mol.GetConformer()
        coord_map = {mol_match[i]: conf.GetAtomPosition(ref_match[i]) for i in range(len(mol_match))}
        AllChem.Compute2DCoords(mol, coordMap=coord_map)
    except Exception:
        AllChem.Compute2DCoords(mol)


@app.route("/api/svg/<mol_id>")
@login_required
def api_svg(mol_id: str):
    m = MOLECULES.get(mol_id)
    if m is None:
        return "not found", 404
    w = _clamp_dim(request.args.get("w"), 360)
    h = _clamp_dim(request.args.get("h"), 300)
    ref_id = request.args.get("ref")
    cache_key = (mol_id, ref_id, w, h)
    if cache_key in _svg_cache:
        return _svg_cache[cache_key], 200, {"Content-Type": "image/svg+xml"}

    mol = Chem.MolFromSmiles(m["smiles"])
    if mol is None:
        return "bad smiles", 400
    ref = MOLECULES.get(ref_id) if ref_id else None
    if ref:
        ref_mol = Chem.MolFromSmiles(ref["smiles"])
        if ref_mol is not None:
            AllChem.Compute2DCoords(ref_mol)
            _aligned_coords(mol, ref_mol)
        else:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    drawer = Draw.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.6
    opts.clearBackground = False
    opts.padding = 0.06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    _svg_cache[cache_key] = svg
    return svg, 200, {"Content-Type": "image/svg+xml"}


_cif_cache: dict = {}


@app.route("/api/pose/<mol_id>")
@login_required
def api_pose(mol_id: str):
    m = MOLECULES.get(mol_id)
    if m is None:
        return jsonify({"available": False}), 404
    pose = m.get("pose")
    if not pose:
        return jsonify({"available": False, "pending": True})
    path = PROJECT / pose
    if not path.exists():
        return jsonify({"available": False, "pending": True})
    if mol_id not in _cif_cache:
        _cif_cache[mol_id] = path.read_text()
    return jsonify(
        {"available": True, "cif": _cif_cache[mol_id], "conf": m.get("conf", {})}
    )


@app.route("/api/submit", methods=["POST"])
def api_submit():
    user = current_user()
    if user is None:
        return jsonify({"error": "not_logged_in"}), 401
    data = request.get_json(silent=True) or {}
    pair_id = data.get("pair_id")
    chosen_id = data.get("chosen_id")  # may be None for skip
    action = data.get("action", "choice")
    if action not in ("choice", "skip"):
        return jsonify({"error": "bad_action"}), 400
    pair = PAIR_BY_ID.get(pair_id)
    if pair is None:
        return jsonify({"error": "bad_pair"}), 400
    if action == "choice" and chosen_id not in (pair["mol_high_id"], pair["mol_low_id"]):
        return jsonify({"error": "bad_choice"}), 400
    uid = user["id"]
    shown_left = _side_for(uid, pair_id)
    shown_left_id = pair["mol_high_id"] if shown_left == "high" else pair["mol_low_id"]
    db = get_db()
    # latest choice wins (re-submit updates rather than being silently dropped)
    cur = db.execute(
        """INSERT INTO judgments
           (user_id, pair_id, mol_high_id, mol_low_id, chosen_id, shown_left_id, action)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(user_id, pair_id) DO UPDATE SET
             chosen_id=excluded.chosen_id, shown_left_id=excluded.shown_left_id,
             action=excluded.action, ts=datetime('now')""",
        (
            uid,
            pair_id,
            pair["mol_high_id"],
            pair["mol_low_id"],
            chosen_id if action == "choice" else None,
            shown_left_id,
            action,
        ),
    )
    db.commit()
    return jsonify({"ok": True, "stored": cur.rowcount})


@app.route("/api/undo", methods=["POST"])
@login_required
def api_undo():
    """Remove the user's most recent judgment so they can re-judge that pair."""
    user = current_user()
    db = get_db()
    row = db.execute(
        "SELECT id, pair_id FROM judgments WHERE user_id=? ORDER BY id DESC LIMIT 1",
        (user["id"],),
    ).fetchone()
    if row is None:
        return jsonify({"ok": False, "empty": True})
    db.execute("DELETE FROM judgments WHERE id=?", (row["id"],))
    db.commit()
    return jsonify({"ok": True, "pair_id": row["pair_id"]})


@app.route("/api/progress")
def api_progress():
    user = current_user()
    if user is None:
        return jsonify({"error": "not_logged_in"}), 401
    return jsonify({"judged": len(_judged_pair_ids(user["id"])), "total": N_PAIRS})


if __name__ == "__main__":
    init_db()
    print(f"RLHF server: {N_PAIRS} pairs, {len(MOLECULES)} molecules")
    print("open http://localhost:5055")
    app.run(host="0.0.0.0", port=5055, debug=False)
