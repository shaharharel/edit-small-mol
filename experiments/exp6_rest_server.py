"""EXP6 retrospective LO — per-target REST scoring server.

One of TWO scorer variants per target:
  * scorer='film'  -> uses target-specific FiLMDelta + anchor pool
  * scorer='dabs'  -> uses target-specific DirectAbsoluteMLP

Composite reward = geometric_mean(pIC50_sigmoid, warhead_match, QED)
  weights: 0.50 / 0.40 / 0.10
  warhead_match: 0.5 (no match) / 1.0 (match)  [soft floor]
  geom_mean uses epsilon floor (default 0.05) so no input zeros the composite

This is the same reward contract as film_rest_server_v3.py, just parameterised
per-target.

Usage:
    CUDA_VISIBLE_DEVICES="" python exp6_rest_server.py \\
        --target_dir /home/shaharh_quris_ai/edit-small-mol/data/exp6_retrospective/egfr_t790m \\
        --src_root /home/shaharh_quris_ai/edit-small-mol \\
        --scorer film \\
        --port 8088
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED

RDLogger.DisableLog("rdApp.*")

SIGMOID_LOW = 5.5
SIGMOID_HIGH = 7.5
SIGMOID_K = 0.5
W_FILM = 0.50
W_WAR = 0.40
W_QED = 0.10
GM_EPS = 0.05

SERVER_VERSION = "exp6-v1"

_MODEL = None
_SCORER = None  # 'film' or 'dabs'
_ANCHOR_FPS = None
_ANCHOR_PICS = None
_SMARTS_STRICT_PAT = None
_SMARTS_GENERIC_PAT = None
_TARGET_LABEL = None
_N_REQ = 0
_N_MOLS = 0


class DirectAbsoluteMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _setup_film_import(src_root: str):
    sys.path.insert(0, src_root)


def _smi_to_fp(smi: str, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return mol, arr


def _double_sigmoid(x, low, high, k):
    import math
    if x is None or not np.isfinite(x):
        return 0.0
    mid = 0.5 * (low + high)
    span = (high - low) if high > low else 1.0
    z = (x - mid) / (span / 4.0)
    return float(1.0 / (1.0 + math.exp(-z * 4.0 * k)))


def _warhead_match(mol) -> float:
    """Soft-floor warhead score in {0.5, 1.0}.
    Uses STRICT smarts (target-specific) for the bonus.
    Falls back to GENERIC if strict missing."""
    if mol is None:
        return 0.5
    pat = _SMARTS_STRICT_PAT if _SMARTS_STRICT_PAT is not None else _SMARTS_GENERIC_PAT
    if pat is None:
        return 0.5
    try:
        matches = bool(mol.HasSubstructMatch(pat))
    except Exception:
        return 0.5
    return 0.5 * (1.0 + float(matches))


def _qed_score(mol) -> float:
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def _geom_mean(values, weights) -> float:
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0
    log_acc = 0.0
    for v, w in zip(values, weights):
        v_clamped = max(float(v), GM_EPS)
        log_acc += (w / w_sum) * float(np.log(v_clamped))
    return float(np.exp(log_acc))


def load_film(target_dir: Path):
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
    import inspect
    ckpt = torch.load(target_dir / "filmdelta.pt", map_location="cpu", weights_only=False)
    sig = inspect.signature(FiLMDeltaMLP.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    hp = {k: v for k, v in ckpt["hyperparameters"].items() if k in allowed}
    model = FiLMDeltaMLP(**hp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_dabs(target_dir: Path):
    import inspect
    ckpt = torch.load(target_dir / "directabs.pt", map_location="cpu", weights_only=False)
    sig = inspect.signature(DirectAbsoluteMLP.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    hp = {k: v for k, v in ckpt["hyperparameters"].items() if k in allowed}
    model = DirectAbsoluteMLP(**hp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_anchors(target_dir: Path):
    df = pd.read_csv(target_dir / "anchor_pool_strategy_b.csv")
    smi_col = "smiles" if "smiles" in df.columns else "anchor_smiles"
    pic_col = "pIC50" if "pIC50" in df.columns else "anchor_pIC50"
    fps, pics = [], []
    for s, p in zip(df[smi_col], df[pic_col]):
        _m, fp = _smi_to_fp(s)
        if fp is None:
            continue
        fps.append(fp)
        pics.append(float(p))
    return np.stack(fps).astype(np.float32), np.array(pics, dtype=np.float32)


def load_warhead(target_dir: Path):
    with open(target_dir / "warhead_smarts.json") as f:
        spec = json.load(f)
    strict = spec.get("smarts_strict")
    generic = spec.get("smarts_generic")
    s_pat = Chem.MolFromSmarts(strict) if strict else None
    g_pat = Chem.MolFromSmarts(generic) if generic else None
    return s_pat, g_pat, spec.get("target_label", "?")


def score_pic50_batch(smiles_list):
    """Returns: list of pIC50 (float; 0.0 for invalid)."""
    out = [0.0] * len(smiles_list)
    valid_idx, valid_fps = [], []
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        _mol, fp = _smi_to_fp(smi)
        if fp is None:
            continue
        valid_fps.append(fp)
        valid_idx.append(i)
    if not valid_fps:
        return out
    fps_arr = np.stack(valid_fps)
    fps_t = torch.from_numpy(fps_arr).float()
    with torch.no_grad():
        if _SCORER == "dabs":
            preds = _MODEL(fps_t).numpy()
            for k, gi in enumerate(valid_idx):
                out[gi] = float(preds[k])
        else:
            # FiLM ensemble: for each query, average pIC50_anchor + delta over anchors
            n_anchors = _ANCHOR_FPS.shape[0]
            anchor_t = torch.from_numpy(_ANCHOR_FPS).float()
            for k, gi in enumerate(valid_idx):
                tgt = fps_t[k:k + 1].expand(n_anchors, -1)
                deltas = _MODEL(anchor_t, tgt).numpy().flatten()
                abs_preds = _ANCHOR_PICS + deltas
                out[gi] = float(np.mean(abs_preds))
    return out


def score_composite_batch(smiles_list):
    pics = score_pic50_batch(smiles_list)
    out = []
    for smi, fp in zip(smiles_list, pics):
        if not smi:
            out.append(GM_EPS)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(GM_EPS)
            continue
        s_pic = _double_sigmoid(fp, SIGMOID_LOW, SIGMOID_HIGH, SIGMOID_K)
        s_war = _warhead_match(mol)
        s_qed = _qed_score(mol)
        gm = _geom_mean([s_pic, s_war, s_qed], [W_FILM, W_WAR, W_QED])
        out.append(gm)
    return out, pics


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "scorer": _SCORER,
        "target_label": _TARGET_LABEL,
        "n_anchors": int(_ANCHOR_FPS.shape[0]) if _ANCHOR_FPS is not None else 0,
        "model_loaded": _MODEL is not None,
        "n_requests": _N_REQ,
        "n_mols_scored": _N_MOLS,
        "server_version": SERVER_VERSION,
        "weights": {"film": W_FILM, "war": W_WAR, "qed": W_QED},
        "gm_eps": GM_EPS,
    })


@app.route("/score", methods=["POST"])
def score():
    global _N_REQ, _N_MOLS
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, list):
        return jsonify({"error": "Body must be a list of {input_string, query_id}"}), 400
    smiles_list = [str(item.get("input_string", "")) for item in body]
    query_ids = [str(item.get("query_id", i)) for i, item in enumerate(body)]
    _N_REQ += 1
    _N_MOLS += len(smiles_list)
    predictor_id = (request.args.get("predictor_id") or "composite").lower()
    try:
        if "raw" in predictor_id or "pic50" in predictor_id:
            scores = score_pic50_batch(smiles_list)
        else:
            scores, _ = score_composite_batch(smiles_list)
    except Exception as e:
        print(f"[exp6-rest] scoring error: {e}", flush=True)
        scores = [0.0] * len(smiles_list)
    successes = []
    for qid, sc in zip(query_ids, scores):
        sv = float(sc) if sc is not None and not (isinstance(sc, float) and np.isnan(sc)) else 0.0
        successes.append({"query_id": qid, "output_value": sv})
    if _N_REQ % 50 == 0:
        gc.collect()
    return jsonify({"output": {"successes_list": successes}})


def main():
    global _MODEL, _SCORER, _ANCHOR_FPS, _ANCHOR_PICS
    global _SMARTS_STRICT_PAT, _SMARTS_GENERIC_PAT, _TARGET_LABEL, GM_EPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--scorer", choices=["film", "dabs"], required=True)
    ap.add_argument("--gm_eps", type=float, default=0.05)
    args = ap.parse_args()

    GM_EPS = float(args.gm_eps)
    _SCORER = args.scorer

    _setup_film_import(args.src_root)
    tgt = Path(args.target_dir)

    if _SCORER == "film":
        _MODEL = load_film(tgt)
        _ANCHOR_FPS, _ANCHOR_PICS = load_anchors(tgt)
    else:
        _MODEL = load_dabs(tgt)
        _ANCHOR_FPS = np.zeros((0, 2048), dtype=np.float32)
        _ANCHOR_PICS = np.zeros(0, dtype=np.float32)

    _SMARTS_STRICT_PAT, _SMARTS_GENERIC_PAT, _TARGET_LABEL = load_warhead(tgt)
    print(
        f"[exp6-rest] target={_TARGET_LABEL} scorer={_SCORER} n_anchors={len(_ANCHOR_PICS)} "
        f"port={args.port} gm_eps={GM_EPS}",
        flush=True,
    )
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=2)
    except ImportError:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
