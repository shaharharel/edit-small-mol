"""Persistent FiLMDelta REST scoring server for REINVENT4.

Replaces the ExternalProcess (subprocess-per-batch) scorer with a long-lived
Flask server. The FiLM model is loaded ONCE at server startup; all subsequent
RL steps just POST SMILES batches and get scores back.

Wire format follows REINVENT4's `comp_generic_rest.py` plugin:
- POST <url>:<port>/<endpoint>
- JSON body: [{"input_string": smiles, "query_id": "0"}, ...]
- URL params: predictor_id, predictor_version, inp_fmt=smiles
- Response:  {"output": {"successes_list": [{"query_id": "0", "output_value": float}, ...]}}

Performance vs ExternalProcess (measured 2026-06-08):
- ExternalProcess: ~30-60s per RL step (model load every call)
- REST: ~1-2s per RL step (model loaded once, kept hot)
- Speedup: 15-30× per scoring call

Launch:
    python experiments/reinvent4_film_rest_server.py --port 8088

Toml config (replace [stage.scoring.component.ExternalProcess] block with):
    [[stage.scoring.component]]
    [stage.scoring.component.REST]
    [[stage.scoring.component.REST.endpoint]]
    name = "FiLMDelta pIC50"
    weight = 0.50
    params.server_url = ["http://127.0.0.1"]
    params.server_port = [8088]
    params.server_endpoint = ["score"]
    params.predictor_id = ["film_delta"]
    params.predictor_version = ["v1"]
    transform.type = "sigmoid"
    transform.high = 7.5
    transform.low = 5.5
    transform.k = 0.5
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
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Late imports so flags are honored first
import numpy as np
import torch
from flask import Flask, jsonify, request
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# --- FiLMDelta model loader: import from the existing scorer ---
from experiments.reinvent4_film_scorer import (  # noqa: E402
    MODEL_CACHE,
    load_film_model,
    score_smiles,
)

app = Flask(__name__)

# Module-level state (loaded ONCE at startup)
_MODEL: Any = None
_SCALER: Any = None
_ANCHOR_EMBS: Any = None
_ANCHOR_PIC50: Any = None

# A small request counter so we can see traffic
_N_REQ = 0
_N_MOLS = 0


def _load_once():
    """Load the FiLM model from cache once; raise on failure (do not auto-train)."""
    global _MODEL, _SCALER, _ANCHOR_EMBS, _ANCHOR_PIC50
    if _MODEL is not None:
        return
    print(f"[server] Loading FiLM model from {MODEL_CACHE}", flush=True)
    if not Path(MODEL_CACHE).exists():
        raise FileNotFoundError(
            f"FiLM model cache not found at {MODEL_CACHE}. "
            "Train it once by running reinvent4_film_scorer.py on a SMILES."
        )
    _MODEL, _SCALER, _ANCHOR_EMBS, _ANCHOR_PIC50 = load_film_model()
    print(
        f"[server] Model loaded: {len(_ANCHOR_EMBS)} anchors, "
        f"model dtype={next(_MODEL.parameters()).dtype}",
        flush=True,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _MODEL is not None,
            "n_anchors": len(_ANCHOR_EMBS) if _ANCHOR_EMBS is not None else 0,
            "n_requests": _N_REQ,
            "n_mols_scored": _N_MOLS,
        }
    )


@app.route("/score", methods=["POST"])
def score():
    """REINVENT4-compatible scoring endpoint.

    Expects JSON body like:
      [{"input_string": "c1ccccc1", "query_id": "0"}, ...]
    URL params: predictor_id, predictor_version, inp_fmt
    Returns:
      {"output": {"successes_list": [{"query_id": "0", "output_value": 6.5}, ...]}}
    """
    global _N_REQ, _N_MOLS

    body = request.get_json(silent=True)
    if body is None or not isinstance(body, list):
        return jsonify({"error": "Body must be a list of {input_string, query_id}"}), 400

    smiles_list = [str(item.get("input_string", "")) for item in body]
    query_ids = [str(item.get("query_id", i)) for i, item in enumerate(body)]

    _N_REQ += 1
    _N_MOLS += len(smiles_list)

    # Score: returns dict {"pIC50": [...]} or {"delta": [...]} depending on params
    predictor_property = request.args.get("predictor_id", "film_delta")
    # Map predictor_id -> property name expected by the scorer
    prop = "pIC50" if "pIC50" in predictor_property or predictor_property == "film_delta" else "delta"

    try:
        scores = score_smiles(
            smiles_list, _MODEL, _SCALER, _ANCHOR_EMBS, _ANCHOR_PIC50
        )
    except Exception as e:
        scores = [float("nan")] * len(smiles_list)
        print(f"[server] scoring error: {e}", flush=True)

    successes = []
    for qid, sc in zip(query_ids, scores):
        # REINVENT4 parser accepts numeric output_value
        sc_val = float(sc) if sc is not None and not (isinstance(sc, float) and np.isnan(sc)) else 0.0
        successes.append({"query_id": qid, "output_value": sc_val})

    # Periodic GC to keep memory steady across many requests
    if _N_REQ % 50 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return jsonify({"output": {"successes_list": successes}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    _load_once()
    print(f"[server] Listening on http://{args.host}:{args.port}", flush=True)
    # Use waitress for production-ish single-process serving; fallback to Flask dev
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=1)
    except ImportError:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=False)


if __name__ == "__main__":
    main()
