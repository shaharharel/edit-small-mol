"""REINVENT4-format REST server exposing the FastGeomScorer.

Same wire protocol as `experiments/reinvent4_film_rest_server.py`:
  POST /score   with body  [{"input_string": smiles, "query_id": "0"}, ...]
  ->   {"output": {"successes_list": [{"query_id": "0", "output_value": 0.5}, ...]}}

Health: GET /health.

Launch:
  python experiments/exp_fast_geom_surrogate/fast_geom_server.py --port 8090

Toml block to wire into the RL run:
  [[stage.scoring.component]]
  [stage.scoring.component.REST]
  [[stage.scoring.component.REST.endpoint]]
  name = "Fast geometry surrogate"
  weight = 0.10
  params.server_url = ["http://127.0.0.1"]
  params.server_port = [8090]
  params.server_endpoint = ["score"]
  params.predictor_id = ["fast_geom"]
  params.predictor_version = ["v1"]
  transform.type = "value_mapping"  # scores are already [0,1]; no transform needed
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
logging.disable(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from flask import Flask, jsonify, request
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from fast_geom_scorer import FastGeomScorer  # noqa: E402

DEFAULT_ANCHOR = PROJECT_ROOT / "data/fast_geom_surrogate/anchor_frame.npz"

app = Flask(__name__)
_SCORER: FastGeomScorer | None = None
_N_REQ = 0
_N_MOLS = 0


def _load_once(anchor_path: Path, n_conf: int):
    global _SCORER
    if _SCORER is not None:
        return
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor frame not found at {anchor_path} — "
                                f"run experiments/exp_fast_geom_surrogate/anchor_frame.py first")
    print(f"[server] loading FastGeomScorer from {anchor_path} (n_conf={n_conf})", flush=True)
    _SCORER = FastGeomScorer(anchor_path, n_conformers=n_conf)
    print(f"[server] ready: anchor_iptm={float(np.load(anchor_path, allow_pickle=True)['anchor_iptm'][0]):.3f}",
          flush=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "scorer_loaded": _SCORER is not None,
        "n_requests": _N_REQ,
        "n_mols_scored": _N_MOLS,
    })


@app.route("/score", methods=["POST"])
def score():
    global _N_REQ, _N_MOLS
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, list):
        return jsonify({"error": "body must be list of {input_string, query_id}"}), 400
    smiles_list = [str(item.get("input_string", "")) for item in body]
    query_ids = [str(item.get("query_id", i)) for i, item in enumerate(body)]
    _N_REQ += 1
    _N_MOLS += len(smiles_list)
    successes = []
    for qid, smi in zip(query_ids, smiles_list):
        try:
            sc = _SCORER.score(smi)
        except Exception as e:
            print(f"[server] scoring error on {smi[:60]}: {e}", flush=True)
            sc = 0.0
        if sc is None or not np.isfinite(sc):
            sc = 0.0
        successes.append({"query_id": qid, "output_value": float(sc)})
    if _N_REQ % 50 == 0:
        gc.collect()
    return jsonify({"output": {"successes_list": successes}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--anchor", type=str, default=str(DEFAULT_ANCHOR))
    ap.add_argument("--n-conf", type=int, default=3)
    args = ap.parse_args()
    _load_once(Path(args.anchor), args.n_conf)
    print(f"[server] listening on http://{args.host}:{args.port}", flush=True)
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=1)
    except ImportError:
        app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
