"""Distilled validator REST scoring server.

Loads the XGBoost iptm/mPAE distilled-validator bundle and exposes a /score
endpoint compatible with the REINVENT4 REST scoring component
(returns score in [0, 1]; we map iptm via a sigmoid).

Run: python experiments/distilled_validator_server.py --port 8089
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "models/boltz_distilled/boltz_distilled_v1.pkl"

# ── Load model bundle ─────────────────────────────────────────────────────
print(f"[init] loading distilled validator bundle from {BUNDLE_PATH}", flush=True)
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
m_iptm = bundle["iptm_model"]
m_mpae = bundle["mpae_model"]
N_BITS = bundle["fp_nbits"]
RADIUS = bundle["fp_radius"]
print(
    f"  loaded: iptm_model (CV r={bundle['cv_pearson_iptm']:.3f}, R²={bundle['cv_r2_iptm']:.3f}); "
    f"mpae_model (CV r={bundle['cv_pearson_mpae']:.3f}); "
    f"trained on {bundle['training_n']} mols",
    flush=True,
)

app = Flask(__name__)


def fp_of(smi):
    if smi is None or not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS))
    except Exception:
        return None


def sigmoid_iptm(x: float, low: float = 0.85, high: float = 0.95, k: float = 30.0) -> float:
    """Sigmoid that maps iptm <=low to ~0 and iptm>=high to ~1."""
    mid = 0.5 * (low + high)
    return float(1.0 / (1.0 + np.exp(-k * (x - mid))))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "distilled_validator", "training_n": bundle["training_n"]})


@app.route("/score", methods=["POST"])
def score():
    """REINVENT4 REST contract:
        Request:  [{"input_string": smiles, "query_id": "0"}, ...]
        Response: {"output": {"successes_list": [{"query_id": "0", "output_value": 0.5}, ...]}}
    """
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, list):
        return jsonify({"output": {"successes_list": []}})

    smiles_list = [str(item.get("input_string", "")) if isinstance(item, dict) else "" for item in body]
    query_ids = [str(item.get("query_id", i)) if isinstance(item, dict) else str(i) for i, item in enumerate(body)]

    fps, valid = [], []
    for s in smiles_list:
        f = fp_of(s)
        if f is None:
            valid.append(False)
            fps.append(np.zeros(N_BITS, dtype=int))
        else:
            valid.append(True)
            fps.append(f)
    X = np.stack(fps)

    iptm_pred = m_iptm.predict(X)
    scores = [sigmoid_iptm(float(p)) if v else 0.0 for p, v in zip(iptm_pred, valid)]

    successes = [{"query_id": qid, "output_value": float(sc)} for qid, sc in zip(query_ids, scores)]
    return jsonify({"output": {"successes_list": successes}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()
    print(f"[distilled_validator] listening on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
