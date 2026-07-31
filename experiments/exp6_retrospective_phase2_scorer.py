#!/usr/bin/env python3
"""
EXP6 Retrospective LO — Phase 2 per-target scorer for REINVENT4.

This single scorer is configured via environment variables and used by REINVENT4
ExternalProcess scoring components. It loads the target-specific FiLMDelta +
DirectAbsoluteMLP checkpoints written by Phase 1, then scores SMILES with both
a delta-based ensemble (anchor pool) and direct-absolute branch and returns
their mean as `pIC50`.

Environment:
  EXP6_TARGET_DIR   : path to data/exp6_retrospective/<target>/
  EXP6_ANCHOR_FILE  : path to a CSV with columns smiles (and pIC50)
                       (defaults to <TARGET_DIR>/anchor_pool_strategy_b.csv)

Protocol: stdin = newline-separated SMILES, stdout = JSON
  {"version": 1, "payload": {"pIC50": [...]}}
"""
from __future__ import annotations
import sys
import os
import json
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

# Project setup
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa


class DirectAbsoluteMLP(nn.Module):
    """Mirror of the prep-script DirectAbsoluteMLP (3-hidden ReLU MLP)."""
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


def smi_to_morgan(smi: str, n_bits=2048, radius=2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _filter_kwargs(klass, hp: dict) -> dict:
    import inspect
    sig = inspect.signature(klass.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in hp.items() if k in valid}


def load_models(target_dir: Path):
    film_ckpt = torch.load(target_dir / "filmdelta.pt", map_location="cpu", weights_only=False)
    dabs_ckpt = torch.load(target_dir / "directabs.pt", map_location="cpu", weights_only=False)

    film_kw = _filter_kwargs(FiLMDeltaMLP, film_ckpt["hyperparameters"])
    film = FiLMDeltaMLP(**film_kw)
    film.load_state_dict(film_ckpt["model_state_dict"])
    film.eval()

    dabs_kw = _filter_kwargs(DirectAbsoluteMLP, dabs_ckpt["hyperparameters"])
    dabs = DirectAbsoluteMLP(**dabs_kw)
    dabs.load_state_dict(dabs_ckpt["model_state_dict"])
    dabs.eval()

    return film, dabs, film_ckpt["fp_config"]


def load_anchors(target_dir: Path, anchor_file: Path | None):
    af = anchor_file or (target_dir / "anchor_pool_strategy_b.csv")
    df = pd.read_csv(af)
    # Accept "smiles"/"pIC50" or "anchor_smiles"/"anchor_pIC50".
    smi_col = "smiles" if "smiles" in df.columns else "anchor_smiles"
    pic_col = "pIC50" if "pIC50" in df.columns else "anchor_pIC50"
    fps = []
    pics = []
    for s, p in zip(df[smi_col], df[pic_col]):
        fp = smi_to_morgan(s)
        if fp is None:
            continue
        fps.append(fp)
        pics.append(float(p))
    return np.stack(fps), np.array(pics, dtype=np.float32)


def score_batch(smiles_list, film, dabs, anchor_fps, anchor_pics):
    """Return list of pIC50 scores (NaN -> 0.0 so REINVENT4 won't choke).
    Score = mean(film_anchor_avg, direct_abs_pred)."""
    n_anchors = anchor_fps.shape[0]
    anchor_fps_t = torch.from_numpy(anchor_fps).float()
    out = []
    valid_fp_buf = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        fp = smi_to_morgan(smi)
        if fp is None:
            out.append(0.0)
            continue
        valid_fp_buf.append(fp)
        valid_idx.append(i)
        out.append(None)

    if not valid_fp_buf:
        return out

    fps_arr = np.stack(valid_fp_buf)
    fps_t = torch.from_numpy(fps_arr).float()
    with torch.no_grad():
        # DirectAbs
        dabs_pred = dabs(fps_t).numpy()
        # FiLMDelta: for each query, expand against anchor pool then take mean
        for k, gi in enumerate(valid_idx):
            tgt = fps_t[k:k+1].expand(n_anchors, -1)
            deltas = film(anchor_fps_t, tgt).numpy().flatten()
            abs_preds = anchor_pics + deltas
            film_mean = float(np.mean(abs_preds))
            dabs_v = float(dabs_pred[k])
            score = 0.5 * film_mean + 0.5 * dabs_v
            out[gi] = score
    return [float(x if x is not None else 0.0) for x in out]


def main():
    target_dir = Path(os.environ.get("EXP6_TARGET_DIR", "")).resolve()
    if not target_dir.exists():
        print(f"[exp6 scorer] EXP6_TARGET_DIR not set or missing: {target_dir}", file=sys.stderr)
        sys.exit(1)
    anchor_file = os.environ.get("EXP6_ANCHOR_FILE")
    anchor_file = Path(anchor_file) if anchor_file else None

    film, dabs, _fp_cfg = load_models(target_dir)
    anchor_fps, anchor_pics = load_anchors(target_dir, anchor_file)
    print(f"[exp6 scorer] target_dir={target_dir.name} n_anchors={len(anchor_pics)}", file=sys.stderr)

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        print(json.dumps({"version": 1, "payload": {"pIC50": []}}))
        return

    scores = score_batch(smiles_list, film, dabs, anchor_fps, anchor_pics)
    print(json.dumps({"version": 1, "payload": {"pIC50": scores}}))
    arr = np.array(scores, dtype=np.float64)
    print(f"[exp6 scorer] Done. Mean pIC50: {float(np.nanmean(arr)):.3f} | n={len(arr)}", file=sys.stderr)


if __name__ == "__main__":
    main()
