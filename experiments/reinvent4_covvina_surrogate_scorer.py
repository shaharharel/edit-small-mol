#!/usr/bin/env python3
"""REINVENT4 ExternalProcess scorer: distilled cov-Vina surrogate.

Predicts log1p(cov-Vina affinity) from Morgan FP via a tiny MLP trained by
experiments/train_covvina_surrogate.py. Transforms to a [0,1] reward where
lower cov-Vina (= better binding) -> higher score.

Reward transform:
    z = (log1p(vina_raw_pred) - y_mean) / y_std            # standardized
    score = sigmoid(-z * 1.5)                              # lower-is-better
This puts the median training mol at ~0.5 and rewards predicted-better binders.

Input  (stdin): one SMILES per line.
Output (stdout JSON): {"version": 1, "payload": {"covvina_score": [...]}}
"""
from __future__ import annotations
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("COVVINA_SURROGATE_PATH",
                                  str(ROOT / "models/covvina_surrogate.pt")))


class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden=(512, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def smi_to_fp(smi: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_model():
    if not MODEL_PATH.exists():
        sys.stderr.write(f"[covvina_surrogate] FATAL: model not found at {MODEL_PATH}\n")
        sys.exit(1)
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    model = MLP(in_dim=int(ckpt.get("n_bits", 2048)),
                hidden=tuple(ckpt.get("hidden", (512, 256, 128))),
                dropout=float(ckpt.get("dropout", 0.2)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["y_mean"]), float(ckpt["y_std"]), int(ckpt.get("n_bits", 2048))


def main():
    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        print(json.dumps({"version": 1, "payload": {"covvina_score": []}}))
        return
    model, y_mean, y_std, n_bits = load_model()

    fps = []
    keep = []
    for i, smi in enumerate(smiles_list):
        fp = smi_to_fp(smi, n_bits)
        if fp is None:
            continue
        fps.append(fp)
        keep.append(i)
    scores = [0.0] * len(smiles_list)
    if fps:
        X = torch.from_numpy(np.array(fps, dtype=np.float32))
        with torch.no_grad():
            pred_z = model(X).numpy()  # standardized
            # lower-is-better: sigmoid(-z * k)
            k = 1.5
            s = 1.0 / (1.0 + np.exp(pred_z * k))
        for j, orig_i in enumerate(keep):
            scores[orig_i] = float(s[j])
    print(json.dumps({"version": 1, "payload": {"covvina_score": scores}}))
    sys.stderr.write(f"[covvina_surrogate] scored n={len(scores)} mean={np.mean(scores):.3f}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # Force model lookup, print 3 test scores
        model, y_mean, y_std, n_bits = load_model()
        tests = [
            ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "Mol1"),
            ("CCO", "ethanol"),
            ("C=CC(=O)NCc1ccc(F)cc1", "small acrylamide"),
        ]
        for smi, label in tests:
            fp = smi_to_fp(smi, n_bits)
            with torch.no_grad():
                pz = model(torch.from_numpy(fp[None, :]).float()).item()
            s = 1.0 / (1.0 + np.exp(pz * 1.5))
            print(f"  {label}: pred_z={pz:+.3f} score={s:.3f}  smi={smi}")
    else:
        main()
