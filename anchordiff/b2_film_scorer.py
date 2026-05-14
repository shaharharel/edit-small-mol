"""FiLMDelta anchor-based scorer (T4-resident).

Loads the cached `reinvent4_film_model.pt` and exposes a single function:

    score_smiles_batch(smiles_list) -> np.ndarray
        returns predicted absolute pIC50 (mean over anchor pairs) per SMILES.
        Invalid / un-fingerprintable SMILES yield -np.inf so they sink in the
        softmax.

This file is intended to live at ~/DiffSBDD/b2_film_scorer.py on T4 — alongside
b2_inpaint.py — and to be imported by it. It uses CPU only; the model is small
(~35 MB, 280 anchors x 2048 dim) and FP scoring 16 mols x 280 anchors takes
<1 s on T4's CPU.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")


# ── Minimal FiLMDeltaMLP (matches src/models/predictors/film_delta_predictor.py) ──
class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, cond):
        return self.gamma_proj(cond) * h + self.beta_proj(cond)


class FiLMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.film = FiLMLayer(hidden_dim, cond_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.linear(x)
        h = self.activation(h)
        h = self.film(h, cond)
        h = self.dropout(h)
        return h


class FiLMDeltaMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(1024, 512, 256), dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList()
        prev = input_dim
        for hd in hidden_dims:
            self.blocks.append(FiLMBlock(prev, hd, delta_hidden, dropout))
            prev = hd
        self.output = nn.Linear(prev, 1)

    def forward_single(self, x, delta_cond):
        h = x
        for blk in self.blocks:
            h = blk(h, delta_cond)
        return self.output(h).squeeze(-1)

    def forward(self, emb_a, emb_b):
        delta = emb_b - emb_a
        delta_cond = self.delta_encoder(delta)
        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        return pred_b - pred_a


class FiLMScorer:
    def __init__(self, ckpt_path: str | Path):
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model = FiLMDeltaMLP(
            input_dim=2048, hidden_dims=(1024, 512, 256), dropout=0.2
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.scaler_mean = ckpt["scaler_mean"].astype(np.float32)
        self.scaler_scale = ckpt["scaler_scale"].astype(np.float32)
        self.anchor_embs = ckpt["anchor_embs"].float()  # already standardized in checkpoint
        self.anchor_pIC50 = ckpt["anchor_pIC50"].astype(np.float32)
        self.n_anchors = len(self.anchor_pIC50)

    def _smi_to_fp(self, smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def score_smiles_batch(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Return predicted pIC50 per SMILES; -inf for invalid."""
        scores = np.full(len(smiles_list), -np.inf, dtype=np.float32)
        valid_idx, fps = [], []
        for i, smi in enumerate(smiles_list):
            arr = self._smi_to_fp(smi)
            if arr is not None:
                valid_idx.append(i)
                fps.append(arr)
        if not fps:
            return scores
        fps = np.array(fps, dtype=np.float32)
        std = (fps - self.scaler_mean) / self.scaler_scale
        embs = torch.from_numpy(std).float()
        with torch.no_grad():
            for k, orig_i in enumerate(valid_idx):
                target = embs[k:k + 1].expand(self.n_anchors, -1)
                deltas = self.model(self.anchor_embs, target).numpy()
                scores[orig_i] = float(np.mean(self.anchor_pIC50 + deltas))
        return scores


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "~/anchordiff/reinvent4_film_model.pt"
    s = FiLMScorer(Path(ckpt).expanduser())
    test = [
        "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1",
        "C=CC(=O)N1CCC[C@@H](C1)n1nc(c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
        "CCO",
        "not a smiles",
    ]
    print(s.score_smiles_batch(test))
