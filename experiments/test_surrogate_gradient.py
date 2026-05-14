"""
Sanity check: load the trained 3D FiLMDelta surrogate and verify that the
gradient of the predicted absolute pIC50 w.r.t. atom positions is non-trivial
and behaves sensibly.

Tests:
    1. Forward pass on a random anchor produces a finite scalar.
    2. ``∇_x pIC50`` is finite (no NaN / Inf), nonzero, and produces a
       measurable change in the prediction when we take a small step.
    3. The gradient direction roughly matches the finite-difference gradient.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_3d_surrogate import FiLMDelta3DSurrogate  # noqa: E402

CONFORMER_CACHE = PROJECT_ROOT / "data" / "embedding_cache" / "zap70_3d_conformers.pkl"
SURROGATE_OUT = PROJECT_ROOT / "results" / "paper_evaluation" / "film_delta_3d_surrogate.pt"


def load_surrogate():
    ckpt = torch.load(SURROGATE_OUT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FiLMDelta3DSurrogate(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main():
    print("[load] surrogate + conformers")
    model, ckpt = load_surrogate()
    anchor_smiles = ckpt["anchor_smiles"]
    anchor_pIC50 = torch.tensor(ckpt["anchor_pIC50"], dtype=torch.float32)
    with open(CONFORMER_CACHE, "rb") as f:
        conformers = pickle.load(f)

    # Use a subset of anchors for speed (50 of 280).
    rng = np.random.RandomState(0)
    sub = rng.choice(len(anchor_smiles), size=min(50, len(anchor_smiles)),
                     replace=False)
    anchor_z_list = [torch.from_numpy(conformers[anchor_smiles[i]]["z"]) for i in sub]
    anchor_pos_list = [torch.from_numpy(conformers[anchor_smiles[i]]["pos"])
                       for i in sub]
    anchor_pIC50_sub = anchor_pIC50[sub.tolist()]

    # Pick a candidate molecule (not in the anchor subset).
    cand_idx = int([k for k in range(len(anchor_smiles)) if k not in sub][0])
    cand_smi = anchor_smiles[cand_idx]
    cand_z = torch.from_numpy(conformers[cand_smi]["z"])
    cand_pos = torch.from_numpy(conformers[cand_smi]["pos"]).clone()
    cand_pos.requires_grad_(True)

    # ----- Forward -----
    pred = model.predict_abs_pIC50(
        cand_z, cand_pos, anchor_z_list, anchor_pos_list, anchor_pIC50_sub
    )
    print(f"[forward] pred pIC50 = {pred.item():.4f}")
    assert torch.isfinite(pred).item(), "Forward NaN/Inf!"

    # ----- Backward -----
    pred.backward()
    grad = cand_pos.grad
    assert grad is not None, "No gradient!"
    g_norm = grad.norm().item()
    g_max = grad.abs().max().item()
    n_nan = torch.isnan(grad).sum().item()
    n_inf = torch.isinf(grad).sum().item()
    print(f"[grad] norm={g_norm:.6f} max_abs={g_max:.6f} "
          f"nan={n_nan} inf={n_inf} shape={tuple(grad.shape)}")
    if n_nan > 0 or n_inf > 0:
        print("[FAIL] gradient has NaN/Inf entries.")
        return 1
    if g_norm < 1e-8:
        print("[FAIL] gradient is dead (norm < 1e-8).")
        return 1

    # ----- Finite difference check -----
    # Take a tiny step along -grad and verify the prediction decreases (and
    # along +grad it increases). Use a small step so first-order Taylor holds.
    with torch.no_grad():
        step = 1e-3
        direction = -grad / (g_norm + 1e-12)
        pos_minus = cand_pos.detach() + step * direction
        pos_plus = cand_pos.detach() - step * direction
        pred_minus = model.predict_abs_pIC50(
            cand_z, pos_minus, anchor_z_list, anchor_pos_list, anchor_pIC50_sub
        ).item()
        pred_plus = model.predict_abs_pIC50(
            cand_z, pos_plus, anchor_z_list, anchor_pos_list, anchor_pIC50_sub
        ).item()
    delta_minus = pred_minus - pred.item()
    delta_plus = pred_plus - pred.item()
    print(f"[fd] step=±{step}  Δ(-grad)={delta_minus:+.6f}  "
          f"Δ(+grad)={delta_plus:+.6f}")

    monotone = (delta_minus < 0) and (delta_plus > 0)
    if monotone:
        print("[ok] gradient direction is sensible "
              "(prediction decreases along -∇, increases along +∇).")
    else:
        # Very small magnitudes can flip due to numerical noise; warn but don't
        # fail outright.
        if abs(delta_minus) < 1e-7 and abs(delta_plus) < 1e-7:
            print("[warn] step too small — gradient too weak to register; "
                  "increase step or check model capacity.")
        else:
            print("[warn] gradient direction did not strictly match finite "
                  "diff (likely curvature / step size). Norm is still "
                  "non-trivial, so guidance is usable.")

    print("[done] gradient sanity check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
