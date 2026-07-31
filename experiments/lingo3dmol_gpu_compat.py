"""GPU compatibility shim for Lingo3DMol on T4/CUDA.

Unlike `lingo3dmol_cpu_shim`, this version keeps `.cuda()` functional so
real GPU is used. It only adds the numpy/torch compatibility fixes the
2023 codebase needs to run on modern torch (2.x) + numpy (1.20+).

Import BEFORE any Lingo3DMol model/inference code:

    import lingo3dmol_gpu_compat  # noqa: F401
    from model.transformer_v1_res_fac2 import TransformerModel
"""
from __future__ import annotations

import torch
import numpy as _np

# 1. numpy 1.20+ removed np.float, np.int, np.bool, np.long, np.object aliases
for _name, _builtin in [("float", float), ("int", int), ("bool", bool),
                        ("long", int), ("object", object), ("str", str)]:
    if not hasattr(_np, _name):
        setattr(_np, _name, _builtin)

# 2. torch.range removed in torch 2.x — alias to arange (inclusive end semantics)
if not hasattr(torch, "range") or not callable(getattr(torch, "range", None)):
    def _range(start, end, step=1, **kwargs):
        # torch.range was inclusive of end; arange is exclusive
        return torch.arange(start, end + step, step, **kwargs)
    torch.range = _range  # type: ignore[assignment]

# 3. torch.cross now requires dim argument in modern torch; old behavior was dim=-1
_orig_cross = torch.cross

def _cross_compat(input, other, dim=None, *args, **kwargs):
    if dim is None:
        dim = -1
    return _orig_cross(input, other, dim=dim, *args, **kwargs)

torch.cross = _cross_compat  # type: ignore[assignment]

print(f"[lingo3dmol_gpu_compat] active: cuda={torch.cuda.is_available()} "
      f"device={'Tesla T4' if torch.cuda.is_available() else 'cpu'} "
      f"torch.range aliased, torch.cross dim defaulted, np.float compat installed")
