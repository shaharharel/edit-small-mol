"""CPU shim for running Lingo3DMol on a Mac without CUDA.

Lingo3DMol's official codebase hard-codes `.cuda()` calls throughout the
model and inference scripts. This module monkey-patches torch so that:

  - tensor.cuda(...)        -> returns self (no-op on CPU)
  - Module.cuda(...)        -> returns self
  - torch.cuda.is_available -> True (so guard code thinks GPU is present)
  - torch.cuda.synchronize  -> no-op
  - DataParallel(model)     -> returns model unchanged (avoid GPU detect)
  - torch.zeros/arange/...  -> remap any `device="cuda"` kwarg to "cpu"

Import this module *before* importing any Lingo3DMol model/inference code.

Usage:
    import lingo3dmol_cpu_shim  # noqa: F401  (must be FIRST)
    from model.transformer_v1_res_fac2 import TransformerModel
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as _np

# numpy 1.20+ removed np.float, np.int, np.bool, np.long, np.object aliases
for _name, _builtin in [("float", float), ("int", int), ("bool", bool),
                        ("long", int), ("object", object), ("str", str)]:
    if not hasattr(_np, _name):
        setattr(_np, _name, _builtin)

# 1. tensor.cuda(...) -> return self
_orig_tensor_cuda = torch.Tensor.cuda

def _noop_tensor_cuda(self, *args, **kwargs):  # noqa: D401
    return self

torch.Tensor.cuda = _noop_tensor_cuda  # type: ignore[assignment]

# 2. Module.cuda(...) -> return self
_orig_module_cuda = nn.Module.cuda

def _noop_module_cuda(self, *args, **kwargs):  # noqa: D401
    return self

nn.Module.cuda = _noop_module_cuda  # type: ignore[assignment]

# 3. torch.cuda.is_available -> True so any conditional uses the "GPU branch"
torch.cuda.is_available = lambda: True  # type: ignore[assignment]
torch.cuda.synchronize = lambda *a, **kw: None  # type: ignore[assignment]
torch.cuda.device_count = lambda: 1  # type: ignore[assignment]
torch.cuda.current_device = lambda: 0  # type: ignore[assignment]

# 4. DataParallel pass-through (don't try to scatter across GPUs)
class _PassthroughDataParallel(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def cuda(self, *args, **kwargs):
        return self

    def eval(self):
        self.module.eval()
        return self

nn.DataParallel = _PassthroughDataParallel  # type: ignore[assignment]

# 5. Remap device="cuda" kwarg to device="cpu" for tensor factory calls
def _patch_factory(fn_name: str):
    orig = getattr(torch, fn_name)

    def wrapper(*args, **kwargs):
        d = kwargs.get("device")
        if d is not None and ("cuda" in str(d)):
            kwargs["device"] = torch.device("cpu")
        return orig(*args, **kwargs)

    wrapper.__name__ = fn_name
    setattr(torch, fn_name, wrapper)

for _fn in ("zeros", "ones", "empty", "arange", "tensor", "rand", "randn", "full", "randint", "LongTensor"):
    try:
        _patch_factory(_fn)
    except Exception:
        pass

# 6. torch.range removed in torch 2.x — alias to arange (inclusive end semantics)
if not hasattr(torch, "range") or not callable(getattr(torch, "range", None)):
    def _range(start, end, step=1, **kwargs):
        # torch.range was inclusive of end; arange is exclusive
        return torch.arange(start, end + step, step, **kwargs)
    torch.range = _range  # type: ignore[assignment]

# 7. torch.cross now requires dim argument in modern torch; old behavior was dim=-1
_orig_cross = torch.cross

def _cross_compat(input, other, dim=None, *args, **kwargs):
    if dim is None:
        # default to last dim (old behavior of torch.cross when both are 3-vectors)
        dim = -1
    return _orig_cross(input, other, dim=dim, *args, **kwargs)

torch.cross = _cross_compat  # type: ignore[assignment]

print("[lingo3dmol_cpu_shim] CPU shim active: torch.cuda is faked, .cuda() = no-op, torch.range aliased, torch.cross dim defaulted")
