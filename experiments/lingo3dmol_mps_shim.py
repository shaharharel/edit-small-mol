"""MPS variant of the cpu_shim.

Lingo3DMol's official codebase hard-codes `.cuda()` calls throughout the
model + inference loop. We redirect `.cuda()` to `.to("mps")` so the model
runs on Apple Silicon GPU instead of CPU. Falls back to CPU if MPS
unavailable.
"""
from __future__ import annotations
import numpy as _np
import torch
import torch.nn as nn

# torch.range -> torch.arange shim (some Lingo code uses .range())
if not hasattr(torch, "range") or getattr(torch.range, "_is_arange_shim", False) is False:
    _orig_range = getattr(torch, "range", None)
    def _range_shim(*args, **kwargs):
        return torch.arange(*args, **kwargs)
    _range_shim._is_arange_shim = True  # type: ignore[attr-defined]
    torch.range = _range_shim  # type: ignore[assignment]

# numpy backfill for old aliases
for _name, _builtin in [("float", float), ("int", int), ("bool", bool),
                        ("long", int), ("object", object), ("str", str)]:
    if not hasattr(_np, _name):
        setattr(_np, _name, _builtin)

# Pick device: prefer MPS, fall back to CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    _DEVICE = torch.device("mps")
    _DEVICE_NAME = "mps"
else:
    _DEVICE = torch.device("cpu")
    _DEVICE_NAME = "cpu"

# 1. tensor.cuda(...) -> tensor.to(device), auto-downcast float64→float32 for MPS
_orig_tensor_cuda = torch.Tensor.cuda
def _tensor_to_device(self, *args, **kwargs):
    if _DEVICE_NAME == "mps" and self.dtype == torch.float64:
        return self.to(torch.float32).to(_DEVICE)
    return self.to(_DEVICE)
torch.Tensor.cuda = _tensor_to_device  # type: ignore[assignment]

# 2. Module.cuda(...) -> Module.to(device), float32 only for MPS
_orig_module_cuda = nn.Module.cuda
def _module_to_device(self, *args, **kwargs):
    if _DEVICE_NAME == "mps":
        return self.to(torch.float32).to(_DEVICE)
    return self.to(_DEVICE)
nn.Module.cuda = _module_to_device  # type: ignore[assignment]

# 3. torch.cuda.is_available -> True so conditional code uses GPU branch
torch.cuda.is_available = lambda: True  # type: ignore[assignment]
torch.cuda.synchronize = lambda *a, **kw: None  # type: ignore[assignment]
torch.cuda.device_count = lambda: 1  # type: ignore[assignment]
torch.cuda.current_device = lambda: 0  # type: ignore[assignment]

# 4. DataParallel pass-through
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
        return self.to(_DEVICE)

    def eval(self):
        self.module.eval()
        return self

nn.DataParallel = _PassthroughDataParallel  # type: ignore[assignment]

# 5. torch.cross with no dim arg: default to dim=-1 (modern torch requires explicit dim)
_orig_cross = torch.cross
def _cross_with_default_dim(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -1
    if out is not None:
        return _orig_cross(input, other, dim=dim, out=out)
    return _orig_cross(input, other, dim=dim)
torch.cross = _cross_with_default_dim  # type: ignore[assignment]

print(f"[lingo3dmol_mps_shim] {_DEVICE_NAME.upper()} shim active: .cuda() = .to({_DEVICE_NAME}), torch.cuda is faked, torch.range aliased, torch.cross dim defaulted")
