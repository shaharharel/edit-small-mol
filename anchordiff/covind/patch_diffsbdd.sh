#!/usr/bin/env bash
# Re-applies the C-arm .detach() removal patch to a fresh DiffSBDD install.
#
# Why: DiffSBDD's `noised_representation` (and `sample_normal_zero_com`) in
# `equivariant_diffusion/conditional_model.py` do
#
#     xh_pocket = xh0_pocket.detach().clone()
#
# This was originally safe because pocket features are "fixed context" and
# the original training never tried to backprop into them. For our C-arm
# (covalent-token conditioning injected into `pocket['one_hot']`) the
# .detach() blocks ALL gradient flow back to the adapter, making the
# adapter weights frozen at init.
#
# This patch replaces both occurrences with `.clone()` (autograd-aware) and
# tags them with `# C-ARM-PATCH`. Idempotent — safe to re-run.
#
# Slicing assumption: in-place writes happen on `xh_pocket[:, :n_dims]`
# (the COORD slice). The adapter's bias lives on `xh_pocket[:, n_dims:]`
# (the ONE-HOT slice). The two slices share no storage with the un-mutated
# `xh0_pocket` once cloned, so removing .detach() is safe for gradient
# flow on the one-hot slice. If a future DiffSBDD change starts using the
# pre-mutation coord slice for backward, this patch may need to be
# revisited.
#
# Usage:  bash anchordiff/covind/patch_diffsbdd.sh
#         (expects DiffSBDD repo at ~/DiffSBDD)
set -euo pipefail
FP="${HOME}/DiffSBDD/equivariant_diffusion/conditional_model.py"
if [[ ! -f "$FP" ]]; then
    echo "[patch] $FP not found"
    exit 1
fi
if grep -q "C-ARM-PATCH" "$FP"; then
    echo "[patch] already applied to $FP"
    exit 0
fi
python3 - "$FP" <<'PYEOF'
import sys, pathlib
fp = pathlib.Path(sys.argv[1])
src = fp.read_text()
old = "xh_pocket = xh0_pocket.detach().clone()"
new = "xh_pocket = xh0_pocket.clone()  # C-ARM-PATCH"
n = src.count(old)
if n == 0:
    print(f"[patch] pattern not found in {fp}")
    sys.exit(2)
fp.write_text(src.replace(old, new))
print(f"[patch] replaced {n} occurrences in {fp}")
PYEOF

# Clear pyc cache so the patch takes effect immediately.
find "${HOME}/DiffSBDD" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[patch] DiffSBDD pyc cache cleared"
echo "[patch] DONE"
