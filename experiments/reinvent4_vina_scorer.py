#!/usr/bin/env python3
"""Vina-in-loop scorer for REINVENT4 ExternalProcess (EXP6_v5).

Reads SMILES from stdin (one per line), runs full AD-CovDock for each:
  3D embed -> meeko CovalentBuilder tether -> Vina --score_only against
  Cys346-stripped ZAP70 receptor (4K2R). Outputs JSON to stdout.

Output schema (REINVENT4 ExternalProcess contract):
  {"version": 1, "payload": {"vina_score": [s1, s2, ...]}}

vina_score convention here: POSITIVE numbers (same as the existing
run_covalent_docking.py output). Lower = better. The TOML uses reverse_sigmoid
to map low scores to high reward.

Failures return NaN — REINVENT4 handles NaN as score=0 (effective penalty).

Multiprocessing: uses N_WORKERS parallel Vina calls per batch. Total per-step
overhead at 620 mols: ~620 * 8s / 6 workers = ~14 min/step.
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore")
os.environ.setdefault("RDK_DEPRECATION_WARNING", "off")

# Reuse the existing battle-tested AD-CovDock functions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Defensive sys.path: subprocess context may have empty PYTHONPATH; add both
# relative (via __file__) AND absolute fallback for both local and V100 layouts.
for _root in (str(PROJECT_ROOT),
              str(PROJECT_ROOT / "experiments"),
              "/home/shaharh_quris_ai/edit-small-mol",
              "/home/shaharh_quris_ai/edit-small-mol/experiments",
              "/Users/shaharharel/Documents/github/edit-small-mol",
              "/Users/shaharharel/Documents/github/edit-small-mol/experiments"):
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

# Direct import (not importlib) so multiprocessing pickling works
from experiments.run_covalent_docking import (  # noqa: E402
    build_cov_tethered_pdbqt,
    vina_dock,
    read_tethered_geom,
)
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK_DIR = PROJECT_ROOT / "data" / "vina_scorer_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
N_WORKERS = int(os.environ.get("VINA_SCORER_WORKERS", "6"))


def _embed_and_score(args):
    # Per-worker fd-level redirect: prody/meeko/openbabel C code prints to
    # stdout (e.g. "CovalentBuilder> searching for residue..."). With 6
    # parallel workers, those messages contaminate REINVENT4's pipe and
    # break json.loads. Redirect fd 1 -> fd 2 once per worker process.
    if not getattr(_embed_and_score, "_redirected", False):
        os.dup2(2, 1)
        _embed_and_score._redirected = True
    idx, smi = args
    if not isinstance(smi, str) or not smi:
        return idx, float("nan")
    # Strip control-token prefix if present (RL passes raw SMILES typically, but
    # be defensive)
    for tok in ("[ACRYLAMIDE]", "[CHLOROACETAMIDE]", "[VINYL_SULFONAMIDE]", "[EPOXIDE]"):
        if smi.startswith(tok):
            smi = smi[len(tok):]
            break
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return idx, float("nan")
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=20) != 0:
            return idx, float("nan")
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=100)
        except Exception:
            pass
        lig_pdbqt = WORK_DIR / f"mol_{idx:05d}_{os.getpid()}_cov.pdbqt"
        prep = build_cov_tethered_pdbqt(m, smi, lig_pdbqt)
        if not prep["ok"]:
            try: lig_pdbqt.unlink(missing_ok=True)
            except Exception: pass
            return idx, float("nan")
        dock = vina_dock(lig_pdbqt, None, mode="score_only", threads=1)
        try: lig_pdbqt.unlink(missing_ok=True)
        except Exception: pass
        if not dock["ok"] or dock.get("score") is None:
            return idx, float("nan")
        return idx, float(dock["score"])
    except Exception:
        return idx, float("nan")


def main():
    # Preflight: explicit meeko import so missing-env errors are loud not silent
    # (QA agent flagged: implicit import via run_covalent_docking can swallow
    # ImportError and produce all-NaN scores).
    try:
        from meeko import CovalentBuilder  # noqa: F401
    except ImportError as e:
        print(f"[vina-scorer] FATAL: meeko not installed: {e}", file=sys.stderr, flush=True)
        print(json.dumps({"version": 1, "payload": {"vina_score": []}}))
        sys.exit(1)

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    n = len(smiles_list)
    if n == 0:
        print(json.dumps({"version": 1, "payload": {"vina_score": []}}))
        return

    print(f"[vina-scorer] {n} mols, {N_WORKERS} workers", file=sys.stderr, flush=True)
    t0 = time.time()

    tasks = list(enumerate(smiles_list))
    results = [float("nan")] * n
    with Pool(N_WORKERS) as pool:
        for idx, score in pool.imap_unordered(_embed_and_score, tasks, chunksize=2):
            results[idx] = score

    n_ok = sum(1 for s in results if s == s)  # NaN-safe
    dt = time.time() - t0
    print(f"[vina-scorer] done: {n_ok}/{n} scored in {dt:.1f}s "
          f"({dt/max(1,n):.2f}s/mol)", file=sys.stderr, flush=True)

    # Keep NaN as Python float('nan'); REINVENT4 treats NaN as score=0.
    # Inf is invalid → replace with NaN. Force flush so REINVENT4 captures
    # before pipe close (multiprocessing Pool can leave stdout buffered).
    import math
    safe = [s if not math.isinf(s) else float("nan") for s in results]
    payload = json.dumps({"version": 1, "payload": {"vina_score": safe}})
    print(f"[vina-scorer] about to emit JSON ({len(payload)} bytes)", file=sys.stderr, flush=True)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()
    os.fsync(sys.stdout.fileno()) if sys.stdout.isatty() else None
    print("[vina-scorer] JSON emitted", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
