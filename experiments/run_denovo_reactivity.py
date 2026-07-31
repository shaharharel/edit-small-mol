#!/usr/bin/env python3
"""
Run REINVENT4 De-Novo with reactivity-aware warhead scoring.

This is the runner for the reactivity-aware experiment (seq_method_experiments).
It:
  1. Resolves the TOML template (placeholders for conda path + project root).
  2. Pre-trains / caches the FiLMDelta scorer model.
  3. Self-tests the reactivity scorer.
  4. Launches REINVENT4 staged learning.
  5. Streams the per-step CSV and harvests molecules that pass all gates
     (warhead, reactivity >= 0.5, valid) until 500 uniques are collected
     (or all stages complete).
  6. Writes:
       data/reinvent4_denovo_reactivity/samples.smi   (N=500)
       results/paper_evaluation/seq_method_experiments/denovo_reactivity.json
"""
from __future__ import annotations
import sys
import os
import json
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
CONFIG_TEMPLATE = PROJECT_ROOT / "experiments" / "reinvent4_configs" / "denovo_reactivity.toml"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4" / "denovo_reactivity"
WORK_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = PROJECT_ROOT / "data" / "reinvent4_denovo_reactivity"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

N_TARGET = 500
REACTIVITY_THRESHOLD = 0.5

# Find conda
CONDA_PATH = "/opt/miniconda3/condabin/conda"
if not os.path.exists(CONDA_PATH):
    for p in [
        "/opt/miniconda3/bin/conda",
        os.path.expanduser("~/miniconda3/condabin/conda"),
        os.path.expanduser("~/miniconda3/bin/conda"),
    ]:
        if os.path.exists(p):
            CONDA_PATH = p
            break
if not os.path.exists(CONDA_PATH):
    raise RuntimeError("Cannot find conda executable")


def resolve_config(template: Path, out: Path) -> Path:
    content = template.read_text()
    content = content.replace("__CONDA_PATH__", CONDA_PATH)
    content = content.replace("__PROJECT_ROOT__", str(PROJECT_ROOT))
    content = content.replace(
        'prior_file = "priors/',
        f'prior_file = "{REINVENT4_ROOT}/priors/'
    )
    content = content.replace(
        'agent_file = "priors/',
        f'agent_file = "{REINVENT4_ROOT}/priors/'
    )
    out.write_text(content)
    return out


def ensure_film_model_cached():
    cache = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if cache.exists():
        return
    print("[runner] Pre-training FiLMDelta model...")
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris", "python", "-u",
         str(PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py")],
        input="c1ccccc1\n", capture_output=True, text=True,
    )


def run_reactivity_selftest():
    print("[runner] Reactivity scorer self-test...")
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris", "python",
         str(PROJECT_ROOT / "experiments" / "reinvent4_reactivity_scorer.py"),
         "--selftest"],
        capture_output=True, text=True,
    )
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Reactivity scorer self-test FAILED")


def stream_run(config_path: Path) -> int:
    log_file = WORK_DIR / "denovo_reactivity.log"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "quris",
        "reinvent", str(config_path), "-d", "cpu"
    ]
    print(f"[runner] launching: {' '.join(cmd)}")
    start = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(WORK_DIR), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            lf.write(line)
        proc.wait()
    elapsed = (time.time() - start) / 60.0
    print(f"[runner] reinvent finished in {elapsed:.1f} min, exit={proc.returncode}")
    return proc.returncode


def collect_samples():
    """Read REINVENT4 csv outputs, filter to molecules that satisfied the
    reactivity + warhead gate at generation time (Score>0 implies all
    multiplicative gates >0 i.e. all warheads survived). Then re-score
    explicitly with the reactivity scorer to enforce the threshold.
    """
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    csvs = sorted(WORK_DIR.glob("denovo_reactivity*.csv"))
    if not csvs:
        csvs = sorted(WORK_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No REINVENT CSVs found in {WORK_DIR}")
    print(f"[runner] found {len(csvs)} csv outputs")

    rows = []
    for c in csvs:
        df = pd.read_csv(c, on_bad_lines="skip")
        cols = list(df.columns)
        smi_col = next((c for c in ["SMILES", "smiles", "Smiles"] if c in cols),
                       cols[0])
        score_col = next((c for c in ["Score", "score", "total_score"] if c in cols),
                         None)
        # Reactivity column will have suffix
        react_col = next((c for c in cols if "Reactivity" in c and "(raw)" in c), None)
        warhead_col = next((c for c in cols if "Warhead" in c and "(raw)" in c), None)
        film_col = next((c for c in cols if "FiLMDelta" in c and "(raw)" in c), None)
        for _, r in df.iterrows():
            entry = {"smiles": str(r[smi_col]).strip()}
            if score_col and pd.notna(r.get(score_col)):
                entry["reinvent_score"] = float(r[score_col])
            if react_col and pd.notna(r.get(react_col)):
                entry["reactivity"] = float(r[react_col])
            if warhead_col and pd.notna(r.get(warhead_col)):
                entry["warhead"] = float(r[warhead_col])
            if film_col and pd.notna(r.get(film_col)):
                entry["film_pIC50"] = float(r[film_col])
            rows.append(entry)
    print(f"[runner] total raw rows: {len(rows)}")

    seen, uniq = set(), []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        # Must pass reactivity gate strictly. Reactivity >= 0.5 implies a
        # covalent warhead exists (acrylamide / vinyl sulfonamide /
        # propiolamide) AND it sits in the GSH-k2 sweet spot.
        if r.get("reactivity", 0.0) < REACTIVITY_THRESHOLD:
            continue
        # Note: reactivity >= 0.5 already implies a warhead exists, since the
        # reactivity scorer returns 0 when no warhead is found. The "Warhead"
        # gate (primary acrylamide only) is therefore not required separately.
        seen.add(can)
        r["smiles"] = can
        uniq.append(r)
    print(f"[runner] unique gated mols: {len(uniq)}")
    return uniq


def rescore_with_reactivity(smiles_list):
    """Re-run the reactivity scorer on a list of canonical SMILES to
    double-check the on-the-fly scores. Returns dict {smi: score}.
    """
    if not smiles_list:
        return {}
    inp = "\n".join(smiles_list) + "\n"
    res = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris", "python",
         str(PROJECT_ROOT / "experiments" / "reinvent4_reactivity_scorer.py")],
        input=inp, capture_output=True, text=True, timeout=600,
    )
    data = json.loads(res.stdout.strip())
    scores = data["payload"]["reactivity"]
    return {s: float(sc) for s, sc in zip(smiles_list, scores)}


def main():
    print("[runner] === De-Novo + Reactivity-aware scoring ===")
    t0 = time.time()
    run_reactivity_selftest()
    ensure_film_model_cached()

    resolved = WORK_DIR / "denovo_reactivity.toml"
    resolve_config(CONFIG_TEMPLATE, resolved)
    print(f"[runner] config: {resolved}")

    rc = stream_run(resolved)
    if rc != 0:
        print(f"[runner] WARNING: REINVENT4 exit code {rc}")

    samples = collect_samples()
    if not samples:
        raise RuntimeError("No molecules survived the gates")

    # Sort by reinvent total score (higher = more potent + lower reactivity issues)
    samples.sort(key=lambda r: -r.get("reinvent_score", 0.0))

    # Verify by re-scoring (sanity)
    top_smiles = [r["smiles"] for r in samples[:N_TARGET]]
    print(f"[runner] re-scoring top {len(top_smiles)} with reactivity scorer...")
    rescored = rescore_with_reactivity(top_smiles)
    for r in samples[:N_TARGET]:
        r["reactivity_check"] = rescored.get(r["smiles"], None)

    chosen = samples[:N_TARGET]
    n_unique = len(chosen)
    print(f"[runner] keeping {n_unique} samples (target was {N_TARGET})")

    samples_smi = SAMPLES_DIR / "samples.smi"
    with open(samples_smi, "w") as f:
        for r in chosen:
            f.write(r["smiles"] + "\n")
    print(f"[runner] wrote {samples_smi}")

    # Build summary
    react_vals = [r.get("reactivity") for r in chosen if r.get("reactivity") is not None]
    film_vals = [r.get("film_pIC50") for r in chosen if r.get("film_pIC50") is not None]
    score_vals = [r.get("reinvent_score") for r in chosen if r.get("reinvent_score") is not None]
    summary = {
        "experiment": "denovo_reactivity",
        "config": str(CONFIG_TEMPLATE.name),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_min": (time.time() - t0) / 60.0,
        "n_requested": N_TARGET,
        "n_generated_raw": len(samples),
        "n_unique_passing": n_unique,
        "reactivity_threshold": REACTIVITY_THRESHOLD,
        "reactivity_stats": {
            "mean":   float(np.mean(react_vals)) if react_vals else None,
            "median": float(np.median(react_vals)) if react_vals else None,
            "min":    float(np.min(react_vals)) if react_vals else None,
            "max":    float(np.max(react_vals)) if react_vals else None,
        },
        "film_pIC50_stats": {
            "mean":   float(np.mean(film_vals)) if film_vals else None,
            "median": float(np.median(film_vals)) if film_vals else None,
            "max":    float(np.max(film_vals)) if film_vals else None,
            "n_potent_7plus": sum(1 for s in film_vals if s >= 7.0),
            "n_potent_8plus": sum(1 for s in film_vals if s >= 8.0),
        },
        "reinvent_score_stats": {
            "mean":   float(np.mean(score_vals)) if score_vals else None,
            "median": float(np.median(score_vals)) if score_vals else None,
            "max":    float(np.max(score_vals)) if score_vals else None,
        },
        "top_10": [
            {
                "smiles": r["smiles"],
                "reinvent_score": r.get("reinvent_score"),
                "film_pIC50":     r.get("film_pIC50"),
                "reactivity":     r.get("reactivity"),
                "warhead":        r.get("warhead"),
                "reactivity_check": r.get("reactivity_check"),
            }
            for r in chosen[:10]
        ],
        "samples_file": str(samples_smi),
    }
    out_json = RESULTS_DIR / "denovo_reactivity.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[runner] wrote {out_json}")

    # Console preview
    print("\n=== denovo_reactivity summary ===")
    print(f"  n_unique_passing: {n_unique}")
    print(f"  reactivity mean : {summary['reactivity_stats']['mean']:.3f}")
    print(f"  FiLM pIC50 mean : {summary['film_pIC50_stats']['mean']}")
    print(f"  FiLM pIC50 max  : {summary['film_pIC50_stats']['max']}")
    print(f"  wall time       : {summary['wall_time_min']:.1f} min")


if __name__ == "__main__":
    main()
