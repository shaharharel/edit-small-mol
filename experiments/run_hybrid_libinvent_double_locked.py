#!/usr/bin/env python3
"""
SEQ-METHOD EXPERIMENT 1: LibInvent with DOUBLE-LOCKED scaffold (warhead + hinge).

Goal: vs the baseline single-locked LibInvent (warhead chassis only),
lock BOTH the acrylamide warhead chassis AND a 3-aminopyridine hinge motif,
forcing the model to place pharmacophores at both anchor positions.

Baseline scaffold (single-locked, [*:1] only):
    C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1

New double-locked scaffold ([*:0] + [*:1]):
    C=CC(=O)N1Cc2cc([*:0])cc(C(=O)Nc3ccc([*:1])nc3)c2C1
                       ^^^^                ^^^^
                       extra R-group       hinge R-group
                       on isoindoline      on aminopyridine
                       benzene (back-pocket) hinge
    Locks: acrylamide warhead + isoindoline + amide + 3-aminopyridine HINGE
    Decorators: [*:0] on the warhead-side benzene, [*:1] off the hinge pyridine

Pipeline:
  1. Sample N_sample candidates from LibInvent prior with dual-anchor scaffold (no RL)
  2. Filter to candidates that retain BOTH warhead AND aminopyridine hinge
  3. Score with FiLMDelta anchor-based pIC50 predictor
  4. Take top 500 by FiLM score, write samples.smi/.sdf + metrics JSON

Run (Mac CPU, ~30 min):
    /opt/miniconda3/envs/quris/bin/python experiments/run_hybrid_libinvent_double_locked.py
"""

import sys
import os
import json
import subprocess
import warnings
import logging
import gc
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog('rdApp.*')

# ── Configuration ─────────────────────────────────────────────────────────────

REINVENT4_ROOT = Path("/Users/shaharharel/Documents/github/REINVENT4")
PRIOR_LIBINVENT = REINVENT4_ROOT / "priors" / "libinvent.prior"
REINVENT_BIN = "/opt/miniconda3/envs/quris/bin/reinvent"
PYTHON = "/opt/miniconda3/envs/quris/bin/python"

OUT_DIR = PROJECT_ROOT / "data" / "reinvent4_libinvent_hybrid_double_locked"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Scaffolds
SCAFFOLD_SINGLE_LOCKED = "C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1"  # baseline (overnight_aigpu_reinvent_jobs.py)
SCAFFOLD_DOUBLE_LOCKED = "C=CC(=O)N1Cc2cc([*:0])cc(C(=O)Nc3ccc([*:1])nc3)c2C1"

WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"
# 3-aminopyridine hinge motif: N-c-c-c-c-n-c (aromatic NH attached to pyridine 3-position)
HINGE_SMARTS = "[NH]c1cccnc1"

# Sampling target — sample 3000, expect ~50-80% to be valid+retain both motifs,
# then top-500 by FiLM score
NUM_SAMPLE = 3000
TOP_N = 500

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: LibInvent sampling ────────────────────────────────────────────────

def write_sampling_config(out_dir: Path) -> Path:
    """Write REINVENT4 LibInvent pure-sampling config."""
    scaffold_file = out_dir / "scaffold.smi"
    scaffold_file.write_text(SCAFFOLD_DOUBLE_LOCKED + "\n")

    config_path = out_dir / "sampling.toml"
    csv_out = out_dir / "raw_samples.csv"

    config = f'''run_type = "sampling"
device = "cpu"

[parameters]
model_file = "{PRIOR_LIBINVENT}"
smiles_file = "{scaffold_file}"
output_file = "{csv_out}"
num_smiles = {NUM_SAMPLE}
unique_molecules = true
randomize_smiles = true
'''
    config_path.write_text(config)
    return config_path


def run_libinvent_sampling(config_path: Path, work_dir: Path) -> Path:
    log(f"Running REINVENT4 LibInvent sampling: {NUM_SAMPLE} mols, dual-anchor scaffold")
    log(f"  scaffold: {SCAFFOLD_DOUBLE_LOCKED}")
    log_file = work_dir / "reinvent_sampling.log"
    cmd = [REINVENT_BIN, str(config_path), "-d", "cpu"]
    start = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(work_dir), stdout=lf,
                              stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = (time.time() - start) / 60
    log(f"  sampling done: exit={proc.returncode}, elapsed={elapsed:.1f} min")
    if proc.returncode != 0:
        log_text = log_file.read_text()
        log(f"  REINVENT4 sampling failed. Tail of log:\n{log_text[-2000:]}")
        raise RuntimeError("REINVENT4 sampling failed")
    return work_dir / "raw_samples.csv"


# ── Step 2: Filter for warhead + hinge retention ──────────────────────────────

def has_warhead(mol: Chem.Mol) -> bool:
    patt = Chem.MolFromSmarts(WARHEAD_SMARTS)
    return mol.HasSubstructMatch(patt)


def has_hinge(mol: Chem.Mol) -> bool:
    patt = Chem.MolFromSmarts(HINGE_SMARTS)
    return mol.HasSubstructMatch(patt)


def filter_and_dedupe(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    log(f"Loaded {len(df)} raw samples")

    smiles_col = next((c for c in ["SMILES", "Smiles", "smiles"] if c in df.columns), None)
    if smiles_col is None:
        raise RuntimeError(f"No SMILES column in {raw_csv}, got cols={df.columns.tolist()}")

    records = []
    seen = set()
    n_invalid = 0
    n_dup = 0
    n_no_warhead = 0
    n_no_hinge = 0
    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            n_dup += 1
            continue
        seen.add(canon)
        w = has_warhead(mol)
        h = has_hinge(mol)
        if not w:
            n_no_warhead += 1
            continue
        if not h:
            n_no_hinge += 1
            continue
        records.append({
            "smiles": canon,
            "scaffold": row.get("Scaffold", ""),
            "r_groups": row.get("R-groups", ""),
            "nll": float(row.get("NLL", 0.0)),
            "warhead_retained": True,
            "hinge_retained": True,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "qed": Descriptors.qed(mol),
            "n_heavy": mol.GetNumHeavyAtoms(),
        })

    log(f"  filter stats: invalid={n_invalid}, dup={n_dup}, no_warhead={n_no_warhead},"
        f" no_hinge={n_no_hinge}, retained={len(records)}")
    return pd.DataFrame(records)


# ── Step 3: FiLMDelta scoring ─────────────────────────────────────────────────

def score_with_filmdelta(df: pd.DataFrame) -> pd.DataFrame:
    """Use experiments/reinvent4_film_scorer.py (cached model) to score pIC50."""
    log(f"Scoring {len(df)} molecules with FiLMDelta anchor-based scorer")

    # We invoke the scorer as a subprocess (same protocol REINVENT4 uses)
    scorer = PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py"
    smis_text = "\n".join(df["smiles"].tolist())
    cmd = [PYTHON, str(scorer)]
    start = time.time()
    proc = subprocess.run(cmd, input=smis_text, capture_output=True, text=True,
                          check=False, timeout=1800)
    elapsed = (time.time() - start) / 60
    log(f"  FiLM scorer done: elapsed={elapsed:.1f} min, returncode={proc.returncode}")
    if proc.returncode != 0:
        log(f"  FiLM scorer stderr (tail):\n{proc.stderr[-2000:]}")
        raise RuntimeError("FiLM scorer failed")

    # Parse stdout — the scorer writes its own logs to stderr; the JSON should be in stdout.
    # In case anything extra leaked in, grab the last JSON object.
    out = proc.stdout.strip()
    # The output is a single JSON line per the scorer's contract
    json_str = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_str = line
            break
    if json_str is None:
        raise RuntimeError(f"Could not parse FiLM scorer output:\n{out[:1000]}")

    payload = json.loads(json_str)
    scores = payload.get("payload", {}).get("pIC50", [])
    if len(scores) != len(df):
        raise RuntimeError(
            f"FiLM scorer returned {len(scores)} scores, expected {len(df)}")

    df = df.copy()
    df["predicted_pIC50"] = scores
    return df


# ── Step 4: Diversity metrics ─────────────────────────────────────────────────

def compute_diversity(df: pd.DataFrame) -> dict:
    """Compute intra-set Tanimoto NN distance using Morgan FP."""
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"].tolist()]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    n = len(fps)
    if n <= 1:
        return {"n": n, "mean_intra_nn_tanimoto": 0.0, "median_intra_nn_tanimoto": 0.0}

    # For each mol, find nearest neighbour Tanimoto
    nn_tans = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i] + fps[i+1:])
        nn_tans.append(max(sims))
    nn_tans = np.array(nn_tans)
    return {
        "n": n,
        "mean_intra_nn_tanimoto": float(np.mean(nn_tans)),
        "median_intra_nn_tanimoto": float(np.median(nn_tans)),
        "p25_intra_nn_tanimoto": float(np.percentile(nn_tans, 25)),
        "p75_intra_nn_tanimoto": float(np.percentile(nn_tans, 75)),
    }


# ── Step 5: SDF writer (with ETKDG conformers) ────────────────────────────────

def write_sdf(df: pd.DataFrame, out_path: Path):
    log(f"Generating ETKDG conformers and writing SDF: {out_path}")
    writer = Chem.SDWriter(str(out_path))
    n_ok = 0
    n_fail = 0
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            n_fail += 1
            continue
        molH = Chem.AddHs(mol)
        # ETKDG conformer embedding
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + i  # deterministic per index
        cid = AllChem.EmbedMolecule(molH, params)
        if cid < 0:
            # Fallback: random coords (still writes a valid SDF entry)
            cid = AllChem.EmbedMolecule(molH, useRandomCoords=True, randomSeed=42 + i)
        if cid >= 0:
            try:
                AllChem.MMFFOptimizeMolecule(molH, maxIters=100)
            except Exception:
                pass
            mol_out = Chem.RemoveHs(molH)
        else:
            # No 3D conformer — write 2D fallback (still counts as 1 SDF entry)
            mol_out = mol
            AllChem.Compute2DCoords(mol_out)
            n_fail += 1

        mol_out.SetProp("_Name", f"hybrid_{i:04d}")
        mol_out.SetProp("predicted_pIC50", f"{row['predicted_pIC50']:.4f}")
        mol_out.SetProp("warhead_retained", "1")
        mol_out.SetProp("hinge_retained", "1")
        mol_out.SetProp("QED", f"{row['qed']:.4f}")
        mol_out.SetProp("MW", f"{row['mw']:.2f}")
        mol_out.SetProp("logP", f"{row['logp']:.3f}")
        mol_out.SetProp("NLL", f"{row['nll']:.3f}")
        writer.write(mol_out)
        n_ok += 1
    writer.close()
    log(f"  SDF written: {n_ok} 3D, {n_fail} 2D-fallback, total={n_ok}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("SEQ-METHOD EXPERIMENT 1: Hybrid LibInvent DOUBLE-locked (warhead + hinge)")
    log("=" * 70)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"REINVENT4 root: {REINVENT4_ROOT}")
    log(f"Output: {OUT_DIR}")
    log(f"Single-locked baseline scaffold:  {SCAFFOLD_SINGLE_LOCKED}")
    log(f"Double-locked NEW scaffold:        {SCAFFOLD_DOUBLE_LOCKED}")
    start_ts = datetime.now()

    if not PRIOR_LIBINVENT.exists():
        raise RuntimeError(f"LibInvent prior missing: {PRIOR_LIBINVENT}")

    # 1. Sample
    config_path = write_sampling_config(OUT_DIR)
    raw_csv = run_libinvent_sampling(config_path, OUT_DIR)

    # 2. Filter
    df_filt = filter_and_dedupe(raw_csv)
    if len(df_filt) < 50:
        log(f"WARNING: only {len(df_filt)} mols retained both motifs; results may be noisy")

    # 3. Score with FiLMDelta
    df_scored = score_with_filmdelta(df_filt)
    df_scored = df_scored.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)

    # 4. Take top N (or all if fewer)
    n_take = min(TOP_N, len(df_scored))
    df_top = df_scored.head(n_take).reset_index(drop=True)
    log(f"Top {n_take} by predicted pIC50:")
    log(f"  max pIC50: {df_top['predicted_pIC50'].max():.3f}")
    log(f"  mean pIC50: {df_top['predicted_pIC50'].mean():.3f}")
    log(f"  median pIC50: {df_top['predicted_pIC50'].median():.3f}")

    # 5. Write samples.smi
    smi_path = OUT_DIR / "samples.smi"
    with open(smi_path, "w") as f:
        f.write("# SMILES\tpredicted_pIC50\tQED\tMW\twarhead\thinge\tnll\n")
        for _, r in df_top.iterrows():
            f.write(f"{r['smiles']}\t{r['predicted_pIC50']:.4f}\t{r['qed']:.4f}\t"
                    f"{r['mw']:.2f}\t1\t1\t{r['nll']:.3f}\n")
    log(f"samples.smi written: {smi_path}")

    # 6. Write SDF
    sdf_path = OUT_DIR / "samples.sdf"
    write_sdf(df_top, sdf_path)

    # 7. Diversity on top-N
    diversity = compute_diversity(df_top)

    # 8. Compute metrics for JSON
    n_total_raw = len(pd.read_csv(raw_csv))
    n_unique_valid = len(df_filt)  # post-dedupe & motif-filter
    warhead_retention_pct = 100.0 * n_unique_valid / max(1, n_total_raw)  # approximate; actual via filter stats
    hinge_retention_pct = warhead_retention_pct  # both required to be retained → same number

    # More precise — recount from raw to break down warhead vs hinge separately
    raw_df = pd.read_csv(raw_csv)
    smi_col = next(c for c in ["SMILES", "Smiles", "smiles"] if c in raw_df.columns)
    n_valid_raw, n_warhead, n_hinge, n_both = 0, 0, 0, 0
    for s in raw_df[smi_col]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        n_valid_raw += 1
        w = has_warhead(m)
        h = has_hinge(m)
        if w: n_warhead += 1
        if h: n_hinge += 1
        if w and h: n_both += 1
    warhead_retention_pct = 100.0 * n_warhead / max(1, n_valid_raw)
    hinge_retention_pct = 100.0 * n_hinge / max(1, n_valid_raw)
    both_retention_pct = 100.0 * n_both / max(1, n_valid_raw)

    metrics = {
        "experiment": "seq_method_exp1_hybrid_libinvent_double_locked",
        "started_at": start_ts.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "scaffold_double_locked": SCAFFOLD_DOUBLE_LOCKED,
        "scaffold_baseline_single_locked": SCAFFOLD_SINGLE_LOCKED,
        "warhead_smarts": WARHEAD_SMARTS,
        "hinge_smarts": HINGE_SMARTS,
        "num_sample_requested": NUM_SAMPLE,
        "n_raw_samples": int(n_total_raw),
        "n_valid_raw": int(n_valid_raw),
        "n_unique_post_dedupe_motif_filter": int(n_unique_valid),
        "n_top_selected": int(n_take),
        "warhead_retention_pct": round(warhead_retention_pct, 2),
        "hinge_retention_pct": round(hinge_retention_pct, 2),
        "both_motifs_retention_pct": round(both_retention_pct, 2),
        "predicted_pIC50": {
            "max": float(df_top["predicted_pIC50"].max()),
            "mean": float(df_top["predicted_pIC50"].mean()),
            "median": float(df_top["predicted_pIC50"].median()),
            "p90": float(df_top["predicted_pIC50"].quantile(0.90)),
            "p10": float(df_top["predicted_pIC50"].quantile(0.10)),
            "n_above_7": int((df_top["predicted_pIC50"] >= 7.0).sum()),
            "n_above_7p5": int((df_top["predicted_pIC50"] >= 7.5).sum()),
            "n_above_8": int((df_top["predicted_pIC50"] >= 8.0).sum()),
        },
        "qed": {
            "mean": float(df_top["qed"].mean()),
            "median": float(df_top["qed"].median()),
        },
        "mw": {
            "mean": float(df_top["mw"].mean()),
            "median": float(df_top["mw"].median()),
            "min": float(df_top["mw"].min()),
            "max": float(df_top["mw"].max()),
        },
        "diversity": diversity,
        "files": {
            "samples_smi": str(smi_path),
            "samples_sdf": str(sdf_path),
            "raw_csv": str(raw_csv),
            "scaffold_file": str(OUT_DIR / "scaffold.smi"),
        },
    }
    out_json = RESULTS_DIR / "hybrid_libinvent_double_locked.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log(f"Metrics → {out_json}")

    # Final SDF entry verification
    sdf_text = sdf_path.read_text()
    n_sdf = sdf_text.count("$$$$")
    log(f"SDF entries (count of '$$$$'): {n_sdf}")

    log("=" * 70)
    log(f"DONE. Total elapsed: {(datetime.now() - start_ts).total_seconds()/60:.1f} min")
    log("=" * 70)
    return metrics


if __name__ == "__main__":
    main()
