#!/usr/bin/env python3
"""
Run REINVENT4 de novo with graph-distance reward, collect 500 unique molecules.

This is the runner for seq_method experiment #5:
    "warhead-Cβ → hinge-N graph distance" reward

Outputs:
    data/reinvent4_denovo_graph_dist/samples.smi          (N=500 SMILES)
    results/paper_evaluation/seq_method_experiments/graph_distance.json
    /tmp/seq_exp5_graph_distance.md                       (run summary)
"""

import sys
import os
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "reinvent4_configs"

OUT_SAMPLES_DIR = PROJECT_ROOT / "data" / "reinvent4_denovo_graph_dist"
OUT_RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
TMP_SUMMARY     = Path("/tmp/seq_exp5_graph_distance.md")

OUT_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TARGET = 500

# Find conda
CONDA_PATH = "/opt/miniconda3/condabin/conda"
if not os.path.exists(CONDA_PATH):
    for p in ["/opt/miniconda3/bin/conda",
              os.path.expanduser("~/miniconda3/condabin/conda"),
              os.path.expanduser("~/miniconda3/bin/conda")]:
        if os.path.exists(p):
            CONDA_PATH = p
            break
if not os.path.exists(CONDA_PATH):
    raise RuntimeError("Cannot find conda executable")


def resolve_config(template_path: Path, output_path: Path) -> Path:
    content = template_path.read_text()
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
    output_path.write_text(content)
    print(f"Resolved config -> {output_path}")
    return output_path


def run_reinvent(config_path: Path, working_dir: Path, log_file: Path) -> int:
    working_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
           "reinvent", str(config_path), "-d", "cpu"]
    print(f"Running REINVENT4: {' '.join(cmd)}")
    print(f"cwd={working_dir}  log={log_file}")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(working_dir),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            lf.write(line)
        proc.wait()
    print(f"REINVENT4 exit {proc.returncode}  elapsed {(time.time()-t0)/60:.1f} min")
    return proc.returncode


def collect_samples(working_dir: Path):
    """Read REINVENT4 CSV(s), return list of (smiles, score, component_dict)."""
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    rows = []
    seen = set()
    csv_files = sorted(working_dir.glob("denovo_graph_distance*.csv"))
    print(f"Found {len(csv_files)} CSV file(s) in {working_dir}")

    for f in csv_files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        print(f"  {f.name}: {len(df)} rows")

        for _, r in df.iterrows():
            smi = str(r.get("SMILES", "")).strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            rec = {"smiles": can, "step": int(r.get("step", -1))}
            for col in ["Score", "FiLMDelta pIC50 (raw)", "QED (raw)",
                        "graph_distance (raw)", "warhead (raw)"]:
                if col in df.columns:
                    try:
                        v = float(r[col])
                        if v == v:  # not NaN
                            rec[col] = v
                    except Exception:
                        pass
            rows.append(rec)
    print(f"Collected {len(rows)} unique valid SMILES")
    return rows


def rescore_with_graph_distance(smiles_list):
    """Re-score every molecule with graph-distance scorer (deterministic)."""
    scorer = PROJECT_ROOT / "experiments" / "reinvent4_graph_distance_scorer.py"
    input_str = "\n".join(smiles_list)
    proc = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris",
         "python", str(scorer)],
        input=input_str, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print("Graph-distance rescore stderr:\n" + proc.stderr[-2000:])
        return [None] * len(smiles_list)
    # Locate JSON line in stdout
    payload = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if payload is None:
        print("Could not parse scorer JSON output.")
        return [None] * len(smiles_list)
    return payload["payload"]["graph_distance"]


def compute_distances(smiles_list):
    """Independent distance recompute using the scorer's debug path."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from reinvent4_graph_distance_scorer import graph_distance_score
    dists = []
    for smi in smiles_list:
        _, dbg = graph_distance_score(smi, return_debug=True)
        dists.append(dbg["distance"])
    return dists


def diversity_stats(smiles_list):
    """Compute Murcko-scaffold and Tanimoto diversity."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem, DataStructs

    scaffolds = set()
    fps = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.add(Chem.MolToSmiles(scaf))
        except Exception:
            pass
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))

    n_pairs = 0
    sum_sim = 0.0
    # Subsample for speed if needed
    if len(fps) > 200:
        import random
        random.seed(0)
        sample = random.sample(fps, 200)
    else:
        sample = fps
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            sum_sim += DataStructs.TanimotoSimilarity(sample[i], sample[j])
            n_pairs += 1
    mean_sim = sum_sim / max(1, n_pairs)
    return {
        "n_unique_scaffolds": len(scaffolds),
        "mean_pairwise_tanimoto_sample": mean_sim,
    }


def write_summary_md(summary: dict):
    lines = []
    lines.append("# Seq Method Experiment 5: graph-distance reward")
    lines.append("")
    lines.append(f"- Generated unique molecules: **{summary['n_unique']}**")
    lines.append(f"- Target N: {summary['n_target']}")
    lines.append(f"- REINVENT4 steps run: {summary.get('steps_observed', 'n/a')}")
    lines.append(f"- Runtime: {summary['runtime_min']:.1f} min")
    lines.append("")
    lines.append("## Graph-distance distribution (warhead Cβ → nearest hinge N)")
    dh = summary["distance_histogram"]
    lines.append("| bond dist | count |")
    lines.append("|----------:|------:|")
    for k in sorted(dh.keys(), key=lambda x: (x == "none", x if x != "none" else 99)):
        lines.append(f"| {k} | {dh[k]} |")
    lines.append("")
    pg = summary["pass_gate_pct"]
    nh = summary["n_in_window"]
    lines.append(f"- In ideal window (6-8 bonds): **{nh}/{summary['n_unique']} = {pg:.1%}**")
    lines.append(f"- Unique Murcko scaffolds: **{summary['diversity']['n_unique_scaffolds']}**")
    lines.append(f"- Mean pairwise Tanimoto (200-mol sample): {summary['diversity']['mean_pairwise_tanimoto_sample']:.3f}")
    if "score_summary" in summary:
        s = summary["score_summary"]
        lines.append("")
        lines.append("## Composite scoring (REINVENT4 reported)")
        for k, v in s.items():
            if v is None:
                continue
            lines.append(f"- {k}: mean={v.get('mean', float('nan')):.3f}, max={v.get('max', float('nan')):.3f}")
    lines.append("")
    lines.append(f"## QA")
    lines.append(f"- 500 generated: {'PASS' if summary['n_unique'] >= 500 else 'FAIL ('+str(summary['n_unique'])+')'}")
    lines.append(f"- >=50% pass distance gate: {'PASS' if pg >= 0.50 else 'FAIL ('+f'{pg:.1%}'+')'}")
    lines.append(f"- >100 unique scaffolds: {'PASS' if summary['diversity']['n_unique_scaffolds'] > 100 else 'FAIL ('+str(summary['diversity']['n_unique_scaffolds'])+')'}")
    TMP_SUMMARY.write_text("\n".join(lines))
    print(f"Wrote summary -> {TMP_SUMMARY}")


def main():
    template = CONFIGS_DIR / "denovo_graph_distance.toml"
    if not template.exists():
        raise FileNotFoundError(template)

    working_dir = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4" / \
                  "graph_distance"
    working_dir.mkdir(parents=True, exist_ok=True)

    resolved = working_dir / "denovo_graph_distance.toml"
    resolve_config(template, resolved)

    log_file = working_dir / "graph_distance.log"

    t_start = time.time()
    rc = run_reinvent(resolved, working_dir, log_file)
    runtime_min = (time.time() - t_start) / 60.0

    # Collect samples even on non-zero exit (training may finish early)
    samples = collect_samples(working_dir)

    # If short, try extending — but for first pass we just take what we got.
    smiles_list = [s["smiles"] for s in samples]

    # Take first N_TARGET unique
    if len(smiles_list) > N_TARGET:
        smiles_top = smiles_list[:N_TARGET]
        samples_top = samples[:N_TARGET]
    else:
        smiles_top = smiles_list
        samples_top = samples

    # Write samples.smi
    out_smi = OUT_SAMPLES_DIR / "samples.smi"
    out_smi.write_text("\n".join(smiles_top) + "\n")
    print(f"Wrote {len(smiles_top)} SMILES -> {out_smi}")

    # Independent rescore (verifies scorer math)
    print("Recomputing graph-distance scores...")
    scores = rescore_with_graph_distance(smiles_top)
    distances = compute_distances(smiles_top)

    # Histogram of distances
    dh = {}
    n_in_window = 0
    for d in distances:
        key = "none" if d is None else str(d)
        dh[key] = dh.get(key, 0) + 1
        if d is not None and 6 <= d <= 8:
            n_in_window += 1

    n_unique = len(smiles_top)
    pass_gate_pct = n_in_window / max(1, n_unique)

    print("Computing diversity...")
    div = diversity_stats(smiles_top)

    # Score column summary
    score_summary = {}
    for col in ["FiLMDelta pIC50 (raw)", "QED (raw)", "graph_distance (raw)",
                "warhead (raw)", "Score"]:
        vals = [s[col] for s in samples_top if col in s and isinstance(s[col], (int, float))]
        if vals:
            score_summary[col] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "n": len(vals),
            }

    # Steps observed = max(step) across collected rows
    steps_observed = max((s.get("step", 0) for s in samples), default=0)

    summary = {
        "experiment": "seq_method_5_graph_distance",
        "n_unique": n_unique,
        "n_target": N_TARGET,
        "n_collected_total": len(samples),
        "steps_observed": steps_observed,
        "runtime_min": runtime_min,
        "reinvent_returncode": rc,
        "distance_histogram": dh,
        "n_in_window": n_in_window,
        "pass_gate_pct": pass_gate_pct,
        "diversity": div,
        "score_summary": score_summary,
        "config_path": str(resolved),
        "samples_path": str(out_smi),
        "scorer_independent_scores_first10": scores[:10],
        "distances_first10": distances[:10],
    }

    out_json = OUT_RESULTS_DIR / "graph_distance.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote results -> {out_json}")

    write_summary_md(summary)

    print("\n=== Final ===")
    print(f"n_unique = {n_unique}  (target {N_TARGET})")
    print(f"in [6,8] = {n_in_window} = {pass_gate_pct:.1%}")
    print(f"unique scaffolds = {div['n_unique_scaffolds']}")


if __name__ == "__main__":
    main()
