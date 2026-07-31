"""EXP7 LO eval — Per-pair RL driver (runs ON a100-b).

For each pair from data/exp7_lo_benchmark/_phase1/*/anchors/*_b_pool.csv:
  - Strategy A: seed=anchor only, 50-step DAP RL, sample 5500
  - Strategy B: seed=100-anchor pool, 50-step DAP RL, sample 5500
  - Plus 3 baselines (NO RL, just sampling from prior with anchor as seed):
      - "prior_anchor": cov FT prior sampled with anchor as single seed
      - "prior_pool":   cov FT prior sampled with 100-anchor B pool as seed
      - "prior_rand":   cov FT prior sampled with randomize_smiles=true to enumerate random analogs

A REST server per target must be running (use exp6_rest_server.py).

Cohort CSVs land at: data/exp7_lo_benchmark/_rl/<pair_id>_<strategy>_cohort.csv
Where strategy in {A, B, baseline_prior_anchor, baseline_prior_pool, baseline_prior_rand}.

Resumable: skips if cohort CSV already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Path conventions (a100-b paths)
PROJECT_ROOT = "/home/shaharh_quris_ai/edit-small-mol"
PHASE1_ROOT = f"{PROJECT_ROOT}/data/exp7_lo_benchmark/_phase1"
RL_OUT_ROOT = f"{PROJECT_ROOT}/data/exp7_lo_benchmark/_rl"
TOML_OUT_ROOT = f"{PROJECT_ROOT}/data/exp7_lo_benchmark/_tomls"
PRIOR_FILE = f"{PROJECT_ROOT}/models/reinvent4_mol2mol_covalent_ft.prior"

# Port allocation per target (server already running)
PORTS = {
    "egfr_t790m": 8088,
    "btk":        8089,
    "jak3":       8090,
    "her2":       8091,
    "fgfr":       8092,
}


def render_rl_toml(target_key, pair_id, strategy, seed_file, out_dir, batch_size=8, n_steps=50, lr=1e-4, sigma=128):
    port = PORTS[target_key]
    chkpt = f"{out_dir}/agent.prior"
    tb_dir = f"{out_dir}/tb"
    csv_prefix = f"{out_dir}/learn"
    cell_id = f"{pair_id}_{strategy}"
    toml = f"""# EXP7 LO {cell_id}
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "{tb_dir}"
json_out_config = "{TOML_OUT_ROOT}/{cell_id}.json"

[parameters]
summary_csv_prefix = "{csv_prefix}"
use_checkpoint = false
purge_memories = true

prior_file = "{PRIOR_FILE}"
agent_file = "{PRIOR_FILE}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = {batch_size}
unique_sequences = false
randomize_smiles = true
tb_isim = false

[learning_strategy]
type = "dap"
sigma = {sigma}
rate = {lr}

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4
penalty_multiplier = 0.5

[[stage]]
chkpt_file = "{chkpt}"
termination = "simple"
max_score = 1.0
max_steps = {n_steps}
min_steps = {n_steps}

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "exp7_film_composite"
weight = 1.0
params.server_url = "http://127.0.0.1"
params.server_port = {port}
params.server_endpoint = "score"
params.predictor_id = "composite"
params.predictor_version = "exp7_v1"
"""
    toml_path = f"{TOML_OUT_ROOT}/{cell_id}.toml"
    Path(toml_path).write_text(toml)
    return toml_path


def render_sample_toml(target_key, pair_id, strategy, agent_file, seed_file, out_dir, num_smiles=5500):
    cell_id = f"{pair_id}_{strategy}_sample"
    out_csv = f"{out_dir}/sampled.csv"
    toml = f"""# EXP7 sample {cell_id}
run_type = "sampling"
device = "cuda:0"
json_out_config = "{TOML_OUT_ROOT}/{cell_id}.json"

[parameters]
model_file = "{agent_file}"
output_file = "{out_csv}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
num_smiles = {num_smiles}
unique_molecules = false
randomize_smiles = true
"""
    toml_path = f"{TOML_OUT_ROOT}/{cell_id}.toml"
    Path(toml_path).write_text(toml)
    return toml_path


def run_reinvent(toml_path, log_path, timeout_sec=2700):
    """Run reinvent <toml>. Returns (ok, dt)."""
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            ["reinvent", toml_path],
            stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_sec,
            env={**os.environ, "PATH": f"{os.environ.get('PATH','')}:/home/shaharh_quris_ai/miniconda3/envs/quris/bin"}
        )
    return proc.returncode == 0, time.time() - t0


def build_seed_file(pair_id, strategy, anchor_smi, pool_csv, out_dir):
    """Returns path to a seed.smi for REINVENT4."""
    seed_path = f"{out_dir}/seed.smi"
    if strategy == "A":
        Path(seed_path).write_text(anchor_smi.strip() + "\n")
    elif strategy == "B":
        # 100-anchor pool
        df = pd.read_csv(pool_csv)
        smis = df["smiles"].astype(str).str.strip().tolist()
        Path(seed_path).write_text("\n".join(smis) + "\n")
    elif strategy in ("baseline_prior_anchor", "baseline_prior_rand"):
        # Anchor only — for sampling from prior
        Path(seed_path).write_text(anchor_smi.strip() + "\n")
    elif strategy == "baseline_prior_pool":
        df = pd.read_csv(pool_csv)
        smis = df["smiles"].astype(str).str.strip().tolist()
        Path(seed_path).write_text("\n".join(smis) + "\n")
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return seed_path


def process_pair_cell(pair: dict, strategy: str, force: bool = False) -> dict:
    """Run one (pair, strategy) cell. Returns metrics dict."""
    pair_id = pair["pair_id"]
    target_key = pair["target_key"]
    anchor_smi = pair["anchor_smiles"]

    cell_dir = f"{RL_OUT_ROOT}/{pair_id}_{strategy}"
    cohort_path = f"{cell_dir}/sampled.csv"
    Path(cell_dir).mkdir(parents=True, exist_ok=True)
    Path(TOML_OUT_ROOT).mkdir(parents=True, exist_ok=True)

    if not force and Path(cohort_path).exists() and Path(cohort_path).stat().st_size > 1000:
        return {"status": "skip", "reason": "cohort_already_exists", "cohort_path": cohort_path}

    # Resolve pool csv (only needed for B-style strategies)
    pool_csv = f"{PHASE1_ROOT}/{target_key}/anchors/{pair_id}_b_pool.csv"

    # Build seed
    try:
        seed_file = build_seed_file(pair_id, strategy, anchor_smi, pool_csv, cell_dir)
    except Exception as e:
        return {"status": "fail", "reason": f"seed_build: {e}"}

    if strategy in ("A", "B"):
        # RL stage
        rl_toml = render_rl_toml(target_key, pair_id, strategy, seed_file, cell_dir)
        rl_log = f"{cell_dir}/rl.log"
        ok, dt_rl = run_reinvent(rl_toml, rl_log, timeout_sec=1800)
        if not ok:
            # Retry at bs=2 (OOM defense)
            rl_toml = render_rl_toml(target_key, pair_id, strategy, seed_file, cell_dir, batch_size=2)
            ok, dt_rl = run_reinvent(rl_toml, rl_log, timeout_sec=2400)
        if not ok:
            return {"status": "fail", "reason": "rl_failed", "log": rl_log, "dt_rl": dt_rl}
        agent_file = f"{cell_dir}/agent.prior"
        if not Path(agent_file).exists():
            return {"status": "fail", "reason": "no_agent_checkpoint", "dt_rl": dt_rl}
        # Sample stage
        sample_toml = render_sample_toml(target_key, pair_id, strategy, agent_file, seed_file, cell_dir)
        sample_log = f"{cell_dir}/sample.log"
        ok2, dt_s = run_reinvent(sample_toml, sample_log, timeout_sec=900)
        if not ok2:
            return {"status": "fail", "reason": "sample_failed", "dt_rl": dt_rl, "dt_sample": dt_s}
        return {"status": "ok", "strategy": strategy, "dt_rl": dt_rl, "dt_sample": dt_s,
                "cohort_path": cohort_path}
    else:
        # Baseline: just sample from prior with the seed
        sample_toml = render_sample_toml(target_key, pair_id, strategy, PRIOR_FILE, seed_file, cell_dir)
        sample_log = f"{cell_dir}/sample.log"
        ok, dt_s = run_reinvent(sample_toml, sample_log, timeout_sec=600)
        if not ok:
            return {"status": "fail", "reason": "baseline_sample_failed", "dt_sample": dt_s}
        return {"status": "ok", "strategy": strategy, "dt_sample": dt_s, "cohort_path": cohort_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: pair_id, target_key, anchor_smiles, drug_smiles, strategy")
    ap.add_argument("--out_results", required=True)
    ap.add_argument("--budget_sec", type=int, default=25200, help="Total wallclock seconds (default 7h)")
    args = ap.parse_args()

    queue = pd.read_csv(args.pairs_csv).to_dict("records")
    results = []
    t_start = time.time()
    for cell in queue:
        elapsed = time.time() - t_start
        if elapsed > args.budget_sec:
            print(f"[exp7-driver] BUDGET HIT after {elapsed/60:.1f} min. Stopping.")
            break
        t0 = time.time()
        out = process_pair_cell(cell, cell["strategy"])
        out["pair_id"] = cell["pair_id"]
        out["target_key"] = cell["target_key"]
        out["strategy"] = cell["strategy"]
        out["dt_total"] = time.time() - t0
        results.append(out)
        # incremental save
        pd.DataFrame(results).to_csv(args.out_results, index=False)
        elapsed = time.time() - t_start
        print(f"[exp7-driver] {cell['pair_id']}_{cell['strategy']} -> {out.get('status','?')} "
              f"({out['dt_total']:.0f}s, total_elapsed={elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
