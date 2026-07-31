"""EXP7-v2 LO eval — mol2mol prior driver for the 54-pair v2 benchmark (runs ON a100-b).

Two conditions per pair:
  - mol2mol_baseline: vanilla mol2mol prior, NO RL, sample 10K mols with anchor as seed
  - mol2mol_RL:       vanilla mol2mol prior + composite RL reward (50 steps), then sample 10K

REST server per target must be running (use exp6_rest_server.py).

Cohort CSVs land at: data/exp7_v2_benchmark/_rl/<pair_id>_<condition>/sampled.csv

Resumable.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = "/home/shaharh_quris_ai/edit-small-mol"
PHASE1_ROOT = f"{PROJECT_ROOT}/data/exp7_v2_benchmark/_phase1"
RL_OUT_ROOT = f"{PROJECT_ROOT}/data/exp7_v2_benchmark/_rl"
TOML_OUT_ROOT = f"{PROJECT_ROOT}/data/exp7_v2_benchmark/_tomls"
PRIOR_FILE_DEFAULT = f"{PROJECT_ROOT}/models/reinvent4_mol2mol_prior.prior"

PORTS = {
    "SOS1":       8101, "KRAS_G12D":  8102, "KRAS_G12C":  8103,
    "CDK7":       8104, "BCL2":       8105, "FGFR1":      8106,
    "EGFR_T790M": 8107, "BTK":        8108, "BTK_Cys481": 8109,
}

NUM_SMILES = 10000


def render_rl_toml(target_key, pair_id, condition, seed_file, out_dir, prior_file,
                   batch_size=8, n_steps=50, lr=1e-4, sigma=128):
    port = PORTS[target_key]
    chkpt = f"{out_dir}/agent.prior"
    tb_dir = f"{out_dir}/tb"
    csv_prefix = f"{out_dir}/learn"
    cell_id = f"{pair_id}_{condition}"
    toml = f"""# EXP7v2 LO {cell_id}
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "{tb_dir}"
json_out_config = "{TOML_OUT_ROOT}/{cell_id}.json"

[parameters]
summary_csv_prefix = "{csv_prefix}"
use_checkpoint = false
purge_memories = true

prior_file = "{prior_file}"
agent_file = "{prior_file}"
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
max_score = 0.7
max_steps = {n_steps}
min_steps = {n_steps}

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "exp7v2_film_composite"
weight = 1.0
params.server_url = "http://127.0.0.1"
params.server_port = {port}
params.server_endpoint = "score"
params.predictor_id = "composite"
params.predictor_version = "exp7v2_v1"
"""
    toml_path = f"{TOML_OUT_ROOT}/{cell_id}.toml"
    Path(toml_path).write_text(toml)
    return toml_path


def render_sample_toml(target_key, pair_id, condition, agent_file, seed_file, out_dir, num_smiles=NUM_SMILES):
    cell_id = f"{pair_id}_{condition}_sample"
    out_csv = f"{out_dir}/sampled.csv"
    toml = f"""# EXP7v2 sample {cell_id}
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
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            proc = subprocess.run(
                ["reinvent", toml_path],
                stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_sec,
                env={**os.environ, "PATH": f"{os.environ.get('PATH','')}:/home/shaharh_quris_ai/miniconda3/envs/quris/bin"}
            )
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    return ok, time.time() - t0


def build_seed_file(anchor_smi, out_dir):
    seed_path = f"{out_dir}/seed.smi"
    Path(seed_path).write_text(anchor_smi.strip() + "\n")
    return seed_path


def process_pair_cell(pair: dict, condition: str, prior_file: str = PRIOR_FILE_DEFAULT, force: bool = False) -> dict:
    pair_id = pair["pair_id"]
    target_key = pair["target_key"]
    anchor_smi = pair["anchor_smiles"]

    cell_dir = f"{RL_OUT_ROOT}/{pair_id}_{condition}"
    cohort_path = f"{cell_dir}/sampled.csv"
    Path(cell_dir).mkdir(parents=True, exist_ok=True)
    Path(TOML_OUT_ROOT).mkdir(parents=True, exist_ok=True)

    if not force and Path(cohort_path).exists() and Path(cohort_path).stat().st_size > 5000:
        return {"status": "skip", "reason": "cohort_already_exists", "cohort_path": cohort_path}

    seed_file = build_seed_file(anchor_smi, cell_dir)

    if condition == "mol2mol_RL":
        rl_toml = render_rl_toml(target_key, pair_id, condition, seed_file, cell_dir, prior_file)
        rl_log = f"{cell_dir}/rl.log"
        ok, dt_rl = run_reinvent(rl_toml, rl_log, timeout_sec=1800)
        if not ok:
            rl_toml = render_rl_toml(target_key, pair_id, condition, seed_file, cell_dir, prior_file, batch_size=2)
            ok, dt_rl = run_reinvent(rl_toml, rl_log, timeout_sec=2400)
        if not ok:
            return {"status": "fail", "reason": "rl_failed", "log": rl_log, "dt_rl": dt_rl}
        agent_file = f"{cell_dir}/agent.prior"
        if not Path(agent_file).exists():
            return {"status": "fail", "reason": "no_agent_checkpoint", "dt_rl": dt_rl}
        sample_toml = render_sample_toml(target_key, pair_id, condition, agent_file, seed_file, cell_dir)
        sample_log = f"{cell_dir}/sample.log"
        ok2, dt_s = run_reinvent(sample_toml, sample_log, timeout_sec=1200)
        if not ok2:
            return {"status": "fail", "reason": "sample_failed", "dt_rl": dt_rl, "dt_sample": dt_s}
        return {"status": "ok", "condition": condition, "dt_rl": dt_rl, "dt_sample": dt_s, "cohort_path": cohort_path}
    elif condition == "mol2mol_baseline":
        sample_toml = render_sample_toml(target_key, pair_id, condition, prior_file, seed_file, cell_dir)
        sample_log = f"{cell_dir}/sample.log"
        ok, dt_s = run_reinvent(sample_toml, sample_log, timeout_sec=800)
        if not ok:
            return {"status": "fail", "reason": "baseline_sample_failed", "dt_sample": dt_s}
        return {"status": "ok", "condition": condition, "dt_sample": dt_s, "cohort_path": cohort_path}
    else:
        return {"status": "fail", "reason": f"unknown condition: {condition}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_results", required=True)
    ap.add_argument("--prior_file", default=PRIOR_FILE_DEFAULT)
    ap.add_argument("--budget_sec", type=int, default=28800)
    args = ap.parse_args()

    queue = pd.read_csv(args.pairs_csv).to_dict("records")
    results = []
    t_start = time.time()
    for cell in queue:
        elapsed = time.time() - t_start
        if elapsed > args.budget_sec:
            print(f"[exp7v2-mol2mol] BUDGET HIT after {elapsed/60:.1f} min. Stopping.", flush=True)
            break
        t0 = time.time()
        out = process_pair_cell(cell, cell["condition"], prior_file=args.prior_file)
        out["pair_id"] = cell["pair_id"]
        out["target_key"] = cell["target_key"]
        out["condition"] = cell["condition"]
        out["dt_total"] = time.time() - t0
        results.append(out)
        pd.DataFrame(results).to_csv(args.out_results, index=False)
        elapsed = time.time() - t_start
        print(f"[exp7v2-mol2mol] {cell['pair_id']}_{cell['condition']} -> {out.get('status','?')} "
              f"({out['dt_total']:.0f}s, elapsed={elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
