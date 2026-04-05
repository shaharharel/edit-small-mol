#!/usr/bin/env python3
"""Benchmark LibInvent: MPS vs CPU, different batch sizes.

Runs 5 steps with each configuration, measures time/step and peak memory.
Then launches the full production run with the best safe config.

Usage:
    conda run --no-capture-output -n quris python -u experiments/benchmark_libinvent.py
"""
import os
import sys
import time
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "reinvent4_configs"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_mps"
BENCH_DIR = RESULTS_DIR / "benchmark"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

CONDA_PATH = "/opt/miniconda3/condabin/conda"
if not os.path.exists(CONDA_PATH):
    for p in ["/opt/miniconda3/bin/conda",
              os.path.expanduser("~/miniconda3/condabin/conda"),
              os.path.expanduser("~/miniconda3/bin/conda")]:
        if os.path.exists(p):
            CONDA_PATH = p
            break

PRIORS_DIR = REINVENT4_ROOT / "priors"

BENCH_STEPS = 5


def get_reference_smiles(n=5):
    actives_file = PROJECT_ROOT / "data" / "zap70_top_actives_clean.smi"
    if not actives_file.exists():
        actives_file = PROJECT_ROOT / "data" / "zap70_top_actives.smi"
    return [line.strip() for line in open(actives_file) if line.strip()][:n]


def write_bench_config(device, batch_size, tag):
    """Write a benchmark config for LibInvent with given device and batch_size."""
    ref_smiles = get_reference_smiles(5)

    config = f"""# LibInvent Benchmark: {tag}
run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_bench_{tag}"
json_out_config = "_bench_{tag}.json"

[parameters]
summary_csv_prefix = "bench_{tag}"
use_checkpoint = false
purge_memories = true

prior_file = "{PRIORS_DIR / 'libinvent.prior'}"
agent_file = "{PRIORS_DIR / 'libinvent.prior'}"
smiles_file = "{PROJECT_ROOT / 'data' / 'zap70_scaffolds.smi'}"

batch_size = {batch_size}
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 15
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "bench_{tag}.chkpt"
termination = "simple"
max_score = 0.99
min_steps = {BENCH_STEPS}
max_steps = {BENCH_STEPS}

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT / 'experiments' / 'reinvent4_film_scorer.py'}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.2

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]
[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular weight"
weight = 0.05
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
"""
    config_path = BENCH_DIR / f"bench_{tag}.toml"
    config_path.write_text(config)
    return config_path


def run_benchmark(config_path, tag, device):
    """Run a benchmark config, return timing and memory info."""
    work_dir = BENCH_DIR / tag
    work_dir.mkdir(parents=True, exist_ok=True)

    device_flag = device if device != "cpu" else "cpu"
    cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
           "reinvent", str(config_path), "-d", device_flag]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {tag} (device={device}, {BENCH_STEPS} steps)")
    print(f"{'='*60}")

    start = time.time()
    step_times = []
    last_step_time = start
    peak_rss_mb = 0

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(work_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            # Track step times
            if "Step:" in line and "Score:" in line:
                now = time.time()
                step_times.append(now - last_step_time)
                last_step_time = now

            # Track peak memory of the process
            try:
                result = subprocess.run(
                    ["ps", "-p", str(proc.pid), "-o", "rss="],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    rss_kb = int(result.stdout.strip())
                    rss_mb = rss_kb / 1024
                    if rss_mb > peak_rss_mb:
                        peak_rss_mb = rss_mb
            except Exception:
                pass

        proc.wait()
        elapsed = time.time() - start

        return {
            "tag": tag,
            "device": device,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "step_times": step_times,
            "avg_step_sec": sum(step_times) / len(step_times) if step_times else 0,
            "peak_rss_mb": peak_rss_mb,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED: {e}")
        return {
            "tag": tag, "device": device, "exit_code": -1,
            "elapsed_sec": elapsed, "step_times": [], "avg_step_sec": 0,
            "peak_rss_mb": peak_rss_mb, "error": str(e),
        }


def write_production_config(device, batch_size):
    """Write the production LibInvent config with chosen device and batch size."""
    ref_smiles = get_reference_smiles(5)

    config = f"""# REINVENT4: LibInvent R-Group — Production Run
# device={device}, batch_size={batch_size}, purge_memories=true
run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_libinvent_prod"
json_out_config = "_libinvent_prod.json"

[parameters]
summary_csv_prefix = "libinvent_prod"
use_checkpoint = false
purge_memories = true

prior_file = "{PRIORS_DIR / 'libinvent.prior'}"
agent_file = "{PRIORS_DIR / 'libinvent.prior'}"
smiles_file = "{PROJECT_ROOT / 'data' / 'zap70_scaffolds.smi'}"

batch_size = {batch_size}
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 15
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "libinvent_prod_stage1.chkpt"
termination = "simple"
max_score = 0.70
min_steps = 50
max_steps = 300

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT / 'experiments' / 'reinvent4_film_scorer.py'}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.2

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]
[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular weight"
weight = 0.05
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
"""
    config_path = RESULTS_DIR / "libinvent" / "libinvent_prod.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    return config_path


def main():
    print("LibInvent Benchmark: MPS vs CPU")
    print(f"Steps per benchmark: {BENCH_STEPS}")
    print(f"Results dir: {BENCH_DIR}\n")

    # Benchmark configurations: (device, batch_size)
    configs = [
        ("mps", 8),
        ("mps", 16),
        ("cpu", 8),
        ("cpu", 16),
        ("cpu", 32),
    ]

    results = []
    for device, bs in configs:
        tag = f"{device}_bs{bs}"
        config_path = write_bench_config(device, bs, tag)
        result = run_benchmark(config_path, tag, device)
        results.append(result)

        print(f"\n  Result: {tag}")
        print(f"    Exit code: {result['exit_code']}")
        print(f"    Total time: {result['elapsed_sec']:.1f}s")
        print(f"    Avg step: {result['avg_step_sec']:.1f}s")
        print(f"    Peak RSS: {result['peak_rss_mb']:.0f} MB")
        print()

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Status':<10} {'Avg/step':<12} {'Total':<12} {'Peak RSS':<12}")
    print("-" * 70)
    for r in results:
        status = "OK" if r["exit_code"] == 0 else f"FAIL({r['exit_code']})"
        avg = f"{r['avg_step_sec']:.1f}s" if r['avg_step_sec'] > 0 else "—"
        total = f"{r['elapsed_sec']:.1f}s"
        rss = f"{r['peak_rss_mb']:.0f} MB" if r['peak_rss_mb'] > 0 else "—"
        print(f"{r['tag']:<20} {status:<10} {avg:<12} {total:<12} {rss:<12}")

    # Save results
    out_file = BENCH_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")

    # Pick best safe config
    ok_results = [r for r in results if r["exit_code"] == 0 and r["avg_step_sec"] > 0]
    if ok_results:
        best = min(ok_results, key=lambda r: r["avg_step_sec"])
        print(f"\nFastest successful config: {best['tag']} ({best['avg_step_sec']:.1f}s/step)")

        # For production, prefer the fastest that used < 20 GB
        safe_results = [r for r in ok_results if r["peak_rss_mb"] < 20000]
        if safe_results:
            best_safe = min(safe_results, key=lambda r: r["avg_step_sec"])
            print(f"Best safe config (<20GB): {best_safe['tag']} ({best_safe['avg_step_sec']:.1f}s/step)")

            # Extract device and batch_size from tag
            parts = best_safe["tag"].split("_")
            prod_device = parts[0]
            prod_bs = int(parts[1].replace("bs", ""))

            # Estimate full run time
            est_hours = (best_safe["avg_step_sec"] * 300) / 3600
            print(f"Estimated 300-step run time: {est_hours:.1f} hours")

            prod_config = write_production_config(prod_device, prod_bs)
            print(f"\nProduction config written: {prod_config}")
            print(f"To launch:")
            print(f"  conda run --no-capture-output -n quris reinvent {prod_config} -d {prod_device}")
        else:
            print("WARNING: All configs used >20GB — consider CPU only")
    else:
        print("\nWARNING: No successful benchmarks!")


if __name__ == "__main__":
    main()
