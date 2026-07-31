"""QA #8: When the executor's same-infra DAP cohort lands on a100-b
(`cohort_dap_zap70.csv.summary.json`), this script adds it to MASTER and produces
the apples-to-apples DAP-vs-PPO comparison.

To run: simply invoke after the cohort_dap_zap70.csv.summary.json appears in
data/exp_ppo_v2plus/.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
PPO_DIR = PROJECT_ROOT / "data" / "exp_ppo_v2plus"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # The DAP cohort summary will be at one of these paths
    candidates = [
        PPO_DIR / "cohort_dap_zap70.csv.summary.json",
        PPO_DIR / "cohort_dap.csv.summary.json",
        PPO_DIR / "dap_zap70_summary.json",
    ]
    found = None
    for p in candidates:
        if p.exists():
            found = p
            break
    if found is None:
        log("DAP cohort summary not yet available locally — re-run when it lands.")
        log(f"Checked: {[str(p) for p in candidates]}")
        return
    log(f"Found DAP same-infra cohort: {found}")
    c = json.loads(found.read_text())
    log(f"  n_valid={c.get('n_valid')}, war={100*c.get('warhead_any_rate',0):.1f}%, "
        f"thiq={100*c.get('thiq_exact_rate',0):.1f}%, "
        f"FiLM_mean={c.get('mean_film_pIC50',0):.2f}, "
        f"FiLM>=7={100*c.get('frac_film_ge_7',0):.1f}%")

    # Append to master (run qa6 which auto-includes all cohort_*.csv.summary.json)
    import subprocess
    r = subprocess.run([
        "/opt/miniconda3/envs/quris/bin/python",
        str(PROJECT_ROOT / "experiments" / "qa_sweep_2026_06_24" / "qa6_assemble_v2.py")],
        capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)

    # Build the headline same-infra comparison table
    import pandas as pd
    df = pd.read_csv(OUT_DIR / "MASTER_metrics_table.csv")
    ppo = df[df.cohort.str.contains("ppo_post_RL")].copy()
    dap_rows = df[df.cohort.str.contains("dap", case=False)].copy()
    log(f"PPO post-RL rows: {len(ppo)}, DAP rows: {len(dap_rows)}")
    print(ppo[["cohort","acrylamide_pct","thiq_core_pct","filmdelta_pic50_mean","filmdelta_pic50_frac_ge_7","qed_mean"]].to_string(index=False))
    print()
    print(dap_rows[["cohort","acrylamide_pct","thiq_core_pct","filmdelta_pic50_mean","filmdelta_pic50_frac_ge_7","qed_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
