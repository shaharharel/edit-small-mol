#!/usr/bin/env python3
"""EXP7 monitor — print one line every check so background watcher sees progress.

Usage:
   python exp7_monitor.py --interval 600 --once
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def vm_status() -> str:
    try:
        r = subprocess.run(
            ["gcloud","compute","instances","describe","ai-gpu-a100-b","--zone=us-central1-b","--format=value(status)"],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception as e:
        return f"ERR:{e}"


def ssh_cmd(cmd: str, timeout=60) -> str:
    try:
        r = subprocess.run(
            ["gcloud","compute","ssh","ai-gpu-a100-b","--zone=us-central1-b","--command", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip()
    except Exception as e:
        return f"ERR:{e}"


def snapshot_local() -> dict:
    rl = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/exp7_lo_benchmark/_rl")
    n_cohorts_local = 0
    if rl.exists():
        for d in rl.iterdir():
            if d.is_dir() and (d / "sampled.csv").exists():
                n_cohorts_local += 1
    return {"n_cohorts_local": n_cohorts_local}


def snapshot_remote() -> dict:
    n_cohorts_remote = ssh_cmd(
        "find ~/edit-small-mol/data/exp7_lo_benchmark/_rl -name sampled.csv 2>/dev/null | wc -l",
        timeout=60,
    )
    driver_log_tail = ssh_cmd(
        "tail -3 ~/edit-small-mol/data/exp7_lo_benchmark/_logs/driver.log 2>/dev/null",
        timeout=60,
    )
    # parse most recent cell finished line
    last_ok_lines = [l for l in driver_log_tail.split("\n") if "->" in l]
    last_ok = last_ok_lines[-1] if last_ok_lines else "(no progress line)"
    return {"n_cohorts_remote": n_cohorts_remote, "last_ok": last_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    while True:
        ts = time.strftime("%H:%M:%S")
        vm = vm_status()
        local = snapshot_local()
        if vm == "RUNNING":
            remote = snapshot_remote()
            line = (f"[monitor {ts}] VM={vm} | local={local['n_cohorts_local']} cohorts | "
                    f"remote={remote['n_cohorts_remote']} cohorts | last_ok=\"{remote['last_ok']}\"")
        else:
            line = f"[monitor {ts}] VM={vm} | local={local['n_cohorts_local']} cohorts"
        print(line, flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
