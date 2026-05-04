#!/usr/bin/env python3
"""
Resume the ai-gpu REINVENT4 orchestrator from where it left off.

Detects which jobs have completed CSVs (≥ 1000 lines = real output) and skips
them. Runs only the remaining jobs.

Usage (on ai-gpu):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aigpu_reinvent_resume.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all the writers from the original orchestrator
from experiments.overnight_aigpu_reinvent_jobs import (
    write_libinvent_locked, write_mol2mol_warhead, write_denovo_warhead,
    write_tier4_denovo_uncon, write_tier4_mol2mol_uncon,
    write_method_a_film_driven, write_method_b_film_driven,
    run_reinvent, collect_smiles_from_csvs, has_uncertainty_ensemble,
    log, OUT_DIR,
)

import json
from datetime import datetime


def is_job_complete(work_dir: Path, prefix: str, min_lines=1000) -> bool:
    """A job is considered complete if its summary CSV has ≥ min_lines."""
    csvs = list(work_dir.glob(f"{prefix}*.csv"))
    if not csvs:
        return False
    # Check the largest CSV
    biggest = max(csvs, key=lambda p: p.stat().st_size)
    try:
        with open(biggest) as f:
            n = sum(1 for _ in f)
        return n >= min_lines
    except Exception:
        return False


def main():
    log("=" * 70)
    log("AI-GPU REINVENT4 RESUME")
    log("=" * 70)
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Uncertainty ensemble available: {has_uncertainty_ensemble()}")

    jobs = [
        ("libinvent_locked", write_libinvent_locked, OUT_DIR / "libinvent_locked", 200),
        ("mol2mol_warhead",  write_mol2mol_warhead,  OUT_DIR / "mol2mol_warhead",  200),
        ("denovo_warhead",   write_denovo_warhead,   OUT_DIR / "denovo_warhead",   200),
        ("tier4_denovo",     write_tier4_denovo_uncon, OUT_DIR / "tier4_denovo",   300),
        ("tier4_mol2mol",    write_tier4_mol2mol_uncon, OUT_DIR / "tier4_mol2mol", 300),
        ("method_a",         write_method_a_film_driven, OUT_DIR / "method_a",     300),
        ("method_b",         write_method_b_film_driven, OUT_DIR / "method_b",     300),
    ]

    log("\nResume status:")
    job_results = []
    for name, write_fn, work_dir, max_steps in jobs:
        # Use prefix that matches the summary_csv_prefix in each TOML
        # Convention: prefix = name (matches the configs we write)
        prefix_map = {
            "libinvent_locked": "libinvent_locked",
            "mol2mol_warhead": "mol2mol_warhead",
            "denovo_warhead": "denovo_warhead",
            "tier4_denovo": "tier4_denovo",
            "tier4_mol2mol": "tier4_mol2mol",
            "method_a": "method_a_filmdriven_denovo",
            "method_b": "method_b_filmdriven_mol2mol",
        }
        prefix = prefix_map.get(name, name)
        if is_job_complete(work_dir, prefix):
            csv_lines = max((sum(1 for _ in open(f)) for f in work_dir.glob(f"{prefix}*.csv")), default=0)
            log(f"  ✓ {name}: SKIP (already done, {csv_lines} CSV lines)")
            job_results.append({"name": name, "ok": True, "skipped": True, "n_smiles": csv_lines})
        else:
            log(f"  ▶ {name}: WILL RUN")

    log("\nLaunching remaining jobs...")
    for name, write_fn, work_dir, max_steps in jobs:
        prefix_map = {
            "libinvent_locked": "libinvent_locked",
            "mol2mol_warhead": "mol2mol_warhead",
            "denovo_warhead": "denovo_warhead",
            "tier4_denovo": "tier4_denovo",
            "tier4_mol2mol": "tier4_mol2mol",
            "method_a": "method_a_filmdriven_denovo",
            "method_b": "method_b_filmdriven_mol2mol",
        }
        prefix = prefix_map.get(name, name)
        if is_job_complete(work_dir, prefix):
            continue
        try:
            # Clean partial state for this job
            for f in work_dir.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass
            cfg = write_fn(work_dir, max_steps=max_steps)
            ok = run_reinvent(cfg, work_dir, name)
            smis = collect_smiles_from_csvs(work_dir, prefix)
            job_results.append({"name": name, "config": str(cfg), "ok": ok,
                                "n_smiles": len(smis), "work_dir": str(work_dir)})
            (work_dir / "smiles.txt").write_text("\n".join(smis))
        except Exception as e:
            log(f"  ERROR in job {name}: {e}")
            job_results.append({"name": name, "ok": False, "error": str(e)})

    out_summary = OUT_DIR / "summary_resume.json"
    out_summary.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "jobs": job_results,
    }, indent=2))
    log(f"Summary → {out_summary}")
    log(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
