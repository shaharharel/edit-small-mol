#!/usr/bin/env python
"""Dock the FT vs base ablation outputs to compare AD-CovDock + cb_sasa.
Reads ft_vs_base_raw.csv, runs AD-CovDock (meeko tether + Vina score_only) on
acrylamide-bearing mols, compares.
"""
from __future__ import annotations
import sys, csv, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT))
from experiments.run_cohort_eval_phase_b import dock_one

# Read ablation outputs
RAW = PROJECT / "results/paper_evaluation/cohort_eval/ft_vs_base_raw.csv"
rows = []
with open(RAW) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        if int(r.get("valid", 0)) and int(r.get("has_acryl", 0)):
            rows.append(r)
print(f"Acrylamide-bearing rows to dock: {len(rows)} (base + ft combined)")

# Build dock jobs (cohort tag = ft_base or ft_exp6)
jobs = []
for r in rows:
    tag = "ft_base" if r["model"] == "base" else "ft_exp6"
    mol_idx = int(r["seed_idx"]) * 10 + abs(hash(r["output_smi"]) % 10)
    jobs.append((tag, mol_idx, r["output_smi"], None, None))
print(f"Total dockings: {len(jobs)}")

with Pool(8) as pool:
    results = list(pool.imap_unordered(dock_one, jobs, chunksize=4))

# Aggregate
per = {"ft_base": [], "ft_exp6": []}
for res in results:
    if res["msg"] == "ok":
        per[res["cohort"]].append(res)

print(f"\n{'tag':<10}  {'n_ok':>6}  {'AD_mean':>8}  {'AD_median':>10}  {'cb_sasa_mean':>12}  {'cb_sasa_median':>14}  {'hinge%':>6}")
for tag, lst in per.items():
    if not lst:
        print(f"{tag}: no dockings succeeded"); continue
    ads = [r["AD_CovDock_score"] for r in lst if r.get("AD_CovDock_score") is not None]
    sasas = [r["cb_sasa_in_pocket"] for r in lst if r.get("cb_sasa_in_pocket") is not None]
    hinges = [r["hinge_hbond_top1"] for r in lst if r.get("hinge_hbond_top1") is not None]
    hinge_pct = 100.0 * sum(1 for h in hinges if h == 1) / len(hinges) if hinges else 0.0
    print(f"{tag:<10}  {len(lst):>6}  {np.mean(ads):>8.1f}  {np.median(ads):>10.1f}  "
          f"{np.mean(sasas) if sasas else 0.0:>12.2f}  {np.median(sasas) if sasas else 0.0:>14.2f}  {hinge_pct:>6.1f}")

# Save per-mol results
OUT = PROJECT / "results/paper_evaluation/cohort_eval/ft_vs_base_dock.csv"
fields = list(results[0].keys()) if results else []
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)
print(f"\nWrote {OUT}")
