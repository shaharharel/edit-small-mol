"""Comprehensive QA for the Mol1-anchored Tier 4 pipeline.

Tests per stream (Track A sampling, Track B' RL, local backfill, configs).
Exits non-zero on any failure; prints a clear pass/fail summary.

Run: python experiments/qa_mol1_pipeline.py

If --remote=ai-gpu,ai-gpu2,ai-gpu-a100-b passed, also probes remote VM state.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import tomllib
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
MOL1_CANON = Chem.MolToSmiles(Chem.MolFromSmiles(MOL1))

results = OrderedDict()


def check(stream: str, name: str, ok: bool, detail: str = ""):
    s = "PASS" if ok else "FAIL"
    print(f"  [{s}] {name}" + (f"  — {detail}" if detail else ""))
    results.setdefault(stream, []).append((name, ok, detail))


# === Stream 1: anchor + seed files ===
def qa_seeds():
    print("\n=== Stream 1: Anchor + Seed files ===")
    anchor = PROJECT / "data/mol1_only_anchor.smi"
    check("seeds", "mol1_only_anchor.smi exists", anchor.exists(), str(anchor.relative_to(PROJECT)))
    if anchor.exists():
        lines = [l for l in anchor.read_text().splitlines() if l]
        check("seeds", "mol1_only_anchor.smi has exactly 1 line", len(lines) == 1, f"got {len(lines)}")
        check("seeds", "mol1_only_anchor.smi line == canonical Mol1", lines[0] == MOL1_CANON if lines else False)

    for fname in ["seed_mol1_only.smi",
                  "seed_zap70_all_plus_mol1.smi",
                  "seed_kinase_zap70x5_mol1x20.smi"]:
        p = PROJECT / "data/mol1_rl_seeds" / fname
        check("seeds", f"{fname} exists", p.exists())
        if p.exists():
            lines = [l for l in p.read_text().splitlines() if l]
            check("seeds", f"{fname}: contains Mol1",
                  MOL1_CANON in set(lines),
                  f"{lines.count(MOL1_CANON)} occurrences")
            # parse-ok rate
            ok_n = sum(1 for l in lines if Chem.MolFromSmiles(l) is not None)
            check("seeds", f"{fname}: 100% parse-ok",
                  ok_n == len(lines), f"{ok_n}/{len(lines)} parse")


# === Stream 2: Track A sampling tomls ===
def qa_track_a_tomls():
    print("\n=== Stream 2: Track A sampling tomls ===")
    tomls_dir = PROJECT / "experiments/mol1_anchored_tomls"
    check("track_a_tomls", "tomls dir exists", tomls_dir.exists())
    if not tomls_dir.exists():
        return
    expected = [
        "exp2_preRL_mol1_50K", "exp6_preRL_mol1_50K", "exp6v3_preRL_mol1_50K",
        "exp2_v2_rl_postRL_mol1_50K", "exp2_v2_rl_v2_postRL_mol1_50K",
        "exp6_v3_postRL_mol1_50K", "exp6_v4_postRL_mol1_50K", "exp6_v5_postRL_mol1_50K",
    ]
    for tag in expected:
        p = tomls_dir / f"{tag}.toml"
        check("track_a_tomls", f"{tag}.toml exists", p.exists())
        if p.exists():
            try:
                d = tomllib.loads(p.read_text())
                params = d["parameters"]
                check("track_a_tomls", f"{tag}: smiles_file points to mol1_only_anchor",
                      params.get("smiles_file", "").endswith("mol1_only_anchor.smi"))
                check("track_a_tomls", f"{tag}: num_smiles = 50000",
                      params.get("num_smiles") == 50000)
                # model_file matches expected
                mf = params.get("model_file", "")
                if "preRL" in tag:
                    ok = mf.endswith(".prior")
                else:
                    ok = mf.endswith(".chkpt")
                check("track_a_tomls", f"{tag}: model_file kind is correct ({'.prior' if 'preRL' in tag else '.chkpt'})", ok, mf)
            except Exception as e:
                check("track_a_tomls", f"{tag}: toml parses", False, str(e))

    # Launcher scripts
    for fn in ["run_aigpu.sh", "run_aigpu2.sh"]:
        p = tomls_dir / fn
        check("track_a_tomls", f"{fn} exists + executable",
              p.exists() and p.stat().st_mode & 0o100)


# === Stream 3: Track B' RL tomls ===
def qa_track_b_tomls():
    print("\n=== Stream 3: Track B' RL tomls ===")
    tomls_dir = PROJECT / "experiments/mol1_rl_tomls"
    check("track_b_tomls", "tomls dir exists", tomls_dir.exists())
    if not tomls_dir.exists():
        return
    expected = [
        ("mol1RL_v5_seed_mol1_only", "seed_mol1_only.smi"),
        ("mol1RL_v5_seed_zap70_all_plus_mol1", "seed_zap70_all_plus_mol1.smi"),
        ("mol1RL_v5_seed_kinase_zap70x5_mol1x20", "seed_kinase_zap70x5_mol1x20.smi"),
    ]
    for tag, seed_file in expected:
        p = tomls_dir / f"{tag}.toml"
        check("track_b_tomls", f"{tag}.toml exists", p.exists())
        if p.exists():
            try:
                d = tomllib.loads(p.read_text())
                params = d["parameters"]
                check("track_b_tomls", f"{tag}: smiles_file ends with {seed_file}",
                      params.get("smiles_file", "").endswith(seed_file))
                check("track_b_tomls", f"{tag}: prior is warhead_tokens (exp6_v5 base)",
                      params.get("prior_file", "").endswith("warhead_tokens.prior"))
                check("track_b_tomls", f"{tag}: run_type = staged_learning",
                      d.get("run_type") == "staged_learning")
                # Must have at least 1 stage
                check("track_b_tomls", f"{tag}: has [stage] section",
                      isinstance(d.get("stage"), list) and len(d["stage"]) >= 1)
            except Exception as e:
                check("track_b_tomls", f"{tag}: toml parses", False, str(e))


# === Stream 4: seed_smi backfill ===
def qa_backfill():
    print("\n=== Stream 4: seed_smi backfill into scored CSVs ===")
    scored_dir = PROJECT / "data/tier4_scored"
    expected = OrderedDict([
        ("exp6_v3_scored.csv",       (48648, 21)),
        ("exp6_v4_scored.csv",       (160,   16)),
        ("exp6_v5_scored.csv",       (90088, 21)),
        ("exp2_v2_rl_v2_scored.csv", (73799, 21)),
        ("exp2_scored.csv",          (500,   1)),
        ("exp6_scored.csv",          (500,   1)),
    ])
    for fname, (exp_rows, exp_seeds) in expected.items():
        p = scored_dir / fname
        check("backfill", f"{fname} exists", p.exists())
        if p.exists():
            df = pd.read_csv(p)
            check("backfill", f"{fname}: has seed_smi column", "seed_smi" in df.columns)
            if "seed_smi" in df.columns:
                cov = df["seed_smi"].notna().mean()
                check("backfill", f"{fname}: 100% seed_smi coverage",
                      cov == 1.0, f"{cov:.1%}")
                n_uniq = df["seed_smi"].nunique(dropna=True)
                check("backfill", f"{fname}: unique seeds = {exp_seeds}",
                      n_uniq == exp_seeds, f"got {n_uniq}")
                check("backfill", f"{fname}: row count = {exp_rows:,}",
                      len(df) == exp_rows, f"got {len(df):,}")
                # Legacy cohorts must show Mol1 as the sole seed
                if exp_seeds == 1:
                    only = df["seed_smi"].unique()[0]
                    check("backfill", f"{fname}: sole seed == Mol1",
                          only == MOL1_CANON)


# === Stream 5: local model files ===
def qa_local_models():
    print("\n=== Stream 5: Local model files ===")
    m = PROJECT / "models"
    for fn, min_bytes in [
        ("reinvent4_mol2mol_covalent_ft.prior", 70_000_000),
        ("reinvent4_mol2mol_warhead_tokens.prior", 70_000_000),
        ("reinvent4_mol2mol_warhead_tokens_v2.prior", 70_000_000),
    ]:
        p = m / fn
        check("local_models", f"{fn} exists ≥{min_bytes//1_000_000}MB",
              p.exists() and p.stat().st_size >= min_bytes,
              f"{p.stat().st_size/1_000_000:.0f}MB" if p.exists() else "missing")

    rl = m / "rl_checkpoints"
    check("local_models", "rl_checkpoints/ dir exists", rl.exists())
    if rl.exists():
        for fn, min_bytes in [
            ("exp2_v2_rl_stage1.chkpt", 70_000_000),
            ("exp2_v2_rl_v2_stage1.chkpt", 70_000_000),
            ("exp6_v3_stage1.chkpt", 70_000_000),
            ("exp6_v4_stage1.chkpt", 70_000_000),
            ("exp6_v5_stage1.chkpt", 70_000_000),
        ]:
            p = rl / fn
            check("local_models", f"rl_checkpoints/{fn} present",
                  p.exists() and p.stat().st_size >= min_bytes,
                  f"{p.stat().st_size/1_000_000:.0f}MB" if p.exists() else "missing")


# === Stream 6: remote VM probes ===
def remote_run(vm: str, zone: str, cmd: str, timeout=60):
    """Run a command on a remote VM via gcloud ssh, return (stdout, ok)."""
    full = ["gcloud", "compute", "ssh", vm, "--zone", zone, "--command", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    except Exception:
        return "", False


def qa_remote(vms: list[str]):
    if not vms:
        return
    print(f"\n=== Stream 6: Remote VM probes ({', '.join(vms)}) ===")
    zone_map = {
        "ai-gpu": "us-central1-c",
        "ai-gpu2": "us-central1-a",
        "ai-gpu-a100": "us-central1-a",
        "ai-gpu-a100-b": "us-central1-b",
        "ai-chem": "us-east1-b",
    }
    for vm in vms:
        zone = zone_map.get(vm)
        if not zone:
            check("remote", f"{vm}: unknown zone", False)
            continue
        # ssh sanity
        out, ok = remote_run(vm, zone, "echo OK && date && which reinvent 2>/dev/null", timeout=30)
        check("remote", f"{vm}: SSH reachable", ok and "OK" in out)
        if not ok:
            continue
        has_reinvent = "/reinvent" in out
        check("remote", f"{vm}: reinvent on PATH", has_reinvent,
              "(may need conda activate)" if not has_reinvent else "")
        # repo + chkpts
        out, _ = remote_run(vm, zone,
            "ls ~/edit-small-mol/models/rl_checkpoints/ 2>/dev/null | wc -l; "
            "ls ~/edit-small-mol/data/mol1_only_anchor.smi 2>/dev/null && echo HAS_ANCHOR; "
            "ls ~/edit-small-mol/experiments/mol1_anchored_tomls/ 2>/dev/null | wc -l",
            timeout=30)
        lines = out.split("\n")
        check("remote", f"{vm}: chkpts present (≥4)",
              len(lines) > 0 and lines[0].strip().isdigit() and int(lines[0]) >= 4,
              f"saw {lines[0] if lines else 0} chkpts")
        check("remote", f"{vm}: mol1_only_anchor.smi synced",
              "HAS_ANCHOR" in out)


# === Stream 7: Track A sampling progress (cohort outputs) ===
def qa_sampling_outputs(vms: list[str]):
    if not vms:
        return
    print(f"\n=== Stream 7: Track A sampling outputs ===")
    zone_map = {"ai-gpu": "us-central1-c", "ai-gpu2": "us-central1-a"}
    for vm in vms:
        zone = zone_map.get(vm)
        if not zone:
            continue
        out, ok = remote_run(vm, zone,
            "for d in ~/edit-small-mol/data/mol1_anchored_tier4/*/; do "
            "  n=$(wc -l <\"$d/sampling.csv\" 2>/dev/null || echo 0); "
            "  echo \"$(basename $d):$n\"; "
            "done", timeout=30)
        if not ok:
            check("sampling", f"{vm}: query outputs", False)
            continue
        cohorts = [l.split(":") for l in out.strip().split("\n") if ":" in l]
        for name, n in cohorts:
            n = int(n.strip()) if n.strip().isdigit() else 0
            # "PASS" criterion: file present and not empty OR not started yet
            label = f"{vm}::{name}"
            if n == 0:
                check("sampling", f"{label}: not yet started or in-flight", True, "0 lines")
            elif n < 100:
                check("sampling", f"{label}: minimal output", False, f"{n} lines")
            else:
                check("sampling", f"{label}: has output", True, f"{n:,} lines")


# === Summary ===
def summary():
    print("\n" + "=" * 60)
    print("QA SUMMARY")
    print("=" * 60)
    total = 0
    failed = 0
    for stream, items in results.items():
        passed = sum(1 for _, ok, _ in items if ok)
        n = len(items)
        total += n
        failed += (n - passed)
        status = "✓" if passed == n else "✗"
        print(f"  {status}  {stream:18s}  {passed}/{n} pass")
        if passed < n:
            for name, ok, det in items:
                if not ok:
                    print(f"        FAIL: {name}{(' — ' + det) if det else ''}")
    print("-" * 60)
    print(f"  TOTAL: {total - failed}/{total} pass ({failed} fail)")
    return 0 if failed == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="",
                    help="Comma-separated VMs to probe (e.g. ai-gpu,ai-gpu2,ai-gpu-a100-b)")
    ap.add_argument("--skip-sampling", action="store_true",
                    help="Skip Stream 7 sampling-output checks")
    args = ap.parse_args()
    vms = [v.strip() for v in args.remote.split(",") if v.strip()]

    qa_seeds()
    qa_track_a_tomls()
    qa_track_b_tomls()
    qa_backfill()
    qa_local_models()
    qa_remote(vms)
    if not args.skip_sampling:
        qa_sampling_outputs([v for v in vms if v in ("ai-gpu", "ai-gpu2")])
    sys.exit(summary())


if __name__ == "__main__":
    main()
