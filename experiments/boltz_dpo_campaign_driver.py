#!/usr/bin/env python3
"""Boltz DPO Campaign Driver.

Per-cohort pipeline (SPOT-safe, resumable):
  1. If --smiles_csv given, use it directly.
     Else if --ckpt given, sample N_RAW mols via sample_dpo_composite_cohort.py.
  2. Filter to valid + unique canonical + acryl-on-largest-frag → N_TARGET SMILES.
  3. Build one Boltz YAML per SMILES with Cys346-SG ↔ warhead-β-C covalent constraint.
  4. Run Boltz predict serially/concurrently, resuming skipped ones.
  5. Harvest CSV every CHECKPOINT_EVERY cofolds → track_A_<cohort>.csv.

Reuses helpers from boltz_verdict_driver.py (build_yamls, filter_valid_acryl,
run_boltz_batch, harvest_cohort).
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
CAMPAIGN_ROOT = PROJECT_ROOT / "data/paper_pair_training/boltz_dpo_campaign"
CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
(CAMPAIGN_ROOT / "logs").mkdir(exist_ok=True)
(CAMPAIGN_ROOT / "yamls").mkdir(exist_ok=True)
(CAMPAIGN_ROOT / "cofolds").mkdir(exist_ok=True)
(CAMPAIGN_ROOT / "samples").mkdir(exist_ok=True)

# Import from sibling
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from boltz_verdict_driver import (  # noqa: E402
    filter_valid_acryl,
    build_yamls,
    run_boltz_batch,
    harvest_cohort,
    MOL1_SMI,
)


def read_smiles_csv(csv_path: Path) -> list[str]:
    """Read SMILES column from a CSV (assumes 'SMILES' col, else col 0)."""
    smis = []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        smi_idx = 0
        if header:
            for i, c in enumerate(header):
                if c.strip().upper() == "SMILES":
                    smi_idx = i
                    break
        for row in reader:
            if not row:
                continue
            s = row[smi_idx].strip()
            if s:
                smis.append(s)
    return smis


def sample_from_ckpt(ckpt: Path, out_csv: Path, n_samples: int, batch_size: int = 32) -> Path:
    """Invoke sample_dpo_composite_cohort.py to generate raw samples."""
    if out_csv.exists():
        try:
            n_have = sum(1 for _ in out_csv.open()) - 1
            if n_have >= n_samples:
                print(f"[sample] {out_csv.name} exists with {n_have} rows — skip")
                return out_csv
        except Exception:
            pass
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/sample_dpo_composite_cohort.py"),
        "--ckpt", str(ckpt),
        "--anchor", MOL1_SMI,
        "--n_samples", str(n_samples),
        "--batch_size", str(batch_size),
        "--out_csv", str(out_csv),
    ]
    print(f"[sample] {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"sampling failed rc={r.returncode}")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="Cohort name (e.g. c1_composite_v1)")
    ap.add_argument("--smiles_csv", help="Path to pre-generated SMILES CSV")
    ap.add_argument("--ckpt", help="DPO chkpt for on-the-fly sampling")
    ap.add_argument("--n_raw", type=int, default=3000, help="Raw samples to generate before filter")
    ap.add_argument("--n_target", type=int, default=400, help="Number of filtered mols to cofold")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_workers", type=int, default=2, help="Concurrent Boltz procs")
    args = ap.parse_args()

    cohort = args.cohort
    print(f"[{cohort}] start")

    # Get SMILES source
    if args.smiles_csv:
        smi_csv = Path(args.smiles_csv)
    elif args.ckpt:
        smi_csv = CAMPAIGN_ROOT / "samples" / f"{cohort}_raw.csv"
        sample_from_ckpt(Path(args.ckpt), smi_csv, args.n_raw, args.batch_size)
    else:
        sys.exit("--smiles_csv or --ckpt required")

    raw = read_smiles_csv(smi_csv)
    print(f"[{cohort}] {len(raw)} raw SMILES")
    smis = filter_valid_acryl(raw, args.n_target)
    print(f"[{cohort}] {len(smis)} after filter (target {args.n_target})")
    if not smis:
        sys.exit(f"[{cohort}] no valid SMILES post-filter")

    # Persist filtered list for reproducibility
    (CAMPAIGN_ROOT / "samples" / f"{cohort}_filtered.smi").write_text(
        "\n".join(smis) + "\n"
    )

    yaml_dir = CAMPAIGN_ROOT / "yamls" / cohort
    cofold_dir = CAMPAIGN_ROOT / "cofolds" / cohort
    log_path = CAMPAIGN_ROOT / "logs" / f"boltz_{cohort}.log"
    out_csv = CAMPAIGN_ROOT / f"track_A_{cohort}.csv"

    manifest = build_yamls(smis, yaml_dir, cohort)
    print(f"[{cohort}] {len(manifest)} YAMLs written")

    def _ckpt_cb():
        harvest_cohort(cofold_dir, cohort, out_csv, manifest)

    t0 = time.time()
    run_boltz_batch(yaml_dir, cofold_dir, log_path,
                     max_workers=args.max_workers, checkpoint_cb=_ckpt_cb)
    # Final harvest
    harvest_cohort(cofold_dir, cohort, out_csv, manifest)
    print(f"[{cohort}] done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
