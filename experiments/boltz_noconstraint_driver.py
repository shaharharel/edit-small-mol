#!/usr/bin/env python3
"""Experiment A: Boltz cofold *without* covalent constraint for v2_curriculum_clean cohorts.

Same Mol1 anchor / ZAP70 pocket / v1_clean samples, but the YAML omits the
`bond: atom1=[A,346,SG], atom2=[B,1,C<n>]` block, letting Boltz pick a
non-covalent pose freely. Warhead is then free to be anywhere.

Then we measure per cofold:
  - d_b_nuc_angstrom      : β-C to Cys346 Sγ (NOT clamped to 1.8)
  - bd_angle_deg          : Bürgi-Dunitz angle at β-C
  - phi_planar_deg        : vinyl-amide planar dihedral
  - complex_iptm, ligand_iptm, complex_plddt, mpae_prot_lig_min

Headline metric:
  f_BD_ready = fraction of cofolds with d ∈ [2.5, 5.5] Å AND bd ∈ [80, 130] °

Usage (per cohort):
    python experiments/boltz_noconstraint_driver.py \
        --cohort v2curr_clean_NOCONSTR_theta_90 \
        --smiles_csv <path>/samples_theta_90.csv \
        --n_target 100 --max_workers 2

Cofolds land in:
    data/paper_pair_training/v2_curriculum_clean/noconstraint_cofolds/<cohort>/
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

# Reuse the well-tested helpers from boltz_verdict_driver.
PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from boltz_verdict_driver import (  # noqa: E402
    filter_valid_acryl,
    run_boltz_batch,
    harvest_cohort,
    boltz_atom_name,
    ZAP70_SEQ,
    MSA_PATH,
    TARGET_CYS,  # noqa: F401
)

OUT_ROOT = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/noconstraint_cofolds"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "logs").mkdir(exist_ok=True)
(OUT_ROOT / "yamls").mkdir(exist_ok=True)
(OUT_ROOT / "cofolds").mkdir(exist_ok=True)
(OUT_ROOT / "samples").mkdir(exist_ok=True)


def build_yamls_no_constraint(smis: list[str], out_dir: Path, cohort_name: str) -> list[dict]:
    """Same as build_yamls but omits the `constraints:` block entirely."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, smi in enumerate(smis):
        # Still compute atom name for reproducibility / downstream comparison,
        # even though we don't emit the bond constraint.
        atom_name, _ = boltz_atom_name(smi)
        name = f"{cohort_name}_{i:04d}"
        yaml = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {ZAP70_SEQ}\n"
            f"      msa: {MSA_PATH}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{smi}'\n"
        )
        (out_dir / f"{name}.yaml").write_text(yaml)
        manifest.append({"name": name, "smiles": smi, "warhead_atom": atom_name})
    return manifest


def read_smiles_csv(csv_path: Path) -> list[str]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True,
                    help="Cohort name (e.g. v2curr_clean_NOCONSTR_theta_90)")
    ap.add_argument("--smiles_csv", required=True)
    ap.add_argument("--n_target", type=int, default=100)
    ap.add_argument("--max_workers", type=int, default=2)
    args = ap.parse_args()

    cohort = args.cohort
    smi_csv = Path(args.smiles_csv)
    raw = read_smiles_csv(smi_csv)
    print(f"[{cohort}] {len(raw)} raw SMILES from {smi_csv}")
    smis = filter_valid_acryl(raw, args.n_target)
    print(f"[{cohort}] {len(smis)} after filter (target {args.n_target})")
    if not smis:
        sys.exit(f"[{cohort}] no valid SMILES post-filter")

    (OUT_ROOT / "samples" / f"{cohort}_filtered.smi").write_text("\n".join(smis) + "\n")

    yaml_dir = OUT_ROOT / "yamls" / cohort
    cofold_dir = OUT_ROOT / "cofolds" / cohort
    log_path = OUT_ROOT / "logs" / f"boltz_{cohort}.log"
    out_csv = OUT_ROOT / f"track_A_{cohort}.csv"

    manifest = build_yamls_no_constraint(smis, yaml_dir, cohort)
    print(f"[{cohort}] {len(manifest)} YAMLs written (no covalent constraint)")

    def _ckpt():
        harvest_cohort(cofold_dir, cohort, out_csv, manifest)

    t0 = time.time()
    run_boltz_batch(yaml_dir, cofold_dir, log_path,
                    max_workers=args.max_workers, checkpoint_cb=_ckpt)
    harvest_cohort(cofold_dir, cohort, out_csv, manifest)
    print(f"[{cohort}] done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
