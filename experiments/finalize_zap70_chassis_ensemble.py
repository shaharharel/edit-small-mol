"""Phase D finalize: evaluate each chassis cohort via eval_lingo3dmol_plans
and merge into the ENSEMBLE summary.

Usage:
    python experiments/finalize_zap70_chassis_ensemble.py \
        --base_dir data/lingo3dmol_zap70_chassis \
        --chassis_ids ZAP_C1 ZAP_C2 ZAP_C3 ZAP_C4 ZAP_C5 ZAP_C6
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def eval_one(chassis_id: str, sdf: Path, anchor_json: Path) -> dict:
    """Run eval_lingo3dmol_plans on a single SDF and return the per-cohort result."""
    if not sdf.exists():
        return {"status": "missing_sdf", "sdf": str(sdf)}
    out_json = sdf.parent / f"{chassis_id}_eval.json"
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "eval_lingo3dmol_plans.py"),
        "--input", str(sdf),
        "--tag", chassis_id,
        "--anchor", str(anchor_json),
        "--out", str(out_json),
    ]
    print(f"[finalize] eval {chassis_id} -> {out_json}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "eval_failed", "rc": proc.returncode,
                "stderr": proc.stderr[-1000:]}
    if not out_json.exists():
        return {"status": "no_output"}
    res = json.loads(out_json.read_text())
    # eval_lingo3dmol_plans writes {tag: result}
    return res.get(chassis_id, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/lingo3dmol_zap70_chassis")
    ap.add_argument("--chassis_ids", nargs="+",
                    default=["ZAP_C1", "ZAP_C2", "ZAP_C3", "ZAP_C4", "ZAP_C5", "ZAP_C6"])
    ap.add_argument("--anchor_json", default="data/lingo3dmol_anchor_zap70_cys346.json")
    args = ap.parse_args()

    base = Path(args.base_dir)
    anchor = Path(args.anchor_json)

    # 1) eval each cohort
    per_cohort = {}
    for cid in args.chassis_ids:
        sdf = base / cid / "samples.sdf"
        per_cohort[cid] = eval_one(cid, sdf, anchor)
        print(f"  {cid}: status={per_cohort[cid].get('status','?')}")

    # 2) merge into ENSEMBLE via merge script
    merge_cmd = [
        sys.executable,
        str(ROOT / "experiments" / "merge_zap70_chassis_ensemble.py"),
        "--base_dir", str(base),
        "--chassis_ids", *args.chassis_ids,
        "--output_dir", str(base / "ENSEMBLE"),
    ]
    print(f"[finalize] merging -> {base/'ENSEMBLE'}")
    proc = subprocess.run(merge_cmd, capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(f"[finalize] merge failed: {proc.stderr[-1000:]}")

    # 3) eval the ENSEMBLE SDF
    ens_sdf = base / "ENSEMBLE" / "samples.sdf"
    ens_eval = eval_one("ENSEMBLE", ens_sdf, anchor)

    # 4) Combine into a single final JSON
    final = {
        "chassis_ids": args.chassis_ids,
        "per_cohort_eval": per_cohort,
        "ensemble_eval": ens_eval,
    }
    final_path = base / "ENSEMBLE" / "final_report.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final, indent=2, default=str))
    print(f"\n[finalize] Wrote {final_path}")

    # 5) Print compact summary
    print("\n=== Per-chassis CHR / scaffold-div / MW_drug / QED-pass ===")
    for cid, res in per_cohort.items():
        s = res.get("summary", {}) if isinstance(res, dict) else {}
        print(f"  {cid:>7}  CHR={s.get('covalent_hit_rate','?'):>6}  "
              f"sca_div={s.get('scaffold_diversity_pct','?'):>5}  "
              f"MW_dr={s.get('MW_in_drug_range_pct','?'):>5}  "
              f"QED={s.get('QED_passing_pct','?'):>5}")
    ens = ens_eval.get("summary", {}) if isinstance(ens_eval, dict) else {}
    print(f"\n  ENSEMBLE  CHR={ens.get('covalent_hit_rate','?')}  "
          f"sca_div={ens.get('scaffold_diversity_pct','?')}  "
          f"MW_dr={ens.get('MW_in_drug_range_pct','?')}  "
          f"QED={ens.get('QED_passing_pct','?')}")


if __name__ == "__main__":
    main()
