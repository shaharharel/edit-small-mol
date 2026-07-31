"""Compute burgi_dunitz_dev_deg + d_SG + n_h_bonds + n_stabilizing_contacts +
pocket_occupancy_pct + atp_pocket_fraction + hinge_hbond for the LEGACY 997
top-1000 cofold cohort. The legacy cofolds use a different dir layout than
the new 3,597 cohort, so this is a small wrapper around compute_pose_geom.

Output: data/tier4_scored/boltz_legacy_geom_metrics_997.csv
Then merge into the live backend DF (via merge_boltz_metrics_into_cohort.py
extension) so the legacy 997 mols pass f5_warhead_dev / etc.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COFOLD_ROOT = PROJECT_ROOT / "data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions"
MANIFEST = PROJECT_ROOT / "data/boltz_poses/top1000_manifest__zap70_cys346.json"
OUT_CSV = PROJECT_ROOT / "data/tier4_scored/boltz_legacy_geom_metrics_997.csv"

sys.path.insert(0, str(PROJECT_ROOT))


def _compute_one(task: tuple[int, str, Path]) -> dict:
    row_id, yaml_name, cif = task
    out = {"row_id": row_id, "yaml_name": yaml_name}
    try:
        from experiments.enrich_mol1_anchor import compute_pose_geom
        # auto-detect warhead — same approach as the new compute
        r1 = compute_pose_geom(cif, warhead_atom_name=None)
        warhead = r1.get("warhead_atom_name")
        if warhead:
            r2 = compute_pose_geom(cif, warhead_atom_name=warhead)
        else:
            r2 = r1
        for k in ("warhead_atom_name", "d_SG", "geom_ok", "burgi_dunitz_dev_deg",
                  "n_h_bonds", "n_stabilizing_contacts", "pocket_occupancy_pct",
                  "atp_pocket_fraction", "hinge_hbond"):
            out[k] = r2.get(k) if r2.get(k) is not None else r1.get(k)
        out["success_flag"] = 1
        out["error"] = ""
    except Exception as e:
        out["success_flag"] = 0
        out["error"] = f"{type(e).__name__}:{e}"
    return out


def main() -> int:
    print("=== Legacy 997 geometry compute ===")
    if not MANIFEST.exists():
        print(f"Missing manifest: {MANIFEST}")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    # Build task list
    tasks = []
    for rid_str, m in manifest.items():
        try:
            rid = int(rid_str)
        except ValueError:
            continue
        yaml_name = m.get("yaml_name") or m.get("name")
        if not yaml_name:
            continue
        cif = COFOLD_ROOT / yaml_name / f"{yaml_name}_model_0.cif"
        if not cif.exists():
            continue
        tasks.append((rid, yaml_name, cif))
    print(f"Tasks: {len(tasks)}")

    t0 = time.perf_counter()
    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Workers: {n_workers}")
    with mp.Pool(n_workers) as pool:
        rows = pool.map(_compute_one, tasks, chunksize=8)
    elapsed = time.perf_counter() - t0
    print(f"Done. {len(rows)} rows in {elapsed:.1f}s ({len(rows)/max(elapsed,1):.1f}/s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print("\n=== Stats ===")
    for c in ("d_SG", "burgi_dunitz_dev_deg", "n_h_bonds", "n_stabilizing_contacts",
              "pocket_occupancy_pct", "atp_pocket_fraction"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                print(f"  {c}: n={len(s)} median={s.median():.3f} p10={s.quantile(0.10):.3f} p90={s.quantile(0.90):.3f}")
    n_ok = (df["success_flag"] == 1).sum()
    print(f"\nSuccess: {n_ok}/{len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
