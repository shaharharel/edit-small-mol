"""Merge kinase classifier scores into F4_boltz_full.csv (in-place backup-then-overwrite).

Mirrors the existing 'backfill' pattern used by experiments/server/backend.py.
Adds (or refills NaN in) columns: P_kinase, P_Tec_family, pIC50_kinase_aux.

Honors the IQR drop recommendation from score_meta.json:
    if P_kinase IQR on the 838 visible < 0.10, do NOT merge P_kinase.

Run AFTER experiments/kinase_clf/score_cohort.py has produced
results/paper_evaluation/kinase_classifier_scores.csv.
"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
F4 = ROOT / "data" / "tier4_scored" / "F4_boltz_full.csv"
SCORES = ROOT / "results" / "paper_evaluation" / "kinase_classifier_scores.csv"
META = ROOT / "results" / "paper_evaluation" / "kinase_classifier" / "score_meta.json"


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Merging kinase classifier scores -> {F4.name}")
    if not F4.exists():
        raise SystemExit(f"Missing {F4}")
    if not SCORES.exists():
        raise SystemExit(f"Missing {SCORES}; run score_cohort.py first")

    meta = json.loads(META.read_text()) if META.exists() else {}
    f4_stats = meta.get("f4_stats", {})
    drop_p_kin = f4_stats.get("P_kinase", {}).get("drop_recommended", False)
    print(f"  P_kinase drop_recommended = {drop_p_kin}")

    backup = F4.with_suffix(".csv.bak.kinase_clf_" + time.strftime("%Y%m%d_%H%M%S"))
    shutil.copyfile(F4, backup)
    print(f"  Backup -> {backup.name}")

    f4 = pd.read_csv(F4, low_memory=False)
    sc = pd.read_csv(SCORES)
    cols_to_merge = []
    if not drop_p_kin:
        cols_to_merge.append("P_kinase")
    else:
        # User specified: still write P_kinase but drop it from the merged CSV
        print("  Skipping P_kinase column due to low IQR (saturation = useless).")
    cols_to_merge += ["P_Tec_family", "pIC50_kinase_aux"]
    sc = sc[["smiles"] + cols_to_merge].drop_duplicates("smiles", keep="first")
    sc = sc.set_index("smiles")

    n_pre = {c: (f4[c].notna().sum() if c in f4.columns else 0) for c in cols_to_merge}
    for c in cols_to_merge:
        new_vals = f4["smiles"].map(sc[c])
        if c in f4.columns:
            f4[c] = f4[c].where(f4[c].notna(), new_vals)
        else:
            f4[c] = new_vals
    n_post = {c: int(f4[c].notna().sum()) for c in cols_to_merge}
    print("  Coverage:")
    for c in cols_to_merge:
        print(f"    {c}: {n_pre[c]:,} -> {n_post[c]:,} of {len(f4):,}")

    # Survivors are visible: keep mol1_murcko_match==True / 1 if column exists
    if "mol1_murcko_match" in f4.columns:
        sm = f4[f4["mol1_murcko_match"].astype(bool, errors="ignore") == True]
        print(f"  mol1_murcko_match survivors: {len(sm):,}")
        for c in cols_to_merge:
            n_pop = sm[c].notna().sum() if c in sm.columns else 0
            print(f"    {c} on survivors: {n_pop:,}/{len(sm):,} = {n_pop/len(sm)*100:.1f}%")

    f4.to_csv(F4, index=False)
    print(f"  Wrote {F4} ({len(f4):,} rows)")

    # Save the merged column list to a sidecar — used by backend.py if you wire it
    sidecar = META.parent / "merged_columns.json"
    sidecar.write_text(json.dumps({"merged": cols_to_merge,
                                     "drop_p_kin_due_to_saturation": drop_p_kin}, indent=2))
    print(f"  Wrote {sidecar}")


if __name__ == "__main__":
    main()
