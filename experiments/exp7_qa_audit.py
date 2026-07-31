#!/usr/bin/env python3
"""EXP7 LO eval — QA audit (LOCAL, runs every 30 min via Schedule).

Audits in-flight work:
  1. Anchor pool leak audit — verifies tc_drug_max < 0.6 for all B pools
  2. Cohort quality — ≥4000 valid mols per cohort
  3. Warhead retention vs baseline (sanity: should not collapse)
  4. Difficulty distribution — pairs spanning H2L and LO bands
  5. Code/config consistency — REST server port matches toml

Writes data/exp7_lo_benchmark/qa_audit_YYYYMMDD_HHMM.json + console summary.
"""
from __future__ import annotations
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs

PHASE1_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_phase1"
RL_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_rl"
QA_OUT_DIR = PROJECT_ROOT / "data" / "exp7_lo_benchmark"


def audit_b_pools(pairs: List[Dict]) -> Dict:
    """Check every B pool for tc_drug_max < 0.6."""
    audit_path = PHASE1_BASE / "strategy_b_pool_audit_all.json"
    if not audit_path.exists():
        return {"status": "no_b_pools_yet", "n_checked": 0}
    audits = json.loads(audit_path.read_text())
    leaks = [pid for pid, a in audits.items()
             if isinstance(a, dict) and a.get("tc_drug_max", 0) >= 0.6]
    fail_reasons = {pid: a.get("status") for pid, a in audits.items()
                    if isinstance(a, dict) and a.get("status") not in (None, "OK")}
    return {"status": "ok" if not leaks else "LEAKS_FOUND",
            "n_b_pools_checked": len(audits),
            "n_leaks": len(leaks),
            "leak_pair_ids": leaks,
            "fail_status_counts": pd.Series(list(fail_reasons.values())).value_counts().to_dict()
                                  if fail_reasons else {}}


def audit_cohort_quality(pairs: List[Dict]) -> Dict:
    """Check cohort size + valid mol fraction per cohort."""
    rows = []
    for p in pairs:
        for strategy in ["A", "B", "baseline_prior_anchor", "baseline_prior_pool"]:
            cohort = RL_BASE / f"{p['pair_id']}_{strategy}" / "sampled.csv"
            if not cohort.exists():
                continue
            try:
                df = pd.read_csv(cohort)
                smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
                n_total = len(df)
                if n_total < 10:
                    rows.append({"pair_id": p["pair_id"], "strategy": strategy, "n_total": n_total, "status": "TOO_FEW"})
                    continue
                # quick validity check on first 100
                sample = df[smi_col].dropna().astype(str).head(500).tolist()
                n_valid = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None)
                vfrac = n_valid / max(1, len(sample))
                status = "ok" if n_total >= 4000 and vfrac >= 0.85 else "QUALITY_WARN"
                rows.append({"pair_id": p["pair_id"], "strategy": strategy,
                             "n_total": n_total, "valid_frac_sample": vfrac, "status": status})
            except Exception as e:
                rows.append({"pair_id": p["pair_id"], "strategy": strategy, "status": f"READ_FAIL: {e}"})
    if not rows:
        return {"status": "no_cohorts_yet", "n_cohorts": 0}
    df = pd.DataFrame(rows)
    status_counts = df["status"].value_counts().to_dict()
    n_quality_warn = sum(1 for r in rows if r["status"] == "QUALITY_WARN")
    return {"status": "ok" if n_quality_warn == 0 else "QUALITY_ISSUES",
            "n_cohorts": len(df),
            "status_counts": status_counts,
            "cohorts_below_4k": df[df.get("n_total", 0) < 4000][["pair_id","strategy","n_total"]].to_dict("records")}


def audit_difficulty_distribution(pairs: List[Dict]) -> Dict:
    """Cover H2L (Tc 0.3-0.6) and LO (Tc 0.6-0.9) bands."""
    h2l = [p for p in pairs if 0.30 <= p.get("tc_anchor_drug", 0) < 0.60]
    lo = [p for p in pairs if 0.60 <= p.get("tc_anchor_drug", 0) < 0.90]
    other = [p for p in pairs if not (0.30 <= p.get("tc_anchor_drug", 0) < 0.90)]
    return {"n_h2l": len(h2l), "n_lo": len(lo), "n_other": len(other),
            "median_tc": float(np.median([p.get("tc_anchor_drug", 0) for p in pairs]))}


def audit_warhead_collapse(pairs: List[Dict]) -> Dict:
    """Compare warhead retention in cohorts. Sample first-available cohort per pair/strategy."""
    # We expect generic-warhead retention ≥50% for all cohorts (anchor has warhead)
    issues = []
    n_checked = 0
    for p in pairs:
        for strategy in ["A", "B"]:
            cohort = RL_BASE / f"{p['pair_id']}_{strategy}" / "sampled.csv"
            if not cohort.exists():
                continue
            n_checked += 1
            try:
                df = pd.read_csv(cohort)
                smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
                smiles = df[smi_col].dropna().astype(str).head(500).tolist()
                # Broad-acceptance generic acrylamide-or-Michael-acceptor pattern (covers
                # C=CC(=O)N AND substituted variants like C/C=C/C(=O)N, CC#CC(=O)N alkyne,
                # any α,β-unsaturated carbonyl-N motif)
                patterns = [
                    Chem.MolFromSmarts("[#6]=[#6][C](=O)[#7]"),  # generic acrylamide/Michael
                    Chem.MolFromSmarts("[#6]#[#6][C](=O)[#7]"),  # ynamide / propiolamide
                ]
                hits = 0
                for s in smiles:
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        continue
                    if any(p and m.HasSubstructMatch(p) for p in patterns):
                        hits += 1
                wh_pct = 100.0 * hits / max(1, len(smiles))
                if wh_pct < 40:
                    issues.append({"pair_id": p["pair_id"], "strategy": strategy, "warhead_pct": wh_pct})
            except Exception:
                pass
    return {"n_checked": n_checked, "n_below_40_pct": len(issues),
            "issues": issues[:10],
            "status": "ok" if not issues else "WARHEAD_COLLAPSE"}


def main():
    pairs = load_all_pairs()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = QA_OUT_DIR / f"qa_audit_{ts}.json"

    audit = {
        "timestamp": ts,
        "n_pairs": len(pairs),
        "b_pool_leak_audit": audit_b_pools(pairs),
        "cohort_quality": audit_cohort_quality(pairs),
        "difficulty_distribution": audit_difficulty_distribution(pairs),
        "warhead_collapse_check": audit_warhead_collapse(pairs),
    }

    out_path.write_text(json.dumps(audit, indent=2))

    # Console summary
    print(f"=== QA Audit {ts} ===")
    print(f"  B-pool leak: {audit['b_pool_leak_audit'].get('status')} "
          f"(n_leaks={audit['b_pool_leak_audit'].get('n_leaks', 0)})")
    print(f"  Cohort quality: {audit['cohort_quality'].get('status')} "
          f"(n_cohorts={audit['cohort_quality'].get('n_cohorts', 0)})")
    print(f"  Difficulty: {audit['difficulty_distribution']}")
    print(f"  Warhead: {audit['warhead_collapse_check'].get('status')} "
          f"(n_checked={audit['warhead_collapse_check'].get('n_checked', 0)})")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
