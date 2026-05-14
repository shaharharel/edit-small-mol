"""Score the Cys346 cofold batch — merge manifest with Boltz confidence JSONs.

Reads:
  - experiments/boltz_inputs/top1000__zap70_cys346/manifest.csv (1000 candidate mols)
  - data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/<name>/confidence_<name>_model_0.json (one per cofolded mol)

Joins on yaml_name and writes:
  - results/anchordiff/cys346_cofold_leaderboard.csv (full)
  - prints top-20 by combined score (FiLMDelta_pIC50 ranked + iptm tiebreak)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

MANIFEST = PROJECT_ROOT / "experiments" / "boltz_inputs" / "top1000__zap70_cys346" / "manifest.csv"
PREDICTIONS = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
OUT_CSV = PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv"


def read_confidence(name: str) -> dict:
    p = PREDICTIONS / name / f"confidence_{name}_model_0.json"
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    out = {}
    for k in ["confidence_score", "ptm", "iptm", "ligand_iptm",
              "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        if k in d:
            out[f"boltz_{k}"] = float(d[k])
    return out


def main():
    df = pd.read_csv(MANIFEST)
    print(f"manifest rows: {len(df)}")
    rows = []
    n_missing = 0
    for _, r in df.iterrows():
        c = read_confidence(r["yaml_name"])
        if not c:
            n_missing += 1
            rec = r.to_dict()
            for k in ["confidence_score", "ptm", "iptm", "ligand_iptm",
                      "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
                rec[f"boltz_{k}"] = None
        else:
            rec = {**r.to_dict(), **c}
        rows.append(rec)
    full = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_CSV, index=False)
    print(f"missing cofold: {n_missing}/{len(df)}")
    print(f"wrote {OUT_CSV}")

    have = full.dropna(subset=["boltz_iptm"]).copy()
    print(f"\ncofolded mols: {len(have)}")
    print(f"\niptm distribution:")
    print(have["boltz_iptm"].describe())
    print(f"\nligand_iptm distribution:")
    print(have["boltz_ligand_iptm"].describe())

    # Combined score: rank_score (FiLMDelta pIC50) + ligand_iptm bonus
    # Candidates that are both potent (high pIC50) and have a confident pose
    # (high ligand_iptm) bubble up.
    have["combined"] = have["rank_score"].rank(pct=True) * 0.6 + \
                       have["boltz_ligand_iptm"].rank(pct=True) * 0.4

    print("\n=== TOP 20 BY COMBINED FiLMDelta+Boltz SCORE ===")
    cols = ["yaml_name", "method", "MW", "rank_score", "boltz_ligand_iptm",
            "boltz_iptm", "boltz_complex_plddt", "combined"]
    top20 = have.nlargest(20, "combined")[cols + ["smiles"]]
    for _, r in top20.iterrows():
        print(f"  pIC50={r['rank_score']:.2f}  ligIPTM={r['boltz_ligand_iptm']:.3f}  "
              f"pLDDT={r['boltz_complex_plddt']:.2f}  MW={r['MW']:.0f}  "
              f"[{r['method'][:18]}]  {r['smiles'][:60]}")


if __name__ == "__main__":
    main()
