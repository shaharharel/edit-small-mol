#!/usr/bin/env python3
"""Backfill cohort_diversity_metrics.csv for every cohort in
`all_cohorts_metrics.csv` using the shared `src.utils.diversity` panel.

Output schema (one row per cohort):
    cohort, group, n_total, n_unique, diversity_ratio,
    mean_intra_nn_tanimoto, max_intra_tanimoto,
    n_bemis_murcko_scaffolds, mean_pairwise_dissim

Reproduces (and extends) the previously emitted `diversity_table.csv`. The
columns common to both must match (n_total / n_unique / diversity_ratio /
mean_intra_NN_tanimoto). The extra columns (max, scaffolds, dissim) are new.

Usage:
    conda run -n quris python experiments/cohort_diversity_backfill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.diversity import compute_diversity_metrics  # noqa: E402

# Group labels copied from cohort_comparison_unique.py so the two artefacts
# agree on which cohort is in which group.
GROUP_A_SEQONLY = {"DeNovo_warhead_gate", "Mol2Mol_warhead_gate"}
GROUP_B_LINGO = {"L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2"}
GROUP_C_HYBRID = {"LibInvent_locked", "LibInvent_locked_FIXED", "Amine_Replacements"}
GROUP_D_AUDIT = {"LibInvent_locked_OLD_pyrrolidinol"}

GROUP_MAP = {
    **{c: "seqonly_A" for c in GROUP_A_SEQONLY},
    **{c: "lingo_B" for c in GROUP_B_LINGO},
    **{c: "hybrid_C" for c in GROUP_C_HYBRID},
    **{c: "audit_OLD" for c in GROUP_D_AUDIT},
}

COHORT_ORDER = [
    "L0_vanilla", "L_locked", "H1", "H2", "H3", "C1", "C5", "L1_FT_H2",
    "DeNovo_warhead_gate", "Mol2Mol_warhead_gate",
    "LibInvent_locked_FIXED", "Amine_Replacements",
    "LibInvent_locked_OLD_pyrrolidinol",
]

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison"
INPUT_CSV = RESULTS_DIR / "all_cohorts_metrics.csv"
OUT_CSV = RESULTS_DIR / "cohort_diversity_metrics.csv"


def main():
    print(f"[load] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  rows = {len(df)}, cohorts = {df['cohort'].nunique()}")

    cohorts_present = [c for c in COHORT_ORDER if c in df["cohort"].unique()]
    others = sorted(set(df["cohort"].unique()) - set(cohorts_present))
    cohorts_present.extend(others)

    rows = []
    for cohort in cohorts_present:
        sub = df[df["cohort"] == cohort]
        smiles_raw = sub["smi"].tolist()
        print(f"[compute] {cohort}: n={len(smiles_raw)} ...", flush=True)
        m = compute_diversity_metrics(smiles_raw)
        rows.append({
            "cohort": cohort,
            "group": GROUP_MAP.get(cohort, "unknown"),
            "n_total": m["n_total"],
            "n_unique": m["n_unique"],
            "diversity_ratio": m["diversity_ratio"],
            "mean_intra_nn_tanimoto": m["mean_intra_nn_tanimoto"],
            "max_intra_tanimoto": m["max_intra_tanimoto"],
            "n_bemis_murcko_scaffolds": m["n_bemis_murcko_scaffolds"],
            "mean_pairwise_dissim": m["mean_pairwise_dissim"],
        })
        last = rows[-1]
        print(
            f"  -> n_uniq={last['n_unique']}, div_ratio={last['diversity_ratio']:.3f}, "
            f"mean_nn_Tc={last['mean_intra_nn_tanimoto']:.3f}, "
            f"max_Tc={last['max_intra_tanimoto']:.3f}, "
            f"BM_scaf={last['n_bemis_murcko_scaffolds']}, "
            f"mean_dissim={last['mean_pairwise_dissim']:.3f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}")
    print(out.to_string())


if __name__ == "__main__":
    main()
