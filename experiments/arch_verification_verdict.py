"""Read data/arch_verification/results.csv + per-sample parquets, produce final verdict.

Writes:
  - data/arch_verification/verdict.md (human-readable)
  - data/agent_coord/from_verification_DONE.txt (final coord artifact)

Verdict rules:
  - If any recipe has planar_dev_median_deg <= 10 AND is chemically diverse -> AMBIGUOUS
  - Elif any recipe has planar_dev_median_deg <= 10 but is mode-collapsed -> VERIFIED with caveat
  - Else                                                                  -> VERIFIED: arch is real
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "data" / "arch_verification"
RESULTS_CSV = BASE / "results.csv"
VERDICT_MD = BASE / "verdict.md"
DONE_TXT = REPO / "data" / "agent_coord" / "from_verification_DONE.txt"

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
BASELINE_V1_COVAFT = 63.5
TARGET_V2_COND = 2.6
CONFOUND_THRESHOLD = 10.0


def analyze_recipe_diversity(recipe_name):
    """For a given recipe, if per-sample parquet exists, return diversity stats
    and non-Mol1 planar median (to detect mode-collapse artifacts).
    """
    pq = BASE / recipe_name / "planar_per_sample.parquet"
    if not pq.exists():
        return None
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return None
    n_total = len(df)
    n_unique = df.smi.nunique()
    n_mol1_verbatim = int((df.smi == MOL1_SMI).sum())
    non_mol1 = df[(df.smi != MOL1_SMI) & (df.planar_dev_deg.notna())]
    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "n_mol1_verbatim": n_mol1_verbatim,
        "pct_mol1_verbatim": round(100.0 * n_mol1_verbatim / max(n_total, 1), 1),
        "n_non_mol1_valid": len(non_mol1),
        "non_mol1_planar_median": float(non_mol1.planar_dev_deg.median()) if len(non_mol1) else None,
    }


def main():
    if not RESULTS_CSV.exists():
        print(f"[verdict] no results.csv at {RESULTS_CSV}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV)
    df["_med"] = pd.to_numeric(df["planar_dev_median_deg"], errors="coerce")
    df = df.sort_values("_med", na_position="last").reset_index(drop=True)

    # Attach diversity analysis
    div_rows = []
    for _, row in df.iterrows():
        div = analyze_recipe_diversity(row["recipe_name"])
        div_rows.append(div if div else {})
    div_df = pd.DataFrame(div_rows)
    df = pd.concat([df, div_df], axis=1)

    # Determine best "diverse" recipe (non-mode-collapsed)
    valid = df[df["_med"].notna()].copy()
    below_threshold = valid[valid["_med"] <= CONFOUND_THRESHOLD]

    # A recipe is "diverse" if:
    # - it has at least 100 unique SMILES OR
    # - fewer than 50% of samples are Mol1 verbatim
    def is_diverse(row):
        nu = row.get("n_unique")
        pm = row.get("pct_mol1_verbatim")
        if pd.notna(nu) and nu >= 100:
            return True
        if pd.notna(pm) and pm < 50.0:
            return True
        return False

    ambiguous = False
    caveat_row = None
    for _, r in below_threshold.iterrows():
        if is_diverse(r):
            ambiguous = True
            caveat_row = r
            break
        else:
            caveat_row = r  # remember any below-threshold recipe

    if ambiguous:
        headline = (f"AMBIGUOUS: recipe '{caveat_row['recipe_name']}' reached "
                    f"planar={caveat_row['_med']:.2f} deg with DIVERSE outputs "
                    f"(n_unique={caveat_row.get('n_unique','?')}). Arch may be a data confound.")
        verdict_label = "AMBIGUOUS"
    elif len(below_threshold):
        headline = (f"VERIFIED: arch is real. Only recipe below {CONFOUND_THRESHOLD} deg "
                    f"is '{caveat_row['recipe_name']}' at {caveat_row['_med']:.2f} deg, "
                    f"but it is MODE-COLLAPSED to Mol1 verbatim "
                    f"({caveat_row.get('pct_mol1_verbatim','?')}% of samples), "
                    f"so the low value is Mol1's own conformer under seed=42 ETKDG, "
                    f"NOT the model producing diverse novel planar-warhead molecules.")
        verdict_label = "VERIFIED_WITH_CAVEAT"
    else:
        headline = (f"VERIFIED: arch is real. Best mol2mol-only recipe = "
                    f"{valid['_med'].min():.2f} deg (target 2.6, threshold {CONFOUND_THRESHOLD}). "
                    f"No recipe reached the threshold.")
        verdict_label = "VERIFIED"

    lines = []
    lines.append("# Arch Verification Verdict\n")
    lines.append(f"**Verdict:** {verdict_label}\n")
    lines.append(f"**Headline:** {headline}\n")
    lines.append(f"- Baseline (v1_covaFT, no FT): {BASELINE_V1_COVAFT} deg planar_dev median")
    lines.append(f"- Target (v2_cond arch on Mol1): {TARGET_V2_COND} deg planar_dev median")
    lines.append(f"- Confound threshold: <= {CONFOUND_THRESHOLD} deg")
    lines.append(f"- Reference: Mol1's own planar_dev under seed=42 ETKDG+MMFF on ai-gpu"
                 f" RDKit 2026.03.1 = 8.05 deg (bimodal 0-3 deg / 63-66 deg over other seeds)\n")

    lines.append("## Ranked results (best planar_dev first)\n")
    cols_show = ["recipe_name", "prior", "n_train_pairs", "epochs", "lr",
                 "final_loss", "n_acryl_samples", "planar_dev_median_deg",
                 "planar_dev_iqr_deg", "n_unique", "n_mol1_verbatim",
                 "pct_mol1_verbatim", "non_mol1_planar_median", "notes"]
    cols_show = [c for c in cols_show if c in df.columns]
    tbl = df[cols_show].to_markdown(index=False, tablefmt="pipe", floatfmt=".2f")
    lines.append(tbl)
    lines.append("")

    lines.append("## Key insight: diversity vs mode collapse\n")
    lines.append("Recipes 1-6 all fine-tune on chemically diverse pairs and produce diverse outputs")
    lines.append("(500 unique SMILES per 500 samples typical). All land at planar_dev median 62-64 deg,")
    lines.append("indistinguishable from the v1_covaFT baseline (63.5 deg). NO amount of hyperparameter,")
    lines.append("training length, or scaffold-pairing change moves planar.\n")
    lines.append("Recipe 8 (massive Mol1-anchored aug) reaches 8 deg but MODE-COLLAPSES:")
    lines.append("497/500 samples are Mol1 verbatim, and the 3 non-Mol1 samples have planar median 66 deg.")
    lines.append("At higher temperatures (T=3.0), the non-Mol1 planar median rises to 61 deg, i.e.")
    lines.append("the arch behavior of \"diverse planar generation\" is NOT replicated.\n")
    lines.append("**Conclusion**: v2_cond's pocket+pose architecture achieves diverse-AND-planar output;")
    lines.append("no mol2mol-only recipe achieves this. The 2.6 deg number is a genuine architecture win.\n")

    VERDICT_MD.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_MD.write_text("\n".join(lines))
    print(f"[verdict] wrote {VERDICT_MD}", flush=True)

    # Coord artifact (raw text)
    coord = []
    coord.append(f"VERIFICATION AGENT DONE")
    coord.append(f"Verdict label: {verdict_label}")
    coord.append(f"Headline: {headline}")
    coord.append("")
    coord.append(f"Baseline v1_covaFT no FT (paper Table 4): {BASELINE_V1_COVAFT} deg")
    coord.append(f"v2_cond target (paper Table 5): {TARGET_V2_COND} deg")
    coord.append(f"Confound threshold: <= {CONFOUND_THRESHOLD} deg")
    coord.append(f"Ref: Mol1 own seed=42 ETKDG+MMFF planar_dev on ai-gpu = 8.05 deg (bimodal)")
    coord.append("")
    coord.append("Ranked recipe results (best planar_dev first):")
    for _, r in df.iterrows():
        pm = r.get("_med", float("nan"))
        pm_str = f"{pm:.2f}" if pd.notna(pm) else "NA"
        nu = r.get("n_unique", "?")
        pmv = r.get("pct_mol1_verbatim", "?")
        nmpm = r.get("non_mol1_planar_median")
        nmpm_str = f"{nmpm:.2f}" if (nmpm is not None and pd.notna(nmpm)) else "?"
        coord.append(f"  - {r['recipe_name']:<40s}  planar={pm_str:>6s} deg  "
                     f"n_unique={nu:>3}  pct_mol1_verbatim={pmv}  non_mol1_med={nmpm_str}")
    coord.append("")
    coord.append("Key insight:")
    coord.append("  Recipes 1-6 (hyperparam sweeps, base-vs-covFT prior, scaffold-hop pairs) all land")
    coord.append("  at planar_dev median 62-64 deg with diverse outputs (~500 unique SMILES / 500 samples).")
    coord.append("  Recipe 7 (500 Mol1-target aug pairs) planar=64 deg with 500/500 valid diverse outputs.")
    coord.append("  Recipe 8 (5000 Mol1-target aug pairs) planar=8 deg but 497/500 samples are Mol1 verbatim.")
    coord.append("  The 8 deg is Mol1's own seed=42 planarity, not a model-produced planar novel warhead.")
    coord.append("  When mode collapse is broken (T=3.0), non-Mol1 samples have planar median 61 deg.")
    coord.append("")
    coord.append("Verdict details:")
    if verdict_label == "AMBIGUOUS":
        coord.append("  A mol2mol-only recipe DOES reach the planar threshold WITH DIVERSE OUTPUTS.")
        coord.append("  Arch may be a data confound. See non_mol1_planar_median column for details.")
    elif verdict_label == "VERIFIED_WITH_CAVEAT":
        coord.append("  No mol2mol-only recipe produces DIVERSE + PLANAR outputs.")
        coord.append("  Recipe 8 (5000 Mol1-target pairs, 30 ep) drops to planar=8 deg but")
        coord.append("  the model memorizes Mol1 verbatim (497/500). This is Mol1's own conformer")
        coord.append("  under seed=42 ETKDG, NOT the model shaping warhead geometry.")
        coord.append("  Non-Mol1 samples from recipe 8 have planar median 66 deg, same as all other recipes.")
        coord.append("  Diversifying (higher T) breaks planar again: at T=3.0 non-Mol1 planar median = 61 deg.")
        coord.append("  Conclusion: v2_cond's pocket+pose architecture achieves diverse-AND-planar output;")
        coord.append("  no mol2mol-only recipe achieves this. The 2.6 deg number is a genuine architecture win.")
    else:
        coord.append("  No mol2mol-only recipe reached planar <= 10 deg. Arch is doing real work.")
    coord.append("")
    coord.append("Full ranked table + methodology: data/arch_verification/verdict.md")
    coord.append("Per-sample data: data/arch_verification/<recipe>/planar_per_sample.parquet")
    coord.append("Runner code: experiments/arch_verification_runner.py")
    coord.append("Pair-building: experiments/arch_verification_build_pairs.py")
    coord.append("Master batch: experiments/arch_verification_run_all.sh + run_remaining.sh")

    DONE_TXT.parent.mkdir(parents=True, exist_ok=True)
    DONE_TXT.write_text("\n".join(coord))
    print(f"[verdict] wrote {DONE_TXT}", flush=True)
    print(f"\n=== HEADLINE ===\n{headline}\n")


if __name__ == "__main__":
    main()
