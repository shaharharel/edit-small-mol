"""QA #4 (2026-06-24): MASTER cross-experiment integration table.

Reads outputs from QA1/2/3 + originals, builds:
  - results/paper_evaluation/MASTER_metrics_table.csv
  - results/paper_evaluation/MASTER_audit.md

One row per cohort (priors, DAP, PPO/DPO variants, distilled-iptm,
reward ablations, sampling baselines). Uniform metric columns where
available (validity, uniqueness, scaffold_uniq, acrylamide_pct,
thiq_core_pct, FiLM pIC50 mean, QED mean, Tc_to_Mol1 mean, FCD v2).

This script is purely an aggregator over already-produced JSON/MD files.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log(f"  failed to load {p}: {e}")
        return None


def main():
    log("=== QA #4: MASTER table assembly ===")

    rows = []

    # --- Exp 1 cohorts (base / covft / warhead_tokens priors), 10k each ---
    exp1 = load_json(OUT_DIR / "exp1_covft_metrics.json")
    exp1plus = load_json(OUT_DIR / "exp1plus_substantive_metrics.json")
    fcdv2 = load_json(OUT_DIR / "exp1plus_fcd_v2.json")
    hinge_masked = load_json(OUT_DIR / "exp1plus_hinge_masked.json")
    for nm in ("base", "covft", "warhead_tokens"):
        r = {"cohort": f"prior_{nm}", "stream": "Exp1"}
        if exp1 and nm in exp1:
            c = exp1[nm]
            r.update({
                "n_total": c.get("n_total"),
                "n_valid": c.get("n_valid"),
                "validity_pct": c.get("validity_pct"),
                "uniqueness_pct_among_valid": c.get("uniqueness_pct_among_valid"),
                "scaffold_uniq_pct": c.get("scaffolds", {}).get("scaffold_uniqueness_pct"),
                "mw_mean": c.get("descriptors", {}).get("MolWt", {}).get("mean"),
                "logp_mean": c.get("descriptors", {}).get("LogP", {}).get("mean"),
                "qed_mean": c.get("descriptors", {}).get("QED", {}).get("mean"),
                "sa_mean": c.get("descriptors", {}).get("SAScore", {}).get("mean"),
                "tpsa_mean": c.get("descriptors", {}).get("TPSA", {}).get("mean"),
                "acrylamide_pct": c.get("warheads", {}).get("by_class_pct", {}).get("acrylamide"),
                "any_warhead_pct": c.get("warheads", {}).get("any_warhead_pct"),
                "mol1_body_pct": c.get("warheads", {}).get("mol1_warhead_body_pct"),
                "tc_mol1_mean": c.get("similarity_to_mol1", {}).get("mean_tc_to_mol1"),
                "tc_mol1_pct_ge_p4": c.get("similarity_to_mol1", {}).get("pct_above_threshold"),
            })
        # Exp1+ adds 4-way, ATP, hinge
        if exp1plus and "results" in exp1plus and nm in exp1plus["results"]:
            c2 = exp1plus["results"][nm]
            r["four_way_pct"] = c2.get("four_way_conjunction", {}).get("pct")
            r["hinge_any_pct_raw"] = c2.get("hinge_pharmacophore", {}).get("any_hinge_pct")
            r["atp_window_pct"] = c2.get("atp_window", {}).get("pct")
        # FCD v1 + v2
        if exp1plus and "fcd" in exp1plus and nm in exp1plus["fcd"]:
            r["fcd_v1"] = exp1plus["fcd"][nm].get("value")
        if fcdv2 and "per_cohort" in fcdv2 and nm in fcdv2["per_cohort"]:
            r["fcd_v2"] = fcdv2["per_cohort"][nm].get("value")
        # Hinge masked
        if hinge_masked and nm in hinge_masked:
            hm = hinge_masked[nm]["pct"]
            r["hinge_masked_any_pct"] = hm.get("any_hinge")
            r["hinge_masked_novel_any_pct"] = hm.get("novel_hinge_any")
            r["hinge_masked_novel_non_mol1_pct"] = hm.get("novel_hinge_other_than_mol1class")
        rows.append(r)

    # --- Exp 3 ablation + Exp B distilled-iptm cohort ---
    exp3 = load_json(OUT_DIR / "exp3_ablation_and_distilled_iptm.json")
    if exp3 and "rows" in exp3:
        for c in exp3["rows"]:
            if c.get("error"):
                rows.append({"cohort": c.get("label"), "stream": "Exp3/B", "note": c["error"]})
                continue
            stream = "Exp3" if "drop" in c["label"] else (
                "ExpB" if "distilled" in c["label"] else (
                "Exp4" if "ppo" in c["label"] else (
                "Exp2" if "prior" in c["label"] else "Exp3" )))
            r = {
                "cohort": c["label"],
                "stream": stream,
                "n_total": c.get("n_total"),
                "n_valid": c.get("n_valid"),
                "validity_pct": c.get("validity_pct"),
                "uniqueness_pct_among_valid": c.get("uniqueness_pct_among_valid"),
                "scaffold_uniq_pct": c.get("scaffold_uniqueness_pct"),
                "mw_mean": c.get("mw_mean"),
                "logp_mean": c.get("logp_mean"),
                "qed_mean": c.get("qed_mean"),
                "sa_mean": c.get("sa_mean"),
                "tpsa_mean": c.get("tpsa_mean"),
                "fsp3_mean": c.get("fsp3_mean"),
                "acrylamide_pct": c.get("acrylamide_pct"),
                "thiq_core_pct": c.get("thiq_core_pct"),
                "tc_mol1_mean": c.get("tc_mol1_mean"),
                "tc_mol1_p90": c.get("tc_mol1_p90"),
                "filmdelta_pic50_mean": c.get("filmdelta_pic50_mean"),
                "filmdelta_pic50_p90": c.get("filmdelta_pic50_p90"),
                "distilled_iptm_mean": c.get("distilled_iptm_mean"),
            }
            rows.append(r)

    # --- Exp 4 PPO/DPO log-level summary (no full cohort scoring yet) ---
    exp4 = load_json(OUT_DIR / "exp4_ppo_dpo_summary.json")
    if exp4 and "variants" in exp4:
        for v in exp4["variants"]:
            if v.get("error"):
                rows.append({"cohort": v.get("label"), "stream": "Exp4", "note": v["error"]})
                continue
            r = {
                "cohort": v["label"],
                "stream": "Exp4",
                "n_steps": v.get("n_steps"),
                "reward_final": v.get("reward_final"),
                "reward_max": v.get("reward_max"),
                "kl_final": v.get("kl_final"),
                "kl_max": v.get("kl_max"),
                "ent_final": v.get("entropy_final"),
                "warhead_final": v.get("warhead_any_rate_final"),
                "warhead_mean": v.get("warhead_any_rate_mean"),
                "warhead_max": v.get("warhead_any_rate_max"),
                "thiq_final": v.get("thiq_exact_rate_final"),
                "thiq_max": v.get("thiq_exact_rate_max"),
            }
            rows.append(r)

    # Convert
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "MASTER_metrics_table.csv"
    df.to_csv(csv_path, index=False)
    log(f"Wrote {csv_path} with {len(df)} rows")

    # Build a streamlined audit MD
    lines = [
        "# MASTER Audit — Overnight Campaign 2026-06-23 to 2026-06-24\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n",
        "## Per-stream verdict\n",
    ]

    streams_verdict = [
        ("Exp1", "Cov FT value — 3 priors × 10k samples",
         "DONE", "READY",
         "All 3 cohort CSVs (10k each, 98-99% valid) + metrics JSON/PNG/MD present.",
         "Warhead retention: base 1.4% → covft 97.2% → warhead_tokens 99.9% (Δ=+98pp). Mol1-body retention 96.5% / 97.5%.",
         "(this campaign) no extra action — Exp1+ deeper analysis covers it"),
        ("Exp1+", "Substantive cov-kinase metrics (FCD, hinge, 4-way conjunction)",
         "DONE+REVISED", "READY (with corrections)",
         "Original JSON/MD present; FCD v2 + hinge-masked recomputed this campaign.",
         "Headline 4-way: base 0.55% → covft 20.8% → warhead_tokens 23.3% (Δ=+22.7pp). "
         "FCD v1 (3k ref): 25.3→24.2→28.2 (only 1.1u shift). FCD v2 (8.8k ref): 23.7→24.1→28.1 — "
         "expanded reference does NOT rescue covft FCD; the prior really is close-but-different. "
         "Hinge MASKED: base 0.9%, covft 4.7%, warhead_tokens 10.4% non-Mol1-class hinge — confirms "
         "original 'hinge presence' was almost entirely parent leakage; the true novel-hinge signal "
         "is small (~10% in warhead_tokens) but cohort-ordered correctly.",
         "Output: exp1plus_fcd_v2.json, exp1plus_hinge_masked.json, exp1plus_fcd_v2_summary.md"),
        ("Exp2", "RL value baselines (4 expected)",
         "PARTIAL", "NEEDS-WORK",
         "2/4 expected baselines present locally (covft, warhead_tokens, Mol1-only seed). 2 with "
         "kinase/zap70 seed are running on a100-b; sampling.json metadata present but no CSV yet "
         "as of 21:30 UTC.",
         "Mol1-only covft baseline metrics on file (acrylamide 96.9%, QED 0.73, Tc_mol1 0.66).",
         "Pull the kinase+zap70 seed CSVs from a100-b once sampling completes; re-run QA2 with all 4."),
        ("Exp3", "Reward ablation (3 cohorts)",
         "DONE+ANALYSIS", "READY",
         "3 ablation cohorts present (drop_film 84k full = 50 steps, drop_smarts 8k = 5 steps "
         "early-term, drop_qed 8k = 5 steps early-term). QA2 computes uniform metrics + subsamples "
         "drop_film to 5-step window for fair comparison.",
         "(see MASTER table for headline numbers)",
         "Output: exp3_ablation_and_distilled_iptm.json, exp3_ablation_and_distilled_iptm.md"),
        ("Exp4-v1", "PPO baseline",
         "DONE (negative result)", "READY (negative)",
         "5500-mol sample cohort + log present. Binary gate kills gradient signal.",
         "warhead retention 0.4% — collapsed, no learning.",
         "Documented as the failure case that motivates v2+ ladder; no additional action."),
        ("Exp4-v2+", "PPO iterative SOTA hunt on a100-b",
         "RUNNING (PARTIAL)", "NEEDS-WORK",
         "v2 (collapsed late, KL=394), v2b (stable, 5% warhead, no improvement over prior), "
         "v3 GRPO (stable, 7-26% warhead), v4 entropy curriculum (28/100 steps, 12% final), "
         "v5 RLOO (50 steps, 10% warhead), **v6 combo (50 steps, 34% mean / 56% max warhead, "
         "1.7% thiq exact — NEW BEST PPO**), DPO-v1 (42 epochs, 9.8% warhead, 0% thiq).",
         "PPO v6 is the only variant approaching the DAP cohort's regime (DAP: 98%/94% warhead/thiq). "
         "Still ~3x worse than DAP on warhead; PPO objective fundamentally harder than DAP.",
         "Executor continues iterating on a100-b. After v6 sample cohort lands, re-run QA3+QA4 "
         "with the full sampled cohort metrics (currently only log-level)."),
        ("Exp5", "FiLMDelta vs direct (basic, ZAP70 small-data)",
         "DONE", "READY",
         "JSON + PNG + log present (3001 pairs, 257 unique mols, 5 train sizes × 3 seeds).",
         "FiLMDelta beats direct by ~7.5% MAE at N=25 (0.675 vs 0.730), 4.7% at N=50, "
         "converging at N>=200. Signal is real, modest in absolute terms.",
         "(this campaign) no extra action; Exp5b/c are deep extensions on ai-gpu2."),
        ("Exp5b/5c", "FiLMDelta deep splits + architecture matrix on ai-gpu2",
         "RUNNING on ai-gpu2 BUT SSH UNREACHABLE", "BLOCKED",
         "ai-gpu2 SSH down across 4 attempts (banner timeout). Serial console shows DHCP "
         "timeouts on ens5 but instance is RUNNING. tmux sessions exp5b/exp5c presumably still "
         "writing results JSON to local disk on ai-gpu2 — but we cannot fetch them.",
         "n/a — no result files retrievable.",
         "**USER FLAG:** ai-gpu2 SSH down. Options: (1) wait for SSH recovery, (2) request "
         "VM restart via console, (3) accept Exp5b/c as out-of-scope for morning deliverable "
         "and ship Exp5 basic only. Recommend (1) — exp5b/c JSON snapshot files exist on disk; "
         "scp once SSH recovers."),
        ("ExpB", "Distilled-iptm 4-component RL on V100 (stopped)",
         "DONE+ANALYSIS", "READY",
         "21,274-row cohort.csv present. QA2 includes uniform metrics for it.",
         "thiq_core retention 99.9%, acrylamide 99.9%, FiLM pIC50 mean ~6.16, QED ~0.42, "
         "distilled iptm 0.86. Best on all measured composite quality vs DAP 3-comp.",
         "Output: row 'distilled_iptm_4comp' in MASTER table."),
    ]

    for nm, desc, status, ready, present, headline, action in streams_verdict:
        lines.append(f"### {nm}: {desc}\n")
        lines.append(f"- **Status**: {status}")
        lines.append(f"- **Paper-readiness**: {ready}")
        lines.append(f"- **Outputs**: {present}")
        lines.append(f"- **Headline**: {headline}")
        lines.append(f"- **Action / outstanding**: {action}\n")

    lines.append("## MASTER table (CSV companion)\n")
    lines.append(f"Full table: `{csv_path.relative_to(PROJECT_ROOT)}` ({len(df)} rows)\n")

    # Inline a small priority subset
    inline_cols = [
        "cohort", "stream", "n_valid", "validity_pct", "uniqueness_pct_among_valid",
        "scaffold_uniq_pct", "acrylamide_pct", "thiq_core_pct",
        "qed_mean", "sa_mean", "tc_mol1_mean",
        "filmdelta_pic50_mean", "distilled_iptm_mean",
        "fcd_v1", "fcd_v2",
        "warhead_mean", "thiq_max", "reward_final",
    ]
    inline_cols = [c for c in inline_cols if c in df.columns]
    lines.append("## Headline cohort-level metrics (subset of MASTER table)\n")
    lines.append("| " + " | ".join(inline_cols) + " |")
    lines.append("|" + "---|" * len(inline_cols))
    for _, r in df.iterrows():
        row_vals = []
        for c in inline_cols:
            v = r.get(c)
            if pd.isna(v):
                row_vals.append("—")
            elif isinstance(v, float):
                if c.endswith("_pct"):
                    row_vals.append(f"{v:.1f}%")
                else:
                    row_vals.append(f"{v:.3f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")

    md = OUT_DIR / "MASTER_audit.md"
    md.write_text("\n".join(lines))
    log(f"Wrote {md}")
    log("DONE")


if __name__ == "__main__":
    main()
