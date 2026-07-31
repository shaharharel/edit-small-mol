"""QA #6 (2026-06-24): Final MASTER table assembly — merges all PPO post-RL
sampled cohorts (prior_baseline, v1c, v2b, v3_grpo) from
data/exp_ppo_v2plus/cohort_*.csv.summary.json and updates the headline tables.

Also folds in:
  - QA1: FCD v2 + hinge masked
  - QA2: cohort uniform metrics (Exp 3, Exp B, plus priors and DAP_full)
  - QA3: PPO/DPO log-level summary
  - QA5: late-step RL stream snapshot
  - PPO variant summary CSV from a100-b (4-row partial)

Outputs:
  - results/paper_evaluation/MASTER_metrics_table.csv (FINAL)
  - results/paper_evaluation/MASTER_audit.md (FINAL)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
PPO_DIR = PROJECT_ROOT / "data" / "exp_ppo_v2plus"


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
    log("=== QA #6: FINAL MASTER table assembly ===")

    rows = []

    # --- Exp 1 cohorts ---
    exp1 = load_json(OUT_DIR / "exp1_covft_metrics.json")
    exp1plus = load_json(OUT_DIR / "exp1plus_substantive_metrics.json")
    fcdv2 = load_json(OUT_DIR / "exp1plus_fcd_v2.json")
    hinge_masked = load_json(OUT_DIR / "exp1plus_hinge_masked.json")
    for nm in ("base", "covft", "warhead_tokens"):
        r = {"cohort": f"prior_{nm}", "stream": "Exp1",
             "comp_class": "prior_sampling", "n_steps": "n/a"}
        if exp1 and nm in exp1:
            c = exp1[nm]
            r.update({
                "n_total": c.get("n_total"),
                "n_valid": c.get("n_valid"),
                "validity_pct": c.get("validity_pct"),
                "uniqueness_pct": c.get("uniqueness_pct_among_valid"),
                "scaffold_uniq_pct": c.get("scaffolds", {}).get("scaffold_uniqueness_pct"),
                "qed_mean": c.get("descriptors", {}).get("QED", {}).get("mean"),
                "sa_mean": c.get("descriptors", {}).get("SAScore", {}).get("mean"),
                "acrylamide_pct": c.get("warheads", {}).get("by_class_pct", {}).get("acrylamide"),
                "mol1_body_pct": c.get("warheads", {}).get("mol1_warhead_body_pct"),
                "tc_mol1_mean": c.get("similarity_to_mol1", {}).get("mean_tc_to_mol1"),
            })
        if exp1plus and "results" in exp1plus and nm in exp1plus["results"]:
            c2 = exp1plus["results"][nm]
            r["four_way_pct"] = c2.get("four_way_conjunction", {}).get("pct")
        if exp1plus and "fcd" in exp1plus and nm in exp1plus["fcd"]:
            r["fcd_v1"] = exp1plus["fcd"][nm].get("value")
        if fcdv2 and "per_cohort" in fcdv2 and nm in fcdv2["per_cohort"]:
            r["fcd_v2"] = fcdv2["per_cohort"][nm].get("value")
        if hinge_masked and nm in hinge_masked:
            hm = hinge_masked[nm]["pct"]
            r["hinge_novel_non_mol1class_pct"] = hm.get("novel_hinge_other_than_mol1class")
        rows.append(r)

    # --- Exp 3 NATIVE ablation analysis (executor's exp3_ablation_analysis.json) ---
    exp3_native = load_json(OUT_DIR / "exp3_ablation_analysis.json")
    if exp3_native and isinstance(exp3_native, list):
        for r in exp3_native:
            rows.append({
                "cohort": f"exp3_native_{r['cohort']}",
                "stream": "Exp3-native",
                "comp_class": "RL_post_sampling" if r['cohort'] == 'full_dap' else "RL_during_training",
                "n_total": r.get("n_total"),
                "n_valid": r.get("n_valid"),
                "validity_pct": r.get("validity_pct"),
                "uniqueness_pct": r.get("uniqueness_pct"),
                "scaffold_uniq_pct": 100 * r.get("scaffold_uniqueness", 0) if r.get("scaffold_uniqueness") else None,
                "qed_mean": r.get("qed_mean"),
                "thiq_core_pct": 100 * r.get("smarts_mean_retention", 0) if r.get("smarts_mean_retention") is not None else None,
                "filmdelta_pic50_mean": r.get("film_mean"),
                "filmdelta_pic50_median": r.get("film_median"),
                "filmdelta_pic50_p90": r.get("film_p90"),
                "filmdelta_pic50_source": r.get("film_source"),
            })

    # --- Exp 2 prior baselines (covft + warhead_tokens, Mol1-only) ---
    exp3_qa2 = load_json(OUT_DIR / "exp3_ablation_and_distilled_iptm.json")
    if exp3_qa2 and "rows" in exp3_qa2:
        for c in exp3_qa2["rows"]:
            if c.get("error"):
                continue
            lbl = c["label"]
            stream = "?"
            cclass = "RL_during_training"
            if lbl.startswith("prior_"):
                stream = "Exp2"
                cclass = "prior_sampling_mol1seed"
            elif lbl == "dap_full_3comp":
                stream = "Exp3-baseline"
                cclass = "RL_post_sampling"
            elif lbl == "distilled_iptm_4comp":
                stream = "ExpB"
                cclass = "RL_during_training"
            elif lbl.startswith("drop_") or lbl == "ppo_v1_collapsed":
                stream = "Exp3" if "drop" in lbl else "Exp4"
                cclass = "RL_during_training"
            r = {
                "cohort": lbl, "stream": stream, "comp_class": cclass,
                "n_total": c.get("n_total"),
                "n_valid": c.get("n_valid"),
                "validity_pct": c.get("validity_pct"),
                "uniqueness_pct": c.get("uniqueness_pct_among_valid"),
                "scaffold_uniq_pct": c.get("scaffold_uniqueness_pct"),
                "qed_mean": c.get("qed_mean"),
                "sa_mean": c.get("sa_mean"),
                "acrylamide_pct": c.get("acrylamide_pct"),
                "thiq_core_pct": c.get("thiq_core_pct"),
                "tc_mol1_mean": c.get("tc_mol1_mean"),
                "filmdelta_pic50_mean": c.get("filmdelta_pic50_mean"),
                "distilled_iptm_mean": c.get("distilled_iptm_mean"),
            }
            rows.append(r)

    # --- PPO post-RL sampled cohorts: per-cohort summary.json files ---
    for sjson in sorted(PPO_DIR.glob("cohort_*.csv.summary.json")):
        try:
            c = json.loads(sjson.read_text())
        except Exception as e:
            log(f"  failed to load {sjson}: {e}")
            continue
        variant = sjson.name.replace("cohort_", "").replace(".csv.summary.json", "")
        r = {
            "cohort": f"ppo_post_RL_{variant}",
            "stream": "Exp4",
            "comp_class": "RL_post_sampling",
            "n_total": c.get("n_total"),
            "n_valid": c.get("n_valid"),
            "validity_pct": 100 * c["valid_rate"] if "valid_rate" in c else None,
            "qed_mean": c.get("mean_QED"),
            "acrylamide_pct": 100 * c.get("warhead_any_rate", 0),
            "thiq_core_pct": 100 * c.get("thiq_exact_rate", 0),
            "tc_mol1_mean": c.get("internal_mean_tc_sub500"),
            "filmdelta_pic50_mean": c.get("mean_film_pIC50"),
            "filmdelta_pic50_median": c.get("median_film_pIC50"),
            "filmdelta_pic50_max": c.get("max_film_pIC50"),
            "filmdelta_pic50_frac_ge_7": 100 * c.get("frac_film_ge_7", 0),
            "filmdelta_pic50_frac_ge_8": 100 * c.get("frac_film_ge_8", 0),
            "scaffolds_per_mol": c.get("scaffolds_per_mol_valid"),
        }
        rows.append(r)

    # --- PPO log-level summary (for in-progress v4/v5/v6/dpo and v6b) ---
    exp4_log = load_json(OUT_DIR / "exp4_ppo_dpo_summary.json")
    if exp4_log and "variants" in exp4_log:
        for v in exp4_log["variants"]:
            if v.get("error"):
                continue
            r = {
                "cohort": f"ppo_log_{v['label']}",
                "stream": "Exp4",
                "comp_class": "RL_training_log",
                "n_steps": v.get("n_steps"),
                "reward_final": v.get("reward_final"),
                "reward_max": v.get("reward_max"),
                "kl_max": v.get("kl_max"),
                "ent_final": v.get("entropy_final"),
                "warhead_pct_log_mean": 100 * v["warhead_any_rate_mean"] if "warhead_any_rate_mean" in v else None,
                "warhead_pct_log_final": 100 * v["warhead_any_rate_final"] if "warhead_any_rate_final" in v else None,
                "warhead_pct_log_max": 100 * v["warhead_any_rate_max"] if "warhead_any_rate_max" in v else None,
                "thiq_pct_log_mean": 100 * v["thiq_exact_rate_mean"] if "thiq_exact_rate_mean" in v else None,
                "thiq_pct_log_max": 100 * v["thiq_exact_rate_max"] if "thiq_exact_rate_max" in v else None,
            }
            rows.append(r)

    # --- v6b log (newest, in-progress on a100-b) ---
    v6b_log_path = PPO_DIR / "ppo_v6b_log.csv"
    if v6b_log_path.exists():
        df = pd.read_csv(v6b_log_path)
        r = {
            "cohort": "ppo_log_ppo_v6b",
            "stream": "Exp4",
            "comp_class": "RL_training_log",
            "n_steps": len(df),
            "reward_final": float(df["mean_reward"].iloc[-1]),
            "reward_max": float(df["mean_reward"].max()),
            "kl_max": float(df["mean_kl_to_prior"].abs().max()),
            "ent_final": float(df["mean_entropy"].iloc[-1]),
            "warhead_pct_log_mean": float(100 * df["warhead_any_rate"].mean()),
            "warhead_pct_log_final": float(100 * df["warhead_any_rate"].iloc[-1]),
            "warhead_pct_log_max": float(100 * df["warhead_any_rate"].max()),
            "thiq_pct_log_mean": float(100 * df["thiq_exact_rate"].mean()),
            "thiq_pct_log_max": float(100 * df["thiq_exact_rate"].max()),
        }
        rows.append(r)

    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "MASTER_metrics_table.csv"
    df_out.to_csv(csv_path, index=False)
    log(f"Wrote {csv_path} with {len(df_out)} rows")

    # MD audit
    lines = [
        "# MASTER Audit — Overnight Campaign 2026-06-23 to 2026-06-24\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n",
        "## Per-stream verdict\n",
    ]

    streams_verdict = [
        ("Exp1", "Cov FT value — 3 priors × 10k samples",
         "DONE", "READY",
         "All 3 cohort CSVs (10k each, 98-99% valid) + metrics JSON/PNG/MD present.",
         "Warhead retention: base 1.4% → covft 97.2% → warhead_tokens 99.9%. "
         "Mol1-body retention 96.5% / 97.5%. Validity 99.6/98.9/99.3%. "
         "Uniqueness 25.3/33.0/10.9%, scaffold uniq 7.9/9.2/4.1%.",
         "None — fully covered by Exp1+ deeper analysis."),
        ("Exp1+", "Substantive cov-kinase metrics (FCD, hinge, 4-way conjunction)",
         "DONE+REVISED", "READY (with corrections)",
         "Original JSON/MD + this campaign's FCD v2 + hinge-masked recompute.",
         "Headline 4-way conjunction: base 0.55% → covft 20.8% → warhead_tokens 23.3% (Δ=+22.7pp). "
         "**FCD v1** (3k ref): 25.3→24.2→28.2. **FCD v2** (8.8k ref incl. CovInDB+broader ChEMBL kinase pIC50>=7): "
         "23.7→24.1→28.1 — expanded reference does NOT change the picture; covft prior is genuinely "
         "close-but-different (Δ_v2 = -0.4 units). **Hinge MASKED** (after subtracting Mol1's 4-aminoimidazole "
         "atoms): base 0.9%, covft 4.7%, warhead_tokens 10.4% non-Mol1-class hinge — confirms the "
         "original 'hinge presence' was almost entirely parent leakage. The true novel-hinge signal "
         "is small but warhead_tokens is the clear winner (10.4% vs 0.9% base, a 11x ratio).",
         "DONE. Outputs: exp1plus_fcd_v2{.json,_summary.md}, exp1plus_hinge_masked.json"),
        ("Exp2", "RL value baselines (4 expected)",
         "PARTIAL", "NEEDS-WORK",
         "Mol1-only baselines (covft, warhead_tokens) PRESENT locally. kinase-seed and zap70-seed "
         "baselines were RUNNING on a100-b (sampling.json metadata) but cohort CSVs not yet written.",
         "Mol1-only covft baseline: acrylamide 96.9%, QED 0.73, tc_mol1 0.66 (10K mols).",
         "After a100-b sampling finishes (post-deadline if not done by 06:30 UTC), re-run QA2 with all 4."),
        ("Exp3", "Reward ablation (3 cohorts) — DEFINITIVE",
         "DONE+ANALYSIS", "READY",
         "Executor wrote `exp3_ablation_analysis.{json,png,summary.md}` at 00:33 with "
         "RESCORED FiLM/QED/SMARTS native scores for all 5 cohorts (prior, drop_film, "
         "drop_smarts, drop_qed, full_dap=41,667-mol post-RL DAP). My QA2 was a different "
         "view based on raw RL training streams; the executor's analysis is the canonical one.",
         "**Per-component contribution to DAP**: Removing FiLM costs ΔpIC50=0.547 "
         "(full_dap 6.597 → drop_film 6.050 — note drop_film goes BELOW prior 6.483, i.e. FiLM "
         "is what actually lifts pIC50 above the covalent-FT baseline). Removing SMARTS costs "
         "Δwarhead=0.957 (full_dap 0.962 → drop_smarts 0.005 — total collapse, confirms SMARTS "
         "is what enforces THIQ-acrylamide preservation). Removing QED costs ΔQED=0.157 (full_dap "
         "0.596 → drop_qed 0.439). **Each reward term controls its own metric with minimal "
         "cross-interference, except SMARTS-removal also drags QED down** (drug-likeness rides "
         "on the THIQ-acrylamide chassis).",
         "DONE. Outputs: exp3_ablation_{analysis.json,summary.md,analysis.png}, "
         "exp3_ablation_and_distilled_iptm.{json,md} (my QA2), exp3_rl_late_step.{json,md}"),
        ("Exp4-v1", "PPO baseline (collapsed)", "DONE", "READY (negative)",
         "5500-mol sampled cohort + log present.",
         "Post-RL: 14% warhead, 0.5% thiq, FiLM mean 6.21 (a bit above prior 6.16). "
         "Log mean_reward ≈ 0 — gradient signal killed by binary gate.",
         "Documented as failure case motivating PPO v2+ ladder; no further action."),
        ("Exp4-v2+", "PPO iterative SOTA hunt on a100-b",
         "DONE (9 variants sampled, v6b still training)", "READY — NEGATIVE FINDING",
         "Variants: v2 (KL=394 collapse, log only), v2b, v3 GRPO, v4 entropy curriculum, "
         "v5 RLOO, v6 combo, v6b (replay+combo, running), DPO-v1. "
         "Post-RL sampled cohorts WRITTEN AND SCORED: prior_baseline, v1_collapsed, v2b, v3_grpo, "
         "v4_entcurr, v5_rloo, v6_combo, dpo_v1, v6b_combo, v7_refine, dpo_v2, **dpo_v3 (winner)**. "
         "DAP same-infra control CRASHED — REINVENT4 staged_learning on Mol2Mol prior CUDA-OOMs "
         "at 32-39GB GPU memory by step ~7. See exp4_ppo_iteration_summary.md (executor's auth doc).",
         "**Winner: DPO-v3 at 22.3% warhead** (3.5× prior 6.4%), FiLM 5.92 (-3% vs baseline 6.16), "
         "QED 0.45, 1052 warhead-positive mols. Recipe: PPO-v6 warmstart → 80 DPO steps β=0.3, "
         "k_pairs=24. **Standalone stable winner: PPO-v6** (warhead 14.9%, FiLM at baseline). "
         "**Quality winner: DPO-v1** (warhead 12.1% + highest QED 0.48). Executor's authoritative "
         "interpretation: (a) reward shaping > algorithm — SMARTS-ladder+Tanimoto-tail lifted "
         "non-zero-reward rate from 0.4% to 80%; (b) NO variant moved thiq_exact above prior's "
         "1.2%; algorithmic tweaks only doubled any-Michael-acceptor rate; (c) DAP_full_3comp "
         "(60K-mol post-RL cohort from earlier run) shows 98.6% warhead — confirms DAP is "
         "currently the only RL approach to reach the >95% chassis-retention regime on this task. "
         "**Paper claim**: 'PPO+DPO algorithms produce 3.5× warhead-enrichment over prior; reaching "
         "the ~98% DAP regime requires either (a) a warhead-enriched prior fine-tune or (b) "
         "retraining the FiLM scorer on a warhead-balanced subset — algorithm choice is not the "
         "bottleneck.'",
         "PPO results FINALIZED. Outputs: data/exp_ppo_v2plus/cohort_*.csv (12 variants), "
         "results/paper_evaluation/exp4_ppo_iteration_summary.md (executor's authoritative doc), "
         "exp4_ppo_dpo_summary.{json,md} (QA3 log-level metrics)."),
        ("Exp5", "FiLMDelta vs direct (basic, ZAP70 small-data)", "DONE", "READY",
         "JSON+PNG+log present. 3001 pairs, 257 mols, 5 sizes × 3 seeds.",
         "FiLMDelta beats direct by 7.5% MAE at N=25 (0.675 vs 0.730), 4.7% at N=50, "
         "converging at N>=200. Spearman gap +1-3 pp consistently in FiLM's favor. "
         "Signal is real, modest in absolute terms.",
         "No action."),
        ("Exp5b", "FiLMDelta deep splits on ai-gpu2",
         "PARTIAL (90 runs landed before OOM)", "READY — nuanced result",
         "ai-gpu2 SSH had a 1h6min outage (21:50-22:55), then stable from 23:58. "
         "OOM-killed at 00:22 UTC (29GB anon-rss exhausted by exp5c competing with exp5b). "
         "Restarted with --resume --workers 1 at 00:26 UTC. Currently ~90 runs in JSON (~109kb).",
         "**Pattern across 90 runs (3 splits × N=25..2401 × ≤5 seeds)**: "
         "FiLMDelta wins on Δ-prediction at all N (5-30% MAE reduction over direct-delta-recon). "
         "vs Direct (absolute pIC50): FiLM wins at LOW-N (N=25, +13-17% MAE; N=50, +6-10%) but "
         "LOSES at HIGH-N (at N=2401 pair_disjoint, direct 0.118 vs film 0.191 = 62% gap; "
         "at N=1942 mol_disjoint, direct 0.185 vs film 0.364 = 97% gap). **Paper framing**: "
         "FiLMDelta is a small-data delta-prediction champion; for absolute prediction at large N, "
         "direct is the strong baseline. Frame as 'within-assay delta prediction at low N' "
         "rather than blanket FiLM superiority.",
         "Output: exp5b_filmdelta_zap70_deep.json (90 runs), exp5b_summary.{csv,md}."),
        ("Exp5c", "FiLMDelta deep architecture matrix on ai-gpu2",
         "BLOCKED (OOM)", "BLOCKED — skip for paper",
         "Launched 4 times this campaign; each launch hit OOM during initialization (it loads "
         "morgan + chemberta2_mtr + molformer_xl + drfp_pair encoders simultaneously, exceeding "
         "ai-gpu2's 29GB RAM — see /var/log dmesg for the kill event at 00:22 UTC). The OOM also "
         "killed exp5b and the tmux server.",
         "n/a — no runs completed.",
         "**USER FLAG:** Exp5c requires either (a) VM upgrade to 64GB+ RAM, or (b) refactor the "
         "script to load encoders one at a time. Recommend skip for this paper — exp5b already "
         "establishes the FiLM-vs-direct nuance — and add Exp5c to future-work."),
        ("ExpB", "Distilled-iptm 4-component RL on V100 (stopped)",
         "DONE+ANALYSIS", "READY (negative finding)",
         "21,274-row cohort.csv present (RL training stream, 50 steps, ~425 mols/step).",
         "Headline (cohort-level, **during RL** — NOT post-RL): acrylamide 14.4%, thiq 0.5%, "
         "FiLM 6.28, QED 0.42, iptm 0.52, Tc-to-Mol1 0.13. Last-5-steps snapshot: "
         "thiq retention raw 0.49, FiLM 6.44 — the IPTM term DOMINATES the policy and pulls it AWAY "
         "from THIQ retention. **Headline finding**: adding iptm to the 3-component DAP reward "
         "destroys warhead/THIQ retention (DAP 3-comp post-RL: 98.6% acryl/94.4% thiq; "
         "distilled 4-comp during-RL: 14.4% acryl/0.5% thiq). The 4-comp reward is over-constrained "
         "and the IPTM term competes destructively with THIQ.",
         "DONE. Outputs in exp3_ablation_and_distilled_iptm.{json,md}, exp3_rl_late_step.{json,md}"),
    ]

    for nm, desc, status, ready, present, headline, action in streams_verdict:
        lines.append(f"### {nm}: {desc}\n")
        lines.append(f"- **Status**: {status}")
        lines.append(f"- **Paper-readiness**: {ready}")
        lines.append(f"- **Outputs**: {present}")
        lines.append(f"- **Headline**: {headline}")
        lines.append(f"- **Action / outstanding**: {action}\n")

    lines.append("## MASTER table (CSV companion)\n")
    lines.append(f"Full table: `results/paper_evaluation/MASTER_metrics_table.csv` ({len(df_out)} rows)\n")

    inline_cols = [
        "cohort", "stream", "comp_class", "n_valid",
        "validity_pct", "qed_mean", "acrylamide_pct", "thiq_core_pct",
        "tc_mol1_mean", "filmdelta_pic50_mean", "filmdelta_pic50_max",
        "filmdelta_pic50_frac_ge_7", "fcd_v2",
        "warhead_pct_log_mean", "warhead_pct_log_max", "thiq_pct_log_max",
        "reward_final", "n_steps",
    ]
    inline_cols = [c for c in inline_cols if c in df_out.columns]
    lines.append("## Headline cohort-level metrics (subset of MASTER table)\n")
    lines.append("| " + " | ".join(inline_cols) + " |")
    lines.append("|" + "---|" * len(inline_cols))
    for _, r in df_out.iterrows():
        row_vals = []
        for c in inline_cols:
            v = r.get(c)
            if pd.isna(v):
                row_vals.append("—")
            elif isinstance(v, float):
                if c.endswith("_pct") or c.endswith("_pct_log_mean") or c.endswith("_pct_log_max") or "_frac_" in c:
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
