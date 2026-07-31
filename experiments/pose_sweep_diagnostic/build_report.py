"""Aggregate Phase 1/2/3 results into a single Markdown report."""
from __future__ import annotations
import json
from pathlib import Path

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = LOCAL_ROOT / "data/paper_pair_training/pose_sweep_diagnostic"
DONE_FLAG = LOCAL_ROOT / "data/agent_coord/from_pose_sweep_diagnostic_DONE.txt"


def format_ks_matrix(mat, features, cohort_order=None):
    """Render KS matrix as markdown table."""
    names = cohort_order or list(mat[features[0]].keys())
    out = []
    for feat in features:
        out.append(f"\n#### KS p-values for {feat}\n")
        header = "| pair |"
        sep = "| --- |"
        for b in names:
            header += f" {b} |"
            sep += " --- |"
        out.append(header)
        out.append(sep)
        for a in names:
            row = f"| **{a}** |"
            for b in names:
                cell = mat[feat][a][b]
                if cell["p"] is None:
                    row += " n/a |"
                else:
                    marker = " (**)" if cell["p"] < 0.01 else (" (*)" if cell["p"] < 0.05 else "")
                    row += f" {cell['p']:.3g}{marker} |"
            out.append(row)
    return "\n".join(out)


def main():
    p1 = json.load(open(OUT_DIR / "phase1_logprob.json"))
    p2 = json.load(open(OUT_DIR / "phase2_free_sampling.json"))
    p3 = json.load(open(OUT_DIR / "phase3_permutation_null.json"))

    lines = []
    def W(s=""):
        lines.append(s)

    W("# Pose-Sweep Diagnostic Report — v2_curriculum_clean")
    W("")
    W("**Mission**: Diagnose whether the `[POSE]` conditioning channel of "
      "v2_curriculum_clean is IGNORED at inference, or is USED but doesn't "
      "propagate to sample distributions. This is Phase A of the pose-sweep "
      "diagnostic protocol.")
    W("")
    W("- **Checkpoint**: `models/v2_curriculum_clean/best.chkpt`")
    W(f"- **Pose normalizer**: mean={p1['pose_mean']}, std={p1['pose_std']}")
    W(f"- **Pocket**: ZAP70-like (cache pocket_idx={p1['pocket_idx']})")
    W(f"- **Anchor**: `C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1` (Mol1)")
    W("- **Hardware**: CPU only")
    W("")

    # ================= Phase 1 =================
    W("## Phase 1 — Teacher-forced log-prob under pose sweep")
    W("")
    W(f"Reference target SMILES (median-pose row from val, i={p1['reference']['target_i_in_val']}):")
    W(f"```")
    W(p1['reference']['target_smi'])
    W(f"```")
    W(f"Reference pose (raw d, θ, φ) = {p1['reference']['pose_raw']}")
    W("")
    W("Base pose for one-at-a-time sweeps = median training pose = "
      f"{p1['median_pose_raw']} (raw).")
    W("")

    def render_sweep(sweep, label):
        W(f"### NLL vs {label}")
        W("")
        W(f"| {label} | NLL | ΔNLL vs min |")
        W("| --- | --- | --- |")
        vals = sweep["values"]
        nlls = sweep["nlls"]
        nll_min = min(nlls)
        for v, nll in zip(vals, nlls):
            W(f"| {v:.2f} | {nll:.3f} | +{nll - nll_min:.3f} |")
        rng = max(nlls) - min(nlls)
        W("")
        W(f"NLL range = **{rng:.3f} nats** across sweep values.")
        W("")

    render_sweep(p1["sweep_d"], "d (Å) — Bürgi-Dunitz distance")
    render_sweep(p1["sweep_theta"], "θ (deg) — Bürgi-Dunitz angle")
    render_sweep(p1["sweep_phi"], "φ (deg) — planar dihedral")

    W("### Reference / control NLLs")
    W("")
    W("| condition | NLL |")
    W("| --- | --- |")
    W(f"| reference val pose (matched to target) | {p1['nll_reference_pose']:.3f} |")
    W(f"| null (zero-raw) pose | {p1['nll_null_pose']:.3f} |")
    W(f"| random val poses (n=20) mean | {p1['nll_random_val_poses']['mean']:.3f} |")
    W(f"| random val poses (n=20) std | {p1['nll_random_val_poses']['std']:.3f} |")
    W(f"| random val poses min | {p1['nll_random_val_poses']['min']:.3f} |")
    W(f"| random val poses max | {p1['nll_random_val_poses']['max']:.3f} |")
    W("")

    W("### Phase 1 interpretation")
    W("")
    d_rng = max(p1['sweep_d']['nlls']) - min(p1['sweep_d']['nlls'])
    t_rng = max(p1['sweep_theta']['nlls']) - min(p1['sweep_theta']['nlls'])
    p_rng = max(p1['sweep_phi']['nlls']) - min(p1['sweep_phi']['nlls'])
    nll_delta = p1['nll_null_pose'] - p1['nll_reference_pose']
    W(f"- NLL swings across pose sweeps: **d={d_rng:.2f}**, **θ={t_rng:.2f}**, **φ={p_rng:.2f}** nats.")
    W(f"- Null-pose NLL is **+{nll_delta:.2f} nats** worse than reference pose.")
    W(f"- Random val-pose distribution has std={p1['nll_random_val_poses']['std']:.2f} nats.")
    W("")
    if d_rng > 3 and t_rng > 3 and nll_delta > 2:
        W("**Verdict (Phase 1)**: `[POSE]` IS used at scoring — teacher-forced "
          "log-prob varies substantially with pose input. The channel is not "
          "ignored by the decoder's gradient flow.")
    else:
        W("**Verdict (Phase 1)**: `[POSE]` produces only small NLL swings — "
          "potentially decorative at scoring level.")
    W("")

    # ================= Phase 2 =================
    W("## Phase 2 — Free-sampling distribution shift under 4 pose conditions")
    W("")
    W(f"- N/cohort = {p2['config']['N_per_cohort']}, temperature = "
      f"{p2['config']['temperature']}, pocket_idx = {p2['config']['pocket_idx']}")
    W("")
    W("### Pose conditions")
    W("")
    W("| cohort | pose_raw (d, θ, φ) |")
    W("| --- | --- |")
    for name, pc in p2["pose_conditions"].items():
        W(f"| {name} | {pc['raw']} |")
    W("")

    W("### Cohort summaries")
    W("")
    keys = ["n_valid", "pct_valid", "MW_median", "TPSA_median", "logP_median",
             "HBA_mean", "HBD_mean", "pct_acryl_largest", "n_unique_scaffolds"]
    W("| feature | " + " | ".join(p2["summaries"].keys()) + " |")
    W("| --- | " + " | ".join(["---"] * len(p2["summaries"])) + " |")
    for k in keys:
        row = f"| {k} |"
        for name, s in p2["summaries"].items():
            v = s.get(k)
            if isinstance(v, float):
                row += f" {v:.3f} |"
            else:
                row += f" {v} |"
        W(row)
    W("")

    cohort_order = list(p2["pose_conditions"].keys())
    W("### Pairwise KS-tests (continuous features)")
    W("(** = p<0.01, * = p<0.05)")
    W(format_ks_matrix(p2["ks_results"],
                        ["MW", "TPSA", "logP", "HBA", "HBD", "RB"],
                        cohort_order))
    W("")

    W("### %acryl (largest fragment) by cohort")
    W("")
    W("| cohort | pct_acryl_largest |")
    W("| --- | --- |")
    for k, v in p2["acryl_pct"].items():
        W(f"| {k} | {v:.3f} |")
    W("")

    W("### First-token distributions")
    W("")
    W("| cohort | top tokens |")
    W("| --- | --- |")
    for k, v in p2["first_token_dist"].items():
        W(f"| {k} | {v} |")
    W("")

    # Phase 2 verdict
    W("### Phase 2 interpretation")
    W("")
    def signif_pairs(feature, alpha=0.01):
        pairs = []
        for a in cohort_order:
            for b in cohort_order:
                if a >= b:
                    continue
                p = p2["ks_results"][feature][a][b]["p"]
                if p is not None and p < alpha:
                    pairs.append((a, b, p))
        return pairs

    for feat in ["MW", "TPSA", "logP"]:
        sigs = signif_pairs(feat)
        if sigs:
            W(f"- **{feat}** shows significant shifts (p<0.01) for pairs: "
              + ", ".join(f"({a} vs {b}, p={p:.2g})" for a, b, p in sigs))
        else:
            W(f"- **{feat}** shows no significant pairwise shifts.")
    W("")
    d25_vs_d50 = p2["ks_results"]["MW"]["d_2.5"]["d_5.0"]["p"]
    d25_vs_zero = p2["ks_results"]["MW"]["d_2.5"]["zeroed"]["p"]
    d25_vs_shuf = p2["ks_results"]["MW"]["d_2.5"]["shuffled"]["p"]
    W(f"- MW: d_2.5 vs d_5.0 p={d25_vs_d50}, d_2.5 vs zeroed p={d25_vs_zero}, d_2.5 vs shuffled p={d25_vs_shuf}")
    W("")

    # ================= Phase 3 =================
    W("## Phase 3 — Permutation null test (GT pose vs distant permuted pose)")
    W("")
    W(f"- GT pose (matched to reference target): {p3['gt_pose_raw']}")
    W(f"- Permuted pose (top-30% farthest in z-score): {p3['perm_pose_raw']}")
    W(f"- z-score distance = {p3['gt_perm_pose_distance_zscore']:.3f}")
    W("")
    W("### Cohort summaries")
    W("")
    keys3 = ["n_valid", "pct_valid", "MW_median", "TPSA_median",
              "logP_median", "pct_acryl_largest", "n_unique_scaffolds"]
    W("| feature | gt_pose | perm_pose |")
    W("| --- | --- | --- |")
    for k in keys3:
        gt = p3["summaries"]["gt_pose"].get(k)
        pv = p3["summaries"]["perm_pose"].get(k)
        gtstr = f"{gt:.3f}" if isinstance(gt, float) else str(gt)
        pvstr = f"{pv:.3f}" if isinstance(pv, float) else str(pv)
        W(f"| {k} | {gtstr} | {pvstr} |")
    W("")
    W("### KS-tests (GT vs PERM)")
    W("")
    W("| feature | stat | p |")
    W("| --- | --- | --- |")
    for feat in ["MW", "TPSA", "logP", "HBA", "HBD", "RB"]:
        r = p3["ks_tests"].get(feat)
        if r is None:
            W(f"| {feat} | n/a | n/a |")
        else:
            marker = "**" if r["p"] is not None and r["p"] < 0.01 else ""
            W(f"| {feat} | {r['stat']:.3f} | {r['p']:.3g}{marker} |")
    W("")
    W(f"- Scaffold Jaccard (GT ∩ PERM) / (GT ∪ PERM) = **{p3['scaffold_jaccard']:.3f}**")
    W(f"- %acryl_largest: GT = {p3['acryl_pct']['gt']:.3f}, PERM = {p3['acryl_pct']['perm']:.3f}")
    W(f"- Significant KS at p<0.01: **{p3['n_significant_ks_p01']}/{p3['n_total_ks']}**")
    W("")

    # ================= Bottom line =================
    W("## Bottom line")
    W("")
    if d_rng > 3 and t_rng > 3 and nll_delta > 2:
        pose_used_at_scoring = True
    else:
        pose_used_at_scoring = False

    # Sampling responsiveness
    sig_features_p2 = sum(1 for feat in ["MW", "TPSA", "logP"]
                          for a in cohort_order for b in cohort_order
                          if a < b and p2["ks_results"][feat][a][b]["p"] is not None
                          and p2["ks_results"][feat][a][b]["p"] < 0.01)
    # Content-sensitive if d_2.5 vs d_5.0 or d_2.5 vs shuffled differ
    content_sensitive_d = any(
        p2["ks_results"][feat]["d_2.5"]["d_5.0"]["p"] is not None and
        p2["ks_results"][feat]["d_2.5"]["d_5.0"]["p"] < 0.01
        for feat in ["MW", "TPSA", "logP"])
    presence_only = any(
        p2["ks_results"][feat]["d_2.5"]["zeroed"]["p"] is not None and
        p2["ks_results"][feat]["d_2.5"]["zeroed"]["p"] < 0.01
        for feat in ["MW", "TPSA", "logP"]) and not content_sensitive_d

    W(f"- **Phase 1 (log-prob)**: `[POSE]` "
      f"{'IS' if pose_used_at_scoring else 'IS NOT'} used at scoring.")
    W(f"- **Phase 2 (free sampling)**: significant KS shifts on {sig_features_p2}"
      f" continuous features across 4 pose conditions.")
    W(f"  - content-sensitive (d=2.5 vs d=5.0 differ): "
      f"{'YES' if content_sensitive_d else 'NO'}")
    W(f"  - presence-only (only zeroed differs): "
      f"{'YES' if presence_only else 'NO'}")
    W(f"- **Phase 3 (GT vs permuted)**: {p3['n_significant_ks_p01']}/"
      f"{p3['n_total_ks']} KS features significant at p<0.01.")
    W("")

    # Final verdict
    if not pose_used_at_scoring:
        verdict = ("`[POSE]` behaves as a **DECORATIVE TOKEN**: log-prob is "
                    "insensitive to pose values, sampling distribution does not "
                    "shift with pose content.")
    elif pose_used_at_scoring and content_sensitive_d and p3['n_significant_ks_p01'] >= 2:
        verdict = ("`[POSE]` is **BEHAVIORALLY CONTENT-SENSITIVE**: log-prob "
                    "responds to pose, and the free-sampling distribution "
                    "shifts with pose content (not just with pose presence).")
    elif pose_used_at_scoring and presence_only and p3['n_significant_ks_p01'] == 0:
        verdict = ("`[POSE]` is **SCORING-SENSITIVE BUT SAMPLING-DECORATIVE**: "
                    "the decoder recognizes the pose token at scoring time (NLL "
                    "changes with input), but this signal DOES NOT propagate to "
                    "free-sampling distributions. Sampling responds to pose "
                    "PRESENCE (zeroed pose vs any real pose) rather than to "
                    "pose CONTENT. Consistent with a nearly-degenerate encoder "
                    "output whose entropy is high enough to score inputs but too "
                    "low to move the argmax at each decode step.")
    elif pose_used_at_scoring and content_sensitive_d and p3['n_significant_ks_p01'] < 2:
        verdict = ("`[POSE]` is **CONTENT-SENSITIVE AT SCORING AND SAMPLING**, "
                    "but the specific GT-vs-permuted contrast does not reach "
                    "statistical significance on N=100 — likely need larger N "
                    "or a more sensitive metric.")
    else:
        verdict = ("`[POSE]` produces MIXED signals: log-prob is pose-sensitive, "
                    "free sampling shifts on some observables but the "
                    "GT-vs-permuted contrast is weak. Requires deeper analysis.")

    W(f"### VERDICT: {verdict}")
    W("")

    # Write report
    report_path = OUT_DIR / "pose_sweep_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {report_path}")

    # Write DONE flag
    DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    with open(DONE_FLAG, "w") as f:
        f.write("pose_sweep_diagnostic DONE\n")
        f.write(f"verdict: {verdict}\n")
        f.write(f"report: {report_path}\n")
    print(f"[report] wrote DONE flag: {DONE_FLAG}")


if __name__ == "__main__":
    main()
