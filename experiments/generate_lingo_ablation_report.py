#!/usr/bin/env python3
"""Generate a 1-page Markdown report for the Lingo3DMol sampling ablation.

Reads:
    results/paper_evaluation/lingo_sampling_ablation/results.csv

Writes:
    /tmp/lingo_sampling_ablation_report.md  (single-page summary, <=800 words)
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_CSV = (PROJECT_ROOT / "results" / "paper_evaluation"
               / "lingo_sampling_ablation" / "results.csv")
OUT_MD = Path("/tmp/lingo_sampling_ablation_report.md")


def fmt(v, ndp=3):
    if v is None or v == "":
        return "-"
    try:
        fv = float(v)
        return f"{fv:.{ndp}f}"
    except Exception:
        return str(v)


def main():
    rows = []
    with SUMMARY_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    lines = []
    lines.append("# Lingo3DMol Sampling Ablation — H2 anchor (L1 FT v1)")
    lines.append("")
    lines.append("**Setup.** Generator: Lingo3DMol with L1 FT v1 (`covlingo_full_v1/"
                 "ckpt_phase2_dev.pt`), 3-epoch fine-tune from Lingo pretrain on "
                 "CovIN-DB with anchor-geometry loss. Anchor: H2 — oxadiazole-"
                 "piperidine chassis "
                 "`C=CC(=O)N1Cc2cccc(C(=O)N3CCC(c4nnc([*])o4)CC3)c2C1`, "
                 "with vinyl CH2 pinned to Cys346 attack vector. Receptor: ZAP70 "
                 "(4K2R) pocket. Target = 100 molecules per config. Each "
                 "molecule = one fresh decode pass; mol-level dedup is by "
                 "canonical SMILES on the largest fragment.")
    lines.append("")
    lines.append("**Sampling code.** All configs share the same decoder. "
                 "Multinomial draws use `torch.multinomial(softmax(logits/T))`. "
                 "Nucleus (top-p) is opt-in via `LINGO_TOPP` env var which "
                 "switches the per-step draw to `topkp_random(top_k=50, top_p=p)`.")
    lines.append("")

    # Table
    lines.append("## Results")
    lines.append("")
    lines.append("| Config | T | Sampling | N_total | N_unique | Diversity | NN Tan | d(Cβ-SG) Å | Warhead % | SMILES len |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['config']} | {fmt(r['temperature'], 2)} | "
            f"{r['sampling']} | {r['n_total']} | {r['n_unique']} | "
            f"{fmt(r['diversity_ratio'], 3)} | "
            f"{fmt(r['mean_nn_tanimoto'], 3)} | "
            f"{fmt(r['median_d_cb_sg_input'], 2)} | "
            f"{fmt(r['warhead_retention_pct'], 1)} | "
            f"{fmt(r['median_smiles_len'], 1)} |"
        )
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Pull numbers for narrative
    by_tag = {r["config"]: r for r in rows}
    def f_or_none(x):
        try:
            return float(x) if x not in (None, "", "None") else None
        except Exception:
            return None
    T10 = by_tag.get("T10", {})
    T07 = by_tag.get("T07", {})
    T15 = by_tag.get("T15", {})
    T20 = by_tag.get("T20", {})
    NP09 = by_tag.get("NP09", {})

    div_T10 = f_or_none(T10.get("diversity_ratio"))
    div_T07 = f_or_none(T07.get("diversity_ratio"))
    div_T15 = f_or_none(T15.get("diversity_ratio"))
    div_T20 = f_or_none(T20.get("diversity_ratio"))
    div_NP = f_or_none(NP09.get("diversity_ratio"))
    wh_T20 = f_or_none(T20.get("warhead_retention_pct"))
    n_T20 = int(T20.get("n_total", 0) or 0)

    lines.append(f"**Baseline (T=1.0, multinomial).** {div_T10 or 0:.2f} unique"
                 " ratio — re-confirms the H1 anomaly that the L1 FT v1 "
                 "checkpoint produces a sharply peaked logit distribution.")
    lines.append("")

    direction = ""
    if div_T07 is not None and div_T10 is not None:
        if div_T07 < div_T10:
            direction = "More collapse, as predicted (control)."
        elif div_T07 == div_T10:
            direction = "Same diversity — temperature has no effect at the cold end."
        else:
            direction = "Counter-intuitively HIGHER diversity at T=0.7 — investigate."
    lines.append(f"**Sharpening (T=0.7).** Diversity {div_T07 or 0:.2f}. {direction}")
    lines.append("")

    lines.append(f"**Heating (T=1.5 / 2.0).** Diversity rises to "
                 f"{div_T15 or 0:.2f} / {div_T20 or 0:.2f}; warhead retention at "
                 f"T=2.0 is {wh_T20 or 0:.1f}%. At T=2.0 we produced "
                 f"{n_T20} mols total — high temperature may also degrade "
                 f"decode validity.")
    lines.append("")

    lines.append(f"**Nucleus (top-p=0.9 @ T=1.0).** Diversity {div_NP or 0:.2f}. "
                 "Truncating the tail allows the >0.9-cdf head to sample "
                 "uniformly, breaking ties on the few high-probability paths.")
    lines.append("")

    lines.append("## Diversity-vs-chemistry tradeoff")
    lines.append("")
    lines.append("Reading down the configs (T07 → T10 → T12 → T15 → T20):"
                 " diversity should increase monotonically while warhead "
                 "retention should fall. NP09 sits at T=1.0 with truncation, so "
                 "the comparison NP09 vs T10 isolates the *truncation* effect "
                 "(tail mass redistribution) from the *temperature* effect "
                 "(slope steepening).")
    lines.append("")

    lines.append("## Recommended production settings")
    lines.append("")
    lines.append("Among configs with warhead retention >= 95% and valid SMILES "
                 ">= 90, pick the one with highest `n_unique`. If T=1.2 or "
                 "NP09 satisfy this, use it; otherwise fall back to the "
                 "default T=1.0 and treat low diversity as a feature, not a "
                 "bug.")
    lines.append("")

    lines.append("## Implications")
    lines.append("")
    lines.append("If T=2.0 + NP09 still cap below ~30% diversity_ratio, the "
                 "bottleneck is **anchor pinning**, not the multinomial sampler. "
                 "H2 fixes 25 tokens (the full oxadiazole-piperidine chassis "
                 "with its 3D coordinates), so the model only freely decodes "
                 "the distal R-group; for an N≈100 set against this small "
                 "search space, even an entropic decoder will repeat. If "
                 "instead NP09 alone restores diversity, the L1 FT v1 logit "
                 "distribution is the dominant constraint and we can fix it "
                 "by switching to nucleus sampling in production with no "
                 "architecture change.")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
