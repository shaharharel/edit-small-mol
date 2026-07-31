"""Generate report.html subsection HTML for each Mol1-anchored cohort with REAL stats.

Reads scored CSVs from data/tier4_scored/ and emits a single HTML block that
replaces the placeholder #tier4-mol1-anchored section.

Writes to /tmp/mol1_subsections.html
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

# Cohort metadata: (tag, family, base_prior, seed_pool, reward_desc)
COHORTS = [
    # Phase 3: pre-THIQ reward (generic acrylamide reward only)
    ("mol1RL_v5_seed_mol1_only",
     "Phase 3 — Mol1-only RL (generic acrylamide reward)",
     "warhead_tokens",
     "Mol1 only (1 seed)",
     "FiLMDelta pIC50 0.45 + acrylamide-SMARTS 0.35 + QED 0.10 + unwanted-SMARTS 0.10",
     "First Mol1-anchored RL. Trained on Mol1 alone, reward only checks for generic acrylamide. Baseline for the THIQ-reward + Murcko-reward variants below."),

    # Phase 4: warhead_tokens + THIQ-acrylamide reward (full Mol1 pharmacophore SMARTS)
    ("thiq_rl_mol1only",
     "Phase 4 — THIQ reward, warhead_tokens base, Mol1-only seed",
     "warhead_tokens",
     "Mol1 only (1 seed)",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "Same setup as B'-1 but reward now requires the full Mol1 THIQ-acrylamide core, not just any acrylamide."),

    ("thiq_rl_zap70",
     "Phase 4 — THIQ reward, warhead_tokens base, 220 ZAP70+Mol1 seeds",
     "warhead_tokens",
     "220 ZAP70 acrylamide leads + Mol1",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "Same reward as mol1only, but RL sees a 220-seed pool of ChEMBL ZAP70 acrylamides (Mol1 included). Tests whether multi-seed RL keeps Mol1 fidelity."),

    ("thiq_rl_kinase",
     "Phase 4 — THIQ reward, warhead_tokens base, 34K kinase + ZAP70×5 + Mol1×20 seeds",
     "warhead_tokens",
     "34,314 kinase panel + ZAP70 ×5 + Mol1 ×20",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "Broad kinase pool with Mol1/ZAP70 over-sampled. Tests whether THIQ reward survives wide chemical diversity."),

    # Phase 5: covalent_ft (EXP2) + THIQ reward
    ("thiq_rl_exp2_mol1only",
     "Phase 5 — THIQ reward, covalent_ft (EXP2) base, Mol1-only seed",
     "covalent_ft (EXP2)",
     "Mol1 only (1 seed)",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "Same reward as Phase 4 but starting from covalent_ft prior (the EXP2 base). Hypothesis: covalent_ft is more Mol1-friendly than warhead_tokens."),

    ("thiq_rl_exp2_zap70",
     "Phase 5 — THIQ reward, covalent_ft (EXP2) base, 220 ZAP70+Mol1 seeds",
     "covalent_ft (EXP2)",
     "220 ZAP70 acrylamide leads + Mol1",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "covalent_ft × 220-seed ZAP70 pool."),

    ("thiq_rl_exp2_kinase",
     "Phase 5 — THIQ reward, covalent_ft (EXP2) base, 34K kinase + ZAP70×5 + Mol1×20",
     "covalent_ft (EXP2)",
     "34,314 kinase panel + ZAP70 ×5 + Mol1 ×20",
     "FiLMDelta pIC50 0.50 + THIQ-acrylamide SMARTS 0.40 + QED 0.10",
     "covalent_ft × broad kinase pool."),

    # Phase 6: Murcko-preserving (FiLM + Mol1-Murcko SMARTS + THIQ + QED)
    ("murcko_rl_zap70",
     "Phase 6 — Mol1-Murcko reward, warhead_tokens base, 220 ZAP70+Mol1 seeds",
     "warhead_tokens",
     "220 ZAP70 acrylamide leads + Mol1",
     "FiLMDelta pIC50 0.35 + Mol1-Murcko SMARTS 0.30 + THIQ-acrylamide SMARTS 0.25 + QED 0.10",
     "Adds the FULL Mol1 Murcko scaffold (THIQ + benzamide + imidazole) as a 30% reward weight. Toml SMARTS: O=C(Nc1cncn1)c1cccc2c1CNC2"),

    ("murcko_rl_kinase",
     "Phase 6 — Mol1-Murcko reward, warhead_tokens base, 34K kinase pool",
     "warhead_tokens",
     "34,314 kinase panel + ZAP70 ×5 + Mol1 ×20",
     "FiLMDelta pIC50 0.35 + Mol1-Murcko SMARTS 0.30 + THIQ-acrylamide SMARTS 0.25 + QED 0.10",
     "Murcko reward + broad kinase pool."),

    ("murcko_rl_exp2_zap70",
     "Phase 6 — Mol1-Murcko reward, covalent_ft (EXP2) base, 220 ZAP70+Mol1 seeds",
     "covalent_ft (EXP2)",
     "220 ZAP70 acrylamide leads + Mol1",
     "FiLMDelta pIC50 0.35 + Mol1-Murcko SMARTS 0.30 + THIQ-acrylamide SMARTS 0.25 + QED 0.10",
     "covalent_ft + Murcko reward."),

    ("murcko_rl_exp2_kinase",
     "Phase 6 — Mol1-Murcko reward, covalent_ft (EXP2) base, 34K kinase pool",
     "covalent_ft (EXP2)",
     "34,314 kinase panel + ZAP70 ×5 + Mol1 ×20",
     "FiLMDelta pIC50 0.35 + Mol1-Murcko SMARTS 0.30 + THIQ-acrylamide SMARTS 0.25 + QED 0.10",
     "covalent_ft + Murcko reward + broad kinase pool."),
]


def stats_for(tag: str) -> dict:
    p = PROJECT / "data/tier4_scored" / f"{tag}_scored.csv"
    if not p.exists():
        return {"missing": True}
    df = pd.read_csv(p)
    n = len(df)
    out = {"missing": False, "n": n}
    def safe(col, op="mean", pct=True):
        if col not in df.columns:
            return None
        s = df[col]
        if op == "mean":
            v = s.mean()
        elif op == "median":
            v = s.median()
        elif op == "max":
            v = s.max()
        return float(v) * (100 if pct else 1)
    out["acryl"] = safe("warhead_intact", pct=True)
    out["thiq"] = safe("thiq_core", pct=True)
    out["murcko_strict"] = safe("mol1_murcko_match", pct=True)
    out["murcko_smarts"] = safe("mol1_murcko_smarts_match", pct=True)
    out["tc_med"] = safe("Tc_to_Mol1", op="median", pct=False)
    out["tc_max"] = safe("Tc_to_Mol1", op="max", pct=False)
    out["tc_ge5"] = 100 * (df["Tc_to_Mol1"] >= 0.5).mean() if "Tc_to_Mol1" in df.columns else None
    out["qed_med"] = safe("QED", op="median", pct=False)
    out["pic50_med"] = safe("pIC50_film", op="median", pct=False)
    out["pic50_ge7"] = 100 * (df["pIC50_film"] >= 7.0).mean() if "pIC50_film" in df.columns else None
    out["max_pubtc"] = safe("max_pubTc", op="median", pct=False)
    out["closest_lead"] = df["closest_lead"].mode()[0] if "closest_lead" in df.columns and len(df) else None
    out["has_film"] = "pIC50_film" in df.columns
    out["has_pubtc"] = "max_pubTc" in df.columns
    out["has_xtb"] = "LUMO_eV" in df.columns
    return out


def fmt_pct(v, decimals=1):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}%"


def fmt_num(v, decimals=2):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def gen_subsection(tag: str, title: str, prior: str, seeds: str, reward: str, note: str) -> str:
    s = stats_for(tag)
    if s["missing"]:
        stats_html = (
            '<p style="font-size:12px;color:#a00;margin:6px 0;">'
            '⚠ Scored CSV missing — sampling not yet finished. Subsection will populate when ready.</p>'
        )
        table = (
            f'<div style="font-size:12px;color:var(--neutral-500);padding:14px;border:1px dashed #ccc;border-radius:4px;">'
            f'(no data yet for cohort <code>{tag}</code>)</div>'
        )
        loader_call = ""
    else:
        coverage = []
        if s["has_film"]: coverage.append("FiLM ✓")
        else: coverage.append('<span style="color:#a00;">FiLM ✗</span>')
        if s["has_pubtc"]: coverage.append("PubTc ✓")
        else: coverage.append('<span style="color:#a00;">PubTc ✗</span>')
        if s["has_xtb"]: coverage.append("xTB ✓")
        else: coverage.append('<span style="color:#a00;">xTB ✗</span>')

        stats_html = (
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;'
            f'margin:10px 0;font-size:12px;background:#f8f8fb;padding:10px;border-radius:6px;">'
            f'<div><b>N mols</b><br>{s["n"]:,}</div>'
            f'<div><b>Acrylamide intact</b><br>{fmt_pct(s["acryl"])}</div>'
            f'<div><b>THIQ-acryl core</b><br>{fmt_pct(s["thiq"])}</div>'
            f'<div><b>Mol1 core (SMARTS)</b><br><span style="font-weight:600;color:#5f3dc4;">{fmt_pct(s["murcko_smarts"])}</span></div>'
            f'<div><b>Mol1 Murcko (strict)</b><br>{fmt_pct(s["murcko_strict"], 2)}</div>'
            f'<div><b>Tc to Mol1 median</b><br>{fmt_num(s["tc_med"], 3)}</div>'
            f'<div><b>Tc ≥ 0.5</b><br>{fmt_pct(s["tc_ge5"])}</div>'
            f'<div><b>Tc max</b><br>{fmt_num(s["tc_max"], 3)}</div>'
            f'<div><b>QED median</b><br>{fmt_num(s["qed_med"], 3)}</div>'
            f'<div><b>FiLM pIC50 median</b><br>{fmt_num(s["pic50_med"], 2) if s["pic50_med"] else "—"}</div>'
            f'<div><b>% pIC50 ≥ 7.0</b><br>{fmt_pct(s["pic50_ge7"])}</div>'
            f'<div><b>Max pubTc median</b><br>{fmt_num(s["max_pubtc"], 3) if s["max_pubtc"] else "—"}</div>'
            f'<div><b>Closest lead</b><br>{s["closest_lead"] or "—"}</div>'
            f'<div><b>Column coverage</b><br>{" · ".join(coverage)}</div>'
            f'</div>'
        )
        table = (
            f'<div class="leader-scroll" id="tier4-{tag}-scroll" style="max-height:480px;">'
            f'  <table id="tier4-{tag}-table" class="display compact" style="width:100%;font-size:12px;">'
            f'    <thead></thead><tbody></tbody>'
            f'  </table>'
            f'</div>'
            f'<div id="tier4-{tag}-pager" style="display:flex;gap:6px;margin-top:10px;justify-content:center;font-size:11px;"></div>'
        )

    return f"""  <div style="margin-bottom:24px;border:1px solid var(--neutral-200);border-radius:8px;padding:14px;">
    <h3 style="font-size:15px;font-weight:600;margin:0 0 4px;">{title} <span style="color:#888;font-weight:400;">— <code>{tag}</code></span></h3>
    <p style="font-size:12px;color:var(--neutral-600);max-width:95ch;margin:0 0 6px;">
      <b>Prior:</b> {prior} · <b>RL seed pool:</b> {seeds}<br>
      <b>Reward:</b> {reward}<br>
      <i>{note}</i>
    </p>
    {stats_html}
    {table}
  </div>"""


def main():
    blocks = []
    for c in COHORTS:
        blocks.append(gen_subsection(*c))
    section = (
        '<section class="page" id="tier4-mol1-anchored" style="padding:32px 24px 0;">\n'
        '  <h2 style="font-size:22px;font-weight:600;margin:0 0 8px;">Mol1-Anchored RL — Lead Optimization Cohorts</h2>\n'
        '  <p style="font-size:12px;color:var(--neutral-600);max-width:95ch;margin:0 0 16px;">'
        'Comparison across <b>3 base priors × 3 reward designs × up to 3 seed pools</b> for Mol1 lead-optimization. '
        'Each subsection shows the cohort\'s real metrics computed from <code>data/tier4_scored/{tag}_scored.csv</code> '
        '(Mol1 anchor for all sampling). <b>Mol1 core (SMARTS)</b> uses the training reward\'s lenient SMARTS '
        '<code>O=C(Nc1cncn1)c1cccc2c1CNC2</code> (matches Mol1\'s pharmacophore + allows substituent variation). '
        '<b>Mol1 Murcko (strict)</b> requires exact Bemis–Murcko equality (rare — useful only as a near-duplicate check).</p>\n'
        + "\n".join(blocks)
        + '\n</section>\n'
    )
    out = Path("/tmp/mol1_subsections.html")
    out.write_text(section)
    print(f"wrote {out} ({len(section.splitlines())} lines)")
    # Also generate the JS loader-call list for DOMContentLoaded
    loader = "\n".join(
        f"      loadTier4Cohort('{c[0]}');" for c in COHORTS if not stats_for(c[0])["missing"]
    )
    Path("/tmp/mol1_loader_calls.js").write_text(loader)
    print(f"wrote loader calls for {sum(not stats_for(c[0])['missing'] for c in COHORTS)} cohorts")


if __name__ == "__main__":
    main()
