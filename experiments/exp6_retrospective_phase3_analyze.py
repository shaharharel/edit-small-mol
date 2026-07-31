#!/usr/bin/env python3
"""
EXP6 Retrospective LO — Phase 3 LOCAL analysis.

For each (target, strategy, iter), reads the cohort CSV written by Phase 2 and computes:
  - max Tc(gen, drug)
  - Murcko match rate vs drug
  - Pharmacophore recovery: warhead match + hinge mimic + key substituent
  - Median FiLM+DAbs predicted pIC50
  - Internal diversity (mean pairwise Tc, sampled)
  - n with Tc >= 0.5 to drug (near-recover)
  - n with Tc == 1.0 (exact recover)

Plots trajectory of max-Tc-to-drug across iterations (per target × strategy).
Writes per-target markdown reports + the top-level summary at
  results/paper_evaluation/exp6_retrospective_summary.md.
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
DATA = PROJ / "data" / "exp6_retrospective"
RESULTS = PROJ / "results" / "paper_evaluation"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGETS = ["egfr_t790m", "btk", "kras_g12c"]
# Rival driver uses A=single anchor, B=100-anchor pool
STRATEGIES = ["A", "B"]
SCORERS = ["film", "dabs"]
ITERS = [1, 2, 3]

# Hinge / key-substituent SMARTS per target — sourced from the actual drug structure
# (these are crude pharmacophore proxies, not perfect — but better than nothing).
PHARMACOPHORE = {
    "egfr_t790m": {
        # 2-anilino-4-substituted pyrimidine — osimertinib hinge mimic
        "hinge_smarts": "Nc1nccc(c)n1",
        # dimethylaminoethyl-methylamine sidechain (osimertinib solubilizer)
        "key_sub_smarts": "N(C)CCN(C)C",
        "drug_smiles":   "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    },
    "btk": {
        # pyrazolo[3,4-d]pyrimidin-4-amine core — ibrutinib hinge
        "hinge_smarts": "Nc1ncnc2n([*])nc(-c)c12",
        "key_sub_smarts": "Oc1ccccc1",        # diaryl ether
        "drug_smiles":  "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    },
    "kras_g12c": {
        # pyridopyrimidinone core (sotorasib) — drugs may share quinazolinone variants
        "hinge_smarts": "O=c1nc2ncccc2cn1",
        "key_sub_smarts": "Oc1cccc(F)c1",       # 2-fluoro-phenol fragment
        "drug_smiles":  "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@@H](C)C1",
    },
}


def smi_to_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def tc(a, b):
    fa, fb = smi_to_fp(a), smi_to_fp(b)
    if fa is None or fb is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def murcko_match_pct(smiles_list, drug_smi):
    target_scf = ""
    md = Chem.MolFromSmiles(drug_smi)
    if md is not None:
        try:
            target_scf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(md))
        except Exception:
            pass
    if not target_scf:
        return 0.0
    n = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        except Exception:
            sc = ""
        if sc == target_scf:
            n += 1
    return 100.0 * n / max(1, len(smiles_list))


def smarts_hit_pct(smiles_list, smarts):
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return 0.0
    n = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if m.HasSubstructMatch(pat):
            n += 1
    return 100.0 * n / max(1, len(smiles_list))


def internal_diversity(smiles_list, sample_n=500):
    if len(smiles_list) < 5:
        return float("nan")
    rng = np.random.default_rng(42)
    if len(smiles_list) > sample_n:
        idx = rng.choice(len(smiles_list), sample_n, replace=False)
        smis = [smiles_list[i] for i in idx]
    else:
        smis = smiles_list
    fps = [smi_to_fp(s) for s in smis]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 5:
        return float("nan")
    tcs = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        tcs.extend(sims)
    return 1.0 - float(np.mean(tcs))


def analyze_cohort(target, strategy, scorer, it):
    csv = DATA / target / f"iter{it}_{strategy}_{scorer}_cohort.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    drug = PHARMACOPHORE[target]["drug_smiles"]
    hinge = PHARMACOPHORE[target]["hinge_smarts"]
    keysub = PHARMACOPHORE[target]["key_sub_smarts"]
    warhead = json.loads((DATA / target / "warhead_smarts.json").read_text())["smarts_generic"]

    # rival driver writes lowercase "smiles" col; compute tc on the fly
    smi_col = "smiles" if "smiles" in df.columns else ("SMILES" if "SMILES" in df.columns else df.columns[0])
    smiles_list = df[smi_col].astype(str).tolist()
    # tc_to_drug column may not exist — compute it
    if "tc_to_drug" not in df.columns:
        df["tc_to_drug"] = [tc(s, drug) for s in smiles_list]
    tc_drug = df["tc_to_drug"].max()
    tc_drug_p95 = df["tc_to_drug"].quantile(0.95)
    n_near_recover = int((df["tc_to_drug"] >= 0.5).sum())
    n_exact_recover = int((df["tc_to_drug"] >= 0.99).sum())
    murcko_pct = murcko_match_pct(smiles_list, drug)
    warhead_pct = smarts_hit_pct(smiles_list, warhead)
    hinge_pct = smarts_hit_pct(smiles_list, hinge)
    keysub_pct = smarts_hit_pct(smiles_list, keysub)
    pharm_recovery = (warhead_pct > 0) + (hinge_pct > 0) + (keysub_pct > 0)
    pic_col = "predicted_pIC50" if "predicted_pIC50" in df.columns else ("pred_pIC50" if "pred_pIC50" in df.columns else None)
    median_pic50 = float(df[pic_col].median()) if pic_col else float("nan")
    diversity = internal_diversity(smiles_list)

    return {
        "n_cohort": len(df),
        "max_tc_drug": float(tc_drug),
        "tc_drug_p95": float(tc_drug_p95),
        "n_near_recover_tc05": n_near_recover,
        "n_exact_recover": n_exact_recover,
        "murcko_match_pct": round(murcko_pct, 2),
        "warhead_pct": round(warhead_pct, 2),
        "hinge_pct": round(hinge_pct, 2),
        "keysub_pct": round(keysub_pct, 2),
        "pharmacophore_recovered_count": pharm_recovery,
        "median_pred_pic50": round(median_pic50, 3),
        "internal_diversity": round(diversity, 3),
    }


def verdict_for_target(target, rows):
    """Verdict string based on best (over strategy×iter) result."""
    best_tc = max((r.get("max_tc_drug", 0) for r in rows.values() if r), default=0.0)
    best_pharm = max((r.get("pharmacophore_recovered_count", 0) for r in rows.values() if r), default=0)
    n_near = sum((r.get("n_near_recover_tc05", 0) for r in rows.values() if r))
    n_exact = sum((r.get("n_exact_recover", 0) for r in rows.values() if r))

    if n_exact > 0:
        return f"EXACT DRUG RE-GENERATED ({n_exact} matches, max_tc={best_tc:.3f})"
    if n_near > 0:
        return f"NEAR-RECOVER ({n_near} mols Tc>=0.5 to drug, max_tc={best_tc:.3f})"
    if best_pharm >= 2 and best_tc >= 0.30:
        return f"PHARMACOPHORE RECOVER (≥2 motifs, max_tc={best_tc:.3f})"
    if best_pharm >= 1:
        return f"PARTIAL: 1 motif hit, max_tc={best_tc:.3f}"
    return f"FAIL: no recovery (max_tc={best_tc:.3f})"


def plot_trajectories(per_target):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, target in zip(axes, TARGETS):
        rows = per_target.get(target, {})
        for strategy in STRATEGIES:
            for scorer in SCORERS:
                xs, ys = [], []
                for it in ITERS:
                    r = rows.get((strategy, scorer, it))
                    if r:
                        xs.append(it)
                        ys.append(r["max_tc_drug"])
                if xs:
                    ls = "-" if scorer == "film" else "--"
                    ax.plot(xs, ys, marker="o", linestyle=ls, label=f"{strategy}/{scorer}")
        ax.set_title(target)
        ax.set_xlabel("iteration")
        ax.set_xticks(ITERS)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("max Tc(gen, drug)")
    fig.suptitle("EXP6 retrospective LO — max Tc-to-drug per iteration")
    fig.tight_layout()
    out = RESULTS / "exp6_trajectory_max_tc_drug.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    per_target = {}
    for target in TARGETS:
        per_target[target] = {}
        for strategy in STRATEGIES:
            for scorer in SCORERS:
                for it in ITERS:
                    r = analyze_cohort(target, strategy, scorer, it)
                    if r is not None:
                        per_target[target][(strategy, scorer, it)] = r

    summary_lines = ["# EXP6 retrospective LO — Phase 3 summary\n\n"]
    summary_lines.append("Honest assessment: did the pipeline regenerate the held-out drug?\n\n")
    summary_lines.append("## Per-target verdicts\n")

    for target in TARGETS:
        rows = per_target[target]
        verdict = verdict_for_target(target, rows)
        summary_lines.append(f"\n### {target}\n")
        summary_lines.append(f"**Verdict: {verdict}**\n\n")
        if not rows:
            summary_lines.append("_no cohorts produced_\n")
            continue
        summary_lines.append("| strategy | scorer | iter | n | max Tc-drug | Tc-drug p95 | n≥0.5 | n exact | Murcko% | warhead% | hinge% | keysub% | median pIC50 | div |\n")
        summary_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for strategy in STRATEGIES:
            for scorer in SCORERS:
                for it in ITERS:
                    r = rows.get((strategy, scorer, it))
                    if not r:
                        continue
                    summary_lines.append(
                        f"| {strategy} | {scorer} | {it} | {r['n_cohort']} | {r['max_tc_drug']:.3f} | {r['tc_drug_p95']:.3f} | {r['n_near_recover_tc05']} | "
                        f"{r['n_exact_recover']} | {r['murcko_match_pct']:.1f} | {r['warhead_pct']:.1f} | "
                        f"{r['hinge_pct']:.1f} | {r['keysub_pct']:.1f} | {r['median_pred_pic50']:.2f} | "
                        f"{r['internal_diversity']:.3f} |\n"
                    )

        per_md = DATA / target / "phase3_analysis.md"
        per_md.write_text(f"# {target} — Phase 3 analysis\n\nVerdict: {verdict}\n\n")

    plot_path = plot_trajectories(per_target)
    summary_lines.append(f"\n\n![max Tc-to-drug trajectory]({plot_path.name})\n")

    serializable = {t: {f"{s}/{sc}/iter{it}": v for (s, sc, it), v in rows.items()}
                    for t, rows in per_target.items()}
    (RESULTS / "exp6_retrospective_summary.json").write_text(json.dumps(serializable, indent=2))

    out_md = RESULTS / "exp6_retrospective_summary.md"
    out_md.write_text("".join(summary_lines))
    print(f"Wrote {out_md}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
