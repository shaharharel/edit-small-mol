"""Compute genchem + covalent-specific metrics on 3 cohorts of 10k generated mols.

CPU-only, single-file. Inputs are REINVENT4 sampling CSVs with columns
`SMILES, SMILES_state, Input_SMILES, Tanimoto, NLL`. The `SMILES` column is
the generated/Output molecule we analyze (per spec: "Output_SMILES").

Writes:
    results/paper_evaluation/exp1_covft_metrics.json
    results/paper_evaluation/exp1_covft_metrics.png
    results/paper_evaluation/exp1_covft_metrics_summary.md
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# SA scorer (rdkit contrib)
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_covft_value"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "base": EXP_DIR / "samples_base.csv",
    "covft": EXP_DIR / "samples_covft.csv",
    "warhead_tokens": EXP_DIR / "samples_warhead_tokens.csv",
}

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
MOL1_WARHEAD_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"  # isoindoline-acrylamide body

# Order matters for plot legend.
WARHEAD_CLASSES = {
    "acrylamide": "C=CC(=O)N",
    "alpha_haloacetamide": "[Cl,Br]CC(=O)N",
    "vinyl_sulfonamide": "C=CS(=O)(=O)N",
    "epoxide": "C1OC1",
    "propargyl_amide": "C#CCC(=O)N",
    "michael_acceptor_generic": "[CX3]=[CX3][CX3]=O",
}

TC_PAIR_SAMPLES = 500
SIM_TO_MOL1_THRESHOLD = 0.4
SEED = 17


def _morgan(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def compute_cohort(name: str, csv_path: Path, mol1_fp) -> dict:
    df = pd.read_csv(csv_path)
    smiles_col = "SMILES"
    raw_smiles = df[smiles_col].astype(str).tolist()
    n_total = len(raw_smiles)

    # Parse + canonicalize.
    mols: list[Chem.Mol] = []
    canonical: list[str] = []
    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        mols.append(mol)
        canonical.append(Chem.MolToSmiles(mol))

    n_valid = len(mols)
    unique_canonical = set(canonical)
    n_unique = len(unique_canonical)

    # Physchem descriptors.
    mw, logp, tpsa, qed, sa = [], [], [], [], []
    scaffolds = set()
    fps = []
    for mol in mols:
        mw.append(Descriptors.MolWt(mol))
        logp.append(Crippen.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
        qed.append(QED.qed(mol))
        try:
            sa.append(sascorer.calculateScore(mol))
        except Exception:
            pass
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.add(scaf)
        fps.append(_morgan(mol))

    # Internal Tanimoto diversity (500 random pairs of distinct indices).
    rng = random.Random(SEED)
    pair_tcs: list[float] = []
    if n_valid >= 2:
        seen_pairs = set()
        max_attempts = TC_PAIR_SAMPLES * 10
        attempts = 0
        while len(pair_tcs) < TC_PAIR_SAMPLES and attempts < max_attempts:
            i = rng.randrange(n_valid)
            j = rng.randrange(n_valid)
            if i == j:
                attempts += 1
                continue
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                attempts += 1
                continue
            seen_pairs.add(key)
            pair_tcs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            attempts += 1

    # Similarity to Mol1.
    mol1_tcs = [DataStructs.TanimotoSimilarity(fp, mol1_fp) for fp in fps]
    pct_sim_mol1 = float(np.mean([t >= SIM_TO_MOL1_THRESHOLD for t in mol1_tcs])) * 100.0 if mol1_tcs else 0.0

    # Warhead retention (matches over VALID mols).
    warhead_patterns = {k: Chem.MolFromSmarts(s) for k, s in WARHEAD_CLASSES.items()}
    mol1_pattern = Chem.MolFromSmarts(MOL1_WARHEAD_SMARTS)
    class_hits = {k: 0 for k in WARHEAD_CLASSES}
    any_warhead_hits = 0
    mol1_body_hits = 0
    for mol in mols:
        any_match = False
        for k, patt in warhead_patterns.items():
            if patt is not None and mol.HasSubstructMatch(patt):
                class_hits[k] += 1
                any_match = True
        if any_match:
            any_warhead_hits += 1
        if mol1_pattern is not None and mol.HasSubstructMatch(mol1_pattern):
            mol1_body_hits += 1

    def pct(num: int) -> float:
        return float(num) / n_valid * 100.0 if n_valid else 0.0

    cohort = {
        "name": name,
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_pct": pct(n_valid) if False else (float(n_valid) / n_total * 100.0 if n_total else 0.0),
        "n_unique_canonical": n_unique,
        "uniqueness_pct_among_valid": (float(n_unique) / n_valid * 100.0) if n_valid else 0.0,
        "descriptors": {
            "MolWt": _summary(mw),
            "LogP": _summary(logp),
            "TPSA": _summary(tpsa),
            "QED": _summary(qed),
            "SAScore": _summary(sa),
        },
        "scaffolds": {
            "n_unique_scaffolds": len(scaffolds),
            "scaffold_uniqueness_pct": (float(len(scaffolds)) / n_valid * 100.0) if n_valid else 0.0,
        },
        "internal_tanimoto": {
            "n_pairs_sampled": len(pair_tcs),
            "mean_tc": float(np.mean(pair_tcs)) if pair_tcs else None,
            "p90_tc": float(np.percentile(pair_tcs, 90)) if pair_tcs else None,
        },
        "similarity_to_mol1": {
            "threshold": SIM_TO_MOL1_THRESHOLD,
            "pct_above_threshold": pct_sim_mol1,
            "mean_tc_to_mol1": float(np.mean(mol1_tcs)) if mol1_tcs else None,
        },
        "warheads": {
            "by_class_count": class_hits,
            "by_class_pct": {k: pct(v) for k, v in class_hits.items()},
            "any_warhead_count": any_warhead_hits,
            "any_warhead_pct": pct(any_warhead_hits),
            "mol1_warhead_body_count": mol1_body_hits,
            "mol1_warhead_body_pct": pct(mol1_body_hits),
        },
    }
    return cohort


def make_plot(results: dict, out_path: Path) -> None:
    cohort_names = list(results.keys())
    classes = list(WARHEAD_CLASSES.keys()) + ["any_warhead", "mol1_body"]

    data = np.zeros((len(classes), len(cohort_names)))
    for j, c in enumerate(cohort_names):
        wh = results[c]["warheads"]
        for i, k in enumerate(WARHEAD_CLASSES.keys()):
            data[i, j] = wh["by_class_pct"][k]
        data[-2, j] = wh["any_warhead_pct"]
        data[-1, j] = wh["mol1_warhead_body_pct"]

    x = np.arange(len(classes))
    bar_w = 0.8 / len(cohort_names)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#888888", "#1f77b4", "#d62728"]
    for j, c in enumerate(cohort_names):
        ax.bar(x + j * bar_w - 0.4 + bar_w / 2, data[:, j], width=bar_w,
               label=c, color=colors[j % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("% of valid molecules")
    ax.set_title("Warhead retention by cohort (10k samples each)")
    ax.legend(title="Cohort")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_summary_md(results: dict, out_path: Path) -> None:
    cohorts = list(results.keys())
    michael = {c: results[c]["warheads"]["by_class_pct"]["michael_acceptor_generic"] for c in cohorts}
    acryl = {c: results[c]["warheads"]["by_class_pct"]["acrylamide"] for c in cohorts}
    any_wh = {c: results[c]["warheads"]["any_warhead_pct"] for c in cohorts}
    mol1body = {c: results[c]["warheads"]["mol1_warhead_body_pct"] for c in cohorts}

    def row(c: str) -> str:
        r = results[c]
        return (
            f"| {c} | {r['n_total']} | {r['validity_pct']:.1f}% | "
            f"{r['uniqueness_pct_among_valid']:.1f}% | "
            f"{r['scaffolds']['scaffold_uniqueness_pct']:.1f}% | "
            f"{r['descriptors']['MolWt']['mean']:.0f} | "
            f"{r['descriptors']['LogP']['mean']:.2f} | "
            f"{r['descriptors']['QED']['mean']:.3f} | "
            f"{r['descriptors']['SAScore']['mean']:.2f} | "
            f"{r['internal_tanimoto']['mean_tc']:.3f} | "
            f"{r['similarity_to_mol1']['pct_above_threshold']:.1f}% |"
        )

    lines = [
        "# Exp1: CovFT value — genchem + covalent metrics",
        "",
        "## Table: cohort-level metrics (10k samples each)",
        "",
        "| Cohort | N | Validity | Uniqueness | Scaffold Uniq | MolWt | LogP | QED | SA | Int. Tc (mean) | %≥0.4 to Mol1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        row("base"),
        row("covft"),
        row("warhead_tokens"),
        "",
        "## Warhead retention (% of valid)",
        "",
        "| Cohort | Acrylamide | α-haloAcAm | Vinyl-sulf | Epoxide | Propargyl-Am | Michael(generic) | ANY warhead | Mol1 body |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cohorts:
        wh = results[c]["warheads"]["by_class_pct"]
        lines.append(
            f"| {c} | {wh['acrylamide']:.1f}% | {wh['alpha_haloacetamide']:.1f}% | "
            f"{wh['vinyl_sulfonamide']:.1f}% | {wh['epoxide']:.1f}% | "
            f"{wh['propargyl_amide']:.1f}% | {wh['michael_acceptor_generic']:.1f}% | "
            f"{results[c]['warheads']['any_warhead_pct']:.1f}% | "
            f"{results[c]['warheads']['mol1_warhead_body_pct']:.1f}% |"
        )

    headline = (
        f"**Headline (Michael acceptor presence):** base {michael['base']:.1f}% "
        f"-> covft {michael['covft']:.1f}% -> warhead_tokens {michael['warhead_tokens']:.1f}%. "
        f"**Acrylamide specifically:** base {acryl['base']:.1f}% -> covft {acryl['covft']:.1f}% "
        f"-> warhead_tokens {acryl['warhead_tokens']:.1f}%. "
        f"**ANY warhead:** base {any_wh['base']:.1f}% -> covft {any_wh['covft']:.1f}% "
        f"-> warhead_tokens {any_wh['warhead_tokens']:.1f}%. "
        f"**Mol1 isoindoline-acrylamide body:** base {mol1body['base']:.1f}% -> covft "
        f"{mol1body['covft']:.1f}% -> warhead_tokens {mol1body['warhead_tokens']:.1f}%."
    )

    # 3-sentence interpretation.
    delta_any = any_wh["covft"] - any_wh["base"]
    if any_wh["covft"] > any_wh["base"] + 5 or any_wh["warhead_tokens"] > any_wh["base"] + 5:
        interp = (
            f"Covalent fine-tuning is doing real work: ANY-warhead presence shifts from "
            f"{any_wh['base']:.1f}% in the base prior to {any_wh['covft']:.1f}% (covft, "
            f"Δ={delta_any:+.1f} pp) and {any_wh['warhead_tokens']:.1f}% (warhead_tokens). "
            f"This is the value statement: a generic Reinvent prior produces almost no "
            f"covalent chemistry, while the covalent-conditioned priors concentrate on "
            f"electrophilic motifs. The Mol1-specific isoindoline-acrylamide body "
            f"({mol1body['warhead_tokens']:.1f}% in warhead_tokens vs {mol1body['base']:.1f}% in base) "
            f"shows that warhead-class control tokens can lock the model onto the "
            f"prompt's exact warhead scaffold."
        )
    else:
        interp = (
            f"Covalent fine-tuning did NOT noticeably raise warhead presence over the base "
            f"prior (base {any_wh['base']:.1f}% vs covft {any_wh['covft']:.1f}%). The "
            f"warhead_tokens cohort at {acryl['warhead_tokens']:.1f}% acrylamide vs base "
            f"{acryl['base']:.1f}% suggests the input prompt (already covalent) is "
            f"dominating, not the fine-tune. Inspect validity, NLL, and Tanimoto-to-input "
            f"distributions: the prior may simply be echoing the input molecule."
        )

    lines += ["", "## Interpretation", "", headline, "", interp, ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    print("Reading Mol1 reference fingerprint...")
    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    assert mol1 is not None, "Mol1 reference SMILES failed to parse"
    mol1_fp = _morgan(mol1)

    results: dict[str, dict] = {}
    for name, path in COHORTS.items():
        print(f"Processing cohort '{name}' from {path.name}...")
        results[name] = compute_cohort(name, path, mol1_fp)
        wh = results[name]["warheads"]
        print(
            f"  n_valid={results[name]['n_valid']}/{results[name]['n_total']} "
            f"({results[name]['validity_pct']:.1f}%); ANY warhead {wh['any_warhead_pct']:.1f}%, "
            f"acrylamide {wh['by_class_pct']['acrylamide']:.1f}%, "
            f"Michael(gen) {wh['by_class_pct']['michael_acceptor_generic']:.1f}%"
        )

    out_json = OUT_DIR / "exp1_covft_metrics.json"
    out_png = OUT_DIR / "exp1_covft_metrics.png"
    out_md = OUT_DIR / "exp1_covft_metrics_summary.md"

    out_json.write_text(json.dumps(results, indent=2))
    make_plot(results, out_png)
    make_summary_md(results, out_md)

    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
