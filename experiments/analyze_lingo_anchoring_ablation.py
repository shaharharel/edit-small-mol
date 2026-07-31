#!/usr/bin/env python3
"""Analyze Lingo3DMol anchoring-ablation cohorts (5 configs x ~100 mols).

Hypothesis: hard-pinning all 50 prefix tokens (FULL) over-constrains the
model and forces convergence on a few completions. Loosening the pin should
restore diversity at the cost of warhead geometry fidelity.

Configs (5; SOFT_COORD skipped per implementation risk):
  FULL              — baseline (pin 50 prefix tokens + coords)
  BC_ONLY           — pin only [start_0, C_0] + β-C voxel
  BC_PLUS_4         — pin acrylamide core (atoms 0..4)
  SMI_ONLY          — pin token codes, coords float (sentinel -1)
  BC_NOCONSTRAINT   — 2-token code pin, coords float

Computes per-config metrics (matches temperature-ablation analyzer):
  - N_total, N_valid, N_unique, diversity_ratio
  - intra-cohort mean nearest-neighbour Tanimoto on Morgan FP-2048
  - median d(C-beta - SG)_input — should INCREASE as anchoring relaxes
  - warhead retention % (acrylamide on largest fragment)
  - N unique Bemis-Murcko scaffolds
  - mean / median SMILES length
  - top-3 most frequent SMILES (mode-collapse sniff test)

Inputs:
    data/lingo_anchoring_ablation/<MODE>/samples.sdf

Outputs:
    results/paper_evaluation/lingo_anchoring_ablation/results.csv
    results/paper_evaluation/lingo_anchoring_ablation/per_mol/<MODE>.csv
    results/paper_evaluation/lingo_anchoring_ablation/report.md
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
ABL_DIR = PROJECT_ROOT / "data" / "lingo_anchoring_ablation"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "lingo_anchoring_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "per_mol").mkdir(parents=True, exist_ok=True)

# ZAP70 Cys346 SG (from cohort_comparison_dock.py, receptor 4K2R after prep)
CYS346_SG = np.array([18.888, -3.650, -29.979])

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
ACRYL_PATT = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)

CONFIGS = [
    # (tag, n_tokens_pinned, coords_pinned, hypothesis_label)
    ("FULL",             50, "all",      "Baseline (50 token + coord pin)"),
    ("BC_ONLY",           2, "beta-C",   "Looser — only β-C atom"),
    ("BC_PLUS_4",        10, "5-atom",   "Acrylamide core (atoms 0..4)"),
    ("SMI_ONLY",         50, "none",     "Tokens pinned, coords float (LibInvent-like)"),
    ("BC_NOCONSTRAINT",   2, "none",     "Lightest possible (start+C only, coords float)"),
]


def canonical_smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def acrylamide_on_largest_frag(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return False
        largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return largest.HasSubstructMatch(ACRYL_PATT)
    except Exception:
        return False


def bemis_murcko_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def morgan_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def mean_nn_tanimoto(fps):
    if len(fps) < 2:
        return None
    nn = []
    for i, fp_i in enumerate(fps):
        others = fps[:i] + fps[i+1:]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, others)
        nn.append(max(sims))
    return float(np.mean(nn))


def cb_sg_distance(sdf_mol):
    """C-beta to Cys346 SG distance (input-pose, atom index 1 of the
    acrylamide SMARTS match within the 3D mol)."""
    if sdf_mol.GetNumConformers() == 0:
        return None
    try:
        match_3d = sdf_mol.GetSubstructMatch(ACRYL_PATT)
        if not match_3d or len(match_3d) < 2:
            return None
        cb_idx = match_3d[1]
        conf = sdf_mol.GetConformer(0)
        pos = conf.GetAtomPosition(cb_idx)
        cb_xyz = np.array([pos.x, pos.y, pos.z])
        return float(np.linalg.norm(cb_xyz - CYS346_SG))
    except Exception:
        return None


def analyze_cohort(tag, sdf_path):
    out = {
        "config": tag,
        "n_total": 0,
        "n_valid": 0,
        "n_unique": 0,
        "diversity_ratio": None,
        "mean_nn_tanimoto": None,
        "median_d_cb_sg_input": None,
        "warhead_retention_pct": None,
        "n_unique_scaffolds": 0,
        "mean_smiles_len": None,
        "median_smiles_len": None,
        "top3_smiles_counts": "",
    }
    per_mol_rows = []
    if not sdf_path.exists():
        return out, per_mol_rows

    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    smis = []
    smiles_lens = []
    warhead_flags = []
    cb_sg_dists = []
    n_total = 0
    for i, mol in enumerate(supp):
        n_total += 1
        if mol is None:
            per_mol_rows.append({
                "mol_idx": i, "smi": None, "smi_canon": None,
                "valid": False, "len": None, "warhead_on_largest": None,
                "d_cb_sg": None, "scaffold": None,
            })
            continue
        smi_canon = canonical_smiles(mol)
        if smi_canon is None:
            per_mol_rows.append({
                "mol_idx": i, "smi": None, "smi_canon": None,
                "valid": False, "len": None, "warhead_on_largest": None,
                "d_cb_sg": None, "scaffold": None,
            })
            continue
        smis.append(smi_canon)
        smiles_lens.append(len(smi_canon))
        wh = acrylamide_on_largest_frag(smi_canon)
        warhead_flags.append(wh)
        d = cb_sg_distance(mol)
        cb_sg_dists.append(d)
        scaf = bemis_murcko_smi(smi_canon)
        per_mol_rows.append({
            "mol_idx": i,
            "smi": smi_canon,
            "smi_canon": smi_canon,
            "valid": True,
            "len": len(smi_canon),
            "warhead_on_largest": wh,
            "d_cb_sg": d,
            "scaffold": scaf,
        })

    out["n_total"] = n_total
    out["n_valid"] = len(smis)
    unique = sorted(set(smis))
    out["n_unique"] = len(unique)
    out["diversity_ratio"] = (out["n_unique"] / out["n_total"]) if out["n_total"] else None

    if smis:
        ctr = Counter(smis)
        top3 = ctr.most_common(3)
        out["top3_smiles_counts"] = "; ".join(f"{c}x:{s[:40]}" for s, c in top3)

    fps = []
    for s in unique:
        fp = morgan_fp(s)
        if fp is not None:
            fps.append(fp)
    out["mean_nn_tanimoto"] = mean_nn_tanimoto(fps)

    if cb_sg_dists:
        ds = [d for d in cb_sg_dists if d is not None]
        if ds:
            out["median_d_cb_sg_input"] = float(np.median(ds))

    if warhead_flags:
        out["warhead_retention_pct"] = 100.0 * sum(warhead_flags) / len(warhead_flags)

    scafs = {bemis_murcko_smi(s) for s in unique}
    scafs.discard(None)
    out["n_unique_scaffolds"] = len(scafs)

    if smiles_lens:
        out["mean_smiles_len"] = float(np.mean(smiles_lens))
        out["median_smiles_len"] = float(np.median(smiles_lens))

    return out, per_mol_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl-dir", type=Path, default=ABL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rows = []
    for tag, _, _, hyp in CONFIGS:
        sdf = args.abl_dir / tag / "samples.sdf"
        cohort_out, per_mol_rows = analyze_cohort(tag, sdf)
        cohort_out["hypothesis"] = hyp
        rows.append(cohort_out)

        per_mol_path = args.out_dir / "per_mol" / f"{tag}.csv"
        with per_mol_path.open("w", newline="") as f:
            cols = ["mol_idx", "smi", "smi_canon", "valid", "len",
                    "warhead_on_largest", "d_cb_sg", "scaffold"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_mol_rows:
                w.writerow(r)
        print(f"[{tag}] wrote {per_mol_path} ({len(per_mol_rows)} rows)")

    out_csv = args.out_dir / "results.csv"
    cols = ["config", "hypothesis",
            "n_total", "n_valid", "n_unique", "diversity_ratio",
            "mean_nn_tanimoto", "median_d_cb_sg_input",
            "warhead_retention_pct", "n_unique_scaffolds",
            "mean_smiles_len", "median_smiles_len",
            "top3_smiles_counts"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\nWrote summary: {out_csv}")

    print("\nSummary table:")
    print(f"{'config':<18}{'n_tot':>6}{'n_uniq':>7}{'div':>7}"
          f"{'NN_tan':>8}{'d_cb_sg':>9}{'wh_%':>7}{'#scaf':>7}{'len_med':>9}")
    for r in rows:
        nt = r.get("n_total", 0)
        nu = r.get("n_unique", 0)
        dr = r.get("diversity_ratio")
        nn = r.get("mean_nn_tanimoto")
        cb = r.get("median_d_cb_sg_input")
        wh = r.get("warhead_retention_pct")
        ns = r.get("n_unique_scaffolds", 0)
        ml = r.get("median_smiles_len")
        print(f"{r['config']:<18}{nt:>6}{nu:>7}"
              f"{(dr if dr is not None else 0):>7.3f}"
              f"{(nn if nn is not None else 0):>8.3f}"
              f"{(cb if cb is not None else 0):>9.2f}"
              f"{(wh if wh is not None else 0):>7.1f}"
              f"{ns:>7}"
              f"{(ml if ml is not None else 0):>9.1f}")

    # Markdown report
    report_path = args.out_dir / "report.md"
    with report_path.open("w") as f:
        f.write("# Lingo3DMol anchoring ablation\n\n")
        f.write("**Date:** 2026-06-01\n")
        f.write("**Hypothesis:** hard-pinning 50 prefix tokens forces mode collapse; "
                "loosening the pin should restore diversity (with possible cost to "
                "warhead geometry fidelity).\n\n")
        f.write("**Setup:**\n")
        f.write("- Anchor: H2 (oxadiazole / acrylamide / isoindoline chassis)\n")
        f.write("- Pocket: ZAP70 Cys346 (4K2R)\n")
        f.write("- Checkpoint: `covlingo_full_v1/ckpt_phase2_dev.pt` (L1 FT v1, "
                "matches temperature ablation)\n")
        f.write("- Temperature: T=1.5 (best diversity in temp ablation per spec)\n")
        f.write("- Target N per config: 100 mols (gennums=200, min_acceptable=100)\n\n")

        f.write("## Configs run\n\n")
        f.write("| Config | n_tokens_pinned | coords_pinned | Hypothesis |\n")
        f.write("|---|---|---|---|\n")
        for tag, nt, cp, hyp in CONFIGS:
            f.write(f"| `{tag}` | {nt} | {cp} | {hyp} |\n")
        f.write("\nSOFT_COORD (Gaussian distance penalty σ=0.5 Å) was **SKIPPED** "
                "because it requires editing the model's forward pass to add a "
                "soft prior term to coord-pred logits — too risky for a one-shot "
                "ablation. Hard token pin + sentinel coords (SMI_ONLY) already "
                "spans most of that hypothesis space.\n\n")

        f.write("## Results\n\n")
        f.write("| Config | N_tot | N_uniq | div | NN_Tc | d(Cβ-SG) | warhead% | #scaf | len_med |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| `{r['config']}` "
                    f"| {r['n_total']} "
                    f"| {r['n_unique']} "
                    f"| {(r.get('diversity_ratio') or 0):.3f} "
                    f"| {(r.get('mean_nn_tanimoto') or 0):.3f} "
                    f"| {(r.get('median_d_cb_sg_input') or 0):.2f} "
                    f"| {(r.get('warhead_retention_pct') or 0):.1f} "
                    f"| {r['n_unique_scaffolds']} "
                    f"| {(r.get('median_smiles_len') or 0):.1f} |\n")
        f.write("\nLegend: `div` = N_uniq/N_tot (higher = less mode collapse); "
                "`NN_Tc` = mean intra-cohort nearest-neighbour Tanimoto (LOWER = "
                "more diverse); `d(Cβ-SG)` = input-pose Å (target 1.85 Å — FULL "
                "should be tightest, looser modes should drift); `warhead%` = "
                "fraction with acrylamide on largest fragment; `#scaf` = unique "
                "Bemis-Murcko scaffolds.\n\n")

        f.write("## Top-3 SMILES per config (mode-collapse sniff)\n\n")
        for r in rows:
            f.write(f"- **{r['config']}**: {r.get('top3_smiles_counts', '')}\n")
        f.write("\n")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
