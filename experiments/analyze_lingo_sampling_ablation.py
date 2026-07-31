#!/usr/bin/env python3
"""Analyze Lingo3DMol sampling-ablation cohorts (6 configs x 100 mols).

Computes per-config metrics:
  - N_total, N_unique (canonical SMILES dedup)
  - diversity_ratio = N_unique / N_total
  - intra-cohort mean nearest-neighbour Tanimoto on Morgan FP-2048
  - median d(C-beta - SG)_input (input-pose, atom 0 is vinyl CH2; the C-beta is
    the second warhead atom — distance to Cys346 SG at (18.888, -3.650, -29.979))
  - warhead retention % (acrylamide on largest fragment)
  - mean / median SMILES length

Skips docking by default (use --dock to enable; expensive).

Inputs:
    data/lingo_sampling_ablation/<TAG>/samples.sdf

Outputs:
    results/paper_evaluation/lingo_sampling_ablation/results.csv
    results/paper_evaluation/lingo_sampling_ablation/per_mol/<TAG>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
ABL_DIR = PROJECT_ROOT / "data" / "lingo_sampling_ablation"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "lingo_sampling_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "per_mol").mkdir(parents=True, exist_ok=True)

# ZAP70 Cys346 SG (from cohort_comparison_dock.py, receptor 4K2R after prep)
CYS346_SG = np.array([18.888, -3.650, -29.979])

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
ACRYL_PATT = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)

CONFIGS = [
    # (tag, temperature, sampling, hypothesis_label)
    ("T07", 0.7, "multinomial", "Sharper — control"),
    ("T10", 1.0, "multinomial", "Default (baseline)"),
    ("T12", 1.2, "multinomial", "Slightly warmer"),
    ("T15", 1.5, "multinomial", "Warmer"),
    ("T20", 2.0, "multinomial", "Hot"),
    ("NP09", 1.0, "nucleus_p0.9", "Truncate tail"),
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


def morgan_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def mean_nn_tanimoto(fps):
    """Mean intra-cohort nearest-neighbour Tanimoto (1 - novelty proxy)."""
    if len(fps) < 2:
        return None
    nn = []
    for i, fp_i in enumerate(fps):
        others = fps[:i] + fps[i+1:]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, others)
        nn.append(max(sims))
    return float(np.mean(nn))


def cb_sg_distance(sdf_mol):
    """Find atom indices for acrylamide (CH2=CH-C(=O)-N) in 3D conf and compute
    distance from the C-beta (second atom in match) to Cys346 SG.

    Returns None if (a) no conformer, (b) no acrylamide substructure, (c) atom 0
    coordinates are zero/invalid.
    """
    if sdf_mol.GetNumConformers() == 0:
        return None
    try:
        smi = Chem.MolToSmiles(sdf_mol)
        mol2 = Chem.MolFromSmiles(smi)
        if mol2 is None:
            return None
        match = mol2.GetSubstructMatch(ACRYL_PATT)
        if not match:
            return None
        # Map SMILES atom order to SDF atom order via canonical ranks.
        # Easier: search in the 3D mol itself.
        match_3d = sdf_mol.GetSubstructMatch(ACRYL_PATT)
        if not match_3d or len(match_3d) < 2:
            return None
        cb_idx = match_3d[1]  # CH (beta)
        conf = sdf_mol.GetConformer(0)
        pos = conf.GetAtomPosition(cb_idx)
        cb_xyz = np.array([pos.x, pos.y, pos.z])
        return float(np.linalg.norm(cb_xyz - CYS346_SG))
    except Exception:
        return None


def analyze_cohort(tag, sdf_path):
    """Compute metrics for one cohort. Returns dict + per-mol rows."""
    out = {
        "config": tag,
        "n_total": 0,
        "n_valid": 0,
        "n_unique": 0,
        "diversity_ratio": None,
        "mean_nn_tanimoto": None,
        "median_d_cb_sg_input": None,
        "warhead_retention_pct": None,
        "mean_smiles_len": None,
        "median_smiles_len": None,
    }
    per_mol_rows = []
    if not sdf_path.exists():
        return out, per_mol_rows

    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    valid_mols = []
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
                "d_cb_sg": None,
            })
            continue
        smi_canon = canonical_smiles(mol)
        if smi_canon is None:
            per_mol_rows.append({
                "mol_idx": i, "smi": None, "smi_canon": None,
                "valid": False, "len": None, "warhead_on_largest": None,
                "d_cb_sg": None,
            })
            continue
        valid_mols.append(mol)
        smis.append(smi_canon)
        smiles_lens.append(len(smi_canon))
        wh = acrylamide_on_largest_frag(smi_canon)
        warhead_flags.append(wh)
        d = cb_sg_distance(mol)
        cb_sg_dists.append(d)
        per_mol_rows.append({
            "mol_idx": i,
            "smi": smi_canon,
            "smi_canon": smi_canon,
            "valid": True,
            "len": len(smi_canon),
            "warhead_on_largest": wh,
            "d_cb_sg": d,
        })

    out["n_total"] = n_total
    out["n_valid"] = len(smis)
    unique = sorted(set(smis))
    out["n_unique"] = len(unique)
    out["diversity_ratio"] = (out["n_unique"] / out["n_total"]) if out["n_total"] else None

    # Top-3 most frequent SMILES (sanity check for mode-collapse pattern)
    from collections import Counter
    if smis:
        ctr = Counter(smis)
        top3 = ctr.most_common(3)
        out["top3_smiles_counts"] = "; ".join(f"{c}x:{s[:40]}" for s, c in top3)
    else:
        out["top3_smiles_counts"] = ""

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
    for tag, T, sampling, hyp in CONFIGS:
        sdf = args.abl_dir / tag / "samples.sdf"
        cohort_out, per_mol_rows = analyze_cohort(tag, sdf)
        cohort_out["temperature"] = T
        cohort_out["sampling"] = sampling
        cohort_out["hypothesis"] = hyp
        rows.append(cohort_out)

        # Write per-mol CSV
        per_mol_path = args.out_dir / "per_mol" / f"{tag}.csv"
        with per_mol_path.open("w", newline="") as f:
            cols = ["mol_idx", "smi", "smi_canon", "valid", "len",
                    "warhead_on_largest", "d_cb_sg"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_mol_rows:
                w.writerow(r)
        print(f"[{tag}] wrote {per_mol_path} ({len(per_mol_rows)} rows)")

    # Write summary CSV
    out_csv = args.out_dir / "results.csv"
    cols = ["config", "temperature", "sampling", "hypothesis",
            "n_total", "n_valid", "n_unique", "diversity_ratio",
            "mean_nn_tanimoto", "median_d_cb_sg_input",
            "warhead_retention_pct", "mean_smiles_len", "median_smiles_len",
            "top3_smiles_counts"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\nWrote summary: {out_csv}")

    # Print a console table
    print("\nSummary table:")
    print(f"{'config':<8}{'T':>5}{'samp':>14}{'n_tot':>6}{'n_uniq':>7}{'div':>7}"
          f"{'NN_tan':>8}{'d_cb_sg':>9}{'wh_%':>7}{'len_med':>9}")
    for r in rows:
        T = r.get("temperature")
        samp = r.get("sampling", "")[:12]
        nt = r.get("n_total", 0)
        nu = r.get("n_unique", 0)
        dr = r.get("diversity_ratio")
        nn = r.get("mean_nn_tanimoto")
        cb = r.get("median_d_cb_sg_input")
        wh = r.get("warhead_retention_pct")
        ml = r.get("median_smiles_len")
        print(f"{r['config']:<8}{T:>5.2f}{samp:>14}{nt:>6}{nu:>7}"
              f"{(dr if dr is not None else 0):>7.3f}"
              f"{(nn if nn is not None else 0):>8.3f}"
              f"{(cb if cb is not None else 0):>9.2f}"
              f"{(wh if wh is not None else 0):>7.1f}"
              f"{(ml if ml is not None else 0):>9.1f}")


if __name__ == "__main__":
    main()
