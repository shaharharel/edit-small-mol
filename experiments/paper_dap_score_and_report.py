#!/usr/bin/env python3
"""Score a REINVENT4 sample CSV with the FiLM scorer and produce a QA/report JSON.

Reports:
  - mean/median/std/min/max pIC50 over the 10k cohort
  - fraction >= 7.0 (potent)
  - top-10 mean pIC50
  - top-scaffold share (diversity check)
  - acrylamide retention fraction (soft floor)
  - QED distribution
"""
import argparse
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

REPO = Path("/home/shaharh_quris_ai/edit-small-mol")
SCORER = REPO / "experiments" / "reinvent4_film_scorer.py"
PY = "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python"
ACRYLAMIDE = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def score_with_film(smiles_list, batch=1500):
    """Call reinvent4_film_scorer.py in chunks. Returns list of floats (NaN on error)."""
    all_scores = []
    for i in range(0, len(smiles_list), batch):
        chunk = smiles_list[i : i + batch]
        proc = subprocess.run(
            [PY, str(SCORER)],
            input="\n".join(chunk),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        # Parse JSON: last line of stdout
        out = proc.stdout.strip().split("\n")[-1]
        try:
            payload = json.loads(out)
            vals = payload["payload"]["pIC50"]
        except Exception as e:
            print(
                f"[score] parse error at batch {i}: {e}\nstdout tail: {proc.stdout[-500:]}\nstderr tail: {proc.stderr[-500:]}",
                file=sys.stderr,
            )
            vals = [float("nan")] * len(chunk)
        all_scores.extend(vals)
        print(
            f"[score] {min(i+batch, len(smiles_list))}/{len(smiles_list)} — mean so far {np.nanmean(all_scores):.3f}",
            flush=True,
        )
    return all_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=10000)
    args = ap.parse_args()

    df = pd.read_csv(args.samples)
    print(f"[load] {args.samples}: {len(df)} rows, columns={list(df.columns)}")

    # REINVENT4 mol2mol sampling CSV usually has columns: SMILES,Input_SMILES,Tanimoto,NLL
    # Use SMILES column
    smi_col = None
    for cand in ["SMILES", "smiles", "Smiles"]:
        if cand in df.columns:
            smi_col = cand
            break
    if smi_col is None:
        smi_col = df.columns[0]
    print(f"[load] using column '{smi_col}'")

    # Canonicalize + dedup
    df["canon"] = df[smi_col].astype(str).map(canonicalize)
    df = df[df["canon"].notnull()].copy()
    df = df.drop_duplicates(subset=["canon"]).reset_index(drop=True)
    print(f"[dedup] {len(df)} unique valid molecules")

    # Cap at 10k
    if len(df) > args.max:
        df = df.iloc[: args.max].reset_index(drop=True)
    n = len(df)

    smiles = df["canon"].tolist()

    # Score with FiLM
    scores = score_with_film(smiles)
    df["pIC50"] = scores

    # RDKit-derived properties
    def qed_of(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return float("nan")
        try:
            return QED.qed(m)
        except Exception:
            return float("nan")

    df["qed"] = df["canon"].map(qed_of)
    df["has_acrylamide"] = df["canon"].map(
        lambda s: bool(Chem.MolFromSmiles(s).HasSubstructMatch(ACRYLAMIDE)) if Chem.MolFromSmiles(s) else False
    )
    df["scaffold"] = df["canon"].map(scaffold_of)

    # Save scored CSV alongside
    scored_csv = Path(args.samples).with_name(f"scored_{args.tag}.csv")
    df.to_csv(scored_csv, index=False)

    pic50 = np.asarray(df["pIC50"].values, dtype=float)
    valid = ~np.isnan(pic50)
    p = pic50[valid]
    top10 = np.sort(p)[-10:]

    scaf_counts = df["scaffold"].value_counts(dropna=True)
    top_scaf_share = float(scaf_counts.iloc[0] / len(df)) if len(scaf_counts) else 0.0

    report = {
        "tag": args.tag,
        "n_cohort": int(n),
        "n_valid_scored": int(valid.sum()),
        "pIC50_mean": float(np.mean(p)),
        "pIC50_median": float(np.median(p)),
        "pIC50_std": float(np.std(p)),
        "pIC50_min": float(np.min(p)),
        "pIC50_max": float(np.max(p)),
        "frac_ge_7": float(np.mean(p >= 7.0)),
        "frac_ge_6_5": float(np.mean(p >= 6.5)),
        "top10_mean_pIC50": float(np.mean(top10)),
        "acrylamide_retention_frac": float(df["has_acrylamide"].mean()),
        "qed_mean": float(np.nanmean(df["qed"].values)),
        "top_scaffold_share": top_scaf_share,
        "n_unique_scaffolds": int(scaf_counts.shape[0]),
        "scored_csv": str(scored_csv),
    }

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
