#!/usr/bin/env python
"""Phase A — fast standalone metrics (1-4, 10) for ZAP70 generation cohorts.

Metrics:
  1. Valid SMILES rate (%)
  2. N unique canonical SMILES
  3. N Bemis-Murcko scaffolds
  4. Acrylamide on largest fragment (%)
  10. Mean predicted pIC50 (FiLMDelta)

Phase B (slow): metrics 5-9 require AD-CovDock — separate script.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
# Headline acrylamide SMARTS (unsubstituted Michael acceptor — CH2=CH-C(=O)-N)
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")

# Cohorts where the FiLMDelta scorer is OOD (e.g. tiny fragment outputs, no rings)
OOD_COHORTS = {"Lingo_BC_ONLY_N500"}

# (name, format, path, [smiles_col], [pIC50_col], RL_flag)
COHORTS = [
    # --- OLD sequence baselines (no covalent training) ---
    ("OLD_LibInvent_NoCovalent",    "csv", PROJECT/"results/paper_evaluation/reinvent4/libinvent/libinvent_rgroup_1.csv",       "SMILES", "FiLMDelta pIC50 (raw)", False),
    ("OLD_Amine_Replacements",      "csv", PROJECT/"results/paper_evaluation/aichem_tier2_scaled/products_top50k.csv",          "smiles", "pIC50_film", False),
    # --- New covalent-aware methods ---
    ("REINVENT4_DeNovo_RL",         "csv", PROJECT/"results/paper_evaluation/reinvent4/reinvent/reinvent_denovo_1.csv",          "SMILES", "FiLMDelta pIC50 (raw)", True),
    ("REINVENT4_Mol2Mol_RL",        "csv", PROJECT/"results/paper_evaluation/reinvent4/mol2mol/mol2mol_optimize_1.csv",          "SMILES", "FiLMDelta pIC50 (raw)", True),
    ("REINVENT4_Mol2Mol_RL_late_only", "csv", PROJECT/"results/paper_evaluation/reinvent4/mol2mol_late/mol2mol_optimize_late_1.csv", "SMILES", "FiLMDelta pIC50 (raw)", True),
    ("EXP1_LibInvent_DoubleLocked", "smi", PROJECT/"data/reinvent4_libinvent_hybrid_double_locked/samples.smi",                  None, None, False),
    ("EXP2_Mol2Mol_CovInDB_FT",     "smi", PROJECT/"data/reinvent4_mol2mol_covalent_ft_samples/samples.smi",                     None, None, False),
    ("EXP3_Reactivity_RL",          "csv", PROJECT/"results/paper_evaluation/reinvent4/denovo_reactivity/denovo_reactivity_1.csv","SMILES", "FiLMDelta pIC50 (raw)", True),
    ("EXP4_Warhead_Prefix",         "smi", PROJECT/"data/reinvent4_warhead_prefix_samples/samples.smi",                          None, None, False),
    ("EXP6_Warhead_Tokens",         "smi", PROJECT/"data/reinvent4_mol2mol_warhead_tokens_samples/samples.smi",                  None, None, False),
    ("Lingo_BC_ONLY_N500",          "sdf", PROJECT/"data/cohort_eval/BC_ONLY/samples.sdf",                                       None, None, False),
    ("Lingo_NP09_N200",             "sdf", PROJECT/"data/cohort_eval/NP09/samples.sdf",                                          None, None, False),
]


def load_anchor_fps():
    """Load FiLMDelta anchor fingerprints (280 ZAP70 actives) for anchor-leak test."""
    sys.path.insert(0, str(PROJECT))
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    anchor_smiles = smiles_df["smiles"].tolist()
    anchor_pIC50 = smiles_df["pIC50"].values.astype(np.float64)
    fps = []
    for s in anchor_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return anchor_smiles, anchor_pIC50, fps


_ANCHOR_CACHE = None
def _anchors():
    global _ANCHOR_CACHE
    if _ANCHOR_CACHE is None:
        _ANCHOR_CACHE = load_anchor_fps()
    return _ANCHOR_CACHE


def load_smiles(fmt: str, path: Path, smiles_col: str | None):
    """Return list of (raw_smiles, prescored_pIC50_or_None)."""
    if fmt == "smi":
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if not tok:
                continue
            # Skip header lines: first token is column name "SMILES"/"smiles"
            if tok[0].lower() in ("smiles",):
                continue
            out.append((tok[0], None))
        return out
    if fmt == "csv":
        import csv
        out = []
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                smi = row.get(smiles_col, "").strip()
                if not smi:
                    continue
                p = row.get("FiLMDelta pIC50 (raw)", "")
                try:
                    p_val = float(p) if p else None
                    if p_val == 0.0:
                        p_val = None  # zero = filter rejection, not a real score
                except ValueError:
                    p_val = None
                out.append((smi, p_val))
        return out
    if fmt == "sdf":
        out = []
        supp = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        for mol in supp:
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    out.append((smi, None))
            except Exception:
                continue
        return out
    raise ValueError(fmt)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def score_film_pIC50(smiles: list[str]) -> list[float]:
    """Pipe SMILES into reinvent4_film_scorer.py subprocess; parse JSON."""
    if not smiles:
        return []
    scorer = PROJECT/"experiments/reinvent4_film_scorer.py"
    proc = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "quris", "python", str(scorer)],
        input="\n".join(smiles),
        capture_output=True, text=True, timeout=3600,
    )
    # scorer prints other stuff to stderr; stdout has the JSON
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{"):
            obj = json.loads(line)
            return obj["payload"]["pIC50"]
    raise RuntimeError(f"scorer failed: {proc.stderr[-400:]}")


def evaluate_cohort(cohort_spec: tuple) -> dict:
    name, fmt, path, smiles_col, pic50_col, rl = cohort_spec
    print(f"[{name}] loading {path}", flush=True)
    raw = load_smiles(fmt, path, smiles_col)
    n_raw = len(raw)

    valid_smiles = []
    canonicals = set()
    scaffolds = set()
    n_acryl = 0
    prescored = []

    for s, ps in raw:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        # canonical SMILES from RDKit
        try:
            can = Chem.MolToSmiles(mol)
        except Exception:
            continue
        valid_smiles.append(can)
        canonicals.add(can)
        # Bemis-Murcko scaffold
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            sc_smi = Chem.MolToSmiles(sc) if sc.GetNumAtoms() else ""
            if sc_smi:
                scaffolds.add(sc_smi)
        except Exception:
            pass
        # acrylamide on LARGEST fragment
        try:
            lf = largest_fragment(mol)
            Chem.SanitizeMol(lf, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
            if lf.HasSubstructMatch(ACRYL_SMARTS):
                n_acryl += 1
        except Exception:
            pass
        if ps is not None:
            prescored.append(ps)

    n_valid = len(valid_smiles)
    n_unique = len(canonicals)
    n_scaffolds = len(scaffolds)
    pct_valid = 100.0 * n_valid / n_raw if n_raw else 0.0
    pct_acryl = 100.0 * n_acryl / n_valid if n_valid else 0.0
    pct_unique = 100.0 * n_unique / n_valid if n_valid else 0.0
    pct_scaffold = 100.0 * n_scaffolds / n_unique if n_unique else 0.0
    mols_per_scaffold = round(n_unique / n_scaffolds, 2) if n_scaffolds else None

    # Intra-cohort diversity: mean nearest-neighbor Tanimoto (sample-capped)
    try:
        sys.path.insert(0, str(PROJECT))
        from src.utils.diversity import compute_intra_nn_tanimoto
        import random
        unique_list = sorted(canonicals)
        if len(unique_list) > 500:
            random.seed(42)
            sample = random.sample(unique_list, 500)
        else:
            sample = unique_list
        intra_stats = compute_intra_nn_tanimoto(sample)
        intra_nn_mean = round(intra_stats["mean_intra_nn_tanimoto"], 3)
        intra_max = round(intra_stats["max_intra_tanimoto"], 3)
    except Exception as e:
        print(f"[{name}] intra-NN Tanimoto failed: {e}", file=sys.stderr)
        intra_nn_mean = None
        intra_max = None
    # Mode-collapse heuristic: <0.7 unique-rate OR >5 mols/scaffold
    unique_rate = n_unique / n_raw if n_raw else 0.0
    mode_collapse_suspect = bool(unique_rate < 0.7 or (mols_per_scaffold and mols_per_scaffold > 5.0))

    # Anchor-leak Tanimoto test: max Tc to any of the 280 ZAP70 anchors,
    # averaged over a sample of unique canonical SMILES.
    try:
        _, _, anchor_fps = _anchors()
        afp = [fp for fp in anchor_fps if fp is not None]
        sample = sorted(canonicals)[:200]
        max_tcs = []
        n_high_leak = 0  # max Tc > 0.6 (very similar)
        for s in sample:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            tcs = DataStructs.BulkTanimotoSimilarity(fp, afp)
            mx = max(tcs) if tcs else 0.0
            max_tcs.append(mx)
            if mx > 0.6:
                n_high_leak += 1
        mean_max_tc = float(np.mean(max_tcs)) if max_tcs else None
        median_max_tc = float(np.median(max_tcs)) if max_tcs else None
        pct_high_leak = (100.0 * n_high_leak / len(max_tcs)) if max_tcs else 0.0
    except Exception as e:
        print(f"[{name}] anchor-leak test failed: {e}", file=sys.stderr)
        mean_max_tc = median_max_tc = pct_high_leak = None

    # FiLMDelta pIC50: use prescored if cohort has it; else call scorer on unique SMILES
    if prescored and len(prescored) >= 0.5 * n_valid:
        mean_pic50 = sum(prescored) / len(prescored)
        n_scored = len(prescored)
        scoring_method = "prescored_in_cohort_csv"
    elif n_unique > 0:
        unique_list = sorted(canonicals)
        # sample up to 2000 for the FiLMDelta call (large cohorts → fast)
        sample = unique_list[:2000] if len(unique_list) > 2000 else unique_list
        try:
            pic50_list = score_film_pIC50(sample)
            mean_pic50 = sum(pic50_list) / len(pic50_list) if pic50_list else float("nan")
            n_scored = len(pic50_list)
            scoring_method = f"filmdelta_scorer(n={len(sample)})"
        except Exception as e:
            print(f"[{name}] FiLMDelta scorer error: {e}", file=sys.stderr)
            mean_pic50 = float("nan")
            n_scored = 0
            scoring_method = "FAILED"
    else:
        mean_pic50 = float("nan"); n_scored = 0; scoring_method = "no_valid_mols"

    return {
        "cohort": name,
        "rl_optimized_for_filmdelta": rl,
        "ood_flag": name in OOD_COHORTS,
        "n_raw": n_raw,
        "n_valid": n_valid,
        "valid_pct": round(pct_valid, 2),
        "n_unique": n_unique,
        "unique_pct": round(pct_unique, 2),
        "n_bemis_murcko_scaffolds": n_scaffolds,
        "scaffold_pct": round(pct_scaffold, 2),
        "mols_per_scaffold": mols_per_scaffold,
        "mode_collapse_suspect": mode_collapse_suspect,
        "intra_nn_tanimoto_mean": intra_nn_mean,
        "intra_max_tanimoto": intra_max,
        "unsubstituted_acrylamide_lf_pct": round(pct_acryl, 2),
        "mean_predicted_pIC50": round(mean_pic50, 4) if mean_pic50 == mean_pic50 else None,
        "n_pIC50_scored": n_scored,
        "pIC50_scoring_method": scoring_method,
        "anchor_leak_mean_max_tc": round(mean_max_tc, 3) if mean_max_tc is not None else None,
        "anchor_leak_median_max_tc": round(median_max_tc, 3) if median_max_tc is not None else None,
        "anchor_leak_high_pct_tc_gt_06": round(pct_high_leak, 1) if pct_high_leak is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default=str(PROJECT/"results/paper_evaluation/cohort_eval/phase_a.json"))
    ap.add_argument("--workers", type=int, default=4)  # 4 to limit concurrent FiLMDelta subprocesses
    args = ap.parse_args()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[phase_a] evaluating {len(COHORTS)} cohorts with {args.workers} workers")
    with Pool(args.workers) as pool:
        results = pool.map(evaluate_cohort, COHORTS)

    summary = {
        "metrics": {
            "1": "valid_pct",
            "2": "n_unique",
            "3": "n_bemis_murcko_scaffolds",
            "4": "acrylamide_largest_frag_pct",
            "10": "mean_predicted_pIC50 (RL* = optimized toward this)",
        },
        "results": sorted(results, key=lambda r: -(r.get("mean_predicted_pIC50") or 0)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_json}\n")

    # console table
    cols = [
        "cohort", "n_raw", "valid_pct", "unique_pct", "scaffold_pct",
        "intra_nn_tanimoto_mean", "unsubstituted_acrylamide_lf_pct",
        "mean_predicted_pIC50", "anchor_leak_mean_max_tc", "anchor_leak_high_pct_tc_gt_06",
        "mode_collapse_suspect", "ood_flag", "rl_optimized_for_filmdelta",
    ]
    print("|" + "|".join(cols) + "|")
    print("|" + "|".join(["---"]*len(cols)) + "|")
    for r in summary["results"]:
        row = []
        for c in cols:
            v = r.get(c)
            if c == "rl_optimized_for_filmdelta":
                row.append("RL*" if v else "")
            elif c in ("mode_collapse_suspect", "ood_flag"):
                row.append("YES" if v else "")
            elif c == "mean_predicted_pIC50" and r.get("rl_optimized_for_filmdelta") and v is not None:
                row.append(f"{v}*")  # RL-optimized for this metric
            else:
                row.append(f"{v}" if v is not None else "n/a")
        print("|" + "|".join(row) + "|")
    print("\n* = pIC50 was the RL training reward; treat with caution when comparing.")
    print("anchor_leak: mean max Tc to 280 ZAP70 anchors. >0.6 = near-duplicate; >0.5 = high similarity to training set.")


if __name__ == "__main__":
    main()
