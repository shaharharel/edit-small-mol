"""Build the final two-group comparison table per user spec.

Group 1 — Covalent-sense metrics:
  - has_acrylamide_strict   ([CH2]=[CH]C(=O)N — H counts enforced)
  - has_acrylamide_soft     (C=CC(=O)N — same skeleton, any H counts)
  - has_skeleton            (CCC(=O)N — skeleton preserved even if C=C reduced)
  - has_any_michael         (any Michael-acceptor/electrophilic warhead)
  - acrylamide_largest_frag (the acrylamide is on the LARGEST fragment, not a stray piece)

Group 2 — Distribution shift / pretrained-model integrity:
  - validity_pct      (sanitize OK)
  - uniqueness_pct    (unique canonical SMILES / N)
  - novelty           (median max-Tc to CovBinder training set; lower = more novel)
  - MW_median         (collapse to fragments would show as low MW)
  - QED_median        (drug-likeness)
  - SAS_median        (synthetic accessibility, RDKit's `sascorer` not always available)
  - scaffold_div_pct  (unique Murcko / N_valid)
  - largest_frag_n_atoms_median (catches "molecules are dust + warhead" failure)

Methods compared:
  DiffSBDD M0-M3 family, DrugFlow M0/Inpaint-v2/Finetuned-Inpaint, PocketFlow M0/Inpaint
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results/covalent_gen_day1"
COVBINDER_CSV = PROJECT_ROOT / "data/covbinder/covind_training_set.csv"

# Methods in display order (best to worst, our hypotheses)
METHODS = [
    ("m0_vanilla",                    "DiffSBDD M0 (vanilla)"),
    ("m1_inpaint",                    "DiffSBDD M1 (hard inpaint)"),
    ("m2_iterated",                   "DiffSBDD M2 (iterated inpaint)"),
    ("m3_baseline",                   "DiffSBDD M3 (CovBinder finetune)"),
    ("m3_jitter_med",                 "DiffSBDD M3 (jitter_med)"),
    ("drugflow_m0",                   "DrugFlow M0 (vanilla, N=25)"),
    ("pocketflow_m0",                 "PocketFlow M0 (ZINC pretrained)"),
    ("pocketflow_inpaint_smoke",      "PocketFlow Inpaint (warhead-seeded) ★"),
    ("drugflow_inpaint_v2",           "DrugFlow Inpaint v2 (NO_BOND fix, N=25) ★★"),
    ("drugflow_finetuned_inpaint",    "DrugFlow C+D-Finetuned + Inpaint"),
    # N=500 scale-up (Day-3 deliverable)
    ("drugflow_m0_500",               "DrugFlow M0 vanilla N=500"),
    ("drugflow_inpaint_v2_500",       "DrugFlow Inpaint v2 N=500 ★★"),
    ("pocketflow_v2_500",             "PocketFlow Cβ-bias N=500 ★"),
    # Day-2 finetune-architecture iteration (parent-SMILES bond fix)
    ("drugflow_dc_v2_smoke",          "DrugFlow DC v2 smoke (SMILES bonds, N=25)"),
    ("drugflow_dc_v2_F",              "DrugFlow DC v2 ablation F (SMILES + freezeBond + no_jitter, N=25)"),
    ("drugflow_dc_v2_F_unfrozen",     "DrugFlow DC v2 (SMILES + unfrozen, N=25)"),
    ("drugflow_dc_v2_G",              "DrugFlow DC v2 ablation G (SMILES + freezeBond + sigma=0.03, N=25)"),
    ("drugflow_dc_winner",            "DrugFlow DC WINNER (N=100)"),
    ("drugflow_dc_winner_500",        "DrugFlow DC WINNER N=500 ★★★"),
]

# SMARTS bank
ACRYL_STRICT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
ACRYL_SOFT = Chem.MolFromSmarts("C=CC(=O)N")
PROPANAMIDE = Chem.MolFromSmarts("CCC(=O)N")
HALOACET = Chem.MolFromSmarts("[Cl,Br,I][CH2]C(=O)N")
VINYL_SULFONE = Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)")
PROPIOLAMIDE = Chem.MolFromSmarts("C#CC(=O)N")
ANY_MICHAEL_LIST = [ACRYL_SOFT, HALOACET, VINYL_SULFONE, PROPIOLAMIDE]


def morgan_fp(m, n_bits=2048, r=2):
    return AllChem.GetMorganFingerprintAsBitVect(m, r, n_bits)


def has_smarts_anywhere(m, smarts):
    """Match SMARTS on the WHOLE mol (covers all fragments) or any single fragment."""
    if m is None:
        return False
    if m.HasSubstructMatch(smarts):
        return True
    for frag in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False):
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            continue
        if frag.HasSubstructMatch(smarts):
            return True
    return False


def acrylamide_on_largest_frag(m):
    """True if the LARGEST fragment contains the acrylamide motif (drug-relevant)."""
    if m is None:
        return False
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return False
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        return False
    return largest.HasSubstructMatch(ACRYL_SOFT)


def load_train_fps():
    if not COVBINDER_CSV.exists():
        return []
    df = pd.read_csv(COVBINDER_CSV)
    col = next((c for c in df.columns if "smi" in c.lower()), None)
    if col is None:
        return []
    fps = []
    for s in df[col].dropna().astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(morgan_fp(m))
    return fps


_TRAIN_FPS = None
def max_tc_to_train(m, train_fps):
    fp = morgan_fp(m)
    return max((TanimotoSimilarity(fp, t) for t in train_fps), default=None)


def count_dust(m, threshold: int = 3) -> bool:
    """A 'dust' molecule has >= `threshold` disconnected fragments.
    Per QA agent: `% acrylamide` can be 100% while every mol is `C.C.C=CC(N)=O.N.N` —
    we need this filter to distinguish real chemistry from inpaint-scaffold-dominating.
    """
    if m is None:
        return False
    return len(Chem.GetMolFrags(m, asMols=False)) >= threshold


def evaluate_method(sdf_path: Path, train_fps: list) -> dict:
    if not sdf_path.exists():
        return {"error": "missing_sdf"}
    mols_raw = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False))
    n_total = len(mols_raw)
    rows = []
    for i, m in enumerate(mols_raw):
        if m is None:
            rows.append({"valid": False}); continue
        try:
            Chem.SanitizeMol(m)
            rec = {"valid": True}
            # QA fix #2 (2026-05-25): dust detector — ≥ 3 disconnected fragments
            # means the SDF record is mostly "loose atoms + warhead", not a real molecule.
            rec["dust"] = count_dust(m)
            # Covalent (use has_smarts_anywhere for consistency across all checks
            # — was inconsistent before: acryl_strict used whole-mol, others used per-frag)
            rec["acryl_strict"] = has_smarts_anywhere(m, ACRYL_STRICT)
            rec["acryl_soft"] = has_smarts_anywhere(m, ACRYL_SOFT)
            rec["skeleton"] = has_smarts_anywhere(m, PROPANAMIDE)
            rec["any_michael"] = any(has_smarts_anywhere(m, s) for s in ANY_MICHAEL_LIST)
            rec["acryl_largest_frag"] = acrylamide_on_largest_frag(m)
            # Distribution — compute on BOTH whole-mol AND largest-fragment so dust
            # is visible. Per QA: novelty on largest-frag of dust = Tc≈1 (warhead-only),
            # masking real "novelty". Only count non-dust mols for honest novelty.
            rec["MW_whole"] = float(Descriptors.MolWt(m))
            rec["n_heavy_whole"] = int(m.GetNumHeavyAtoms())
            frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
            if frags:
                largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
                try:
                    Chem.SanitizeMol(largest)
                except Exception:
                    largest = None
            else:
                largest = None
            if largest is not None:
                rec["MW"] = float(Descriptors.MolWt(largest))
                rec["QED"] = float(QED.qed(largest))
                rec["n_heavy"] = int(largest.GetNumHeavyAtoms())
                rec["scaffold"] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(largest))
                rec["smiles"] = Chem.MolToSmiles(largest)
                if train_fps and not rec["dust"]:
                    # Only meaningful on non-dust mols; otherwise largest-frag = warhead-only → Tc=1.0
                    rec["max_tc_train"] = max_tc_to_train(largest, train_fps)
            rows.append(rec)
        except Exception:
            rows.append({"valid": False})
    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum()) if "valid" in df.columns else 0
    valid = df[df["valid"] == True] if n_valid > 0 else df.iloc[0:0]

    def pct(col):
        if col not in valid.columns or len(valid) == 0:
            return 0.0
        return 100.0 * int(valid[col].sum()) / n_total

    def med(col):
        if col not in valid.columns or valid[col].notna().sum() == 0:
            return None
        return float(valid[col].dropna().median())

    n_unique = valid["smiles"].nunique() if "smiles" in valid.columns else 0
    n_unique_scaff = valid["scaffold"].nunique() if "scaffold" in valid.columns else 0

    return {
        "n_total": n_total,
        # Covalent metrics
        "validity_pct": 100.0 * n_valid / n_total,
        "acryl_strict_pct": pct("acryl_strict"),
        "acryl_soft_pct": pct("acryl_soft"),
        "skeleton_pct": pct("skeleton"),
        "any_michael_pct": pct("any_michael"),
        "acryl_largest_frag_pct": pct("acryl_largest_frag"),
        # QA fix #2: dust filter — % of records that are ≥3 disconnected fragments
        "dust_pct": pct("dust"),
        # Distribution (largest fragment basis — historic)
        "uniqueness_pct": 100.0 * n_unique / max(1, n_valid),
        "novelty_median_tc": med("max_tc_train"),  # only computed on non-dust
        "MW_median": med("MW"),
        "QED_median": med("QED"),
        "n_heavy_median": med("n_heavy"),
        "scaffold_div_pct": 100.0 * n_unique_scaff / max(1, n_valid),
        # Whole-mol shadow metrics — catches dust by showing total MW + n_heavy across all fragments
        "MW_whole_median": med("MW_whole"),
        "n_heavy_whole_median": med("n_heavy_whole"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix for output CSV (e.g. 'n500' → final_comparison_n500.csv).")
    args = ap.parse_args()
    train_fps = load_train_fps()
    print(f"Loaded {len(train_fps)} training FPs for novelty calc")
    print()
    rows = []
    for d, label in METHODS:
        sdf = RESULTS_ROOT / d / "samples.sdf"
        r = evaluate_method(sdf, train_fps)
        if "error" in r:
            print(f"{label:<48} MISSING SDF")
            continue
        rows.append({"method": label, **r})

    df = pd.DataFrame(rows)
    # COVALENT TABLE
    print("=" * 130)
    print("GROUP 1 — Covalent-sense metrics (does the warhead chemistry exist?)")
    print("=" * 130)
    cov_cols = ["method", "n_total", "validity_pct",
                "acryl_strict_pct", "acryl_soft_pct", "skeleton_pct",
                "acryl_largest_frag_pct", "any_michael_pct", "dust_pct"]
    sub = df[cov_cols].copy()
    sub.columns = ["Method", "N", "Valid%(of N)",
                   "Acryl-strict%(of N)", "Acryl-soft%(of N)", "Skel%(of N)",
                   "Acryl-on-largest%(of N)", "Any-Michael%(of N)", "Dust%(of N)"]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print()
    print("=" * 130)
    print("GROUP 2 — Distribution / pretrained-model integrity (is the foundation intact?)")
    print("(Dust gate: novelty Tc only computed on non-dust mols to avoid Tc=1 from warhead-only fragments)")
    print("=" * 130)
    dist_cols = ["method", "uniqueness_pct", "novelty_median_tc",
                 "MW_median", "MW_whole_median", "QED_median",
                 "n_heavy_median", "n_heavy_whole_median", "scaffold_div_pct"]
    sub = df[dist_cols].copy()
    sub.columns = ["Method", "Unique%", "Novelty(med Tc-train, non-dust)",
                   "MW(largest)", "MW(whole)", "QED",
                   "n_heavy(largest)", "n_heavy(whole)", "Scaff div%"]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

    # Save (parameterized output path per QA fix #1 — was overwriting N=25 file)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = PROJECT_ROOT / f"results/covalent_gen_day1/final_comparison{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
