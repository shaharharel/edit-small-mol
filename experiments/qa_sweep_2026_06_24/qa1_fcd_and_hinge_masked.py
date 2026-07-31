"""QA sweep #1 (2026-06-24): two follow-ups to Exp 1+

(A) Recompute FCD with an EXPANDED ChEMBL kinase covalent reference set.
    Old reference: 855 covalent_smiles.smi + ~2129 kinase_within_pairs pIC50>=7
                   = 2984 mols (after dedup).
    New reference: same union PLUS all unique CHEMBL kinase-target SMILES with
    pIC50>=7 from molecule_pIC50_minimal.csv, plus the CovInDB training
    set (1200 covalent mols). Target size 5k-15k.

(B) Recompute hinge-pharmacophore presence with the Mol1 4-aminoimidazole
    hinge atoms MASKED. A cohort molecule "has a NOVEL hinge" only if it
    contains a hinge-pharmacophore match on atoms NOT overlapping with the
    Mol1 anchor's hinge motif (when Mol1 substructure is matched).

Outputs:
    results/paper_evaluation/exp1plus_fcd_v2.json
    results/paper_evaluation/exp1plus_hinge_masked.json
    results/paper_evaluation/exp1plus_fcd_v2_summary.md
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_covft_value"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

COHORTS = {
    "base": EXP_DIR / "samples_base.csv",
    "covft": EXP_DIR / "samples_covft.csv",
    "warhead_tokens": EXP_DIR / "samples_warhead_tokens.csv",
}

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

HINGE_SMARTS = {
    "2_aminopyridine":      "Nc1ccccn1",
    "2_aminopyrimidine":    "Nc1ncccn1",
    "7_azaindole":          "c1ccc2[nH]ccc2n1",
    "pyrrolopyrimidine":    "c1ncc2[nH]ccc2n1",
    "4_aminoquinazoline":   "Nc1ncnc2ccccc12",
    "4_aminoimidazole":     "[nH0]1cnc(N)c1",  # Mol1's hinge
    "2_aminothiazole":      "Nc1nccs1",
}

# Kinase target ChEMBL IDs from kinase_within_pairs.csv (broader set).
KINASE_PAIRS = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
COV_FT_SMI = PROJECT_ROOT / "data" / "reinvent4_mol2mol_covalent_ft_work" / "covalent_smiles.smi"
COVIND_CSV = PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"
MOL_PIC50 = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "molecule_pIC50_minimal.csv"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_smiles_column(csv: Path) -> list[str]:
    df = pd.read_csv(csv)
    col = "Output_SMILES" if "Output_SMILES" in df.columns else "SMILES"
    return df[col].astype(str).tolist()


def parse_mols(smiles):
    mols, canonical = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        mols.append(m)
        canonical.append(Chem.MolToSmiles(m))
    return mols, canonical


def canonical_unique(smiles_iter):
    seen = set()
    out = []
    for s in smiles_iter:
        m = Chem.MolFromSmiles(str(s))
        if m is None or m.GetNumAtoms() == 0:
            continue
        c = Chem.MolToSmiles(m)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


# ---------- (A) FCD reference set v2 ----------

def get_kinase_targets() -> set[str]:
    """Get all CHEMBL kinase target IDs that appear in the within-pairs dataset."""
    df = pd.read_csv(KINASE_PAIRS, usecols=["target_chembl_id"])
    return set(df["target_chembl_id"].dropna().astype(str).unique())


def build_fcd_reference_v2():
    log("Building FCD reference v2 (expanded ChEMBL kinase covalent set)")
    parts = {}

    # (1) Covalent FT smiles (855)
    cov_ft = []
    if COV_FT_SMI.exists():
        with open(COV_FT_SMI) as fh:
            for line in fh:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    cov_ft.append(s)
    parts["covalent_ft_smi"] = cov_ft
    log(f"  covalent_ft_smi: {len(cov_ft)}")

    # (2) CovInDB v2 train (1200 covalent mols, all targets)
    covind = []
    if COVIND_CSV.exists():
        cd = pd.read_csv(COVIND_CSV, usecols=["smiles"])
        covind = cd["smiles"].dropna().astype(str).tolist()
    parts["covindb_v2_all"] = covind
    log(f"  covindb_v2_all: {len(covind)}")

    # (3) Kinase pIC50>=7 within-pairs (existing path)
    kinase_within = []
    if KINASE_PAIRS.exists():
        kp = pd.read_csv(KINASE_PAIRS, usecols=["mol_a", "mol_b", "value_a", "value_b"])
        ab = pd.concat([
            kp[["mol_a", "value_a"]].rename(columns={"mol_a": "smi", "value_a": "v"}),
            kp[["mol_b", "value_b"]].rename(columns={"mol_b": "smi", "value_b": "v"}),
        ]).dropna()
        ab = ab[ab["v"] >= 7.0]
        kinase_within = ab["smi"].astype(str).tolist()
    parts["kinase_within_pairs_pic50_ge7"] = kinase_within
    log(f"  kinase_within_pairs_pic50_ge7: {len(kinase_within)}")

    # (4) Broader: ALL pIC50>=7 molecules from molecule_pIC50_minimal.csv
    #     restricted to kinase target chembl IDs.
    kinase_targets = get_kinase_targets()
    log(f"  kinase_target_universe (n_targets): {len(kinase_targets)}")
    if MOL_PIC50.exists():
        chunks = []
        for i, ch in enumerate(pd.read_csv(MOL_PIC50, chunksize=200_000,
                                          usecols=["canonical_smiles", "pchembl_value",
                                                   "target_chembl_id"])):
            ch = ch.dropna(subset=["canonical_smiles", "pchembl_value", "target_chembl_id"])
            ch = ch[ch["target_chembl_id"].astype(str).isin(kinase_targets)]
            ch = ch[ch["pchembl_value"] >= 7.0]
            if len(ch):
                chunks.append(ch[["canonical_smiles"]].copy())
        kinase_broad = pd.concat(chunks)["canonical_smiles"].astype(str).tolist() if chunks else []
    else:
        kinase_broad = []
    parts["kinase_broad_pic50_ge7"] = kinase_broad
    log(f"  kinase_broad_pic50_ge7: {len(kinase_broad)}")

    # Combine + canonicalize + dedup
    combined = cov_ft + covind + kinase_within + kinase_broad
    log(f"  combined raw: {len(combined)}")
    unique = canonical_unique(combined)
    log(f"  combined unique (canonical): {len(unique)}")

    parts_sizes = {k: len(v) for k, v in parts.items()}
    return unique, parts_sizes


def compute_fcd(cohort_smiles, reference_smiles):
    try:
        from fcd_torch import FCD
    except Exception as e:
        try:
            from fcd import FCD  # noqa
        except Exception:
            log(f"FCD library not importable: {e}")
            return {"fallback": True, "error": str(e), "value": None}

    try:
        # Subsample reference if too large to keep it manageable
        rng = np.random.default_rng(42)
        if len(reference_smiles) > 15000:
            idx = rng.choice(len(reference_smiles), 15000, replace=False)
            reference_smiles = [reference_smiles[i] for i in idx]
        if len(cohort_smiles) > 15000:
            idx = rng.choice(len(cohort_smiles), 15000, replace=False)
            cohort_smiles = [cohort_smiles[i] for i in idx]
        fcd = FCD(device="cpu", n_jobs=1)
        v = float(fcd(cohort_smiles, reference_smiles))
        return {"fallback": False, "value": v,
                "n_cohort": len(cohort_smiles), "n_reference": len(reference_smiles)}
    except Exception as e:
        log(f"FCD computation failed: {e}")
        return {"fallback": True, "error": str(e), "value": None}


def run_fcd_v2(cohort_unique):
    ref, parts_sizes = build_fcd_reference_v2()
    out = {"reference_size": len(ref), "reference_composition": parts_sizes,
           "per_cohort": {}}
    for name, smis in cohort_unique.items():
        log(f"Computing FCD for {name} (n_cohort_unique={len(smis)}) vs v2 ref (n={len(ref)})")
        res = compute_fcd(smis, ref)
        out["per_cohort"][name] = res
        log(f"  {name}: FCD={res.get('value')}")
    return out


# ---------- (B) Hinge masking ----------

def hinge_masked_metrics(mols, mol1_smiles):
    """For each cohort mol:
       1. find Mol1 substructure match (if any) -> mol1_atoms set
       2. for each hinge SMARTS, find matches -> hinge_atoms sets
       3. count match as "novel" only if hinge_atoms - mol1_atoms is non-empty
          (i.e. the hinge match is NOT entirely on Mol1-anchor atoms)
       Also: if no Mol1 match found, all hinge matches count as novel.
    """
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    # Use a more specific Mol1 hinge query: just the 4-aminoimidazole region.
    mol1_hinge = Chem.MolFromSmarts("[nH0]1cnc(N)c1")
    hinge_patts = {k: Chem.MolFromSmarts(s) for k, s in HINGE_SMARTS.items()}

    cohort_results = {
        "by_class": {k: 0 for k in HINGE_SMARTS},
        "by_class_novel": {k: 0 for k in HINGE_SMARTS},
        "any_hinge": 0,
        "novel_hinge_any": 0,
        "novel_hinge_other_than_mol1class": 0,  # excludes 4-aminoimidazole entirely
        "had_mol1_match": 0,
        "n": len(mols),
    }

    for m in mols:
        # 1. find Mol1 atoms in this mol (if any)
        mol1_match = m.GetSubstructMatch(mol1)
        mol1_atoms = set(mol1_match) if mol1_match else set()
        if mol1_match:
            cohort_results["had_mol1_match"] += 1
            # If Mol1 matches, also mark the Mol1-hinge subset of those atoms
            sub_hinge = m.GetSubstructMatch(mol1_hinge)
            mol1_hinge_atoms = set(sub_hinge) if sub_hinge else set()
        else:
            mol1_hinge_atoms = set()

        any_present = False
        any_novel = False
        any_novel_non_mol1class = False
        for cls, patt in hinge_patts.items():
            all_matches = m.GetSubstructMatches(patt)
            if not all_matches:
                continue
            cohort_results["by_class"][cls] += 1
            any_present = True
            # novel: at least one match has atoms outside mol1_hinge_atoms
            novel_match = any(set(am) - mol1_hinge_atoms for am in all_matches)
            if novel_match:
                cohort_results["by_class_novel"][cls] += 1
                any_novel = True
                if cls != "4_aminoimidazole":
                    any_novel_non_mol1class = True
        if any_present:
            cohort_results["any_hinge"] += 1
        if any_novel:
            cohort_results["novel_hinge_any"] += 1
        if any_novel_non_mol1class:
            cohort_results["novel_hinge_other_than_mol1class"] += 1

    n = cohort_results["n"]
    pct = lambda c: 100 * c / n if n else 0.0
    cohort_results["pct"] = {
        "any_hinge": pct(cohort_results["any_hinge"]),
        "novel_hinge_any": pct(cohort_results["novel_hinge_any"]),
        "novel_hinge_other_than_mol1class": pct(cohort_results["novel_hinge_other_than_mol1class"]),
        "had_mol1_match": pct(cohort_results["had_mol1_match"]),
        "by_class": {k: pct(v) for k, v in cohort_results["by_class"].items()},
        "by_class_novel": {k: pct(v) for k, v in cohort_results["by_class_novel"].items()},
    }
    return cohort_results


def main():
    log("=== QA sweep #1: FCD v2 + hinge masking ===")
    cohort_unique = {}
    cohort_mols = {}
    for name, path in COHORTS.items():
        log(f"Loading {name} from {path}")
        smis = load_smiles_column(path)
        mols, canonical = parse_mols(smis)
        log(f"  {name}: {len(mols)} valid / {len(smis)} total")
        cohort_mols[name] = mols
        # dedup canonical for FCD
        seen = set()
        uniq = []
        for c in canonical:
            if c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        cohort_unique[name] = uniq
        log(f"  {name}: {len(uniq)} unique canonical")

    # (A) FCD v2
    log("\n--- (A) FCD v2 ---")
    fcd_out = run_fcd_v2(cohort_unique)
    fcd_path = OUT_DIR / "exp1plus_fcd_v2.json"
    fcd_path.write_text(json.dumps(fcd_out, indent=2))
    log(f"Wrote {fcd_path}")

    # (B) Hinge masking
    log("\n--- (B) Hinge masking ---")
    hinge_out = {}
    for name, mols in cohort_mols.items():
        log(f"Computing hinge-masked metrics for {name}")
        hinge_out[name] = hinge_masked_metrics(mols, MOL1_SMILES)
    hinge_path = OUT_DIR / "exp1plus_hinge_masked.json"
    hinge_path.write_text(json.dumps(hinge_out, indent=2))
    log(f"Wrote {hinge_path}")

    # Summary md
    lines = []
    lines.append("# Exp 1+ FCD v2 + hinge-masked (2026-06-24)\n")
    lines.append("## (A) FCD v2 — expanded ChEMBL kinase covalent reference\n")
    lines.append(f"Reference size: {fcd_out['reference_size']} unique mols")
    lines.append(f"Composition: {fcd_out['reference_composition']}\n")
    lines.append("| Cohort | FCD v2 | N cohort | N reference |")
    lines.append("|---|---|---|---|")
    for name in ["base", "covft", "warhead_tokens"]:
        r = fcd_out["per_cohort"].get(name, {})
        v = r.get("value")
        vstr = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"| {name} | {vstr} | {r.get('n_cohort')} | {r.get('n_reference')} |")
    lines.append("")
    lines.append("## (B) Hinge presence — with Mol1 anchor hinge atoms MASKED\n")
    lines.append("`novel_hinge_other_than_mol1class` = hinge present from a class OTHER than 4-aminoimidazole (Mol1's hinge)")
    lines.append("")
    lines.append("| Cohort | any hinge | novel hinge (any class) | novel hinge (non-4-aminoimid) | Mol1-anchor present |")
    lines.append("|---|---|---|---|---|")
    for name in ["base", "covft", "warhead_tokens"]:
        r = hinge_out[name]["pct"]
        lines.append(
            f"| {name} | {r['any_hinge']:.1f}% | {r['novel_hinge_any']:.1f}% | "
            f"{r['novel_hinge_other_than_mol1class']:.1f}% | {r['had_mol1_match']:.1f}% |"
        )
    lines.append("")
    lines.append("## (B) Per-class novel hinge (% with novel hinge of that class)")
    cls_list = list(HINGE_SMARTS.keys())
    header = "| Cohort | " + " | ".join(cls_list) + " |"
    sep = "|---|" + "---|" * len(cls_list)
    lines.append(header)
    lines.append(sep)
    for name in ["base", "covft", "warhead_tokens"]:
        r = hinge_out[name]["pct"]["by_class_novel"]
        row = f"| {name} | " + " | ".join(f"{r[c]:.1f}%" for c in cls_list) + " |"
        lines.append(row)
    lines.append("")

    md_path = OUT_DIR / "exp1plus_fcd_v2_summary.md"
    md_path.write_text("\n".join(lines))
    log(f"Wrote {md_path}")
    log("DONE")


if __name__ == "__main__":
    main()
