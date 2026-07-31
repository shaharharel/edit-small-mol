"""Phase A — Cluster CovInDB 2.0 into chassis families.

A "chassis" is the central scaffold + warhead unit of a covalent inhibitor.
This script:
  1. Loads ligand SMILES from CovInDB.
  2. Identifies the warhead via SMARTS (acrylamide, chloroacetamide,
     vinyl sulfonamide, propionitrile, etc. — 10 warhead patterns).
  3. Strips the warhead, leaving the "chassis body".
  4. Computes the Bemis–Murcko scaffold of the chassis body.
  5. Groups by exact scaffold match, ranks by frequency.
  6. Builds a label table per CovInDB ligand and saves:
        data/covindb_chassis_labels.csv
        data/covindb_chassis_families.json

Top-N families are mapped to ids 0..N-1; ligands not belonging to the
top-N receive chassis_family_id = -1 (will be ignored by the chassis
head via cross_entropy(ignore_index=-1)).

CPU only. Standalone; no torch needed.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Warhead SMARTS. Order matters: more specific patterns first. After matching,
# the atoms in the SMARTS hit are deleted from the molecule (warhead stripped)
# leaving the chassis body.
# ---------------------------------------------------------------------------
# References:
#  - Acrylamide / Michael acceptor:  C=C-C(=O)-N  (common BTK/ZAP70/EGFR class)
#  - Chloroacetamide / Halohydrocarbon: Cl-CH2-C(=O)-N
#  - Vinyl sulfonamide / sulfone:     C=C-S(=O)(=O)
#  - Acrylonitrile / propionitrile:   C=C-C#N
#  - Boronic acid:                    B(O)(O)
#  - Epoxide:                         C1OC1
#  - Aldehyde:                        [CX3H1](=O)
#  - Sulfonyl fluoride:               S(=O)(=O)F
#  - Disulfide:                       S-S
#  - Beta-lactam:                     C1C(=O)NC1 (4-membered ring N–C(=O)–C–C)
#
# We match each pattern, record the warhead class, and strip the matching
# atoms. The remaining largest connected fragment is the "chassis body".
WARHEAD_SMARTS: list[tuple[str, str]] = [
    # ----- Michael acceptors (acrylamide, acrylate, vinyl ketone) -----
    ("acrylamide",         "C=CC(=O)N"),                  # amide variant
    ("acrylate",           "C=CC(=O)O"),                  # ester variant
    ("acrylyl",            "C=CC(=O)"),                   # generic Michael acceptor C=C-C=O
    ("vinyl_sulfonamide",  "C=CS(=O)(=O)N"),
    ("vinyl_sulfone",      "C=CS(=O)(=O)"),
    ("acrylonitrile",      "C=CC#N"),                     # propionitrile / α,β-unsat nitrile
    # ----- Halocarbonyl (haloacetamide / haloketone) -----
    ("chloroacetamide",    "ClCC(=O)N"),
    ("bromoacetamide",     "BrCC(=O)N"),
    ("fluoroacetamide",    "FCC(=O)N"),
    ("chloroacetyl",       "ClCC(=O)"),                   # broader Cl-CH2-C(=O)
    ("bromoacetyl",        "BrCC(=O)"),
    # ----- Special carbonyl warheads (β-lactam, lactone, urea) -----
    ("beta_lactam",        "C1CC(=O)N1"),                 # 4-membered β-lactam
    ("gamma_lactam",       "C1CCC(=O)N1"),                # 5-membered γ-lactam
    ("lactone",            "C1CCCC(=O)O1"),               # 6-membered lactone (rough)
    ("urea_carbonyl",      "NC(=O)N"),
    ("carbamate",          "OC(=O)N"),
    # ----- Aldehyde / ketoamide -----
    ("aldehyde",           "[CX3H1](=O)[#6]"),
    ("ketoamide",          "O=CC(=O)N"),                  # α-ketoamide
    # ----- Phosphonate / phosphate -----
    ("phosphonate_fluoride","P(=O)F"),
    ("phosphonate",        "P(=O)(O)O"),
    ("phosphonyl",         "P(=O)"),                      # generic P=O fallback
    # ----- Sulfonyl warheads -----
    ("sulfonyl_fluoride",  "S(=O)(=O)F"),
    ("sulfonyl_halide",    "S(=O)(=O)Cl"),
    ("sulfonamide",        "S(=O)(=O)N"),                 # sulfonamide (less reactive but used)
    ("sulfonic_acid",      "S(=O)(=O)O"),
    # ----- Boronic acid -----
    ("boronic_acid",       "B(O)O"),
    # ----- Epoxide / aziridine -----
    ("epoxide",            "C1OC1"),
    ("aziridine",          "C1CN1"),
    # ----- Disulfide / thiol -----
    ("disulfide",          "SS"),
    ("thiol",              "[SH]"),
    # ----- Nitrile fallback (least specific so last) -----
    ("nitrile",            "C#N"),
    # ----- Last-resort generic carbonyl -----
    ("carbonyl_amide",     "C(=O)N"),                     # generic amide carbonyl
    ("carbonyl",           "C(=O)"),                      # final fallback
]


def find_warhead(mol: Chem.Mol) -> tuple[str, tuple[int, ...]] | None:
    """Return (warhead_class, atom_indices) for the FIRST matching SMARTS, or None."""
    for cls, sma in WARHEAD_SMARTS:
        pat = Chem.MolFromSmarts(sma)
        if pat is None:
            continue
        m = mol.GetSubstructMatch(pat)
        if m:
            return cls, m
    return None


def strip_warhead(mol: Chem.Mol, atom_idxs: tuple[int, ...]) -> Chem.Mol | None:
    """Delete the warhead atoms; return the largest connected fragment as a Mol."""
    em = Chem.EditableMol(mol)
    # Sort descending so removal does not invalidate earlier indices.
    for a in sorted(atom_idxs, reverse=True):
        em.RemoveAtom(a)
    body = em.GetMol()
    try:
        Chem.SanitizeMol(body)
    except Exception:
        return None
    frags = Chem.GetMolFrags(body, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    # Largest connected fragment by heavy atom count.
    frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return frags[0]


def compute_chassis_scaffold(smiles: str) -> dict | None:
    """Full pipeline for ONE ligand SMILES.

    Returns dict(warhead_class, chassis_body_smi, chassis_scaffold_smi)
    or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() < 4:
        return None

    wh = find_warhead(mol)
    if wh is None:
        return None
    warhead_class, atom_idxs = wh

    body = strip_warhead(mol, atom_idxs)
    if body is None or body.GetNumHeavyAtoms() < 3:
        return None

    # Murcko scaffold — captures the ring system & their linkers.
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(body)
    except Exception:
        return None
    if scaff is None or scaff.GetNumHeavyAtoms() == 0:
        # Body is acyclic — keep body as its own "scaffold" (will collapse to
        # a small handful of acyclic chassis classes).
        scaff_smi = Chem.MolToSmiles(body)
    else:
        scaff_smi = Chem.MolToSmiles(scaff)

    if not scaff_smi:
        return None

    return {
        "warhead_class":      warhead_class,
        "chassis_body_smi":   Chem.MolToSmiles(body),
        "chassis_scaffold_smi": scaff_smi,
    }


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.complex_csv)
    print(f"[chassis] loaded {len(df)} complex records from {args.complex_csv}")
    df = df.dropna(subset=["SMILES", "PDB"])
    print(f"[chassis]   {len(df)} after dropping NA SMILES/PDB")

    # Process each row: identify chassis.
    per_row: list[dict] = []
    fail_counts: Counter = Counter()
    for i, row in df.reset_index(drop=True).iterrows():
        smi = str(row["SMILES"]).strip()
        out = compute_chassis_scaffold(smi)
        if out is None:
            fail_counts["no_warhead_or_invalid"] += 1
            per_row.append({
                "pdb_id":          str(row["PDB"]).strip().upper(),
                "ligand_smiles":   smi,
                "warhead_class":   None,
                "chassis_body_smi": None,
                "chassis_scaffold_smi": None,
            })
            continue
        per_row.append({
            "pdb_id":            str(row["PDB"]).strip().upper(),
            "ligand_smiles":     smi,
            "warhead_class":     out["warhead_class"],
            "chassis_body_smi":  out["chassis_body_smi"],
            "chassis_scaffold_smi": out["chassis_scaffold_smi"],
        })

    print(f"[chassis] warhead-found: {len(per_row) - fail_counts['no_warhead_or_invalid']}/{len(per_row)} "
          f"(failures: {dict(fail_counts)})")

    # ----- Cluster by chassis scaffold (exact match on canonical SMILES) ----
    scaffold_counter: Counter = Counter()
    scaffold_examples: dict[str, dict] = {}
    for r in per_row:
        if r["chassis_scaffold_smi"] is None:
            continue
        scf = r["chassis_scaffold_smi"]
        scaffold_counter[scf] += 1
        # Remember first example of each scaffold for the dump.
        if scf not in scaffold_examples:
            scaffold_examples[scf] = {
                "pdb_id":        r["pdb_id"],
                "ligand_smiles": r["ligand_smiles"],
                "warhead_class": r["warhead_class"],
            }

    print(f"[chassis] unique scaffolds: {len(scaffold_counter)}")
    top = scaffold_counter.most_common(args.n_classes)
    print(f"[chassis] top-{args.n_classes} scaffold frequencies:")
    for i, (scf, c) in enumerate(top):
        ex = scaffold_examples[scf]
        print(f"  [{i:2d}] count={c:4d}  scaffold={scf}  example_pdb={ex['pdb_id']}  warhead={ex['warhead_class']}")

    # Coverage check: fraction of warhead-found rows that land in top-N.
    in_top = sum(c for _, c in top)
    n_warhead = sum(scaffold_counter.values())
    print(f"[chassis] top-{args.n_classes} cover {in_top}/{n_warhead} "
          f"({100 * in_top / max(n_warhead, 1):.1f}%) of warhead-found rows")

    # If too few families, document and shrink n_classes.
    viable = sum(1 for _, c in top if c >= args.min_family_size)
    if viable < args.n_classes:
        print(f"[chassis] WARNING: only {viable} families have count >= "
              f"{args.min_family_size}. Reducing n_classes from {args.n_classes} to {viable}.")
        top = top[:viable]

    # ----- Build family table -----
    family_id_of_scaffold: dict[str, int] = {scf: i for i, (scf, _) in enumerate(top)}
    families_json = {
        "n_classes":    len(top),
        "n_classes_arg": args.n_classes,
        "min_family_size": args.min_family_size,
        "coverage_top_n_pct": 100 * in_top / max(n_warhead, 1),
        "families": [
            {
                "id":              i,
                "scaffold_smiles": scf,
                "count":           c,
                "example_pdb":     scaffold_examples[scf]["pdb_id"],
                "example_ligand":  scaffold_examples[scf]["ligand_smiles"],
                "example_warhead": scaffold_examples[scf]["warhead_class"],
            }
            for i, (scf, c) in enumerate(top)
        ],
        "warhead_smarts": WARHEAD_SMARTS,
    }
    fam_json_path = out_dir / "covindb_chassis_families.json"
    with open(fam_json_path, "w") as f:
        json.dump(families_json, f, indent=2)
    print(f"[chassis] wrote families: {fam_json_path}")

    # ----- Per-ligand label rows -----
    rows_out: list[dict] = []
    for r in per_row:
        fid = -1
        fid_smi = None
        if r["chassis_scaffold_smi"] is not None:
            fid = family_id_of_scaffold.get(r["chassis_scaffold_smi"], -1)
            if fid != -1:
                fid_smi = r["chassis_scaffold_smi"]
        rows_out.append({
            "pdb_id":               r["pdb_id"],
            "ligand_smiles":        r["ligand_smiles"],
            "chassis_family_id":    fid,
            "chassis_family_smiles": fid_smi if fid_smi is not None else "",
            "warhead_class":        r["warhead_class"] if r["warhead_class"] else "",
        })
    out_df = pd.DataFrame(rows_out)
    labels_path = out_dir / "covindb_chassis_labels.csv"
    out_df.to_csv(labels_path, index=False)
    print(f"[chassis] wrote labels: {labels_path}  rows={len(out_df)}  "
          f"in-top-N={(out_df['chassis_family_id'] >= 0).sum()}")
    print(f"[chassis] family id distribution:")
    print(out_df["chassis_family_id"].value_counts().sort_index().to_string())

    return families_json, out_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--complex_csv",
                   default=str(Path.home() / "Downloads/CovInDB2/CovInDB/Covalent_Complex_Records.csv"))
    p.add_argument("--out_dir",
                   default="data/")
    p.add_argument("--n_classes", type=int, default=15,
                   help="Number of chassis families to keep (top-N by frequency).")
    p.add_argument("--min_family_size", type=int, default=8,
                   help="Drop families smaller than this from the top-N if "
                   "fewer pass the threshold (sparse-data fallback).")
    args = p.parse_args()
    main(args)
