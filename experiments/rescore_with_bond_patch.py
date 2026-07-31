"""Rescore all generation method outputs with a post-hoc bond patch.

The patch finds any C-C-C(=O)-N skeleton (propanamide) near a Michael
position in the molecule and tries the C=C double bond reinstatement.
Reports both "as-output" and "after-patch" warhead detection rates.

This corrects for OpenBabel/RDKit's tendency to write single bonds when atom
distances are ambiguous (between 1.34 Å double-bond and 1.54 Å single-bond
canonicals), which collapses every method's M1 hard inpaint to 0% warhead
even when the 5-atom skeleton survives.

Usage:
  python rescore_with_bond_patch.py
  (Reads results/covalent_gen_day1/<method>/samples.sdf for all methods.)
"""
from __future__ import annotations
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results/covalent_gen_day1"

# Method dirs to process
METHODS = [
    ("m0_vanilla",         "DiffSBDD M0 vanilla"),
    ("m1_inpaint",         "DiffSBDD M1 hard inpaint"),
    ("m2_iterated",        "DiffSBDD M2 iterated inpaint"),
    ("m3_baseline",        "DiffSBDD M3 baseline finetune"),
    ("m3_jitter_small",    "DiffSBDD M3 jitter_small (sigma 0.05Å, 2.5°)"),
    ("m3_jitter_med",      "DiffSBDD M3 jitter_med (sigma 0.1Å, 5°)"),
    ("m3_jitter_large",    "DiffSBDD M3 jitter_large (sigma 0.2Å, 10°)"),
    ("drugflow_m0",        "DrugFlow M0 vanilla"),
    ("pocketflow_m0",      "PocketFlow M0 ZINC-pretrained"),
    ("pocketflow_inpaint_smoke", "PocketFlow Inpaint (warhead-seeded) ★"),
    ("drugflow_inpaint_v2",     "DrugFlow Inpaint v2 (C+D, NO_BOND fix) ★★"),
]

# Strict acrylamide
ACRYL_STRICT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
ACRYL_SOFT = Chem.MolFromSmarts("C=CC(=O)N")
# Propanamide skeleton — same 5 atoms but single C-C bond. This is what the
# bond patch tries to upgrade.
PROPANAMIDE = Chem.MolFromSmarts("[CH3]-[CH2]-C(=O)-N")  # CH3-CH2-C(=O)-N
PROPANAMIDE_LOOSE = Chem.MolFromSmarts("CCC(=O)N")
# Even more permissive (catches reduced acrylamide regardless of H counts)
ACRYL_REDUCED = Chem.MolFromSmarts("CCC(=O)N")


def patch_acrylamide(mol):
    """If a CCC(=O)N motif exists, attempt to set the first C-C to a double bond
    (acrylamide reconstruction). Returns (patched_mol, was_patched).
    Conservative: only flips bond order if the resulting molecule sanitizes.
    """
    if mol is None:
        return None, False
    # Find CCC(=O)N substructure
    matches = mol.GetSubstructMatches(ACRYL_REDUCED)
    if not matches:
        return mol, False
    # Try patching: for each match, the first two atoms are CB(beta) and CA.
    # We want CB=CA (double bond).
    for match in matches:
        c_beta, c_alpha = match[0], match[1]
        # Verify these atoms are sp3 currently (i.e., currently single bond)
        bond = mol.GetBondBetweenAtoms(c_beta, c_alpha)
        if bond is None or bond.GetBondTypeAsDouble() == 2.0:
            continue  # already double, or not bonded directly
        # Check valences: CB should have <=3 connections (so we can turn one single → double)
        # CA should have <=3 connections
        cb_atom = mol.GetAtomWithIdx(c_beta)
        ca_atom = mol.GetAtomWithIdx(c_alpha)
        if cb_atom.GetDegree() > 3 or ca_atom.GetDegree() > 3:
            continue
        # Attempt patch on a writable copy
        rw = Chem.RWMol(mol)
        rw_bond = rw.GetBondBetweenAtoms(c_beta, c_alpha)
        rw_bond.SetBondType(Chem.BondType.DOUBLE)
        # Lower implicit H count on each atom by 1 to keep valence sane
        for idx in (c_beta, c_alpha):
            a = rw.GetAtomWithIdx(idx)
            try:
                cur_h = a.GetNumExplicitHs()
                if cur_h >= 1:
                    a.SetNumExplicitHs(cur_h - 1)
            except Exception:
                pass
        patched = rw.GetMol()
        try:
            Chem.SanitizeMol(patched)
            return patched, True
        except Exception:
            continue
    return mol, False


def load_largest_frag(sdf_path: Path):
    """Same load_sdf logic as eval pipeline — largest fragment per record."""
    if not sdf_path.exists():
        return []
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    mols = []
    for m in suppl:
        if m is None:
            mols.append(None)
            continue
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            m = sorted(frags, key=lambda f: f.GetNumHeavyAtoms(), reverse=True)[0]
        try:
            Chem.SanitizeMol(m)
            mols.append(m)
        except Exception:
            mols.append(None)
    return mols


def evaluate(mols):
    n_total = len(mols)
    n_valid = sum(1 for m in mols if m is not None)
    # As-output detection
    asout_strict = sum(1 for m in mols if m is not None and m.HasSubstructMatch(ACRYL_STRICT))
    asout_soft = sum(1 for m in mols if m is not None and m.HasSubstructMatch(ACRYL_SOFT))
    skeleton = sum(1 for m in mols if m is not None and m.HasSubstructMatch(ACRYL_REDUCED))
    # After-patch detection
    patched_mols = []
    n_patched = 0
    for m in mols:
        if m is None:
            patched_mols.append(None); continue
        pm, p = patch_acrylamide(m)
        patched_mols.append(pm)
        if p: n_patched += 1
    patched_strict = sum(1 for m in patched_mols if m is not None and m.HasSubstructMatch(ACRYL_STRICT))
    patched_soft = sum(1 for m in patched_mols if m is not None and m.HasSubstructMatch(ACRYL_SOFT))
    return {
        "n_total": n_total, "n_valid": n_valid,
        "skeleton_pct": 100.0 * skeleton / max(1, n_total),
        "asout_strict_pct": 100.0 * asout_strict / max(1, n_total),
        "asout_soft_pct": 100.0 * asout_soft / max(1, n_total),
        "patched_strict_pct": 100.0 * patched_strict / max(1, n_total),
        "patched_soft_pct": 100.0 * patched_soft / max(1, n_total),
        "n_patched": n_patched,
    }


def main():
    print(f"\n{'Method':<46} {'N':>3} {'skel':>5} {'A-out':>6} {'patch':>6} {'+Δ':>5}")
    print("-" * 80)
    for dir_name, label in METHODS:
        sdf = RESULTS_ROOT / dir_name / "samples.sdf"
        mols = load_largest_frag(sdf)
        if not mols:
            print(f"{label:<46} (no SDF at {sdf})")
            continue
        r = evaluate(mols)
        delta = r["patched_soft_pct"] - r["asout_soft_pct"]
        print(f"{label:<46} {r['n_total']:>3} "
              f"{r['skeleton_pct']:>4.0f}% "
              f"{r['asout_soft_pct']:>5.1f}% "
              f"{r['patched_soft_pct']:>5.1f}% "
              f"{delta:>+4.1f}")

    print()
    print("Legend:")
    print("  skel  = skeleton (CCC(=O)N) preserved — upper bound on what bond patch could fix")
    print("  A-out = acrylamide (C=CC(=O)N) detected in raw output")
    print("  patch = acrylamide detected after post-hoc C-C → C=C bond patch")
    print("  +Δ    = improvement from bond patch")


if __name__ == "__main__":
    main()
