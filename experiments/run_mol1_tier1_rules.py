#!/usr/bin/env python3
"""
Tier 1 — Med-Chem Playbook (rule-based, deterministic).

Acrylamide WARHEAD IS PRESERVED for every output. All transformations
operate on the binder side, the linker, or the isoindoline core, never on
the warhead atoms.

Implementation: reaction SMARTS via `AllChem.ReactionFromSmarts`. Each rule is
a stateless transformation; products are deduped + warhead-checked.

Rule categories (refined per med-chem agent review, 2026-04-30):
  A. Aryl C-H scan      — Me, F, Cl, CN, CF3, OMe, OH on isoindoline aryl
  B. Aryl N-walk        — C → N at each isoindoline aryl position
  C. Imidazole C2 scan  — H/Me/Cl/NH2/cPr/CHO at imidazole C2 (NEW)
  D. Imidazole isostere — pyrazol/triazol/oxazol/thiazol swaps (FIXED)
  E. N1 sub on imidazole — Me, Et, nPr, iBu, tBu, cPr, cBu, allyl, 2-(NMe2)Et, 2-morpholino-Et (DMPK)
  F. Linker variants    — N-Me, N-cPr, +CH2 spacer, reverse amide, oxadiazol, sulfonamide, urea (FIXED + EXTENDED)
  G. Isoindoline ring   — gem-dimethyl on CH2, gem-difluoro on CH2, ring expansion → THIQ (NEW)
  H. Deuteration        — d1-iPr (methine), d7-iPr (full CYP-soft block) (NEW)
  I. N-cyclopropyl amide — already partly in F

Warhead-MODIFYING controls (alpha/beta/saturated) are in `run_mol1_tier1_5_warhead_panel.py`.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tier1_rules.py
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_film_predictor, load_zap70_train_smiles,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_tier1_rules"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "tier1_results.json"
OUT_CSV = RESULTS_DIR / "tier1_candidates.csv"


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def warhead_atom_set(mol):
    pat = Chem.MolFromSmarts("[CH2]=[CH]-C(=O)-[N;X3]")
    matches = mol.GetSubstructMatches(pat)
    if not matches:
        return set()
    atoms = set()
    for m in matches:
        atoms.update(m)
    return atoms


def _all_aryl_ch(mol):
    warhead = warhead_atom_set(mol)
    out = []
    for a in mol.GetAtoms():
        if (a.GetSymbol() == 'C' and a.GetIsAromatic()
                and a.GetTotalNumHs() > 0 and a.GetIdx() not in warhead):
            out.append(a.GetIdx())
    return out


def _add_substituent_at_idx(mol, idx, sub_smiles):
    rw = Chem.RWMol(mol)
    sub = Chem.MolFromSmiles(sub_smiles)
    if sub is None:
        return None
    offset = rw.GetNumAtoms()
    for atom in sub.GetAtoms():
        rw.AddAtom(atom)
    for bond in sub.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset, bond.GetBondType())
    rw.AddBond(idx, offset, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def _replace_atom(mol, idx, new_symbol):
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def _run_reaction(seed_smi: str, smarts: str):
    """Run a reaction-SMARTS on Mol 1 and return list of unique product SMILES."""
    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn is None:
        return []
    mol = Chem.MolFromSmiles(seed_smi)
    if mol is None:
        return []
    out = []
    seen = set()
    try:
        prods = rxn.RunReactants((mol,))
    except Exception:
        return []
    for prod_set in prods:
        if not prod_set:
            continue
        p = prod_set[0]
        try:
            Chem.SanitizeMol(p)
            s = Chem.MolToSmiles(p)
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        except Exception:
            continue
    return out


# ── Rule definitions ──────────────────────────────────────────────────────────

# A. Aryl C-H scan (legacy, working) — kept the original implementation
def rule_aryl_ch_scan(seed_smi):
    mol = Chem.MolFromSmiles(seed_smi)
    out = []
    subs = [("C", "Me"), ("F", "F"), ("Cl", "Cl"), ("OC", "OMe"),
            ("C(F)(F)F", "CF3"), ("C#N", "CN"), ("O", "OH")]
    for idx in _all_aryl_ch(mol):
        for sub_smi, name in subs:
            s = _add_substituent_at_idx(mol, idx, sub_smi)
            if s:
                out.append((s, f"Tier1_A_aryl_{name}", f"+{name} at idx {idx}"))
    return out


# B. Aryl N-walk (legacy, working)
def rule_aryl_n_walk(seed_smi):
    mol = Chem.MolFromSmiles(seed_smi)
    warhead = warhead_atom_set(mol)
    out = []
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) != 6 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        # Skip imidazole — only operate on the isoindoline carbocycle
        is_carbocycle = all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in ring)
        if not is_carbocycle:
            continue
        for idx in ring:
            if idx in warhead:
                continue
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != 'C' or atom.GetTotalNumHs() == 0:
                continue
            s = _replace_atom(mol, idx, "N")
            if s:
                out.append((s, "Tier1_B_aryl_N_walk", f"C→N at idx {idx}"))
    return out


# C. Imidazole C2-H scan (NEW, per med-chem agent recommendation)
def rule_imidazole_c2_scan(seed_smi):
    """Substitute the imidazole C2 position (between the two N atoms)."""
    out = []
    # Mol 1 imidazole: c3cn(C(C)C)cn3 — C2 is the C between the two N's (cn3-cn pattern)
    # Reaction SMARTS using atom maps. C2 in 1-substituted imidazole = ncn (middle C)
    subs_at_c2 = [
        ("[c:1]([H])([n:2][R:3])[n:4][R:5]>>[c:1](C)([n:2][R:3])[n:4][R:5]", "C2-Me"),
        # Simpler: direct SMILES replacement of fragment Nc1cn(C(C)C)cn1 → Nc1cn(C(C)C)c(R)n1
    ]
    base = Chem.MolFromSmiles("Nc1cn(C(C)C)cn1")
    if base is None or not Chem.MolFromSmiles(seed_smi).HasSubstructMatch(base):
        return out
    new_variants = [
        ("Nc1cn(C(C)C)c(C)n1", "C2-Me"),
        ("Nc1cn(C(C)C)c(Cl)n1", "C2-Cl"),
        ("Nc1cn(C(C)C)c(N)n1", "C2-NH2"),
        ("Nc1cn(C(C)C)c(C2CC2)n1", "C2-cPr"),
        ("Nc1cn(C(C)C)c(C(F)(F)F)n1", "C2-CF3"),
        ("Nc1cn(C(C)C)c(C#N)n1", "C2-CN"),
    ]
    mol = Chem.MolFromSmiles(seed_smi)
    for new_smi, name in new_variants:
        new_mol = Chem.MolFromSmiles(new_smi)
        if new_mol is None:
            continue
        try:
            prods = AllChem.ReplaceSubstructs(mol, base, new_mol, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_C_imidazoleC2_scan", name))
        except Exception:
            continue
    return out


# D. Imidazole bioisostere (FIXED — use proper SMILES patterns and ReplaceSubstructs)
def rule_imidazole_isostere(seed_smi):
    """Swap 1-iPr-imidazol-4-yl for triazole/pyrazole/oxazole/thiazole isosteres."""
    mol = Chem.MolFromSmiles(seed_smi)
    base = Chem.MolFromSmiles("Nc1cn(C(C)C)cn1")  # 4-amino-1-iPr-imidazol
    if base is None or not mol.HasSubstructMatch(base):
        return []

    isosteres = [
        ("Nc1nn(C(C)C)cn1", "1,2,4-triazol-3-amine_iPr"),
        ("Nc1cn(C(C)C)nn1", "1,2,3-triazol-4-amine_iPr"),
        ("Nc1ccn(C(C)C)n1", "pyrazol-3-amine_iPr"),
        ("Nc1cnn(C(C)C)c1", "pyrazol-5-amine_iPr"),
        ("Nc1cn(C(C)C)nc1=O", "1,2,4-oxadiazinone"),  # reaches into amide-iso territory
        ("Nc1nc(C(C)C)on1", "1,2,4-oxadiazol-3-amine"),
        ("Nc1cnc(C(C)C)s1", "thiazol-2-amine"),
        ("Nc1cnc(C(C)C)o1", "oxazol-2-amine"),
    ]
    out = []
    for new_smi, name in isosteres:
        new_mol = Chem.MolFromSmiles(new_smi)
        if new_mol is None:
            continue
        try:
            prods = AllChem.ReplaceSubstructs(mol, base, new_mol, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_D_imidazole_isostere", name))
        except Exception:
            continue
    return out


# E. N1 substituent scan on imidazole (legacy + DMPK additions)
def rule_n1_substituent_scan(seed_smi):
    mol = Chem.MolFromSmiles(seed_smi)
    base = Chem.MolFromSmiles("CC(C)n1cncc1")  # 1-iPr-imidazol stand-in
    if base is None or not mol.HasSubstructMatch(base):
        return []
    new_subs = [
        ("Cn1cncc1", "Me"),
        ("CCn1cncc1", "Et"),
        ("CCCn1cncc1", "nPr"),
        ("CC(C)Cn1cncc1", "iBu"),
        ("CC(C)(C)n1cncc1", "tBu"),
        ("C1CC1n1cncc1", "cPr"),
        ("C1CCC1n1cncc1", "cBu"),
        ("C=CCn1cncc1", "allyl"),
        # DMPK / solubility additions per med-chem agent
        ("CN(C)CCn1cncc1", "2-(NMe2)Et"),
        ("O1CCN(CCn2cncc2)CC1", "2-morpholino-Et"),
        ("OCCn1cncc1", "2-OH-Et"),
        ("C1CN(CCn2cncc2)CCC1", "2-piperidinyl-Et"),
    ]
    out = []
    for new_smi, name in new_subs:
        new_mol = Chem.MolFromSmiles(new_smi)
        if new_mol is None:
            continue
        try:
            prods = AllChem.ReplaceSubstructs(mol, base, new_mol, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_E_N1_subst", name))
        except Exception:
            continue
    return out


# F. Linker variants — uses reaction SMARTS (more robust than ReplaceSubstructs on aromatic frag)
def rule_linker_variants(seed_smi):
    rules = [
        # N-methyl amide
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1][C:2](=O)N(C)[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "N-Me amide"),
        # N-cyclopropyl amide
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1][C:2](=O)N(C9CC9)[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "N-cPr amide"),
        # +CH2 spacer between aryl and carbonyl
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]C[C:2](=O)N[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "+CH2 spacer"),
        # +CH2CH2 homologation
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]CC[C:2](=O)N[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "+CH2CH2 homologation"),
        # Reverse amide direction
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]NC(=O)[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "reverse amide"),
        # Sulfonamide
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]S(=O)(=O)N[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "sulfonamide"),
        # Urea
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]NC(=O)N[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "urea"),
        # Reductive amination (CH2-NH instead of C(=O)-NH)
        ("[c:1][C:2](=O)[N;H1][c:3]1[c:4][n:5]([C:6])[c:7][n:8]1>>"
         "[c:1]CN[c:3]1[c:4][n:5]([C:6])[c:7][n:8]1", "CH2-NH reductive amination"),
    ]
    out = []
    for smarts, name in rules:
        prods = _run_reaction(seed_smi, smarts)
        for s in prods:
            out.append((s, "Tier1_F_linker", name))
    return out


# G. Isoindoline ring modifications (NEW)
def rule_isoindoline_core(seed_smi):
    mol = Chem.MolFromSmiles(seed_smi)
    out = []

    # G1. gem-dimethyl on isoindoline saturated CH2 carbons
    base_iso = Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C)c2C1")
    new_iso_dimethyl = Chem.MolFromSmiles("C=CC(=O)N1C(C)(C)c2cccc(C)c2C1")
    if base_iso and new_iso_dimethyl and mol.HasSubstructMatch(base_iso):
        try:
            prods = AllChem.ReplaceSubstructs(mol, base_iso, new_iso_dimethyl, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_G_isoind_core", "gem-diMe at C1"))
        except Exception:
            pass
    new_iso_dimethyl_3 = Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C)c2C1(C)C")
    if base_iso and new_iso_dimethyl_3 and mol.HasSubstructMatch(base_iso):
        try:
            prods = AllChem.ReplaceSubstructs(mol, base_iso, new_iso_dimethyl_3, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_G_isoind_core", "gem-diMe at C3"))
        except Exception:
            pass

    # G2. gem-difluoro on isoindoline saturated CH2 (block benzylic oxidation)
    new_iso_difluoro = Chem.MolFromSmiles("C=CC(=O)N1C(F)(F)c2cccc(C)c2C1")
    if base_iso and new_iso_difluoro and mol.HasSubstructMatch(base_iso):
        try:
            prods = AllChem.ReplaceSubstructs(mol, base_iso, new_iso_difluoro, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_G_isoind_core", "gem-diF at C1"))
        except Exception:
            pass

    # G3. Ring expansion: isoindoline (5-ring fused) → tetrahydroisoquinoline (6-ring fused)
    base_full = Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1")
    new_thiq = Chem.MolFromSmiles("C=CC(=O)N1CCc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1")
    if base_full and new_thiq and mol.HasSubstructMatch(base_full):
        try:
            prods = AllChem.ReplaceSubstructs(mol, base_full, new_thiq, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_G_isoind_core", "ring expand → THIQ"))
        except Exception:
            pass

    return out


# H. Deuteration on the imidazole iPr CH (CYP3A4 soft spot, per agent rec)
def rule_deuteration(seed_smi):
    mol = Chem.MolFromSmiles(seed_smi)
    base = Chem.MolFromSmiles("CC([H])(C)n1cncc1")  # 1-iPr-imidazol — methine has 1 H
    out = []
    variants = [
        ("[2H][C](C)(C)n1cncc1", "d1-iPr (methine D)"),
        ("[2H]C([2H])([2H])C([2H])(C([2H])([2H])[2H])n1cncc1", "d7-iPr (full D)"),
    ]
    base2 = Chem.MolFromSmiles("CC(C)n1cncc1")
    for new_smi, name in variants:
        new_mol = Chem.MolFromSmiles(new_smi)
        if new_mol is None or base2 is None or not mol.HasSubstructMatch(base2):
            continue
        try:
            prods = AllChem.ReplaceSubstructs(mol, base2, new_mol, replaceAll=True)
            for p in prods:
                Chem.SanitizeMol(p)
                out.append((Chem.MolToSmiles(p), "Tier1_H_deuteration", name))
        except Exception:
            continue
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MOL 1 — TIER 1: MED-CHEM PLAYBOOK (warhead-preserving)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {MOL1_SMILES}")

    rules = [
        ("A: Aryl C-H scan", rule_aryl_ch_scan),
        ("B: Aryl N-walk", rule_aryl_n_walk),
        ("C: Imidazole C2 scan", rule_imidazole_c2_scan),
        ("D: Imidazole isostere", rule_imidazole_isostere),
        ("E: N1 substituent scan", rule_n1_substituent_scan),
        ("F: Linker variants", rule_linker_variants),
        ("G: Isoindoline core", rule_isoindoline_core),
        ("H: Deuteration", rule_deuteration),
    ]

    all_candidates = []
    seen = {canon(MOL1_SMILES)}
    per_rule = {}

    for label, fn in rules:
        try:
            prods = fn(MOL1_SMILES)
        except Exception as e:
            print(f"  {label:<28s} ERROR: {e}")
            per_rule[label] = 0
            continue
        kept = 0
        for smi, method, note in prods:
            try:
                cs = canon(smi)
                if cs in seen:
                    continue
                m = Chem.MolFromSmiles(cs)
                if m is None or not warhead_intact(m):
                    continue
                seen.add(cs)
                all_candidates.append({"smiles": cs, "method": method, "note": note})
                kept += 1
            except Exception:
                continue
        per_rule[label] = kept
        print(f"  {label:<28s} → {kept:3d} new candidates")

    print(f"\nTotal Tier 1 candidates (deduped, warhead-intact): {len(all_candidates)}")

    if not all_candidates:
        print("ERROR: no candidates produced.")
        return

    print("\n[+] Scoring with FiLMDelta + 3D + descriptors")
    df = pd.DataFrame(all_candidates)
    train = load_zap70_train_smiles()
    pred = load_film_predictor()
    df = score_dataframe(df, train_smiles=train, compute_3d=True, pIC50_predictor=pred)
    df = df.sort_values("pIC50", ascending=False).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    out = {
        "tier": 1, "method": "med_chem_playbook_rules", "seed": MOL1_SMILES,
        "n_candidates": len(df),
        "per_rule": per_rule,
        "timestamp": datetime.now().isoformat(),
        "candidates": df.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2))
    print(f"[+] Saved → {OUT_JSON}")

    print("\nTop 15 by FiLMDelta pIC50:")
    print(f"{'#':<3}{'pIC50':<7}{'SAS':<6}{'shape_Tc':<10}{'wrhd_dev':<10}{'method':<28s}{'note':<30s}")
    for i, r in df.head(15).iterrows():
        note = str(r.get('note',''))[:28]
        print(f"{i+1:<3}{r['pIC50']:<7.3f}{r['SAScore']:<6.2f}{r['shape_Tc_seed']:<10.3f}"
              f"{r['warhead_dev_deg']:<10.1f}{r['method']:<28s}{note:<30s}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
