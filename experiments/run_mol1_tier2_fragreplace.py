#!/usr/bin/env python3
"""
Tier 2 — Fragment-Replacement (synthesis-aware).

Holds Fragment A (acrylamide-isoindoline carboxylic acid) FIXED:
    C=CC(=O)N1Cc2cccc(C(=O)O)c2C1

Replaces Fragment B (the amine partner) from three sources:
  (a) Hand-curated kinase-relevant aminoheterocycles (~30)
  (b) ChEMBL miner: primary/secondary amines from kinase training mols (~500–1500)
  (c) ML-novel amines: REINVENT4 Mol2Mol seeded on (a)+(b) (deferred — runs separately)

Synthesis: single amide coupling step (HATU/EDC) → unambiguous route.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tier2_fragreplace.py
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
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_film_predictor, load_zap70_train_smiles,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_tier2_fragreplace"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "tier2_results.json"
OUT_CSV = RESULTS_DIR / "tier2_candidates.csv"
OUT_HTML = RESULTS_DIR / "tier2_report.html"

# Fragment A: acrylamide-isoindoline carboxylic acid
FRAG_A_ACID = "C=CC(=O)N1Cc2cccc(C(=O)O)c2C1"

# ── Hand-curated amine library (~30) ──────────────────────────────────────────
HAND_AMINES = [
    # Original Mol 1 amine
    ("Nc1cn(C(C)C)cn1", "imidazol-4-amine_iPr", "Mol 1 native"),
    # Imidazole variants
    ("Nc1cncn1C", "imidazol-4-amine_Me", "1-methyl"),
    ("Nc1cn(CC)cn1", "imidazol-4-amine_Et", "1-ethyl"),
    ("Nc1cn(C2CC2)cn1", "imidazol-4-amine_cPr", "1-cyclopropyl"),
    ("Nc1cn(C(C)(C)C)cn1", "imidazol-4-amine_tBu", "1-tert-butyl"),
    # Pyrazole variants
    ("Nc1ccn(C(C)C)n1", "pyrazol-3-amine_iPr", ""),
    ("Nc1cnn(C)c1", "pyrazol-3-amine_Me", ""),
    ("Nc1ccnn1C(C)C", "pyrazol-5-amine_iPr", ""),
    # Triazole variants
    ("Nc1nn(C(C)C)cn1", "1,2,4-triazol-3-amine", ""),
    ("Nc1cn(C(C)C)nn1", "1,2,3-triazol-4-amine", ""),
    # Other aminoheterocycles
    ("Nc1cnoc1", "isoxazol-3-amine", ""),
    ("Nc1cnsc1", "isothiazol-3-amine", ""),
    ("Nc1ncc(C)s1", "thiazol-2-amine", ""),
    ("Nc1ccnc(N)n1", "2,4-diaminopyrimidine", ""),
    ("Nc1ncccn1", "pyrimidin-2-amine", ""),
    ("Nc1ccncn1", "pyrimidin-4-amine", ""),
    ("Nc1ccncc1", "pyridin-4-amine", ""),
    ("Nc1cccnc1", "pyridin-3-amine", ""),
    ("Nc1ccccn1", "pyridin-2-amine", ""),
    # Aniline variants
    ("Nc1ccccc1", "aniline", ""),
    ("Nc1ccc(F)cc1", "4-F-aniline", ""),
    ("Nc1ccc(Cl)cc1", "4-Cl-aniline", ""),
    ("Nc1ccc(OC)cc1", "4-OMe-aniline", ""),
    ("Nc1cccc(C)c1", "3-Me-aniline", ""),
    ("Nc1cc(C)cc(C)c1", "3,5-Me2-aniline", ""),
    # Benzylamine
    ("NCc1ccccc1", "benzylamine", ""),
    ("NCc1cnc(C(C)C)cn1", "amino-Me-pyrazinyl", ""),
    # Aliphatic small amines (controls)
    ("NC1CCCCC1", "cyclohexylamine", ""),
    ("NCC1CCCO1", "tetrahydrofuran-2-yl-methylamine", ""),
    ("NC1CCNC1", "pyrrolidin-3-amine", ""),
    ("NCCN1CCCC1", "pyrrolidinyl-ethylamine", ""),
    # Aminoindazole (kinase-classic)
    ("Nc1cc2ccccc2[nH]1", "1H-indol-3-amine", ""),
    ("Nc1[nH]c2ccccc2n1", "1H-benzimidazol-2-amine", ""),
    ("Nc1nc2ccccc2[nH]1", "1H-benzimidazol-2-amine_alt", ""),
]


def has_amine_for_coupling(mol):
    """Return atom index of a primary or secondary amine N suitable for amide coupling, or None."""
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N':
            continue
        # Skip aromatic N, charged N, N-oxide
        if atom.GetIsAromatic() or atom.GetFormalCharge() != 0:
            continue
        # Skip N already in an amide/imide
        is_amide = False
        for nb in atom.GetNeighbors():
            for nbb in nb.GetBonds():
                if nbb.GetBeginAtomIdx() != atom.GetIdx() and nbb.GetEndAtomIdx() != atom.GetIdx():
                    continue
                if nb.GetSymbol() == 'C':
                    for nbb2 in nb.GetBonds():
                        other = nbb2.GetOtherAtom(nb)
                        if other.GetSymbol() == 'O' and nbb2.GetBondType() == Chem.BondType.DOUBLE:
                            is_amide = True
        if is_amide:
            continue
        # Need at least 1 H (primary or secondary amine for amide coupling)
        if atom.GetTotalNumHs() < 1:
            continue
        return atom.GetIdx()
    return None


_AMIDE_RXN = AllChem.ReactionFromSmarts(
    "[C:1](=[O:2])[O;H1].[N;H1,H2;!$(N=*);!$(N#*);!$(N*=O):3] >> [C:1](=[O:2])[N:3]"
)


def couple_amide(acid_smi: str, amine_smi: str):
    """Produce the amide coupling product via a reaction SMARTS:
    acid-COOH + H-NR → acid-C(=O)-NR.
    Returns canonical SMILES or None.
    """
    acid = Chem.MolFromSmiles(acid_smi)
    amine = Chem.MolFromSmiles(amine_smi)
    if acid is None or amine is None:
        return None
    try:
        products = _AMIDE_RXN.RunReactants((acid, amine))
    except Exception:
        return None
    if not products:
        return None
    # Pick first valid product
    for prod_set in products:
        if not prod_set:
            continue
        p = prod_set[0]
        try:
            Chem.SanitizeMol(p)
            return Chem.MolToSmiles(p)
        except Exception:
            continue
    return None


# ── ChEMBL miner ──────────────────────────────────────────────────────────────

def mine_chembl_amines(max_amines=2000, mw_max=300):
    """Mine primary/secondary amines from kinase training molecules.
    Two strategies:
      (a) whole-mol: small kinase mols that are themselves amines (MW ≤ mw_max)
      (b) BRICS decomposition: extract small amine fragments from larger mols
    """
    from rdkit.Chem import BRICS

    sources = []
    try:
        sources.extend(load_zap70_train_smiles())
    except Exception:
        pass
    kpairs_path = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
    if kpairs_path.exists():
        try:
            df = pd.read_csv(kpairs_path, usecols=["mol_a", "mol_b"], nrows=500_000)
            sources.extend(df['mol_a'].tolist()[:30000])
            sources.extend(df['mol_b'].tolist()[:30000])
        except Exception:
            pass

    print(f"  ChEMBL miner: scanning {len(sources):,} source molecules")
    seen = set()
    amines = []
    acryl_pat = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")

    # Strategy (a): whole molecule
    for smi in sources:
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if Descriptors.MolWt(mol) > mw_max:
            continue
        if has_amine_for_coupling(mol) is None:
            continue
        if mol.HasSubstructMatch(acryl_pat):
            continue
        cs = Chem.MolToSmiles(mol)
        if cs in seen:
            continue
        seen.add(cs)
        amines.append((cs, "ChEMBL_whole", f"MW={Descriptors.MolWt(mol):.0f}"))
        if len(amines) >= max_amines // 2:
            break
    print(f"  ChEMBL whole-mol amines: {len(amines)}")

    # Strategy (b): BRICS decomposition of bigger kinase mols
    pre_brics = len(amines)
    for smi in sources[:20_000]:
        if not isinstance(smi, str):
            continue
        if len(amines) >= max_amines:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None or Descriptors.MolWt(mol) < 200:
            continue
        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
        except Exception:
            continue
        for f in frags:
            # Strip BRICS dummy atoms ([n*]) — keep the core fragment
            cleaned = Chem.MolFromSmiles(f)
            if cleaned is None:
                continue
            # Replace dummy atoms with [H] to get a clean amine
            cleaned = Chem.DeleteSubstructs(cleaned, Chem.MolFromSmarts("[*;!#1;!#6;!#7;!#8;!#9;!#16;!#17;!#35]"))
            try:
                Chem.SanitizeMol(cleaned)
            except Exception:
                continue
            mw = Descriptors.MolWt(cleaned)
            if mw < 60 or mw > mw_max:
                continue
            if has_amine_for_coupling(cleaned) is None:
                continue
            if cleaned.HasSubstructMatch(acryl_pat):
                continue
            cs = Chem.MolToSmiles(cleaned)
            if cs in seen:
                continue
            seen.add(cs)
            amines.append((cs, "ChEMBL_BRICS", f"MW={mw:.0f}"))
            if len(amines) >= max_amines:
                break
    print(f"  BRICS-derived amines: {len(amines) - pre_brics}")
    print(f"  ChEMBL miner: total {len(amines)} amines (MW ≤ {mw_max})")
    return amines


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MOL 1 — TIER 2: FRAGMENT REPLACEMENT (AMIDE COUPLING)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fragment A (fixed): {FRAG_A_ACID}")

    # Build amine library
    print("\n[+] Building amine library")
    amines = []
    print(f"  Hand-curated: {len(HAND_AMINES)}")
    amines.extend([(s, src, note) for s, src, note in HAND_AMINES])

    print("  Mining ChEMBL kinase amines...")
    amines.extend(mine_chembl_amines(max_amines=1500))
    print(f"  TOTAL amine library: {len(amines)}")

    # Couple
    print("\n[+] Enumerating amide-coupling products")
    products = []
    seen = {Chem.CanonSmiles(MOL1_SMILES)}
    for am_smi, src, note in amines:
        prod = couple_amide(FRAG_A_ACID, am_smi)
        if prod is None:
            continue
        try:
            csm = Chem.CanonSmiles(prod)
        except Exception:
            continue
        if csm in seen:
            continue
        seen.add(csm)
        if not warhead_intact(csm):
            continue
        products.append({
            "smiles": csm, "method": "Tier2_frag_replace",
            "amine_source": src, "amine_smiles": am_smi, "note": note,
        })
    print(f"  Coupling products (unique, warhead-intact): {len(products)}")

    if not products:
        print("ERROR: no products generated.")
        return

    # Score
    print("\n[+] Scoring with FiLMDelta + 3D + descriptors")
    df = pd.DataFrame(products)
    train = load_zap70_train_smiles()
    pred = load_film_predictor()
    df = score_dataframe(df, train_smiles=train, compute_3d=True, pIC50_predictor=pred)
    df = df.sort_values("pIC50", ascending=False).reset_index(drop=True)

    # Save
    df.to_csv(OUT_CSV, index=False)
    out = {
        "tier": 2, "method": "fragment_replacement_amide_coupling",
        "seed": MOL1_SMILES,
        "fragment_A": FRAG_A_ACID,
        "n_amines": len(amines),
        "n_products": len(df),
        "per_amine_source": dict(df["amine_source"].value_counts()),
        "timestamp": datetime.now().isoformat(),
        "candidates": df.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2))
    print(f"[+] Saved → {OUT_JSON}")

    # Print top-10
    print("\nTop 10 by FiLMDelta pIC50:")
    print(f"{'#':<3}{'pIC50':<7}{'SAS':<6}{'shape_Tc':<10}{'wrhd_dev':<10}{'src':<18s}{'note':<24s}")
    for i, r in df.head(10).iterrows():
        print(f"{i+1:<3}{r['pIC50']:<7.3f}{r['SAScore']:<6.2f}"
              f"{r['shape_Tc_seed']:<10.3f}{r['warhead_dev_deg']:<10.1f}"
              f"{r['amine_source']:<18s}{str(r.get('note',''))[:22]:<24s}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
