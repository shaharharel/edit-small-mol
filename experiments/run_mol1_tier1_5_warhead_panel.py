#!/usr/bin/env python3
"""
Tier 1.5 — Warhead Control Panel + Med-Chem Tricks.

Two subsections, both deterministic / rule-based, both small (~10–30 mols).

1.5a — Warhead Control Panel
    The acrylamide is REPLACED with a panel of related electrophiles + a
    saturated null. These are NOT warhead-preserved; they're controls/comparators
    for SAR interpretation in a covalent series.

    Variants:
      α-Me, α-F, α-CN, α-CF3 acrylamide        (steric/electronic α-tuning)
      β-Me (E-crotonamide), β,β-diMe            (steric β-tuning)
      β-NMe2 acrylamide (afatinib-style trick)  (basic-amine PK lever)
      Propiolamide, 2-butynamide                (alkyne warheads, geometry comparator)
      Propanamide                               (saturated null — mandatory comparator)

1.5b — Med-Chem Tricks (warhead-preserved)
    Specific transformations the agents flagged as missing from Tier 1.

    Variants:
      Imidazole C5 scan                        (Me, Cl, NH2, CN at imidazole C5)
      Ortho-to-amide on isoindoline aryl       (force ortho-Me, F, Cl, OMe on the aryl-C
                                                immediately adjacent to the carboxamide)
      1-Substituted isoindoline                (introduces stereocenter — 1-Me, 1-Ph, 1-cPr)
      Ring contraction → pyrrolinone           (5-ring carbocycle of isoindoline → 4-ring)

Outputs:
    results/paper_evaluation/mol1_tier1_5_warhead_panel/tier1_5_results.json
    plus separate CSVs for the two subsections.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tier1_5_warhead_panel.py
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
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_tier1_5_warhead_panel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def _replace_smiles_pattern(seed_smi: str, old: str, new: str):
    """Replace exact SMILES pattern in seed via ReplaceSubstructs."""
    mol = Chem.MolFromSmiles(seed_smi)
    base = Chem.MolFromSmiles(old)
    repl = Chem.MolFromSmiles(new)
    if mol is None or base is None or repl is None or not mol.HasSubstructMatch(base):
        return []
    out = []
    seen = set()
    try:
        prods = AllChem.ReplaceSubstructs(mol, base, repl, replaceAll=True)
        for p in prods:
            try:
                Chem.SanitizeMol(p)
                s = Chem.MolToSmiles(p)
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _run_reaction(seed_smi: str, smarts: str):
    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn is None:
        return []
    mol = Chem.MolFromSmiles(seed_smi)
    if mol is None:
        return []
    out, seen = [], set()
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
            if s not in seen:
                seen.add(s)
                out.append(s)
        except Exception:
            continue
    return out


# ── 1.5a: Warhead Control Panel ───────────────────────────────────────────────

def warhead_panel(seed_smi):
    """Replace the acrylamide with related electrophiles + saturated null.
    Each variant keeps the rest of Mol 1 fixed."""
    # Old acrylamide group attached to isoindoline N: C=CC(=O)N
    # Use full SMILES including the ring N to be specific.
    candidates = [
        # α-substituted acrylamides
        ("CC(=C)C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_alpha_Me", "α-methyl acrylamide"),
        ("FC(=C)C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_alpha_F", "α-fluoro acrylamide (Astellas)"),
        ("N#CC(=C)C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_alpha_CN", "α-cyano (rev. covalent, Taunton)"),
        ("FC(F)(F)C(=C)C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_alpha_CF3", "α-trifluoromethyl acrylamide"),
        # β-substituted acrylamides
        ("C/C=C/C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_beta_Me", "(E)-β-methyl (crotonamide)"),
        ("CC(C)=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_beta_diMe", "β,β-dimethyl"),
        ("CN(C)/C=C/C(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_beta_NMe2", "β-NMe2 (afatinib trick)"),
        # Alkyne warheads
        ("C#CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_propiolamide", "propiolamide (alkyne)"),
        ("CC#CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_2_butynamide", "2-butynamide"),
        # Saturated null (mandatory control)
        ("CCC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "warhead_a_propanamide_null", "propanamide (saturated null)"),
        # Reverse-direction warhead (acryloyl attached via C, not N)
        # Skipped — would change connectivity entirely.
    ]
    return [(canon(s), m, n) for s, m, n in candidates if Chem.MolFromSmiles(s)]


# ── 1.5b: Med-Chem Tricks (warhead-preserved) ─────────────────────────────────

def imidazole_c5_scan(seed_smi):
    """Substitute the imidazole C5-H position. C5 is the carbon adjacent to N1-iPr."""
    base = "Nc1cn(C(C)C)cn1"
    variants = [
        ("Nc1c(C)n(C(C)C)cn1", "C5-Me"),
        ("Nc1c(Cl)n(C(C)C)cn1", "C5-Cl"),
        ("Nc1c(N)n(C(C)C)cn1", "C5-NH2"),
        ("Nc1c(C#N)n(C(C)C)cn1", "C5-CN"),
        ("Nc1c(C(F)(F)F)n(C(C)C)cn1", "C5-CF3"),
    ]
    out = []
    for new_smi, name in variants:
        for prod in _replace_smiles_pattern(seed_smi, base, new_smi):
            out.append((prod, "Tier1_5b_imidazole_C5", name))
    return out


def ortho_to_amide_scan(seed_smi):
    """Force ortho substitutents on the isoindoline aryl C adjacent to C(=O)NH-imidazole.
    In Mol 1 the aryl pattern is c2cccc(C(=O)Nc3...)c2 — the ortho positions to the
    carboxamide-attached C are positions 4 and 6 of the isoindoline aryl (idx 5, 7 in
    canonical SMILES).
    """
    # Use reaction SMARTS to add a substituent to the aryl-C ortho to the amide carbonyl.
    rules = [
        ("[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[cH:10][c:11][c:12][c:13][c:14]1>>"
         "[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[c:10](C)[c:11][c:12][c:13][c:14]1",
         "ortho-Me"),
        ("[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[cH:10][c:11][c:12][c:13][c:14]1>>"
         "[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[c:10](F)[c:11][c:12][c:13][c:14]1",
         "ortho-F"),
        ("[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[cH:10][c:11][c:12][c:13][c:14]1>>"
         "[c:1]1([c:2][C:3](=O)N[c:4]2[c:5][n:6]([C:7])[c:8][n:9]2)[c:10](Cl)[c:11][c:12][c:13][c:14]1",
         "ortho-Cl"),
    ]
    out = []
    for smarts, name in rules:
        for prod in _run_reaction(seed_smi, smarts):
            out.append((prod, "Tier1_5b_ortho_to_amide", name))
    return out


def isoindoline_1_sub_scan(seed_smi):
    """Add 1-substituent on isoindoline saturated CH2 (the one closer to C-C(=O)
    of the warhead) — introduces a stereocenter."""
    # isoindoline carbon adjacent to the warhead N — `N1Cc2...c2C1` → `N1C(R)c2...c2C1`
    base = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
    variants = [
        ("C=CC(=O)N1[C@@H](C)c2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "1-Me (S)"),
        ("C=CC(=O)N1[C@H](C)c2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "1-Me (R)"),
        ("C=CC(=O)N1C(C)(C)c2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "1,1-diMe"),
        ("C=CC(=O)N1C(c2ccccc2)c3cccc(C(=O)Nc4cn(C(C)C)cn4)c3C1", "1-Ph"),
    ]
    out = []
    for new_smi, name in variants:
        m = Chem.MolFromSmiles(new_smi)
        if m is None:
            continue
        s = canon(new_smi)
        if s == canon(seed_smi):
            continue
        out.append((s, "Tier1_5b_isoindoline_1sub", name))
    return out


def aza_isoindoline(seed_smi):
    """Replace isoindoline aryl C with N at specific positions — fused-N variants
    (different from generic Tier 1 N-walk because positions are pharmacology-relevant)."""
    base = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
    variants = [
        ("C=CC(=O)N1Cc2cncc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "5-aza (pyrid-3-yl-fused)"),
        ("C=CC(=O)N1Cc2ccnc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "6-aza"),
        ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2N1", "isoindoline N-N (replaces CH2)"),  # unusual
    ]
    out = []
    for new_smi, name in variants:
        m = Chem.MolFromSmiles(new_smi)
        if m is None:
            continue
        s = canon(new_smi)
        if s == canon(seed_smi):
            continue
        out.append((s, "Tier1_5b_aza_isoindoline", name))
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MOL 1 — TIER 1.5: WARHEAD CONTROLS + MED-CHEM TRICKS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1.5a Warhead controls ─────────────────────────────────────────────────
    print("\n--- 1.5a: Warhead Control Panel ---")
    panel = warhead_panel(MOL1_SMILES)
    panel_a = []
    seen = {canon(MOL1_SMILES)}
    for smi, method, note in panel:
        if smi in seen:
            continue
        seen.add(smi)
        panel_a.append({"smiles": smi, "method": method, "note": note,
                        "subsection": "1.5a_warhead_controls"})
    print(f"  Generated {len(panel_a)} warhead-control variants")
    for c in panel_a:
        print(f"    {c['method']:<32s} {c['note']}")

    # ── 1.5b Med-chem tricks ──────────────────────────────────────────────────
    print("\n--- 1.5b: Med-Chem Tricks (warhead-preserved) ---")
    rules_b = [
        ("Imidazole C5 scan", imidazole_c5_scan),
        ("Ortho-to-amide", ortho_to_amide_scan),
        ("Isoindoline 1-sub", isoindoline_1_sub_scan),
        ("Aza-isoindoline (specific)", aza_isoindoline),
    ]
    panel_b = []
    seen_b = {c["smiles"] for c in panel_a} | seen
    for label, fn in rules_b:
        try:
            prods = fn(MOL1_SMILES)
        except Exception as e:
            print(f"  {label:<28s} ERROR: {e}")
            continue
        kept = 0
        for smi, method, note in prods:
            if smi in seen_b:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None or not warhead_intact(m):
                continue
            seen_b.add(smi)
            panel_b.append({"smiles": smi, "method": method, "note": note,
                           "subsection": "1.5b_med_chem_tricks"})
            kept += 1
        print(f"  {label:<28s} → {kept} candidates")
    print(f"  Total 1.5b: {len(panel_b)}")

    all_candidates = panel_a + panel_b
    print(f"\nTotal Tier 1.5: {len(all_candidates)} (1.5a={len(panel_a)} + 1.5b={len(panel_b)})")

    if not all_candidates:
        print("ERROR: no candidates produced.")
        return

    # ── Score ─────────────────────────────────────────────────────────────────
    print("\n[+] Scoring with FiLMDelta + 3D (shape Tc + ESP-Sim + warhead Δ°) + descriptors")
    df = pd.DataFrame(all_candidates)
    train = load_zap70_train_smiles()
    pred = load_film_predictor()
    df = score_dataframe(df, train_smiles=train, compute_3d=True, pIC50_predictor=pred)

    # 1.5a warhead variants will have warhead_intact=False — that's expected
    df = df.sort_values("pIC50", ascending=False).reset_index(drop=True)

    df.to_csv(RESULTS_DIR / "tier1_5_candidates.csv", index=False)
    out = {
        "tier": 1.5, "method": "warhead_panel_plus_med_chem_tricks",
        "seed": MOL1_SMILES,
        "n_total": len(df),
        "n_1_5a_warhead_controls": len(panel_a),
        "n_1_5b_med_chem_tricks": len(panel_b),
        "timestamp": datetime.now().isoformat(),
        "candidates": df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "tier1_5_results.json").write_text(
        json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2)
    )
    print(f"[+] Saved → {RESULTS_DIR / 'tier1_5_results.json'}")

    print("\nTop 15 by FiLMDelta pIC50 (across both subsections):")
    print(f"{'#':<3}{'pIC50':<7}{'SAS':<6}{'shape_Tc':<10}{'esp_sim':<10}{'wrhd_dev':<10}{'subs':<8s}{'note':<30s}")
    for i, r in df.head(15).iterrows():
        sub = "1.5a" if r["subsection"] == "1.5a_warhead_controls" else "1.5b"
        wd = r.get('warhead_dev_deg', float('nan'))
        wd_s = f"{wd:.1f}" if not np.isnan(wd) else "—"
        es = r.get('esp_sim_seed', float('nan'))
        es_s = f"{es:+.3f}" if not np.isnan(es) else "—"
        st = r.get('shape_Tc_seed', float('nan'))
        st_s = f"{st:.3f}" if not np.isnan(st) else "—"
        print(f"{i+1:<3}{r['pIC50']:<7.3f}{r['SAScore']:<6.2f}{st_s:<10s}{es_s:<10s}{wd_s:<10s}{sub:<8s}{str(r.get('note',''))[:28]:<30s}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
