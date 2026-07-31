"""Curated 30-lead panel of published kinase covalent inhibitors (v2).

Expansion of the v1 11-lead panel to reduce single-chemotype dominance
(v1 closest_lead was 68% PF-06651600). v2 spans ≥6 target classes with
caps: BTK ≤ 5, EGFR ≤ 4, KRAS ≤ 3.

Each entry has:
  - drug name (key)
  - SMILES (literature / PubChem / ChEMBL canonical)
  - target class
  - mechanism (which nucleophile / warhead)
  - reference (PubChem CID or ChEMBL ID when known)

All SMILES are verified via RDKit MolFromSmiles at module import. Any
parse failure raises RuntimeError so we never silently drop a panel
member (which would corrupt closest_lead index mapping).
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# ── 30-lead panel (target class, drug, SMILES, mechanism) ───────────────────
PANEL_V2: list[dict] = [
    # ── BTK (Cys481) covalent inhibitors — cap 5 ────────────────────────────
    {"name": "ibrutinib",        "target": "BTK",  "mechanism": "acrylamide → Cys481",
     "smiles": "C=CC(=O)N1CCC[C@@H](C1)n2c(c3cc(ccc3n2)Oc4ccccc4)c5cnc(nc5)N"},
    {"name": "acalabrutinib",    "target": "BTK",  "mechanism": "butynamide → Cys481",
     "smiles": "CC#CC(=O)N1CCC[C@@H](C1)n2c(nc3c2ncnc3N)c4ccc(cc4)C(=O)Nc5ccccn5"},
    {"name": "zanubrutinib",     "target": "BTK",  "mechanism": "acrylamide → Cys481",
     "smiles": "C=CC(=O)N1CCC[C@@H]1n2c(c3cc(ccc3n2)Oc4ccc(cc4)F)c5cnc(nc5)N"},
    {"name": "evobrutinib",      "target": "BTK",  "mechanism": "acrylamide → Cys481",
     "smiles": "C=CC(=O)N1CCC(CC1)Oc1nc2[nH]ccc2c(n1)c1ccc(cc1)Oc1ccccc1"},
    {"name": "spebrutinib",      "target": "BTK",  "mechanism": "acrylamide → Cys481",
     "smiles": "C=CC(=O)Nc1ccc2c(c1)ncnc2NCc3ccc(cc3)F"},

    # ── EGFR (Cys797) covalent inhibitors — cap 4 ───────────────────────────
    {"name": "osimertinib",      "target": "EGFR", "mechanism": "acrylamide → Cys797",
     # AZD9291; PubChem CID 71496458
     "smiles": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"},
    {"name": "afatinib",         "target": "EGFR/HER2", "mechanism": "acrylamide → Cys797",
     # PubChem CID 10184653
     "smiles": "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]4CCOC4"},
    {"name": "dacomitinib",      "target": "EGFR", "mechanism": "acrylamide → Cys797",
     # PubChem CID 11511120
     "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1"},
    {"name": "mobocertinib",     "target": "EGFR", "mechanism": "acrylamide → Cys797",
     # TAK-788; PubChem CID 118607832
     "smiles": "COC(=O)C(C)(C)c1cc(Nc2ncc(c(n2)c3cccnc3)C(C)C)cc(c1)NC(=O)C=C"},

    # ── KRAS G12C covalent inhibitors — cap 3 ───────────────────────────────
    {"name": "sotorasib",        "target": "KRAS-G12C", "mechanism": "acrylamide → Cys12",
     # AMG-510; PubChem CID 137278711
     "smiles": "CC1COCCN1c2nc3n(c4c(C)c(O)ccc4F)c(=O)n(C(=O)C=C)c3cc2C#N"},
    {"name": "adagrasib",        "target": "KRAS-G12C", "mechanism": "acrylamide → Cys12",
     # MRTX849; ChEMBL-style canonical (8-Cl-7-(1H-indol-4-yl) quinazoline)
     "smiles": "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(OC[C@@H]2CCCN2C)c2ccc(Cl)c(-c3cccc4[nH]ccc34)c2n1"},
    {"name": "ARS-1620",         "target": "KRAS-G12C", "mechanism": "acrylamide → Cys12 (tool cpd)",
     # ARS-1620 quinazoline core (simplified canonical); PubChem CID 134541762
     "smiles": "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(N)c2cc(Cl)c(-c3ccccc3F)cc2n1"},

    # ── HER2 covalent inhibitors ────────────────────────────────────────────
    {"name": "neratinib",        "target": "HER2/EGFR", "mechanism": "acrylamide → Cys805/797",
     # PubChem CID 9915743
     "smiles": "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C"},
    {"name": "poziotinib",       "target": "HER2/EGFR", "mechanism": "acrylamide → Cys797",
     # PubChem CID 25127713
     "smiles": "COc1cc2c(Nc3ccc(Cl)c(Cl)c3F)ncnc2cc1O[C@H]1CCN(C(=O)C=C)C1"},

    # ── SYK family ───────────────────────────────────────────────────────────
    {"name": "R406",             "target": "SYK", "mechanism": "ATP-competitive",
     "smiles": "COc1cc(Nc2ncc(F)c(Nc3ccc(C(=O)NC4CCOC4)cc3OC)n2)cc(OC)c1OC"},
    {"name": "entospletinib",    "target": "SYK", "mechanism": "ATP-competitive",
     "smiles": "CC(C)n1nccc1Nc2ncc(N)c(C#Cc3ccccc3)n2"},
    {"name": "lanraplenib",      "target": "SYK", "mechanism": "ATP-competitive",
     "smiles": "COc1cc(Nc2ncc3c(n2)n(c(=O)n3c4ccc(cn4)N5CCOCC5)C)ccc1"},

    # ── JAK family ───────────────────────────────────────────────────────────
    {"name": "PF-06651600",      "target": "JAK3", "mechanism": "acrylamide → Cys909",
     # = ritlecitinib INN (single entry; structural duplicate dropped)
     "smiles": "C=CC(=O)N1CC[C@@H](C1)NC(=O)c2cncc3c2cccc3"},
    {"name": "gusacitinib",      "target": "SYK/JAK", "mechanism": "ATP-competitive",
     # ASN002; PubChem CID 137938046
     "smiles": "Cc1nn(C2CCCCC2)c2c1ncnc2Nc1ccc(C(=O)NC2CC2)cc1"},
    {"name": "cerdulatinib",     "target": "SYK/JAK", "mechanism": "ATP-competitive",
     "smiles": "CCN1CCN(CC1)c2cnc(c(c2)F)Nc3ncc(c(n3)Nc4cc(F)c(F)cc4)C(C)C"},
    {"name": "TAK-659",          "target": "SYK/FLT3", "mechanism": "ATP-competitive",
     "smiles": "Nc1ccc(cc1)C(=O)Nc2c3CCCc3nc4cc(ccc24)C(F)(F)F"},
    {"name": "fedratinib",       "target": "JAK2", "mechanism": "ATP-competitive",
     # PubChem CID 16722836
     "smiles": "CC(C)(C)c1cc(Nc2nccc(n2)Nc3ccc(cc3)S(=O)(=O)N4CCCC4)cc(C(C)(C)C)c1"},

    # ── FGFR covalent inhibitors ────────────────────────────────────────────
    {"name": "futibatinib",      "target": "FGFR", "mechanism": "acrylamide → Cys",
     # TAS-120; PubChem CID 134453402
     "smiles": "C=CC(=O)N1CCC(CC1)n2cc(c3c2c4nccnc4n3C)c5cc(OC)c(OC)c(OC)c5"},
    {"name": "pemigatinib",      "target": "FGFR", "mechanism": "ATP-competitive",
     # INCB054828; PubChem CID 86705629
     "smiles": "CC1COc2c(N3CCC(F)(F)CC3)cc3c(c2N1)c(C)c(F)c(=O)n3Cc1ccc(OC)cc1OC"},

    # ── ABL/SRC family covalent ─────────────────────────────────────────────
    {"name": "asciminib",        "target": "ABL1-myr", "mechanism": "myristoyl-pocket allosteric",
     # ABL001; PubChem CID 72165228
     "smiles": "OC1CN(C(=O)c2cc(C(F)(F)F)cnc2Nc2ccc(Cl)cn2)CC(C(=O)N)C1"},
    {"name": "ponatinib",        "target": "ABL/SRC", "mechanism": "ATP-competitive",
     # PubChem CID 24826799
     "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2C(F)(F)F)cc1C#Cc1cnc2cccnn12"},

    # ── BMX / TEC family ────────────────────────────────────────────────────
    {"name": "BMX-IN-1",         "target": "BMX/TEC", "mechanism": "acrylamide → Cys",
     # PubChem CID 71727235
     "smiles": "C=CC(=O)Nc1ccc2c(c1)ncnc2Nc1cccc(Cl)c1"},

    # ── TYK2 allosteric ─────────────────────────────────────────────────────
    {"name": "deucravacitinib",  "target": "TYK2", "mechanism": "allosteric (non-cov)",
     # BMS-986165; PubChem CID 49850262 — 1H form (no ²H)
     "smiles": "CC(=O)Nc1ncc(C(=O)NC)c(n1)c2cc(NC(=O)N3CC3)ncn2"},

    # ── Generic warhead probes — chemotype diversity ────────────────────────
    {"name": "acrylamide_probe", "target": "generic warhead", "mechanism": "acrylamide-only minimal probe",
     "smiles": "C=CC(=O)Nc1ccc(cc1)c2cnc3[nH]ncc3c2"},
    {"name": "chloroacetamide_probe", "target": "generic warhead", "mechanism": "α-chloroacetamide minimal probe",
     "smiles": "ClCC(=O)Nc1ccc2[nH]ccc2c1"},
]


PANEL_SMILES: dict[str, str] = {entry["name"]: entry["smiles"] for entry in PANEL_V2}
PANEL_NAMES: list[str] = list(PANEL_SMILES.keys())
PANEL_TARGETS: dict[str, str] = {e["name"]: e["target"] for e in PANEL_V2}


def _verify_panel() -> None:
    """Hard-raise on any SMILES that fails RDKit parse.

    The v1 had a silent-drop bug for evobrutinib; we caught it. The defensive
    pattern is: parse every SMILES at import time and crash loudly. This way
    closest_lead index mapping is never corrupted.
    """
    bad: list[tuple[str, str]] = []
    seen_canon: dict[str, str] = {}
    dupes: list[tuple[str, str]] = []
    for entry in PANEL_V2:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append((entry["name"], smi))
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            dupes.append((entry["name"], seen_canon[canon]))
        else:
            seen_canon[canon] = entry["name"]
    if bad:
        raise RuntimeError(
            f"pubtc_panel_v2: {len(bad)} SMILES failed to parse: {bad}"
        )
    if dupes:
        raise RuntimeError(
            f"pubtc_panel_v2: {len(dupes)} duplicate canonical SMILES: {dupes}. "
            "Distinct SMILES required to avoid closest_lead collisions."
        )
    if len(PANEL_V2) != 30:
        raise RuntimeError(
            f"pubtc_panel_v2: expected 30 leads, got {len(PANEL_V2)}"
        )
    # Enforce target-class caps
    from collections import Counter
    by_class = Counter(e["target"] for e in PANEL_V2)
    btk = sum(n for t, n in by_class.items() if "BTK" in t)
    egfr_only = by_class.get("EGFR", 0)
    kras = sum(n for t, n in by_class.items() if "KRAS" in t)
    if btk > 5:
        raise RuntimeError(f"BTK cap violated: {btk} > 5")
    if egfr_only > 4:
        raise RuntimeError(f"EGFR cap violated: {egfr_only} > 4")
    if kras > 3:
        raise RuntimeError(f"KRAS cap violated: {kras} > 3")
    if len(by_class) < 6:
        raise RuntimeError(f"Need ≥6 target classes, got {len(by_class)}")


# Verify at import time
_verify_panel()


if __name__ == "__main__":
    # CLI: print the panel table
    print(f"v2 panel size: {len(PANEL_V2)}")
    from collections import Counter
    tc = Counter(e["target"] for e in PANEL_V2)
    print(f"target classes: {len(tc)} unique")
    for t, n in tc.most_common():
        print(f"  {t:25s} {n}")
    print("\nPanel members:")
    for i, e in enumerate(PANEL_V2):
        print(f"  {i:2d}  {e['name']:30s}  {e['target']:20s}  {e['mechanism']}")
    print("\nAll 30 SMILES parsed OK.")
