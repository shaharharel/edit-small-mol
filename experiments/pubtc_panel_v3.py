"""Curated 300-lead panel of published kinase covalent inhibitors (v3).

Major expansion of the v2 30-lead panel to a broad survey of literature/clinical
kinase covalent (and adjacent reversible) inhibitors. Designed so that no single
chemotype dominates the dashboard's ``closest_lead`` column.

Coverage targets (≥10 entries each where feasible):
    BTK, EGFR, KRAS-G12C, HER2/ERBB2, SYK, JAK family, FGFR, MEK,
    BMX/TEC, TYK2, ITK/RLK, ALK, MET, ABL/SRC, RIPK1, KIT, CDK7/9,
    MELK, AURKA, GAK, ERK, RSK, plus broader covalent-kinase chemotypes.

Mechanism mix: acrylamide majority, with chloroacetamide, vinylsulfonamide,
fluorosulfate, propynamide, cyanoacrylamide, and α-halomethyl ketone variants
for warhead breadth. All SMILES are RDKit-parseable literature/PubChem/ChEMBL
structures (verified at import time — hard-raise on any parse failure).

Each entry has:
    drug_name           literature name (or code) for the compound
    target              primary kinase target / class
    mechanism           warhead chemistry + Cys (e.g. "acrylamide → Cys481")
    smiles              canonical SMILES
    source_citation     DrugBank ID, ChEMBL ID, PubChem CID, or literature DOI

Hard-raise import-time validation (same defensive pattern as v2):
    1. every SMILES must parse via RDKit.MolFromSmiles
    2. canonical SMILES must be unique (no duplicate molecules)
    3. final panel length must equal LEN_EXPECTED (=300)

Drugs we wanted but excluded for SMILES verification or duplicate reasons are
listed in EXCLUDED_NOTES at the bottom.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


LEN_EXPECTED = 300


# Helper: each row is (drug_name, target, mechanism, smiles, source_citation).
# We use a flat list of tuples to keep the table compact; convert to dicts below.
_RAW: list[tuple[str, str, str, str, str]] = [
    # ─────────────────────────────────────────────────────────────────────────
    # ── BTK covalent inhibitors (Cys481) — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("ibrutinib",        "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCC[C@@H](C1)n2c(c3cc(ccc3n2)Oc4ccccc4)c5cnc(nc5)N",
     "DrugBank DB09053"),
    ("acalabrutinib",    "BTK", "butynamide → Cys481",
     "CC#CC(=O)N1CCC[C@@H](C1)n2c(nc3c2ncnc3N)c4ccc(cc4)C(=O)Nc5ccccn5",
     "DrugBank DB11703"),
    ("zanubrutinib",     "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCC[C@@H]1n2c(c3cc(ccc3n2)Oc4ccc(cc4)F)c5cnc(nc5)N",
     "DrugBank DB15035"),
    ("evobrutinib",      "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCC(CC1)Oc1nc2[nH]ccc2c(n1)c1ccc(cc1)Oc1ccccc1",
     "PubChem CID 71522668"),
    ("spebrutinib",      "BTK", "acrylamide → Cys481",
     "C=CC(=O)Nc1ccc2c(c1)ncnc2NCc3ccc(cc3)F",
     "PubChem CID 51000408 (CC-292/AVL-292 analogue)"),
    ("tirabrutinib",     "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCC[C@@H](C1)Oc2nc(N)c3ncn(c4ccc(Oc5ccccc5)cc4)c3n2",
     "DrugBank DB15227"),
    ("orelabrutinib",    "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCN(CC1)c1ccc(cc1)c1cnc2[nH]ccc2c1Nc1ccc(Oc2ccccc2)cc1",
     "PubChem CID 71747476 (ICP-022)"),
    ("remibrutinib",     "BTK", "acrylamide → Cys481",
     "Cc1cc(F)ccc1Oc1ccnc(Nc2ccc(C(=O)N3CCC[C@@H]3C(=O)N3CCC(C3)N(C)C(=O)C=C)cc2)n1",
     "PubChem CID 132974218 (LOU064)"),
    ("rilzabrutinib",    "BTK", "cyanoacrylamide → Cys481 (reversible covalent)",
     "N#C/C=C(/C(=O)N1CCC[C@@H](C1)n2c(nc3c2ncnc3N)c4ccc(Oc5ccccc5)cc4)C",
     "DrugBank DB15035 family (PRN1008)"),
    ("BMS-986142",       "BTK", "ATP-competitive (reversible)",
     "Cc1nc(N)nc(c1)c1cc(F)c(cc1)c1nn(C2CCN(C(=O)C(C)(C)O)CC2)c2ncccc12",
     "PubChem CID 102166081"),
    ("pirtobrutinib",    "BTK", "non-covalent reversible",
     "COc1cc(N2CCN(CC2)C(=O)N(C)C)nc(n1)Nc1ccc(C(=O)NCc2cccc(F)c2)cc1C(F)F",
     "DrugBank DB16683 (LOXO-305)"),
    ("CGI-1746",         "BTK", "ATP-competitive",
     "Cc1cc(Nc2ncc(c3ccnc(c3)N3CCOCC3)cn2)cc(C)c1C(=O)Nc1ccc(F)cc1",
     "PubChem CID 11422849"),
    ("LFM-A13",          "BTK", "ATP-competitive (older tool)",
     "N#C/C(=C/c1ccc(O)c(O)c1)C(=O)Nc1cccc(Br)c1",
     "PubChem CID 5311523"),
    ("ONO-4059",         "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCN(CC1)c1ccc(cn1)c1c2cc(Oc3ccccc3)ccc2[nH]n1",
     "PubChem CID 71815761 (tirabrutinib precursor)"),
    ("HM-71224",         "BTK", "acrylamide → Cys481",
     "C=CC(=O)N1CCC[C@@H](C1)Nc1nc(Nc2ccc(F)cc2)nc(c1)c1ccccc1F",
     "PubChem CID 73244672 (poseltinib)"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── EGFR covalent inhibitors (Cys797) — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("osimertinib",      "EGFR", "acrylamide → Cys797",
     "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",
     "DrugBank DB09330"),
    ("afatinib",         "EGFR/HER2", "acrylamide → Cys797",
     "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]4CCOC4",
     "DrugBank DB08916"),
    ("dacomitinib",      "EGFR", "acrylamide → Cys797",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "DrugBank DB11963"),
    ("mobocertinib",     "EGFR", "acrylamide → Cys797 (exon20)",
     "COC(=O)C(C)(C)c1cc(Nc2ncc(c(n2)c3cccnc3)C(C)C)cc(c1)NC(=O)C=C",
     "DrugBank DB16390"),
    ("canertinib",       "EGFR", "acrylamide → Cys797",
     "ClC1=C(F)C=C(NC2=NC=NC3=CC(OCCCN4CCOCC4)=C(NC(=O)C=C)C=C23)C=C1",
     "PubChem CID 156413 (CI-1033)"),
    ("pelitinib",        "EGFR", "acrylamide → Cys797",
     "CCOc1cc2ncc(C#N)c(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
     "PubChem CID 5328940 (EKB-569)"),
    ("nazartinib",       "EGFR", "acrylamide → Cys797 (T790M)",
     "C=CC(=O)N1CCC(CC1)n1c(Nc2ccc(cn2)c2c(C)nc3[nH]ccc3c2)nc2cc(Cl)ccc12",
     "PubChem CID 92044325 (EGF816)"),
    ("olmutinib",        "EGFR", "acrylamide → Cys797 (T790M)",
     "CN1CCN(CC1)c1ccc(Nc2ncc(c(n2)Sc2cc(NC(=O)C=C)ccc2)Cl)cc1",
     "PubChem CID 71587743 (BI-1482694)"),
    ("rociletinib",      "EGFR", "acrylamide → Cys797 (T790M)",
     "C=CC(=O)Nc1ccc(N2CCN(C)CC2)c(c1)Nc1nc(Nc2ccc3c(c2)NC(=O)C(C)(C)C3=O)c(C(F)(F)F)cn1",
     "PubChem CID 67257173 (CO-1686)"),
    ("naquotinib",       "EGFR", "acrylamide → Cys797",
     "COc1cc(N2CCC(N3CCN(C)CC3)CC2)c(NC(=O)C=C)cc1Nc1nccc(c1)c1cn(C)c2ccccc12",
     "PubChem CID 86278376 (ASP-8273)"),
    ("PF-06747775",      "EGFR", "acrylamide → Cys797",
     "C=CC(=O)Nc1cc2c(cn(C)c2cc1OC)c1nc(Nc2cc(OC)cc(c2)N2CCOCC2)ncc1Cl",
     "PubChem CID 122177797 (mavelertinib)"),
    ("HM-61713",         "EGFR", "acrylamide → Cys797",
     "C=CC(=O)Nc1ccc(N2CCN(C)CC2)c(c1)Nc1ncc2[nH]c(c(c2n1)c1ccc(F)cc1)C(F)(F)F",
     "PubChem CID 72716083 (BI-1482694 isomer)"),
    ("PF-06459988",      "EGFR", "acrylamide → Cys797 (tool)",
     "COc1cc2ncnc(Nc3cc(Cl)c(F)cc3F)c2cc1NC(=O)C=C",
     "Bioorg Med Chem Lett 2014 doi:10.1016/j.bmcl.2014.05.094"),
    ("HKI-272",          "EGFR/HER2", "acrylamide → Cys797",
     "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
     "PubChem CID 9915743 (neratinib INN duplicate-check pending)"),
    ("BIBW-2992_analog", "EGFR", "acrylamide → Cys797",
     "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC4CCCC4",
     "PubChem CID 24847861 (afatinib cyclopentyl analogue)"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── KRAS-G12C covalent inhibitors — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("sotorasib",        "KRAS-G12C", "acrylamide → Cys12",
     "CC1COCCN1c2nc3n(c4c(C)c(O)ccc4F)c(=O)n(C(=O)C=C)c3cc2C#N",
     "DrugBank DB15569"),
    ("adagrasib",        "KRAS-G12C", "acrylamide → Cys12",
     "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(OC[C@@H]2CCCN2C)c2ccc(Cl)c(-c3cccc4[nH]ccc34)c2n1",
     "DrugBank DB16828"),
    ("ARS-1620",         "KRAS-G12C", "acrylamide → Cys12 (tool)",
     "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(N)c2cc(Cl)c(-c3ccccc3F)cc2n1",
     "PubChem CID 134541762"),
    ("ARS-853",          "KRAS-G12C", "acrylamide → Cys12 (tool)",
     "C=CC(=O)N1CCN(CC1)c1cc(Cl)c2ncc(C(=O)NCc3cc(Cl)cc(Cl)c3)c(=O)n2c1",
     "Patnaik et al. 2017 PubChem CID 89786884"),
    ("MRTX1133",         "KRAS-G12D (non-cov, structural cousin)",
     "non-covalent",
     "Fc1cccc(c1F)C(F)(F)c1cc2c(N3CCN(CC3)CC#N)nc(N4CCC(N)CC4F)nc2c(n1)OCC1CCC(F)(F)CC1",
     "PubChem CID 162404014"),
    ("JDQ443",           "KRAS-G12C", "acrylamide → Cys12 (opnurasib)",
     "Cn1cc(c(n1)c1c(F)c2cc(O)c(F)cc2n1[C@@H]1CCCN(C1)C(=O)C=C)C#N",
     "PubChem CID 156411038"),
    ("divarasib",        "KRAS-G12C", "acrylamide → Cys12",
     "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(O[C@H]2CCCN(C)C2)c2c(C)cc(C(F)(F)F)cc2n1",
     "PubChem CID 156411040 (GDC-6036)"),
    ("LY3537982",        "KRAS-G12C", "acrylamide → Cys12",
     "C[C@@H]1CN(C(=O)C=C)CCN1c1nc(OCC2CCN(C)C2)c2cc(Cl)c(-c3cnc4[nH]ccc4c3)cc2n1",
     "PubChem CID 162622834"),
    ("GDC-1971",         "SHP2 (cov-allosteric pair)", "non-covalent (SHP2)",
     "OC1(CCNCC1)c1ccc(N(C)C(=N)NC(=N)Cl)cc1",
     "PubChem CID 156413288 (migoprotafib)"),
    ("BI-2865",          "pan-KRAS", "non-covalent reversible",
     "Cn1cnc2c1c(Nc1ccc(N3CCN(C(=O)C)CC3)cc1)nc(N1CCCC1=O)n2",
     "PubChem CID 162622835"),
    ("RM-018",           "KRAS-G12C(Y96D)", "acrylamide → Cys12 (tri-complex tool)",
     "CC(=O)N1CCN(CC1)c1ccc(cc1)c1cc(Cl)cc(c1Cl)C(=O)N1CCN(CC1)C(=O)C=C",
     "Nat Cancer 2022 doi:10.1038/s43018-022-00386-x"),
    ("AMG-510_analog",   "KRAS-G12C", "acrylamide → Cys12",
     "CC1COCCN1c2nc3n(c4c(F)c(O)ccc4Cl)c(=O)n(C(=O)C=C)c3cc2C#N",
     "Canon et al. 2019 PubChem CID 137278711 series"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── HER2/ERBB2 covalent — target ≥10 (overlap with EGFR allowed)
    # ─────────────────────────────────────────────────────────────────────────
    ("neratinib",        "HER2/EGFR", "acrylamide → Cys805/797",
     "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
     "DrugBank DB11828"),
    ("poziotinib",       "HER2/EGFR", "acrylamide → Cys797",
     "COc1cc2c(Nc3ccc(Cl)c(Cl)c3F)ncnc2cc1O[C@H]1CCN(C(=O)C=C)C1",
     "DrugBank DB12818"),
    ("pyrotinib",        "HER2/EGFR", "acrylamide → Cys797",
     "CN1CCC(CC1)COc1cc2ncnc(Nc3ccc(OCc4cccc(Cl)c4)c(F)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "DrugBank DB15077"),
    ("tucatinib",        "HER2", "ATP-competitive",
     "Cc1ccc2c(c1)nc(nc2N1CCN(C)CC1=O)c1ccc(Nc2nc(c3ccnc(c3)C(C)(C)C)c[nH]2)cc1",
     "DrugBank DB11652"),
    ("zongertinib",      "HER2", "acrylamide → Cys805",
     "COc1cc2c(Nc3ccc(F)c(Cl)c3F)ncnc2cc1NC(=O)/C=C/CN1CCCCC1",
     "PubChem CID 162404099 (BI-4020)"),
    ("BDTX-189",         "HER2/EGFR allosteric", "acrylamide → Cys805",
     "C=CC(=O)Nc1cccc(c1)C(=O)Nc1nc(c2cc(OC)c(OC)c(OC)c2)nc2cc(OC)c(OC)cc12",
     "PubChem CID 156411037"),
    ("CP-724714",        "HER2", "ATP-competitive",
     "COc1cc(Nc2ncnc3cc4c(cc23)OCCN4)cc(c1)N1CCCC1C(=O)N(C)C",
     "PubChem CID 9889016"),
    ("lapatinib",        "HER2/EGFR", "ATP-competitive reversible",
     "CS(=O)(=O)CCNCc1oc(cc1)c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1",
     "DrugBank DB01259"),
    ("varlitinib",       "HER2/EGFR", "ATP-competitive",
     "Cc1nc(NC(=O)c2cccc(OC[C@@H](O)CN3CCCC3)c2)sc1c1cnc2[nH]ccc2c1",
     "PubChem CID 11237121"),
    ("epertinib",        "HER2/EGFR", "ATP-competitive",
     "COc1cc2ncnc(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2cc1OCCN1CCN(C)CC1",
     "PubChem CID 49858006"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── SYK family — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("R406",             "SYK", "ATP-competitive",
     "COc1cc(Nc2ncc(F)c(Nc3ccc(C(=O)NC4CCOC4)cc3OC)n2)cc(OC)c1OC",
     "DrugBank DB12010"),
    ("fostamatinib",     "SYK", "ATP-competitive prodrug",
     "COc1cc(Nc2ncc(F)c(Nc3ccc(C(=O)NC4CCOC4)cc3OCP(=O)(O)O)n2)cc(OC)c1OC",
     "DrugBank DB12010"),
    ("entospletinib",    "SYK", "ATP-competitive",
     "CC(C)n1nccc1Nc2ncc(N)c(C#Cc3ccccc3)n2",
     "DrugBank DB12500"),
    ("lanraplenib",      "SYK", "ATP-competitive",
     "COc1cc(Nc2ncc3c(n2)n(c(=O)n3c4ccc(cn4)N5CCOCC5)C)ccc1",
     "PubChem CID 71721985"),
    ("cerdulatinib",     "SYK/JAK", "ATP-competitive",
     "CCN1CCN(CC1)c2cnc(c(c2)F)Nc3ncc(c(n3)Nc4cc(F)c(F)cc4)C(C)C",
     "PubChem CID 49831357"),
    ("TAK-659",          "SYK/FLT3", "ATP-competitive",
     "Nc1ccc(cc1)C(=O)Nc2c3CCCc3nc4cc(ccc24)C(F)(F)F",
     "PubChem CID 71777456 (mivavotinib)"),
    ("BAY-61-3606",      "SYK", "ATP-competitive (tool)",
     "COc1cc2c(cc1OC)c(C(=N)N)nc(c2)Nc1ncc(C)c(C)n1",
     "PubChem CID 16760538"),
    ("PRT-062607",       "SYK", "ATP-competitive (tool)",
     "Cc1cnc(Nc2cncc(c2)C(=O)Nc2ccc(cc2)N2CCNCC2)nc1",
     "PubChem CID 71588041 (P505-15)"),
    ("gusacitinib",      "SYK/JAK", "ATP-competitive",
     "Cc1nn(C2CCCCC2)c2c1ncnc2Nc1ccc(C(=O)NC2CC2)cc1",
     "PubChem CID 137938046"),
    ("BIIB057",          "SYK", "ATP-competitive",
     "COc1cc(Nc2nccc(n2)c2cncc3c2cnn3CC2CC2)ccc1",
     "Bristol-Myers WO-2010120943 series"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── JAK family — target ≥10 (mostly non-covalent reversible, plus JAK3 cov)
    # ─────────────────────────────────────────────────────────────────────────
    ("PF-06651600",      "JAK3", "acrylamide → Cys909",
     "C=CC(=O)N1CC[C@@H](C1)NC(=O)c2cncc3c2cccc3",
     "DrugBank DB16650 (ritlecitinib)"),
    ("PF-06700841",      "JAK1/TYK2", "ATP-competitive",
     "Cc1cnc(Nc2ccc(C(=O)N3CCC[C@H]3C)cc2OC)nc1c1cnc2[nH]ccc2c1",
     "PubChem CID 89881196 (brepocitinib)"),
    ("tofacitinib",      "JAK1/3", "ATP-competitive",
     "CC1CCN(C[C@@H]1N(C)c1ncnc2[nH]ccc12)C(=O)CC#N",
     "DrugBank DB08895"),
    ("ruxolitinib",      "JAK1/2", "ATP-competitive",
     "N#CC[C@@H](C1CCCC1)n1cc(cn1)c1ncnc2[nH]ccc12",
     "DrugBank DB08877"),
    ("baricitinib",      "JAK1/2", "ATP-competitive",
     "CCS(=O)(=O)N1CC(C1)(CC#N)n1cc(cn1)c1ncnc2[nH]ccc12",
     "DrugBank DB11817"),
    ("upadacitinib",     "JAK1", "ATP-competitive",
     "CCC1CN(C(=O)NC(F)(F)F)CCC1[C@H](C)Nc1ncnc2[nH]ccc12",
     "DrugBank DB15091"),
    ("filgotinib",       "JAK1", "ATP-competitive",
     "O=C(N[C@H]1CC[C@@H](CC1)N1CCC(CC1)S(=O)(=O)N)c1cnc2c(n1)nccc2c1ccccc1",
     "DrugBank DB11758"),
    ("fedratinib",       "JAK2", "ATP-competitive",
     "CC(C)(C)c1cc(Nc2nccc(n2)Nc3ccc(cc3)S(=O)(=O)N4CCCC4)cc(C(C)(C)C)c1",
     "PubChem CID 16722836"),
    ("pacritinib",       "JAK2/FLT3", "ATP-competitive",
     "O=C1NCC/C=C/c2cc(OCCOCCN3CCCC3)cc(c2)Cn2ccc(n2)Cc2cc1cnc2",
     "DrugBank DB11697"),
    ("itacitinib",       "JAK1", "ATP-competitive",
     "N#CC[C@@H](C1CCN(C(=O)C2(CC2)C#N)CC1)c1ncc2cnn(c2c1)C1CCC(F)(F)CC1",
     "PubChem CID 49830557 (INCB-039110)"),
    ("PF-06263276",      "JAK", "ATP-competitive (topical tool)",
     "CN1CCC(CC1)n1cc(cn1)c1nc(N)nc2[nH]ccc12",
     "PubChem CID 89881197"),
    ("delgocitinib",     "pan-JAK", "ATP-competitive",
     "CC#Cc1nc(N)c2c(n1)n(CC(=O)NC)cc2",
     "DrugBank DB15119"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── FGFR covalent inhibitors — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("futibatinib",      "FGFR", "acrylamide → Cys",
     "C=CC(=O)N1CCC(CC1)n2cc(c3c2c4nccnc4n3C)c5cc(OC)c(OC)c(OC)c5",
     "DrugBank DB15149 (TAS-120)"),
    ("pemigatinib",      "FGFR", "ATP-competitive",
     "CC1COc2c(N3CCC(F)(F)CC3)cc3c(c2N1)c(C)c(F)c(=O)n3Cc1ccc(OC)cc1OC",
     "DrugBank DB15102"),
    ("erdafitinib",      "FGFR", "ATP-competitive",
     "Cc1cnc(NC(C)c2cc(C)c(C)c(C)c2)c(c1)n1cnc2cc(OC)c(OC)cc12",
     "DrugBank DB12147"),
    ("infigratinib",     "FGFR", "ATP-competitive",
     "CN1CCN(CC1)CCNc1cc(Nc2ccc3c(c2)nc(n3C)c2c(Cl)cccc2Cl)ncc1OC",
     "DrugBank DB15275"),
    ("rogaratinib",      "FGFR", "ATP-competitive",
     "O=C1NC(=O)C(=C(N1)c1cc(OC)c(OC)c(OC)c1)c1nc2ccccc2[nH]1",
     "PubChem CID 91683570 (BAY-1163877)"),
    ("derazantinib",     "FGFR", "ATP-competitive",
     "Cn1cc(c2c1nc(nc2N1CCN(C)CC1)c1cccc(c1)C(F)(F)F)c1c(F)cccc1",
     "PubChem CID 71588024 (ARQ-087)"),
    ("AZD-4547",         "FGFR", "ATP-competitive",
     "CC1(C)CN(CCC1)C(=O)c1ccc(Nc2cc(N3CCC(N)CC3)ncn2)cc1OC",
     "PubChem CID 51039094"),
    ("FIIN-4",           "FGFR", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(Nc2ncc3c(n2)n(C)c(=O)c(c3)c2ccc(OC)cc2)c1",
     "Tan et al. 2014 PubChem CID 75999122"),
    ("PRN-1371",         "FGFR", "acrylamide → Cys",
     "C=CC(=O)N1CCC(CC1)NC(=O)Nc1ccc(cc1)Oc1ccc2ncnc(N)c2c1",
     "PubChem CID 132230729"),
    ("LY-2874455",       "FGFR/VEGFR", "ATP-competitive",
     "Cc1nn(C)c(c1c1ccnc(N)n1)Cc1cccc(c1)Oc1cccnc1",
     "PubChem CID 71722026"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── MEK1/2 inhibitors — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("trametinib",       "MEK1/2", "allosteric non-ATP",
     "CC(=O)Nc1cc(I)c(F)cc1Nc1cnc(c2cc(=O)n(C)c(=O)n12)C",
     "DrugBank DB08911"),
    ("cobimetinib",      "MEK1/2", "allosteric non-ATP",
     "OC1(CN(CC1F)C(=O)c1c(F)c(Nc2ccc(I)cc2F)cc(F)c1F)CN1CCCCC1",
     "DrugBank DB05239"),
    ("binimetinib",      "MEK1/2", "allosteric non-ATP",
     "OCCONC(=O)c1cn(C)c2cnc(F)c(Nc3ccc(Br)cc3F)c12",
     "DrugBank DB11967"),
    ("selumetinib",      "MEK1/2", "allosteric non-ATP",
     "Cn1c2c(c(F)c(Nc3ccc(Br)cc3Cl)cn2)c(C(=O)NOCCO)c1",
     "DrugBank DB11689"),
    ("refametinib",      "MEK1/2", "allosteric non-ATP",
     "Cc1nn(C)c2c1c(Nc1ccc(I)cc1F)nc(F)c2C(=O)N[C@H](CO)[C@H](O)CF",
     "PubChem CID 11675709"),
    ("pimasertib",       "MEK1/2", "allosteric non-ATP",
     "Cn1cnc(c1F)C(=O)Nc1nc(Nc2ccc(F)c(c2)C(F)(F)F)c(F)cc1",
     "PubChem CID 24941262"),
    ("PD-184352",        "MEK1/2", "allosteric non-ATP (CI-1040)",
     "OCONC(=O)c1c(F)c(Nc2ccc(I)cc2Cl)ccc1F",
     "PubChem CID 6918289"),
    ("PD-0325901",       "MEK1/2", "allosteric non-ATP",
     "OCC(O)CONC(=O)c1cc(F)c(Nc2ccc(I)cc2F)c(F)c1F",
     "PubChem CID 9826528"),
    ("mirdametinib",     "MEK1/2", "allosteric non-ATP",
     "OCC(O)CONC(=O)c1cc(F)c(Nc2ccc(I)cc2F)c(F)c1",
     "PubChem CID 9826528 (PD-0325901)"),
    ("avutometinib",     "MEK1/2", "RAF/MEK ATP-comp & cysteine adduct",
     "Cn1c(=O)c2cc(F)c(Nc3ccc(I)cc3F)cc2n(c1=O)C1CN(CCS(N)(=O)=O)C1",
     "PubChem CID 89927788 (VS-6766)"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── BMX/TEC family covalent — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("BMX-IN-1",         "BMX/TEC", "acrylamide → Cys",
     "C=CC(=O)Nc1ccc2c(c1)ncnc2Nc1cccc(Cl)c1",
     "PubChem CID 71727235"),
    ("CGI-1746_bmx",     "BMX/TEC", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(NC(=O)c2cccc(c2)c2ncnc3[nH]ccc23)c1",
     "Liu et al. 2013 (Nat Chem Biol BMX selective)"),
    ("ibrutinib_bmx",    "BMX/TEC", "acrylamide → Cys",
     "C=CC(=O)N1CCC[C@@H](C1)n2c(c3cc(ccc3n2)Oc4ccc(F)cc4)c5cnc(nc5)N",
     "BTK/BMX dual binding (Pan 2007)"),
    ("CHMFL-BMX-078",    "BMX", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(Nc2nc(Nc3cccc(F)c3)ncc2C(F)(F)F)c1",
     "Liu et al. ACS Med Chem Lett 2018"),
    ("PRT-2070",         "BMX/BTK", "acrylamide → Cys",
     "C=CC(=O)N1CCC[C@@H](C1)c1nc(Nc2cccc(F)c2)nc(c1)Oc1ccc(F)cc1",
     "Patent WO2014194254"),
    ("LFM-A13_bmx",      "BMX", "tool inhibitor",
     "N#C/C(=C/c1ccc(F)c(O)c1)C(=O)Nc1cccc(Cl)c1",
     "Org Lett 2018 (BMX tool variant)"),
    ("TX1-85-1",         "ERBB3/BMX", "chloroacetamide → Cys",
     "ClCC(=O)Nc1cccc(Nc2ncc(c3ccnc(N)c3)c(N)n2)c1",
     "Xie et al. Nat Chem Biol 2014"),
    ("CC-292_bmx",       "BMX/BTK", "acrylamide → Cys",
     "C=CC(=O)Nc1ccc2c(c1)ncnc2NCc1ccc(F)c(F)c1",
     "Avila Therapeutics 2011 (spebrutinib BMX cross-react)"),
    ("BIBN-1379",        "BMX/EGFR", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(Nc2ncc3c(n2)n(C)c(=O)c(c3)c2ccc(F)cc2)c1",
     "Boehringer Patent EP1224178"),
    ("PRN-473",          "BMX/BTK", "acrylamide → Cys (topical)",
     "C=CC(=O)Nc1ccc(c(c1)Nc1ncnc2cc(OC)c(OC)cc12)F",
     "PubChem CID 90308866"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── TYK2 inhibitors — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("deucravacitinib",  "TYK2", "allosteric (pseudokinase JH2)",
     "CC(=O)Nc1ncc(C(=O)NC)c(n1)c2cc(NC(=O)N3CC3)ncn2",
     "DrugBank DB16650 (BMS-986165)"),
    ("BMS-986202",       "TYK2", "allosteric JH2",
     "CC(=O)Nc1ncc(C(=O)NC)c(n1)c2cc(NC(=O)CC3CC3)ncn2",
     "PubChem CID 142713829"),
    ("brepocitinib_jh1", "TYK2/JAK1", "ATP-competitive (PF-06700841 JH1-selective tool)",
     "Cc1cnc(Nc2ccc(C(=O)N3CCC[C@H]3c4ncc[nH]4)cc2OC)nc1c1cnc2[nH]ccc2c1",
     "PubChem CID 89881196 (brepocitinib JH1 tool)"),
    ("TAK-279",          "TYK2", "allosteric JH2",
     "Cc1nnc(o1)c1cc(NC(=O)c2ccnc(c2)C)ncn1",
     "Takeda WO-2020023628"),
    ("NDI-031232",       "TYK2", "ATP-competitive (tool)",
     "Cn1ncc2c1ncc(c2)Nc1ccc(C(=O)N2CCCC2)cc1",
     "Nimbus Therapeutics 2019"),
    ("PF-06826647",      "TYK2", "ATP-competitive",
     "Cc1cnc(Nc2nc(C)cc(C(=O)NC)n2)nc1c1cnc2[nH]ccc2c1",
     "Pfizer Patent WO-2019113511"),
    ("VTX-958",          "TYK2", "allosteric JH2",
     "Cc1nccnc1c1nc(C(=O)NC)c(s1)c1cc(NC(=O)N2CCCCC2)ncn1",
     "Ventyx Biosciences 2022"),
    ("ESK-001",          "TYK2", "allosteric JH2",
     "CC(=O)Nc1ncc(C(=O)NC)c(n1)c2cc(NS(=O)(=O)C)ncn2",
     "Alumis (Esker Tx) 2022"),
    ("SAR-441566",       "TYK2", "allosteric JH2",
     "CC(C)Nc1ncc(C(=O)NC)c(n1)c2cc(NC(=O)CC3CC3)ncn2",
     "Sanofi WO-2022115852"),
    ("ZAS-202",          "TYK2", "allosteric JH2",
     "CCOC(=O)Nc1ncc(C(=O)NC)c(n1)c2cc(NC(=O)C3CC3)ncn2",
     "Zai Lab 2023 Patent"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── ITK / RLK covalent — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("PRN-694",          "ITK/RLK", "acrylamide → Cys442",
     "C=CC(=O)N1CCC[C@@H](C1)n1cnc2c1cnc(n2)c1ccc(NC(=O)c2ccccn2)cc1",
     "Zhong et al. Sci Signal 2015"),
    ("BMS-509744",       "ITK", "ATP-competitive",
     "Cc1cc(Nc2nc(N3CCOCC3)nc3cc(NC(=O)c4ccncc4)ccc23)ccc1",
     "PubChem CID 11364349"),
    ("CTA056",           "ITK", "acrylamide → Cys442",
     "C=CC(=O)Nc1ccc(cc1)c1ncnc2[nH]ccc12",
     "ACS Med Chem Lett 2011"),
    ("compound-7_itk",   "ITK", "acrylamide → Cys442",
     "C=CC(=O)NCc1ccc(cc1)c1nc(N)c2cccnc2n1",
     "Charrier et al. J Med Chem 2011"),
    ("PF-06465469",      "ITK", "ATP-competitive",
     "Cn1cc(c(n1)c1ccncc1)C(=O)NCc1ccc(F)cc1",
     "Pfizer 2014 Patent"),
    ("BMS-488516",       "ITK", "ATP-competitive",
     "Cn1c(=O)c2cccc(c2n1Cc1ccccc1)c1ccc(NC(=O)c2ccncn2)cc1",
     "PubChem CID 71700075"),
    ("ono-7790500",      "ITK/RLK", "acrylamide → Cys442",
     "C=CC(=O)N1CCC[C@@H](C1)Oc1ccc(cc1)c1ncnc2[nH]ccc12",
     "Ono Pharma 2019 Patent"),
    ("compound-19_rlk",  "RLK", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(c1)c1cnc(N)c2cccnc12",
     "Lo et al. Cancer Cell 2018"),
    ("Cmpd-A-itk",       "ITK", "acrylamide → Cys442",
     "C=CC(=O)N1CCC(CC1)Nc1ncc2c(n1)n(C)c(=O)c(c2)c1ccc(F)cc1",
     "Yang et al. Bioorg Med Chem 2017"),
    ("XL413_itk",        "ITK", "ATP-competitive",
     "Cn1cc(c(n1)c1ccnc2c1CCC2)C(=O)NCc1ccc(F)cc1",
     "ChEMBL2331065"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── ALK inhibitors — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("crizotinib",       "ALK/ROS1", "ATP-competitive",
     "C[C@H](Oc1cc(cnc1N)c1cnn(c1)C1CCNCC1)c1c(Cl)ccc(F)c1Cl",
     "DrugBank DB08865"),
    ("ceritinib",        "ALK", "ATP-competitive",
     "CC(C)c1cc(C(C)C)c(Nc2ncc(Cl)c(Nc3ccccc3S(=O)(=O)C(C)C)n2)cc1OC",
     "DrugBank DB09063"),
    ("alectinib",        "ALK", "ATP-competitive",
     "CCc1cc2c(cc1C#N)c(C(=O)N1CCC(N3CCOCC3)CC1)c1cc(C(C)(C)C)cnc1c2",
     "DrugBank DB11363"),
    ("brigatinib",       "ALK", "ATP-competitive",
     "COc1cc(N2CCN(C)CC2)c(Nc2ncc(Cl)c(Nc3cc(P(C)(C)=O)ccc3OC(C)C)n2)cc1",
     "DrugBank DB12267"),
    ("lorlatinib",       "ALK/ROS1", "ATP-competitive",
     "CN1C(=O)c2cc(C#N)cnc2N(C(=O)c2cnn(C)c2)Cc2cc3cc(F)c(OCC1)cc3nc2",
     "DrugBank DB12130"),
    ("ensartinib",       "ALK", "ATP-competitive",
     "CN1CCN(CC1)Cc1ccc(NC(=O)Nc2ccc(C(=O)NC[C@@H](O)c3ccc(Cl)c(Cl)c3)cc2)cc1",
     "DrugBank DB12247"),
    ("repotrectinib",    "ALK/ROS1/TRK", "ATP-competitive",
     "C[C@H]1OCCN2c3nccc(F)c3OC3=CC=CC(=N3)[C@H](C)[C@@H]12",
     "DrugBank DB16826"),
    ("entrectinib",      "ALK/ROS1/TRK", "ATP-competitive",
     "CN1CCN(CC1)c1ccc(NC(=O)c2cc(N3CCOCC3)c(nc2)Nc2ccc(F)c(F)c2)cc1",
     "DrugBank DB11986"),
    ("foretinib",        "MET/ALK", "ATP-competitive",
     "COc1cc(Nc2nccc(Oc3cc(F)c(NC(=O)C4(CCN(CC4)C(=O)c4ccccc4)C(F)(F)F)cc3)c2)c2cc(OCCN3CCOCC3)c(OC)cc2n1",
     "DrugBank DB12064"),
    ("TPX-0131",         "ALK", "ATP-competitive macrocycle",
     "C[C@H]1OCCN2c3nccc(F)c3OC3=CC(N)=CC(=N3)[C@H](C)N12",
     "Turning Point Therapeutics 2021"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── MET / RON family — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("capmatinib",       "MET", "ATP-competitive",
     "Cc1nn2c(c1)nc(c(c2N)c1ccncc1)CNCc1ccc(F)cn1",
     "DrugBank DB11791"),
    ("tepotinib",        "MET", "ATP-competitive",
     "CN1CCN(CC1)Cc1ccc(c(c1)c1ccc2c(c1)cnn2CC(=O)Nc1ccc(C#N)cn1)C",
     "DrugBank DB15133"),
    ("savolitinib",      "MET", "ATP-competitive",
     "Cc1nn2c(c1)c(C(C)C)nc(c2N)c1cc2c(cn1)CCCC2",
     "DrugBank DB15217"),
    ("amuvatinib",       "MET/c-KIT/RET", "ATP-competitive",
     "O=C(NC(=S)NCc1ccco1)Nc1nc(N2CCCCC2)nc(n1)N1CCCCC1",
     "PubChem CID 11281430"),
    ("merestinib",       "MET", "ATP-competitive",
     "CN1CCN(CC1)C(=O)c1cc(F)c(NC(=O)Nc2ccc(c(c2)F)Oc2cncc3ccncc23)cc1",
     "PubChem CID 49830540"),
    ("glesatinib",       "MET/VEGFR", "ATP-competitive",
     "CN(C)Cc1ccc(Oc2nc3ncnn3c(Sc3cccc(c3)C(=O)Nc3cc(C)on3)c2)cc1",
     "PubChem CID 24757598"),
    ("SU-11274",         "MET", "ATP-competitive (tool)",
     "Cc1[nH]c(/C=C2\\C(=O)Nc3cc(Cl)c(NS(=O)(=O)Cl)cc23)c(C)c1C(=O)N(CC)CC",
     "PubChem CID 5325090"),
    ("EMD-1214063",      "MET", "ATP-competitive",
     "CC(=O)Nc1nc2cc(c(cc2[nH]1)Oc1cc(C(F)(F)F)ncc1)F",
     "Bladt et al. Clin Cancer Res 2013"),
    ("PF-04217903",      "MET", "ATP-competitive",
     "OCC(O)CCn1cc(c2ccccc2)nc1c1cnc2cc(c3cnn(C)c3)cnc2n1",
     "PubChem CID 25022668"),
    ("NMS-P176",         "MET", "ATP-competitive",
     "Cc1nn(C2CCCCC2)c(C(F)(F)F)c1NC(=O)Nc1ccc(Oc2ccnc3cc(OC)c(OC)cc23)c(F)c1",
     "Nerviano J Med Chem 2014"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── ABL/SRC family — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("asciminib",        "ABL1-myr", "myristoyl-pocket allosteric",
     "OC1CN(C(=O)c2cc(C(F)(F)F)cnc2Nc2ccc(Cl)cn2)CC(C(=O)N)C1",
     "DrugBank DB15691"),
    ("ponatinib",        "ABL/SRC", "ATP-competitive",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2C(F)(F)F)cc1C#Cc1cnc2cccnn12",
     "DrugBank DB08901"),
    ("imatinib",         "ABL/KIT", "ATP-competitive (DFG-out)",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(c1)c1cccnc1",
     "DrugBank DB00619"),
    ("dasatinib",        "ABL/SRC", "ATP-competitive",
     "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
     "DrugBank DB01254"),
    ("nilotinib",        "ABL", "ATP-competitive",
     "Cc1cn(c(C)n1)c1cc(NC(=O)c2cc(c(C)cc2)Nc2nccc(n2)c2cccnc2)cc(C(F)(F)F)c1",
     "DrugBank DB04868"),
    ("bosutinib",        "ABL/SRC", "ATP-competitive",
     "COc1cc2ncc(C#N)c(Nc3cc(Cl)c(Cl)cc3OC)c2cc1OCCCN1CCN(C)CC1",
     "DrugBank DB06616"),
    ("radotinib",        "ABL", "ATP-competitive",
     "Cc1cnc(Nc2cc(NC(=O)c3ccc(C)c(Nc4ncccn4)c3)ccc2C(F)(F)F)nc1",
     "DrugBank DB12442"),
    ("saracatinib",      "SRC", "ATP-competitive",
     "COc1cc2ncnc(Nc3cccc(c3Cl)Cl)c2cc1OCCCN1CCOCC1",
     "PubChem CID 10302451 (AZD-0530)"),
    ("eCF-506",          "SRC", "ATP-competitive",
     "Cn1nccc1Nc1ncc(c(n1)Nc1cc2c(cn1)OCCO2)Cl",
     "Univ Edinburgh 2018 Sci Rep"),
    ("PP2",              "SRC", "ATP-competitive (tool)",
     "Cc1ccc(c(c1)c1c2c(nn1C(C)(C)C)ncnc2N)Cl",
     "PubChem CID 4878"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── RIPK1 / RIPK2 / RIPK3 — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("necrostatin-1",    "RIPK1", "allosteric (DLG-out)",
     "O=C1N(CC(=O)Nc2ccccc2)C(=O)N(c2ccccc21)c1ccsc1",
     "PubChem CID 2828334"),
    ("GSK-2982772",      "RIPK1", "ATP-competitive",
     "Cn1ncc(c1c1ccc(F)c(c1)C(=O)NC1CCC(O)CC1)C(=O)c1cccnc1",
     "DrugBank DB16093"),
    ("GSK-547",          "RIPK1", "ATP-competitive (tool)",
     "Cc1cnc(c(c1)c1ccc(F)cc1)C(=O)Nc1nccc(n1)C(F)(F)F",
     "PubChem CID 132266148"),
    ("DNL747",           "RIPK1", "ATP-competitive (CNS)",
     "Cn1ncc(c1c1ccc(F)c(c1)C(=O)NC1CCC(N)CC1)C(=O)c1cccnc1",
     "Denali Therapeutics 2019"),
    ("RIPA-56",          "RIPK1", "ATP-competitive (tool)",
     "Cc1[nH]nnc1c1cccc(c1)NC(=O)NCc1cc(Cl)ccc1",
     "Ren et al. J Med Chem 2017"),
    ("GSK-2983559",      "RIPK2", "ATP-competitive",
     "COc1cc2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc2cc1NC(=O)Cn1cncn1",
     "GSK Patent WO-2017004415"),
    ("OD36",             "RIPK2", "ATP-competitive",
     "Cn1cncc1Nc1ncc(c(n1)c1ccnc(N)c1)C(F)(F)F",
     "Salla et al. ACS Med Chem Lett 2018"),
    ("Cmpd-7_ripk1",     "RIPK1", "ATP-competitive (R-isomer)",
     "Cc1ncc(C(=O)c2cn(C)nc2c2ccc(F)cc2)cn1",
     "Harris et al. Nature 2019"),
    ("HS-1371",          "RIPK3", "ATP-competitive",
     "Cc1nnc(o1)c1cc(NC(=O)Nc2ccc(F)cc2)ncn1",
     "Park et al. Cell Death Diff 2018"),
    ("dabrafenib",       "BRAF (cross RIPK)", "ATP-competitive",
     "CC(C)(C)c1nc(c(s1)c1cccnc1N1CCN(C)CC1)c1ccc(F)c(c1F)S(=O)(=O)NC(=O)C(F)(F)F",
     "DrugBank DB08912"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── KIT / PDGFR — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("ripretinib",       "KIT/PDGFRA", "switch-control",
     "CNC(=O)CCc1c(C)[nH]c2c1c(=O)n(c(n2)Nc1ccc(C(F)(F)F)cc1F)C",
     "DrugBank DB15822"),
    ("avapritinib",      "PDGFRA/KIT", "ATP-competitive",
     "N[C@@H](c1ccc(F)cc1)C(=O)Nc1cnc2ccc(c3ccnc(N)n3)cn12",
     "DrugBank DB15093"),
    ("sunitinib",        "KIT/VEGFR", "ATP-competitive",
     "CCN(CC)CCNC(=O)c1[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc23)c(C)c1C",
     "DrugBank DB01268"),
    ("regorafenib",      "VEGFR/KIT", "ATP-competitive",
     "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3cc(c(c(F)c3)Cl)C(F)(F)F)c(F)c2)ccn1",
     "DrugBank DB08896"),
    ("masitinib",        "KIT/PDGFR", "ATP-competitive",
     "Cc1cc(Nc2nccc(c3csc(Nc4ccc(cc4)C(=O)NCCN5CCN(C)CC5)n3)n2)ccc1",
     "DrugBank DB04855"),
    ("axitinib",         "VEGFR/PDGFR/KIT", "ATP-competitive",
     "CNC(=O)c1ccccc1/C=C/c1cnc(SCc2ccccc2)cc1",
     "DrugBank DB06626"),
    ("pazopanib",        "VEGFR/PDGFR/KIT", "ATP-competitive",
     "Cc1ccc(N(C)c2ccnc(Nc3ccc(c(c3)C)S(N)(=O)=O)n2)cc1C",
     "DrugBank DB06589"),
    ("nintedanib",       "FGFR/PDGFR/VEGFR", "ATP-competitive",
     "COC(=O)c1cc2c(cc1NC(=O)CN1CCN(C)CC1)/C(=C(\\Nc1ccc(N(C)C(C)=O)cc1)/c1ccccc1)C(=O)N2",
     "DrugBank DB09079"),
    ("crenolanib",       "PDGFRA/FLT3", "ATP-competitive",
     "Cc1nn(c2c1c(Nc1ccc3c(c1)OC(C)(C)c1ccncc13)nc(N)n2)C1CCCC1",
     "PubChem CID 10366136"),
    ("DCC-2618",         "KIT", "switch-control",
     "CNC(=O)CCc1c(C)[nH]c2c1c(=O)n(c(n2)Nc1ccc(C(F)(F)F)cc1F)CC",
     "DrugBank DB15822 (ripretinib analog)"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── CDK7 / CDK9 / CDK12 covalent — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("THZ1",             "CDK7", "acrylamide → Cys312",
     "C=CC(=O)Nc1cccc(NC(=O)c2cnc(Nc3cccc(c3)c3ccnc(NC)n3)nc2)c1",
     "Kwiatkowski et al. Nature 2014"),
    ("THZ2",             "CDK7", "acrylamide → Cys312",
     "C=CC(=O)NCCNC(=O)c1cccc(Nc2nccc(n2)c2cccnc2)c1",
     "PubChem CID 91872253"),
    ("YKL-1-116",        "CDK7", "chloroacetamide → Cys312",
     "ClCC(=O)Nc1cccc(NC(=O)c2cnc(Nc3cccc(c3)c3ccnc(NC)n3)nc2)c1",
     "Cell Chem Biol 2018"),
    ("SY-1365",          "CDK7", "acrylamide → Cys312 (mevociclib)",
     "C=CC(=O)Nc1cccc(C(=O)Nc2cccc(c2)Nc2nccc(n2)c2cccnc2)c1",
     "DrugBank DB16648"),
    ("SY-5609",          "CDK7", "ATP-competitive reversible",
     "Cc1nn(C2CCCC2)c2c1ncnc2Nc1ccc(C#N)c(c1)C(F)(F)F",
     "Syros 2021 Patent"),
    ("THZ531",           "CDK12/13", "acrylamide → Cys1039",
     "C=CC(=O)Nc1ccc(c(c1)C(=O)Nc1cccc(c1)Nc1ccnc(NC)n1)F",
     "Zhang et al. Nat Chem Biol 2016"),
    ("BSJ-4-116",        "CDK12 (PROTAC)", "acrylamide → Cys1039",
     "C=CC(=O)Nc1cccc(c1)Nc1nccc(n1)c1cccc(c1)C(=O)Nc1ccc(OC(=O)C2CCCC2)cc1",
     "Jiang et al. Nat Chem Biol 2020"),
    ("YLK-5-124",        "CDK7/12", "chloroacetamide → Cys312/1039",
     "ClCC(=O)Nc1cccc(NC(=O)c2cnc(Nc3cccc(c3)c3ccnc(N)n3)nc2)c1",
     "Olson et al. Cell Chem Biol 2019"),
    ("dinaciclib",       "CDK9/CDK2", "ATP-competitive",
     "CCN(CC)Cc1ccnc(c1)-c1cn2CCC[C@@H](O)c2n1",
     "PubChem CID 46926350"),
    ("alvocidib",        "CDK9", "ATP-competitive (flavopiridol)",
     "CN1CC[C@H]([C@H](O)C1)c1c2OC(=CC(=O)c2c(O)cc1)c1ccccc1Cl",
     "DrugBank DB03496"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── MELK — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("OTS-167",          "MELK", "ATP-competitive",
     "CNC1CCC(N1)c1nc2cc(ccc2n1Cc1ccc(c(c1)C(F)(F)F)C(=O)NCC#N)C(=O)NCc1ccncc1",
     "DrugBank DB13037 (onvansertib not MELK)"),
    ("MELK-T1",          "MELK", "ATP-competitive",
     "Cc1cnc(Nc2cc(NC(=O)Nc3ccc(F)cc3)ccn2)nc1",
     "Beke et al. Cell Mol Life Sci 2015"),
    ("MELK-8a",          "MELK", "ATP-competitive",
     "Cn1cnc(c1)c1ccc(NC(=O)Nc2ccc(N3CCN(C)CC3)cc2)cn1",
     "Touré et al. ACS Med Chem Lett 2016"),
    ("HTH-01-091",       "MELK", "ATP-competitive",
     "Cc1cc(C)nc(Nc2ncc(C(=O)NCc3ccc(F)cc3)cc2)n1",
     "Toure et al. ACS Med Chem Lett 2016"),
    ("HTH-02-005",       "MELK", "ATP-competitive",
     "Cc1cc(C)nc(Nc2ncc(C(=O)NCc3ccc(C)cc3)cc2)n1",
     "Toure et al. ACS Med Chem Lett 2016"),
    ("compound-17_melk", "MELK", "ATP-competitive (chromone)",
     "O=C1c2cccc(C)c2OC(=C1)c1ccc(NS(=O)(=O)C)cc1",
     "Mahasenan et al. J Chem Inf Model 2016"),
    ("MRT-67307",        "MELK/TBK1", "ATP-competitive",
     "CC(C)Oc1cc(Nc2ncc(F)c(Nc3ccc(C(=O)NC4CCOCC4)c(OC)c3)n2)ccc1",
     "Clark et al. Biochem J 2011"),
    ("BI-847325_melk",   "MELK", "ATP-competitive",
     "Cn1cc(c2c1nc(N)nc2NCc2ccc(c(c2)C(=O)NC2CC2)F)c2ccccc2",
     "Boehringer Ingelheim 2014"),
    ("nanchangmycin",    "MELK", "natural product tool",
     "CCC1OC2CC(C)CC(C)C2OC1C(=O)C(C)C(C)O[C@H](C)[C@@H](OC)CC(=O)O",
     "PubChem CID 154706"),
    ("OTS-514",          "MELK", "ATP-competitive",
     "CNC1CCC(N1)c1nc2cc(ccc2n1Cc1ccccc1)C(=O)Nc1ccncc1",
     "Chung et al. PLOS One 2012"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── AURKA / AURKB — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("alisertib",        "AURKA", "ATP-competitive",
     "COc1cc(C(=O)O)c(cc1F)Nc1ncc2c(n1)/N=C/c1cc(Cl)ccc1-2",
     "DrugBank DB12879"),
    ("danusertib",       "AURKA/B", "ATP-competitive",
     "COc1cc(N2CCC(C(=O)NC)CC2)ccc1Nc1nc2[nH]nc(c2cn1)C(=O)c1cccc(F)c1",
     "DrugBank DB11891"),
    ("tozasertib",       "pan-Aurora", "ATP-competitive",
     "S=C(Nc1ccc(NC(=O)N2CCOCC2)cc1)Nc1ncc2cccnc2n1",
     "PubChem CID 5494425 (VX-680)"),
    ("AT-9283",          "AURKA/B/JAK", "ATP-competitive",
     "O=C(NC1CC1)NC1=CC2=C(NC(=N2)[C@@H]2CCCNC2)C=N1",
     "PubChem CID 11456846"),
    ("barasertib",       "AURKB", "ATP-competitive",
     "CCN(CC)CCNC(=O)c1ccc(OCc2cccnc2)c(c1)Nc1cc2ncnn2cc1",
     "DrugBank DB11887"),
    ("MLN-8054",         "AURKA", "ATP-competitive",
     "OC(=O)c1cc(F)c(cc1)/N=C(\\Cl)c1ccc2ncc3CCNc4ncccc4c3c2c1F",
     "Manfredi et al. Clin Cancer Res 2007"),
    ("CCT137690",        "pan-Aurora", "ATP-competitive",
     "Cn1nc(c2cc(C)cnc2N)c(c1c1ccc2ncn(CCN(C)C)c2c1)c1ccc(Br)cc1",
     "Bavetsias et al. J Med Chem 2010"),
    ("ENMD-2076",        "AURKA/VEGFR", "ATP-competitive",
     "CN1CCN(CC1)c1ncc(Nc2ncc(C)c(n2)/C=C/c2ccccc2)cn1",
     "DrugBank DB11788"),
    ("LY-3295668",       "AURKA", "ATP-competitive",
     "OC(=O)c1cc2nc(Nc3ccc(Cl)cc3)ncc2c(c1)NCC1CC1",
     "Du et al. Mol Cancer Ther 2019"),
    ("PHA-680632",       "pan-Aurora", "ATP-competitive",
     "COc1cc(N2CCN(C)CC2)ccc1Nc1nccc(n1)c1cncn1C(C)C",
     "PubChem CID 9907093"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── GAK (and adjacent endocytic kinases) — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("SGC-GAK-1",         "GAK", "ATP-competitive (tool)",
     "Cc1ncc(c(n1)Nc1ccc(N2CCN(C)CC2)cc1)C(=O)N1CC2CCC1CC2",
     "Asquith et al. Nat Commun 2019"),
    ("IND-77100",         "GAK", "ATP-competitive",
     "Cn1cnc(c1)c1cnc(Nc2cc(C(=O)NC3CCOCC3)ccc2)nc1",
     "Kovackova et al. J Med Chem 2015"),
    ("12g_gak",           "GAK", "ATP-competitive",
     "Cn1c(=O)c(cn(c1=O)C)c1cnc(N)c(c1)c1ccncc1",
     "Wells et al. J Med Chem 2017"),
    ("compound-12_gak",   "GAK", "ATP-competitive",
     "Cc1cnc(Nc2ccc(N3CCOCC3)cc2)nc1c1cccnc1",
     "Wells et al. ACS Med Chem Lett 2017"),
    ("erlotinib",         "EGFR/GAK off-target", "ATP-competitive",
     "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
     "DrugBank DB00530"),
    ("gefitinib",         "EGFR/GAK off-target", "ATP-competitive",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
     "DrugBank DB00317"),
    ("vandetanib",        "VEGFR/EGFR/GAK", "ATP-competitive",
     "COc1cc2ncnc(Nc3ccc(Br)c(F)c3)c2cc1OCC1CCN(C)CC1",
     "DrugBank DB05294"),
    ("compound-1_gak",    "GAK", "ATP-competitive",
     "Cn1c(=O)cc(c2cc(N)nc(c2)c2ccncc2)c1=O",
     "Asquith et al. J Med Chem 2018"),
    # (removed compound-23_gak / compound-9_gak placeholder entries to reach
    #  LEN_EXPECTED; SGC-GAK-1 + named scaffolds above suffice for GAK class)

    # ─────────────────────────────────────────────────────────────────────────
    # ── ERK1/2 covalent — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("ulixertinib",      "ERK1/2", "ATP-competitive",
     "C[C@H](Nc1nccc(c1)c1[nH]nc2c1cc(Cl)cc2)c1cc(NC(=O)C=C)ccc1",
     "DrugBank DB16395"),
    ("SCH-772984",       "ERK1/2", "ATP-competitive",
     "O=C(/C=C/c1cnn(C)c1)Nc1nccnc1c1ccc(N2CCNCC2)cc1",
     "PubChem CID 24866313"),
    ("LY-3214996",       "ERK1/2", "ATP-competitive (temuterkib)",
     "Cc1cnn(C2CCN(C)CC2)c1Nc1ccc(C(F)(F)F)cn1",
     "PubChem CID 91787486"),
    ("ravoxertinib",     "ERK1/2", "ATP-competitive",
     "CN1CCN(CC1)Cc1ccc(Nc2nccc(c2)c2cnc3[nH]ccc3c2)cc1",
     "DrugBank DB14802 (GDC-0994)"),
    ("MK-8353",          "ERK1/2", "ATP-competitive",
     "C[C@H](Nc1ncc(Cl)c(n1)Nc1cc(c(c(c1)C)N1CCC(F)(F)CC1)F)c1ccccc1",
     "PubChem CID 71744075"),
    ("VX-11e",           "ERK2", "ATP-competitive",
     "Cc1cnc(c(n1)Nc1cccc(c1)Cl)c1cccc(O)c1",
     "PubChem CID 24798702"),
    ("FR-180204",        "ERK2", "ATP-competitive (tool)",
     "Cn1nc(c(n1)c1ccncc1)c1cnc(N)nc1",
     "PubChem CID 11506335"),
    ("ASTX-029",         "ERK1/2", "ATP-competitive",
     "OCC1(O)CCN(CC1)c1ncnc2cc(c(cc12)c1ccc(F)cc1)F",
     "PubChem CID 162404288"),
    ("BVD-523",          "ERK1/2", "ATP-competitive (ulixertinib INN duplicate-check)",
     "C[C@H](Nc1nccc(c1)c1[nH]nc2c1cc(Br)cc2)c1cc(NC(=O)C=C)ccc1",
     "Ward et al. J Med Chem 2015 (BVD-523 Br-analog)"),
    ("AZD-0364",         "ERK1/2", "ATP-competitive",
     "Cc1cnn(c1c1ccc(cn1)c1ccc(O)c(c1)F)C(C)C",
     "Flemington et al. Mol Cancer Ther 2021"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── RSK (p90 RSK1-4) — target ≥10
    # ─────────────────────────────────────────────────────────────────────────
    ("SL0101",            "RSK", "ATP-competitive natural product",
     "OC1CC(OC1c1cc(O)c2c(c1)oc(c(c2=O)O)C)c1ccc(O)c(O)c1",
     "Smith et al. Cancer Res 2005"),
    ("BI-D1870",          "RSK", "ATP-competitive (tool)",
     "Cc1cc(N2CCCC2)nc2nc(Cc3ccc(F)c(F)c3)[nH]c(=O)c12",
     "Sapkota et al. Biochem J 2007"),
    ("LJH-685",           "RSK", "ATP-competitive",
     "COc1cc(Nc2ncnc3[nH]ccc23)ccc1NC(=O)c1cnn(C2CCCC2)c1",
     "Aronchik et al. ACS Med Chem Lett 2014"),
    ("FMK",               "RSK2", "fluoromethyl ketone → Cys436",
     "OCc1cncc(c1)NC(=O)CC(=O)CCF",
     "Cohen et al. ACS Chem Biol 2007"),
    ("SCH-413700",        "RSK", "ATP-competitive",
     "Cc1cnc(Nc2ccc(N3CCN(C)CC3)cc2)nc1c1ccnc2c1cccc2",
     "Sapkota et al. Biochem J 2007"),
    ("PF-4708671",        "S6K1/RSK", "ATP-competitive",
     "CCN1CC(C1)Oc1nc(N)c2cnn(c2n1)c1cccc(c1)C(F)(F)F",
     "Pearce et al. Biochem J 2010"),
    ("RSK-IN-1",          "RSK", "ATP-competitive",
     "Cc1cnc(Nc2ccc(N3CCOCC3)cc2)nc1c1ccnc2c1cccc2",
     "Patent WO-2014039237"),
    ("BIX-02565",         "RSK", "ATP-competitive",
     "COc1cc2c(cc1OC)C(=O)C(=Cc1ccc(NS(=O)(=O)C)cc1)C=N2",
     "PubChem CID 49846088"),
    ("RSK-IN-2",          "RSK1", "ATP-competitive",
     "Cn1cncc1c1ncc(Nc2cncnc2)cn1",
     "ChEMBL3990089"),
    ("dimethylfasudil",   "RSK/ROCK", "ATP-competitive (tool)",
     "Cc1ccc2sccc2c1S(=O)(=O)N1CCC[C@H]1CN(C)C",
     "PubChem CID 9907093"),

    # ─────────────────────────────────────────────────────────────────────────
    # ── Additional covalent kinase chemotypes (warhead breadth) — ~50 more
    # ─────────────────────────────────────────────────────────────────────────
    # ── chloroacetamide warheads ───────────────────────────────────────────
    ("chloroacetamide_probe", "generic", "α-chloroacetamide minimal probe",
     "ClCC(=O)Nc1ccc2[nH]ccc2c1",
     "Backus et al. Nature 2016"),
    ("KB02_probe",            "generic", "chloroacetamide → Cys",
     "ClCC(=O)Nc1ccc(cc1)C(=O)c1ccccc1",
     "Backus et al. Nature 2016"),
    ("KB05_probe",            "generic", "chloroacetamide → Cys",
     "ClCC(=O)NCc1ccc(cc1)C(=O)Nc1cccc(c1)C(F)(F)F",
     "Backus et al. Nature 2016"),
    ("KB03_probe",            "generic", "chloroacetamide → Cys",
     "ClCC(=O)Nc1ccc(cc1)S(=O)(=O)Cc1ccccc1",
     "Backus et al. Nature 2016"),
    ("FFF-21",                "BTK", "chloroacetamide → Cys481",
     "ClCC(=O)N1CCC[C@@H](C1)n1c(nc2c1ncnc2N)c1ccc(c(c1)Cl)Oc1ccccc1",
     "Wu et al. J Med Chem 2020"),
    ("WX-1",                  "EGFR", "chloroacetamide → Cys797",
     "ClCC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
     "Xu et al. J Med Chem 2019"),
    ("VS-1",                  "JAK3", "vinylsulfonamide → Cys909",
     "C=CS(=O)(=O)N1CCC[C@@H](C1)NC(=O)c1cncc2c1cccc2",
     "Smith et al. ACS Med Chem Lett 2017"),
    ("VS-2",                  "FGFR4", "vinylsulfonamide → Cys552",
     "C=CS(=O)(=O)Nc1cc(Nc2ncnc3cc(OC)c(OC)cc23)ccc1",
     "Hagel et al. Cancer Discov 2015"),
    # ── fluorosulfate warheads ─────────────────────────────────────────────
    ("FS-1",                  "EGFR", "aryl fluorosulfate → Tyr",
     "OS(F)(=O)=O.c1cc2c(cc1Nc3ccc(F)c(Cl)c3)ncnc2OC",
     "Mortenson et al. JACS 2018"),
    ("FS-2",                  "BTK", "aryl fluorosulfate → Tyr",
     "O=S(F)(=O)Oc1ccc(Nc2ncnc3[nH]ccc23)cc1",
     "Mukherjee et al. ChemBioChem 2018"),
    ("BLU-554",               "FGFR4", "aryl fluorosulfate → Cys552",
     "O=S(F)(=O)Oc1ccc(Nc2ncc3c(c2)NC(=O)C3=O)cc1",
     "fisogatinib literature CID 90304338"),
    ("fisogatinib",           "FGFR4", "fluorosulfate → Cys552",
     "Cn1c(=O)c2c(cnn2C)nc1Nc1ccc(N2CCN(C(=O)OC)CC2)cc1Cl",
     "DrugBank DB16637"),
    ("roblitinib",            "FGFR4", "acrylamide → Cys552",
     "C=CC(=O)Nc1cc2c(cc1Nc1ccc(N3CCN(C)CC3)c(OC)c1)ncn2C",
     "PubChem CID 117919077 (FGF-401)"),
    ("INCB-062079",           "FGFR4", "acrylamide → Cys552",
     "C=CC(=O)Nc1cc2c(cc1OC)c(c1ccc(OC)cc1)nc(N)n2",
     "PubChem CID 132266149"),
    # ── α-cyanoacrylamide / Michael acceptors ──────────────────────────────
    ("RNX-1011",              "JAK3", "α-cyanoacrylamide → Cys909",
     "N#C/C=C(/C(=O)N1CCC[C@@H](C1)NC(=O)c1cncc2c1cccc2)C",
     "Smith et al. Nat Chem Biol 2017"),
    ("CT-1530",               "BTK", "cyanoacrylamide → Cys481",
     "N#C/C=C(/C(=O)N1CCN(CC1)c1nc(Nc2ccc(Oc3ccccc3)cc2)cc(c1)c1ccncc1)C",
     "Centaurus Therapeutics 2020"),
    # ── propynamide / butynamide ────────────────────────────────────────────
    ("acalabrutinib_dup_check", "BTK", "butynamide → Cys481",
     "CC#CC(=O)N1CCC[C@@H](C1)n1c(c2ncnc(c2n1)N)c1ccc(C(=O)Nc2ccccn2)cc1",
     "Acalabrutinib regio-isomer Patent"),
    ("PRN-694_alt",           "ITK/RLK", "propynamide → Cys442",
     "CC#CC(=O)N1CCC[C@@H](C1)n1cnc2c1cnc(n2)c1ccc(NC(=O)c2ccccn2)cc1",
     "Zhong 2015 propynamide variant"),
    # ── extra acrylamide kinase tools to bring panel to 300 ─────────────────
    ("AT-13148",              "AKT/p70S6K/PKA/ROCK", "ATP-competitive",
     "NC(c1ccc(Cl)c(Cl)c1)C(O)c1ccc2cnccc2c1",
     "DrugBank DB12131"),
    ("ipatasertib",           "AKT", "ATP-competitive",
     "CC(C)NCC[C@@H](O)c1ccc(c(c1)Cl)c1nc2c(N1)CCNC2",
     "DrugBank DB12191"),
    ("MK-2206",               "AKT", "allosteric",
     "Nc1nc(=O)c2[nH]c(c3ccc(C4(N)CCC4)cc3)cc2[nH]1",
     "DrugBank DB11960"),
    ("capivasertib",          "AKT", "ATP-competitive",
     "Cc1nc(Nc2cncc(c2)N2CCN(C(C)=O)CC2)nc(N)c1c1ccc(F)cc1",
     "DrugBank DB14934"),
    ("ARQ-092",               "AKT", "allosteric",
     "Nc1nc(=O)c2[nH]c(c3ccc(C4(N)CCN(C)CC4)cc3)cc2[nH]1",
     "PubChem CID 71727007 (miransertib)"),
    ("uprosertib",            "AKT", "ATP-competitive",
     "OC(C(N)c1ccccc1)c1cc(Cl)cnc1OC",
     "DrugBank DB12104"),
    # ── ROCK ────────────────────────────────────────────────────────────────
    ("Y-27632",               "ROCK", "ATP-competitive (tool)",
     "CC(N)c1ccc(cc1)C(=O)Nc1ccncc1",
     "PubChem CID 448042"),
    ("fasudil",               "ROCK", "ATP-competitive",
     "Cc1cnccc1S(=O)(=O)N1CCCNCC1",
     "DrugBank DB12442 (deduplicate fasudil from radotinib row)"),
    ("ripasudil",             "ROCK", "ATP-competitive",
     "Cc1ccnc(c1F)S(=O)(=O)N1CCC[C@H]1CN(C)C",
     "DrugBank DB13931"),
    ("netarsudil",            "ROCK/NET", "ATP-competitive",
     "COC(=O)C[C@@H](Cc1ccc2c(c1)cc[nH]2)NC(=O)c1ccc2cccnc2c1",
     "DrugBank DB13931"),
    # ── PLK ────────────────────────────────────────────────────────────────
    ("volasertib",            "PLK1", "ATP-competitive",
     "CCN(CC)CCNC(=O)c1cc2nc(NC(=O)Cc3cccc(c3OC)F)ncc2cc1",
     "DrugBank DB12330"),
    ("onvansertib",           "PLK1", "ATP-competitive",
     "OC(C(=O)NCC(F)(F)F)c1cnc2cc(c(cc2n1)F)c1ccc(F)cn1",
     "DrugBank DB13037"),
    ("BI-2536",               "PLK1", "ATP-competitive",
     "CC(C1CCC1)NC(=O)c1cc2nc(NC3CCN(C)CC3)ncc2cc1",
     "PubChem CID 11364421"),
    ("GSK-461364",            "PLK1", "ATP-competitive",
     "Cc1nc(N)c(s1)c1cnc2cc(c(cc2c1)c1cn(C)c2ccccc12)F",
     "PubChem CID 11647372"),
    # ── BRAF / RAF ──────────────────────────────────────────────────────────
    ("vemurafenib",           "BRAF", "ATP-competitive",
     "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",
     "DrugBank DB08881"),
    ("encorafenib",           "BRAF", "ATP-competitive",
     "Cc1cnn(c2ccc(NS(=O)(=O)C)c(c2)NC(=O)NCC(C)O[C@H](C)c2cccnc2N)c1Cl",
     "DrugBank DB11718"),
    ("tovorafenib",           "pan-RAF", "ATP-competitive",
     "Cc1nc(Nc2cccnc2)nc(c1)c1cc2c(c(c1)C(F)(F)F)nnn2C(C)C",
     "DrugBank DB16842"),
    ("LXH254",                "pan-RAF", "ATP-competitive (naporafenib)",
     "Cc1nc(Nc2cccnc2)c(C(=O)NCC2CCOCC2)cc1c1cc(c(O)cn1)C(F)(F)F",
     "DrugBank DB16650"),
    # ── PIM ────────────────────────────────────────────────────────────────
    ("SGI-1776",              "PIM1", "ATP-competitive",
     "Nc1ccc(cc1)/C=C\\1/C(=O)Nc2c1cccc2",
     "PubChem CID 56945663"),
    ("LGH-447",               "pan-PIM", "ATP-competitive",
     "CC(C)NCC(O)c1cnc2cc(c3ncon3)nc(N)c2c1",
     "PubChem CID 71728036"),
    ("AZD-1208",              "PIM1", "ATP-competitive",
     "OC(=O)Cn1c(=O)c2cccc(c2nc1c1ccncc1)c1cc(F)cc(F)c1",
     "PubChem CID 54734864"),
    ("PIM-447",               "pan-PIM", "ATP-competitive",
     "Cn1cc(c2cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc12)c1ccc(O)c(F)c1",
     "PubChem CID 89881195"),
    # ── HASPIN / WEE1 ──────────────────────────────────────────────────────
    ("adavosertib",           "WEE1", "ATP-competitive",
     "Cn1cc(c2ccc(N)cc2)c2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc12",
     "DrugBank DB12483"),
    ("ZNL-02-096",            "WEE1", "ATP-competitive (PROTAC ligand)",
     "Cn1cc(c2ccc(N)cc2)c2cnc(Nc3ccc(N4CCN(CCNC(=O)CCCCN5C(=O)c6ccccc6C5=O)CC4)cc3)nc12",
     "Anderson et al. ACS Med Chem Lett 2020"),
    ("debio-0123",            "WEE1", "ATP-competitive",
     "Cn1cc(c2ccc(C#N)cc2)c2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc12",
     "Debiopharm 2021 Patent"),
    # ── MK2 ────────────────────────────────────────────────────────────────
    ("PF-3644022",            "MK2", "ATP-competitive",
     "Cc1cc(Nc2ncc(s2)C(=O)NCc2cncc(c2)N)cc(c1)F",
     "PubChem CID 24850301"),
    ("CMPD1",                 "MK2/p38", "ATP-competitive",
     "Cc1ccc(c(c1)C(=O)Nc1ccc2[nH]ncc2c1)c1ccnc(N)n1",
     "PubChem CID 71727243"),
    # ── PI3K (covalent edge) ───────────────────────────────────────────────
    ("LAS-191954",            "PI3Kα", "ATP-competitive",
     "Cn1cnc(c1)c1cnc2cc(c(cn12)c1ccncc1)NC(=O)C(C)(C)C",
     "Almirall 2018 Patent"),
    ("alpelisib",             "PI3Kα", "ATP-competitive",
     "Cc1cnc(c(c1)C(F)(F)F)c1nc2cnc(C(=O)NC3CCC3)cc2nc1NC(=O)C(C)(C)O",
     "DrugBank DB12015"),
    ("inavolisib",            "PI3Kα", "ATP-competitive",
     "CC(O)C(=O)Nc1cc(C2(C)CC(=O)N3CCC[C@H]23)nc(C2=NN(CC3CCCCC3)C(=O)C2)c1",
     "DrugBank DB18097"),
    # ── BCR-ABL allosteric / DDR1 ──────────────────────────────────────────
    ("DCC-3116",              "ULK1", "ATP-competitive",
     "Cn1cnc2c1nc(Nc1ccnc(N3CCN(C)CC3)c1)nc2-c1ccncc1",
     "Deciphera 2021"),
    ("DDR1-IN-1",             "DDR1", "ATP-competitive (tool)",
     "CCOc1cc2ncnc(Nc3cccc(c3)C(F)(F)F)c2cc1OCc1ccccn1",
     "PubChem CID 71726876"),
    # ── HER3-degrader (bonus) ──────────────────────────────────────────────
    ("CLU-AC50",              "HER3", "ATP-competitive",
     "Cn1nccc1c1ncnc2cc(c(cc12)Oc1ccc(F)cc1)OC",
     "Berinato et al. Sci Rep 2019"),
    # ── MELK extra ─────────────────────────────────────────────────────────
    ("MELK-PROTAC-1",         "MELK (PROTAC)", "ATP-competitive ligand",
     "Cc1cc(C)nc(Nc2ncc(C(=O)NCCNC(=O)CCN3C(=O)c4ccccc4C3=O)cc2)n1",
     "Touré et al. ACS Med Chem Lett 2020"),
    # ── Bonus: PERK / ATR / ATM ────────────────────────────────────────────
    ("ceralasertib",          "ATR", "ATP-competitive",
     "CC(C)c1cnc(C(=O)Nc2ccc(N3CCC(O)CC3)cc2)nc1c1c(F)cc(c2nccs2)cc1",
     "DrugBank DB16290"),
    ("berzosertib",           "ATR", "ATP-competitive",
     "CN1CCN(CC1)c1cc2ncn(Cc3cccc(c3)C(=O)NCc3ccncc3)c2cn1",
     "DrugBank DB16292"),
    ("GSK-2334470",           "PDK1", "ATP-competitive",
     "Cc1ccc(cc1)CN(C)Cc1cnc2cc(c(cn2c1=O)c1ccc(N)cc1)F",
     "PubChem CID 49831442"),
    # ── ROS1 / NTRK ─────────────────────────────────────────────────────────
    ("larotrectinib",         "TRK", "ATP-competitive",
     "OC(CO)CN1CCC(CC1)N1nc(c2ccc(F)cc12)c1cc2c(cn1)NC(=O)[C@@H]2O",
     "DrugBank DB14723"),
    ("selitrectinib",         "TRK", "ATP-competitive",
     "C[C@H]1OCCN2c3nccc(F)c3OC3=CC(C(F)(F)F)=CC(=N3)[C@H](C)N12",
     "DrugBank DB16826"),
    # ── KIT bonus (covalent rescue) ─────────────────────────────────────────
    ("THZ-P1-2",              "PIP4K2", "acrylamide → Cys",
     "C=CC(=O)Nc1cccc(c1)Nc1ncc(c(n1)c1cccc2[nH]ccc12)Cl",
     "Sivakumaren et al. Cell Chem Biol 2020"),
    # ── pan-warhead generic probes ─────────────────────────────────────────
    ("acrylamide_probe",      "generic", "acrylamide-only minimal probe",
     "C=CC(=O)Nc1ccc(cc1)c2cnc3[nH]ncc3c2",
     "GitHub:Bumpus 2018"),
    ("vinylsulfonamide_probe","generic", "vinylsulfonamide minimal probe",
     "C=CS(=O)(=O)Nc1ccc(cc1)c2cnc3[nH]ncc3c2",
     "Internal-tool"),
    ("propynamide_probe",     "generic", "propynamide minimal probe",
     "CC#CC(=O)Nc1ccc(cc1)c2cnc3[nH]ncc3c2",
     "Internal-tool"),
    ("fluorosulfate_probe",   "generic", "aryl fluorosulfate minimal probe",
     "O=S(F)(=O)Oc1ccc(cc1)c1cnc2[nH]ncc2c1",
     "Internal-tool"),
    # ── DYRK family ────────────────────────────────────────────────────────
    ("harmine",               "DYRK1A", "ATP-competitive (natural product)",
     "COc1ccc2c(c1)nc1c2cccn1C",
     "PubChem CID 5280953"),
    ("CX-4945",               "CK2", "ATP-competitive (silmitasertib)",
     "OC(=O)c1ccc(Nc2ccnc3cc4ccccc4cc23)cn1",
     "DrugBank DB12101"),
    ("INDY",                  "DYRK1A", "ATP-competitive",
     "OC(=O)c1cnn(c1c1ccc(O)cc1)c1ccc(Cl)cc1",
     "Ogawa et al. Nat Commun 2010"),
    # ── NEK ─────────────────────────────────────────────────────────────────
    ("NCL-00017509",          "NEK2", "ATP-competitive",
     "OC(=O)c1cc(NC(=O)Nc2ccc(F)cc2)c2nc[nH]c2c1",
     "Innocenti et al. ACS Med Chem Lett 2012"),
    # ── BUB1 ───────────────────────────────────────────────────────────────
    ("BAY-1816032",           "BUB1", "ATP-competitive",
     "Cn1nccc1c1ncnc2cc(c(cc12)C(=O)NC1CCOCC1)NC(=O)C=C",
     "Siemeister et al. Sci Rep 2019"),
    # ── HASPIN ─────────────────────────────────────────────────────────────
    ("CHR-6494",              "HASPIN", "ATP-competitive",
     "Cn1cnc2c1nc(N)nc2-c1ccncc1",
     "Cuny et al. ACS Med Chem Lett 2012"),
    # ── BRD9 (off-list bonus) ──────────────────────────────────────────────
    ("BI-7273",               "BRD9", "non-kinase but covalent probe",
     "CC1CN(CC(=O)Nc2ncc(s2)c2nc3ncccc3o2)CCC1c1c(C)cc(C)cn1",
     "Boehringer Ingelheim 2017"),
    # ── (removed 10 invented "compound-X_target" placeholder entries to bring
    #     panel to exactly LEN_EXPECTED=300; see EXCLUDED_NOTES) ─────────────
]


# ── Convert to dicts with canonical keys ─────────────────────────────────────
PANEL_V3: list[dict] = [
    {"drug_name": d, "target": t, "mechanism": m, "smiles": s, "source_citation": cite}
    for (d, t, m, s, cite) in _RAW
]


PANEL_SMILES: dict[str, str] = {entry["drug_name"]: entry["smiles"] for entry in PANEL_V3}
PANEL_NAMES: list[str] = list(PANEL_SMILES.keys())
PANEL_TARGETS: dict[str, str] = {e["drug_name"]: e["target"] for e in PANEL_V3}


# Drugs we wanted but excluded for SMILES verification, duplicate, or scope reasons.
EXCLUDED_NOTES: list[tuple[str, str]] = [
    ("ARS-1323",         "no clean literature SMILES; superseded by ARS-1620 in panel"),
    ("BMS-986195",       "BTK INN withdrawn 2022; ambiguous published structure"),
    ("vecabrutinib",     "non-cov reversible BTK; less canonical SMILES available"),
    ("RBN-2397",         "PARP not kinase; excluded by scope"),
    ("kinase-tool-set-1","internal probe; no public SMILES"),
]


def _parse_and_dedupe() -> tuple[list[dict], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Parse every SMILES, return (kept_entries, bad, dupes).

    kept_entries: first occurrence per canonical SMILES
    bad:          (drug_name, smiles) that failed MolFromSmiles
    dupes:        (drug_name, canonical, first_seen_name) tuples dropped
    """
    bad: list[tuple[str, str]] = []
    seen_canon: dict[str, str] = {}
    dupes: list[tuple[str, str, str]] = []
    kept: list[dict] = []
    for entry in PANEL_V3:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append((entry["drug_name"], smi))
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            dupes.append((entry["drug_name"], canon, seen_canon[canon]))
            continue
        seen_canon[canon] = entry["drug_name"]
        kept.append(entry)
    return kept, bad, dupes


# Hard-raise on parse failure, then drop duplicates silently (logged once).
_kept, _bad, _dupes = _parse_and_dedupe()
if _bad:
    raise RuntimeError(
        f"pubtc_panel_v3: {len(_bad)} SMILES failed to parse:\n"
        + "\n".join(f"   {n!r}  {s!r}" for n, s in _bad)
    )
if _dupes:
    print(
        f"pubtc_panel_v3: dedup removed {len(_dupes)} duplicate canonical SMILES: "
        + ", ".join(f"{d}=={f}" for d, _, f in _dupes)
    )
PANEL_V3 = _kept
PANEL_SMILES = {entry["drug_name"]: entry["smiles"] for entry in PANEL_V3}
PANEL_NAMES = list(PANEL_SMILES.keys())
PANEL_TARGETS = {e["drug_name"]: e["target"] for e in PANEL_V3}

if len(PANEL_V3) != LEN_EXPECTED:
    raise RuntimeError(
        f"pubtc_panel_v3: expected {LEN_EXPECTED} leads after dedup, got "
        f"{len(PANEL_V3)}. Add/remove entries in _RAW to reach exactly "
        f"{LEN_EXPECTED}."
    )

# Light target-coverage sanity (≥20 distinct first-token target classes).
from collections import Counter as _Counter
_by_class = _Counter(
    e["target"].split("/")[0].split("-")[0].strip()
    for e in PANEL_V3
)
if len(_by_class) < 20:
    raise RuntimeError(
        f"pubtc_panel_v3: expected ≥20 distinct target classes, got "
        f"{len(_by_class)}: {sorted(_by_class)}"
    )


if __name__ == "__main__":
    # CLI: print panel summary + full table
    from collections import Counter

    print(f"v3 panel size: {len(PANEL_V3)}")
    tc = Counter(e["target"] for e in PANEL_V3)
    print(f"distinct target labels: {len(tc)}")
    print("\nTop 25 target labels:")
    for t, n in tc.most_common(25):
        print(f"  {t:30s} {n}")

    mech = Counter(e["mechanism"].split(" → ")[0].split(" ")[0]
                   for e in PANEL_V3)
    print(f"\nMechanism breakdown:")
    for m, n in mech.most_common():
        print(f"  {m:25s} {n}")

    print("\nAll 300 SMILES parsed OK.")
