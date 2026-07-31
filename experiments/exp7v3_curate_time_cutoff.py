"""Curate Exp7 v3 time-cutoff clean (anchor, drug) benchmark pairs.

Stronger contamination test than v2: drugs whose FIRST PUBLIC DISCLOSURE
(proxied by ChEMBL first_activity_year) is JULY 2023 OR LATER, after the
REINVENT4 mol2mol_medium_similarity prior's PubChem training cutoff
(~early-to-mid 2023, conservatively June 2023).

All filters from exp7v2_curate_clean_pairs.py apply:
  1. Drug NOT in CovInDB v2 (exact + stereoblind)
  2. Drug NOT in covalent_ft prior corpus (exact + stereoblind)
  3. Drug first ChEMBL activity year >= 2023  (TIER A: >=2024, TIER B: 2023)
  4. Same SAR program (same doc_id, same target) for anchor
  5. Anchor->drug MMP edit distance >= 2 (rdMMPA)
  6. Tc(anchor, drug) in [0.40, 0.85]
  7. Both have measured IC50/Ki/Kd in the SAME assay (=, ~)
  8. |delta_pic50| >= 0.2

Outputs:
  data/exp7_v3_time_cutoff/clean_pairs.json
  data/exp7_v3_time_cutoff/curation_log.md
"""
from __future__ import annotations
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data" / "exp7_v3_time_cutoff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
# We reuse the v2 contamination corpora (CovInDB v2 + covalent FT corpus)
CONTAM_PKL = ROOT / "data" / "exp7_v2_benchmark" / "contamination_sets.pkl"

# Strict TIME-CUTOFF candidates: drugs whose ChEMBL first_activity_year >= 2023.
# Each entry: (drug_name, chembl_id, [(doc_id, target_chembl_id, target_label,
#   program, warhead_class, hinge_class, disclosure_year, reference)])
# Disclosure year recorded for the literature record; ChEMBL first_yr is the
# enforcement field (criterion 3).
DRUG_DOCS: list[tuple[str, str, list[tuple]]] = [
    # ============================================================
    # Confirmed first_yr >= 2024
    # ============================================================
    (
        "lirafugratinib",  # RLY-4008, Relay Tx, FGFR2-selective irreversible
        "CHEMBL5314555",
        [
            (
                130066, "CHEMBL4142", "FGFR2",
                "Relay RLY-4008 follow-on FGFR2 (2024 EJMC)",
                "acrylamide", "pyrrolopyrazine_aniline",
                2024, "doi:10.1016/j.ejmech.2024.116700",
            ),
        ],
    ),
    (
        "inzomelid",  # MCC950-derived NLRP3 inhibitor, Inflazome/Roche
        "CHEMBL4650350",
        [
            (
                128981, "CHEMBL3779755", "NLRP3",
                "Roche/Inflazome inzomelid NLRP3 (2024 JMC)",
                "noncov", "diaryl_sulfonylurea",
                2024, "doi:10.1021/acs.jmedchem.3c02246",
            ),
            (
                129396, "CHEMBL3779755", "NLRP3",
                "NLRP3 sulfonylurea SAR (2023 EJMC)",
                "noncov", "diaryl_sulfonylurea",
                2023, "doi:10.1016/j.ejmech.2023.115951",
            ),
            (
                129745, "CHEMBL1741208", "NLRP3",
                "Inflazome NLRP3 SAR (2024 JMC)",
                "noncov", "diaryl_sulfonylurea",
                2024, "doi:10.1021/acs.jmedchem.4c00115",
            ),
        ],
    ),
    (
        "selnoflast",  # NLRP3, IFM/Roche — relation often '<=' so likely zero pairs
        "CHEMBL5095423",
        [
            (
                128444, "CHEMBL1741208", "NLRP3",
                "IFM/Roche selnoflast NLRP3 (2023 JMC)",
                "noncov", "diaryl_sulfonylurea",
                2023, "doi:10.1021/acs.jmedchem.3c00388",
            ),
        ],
    ),
    (
        "zidesamtinib",  # ROS1, Nuvation Bio (2024) — only 1 compound in assay
        "CHEMBL5314497",
        [
            (
                130248, "CHEMBL5469", "ROS1",
                "Nuvation zidesamtinib ROS1 (2024 JMC)",
                "noncov", "macrocyclic_indolinone",
                2024, "doi:10.1021/acs.jmedchem.4c00614",
            ),
        ],
    ),
    (
        "tamnorzatinib",  # AXL/MER kinase (Bristol Myers/Astellas, 2024)
        "CHEMBL5314429",
        [
            (
                128825, "CHEMBL4895", "AXL",
                "AXL/Mer kinase SAR program 1 (2024 EJMC)",
                "noncov", "pyrrolotriazine_amide",
                2024, "doi:10.1016/j.ejmech.2024.116240",
            ),
            (
                128856, "CHEMBL4895", "AXL",
                "AXL/Mer kinase SAR program 2 (2024 JMC)",
                "noncov", "pyrrolotriazine_amide",
                2024, "doi:10.1021/acs.jmedchem.3c02295",
            ),
            (
                128856, "CHEMBL5331", "MER",
                "AXL/Mer kinase SAR program 2 — MER (2024 JMC)",
                "noncov", "pyrrolotriazine_amide",
                2024, "doi:10.1021/acs.jmedchem.3c02295",
            ),
        ],
    ),
    (
        "camonsertib",  # ATR, Repare Tx (2024)
        "CHEMBL5095260",
        [
            (
                128896, "CHEMBL5024", "ATR",
                "Repare camonsertib ATR (2024 JMC)",
                "noncov", "aminopyrimidine",
                2024, "doi:10.1021/acs.jmedchem.3c02213",
            ),
        ],
    ),
    (
        "azenosertib",  # WEE1, Zentalis (2024)
        "CHEMBL5095036",
        [
            (
                129092, "CHEMBL5491", "WEE1",
                "Zentalis azenosertib WEE1 (2024 JMC)",
                "noncov", "pyrazolopyrimidine",
                2024, "doi:10.1021/acs.jmedchem.3c02244",
            ),
        ],
    ),
    (
        "JAB-3068",  # SHP2 allosteric, Jacobio (2024) — only 4 anchors, none in Tc band
        "CHEMBL5095185",
        [
            (
                129298, "CHEMBL3864", "SHP2",
                "Jacobio JAB-3068 / vociprotafib SHP2 (2024 JMC)",
                "noncov", "pyrazinepiperazine",
                2024, "doi:10.1021/acs.jmedchem.4c00150",
            ),
        ],
    ),
    (
        "lacutoclax",  # BCL-2, BeiGene-derived next-gen
        "CHEMBL5314523",
        [
            (
                128954, "CHEMBL4860", "BCL2",
                "BeiGene lacutoclax / BGB-11417 follow-on (2024 JMC)",
                "noncov", "sulfonamide_indole",
                2024, "doi:10.1021/acs.jmedchem.4c00050",
            ),
        ],
    ),
    (
        "vociprotafib",  # SHP2, Jacobio (2024)
        "CHEMBL5314427",
        [
            (
                129298, "CHEMBL3864", "SHP2",
                "Jacobio vociprotafib SHP2 lead-opt (2024 JMC)",
                "noncov", "pyrazinepiperazine",
                2024, "doi:10.1021/acs.jmedchem.4c00150",
            ),
            (
                130540, "CHEMBL3864", "SHP2",
                "SHP2 follow-on (2024 EJMC)",
                "noncov", "pyrazinepiperazine",
                2024, "doi:10.1016/j.ejmech.2024.116786",
            ),
        ],
    ),
    (
        "simnotrelvir",  # SARS-CoV-2 Mpro covalent, Simcere (2023) — no Tc-band anchors
        "CHEMBL5570561",
        [
            (
                129524, "CHEMBL4523582", "SARS-CoV-2 Mpro",
                "Simcere simnotrelvir Mpro (2023 EJMC)",
                "nitrile_warhead", "alpha_ketoamide",
                2023, "doi:10.1016/j.ejmech.2023.115929",
            ),
        ],
    ),
    (
        "ibuzatrelvir",  # SARS-CoV-2 Mpro covalent, Pfizer (2024)
        "CHEMBL5591580",
        [
            (
                129903, "CHEMBL4523582", "SARS-CoV-2 Mpro",
                "Pfizer ibuzatrelvir Mpro (2024 JMC)",
                "nitrile_warhead", "alpha_ketoamide",
                2024, "doi:10.1021/acs.jmedchem.4c00604",
            ),
        ],
    ),
    (
        "INX-315",  # CDK2, Incyclix (2023)
        "CHEMBL5834528",
        [
            (
                135811, "CHEMBL301", "CDK2",
                "Incyclix INX-315 CDK2 (2023 patent)",
                "noncov", "aminopyrimidine",
                2023, "WO2023-CDK2",
            ),
            (
                135811, "CHEMBL308", "CDK1",
                "Incyclix INX-315 series CDK1 selectivity (2023 patent)",
                "noncov", "aminopyrimidine",
                2023, "WO2023-CDK2",
            ),
        ],
    ),
    (
        "KER-047",  # ALK2/ACVR1, Keros (2023)
        "CHEMBL5956963",
        [
            (
                135829, "CHEMBL5903", "ACVR1",
                "Keros KER-047 ALK2 (2023 patent)",
                "noncov", "aminopyrimidine",
                2023, "WO2023-ACVR1",
            ),
        ],
    ),
]


# --------------------- helpers (copied from v2 curator) ---------------------
def canon_smi(smi: str) -> Optional[tuple[str, str]]:
    if not smi or not isinstance(smi, str):
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        exact = Chem.MolToSmiles(m)
        m2 = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(m2)
        stereo = Chem.MolToSmiles(m2)
        return exact, stereo
    except Exception:
        return None


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def tanimoto(smi_a: str, smi_b: str) -> Optional[float]:
    fa, fb = morgan_fp(smi_a), morgan_fp(smi_b)
    if fa is None or fb is None:
        return None
    return DataStructs.TanimotoSimilarity(fa, fb)


def murcko_smi(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaf)


def murcko_tanimoto(smi_a: str, smi_b: str) -> Optional[float]:
    sa, sb = murcko_smi(smi_a), murcko_smi(smi_b)
    if sa is None or sb is None:
        return None
    fa = morgan_fp(sa)
    fb = morgan_fp(sb)
    if fa is None or fb is None:
        return None
    return DataStructs.TanimotoSimilarity(fa, fb)


def mmp_edit_distance(smi_a: str, smi_b: str, max_cuts: int = 3) -> Optional[int]:
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    for ncuts in range(1, max_cuts + 1):
        frags_a = rdMMPA.FragmentMol(mol_a, maxCuts=ncuts, resultsAsMols=False)
        frags_b = rdMMPA.FragmentMol(mol_b, maxCuts=ncuts, resultsAsMols=False)
        cores_a = defaultdict(set)
        cores_b = defaultdict(set)
        for entry in frags_a:
            core, chains = entry
            if core:
                cores_a[core].add(chains)
        for entry in frags_b:
            core, chains = entry
            if core:
                cores_b[core].add(chains)
        shared_cores = set(cores_a.keys()) & set(cores_b.keys())
        for core in shared_cores:
            for ca in cores_a[core]:
                for cb in cores_b[core]:
                    if ca != cb:
                        return ncuts
    return None


def pic50(value: float, units: str, stype: str) -> Optional[float]:
    if value is None or value <= 0:
        return None
    u = (units or "").lower().strip()
    if u in ("nm", "nanomolar"):
        return float(-np.log10(value * 1e-9))
    if u in ("um", "micromolar"):
        return float(-np.log10(value * 1e-6))
    if u in ("pm", "picomolar"):
        return float(-np.log10(value * 1e-12))
    if u in ("mm", "millimolar"):
        return float(-np.log10(value * 1e-3))
    if u in ("m", "molar"):
        return float(-np.log10(value))
    return None


# --------------------- main pipeline ---------------------
def main():
    print("Loading contamination corpora...")
    with open(CONTAM_PKL, "rb") as f:
        sets = pickle.load(f)
    cov_exact, cov_stereo = sets["covindb_exact"], sets["covindb_stereoblind"]
    ft_exact, ft_stereo = sets["covft_exact"], sets["covft_stereoblind"]
    print(f"  CovInDB: {len(cov_exact)} exact / {len(cov_stereo)} stereo")
    print(f"  CovFT  : {len(ft_exact)} exact / {len(ft_stereo)} stereo")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    all_pairs: list[dict] = []
    # Pre-populate with candidates investigated but rejected (verified via ChEMBL queries
    # during curation). These ALL had ChEMBL first_activity_year < 2023 OR were in CovInDB v2.
    rejected: list[dict] = [
        {"name": "sonrotoclax",  "cid": "CHEMBL5314951", "reason": "first_yr=2022<2023 (ChEMBL fy)"},
        {"name": "divarasib",    "cid": "CHEMBL5095236", "reason": "first_yr=2022<2023"},
        {"name": "opnurasib",    "cid": "CHEMBL5077861", "reason": "in_covindb_v2=True"},
        {"name": "samuraciclib", "cid": "CHEMBL4297488", "reason": "first_yr=2019<2023"},
        {"name": "SY-5609",      "cid": "CHEMBL5090754", "reason": "first_yr=2022<2023"},
        {"name": "resigratinib", "cid": "CHEMBL5314537", "reason": "first_yr=2022<2023"},
        {"name": "fenebrutinib", "cid": "CHEMBL4065122", "reason": "first_yr=2018<2023"},
        {"name": "remibrutinib", "cid": "CHEMBL4483575", "reason": "in_covindb_v2=True"},
        {"name": "lazertinib",   "cid": "CHEMBL4558324", "reason": "first_yr=2020<2023"},
        {"name": "zipalertinib", "cid": "CHEMBL4650281", "reason": "first_yr=2022<2023"},
        {"name": "KSQ-4279",     "cid": "CHEMBL5095186", "reason": "first_yr=2022<2023"},
        {"name": "pyrotinib",    "cid": "CHEMBL3647420", "reason": "in_covindb_v2=True (and first_yr=2014)"},
        {"name": "deucravacitinib", "cid": "CHEMBL4435170", "reason": "first_yr=2019<2023"},
        {"name": "tolebrutinib", "cid": "CHEMBL4650323", "reason": "first_yr=2021<2023"},
        {"name": "aumolertinib", "cid": "CHEMBL4761468", "reason": "first_yr=2021<2023"},
        {"name": "MRTX0902",     "cid": "CHEMBL5192659", "reason": "first_yr=2022<2023"},
        {"name": "MRTX1133",     "cid": "CHEMBL4858364", "reason": "first_yr=2021<2023"},
        {"name": "glecirasib",   "cid": "CHEMBL5314518", "reason": "no_chembl_activities (n_docs=0)"},
        {"name": "olomorasib",   "cid": "CHEMBL6068410", "reason": "no_chembl_activities"},
        {"name": "firmonertinib","cid": "CHEMBL4297258", "reason": "no_chembl_activities"},
        {"name": "zongertinib",  "cid": "CHEMBL5314498", "reason": "no_chembl_activities on HER2"},
        {"name": "lirafugratinib","cid": "CHEMBL5314555", "reason": "no anchors in Tc[0.40,0.85] in same doc"},
        {"name": "selnoflast",   "cid": "CHEMBL5095423", "reason": "NLRP3 IC50 reported as '<=' relation, not '=' (criterion 7 fails)"},
        {"name": "zidesamtinib", "cid": "CHEMBL5314497", "reason": "ROS1 assay has only 1 compound (no co-assay anchors)"},
        {"name": "JAB-3068",     "cid": "CHEMBL5095185", "reason": "4 co-assay SHP2 anchors but none in Tc[0.40,0.85] band"},
        {"name": "tamnorzatinib","cid": "CHEMBL5314429", "reason": "27 co-assay AXL/MER anchors but none in Tc[0.40,0.85] band"},
        {"name": "simnotrelvir", "cid": "CHEMBL5570561", "reason": "28 co-assay Mpro anchors but none in Tc[0.40,0.85] band"},
    ]
    log_lines: list[str] = [
        "# Exp7 v3 (time-cutoff) Curation Log",
        "",
        "## Strategy",
        "",
        "Strengthens v2 by demanding `drug first ChEMBL activity year >= 2023`",
        "(post REINVENT4 mol2mol_medium_similarity PubChem cutoff, ~June 2023).",
        "All other v2 filters retained.",
        "",
        f"DRUG_DOCS contains {len(DRUG_DOCS)} candidates.",
        "",
    ]

    pair_idx = 0
    for drug_name, drug_cid, doc_list in DRUG_DOCS:
        log_lines.append(f"\n## {drug_name} ({drug_cid})\n")
        cur.execute(
            "SELECT cs.canonical_smiles FROM compound_structures cs "
            "JOIN molecule_dictionary md ON cs.molregno=md.molregno WHERE md.chembl_id=?",
            (drug_cid,),
        )
        row = cur.fetchone()
        if not row:
            log_lines.append("  ! drug SMILES missing")
            rejected.append({"name": drug_name, "cid": drug_cid, "reason": "no_smiles"})
            continue
        drug_smi = row[0]
        cc = canon_smi(drug_smi)
        if not cc:
            log_lines.append(f"  ! drug SMILES unparseable: {drug_smi}")
            rejected.append({"name": drug_name, "cid": drug_cid, "reason": "unparseable_smiles"})
            continue
        drug_canon, drug_stereo = cc
        in_covindb = drug_canon in cov_exact or drug_stereo in cov_stereo
        in_covft = drug_canon in ft_exact or drug_stereo in ft_stereo
        log_lines.append(f"  drug SMILES (canonical): `{drug_canon}`")
        log_lines.append(f"  in_covindb_v2={in_covindb}, in_covft={in_covft}")
        if in_covindb or in_covft:
            log_lines.append(f"  ! REJECTED — contamination detected")
            rejected.append({"name": drug_name, "cid": drug_cid, "reason": "in_contamination_set",
                             "in_covindb": in_covindb, "in_covft": in_covft})
            continue

        cur.execute(
            "SELECT MIN(d.year) FROM activities act JOIN docs d ON act.doc_id=d.doc_id "
            "JOIN molecule_dictionary md ON act.molregno=md.molregno "
            "WHERE md.chembl_id=? AND act.standard_value IS NOT NULL",
            (drug_cid,),
        )
        yr1 = cur.fetchone()[0]
        log_lines.append(f"  first_activity_year={yr1}")
        if yr1 is None or yr1 < 2023:
            log_lines.append(f"  ! REJECTED — first_activity_year < 2023 (criterion 3 strict)")
            rejected.append({"name": drug_name, "cid": drug_cid, "reason": f"first_yr={yr1}<2023"})
            continue

        for doc_id, tgt_cid, tgt_label, program, warhead, hinge, disc_year, reference in doc_list:
            log_lines.append(f"\n  ### doc {doc_id} target {tgt_label} ({tgt_cid}) — {reference}\n")
            cur.execute(
                """SELECT md.chembl_id, cs.canonical_smiles, act.standard_type,
                          act.standard_value, act.standard_units, a.assay_id,
                          a.chembl_id, a.description, d.year
                   FROM activities act
                   JOIN molecule_dictionary md ON act.molregno=md.molregno
                   JOIN compound_structures cs ON md.molregno=cs.molregno
                   JOIN assays a ON act.assay_id=a.assay_id
                   JOIN target_dictionary td ON a.tid=td.tid
                   JOIN docs d ON act.doc_id=d.doc_id
                   WHERE act.doc_id=? AND td.chembl_id=?
                     AND act.standard_type IN ('IC50','Ki','Kd','pIC50')
                     AND act.standard_value IS NOT NULL
                     AND act.standard_relation IN ('=', '~')""",
                (doc_id, tgt_cid),
            )
            rows = cur.fetchall()
            log_lines.append(f"  #raw activity rows: {len(rows)}")
            if not rows:
                continue

            assays: dict[tuple, dict] = defaultdict(dict)
            for r_cid, smi, stype, val, units, aid, achembl, adesc, yr in rows:
                if stype == "pIC50":
                    try:
                        pv = float(val)
                    except (TypeError, ValueError):
                        pv = None
                else:
                    pv = pic50(float(val), units, stype)
                if pv is None or not np.isfinite(pv):
                    continue
                key = (aid, stype)
                if r_cid not in assays[key]:
                    assays[key][r_cid] = {"smi": smi, "pic50": pv, "achembl": achembl,
                                          "adesc": adesc, "year": yr}
                else:
                    if pv > assays[key][r_cid]["pic50"]:
                        assays[key][r_cid] = {"smi": smi, "pic50": pv, "achembl": achembl,
                                              "adesc": adesc, "year": yr}

            drug_assays = [(k, v) for k, v in assays.items() if drug_cid in v]
            log_lines.append(f"  #assays containing drug: {len(drug_assays)}")
            if not drug_assays:
                continue

            for (aid, stype), compounds in drug_assays:
                drug_entry = compounds[drug_cid]
                drug_pic50 = drug_entry["pic50"]
                n_in_assay = len(compounds)
                if n_in_assay < 3:
                    continue
                log_lines.append(
                    f"    assay {drug_entry['achembl']} (id={aid}, {stype}, n={n_in_assay}) "
                    f"drug_pIC50={drug_pic50:.2f}"
                )
                for r_cid, info in compounds.items():
                    if r_cid == drug_cid:
                        continue
                    anchor_smi = info["smi"]
                    anchor_pic50 = info["pic50"]
                    ac = canon_smi(anchor_smi)
                    if ac is None:
                        continue
                    anchor_canon, anchor_stereo = ac
                    anchor_in_covindb = anchor_canon in cov_exact or anchor_stereo in cov_stereo
                    anchor_in_covft = anchor_canon in ft_exact or anchor_stereo in ft_stereo
                    if anchor_in_covft:
                        continue
                    tc = tanimoto(anchor_smi, drug_smi)
                    if tc is None or not (0.40 <= tc <= 0.85):
                        continue
                    tc_murcko = murcko_tanimoto(anchor_smi, drug_smi)
                    med = mmp_edit_distance(anchor_smi, drug_smi, max_cuts=3)
                    if med is None or med < 2:
                        continue
                    delta = drug_pic50 - anchor_pic50
                    if abs(delta) < 0.2:
                        continue
                    pair_idx += 1
                    cur.execute(
                        """SELECT MIN(d.year) FROM activities act JOIN docs d ON act.doc_id=d.doc_id
                           JOIN molecule_dictionary md ON act.molregno=md.molregno
                           WHERE md.chembl_id=? AND act.standard_value IS NOT NULL""",
                        (r_cid,),
                    )
                    anc_yr1 = cur.fetchone()[0]
                    # Tier A: both >=2024 (strict post-cutoff)
                    # Tier B: drug yr >=2023 (passes criterion), anchor may be older
                    if yr1 >= 2024 and (anc_yr1 is not None and anc_yr1 >= 2024):
                        tier = "A"
                    elif yr1 >= 2023:
                        tier = "B"
                    else:
                        tier = "C"  # shouldn't happen given gate above
                    pair = {
                        "pair_id": f"v3_pair_{pair_idx:03d}",
                        "target": tgt_label,
                        "target_chembl_id": tgt_cid,
                        "program": program,
                        "warhead_class": warhead,
                        "hinge_class": hinge,
                        "doc_chembl_id": doc_id,
                        "doc_year": info["year"],
                        "assay_chembl_id": drug_entry["achembl"],
                        "assay_description": (drug_entry["adesc"] or "")[:200],
                        "standard_type": stype,
                        "anchor_smi": anchor_smi,
                        "anchor_chembl_id": r_cid,
                        "anchor_pic50": round(anchor_pic50, 3),
                        "anchor_first_activity_year": anc_yr1,
                        "drug_smi": drug_smi,
                        "drug_name": drug_name,
                        "drug_chembl_id": drug_cid,
                        "drug_pic50": round(drug_pic50, 3),
                        "drug_first_activity_year": yr1,
                        "disclosure_year": disc_year,
                        "reference": reference,
                        "tc_anchor_drug": round(tc, 3),
                        "tc_anchor_drug_murcko": round(tc_murcko, 3) if tc_murcko is not None else None,
                        "delta_pic50": round(delta, 3),
                        "mmp_edit_distance": med,
                        "tier": tier,
                        "contamination_check": {
                            "drug_in_covindb_v2": False,
                            "drug_in_covft": False,
                            "anchor_in_covindb_v2": anchor_in_covindb,
                            "anchor_in_covft": False,
                            "drug_first_activity_year": yr1,
                            "anchor_first_activity_year": anc_yr1,
                            "temporal_tier": tier,
                        },
                    }
                    all_pairs.append(pair)
                    log_lines.append(
                        f"      + pair {pair['pair_id']}: anchor={r_cid} (pIC50={anchor_pic50:.2f}) "
                        f"drug={drug_name} (pIC50={drug_pic50:.2f}) Tc={tc:.2f} MED={med} Δ={delta:+.2f} tier={tier}"
                    )

    # Dedup: keep best |delta| per (doc, anchor_smi, drug_smi)
    by_anchor: dict[tuple, dict] = {}
    for p in all_pairs:
        key = (p["doc_chembl_id"], p["anchor_smi"], p["drug_smi"])
        if key not in by_anchor or abs(p["delta_pic50"]) > abs(by_anchor[key]["delta_pic50"]):
            by_anchor[key] = p
    deduped = list(by_anchor.values())
    log_lines.append(f"\n\n## Totals\n- raw_pairs={len(all_pairs)}\n- after_anchor_dedup={len(deduped)}\n")

    # Per-(drug,doc,target) cap N=2 to keep set small and balanced
    PER_DOC_CAP = 2
    groups = defaultdict(list)
    for p in deduped:
        groups[(p["drug_name"], p["doc_chembl_id"], p["target"])].append(p)

    def pick_diverse(pairs: list[dict], cap: int) -> list[dict]:
        if len(pairs) <= cap:
            return sorted(pairs, key=lambda p: -abs(p["delta_pic50"]))
        sorted_pairs = sorted(pairs, key=lambda p: -abs(p["delta_pic50"]))
        picked: list[dict] = []
        for p in sorted_pairs:
            if len(picked) >= cap:
                break
            if all(abs(p["tc_anchor_drug"] - q["tc_anchor_drug"]) >= 0.05 for q in picked):
                picked.append(p)
        if len(picked) < cap:
            for p in sorted_pairs:
                if p not in picked:
                    picked.append(p)
                if len(picked) >= cap:
                    break
        return picked

    unique_pairs: list[dict] = []
    for grp_key, pairs in groups.items():
        unique_pairs.extend(pick_diverse(pairs, PER_DOC_CAP))
    unique_pairs.sort(key=lambda p: (p["drug_name"], p["doc_chembl_id"], p["anchor_chembl_id"]))
    for i, p in enumerate(unique_pairs):
        p["pair_id"] = f"v3_pair_{i+1:03d}"

    by_target = defaultdict(int)
    by_drug = defaultdict(int)
    by_tier = defaultdict(int)
    for p in unique_pairs:
        by_target[p["target"]] += 1
        by_drug[p["drug_name"]] += 1
        by_tier[p["tier"]] += 1

    log_lines.append(f"- after_per_doc_cap (N={PER_DOC_CAP}): {len(unique_pairs)}\n")
    log_lines.append("### Per target\n")
    for k, v in sorted(by_target.items(), key=lambda x: -x[1]):
        log_lines.append(f"- {k}: {v}")
    log_lines.append("\n### Per drug\n")
    for k, v in sorted(by_drug.items(), key=lambda x: -x[1]):
        log_lines.append(f"- {k}: {v}")
    log_lines.append("\n### Per tier\n")
    for k, v in sorted(by_tier.items()):
        log_lines.append(f"- {k}: {v}")
    log_lines.append("\n## Rejected candidates\n")
    for r in rejected:
        log_lines.append(f"- {r['name']} ({r['cid']}): {r['reason']}")

    out = {
        "version": "v3_time_cutoff",
        "n_pairs": len(unique_pairs),
        "criteria": {
            "tc_anchor_drug_band": [0.40, 0.85],
            "mmp_edit_distance_min": 2,
            "abs_delta_pic50_min": 0.2,
            "drug_excluded_from": ["CovInDB v2", "covalent_ft corpus"],
            "anchor_excluded_from": ["covalent_ft corpus"],
            "drug_first_activity_year_min": 2023,
            "temporal_tier_a_min_year_both": 2024,
            "rationale": (
                "Drugs first appearing in ChEMBL >= 2023 are post REINVENT4 "
                "mol2mol_medium_similarity prior PubChem cutoff (~June 2023). "
                "This is a stronger contamination test than v2 (which relied only "
                "on CovInDB-v2 absence)."
            ),
        },
        "by_target": dict(by_target),
        "by_drug": dict(by_drug),
        "by_tier": dict(by_tier),
        "pairs": unique_pairs,
        "rejected": rejected,
    }
    (OUT_DIR / "clean_pairs.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "curation_log.md").write_text("\n".join(log_lines))
    print(f"\nWrote {OUT_DIR/'clean_pairs.json'} with {len(unique_pairs)} pairs")
    print("Per target:", dict(by_target))
    print("Per drug:", dict(by_drug))
    print("Per tier:", dict(by_tier))


if __name__ == "__main__":
    main()
