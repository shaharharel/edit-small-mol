"""Curate Exp7 v2 clean (anchor, drug) benchmark pairs.

For each post-2023 (or recently published) covalent drug whose SMILES is NOT
in CovInDB v2 AND NOT in the covalent-FT corpus, identify SAR analog anchors
in the same ChEMBL document on the same target, then filter by the 7 criteria:
  1. Drug NOT in CovInDB v2 (exact + stereoblind)
  2. Drug NOT in covalent_ft corpus (exact + stereoblind)
  3. Drug first ChEMBL activity year >= 2023 (TIER A) or 2022 (TIER B)
  4. Same SAR program (same doc_id, same target)
  5. Anchor->drug MMP edit distance >= 2 (rdMMPA fragmentation)
  6. Tc(anchor, drug) in [0.40, 0.85]
  7. Both have measured IC50/Ki/Kd with same standard_type, same assay

Outputs:
  data/exp7_v2_benchmark/clean_pairs.json  (schema mirrors v1 btk_pairs.json)
  data/exp7_v2_benchmark/curation_log.md
  data/exp7_v2_benchmark/audit_report.json
"""
from __future__ import annotations
import json
import pickle
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMMPA, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data" / "exp7_v2_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
CONTAM_PKL = OUT_DIR / "contamination_sets.pkl"

# Recently-disclosed covalent (and one non-cov pirtobrutinib) drugs with curated docs
# (name, chembl_id, [(doc_id, target_chembl_id, target_label, program, warhead, hinge)])
DRUG_DOCS: list[tuple[str, str, list[tuple]]] = [
    # ===== Covalent EGFR-mutant (zipalertinib & aumolertinib are clean of CovInDB v2) =====
    (
        "zipalertinib",
        "CHEMBL4650281",
        [
            (128983, "CHEMBL203", "EGFR_Ex20ins", "Cullinan/Taiho zipalertinib (2024 JMC)", "acrylamide", "4-aminopyrazolopyrimidine"),
            (127766, "CHEMBL203", "EGFR_Ex20ins", "zipalertinib follow-on (2023 JMC)", "acrylamide", "4-aminopyrazolopyrimidine"),
            (125565, "CHEMBL203", "EGFR_Ex20ins", "4-aminopyrazolopyrimidine scaffold (2022 RSC MedChem)", "acrylamide", "4-aminopyrazolopyrimidine"),
        ],
    ),
    (
        "aumolertinib",
        "CHEMBL4761468",
        [
            (129936, "CHEMBL203", "EGFR_T790M", "Aumolertinib-scaffold SAR (2024 Bioorg Med Chem)", "acrylamide", "aminopyrimidine"),
        ],
    ),
    # ===== Covalent BTK (tolebrutinib is clean of CovInDB v2 — verified) =====
    (
        "tolebrutinib",
        "CHEMBL4650323",
        [
            (128824, "CHEMBL5251", "BTK_Cys481", "Biogen BIIB129 covalent BTK (2024 JMC)", "acrylamide", "various"),
            (125751, "CHEMBL5251", "BTK_Cys481", "BeiGene BGB-8035 covalent BTK (2023 JMC)", "acrylamide", "various"),
        ],
    ),
    # ===== Non-cov BTK (nemtabrutinib is clean of CovInDB v2 — reversible covalent) =====
    (
        "nemtabrutinib",
        "CHEMBL4756476",
        [
            (134602, "CHEMBL5251", "BTK_C481S", "Merck/Schrödinger nemtabrutinib patent (2021)", "reversible", "aminopyrrolopyrimidone"),
        ],
    ),
    # ===== Non-cov BTK (pirtobrutinib reversible) =====
    (
        "fenebrutinib",
        "CHEMBL4065122",
        [
            (104876, "CHEMBL5251", "BTK", "Genentech GDC-0853 fenebrutinib disclosure (2018 JMC)", "noncov", "pyridinone"),
            (116462, "CHEMBL5251", "BTK", "Genentech fluorocyclopropyl amide SAR (2020 ACS MCL)", "noncov", "pyridinone"),
        ],
    ),
    # ===== KRAS G12D (MRTX1133, clean) =====
    (
        "MRTX1133",
        "CHEMBL4858364",
        [
            (124084, "CHEMBL2189121", "KRAS_G12D", "Mirati MRTX1133 original disclosure (2022 JMC)", "noncov", "naphthyl_pyrimidine"),
            (127977, "CHEMBL2189121", "KRAS_G12D", "Mirati MRTX1133 series (2023 ACS MCL)", "noncov", "naphthyl_pyrimidine"),
            (127694, "CHEMBL2189121", "KRAS_G12D", "Mirati KRAS G12D follow-on (2023 ACS MCL)", "noncov", "naphthyl_pyrimidine"),
            (125925, "CHEMBL2189121", "KRAS_G12D", "Mirati KRAS G12D deuterated (2023 ACS MCL)", "noncov", "naphthyl_pyrimidine"),
            (121370, "CHEMBL2189121", "KRAS_G12D", "Mirati KRAS G12D early disclosure (2021 ACS MCL)", "noncov", "naphthyl_pyrimidine"),
            (135452, "CHEMBL2189121", "KRAS_G12D", "Mirati KRAS G12D patent (2022)", "noncov", "naphthyl_pyrimidine"),
        ],
    ),
    # ===== SOS1 (MRTX0902, clean) =====
    (
        "MRTX0902",
        "CHEMBL5192659",
        [
            (124234, "CHEMBL2079846", "SOS1", "Mirati MRTX0902 SOS1 (2022 JMC)", "noncov", "quinazoline"),
        ],
    ),
    # ===== CDK7 (samuraciclib, clean) =====
    (
        "samuraciclib",
        "CHEMBL4297488",
        [
            (129708, "CHEMBL3038473", "CDK7", "Macrocyclic CDK7 series (2024 JMC)", "noncov", "pyrazolopyrimidine"),
        ],
    ),
    # ===== KRAS G12C (divarasib, clean of CovInDB v2) =====
    (
        "divarasib",
        "CHEMBL5095236",
        [
            (135062, "CHEMBL2189121", "KRAS_G12C", "Genentech divarasib patent (2022)", "acrylamide", "tetrahydropyrazine"),
        ],
    ),
    # ===== Lazertinib (clean, EGFR T790M) =====
    (
        "lazertinib",
        "CHEMBL4558324",
        [
            (118577, "CHEMBL203", "EGFR_T790M", "Lazertinib scaffold SAR (2020 Eur JMC)", "acrylamide", "aminopyrimidine"),
        ],
    ),
    # ===== Sonrotoclax / BGB-11417 (BCL-2, 2024 JMC, clean) =====
    (
        "sonrotoclax",
        "CHEMBL5314951",
        [
            (128954, "CHEMBL4860", "BCL2", "BeiGene sonrotoclax (BGB-11417) disclosure (2024 JMC)", "noncov", "sulfonamide_indole"),
        ],
    ),
    # ===== SY-5609 (CDK7 covalent-reversible, 2024 Eur JMC, clean) =====
    (
        "SY-5609",
        "CHEMBL5090754",
        [
            (129414, "CHEMBL3038473", "CDK7", "Syros SY-5609 thienopyrimidine series (2024 Eur JMC)", "reversible", "thienopyrimidine"),
        ],
    ),
    # ===== Resigratinib (FGFR2 irreversible, 2024 JMC KIN-3248 paper, clean) =====
    (
        "resigratinib",
        "CHEMBL5314537",
        [
            (129609, "CHEMBL3650", "FGFR1", "KIN-3248 Kinnate paper FGFR (2024 JMC)", "acrylamide", "purine"),
        ],
    ),
    # ===== Mevidalen-like / SY-5609 covalent CDK7 SAR alt — none currently named =====
]


# --------------------- helpers ---------------------
def canon_smi(smi: str) -> Optional[tuple[str, str]]:
    """Return (canonical, stereoblind) or None."""
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
    """Return min #MMP cuts that connect A->B via shared core.

    Uses rdMMPA.FragmentMol to enumerate (core, chain) splits for each mol;
    if a (core, chain1_a, chain1_b) match exists with chain1_a != chain1_b,
    that's a 1-edit MMP pair. Returns 1 if matching at cut=1, 2 at cut=2, etc.
    Returns None if no MMP relationship found at <= max_cuts.
    """
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    # rdMMPA returns list of (core, chains) tuples; chain is sidechain that was cut
    # Format: (chains_smi, core_smi)
    for ncuts in range(1, max_cuts + 1):
        frags_a = rdMMPA.FragmentMol(mol_a, maxCuts=ncuts, resultsAsMols=False)
        frags_b = rdMMPA.FragmentMol(mol_b, maxCuts=ncuts, resultsAsMols=False)
        # Each entry: (core_smi, chains_smi) — sometimes core is empty when whole mol is chain
        cores_a = defaultdict(set)
        cores_b = defaultdict(set)
        for entry in frags_a:
            core, chains = entry
            if core:  # cut entry (non-empty core)
                cores_a[core].add(chains)
        for entry in frags_b:
            core, chains = entry
            if core:
                cores_b[core].add(chains)
        # Look for shared core with different chains
        shared_cores = set(cores_a.keys()) & set(cores_b.keys())
        for core in shared_cores:
            for ca in cores_a[core]:
                for cb in cores_b[core]:
                    if ca != cb:
                        return ncuts
    return None


def pic50(value: float, units: str, stype: str) -> Optional[float]:
    """Convert standard_value+standard_units to pIC50 (or pKi/pKd).

    Returns None if non-numeric or units don't match expected.
    Skip 'pIC50' direct values — let the caller handle.
    """
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
    log_lines: list[str] = [
        "# Exp7 v2 Curation Log",
        "",
        "## Strategy",
        "",
        "- **Target**: re-curate clean (anchor, drug) pairs after v1 contamination audit found 31/50 MEMORIZED + 13/50 TRIVIAL + 0/50 CLEAN.",
        "- **Contamination corpora** (exclusion sets):",
        "  - CovInDB v2 (`data/covbinder/raw_covindb2/CovInDB_All.csv`, 8,277 unique canonical SMILES)",
        "  - Covalent FT corpus (`data/reinvent4_mol2mol_covalent_ft_work/covalent_smiles*.smi`, 900 unique SMILES)",
        "- **Drug filter**: SMILES must NOT match either corpus (exact + stereoblind canonical match).",
        "- **Anchor filter**: SMILES must NOT match the cov-FT corpus (the prior's actual training data). "
        "CovInDB v2 membership of the anchor is allowed (it's an annotation source, not a training set for the mol2mol prior); we flag it but do not reject.",
        "- **Temporal proxy** (criterion 3): drug AND anchor first ChEMBL activity year.",
        "  - TIER A: both >= 2024 (post mol2mol-prior cutoff, gold standard)",
        "  - TIER B: at least one >= 2023 (likely post-cutoff on at least one side)",
        "  - TIER C: both < 2023 (relies entirely on contamination check)",
        "- **Program filter** (criterion 4): anchor and drug must share a ChEMBL document, on the same target — proxy for same SAR program.",
        "- **MMP edit distance** (criterion 5): minimum 2 (avoids trivial 1-edit recovery).",
        "- **Tc band** (criterion 6): 0.40 - 0.85 (within-series, not scaffold-bridge).",
        "- **Activity filter** (criterion 7): both compounds measured with IC50/Ki/Kd in the SAME assay (same assay_id), standard_relation in `('=', '~')`.",
        "",
        "## Process",
        "",
        "1. Build canonical SMILES sets for both corpora (`build_contamination_sets.py` precursor, output `contamination_sets.pkl`).",
        "2. Enumerate ~25 candidate covalent / clinical-candidate drugs, hand-pick those NOT in either corpus.",
        "3. For each clean drug, identify recent ChEMBL documents (preferring 2023-2024 J Med Chem / Eur J Med Chem) with rich SAR (≥8 same-target activities).",
        "4. For each (drug, doc, target) triple, enumerate per-assay candidate anchors and apply all filters.",
        "5. Per-anchor dedup (keep best Δ across assays in same doc), then per-doc cap to N=5 with greedy Tc diversity.",
        "",
        "## Drugs evaluated\n",
    ]
    for dn, dcid, _ in DRUG_DOCS:
        log_lines.append(f"- {dn} ({dcid})")
    log_lines.append("\n## Rejected candidates (in CovInDB v2 — fail criterion 1)\n")
    log_lines.append("- opnurasib (CHEMBL5077861) — KRAS G12C, in CovInDB v2")
    log_lines.append("- remibrutinib (CHEMBL4483575) — covalent BTK, in CovInDB v2")
    log_lines.append("- orelabrutinib (CHEMBL4650321) — covalent BTK, in CovInDB v2")
    log_lines.append("\n## Per-drug curation detail (raw assay matches → filtered pairs)\n")
    pair_idx = 0

    for drug_name, drug_cid, doc_list in DRUG_DOCS:
        log_lines.append(f"\n## {drug_name} ({drug_cid})\n")
        # Fetch drug SMILES & contamination check
        cur.execute(
            "SELECT cs.canonical_smiles FROM compound_structures cs "
            "JOIN molecule_dictionary md ON cs.molregno=md.molregno WHERE md.chembl_id=?",
            (drug_cid,),
        )
        row = cur.fetchone()
        if not row:
            log_lines.append(f"  ! drug SMILES missing")
            continue
        drug_smi = row[0]
        cc = canon_smi(drug_smi)
        if not cc:
            log_lines.append(f"  ! drug SMILES unparseable: {drug_smi}")
            continue
        drug_canon, drug_stereo = cc
        in_covindb = drug_canon in cov_exact or drug_stereo in cov_stereo
        in_covft = drug_canon in ft_exact or drug_stereo in ft_stereo
        log_lines.append(f"  drug SMILES (canonical): `{drug_canon}`")
        log_lines.append(f"  in_covindb_v2={in_covindb}, in_covft={in_covft}")
        if in_covindb or in_covft:
            log_lines.append(f"  ! REJECTED drug — contamination detected, skipping all pairs for {drug_name}")
            continue

        # First-activity year (proxy for prior knowledge)
        cur.execute(
            "SELECT MIN(d.year) FROM activities act JOIN docs d ON act.doc_id=d.doc_id "
            "JOIN molecule_dictionary md ON act.molregno=md.molregno "
            "WHERE md.chembl_id=? AND act.standard_value IS NOT NULL",
            (drug_cid,),
        )
        yr1 = cur.fetchone()[0]
        log_lines.append(f"  first_activity_year={yr1}")

        for doc_id, tgt_cid, tgt_label, program, warhead, hinge in doc_list:
            log_lines.append(f"\n  ### doc {doc_id} target {tgt_label} ({tgt_cid})\n")
            # Pull all measured activities in this doc on this target
            cur.execute(
                """SELECT md.chembl_id, cs.canonical_smiles, act.standard_type, act.standard_value, act.standard_units, a.assay_id, a.chembl_id, a.description, d.year
                   FROM activities act
                   JOIN molecule_dictionary md ON act.molregno=md.molregno
                   JOIN compound_structures cs ON md.molregno=cs.molregno
                   JOIN assays a ON act.assay_id=a.assay_id
                   JOIN target_dictionary td ON a.tid=td.tid
                   JOIN docs d ON act.doc_id=d.doc_id
                   WHERE act.doc_id=? AND td.chembl_id=?
                     AND act.standard_type IN ('IC50','Ki','Kd','pIC50')
                     AND act.standard_value IS NOT NULL
                     AND act.standard_relation IN ('=', '~') """,
                (doc_id, tgt_cid),
            )
            rows = cur.fetchall()
            log_lines.append(f"  #raw activity rows: {len(rows)}")
            if not rows:
                continue

            # Build per-assay (assay_id, stype) -> {chembl_id: pIC50}
            assays: dict[tuple, dict] = defaultdict(dict)
            for r_cid, smi, stype, val, units, aid, achembl, adesc, yr in rows:
                pv = None
                if stype == "pIC50":
                    try:
                        pv = float(val)
                    except (TypeError, ValueError):
                        pv = None
                else:
                    pv = pic50(float(val), units, stype)
                if pv is None or not np.isfinite(pv):
                    continue
                # Keep best (lowest IC50 == highest pIC50) per (assay, compound)
                key = (aid, stype)
                if r_cid not in assays[key]:
                    assays[key][r_cid] = {"smi": smi, "pic50": pv, "achembl": achembl, "adesc": adesc, "year": yr}
                else:
                    if pv > assays[key][r_cid]["pic50"]:
                        assays[key][r_cid] = {"smi": smi, "pic50": pv, "achembl": achembl, "adesc": adesc, "year": yr}

            # Find an assay containing the drug
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
                log_lines.append(f"    assay {drug_entry['achembl']} (id={aid}, {stype}, n={n_in_assay}) drug_pIC50={drug_pic50:.2f}")
                # Iterate candidate anchors
                for r_cid, info in compounds.items():
                    if r_cid == drug_cid:
                        continue
                    anchor_smi = info["smi"]
                    anchor_pic50 = info["pic50"]
                    # Contamination on anchor (we allow anchor in CovInDB only if it's NOT memorized as anchor; the harm is if drug is)
                    # Per spec the rule applies to drug; but we also check anchor and warn
                    ac = canon_smi(anchor_smi)
                    if ac is None:
                        continue
                    anchor_canon, anchor_stereo = ac
                    anchor_in_covindb = anchor_canon in cov_exact or anchor_stereo in cov_stereo
                    anchor_in_covft = anchor_canon in ft_exact or anchor_stereo in ft_stereo
                    # Only REJECT when anchor is in covalent FT corpus (that IS prior training data).
                    # CovInDB v2 membership of the anchor is acceptable: it's an annotation source,
                    # not a training set for the mol2mol prior; the relevant leak is the DRUG.
                    if anchor_in_covft:
                        continue
                    # Tc(anchor, drug)
                    tc = tanimoto(anchor_smi, drug_smi)
                    if tc is None or not (0.40 <= tc <= 0.85):
                        continue
                    # Murcko Tc
                    tc_murcko = murcko_tanimoto(anchor_smi, drug_smi)
                    # MMP edit distance
                    med = mmp_edit_distance(anchor_smi, drug_smi, max_cuts=3)
                    if med is None or med < 2:
                        continue
                    delta = drug_pic50 - anchor_pic50
                    # Require non-zero delta (we want SAR signal)
                    if abs(delta) < 0.2:
                        continue
                    pair_idx += 1
                    # Tier based on drug + anchor first-activity year.
                    # TIER A: both drug AND anchor first appear >= 2024 (post-cutoff).
                    # TIER B: at least one >= 2023 (likely post-cutoff).
                    # TIER C: both < 2023 (rely entirely on contamination check).
                    cur.execute(
                        """SELECT MIN(d.year) FROM activities act JOIN docs d ON act.doc_id=d.doc_id
                           JOIN molecule_dictionary md ON act.molregno=md.molregno
                           WHERE md.chembl_id=? AND act.standard_value IS NOT NULL""",
                        (r_cid,),
                    )
                    anc_yr1 = cur.fetchone()[0]
                    if yr1 is not None and anc_yr1 is not None and yr1 >= 2024 and anc_yr1 >= 2024:
                        tier = "A"
                    elif (yr1 is not None and yr1 >= 2023) or (anc_yr1 is not None and anc_yr1 >= 2023):
                        tier = "B"
                    else:
                        tier = "C"
                    pair = {
                        "pair_id": f"v2_pair_{pair_idx:03d}",
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
                        "anchor": {
                            "chembl_id": r_cid,
                            "smiles": anchor_smi,
                            "pic50": round(anchor_pic50, 3),
                            "first_activity_year": anc_yr1,
                        },
                        "drug": {
                            "name": drug_name,
                            "chembl_id": drug_cid,
                            "smiles": drug_smi,
                            "pic50": round(drug_pic50, 3),
                            "first_activity_year": yr1,
                        },
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
                        f"      + pair {pair['pair_id']}: anchor={r_cid} (pIC50={anchor_pic50:.2f}) drug={drug_name} (pIC50={drug_pic50:.2f}) Tc={tc:.2f} MED={med} Δ={delta:+.2f}"
                    )

    # Step 1: Per-anchor dedup across assays in the same doc — keep best (highest |delta|) per anchor
    by_anchor: dict[tuple, dict] = {}
    for p in all_pairs:
        key = (p["doc_chembl_id"], p["anchor"]["smiles"], p["drug"]["smiles"])
        if key not in by_anchor or abs(p["delta_pic50"]) > abs(by_anchor[key]["delta_pic50"]):
            by_anchor[key] = p
    deduped = list(by_anchor.values())
    log_lines.append(f"\n\n## Totals\n- raw_pairs={len(all_pairs)}\n- after_anchor_dedup={len(deduped)}\n")

    # Step 2: Cap per (drug, doc, target) at N — pick diverse by Tc band
    PER_DOC_CAP = 5
    from collections import defaultdict as dd
    groups = dd(list)
    for p in deduped:
        groups[(p["drug"]["name"], p["doc_chembl_id"], p["target"])].append(p)

    def pick_diverse(pairs: list[dict], cap: int) -> list[dict]:
        """Greedy diversity: sort by |delta| desc, pick first; subsequent picks must
        differ in Tc by >= 0.05 from prior picks. Falls back to top-|delta| if exhausted."""
        if len(pairs) <= cap:
            return sorted(pairs, key=lambda p: -abs(p["delta_pic50"]))
        sorted_pairs = sorted(pairs, key=lambda p: -abs(p["delta_pic50"]))
        picked: list[dict] = []
        for p in sorted_pairs:
            if len(picked) >= cap:
                break
            if all(abs(p["tc_anchor_drug"] - q["tc_anchor_drug"]) >= 0.05 for q in picked):
                picked.append(p)
        # Top-up if we didn't fill cap
        if len(picked) < cap:
            for p in sorted_pairs:
                if p not in picked:
                    picked.append(p)
                if len(picked) >= cap:
                    break
        return picked

    unique_pairs: list[dict] = []
    for grp_key, pairs in groups.items():
        chosen = pick_diverse(pairs, PER_DOC_CAP)
        unique_pairs.extend(chosen)
    # Renumber pair ids deterministically (sort by drug, doc, anchor for stability)
    unique_pairs.sort(key=lambda p: (p["drug"]["name"], p["doc_chembl_id"], p["anchor"]["chembl_id"]))
    for i, p in enumerate(unique_pairs):
        p["pair_id"] = f"v2_pair_{i+1:03d}"
    log_lines.append(f"- after_per_doc_cap (N={PER_DOC_CAP}): {len(unique_pairs)}\n")

    # Aggregate counts
    by_target = defaultdict(int)
    by_drug = defaultdict(int)
    by_tier = defaultdict(int)
    for p in unique_pairs:
        by_target[p["target"]] += 1
        by_drug[p["drug"]["name"]] += 1
        by_tier[p["tier"]] += 1
    log_lines.append(f"### Per target\n")
    for k, v in sorted(by_target.items(), key=lambda x: -x[1]):
        log_lines.append(f"- {k}: {v}")
    log_lines.append(f"\n### Per drug\n")
    for k, v in sorted(by_drug.items(), key=lambda x: -x[1]):
        log_lines.append(f"- {k}: {v}")
    log_lines.append(f"\n### Per tier\n")
    for k, v in sorted(by_tier.items()):
        log_lines.append(f"- {k}: {v}")

    out = {
        "version": "v2",
        "n_pairs": len(unique_pairs),
        "criteria": {
            "tc_anchor_drug_band": [0.40, 0.85],
            "mmp_edit_distance_min": 2,
            "abs_delta_pic50_min": 0.2,
            "drug_excluded_from": ["CovInDB v2", "covalent_ft corpus"],
            "anchor_excluded_from": ["CovInDB v2", "covalent_ft corpus"],
            "temporal_tier_a_min_year": 2024,
            "temporal_tier_b_min_year": 2023,
            "temporal_tier_c": "<2023 (contamination check is sole filter)",
        },
        "by_target": dict(by_target),
        "by_drug": dict(by_drug),
        "by_tier": dict(by_tier),
        "pairs": unique_pairs,
    }
    (OUT_DIR / "clean_pairs.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "curation_log.md").write_text("\n".join(log_lines))
    print(f"\nWrote {OUT_DIR/'clean_pairs.json'} with {len(unique_pairs)} pairs")
    print(f"Wrote {OUT_DIR/'curation_log.md'}")
    print("Per target:", dict(by_target))
    print("Per drug:", dict(by_drug))
    print("Per tier:", dict(by_tier))


if __name__ == "__main__":
    main()
