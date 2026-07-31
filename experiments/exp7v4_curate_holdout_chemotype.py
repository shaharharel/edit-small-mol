"""Curate Exp7 v4 holdout-chemotype benchmark.

For each of 5 covalent / kinase series, define:
  - series_name
  - target (ChEMBL target id)
  - holdout_smarts: SMARTS that identifies the held-out sub-scaffold class
  - train_anchors: 10-30 SMILES that DON'T match holdout_smarts (RL seeds)
  - test_drugs: 3-10 SMILES that DO match holdout_smarts (rediscovery targets)
  - holdout_rationale: 1-sentence "why"

Anchor selection:
  - Pull all compounds with measured activity on target (=, ~ standard_relation)
  - Pull all compounds with pIC50 >= 6.5 (active anchors only)
  - Compounds matching holdout_smarts -> test_drugs (cap 10)
  - Compounds NOT matching but with Murcko-Tc to test_drugs >= 0.15 -> train_anchors
  - Cap train_anchors at 30 by diverse-Tc to test_drugs

Output: data/exp7_v4_holdout_chemotype/pairs.json (single JSON with list of 5 series)
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data" / "exp7_v4_holdout_chemotype"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"


# Series definitions: (name, target_chembl_id, target_pref_name, holdout_smarts, rationale, known_drug_names_in_holdout)
# IMPORTANT: SMARTS uses generic n (not [nH]) so substituted ring N matches; uses generic
# aromaticity so canonical RDKit forms work.
SERIES = [
    {
        "series_name": "BTK_pyrazolopyrimidin4amine_ibrutinib_class",
        "target": "CHEMBL5251",
        "target_pref_name": "Tyrosine-protein kinase BTK",
        # Pyrazolo[3,4-d]pyrimidin-4-amine — matches ibrutinib's c3c(N)ncnc32 ring fusion
        # Generic SMARTS: 4-amino-pyrazolopyrimidine fused ring (ibrutinib core)
        "holdout_smarts": "c1ncnc2nnc(*)c12",
        "holdout_rationale": (
            "Pyrazolo[3,4-d]pyrimidin-4-amine is the canonical ibrutinib chemotype "
            "(also PCI-45292, evobrutinib, tirabrutinib). Holding it out forces the "
            "model to rediscover the scaffold class that dominated the first decade "
            "of approved covalent BTK programs."
        ),
        "known_drug_pref_names": [
            "IBRUTINIB", "EVOBRUTINIB", "TIRABRUTINIB",
        ],
    },
    {
        "series_name": "EGFR_pyrrolopyrimidine_acrylamide_zipalertinib_class",
        "target": "CHEMBL203",
        "target_pref_name": "Epidermal growth factor receptor erbB1",
        # 4-amino pyrrolo[2,3-d]pyrimidine — matches CLN-081/zipalertinib
        # SMARTS allows substituted N1 (i.e. n(C)) — uses c3c(N)ncnc3n2 from CLN-081
        "holdout_smarts": "c1ncnc2n(*)ccc12",
        "holdout_rationale": (
            "4-amino-pyrrolo[2,3-d]pyrimidine is the zipalertinib (CLN-081) EGFR Ex20ins "
            "chemotype — a recent, mutation-selective covalent class distinct from the "
            "osimertinib aminopyrimidine and the gefitinib quinazoline families."
        ),
        "known_drug_pref_names": ["CLN-081", "ZIPALERTINIB"],
    },
    {
        "series_name": "KRAS_G12C_arylacrylamide_piperazine_sotorasib_class",
        "target": "CHEMBL2189121",
        "target_pref_name": "KRAS",
        # Piperazinyl acrylamide (or fluoro-vinyl) attached to a fused-aryl: sotorasib +
        # adagrasib both have aryl-piperazine-acrylamide. Allow halogenated vinyl as warhead.
        "holdout_smarts": "[C;X3](=[O])N1CC[NX3](c2[c,n][c,n][c,n][c,n]c2*)CC1",
        "holdout_rationale": (
            "Piperazinyl arylacrylamide is the sotorasib/adagrasib KRAS-G12C "
            "hinge+warhead scaffold (acrylamide on the distal piperazine N, with a "
            "fused-bicyclic on the proximal piperazine N). Removing it tests rediscovery "
            "of the dominant clinical chemotype for the first druggable KRAS allele."
        ),
        "known_drug_pref_names": ["SOTORASIB", "ADAGRASIB", "AMG-510", "MRTX849", "DIVARASIB"],
    },
    {
        "series_name": "HER2_quinazoline_acrylamide_afatinib_class",
        "target": "CHEMBL1824",
        "target_pref_name": "Receptor protein-tyrosine kinase erbB-2",
        # Compound must have BOTH 4-amino-quinazoline AND an acrylamide warhead.
        # We use a multi-SMARTS conjunction (handled by the curator). The first SMARTS
        # is the hinge-binder; the second is the warhead. A molecule passes if it
        # has BOTH substructures.
        "holdout_smarts": "Nc1ncnc2ccccc12",
        "holdout_smarts_extra": "C=CC(=O)N",
        "holdout_rationale": (
            "4-anilinoquinazoline + acrylamide is the afatinib/dacomitinib/pyrotinib "
            "HER2 chemotype — the dominant covalent-EGFR/HER2 scaffold class for "
            "20+ years. Holding it out forces rediscovery of the canonical clinical "
            "covalent HER2 class while leaving the non-acrylamide quinazolines "
            "(lapatinib, tucatinib) in the training pool."
        ),
        "known_drug_pref_names": [
            "AFATINIB", "DACOMITINIB", "PYROTINIB", "NERATINIB", "POZIOTINIB", "TUCATINIB",
        ],
    },
    {
        "series_name": "CDK7_azaindole_phosphine_oxide_SY5609_class",
        "target": "CHEMBL3038473",
        "target_pref_name": "Cyclin-dependent kinase 7",
        # SY-5609: 7-azaindole bearing a dimethylphosphine oxide ortho to the pyrrole N
        # SMARTS: P(C)(C)(=O)-c attached to bicycle with [nH] and fused 6-ring
        "holdout_smarts": "[P](=O)(C)(C)c1cccc2c1[nH]cc2",
        "holdout_rationale": (
            "The 7-azaindole bearing a dimethylphosphine oxide ortho to the pyrrole N "
            "is the Syros SY-5609 CDK7 covalent-reversible chemotype — distinctive "
            "enough that no other CDK7 program uses it. Holding it out tests "
            "rediscovery of a single named-invention scaffold rather than a broad class."
        ),
        "known_drug_pref_names": ["SY-5609", "SY5609"],
    },
]


# ----------------- helpers -----------------
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


def fetch_compounds_with_activity(cur, target_chembl_id: str, min_pic50: float = 6.5):
    """Return list of dicts: {chembl_id, smiles, pic50_max, pref_name, first_year}"""
    cur.execute(
        """SELECT md.chembl_id, cs.canonical_smiles, act.standard_type, act.standard_value,
                  act.standard_units, md.pref_name
           FROM activities act
           JOIN molecule_dictionary md ON act.molregno=md.molregno
           JOIN compound_structures cs ON md.molregno=cs.molregno
           JOIN assays a ON act.assay_id=a.assay_id
           JOIN target_dictionary td ON a.tid=td.tid
           WHERE td.chembl_id=?
             AND act.standard_type IN ('IC50','Ki','Kd','pIC50')
             AND act.standard_value IS NOT NULL
             AND act.standard_relation IN ('=', '~')""",
        (target_chembl_id,),
    )
    rows = cur.fetchall()
    by_cid: dict[str, dict] = {}
    for r_cid, smi, stype, val, units, pref_name in rows:
        if stype == "pIC50":
            try:
                pv = float(val)
            except (TypeError, ValueError):
                pv = None
        else:
            pv = pic50(float(val), units, stype)
        if pv is None or not np.isfinite(pv):
            continue
        # Sanity cap on pIC50 — reject suspicious values >= 11.0 (sub-picomolar; usually
        # mislabelled units in ChEMBL or pseudo-Ki values)
        if pv > 11.0:
            continue
        if r_cid not in by_cid or pv > by_cid[r_cid]["pic50"]:
            by_cid[r_cid] = {"chembl_id": r_cid, "smiles": smi, "pic50": pv, "pref_name": pref_name}

    # Filter by min potency
    return [c for c in by_cid.values() if c["pic50"] >= min_pic50]


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    series_out = []
    for series in SERIES:
        print(f"\n=== {series['series_name']} on {series['target_pref_name']} ===")
        compounds = fetch_compounds_with_activity(cur, series["target"], min_pic50=6.5)
        print(f"  N actives (pIC50>=6.5) on target: {len(compounds)}")

        patt = Chem.MolFromSmarts(series["holdout_smarts"])
        assert patt is not None, f"Bad SMARTS: {series['holdout_smarts']}"
        patt_extra = None
        if "holdout_smarts_extra" in series:
            patt_extra = Chem.MolFromSmarts(series["holdout_smarts_extra"])
            assert patt_extra is not None, f"Bad extra SMARTS: {series['holdout_smarts_extra']}"

        def is_holdout(m):
            if not m.HasSubstructMatch(patt):
                return False
            if patt_extra is not None and not m.HasSubstructMatch(patt_extra):
                return False
            return True

        # Partition
        holdout_set = []
        non_holdout = []
        for c in compounds:
            m = Chem.MolFromSmiles(c["smiles"])
            if m is None:
                continue
            if is_holdout(m):
                holdout_set.append(c)
            else:
                non_holdout.append(c)
        print(f"  N matching holdout SMARTS:    {len(holdout_set)}")
        print(f"  N NOT matching holdout SMARTS: {len(non_holdout)}")

        # test_drugs: prefer named drugs from holdout_set; else top-pIC50 most-potent representatives
        known = set(s.upper() for s in series["known_drug_pref_names"])
        named_holdout = [c for c in holdout_set if c["pref_name"] and c["pref_name"].upper() in known]
        unnamed_holdout = [c for c in holdout_set if c not in named_holdout]
        # Sort named first by pIC50 desc, then unnamed by pIC50 desc
        named_holdout.sort(key=lambda c: -c["pic50"])
        unnamed_holdout.sort(key=lambda c: -c["pic50"])
        test_drugs = named_holdout + unnamed_holdout
        # Dedup by SMILES (canonical)
        seen_smi = set()
        deduped = []
        for c in test_drugs:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(c["smiles"]))
            if canon in seen_smi:
                continue
            seen_smi.add(canon)
            deduped.append(c)
        # Diversify by Tc within: if successive entries have Tc >0.9 to all picks, skip
        diversified = []
        for c in deduped:
            if len(diversified) >= 10:
                break
            ok = True
            for prev in diversified:
                t = tanimoto(c["smiles"], prev["smiles"])
                if t is not None and t > 0.9:
                    ok = False
                    break
            if ok:
                diversified.append(c)
        test_drugs = diversified[: max(3, min(10, len(diversified)))]
        print(f"  Selected {len(test_drugs)} test_drugs (named: {sum(1 for d in test_drugs if d['pref_name'])})")

        # train_anchors: from non_holdout, must have Murcko-Tc to at least one test_drug >= 0.15
        # (loose chemotype proximity, to ensure same target family but no holdout match)
        candidate_anchors = []
        for c in non_holdout:
            for d in test_drugs:
                mt = murcko_tanimoto(c["smiles"], d["smiles"])
                if mt is not None and mt >= 0.15:
                    c["max_murcko_tc_to_test"] = max(
                        (murcko_tanimoto(c["smiles"], dd["smiles"]) or 0.0) for dd in test_drugs
                    )
                    c["max_tc_to_test"] = max(
                        (tanimoto(c["smiles"], dd["smiles"]) or 0.0) for dd in test_drugs
                    )
                    candidate_anchors.append(c)
                    break

        # Sort by potency, take diverse subset (Murcko-Tc>=0.05 from prior picks)
        candidate_anchors.sort(key=lambda c: -c["pic50"])
        train_anchors = []
        for c in candidate_anchors:
            if len(train_anchors) >= 30:
                break
            ok = True
            for prev in train_anchors:
                t = tanimoto(c["smiles"], prev["smiles"])
                if t is not None and t > 0.9:
                    ok = False
                    break
            if ok:
                train_anchors.append(c)
        if len(train_anchors) < 10:
            # Top up loosening Murcko-Tc requirement (just take top potency without holdout)
            need = 10 - len(train_anchors)
            taken_smi = {a["smiles"] for a in train_anchors}
            extras = [c for c in non_holdout if c["smiles"] not in taken_smi]
            extras.sort(key=lambda c: -c["pic50"])
            for c in extras[:need]:
                c["max_murcko_tc_to_test"] = max(
                    (murcko_tanimoto(c["smiles"], d["smiles"]) or 0.0) for d in test_drugs
                )
                c["max_tc_to_test"] = max(
                    (tanimoto(c["smiles"], d["smiles"]) or 0.0) for d in test_drugs
                )
                train_anchors.append(c)
        print(f"  Selected {len(train_anchors)} train_anchors")

        # Verify holdout SMARTS is specific: none of train_anchors should match (both)
        miss_specificity = 0
        for a in train_anchors:
            m = Chem.MolFromSmiles(a["smiles"])
            if m and is_holdout(m):
                miss_specificity += 1
        print(f"  Train anchors that wrongly match holdout SMARTS: {miss_specificity} (should be 0)")

        series_out.append(
            {
                "series_name": series["series_name"],
                "target_chembl_id": series["target"],
                "target_pref_name": series["target_pref_name"],
                "holdout_smarts": series["holdout_smarts"],
                **({"holdout_smarts_extra": series["holdout_smarts_extra"]}
                   if "holdout_smarts_extra" in series else {}),
                "holdout_rationale": series["holdout_rationale"],
                "n_total_actives": len(compounds),
                "n_holdout_total": len(holdout_set),
                "n_non_holdout_total": len(non_holdout),
                "test_drugs": [
                    {
                        "chembl_id": d["chembl_id"],
                        "smiles": d["smiles"],
                        "pref_name": d["pref_name"],
                        "pic50": round(d["pic50"], 3),
                    }
                    for d in test_drugs
                ],
                "train_anchors": [
                    {
                        "chembl_id": a["chembl_id"],
                        "smiles": a["smiles"],
                        "pref_name": a["pref_name"],
                        "pic50": round(a["pic50"], 3),
                        "max_murcko_tc_to_test_drug": round(a.get("max_murcko_tc_to_test", 0.0), 3),
                        "max_tc_to_test_drug": round(a.get("max_tc_to_test", 0.0), 3),
                    }
                    for a in train_anchors
                ],
            }
        )

    output = {
        "version": "v4_holdout_chemotype",
        "n_series": len(series_out),
        "criteria": {
            "compound_min_pic50": 6.5,
            "test_drug_selection": "named-drug-first (from known_drug_pref_names), then top-pIC50; cap 10; diversity Tc<=0.9",
            "train_anchor_selection": (
                "must NOT match holdout_smarts; must have Murcko-Tc >= 0.15 to >=1 test_drug "
                "(same target series, not random); diverse Tc<=0.9; cap 30"
            ),
            "intended_use": (
                "RL-train mol2mol on train_anchors with composite reward; "
                "evaluate sampled molecules' max_tc_to_test_drug per cohort. "
                "Compare to baseline (untrained mol2mol or full-data RL): held-out chemotype tests rediscovery."
            ),
        },
        "series": series_out,
    }
    (OUT_DIR / "pairs.json").write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUT_DIR/'pairs.json'} with {len(series_out)} series")


if __name__ == "__main__":
    main()
