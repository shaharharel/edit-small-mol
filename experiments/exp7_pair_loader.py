#!/usr/bin/env python3
"""EXP7 LO eval — load all 50 curated pairs into a unified schema.

Each JSON file has slightly different conventions. Output: List[Dict] with keys
    pair_id, target_key, target_chembl_ids (list[str]), target_label,
    anchor_smiles, anchor_name, anchor_chembl_id, anchor_pIC50,
    drug_smiles, drug_name, drug_chembl_id, drug_pIC50,
    tc_anchor_drug, delta_pIC50,
    successors_smiles (list[str]),     # exp7 exclude policy = drug + named successors only
    successors_names (list[str]).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List


PAIR_DIR = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/exp7_lo_benchmark")

# Named clinical successors per target (paper Exp6 policy + spec doc).
NAMED_SUCCESSORS = {
    "egfr_t790m": {
        "rociletinib":  "C=CC(=O)Nc1cc(Nc2nc(Nc3ccc(N4CCN(C)C(=O)C4)cc3)nc(C(F)(F)F)c2)c(OC)cc1N1CCOCC1",
        "naquotinib":   "C=CC(=O)N(C)c1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N1CCN(C(C)C)CC1",
        "lazertinib":   "C=CC(=O)Nc1cc(Nc2ncc(Cl)c(Oc3ccc(N4CCN(C)CC4)c(OC)c3)n2)c(OC)cc1N1CCOCC1",
        "mavelertinib": "C=CC(=O)Nc1cccc(Nc2nccc(-c3cccnc3)n2)c1",
    },
    "btk": {
        "zanubrutinib":  "C=CC(=O)N1CCC[C@H]1Cn1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
        "acalabrutinib": "CC#CC(=O)N1CCC[C@H]1c1ncc(-c2ccc(C(=O)Nc3ccccn3)cc2)n1-c1ncccn1",
        "tirabrutinib":  "C=CC(=O)N1CCCC1Cn1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
        "spebrutinib":   "C=CC(=O)Nc1ccc(C(=O)NC2CCCCC2)cc1Nc1ncc(Cl)c(Nc2ccccc2)n1",
        # ibrutinib itself is the drug for some pairs; handled at canonicalization level
    },
    "jak3": {
        # ritlecitinib (drug) — successors are pretty rare; include known JAK3 covalent leads
        # ritlecitinib was first-in-class; no FDA successors yet (as of 2026)
    },
    "her2": {
        # afatinib drug; HER2-active named successors:
        "neratinib":     "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
        "dacomitinib":   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
        "mobocertinib":  "CC(C)OC(=O)c1cnc(Nc2ccc(N(C)CCN(C)C)c(OC)c2)nc1NC(=O)/C=C/C",
    },
    "fgfr": {
        # Drug varies per pair — futibatinib/erdafitinib/pemigatinib/infigratinib/roblitinib
        # Include the major FGFR covalent and reversible kinase drugs:
        "futibatinib":   "C=CC(=O)N1CC[C@H](n2nc(C#Cc3cc(OC)cc(OC)c3)c3c(N)ncnc32)C1",
        "fisogatinib":   "C=CC(=O)Nc1ccccc1Nc1ncc2cc(-c3c(Cl)c(OC)cc(OC)c3Cl)ccc2n1",
        "roblitinib":    "COCCNc1cc(NC(=O)N2CCCc3cc(CN4CCN(C)CC4=O)c(C=O)nc32)ncc1C#N",
    },
}

# Target ChEMBL ID mapping
TARGET_CHEMBL = {
    "egfr_t790m": (["CHEMBL203"], "EGFR T790M"),
    "btk":        (["CHEMBL5251"], "BTK Cys481"),
    "jak3":       (["CHEMBL2148"], "JAK3 Cys909"),
    "her2":       (["CHEMBL1824", "CHEMBL203"], "HER2 / EGFR (pan-ErbB)"),
    "fgfr":       (["CHEMBL2742", "CHEMBL4296", "CHEMBL3650", "CHEMBL3717"], "FGFR1-4"),
}


def _load_egfr_t790m() -> List[Dict]:
    d = json.load(open(PAIR_DIR / "egfr_t790m_pairs.json"))
    out = []
    for p in d["pairs"]:
        a, dr = p["anchor"], p["drug"]
        out.append(dict(
            pair_id=p["pair_id"], target_key="egfr_t790m",
            target_chembl_ids=TARGET_CHEMBL["egfr_t790m"][0],
            target_label=TARGET_CHEMBL["egfr_t790m"][1],
            anchor_smiles=a["smiles"], anchor_name=a["name"],
            anchor_chembl_id=a.get("chembl_id"),
            anchor_pIC50=a.get("pIC50_T790M_H1975", float("nan")),
            drug_smiles=dr["smiles"], drug_name=dr["name"],
            drug_chembl_id=dr.get("chembl_id"),
            drug_pIC50=dr.get("pIC50_T790M_H1975", float("nan")),
            tc_anchor_drug=p.get("tc_anchor_drug", float("nan")),
            delta_pIC50=p.get("delta_pic50", float("nan")),
            successors_smiles=list(NAMED_SUCCESSORS["egfr_t790m"].values()),
            successors_names=list(NAMED_SUCCESSORS["egfr_t790m"].keys()),
        ))
    return out


def _load_btk() -> List[Dict]:
    d = json.load(open(PAIR_DIR / "btk_pairs.json"))
    out = []
    for p in d["pairs"]:
        a, dr = p["anchor"], p["drug"]
        out.append(dict(
            pair_id=p["pair_id"], target_key="btk",
            target_chembl_ids=TARGET_CHEMBL["btk"][0],
            target_label=TARGET_CHEMBL["btk"][1],
            anchor_smiles=a["smiles"], anchor_name=a["name"],
            anchor_chembl_id=a.get("chembl_id"),
            anchor_pIC50=a.get("pIC50_BTK", float("nan")),
            drug_smiles=dr["smiles"], drug_name=dr["name"],
            drug_chembl_id=dr.get("chembl_id"),
            drug_pIC50=dr.get("pIC50_BTK", float("nan")),
            tc_anchor_drug=p.get("tc_anchor_drug", float("nan")),
            delta_pIC50=p.get("delta_pic50", float("nan")),
            successors_smiles=list(NAMED_SUCCESSORS["btk"].values()),
            successors_names=list(NAMED_SUCCESSORS["btk"].keys()),
        ))
    return out


def _load_jak3() -> List[Dict]:
    d = json.load(open(PAIR_DIR / "jak3_pairs.json"))
    drug = d["drug"]
    out = []
    for p in d["pairs"]:
        a = p["anchor"]
        out.append(dict(
            pair_id=p["pair_id"], target_key="jak3",
            target_chembl_ids=TARGET_CHEMBL["jak3"][0],
            target_label=TARGET_CHEMBL["jak3"][1],
            anchor_smiles=a["smiles"], anchor_name=a.get("chembl_id", "?"),
            anchor_chembl_id=a.get("chembl_id"),
            anchor_pIC50=a.get("pIC50", float("nan")),
            drug_smiles=drug["smiles"], drug_name=drug["name"],
            drug_chembl_id=drug.get("chembl_id"),
            drug_pIC50=p["drug"].get("pIC50", float("nan")),
            tc_anchor_drug=p.get("tc_morgan_r2_2048", float("nan")),
            delta_pIC50=p.get("delta_pIC50", float("nan")),
            successors_smiles=list(NAMED_SUCCESSORS["jak3"].values()),
            successors_names=list(NAMED_SUCCESSORS["jak3"].keys()),
        ))
    return out


def _load_her2() -> List[Dict]:
    d = json.load(open(PAIR_DIR / "her2_pairs.json"))
    out = []
    for p in d["pairs"]:
        out.append(dict(
            pair_id=p.get("anchor_name","") + "->" + p.get("drug_name",""),
            target_key="her2",
            target_chembl_ids=TARGET_CHEMBL["her2"][0],
            target_label=TARGET_CHEMBL["her2"][1],
            anchor_smiles=p["anchor_smiles"], anchor_name=p["anchor_name"],
            anchor_chembl_id=p.get("anchor_chembl_id"),
            anchor_pIC50=p.get("anchor_pIC50_HER2", p.get("anchor_pIC50_EGFR", float("nan"))),
            drug_smiles=p["drug_smiles"], drug_name=p["drug_name"],
            drug_chembl_id=p.get("drug_chembl_id"),
            drug_pIC50=p.get("drug_pIC50_HER2", p.get("drug_pIC50_EGFR", float("nan"))),
            tc_anchor_drug=p.get("tc_morgan_r2_2048", float("nan")),
            delta_pIC50=p.get("delta_pIC50_HER2", p.get("delta_pIC50_EGFR", float("nan"))),
            successors_smiles=list(NAMED_SUCCESSORS["her2"].values()),
            successors_names=list(NAMED_SUCCESSORS["her2"].keys()),
        ))
    return out


def _load_fgfr() -> List[Dict]:
    d = json.load(open(PAIR_DIR / "fgfr_pairs.json"))
    out = []
    for p in d:
        out.append(dict(
            pair_id=p["pair_id"], target_key="fgfr",
            target_chembl_ids=TARGET_CHEMBL["fgfr"][0],
            target_label=p.get("target_name", "FGFR"),
            anchor_smiles=p["anchor_smiles"], anchor_name=p.get("anchor_id", "?"),
            anchor_chembl_id=p.get("anchor_id"),
            anchor_pIC50=p.get("anchor_pIC50_max", float("nan")),
            drug_smiles=p["drug_smiles"], drug_name=p["drug_name"],
            drug_chembl_id=p.get("drug_id"),
            drug_pIC50=p.get("drug_pIC50_max", float("nan")),
            tc_anchor_drug=p.get("tanimoto", float("nan")),
            delta_pIC50=p.get("delta_pIC50_max", float("nan")),
            successors_smiles=list(NAMED_SUCCESSORS["fgfr"].values()),
            successors_names=list(NAMED_SUCCESSORS["fgfr"].keys()),
        ))
    return out


def load_all_pairs() -> List[Dict]:
    pairs = []
    pairs.extend(_load_egfr_t790m())
    pairs.extend(_load_btk())
    pairs.extend(_load_jak3())
    pairs.extend(_load_her2())
    pairs.extend(_load_fgfr())
    return pairs


if __name__ == "__main__":
    pairs = load_all_pairs()
    print(f"Total: {len(pairs)} pairs across {len(set(p['target_key'] for p in pairs))} targets")
    by_target = {}
    for p in pairs:
        by_target.setdefault(p["target_key"], []).append(p)
    for tk, ps in by_target.items():
        print(f"  {tk}: {len(ps)} pairs")
    # sanity
    for p in pairs[:3]:
        print(f"  -> {p['pair_id']}: tc={p['tc_anchor_drug']:.2f}, Δ={p['delta_pIC50']:.2f}, drug={p['drug_name']}")
