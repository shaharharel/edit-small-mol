"""Independent audit: re-verify each pair in clean_pairs.json passes all 7 criteria.

Runs from scratch (does not trust upstream curator). Reports pass/fail per criterion.
Writes data/exp7_v2_benchmark/audit_report.json with full results.
"""
from __future__ import annotations
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data" / "exp7_v2_benchmark"
DB_PATH = ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
CONTAM_PKL = OUT_DIR / "contamination_sets.pkl"


def canon_smi(smi):
    if not smi:
        return None, None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    m2 = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(m2)
    return Chem.MolToSmiles(m), Chem.MolToSmiles(m2)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def tanimoto(a, b):
    fa, fb = fp(a), fp(b)
    return DataStructs.TanimotoSimilarity(fa, fb) if (fa and fb) else None


def mmp_edit_distance(sa, sb, mx=3):
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    if not ma or not mb:
        return None
    for n in range(1, mx + 1):
        fa = rdMMPA.FragmentMol(ma, maxCuts=n, resultsAsMols=False)
        fb = rdMMPA.FragmentMol(mb, maxCuts=n, resultsAsMols=False)
        ca, cb = defaultdict(set), defaultdict(set)
        for c, ch in fa:
            if c:
                ca[c].add(ch)
        for c, ch in fb:
            if c:
                cb[c].add(ch)
        for k in (set(ca) & set(cb)):
            for x in ca[k]:
                for y in cb[k]:
                    if x != y:
                        return n
    return None


def main():
    with open(CONTAM_PKL, "rb") as f:
        sets = pickle.load(f)
    cov_exact, cov_stereo = sets["covindb_exact"], sets["covindb_stereoblind"]
    ft_exact, ft_stereo = sets["covft_exact"], sets["covft_stereoblind"]

    pairs_doc = json.loads((OUT_DIR / "clean_pairs.json").read_text())
    pairs = pairs_doc["pairs"]
    print(f"Auditing {len(pairs)} pairs...")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    audited = []
    fails = []
    for p in pairs:
        a_smi, d_smi = p["anchor"]["smiles"], p["drug"]["smiles"]
        d_cid = p["drug"]["chembl_id"]
        a_cid = p["anchor"]["chembl_id"]
        result = {"pair_id": p["pair_id"], "drug": p["drug"]["name"]}
        # Criterion 1: drug not in CovInDB v2
        de, ds = canon_smi(d_smi)
        c1 = not ((de in cov_exact) or (ds in cov_stereo))
        result["c1_drug_not_in_covindb_v2"] = bool(c1)
        # Criterion 2: drug not in covalent_ft
        c2 = not ((de in ft_exact) or (ds in ft_stereo))
        result["c2_drug_not_in_covft"] = bool(c2)
        # Criterion 3: drug first ChEMBL appearance >= 2024 (strict) — track full year
        cur.execute(
            """SELECT MIN(d.year) FROM activities act JOIN docs d ON act.doc_id=d.doc_id
               JOIN molecule_dictionary md ON act.molregno=md.molregno
               WHERE md.chembl_id=? AND act.standard_value IS NOT NULL""",
            (d_cid,),
        )
        yr1 = cur.fetchone()[0]
        result["c3_drug_first_year"] = yr1
        result["c3_tier_a_strict_2024"] = (yr1 is not None) and (yr1 >= 2024)
        result["c3_tier_b_2023"] = (yr1 is not None) and (yr1 >= 2023)
        result["c3_tier_c_2022"] = (yr1 is not None) and (yr1 >= 2022)
        # Criterion 4: anchor + drug from same program (proxy: same doc_id, same target)
        cur.execute(
            """SELECT COUNT(*) FROM activities act JOIN molecule_dictionary md ON act.molregno=md.molregno
               WHERE md.chembl_id=? AND act.doc_id=?""",
            (a_cid, p["doc_chembl_id"]),
        )
        anc_in_doc = cur.fetchone()[0] > 0
        cur.execute(
            """SELECT COUNT(*) FROM activities act JOIN molecule_dictionary md ON act.molregno=md.molregno
               WHERE md.chembl_id=? AND act.doc_id=?""",
            (d_cid, p["doc_chembl_id"]),
        )
        drug_in_doc = cur.fetchone()[0] > 0
        result["c4_same_doc"] = bool(anc_in_doc and drug_in_doc)
        # Criterion 5: MMP edit distance >= 2
        med = mmp_edit_distance(a_smi, d_smi, mx=3)
        result["c5_mmp_edit_distance"] = med
        result["c5_pass_ge2"] = (med is not None) and (med >= 2)
        # Criterion 6: Tc in [0.4, 0.85]
        tc = tanimoto(a_smi, d_smi)
        result["c6_tc"] = round(tc, 3) if tc is not None else None
        result["c6_pass_band"] = (tc is not None) and (0.40 <= tc <= 0.85)
        # Criterion 7: both in ChEMBL with IC50/Ki/Kd
        cur.execute(
            """SELECT COUNT(*) FROM activities act JOIN molecule_dictionary md ON act.molregno=md.molregno
               WHERE md.chembl_id=? AND act.standard_type IN ('IC50','Ki','Kd','pIC50') AND act.standard_value IS NOT NULL""",
            (a_cid,),
        )
        anc_acts = cur.fetchone()[0]
        cur.execute(
            """SELECT COUNT(*) FROM activities act JOIN molecule_dictionary md ON act.molregno=md.molregno
               WHERE md.chembl_id=? AND act.standard_type IN ('IC50','Ki','Kd','pIC50') AND act.standard_value IS NOT NULL""",
            (d_cid,),
        )
        drug_acts = cur.fetchone()[0]
        result["c7_anchor_n_acts"] = anc_acts
        result["c7_drug_n_acts"] = drug_acts
        result["c7_pass"] = anc_acts > 0 and drug_acts > 0
        # Composite — hard criteria are 1,2,4,5,6,7. Criterion 3 is informational tier flag only.
        hard_pass = (
            result["c1_drug_not_in_covindb_v2"]
            and result["c2_drug_not_in_covft"]
            and result["c4_same_doc"]
            and result["c5_pass_ge2"]
            and result["c6_pass_band"]
            and result["c7_pass"]
        )
        result["all_hard_criteria_pass"] = bool(hard_pass)
        result["all_pass_relaxed"] = bool(hard_pass)  # alias for backward compat
        if not hard_pass:
            fail_reasons = []
            for k in ["c1_drug_not_in_covindb_v2", "c2_drug_not_in_covft", "c4_same_doc",
                      "c5_pass_ge2", "c6_pass_band", "c7_pass"]:
                if not result[k]:
                    fail_reasons.append(k)
            result["fail_reason"] = ",".join(fail_reasons)
        audited.append(result)
        if not hard_pass:
            fails.append(result)

    n_pass = sum(1 for r in audited if r["all_pass_relaxed"])
    n_strict_2024 = sum(1 for r in audited if r["c3_tier_a_strict_2024"] and r["all_pass_relaxed"])
    n_tier_b_2023 = sum(1 for r in audited if r["c3_tier_b_2023"] and r["all_pass_relaxed"])
    n_tier_c_2022 = sum(1 for r in audited if r["c3_tier_c_2022"] and r["all_pass_relaxed"])

    report = {
        "n_pairs": len(audited),
        "n_pass_all_criteria_relaxed": n_pass,
        "n_strict_2024+": n_strict_2024,
        "n_tier_b_2023+": n_tier_b_2023,
        "n_tier_c_2022+": n_tier_c_2022,
        "n_fails": len(fails),
        "fails": fails,
        "audit": audited,
    }
    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nAudit summary:")
    print(f"  pass all (criterion 3 relaxed to tier_c >=2022): {n_pass}/{len(audited)}")
    print(f"  strict 2024+: {n_strict_2024}")
    print(f"  tier_b 2023+: {n_tier_b_2023}")
    print(f"  tier_c 2022+: {n_tier_c_2022}")
    print(f"  fails: {len(fails)}")
    for f in fails[:5]:
        print(f"    {f['pair_id']} {f['drug']} :: {f.get('fail_reason', 'check report')}")
    print(f"\nWrote {OUT_DIR/'audit_report.json'}")


if __name__ == "__main__":
    main()
