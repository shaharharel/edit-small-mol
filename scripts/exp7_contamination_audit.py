"""
exp7 contamination diagnostic.

For each of the 50 LO benchmark pairs (egfr_t790m, btk, jak3, her2, fgfr):
  1. Look up baseline_prior_anchor cohort and find generated mol with max Tc to drug.
  2. Confirm whether the canonical SMILES is identical to the drug SMILES.
  3. Check whether the anchor SMILES appears in the cohort and whether it equals the drug.
  4. Search CovInDB v2 (CovInDB_All.csv) for exact and fuzzy matches (Tc>=0.9).
  5. Search the actual Mol2Mol covalent FT corpus
     (data/reinvent4_mol2mol_covalent_ft_work/covalent_smiles.smi + val) for exact / fuzzy matches.
  6. Compute MMP edit distance from anchor -> drug (rdMMPA).
  7. Categorize each pair: TRIVIAL / MEMORIZED / CHEMBL_NATIVE / CLEAN.

Writes:
  data/exp7_lo_benchmark/contamination_audit.csv
  results/paper_evaluation/exp7_contamination_diagnosis.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMMPA

RDLogger.DisableLog("rdApp.*")

REPO = Path("/Users/shaharharel/Documents/github/edit-small-mol")
BENCH = REPO / "data" / "exp7_lo_benchmark"
RL_DIR = BENCH / "_rl"
COVINDB_CSV = REPO / "data" / "covbinder" / "raw_covindb2" / "CovInDB_All.csv"
FT_CORPUS = REPO / "data" / "reinvent4_mol2mol_covalent_ft_work" / "covalent_smiles.smi"
FT_CORPUS_VAL = REPO / "data" / "reinvent4_mol2mol_covalent_ft_work" / "covalent_smiles_val.smi"

OUT_CSV = BENCH / "contamination_audit.csv"
OUT_MD = REPO / "results" / "paper_evaluation" / "exp7_contamination_diagnosis.md"

FUZZY_TC = 0.90


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def canon(smi: str) -> str | None:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def canon_no_stereo(smi: str) -> str | None:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def tc(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mmp_edits(anchor: str, drug: str) -> int:
    """Count chemically distinct fragmentations between anchor and drug.

    Heuristic: enumerate rdMMPA fragmentations on the union molecule; the smallest
    pair-difference fragment count between matched cores approximates the edit
    distance. Falls back to atom-count diff if MMP fails.
    """
    try:
        a = Chem.MolFromSmiles(anchor)
        d = Chem.MolFromSmiles(drug)
        if a is None or d is None:
            return -1
        # Enumerate MMPA fragmentations
        a_frags = rdMMPA.FragmentMol(a, resultsAsMols=False, maxCuts=3)
        d_frags = rdMMPA.FragmentMol(d, resultsAsMols=False, maxCuts=3)
        if not a_frags or not d_frags:
            return abs(a.GetNumHeavyAtoms() - d.GetNumHeavyAtoms())
        # build core->chain maps; cores are unique per cut pattern
        def _frag_map(frags):
            m = {}
            for core, chains in frags:
                if core is None or chains is None:
                    continue
                # core can be like "" if zero-cut.
                m.setdefault(core, set()).add(chains)
            return m

        amap = _frag_map(a_frags)
        dmap = _frag_map(d_frags)
        # find a common core; count chain differences (symmetric difference)
        best = None
        for core in amap.keys() & dmap.keys():
            chains_a = amap[core]
            chains_d = dmap[core]
            diff = len(chains_a.symmetric_difference(chains_d))
            if diff == 0:
                # identical chains given same core fragmentation → same molecule
                if anchor != drug:
                    diff = 0  # but molecules differ → keep searching
            if best is None or diff < best:
                best = diff
        if best is None:
            return abs(a.GetNumHeavyAtoms() - d.GetNumHeavyAtoms())
        # Each chain in symmetric difference contributes 1 to "fragments differ".
        # An "edit" = 1 swap = 2 chains differ (one removed, one added).
        return max(1, best // 2)
    except Exception:
        return -1


# ----------------------------------------------------------------------------
# Load benchmark pairs (50 pairs)
# ----------------------------------------------------------------------------
def load_pairs() -> list[dict]:
    """Normalise pairs from all 5 target files into a uniform schema."""
    pairs = []

    # EGFR T790M
    j = json.loads((BENCH / "egfr_t790m_pairs.json").read_text())
    for p in j["pairs"]:
        pairs.append({
            "pair_id": p["pair_id"],
            "target": "EGFR_T790M",
            "drug_name": p["drug"]["name"],
            "drug_smiles": p["drug"]["smiles"],
            "anchor_smiles": p["anchor"]["smiles"],
            "cohort_dir_basename": p["pair_id"] + "_baseline_prior_anchor",
        })

    # BTK
    j = json.loads((BENCH / "btk_pairs.json").read_text())
    for p in j["pairs"]:
        pairs.append({
            "pair_id": p["pair_id"],
            "target": "BTK",
            "drug_name": p["drug"]["name"],
            "drug_smiles": p["drug"]["smiles"],
            "anchor_smiles": p["anchor"]["smiles"],
            "cohort_dir_basename": p["pair_id"] + "_baseline_prior_anchor",
        })

    # JAK3
    j = json.loads((BENCH / "jak3_pairs.json").read_text())
    drug_smi = j["drug"]["smiles"]
    drug_name = j["drug"]["name"]
    for p in j["pairs"]:
        pairs.append({
            "pair_id": p["pair_id"],
            "target": "JAK3",
            "drug_name": drug_name,
            "drug_smiles": drug_smi,
            "anchor_smiles": p["anchor"]["smiles"],
            "cohort_dir_basename": p["pair_id"] + "_baseline_prior_anchor",
        })

    # HER2 (single flat list of pairs with anchor_smiles / drug_smiles at top level)
    j = json.loads((BENCH / "her2_pairs.json").read_text())
    her2_list = j["pairs"] if isinstance(j, dict) and "pairs" in j else j
    for i, p in enumerate(her2_list, 1):
        # try several conventions
        if "pair_id" in p:
            pid = p["pair_id"]
        else:
            pid = f"her2_pair_{i:03d}"
        anchor = p.get("anchor_smiles") or p.get("anchor", {}).get("smiles")
        drug = p.get("drug_smiles") or p.get("drug", {}).get("smiles")
        drug_name = p.get("drug_name") or p.get("drug", {}).get("name") or ""
        pairs.append({
            "pair_id": pid,
            "target": "HER2",
            "drug_name": drug_name,
            "drug_smiles": drug,
            "anchor_smiles": anchor,
            "cohort_dir_basename": pid + "_baseline_prior_anchor",
        })

    # FGFR (flat list)
    j = json.loads((BENCH / "fgfr_pairs.json").read_text())
    fgfr_list = j if isinstance(j, list) else j.get("pairs", [])
    for p in fgfr_list:
        pid = p["pair_id"]
        pairs.append({
            "pair_id": pid,
            "target": "FGFR",
            "drug_name": p.get("drug_name", ""),
            "drug_smiles": p["drug_smiles"],
            "anchor_smiles": p["anchor_smiles"],
            "cohort_dir_basename": pid + "_baseline_prior_anchor",
        })

    return pairs


# ----------------------------------------------------------------------------
# Load reference corpora
# ----------------------------------------------------------------------------
def load_covindb_smiles() -> tuple[set[str], set[str], list[tuple[str, object]]]:
    """Return (canon set with stereo, canon set without stereo, list of (canon, fp) for Tc)."""
    df = pd.read_csv(COVINDB_CSV, low_memory=False)
    if "SMILES" not in df.columns:
        raise RuntimeError(f"No SMILES column in {COVINDB_CSV}; got {df.columns.tolist()}")
    canon_set, canon_ns_set, fps = set(), set(), []
    for smi in df["SMILES"].dropna().astype(str):
        c = canon(smi)
        if c:
            canon_set.add(c)
        cns = canon_no_stereo(smi)
        if cns:
            canon_ns_set.add(cns)
        fp = morgan_fp(smi)
        if fp is not None and c:
            fps.append((c, fp))
    return canon_set, canon_ns_set, fps


def load_ft_corpus() -> tuple[set[str], set[str], list[tuple[str, object]]]:
    smis = []
    for path in (FT_CORPUS, FT_CORPUS_VAL):
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    smis.append(line.split()[0])
    canon_set, canon_ns_set, fps = set(), set(), []
    for smi in smis:
        c = canon(smi)
        if c:
            canon_set.add(c)
        cns = canon_no_stereo(smi)
        if cns:
            canon_ns_set.add(cns)
        fp = morgan_fp(smi)
        if fp is not None and c:
            fps.append((c, fp))
    return canon_set, canon_ns_set, fps


# ----------------------------------------------------------------------------
# Cohort analysis
# ----------------------------------------------------------------------------
def analyze_cohort(cohort_dir: Path, drug_canon: str, drug_canon_ns: str, drug_fp, anchor_canon: str) -> dict:
    """Read sampled.csv; find max-Tc-to-drug row; report identity / anchor inclusion."""
    sampled_csv = cohort_dir / "sampled.csv"
    info = {
        "cohort_exists": cohort_dir.exists(),
        "sampled_rows": 0,
        "max_tc_smiles": "",
        "max_tc_canon": "",
        "max_tc_value": 0.0,
        "max_tc_is_drug_exact": False,
        "max_tc_is_drug_no_stereo": False,
        "anchor_in_cohort_input": False,
        "anchor_equals_drug": (anchor_canon == drug_canon if anchor_canon and drug_canon else False),
        "n_exact_drug_in_cohort": 0,
    }
    if not sampled_csv.exists():
        return info
    try:
        df = pd.read_csv(sampled_csv)
    except Exception:
        return info
    if "SMILES" not in df.columns:
        return info
    info["sampled_rows"] = len(df)
    # anchor input
    if "Input_SMILES" in df.columns and not df.empty:
        in_canon = canon(str(df["Input_SMILES"].iloc[0]))
        info["anchor_in_cohort_input"] = (in_canon == anchor_canon)
        info["input_smiles_canon"] = in_canon or ""
    # iterate
    max_tc = 0.0
    max_smi = ""
    max_canon = ""
    exact_count = 0
    for smi in df["SMILES"].dropna().astype(str):
        c = canon(smi)
        if not c:
            continue
        if c == drug_canon:
            exact_count += 1
        fp = morgan_fp(smi)
        t = tc(fp, drug_fp)
        if t > max_tc:
            max_tc = t
            max_smi = smi
            max_canon = c
    info["max_tc_smiles"] = max_smi
    info["max_tc_canon"] = max_canon
    info["max_tc_value"] = max_tc
    info["max_tc_is_drug_exact"] = (max_canon == drug_canon)
    info["max_tc_is_drug_no_stereo"] = (canon_no_stereo(max_smi) == drug_canon_ns) if max_smi else False
    info["n_exact_drug_in_cohort"] = exact_count
    return info


def best_fuzzy(drug_fp, ref_fps: list[tuple[str, object]]) -> tuple[float, str]:
    best, best_smi = 0.0, ""
    for c, fp in ref_fps:
        t = tc(drug_fp, fp)
        if t > best:
            best = t
            best_smi = c
    return best, best_smi


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("[load] pairs ...", flush=True)
    pairs = load_pairs()
    print(f"[load] {len(pairs)} pairs", flush=True)

    print("[load] CovInDB v2 corpus ...", flush=True)
    cov_canon, cov_canon_ns, cov_fps = load_covindb_smiles()
    print(f"[load] CovInDB v2: {len(cov_canon)} unique canonical SMILES", flush=True)

    print("[load] Mol2Mol covalent FT corpus ...", flush=True)
    ft_canon, ft_canon_ns, ft_fps = load_ft_corpus()
    print(f"[load] FT corpus: {len(ft_canon)} unique canonical SMILES", flush=True)

    rows = []
    for i, p in enumerate(pairs, 1):
        pid = p["pair_id"]
        drug_smi = p["drug_smiles"]
        anchor_smi = p["anchor_smiles"]
        drug_canon = canon(drug_smi) or ""
        drug_canon_ns = canon_no_stereo(drug_smi) or ""
        anchor_canon = canon(anchor_smi) or ""
        drug_fp = morgan_fp(drug_smi)

        # cohort
        cohort_dir = RL_DIR / p["cohort_dir_basename"]
        cohort = analyze_cohort(cohort_dir, drug_canon, drug_canon_ns, drug_fp, anchor_canon)

        # CovInDB v2 audit
        in_cov = drug_canon in cov_canon
        in_cov_ns = drug_canon_ns in cov_canon_ns
        max_tc_cov, max_tc_cov_smi = best_fuzzy(drug_fp, cov_fps)

        # FT corpus audit
        in_ft = drug_canon in ft_canon
        in_ft_ns = drug_canon_ns in ft_canon_ns
        max_tc_ft, max_tc_ft_smi = best_fuzzy(drug_fp, ft_fps)

        # MMP edit distance anchor->drug
        edits = mmp_edits(anchor_smi, drug_smi)

        # Categorize. Order matters: MEMORIZED dominates if drug is literally in FT;
        # TRIVIAL dominates only if drug is NOT memorized but anchor is ~one edit away.
        category = "CLEAN"
        evidence_bits = []
        if in_ft:
            category = "MEMORIZED"
            evidence_bits.append("drug exact in FT corpus")
        elif in_cov:
            category = "MEMORIZED"
            evidence_bits.append("drug exact in CovInDB v2 (likely in prior training)")
        elif in_ft_ns:
            category = "MEMORIZED"
            evidence_bits.append("drug stereo-blind match in FT corpus")
        elif in_cov_ns:
            category = "CHEMBL_NATIVE"
            evidence_bits.append(f"drug stereo-blind match in CovInDB v2")
        elif max_tc_cov >= FUZZY_TC:
            category = "CHEMBL_NATIVE"
            evidence_bits.append(f"CovInDB v2 near-neighbor Tc={max_tc_cov:.3f}")
        elif max_tc_ft >= FUZZY_TC:
            category = "MEMORIZED"
            evidence_bits.append(f"FT corpus near-neighbor Tc={max_tc_ft:.3f}")
        # TRIVIAL is layered on: even if not memorized, a 1-edit anchor still makes any
        # prior trivially recover it. So TRIVIAL takes priority over CLEAN, and is
        # additionally flagged when it co-occurs with MEMORIZED.
        if edits >= 0 and edits <= 1:
            evidence_bits.append(f"anchor->drug MMP edit distance ~{edits}")
            if category == "CLEAN":
                category = "TRIVIAL"

        # Anchor-equals-drug bug check
        if cohort.get("anchor_equals_drug"):
            category = "ANCHOR_IS_DRUG_BUG"
            evidence_bits.insert(0, "ANCHOR == DRUG (canon)")

        rows.append({
            "pair_id": pid,
            "target": p["target"],
            "drug_name": p.get("drug_name", ""),
            "drug_smiles": drug_smi,
            "anchor_smiles": anchor_smi,
            "category": category,
            "in_covindb_v2_exact": in_cov,
            "in_covindb_v2_stereoblind": in_cov_ns,
            "max_tc_covindb_v2": round(max_tc_cov, 4),
            "max_tc_covindb_v2_match": max_tc_cov_smi,
            "in_ft_corpus_exact": in_ft,
            "in_ft_corpus_stereoblind": in_ft_ns,
            "max_tc_ft_corpus": round(max_tc_ft, 4),
            "max_tc_ft_corpus_match": max_tc_ft_smi,
            "anchor_drug_mmp_edits": edits,
            "cohort_max_tc_to_drug": round(cohort.get("max_tc_value", 0.0), 4),
            "cohort_max_tc_is_drug_exact": cohort.get("max_tc_is_drug_exact", False),
            "cohort_max_tc_is_drug_no_stereo": cohort.get("max_tc_is_drug_no_stereo", False),
            "cohort_max_tc_smiles": cohort.get("max_tc_smiles", ""),
            "n_drug_exact_in_cohort": cohort.get("n_exact_drug_in_cohort", 0),
            "anchor_in_cohort_input": cohort.get("anchor_in_cohort_input", False),
            "anchor_equals_drug": cohort.get("anchor_equals_drug", False),
            "cohort_sampled_rows": cohort.get("sampled_rows", 0),
            "cohort_exists": cohort.get("cohort_exists", False),
            "evidence": "; ".join(evidence_bits) if evidence_bits else "no contamination signal",
        })
        if i % 5 == 0 or i == len(pairs):
            print(f"[scan] {i}/{len(pairs)} pairs processed", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}", flush=True)

    # ------------------ MD report ------------------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# exp7 prior-baseline contamination diagnosis")
    lines.append("")
    lines.append(f"Audit of {len(df)} curated LO benchmark pairs (egfr_t790m / btk / jak3 / her2 / fgfr).")
    lines.append("")
    lines.append("Sources:")
    lines.append(f"- CovInDB v2: `{COVINDB_CSV}` ({len(cov_canon)} unique canonical SMILES)")
    lines.append(f"- Mol2Mol covalent FT corpus: `{FT_CORPUS}` + val ({len(ft_canon)} unique canonical SMILES)")
    lines.append(f"- Cohort sampling dirs: `{RL_DIR}/<pair_id>_baseline_prior_anchor/sampled.csv`")
    lines.append("")
    lines.append("Categorization rules (first matching rule wins, then TRIVIAL annotation overrides CLEAN):")
    lines.append("1. ANCHOR_IS_DRUG_BUG: canonical anchor == canonical drug.")
    lines.append("2. MEMORIZED: drug canonical SMILES (or stereo-blind) appears in Mol2Mol covalent FT corpus, "
                 "OR FT corpus has a Tc>=0.90 near-neighbor.")
    lines.append("3. MEMORIZED (CovInDB): drug exact-matches CovInDB v2.")
    lines.append("4. CHEMBL_NATIVE: drug is a stereo-blind match in CovInDB v2, OR CovInDB v2 has a Tc>=0.90 near-neighbor "
                 "(known molecule, the FT prior may have seen it indirectly).")
    lines.append("5. TRIVIAL: anchor is <=1 MMP edit from drug (any Mol2Mol prior can hit it without memorization).")
    lines.append("6. CLEAN: drug not in CovInDB v2, low Tc to FT corpus, anchor needs >=2 edits.")
    lines.append("")

    # aggregate
    lines.append("## Aggregate counts")
    lines.append("")
    counts = df["category"].value_counts().to_dict()
    for cat in ["CLEAN", "CHEMBL_NATIVE", "TRIVIAL", "MEMORIZED", "ANCHOR_IS_DRUG_BUG"]:
        lines.append(f"- **{cat}**: {counts.get(cat, 0)}")
    lines.append("")
    lines.append("Per target:")
    lines.append("")
    pt = df.groupby(["target", "category"]).size().unstack(fill_value=0)
    lines.append("```")
    lines.append(pt.to_string())
    lines.append("```")
    lines.append("")

    # The 4 Tc=1.0 hits
    lines.append("## The four Tc=1.000 baseline_prior_anchor hits")
    lines.append("")
    spotlight = ["JAK3_T01_4083165", "JAK3_T02_4062806", "FGFR_P07", "FGFR_P08"]
    for pid in spotlight:
        row = df[df["pair_id"] == pid]
        if row.empty:
            lines.append(f"- **{pid}**: NO MATCHING ROW (pair_id mismatch)")
            continue
        r = row.iloc[0]
        lines.append(f"### {pid} ({r['target']} → {r['drug_name']})")
        lines.append("")
        lines.append(f"- Category: **{r['category']}**")
        lines.append(f"- Cohort max Tc to drug: {r['cohort_max_tc_to_drug']:.4f}")
        lines.append(f"- Cohort max-Tc mol IS drug (exact canonical): **{r['cohort_max_tc_is_drug_exact']}**")
        lines.append(f"- Cohort max-Tc mol IS drug (stereo-blind): {r['cohort_max_tc_is_drug_no_stereo']}")
        lines.append(f"- # exact-drug copies in cohort: {r['n_drug_exact_in_cohort']}")
        lines.append(f"- Anchor SMILES present as Input_SMILES: {r['anchor_in_cohort_input']}")
        lines.append(f"- Anchor == drug (bug check): {r['anchor_equals_drug']}")
        lines.append(f"- Drug in CovInDB v2 (exact): {r['in_covindb_v2_exact']}; "
                     f"stereo-blind: {r['in_covindb_v2_stereoblind']}; "
                     f"max-Tc in CovInDB: {r['max_tc_covindb_v2']}")
        lines.append(f"- Drug in FT corpus (exact): {r['in_ft_corpus_exact']}; "
                     f"stereo-blind: {r['in_ft_corpus_stereoblind']}; "
                     f"max-Tc in FT: {r['max_tc_ft_corpus']}")
        lines.append(f"- Anchor → drug MMP edit distance: {r['anchor_drug_mmp_edits']}")
        lines.append(f"- Evidence: {r['evidence']}")
        lines.append("")

    # Recommended clean subset
    lines.append("## Recommended clean subset for paper claims")
    lines.append("")
    clean = df[df["category"].isin(["CLEAN", "CHEMBL_NATIVE"])]
    lines.append(f"Allowed categories: CLEAN + CHEMBL_NATIVE = **{len(clean)}/{len(df)}** pairs.")
    lines.append("(Exclude TRIVIAL — any prior recovers a 1-edit drug. Exclude MEMORIZED — drug is in the FT corpus.")
    lines.append(" Exclude ANCHOR_IS_DRUG_BUG — pipeline bug, not a model behavior.)")
    lines.append("")
    if not clean.empty:
        lines.append("Per target in clean subset:")
        lines.append("```")
        lines.append(clean.groupby("target").size().to_string())
        lines.append("```")
        lines.append("")
        lines.append("Clean pair IDs:")
        for t, sub in clean.groupby("target"):
            ids = ", ".join(sorted(sub["pair_id"]))
            lines.append(f"- {t}: {ids}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Full table")
    lines.append("")
    lines.append("See `data/exp7_lo_benchmark/contamination_audit.csv`.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"[save] {OUT_MD}", flush=True)
    print(f"[done] CLEAN={counts.get('CLEAN',0)} CHEMBL_NATIVE={counts.get('CHEMBL_NATIVE',0)} "
          f"TRIVIAL={counts.get('TRIVIAL',0)} MEMORIZED={counts.get('MEMORIZED',0)} "
          f"ANCHOR_BUG={counts.get('ANCHOR_IS_DRUG_BUG',0)}", flush=True)


if __name__ == "__main__":
    main()
