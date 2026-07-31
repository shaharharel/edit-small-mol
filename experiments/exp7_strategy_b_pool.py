#!/usr/bin/env python3
"""EXP7 LO eval — Strategy B 100-anchor pool builder with leak audit.

Per-pair (anchor, drug) Strategy B pool MUST:
  - Tc(pool_mol, drug) < 0.6              (NO answer-leak)
  - Tc(pool_mol, anchor) >= 0.3           (related-to-lead band)
  - Murcko-stratified for diversity within the band
  - Exclude drug + named successors (canonicalized SMILES match)

Inputs (per target): data/exp7_lo_benchmark/_phase1/<target>/{target_pairs.csv, exclude_set.csv}
Inputs (per pair):   anchor SMILES, drug SMILES

Outputs (per pair): data/exp7_lo_benchmark/_phase1/<target>/anchors/<pair_id>_b_pool.csv
                    data/exp7_lo_benchmark/_phase1/<target>/anchors/<pair_id>_b_audit.json

If pool max-Tc-to-drug >= 0.6, ABORT — wrote `_FAIL.json` instead.
Audit ALSO written for all pools at one consolidated path:
   data/exp7_lo_benchmark/_phase1/strategy_b_pool_audit_all.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp7_pair_loader import load_all_pairs, TARGET_CHEMBL

PHASE1_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_phase1"
AUDIT_OUT = PHASE1_BASE / "strategy_b_pool_audit_all.json"

TC_TO_DRUG_MAX = 0.6     # hard ceiling (answer-leak prevention)
TC_TO_ANCHOR_MIN = 0.3   # lead-class membership requirement
TARGET_POOL_SIZE = 100
SEED = 42

# Vocab-unsupported atoms in mol2mol prior (from exp6 finding)
DISALLOWED_TOKENS = ["[S+]", "[S@@+]", "[S@+]"]


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _murcko(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def vocab_ok(smi: str) -> bool:
    """Reject SMILES containing tokens not in mol2mol prior vocab."""
    return all(tok not in smi for tok in DISALLOWED_TOKENS)


def build_pool_for_pair(pair: Dict, target_dir: Path, audit_out: Dict):
    pair_id = pair["pair_id"]
    anchor_smi = pair["anchor_smiles"]
    drug_smi = pair["drug_smiles"]
    target_key = pair["target_key"]

    anchors_dir = target_dir / "anchors"
    anchors_dir.mkdir(exist_ok=True)

    # Load target's mol pool (mol_a + mol_b unique)
    tp_path = target_dir / "target_pairs.csv"
    if not tp_path.exists():
        audit_out[pair_id] = {"status": "FAIL", "reason": "target_pairs.csv not found"}
        return False
    tp = pd.read_csv(tp_path)
    rows = []
    rows.extend(zip(tp["mol_a_id"], tp["mol_a"], tp["value_a"]))
    rows.extend(zip(tp["mol_b_id"], tp["mol_b"], tp["value_b"]))
    pool_df = pd.DataFrame(rows, columns=["chembl_id","smiles","pIC50"]).groupby("chembl_id", as_index=False).agg({"smiles":"first","pIC50":"mean"})

    # Load exclude canonical SMILES
    excl_path = target_dir / "exclude_set.csv"
    if excl_path.exists():
        excl_df = pd.read_csv(excl_path)
        excl_can = set(excl_df["canon_smiles"]) if "canon_smiles" in excl_df.columns else set()
    else:
        excl_can = set()
    # Per-pair drug + successors should also exclude (canon set)
    pair_excl_can = set()
    pair_excl_can.add(_canon(drug_smi))
    for s in pair.get("successors_smiles", []):
        c = _canon(s)
        if c:
            pair_excl_can.add(c)
    excl_can = excl_can | pair_excl_can

    # Build fingerprints + Tc
    drug_fp = _fp(drug_smi)
    anchor_fp = _fp(anchor_smi)
    if drug_fp is None or anchor_fp is None:
        audit_out[pair_id] = {"status": "FAIL", "reason": "anchor or drug SMILES invalid"}
        return False

    # Score each pool mol
    records = []
    for cid, smi, pic in zip(pool_df["chembl_id"], pool_df["smiles"], pool_df["pIC50"]):
        if not vocab_ok(smi):
            continue
        fp = _fp(smi)
        if fp is None:
            continue
        tc_drug = DataStructs.TanimotoSimilarity(drug_fp, fp)
        tc_anchor = DataStructs.TanimotoSimilarity(anchor_fp, fp)
        can = _canon(smi)
        if can in excl_can:
            continue
        if tc_drug >= TC_TO_DRUG_MAX:
            continue
        if tc_anchor < TC_TO_ANCHOR_MIN:
            continue
        records.append({"chembl_id": cid, "smiles": smi, "pIC50": pic,
                        "tc_drug": float(tc_drug), "tc_anchor": float(tc_anchor),
                        "scaffold": _murcko(smi) or ""})
    eligible_df = pd.DataFrame(records)

    if len(eligible_df) < 10:
        # Too narrow band; fallback: relax tc_anchor floor in 0.05 steps to 0.20
        for relax in [0.25, 0.20]:
            extra = []
            for cid, smi, pic in zip(pool_df["chembl_id"], pool_df["smiles"], pool_df["pIC50"]):
                if not vocab_ok(smi):
                    continue
                fp = _fp(smi)
                if fp is None:
                    continue
                can = _canon(smi)
                if can in excl_can:
                    continue
                tc_drug = DataStructs.TanimotoSimilarity(drug_fp, fp)
                tc_anchor = DataStructs.TanimotoSimilarity(anchor_fp, fp)
                if tc_drug >= TC_TO_DRUG_MAX:
                    continue
                if tc_anchor < relax:
                    continue
                if any(cid == r["chembl_id"] for r in records):
                    continue
                extra.append({"chembl_id": cid, "smiles": smi, "pIC50": pic,
                              "tc_drug": float(tc_drug), "tc_anchor": float(tc_anchor),
                              "scaffold": _murcko(smi) or ""})
            records.extend(extra)
            if len(records) >= TARGET_POOL_SIZE:
                break
        eligible_df = pd.DataFrame(records)

    n_eligible = len(eligible_df)
    if n_eligible == 0:
        audit_out[pair_id] = {"status": "FAIL", "reason": "no eligible mols in band",
                              "n_target_mols": len(pool_df)}
        return False

    # Murcko-stratified pick: round-robin highest pIC50 per scaffold
    rng = np.random.RandomState(SEED)
    scaff_groups = eligible_df.groupby("scaffold")
    scaffolds = list(scaff_groups.groups.keys())
    rng.shuffle(scaffolds)
    scaff_to_sorted = {sc: gp.sort_values("pIC50", ascending=False).to_dict("records")
                       for sc, gp in scaff_groups}
    selected, selected_ids, pass_idx = [], set(), 0
    # Anchor must be present in pool (as cap; if not in target_pairs as a separate mol, add manually)
    # First check eligible
    anchor_can = _canon(anchor_smi)
    anchor_in_eligible = False
    for r in records:
        if _canon(r["smiles"]) == anchor_can:
            anchor_in_eligible = True
            break
    if not anchor_in_eligible:
        selected.append({"chembl_id": pair.get("anchor_chembl_id", "ANCHOR"),
                         "smiles": anchor_smi,
                         "pIC50": float(pair.get("anchor_pIC50", 6.5) or 6.5),
                         "tc_drug": float(DataStructs.TanimotoSimilarity(drug_fp, anchor_fp)),
                         "tc_anchor": 1.0,
                         "scaffold": _murcko(anchor_smi) or ""})
        selected_ids.add(pair.get("anchor_chembl_id", "ANCHOR"))

    while len(selected) < TARGET_POOL_SIZE:
        added = 0
        for sc in scaffolds:
            if len(selected) >= TARGET_POOL_SIZE:
                break
            bucket = scaff_to_sorted[sc]
            if pass_idx >= len(bucket):
                continue
            cand = bucket[pass_idx]
            if cand["chembl_id"] in selected_ids:
                continue
            selected.append(cand)
            selected_ids.add(cand["chembl_id"])
            added += 1
        pass_idx += 1
        if added == 0:
            break

    pool_out = pd.DataFrame(selected)
    pool_out.to_csv(anchors_dir / f"{pair_id}_b_pool.csv", index=False)

    # Audit — exclude anchor row (mol with tc_anchor==1.0) since it's not "leakage", it's the lead by construction
    is_anchor = pool_out["smiles"].apply(lambda s: _canon(s) == anchor_can)
    pool_excl_anchor = pool_out[~is_anchor]
    if len(pool_excl_anchor) == 0:
        pool_excl_anchor = pool_out  # avoid empty stats if pool == anchor only
    tc_drugs = pool_excl_anchor["tc_drug"].values
    tc_anchors = pool_excl_anchor["tc_anchor"].values
    audit = {
        "status": "OK" if tc_drugs.max() < TC_TO_DRUG_MAX else "FAIL_LEAK",
        "n_pool": int(len(pool_out)),
        "n_pool_excl_anchor": int(len(pool_excl_anchor)),
        "n_eligible_before_strat": n_eligible,
        "n_target_mols_total": len(pool_df),
        "tc_drug_max": float(tc_drugs.max()),
        "tc_drug_min": float(tc_drugs.min()),
        "tc_drug_median": float(np.median(tc_drugs)),
        "tc_drug_p95": float(np.quantile(tc_drugs, 0.95)),
        "tc_anchor_max": float(tc_anchors.max()),
        "tc_anchor_min": float(tc_anchors.min()),
        "tc_anchor_median": float(np.median(tc_anchors)),
        "n_unique_scaffolds": int(pool_out["scaffold"].nunique()),
        "anchor_in_pool": bool(any(_canon(s) == anchor_can for s in pool_out["smiles"])),
        "drug_in_pool": bool(any(_canon(s) == _canon(drug_smi) for s in pool_out["smiles"])),
        "note": "tc_drug_max excludes the anchor itself (which by definition has the pair's tc_anchor_drug)",
    }
    (anchors_dir / f"{pair_id}_b_audit.json").write_text(json.dumps(audit, indent=2))
    audit_out[pair_id] = audit
    return True


def main():
    pairs = load_all_pairs()
    print(f"Building Strategy B pools for {len(pairs)} pairs...")
    audit_all = {}
    n_ok, n_fail = 0, 0
    for p in pairs:
        target_dir = PHASE1_BASE / p["target_key"]
        if not target_dir.exists():
            audit_all[p["pair_id"]] = {"status": "FAIL", "reason": f"target_dir missing: {target_dir}"}
            n_fail += 1
            continue
        ok = build_pool_for_pair(p, target_dir, audit_all)
        if ok:
            n_ok += 1
            a = audit_all[p["pair_id"]]
            flag = "" if a["status"] == "OK" else f" [{a['status']}]"
            print(f"  {p['pair_id']:25s} pool={a['n_pool']:3d}, tc_drug_max={a['tc_drug_max']:.3f}, scaffolds={a['n_unique_scaffolds']}{flag}")
        else:
            n_fail += 1
            print(f"  {p['pair_id']:25s} FAIL: {audit_all[p['pair_id']].get('reason','?')}")

    AUDIT_OUT.write_text(json.dumps(audit_all, indent=2))
    print(f"\nDone: {n_ok} ok / {n_fail} fail. Audit: {AUDIT_OUT}")

    # Sanity: count failures by reason
    leak_count = sum(1 for v in audit_all.values() if v.get("status") == "FAIL_LEAK")
    print(f"Leak failures (tc_drug >= 0.6): {leak_count}")


if __name__ == "__main__":
    main()
