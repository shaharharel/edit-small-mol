#!/usr/bin/env python3
"""Build the ZAP70 RLHF preference-collection pair dataset.

Reads the ZAP70 (CHEMBL2803) slice of molecule_pIC50_minimal.csv, keeps only
molecules that already have a Boltz cofold pose on disk, and forms SAME-LAB
similar / one-higher-one-lower-potency pairs.

A "same lab" cohort = molecules sharing an assay_chembl_id (which for ZAP70 maps
1:1 to a publication / doc_id). Within a cohort we keep a pair when:

  * Tanimoto(Morgan r2, 2048-bit) >= TANIMOTO_MIN   (similar)
  * |delta pIC50| >= GAP_MIN                         (meaningful potency gap)
  * BOTH molecules have a Boltz pose                 (3D viewable)

Outputs (NO existing data is moved or deleted; everything is written fresh under
data/rlhf_demo/):
  * data/rlhf_demo/molecules.json  -- id -> smiles, pIC50, assay, doc, pose, conf
  * data/rlhf_demo/pairs.json      -- pair_id, mol ids, tanimoto, delta, cohort

Run:  conda run -n quris python experiments/rlhf_server/build_pairs.py
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

_ALERT_CATALOG = None


def _alert_catalog():
    """PAINS + Brenk structural-alert catalog (built once)."""
    global _ALERT_CATALOG
    if _ALERT_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _ALERT_CATALOG = FilterCatalog(params)
    return _ALERT_CATALOG


def compute_props(smiles: str) -> dict:
    """Physchem descriptors shown to chemists. NO potency / LE (those leak the
    hidden activity signal). Deltas are computed client-side per pair."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    try:
        n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol) + \
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    except Exception:
        n_stereo = 0
    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "ArRings": Lipinski.NumAromaticRings(mol),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
        "stereo": int(n_stereo),
        "alerts": len(_alert_catalog().GetMatches(mol)),
    }

PROJECT = Path(__file__).resolve().parents[2]
SRC_CSV = PROJECT / "data/overlapping_assays/molecule_pIC50_minimal.csv"
POSE_DIR = PROJECT / "data/rlhf_demo/boltz_poses"
OUT_DIR = PROJECT / "data/rlhf_demo"
ZAP70 = "CHEMBL2803"

TANIMOTO_MIN = 0.6
TANIMOTO_MAX = 0.999   # drop stereo/regio-only "identical-FP" pairs (poor visual items)
GAP_MIN = 0.5
# Pose-confidence floor (per scientist review): Boltz confidence, not correctness,
# but cleanly separates trustworthy kinase poses from low-confidence phosphonate
# peptidomimetics. A pair survives only if BOTH molecules clear the floor.
LIGAND_IPTM_MIN = 0.80
PLDDT_MIN = 0.78
COHORT_CAP = 40        # avoid one publication series dominating the session


def index_poses() -> dict[str, dict]:
    """Map CHEMBL id -> {cif: path, conf: {...}} by scanning pose dirs NN_CHEMBLID."""
    poses: dict[str, dict] = {}
    for d in sorted(POSE_DIR.glob("*_CHEMBL*")):
        if not d.is_dir():
            continue
        m = re.match(r"\d+_(CHEMBL\d+)$", d.name)
        if not m:
            continue
        chembl = m.group(1)
        cif = d / f"{d.name}_model_0.cif"
        if not cif.exists():
            continue
        conf_path = d / f"confidence_{d.name}_model_0.json"
        conf = {}
        if conf_path.exists():
            try:
                c = json.loads(conf_path.read_text())
                conf = {
                    "confidence_score": round(float(c.get("confidence_score", 0.0)), 3),
                    "iptm": round(float(c.get("iptm", 0.0)), 3),
                    "ligand_iptm": round(float(c.get("ligand_iptm", 0.0)), 3),
                    "complex_plddt": round(float(c.get("complex_plddt", 0.0)), 3),
                }
            except Exception:
                pass
        poses[chembl] = {"cif": str(cif.relative_to(PROJECT)), "conf": conf}
    return poses


def main() -> None:
    poses = index_poses()
    print(f"Boltz poses on disk: {len(poses)} molecules")

    df = pd.read_csv(SRC_CSV)
    z = df[df.target_chembl_id == ZAP70].copy()
    # one pIC50 per (molecule, assay) = a single same-lab measurement
    g = (
        z.groupby(["molecule_chembl_id", "assay_chembl_id", "doc_id"])
        .agg(smiles=("smiles", "first"), pIC50=("pIC50", "median"))
        .reset_index()
    )

    # canonical smiles + fingerprints, only for molecules that have a pose
    fps: dict[str, object] = {}
    canon: dict[str, str] = {}
    for chembl, smi in g[["molecule_chembl_id", "smiles"]].drop_duplicates().values:
        if chembl not in poses or chembl in fps:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps[chembl] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        canon[chembl] = Chem.MolToSmiles(mol)
    print(f"Pose-backed molecules with valid fingerprints: {len(fps)}")

    def pose_ok(chembl: str) -> bool:
        c = poses.get(chembl, {}).get("conf", {})
        return (
            c.get("ligand_iptm", 0.0) >= LIGAND_IPTM_MIN
            and c.get("complex_plddt", 0.0) >= PLDDT_MIN
        )

    n_conf_ok = sum(pose_ok(m) for m in fps)
    print(f"Molecules passing pose-confidence floor: {n_conf_ok}/{len(fps)}")

    # build same-lab pairs
    seen: set[frozenset] = set()
    pairs = []
    used_mols: set[str] = set()
    for assay, sub in g.groupby("assay_chembl_id"):
        sub = sub.drop_duplicates("molecule_chembl_id")
        rows = sub[["molecule_chembl_id", "pIC50", "doc_id"]].values
        for (a, pa, doc), (b, pb, _) in itertools.combinations(rows, 2):
            if a not in fps or b not in fps:
                continue
            if not (pose_ok(a) and pose_ok(b)):
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            if abs(pa - pb) < GAP_MIN:
                continue
            tan = DataStructs.TanimotoSimilarity(fps[a], fps[b])
            if tan < TANIMOTO_MIN or tan >= TANIMOTO_MAX:
                continue
            seen.add(key)
            # order so mol_high is the more potent one (higher pIC50)
            (mh, ph), (ml, pl) = ((a, pa), (b, pb)) if pa >= pb else ((b, pb), (a, pa))
            pairs.append(
                {
                    "pair_id": f"P{len(pairs):04d}",
                    "mol_high_id": mh,
                    "mol_low_id": ml,
                    "pIC50_high": round(float(ph), 2),
                    "pIC50_low": round(float(pl), 2),
                    "delta_pIC50": round(float(ph - pl), 2),
                    "tanimoto": round(float(tan), 3),
                    "assay_chembl_id": str(assay),
                    "doc_id": int(doc),
                }
            )
            used_mols.update((mh, ml))

    # quality/diversity ordering: present higher-quality (clear gap, good pose,
    # high similarity) pairs first, and spread across cohorts so a session is varied.
    def quality(p):
        cmin = min(
            poses[p["mol_high_id"]]["conf"].get("ligand_iptm", 0.0),
            poses[p["mol_low_id"]]["conf"].get("ligand_iptm", 0.0),
        )
        return p["delta_pIC50"] * 0.5 + p["tanimoto"] + cmin

    pairs.sort(key=quality, reverse=True)
    # cap each cohort (keep its highest-quality pairs) so no single publication
    # series dominates the labeling session
    capped: list[dict] = []
    per_cohort_count: dict[str, int] = {}
    for p in pairs:
        c = p["assay_chembl_id"]
        if per_cohort_count.get(c, 0) >= COHORT_CAP:
            continue
        per_cohort_count[c] = per_cohort_count.get(c, 0) + 1
        capped.append(p)
    pairs = capped
    # round-robin across cohorts for variety in the first pairs shown
    by_cohort: dict[str, list] = {}
    for p in pairs:
        by_cohort.setdefault(p["assay_chembl_id"], []).append(p)
    ordered, idx = [], 0
    while any(by_cohort.values()):
        for c in list(by_cohort):
            if by_cohort[c]:
                p = by_cohort[c].pop(0)
                p["order"] = idx
                ordered.append(p)
                idx += 1

    # only molecules that appear in a SURVIVING pair (after cohort cap)
    final_mols = {x for p in ordered for x in (p["mol_high_id"], p["mol_low_id"])}
    molecules = {
        m: {
            "chembl_id": m,
            "smiles": canon[m],
            "pose": poses[m]["cif"],
            "conf": poses[m]["conf"],
            "props": compute_props(canon[m]),
        }
        for m in sorted(final_mols)
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "molecules.json").write_text(json.dumps(molecules, indent=2))
    (OUT_DIR / "pairs.json").write_text(json.dumps(ordered, indent=2))

    n_cohorts = len({p["assay_chembl_id"] for p in ordered})
    print(f"\nWrote {len(ordered)} pairs across {n_cohorts} same-lab cohorts")
    print(f"      {len(molecules)} unique molecules (all pose-backed)")
    print(f"  ->  {OUT_DIR/'pairs.json'}")
    print(f"  ->  {OUT_DIR/'molecules.json'}")
    if ordered:
        ex = ordered[0]
        print(
            f"\nExample pair {ex['pair_id']}: {ex['mol_high_id']} (pIC50 {ex['pIC50_high']}) "
            f"vs {ex['mol_low_id']} (pIC50 {ex['pIC50_low']}), "
            f"Δ={ex['delta_pIC50']}, Tc={ex['tanimoto']}, assay {ex['assay_chembl_id']}"
        )


if __name__ == "__main__":
    main()
