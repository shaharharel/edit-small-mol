"""Per-cohort BD-angle eval for the BD-corrected Lingo3DMol cohorts.

Measures the Bürgi-Dunitz angle at atom0 (the chemistry-standard convention
for the electrophilic Cβ carbon) — this is the metric that exposed the
collinear rotation bug fixed in ``experiments/anchor_geometry.py``.

Also reports the legacy angle-at-SG measurement for direct comparison to the
pre-fix evaluation pipeline in ``experiments/eval_lingo3dmol_plans.py``.

Usage::

    python experiments/eval_bd_corrected_cohorts.py \
        --root data/lingo3dmol_BD_corrected \
        --output /tmp/bd_corrected_cohorts_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# ZAP70 Cys346 anchor reference points (matches data/lingo3dmol_anchor_zap70_cys346.json)
SG = np.array([18.888, -3.65, -29.979])
CA = np.array([16.292, -4.283, -28.999])

BD_TARGET_DEG = 107.0
BD_TOL_DEG = 5.0

# Cohorts produced by the BD-fix regeneration.  Pre-fix counterparts are listed
# so the report can show the BD-at-atom0 delta directly.
COHORTS = [
    ("L2_scaffold_C5_BD",   "data/lingo3dmol_BD_corrected/L2_scaffold_C5_BD/samples.sdf",
     "data/lingo3dmol_L2_scaffold_C5_N500/samples.sdf"),
    ("L2_extended_H2_BD",   "data/lingo3dmol_BD_corrected/L2_extended_H2_BD/samples.sdf",
     "data/lingo3dmol_L2_extended_H2_N500/samples.sdf"),
    ("L2_extended_L_BD",    "data/lingo3dmol_BD_corrected/L2_extended_L_BD/samples.sdf",
     "data/lingo3dmol_L2_extended_L_v2/samples_T10.sdf"),
    ("L2_extended_H1_BD",   "data/lingo3dmol_BD_corrected/L2_extended_H1_BD/samples.sdf",
     "data/lingo3dmol_L2_extended_H1/samples_T10.sdf"),
    ("L2_extended_H3_BD",   "data/lingo3dmol_BD_corrected/L2_extended_H3_BD/samples.sdf",
     "data/lingo3dmol_L2_extended_H3/samples_T10.sdf"),
    ("L2_scaffold_C1_BD",   "data/lingo3dmol_BD_corrected/L2_scaffold_C1_BD/samples.sdf",
     "data/lingo3dmol_L2_scaffold_anchor/samples_T10_N500.sdf"),
]


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _angle_at_atom0(conf, sg: np.ndarray) -> float:
    """Angle SG–atom0–atom1 measured AT atom0 (chemistry-standard convention)."""
    a0 = np.array(conf.GetAtomPosition(0))
    a1 = np.array(conf.GetAtomPosition(1))
    v_sg = sg - a0
    v_a1 = a1 - a0
    return _angle_deg(v_sg, v_a1)


def _angle_at_sg(conf, sg: np.ndarray, ca: np.ndarray) -> float:
    """Angle (SG->atom0) ∠ (SG->CA_cys), measured AT SG (legacy eval convention)."""
    a0 = np.array(conf.GetAtomPosition(0))
    v_attack = a0 - sg
    v_back = ca - sg
    return _angle_deg(v_attack, v_back)


def _d_sg(conf, sg: np.ndarray) -> float:
    a0 = np.array(conf.GetAtomPosition(0))
    return float(np.linalg.norm(a0 - sg))


def _largest_fragment(m: Chem.Mol) -> Optional[Chem.Mol]:
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    return max(frags, key=lambda x: x.GetNumHeavyAtoms())


def _murcko_smiles(m: Chem.Mol) -> Optional[str]:
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s and s.GetNumAtoms() > 0 else None
    except Exception:
        return None


# Approximate pocket centre for ZAP70 (used for pocket-clash count). Use
# the SG position as a stand-in for the catalytic centre; "clash" is taken
# as any heavy atom of the ligand within < 2.0 Å of the SG (other than
# atom0, which sits ~1.85 Å by design).
POCKET_CLASH_RADIUS = 2.0


def _pocket_clash_count(m: Chem.Mol, sg: np.ndarray) -> int:
    if m.GetNumConformers() == 0:
        return -1
    conf = m.GetConformer()
    n_clash = 0
    for idx in range(m.GetNumAtoms()):
        if idx == 0:
            # atom0 is the warhead Cβ — its 1.85 Å distance is intentional.
            continue
        p = np.array(conf.GetAtomPosition(idx))
        if np.linalg.norm(p - sg) < POCKET_CLASH_RADIUS:
            n_clash += 1
    return n_clash


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    keys_num = [
        "bd_angle_at_atom0",
        "bd_angle_at_sg",
        "d_SG",
        "MW",
        "QED",
        "pocket_clash_count",
    ]
    out: dict = {"n": len(records)}
    for k in keys_num:
        xs = [r[k] for r in records if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        if not xs:
            out[f"{k}_mean"] = None
            continue
        out[f"{k}_mean"] = float(np.mean(xs))
        out[f"{k}_median"] = float(np.median(xs))
        out[f"{k}_std"] = float(np.std(xs))
        out[f"{k}_min"] = float(np.min(xs))
        out[f"{k}_max"] = float(np.max(xs))
    # BD-fix passes (within ±5° of 107)
    bd_ok = [r for r in records if r.get("bd_angle_at_atom0") is not None
             and abs(r["bd_angle_at_atom0"] - BD_TARGET_DEG) <= BD_TOL_DEG]
    out["bd_at_atom0_pass_pct"] = float(len(bd_ok) / len(records)) if records else 0.0
    # Drug-likeness pass
    qed_ok = [r for r in records if r.get("QED") is not None and r["QED"] >= 0.5]
    out["QED_pass_pct"] = float(len(qed_ok) / len(records)) if records else 0.0
    mw_ok = [r for r in records if r.get("MW") is not None and 300 <= r["MW"] <= 600]
    out["MW_drug_range_pct"] = float(len(mw_ok) / len(records)) if records else 0.0
    # Scaffold diversity
    scaff_set = {r["murcko_smiles"] for r in records if r.get("murcko_smiles")}
    out["scaffold_diversity_pct"] = float(len(scaff_set) / len(records)) if records else 0.0
    # Pocket-clash mean across cohort
    pcs = [r["pocket_clash_count"] for r in records if r.get("pocket_clash_count") is not None and r["pocket_clash_count"] >= 0]
    if pcs:
        out["mean_pocket_clash"] = float(np.mean(pcs))
        out["pct_with_any_pocket_clash"] = float(sum(1 for x in pcs if x > 0) / len(pcs))
    return out


def _eval_sdf(path: str) -> dict:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"path": str(p), "missing_or_empty": True, "summary": {"n": 0}}
    sup = Chem.SDMolSupplier(str(p))
    records: list[dict] = []
    for m in sup:
        if m is None:
            continue
        largest = _largest_fragment(m)
        target = largest if largest is not None else m
        try:
            if target.GetNumConformers() == 0:
                continue
            conf = target.GetConformer(0) if target.GetNumConformers() > 0 else None
            # Use the original mol for atom-0/atom-1 (the warhead-locked atoms);
            # falling back to largest fragment if the original is multi-fragment.
            conf_for_anchor = m.GetConformer(0) if m.GetNumConformers() > 0 else conf
            bd_a0 = _angle_at_atom0(conf_for_anchor, SG)
            bd_sg = _angle_at_sg(conf_for_anchor, SG, CA)
            d_sg = _d_sg(conf_for_anchor, SG)
            mw = float(Descriptors.MolWt(target))
            qed = float(QED.qed(target))
            scaff = _murcko_smiles(target)
            pc = _pocket_clash_count(m, SG)
        except Exception:
            continue
        records.append({
            "smiles": Chem.MolToSmiles(target),
            "bd_angle_at_atom0": bd_a0,
            "bd_angle_at_sg": bd_sg,
            "d_SG": d_sg,
            "MW": mw,
            "QED": qed,
            "murcko_smiles": scaff,
            "pocket_clash_count": pc,
        })
    return {"path": str(p), "missing_or_empty": False,
            "n_records": len(records), "summary": _summarize(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                        help="repo root (paths in COHORTS are repo-relative)")
    parser.add_argument("--output", default="/tmp/bd_corrected_cohorts_eval.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results = []
    for name, post_path, pre_path in COHORTS:
        post = _eval_sdf(str(root / post_path))
        pre = _eval_sdf(str(root / pre_path)) if pre_path else {"missing_or_empty": True}
        results.append({"cohort": name, "post_fix": post, "pre_fix": pre})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")

    # Print compact report
    print()
    print(f"{'cohort':<22} {'n_post':>6} {'n_pre':>6} {'BD@a0 post':>12} {'BD@a0 pre':>12} {'BD@SG post':>12} {'BD@SG pre':>12} {'pass_pct':>9}")
    for r in results:
        post_n = r["post_fix"].get("n_records", 0)
        pre_n = r["pre_fix"].get("n_records", 0)
        post_bd = r["post_fix"].get("summary", {}).get("bd_angle_at_atom0_mean")
        pre_bd = r["pre_fix"].get("summary", {}).get("bd_angle_at_atom0_mean")
        post_sg = r["post_fix"].get("summary", {}).get("bd_angle_at_sg_mean")
        pre_sg = r["pre_fix"].get("summary", {}).get("bd_angle_at_sg_mean")
        pass_pct = r["post_fix"].get("summary", {}).get("bd_at_atom0_pass_pct", 0.0)
        def f(x):
            return f"{x:.1f}" if isinstance(x, (int, float)) and not math.isnan(x) else "-"
        print(f"{r['cohort']:<22} {post_n:>6} {pre_n:>6} {f(post_bd):>12} {f(pre_bd):>12} {f(post_sg):>12} {f(pre_sg):>12} {pass_pct:>9.1%}")


if __name__ == "__main__":
    main()
