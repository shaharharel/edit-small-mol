"""Phase D — Merge multi-chassis Lingo3DMol cohorts into a single deduped
ensemble SDF, eval each cohort, compute ensemble statistics, and rank
top-N mols by composite score.

Usage:
    python experiments/merge_zap70_chassis_ensemble.py \
        --base_dir data/lingo3dmol_zap70_chassis \
        --chassis_ids ZAP_C1 ZAP_C2 ZAP_C3 ZAP_C4 ZAP_C5 ZAP_C6 \
        --output_dir data/lingo3dmol_zap70_chassis/ENSEMBLE
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def _load_sdf(path: Path):
    if not path.exists():
        return []
    if path.stat().st_size < 10:
        return []
    try:
        sup = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        return [m for m in sup if m is not None]
    except Exception as e:
        print(f"[merge] failed to load {path}: {e}")
        return []


def _canon(m):
    try:
        s = Chem.MolToSmiles(m)
        return s
    except Exception:
        return None


def _murcko(m):
    try:
        sca = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sca)
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/lingo3dmol_zap70_chassis")
    ap.add_argument("--chassis_ids", nargs="+",
                    default=["ZAP_C1", "ZAP_C2", "ZAP_C3", "ZAP_C4", "ZAP_C5", "ZAP_C6"])
    ap.add_argument("--output_dir", default="data/lingo3dmol_zap70_chassis/ENSEMBLE")
    ap.add_argument("--sdf_basename", default="samples.sdf",
                    help="SDF filename inside each chassis directory.")
    ap.add_argument("--top_n_per_chassis_for_score", type=int, default=50)
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_chassis_stats = {}
    seen_canon = set()
    merged_mols = []
    merged_chassis_labels = []

    for cid in args.chassis_ids:
        sdf_path = base / cid / args.sdf_basename
        mols = _load_sdf(sdf_path)
        # also include any "initial_partial" runs (from prior killed runs)
        for ip_dir in sorted((base / cid).glob("initial_partial*")):
            ip_path = ip_dir / "samples_initial.sdf"
            if ip_path.exists():
                mols += _load_sdf(ip_path)
        # also include the smoke mols (these are real samples too)
        smoke_path = base / f"{cid}_smoke" / "samples.sdf"
        if smoke_path.exists():
            mols += _load_sdf(smoke_path)
        if not mols:
            per_chassis_stats[cid] = {
                "n_mols_raw": 0,
                "sdf_path": str(sdf_path),
                "status": "no_sdf_or_empty",
            }
            print(f"[merge] {cid}: SDF empty or missing at {sdf_path}")
            continue
        # canonicalize, dedup within cohort
        within_seen = set()
        within_mols = []
        for m in mols:
            c = _canon(m)
            if c is None or c in within_seen:
                continue
            within_seen.add(c)
            within_mols.append((c, m))
        # compute per-chassis metrics
        mws = [Descriptors.MolWt(m) for _, m in within_mols]
        qeds = [QED.qed(m) for _, m in within_mols]
        scas = Counter(_murcko(m) for _, m in within_mols)
        mw_in_drug = sum(1 for w in mws if 320 <= w <= 480) / max(1, len(mws))
        qed_pass = sum(1 for q in qeds if q >= 0.4) / max(1, len(qeds))
        scaffold_div = len(scas) / max(1, len(within_mols))
        per_chassis_stats[cid] = {
            "n_mols_raw": len(mols),
            "n_mols_unique_in_cohort": len(within_mols),
            "MW_mean": float(sum(mws) / max(1, len(mws))),
            "MW_in_drug_range_pct": float(mw_in_drug * 100),
            "QED_mean": float(sum(qeds) / max(1, len(qeds))),
            "QED_passing_pct": float(qed_pass * 100),
            "n_unique_murcko": len(scas),
            "scaffold_diversity_pct": float(scaffold_div * 100),
            "sdf_path": str(sdf_path),
            "status": "ok",
        }
        # dedup into global merged set
        for canon_smi, m in within_mols:
            if canon_smi in seen_canon:
                continue
            seen_canon.add(canon_smi)
            merged_mols.append(m)
            merged_chassis_labels.append(cid)

    # write ensemble SDF
    ens_sdf = out_dir / "samples.sdf"
    with Chem.SDWriter(str(ens_sdf)) as w:
        for m, label in zip(merged_mols, merged_chassis_labels):
            m.SetProp("source_chassis", label)
            w.write(m)
    print(f"[merge] ensemble: {len(merged_mols)} unique mols -> {ens_sdf}")

    # ensemble statistics
    ens_scas = set()
    ens_mw = []
    ens_qed = []
    ens_composite = []
    for m in merged_mols:
        mw = Descriptors.MolWt(m)
        q = QED.qed(m)
        sca = _murcko(m)
        ens_scas.add(sca)
        ens_mw.append(mw)
        ens_qed.append(q)
        # composite score = QED * (MW_in_drug ? 1 : 0.5) * (1 if has heteroaryl arm else 0.7)
        mw_factor = 1.0 if 320 <= mw <= 480 else 0.5
        # heteroaryl arm = at least 2 aromatic rings with N
        het_arom = 0
        ri = m.GetRingInfo()
        for r in ri.AtomRings():
            atoms = [m.GetAtomWithIdx(i) for i in r]
            if all(a.GetIsAromatic() for a in atoms):
                if any(a.GetAtomicNum() == 7 for a in atoms):
                    het_arom += 1
        het_factor = 1.0 if het_arom >= 1 else 0.7
        comp = q * mw_factor * het_factor
        ens_composite.append(comp)

    # top-50 by composite
    idx_sorted = sorted(range(len(merged_mols)), key=lambda i: -ens_composite[i])[:50]
    top50 = []
    for rank, i in enumerate(idx_sorted, 1):
        m = merged_mols[i]
        top50.append({
            "rank": rank,
            "smiles": Chem.MolToSmiles(m),
            "source_chassis": merged_chassis_labels[i],
            "MW": float(Descriptors.MolWt(m)),
            "QED": float(QED.qed(m)),
            "composite_score": float(ens_composite[i]),
        })

    # CHR per chassis is computed by external eval script; just summarize
    summary = {
        "n_chassis_attempted": len(args.chassis_ids),
        "chassis_ids": args.chassis_ids,
        "n_unique_mols_ensemble": len(merged_mols),
        "n_unique_murcko_ensemble": len(ens_scas),
        "MW_mean": float(sum(ens_mw) / max(1, len(ens_mw))),
        "MW_in_drug_range_pct": float(
            sum(1 for w in ens_mw if 320 <= w <= 480) / max(1, len(ens_mw)) * 100
        ),
        "QED_mean": float(sum(ens_qed) / max(1, len(ens_qed))),
        "QED_passing_pct": float(
            sum(1 for q in ens_qed if q >= 0.4) / max(1, len(ens_qed)) * 100
        ),
        "per_chassis": per_chassis_stats,
        "top_50_by_composite": top50,
    }
    (out_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"[merge] wrote summary -> {out_dir / 'ensemble_summary.json'}")
    print(f"[merge] unique murcko scaffolds: {len(ens_scas)}")
    print(f"[merge] MW_in_drug_range: {summary['MW_in_drug_range_pct']:.1f}%")
    print(f"[merge] QED_passing: {summary['QED_passing_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
