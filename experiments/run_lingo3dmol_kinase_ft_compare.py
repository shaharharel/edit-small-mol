"""Strategy 1 QA — Compare Lingo3DMol L1 kinase-FT ckpt vs v1 ckpt on
kinase-pharmacophore score.

Samples N mols from each ckpt with the H2 anchor on the ZAP70 Cys346 pocket,
scores each cohort with the Strategy 2 kinase pharmacophore classifier, and
emits a comparison table.

Usage:
    conda run -n quris python -u experiments/run_lingo3dmol_kinase_ft_compare.py \\
        --old_ckpt data/covlingo_full_v1/ckpt_phase2_dev.pt \\
        --new_ckpt data/lingo3dmol_L1_kinase_FT/ckpt_phase2_dev.pt \\
        --n_samples 30

Note: sampling itself must run via the Lingo3DMol env on a CUDA/MPS box.
This script is the SCORING / COMPARISON half — sampling produces SDFs which
this script reads. If running locally on Mac, you'd already have produced
the two SDFs via the lingo3dmol env.
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")


def morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def smiles_from_sdf(path: Path) -> list[str]:
    if not path.exists():
        return []
    from rdkit.Chem import SDMolSupplier
    out = []
    try:
        supp = SDMolSupplier(str(path), sanitize=True, removeHs=False)
        for m in supp:
            if m is None:
                continue
            try:
                out.append(Chem.MolToSmiles(m))
            except Exception:
                pass
    except Exception as e:
        print(f"  WARN reading {path}: {e}")
    return out


def score_smiles(smiles_list, clf, n_bits=2048):
    feats, kept = [], []
    for s in smiles_list:
        fp = morgan_fp(s, n_bits)
        if fp is not None:
            feats.append(fp)
            kept.append(s)
    if not feats:
        return [], []
    X = np.stack(feats).astype(np.float32)
    probs = clf.predict_proba(X)[:, 1]
    return kept, probs.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old_sdf",
                   default="data/covlingo_v1_H2/samples.sdf",
                   help="Pre-FT cohort SDF.")
    p.add_argument("--new_sdf",
                   default="data/lingo3dmol_L1_kinase_FT/samples_H2.sdf",
                   help="Post-FT cohort SDF.")
    p.add_argument("--clf_path",
                   default="models/kinase_pharmacophore_clf.pkl")
    p.add_argument("--out_json",
                   default="results/paper_evaluation/kinase_ft_compare.json")
    args = p.parse_args()

    print(f"Loading classifier: {args.clf_path}")
    with open(args.clf_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]
    n_bits = bundle.get("n_bits", 2048)
    print(f"  Classifier: ROC-AUC(holdout)={bundle.get('holdout_roc_auc', 'NA'):.4f}, "
          f"ROC-AUC(scaffold)={bundle.get('scaffold_holdout_roc_auc', 'NA')}")

    old_smi = smiles_from_sdf(Path(args.old_sdf))
    new_smi = smiles_from_sdf(Path(args.new_sdf))
    print(f"  old_sdf={args.old_sdf}: {len(old_smi)} mols")
    print(f"  new_sdf={args.new_sdf}: {len(new_smi)} mols")

    old_keep, old_scores = score_smiles(old_smi, clf, n_bits)
    new_keep, new_scores = score_smiles(new_smi, clf, n_bits)

    def summary(scores, label):
        if not scores:
            return {"label": label, "n": 0, "mean": None, "median": None,
                    "p25": None, "p75": None, "frac_gt_05": None,
                    "frac_gt_07": None}
        a = np.array(scores)
        return {
            "label": label,
            "n": int(len(a)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "frac_gt_05": float((a > 0.5).mean()),
            "frac_gt_07": float((a > 0.7).mean()),
        }

    old_s = summary(old_scores, "L1_FT_v1 (pre-kinase)")
    new_s = summary(new_scores, "L1_kinase_FT (post)")

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for s in [old_s, new_s]:
        print(f"  {s['label']:35s}  n={s['n']:3d}  mean={s['mean']}  "
              f"frac>0.5={s['frac_gt_05']}  frac>0.7={s['frac_gt_07']}")

    # Mann-Whitney test
    from scipy.stats import mannwhitneyu
    if old_scores and new_scores:
        mw = mannwhitneyu(new_scores, old_scores, alternative="greater")
        print(f"  Mann-Whitney U test (new > old, one-sided): p={mw.pvalue:.4g}")
        mw_p = float(mw.pvalue)
    else:
        mw_p = None

    out = {
        "old_summary": old_s,
        "new_summary": new_s,
        "mw_p_new_gt_old": mw_p,
        "old_scores": old_scores,
        "new_scores": new_scores,
        "old_smiles": old_keep,
        "new_smiles": new_keep,
        "clf_holdout_roc_auc": float(bundle.get("holdout_roc_auc", float("nan"))),
        "clf_scaffold_holdout_roc_auc": float(bundle.get("scaffold_holdout_roc_auc", float("nan"))),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Wrote {args.out_json}")

    if new_s["mean"] is not None and old_s["mean"] is not None:
        delta = new_s["mean"] - old_s["mean"]
        improved = "IMPROVED" if delta > 0 else "REGRESSED"
        print(f"\n  Kinase pharmacophore score delta: {delta:+.4f}  ({improved})")
        if mw_p is not None and mw_p < 0.05:
            print(f"  Significant at p<0.05 (Mann-Whitney one-sided)")
    return out


if __name__ == "__main__":
    main()
