"""QA sweep #2 (2026-06-24): Exp 3 reward ablation + Exp B distilled-iptm
cohort analyses, side-by-side on uniform metrics.

For each cohort (per row), compute:
  - validity, uniqueness, scaffold uniqueness
  - QED, MW, logP, SA, TPSA, Fsp3 (descriptors)
  - Tc to Mol1 (anchor)
  - warhead presence (acrylamide SMARTS C=CC(=O)N)
  - THIQ-acrylamide core retention (SMARTS C=CC(=O)N1Cc2ccccc2C1)
  - FiLMDelta pIC50 distribution (use the 'FiLMDelta pIC50 (raw)' column if present;
    otherwise re-score with the FiLM REST server if reachable, else NA)
  - Distilled Boltz iptm (raw) if present

Cohorts compared:
  - prior_covft_baseline        : data/exp_rl_value/baseline_covft_samples.csv
  - prior_warhead_tokens        : data/exp_rl_value/baseline_warhead_tokens_samples.csv
  - dap_full_3comp              : data/_backups/.../thiq_rl_zap70_scored.csv
                                  (FiLM + SMARTS + QED; tier1 scored cohort)
  - distilled_iptm_4comp        : data/exp_distilled_iptm/cohort.csv
                                  (FiLM + iptm + SMARTS + QED)
  - drop_film (no FiLM)         : data/exp_reward_ablation/drop_film/thiq_rl_ablation_drop_film_1.csv
  - drop_smarts (no SMARTS)     : data/exp_reward_ablation/drop_smarts/thiq_rl_ablation_drop_smarts_1.csv
  - drop_qed   (no QED)         : data/exp_reward_ablation/drop_qed/thiq_rl_ablation_drop_qed_1.csv
  - ppo_v1 (collapsed PPO)      : data/exp_ppo/ppo_zap70_samples.csv

For fair comparison: subsample drop_film to its first 5 steps' worth of rows
(matching the natural early termination of drop_smarts/drop_qed at step 5).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys
try:
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import RDConfig
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer
    def sa_score(m):
        return sascorer.calculateScore(m)
except Exception:
    def sa_score(m):
        return float("nan")

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYLAMIDE_SMARTS = "C=CC(=O)N"
THIQ_CORE_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"

COHORTS = [
    # (label, path, smiles_col, optional_subsample_step_max)
    ("prior_covft_baseline",
     PROJECT_ROOT / "data" / "exp_rl_value" / "baseline_covft_samples.csv", "SMILES", None),
    ("prior_warhead_tokens",
     PROJECT_ROOT / "data" / "exp_rl_value" / "baseline_warhead_tokens_samples.csv", "SMILES", None),
    ("dap_full_3comp",
     PROJECT_ROOT / "data" / "_backups" / "20260623_025449" / "tier1_scored_cohorts"
     / "thiq_rl_zap70_scored.csv", "smiles", None),
    ("distilled_iptm_4comp",
     PROJECT_ROOT / "data" / "exp_distilled_iptm" / "cohort.csv", "SMILES", None),
    ("drop_film_full",
     PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_film"
     / "thiq_rl_ablation_drop_film_1.csv", "SMILES", None),
    ("drop_film_5step",  # fair subsample
     PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_film"
     / "thiq_rl_ablation_drop_film_1.csv", "SMILES", 5),
    ("drop_smarts",
     PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_smarts"
     / "thiq_rl_ablation_drop_smarts_1.csv", "SMILES", None),
    ("drop_qed",
     PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_qed"
     / "thiq_rl_ablation_drop_qed_1.csv", "SMILES", None),
    ("ppo_v1_collapsed",
     PROJECT_ROOT / "data" / "exp_ppo" / "ppo_zap70_samples.csv", "SMILES", None),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def morgan_fp(mol, n_bits=2048, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2):
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_cohort(label, path, smiles_col, max_step):
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        log(f"  WARN: '{smiles_col}' not in {label}; using first col")
        smiles_col = df.columns[0]
    if max_step is not None and "step" in df.columns:
        df = df[df["step"] <= max_step].copy()
    smis = df[smiles_col].astype(str).tolist()
    n_total = len(smis)

    # parse
    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    mol1_fp = morgan_fp(mol1)
    acryl = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    thiq = Chem.MolFromSmarts(THIQ_CORE_SMARTS)

    valid_mols = []
    canonical = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        valid_mols.append(m)
        canonical.append(Chem.MolToSmiles(m))
    n_valid = len(valid_mols)

    # uniqueness
    canonical_set = set(canonical)
    n_unique = len(canonical_set)

    # scaffolds
    scaffolds = set()
    for m in valid_mols:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.add(Chem.MolToSmiles(scaf))
        except Exception:
            pass
    n_scaf = len(scaffolds)

    # descriptors / warhead / Tc to Mol1
    mw, logp, qed_v, tpsa, fsp3 = [], [], [], [], []
    sa_l = []
    has_acryl = 0
    has_thiq = 0
    tc_mol1 = []
    for m in valid_mols:
        try:
            mw.append(Descriptors.MolWt(m))
            logp.append(Crippen.MolLogP(m))
            qed_v.append(QED.qed(m))
            tpsa.append(Descriptors.TPSA(m))
            fsp3.append(Lipinski.FractionCSP3(m))
            sa_l.append(sa_score(m))
            if m.HasSubstructMatch(acryl):
                has_acryl += 1
            if m.HasSubstructMatch(thiq):
                has_thiq += 1
            fp = morgan_fp(m)
            tc_mol1.append(tanimoto(fp, mol1_fp))
        except Exception:
            pass

    res = {
        "label": label,
        "path": str(path),
        "max_step_filter": max_step,
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_pct": 100 * n_valid / n_total if n_total else 0.0,
        "n_unique": n_unique,
        "uniqueness_pct_among_valid": 100 * n_unique / n_valid if n_valid else 0.0,
        "n_scaffolds": n_scaf,
        "scaffold_uniqueness_pct": 100 * n_scaf / n_valid if n_valid else 0.0,
        "acrylamide_pct": 100 * has_acryl / n_valid if n_valid else 0.0,
        "thiq_core_pct": 100 * has_thiq / n_valid if n_valid else 0.0,
        "mw_mean": float(np.mean(mw)) if mw else None,
        "logp_mean": float(np.mean(logp)) if logp else None,
        "qed_mean": float(np.mean(qed_v)) if qed_v else None,
        "tpsa_mean": float(np.mean(tpsa)) if tpsa else None,
        "fsp3_mean": float(np.mean(fsp3)) if fsp3 else None,
        "sa_mean": float(np.mean(sa_l)) if sa_l else None,
        "tc_mol1_mean": float(np.mean(tc_mol1)) if tc_mol1 else None,
        "tc_mol1_median": float(np.median(tc_mol1)) if tc_mol1 else None,
        "tc_mol1_p90": float(np.percentile(tc_mol1, 90)) if tc_mol1 else None,
    }

    # FiLMDelta pIC50 if present
    pic50_col = None
    for c in ["FiLMDelta pIC50 (raw)", "pIC50_film", "pIC50_mean"]:
        if c in df.columns:
            pic50_col = c
            break
    if pic50_col:
        v = df[pic50_col].astype(float).dropna()
        res["filmdelta_pic50_mean"] = float(v.mean()) if len(v) else None
        res["filmdelta_pic50_median"] = float(v.median()) if len(v) else None
        res["filmdelta_pic50_p90"] = float(v.quantile(0.9)) if len(v) else None
        res["filmdelta_pic50_n"] = int(len(v))
        res["filmdelta_pic50_col"] = pic50_col
    else:
        res["filmdelta_pic50_mean"] = None
        res["filmdelta_pic50_col"] = None

    # Distilled iptm if present
    if "Distilled Boltz iptm (raw)" in df.columns:
        v = df["Distilled Boltz iptm (raw)"].astype(float).dropna()
        res["distilled_iptm_mean"] = float(v.mean()) if len(v) else None
        res["distilled_iptm_median"] = float(v.median()) if len(v) else None
        res["distilled_iptm_p90"] = float(v.quantile(0.9)) if len(v) else None
    else:
        res["distilled_iptm_mean"] = None

    return res


def main():
    log("=== QA #2: ablation + distilled iptm cohort comparison ===")
    out = []
    for label, path, scol, ms in COHORTS:
        if not Path(path).exists():
            log(f"MISSING: {label} -> {path}")
            out.append({"label": label, "path": str(path), "error": "file missing"})
            continue
        log(f"Processing {label} from {path} (max_step={ms})")
        try:
            r = compute_cohort(label, path, scol, ms)
            out.append(r)
            log(f"  done: n_valid={r['n_valid']}, qed={r['qed_mean']:.3f}, "
                f"acryl={r['acrylamide_pct']:.1f}%, thiq={r['thiq_core_pct']:.1f}%, "
                f"tc_mol1={r['tc_mol1_mean']:.3f}")
        except Exception as e:
            log(f"  ERROR processing {label}: {e}")
            out.append({"label": label, "path": str(path), "error": str(e)})

    out_path = OUT_DIR / "exp3_ablation_and_distilled_iptm.json"
    out_path.write_text(json.dumps({"rows": out}, indent=2))
    log(f"Wrote {out_path}")

    # Markdown table
    headers = [
        "label", "n_total", "n_valid", "validity_pct", "uniqueness_pct_among_valid",
        "scaffold_uniqueness_pct",
        "acrylamide_pct", "thiq_core_pct",
        "qed_mean", "mw_mean", "logp_mean", "sa_mean",
        "tc_mol1_mean", "tc_mol1_p90",
        "filmdelta_pic50_mean", "filmdelta_pic50_p90",
        "distilled_iptm_mean",
    ]
    lines = ["# Exp 3 ablation + Exp B distilled-iptm — uniform metrics\n",
             "## Cohort table (Mol1 ZAP70 RL family)\n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for r in out:
        row = []
        for h in headers:
            v = r.get(h)
            if v is None:
                row.append("—")
            elif isinstance(v, float):
                if h.endswith("_pct"):
                    row.append(f"{v:.1f}%")
                else:
                    row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    md = OUT_DIR / "exp3_ablation_and_distilled_iptm.md"
    md.write_text("\n".join(lines))
    log(f"Wrote {md}")
    log("DONE")


if __name__ == "__main__":
    main()
