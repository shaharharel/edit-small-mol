"""QA #5 (2026-06-24): For each during-RL-stream cohort, report metrics from the
LAST 5 STEPS only — gives a cleaner late-policy snapshot for comparing reward
configurations side-by-side.

Inputs (all have a `step` column):
  - data/exp_reward_ablation/drop_film/thiq_rl_ablation_drop_film_1.csv (steps 1-50)
  - data/exp_reward_ablation/drop_smarts/thiq_rl_ablation_drop_smarts_1.csv (steps 1-5)
  - data/exp_reward_ablation/drop_qed/thiq_rl_ablation_drop_qed_1.csv (steps 1-5)
  - data/exp_distilled_iptm/cohort.csv (steps 1-50)

For each: compute LAST 5 STEPS subset (or all if <5).
  - n_samples
  - mean validity, mean THIQ retention, mean FiLMDelta pIC50, mean QED
  - mean iptm if present
  - mean Tc to Mol1

Also writes a single TSV/MD table.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED, Descriptors

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

MOL1 = Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1")
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(MOL1, 2, nBits=2048)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tanimoto(fp1, fp2):
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


COHORTS = [
    ("drop_film_50step_last5", PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_film"
     / "thiq_rl_ablation_drop_film_1.csv"),
    ("drop_smarts_5step_last5", PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_smarts"
     / "thiq_rl_ablation_drop_smarts_1.csv"),
    ("drop_qed_5step_last5", PROJECT_ROOT / "data" / "exp_reward_ablation" / "drop_qed"
     / "thiq_rl_ablation_drop_qed_1.csv"),
    ("distilled_iptm_50step_last5", PROJECT_ROOT / "data" / "exp_distilled_iptm" / "cohort.csv"),
]


def cohort_late_metrics(label, path):
    df = pd.read_csv(path)
    if "step" not in df.columns:
        return {"label": label, "error": "no step column"}
    max_step = int(df["step"].max())
    cutoff = max_step - 4  # last 5
    late = df[df["step"] >= cutoff].copy()
    n = len(late)
    # Validity from SMILES_state if present
    n_valid = int((late["SMILES_state"] == 1).sum()) if "SMILES_state" in late.columns else n
    res = {
        "label": label,
        "path": str(path),
        "n_steps_total": max_step,
        "last_step_window": [int(cutoff), int(max_step)],
        "n_samples": int(n),
        "validity_pct": 100 * n_valid / n if n else 0.0,
    }
    # Computed columns from REINVENT 'raw' suffix
    for c, key in [
        ("FiLMDelta pIC50 (raw)", "film_pic50"),
        ("THIQ-acrylamide core retained (raw)", "thiq_retain"),
        ("QED (raw)", "qed"),
        ("Distilled Boltz iptm (raw)", "iptm"),
        ("Score", "score"),
    ]:
        if c in late.columns:
            v = late[c].astype(float).dropna()
            res[f"{key}_mean"] = float(v.mean()) if len(v) else None
            res[f"{key}_median"] = float(v.median()) if len(v) else None
            res[f"{key}_p90"] = float(v.quantile(0.9)) if len(v) else None
    # Direct compute: acrylamide presence (SMARTS) for those mols
    n_acryl = 0
    tc_vals = []
    acryl = Chem.MolFromSmarts("C=CC(=O)N")
    smiles_col = "SMILES" if "SMILES" in late.columns else late.columns[0]
    smis = late[smiles_col].astype(str).tolist()
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if m.HasSubstructMatch(acryl):
            n_acryl += 1
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        tc_vals.append(tanimoto(fp, MOL1_FP))
    res["acrylamide_pct_late"] = 100 * n_acryl / max(1, len(smis))
    res["tc_mol1_mean_late"] = float(np.mean(tc_vals)) if tc_vals else None
    res["tc_mol1_p90_late"] = float(np.percentile(tc_vals, 90)) if tc_vals else None
    return res


def main():
    log("=== QA #5: late-step (last 5) RL stream metrics ===")
    rows = []
    for label, path in COHORTS:
        if not path.exists():
            log(f"MISSING: {label} -> {path}")
            rows.append({"label": label, "error": "file missing"})
            continue
        log(f"Processing {label}")
        r = cohort_late_metrics(label, path)
        rows.append(r)
        log(f"  {label}: steps {r.get('last_step_window')}, n={r.get('n_samples')}, "
            f"film={r.get('film_pic50_mean')}, thiq={r.get('thiq_retain_mean')}, "
            f"acryl_late={r.get('acrylamide_pct_late'):.1f}%")

    out_path = OUT_DIR / "exp3_rl_late_step.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    log(f"Wrote {out_path}")

    # MD
    headers = ["label", "last_step_window", "n_samples",
               "film_pic50_mean", "film_pic50_p90",
               "thiq_retain_mean", "qed_mean", "iptm_mean",
               "score_mean", "acrylamide_pct_late", "tc_mol1_mean_late"]
    lines = ["# Exp 3 + Exp B — LATE-STEP (last 5 steps) reward-ablation snapshot\n",
             "Compares the per-reward configuration's BEHAVIOR AT THE LATE POLICY only,\n"
             "removing the early-policy noise that dominated the cohort-level QA2 numbers.\n",
             ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['label']} | ERROR: {r['error']} |" + " - |" * (len(headers) - 1))
            continue
        row = []
        for h in headers:
            v = r.get(h)
            if v is None:
                row.append("—")
            elif isinstance(v, float):
                if h.endswith("_pct_late"):
                    row.append(f"{v:.1f}%")
                else:
                    row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    md = OUT_DIR / "exp3_rl_late_step.md"
    md.write_text("\n".join(lines))
    log(f"Wrote {md}")
    log("DONE")


if __name__ == "__main__":
    main()
