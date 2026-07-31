"""Compare RL-tuned cohort vs pre-RL baseline cohort for the distilled-validator experiment.

WHAT THIS TESTS — be honest about circularity:

  REWARD COMPONENTS (expected to go UP on RL cohort by construction; this only
  validates that the RL actually optimized the chosen objective — not that the
  optimized objective corresponds to better geometry in reality):
    - distilled iptm prediction  (the reward we added)
    - FiLM pIC50  (existing reward)
    - THIQ-SMARTS warhead retention  (existing reward)
    - QED  (existing reward)

  INDEPENDENT QUESTIONS (these are the actual experiment):
    - Mode-collapse: did RL collapse to one scaffold? (Bemis-Murcko unique count,
      intra-cohort Tanimoto distribution, scaffolds-per-mol ratio)
    - Drug-likeness drift: did MW/LogP/TPSA blow up to chase reward?
    - True geometry shift (gold standard): requires actual Boltz-2 cofold on
      samples from BOTH arms. NOT done here — deferred to a follow-up step.

Usage:
    python distilled_rl_compare.py \\
        --rl_smi  data/distilled_rl_v1/rl_cohort.smi \\
        --base_smi data/distilled_rl_v1/baseline_cohort.smi \\
        --bundle  models/boltz_distilled/boltz_distilled_v1.pkl \\
        --out     data/distilled_rl_v1/comparison.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def fp_of(s, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def score_cohort(smiles, m_iptm, m_mpae, label):
    rows = []
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        rows.append({
            "smiles": s,
            "scaffold": scaf,
            "MW": Descriptors.MolWt(m),
            "LogP": Descriptors.MolLogP(m),
            "TPSA": Descriptors.TPSA(m),
            "HBD": Descriptors.NumHDonors(m),
            "HBA": Descriptors.NumHAcceptors(m),
            "QED": QED.qed(m),
        })
        fps.append(np.array(fp))
    X = np.stack(fps)
    iptm_pred = m_iptm.predict(X)
    mpae_pred = m_mpae.predict(X)
    df = pd.DataFrame(rows)
    df["iptm_pred"] = iptm_pred
    df["mpae_pred"] = mpae_pred
    df["cohort"] = label

    # SMARTS retention — THIQ-acrylamide warhead body
    wh = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
    df["has_warhead"] = df["smiles"].map(
        lambda s: bool(Chem.MolFromSmiles(s) and Chem.MolFromSmiles(s).HasSubstructMatch(wh))
    )

    # Diversity metrics
    n = len(df)
    n_unique_smi = df["smiles"].nunique()
    n_unique_scaf = df["scaffold"].nunique()
    # Intra-cohort mean Tc (sample 200 random pairs to bound compute)
    rng = np.random.default_rng(42)
    sample_n = min(n, 200)
    sample_idx = rng.choice(n, sample_n, replace=False)
    sample_fps = [fps[i] for i in sample_idx]
    tcs = []
    for i in range(sample_n):
        sims = DataStructs.BulkTanimotoSimilarity(sample_fps[i], sample_fps[i + 1:])
        tcs.extend(sims)
    mean_intra_tc = float(np.mean(tcs)) if tcs else None
    p90_intra_tc = float(np.percentile(tcs, 90)) if tcs else None

    summary = {
        "label": label,
        "n_input": len(smiles),
        "n_valid": n,
        "n_unique_smiles": n_unique_smi,
        "n_unique_murcko": n_unique_scaf,
        "frac_warhead_retained": float(df["has_warhead"].mean()),
        "iptm_pred_mean": float(df["iptm_pred"].mean()),
        "iptm_pred_p50": float(df["iptm_pred"].median()),
        "iptm_pred_p90": float(df["iptm_pred"].quantile(0.9)),
        "mpae_pred_mean": float(df["mpae_pred"].mean()),
        "mpae_pred_p50": float(df["mpae_pred"].median()),
        "MW_mean": float(df["MW"].mean()),
        "MW_p50": float(df["MW"].median()),
        "LogP_mean": float(df["LogP"].mean()),
        "QED_mean": float(df["QED"].mean()),
        "mean_intra_cohort_Tc": mean_intra_tc,
        "p90_intra_cohort_Tc": p90_intra_tc,
        "diversity_index_scaf_per_mol": n_unique_scaf / max(n, 1),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_smi", required=True)
    ap.add_argument("--base_smi", required=True)
    ap.add_argument("--bundle", default="models/boltz_distilled/boltz_distilled_v1.pkl")
    ap.add_argument("--out", default="data/distilled_rl_v1/comparison.json")
    args = ap.parse_args()

    print(f"loading bundle {args.bundle}")
    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    m_iptm = bundle["iptm_model"]
    m_mpae = bundle["mpae_model"]

    rl_smiles = [l.strip().split()[0] for l in open(args.rl_smi) if l.strip()]
    base_smiles = [l.strip().split()[0] for l in open(args.base_smi) if l.strip()]
    print(f"  rl cohort: {len(rl_smiles)} smiles")
    print(f"  base cohort: {len(base_smiles)} smiles")

    rl_df, rl_summary = score_cohort(rl_smiles, m_iptm, m_mpae, "RL")
    base_df, base_summary = score_cohort(base_smiles, m_iptm, m_mpae, "baseline")

    # ── REWARD-CONVERGENCE CHECKS (circular by construction — confirms RL worked) ──
    reward_convergence = {
        "warhead_retention_shift": rl_summary["frac_warhead_retained"] - base_summary["frac_warhead_retained"],
        # iptm_pred is the reward — going up is tautological, not validation
        "_REWARD_distilled_iptm_pred_shift_mean": rl_summary["iptm_pred_mean"] - base_summary["iptm_pred_mean"],
        "_REWARD_distilled_iptm_pred_shift_p50": rl_summary["iptm_pred_p50"] - base_summary["iptm_pred_p50"],
        "_REWARD_distilled_iptm_pred_shift_p90": rl_summary["iptm_pred_p90"] - base_summary["iptm_pred_p90"],
    }

    # ── INDEPENDENT CHECKS (the actual experiment) ──
    independent = {
        # Mode-collapse: did the RL collapse to a narrow chemical neighborhood?
        "scaf_diversity_ratio_RL_over_base": (
            rl_summary["diversity_index_scaf_per_mol"]
            / max(base_summary["diversity_index_scaf_per_mol"], 1e-9)
        ),
        "intra_tc_shift_mean": (
            (rl_summary["mean_intra_cohort_Tc"] or 0)
            - (base_summary["mean_intra_cohort_Tc"] or 0)
        ),
        "unique_smiles_ratio_RL_over_base": (
            rl_summary["n_unique_smiles"] / max(base_summary["n_unique_smiles"], 1)
        ),
        # Drug-likeness drift
        "MW_shift_mean": rl_summary["MW_mean"] - base_summary["MW_mean"],
        "LogP_shift_mean": rl_summary["LogP_mean"] - base_summary["LogP_mean"],
        "QED_shift_mean": rl_summary["QED_mean"] - base_summary["QED_mean"],
    }

    # Verdict
    verdict = []
    # Reward convergence (circular check)
    if reward_convergence["_REWARD_distilled_iptm_pred_shift_mean"] > 0.005:
        verdict.append("reward_converged_iptm_UP")
    else:
        verdict.append("reward_NOT_converged_iptm_flat")
    if reward_convergence["warhead_retention_shift"] > 0.1:
        verdict.append("warhead_grafted_more")
    elif reward_convergence["warhead_retention_shift"] < -0.1:
        verdict.append("warhead_grafted_less")
    else:
        verdict.append("warhead_unchanged")
    # Independent (actual experiment)
    if independent["scaf_diversity_ratio_RL_over_base"] < 0.5:
        verdict.append("DIVERSITY_COLLAPSE")
    elif independent["scaf_diversity_ratio_RL_over_base"] < 0.8:
        verdict.append("diversity_reduced")
    else:
        verdict.append("diversity_preserved")
    if abs(independent["MW_shift_mean"]) > 30 or abs(independent["LogP_shift_mean"]) > 0.5:
        verdict.append("druglikeness_drift")
    else:
        verdict.append("druglikeness_stable")

    out = {
        "_design_note": (
            "REWARD-CONVERGENCE shifts (prefixed _REWARD_) are circular: they measure "
            "whether RL optimized the chosen objective, not whether the chosen objective "
            "corresponds to better real-world geometry. INDEPENDENT shifts test mode-"
            "collapse + drug-likeness drift — the actual scientific questions. True "
            "geometry validation (Boltz cofold iptm) is deferred to a follow-up step."
        ),
        "RL_summary": rl_summary,
        "baseline_summary": base_summary,
        "reward_convergence_circular": reward_convergence,
        "independent_shifts": independent,
        "verdict_tags": verdict,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    print("\n=== Reward-convergence (CIRCULAR — confirms RL ran) ===")
    for k, v in reward_convergence.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== Independent shifts (the actual experiment) ===")
    for k, v in independent.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nverdict: {' | '.join(verdict)}")
    print("\nNOTE: true geometry validation needs Boltz-cofold sample on both cohorts (not run here).")

    # Save per-cohort dfs
    rl_df.to_csv(out_path.parent / "rl_cohort_scored.csv", index=False)
    base_df.to_csv(out_path.parent / "baseline_cohort_scored.csv", index=False)
    print(f"per-mol CSVs in {out_path.parent}")


if __name__ == "__main__":
    main()
