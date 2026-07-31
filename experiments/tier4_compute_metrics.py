#!/usr/bin/env python3
"""Tier 4 cheap metrics — RL-fine-tuned Mol2Mol cohorts.

Computes per-cohort: N total / unique-canonical / valid; acryl% on largest
fragment (strict SMARTS); mean/std of MW, QED, FiLMDelta pIC50; unique Murcko
scaffolds; Brenk/Lipinski/Veber flags.

Does NOT compute: AD-CovDock Vina, d(Cβ-SG), BD angle, hinge, Cβ-SASA, Boltz.
Vina cost estimate is printed at the bottom.

Inputs: each cohort's `cohort_all.csv` (columns: smiles, nll, warhead_acryl,
mw, qed, seed_idx, seed_smi).

Output: JSON per cohort + summary at `results/paper_evaluation/tier4_rl_cohorts/`.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT / "results/paper_evaluation/tier4_rl_cohorts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACRYL_STRICT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N;!H2]")
BRENK_CAT = FilterCatalog.FilterCatalog(
    FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK
)

COHORTS = {
    "EXP6_v3": {
        "path": PROJECT / "data/reinvent4_mol2mol_exp6_v3_5k_per_seed/cohort_all.csv",
        "config": "Warhead-tokens v1 prior, **no RL**; sampled with `[ACRYLAMIDE]` token prefix from 21 ZAP70-acryl seeds × 5K each. Pre-RL baseline cohort (reference for RL-induced shifts).",
        "rl_steps": 0,
        "prior": "Mol2Mol warhead-tokens v1 (vocab 132 with `[ACRYLAMIDE]`/`[CHLOROACETAMIDE]`/... tokens)",
        "reward": "n/a (no RL)",
    },
    "EXP6_v4": {
        "path": PROJECT / "data/exp6_v4_5k_per_seed/cohort_all.csv",
        "config": "EXP2 covalent FT prior (**no warhead tokens**) + RL with FiLMDelta pIC50 0.50 + AD-CovDock Vina 0.40 + QED 0.10, 50 steps. **RL hijacked the reward**: policy abandoned acrylamide (99%→0.2%) so the meeko CovalentBuilder tether failed → Vina success collapsed 76%→2% over the run.",
        "rl_steps": 50,
        "prior": "Mol2Mol EXP2 covalent FT (no warhead tokens, vocab 128)",
        "reward": "FiLMDelta 0.50 + Vina 0.40 + QED 0.10",
    },
    "EXP6_v5": {
        "path": PROJECT / "data/exp6_v5_5k_per_seed/cohort_all.csv",
        "config": "Warhead-tokens v1 prior + **same Vina-in-loop reward as v4** (FiLM 0.50 + Vina 0.40 + QED 0.10), 50 steps. Warhead-token conditioning prevented reward-hijacking: acryl retained 99%→99%, Vina success held 97%, and drug-likeness IMPROVED (QED 0.27→0.50, Lipinski-ok 27%→86%, MW −117 Da).",
        "rl_steps": 50,
        "prior": "Mol2Mol warhead-tokens v1 (vocab 132)",
        "reward": "FiLMDelta 0.50 + Vina 0.40 + QED 0.10",
    },
    "EXP2_V2_RL_v2": {
        "path": PROJECT / "data/exp2_v2_rl_v2_5k_per_seed/cohort_all.csv",
        "config": "EXP2 covalent FT prior (no warhead tokens) + RL ablation **without Vina-in-loop** (FiLM 0.50 + acryl-SMARTS 0.40 + QED 0.10), 50 steps. Tests whether SMARTS-matching alone (no expensive Vina) suffices for covalent constraint when there are no warhead tokens.",
        "rl_steps": 50,
        "prior": "Mol2Mol EXP2 covalent FT (no warhead tokens, vocab 128)",
        "reward": "FiLMDelta 0.50 + acryl-SMARTS 0.40 + QED 0.10",
    },
}


def murcko_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s.GetNumAtoms() > 0 else ""
    except Exception:
        return None


def lipinski_ok(mw, logp, hbd, hba):
    return int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)


def veber_ok(rotb, tpsa):
    return int(rotb <= 10 and tpsa <= 140)


def compute_one(name, meta):
    path = meta["path"]
    if not path.exists():
        print(f"[{name}] SKIP: {path} does not exist", flush=True)
        return None
    print(f"\n=== {name} ===", flush=True)
    print(f"  loading {path}...", flush=True)
    df = pd.read_csv(path)
    n_total = len(df)
    print(f"  total rows: {n_total}", flush=True)

    valid_smiles = []
    largest_frag_acryl = []
    n_disconnected = 0
    mws, logps, qeds, tpsas, hbas, hbds, rotbs = [], [], [], [], [], [], []
    ring_scaffolds = []  # only mols with ≥1 ring
    n_acyclic = 0
    brenk_alerts = []
    t0 = time.time()
    for i, smi in enumerate(df["smiles"].astype(str)):
        if i % 10000 == 0 and i > 0:
            dt = time.time() - t0
            print(f"  ...{i}/{n_total}  ({dt:.1f}s)", flush=True)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        valid_smiles.append(smi)
        frags = Chem.GetMolFrags(m, asMols=True)
        if len(frags) > 1:
            n_disconnected += 1
        big = max(frags, key=lambda x: x.GetNumHeavyAtoms()) if frags else m
        largest_frag_acryl.append(int(big.HasSubstructMatch(ACRYL_STRICT)))
        # All physchem descriptors on LARGEST fragment for consistency with acryl% and Brenk
        mws.append(Descriptors.MolWt(big))
        try:
            logps.append(Descriptors.MolLogP(big))
        except Exception:
            logps.append(np.nan)
        try:
            qeds.append(QED.qed(big))
        except Exception:
            qeds.append(np.nan)
        tpsas.append(Descriptors.TPSA(big))
        hbas.append(Descriptors.NumHAcceptors(big))
        hbds.append(Descriptors.NumHDonors(big))
        rotbs.append(Descriptors.NumRotatableBonds(big))
        sc = murcko_scaffold(Chem.MolToSmiles(big))
        if sc is None or sc == "":
            n_acyclic += 1
        else:
            ring_scaffolds.append(sc)
        brenk_alerts.append(int(BRENK_CAT.HasMatch(big)))

    n_valid = len(valid_smiles)
    if n_valid == 0:
        out = {
            "name": name, "config": meta["config"], "prior": meta["prior"],
            "reward": meta["reward"], "rl_steps": meta["rl_steps"],
            "n_total": int(n_total), "n_valid": 0,
            "validity_pct": 0.0, "wallclock_s": round(time.time() - t0, 1),
            "_warning": "no valid molecules — all other metrics omitted",
        }
        out_path = OUT_DIR / f"{name}_metrics.json"
        out_path.write_text(json.dumps(out, indent=2))
        return out

    mws = np.array(mws); logps = np.array(logps); qeds = np.array(qeds)
    tpsas = np.array(tpsas); hbas = np.array(hbas); hbds = np.array(hbds)
    rotbs = np.array(rotbs); brenk_alerts = np.array(brenk_alerts)
    largest_frag_acryl = np.array(largest_frag_acryl)

    lip = np.array([
        lipinski_ok(mws[i], logps[i], hbds[i], hbas[i])
        for i in range(len(mws))
    ])
    veb = np.array([veber_ok(rotbs[i], tpsas[i]) for i in range(len(mws))])
    out = {
        "name": name,
        "config": meta["config"],
        "prior": meta["prior"],
        "reward": meta["reward"],
        "rl_steps": meta["rl_steps"],
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "validity_pct": round(100 * n_valid / max(1, n_total), 2),
        "n_disconnected": int(n_disconnected),
        "disconnected_pct": round(100 * n_disconnected / max(1, n_valid), 2),
        "n_acryl_largest_frag": int(largest_frag_acryl.sum()),
        "acryl_largest_frag_pct": round(100 * largest_frag_acryl.mean(), 2),
        "n_unique_murcko_scaffolds": int(len(set(ring_scaffolds))),
        "n_acyclic": int(n_acyclic),
        "mw_mean": round(float(mws.mean()), 2),
        "mw_std": round(float(mws.std()), 2),
        "mw_median": round(float(np.median(mws)), 2),
        "mw_p10": round(float(np.percentile(mws, 10)), 2),
        "mw_p90": round(float(np.percentile(mws, 90)), 2),
        "logp_mean": round(float(np.nanmean(logps)), 3),
        "qed_mean": round(float(np.nanmean(qeds)), 3),
        "qed_median": round(float(np.nanmedian(qeds)), 3),
        "qed_std": round(float(np.nanstd(qeds)), 3),
        "tpsa_mean": round(float(tpsas.mean()), 2),
        "rotbonds_mean": round(float(rotbs.mean()), 2),
        "lipinski_compliant_pct": round(100 * lip.mean(), 2),
        "veber_compliant_pct": round(100 * veb.mean(), 2),
        "brenk_alert_pct": round(100 * brenk_alerts.mean(), 2),
        "wallclock_s": round(time.time() - t0, 1),
    }
    out_path = OUT_DIR / f"{name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {out_path}", flush=True)
    print(f"  validity: {out['validity_pct']}%, acryl-largest: {out['acryl_largest_frag_pct']}%, scaffolds: {out['n_unique_murcko_scaffolds']}", flush=True)
    print(f"  MW: {out['mw_mean']}±{out['mw_std']}, QED: {out['qed_mean']}±{out['qed_std']}", flush=True)
    print(f"  Brenk-alert: {out['brenk_alert_pct']}%, Lipinski-ok: {out['lipinski_compliant_pct']}%, Veber-ok: {out['veber_compliant_pct']}%", flush=True)
    return out


def main():
    summary = []
    for name, meta in COHORTS.items():
        r = compute_one(name, meta)
        if r:
            summary.append(r)

    print("\n=== Vina cost estimate ===", flush=True)
    total_for_vina = sum(r["n_valid"] for r in summary)
    sec_per_mol = 1.83  # measured on V100: 6 parallel workers, already amortized
    single_v100_hours = total_for_vina * sec_per_mol / 3600
    print(f"  Mols to dock: {total_for_vina:,}", flush=True)
    print(f"  Per-mol time (6 workers, V100): {sec_per_mol} s", flush=True)
    print(f"  Single-V100 wallclock: {single_v100_hours:.1f} h", flush=True)
    print(f"  Two-V100 wallclock (parallel, assumes linear scaling): {single_v100_hours/2:.1f} h", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "cohorts": summary,
            "vina_cost_estimate": {
                "total_mols": total_for_vina,
                "sec_per_mol_6workers": sec_per_mol,
                "single_v100_hours": round(total_for_vina * sec_per_mol / 3600, 1),
                "two_v100_hours": round(total_for_vina * sec_per_mol / 3600 / 2, 1),
            },
        }, f, indent=2)
    print(f"\n  → {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
