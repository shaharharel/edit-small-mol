#!/usr/bin/env python3
"""Build a backend-ready CSV for the 4 Tier 4 overnight cohorts.

Output schema matches `results/paper_evaluation/all_methods_bulk_scored_v4.csv`
so backend.py can merge it without code changes — we just point it at one
extra DATA_FILE.

Columns produced (filled when possible, NaN otherwise):
  smiles, method, MW, LogP, TPSA, HBA, HBD, RotBonds, QED, HeavyAtoms, Rings,
  Tc_to_Mol1, max_Tc_train (NaN), mean_top10_Tc_train (NaN),
  SAScore, PAINS_alerts, warhead_intact, row_id (assigned by backend),
  pIC50_method (NaN), pIC50_mean (NaN), pIC50_std (NaN),
  delta_vs_mol1 (NaN), direct_delta_from_mol1 (NaN),
  anchor_wins (NaN), anchor_wins_ge7 (NaN),
  shape_Tc_seed (NaN), esp_sim_seed (NaN), warhead_dev_deg (NaN)

The user decision (2026-06-05): KEEP backend's warhead_intact==True filter.
EXP6_v4 will collapse to ~160 surviving mols in the dashboard; the full
80K and the 0.2% acryl-retention story lives in tier4_rl_cohorts/tier4_report.md.

All physchem is computed on the LARGEST FRAGMENT (which is the whole molecule
since our sampling scripts already drop multi-fragment SMILES at write-time —
verified 0 disconnected rows across all 4 cohorts).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, FilterCatalog
from rdkit.DataStructs import TanimotoSimilarity

# SAScore: ships with RDKit Contrib
try:
    sys.path.insert(0, str(Path(Chem.__file__).parent / ".." / ".." / ".." / "share" / "RDKit" / "Contrib" / "SA_Score"))
    import sascorer
    HAVE_SASCORE = True
except Exception:
    HAVE_SASCORE = False

RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_CSV = PROJECT / "results/paper_evaluation/tier4_overnight_bulk.csv"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(
    Chem.MolFromSmiles(MOL1_SMILES), 2, 2048
)
ACRYL_STRICT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N;!H2]")
PAINS_CAT = FilterCatalog.FilterCatalog(
    FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
)

# (cohort_name, cohort_path, method_label_for_dashboard)
COHORTS = [
    ("EXP6_v3",
     PROJECT / "data/reinvent4_mol2mol_exp6_v3_5k_per_seed/cohort_all.csv",
     "Tier 4 — EXP6_v3 (Warhead-tokens v1, no RL, 58.8K)"),
    ("EXP2_V2_RL_v2",
     PROJECT / "data/exp2_v2_rl_v2_5k_per_seed/cohort_all.csv",
     "Tier 4 — EXP2_V2_RL_v2 (RL no Vina, SMARTS reward, 82.7K)"),
    ("EXP6_v4",
     PROJECT / "data/exp6_v4_5k_per_seed/cohort_all.csv",
     "Tier 4 — EXP6_v4 (RL+Vina+QED, NO warhead tokens, 80.2K)"),
    ("EXP6_v5",
     PROJECT / "data/exp6_v5_5k_per_seed/cohort_all.csv",
     "Tier 4 — EXP6_v5 (RL+Vina+QED, warhead tokens, 95.1K)"),
]


def compute_one(name, path, method_label):
    print(f"\n=== {name}  ({method_label}) ===", flush=True)
    if not path.exists():
        print(f"  SKIP — {path} not found", flush=True)
        return None
    df = pd.read_csv(path)
    n = len(df)
    print(f"  {n:,} input rows", flush=True)
    rows = []
    t0 = time.time()
    n_warhead = 0
    for i, smi in enumerate(df["smiles"].astype(str)):
        if i % 10000 == 0 and i > 0:
            dt = time.time() - t0
            print(f"    ...{i:,}/{n:,}  ({dt:.0f}s, ~{int(i/dt)}/s)", flush=True)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        # All cohorts already deduped to single-fragment by the sampling script;
        # be defensive with GetMolFrags here too — picks largest if any escapes.
        frags = Chem.GetMolFrags(m, asMols=True)
        big = max(frags, key=lambda x: x.GetNumHeavyAtoms()) if frags else m
        warhead = int(big.HasSubstructMatch(ACRYL_STRICT))
        if warhead:
            n_warhead += 1
        try:
            sa = float(sascorer.calculateScore(big)) if HAVE_SASCORE else np.nan
        except Exception:
            sa = np.nan
        # Tanimoto to Mol-1 (Morgan r=2, 2048 bits)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(big, 2, 2048)
            tc = float(TanimotoSimilarity(fp, MOL1_FP))
        except Exception:
            tc = np.nan
        pains = int(PAINS_CAT.HasMatch(big))
        rows.append({
            "smiles": Chem.MolToSmiles(big, canonical=True, isomericSmiles=False),
            "method": method_label,
            "MW": round(Descriptors.MolWt(big), 2),
            "LogP": round(Descriptors.MolLogP(big), 3),
            "TPSA": round(Descriptors.TPSA(big), 2),
            "HBA": int(Descriptors.NumHAcceptors(big)),
            "HBD": int(Descriptors.NumHDonors(big)),
            "RotBonds": int(Descriptors.NumRotatableBonds(big)),
            "QED": round(QED.qed(big), 4),
            "HeavyAtoms": int(big.GetNumHeavyAtoms()),
            "Rings": int(Descriptors.RingCount(big)),
            "Tc_to_Mol1": round(tc, 4) if not np.isnan(tc) else np.nan,
            "max_Tc_train": np.nan,
            "mean_top10_Tc_train": np.nan,
            "SAScore": round(sa, 3) if not np.isnan(sa) else np.nan,
            "PAINS_alerts": pains,
            "warhead_intact": bool(warhead),
            "pIC50_method": np.nan,
            "pIC50_mean": np.nan,
            "pIC50_std": np.nan,
            "delta_vs_mol1": np.nan,
            "direct_delta_from_mol1": np.nan,
            "anchor_wins": np.nan,
            "anchor_wins_ge7": np.nan,
            "shape_Tc_seed": np.nan,
            "esp_sim_seed": np.nan,
            "warhead_dev_deg": np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"  computed {len(out):,} rows in {time.time()-t0:.0f}s", flush=True)
    print(f"  warhead_intact survivors: {n_warhead:,} ({100*n_warhead/max(1,len(out)):.1f}%)", flush=True)
    return out


def add_film_pIC50(df: pd.DataFrame) -> pd.DataFrame:
    """Run FiLMDelta anchor-based pIC50 inference on the SMILES column.

    Uses the cached model at results/paper_evaluation/reinvent4_film_model.pt.
    CPU only (CUDA_VISIBLE_DEVICES gets cleared by the scorer module). For
    317K mols on Mac CPU this takes ~30-60 min.
    """
    print("\n=== FiLM pIC50 inference ===", flush=True)
    sys.path.insert(0, str(PROJECT / "experiments"))
    # Importing the module loads the model
    import reinvent4_film_scorer as fs
    model, scaler, anchor_embs, anchor_pIC50 = fs.load_film_model()
    print(f"  model loaded with {len(anchor_pIC50)} anchors", flush=True)
    smis = df["smiles"].astype(str).tolist()
    n = len(smis)
    pic50s = [float("nan")] * n
    CHUNK = 2000
    t0 = time.time()
    for i in range(0, n, CHUNK):
        chunk = smis[i:i + CHUNK]
        scores = fs.score_smiles(chunk, model, scaler, anchor_embs, anchor_pIC50)
        pic50s[i:i + len(scores)] = scores
        if (i // CHUNK) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(1, elapsed)
            eta = (n - i - len(chunk)) / max(1, rate) / 60
            print(f"  ...{i+len(chunk):,}/{n:,}  ({elapsed:.0f}s, "
                  f"{rate:.0f}/s, ETA {eta:.1f} min)", flush=True)
    df = df.copy()
    df["pIC50_method"] = pic50s
    df["pIC50_film"] = pic50s
    # We don't have a Mol-1 absolute pIC50 baseline here, so delta is just
    # FiLM(mol_target) - FiLM(Mol-1). Compute once.
    mol1_score = fs.score_smiles([MOL1_SMILES], model, scaler,
                                  anchor_embs, anchor_pIC50)[0]
    df["delta_vs_mol1"] = df["pIC50_method"] - mol1_score
    df["direct_delta_from_mol1"] = df["delta_vs_mol1"]
    df["anchor_wins"] = (df["pIC50_method"] > mol1_score).astype("Int64")
    df["anchor_wins_ge7"] = ((df["pIC50_method"] > mol1_score)
                              & (df["pIC50_method"] >= 7.0)).astype("Int64")
    print(f"  Mol-1 FiLM pIC50 (reference): {mol1_score:.3f}", flush=True)
    valid = ~np.isnan(pic50s)
    print(f"  Mean pIC50: {np.nanmean(pic50s):.3f}  "
          f"·  mean delta vs Mol-1: {np.nanmean(df['delta_vs_mol1']):.3f}  "
          f"·  anchor_wins: {df['anchor_wins'].sum()}/{n}", flush=True)
    return df


def main():
    parts = []
    for name, path, label in COHORTS:
        sub = compute_one(name, path, label)
        if sub is not None:
            parts.append(sub)
    if not parts:
        print("nothing computed", flush=True)
        return
    combined = pd.concat(parts, ignore_index=True)

    # Phase 1 — physchem (already in combined). Snapshot to disk before FiLM in case FiLM crashes.
    snapshot = OUT_CSV.with_suffix(".phase1_physchem.csv")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(snapshot, index=False)
    print(f"\n  Phase-1 snapshot written → {snapshot}  ({len(combined):,} rows)", flush=True)

    # Phase 2 — FiLM pIC50 + deltas. SKIP if SKIP_FILM env var set (used when
    # we want to ship FiLM to a faster machine like ai-chem / V100).
    import os as _os
    if _os.environ.get("SKIP_FILM"):
        print("  SKIP_FILM set — exiting after phase-1 snapshot. Run "
              "tier4_film_phase2.py on the target machine.", flush=True)
        return
    combined = add_film_pIC50(combined)

    combined.to_csv(OUT_CSV, index=False)
    print(f"\n=== TOTAL: {len(combined):,} rows  →  {OUT_CSV} ===", flush=True)
    print("Per-method counts:")
    print(combined["method"].value_counts().to_string(), flush=True)
    print("Per-method warhead_intact counts (post-filter survivors):")
    print(combined.groupby("method")["warhead_intact"].sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
