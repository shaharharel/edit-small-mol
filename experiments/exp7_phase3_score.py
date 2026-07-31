#!/usr/bin/env python3
"""EXP7 LO eval — Phase 3 LOCAL scoring of cohorts.

For each cohort CSV in data/exp7_lo_benchmark/_rl/<pair_id>_<strategy>/sampled.csv:
  - Load target-specific FiLMDelta + 100-anchor pool (or anchor only) for pIC50
  - Compute: warhead match (strict + generic), Tc-to-drug (max + p95),
             Murcko match (drug), pred pIC50 (median, p95), QED, hit rates,
             internal diversity (sample 500)
  - Save per-cohort {pair_id}_{strategy}_metrics.json + cohort.csv with extra columns

Writes consolidated `exp7_all_cohorts_scored.csv` and `exp7_phase3_summary.json`.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

from experiments.exp7_pair_loader import load_all_pairs, TARGET_CHEMBL

PHASE1_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_phase1"
RL_BASE = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "_rl"
OUT_DIR = PROJECT_ROOT / "data" / "exp7_lo_benchmark"
SUMMARY_JSON = OUT_DIR / "exp7_phase3_summary.json"
ALL_COHORTS_CSV = OUT_DIR / "exp7_all_cohorts_scored.csv"

STRATEGIES = ["A", "B", "baseline_prior_anchor", "baseline_prior_pool", "baseline_prior_rand",
              "mol2mol_baseline", "mol2mol_RL"]


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


def smarts_match_pct(smiles_list, smarts):
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return 0.0
    hits = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if m.HasSubstructMatch(pat):
            hits += 1
    return 100.0 * hits / max(1, len(smiles_list))


# Broad-acceptance Michael-acceptor patterns for "generic warhead" check (handles
# substituted acrylamides C/C=C/C(=O)N, ynamides CC#CC(=O)N, etc. — anything that
# would actually be a covalent kinase warhead).
_BROAD_WH_PATS = [
    Chem.MolFromSmarts("[#6]=[#6][C](=O)[#7]"),
    Chem.MolFromSmarts("[#6]#[#6][C](=O)[#7]"),
]


def broad_warhead_pct(smiles_list):
    hits = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if any(p and m.HasSubstructMatch(p) for p in _BROAD_WH_PATS):
            hits += 1
    return 100.0 * hits / max(1, len(smiles_list))


def load_film_model(target_dir: Path):
    ckpt = torch.load(target_dir / "filmdelta.pt", map_location="cpu", weights_only=False)
    sig = inspect.signature(FiLMDeltaMLP.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    hp = {k: v for k, v in ckpt["hyperparameters"].items() if k in allowed}
    m = FiLMDeltaMLP(**hp)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def score_pic50_film(model, anchor_fps_arr, anchor_pics, query_smiles, query_batch=64):
    """Returns array of predicted pIC50 for each query, averaging delta over anchor pool.

    Batched: for each batch of Q queries, expand to (Q*A, 2048) for both anchor and target,
    one forward pass per batch. Speeds up >10x vs per-query loop.
    """
    out = np.zeros(len(query_smiles), dtype=np.float32)
    valid_idx, valid_fps = [], []
    for i, smi in enumerate(query_smiles):
        fp = _fp(smi)
        if fp is None:
            continue
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_fps.append(arr)
        valid_idx.append(i)
    if not valid_fps:
        return out
    fps_arr = np.stack(valid_fps)
    n_anchors = anchor_fps_arr.shape[0]
    anchor_t = torch.from_numpy(anchor_fps_arr).float()
    anchor_pics_t = torch.from_numpy(anchor_pics).float()
    with torch.no_grad():
        for start in range(0, len(valid_idx), query_batch):
            end = min(start + query_batch, len(valid_idx))
            Q = end - start
            q_fps = torch.from_numpy(fps_arr[start:end]).float()  # (Q, 2048)
            # Build (Q*A, 2048) tensors
            anchors_rep = anchor_t.unsqueeze(0).expand(Q, n_anchors, -1).reshape(Q * n_anchors, -1)
            queries_rep = q_fps.unsqueeze(1).expand(Q, n_anchors, -1).reshape(Q * n_anchors, -1)
            deltas = model(anchors_rep, queries_rep).reshape(Q, n_anchors)
            abs_per_query = (anchor_pics_t.unsqueeze(0) + deltas).mean(dim=1).numpy()
            for k in range(Q):
                out[valid_idx[start + k]] = float(abs_per_query[k])
    return out


def load_anchor_arrays_for_pair(pair: Dict) -> tuple[np.ndarray, np.ndarray]:
    """Use the pair's anchor + its 100-anchor B pool combined for FiLM scoring."""
    target_dir = PHASE1_BASE / pair["target_key"]
    anchors_dir = target_dir / "anchors"
    pool_csv = anchors_dir / f"{pair['pair_id']}_b_pool.csv"
    if pool_csv.exists():
        df = pd.read_csv(pool_csv)
        smis = df["smiles"].tolist()
        pics = df["pIC50"].astype(float).tolist()
    else:
        smis = [pair["anchor_smiles"]]
        pics = [float(pair.get("anchor_pIC50", 6.5) or 6.5)]
    fps, ps = [], []
    for s, p in zip(smis, pics):
        fp = _fp(s)
        if fp is None:
            continue
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        ps.append(float(p) if np.isfinite(p) else 6.5)
    return np.stack(fps).astype(np.float32), np.array(ps, dtype=np.float32)


def internal_diversity(smiles_list, sample_n=200):
    if len(smiles_list) < 5:
        return float("nan")
    rng = np.random.default_rng(42)
    if len(smiles_list) > sample_n:
        idx = rng.choice(len(smiles_list), sample_n, replace=False)
        smis = [smiles_list[i] for i in idx]
    else:
        smis = smiles_list
    fps = [_fp(s) for s in smis]
    fps = [f for f in fps if f is not None]
    if len(fps) < 5:
        return float("nan")
    tcs = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        tcs.extend(sims)
    return 1.0 - float(np.mean(tcs))


def score_cohort(pair: Dict, strategy: str, film_model, anchor_fps, anchor_pics, warhead_specs: Dict) -> Dict | None:
    cell_dir = RL_BASE / f"{pair['pair_id']}_{strategy}"
    cohort_path = cell_dir / "sampled.csv"
    if not cohort_path.exists() or cohort_path.stat().st_size < 500:
        return None
    try:
        df = pd.read_csv(cohort_path)
    except Exception as e:
        return {"status": "csv_read_fail", "error": str(e)}
    smi_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[0])
    smiles_list = df[smi_col].dropna().astype(str).tolist()
    if len(smiles_list) < 100:
        return {"status": "too_few", "n": len(smiles_list)}

    drug_smi = pair["drug_smiles"]
    drug_fp = _fp(drug_smi)
    drug_scf = _murcko(drug_smi)

    # Filter to valid mols
    valid_smis = []
    valid_fps = []
    for s in smiles_list:
        f = _fp(s)
        if f is None:
            continue
        valid_smis.append(s)
        valid_fps.append(f)

    if not valid_smis:
        return {"status": "no_valid_mols"}

    # Tc to drug
    tc_drug = np.array(DataStructs.BulkTanimotoSimilarity(drug_fp, valid_fps), dtype=float)

    # Predicted pIC50 (FiLM)
    pred_pic = score_pic50_film(film_model, anchor_fps, anchor_pics, valid_smis)

    # Warhead match
    warhead_strict_pct = smarts_match_pct(valid_smis, warhead_specs["smarts_strict"])
    warhead_generic_pct = smarts_match_pct(valid_smis, warhead_specs["smarts_generic"])
    warhead_broad_pct = broad_warhead_pct(valid_smis)  # any C=C–C(=O)–N or C#C-C(=O)-N

    # Murcko match to drug
    murcko_hits = 0
    for s in valid_smis:
        if _murcko(s) == drug_scf:
            murcko_hits += 1
    murcko_pct = 100.0 * murcko_hits / len(valid_smis)

    # QED
    qeds = []
    for s in valid_smis[:1500]:  # sample for speed
        try:
            qeds.append(QED.qed(Chem.MolFromSmiles(s)))
        except Exception:
            pass
    qed_median = float(np.median(qeds)) if qeds else float("nan")

    # Diversity
    div = internal_diversity(valid_smis)

    metrics = {
        "status": "ok",
        "pair_id": pair["pair_id"],
        "target_key": pair["target_key"],
        "strategy": strategy,
        "drug_name": pair["drug_name"],
        "anchor_pIC50": pair.get("anchor_pIC50"),
        "drug_pIC50": pair.get("drug_pIC50"),
        "tc_anchor_drug": pair.get("tc_anchor_drug"),
        "delta_pIC50_true": pair.get("delta_pIC50"),
        "n_cohort_total": len(smiles_list),
        "n_cohort_valid": len(valid_smis),
        "valid_frac": len(valid_smis) / max(1, len(smiles_list)),
        "max_tc_drug": float(tc_drug.max()),
        "tc_drug_p95": float(np.quantile(tc_drug, 0.95)),
        "tc_drug_median": float(np.median(tc_drug)),
        "n_tc05": int((tc_drug >= 0.5).sum()),
        "n_tc06": int((tc_drug >= 0.6).sum()),
        "n_tc07": int((tc_drug >= 0.7).sum()),
        "n_exact_drug": int((tc_drug >= 0.99).sum()),
        "murcko_match_pct": murcko_pct,
        "warhead_strict_pct": warhead_strict_pct,
        "warhead_generic_pct": warhead_generic_pct,
        "warhead_broad_pct": warhead_broad_pct,
        "pred_pic50_median": float(np.median(pred_pic)),
        "pred_pic50_p95": float(np.quantile(pred_pic, 0.95)),
        "pred_pic50_max": float(pred_pic.max()),
        "n_predpic_ge7": int((pred_pic >= 7.0).sum()),
        "qed_median": qed_median,
        "diversity": div,
        # best candidate (max tc_drug)
        "best_tc_drug_smiles": valid_smis[int(np.argmax(tc_drug))],
        "best_pred_pic_smiles": valid_smis[int(np.argmax(pred_pic))],
    }

    # save scored cohort csv (with tc_drug + pred_pic50 columns)
    out = pd.DataFrame({"smiles": valid_smis, "tc_drug": tc_drug, "pred_pIC50": pred_pic})
    out.to_csv(cell_dir / "cohort_scored.csv", index=False)
    return metrics


def main():
    pairs = load_all_pairs()
    print(f"Phase 3 scoring: {len(pairs)} pairs x up to {len(STRATEGIES)} strategies")

    # Load FiLM model per target
    target_keys = sorted(set(p["target_key"] for p in pairs))
    film_models = {}
    warhead_specs = {}
    for tk in target_keys:
        td = PHASE1_BASE / tk
        if not (td / "filmdelta.pt").exists():
            print(f"  WARNING: {tk} filmdelta.pt missing — will skip")
            continue
        film_models[tk] = load_film_model(td)
        with open(td / "warhead_smarts.json") as f:
            warhead_specs[tk] = json.load(f)
        print(f"  loaded {tk} film + warhead specs")

    all_metrics = []
    for pair in pairs:
        tk = pair["target_key"]
        if tk not in film_models:
            continue
        anchor_fps, anchor_pics = load_anchor_arrays_for_pair(pair)
        for strategy in STRATEGIES:
            m = score_cohort(pair, strategy, film_models[tk], anchor_fps, anchor_pics, warhead_specs[tk])
            if m is None:
                continue  # no cohort yet
            if m.get("status") != "ok":
                print(f"  {pair['pair_id']}/{strategy}: {m.get('status')}")
                continue
            all_metrics.append(m)
            print(f"  {pair['pair_id']}/{strategy}: n={m['n_cohort_valid']}, max_tc={m['max_tc_drug']:.3f}, "
                  f"n_tc05={m['n_tc05']}, pred_pIC50_med={m['pred_pic50_median']:.2f}, warhead={m['warhead_strict_pct']:.0f}%")

    if not all_metrics:
        print("\nNo cohorts scored. Exiting.")
        SUMMARY_JSON.write_text(json.dumps({"n_scored": 0, "note": "no cohorts found yet"}, indent=2))
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(ALL_COHORTS_CSV, index=False)

    summary = {
        "n_pairs": len(pairs),
        "n_cohorts_scored": len(all_metrics),
        "by_strategy": df["strategy"].value_counts().to_dict(),
        "by_target": df["target_key"].value_counts().to_dict(),
        "median_max_tc_drug": float(df["max_tc_drug"].median()),
        "median_n_tc05": float(df["n_tc05"].median()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {ALL_COHORTS_CSV} ({len(df)} cohorts)")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
