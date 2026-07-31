"""Score the DAP-finetuned 10k cohort AND the v2-cond baseline 10k with the same
pipeline (basic + geometric + FiLM pIC50), then emit md + json deltas.
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

from experiments.covft_geometric_options_bc import _compute_2d_one  # noqa: E402
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_PAT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def morgan_bit(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)


def morgan_np(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)
    arr = np.zeros(n, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def largest_frag_smi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags: return None
    return Chem.MolToSmiles(max(frags, key=lambda mm: mm.GetNumHeavyAtoms()))


def acryl_on_largest(smi):
    lf = largest_frag_smi(smi)
    if lf is None: return False
    m = Chem.MolFromSmiles(lf)
    if m is None: return False
    return bool(m.GetSubstructMatches(ACRYL_PAT))


def basic_metrics(df):
    smis = df["SMILES"].astype(str).tolist()
    n = len(smis)
    canons = [canon(s) for s in smis]
    n_valid = sum(1 for c in canons if c is not None)
    valid_c = [c for c in canons if c is not None]
    n_unique = len(set(valid_c))
    scs = [scaffold(s) for s in smis]
    sc_counter = Counter([s for s in scs if s])
    top_sc = sc_counter.most_common(1)[0] if sc_counter else (None, 0)
    n_acryl_anywhere = 0
    n_acryl_largest = 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m and m.GetSubstructMatches(ACRYL_PAT):
            n_acryl_anywhere += 1
        if acryl_on_largest(s):
            n_acryl_largest += 1
    fp_mol1 = morgan_bit(MOL1_SMI)
    tans = []
    qeds = []
    for c in canons:
        if c is None: continue
        fp = morgan_bit(c)
        if fp is not None:
            tans.append(float(DataStructs.TanimotoSimilarity(fp, fp_mol1)))
        m = Chem.MolFromSmiles(c)
        if m:
            try: qeds.append(float(QED.qed(m)))
            except Exception: pass
    return {
        "n_total": int(n),
        "n_valid": int(n_valid),
        "validity_rate": float(n_valid / n) if n else 0.0,
        "n_unique_canonical": int(n_unique),
        "unique_canonical_frac": float(n_unique / n_valid) if n_valid else 0.0,
        "top_scaffold_smi": top_sc[0],
        "top_scaffold_share": float(top_sc[1] / n_valid) if n_valid else 0.0,
        "acryl_retention_rate_anywhere": float(n_acryl_anywhere / n_valid) if n_valid else 0.0,
        "acryl_retention_rate_largest": float(n_acryl_largest / n_valid) if n_valid else 0.0,
        "tanimoto_to_mol1_median": float(np.median(tans)) if tans else None,
        "tanimoto_to_mol1_mean": float(np.mean(tans)) if tans else None,
        "qed_median": float(np.median(qeds)) if qeds else None,
        "qed_mean": float(np.mean(qeds)) if qeds else None,
    }


def geom_panel(df, cohort_id, workers, cache_csv):
    if cache_csv.exists():
        print(f"  [geom cached] {cache_csv}", flush=True)
        return pd.read_csv(cache_csv)
    tasks = [(i, smi) for i, smi in enumerate(df["SMILES"].astype(str).tolist())]
    print(f"  [geom] cohort={cohort_id} N={len(tasks)} workers={workers}", flush=True)
    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futures = [exc.submit(_compute_2d_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"idx": -1, "smi": "", "acryl_match": False,
                              "embed_ok": False, "dihedral_deg": None,
                              "planar_dev_deg": None,
                              "pre_reactivity_score": None,
                              "msg": f"fut_exc:{e}"})
            done += 1
            if done % 1000 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"    [geom {cohort_id}] {done}/{len(tasks)}  {rate:.1f}/s  ETA {eta/60:.1f}m", flush=True)
    df_g = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df_g.to_csv(cache_csv, index=False)
    return df_g


def geom_summary(df_g):
    ok = df_g["embed_ok"].astype(bool) & df_g["pre_reactivity_score"].notna()
    pre = df_g.loc[ok, "pre_reactivity_score"].astype(float).values
    dev = df_g.loc[ok, "planar_dev_deg"].astype(float).values
    return {
        "n_geom_evaluated": int(ok.sum()),
        "pre_reactivity_score_frac_ge_0p5": float((pre >= 0.5).mean()) if len(pre) else None,
        "planar_dev_median_deg": float(np.median(dev)) if len(dev) else None,
        "planar_dev_q25_deg": float(np.percentile(dev, 25)) if len(dev) else None,
        "planar_dev_q75_deg": float(np.percentile(dev, 75)) if len(dev) else None,
    }


def load_film(cache_path):
    ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    sc = StandardScaler()
    sc.mean_ = ckpt["scaler_mean"]; sc.scale_ = ckpt["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ckpt["anchor_embs"], ckpt["anchor_pIC50"]


def score_pIC50(smiles_list, film, sc, anchor_embs, anchor_pIC50, chunk_size=64):
    """Batch scoring: for each target chunk, tile anchors K times → single big
    forward pass. Anchor tensor shape (K*A, D); target tensor shape (K*A, D).
    """
    n = len(smiles_list)
    out = np.full(n, np.nan, dtype=np.float64)
    valid_fps = []; valid_idx = []
    for i, s in enumerate(smiles_list):
        arr = morgan_np(s)
        if arr is None: continue
        valid_fps.append(arr); valid_idx.append(i)
    if not valid_fps: return out
    fps_arr = np.array(valid_fps, dtype=np.float32)
    embs = torch.FloatTensor(sc.transform(fps_arr))
    n_anch = len(anchor_pIC50)
    n_val = len(valid_idx)
    anchor_pIC50_t = torch.FloatTensor(anchor_pIC50)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_val, chunk_size):
            end = min(start + chunk_size, n_val)
            K = end - start
            # anchor tile: (K*A, D) — anchors repeat inner
            anch_tile = anchor_embs.unsqueeze(0).expand(K, -1, -1).reshape(K * n_anch, -1)
            # target tile: (K*A, D) — each target repeats A times
            tgt_tile = embs[start:end].unsqueeze(1).expand(-1, n_anch, -1).reshape(K * n_anch, -1)
            deltas = film(anch_tile, tgt_tile).reshape(K, n_anch)
            preds = anchor_pIC50_t.unsqueeze(0) + deltas  # (K, A)
            means = preds.mean(dim=1).numpy()
            for k in range(K):
                out[valid_idx[start + k]] = float(means[k])
            # Print every ~1000 mols. chunk_size=64 → print every 16 chunks.
            if ((start // chunk_size) % 16 == 0) or (end == n_val):
                dt = time.time() - t0
                rate = end / max(dt, 1e-6)
                eta = (n_val - end) / max(rate, 1e-6)
                print(f"    [pIC50] {end}/{n_val}  {rate:.0f} mol/s  ETA {eta:.0f}s", flush=True)
    return out


def score_cohort(cohort_id, csv_path, workers, film_bundle, geom_dir):
    print(f"\n=== {cohort_id} ({csv_path.name}) ===", flush=True)
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} rows", flush=True)

    basic = basic_metrics(df)
    print(f"  basic: valid={basic['validity_rate']:.3f}  uniq={basic['unique_canonical_frac']:.3f}  "
          f"acryl_any={basic['acryl_retention_rate_anywhere']:.3f}  "
          f"acryl_largest={basic['acryl_retention_rate_largest']:.3f}  "
          f"top_sc={basic['top_scaffold_share']:.3f}  "
          f"Tc_mol1={basic['tanimoto_to_mol1_median']:.3f}  "
          f"QED={basic['qed_median']:.3f}", flush=True)

    geom = geom_summary(geom_panel(df, cohort_id, workers,
                                    geom_dir / f"m1a_v2_dap_score_geom_{cohort_id}.csv"))
    print(f"  geom : planar_med={geom['planar_dev_median_deg']}  "
          f"pre_react_ge_0p5={geom['pre_reactivity_score_frac_ge_0p5']}", flush=True)

    print(f"  scoring pIC50 with FiLM (this dominates runtime — ~5 mol/s)", flush=True)
    pIC50 = score_pIC50(df["SMILES"].astype(str).tolist(), *film_bundle)
    finite = pIC50[np.isfinite(pIC50)]
    pIC_summary = {
        "n_pIC50_evaluated": int(len(finite)),
        "pIC50_median": float(np.median(finite)) if len(finite) else None,
        "pIC50_mean": float(np.mean(finite)) if len(finite) else None,
        "pIC50_q75": float(np.percentile(finite, 75)) if len(finite) else None,
        "pIC50_q95": float(np.percentile(finite, 95)) if len(finite) else None,
        "frac_pIC50_ge_7p0": float((finite >= 7.0).mean()) if len(finite) else None,
        "frac_pIC50_ge_7p5": float((finite >= 7.5).mean()) if len(finite) else None,
    }
    print(f"  pIC50: median={pIC_summary['pIC50_median']:.3f}  "
          f"mean={pIC_summary['pIC50_mean']:.3f}  "
          f"frac>=7.0={pIC_summary['frac_pIC50_ge_7p0']:.3f}", flush=True)

    return {**basic, **geom, **pIC_summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dap_csv", default=str(PROJECT_ROOT / "data/m1a_v2_dap/cohort_A_10k.csv"))
    ap.add_argument("--baseline_csv", default=str(PROJECT_ROOT / "data/m1a_v2_ablation/cohort_A_10k.csv"))
    ap.add_argument("--film_cache", default=str(PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_model.pt"))
    ap.add_argument("--geom_dir", default=str(PROJECT_ROOT / "results/paper_evaluation"))
    ap.add_argument("--out_json", default=str(PROJECT_ROOT / "results/paper_evaluation/m1a_v2_dap_metrics.json"))
    ap.add_argument("--out_md", default=str(PROJECT_ROOT / "results/paper_evaluation/m1a_v2_dap_metrics.md"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    geom_dir = Path(args.geom_dir); geom_dir.mkdir(parents=True, exist_ok=True)
    film_bundle = load_film(Path(args.film_cache))
    print(f"[score] FiLM anchors: {len(film_bundle[3])}", flush=True)

    dap_res = score_cohort("dap", Path(args.dap_csv), args.workers, film_bundle, geom_dir)
    gc.collect()
    base_res = score_cohort("baseline", Path(args.baseline_csv), args.workers, film_bundle, geom_dir)

    # Deltas
    deltas = {}
    for k in dap_res:
        if isinstance(dap_res[k], (int, float)) and isinstance(base_res.get(k), (int, float)):
            deltas[k] = dap_res[k] - base_res[k]

    out = {
        "config": vars(args),
        "dap_cohort_A_10k": dap_res,
        "baseline_v2cond_cohort_A_10k": base_res,
        "deltas_dap_minus_baseline": deltas,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[score] wrote {args.out_json}", flush=True)

    # md
    md = []
    md.append("# M1a v2-cond + DAP RL — 10k Mol1-anchored cohort A metrics\n")
    md.append("| Metric | v2-cond (baseline) | v2-cond + DAP | Δ (DAP − base) |")
    md.append("|---|---|---|---|")
    metric_order = [
        ("validity_rate", 3), ("unique_canonical_frac", 3), ("top_scaffold_share", 3),
        ("acryl_retention_rate_anywhere", 3), ("acryl_retention_rate_largest", 3),
        ("tanimoto_to_mol1_median", 3), ("qed_median", 3),
        ("planar_dev_median_deg", 2), ("pre_reactivity_score_frac_ge_0p5", 3),
        ("pIC50_median", 3), ("pIC50_mean", 3), ("pIC50_q75", 3), ("pIC50_q95", 3),
        ("frac_pIC50_ge_7p0", 3), ("frac_pIC50_ge_7p5", 3),
    ]
    for k, nd in metric_order:
        b = base_res.get(k); d = dap_res.get(k); delta = deltas.get(k)
        b_s = f"{b:.{nd}f}" if isinstance(b, (int, float)) else "—"
        d_s = f"{d:.{nd}f}" if isinstance(d, (int, float)) else "—"
        delta_s = f"{delta:+.{nd}f}" if isinstance(delta, (int, float)) else "—"
        md.append(f"| {k} | {b_s} | {d_s} | {delta_s} |")
    md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[score] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
