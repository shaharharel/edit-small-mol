"""Score Variant 3 (v2-cond + DAP + DPO) 10k cohort with full metric panel.

Compares against:
  - v2-cond baseline (data/m1a_v2_ablation_10k/cohort_A.csv)
  - Variant 1 DAP (data/m1a_v2_dap/cohort_A_10k.csv)   [if present]

Metrics (per cohort):
  - Validity / uniqueness / top-scaffold share
  - Acrylamide retention (SMARTS on largest fragment)
  - Tanimoto to Mol1 (median/mean)
  - FiLMDelta pIC50 (mean/median/max) — anchor-based prediction
  - Composite Boltz-pose-quality is NOT computed here (needs cofolding).
    Instead we compute FiLM pIC50, since that's the closest cheap proxy for
    what DAP is optimizing.

Output: results/variant3/variant3_metrics.json + variant3_metrics.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from train_m1a_v2_dap import (  # noqa: E402
    MOL1_SMI, acryl_on_largest, qed_score, morgan_fp_np,
    load_film_predictor,
)

OSIMERTINIB_SMI = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1"


def score_pIC50_batch_fast(smiles_list, film_model, scaler, anchor_embs, anchor_pIC50,
                            device: str = "cuda", batch_size: int = 512) -> np.ndarray:
    """GPU-batched FiLMDelta pIC50 predictions.

    For each valid target, compute pIC50 = mean over anchors of (anchor_pIC50 + delta(anchor, target)).
    Vectorized: process `batch_size` targets at once → (n_anch × batch_size) FiLM calls per batch.
    """
    n = len(smiles_list)
    out = np.full(n, np.nan, dtype=np.float64)
    valid_fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        arr = morgan_fp_np(smi)
        if arr is None:
            continue
        valid_fps.append(arr)
        valid_idx.append(i)
    if not valid_fps:
        return out
    fps_arr = np.asarray(valid_fps, dtype=np.float32)
    tgt_all = torch.FloatTensor(scaler.transform(fps_arr))
    n_anch = len(anchor_pIC50)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    film_model = film_model.to(dev)
    anchor_embs_d = anchor_embs.to(dev)
    anchor_pIC50_t = torch.from_numpy(np.asarray(anchor_pIC50, dtype=np.float32)).to(dev)
    with torch.no_grad():
        n_valid = tgt_all.size(0)
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            tgt_batch = tgt_all[start:end].to(dev)  # (B, D)
            B = tgt_batch.size(0)
            # Expand anchors × targets into flat batch
            # anchors: (n_anch, D)  → (n_anch, B, D) → (n_anch*B, D)
            a_exp = anchor_embs_d.unsqueeze(1).expand(n_anch, B, -1).reshape(-1, anchor_embs_d.size(-1))
            t_exp = tgt_batch.unsqueeze(0).expand(n_anch, B, -1).reshape(-1, tgt_batch.size(-1))
            deltas = film_model(a_exp, t_exp).view(n_anch, B).cpu()
            preds = (anchor_pIC50_t.unsqueeze(1) + deltas.to(dev)).mean(dim=0).cpu().numpy()
            for j, orig_i in enumerate(valid_idx[start:end]):
                out[orig_i] = float(preds[j])
    return out


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def score_cohort(csv_path: Path, name: str, film_bundle) -> dict:
    df = pd.read_csv(csv_path)
    smis = df["SMILES"].astype(str).tolist()
    n = len(smis)
    print(f"[score] {name}: N={n}", flush=True)

    # Basic metrics
    canons = [canonicalize(s) for s in smis]
    n_valid = sum(1 for c in canons if c is not None)
    valid_canons = [c for c in canons if c is not None]
    n_unique = len(set(valid_canons)) if valid_canons else 0
    scs = [murcko(s) for s in smis]
    sc_counter = Counter([s for s in scs if s])
    top_sc_share = (sc_counter.most_common(1)[0][1] / n_valid) if (sc_counter and n_valid) else None

    # Acryl retention (largest fragment)
    n_acryl_largest = sum(1 for s in smis if acryl_on_largest(s))

    # Tanimoto to Mol1
    fp_mol1 = morgan_fp(MOL1_SMI)
    fp_osi = morgan_fp(OSIMERTINIB_SMI)
    tan_mol1, tan_osi = [], []
    qed_vals = []
    for c in canons:
        if c is None:
            continue
        fp = morgan_fp(c)
        if fp is not None:
            tan_mol1.append(float(DataStructs.TanimotoSimilarity(fp, fp_mol1)))
            tan_osi.append(float(DataStructs.TanimotoSimilarity(fp, fp_osi)))
        q = qed_score(c)
        if np.isfinite(q):
            qed_vals.append(float(q))

    # FiLMDelta pIC50 (batch valid canons)
    film_model, scaler, anchor_embs, anchor_pIC50 = film_bundle
    print(f"[score] {name}: scoring {len(valid_canons)} valid SMILES with FiLM", flush=True)
    t0 = time.time()
    pIC50_arr = score_pIC50_batch_fast(valid_canons, film_model, scaler,
                                        anchor_embs, anchor_pIC50)
    pIC50_finite = pIC50_arr[np.isfinite(pIC50_arr)]
    print(f"[score] {name}: FiLM done in {time.time()-t0:.1f}s  "
          f"n_finite={len(pIC50_finite)}", flush=True)

    return {
        "n_total": int(n),
        "n_valid": int(n_valid),
        "validity_rate": float(n_valid / n) if n else None,
        "n_unique_canonical": int(n_unique),
        "unique_canonical_frac": float(n_unique / n_valid) if n_valid else None,
        "top_scaffold_share": float(top_sc_share) if top_sc_share is not None else None,
        "top_scaffold_smi": sc_counter.most_common(1)[0][0] if sc_counter else None,
        "n_acryl_largest": int(n_acryl_largest),
        "acryl_largest_frag_pct": float(n_acryl_largest / n_valid) if n_valid else None,
        "tanimoto_to_mol1_median": float(np.median(tan_mol1)) if tan_mol1 else None,
        "tanimoto_to_mol1_mean": float(np.mean(tan_mol1)) if tan_mol1 else None,
        "tanimoto_to_osimertinib_median": float(np.median(tan_osi)) if tan_osi else None,
        "qed_mean": float(np.mean(qed_vals)) if qed_vals else None,
        "qed_median": float(np.median(qed_vals)) if qed_vals else None,
        "pIC50_mean": float(np.mean(pIC50_finite)) if len(pIC50_finite) else None,
        "pIC50_median": float(np.median(pIC50_finite)) if len(pIC50_finite) else None,
        "pIC50_p90": float(np.percentile(pIC50_finite, 90)) if len(pIC50_finite) else None,
        "pIC50_max": float(np.max(pIC50_finite)) if len(pIC50_finite) else None,
        "pIC50_frac_ge_7": float((pIC50_finite >= 7.0).mean()) if len(pIC50_finite) else None,
        "pIC50_frac_ge_7_5": float((pIC50_finite >= 7.5).mean()) if len(pIC50_finite) else None,
        "pIC50_frac_ge_8": float((pIC50_finite >= 8.0).mean()) if len(pIC50_finite) else None,
        "n_pIC50_scored": int(len(pIC50_finite)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant3_csv", default=str(PROJECT_ROOT / "data/m1a_v2_variant3/cohort_A_10k.csv"))
    ap.add_argument("--v2cond_csv", default=str(PROJECT_ROOT / "data/m1a_v2_ablation_10k/cohort_A.csv"))
    ap.add_argument("--v1dap_csv", default=str(PROJECT_ROOT / "data/m1a_v2_dap/cohort_A_10k.csv"))
    ap.add_argument("--film_cache", default=str(PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_model.pt"))
    ap.add_argument("--out_json", default=str(PROJECT_ROOT / "results/variant3/variant3_metrics.json"))
    ap.add_argument("--out_md", default=str(PROJECT_ROOT / "results/variant3/variant3_metrics.md"))
    args = ap.parse_args()

    print(f"[score] loading FiLM predictor from {args.film_cache}", flush=True)
    film_bundle = load_film_predictor(Path(args.film_cache))
    print(f"[score] FiLM anchors: {len(film_bundle[3])}", flush=True)

    cohorts = {}
    for name, path_s in [("variant3", args.variant3_csv),
                          ("v2cond_baseline", args.v2cond_csv),
                          ("v1_dap", args.v1dap_csv)]:
        p = Path(path_s)
        if not p.exists():
            print(f"[score] SKIP {name}: {p} not found", flush=True)
            continue
        cohorts[name] = score_cohort(p, name, film_bundle)

    # Deltas vs v2cond
    if "v2cond_baseline" in cohorts:
        base = cohorts["v2cond_baseline"]
        for name in cohorts:
            if name == "v2cond_baseline":
                continue
            d = {}
            for k, v in cohorts[name].items():
                if isinstance(v, (int, float)) and isinstance(base.get(k), (int, float)) and base.get(k) is not None:
                    try:
                        d[f"Δ_{k}"] = float(v - base[k])
                    except Exception:
                        pass
            cohorts[name]["deltas_vs_v2cond"] = d

    out = {"cohorts": cohorts, "config": vars(args)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[score] wrote {args.out_json}", flush=True)

    # Markdown side-by-side
    lines = ["# Variant 3 (v2-cond + DAP + DPO) — Metric Panel\n"]
    metrics_order = ["n_valid", "validity_rate", "unique_canonical_frac",
                     "top_scaffold_share", "acryl_largest_frag_pct",
                     "tanimoto_to_mol1_median", "qed_mean",
                     "pIC50_mean", "pIC50_median", "pIC50_p90", "pIC50_max",
                     "pIC50_frac_ge_7", "pIC50_frac_ge_7_5", "pIC50_frac_ge_8"]
    header = "| Metric |" + "".join(f" {n} |" for n in cohorts) + "\n"
    sep = "|---|" + "---|" * len(cohorts) + "\n"
    lines += [header, sep]
    for m in metrics_order:
        row = f"| {m} |"
        for name in cohorts:
            v = cohorts[name].get(m)
            if v is None:
                row += " NA |"
            elif isinstance(v, float):
                row += f" {v:.4f} |"
            else:
                row += f" {v} |"
        row += "\n"
        lines.append(row)
    if "v2cond_baseline" in cohorts and "variant3" in cohorts:
        lines.append("\n## Deltas vs v2-cond baseline (variant3 - v2cond)\n\n")
        d = cohorts["variant3"].get("deltas_vs_v2cond", {})
        for m in metrics_order:
            key = f"Δ_{m}"
            if key in d:
                lines.append(f"- **{m}**: Δ = {d[key]:+.4f}\n")
    Path(args.out_md).write_text("".join(lines))
    print(f"[score] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
