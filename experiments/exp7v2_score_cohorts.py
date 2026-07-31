"""EXP7-v2 cohort scorer: compute the 5 paper metrics per cohort.

Metrics per cohort (sampled.csv):
  1. max_tc_to_drug           — Morgan FP Tc-to-drug, max over cohort
  2. warhead_retention_rate   — fraction matching pair's warhead SMARTS
  3. composite_reward_hit_rate — fraction with composite score >= 0.5
  4. pareto_fraction          — fraction with warhead AND QED>=0.5 AND pred_delta>0
  5. predicted_delta_pic50_p90 — 90th percentile of FiLMDelta-predicted Δ pIC50

Plus extras used by the HTML report:
  - n_valid, valid_frac, qed_median, mean_pred_pic50, max_pred_pic50
  - top10 mols by composite score (smi + score components)

Output: data/exp7_v2_benchmark/exp7_v2_all_cohorts_scored.csv (+ top10 JSON per cohort)
"""
from __future__ import annotations
import gc, inspect, json, math, os, sys, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(4)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
RDLogger.DisableLog("rdApp.*")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

V2 = PROJ / "data" / "exp7_v2_benchmark"
PHASE1 = V2 / "_phase1"
RL_BASE = V2 / "_rl"
OUT_CSV = V2 / "exp7_v2_all_cohorts_scored.csv"
TOP10_JSON = V2 / "exp7_v2_cohort_top10.json"

METHODS = [
    ("covFT_RL", "A"),
    ("covFT_baseline", "baseline_prior_anchor"),
    ("mol2mol_baseline", "mol2mol_baseline"),
    ("mol2mol_RL", "mol2mol_RL"),
]

# Composite reward constants (per spec 2026-06-27 post-process task):
#   score = sigm(pred_pIC50; low=6.0, high=8.0, k=1.0)^0.5 * warhead_indicator^0.4 * QED^0.1
SIGMOID_LOW, SIGMOID_HIGH, SIGMOID_K = 6.0, 8.0, 1.0
W_FILM, W_WAR, W_QED = 0.50, 0.40, 0.10
GM_EPS = 0.05

# Warhead SMARTS per the spec (overrides per-target warhead_smarts.json so all
# 49 pairs use the same patterns). Falls back to "noncov"/"reversible" => no
# constraint, NaN retention metric.
SPEC_WARHEAD_SMARTS = {
    "acrylamide": "[CH2;X3]=[CH;X3]C(=O)",
    "chloroacetamide": "ClCC(=O)N",
    "vinyl_sulfonamide": "[CH2;X3]=[CH;X3]S(=O)(=O)",
    "epoxide": "C1CO1",
}
# 5 pairs that failed the prior P-token vocab issue (CDK7 SY-5609 series, no cohorts)
SKIP_PAIRS = {"v2_pair_015", "v2_pair_016", "v2_pair_017", "v2_pair_018", "v2_pair_019"}


def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _fp_arr(smi):
    fp = _fp(smi)
    if fp is None:
        return None
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _double_sigmoid(x, low=SIGMOID_LOW, high=SIGMOID_HIGH, k=SIGMOID_K):
    if x is None or not np.isfinite(x):
        return 0.0
    mid = 0.5 * (low + high)
    span = (high - low) if high > low else 1.0
    z = (x - mid) / (span / 4.0)
    return float(1.0 / (1.0 + math.exp(-z * 4.0 * k)))


def _geom_mean(values, weights):
    w_sum = sum(weights)
    log_acc = 0.0
    for v, w in zip(values, weights):
        v_clamped = max(float(v), GM_EPS)
        log_acc += (w / w_sum) * math.log(v_clamped)
    return math.exp(log_acc)


def _largest_frag(smi):
    if not smi or "." not in smi:
        return smi
    parts = smi.split(".")
    parts.sort(key=len, reverse=True)
    return parts[0]


def load_film(target_dir: Path) -> Optional[FiLMDeltaMLP]:
    ckpt_path = target_dir / "filmdelta.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = inspect.signature(FiLMDeltaMLP.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    hp = {k: v for k, v in ckpt["hyperparameters"].items() if k in allowed}
    m = FiLMDeltaMLP(**hp)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def load_anchor_arrays(target_dir: Path, anchor_smi: str, anchor_pic: float):
    """Use the anchor pool for FiLM ensemble averaging.

    Anchor pool CSV provides smiles + pIC50 — used as the "B" molecules in FiLM(A,B).
    For predicting (anchor -> candidate) we treat the candidate as B and anchor pool as A.
    But for matching the reward used in RL training, we use the same convention
    as the REST server: A = anchor_pool, B = candidate, delta = pIC50(B) - pIC50(A).
    Predicted candidate pIC50 = pIC50(A) + delta, averaged over the pool.
    """
    pool_csv = target_dir / "anchor_pool_strategy_b.csv"
    if pool_csv.exists():
        df = pd.read_csv(pool_csv)
        smis = df["smiles"].tolist()
        pics = df["pIC50"].astype(float).tolist()
    else:
        smis = [anchor_smi]
        pics = [anchor_pic if np.isfinite(anchor_pic) else 6.5]
    fps, ps = [], []
    for s, p in zip(smis, pics):
        arr = _fp_arr(s)
        if arr is None:
            continue
        fps.append(arr)
        ps.append(float(p) if np.isfinite(p) else 6.5)
    return np.stack(fps).astype(np.float32), np.array(ps, dtype=np.float32)


def score_pic50_batch(model, anchor_fps, anchor_pics, query_smis, qbatch=64):
    """Predict candidate pIC50 = mean over anchor pool of (anchor_pIC50 + delta(anchor, candidate))."""
    out = np.full(len(query_smis), np.nan, dtype=np.float32)
    valid_idx, valid_fps = [], []
    for i, smi in enumerate(query_smis):
        arr = _fp_arr(smi)
        if arr is None:
            continue
        valid_fps.append(arr)
        valid_idx.append(i)
    if not valid_fps:
        return out
    fps_arr = np.stack(valid_fps)
    n_a = anchor_fps.shape[0]
    a_t = torch.from_numpy(anchor_fps).float()
    p_t = torch.from_numpy(anchor_pics).float()
    with torch.no_grad():
        for st in range(0, len(valid_idx), qbatch):
            en = min(st + qbatch, len(valid_idx))
            Q = en - st
            q_fps = torch.from_numpy(fps_arr[st:en]).float()
            a_rep = a_t.unsqueeze(0).expand(Q, n_a, -1).reshape(Q * n_a, -1)
            q_rep = q_fps.unsqueeze(1).expand(Q, n_a, -1).reshape(Q * n_a, -1)
            deltas = model(a_rep, q_rep).reshape(Q, n_a)
            abs_per = (p_t.unsqueeze(0) + deltas).mean(dim=1).numpy()
            for k in range(Q):
                out[valid_idx[st + k]] = float(abs_per[k])
    return out


def smarts_matches(smis, smarts):
    """Returns boolean array of matches per SMILES."""
    pat = Chem.MolFromSmarts(smarts) if smarts else None
    if pat is None:
        return np.zeros(len(smis), dtype=bool)
    out = np.zeros(len(smis), dtype=bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            out[i] = m.HasSubstructMatch(pat)
        except Exception:
            pass
    return out


def resolve_warhead(wh_class: Optional[str]) -> tuple[str, bool]:
    """Returns (smarts, metric_used). If wh_class is missing -> default acrylamide
    AND set metric_used=True (per spec). For noncov/reversible -> always-match
    wildcard, metric_used=False (retention metric is NaN). For known covalent
    classes -> use SPEC_WARHEAD_SMARTS pattern, metric_used=True.
    """
    if wh_class is None or wh_class == "":
        return SPEC_WARHEAD_SMARTS["acrylamide"], True
    if wh_class in SPEC_WARHEAD_SMARTS:
        return SPEC_WARHEAD_SMARTS[wh_class], True
    if wh_class in ("noncov", "reversible"):
        return "[*]", False
    # Unknown covalent class - default to acrylamide but still flag covalent
    return SPEC_WARHEAD_SMARTS["acrylamide"], True


def score_cohort(cell_dir: Path, anchor_smi: str, anchor_pic: float, drug_smi: str,
                 film_model, anchor_fps, anchor_pics, wh_strict_pat: str,
                 wh_class: str) -> Optional[dict]:
    cohort_path = cell_dir / "sampled.csv"
    if not cohort_path.exists() or cohort_path.stat().st_size < 500:
        return None
    df = pd.read_csv(cohort_path)
    smi_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[0])
    smis_raw = df[smi_col].dropna().astype(str).tolist()
    # Validate
    valid_smis, valid_fps = [], []
    for s in smis_raw:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = _fp(s)
        if fp is None:
            continue
        valid_smis.append(s)
        valid_fps.append(fp)
    if not valid_smis:
        return None

    # 1) max_tc_to_drug
    drug_fp = _fp(_largest_frag(drug_smi))
    tc_drug = np.array(DataStructs.BulkTanimotoSimilarity(drug_fp, valid_fps), dtype=float)
    max_tc_drug = float(tc_drug.max())

    # 2) warhead_retention_rate (uses spec-resolved SMARTS)
    spec_smarts, warhead_metric_used = resolve_warhead(wh_class)
    wh_matches = smarts_matches(valid_smis, spec_smarts)
    if warhead_metric_used:
        warhead_retention_rate = float(wh_matches.mean())
    else:
        warhead_retention_rate = float("nan")

    # FiLM-predicted candidate pIC50 (per query)
    pred_pic = score_pic50_batch(film_model, anchor_fps, anchor_pics, valid_smis)

    # Predicted delta: candidate_pIC50 - anchor_pIC50 (anchor_pic from pair)
    anchor_pic_base = float(anchor_pic) if np.isfinite(anchor_pic) else 6.5
    pred_delta = pred_pic - anchor_pic_base

    # QED
    qeds = np.zeros(len(valid_smis), dtype=float)
    for i, s in enumerate(valid_smis):
        try:
            qeds[i] = QED.qed(Chem.MolFromSmiles(s))
        except Exception:
            qeds[i] = float("nan")

    # 3) composite reward (per query, mirroring REST server)
    composite = np.zeros(len(valid_smis), dtype=float)
    for i in range(len(valid_smis)):
        s_pic = _double_sigmoid(pred_pic[i])
        s_war = 1.0 if wh_matches[i] else 0.5
        s_qed = float(qeds[i]) if np.isfinite(qeds[i]) else 0.0
        composite[i] = _geom_mean([s_pic, s_war, s_qed], [W_FILM, W_WAR, W_QED])

    composite_hit_rate = float((composite >= 0.5).mean())

    # 4) pareto_fraction: warhead AND qed>=0.5 AND pred_delta>0
    if warhead_metric_used:
        pareto_mask = wh_matches & (qeds >= 0.5) & (pred_delta > 0)
    else:
        # No warhead constraint for noncov/reversible
        pareto_mask = (qeds >= 0.5) & (pred_delta > 0)
    pareto_fraction = float(pareto_mask.mean())

    # 5) predicted_delta_pic50_p90
    valid_pred_delta = pred_delta[np.isfinite(pred_delta)]
    pred_delta_p90 = float(np.percentile(valid_pred_delta, 90)) if len(valid_pred_delta) else float("nan")

    # Extras
    qed_median = float(np.nanmedian(qeds)) if np.any(np.isfinite(qeds)) else float("nan")
    mean_pred_pic50 = float(np.nanmean(pred_pic)) if np.any(np.isfinite(pred_pic)) else float("nan")
    max_pred_pic50 = float(np.nanmax(pred_pic)) if np.any(np.isfinite(pred_pic)) else float("nan")

    # Top-10 by composite
    top_idx = np.argsort(-composite)[:10]
    top10 = []
    for idx in top_idx:
        top10.append({
            "smi": valid_smis[int(idx)],
            "composite": float(composite[idx]),
            "pred_pic50": float(pred_pic[idx]),
            "pred_delta_pic50": float(pred_delta[idx]),
            "qed": float(qeds[idx]),
            "tc_to_drug": float(tc_drug[idx]),
            "warhead_match": bool(wh_matches[idx]),
        })

    # Best Tc + best pIC50 (for HTML)
    best_tc_idx = int(np.argmax(tc_drug))
    valid_pic_idx = np.where(np.isfinite(pred_pic))[0]
    best_pic_idx = int(valid_pic_idx[np.argmax(pred_pic[valid_pic_idx])]) if len(valid_pic_idx) else 0

    return {
        "n_total": len(smis_raw),
        "n_valid": len(valid_smis),
        "valid_frac": len(valid_smis) / max(1, len(smis_raw)),
        # Paper metrics
        "max_tc_to_drug": max_tc_drug,
        "warhead_retention_rate": warhead_retention_rate,
        "composite_reward_hit_rate": composite_hit_rate,
        "pareto_fraction": pareto_fraction,
        "predicted_delta_pic50_p90": pred_delta_p90,
        # Extras
        "qed_median": qed_median,
        "mean_pred_pic50": mean_pred_pic50,
        "max_pred_pic50": max_pred_pic50,
        "n_tc05": int((tc_drug >= 0.5).sum()),
        "n_tc06": int((tc_drug >= 0.6).sum()),
        "n_tc07": int((tc_drug >= 0.7).sum()),
        "best_tc_smi": valid_smis[best_tc_idx],
        "best_tc_value": float(tc_drug[best_tc_idx]),
        "best_pic_smi": valid_smis[best_pic_idx],
        "best_pic_value": float(pred_pic[best_pic_idx]),
        "warhead_metric_used": warhead_metric_used,
        "top10": top10,
    }


def main():
    with open(V2 / "clean_pairs.json") as f:
        bench = json.load(f)
    pairs = bench["pairs"]

    # Load FiLM models + warhead per target
    targets = sorted({p["target"] for p in pairs})
    film, anchor_arrays, wh = {}, {}, {}
    for tk in targets:
        td = PHASE1 / tk
        film[tk] = load_film(td)
        if film[tk] is None:
            print(f"[score] WARN: no FiLM for {tk} — skipping its pairs")
            continue
        wh_spec = json.load(open(td / "warhead_smarts.json"))
        wh[tk] = wh_spec
        print(f"[load] {tk}: film + warhead OK ({wh_spec.get('warhead_class','?')})")

    rows = []
    top10_all = {}
    n_skipped = 0
    n_pairs_processed = 0
    for pair in pairs:
        pid = pair["pair_id"]
        if pid in SKIP_PAIRS:
            print(f"[skip] {pid} (5-pair P-token failure cohort)")
            n_skipped += 1
            continue
        tk = pair["target"]
        if tk not in film or film[tk] is None:
            continue
        anchor_smi = pair["anchor"]["smiles"]
        anchor_pic_raw = pair["anchor"].get("pic50")
        if anchor_pic_raw is None or (isinstance(anchor_pic_raw, float) and not np.isfinite(anchor_pic_raw)):
            anchor_pic = 7.0
            anchor_pic_fallback = True
        else:
            anchor_pic = float(anchor_pic_raw)
            anchor_pic_fallback = False
        drug_smi = pair["drug"]["smiles"]

        anchor_fps, anchor_pics = load_anchor_arrays(PHASE1 / tk, anchor_smi, anchor_pic)
        # Use clean_pairs warhead_class (overrides per-target JSON for the spec patterns)
        wh_class_pair = pair.get("warhead_class")
        wh_class_target = wh[tk].get("warhead_class") if tk in wh else None
        wh_class = wh_class_pair or wh_class_target

        print(f"\n[pair {n_pairs_processed+1}] {pid} ({tk}, warhead_class={wh_class}, anchor_pic={anchor_pic:.2f}"
              f"{' [FALLBACK]' if anchor_pic_fallback else ''}, n_anchors={anchor_fps.shape[0]})")

        for nice, raw in METHODS:
            cell_dir = RL_BASE / f"{pid}_{raw}"
            res = score_cohort(cell_dir, anchor_smi, anchor_pic, drug_smi,
                               film[tk], anchor_fps, anchor_pics,
                               "[*]",  # legacy arg, unused now (resolve_warhead does it)
                               wh_class)
            if res is None:
                print(f"  {nice:<18} -- NO COHORT or empty")
                rows.append({
                    "pair_id": pid, "target": tk, "method": nice,
                    "warhead_class": wh_class, "anchor_pic_fallback": anchor_pic_fallback,
                    "n_total": 0, "n_valid": 0,
                    "max_tc_to_drug": float("nan"),
                    "warhead_retention_rate": float("nan"),
                    "composite_reward_hit_rate": float("nan"),
                    "pareto_fraction": float("nan"),
                    "predicted_delta_pic50_p90": float("nan"),
                })
                continue
            print(f"  {nice:<18} n={res['n_valid']:5d}  max_tc={res['max_tc_to_drug']:.3f}  "
                  f"war_ret={res['warhead_retention_rate']:.3f}  comp_hit={res['composite_reward_hit_rate']:.3f}  "
                  f"pareto={res['pareto_fraction']:.3f}  dp90={res['predicted_delta_pic50_p90']:.2f}")
            row = {
                "pair_id": pid, "target": tk, "method": nice,
                "warhead_class": wh_class,
                "hinge_class": pair.get("hinge_class"),
                "tier": pair.get("tier"),
                "tc_anchor_drug": pair["tc_anchor_drug"],
                "delta_pic50_gt": pair["delta_pic50"],
                "anchor_pic": anchor_pic,
                "anchor_pic_fallback": anchor_pic_fallback,
                "anchor_smi": anchor_smi,
                "drug_smi": drug_smi,
                "anchor_name": pair["anchor"].get("chembl_id"),
                "drug_name": pair["drug"]["name"],
                **{k: v for k, v in res.items() if k != "top10"},
            }
            rows.append(row)
            top10_all[f"{pid}_{nice}"] = res["top10"]

        # Incremental write after each pair
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        TOP10_JSON.write_text(json.dumps(top10_all, indent=2))
        n_pairs_processed += 1
        gc.collect()

    print(f"\n[done] pairs_processed={n_pairs_processed}  cohorts_total={len(rows)}  pairs_skipped={n_skipped}")
    print(f"[write] {OUT_CSV} ({len(rows)} rows)")
    print(f"[write] {TOP10_JSON} ({len(top10_all)} cohorts)")


if __name__ == "__main__":
    main()
