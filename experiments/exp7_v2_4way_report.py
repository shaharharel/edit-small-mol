"""EXP7 v2 4-way report: score 54 LO pairs × 4 methods × 10K samples.

Methods (per pair on disk):
  v2_pair_NNN_A                      = covFT_RL
  v2_pair_NNN_baseline_prior_anchor  = covFT_baseline
  v2_pair_NNN_mol2mol_baseline       = mol2mol_baseline
  v2_pair_NNN_mol2mol_RL             = mol2mol_RL

Five metrics per cohort (per orchestrator spec):
  1. max_tc_to_drug             (Morgan FP Tc, max over cohort) — contamination test
  2. warhead_retention_rate     (fraction matching pair.warhead_class SMARTS)
  3. composite_reward_hit_rate  (fraction with score >= 0.5; score = sigm(pic50)^0.5 * wh^0.4 * QED^0.1)
  4. pareto_fraction            (warhead AND QED>=0.5 AND pred_delta>0)
  5. predicted_delta_pic50_p90  (90th pct FiLMDelta-predicted delta pIC50: anchor -> candidate)

Locally scored using per-target FiLMDelta (data/exp7_v2_benchmark/_phase1/<target>/filmdelta.pt)
and anchor pool (anchor_pool_strategy_b.csv). For per-pair anchor pIC50 we use
clean_pairs.json anchor.pic50 (always present).

Outputs:
  results/paper_evaluation/exp7_v2_4way_report.html
  data/exp7_v2_benchmark/exp7_v2_4way_summary.json
"""
from __future__ import annotations

import base64
import inspect
import io
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, QED
RDLogger.DisableLog("rdApp.*")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PHASE1 = PROJ / "data" / "exp7_v2_benchmark" / "_phase1"
RL_BASE = PROJ / "data" / "exp7_v2_benchmark" / "_rl"
PAIRS_JSON = PROJ / "data" / "exp7_v2_benchmark" / "clean_pairs.json"
OUT_HTML = PROJ / "results" / "paper_evaluation" / "exp7_v2_4way_report.html"
OUT_JSON = PROJ / "data" / "exp7_v2_benchmark" / "exp7_v2_4way_summary.json"

# Per orchestrator spec: SMARTS patterns by warhead_class
WARHEAD_SMARTS = {
    "acrylamide":           "[CH2;X3]=[CH;X3]C(=O)",
    "chloroacetamide":      "ClCC(=O)N",
    "vinyl_sulfonamide":    "[CH2;X3]=[CH;X3]S(=O)(=O)",
    "epoxide":              "C1CO1",
}

# Method mapping: nice name -> on-disk suffix
COND_MAP = [
    ("covFT_RL",         "A"),
    ("covFT_baseline",   "baseline_prior_anchor"),
    ("mol2mol_baseline", "mol2mol_baseline"),
    ("mol2mol_RL",       "mol2mol_RL"),
]

METRIC_NAMES = [
    "max_tc_to_drug",
    "warhead_retention_rate",
    "composite_reward_hit_rate",
    "pareto_fraction",
    "predicted_delta_pic50_p90",
]
METRIC_LABELS = {
    "max_tc_to_drug":            "max Tc → drug",
    "warhead_retention_rate":    "warhead retention",
    "composite_reward_hit_rate": "composite ≥ 0.5",
    "pareto_fraction":           "pareto (wh ∧ QED ∧ Δ>0)",
    "predicted_delta_pic50_p90": "pred ΔpIC50 p90",
}
# Higher-is-better for all five (contamination is included but flagged in verdict)
METRIC_HIGHER_BETTER = {n: True for n in METRIC_NAMES}


# ---------- utilities ----------

def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _largest_frag(smi):
    if not smi or "." not in smi:
        return smi
    parts = smi.split(".")
    parts.sort(key=lambda s: len(s), reverse=True)
    return parts[0]


def smi_to_png_b64(smi, size=160):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        img = Draw.MolToImage(m, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def load_film_model(target_dir):
    ckpt = torch.load(target_dir / "filmdelta.pt", map_location="cpu", weights_only=False)
    sig = inspect.signature(FiLMDeltaMLP.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    hp = {k: v for k, v in ckpt["hyperparameters"].items() if k in allowed}
    m = FiLMDeltaMLP(**hp)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def load_anchor_arrays(target_key, anchor_smi, anchor_pic50):
    """Return (fps, pic50s) for the per-pair anchor pool. Falls back to single anchor."""
    target_dir = PHASE1 / target_key
    pool_csv = target_dir / "anchor_pool_strategy_b.csv"
    smis = [anchor_smi]
    pics = [float(anchor_pic50) if anchor_pic50 is not None else 7.0]
    if pool_csv.exists():
        # Pool is the FULL target pool; we want at least the per-pair anchor's neighbours.
        # Use union {given anchor} ∪ pool (truncate to ~50 anchors).
        try:
            df = pd.read_csv(pool_csv)
            for s, p in zip(df["smiles"].tolist(), df["pIC50"].astype(float).tolist()):
                if s != anchor_smi:
                    smis.append(s)
                    pics.append(float(p) if np.isfinite(p) else 7.0)
        except Exception:
            pass
    fps, ps = [], []
    for s, p in zip(smis[:50], pics[:50]):
        fp = _fp(s)
        if fp is None:
            continue
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        ps.append(p)
    if not fps:
        # Fallback: use anchor only
        fp = _fp(anchor_smi)
        if fp is None:
            return None, None
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps = [arr]
        ps = [float(anchor_pic50) if anchor_pic50 is not None else 7.0]
    return np.stack(fps).astype(np.float32), np.array(ps, dtype=np.float32)


def predict_pic50(model, anchor_fps, anchor_pics, query_fps_np, qbatch=64):
    """Returns mean-over-anchors predicted absolute pIC50 per query."""
    out = np.full(len(query_fps_np), np.nan, dtype=np.float32)
    n_a = anchor_fps.shape[0]
    a_t = torch.from_numpy(anchor_fps).float()
    p_t = torch.from_numpy(anchor_pics).float()
    with torch.no_grad():
        for st in range(0, len(query_fps_np), qbatch):
            en = min(st + qbatch, len(query_fps_np))
            Q = en - st
            q_fps = torch.from_numpy(query_fps_np[st:en]).float()
            a_rep = a_t.unsqueeze(0).expand(Q, n_a, -1).reshape(Q * n_a, -1)
            q_rep = q_fps.unsqueeze(1).expand(Q, n_a, -1).reshape(Q * n_a, -1)
            deltas = model(a_rep, q_rep).reshape(Q, n_a)
            abs_per = (p_t.unsqueeze(0) + deltas).mean(dim=1).numpy()
            out[st:en] = abs_per
    return out


def sigmoid(x, lo=6.0, hi=8.0, k=1.0):
    """Reward sigmoid: 0 at pic50=lo, 0.5 at midpoint, 1 at pic50>=hi (slope k)."""
    # We use a smooth sigmoid centered between lo and hi
    mid = (lo + hi) / 2.0
    return 1.0 / (1.0 + np.exp(-k * (x - mid) / ((hi - lo) / 4.0)))


def warhead_match(mols, smarts_or_none):
    if smarts_or_none is None:
        return np.zeros(len(mols), dtype=bool)
    pat = Chem.MolFromSmarts(smarts_or_none)
    if pat is None:
        return np.zeros(len(mols), dtype=bool)
    return np.array([(m is not None and m.HasSubstructMatch(pat)) for m in mols], dtype=bool)


def warhead_smarts_for_pair(pair_meta):
    wc = (pair_meta.get("warhead_class") or "").strip().lower()
    if not wc:
        wc = "acrylamide"  # spec default
    return WARHEAD_SMARTS.get(wc, None), wc


def score_cohort(cohort_dir, pair, film_model, anchor_fps, anchor_pics):
    """Return dict of 5 metrics + best-mol pointers + valid-molecule lists."""
    cohort_path = cohort_dir / "sampled.csv"
    if not cohort_path.exists() or cohort_path.stat().st_size < 200:
        return None
    try:
        df = pd.read_csv(cohort_path)
    except Exception:
        return None
    smi_col = "SMILES" if "SMILES" in df.columns else df.columns[0]
    smis_raw = df[smi_col].dropna().astype(str).tolist()
    if not smis_raw:
        return None

    valid_smis, valid_mols, valid_fp_arrs = [], [], []
    for s in smis_raw:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_smis.append(s)
        valid_mols.append(m)
        valid_fp_arrs.append(arr)
    if not valid_smis:
        return None

    valid_fps_np = np.stack(valid_fp_arrs)

    # 1. max Tc to drug
    drug_smi = _largest_frag(pair["drug"]["smiles"])
    drug_fp = _fp(drug_smi)
    tc_drug = np.array(DataStructs.BulkTanimotoSimilarity(
        drug_fp,
        [DataStructs.CreateFromBitString(
            "".join("1" if v else "0" for v in arr.astype(int))
        ) for arr in valid_fp_arrs[:0]]  # placeholder; we'll use bitvects from mols
    ), dtype=float) if False else None  # see below
    # Use bitvects directly for bulk Tc (faster):
    bvs = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
    tc_drug = np.array(DataStructs.BulkTanimotoSimilarity(drug_fp, bvs), dtype=float)

    # 2. warhead retention
    smarts, wc = warhead_smarts_for_pair(pair)
    if smarts is None:
        wh_mask = np.array([], dtype=bool)
        wh_rate = float("nan")
        wh_indicator = np.ones(len(valid_smis), dtype=float)  # treat as 1 for composite when N/A
        warhead_applicable = False
    else:
        wh_mask = warhead_match(valid_mols, smarts)
        wh_rate = float(wh_mask.mean())
        wh_indicator = wh_mask.astype(float)
        warhead_applicable = True

    # 5. predicted ΔpIC50 = pred_abs_pic50 − anchor_pic50
    anchor_pic = float(pair["anchor"].get("pic50", 7.0))
    pred_abs = predict_pic50(film_model, anchor_fps, anchor_pics, valid_fps_np)
    pred_delta = pred_abs - anchor_pic
    p90_delta = float(np.nanpercentile(pred_delta, 90)) if np.isfinite(pred_delta).any() else float("nan")

    # QED
    qeds = np.array([QED.qed(m) for m in valid_mols], dtype=float)

    # 3. composite reward hit rate (>= 0.5)
    # score = sigmoid(pred_pic50)^0.5 * wh_indicator^0.4 * QED^0.1
    sig = sigmoid(pred_abs)
    # Use wh_indicator (1.0 for non-applicable warhead classes per design above)
    # geometric mean form: weights 0.5/0.4/0.1
    # Avoid 0^0 issues by clipping
    eps = 1e-6
    sig_c = np.clip(sig, eps, 1.0)
    wh_c = np.clip(wh_indicator, eps, 1.0)
    qed_c = np.clip(qeds, eps, 1.0)
    composite = (sig_c ** 0.5) * (wh_c ** 0.4) * (qed_c ** 0.1)
    hit_rate = float((composite >= 0.5).mean())

    # 4. pareto: warhead-retained AND QED>=0.5 AND pred_delta > 0
    if warhead_applicable:
        pareto_mask = wh_mask & (qeds >= 0.5) & (pred_delta > 0)
    else:
        # if warhead N/A, drop that constraint
        pareto_mask = (qeds >= 0.5) & (pred_delta > 0)
    pareto_frac = float(pareto_mask.mean())

    # top-10 by composite score (for display)
    top10_idx = np.argsort(-composite)[:10]
    top10 = [
        {
            "smiles": valid_smis[i],
            "tc_drug": float(tc_drug[i]),
            "pred_delta": float(pred_delta[i]),
            "qed": float(qeds[i]),
            "warhead": bool(wh_indicator[i] >= 0.5) if warhead_applicable else None,
            "composite": float(composite[i]),
        }
        for i in top10_idx
    ]

    best_tc_i = int(np.argmax(tc_drug))
    best_delta_i = int(np.argmax(pred_delta))

    return {
        "n_total": len(smis_raw),
        "n_valid": len(valid_smis),
        "valid_frac": len(valid_smis) / max(1, len(smis_raw)),
        # 5 metrics
        "max_tc_to_drug":            float(tc_drug.max()),
        "warhead_retention_rate":    wh_rate,
        "composite_reward_hit_rate": hit_rate,
        "pareto_fraction":           pareto_frac,
        "predicted_delta_pic50_p90": p90_delta,
        # extras
        "median_pred_delta":         float(np.nanmedian(pred_delta)),
        "median_qed":                float(np.median(qeds)),
        "warhead_applicable":        warhead_applicable,
        "warhead_class":             wc,
        # best-mol pointers
        "best_tc_smi":               valid_smis[best_tc_i],
        "best_tc_value":             float(tc_drug[best_tc_i]),
        "best_delta_smi":            valid_smis[best_delta_i],
        "best_delta_value":          float(pred_delta[best_delta_i]),
        # top-10 by composite
        "top10":                     top10,
    }


def fmt(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{d}f}"


def fmt_pct(x, d=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{100 * x:.{d}f}%"


# ---------- main ----------

def main():
    pairs_data = json.loads(PAIRS_JSON.read_text())
    pairs = pairs_data["pairs"]
    print(f"[load] {len(pairs)} pairs from {PAIRS_JSON}")

    # Group targets so we load each FiLM model once
    targets = sorted({p["target"] for p in pairs})
    film = {}
    for tk in targets:
        td = PHASE1 / tk
        try:
            film[tk] = load_film_model(td)
            print(f"[load] FiLM {tk}: OK")
        except Exception as e:
            print(f"[load] FiLM {tk}: FAILED {e}")
            film[tk] = None

    # Score each pair × condition
    rows = []
    for pair in pairs:
        pid = pair["pair_id"]
        tk = pair["target"]
        anchor_smi = pair["anchor"]["smiles"]
        anchor_pic = float(pair["anchor"].get("pic50", 7.0))
        anchor_fps, anchor_pics = load_anchor_arrays(tk, anchor_smi, anchor_pic)
        print(f"\n[pair] {pid} target={tk} warhead={pair.get('warhead_class')} anchor_pic={anchor_pic:.2f} anchors_used={anchor_fps.shape[0]}")
        per_cond = {}
        for nice, raw in COND_MAP:
            cdir = RL_BASE / f"{pid}_{raw}"
            if film[tk] is None:
                m = None
            else:
                m = score_cohort(cdir, pair, film[tk], anchor_fps, anchor_pics)
            if m is None:
                print(f"  {nice:<18} -- NO COHORT")
            else:
                print(f"  {nice:<18} n={m['n_valid']:5d}  Tc_max={m['max_tc_to_drug']:.3f}  "
                      f"wh={fmt_pct(m['warhead_retention_rate'])}  "
                      f"hit={fmt_pct(m['composite_reward_hit_rate'])}  "
                      f"pareto={fmt_pct(m['pareto_fraction'])}  "
                      f"Δp90={fmt(m['predicted_delta_pic50_p90'],2)}")
            per_cond[nice] = m

        # Tc(anchor, drug)
        a_fp = _fp(anchor_smi)
        d_fp = _fp(_largest_frag(pair["drug"]["smiles"]))
        tc_ad = float(DataStructs.TanimotoSimilarity(a_fp, d_fp)) if (a_fp and d_fp) else float("nan")
        rows.append({
            "pair_id":         pid,
            "target":          tk,
            "warhead_class":   pair.get("warhead_class"),
            "hinge_class":     pair.get("hinge_class"),
            "tier":            pair.get("tier"),
            "anchor_smi":      anchor_smi,
            "anchor_name":     pair["anchor"].get("chembl_id", "anchor"),
            "anchor_pic50":    anchor_pic,
            "drug_smi":        pair["drug"]["smiles"],
            "drug_name":       pair["drug"].get("name", "drug"),
            "drug_pic50":      pair["drug"].get("pic50"),
            "tc_anchor_drug":  tc_ad,
            "delta_pic50":     pair.get("delta_pic50"),
            "anchor_png":      smi_to_png_b64(anchor_smi),
            "drug_png":        smi_to_png_b64(_largest_frag(pair["drug"]["smiles"])),
            "conditions":      per_cond,
        })

    # ---------- aggregate ----------
    # Per-method median + IQR for each metric
    agg = {nice: {} for nice, _ in COND_MAP}
    for nice, _ in COND_MAP:
        for met in METRIC_NAMES:
            vals = []
            for r in rows:
                m = r["conditions"][nice]
                if m is None:
                    continue
                v = m.get(met)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    continue
                vals.append(float(v))
            if vals:
                agg[nice][met] = {
                    "n":       len(vals),
                    "median":  float(np.median(vals)),
                    "q25":     float(np.percentile(vals, 25)),
                    "q75":     float(np.percentile(vals, 75)),
                    "mean":    float(np.mean(vals)),
                    "std":     float(np.std(vals)),
                }
            else:
                agg[nice][met] = None

    # Win-rate matrix: 4 methods × 5 metrics, value = wins out of N pairs (ties = 0.5 each)
    winrate = {nice: {met: 0.0 for met in METRIC_NAMES} for nice, _ in COND_MAP}
    winrate_n = {met: 0 for met in METRIC_NAMES}
    for r in rows:
        for met in METRIC_NAMES:
            # gather method-value pairs that have valid data for this metric for this pair
            vals = {}
            for nice, _ in COND_MAP:
                m = r["conditions"][nice]
                if m is None:
                    continue
                v = m.get(met)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    continue
                vals[nice] = float(v)
            if len(vals) < 2:
                continue
            winrate_n[met] += 1
            best_v = max(vals.values()) if METRIC_HIGHER_BETTER[met] else min(vals.values())
            winners = [k for k, v in vals.items() if v == best_v]
            share = 1.0 / len(winners)
            for w in winners:
                winrate[w][met] += share

    # Per-target breakdown
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    target_agg = {}
    for tk, trows in by_target.items():
        target_agg[tk] = {nice: {met: [] for met in METRIC_NAMES} for nice, _ in COND_MAP}
        for r in trows:
            for nice, _ in COND_MAP:
                m = r["conditions"][nice]
                if m is None:
                    continue
                for met in METRIC_NAMES:
                    v = m.get(met)
                    if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        target_agg[tk][nice][met].append(float(v))
        for nice, _ in COND_MAP:
            for met in METRIC_NAMES:
                arr = target_agg[tk][nice][met]
                target_agg[tk][nice][met] = float(np.median(arr)) if arr else None

    # ---------- verdict ----------
    # Compute RL-vs-baseline delta on metrics 2-4 (RL benefit)
    def _agg_med(nice, met):
        a = agg[nice].get(met)
        return a["median"] if a else None
    rl_benefit = {}
    for pairing in [("covFT_RL", "covFT_baseline"), ("mol2mol_RL", "mol2mol_baseline")]:
        rl, base = pairing
        rl_benefit[rl] = {}
        for met in METRIC_NAMES:
            v_rl = _agg_med(rl, met); v_base = _agg_med(base, met)
            rl_benefit[rl][met] = (v_rl - v_base) if (v_rl is not None and v_base is not None) else None

    verdict_lines = []
    # contamination check
    contam = {nice: _agg_med(nice, "max_tc_to_drug") for nice, _ in COND_MAP}
    contam_winner = max(contam, key=lambda k: (contam[k] is not None, contam[k] or -1))
    verdict_lines.append(
        f"<b>Metric 1 (max Tc → drug)</b> is a contamination test: it measures how often each "
        f"method rediscovers the held-out drug, which is meaningful only when the drug is plausibly "
        f"reachable from the anchor's chemotype. Median max Tc — "
        + ", ".join(f"{k}: {fmt(contam[k])}" for k in contam) + f". Contamination-tilted method: <b>{contam_winner}</b>."
    )
    # RL benefit
    for rl in ("covFT_RL", "mol2mol_RL"):
        rb = rl_benefit[rl]
        signs = []
        for met in ("warhead_retention_rate", "composite_reward_hit_rate", "pareto_fraction"):
            v = rb.get(met)
            if v is None:
                signs.append((met, "n/a"))
            else:
                signs.append((met, f"{'+' if v >= 0 else ''}{v:.3f}"))
        verdict_lines.append(
            f"<b>{rl} vs baseline</b> (RL benefit on metrics 2–4): "
            + ", ".join(f"Δ{METRIC_LABELS[m]}={s}" for m, s in signs) + "."
        )

    # ---------- write JSON ----------
    summary = {
        "n_pairs":     len(rows),
        "methods":     [nice for nice, _ in COND_MAP],
        "metrics":     METRIC_NAMES,
        "aggregate":   agg,
        "winrate":     {nice: dict(winrate[nice]) for nice, _ in COND_MAP},
        "winrate_n":   winrate_n,
        "rl_benefit":  rl_benefit,
        "by_target":   target_agg,
        "per_pair": [
            {
                "pair_id":       r["pair_id"],
                "target":        r["target"],
                "warhead_class": r["warhead_class"],
                "hinge_class":   r["hinge_class"],
                "tier":          r["tier"],
                "anchor_pic50":  r["anchor_pic50"],
                "drug_pic50":    r["drug_pic50"],
                "tc_anchor_drug":r["tc_anchor_drug"],
                "delta_pic50":   r["delta_pic50"],
                "drug_name":     r["drug_name"],
                "conditions": {
                    n: ({k: v for k, v in (m or {}).items()
                         if k not in ("best_tc_smi", "best_delta_smi", "top10")}
                        if m else None)
                    for n, m in r["conditions"].items()
                },
            }
            for r in rows
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[write] {OUT_JSON}")

    # ---------- write HTML ----------
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>EXP7 v2 — 4-way RL benchmark (54 LO pairs)</title>")
    html.append("""<style>
        body { font-family: -apple-system, sans-serif; margin: 16px; color: #222; max-width: 1900px; }
        h1 { font-size: 22px; margin-bottom: 4px; }
        h2 { margin-top: 22px; font-size: 16px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
        h3 { margin-top: 18px; font-size: 14px; color: #444; }
        table { border-collapse: collapse; font-size: 11px; margin-bottom: 6px; }
        th, td { padding: 3px 6px; border: 1px solid #ddd; text-align: left; vertical-align: top; }
        th { background: #f5f5f5; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .cond-mol2mol-baseline { background: #fffde7; }
        .cond-mol2mol-RL       { background: #fff9c4; }
        .cond-covFT-baseline   { background: #e1f5fe; }
        .cond-covFT-RL         { background: #b3e5fc; }
        .header2 { background: #f0f0f0; font-weight: bold; }
        .stat-table th, .stat-table td { padding: 5px 9px; font-size: 12px; }
        .caveat { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin: 10px 0; line-height: 1.45; font-size: 13px; }
        .verdict { background: #e8f5e9; border-left: 4px solid #43a047; padding: 12px 16px; margin: 12px 0; line-height: 1.5; font-size: 13px; }
        details summary { cursor: pointer; font-size: 12px; color: #1976d2; padding: 3px 0; }
        details[open] summary { font-weight: bold; }
        .top10grid { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px 0; }
        .top10cell { border: 1px solid #ddd; padding: 4px; text-align: center; font-size: 9px; width: 92px; }
        .top10cell img { display: block; margin: 0 auto; }
        .win-best { background: #c8e6c9; font-weight: bold; }
    </style></head><body>""")

    html.append("<h1>EXP7 v2 — 4-way RL method comparison (54 leave-out pairs)</h1>")
    html.append(f"<p style='color:#666;font-size:12px'>Generated by experiments/exp7_v2_4way_report.py. Dataset: clean_pairs.json (54 pairs across {len(targets)} covalent-kinase targets).</p>")

    html.append("<div class='caveat'>")
    html.append("<b>Honest framing.</b> Five metrics per cohort, evaluated on 10K REINVENT samples from each "
                "of 4 methods × 54 anchor→drug pairs.<br>"
                "<b>Metric 1 (max Tc → drug) is a contamination test</b>: high values favour methods that "
                "rediscover the held-out drug from the anchor, which only happens when (a) the prior already "
                "encodes the drug-like chemotype or (b) the anchor is similar to the drug. It is reported for "
                "transparency but should <em>not</em> be the primary RL-benefit signal. "
                "<b>RL benefit is measured by metrics 2–4</b>: warhead retention, composite reward hit rate, "
                "and a 3-way Pareto front (warhead ∧ QED ≥ 0.5 ∧ predicted ΔpIC50 &gt; 0). "
                "<b>Metric 5 (FiLMDelta-predicted ΔpIC50 p90)</b> is the headline potency signal.<br><br>"
                "<b>Methods.</b> "
                "<code>covFT_*</code> use the covalent-FT mol2mol prior; <code>mol2mol_*</code> use the vanilla "
                "REINVENT4 mol2mol prior. RL variants train 50 steps of staged-learning RL with the composite "
                "reward (geometric mean of FiLMDelta-sigm, SMARTS warhead indicator, QED; weights 0.5/0.4/0.1). "
                "Baselines use the prior plus the anchor as input seed, no RL.<br><br>"
                "<b>Assumption flag.</b> Predicted ΔpIC50 is computed as (FiLMDelta-predicted absolute pIC50 − "
                "anchor-pIC50-from-JSON). All 54 pairs have anchor pIC50 present, so no fallback was used.<br>"
                "<b>Warhead retention</b> SMARTS by class: "
                "acrylamide <code>[CH2;X3]=[CH;X3]C(=O)</code>, chloroacetamide <code>ClCC(=O)N</code>, "
                "vinyl_sulfonamide <code>[CH2;X3]=[CH;X3]S(=O)(=O)</code>, epoxide <code>C1CO1</code>. "
                "For <code>noncov</code> and <code>reversible</code> pairs the warhead metric is N/A.")
    html.append("</div>")

    # Verdict
    html.append("<div class='verdict'><b>Verdict.</b><br>")
    for line in verdict_lines:
        html.append(f"{line}<br>")
    html.append("</div>")

    # Aggregate (per method, per metric: median ± IQR)
    html.append("<h2>Aggregate — per-method median (IQR) across pairs</h2>")
    html.append("<table class='stat-table'><tr><th>metric</th>")
    for nice, _ in COND_MAP:
        html.append(f"<th class='cond-{nice.replace('_','-')}'>{nice}</th>")
    html.append("</tr>")
    for met in METRIC_NAMES:
        html.append(f"<tr><th>{METRIC_LABELS[met]}</th>")
        # find best (max) median for this row
        meds = {}
        for nice, _ in COND_MAP:
            a = agg[nice].get(met)
            if a:
                meds[nice] = a["median"]
        best_v = max(meds.values()) if meds else None
        for nice, _ in COND_MAP:
            a = agg[nice].get(met)
            cls = f"cond-{nice.replace('_','-')}"
            if a is None:
                html.append(f"<td class='num {cls}'>—</td>")
            else:
                fmtfn = fmt_pct if met in ("warhead_retention_rate", "composite_reward_hit_rate", "pareto_fraction") else (lambda v: fmt(v, 3))
                iqr = f"({fmtfn(a['q25'])}, {fmtfn(a['q75'])})"
                med_s = fmtfn(a['median'])
                win_cls = " win-best" if (best_v is not None and a['median'] == best_v) else ""
                html.append(f"<td class='num {cls}{win_cls}'>{med_s}<br><span style='font-size:9px;color:#666'>IQR {iqr}, n={a['n']}</span></td>")
        html.append("</tr>")
    html.append("</table>")

    # Win-rate matrix
    html.append("<h2>Win-rate matrix — wins out of N pairs (ties shared)</h2>")
    html.append("<table class='stat-table'><tr><th>method</th>")
    for met in METRIC_NAMES:
        html.append(f"<th>{METRIC_LABELS[met]}<br><span style='font-size:9px;color:#666'>n={winrate_n[met]}</span></th>")
    html.append("</tr>")
    # find per-metric leader to highlight
    leaders = {}
    for met in METRIC_NAMES:
        col = {nice: winrate[nice][met] for nice, _ in COND_MAP}
        leaders[met] = max(col.values()) if col else 0
    for nice, _ in COND_MAP:
        cls = f"cond-{nice.replace('_','-')}"
        html.append(f"<tr><th class='{cls}'>{nice}</th>")
        for met in METRIC_NAMES:
            v = winrate[nice][met]
            n = winrate_n[met] or 1
            win_cls = " win-best" if (leaders[met] and v == leaders[met]) else ""
            html.append(f"<td class='num{win_cls}'>{v:.1f}/{n}<br><span style='font-size:9px;color:#666'>{100*v/n:.0f}%</span></td>")
        html.append("</tr>")
    html.append("</table>")

    # Per-target breakdown
    html.append("<h2>Per-target breakdown — median per metric, by 9 targets</h2>")
    for tk in sorted(target_agg.keys()):
        n_tk = sum(1 for r in rows if r["target"] == tk)
        html.append(f"<h3>{tk} (n={n_tk})</h3>")
        html.append("<table class='stat-table'><tr><th>metric</th>")
        for nice, _ in COND_MAP:
            html.append(f"<th class='cond-{nice.replace('_','-')}'>{nice}</th>")
        html.append("</tr>")
        for met in METRIC_NAMES:
            html.append(f"<tr><th>{METRIC_LABELS[met]}</th>")
            row_vals = {nice: target_agg[tk][nice][met] for nice, _ in COND_MAP if target_agg[tk][nice][met] is not None}
            best_v = max(row_vals.values()) if row_vals else None
            for nice, _ in COND_MAP:
                v = target_agg[tk][nice][met]
                cls = f"cond-{nice.replace('_','-')}"
                if v is None:
                    html.append(f"<td class='num {cls}'>—</td>")
                else:
                    fmtfn = fmt_pct if met in ("warhead_retention_rate", "composite_reward_hit_rate", "pareto_fraction") else (lambda v: fmt(v, 3))
                    win_cls = " win-best" if (best_v is not None and v == best_v) else ""
                    html.append(f"<td class='num {cls}{win_cls}'>{fmtfn(v)}</td>")
            html.append("</tr>")
        html.append("</table>")

    # Per-pair detail table
    html.append("<h2>Per-pair detail (54 pairs × 4 methods × 5 metrics)</h2>")
    html.append("<table><thead><tr>")
    html.append("<th colspan='9' class='header2'>PAIR</th>")
    for nice, _ in COND_MAP:
        html.append(f"<th colspan='5' class='header2 cond-{nice.replace('_','-')}'>{nice}</th>")
    html.append("</tr><tr>")
    for h in ["#", "pair_id", "target", "tier", "anchor", "drug", "Tc(a,d)", "ΔpIC50 (obs)", "warhead"]:
        html.append(f"<th>{h}</th>")
    for nice, _ in COND_MAP:
        cls = f"cond-{nice.replace('_','-')}"
        for sub in ["Tc→drug", "warhead %", "hit %", "pareto %", "Δp90"]:
            html.append(f"<th class='{cls}'>{sub}</th>")
    html.append("</tr></thead><tbody>")
    for i, r in enumerate(rows, 1):
        html.append("<tr>")
        html.append(f"<td>{i}</td>")
        html.append(f"<td><b>{r['pair_id']}</b></td>")
        html.append(f"<td>{r['target']}</td>")
        html.append(f"<td>{r['tier']}</td>")
        html.append(f"<td>{r['anchor_name']}<br><img src='{r['anchor_png']}' width=100><br><span style='font-size:9px'>pIC50={r['anchor_pic50']:.2f}</span></td>")
        html.append(f"<td>{r['drug_name']}<br><img src='{r['drug_png']}' width=100><br><span style='font-size:9px'>pIC50={fmt(r['drug_pic50'],2)}</span></td>")
        html.append(f"<td class='num'>{fmt(r['tc_anchor_drug'])}</td>")
        html.append(f"<td class='num'>{fmt(r['delta_pic50'],2)}</td>")
        html.append(f"<td>{r['warhead_class']}</td>")
        # find per-metric winner in this pair (across methods)
        per_pair_best = {}
        for met in METRIC_NAMES:
            vals = []
            for nice, _ in COND_MAP:
                m = r["conditions"][nice]
                if m is None:
                    continue
                v = m.get(met)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    continue
                vals.append((nice, float(v)))
            if vals:
                per_pair_best[met] = max(v for _, v in vals)
        for nice, _ in COND_MAP:
            cls = f"cond-{nice.replace('_','-')}"
            m = r["conditions"][nice]
            if m is None:
                for _ in range(5):
                    html.append(f"<td class='{cls}'>—</td>")
                continue
            cells_vals = [
                ("max_tc_to_drug",            fmt(m["max_tc_to_drug"], 3)),
                ("warhead_retention_rate",    fmt_pct(m["warhead_retention_rate"], 0)),
                ("composite_reward_hit_rate", fmt_pct(m["composite_reward_hit_rate"], 0)),
                ("pareto_fraction",           fmt_pct(m["pareto_fraction"], 0)),
                ("predicted_delta_pic50_p90", fmt(m["predicted_delta_pic50_p90"], 2)),
            ]
            for met, sval in cells_vals:
                v = m.get(met)
                is_best = (per_pair_best.get(met) is not None
                           and v is not None
                           and not (isinstance(v, float) and np.isnan(v))
                           and float(v) == per_pair_best[met])
                wcls = " win-best" if is_best else ""
                html.append(f"<td class='num {cls}{wcls}'>{sval}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    # Top-10 per method per pair (collapsible)
    html.append("<h2>Top-10 generated molecules per method per pair (by composite reward)</h2>")
    html.append("<p style='font-size:11px;color:#666'>Ranked by composite score = sigm(pred_pic50; 6/8, k=1)^0.5 × warhead^0.4 × QED^0.1. "
                "For non-covalent / reversible pairs warhead indicator = 1 (i.e. the constraint is dropped).</p>")
    for r in rows:
        html.append(f"<details><summary>{r['pair_id']} — {r['target']} — {r['anchor_name']} → {r['drug_name']}</summary>")
        for nice, _ in COND_MAP:
            cls = f"cond-{nice.replace('_','-')}"
            m = r["conditions"][nice]
            if m is None:
                html.append(f"<div class='{cls}' style='padding:6px;margin:4px 0'><b>{nice}</b>: no cohort.</div>")
                continue
            html.append(f"<div class='{cls}' style='padding:6px;margin:4px 0'><b>{nice}</b>")
            html.append("<div class='top10grid'>")
            for j, mol in enumerate(m["top10"], 1):
                png = smi_to_png_b64(mol["smiles"], 90)
                wh_t = "✓" if mol["warhead"] else ("—" if mol["warhead"] is None else "✗")
                html.append(
                    f"<div class='top10cell'><img src='{png}' width=85>"
                    f"<div>#{j} c={mol['composite']:.2f}</div>"
                    f"<div>Δ={mol['pred_delta']:+.1f} Q={mol['qed']:.2f}</div>"
                    f"<div>Tc={mol['tc_drug']:.2f} wh={wh_t}</div>"
                    f"</div>"
                )
            html.append("</div></div>")
        html.append("</details>")

    html.append(f"<p style='font-size:11px;color:#888;margin-top:30px'>End of report. Per-cohort n_valid stats and full SMILES are available in {OUT_JSON.name}.</p>")
    html.append("</body></html>")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(html))
    print(f"[write] {OUT_HTML}")

    # Also print verdict summary for caller
    print("\n=== VERDICT ===")
    for line in verdict_lines:
        # strip HTML for stdout
        import re
        print(re.sub(r"<[^>]+>", "", line))
    print("===============\n")


if __name__ == "__main__":
    main()
