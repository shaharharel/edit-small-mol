"""EXP7 clean-6 4-way report: score cohorts for 6 audit-clean pairs and emit HTML.

Pairs (audit_id -> cohort name on disk):
  egfr_pair_008                              -> egfr_pair_008
  her2_pair_002 (PD-168393 -> dacomitinib)   -> "PD-168393->dacomitinib"
  her2_pair_004 (canertinib (CI-1033)        -> "canertinib (CI-1033)->dacomitinib"
  her2_pair_007 (afatinib -> dacomitinib)    -> "afatinib->dacomitinib"
  her2_pair_010 (erlotinib -> dacomitinib)   -> "erlotinib->dacomitinib"
  her2_pair_012 (gefitinib -> dacomitinib)   -> "gefitinib->dacomitinib"

For each of 4 conditions (covFT_RL = A, covFT_baseline = baseline_prior_anchor,
mol2mol_baseline, mol2mol_RL) we read cohort CSVs from
data/exp7_lo_benchmark/_rl/{cohort_name}_{condition}/sampled.csv.

Scores:
  n_valid, valid_frac, max_tc_drug, n@tc {0.5,0.6,0.7},
  pred_pic50_mean+median+max, warhead %, qed median,
  best mol by Tc, best by pred pIC50.

HTML output: results/paper_evaluation/exp7_clean6_4way_report.html
"""
from __future__ import annotations

import base64
import inspect
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PHASE1 = PROJ / "data" / "exp7_lo_benchmark" / "_phase1"
RL_BASE = PROJ / "data" / "exp7_lo_benchmark" / "_rl"
OUT_HTML = PROJ / "results" / "paper_evaluation" / "exp7_clean6_4way_report.html"
OUT_JSON = PROJ / "data" / "exp7_lo_benchmark" / "exp7_clean6_4way_summary.json"

# audit_id -> (cohort_basename, target_key, anchor_smi, drug_smi, drug_name, anchor_name)
CLEAN_6 = [
    ("egfr_pair_008", "egfr_pair_008", "egfr_t790m",
     "C=CC(=O)Nc1cccc(Oc2nc(Nc3ccc(N4CCN(C)CC4)cc3OC)ncc2Cl)c1",
     "C=CC(=O)Nc1cccc(Nc2ncc(C(F)(F)F)c(Nc3ccc(N4CCN(C(C)=O)CC4)cc3OC)n2)c1",
     "rociletinib", "anchor"),
    ("her2_pair_002", "PD-168393->dacomitinib", "her2",
     "C=CC(=O)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",  # strip ".O" salt
     "dacomitinib", "PD-168393"),
    ("her2_pair_004", "canertinib (CI-1033)->dacomitinib", "her2",
     "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "dacomitinib", "canertinib (CI-1033)"),
    ("her2_pair_007", "afatinib->dacomitinib", "her2",
     "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "dacomitinib", "afatinib"),
    ("her2_pair_010", "erlotinib->dacomitinib", "her2",
     "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "dacomitinib", "erlotinib"),
    ("her2_pair_012", "gefitinib->dacomitinib", "her2",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "dacomitinib", "gefitinib"),
]

# 4 conditions: covFT_RL, covFT_baseline, mol2mol_baseline, mol2mol_RL
COND_MAP = [
    ("covFT_RL", "A"),
    ("covFT_baseline", "baseline_prior_anchor"),
    ("mol2mol_baseline", "mol2mol_baseline"),
    ("mol2mol_RL", "mol2mol_RL"),
]


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


def score_pic50(model, anchor_fps, anchor_pics, query_smiles, qbatch=64):
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


def load_anchor_arrays(target_key, cohort_basename, anchor_smi):
    target_dir = PHASE1 / target_key
    pool_csv = target_dir / "anchors" / f"{cohort_basename}_b_pool.csv"
    if pool_csv.exists():
        df = pd.read_csv(pool_csv)
        smis = df["smiles"].tolist()
        pics = df["pIC50"].astype(float).tolist()
    else:
        smis = [anchor_smi]
        pics = [6.5]
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


def smarts_pct(smis, smarts):
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return 0.0
    hits = 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if m.HasSubstructMatch(pat):
            hits += 1
    return 100.0 * hits / max(1, len(smis))


def score_cohort(cohort_dir, drug_smi, film_model, anchor_fps, anchor_pics, warhead_specs):
    cohort_path = cohort_dir / "sampled.csv"
    if not cohort_path.exists() or cohort_path.stat().st_size < 500:
        return None
    df = pd.read_csv(cohort_path)
    smi_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[0])
    smis_raw = df[smi_col].dropna().astype(str).tolist()
    valid_smis, valid_fps = [], []
    for s in smis_raw:
        f = _fp(s)
        if f is None:
            continue
        valid_smis.append(s)
        valid_fps.append(f)
    if not valid_smis:
        return None

    drug_fp = _fp(_largest_frag(drug_smi))
    tc_drug = np.array(DataStructs.BulkTanimotoSimilarity(drug_fp, valid_fps), dtype=float)
    pred_pic = score_pic50(film_model, anchor_fps, anchor_pics, valid_smis)

    wh_strict = smarts_pct(valid_smis, warhead_specs["smarts_strict"])
    wh_generic = smarts_pct(valid_smis, warhead_specs["smarts_generic"])

    qeds = []
    for s in valid_smis[:1500]:
        try:
            qeds.append(QED.qed(Chem.MolFromSmiles(s)))
        except Exception:
            pass
    qed_med = float(np.median(qeds)) if qeds else float("nan")

    return {
        "n_total": len(smis_raw),
        "n_valid": len(valid_smis),
        "valid_frac": len(valid_smis) / max(1, len(smis_raw)),
        "max_tc_drug": float(tc_drug.max()),
        "n_tc05": int((tc_drug >= 0.5).sum()),
        "n_tc06": int((tc_drug >= 0.6).sum()),
        "n_tc07": int((tc_drug >= 0.7).sum()),
        "mean_pred_pic50": float(np.mean(pred_pic)),
        "median_pred_pic50": float(np.median(pred_pic)),
        "max_pred_pic50": float(pred_pic.max()),
        "n_predpic_ge7": int((pred_pic >= 7.0).sum()),
        "warhead_strict_pct": wh_strict,
        "warhead_generic_pct": wh_generic,
        "qed_median": qed_med,
        "best_tc_smi": valid_smis[int(np.argmax(tc_drug))],
        "best_tc_value": float(tc_drug.max()),
        "best_pic_smi": valid_smis[int(np.argmax(pred_pic))],
        "best_pic_value": float(pred_pic.max()),
    }


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{d}f}"


def main():
    # Load FiLM models + warhead specs
    targets = sorted({c[2] for c in CLEAN_6})
    film, warhead = {}, {}
    for tk in targets:
        td = PHASE1 / tk
        film[tk] = load_film_model(td)
        warhead[tk] = json.load(open(td / "warhead_smarts.json"))
        print(f"[load] {tk}: film + warhead OK")

    # Score all cohorts
    rows = []
    for audit_id, cohort_base, tk, anchor_smi, drug_smi, drug_name, anchor_name in CLEAN_6:
        anchor_fps, anchor_pics = load_anchor_arrays(tk, cohort_base, anchor_smi)
        print(f"\n[pair] {audit_id} (cohort={cohort_base!r}, target={tk}, anchors={anchor_fps.shape[0]})")
        per_cond = {}
        for nice, raw in COND_MAP:
            d = RL_BASE / f"{cohort_base}_{raw}"
            m = score_cohort(d, drug_smi, film[tk], anchor_fps, anchor_pics, warhead[tk])
            if m is None:
                print(f"  {nice:<18} -- NO COHORT")
            else:
                print(f"  {nice:<18} n={m['n_valid']:5d}, maxTc={m['max_tc_drug']:.3f}, "
                      f"n@Tc05={m['n_tc05']:4d}, mean_pIC50={m['mean_pred_pic50']:.2f}, "
                      f"warhead_generic={m['warhead_generic_pct']:.0f}%")
            per_cond[nice] = m

        # Compute Tc(anchor, drug) and delta if available
        a_fp = _fp(anchor_smi)
        d_fp = _fp(_largest_frag(drug_smi))
        tc_ad = float(DataStructs.TanimotoSimilarity(a_fp, d_fp)) if (a_fp and d_fp) else float("nan")
        rows.append({
            "audit_id": audit_id,
            "cohort_base": cohort_base,
            "target_key": tk,
            "anchor_smi": anchor_smi,
            "drug_smi": drug_smi,
            "anchor_name": anchor_name,
            "drug_name": drug_name,
            "tc_anchor_drug": tc_ad,
            "anchor_png": smi_to_png_b64(anchor_smi),
            "drug_png": smi_to_png_b64(_largest_frag(drug_smi)),
            "conditions": per_cond,
        })

    # Aggregate stats per condition
    agg = {}
    for nice, _ in COND_MAP:
        max_tcs = [r["conditions"][nice]["max_tc_drug"] for r in rows if r["conditions"][nice]]
        n05 = [r["conditions"][nice]["n_tc05"] for r in rows if r["conditions"][nice]]
        mean_pic = [r["conditions"][nice]["mean_pred_pic50"] for r in rows if r["conditions"][nice]]
        wh_gen = [r["conditions"][nice]["warhead_generic_pct"] for r in rows if r["conditions"][nice]]
        n_valid = [r["conditions"][nice]["n_valid"] for r in rows if r["conditions"][nice]]
        agg[nice] = {
            "n_pairs": len(max_tcs),
            "median_max_tc": float(np.median(max_tcs)) if max_tcs else float("nan"),
            "mean_max_tc": float(np.mean(max_tcs)) if max_tcs else float("nan"),
            "median_n_tc05": float(np.median(n05)) if n05 else float("nan"),
            "mean_pred_pic50_avg": float(np.mean(mean_pic)) if mean_pic else float("nan"),
            "mean_warhead_pct": float(np.mean(wh_gen)) if wh_gen else float("nan"),
            "mean_n_valid": float(np.mean(n_valid)) if n_valid else float("nan"),
        }

    # ----- HTML -----
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>EXP7 clean-6 4-way comparison</title>")
    html.append("""<style>
        body { font-family: -apple-system, sans-serif; margin: 20px; color: #222; max-width: 1800px; }
        h1 { font-size: 22px; }
        h2 { margin-top: 24px; font-size: 16px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
        table { border-collapse: collapse; font-size: 11px; }
        th, td { padding: 4px 6px; border: 1px solid #ddd; text-align: left; vertical-align: top; }
        th { background: #f5f5f5; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .cond-mol2mol-baseline { background: #fffde7; }
        .cond-mol2mol-RL       { background: #fff9c4; }
        .cond-covFT-baseline   { background: #e1f5fe; }
        .cond-covFT-RL         { background: #b3e5fc; }
        .header2 { background: #f0f0f0; font-weight: bold; }
        .stat-table th, .stat-table td { padding: 6px 10px; font-size: 13px; }
        .caveat { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 12px 0; line-height: 1.4; }
    </style></head><body>""")

    html.append("<h1>EXP7 clean-6 LO benchmark — 4-condition comparison</h1>")
    html.append("""<div class='caveat'>
    <b>What's in this report:</b> 6 (anchor, drug) pairs that passed the EXP7 contamination
    audit. <b>Audit criteria</b>: (a) drug NOT in CovInDB v2 (FT corpus); (b) anchor-&gt;drug
    distance &ge; 1 MMP edit (no trivial chirality flips). Categories: CHEMBL_NATIVE (5) +
    CHEMBL_NATIVE-equivalent (1).
    <br><br>
    <b>Limitations:</b> these are the only audit-clean pairs out of 50 curated. A re-curation
    effort is in flight to expand the clean set. All 5 HER2 anchors here are pre-2010 quinazolines;
    EGFR T790M anchor is rociletinib's CHEMBL precursor.
    <br><br>
    <b>4 generation conditions per pair</b> (cohort size 5500):
    <ul style='margin: 6px 0;'>
      <li><b>covFT_RL</b> — covalent fine-tuned mol2mol + composite RL reward (FiLM+SMARTS+QED), 50 steps</li>
      <li><b>covFT_baseline</b> — covalent FT prior, NO RL, anchor as seed</li>
      <li><b>mol2mol_baseline</b> — vanilla REINVENT4 mol2mol prior, NO RL, anchor as seed</li>
      <li><b>mol2mol_RL</b> — vanilla mol2mol + composite RL reward, 50 steps</li>
    </ul>
    </div>""")

    # Aggregate stats
    html.append("<h2>Aggregate stats — 6 clean pairs</h2>")
    html.append("<table class='stat-table'><tr><th>condition</th><th>n with data</th>"
                "<th>median max Tc</th><th>mean max Tc</th><th>median n@Tc≥0.5</th>"
                "<th>avg mean pred pIC50</th><th>avg warhead %</th><th>avg n valid</th></tr>")
    for nice, _ in COND_MAP:
        v = agg[nice]
        cls = f"cond-{nice.replace('_', '-')}"
        html.append(f"<tr class='{cls}'><td>{nice}</td><td class='num'>{v['n_pairs']}</td>"
                    f"<td class='num'>{fmt(v['median_max_tc'])}</td>"
                    f"<td class='num'>{fmt(v['mean_max_tc'])}</td>"
                    f"<td class='num'>{fmt(v['median_n_tc05'],1)}</td>"
                    f"<td class='num'>{fmt(v['mean_pred_pic50_avg'],2)}</td>"
                    f"<td class='num'>{fmt(v['mean_warhead_pct'],1)}</td>"
                    f"<td class='num'>{fmt(v['mean_n_valid'],0)}</td></tr>")
    html.append("</table>")

    # Per-pair detail
    html.append("<h2>Per-pair detail</h2>")
    html.append("<table><thead><tr>"
                "<th colspan='8' class='header2'>PAIR</th>")
    for nice, _ in COND_MAP:
        html.append(f"<th colspan='6' class='header2 cond-{nice.replace('_','-')}'>{nice}</th>")
    html.append("</tr><tr>")
    for h in ["#", "audit_id", "target", "anchor", "drug", "Tc(a,d)", "n_valid", ""]:
        html.append(f"<th>{h}</th>")
    for nice, _ in COND_MAP:
        cls = f"cond-{nice.replace('_','-')}"
        for sub in ["max Tc", "n@Tc≥0.5", "mean pIC50", "warhead %", "best Tc mol", "best pIC50 mol"]:
            html.append(f"<th class='{cls}'>{sub}</th>")
    html.append("</tr></thead><tbody>")
    for i, r in enumerate(rows, 1):
        html.append("<tr>")
        html.append(f"<td>{i}</td>")
        html.append(f"<td><b>{r['audit_id']}</b><br><span style='font-size:10px;color:#666'>{r['cohort_base']}</span></td>")
        html.append(f"<td>{r['target_key']}</td>")
        html.append(f"<td>{r['anchor_name']}<br><img src='{r['anchor_png']}' width=120></td>")
        html.append(f"<td>{r['drug_name']}<br><img src='{r['drug_png']}' width=120></td>")
        html.append(f"<td class='num'>{fmt(r['tc_anchor_drug'])}</td>")
        html.append("<td></td><td></td>")
        for nice, _ in COND_MAP:
            cls = f"cond-{nice.replace('_','-')}"
            m = r["conditions"][nice]
            if m is None:
                for _ in range(6):
                    html.append(f"<td class='{cls}'>—</td>")
                continue
            tc_png = smi_to_png_b64(m["best_tc_smi"], 110)
            pic_png = smi_to_png_b64(m["best_pic_smi"], 110)
            html.append(f"<td class='num {cls}'>{fmt(m['max_tc_drug'])}</td>")
            html.append(f"<td class='num {cls}'>{m['n_tc05']}</td>")
            html.append(f"<td class='num {cls}'>{fmt(m['mean_pred_pic50'],2)}</td>")
            html.append(f"<td class='num {cls}'>{fmt(m['warhead_generic_pct'],0)}%</td>")
            html.append(f"<td class='{cls}'><img src='{tc_png}' width=100 title='Tc={m['best_tc_value']:.3f}'><br>"
                        f"<span style='font-size:9px'>Tc={m['best_tc_value']:.2f}</span></td>")
            html.append(f"<td class='{cls}'><img src='{pic_png}' width=100 title='pIC50={m['best_pic_value']:.2f}'><br>"
                        f"<span style='font-size:9px'>pIC50={m['best_pic_value']:.2f}</span></td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    html.append("<p style='font-size:11px;color:#666;margin-top:20px'>"
                "Generated by experiments/exp7_clean6_report.py. Scoring uses target-specific FiLMDelta "
                "+ 100-anchor B pool (mean over anchors) for pred pIC50. warhead %% uses generic "
                "Michael-acceptor SMARTS C(=O)C=C.</p>")
    html.append("</body></html>")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(html))
    print(f"\n[write] {OUT_HTML}")

    summary = {
        "n_pairs": len(rows),
        "aggregate": agg,
        "per_pair": [{
            "audit_id": r["audit_id"], "cohort_base": r["cohort_base"],
            "target_key": r["target_key"], "tc_anchor_drug": r["tc_anchor_drug"],
            "conditions": {n: ({k: v for k, v in (m or {}).items()
                                if k not in ("best_tc_smi", "best_pic_smi")}
                               if m else None)
                           for n, m in r["conditions"].items()},
        } for r in rows],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[write] {OUT_JSON}")


if __name__ == "__main__":
    main()
