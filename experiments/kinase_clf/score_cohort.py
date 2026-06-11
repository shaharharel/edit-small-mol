"""Score F4 visible + extended pool with the trained 3-head kinase classifier.

Loads:
  models/kinase_clf/trunk.pt
  models/kinase_clf/calibrators.pkl
Featurizes (Morgan + ChemBERTa MTR) — cache-aware.

Inputs:
  data/tier4_scored/F4_boltz_full.csv   (2,221 mols)
  optional extended pool CSVs (specified via --pools)

Outputs:
  results/paper_evaluation/kinase_classifier_scores.csv
  results/paper_evaluation/kinase_classifier/score_meta.json
"""
from __future__ import annotations
import os, sys, json, time, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.kinase_clf.featurize_and_train import (
    ThreeHeadFFN, featurize_morgan, featurize_chemberta, load_cache_dict,
    EMB_CACHE_MORGAN, EMB_CACHE_MTR
)

MODEL_DIR = ROOT / "models" / "kinase_clf"
OUT_CSV   = ROOT / "results" / "paper_evaluation" / "kinase_classifier_scores.csv"
OUT_META  = ROOT / "results" / "paper_evaluation" / "kinase_classifier" / "score_meta.json"
F4        = ROOT / "data" / "tier4_scored" / "F4_boltz_full.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=[])
    args = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Collect SMILES to score
    sm_set = set()
    src_rows = []
    df_f4 = pd.read_csv(F4)
    f4_sm = df_f4["smiles"].dropna().drop_duplicates().tolist()
    sm_set.update(f4_sm)
    src_rows.append({"source": "F4_boltz_full", "n": len(f4_sm)})
    for p in args.pools:
        pp = Path(p)
        if not pp.exists():
            print(f"  skip {p} (not found)")
            continue
        d = pd.read_csv(pp)
        s_col = None
        for c in ("smiles", "SMILES", "canonical_smiles"):
            if c in d.columns: s_col = c; break
        if s_col is None:
            print(f"  skip {p} (no smiles col)")
            continue
        ms = d[s_col].dropna().drop_duplicates().tolist()
        sm_set.update(ms)
        src_rows.append({"source": pp.name, "n": len(ms)})
    smiles = sorted(sm_set)
    print(f"Scoring {len(smiles):,} unique SMILES")

    # Featurize
    morgan_lookup = load_cache_dict(EMB_CACHE_MORGAN, expected_dim=2048)
    X_morgan = featurize_morgan(smiles, morgan_lookup)
    print(f"  Morgan X: {X_morgan.shape}")
    X_cb = featurize_chemberta(smiles, batch_size=256)
    print(f"  ChemBERTa X: {X_cb.shape}")
    X = np.concatenate([X_morgan, X_cb], axis=1).astype(np.float32)

    # Load model
    ckpt = torch.load(MODEL_DIR / "trunk.pt", map_location="cpu")
    in_dim = ckpt["in_dim"]
    pic_mu = ckpt["pic_mu"]; pic_sd = ckpt["pic_sd"]
    model = ThreeHeadFFN(in_dim=in_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with open(MODEL_DIR / "calibrators.pkl", "rb") as fh:
        cal = pickle.load(fh)
    platt_kin = cal["platt_kin"]; platt_tec = cal["platt_tec"]

    # Score
    lk_all, lt_all, lp_all = [], [], []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            xb = torch.from_numpy(X[b:b+4096]).to(device)
            lk, lt, lp = model(xb)
            lk_all.append(lk.cpu().numpy()); lt_all.append(lt.cpu().numpy()); lp_all.append(lp.cpu().numpy())
    lk_all = np.concatenate(lk_all); lt_all = np.concatenate(lt_all); lp_all = np.concatenate(lp_all)

    p_kin = platt_kin.predict_proba(lk_all.reshape(-1,1))[:,1]
    if platt_tec is not None:
        p_tec = platt_tec.predict_proba(lt_all.reshape(-1,1))[:,1]
    else:
        p_tec = 1/(1+np.exp(-lt_all))
    pic50_pred = lp_all * pic_sd + pic_mu

    out = pd.DataFrame({
        "smiles": smiles,
        "P_kinase": p_kin.astype(np.float32),
        "P_Tec_family": p_tec.astype(np.float32),
        "pIC50_kinase_aux": pic50_pred.astype(np.float32),
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out):,} rows)")

    # IQR analysis on 838 (use mol1_murcko_match==True survivors when present)
    f4_scored = out.merge(df_f4[["smiles"] + [c for c in ("mol1_murcko_match","thiq_core","warhead_intact") if c in df_f4.columns]],
                          on="smiles", how="right")
    iqr_kin = float(f4_scored["P_kinase"].quantile(0.75) - f4_scored["P_kinase"].quantile(0.25))
    iqr_tec = float(f4_scored["P_Tec_family"].quantile(0.75) - f4_scored["P_Tec_family"].quantile(0.25))
    iqr_pic = float(f4_scored["pIC50_kinase_aux"].quantile(0.75) - f4_scored["pIC50_kinase_aux"].quantile(0.25))
    med_kin = float(f4_scored["P_kinase"].median())
    med_tec = float(f4_scored["P_Tec_family"].median())
    med_pic = float(f4_scored["pIC50_kinase_aux"].median())
    drop_p_kin = iqr_kin < 0.10
    print(f"\nF4 cohort score distributions:")
    print(f"  P_kinase        IQR={iqr_kin:.4f}  median={med_kin:.4f}  -> drop={'YES' if drop_p_kin else 'NO'}")
    print(f"  P_Tec_family    IQR={iqr_tec:.4f}  median={med_tec:.4f}")
    print(f"  pIC50_kinase_aux IQR={iqr_pic:.4f}  median={med_pic:.4f}")

    meta = dict(
        n_scored=int(len(out)),
        sources=src_rows,
        f4_stats=dict(
            P_kinase=dict(iqr=iqr_kin, median=med_kin, drop_recommended=drop_p_kin),
            P_Tec_family=dict(iqr=iqr_tec, median=med_tec),
            pIC50_kinase_aux=dict(iqr=iqr_pic, median=med_pic),
        ),
        wall_clock_s=time.time()-t0,
    )
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_META}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
