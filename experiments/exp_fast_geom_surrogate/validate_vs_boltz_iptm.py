"""Score F4 pool with FastGeomScorer and report Pearson/Spearman vs boltz_iptm.

Acceptance bar: Pearson r ≥ 0.30 (per the user's pre-stated cutoff).
If we beat the bar, the scorer is shippable as an RL reward.
If not, redesign before deployment.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_geom_scorer import FastGeomScorer

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
F4_CSV = PROJECT_ROOT / "data/tier4_scored/F4_boltz_full.csv"
ANCHOR = PROJECT_ROOT / "data/fast_geom_surrogate/anchor_frame.npz"
OUT_CSV = PROJECT_ROOT / "data/fast_geom_surrogate/validation_F4_scores.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only first N rows (for fast iteration).")
    ap.add_argument("--n-conf", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Pearson r acceptance threshold.")
    args = ap.parse_args()

    if not F4_CSV.exists():
        print(f"missing {F4_CSV}", file=sys.stderr); sys.exit(2)
    if not ANCHOR.exists():
        print(f"missing {ANCHOR} — run anchor_frame.py first", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(F4_CSV)
    df = df[df["boltz_iptm"].notna()].reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"[validate] scoring {len(df)} F4 molecules with n_conf={args.n_conf}")

    sc = FastGeomScorer(ANCHOR, n_conformers=args.n_conf)

    scores = np.zeros(len(df), dtype=np.float64)
    shape_s = np.zeros_like(scores); hin_s = np.zeros_like(scores); bd_s = np.zeros_like(scores)
    cl_g = np.zeros_like(scores); rmsd = np.zeros_like(scores)
    shape_md = np.zeros_like(scores); shape_cv = np.zeros_like(scores)
    reasons = [""] * len(df)
    t0 = time.time()
    for i, smi in enumerate(tqdm(df["smiles"].tolist(), desc="scoring")):
        det = sc._score_internal(str(smi))
        scores[i] = det.get("composite", 0.0)
        shape_s[i] = det.get("shape_score", 0.0)
        hin_s[i] = det.get("hinge_score", 0.0)
        bd_s[i] = det.get("bd_score", 0.0)
        cl_g[i] = det.get("clash_gate", 0.0)
        rmsd[i] = det.get("rmsd_align", np.nan)
        shape_md[i] = det.get("shape_mean_d", np.nan)
        shape_cv[i] = det.get("shape_cov", 0.0)
        reasons[i] = det.get("reason", "")
    dt = time.time() - t0
    print(f"[validate] scored in {dt:.1f}s ({dt/len(df)*1000:.1f} ms/mol, "
          f"{len(df)/dt:.1f} mol/sec)")

    df_out = df[["row_id", "smiles", "boltz_iptm"]].copy()
    for col, vals in [("composite", scores), ("shape_score", shape_s),
                       ("hinge_score", hin_s), ("bd_score", bd_s),
                       ("clash_gate", cl_g), ("rmsd_align", rmsd),
                       ("shape_mean_d", shape_md), ("shape_cov", shape_cv)]:
        df_out[col] = vals
    df_out["reason"] = reasons
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[validate] wrote {OUT_CSV}")

    # Correlations on rows where the scorer ran successfully.
    ok = df_out[df_out["reason"] == "ok"]
    print(f"[validate] {len(ok)}/{len(df_out)} scored successfully")
    if len(ok) < 10:
        print("[validate] too few rows for correlation"); sys.exit(3)

    iptm = ok["boltz_iptm"].values
    s = ok["composite"].values
    pr, pp = pearsonr(iptm, s)
    sr, sp = spearmanr(iptm, s)
    print(f"\n[validate] composite vs boltz_iptm:")
    print(f"  Pearson r  = {pr:+.4f}  p={pp:.2e}")
    print(f"  Spearman ρ = {sr:+.4f}  p={sp:.2e}")
    print()
    print("[validate] per-channel correlations vs boltz_iptm:")
    for col in ["shape_score", "shape_mean_d", "shape_cov", "hinge_score",
                "bd_score", "clash_gate", "rmsd_align"]:
        v = ok[col].values
        if np.std(v) < 1e-9:
            print(f"  {col:12s}: (constant)")
            continue
        prc, _ = pearsonr(iptm, v)
        src, _ = spearmanr(iptm, v)
        print(f"  {col:12s}: Pearson={prc:+.4f}  Spearman={src:+.4f}")
    print()

    # Also correlate against the more direct geometry metrics if present.
    for tgt_col in ["d_SG", "burgi_dunitz_dev_deg", "boltz_ligand_iptm"]:
        if tgt_col in df.columns:
            tv = df.loc[ok.index, tgt_col].values
            mask = ~np.isnan(tv)
            if mask.sum() >= 10 and np.std(tv[mask]) > 1e-9:
                prc, _ = pearsonr(tv[mask], s[mask])
                src, _ = spearmanr(tv[mask], s[mask])
                print(f"[validate] composite vs {tgt_col}: Pearson={prc:+.4f} Spearman={src:+.4f}")

    print()
    ok_bar = pr >= args.threshold
    print(f"[validate] ACCEPTANCE: Pearson r {pr:+.4f} {'>=' if ok_bar else '<'} {args.threshold:+.2f}  "
          f"{'PASS — ship to V100' if ok_bar else 'FAIL — redesign needed'}")
    sys.exit(0 if ok_bar else 1)


if __name__ == "__main__":
    main()
