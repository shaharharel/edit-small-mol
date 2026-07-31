#!/usr/bin/env python3
"""Phase 2 of the tier4 backend pipeline: FiLM pIC50 inference only.

Reads the phase-1 snapshot CSV produced by `tier4_to_backend.py` and adds:
  pIC50_method, pIC50_film, delta_vs_mol1, direct_delta_from_mol1,
  anchor_wins, anchor_wins_ge7.

Designed to run on a V100 (CPU inference, ~10 min on the V100's 8 CPUs) or
ai-chem (16 CPUs, ~5-6 min). Output replaces the final CSV at
`results/paper_evaluation/tier4_overnight_bulk.csv`.

CLI:
    python tier4_film_phase2.py <phase1_csv> <out_csv>
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_LOCAL = Path("/Users/shaharharel/Documents/github/edit-small-mol")
PROJECT_V100 = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_AICHEM = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = next((p for p in (PROJECT_LOCAL, PROJECT_V100, PROJECT_AICHEM) if p.exists()), None)
if PROJECT_ROOT is None:
    raise SystemExit("project root not found in any known location")

sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"


def main():
    if len(sys.argv) < 3:
        print("usage: tier4_film_phase2.py <phase1_csv> <out_csv>", file=sys.stderr)
        sys.exit(2)
    phase1 = Path(sys.argv[1])
    out = Path(sys.argv[2])
    print(f"loading phase-1 CSV: {phase1} ({phase1.stat().st_size/1e6:.1f} MB)", flush=True)
    df = pd.read_csv(phase1)
    n = len(df)
    print(f"  {n:,} rows", flush=True)
    print(f"  cohorts: {df['method'].value_counts().to_dict()}", flush=True)

    import reinvent4_film_scorer as fs
    model, scaler, anchor_embs, anchor_pIC50 = fs.load_film_model()
    print(f"  FiLM model loaded with {len(anchor_pIC50)} anchors", flush=True)

    smis = df["smiles"].astype(str).tolist()
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
            print(f"  ...{i+len(chunk):,}/{n:,}  ({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.1f} min)", flush=True)

    df["pIC50_method"] = pic50s
    df["pIC50_film"] = pic50s
    # Set pIC50_mean = pIC50_method (anchor-based prediction is mean over 280 anchors).
    # Setting pIC50_std = NaN since we don't have ensemble variance from a single FiLM model.
    df["pIC50_mean"] = pic50s
    df["pIC50_std"] = np.nan
    mol1_score = fs.score_smiles([MOL1_SMILES], model, scaler, anchor_embs, anchor_pIC50)[0]
    df["delta_vs_mol1"] = df["pIC50_method"] - mol1_score
    df["direct_delta_from_mol1"] = df["delta_vs_mol1"]
    df["anchor_wins"] = (df["pIC50_method"] > mol1_score).astype("Int64")
    df["anchor_wins_ge7"] = ((df["pIC50_method"] > mol1_score) & (df["pIC50_method"] >= 7.0)).astype("Int64")
    print(f"  Mol-1 FiLM pIC50 reference: {mol1_score:.3f}", flush=True)
    print(f"  Mean pIC50 (all): {np.nanmean(pic50s):.3f}", flush=True)
    print(f"  anchor_wins: {df['anchor_wins'].sum()}/{n}  ·  anchor_wins_ge7: {df['anchor_wins_ge7'].sum()}/{n}", flush=True)

    df.to_csv(out, index=False)
    print(f"\n  → {out}  ({len(df):,} rows, {out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
