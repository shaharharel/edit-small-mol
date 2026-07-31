"""Stream 2: SAScore for missing rows. ~1-2 min on 8 cores."""
import sys, os, time, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/shaharharel/Documents/github/edit-small-mol')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'external/DiffSBDD/analysis/SA_Score'))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = ROOT/'data/paper_evaluation/l3_cheap_backfill.csv'

_SA = None
def _init():
    global _SA
    import sascorer
    _SA = sascorer

def _work(chunk):
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan)); continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append((rid, np.nan)); continue
        try:
            out.append((rid, float(_SA.calculateScore(m))))
        except Exception:
            out.append((rid, np.nan))
    return out

def main():
    t0 = time.time()
    out_df = pd.read_csv(OUT)
    print(f"[load] CSV {len(out_df):,} rows in {time.time()-t0:.1f}s")
    miss = out_df[out_df["SAScore"].isna()].copy()
    print(f"[miss] {len(miss):,} rows need SAScore")
    if len(miss) == 0:
        print("nothing to do"); return

    pairs = list(zip(miss["row_id"].tolist(), miss["smiles"].tolist()))
    chunk_size = 2000
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]
    print(f"[work] {len(chunks)} chunks of {chunk_size}, 8 workers")

    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(8, initializer=_init) as pool:
        for i, c in enumerate(pool.imap_unordered(_work, chunks, chunksize=1)):
            results.extend(c)
            if (i+1) % 10 == 0 or (i+1) == len(chunks):
                done = len(results)
                rate = done / (time.time()-t0)
                eta = (len(pairs)-done) / rate if rate > 0 else 0
                print(f"  chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  rate={rate:.0f}/s eta={eta:.0f}s", flush=True)

    rids, vals = zip(*results)
    new_sa = pd.Series(vals, index=rids)
    out_df.loc[out_df["row_id"].isin(new_sa.index), "SAScore"] = (
        out_df.loc[out_df["row_id"].isin(new_sa.index), "row_id"].map(new_sa.to_dict())
    )
    out_df.to_csv(OUT, index=False)
    n_ok = out_df["SAScore"].notna().sum()
    print(f"[done] SAScore now {n_ok:,}/{len(out_df):,}  ({100*n_ok/len(out_df):.1f}%)  "
          f"mean={out_df['SAScore'].mean():.2f} min={out_df['SAScore'].min():.2f} max={out_df['SAScore'].max():.2f}")
    print(f"[wall] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
