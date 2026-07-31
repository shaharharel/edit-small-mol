"""Stream 4: rdkit_strain_posefree for missing rows. Lift cache by SMILES first,
then compute remainder on Mac CPU."""
import sys, os, time, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/shaharharel/Documents/github/edit-small-mol')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = ROOT/'data/paper_evaluation/l3_cheap_backfill.csv'
V4 = ROOT/'results/paper_evaluation/all_methods_bulk_scored_v4.csv'
STRAIN_CACHE = ROOT/'data/paper_evaluation/rdkit_strain_posefree.csv'
COL = "rdkit_strain_posefree_kcal_mol"

_SCORE = None
def _init():
    global _SCORE
    from src.utils.rdkit_strain import score_posefree
    _SCORE = score_posefree

def _work(chunk):
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan)); continue
        try:
            r = _SCORE(smi, num_conf_free=2, seed=42, free_max_its=100)
            if r.get("success_flag") == 1:
                out.append((rid, float(r["strain_kcal_mol"])))
            else:
                out.append((rid, np.nan))
        except Exception:
            out.append((rid, np.nan))
    return out

def main():
    t0 = time.time()
    out_df = pd.read_csv(OUT)
    print(f"[load] CSV {len(out_df):,} rows in {time.time()-t0:.1f}s")

    # Lift from cache by SMILES
    if STRAIN_CACHE.exists() and out_df[COL].isna().all():
        print(f"[cache] lifting from {STRAIN_CACHE.name} by SMILES match ...")
        st = pd.read_csv(STRAIN_CACHE, usecols=["row_id","strain_kcal_mol","success_flag"])
        st = st[st["success_flag"]==1][["row_id","strain_kcal_mol"]]
        v4 = pd.read_csv(V4, usecols=["row_id","smiles"])
        st = st.merge(v4, on="row_id", how="left").dropna(subset=["smiles","strain_kcal_mol"])
        st = st.drop_duplicates("smiles")
        lookup = dict(zip(st["smiles"], st["strain_kcal_mol"]))
        mapped = out_df["smiles"].map(lookup)
        out_df.loc[mapped.notna(), COL] = mapped[mapped.notna()].values
        n_lifted = out_df[COL].notna().sum()
        print(f"[cache] lifted {n_lifted:,} rows by SMILES")
        out_df.to_csv(OUT, index=False)
        del st, v4, lookup, mapped
        import gc; gc.collect()

    miss = out_df[out_df[COL].isna()].copy()
    print(f"[miss] {len(miss):,} rows need {COL}")
    if len(miss) == 0:
        return

    pairs = list(zip(miss["row_id"].tolist(), miss["smiles"].tolist()))
    chunk_size = 100
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]
    workers = int(os.environ.get("STRAIN_WORKERS", "12"))
    print(f"[work] {len(chunks)} chunks of {chunk_size}, {workers} workers")

    t1 = time.time()
    results = []
    last_write = t1
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        for i, c in enumerate(pool.imap_unordered(_work, chunks, chunksize=1)):
            results.extend(c)
            if (i+1) % 10 == 0 or (i+1) == len(chunks):
                done = len(results)
                dt = time.time()-t1
                rate = done/dt if dt > 0 else 0
                eta = (len(pairs)-done)/rate if rate > 0 else 0
                print(f"  chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.1f}/s eta={eta:.0f}s", flush=True)
            # Periodic partial CSV write every 5 min
            if time.time() - last_write > 300:
                rids, vals = zip(*results)
                new_v = pd.Series(vals, index=rids)
                out_df.loc[out_df["row_id"].isin(new_v.index), COL] = (
                    out_df.loc[out_df["row_id"].isin(new_v.index), "row_id"].map(new_v.to_dict())
                )
                out_df.to_csv(OUT, index=False)
                last_write = time.time()
                print(f"  [partial-write] CSV saved with {out_df[COL].notna().sum():,} non-null", flush=True)

    rids, vals = zip(*results)
    new_v = pd.Series(vals, index=rids)
    out_df.loc[out_df["row_id"].isin(new_v.index), COL] = (
        out_df.loc[out_df["row_id"].isin(new_v.index), "row_id"].map(new_v.to_dict())
    )
    out_df.to_csv(OUT, index=False)
    n_ok = out_df[COL].notna().sum()
    print(f"[done] {COL} now {n_ok:,}/{len(out_df):,}  ({100*n_ok/len(out_df):.1f}%)  "
          f"mean={out_df[COL].mean():.2f} min={out_df[COL].min():.2f} max={out_df[COL].max():.2f}")
    print(f"[wall] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
