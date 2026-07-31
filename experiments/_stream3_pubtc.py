"""Stream 3: max_pubTc vs 300-lead panel for missing rows. ~10 min on 8 cores."""
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

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

OUT = ROOT/'data/paper_evaluation/l3_cheap_backfill.csv'
V4 = ROOT/'results/paper_evaluation/all_methods_bulk_scored_v4.csv'
PUBTC_CACHE = ROOT/'data/paper_evaluation/pubtc_scores_v3.csv'

_FPS = []
def _init(panel_bytes):
    global _FPS
    _FPS = [DataStructs.CreateFromBinaryText(b) for b in panel_bytes]

def _work(chunk):
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan)); continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append((rid, np.nan)); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _FPS)
        out.append((rid, float(max(sims))))
    return out

def main():
    t0 = time.time()
    out_df = pd.read_csv(OUT)
    print(f"[load] CSV {len(out_df):,} rows in {time.time()-t0:.1f}s")

    # Lift cached values by SMILES match (avoid recomputing 620K legacy rows)
    if PUBTC_CACHE.exists() and out_df["max_pubTc"].isna().all():
        print(f"[cache] lifting from {PUBTC_CACHE.name} by SMILES match ...")
        pt = pd.read_csv(PUBTC_CACHE, usecols=["row_id", "max_pubTc"])
        v4_smi = pd.read_csv(V4, usecols=["row_id", "smiles"])
        pt_smi = pt.merge(v4_smi, on="row_id", how="left").dropna(subset=["smiles","max_pubTc"])
        pt_smi = pt_smi.drop_duplicates("smiles")
        lookup = dict(zip(pt_smi["smiles"], pt_smi["max_pubTc"]))
        mapped = out_df["smiles"].map(lookup)
        out_df.loc[mapped.notna(), "max_pubTc"] = mapped[mapped.notna()].values
        n_lifted = out_df["max_pubTc"].notna().sum()
        print(f"[cache] lifted {n_lifted:,} rows by SMILES")
        # Write partial
        out_df.to_csv(OUT, index=False)
        del pt, v4_smi, pt_smi, lookup, mapped
        import gc; gc.collect()

    miss = out_df[out_df["max_pubTc"].isna()].copy()
    print(f"[miss] {len(miss):,} rows need max_pubTc")
    if len(miss) == 0:
        return

    # Build panel FPs
    from pubtc_panel_v3 import PANEL_SMILES
    panel_fps = []
    for name, smi in PANEL_SMILES.items():
        m = Chem.MolFromSmiles(smi)
        if m is None: raise RuntimeError(f"panel parse failed: {name}")
        panel_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
    print(f"[panel] {len(panel_fps)} panel FPs built")
    panel_bytes = [DataStructs.BitVectToBinaryText(p) for p in panel_fps]

    pairs = list(zip(miss["row_id"].tolist(), miss["smiles"].tolist()))
    chunk_size = 500
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]
    print(f"[work] {len(chunks)} chunks of {chunk_size}, 8 workers")

    t1 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(8, initializer=_init, initargs=(panel_bytes,)) as pool:
        for i, c in enumerate(pool.imap_unordered(_work, chunks, chunksize=1)):
            results.extend(c)
            if (i+1) % 20 == 0 or (i+1) == len(chunks):
                done = len(results)
                rate = done / (time.time()-t1)
                eta = (len(pairs)-done) / rate if rate > 0 else 0
                print(f"  chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  rate={rate:.0f}/s eta={eta:.0f}s", flush=True)

    rids, vals = zip(*results)
    new_v = pd.Series(vals, index=rids)
    out_df.loc[out_df["row_id"].isin(new_v.index), "max_pubTc"] = (
        out_df.loc[out_df["row_id"].isin(new_v.index), "row_id"].map(new_v.to_dict())
    )
    out_df.to_csv(OUT, index=False)
    n_ok = out_df["max_pubTc"].notna().sum()
    print(f"[done] max_pubTc now {n_ok:,}/{len(out_df):,}  ({100*n_ok/len(out_df):.1f}%)  "
          f"mean={out_df['max_pubTc'].mean():.3f} min={out_df['max_pubTc'].min():.3f} max={out_df['max_pubTc'].max():.3f}")
    print(f"[wall] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
