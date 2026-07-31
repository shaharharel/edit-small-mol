"""Strain compute on an input CSV with (row_id, smiles). Writes (row_id, strain).
Used to shard work across machines."""
import sys, os, time, multiprocessing as mp, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

_SCORE = None
def _init():
    global _SCORE
    # Avoid triggering src/utils/__init__.py (pulls sklearn which may be missing
    # on remote machines). Import rdkit_strain.py directly by file path.
    import importlib.util
    rs_path = Path(__file__).resolve().parents[1] / "src" / "utils" / "rdkit_strain.py"
    spec = importlib.util.spec_from_file_location("rdkit_strain", rs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SCORE = mod.score_posefree

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with row_id,smiles")
    ap.add_argument("--output", required=True, help="output CSV row_id,strain")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.input)
    print(f"[load] {len(df):,} rows -> {args.input}", flush=True)

    pairs = list(zip(df["row_id"].tolist(), df["smiles"].tolist()))
    chunks = [pairs[i:i+args.chunk] for i in range(0, len(pairs), args.chunk)]
    print(f"[work] {len(chunks)} chunks of {args.chunk}, {args.workers} workers", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    last_write = time.time()
    t1 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init) as pool:
        for i, c in enumerate(pool.imap_unordered(_work, chunks, chunksize=1)):
            results.extend(c)
            if (i+1) % 10 == 0 or (i+1) == len(chunks):
                done = len(results)
                dt = time.time()-t1
                rate = done/dt if dt>0 else 0
                eta = (len(pairs)-done)/rate if rate>0 else 0
                print(f"  chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.1f}/s eta={eta:.0f}s", flush=True)
            # Partial write every 2 min
            if time.time() - last_write > 120:
                pd.DataFrame(results, columns=["row_id","rdkit_strain_posefree_kcal_mol"]).to_csv(args.output, index=False)
                last_write = time.time()
                print(f"  [partial-write] {len(results):,} rows -> {args.output}", flush=True)

    pd.DataFrame(results, columns=["row_id","rdkit_strain_posefree_kcal_mol"]).to_csv(args.output, index=False)
    n_ok = sum(1 for _, v in results if not np.isnan(v))
    print(f"[done] {n_ok:,}/{len(results):,} success in {time.time()-t0:.1f}s -> {args.output}", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
