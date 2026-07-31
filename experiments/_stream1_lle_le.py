"""Stream 1: LLE + LE — vectorized, writes initial CSV in <30s."""
import sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/shaharharel/Documents/github/edit-small-mol')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))

from backfill_l3_cheap_metrics import build_backend_df

OUT = ROOT/'data/paper_evaluation/l3_cheap_backfill.csv'

t0 = time.time()
DF = build_backend_df()
print(f"[build] DF in {time.time()-t0:.1f}s")

pic = pd.to_numeric(DF["pIC50_method"], errors="coerce")
logp = pd.to_numeric(DF["LogP"], errors="coerce")
ha = pd.to_numeric(DF["HeavyAtoms"], errors="coerce")

lle = pic - logp
le = 1.4 * pic / ha.where(ha > 0)

out = pd.DataFrame({
    "row_id": DF["row_id"].values,
    "smiles": DF["smiles"].values,
    "SAScore": np.nan,
    "LLE_method": lle.values,
    "LE_method": le.values,
    "max_pubTc": np.nan,
    "rdkit_strain_posefree_kcal_mol": np.nan,
})

# Also lift SAScore from v4 directly (already in DF column)
sa_src = pd.to_numeric(DF["SAScore"], errors="coerce")
out["SAScore"] = sa_src.values

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
n_lle = out["LLE_method"].notna().sum()
n_le = out["LE_method"].notna().sum()
n_sa = out["SAScore"].notna().sum()
print(f"[done] {len(out):,} rows -> {OUT}")
print(f"  LLE_method: {n_lle:,}  ({100*n_lle/len(out):.1f}%)  "
      f"mean={out['LLE_method'].mean():.3f}  min={out['LLE_method'].min():.3f}  max={out['LLE_method'].max():.3f}")
print(f"  LE_method:  {n_le:,}  ({100*n_le/len(out):.1f}%)  "
      f"mean={out['LE_method'].mean():.3f}  min={out['LE_method'].min():.3f}  max={out['LE_method'].max():.3f}")
print(f"  SAScore:    {n_sa:,}  ({100*n_sa/len(out):.1f}%)  "
      f"mean={out['SAScore'].mean():.2f}  min={out['SAScore'].min():.2f}  max={out['SAScore'].max():.2f}")
print(f"[wall] {time.time()-t0:.1f}s")
