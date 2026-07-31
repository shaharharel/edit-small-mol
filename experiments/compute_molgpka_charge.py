"""Compute MolGpKa fractional + integer charge at pH 7.4 for F4_boltz_full.csv.

Why this is better than Dimorphite-DL (run earlier):
  Dimorphite returns the DOMINANT microstate as an integer. MolGpKa predicts
  the pKa of every ionizable site, then we Henderson-Hasselbalch to get the
  population-weighted fractional charge — the right physical quantity.

  Concrete sanity case: Mol1's 2-amino-N-isopropyl-imidazole. Dimorphite uses a
  generic imidazole rule and returns +1. MolGpKa correctly predicts pKa ≈ 5.5
  (the 2-amino donates into the ring, shifting pKa down by ~2 units) → frac
  charge +0.017 (essentially neutral). Dasatinib +1.196 (matches the literature
  ~+1.1 for the piperazine at pH 7.4). Imatinib +1.428.

Adds three columns:
  frac_charge_pH74      : population-weighted charge (real-valued)
  net_charge_pH74_mg    : rounded integer (for display / threshold filters)
  pKa_basic_max         : highest predicted basic pKa (most basic site)

Writes back to data/tier4_scored/F4_boltz_full.csv in place.
"""
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# MolGpKa lives at /tmp/MolGpKa/src (cloned from github.com/Xundrug/MolGpKa).
# Its modules import as `from utils.X import Y` so we have to cd into src.
MOLGPKA_SRC = Path("/tmp/MolGpKa/src")
sys.path.insert(0, str(MOLGPKA_SRC))
import os
_prev_cwd = os.getcwd()
os.chdir(MOLGPKA_SRC)
from predict_pka import predict  # noqa: E402

PH = 7.4


def frac_charge(smi: str):
    """Return (frac_charge, integer_rounded, pKa_basic_max)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    try:
        base_dict, acid_dict = predict(m)
    except Exception:
        return None, None, None
    frac_b = sum(1.0 / (1 + 10 ** (PH - pka)) for pka in base_dict.values())
    frac_a = sum(1.0 / (1 + 10 ** (pka - PH)) for pka in acid_dict.values())
    frac = frac_b - frac_a
    pka_basic_max = max(base_dict.values()) if base_dict else None
    return frac, round(frac), pka_basic_max


def main():
    csv = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/tier4_scored/F4_boltz_full.csv")
    df = pd.read_csv(csv)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

    t0 = time.time()
    frac, integer, pka_b = [], [], []
    for i, smi in enumerate(df["smiles"].tolist()):
        f, n, p = frac_charge(smi)
        frac.append(f)
        integer.append(n)
        pka_b.append(p)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  {i+1:,}/{len(df):,}  ({rate:.1f} mol/s, ETA {eta/60:.1f} min)")

    df["frac_charge_pH74"] = frac
    df["net_charge_pH74_mg"] = integer
    df["pKa_basic_max"] = pka_b
    df.to_csv(csv, index=False)
    print(f"\nSaved (+3 columns) in {(time.time()-t0)/60:.1f} min")

    print("\nfrac_charge_pH74 distribution:")
    print(df["frac_charge_pH74"].describe().to_string())
    print("\nnet_charge_pH74_mg (rounded) distribution:")
    print(df["net_charge_pH74_mg"].value_counts().sort_index().to_string())
    print(f"\nNaN count: frac={df['frac_charge_pH74'].isna().sum()}, mg={df['net_charge_pH74_mg'].isna().sum()}")

    os.chdir(_prev_cwd)


if __name__ == "__main__":
    main()
