"""Compute formal_charge_drawn + net_charge_pH74 (Dimorphite-DL) for F4_boltz_full.csv.

Adds two columns:
- formal_charge_drawn: RDKit Chem.GetFormalCharge() on the input SMILES (almost
  always 0 for generator output; reflects how the SMILES is drawn, not biology).
- net_charge_pH74: formal charge of the Dimorphite-DL dominant microspecies at
  pH 7.4 (the biologically meaningful "net charge").

Why both: formal_charge_drawn is the cheap RDKit baseline London asked about;
net_charge_pH74 is what discriminates 0 vs +1 vs +2 (e.g. ibrutinib +1, dasatinib
+1, Mol1 +1, drugs with piperazine ~+2).

Reads/writes data/tier4_scored/F4_boltz_full.csv in place.
"""
import sys
import time
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from dimorphite_dl import protonate_smiles

RDLogger.DisableLog('rdApp.*')


def compute_charges(smi: str):
    """Return (formal_charge_drawn, net_charge_pH74). NaN on parse failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    fc = Chem.GetFormalCharge(m)
    try:
        out = protonate_smiles(smi, ph_min=7.4, ph_max=7.4, precision=0.5)
    except Exception:
        return fc, None
    if not out:
        return fc, None
    m2 = Chem.MolFromSmiles(out[0])
    if m2 is None:
        return fc, None
    return fc, Chem.GetFormalCharge(m2)


def main():
    csv = Path('/Users/shaharharel/Documents/github/edit-small-mol/data/tier4_scored/F4_boltz_full.csv')
    df = pd.read_csv(csv)
    print(f'Loaded {len(df):,} rows × {len(df.columns)} cols')

    t0 = time.time()
    drawn, ph74 = [], []
    for i, smi in enumerate(df['smiles'].tolist()):
        d, p = compute_charges(smi)
        drawn.append(d)
        ph74.append(p)
        if (i + 1) % 200 == 0:
            print(f'  {i+1:,}/{len(df):,} ({(i+1)/(time.time()-t0):.0f} mol/s)')

    df['formal_charge_drawn'] = drawn
    df['net_charge_pH74'] = ph74
    df.to_csv(csv, index=False)
    print(f'Saved (+2 columns) in {time.time()-t0:.1f}s')

    # Distribution summary
    print('\nformal_charge_drawn distribution:')
    print(df['formal_charge_drawn'].value_counts().sort_index().to_string())
    print('\nnet_charge_pH74 distribution (the one London cares about):')
    print(df['net_charge_pH74'].value_counts().sort_index().to_string())
    print(f'\nNaN: drawn={df["formal_charge_drawn"].isna().sum()}, pH74={df["net_charge_pH74"].isna().sum()}')


if __name__ == '__main__':
    main()
