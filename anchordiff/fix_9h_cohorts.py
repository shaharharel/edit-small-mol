"""Apply warhead bond-order fix + rescore all 5 cohorts from the 9h run.

The 5 cohort SDFs have the warhead atoms at correct positions but DiffSBDD's
bond-perceiver mis-assigned the orders → SMARTS `[CH2]=[CH]C(=O)N` doesn't
match. The fix script re-forces canonical orders + keeps only the largest
warhead-containing fragment.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs, Crippen
RDLogger.DisableLog("rdApp.*")
from anchordiff.fix_inpaint_warhead_bonds import fix_warhead_bonds

ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
COHORTS = {
    "ZAP70 vanilla":    "phase2/zap70_vanilla.sdf",
    "ZAP70 fine-tuned": "phase4/zap70_finetuned.sdf",
    "BTK vanilla":      "phase3/btk_vanilla.sdf",
    "EGFR fine-tuned":  "phase5/egfr_finetuned.sdf",
    "KRAS fine-tuned":  "phase6/kras_finetuned.sdf",
}


def main():
    base = PROJECT_ROOT / "results" / "9h_run"
    rows = []
    for name, rel in COHORTS.items():
        in_sdf = base / rel
        out_sdf = in_sdf.with_name(in_sdf.stem + "_fixed.sdf")
        if not in_sdf.exists():
            print(f"  MISSING: {in_sdf}")
            continue
        suppl = Chem.SDMolSupplier(str(in_sdf), sanitize=False)
        n_in = n_fixed = n_warhead = 0
        w = Chem.SDWriter(str(out_sdf))
        recs = []
        for m in suppl:
            if m is None: continue
            n_in += 1
            new_mol = fix_warhead_bonds(m)
            if new_mol is None: continue
            n_fixed += 1
            if new_mol.HasSubstructMatch(ACRYL):
                n_warhead += 1
                smi = Chem.MolToSmiles(new_mol)
                recs.append({
                    "smi": smi,
                    "MW": Descriptors.MolWt(new_mol),
                    "QED": float(Descriptors.qed(new_mol)),
                    "logP": Crippen.MolLogP(new_mol),
                })
                w.write(new_mol)
        w.close()
        rate = 100 * n_warhead / max(n_in, 1)
        print(f"\n{name}")
        print(f"  in={n_in}  fixed={n_fixed}  acrylamide-matching={n_warhead} ({rate:.0f}%)")
        if recs:
            df = pd.DataFrame(recs)
            df["lipinski"] = ((df.MW <= 500) & (df.logP <= 5)).astype(int)
            print(f"  MW med={df.MW.median():.0f}  QED med={df.QED.median():.2f}  Lipinski={100*df.lipinski.mean():.0f}%")
            df["cohort"] = name
            rows.extend(df.to_dict("records"))
    if rows:
        full = pd.DataFrame(rows)
        out_csv = base / "all_cohorts_warhead_fixed.csv"
        full.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
        print("\n=== HEAD-TO-HEAD (post-bond-fix) ===")
        za_v = full[full.cohort == "ZAP70 vanilla"]
        za_f = full[full.cohort == "ZAP70 fine-tuned"]
        print(f"  ZAP70 vanilla   :  n={len(za_v):3d}  QED med={za_v.QED.median():.2f}  MW med={za_v.MW.median():.0f}  Lipinski={100*za_v.lipinski.mean():.0f}%")
        print(f"  ZAP70 fine-tuned:  n={len(za_f):3d}  QED med={za_f.QED.median():.2f}  MW med={za_f.MW.median():.0f}  Lipinski={100*za_f.lipinski.mean():.0f}%")


if __name__ == "__main__":
    main()
