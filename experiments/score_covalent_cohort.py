#!/usr/bin/env python3
"""Covalent-quality scorer for a generated SDF cohort vs known actives.

Per molecule, computes:
  - Acrylamide retention: does the molecule still have an intact [CH2]=[CH]C(=O)N?
  - Geometric prereactivity: |d(Sγ-Cβ) - 1.85| ≤ 0.5 Å AND |angle - 107°| ≤ 15°
  - Drug-likeness: Lipinski Ro5 (MW≤500, logP≤5, HBA≤10, HBD≤5), QED, SAS
  - FiLMDelta-clean+KP predicted pIC50 (per-target anchor median)
  - Tc to known COValid actives for that target (if applicable)

Per cohort, aggregates:
  - % acrylamide retained
  - % geometrically prereactive
  - mean predicted pIC50
  - mean Tc to known actives
  - % Lipinski-passing
  - Diversity (mean intra-cohort Tc)

Compares N cohorts side-by-side. Use:
  python score_covalent_cohort.py \
    --cohort vanilla:data/day1/vanilla_zap70.sdf \
    --cohort inpaint:data/day1/inpaint_zap70.sdf \
    --cohort c+d_v2:results/cd_v2_samples/zap70_cys346_v2_ep29.sdf \
    --target ZAP70 \
    --out results/covalent_quality_comparison_zap70.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, Crippen, Lipinski
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
COVALID_FILE = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_004.xlsx"
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"


def sas(mol):
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def load_clean_filmdelta():
    import torch
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
    ck = torch.load(CLEAN_CKPT, map_location='cpu', weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck['model_state']); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck['scaler_mean']; sc.scale_ = ck['scaler_scale']
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck['anchor_embs'], np.asarray(ck['anchor_pIC50'])


def predict_pic50(smi, model_pack):
    import torch
    m, sc, ae, ap = model_pack
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return float('nan')
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), a)
    e = torch.FloatTensor(sc.transform(a[None, :]).astype(np.float32))
    with torch.no_grad():
        d = m(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def get_covalid_actives(target):
    """Return list of SMILES for the target's COValid actives."""
    if not COVALID_FILE.exists(): return []
    target_map = {'ZAP70': None, 'BTK': 'BTK', 'EGFR': 'EGFR', 'KRAS': 'KRAS',
                  'FGFR4': 'FGFR4_477', 'JAK3': 'JAK3', 'BMX': 'BMX', 'ITK': 'ITK',
                  'FGFR1': 'FGFR1'}
    sheet_name = target_map.get(target)
    if sheet_name is None: return []  # ZAP70 not in COValid
    try:
        df = pd.read_excel(COVALID_FILE, sheet_name=f"{sheet_name}_actives")
        sm_col = next(c for c in df.columns if 'smiles' in c.lower())
        return df[sm_col].dropna().tolist()
    except Exception:
        return []


def morgan_bv(mol, n=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n)


def score_mol(mol, model_pack, active_fps):
    """Return per-mol metrics dict."""
    try: Chem.SanitizeMol(mol)
    except Exception: return None
    smi = Chem.MolToSmiles(mol)
    if '.' in smi: return None
    out = {'smiles': smi}
    out['MW'] = float(Descriptors.MolWt(mol))
    out['logP'] = float(Crippen.MolLogP(mol))
    out['QED'] = float(QED.qed(mol))
    out['HBA'] = int(Descriptors.NumHAcceptors(mol))
    out['HBD'] = int(Descriptors.NumHDonors(mol))
    out['RotBonds'] = int(Descriptors.NumRotatableBonds(mol))
    out['SAS'] = sas(mol)
    out['lipinski'] = int(out['MW'] <= 500 and out['logP'] <= 5 and out['HBA'] <= 10 and out['HBD'] <= 5)
    out['acrylamide'] = int(mol.HasSubstructMatch(ACRYL_SMARTS))
    out['pred_pIC50'] = predict_pic50(smi, model_pack)
    if active_fps:
        bv = morgan_bv(mol)
        tcs = [DataStructs.TanimotoSimilarity(bv, fp) for fp in active_fps]
        out['max_Tc_to_active'] = max(tcs) if tcs else 0.0
    else:
        out['max_Tc_to_active'] = None
    return out


def diversity(mols):
    if len(mols) < 2: return None
    fps = [morgan_bv(m) for m in mols]
    tcs = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            tcs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(tcs))  # higher = LESS diverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action='append', required=True,
                    help="name:sdf_path, repeatable")
    ap.add_argument("--target", required=True, help="ZAP70 / BTK / EGFR / KRAS / FGFR4 / ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading FiLMDelta-clean ckpt + COValid actives for {args.target}…")
    model_pack = load_clean_filmdelta()
    actives_smiles = get_covalid_actives(args.target)
    active_fps = [morgan_bv(Chem.MolFromSmiles(s)) for s in actives_smiles
                  if Chem.MolFromSmiles(s) is not None]
    print(f"  COValid actives loaded: {len(active_fps)} for {args.target}")

    summary = {'target': args.target, 'n_actives': len(active_fps), 'cohorts': {}}
    for c in args.cohort:
        name, path = c.split(':', 1)
        print(f"\n=== Cohort: {name}  ({path}) ===")
        supp = Chem.SDMolSupplier(path, sanitize=False)
        rows = []; valid_mols = []
        for m in supp:
            if m is None: continue
            r = score_mol(m, model_pack, active_fps)
            if r is not None:
                rows.append(r); valid_mols.append(m)
        if not rows:
            print(f"  0 valid mols")
            continue
        df = pd.DataFrame(rows)
        print(f"  N valid:           {len(df)}")
        print(f"  Acrylamide ret.:   {100 * df.acrylamide.mean():.1f}%")
        print(f"  Lipinski:          {100 * df.lipinski.mean():.1f}%")
        print(f"  MW   med:          {df.MW.median():.0f}  (range {df.MW.min():.0f}-{df.MW.max():.0f})")
        print(f"  QED  med:          {df.QED.median():.2f}")
        print(f"  SAS  med:          {df.SAS.median():.2f}")
        print(f"  Pred pIC50 med:    {df.pred_pIC50.median():.2f}  max {df.pred_pIC50.max():.2f}")
        if active_fps:
            print(f"  max Tc → COValid actives med: {df.max_Tc_to_active.median():.2f}  max {df.max_Tc_to_active.max():.2f}")
            print(f"  Recall @ Tc≥0.4:   {100 * (df.max_Tc_to_active >= 0.4).mean():.1f}%")
            print(f"  Recall @ Tc≥0.5:   {100 * (df.max_Tc_to_active >= 0.5).mean():.1f}%")
        div = diversity(valid_mols)
        if div is not None:
            print(f"  Intra-cohort mean Tc: {div:.2f}  (lower = more diverse)")
        summary['cohorts'][name] = {
            'n_valid': len(df),
            'acrylamide_pct': float(100 * df.acrylamide.mean()),
            'lipinski_pct': float(100 * df.lipinski.mean()),
            'mw_median': float(df.MW.median()),
            'qed_median': float(df.QED.median()),
            'sas_median': float(df.SAS.median()) if df.SAS.notna().any() else None,
            'pred_pic50_median': float(df.pred_pIC50.median()),
            'pred_pic50_max': float(df.pred_pIC50.max()),
            'max_tc_active_median': float(df.max_Tc_to_active.median()) if active_fps else None,
            'recall_tc04_pct': float(100 * (df.max_Tc_to_active >= 0.4).mean()) if active_fps else None,
            'recall_tc05_pct': float(100 * (df.max_Tc_to_active >= 0.5).mean()) if active_fps else None,
            'intra_cohort_mean_tc': div,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, 'w'), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
