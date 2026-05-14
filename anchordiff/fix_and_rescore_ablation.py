"""Fix warhead bonds on each ablation cohort SDF, then rescore with the clean
FiLMDelta. Mirrors the Day-1 pipeline (which manually applied the bond fix
after DiffSBDD output) — the 24h master script skipped that step.

For acrylamide-warhead cohorts (5 of 6) the canonical bond pattern is
0=1 / 1-2 / 2=3 / 2-4 with atoms (C, C, C, O, N). For the chloroacetamide
cohort it's 0-1 / 0-2 / 2=3 / 2-4 with atoms (C, Cl, C, O, N).

For each cohort:
  1. Run fix_warhead_bonds (auto-detects acrylamide vs chloroacetamide by
     element-pattern at atoms 0-4).
  2. Rescore with clean FiLMDelta.
  3. Aggregate + report headline numbers, especially the kill-shot
     acrylamide-vs-chloroacetamide A/B test.
"""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog("rdApp.*")
from scipy.stats import ks_2samp, mannwhitneyu

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL  = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CHLORO = Chem.MolFromSmarts("[Cl][CH2]C(=O)N")
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"


def fix_warhead_bonds(mol):
    """Auto-detect acrylamide vs chloroacetamide from atoms 0-4 element pattern,
    rebuild canonical warhead bond order + connectivity, keep warhead-containing
    fragment. Returns fixed Mol or None."""
    if mol is None or mol.GetNumAtoms() < 6: return None
    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(5)]

    is_acryl  = elems == ["C", "C", "C", "O", "N"]
    is_chloro = elems == ["C", "Cl", "C", "O", "N"]
    if not (is_acryl or is_chloro):
        return None

    rwm = Chem.RWMol(mol)
    # Step: remove all bonds involving warhead-internal atoms (acryl: 0-3, chloro: 0,2,3)
    if is_acryl:
        internal = {0, 1, 2, 3}; n_idx = 4
    else:  # chloroacetamide: 0=Cα, 1=Cl (leaving group), 2=Ccarb, 3=O, 4=N
        internal = {0, 2, 3}; n_idx = 4
    bonds_to_remove = []
    misbonded_scaffold = set()
    for b in rwm.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in internal or j in internal:
            # track scaffold attachments to warhead-internal C's (perceiver bug)
            if i in internal and j >= 5: misbonded_scaffold.add(j)
            if j in internal and i >= 5: misbonded_scaffold.add(i)
            bonds_to_remove.append((i, j))
    for i, j in bonds_to_remove:
        if rwm.GetBondBetweenAtoms(i, j) is not None:
            rwm.RemoveBond(i, j)
    # Also remove Cl-* bonds for chloroacetamide (Cl only bonds to Cα)
    if is_chloro:
        for b in list(rwm.GetBonds()):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (i == 1 and j != 0) or (j == 1 and i != 0):
                rwm.RemoveBond(i, j)

    # Add canonical bonds
    def _add(a, b, bt):
        if rwm.GetBondBetweenAtoms(a, b) is None:
            rwm.AddBond(a, b, bt)
        else:
            rwm.GetBondBetweenAtoms(a, b).SetBondType(bt)

    if is_acryl:
        _add(0, 1, Chem.BondType.DOUBLE)  # C_β=C_α
        _add(1, 2, Chem.BondType.SINGLE)  # C_α-C_carb
        _add(2, 3, Chem.BondType.DOUBLE)  # C_carb=O
        _add(2, 4, Chem.BondType.SINGLE)  # C_carb-N
    else:
        _add(0, 1, Chem.BondType.SINGLE)  # C_α-Cl
        _add(0, 2, Chem.BondType.SINGLE)  # C_α-C_carb
        _add(2, 3, Chem.BondType.DOUBLE)  # C_carb=O
        _add(2, 4, Chem.BondType.SINGLE)  # C_carb-N

    # Re-attach scaffold to N if dangling
    n_has = any(b.GetOtherAtom(rwm.GetAtomWithIdx(n_idx)).GetIdx() >= 5
                for b in rwm.GetAtomWithIdx(n_idx).GetBonds())
    if not n_has:
        conf = rwm.GetConformer()
        n_pos = np.array(list(conf.GetAtomPosition(n_idx)))
        cands = []
        for k in range(5, rwm.GetNumAtoms()):
            d = np.linalg.norm(np.array(list(conf.GetAtomPosition(k))) - n_pos)
            if d < 2.5: cands.append((d, k))
        cands.sort()
        misbonded_in_cands = [c for c in cands if c[1] in misbonded_scaffold]
        chosen = (misbonded_in_cands[0] if misbonded_in_cands else (cands[0] if cands else None))
        if chosen is not None:
            rwm.AddBond(n_idx, chosen[1], Chem.BondType.SINGLE)

    # reset warhead atom flags
    for i in range(5):
        a = rwm.GetAtomWithIdx(i)
        a.SetNoImplicit(False); a.SetNumExplicitHs(0)
        a.SetFormalCharge(0); a.SetIsAromatic(False)

    new_mol = rwm.GetMol()
    try: Chem.SanitizeMol(new_mol)
    except Exception: return None

    # keep warhead-containing fragment only
    frags = Chem.GetMolFrags(new_mol, asMols=False)
    if len(frags) > 1:
        wh_frag = next((f for f in frags if 0 in f), None)
        if wh_frag is None or len(wh_frag) < 6: return None
        keep = set(wh_frag)
        to_drop = sorted([i for i in range(new_mol.GetNumAtoms()) if i not in keep], reverse=True)
        rwm2 = Chem.RWMol(new_mol)
        for i in to_drop: rwm2.RemoveAtom(i)
        new_mol = rwm2.GetMol()
        try: Chem.SanitizeMol(new_mol)
        except Exception: return None
    return new_mol


def load_clean_scorer():
    ck = torch.load(CLEAN_CKPT, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck["anchor_embs"], np.asarray(ck["anchor_pIC50"])


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
    return a


def score(smi, model, scaler, ae, ap):
    a = fp(smi)
    if a is None: return float("nan")
    e = torch.FloatTensor(scaler.transform(a[None, :]))
    with torch.no_grad():
        d = model(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def tc(smi_a, smi_b):
    a = fp(smi_a); b = fp(smi_b)
    if a is None or b is None: return float("nan")
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi_a), 2, nBits=2048),
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi_b), 2, nBits=2048),
    )


def main():
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    model, scaler, ae, ap = load_clean_scorer()
    print(f"loaded clean FiLMDelta ({len(ap)} anchors)")
    ablation_dir = PROJECT_ROOT / "anchordiff_results" / "24h" / "ablation"
    rows = []; summary = []
    for sdf in sorted(ablation_dir.glob("*.sdf")):
        cohort = sdf.stem
        suppl = Chem.SDMolSupplier(str(sdf), sanitize=False)
        n_total = n_fixed = n_warhead = 0
        cohort_rows = []
        warhead_type = "chloroacetamide" if cohort == "warhead_chloroacetamide" else "acrylamide"
        warhead_match = CHLORO if warhead_type == "chloroacetamide" else ACRYL
        out_fixed = ablation_dir / f"{cohort}_fixed.sdf"
        wfix = Chem.SDWriter(str(out_fixed))
        for m in suppl:
            if m is None: continue
            n_total += 1
            fixed = fix_warhead_bonds(m)
            if fixed is None: continue
            n_fixed += 1
            if not fixed.HasSubstructMatch(warhead_match): continue
            n_warhead += 1
            wfix.write(fixed)
            smi = Chem.MolToSmiles(fixed)
            cohort_rows.append({
                "cohort": cohort, "warhead_type": warhead_type, "smiles": smi,
                "MW": Descriptors.MolWt(fixed),
                "QED": float(Descriptors.qed(fixed)),
                "pIC50_clean": score(smi, model, scaler, ae, ap),
                "tc_to_mol1": tc(smi, MOL1),
            })
        wfix.close()
        df = pd.DataFrame(cohort_rows)
        if not df.empty:
            print(f"\n=== {cohort} ({warhead_type}) ===")
            print(f"  read={n_total}  fixed={n_fixed}  warhead_OK={n_warhead}")
            print(f"  pIC50 (clean):  min={df['pIC50_clean'].min():.2f}  "
                  f"med={df['pIC50_clean'].median():.2f}  "
                  f"max={df['pIC50_clean'].max():.2f}")
            print(f"  Top 5:")
            for _, r in df.nlargest(5, "pIC50_clean").iterrows():
                print(f"    pIC50={r['pIC50_clean']:.2f}  MW={r['MW']:.0f}  "
                      f"QED={r['QED']:.2f}  Tc={r['tc_to_mol1']:.2f}  {r['smiles'][:60]}")
            rows.extend(cohort_rows)
            summary.append({"cohort": cohort, "warhead_type": warhead_type,
                            "n_total": n_total, "n_fixed": n_fixed, "n_warhead": n_warhead,
                            "pIC50_med": df["pIC50_clean"].median(),
                            "pIC50_max": df["pIC50_clean"].max(),
                            "tc_med": df["tc_to_mol1"].median()})
        else:
            print(f"\n=== {cohort}: 0 mols passed fix + warhead check ===")

    pd.DataFrame(rows).to_csv(out_dir / "ablation_scored_clean.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "ablation_summary_clean.csv", index=False)
    print(f"\nwrote ablation_scored_clean.csv + ablation_summary_clean.csv")

    full = pd.DataFrame(rows)
    if {"warhead_chloroacetamide", "baseline"} <= set(full["cohort"].unique()):
        a = full[full["cohort"] == "baseline"]["pIC50_clean"].dropna()
        b = full[full["cohort"] == "warhead_chloroacetamide"]["pIC50_clean"].dropna()
        ks_p = ks_2samp(a, b).pvalue
        mwu_p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"\n=== KILL-SHOT (acrylamide baseline vs chloroacetamide) ===")
        print(f"  baseline   n={len(a)}  med={a.median():.2f}  top1={a.max():.2f}")
        print(f"  chloro     n={len(b)}  med={b.median():.2f}  top1={b.max():.2f}")
        print(f"  KS p={ks_p:.4f}  MWU p={mwu_p:.4f}  ΔMed={b.median()-a.median():+.3f}")


if __name__ == "__main__":
    main()
