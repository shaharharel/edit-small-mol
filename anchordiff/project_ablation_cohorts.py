"""Apply constraint-manifold projection to the (already-warhead-bond-fixed)
ablation cohort SDFs. This completes the kill-shot — measures whether the
projector actually does work on perturbed warhead inputs.

Per cohort:
  perturbed_fixed.sdf  →  perturbed_fixed_projected.sdf
                      +  per-mol geometry residuals before / after projection

Then re-score both with clean FiLMDelta and compare:
  - Δ pIC50 = projected − perturbed = direct measure of projector contribution
  - Hypothesis: perturbed cohorts (offset/rot) gain more from projection than
    the baseline cohort (which is already on-manifold).
"""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog("rdApp.*")
from Bio.PDB import PDBParser

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from anchordiff.config import ZAP70_CYS346
from anchordiff.covalent_constraint_manifold import (
    WarheadAtoms, measure, project,
)

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL  = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CHLORO = Chem.MolFromSmarts("[Cl][CH2]C(=O)N")
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"
RECEPTOR  = PROJECT_ROOT / "anchordiff" / "pockets" / "zap70_cys346" / "receptor.pdb"


def get_sg_xyz():
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("", str(RECEPTOR))[0]
    for chain in s:
        for res in chain:
            if res.get_resname() == "CYS" and res.id[1] == ZAP70_CYS346.cys_residue and "SG" in res:
                return np.array(res["SG"].get_coord(), dtype=float)
    raise RuntimeError("Cys346 SG not found")


def find_warhead_atoms(mol, kind):
    """For acrylamide: SMARTS gives (cb, ca, c_carb, n_amide).
    For chloroacetamide: SMARTS gives (cl, ca, c_carb, n_amide); we treat
    Cα (atom 1 of the warhead.sdf) as the projection anchor: d(SG, Cα) = 1.85.
    Returns a WarheadAtoms instance or None."""
    if kind == "acrylamide":
        m = mol.GetSubstructMatches(ACRYL)
        if not m: return None
        tup = m[0]
        # SMARTS [CH2]=[CH]C(=O)N matches 5 atoms (incl. =O) on most builds;
        # accept either 4 or 5-atom returns.
        if len(tup) == 5:
            cb, ca, c_carb, _o, n = tup
        elif len(tup) == 4:
            cb, ca, c_carb, n = tup
        else: return None
        return WarheadAtoms(cb=cb, ca=ca, c_carb=c_carb, n_amide=n)
    else:
        m = mol.GetSubstructMatches(CHLORO)
        if not m: return None
        tup = m[0]
        if len(tup) == 5:
            cl, ca, c_carb, _o, n = tup
        elif len(tup) == 4:
            cl, ca, c_carb, n = tup
        else: return None
        # Reuse the WarheadAtoms struct but interpret cb→Cα (attack site)
        # since chloroacetamide's reactive carbon is Cα, not Cβ.
        return WarheadAtoms(cb=ca, ca=cl, c_carb=c_carb, n_amide=n)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
    return a


def score_smi(smi, model, scaler, ae, ap):
    a = fp(smi)
    if a is None: return float("nan")
    e = torch.FloatTensor(scaler.transform(a[None, :]))
    with torch.no_grad():
        d = model(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def load_clean_scorer():
    ck = torch.load(CLEAN_CKPT, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck["anchor_embs"], np.asarray(ck["anchor_pIC50"])


def main():
    sg = get_sg_xyz()
    print(f"Cys346 SG = ({sg[0]:.3f}, {sg[1]:.3f}, {sg[2]:.3f})")
    model, scaler, ae, ap = load_clean_scorer()
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    out_dir.mkdir(exist_ok=True, parents=True)

    ablation_dir = PROJECT_ROOT / "anchordiff_results" / "24h" / "ablation"
    rows = []
    summary = []
    for fixed_sdf in sorted(ablation_dir.glob("*_fixed.sdf")):
        cohort = fixed_sdf.stem.replace("_fixed", "")
        kind = "chloroacetamide" if "chloroacetamide" in cohort else "acrylamide"
        out_proj = ablation_dir / f"{cohort}_fixed_projected.sdf"
        suppl = Chem.SDMolSupplier(str(fixed_sdf), sanitize=True)
        w = Chem.SDWriter(str(out_proj))
        n = 0
        cohort_rows = []
        for mol in suppl:
            if mol is None: continue
            warh = find_warhead_atoms(mol, kind)
            if warh is None: continue
            if mol.GetNumConformers() == 0: continue
            n += 1
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            before = measure(coords, sg, warh)
            new_coords = project(coords, sg, warh)
            after = measure(new_coords, sg, warh)
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, tuple(new_coords[i]))
            w.write(mol)
            smi = Chem.MolToSmiles(mol)
            pIC50_proj = score_smi(smi, model, scaler, ae, ap)
            cohort_rows.append({
                "cohort": cohort, "warhead_type": kind, "smiles": smi,
                "MW": Descriptors.MolWt(mol),
                "pIC50_after_projection": pIC50_proj,
                "before_d_S_Cb": before["d_S_Cb"],
                "after_d_S_Cb":  after["d_S_Cb"],
                "before_angle":  before["angle_S_Cb_Ca"],
                "after_angle":   after["angle_S_Cb_Ca"],
                "before_phi":    before["dihedral_S_Cb_Ca_Ccarb"],
                "after_phi":     after["dihedral_S_Cb_Ca_Ccarb"],
            })
        w.close()
        if not cohort_rows: continue
        df = pd.DataFrame(cohort_rows)
        print(f"\n=== {cohort} ({kind}) ===")
        print(f"  projected: {n} mols → {out_proj.name}")
        print(f"  geometry (before → after):")
        print(f"    d_S_Cb:     [{df.before_d_S_Cb.min():.2f}, {df.before_d_S_Cb.max():.2f}] "
              f"→ [{df.after_d_S_Cb.min():.3f}, {df.after_d_S_Cb.max():.3f}]")
        print(f"    angle:      [{df.before_angle.min():.1f}, {df.before_angle.max():.1f}] "
              f"→ [{df.after_angle.min():.2f}, {df.after_angle.max():.2f}]")
        print(f"    |phi|max:   {df.before_phi.abs().max():.1f}° → {df.after_phi.abs().max():.2f}°")
        print(f"  pIC50_after_projection: med={df['pIC50_after_projection'].median():.2f}  "
              f"max={df['pIC50_after_projection'].max():.2f}")
        rows.extend(cohort_rows)
        summary.append({
            "cohort": cohort, "warhead_type": kind, "n_projected": n,
            "med_pIC50_after_proj": df["pIC50_after_projection"].median(),
            "max_pIC50_after_proj": df["pIC50_after_projection"].max(),
            "before_d_med": df["before_d_S_Cb"].median(),
            "before_angle_med": df["before_angle"].median(),
            "before_phi_absmed": df["before_phi"].abs().median(),
        })

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "ablation_projected.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "ablation_projection_summary.csv", index=False)
    print(f"\nwrote ablation_projected.csv + ablation_projection_summary.csv")

    # Compare pre-projection (from ablation_scored_clean) vs post-projection
    pre = pd.read_csv(out_dir / "ablation_scored_clean.csv")
    print("\n=== PROJECTION GAIN (post − pre median pIC50 per cohort) ===")
    print(f"{'cohort':30s}  {'pre_med':>8}  {'post_med':>9}  {'gain':>6}")
    for c in summary:
        pre_med = pre[pre["cohort"] == c["cohort"]]["pIC50_clean"].median()
        post_med = c["med_pIC50_after_proj"]
        print(f"{c['cohort']:30s}  {pre_med:>8.3f}  {post_med:>9.3f}  {post_med-pre_med:>+6.3f}")


if __name__ == "__main__":
    main()
