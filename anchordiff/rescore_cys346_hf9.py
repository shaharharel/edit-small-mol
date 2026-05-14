"""HF9 — rebuild top1000_manifest__zap70_cys346.json with London's mPAE metric.

Changes vs build_cys346_manifest_json.py:
  1. combined_score = α · z(FiLMDelta_pIC50) + (1−α) · (−z(mPAE))     α = 0.4
     (LOWER mPAE is better — London JACS 2026 Fig 2C). We drop lig-iPTM
     entirely; the paper showed it has poor ranking resolution.
  2. Add d_SG ∈ [1.5, 2.5] Å filter: 50% of existing 997 poses have a
     broken covalent bond (Boltz-2 soft-constraint quirk). They get
     `combined_score = None` so the report can exclude them or mark them.
  3. FiLMDelta source: the rank_score column in the CSV is from the
     LEAKY (pair-row-val) model. If `reinvent4_film_model_clean.pt` is
     available, we recompute pIC50 with the clean ensemble.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, math, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

CSV = PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv"
PRED_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
OUT_JSON = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
CLEAN_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model_clean.pt"

ALPHA = 0.4  # FiLMDelta weight; (1-ALPHA) = mPAE weight
D_SG_MIN, D_SG_MAX = 1.5, 2.5  # Å — bond-geometry filter


def parse_cif_chain_split(cif_path: Path):
    lines = open(cif_path).read().splitlines()
    cols, n_A, n_B = [], 0, 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols_local = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols_local.append(lines[j].strip().removeprefix("_atom_site.")); j += 1
            if cols_local:
                cols = cols_local
                try: chain_idx = cols.index("auth_asym_id")
                except ValueError: chain_idx = cols.index("label_asym_id")
                i = j
                while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) == len(cols):
                        if parts[chain_idx] == "A": n_A += 1
                        elif parts[chain_idx] == "B": n_B += 1
                    i += 1
                continue
        i += 1
    return n_A, n_B


def mpae_from_npz(pae_npz: Path, n_ligand: int) -> float | None:
    if not pae_npz.exists(): return None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception: return None
    if pae.ndim != 2: return None
    N = pae.shape[0]; lig_lo = N - n_ligand
    if lig_lo <= 0 or lig_lo >= N: return float(np.min(pae))
    cross = pae[:lig_lo, lig_lo:]
    return float(np.min(cross)) if cross.size else None


def sg_anchor_distance(cif_path: Path, cys_resi: int, warhead_atom: str | None):
    """Read SG and the named ligand atom (warhead Cβ) from the CIF and return distance.
    Strategy: get SG first; then if warhead_atom is given and present, use it.
    Else take the NEAREST C atom in chain B to SG (correct for acrylamide adducts)."""
    import gemmi
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception: return None
    sg = None
    # Pass 1: SG
    for chain in st[0]:
        if chain.name != "A": continue
        for res in chain:
            if res.seqid.num == cys_resi and res.name == "CYS":
                for atom in res:
                    if atom.name == "SG": sg = atom.pos
    if sg is None: return None
    # Pass 2: ligand anchor
    target_atom = None
    nearest_d = 1e9; nearest_atom = None
    for chain in st[0]:
        if chain.name != "B": continue
        for res in chain:
            for atom in res:
                if warhead_atom is not None and atom.name == warhead_atom:
                    target_atom = atom.pos
                if atom.element.name == "C":
                    d = sg.dist(atom.pos)
                    if d < nearest_d:
                        nearest_d, nearest_atom = d, atom.pos
    if target_atom is not None:
        return float(sg.dist(target_atom))
    if nearest_atom is not None:
        return float(nearest_d)
    return None


def load_clean_filmdelta():
    if not CLEAN_CKPT.exists(): return None
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
    ck = torch.load(CLEAN_CKPT, map_location="cpu", weights_only=False)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    m.load_state_dict(ck["model_state"]); m.eval()
    sc = StandardScaler()
    sc.mean_ = ck["scaler_mean"]; sc.scale_ = ck["scaler_scale"]; sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = len(sc.mean_)
    return m, sc, ck["anchor_embs"], np.asarray(ck["anchor_pIC50"])


def score_pic50_clean(smi, scorer):
    m, sc, ae, ap = scorer
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return float('nan')
    a = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), a)
    e = torch.FloatTensor(sc.transform(a[None, :]).astype(np.float32))
    with torch.no_grad():
        d = m(ae, e.expand(len(ap), -1)).numpy().flatten()
    return float(np.mean(ap + d))


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["boltz_iptm"]).copy()
    print(f"loaded {len(df)} cofolded rows")
    print(f"Recomputing pIC50 with clean FiLMDelta ckpt: {CLEAN_CKPT.exists()}")
    scorer = load_clean_filmdelta()
    if scorer:
        df["pIC50_clean"] = df["smiles"].apply(lambda s: score_pic50_clean(s, scorer))
        print(f"  pIC50_clean range: {df.pIC50_clean.min():.2f} … {df.pIC50_clean.max():.2f}")
    else:
        df["pIC50_clean"] = df["rank_score"]
        print(f"  (clean ckpt missing — falling back to leaky rank_score)")

    # Need warhead atom name per row. Manifest CSV has it; read.
    MAN = PROJECT_ROOT / "experiments" / "boltz_inputs" / "top1000__zap70_cys346" / "manifest.csv"
    if MAN.exists():
        mdf = pd.read_csv(MAN, usecols=['yaml_name', 'warhead_atom_name'])
        df = df.merge(mdf, on='yaml_name', how='left')
    else:
        df["warhead_atom_name"] = None

    # Compute mPAE + d_SG per row
    print("Parsing CIFs + PAE NPZ for mPAE + d_SG …")
    mpae_list = []; d_list = []; n_mpae = 0; n_d_ok = 0
    for _, r in df.iterrows():
        name = r["yaml_name"]
        cif = PRED_DIR / name / f"{name}_model_0.cif"
        pae = PRED_DIR / name / f"pae_{name}_model_0.npz"
        # d_SG
        d = None
        if cif.exists():
            d = sg_anchor_distance(cif, cys_resi=346, warhead_atom=r.get('warhead_atom_name'))
            if d is not None and D_SG_MIN <= d <= D_SG_MAX:
                n_d_ok += 1
        d_list.append(d)
        # mPAE
        mpae = None
        if cif.exists() and pae.exists():
            try:
                _, n_B = parse_cif_chain_split(cif)
                if n_B > 0:
                    mpae = mpae_from_npz(pae, n_B)
                    if mpae is not None: n_mpae += 1
            except Exception: pass
        mpae_list.append(mpae)
    df["mPAE"] = mpae_list
    df["d_SG"] = d_list
    df["geom_ok"] = df["d_SG"].between(D_SG_MIN, D_SG_MAX, inclusive='both')
    print(f"  mPAE computed for {n_mpae}/{len(df)}")
    print(f"  d_SG in [{D_SG_MIN},{D_SG_MAX}] Å: {n_d_ok}/{len(df)}  ({100 * n_d_ok / len(df):.0f}%)")

    # Combined score (only on geom-ok rows; others get None)
    mask = df["geom_ok"] & df["mPAE"].notna() & df["pIC50_clean"].notna()
    sub = df.loc[mask].copy()
    z_pic = (sub["pIC50_clean"] - sub["pIC50_clean"].mean()) / (sub["pIC50_clean"].std() + 1e-9)
    z_mpae = (sub["mPAE"] - sub["mPAE"].mean()) / (sub["mPAE"].std() + 1e-9)
    sub["combined_score"] = ALPHA * z_pic - (1 - ALPHA) * z_mpae  # mPAE inverted
    df["combined_score"] = np.nan
    df.loc[sub.index, "combined_score"] = sub["combined_score"]
    n_scored = int(sub.shape[0])
    print(f"  combined_score computed for {n_scored}/{len(df)} (others have broken geom or missing mPAE)")

    out = {}
    for _, r in df.iterrows():
        out[str(int(r["row_id"]))] = {
            "row_id": int(r["row_id"]),
            "yaml_name": r["yaml_name"],
            "smiles": r["smiles"],
            "MW": float(r["MW"]) if pd.notna(r["MW"]) else None,
            "method": r["method"],
            "mPAE": float(r["mPAE"]) if pd.notna(r["mPAE"]) else None,
            "iptm": float(r["boltz_iptm"]),
            "ligand_iptm": float(r["boltz_ligand_iptm"]),
            "complex_plddt": float(r["boltz_complex_plddt"]),
            "complex_pde": float(r["boltz_complex_pde"]),
            "confidence_score": float(r["boltz_confidence_score"]),
            "combined_score": float(r["combined_score"]) if pd.notna(r["combined_score"]) else None,
            "rank_score": float(r["pIC50_clean"]) if pd.notna(r["pIC50_clean"]) else None,
            "rank_score_legacy_leaky": float(r["rank_score"]),
            "d_SG": float(r["d_SG"]) if pd.notna(r["d_SG"]) else None,
            "geom_ok": bool(r["geom_ok"]),
            "warhead_atom_name": r.get("warhead_atom_name"),
            "Tc_to_Mol1": float(r["Tc_to_Mol1"]) if pd.notna(r["Tc_to_Mol1"]) else None,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}  ({len(out)} rows, {n_scored} with combined_score)")
    print(f"Top-10 by NEW combined_score (FiLMDelta_clean + mPAE, geom-filtered):")
    top = df.loc[df["combined_score"].notna()].nlargest(10, "combined_score")[
        ["yaml_name", "pIC50_clean", "mPAE", "d_SG", "combined_score"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
