#!/usr/bin/env python3
"""Build labeled_mols.npz for the mpae-geometry experiment.

Reads:
    data/covalid_mv_cofolds/warhead_metrics.csv   (1,479 rows w/ mpae_warhead_cys)
    data/covalid_mv_cofolds/cifs/<sub>/<TARGET>/boltz_results_<name>/predictions/<name>/
        <name>_model_0.lig.sdf    (ligand 3D coords → SMILES)
        <name>_model_0.pdb        (protein w/ target Cys — used to locate S atom)

Extracts per row:
    smiles         : canonical SMILES from the ligand SDF
    d_b_nuc        : distance Cβ ↔ target Cys Sγ (Å)
    bd_angle_deg   : ∠S-Cβ-Cα Bürgi-Dunitz angle (°)
    planar_dihedral: Cβ=Cα-C(=O)-N dihedral (°)
    mpae_warhead_cys: from CSV
    q_bin          : quartile bin (good=Q1 best-mpae, mid=Q2Q3, bad=Q4 worst-mpae)
    target         : kinase (BMX/BTK/EGFR/FGFR1/FGFR4_477/FGFR4_552/ITK/JAK3)

Filters:
    mpae_warhead_cys finite
    Passes acrylamide SMARTS
    MW 200-800

Outputs to <out_dir>/labeled_mols.npz and labeled_mols.csv.
"""
from __future__ import annotations
import argparse
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolTransforms

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")

# Acrylamide SMARTS: β-C=α-C-C(=O)-N
ACRYL_PATTERNS = [
    ("acrylamide_strict", "[CH2;X3]=[CH;X3][C;X3](=O)[N]",   (0, 1, 2, 4)),
    ("acrylamide_loose",  "[CH2]=C[C](=O)[N,n]",             (0, 1, 2, 4)),
]

# Boltz name → target dir mapping
TARGET_DIRS = {
    "BMX":       "covalid_mv_cofolds/BMX",
    "BTK":       "covalid_mv_cofolds/BTK",
    "EGFR":      "covalid_mv_cofolds/EGFR",
    "FGFR1":     "covalid_mv_cofolds/FGFR1",
    "FGFR4_477": "covalid_mv_cofolds/FGFR4_477",
    "FGFR4_552": "covalid_mv_cofolds/FGFR4_552",
    "ITK":       "covalid_mv_cofolds/ITK",
    "JAK3":      "covalid_mv_cofolds/JAK3",
}


def read_sdf_first(sdf_path: Path) -> Chem.Mol | None:
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
        for m in suppl:
            if m is not None:
                return m
    except Exception:
        pass
    return None


def match_acryl(mol: Chem.Mol) -> tuple[int, int, int, int] | None:
    """Return (b_idx, a_idx, gamma_C_idx, N_idx) for the FIRST acryl match."""
    for name, sma, order in ACRYL_PATTERNS:
        pat = Chem.MolFromSmarts(sma)
        matches = mol.GetSubstructMatches(pat)
        if matches:
            m0 = matches[0]
            return (m0[order[0]], m0[order[1]], m0[order[2]], m0[order[3]])
    return None


def parse_cys_s_from_pdb(pdb_path: Path, cys_res: int) -> np.ndarray | None:
    """Extract the SG atom XYZ from a protein PDB for the target Cys."""
    try:
        with open(pdb_path) as f:
            for ln in f:
                if not ln.startswith("ATOM"):
                    continue
                # PDB fixed columns
                atom = ln[12:16].strip()
                resname = ln[17:20].strip()
                try:
                    resnum = int(ln[22:26])
                except Exception:
                    continue
                if resname == "CYS" and atom == "SG" and resnum == cys_res:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                    return np.array([x, y, z], dtype=np.float64)
    except Exception:
        return None
    return None


def compute_pose(mol: Chem.Mol, s_xyz: np.ndarray,
                  acryl_atoms: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    """Return (d_b_nuc, bd_angle_deg, planar_dihedral_deg)."""
    conf = mol.GetConformer(0)
    b_idx, a_idx, g_idx, n_idx = acryl_atoms
    def xyz(i):
        p = conf.GetAtomPosition(i)
        return np.array([p.x, p.y, p.z], dtype=np.float64)
    b = xyz(b_idx); a = xyz(a_idx); g = xyz(g_idx); n = xyz(n_idx)
    # d_b_nuc = |β − Sγ|
    d = float(np.linalg.norm(b - s_xyz))
    # Bürgi-Dunitz angle ∠S-Cβ-Cα (deg)
    v1 = s_xyz - b; v2 = a - b
    cos_bd = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    bd = float(np.degrees(np.arccos(np.clip(cos_bd, -1.0, 1.0))))
    # Vinyl-amide planar dihedral Cβ=Cα-C(=O)-N
    # Standard 4-atom dihedral formula
    b1 = a - b; b2 = g - a; b3 = n - g
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-9))
    x = float(np.dot(n1, n2)); y = float(np.dot(m1, n2))
    phi = float(np.degrees(np.arctan2(y, x)))
    return (d, bd, phi)


def find_boltz_dir_for_name(name: str) -> tuple[Path, str] | None:
    """Locate the Boltz output directory + target for a row `name`."""
    # covalid_mv_cofolds names look like BMX_act_00001, EGFR_dec_00023, FGFR4_477_...
    # bmx_stratified subset: BMX_dec_strat_00130
    # Target key is prefix before _act/_dec/_dec_strat
    for tgt, subdir in TARGET_DIRS.items():
        if name.startswith(tgt + "_"):
            root = PROJECT_ROOT / "data/covalid_mv_cofolds/cifs" / subdir / f"boltz_results_{name}" / "predictions" / name
            if root.exists():
                return root, tgt
    # Try bmx_stratified
    if name.startswith("BMX_dec_strat_"):
        root = PROJECT_ROOT / "data/covalid_mv_cofolds/cifs/bmx_stratified_cofolds/BMX" / f"boltz_results_{name}" / "predictions" / name
        if root.exists():
            return root, "BMX"
    return None


def process_row(row: pd.Series) -> dict | None:
    name = row["name"]
    cys_res = int(row["cys_res"])
    mpae = float(row["mpae_warhead_cys"])
    if not np.isfinite(mpae):
        return None
    loc = find_boltz_dir_for_name(name)
    if loc is None:
        return None
    boltz_dir, target = loc
    sdf = boltz_dir / f"{name}_model_0.lig.sdf"
    pdb = boltz_dir / f"{name}_model_0.prot.pdb"
    if not sdf.exists() or not pdb.exists():
        return None
    mol = read_sdf_first(sdf)
    if mol is None:
        return None
    # SMILES + filters
    try:
        smi = Chem.MolToSmiles(mol)
        mol_h = Chem.MolFromSmiles(smi)
        if mol_h is None:
            return None
        mw = Descriptors.MolWt(mol_h)
    except Exception:
        return None
    if mw < 200 or mw > 800:
        return None
    acryl = match_acryl(mol)
    if acryl is None:
        return None
    s_xyz = parse_cys_s_from_pdb(pdb, cys_res)
    if s_xyz is None:
        return None
    pose = compute_pose(mol, s_xyz, acryl)
    if pose is None:
        return None
    return {
        "name": name,
        "target": target,
        "smiles": smi,
        "mw": float(mw),
        "d_b_nuc": pose[0],
        "bd_angle_deg": pose[1],
        "planar_dihedral_deg": pose[2],
        "mpae_warhead_cys": mpae,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warhead_csv", default=str(PROJECT_ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"))
    ap.add_argument("--out_dir",     default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.warhead_csv)
    print(f"[in] {len(df)} rows from warhead_metrics.csv")
    df = df.dropna(subset=["mpae_warhead_cys", "cys_res"])
    print(f"[filt] {len(df)} rows with finite mpae + cys_res")

    records = []
    skipped: Counter[str] = Counter()
    for i, row in df.iterrows():
        rec = process_row(row)
        if rec is None:
            skipped["failed"] += 1
        else:
            records.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(df)}  kept={len(records)}  skipped={sum(skipped.values())}")

    print(f"[done] kept={len(records)} skipped={sum(skipped.values())}")
    out = pd.DataFrame.from_records(records)
    # Quartile binning
    q25 = out["mpae_warhead_cys"].quantile(0.25)
    q75 = out["mpae_warhead_cys"].quantile(0.75)
    print(f"[quartiles] q25={q25:.3f}  q75={q75:.3f}  median={out['mpae_warhead_cys'].median():.3f}")

    def qbin(v: float) -> str:
        if v <= q25: return "good"
        if v >= q75: return "bad"
        return "mid"
    out["q_bin"] = out["mpae_warhead_cys"].apply(qbin)
    print(f"[q_bin counts] {dict(out['q_bin'].value_counts())}")
    print(f"[target counts] {dict(out['target'].value_counts())}")

    out.to_csv(out_dir / "labeled_mols.csv", index=False)
    print(f"[write] {out_dir / 'labeled_mols.csv'}  ({len(out)} rows)")

    pose_boltz = out[["d_b_nuc", "bd_angle_deg", "planar_dihedral_deg"]].to_numpy(dtype=np.float32)
    smi_arr = out["smiles"].to_numpy()
    mpae_arr = out["mpae_warhead_cys"].to_numpy(dtype=np.float32)
    qbin_arr = out["q_bin"].to_numpy()
    target_arr = out["target"].to_numpy()
    np.savez_compressed(
        out_dir / "labeled_mols.npz",
        smiles=smi_arr,
        pose_boltz=pose_boltz,
        mpae_warhead_cys=mpae_arr,
        q_bin=qbin_arr,
        target=target_arr,
    )
    print(f"[write] {out_dir / 'labeled_mols.npz'}")

    # Pose stats for downstream normalization
    stats = {
        "pose_mean": pose_boltz.mean(axis=0).tolist(),
        "pose_std":  pose_boltz.std(axis=0).tolist(),
        "pose_names": ["d_b_nuc", "bd_angle_deg", "planar_dihedral_deg"],
        "N": len(out),
        "quartiles": {"q25": float(q25), "q75": float(q75)},
        "q_bin_counts": {k: int(v) for k, v in out["q_bin"].value_counts().items()},
        "target_counts": {k: int(v) for k, v in out["target"].value_counts().items()},
    }
    import json
    (out_dir / "pose_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[write] {out_dir / 'pose_stats.json'}")


if __name__ == "__main__":
    main()
