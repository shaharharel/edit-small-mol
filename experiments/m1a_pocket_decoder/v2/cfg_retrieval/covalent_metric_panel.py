"""Covalent metric panel for CFG × retrieval samples.

Reads all 8 (or 9 including baseline) samples_*.csv files under
`data/paper_pair_training/cfg_retrieval/` and produces
`covalent_metric_panel.csv` with per-sample metrics:

  - cell, cfg_scale, retrieval, sample_idx, raw_SMILES, largest_frag_SMILES
  - valid : bool (parses as a molecule with >=1 heavy atom)
  - hac   : heavy atom count of largest fragment
  - qed   : QED score
  - tc_mol1 : Morgan(r=2, 2048) Tanimoto to Mol1 canonical SMILES
  - acryl_largest : bool (acrylamide [CH2]=[CH]-C(=O)-N SMARTS on largest fragment)
  - planar_dihedral_deg : ETKDG single-conformer C=C-C(=O)-N torsion,
                            wrapped to [0,90]. NaN if not applicable.
  - fukui_fplus_proxy : signed Gasteiger charge on the β-C (Cβ = CH2 of the
                          acrylamide). NaN if not applicable. Higher = more
                          electrophilic. This is a proxy — the real Fukui f+
                          would require an xTB single-point + FMO analysis
                          which is not available on this machine. This column
                          is the strongest RDKit-only surrogate.
  - vina_cov_score : Vina covalent-tethered docking score (kcal/mol; lower is
                       better). Only computed for the top-`--vina_topn` valid,
                       BD-ready samples per cell to keep wall clock < 12h.
                       Uses experiments/run_covalent_docking.py's
                       build_cov_tethered_pdbqt + vina_dock. NaN otherwise.
  - vina_d_sg, vina_bd_angle : geometric quality of the docked pose. NaN if
                                  vina_cov_score is NaN.

We do NOT compute:
  - xTB k_inact (pred_log_k2_GSH): xTB is not installed on ai-gpu.
  - Boltz-2 cofolds per sample: 20+ min each; 1600 samples => infeasible.
    A separate Boltz cofold worker can consume this CSV to enrich later.

Writes incrementally per cell so preemption is safe.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # scorer runs CPU
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")


def canon(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(m)


def largest_frag(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None, None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None, None
    lf = max(frags, key=lambda mm: mm.GetNumHeavyAtoms())
    return lf, Chem.MolToSmiles(lf)


def morgan_bv(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def compute_planar_and_fukui_proxy(lf_mol, seed=42):
    """Return (dihedral_deg_wrapped, fplus_proxy_Cbeta)."""
    if lf_mol is None:
        return np.nan, np.nan
    matches = lf_mol.GetSubstructMatches(ACRYL_SMARTS)
    if not matches:
        return np.nan, np.nan
    a_b, a_ca, a_co, _o, a_n = matches[0]
    dih = np.nan
    try:
        mH = Chem.AddHs(lf_mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        cid = AllChem.EmbedMolecule(mH, params)
        if cid >= 0:
            conf = mH.GetConformer(cid)
            phi = AllChem.GetDihedralDeg(conf, a_b, a_ca, a_co, a_n)
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            dih = float(min(abs(phi_wrap), abs(180.0 - abs(phi_wrap))))
    except Exception:
        pass
    # Fukui f+ proxy: signed Gasteiger charge on the β-C.  In practice, more
    # positive charge => more electrophilic C=C => higher Fukui f+.  This is
    # a rough approximation, NOT a real Fukui function.  Real Fukui requires
    # xTB single-point + FMO or a DFT calc.
    fplus = np.nan
    try:
        m2 = Chem.Mol(lf_mol)
        AllChem.ComputeGasteigerCharges(m2)
        atom = m2.GetAtomWithIdx(a_b)
        q = atom.GetDoubleProp("_GasteigerCharge")
        if np.isfinite(q):
            fplus = float(q)
    except Exception:
        pass
    return dih, fplus


def score_one_sample(rec, mol1_bv):
    raw = rec["SMILES"]
    if not isinstance(raw, str) or len(raw) < 2:
        return dict(rec, valid=False)
    m = Chem.MolFromSmiles(raw)
    if m is None or m.GetNumHeavyAtoms() == 0:
        return dict(rec, valid=False)
    lf_mol, lf_smi = largest_frag(raw)
    if lf_mol is None:
        return dict(rec, valid=False)
    hac = lf_mol.GetNumHeavyAtoms()
    try:
        q_val = QED.qed(lf_mol)
    except Exception:
        q_val = np.nan
    fp = AllChem.GetMorganFingerprintAsBitVect(lf_mol, 2, nBits=2048)
    tc = DataStructs.TanimotoSimilarity(fp, mol1_bv)
    acryl = bool(lf_mol.GetSubstructMatch(ACRYL_SMARTS))
    dih, fplus = (np.nan, np.nan)
    if acryl:
        dih, fplus = compute_planar_and_fukui_proxy(lf_mol)
    return {
        **rec,
        "valid": True,
        "canonical_SMILES": Chem.MolToSmiles(m),
        "largest_frag_SMILES": lf_smi,
        "hac": int(hac),
        "qed": float(q_val),
        "tc_mol1": float(tc),
        "acryl_largest": acryl,
        "planar_dihedral_deg": float(dih) if np.isfinite(dih) else np.nan,
        "fukui_fplus_proxy": float(fplus) if np.isfinite(fplus) else np.nan,
    }


def run_vina_for_topn(rows_df, top_n, receptor_pdbqt, box_center, box_size,
                        vina_bin, n_workers=6):
    """Run Vina covalent docking on the top_n valid+BD-ready+lowest-NLL rows
    per cell.  Returns updated df with vina_cov_score / vina_d_sg /
    vina_bd_angle columns filled where docking succeeded.

    Requires the run_covalent_docking module.  We import lazily so the panel
    still writes CSV outputs if Vina is broken.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from experiments.run_covalent_docking import (
            build_cov_tethered_pdbqt, vina_dock, read_tethered_geom,
        )
    except Exception as e:
        print(f"[vina] import failed: {e}; skipping Vina", flush=True)
        return rows_df

    rows_df["vina_cov_score"] = np.nan
    rows_df["vina_d_sg"] = np.nan
    rows_df["vina_bd_angle"] = np.nan

    # Rank per cell: BD-ready samples (acryl_largest & planar_dihedral <=30)
    # first; among those, lowest NLL is highest priority.  Then acryl-only.
    for cell, cell_df in rows_df.groupby("cell"):
        pool = cell_df[cell_df["valid"] & cell_df["acryl_largest"]].copy()
        pool["bd_ready"] = (pool["planar_dihedral_deg"] <= 30.0).astype(int)
        pool = pool.sort_values(["bd_ready", "NLL"],
                                     ascending=[False, True]).head(top_n)
        print(f"[vina] cell={cell}: docking {len(pool)} samples", flush=True)
        for row_ix in pool.index:
            smi = rows_df.at[row_ix, "largest_frag_SMILES"]
            try:
                # Build tethered PDBQT.
                res = build_cov_tethered_pdbqt(smi)
                if res is None:
                    continue
                lig_pdbqt = res.get("pdbqt")
                if lig_pdbqt is None or not Path(lig_pdbqt).exists():
                    continue
                score, pose = vina_dock(lig_pdbqt, receptor_pdbqt,
                                             box_center, box_size,
                                             exhaustiveness=4)
                if score is None:
                    continue
                geom = read_tethered_geom(pose) if pose is not None else None
                rows_df.at[row_ix, "vina_cov_score"] = float(score)
                if geom:
                    rows_df.at[row_ix, "vina_d_sg"] = float(geom.get("d_sg", np.nan))
                    rows_df.at[row_ix, "vina_bd_angle"] = float(
                        geom.get("bd_angle", np.nan))
            except Exception as e:
                print(f"[vina] {smi[:40]}...: {type(e).__name__}: {e}",
                       flush=True)
    return rows_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/covalent_metric_panel.csv"))
    ap.add_argument("--vina_topn", type=int, default=0,
                    help="Per cell, how many top-BD-ready samples to Vina-dock. "
                          "Default 0 = skip Vina.  Set 50 for full experiment.")
    ap.add_argument("--limit_per_cell", type=int, default=None,
                    help="Limit rows per cell (debug).")
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    out_csv = Path(args.out_csv)
    progress_path = samples_dir / "progress.json"

    mol1_m = Chem.MolFromSmiles(MOL1_SMI)
    mol1_bv = AllChem.GetMorganFingerprintAsBitVect(mol1_m, 2, nBits=2048)

    all_rows = []
    files = sorted(samples_dir.glob("samples_*.csv"))
    print(f"Scoring {len(files)} sample files:", flush=True)
    for f in files:
        print(f"  - {f.name}", flush=True)
    for fp in files:
        df = pd.read_csv(fp)
        if args.limit_per_cell is not None:
            df = df.head(args.limit_per_cell)
        # Ensure required cols
        for c in ("cell", "cfg_scale", "retrieval", "sample_idx",
                    "SMILES", "Input_SMILES", "NLL"):
            if c not in df.columns:
                df[c] = None
        t0 = time.time()
        scored = []
        for i, rec in enumerate(df.to_dict(orient="records")):
            scored.append(score_one_sample(rec, mol1_bv))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"    {fp.name}  {i+1}/{len(df)}  "
                       f"{(i+1)/el:.1f} rec/s", flush=True)
                progress_path.write_text(json.dumps({
                    "phase": "scoring", "file": fp.name,
                    "done": i + 1, "total": len(df),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, indent=2))
        all_rows.extend(scored)
        print(f"  done {fp.name} in {time.time()-t0:.1f}s "
               f"({len(df)} rows)", flush=True)

    rows_df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(rows_df)}", flush=True)
    print(f"Valid fraction: {rows_df['valid'].mean():.3f}", flush=True)
    if "acryl_largest" in rows_df.columns:
        print(f"Acryl-largest fraction (of valid): "
               f"{rows_df.loc[rows_df['valid'], 'acryl_largest'].mean():.3f}",
               flush=True)

    if args.vina_topn > 0:
        print(f"\n[vina] docking top {args.vina_topn} per cell (BD-ready first)",
               flush=True)
        RECEPTOR_PDBQT = str(PROJECT_ROOT / "data/docking_500/"
                                "receptor_cys346_stripped.pdbqt")
        BOX_CENTER = np.array([18.888, -3.650, -29.979])
        BOX_SIZE = np.array([20.0, 20.0, 20.0])
        rows_df = run_vina_for_topn(rows_df, args.vina_topn,
                                        RECEPTOR_PDBQT, BOX_CENTER, BOX_SIZE,
                                        vina_bin=str(PROJECT_ROOT /
                                                       "tools/vina"))

    rows_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)

    progress_path.write_text(json.dumps({
        "phase": "scoring_done",
        "total_rows": int(len(rows_df)),
        "valid_frac": float(rows_df["valid"].mean()),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
