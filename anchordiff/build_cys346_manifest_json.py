"""Build top1000_manifest__zap70_cys346.json from the Cys346 cofold leaderboard
+ compute mPAE per molecule from the PAE NPZ files.

The Flask backend at port 5001 expects this JSON to populate the Cys346 cofold
columns + the 3D viewer. Until this JSON exists, all Boltz columns are NaN
and the 3D viewer has nothing to show.

Output:
  data/boltz_poses/top1000_manifest__zap70_cys346.json
    {
      "<row_id>": {
        "row_id": int,
        "yaml_name": str,
        "smiles": str,
        "MW": float,
        "method": str,
        "mPAE": float (min PAE between any ligand atom and any protein residue),
        "iptm": float,
        "ligand_iptm": float,
        "complex_plddt": float,
        "complex_pde": float,
        "confidence_score": float,
        "combined_score": float (FiLMDelta pIC50 percentile × 0.6 + lig-iptm percentile × 0.4),
        "rank_score": float (FiLMDelta pIC50)
      }, ...
    }
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv"
PRED_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
OUT_JSON = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"


def parse_cif_for_chain_split(cif_path: Path):
    """Count atoms in chain A (protein) and chain B (ligand) from the CIF."""
    lines = open(cif_path).read().splitlines()
    in_loop = False
    cols = []
    n_A = n_B = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols_local = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols_local.append(lines[j].strip().removeprefix("_atom_site."))
                j += 1
            if cols_local:
                cols = cols_local
                try:
                    chain_idx = cols.index("auth_asym_id")
                except ValueError:
                    chain_idx = cols.index("label_asym_id")
                i = j
                while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) == len(cols):
                        if parts[chain_idx] == "A":
                            n_A += 1
                        elif parts[chain_idx] == "B":
                            n_B += 1
                    i += 1
                continue
        i += 1
    return n_A, n_B


def mPAE_from_npz(pae_npz: Path, n_protein: int, n_ligand: int) -> float | None:
    """Compute mPAE = min PAE between any protein residue and any ligand atom.
    Boltz writes pae as a square matrix in token order (atom-level for ligand,
    residue-level for protein).
    """
    if not pae_npz.exists():
        return None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None
    if pae.ndim != 2:
        return None
    N = pae.shape[0]
    # Boltz PAE indexing: tokens 0..N-1 where the LAST n_ligand tokens are
    # the ligand atoms. Take min of the cross-block PAE[protein, ligand].
    lig_lo = N - n_ligand
    if lig_lo <= 0 or lig_lo >= N:
        # fallback: assume first chunk is protein, second is ligand
        return float(np.min(pae))
    cross = pae[:lig_lo, lig_lo:]
    if cross.size == 0:
        return None
    return float(np.min(cross))


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["boltz_iptm"]).copy()
    print(f"loaded {len(df)} cofolded rows")

    # Combined score (percentile based, robust to scale)
    df["combined_score"] = df["rank_score"].rank(pct=True) * 0.6 + \
                           df["boltz_ligand_iptm"].rank(pct=True) * 0.4

    out = {}
    n_with_mpae = 0
    for _, r in df.iterrows():
        name = r["yaml_name"]
        pred_subdir = PRED_DIR / name
        cif = pred_subdir / f"{name}_model_0.cif"
        pae = pred_subdir / f"pae_{name}_model_0.npz"

        # mPAE — best-effort. If CIF missing skip; if PAE missing leave None.
        mpae = None
        if cif.exists() and pae.exists():
            try:
                n_A, n_B = parse_cif_for_chain_split(cif)
                if n_A > 0 and n_B > 0:
                    mpae = mPAE_from_npz(pae, n_A, n_B)
                    if mpae is not None:
                        n_with_mpae += 1
            except Exception:
                pass

        out[str(int(r["row_id"]))] = {
            "row_id": int(r["row_id"]),
            "yaml_name": name,
            "smiles": r["smiles"],
            "MW": float(r["MW"]) if pd.notna(r["MW"]) else None,
            "method": r["method"],
            "mPAE": mpae,
            "iptm": float(r["boltz_iptm"]),
            "ligand_iptm": float(r["boltz_ligand_iptm"]),
            "complex_plddt": float(r["boltz_complex_plddt"]),
            "complex_pde": float(r["boltz_complex_pde"]),
            "confidence_score": float(r["boltz_confidence_score"]),
            "combined_score": float(r["combined_score"]),
            "rank_score": float(r["rank_score"]),
            "Tc_to_Mol1": float(r["Tc_to_Mol1"]) if pd.notna(r["Tc_to_Mol1"]) else None,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}  ({len(out)} rows, {n_with_mpae} with mPAE)")


if __name__ == "__main__":
    main()
