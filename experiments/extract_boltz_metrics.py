"""Extract mPAE + confidence metrics from a Boltz cofold result tree.

Produces a CSV with the same column subset used by `run_bmx_stratified_d3_with_energy.py`
and `run_covalid_d3_london_mpae.py`:

  target, name, n_prot, n_lig, n_pae,
  complex_iptm, ligand_iptm, complex_plddt, complex_ipde,
  mpae_london_min, mpae_prot_lig_mean

Walks <root>/<target>/boltz_results_<name>/predictions/<name>/{*_model_0.cif, pae_*_model_0.npz, confidence_*_model_0.json}.

Usage:
  python extract_boltz_metrics.py <cofold_root> <output_csv>
  python extract_boltz_metrics.py data/covalid_mv_cofolds/cifs/bmx_stratified_cofolds \
      data/covalid_mv_cofolds/warhead_metrics_bmx_strat.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def chain_split(cif_path: Path) -> tuple[int, int]:
    """Return (n_protein_atoms_chainA, n_ligand_atoms_chainB) from CIF atom_site loop."""
    lines = cif_path.read_text().splitlines()
    n_A = n_B = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().removeprefix("_atom_site."))
                j += 1
            if cols:
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


def mpae_metrics(pae_npz: Path, n_ligand: int) -> tuple[float | None, float | None, int | None]:
    """Return (london_min, mean_cross, n_pae) — min and mean over protein × ligand block."""
    if not pae_npz.exists():
        return None, None, None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None, None, None
    if pae.ndim != 2:
        return None, None, None
    N = pae.shape[0]
    lig_lo = N - n_ligand
    if lig_lo <= 0 or lig_lo >= N:
        return float(np.min(pae)), float(np.mean(pae)), N
    cross = pae[:lig_lo, lig_lo:]
    if cross.size == 0:
        return None, None, N
    return float(np.min(cross)), float(np.mean(cross)), N


def conf_metrics(conf_json: Path) -> dict:
    if not conf_json.exists():
        return {}
    try:
        d = json.loads(conf_json.read_text())
    except Exception:
        return {}
    return {
        "complex_iptm": d.get("iptm") or d.get("complex_iptm"),
        "ligand_iptm": d.get("ligand_iptm"),
        "complex_plddt": d.get("complex_plddt"),
        "complex_ipde": d.get("complex_pde") or d.get("complex_ipde"),
    }


def walk_cofolds(root: Path):
    """Yield (target, name, cif, pae, conf) tuples for each cofold under root."""
    for cif in root.rglob("*_model_0.cif"):
        try:
            name = cif.parent.name
            target = cif.parents[3].name  # <root>/<target>/boltz_results_<name>/predictions/<name>/cif
        except IndexError:
            continue
        pae = cif.parent / f"pae_{name}_model_0.npz"
        conf = cif.parent / f"confidence_{name}_model_0.json"
        yield target, name, cif, pae, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Cofold root dir")
    ap.add_argument("output", type=Path)
    ap.add_argument("--target-override", default=None,
                    help="If set, force target column to this string (useful when root contains one target only)")
    args = ap.parse_args()

    rows = []
    for target, name, cif, pae, conf in walk_cofolds(args.root):
        n_prot, n_lig = chain_split(cif)
        if n_lig == 0:
            continue
        london_min, prot_lig_mean, n_pae = mpae_metrics(pae, n_lig)
        c = conf_metrics(conf)
        rows.append({
            "target": args.target_override or target,
            "name": name,
            "n_prot": n_prot,
            "n_lig": n_lig,
            "n_pae": n_pae,
            "mpae_london_min": london_min,
            "mpae_prot_lig_mean": prot_lig_mean,
            **c,
        })
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows → {args.output}")
    if len(df):
        print(df.groupby("target").size().to_string())
        print(f"  mpae_london_min NaN: {df['mpae_london_min'].isna().sum()}")
        print(f"  ligand_iptm NaN:    {df['ligand_iptm'].isna().sum()}")


if __name__ == "__main__":
    main()
