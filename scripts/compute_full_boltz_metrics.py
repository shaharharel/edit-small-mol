"""Compute geometry + contacts + mPAE-proxy metrics for every Boltz cofold in
`data/boltz_results/cohort_3597_full/from_*/<row_id>/`.

What this writes per cofold (one row in the output CSV):
  - row_id
  - warhead_class, warhead_atom_name (derived from SMILES)
  - d_SG (Å, Cys346.SG to ligand warhead Cβ; None if no warhead match)
  - burgi_dunitz_dev_deg (|angle(SG, Cβ, Cα) − 107°|)
  - geom_ok (d_SG <= 4.0 Å)
  - n_h_bonds (anchordiff v2 count; polar N/O ↔ N/O, 2.5–3.5 Å)
  - n_stabilizing_contacts (= n_cov + n_hb + n_sb + n_pi from compute_contacts_occupancy)
  - n_covalent, n_salt_bridges, n_pi_pi (sub-counts)
  - n_contacts_total (= n_stabilizing_contacts, alias for clarity)
  - pocket_occupancy_pct (lig hull vol / 8 Å pocket-shell hull vol × 100)
  - atp_pocket_fraction (frac of lig heavy atoms within 4 Å of ATP-pocket residues)
  - hinge_hbond (bool: any H-bond to backbone N/O of MET414/GLU415/MET416)
  - mPAE_paper  → confidence_score's complex_pde (PROXY; see note below)
  - mPAE_full   → confidence_score's complex_pde (PROXY; full-complex predicted dist error)
  - mPAE_interface → confidence_score's complex_ipde (PROXY; interface-only pred dist error)
  - mPAE_min    → London et al definition: min(PAE[protein_rows, ligand_cols]).
                  NOT available — Boltz was run with `--output_format mmcif` only,
                  which does NOT emit a `pae_<name>_model_0.npz`. Set to NaN with
                  a `mPAE_min_available=False` flag. To recover, Boltz must be re-run
                  with `--write_pae` (≈3 min/cofold × 2010 = ~100 GPU-hr).
  - mPAE_proxy_kind = "complex_pde" (sentinel; future-proofs if we mix sources)

------------------------------------------------------------------
LONDON ET AL mPAE DEFINITION (from `run_covalid_d3_london_mpae.py` &
`analyze_covalid_boltz_mpae.py` in this repo, which reproduce it):

  mPAE = min( PAE[protein_rows, ligand_cols] )                       (eq. London S2)

  where PAE is the AlphaFold3/Boltz predicted-aligned-error matrix
  (token × token, in Å), protein_rows is the contiguous block of
  protein-token rows, ligand_cols is the contiguous block of ligand-
  token cols. London et al apply a per-compound worst-case-protomer
  reduction (max over tautomers) for screening.

  Per-target adj_LogAUC results in London Table S2:
    BMX 56.4 | FGFR1 79.4 | FGFR4_477 83.8 | FGFR4_552 82.5 | JAK3 71.6 |
    EGFR 68.1 | MAP3K7 78.4 | KRAS 74.7 | BTK 72.3 | ITK 65.8.

  Source: London et al. 2024/2025, JACS — "Discovery of covalent ligands
  with AlphaFold3", Table S2 row "AF3-mPAE".

  In this run we use Boltz's per-cofold `confidence_score.complex_pde`
  (mean predicted DISTANCE error) as a proxy because the PAE NPZ block
  was not emitted. complex_pde is correlated with mPAE but is full-matrix
  AVERAGE not minimum over the cross-block, so this is NOT a faithful
  reproduction of London's metric. Use `boltz_ligand_iptm` and
  `boltz_iptm` (already in the cohort CSV) as orthogonal pose-quality
  signals.
------------------------------------------------------------------

Usage:
    python scripts/compute_full_boltz_metrics.py [--limit N]

Output:
    data/tier4_scored/boltz_full_metrics_3597.csv  (resumable)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

COFOLD_ROOT = PROJECT_ROOT / "data/boltz_results/cohort_3597_full"
COHORT_CSV = PROJECT_ROOT / "data/tier4_scored/boltz2_cohort_A_relaxed.csv"
OUT_CSV = PROJECT_ROOT / "data/tier4_scored/boltz_full_metrics_3597.csv"

# Silence RDKit/scipy
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _load_smiles_map() -> Dict[int, str]:
    """Cohort row_id → SMILES (used for SMARTS warhead derivation)."""
    df = pd.read_csv(COHORT_CSV, low_memory=False, usecols=["row_id", "smiles"])
    df["row_id"] = df["row_id"].astype(int)
    return dict(zip(df["row_id"], df["smiles"]))


def _confidence_proxies(conf_json: Path) -> dict:
    """Pull confidence-JSON metrics and derive mPAE proxies from them.

    Returns NaN-filled dict if file missing/corrupt."""
    keys = (
        "boltz_confidence_score", "boltz_ptm", "boltz_iptm", "boltz_ligand_iptm",
        "boltz_protein_iptm", "boltz_complex_plddt", "boltz_complex_iplddt",
        "boltz_complex_pde", "boltz_complex_ipde",
        "mPAE_paper", "mPAE_full", "mPAE_interface", "mPAE_min",
    )
    out: dict = {k: None for k in keys}
    out["mPAE_proxy_kind"] = None
    out["mPAE_min_available"] = False
    try:
        d = json.loads(conf_json.read_text())
    except Exception:
        return out
    out["boltz_confidence_score"] = d.get("confidence_score")
    out["boltz_ptm"] = d.get("ptm")
    out["boltz_iptm"] = d.get("iptm")
    out["boltz_ligand_iptm"] = d.get("ligand_iptm")
    out["boltz_protein_iptm"] = d.get("protein_iptm")
    out["boltz_complex_plddt"] = d.get("complex_plddt")
    out["boltz_complex_iplddt"] = d.get("complex_iplddt")
    out["boltz_complex_pde"] = d.get("complex_pde")
    out["boltz_complex_ipde"] = d.get("complex_ipde")
    # PAE proxies — see header for caveat
    # mPAE_paper = London min(PAE[prot, lig]).  Not available (no NPZ);
    # we fall back to complex_pde so the column is populated but flag as proxy.
    out["mPAE_paper"] = d.get("complex_pde")
    out["mPAE_full"] = d.get("complex_pde")          # mean over all pairs
    out["mPAE_interface"] = d.get("complex_ipde")    # mean over interface pairs
    out["mPAE_min"] = None                           # honest: cannot compute
    out["mPAE_proxy_kind"] = "complex_pde"
    out["mPAE_min_available"] = False
    return out


def _geom_metrics(cif: Path, smiles: str) -> dict:
    """Run anchordiff geometry + contacts on a single cofold CIF.

    Returns dict with d_SG/burgi/geom_ok + n_h_bonds + n_stabilizing_contacts +
    pocket_occupancy_pct + atp_pocket_fraction + hinge_hbond + warhead atom name."""
    from anchordiff.compute_pose_quality_v2 import (
        parse_cofold, d_SG_honest, burgi_dunitz_dev, n_h_bonds as count_h_bonds,
        atp_pocket_fraction, hinge_hbond_present, derive_warhead_atom_name,
    )
    from anchordiff.compute_contacts_occupancy import (
        find_covalent, find_h_bonds, find_salt_bridges, find_pi_pi,
        pocket_occupancy,
    )

    out: dict = {
        "warhead_class": None, "warhead_atom_name": None,
        "d_SG": None, "geom_ok": None, "burgi_dunitz_dev_deg": None,
        "n_h_bonds": None, "n_covalent": None, "n_salt_bridges": None,
        "n_pi_pi": None, "n_stabilizing_contacts": None,
        "n_contacts_total": None,
        "pocket_occupancy_pct": None,
        "atp_pocket_fraction": None, "hinge_hbond": None,
    }
    # Detect warhead from SMILES (no manifest column for this cohort).
    wh_class, wh_atom = derive_warhead_atom_name(smiles)
    out["warhead_class"] = wh_class
    out["warhead_atom_name"] = wh_atom

    prot, lig = parse_cofold(cif)
    if not lig:
        return out

    d_sg = d_SG_honest(prot, lig, wh_atom)
    bd = burgi_dunitz_dev(prot, lig, wh_atom)
    nhb_v2, _ = count_h_bonds(prot, lig)
    apf = atp_pocket_fraction(prot, lig, cutoff_A=4.0)
    hh = hinge_hbond_present(prot, lig)
    n_cov, _ = find_covalent(prot, lig)
    n_hb_c, _ = find_h_bonds(prot, lig)
    n_sb, _ = find_salt_bridges(prot, lig)
    n_pi, _ = find_pi_pi(prot, lig, smiles)
    occ = pocket_occupancy(prot, lig)

    n_stab = int(n_cov + n_hb_c + n_sb + n_pi)
    out.update({
        "d_SG": d_sg,
        "geom_ok": (None if d_sg is None else bool(d_sg <= 4.0)),
        "burgi_dunitz_dev_deg": bd,
        "n_h_bonds": int(nhb_v2),
        "n_covalent": int(n_cov),
        "n_salt_bridges": int(n_sb),
        "n_pi_pi": int(n_pi),
        "n_stabilizing_contacts": n_stab,
        "n_contacts_total": n_stab,
        "pocket_occupancy_pct": (float(occ) if occ is not None else None),
        "atp_pocket_fraction": float(apf),
        "hinge_hbond": bool(hh),
    })
    return out


def _process_one(args) -> dict:
    """Per-cofold worker. Returns one row dict (or sentinel on failure)."""
    row_id, cif_path, conf_path, smiles = args
    rec: dict = {"row_id": int(row_id), "cif_dir": str(cif_path.parent.name),
                 "error": None}
    try:
        rec.update(_confidence_proxies(conf_path))
    except Exception as e:
        rec["error"] = f"conf:{e}"
    try:
        rec.update(_geom_metrics(cif_path, smiles))
    except Exception as e:
        prev = rec.get("error")
        rec["error"] = (prev + "; " if prev else "") + f"geom:{e}"
    return rec


def _discover_cofolds(smiles_map: Dict[int, str]) -> list:
    """Walk every from_* dir → list of (row_id, cif, conf, smiles) tuples."""
    tasks = []
    for from_dir in sorted(COFOLD_ROOT.glob("from_*")):
        if not from_dir.is_dir():
            continue
        for cofold_dir in sorted(from_dir.iterdir()):
            if not cofold_dir.is_dir() or not cofold_dir.name.isdigit():
                continue
            row_id = int(cofold_dir.name)
            cif = cofold_dir / f"{row_id}_model_0.cif"
            conf = cofold_dir / f"confidence_{row_id}_model_0.json"
            if not cif.exists() or not conf.exists():
                continue
            smi = smiles_map.get(row_id)
            if smi is None:
                continue
            tasks.append((row_id, cif, conf, smi))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if row already in OUT_CSV")
    args = ap.parse_args()

    print(f"=== compute_full_boltz_metrics ===")
    print(f"cohort csv : {COHORT_CSV}")
    print(f"cofold root: {COFOLD_ROOT}")
    print(f"output     : {OUT_CSV}")

    smiles_map = _load_smiles_map()
    print(f"cohort SMILES loaded: {len(smiles_map):,} rows")

    tasks = _discover_cofolds(smiles_map)
    print(f"discovered cofolds   : {len(tasks):,}")

    # Resume support — skip row_ids already present in OUT_CSV
    done_ids: set = set()
    if OUT_CSV.exists() and not args.force:
        prev = pd.read_csv(OUT_CSV)
        done_ids = set(prev["row_id"].astype(int).tolist())
        print(f"resume: {len(done_ids):,} rows already in output, will skip them")
        tasks = [t for t in tasks if t[0] not in done_ids]
        print(f"remaining            : {len(tasks):,}")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"limit applied        : {len(tasks):,}")

    if not tasks:
        print("nothing to do")
        return 0

    t0 = time.perf_counter()
    rows: list = []
    n_done = 0
    n_err = 0
    BATCH = 200  # flush to CSV every N
    with mp.Pool(args.workers) as pool:
        for rec in pool.imap_unordered(_process_one, tasks, chunksize=4):
            rows.append(rec)
            n_done += 1
            if rec.get("error"):
                n_err += 1
            if n_done % BATCH == 0:
                elapsed = time.perf_counter() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(tasks) - n_done) / max(rate, 1e-6)
                print(f"  [{n_done:5d}/{len(tasks)}]  err={n_err}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min")

    print(f"\nfinished {n_done} cofolds in {time.perf_counter()-t0:.1f}s "
          f"({n_done / max(time.perf_counter()-t0, 1e-6):.1f}/s)  errors={n_err}")

    df_new = pd.DataFrame(rows)

    # Merge with existing
    if OUT_CSV.exists() and not args.force:
        df_prev = pd.read_csv(OUT_CSV)
        # Ensure schema parity
        for c in df_new.columns:
            if c not in df_prev.columns:
                df_prev[c] = None
        for c in df_prev.columns:
            if c not in df_new.columns:
                df_new[c] = None
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset="row_id", keep="last")
    else:
        df_all = df_new

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df_all):,} rows × {len(df_all.columns)} cols)")

    # Quick stats
    print(f"\n=== coverage / sanity ===")
    sub = df_all
    for c in ("d_SG", "burgi_dunitz_dev_deg", "n_h_bonds",
              "n_stabilizing_contacts", "pocket_occupancy_pct",
              "atp_pocket_fraction", "mPAE_paper", "mPAE_full",
              "mPAE_interface", "boltz_iptm", "boltz_complex_plddt"):
        if c not in sub.columns:
            continue
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if not len(s):
            print(f"  {c}: empty")
            continue
        print(f"  {c:28s}  n={len(s):5d}  med={s.median():7.3f}  "
              f"p10={s.quantile(0.1):7.3f}  p90={s.quantile(0.9):7.3f}  "
              f"min={s.min():7.3f}  max={s.max():7.3f}")
    # bool / categorical coverage
    for c in ("hinge_hbond", "geom_ok", "warhead_class", "mPAE_min_available"):
        if c not in sub.columns:
            continue
        v = sub[c].value_counts(dropna=False).to_dict()
        print(f"  {c}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
