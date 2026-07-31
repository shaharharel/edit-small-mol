"""Phase A — compute xTB-derived warhead reactivity labels for CovInDB 2.0.

For every UNIQUE SMILES in the union of `Covalent_Complex_Records.csv` and
`CovInDB_All.csv` we:

  1. Try to parse the SMILES with RDKit. Skip on parse fail.
  2. Classify the warhead:
       'acrylamide'      — bears [CH2]=[CH]-[C](=O)-[N]   (labelable)
       'other_michael'   — bears C=CC(=O)*               (skipped; QSAR is
                           acrylamide-specific)
       'chloroacetamide' — bears Cl-CH2-C(=O)-N          (skipped)
       'other'           — anything else                  (skipped)
       'no_match'        — no recognised warhead          (skipped)
  3. For 'acrylamide' rows, call the existing `compute_warhead_descriptors`
     from `experiments/xtb_warhead_electrophilicity.py` (the canonical
     pipeline that produced the 23K cofold-dashboard log_k2_GSH values).
     We take the heuristic `pred_log_k2_GSH` field as the regression target.
  4. Parallelise via `multiprocessing.Pool(n_workers)` — xTB single-points
     on small ligands take ~0.05-0.6 s wall-clock per molecule on Mac CPU,
     so a few thousand mols should take a few minutes wall, not hours.

Output: a CSV with columns
    smiles, log_k2_GSH, warhead_class, xtb_status, n_heavy, elapsed_s,
    LUMO_eV, q_Cb, fukui_plus_Cb, error

xtb_status is one of: 'ok' | 'skipped_nonacryl' | 'skipped_no_warhead' |
                      'skipped_parse_fail' | 'failed'

We DO NOT re-implement the QSAR — we re-use `compute_warhead_descriptors`
which already returns `pred_log_k2_GSH`.

Usage:
    python experiments/compute_covindb_xtb_reactivity.py \
        --out data/covindb_xtb_reactivity.csv --workers 8

Mac CPU only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make `experiments/xtb_warhead_electrophilicity` importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from xtb_warhead_electrophilicity import (  # noqa: E402
    ACRYLAMIDE_SMARTS,
    compute_warhead_descriptors,
)

# Warhead-class SMARTS (priority order). 'acrylamide' is the only labelable
# class — the QSAR was calibrated on acrylamides.
_MICHAEL_GENERAL = Chem.MolFromSmarts("C=CC(=O)*")
_CHLOROACET     = Chem.MolFromSmarts("ClCC(=O)N")
_VINYL_SULFON   = Chem.MolFromSmarts("C=CS(=O)(=O)*")
_NITRILE        = Chem.MolFromSmarts("[#6]C#N")


def classify_warhead(mol: Chem.Mol) -> str:
    """Return one of: 'acrylamide', 'other_michael', 'chloroacetamide',
    'vinyl_sulfone', 'nitrile', 'no_match'."""
    if mol.HasSubstructMatch(ACRYLAMIDE_SMARTS):
        return "acrylamide"
    if mol.HasSubstructMatch(_MICHAEL_GENERAL):
        return "other_michael"
    if mol.HasSubstructMatch(_CHLOROACET):
        return "chloroacetamide"
    if mol.HasSubstructMatch(_VINYL_SULFON):
        return "vinyl_sulfone"
    if mol.HasSubstructMatch(_NITRILE):
        return "nitrile"
    return "no_match"


def _collect_unique_smiles(complex_csv: Path, all_csv: Path) -> List[str]:
    """Return the union of SMILES from the two CovInDB files, deduplicated
    and stripped. Order is deterministic (sorted)."""
    smis: set[str] = set()
    for p in (complex_csv, all_csv):
        if not p.exists():
            print(f"[warn] {p} does not exist — skipping")
            continue
        df = pd.read_csv(p)
        if "SMILES" not in df.columns:
            print(f"[warn] {p} has no SMILES column — skipping")
            continue
        for s in df["SMILES"].dropna().astype(str).str.strip():
            if s and s.lower() not in ("nan", "none"):
                smis.add(s)
    return sorted(smis)


def _worker(job: Tuple[int, str]) -> Dict:
    """Per-SMILES worker: classify, then run xtb if acrylamide."""
    idx, smi = job
    # Force single-thread xtb in workers (compute_warhead_descriptors sets
    # these too, but redundancy is cheap).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("XTB_NUM_THREADS", "1")

    rec: Dict = {
        "idx": idx,
        "smiles": smi,
        "log_k2_GSH": np.nan,
        "warhead_class": "",
        "xtb_status": "",
        "n_heavy": -1,
        "elapsed_s": 0.0,
        "LUMO_eV": np.nan,
        "q_Cb": np.nan,
        "fukui_plus_Cb": np.nan,
        "error": "",
    }
    t0 = time.time()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        rec["warhead_class"] = "parse_fail"
        rec["xtb_status"] = "skipped_parse_fail"
        rec["elapsed_s"] = round(time.time() - t0, 3)
        return rec
    rec["n_heavy"] = int(mol.GetNumHeavyAtoms())
    cls = classify_warhead(mol)
    rec["warhead_class"] = cls
    if cls != "acrylamide":
        rec["xtb_status"] = (
            "skipped_no_warhead" if cls == "no_match" else "skipped_nonacryl"
        )
        rec["elapsed_s"] = round(time.time() - t0, 3)
        return rec

    # Acrylamide → run xtb.
    try:
        out = compute_warhead_descriptors(smi)
    except Exception as e:
        rec["xtb_status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = round(time.time() - t0, 3)
        return rec

    if int(out.get("success_flag", 0)) == 1 and np.isfinite(
        out.get("pred_log_k2_GSH", np.nan)
    ):
        rec["log_k2_GSH"] = float(out["pred_log_k2_GSH"])
        rec["LUMO_eV"] = float(out.get("LUMO_eV", np.nan))
        rec["q_Cb"] = float(out.get("q_Cb", np.nan))
        rec["fukui_plus_Cb"] = float(out.get("fukui_plus_Cb", np.nan))
        rec["xtb_status"] = "ok"
    else:
        rec["xtb_status"] = "failed"
        rec["error"] = str(out.get("error", ""))[:200]
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--complex_csv",
        default=str(Path.home() / "Downloads/CovInDB2/CovInDB/Covalent_Complex_Records.csv"),
    )
    ap.add_argument(
        "--all_csv",
        default=str(Path.home() / "Downloads/CovInDB2/CovInDB/CovInDB_All.csv"),
    )
    ap.add_argument("--out", default="data/covindb_xtb_reactivity.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N unique SMILES (dev/smoke).",
    )
    ap.add_argument(
        "--flush_every",
        type=int,
        default=200,
        help="Write CSV partial every N completed jobs.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_list = _collect_unique_smiles(Path(args.complex_csv), Path(args.all_csv))
    print(f"[main] unique SMILES across CovInDB: {len(smiles_list)}")
    if args.limit > 0:
        smiles_list = smiles_list[: args.limit]
        print(f"[main] limited to first {args.limit}")

    jobs = list(enumerate(smiles_list))

    rows: List[Dict] = []
    t_start = time.time()
    print(
        f"[main] starting xTB labelling: {len(jobs)} mols, {args.workers} workers, "
        f"out={out_path}"
    )
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            rows.append(rec)
            if i % args.flush_every == 0 or i == len(jobs):
                df = pd.DataFrame(rows).sort_values("idx").drop(columns=["idx"])
                df.to_csv(out_path, index=False)
                ok = sum(1 for r in rows if r["xtb_status"] == "ok")
                acr_total = sum(1 for r in rows if r["warhead_class"] == "acrylamide")
                el = time.time() - t_start
                eta = (el / i) * (len(jobs) - i) / 60 if i < len(jobs) else 0.0
                print(
                    f"  [{i}/{len(jobs)}] acryl_seen={acr_total} ok={ok} "
                    f"elapsed={el/60:.1f}m  eta={eta:.1f}m",
                    flush=True,
                )

    df = pd.DataFrame(rows).sort_values("idx").drop(columns=["idx"])
    df.to_csv(out_path, index=False)
    el = time.time() - t_start

    # Summary
    n_total = len(df)
    n_acryl = int((df["warhead_class"] == "acrylamide").sum())
    n_ok = int((df["xtb_status"] == "ok").sum())
    n_fail = int((df["xtb_status"] == "failed").sum())
    if n_acryl > 0:
        per_mol_acryl_s = (
            df.loc[df["warhead_class"] == "acrylamide", "elapsed_s"].mean()
        )
    else:
        per_mol_acryl_s = 0.0

    print()
    print("=" * 72)
    print("[summary] xTB CovInDB labelling complete")
    print(f"  total SMILES processed:    {n_total}")
    print(f"  acrylamide warheads:       {n_acryl}")
    print(f"  successful xTB labels:     {n_ok}")
    print(f"  failed xTB:                {n_fail}")
    print(f"  total wall time:           {el/60:.2f} min")
    print(f"  mean per-acrylamide:       {per_mol_acryl_s:.2f} s")
    print(f"  output:                    {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
