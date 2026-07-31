"""Compute 3D shape and ESP-like (USRCAT) similarity vs the Mol1 anchor for the
3,597-mol ZAP70 cohort.

Reference (anchor): MOL1_SMILES from `experiments/server/backend.py`
    C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1

Per-mol pipeline:
  SMILES → AddHs → ETKDGv3 embed → MMFF94 optimize (200 iter)
  shape_Tc_seed = 1 - ShapeTanimotoDist(ref, mol)         # 3D shape Tanimoto
  esp_sim_seed  = GetUSRScore(ref_usrcat, mol_usrcat)    # USRCAT proxy for ESP/pharmacophore

Output: data/tier4_scored/shape_sim_3597.csv
        columns row_id, shape_Tc_seed, esp_sim_seed, success_flag, error

Resumable. Heartbeat to /tmp/shape_progress.json every 500 rows.

Run: /opt/miniconda3/envs/quris/bin/python scripts/compute_shape_sim_3597.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

# Silence RDKit chatter from worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.rdMolDescriptors import GetUSRCAT, GetUSRScore  # noqa: E402
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT_ROOT / "data/tier4_scored/boltz2_cohort_A_relaxed.csv"
OUT_CSV = PROJECT_ROOT / "data/tier4_scored/shape_sim_3597.csv"
HEARTBEAT_PATH = Path("/tmp/shape_progress.json")

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
N_WORKERS = 12
SEED = 42
MMFF_ITERS = 200


# ---------------------------------------------------------------------------
# Reference (Mol1) — embedded once, USRCAT and 3D ref-mol passed to workers.
# Workers receive the (heavy-atom-only) Mol via pickling.
# ---------------------------------------------------------------------------
def build_reference() -> tuple[Chem.Mol, tuple[float, ...]]:
    mol = Chem.MolFromSmiles(MOL1_SMILES)
    if mol is None:
        raise RuntimeError(f"Failed to parse MOL1_SMILES: {MOL1_SMILES}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDG embed failed for MOL1 reference")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=MMFF_ITERS)
    usrcat = GetUSRCAT(mol)
    return mol, tuple(usrcat)


# Globals populated in each worker via initializer.
_REF_MOL: Chem.Mol | None = None
_REF_USRCAT: tuple[float, ...] | None = None


def _worker_init(ref_mol_pkl: bytes, ref_usrcat: tuple[float, ...]) -> None:
    global _REF_MOL, _REF_USRCAT
    _REF_MOL = Chem.Mol(ref_mol_pkl)
    _REF_USRCAT = ref_usrcat
    RDLogger.DisableLog("rdApp.*")


def _compute_one(args: tuple[int, str]) -> dict:
    row_id, smi = args
    out = {
        "row_id": row_id,
        "shape_Tc_seed": None,
        "esp_sim_seed": None,
        "success_flag": False,
        "error": "",
    }
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out["error"] = "smiles_parse_fail"
            return out
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        embed_ret = AllChem.EmbedMolecule(mol, params)
        if embed_ret != 0:
            # Retry with random coords as a fallback
            params2 = AllChem.ETKDGv3()
            params2.randomSeed = SEED + 1
            params2.useRandomCoords = True
            embed_ret = AllChem.EmbedMolecule(mol, params2)
            if embed_ret != 0:
                out["error"] = "etkdg_embed_fail"
                return out
        AllChem.MMFFOptimizeMolecule(mol, maxIters=MMFF_ITERS)
        shape_tc = 1.0 - ShapeTanimotoDist(_REF_MOL, mol)
        usr = GetUSRCAT(mol)
        esp_tc = GetUSRScore(list(_REF_USRCAT), list(usr))
        out["shape_Tc_seed"] = float(shape_tc)
        out["esp_sim_seed"] = float(esp_tc)
        out["success_flag"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}:{e}"
    return out


def main() -> int:
    print("=== Shape + USRCAT similarity vs Mol1 (3,597 cohort) ===")
    if not COHORT_CSV.exists():
        print(f"Cohort CSV missing: {COHORT_CSV}")
        return 1

    cohort = pd.read_csv(COHORT_CSV, usecols=["row_id", "smiles"], low_memory=False)
    cohort["row_id"] = cohort["row_id"].astype(int)
    print(f"Cohort: {len(cohort):,} rows")

    # Resume
    done_ids: set[int] = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done_ids = set(prev["row_id"].astype(int).tolist())
        print(f"Resume: {len(done_ids):,} row_ids already in {OUT_CSV.name}")

    todo = cohort[~cohort["row_id"].isin(done_ids)].reset_index(drop=True)
    print(f"To compute: {len(todo):,}")
    if len(todo) == 0:
        print("Nothing to do; sanity-check existing CSV.")
        _print_dist(pd.read_csv(OUT_CSV))
        return 0

    ref_mol, ref_usrcat = build_reference()
    ref_pkl = ref_mol.ToBinary()
    print(f"Reference Mol1 embedded (atoms={ref_mol.GetNumAtoms()}, USRCAT dim={len(ref_usrcat)})")

    args_iter = list(zip(todo["row_id"].tolist(), todo["smiles"].tolist()))

    t0 = time.perf_counter()
    rows: list[dict] = []
    # Write header upfront if no prior output
    write_header = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    BATCH = 500
    with Pool(N_WORKERS, initializer=_worker_init, initargs=(ref_pkl, ref_usrcat)) as pool:
        for i, res in enumerate(pool.imap_unordered(_compute_one, args_iter, chunksize=8), start=1):
            rows.append(res)
            if i % BATCH == 0 or i == len(args_iter):
                # Flush to CSV
                batch_df = pd.DataFrame(rows)
                batch_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
                write_header = False
                rows = []
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(args_iter) - i) / max(rate, 1e-6)
                heartbeat = {
                    "done": i,
                    "total": len(args_iter),
                    "elapsed_s": round(elapsed, 1),
                    "rate_per_s": round(rate, 2),
                    "eta_s": round(eta, 1),
                    "timestamp": time.time(),
                }
                try:
                    HEARTBEAT_PATH.write_text(json.dumps(heartbeat))
                except OSError:
                    pass
                print(
                    f"  [{i:5,}/{len(args_iter):,}] "
                    f"elapsed={elapsed:6.1f}s rate={rate:5.2f}/s eta={eta:5.1f}s"
                )

    print(f"\nDone in {time.perf_counter() - t0:.1f}s → {OUT_CSV}")
    final = pd.read_csv(OUT_CSV)
    print(f"Total rows in output: {len(final):,}")
    _print_dist(final)
    return 0


def _print_dist(df: pd.DataFrame) -> None:
    print("\n=== Distribution sanity check ===")
    print(f"  success_flag: {df['success_flag'].sum():,}/{len(df):,}")
    fails = df[~df["success_flag"].astype(bool)]
    if len(fails):
        print(f"  failures by error:")
        print(fails["error"].value_counts().head(10).to_string())
    for c in ("shape_Tc_seed", "esp_sim_seed"):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            print(f"  {c}: no successful rows")
            continue
        print(
            f"  {c}: n={len(s):,}  min={s.min():.3f}  p10={s.quantile(0.10):.3f}  "
            f"median={s.median():.3f}  p90={s.quantile(0.90):.3f}  max={s.max():.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
