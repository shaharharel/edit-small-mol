"""Rescue-78 Phase 3 driver.

Adapts experiments/compute_full_boltz_metrics.py + compute_pose_extras_for_f4.py
+ run_mmgbsa_838.py to operate on the 78 rescue cofold CIFs at
data/boltz_rescue_78/<rescue_row_id>/<rescue_row_id>_model_0.cif.

Phases:
  3.1 boltz_iptm/ligand_iptm/protein_iptm/ptm/confidence/plddt/iplddt/pde/ipde
  3.2 mPAE_paper + mPAE_london
  3.3 d_SG, burgi_dunitz_dev_deg, geom_ok, n_h_bonds, n_stabilizing_contacts,
      pocket_occupancy_pct, warhead_dev_deg (3D), warhead_atom_name, warhead_class,
      n_contacts_total, n_salt_bridges, atp_pocket_fraction, hinge_hbond
  3.4 vina_rescore_affinity/intra/inter/torsions
  3.5 covvina_rescore_*
  3.6 pKa_Cys346 (PROPKA), rdkit_strain_kcal_mol (MMFF94)
  3.7 dG_GB_kcalmol, ggas, gsolv, E_vdw, E_eel  (single-frame MM-GBSA igb=8)
  3.8 k_inact_proxy = thiolate_fraction(pKa) × geom_factor(d_SG, BD)

All writes go into data/tier4_scored/rescue_78_working.csv.
F4_boltz_full.csv is NEVER touched.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "anchordiff"))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

import gemmi  # noqa: E402

WORKING = ROOT / "data/tier4_scored/rescue_78_working.csv"
STATE = ROOT / "data/tier4_scored/rescue_78_state.json"
LOG = ROOT / "data/tier4_scored/rescue_78_progress.log"
BOLTZ_DIR = ROOT / "data/boltz_rescue_78"


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{now_z()}] {msg}\n")


def update_state(phase: str, status: str, **extra) -> None:
    s = json.loads(STATE.read_text())
    s["phases"][phase] = {"status": status, "ts": now_z(), **extra}
    STATE.write_text(json.dumps(s, indent=2))


def load_working() -> pd.DataFrame:
    return pd.read_csv(WORKING)


def save_working(df: pd.DataFrame) -> None:
    df.to_csv(WORKING, index=False)


# Module-scope worker for ProcessPoolExecutor pickling
def _phase3_compute_row_worker(args):
    from compute_full_boltz_metrics import compute_row
    rid, pred_dir_str, smi = args
    return compute_row(rid, Path(pred_dir_str), smi,
                       enable_vina=True, enable_propka=True, enable_shape=True)


# ─────────────── Extended mPAE family (parity with 838 schema) ───────────────

def compute_extended_mpae(pred_dir: Path, rid: str, smiles: str) -> dict:
    """Compute the extended mPAE family from pae_*.npz so the rescue-78 cohort
    has feature parity with the 838 cohort's mPAE columns:

      - mPAE_full           : mean over the full protein×ligand block
      - mPAE_interface      : mean over interface-only pairs (proxy: same block)
      - mPAE_min            : min over the protein×ligand block (London 2026)
      - mPAE_min_london     : same as mPAE_min (alias used elsewhere)
      - mPAE_min_raw_atompair : same as mPAE_min when no separate atompair PAE
                                exists (Boltz emits token-level PAE only)
      - mPAE_mean_lig_to_prot : mean over ligand×protein submatrix
      - mPAE_proxy_kind     : "pae_npz_full" once we have the raw matrix
      - mPAE_min_available  : True once we have the raw matrix
    """
    out = {
        "mPAE_full": None,
        "mPAE_interface": None,
        "mPAE_min": None,
        "mPAE_min_london": None,
        "mPAE_min_raw_atompair": None,
        "mPAE_mean_lig_to_prot": None,
        "mPAE_proxy_kind": None,
        "mPAE_min_available": False,
    }
    pae_npz = pred_dir / f"pae_{rid}_model_0.npz"
    if not pae_npz.exists():
        return out
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return out
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return out

    n_total = pae.shape[0]
    n_lig_candidates = []
    try:
        m = Chem.MolFromSmiles(smiles) if smiles else None
        if m is not None:
            n_lig_candidates.append(m.GetNumHeavyAtoms())
            m_h = Chem.AddHs(m)
            n_lig_candidates.append(m_h.GetNumAtoms())
    except Exception:
        pass

    n_lig = None
    for cand in n_lig_candidates:
        if 0 < cand < n_total - 50:
            n_lig = cand
            break
    if n_lig is None:
        return out

    lig_lo = n_total - n_lig
    cross_prot_to_lig = pae[:lig_lo, lig_lo:]
    cross_lig_to_prot = pae[lig_lo:, :lig_lo]
    if cross_prot_to_lig.size == 0:
        return out

    out["mPAE_full"] = float(pae.mean())
    out["mPAE_interface"] = float(cross_prot_to_lig.mean())
    out["mPAE_min"] = float(np.min(cross_prot_to_lig))
    out["mPAE_min_london"] = out["mPAE_min"]
    out["mPAE_min_raw_atompair"] = out["mPAE_min"]
    out["mPAE_mean_lig_to_prot"] = float(cross_lig_to_prot.mean())
    out["mPAE_proxy_kind"] = "pae_npz_full"
    out["mPAE_min_available"] = True
    return out


# ─────────────── Build pred_dir index for rescue_78 ────────────────

def build_rescue_index() -> dict:
    """Return dict rescue_row_id → pred_dir Path.

    Expected layout (after Phase 2.4 pull):
      data/boltz_rescue_78/boltz_results_<rid>/predictions/<rid>/<rid>_model_0.cif
    OR (simpler layout used by pull script):
      data/boltz_rescue_78/<rid>/<rid>_model_0.cif

    Both are supported.
    """
    index = {}
    if not BOLTZ_DIR.exists():
        return index
    for sub in BOLTZ_DIR.iterdir():
        if not sub.is_dir():
            continue
        if sub.name.startswith("boltz_results_"):
            rid = sub.name[len("boltz_results_"):]
            pred_dir = sub / "predictions" / rid
            if (pred_dir / f"{rid}_model_0.cif").exists():
                index[rid] = pred_dir
        else:
            rid = sub.name
            if (sub / f"{rid}_model_0.cif").exists():
                index[rid] = sub
    return index


# ─────────────── 3.1 + 3.2 + part of 3.3 via compute_full_boltz_metrics ───────

def phase_3_main():
    """Run the bundled boltz metrics + pose extras + MM-GBSA on the 78 CIFs."""
    df = load_working()
    rids = df["rescue_row_id"].tolist()
    smis = df["smiles"].tolist()
    smi_by_rid = dict(zip(rids, smis))
    index = build_rescue_index()
    have = [r for r in rids if r in index]
    missing = [r for r in rids if r not in index]
    append_log(f"Phase 3 start: {len(have)}/{len(rids)} cofolds present, missing={missing[:5]}...")

    to_do = [(rid, str(index[rid]), smi_by_rid[rid]) for rid in have]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_phase3_compute_row_worker, t): t[0] for t in to_do}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"row_id": rid, "boltz_compute_error": f"{type(e).__name__}:{e}"})
            if len(results) % 10 == 0:
                append_log(f"  full_boltz {len(results)}/{len(to_do)} ({time.time()-t0:.0f}s)")
    fb_df = pd.DataFrame(results).rename(columns={"row_id": "rescue_row_id"})
    append_log(f"full_boltz_metrics done: {len(fb_df)} rows ({time.time()-t0:.0f}s)")

    # Merge into df by rescue_row_id
    overlap = [c for c in fb_df.columns if c in df.columns and c != "rescue_row_id"]
    df = df.drop(columns=overlap, errors="ignore")
    df = df.merge(fb_df, on="rescue_row_id", how="left")
    save_working(df)

    # Extended mPAE family (matches 838's mPAE_full / mPAE_interface /
    # mPAE_min_london / mPAE_min_raw_atompair / mPAE_mean_lig_to_prot)
    ext_rows = []
    for rid in have:
        rec = compute_extended_mpae(index[rid], rid, smi_by_rid[rid])
        rec["rescue_row_id"] = rid
        ext_rows.append(rec)
    ext_df = pd.DataFrame(ext_rows)
    ext_overlap = [c for c in ext_df.columns if c in df.columns and c != "rescue_row_id"]
    df = df.drop(columns=ext_overlap, errors="ignore")
    df = df.merge(ext_df, on="rescue_row_id", how="left")
    save_working(df)

    # ── 3.1 & 3.2 update state ────────────────────────────────────────────
    update_state("3.1_boltz_metrics", "done",
                 n_cofolds=int(df["boltz_iptm"].notna().sum()))
    update_state("3.2_mpae_family", "done",
                 n_mpae_paper=int(df["mPAE_paper"].notna().sum()),
                 n_mpae_london=int(df["mPAE_london"].notna().sum()),
                 n_mpae_min_london=int(df["mPAE_min_london"].notna().sum()),
                 n_mpae_full=int(df["mPAE_full"].notna().sum()),
                 n_mpae_mean_lig_to_prot=int(df["mPAE_mean_lig_to_prot"].notna().sum()))

    # ── 3.3 extra pose columns via compute_pose_extras_for_f4 ──
    from compute_pose_extras_for_f4 import compute_pose_cols
    pose_args = []
    for rid in have:
        cif = index[rid] / f"{rid}_model_0.cif"
        pose_args.append((rid, smi_by_rid[rid], str(cif)))

    pose_results = []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(compute_pose_cols, a): a[0] for a in pose_args}
        for fut in as_completed(futs):
            try:
                pose_results.append(fut.result())
            except Exception as e:
                pose_results.append({"row_id": futs[fut], "error": str(e)})
    pose_df = pd.DataFrame(pose_results).rename(columns={"row_id": "rescue_row_id"})
    pose_df = pose_df.drop(columns=["error"], errors="ignore")
    # Overwrite warhead_dev_deg + add n_contacts_total etc.; pose-extras has
    # the canonical values, so just drop the existing columns and merge fresh.
    drop_cols = [c for c in pose_df.columns
                 if c in df.columns and c != "rescue_row_id"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.merge(pose_df, on="rescue_row_id", how="left")
    save_working(df)
    update_state("3.3_cofold_geom", "done",
                 n_warhead_dev=int(df["warhead_dev_deg"].notna().sum()),
                 n_contacts=int(df["n_contacts_total"].notna().sum()))
    append_log(f"pose_extras done ({time.time()-t1:.0f}s)")

    # ── 3.4 vina_rescore is already in compute_row ────────────────────────
    update_state("3.4_vina_rescore", "done",
                 n_vina=int(df["vina_rescore_affinity_kcalmol"].notna().sum()))
    # ── 3.5 cov-Vina rescore via covvina_rescore_cohort_3597.py ─────────────
    # The script expects `--cofold-root` with a `from_<letter>/<cofold_name>/cif`
    # layout. We build a synthetic from_r/ directory of symlinks pointing to the
    # rescue_78 CIFs.
    cov_root = ROOT / "data/tier4_scored/rescue_78_covvina_root"
    (cov_root / "from_r").mkdir(parents=True, exist_ok=True)
    for rid in have:
        cof_dir = cov_root / "from_r" / rid
        cof_dir.mkdir(exist_ok=True)
        src_cif = index[rid] / f"{rid}_model_0.cif"
        dst_cif = cof_dir / f"{rid}_model_0.cif"
        if not dst_cif.exists():
            try:
                dst_cif.symlink_to(src_cif.resolve())
            except Exception:
                import shutil; shutil.copy(src_cif, dst_cif)

    cov_out = ROOT / "data/tier4_scored/rescue_78_covvina_rescore.csv"
    cov_cmd = [
        "/opt/miniconda3/envs/quris/bin/python",
        str(ROOT / "experiments/covvina_rescore_cohort_3597.py"),
        "--cofold-root", str(cov_root),
        "--out", str(cov_out),
        "--workers", "4",
    ]
    append_log(f"covVina cmd: {' '.join(cov_cmd)}")
    t_cov = time.time()
    try:
        r = subprocess.run(cov_cmd, capture_output=True, text=True, timeout=3600)
        append_log(f"covVina rc={r.returncode} wall={time.time()-t_cov:.0f}s")
        if r.returncode != 0:
            append_log(f"  covVina stderr tail: {r.stderr[-400:]}")
    except subprocess.TimeoutExpired:
        append_log("covVina TIMED OUT (1h cap)")

    if cov_out.exists():
        cv = pd.read_csv(cov_out)
        # rename column-headers to match the 838 schema
        cv_keep = cv.rename(columns={
            "yaml_name": "rescue_row_id",
            "covvina_affinity_kcalmol": "covvina_rescore_affinity_kcalmol",
            "covvina_inter_kcalmol":   "covvina_rescore_inter_kcalmol",
            "covvina_intra_kcalmol":   "covvina_rescore_intra_kcalmol",
            "covvina_torsions_kcalmol":"covvina_rescore_torsions_kcalmol",
        })
        keep_cols = ["rescue_row_id",
                     "covvina_rescore_affinity_kcalmol",
                     "covvina_rescore_inter_kcalmol",
                     "covvina_rescore_intra_kcalmol",
                     "covvina_rescore_torsions_kcalmol"]
        cv_keep = cv_keep[[c for c in keep_cols if c in cv_keep.columns]]
        # Drop existing covvina cols then merge
        drop = [c for c in cv_keep.columns if c in df.columns and c != "rescue_row_id"]
        df = df.drop(columns=drop, errors="ignore")
        df = df.merge(cv_keep, on="rescue_row_id", how="left")
        save_working(df)
        n_cv = int(df["covvina_rescore_affinity_kcalmol"].notna().sum()) if "covvina_rescore_affinity_kcalmol" in df.columns else 0
        update_state("3.5_covvina", "done", n_covvina=n_cv)
        append_log(f"covVina merged: {n_cv} rows non-null")
    else:
        update_state("3.5_covvina", "failed", reason="covVina output CSV missing")

    # ── 3.6 PROPKA + RDKit strain — already in compute_row ────────────────
    update_state("3.6_propka_strain", "done",
                 n_pKa=int(df["pKa_Cys346"].notna().sum()),
                 n_strain=int(df["rdkit_strain_kcal_mol"].notna().sum()))

    # ── 3.8 Recompute k_inact_proxy with real pKa+geometry ───────────────
    def k_inact(pKa, d, bd):
        if pd.isna(pKa) or pd.isna(d):
            return np.nan
        if pd.isna(bd):
            bd = 30.0
        try:
            thiolate = 1.0 / (1.0 + 10 ** (float(pKa) - 7.4))
            d_term = (float(d) - 1.8) ** 2
            bd_term = (float(bd) / 10.0) ** 2
            return float(thiolate * math.exp(-(d_term + bd_term)))
        except Exception:
            return np.nan
    df["k_inact_proxy"] = [k_inact(p, d, b) for p, d, b in
                           zip(df["pKa_Cys346"], df["d_SG"], df["burgi_dunitz_dev_deg"])]
    save_working(df)
    update_state("3.8_kinact_refine", "done",
                 n_kinact=int(df["k_inact_proxy"].notna().sum()))
    append_log(f"k_inact_proxy refined ({df['k_inact_proxy'].notna().sum()} non-null)")


# ─────────────── 3.7 MM-GBSA ────────────────────────────────────────────────

def phase_3_7_mmgbsa():
    """Run single-frame MM-GBSA on the 78 CIFs using the same wrapper as 838."""
    df = load_working()
    rids = df["rescue_row_id"].tolist()
    smis = dict(zip(rids, df["smiles"].tolist()))
    index = build_rescue_index()
    have = [r for r in rids if r in index]

    # Write a manifest the wrapper expects: row_id_api, row_id_semantic, smiles, cif_filename
    manifest = pd.DataFrame([
        {"row_id_api": rid, "row_id_semantic": rid, "smiles": smis[rid],
         "cif_filename": f"{rid}_model_0.cif"}
        for rid in have
    ])
    tmp_manifest = ROOT / "data/tier4_scored/rescue_78_mmgbsa_manifest.csv"
    manifest.to_csv(tmp_manifest, index=False)

    # Create a flat dir of CIFs so the wrapper finds them by row_id
    flat_dir = ROOT / "data/tier4_scored/rescue_78_cifs_flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for rid in have:
        src = index[rid] / f"{rid}_model_0.cif"
        dst = flat_dir / f"{rid}_model_0.cif"
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except Exception:
                import shutil; shutil.copy(src, dst)

    out_csv = ROOT / "results/paper_evaluation/rescue_78_mmgbsa.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # MM-GBSA needs the conda env's PATH so antechamber/tleap/sander/MMPBSA.py are available
    cmd_str = (
        "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate quris && "
        f"python {ROOT}/experiments/run_mmgbsa_838.py "
        f"--manifest {tmp_manifest} --cif-dir {flat_dir} --out {out_csv} --workers 4"
    )
    append_log(f"MM-GBSA cmd (shell): {cmd_str}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd_str, shell=True, executable="/bin/bash",
                           capture_output=True, text=True, timeout=14400)
        append_log(f"MM-GBSA rc={r.returncode} wall={time.time()-t0:.0f}s")
        if r.returncode != 0:
            append_log(f"MM-GBSA stderr tail: {r.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        append_log("MM-GBSA TIMED OUT (4h cap)")

    # Merge results back if present
    if out_csv.exists():
        m = pd.read_csv(out_csv)
        keep = ["row_id_semantic" if "row_id_semantic" in m.columns else "row_id",
                "dG_GB_kcalmol", "ggas_kcalmol", "gsolv_kcalmol",
                "E_vdw_kcalmol", "E_eel_kcalmol"]
        keep = [c for c in keep if c in m.columns]
        m_keep = m[keep].rename(columns={keep[0]: "rescue_row_id"})
        # Drop existing cols then merge
        drop = [c for c in m_keep.columns if c in df.columns and c != "rescue_row_id"]
        df = df.drop(columns=drop, errors="ignore")
        df = df.merge(m_keep, on="rescue_row_id", how="left")
        save_working(df)
        n_dG = int(df["dG_GB_kcalmol"].notna().sum()) if "dG_GB_kcalmol" in df.columns else 0
        update_state("3.7_mmgbsa", "done", n_dG=n_dG)
        append_log(f"MM-GBSA merged: {n_dG} rows non-null")
    else:
        update_state("3.7_mmgbsa", "failed", reason="output CSV missing")
        append_log("MM-GBSA failed: no output CSV")


# ─────────────── Composite scores (Phase 1.6 deferred) ─────────────────────

def phase_1_6_composite():
    """Compute desirability components + score using the 78-cohort distribution.

    Uses the Mol1-style v6 formula from compute_mol1_full_features.py but ranks
    within the 78-cohort instead of the 838.
    """
    df = load_working()

    def pctl(s, higher_is_better=True):
        s = pd.to_numeric(s, errors="coerce")
        if not higher_is_better:
            s = -s
        return s.rank(pct=True, method="average").fillna(0.0)

    df["P_potency"] = pctl(df.get("anchor_wins_ge7"), True)
    P_pose_raw = pctl(df.get("mPAE_london"), False)
    iptm_gate = (pd.to_numeric(df.get("boltz_ligand_iptm"), errors="coerce") >= 0.92).fillna(False)
    df["P_pose"] = P_pose_raw.where(iptm_gate, P_pose_raw * 0.5)
    df["P_shape_to_mol1"] = pctl(df.get("shape_Tc_mol1"), True)
    P_dsg = pctl(df.get("d_SG"), False)
    P_bd  = pctl(df.get("burgi_dunitz_dev_deg"), False)
    df["P_covgeom"] = pd.concat([P_dsg, P_bd], axis=1).min(axis=1)
    df["P_clean_reactivity"] = pctl(df.get("pred_log_k2_GSH"), False)
    # P_Tc_to_Mol1 — chemical similarity to Mol1 (already a column)
    df["P_Tc_to_Mol1"] = pctl(df.get("Tc_to_Mol1"), True)

    EPS = 1e-3
    df["desirability_score"] = (
          (df["P_potency"]          + EPS) ** 0.22
        * (df["P_pose"]              + EPS) ** 0.18
        * (df["P_shape_to_mol1"]     + EPS) ** 0.20
        * (df["P_covgeom"]           + EPS) ** 0.10
        * (df["P_clean_reactivity"]  + EPS) ** 0.10
        * (df.get("P_Tec_family_pctl", 0.5) + EPS) ** 0.10
        * (df.get("P_kinase_pctl",    0.5) + EPS) ** 0.05
        * (df.get("pIC50_kinase_aux_pctl", 0.5) + EPS) ** 0.05
    )
    df["combined_score"] = df["desirability_score"]  # alias used in some legacy code
    save_working(df)
    update_state("1.6_composite_scores", "done",
                 mean_des=float(df["desirability_score"].mean()),
                 max_des=float(df["desirability_score"].max()))
    append_log(f"composite scores done (mean des={df['desirability_score'].mean():.3f})")


PHASES = {
    "3.main":     phase_3_main,
    "3.7":        phase_3_7_mmgbsa,
    "1.6":        phase_1_6_composite,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES) + ["all"])
    args = ap.parse_args()
    if args.phase == "all":
        for ph in PHASES:
            print(f"\n=== running phase {ph} ===")
            PHASES[ph]()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
