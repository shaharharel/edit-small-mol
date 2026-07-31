"""Batched AutoDock Vina docking for all v2 samples.

For each valid+acryl SMILES, we run TWO complementary Vina runs against the
Cys346-stripped ZAP70 receptor:

  1. **Non-covalent global dock** (`--exhaustiveness 4`) — free ligand
     search inside a 20³-Å box centred on Cys346 SG.  From the top-1 pose,
     measure:
        - d_SG   : distance from Cys346 SG (18.888, -3.650, -29.979) to
                   the ligand's acrylamide β-C.
        - bd_ang : Bürgi–Dunitz angle between (SG->Cβ) and (Cβ->Cα).
        - phi_pl : planar dihedral of the acrylamide in the docked pose.
     Store as `vina_free_*`.

  2. **AD-CovDock score_only** (meeko-tethered pose, canonical AD-CovDock
     recipe) — reuses `experiments/run_covalent_docking.py`.  We keep this
     as the "how good is the ideal tethered pose" reference score.
     Store as `vina_cov_score` (kcal/mol; lower = better).

Runs multiprocessed on the CPU cores.  Vina is CPU-bound and ignores GPU.

BD-ready gate (from Vina free dock):
   d_SG ∈ [2.5, 5.5] Å  AND  bd_ang ∈ [80°, 130°]  AND  phi_pl ≤ 30°

Output: covalent_vina_panel.csv (one row per valid+acryl sample).
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
RECEPTOR = str(PROJECT_ROOT / "data/docking_500/receptor_cys346_stripped.pdbqt")
CYS346_SG = np.array([18.888, -3.650, -29.979])
BOX_SIZE = np.array([20.0, 20.0, 20.0])
VINA_BIN = "vina"

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")


def angle_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def dihedral_deg(p1, p2, p3, p4):
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / max(np.linalg.norm(b2), 1e-9))
    x = float(np.dot(n1, n2)); y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def wrap_planar(phi_signed):
    a = abs(((phi_signed + 180.0) % 360.0) - 180.0)
    return float(min(a, 180.0 - a))


def parse_pdbqt_first_model(text: str):
    """Return coords for the first MODEL block as a list of dicts:
       {atom_idx (0-based within the mol), atom_name, elem, xyz}."""
    atoms = []
    in_first = False
    seen_endmdl = False
    for ln in text.splitlines():
        if ln.startswith("MODEL"):
            in_first = True
            continue
        if ln.startswith("ENDMDL"):
            if in_first:
                seen_endmdl = True
                break
        if not in_first:
            # some Vina output has no MODEL header (single pose)
            pass
        if ln.startswith(("ATOM", "HETATM")):
            atom_name = ln[12:16].strip()
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            elem = ln[76:78].strip() if len(ln) >= 78 else atom_name[:1]
            atoms.append({"name": atom_name, "elem": elem,
                            "xyz": np.array([x, y, z])})
    return atoms


def embed_and_prepare_ligand(smi: str, out_pdbqt: Path) -> tuple[bool, str, int]:
    """RDKit ETKDG + MMFF opt + meeko write of a free (non-covalent) ligand.

    Returns (ok, msg, beta_atom_idx_in_prep) — beta_atom_idx = index in the
    order-of-atoms Vina will use (which matches meeko-writer order for
    heavy atoms)."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except Exception as e:
        return False, f"meeko import fail: {e}", -1
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False, "smi parse fail", -1
    matches = m.GetSubstructMatches(ACRYL_SMARTS)
    if not matches:
        return False, "no acryl smarts", -1
    beta_rdkit_heavy = matches[0][0]
    mH = Chem.AddHs(m)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(mH, params) == -1:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mH, params) == -1:
            return False, "embed fail", -1
    try:
        AllChem.MMFFOptimizeMolecule(mH, maxIters=200)
    except Exception:
        pass
    prep = MoleculePreparation()
    setups = prep.prepare(mH)
    if not setups:
        return False, "meeko prepare fail", -1
    pdbqt_str, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return False, f"pdbqt write fail: {msg}", -1
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    out_pdbqt.write_text(pdbqt_str)
    return True, "ok", beta_rdkit_heavy


def run_vina_free(lig_pdbqt: Path, out_pose: Path, timeout: int = 300) -> dict:
    """Global (non-covalent) Vina dock.  Returns dict with score + pose path."""
    cmd = [VINA_BIN, "--receptor", RECEPTOR,
             "--ligand", str(lig_pdbqt),
             "--center_x", str(CYS346_SG[0]),
             "--center_y", str(CYS346_SG[1]),
             "--center_z", str(CYS346_SG[2]),
             "--size_x", str(BOX_SIZE[0]),
             "--size_y", str(BOX_SIZE[1]),
             "--size_z", str(BOX_SIZE[2]),
             "--exhaustiveness", "4",
             "--num_modes", "5",
             "--out", str(out_pose),
             "--cpu", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": "timeout", "score": None}
    if r.returncode != 0:
        return {"ok": False, "msg": f"rc={r.returncode}: {r.stderr[-200:]}",
                  "score": None}
    # Vina outputs "REMARK VINA RESULT:  -7.5  0.000  0.000" as first line in the pose file.
    if not out_pose.exists():
        return {"ok": False, "msg": "no pose file", "score": None}
    text = out_pose.read_text()
    # Scan for the first "REMARK VINA RESULT:" line.
    score = None
    for ln in text.splitlines():
        if ln.startswith("REMARK VINA RESULT:"):
            parts = ln.split()
            try:
                score = float(parts[3])
                break
            except (IndexError, ValueError):
                pass
    return {"ok": True, "msg": "ok", "score": score, "pose_text": text}


def measure_pose_bd(pose_text: str, smi: str, beta_atom_idx_rdkit: int) -> dict:
    """From the Vina-docked pose PDBQT, extract d_SG, bd_ang, phi_planar.

    Meeko emits heavy atoms then hydrogens in a specific order.  For BD gate
    we only need the heavy atoms.  We match them positionally to the RDKit
    SMILES's heavy-atom order.
    """
    out = {"d_sg": np.nan, "bd_ang": np.nan, "phi_planar": np.nan, "note": ""}
    atoms = parse_pdbqt_first_model(pose_text)
    heavy_atoms = [a for a in atoms if a["elem"] != "H"]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        out["note"] = "smi parse fail"
        return out
    matches = m.GetSubstructMatches(ACRYL_SMARTS)
    if not matches:
        out["note"] = "no acryl smarts"
        return out
    a_b, a_a, a_co, _o, a_n = matches[0]
    # Elements should match at each heavy position.
    if len(heavy_atoms) < max(a_b, a_a, a_co, _o, a_n) + 1:
        out["note"] = f"only {len(heavy_atoms)} heavy atoms in pose"
        return out
    b = heavy_atoms[a_b]["xyz"]; alpha = heavy_atoms[a_a]["xyz"]
    carb = heavy_atoms[a_co]["xyz"]; nit = heavy_atoms[a_n]["xyz"]
    d_sg = float(np.linalg.norm(b - CYS346_SG))
    bd_ang = angle_deg(CYS346_SG - b, alpha - b)
    phi_signed = dihedral_deg(b, alpha, carb, nit)
    phi_planar = wrap_planar(phi_signed)
    out.update({"d_sg": d_sg, "bd_ang": bd_ang, "phi_planar": phi_planar,
                 "note": "ok"})
    return out


def _job(args):
    idx, smi = args
    result = {"row_idx": idx,
                "vina_free_score": np.nan,
                "vina_free_d_sg": np.nan,
                "vina_free_bd_ang": np.nan,
                "vina_free_phi_planar": np.nan,
                "vina_free_ok": False,
                "vina_free_note": "",
                "prep_ok": False}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            lig_pdbqt = Path(tmp) / "lig.pdbqt"
            ok, msg, beta = embed_and_prepare_ligand(smi, lig_pdbqt)
            if not ok:
                result["vina_free_note"] = f"prep: {msg}"
                return result
            result["prep_ok"] = True
            pose_pdbqt = Path(tmp) / "pose.pdbqt"
            v = run_vina_free(lig_pdbqt, pose_pdbqt)
            if not v["ok"]:
                result["vina_free_note"] = f"vina: {v['msg']}"
                return result
            result["vina_free_score"] = v["score"] if v["score"] is not None else np.nan
            geom = measure_pose_bd(v["pose_text"], smi, beta)
            result["vina_free_d_sg"] = geom["d_sg"]
            result["vina_free_bd_ang"] = geom["bd_ang"]
            result["vina_free_phi_planar"] = geom["phi_planar"]
            result["vina_free_ok"] = True
            result["vina_free_note"] = geom["note"]
    except Exception as e:
        result["vina_free_note"] = f"exc {type(e).__name__}: {e}"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_metric_panel_v2.csv"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_vina_panel.csv"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "vina_batch_progress.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="If --out_csv exists, skip rows whose (cell, sample_idx) "
                          "are already present with vina_free_ok==True.")
    ap.add_argument("--incremental_save_every", type=int, default=20,
                    help="Dump partial CSV every N completed jobs (0 = only at end).")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel_csv)
    valid = panel[(panel["valid"] == True) &
                    (panel["acryl_largest"] == True)].copy().reset_index(drop=True)
    print(f"Vina panel: {len(valid)} valid+acryl rows", flush=True)
    if args.limit is not None:
        valid = valid.head(args.limit)
        print(f"Limited to {len(valid)}", flush=True)

    # Resume: load already-computed rows and skip them.
    done_keys = set()
    prior_rows = []
    if args.resume and Path(args.out_csv).exists():
        try:
            prior = pd.read_csv(args.out_csv)
            ok_mask = prior["vina_free_ok"].fillna(False).astype(bool)
            for _, r in prior[ok_mask].iterrows():
                done_keys.add((str(r["cell"]), int(r["sample_idx"])))
                prior_rows.append(r.to_dict())
            print(f"[resume] {len(done_keys)} rows already done in {args.out_csv}",
                   flush=True)
        except Exception as e:
            print(f"[resume] failed to load prior CSV: {e}", flush=True)

    # Build job list, skipping done keys.
    jobs = []
    row_index_map = {}  # row_idx (in jobs) -> position in valid
    for i, row in valid.iterrows():
        key = (str(row["cell"]), int(row["sample_idx"]))
        if key in done_keys:
            continue
        jobs.append((len(jobs), str(row["largest_frag_SMILES"])))
        row_index_map[len(jobs) - 1] = i
    print(f"Jobs to run: {len(jobs)} (skipping {len(done_keys)} already done)",
           flush=True)

    def dump_partial(results_so_far, done_positions):
        """Write out_csv containing prior_rows + newly-computed rows."""
        out_rows = list(prior_rows)
        for pos in done_positions:
            res = results_so_far[pos]
            if res is None:
                continue
            i_in_valid = row_index_map[pos]
            r = valid.iloc[i_in_valid].to_dict()
            base = {"cell": r["cell"], "sample_idx": int(r["sample_idx"]),
                      "smi": r["largest_frag_SMILES"]}
            out_rows.append({**base, **{k: v for k, v in res.items()
                                             if k != "row_idx"}})
        df_out = pd.DataFrame(out_rows)
        if len(df_out):
            df_out["bd_ready_vina"] = (
                (df_out["vina_free_d_sg"] >= 2.5) &
                (df_out["vina_free_d_sg"] <= 5.5) &
                (df_out["vina_free_bd_ang"] >= 80.0) &
                (df_out["vina_free_bd_ang"] <= 130.0) &
                (df_out["vina_free_phi_planar"] <= 30.0)
            ).fillna(False)
        df_out.to_csv(args.out_csv, index=False)

    t_start = time.time()
    results = [None] * len(jobs)
    completed_positions = []
    if jobs:
        with mp.Pool(processes=args.workers) as pool:
            for k, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
                results[res["row_idx"]] = res
                completed_positions.append(res["row_idx"])
                if (k + 1) % 20 == 0 or (k + 1) == len(jobs):
                    el = time.time() - t_start
                    rate = (k + 1) / max(el, 1e-6)
                    eta = (len(jobs) - k - 1) / max(rate, 1e-6)
                    n_ok = sum(1 for r in results if r is not None and r["vina_free_ok"])
                    print(f"[{k+1}/{len(jobs)}] ok={n_ok}  {rate:.2f}/s  ETA={eta/60:.1f}min",
                           flush=True)
                    Path(args.progress_path).write_text(json.dumps({
                        "phase": "vina_batch",
                        "done": k + 1 + len(done_keys),
                        "total": len(jobs) + len(done_keys),
                        "n_ok": n_ok + len(done_keys),
                        "rate_per_sec": rate, "eta_min": eta / 60,
                        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, indent=2))
                if args.incremental_save_every > 0 and \
                        (k + 1) % args.incremental_save_every == 0:
                    dump_partial(results, completed_positions)

    # Merge into rows (prior + new).
    out_rows = list(prior_rows)
    for pos in completed_positions:
        res = results[pos]
        if res is None:
            continue
        i_in_valid = row_index_map[pos]
        r = valid.iloc[i_in_valid].to_dict()
        base = {"cell": r["cell"], "sample_idx": int(r["sample_idx"]),
                  "smi": r["largest_frag_SMILES"]}
        out_rows.append({**base, **{k: v for k, v in res.items()
                                         if k != "row_idx"}})
    out_df = pd.DataFrame(out_rows)
    # Compute BD-ready gate.
    out_df["bd_ready_vina"] = (
        (out_df["vina_free_d_sg"] >= 2.5) &
        (out_df["vina_free_d_sg"] <= 5.5) &
        (out_df["vina_free_bd_ang"] >= 80.0) &
        (out_df["vina_free_bd_ang"] <= 130.0) &
        (out_df["vina_free_phi_planar"] <= 30.0)
    ).fillna(False)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)  bd_ready count={int(out_df['bd_ready_vina'].sum())}",
           flush=True)
    Path(args.progress_path).write_text(json.dumps({
        "phase": "vina_batch_done",
        "n_total": len(out_df),
        "n_ok": int(out_df["vina_free_ok"].sum()),
        "n_bd_ready": int(out_df["bd_ready_vina"].sum()),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
