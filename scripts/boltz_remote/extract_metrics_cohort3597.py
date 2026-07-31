#!/usr/bin/env python3
"""Extract per-mol Boltz metrics from cohort_3597 cofold results.

Adapted from extract_metrics.py to read row_id-based YAML naming and the cohort CSV.
Merges results from all three machines (from_a, from_b, from_c) into one summary.csv.

Inputs:
  - data/tier4_scored/boltz2_cohort_A_relaxed.csv (3,597 mols)
  - data/boltz_results/cohort_3597_full/from_<machine>/results/boltz_results_<row_id>/...

Outputs (under data/boltz_results/cohort_3597_full/):
  - summary.csv (row_id, smiles, method, iptm, ligand_iptm, plddt,
                 confidence_score, d_SG, BD_angle, met414_hinge_dist,
                 success_flag, machine)
  - json/<row_id>.json   (raw confidence JSON, copied)
  - pdb/<row_id>.pdb     (model_0 pdb copy if present, else cif)
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COHORT_CSV = REPO / "data" / "tier4_scored" / "boltz2_cohort_A_relaxed.csv"
BASE = REPO / "data" / "boltz_results" / "cohort_3597_full"
PDB_OUT = BASE / "pdb"
JSON_OUT = BASE / "json"
SUMMARY = BASE / "summary.csv"

MACHINES = [
    ("a", BASE / "from_a"),
    ("b", BASE / "from_b"),
    ("c", BASE / "from_c"),
]


def parse_cif_atoms(cif: Path):
    lines = cif.read_text().splitlines()
    atoms = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().removeprefix("_atom_site."))
                j += 1
            if not cols:
                i = j
                continue
            try:
                ci = {n: cols.index(n) for n in [
                    "auth_asym_id", "label_atom_id", "auth_comp_id",
                    "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "type_symbol",
                ]}
            except ValueError:
                try:
                    ci = {
                        "auth_asym_id": cols.index("label_asym_id"),
                        "label_atom_id": cols.index("label_atom_id"),
                        "auth_comp_id": cols.index("label_comp_id"),
                        "auth_seq_id": cols.index("label_seq_id"),
                        "Cartn_x": cols.index("Cartn_x"),
                        "Cartn_y": cols.index("Cartn_y"),
                        "Cartn_z": cols.index("Cartn_z"),
                        "type_symbol": cols.index("type_symbol"),
                    }
                except ValueError:
                    i = j
                    continue
            i = j
            while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= len(cols):
                    try:
                        atoms.append({
                            "chain": parts[ci["auth_asym_id"]].strip('"'),
                            "atom": parts[ci["label_atom_id"]].strip('"'),
                            "resn": parts[ci["auth_comp_id"]].strip('"'),
                            "resi": int(parts[ci["auth_seq_id"]].strip('"')),
                            "elem": parts[ci["type_symbol"]].strip('"'),
                            "x": float(parts[ci["Cartn_x"]]),
                            "y": float(parts[ci["Cartn_y"]]),
                            "z": float(parts[ci["Cartn_z"]]),
                        })
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue
        i += 1
    return atoms


def vec(a, b):
    return np.array([b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]])


def norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def angle_deg(p1, p2, p3):
    v1 = norm(vec(p2, p1))
    v2 = norm(vec(p2, p3))
    c = np.clip(np.dot(v1, v2), -1, 1)
    return float(np.degrees(np.arccos(c)))


def closest_lig_to_resN(lig, prot, resi):
    bN = [a for a in prot if a["resi"] == resi and a["atom"] == "N"]
    if not bN:
        return None
    N = np.array([bN[0]["x"], bN[0]["y"], bN[0]["z"]])
    dists = []
    for a in lig:
        if a["elem"] == "H":
            continue
        p = np.array([a["x"], a["y"], a["z"]])
        dists.append(float(np.linalg.norm(p - N)))
    return min(dists) if dists else None


def process_one(uid: str, smi: str, method: str, pred_dir: Path, machine: str):
    cif = pred_dir / f"{uid}_model_0.cif"
    conf_json = pred_dir / f"confidence_{uid}_model_0.json"
    pdb = pred_dir / f"{uid}_model_0.pdb"
    if not (cif.exists() and conf_json.exists()):
        return {"row_id": uid, "smiles": smi, "method": method, "machine": machine,
                "success_flag": "missing"}
    try:
        conf = json.loads(conf_json.read_text())
        # Copy raw json + pose to centralized cache
        shutil.copy(conf_json, JSON_OUT / f"{uid}.json")
        if pdb.exists():
            shutil.copy(pdb, PDB_OUT / f"{uid}.pdb")
        else:
            shutil.copy(cif, PDB_OUT / f"{uid}.cif")
        atoms = parse_cif_atoms(cif)
        prot = [a for a in atoms if a["chain"] == "A"]
        lig = [a for a in atoms if a["chain"] == "B"]
        sg = next((a for a in prot if a["resi"] == 346 and a["atom"] == "SG"), None)
        cb = next((a for a in prot if a["resi"] == 346 and a["atom"] == "CB"), None)
        warhead = None
        if sg and lig:
            S = np.array([sg["x"], sg["y"], sg["z"]])
            heavy = [a for a in lig if a["elem"] != "H"]
            heavy_sorted = sorted(heavy, key=lambda a: np.linalg.norm(np.array([a["x"], a["y"], a["z"]]) - S))
            if heavy_sorted:
                warhead = heavy_sorted[0]
        d_SG = None
        BD = None
        if sg and warhead:
            d_SG = float(np.linalg.norm(np.array([sg["x"], sg["y"], sg["z"]]) - np.array([warhead["x"], warhead["y"], warhead["z"]])))
        if cb and sg and warhead:
            BD = angle_deg(cb, sg, warhead)
        met414 = closest_lig_to_resN(lig, prot, 414)
        return {
            "row_id": uid, "smiles": smi, "method": method, "machine": machine,
            "iptm": conf.get("iptm") or conf.get("complex_iptm"),
            "ligand_iptm": conf.get("ligand_iptm"),
            "plddt": conf.get("complex_plddt"),
            "confidence_score": conf.get("confidence_score"),
            "pae_mean": conf.get("complex_pde") or conf.get("pae_mean"),
            "d_SG": d_SG,
            "BD_angle": BD,
            "met414_hinge_dist": met414,
            "success_flag": "ok",
        }
    except Exception as e:
        return {"row_id": uid, "smiles": smi, "method": method, "machine": machine,
                "success_flag": f"err:{type(e).__name__}:{e}"}


def main():
    PDB_OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    # Load cohort metadata
    meta = {}
    with COHORT_CSV.open() as fh:
        for r in csv.DictReader(fh):
            meta[str(r["row_id"]).strip()] = {
                "smi": r["smiles"].strip(),
                "method": r.get("method", "").strip(),
            }
    print(f"Loaded {len(meta)} cohort rows")

    rows = []
    n_ok = n_fail = n_missing = 0
    for machine, base in MACHINES:
        results_dir = base / "results"
        if not results_dir.exists():
            print(f"Skip {machine}: no results dir at {results_dir}")
            continue
        # Each subdir is boltz_results_<uid>/predictions/<uid>/...
        for boltz_dir in sorted(results_dir.glob("boltz_results_*")):
            uid = boltz_dir.name.removeprefix("boltz_results_")
            if uid not in meta:
                continue
            pred_dir = boltz_dir / "predictions" / uid
            md = meta[uid]
            res = process_one(uid, md["smi"], md["method"], pred_dir, machine)
            rows.append(res)
            sf = res["success_flag"]
            if sf == "ok":
                n_ok += 1
            elif sf == "missing":
                n_missing += 1
            else:
                n_fail += 1

    # Add absent rows
    seen = {r["row_id"] for r in rows}
    for uid, md in meta.items():
        if uid not in seen:
            rows.append({
                "row_id": uid, "smiles": md["smi"], "method": md["method"],
                "machine": "", "success_flag": "absent",
            })
            n_missing += 1

    fields = ["row_id", "smiles", "method", "machine",
              "iptm", "ligand_iptm", "plddt", "pae_mean", "confidence_score",
              "d_SG", "BD_angle", "met414_hinge_dist", "success_flag"]
    with SUMMARY.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {SUMMARY}  ok={n_ok} fail={n_fail} missing={n_missing} total={len(rows)}")


if __name__ == "__main__":
    main()
