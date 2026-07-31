#!/usr/bin/env python3
"""Run AutoDock Vina for ONE cohort SDF against ZAP70 (4K2R) and compute the
9 metrics required for the pocket+covalent-aware comparison.

Designed to be invoked many times in parallel (one process per cohort, or
multiple shards across machines). Uses an internal multiprocessing Pool of N
workers (each Vina run uses 2 cpu threads, so 4 workers ≈ 8 cores).

Usage:
    conda run --no-capture-output -n quris python -u \
        experiments/cohort_comparison_dock.py \
        --cohort H2 \
        --input-sdf data/cohort_comparison/cohorts/H2/input.sdf \
        --out-dir results/paper_evaluation/cohort_comparison/per_cohort \
        --workers 4 --threads 2

Outputs (under out-dir/cohort/):
    per_mol.csv     # one row per molecule with 9 metrics + dock metadata
    poses/*.pdbqt   # docked poses (one file per mol)
    top10.sdf       # top-10 by Vina score, with metadata

Receptor / box / hinge / Cys346 references are kept in this file (hard-coded
from the existing 4K2R setup).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent

# --- Vina binary discovery (Mac local OR Linux ai-chem) ---
def _find_vina():
    for cand in [
        PROJECT_ROOT / "tools" / "vina",
        Path("/usr/bin/vina"),
        Path("/usr/local/bin/vina"),
        shutil.which("vina") and Path(shutil.which("vina")),
    ]:
        if cand and Path(cand).exists():
            return str(cand)
    return None

# --- Pocket / receptor (ZAP70 4K2R) ---
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"
RECEPTOR_PDB = PROJECT_ROOT / "data" / "docking_500" / "receptor_clean.pdb"
CYS346_SG = np.array([18.888, -3.650, -29.979])
MET414_N = np.array([1.671, -5.312, -27.925])
MET414_C = np.array([1.313, -3.739, -26.006])
# Hinge proxies (ZAP70: Met414 N is hinge N-H donor; Glu413 backbone O is hinge acceptor)
HINGE_DONORS = [MET414_N]              # backbone NH
HINGE_ACCEPTORS = [MET414_C]           # use carbonyl C as proxy for O (within 1.2 A)
HINGE_HBOND_DIST_MAX = 5.0             # A — relaxed because Met414 lies ~17A from box center;
                                       # Vina lattice search keeps mol COG inside box, but the
                                       # recognition arm can extend outward and reach Met414 in
                                       # large mols. 5.0 A captures these "hinge-pointing" poses.
# Expected Cβ attack position: 1.85 A from SG along the attack vector (using a
# very rough proxy: 1.85A from SG in the direction toward the pocket centroid
# from SG). For simplicity, the "expected" is just 1.85A — we measure ABS
# distance from observed Cβ to a sphere of that radius (i.e. d_diff_from_185)
EXPECTED_CB_SG_DIST = 1.85  # A
BD_TARGET_DEG = 107.0

BOX_CENTER = CYS346_SG.copy()
BOX_SIZE = np.array([20.0, 20.0, 20.0])

EXHAUSTIVENESS = 4
NUM_MODES = 9
TIMEOUT_S = 600

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"


# ---------------- Ligand preparation ----------------

def mol_to_pdbqt(mol, out_path: Path) -> bool:
    try:
        mol_h = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol_h)
        pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if is_ok:
            out_path.write_text(pdbqt_str)
            return True
    except Exception:
        return False
    return False


def reembed_and_pdbqt(smiles: str, out_path: Path, seed: int = 42) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1:
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1:
            return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol_to_pdbqt(mol, out_path)


# ---------------- PDBQT pose parsing ----------------

def parse_pdbqt_scores(pose_path: Path):
    if not pose_path.exists():
        return {"vina_score": None, "vina_all": []}
    scores = []
    for line in pose_path.read_text().split("\n"):
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        scores.append(float(parts[i + 1]))
                    except Exception:
                        pass
                    break
    if not scores:
        return {"vina_score": None, "vina_all": []}
    return {"vina_score": scores[0], "vina_all": scores}


def parse_pdbqt_all_models(pose_path: Path):
    """Parse all MODEL blocks; return list of dicts {coords, elements, score}."""
    if not pose_path.exists():
        return []
    out = []
    cur_coords = []
    cur_elems = []
    cur_score = None
    in_model = False
    txt = pose_path.read_text()
    for line in txt.split("\n"):
        if line.startswith("MODEL"):
            in_model = True
            cur_coords = []
            cur_elems = []
            cur_score = None
            continue
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        cur_score = float(parts[i + 1])
                    except Exception:
                        pass
        if line.startswith("ENDMDL"):
            if cur_coords:
                out.append({
                    "coords": np.array(cur_coords, dtype=float),
                    "elements": list(cur_elems),
                    "score": cur_score,
                })
            in_model = False
            continue
        if in_model and (line.startswith("ATOM") or line.startswith("HETATM")):
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                ad_type = line[77:79].strip() if len(line) >= 79 else ""
                elem = ad_type[0] if ad_type else line[12:14].strip()[0]
                cur_coords.append([x, y, z])
                cur_elems.append(elem)
            except Exception:
                pass
    return out


# ---------------- Geometry helpers ----------------

def find_warhead_indices(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    m = matches[0]
    return {
        "ch2_terminal": m[0],
        "ch_beta": m[1],
        "c_carbonyl": m[2],
        "n_amide": m[3],
        "all_warhead": list(m),
    }


def acrylamide_on_largest_frag(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return False
        largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
        return largest.HasSubstructMatch(patt)
    except Exception:
        return False


def map_mol_to_pose_indices(mol, pose_elements):
    """Fallback: match RDKit heavy atom sequence to PDBQT heavy atom sequence.
    Returns mol_idx -> pose_abs_idx map, or None on mismatch.

    Treats 'A' (aromatic C) as 'C' for element comparison.
    """
    heavy_rdkit = [a.GetSymbol() for a in mol.GetAtoms()]
    norm = lambda e: "C" if e == "A" else e
    pose_heavy_idx = [i for i, e in enumerate(pose_elements) if e != "H"]
    pose_heavy_elems = [norm(pose_elements[i]) for i in pose_heavy_idx]
    if len(pose_heavy_elems) != len(heavy_rdkit):
        return None
    if not all(a == b for a, b in zip(pose_heavy_elems, heavy_rdkit)):
        return None
    return {i: pose_heavy_idx[i] for i in range(len(heavy_rdkit))}


def parse_meeko_smiles_idx(pdbqt_path: Path):
    """Parse meeko's REMARK SMILES IDX lines from a ligand PDBQT.

    Meeko writes the mapping across MULTIPLE 'REMARK SMILES IDX' lines (one per
    line ~10 atoms). We concatenate them all.

    Returns dict: smi_idx_0based -> pdbqt_atom_num_1based (all-atom serial,
    NOT heavy-only index). Caller is responsible for converting via the
    serial→parsed_idx remap if measuring against H-filtered coords.

    Returns None if no REMARK lines present.
    """
    if not pdbqt_path.exists():
        return None
    all_ints = []
    found = False
    for line in pdbqt_path.read_text().split("\n"):
        if line.startswith("REMARK SMILES IDX"):
            found = True
            parts = line.split()
            for p in parts[3:]:
                try:
                    all_ints.append(int(p))
                except ValueError:
                    pass
    if not found:
        return None
    if len(all_ints) % 2 != 0:
        return None
    mapping = {}
    for i in range(0, len(all_ints), 2):
        smi_idx_1b = all_ints[i]
        pdbqt_num_1b = all_ints[i + 1]
        mapping[smi_idx_1b - 1] = pdbqt_num_1b
    return mapping


def _serial_to_parsed_idx(pdbqt_path: Path) -> dict:
    """Build serial(1b, all atoms) → parsed_idx(0b, heavy-only) mapping.

    `parse_pdbqt_all_models` filters out HD/H atoms, so a PDBQT atom serial of
    N (1-based, counting Hs) corresponds to parsed_idx = N - 1 - (number of
    Hs before serial N). This helper returns the explicit map.

    Bug history: prior to 2026-06-01, downstream consumers used `serial - 1`
    naively. Mostly safe for Lingo cohorts whose meeko output places explicit
    Hs AFTER the β-C; broken for LibInvent / Amine / DeNovo / Mol2Mol where
    Hs appear before the β-C.
    """
    if not pdbqt_path.exists():
        return {}
    serial_to_parsed = {}
    parsed_idx = 0
    serial = 0
    for ln in pdbqt_path.read_text().split("\n"):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        serial += 1
        ad_type = ln[77:79].strip() if len(ln) >= 79 else ""
        if ad_type in ("HD", "H"):
            continue
        serial_to_parsed[serial] = parsed_idx
        parsed_idx += 1
    return serial_to_parsed


def compute_pose_metrics(pose_path: Path, lig_pdbqt_path: Path, smiles: str):
    """Compute pose metrics using the meeko REMARK-based SMILES atom mapping
    (fall back to element-sequence mapping if meeko REMARK absent).
    """
    poses = parse_pdbqt_all_models(pose_path)
    out = {
        "n_poses": len(poses),
        "warhead_found": False,
        "d_cb_sg_top1": None,
        "d_cb_sg_diff_from_185": None,
        "bd_angle_top1": None,
        "hinge_hbond_top1": 0,
        "d_hinge_top1": None,
        "clash_count_top1": None,
        "any_pose_feasible": 0,
    }
    if not poses:
        return out
    top = poses[0]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    wh = find_warhead_indices(smiles)

    # Atom mapping: prefer meeko REMARK SMILES IDX.
    # FIX 2026-06-01: meeko REMARK serials count ALL atoms (incl. polar H/HD),
    # but parse_pdbqt_all_models filters Hs out. So `serial - 1` only works
    # when no Hs appear before the β-C; otherwise the index is off by N. Use
    # the explicit serial→parsed_idx remap built from the ligand PDBQT itself.
    meeko_map = parse_meeko_smiles_idx(lig_pdbqt_path)
    if meeko_map:
        serial_to_parsed = _serial_to_parsed_idx(lig_pdbqt_path)
        if serial_to_parsed:
            mapping = {
                smi_idx: serial_to_parsed[pdbqt_num]
                for smi_idx, pdbqt_num in meeko_map.items()
                if pdbqt_num in serial_to_parsed
            }
        else:
            # No PDBQT to remap against — fall back to naive serial-1
            mapping = {smi_idx: pdbqt_num - 1 for smi_idx, pdbqt_num in meeko_map.items()}
    else:
        mapping = map_mol_to_pose_indices(mol, top["elements"])

    if wh and mapping and (wh["ch_beta"] in mapping) and (wh["ch2_terminal"] in mapping):
        out["warhead_found"] = True
        cbeta_idx = mapping[wh["ch_beta"]]
        cterm_idx = mapping[wh["ch2_terminal"]]
        if cbeta_idx < len(top["coords"]) and cterm_idx < len(top["coords"]):
            cb_xyz = top["coords"][cbeta_idx]
            ct_xyz = top["coords"][cterm_idx]
            d_sg = float(np.linalg.norm(cb_xyz - CYS346_SG))
            out["d_cb_sg_top1"] = d_sg
            out["d_cb_sg_diff_from_185"] = abs(d_sg - EXPECTED_CB_SG_DIST)
            v1 = CYS346_SG - cb_xyz
            v2 = ct_xyz - cb_xyz
            nv1 = np.linalg.norm(v1); nv2 = np.linalg.norm(v2)
            if nv1 > 1e-6 and nv2 > 1e-6:
                cosang = float(np.dot(v1, v2) / (nv1 * nv2))
                cosang = max(-1.0, min(1.0, cosang))
                out["bd_angle_top1"] = float(np.degrees(np.arccos(cosang)))

    # Hinge proximity: minimum distance from ANY ligand polar atom (N, O, F) to
    # the Met414 hinge (N or C). The 20-A pocket box keeps ligand COG within ~10 A
    # of Cys346 SG, leaving ~7 A between box edge and Met414 N. So we record the
    # MINIMUM distance as a continuous metric, plus a binary hinge_hbond_top1
    # using a generous 5 A threshold (for completeness).
    coords = top["coords"]
    elems = top["elements"]
    polar_mask = np.array([e in ("N", "O", "F", "OA", "NA") for e in elems])
    if polar_mask.sum() > 0:
        polar = coords[polar_mask]
        d_to_N = float(np.linalg.norm(polar - MET414_N, axis=1).min())
        d_to_C = float(np.linalg.norm(polar - MET414_C, axis=1).min())
        d_min = min(d_to_N, d_to_C)
        out["d_hinge_top1"] = d_min
        out["hinge_hbond_top1"] = int(d_min < HINGE_HBOND_DIST_MAX)
    else:
        out["d_hinge_top1"] = None

    # Scan all poses for BD + d_SG feasibility
    if wh and mapping:
        for p in poses:
            cbi = mapping.get(wh["ch_beta"])
            cti = mapping.get(wh["ch2_terminal"])
            if cbi is None or cti is None:
                continue
            if cbi >= len(p["coords"]) or cti >= len(p["coords"]):
                continue
            cb = p["coords"][cbi]
            ct = p["coords"][cti]
            d_sg = float(np.linalg.norm(cb - CYS346_SG))
            v1 = CYS346_SG - cb; v2 = ct - cb
            nv1 = np.linalg.norm(v1); nv2 = np.linalg.norm(v2)
            if nv1 < 1e-6 or nv2 < 1e-6:
                continue
            cosang = max(-1.0, min(1.0, float(np.dot(v1, v2) / (nv1 * nv2))))
            bd = float(np.degrees(np.arccos(cosang)))
            if 1.55 <= d_sg <= 2.15 and 102.0 <= bd <= 112.0:
                out["any_pose_feasible"] = 1
                break
    return out


# ---------------- Pocket atom loader (for clash count) ----------------

def load_pocket_atoms(receptor_pdb: Path, center: np.ndarray, radius: float = 12.0):
    """Return Nx3 array of heavy-atom coords within `radius` A of `center`."""
    if not receptor_pdb.exists():
        return None
    coords = []
    for line in receptor_pdb.read_text().split("\n"):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            elem = line[76:78].strip() or line[12:14].strip()[0]
        except Exception:
            continue
        if elem.upper() == "H":
            continue
        if np.linalg.norm(np.array([x, y, z]) - center) <= radius:
            coords.append([x, y, z])
    return np.array(coords) if coords else None


# ---------------- Single docking job ----------------

def dock_one(args):
    (cohort, idx, name, smi, mol_block, vina_bin, lig_dir, pose_dir,
     receptor_pdbqt, threads) = args
    lig_pdbqt = lig_dir / f"{name}.pdbqt"
    pose_pdbqt = pose_dir / f"{name}_pose.pdbqt"
    cache = pose_dir / f"{name}_meta.json"

    res = {
        "cohort": cohort, "mol_idx": idx, "name": name, "smi": smi,
        "vina_score": None, "vina_all": [],
        "dock_time_s": 0.0, "success": False, "error": None,
    }

    if cache.exists():
        try:
            r = json.loads(cache.read_text())
            res.update(r)
            res["success"] = r.get("vina_score") is not None
            return res
        except Exception:
            pass

    # Prepare ligand PDBQT
    if not lig_pdbqt.exists():
        ok = False
        if mol_block:
            try:
                m = Chem.MolFromMolBlock(mol_block, removeHs=False)
                if m is not None:
                    ok = mol_to_pdbqt(m, lig_pdbqt)
            except Exception:
                ok = False
        if not ok:
            ok = reembed_and_pdbqt(smi, lig_pdbqt)
        if not ok:
            res["error"] = "ligand_prep_failed"
            cache.write_text(json.dumps(res, indent=2))
            return res

    cmd = [
        vina_bin,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(lig_pdbqt),
        "--center_x", f"{BOX_CENTER[0]:.3f}",
        "--center_y", f"{BOX_CENTER[1]:.3f}",
        "--center_z", f"{BOX_CENTER[2]:.3f}",
        "--size_x", f"{BOX_SIZE[0]:.1f}",
        "--size_y", f"{BOX_SIZE[1]:.1f}",
        "--size_z", f"{BOX_SIZE[2]:.1f}",
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--cpu", str(threads),
        "--out", str(pose_pdbqt),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        res["dock_time_s"] = time.time() - t0
        parsed = parse_pdbqt_scores(pose_pdbqt)
        res["vina_score"] = parsed["vina_score"]
        res["vina_all"] = parsed["vina_all"]
        if res["vina_score"] is not None:
            res["success"] = True
        elif proc.returncode != 0:
            res["error"] = f"vina_exit_{proc.returncode}"
        else:
            res["error"] = "no_score_parsed"
    except subprocess.TimeoutExpired:
        res["dock_time_s"] = time.time() - t0
        res["error"] = f"timeout_{TIMEOUT_S}s"
    except Exception as e:
        res["dock_time_s"] = time.time() - t0
        res["error"] = str(e)[:200]

    cache.write_text(json.dumps(res, indent=2))
    return res


# ---------------- Cohort driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--input-sdf", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--receptor-pdbqt", type=Path, default=RECEPTOR_PDBQT)
    ap.add_argument("--receptor-pdb", type=Path, default=RECEPTOR_PDB)
    args = ap.parse_args()

    vina = _find_vina()
    if not vina:
        print("ERROR: Vina not found")
        sys.exit(1)
    if not args.receptor_pdbqt.exists():
        print(f"ERROR: receptor PDBQT missing: {args.receptor_pdbqt}")
        sys.exit(1)
    if not args.input_sdf.exists():
        print(f"ERROR: input SDF missing: {args.input_sdf}")
        sys.exit(1)

    cohort_out = args.out_dir / args.cohort
    lig_dir = cohort_out / "ligands_pdbqt"
    pose_dir = cohort_out / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Load mols
    supp = Chem.SDMolSupplier(str(args.input_sdf), removeHs=False, sanitize=True)
    jobs = []
    rec_meta = []
    for i, mol in enumerate(supp):
        if mol is None:
            continue
        try:
            smi = mol.GetProp("smi") if mol.HasProp("smi") else Chem.MolToSmiles(mol)
        except Exception:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{args.cohort}_{i}"
        try:
            mb = Chem.MolToMolBlock(mol)
        except Exception:
            mb = None
        jobs.append((args.cohort, i, name, smi, mb, vina, lig_dir, pose_dir,
                     args.receptor_pdbqt, args.threads))
        rec_meta.append({"mol_idx": i, "name": name, "smi": smi})

    print(f"[{args.cohort}] Docking {len(jobs)} mols, {args.workers} workers x {args.threads} threads, vina={vina}")
    t0 = time.time()
    results = []
    with Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(dock_one, jobs), start=1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                n_ok = sum(1 for x in results if x["success"])
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - i) / rate if rate > 0 else 0
                print(f"  [{args.cohort}] [{i}/{len(jobs)}] ok={n_ok} rate={rate:.2f}/s ETA={eta/60:.1f}min", flush=True)
    total = time.time() - t0
    print(f"[{args.cohort}] Docked in {total/60:.1f} min")

    # Geometry & metrics
    pocket_coords = load_pocket_atoms(args.receptor_pdb, BOX_CENTER, radius=12.0)
    print(f"[{args.cohort}] Pocket atoms loaded: {0 if pocket_coords is None else len(pocket_coords)}")

    # Build a smi -> input-pose-Cb mapping from the SDF (only meaningful for
    # cohorts whose input.sdf carries a 3D pose; for ETKDG-embedded baselines
    # this measures the "ETKDG arbitrary" coords, which we'll record but not
    # as a covalent claim).
    name_to_input_geo = {}
    supp = Chem.SDMolSupplier(str(args.input_sdf), removeHs=False, sanitize=True)
    for i, m in enumerate(supp):
        if m is None: continue
        try:
            smi = m.GetProp("smi") if m.HasProp("smi") else Chem.MolToSmiles(m)
            n = m.GetProp("_Name") if m.HasProp("_Name") else f"{args.cohort}_{i}"
            wh = find_warhead_indices(smi)
            if not wh or m.GetNumConformers() == 0:
                continue
            conf = m.GetConformer()
            cb = conf.GetAtomPosition(wh["ch_beta"])
            ct = conf.GetAtomPosition(wh["ch2_terminal"])
            cb_np = np.array([cb.x, cb.y, cb.z])
            ct_np = np.array([ct.x, ct.y, ct.z])
            d_sg = float(np.linalg.norm(cb_np - CYS346_SG))
            v1 = CYS346_SG - cb_np; v2 = ct_np - cb_np
            nv1 = np.linalg.norm(v1); nv2 = np.linalg.norm(v2)
            bd = None
            if nv1 > 1e-6 and nv2 > 1e-6:
                cosang = max(-1.0, min(1.0, float(np.dot(v1, v2) / (nv1 * nv2))))
                bd = float(np.degrees(np.arccos(cosang)))
            name_to_input_geo[n] = {
                "d_cb_sg_input": d_sg,
                "bd_angle_input": bd,
            }
        except Exception:
            continue

    rows = []
    for r in results:
        smi = r["smi"]
        wh_largest = acrylamide_on_largest_frag(smi)
        mol = Chem.MolFromSmiles(smi) if smi else None
        n_heavy = mol.GetNumHeavyAtoms() if mol else 0
        input_geo = name_to_input_geo.get(r["name"], {})
        row = {
            "cohort": args.cohort,
            "mol_idx": r["mol_idx"],
            "name": r["name"],
            "smi": smi,
            "n_heavy_atoms": n_heavy,
            "vina_score": r["vina_score"],
            "vina_all": ";".join(f"{x:.2f}" for x in r["vina_all"]),
            "dock_time_s": round(r["dock_time_s"], 2),
            "success": r["success"],
            "error": r["error"],
            "warhead_largest_frag": int(wh_largest),
            # input (pre-dock) pose geometry — meaningful for cohorts with 3D-designed poses
            "d_cb_sg_input": input_geo.get("d_cb_sg_input"),
            "bd_angle_input": input_geo.get("bd_angle_input"),
            "d_cb_sg_input_diff_from_185": (
                abs(input_geo["d_cb_sg_input"] - EXPECTED_CB_SG_DIST)
                if input_geo.get("d_cb_sg_input") is not None else None
            ),
            "bd_angle_input_diff_from_107": (
                abs(input_geo["bd_angle_input"] - BD_TARGET_DEG)
                if input_geo.get("bd_angle_input") is not None else None
            ),
            # post-Vina pose geometry
            "ligand_efficiency": None,
            "d_cb_sg_top1": None,
            "d_cb_sg_diff_from_185": None,
            "bd_angle_top1": None,
            "bd_angle_diff_from_107": None,
            "hinge_hbond_top1": None,
            "any_pose_feasible": None,
            "n_poses": None,
            "clash_count_top1": None,
            "warhead_found_in_pose": None,
        }
        if r["success"]:
            pose_pdbqt = pose_dir / f"{r['name']}_pose.pdbqt"
            lig_pdbqt = lig_dir / f"{r['name']}.pdbqt"
            geo = compute_pose_metrics(pose_pdbqt, lig_pdbqt, smi)
            row["d_cb_sg_top1"] = geo["d_cb_sg_top1"]
            row["d_cb_sg_diff_from_185"] = geo["d_cb_sg_diff_from_185"]
            row["bd_angle_top1"] = geo["bd_angle_top1"]
            row["bd_angle_diff_from_107"] = (
                abs(geo["bd_angle_top1"] - BD_TARGET_DEG) if geo["bd_angle_top1"] is not None else None
            )
            row["hinge_hbond_top1"] = geo["hinge_hbond_top1"]
            row["d_hinge_top1"] = geo.get("d_hinge_top1")
            row["any_pose_feasible"] = geo["any_pose_feasible"]
            row["n_poses"] = geo["n_poses"]
            row["warhead_found_in_pose"] = int(geo["warhead_found"])
            if n_heavy > 0 and r["vina_score"] is not None:
                row["ligand_efficiency"] = r["vina_score"] / n_heavy
            # Clash count: top model heavy atoms vs pocket atoms < 1.8 A
            if pocket_coords is not None:
                poses_all = parse_pdbqt_all_models(pose_pdbqt)
                if poses_all:
                    top = poses_all[0]
                    heavy_mask = np.array([e != "H" for e in top["elements"]])
                    if heavy_mask.sum() > 0:
                        lig = top["coords"][heavy_mask]
                        # vectorized pairwise distance
                        dmat = np.linalg.norm(lig[:, None, :] - pocket_coords[None, :, :], axis=2)
                        row["clash_count_top1"] = int((dmat < 2.0).sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = cohort_out / "per_mol.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{args.cohort}] Wrote {out_csv} ({len(df)} rows, {df['success'].sum()} successful)")

    # Top-10 SDF: by Vina score (best first)
    df_ok = df[df["success"] & df["vina_score"].notna()].sort_values("vina_score").head(10)
    sdf_path = cohort_out / "top10.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for _, row in df_ok.iterrows():
        try:
            m = Chem.MolFromSmiles(row["smi"])
            if m is None:
                continue
            m.SetProp("_Name", str(row["name"]))
            m.SetProp("vina_score", f"{row['vina_score']:.3f}")
            m.SetProp("cohort", args.cohort)
            writer.write(m)
        except Exception:
            continue
    writer.close()
    print(f"[{args.cohort}] Wrote {sdf_path}")


if __name__ == "__main__":
    main()
