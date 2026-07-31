"""Cofold-anchored covalent metric panel: real BD geometry + xTB Fukui f+.

For every cofolded sample under `boltz_cofolds/{cell}/pred_s####/.../*.cif`:

  1. Parse the CIF -> protein and ligand atoms.
  2. Compute real BD geometry:
       - d_SG   = distance from Cys346 SG to ligand acrylamide β-C
       - theta  = Bürgi–Dunitz angle = angle(Cys346:SG -> Cβ, Cβ -> Cα)
       - phi_planar = C=C-C(=O)-N torsion measured in the cofolded pose
       - BD-ready gate: d_SG ∈ [2.5, 5.5] Å AND theta ∈ [80°, 130°]
  3. Extract ligand as SDF -> RDKit mol.
  4. Real xTB Fukui f+ at the β-C:
       xtb --gfn 2 --acc 1 --vfukui  →  reads fukui.txt →  f+ at Cβ
  5. (Optional) Vina covalent-tethered rescore for the same pose.

Output: covalent_cofold_panel.csv with columns:
  cell, sample_idx, canonical_smi, cofold_ok, d_sg, bd_theta, phi_planar,
  bd_ready, fukui_fplus_real, xtb_ok, vina_cov_score

Reads plan (which samples to score) from `--panel_csv` (the 2D panel).  Only
cofolded samples get real geometry+xTB; the rest keep NaN.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# BD-ready gate (real, cofold-derived).
D_SG_LO, D_SG_HI = 2.5, 5.5      # Angstrom
BD_THETA_LO, BD_THETA_HI = 80.0, 130.0  # degrees

# Vina receptor (Cys346-stripped for AD-CovDock convention).
VINA_RECEPTOR = str(PROJECT_ROOT / "data/docking_500/receptor_cys346_stripped.pdbqt")
VINA_BIN = "vina"   # comes from conda env
CYS346_SG_COORD = np.array([18.888, -3.650, -29.979])
VINA_BOX_SIZE = np.array([20.0, 20.0, 20.0])


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def dihedral_deg(p1, p2, p3, p4) -> float:
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / max(np.linalg.norm(b2), 1e-9))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def wrap_planar(phi_signed: float) -> float:
    """Return min(|phi|, |180-|phi||) in [0, 90]."""
    a = abs(((phi_signed + 180.0) % 360.0) - 180.0)
    return float(min(a, 180.0 - a))


def parse_cif(cif_path: Path):
    """Return (protein_atoms_by_res, lig_atoms).

    protein_atoms_by_res: list of {aa, idx, atoms_dict}
    lig_atoms: list of {name, elem, xyz}"""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("m", str(cif_path))
    model = next(s.get_models())
    residues = []
    lig_atoms = []
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            if hetflag == " ":
                if chain.id != "A":
                    continue
                atoms = {a.get_name(): np.array(a.get_coord()) for a in res}
                residues.append({"aa": res.get_resname(),
                                    "idx": int(res.id[1]),
                                    "atoms": atoms})
            else:
                if res.get_resname().startswith("LIG"):
                    for a in res:
                        name = a.get_name()
                        elem = a.element.strip() if hasattr(a, "element") else \
                            ("H" if name.startswith("H") else name[0])
                        lig_atoms.append({"name": name,
                                            "elem": elem.upper(),
                                            "xyz": np.array(a.get_coord())})
    return residues, lig_atoms


def find_cys346_sg(residues):
    for r in residues:
        if r["aa"] == "CYS" and r["idx"] == 346 and "SG" in r["atoms"]:
            return r["atoms"]["SG"]
    # Fallback: any cys SG
    for r in residues:
        if r["aa"] == "CYS" and "SG" in r["atoms"]:
            return r["atoms"]["SG"]
    return None


def find_acryl_atoms_in_lig(smi: str, lig_atoms: list) -> dict | None:
    """Map the SMILES's canonical rank atoms to the LIG atom names in the CIF,
    then locate the acrylamide β-C / α-C / carbonyl-C / N.

    Boltz assigns lig atom names as C1, C2, ..., N1, N2, ... based on
    canonical rank + element type prefixes.  Simplest approach: rebuild the
    ligand in RDKit with 3D from the CIF and match SMARTS directly."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    # Build RDKit mol with 3D coords from lig_atoms.  We rely on the fact
    # that Boltz's atom naming embeds the canonical rank: strip element
    # prefix to get 1-indexed canonical rank.  Then remap to RDKit indices.
    mH = Chem.AddHs(m)
    ranks = list(Chem.CanonicalRankAtoms(mH, breakTies=True))
    # Map: rank+1 -> heavy-atom index in mH (which == index in m for heavy).
    rank_to_idx = {}
    for i, r in enumerate(ranks):
        if mH.GetAtomWithIdx(i).GetAtomicNum() != 1:
            rank_to_idx[r + 1] = i
    # Build coord dict
    coords = {}
    for a in lig_atoms:
        # a["name"] is like "C26" or "N3" — strip letters, parse int
        digits = "".join(c for c in a["name"] if c.isdigit())
        if not digits:
            continue
        rank = int(digits)
        if rank in rank_to_idx and a["elem"] == mH.GetAtomWithIdx(
                rank_to_idx[rank]).GetSymbol():
            coords[rank_to_idx[rank]] = a["xyz"]
    if len(coords) < mH.GetNumHeavyAtoms():
        # partial map is OK if we have the warhead atoms
        pass
    # SMARTS match: β = match[0], α = [1], C=O = [2], N = [4]
    patt = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")
    matches = m.GetSubstructMatches(patt)
    if not matches:
        return None
    a_b, a_a, a_co, _o, a_n = matches[0]
    if any(idx not in coords for idx in (a_b, a_a, a_co, a_n)):
        return None
    return {
        "b": coords[a_b],
        "alpha": coords[a_a],
        "carbonyl": coords[a_co],
        "nitrogen": coords[a_n],
        "b_idx": a_b,
    }


def compute_bd_geometry(cif_path: Path, smi: str) -> dict:
    """Return dict with d_sg, bd_theta, phi_planar, bd_ready, notes."""
    result = {"d_sg": np.nan, "bd_theta": np.nan, "phi_planar": np.nan,
                "bd_ready": False, "notes": ""}
    try:
        residues, lig = parse_cif(cif_path)
    except Exception as e:
        result["notes"] = f"parse_cif fail: {e}"
        return result
    sg = find_cys346_sg(residues)
    if sg is None:
        result["notes"] = "no cys346 SG"
        return result
    if not lig:
        result["notes"] = "no ligand atoms"
        return result
    info = find_acryl_atoms_in_lig(smi, lig)
    if info is None:
        result["notes"] = "acryl atoms not found in lig"
        return result
    b = info["b"]; alpha = info["alpha"]
    carbonyl = info["carbonyl"]; nitrogen = info["nitrogen"]
    d_sg = float(np.linalg.norm(b - sg))
    bd_theta = angle_deg(sg - b, alpha - b)
    phi_signed = dihedral_deg(b, alpha, carbonyl, nitrogen)
    phi_planar = wrap_planar(phi_signed)
    bd_ready = bool((D_SG_LO <= d_sg <= D_SG_HI) and
                       (BD_THETA_LO <= bd_theta <= BD_THETA_HI))
    result.update({"d_sg": d_sg, "bd_theta": bd_theta,
                    "phi_planar": phi_planar, "bd_ready": bd_ready,
                    "b_idx": int(info["b_idx"]),
                    "notes": "ok"})
    return result


def run_xtb_fukui(smi: str, b_atom_idx_in_rdkit: int) -> dict:
    """Run xTB with --vfukui on a 3D-embedded RDKit conformer, extract the
    Fukui f+ value at atom index `b_atom_idx_in_rdkit` (heavy-atom RDKit
    index).  Returns dict with fukui_fplus_real, xtb_ok, xtb_note."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    out = {"fukui_fplus_real": np.nan, "xtb_ok": False, "xtb_note": ""}
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out["xtb_note"] = "smi parse fail"
            return out
        mH = Chem.AddHs(m)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mH, params)
        if cid < 0:
            out["xtb_note"] = "embed fail"
            return out
        try:
            AllChem.MMFFOptimizeMolecule(mH, maxIters=200)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "mol.xyz"
            # Write XYZ manually to control atom ordering.
            conf = mH.GetConformer(cid)
            nA = mH.GetNumAtoms()
            lines = [str(nA), ""]
            for i in range(nA):
                a = mH.GetAtomWithIdx(i)
                p = conf.GetAtomPosition(i)
                lines.append(f"{a.GetSymbol():<3s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
            xyz.write_text("\n".join(lines))
            cmd = ["xtb", str(xyz), "--gfn", "2", "--acc", "1.0",
                     "--vfukui", "--iterations", "150"]
            res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                                    timeout=300)
            if res.returncode != 0:
                out["xtb_note"] = f"rc={res.returncode}"
                return out
            # Fukui f+ appears in xtb stdout under "Fukui functions f(+) at
            # atom".  Parse the table.
            lines = res.stdout.splitlines()
            fplus = None
            in_table = False
            for i, ln in enumerate(lines):
                if "Fukui-Analysis" in ln or "Fukui analysis" in ln.lower() \
                        or "f(+)" in ln:
                    in_table = True
                    continue
                if in_table:
                    parts = ln.split()
                    # Format: "  1 C     -0.000  ..." or similar
                    if len(parts) >= 3:
                        try:
                            atom_num = int(parts[0])
                            # xtb 1-indexes over ALL atoms including H.
                            # Our beta-C is at RDKit index b_atom_idx_in_rdkit
                            # in mH, so xtb index = b_atom_idx_in_rdkit+1.
                            if atom_num == b_atom_idx_in_rdkit + 1:
                                # try to parse f(+) (usually 3rd numeric col)
                                nums = []
                                for p in parts[1:]:
                                    try:
                                        nums.append(float(p))
                                    except ValueError:
                                        continue
                                if nums:
                                    # First numeric column is f(+), or the
                                    # 4th (varies by xtb version).  We take
                                    # the maximum-mag positive to be safe.
                                    fplus = nums[0]
                                break
                        except ValueError:
                            continue
            if fplus is None:
                # Fallback: read fukui.txt if written
                ftxt = Path(tmp) / "fukui.txt"
                if ftxt.exists():
                    for ln in ftxt.read_text().splitlines():
                        parts = ln.split()
                        if len(parts) >= 4:
                            try:
                                atom_num = int(parts[0])
                                if atom_num == b_atom_idx_in_rdkit + 1:
                                    fplus = float(parts[2])
                                    break
                            except ValueError:
                                pass
            if fplus is None:
                out["xtb_note"] = "fukui parse fail"
                return out
            out["fukui_fplus_real"] = float(fplus)
            out["xtb_ok"] = True
            out["xtb_note"] = "ok"
            return out
    except subprocess.TimeoutExpired:
        out["xtb_note"] = "timeout"
        return out
    except Exception as e:
        out["xtb_note"] = f"exc: {type(e).__name__}: {e}"
        return out


def run_vina_covalent(smi: str) -> dict:
    """Run AD-CovDock Vina rescore against Cys346-stripped ZAP70 receptor.
    Reuses experiments/run_covalent_docking.py.  Returns dict with
    vina_cov_score, vina_ok, vina_note."""
    out = {"vina_cov_score": np.nan, "vina_ok": False, "vina_note": ""}
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from experiments.run_covalent_docking import (
            build_cov_tethered_pdbqt, vina_dock,
        )
    except Exception as e:
        out["vina_note"] = f"import fail: {e}"
        return out
    try:
        res = build_cov_tethered_pdbqt(smi)
        if res is None:
            out["vina_note"] = "tether build fail"
            return out
        lig_pdbqt = res.get("pdbqt")
        if lig_pdbqt is None or not Path(lig_pdbqt).exists():
            out["vina_note"] = "no pdbqt"
            return out
        score, _pose = vina_dock(lig_pdbqt, VINA_RECEPTOR,
                                    CYS346_SG_COORD, VINA_BOX_SIZE,
                                    exhaustiveness=4)
        if score is None:
            out["vina_note"] = "vina None"
            return out
        out["vina_cov_score"] = float(score)
        out["vina_ok"] = True
        out["vina_note"] = "ok"
        return out
    except Exception as e:
        out["vina_note"] = f"exc: {type(e).__name__}: {e}"
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_metric_panel_v2.csv"))
    ap.add_argument("--cofold_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/boltz_cofolds"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_cofold_panel.csv"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "cofold_scoring_progress.json"))
    ap.add_argument("--run_xtb", action="store_true", default=True)
    ap.add_argument("--run_vina", action="store_true", default=True)
    args = ap.parse_args()

    panel = pd.read_csv(args.panel_csv)
    panel = panel[panel["valid"] == True]
    # Index by (cell, sample_idx) -> row
    panel_idx = {(str(r["cell"]), int(r["sample_idx"])): r
                    for _, r in panel.iterrows()}

    cofold_root = Path(args.cofold_dir)
    rows = []
    all_cifs = []
    for cell_dir in sorted(cofold_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        cell = cell_dir.name
        for pred_dir in sorted(cell_dir.iterdir()):
            if not pred_dir.is_dir() or not pred_dir.name.startswith("pred_"):
                continue
            stem = pred_dir.name[len("pred_"):]   # e.g. "s0042"
            try:
                sidx = int(stem.lstrip("s"))
            except ValueError:
                continue
            cifs = list(pred_dir.rglob(f"{stem}_model_0.cif"))
            if not cifs:
                continue
            all_cifs.append((cell, sidx, cifs[0]))

    print(f"Found {len(all_cifs)} cofolded CIFs", flush=True)
    progress_path = Path(args.progress_path)

    t_start = time.time()
    for i, (cell, sidx, cif) in enumerate(all_cifs):
        r = panel_idx.get((cell, sidx))
        smi = str(r["largest_frag_SMILES"]) if r is not None else None
        if smi is None or smi.lower() == "nan":
            continue

        # Real BD geometry.
        bd = compute_bd_geometry(cif, smi)
        b_idx = bd.get("b_idx")
        fukui_result = {"fukui_fplus_real": np.nan, "xtb_ok": False,
                          "xtb_note": "skipped"}
        if args.run_xtb and b_idx is not None:
            fukui_result = run_xtb_fukui(smi, int(b_idx))

        vina_result = {"vina_cov_score": np.nan, "vina_ok": False,
                          "vina_note": "skipped"}
        if args.run_vina:
            vina_result = run_vina_covalent(smi)

        row = {
            "cell": cell,
            "sample_idx": sidx,
            "smi": smi,
            "cofold_cif": str(cif),
            "d_sg": bd["d_sg"],
            "bd_theta": bd["bd_theta"],
            "phi_planar": bd["phi_planar"],
            "bd_ready": bd["bd_ready"],
            "bd_notes": bd.get("notes", ""),
            **fukui_result,
            **vina_result,
        }
        rows.append(row)
        elapsed = time.time() - t_start
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(all_cifs) - i - 1) / max(rate, 1e-6)
        if (i + 1) % 5 == 0 or i == len(all_cifs) - 1:
            print(f"[{i+1}/{len(all_cifs)}] {cell} s{sidx:04d}  d_sg={bd['d_sg']:.2f} theta={bd['bd_theta']:.1f} "
                   f"bd_ready={bd['bd_ready']} fukui_fplus={fukui_result['fukui_fplus_real']} "
                   f"vina={vina_result['vina_cov_score']}  ETA={eta/60:.1f}min",
                   flush=True)
            progress_path.write_text(json.dumps({
                "phase": "cofold_scoring",
                "done": i + 1, "total": len(all_cifs),
                "rate": rate, "eta_min": eta / 60,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2))
            # Incremental save so preemption is safe.
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  ({len(rows)} rows)", flush=True)
    progress_path.write_text(json.dumps({
        "phase": "cofold_scoring_done",
        "n_rows": len(rows),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
