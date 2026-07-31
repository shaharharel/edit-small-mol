"""Vina dock the L2 scaffold-anchor C1 (n=50) vs C5 (n=28) cohorts.

Targeted box around Cys346 SG (radius ~12 Å) to test the regiochemistry fix
in binding pose, not in CHR. Reuses the receptor.pdbqt from docking_500.
"""
import json, subprocess, time, os
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from meeko import MoleculePreparation, PDBQTWriterLegacy

RDLogger.logger().setLevel(RDLogger.ERROR)
ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
VINA = ROOT / "tools" / "vina"
RECEPTOR = ROOT / "data" / "docking_500" / "receptor.pdbqt"

# Targeted box around Cys346 SG. Box 24x24x24 Å — covers warhead + recognition arm to Met414.
CYS346_SG = np.array([18.888, -3.650, -29.979])
BOX_SIZE = (24.0, 24.0, 24.0)

COHORTS = {
    "C1": ROOT / "data" / "lingo3dmol_L2_scaffold_anchor" / "samples_T10.sdf",
    "C5": ROOT / "data" / "lingo3dmol_L2_scaffold_C5"     / "samples_T10.sdf",
}

OUT_DIR = ROOT / "data" / "lingo3dmol_docking_v2"


def prep_pdbqt(mol, out_path):
    """Embed if no 3D, write PDBQT via meeko."""
    if mol.GetNumConformers() == 0:
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=42) < 0:
            return False
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    else:
        m = Chem.AddHs(mol, addCoords=True)
    prep = MoleculePreparation()
    setups = prep.prepare(m)
    if not setups:
        return False
    pdbqt, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        return False
    out_path.write_text(pdbqt)
    return True


def run_vina(lig_pdbqt, pose_pdbqt, timeout=180):
    cmd = [str(VINA),
           "--receptor", str(RECEPTOR),
           "--ligand", str(lig_pdbqt),
           "--center_x", str(CYS346_SG[0]),
           "--center_y", str(CYS346_SG[1]),
           "--center_z", str(CYS346_SG[2]),
           "--size_x", str(BOX_SIZE[0]),
           "--size_y", str(BOX_SIZE[1]),
           "--size_z", str(BOX_SIZE[2]),
           "--exhaustiveness", "8",
           "--num_modes", "5",
           "--cpu", "4",
           "--out", str(pose_pdbqt)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def parse_score(pose_pdbqt):
    if not pose_pdbqt.exists():
        return None
    for line in pose_pdbqt.read_text().splitlines():
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:" and i + 1 < len(parts):
                    try: return float(parts[i + 1])
                    except: pass
    return None


def dock_cohort(name, sdf_path):
    print(f"\n=== Cohort {name}: {sdf_path.name} ===")
    cohort_dir = OUT_DIR / name
    (cohort_dir / "ligands_pdbqt").mkdir(parents=True, exist_ok=True)
    (cohort_dir / "poses").mkdir(parents=True, exist_ok=True)

    results = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    t0 = time.time()
    for i, m in enumerate(suppl):
        if m is None: continue
        smi = Chem.MolToSmiles(Chem.RemoveHs(m))
        lig = cohort_dir / "ligands_pdbqt" / f"{name}_{i:03d}.pdbqt"
        pose = cohort_dir / "poses" / f"{name}_{i:03d}_pose.pdbqt"

        ok_prep = prep_pdbqt(m, lig)
        if not ok_prep:
            results.append({"idx": i, "smi": smi, "prep_ok": False, "vina": None})
            print(f"  [{i:03d}] PREP FAIL")
            continue
        ok_dock = run_vina(lig, pose)
        score = parse_score(pose) if ok_dock else None
        results.append({"idx": i, "smi": smi, "prep_ok": True, "vina": score})
        elapsed = time.time() - t0
        print(f"  [{i:03d}] vina={score} ({elapsed:.0f}s elapsed)")

    out_json = cohort_dir / "vina_scores.json"
    out_json.write_text(json.dumps(results, indent=2))
    scores = [r["vina"] for r in results if r["vina"] is not None]
    print(f"\n  {name}: {len(scores)}/{len(results)} docked, "
          f"median={np.median(scores):.2f}, mean={np.mean(scores):.2f}, "
          f"min={np.min(scores):.2f}, max={np.max(scores):.2f}")
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, sdf in COHORTS.items():
        all_results[name] = dock_cohort(name, sdf)

    # Comparison
    print("\n" + "="*60 + "\nFINAL COMPARISON\n" + "="*60)
    for name in ["C1", "C5"]:
        scores = [r["vina"] for r in all_results[name] if r["vina"] is not None]
        print(f"  {name}: n={len(scores)}, median={np.median(scores):+.2f}, "
              f"p25={np.percentile(scores, 25):+.2f}, p75={np.percentile(scores, 75):+.2f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
