"""Capped-analog MM-GBSA recognition-energy scorer.

Methodology — Cheng JCTC 2017 (EGFR) + Zhu 2015 (Schrödinger):
  1. Take the covalent cofold pose (Cys-SG ≈ 1.8 Å from warhead Cβ).
  2. Saturate the acrylamide C=C to a propionamide C-C (add 2 H atoms).
  3. Score MM-GBSA WITHOUT any covalent-bond restraint.
  4. The result `dG_recognition_kcalmol` is the K_I proxy — what binding free
     energy WOULD be if the bond never formed. Clean and interpretable.

This replaces the artifact-laden covalent-complex MM-GBSA in
`zap70_cys346_energy_scores_v2.csv`, which is biased by spring restraint
energy + close-contact GBSA penalty + ligand strain at the warhead Cβ.

Usage as library:
    from score_cofold_energy_capped import score_one_capped
    res = score_one_capped(pdb_path, target, name, smiles)
    # res["dG_recognition_kcalmol"] is the K_I proxy
    # res["capped_smiles"] is the propionamide SMILES that was scored
    # res["success_flag"] == 0 if capping or scoring failed

If the SMILES does not contain an acrylamide warhead, score_one_capped
returns a failure dict with error="no_acrylamide_warhead". The batch
driver treats these as skips (success_flag=0) and they do not contribute
to coverage.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from cap_warhead_to_analog import cap_acrylamide_to_propionamide  # noqa: E402
from score_cofold_energy import score_one  # noqa: E402


def score_one_capped(pdb_path: Path, target: str, name: str, smiles: str) -> dict:
    """Cap acrylamide → propionamide and run MM-GBSA without covalent bond.

    Returns a dict with `_capped`-suffixed columns to avoid colliding with
    the covalent v2 CSV column names. All energy values are in kcal/mol.

    Failure modes:
      - error="no_acrylamide_warhead" : SMILES has no [CH2]=[CH]-C(=O)-N
        match; caller should skip. success_flag=0.
      - error="capping_failed" : SMARTS matched but PDB rewrite failed
        (atom-name mismatch, embed failure, etc.). success_flag=0.
      - Any antechamber/parmchk2/OpenMM exception: bubbled up through
        score_one(). success_flag=0.
    """
    out = {
        "target": target, "name": name, "smiles": smiles,
        "capped_smiles": None,
        "dG_recognition_kcalmol": None,
        "ligand_strain_capped_kcalmol": None,
        "E_complex_capped": None,
        "E_protein_capped": None,
        "E_ligand_bound_capped": None,
        "E_ligand_free_capped": None,
        "rmsd_min_capped_A": None,
        "E_complex_raw_capped": None,
        "E_far_restraint_capped": None,
        "n_near_residues_capped": None,
        "success_flag": 0,
        "error": None,
    }
    try:
        # Step 1: cap the warhead
        capped_smi, capped_pdb = cap_acrylamide_to_propionamide(smiles, pdb_path)
        if capped_smi is None:
            out["error"] = "no_acrylamide_warhead"
            return out
        out["capped_smiles"] = capped_smi

        # Step 2: score the capped complex WITHOUT a covalent bond
        res = score_one(
            capped_pdb,
            target=target,
            name=name + "_capped",
            smiles=capped_smi,
            no_covalent_bond=True,
        )
        if not res.get("success_flag"):
            out["error"] = res.get("error") or "score_one_failed"
            # Keep traceback for debugging if present
            if "traceback" in res:
                out["traceback"] = res["traceback"]
            return out

        # Step 3: re-key columns with _capped suffix
        out["dG_recognition_kcalmol"] = res["dG_bind_kcalmol"]
        out["ligand_strain_capped_kcalmol"] = res["ligand_strain_kcalmol"]
        out["E_complex_capped"] = res["E_complex"]
        out["E_protein_capped"] = res["E_protein"]
        out["E_ligand_bound_capped"] = res["E_ligand_bound"]
        out["E_ligand_free_capped"] = res["E_ligand_free"]
        out["rmsd_min_capped_A"] = res["rmsd_min_A"]
        out["E_complex_raw_capped"] = res.get("_E_complex_raw")
        out["E_far_restraint_capped"] = res.get("_E_far_restraint")
        out["n_near_residues_capped"] = res.get("_n_near_residues")
        out["success_flag"] = 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-500:]
    return out


def main():
    """Smoke test on the same fixture as score_cofold_energy.py (5P9J BTK+ibrutinib)."""
    # Use a Boltz cofold — the BTK 5P9J PDB uses crystallographic N1/C2/N3
    # atom names that don't match our Boltz-CanonicalRank atom-naming scheme.
    test_pdb = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/boltz_poses/"
                    "boltz_results_top1000__zap70_cys346/predictions/"
                    "003889_Amine_Replacements_3889/003889_Amine_Replacements_3889_model_0.pdb")
    smi = "C=CC(=O)N1Cc2cccc(C(=O)NC3=C(Cl)C(=O)c4[nH]ncc4C3=O)c2C1"

    print("=" * 80)
    print(f"Capped-analog smoke test: {test_pdb.name}")
    print("=" * 80)
    t0 = time.time()
    res = score_one_capped(test_pdb, target="smoke", name=test_pdb.stem, smiles=smi)
    dt = time.time() - t0
    print(f"\n=== Result ({dt:.1f}s) ===")
    for k, v in res.items():
        if k == "traceback":
            continue
        if isinstance(v, float):
            print(f"  {k:35s} {v:+10.3f}")
        else:
            print(f"  {k:35s} {v}")
    if not res["success_flag"]:
        print(f"\n  traceback:\n{res.get('traceback','')}")


if __name__ == "__main__":
    main()
