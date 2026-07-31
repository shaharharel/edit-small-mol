#!/usr/bin/env python3
"""Compute Mol1 (seed/design anchor) per-pose enriched metrics.

Mol1 is NOT in the 520K bulk-scored CSV and NOT in the 997 cofold pipeline
(it's the anchor, by definition). To populate its row in the dashboard
without leaving most columns as `—`, we compute its enriched values
ONCE here and write them to a JSON cache the backend loads at startup.

Output: data/boltz_poses/mol1_anchor_enriched.json

Computed columns:
  - d_SG, burgi_dunitz_dev_deg, n_h_bonds, n_stabilizing_contacts,
    pocket_occupancy_pct        (cofold geom — from compute_pose_quality_v2
                                  + compute_contacts_occupancy logic)
  - vina_kcalmol, vina_inter_kcalmol, vina_intra_kcalmol  (Vina rescore)
  - pKa_Cys346                                            (PROPKA3)
  - LUMO_eV, HOMO_eV, gap_eV, omega_eV, q_Cb, fukui_plus_Cb,
    pred_log_k2_GSH                                        (xTB warhead)
  - rdkit_strain_kcal_mol, vdw_interaction_kcal_mol      (RDKit MMFF + UFF)
  - rdkit_strain_posefree_kcal_mol                       (RDKit posefree)
  - shape_Tc_seed, esp_sim_seed, warhead_dev_deg          (= 1, 1, 0)

The script is idempotent: if mol1_anchor_enriched.json exists and is
fresh, it skips compute (use --force to recompute).

Usage:
  python experiments/enrich_mol1_anchor.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
MOL1_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "mol1__zap70_cys346"
CIF_PATH = MOL1_DIR / "mol1" / "mol1_model_0.cif"
OUT_JSON = PROJECT_ROOT / "data" / "boltz_poses" / "mol1_anchor_enriched.json"


def log(msg: str) -> None:
    print(f"[mol1_anchor] {msg}", flush=True)


def cif_to_pdb(cif: Path, pdb: Path) -> None:
    """Convert mol1's Boltz CIF into a PDB suitable for Vina / PROPKA / strain.

    Uses the same residue-renaming logic as experiments/cif_to_pdb_for_mmgbsa.py
    (LIG1 → LIG, mark as non-polymer HETATM).
    """
    import gemmi
    st = gemmi.read_structure(str(cif))
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "LIG1":
                    res.name = "LIG"
                    res.het_flag = "H"
                    res.entity_type = gemmi.EntityType.NonPolymer
    st.write_pdb(str(pdb))


def compute_pose_geom(cif: Path, warhead_atom_name: str) -> dict:
    """Cofold geometry — d_SG, n_h_bonds, n_stabilizing_contacts, occupancy.

    Combines compute_pose_quality_v2 (d_SG, burgi-dunitz, n_h_bonds, atp_pocket)
    with compute_contacts_occupancy (n_stabilizing_contacts breakdown, pocket %).
    """
    from anchordiff.compute_pose_quality_v2 import (
        parse_cofold, d_SG_honest, burgi_dunitz_dev, n_h_bonds as count_h_bonds,
        atp_pocket_fraction, hinge_hbond_present, derive_warhead_atom_name,
    )
    from anchordiff.compute_contacts_occupancy import (
        find_covalent, find_h_bonds, find_salt_bridges, find_pi_pi,
        pocket_occupancy,
    )

    prot, lig = parse_cofold(cif)

    # Use manifest's warhead_atom_name first; fall back to SMILES derivation.
    if not warhead_atom_name:
        _, warhead_atom_name = derive_warhead_atom_name(MOL1_SMILES)

    d_sg = d_SG_honest(prot, lig, warhead_atom_name)
    bd = burgi_dunitz_dev(prot, lig, warhead_atom_name)
    nhb_v2, _ = count_h_bonds(prot, lig)
    apf = atp_pocket_fraction(prot, lig, cutoff_A=4.0)
    hh = hinge_hbond_present(prot, lig)

    # Composite contacts (matches the 997-batch pipeline's accounting).
    n_cov, _ = find_covalent(prot, lig)
    n_hb_c, _ = find_h_bonds(prot, lig)
    n_sb, _ = find_salt_bridges(prot, lig)
    n_pi, _ = find_pi_pi(prot, lig, MOL1_SMILES)
    occ = pocket_occupancy(prot, lig)

    return {
        "d_SG": d_sg,
        "geom_ok": True if (d_sg is not None and d_sg <= 4.0) else (False if d_sg is not None else None),
        "burgi_dunitz_dev_deg": bd,
        "n_h_bonds": int(nhb_v2),
        "atp_pocket_fraction": float(apf),
        "hinge_hbond": bool(hh),
        "n_stabilizing_contacts": int(n_cov + n_hb_c + n_sb + n_pi),
        "pocket_occupancy_pct": float(occ) if occ is not None else None,
        "warhead_atom_name": warhead_atom_name,
    }


def compute_vina(pdb: Path) -> dict:
    """Run vina --score_only on a one-cofold dir (must contain a *_model_0.pdb)."""
    from experiments.vina_rescore_cofolds import rescore_cofold
    return rescore_cofold(pdb.parent)


def compute_propka(pdb: Path) -> dict:
    from experiments.propka_cys346_pka import compute_cys346_pka
    return compute_cys346_pka(pdb)


def compute_xtb(smiles: str) -> dict:
    from experiments.xtb_warhead_electrophilicity import compute_warhead_descriptors
    return compute_warhead_descriptors(smiles)


def compute_strain_cofold(pdb: Path, smiles: str) -> dict:
    from src.utils.rdkit_strain import score_cofold
    return score_cofold(pdb, smiles=smiles, sdf_path=None, num_conf=5,
                        seed=42, compute_interaction=True)


def compute_strain_posefree(smiles: str) -> dict:
    from src.utils.rdkit_strain import score_posefree
    return score_posefree(smiles, num_conf_free=5, seed=42)


def compute_3d_seed_metrics(smiles: str) -> dict:
    """shape_Tc_seed, esp_sim_seed, warhead_dev_deg for Mol1 vs the seed (= Mol1)."""
    # By construction: Mol1 IS the seed, so these are 1, 1, 0 — but we recompute
    # for honesty (sanity check the seed-3D pipeline returns the expected values).
    try:
        from src.utils.mol1_scoring import (
            shape_tanimoto_seed, esp_sim_seed, warhead_vector_deviation,
        )
        return {
            "shape_Tc_seed": float(shape_tanimoto_seed(smiles)),
            "esp_sim_seed": float(esp_sim_seed(smiles)),
            "warhead_dev_deg": float(warhead_vector_deviation(smiles)),
        }
    except Exception as e:
        log(f"  3d-seed metrics failed (using definitional values): {e}")
        return {"shape_Tc_seed": 1.0, "esp_sim_seed": 1.0, "warhead_dev_deg": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = ap.parse_args()

    if OUT_JSON.exists() and not args.force:
        log(f"cache exists at {OUT_JSON} — use --force to recompute")
        return 0

    if not CIF_PATH.exists():
        log(f"ERROR: Mol1 cofold CIF not found at {CIF_PATH}")
        return 2

    out: dict = {
        "smiles": MOL1_SMILES,
        "cif": str(CIF_PATH),
        "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Pull Boltz fields from manifest for completeness.
    manifest = json.loads((MOL1_DIR / "mol1_manifest.json").read_text())["mol1"]
    warhead_atom = manifest.get("warhead_atom_name", "C26")

    # Convert CIF → PDB ONCE in a stable place (next to the CIF). Vina, PROPKA,
    # and RDKit strain all need a PDB.
    pdb_path = CIF_PATH.with_suffix(".pdb")
    if not pdb_path.exists():
        log(f"converting CIF → PDB: {pdb_path.name}")
        cif_to_pdb(CIF_PATH, pdb_path)
    else:
        log(f"reusing existing PDB: {pdb_path.name}")

    # ── 1. Cofold geometry ────────────────────────────────────────────────
    log("computing cofold geometry (d_SG, h-bonds, contacts, occupancy)...")
    t0 = time.time()
    try:
        geom = compute_pose_geom(CIF_PATH, warhead_atom)
        out.update(geom)
        log(f"  d_SG={geom['d_SG']:.2f}Å  n_h_bonds={geom['n_h_bonds']}  "
            f"n_stab={geom['n_stabilizing_contacts']}  occ={geom['pocket_occupancy_pct']:.1f}%  ({time.time()-t0:.1f}s)")
    except Exception as e:
        log(f"  geom failed: {e}")
        for k in ("d_SG", "geom_ok", "n_h_bonds", "n_stabilizing_contacts",
                  "pocket_occupancy_pct", "burgi_dunitz_dev_deg"):
            out[k] = None
        out["warhead_atom_name"] = warhead_atom

    # ── 2. Vina rescore ────────────────────────────────────────────────────
    log("running Vina --score_only ...")
    t0 = time.time()
    try:
        vina = compute_vina(pdb_path)
        if vina.get("success_flag", 0) == 1:
            out["vina_kcalmol"] = vina["vina_kcalmol"]
            out["vina_inter_kcalmol"] = vina["vina_inter_kcalmol"]
            out["vina_intra_kcalmol"] = vina["vina_intra_kcalmol"]
            log(f"  vina={vina['vina_kcalmol']:.3f}  inter={vina['vina_inter_kcalmol']:.3f}  intra={vina['vina_intra_kcalmol']:.3f}  ({time.time()-t0:.1f}s)")
        else:
            log(f"  vina failed: {vina.get('error')}")
            out["vina_kcalmol"] = None
            out["vina_inter_kcalmol"] = None
            out["vina_intra_kcalmol"] = None
    except Exception as e:
        log(f"  vina failed: {e}")
        out["vina_kcalmol"] = None
        out["vina_inter_kcalmol"] = None
        out["vina_intra_kcalmol"] = None

    # ── 3. PROPKA3 ─────────────────────────────────────────────────────────
    log("running PROPKA3 ...")
    t0 = time.time()
    try:
        # PROPKA needs PROTEIN only — strip HETATM (the LIG) and use the SAME prot.pdb
        # the vina_rescore_cofolds split-out produces (it stays next to the cofold).
        prot_pdb = pdb_path.with_name(pdb_path.stem + ".prot.pdb")
        if not prot_pdb.exists():
            # Re-split if not done already.
            from experiments.vina_rescore_cofolds import split_pdb_to_protein_and_ligand
            lig_sdf = pdb_path.with_name(pdb_path.stem + ".lig.sdf")
            split_pdb_to_protein_and_ligand(pdb_path, prot_pdb, lig_sdf)
        propka_res = compute_propka(prot_pdb)
        if propka_res.get("success_flag", 0) == 1:
            out["pKa_Cys346"] = propka_res["pKa_Cys346"]
            log(f"  pKa_Cys346={propka_res['pKa_Cys346']:.2f}  ({time.time()-t0:.1f}s)")
        else:
            log(f"  propka failed: {propka_res.get('error')}")
            out["pKa_Cys346"] = None
    except Exception as e:
        log(f"  propka failed: {e}")
        out["pKa_Cys346"] = None

    # ── 4. xTB warhead electrophilicity ────────────────────────────────────
    log("running xTB GFN2 warhead descriptors ...")
    t0 = time.time()
    try:
        xtb_res = compute_xtb(MOL1_SMILES)
        if xtb_res.get("success_flag", 0) == 1:
            for k in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
                      "fukui_plus_Cb", "pred_log_k2_GSH"):
                v = xtb_res.get(k)
                out[k] = float(v) if v is not None and np.isfinite(v) else None
            log(f"  LUMO={out['LUMO_eV']:.3f}eV  q_Cb={out['q_Cb']:.3f}  "
                f"log_k2={out['pred_log_k2_GSH']:.3f}  ({time.time()-t0:.1f}s)")
        else:
            log(f"  xtb failed: {xtb_res.get('error')}")
            for k in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
                      "fukui_plus_Cb", "pred_log_k2_GSH"):
                out[k] = None
    except Exception as e:
        log(f"  xtb failed: {e}")
        for k in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
                  "fukui_plus_Cb", "pred_log_k2_GSH"):
            out[k] = None

    # ── 5. RDKit strain (cofold + posefree) + UFF vdW ──────────────────────
    log("running RDKit MMFF cofold strain + UFF vdW ...")
    t0 = time.time()
    try:
        strain = compute_strain_cofold(pdb_path, MOL1_SMILES)
        if strain.get("success_flag", 0) == 1:
            out["rdkit_strain_kcal_mol"] = float(strain["strain_kcal_mol"])
            vdw = strain.get("vdw_interaction_kcal_mol")
            out["vdw_interaction_kcal_mol"] = float(vdw) if vdw is not None and np.isfinite(vdw) else None
            log(f"  strain={out['rdkit_strain_kcal_mol']:.2f} kcal/mol  vdW={out['vdw_interaction_kcal_mol']}  ({time.time()-t0:.1f}s)")
        else:
            log(f"  cofold strain failed: {strain.get('error')}")
            out["rdkit_strain_kcal_mol"] = None
            out["vdw_interaction_kcal_mol"] = None
    except Exception as e:
        log(f"  cofold strain failed: {e}")
        out["rdkit_strain_kcal_mol"] = None
        out["vdw_interaction_kcal_mol"] = None

    log("running RDKit posefree strain ...")
    t0 = time.time()
    try:
        pf = compute_strain_posefree(MOL1_SMILES)
        if pf.get("success_flag", 0) == 1:
            out["rdkit_strain_posefree_kcal_mol"] = float(pf["strain_kcal_mol"])
            log(f"  posefree strain={out['rdkit_strain_posefree_kcal_mol']:.2f} kcal/mol  ({time.time()-t0:.1f}s)")
        else:
            out["rdkit_strain_posefree_kcal_mol"] = None
            log(f"  posefree failed: {pf.get('error')}")
    except Exception as e:
        out["rdkit_strain_posefree_kcal_mol"] = None
        log(f"  posefree failed: {e}")

    # ── 6. 3D seed metrics (sanity: Mol1 vs Mol1 = 1.0, 1.0, 0) ────────────
    log("computing 3D-seed-similarity sanity (Mol1 vs Mol1) ...")
    out.update(compute_3d_seed_metrics(MOL1_SMILES))
    log(f"  shape_Tc_seed={out['shape_Tc_seed']:.3f}  esp_sim_seed={out['esp_sim_seed']:.3f}  warhead_dev={out['warhead_dev_deg']:.2f}°")

    # ── Write ────────────────────────────────────────────────────────────
    # Cleanup numpy types so JSON serializes cleanly.
    clean: dict = {}
    for k, v in out.items():
        if isinstance(v, (np.floating,)):
            clean[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        elif isinstance(v, float) and not np.isfinite(v):
            clean[k] = None
        else:
            clean[k] = v

    OUT_JSON.write_text(json.dumps(clean, indent=2, default=str))
    log(f"WROTE {OUT_JSON}")
    log(f"keys populated: {sum(1 for v in clean.values() if v not in (None, ''))}  /  {len(clean)}")
    nulls = [k for k, v in clean.items() if v in (None, "")]
    if nulls:
        log(f"NULLs ({len(nulls)}): {nulls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
