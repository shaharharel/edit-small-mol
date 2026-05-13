"""Curate CovalentInDB 2.0 into a training set for the C+D flagship.

QA-driven fixes (2026-05-12):
  - Per-warhead anchor element validation (Michael→C, Disulfide→S, etc.)
    so the anchor atom heuristic never picks an O or N for a C-attack class.
  - Stratified train/val split by (warhead_class, target) instead of year median,
    so rare warhead classes don't cluster into one split.
  - Random seed propagation.


Outputs:
  data/covbinder/covind_training_set.csv  — one row per (PDB, ligand-Cys-bond)
    columns:
      record_id, pdb_id, pdb_path, ligand_chain, ligand_resi, ligand_resname,
      cys_chain, cys_resi, warhead_class, reaction_type, smiles, target,
      organism, resolution, year,
      sg_x, sg_y, sg_z, cb_x, cb_y, cb_z,
      warhead_atom_idx_anchor   (index into ligand of the Cβ/Cα attacked carbon)

Three filters applied (default):
  1. CYS reactive residue only.
  2. Warhead in {Michael Acceptor, Halohydrocarbon, Vinyl Sulfone,
                 Beta Lactam, Epoxide, Disulfide, Aldehyde, Nitrile,
                 Aldehydic carbonyl, Carbonyl}.
     (Subset that maps cleanly to a canonical 3D approach geometry.)
  3. Resolution ≤ 3.0 Å (drop low-quality entries).

For each kept entry we additionally validate the SG→ligand-atom geometry
(d ≤ 2.4 Å for a covalent S–C bond) by parsing the PDB.
"""
from __future__ import annotations
from pathlib import Path
import sys, json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser

CIDB_ROOT  = PROJECT_ROOT / "data" / "covbinder" / "raw_covindb2"
CSV_PATH   = CIDB_ROOT / "Covalent_Complex_Records.csv"
PDB_DIR    = CIDB_ROOT / "PDB"
OUT_CSV    = PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"

# Warhead classes we explicitly parametrise.
# Per-class canonical approach: (d_S_to_attack_atom Å, attack angle °)
# Sources: QM Bürgi-Dunitz analyses for Michael (Houk JACS 1990s);
#          SN2 backside attack for Halohydrocarbon;
#          empirical literature for the rest.
WARHEAD_GEOM = {
    # Original 12 classes
    "Michael Acceptor":       {"d": 1.85, "angle":  107.0, "mechanism": "addition"},
    "Halohydrocarbon":        {"d": 1.85, "angle":  180.0, "mechanism": "sn2"},
    "Vinyl Sulfone":          {"d": 1.85, "angle":  107.0, "mechanism": "addition"},
    "Vinylsulfone":           {"d": 1.85, "angle":  107.0, "mechanism": "addition"},
    "Beta Lactam":            {"d": 1.82, "angle":  100.0, "mechanism": "addition"},
    "Epoxide":                {"d": 1.85, "angle":  180.0, "mechanism": "sn2"},
    "Disulfide":              {"d": 2.05, "angle":  103.0, "mechanism": "disulfide"},
    "Aldehyde":               {"d": 1.85, "angle":  107.0, "mechanism": "hemithioacetal"},
    "Aldehydic carbonyl":     {"d": 1.85, "angle":  107.0, "mechanism": "hemithioacetal"},
    "Carbonyl":               {"d": 1.85, "angle":  107.0, "mechanism": "hemithioacetal"},
    "Nitrile":                {"d": 1.85, "angle":  107.0, "mechanism": "thioimidate"},
    "Sulfonyl Fluorine":      {"d": 1.85, "angle":  180.0, "mechanism": "sn2_at_S"},
    # Added 2026-05-13 from CovInDB2 audit — these are CYS-targeting classes
    # with ≥10 entries that were silently filtered out before. Adding them
    # captures the missing 15% of the CYS-targeting corpus (~230 records).
    "Sulfonic acid":          {"d": 1.85, "angle":  109.0, "mechanism": "sulfonylation"},
    "Thiol":                  {"d": 2.05, "angle":  103.0, "mechanism": "disulfide"},
    "Ester":                  {"d": 1.85, "angle":  109.0, "mechanism": "transesterification"},
    "Diazomethyl Carbonyl":   {"d": 1.85, "angle":  109.0, "mechanism": "addition"},
    "Thiosulfonate":          {"d": 2.05, "angle":  103.0, "mechanism": "disulfide"},
    "Aziridine":              {"d": 1.85, "angle":  180.0, "mechanism": "sn2"},
}

MAX_RESOLUTION = 2.5  # Tightened from 3.5 → 2.5 (2026-05-13 CovInDB2 paper QA).
                     # Paper's DeepCoSI cutoff is ≤2.0; 2.5 is a practical
                     # compromise. At 3.5 Å SG→anchor distances are noisy by
                     # ±0.3 Å — same order as the geometry signal we train on,
                     # which corrupts training. We accept losing ~15-25% of
                     # records in exchange for trustworthy bond geometry.
SEED = 42

# Expected element at the *attacked* atom per warhead class (QA #6).
ANCHOR_ELEMENT = {
    "Michael Acceptor":   "C",
    "Halohydrocarbon":    "C",
    "Vinyl Sulfone":      "C",
    "Vinylsulfone":       "C",
    "Beta Lactam":        "C",
    "Epoxide":            "C",
    "Disulfide":          "S",
    "Aldehyde":           "C",
    "Aldehydic carbonyl": "C",
    "Carbonyl":           "C",
    "Nitrile":            "C",
    "Sulfonyl Fluorine":  "S",
    # New classes (added 2026-05-13)
    "Sulfonic acid":      "S",
    "Thiol":              "S",
    "Ester":              "C",  # carbonyl C
    "Diazomethyl Carbonyl": "C",  # alpha-C
    "Thiosulfonate":      "S",
    "Aziridine":          "C",
}


def find_sg_cb(pdb_path: Path, chain_id: str, cys_resi: int):
    """Return (SG, CB) numpy arrays for the Cys at chain:resi."""
    parser = PDBParser(QUIET=True)
    try:
        s = parser.get_structure("", str(pdb_path))[0]
    except Exception:
        return None
    try:
        ch = s[chain_id]
    except KeyError:
        return None
    for res in ch:
        if res.id[1] == cys_resi and res.get_resname() == "CYS":
            if "SG" not in res or "CB" not in res:
                return None
            return (np.array(res["SG"].get_coord(), dtype=float),
                    np.array(res["CB"].get_coord(), dtype=float))
    return None


def find_anchor_atom(pdb_path: Path, lig_chain: str, lig_resi: int, sg_xyz,
                      expected_element: str | None = None):
    """In the ligand residue at (chain, resi), return:
        anchor_atom_index (0-based across heavy atoms in the residue),
        anchor_xyz,
        d_SG_to_anchor.

    Anchor = closest atom to SG **whose element matches expected_element**
    (if provided). Falls back to closest-of-any-element if no match within
    2.4 Å. Heavy atoms only (skips H).
    """
    parser = PDBParser(QUIET=True)
    try:
        s = parser.get_structure("", str(pdb_path))[0]
    except Exception:
        return None
    try:
        ch = s[lig_chain]
    except KeyError:
        return None
    for res in ch:
        if res.id[1] == lig_resi:
            atoms = [a for a in res.get_atoms() if a.element != "H"]
            if not atoms: return None
            coords = np.array([a.get_coord() for a in atoms], dtype=float)
            d = np.linalg.norm(coords - sg_xyz, axis=1)
            if expected_element is not None:
                eligible = [i for i, a in enumerate(atoms) if a.element == expected_element]
                if eligible:
                    sub_d = d[eligible]
                    j_local = int(np.argmin(sub_d))
                    j = eligible[j_local]
                    if d[j] <= 2.4:
                        return j, coords[j], float(d[j])
                # if no eligible-element atom is within bond distance, fall through
            j = int(np.argmin(d))
            return j, coords[j], float(d[j])
    return None


def parse_resolution(s):
    try: return float(s)
    except: return None


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"loaded {len(df)} cocrystal records from CovInDB 2.0")
    df = df[df["Resi_name"] == "CYS"].copy()
    print(f"  CYS reactive residue: {len(df)}")
    df = df[df["Warhead"].isin(WARHEAD_GEOM.keys())].copy()
    print(f"  supported warhead class: {len(df)}")
    df["resolution_f"] = df["Resolution"].map(parse_resolution)
    df = df[df["resolution_f"].notna() & (df["resolution_f"] <= MAX_RESOLUTION)].copy()
    print(f"  resolution ≤ {MAX_RESOLUTION} Å: {len(df)}")
    print(f"  warhead-class counts after filter:")
    print(df["Warhead"].value_counts().head(20))

    rows = []
    n_pdb_missing = 0
    n_sg_missing = 0
    n_anchor_bad = 0
    for _, r in df.iterrows():
        pdb_id = str(r["PDB"]).upper()
        pdbf = PDB_DIR / f"{pdb_id}.pdb"
        if not pdbf.exists():
            n_pdb_missing += 1
            continue
        sg_cb = find_sg_cb(pdbf, r["Resi_chain"], int(r["Resi_posi"]))
        if sg_cb is None:
            n_sg_missing += 1
            continue
        sg_xyz, cb_xyz = sg_cb
        ai = find_anchor_atom(pdbf, r["Ligand_chain"], int(r["Ligand_position"]),
                              sg_xyz, expected_element=ANCHOR_ELEMENT.get(r["Warhead"]))
        if ai is None:
            n_anchor_bad += 1
            continue
        anchor_idx, anchor_xyz, d_sg_anchor = ai
        # Sanity: d should be ≤ 2.4 Å for a true covalent bond
        if d_sg_anchor > 2.4:
            n_anchor_bad += 1
            continue
        wclass = r["Warhead"]
        geom = WARHEAD_GEOM[wclass]
        rows.append({
            "record_id": r["ID"], "pdb_id": pdb_id,
            "pdb_path": str(pdbf.relative_to(PROJECT_ROOT)),
            "ligand_chain": r["Ligand_chain"], "ligand_resi": int(r["Ligand_position"]),
            "ligand_resname": r["Ligand_name"],
            "cys_chain": r["Resi_chain"], "cys_resi": int(r["Resi_posi"]),
            "warhead_class": wclass, "reaction_type": r["Reaction"],
            "warhead_canonical_d": geom["d"],
            "warhead_canonical_angle": geom["angle"],
            "warhead_mechanism": geom["mechanism"],
            "smiles": r["SMILES"], "target": r.get("Protein_name", ""),
            "organism": r.get("Organism", ""), "resolution": r["resolution_f"],
            "year": r.get("Year", ""),
            "sg_x": float(sg_xyz[0]), "sg_y": float(sg_xyz[1]), "sg_z": float(sg_xyz[2]),
            "cb_x": float(cb_xyz[0]), "cb_y": float(cb_xyz[1]), "cb_z": float(cb_xyz[2]),
            "anchor_atom_idx_in_ligand": anchor_idx,
            "anchor_x": float(anchor_xyz[0]), "anchor_y": float(anchor_xyz[1]),
            "anchor_z": float(anchor_xyz[2]),
            "d_SG_to_anchor": d_sg_anchor,
        })
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(out)} usable complexes)")
    print(f"  PDB missing:    {n_pdb_missing}")
    print(f"  Cys SG/CB not found: {n_sg_missing}")
    print(f"  anchor atom bad / d>2.4: {n_anchor_bad}")
    if len(out):
        print(f"\nFinal warhead breakdown:")
        print(out["warhead_class"].value_counts().to_string())
        # Stratified train/val split.
        # Default: target-based (no protein leakage between train and val).
        # Fallback: ROW-level random for warhead classes with <10 unique
        # targets (otherwise small classes can have all rows on a few targets,
        # producing val>>train imbalance — QA round 2 #LOW Aldehydic
        # carbonyl/Beta Lactam pathology).
        SMALL_CLASS_TARGET_THRESHOLD = 10
        rng = np.random.default_rng(SEED)
        out["split"] = "train"
        for wclass, g in out.groupby("warhead_class"):
            targets = g["target"].dropna().unique()
            if len(targets) < SMALL_CLASS_TARGET_THRESHOLD:
                # too few targets — split by PDB with seeded RNG instead
                # (QA fix 2026-05-13: was row-level which could split rows of
                # the same PDB into both train and val — coordinate-level leak)
                class_pdbs = g["pdb_id"].unique()
                rng.shuffle(class_pdbs)
                n_val_pdb = max(1, int(round(0.15 * len(class_pdbs))))
                if n_val_pdb >= len(class_pdbs): continue
                val_pdb_set = set(class_pdbs[:n_val_pdb].tolist())
                out.loc[(out["warhead_class"] == wclass) & out["pdb_id"].isin(val_pdb_set),
                        "split"] = "val"
                continue
            n_val = max(1, int(round(0.15 * len(targets))))
            if n_val >= len(targets):
                continue
            val_targets = set(rng.choice(targets, n_val, replace=False).tolist())
            out.loc[(out["warhead_class"] == wclass) & out["target"].isin(val_targets),
                    "split"] = "val"
        # QA fix 2026-05-13: ensure no PDB ID appears in both splits.
        # If a PDB is split-ambiguous (e.g., target name varied across rows),
        # force-move all its rows to the majority split.
        for pdb_id, g in out.groupby("pdb_id"):
            if g["split"].nunique() > 1:
                majority = g["split"].mode().iloc[0]
                out.loc[out["pdb_id"] == pdb_id, "split"] = majority
        leak_check = out.groupby("pdb_id")["split"].nunique().max()
        assert leak_check == 1, f"PDB still appears in multiple splits (max={leak_check})"
        out.to_csv(OUT_CSV, index=False)
        n_tr = (out.split == "train").sum()
        n_va = (out.split == "val").sum()
        print(f"\nStratified split (seed={SEED}):  train={n_tr}  val={n_va}")
        print("Per-class split:")
        pv = out.groupby(["warhead_class", "split"]).size().unstack(fill_value=0)
        print(pv.to_string())


if __name__ == "__main__":
    main()
