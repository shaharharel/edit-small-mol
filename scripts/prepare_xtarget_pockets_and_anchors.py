"""Stage pocket PDBs + Bürgi-Dunitz anchor JSONs for cross-target generalization.

Two targets:
  BTK Cys481 — uses PDB 5P9J (ibrutinib-bound; 5P9I unavailable on disk under the
      raw_covindb2 set; 5P9J is the same series, ibrutinib-bound, equivalent for
      pocket geometry).
  KRAS G12C Cys12 — uses PDB 6OIM (sotorasib-bound, Switch-II pocket).

Output (all paths absolute to project root):
  data/lingo3dmol_smoke/btk_pocket_cys481.pdb
  data/lingo3dmol_smoke/kras_pocket_cys12.pdb
  data/lingo3dmol_anchor_btk_cys481.json
  data/lingo3dmol_anchor_kras_cys12.json

Pocket extraction: 5 Å around the target Cys SG. Saves only ATOM records (no
HETATM) so RDKit MolFromPDBFile can parse cleanly via the same code path used
by Lingo3DMol's pocket_code_all.loadMacroPDBInfo.

Anchor JSON: same schema as data/lingo3dmol_anchor_zap70_cys346.json. For each
target we compute:
  sg_pos, ca_pos, cb_pos_cys (from the Cys residue)
  cb_pos_target — the ideal β-carbon position of the warhead, 1.85 Å from SG,
                  along the Bürgi-Dunitz attack vector (107° from S->CA backbone)
  anchor_attack_vector — unit vector from CB_target toward SG (the bond being
                          made), used by the inpaint scaffold builder.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POCKET_OUT_DIR = PROJECT_ROOT / "data" / "lingo3dmol_smoke"
ANCHOR_OUT_DIR = PROJECT_ROOT / "data"

# --- Target definitions --------------------------------------------------
TARGETS = {
    "btk_cys481": {
        "target_name": "BTK Cys481 (CHEMBL5251)",
        "pdb_src": PROJECT_ROOT / "data/covbinder/raw/5P9I.pdb",
        # Fallback if 5P9I is missing.
        "pdb_src_fallback": PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB/5P9J.pdb",
        "chain": "A",
        "resi": 481,
        "pocket_out": POCKET_OUT_DIR / "btk_pocket_cys481.pdb",
        "anchor_out": ANCHOR_OUT_DIR / "lingo3dmol_anchor_btk_cys481.json",
    },
    "kras_cys12": {
        "target_name": "KRAS-G12C Cys12 (CHEMBL1075094)",
        "pdb_src": PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB/6OIM.pdb",
        "pdb_src_fallback": None,
        "chain": "A",
        "resi": 12,
        "pocket_out": POCKET_OUT_DIR / "kras_pocket_cys12.pdb",
        "anchor_out": ANCHOR_OUT_DIR / "lingo3dmol_anchor_kras_cys12.json",
    },
}

POCKET_RADIUS_A = 10.0      # Å around CYS SG; matches the ZAP70 reference (~49 residues)
WARHEAD_CB_DIST = 1.85      # ideal β-C to SG distance for a Michael adduct
BURGI_DUNITZ_DEG = 107.0    # ideal attack angle (S -> CA backbone vs S -> CB_target)


# --- PDB parser ----------------------------------------------------------
def _parse_atom(line: str) -> dict | None:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None
    try:
        return {
            "record": line[0:6].strip(),
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "altloc": line[16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21].strip(),
            "resi": int(line[22:26]),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "raw": line.rstrip(),
        }
    except (ValueError, IndexError):
        return None


def _read_pdb_atoms(pdb_path: Path) -> list[dict]:
    out: list[dict] = []
    with pdb_path.open() as f:
        for line in f:
            if line.startswith("ENDMDL"):
                # Only keep first model.
                break
            atom = _parse_atom(line)
            if atom is None:
                continue
            # Take primary alt-loc only.
            if atom["altloc"] not in ("", "A"):
                continue
            out.append(atom)
    return out


def _find_cys_atoms(atoms: list[dict], chain: str, resi: int) -> dict:
    """Return {name: atom} for the target Cys residue."""
    hits = [a for a in atoms if a["chain"] == chain and a["resi"] == resi
            and a["resname"] == "CYS" and a["record"] == "ATOM"]
    if not hits:
        # Loosen chain filter for single-chain structures.
        hits = [a for a in atoms if a["resi"] == resi and a["resname"] == "CYS"
                and a["record"] == "ATOM"]
    by_name = {a["name"]: a for a in hits}
    for needed in ("SG", "CB", "CA", "N", "C"):
        assert needed in by_name, f"missing CYS atom {needed!r} for chain={chain} resi={resi}; have {list(by_name)}"
    return by_name


def _atom_xyz(atom: dict) -> np.ndarray:
    return np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float64)


def _extract_pocket(atoms: list[dict], sg_xyz: np.ndarray, radius_a: float, chain: str) -> list[dict]:
    """Return ATOM records of all residues with ANY atom within radius_a of SG."""
    keep_residues = set()
    for a in atoms:
        if a["record"] != "ATOM":
            continue
        if a["chain"] != chain:
            continue
        d = np.linalg.norm(_atom_xyz(a) - sg_xyz)
        if d <= radius_a:
            keep_residues.add((a["chain"], a["resi"]))
    pocket = [a for a in atoms
              if a["record"] == "ATOM"
              and (a["chain"], a["resi"]) in keep_residues]
    pocket.sort(key=lambda a: (a["chain"], a["resi"], a["serial"]))
    return pocket


def _compute_bd_frame(sg: np.ndarray, ca: np.ndarray, cb_cys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Bürgi-Dunitz attack vector for a Michael addition to a Cys SG.

    The new warhead β-carbon (CB_target) lies WARHEAD_CB_DIST away from SG,
    along a direction that makes BURGI_DUNITZ_DEG with the S->CA backbone bond
    (matching the ZAP70 anchor convention).

    Returns (cb_pos_target, anchor_attack_vector, frame_e1, frame_e2).
    """
    # Build the BD frame: e1 along S->CA, e2 orthogonal in the S-CA-CB_cys plane.
    v_sg_ca = ca - sg
    e1 = v_sg_ca / np.linalg.norm(v_sg_ca)
    v_sg_cb = cb_cys - sg
    # Remove e1 component.
    e2_raw = v_sg_cb - np.dot(v_sg_cb, e1) * e1
    if np.linalg.norm(e2_raw) < 1e-6:
        # Degenerate; fall back to an arbitrary orthogonal direction.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, e1)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0])
        e2_raw = helper - np.dot(helper, e1) * e1
    e2 = e2_raw / np.linalg.norm(e2_raw)

    # Direction from SG to CB_target: rotate -e1 (towards CA reversed? no -- the
    # ZAP70 convention is that anchor_attack_vector points FROM CB_target TOWARDS
    # SG, i.e. it's the unit vector along the forming bond, INTO the cysteine.
    # The cb_target sits OPPOSITE the e1 direction at angle (180-BD) from -e1.
    # Easier formulation: angle between S->CA and S->CB_target is BURGI_DUNITZ_DEG.
    # We construct S->CB_target = cos(BD)*e1 + sin(BD)*e2 (in the e1,e2 plane).
    theta = np.radians(BURGI_DUNITZ_DEG)
    s_to_cb_target = np.cos(theta) * e1 + np.sin(theta) * e2
    s_to_cb_target /= np.linalg.norm(s_to_cb_target)
    cb_target = sg + WARHEAD_CB_DIST * s_to_cb_target
    # Anchor attack vector: from CB_target back to SG (forming bond direction).
    av = sg - cb_target
    av /= np.linalg.norm(av)
    return cb_target, av, e1, e2


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu = u / np.linalg.norm(u)
    cv = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(cu, cv), -1.0, 1.0))))


def _write_pocket_pdb(pocket: list[dict], out_path: Path, header: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"REMARK   1 {header}\n")
        for a in pocket:
            f.write(a["raw"] + "\n")
        f.write("END\n")


def process_target(tag: str, spec: dict) -> dict:
    src = spec["pdb_src"] if spec["pdb_src"].exists() else spec["pdb_src_fallback"]
    assert src is not None and src.exists(), f"no PDB source for {tag}: tried {spec['pdb_src']} and {spec['pdb_src_fallback']}"
    print(f"\n[{tag}] using PDB: {src}")

    atoms = _read_pdb_atoms(src)
    print(f"[{tag}] read {len(atoms)} atom records (ATOM+HETATM)")

    cys = _find_cys_atoms(atoms, spec["chain"], spec["resi"])
    sg = _atom_xyz(cys["SG"])
    ca = _atom_xyz(cys["CA"])
    cb = _atom_xyz(cys["CB"])
    print(f"[{tag}] CYS {spec['resi']} SG={sg.tolist()}  CA={ca.tolist()}  CB={cb.tolist()}")

    pocket = _extract_pocket(atoms, sg, POCKET_RADIUS_A, spec["chain"])
    residues = sorted({(a["chain"], a["resi"], a["resname"]) for a in pocket})
    print(f"[{tag}] pocket: {len(pocket)} atoms, {len(residues)} residues within {POCKET_RADIUS_A} Å of SG")

    _write_pocket_pdb(pocket, spec["pocket_out"],
                      header=f"{spec['target_name']} pocket (5A around Cys{spec['resi']}) from {src.name}")
    print(f"[{tag}] wrote pocket -> {spec['pocket_out']}")

    cb_target, av, e1, e2 = _compute_bd_frame(sg, ca, cb)
    bd_check = _angle_deg(ca - sg, cb_target - sg)
    d_check = float(np.linalg.norm(cb_target - sg))
    print(f"[{tag}] cb_target={cb_target.tolist()}")
    print(f"[{tag}] anchor_attack_vector={av.tolist()}")
    print(f"[{tag}] |CB_target - SG|={d_check:.3f} Å (target 1.85)")
    print(f"[{tag}] angle(CA-S-CB_target)={bd_check:.2f}° (target 107)")

    anchor = {
        "target": spec["target_name"],
        "pdb_source": str(src.relative_to(PROJECT_ROOT)),
        "chain": spec["chain"],
        "resi": spec["resi"],
        "resname": "CYS",
        "sg_pos": sg.tolist(),
        "ca_pos": ca.tolist(),
        "cb_pos_cys": cb.tolist(),
        "cb_pos_target": cb_target.tolist(),
        "anchor_attack_vector": av.tolist(),
        "ideal_warhead_smiles_prefix": "C=CC(=O)N",
        "warhead_cb_distance_A": WARHEAD_CB_DIST,
        "burgi_dunitz_angle_deg": BURGI_DUNITZ_DEG,
        "angle_check_deg": bd_check,
        "frame_e1_minus_v_sg_ca": e1.tolist(),
        "frame_e2_in_plane": e2.tolist(),
        "notes": [
            "Anchor describes the IDEAL position of the warhead's β-carbon (the CH2= end",
            "of CH2=CH-C(=O)-NH-...) so that, upon Michael addition, the new C-S bond to",
            f"Cys{spec['resi']} SG forms at {WARHEAD_CB_DIST:.2f} Å along the Bürgi-Dunitz attack vector",
            f"({BURGI_DUNITZ_DEG:.0f}° from the S->CA backbone vector).",
            "Generated by scripts/prepare_xtarget_pockets_and_anchors.py.",
        ],
    }
    spec["anchor_out"].parent.mkdir(parents=True, exist_ok=True)
    spec["anchor_out"].write_text(json.dumps(anchor, indent=2))
    print(f"[{tag}] wrote anchor JSON -> {spec['anchor_out']}")

    return {
        "tag": tag,
        "pocket": str(spec["pocket_out"]),
        "anchor": str(spec["anchor_out"]),
        "n_pocket_atoms": len(pocket),
        "n_residues": len(residues),
        "sg_pos": sg.tolist(),
        "cb_pos_target": cb_target.tolist(),
        "bd_check_deg": bd_check,
        "warhead_cb_dist_A": d_check,
    }


def main():
    summary = {}
    for tag, spec in TARGETS.items():
        summary[tag] = process_target(tag, spec)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
