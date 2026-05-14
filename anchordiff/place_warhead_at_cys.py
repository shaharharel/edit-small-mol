"""Place an acrylamide warhead in the prereactive covalent complex pose
near a target cysteine SG, then write a 5-atom warhead SDF.

This is the CORRECT setup for DiffSBDD inpaint: instead of using the warhead's
position from a docked parent ligand (which may be far from the target Cys),
we place the warhead atoms at canonical Michael-addition geometry relative to
the target cysteine SG, and use that SDF as --fix_atoms.

Geometry (the prereactive complex, just before SG-Cβ bond formation):
  - SG is at the cysteine SG position (read from receptor.pdb)
  - C_β at 1.85 Å from SG, along an outward direction (pointing away from
    Cys CB into pocket)
  - C_α placed so ∠(SG, C_β, C_α) = 107° (Bürgi-Dunitz)
  - C_carb placed so dihedral(SG, Cβ, Cα, Ccarb) = 0° (planar approach)
  - O carbonyl + N amide placed at canonical sp2 geometry
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser

from anchordiff.config import ZAP70_CYS346, BTK_CYS481, EGFR_CYS797, KRAS_G12C

# Bond lengths (Å) and angles (°)
SG_CB     = 1.85   # S—C single bond at prereactive
CB_CA     = 1.34   # C=C double (sp2 carbon-carbon)
CA_CCARB  = 1.49   # C—C single (sp2-sp2)
CCARB_O   = 1.23   # C=O carbonyl
CCARB_N   = 1.34   # C—N amide

ANG_S_CB_CA   = 107.0  # Bürgi-Dunitz
ANG_CB_CA_CC  = 120.0  # sp2 backbone
ANG_CA_CC_O   = 120.0  # sp2 carbonyl
ANG_CA_CC_N   = 116.0  # sp2 amide


def get_cys_sg_and_cb(receptor_pdb: Path, resi: int, chain: str = "A"):
    """Return (SG, CB) coords of the target cysteine."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", str(receptor_pdb))
    model = struct[0]
    if chain not in model:
        chain = list(model.child_dict.keys())[0]
    res = model[chain][resi]
    if res.get_resname() != "CYS":
        raise ValueError(f"{chain}:{resi} is {res.get_resname()}, not CYS")
    sg = np.array(res["SG"].get_coord(), dtype=float)
    cb = np.array(res["CB"].get_coord(), dtype=float)
    return sg, cb


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _perp_to(v):
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, a))


def build_prereactive_warhead(sg: np.ndarray, cb_cys: np.ndarray):
    """Return (5,3) array of warhead atom coords at the prereactive complex.

    Atom order: [C_β, C_α, C_carb, O, N]

    Geometry:
      - C_β at 1.85 Å from SG, along the Cys CB→SG direction continued outward
        (so the warhead approaches SG from the side opposite Cys CB).
      - C_α placed at angle 107° at C_β, in a plane containing SG.
      - C_carb continues the sp2 backbone with 120° at C_α.
      - O and N branch off C_carb at canonical amide geometry.
    """
    # Direction: from Cys CB to SG, continued (so warhead approaches from outside)
    out_dir = _unit(sg - cb_cys)
    p_cb = sg + SG_CB * out_dir

    # Place C_α at angle ∠(S, C_β, C_α) = 107°.
    # The angle is between (S - C_β) and (C_α - C_β). Build (C_α - C_β) by
    # rotating (S - C_β) by 107° about a perpendicular axis (so the angle
    # between the rotated vec and (S - C_β) is 107°).
    v_Cb_to_S = sg - p_cb
    perp = _perp_to(v_Cb_to_S)
    K = _rot(perp, np.radians(ANG_S_CB_CA))
    direction_Cb_to_Ca = K @ _unit(v_Cb_to_S)
    p_ca = p_cb + CB_CA * direction_Cb_to_Ca

    # Place C_carb so dihedral(S, Cβ, Cα, Ccarb) = 0 (planar) and
    # ∠(C_β, C_α, C_carb) = 120°.
    # The angle is between (C_β - C_α) and (C_carb - C_α). Build it by rotating
    # (C_β - C_α) by 120° about the normal to plane(S, Cβ, Cα). For dihedral=0
    # the rotation axis must be the plane normal pointing such that C_carb
    # ends up on the SG side of the C_β-C_α axis.
    v_Ca_to_Cb = _unit(p_cb - p_ca)
    n_plane = _unit(np.cross(p_cb - sg, p_ca - p_cb))
    # Negative rotation puts Ccarb in cis relative to SG (dihedral=0).
    K = _rot(n_plane, np.radians(-ANG_CB_CA_CC))
    direction_Ca_to_CC = K @ v_Ca_to_Cb
    p_carb = p_ca + CA_CCARB * direction_Ca_to_CC

    # Place O carbonyl: angle ∠(C_α, C_carb, O) = 120°, in same plane.
    v_CC_to_Ca = _unit(p_ca - p_carb)
    K = _rot(n_plane, np.radians(-ANG_CA_CC_O))
    direction_CC_to_O = K @ v_CC_to_Ca
    p_o = p_carb + CCARB_O * direction_CC_to_O

    # Place N amide: angle ∠(C_α, C_carb, N) = 116°, on the opposite side
    # of the C_α-C_carb axis from O (cis amide).
    K = _rot(n_plane, np.radians(ANG_CA_CC_N))
    direction_CC_to_N = K @ v_CC_to_Ca
    p_n = p_carb + CCARB_N * direction_CC_to_N

    return np.array([p_cb, p_ca, p_carb, p_o, p_n])


def _rot(axis, theta):
    """Rodrigues rotation matrix."""
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def write_warhead_sdf(coords: np.ndarray, out_sdf: Path):
    """Write a 5-atom acrylamide warhead SDF with bonds 0=1, 1-2, 2=3, 2-4."""
    rwm = Chem.RWMol()
    syms = ["C", "C", "C", "O", "N"]
    for s in syms:
        a = Chem.Atom(s)
        a.SetNoImplicit(False)
        rwm.AddAtom(a)
    rwm.AddBond(0, 1, Chem.BondType.DOUBLE)
    rwm.AddBond(1, 2, Chem.BondType.SINGLE)
    rwm.AddBond(2, 3, Chem.BondType.DOUBLE)
    rwm.AddBond(2, 4, Chem.BondType.SINGLE)
    mol = rwm.GetMol()
    conf = Chem.Conformer(5)
    for i in range(5):
        conf.SetAtomPosition(i, coords[i].tolist())
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    w.write(mol)
    w.close()


def main():
    pockets = PROJECT_ROOT / "anchordiff" / "pockets"
    for tgt in [ZAP70_CYS346, BTK_CYS481, EGFR_CYS797, KRAS_G12C]:
        pock_dir = pockets / tgt.name
        receptor = pock_dir / "receptor.pdb"
        sg, cb = get_cys_sg_and_cb(receptor, tgt.cys_residue)
        print(f"=== {tgt.name} ===")
        print(f"  Cys{tgt.cys_residue} SG = ({sg[0]:.3f}, {sg[1]:.3f}, {sg[2]:.3f})")
        print(f"  Cys{tgt.cys_residue} CB = ({cb[0]:.3f}, {cb[1]:.3f}, {cb[2]:.3f})")
        coords = build_prereactive_warhead(sg, cb)
        # Sanity check
        d_S_Cb = np.linalg.norm(sg - coords[0])
        # angle ∠(S, C_β, C_α)
        v1 = sg - coords[0]; v2 = coords[1] - coords[0]
        ang = np.degrees(np.arccos(np.clip(np.dot(_unit(v1), _unit(v2)), -1, 1)))
        # dihedral
        b1 = coords[0] - sg
        b2 = coords[1] - coords[0]
        b3 = coords[2] - coords[1]
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        m1 = np.cross(n1, _unit(b2))
        x = np.dot(n1, n2); y = np.dot(m1, n2)
        dih = np.degrees(np.arctan2(y, x))
        print(f"  prereactive: d(S,Cβ)={d_S_Cb:.3f} Å,  ∠(S,Cβ,Cα)={ang:.2f}°,  φ={dih:.2f}°")
        out_sdf = pock_dir / "warhead_at_cys.sdf"
        write_warhead_sdf(coords, out_sdf)
        print(f"  wrote {out_sdf}")


if __name__ == "__main__":
    main()
