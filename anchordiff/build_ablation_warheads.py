"""Day 3 task: build warhead-pose ablation SDFs.

7 cohorts per target:
  (a) C_β position along Cys CB → SG direction, offset = -0.3, 0, +0.3 Å
  (b) warhead rotated about the S-Cβ axis by -20°, 0°, +20°
  (c) chloroacetamide replacing acrylamide (Cl-CH2-C(=O)-N, SN2 geometry — angle 180°)

For each variant we write a 5-atom SDF (acrylamide) or 5-atom SDF (chloroacetamide)
into `anchordiff/pockets/<target>/ablation/<variant>.sdf`. The downstream inpaint
script then references these via --fix_atoms.

(0, 0, acrylamide) is the baseline already used in Day 1 — confirmed by writing
an identical SDF as the "baseline" entry for completeness.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rdkit import Chem

from anchordiff.config import ZAP70_CYS346, BTK_CYS481
from anchordiff.place_warhead_at_cys import (
    get_cys_sg_and_cb, build_prereactive_warhead, write_warhead_sdf,
    _unit, _rot, SG_CB,
)
S_C_BOND = SG_CB  # alias

# Chloroacetamide canonical geometry (Cl-CH2-C(=O)-N, SN2 to Cys SG):
#  - S–Cα = 1.85 Å (SN2 attack distance equals new C-S bond length)
#  - ∠(S, Cα, Cl) = 180° (SN2 backside; Cl departs opposite to incoming S)
#  - ∠(S, Cα, C_carb) ≈ 109° (sp3 tetrahedral)
#  - The 5 atoms recorded in the SDF: Cα, Cl (leaving group), C_carb, O, N
#    Same atom order as acrylamide for downstream pipeline reuse: atom 0 is
#    the carbon attacked by SG; atom 1 holds the leaving group / second sp2
#    placeholder; atoms 2-4 = C_carb, O, N.
CL_C_BOND   = 1.79   # Å, C-Cl
C_C_SP3_SP2 = 1.51   # Cα-C_carb single (sp3 to sp2)
SN2_ANGLE   = 180.0  # ° backside attack
TETRA_ANGLE = 109.5  # ° tetrahedral S–Cα–Ccarb


def build_chloroacetamide_warhead(sg: np.ndarray, cb_cys: np.ndarray) -> np.ndarray:
    """Return 5×3 array: [Cα, Cl, C_carb, O, N] at SN2 prereactive geometry."""
    out_dir = _unit(sg - cb_cys)
    p_calpha = sg + S_C_BOND * out_dir          # the carbon being attacked
    # Cl directly opposite SG (backside): place along out_dir continuing
    p_cl     = p_calpha + CL_C_BOND * out_dir
    # C_carb at 109.5° from S-Cα, in some plane
    perp = _unit(np.cross(out_dir, np.array([0.0, 0.0, 1.0]) if abs(out_dir[2]) < 0.9
                                            else np.array([1.0, 0.0, 0.0])))
    K = _rot(perp, np.radians(TETRA_ANGLE))
    direction_to_Ccarb = K @ (-out_dir)         # rotate (S-attack dir) by 109.5°
    p_carb = p_calpha + C_C_SP3_SP2 * direction_to_Ccarb
    # O and N branching off C_carb (sp2 carbonyl, 120°)
    v_CC_to_Ca = _unit(p_calpha - p_carb)
    n_plane = _unit(np.cross(p_calpha - sg, p_carb - p_calpha))
    K = _rot(n_plane, np.radians(-120))
    p_o = p_carb + 1.23 * (K @ v_CC_to_Ca)
    K = _rot(n_plane, np.radians(116))
    p_n = p_carb + 1.34 * (K @ v_CC_to_Ca)
    return np.array([p_calpha, p_cl, p_carb, p_o, p_n])


def write_chloroacetamide_sdf(coords: np.ndarray, out_sdf: Path):
    """5-atom SDF: Cα-Cl, Cα-Ccarb, Ccarb=O, Ccarb-N (bonds 0-1 single, 0-2 single,
    2-3 double, 2-4 single)."""
    rwm = Chem.RWMol()
    for s in ["C", "Cl", "C", "O", "N"]:
        a = Chem.Atom(s)
        a.SetNoImplicit(False)
        rwm.AddAtom(a)
    rwm.AddBond(0, 1, Chem.BondType.SINGLE)
    rwm.AddBond(0, 2, Chem.BondType.SINGLE)
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


def apply_offset(coords: np.ndarray, sg: np.ndarray, cb_cys: np.ndarray, d_offset: float):
    """Translate all 5 warhead atoms by d_offset Å along SG → C_β outward direction."""
    out_dir = _unit(sg - cb_cys)
    return coords + d_offset * out_dir


def apply_rotation(coords: np.ndarray, sg: np.ndarray, deg: float):
    """Rotate atoms 1-4 (everything except Cβ) about the S–Cβ axis by `deg`."""
    p_cb = coords[0]
    axis = _unit(p_cb - sg)
    R = _rot(axis, np.radians(deg))
    out = coords.copy()
    for i in [1, 2, 3, 4]:
        out[i] = p_cb + R @ (coords[i] - p_cb)
    return out


def main():
    pockets = PROJECT_ROOT / "anchordiff" / "pockets"
    targets = [ZAP70_CYS346, BTK_CYS481]
    variants = []
    # (a) C_β offset
    for d in [-0.3, 0.0, 0.3]:
        variants.append(("offset", d, {"d_offset": d}))
    # (b) rotation about SG axis
    for deg in [-20.0, 20.0]:
        variants.append(("rot", deg, {"rot_deg": deg}))
    # (c) chloroacetamide
    variants.append(("warhead", "chloro", {"warhead": "chloroacetamide"}))

    for tgt in targets:
        pock_dir = pockets / tgt.name
        receptor = pock_dir / "receptor.pdb"
        sg, cb = get_cys_sg_and_cb(receptor, tgt.cys_residue)
        ablation_dir = pock_dir / "ablation"
        ablation_dir.mkdir(exist_ok=True)
        print(f"\n=== {tgt.name} ===")
        for kind, value, params in variants:
            base = build_prereactive_warhead(sg, cb)
            if kind == "offset":
                if params["d_offset"] == 0:
                    label = "baseline"
                else:
                    sign = "p" if params["d_offset"] > 0 else "m"
                    label = f"offset_{sign}{abs(params['d_offset']):.1f}"
                coords = apply_offset(base, sg, cb, params["d_offset"])
                out = ablation_dir / f"{label}.sdf"
                write_warhead_sdf(coords, out)
            elif kind == "rot":
                sign = "p" if value > 0 else "m"
                label = f"rot_{sign}{abs(int(value))}"
                coords = apply_rotation(base, sg, value)
                out = ablation_dir / f"{label}.sdf"
                write_warhead_sdf(coords, out)
            elif kind == "warhead":
                coords = build_chloroacetamide_warhead(sg, cb)
                out = ablation_dir / "warhead_chloroacetamide.sdf"
                write_chloroacetamide_sdf(coords, out)
            d_S_Cb = np.linalg.norm(sg - coords[0])
            print(f"  {label if kind != 'warhead' else 'chloroacetamide':20s}  d(S, atom0)={d_S_Cb:.3f}  →  {out.name}")


if __name__ == "__main__":
    main()
