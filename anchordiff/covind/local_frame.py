"""D: Cβ-anchored SE(3) local frame.

The flagship architectural change. Every input coordinate is transformed
into a Cys-SG-anchored local frame BEFORE feeding to DiffSBDD:

  origin  = SG
  z-axis  = SG → Cβ direction (the privileged "outward to pocket" axis)
  x-axis  = perpendicular component of (Cα − Cβ) → in the (SG, Cβ, Cα) plane
  y-axis  = z × x  (right-hand rule)

In this frame the canonical prereactive complex sits at:
  C_β  = (0, 0, d_warhead)            ≈ (0, 0, 1.85)
  C_α  = at angle θ_warhead from +z, in the xz-plane

So canonical geometry is the **origin and a small set of fixed displacements**,
not a constraint to be enforced.

Forward / inverse transforms:
  to_local(x_global) = R^T @ (x_global − SG)
  to_global(x_local) = R @ x_local + SG
"""
from __future__ import annotations
import numpy as np
import torch


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_frame(sg: np.ndarray, cb: np.ndarray, ca: np.ndarray | None = None):
    """Return (R, t) so that x_local = R^T @ (x_global − t), with:
      - origin = t = SG
      - z-axis = SG → Cβ direction
      - x-axis = in plane (SG, Cβ, Cα), perpendicular to z
      - y-axis = z × x
    If Cα is None, falls back to an arbitrary perpendicular (then the frame
    is fixed up to rotation about z — fine for unconditional generation, but
    not preserved across training examples; pass Cα when you have it).

    Returns R (3,3) and t (3,) as numpy float64 arrays.
    """
    sg = np.asarray(sg, dtype=float)
    cb = np.asarray(cb, dtype=float)
    z = _unit(cb - sg)
    if ca is not None:
        ca = np.asarray(ca, dtype=float)
        v = ca - sg
        v_perp = v - np.dot(v, z) * z
        # CRITICAL: check colinearity *before* the unit-vector divide, otherwise
        # we get a NaN-poisoned frame on degenerate Cys geometry (QA #8).
        if np.linalg.norm(v_perp) < 1e-6:
            ca = None
        else:
            x = _unit(v_perp)
    if ca is None:
        # arbitrary perpendicular
        a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = _unit(np.cross(z, a))
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])           # columns = frame axes in global coords
    return R, sg


def to_local(coords_global: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """coords_global: (N, 3) → coords_local: (N, 3)."""
    return (coords_global - t) @ R           # R^T @ v == v @ R when R is orthonormal


def to_global(coords_local: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """coords_local: (N, 3) → coords_global: (N, 3)."""
    return coords_local @ R.T + t


# Torch wrappers — same math, but on tensors
def to_local_torch(coords_global: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    return (coords_global - t.unsqueeze(0)) @ R


def to_global_torch(coords_local: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    return coords_local @ R.T + t.unsqueeze(0)


def canonical_attack_and_companion(d: float, theta_deg: float,
                                   companion_bond_len: float = 1.34) -> tuple[np.ndarray, np.ndarray]:
    """Return (attacked_atom, companion_atom) positions in the local frame.

    *attacked_atom* is at (0, 0, d) along the +z axis (the privileged direction
    from SG into the pocket).  This is the carbon that forms the new S-C bond
    (Cβ of an acrylamide, Cα of a chloroacetamide, carbonyl C of an aldehyde,
    etc.).

    *companion_atom* is the second warhead atom that defines the warhead's
    orientation (breaks rotational symmetry around the +z axis):
      - Michael Acceptor (θ≈107°): the sp2 partner (Cα of the alkene). Ends up
        in the xz-plane, "tilted" away from +z by (180°-θ) toward +x.
      - Halohydrocarbon (θ=180°, SN2 backside): the **leaving group** (Cl).
        Ends up directly above the attacked carbon at (0, 0, d + 1.34).
      - Aldehyde / Nitrile / Carbonyl: the partner heavy atom (O or N) tilted
        by (180°-θ) from +z.

    The math `companion = attacked + L * (sin θ, 0, -cos θ)` is identical for
    all classes — the *meaning* of the companion atom is what varies.
    """
    theta = np.radians(theta_deg)
    attacked  = np.array([0.0, 0.0, d])
    companion = attacked + companion_bond_len * np.array([np.sin(theta), 0.0, -np.cos(theta)])
    return attacked, companion


# Backward-compat alias for older callers — kept for one release, will remove
canonical_warhead_origin = canonical_attack_and_companion


if __name__ == "__main__":
    # Smoke test: round-trip transformation
    sg = np.array([18.888, -3.650, -29.979])      # ZAP70 Cys346 SG
    cb = np.array([17.193, -4.221, -30.247])      # CB
    ca = np.array([16.292, -4.283, -28.999])      # CA
    R, t = compute_frame(sg, cb, ca)
    print("Frame R:\n", R)
    print("Frame t:", t)
    # Check Cb is at (0, 0, d) in local frame
    cb_local = to_local(cb[None, :], R, t)[0]
    print(f"CB in local: ({cb_local[0]:.3f}, {cb_local[1]:.3f}, {cb_local[2]:.3f})")
    # Round-trip
    rt = to_global(to_local(np.array([sg, cb, ca]), R, t), R, t)
    print(f"Round-trip max error: {np.max(np.abs(rt - np.array([sg, cb, ca]))):.2e}")
    # Canonical (attack, companion) at d=1.85
    for label, theta in [("acrylamide", 107), ("chloroacetamide (SN2)", 180),
                          ("aldehyde (BD-like)", 107), ("nitrile", 107)]:
        att, comp = canonical_attack_and_companion(1.85, theta)
        d_pair = np.linalg.norm(att - comp)
        print(f"  {label:30s}  attack={att}  companion={comp}  d={d_pair:.3f}")
