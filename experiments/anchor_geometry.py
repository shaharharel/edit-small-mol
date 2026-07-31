"""Anchor-rotation helper for Lingo3DMol covalent-warhead inpaint scripts.

Implements the Bürgi-Dunitz-corrected scaffold placement:

  atom0 (warhead Cβ) sits at cb_pos_target = sg + 1.85 * attack_vector.
  atom1 (vinyl Cα) is placed so that the angle SG–atom0–atom1 ≈ 107°,
  i.e. the angle at atom0 between (atom0→SG) and (atom0→atom1) is the
  Bürgi-Dunitz angle measured at the electrophilic carbon, which is the
  chemistry-standard convention.

The previous (buggy) recipe aligned (atom1−atom0) COLINEAR with
attack_vector, which forced SG–atom0–atom1 = 180° (collinear) — clearly
wrong for a Michael addition.

Public API:

  rotate_scaffold_to_bd(
      coords:        np.ndarray [n_atoms, 3] — ETKDG-embedded scaffold
                                                xyz in any frame.
      cb_target:     np.ndarray [3]          — 3D point where atom0 must land
                                                (cb_pos_target from anchor JSON).
      av:            np.ndarray [3]          — unit attack vector
                                                (anchor_attack_vector).
      perp_in_plane: Optional[np.ndarray]    — preferred in-plane perpendicular
                                                (e.g. frame_e2_in_plane from
                                                anchor JSON). Falls back to a
                                                world-Z-derived perpendicular if
                                                None.
      bd_deg:        float                   — target Bürgi-Dunitz angle in
                                                degrees (default 107.0).
  )
    Returns rotated np.ndarray [n_atoms, 3] in the receptor frame, with
    atom0 at cb_target and atom1 placed at the BD angle.

  assert_bd_angle(rotated, sg_pos, bd_deg=107.0, tol_deg=5.0)
    Sanity-check the angle at atom0. Raises AssertionError if off.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def _rotation_matrix_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation that takes unit vector a to unit vector b."""
    a = _unit(a)
    b = _unit(b)
    cross = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(cross))
    if s < 1e-9:
        # parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180° flip — pick any axis perpendicular to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * float(np.dot(axis, a))
        axis = _unit(axis)
        K = np.array([
            [0.0,       -axis[2], axis[1]],
            [axis[2],   0.0,      -axis[0]],
            [-axis[1],  axis[0], 0.0],
        ])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([
        [0.0,       -cross[2], cross[1]],
        [cross[2],   0.0,      -cross[0]],
        [-cross[1],  cross[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _pick_perpendicular(av_norm: np.ndarray,
                        preferred: np.ndarray | None) -> np.ndarray:
    """Return a unit vector perpendicular to av_norm.

    If preferred is given and not parallel to av_norm, use its component
    orthogonal to av_norm (Gram-Schmidt). Otherwise fall back to cross with
    world Z (or world X if av_norm is along Z).
    """
    if preferred is not None:
        p = np.asarray(preferred, dtype=np.float64)
        # Project out the av_norm component
        p_perp = p - av_norm * float(np.dot(p, av_norm))
        if np.linalg.norm(p_perp) >= 1e-6:
            return _unit(p_perp)
    perp = np.cross(av_norm, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(av_norm, np.array([1.0, 0.0, 0.0]))
    return _unit(perp)


def rotate_scaffold_to_bd(coords: np.ndarray,
                          cb_target: np.ndarray,
                          av: np.ndarray,
                          perp_in_plane: np.ndarray | None = None,
                          bd_deg: float = 107.0) -> np.ndarray:
    """Rotate an ETKDG-embedded scaffold so that:
      - atom 0 sits at cb_target,
      - atom 1 makes a Bürgi-Dunitz angle (bd_deg ≈ 107°) at atom 0 between
        the (atom0->SG) direction (which is -av) and (atom0->atom1).

    This requires that (atom0->atom1) makes (180° - bd_deg) ≈ 73° with the
    attack vector av (i.e. atom1 leans slightly back toward the pocket but
    is NOT collinear with av).

    coords[0] and coords[1] are taken as the warhead Cβ (atom0) and vinyl Cα
    (atom1) respectively — this is the Lingo3DMol scaffold convention.

    The rotation is done in two stages:
      1) Rigid rotation R1 that maps (coords[1]-coords[0]) to the BD target
         direction (cos(180-bd)·av + sin(180-bd)·perp).
      2) (No second-axis disambiguation here — downstream callers can do
         their own in-plane rotation around the (atom0->atom1) axis, e.g.
         to centre the body on the pocket.)
    """
    coords = np.asarray(coords, dtype=np.float64)
    cb_target = np.asarray(cb_target, dtype=np.float64)
    av = np.asarray(av, dtype=np.float64)
    av_norm = _unit(av)

    perp = _pick_perpendicular(av_norm, perp_in_plane)

    # Target direction for (atom0 -> atom1).
    theta = np.radians(bd_deg)
    # angle between (atom0 -> atom1) and (atom0 -> SG) is theta;
    # (atom0 -> SG) = -av_norm, so the (atom0 -> atom1) direction is at
    # angle (pi - theta) with av_norm.
    alpha = np.pi - theta
    a1_unit = np.cos(alpha) * av_norm + np.sin(alpha) * perp
    a1_unit = _unit(a1_unit)

    v0 = coords[1] - coords[0]
    if np.linalg.norm(v0) < 1e-9:
        raise ValueError("atom1 == atom0 in scaffold coords")
    R = _rotation_matrix_from_to(v0, a1_unit)

    rotated = (coords - coords[0]) @ R.T + cb_target
    return rotated


def assert_bd_angle(rotated: np.ndarray,
                    sg_pos: np.ndarray,
                    bd_deg: float = 107.0,
                    tol_deg: float = 5.0) -> float:
    """Sanity-check the BD angle at atom0.

    Returns the measured angle in degrees. Raises AssertionError if not
    within tol_deg of bd_deg.
    """
    rotated = np.asarray(rotated, dtype=np.float64)
    sg_pos = np.asarray(sg_pos, dtype=np.float64)
    v_sg = sg_pos - rotated[0]
    v_a1 = rotated[1] - rotated[0]
    n_sg = float(np.linalg.norm(v_sg))
    n_a1 = float(np.linalg.norm(v_a1))
    if n_sg < 1e-9 or n_a1 < 1e-9:
        raise AssertionError("degenerate BD geometry (zero-length vector)")
    c = float(np.dot(v_sg, v_a1) / (n_sg * n_a1))
    c = max(-1.0, min(1.0, c))
    ang = float(np.degrees(np.arccos(c)))
    assert abs(ang - bd_deg) < tol_deg, (
        f"BD angle wrong: {ang:.1f}° (expected {bd_deg:.1f}° ± {tol_deg:.1f}°)"
    )
    return ang


def measured_bd_angle(rotated: np.ndarray, sg_pos: np.ndarray) -> float:
    """Compute the BD angle at atom0 without asserting. Returns degrees."""
    rotated = np.asarray(rotated, dtype=np.float64)
    sg_pos = np.asarray(sg_pos, dtype=np.float64)
    v_sg = sg_pos - rotated[0]
    v_a1 = rotated[1] - rotated[0]
    n_sg = float(np.linalg.norm(v_sg))
    n_a1 = float(np.linalg.norm(v_a1))
    if n_sg < 1e-9 or n_a1 < 1e-9:
        return float("nan")
    c = float(np.dot(v_sg, v_a1) / (n_sg * n_a1))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))
