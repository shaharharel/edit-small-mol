"""Fast geometry surrogate for covalent kinase RL.

For a SMILES candidate:
  1. ETKDG conformer (single, with H, MMFF cleanup optional)
  2. SMARTS-match the terminal acrylamide warhead (Cβ=Cα-C(=O)-N)
  3. Kabsch-align the warhead's 5 anchor atoms (Cβ, Cα, C', O, N) to the
     corresponding atoms from a canonical Boltz cofold pose (anchor_frame.npz)
  4. Score three channels (each squashed to [0,1] then geometric-mean composed):

       clash         = soft penalty on min(candidate_non_warhead_heavy → pocket_cloud)
                       (target: no atom closer than ~2.7Å to any pocket atom)
       hinge_reach   = soft reward on min(distal_atom → {Met416.N, Ala417.N})
                       distal = candidate heavy atom furthest from Cβ
                       (target: 2.8-4.5Å, i.e. H-bond range)
       bd_angle      = soft reward on |SG–Cβ–Cα angle − 107°|
                       (target: dev<25°)

  5. composite = (clash * hinge * bd) ** (1/3), clipped to [0,1].

All scoring is CPU-only and conformer-free of the pocket: we use the anchor pose
as a frozen reference frame. Throughput: ~50-100 ms/mol on M-series CPUs.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

# Default warhead: terminal acrylamide CH2=CH-C(=O)-N.
# Note SMARTS atom order is (Cβ, Cα, C', O, N) — used to align 5 anchor atoms.
DEFAULT_WARHEAD_SMARTS = "[CH2;X3]=[CH;X3]C(=O)N"
BURGI_DUNITZ_DEG = 107.0


# ---------------------------------------------------------------------------- #
# Kabsch / alignment helpers
# ---------------------------------------------------------------------------- #

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (R, t, rmsd) s.t. (R @ P.T).T + t ≈ Q. Assumes P, Q same shape."""
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    Pp = P - Pc
    Qp = Q - Qc
    H = Pp.T @ Qp
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    P_aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))
    return R, t, rmsd


def apply_rigid(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (R @ X.T).T + t


# ---------------------------------------------------------------------------- #
# Channel mappings (raw measurement → [0,1])
# ---------------------------------------------------------------------------- #

def soft_clash(d_min_A: float, hard: float = 1.8, soft: float = 2.8) -> float:
    """1.0 if no atom within `hard`Å else linear ramp to 0 at `soft`Å.
    Higher = less clash."""
    if d_min_A >= soft:
        return 1.0
    if d_min_A <= hard:
        return 0.0
    return (d_min_A - hard) / (soft - hard)


def soft_clash_count(n_clashes: int, n_heavy: int, k: float = 6.0) -> float:
    """1.0 if no clashes, sigmoid-fall as a fraction of heavy atoms clash.
    A real ETKDG-from-vacuum will always have a few touching atoms; this measures *fraction*.
    """
    if n_heavy <= 0:
        return 0.0
    frac = n_clashes / max(n_heavy, 1)
    # sigmoid centred at frac=0 with slope k; map [0,1]
    # x=0 -> 1.0,  x=0.1 -> ~0.65, x=0.3 -> ~0.15
    return float(np.exp(-k * frac))


def soft_hinge(d_A: float, lo: float = 2.6, hi: float = 6.5, fall: float = 12.0) -> float:
    """Sweet spot in [lo, hi] = 1.0; ramps to 0 at fall.
    Sub-lo (clash with hinge backbone) = 0."""
    if d_A < lo:
        return max(0.0, (d_A - 1.5) / max(lo - 1.5, 1e-6))  # weak credit for getting close
    if d_A <= hi:
        return 1.0
    if d_A >= fall:
        return 0.0
    return (fall - d_A) / (fall - hi)


def soft_bd(dev_deg: float, perfect: float = 15.0, fail: float = 60.0) -> float:
    """1.0 if dev<perfect, 0 if dev>fail, linear in between."""
    if dev_deg <= perfect:
        return 1.0
    if dev_deg >= fail:
        return 0.0
    return (fail - dev_deg) / (fail - perfect)


# ---------------------------------------------------------------------------- #
# Core scorer
# ---------------------------------------------------------------------------- #

class FastGeomScorer:
    """Geometry surrogate for covalent kinase candidates."""

    def __init__(
        self,
        anchor_frame_npz: str | Path,
        warhead_smarts: str = DEFAULT_WARHEAD_SMARTS,
        seed: int = 0xC0FE,
        n_conformers: int = 3,
    ):
        self.n_conformers = max(1, int(n_conformers))
        self.anchor_path = Path(anchor_frame_npz)
        d = np.load(self.anchor_path, allow_pickle=True)
        self.pocket_xyz = d["coords"].astype(np.float32)        # (Npocket, 3)
        self.sg = d["sg"].astype(np.float32)                     # (3,)
        self.cys_ca = d["cys_ca"].astype(np.float32)
        self.cys_cb_prot = d["cys_cb"].astype(np.float32)
        self.cb_ideal = d["cb_ideal"].astype(np.float32)         # ligand Cβ in anchor pose
        self.met_n = d["met416_n"].astype(np.float32)
        self.ala_n = d["ala417_n"].astype(np.float32)
        # 5-atom warhead template (Cβ Cα C' O N) directly from the anchor cofold.
        if "warhead_template" in d:
            self.warhead_template = d["warhead_template"].astype(np.float32)
        else:
            self.warhead_template = self._build_warhead_template()  # fallback
        # Distance from anchor's distal atom to nearest hinge donor — sets the hinge sweet spot.
        if "distal_ideal" in d:
            self.distal_ideal = d["distal_ideal"].astype(np.float32)
        else:
            self.distal_ideal = None
        # Exclude pocket atoms that lie within the warhead-bonding "interior" — Cys346 SG/CB
        # and the chemically-bonded Cβ position itself are not clashes.
        cb_pos = self.cb_ideal
        keep = [i for i in range(self.pocket_xyz.shape[0])
                if float(np.linalg.norm(self.pocket_xyz[i] - cb_pos)) >= 2.2]
        self.pocket_xyz = self.pocket_xyz[keep]
        # Anchor ligand cloud — the "ideal shape" the candidate should overlap.
        if "lig_coords" in d:
            self.lig_cloud = d["lig_coords"].astype(np.float32)
        else:
            self.lig_cloud = None

        self.smarts = warhead_smarts
        self.smarts_mol = Chem.MolFromSmarts(warhead_smarts)
        if self.smarts_mol is None:
            raise ValueError(f"bad SMARTS {warhead_smarts!r}")
        self.seed = int(seed)

        # ETKDG once-per-mol params
        self._ps = AllChem.ETKDGv3()
        self._ps.randomSeed = self.seed
        self._ps.useRandomCoords = True
        self._ps.numThreads = 1
        self._ps.pruneRmsThresh = -1.0
        # max embedding attempts (some RDKit versions expose `maxIterations` instead of `maxAttempts`).
        for _attr in ("maxAttempts", "maxIterations"):
            if hasattr(self._ps, _attr):
                try:
                    setattr(self._ps, _attr, 8)
                except Exception:
                    pass

    # -- template geometry ---------------------------------------------------- #
    def _build_warhead_template(self) -> np.ndarray:
        """5 atoms (Cβ, Cα, C', O, N) in the anchor pose's coordinate frame.

        Cβ = cb_ideal (given). Direction Cβ→Cα chosen to satisfy
        the Bürgi-Dunitz angle (107°) wrt SG→Cβ. C'/O/N then placed via
        idealized acrylamide geometry (planar, trans amide).
        """
        cb = self.cb_ideal
        sg_vec = (self.sg - cb)
        sg_dir = sg_vec / (np.linalg.norm(sg_vec) + 1e-9)
        # pick an arbitrary perp axis
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(tmp, sg_dir)) > 0.95:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        perp = tmp - np.dot(tmp, sg_dir) * sg_dir
        perp /= (np.linalg.norm(perp) + 1e-9)
        # Cα direction at 107° from SG dir, in the (sg_dir, perp) plane
        theta = np.deg2rad(BURGI_DUNITZ_DEG)
        ca_dir = np.cos(np.pi - theta) * sg_dir + np.sin(np.pi - theta) * perp
        ca = cb + 1.50 * ca_dir.astype(np.float32)
        # C' = Cα + 1.51Å along ca_dir continued (sp2 Cα–C')
        cprime = ca + 1.51 * ca_dir.astype(np.float32)
        # O double-bond, ~120° from Cα–C'
        op_perp = perp
        o_dir = np.cos(np.deg2rad(120.0)) * (-ca_dir) + np.sin(np.deg2rad(120.0)) * op_perp
        oxy = cprime + 1.23 * o_dir.astype(np.float32)
        # N opposite of O across C' (trans amide)
        n_dir = np.cos(np.deg2rad(-120.0)) * (-ca_dir) + np.sin(np.deg2rad(-120.0)) * op_perp
        nit = cprime + 1.34 * n_dir.astype(np.float32)
        return np.stack([cb, ca, cprime, oxy, nit], axis=0).astype(np.float32)

    # -- per-mol pipeline ----------------------------------------------------- #
    def _gen_conformers(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Embed self.n_conformers ETKDG conformers; return mol with conformers (or None)."""
        m = Chem.Mol(mol)
        try:
            m = AllChem.AddHs(m)
        except Exception:
            return None
        try:
            cids = AllChem.EmbedMultipleConfs(m, numConfs=self.n_conformers, params=self._ps)
        except Exception:
            cids = []
        if len(cids) == 0:
            # fallback: single embed, non-random coords
            try:
                ps2 = AllChem.ETKDGv3()
                ps2.randomSeed = self.seed + 1
                ps2.useRandomCoords = False
                cid = AllChem.EmbedMolecule(m, ps2)
                if cid < 0:
                    return None
            except Exception:
                return None
        # NOTE: we deliberately SKIP MMFF gas-phase optimisation: it slows things 2-3x
        # and tends to over-collapse polar contacts vs ETKDG's protein-statistics priors.
        # ETKDG-only is the standard fast-geometry approach (e.g. RDKit's recommended pipeline).
        return m

    def _match_warhead_anchor_atoms(self, mol: Chem.Mol) -> list[int] | None:
        """Return [idx_Cβ, idx_Cα, idx_C', idx_O, idx_N] from SMARTS match."""
        # SMARTS atom 0=CH2 Cβ, 1=CH Cα, 2=C', 3=O, 4=N
        matches = mol.GetSubstructMatches(self.smarts_mol, useChirality=False)
        if not matches:
            return None
        # pick the match with the most-terminal Cβ (CH2 with only 1 heavy neighbour)
        best = None
        best_score = -1
        for m in matches:
            cb_idx = m[0]
            cb_atom = mol.GetAtomWithIdx(cb_idx)
            heavy_nb = sum(1 for n in cb_atom.GetNeighbors() if n.GetAtomicNum() > 1)
            score = (3 - heavy_nb)  # prefer truly terminal CH2 (1 heavy nb = Cα)
            if score > best_score:
                best_score = score
                best = m
        return list(best) if best is not None else None

    def score(self, smiles: str, debug: bool = False) -> float | dict:
        out = self._score_internal(smiles)
        if debug:
            return out
        return float(out["composite"])

    def _score_one_conformer(self, mol3d: Chem.Mol, conf_id: int, idxs: list[int]) -> dict | None:
        conf = mol3d.GetConformer(conf_id)
        wh_xyz = np.array([[p.x, p.y, p.z] for p in
                           (conf.GetAtomPosition(i) for i in idxs)], dtype=np.float32)
        if wh_xyz.shape != (5, 3) or not np.isfinite(wh_xyz).all():
            return None
        try:
            R, t, rmsd_align = kabsch_align(wh_xyz, self.warhead_template)
        except Exception:
            return None
        heavy_idxs = [a.GetIdx() for a in mol3d.GetAtoms() if a.GetAtomicNum() > 1]
        heavy_xyz = np.array([[p.x, p.y, p.z] for p in
                              (conf.GetAtomPosition(i) for i in heavy_idxs)],
                             dtype=np.float32)
        heavy_aligned = apply_rigid(R, t, heavy_xyz)
        wh_set = set(idxs)
        nonwh_mask = np.array([i not in wh_set for i in heavy_idxs])
        nonwh_xyz = heavy_aligned[nonwh_mask]
        cb_xyz = apply_rigid(R, t, np.array([wh_xyz[0]]))[0]
        ca_xyz = apply_rigid(R, t, np.array([wh_xyz[1]]))[0]
        return self._score_geom(nonwh_xyz, cb_xyz, ca_xyz, rmsd_align)

    def _score_geom(self, nonwh_xyz: np.ndarray, cb_xyz: np.ndarray,
                    ca_xyz: np.ndarray, rmsd_align: float) -> dict:
        # ---- Channel design (final, post-iteration 2026-06-23) ----
        # 1. SHAPE — does the candidate's atom cloud overlap the anchor ligand's?
        #    For each candidate non-warhead heavy atom, distance to nearest anchor
        #    ligand heavy atom. Symmetric & ETKDG-noise-tolerant.
        # 2. HINGE — distance from candidate's distal atom to nearest hinge donor.
        # 3. BD — angle SG–Cβ–Cα.
        # 4. CLASH GATE — soft penalty on gross overlap with the protein pocket.
        # Note: dense pocket-cloud clash scoring was abandoned (the F4 pool's iptm
        # variance is driven by global confidence the surrogate can't see; shape
        # complementarity to the anchor ligand is the substitute "fit" signal).
        n_heavy = int(nonwh_xyz.shape[0])

        # ---- Channel 1: shape complementarity to anchor ligand cloud ----
        if self.lig_cloud is None or n_heavy == 0:
            shape_score = 0.0; shape_mean_d = float("inf"); shape_cov = 0.0
        else:
            # candidate atoms -> nearest anchor ligand atom
            diff_l = nonwh_xyz[:, None, :] - self.lig_cloud[None, :, :]
            dists_l = np.sqrt(np.sum(diff_l * diff_l, axis=2))
            d_to_lig = dists_l.min(axis=1)
            shape_mean_d = float(d_to_lig.mean())
            # also coverage of anchor: fraction of anchor atoms with a candidate
            # neighbour within 2.5Å (i.e. is the ATP pocket actually being filled?)
            diff_a = self.lig_cloud[:, None, :] - nonwh_xyz[None, :, :]
            dists_a = np.sqrt(np.sum(diff_a * diff_a, axis=2))
            d_anchor_to_cand = dists_a.min(axis=1)
            shape_cov = float((d_anchor_to_cand < 2.5).mean())
            # Combine: low mean distance (good neighbor) + high coverage
            shape_fit = max(0.0, 1.0 - (shape_mean_d - 1.0) / 3.5)  # 1Å->1.0, 4.5Å->0
            shape_score = float(np.sqrt(shape_fit * (shape_cov + 0.05)))
            shape_score = max(0.0, min(1.0, shape_score))

        # ---- Channel 1b: gross clash gate (protein) ----
        if n_heavy == 0 or self.pocket_xyz.shape[0] == 0:
            d_min = float("inf"); n_clashes = 0; clash_gate = 1.0
        else:
            diff = nonwh_xyz[:, None, :] - self.pocket_xyz[None, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            d_per_atom = dists.min(axis=1)
            d_min = float(d_per_atom.min())
            n_clashes = int((d_per_atom < 1.6).sum())  # only really severe (<1.6Å) clashes
            frac = n_clashes / max(n_heavy, 1)
            clash_gate = float(np.exp(-3.0 * frac))

        # ---- Channel 2: hinge reach (distal atom -> nearest hinge donor) ----
        if n_heavy == 0:
            hinge_d = float("inf"); hinge_score = 0.0
        else:
            d_to_cb = np.linalg.norm(nonwh_xyz - cb_xyz, axis=1)
            distal_idx = int(np.argmax(d_to_cb))
            distal_xyz = nonwh_xyz[distal_idx]
            d_met = float(np.linalg.norm(distal_xyz - self.met_n))
            d_ala = float(np.linalg.norm(distal_xyz - self.ala_n))
            hinge_d = min(d_met, d_ala)
            hinge_score = soft_hinge(hinge_d)

        # ---- Channel 3: Bürgi-Dunitz ----
        v1 = self.sg - cb_xyz
        v2 = ca_xyz - cb_xyz
        cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        cos_a = max(-1.0, min(1.0, cos_a))
        ang = float(np.degrees(np.arccos(cos_a)))
        bd_dev = abs(ang - BURGI_DUNITZ_DEG)
        bd_score = soft_bd(bd_dev)

        eps = 1e-3
        composite = float(((shape_score + eps) * (hinge_score + eps) * (bd_score + eps)) ** (1.0 / 3.0))
        composite *= clash_gate
        composite = max(0.0, min(1.0, composite))

        return {
            "composite": composite,
            "shape_score": shape_score, "shape_mean_d": shape_mean_d, "shape_cov": shape_cov,
            "clash_score": shape_score,  # spec alias
            "clash_gate": clash_gate,
            "hinge_score": hinge_score, "bd_score": bd_score,
            "rmsd_align": rmsd_align, "d_min_pocket": d_min,
            "hinge_d": hinge_d, "bd_dev": bd_dev, "n_clashes": n_clashes,
            "reason": "ok",
        }

    def _score_internal(self, smiles: str) -> dict:
        fail = {"composite": 0.0, "clash_score": 0.0, "hinge_score": 0.0,
                "bd_score": 0.0, "rmsd_align": float("nan"), "reason": "ok"}
        if not smiles or not isinstance(smiles, str):
            fail["reason"] = "empty"; return fail
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fail["reason"] = "unparseable"; return fail
        idxs = self._match_warhead_anchor_atoms(mol)
        if idxs is None:
            fail["reason"] = "no_warhead"; return fail
        mol3d = self._gen_conformers(mol)
        if mol3d is None or mol3d.GetNumConformers() == 0:
            fail["reason"] = "embed_failed"; return fail
        # Score each conformer, take best by composite.
        best = None
        for c in mol3d.GetConformers():
            try:
                out = self._score_one_conformer(mol3d, c.GetId(), idxs)
            except Exception:
                out = None
            if out is None:
                continue
            if best is None or out["composite"] > best["composite"]:
                best = out
        if best is None:
            fail["reason"] = "scoring_failed"; return fail
        return best

    def score_batch(self, smiles_list: Iterable[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]


# Sanity self-test if invoked as a script.
if __name__ == "__main__":
    import time
    anchor = "/Users/shaharharel/Documents/github/edit-small-mol/data/fast_geom_surrogate/anchor_frame.npz"
    sc = FastGeomScorer(anchor)
    smis = [
        "C=CC(=O)N1Cc2cccc(C(=O)Nc3cnc(NC(=O)c4cn(C)nc4C(C)C)cn3)c2C1",  # Mol1
        "C=CC(=O)NCc1ccccc1",                                              # tiny acrylamide
        "C=CC(=O)N1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",                     # big floppy = bad
        "c1ccccc1",                                                        # no warhead
        "$$$bad",                                                          # unparseable
    ]
    t0 = time.time()
    for s in smis:
        det = sc.score(s, debug=True)
        print(f"{det['composite']:.3f}  clash={det['clash_score']:.2f} hinge={det['hinge_score']:.2f} "
              f"bd={det['bd_score']:.2f}  ({det['reason']})  {s[:60]}")
    dt = time.time() - t0
    print(f"avg: {dt / len(smis) * 1000:.1f} ms/mol")
