"""Shared evaluation suite for Lingo3DMol Plans L0-L3 (ZAP70 Cys346 campaign).

Computes 14 metrics + 1 headline aggregate (Covalent Hit Rate) on a generated
cohort. Each cohort can be either a 3D SDF (one mol per record, with a pose) or
a list of SMILES (in which case Cat-2 geometry metrics are reported as N/A).

  Cat 1 — Connectivity & warhead structural:
    1.  connectivity_largest_pct   — largest fragment >= 80% heavy atoms
    2.  acryl_strict_pct           — exact SMARTS [CH2]=[CH]C(=O)N where the
                                       amide N is either NH1/NH2 (terminal) OR
                                       tertiary with both substituents on carbon
                                       (sp3 or aromatic) — covers isoindoline-N
                                       and N,N-dialkyl/aryl acrylamides.
    3.  acryl_largest_pct          — warhead must be on largest fragment
    4.  body_atom_C_pct            — atom on the amide-N (not C=O) must be C

  Cat 2 — Geometry (requires 3D pose):
    5.  d_SG_in_range_pct          — Cbeta-SG distance in [1.7, 3.0] A
    6.  burgi_dunitz_angle_pct     — attack angle within 30 deg of 107
    7.  warhead_orientation_score  — composite (mean of (5) + (6))

  Cat 3 — Chemistry (xTB descriptors, optional):
    8.  pred_log_k2_GSH_in_drug_range_pct
    9.  omega_diversity_kl
    10. body_atom_validity_pct     — body atom is C, not heavily substituted

  Cat 4 — Drug-likeness + diversity:
    11. QED_passing_pct            — QED >= 0.4
    12. MW_in_drug_range_pct       — MW in [320, 480]
    13. scaffold_diversity_pct     — unique Murcko / total
    14. tanimoto_to_train_max_0.5_pct — Tc to Tier-2 train set < 0.5

  Headline:
    15. Covalent_Hit_Rate          — composite gate (see CHR_GATES)

CLI:

    python experiments/eval_lingo3dmol_plans.py \
        --input cohort.sdf --tag L0_baseline \
        [--smiles-list cohort.csv] \
        [--anchor data/lingo3dmol_anchor_zap70_cys346.json] \
        [--train-csv data/retrain_covalent/acrylamide_only.csv] \
        [--xtb-results <path>] \
        [--compare cohortA.sdf cohortB.sdf ...] \
        [--out results/lingo3dmol/<tag>.json]

Library use:

    from experiments.eval_lingo3dmol_plans import eval_cohort
    rec = eval_cohort(sdf_or_smiles, anchor_json, train_csv, xtb_results=None)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

# FilterCatalog is optional — fall back to hand-written SMARTS if unavailable.
try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # type: ignore
    _HAS_FILTER_CATALOG = True
except Exception:  # pragma: no cover
    _HAS_FILTER_CATALOG = False

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANCHOR = PROJECT_ROOT / "data/lingo3dmol_anchor_zap70_cys346.json"
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data/retrain_covalent/acrylamide_only.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/lingo3dmol"

# ---------------------------------------------------------------------------
# SMARTS bank
# ---------------------------------------------------------------------------
# Strict, anchored, no internal alkenes: terminal CH2=CH- vinyl on the carbonyl
# C, amide N with 1-2 H. Specifically excludes internal C=C (R-CH=CH-...).
ACRYL_STRICT_NH = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[NX3;H1,H2]")
# Tertiary N-on-carbon variant: amide N has no H but both substituents are
# carbon (sp3 or aromatic). Captures isoindoline-N-acrylamide and the N,N-
# disubstituted acrylamide motif used by several covalent drugs.
ACRYL_STRICT_TERT_C = Chem.MolFromSmarts(
    "[CH2]=[CH][C](=O)[NX3;H0]([$([CX4]),$([cX3])])[$([CX4]),$([cX3])]"
)
# Legacy alias (kept for any external imports that referenced ACRYL_STRICT).
ACRYL_STRICT = ACRYL_STRICT_NH
# Soft variant for "warhead present at all" diagnostics.
ACRYL_SOFT = Chem.MolFromSmarts("C=CC(=O)N")

# ---------------------------------------------------------------------------
# Eval-suite v2 — medchem-relevant SMARTS bank
# ---------------------------------------------------------------------------
# M1. Privileged kinase hinge-binder motifs. Match ANY -> hinge_clamp = True.
HINGE_CLAMP_SMARTS = {
    "2-aminopyridine":    "[#6]:[#6]:[#6](:[#7])[NH2]",
    "aminopyrimidine":    "[#7]:[#6]:[#6]:[#7]:[#6][NH2]",
    "7-azaindole":        "c1ccc2[nH]ccc2n1",
    "1,3,4-oxadiazole":   "c1nnoc1",
    "Amide-anilide":      "[NH]C(=O)c1ccccc1",
    "Pyrimidin-2-amine":  "Nc1nccnc1",
}
HINGE_CLAMP_PATTERNS = [(name, Chem.MolFromSmarts(s)) for name, s in HINGE_CLAMP_SMARTS.items()]
HINGE_CLAMP_PATTERNS = [(n, p) for n, p in HINGE_CLAMP_PATTERNS if p is not None]

# M2/M6. Isoindoline-acrylamide chassis used to root the arm / measure linker.
ISOINDOLINE_ACRYL_CHASSIS = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")

# M2. H-bond donor pattern: any heavy atom with >=1 H on N/O/S.
H_DONOR = Chem.MolFromSmarts("[$([N!H0]),$([O!H0]),$([S!H0])]")

# M6. Aromatic heteroatom (not aromatic carbon).
AROM_HETERO = Chem.MolFromSmarts("[a;!c]")

# M3. Charged-group SMARTS used as fallback / sanity. Net charge is computed
# from formal charges directly (the cohort SMILES already encode protonation
# states such as `[O-]` for carboxylate and `[N+]` for ammonium).
CHARGED_ANION_SMARTS = (
    "[CX3](=O)[O-]",     # carboxylate
    "[S](=O)(=O)[O-]",   # sulfonate
    "P(=O)([O-])[O-]",   # phosphonate
)
CHARGED_CATION_SMARTS = (
    "[N+;H3,H2,H1,H0]",  # protonated amine
)

# M4. Brenk/PAINS fallback patterns (used only if RDKit FilterCatalog missing).
# Hand-picked subset matching the medchem failure modes called out in the C5
# review (trimethoxy, furan, naphthylamine, sugar, hydrazide, nitro).
FALLBACK_BRENK_SMARTS = {
    "trimethoxy":     "c(OC)(OC)c(OC)",
    "furan":          "c1ccoc1",
    "naphthylamine":  "Nc1ccc2ccccc2c1",
    "sugar_ring":     "OC1OCC(O)C(O)C1O",
    "hydrazide":      "[NX3][NX3]",
    "nitro":          "[N+](=O)[O-]",
    "aldehyde":       "[CX3H1](=O)[#6]",
    "epoxide":        "C1OC1",
    "thiocarbonyl":   "[CX3]=[SX1]",
    "isocyanate":     "N=C=O",
}
FALLBACK_BRENK_PATTERNS = [
    (name, Chem.MolFromSmarts(s)) for name, s in FALLBACK_BRENK_SMARTS.items()
]
FALLBACK_BRENK_PATTERNS = [(n, p) for n, p in FALLBACK_BRENK_PATTERNS if p is not None]

# Acrylamide warhead is a Michael acceptor by design — the Brenk catalog flags
# `Michael_acceptor_1` on every covalent mol in this campaign. We exempt that
# single alert so the pass/fail signal reflects *unintended* alerts only.
_BRENK_EXEMPT_DESCRIPTIONS = {"Michael_acceptor_1"}

# Build the RDKit FilterCatalog once (Brenk + PAINS A/B/C) for M4.
_FILTER_CATALOG = None
if _HAS_FILTER_CATALOG:
    try:
        _fc_params = FilterCatalogParams()
        _fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        _fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        _fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        _FILTER_CATALOG = FilterCatalog(_fc_params)
    except Exception:
        _FILTER_CATALOG = None
        _HAS_FILTER_CATALOG = False

# M5. Veber thresholds.
VEBER_TPSA_MAX = 140.0
VEBER_ROTBONDS_MAX = 10

# M3. Net-charge acceptable window (medchem developability).
NET_CHARGE_ACCEPTABLE = {-1, 0, +1}

# M2. Linker-length cutoff used by chr_v2 composite (bond-count from arm root
# to first H-bond donor in the arm).
LINKER_LENGTH_MAX = 4

# Reference set: known drug-like covalent warhead omega values (heuristic anchor).
# Used as the "drug-like" omega distribution for Cat 3 metric (10). Values are
# illustrative; users should overlay their own xTB-derived omega distribution.
REF_OMEGA_DRUG_LIKE = np.array(
    [
        # Approx omega (electron-accepting power, eV) for known acrylamide drugs.
        2.40,  # afatinib
        2.55,  # ibrutinib
        2.45,  # acalabrutinib
        2.50,  # zanubrutinib
        2.60,  # neratinib
        2.35,  # osimertinib
        2.45,  # rupatadine analog ref
        2.50,  # poziotinib
        2.55,  # dacomitinib
        2.40,  # olmutinib
    ],
    dtype=float,
)

# Drug-range for xTB log_k2_GSH (from MEMORY: k_inact in [0.5, 2.0])
DRUG_LOG_K2_GSH_RANGE = (0.5, 2.0)

# Warhead-geometry tolerances.
D_SG_RANGE_A = (1.7, 3.0)
BURGI_DUNITZ_TARGET_DEG = 107.0
BURGI_DUNITZ_TOL_DEG = 30.0

# Drug-likeness gates.
QED_THRESHOLD = 0.4
MW_RANGE = (320.0, 480.0)
LARGEST_FRAG_HEAVY_FRACTION = 0.8

# CHR (headline) — must pass ALL listed sub-gates.
CHR_GATES = (
    "connectivity_largest",
    "acryl_largest",
    "body_atom_C",
    "d_SG_in_range",     # if pose available; else this gate is skipped
    "body_atom_validity",
    "QED_passing",
    "MW_in_drug_range",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _frags(mol: Chem.Mol) -> list[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def _largest(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    frags = _frags(mol)
    if not frags:
        return None
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        return None
    return largest


def _morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)


def _murcko_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        if scaff is None or scaff.GetNumHeavyAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaff)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mol loading
# ---------------------------------------------------------------------------
def load_input(path: Path) -> tuple[list[Optional[Chem.Mol]], bool]:
    """Return (mols_with_optional_3D, has_3d_flag).

    .sdf  -> SDMolSupplier (3D pose preserved when present)
    .csv  -> reads 'smiles' col; no 3D
    .smi  -> one SMILES per line; no 3D
    """
    path = Path(path)
    mols: list[Optional[Chem.Mol]] = []
    if path.suffix.lower() == ".sdf":
        for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False):
            mols.append(m)
        # Decide has_3d by inspecting first non-None mol's conformer.
        has_3d = False
        for m in mols:
            if m is not None and m.GetNumConformers() > 0:
                conf = m.GetConformer(0)
                if conf.Is3D():
                    has_3d = True
                    break
        return mols, has_3d
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        for s in df[col].astype(str):
            mols.append(Chem.MolFromSmiles(s))
        return mols, False
    elif path.suffix.lower() in {".smi", ".txt"}:
        with path.open() as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    mols.append(Chem.MolFromSmiles(s))
        return mols, False
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Per-mol metric primitives
# ---------------------------------------------------------------------------
def m_connectivity_largest(raw: Chem.Mol, largest: Chem.Mol) -> bool:
    """True if the largest fragment contains >= LARGEST_FRAG_HEAVY_FRACTION of
    the raw molecule's heavy atoms.
    """
    total = raw.GetNumHeavyAtoms() if raw else 0
    if total == 0:
        return False
    return largest.GetNumHeavyAtoms() / total >= LARGEST_FRAG_HEAVY_FRACTION


def _acryl_strict_matches(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """Return SMARTS matches for an "anchored" acrylamide warhead.

    Accepts EITHER the terminal-NH form `[CH2]=[CH]C(=O)[NX3;H1,H2]` OR the
    tertiary-N form `[CH2]=[CH]C(=O)N(R)R'` where both N substituents are
    carbon (sp3 or aromatic). The latter covers isoindoline-N-acrylamide
    style warheads (this codebase's L2 scaffold-anchor cohort) and N,N-
    disubstituted acrylamides used by several covalent drugs.

    Both atom tuples follow the same ordering convention:
        (CH2_beta, CH_alpha, C_carbonyl, O_carbonyl, N).
    Downstream helpers (body atom detection, β-carbon coords) rely on this.
    """
    if mol is None:
        return []
    matches: list[tuple[int, ...]] = []
    if ACRYL_STRICT_NH is not None:
        matches.extend(mol.GetSubstructMatches(ACRYL_STRICT_NH))
    if ACRYL_STRICT_TERT_C is not None:
        # The tertiary SMARTS has the same first five atoms (CH2, CH, C, O, N);
        # the two trailing carbon-substituent atoms (positions 5 and 6) are
        # branches outside the warhead — we drop them so the returned tuple
        # matches the NH variant's shape and the body-atom helper still works.
        for tup in mol.GetSubstructMatches(ACRYL_STRICT_TERT_C):
            matches.append(tup[:5])
    # De-duplicate while preserving order.
    seen: set[tuple[int, ...]] = set()
    uniq: list[tuple[int, ...]] = []
    for tup in matches:
        if tup not in seen:
            seen.add(tup)
            uniq.append(tup)
    return uniq


def m_acryl_strict(raw: Chem.Mol) -> bool:
    """Strict anchored acrylamide present anywhere in the raw mol."""
    return bool(_acryl_strict_matches(raw))


def m_acryl_largest(largest: Chem.Mol) -> bool:
    """Strict acrylamide present on the largest fragment."""
    return bool(_acryl_strict_matches(largest))


def _body_atom_of_warhead(largest: Chem.Mol) -> Optional[Chem.Atom]:
    """The body atom = the heavy-atom neighbor of the amide N that is NOT in
    the warhead match (i.e. the first atom of the body where it attaches to
    the warhead amide N).

    Returns None if no warhead is present.
    """
    matches = _acryl_strict_matches(largest)
    if not matches:
        return None
    match = matches[0]
    warhead_set = set(match)
    # SMARTS atoms (0..N-1): [CH2]=[CH][C](=O)[NX3;H1,H2]
    #   0=CH2, 1=CH, 2=C, 3=O (the =O), 4=N. Find the actual N (atomic #7).
    n_idx = None
    for ai in match:
        if largest.GetAtomWithIdx(ai).GetAtomicNum() == 7:
            n_idx = ai
            break
    if n_idx is None:
        return None
    n_atom = largest.GetAtomWithIdx(n_idx)
    for nb in n_atom.GetNeighbors():
        if nb.GetIdx() not in warhead_set and nb.GetAtomicNum() > 1:
            return nb
    return None


def m_body_atom_C(largest: Chem.Mol) -> Optional[bool]:
    """True if body atom (the heavy neighbor of amide N outside the warhead)
    is carbon. None if no warhead.
    """
    body = _body_atom_of_warhead(largest)
    if body is None:
        return None
    return body.GetAtomicNum() == 6


def m_body_atom_validity(largest: Chem.Mol) -> Optional[bool]:
    """Stricter: body atom is C AND has <= 3 heavy-atom substituents (i.e. not
    a quaternary carbon)."""
    body = _body_atom_of_warhead(largest)
    if body is None:
        return None
    if body.GetAtomicNum() != 6:
        return False
    heavy_nbrs = sum(1 for nb in body.GetNeighbors() if nb.GetAtomicNum() > 1)
    return heavy_nbrs <= 3


# ---------------------------------------------------------------------------
# Geometry primitives (Cat 2 — only valid when input is a 3D SDF with a
# warhead-pose AND the user supplies the Cys346 SG coordinate via --anchor.)
# ---------------------------------------------------------------------------
def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _warhead_cb_pos(largest_3d: Chem.Mol) -> Optional[np.ndarray]:
    """Return the 3D coords of the warhead CH2= carbon (the β-carbon)."""
    matches = _acryl_strict_matches(largest_3d)
    if not matches:
        return None
    cb_idx = matches[0][0]   # first SMARTS atom = [CH2]=
    if largest_3d.GetNumConformers() == 0:
        return None
    conf = largest_3d.GetConformer(0)
    p = conf.GetAtomPosition(cb_idx)
    return np.array([p.x, p.y, p.z])


def m_d_SG_in_range(largest_3d: Chem.Mol, sg_pos: np.ndarray) -> Optional[bool]:
    cb = _warhead_cb_pos(largest_3d)
    if cb is None:
        return None
    d = float(np.linalg.norm(cb - sg_pos))
    return D_SG_RANGE_A[0] <= d <= D_SG_RANGE_A[1]


def m_burgi_dunitz_angle(largest_3d: Chem.Mol, sg_pos: np.ndarray, ca_pos: np.ndarray) -> Optional[bool]:
    cb = _warhead_cb_pos(largest_3d)
    if cb is None:
        return None
    # Angle between (SG->CB_warhead) and (SG->CA_cys) — should be ~107°.
    # NOTE: legacy metric. Measured AT SG. Will pass by construction whenever
    # CB sits at 1.85 Å along the (SG->CB) attack vector defined in the anchor
    # JSON, regardless of where atom1 (vinyl Cα) is — so this metric did NOT
    # catch the BD-collinear bug in the pre-fix Lingo3DMol pipeline.
    v_attack = cb - sg_pos
    v_backbone = ca_pos - sg_pos
    ang = _angle_deg(v_attack, v_backbone)
    return abs(ang - BURGI_DUNITZ_TARGET_DEG) <= BURGI_DUNITZ_TOL_DEG


def m_burgi_dunitz_angle_at_atom0(largest_3d: Chem.Mol,
                                  sg_pos: np.ndarray) -> Optional[bool]:
    """Bürgi-Dunitz angle measured AT atom0 (the electrophilic Cβ carbon).

    This is the chemistry-standard convention and is the metric that exposed
    the collinear rotation bug fixed in ``experiments/anchor_geometry.py``.
    The angle is between (atom0 -> SG) and (atom0 -> atom1), where atom1 is
    the vinyl Cα immediately bonded to atom0 in the warhead. Returns True iff
    the measured angle is within BURGI_DUNITZ_TOL_DEG of BURGI_DUNITZ_TARGET_DEG.
    """
    if largest_3d.GetNumConformers() == 0:
        return None
    conf = largest_3d.GetConformer(0)
    if largest_3d.GetNumAtoms() < 2:
        return None
    a0 = np.array(conf.GetAtomPosition(0))
    a1 = np.array(conf.GetAtomPosition(1))
    v_sg = sg_pos - a0
    v_a1 = a1 - a0
    ang = _angle_deg(v_sg, v_a1)
    return abs(ang - BURGI_DUNITZ_TARGET_DEG) <= BURGI_DUNITZ_TOL_DEG


# ---------------------------------------------------------------------------
# Drug-likeness primitives
# ---------------------------------------------------------------------------
def m_QED(largest: Chem.Mol) -> tuple[bool, float]:
    q = float(QED.qed(largest))
    return q >= QED_THRESHOLD, q


def m_MW(largest: Chem.Mol) -> tuple[bool, float]:
    mw = float(Descriptors.MolWt(largest))
    return MW_RANGE[0] <= mw <= MW_RANGE[1], mw


# ---------------------------------------------------------------------------
# Eval-suite v2 — six medchem-relevant metrics + chr_v2 composite.
# ---------------------------------------------------------------------------
def m_hinge_clamp(mol: Chem.Mol) -> Optional[bool]:
    """M1. True if mol matches >=1 privileged kinase hinge-binder SMARTS."""
    if mol is None:
        return None
    for _, pat in HINGE_CLAMP_PATTERNS:
        if mol.HasSubstructMatch(pat):
            return True
    return False


def _chassis_anchor_idx(mol: Chem.Mol) -> Optional[int]:
    """Find the chassis atom (the "C5 anchor") whose substituent leaves the
    chassis into the recognition arm. Aromatic chassis atoms (benzene C5/C6)
    are preferred (C5 cohort); aliphatic isoindoline C1 is the fall-back
    (C1 cohort). Warhead atoms (SMARTS positions 0..4) are excluded.
    Returns None if no chassis or no arm.
    """
    if mol is None or ISOINDOLINE_ACRYL_CHASSIS is None:
        return None
    match = mol.GetSubstructMatch(ISOINDOLINE_ACRYL_CHASSIS)
    if not match:
        return None
    chassis_set = set(match)
    warhead_set = set(match[:5])  # C=C-C(=O)-N of the acrylamide
    aromatic_anchor: Optional[int] = None
    aliphatic_anchor: Optional[int] = None
    for ai in match:
        if ai in warhead_set:
            continue
        a = mol.GetAtomWithIdx(ai)
        has_arm = any(
            nb.GetIdx() not in chassis_set and nb.GetAtomicNum() > 1
            for nb in a.GetNeighbors()
        )
        if not has_arm:
            continue
        if a.GetIsAromatic() and aromatic_anchor is None:
            aromatic_anchor = ai
        elif (not a.GetIsAromatic()) and aliphatic_anchor is None:
            aliphatic_anchor = ai
    return aromatic_anchor if aromatic_anchor is not None else aliphatic_anchor


def m_linker_length(mol: Chem.Mol) -> Optional[int]:
    """M2. BFS bond-distance from the chassis anchor (the C5 atom of the
    isoindoline-benzene, or the C1 sp3 atom in the C1 cohort) to the FIRST
    H-bond donor encountered in the arm (chassis atoms are *not* traversed
    after the first step).

    Counting convention: the bond from the anchor into the first arm atom
    counts as 1. So a direct anilide `aryl-N(H)C(=O)-` measures 1, and an
    ethylene-amide `aryl-CH2-CH2-N(H)C(=O)-` measures 3.

    Returns None if (a) the chassis SMARTS does not match, (b) no arm
    exists, or (c) the arm contains no donor atoms. The chr_v2 gate treats
    None as a fail.
    """
    anchor = _chassis_anchor_idx(mol)
    if anchor is None:
        return None
    if H_DONOR is None:
        return None
    donor_set = {hit[0] for hit in mol.GetSubstructMatches(H_DONOR)}
    if not donor_set:
        return None
    match = mol.GetSubstructMatch(ISOINDOLINE_ACRYL_CHASSIS)
    chassis_set = set(match)
    visited = {anchor}
    q: deque[tuple[int, int]] = deque([(anchor, 0)])
    while q:
        cur, d = q.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            # Allow leaving the chassis exactly once (via the anchor->arm bond);
            # once we are in the arm, do not re-enter the chassis.
            if cur != anchor and ni in chassis_set:
                continue
            if ni in visited:
                continue
            if cur == anchor and ni in chassis_set:
                # Don't traverse along the chassis from the anchor itself.
                continue
            if ni in donor_set:
                return d + 1
            visited.add(ni)
            q.append((ni, d + 1))
    return None


def m_net_charge(mol: Chem.Mol) -> Optional[int]:
    """M3. Net formal charge of the molecule as written.

    Cohort SMILES already encode the ionised state (`[O-]` for carboxylate,
    `[N+]` for ammonium, etc.), so we simply sum atomic formal charges. The
    aggregate metric reports the % cohort with net charge in {-1, 0, +1}.
    """
    if mol is None:
        return None
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def m_brenk_pains(mol: Chem.Mol) -> Optional[bool]:
    """M4. True if mol has NO Brenk/PAINS alerts (after exempting the
    Michael-acceptor alert that the covalent warhead is guaranteed to trip).

    Uses RDKit FilterCatalog when available; otherwise falls back to a small
    hand-written SMARTS panel covering the medchem-flagged failure modes
    (trimethoxy, furan, naphthylamine, sugar ring, hydrazide, nitro, ...).
    """
    if mol is None:
        return None
    if _FILTER_CATALOG is not None:
        matches = list(_FILTER_CATALOG.GetMatches(mol))
        unintended = [
            m for m in matches if m.GetDescription() not in _BRENK_EXEMPT_DESCRIPTIONS
        ]
        return len(unintended) == 0
    # Fallback: hand-written panel.
    for _, pat in FALLBACK_BRENK_PATTERNS:
        if mol.HasSubstructMatch(pat):
            return False
    return True


def m_veber(mol: Chem.Mol) -> Optional[bool]:
    """M5. Veber: TPSA <= 140 AND rotatable bonds <= 10."""
    if mol is None:
        return None
    tpsa = Descriptors.TPSA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return (tpsa <= VEBER_TPSA_MAX) and (rot <= VEBER_ROTBONDS_MAX)


def m_arm_heteroaryl(mol: Chem.Mol) -> Optional[bool]:
    """M6. True iff the recognition arm (everything outside the chassis match)
    contains >=1 aromatic heteroatom. Returns None when chassis does not match.
    """
    if mol is None or ISOINDOLINE_ACRYL_CHASSIS is None:
        return None
    match = mol.GetSubstructMatch(ISOINDOLINE_ACRYL_CHASSIS)
    if not match:
        return None
    chassis_set = set(match)
    if AROM_HETERO is None:
        return None
    for hit in mol.GetSubstructMatches(AROM_HETERO):
        if hit[0] not in chassis_set:
            return True
    return False


# ---------------------------------------------------------------------------
# Cat 3 — xTB descriptors (post-hoc).
# ---------------------------------------------------------------------------
def _kl_divergence_gaussian(p: np.ndarray, q: np.ndarray) -> float:
    """Closed-form symmetric KL between two empirical 1D distributions, modeled
    as Gaussians (matched moments). Returns the symmetric form (KL(p||q)+KL(q||p))/2.
    """
    if len(p) < 2 or len(q) < 2:
        return float("nan")
    mp, sp = float(np.mean(p)), float(np.std(p) + 1e-6)
    mq, sq = float(np.mean(q)), float(np.std(q) + 1e-6)
    kl_pq = math.log(sq / sp) + (sp * sp + (mp - mq) ** 2) / (2 * sq * sq) - 0.5
    kl_qp = math.log(sp / sq) + (sq * sq + (mq - mp) ** 2) / (2 * sp * sp) - 0.5
    return 0.5 * (kl_pq + kl_qp)


def _xtb_load(xtb_results_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Best-effort load of xTB descriptor table. Expected columns:
       smiles, log_k2_GSH, omega
    """
    if xtb_results_path is None:
        return None
    p = Path(xtb_results_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


# ---------------------------------------------------------------------------
# Build per-mol records
# ---------------------------------------------------------------------------
@dataclass
class AnchorSpec:
    sg_pos: np.ndarray
    ca_pos: np.ndarray


def _load_anchor(path: Path) -> AnchorSpec:
    with Path(path).open() as f:
        d = json.load(f)
    return AnchorSpec(
        sg_pos=np.array(d["sg_pos"], dtype=float),
        ca_pos=np.array(d["ca_pos"], dtype=float),
    )


def build_records(
    mols_raw: list[Optional[Chem.Mol]],
    has_3d: bool,
    anchor: Optional[AnchorSpec],
    xtb_df: Optional[pd.DataFrame],
) -> list[dict]:
    xtb_lookup: dict[str, dict] = {}
    if xtb_df is not None:
        for _, row in xtb_df.iterrows():
            try:
                m = Chem.MolFromSmiles(str(row["smiles"]))
                if m is not None:
                    xtb_lookup[Chem.MolToSmiles(m)] = row.to_dict()
            except Exception:
                continue

    records: list[dict] = []
    for raw in mols_raw:
        raw_san = _sanitize(raw)
        largest = _largest(raw_san)
        rec: dict = {"valid": largest is not None}
        if not rec["valid"]:
            records.append(rec)
            continue

        rec["smiles"] = Chem.MolToSmiles(largest)
        rec["n_heavy"] = int(largest.GetNumHeavyAtoms())
        rec["n_frags"] = len(_frags(raw_san)) if raw_san else 1

        # Cat 1
        rec["connectivity_largest"] = m_connectivity_largest(raw_san, largest)
        rec["acryl_strict"] = m_acryl_strict(raw_san)
        rec["acryl_largest"] = m_acryl_largest(largest)
        body_c = m_body_atom_C(largest)
        rec["body_atom_C"] = body_c if body_c is not None else False
        body_v = m_body_atom_validity(largest)
        rec["body_atom_validity"] = body_v if body_v is not None else False

        # Cat 2 — geometry
        if has_3d and anchor is not None and largest.GetNumConformers() > 0:
            d_ok = m_d_SG_in_range(largest, anchor.sg_pos)
            ang_ok = m_burgi_dunitz_angle(largest, anchor.sg_pos, anchor.ca_pos)
            ang_a0_ok = m_burgi_dunitz_angle_at_atom0(largest, anchor.sg_pos)
            rec["d_SG_in_range"] = d_ok
            rec["burgi_dunitz_angle"] = ang_ok
            rec["burgi_dunitz_angle_at_atom0"] = ang_a0_ok
        else:
            rec["d_SG_in_range"] = None
            rec["burgi_dunitz_angle"] = None
            rec["burgi_dunitz_angle_at_atom0"] = None

        # Cat 3 — xTB descriptors (post-hoc; require xtb_df).
        xtb_row = xtb_lookup.get(rec["smiles"])
        if xtb_row is not None and "log_k2_GSH" in xtb_row:
            lk = float(xtb_row["log_k2_GSH"])
            rec["log_k2_GSH"] = lk
            rec["log_k2_GSH_in_drug_range"] = (
                DRUG_LOG_K2_GSH_RANGE[0] <= lk <= DRUG_LOG_K2_GSH_RANGE[1]
            )
            if "omega" in xtb_row:
                rec["omega"] = float(xtb_row["omega"])
        else:
            rec["log_k2_GSH"] = None
            rec["log_k2_GSH_in_drug_range"] = None

        # Cat 4
        qed_pass, qed_v = m_QED(largest)
        rec["QED"] = qed_v
        rec["QED_passing"] = qed_pass
        mw_pass, mw_v = m_MW(largest)
        rec["MW"] = mw_v
        rec["MW_in_drug_range"] = mw_pass
        rec["murcko_smiles"] = _murcko_smiles(largest)

        # Eval-suite v2 — medchem metrics (computed on the largest fragment).
        rec["hinge_clamp"] = m_hinge_clamp(largest)
        rec["linker_length_to_donor"] = m_linker_length(largest)
        rec["net_charge"] = m_net_charge(largest)
        nc = rec["net_charge"]
        rec["net_charge_acceptable"] = (
            nc in NET_CHARGE_ACCEPTABLE if nc is not None else None
        )
        rec["brenk_pains_pass"] = m_brenk_pains(largest)
        rec["veber_pass"] = m_veber(largest)
        rec["arm_heteroaryl"] = m_arm_heteroaryl(largest)

        # chr_v2 — strict composite. AND of: existing CHR gate set + hinge
        # clamp + arm heteroaryl + net charge in {-1,0,+1} + Brenk/PAINS pass
        # + Veber pass + linker_length <= LINKER_LENGTH_MAX.
        # We only set chr_v2 here per-mol; the aggregate uses the same logic.
        chr_v2_ok = True
        # Reuse the existing CHR gates (skip d_SG when 3D unavailable).
        for gate in CHR_GATES:
            if gate == "d_SG_in_range" and not has_3d:
                continue
            v = rec.get(gate)
            if v is None or v is False:
                chr_v2_ok = False
                break
        # New gates:
        v2_gates = (
            ("hinge_clamp", rec["hinge_clamp"]),
            ("arm_heteroaryl", rec["arm_heteroaryl"]),
            ("net_charge_acceptable", rec["net_charge_acceptable"]),
            ("brenk_pains_pass", rec["brenk_pains_pass"]),
            ("veber_pass", rec["veber_pass"]),
        )
        for _, v in v2_gates:
            if v is None or v is False:
                chr_v2_ok = False
                break
        ll = rec["linker_length_to_donor"]
        if ll is None or ll > LINKER_LENGTH_MAX:
            chr_v2_ok = False
        rec["chr_v2"] = chr_v2_ok

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Cohort-level aggregation
# ---------------------------------------------------------------------------
def _frac(values: Iterable, true_value=True) -> float:
    arr = [v for v in values if v is not None]
    if not arr:
        return float("nan")
    return float(sum(1 for v in arr if v == true_value) / len(arr))


def aggregate(records: list[dict], has_3d: bool, train_fps: list, xtb_omegas: Optional[np.ndarray]) -> dict:
    valid = [r for r in records if r.get("valid")]
    n = len(records)
    n_valid = len(valid)

    out: dict = {
        "n_input": n,
        "n_valid": n_valid,
        "validity_pct": float(n_valid / n) if n else 0.0,
    }

    # Cat 1
    out["connectivity_largest_pct"] = _frac([r["connectivity_largest"] for r in valid])
    out["acryl_strict_pct"] = _frac([r["acryl_strict"] for r in valid])
    out["acryl_largest_pct"] = _frac([r["acryl_largest"] for r in valid])
    out["body_atom_C_pct"] = _frac([r["body_atom_C"] for r in valid])

    # Cat 2
    if has_3d:
        out["d_SG_in_range_pct"] = _frac([r["d_SG_in_range"] for r in valid])
        out["burgi_dunitz_angle_pct"] = _frac([r["burgi_dunitz_angle"] for r in valid])
        out["burgi_dunitz_angle_at_atom0_pct"] = _frac(
            [r.get("burgi_dunitz_angle_at_atom0") for r in valid]
        )
        d_ok = out["d_SG_in_range_pct"]
        a_ok = out["burgi_dunitz_angle_pct"]
        if math.isnan(d_ok) or math.isnan(a_ok):
            out["warhead_orientation_score"] = float("nan")
        else:
            out["warhead_orientation_score"] = 0.5 * (d_ok + a_ok)
    else:
        out["d_SG_in_range_pct"] = "N/A (no 3D pose)"
        out["burgi_dunitz_angle_pct"] = "N/A (no 3D pose)"
        out["burgi_dunitz_angle_at_atom0_pct"] = "N/A (no 3D pose)"
        out["warhead_orientation_score"] = "N/A (no 3D pose)"

    # Cat 3
    lk_records = [r["log_k2_GSH_in_drug_range"] for r in valid if r["log_k2_GSH_in_drug_range"] is not None]
    if lk_records:
        out["pred_log_k2_GSH_in_drug_range_pct"] = _frac(lk_records)
    else:
        out["pred_log_k2_GSH_in_drug_range_pct"] = "N/A (xTB not provided — TODO: run experiments/xtb_warhead_electrophilicity.py)"

    if xtb_omegas is not None and len(xtb_omegas) >= 2:
        out["omega_diversity_kl"] = _kl_divergence_gaussian(xtb_omegas, REF_OMEGA_DRUG_LIKE)
    else:
        out["omega_diversity_kl"] = "N/A (xTB omega not provided — TODO: run experiments/xtb_warhead_electrophilicity.py)"

    out["body_atom_validity_pct"] = _frac([r["body_atom_validity"] for r in valid])

    # Cat 4
    out["QED_passing_pct"] = _frac([r["QED_passing"] for r in valid])
    out["MW_in_drug_range_pct"] = _frac([r["MW_in_drug_range"] for r in valid])

    scaff_set: set[str] = set()
    for r in valid:
        if r.get("murcko_smiles"):
            scaff_set.add(r["murcko_smiles"])
    out["scaffold_diversity_pct"] = float(len(scaff_set) / n_valid) if n_valid else 0.0

    # Tanimoto novelty vs train.
    if train_fps:
        novel = 0
        considered = 0
        for r in valid:
            m = Chem.MolFromSmiles(r["smiles"]) if r.get("smiles") else None
            if m is None:
                continue
            fp = _morgan_fp(m)
            max_tc = max((TanimotoSimilarity(fp, t) for t in train_fps), default=0.0)
            considered += 1
            if max_tc < 0.5:
                novel += 1
        out["tanimoto_to_train_max_0.5_pct"] = float(novel / considered) if considered else float("nan")
    else:
        out["tanimoto_to_train_max_0.5_pct"] = "N/A (no train FPs)"

    # Headline: Covalent Hit Rate.
    chr_passes = 0
    chr_total = 0
    for r in valid:
        chr_total += 1
        ok = True
        for gate in CHR_GATES:
            if gate in ("d_SG_in_range",):
                if not has_3d:
                    continue  # skip the gate if 3D not available
            v = r.get(gate)
            if v is None or v is False:
                ok = False
                break
        if ok:
            chr_passes += 1
    out["Covalent_Hit_Rate"] = float(chr_passes / chr_total) if chr_total else float("nan")
    out["chr_gates_used"] = [g for g in CHR_GATES if (has_3d or g != "d_SG_in_range")]

    # ---------------------------------------------------------------
    # Eval-suite v2 — medchem-relevant aggregates.
    # ---------------------------------------------------------------
    out["hinge_clamp_pct"] = _frac([r["hinge_clamp"] for r in valid])

    ll_vals = [
        r["linker_length_to_donor"]
        for r in valid
        if r.get("linker_length_to_donor") is not None
    ]
    if ll_vals:
        ll_arr = np.array(ll_vals, dtype=float)
        out["linker_length_to_donor_n"] = int(ll_arr.size)
        out["linker_length_to_donor_median"] = float(np.median(ll_arr))
        out["linker_length_to_donor_p25"] = float(np.percentile(ll_arr, 25))
        out["linker_length_to_donor_p75"] = float(np.percentile(ll_arr, 75))
        out["linker_length_to_donor_le_max_pct"] = float(
            np.mean(ll_arr <= LINKER_LENGTH_MAX)
        )
    else:
        out["linker_length_to_donor_n"] = 0
        out["linker_length_to_donor_median"] = "N/A (no chassis match)"
        out["linker_length_to_donor_p25"] = "N/A (no chassis match)"
        out["linker_length_to_donor_p75"] = "N/A (no chassis match)"
        out["linker_length_to_donor_le_max_pct"] = "N/A (no chassis match)"

    out["net_charge_acceptable_pct"] = _frac(
        [r["net_charge_acceptable"] for r in valid]
    )
    # Developability fail = net charge != 0 (anion or cation at pH 7.4 proxy).
    nz = [
        (r["net_charge"] != 0) if r.get("net_charge") is not None else None
        for r in valid
    ]
    out["net_charge_nonzero_pct"] = _frac(nz)

    out["brenk_pains_pass_pct"] = _frac([r["brenk_pains_pass"] for r in valid])
    out["brenk_pains_backend"] = (
        "rdkit_FilterCatalog" if _FILTER_CATALOG is not None else "fallback_smarts_panel"
    )

    out["veber_pass_pct"] = _frac([r["veber_pass"] for r in valid])

    # arm_heteroaryl_pct is computed only over mols where chassis matches.
    out["arm_heteroaryl_pct"] = _frac([r["arm_heteroaryl"] for r in valid])

    # chr_v2 — stricter composite over all valid mols.
    out["Covalent_Hit_Rate_v2"] = _frac([r.get("chr_v2") for r in valid])

    return out


# ---------------------------------------------------------------------------
# Train-set fingerprint helper
# ---------------------------------------------------------------------------
def load_train_fps(train_csv: Optional[Path]) -> list:
    if train_csv is None:
        return []
    p = Path(train_csv)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    fps = []
    for s in df[col].dropna().astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(_morgan_fp(m))
    return fps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def eval_cohort(
    input_path: Union[str, Path],
    anchor_json: Union[str, Path] = DEFAULT_ANCHOR,
    train_csv: Union[str, Path, None] = DEFAULT_TRAIN_CSV,
    xtb_results: Union[str, Path, None] = None,
) -> dict:
    """Evaluate a single cohort. Returns a dict with summary + per_mol breakdown."""
    mols, has_3d = load_input(Path(input_path))
    anchor = _load_anchor(Path(anchor_json)) if Path(anchor_json).exists() else None
    train_fps = load_train_fps(Path(train_csv)) if train_csv else []
    xtb_df = _xtb_load(Path(xtb_results)) if xtb_results else None

    records = build_records(mols, has_3d, anchor, xtb_df)
    xtb_omegas = None
    if xtb_df is not None and "omega" in xtb_df.columns:
        xtb_omegas = xtb_df["omega"].dropna().to_numpy(dtype=float)

    summary = aggregate(records, has_3d, train_fps, xtb_omegas)
    summary["has_3d_pose"] = has_3d
    summary["input"] = str(input_path)
    summary["anchor_json"] = str(anchor_json)
    summary["train_csv"] = str(train_csv) if train_csv else None
    summary["xtb_results"] = str(xtb_results) if xtb_results else None

    return {"summary": summary, "per_mol": records}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{100*v:.1f}%"
    return str(v)


def _print_markdown(name: str, summary: dict) -> None:
    print(f"\n## Eval — {name}\n")
    rows = [
        ("n_input", summary["n_input"]),
        ("n_valid", summary["n_valid"]),
        ("validity_pct", _fmt_pct(summary["validity_pct"])),
        ("connectivity_largest_pct", _fmt_pct(summary["connectivity_largest_pct"])),
        ("acryl_strict_pct", _fmt_pct(summary["acryl_strict_pct"])),
        ("acryl_largest_pct", _fmt_pct(summary["acryl_largest_pct"])),
        ("body_atom_C_pct", _fmt_pct(summary["body_atom_C_pct"])),
        ("d_SG_in_range_pct", _fmt_pct(summary["d_SG_in_range_pct"])),
        ("burgi_dunitz_angle_pct", _fmt_pct(summary["burgi_dunitz_angle_pct"])),
        ("burgi_dunitz_angle_at_atom0_pct",
         _fmt_pct(summary.get("burgi_dunitz_angle_at_atom0_pct", "N/A"))),
        ("warhead_orientation_score", _fmt_pct(summary["warhead_orientation_score"])
            if isinstance(summary["warhead_orientation_score"], float) else summary["warhead_orientation_score"]),
        ("pred_log_k2_GSH_in_drug_range_pct", _fmt_pct(summary["pred_log_k2_GSH_in_drug_range_pct"])),
        ("omega_diversity_kl", summary["omega_diversity_kl"]),
        ("body_atom_validity_pct", _fmt_pct(summary["body_atom_validity_pct"])),
        ("QED_passing_pct", _fmt_pct(summary["QED_passing_pct"])),
        ("MW_in_drug_range_pct", _fmt_pct(summary["MW_in_drug_range_pct"])),
        ("scaffold_diversity_pct", _fmt_pct(summary["scaffold_diversity_pct"])),
        ("tanimoto_to_train_max_0.5_pct", _fmt_pct(summary["tanimoto_to_train_max_0.5_pct"])),
        ("Covalent_Hit_Rate", _fmt_pct(summary["Covalent_Hit_Rate"])),
        # --- eval-suite v2 ---
        ("hinge_clamp_pct", _fmt_pct(summary.get("hinge_clamp_pct"))),
        (
            "linker_length_to_donor_median",
            summary.get("linker_length_to_donor_median"),
        ),
        (
            "linker_length_to_donor_p25/p75",
            f"{summary.get('linker_length_to_donor_p25')}/"
            f"{summary.get('linker_length_to_donor_p75')}",
        ),
        (
            "linker_length_to_donor_le4_pct",
            _fmt_pct(summary.get("linker_length_to_donor_le_max_pct")),
        ),
        (
            "net_charge_acceptable_pct",
            _fmt_pct(summary.get("net_charge_acceptable_pct")),
        ),
        (
            "net_charge_nonzero_pct",
            _fmt_pct(summary.get("net_charge_nonzero_pct")),
        ),
        ("brenk_pains_pass_pct", _fmt_pct(summary.get("brenk_pains_pass_pct"))),
        ("brenk_pains_backend", summary.get("brenk_pains_backend")),
        ("veber_pass_pct", _fmt_pct(summary.get("veber_pass_pct"))),
        ("arm_heteroaryl_pct", _fmt_pct(summary.get("arm_heteroaryl_pct"))),
        ("Covalent_Hit_Rate_v2", _fmt_pct(summary.get("Covalent_Hit_Rate_v2"))),
    ]
    print("| metric | value |")
    print("|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", "-i", type=Path, required=False, help="Primary cohort SDF/CSV/SMI.")
    ap.add_argument("--tag", type=str, default="cohort", help="Tag used in output filename.")
    ap.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--xtb-results", type=Path, default=None,
                    help="Optional CSV of xTB descriptors (smiles, log_k2_GSH, omega).")
    ap.add_argument("--compare", type=Path, nargs="*", default=None,
                    help="Optional list of additional cohorts to compare against --input.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.input and not args.compare:
        ap.error("Provide at least --input or --compare.")

    inputs: list[tuple[str, Path]] = []
    if args.input:
        inputs.append((args.tag, args.input))
    if args.compare:
        for i, p in enumerate(args.compare):
            inputs.append((p.stem, p))

    all_results: dict[str, dict] = {}
    for name, path in inputs:
        result = eval_cohort(
            path,
            anchor_json=args.anchor,
            train_csv=args.train_csv,
            xtb_results=args.xtb_results,
        )
        all_results[name] = result
        _print_markdown(name, result["summary"])

    out_path = args.out
    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / f"{args.tag}_eval.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
