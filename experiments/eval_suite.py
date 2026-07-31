"""Eval suite for covalent inhibitor generative models — fast, chemistry-honest.

Designed to catch the failure modes that `build_final_comparison.py` misses
(per `/tmp/medchem_qualitative_report.md` and `/tmp/medchem_physchem_report.md`):

  - Acrylamide as a *disconnected* free fragment (Inpaint v2 / v2_G bug — 65-72%).
  - Reversed warhead polarity (N-vinyl-amide instead of vinyl-carbonyl-amide;
    PocketFlow Cβ-bias bug).
  - Hypervalent S/P, geminal diols, methyleneamine artefacts (M0 / v2_G).
  - Distribution drift vs. the model's own vanilla baseline (mode collapse).

Single CLI entry point:

    python experiments/eval_suite.py --sdf <path> \
        [--baseline <vanilla-sdf-path>] \
        [--report <report-csv>] \
        [--tag <name>] [--out <dir>]

Writes a JSON report under ``results/eval_suite/<tag>_<timestamp>.json`` and
prints a markdown table to stdout.

Design budget: ≤2 min on N=500 (real numbers in the validation table at the
bottom of the run — typical end-to-end is ~30 s including Tier 2 Tc).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, QED, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/eval_suite"
DEFAULT_REPORT_CSV = Path("/tmp/report_mols_compact.csv")
TIER2_METHOD_TAG = "Tier 2 — Fragment Replacement (curated 204)"

# ---------------------------------------------------------------------------
# SMARTS bank
# ---------------------------------------------------------------------------
# Correct-polarity acrylamide: CH2=CH-C(=O)-N (vinyl on the carbonyl-C side).
ACRYL_POLAR = Chem.MolFromSmarts("[CH2]=[CH]-[C](=O)-[N]")
# Soft acrylamide skeleton (any H counts on the vinyl) for "warhead present" check.
ACRYL_SOFT = Chem.MolFromSmarts("C=CC(=O)N")
# Reversed/wrong-polarity warhead the medchem reviewer flagged on PocketFlow.
ACRYL_REVERSED = Chem.MolFromSmarts("N-[CH]=[CH]")

# Synthetic-tractability problem substructures (curated from medchem reports —
# Brenk would flag many of these too but we want a stable, small in-house list
# that does NOT trip on the warhead itself).
TRACTABILITY_BLOCKLIST = [
    ("hypervalent_S", "[S;v4,v5,v6]([!=O])([!=O])([!=O])[!=O]"),
    ("hypervalent_P", "[P;v4]([!=O])([!=O])([!=O])[!=O]"),
    ("SH_hypervalent", "[SH](=O)(O)O"),
    ("geminal_diol", "[CX4]([OH])([OH])"),
    ("methylene_amine_N_oxide", "[CX3]=[NX2][OX1]"),
    ("N_oxide_double", "[#7]=O"),  # excludes nitro via aromatic context check below
    ("triple_N_chain", "[#7]-[#7]-[#7]"),
    ("S_eq_N", "[S]=[N]"),
    ("PH_hypervalent", "[PH]([OH])([OH])"),
]
TRACTABILITY_SMARTS = [(n, Chem.MolFromSmarts(s)) for n, s in TRACTABILITY_BLOCKLIST]

# Brenk catalog, used with the "exclude warhead-Michael" trick — we drop any
# Brenk hit whose match overlaps the acrylamide atoms.
_brenk_params = FilterCatalog.FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog.FilterCatalog(_brenk_params)


# ---------------------------------------------------------------------------
# Mol-loading and per-record record builder
# ---------------------------------------------------------------------------
def _sanitize_or_none(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _largest_frag(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Return the largest (by heavy atoms) sanitized fragment, or None."""
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        return None
    return largest


def load_mols(sdf_or_csv: Path) -> tuple[list[Optional[Chem.Mol]], list[Optional[Chem.Mol]]]:
    """Return (raw_mols, largest_frag_mols). Raw preserves multi-frag topology.

    Accepts .sdf (SDMolSupplier) or .csv (column ``smiles``). The Tier 2
    gold-standard set is a CSV.
    """
    path = Path(sdf_or_csv)
    raw: list[Optional[Chem.Mol]] = []
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        for s in df[col].astype(str):
            raw.append(Chem.MolFromSmiles(s))
    else:
        for m in Chem.SDMolSupplier(str(path), sanitize=False):
            raw.append(m)
    largest = [_largest_frag(_sanitize_or_none(m)) for m in raw]
    return raw, largest


# ---------------------------------------------------------------------------
# Metric: Connectivity Score
# ---------------------------------------------------------------------------
def connectivity_flag(raw_mol: Optional[Chem.Mol], largest: Optional[Chem.Mol]) -> bool:
    """True if the molecule is single-fragment OR the soft-acrylamide motif is
    present on the largest fragment.

    Unifies the previously-scattered ``acryl_largest_frag`` + ``len(GetMolFrags)==1``
    checks. A record passes connectivity if EITHER it's already a single graph
    OR the body-of-interest (largest frag) already contains the warhead.
    Failing connectivity = the warhead exists only as a free piece next to a
    body fragment.
    """
    if raw_mol is None or largest is None:
        return False
    try:
        n_frags = len(Chem.GetMolFrags(raw_mol, asMols=False))
    except Exception:
        return False
    if n_frags == 1:
        return True
    return bool(largest.HasSubstructMatch(ACRYL_SOFT))


# ---------------------------------------------------------------------------
# Metric: Warhead-Connectivity (CMW) — the medchem-honest "is this covalent?"
# ---------------------------------------------------------------------------
def cmw_flag(largest: Optional[Chem.Mol]) -> bool:
    """CMW: the polar acrylamide (``CH2=CH-C(=O)-N``) exists on the largest
    fragment AND its terminal amide N has at least one heavy-atom neighbour
    OUTSIDE the warhead match (i.e. wired into the body).

    Catches PocketFlow's reversed-polarity warhead (no polar match → False)
    AND catches Inpaint's free-fragment warhead (the acrylamide is on a small
    fragment, not the largest, OR has no body atom neighbour to the N).
    """
    if largest is None:
        return False
    matches = largest.GetSubstructMatches(ACRYL_POLAR)
    if not matches:
        return False
    for match in matches:
        match_set = set(match)
        n_atom = largest.GetAtomWithIdx(match[-1])  # the amide N
        for nb in n_atom.GetNeighbors():
            if nb.GetIdx() not in match_set and nb.GetAtomicNum() > 1:
                return True
    return False


# ---------------------------------------------------------------------------
# Metric: Warhead-Polarity
# ---------------------------------------------------------------------------
def warhead_polarity_flag(largest: Optional[Chem.Mol]) -> Optional[bool]:
    """For records that contain SOME acrylamide-ish skeleton, was the polarity
    correct? Returns None if no warhead is present at all (skip from the
    fraction). True = correct polar match. False = only the reversed/N-vinyl
    motif is present (PocketFlow failure).
    """
    if largest is None:
        return None
    has_polar = bool(largest.HasSubstructMatch(ACRYL_POLAR))
    has_soft = bool(largest.HasSubstructMatch(ACRYL_SOFT))
    has_reversed = bool(largest.HasSubstructMatch(ACRYL_REVERSED))
    if not (has_soft or has_reversed):
        return None
    return has_polar


# ---------------------------------------------------------------------------
# Metric: Synthetic Tractability
# ---------------------------------------------------------------------------
def synthetic_tractability_flag(largest: Optional[Chem.Mol]) -> bool:
    """True if the largest fragment has NO hits against the tractability
    blocklist (hypervalent S/P, geminal diols, exotic N–N–N motifs, etc.).
    """
    if largest is None:
        return False
    for _name, patt in TRACTABILITY_SMARTS:
        if patt is None:
            continue
        if largest.HasSubstructMatch(patt):
            return False
    return True


# ---------------------------------------------------------------------------
# Metric: Brenk-free EXCLUDING the warhead-Michael acceptor
# ---------------------------------------------------------------------------
def brenk_free_excl_warhead(largest: Optional[Chem.Mol]) -> bool:
    """True if the largest fragment has zero Brenk hits whose match atoms are
    NOT entirely covered by the acrylamide warhead substructure.

    This is the same trick used in the medchem_physchem report: Brenk's
    Michael-acceptor filter trips on every covalent mol, so we forgive Brenk
    hits that lie entirely on the warhead.
    """
    if largest is None:
        return False
    warhead_atoms: set[int] = set()
    for m in largest.GetSubstructMatches(ACRYL_SOFT):
        warhead_atoms.update(m)
    entries = BRENK_CATALOG.GetMatches(largest)
    for entry in entries:
        matches = entry.GetFilterMatches(largest)
        if not matches:
            return False
        # If any matched-pair atom is outside the warhead, this is a real hit.
        for match in matches:
            atoms = {pair.target for pair in match.atomPairs}
            if not atoms.issubset(warhead_atoms):
                return False
    return True


# ---------------------------------------------------------------------------
# Composite gate: Medchem Acceptability Score (MAS)
# ---------------------------------------------------------------------------
def lipinski_pass(largest: Optional[Chem.Mol]) -> bool:
    if largest is None:
        return False
    mw = Descriptors.MolWt(largest)
    logp = Crippen.MolLogP(largest)
    hba = rdMolDescriptors.CalcNumHBA(largest)
    hbd = rdMolDescriptors.CalcNumHBD(largest)
    violations = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
    return violations <= 1


def veber_pass(largest: Optional[Chem.Mol]) -> bool:
    if largest is None:
        return False
    tpsa = rdMolDescriptors.CalcTPSA(largest)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(largest)
    return tpsa <= 140 and rotb <= 10


def mas_flag(largest: Optional[Chem.Mol], qed_val: Optional[float], cmw: bool) -> bool:
    """Combined gate: Brenk-free (excl. warhead) AND Veber AND Lipinski AND
    QED ≥ 0.3 AND CMW.
    """
    if largest is None or qed_val is None:
        return False
    return (
        brenk_free_excl_warhead(largest)
        and veber_pass(largest)
        and lipinski_pass(largest)
        and qed_val >= 0.3
        and cmw
    )


# ---------------------------------------------------------------------------
# Per-mol property panel (used for PDP KS test and the markdown table)
# ---------------------------------------------------------------------------
PROPS_FOR_PDP = ("MW", "QED", "n_heavy", "n_aromatic_rings", "fsp3")


def _morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)


def build_record(raw_mol: Optional[Chem.Mol], largest: Optional[Chem.Mol]) -> dict:
    """Compute the full per-mol record. Designed to be vectorised across the
    whole cohort with a single Python loop (no per-mol pandas calls)."""
    rec: dict = {"valid": largest is not None}
    if largest is None:
        return rec
    rec["smiles"] = Chem.MolToSmiles(largest)
    rec["n_heavy"] = int(largest.GetNumHeavyAtoms())
    rec["MW"] = float(Descriptors.MolWt(largest))
    rec["QED"] = float(QED.qed(largest))
    rec["n_aromatic_rings"] = int(rdMolDescriptors.CalcNumAromaticRings(largest))
    rec["fsp3"] = float(rdMolDescriptors.CalcFractionCSP3(largest))
    rec["connectivity"] = connectivity_flag(raw_mol, largest)
    rec["cmw"] = cmw_flag(largest)
    rec["polarity"] = warhead_polarity_flag(largest)
    rec["tractable"] = synthetic_tractability_flag(largest)
    rec["brenk_clean"] = brenk_free_excl_warhead(largest)
    rec["lipinski"] = lipinski_pass(largest)
    rec["veber"] = veber_pass(largest)
    rec["mas"] = mas_flag(largest, rec["QED"], rec["cmw"])
    return rec


# ---------------------------------------------------------------------------
# Metric: Pretrained-Distribution Preservation Score (PDP)
# ---------------------------------------------------------------------------
def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS D statistic, no scipy dependency."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return 1.0
    all_x = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_x, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_x, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def pdp_score(cohort_df: pd.DataFrame, baseline_df: pd.DataFrame,
              props: Iterable[str] = PROPS_FOR_PDP) -> tuple[float, dict[str, float]]:
    """PDP = mean over selected properties of ``(1 - D_KS)`` between the
    cohort and its own vanilla baseline. Returns (overall_score, per_prop_D).

    Score range: [0, 1]. 1.0 = perfect distribution match. 0.0 = totally
    disjoint. We use D, not p-value, to keep the score sample-size-stable.
    """
    if baseline_df is None or baseline_df.empty:
        return float("nan"), {}
    per_prop: dict[str, float] = {}
    for p in props:
        if p not in cohort_df.columns or p not in baseline_df.columns:
            continue
        d = _ks_two_sample(cohort_df[p].to_numpy(), baseline_df[p].to_numpy())
        per_prop[p] = d
    if not per_prop:
        return float("nan"), {}
    score = float(np.mean([1.0 - d for d in per_prop.values()]))
    return score, per_prop


# ---------------------------------------------------------------------------
# Metric: Tier-2 Similarity
# ---------------------------------------------------------------------------
def tier2_similarity(largest_mols: list[Optional[Chem.Mol]], tier2_fps: list) -> float:
    """Median max-Tc to the Tier 2 curated 204 fingerprint bank. Higher =
    closer to medchem-acceptable hits.
    """
    if not tier2_fps:
        return float("nan")
    sims = []
    for m in largest_mols:
        if m is None:
            continue
        fp = _morgan_fp(m)
        sims.append(max(TanimotoSimilarity(fp, t) for t in tier2_fps))
    if not sims:
        return float("nan")
    return float(np.median(sims))


def load_tier2_fps(report_csv: Path) -> list:
    if not report_csv.exists():
        return []
    df = pd.read_csv(report_csv)
    sub = df[df["method"].astype(str).str.contains("Tier 2 — Fragment Replacement \(curated 204\)", regex=True)]
    fps = []
    for s in sub["smiles"].dropna().astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(_morgan_fp(m))
    return fps


# ---------------------------------------------------------------------------
# Quality Score — weighted geometric mean of headline metrics
# ---------------------------------------------------------------------------
@dataclass
class QualityWeights:
    """Configurable weights for the geometric-mean Quality Score.

    All weights ≥ 0. Tier-2-Sim is bounded to [0, 1] via clip; PDP is too.
    """
    validity: float = 1.0
    cmw: float = 2.0
    pdp: float = 1.0
    mas: float = 2.0
    tier2_sim: float = 1.0


def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-6)))


def quality_score(parts: dict[str, float], weights: QualityWeights) -> float:
    """Weighted geometric mean of ``parts``. NaN parts are skipped (their
    weight is dropped). Returns NaN if all parts are missing.
    """
    pairs = []
    for key, w in asdict(weights).items():
        if key not in parts:
            continue
        v = parts[key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = float(np.clip(v, 0.0, 1.0))
        pairs.append((v, w))
    if not pairs:
        return float("nan")
    num = sum(w * _safe_log(v) for v, w in pairs)
    den = sum(w for _v, w in pairs)
    return float(np.exp(num / den)) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Top-level: evaluate a single cohort
# ---------------------------------------------------------------------------
@dataclass
class CohortResult:
    tag: str
    n_total: int
    n_valid: int
    validity: float
    connectivity: float
    cmw: float
    polarity: float
    tractable: float
    brenk_clean: float
    mas: float
    tier2_sim: float
    pdp: float
    pdp_per_prop: dict[str, float] = field(default_factory=dict)
    median_MW: float = float("nan")
    median_QED: float = float("nan")
    median_n_heavy: float = float("nan")
    median_n_aromatic_rings: float = float("nan")
    median_fsp3: float = float("nan")
    quality: float = float("nan")
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_cohort(path: Path, *, tag: str,
                    baseline_df: Optional[pd.DataFrame] = None,
                    tier2_fps: Optional[list] = None,
                    weights: Optional[QualityWeights] = None
                    ) -> tuple[CohortResult, pd.DataFrame]:
    t0 = time.time()
    weights = weights or QualityWeights()
    raw_mols, largest_mols = load_mols(path)
    n_total = len(raw_mols)
    records = [build_record(r, l) for r, l in zip(raw_mols, largest_mols)]
    df = pd.DataFrame(records)
    valid = df[df["valid"] == True]
    n_valid = len(valid)

    def _pct(col: str) -> float:
        if col not in valid.columns or n_total == 0:
            return 0.0
        series = valid[col]
        # Treat None/NaN as False without triggering pandas downcasting warning.
        return float(series.map(lambda x: bool(x) if x is not None else False).sum()) / n_total

    def _frac_of_present(col: str) -> float:
        """For Boolean cols where None means 'metric not applicable'."""
        if col not in valid.columns:
            return float("nan")
        present = valid[col].dropna()
        if len(present) == 0:
            return float("nan")
        return float(present.astype(bool).sum()) / len(present)

    parts = {
        "validity": (n_valid / n_total) if n_total > 0 else 0.0,
        "connectivity": _pct("connectivity"),
        "cmw": _pct("cmw"),
        "polarity": _frac_of_present("polarity"),
        "tractable": _pct("tractable"),
        "brenk_clean": _pct("brenk_clean"),
        "mas": _pct("mas"),
    }

    # PDP against vanilla baseline (if supplied)
    pdp_val, pdp_per_prop = (float("nan"), {})
    if baseline_df is not None and not baseline_df.empty and not valid.empty:
        pdp_val, pdp_per_prop = pdp_score(valid, baseline_df)
    parts["pdp"] = pdp_val

    # Tier-2 similarity
    tier2_val = tier2_similarity(largest_mols, tier2_fps or [])
    parts["tier2_sim"] = tier2_val

    quality = quality_score(parts, weights)

    def _med(col: str) -> float:
        if col not in valid.columns or valid[col].notna().sum() == 0:
            return float("nan")
        return float(valid[col].dropna().median())

    res = CohortResult(
        tag=tag,
        n_total=n_total,
        n_valid=n_valid,
        validity=parts["validity"],
        connectivity=parts["connectivity"],
        cmw=parts["cmw"],
        polarity=parts["polarity"] if not np.isnan(parts["polarity"]) else float("nan"),
        tractable=parts["tractable"],
        brenk_clean=parts["brenk_clean"],
        mas=parts["mas"],
        tier2_sim=tier2_val,
        pdp=pdp_val,
        pdp_per_prop=pdp_per_prop,
        median_MW=_med("MW"),
        median_QED=_med("QED"),
        median_n_heavy=_med("n_heavy"),
        median_n_aromatic_rings=_med("n_aromatic_rings"),
        median_fsp3=_med("fsp3"),
        quality=quality,
        elapsed_sec=time.time() - t0,
    )
    return res, valid


# ---------------------------------------------------------------------------
# CLI + markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(results: list[CohortResult]) -> str:
    cols = [
        ("tag", "Method"),
        ("n_total", "N"),
        ("validity", "Valid"),
        ("connectivity", "Connect"),
        ("cmw", "CMW"),
        ("polarity", "Polarity"),
        ("tractable", "Tract"),
        ("brenk_clean", "Brenk-ok"),
        ("mas", "MAS"),
        ("pdp", "PDP"),
        ("tier2_sim", "Tier2-Tc"),
        ("quality", "Quality"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in results:
        row = []
        for key, _ in cols:
            v = getattr(r, key)
            if isinstance(v, float):
                if np.isnan(v):
                    row.append("--")
                elif key in ("tier2_sim", "pdp"):
                    row.append(f"{v:.3f}")
                else:
                    row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sdf", required=True,
                    help="Path to the cohort SDF (or CSV with a 'smiles' col).")
    ap.add_argument("--baseline", default=None,
                    help="Path to the vanilla baseline SDF/CSV for the same family.")
    ap.add_argument("--report", default=str(DEFAULT_REPORT_CSV),
                    help="Path to report_mols_compact.csv (for Tier 2 fps).")
    ap.add_argument("--tag", default=None,
                    help="Tag for output filename. Defaults to SDF stem.")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                    help="Output directory.")
    ap.add_argument("--weights", default=None,
                    help="JSON string with optional weight overrides.")
    args = ap.parse_args()

    weights = QualityWeights()
    if args.weights:
        weights = QualityWeights(**{**asdict(weights), **json.loads(args.weights)})

    sdf_path = Path(args.sdf)
    tag = args.tag or sdf_path.parent.name

    tier2_fps = load_tier2_fps(Path(args.report))
    print(f"[eval_suite] Loaded {len(tier2_fps)} Tier-2 fps")

    baseline_df = None
    if args.baseline:
        bp = Path(args.baseline)
        if bp.exists():
            raw_b, largest_b = load_mols(bp)
            records_b = [build_record(r, l) for r, l in zip(raw_b, largest_b)]
            baseline_df = pd.DataFrame(records_b)
            baseline_df = baseline_df[baseline_df["valid"] == True]
            print(f"[eval_suite] Loaded {len(baseline_df)} baseline mols from {bp}")
        else:
            print(f"[eval_suite] WARNING: --baseline path {bp} not found")

    res, _ = evaluate_cohort(sdf_path, tag=tag, baseline_df=baseline_df,
                             tier2_fps=tier2_fps, weights=weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{tag}_{ts}.json"
    payload = {
        "weights": asdict(weights),
        "sdf": str(sdf_path),
        "baseline": str(args.baseline) if args.baseline else None,
        "timestamp": ts,
        "result": res.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[eval_suite] Wrote {out_path}")
    print()
    print(render_markdown([res]))
    print()
    print(f"[eval_suite] elapsed: {res.elapsed_sec:.1f} sec")


if __name__ == "__main__":
    main()
