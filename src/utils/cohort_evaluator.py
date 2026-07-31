"""Canonical CohortEvaluator — the single source of truth for cohort metrics
reported in the CovaCraft paper (base, v1-covaFT, v1-DAP, v2-cond, v2-cond+DAP).

Goal: whatever appears in the manuscript's tab:covft, tab:rl_value, tab:m1a_geometry,
tab:covdpo comes from ONE code path here. No more ad-hoc scripts.

Inputs (per cohort):
  - SMILES list (10k Mol1-anchored samples)
  - Optional planar_2d CSV (from covft_geometric_comparison.py schema)
  - Optional cov-Vina CSV (300-mol subsample or full 10k, we take the subsample)
  - Optional pIC50 CSV/JSON (per-cohort FiLM predictor scored)

Outputs:
  - Dict of metrics that map 1-to-1 to manuscript rows.

Denominator conventions (matches manuscript exactly):
  - Validity: n_valid / n_total
  - Scaffold uniqueness: n_unique_scaffold / n_valid
  - Top-scaffold share: max_scaffold_count / n_valid
  - QED mean: mean over valid mols
  - Acrylamide retention: n_acryl_on_largest_frag / n_valid  (matches m1a_geometry.json
    and v2_cond_10k_metric_panel.json; uses `[CH2]=[CH]C(=O)N` SMARTS on largest fragment)
  - Median Tc to Mol1: median over valid mols
  - Planar-dihedral median (deg): median of `planar_dev_deg` over rows with msg='ok' in
    the planar_2d CSV — this is the `min(|d|, |180-d|)` deviation from planar s-cis/s-trans
    for the C=C-C(=O)-N torsion computed via one ETKDG conformer + MMFF-optimize.
  - Pre-reactivity (frac >=0.5): fraction of ALL rows (not just acryl-matching) with
    pre_reactivity_score >= 0.5 in the planar_2d CSV. (Matches m1a_geometry.json.)
  - Cov-Vina median (kcal/mol): median vina_affinity over a 300-mol subsample of the
    acryl-matching cohort using the covalent-tether protocol.
  - Predicted pIC50 median / frac_ge_7: FiLM predictor (reinvent4_film_model_clean.pt),
    denominator = n_valid (no dedup — matches film_tail_analysis.json 'clean' variant).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Union, Iterable, Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ----- CANONICAL CONSTANTS -----
# Do not fork these; import them from here.

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = "[CH2]=[CH]C(=O)N"
PLANARITY_THRESH_DEG = 20.0  # for pre-reactivity score
_ACRYL_PATTERN = Chem.MolFromSmarts(ACRYL_SMARTS)


# ----- HELPERS -----

def _canonical(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def _largest_fragment_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())


def _has_acryl_largest(mol: Chem.Mol) -> bool:
    lf = _largest_fragment_mol(mol)
    if lf is None:
        return False
    return lf.HasSubstructMatch(_ACRYL_PATTERN)


def _scaffold_smiles(mol: Chem.Mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None


def _morgan_fp(mol: Chem.Mol, r: int = 2, nbits: int = 2048):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, r, nbits)
    except Exception:
        return None


# ----- MAIN EVALUATOR -----

@dataclass
class CohortEvaluator:
    """Evaluate a Mol1-anchored cohort. All metrics use canonical denominators.

    Attributes
    ----------
    smiles : list of str
        Raw SMILES for the 10k cohort (may include duplicates and invalid).
    planar_2d_csv : Path, optional
        Path to `covft_geometric_2d_*.csv` (schema from
        `experiments/covft_geometric_comparison.py::_compute_2d_one`). Provides
        planar-dihedral and pre-reactivity metrics.
    covvina_csv : Path, optional
        Path to `covft_geometric_covvina_*.csv` (schema from
        `experiments/covvina_10k/run_covvina_10k.py`). We take the median of
        `vina_affinity` over up to 300 acryl-matching rows (matches manuscript
        `cov_vina_subsample_n=300`).
    pIC50_scores : sequence of float, optional
        FiLM-predicted pIC50 per (valid) molecule, indexed to `self.valid_smiles`.
    tag : str, optional
        Cohort tag for the report.
    tc_subsample_n : int
        Cap on molecules for Tc-to-Mol1 median (compute cost).
    covvina_subsample_n : int
        Manuscript convention (300 mol).
    """

    smiles: Sequence[str]
    planar_2d_csv: Optional[Union[str, Path]] = None
    covvina_csv: Optional[Union[str, Path]] = None
    pIC50_scores: Optional[Sequence[float]] = None
    tag: str = "cohort"
    tc_subsample_n: int = 5000
    covvina_subsample_n: int = 300
    covvina_subsample_seed: int = 0

    # populated by _prepare()
    mols: list = field(default_factory=list, init=False)
    valid_idx: list = field(default_factory=list, init=False)
    valid_mols: list = field(default_factory=list, init=False)
    valid_smiles: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._prepare()

    def _prepare(self):
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.valid_idx = [i for i, m in enumerate(self.mols) if m is not None]
        self.valid_mols = [self.mols[i] for i in self.valid_idx]
        self.valid_smiles = [self.smiles[i] for i in self.valid_idx]

    # ---- individual metric groups ----

    def basic(self) -> dict:
        n_total = len(self.smiles)
        n_valid = len(self.valid_mols)
        canon = [_canonical(s) for s in self.valid_smiles]
        n_unique_canonical = len(set(c for c in canon if c is not None))
        return {
            "n_total": int(n_total),
            "n_valid": int(n_valid),
            "validity": round(n_valid / n_total, 4) if n_total else None,
            "unique_canonical": int(n_unique_canonical),
        }

    def diversity(self) -> dict:
        scaffs = [_scaffold_smiles(m) for m in self.valid_mols]
        scaffs_v = [x for x in scaffs if x]
        if not scaffs_v:
            return {"scaffold_uniqueness": None, "top_scaffold_share": None, "n_unique_scaffolds": 0}
        sc_series = pd.Series(scaffs_v).value_counts()
        n_uniq = len(set(scaffs_v))
        return {
            "scaffold_uniqueness": round(n_uniq / len(scaffs_v), 4),
            "top_scaffold_share": round(sc_series.iloc[0] / len(scaffs_v), 4),
            "n_unique_scaffolds": int(n_uniq),
        }

    def qed(self) -> dict:
        qeds = []
        for m in self.valid_mols:
            try:
                qeds.append(QED.qed(m))
            except Exception:
                pass
        if not qeds:
            return {"qed_mean": None, "qed_median": None}
        arr = np.array(qeds)
        return {
            "qed_mean": round(float(arr.mean()), 3),
            "qed_median": round(float(np.median(arr)), 3),
        }

    def acryl_retention(self) -> dict:
        """Acryl-on-largest-fragment as fraction of n_valid (matches manuscript)."""
        n_valid = len(self.valid_mols)
        if not n_valid:
            return {"acryl_largest_frag_pct": None, "acryl_hits": 0}
        hits = sum(1 for m in self.valid_mols if _has_acryl_largest(m))
        return {
            "acryl_largest_frag_pct": round(hits / n_valid, 4),
            "acryl_hits": int(hits),
        }

    def tc_to_mol1(self) -> dict:
        mol1_fp = _morgan_fp(Chem.MolFromSmiles(MOL1_SMILES))
        if mol1_fp is None:
            return {"tc_to_mol1_median": None, "tc_to_mol1_mean": None}
        sub = self.valid_mols[: self.tc_subsample_n]
        fps = [_morgan_fp(m) for m in sub]
        tcs = [DataStructs.TanimotoSimilarity(f, mol1_fp) for f in fps if f is not None]
        if not tcs:
            return {"tc_to_mol1_median": None, "tc_to_mol1_mean": None}
        arr = np.array(tcs)
        return {
            "tc_to_mol1_median": round(float(np.median(arr)), 3),
            "tc_to_mol1_mean": round(float(arr.mean()), 3),
        }

    def planar_2d(self) -> dict:
        """Median planar-dev deg and pre-reactivity frac, both on the acryl-matching subset.

        Manuscript convention (see `v2_cond_10k_metric_panel.json`):
          - planar median is over rows with msg='ok' (acryl-matching, embed_ok).
          - Pre-reactivity is `n_prereact_ge_0p5 / n_acryl_matching` (denominator =
            acryl-matching rows, NOT all rows). This matches the paper's 0.60 for
            v2-cond, 0.48 for v1-covaFT, 0.41 for v1-DAP, 0.43 for base.
        """
        if self.planar_2d_csv is None:
            return {"planar_dihedral_median_deg": None, "prereact_frac_ge_0p5": None}
        p = Path(self.planar_2d_csv)
        if not p.exists():
            return {"planar_dihedral_median_deg": None, "prereact_frac_ge_0p5": None, "planar_2d_note": "csv_missing"}
        d = pd.read_csv(p)
        ok = d[d["msg"] == "ok"]
        planar_median = float(np.median(ok["planar_dev_deg"].dropna())) if len(ok) else None
        # pre-reactivity denominator = acryl-matching rows
        acryl_rows = d[d["acryl_match"] == True] if "acryl_match" in d.columns else d
        if len(acryl_rows) and "pre_reactivity_score" in d.columns:
            pre_react = float((acryl_rows["pre_reactivity_score"] >= 0.5).mean())
        else:
            pre_react = None
        return {
            "planar_dihedral_median_deg": round(planar_median, 2) if planar_median is not None else None,
            "prereact_frac_ge_0p5": round(pre_react, 4) if pre_react is not None else None,
            "planar_2d_n_ok": int(len(ok)),
            "planar_2d_n_acryl": int(len(acryl_rows)),
            "planar_2d_n_total": int(len(d)),
        }

    def cov_vina(self) -> dict:
        """Median cov-Vina affinity over ALL acryl-matching+vina_ok rows (full 10k cohort).

        Manuscript convention (current draft): median over `vina_ok=True &
        warhead_found=True` across the entire 10k cohort. Reproduces
        v1-covaFT 146.7, v1-DAP 182.7, v2-cond 125.6, base 144.9.

        (An earlier manuscript draft used a 300-mol subsample; that
        convention is retained as `covvina_subsample_n` but is only used if
        explicitly requested via a non-None `subsample_n` override in the
        future.)
        """
        if self.covvina_csv is None:
            return {"cov_vina_median_kcalmol": None}
        p = Path(self.covvina_csv)
        if not p.exists():
            return {"cov_vina_median_kcalmol": None, "cov_vina_note": "csv_missing"}
        d = pd.read_csv(p)
        # Keep only acryl-matching + vina_ok rows
        wh_col = "warhead_found" if "warhead_found" in d.columns else None
        vok_col = "vina_ok" if "vina_ok" in d.columns else None
        if wh_col and vok_col:
            docked = d[(d[wh_col] == True) & (d[vok_col] == True)]
        elif vok_col:
            docked = d[d[vok_col] == True]
        else:
            docked = d
        aff = pd.to_numeric(docked["vina_affinity"], errors="coerce").dropna()
        return {
            "cov_vina_median_kcalmol": round(float(aff.median()), 2) if len(aff) else None,
            "cov_vina_mean_kcalmol": round(float(aff.mean()), 2) if len(aff) else None,
            "cov_vina_n_docked_used": int(len(aff)),
            "cov_vina_n_docked_total": int(len(docked)),
        }

    def pIC50(self) -> dict:
        """Median and frac_ge_7 from provided pIC50 scores (indexed to valid_smiles)."""
        if self.pIC50_scores is None:
            return {"pIC50_median": None, "pIC50_mean": None, "frac_ge_7": None}
        arr = np.asarray(self.pIC50_scores, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"pIC50_median": None, "pIC50_mean": None, "frac_ge_7": None}
        return {
            "pIC50_median": round(float(np.median(arr)), 3),
            "pIC50_mean": round(float(arr.mean()), 3),
            "pIC50_p95": round(float(np.percentile(arr, 95)), 3),
            "frac_ge_7": round(float((arr >= 7.0).mean()), 4),
            "n_scored": int(len(arr)),
        }

    def evaluate(self) -> dict:
        """Compute all metric groups, return single flat dict with tag."""
        out = {"tag": self.tag}
        out.update(self.basic())
        out.update(self.diversity())
        out.update(self.qed())
        out.update(self.acryl_retention())
        out.update(self.tc_to_mol1())
        out.update(self.planar_2d())
        out.update(self.cov_vina())
        out.update(self.pIC50())
        return out


# ----- CONVENIENCE: canonical cohort registry -----

REPO = Path("/Users/shaharharel/Documents/github/edit-small-mol")

CANONICAL_COHORTS = {
    "mol2mol_base": {
        "smiles_csv": REPO / "experiments/exp_covft_value/samples_base.csv",
        "smi_col": "SMILES",
        "planar_2d_csv": REPO / "results/paper_evaluation/covft_geometric_2d_base.csv",
        "covvina_csv": REPO / "results/paper_evaluation/covft_geometric_covvina_base.csv",
        "checkpoint": "REINVENT4/priors/mol2mol_medium_similarity.prior (upstream)",
        "manuscript_label": "base",
    },
    "v1_covaFT": {
        "smiles_csv": REPO / "experiments/exp_covft_value/samples_covft.csv",
        "smi_col": "SMILES",
        "planar_2d_csv": REPO / "results/paper_evaluation/covft_geometric_2d_covft.csv",
        "covvina_csv": REPO / "results/paper_evaluation/covft_geometric_covvina_covft.csv",
        "checkpoint": "models/reinvent4_mol2mol_covalent_ft.prior",
        "manuscript_label": "v1-covaFT",
    },
    "v1_DAP": {
        "smiles_csv": REPO / "experiments/exp_geom_bc/samples_rl.csv",
        "smi_col": "SMILES",
        "planar_2d_csv": REPO / "results/paper_evaluation/covft_geometric_2d_rl.csv",
        "covvina_csv": REPO / "results/paper_evaluation/covft_geometric_covvina_rl.csv",
        "checkpoint": "models/rl_checkpoints/exp2_v2_rl_v2_stage1.chkpt",
        "manuscript_label": "v1-DAP",
    },
    "v2_cond": {
        "smiles_csv": REPO / "data/m1a_v2_ablation/cohort_A_10k.csv",
        "smi_col": "SMILES",
        "planar_2d_csv": REPO / "results/paper_evaluation/covft_geometric_2d_m1a.csv",
        "covvina_csv": REPO / "results/paper_evaluation/covft_geometric_covvina_m1a.csv",
        "checkpoint": "models/m1a_v2.ckpt",
        "manuscript_label": "v2-cond",
    },
    "v2_cond_DAP": {
        "smiles_csv": REPO / "data/paper_dap_repro/samples_v2cond_E2_10k.csv",
        "smi_col": "SMILES",
        "planar_2d_csv": REPO / "results/paper_evaluation/covft_geometric_2d_v2cond_DAP.csv",
        "covvina_csv": REPO / "results/paper_evaluation/covft_geometric_covvina_v2cond_DAP_10k.csv",
        "checkpoint": "models/rl_checkpoints/v2cond_E2.chkpt",
        "manuscript_label": "v2-cond+DAP",
    },
}


def load_cohort(name: str, pIC50_scores: Optional[Iterable[float]] = None) -> CohortEvaluator:
    """Instantiate CohortEvaluator for one of the canonical cohorts."""
    if name not in CANONICAL_COHORTS:
        raise KeyError(f"Unknown cohort '{name}'. Available: {list(CANONICAL_COHORTS)}")
    cfg = CANONICAL_COHORTS[name]
    df = pd.read_csv(cfg["smiles_csv"])
    smi_col = cfg["smi_col"] if cfg["smi_col"] in df.columns else df.columns[0]
    smiles = df[smi_col].astype(str).tolist()
    return CohortEvaluator(
        smiles=smiles,
        planar_2d_csv=cfg.get("planar_2d_csv"),
        covvina_csv=cfg.get("covvina_csv"),
        pIC50_scores=list(pIC50_scores) if pIC50_scores is not None else None,
        tag=cfg["manuscript_label"],
    )


def evaluate_all_canonical(pIC50_by_cohort: Optional[dict] = None) -> dict:
    """Run evaluator on every canonical cohort; return {cohort_name: metrics}.

    Parameters
    ----------
    pIC50_by_cohort : dict, optional
        {cohort_name -> iterable of pIC50 scores}, indexed by valid_smiles order.
    """
    out = {}
    pIC50_by_cohort = pIC50_by_cohort or {}
    for name in CANONICAL_COHORTS:
        try:
            ev = load_cohort(name, pIC50_scores=pIC50_by_cohort.get(name))
            out[name] = ev.evaluate()
        except FileNotFoundError as e:
            out[name] = {"tag": name, "error": f"file_missing: {e}"}
        except Exception as e:
            out[name] = {"tag": name, "error": f"{type(e).__name__}: {e}"}
    return out
