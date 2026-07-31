"""Cohort-level diversity metrics.

Computes per-cohort observability metrics to flag mode-collapse. All metrics
are computed on UNIQUE canonical SMILES within the cohort (the duplicates are
informative for `n_total` / `diversity_ratio` only).

Public API:
    canonicalize_smiles(smi) -> str | None
    morgan_fp(smi, radius=2, nbits=2048) -> ExplicitBitVect | None
    compute_intra_nn_tanimoto(smiles_list, radius=2, nbits=2048) -> dict
        returns {"mean_intra_nn_tanimoto": float, "max_intra_tanimoto": float,
                 "n_unique_with_fp": int}
    count_bemis_murcko_scaffolds(smiles_list) -> int
    compute_diversity_metrics(smiles_list_raw, radius=2, nbits=2048) -> dict
        the headline aggregator used by the cohort-comparison pipeline and
        the dashboard backend. Returns:
            n_total, n_unique, diversity_ratio,
            mean_intra_nn_tanimoto, max_intra_tanimoto,
            n_bemis_murcko_scaffolds, mean_pairwise_dissim

All metrics are OBSERVE-ONLY — no filtering happens here.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_FP_CACHE: dict[tuple[str, int, int], object] = {}


def canonicalize_smiles(smi: str | None) -> str | None:
    """RDKit canonical SMILES; returns None on parse failure / empty input."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    """Morgan FP (cached). Default 2048 bits per the cohort-diversity spec."""
    if not isinstance(smi, str) or not smi:
        return None
    key = (smi, radius, nbits)
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _FP_CACHE[key] = None
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    _FP_CACHE[key] = fp
    return fp


def bemis_murcko_scaffold(smi: str) -> str | None:
    """Return canonical SMILES of the Bemis-Murcko scaffold of `smi`."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        # Empty scaffold (acyclic mol) -> return empty SMILES as the scaffold key.
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def count_bemis_murcko_scaffolds(smiles_list: Iterable[str]) -> int:
    """Number of distinct Bemis-Murcko scaffolds in the input list.

    Includes the "empty-scaffold" bucket (acyclic mols) as a single bucket if
    present. Unparseable SMILES are skipped.
    """
    scaffolds: set[str] = set()
    for s in smiles_list:
        scaf = bemis_murcko_scaffold(s)
        if scaf is None:
            continue
        scaffolds.add(scaf)
    return len(scaffolds)


def compute_intra_nn_tanimoto(
    smiles_list: Sequence[str], radius: int = 2, nbits: int = 2048
) -> dict:
    """Compute intra-cohort nearest-neighbor Tanimoto stats.

    Operates on the input list directly — caller is responsible for deduping
    if desired. Returns the mean of per-molecule max-Tanimoto-to-any-other-mol
    and the global max pairwise Tanimoto (excluding self).

    Returns:
        {"mean_intra_nn_tanimoto": float, "max_intra_tanimoto": float,
         "n_unique_with_fp": int}
    """
    fps = []
    for s in smiles_list:
        fp = morgan_fp(s, radius=radius, nbits=nbits)
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 2:
        return {
            "mean_intra_nn_tanimoto": float("nan"),
            "max_intra_tanimoto": float("nan"),
            "n_unique_with_fp": n,
        }
    nn_vals = np.zeros(n, dtype=float)
    global_max = 0.0
    for i in range(n):
        others = fps[:i] + fps[i + 1:]
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], others)
        if sims:
            m = max(sims)
            nn_vals[i] = m
            if m > global_max:
                global_max = m
    return {
        "mean_intra_nn_tanimoto": float(nn_vals.mean()),
        "max_intra_tanimoto": float(global_max),
        "n_unique_with_fp": n,
    }


def compute_diversity_metrics(
    smiles_list_raw: Sequence[str | None],
    radius: int = 2,
    nbits: int = 2048,
) -> dict:
    """Compute the full per-cohort diversity panel.

    Args:
        smiles_list_raw: every SMILES emitted by the cohort (before dedup).
            Unparseable / empty entries are tolerated.

    Returns dict with keys (always present, NaN/0 on empty input):
        n_total                    — raw count (len(smiles_list_raw))
        n_unique                   — distinct canonical SMILES count
        diversity_ratio            — n_unique / n_total
        mean_intra_nn_tanimoto     — mean of per-mol NN Tc over uniques
        max_intra_tanimoto         — global max pairwise Tc over uniques
        n_bemis_murcko_scaffolds   — distinct BM scaffold count over uniques
        mean_pairwise_dissim       — 1 - mean_intra_nn_tanimoto
    """
    n_total = int(len(smiles_list_raw))
    canon = []
    for s in smiles_list_raw:
        c = canonicalize_smiles(s)
        if c is not None:
            canon.append(c)
    unique = sorted(set(canon))
    n_unique = len(unique)

    if n_unique == 0:
        return {
            "n_total": n_total,
            "n_unique": 0,
            "diversity_ratio": float("nan") if n_total == 0 else 0.0,
            "mean_intra_nn_tanimoto": float("nan"),
            "max_intra_tanimoto": float("nan"),
            "n_bemis_murcko_scaffolds": 0,
            "mean_pairwise_dissim": float("nan"),
        }

    nn_stats = compute_intra_nn_tanimoto(unique, radius=radius, nbits=nbits)
    n_scaf = count_bemis_murcko_scaffolds(unique)

    mean_nn = nn_stats["mean_intra_nn_tanimoto"]
    mean_dissim = (1.0 - mean_nn) if not np.isnan(mean_nn) else float("nan")

    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "diversity_ratio": n_unique / max(1, n_total),
        "mean_intra_nn_tanimoto": mean_nn,
        "max_intra_tanimoto": nn_stats["max_intra_tanimoto"],
        "n_bemis_murcko_scaffolds": int(n_scaf),
        "mean_pairwise_dissim": mean_dissim,
    }


__all__ = [
    "canonicalize_smiles",
    "morgan_fp",
    "bemis_murcko_scaffold",
    "count_bemis_murcko_scaffolds",
    "compute_intra_nn_tanimoto",
    "compute_diversity_metrics",
]
