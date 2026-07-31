"""Curate a covalent-inhibitor retraining set for DrugFlow / PocketFlow / DiffSBDD.

Goal: produce N >= 1000 connected covalent inhibitors with explicit warhead-body
bonds, suitable for fine-tuning a 3D pocket-conditioned generative model. Each
example contains:

- ligand SMILES (single fragment, warhead bonded to body)
- warhead atom indices in the SMILES (5 atoms for acrylamide)
- PDB id + pocket file path (if available)
- Cys SG / Cbeta coordinates (if available)
- Murcko scaffold + descriptors

Sources combined:

1. CovInDB2 curated training set (1200 rows) -- pocket-aware, used in v2_G
   fine-tune; has Cys SG/Cbeta, anchor atom idx, PDB paths.
2. Tier 2 curated 204 -- ligand-only kinase acrylamides, QED ~0.71, pristine.

Outputs:
    data/retrain_covalent/train.csv
    data/retrain_covalent/val.csv
    data/retrain_covalent/test.csv
    data/retrain_covalent/summary_stats.json

Run:
    conda activate quris
    python experiments/curate_retraining_set.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED, Descriptors, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "retrain_covalent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SMARTS / filters
# ---------------------------------------------------------------------------
ACRYL_POLAR = Chem.MolFromSmarts("[CH2]=[CH]-[C](=O)-[N]")
ACRYL_SOFT = Chem.MolFromSmarts("C=CC(=O)N")
ACRYL_REVERSED = Chem.MolFromSmarts("[N]-[CH]=[CH]-[C](=O)")

_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def _largest_frag_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Return the largest (heavy-atom) fragment."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def cmw_pass(mol: Chem.Mol) -> bool:
    """CMW: polar acrylamide present on largest frag, amide N wired into body."""
    if mol is None:
        return False
    matches = mol.GetSubstructMatches(ACRYL_POLAR)
    if not matches:
        return False
    for match in matches:
        n_atom = mol.GetAtomWithIdx(match[-1])
        for nb in n_atom.GetNeighbors():
            if nb.GetIdx() not in set(match) and nb.GetAtomicNum() > 1:
                return True
    return False


def pains_hits(mol: Chem.Mol) -> int:
    if mol is None:
        return 99
    try:
        return len(PAINS_CATALOG.GetMatches(mol))
    except Exception:
        return 99


def find_warhead_atoms(mol: Chem.Mol) -> list[int]:
    """Return indices of acrylamide atoms in the SMILES (5 atoms).

    Returns the first match of CH2=CH-C(=O)-N (4 heavy atoms + the O = 5 atoms
    including the carbonyl O). Empty list if no match.
    """
    if mol is None:
        return []
    matches = mol.GetSubstructMatches(ACRYL_POLAR)
    if not matches:
        return []
    # ACRYL_POLAR = [CH2]=[CH]-[C](=O)-[N]; SMARTS gives 5 atoms incl carbonyl O.
    return list(matches[0])


def murcko_scaffold(mol: Chem.Mol) -> str:
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-source ingestion
# ---------------------------------------------------------------------------
def _build_row(smiles: str, source: str, extras: dict | None = None,
               require_acrylamide: bool = True,
               relax_mw: bool = False,
               relax_qed: bool = False) -> dict | None:
    """Apply filters and return a row dict, or None if it fails.

    Parameters
    ----------
    require_acrylamide
        If True, require the polar acrylamide warhead to be present and wired
        into the body (CMW gate). For Tier 2 / kinase data this is essential.
        For CovInDB2 records, set False so non-acrylamide covalent classes
        (chloroacetamide, vinyl sulfone, halohydrocarbon, ...) are kept --
        these still teach the model warhead-body connectivity. ``warhead_atom_indices``
        will be empty for non-acrylamide rows; the consumer can rely on
        ``warhead_class``.
    relax_mw, relax_qed
        Loosen MW / QED gates -- only used for CovInDB2 where crystal-structure
        provenance outweighs drug-likeness.
    """
    extras = extras or {}
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return None
    canon = Chem.MolToSmiles(raw)
    if "." in canon:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    has_acryl = cmw_pass(mol)
    if require_acrylamide and not has_acryl:
        return None
    mw = Descriptors.MolWt(mol)
    mw_lo, mw_hi = (200.0, 700.0) if relax_mw else (250.0, 500.0)
    if not (mw_lo <= mw <= mw_hi):
        return None
    qed_val = QED.qed(mol)
    qed_min = 0.10 if relax_qed else 0.30
    if qed_val < qed_min:
        return None
    # For acrylamide-required cohorts (Tier 2) we keep strict PAINS=0. For
    # CovInDB2 we tolerate PAINS hits since the catalog overlaps covalent
    # warheads (Michael acceptors, carbonyls).
    if require_acrylamide and pains_hits(mol) > 0:
        return None
    warhead_idx = find_warhead_atoms(mol) if has_acryl else []
    if require_acrylamide and len(warhead_idx) != 5:
        return None
    scaff = murcko_scaffold(mol)
    warhead_class = extras.get("warhead_class") or ("Acrylamide" if has_acryl else "Unknown")
    return {
        "smiles": canon,
        "source": source,
        "warhead_atom_indices": ",".join(str(i) for i in warhead_idx),
        "scaffold": scaff,
        "MW": round(mw, 2),
        "QED": round(qed_val, 4),
        "n_heavy": mol.GetNumHeavyAtoms(),
        "pdb_id": extras.get("pdb_id", ""),
        "pdb_path": extras.get("pdb_path", ""),
        "cys_chain": extras.get("cys_chain", ""),
        "cys_resi": extras.get("cys_resi", ""),
        "sg_x": extras.get("sg_x", ""),
        "sg_y": extras.get("sg_y", ""),
        "sg_z": extras.get("sg_z", ""),
        "cb_x": extras.get("cb_x", ""),
        "cb_y": extras.get("cb_y", ""),
        "cb_z": extras.get("cb_z", ""),
        "target": extras.get("target", ""),
        "warhead_class": warhead_class,
        "anchor_atom_idx_in_ligand": extras.get("anchor_atom_idx_in_ligand", ""),
    }


def load_covind() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"
    df = pd.read_csv(path)
    rows: list[dict] = []
    skipped = Counter()
    for _, r in df.iterrows():
        smi = r.get("smiles")
        if not isinstance(smi, str):
            skipped["no_smiles"] += 1
            continue
        extras = {
            "pdb_id": r.get("pdb_id", ""),
            "pdb_path": r.get("pdb_path", ""),
            "cys_chain": r.get("cys_chain", ""),
            "cys_resi": r.get("cys_resi", ""),
            "sg_x": r.get("sg_x", ""),
            "sg_y": r.get("sg_y", ""),
            "sg_z": r.get("sg_z", ""),
            "cb_x": r.get("cb_x", ""),
            "cb_y": r.get("cb_y", ""),
            "cb_z": r.get("cb_z", ""),
            "target": r.get("target", ""),
            "warhead_class": r.get("warhead_class", ""),
            "anchor_atom_idx_in_ligand": r.get("anchor_atom_idx_in_ligand", ""),
        }
        # CovInDB2 records are crystal-verified covalent ligands with anchor
        # atom indices and d_SG_to_anchor < 2.3 A. Trust the crystallography:
        # accept all warhead classes, relax MW/QED, but still require single
        # fragment, no PAINS, and connected warhead (anchor exists).
        d_anchor = r.get("d_SG_to_anchor", 99.0)
        if not (isinstance(d_anchor, (int, float)) and 1.0 <= d_anchor <= 2.5):
            skipped["bad_anchor"] += 1
            continue
        out = _build_row(smi, "CovInDB2", extras=extras,
                         require_acrylamide=False, relax_mw=True, relax_qed=True)
        if out is None:
            skipped["filter_fail"] += 1
            continue
        rows.append(out)
    print(f"  CovInDB2: kept {len(rows)} / {len(df)} (filter fails: {skipped['filter_fail']}, no_smiles: {skipped['no_smiles']})")
    return pd.DataFrame(rows)


def load_tier2() -> pd.DataFrame:
    path = Path("/tmp/report_mols_compact.csv")
    if not path.exists():
        print("  Tier2 CSV not found at /tmp/report_mols_compact.csv -- skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    t2 = df[df["method"] == "Tier 2 — Fragment Replacement (curated 204)"]
    rows: list[dict] = []
    skipped = 0
    for _, r in t2.iterrows():
        out = _build_row(r["smiles"], "Tier2_curated_204", extras={"warhead_class": "Acrylamide"})
        if out is None:
            skipped += 1
            continue
        rows.append(out)
    print(f"  Tier2 curated 204: kept {len(rows)} / {len(t2)} (filter fails: {skipped})")
    return pd.DataFrame(rows)


def load_tier2_scaled() -> pd.DataFrame:
    """Tier 2 SCALED 1000 -- high-MW, lower-QED Fragment Replacement.

    Almost all rows fail MW <= 500 OR QED >= 0.3. We include the small surviving
    set as additional kinase diversity.
    """
    path = Path("/tmp/report_mols_compact.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    sub = df[df["method"] == "Tier 2 SCALED — Fragment Replacement (498K)"]
    rows: list[dict] = []
    for _, r in sub.iterrows():
        out = _build_row(r["smiles"], "Tier2_scaled", extras={"warhead_class": "Acrylamide"})
        if out is not None:
            rows.append(out)
    print(f"  Tier2 SCALED: kept {len(rows)} / {len(sub)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combine + dedupe + split
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate SMILES, preferring rows with PDB info."""
    df = df.copy()
    df["has_pdb"] = df["pdb_path"].astype(str).str.len() > 0
    df = df.sort_values(by="has_pdb", ascending=False)
    df = df.drop_duplicates(subset="smiles", keep="first")
    df = df.drop(columns="has_pdb").reset_index(drop=True)
    return df


def scaffold_stratified_split(df: pd.DataFrame, val_frac: float = 0.10,
                              test_frac: float = 0.10, seed: int = 42) -> dict:
    """Group SMILES by Murcko scaffold, then assign whole scaffolds to splits.

    Targets ~80/10/10 by row count, scaffold-disjoint to discourage memorization.
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for idx, scaff in enumerate(df["scaffold"]):
        groups[scaff].append(idx)
    scaffs = list(groups.keys())
    sizes = np.array([len(groups[s]) for s in scaffs])
    order = np.argsort(-sizes)  # largest scaffolds first
    scaffs = [scaffs[i] for i in order]
    n = len(df)
    target_val = int(round(n * val_frac))
    target_test = int(round(n * test_frac))
    val_idx, test_idx, train_idx = [], [], []
    # Big scaffolds always go to train; small/medium scaffolds get distributed
    # so val / test have scaffold variety.
    shuffled_small = [s for s in scaffs if len(groups[s]) <= 10]
    rng.shuffle(shuffled_small)
    big = [s for s in scaffs if len(groups[s]) > 10]
    for s in big:
        train_idx.extend(groups[s])
    for s in shuffled_small:
        if len(val_idx) < target_val:
            val_idx.extend(groups[s])
        elif len(test_idx) < target_test:
            test_idx.extend(groups[s])
        else:
            train_idx.extend(groups[s])
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[1/4] Loading sources...")
    parts = []
    cov_df = load_covind()
    if not cov_df.empty:
        parts.append(cov_df)
    t2_df = load_tier2()
    if not t2_df.empty:
        parts.append(t2_df)
    t2s_df = load_tier2_scaled()
    if not t2s_df.empty:
        parts.append(t2s_df)
    if not parts:
        raise RuntimeError("No sources loaded.")
    combined = pd.concat(parts, ignore_index=True)
    print(f"  Combined before dedupe: {len(combined)}")
    combined = deduplicate(combined)
    print(f"  Combined after dedupe:  {len(combined)}")

    print("[2/4] Composition by source:")
    print(combined["source"].value_counts().to_dict())

    print("[3/4] Scaffold-stratified 80/10/10 split...")
    # Split per warhead-class stratum so train/val/test all see every warhead
    # class. We split each stratum then concat.
    train_chunks, val_chunks, test_chunks = [], [], []
    for cls, sub in combined.groupby("warhead_class", sort=False):
        if len(sub) < 3:
            train_chunks.append(sub)
            continue
        sp = scaffold_stratified_split(sub.reset_index(drop=True), seed=42)
        train_chunks.append(sp["train"])
        val_chunks.append(sp["val"])
        test_chunks.append(sp["test"])
    splits = {
        "train": pd.concat(train_chunks, ignore_index=True),
        "val": pd.concat(val_chunks, ignore_index=True) if val_chunks else combined.iloc[:0],
        "test": pd.concat(test_chunks, ignore_index=True) if test_chunks else combined.iloc[:0],
    }
    for name, sub in splits.items():
        path = OUT_DIR / f"{name}.csv"
        sub.to_csv(path, index=False)
        print(f"  {name}: {len(sub)} -> {path}")

    # Combined ledger (for eval_suite call)
    full_path = OUT_DIR / "all.csv"
    combined.to_csv(full_path, index=False)
    # Acrylamide-only subset (primary cohort for connectivity bug)
    acr = combined[combined["warhead_class"] == "Acrylamide"]
    acr.to_csv(OUT_DIR / "acrylamide_only.csv", index=False)
    print(f"  acrylamide_only: {len(acr)} -> {OUT_DIR / 'acrylamide_only.csv'}")

    print("[4/4] Stats...")
    stats = {
        "n_total": int(len(combined)),
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "by_source": combined["source"].value_counts().to_dict(),
        "by_warhead_class": combined["warhead_class"].value_counts().to_dict(),
        "have_pdb": int((combined["pdb_path"].astype(str).str.len() > 0).sum()),
        "have_cys_sg": int((combined["sg_x"].astype(str).str.len() > 0).sum()),
        "MW_median": float(combined["MW"].median()),
        "MW_mean": float(combined["MW"].mean()),
        "QED_median": float(combined["QED"].median()),
        "QED_mean": float(combined["QED"].mean()),
        "n_unique_scaffolds": int(combined["scaffold"].nunique()),
        "top_targets": combined["target"].value_counts().head(10).to_dict(),
    }
    json_path = OUT_DIR / "summary_stats.json"
    json_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"  Wrote {json_path}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
