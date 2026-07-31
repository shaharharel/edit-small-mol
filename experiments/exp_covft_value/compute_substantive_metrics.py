"""Substantive metric panel on 3 cohorts of 10k mols (Local CPU only).

Headline trio:
  1. FCD vs covalent-kinase reference set
  2. Hinge-binder pharmacophore hit rate
  3. Four-way conjunction (acrylamide AND hinge AND Tc-band-to-Mol1 AND ATP-pocket window)

Plus additional metrics (1-similarity band distribution, clinical-drug max-Tc,
ZAP70-seed max-Tc, CovInDB v2 training-memorization, scaffold novelty,
warhead-to-hinge topological distance, ATP-pocket window, SAScore + KS).

Inputs:
  experiments/exp_covft_value/samples_{base,covft,warhead_tokens}.csv

Outputs:
  results/paper_evaluation/exp1plus_substantive_metrics.json
  results/paper_evaluation/exp1plus_substantive_metrics.png
  results/paper_evaluation/exp1plus_substantive_metrics_summary.md
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_covft_value"
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "base": EXP_DIR / "samples_base.csv",
    "covft": EXP_DIR / "samples_covft.csv",
    "warhead_tokens": EXP_DIR / "samples_warhead_tokens.csv",
}

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYLAMIDE_SMARTS = "C=CC(=O)N"

# --- Hinge pharmacophore SMARTS panel (kinase recognition) ---
HINGE_SMARTS = {
    "2_aminopyridine":      "Nc1ccccn1",
    "2_aminopyrimidine":    "Nc1ncccn1",
    "7_azaindole":          "c1ccc2[nH]ccc2n1",
    "pyrrolopyrimidine":    "c1ncc2[nH]ccc2n1",
    "4_aminoquinazoline":   "Nc1ncnc2ccccc12",
    "4_aminoimidazole":     "[nH0]1cnc(N)c1",  # Mol1's hinge
    "2_aminothiazole":      "Nc1nccs1",
}

# --- Clinical covalent kinase inhibitors (canonical SMILES from PubChem) ---
CLINICAL_COVALENT_KINASE = {
    "ibrutinib":      "C=CC(=O)N1CCC[C@H]1C1=NC(c2ccc(Oc3ccccc3)cc2)=C2N1N=CN=C2N",
    "acalabrutinib":  "CC#CC(=O)N1CCC[C@H]1-c1nc(-c2ccc(C(=O)Nc3ccncc3)cc2)c2[nH]cnc2n1",
    "zanubrutinib":   "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "osimertinib":    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "afatinib":       "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1",
    "neratinib":      "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
    "dacomitinib":    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
    "mobocertinib":   "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C(C)C)c3ncccc23)n1",
}

# --- Reference set for FCD (covalent kinase chemistry) ---
COVALENT_FT_TRAIN = PROJECT_ROOT / "data" / "reinvent4_mol2mol_covalent_ft_work" / "covalent_smiles.smi"
KINASE_PAIRS = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"

COVINDB_TRAIN = PROJECT_ROOT / "models" / "train.csv"
ZAP70_SEEDS = PROJECT_ROOT / "data" / "mol1_rl_seeds" / "seed_zap70_acryl_plus_mol1.smi"

# --- ATP-pocket physchem window ---
ATP_WINDOW = {
    "MW":   (350.0, 550.0),
    "logP": (1.5, 4.0),
    "HBA":  (0, 8),
    "HBD":  (0, 3),
    "RotB": (0, 8),
    "Fsp3": (0.25, 1.0),
    "TPSA": (60.0, 120.0),
}

TC_HOP_BAND = (0.3, 0.7)


def now() -> str:
    return time.strftime("%H:%M:%S")


def log_progress(msg: str) -> None:
    print(f"PROGRESS: [{now()}] {msg}", flush=True)


# --------------------------- helpers ---------------------------


def morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def load_smiles_column(csv: Path) -> list[str]:
    df = pd.read_csv(csv)
    col = "Output_SMILES" if "Output_SMILES" in df.columns else "SMILES"
    return df[col].astype(str).tolist()


def parse_mols(smiles: Iterable[str]) -> tuple[list[Chem.Mol], list[str]]:
    mols: list[Chem.Mol] = []
    canonical: list[str] = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        mols.append(m)
        canonical.append(Chem.MolToSmiles(m))
    return mols, canonical


def passes_atp_window(mw, logp, hba, hbd, rotb, fsp3, tpsa) -> bool:
    a, b = ATP_WINDOW["MW"]
    if not (a <= mw <= b):
        return False
    a, b = ATP_WINDOW["logP"]
    if not (a <= logp <= b):
        return False
    if not (hba <= ATP_WINDOW["HBA"][1]):
        return False
    if not (hbd <= ATP_WINDOW["HBD"][1]):
        return False
    if not (rotb <= ATP_WINDOW["RotB"][1]):
        return False
    if fsp3 < ATP_WINDOW["Fsp3"][0]:
        return False
    a, b = ATP_WINDOW["TPSA"]
    if not (a <= tpsa <= b):
        return False
    return True


def shortest_warhead_to_hinge(mol: Chem.Mol, acryl_patt: Chem.Mol,
                              hinge_patts: list[Chem.Mol]) -> int | None:
    """Shortest bond path between warhead C_beta and ANY hinge anchor atom."""
    am = mol.GetSubstructMatches(acryl_patt)
    if not am:
        return None
    # acryl SMARTS "C=CC(=O)N": atoms 0:C_beta 1:C_alpha 2:C=O 3:O 4:N
    c_betas = [match[0] for match in am]
    hinge_anchors: list[int] = []
    for hp in hinge_patts:
        for m in mol.GetSubstructMatches(hp):
            hinge_anchors.extend(m)
    if not hinge_anchors:
        return None
    best = None
    for cb in c_betas:
        for ha in hinge_anchors:
            if cb == ha:
                continue
            path = Chem.GetShortestPath(mol, cb, ha)
            if not path:
                continue
            d = len(path) - 1
            if best is None or d < best:
                best = d
    return best


# --------------------------- reference set for FCD ---------------------------


def build_fcd_reference() -> list[str]:
    """Compose the FCD reference set: kinase actives (>=7 pIC50) UNION CovFT
    training-set covalent kinase smiles, deduplicated.
    """
    covalent: list[str] = []
    if COVALENT_FT_TRAIN.exists():
        with open(COVALENT_FT_TRAIN) as fh:
            for line in fh:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    covalent.append(s)

    kinase_pos: list[str] = []
    if KINASE_PAIRS.exists():
        df = pd.read_csv(KINASE_PAIRS, usecols=["mol_a", "mol_b", "value_a", "value_b"])
        ab = pd.concat([
            df[["mol_a", "value_a"]].rename(columns={"mol_a": "smi", "value_a": "v"}),
            df[["mol_b", "value_b"]].rename(columns={"mol_b": "smi", "value_b": "v"}),
        ])
        ab = ab.dropna()
        ab = ab[ab["v"] >= 7.0]
        kinase_pos = ab["smi"].astype(str).drop_duplicates().tolist()

    canonical = set()
    valid: list[str] = []
    for s in covalent + kinase_pos:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        c = Chem.MolToSmiles(m)
        if c in canonical:
            continue
        canonical.add(c)
        valid.append(c)
    return valid


# --------------------------- main cohort computation ---------------------------


def compute_cohort_metrics(name: str, csv_path: Path,
                           mol1_fp,
                           clinical_fps: dict[str, object],
                           zap70_fps: list[object],
                           train_fps: list[object],
                           train_scaffolds: set[str]) -> dict:
    raw = load_smiles_column(csv_path)
    n_total = len(raw)
    mols, canonical = parse_mols(raw)
    n_valid = len(mols)
    log_progress(f"[{name}] parsed {n_valid}/{n_total} mols")

    acryl_patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    hinge_patts = {k: Chem.MolFromSmarts(s) for k, s in HINGE_SMARTS.items()}
    hinge_patt_list = list(hinge_patts.values())

    # ---- per-mol descriptors / flags ----
    fps = []
    mw_list, logp_list, tpsa_list = [], [], []
    hba_list, hbd_list, rotb_list, fsp3_list = [], [], [], []
    sa_list = []
    scaffolds = []
    has_acryl = []
    has_hinge = []
    hinge_per_class = {k: 0 for k in HINGE_SMARTS}
    pass_atp = []
    tc_mol1 = []
    max_tc_clinical = []  # best across drugs
    max_tc_zap70 = []
    max_tc_train = []
    warhead_hinge_dist = []

    canonical_smiles_list = canonical
    canonical_smiles_set = set()
    n_unique = 0
    for c in canonical_smiles_list:
        if c not in canonical_smiles_set:
            canonical_smiles_set.add(c)
            n_unique += 1

    t0 = time.time()
    last_log = t0

    for idx, mol in enumerate(mols):
        # descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rotb = Lipinski.NumRotatableBonds(mol)
        fsp3 = Lipinski.FractionCSP3(mol)
        mw_list.append(mw)
        logp_list.append(logp)
        tpsa_list.append(tpsa)
        hba_list.append(hba)
        hbd_list.append(hbd)
        rotb_list.append(rotb)
        fsp3_list.append(fsp3)

        # SA
        try:
            sa_list.append(sascorer.calculateScore(mol))
        except Exception:
            sa_list.append(np.nan)

        # scaffold
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        scaffolds.append(scaf)

        # warhead
        has_a = mol.HasSubstructMatch(acryl_patt)
        has_acryl.append(has_a)

        # hinge
        any_hinge = False
        for k, patt in hinge_patts.items():
            if patt is not None and mol.HasSubstructMatch(patt):
                hinge_per_class[k] += 1
                any_hinge = True
        has_hinge.append(any_hinge)

        # ATP window
        pass_atp.append(passes_atp_window(mw, logp, hba, hbd, rotb, fsp3, tpsa))

        # fingerprint -> similarities
        fp = morgan_fp(mol)
        fps.append(fp)
        tc_mol1.append(DataStructs.TanimotoSimilarity(fp, mol1_fp))
        if clinical_fps:
            sims = [DataStructs.TanimotoSimilarity(fp, cfp) for cfp in clinical_fps.values()]
            max_tc_clinical.append(max(sims))
        if zap70_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, zap70_fps)
            max_tc_zap70.append(max(sims))
        if train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_tc_train.append(max(sims))

        # warhead-to-hinge bond path (only if both present)
        if has_a and any_hinge:
            d = shortest_warhead_to_hinge(mol, acryl_patt, hinge_patt_list)
            if d is not None:
                warhead_hinge_dist.append(d)

        if time.time() - last_log > 600:  # every 10 min
            log_progress(f"[{name}] processed {idx + 1}/{n_valid}")
            last_log = time.time()

    log_progress(f"[{name}] descriptors+sims done in {time.time() - t0:.0f}s")

    # ---- aggregate ----
    n = n_valid
    def pct(num: int) -> float:
        return (num / n * 100.0) if n else 0.0

    acryl_pct = pct(sum(has_acryl))
    hinge_pct = pct(sum(has_hinge))
    hinge_class_pct = {k: pct(v) for k, v in hinge_per_class.items()}
    atp_pct = pct(sum(pass_atp))

    # similarity bands to Mol1
    arr_mol1 = np.asarray(tc_mol1)
    band_analog = int((arr_mol1 > 0.7).sum())
    band_hop = int(((arr_mol1 >= 0.3) & (arr_mol1 <= 0.7)).sum())
    band_novel = int((arr_mol1 < 0.3).sum())

    # 4-way conjunction
    four_way = 0
    for i in range(n):
        if (has_acryl[i] and has_hinge[i]
                and TC_HOP_BAND[0] <= tc_mol1[i] <= TC_HOP_BAND[1]
                and pass_atp[i]):
            four_way += 1
    four_way_pct = pct(four_way)

    # clinical max-tc
    arr_clin = np.asarray(max_tc_clinical) if max_tc_clinical else np.array([])
    arr_z = np.asarray(max_tc_zap70) if max_tc_zap70 else np.array([])
    arr_train = np.asarray(max_tc_train) if max_tc_train else np.array([])

    # CovInDB memorization bands
    mem_pct = float((arr_train > 0.8).mean() * 100.0) if arr_train.size else None
    interp_pct = float(((arr_train >= 0.5) & (arr_train <= 0.8)).mean() * 100.0) if arr_train.size else None
    novel_train_pct = float((arr_train < 0.5).mean() * 100.0) if arr_train.size else None

    # scaffold novelty vs CovInDB
    cohort_scafs = set(scaffolds)
    novel_scaf_pct = (
        float(len([s for s in cohort_scafs if s and s not in train_scaffolds]))
        / max(len(cohort_scafs), 1) * 100.0
    )

    return {
        "name": name,
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_pct": pct(n_valid) if n_total == 0 else (n_valid / n_total * 100.0),
        "n_unique_canonical": n_unique,
        "uniqueness_pct": (n_unique / n * 100.0) if n else 0.0,
        # headline trio (rates only here; FCD computed once cohorts loaded)
        "hinge_pharmacophore": {
            "any_hit_pct": hinge_pct,
            "by_class_pct": hinge_class_pct,
        },
        "four_way_conjunction": {
            "count": four_way,
            "pct": four_way_pct,
            "criteria": "acrylamide AND hinge AND 0.3<=Tc(Mol1)<=0.7 AND ATP_window",
        },
        # supporting
        "acrylamide_pct": acryl_pct,
        "mol1_similarity_bands": {
            "n": int(arr_mol1.size),
            "analog_gt_0.7_pct": float(band_analog / n * 100.0) if n else 0,
            "hop_0.3_0.7_pct": float(band_hop / n * 100.0) if n else 0,
            "novel_lt_0.3_pct": float(band_novel / n * 100.0) if n else 0,
            "mean_tc_mol1": float(arr_mol1.mean()) if arr_mol1.size else None,
        },
        "clinical_drug_max_tc": {
            "mean": float(arr_clin.mean()) if arr_clin.size else None,
            "median": float(np.median(arr_clin)) if arr_clin.size else None,
            "p90": float(np.percentile(arr_clin, 90)) if arr_clin.size else None,
            "frac_ge_0.3_pct": float((arr_clin >= 0.3).mean() * 100.0) if arr_clin.size else None,
        },
        "zap70_seed_max_tc": {
            "n_seeds": len(zap70_fps),
            "mean": float(arr_z.mean()) if arr_z.size else None,
            "median": float(np.median(arr_z)) if arr_z.size else None,
            "frac_ge_0.4_pct": float((arr_z >= 0.4).mean() * 100.0) if arr_z.size else None,
        },
        "covindb_memorization": {
            "n_train": len(train_fps),
            "memorized_gt_0.8_pct": mem_pct,
            "interpolated_0.5_0.8_pct": interp_pct,
            "novel_lt_0.5_pct": novel_train_pct,
        },
        "scaffold_novelty_vs_covindb": {
            "n_unique_scaffolds": len(cohort_scafs),
            "novel_scaffold_pct": novel_scaf_pct,
        },
        "warhead_to_hinge_topo": {
            "n": len(warhead_hinge_dist),
            "mean": float(np.mean(warhead_hinge_dist)) if warhead_hinge_dist else None,
            "median": float(np.median(warhead_hinge_dist)) if warhead_hinge_dist else None,
            "frac_in_6_10": float(np.mean([(6 <= d <= 10) for d in warhead_hinge_dist]) * 100.0)
                if warhead_hinge_dist else None,
        },
        "atp_pocket_window_pct": atp_pct,
        "physchem_summary": {
            "MW_mean": float(np.mean(mw_list)) if mw_list else None,
            "logP_mean": float(np.mean(logp_list)) if logp_list else None,
            "TPSA_mean": float(np.mean(tpsa_list)) if tpsa_list else None,
            "Fsp3_mean": float(np.mean(fsp3_list)) if fsp3_list else None,
        },
        "sa_score": {
            "mean": float(np.nanmean(sa_list)) if sa_list else None,
            "median": float(np.nanmedian(sa_list)) if sa_list else None,
            # KS computed at the end with the training distribution
            "values": [float(x) for x in sa_list if not np.isnan(x)],
        },
        # raw arrays kept temporarily to compute FCD efficiently
        "_canonical_unique": list(canonical_smiles_set),
    }


# --------------------------- FCD ---------------------------


def compute_fcd(cohort_smiles: list[str], reference_smiles: list[str]) -> dict:
    try:
        from fcd_torch import FCD
        fcd = FCD(device="cpu", n_jobs=4)
        # cap cohort at ~5000 to fit time budget (CPU)
        if len(cohort_smiles) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cohort_smiles), size=5000, replace=False)
            cohort_smiles = [cohort_smiles[i] for i in idx]
        if len(reference_smiles) > 5000:
            rng = np.random.default_rng(43)
            idx = rng.choice(len(reference_smiles), size=5000, replace=False)
            reference_smiles = [reference_smiles[i] for i in idx]
        val = float(fcd(cohort_smiles, reference_smiles))
        return {"value": val, "fallback": False, "n_cohort": len(cohort_smiles),
                "n_reference": len(reference_smiles)}
    except Exception as e:
        return {"value": None, "fallback": False, "error": str(e)}


def fallback_distribution_distance(cohort_fps: list, ref_fps: list) -> float:
    """Wasserstein on top-100 Morgan-FP-bit frequency vectors."""
    def freq(fps):
        arr = np.zeros(2048, dtype=np.float64)
        for fp in fps:
            on_bits = list(fp.GetOnBits())
            for b in on_bits:
                arr[b] += 1
        arr /= max(len(fps), 1)
        return arr
    a = freq(cohort_fps)
    b = freq(ref_fps)
    top_idx = np.argsort(-(a + b))[:100]
    return float(scipy_stats.wasserstein_distance(a[top_idx], b[top_idx]))


# --------------------------- training set helpers ---------------------------


def load_covindb_train() -> tuple[list, set[str]]:
    """Load CovInDB v2 unique training SMILES → fps, scaffolds."""
    df = pd.read_csv(COVINDB_TRAIN)
    src = df["Source_Mol"].dropna().astype(str).tolist()
    tgt = df["Target_Mol"].dropna().astype(str).tolist()
    canonical = set()
    mols = []
    scafs = set()
    for s in src + tgt:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        c = Chem.MolToSmiles(m)
        if c in canonical:
            continue
        canonical.add(c)
        mols.append(m)
        try:
            scafs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
        except Exception:
            pass
    # FPs (limit to avoid OOM in BulkTanimoto -- 8K x 30K is fine)
    fps = [morgan_fp(m) for m in mols]
    return fps, scafs


def load_zap70_seeds() -> list:
    if not ZAP70_SEEDS.exists():
        return []
    fps = []
    with open(ZAP70_SEEDS) as fh:
        for line in fh:
            s = line.strip().split()[0] if line.strip() else ""
            if not s:
                continue
            m = Chem.MolFromSmiles(s)
            if m is not None and m.GetNumAtoms() > 0:
                fps.append(morgan_fp(m))
    return fps


def load_clinical_drug_fps() -> dict:
    out = {}
    for name, smi in CLINICAL_COVALENT_KINASE.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"WARN: clinical drug {name} SMILES failed to parse — skipping")
            continue
        out[name] = morgan_fp(m)
    return out


# --------------------------- plotting + summary ---------------------------


def make_headline_plot(results: dict, fcd_per_cohort: dict, out_path: Path) -> None:
    cohorts = list(results.keys())
    fcds = [fcd_per_cohort[c].get("value") for c in cohorts]
    # invert FCD for "higher = better" visualization, normalised against max
    valid_fcds = [v for v in fcds if v is not None]
    fcd_inv = [None if v is None else (max(valid_fcds) - v) for v in fcds]
    hinge = [results[c]["hinge_pharmacophore"]["any_hit_pct"] for c in cohorts]
    four_way = [results[c]["four_way_conjunction"]["pct"] for c in cohorts]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#888888", "#1f77b4", "#d62728"]

    # FCD raw (lower = better)
    ax = axes[0]
    vals = [v if v is not None else 0 for v in fcds]
    ax.bar(cohorts, vals, color=colors)
    ax.set_title("FCD vs covalent-kinase ref (lower=better)")
    ax.set_ylabel("FCD")
    for i, v in enumerate(fcds):
        ax.text(i, vals[i] + 0.5, f"{v:.2f}" if v is not None else "n/a",
                ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(cohorts, hinge, color=colors)
    ax.set_title("Hinge pharmacophore hit rate")
    ax.set_ylabel("% of valid mols")
    for i, v in enumerate(hinge):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[2]
    ax.bar(cohorts, four_way, color=colors)
    ax.set_title("Four-way conjunction (HEADLINE)")
    ax.set_ylabel("% of valid mols")
    for i, v in enumerate(four_way):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)

    fig.suptitle("Exp1+ Substantive metric panel: headline trio per cohort (10k mols)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_summary_md(results: dict, fcd_per_cohort: dict,
                    fcd_meta: dict,
                    sa_ks: dict,
                    out_path: Path) -> None:
    cohorts = list(results.keys())

    def g(c, *keys):
        x = results[c]
        for k in keys:
            x = x[k]
        return x

    headline_lines = [
        "# Exp1+ Substantive metric panel — 3 cohorts of 10k mols",
        "",
        "## HEADLINE: Four-way conjunction (acrylamide AND hinge AND Tc-hop AND ATP-window)",
        "",
        "| Cohort | 4-way conjunction | acrylamide | hinge any | Tc 0.3-0.7 to Mol1 | ATP window |",
        "|---|---|---|---|---|---|",
    ]
    for c in cohorts:
        headline_lines.append(
            f"| {c} | **{g(c, 'four_way_conjunction', 'pct'):.2f}%** "
            f"| {g(c, 'acrylamide_pct'):.1f}% "
            f"| {g(c, 'hinge_pharmacophore', 'any_hit_pct'):.1f}% "
            f"| {g(c, 'mol1_similarity_bands', 'hop_0.3_0.7_pct'):.1f}% "
            f"| {g(c, 'atp_pocket_window_pct'):.1f}% |"
        )

    headline_lines += [
        "",
        "## FCD vs covalent-kinase reference (lower = better)",
        f"_Reference set: {fcd_meta['n_reference_raw']} unique covalent-kinase mols "
        f"(covalent FT corpus + ChEMBL kinase pIC50≥7 within-pairs)._",
        "",
        "| Cohort | FCD | Fallback? |",
        "|---|---|---|",
    ]
    for c in cohorts:
        f = fcd_per_cohort[c]
        v = f.get("value")
        v_str = f"{v:.2f}" if v is not None else "ERROR"
        fb = "yes" if f.get("fallback") else "no"
        headline_lines.append(f"| {c} | {v_str} | {fb} |")

    headline_lines += [
        "",
        "## Hinge pharmacophore — by class (% of valid)",
        "",
        "| Cohort | 2-amPy | 2-amPm | 7-aza-Ind | PyrPm | 4-amQz | 4-amImid (Mol1) | 2-amThz |",
        "|---|---|---|---|---|---|---|---|",
    ]
    keys = list(HINGE_SMARTS.keys())
    for c in cohorts:
        cls = results[c]["hinge_pharmacophore"]["by_class_pct"]
        row = " | ".join(f"{cls[k]:.1f}%" for k in keys)
        headline_lines.append(f"| {c} | {row} |")

    headline_lines += [
        "",
        "## Mol1-similarity band distribution",
        "",
        "| Cohort | analog (>0.7) | hop (0.3-0.7) | novel (<0.3) | mean Tc |",
        "|---|---|---|---|---|",
    ]
    for c in cohorts:
        b = results[c]["mol1_similarity_bands"]
        headline_lines.append(
            f"| {c} | {b['analog_gt_0.7_pct']:.1f}% | {b['hop_0.3_0.7_pct']:.1f}% | "
            f"{b['novel_lt_0.3_pct']:.1f}% | {b['mean_tc_mol1']:.3f} |"
        )

    headline_lines += [
        "",
        "## Clinical covalent kinase drug max-Tc",
        "",
        "| Cohort | mean | median | p90 | frac ≥0.3 |",
        "|---|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["clinical_drug_max_tc"]
        headline_lines.append(
            f"| {c} | {d['mean']:.3f} | {d['median']:.3f} | {d['p90']:.3f} | {d['frac_ge_0.3_pct']:.1f}% |"
        )

    headline_lines += [
        "",
        f"## ZAP70 acrylamide seed max-Tc (n_seeds={results[cohorts[0]]['zap70_seed_max_tc']['n_seeds']})",
        "",
        "| Cohort | mean | median | frac ≥0.4 |",
        "|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["zap70_seed_max_tc"]
        if d["mean"] is None:
            headline_lines.append(f"| {c} | n/a | n/a | n/a |")
            continue
        headline_lines.append(
            f"| {c} | {d['mean']:.3f} | {d['median']:.3f} | {d['frac_ge_0.4_pct']:.1f}% |"
        )

    headline_lines += [
        "",
        "## CovInDB v2 training-memorization bands",
        "",
        "| Cohort | memorized (>0.8) | interpolated (0.5-0.8) | novel (<0.5) |",
        "|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["covindb_memorization"]
        if d["memorized_gt_0.8_pct"] is None:
            headline_lines.append(f"| {c} | n/a | n/a | n/a |")
            continue
        headline_lines.append(
            f"| {c} | {d['memorized_gt_0.8_pct']:.1f}% | "
            f"{d['interpolated_0.5_0.8_pct']:.1f}% | {d['novel_lt_0.5_pct']:.1f}% |"
        )

    headline_lines += [
        "",
        "## Scaffold novelty vs CovInDB v2 (unique scaffolds)",
        "",
        "| Cohort | n_unique_scaffolds | novel_scaffold_pct |",
        "|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["scaffold_novelty_vs_covindb"]
        headline_lines.append(
            f"| {c} | {d['n_unique_scaffolds']} | {d['novel_scaffold_pct']:.1f}% |"
        )

    headline_lines += [
        "",
        "## Warhead-to-hinge topological bond distance",
        "(Clinical covalent kinase drugs cluster at 6-10 bonds.)",
        "",
        "| Cohort | n_with_both | mean | median | frac in [6,10] |",
        "|---|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["warhead_to_hinge_topo"]
        if d["n"] == 0 or d["mean"] is None:
            headline_lines.append(f"| {c} | 0 | n/a | n/a | n/a |")
            continue
        headline_lines.append(
            f"| {c} | {d['n']} | {d['mean']:.2f} | {d['median']:.1f} | {d['frac_in_6_10']:.1f}% |"
        )

    headline_lines += [
        "",
        "## SAScore + KS-stat vs CovInDB v2 training set",
        "",
        "| Cohort | mean SA | median SA | KS-stat (vs CovInDB) | KS p-value |",
        "|---|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["sa_score"]
        ks = sa_ks.get(c, {})
        headline_lines.append(
            f"| {c} | {d['mean']:.2f} | {d['median']:.2f} | "
            f"{ks.get('stat', float('nan')):.3f} | {ks.get('pval', float('nan')):.2e} |"
        )

    headline_lines += [
        "",
        "## Physchem (means)",
        "",
        "| Cohort | MW | logP | TPSA | Fsp3 |",
        "|---|---|---|---|---|",
    ]
    for c in cohorts:
        d = results[c]["physchem_summary"]
        headline_lines.append(
            f"| {c} | {d['MW_mean']:.0f} | {d['logP_mean']:.2f} | "
            f"{d['TPSA_mean']:.0f} | {d['Fsp3_mean']:.2f} |"
        )

    # ---- interpretation ----
    fw = {c: g(c, 'four_way_conjunction', 'pct') for c in cohorts}
    fc = {c: fcd_per_cohort[c].get('value') for c in cohorts}
    hp = {c: g(c, 'hinge_pharmacophore', 'any_hit_pct') for c in cohorts}
    ac = {c: g(c, 'acrylamide_pct') for c in cohorts}

    interp = (
        "## Interpretation\n\n"
        f"**Headline (4-way conjunction):** base {fw.get('base', 0):.2f}% → "
        f"covft {fw.get('covft', 0):.2f}% → warhead_tokens {fw.get('warhead_tokens', 0):.2f}%. "
        f"FCD: base {fc.get('base')}, covft {fc.get('covft')}, "
        f"warhead_tokens {fc.get('warhead_tokens')} (lower = closer to covalent-kinase chemistry). "
        f"Hinge pharmacophore presence: base {hp.get('base'):.1f}%, covft {hp.get('covft'):.1f}%, "
        f"warhead_tokens {hp.get('warhead_tokens'):.1f}%. "
        f"Acrylamide warhead retention: base {ac.get('base'):.1f}%, covft {ac.get('covft'):.1f}%, "
        f"warhead_tokens {ac.get('warhead_tokens'):.1f}%.\n\n"
    )

    # narrative
    base_low_cov = (
        fw.get('covft', 0) > fw.get('base', 0) + 2
        or fw.get('warhead_tokens', 0) > fw.get('base', 0) + 2
    )
    if base_low_cov:
        interp += (
            "Covalent fine-tuning and warhead-token conditioning materially raise the headline "
            "4-way kinase covalent-inhibitor rate above the vanilla mol2mol prior, which "
            "produces almost no kinase covalent chemistry. This is the value statement.\n"
        )
    else:
        interp += (
            "Covalent fine-tuning did NOT meaningfully raise the 4-way conjunction rate over the "
            "vanilla prior. This is counter-narrative — check whether the input prompt (Mol1) "
            "is dominating sampling rather than the fine-tune.\n"
        )

    if hp.get('covft', 0) < hp.get('warhead_tokens', 0) - 5:
        interp += (
            "\nFLAG: warhead_tokens shows stronger hinge pharmacophore presence than covft. "
            "If sampling temperature/parents are matched, this suggests covft transfer-learning "
            "is diluting the kinase recognition motifs that warhead_tokens preserves.\n"
        )
    headline_lines.append("")
    headline_lines.append(interp)

    out_path.write_text("\n".join(headline_lines))


# --------------------------- driver ---------------------------


def main() -> None:
    log_progress("loading reference Mol1 + clinical drugs + ZAP70 seeds")
    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    assert mol1 is not None
    mol1_fp = morgan_fp(mol1)
    clinical_fps = load_clinical_drug_fps()
    log_progress(f"clinical covalent kinase drugs: {len(clinical_fps)}")

    zap70_fps = load_zap70_seeds()
    log_progress(f"ZAP70 acrylamide seeds: {len(zap70_fps)}")

    log_progress("loading CovInDB v2 training set...")
    train_fps, train_scafs = load_covindb_train()
    log_progress(f"CovInDB v2 unique mols: {len(train_fps)}; scaffolds: {len(train_scafs)}")

    log_progress("building FCD reference set (covalent FT corpus + ChEMBL kinase pIC50≥7)...")
    fcd_reference = build_fcd_reference()
    log_progress(f"FCD reference size (unique): {len(fcd_reference)}")

    # ---- training SA distribution for KS ----
    log_progress("computing training-set SAScore distribution (sample)")
    rng = np.random.default_rng(0)
    if len(train_fps) > 2000:
        sub_idx = rng.choice(len(train_fps), size=2000, replace=False)
    else:
        sub_idx = np.arange(len(train_fps))
    train_sa = []
    train_smiles_for_sa: list[str] = []
    # We need mol objects to compute SA — re-parse a subsample from the CSV
    df_t = pd.read_csv(COVINDB_TRAIN)
    src = df_t["Source_Mol"].dropna().astype(str).tolist()
    tgt = df_t["Target_Mol"].dropna().astype(str).tolist()
    combined = list(set(src + tgt))
    rng.shuffle(combined)
    for s in combined[:3000]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            train_sa.append(sascorer.calculateScore(m))
        except Exception:
            continue
        if len(train_sa) >= 2000:
            break
    train_sa = np.asarray(train_sa)
    log_progress(f"training SA samples: {train_sa.size}")

    # ---- per-cohort metrics ----
    results: dict[str, dict] = {}
    for name, csv in COHORTS.items():
        log_progress(f"start cohort {name}")
        results[name] = compute_cohort_metrics(
            name, csv, mol1_fp, clinical_fps, zap70_fps, train_fps, train_scafs
        )
        log_progress(f"done cohort {name}")

    # ---- FCD per cohort ----
    fcd_per_cohort: dict[str, dict] = {}
    fcd_meta = {"n_reference_raw": len(fcd_reference)}
    for name in COHORTS:
        cohort_unique = results[name].get("_canonical_unique", [])
        log_progress(f"computing FCD for {name} (n_cohort={len(cohort_unique)})")
        out = compute_fcd(cohort_unique, fcd_reference)
        fcd_per_cohort[name] = out
        log_progress(f"FCD[{name}] = {out.get('value')}")

    # ---- SA KS vs training ----
    sa_ks = {}
    for name in COHORTS:
        sa_vals = np.asarray(results[name]["sa_score"]["values"])
        if sa_vals.size == 0 or train_sa.size == 0:
            sa_ks[name] = {"stat": float("nan"), "pval": float("nan")}
            continue
        if sa_vals.size > 5000:
            idx = rng.choice(sa_vals.size, size=5000, replace=False)
            sa_vals = sa_vals[idx]
        s, p = scipy_stats.ks_2samp(sa_vals, train_sa)
        sa_ks[name] = {"stat": float(s), "pval": float(p)}
        # don't ship raw values to JSON
        results[name]["sa_score"].pop("values", None)

    # ---- strip large internals ----
    for name in COHORTS:
        results[name].pop("_canonical_unique", None)

    # ---- write outputs ----
    out_json = OUT_DIR / "exp1plus_substantive_metrics.json"
    out_png = OUT_DIR / "exp1plus_substantive_metrics.png"
    out_md = OUT_DIR / "exp1plus_substantive_metrics_summary.md"

    payload = {
        "results": results,
        "fcd": fcd_per_cohort,
        "fcd_meta": fcd_meta,
        "sa_ks_vs_covindb": sa_ks,
        "notes": {
            "hinge_smarts": HINGE_SMARTS,
            "atp_window": ATP_WINDOW,
            "tc_hop_band": TC_HOP_BAND,
            "clinical_drugs": list(CLINICAL_COVALENT_KINASE.keys()),
            "schwobel_gsh_model": "deferred to later run (not installable in 5 min)",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    make_headline_plot(results, fcd_per_cohort, out_png)
    make_summary_md(results, fcd_per_cohort, fcd_meta, sa_ks, out_md)

    log_progress(f"wrote {out_json}")
    log_progress(f"wrote {out_png}")
    log_progress(f"wrote {out_md}")

    print("\n=== SHORT REPORT ===")
    for c in COHORTS:
        r = results[c]
        f = fcd_per_cohort[c].get("value")
        print(f"  {c}: 4-way={r['four_way_conjunction']['pct']:.2f}%  "
              f"FCD={f:.2f}  hinge={r['hinge_pharmacophore']['any_hit_pct']:.1f}%  "
              f"acryl={r['acrylamide_pct']:.1f}%")


if __name__ == "__main__":
    main()
