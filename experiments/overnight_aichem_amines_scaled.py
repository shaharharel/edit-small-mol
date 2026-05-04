#!/usr/bin/env python3
"""
ai-chem scaled-up amine pipeline (overnight).

Builds a massive amine library by combining:
  (a) Existing 1,326 amines from initial mine (cached)
  (b) ChEMBL full chemreps download → mine for primary/secondary amines, MW ≤ 500
  (c) Hand-curated set (already in script)

Then couples all amines with frag A (acrylamide-isoindoline-COOH), filters for
warhead intact, and scores with FiLMDelta single-seed + RDKit descriptors.
Top-50K saved.

Designed for 16-CPU n2-standard-16. Parallelizes via multiprocessing.

Usage (on ai-chem):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aichem_amines_scaled.py
"""

import sys
import os
import gc
import json
import gzip
import urllib.request
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, DataStructs
RDLogger.DisableLog('rdApp.*')

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "aichem_tier2_scaled"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz"
CHEMBL_FILE = DATA_DIR / "chembl_35_chemreps.txt.gz"

FRAG_A_ACID = "C=CC(=O)N1Cc2cccc(C(=O)O)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"
ACRYL_PAT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
WARHEAD_PAT = Chem.MolFromSmarts(WARHEAD_SMARTS)

_AMIDE_RXN = AllChem.ReactionFromSmarts(
    "[C:1](=[O:2])[O;H1].[N;H1,H2;!$(N=*);!$(N#*);!$(N*=O):3] >> [C:1](=[O:2])[N:3]"
)

N_PROCS = max(1, mp.cpu_count() - 2)
print(f"Workers: {N_PROCS}")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Utility: amine detection ──────────────────────────────────────────────────

def has_amine_for_coupling(mol):
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N':
            continue
        if atom.GetIsAromatic() or atom.GetFormalCharge() != 0:
            continue
        is_amide = False
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == 'C':
                for nbb in nb.GetBonds():
                    other = nbb.GetOtherAtom(nb)
                    if other.GetSymbol() == 'O' and nbb.GetBondType() == Chem.BondType.DOUBLE:
                        is_amide = True
        if is_amide:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        return True
    return False


# ── Step 1: Download ChEMBL ───────────────────────────────────────────────────

def download_chembl():
    if CHEMBL_FILE.exists() and CHEMBL_FILE.stat().st_size > 200_000_000:
        log(f"  ChEMBL chemreps already downloaded: {CHEMBL_FILE} ({CHEMBL_FILE.stat().st_size / 1024 / 1024:.0f} MB)")
        return CHEMBL_FILE
    log(f"  Downloading ChEMBL 35 chemreps from EBI: {CHEMBL_URL}")
    log(f"  → {CHEMBL_FILE}")
    urllib.request.urlretrieve(CHEMBL_URL, CHEMBL_FILE)
    log(f"  Downloaded {CHEMBL_FILE.stat().st_size / 1024 / 1024:.0f} MB")
    return CHEMBL_FILE


# ── Step 2: Mine ChEMBL for amines ────────────────────────────────────────────

def _check_amine_worker(args):
    """Worker: given a SMILES, return (smiles, MW, source) if it's a viable amine candidate."""
    smi, mw_max = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        if mw > mw_max or mw < 60:
            return None
        if not has_amine_for_coupling(mol):
            return None
        if mol.HasSubstructMatch(ACRYL_PAT):
            return None
        # Drug-likeness
        if Descriptors.NumRotatableBonds(mol) > 12:
            return None
        return (Chem.MolToSmiles(mol), float(mw), "ChEMBL")
    except Exception:
        return None


def mine_chembl_full(chembl_file, mw_max=500, max_amines=1_000_000):
    log(f"  Reading ChEMBL chemreps...")
    smiles = []
    with gzip.open(chembl_file, 'rt') as f:
        header = f.readline()  # skip header line
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                smiles.append(parts[1])  # canonical_smiles column
    log(f"  Read {len(smiles):,} ChEMBL SMILES")

    # Pre-filter by string heuristics: must contain N and not too long
    pre_filtered = [s for s in smiles if 'N' in s and len(s) < 250]
    log(f"  Pre-filtered (contains N, len<250): {len(pre_filtered):,}")

    # Parallel amine check
    args = [(s, mw_max) for s in pre_filtered]
    log(f"  Checking amines on {N_PROCS} workers (mw≤{mw_max})...")

    seen = set()
    amines = []
    n_processed = 0
    n_log_step = 200_000
    last_log = 0

    with mp.Pool(N_PROCS) as pool:
        for r in pool.imap_unordered(_check_amine_worker, args, chunksize=500):
            n_processed += 1
            if r is not None:
                smi, mw, source = r
                if smi not in seen:
                    seen.add(smi)
                    amines.append({"smiles": smi, "MW": mw, "source": source})
                    if len(amines) >= max_amines:
                        break
            if n_processed - last_log >= n_log_step:
                last_log = n_processed
                log(f"    processed {n_processed:,}/{len(args):,}, amines so far: {len(amines):,}")

    log(f"  Mined {len(amines):,} unique amines (MW≤{mw_max})")
    return amines


# ── Step 3: Hand-curated set (always include) ──────────────────────────────────

HAND_AMINES = [
    "Nc1cn(C(C)C)cn1", "Nc1cncn1C", "Nc1cn(CC)cn1", "Nc1cn(C2CC2)cn1",
    "Nc1cn(C(C)(C)C)cn1", "Nc1ccn(C(C)C)n1", "Nc1cnn(C)c1", "Nc1ccnn1C(C)C",
    "Nc1nn(C(C)C)cn1", "Nc1cn(C(C)C)nn1", "Nc1cnoc1", "Nc1cnsc1",
    "Nc1ncc(C)s1", "Nc1ccnc(N)n1", "Nc1ncccn1", "Nc1ccncn1",
    "Nc1ccncc1", "Nc1cccnc1", "Nc1ccccn1",
    "Nc1ccccc1", "Nc1ccc(F)cc1", "Nc1ccc(Cl)cc1", "Nc1ccc(OC)cc1",
    "Nc1cccc(C)c1", "Nc1cc(C)cc(C)c1",
    "NCc1ccccc1", "NCc1cnc(C(C)C)cn1",
    "NC1CCCCC1", "NCC1CCCO1", "NC1CCNC1", "NCCN1CCCC1",
    "Nc1cc2ccccc2[nH]1", "Nc1[nH]c2ccccc2n1", "Nc1nc2ccccc2[nH]1",
    "Nc1ccnc(N2CCOCC2)n1", "Nc1ncn(C2CCOCC2)c1", "Nc1cnc(N2CCN(C)CC2)nc1",
    "Nc1cccc(N2CCOCC2)c1", "Nc1cccc(F)c1F", "Nc1ccc(C(F)(F)F)cc1",
    "Nc1ccc(C#N)cc1", "Nc1ccc(N(C)C)cc1",
    "NCC(F)F", "NCC(F)(F)F", "NCC1CCC1", "NCC1CC1",
    "Nc1ccc2[nH]ncc2c1", "Nc1ccc2[nH]ccc2c1", "Nc1cnc2[nH]ccc2c1",
    "Nc1ccnc(C)c1", "Nc1ccnc(F)c1", "Nc1ccnc(Cl)c1", "Nc1ccnc(OC)c1",
    "Nc1ccnc2ccccc12", "Nc1ncnc2ccccc12",
]


# ── Step 4: Couple ────────────────────────────────────────────────────────────

def _couple_one(args):
    acid_smi, amine_smi = args
    acid = Chem.MolFromSmiles(acid_smi)
    amine = Chem.MolFromSmiles(amine_smi)
    if acid is None or amine is None:
        return None
    try:
        prods = _AMIDE_RXN.RunReactants((acid, amine))
    except Exception:
        return None
    if not prods:
        return None
    for prod_set in prods:
        if not prod_set:
            continue
        p = prod_set[0]
        try:
            Chem.SanitizeMol(p)
            s = Chem.MolToSmiles(p)
            mol_p = Chem.MolFromSmiles(s)
            if mol_p is None or not mol_p.HasSubstructMatch(WARHEAD_PAT):
                return None
            return s
        except Exception:
            continue
    return None


def couple_all(amines, frag_a_acid, max_products=2_000_000):
    log(f"  Coupling {len(amines):,} amines with frag A...")
    args = [(frag_a_acid, a["smiles"]) for a in amines]
    seen = set()
    products = []
    n_log_step = 100_000

    with mp.Pool(N_PROCS) as pool:
        for i, prod in enumerate(pool.imap_unordered(_couple_one, args, chunksize=500)):
            if prod is None:
                continue
            if prod in seen:
                continue
            seen.add(prod)
            products.append({"smiles": prod})
            if len(products) >= max_products:
                break
            if (i + 1) % n_log_step == 0:
                log(f"    {i+1:,}/{len(args):,} processed, unique products: {len(products):,}")

    log(f"  Final unique coupled products: {len(products):,}")
    return products


# ── Step 5: Score descriptors + FiLMDelta ─────────────────────────────────────

def _score_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return {
            "smiles": smi,
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "QED": Chem.QED.qed(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "Rings": Descriptors.RingCount(mol),
        }
    except Exception:
        return None


def _fp_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def score_pipeline(products, model_path):
    log(f"  Scoring {len(products):,} products...")

    # Step 5a: descriptors (parallel)
    log(f"    Computing descriptors...")
    smiles = [p["smiles"] for p in products]
    with mp.Pool(N_PROCS) as pool:
        out_desc = []
        for i, r in enumerate(pool.imap_unordered(_score_descriptors, smiles, chunksize=500)):
            if r is not None:
                out_desc.append(r)
            if (i + 1) % 200_000 == 0:
                log(f"      descriptors {i+1:,}/{len(smiles):,}")
    df_desc = pd.DataFrame(out_desc)
    log(f"    Descriptors done: {len(df_desc):,}")

    # Step 5b: FiLMDelta scoring
    if not model_path.exists():
        log(f"    WARN: no FiLMDelta checkpoint at {model_path}; skipping")
        df_desc["pIC50_film"] = np.nan
        return df_desc

    log(f"    FiLMDelta scoring (single-seed)...")
    import torch
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    anchor_embs = ckpt["anchor_embs"]
    anchor_pIC50 = np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)
    n_anchors = anchor_embs.shape[0]
    log(f"      Loaded model: {n_anchors} anchors")

    # Compute FPs in parallel
    log(f"      Computing FPs...")
    desc_smiles = df_desc["smiles"].tolist()
    with mp.Pool(N_PROCS) as pool:
        fps = pool.map(_fp_one, desc_smiles, chunksize=500)
    valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
    log(f"      Valid FPs: {len(valid):,}")

    scores = np.full(len(desc_smiles), np.nan)
    bs = 512
    log(f"      Scoring (bs={bs})...")
    valid_arr = np.array([v[1] for v in valid], dtype=np.float32)
    valid_idx = [v[0] for v in valid]
    with torch.no_grad():
        for start in range(0, len(valid_arr), bs):
            chunk = valid_arr[start:start+bs]
            embs = torch.FloatTensor(scaler.transform(chunk))
            for j in range(len(chunk)):
                tgt = embs[j:j+1].expand(n_anchors, -1)
                deltas = model(anchor_embs, tgt).numpy().flatten()
                scores[valid_idx[start + j]] = float(np.mean(anchor_pIC50 + deltas))
            if (start + bs) % 50000 == 0:
                log(f"        scored {start+bs:,}/{len(valid_arr):,}")
    df_desc["pIC50_film"] = scores
    log(f"    FiLMDelta scoring done")
    return df_desc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("AI-CHEM SCALED TIER 2 ENRICHMENT")
    log("=" * 70)
    log(f"Started: {datetime.now().isoformat()}")

    # Step 1: download ChEMBL
    log("\n--- Step 1: Download ChEMBL chemreps ---")
    chembl_file = download_chembl()

    # Step 2: mine amines
    log("\n--- Step 2: Mine ChEMBL for amines ---")
    chembl_amines = mine_chembl_full(chembl_file, mw_max=500, max_amines=1_000_000)

    # Step 3: combine with hand-curated + cached
    log("\n--- Step 3: Build full amine library ---")
    seen = set()
    all_amines = []

    # Hand-curated first
    for s in HAND_AMINES:
        m = Chem.MolFromSmiles(s)
        if m is not None and s not in seen:
            seen.add(s)
            all_amines.append({"smiles": s, "MW": Descriptors.MolWt(m), "source": "hand_curated"})
    log(f"  Hand-curated: {len(all_amines)}")

    # ChEMBL
    for a in chembl_amines:
        if a["smiles"] not in seen:
            seen.add(a["smiles"])
            all_amines.append(a)
    log(f"  Total amine library: {len(all_amines):,}")

    # Save
    pd.DataFrame(all_amines).to_csv(RESULTS_DIR / "amine_library.csv", index=False)

    # Step 4: couple
    log("\n--- Step 4: Couple all amines with frag A ---")
    products = couple_all(all_amines, FRAG_A_ACID, max_products=2_000_000)
    pd.DataFrame(products).to_csv(RESULTS_DIR / "coupled_products_smiles.csv", index=False)

    # Step 5: score
    log("\n--- Step 5: Score products ---")
    model_path = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    df_scored = score_pipeline(products, model_path)
    df_scored = df_scored.sort_values("pIC50_film", ascending=False, na_position='last')
    df_scored.to_csv(RESULTS_DIR / "products_scored.csv", index=False)

    # Save top 50K
    top = df_scored.head(50_000)
    top.to_csv(RESULTS_DIR / "products_top50k.csv", index=False)

    # Summary
    summary = {
        "n_amines": len(all_amines),
        "n_hand_amines": len([a for a in all_amines if a.get("source") == "hand_curated"]),
        "n_chembl_amines": len([a for a in all_amines if a.get("source") == "ChEMBL"]),
        "n_products": len(products),
        "n_scored": int(df_scored["pIC50_film"].notna().sum()),
        "pIC50_max": float(df_scored["pIC50_film"].max()),
        "pIC50_median": float(df_scored["pIC50_film"].median()),
        "n_potent_7": int((df_scored["pIC50_film"] >= 7.0).sum()),
        "n_potent_8": int((df_scored["pIC50_film"] >= 8.0).sum()),
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n{json.dumps(summary, indent=2)}")
    log(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
