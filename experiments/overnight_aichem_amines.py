#!/usr/bin/env python3
"""
ai-chem overnight pipeline: Tier 2 enrichment to ~1M amines.

Steps:
  1. Mine ChEMBL kinase ligands deep — extract primary/secondary amines via BRICS
     decomposition + whole-mol filter. No MW cap; relaxed criteria.
  2. (Skipped — Enamine BB download often blocked from compute VMs; we'll do that
      from local if needed.)
  3. Couple all amines with frag A (acrylamide-isoindoline-COOH) → products.
  4. Score products with FiLMDelta (single-seed, fast batch on CPU) +
     descriptors. NO 3D scoring on full set.
  5. Save top-50K product CSV for downstream 3D scoring on local/ai-gpu.

Designed for 16-CPU n2-standard-16 box. Parallelizes BRICS decomposition + scoring
via multiprocessing.

Usage (on ai-chem):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aichem_amines.py
"""

import sys
import os
import gc
import json
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

KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
ZAP70_CSV = PROJECT_ROOT / "data" / "molecule_pIC50_minimal.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "aichem_tier2_enrichment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FRAG_A_ACID = "C=CC(=O)N1Cc2cccc(C(=O)O)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"
ACRYL_PAT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
WARHEAD_PAT = Chem.MolFromSmarts(WARHEAD_SMARTS)

_AMIDE_RXN = AllChem.ReactionFromSmarts(
    "[C:1](=[O:2])[O;H1].[N;H1,H2;!$(N=*);!$(N#*);!$(N*=O):3] >> [C:1](=[O:2])[N:3]"
)

N_PROCS = max(1, mp.cpu_count() - 2)
print(f"Workers: {N_PROCS}")


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


# ── Step 1: Mine ChEMBL deep ──────────────────────────────────────────────────

def _process_one_molecule_for_amines(args):
    """Worker: take a SMILES, return (whole-mol amine if applicable, BRICS amine fragments)."""
    smi, mw_max_whole, mw_max_brics = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [], []
    except Exception:
        return [], []

    whole, frags = [], []
    mw = Descriptors.MolWt(mol)

    # Whole-mol amine (small enough)
    if mw <= mw_max_whole:
        if has_amine_for_coupling(mol) and not mol.HasSubstructMatch(ACRYL_PAT):
            whole.append((Chem.MolToSmiles(mol), mw, "whole"))

    # BRICS decomposition (only for larger mols)
    if mw >= 200:
        try:
            decomp = BRICS.BRICSDecompose(mol, returnMols=False)
            for f in decomp:
                cleaned = Chem.MolFromSmiles(f)
                if cleaned is None:
                    continue
                cleaned = Chem.DeleteSubstructs(
                    cleaned, Chem.MolFromSmarts("[*;!#1;!#6;!#7;!#8;!#9;!#16;!#17;!#35]")
                )
                try:
                    Chem.SanitizeMol(cleaned)
                except Exception:
                    continue
                fmw = Descriptors.MolWt(cleaned)
                if fmw < 60 or fmw > mw_max_brics:
                    continue
                if not has_amine_for_coupling(cleaned):
                    continue
                if cleaned.HasSubstructMatch(ACRYL_PAT):
                    continue
                frags.append((Chem.MolToSmiles(cleaned), fmw, "brics"))
        except Exception:
            pass

    return whole, frags


def mine_chembl_amines(mw_whole=350, mw_brics=300, max_amines=2_000_000):
    print("\n=== STEP 1: ChEMBL deep miner ===")
    sources = []

    # Kinase pairs source
    if KINASE_PAIRS_FILE.exists():
        kp = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b"], nrows=1_000_000)
        kpairs = list(set(kp['mol_a'].dropna().astype(str).tolist() +
                          kp['mol_b'].dropna().astype(str).tolist()))
        print(f"  Kinase pairs sources: {len(kpairs):,}")
        sources.extend(kpairs)
    # ZAP70 source (already small)
    if ZAP70_CSV.exists():
        try:
            zdf = pd.read_csv(ZAP70_CSV, usecols=["canonical_smiles"]).dropna()
            zlist = list(set(zdf['canonical_smiles'].astype(str).tolist()))
            print(f"  ZAP70/ChEMBL whole CSV sources: {len(zlist):,}")
            sources.extend(zlist)
        except Exception as e:
            print(f"  WARN reading {ZAP70_CSV}: {e}")

    sources = list(set(sources))
    print(f"  Unique source SMILES: {len(sources):,}")
    if len(sources) > max_amines:
        # Sample
        np.random.seed(0)
        idx = np.random.choice(len(sources), size=max_amines, replace=False)
        sources = [sources[i] for i in idx]

    # Parallel worker
    args = [(s, mw_whole, mw_brics) for s in sources]
    print(f"  Decomposing on {N_PROCS} workers...")
    seen = set()
    amines = []
    chunk_size = 5000
    n_processed = 0
    n_log_step = 50_000
    last_log = 0

    with mp.Pool(N_PROCS) as pool:
        for i, (whole, frags) in enumerate(pool.imap_unordered(
                _process_one_molecule_for_amines, args, chunksize=200)):
            n_processed += 1
            for sm, mw, src in whole + frags:
                if sm in seen:
                    continue
                seen.add(sm)
                amines.append({"smiles": sm, "MW": mw, "source": f"ChEMBL_{src}"})
            if n_processed - last_log >= n_log_step:
                last_log = n_processed
                print(f"    processed {n_processed:,}/{len(args):,}  amines so far: {len(amines):,}")

    print(f"  Mined {len(amines):,} unique amines (MW≤{mw_whole}/{mw_brics})")
    return amines


# ── Step 2: Hand-curated amines (always include) ──────────────────────────────

HAND_AMINES = [
    # Original Mol 1 amine
    "Nc1cn(C(C)C)cn1",
    # Imidazole variants
    "Nc1cncn1C", "Nc1cn(CC)cn1", "Nc1cn(C2CC2)cn1", "Nc1cn(C(C)(C)C)cn1",
    # Pyrazole variants
    "Nc1ccn(C(C)C)n1", "Nc1cnn(C)c1", "Nc1ccnn1C(C)C",
    # Triazole
    "Nc1nn(C(C)C)cn1", "Nc1cn(C(C)C)nn1",
    # Other heterocycles
    "Nc1cnoc1", "Nc1cnsc1", "Nc1ncc(C)s1", "Nc1ccnc(N)n1",
    "Nc1ncccn1", "Nc1ccncn1", "Nc1ccncc1", "Nc1cccnc1", "Nc1ccccn1",
    # Anilines
    "Nc1ccccc1", "Nc1ccc(F)cc1", "Nc1ccc(Cl)cc1", "Nc1ccc(OC)cc1",
    "Nc1cccc(C)c1", "Nc1cc(C)cc(C)c1",
    "NCc1ccccc1", "NCc1cnc(C(C)C)cn1",
    "NC1CCCCC1", "NCC1CCCO1", "NC1CCNC1", "NCCN1CCCC1",
    "Nc1cc2ccccc2[nH]1", "Nc1[nH]c2ccccc2n1", "Nc1nc2ccccc2[nH]1",
    # Med-chem favorites
    "Nc1ccnc(N2CCOCC2)n1",  # 4-amino-2-morpholino-pyrimidine
    "Nc1ncn(C2CCOCC2)c1",
    "Nc1cnc(N2CCN(C)CC2)nc1",
    "Nc1cccc(N2CCOCC2)c1",
    "Nc1cccc(F)c1F", "Nc1ccc(C(F)(F)F)cc1",
    "Nc1ccc(C#N)cc1", "Nc1ccc(N(C)C)cc1",
    # Aliphatic small
    "NCC(F)F", "NCC(F)(F)F", "NCC1CCC1", "NCC1CC1",
    # Aminoindazoles, aminoindoles
    "Nc1ccc2[nH]ncc2c1", "Nc1ccc2[nH]ccc2c1",
    "Nc1cnc2[nH]ccc2c1",
    # 4-aminopyridines with substituents
    "Nc1ccnc(C)c1", "Nc1ccnc(F)c1", "Nc1ccnc(Cl)c1", "Nc1ccnc(OC)c1",
    # 4-aminoquinoline / quinazoline scaffolds (kinase-classic)
    "Nc1ccnc2ccccc12", "Nc1ncnc2ccccc12",
]


# ── Step 3: Couple ────────────────────────────────────────────────────────────

def _couple_one(args):
    """Worker: couple frag_A_acid + amine."""
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


def couple_all(amines, frag_a_acid):
    print(f"\n=== STEP 3: Coupling {len(amines):,} amines with frag A ===")
    args = [(frag_a_acid, a["smiles"]) for a in amines]
    seen = set()
    products = []
    n_log_step = 50_000

    with mp.Pool(N_PROCS) as pool:
        for i, prod in enumerate(pool.imap_unordered(_couple_one, args, chunksize=500)):
            if prod is None:
                continue
            if prod in seen:
                continue
            seen.add(prod)
            products.append({"smiles": prod, "amine_idx": i})
            if (i + 1) % n_log_step == 0:
                print(f"    coupled {i+1:,}/{len(args):,}  unique products: {len(products):,}")

    print(f"  Final unique coupled products: {len(products):,}")
    return products


# ── Step 4: Score with FiLMDelta + descriptors (cheap, batch) ─────────────────

def _score_one_mol(args):
    """Worker: compute Morgan FP + RDKit descriptors for one SMILES."""
    smi = args
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


def score_descriptors_batch(products):
    print(f"\n=== STEP 4: Scoring descriptors for {len(products):,} products ===")
    with mp.Pool(N_PROCS) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(
                _score_one_mol, [p["smiles"] for p in products], chunksize=500)):
            if r is not None:
                out.append(r)
            if (i + 1) % 100_000 == 0:
                print(f"    scored {i+1:,}/{len(products):,}")
    print(f"  Final scored: {len(out):,}")
    return out


def _fp_one_module(smi):
    """Module-level worker (must be picklable for mp.Pool) — Morgan FP for one SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def score_film_batch(products, model_path):
    """Score products with FiLMDelta in batch (CPU).
    For 1M products, this is ~1 hr on 16-CPU.
    """
    print(f"\n=== STEP 5: FiLMDelta scoring on {len(products):,} products ===")
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
    print(f"  Loaded model: {n_anchors} anchors")

    print("  Computing FPs in parallel...")
    smiles = [p["smiles"] for p in products]
    with mp.Pool(N_PROCS) as pool:
        fps = pool.map(_fp_one_module, smiles, chunksize=500)
    valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
    print(f"  Valid FPs: {len(valid):,}")

    scores = np.full(len(products), np.nan)
    bs = 256
    print(f"  Scoring (bs={bs})...")
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
            if (start + bs) % 20000 == 0:
                print(f"    {start+bs:,}/{len(valid_arr):,} scored")
    return scores


def main():
    print("=" * 70)
    print("AI-CHEM TIER 2 ENRICHMENT — ~1M AMINES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Step 1+2: amines
    # MW cap reasoning (med-chem permissive): final product MW = frag_A_acid (~284)
    # + amine - 17 (loss of OH). With amine MW ≤ 500, final ≤ ~770. With amine
    # MW ≤ 400 (BRICS), final ≤ ~670. These are roomy for early-stage screening.
    chembl_amines = mine_chembl_amines(mw_whole=500, mw_brics=400, max_amines=300_000)
    hand_amines = [{"smiles": s, "MW": Descriptors.MolWt(Chem.MolFromSmiles(s)),
                    "source": "hand_curated"} for s in HAND_AMINES if Chem.MolFromSmiles(s)]
    seen = set()
    all_amines = []
    for a in hand_amines + chembl_amines:
        if a["smiles"] in seen:
            continue
        seen.add(a["smiles"])
        all_amines.append(a)
    print(f"\nTotal amine library: {len(all_amines):,} (hand={len(hand_amines)} + ChEMBL={len(chembl_amines):,})")

    # Save amine library
    pd.DataFrame(all_amines).to_csv(RESULTS_DIR / "amine_library.csv", index=False)

    # Step 3: couple
    products = couple_all(all_amines, FRAG_A_ACID)
    pd.DataFrame(products).to_csv(RESULTS_DIR / "coupled_products_smiles_only.csv", index=False)

    # Step 4: descriptors
    scored_descriptors = score_descriptors_batch(products)
    df = pd.DataFrame(scored_descriptors)
    df.to_csv(RESULTS_DIR / "products_descriptors.csv", index=False)

    # Step 5: FiLMDelta scoring (single seed for speed)
    model_path = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if model_path.exists():
        scores = score_film_batch(products, model_path)
        df_final = df.copy()
        df_final["pIC50_film"] = [scores[i] if i < len(scores) else np.nan
                                    for i in range(len(df_final))]
        df_final = df_final.sort_values("pIC50_film", ascending=False)
        df_final.to_csv(RESULTS_DIR / "products_scored.csv", index=False)
        # Save top 50K to its own file
        df_top = df_final.head(50_000)
        df_top.to_csv(RESULTS_DIR / "products_top50k.csv", index=False)
        print(f"\nTop 50K saved → {RESULTS_DIR / 'products_top50k.csv'}")
    else:
        print(f"  WARN: FiLMDelta checkpoint not found at {model_path}; skipping scoring")

    # Summary
    summary = {
        "n_amines": len(all_amines),
        "n_hand_amines": len(hand_amines),
        "n_chembl_amines": len(chembl_amines),
        "n_products": len(products),
        "n_descriptors_scored": len(scored_descriptors),
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {RESULTS_DIR / 'summary.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
