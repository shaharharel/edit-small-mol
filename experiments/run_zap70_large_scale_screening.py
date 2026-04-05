#!/usr/bin/env python3
"""
ZAP70 (CHEMBL2803) Large-Scale Molecular Screening Pipeline.

Target: Tyrosine-protein kinase ZAP-70 (CHEMBL2803), 280 training molecules.
Two scoring models: XGB ensemble (v5 style) and FiLMDelta + Kinase PT (anchor-based).
Goal: screen ~1M molecules with confidence scoring.

Tier 1: Neighborhood Expansion (~100K)
  - MMP enumeration: apply beneficial edits to all training mols
  - BRICS recombination from all 280 molecules
  - Kinase cross-pollination from ChEMBL
  - R-group enumeration on top scaffolds

Tier 2: Generative Expansion (~500K-1M)
  - CReM mutate_mol on each training molecule (if DB available)
  - Genetic algorithm with BRICS crossover + CReM/MMP mutation

Tier 3: Confidence Scoring
  - Multi-metric confidence: NN similarity, ensemble std, Mahalanobis distance

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_zap70_large_scale_screening.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import gc
import json
import os
import sqlite3
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints,
    train_xgboost, train_rf, N_JOBS,
)
from experiments.run_zap70_v5 import (
    compute_multi_fp, train_ensemble, compute_druglikeness,
    BEST_XGB_PARAMS, BEST_RF_PARAMS,
)
from experiments.run_zap70_edit_unified import FiLMDeltaAnchorModel
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
CHEMBL_DB = PROJECT_ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
CREM_DB_PATH = PROJECT_ROOT / "data" / "crem_db" / "chembl33_sa2_f5.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_FILE = RESULTS_DIR / "zap70_large_scale_screening.json"

TARGET_ID = "CHEMBL2803"

# Use MPS for FiLMDelta inference (simple MLP, no transformer issues)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Common R-groups for enumeration
COMMON_R_GROUPS = [
    "C", "CC", "CCC", "C(C)C",                     # Alkyl: Me, Et, nPr, iPr
    "F", "Cl", "Br",                                 # Halogens
    "C(F)(F)F", "C(F)(F)Cl",                         # CF3, CF2Cl
    "OC", "OCC",                                      # OMe, OEt
    "N", "NC", "NC(C)C",                              # NH2, NHMe, NMe2
    "O",                                               # OH
    "C#N",                                             # CN (nitrile)
    "C(=O)C", "C(=O)OC",                             # Acetyl, ester
    "S(=O)(=O)C",                                      # Methanesulfonyl
    "C1CCOCC1",                                        # Morpholine (ring)
    "C1CCNCC1",                                        # Piperidine (ring)
    "C1CCNC1",                                         # Pyrrolidine (ring)
    "c1ccccc1",                                        # Phenyl
    "c1ccncc1",                                        # Pyridine
    "c1cnc2ccccc2n1",                                  # Quinazoline
]


def timer(msg):
    """Simple timer context manager."""
    class Timer:
        def __init__(self, m):
            self.msg = m
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n>>> {self.msg}...")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            print(f"    [{self.msg}] completed in {elapsed:.1f}s")
            self.elapsed = elapsed
    return Timer(msg)


def canonicalize(smi):
    """Canonicalize SMILES, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def batch_canonicalize(smiles_list):
    """Canonicalize a list of SMILES, filtering out invalids."""
    result = set()
    for smi in smiles_list:
        can = canonicalize(smi)
        if can is not None:
            result.add(can)
    return result


def tanimoto_kernel_matrix(X, Y=None):
    """Vectorized Tanimoto kernel for binary fingerprints."""
    if Y is None:
        Y = X
    XY = X @ Y.T
    X2 = np.sum(X, axis=1, keepdims=True)
    Y2 = np.sum(Y, axis=1, keepdims=True)
    denom = X2 + Y2.T - XY + 1e-10
    return XY / denom


def save_results(results):
    """Save intermediate results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_FILE}")


# =============================================================================
# DualModelScreener: Wraps XGB ensemble + FiLMDelta anchor model
# =============================================================================

class DualModelScreener:
    """
    Wraps both scoring models for large-scale screening:
    1. XGB ensemble (5 models on diverse FPs) -- fast, good absolute prediction
    2. FiLMDelta + Kinase PT (anchor-based) -- delta-aware, noise-robust
    """

    def __init__(self, mol_data, fp_cache):
        self.mol_data = mol_data
        self.fp_cache = fp_cache
        self.train_smiles = mol_data["smiles"].tolist()
        self.train_pIC50 = mol_data["pIC50"].values
        self.n_train = len(self.train_smiles)

        # Models (populated by train_models)
        self.xgb_models = {}
        self.xgb_fps = {}  # FP arrays for training set, keyed by FP type
        self.film_model = None  # FiLMDeltaAnchorModel

        # Training FPs for similarity computation
        self.train_morgan_fp = None

    def train_models(self):
        """Train both scoring models on all training data."""

        # --- XGB Ensemble ---
        with timer("Training XGB ensemble (5 models)"):
            fps_dict = compute_multi_fp(self.train_smiles)
            self.xgb_fps = fps_dict
            self.train_morgan_fp = fps_dict["morgan"]

            # We train each sub-model on full training data.
            # For scoring, we retrain each on full data and predict directly.
            # Store the trained models for later batch scoring.
            import xgboost as xgb
            from sklearn.ensemble import RandomForestRegressor

            y = self.train_pIC50

            # 1. XGB on AtomPair
            m = xgb.XGBRegressor(**BEST_XGB_PARAMS, random_state=42, n_jobs=N_JOBS)
            m.fit(fps_dict["atompair"], y, verbose=False)
            self.xgb_models["xgb_atompair"] = ("atompair", m)

            # 2. RF on RDKit FP
            m = RandomForestRegressor(**BEST_RF_PARAMS, n_jobs=N_JOBS, random_state=42)
            m.fit(fps_dict["rdkit"], y)
            self.xgb_models["rf_rdkit"] = ("rdkit", m)

            # 3. XGB on ECFP6
            m = xgb.XGBRegressor(**BEST_XGB_PARAMS, random_state=42, n_jobs=N_JOBS)
            m.fit(fps_dict["ecfp6"], y, verbose=False)
            self.xgb_models["xgb_ecfp6"] = ("ecfp6", m)

            # 4. XGB on Morgan
            m = xgb.XGBRegressor(**BEST_XGB_PARAMS, random_state=42, n_jobs=N_JOBS)
            m.fit(fps_dict["morgan"], y, verbose=False)
            self.xgb_models["xgb_morgan"] = ("morgan", m)

            # 5. RF on AtomPair
            m = RandomForestRegressor(**BEST_RF_PARAMS, n_jobs=N_JOBS, random_state=42)
            m.fit(fps_dict["atompair"], y)
            self.xgb_models["rf_atompair"] = ("atompair", m)

            print(f"    Trained {len(self.xgb_models)} XGB/RF models")

        # --- FiLMDelta + Kinase PT ---
        with timer("Training FiLMDelta + Kinase PT"):
            self.film_model = FiLMDeltaAnchorModel()

            # Load kinase pairs
            if KINASE_PAIRS_FILE.exists():
                kinase_pairs = pd.read_csv(
                    KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"]
                )
                print(f"    Kinase pairs: {len(kinase_pairs):,}")

                # Compute FP for all kinase molecules
                all_kinase_smi = list(set(
                    kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
                ))
                # Add training SMILES if not already present
                all_smi_needed = list(set(all_kinase_smi + self.train_smiles))
                missing = [s for s in all_smi_needed if s not in self.fp_cache]
                if missing:
                    print(f"    Computing FP for {len(missing)} additional molecules...")
                    X_missing = compute_fingerprints(missing, "morgan", radius=2, n_bits=2048)
                    for i, smi in enumerate(missing):
                        self.fp_cache[smi] = X_missing[i]
                    del X_missing

                # Filter to molecules with FP
                mask = (kinase_pairs["mol_a"].apply(lambda s: s in self.fp_cache) &
                        kinase_pairs["mol_b"].apply(lambda s: s in self.fp_cache))
                kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
                print(f"    Kinase pairs (with FP): {len(kinase_pairs):,}")

                # Pretrain
                self.film_model.pretrain_on_kinase(kinase_pairs, self.fp_cache)
                del kinase_pairs
                gc.collect()
            else:
                print("    WARNING: kinase_within_pairs.csv not found, skipping pretraining")
                # Initialize without pretraining
                from sklearn.preprocessing import StandardScaler
                all_embs = np.array([self.fp_cache[s] for s in self.train_smiles])
                self.film_model.scaler = StandardScaler()
                self.film_model.scaler.fit(all_embs)
                self.film_model.model = FiLMDeltaMLP(
                    input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2
                )
                del all_embs

            # Fine-tune on ZAP70
            self.film_model.finetune_on_zap70(
                self.train_smiles, self.train_pIC50, self.fp_cache
            )

            # Move model to device for inference
            self.film_model.model = self.film_model.model.to(DEVICE)
            self.film_model.anchor_embs = self.film_model.anchor_embs.to(DEVICE)

        print(f"\n  Both models trained. Device for FiLMDelta: {DEVICE}")

    def score_batch(self, smiles_list, batch_size=10000):
        """
        Score a batch of SMILES with both models.

        Returns DataFrame with columns:
          smiles, xgb_pred, xgb_std, film_pred, film_std,
          consensus_pred, nn_similarity
        """
        if not smiles_list:
            return pd.DataFrame()

        n = len(smiles_list)
        results = {
            "smiles": smiles_list,
            "xgb_pred": np.zeros(n),
            "xgb_std": np.zeros(n),
            "film_pred": np.zeros(n),
            "film_std": np.zeros(n),
            "nn_similarity": np.zeros(n),
        }

        # Process in chunks to manage memory
        for chunk_start in range(0, n, batch_size):
            chunk_end = min(chunk_start + batch_size, n)
            chunk_smiles = smiles_list[chunk_start:chunk_end]
            chunk_size = len(chunk_smiles)

            # --- XGB Ensemble Scoring ---
            # Compute FPs for this chunk
            fps_chunk = {}
            for fp_type in ["morgan", "atompair", "rdkit", "ecfp6"]:
                radius = 3 if fp_type == "ecfp6" else 2
                actual_type = "morgan" if fp_type == "ecfp6" else fp_type
                fps_chunk[fp_type] = compute_fingerprints(
                    chunk_smiles, actual_type, radius=radius, n_bits=2048
                )

            # Get predictions from each sub-model
            xgb_preds = []
            for model_name, (fp_key, model) in self.xgb_models.items():
                preds = model.predict(fps_chunk[fp_key])
                xgb_preds.append(preds)

            xgb_preds = np.array(xgb_preds)  # (n_models, chunk_size)
            results["xgb_pred"][chunk_start:chunk_end] = np.mean(xgb_preds, axis=0)
            results["xgb_std"][chunk_start:chunk_end] = np.std(xgb_preds, axis=0)

            # --- NN Similarity ---
            chunk_morgan = fps_chunk["morgan"]
            # Compute max Tanimoto to any training molecule
            sim_matrix = tanimoto_kernel_matrix(chunk_morgan, self.train_morgan_fp)
            results["nn_similarity"][chunk_start:chunk_end] = np.max(sim_matrix, axis=1)
            del sim_matrix

            # --- FiLMDelta Scoring ---
            # Ensure all chunk SMILES have FP in cache
            for i, smi in enumerate(chunk_smiles):
                if smi not in self.fp_cache:
                    self.fp_cache[smi] = fps_chunk["morgan"][i]

            # Anchor-based prediction using FiLMDelta
            film_preds, film_stds = self._film_predict_batch(chunk_smiles)
            results["film_pred"][chunk_start:chunk_end] = film_preds
            results["film_std"][chunk_start:chunk_end] = film_stds

            del fps_chunk
            if chunk_end % 50000 == 0 or chunk_end == n:
                print(f"    Scored {chunk_end}/{n} molecules")
            gc.collect()

        # Consensus = mean of XGB and FiLMDelta
        results["consensus_pred"] = 0.5 * (results["xgb_pred"] + results["film_pred"])

        df = pd.DataFrame(results)
        return df

    def _film_predict_batch(self, smiles_list):
        """FiLMDelta anchor-based absolute prediction on device."""
        self.film_model.model.eval()
        n_targets = len(smiles_list)
        n_anchors = len(self.film_model.anchor_smiles)

        target_embs_raw = np.array([self.fp_cache[s] for s in smiles_list])
        target_embs = torch.FloatTensor(
            self.film_model.scaler.transform(target_embs_raw)
        ).to(DEVICE)
        del target_embs_raw

        anchor_embs = self.film_model.anchor_embs  # already on DEVICE
        anchor_pIC50 = self.film_model.anchor_pIC50

        predictions = np.zeros(n_targets)
        uncertainties = np.zeros(n_targets)

        # Process targets in batches
        TBATCH = 200
        for start in range(0, n_targets, TBATCH):
            end = min(start + TBATCH, n_targets)
            batch_preds = np.zeros((end - start, n_anchors))

            for j_local, j_global in enumerate(range(start, end)):
                target_expanded = target_embs[j_global:j_global + 1].expand(n_anchors, -1)
                with torch.no_grad():
                    deltas = self.film_model.model(anchor_embs, target_expanded).cpu().numpy()
                batch_preds[j_local] = anchor_pIC50 + deltas

            predictions[start:end] = np.mean(batch_preds, axis=1)
            uncertainties[start:end] = np.std(batch_preds, axis=1)

        del target_embs
        return predictions, uncertainties


# =============================================================================
# Tier 1: Neighborhood Expansion
# =============================================================================

def tier1_mmp_enumeration(mol_data, fp_cache):
    """Apply known beneficial MMP edits to training molecules."""
    with timer("Tier 1a: MMP enumeration"):
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values

        # Identify beneficial edits from all pairs: find edits that improve pIC50
        # We build a SMARTS-based edit catalog from training MMP pairs
        # For each pair (A, B) with delta > 0.5, extract the structural change
        # Instead of full MMP analysis, use a curated set of common kinase medicinal chemistry edits

        beneficial_edits_smarts = [
            # Halogenation
            ("[cH:1]", "[c:1]F"),
            ("[cH:1]", "[c:1]Cl"),
            ("[cH:1]", "[c:1]Br"),
            # Fluorination of methyl
            ("[CH3:1]", "[C:1](F)(F)F"),
            # Methyl to ethyl
            ("[CH3:1]", "[CH2:1]C"),
            # Add methyl to aromatic
            ("[cH:1]", "[c:1]C"),
            # Methoxy
            ("[cH:1]", "[c:1]OC"),
            # Amino
            ("[cH:1]", "[c:1]N"),
            ("[cH:1]", "[c:1]NC"),
            # Nitrile
            ("[cH:1]", "[c:1]C#N"),
            # Sulfonamide
            ("[cH:1]", "[c:1]S(=O)(=O)N"),
            # OH to OMe
            ("[OH:1]", "[O:1]C"),
            # NH to NMe
            ("[NH:1]", "[N:1]C"),
            # Ring expansion: pyrrolidine to piperidine
            # Amine to amide
            ("[NH2:1]", "[NH:1]C(=O)C"),
            # Carboxylic acid to amide
            ("[C:1](=O)[OH]", "[C:1](=O)N"),
            # Ester to amide
            ("[C:1](=O)[O]C", "[C:1](=O)N"),
        ]

        generated = set()
        n_valid = 0

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            for reactant_smarts, product_smarts in beneficial_edits_smarts:
                try:
                    pattern = Chem.MolFromSmarts(reactant_smarts)
                    if pattern is None:
                        continue
                    matches = mol.GetSubstructMatches(pattern)
                    if not matches:
                        continue

                    replacement = Chem.MolFromSmarts(product_smarts)
                    if replacement is None:
                        # Try as SMILES
                        replacement = Chem.MolFromSmiles(product_smarts)
                    if replacement is None:
                        continue

                    products = AllChem.ReplaceSubstructs(mol, pattern, replacement)
                    for prod in products:
                        try:
                            Chem.SanitizeMol(prod)
                            can = Chem.MolToSmiles(prod)
                            if can and can not in generated:
                                # Basic validity check
                                check = Chem.MolFromSmiles(can)
                                if check is not None:
                                    generated.add(can)
                                    n_valid += 1
                        except Exception:
                            continue
                except Exception:
                    continue

        # Remove training molecules
        train_set = set(canonicalize(s) for s in smiles_list)
        generated -= train_set
        generated.discard(None)

        print(f"    MMP enumeration: {len(generated)} unique valid molecules")
        return generated


def tier1_brics_recombination(mol_data):
    """BRICS decomposition + recombination from all training molecules."""
    with timer("Tier 1b: BRICS recombination"):
        smiles_list = mol_data["smiles"].tolist()

        # Decompose all molecules
        all_fragments = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_fragments.update(frags)
            except Exception:
                continue

        print(f"    BRICS fragments: {len(all_fragments)}")

        # Recombine (cap at 50K to control runtime)
        generated = set()
        MAX_BRICS = 50000

        try:
            # BRICSDecompose returns SMILES strings; BRICSBuild needs mol objects
            frag_mols = [Chem.MolFromSmiles(f) for f in all_fragments]
            frag_mols = [m for m in frag_mols if m is not None]
            print(f"    BRICS fragment mols: {len(frag_mols)}")
            builder = BRICS.BRICSBuild(frag_mols)
            for i, mol in enumerate(builder):
                if i >= MAX_BRICS:
                    break
                try:
                    Chem.SanitizeMol(mol)
                    can = Chem.MolToSmiles(mol)
                    # Filter for reasonable size (not too small or huge)
                    check = Chem.MolFromSmiles(can)
                    if check and 10 <= check.GetNumHeavyAtoms() <= 80:
                        generated.add(can)
                except Exception:
                    continue
        except Exception as e:
            print(f"    BRICS build error: {e}")

        # Remove training molecules
        train_set = set(canonicalize(s) for s in smiles_list)
        generated -= train_set
        generated.discard(None)

        print(f"    BRICS recombination: {len(generated)} unique molecules")
        return generated


def tier1_kinase_crosspolination(mol_data):
    """Get compounds from related kinase targets in ChEMBL."""
    with timer("Tier 1c: Kinase cross-pollination"):
        if not CHEMBL_DB.exists():
            print("    ChEMBL database not found, skipping.")
            return set()

        train_set = set(canonicalize(s) for s in mol_data["smiles"].tolist())
        train_set.discard(None)

        db = sqlite3.connect(str(CHEMBL_DB))

        # Query all kinase targets with significant activity data
        query = """
            SELECT DISTINCT cs.canonical_smiles
            FROM activities a
            JOIN assays ass ON a.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_class cc ON tc.component_id = cc.component_id
            JOIN molecule_dictionary md ON a.molregno = md.molregno
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE cc.protein_class_desc LIKE '%Kinase%'
            AND a.pchembl_value >= 6.0
            AND td.chembl_id != ?
            AND cs.canonical_smiles IS NOT NULL
            LIMIT 100000
        """
        try:
            df = pd.read_sql_query(query, db, params=[TARGET_ID])
            print(f"    Kinase compounds from ChEMBL: {len(df)}")
        except Exception as e:
            print(f"    Query error (trying simpler query): {e}")
            # Fallback: just get related kinase targets
            related_kinases = [
                "CHEMBL3009", "CHEMBL2599", "CHEMBL1841", "CHEMBL258",
                "CHEMBL5251", "CHEMBL2971", "CHEMBL1862", "CHEMBL267",
                "CHEMBL1906", "CHEMBL399", "CHEMBL4045", "CHEMBL203",
                "CHEMBL4722", "CHEMBL1824", "CHEMBL2148", "CHEMBL279",
                "CHEMBL4015", "CHEMBL325", "CHEMBL4860", "CHEMBL2111",
            ]
            placeholders = ",".join("?" * len(related_kinases))
            query = f"""
                SELECT DISTINCT cs.canonical_smiles
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                JOIN compound_structures cs ON md.molregno = cs.molregno
                WHERE td.chembl_id IN ({placeholders})
                AND a.pchembl_value >= 6.0
                AND cs.canonical_smiles IS NOT NULL
            """
            df = pd.read_sql_query(query, db, params=related_kinases)
            print(f"    Related kinase compounds: {len(df)}")

        db.close()

        generated = set()
        for smi in df["canonical_smiles"]:
            can = canonicalize(smi)
            if can and can not in train_set:
                generated.add(can)

        print(f"    Kinase cross-pollination: {len(generated)} unique novel molecules")
        return generated


def tier1_rgroup_enumeration(mol_data):
    """R-group enumeration on top scaffolds."""
    with timer("Tier 1d: R-group enumeration"):
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values

        # Extract Murcko scaffolds
        scaffold_mols = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smi = Chem.MolToSmiles(scaffold)
                scaffold_mols[scaffold_smi].append((smi, pIC50[i]))
            except Exception:
                continue

        # Top 10 scaffolds by molecule count (or pIC50)
        top_scaffolds = sorted(
            scaffold_mols.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]

        print(f"    Top 10 scaffolds (of {len(scaffold_mols)} total):")
        for scaffold_smi, mols in top_scaffolds:
            mean_pic50 = np.mean([p for _, p in mols])
            print(f"      {scaffold_smi[:60]:60s} ({len(mols)} mols, mean pIC50={mean_pic50:.2f})")

        # For each scaffold, find attachment points and enumerate R-groups
        generated = set()
        for scaffold_smi, mols in top_scaffolds:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
            if scaffold_mol is None:
                continue

            # Find aromatic CH positions (common substitution points)
            for r_smi in COMMON_R_GROUPS:
                try:
                    pattern = Chem.MolFromSmarts("[cH]")
                    if pattern is None:
                        continue
                    r_mol = Chem.MolFromSmiles(r_smi)
                    if r_mol is None:
                        continue

                    # For each molecule with this scaffold, try R-group additions
                    for mol_smi, _ in mols:
                        mol = Chem.MolFromSmiles(mol_smi)
                        if mol is None:
                            continue

                        matches = mol.GetSubstructMatches(pattern)
                        if not matches:
                            continue

                        # Try replacing each aromatic H with R-group
                        r_pattern = Chem.MolFromSmarts("[cH:1]")
                        r_replacement = Chem.MolFromSmarts(f"[c:1]{r_smi}")
                        if r_replacement is None:
                            # Build by attachment
                            r_replacement = Chem.MolFromSmiles(f"c({r_smi})")
                        if r_replacement is None:
                            continue

                        try:
                            products = AllChem.ReplaceSubstructs(mol, r_pattern, r_replacement)
                            for prod in products[:3]:  # Limit per position
                                try:
                                    Chem.SanitizeMol(prod)
                                    can = Chem.MolToSmiles(prod)
                                    check = Chem.MolFromSmiles(can)
                                    if check and 10 <= check.GetNumHeavyAtoms() <= 80:
                                        generated.add(can)
                                except Exception:
                                    continue
                        except Exception:
                            continue
                except Exception:
                    continue

        # Remove training molecules
        train_set = set(canonicalize(s) for s in smiles_list)
        generated -= train_set
        generated.discard(None)

        print(f"    R-group enumeration: {len(generated)} unique molecules")
        return generated


def run_tier1(mol_data, fp_cache):
    """Run all Tier 1 expansions and merge."""
    print("\n" + "=" * 70)
    print("TIER 1: Neighborhood Expansion")
    print("=" * 70)

    tier1_results = {}
    all_generated = set()

    # 1a. MMP enumeration
    mmp_mols = tier1_mmp_enumeration(mol_data, fp_cache)
    tier1_results["mmp"] = len(mmp_mols)
    all_generated.update(mmp_mols)
    del mmp_mols
    gc.collect()

    # 1b. BRICS recombination
    brics_mols = tier1_brics_recombination(mol_data)
    tier1_results["brics"] = len(brics_mols)
    all_generated.update(brics_mols)
    del brics_mols
    gc.collect()

    # 1c. Kinase cross-pollination
    kinase_mols = tier1_kinase_crosspolination(mol_data)
    tier1_results["kinase"] = len(kinase_mols)
    all_generated.update(kinase_mols)
    del kinase_mols
    gc.collect()

    # 1d. R-group enumeration
    rgroup_mols = tier1_rgroup_enumeration(mol_data)
    tier1_results["rgroup"] = len(rgroup_mols)
    all_generated.update(rgroup_mols)
    del rgroup_mols
    gc.collect()

    # Remove training set
    train_set = set(canonicalize(s) for s in mol_data["smiles"].tolist())
    all_generated -= train_set
    all_generated.discard(None)

    tier1_results["n_unique"] = len(all_generated)
    print(f"\n  Tier 1 total unique molecules: {len(all_generated):,}")
    print(f"    MMP: {tier1_results['mmp']}, BRICS: {tier1_results['brics']}, "
          f"Kinase: {tier1_results['kinase']}, R-group: {tier1_results['rgroup']}")

    return list(all_generated), tier1_results


# =============================================================================
# Tier 2: Generative Expansion
# =============================================================================

def tier2_crem_generation(mol_data):
    """Generate analogs using CReM (if DB available)."""
    with timer("Tier 2a: CReM generation"):
        if not CREM_DB_PATH.exists():
            # Also check common alternative locations
            alt_paths = [
                PROJECT_ROOT / "data" / "crem_db" / "replacements.db",
                PROJECT_ROOT / "data" / "crem_db" / "replacements_sa2.db",
                PROJECT_ROOT / "data" / "crem_db" / "chembl22_sa2.db",
                PROJECT_ROOT / "data" / "replacements.db",
                Path.home() / "data" / "crem" / "replacements.db",
            ]
            db_path = None
            for p in alt_paths:
                if p.exists():
                    db_path = p
                    break
            if db_path is None:
                print("    CReM database not found at any of:")
                print(f"      {CREM_DB_PATH}")
                for p in alt_paths:
                    print(f"      {p}")
                print("    Skipping CReM generation.")
                print("    To use CReM, download from https://github.com/DrrDom/crem")
                return set()
        else:
            db_path = CREM_DB_PATH

        try:
            from crem.crem import mutate_mol, grow_mol
        except ImportError:
            print("    CReM not installed, skipping.")
            return set()

        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values
        db_str = str(db_path)

        generated = set()
        n_attempted = 0
        n_errors = 0

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            n_attempted += 1
            try:
                # mutate_mol: replace fragments
                muts = list(mutate_mol(
                    mol, db_name=db_str,
                    radius=3, min_size=0, max_size=10,
                    max_replacements=1000,
                    return_mol=False,
                ))
                for new_smi in muts:
                    can = canonicalize(new_smi)
                    if can:
                        generated.add(can)
            except Exception:
                n_errors += 1

            # Also try grow_mol on potent molecules
            if pIC50[i] >= 7.0:
                try:
                    growths = list(grow_mol(
                        mol, db_name=db_str,
                        radius=2, min_atoms=1, max_atoms=8,
                        max_replacements=500,
                        return_mol=False,
                    ))
                    for new_smi in growths:
                        can = canonicalize(new_smi)
                        if can:
                            generated.add(can)
                except Exception:
                    pass

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(smiles_list)} molecules, "
                      f"{len(generated):,} unique generated")

        # Remove training molecules
        train_set = set(canonicalize(s) for s in smiles_list)
        generated -= train_set
        generated.discard(None)

        print(f"    CReM: {len(generated):,} unique molecules "
              f"(from {n_attempted} molecules, {n_errors} errors)")
        return generated


def tier2_genetic_algorithm(mol_data, scorer, n_generations=50, pop_size=2000):
    """Genetic algorithm for molecule optimization."""
    with timer(f"Tier 2b: Genetic algorithm ({n_generations} gens, pop {pop_size})"):
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values

        # Check if CReM is available for mutation
        has_crem = False
        crem_db_str = None
        if CREM_DB_PATH.exists():
            try:
                from crem.crem import mutate_mol
                has_crem = True
                crem_db_str = str(CREM_DB_PATH)
            except ImportError:
                pass

        # Initialize population from training molecules (bias toward potent ones)
        # Sample with replacement, weighted by pIC50
        weights = np.exp(pIC50 - pIC50.max())  # Softmax-like weighting
        weights /= weights.sum()
        init_indices = np.random.choice(len(smiles_list), size=pop_size, replace=True, p=weights)
        population = [smiles_list[i] for i in init_indices]

        # Pre-decompose all training molecules for BRICS crossover
        all_fragments = set()
        mol_fragments = {}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_fragments.update(frags)
                mol_fragments[smi] = list(frags)
            except Exception:
                continue

        all_fragments_list = list(all_fragments)
        all_unique_molecules = set()  # Track all unique molecules across generations

        # Beneficial edits from MMP catalog (simplified SMARTS)
        edit_patterns = [
            ("[cH:1]", "[c:1]F"),
            ("[cH:1]", "[c:1]Cl"),
            ("[cH:1]", "[c:1]C"),
            ("[cH:1]", "[c:1]OC"),
            ("[cH:1]", "[c:1]C#N"),
            ("[CH3:1]", "[C:1](F)(F)F"),
            ("[OH:1]", "[O:1]C"),
            ("[NH:1]", "[N:1]C"),
        ]

        def mutate(smi):
            """Apply a random mutation to a molecule."""
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi

            # Strategy 1: CReM mutation (if available)
            if has_crem and np.random.random() < 0.3:
                try:
                    from crem.crem import mutate_mol as crem_mutate
                    muts = list(crem_mutate(
                        mol, db_name=crem_db_str,
                        radius=3, min_size=0, max_size=8,
                        max_replacements=10,
                        return_mol=False,
                    ))
                    if muts:
                        new_smi = np.random.choice(muts)
                        can = canonicalize(new_smi)
                        if can:
                            return can
                except Exception:
                    pass

            # Strategy 2: SMARTS-based edit
            if np.random.random() < 0.5:
                pattern_smi, replacement_smi = edit_patterns[
                    np.random.randint(len(edit_patterns))
                ]
                try:
                    pattern = Chem.MolFromSmarts(pattern_smi)
                    replacement = Chem.MolFromSmarts(replacement_smi)
                    if pattern and replacement:
                        matches = mol.GetSubstructMatches(pattern)
                        if matches:
                            products = AllChem.ReplaceSubstructs(mol, pattern, replacement)
                            if products:
                                prod = products[np.random.randint(len(products))]
                                try:
                                    Chem.SanitizeMol(prod)
                                    can = Chem.MolToSmiles(prod)
                                    check = Chem.MolFromSmiles(can)
                                    if check:
                                        return can
                                except Exception:
                                    pass
                except Exception:
                    pass

            # Strategy 3: Random atom replacement (fallback)
            try:
                # Remove a random substituent or add one
                rwmol = Chem.RWMol(mol)
                if rwmol.GetNumAtoms() > 5:
                    # Try removing a terminal atom
                    terminal_atoms = [
                        a.GetIdx() for a in rwmol.GetAtoms()
                        if a.GetDegree() == 1 and a.GetAtomicNum() != 1
                    ]
                    if terminal_atoms:
                        remove_idx = np.random.choice(terminal_atoms)
                        rwmol.RemoveAtom(remove_idx)
                        try:
                            Chem.SanitizeMol(rwmol)
                            can = Chem.MolToSmiles(rwmol)
                            check = Chem.MolFromSmiles(can)
                            if check and check.GetNumHeavyAtoms() >= 5:
                                return can
                        except Exception:
                            pass
            except Exception:
                pass

            return smi  # Return unchanged if all mutations fail

        def crossover(smi1, smi2):
            """BRICS-based crossover between two molecules."""
            try:
                mol1 = Chem.MolFromSmiles(smi1)
                mol2 = Chem.MolFromSmiles(smi2)
                if mol1 is None or mol2 is None:
                    return smi1

                frags1 = list(BRICS.BRICSDecompose(mol1))
                frags2 = list(BRICS.BRICSDecompose(mol2))

                if not frags1 or not frags2:
                    return smi1

                # Mix fragments from both parents
                mixed = list(set(frags1[:len(frags1)//2] + frags2[:len(frags2)//2]))
                if len(mixed) < 2:
                    mixed = frags1 + frags2

                builder = BRICS.BRICSBuild(mixed)
                for j, mol in enumerate(builder):
                    if j >= 5:  # Try up to 5 builds
                        break
                    try:
                        Chem.SanitizeMol(mol)
                        can = Chem.MolToSmiles(mol)
                        check = Chem.MolFromSmiles(can)
                        if check and 10 <= check.GetNumHeavyAtoms() <= 80:
                            return can
                    except Exception:
                        continue
            except Exception:
                pass
            return smi1

        # GA loop
        best_fitness = -np.inf
        for gen in range(n_generations):
            # Score current population
            # Use XGB ensemble only for speed (FiLMDelta is too slow for GA)
            pop_fps = {}
            for fp_type in ["morgan", "atompair", "rdkit", "ecfp6"]:
                radius = 3 if fp_type == "ecfp6" else 2
                actual_type = "morgan" if fp_type == "ecfp6" else fp_type
                pop_fps[fp_type] = compute_fingerprints(
                    population, actual_type, radius=radius, n_bits=2048
                )

            fitness_preds = []
            for model_name, (fp_key, model) in scorer.xgb_models.items():
                fitness_preds.append(model.predict(pop_fps[fp_key]))
            fitness = np.mean(fitness_preds, axis=0)

            # Track all unique molecules
            for smi in population:
                can = canonicalize(smi)
                if can:
                    all_unique_molecules.add(can)

            # Stats
            gen_best = np.max(fitness)
            gen_mean = np.mean(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"    Gen {gen + 1}/{n_generations}: "
                      f"best={gen_best:.2f}, mean={gen_mean:.2f}, "
                      f"unique total={len(all_unique_molecules):,}")

            # Selection: tournament, keep top 50%
            n_keep = pop_size // 2
            sorted_idx = np.argsort(fitness)[::-1]
            survivors = [population[i] for i in sorted_idx[:n_keep]]

            # Generate new population
            new_pop = list(survivors)  # Start with survivors

            # Fill remaining slots with mutation and crossover
            while len(new_pop) < pop_size:
                if np.random.random() < 0.7:
                    # Mutation
                    parent = survivors[np.random.randint(len(survivors))]
                    child = mutate(parent)
                    new_pop.append(child)
                else:
                    # Crossover
                    p1 = survivors[np.random.randint(len(survivors))]
                    p2 = survivors[np.random.randint(len(survivors))]
                    child = crossover(p1, p2)
                    new_pop.append(child)

            population = new_pop[:pop_size]
            del pop_fps
            gc.collect()

        # Remove training molecules
        train_set = set(canonicalize(s) for s in smiles_list)
        all_unique_molecules -= train_set
        all_unique_molecules.discard(None)

        print(f"    GA: {len(all_unique_molecules):,} unique molecules "
              f"over {n_generations} generations")
        return all_unique_molecules


def run_tier2(mol_data, scorer):
    """Run all Tier 2 generative expansions."""
    print("\n" + "=" * 70)
    print("TIER 2: Generative Expansion")
    print("=" * 70)

    tier2_results = {}
    all_generated = set()

    # 2a. CReM generation
    crem_mols = tier2_crem_generation(mol_data)
    tier2_results["crem"] = len(crem_mols)
    all_generated.update(crem_mols)
    del crem_mols
    gc.collect()

    # 2b. Genetic algorithm
    ga_mols = tier2_genetic_algorithm(mol_data, scorer)
    tier2_results["ga"] = len(ga_mols)
    all_generated.update(ga_mols)
    del ga_mols
    gc.collect()

    # Remove training set
    train_set = set(canonicalize(s) for s in mol_data["smiles"].tolist())
    all_generated -= train_set
    all_generated.discard(None)

    tier2_results["n_unique"] = len(all_generated)
    print(f"\n  Tier 2 total unique molecules: {len(all_generated):,}")
    print(f"    CReM: {tier2_results['crem']}, GA: {tier2_results['ga']}")

    return list(all_generated), tier2_results


# =============================================================================
# Tier 3: Confidence Scoring
# =============================================================================

def compute_confidence(scored_df, train_fps, train_pIC50):
    """
    Compute multi-metric confidence scores for scored molecules.

    Adds columns: local_density, mahalanobis_dist, confidence_tier
    """
    with timer("Computing confidence metrics"):
        n = len(scored_df)

        # Already have nn_similarity from scoring
        nn_sim = scored_df["nn_similarity"].values

        # Local density: mean Tanimoto to top-5 nearest neighbors
        print("    Computing local density...")
        local_density = np.zeros(n)

        # Process in chunks for memory
        CHUNK = 5000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_smiles = scored_df["smiles"].iloc[start:end].tolist()
            chunk_fps = compute_fingerprints(chunk_smiles, "morgan", radius=2, n_bits=2048)
            sim_matrix = tanimoto_kernel_matrix(chunk_fps, train_fps)

            for i in range(len(chunk_smiles)):
                top5 = np.sort(sim_matrix[i])[-5:]
                local_density[start + i] = np.mean(top5)

            del chunk_fps, sim_matrix
            gc.collect()

        # Mahalanobis distance from training centroid
        print("    Computing Mahalanobis distance...")
        train_mean = np.mean(train_fps, axis=0)
        # Use pseudo-inverse for covariance (FPs are sparse, full cov is singular)
        # Simplified: use diagonal covariance (much faster for 2048-dim)
        train_var = np.var(train_fps, axis=0) + 1e-6
        mahal_dist = np.zeros(n)

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_smiles = scored_df["smiles"].iloc[start:end].tolist()
            chunk_fps = compute_fingerprints(chunk_smiles, "morgan", radius=2, n_bits=2048)

            for i in range(len(chunk_smiles)):
                diff = chunk_fps[i] - train_mean
                mahal_dist[start + i] = np.sqrt(np.sum(diff**2 / train_var))

            del chunk_fps
            gc.collect()

        # Assign confidence tiers
        ensemble_std = np.maximum(scored_df["xgb_std"].values, scored_df["film_std"].values)

        confidence_tier = []
        for i in range(n):
            if nn_sim[i] > 0.4 and ensemble_std[i] < 0.3:
                confidence_tier.append("HIGH")
            elif nn_sim[i] > 0.2 and ensemble_std[i] < 0.5:
                confidence_tier.append("MEDIUM")
            elif nn_sim[i] > 0.1:
                confidence_tier.append("LOW")
            else:
                confidence_tier.append("SPECULATIVE")

        scored_df = scored_df.copy()
        scored_df["local_density"] = local_density
        scored_df["mahalanobis_dist"] = mahal_dist
        scored_df["confidence_tier"] = confidence_tier

        # Tier counts
        tier_counts = Counter(confidence_tier)
        print(f"    Confidence tiers: HIGH={tier_counts.get('HIGH', 0)}, "
              f"MEDIUM={tier_counts.get('MEDIUM', 0)}, "
              f"LOW={tier_counts.get('LOW', 0)}, "
              f"SPECULATIVE={tier_counts.get('SPECULATIVE', 0)}")

        return scored_df


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    pipeline_start = time.time()
    print("=" * 70)
    print("ZAP70 Large-Scale Molecular Screening Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    results = {
        "target": TARGET_ID,
        "target_name": "ZAP70 (Tyrosine-protein kinase ZAP-70)",
        "started": datetime.now().isoformat(),
        "device": str(DEVICE),
    }

    # === Step 1: Load data ===
    with timer("Loading ZAP70 data"):
        mol_data, _ = load_zap70_molecules()
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values
        n_train = len(smiles_list)
        print(f"    {n_train} training molecules")

    # Compute fingerprints for training set
    with timer("Computing training fingerprints"):
        train_fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
        fp_cache = {smi: train_fps[i] for i, smi in enumerate(smiles_list)}

    results["n_training"] = n_train
    results["pIC50_range"] = [float(pIC50.min()), float(pIC50.max())]

    # === Step 2: Train both models ===
    scorer = DualModelScreener(mol_data, fp_cache)
    scorer.train_models()
    results["models_trained"] = True
    save_results(results)

    # === Step 3: Tier 1 expansion ===
    tier1_smiles, tier1_info = run_tier1(mol_data, fp_cache)
    results["tier1"] = tier1_info
    save_results(results)

    # === Step 4: Score Tier 1 ===
    if tier1_smiles:
        with timer(f"Scoring Tier 1 ({len(tier1_smiles):,} molecules)"):
            tier1_scored = scorer.score_batch(tier1_smiles)
            tier1_scored = compute_confidence(tier1_scored, train_fps, pIC50)

        # Save top candidates
        tier1_scored_sorted = tier1_scored.sort_values("consensus_pred", ascending=False)
        top_tier1 = []
        for _, row in tier1_scored_sorted.head(100).iterrows():
            top_tier1.append({
                "smiles": row["smiles"],
                "consensus_pred": round(float(row["consensus_pred"]), 3),
                "xgb_pred": round(float(row["xgb_pred"]), 3),
                "film_pred": round(float(row["film_pred"]), 3),
                "xgb_std": round(float(row["xgb_std"]), 3),
                "film_std": round(float(row["film_std"]), 3),
                "nn_similarity": round(float(row["nn_similarity"]), 3),
                "confidence_tier": row["confidence_tier"],
            })

        tier1_stats = {
            "n_scored": len(tier1_scored),
            "n_high_confidence": int((tier1_scored["confidence_tier"] == "HIGH").sum()),
            "n_medium_confidence": int((tier1_scored["confidence_tier"] == "MEDIUM").sum()),
            "n_predicted_potent_7": int((tier1_scored["consensus_pred"] >= 7.0).sum()),
            "n_predicted_potent_6": int((tier1_scored["consensus_pred"] >= 6.0).sum()),
            "mean_consensus": round(float(tier1_scored["consensus_pred"].mean()), 3),
            "max_consensus": round(float(tier1_scored["consensus_pred"].max()), 3),
            "top_candidates": top_tier1,
        }
        results["tier1"].update(tier1_stats)

        # Print summary
        print(f"\n  Tier 1 scoring summary:")
        print(f"    Molecules scored: {len(tier1_scored):,}")
        print(f"    Predicted potent (pIC50>=7): {tier1_stats['n_predicted_potent_7']}")
        print(f"    Predicted active (pIC50>=6): {tier1_stats['n_predicted_potent_6']}")
        print(f"    High confidence: {tier1_stats['n_high_confidence']}")
        print(f"    Medium confidence: {tier1_stats['n_medium_confidence']}")

        save_results(results)
    else:
        tier1_scored = pd.DataFrame()
        print("  No Tier 1 molecules generated.")

    gc.collect()

    # === Step 5: Tier 2 expansion ===
    tier2_smiles, tier2_info = run_tier2(mol_data, scorer)
    results["tier2"] = tier2_info
    save_results(results)

    # === Step 6: Score Tier 2 ===
    if tier2_smiles:
        # Remove any already scored in Tier 1
        tier1_set = set(tier1_scored["smiles"].tolist()) if len(tier1_scored) > 0 else set()
        tier2_novel = [s for s in tier2_smiles if s not in tier1_set]
        print(f"  Tier 2 novel (not in Tier 1): {len(tier2_novel):,}")

        if tier2_novel:
            with timer(f"Scoring Tier 2 ({len(tier2_novel):,} molecules)"):
                tier2_scored = scorer.score_batch(tier2_novel)
                tier2_scored = compute_confidence(tier2_scored, train_fps, pIC50)

            # Save top candidates
            tier2_scored_sorted = tier2_scored.sort_values("consensus_pred", ascending=False)
            top_tier2 = []
            for _, row in tier2_scored_sorted.head(100).iterrows():
                top_tier2.append({
                    "smiles": row["smiles"],
                    "consensus_pred": round(float(row["consensus_pred"]), 3),
                    "xgb_pred": round(float(row["xgb_pred"]), 3),
                    "film_pred": round(float(row["film_pred"]), 3),
                    "xgb_std": round(float(row["xgb_std"]), 3),
                    "film_std": round(float(row["film_std"]), 3),
                    "nn_similarity": round(float(row["nn_similarity"]), 3),
                    "confidence_tier": row["confidence_tier"],
                })

            tier2_stats = {
                "n_scored": len(tier2_scored),
                "n_high_confidence": int((tier2_scored["confidence_tier"] == "HIGH").sum()),
                "n_medium_confidence": int((tier2_scored["confidence_tier"] == "MEDIUM").sum()),
                "n_predicted_potent_7": int((tier2_scored["consensus_pred"] >= 7.0).sum()),
                "n_predicted_potent_6": int((tier2_scored["consensus_pred"] >= 6.0).sum()),
                "mean_consensus": round(float(tier2_scored["consensus_pred"].mean()), 3),
                "max_consensus": round(float(tier2_scored["consensus_pred"].max()), 3),
                "top_candidates": top_tier2,
            }
            results["tier2"].update(tier2_stats)

            print(f"\n  Tier 2 scoring summary:")
            print(f"    Molecules scored: {len(tier2_scored):,}")
            print(f"    Predicted potent (pIC50>=7): {tier2_stats['n_predicted_potent_7']}")
            print(f"    Predicted active (pIC50>=6): {tier2_stats['n_predicted_potent_6']}")
            print(f"    High confidence: {tier2_stats['n_high_confidence']}")
        else:
            tier2_scored = pd.DataFrame()
    else:
        tier2_scored = pd.DataFrame()
        print("  No Tier 2 molecules generated.")

    save_results(results)
    gc.collect()

    # === Step 7: Merge, deduplicate, final ranking ===
    with timer("Final ranking and merging"):
        dfs_to_merge = []
        if len(tier1_scored) > 0:
            tier1_scored = tier1_scored.copy()
            tier1_scored["source_tier"] = "tier1"
            dfs_to_merge.append(tier1_scored)
        if len(tier2_scored) > 0:
            tier2_scored = tier2_scored.copy()
            tier2_scored["source_tier"] = "tier2"
            dfs_to_merge.append(tier2_scored)

        if not dfs_to_merge:
            print("    No molecules to merge. Pipeline complete (no molecules generated).")
            results["combined"] = {"n_total_unique": 0}
            save_results(results)
            return

        combined = pd.concat(dfs_to_merge, ignore_index=True)
        # Deduplicate by SMILES, keeping higher consensus prediction
        combined = combined.sort_values("consensus_pred", ascending=False)
        combined = combined.drop_duplicates(subset="smiles", keep="first")
        combined = combined.reset_index(drop=True)

        n_total = len(combined)
        n_high = int((combined["confidence_tier"] == "HIGH").sum())
        n_medium = int((combined["confidence_tier"] == "MEDIUM").sum())
        n_low = int((combined["confidence_tier"] == "LOW").sum())
        n_speculative = int((combined["confidence_tier"] == "SPECULATIVE").sum())
        n_potent_7 = int((combined["consensus_pred"] >= 7.0).sum())
        n_potent_6 = int((combined["consensus_pred"] >= 6.0).sum())

        # Compute druglikeness for top 50
        top50_entries = []
        for i, (_, row) in enumerate(combined.head(50).iterrows()):
            dl = compute_druglikeness(row["smiles"])
            entry = {
                "rank": i + 1,
                "smiles": row["smiles"],
                "consensus_pred": round(float(row["consensus_pred"]), 3),
                "xgb_pred": round(float(row["xgb_pred"]), 3),
                "film_pred": round(float(row["film_pred"]), 3),
                "xgb_std": round(float(row["xgb_std"]), 3),
                "film_std": round(float(row["film_std"]), 3),
                "nn_similarity": round(float(row["nn_similarity"]), 3),
                "confidence_tier": row["confidence_tier"],
                "source_tier": row["source_tier"],
            }
            if dl:
                entry.update({
                    "MW": round(dl.get("MW", 0), 1),
                    "LogP": round(dl.get("LogP", 0), 2),
                    "QED": round(dl.get("QED", 0), 3),
                    "SA_score": round(dl.get("SA_score", 0), 2),
                    "Lipinski_violations": dl.get("Lipinski_violations", None),
                })
            top50_entries.append(entry)

        results["combined"] = {
            "n_total_unique": n_total,
            "n_high_confidence": n_high,
            "n_medium_confidence": n_medium,
            "n_low_confidence": n_low,
            "n_speculative": n_speculative,
            "n_predicted_potent_7": n_potent_7,
            "n_predicted_potent_6": n_potent_6,
            "consensus_mean": round(float(combined["consensus_pred"].mean()), 3),
            "consensus_max": round(float(combined["consensus_pred"].max()), 3),
            "top_50": top50_entries,
        }

    # === Step 8: Final summary ===
    elapsed = time.time() - pipeline_start
    results["completed"] = datetime.now().isoformat()
    results["elapsed_seconds"] = round(elapsed, 1)
    save_results(results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total unique molecules screened: {n_total:,}")
    print(f"  Tier 1 (neighborhood): {results['tier1'].get('n_unique', 0):,}")
    print(f"  Tier 2 (generative):   {results['tier2'].get('n_unique', 0):,}")
    print(f"")
    print(f"  Confidence distribution:")
    print(f"    HIGH:         {n_high:,}")
    print(f"    MEDIUM:       {n_medium:,}")
    print(f"    LOW:          {n_low:,}")
    print(f"    SPECULATIVE:  {n_speculative:,}")
    print(f"")
    print(f"  Predicted potent (pIC50 >= 7.0): {n_potent_7:,}")
    print(f"  Predicted active (pIC50 >= 6.0): {n_potent_6:,}")
    print(f"")
    print(f"  Top 10 candidates:")
    print(f"  {'Rank':>4} {'Consensus':>9} {'XGB':>6} {'FiLM':>6} {'Sim':>5} {'Conf':>12} {'Tier':>5}")
    for c in top50_entries[:10]:
        print(f"  {c['rank']:4d} {c['consensus_pred']:9.3f} {c['xgb_pred']:6.3f} "
              f"{c['film_pred']:6.3f} {c['nn_similarity']:5.3f} "
              f"{c['confidence_tier']:>12s} {c['source_tier']:>5s}")
    print(f"")
    print(f"  Total elapsed time: {elapsed/60:.1f} min")
    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
