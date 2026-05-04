#!/usr/bin/env python3
"""
Mol 1 Expansion Pipeline — End-to-End Molecule Generation & Scoring.

Seed compound (Mol 1, free base):
    C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1
    Pyrimidine-piperazine with acrylamide warhead, ZAP70 inhibitor.

Phase 1: Conservative Expansion (MMP + CReM + BRICS)
Phase 2: Directed Generation (REINVENT4 Mol2Mol + LibInvent — config only if not installed)
Phase 3: Scoring & Ranking all candidates with FiLMDelta anchor-based prediction

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_expansion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import gc
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
torch.backends.mps.is_available = lambda: False  # Force CPU for stability

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, BRICS, QED, FilterCatalog
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "overlapping_assays" / "molecule_pIC50_minimal.csv"
SHARED_PAIRS_FILE = DATA_DIR / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
KINASE_PAIRS_FILE = DATA_DIR / "kinase_within_pairs.csv"
CREM_DB_PATH = DATA_DIR / "crem_db" / "chembl33_sa2_f5.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_expansion"
RESULTS_FILE = RESULTS_DIR / "expansion_results.json"
REPORT_FILE = RESULTS_DIR / "expansion_report.html"

TARGET_ID = "CHEMBL2803"
MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
N_SEEDS = 3
DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def timer(msg):
    class Timer:
        def __init__(self, m):
            self.msg = m
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n>>> {self.msg}...")
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.t0
            print(f"    [{self.msg}] completed in {self.elapsed:.1f}s")
    return Timer(msg)


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for a list of SMILES."""
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)
    return np.array(results, dtype=np.float32)


def tanimoto_similarity(fp1, fp2):
    """Tanimoto similarity between two fingerprint vectors."""
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    if union == 0:
        return 0.0
    return float(intersection / union)


def tanimoto_kernel_matrix(X, Y=None):
    if Y is None:
        Y = X
    XY = X @ Y.T
    X2 = np.sum(X, axis=1, keepdims=True)
    Y2 = np.sum(Y, axis=1, keepdims=True)
    denom = X2 + Y2.T - XY + 1e-10
    return XY / denom


# Load SA scorer once at module level
_SASCORER = None
def _get_sascorer():
    global _SASCORER
    if _SASCORER is None:
        try:
            from rdkit.Chem import RDConfig
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score', 'sascorer.py')
            if os.path.exists(sa_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("sascorer", sa_path)
                _SASCORER = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_SASCORER)
        except Exception:
            pass
    return _SASCORER


def compute_mol_properties(smi):
    """Compute molecular properties for a SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        props = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "TPSA": Descriptors.TPSA(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "QED": QED.qed(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "Rings": Descriptors.RingCount(mol),
        }
        sascorer = _get_sascorer()
        if sascorer is not None:
            try:
                props["SA_Score"] = sascorer.calculateScore(mol)
            except Exception:
                props["SA_Score"] = None
        else:
            props["SA_Score"] = None
        return props
    except Exception:
        return None


# Load PAINS filter catalog once at module level
_PAINS_CATALOG = None
def _get_pains_catalog():
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def check_pains(mol):
    """Check if molecule has PAINS alerts. Returns True if PAINS-free."""
    if mol is None:
        return False
    fc = _get_pains_catalog()
    entry = fc.GetFirstMatch(mol)
    return entry is None  # True = no PAINS match = good


def check_lipinski(props):
    """Check Lipinski's Rule of Five."""
    if props is None:
        return False
    violations = 0
    if props.get("MW", 999) > 500: violations += 1
    if props.get("LogP", 99) > 5: violations += 1
    if props.get("HBA", 99) > 10: violations += 1
    if props.get("HBD", 99) > 5: violations += 1
    return violations <= 1


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_zap70_molecules():
    """Load ZAP70 molecule-level data (averaged across assays)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == TARGET_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  ZAP70: {len(mol_data)} molecules, pIC50 {mol_data['pIC50'].min():.2f}-"
          f"{mol_data['pIC50'].max():.2f} (mean={mol_data['pIC50'].mean():.2f})")
    return mol_data


def load_zap70_pairs():
    """Load ZAP70 MMP pairs from shared pairs dataset."""
    df = pd.read_csv(SHARED_PAIRS_FILE, usecols=[
        "mol_a", "mol_b", "delta", "target_chembl_id", "is_within_assay"
    ])
    zap = df[df["target_chembl_id"] == TARGET_ID].copy()
    print(f"  ZAP70 pairs: {len(zap)} total, {zap['is_within_assay'].sum()} within-assay")
    return zap


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Conservative Expansion
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_mmp_enumeration(mol1_smi, zap70_pairs):
    """Apply known MMP transforms from ZAP70 training data to Mol 1."""
    with timer("Phase 1a: MMP enumeration from ZAP70 transforms"):
        mol1 = Chem.MolFromSmiles(mol1_smi)
        if mol1 is None:
            print("    ERROR: Mol 1 is invalid!")
            return set()

        # Extract unique transforms from ZAP70 pairs
        # Use SMARTS-based edits: common medicinal chemistry transformations
        # that appear in kinase SAR
        beneficial_edits = [
            # Halogenation of aromatic H
            ("[cH:1]", "[c:1]F"),
            ("[cH:1]", "[c:1]Cl"),
            ("[cH:1]", "[c:1]Br"),
            # Methyl additions
            ("[cH:1]", "[c:1]C"),
            ("[NH:1]", "[N:1]C"),
            # Methoxy / Ethoxy
            ("[cH:1]", "[c:1]OC"),
            ("[cH:1]", "[c:1]OCC"),
            # Amino groups
            ("[cH:1]", "[c:1]N"),
            ("[cH:1]", "[c:1]NC"),
            ("[cH:1]", "[c:1]N(C)C"),
            # Nitrile
            ("[cH:1]", "[c:1]C#N"),
            # CF3
            ("[CH3:1]", "[C:1](F)(F)F"),
            # Fluorination
            ("[CH3:1]", "[CH2:1]F"),
            # OH to OMe
            ("[OH:1]", "[O:1]C"),
            # NH to NMe
            ("[NH2:1]", "[NH:1]C"),
            # Ethyl to iPr
            ("[CH2:1][CH3]", "[CH:1]([CH3])[CH3]"),
            # Piperazine N-methylation
            ("[NH:1]", "[N:1]C"),
            # Acrylamide variations
            ("[CH:1]=[CH2]", "[C:1](=O)"),
            # Sulfonamide
            ("[cH:1]", "[c:1]S(=O)(=O)N"),
            ("[cH:1]", "[c:1]S(=O)(=O)C"),
            # Pyridine/pyrimidine nitrogen
            ("[cH:1]", "[c:1]c1ccncc1"),
            # Small ring additions
            ("[cH:1]", "[c:1]C1CC1"),  # cyclopropyl
        ]

        generated = set()
        for reactant_smarts, product_smarts in beneficial_edits:
            try:
                pattern = Chem.MolFromSmarts(reactant_smarts)
                if pattern is None:
                    continue
                matches = mol1.GetSubstructMatches(pattern)
                if not matches:
                    continue

                replacement = Chem.MolFromSmarts(product_smarts)
                if replacement is None:
                    replacement = Chem.MolFromSmiles(product_smarts)
                if replacement is None:
                    continue

                products = AllChem.ReplaceSubstructs(mol1, pattern, replacement)
                for prod in products:
                    try:
                        Chem.SanitizeMol(prod)
                        can = Chem.MolToSmiles(prod)
                        if can:
                            check = Chem.MolFromSmiles(can)
                            if check is not None:
                                generated.add(can)
                    except Exception:
                        continue
            except Exception:
                continue

        # Also apply transforms extracted from actual ZAP70 training pairs
        # For each pair where mol_a or mol_b is structurally similar to Mol 1,
        # extract the structural difference and try to apply it
        mol1_fp = compute_fingerprints([mol1_smi])[0]

        # Get all unique molecules from pairs
        all_pair_smiles = list(set(zap70_pairs["mol_a"].tolist() + zap70_pairs["mol_b"].tolist()))
        if len(all_pair_smiles) > 0:
            pair_fps = compute_fingerprints(all_pair_smiles)
            # Find molecules similar to Mol 1
            sims = tanimoto_kernel_matrix(mol1_fp.reshape(1, -1), pair_fps)[0]
            sim_mask = sims > 0.3  # At least 30% similar
            similar_mols = [all_pair_smiles[i] for i in range(len(all_pair_smiles)) if sim_mask[i]]
            print(f"    Found {len(similar_mols)} molecules with Tanimoto > 0.3 to Mol 1")

            # For pairs involving similar molecules, try ReplaceSubstructs approach
            for _, row in zap70_pairs.iterrows():
                mol_a_smi, mol_b_smi = row["mol_a"], row["mol_b"]
                # If mol_a is similar to mol1, apply the transform mol_a -> mol_b to mol1
                if mol_a_smi in similar_mols:
                    mol_a = Chem.MolFromSmiles(mol_a_smi)
                    mol_b = Chem.MolFromSmiles(mol_b_smi)
                    if mol_a is None or mol_b is None:
                        continue
                    # Simple approach: find MCS and try to swap the non-MCS part
                    # This is approximate; for real MMP transforms, use RDKit MMP code
                    can = canonicalize(mol_b_smi)
                    if can:
                        generated.add(can)

            del pair_fps, sims
            gc.collect()

        mol1_can = canonicalize(mol1_smi)
        generated.discard(mol1_can)
        generated.discard(None)

        print(f"    MMP enumeration: {len(generated)} unique molecules")
        return generated


def phase1_crem_generation(mol1_smi):
    """Generate analogs of Mol 1 using CReM context-aware fragment replacement."""
    with timer("Phase 1b: CReM generation"):
        if not CREM_DB_PATH.exists():
            print("    CReM database not found, skipping.")
            return set()

        try:
            from crem.crem import mutate_mol, grow_mol
        except ImportError:
            print("    CReM not installed, skipping.")
            return set()

        mol1 = Chem.MolFromSmiles(mol1_smi)
        if mol1 is None:
            return set()

        db_str = str(CREM_DB_PATH)
        generated = set()

        # 1. mutate_mol: replace fragments
        try:
            muts = list(mutate_mol(
                mol1, db_name=db_str,
                radius=3, min_size=0, max_size=10,
                max_replacements=5000,
                return_mol=False,
            ))
            for new_smi in muts:
                can = canonicalize(new_smi)
                if can:
                    generated.add(can)
            print(f"    CReM mutate (r=3): {len(muts)} raw, {len(generated)} unique valid")
        except Exception as e:
            print(f"    CReM mutate error: {e}")

        # 2. Also try radius=2 for more diverse replacements
        n_before = len(generated)
        try:
            muts2 = list(mutate_mol(
                mol1, db_name=db_str,
                radius=2, min_size=0, max_size=12,
                max_replacements=5000,
                return_mol=False,
            ))
            for new_smi in muts2:
                can = canonicalize(new_smi)
                if can:
                    generated.add(can)
            print(f"    CReM mutate (r=2): {len(muts2)} raw, {len(generated) - n_before} new unique")
        except Exception as e:
            print(f"    CReM mutate r=2 error: {e}")

        # 3. grow_mol: add fragments at open positions
        n_before = len(generated)
        try:
            growths = list(grow_mol(
                mol1, db_name=db_str,
                radius=2, min_atoms=1, max_atoms=8,
                max_replacements=3000,
                return_mol=False,
            ))
            for new_smi in growths:
                can = canonicalize(new_smi)
                if can:
                    generated.add(can)
            print(f"    CReM grow: {len(growths)} raw, {len(generated) - n_before} new unique")
        except Exception as e:
            print(f"    CReM grow error: {e}")

        mol1_can = canonicalize(mol1_smi)
        generated.discard(mol1_can)
        generated.discard(None)

        print(f"    CReM total: {len(generated)} unique molecules")
        return generated


def phase1_brics_recombination(mol1_smi, mol_data):
    """BRICS decomposition of Mol 1 + recombination with fragments from potent ZAP70 mols."""
    with timer("Phase 1c: BRICS recombination"):
        mol1 = Chem.MolFromSmiles(mol1_smi)
        if mol1 is None:
            return set()

        # Decompose Mol 1
        mol1_frags = set()
        try:
            mol1_frags = BRICS.BRICSDecompose(mol1)
            print(f"    Mol 1 BRICS fragments: {len(mol1_frags)}")
            for f in mol1_frags:
                print(f"      {f}")
        except Exception as e:
            print(f"    Mol 1 decomposition error: {e}")
            return set()

        # Collect fragments from potent ZAP70 molecules (pIC50 >= 6.5)
        potent = mol_data[mol_data["pIC50"] >= 6.5]
        print(f"    Potent ZAP70 molecules (pIC50 >= 6.5): {len(potent)}")

        all_fragments = set(mol1_frags)
        for smi in potent["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_fragments.update(frags)
            except Exception:
                continue

        print(f"    Total BRICS fragments: {len(all_fragments)}")

        # Recombine — only use Mol 1 fragments + top 30 most common other fragments
        # to keep combinatorial explosion manageable
        generated = set()
        MAX_BRICS = 5000

        # Prioritize fragments: all Mol 1 frags + top fragments from potent molecules
        other_frags = all_fragments - mol1_frags
        # Limit other fragments to keep BRICSBuild tractable
        other_frags_list = list(other_frags)[:40]
        selected_frags = list(mol1_frags) + other_frags_list
        print(f"    Selected {len(selected_frags)} fragments for recombination "
              f"({len(mol1_frags)} from Mol 1 + {len(other_frags_list)} from training)")

        try:
            frag_mols = [Chem.MolFromSmiles(f) for f in selected_frags]
            frag_mols = [m for m in frag_mols if m is not None]
            print(f"    BRICS fragment mols: {len(frag_mols)}")
            builder = BRICS.BRICSBuild(frag_mols)
            t0 = time.time()
            for i, mol in enumerate(builder):
                if i >= MAX_BRICS:
                    break
                # Timeout after 5 minutes
                if time.time() - t0 > 300:
                    print(f"      BRICS timeout after 5 minutes at {i} products")
                    break
                try:
                    Chem.SanitizeMol(mol)
                    can = Chem.MolToSmiles(mol)
                    check = Chem.MolFromSmiles(can)
                    if check and 15 <= check.GetNumHeavyAtoms() <= 60:
                        generated.add(can)
                except Exception:
                    continue
                if (i + 1) % 1000 == 0:
                    print(f"      Processed {i + 1} BRICS products, {len(generated)} valid...")
        except Exception as e:
            print(f"    BRICS build error: {e}")

        mol1_can = canonicalize(mol1_smi)
        generated.discard(mol1_can)
        generated.discard(None)

        # Also remove training molecules
        train_set = set(canonicalize(s) for s in mol_data["smiles"])
        generated -= train_set

        print(f"    BRICS recombination: {len(generated)} unique novel molecules")
        return generated


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Directed Generation (REINVENT4 — config generation only)
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_reinvent_configs(mol1_smi):
    """Generate REINVENT4 configs for Mol2Mol and LibInvent. Does not run them."""
    with timer("Phase 2: REINVENT4 config generation"):
        configs_dir = RESULTS_DIR / "reinvent4_configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        # Check if REINVENT4 is available
        reinvent_dir = Path("/Users/shaharharel/Documents/github/REINVENT4")
        reinvent_available = reinvent_dir.exists()
        print(f"    REINVENT4 available: {reinvent_available}")

        # Generate Mol2Mol config
        mol2mol_config = f"""# REINVENT4: Mol2Mol Optimization for Mol 1 Expansion
# Seed: {mol1_smi}
# Target: ZAP70 (CHEMBL2803)

run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol1_mol2mol"
json_out_config = "_mol1_mol2mol.json"

[parameters]
summary_csv_prefix = "mol1_mol2mol"
use_checkpoint = false
purge_memories = false

prior_file = "{reinvent_dir}/priors/mol2mol_medium_similarity.prior"
agent_file = "{reinvent_dir}/priors/mol2mol_medium_similarity.prior"
smiles_file = "{RESULTS_DIR}/mol1_seed.smi"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "mol1_mol2mol_stage1.chkpt"
termination = "simple"
max_score = 0.7
min_steps = 10
max_steps = 30

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "conda"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.25

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
"""
        with open(configs_dir / "mol1_mol2mol.toml", "w") as f:
            f.write(mol2mol_config)

        # Write seed SMILES file
        with open(RESULTS_DIR / "mol1_seed.smi", "w") as f:
            f.write(f"{mol1_smi}\n")

        print(f"    Mol2Mol config written to {configs_dir / 'mol1_mol2mol.toml'}")
        print(f"    Seed SMILES written to {RESULTS_DIR / 'mol1_seed.smi'}")

        if not reinvent_available:
            print("    REINVENT4 not available for execution — configs generated for manual use")

        return {
            "reinvent_available": reinvent_available,
            "mol2mol_config": str(configs_dir / "mol1_mol2mol.toml"),
            "seed_file": str(RESULTS_DIR / "mol1_seed.smi"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: FiLMDelta Scoring
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMDeltaScorer:
    """
    Trains FiLMDelta on ZAP70 data and scores candidates via anchor-based prediction.
    pIC50(j) = median(pIC50(i) + delta(i->j)) across all training anchors.
    """

    def __init__(self, mol_data, seed=42):
        self.mol_data = mol_data
        self.seed = seed
        self.model = None
        self.scaler = None
        self.anchor_smiles = None
        self.anchor_embs = None
        self.anchor_pIC50 = None
        self.fp_cache = {}

    def _build_fp_cache(self, smiles_list):
        """Build/extend fingerprint cache."""
        missing = [s for s in smiles_list if s not in self.fp_cache]
        if missing:
            fps = compute_fingerprints(missing)
            for i, smi in enumerate(missing):
                self.fp_cache[smi] = fps[i]

    def train(self):
        """Train FiLMDelta with optional kinase pretraining + ZAP70 fine-tuning."""
        from sklearn.preprocessing import StandardScaler

        train_smiles = self.mol_data["smiles"].tolist()
        train_pIC50 = self.mol_data["pIC50"].values
        n = len(train_smiles)

        self._build_fp_cache(train_smiles)

        # Try kinase pretraining
        pretrained = False
        if KINASE_PAIRS_FILE.exists():
            print("    Loading kinase pairs for pretraining...")
            kinase_pairs = pd.read_csv(
                KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"]
            )
            all_kinase_smi = list(set(
                kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
            ))
            self._build_fp_cache(all_kinase_smi + train_smiles)

            # Filter to molecules with FP
            mask = (kinase_pairs["mol_a"].apply(lambda s: s in self.fp_cache) &
                    kinase_pairs["mol_b"].apply(lambda s: s in self.fp_cache))
            kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
            print(f"    Kinase pairs: {len(kinase_pairs):,}")

            # Build tensors
            emb_a = np.array([self.fp_cache[s] for s in kinase_pairs["mol_a"]])
            emb_b = np.array([self.fp_cache[s] for s in kinase_pairs["mol_b"]])
            delta = kinase_pairs["delta"].values.astype(np.float32)

            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack([emb_a, emb_b]))

            Xa = torch.FloatTensor(self.scaler.transform(emb_a))
            Xb = torch.FloatTensor(self.scaler.transform(emb_b))
            yd = torch.FloatTensor(delta)
            del emb_a, emb_b, delta, kinase_pairs
            gc.collect()

            n_val = len(Xa) // 10
            input_dim = Xa.shape[1]

            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.model = FiLMDeltaMLP(
                input_dim=input_dim, hidden_dims=[1024, 512, 256], dropout=0.2
            )
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.MSELoss()

            best_vl, best_state, wait = float("inf"), None, 0
            for epoch in range(100):
                self.model.train()
                perm = np.random.permutation(len(Xa) - n_val) + n_val
                for start in range(0, len(perm), 256):
                    bi = perm[start:start + 256]
                    optimizer.zero_grad()
                    loss = criterion(self.model(Xa[bi], Xb[bi]), yd[bi])
                    loss.backward()
                    optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    vl = criterion(self.model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
                if vl < best_vl:
                    best_vl = vl
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= 15:
                        print(f"    Pretrain early stop at epoch {epoch + 1}")
                        break
                if (epoch + 1) % 25 == 0:
                    print(f"    Pretrain epoch {epoch + 1}: val_loss={vl:.4f}")

            self.model.load_state_dict(best_state)
            del Xa, Xb, yd
            gc.collect()
            pretrained = True
            print("    Kinase pretraining complete")
        else:
            print("    No kinase pairs file, training from scratch")
            all_embs = np.array([self.fp_cache[s] for s in train_smiles])
            self.scaler = StandardScaler()
            self.scaler.fit(all_embs)

            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.model = FiLMDeltaMLP(
                input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2
            )
            del all_embs

        # Fine-tune on ZAP70 all-pairs
        print(f"    Fine-tuning on {n} ZAP70 molecules (all-pairs)...")
        rows_a, rows_b, deltas = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rows_a.append(train_smiles[i])
                rows_b.append(train_smiles[j])
                deltas.append(float(train_pIC50[j] - train_pIC50[i]))

        emb_a = np.array([self.fp_cache[s] for s in rows_a])
        emb_b = np.array([self.fp_cache[s] for s in rows_b])
        Xa = torch.FloatTensor(self.scaler.transform(emb_a))
        Xb = torch.FloatTensor(self.scaler.transform(emb_b))
        yd = torch.FloatTensor(np.array(deltas, dtype=np.float32))
        del emb_a, emb_b, rows_a, rows_b, deltas
        gc.collect()

        n_val = max(len(Xa) // 10, 1)
        perm_all = np.random.RandomState(self.seed).permutation(len(Xa))
        val_idx = perm_all[:n_val]
        train_idx = perm_all[n_val:]

        ft_model = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(ft_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_vl, best_state, wait = float("inf"), None, 0
        for epoch in range(50):
            ft_model.train()
            perm = np.random.permutation(len(train_idx))
            for start in range(0, len(perm), 256):
                bi = train_idx[perm[start:start + 256]]
                optimizer.zero_grad()
                loss = criterion(ft_model(Xa[bi], Xb[bi]), yd[bi])
                loss.backward()
                optimizer.step()

            ft_model.eval()
            with torch.no_grad():
                vl = criterion(ft_model(Xa[val_idx], Xb[val_idx]), yd[val_idx]).item()
            if vl < best_vl:
                best_vl = vl
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 15:
                    print(f"    Fine-tune early stop at epoch {epoch + 1}")
                    break
            if (epoch + 1) % 10 == 0:
                print(f"    Fine-tune epoch {epoch + 1}: val_loss={vl:.4f}")

        if best_state:
            ft_model.load_state_dict(best_state)
        ft_model.eval()
        self.model = ft_model

        del Xa, Xb, yd
        gc.collect()

        # Store anchors
        self.anchor_smiles = list(train_smiles)
        self.anchor_pIC50 = np.array(train_pIC50, dtype=np.float32)
        self.anchor_embs = torch.FloatTensor(
            self.scaler.transform(np.array([self.fp_cache[s] for s in train_smiles]))
        )

        print(f"    FiLMDelta trained (seed={self.seed}), {n} anchors")
        return pretrained

    def predict(self, smiles_list, batch_size=200):
        """Anchor-based absolute prediction: median(pIC50_i + delta(i->j))."""
        self._build_fp_cache(smiles_list)
        self.model.eval()

        n_targets = len(smiles_list)
        n_anchors = len(self.anchor_smiles)

        target_embs_raw = np.array([self.fp_cache[s] for s in smiles_list])
        target_embs = torch.FloatTensor(self.scaler.transform(target_embs_raw))
        del target_embs_raw

        predictions = np.zeros(n_targets)
        uncertainties = np.zeros(n_targets)

        for start in range(0, n_targets, batch_size):
            end = min(start + batch_size, n_targets)
            batch_preds = np.zeros((end - start, n_anchors))

            for j_local, j_global in enumerate(range(start, end)):
                target_expanded = target_embs[j_global:j_global + 1].expand(n_anchors, -1)
                with torch.no_grad():
                    deltas = self.model(self.anchor_embs, target_expanded).numpy()
                batch_preds[j_local] = self.anchor_pIC50 + deltas

            # Use median for robustness (less sensitive to anchor outliers)
            predictions[start:end] = np.median(batch_preds, axis=1)
            uncertainties[start:end] = np.std(batch_preds, axis=1)

            if (end % 1000 == 0 or end == n_targets) and end > start:
                print(f"      Scored {end}/{n_targets} candidates...")

        del target_embs
        return predictions, uncertainties


def score_all_candidates(smiles_list, mol_data, mol1_smi, n_seeds=N_SEEDS):
    """Score all candidates with FiLMDelta (multi-seed) + properties."""

    # Remove invalid molecules first
    print(f"\n  Validating {len(smiles_list)} candidates...")
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(canonicalize(smi))
    valid_smiles = list(set(s for s in valid_smiles if s is not None))
    print(f"  Valid unique SMILES: {len(valid_smiles)}")

    # Compute properties
    print("  Computing molecular properties...")
    props_list = []
    for idx, smi in enumerate(valid_smiles):
        props = compute_mol_properties(smi)
        if props is None:
            props = {}
        props["smiles"] = smi
        props_list.append(props)
        if (idx + 1) % 5000 == 0:
            print(f"    Properties computed for {idx + 1}/{len(valid_smiles)}...")

    props_df = pd.DataFrame(props_list)

    # PAINS filter
    print("  Checking PAINS alerts...")
    pains_free = []
    for idx, smi in enumerate(valid_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and check_pains(mol):
            pains_free.append(smi)
        if (idx + 1) % 5000 == 0:
            print(f"    PAINS checked {idx + 1}/{len(valid_smiles)}...")
    print(f"  PAINS-free: {len(pains_free)} / {len(valid_smiles)}")

    # Apply filters
    filtered = props_df[
        (props_df["smiles"].isin(pains_free)) &
        (props_df["QED"] > 0.3) &
        (props_df["MW"] < 600)
    ].copy()
    filtered_smiles = filtered["smiles"].tolist()
    print(f"  After filters (PAINS-free, QED>0.3, MW<600): {len(filtered_smiles)}")

    if len(filtered_smiles) == 0:
        print("  WARNING: No molecules passed filters!")
        return pd.DataFrame()

    # FiLMDelta scoring with multiple seeds
    print(f"\n  FiLMDelta scoring ({n_seeds} seeds)...")
    all_preds = []
    all_uncerts = []

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed + 1}/{n_seeds} ---")
        scorer = FiLMDeltaScorer(mol_data, seed=seed * 42 + 1)
        scorer.train()
        preds, uncerts = scorer.predict(filtered_smiles)
        all_preds.append(preds)
        all_uncerts.append(uncerts)
        del scorer
        gc.collect()

    # Ensemble predictions
    all_preds = np.array(all_preds)
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_std = np.std(all_preds, axis=0)
    anchor_uncert = np.mean(all_uncerts, axis=0)

    # Tanimoto similarity
    print("\n  Computing Tanimoto similarities...")
    mol1_fp = compute_fingerprints([mol1_smi])[0]
    train_fps = compute_fingerprints(mol_data["smiles"].tolist())
    cand_fps = compute_fingerprints(filtered_smiles)

    sim_to_mol1 = tanimoto_kernel_matrix(cand_fps, mol1_fp.reshape(1, -1))[:, 0]
    sim_to_train = np.max(tanimoto_kernel_matrix(cand_fps, train_fps), axis=1)

    # Build final results DataFrame
    results_df = filtered[filtered["smiles"].isin(filtered_smiles)].copy()
    results_df = results_df.set_index("smiles").loc[filtered_smiles].reset_index()

    results_df["film_pIC50"] = ensemble_pred
    results_df["film_std"] = ensemble_std
    results_df["anchor_uncertainty"] = anchor_uncert
    results_df["tanimoto_to_mol1"] = sim_to_mol1
    results_df["tanimoto_to_train_max"] = sim_to_train

    # Sort by predicted pIC50
    results_df = results_df.sort_values("film_pIC50", ascending=False).reset_index(drop=True)

    print(f"\n  Scoring complete: {len(results_df)} candidates")
    print(f"  Predicted pIC50 range: {results_df['film_pIC50'].min():.2f} - {results_df['film_pIC50'].max():.2f}")
    print(f"  Mean pIC50: {results_df['film_pIC50'].mean():.2f}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════════════

def mol_to_svg(smi, size=(250, 200)):
    """Convert SMILES to SVG image string."""
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "<p>Invalid</p>"
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return "<p>Render error</p>"


def generate_report(results_df, generation_stats, mol1_smi, mol_data):
    """Generate comprehensive HTML report."""
    with timer("Generating HTML report"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Mol 1 properties
        mol1_props = compute_mol_properties(mol1_smi)
        mol1_svg = mol_to_svg(mol1_smi, size=(350, 280))

        # Top 50 candidates
        top50 = results_df.head(50).copy()

        # Per-method analysis
        method_stats = {}
        if "source" in results_df.columns:
            for method in results_df["source"].unique():
                subset = results_df[results_df["source"] == method]
                method_stats[method] = {
                    "count": len(subset),
                    "mean_pIC50": float(subset["film_pIC50"].mean()),
                    "max_pIC50": float(subset["film_pIC50"].max()),
                    "mean_sim_mol1": float(subset["tanimoto_to_mol1"].mean()),
                    "mean_QED": float(subset["QED"].mean()) if "QED" in subset.columns else 0,
                }

        # Build HTML
        html = []
        html.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mol 1 Expansion Pipeline Report</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }
h3 { color: #34495e; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.mol1-card { display: flex; align-items: flex-start; gap: 30px; }
.mol1-svg { flex-shrink: 0; }
.mol1-info { flex-grow: 1; }
.mol1-info table { border-collapse: collapse; width: 100%; }
.mol1-info td { padding: 4px 12px; border-bottom: 1px solid #eee; }
.mol1-info td:first-child { font-weight: bold; color: #555; width: 140px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
.stat-box { background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; border-left: 4px solid #3498db; }
.stat-box .value { font-size: 24px; font-weight: bold; color: #2c3e50; }
.stat-box .label { font-size: 12px; color: #777; margin-top: 5px; }
table.results { border-collapse: collapse; width: 100%; font-size: 13px; }
table.results th { background: #2c3e50; color: white; padding: 8px 6px; text-align: left; position: sticky; top: 0; }
table.results td { padding: 6px; border-bottom: 1px solid #eee; vertical-align: middle; }
table.results tr:hover { background: #f0f8ff; }
table.results tr:nth-child(even) { background: #fafafa; }
table.results tr:hover { background: #e8f4fd; }
.smi { font-family: monospace; font-size: 11px; word-break: break-all; max-width: 300px; }
.good { color: #27ae60; font-weight: bold; }
.warn { color: #e67e22; }
.bad { color: #e74c3c; }
.method-table { border-collapse: collapse; width: 100%; }
.method-table th, .method-table td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
.method-table th { background: #34495e; color: white; }
.method-table tr:nth-child(even) { background: #f9f9f9; }
.mol-svg { display: inline-block; }
.mol-svg svg { width: 180px; height: 140px; }
.top10-card { background: #f0fff0; border-left: 4px solid #27ae60; padding: 15px; margin: 10px 0; border-radius: 4px; }
.top10-rank { font-size: 20px; font-weight: bold; color: #27ae60; float: left; margin-right: 15px; }
.footer { margin-top: 40px; padding: 20px; background: #2c3e50; color: #ecf0f1; border-radius: 8px; text-align: center; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
""")

        # Title
        html.append(f"""
<h1>Mol 1 Expansion Pipeline Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Target: ZAP70 (CHEMBL2803) | Seed: Mol 1 (pyrimidine-piperazine acrylamide)</p>
""")

        # Summary stats
        total_generated = sum(v for k, v in generation_stats.items() if k.startswith("n_"))
        n_filtered = len(results_df)
        html.append(f"""
<h2>1. Summary</h2>
<div class="card">
<div class="stats-grid">
<div class="stat-box"><div class="value">{generation_stats.get('n_mmp', 0):,}</div><div class="label">MMP Enumeration</div></div>
<div class="stat-box"><div class="value">{generation_stats.get('n_crem', 0):,}</div><div class="label">CReM Generation</div></div>
<div class="stat-box"><div class="value">{generation_stats.get('n_brics', 0):,}</div><div class="label">BRICS Recombination</div></div>
<div class="stat-box"><div class="value">{generation_stats.get('n_total_raw', 0):,}</div><div class="label">Total Raw</div></div>
<div class="stat-box"><div class="value">{generation_stats.get('n_unique', 0):,}</div><div class="label">Unique Valid</div></div>
<div class="stat-box"><div class="value">{n_filtered:,}</div><div class="label">After Filters</div></div>
</div>
<p style="margin-top:15px; color:#666;">
Filters applied: valid RDKit parse, PAINS-free, QED &gt; 0.3, MW &lt; 600, deduplicated by canonical SMILES.
</p>
</div>
""")

        # Mol 1 card
        html.append(f"""
<h2>2. Mol 1 (Seed Compound)</h2>
<div class="card mol1-card">
<div class="mol1-svg">{mol1_svg}</div>
<div class="mol1-info">
<table>
<tr><td>SMILES</td><td class="smi">{mol1_smi}</td></tr>
<tr><td>MW</td><td>{mol1_props.get('MW', 'N/A'):.1f}</td></tr>
<tr><td>LogP</td><td>{mol1_props.get('LogP', 'N/A'):.2f}</td></tr>
<tr><td>HBA / HBD</td><td>{mol1_props.get('HBA', 'N/A')} / {mol1_props.get('HBD', 'N/A')}</td></tr>
<tr><td>TPSA</td><td>{mol1_props.get('TPSA', 'N/A'):.1f}</td></tr>
<tr><td>Rot. Bonds</td><td>{mol1_props.get('RotBonds', 'N/A')}</td></tr>
<tr><td>QED</td><td>{mol1_props.get('QED', 'N/A'):.3f}</td></tr>
<tr><td>SA Score</td><td>{mol1_props.get('SA_Score', 'N/A')}</td></tr>
<tr><td>Heavy Atoms</td><td>{mol1_props.get('HeavyAtoms', 'N/A')}</td></tr>
<tr><td>Rings</td><td>{mol1_props.get('Rings', 'N/A')}</td></tr>
</table>
</div>
</div>
""")

        # Top 50 candidates table
        html.append("""
<h2>3. Top 50 Candidates (by FiLMDelta pIC50)</h2>
<div class="card" style="overflow-x: auto;">
<table class="results">
<thead>
<tr>
<th>Rank</th>
<th>Structure</th>
<th>SMILES</th>
<th>pIC50</th>
<th>Std</th>
<th>Sim(Mol18)</th>
<th>Sim(Train)</th>
<th>MW</th>
<th>LogP</th>
<th>QED</th>
<th>HBA</th>
<th>HBD</th>
<th>TPSA</th>
<th>Source</th>
</tr>
</thead>
<tbody>
""")

        for i, row in top50.iterrows():
            rank = i + 1
            smi = row["smiles"]
            svg = mol_to_svg(smi, size=(180, 140))
            pic50 = row.get("film_pIC50", 0)
            std = row.get("film_std", 0)
            sim_m18 = row.get("tanimoto_to_mol1", 0)
            sim_tr = row.get("tanimoto_to_train_max", 0)

            pic50_class = "good" if pic50 >= 7.0 else ("warn" if pic50 >= 6.0 else "")
            source = row.get("source", "unknown")

            html.append(f"""<tr>
<td><b>{rank}</b></td>
<td class="mol-svg">{svg}</td>
<td class="smi">{smi}</td>
<td class="{pic50_class}">{pic50:.2f}</td>
<td>{std:.2f}</td>
<td>{sim_m18:.3f}</td>
<td>{sim_tr:.3f}</td>
<td>{row.get('MW', 0):.0f}</td>
<td>{row.get('LogP', 0):.1f}</td>
<td>{row.get('QED', 0):.2f}</td>
<td>{row.get('HBA', 0):.0f}</td>
<td>{row.get('HBD', 0):.0f}</td>
<td>{row.get('TPSA', 0):.0f}</td>
<td>{source}</td>
</tr>""")

        html.append("</tbody></table></div>")

        # Per-method analysis
        html.append("""<h2>4. Per-Method Analysis</h2>""")
        if method_stats:
            html.append("""<div class="card"><table class="method-table">
<thead><tr><th>Method</th><th>Count</th><th>Mean pIC50</th><th>Max pIC50</th><th>Mean Sim(Mol18)</th><th>Mean QED</th></tr></thead><tbody>""")
            for method, stats in sorted(method_stats.items(), key=lambda x: -x[1]["mean_pIC50"]):
                html.append(f"""<tr>
<td>{method}</td>
<td>{stats['count']:,}</td>
<td>{stats['mean_pIC50']:.3f}</td>
<td class="{'good' if stats['max_pIC50'] >= 7.0 else ''}">{stats['max_pIC50']:.3f}</td>
<td>{stats['mean_sim_mol1']:.3f}</td>
<td>{stats['mean_QED']:.3f}</td>
</tr>""")
            html.append("</tbody></table></div>")
        else:
            html.append('<div class="card"><p>No per-method source tracking available.</p></div>')

        # Score distribution summary
        if len(results_df) > 0:
            html.append("""<h2>5. Score Distribution</h2><div class="card">""")
            bins = [(9, float("inf")), (8, 9), (7, 8), (6, 7), (5, 6), (0, 5)]
            html.append("<table class='method-table'><thead><tr><th>pIC50 Range</th><th>Count</th><th>%</th></tr></thead><tbody>")
            for lo, hi in bins:
                count = len(results_df[(results_df["film_pIC50"] >= lo) & (results_df["film_pIC50"] < hi)])
                pct = 100 * count / len(results_df)
                label = f"{lo}+" if hi == float("inf") else f"{lo}-{hi}"
                html.append(f"<tr><td>{label}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>")
            html.append("</tbody></table></div>")

        # Top 10 recommendations
        html.append("""<h2>6. Top 10 Recommendations</h2>""")
        top10 = results_df.head(10)
        for i, row in top10.iterrows():
            rank = i + 1
            smi = row["smiles"]
            svg = mol_to_svg(smi, size=(220, 170))
            pic50 = row.get("film_pIC50", 0)
            std = row.get("film_std", 0)
            sim = row.get("tanimoto_to_mol1", 0)
            source = row.get("source", "unknown")

            # Generate rationale
            rationale_parts = []
            if pic50 >= 7.5:
                rationale_parts.append("Predicted highly potent (pIC50 >= 7.5)")
            elif pic50 >= 7.0:
                rationale_parts.append("Predicted potent (pIC50 >= 7.0)")
            else:
                rationale_parts.append(f"Moderate predicted potency (pIC50={pic50:.2f})")

            if sim >= 0.7:
                rationale_parts.append(f"Very similar to Mol 1 (Tc={sim:.2f})")
            elif sim >= 0.4:
                rationale_parts.append(f"Moderate similarity to Mol 1 (Tc={sim:.2f})")
            else:
                rationale_parts.append(f"Novel scaffold vs Mol 1 (Tc={sim:.2f})")

            if std < 0.3:
                rationale_parts.append("Low prediction uncertainty")
            elif std > 0.5:
                rationale_parts.append("High uncertainty — experimental validation critical")

            qed = row.get("QED", 0)
            if qed >= 0.6:
                rationale_parts.append(f"Good drug-likeness (QED={qed:.2f})")

            rationale = ". ".join(rationale_parts) + "."

            html.append(f"""
<div class="top10-card">
<div class="top10-rank">#{rank}</div>
<div style="display:flex; gap:20px; align-items:flex-start;">
<div>{svg}</div>
<div>
<p><b>SMILES:</b> <span class="smi">{smi}</span></p>
<p><b>pIC50:</b> <span class="good">{pic50:.2f}</span> +/- {std:.2f} | <b>Source:</b> {source}</p>
<p><b>MW:</b> {row.get('MW', 0):.0f} | <b>LogP:</b> {row.get('LogP', 0):.1f} | <b>QED:</b> {qed:.2f} | <b>TPSA:</b> {row.get('TPSA', 0):.0f}</p>
<p><em>{rationale}</em></p>
</div>
</div>
</div>
""")

        # Footer
        html.append(f"""
<div class="footer">
<p>Mol 1 Expansion Pipeline | Edit-Small-Mol Project | {datetime.now().strftime('%Y-%m-%d')}</p>
<p>Scoring: FiLMDelta anchor-based prediction ({N_SEEDS} seeds) | Embedder: Morgan FP 2048d</p>
<p>Training set: {len(mol_data)} ZAP70 molecules with known pIC50</p>
</div>
</div>
</body>
</html>""")

        report_html = "\n".join(html)
        with open(REPORT_FILE, "w") as f:
            f.write(report_html)

        print(f"  Report written to {REPORT_FILE}")
        return str(REPORT_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MOL 1 EXPANSION PIPELINE")
    print(f"Seed: {MOL1_SMILES}")
    print(f"Target: ZAP70 (CHEMBL2803)")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "mol1_smiles": MOL1_SMILES,
        "target": TARGET_ID,
        "timestamp": datetime.now().isoformat(),
        "generation": {},
        "scoring": {},
    }

    # Load ZAP70 data
    print("\n--- Loading ZAP70 Data ---")
    mol_data = load_zap70_molecules()
    zap70_pairs = load_zap70_pairs()

    # Track sources for each generated molecule
    source_map = {}  # smiles -> source method

    # ─────────────────────────────────────────────────────
    # Phase 1: Conservative Expansion (with caching)
    # ─────────────────────────────────────────────────────
    phase1_cache = RESULTS_DIR / "phase1_cache.json"
    if phase1_cache.exists():
        print("\n  Loading Phase 1 results from cache...")
        with open(phase1_cache) as f:
            cache = json.load(f)
        all_generated = set(cache["smiles"])
        source_map = cache["source_map"]
        results["generation"] = cache["stats"]
        print(f"  Loaded {len(all_generated):,} molecules from Phase 1 cache")
    else:
        print("\n" + "=" * 70)
        print("PHASE 1: CONSERVATIVE EXPANSION")
        print("=" * 70)

        all_generated = set()

        # 1a. MMP Enumeration
        mmp_mols = phase1_mmp_enumeration(MOL1_SMILES, zap70_pairs)
        for smi in mmp_mols:
            source_map[smi] = "MMP"
        all_generated.update(mmp_mols)
        results["generation"]["n_mmp"] = len(mmp_mols)
        del mmp_mols
        gc.collect()

        # 1b. CReM Generation
        crem_mols = phase1_crem_generation(MOL1_SMILES)
        for smi in crem_mols:
            if smi not in source_map:
                source_map[smi] = "CReM"
        all_generated.update(crem_mols)
        results["generation"]["n_crem"] = len(crem_mols)
        del crem_mols
        gc.collect()

        # 1c. BRICS Recombination
        brics_mols = phase1_brics_recombination(MOL1_SMILES, mol_data)
        for smi in brics_mols:
            if smi not in source_map:
                source_map[smi] = "BRICS"
        all_generated.update(brics_mols)
        results["generation"]["n_brics"] = len(brics_mols)
        del brics_mols
        gc.collect()

        # Remove training molecules and Mol 1 itself
        train_set = set(canonicalize(s) for s in mol_data["smiles"])
        mol1_can = canonicalize(MOL1_SMILES)
        all_generated -= train_set
        all_generated.discard(mol1_can)
        all_generated.discard(None)

        results["generation"]["n_total_raw"] = (
            results["generation"]["n_mmp"] +
            results["generation"]["n_crem"] +
            results["generation"]["n_brics"]
        )
        results["generation"]["n_unique"] = len(all_generated)

        # Cache Phase 1 results
        with open(phase1_cache, "w") as f:
            json.dump({
                "smiles": list(all_generated),
                "source_map": source_map,
                "stats": results["generation"],
            }, f)
        print(f"  Phase 1 cached to {phase1_cache}")

        print(f"\n  Phase 1 total unique molecules: {len(all_generated):,}")
        print(f"    MMP: {results['generation']['n_mmp']}")
        print(f"    CReM: {results['generation']['n_crem']}")
        print(f"    BRICS: {results['generation']['n_brics']}")

    # ─────────────────────────────────────────────────────
    # Phase 2: REINVENT4 Config Generation (no execution)
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: DIRECTED GENERATION (REINVENT4)")
    print("=" * 70)

    reinvent_info = phase2_reinvent_configs(MOL1_SMILES)
    results["generation"]["reinvent"] = reinvent_info

    # ─────────────────────────────────────────────────────
    # Phase 3: Scoring & Ranking
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: SCORING & RANKING")
    print("=" * 70)

    smiles_list = list(all_generated)
    scored_df = score_all_candidates(smiles_list, mol_data, MOL1_SMILES, n_seeds=N_SEEDS)

    # Add source information
    if len(scored_df) > 0:
        scored_df["source"] = scored_df["smiles"].map(
            lambda s: source_map.get(s, "unknown")
        )

        # Save scored results as CSV
        scored_csv = RESULTS_DIR / "scored_candidates.csv"
        scored_df.to_csv(scored_csv, index=False)
        print(f"  Scored candidates saved to {scored_csv}")

        # Update results
        results["scoring"] = {
            "n_scored": len(scored_df),
            "n_seeds": N_SEEDS,
            "pIC50_mean": float(scored_df["film_pIC50"].mean()),
            "pIC50_std": float(scored_df["film_pIC50"].std()),
            "pIC50_max": float(scored_df["film_pIC50"].max()),
            "pIC50_min": float(scored_df["film_pIC50"].min()),
            "n_potent_7": int((scored_df["film_pIC50"] >= 7.0).sum()),
            "n_potent_8": int((scored_df["film_pIC50"] >= 8.0).sum()),
            "top10_smiles": scored_df.head(10)["smiles"].tolist(),
            "top10_pIC50": scored_df.head(10)["film_pIC50"].tolist(),
        }

        # Per-source stats
        source_stats = {}
        for src in scored_df["source"].unique():
            subset = scored_df[scored_df["source"] == src]
            source_stats[src] = {
                "count": len(subset),
                "mean_pIC50": float(subset["film_pIC50"].mean()),
                "max_pIC50": float(subset["film_pIC50"].max()),
            }
        results["scoring"]["per_source"] = source_stats

    save_results(results)

    # ─────────────────────────────────────────────────────
    # Generate HTML Report
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    if len(scored_df) > 0:
        report_path = generate_report(scored_df, results["generation"], MOL1_SMILES, mol_data)
        results["report_path"] = report_path
    else:
        print("  No scored candidates — skipping report generation")

    save_results(results)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total generated: {results['generation'].get('n_unique', 0):,}")
    print(f"  After filters: {results['scoring'].get('n_scored', 0):,}")
    print(f"  Predicted potent (pIC50 >= 7.0): {results['scoring'].get('n_potent_7', 0):,}")
    print(f"  Top predicted pIC50: {results['scoring'].get('pIC50_max', 0):.2f}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
