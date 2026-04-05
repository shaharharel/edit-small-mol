#!/usr/bin/env python3
"""
ZAP70 1M Molecule Screening + Full Analysis Pipeline.

Generates ~1M molecules, scores ALL with dual model (XGB + FiLMDelta),
saves per-molecule scores to parquet, then runs comprehensive analyses:
1. HTML report
2. Chemical diversity / series analysis
3. Drug-likeness filtering
4. XGB vs FiLMDelta agreement analysis

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_zap70_screen_and_analyze.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import os
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

# Set seeds FIRST for reproducibility
np.random.seed(42)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

import selfies as sf

# Save real MPS check, then import modules that disable it
_real_mps = torch.backends.mps.is_available
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints, train_xgboost, train_rf, N_JOBS
)
from experiments.run_zap70_v5 import BEST_XGB_PARAMS, BEST_RF_PARAMS
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
torch.backends.mps.is_available = _real_mps

PROJECT_ROOT = Path(__file__).parent.parent
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
CREM_DB_PATH = PROJECT_ROOT / "data" / "crem_db" / "chembl33_sa2_f5.db"
CHEMBL_DB = PROJECT_ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

PARQUET_FILE = RESULTS_DIR / "zap70_1M_scored.parquet"
RESULTS_JSON = RESULTS_DIR / "zap70_1M_screening_results.json"
REPORT_HTML = RESULTS_DIR / "zap70_1M_screening_report.html"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def timer(msg):
    class Timer:
        def __init__(self, m): self.msg = m
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n>>> {self.msg}...", flush=True)
            return self
        def __exit__(self, *a):
            self.elapsed = time.time() - self.t0
            print(f"    [{self.msg}] {self.elapsed:.1f}s", flush=True)
    return Timer(msg)


# =============================================================================
# GENERATION METHODS
# =============================================================================

def gen_mmp_enumeration(smiles_list, pIC50):
    """Apply beneficial MMP edits to all training molecules."""
    edit_patterns = [
        ("[cH:1]", "[c:1]F"), ("[cH:1]", "[c:1]Cl"), ("[cH:1]", "[c:1]C"),
        ("[cH:1]", "[c:1]OC"), ("[cH:1]", "[c:1]C#N"), ("[cH:1]", "[c:1]N"),
        ("[CH3:1]", "[C:1](F)(F)F"), ("[OH:1]", "[O:1]C"), ("[NH:1]", "[N:1]C"),
        ("[cH:1]", "[c:1]C(=O)N"), ("[cH:1]", "[c:1]S(=O)(=O)C"),
        ("[cH:1]", "[c:1]C(F)(F)F"), ("[cH:1]", "[c:1]NC(=O)C"),
        ("[CH3:1]", "[C:1]CC"), ("[F:1]", "[Cl:1]"), ("[Cl:1]", "[F:1]"),
        ("[NH2:1]", "[N:1](C)C"),
    ]
    generated = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for pat_smi, rep_smi in edit_patterns:
            try:
                pat = Chem.MolFromSmarts(pat_smi)
                rep = Chem.MolFromSmarts(rep_smi)
                if not pat or not rep:
                    continue
                products = AllChem.ReplaceSubstructs(mol, pat, rep)
                for prod in products:
                    try:
                        Chem.SanitizeMol(prod)
                        can = Chem.MolToSmiles(prod)
                        if Chem.MolFromSmiles(can):
                            generated.add(can)
                    except Exception:
                        pass
            except Exception:
                pass
    return generated


def gen_brics(smiles_list, max_build=100000):
    """BRICS decompose + recombine."""
    frags = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags.update(BRICS.BRICSDecompose(mol))
            except Exception:
                pass
    frag_mols = [Chem.MolFromSmiles(f) for f in frags]
    frag_mols = [m for m in frag_mols if m is not None]
    print(f"    {len(frag_mols)} fragment mols", flush=True)
    generated = set()
    try:
        for i, mol in enumerate(BRICS.BRICSBuild(frag_mols)):
            if i >= max_build:
                break
            try:
                Chem.SanitizeMol(mol)
                can = Chem.MolToSmiles(mol)
                check = Chem.MolFromSmiles(can)
                if check and 10 <= check.GetNumHeavyAtoms() <= 80:
                    generated.add(can)
            except Exception:
                pass
            if (i + 1) % 20000 == 0:
                print(f"    BRICS: {i+1}/{max_build} built, {len(generated)} valid", flush=True)
    except Exception as e:
        print(f"    BRICS error: {e}")
    return generated


def gen_rgroup(smiles_list, pIC50):
    """R-group enumeration on top scaffolds."""
    R_GROUPS = [
        "C", "CC", "CCC", "C(C)C", "F", "Cl", "Br", "C(F)(F)F",
        "OC", "OCC", "N", "NC", "NC(C)C", "O", "C#N", "C(=O)C",
        "C(=O)OC", "S(=O)(=O)C", "c1ccccc1", "c1ccncc1",
        "C1CCOCC1", "C1CCNCC1", "C1CCNC1",
    ]
    scaffolds = {}
    for smi, y in zip(smiles_list, pIC50):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
            scaf_smi = Chem.MolToSmiles(scaf)
            scaffolds.setdefault(scaf_smi, []).append((smi, y))
        except Exception:
            pass

    top_scaffolds = sorted(scaffolds.items(), key=lambda x: -len(x[1]))[:15]
    generated = set()
    for scaf_smi, mols in top_scaffolds:
        for ref_smi, _ in mols:
            ref_mol = Chem.MolFromSmiles(ref_smi)
            if ref_mol is None:
                continue
            for rg in R_GROUPS:
                rg_mol = Chem.MolFromSmiles(rg)
                if rg_mol is None:
                    continue
                # Replace H on aromatic atoms
                pat = Chem.MolFromSmarts("[cH]")
                if pat and ref_mol.HasSubstructMatch(pat):
                    try:
                        prods = AllChem.ReplaceSubstructs(ref_mol, pat, rg_mol)
                        if prods:
                            prod = prods[0]
                            Chem.SanitizeMol(prod)
                            can = Chem.MolToSmiles(prod)
                            if Chem.MolFromSmiles(can):
                                generated.add(can)
                    except Exception:
                        pass
    return generated


def gen_kinase_crosspolination():
    """Get kinase compounds from ChEMBL."""
    if not CHEMBL_DB.exists():
        print("    ChEMBL DB not found, skipping kinase cross-pollination")
        return set()

    import sqlite3
    conn = sqlite3.connect(str(CHEMBL_DB))
    try:
        query = """
            SELECT DISTINCT cs.canonical_smiles
            FROM activities a
            JOIN assays ass ON a.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN molecule_dictionary md ON a.molregno = md.molregno
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE td.pref_name LIKE '%kinase%'
            AND a.pchembl_value >= 6.0
            AND td.chembl_id != 'CHEMBL2803'
            AND cs.canonical_smiles IS NOT NULL
            LIMIT 100000
        """
        df = pd.read_sql(query, conn)
        generated = set()
        for smi in df["canonical_smiles"]:
            can = canonicalize(smi)
            if can:
                generated.add(can)
        return generated
    except Exception as e:
        print(f"    ChEMBL query error: {e}")
        return set()
    finally:
        conn.close()


def gen_crem(smiles_list, pIC50, existing, max_total=300000):
    """CReM mutations with multiple configs."""
    if not CREM_DB_PATH.exists():
        print("    CReM DB not found, skipping")
        return set()
    try:
        from crem.crem import mutate_mol
    except ImportError:
        print("    CReM not installed, skipping")
        return set()

    db = str(CREM_DB_PATH)
    generated = set()
    configs = [
        {"radius": 2, "min_size": 0, "max_size": 8, "max_replacements": 500},
        {"radius": 3, "min_size": 0, "max_size": 10, "max_replacements": 1000},
        {"radius": 3, "min_size": 1, "max_size": 12, "max_replacements": 500},
    ]
    for ci, cfg in enumerate(configs):
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                for new_smi in mutate_mol(mol, db_name=db, return_mol=False, **cfg):
                    can = canonicalize(new_smi)
                    if can and can not in existing:
                        generated.add(can)
            except Exception:
                pass
            if len(generated) >= max_total:
                break
        print(f"    CReM config {ci+1}/{len(configs)}: {len(generated):,} total")
        if len(generated) >= max_total:
            break
    return generated


def gen_selfies(smiles_list, n_per_mol=2000, max_total=500000):
    """SELFIES-based random perturbation."""
    alphabet = list(sf.get_semantic_robust_alphabet())
    generated = set()
    for i, smi in enumerate(smiles_list):
        try:
            selfies_str = sf.encoder(smi)
            if selfies_str is None:
                continue
            tokens = list(sf.split_selfies(selfies_str))
            if len(tokens) < 3:
                continue
        except Exception:
            continue

        count = 0
        for _ in range(n_per_mol * 5):
            if count >= n_per_mol:
                break
            new_tokens = list(tokens)
            for _ in range(np.random.randint(1, 4)):
                op = np.random.random()
                if op < 0.4 and len(new_tokens) > 3:
                    idx = np.random.randint(len(new_tokens))
                    new_tokens[idx] = np.random.choice(alphabet)
                elif op < 0.7:
                    idx = np.random.randint(len(new_tokens) + 1)
                    new_tokens.insert(idx, np.random.choice(alphabet))
                elif len(new_tokens) > 4:
                    new_tokens.pop(np.random.randint(len(new_tokens)))

            try:
                new_smi = sf.decoder("".join(new_tokens))
                if new_smi is None:
                    continue
                can = canonicalize(new_smi)
                if can and Chem.MolFromSmiles(can):
                    mol = Chem.MolFromSmiles(can)
                    if 8 <= mol.GetNumHeavyAtoms() <= 80:
                        generated.add(can)
                        count += 1
            except Exception:
                pass

            if len(generated) >= max_total:
                break
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(smiles_list)} seeds, {len(generated):,} unique")
        if len(generated) >= max_total:
            break
    return generated


# =============================================================================
# MODEL TRAINING + SCORING
# =============================================================================

class DualScorer:
    def __init__(self, smiles_list, pIC50, fp_cache):
        self.train_smiles = smiles_list
        self.train_pIC50 = pIC50
        self.fp_cache = fp_cache
        self.xgb_models = {}
        self.film_model = None
        self.scaler = None
        self.anchor_embs = None

    def train(self):
        # XGB ensemble (5 models)
        with timer("Training XGB ensemble (5 models)"):
            fp_types = {
                "morgan": ("morgan", 2), "atompair": ("atompair", 2),
                "rdkit": ("rdkit", 2), "ecfp6": ("morgan", 3),
            }
            X_train = {}
            for name, (ft, r) in fp_types.items():
                X_train[name] = compute_fingerprints(self.train_smiles, ft, radius=r, n_bits=2048)

            y = self.train_pIC50
            for model_name, fp_key, train_fn, params in [
                ("xgb_atompair", "atompair", train_xgboost, BEST_XGB_PARAMS),
                ("rf_rdkit", "rdkit", train_rf, BEST_RF_PARAMS),
                ("xgb_ecfp6", "ecfp6", train_xgboost, BEST_XGB_PARAMS),
                ("xgb_morgan", "morgan", train_xgboost, BEST_XGB_PARAMS),
                ("rf_atompair", "atompair", train_rf, BEST_RF_PARAMS),
            ]:
                _, model = train_fn(X_train[fp_key], y, X_train[fp_key], **params)
                self.xgb_models[model_name] = (fp_key, model)
            print(f"    {len(self.xgb_models)} models trained")

        # FiLMDelta + Kinase PT
        with timer("Training FiLMDelta + Kinase PT"):
            kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
            all_kinase_smi = list(set(kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()))
            extra = [s for s in all_kinase_smi if s not in self.fp_cache]
            if extra:
                fps = compute_fingerprints(extra, "morgan", radius=2, n_bits=2048)
                for i, s in enumerate(extra):
                    self.fp_cache[s] = fps[i]

            mask = kinase_pairs["mol_a"].apply(lambda s: s in self.fp_cache) & \
                   kinase_pairs["mol_b"].apply(lambda s: s in self.fp_cache)
            kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
            print(f"    Kinase pairs: {len(kinase_pairs):,}")

            ea = np.array([self.fp_cache[s] for s in kinase_pairs["mol_a"]])
            eb = np.array([self.fp_cache[s] for s in kinase_pairs["mol_b"]])
            d = kinase_pairs["delta"].values.astype(np.float32)

            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack([ea, eb]))
            Xa, Xb, yd = (torch.FloatTensor(self.scaler.transform(ea)),
                          torch.FloatTensor(self.scaler.transform(eb)),
                          torch.FloatTensor(d))
            del ea, eb, d, kinase_pairs; gc.collect()

            # Pretrain
            model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.MSELoss()
            n_val = len(Xa) // 10
            best_vl, best_st, wait = float("inf"), None, 0
            for ep in range(100):
                model.train()
                perm = np.random.permutation(len(Xa) - n_val) + n_val
                for s in range(0, len(perm), 256):
                    bi = perm[s:s+256]
                    opt.zero_grad()
                    crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = crit(model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
                if vl < best_vl:
                    best_vl, best_st, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= 15:
                        print(f"    Pretrain early stop ep {ep+1}, val={best_vl:.4f}")
                        break
            model.load_state_dict(best_st)
            del Xa, Xb, yd; gc.collect()

            # Fine-tune on ZAP70 all-pairs
            n = len(self.train_smiles)
            pairs_a, pairs_b, pairs_d = [], [], []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pairs_a.append(self.fp_cache[self.train_smiles[i]])
                        pairs_b.append(self.fp_cache[self.train_smiles[j]])
                        pairs_d.append(float(self.train_pIC50[j] - self.train_pIC50[i]))
            print(f"    Fine-tuning on {len(pairs_a):,} ZAP70 all-pairs")
            Xa = torch.FloatTensor(self.scaler.transform(np.array(pairs_a)))
            Xb = torch.FloatTensor(self.scaler.transform(np.array(pairs_b)))
            yd = torch.FloatTensor(np.array(pairs_d, dtype=np.float32))
            del pairs_a, pairs_b, pairs_d; gc.collect()

            opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            n_val2 = len(Xa) // 10
            best_vl2, best_st2, w2 = float("inf"), None, 0
            for ep in range(50):
                model.train()
                perm = np.random.permutation(len(Xa) - n_val2) + n_val2
                for s in range(0, len(perm), 256):
                    bi = perm[s:s+256]
                    opt2.zero_grad()
                    crit(model(Xa[bi], Xb[bi]), yd[bi]).backward()
                    opt2.step()
                model.eval()
                with torch.no_grad():
                    vl = crit(model(Xa[:n_val2], Xb[:n_val2]), yd[:n_val2]).item()
                if vl < best_vl2:
                    best_vl2, best_st2, w2 = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    w2 += 1
                    if w2 >= 15:
                        break
            if best_st2:
                model.load_state_dict(best_st2)
            model.eval()
            self.film_model = model
            del Xa, Xb, yd; gc.collect()

            # Precompute anchor embeddings
            te = np.array([self.fp_cache[s] for s in self.train_smiles])
            self.anchor_embs = torch.FloatTensor(self.scaler.transform(te))
            print(f"    FiLMDelta ready, device={DEVICE}")

    def score_batch(self, smiles_list, batch_size=10000):
        """Score molecules, return full DataFrame."""
        all_rows = []
        train_fps_arr = np.array([self.fp_cache[s] for s in self.train_smiles])

        for bs in range(0, len(smiles_list), batch_size):
            batch = smiles_list[bs:bs+batch_size]

            # XGB predictions
            fps = {}
            for name, (ft, r) in [("morgan", ("morgan", 2)), ("atompair", ("atompair", 2)),
                                   ("rdkit", ("rdkit", 2)), ("ecfp6", ("morgan", 3))]:
                fps[name] = compute_fingerprints(batch, ft, radius=r, n_bits=2048)

            xgb_preds_list = []
            for mn, (fk, m) in self.xgb_models.items():
                xgb_preds_list.append(m.predict(fps[fk]))
            xgb_mean = np.mean(xgb_preds_list, axis=0)
            xgb_std = np.std(xgb_preds_list, axis=0)

            # FiLMDelta anchor-based
            morgan_fps = fps["morgan"]
            batch_embs = torch.FloatTensor(self.scaler.transform(morgan_fps))
            film_preds = np.zeros(len(batch))
            film_stds = np.zeros(len(batch))
            with torch.no_grad():
                for j in range(len(batch)):
                    target_emb = batch_embs[j:j+1].expand(len(self.train_smiles), -1)
                    deltas = self.film_model(self.anchor_embs, target_emb).numpy().flatten()
                    abs_preds = self.train_pIC50 + deltas
                    film_preds[j] = np.mean(abs_preds)
                    film_stds[j] = np.std(abs_preds)

            # NN similarity (Tanimoto)
            sims = morgan_fps @ train_fps_arr.T
            b_sum = np.sum(morgan_fps, axis=1, keepdims=True)
            t_sum = np.sum(train_fps_arr, axis=1, keepdims=True)
            tani = sims / (b_sum + t_sum.T - sims + 1e-10)
            nn_sim = np.max(tani, axis=1)

            consensus = 0.5 * xgb_mean + 0.5 * film_preds

            for j in range(len(batch)):
                all_rows.append({
                    "smiles": batch[j],
                    "xgb_pred": float(xgb_mean[j]),
                    "film_pred": float(film_preds[j]),
                    "consensus_pred": float(consensus[j]),
                    "xgb_std": float(xgb_std[j]),
                    "film_std": float(film_stds[j]),
                    "nn_similarity": float(nn_sim[j]),
                })

            done = min(bs + batch_size, len(smiles_list))
            if done % 50000 == 0 or done == len(smiles_list):
                print(f"    Scored {done:,}/{len(smiles_list):,}")

        df = pd.DataFrame(all_rows)

        # Confidence tiers
        df["confidence_tier"] = "SPECULATIVE"
        df.loc[df["nn_similarity"] > 0.1, "confidence_tier"] = "LOW"
        df.loc[(df["nn_similarity"] > 0.2) & (df["film_std"] < 0.5), "confidence_tier"] = "MEDIUM"
        df.loc[(df["nn_similarity"] > 0.4) & (df["film_std"] < 0.3), "confidence_tier"] = "HIGH"

        return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_druglikeness_batch(df, max_n=None):
    """Add druglikeness columns to DataFrame."""
    if max_n:
        subset = df.head(max_n).copy()
    else:
        subset = df.copy()

    props = {"MW": [], "LogP": [], "TPSA": [], "HBA": [], "HBD": [],
             "RotBonds": [], "QED": [], "HeavyAtoms": [], "Lipinski_violations": []}

    for smi in subset["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                props["MW"].append(mw)
                props["LogP"].append(logp)
                props["TPSA"].append(Descriptors.TPSA(mol))
                props["HBA"].append(Descriptors.NumHAcceptors(mol))
                props["HBD"].append(Descriptors.NumHDonors(mol))
                props["RotBonds"].append(Descriptors.NumRotatableBonds(mol))
                props["QED"].append(Descriptors.qed(mol))
                props["HeavyAtoms"].append(mol.GetNumHeavyAtoms())
                viol = sum([mw > 500, logp > 5,
                            Descriptors.NumHAcceptors(mol) > 10,
                            Descriptors.NumHDonors(mol) > 5])
                props["Lipinski_violations"].append(viol)
                continue
            except Exception:
                pass
        for k in props:
            props[k].append(None)

    for k, v in props.items():
        subset[k] = v
    return subset


def analyze_diversity(df, train_smiles):
    """Analyze chemical diversity of scored molecules."""
    results = {}

    # Scaffold diversity
    scaffolds = Counter()
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                scaffolds[scaf] += 1
            except Exception:
                pass

    results["n_unique_scaffolds"] = len(scaffolds)
    results["scaffold_diversity"] = len(scaffolds) / len(df) if len(df) > 0 else 0
    results["top_20_scaffolds"] = [
        {"scaffold": s, "count": c, "pct": round(100*c/len(df), 2)}
        for s, c in scaffolds.most_common(20)
    ]

    # Novelty vs training set
    train_set = set(canonicalize(s) for s in train_smiles)
    n_novel = sum(1 for s in df["smiles"] if s not in train_set)
    results["n_novel_vs_training"] = n_novel
    results["pct_novel"] = round(100 * n_novel / len(df), 2)

    # Similarity distribution
    sims = df["nn_similarity"].values
    results["similarity_stats"] = {
        "mean": round(float(np.mean(sims)), 4),
        "median": round(float(np.median(sims)), 4),
        "std": round(float(np.std(sims)), 4),
        "pct_below_0.2": round(100 * np.mean(sims < 0.2), 2),
        "pct_0.2_0.4": round(100 * np.mean((sims >= 0.2) & (sims < 0.4)), 2),
        "pct_0.4_0.6": round(100 * np.mean((sims >= 0.4) & (sims < 0.6)), 2),
        "pct_above_0.6": round(100 * np.mean(sims >= 0.6), 2),
    }

    return results


def analyze_model_agreement(df):
    """Analyze XGB vs FiLMDelta agreement/disagreement."""
    results = {}

    xgb = df["xgb_pred"].values
    film = df["film_pred"].values
    diff = film - xgb

    results["correlation"] = {
        "pearson": round(float(np.corrcoef(xgb, film)[0, 1]), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(diff))), 4),
        "mean_diff_film_minus_xgb": round(float(np.mean(diff)), 4),
        "std_diff": round(float(np.std(diff)), 4),
    }

    # Agreement on activity calls
    xgb_active = xgb >= 6.0
    film_active = film >= 6.0
    both_active = xgb_active & film_active
    either_active = xgb_active | film_active
    results["activity_agreement"] = {
        "both_predict_active": int(both_active.sum()),
        "only_xgb_active": int((xgb_active & ~film_active).sum()),
        "only_film_active": int((~xgb_active & film_active).sum()),
        "neither_active": int((~xgb_active & ~film_active).sum()),
        "agreement_rate": round(float(
            (both_active.sum() + (~xgb_active & ~film_active).sum()) / len(df)), 4),
    }

    # High-confidence disagreements (interesting for analysis)
    high_conf = df["confidence_tier"] == "HIGH"
    big_diff = np.abs(diff) > 1.0
    disagreements = df[high_conf & big_diff].copy()
    disagreements["model_diff"] = diff[high_conf & big_diff]
    disagreements = disagreements.sort_values("model_diff", key=abs, ascending=False)
    results["n_high_conf_disagreements"] = len(disagreements)

    top_disagree = []
    for _, row in disagreements.head(20).iterrows():
        top_disagree.append({
            "smiles": row["smiles"],
            "xgb_pred": round(float(row["xgb_pred"]), 3),
            "film_pred": round(float(row["film_pred"]), 3),
            "diff": round(float(row["model_diff"]), 3),
            "nn_similarity": round(float(row["nn_similarity"]), 3),
        })
    results["top_disagreements"] = top_disagree

    return results


def filter_druglike(df_with_props):
    """Filter for drug-like, synthesizable hits."""
    mask = (
        (df_with_props["MW"].notna()) &
        (df_with_props["MW"] <= 550) &
        (df_with_props["LogP"] <= 5.5) &
        (df_with_props["LogP"] >= -1) &
        (df_with_props["HBA"] <= 10) &
        (df_with_props["HBD"] <= 5) &
        (df_with_props["RotBonds"] <= 10) &
        (df_with_props["QED"] >= 0.3) &
        (df_with_props["Lipinski_violations"] <= 1)
    )
    return df_with_props[mask].copy()


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(results, df_top, df_druglike, train_smiles):
    """Generate comprehensive HTML report."""
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ZAP70 1M Molecule Screening Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
h3 {{ color: #0f3460; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #16213e; color: white; padding: 10px 12px; text-align: left; font-size: 0.9em; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.85em; }}
tr:hover {{ background: #f0f4ff; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
.stat-card .value {{ font-size: 2em; font-weight: bold; color: #e94560; }}
.stat-card .label {{ color: #666; margin-top: 5px; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
.tag-high {{ background: #d4edda; color: #155724; }}
.tag-medium {{ background: #fff3cd; color: #856404; }}
.tag-low {{ background: #f8d7da; color: #721c24; }}
.tag-spec {{ background: #e2e3e5; color: #383d41; }}
.smi {{ font-family: monospace; font-size: 0.8em; word-break: break-all; max-width: 400px; display: inline-block; }}
</style>
</head><body>
<h1>ZAP70 Large-Scale Molecular Screening Report</h1>
<p><strong>Target:</strong> ZAP-70 Tyrosine Kinase (CHEMBL2803) | <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} | <strong>Training set:</strong> {len(train_smiles)} molecules</p>

<h2>1. Pipeline Overview</h2>
<div class="stat-grid">
<div class="stat-card"><div class="value">{results['generation']['total_generated']:,}</div><div class="label">Total Molecules Generated</div></div>
<div class="stat-card"><div class="value">{results['generation']['total_scored']:,}</div><div class="label">Total Scored (Dual Model)</div></div>
<div class="stat-card"><div class="value">{results['scoring']['n_potent_7']:,}</div><div class="label">Predicted Potent (pIC50≥7)</div></div>
<div class="stat-card"><div class="value">{results['scoring']['n_active_6']:,}</div><div class="label">Predicted Active (pIC50≥6)</div></div>
<div class="stat-card"><div class="value">{results['scoring']['confidence']['HIGH']:,}</div><div class="label">HIGH Confidence</div></div>
<div class="stat-card"><div class="value">{results['druglike']['n_druglike_hits']:,}</div><div class="label">Drug-like Hits (pIC50≥6)</div></div>
</div>

<h3>Generation Methods</h3>
<table>
<tr><th>Method</th><th>Count</th><th>Description</th></tr>
"""
    for method, count in results['generation']['methods'].items():
        descs = {
            'mmp': 'Beneficial MMP edits applied to training mols',
            'brics': 'BRICS fragment recombination',
            'rgroup': 'R-group enumeration on top scaffolds',
            'kinase': 'Kinase cross-pollination from ChEMBL',
            'crem': 'CReM fragment-based mutation (3 configs)',
            'selfies': 'SELFIES random perturbation',
        }
        html += f"<tr><td><strong>{method}</strong></td><td>{count:,}</td><td>{descs.get(method, '')}</td></tr>\n"
    html += "</table>\n"

    # Scoring summary
    html += """<h2>2. Scoring Summary</h2>
<p>Each molecule scored by two independent models: <strong>XGB Ensemble</strong> (5 models on diverse fingerprints) and
<strong>FiLMDelta + Kinase PT</strong> (anchor-based, noise-robust). Consensus = average of both.</p>
<table><tr><th>Confidence Tier</th><th>Count</th><th>%</th><th>Criteria</th></tr>
"""
    total = results['generation']['total_scored']
    for tier, criteria in [
        ("HIGH", "NN sim > 0.4, FiLM std < 0.3"),
        ("MEDIUM", "NN sim > 0.2, FiLM std < 0.5"),
        ("LOW", "NN sim > 0.1"),
        ("SPECULATIVE", "NN sim ≤ 0.1"),
    ]:
        c = results['scoring']['confidence'].get(tier, 0)
        pct = 100 * c / total if total > 0 else 0
        tag_cls = {"HIGH": "tag-high", "MEDIUM": "tag-medium", "LOW": "tag-low", "SPECULATIVE": "tag-spec"}[tier]
        html += f'<tr><td><span class="tag {tag_cls}">{tier}</span></td><td>{c:,}</td><td>{pct:.1f}%</td><td>{criteria}</td></tr>\n'
    html += "</table>\n"

    # Model agreement
    ma = results['model_agreement']
    html += f"""<h2>3. Model Agreement Analysis</h2>
<p>XGB vs FiLMDelta correlation: <strong>Pearson r = {ma['correlation']['pearson']}</strong>,
Mean |diff| = {ma['correlation']['mean_abs_diff']:.3f},
FiLM tends to predict {'higher' if ma['correlation']['mean_diff_film_minus_xgb'] > 0 else 'lower'} by {abs(ma['correlation']['mean_diff_film_minus_xgb']):.3f}.</p>
<table><tr><th>Category</th><th>Count</th><th>%</th></tr>
<tr><td>Both predict active (pIC50≥6)</td><td>{ma['activity_agreement']['both_predict_active']:,}</td><td>{100*ma['activity_agreement']['both_predict_active']/total:.1f}%</td></tr>
<tr><td>Only XGB predicts active</td><td>{ma['activity_agreement']['only_xgb_active']:,}</td><td>{100*ma['activity_agreement']['only_xgb_active']/total:.1f}%</td></tr>
<tr><td>Only FiLM predicts active</td><td>{ma['activity_agreement']['only_film_active']:,}</td><td>{100*ma['activity_agreement']['only_film_active']/total:.1f}%</td></tr>
<tr><td>Neither predicts active</td><td>{ma['activity_agreement']['neither_active']:,}</td><td>{100*ma['activity_agreement']['neither_active']/total:.1f}%</td></tr>
</table>
<p>Agreement rate: <strong>{100*ma['activity_agreement']['agreement_rate']:.1f}%</strong></p>
<p>High-confidence disagreements (|diff| > 1.0, HIGH tier): <strong>{ma['n_high_conf_disagreements']}</strong></p>
"""
    if ma['top_disagreements']:
        html += "<h3>Top Disagreements (HIGH confidence, largest |diff|)</h3>\n<table><tr><th>#</th><th>SMILES</th><th>XGB</th><th>FiLM</th><th>Diff</th><th>NN Sim</th></tr>\n"
        for i, d in enumerate(ma['top_disagreements'][:10]):
            html += f'<tr><td>{i+1}</td><td class="smi">{d["smiles"][:80]}</td><td>{d["xgb_pred"]:.3f}</td><td>{d["film_pred"]:.3f}</td><td>{d["diff"]:.3f}</td><td>{d["nn_similarity"]:.3f}</td></tr>\n'
        html += "</table>\n"

    # Diversity
    div = results['diversity']
    html += f"""<h2>4. Chemical Diversity</h2>
<div class="stat-grid">
<div class="stat-card"><div class="value">{div['n_unique_scaffolds']:,}</div><div class="label">Unique Scaffolds</div></div>
<div class="stat-card"><div class="value">{div['scaffold_diversity']:.4f}</div><div class="label">Scaffold Diversity (scaffolds/mols)</div></div>
<div class="stat-card"><div class="value">{div['pct_novel']:.1f}%</div><div class="label">Novel vs Training</div></div>
</div>
<h3>Similarity Distribution</h3>
<table><tr><th>Sim Range</th><th>%</th></tr>
<tr><td>&lt; 0.2 (far)</td><td>{div['similarity_stats']['pct_below_0.2']}%</td></tr>
<tr><td>0.2–0.4</td><td>{div['similarity_stats']['pct_0.2_0.4']}%</td></tr>
<tr><td>0.4–0.6</td><td>{div['similarity_stats']['pct_0.4_0.6']}%</td></tr>
<tr><td>&gt; 0.6 (close)</td><td>{div['similarity_stats']['pct_above_0.6']}%</td></tr>
</table>
<h3>Top 20 Scaffolds</h3>
<table><tr><th>#</th><th>Scaffold</th><th>Count</th><th>%</th></tr>
"""
    for i, s in enumerate(div['top_20_scaffolds']):
        html += f'<tr><td>{i+1}</td><td class="smi">{s["scaffold"][:80]}</td><td>{s["count"]:,}</td><td>{s["pct"]}%</td></tr>\n'
    html += "</table>\n"

    # Drug-likeness
    dl = results['druglike']
    html += f"""<h2>5. Drug-likeness Filtering</h2>
<p>Criteria: MW≤550, LogP∈[-1,5.5], HBA≤10, HBD≤5, RotBonds≤10, QED≥0.3, Lipinski≤1</p>
<div class="stat-grid">
<div class="stat-card"><div class="value">{dl['n_druglike_total']:,}</div><div class="label">Pass Drug-likeness Filter</div></div>
<div class="stat-card"><div class="value">{dl['pct_druglike']:.1f}%</div><div class="label">Pass Rate</div></div>
<div class="stat-card"><div class="value">{dl['n_druglike_hits']:,}</div><div class="label">Drug-like + Active (pIC50≥6)</div></div>
<div class="stat-card"><div class="value">{dl['n_druglike_potent']:,}</div><div class="label">Drug-like + Potent (pIC50≥7)</div></div>
</div>
"""
    if dl.get('property_stats'):
        ps = dl['property_stats']
        html += "<h3>Property Distribution (Drug-like Hits)</h3>\n<table><tr><th>Property</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>\n"
        for prop in ['MW', 'LogP', 'TPSA', 'QED', 'HBA', 'HBD', 'RotBonds']:
            if prop in ps:
                html += f'<tr><td>{prop}</td><td>{ps[prop]["mean"]:.2f}</td><td>{ps[prop]["std"]:.2f}</td><td>{ps[prop]["min"]:.2f}</td><td>{ps[prop]["max"]:.2f}</td></tr>\n'
        html += "</table>\n"

    # Top candidates
    html += "<h2>6. Top 50 Candidates</h2>\n"
    html += "<p>Ranked by consensus prediction (avg of XGB + FiLMDelta). Drug-likeness computed for top candidates.</p>\n"
    html += "<table><tr><th>#</th><th>SMILES</th><th>Consensus</th><th>XGB</th><th>FiLM</th><th>NN Sim</th><th>Conf</th><th>MW</th><th>QED</th><th>LogP</th></tr>\n"
    for i, (_, row) in enumerate(df_top.head(50).iterrows()):
        tag_cls = {"HIGH": "tag-high", "MEDIUM": "tag-medium", "LOW": "tag-low", "SPECULATIVE": "tag-spec"}.get(row["confidence_tier"], "tag-spec")
        mw = f'{row["MW"]:.0f}' if pd.notna(row.get("MW")) else "?"
        qed = f'{row["QED"]:.3f}' if pd.notna(row.get("QED")) else "?"
        logp = f'{row["LogP"]:.2f}' if pd.notna(row.get("LogP")) else "?"
        html += f'<tr><td>{i+1}</td><td class="smi">{row["smiles"][:80]}</td><td><strong>{row["consensus_pred"]:.3f}</strong></td>'
        html += f'<td>{row["xgb_pred"]:.3f}</td><td>{row["film_pred"]:.3f}</td><td>{row["nn_similarity"]:.3f}</td>'
        html += f'<td><span class="tag {tag_cls}">{row["confidence_tier"]}</span></td>'
        html += f'<td>{mw}</td><td>{qed}</td><td>{logp}</td></tr>\n'
    html += "</table>\n"

    # Drug-like hits table
    if len(df_druglike) > 0:
        html += "<h2>7. Top 50 Drug-like Hits (pIC50≥6, pass all filters)</h2>\n"
        html += "<table><tr><th>#</th><th>SMILES</th><th>Consensus</th><th>XGB</th><th>FiLM</th><th>Conf</th><th>MW</th><th>QED</th><th>LogP</th><th>Lipinski</th></tr>\n"
        dl_sorted = df_druglike[df_druglike["consensus_pred"] >= 6.0].sort_values("consensus_pred", ascending=False)
        for i, (_, row) in enumerate(dl_sorted.head(50).iterrows()):
            tag_cls = {"HIGH": "tag-high", "MEDIUM": "tag-medium", "LOW": "tag-low"}.get(row["confidence_tier"], "tag-spec")
            html += f'<tr><td>{i+1}</td><td class="smi">{row["smiles"][:80]}</td><td><strong>{row["consensus_pred"]:.3f}</strong></td>'
            html += f'<td>{row["xgb_pred"]:.3f}</td><td>{row["film_pred"]:.3f}</td>'
            html += f'<td><span class="tag {tag_cls}">{row["confidence_tier"]}</span></td>'
            html += f'<td>{row["MW"]:.0f}</td><td>{row["QED"]:.3f}</td><td>{row["LogP"]:.2f}</td><td>{int(row["Lipinski_violations"])}</td></tr>\n'
        html += "</table>\n"

    html += f"""
<hr>
<p style="color:#888; font-size:0.8em;">Generated by ZAP70 Screening Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
Dual scoring: XGB Ensemble (5 models) + FiLMDelta + Kinase PT (anchor-based)</p>
</body></html>"""

    with open(REPORT_HTML, "w") as f:
        f.write(html)
    print(f"    Report saved to {REPORT_HTML}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ZAP70 1M Screening + Analysis Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    results = {"target": "CHEMBL2803", "started": datetime.now().isoformat()}

    # Load data
    with timer("Loading ZAP70 data"):
        mol_data, _ = load_zap70_molecules()
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values
        print(f"    {len(smiles_list)} training molecules")

    train_fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: train_fps[i] for i, smi in enumerate(smiles_list)}
    train_set = set(canonicalize(s) for s in smiles_list)

    # Train models
    scorer = DualScorer(smiles_list, pIC50, fp_cache)
    scorer.train()

    # =========================================================================
    # GENERATION
    # =========================================================================
    all_mols = set()
    method_counts = {}

    with timer("MMP enumeration"):
        mmp = gen_mmp_enumeration(smiles_list, pIC50)
        mmp -= train_set; mmp.discard(None)
        method_counts["mmp"] = len(mmp)
        all_mols.update(mmp)
        print(f"    MMP: {len(mmp):,}")
        del mmp

    with timer("BRICS recombination"):
        brics = gen_brics(smiles_list, max_build=100000)
        brics -= train_set; brics.discard(None)
        method_counts["brics"] = len(brics)
        all_mols.update(brics)
        print(f"    BRICS: {len(brics):,}")
        del brics

    with timer("R-group enumeration"):
        rgroup = gen_rgroup(smiles_list, pIC50)
        rgroup -= train_set; rgroup.discard(None)
        method_counts["rgroup"] = len(rgroup)
        all_mols.update(rgroup)
        print(f"    R-group: {len(rgroup):,}")
        del rgroup

    with timer("Kinase cross-pollination"):
        kinase = gen_kinase_crosspolination()
        kinase -= train_set; kinase.discard(None)
        method_counts["kinase"] = len(kinase)
        all_mols.update(kinase)
        print(f"    Kinase: {len(kinase):,}")
        del kinase
    gc.collect()

    with timer("CReM generation (3 configs)"):
        crem = gen_crem(smiles_list, pIC50, all_mols | train_set, max_total=300000)
        method_counts["crem"] = len(crem)
        all_mols.update(crem)
        print(f"    CReM: {len(crem):,}")
        del crem
    gc.collect()

    # SELFIES to reach ~1M
    target = 1000000
    shortfall = max(0, target - len(all_mols))
    n_per_mol = max(200, shortfall // len(smiles_list) + 1)
    with timer(f"SELFIES perturbation ({n_per_mol}/mol, target {shortfall:,})"):
        selfies_mols = gen_selfies(smiles_list, n_per_mol=n_per_mol, max_total=shortfall)
        selfies_mols -= train_set; selfies_mols -= all_mols; selfies_mols.discard(None)
        method_counts["selfies"] = len(selfies_mols)
        all_mols.update(selfies_mols)
        print(f"    SELFIES: {len(selfies_mols):,}")
        del selfies_mols
    gc.collect()

    all_mols_list = list(all_mols)
    del all_mols
    gc.collect()

    results["generation"] = {
        "total_generated": len(all_mols_list),
        "total_scored": len(all_mols_list),
        "methods": method_counts,
    }
    print(f"\n  Total molecules to score: {len(all_mols_list):,}")

    # =========================================================================
    # SCORING — save full results to parquet
    # =========================================================================
    with timer(f"Scoring {len(all_mols_list):,} molecules"):
        scored_df = scorer.score_batch(all_mols_list)

    # Add source info
    scored_df["source"] = "generated"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with timer("Saving scored parquet"):
        scored_df.to_parquet(PARQUET_FILE, index=False)
        print(f"    Saved {len(scored_df):,} rows to {PARQUET_FILE}")
        print(f"    File size: {PARQUET_FILE.stat().st_size / 1e6:.1f} MB")

    # Scoring stats
    conf = scored_df["confidence_tier"].value_counts().to_dict()
    results["scoring"] = {
        "n_potent_7": int((scored_df["consensus_pred"] >= 7.0).sum()),
        "n_active_6": int((scored_df["consensus_pred"] >= 6.0).sum()),
        "confidence": {k: int(v) for k, v in conf.items()},
        "consensus_mean": round(float(scored_df["consensus_pred"].mean()), 3),
        "consensus_max": round(float(scored_df["consensus_pred"].max()), 3),
    }

    # =========================================================================
    # ANALYSIS 1: Chemical Diversity
    # =========================================================================
    with timer("Analyzing chemical diversity"):
        results["diversity"] = analyze_diversity(scored_df, smiles_list)
        print(f"    Scaffolds: {results['diversity']['n_unique_scaffolds']:,}")

    # =========================================================================
    # ANALYSIS 2: Drug-likeness filtering
    # =========================================================================
    with timer("Computing drug-likeness (top 200K by consensus)"):
        top_200k = scored_df.nlargest(200000, "consensus_pred")
        dl_df = compute_druglikeness_batch(top_200k)
        druglike_df = filter_druglike(dl_df)
        druglike_hits = druglike_df[druglike_df["consensus_pred"] >= 6.0]
        druglike_potent = druglike_df[druglike_df["consensus_pred"] >= 7.0]

        results["druglike"] = {
            "n_druglike_total": len(druglike_df),
            "pct_druglike": round(100 * len(druglike_df) / len(dl_df), 1),
            "n_druglike_hits": len(druglike_hits),
            "n_druglike_potent": len(druglike_potent),
        }

        # Property stats for drug-like hits
        if len(druglike_hits) > 0:
            pstats = {}
            for prop in ['MW', 'LogP', 'TPSA', 'QED', 'HBA', 'HBD', 'RotBonds']:
                vals = druglike_hits[prop].dropna()
                if len(vals) > 0:
                    pstats[prop] = {
                        "mean": round(float(vals.mean()), 2),
                        "std": round(float(vals.std()), 2),
                        "min": round(float(vals.min()), 2),
                        "max": round(float(vals.max()), 2),
                    }
            results["druglike"]["property_stats"] = pstats

        print(f"    Drug-like: {len(druglike_df):,}, Hits: {len(druglike_hits):,}, "
              f"Potent: {len(druglike_potent):,}")

    # =========================================================================
    # ANALYSIS 3: Model Agreement
    # =========================================================================
    with timer("Analyzing model agreement"):
        results["model_agreement"] = analyze_model_agreement(scored_df)
        print(f"    Agreement rate: {results['model_agreement']['activity_agreement']['agreement_rate']:.1%}")

    # =========================================================================
    # ANALYSIS 4: Top candidates with drug-likeness
    # =========================================================================
    with timer("Preparing top candidates"):
        top_500 = scored_df.nlargest(500, "consensus_pred")
        top_500_dl = compute_druglikeness_batch(top_500)

    # =========================================================================
    # HTML REPORT
    # =========================================================================
    with timer("Generating HTML report"):
        generate_html_report(results, top_500_dl, druglike_hits, smiles_list)

    # Save JSON results
    # Trim large lists for JSON
    results_json = dict(results)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  JSON results saved to {RESULTS_JSON}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Molecules generated: {results['generation']['total_generated']:,}")
    print(f"  Molecules scored:    {results['generation']['total_scored']:,}")
    print(f"  Potent (pIC50≥7):    {results['scoring']['n_potent_7']:,}")
    print(f"  Active (pIC50≥6):    {results['scoring']['n_active_6']:,}")
    print(f"  Drug-like hits:      {results['druglike']['n_druglike_hits']:,}")
    print(f"  Drug-like potent:    {results['druglike']['n_druglike_potent']:,}")
    print(f"  Unique scaffolds:    {results['diversity']['n_unique_scaffolds']:,}")
    print(f"  Model agreement:     {results['model_agreement']['activity_agreement']['agreement_rate']:.1%}")
    print(f"  Parquet: {PARQUET_FILE}")
    print(f"  Report:  {REPORT_HTML}")
    print(f"  Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
