#!/usr/bin/env python3
"""
Expand ZAP70 screening from ~210K to ~1M molecules.

Adds:
1. SELFIES-based random perturbation (proven scalable to 500K+)
2. CReM with expanded parameters (multiple radii, larger fragments)
3. Fixed BRICS recombination
4. Score all new molecules with dual model (XGB + FiLMDelta)

Picks up where run_zap70_large_scale_screening.py left off.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_zap70_expand_screening.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, BRICS
RDLogger.DisableLog('rdApp.*')

import selfies as sf

# Re-enable MPS BEFORE importing modules that disable it
_real_mps_available = torch.backends.mps.is_available
# Import needed functions
from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints, train_xgboost, train_rf
from experiments.run_zap70_v5 import compute_druglikeness
from experiments.run_zap70_edit_unified import FiLMDeltaAnchorModel
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

# Restore MPS availability
torch.backends.mps.is_available = _real_mps_available

PROJECT_ROOT = Path(__file__).parent.parent
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
CREM_DB_PATH = PROJECT_ROOT / "data" / "crem_db" / "chembl33_sa2_f5.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
PREV_RESULTS = RESULTS_DIR / "zap70_large_scale_screening.json"
EXPAND_RESULTS = RESULTS_DIR / "zap70_screening_expanded.json"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def timer(msg):
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
    return Timer(msg)


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPAND_RESULTS, "w") as f:
        json.dump(results, f, indent=2, default=str)


# =============================================================================
# SELFIES-based random perturbation
# =============================================================================

def selfies_perturbation(seed_smiles, n_per_mol=200, max_total=500000):
    """
    Generate molecular analogs by random SELFIES mutations.

    For each seed molecule:
    1. Convert SMILES -> SELFIES
    2. Randomly replace/insert/delete SELFIES tokens
    3. Convert back to SMILES, validate with RDKit
    """
    with timer(f"SELFIES perturbation ({len(seed_smiles)} seeds, {n_per_mol}/mol)"):
        alphabet = list(sf.get_semantic_robust_alphabet())
        generated = set()
        n_valid = 0
        n_invalid = 0

        for i, smi in enumerate(seed_smiles):
            try:
                selfies_str = sf.encoder(smi)
                if selfies_str is None:
                    continue
                tokens = list(sf.split_selfies(selfies_str))
                if len(tokens) < 3:
                    continue
            except Exception:
                continue

            mol_generated = 0
            attempts = 0
            max_attempts = n_per_mol * 5

            while mol_generated < n_per_mol and attempts < max_attempts:
                attempts += 1
                new_tokens = list(tokens)

                # Apply 1-3 random mutations
                n_mutations = np.random.randint(1, 4)
                for _ in range(n_mutations):
                    op = np.random.random()
                    if op < 0.4 and len(new_tokens) > 3:
                        # Replace a random token
                        idx = np.random.randint(len(new_tokens))
                        new_tokens[idx] = np.random.choice(alphabet)
                    elif op < 0.7:
                        # Insert a random token
                        idx = np.random.randint(len(new_tokens) + 1)
                        new_tokens.insert(idx, np.random.choice(alphabet))
                    elif len(new_tokens) > 4:
                        # Delete a random token
                        idx = np.random.randint(len(new_tokens))
                        new_tokens.pop(idx)

                new_selfies = "".join(new_tokens)
                try:
                    new_smi = sf.decoder(new_selfies)
                    if new_smi is None:
                        n_invalid += 1
                        continue
                    can = canonicalize(new_smi)
                    if can is None:
                        n_invalid += 1
                        continue
                    mol = Chem.MolFromSmiles(can)
                    if mol is None:
                        n_invalid += 1
                        continue
                    ha = mol.GetNumHeavyAtoms()
                    if 8 <= ha <= 80:
                        generated.add(can)
                        mol_generated += 1
                        n_valid += 1
                except Exception:
                    n_invalid += 1

                if len(generated) >= max_total:
                    break

            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(seed_smiles)} seeds, "
                      f"{len(generated):,} unique generated")

            if len(generated) >= max_total:
                print(f"    Reached {max_total:,} target, stopping early at seed {i+1}")
                break

        print(f"    SELFIES: {len(generated):,} unique molecules "
              f"({n_valid:,} valid, {n_invalid:,} invalid)")
        return generated


# =============================================================================
# Expanded CReM generation
# =============================================================================

def expanded_crem_generation(seed_smiles, existing_mols, max_total=300000):
    """CReM with multiple radii and larger fragment sizes."""
    with timer("Expanded CReM generation"):
        if not CREM_DB_PATH.exists():
            print("    CReM DB not found, skipping.")
            return set()

        try:
            from crem.crem import mutate_mol
        except ImportError:
            print("    CReM not installed, skipping.")
            return set()

        db_str = str(CREM_DB_PATH)
        generated = set()

        # Multiple radius/size configurations for diversity
        configs = [
            {"radius": 2, "min_size": 0, "max_size": 8, "max_replacements": 500},
            {"radius": 3, "min_size": 1, "max_size": 12, "max_replacements": 500},
            {"radius": 4, "min_size": 2, "max_size": 15, "max_replacements": 300},
        ]

        for cfg_i, cfg in enumerate(configs):
            print(f"    Config {cfg_i+1}/{len(configs)}: radius={cfg['radius']}, "
                  f"max_size={cfg['max_size']}")
            cfg_count = 0

            for i, smi in enumerate(seed_smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                try:
                    muts = list(mutate_mol(
                        mol, db_name=db_str,
                        return_mol=False,
                        **cfg,
                    ))
                    for new_smi in muts:
                        can = canonicalize(new_smi)
                        if can and can not in existing_mols:
                            generated.add(can)
                            cfg_count += 1
                except Exception:
                    pass

                if len(generated) >= max_total:
                    break

            print(f"      Config {cfg_i+1}: +{cfg_count:,} new molecules "
                  f"(total: {len(generated):,})")

            if len(generated) >= max_total:
                print(f"    Reached {max_total:,} target")
                break

        print(f"    Expanded CReM: {len(generated):,} unique new molecules")
        return generated


# =============================================================================
# Fixed BRICS recombination
# =============================================================================

def brics_recombination(seed_smiles, existing_mols, max_build=100000):
    """BRICS decompose + recombine with proper mol object handling."""
    with timer("BRICS recombination"):
        all_fragments = set()
        for smi in seed_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_fragments.update(frags)
            except Exception:
                continue

        print(f"    BRICS fragments: {len(all_fragments)}")

        # Convert to mol objects (the fix!)
        frag_mols = [Chem.MolFromSmiles(f) for f in all_fragments]
        frag_mols = [m for m in frag_mols if m is not None]
        print(f"    Valid fragment mols: {len(frag_mols)}")

        generated = set()
        try:
            builder = BRICS.BRICSBuild(frag_mols)
            for i, mol in enumerate(builder):
                if i >= max_build:
                    break
                try:
                    Chem.SanitizeMol(mol)
                    can = Chem.MolToSmiles(mol)
                    check = Chem.MolFromSmiles(can)
                    if check and 10 <= check.GetNumHeavyAtoms() <= 80:
                        if can not in existing_mols:
                            generated.add(can)
                except Exception:
                    continue
        except Exception as e:
            print(f"    BRICS build error: {e}")

        print(f"    BRICS: {len(generated):,} unique new molecules")
        return generated


# =============================================================================
# Scoring
# =============================================================================

class DualScorer:
    """Lightweight dual scorer for expansion."""

    def __init__(self, mol_data, fp_cache):
        self.train_smiles = mol_data["smiles"].tolist()
        self.train_pIC50 = mol_data["pIC50"].values
        self.fp_cache = fp_cache
        self.xgb_models = {}
        self.film_model = None
        self.scaler = None

    def train_models(self):
        from sklearn.preprocessing import StandardScaler
        from experiments.run_zap70_v5 import BEST_XGB_PARAMS, BEST_RF_PARAMS

        # XGB ensemble — train directly since train_ensemble needs test data
        with timer("Training XGB ensemble"):
            fp_types = {
                "morgan": {"fp_type": "morgan", "radius": 2},
                "atompair": {"fp_type": "atompair", "radius": 2},
                "rdkit": {"fp_type": "rdkit", "radius": 2},
                "ecfp6": {"fp_type": "morgan", "radius": 3},
            }
            X_train = {}
            for name, cfg in fp_types.items():
                X_train[name] = compute_fingerprints(
                    self.train_smiles, cfg["fp_type"],
                    radius=cfg["radius"], n_bits=2048
                )

            # Train 5 models: XGB on atompair/ecfp6/morgan, RF on rdkit/atompair
            y = self.train_pIC50
            configs = [
                ("xgb_atompair", "atompair", train_xgboost, BEST_XGB_PARAMS),
                ("rf_rdkit", "rdkit", train_rf, BEST_RF_PARAMS),
                ("xgb_ecfp6", "ecfp6", train_xgboost, BEST_XGB_PARAMS),
                ("xgb_morgan", "morgan", train_xgboost, BEST_XGB_PARAMS),
                ("rf_atompair", "atompair", train_rf, BEST_RF_PARAMS),
            ]
            for name, fp_key, train_fn, params in configs:
                # Use train data as dummy test (we only need the model)
                _, model = train_fn(X_train[fp_key], y, X_train[fp_key], **params)
                self.xgb_models[name] = (fp_key, model)
            print(f"    Trained {len(self.xgb_models)} XGB/RF models")

        # FiLMDelta
        with timer("Training FiLMDelta + Kinase PT"):
            kinase_pairs = pd.read_csv(
                KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"]
            )
            print(f"    Kinase pairs: {len(kinase_pairs):,}")

            # Get all needed FPs
            all_kinase_smi = list(set(
                kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
            ))
            extra_smi = [s for s in all_kinase_smi if s not in self.fp_cache]
            if extra_smi:
                print(f"    Computing FP for {len(extra_smi)} additional molecules...")
                extra_fps = compute_fingerprints(extra_smi, "morgan", radius=2, n_bits=2048)
                for i, smi in enumerate(extra_smi):
                    self.fp_cache[smi] = extra_fps[i]

            # Filter valid pairs
            mask = kinase_pairs["mol_a"].apply(lambda s: s in self.fp_cache) & \
                   kinase_pairs["mol_b"].apply(lambda s: s in self.fp_cache)
            kinase_pairs = kinase_pairs[mask].reset_index(drop=True)

            # Pretrain
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

            print("  Pretraining...")
            n_val = len(Xa) // 10
            model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.MSELoss()

            best_vl, best_state, wait = float("inf"), None, 0
            for epoch in range(100):
                model.train()
                perm = np.random.permutation(len(Xa) - n_val) + n_val
                for start in range(0, len(perm), 256):
                    bi = perm[start:start + 256]
                    optimizer.zero_grad()
                    loss = criterion(model(Xa[bi], Xb[bi]), yd[bi])
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    vl = criterion(model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()
                if vl < best_vl:
                    best_vl, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= 15:
                        print(f"    Early stop at epoch {epoch + 1}")
                        break

            model.load_state_dict(best_state)
            del Xa, Xb, yd
            gc.collect()

            # Fine-tune on ZAP70 all-pairs
            n_mols = len(self.train_smiles)
            pairs = []
            for i in range(n_mols):
                for j in range(n_mols):
                    if i != j:
                        pairs.append((self.train_smiles[i], self.train_smiles[j],
                                      float(self.train_pIC50[j] - self.train_pIC50[i])))

            print(f"    Fine-tuning on {len(pairs):,} ZAP70 all-pairs...")
            emb_a = np.array([self.fp_cache[p[0]] for p in pairs])
            emb_b = np.array([self.fp_cache[p[1]] for p in pairs])
            delta = np.array([p[2] for p in pairs], dtype=np.float32)

            Xa = torch.FloatTensor(self.scaler.transform(emb_a))
            Xb = torch.FloatTensor(self.scaler.transform(emb_b))
            yd = torch.FloatTensor(delta)
            del emb_a, emb_b, delta, pairs
            gc.collect()

            n_val2 = len(Xa) // 10
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            best_vl2, best_st2, w2 = float("inf"), None, 0
            for ep in range(50):
                model.train()
                perm = np.random.permutation(len(Xa) - n_val2) + n_val2
                for start in range(0, len(perm), 256):
                    bi = perm[start:start + 256]
                    opt.zero_grad()
                    loss = criterion(model(Xa[bi], Xb[bi]), yd[bi])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = criterion(model(Xa[:n_val2], Xb[:n_val2]), yd[:n_val2]).item()
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

            # Precompute anchor embeddings
            train_embs = np.array([self.fp_cache[s] for s in self.train_smiles])
            self.anchor_embs = torch.FloatTensor(self.scaler.transform(train_embs))

            del Xa, Xb, yd
            gc.collect()
            print(f"    FiLMDelta ready. Device: {DEVICE}")

    def score_batch(self, smiles_list, batch_size=10000):
        """Score a batch of SMILES with both models."""
        results = []

        for batch_start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[batch_start:batch_start + batch_size]

            # XGB predictions
            fps = {}
            for fp_type in ["morgan", "atompair", "rdkit", "ecfp6"]:
                radius = 3 if fp_type == "ecfp6" else 2
                actual_type = "morgan" if fp_type == "ecfp6" else fp_type
                fps[fp_type] = compute_fingerprints(batch, actual_type, radius=radius, n_bits=2048)

            xgb_preds = []
            for model_name, (fp_key, model) in self.xgb_models.items():
                xgb_preds.append(model.predict(fps[fp_key]))
            xgb_mean = np.mean(xgb_preds, axis=0)
            xgb_std = np.std(xgb_preds, axis=0)

            # FiLMDelta anchor-based predictions
            batch_fps = compute_fingerprints(batch, "morgan", radius=2, n_bits=2048)
            batch_embs = torch.FloatTensor(self.scaler.transform(batch_fps))

            film_preds = np.zeros(len(batch))
            film_stds = np.zeros(len(batch))

            with torch.no_grad():
                for j in range(len(batch)):
                    target_emb = batch_embs[j:j+1].expand(len(self.train_smiles), -1)
                    deltas = self.film_model(self.anchor_embs, target_emb).numpy()
                    abs_preds = self.train_pIC50 + deltas.flatten()
                    film_preds[j] = np.mean(abs_preds)
                    film_stds[j] = np.std(abs_preds)

            # NN similarity
            train_fps = np.array([self.fp_cache[s] for s in self.train_smiles])
            sims = batch_fps @ train_fps.T
            batch_sum = np.sum(batch_fps, axis=1, keepdims=True)
            train_sum = np.sum(train_fps, axis=1, keepdims=True)
            denom = batch_sum + train_sum.T - sims + 1e-10
            tanimoto = sims / denom
            nn_sim = np.max(tanimoto, axis=1)

            consensus = 0.5 * xgb_mean + 0.5 * film_preds

            for j in range(len(batch)):
                results.append({
                    "smiles": batch[j],
                    "xgb_pred": float(xgb_mean[j]),
                    "film_pred": float(film_preds[j]),
                    "consensus_pred": float(consensus[j]),
                    "xgb_std": float(xgb_std[j]),
                    "film_std": float(film_stds[j]),
                    "nn_similarity": float(nn_sim[j]),
                })

            if (batch_start + batch_size) % 50000 == 0 or batch_start + batch_size >= len(smiles_list):
                print(f"    Scored {min(batch_start + batch_size, len(smiles_list)):,}/{len(smiles_list):,}")

        df = pd.DataFrame(results)

        # Confidence tiers
        df["confidence_tier"] = "SPECULATIVE"
        df.loc[(df["nn_similarity"] > 0.1), "confidence_tier"] = "LOW"
        df.loc[(df["nn_similarity"] > 0.2) & (df["film_std"] < 0.5), "confidence_tier"] = "MEDIUM"
        df.loc[(df["nn_similarity"] > 0.4) & (df["film_std"] < 0.3), "confidence_tier"] = "HIGH"

        return df


def main():
    start_time = time.time()
    print("=" * 70)
    print("ZAP70 Screening Expansion: 210K -> 1M molecules")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load previous results to get existing molecules
    prev_existing = set()
    if PREV_RESULTS.exists():
        with open(PREV_RESULTS) as f:
            prev = json.load(f)
        # Extract all previously scored SMILES from top candidates
        for tier_key in ["tier1", "tier2"]:
            if tier_key in prev and "top_candidates" in prev[tier_key]:
                for c in prev[tier_key]["top_candidates"]:
                    prev_existing.add(c["smiles"])
        print(f"  Previous run: {prev.get('combined', {}).get('n_total_unique', 0):,} molecules")

    results = {
        "target": "CHEMBL2803",
        "started": datetime.now().isoformat(),
        "previous_total": prev.get("combined", {}).get("n_total_unique", 0) if PREV_RESULTS.exists() else 0,
    }

    # Load ZAP70 data
    with timer("Loading ZAP70 data"):
        mol_data, _ = load_zap70_molecules()
        smiles_list = mol_data["smiles"].tolist()
        pIC50 = mol_data["pIC50"].values
        print(f"    {len(smiles_list)} training molecules")

    # Compute FP cache
    train_fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: train_fps[i] for i, smi in enumerate(smiles_list)}
    train_set = set(canonicalize(s) for s in smiles_list)

    # Train dual scorer
    scorer = DualScorer(mol_data, fp_cache)
    scorer.train_models()

    # === Generation Phase ===
    all_new = set()

    # 1. SELFIES perturbation (target 500K)
    selfies_mols = selfies_perturbation(smiles_list, n_per_mol=2000, max_total=500000)
    selfies_mols -= train_set
    selfies_mols.discard(None)
    all_new.update(selfies_mols)
    results["selfies_generated"] = len(selfies_mols)
    print(f"  SELFIES novel: {len(selfies_mols):,}")
    del selfies_mols
    gc.collect()

    # 2. Expanded CReM (target 300K additional)
    crem_mols = expanded_crem_generation(smiles_list, all_new | train_set, max_total=300000)
    all_new.update(crem_mols)
    results["expanded_crem_generated"] = len(crem_mols)
    print(f"  Expanded CReM novel: {len(crem_mols):,}")
    del crem_mols
    gc.collect()

    # 3. BRICS recombination (fixed)
    brics_mols = brics_recombination(smiles_list, all_new | train_set, max_build=100000)
    all_new.update(brics_mols)
    results["brics_generated"] = len(brics_mols)
    print(f"  BRICS novel: {len(brics_mols):,}")
    del brics_mols
    gc.collect()

    # 4. Second-generation SELFIES from top CReM/BRICS (if under target)
    target_total = 800000  # aim for 800K new + 210K previous = ~1M
    if len(all_new) < target_total:
        shortfall = target_total - len(all_new)
        print(f"\n  Need {shortfall:,} more to reach {target_total:,} target")
        # Take a sample of the generated molecules as new seeds
        sample_size = min(500, len(all_new))
        sample_seeds = list(all_new)[:sample_size]
        extra = selfies_perturbation(sample_seeds, n_per_mol=int(shortfall / sample_size) + 1,
                                     max_total=shortfall)
        extra -= train_set
        extra -= all_new
        extra.discard(None)
        all_new.update(extra)
        results["selfies_2nd_gen"] = len(extra)
        print(f"  2nd-gen SELFIES: {len(extra):,}")
        del extra
        gc.collect()

    all_new_list = list(all_new)
    results["total_new_generated"] = len(all_new_list)
    print(f"\n  Total new molecules to score: {len(all_new_list):,}")
    save_results(results)

    # === Scoring Phase ===
    with timer(f"Scoring {len(all_new_list):,} molecules"):
        scored_df = scorer.score_batch(all_new_list)

    # Summary stats
    n_total = len(scored_df)
    conf_counts = scored_df["confidence_tier"].value_counts().to_dict()
    n_potent7 = int((scored_df["consensus_pred"] >= 7.0).sum())
    n_active6 = int((scored_df["consensus_pred"] >= 6.0).sum())

    # Top 100 candidates
    scored_sorted = scored_df.sort_values("consensus_pred", ascending=False)
    top100 = []
    for i, (_, row) in enumerate(scored_sorted.head(100).iterrows()):
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
        }
        if dl:
            entry.update({
                "MW": round(dl.get("MW", 0), 1),
                "LogP": round(dl.get("LogP", 0), 2),
                "QED": round(dl.get("QED", 0), 3),
            })
        top100.append(entry)

    results["scoring"] = {
        "n_scored": n_total,
        "confidence": {k: int(v) for k, v in conf_counts.items()},
        "n_predicted_potent_7": n_potent7,
        "n_predicted_active_6": n_active6,
        "consensus_mean": round(float(scored_df["consensus_pred"].mean()), 3),
        "consensus_max": round(float(scored_df["consensus_pred"].max()), 3),
        "top_100": top100,
    }

    # Combined total with previous run
    prev_total = results.get("previous_total", 0)
    grand_total = prev_total + n_total
    results["grand_total"] = grand_total

    elapsed = time.time() - start_time
    results["completed"] = datetime.now().isoformat()
    results["elapsed_seconds"] = round(elapsed, 1)
    save_results(results)

    print("\n" + "=" * 70)
    print("EXPANSION COMPLETE")
    print("=" * 70)
    print(f"  New molecules scored: {n_total:,}")
    print(f"  Previous run:         {prev_total:,}")
    print(f"  GRAND TOTAL:          {grand_total:,}")
    print(f"")
    print(f"  New confidence distribution:")
    for tier in ["HIGH", "MEDIUM", "LOW", "SPECULATIVE"]:
        print(f"    {tier:12s}: {conf_counts.get(tier, 0):,}")
    print(f"")
    print(f"  New predicted potent (pIC50>=7): {n_potent7:,}")
    print(f"  New predicted active (pIC50>=6): {n_active6:,}")
    print(f"")
    print(f"  Top 10 NEW candidates:")
    print(f"  {'Rank':>4} {'Consensus':>9} {'XGB':>6} {'FiLM':>6} {'Sim':>5} {'Conf':>12}")
    for c in top100[:10]:
        print(f"  {c['rank']:4d} {c['consensus_pred']:9.3f} {c['xgb_pred']:6.3f} "
              f"{c['film_pred']:6.3f} {c['nn_similarity']:5.3f} "
              f"{c['confidence_tier']:>12s}")
    print(f"")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Results: {EXPAND_RESULTS}")


if __name__ == "__main__":
    main()
