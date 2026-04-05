#!/usr/bin/env python3
"""
ZAP70 (CHEMBL2803) Unified Virtual Screening & Analysis — FiLMDelta + Kinase PT.

Target: Tyrosine-protein kinase ZAP-70 (CHEMBL2803).

Uses FiLMDelta pretrained on kinase within-assay pairs, fine-tuned on ZAP70 all-pairs,
with anchor-based absolute prediction: pred(j) = mean_i(known_pIC50_i + delta(i->j)).

Phase A: Score & rank ChEMBL compounds NOT in training set
Phase B: Kinase similarity search — repurpose potent kinase inhibitors
Phase C: SAR-guided enumeration — BRICS recombination from potent molecules
Phase D: MMP-inspired optimization — apply beneficial edits with FiLMDelta-predicted deltas
Phase E: Multi-objective ranking — combine all candidates
Phase F: Model interpretation — anchor analysis, per-edit accuracy, FiLM layer statistics
Phase G: HTML report

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_zap70_edit_unified.py
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
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
CHEMBL_DB = PROJECT_ROOT / "data" / "chembl_db" / "chembl" / "36" / "chembl_36.db"
KINASE_PAIRS_FILE = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
SHARED_PAIRS_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_FILE = RESULTS_DIR / "zap70_edit_unified_results.json"
REPORT_FILE = RESULTS_DIR / "zap70_edit_unified_report.html"

TARGET_ID = "CHEMBL2803"
TARGET_NAME = "ZAP70 (Tyrosine-protein kinase ZAP-70)"


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def tanimoto_kernel_matrix(X, Y=None):
    """Tanimoto kernel for binary fingerprints."""
    if Y is None:
        Y = X
    XY = X @ Y.T
    X2 = np.sum(X, axis=1, keepdims=True)
    Y2 = np.sum(Y, axis=1, keepdims=True)
    denom = X2 + Y2.T - XY + 1e-10
    return XY / denom


def compute_druglikeness(smi):
    """Compute druglikeness properties for a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        lip_v = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
        return {
            "MW": mw,
            "LogP": logp,
            "TPSA": Descriptors.TPSA(mol),
            "HBA": hba,
            "HBD": hbd,
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "Rings": Descriptors.RingCount(mol),
            "AromaticRings": Descriptors.NumAromaticRings(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "QED": Descriptors.qed(mol),
            "SA_score": _sa_score(mol),
            "Lipinski_violations": lip_v,
        }
    except Exception:
        return {}


def _sa_score(mol):
    """Synthetic accessibility score (1=easy, 10=hard)."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score', 'sascorer.py')
        if os.path.exists(sa_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("sascorer", sa_path)
            sascorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sascorer)
            return sascorer.calculateScore(mol)
    except Exception:
        pass
    n_rings = Descriptors.RingCount(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_heavy = mol.GetNumHeavyAtoms()
    return min(10, max(1, 1 + n_heavy * 0.05 + n_rings * 0.3 + n_stereo * 0.5))


# ═══════════════════════════════════════════════════════════════════════════
# Core Model: FiLMDelta + Kinase Pretraining + Anchor Prediction
# ═══════════════════════════════════════════════════════════════════════════

class FiLMDeltaAnchorModel:
    """
    FiLMDelta pretrained on kinase within-assay pairs, fine-tuned on ZAP70
    all-pairs, with anchor-based absolute prediction.

    pred(j) = mean_i(known_pIC50_i + FiLMDelta(anchor_i -> target_j))
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.anchor_smiles = None
        self.anchor_embs = None  # scaled, as tensor
        self.anchor_pIC50 = None

    def pretrain_on_kinase(self, kinase_pairs_df, fp_cache,
                           epochs=100, batch_size=256, lr=1e-3, patience=15):
        """Pretrain FiLMDelta on kinase within-assay pairs."""
        from sklearn.preprocessing import StandardScaler

        print("  Building kinase pair tensors...")
        emb_a = np.array([fp_cache[s] for s in kinase_pairs_df["mol_a"]])
        emb_b = np.array([fp_cache[s] for s in kinase_pairs_df["mol_b"]])
        delta = kinase_pairs_df["delta"].values.astype(np.float32)

        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack([emb_a, emb_b]))

        Xa = torch.FloatTensor(self.scaler.transform(emb_a))
        Xb = torch.FloatTensor(self.scaler.transform(emb_b))
        yd = torch.FloatTensor(delta)
        del emb_a, emb_b, delta
        gc.collect()

        n_val = len(Xa) // 10
        input_dim = Xa.shape[1]

        self.model = FiLMDeltaMLP(
            input_dim=input_dim, hidden_dims=[1024, 512, 256], dropout=0.2
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_vl, best_state, wait = float("inf"), None, 0
        for epoch in range(epochs):
            self.model.train()
            perm = np.random.permutation(len(Xa) - n_val) + n_val
            epoch_losses = []
            for start in range(0, len(perm), batch_size):
                bi = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(Xa[bi], Xb[bi]), yd[bi])
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                vl = criterion(self.model(Xa[:n_val], Xb[:n_val]), yd[:n_val]).item()

            if vl < best_vl:
                best_vl = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early stop at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}: train={np.mean(epoch_losses):.4f}, val={vl:.4f}")

        self.model.load_state_dict(best_state)
        self.model.eval()

        # Compute validation metrics
        with torch.no_grad():
            val_pred = self.model(Xa[:n_val], Xb[:n_val]).numpy()
        val_true = yd[:n_val].numpy()
        pt_mae = float(np.mean(np.abs(val_true - val_pred)))
        pt_spr = float(spearmanr(val_true, val_pred).statistic)
        print(f"    Pretrain validation: MAE={pt_mae:.4f}, Spr={pt_spr:.3f}")

        del Xa, Xb, yd
        gc.collect()
        return {"mae": pt_mae, "spearman": pt_spr, "best_val_loss": best_vl}

    def finetune_on_zap70(self, train_smiles, train_pIC50, fp_cache,
                          epochs=50, batch_size=256, lr=1e-4, patience=15):
        """Fine-tune on ZAP70 all-pairs (full training set for screening)."""
        if self.model is None:
            raise RuntimeError("Must pretrain first.")

        # Generate all-pairs from training molecules
        n = len(train_smiles)
        rows_a, rows_b, deltas = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rows_a.append(train_smiles[i])
                rows_b.append(train_smiles[j])
                deltas.append(float(train_pIC50[j] - train_pIC50[i]))

        emb_a = np.array([fp_cache[s] for s in rows_a])
        emb_b = np.array([fp_cache[s] for s in rows_b])

        Xa = torch.FloatTensor(self.scaler.transform(emb_a))
        Xb = torch.FloatTensor(self.scaler.transform(emb_b))
        yd = torch.FloatTensor(np.array(deltas, dtype=np.float32))
        del emb_a, emb_b, rows_a, rows_b, deltas
        gc.collect()

        print(f"    Fine-tuning on {len(Xa):,} ZAP70 all-pairs ({n} molecules)...")

        # Use 10% for validation
        n_val = max(len(Xa) // 10, 1)
        perm_all = np.random.RandomState(42).permutation(len(Xa))
        val_idx = perm_all[:n_val]
        train_idx = perm_all[n_val:]

        ft_model = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_vl, best_state, wait = float("inf"), None, 0
        for epoch in range(epochs):
            ft_model.train()
            perm = np.random.permutation(len(train_idx))
            for start in range(0, len(perm), batch_size):
                bi = train_idx[perm[start:start + batch_size]]
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
                if wait >= patience:
                    print(f"    Early stop at epoch {epoch + 1}")
                    break
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}: val_loss={vl:.4f}")

        if best_state:
            ft_model.load_state_dict(best_state)
        ft_model.eval()
        self.model = ft_model

        # Compute validation metrics
        with torch.no_grad():
            val_pred = self.model(Xa[val_idx], Xb[val_idx]).numpy()
        val_true = yd[val_idx].numpy()
        ft_mae = float(np.mean(np.abs(val_true - val_pred)))
        ft_spr = float(spearmanr(val_true, val_pred).statistic) if len(val_true) > 2 else 0.0
        print(f"    Fine-tune validation: MAE={ft_mae:.4f}, Spr={ft_spr:.3f}")

        del Xa, Xb, yd
        gc.collect()

        # Store anchor info
        self.anchor_smiles = list(train_smiles)
        self.anchor_pIC50 = np.array(train_pIC50, dtype=np.float32)
        self.anchor_embs = torch.FloatTensor(
            self.scaler.transform(np.array([fp_cache[s] for s in train_smiles]))
        )

        return {"mae": ft_mae, "spearman": ft_spr, "n_pairs": int(n_val + len(train_idx)),
                "best_val_loss": best_vl}

    def predict_absolute(self, target_smiles_list, fp_cache):
        """
        Anchor-based absolute prediction:
        pred(j) = mean_i(anchor_pIC50_i + delta(i->j))

        Returns (predictions, uncertainties) where uncertainty = std of anchor predictions.
        """
        if self.model is None or self.anchor_embs is None:
            raise RuntimeError("Must pretrain and fine-tune first.")

        self.model.eval()
        n_targets = len(target_smiles_list)
        n_anchors = len(self.anchor_smiles)

        target_embs_raw = np.array([fp_cache[s] for s in target_smiles_list])
        target_embs = torch.FloatTensor(self.scaler.transform(target_embs_raw))
        del target_embs_raw

        predictions = np.zeros(n_targets)
        uncertainties = np.zeros(n_targets)

        # Process in batches of targets to control memory
        BATCH = 50
        for start in range(0, n_targets, BATCH):
            end = min(start + BATCH, n_targets)
            batch_preds = np.zeros((end - start, n_anchors))

            for j_local, j_global in enumerate(range(start, end)):
                target_expanded = target_embs[j_global:j_global + 1].expand(n_anchors, -1)
                with torch.no_grad():
                    deltas = self.model(self.anchor_embs, target_expanded).numpy()
                anchor_preds = self.anchor_pIC50 + deltas
                batch_preds[j_local] = anchor_preds

            predictions[start:end] = np.mean(batch_preds, axis=1)
            uncertainties[start:end] = np.std(batch_preds, axis=1)

        return predictions, uncertainties

    def predict_delta(self, smiles_a_list, smiles_b_list, fp_cache):
        """Predict delta between pairs of molecules."""
        self.model.eval()
        emb_a = np.array([fp_cache[s] for s in smiles_a_list])
        emb_b = np.array([fp_cache[s] for s in smiles_b_list])
        Xa = torch.FloatTensor(self.scaler.transform(emb_a))
        Xb = torch.FloatTensor(self.scaler.transform(emb_b))
        with torch.no_grad():
            return self.model(Xa, Xb).numpy()

    def get_anchor_predictions_matrix(self, target_smiles_list, fp_cache):
        """
        Get the full anchor prediction matrix [n_targets x n_anchors].
        Each entry (j, i) = anchor_pIC50_i + delta(i->j).
        """
        self.model.eval()
        n_targets = len(target_smiles_list)
        n_anchors = len(self.anchor_smiles)

        target_embs_raw = np.array([fp_cache[s] for s in target_smiles_list])
        target_embs = torch.FloatTensor(self.scaler.transform(target_embs_raw))

        matrix = np.zeros((n_targets, n_anchors))
        for j in range(n_targets):
            target_expanded = target_embs[j:j + 1].expand(n_anchors, -1)
            with torch.no_grad():
                deltas = self.model(self.anchor_embs, target_expanded).numpy()
            matrix[j] = self.anchor_pIC50 + deltas

        return matrix


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: ChEMBL Compound Screening
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_a(mol_data, model, fp_cache, results):
    """Score all ChEMBL compounds for ZAP70 not in training set."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE A: Score & Rank ChEMBL Compounds (FiLMDelta Anchor)")
    print("=" * 70)

    if not CHEMBL_DB.exists():
        print("  ChEMBL database not found, skipping.")
        results["phase_a"] = {"error": "ChEMBL DB not found"}
        return results

    # Query ChEMBL for ZAP70 compounds
    db = sqlite3.connect(str(CHEMBL_DB))
    query = """
        SELECT DISTINCT cs.canonical_smiles, md.chembl_id as mol_chembl_id,
               a.standard_type, a.standard_value, a.standard_units,
               a.pchembl_value
        FROM activities a
        JOIN assays ass ON a.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE td.chembl_id = ?
        AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
        AND a.standard_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
    """
    chembl_df = pd.read_sql_query(query, db, params=[TARGET_ID])
    db.close()

    print(f"  ChEMBL activities: {len(chembl_df)}, unique mols: {chembl_df['mol_chembl_id'].nunique()}")

    # Identify molecules NOT in training set
    train_canonical = set()
    for smi in mol_data["smiles"].values:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    chembl_df["canonical"] = chembl_df["canonical_smiles"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    new_mols = chembl_df[~chembl_df["canonical"].isin(train_canonical)].drop_duplicates(subset=["canonical"])
    known_mols = chembl_df[chembl_df["canonical"].isin(train_canonical)].drop_duplicates(subset=["canonical"])

    print(f"  Known (in training): {len(known_mols)}")
    print(f"  New (not in training): {len(new_mols)}")

    if len(new_mols) == 0:
        results["phase_a"] = {"n_new": 0, "message": "No new molecules found"}
        return results

    # Compute FP for new molecules and add to cache
    new_smiles_list = new_mols["canonical"].tolist()
    missing = [s for s in new_smiles_list if s not in fp_cache]
    if missing:
        X_missing = compute_fingerprints(missing, "morgan", radius=2, n_bits=2048)
        for i, smi in enumerate(missing):
            fp_cache[smi] = X_missing[i]
        del X_missing

    # Score with anchor-based prediction
    print("  Scoring with FiLMDelta anchor model...")
    new_preds, new_stds = model.predict_absolute(new_smiles_list, fp_cache)

    # NN similarity to training set
    print("  Computing NN similarity to training set...")
    train_smiles_list = mol_data["smiles"].tolist()
    train_morgan = np.array([fp_cache[s] for s in train_smiles_list])
    new_morgan = np.array([fp_cache[s] for s in new_smiles_list])

    nn_sims, nn_smiles = [], []
    for i in range(len(new_smiles_list)):
        sims = tanimoto_kernel_matrix(new_morgan[i:i + 1], train_morgan)[0]
        best_idx = np.argmax(sims)
        nn_sims.append(float(sims[best_idx]))
        nn_smiles.append(train_smiles_list[best_idx])

    # Druglikeness
    print("  Computing druglikeness...")
    drug_props = [compute_druglikeness(smi) for smi in new_smiles_list]

    # Build candidate table
    candidates = []
    for i in range(len(new_smiles_list)):
        dp = drug_props[i] if drug_props[i] else {}
        cand = {
            "rank": 0,
            "smiles": new_smiles_list[i],
            "chembl_id": new_mols.iloc[i]["mol_chembl_id"],
            "predicted_pIC50": round(float(new_preds[i]), 3),
            "uncertainty": round(float(new_stds[i]), 3),
            "nn_similarity": round(nn_sims[i], 3),
            "nn_train_smiles": nn_smiles[i],
            "known_value": None,
            "MW": dp.get("MW"),
            "LogP": dp.get("LogP"),
            "TPSA": dp.get("TPSA"),
            "QED": dp.get("QED"),
            "SA_score": dp.get("SA_score"),
            "Lipinski_violations": dp.get("Lipinski_violations"),
        }
        matching = new_mols[new_mols["canonical"] == new_smiles_list[i]]
        pchembl_vals = matching["pchembl_value"].dropna()
        if len(pchembl_vals) > 0:
            cand["known_value"] = round(float(pchembl_vals.mean()), 2)
        candidates.append(cand)

    candidates.sort(key=lambda x: x["predicted_pIC50"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    n_potent = sum(1 for c in candidates if c["predicted_pIC50"] >= 7.0)
    n_moderate = sum(1 for c in candidates if 6.0 <= c["predicted_pIC50"] < 7.0)
    n_weak = sum(1 for c in candidates if c["predicted_pIC50"] < 6.0)
    high_conf_potent = [c for c in candidates
                        if c["predicted_pIC50"] >= 7.0 and c["nn_similarity"] >= 0.3]

    print(f"\n  === Screening Results ===")
    print(f"  Total new candidates: {len(candidates)}")
    print(f"  Predicted potent (pIC50>=7): {n_potent}")
    print(f"  Predicted moderate (6-7): {n_moderate}")
    print(f"  Predicted weak (<6): {n_weak}")
    print(f"  High-confidence potent (pIC50>=7, sim>=0.3): {len(high_conf_potent)}")

    print(f"\n  Top 20 candidates:")
    print(f"  {'Rank':>4} {'ChEMBL ID':>14} {'Pred pIC50':>10} {'Unc':>6} {'NN Sim':>7} {'Known':>6} {'QED':>5}")
    for c in candidates[:20]:
        known_str = f"{c['known_value']:.1f}" if c["known_value"] else "   -"
        qed_str = f"{c['QED']:.2f}" if c["QED"] else "  -"
        print(f"  {c['rank']:4d} {c['chembl_id']:>14s} {c['predicted_pIC50']:10.2f} "
              f"+-{c['uncertainty']:.2f} {c['nn_similarity']:7.3f} {known_str:>6s} {qed_str:>5s}")

    # Validate where known
    with_known = [c for c in candidates if c["known_value"] is not None]
    val_result = {}
    if with_known:
        known_true = np.array([c["known_value"] for c in with_known])
        known_pred = np.array([c["predicted_pIC50"] for c in with_known])
        mae = np.mean(np.abs(known_true - known_pred))
        sr, _ = spearmanr(known_true, known_pred) if len(known_true) > 2 else (0, 1)
        print(f"\n  Validation on {len(with_known)} mols with known pChEMBL:")
        print(f"  MAE={mae:.3f}, Spearman={sr:.3f}")
        val_result = {"n_with_known": len(with_known), "mae": float(mae), "spearman": float(sr)}

    elapsed = time.time() - t0
    results["phase_a"] = {
        "n_candidates": len(candidates),
        "n_potent": n_potent,
        "n_moderate": n_moderate,
        "n_weak": n_weak,
        "n_high_conf_potent": len(high_conf_potent),
        "candidates": candidates[:50],
        "validation": val_result,
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase A complete ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Kinase Similarity Search
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b(mol_data, model, fp_cache, results):
    """Search for related molecules from similar kinase targets."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE B: Kinase Similarity Search (FiLMDelta Anchor)")
    print("=" * 70)

    if not CHEMBL_DB.exists():
        print("  ChEMBL database not found, skipping.")
        results["phase_b"] = {"error": "ChEMBL DB not found"}
        return results

    RELATED_KINASES = {
        "ITK": "CHEMBL3009",
        "SYK": "CHEMBL2599",
        "FYN": "CHEMBL1841",
        "RAF1": "CHEMBL1906",
        "MEK1": "CHEMBL399",
        "MEK2": "CHEMBL4045",
    }

    db = sqlite3.connect(str(CHEMBL_DB))
    kinase_compounds = {}
    for name, chembl_id in RELATED_KINASES.items():
        query = """
            SELECT DISTINCT cs.canonical_smiles, md.chembl_id, a.pchembl_value
            FROM activities a
            JOIN assays ass ON a.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN molecule_dictionary md ON a.molregno = md.molregno
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE td.chembl_id = ?
            AND a.pchembl_value >= 7.0
            AND cs.canonical_smiles IS NOT NULL
            LIMIT 500
        """
        df = pd.read_sql_query(query, db, params=[chembl_id])
        kinase_compounds[name] = df
        print(f"  {name} ({chembl_id}): {len(df)} potent compounds")
    db.close()

    # Canonical set for exclusion
    train_canonical = set()
    for smi in mol_data["smiles"].values:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    phase_a_smiles = set()
    if "phase_a" in results and "candidates" in results["phase_a"]:
        phase_a_smiles = set(c["smiles"] for c in results["phase_a"]["candidates"])

    novel_kinase_mols = []
    for kinase_name, df in kinase_compounds.items():
        for _, row in df.iterrows():
            smi = row["canonical_smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol)
            if can_smi in train_canonical or can_smi in phase_a_smiles:
                continue
            novel_kinase_mols.append({
                "smiles": can_smi,
                "chembl_id": row["chembl_id"],
                "source_kinase": kinase_name,
                "source_pchembl": float(row["pchembl_value"]) if pd.notna(row["pchembl_value"]) else None,
            })

    # Deduplicate
    seen = set()
    unique_kinase_mols = []
    for m in novel_kinase_mols:
        if m["smiles"] not in seen:
            seen.add(m["smiles"])
            unique_kinase_mols.append(m)

    print(f"\n  Novel kinase molecules: {len(unique_kinase_mols)}")

    if len(unique_kinase_mols) == 0:
        results["phase_b"] = {"n_novel": 0, "completed": True}
        return results

    # Compute FP and add to cache
    novel_smiles = [m["smiles"] for m in unique_kinase_mols]
    missing = [s for s in novel_smiles if s not in fp_cache]
    if missing:
        X_missing = compute_fingerprints(missing, "morgan", radius=2, n_bits=2048)
        for i, smi in enumerate(missing):
            fp_cache[smi] = X_missing[i]
        del X_missing

    # Score
    print("  Scoring with FiLMDelta anchor model...")
    novel_preds, novel_stds = model.predict_absolute(novel_smiles, fp_cache)

    # NN similarity
    train_smiles_list = mol_data["smiles"].tolist()
    train_morgan = np.array([fp_cache[s] for s in train_smiles_list])
    novel_morgan = np.array([fp_cache[s] for s in novel_smiles])

    kinase_candidates = []
    for i, m in enumerate(unique_kinase_mols):
        sims = tanimoto_kernel_matrix(novel_morgan[i:i + 1], train_morgan)[0]
        nn_sim = float(np.max(sims))
        dp = compute_druglikeness(m["smiles"])
        kinase_candidates.append({
            "rank": 0,
            "smiles": m["smiles"],
            "chembl_id": m["chembl_id"],
            "source_kinase": m["source_kinase"],
            "source_pchembl": m["source_pchembl"],
            "predicted_zap70_pIC50": round(float(novel_preds[i]), 3),
            "uncertainty": round(float(novel_stds[i]), 3),
            "nn_similarity_to_zap70": round(nn_sim, 3),
            "QED": round(dp.get("QED", 0), 3) if dp else None,
            "MW": round(dp.get("MW", 0), 1) if dp else None,
        })

    kinase_candidates.sort(key=lambda x: x["predicted_zap70_pIC50"], reverse=True)
    for i, c in enumerate(kinase_candidates):
        c["rank"] = i + 1

    n_repurpose = sum(1 for c in kinase_candidates if c["predicted_zap70_pIC50"] >= 6.5)
    print(f"\n  Kinase compounds predicted active on ZAP70 (pIC50>=6.5): {n_repurpose}")
    print(f"\n  Top 15 repurposing candidates:")
    print(f"  {'Rank':>4} {'Source':>6} {'Pred':>5} {'Unc':>5} {'Sim':>5} {'Src pIC50':>9}")
    for c in kinase_candidates[:15]:
        src_val = f"{c['source_pchembl']:.1f}" if c["source_pchembl"] else "  -"
        print(f"  {c['rank']:4d} {c['source_kinase']:>6s} {c['predicted_zap70_pIC50']:5.2f} "
              f"+-{c['uncertainty']:.2f} {c['nn_similarity_to_zap70']:.3f} {src_val:>9s}")

    elapsed = time.time() - t0
    results["phase_b"] = {
        "n_novel_kinase_mols": len(unique_kinase_mols),
        "n_predicted_active": n_repurpose,
        "kinase_sources": {k: len(v) for k, v in kinase_compounds.items()},
        "candidates": kinase_candidates[:30],
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase B complete ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: SAR-Guided Enumeration
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c(mol_data, model, fp_cache, results):
    """Enumerate R-group variations via BRICS recombination."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE C: SAR-Guided Enumeration (FiLMDelta Anchor)")
    print("=" * 70)

    potent = mol_data[mol_data["pIC50"] >= 7.0].copy()
    print(f"  Potent molecules (pIC50>=7): {len(potent)}")

    # Murcko scaffolds
    scaffold_groups = defaultdict(list)
    for _, row in potent.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_smi = Chem.MolToSmiles(scaf)
                scaffold_groups[scaf_smi].append({
                    "smiles": row["smiles"], "pIC50": row["pIC50"],
                })
            except Exception:
                continue

    top_scaffolds = sorted(scaffold_groups.items(), key=lambda x: -len(x[1]))[:5]
    print(f"  Top scaffolds:")
    for scaf, mols in top_scaffolds:
        mean_pic = np.mean([m["pIC50"] for m in mols])
        print(f"    {scaf[:60]}... (n={len(mols)}, mean={mean_pic:.2f})")

    # BRICS decomposition
    print("\n  BRICS decomposition of top molecules...")
    all_fragments = Counter()
    for _, row in potent.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                for f in frags:
                    all_fragments[f] += 1
            except Exception:
                continue

    print(f"  Unique BRICS fragments: {len(all_fragments)}")

    # BRICS recombination from top-20 potent molecules
    print("\n  Generating analogs via BRICS recombination...")
    generated_mols = set()
    train_canonical = set()
    for smi in mol_data["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_canonical.add(Chem.MolToSmiles(mol))

    top_mols = potent.nlargest(20, "pIC50")
    all_brics_frags = set()
    for _, row in top_mols.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_brics_frags.update(frags)
            except Exception:
                continue

    if all_brics_frags:
        print(f"  BRICS fragments from top-20: {len(all_brics_frags)}")
        try:
            frag_mols = []
            for frag_smi in list(all_brics_frags)[:20]:
                fmol = Chem.MolFromSmiles(frag_smi)
                if fmol is not None:
                    frag_mols.append(fmol)
            builder = BRICS.BRICSBuild(frag_mols)
            for count, mol in enumerate(builder):
                if count >= 500:
                    break
                try:
                    smi = Chem.MolToSmiles(mol)
                    if smi not in train_canonical and len(smi) < 200:
                        if 15 <= mol.GetNumHeavyAtoms() <= 50:
                            generated_mols.add(smi)
                except Exception:
                    continue
            print(f"  Generated {len(generated_mols)} novel molecules via BRICS")
        except Exception as e:
            print(f"  BRICS build failed: {e}")

    if not generated_mols:
        print("  No novel molecules generated.")
        results["phase_c"] = {"n_generated": 0, "completed": True}
        save_results(results)
        return results

    # Compute FP and score
    gen_smiles_list = list(generated_mols)
    missing = [s for s in gen_smiles_list if s not in fp_cache]
    if missing:
        X_missing = compute_fingerprints(missing, "morgan", radius=2, n_bits=2048)
        for i, smi in enumerate(missing):
            fp_cache[smi] = X_missing[i]
        del X_missing

    print(f"\n  Scoring {len(gen_smiles_list)} generated molecules...")
    gen_preds, gen_stds = model.predict_absolute(gen_smiles_list, fp_cache)

    # Similarity + druglikeness
    train_smiles_list = mol_data["smiles"].tolist()
    train_morgan = np.array([fp_cache[s] for s in train_smiles_list])
    gen_morgan = np.array([fp_cache[s] for s in gen_smiles_list])

    gen_candidates = []
    for i, smi in enumerate(gen_smiles_list):
        sims = tanimoto_kernel_matrix(gen_morgan[i:i + 1], train_morgan)[0]
        best_idx = np.argmax(sims)
        nn_sim = float(sims[best_idx])
        dp = compute_druglikeness(smi)
        gen_candidates.append({
            "rank": 0,
            "smiles": smi,
            "predicted_pIC50": round(float(gen_preds[i]), 3),
            "uncertainty": round(float(gen_stds[i]), 3),
            "nn_similarity": round(nn_sim, 3),
            "nn_train_smiles": train_smiles_list[best_idx],
            "QED": round(dp.get("QED", 0), 3) if dp else None,
            "MW": round(dp.get("MW", 0), 1) if dp else None,
            "SA_score": round(dp.get("SA_score", 3), 2) if dp else None,
            "Lipinski_violations": dp.get("Lipinski_violations"),
        })

    gen_candidates.sort(key=lambda x: x["predicted_pIC50"], reverse=True)
    for i, c in enumerate(gen_candidates):
        c["rank"] = i + 1

    druglike = [c for c in gen_candidates
                if c.get("Lipinski_violations", 5) <= 1 and
                c.get("QED", 0) >= 0.3 and
                c.get("SA_score", 10) <= 5]

    n_potent_gen = sum(1 for c in gen_candidates if c["predicted_pIC50"] >= 7.0)
    print(f"\n  === Generated Molecule Results ===")
    print(f"  Total: {len(gen_candidates)}, Potent: {n_potent_gen}, Drug-like: {len(druglike)}")

    print(f"\n  Top 15:")
    print(f"  {'Rank':>4} {'Pred':>6} {'Unc':>5} {'Sim':>5} {'QED':>5}")
    for c in gen_candidates[:15]:
        qed_str = f"{c['QED']:.2f}" if c["QED"] else "  -"
        print(f"  {c['rank']:4d} {c['predicted_pIC50']:6.2f} +-{c['uncertainty']:.2f} "
              f"{c['nn_similarity']:.3f} {qed_str:>5s}")

    elapsed = time.time() - t0
    results["phase_c"] = {
        "n_brics_fragments": len(all_brics_frags),
        "n_generated": len(gen_candidates),
        "n_potent": n_potent_gen,
        "n_druglike": len(druglike),
        "top_scaffolds": [{"scaffold": s, "n_mols": len(m),
                           "mean_pIC50": round(np.mean([x["pIC50"] for x in m]), 2)}
                          for s, m in top_scaffolds],
        "top_fragments": [{"fragment": f, "count": c} for f, c in all_fragments.most_common(10)],
        "candidates": gen_candidates[:30],
        "druglike_candidates": druglike[:20] if druglike else [],
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase C complete ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: MMP-Inspired Optimization (with FiLMDelta-predicted deltas)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_d(mol_data, model, fp_cache, results):
    """Apply beneficial edits with FiLMDelta-predicted deltas."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE D: MMP-Inspired Optimization (FiLMDelta Deltas)")
    print("=" * 70)

    # Load MMP pairs — use shared_pairs_deduped.csv with minimal columns
    print("  Loading MMP edits...")
    pairs = pd.read_csv(
        SHARED_PAIRS_FILE,
        usecols=["mol_a", "mol_b", "edit_smiles", "delta", "is_within_assay", "target_chembl_id"]
    )

    within = pairs[pairs["is_within_assay"] == True].copy()
    print(f"  Within-assay pairs: {len(within):,}")
    del pairs
    gc.collect()

    # Find beneficial edits
    edit_stats = within.groupby("edit_smiles").agg(
        n_pairs=("delta", "count"),
        mean_delta=("delta", "mean"),
        std_delta=("delta", "std"),
        n_targets=("target_chembl_id", "nunique"),
    ).reset_index()

    beneficial_edits = edit_stats[
        (edit_stats["n_pairs"] >= 10) &
        (edit_stats["mean_delta"] > 0.3) &
        (edit_stats["n_targets"] >= 2)
    ].sort_values("mean_delta", ascending=False)

    print(f"  Beneficial edits (n>=10, delta>0.3, targets>=2): {len(beneficial_edits)}")
    print(f"\n  Top 15 beneficial edits:")
    print(f"  {'Edit SMILES':>50} {'N':>5} {'Mean D':>7} {'Targets':>7}")
    for _, row in beneficial_edits.head(15).iterrows():
        edit_smi = str(row["edit_smiles"])
        if len(edit_smi) > 50:
            edit_smi = edit_smi[:47] + "..."
        print(f"  {edit_smi:>50s} {row['n_pairs']:5d} {row['mean_delta']:+7.2f} {row['n_targets']:7d}")

    # Seed molecules: top potent + some moderate
    top_mols = mol_data.nlargest(15, "pIC50")
    moderate_pool = mol_data[(mol_data["pIC50"] >= 5.5) & (mol_data["pIC50"] < 7.0)]
    moderate_mols = moderate_pool.sample(
        min(10, len(moderate_pool)), random_state=42
    ) if len(moderate_pool) > 0 else pd.DataFrame()
    seed_mols = pd.concat([top_mols, moderate_mols]).drop_duplicates(subset=["smiles"])

    print(f"\n  Seed molecules for optimization: {len(seed_mols)}")

    # For each seed, check for applicable edits (substructure match)
    print("  Finding applicable edits and predicting deltas with FiLMDelta...")
    optimization_results = []

    for _, seed_row in seed_mols.iterrows():
        seed_smi = seed_row["smiles"]
        seed_pic = seed_row["pIC50"]
        seed_mol = Chem.MolFromSmiles(seed_smi)
        if seed_mol is None:
            continue

        for _, edit_row in beneficial_edits.head(30).iterrows():
            edit_smi = str(edit_row["edit_smiles"])
            if ">>" not in edit_smi:
                continue
            parts = edit_smi.split(">>")
            if len(parts) != 2:
                continue
            leaving, incoming = parts
            leaving_mol = Chem.MolFromSmiles(leaving)
            incoming_mol = Chem.MolFromSmiles(incoming)
            if leaving_mol is None or incoming_mol is None:
                continue
            if not seed_mol.HasSubstructMatch(leaving_mol):
                continue

            # Use FiLMDelta to predict the actual delta for this molecule context
            # We need to find example mol_b from the MMP database for this edit
            # applied to this seed. Since we can't do reaction transforms easily,
            # we predict using the database average AND the model-predicted delta
            # for an exemplar pair with this edit.
            edit_pairs = within[within["edit_smiles"] == edit_row["edit_smiles"]].head(5)
            model_deltas = []
            for _, ep in edit_pairs.iterrows():
                if ep["mol_a"] in fp_cache and ep["mol_b"] in fp_cache:
                    d = model.predict_delta([ep["mol_a"]], [ep["mol_b"]], fp_cache)
                    model_deltas.append(float(d[0]))

            model_predicted_delta = np.mean(model_deltas) if model_deltas else None

            optimization_results.append({
                "seed_smiles": seed_smi,
                "seed_pIC50": round(float(seed_pic), 2),
                "edit": edit_smi,
                "db_mean_delta": round(float(edit_row["mean_delta"]), 3),
                "model_predicted_delta": round(float(model_predicted_delta), 3) if model_predicted_delta is not None else None,
                "expected_new_pIC50_db": round(float(seed_pic + edit_row["mean_delta"]), 3),
                "expected_new_pIC50_model": round(float(seed_pic + model_predicted_delta), 3) if model_predicted_delta is not None else None,
                "edit_confidence": int(edit_row["n_pairs"]),
                "edit_n_targets": int(edit_row["n_targets"]),
            })

    del within
    gc.collect()

    print(f"\n  Applicable optimization suggestions: {len(optimization_results)}")
    optimization_results.sort(key=lambda x: x["expected_new_pIC50_db"], reverse=True)

    if optimization_results:
        print(f"\n  Top 15 optimization suggestions:")
        print(f"  {'Seed':>7} {'DB D':>6} {'Model D':>8} {'Edit':>40} {'Conf':>5}")
        for o in optimization_results[:15]:
            edit_str = o["edit"][:40]
            md = f"{o['model_predicted_delta']:+.2f}" if o["model_predicted_delta"] is not None else "   N/A"
            print(f"  {o['seed_pIC50']:7.2f} {o['db_mean_delta']:+6.2f} {md:>8s} {edit_str:>40s} {o['edit_confidence']:5d}")

    elapsed = time.time() - t0
    results["phase_d"] = {
        "n_beneficial_edits": len(beneficial_edits),
        "n_seed_molecules": len(seed_mols),
        "n_applicable_optimizations": len(optimization_results),
        "top_edits": beneficial_edits.head(20).to_dict(orient="records"),
        "optimization_suggestions": optimization_results[:30],
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase D complete ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase E: Multi-Objective Ranking
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_e(mol_data, results):
    """Combine all candidates with multi-objective scoring."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE E: Multi-Objective Ranking & Final Candidates")
    print("=" * 70)

    all_candidates = []

    # Collect from Phase A
    if "phase_a" in results and "candidates" in results["phase_a"]:
        for c in results["phase_a"]["candidates"]:
            c["source"] = "ChEMBL (untested)"
            all_candidates.append(c)

    # Collect from Phase B
    if "phase_b" in results and "candidates" in results["phase_b"]:
        for c in results["phase_b"]["candidates"]:
            c["source"] = f"Kinase ({c.get('source_kinase', '?')})"
            c["predicted_pIC50"] = c.get("predicted_zap70_pIC50", 0)
            c["nn_similarity"] = c.get("nn_similarity_to_zap70", 0.3)
            all_candidates.append(c)

    # Collect from Phase C
    if "phase_c" in results and "candidates" in results["phase_c"]:
        for c in results["phase_c"]["candidates"]:
            c["source"] = "BRICS generated"
            all_candidates.append(c)

    print(f"  Total candidates from all phases: {len(all_candidates)}")

    if not all_candidates:
        results["phase_e"] = {"message": "No candidates to rank", "completed": True}
        return results

    # Multi-objective scoring:
    # 50% potency + 20% QED + 15% novelty + 15% certainty
    for c in all_candidates:
        potency_score = min(10, max(0, c.get("predicted_pIC50", 0) - 4.0)) / 6.0
        qed_score = c.get("QED", 0.5) if c.get("QED") else 0.5
        sim = c.get("nn_similarity", 0.3)
        novelty_score = 1.0 - abs(sim - 0.5) * 2  # Peaks at sim=0.5
        unc = c.get("uncertainty", 0.5)
        certainty_score = max(0, 1.0 - unc)

        c["composite_score"] = round(
            0.50 * potency_score +
            0.20 * qed_score +
            0.15 * novelty_score +
            0.15 * certainty_score,
            4
        )

    all_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, c in enumerate(all_candidates):
        c["final_rank"] = i + 1

    print(f"\n  === FINAL SCREENING CANDIDATES ===")
    print(f"  {'Rank':>4} {'Source':>20} {'Pred pIC50':>10} {'Score':>6} {'QED':>5}")
    for c in all_candidates[:25]:
        src = c.get("source", "?")[:20]
        qed_str = f"{c['QED']:.2f}" if c.get("QED") else "  -"
        print(f"  {c['final_rank']:4d} {src:>20s} {c.get('predicted_pIC50', 0):10.2f} "
              f"{c['composite_score']:.3f} {qed_str:>5s}")

    elapsed = time.time() - t0
    results["phase_e"] = {
        "n_total_candidates": len(all_candidates),
        "top_candidates": all_candidates[:50],
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase E complete ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase F: Model Interpretation
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_f(mol_data, model, fp_cache, results):
    """Analyze what the FiLMDelta model has learned."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE F: Model Interpretation")
    print("=" * 70)

    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n_mols = len(smiles_list)

    # F1: Anchor prediction analysis — which anchors contribute most/least?
    print("\n  F1. Anchor Prediction Analysis...")
    pred_matrix = model.get_anchor_predictions_matrix(list(smiles_list), fp_cache)
    # pred_matrix[j, i] = anchor_pIC50_i + delta(i->j)

    # For each target molecule, compute mean and std across anchors
    mol_pred_mean = np.mean(pred_matrix, axis=1)
    mol_pred_std = np.std(pred_matrix, axis=1)
    mol_errors = np.abs(mol_pred_mean - pIC50)

    # Identify easy vs hard molecules
    easy_idx = np.argsort(mol_pred_std)[:10]
    hard_idx = np.argsort(mol_pred_std)[-10:]

    print(f"  Molecules: {n_mols}")
    print(f"  Overall anchor prediction: MAE={np.mean(mol_errors):.4f}, "
          f"mean std={np.mean(mol_pred_std):.4f}")

    print(f"\n  'Easy' molecules (lowest prediction variance across anchors):")
    print(f"  {'SMILES':>40} {'True pIC50':>10} {'Pred':>6} {'Std':>6} {'Error':>6}")
    for idx in easy_idx[:5]:
        smi_short = smiles_list[idx][:40]
        print(f"  {smi_short:>40s} {pIC50[idx]:10.2f} {mol_pred_mean[idx]:6.2f} "
              f"{mol_pred_std[idx]:6.3f} {mol_errors[idx]:6.3f}")

    print(f"\n  'Hard' molecules (highest prediction variance across anchors):")
    for idx in hard_idx[-5:]:
        smi_short = smiles_list[idx][:40]
        print(f"  {smi_short:>40s} {pIC50[idx]:10.2f} {mol_pred_mean[idx]:6.2f} "
              f"{mol_pred_std[idx]:6.3f} {mol_errors[idx]:6.3f}")

    # Anchor quality: which anchors are the best/worst predictors overall?
    anchor_errors = np.zeros(n_mols)
    for i in range(n_mols):
        # For anchor i, compute mean abs error of its predictions across all targets
        # pred_matrix[:, i] = predictions of all targets using anchor i
        anchor_errors[i] = np.mean(np.abs(pred_matrix[:, i] - pIC50))

    best_anchors = np.argsort(anchor_errors)[:5]
    worst_anchors = np.argsort(anchor_errors)[-5:]

    print(f"\n  Best anchors (lowest MAE as predictor):")
    print(f"  {'SMILES':>40} {'pIC50':>6} {'Anchor MAE':>10}")
    for idx in best_anchors:
        smi_short = smiles_list[idx][:40]
        print(f"  {smi_short:>40s} {pIC50[idx]:6.2f} {anchor_errors[idx]:10.4f}")

    print(f"\n  Worst anchors (highest MAE as predictor):")
    for idx in worst_anchors:
        smi_short = smiles_list[idx][:40]
        print(f"  {smi_short:>40s} {pIC50[idx]:6.2f} {anchor_errors[idx]:10.4f}")

    anchor_analysis = {
        "overall_mae": float(np.mean(mol_errors)),
        "mean_pred_std": float(np.mean(mol_pred_std)),
        "easy_molecules": [
            {"smiles": smiles_list[idx], "true_pIC50": float(pIC50[idx]),
             "pred": float(mol_pred_mean[idx]), "std": float(mol_pred_std[idx])}
            for idx in easy_idx[:10]
        ],
        "hard_molecules": [
            {"smiles": smiles_list[idx], "true_pIC50": float(pIC50[idx]),
             "pred": float(mol_pred_mean[idx]), "std": float(mol_pred_std[idx])}
            for idx in hard_idx[-10:]
        ],
        "best_anchors": [
            {"smiles": smiles_list[idx], "pIC50": float(pIC50[idx]),
             "anchor_mae": float(anchor_errors[idx])}
            for idx in best_anchors
        ],
        "worst_anchors": [
            {"smiles": smiles_list[idx], "pIC50": float(pIC50[idx]),
             "anchor_mae": float(anchor_errors[idx])}
            for idx in worst_anchors
        ],
    }

    # F2: Per-edit-type accuracy
    print("\n  F2. Per-Edit-Type Accuracy...")
    # Generate all-pairs and compute edit_smiles for ZAP70
    edit_type_results = []
    pair_data = []
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            mol_a = Chem.MolFromSmiles(smiles_list[i])
            mol_b = Chem.MolFromSmiles(smiles_list[j])
            if mol_a is None or mol_b is None:
                continue
            delta_true = float(pIC50[j] - pIC50[i])
            pair_data.append({
                "smi_a": smiles_list[i], "smi_b": smiles_list[j],
                "delta_true": delta_true,
            })

    if pair_data:
        smi_a_list = [p["smi_a"] for p in pair_data]
        smi_b_list = [p["smi_b"] for p in pair_data]
        delta_true_arr = np.array([p["delta_true"] for p in pair_data])

        delta_pred = model.predict_delta(smi_a_list, smi_b_list, fp_cache)
        pair_mae = float(np.mean(np.abs(delta_true_arr - delta_pred)))
        pair_spr = float(spearmanr(delta_true_arr, delta_pred).statistic) if len(delta_true_arr) > 2 else 0.0

        print(f"  All-pairs delta prediction: MAE={pair_mae:.4f}, Spr={pair_spr:.3f} "
              f"(n={len(pair_data):,} pairs)")

        # Group by delta magnitude bins
        abs_delta = np.abs(delta_true_arr)
        bins = [(0, 0.5, "small (|d|<0.5)"), (0.5, 1.0, "medium (0.5-1.0)"),
                (1.0, 2.0, "large (1.0-2.0)"), (2.0, float("inf"), "very large (>2.0)")]
        delta_bin_results = []
        for lo, hi, label in bins:
            mask = (abs_delta >= lo) & (abs_delta < hi)
            if np.sum(mask) > 0:
                bin_mae = float(np.mean(np.abs(delta_true_arr[mask] - delta_pred[mask])))
                bin_n = int(np.sum(mask))
                print(f"    {label}: n={bin_n}, MAE={bin_mae:.4f}")
                delta_bin_results.append({"bin": label, "n": bin_n, "mae": bin_mae})

        edit_type_results = {
            "all_pairs_mae": pair_mae,
            "all_pairs_spearman": pair_spr,
            "n_pairs": len(pair_data),
            "delta_bins": delta_bin_results,
        }
    else:
        edit_type_results = {"error": "No pairs generated"}

    # F3: FiLM layer analysis — extract gamma/beta statistics
    print("\n  F3. FiLM Layer Analysis...")
    film_analysis = analyze_film_layers(model, smiles_list, pIC50, fp_cache)

    # F4: Context dependence — same delta magnitude, different scaffold contexts
    print("\n  F4. Context Dependence Analysis...")
    context_analysis = analyze_context_dependence(model, smiles_list, pIC50, fp_cache)

    elapsed = time.time() - t0
    results["phase_f"] = {
        "anchor_analysis": anchor_analysis,
        "edit_type_accuracy": edit_type_results,
        "film_layer_analysis": film_analysis,
        "context_dependence": context_analysis,
        "time_seconds": round(elapsed, 1),
        "completed": True,
    }
    save_results(results)
    print(f"  Phase F complete ({elapsed:.0f}s)")
    return results


def analyze_film_layers(model, smiles_list, pIC50, fp_cache):
    """Extract gamma/beta statistics from FiLM layers for different edit types."""
    mdl = model.model
    mdl.eval()

    n_mols = len(smiles_list)
    # Sample some pairs for analysis
    np.random.seed(42)
    n_sample = min(1000, n_mols * (n_mols - 1) // 2)
    sample_pairs = []
    all_indices = [(i, j) for i in range(n_mols) for j in range(i + 1, n_mols)]
    if len(all_indices) > n_sample:
        chosen = np.random.choice(len(all_indices), n_sample, replace=False)
        sample_pairs = [all_indices[c] for c in chosen]
    else:
        sample_pairs = all_indices

    smi_a = [smiles_list[i] for i, j in sample_pairs]
    smi_b = [smiles_list[j] for i, j in sample_pairs]
    deltas_true = np.array([pIC50[j] - pIC50[i] for i, j in sample_pairs])

    emb_a = np.array([fp_cache[s] for s in smi_a])
    emb_b = np.array([fp_cache[s] for s in smi_b])
    Xa = torch.FloatTensor(model.scaler.transform(emb_a))
    Xb = torch.FloatTensor(model.scaler.transform(emb_b))

    # Compute delta and encode it
    delta_vec = Xb - Xa
    with torch.no_grad():
        delta_cond = mdl.delta_encoder(delta_vec)

    # Extract gamma/beta for each FiLM layer
    layer_stats = []
    for block_idx, block in enumerate(mdl.blocks):
        film = block.film
        with torch.no_grad():
            gamma = film.gamma_proj(delta_cond).numpy()
            beta = film.beta_proj(delta_cond).numpy()

        stats = {
            "layer": block_idx,
            "gamma_mean": float(np.mean(gamma)),
            "gamma_std": float(np.std(gamma)),
            "gamma_min": float(np.min(gamma)),
            "gamma_max": float(np.max(gamma)),
            "beta_mean": float(np.mean(beta)),
            "beta_std": float(np.std(beta)),
            "beta_min": float(np.min(beta)),
            "beta_max": float(np.max(beta)),
        }
        layer_stats.append(stats)
        print(f"    Layer {block_idx}: gamma={stats['gamma_mean']:.3f}+-{stats['gamma_std']:.3f}, "
              f"beta={stats['beta_mean']:.3f}+-{stats['beta_std']:.3f}")

    # Gamma variance across edits — higher = more edit-specific modulation
    # Partition by delta sign (beneficial vs detrimental edits)
    pos_mask = deltas_true > 0
    neg_mask = deltas_true < 0

    direction_stats = {}
    for label, mask in [("beneficial", pos_mask), ("detrimental", neg_mask)]:
        if np.sum(mask) < 10:
            continue
        subset_cond = delta_cond[mask]
        with torch.no_grad():
            g0 = mdl.blocks[0].film.gamma_proj(subset_cond).numpy()
        direction_stats[label] = {
            "n_pairs": int(np.sum(mask)),
            "layer0_gamma_mean": float(np.mean(g0)),
            "layer0_gamma_std": float(np.std(g0)),
        }
        print(f"    {label.capitalize()} edits (n={np.sum(mask)}): "
              f"Layer0 gamma={np.mean(g0):.3f}+-{np.std(g0):.3f}")

    return {"layer_stats": layer_stats, "direction_stats": direction_stats}


def analyze_context_dependence(model, smiles_list, pIC50, fp_cache):
    """Check if same-magnitude deltas produce different predictions in different contexts."""
    n_mols = len(smiles_list)

    # Find pairs with similar true deltas but different scaffolds
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds[smi] = Chem.MolToSmiles(scaf)
            except Exception:
                scaffolds[smi] = "unknown"
        else:
            scaffolds[smi] = "unknown"

    # Group pairs by approximate delta magnitude (bins of 0.25 pIC50)
    bin_size = 0.25
    pair_bins = defaultdict(list)
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            d = pIC50[j] - pIC50[i]
            bin_key = round(d / bin_size) * bin_size
            pair_bins[bin_key].append((i, j, d))

    # Find bins with pairs from different scaffolds
    context_examples = []
    for bin_key, pairs in sorted(pair_bins.items(), key=lambda x: -len(x[1])):
        if len(pairs) < 5:
            continue
        # Compute model predictions for these pairs
        smi_a = [smiles_list[p[0]] for p in pairs[:20]]
        smi_b = [smiles_list[p[1]] for p in pairs[:20]]
        true_d = np.array([p[2] for p in pairs[:20]])
        pred_d = model.predict_delta(smi_a, smi_b, fp_cache)
        errors = np.abs(true_d - pred_d)

        # Check scaffold diversity
        scaf_set = set()
        for p in pairs[:20]:
            scaf_set.add(scaffolds.get(smiles_list[p[0]], "?"))
            scaf_set.add(scaffolds.get(smiles_list[p[1]], "?"))

        context_examples.append({
            "delta_bin": float(bin_key),
            "n_pairs": len(pairs),
            "n_analyzed": len(smi_a),
            "n_scaffolds": len(scaf_set),
            "mae": float(np.mean(errors)),
            "pred_std": float(np.std(pred_d)),
        })

        if len(context_examples) >= 10:
            break

    print(f"  Analyzed {len(context_examples)} delta bins for context dependence:")
    print(f"  {'Delta Bin':>10} {'N pairs':>8} {'Scaffolds':>9} {'MAE':>6} {'Pred Std':>9}")
    for ex in context_examples[:8]:
        print(f"  {ex['delta_bin']:+10.2f} {ex['n_pairs']:8d} {ex['n_scaffolds']:9d} "
              f"{ex['mae']:6.3f} {ex['pred_std']:9.3f}")

    return {"context_bins": context_examples}


# ═══════════════════════════════════════════════════════════════════════════
# Phase G: HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_g(mol_data, results):
    """Generate comprehensive HTML report."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE G: Generating HTML Report")
    print("=" * 70)

    target_info = {
        "name": "ZAP70 (Tyrosine-protein kinase ZAP-70)",
        "full_name": "Zeta-chain-associated protein kinase 70",
        "gene": "ZAP70",
        "uniprot": "P43403",
        "function": "Non-receptor tyrosine kinase essential for T-cell receptor (TCR) signaling",
        "disease_relevance": [
            "Autoimmune diseases (T-cell mediated inflammation)",
            "Transplant rejection (T-cell activation)",
            "CLL prognosis (ZAP-70 expression = aggressive disease)",
            "Severe combined immunodeficiency (ZAP-70 deficiency causes SCID)",
        ],
    }

    html = []
    html.append("<!DOCTYPE html><html><head>")
    html.append("<title>ZAP70 Virtual Screening — FiLMDelta + Kinase PT</title>")
    html.append("""<style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto;
               padding: 20px; background: #f5f5f5; color: #333; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #1a5276, #2980b9); color: white;
                  padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 28px; }
        .header p { margin: 5px 0 0; opacity: 0.9; font-size: 15px; }
        .section { background: white; padding: 25px; border-radius: 8px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }
        h2 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 8px; }
        h3 { color: #2c3e50; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
        th { background: #2c3e50; color: white; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f0f7ff; }
        .metric-box { display: inline-block; background: #eaf2f8; padding: 15px 25px;
                      border-radius: 8px; margin: 5px; text-align: center; min-width: 120px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #1a5276; }
        .metric-label { font-size: 12px; color: #666; margin-top: 4px; }
        .highlight { background: #d5f5e3; padding: 2px 6px; border-radius: 3px; }
        .warning { background: #fdebd0; padding: 2px 6px; border-radius: 3px; }
        .good { color: #27ae60; font-weight: bold; }
        .smi { font-family: monospace; font-size: 11px; word-break: break-all;
               max-width: 300px; display: inline-block; }
        .note { background: #fef9e7; padding: 15px; border-left: 4px solid #f39c12;
                border-radius: 4px; margin: 15px 0; }
        .clinical { background: #eaf2f8; padding: 15px; border-left: 4px solid #2980b9;
                    border-radius: 4px; margin: 15px 0; }
        ul { margin: 5px 0; padding-left: 20px; }
    </style>""")
    html.append("</head><body>")

    # Header
    html.append('<div class="header">')
    html.append(f'<h1>Virtual Screening Report: {TARGET_NAME}</h1>')
    html.append(f'<p>FiLMDelta + Kinase Pretraining with Anchor-Based Prediction</p>')
    html.append(f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
                f'Training set: {len(mol_data)} molecules | '
                f'Model: FiLMDelta [1024,512,256] + kinase PT + ZAP70 fine-tune</p>')
    html.append('</div>')

    # Section 1: Target Overview + Model
    html.append('<div class="section">')
    html.append('<h2>1. Target & Model Overview</h2>')
    html.append('<div class="clinical">')
    html.append(f'<strong>{target_info["name"]}</strong> ({target_info["gene"]}, UniProt: {target_info["uniprot"]})<br>')
    html.append(f'{target_info["function"]}')
    html.append('</div>')
    html.append('<h3>Disease Relevance</h3><ul>')
    for d in target_info["disease_relevance"]:
        html.append(f'<li>{d}</li>')
    html.append('</ul>')

    html.append('<h3>Prediction Model</h3>')
    html.append('''<ul>
        <li><strong>Architecture</strong>: FiLMDelta (Feature-wise Linear Modulation) with hidden dims [1024, 512, 256]</li>
        <li><strong>Pretraining</strong>: Kinase within-assay MMP pairs (~32K pairs, 8 kinase targets)</li>
        <li><strong>Fine-tuning</strong>: ZAP70 all-pairs (280 molecules, ~78K ordered pairs)</li>
        <li><strong>Inference</strong>: Anchor-based absolute prediction: pred(j) = mean_i(known_pIC50_i + delta(i&rarr;j))</li>
        <li><strong>Uncertainty</strong>: Standard deviation of individual anchor predictions</li>
        <li><strong>Embeddings</strong>: Morgan fingerprints (2048-bit, radius 2)</li>
    </ul>''')

    # Training summary
    html.append('<h3>Training Data</h3>')
    html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
    model_info = results.get("model_training", {})
    for label, value in [
        ("Molecules", str(len(mol_data))),
        ("pIC50 Range", f"{mol_data['pIC50'].min():.1f}&ndash;{mol_data['pIC50'].max():.1f}"),
        ("Mean pIC50", f"{mol_data['pIC50'].mean():.2f}"),
        ("Kinase PT Loss", f"{model_info.get('pretrain_val_loss', '?')}"),
        ("Fine-tune MAE", f"{model_info.get('finetune_mae', '?')}"),
    ]:
        html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>')
    html.append('</div>')
    html.append('</div>')

    # Section 2: Phase A
    if "phase_a" in results and results["phase_a"].get("completed"):
        pa = results["phase_a"]
        html.append('<div class="section">')
        html.append('<h2>2. ChEMBL Compound Screening</h2>')
        html.append(f'<p>Scored <strong>{pa["n_candidates"]}</strong> ChEMBL compounds not in training set '
                    f'using FiLMDelta anchor-based prediction.</p>')

        html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
        for label, value in [
            ("Total Screened", str(pa["n_candidates"])),
            ("Predicted Potent", f'{pa["n_potent"]} (pIC50&ge;7)'),
            ("Moderate", f'{pa["n_moderate"]} (6-7)'),
            ("High Confidence", str(pa.get("n_high_conf_potent", 0))),
            ("Time", f'{pa.get("time_seconds", 0):.0f}s'),
        ]:
            html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div></div>')
        html.append('</div>')

        val = pa.get("validation", {})
        if val.get("n_with_known", 0) > 0:
            html.append(f'<div class="note"><strong>Validation:</strong> '
                        f'{val["n_with_known"]} molecules had known pChEMBL values. '
                        f'Prediction MAE = {val.get("mae", "?"):.3f}, '
                        f'Spearman = {val.get("spearman", "?"):.3f}</div>')

        if pa.get("candidates"):
            html.append('<h3>Top Screening Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>ChEMBL ID</th><th>Pred pIC50</th>'
                        '<th>Uncertainty</th><th>NN Sim</th><th>Known</th>'
                        '<th>QED</th><th>SMILES</th></tr>')
            for c in pa["candidates"][:30]:
                known = f"{c['known_value']:.1f}" if c.get("known_value") else "&mdash;"
                qed = f"{c['QED']:.2f}" if c.get("QED") else "&mdash;"
                cls = "highlight" if c["predicted_pIC50"] >= 7.0 else ""
                html.append(f'<tr><td>{c["rank"]}</td><td>{c["chembl_id"]}</td>'
                            f'<td class="{cls}">{c["predicted_pIC50"]:.2f}</td>'
                            f'<td>{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity"]:.3f}</td><td>{known}</td>'
                            f'<td>{qed}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 3: Phase B
    if "phase_b" in results and results["phase_b"].get("completed"):
        pb = results["phase_b"]
        html.append('<div class="section">')
        html.append('<h2>3. Kinase Compound Repurposing</h2>')
        html.append(f'<p>Screened potent compounds from related kinases for ZAP70 activity.</p>')

        html.append('<h3>Source Kinases</h3>')
        html.append('<table><tr><th>Kinase</th><th>Potent Compounds</th></tr>')
        for k, n in pb.get("kinase_sources", {}).items():
            html.append(f'<tr><td>{k}</td><td>{n}</td></tr>')
        html.append('</table>')

        html.append(f'<p>Novel kinase compounds scored: <strong>{pb["n_novel_kinase_mols"]}</strong>, '
                    f'predicted active (pIC50&ge;6.5): <strong class="good">{pb["n_predicted_active"]}</strong></p>')

        if pb.get("candidates"):
            html.append('<h3>Top Repurposing Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>Source</th><th>Pred ZAP70 pIC50</th>'
                        '<th>Uncertainty</th><th>Similarity</th><th>Source pIC50</th><th>SMILES</th></tr>')
            for c in pb["candidates"][:20]:
                src_val = f"{c['source_pchembl']:.1f}" if c.get("source_pchembl") else "&mdash;"
                html.append(f'<tr><td>{c["rank"]}</td><td>{c["source_kinase"]}</td>'
                            f'<td>{c["predicted_zap70_pIC50"]:.2f}</td>'
                            f'<td>{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity_to_zap70"]:.3f}</td>'
                            f'<td>{src_val}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 4: Phase C
    if "phase_c" in results and results["phase_c"].get("completed"):
        pc = results["phase_c"]
        html.append('<div class="section">')
        html.append('<h2>4. BRICS-Generated Molecules</h2>')
        html.append(f'<p>Generated <strong>{pc["n_generated"]}</strong> novel molecules via BRICS '
                    f'recombination of fragments from top-20 potent molecules.</p>')

        html.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
        for label, value in [
            ("Generated", str(pc["n_generated"])),
            ("Predicted Potent", str(pc["n_potent"])),
            ("Drug-like", str(pc["n_druglike"])),
            ("BRICS Fragments", str(pc["n_brics_fragments"])),
        ]:
            html.append(f'<div class="metric-box"><div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div></div>')
        html.append('</div>')

        if pc.get("top_scaffolds"):
            html.append('<h3>Source Scaffolds</h3>')
            html.append('<table><tr><th>Scaffold</th><th>N Mols</th><th>Mean pIC50</th></tr>')
            for s in pc["top_scaffolds"]:
                html.append(f'<tr><td><span class="smi">{s["scaffold"][:80]}</span></td>'
                            f'<td>{s["n_mols"]}</td><td>{s["mean_pIC50"]:.2f}</td></tr>')
            html.append('</table>')

        if pc.get("candidates"):
            html.append('<h3>Top Generated Candidates</h3>')
            html.append('<table><tr><th>Rank</th><th>Pred pIC50</th><th>Uncertainty</th>'
                        '<th>Similarity</th><th>QED</th><th>SA</th><th>SMILES</th></tr>')
            for c in pc["candidates"][:20]:
                qed = f"{c['QED']:.2f}" if c.get("QED") else "&mdash;"
                sa = f"{c['SA_score']:.1f}" if c.get("SA_score") else "&mdash;"
                html.append(f'<tr><td>{c["rank"]}</td>'
                            f'<td>{c["predicted_pIC50"]:.2f}</td>'
                            f'<td>{c["uncertainty"]:.2f}</td>'
                            f'<td>{c["nn_similarity"]:.3f}</td>'
                            f'<td>{qed}</td><td>{sa}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 5: Phase D
    if "phase_d" in results and results["phase_d"].get("completed"):
        pd_res = results["phase_d"]
        html.append('<div class="section">')
        html.append('<h2>5. MMP-Guided Optimization (FiLMDelta Deltas)</h2>')
        html.append(f'<p>Identified <strong>{pd_res["n_beneficial_edits"]}</strong> beneficial edits. '
                    f'Unlike database averages, FiLMDelta predicts context-specific deltas.</p>')

        if pd_res.get("top_edits"):
            html.append('<h3>Top Beneficial Edits</h3>')
            html.append('<table><tr><th>Edit</th><th>N Pairs</th>'
                        '<th>Mean &Delta; pIC50</th><th>N Targets</th></tr>')
            for e in pd_res["top_edits"][:15]:
                edit_str = str(e.get("edit_smiles", "?"))
                if len(edit_str) > 60:
                    edit_str = edit_str[:57] + "..."
                html.append(f'<tr><td><span class="smi">{edit_str}</span></td>'
                            f'<td>{e.get("n_pairs", 0)}</td>'
                            f'<td class="good">{e.get("mean_delta", 0):+.2f}</td>'
                            f'<td>{e.get("n_targets", 0)}</td></tr>')
            html.append('</table>')

        if pd_res.get("optimization_suggestions"):
            html.append(f'<h3>Optimization Suggestions ({pd_res["n_applicable_optimizations"]} found)</h3>')
            html.append('<table><tr><th>Seed pIC50</th><th>Edit</th><th>DB &Delta;</th>'
                        '<th>Model &Delta;</th><th>Expected pIC50</th><th>Confidence</th></tr>')
            for o in pd_res["optimization_suggestions"][:15]:
                md = f"{o['model_predicted_delta']:+.2f}" if o.get("model_predicted_delta") is not None else "&mdash;"
                exp = o.get("expected_new_pIC50_model") or o["expected_new_pIC50_db"]
                html.append(f'<tr><td>{o["seed_pIC50"]:.2f}</td>'
                            f'<td><span class="smi">{o["edit"][:50]}</span></td>'
                            f'<td>{o["db_mean_delta"]:+.2f}</td>'
                            f'<td>{md}</td>'
                            f'<td><strong>{exp:.2f}</strong></td>'
                            f'<td>{o["edit_confidence"]} pairs</td></tr>')
            html.append('</table>')
        else:
            html.append('<div class="note">No directly applicable edits found for seed molecules.</div>')
        html.append('</div>')

    # Section 6: Phase E
    if "phase_e" in results and results["phase_e"].get("completed"):
        pe = results["phase_e"]
        html.append('<div class="section">')
        html.append('<h2>6. Final Multi-Objective Rankings</h2>')
        html.append('<p>Combined candidates scored by: potency (50%), druglikeness (20%), '
                    'novelty (15%), prediction confidence (15%).</p>')
        html.append(f'<p>Total candidates: <strong>{pe["n_total_candidates"]}</strong></p>')

        if pe.get("top_candidates"):
            html.append('<table><tr><th>Rank</th><th>Source</th><th>Pred pIC50</th>'
                        '<th>Composite Score</th><th>Uncertainty</th><th>QED</th><th>SMILES</th></tr>')
            for c in pe["top_candidates"][:30]:
                src = c.get("source", "?")
                qed = f"{c['QED']:.2f}" if c.get("QED") else "&mdash;"
                unc = f"{c.get('uncertainty', 0):.2f}"
                html.append(f'<tr><td>{c["final_rank"]}</td><td>{src}</td>'
                            f'<td><strong>{c.get("predicted_pIC50", 0):.2f}</strong></td>'
                            f'<td>{c["composite_score"]:.3f}</td>'
                            f'<td>{unc}</td><td>{qed}</td>'
                            f'<td><span class="smi">{c["smiles"][:80]}</span></td></tr>')
            html.append('</table>')
        html.append('</div>')

    # Section 7: Phase F — Model Interpretation
    if "phase_f" in results and results["phase_f"].get("completed"):
        pf = results["phase_f"]
        html.append('<div class="section">')
        html.append('<h2>7. Model Interpretation</h2>')

        # Anchor analysis
        aa = pf.get("anchor_analysis", {})
        html.append('<h3>7.1 Anchor Prediction Analysis</h3>')
        html.append(f'<p>Overall anchor-based MAE: <strong>{aa.get("overall_mae", "?")}</strong>, '
                    f'mean prediction std across anchors: <strong>{aa.get("mean_pred_std", "?")}</strong></p>')

        html.append('<h4>"Easy" Molecules (lowest variance across anchors)</h4>')
        html.append('<table><tr><th>SMILES</th><th>True pIC50</th><th>Predicted</th><th>Std</th></tr>')
        for m in aa.get("easy_molecules", [])[:5]:
            html.append(f'<tr><td><span class="smi">{m["smiles"][:60]}</span></td>'
                        f'<td>{m["true_pIC50"]:.2f}</td><td>{m["pred"]:.2f}</td>'
                        f'<td>{m["std"]:.3f}</td></tr>')
        html.append('</table>')

        html.append('<h4>"Hard" Molecules (highest variance across anchors)</h4>')
        html.append('<table><tr><th>SMILES</th><th>True pIC50</th><th>Predicted</th><th>Std</th></tr>')
        for m in aa.get("hard_molecules", [])[-5:]:
            html.append(f'<tr><td><span class="smi">{m["smiles"][:60]}</span></td>'
                        f'<td>{m["true_pIC50"]:.2f}</td><td>{m["pred"]:.2f}</td>'
                        f'<td>{m["std"]:.3f}</td></tr>')
        html.append('</table>')

        html.append('<h4>Best & Worst Anchors</h4>')
        html.append('<table><tr><th>Type</th><th>SMILES</th><th>pIC50</th><th>Anchor MAE</th></tr>')
        for m in aa.get("best_anchors", []):
            html.append(f'<tr><td class="highlight">Best</td>'
                        f'<td><span class="smi">{m["smiles"][:60]}</span></td>'
                        f'<td>{m["pIC50"]:.2f}</td><td>{m["anchor_mae"]:.4f}</td></tr>')
        for m in aa.get("worst_anchors", []):
            html.append(f'<tr><td class="warning">Worst</td>'
                        f'<td><span class="smi">{m["smiles"][:60]}</span></td>'
                        f'<td>{m["pIC50"]:.2f}</td><td>{m["anchor_mae"]:.4f}</td></tr>')
        html.append('</table>')

        # Per-edit-type accuracy
        eta = pf.get("edit_type_accuracy", {})
        html.append('<h3>7.2 Delta Prediction Accuracy by Magnitude</h3>')
        html.append(f'<p>All-pairs delta prediction: MAE={eta.get("all_pairs_mae", "?")}, '
                    f'Spearman={eta.get("all_pairs_spearman", "?")}, '
                    f'n={eta.get("n_pairs", "?"):,} pairs</p>')
        if eta.get("delta_bins"):
            html.append('<table><tr><th>Delta Bin</th><th>N Pairs</th><th>MAE</th></tr>')
            for b in eta["delta_bins"]:
                html.append(f'<tr><td>{b["bin"]}</td><td>{b["n"]}</td><td>{b["mae"]:.4f}</td></tr>')
            html.append('</table>')

        # FiLM layer analysis
        fla = pf.get("film_layer_analysis", {})
        html.append('<h3>7.3 FiLM Layer Statistics</h3>')
        html.append('<p>Gamma (&gamma;) and beta (&beta;) parameters from FiLM layers, '
                    'showing how edits modulate the prediction network.</p>')
        if fla.get("layer_stats"):
            html.append('<table><tr><th>Layer</th><th>&gamma; mean</th><th>&gamma; std</th>'
                        '<th>&beta; mean</th><th>&beta; std</th></tr>')
            for ls in fla["layer_stats"]:
                html.append(f'<tr><td>{ls["layer"]}</td>'
                            f'<td>{ls["gamma_mean"]:.3f}</td><td>{ls["gamma_std"]:.3f}</td>'
                            f'<td>{ls["beta_mean"]:.3f}</td><td>{ls["beta_std"]:.3f}</td></tr>')
            html.append('</table>')

        if fla.get("direction_stats"):
            html.append('<h4>Beneficial vs Detrimental Edits</h4>')
            html.append('<table><tr><th>Direction</th><th>N Pairs</th><th>Layer0 &gamma; mean</th>'
                        '<th>Layer0 &gamma; std</th></tr>')
            for label, stats in fla["direction_stats"].items():
                html.append(f'<tr><td>{label.capitalize()}</td><td>{stats["n_pairs"]}</td>'
                            f'<td>{stats["layer0_gamma_mean"]:.3f}</td>'
                            f'<td>{stats["layer0_gamma_std"]:.3f}</td></tr>')
            html.append('</table>')

        # Context dependence
        cd = pf.get("context_dependence", {})
        if cd.get("context_bins"):
            html.append('<h3>7.4 Context Dependence</h3>')
            html.append('<p>Do same-magnitude deltas produce different predictions in different scaffold contexts?</p>')
            html.append('<table><tr><th>&Delta; Bin</th><th>N Pairs</th><th>N Scaffolds</th>'
                        '<th>MAE</th><th>Pred Std</th></tr>')
            for ex in cd["context_bins"]:
                html.append(f'<tr><td>{ex["delta_bin"]:+.2f}</td><td>{ex["n_pairs"]}</td>'
                            f'<td>{ex["n_scaffolds"]}</td><td>{ex["mae"]:.3f}</td>'
                            f'<td>{ex["pred_std"]:.3f}</td></tr>')
            html.append('</table>')

        html.append('</div>')

    # Section 8: Methodology & Limitations
    html.append('<div class="section">')
    html.append('<h2>8. Methodology & Limitations</h2>')
    html.append('''
    <h3>Key Differences from v5 (XGB Ensemble)</h3>
    <ul>
        <li><strong>Model</strong>: FiLMDelta neural network (vs. XGBoost/RF ensemble)</li>
        <li><strong>Pretraining</strong>: Kinase within-assay MMP pairs provide chemical transformation knowledge</li>
        <li><strong>Prediction</strong>: Anchor-based via delta prediction (vs. direct property prediction)</li>
        <li><strong>Uncertainty</strong>: Std of anchor predictions (vs. ensemble model disagreement)</li>
        <li><strong>MMP optimization</strong>: FiLMDelta-predicted deltas (vs. database averages only)</li>
    </ul>

    <h3>Limitations</h3>
    <div class="note">
        <ul>
            <li><strong>Applicability domain</strong>: Predictions are most reliable for molecules
                structurally similar to the training set (Tanimoto &ge; 0.3).</li>
            <li><strong>Anchor prediction variance</strong>: High variance indicates the molecule is
                "far" from consensus — these predictions are less reliable.</li>
            <li><strong>Kinase pretraining bias</strong>: The model has learned kinase-specific SAR patterns
                which may not transfer to all chemical scaffolds.</li>
            <li><strong>BRICS molecules</strong>: Generated molecules may not be synthetically accessible.</li>
            <li><strong>No 3D/docking</strong>: This is a ligand-based screen. Structure-based
                validation recommended for top candidates.</li>
        </ul>
    </div>
    ''')
    html.append('</div>')

    # Timing summary
    html.append('<div class="section">')
    html.append('<h2>9. Timing Summary</h2>')
    html.append('<table><tr><th>Phase</th><th>Time (s)</th></tr>')
    total_time = 0
    for phase_key in ["model_training", "phase_a", "phase_b", "phase_c", "phase_d", "phase_e", "phase_f"]:
        if phase_key in results:
            t = results[phase_key].get("time_seconds", 0)
            total_time += t
            label = phase_key.replace("_", " ").title()
            html.append(f'<tr><td>{label}</td><td>{t:.1f}</td></tr>')
    html.append(f'<tr><td><strong>Total</strong></td><td><strong>{total_time:.1f}</strong></td></tr>')
    html.append('</table>')
    html.append('</div>')

    # Footer
    html.append('<div style="text-align: center; color: #999; padding: 20px; font-size: 12px;">')
    html.append(f'Generated by Edit Effect Framework | {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
                f'FiLMDelta + Kinase PT Virtual Screening Pipeline')
    html.append('</div>')

    html.append("</body></html>")

    report_text = "\n".join(html)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)

    elapsed = time.time() - t0
    print(f"  Report saved to {REPORT_FILE}")
    print(f"  Report size: {len(report_text):,} characters ({elapsed:.0f}s)")
    results["report"] = {"path": str(REPORT_FILE), "size": len(report_text)}
    save_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("=" * 70)
    print(f"ZAP70 (CHEMBL2803) Virtual Screening — FiLMDelta + Kinase PT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load ZAP70 data
    print("\n--- Loading ZAP70 data ---")
    mol_data, per_assay = load_zap70_molecules()
    smiles_list = mol_data["smiles"].values
    pIC50 = mol_data["pIC50"].values
    n_mols = len(smiles_list)

    # Load or initialize results
    results = {}

    results["data_summary"] = {
        "target": TARGET_ID,
        "target_name": TARGET_NAME,
        "n_molecules": n_mols,
        "pIC50_range": [float(pIC50.min()), float(pIC50.max())],
        "pIC50_mean": float(pIC50.mean()),
        "pIC50_std": float(pIC50.std()),
    }

    # ── Core Model Training ──
    t_model = time.time()
    print("\n--- Core Model Training ---")

    # Load kinase pairs
    print("  Loading kinase within-assay pairs...")
    kinase_pairs = pd.read_csv(KINASE_PAIRS_FILE, usecols=["mol_a", "mol_b", "delta"])
    print(f"  Kinase pairs: {len(kinase_pairs):,}")

    # Compute Morgan FP for all needed molecules
    print("  Computing Morgan FP for all molecules...")
    all_kinase_smi = list(set(
        kinase_pairs["mol_a"].tolist() + kinase_pairs["mol_b"].tolist()
    ))
    all_smi = list(set(all_kinase_smi + list(smiles_list)))
    print(f"  {len(all_smi):,} unique molecules")
    X_all = compute_fingerprints(all_smi, "morgan", radius=2, n_bits=2048)
    fp_cache = {smi: X_all[i] for i, smi in enumerate(all_smi)}
    del X_all, all_kinase_smi, all_smi
    gc.collect()

    # Filter kinase pairs to those with FP
    mask = kinase_pairs["mol_a"].apply(lambda s: s in fp_cache) & \
           kinase_pairs["mol_b"].apply(lambda s: s in fp_cache)
    kinase_pairs = kinase_pairs[mask].reset_index(drop=True)
    print(f"  Kinase pairs (with FP): {len(kinase_pairs):,}")

    # Build model
    model = FiLMDeltaAnchorModel()

    # Step 1: Pretrain on kinase pairs
    print("\n  Step 1: Pretrain on kinase within-assay pairs...")
    pt_metrics = model.pretrain_on_kinase(
        kinase_pairs, fp_cache,
        epochs=100, batch_size=256, lr=1e-3, patience=15
    )
    del kinase_pairs
    gc.collect()

    # Step 2: Fine-tune on ZAP70 all-pairs (ALL 280 molecules for screening)
    print("\n  Step 2: Fine-tune on ZAP70 all-pairs (full dataset)...")
    ft_metrics = model.finetune_on_zap70(
        list(smiles_list), pIC50, fp_cache,
        epochs=50, batch_size=256, lr=1e-4, patience=15
    )

    model_time = time.time() - t_model
    results["model_training"] = {
        "pretrain_metrics": pt_metrics,
        "pretrain_val_loss": round(pt_metrics["best_val_loss"], 4),
        "finetune_metrics": ft_metrics,
        "finetune_mae": round(ft_metrics["mae"], 4),
        "finetune_spearman": round(ft_metrics["spearman"], 3),
        "n_kinase_pairs": int(mask.sum()),
        "n_zap70_molecules": n_mols,
        "architecture": "FiLMDelta [1024, 512, 256]",
        "time_seconds": round(model_time, 1),
    }
    save_results(results)
    print(f"\n  Model training complete ({model_time:.0f}s)")
    gc.collect()

    # ── Run all phases ──
    results = run_phase_a(mol_data, model, fp_cache, results)
    gc.collect()

    results = run_phase_b(mol_data, model, fp_cache, results)
    gc.collect()

    results = run_phase_c(mol_data, model, fp_cache, results)
    gc.collect()

    results = run_phase_d(mol_data, model, fp_cache, results)
    gc.collect()

    results = run_phase_e(mol_data, results)
    gc.collect()

    results = run_phase_f(mol_data, model, fp_cache, results)
    gc.collect()

    results = run_phase_g(mol_data, results)

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed
    results["completed"] = True
    save_results(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print(f"Results: {RESULTS_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
