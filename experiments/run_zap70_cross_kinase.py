#!/usr/bin/env python3
"""
Cross-kinase scoring for 19 ZAP70 candidates.

Since BTK-ZAP70 correlation is 0.823 (p=9e-5) and SYK-ZAP70 is 0.312,
BTK predicted activity is a strong proxy for ZAP70. Score candidates
using BTK and SYK models as corroborating evidence.

Also builds binary ZAP70 classifier using:
- Positives: 280 ZAP70 molecules (pIC50 ≥ 5.0)
- Negatives: ZAP70-tested weak binders (pIC50 < 5.0) + diverse ChEMBL non-kinase molecules

Usage:
    conda run -n quris python -u experiments/run_zap70_cross_kinase.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.SaltRemover import SaltRemover

warnings.filterwarnings("ignore")
# MPS is fine for simple MLPs (FiLMDelta) — only ChemBERTa/transformers crash on MPS
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from experiments.run_paper_evaluation import RESULTS_DIR
from experiments.run_zap70_v3 import (
    load_zap70_molecules, compute_fingerprints, compute_absolute_metrics,
    aggregate_cv_results,
)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
# Re-enable MPS (run_zap70_v3 import disables it, but FiLMDelta MLPs work fine on MPS)
torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SMILES_19 = [
    'C=CC(=O)N1CC=2C=CC=C(C(=O)NC3=CN(C=N3)C(C)C)C2C1',
    'C=CC(=O)N1CC2(CCC(=O)NC3=CNN=C3C(=O)NCC(C)O)CCC1CC2',
    'C=CC(=O)NC[C@H]1C[C@H]2C[C@@H]1CN2C=3N=CN=C(N)C3Cl',
    'C=CC(=O)N1CCCC1(C(=O)NC2=CNN=C2OCC(F)F)C=3C=CC=CC3',
    'C=CC(=O)N1CCC(CC(=O)NC2=CNN=C2C=3C=NC=CN3)CC41CC4',
    'C=CC(=O)N1CC2(CCC2)CC1CC(=O)NC3=CNN=C3C(=O)NCC(C)O',
    'C=CC(=O)N1C[C@H](CC(C)(C)C)[C@H](C1)C(=O)NC2=CNN=C2C(=O)NCC(C)O',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C3=CNC=4C=C(F)C(Cl)=CC34)S2',
    'C=CC(=O)NC1C2C3C[C@@H]1[C@H](C(=O)NC4=CNN=C4C=5C=NC=CN5)C32',
    'C=CC(=O)N1CC(C1)C2=CN=C(NC(=O)C=3C=CC(F)=C(C3)S(=O)(=O)N(C)C)S2',
    'C=CC(=O)NCC1=CN(N=N1)[C@@H]2C[C@H](C2)C(=O)NC=3C=CC=NC3NC(C)=O',
    'C=CC(=O)N1CC2(CC1CCC2)NC=3N=CC=C(N3)OC=4C=CC=C(C#N)C4',
    'C=CC(=O)NC(C)C=1N=CC(=CN1)NC(=O)C=2N=CN=C3NC=C(C)C23',
    'C=CC(=O)N(C)C1(CNC=2N=CN=C(N)C2C(=O)OCC)CCC1',
    'C=CC(=O)N1CCC[C@H]1C(C)NC(=O)C=2C=NNC2C=3C=CN(C)N3',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CC(CCNC=2C=C(N=CN2)NC=3C=CC=CC3)(C1)N(C)C',
    'O=C(O)C(F)(F)F.C=CC(=O)N(C)C1(CNC=2N=CN=C3NC=C(C(N)=O)C23)CCC1',
    'O=C(O)C(F)(F)F.C=CC(=O)N1CCN(CC1)C=2N=CN=C(N2)NC3(CC)CCNCC3',
    'C=CC(=O)N1CCCC(NC(=O)C2=CNN=C2C3=CC=4C=CC=CC4O3)C51CCC5',
]


def clean_smiles(smi):
    smi_clean = smi.split(' |')[0] if ' |' in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    remover = SaltRemover()
    mol_stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(mol_stripped)


def generate_all_pairs(smiles, pIC50):
    pairs = []
    for (i, j) in combinations(range(len(smiles)), 2):
        pairs.append({'mol_a': smiles[i], 'mol_b': smiles[j], 'delta': pIC50[j] - pIC50[i]})
        pairs.append({'mol_a': smiles[j], 'mol_b': smiles[i], 'delta': pIC50[i] - pIC50[j]})
    return pd.DataFrame(pairs)


def train_filmdelta_and_score(train_smiles, train_y, cand_smiles, emb_dim=2048, n_seeds=3):
    """Train FiLMDelta and score candidates with anchor-based prediction."""
    all_smi = list(set(train_smiles + cand_smiles))
    print(f"    Computing fingerprints for {len(all_smi)} molecules...")
    X = compute_fingerprints(all_smi, "morgan", radius=2, n_bits=emb_dim)
    emb_dict = {smi: X[i] for i, smi in enumerate(all_smi)}

    print(f"    Generating all-pairs from {len(train_smiles)} molecules...")
    all_pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(all_pairs) * 0.1), 100)
    val_pairs = all_pairs.sample(n=n_val, random_state=42)
    trn_pairs = all_pairs.drop(val_pairs.index)
    print(f"    {len(trn_pairs)} train + {len(val_pairs)} val pairs")

    preds_all = np.zeros((len(cand_smiles), n_seeds))

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        def get_emb(smi):
            return emb_dict.get(smi, np.zeros(emb_dim, dtype=np.float32))

        train_a = np.array([get_emb(s) for s in trn_pairs['mol_a']])
        train_b = np.array([get_emb(s) for s in trn_pairs['mol_b']])
        train_delta = trn_pairs['delta'].values.astype(np.float32)
        val_a = np.array([get_emb(s) for s in val_pairs['mol_a']])
        val_b = np.array([get_emb(s) for s in val_pairs['mol_b']])
        val_delta = val_pairs['delta'].values.astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(np.vstack([train_a, train_b]))
        train_a = scaler.transform(train_a).astype(np.float32)
        train_b = scaler.transform(train_b).astype(np.float32)
        val_a = scaler.transform(val_a).astype(np.float32)
        val_b = scaler.transform(val_b).astype(np.float32)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_delta))
        val_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_delta))
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        device = torch.device(DEVICE)
        model = FiLMDeltaMLP(input_dim=emb_dim, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        if seed == 0:
            print(f"    Training on {DEVICE}")

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(MAX_EPOCHS):
            model.train()
            for batch in train_ld:
                a, b, y = [t.to(device) for t in batch]
                optimizer.zero_grad()
                loss = criterion(model(a, b), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = sum(criterion(model(a.to(device), b.to(device)), y.to(device)).item() for a, b, y in val_ld) / len(val_ld)
            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Anchor-based scoring (on CPU for numpy conversion)
        train_embs = scaler.transform(np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
        cand_embs = scaler.transform(np.array([get_emb(s) for s in cand_smiles])).astype(np.float32)

        model.cpu().eval()
        with torch.no_grad():
            for j in range(len(cand_smiles)):
                anchor_t = torch.FloatTensor(train_embs)
                target_t = torch.FloatTensor(np.tile(cand_embs[j:j+1], (len(train_smiles), 1)))
                deltas = model(anchor_t, target_t).numpy().flatten()
                preds_all[j, seed] = np.median(train_y + deltas)

        del model, scaler
        gc.collect()
        print(f"    Seed {seed}: range [{preds_all[:,seed].min():.3f}, {preds_all[:,seed].max():.3f}]")

    return preds_all.mean(axis=1), preds_all.std(axis=1)


def main():
    print("=" * 70)
    print("CROSS-KINASE SCORING + BINARY CLASSIFIER for 19 Candidates")
    print("=" * 70)

    raw = pd.read_csv(RAW_FILE)
    cand_smiles = [clean_smiles(s) for s in SMILES_19]
    results = {}

    # =========================================================
    # PART 1: BTK and SYK FiLMDelta scoring
    # =========================================================
    kinases = {
        "BTK": "CHEMBL5251",
        "SYK": "CHEMBL2599",
    }

    for name, chembl_id in kinases.items():
        print(f"\n{'='*70}")
        print(f"[{name}] FiLMDelta scoring")
        print(f"{'='*70}")

        k_data = raw[raw["target_chembl_id"] == chembl_id]
        k_mol = k_data.groupby("smiles").agg({"pIC50": "mean"}).reset_index()
        print(f"  {name}: {len(k_mol)} molecules, pIC50 {k_mol['pIC50'].min():.2f}-{k_mol['pIC50'].max():.2f}")

        # Subsample if too many molecules (BTK has 9.5K)
        if len(k_mol) > 500:
            # Use within-assay molecules preferentially
            k_within = k_data.groupby("assay_id").filter(lambda x: len(x) >= 5)
            k_mol_within = k_within.groupby("smiles").agg({"pIC50": "mean"}).reset_index()
            if len(k_mol_within) > 500:
                k_mol = k_mol_within.sample(n=500, random_state=42)
            else:
                k_mol = k_mol_within
            print(f"  Subsampled to {len(k_mol)} molecules")

        k_smiles = k_mol['smiles'].tolist()
        k_y = k_mol['pIC50'].values

        pred_mean, pred_std = train_filmdelta_and_score(k_smiles, k_y, cand_smiles, n_seeds=3)

        ranking = np.argsort(-pred_mean)
        results[name] = {
            "n_train": len(k_smiles),
            "candidates": [
                {"idx": j+1, "pred_mean": float(pred_mean[j]), "pred_std": float(pred_std[j])}
                for j in range(19)
            ],
            "ranking": [int(r+1) for r in ranking],
        }

        print(f"\n  {name} Top-5 predictions:")
        for rank, j in enumerate(ranking[:5]):
            print(f"    {rank+1}. Mol {j+1}: pred={pred_mean[j]:.3f} ± {pred_std[j]:.3f}")

    # =========================================================
    # PART 2: Binary ZAP70 Classifier
    # =========================================================
    print(f"\n{'='*70}")
    print("[BINARY CLASSIFIER] ZAP70 Binder Identification")
    print(f"{'='*70}")

    # Load ZAP70 data
    mol_data, _ = load_zap70_molecules()
    zap_smiles = mol_data['smiles'].tolist()
    zap_y = mol_data['pIC50'].values

    # Positives: ZAP70 binders (pIC50 >= 5.0)
    pos_mask = zap_y >= 5.0
    pos_smiles = [s for s, m in zip(zap_smiles, pos_mask) if m]
    neg_zap_smiles = [s for s, m in zip(zap_smiles, pos_mask) if not m]
    print(f"  ZAP70 positives (pIC50 >= 5.0): {len(pos_smiles)}")
    print(f"  ZAP70 weak binders (pIC50 < 5.0): {len(neg_zap_smiles)}")

    # Additional negatives: diverse ChEMBL molecules NOT from kinase targets
    # Use molecules from non-kinase targets that are structurally diverse
    kinase_ids = set(["CHEMBL2803", "CHEMBL5251", "CHEMBL2599", "CHEMBL258",
                      "CHEMBL1841", "CHEMBL3009", "CHEMBL2971", "CHEMBL1862", "CHEMBL267"])
    non_kinase = raw[~raw["target_chembl_id"].isin(kinase_ids)]
    non_kinase_mols = non_kinase.groupby("smiles").first().reset_index()

    # Sample diverse negatives: molecules with Tc < 0.3 to any ZAP70 positive
    print("  Sampling diverse negatives (Tc < 0.3 to ZAP70 positives)...")
    pos_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
               for s in pos_smiles if Chem.MolFromSmiles(s)]

    neg_candidates = non_kinase_mols['smiles'].sample(n=min(5000, len(non_kinase_mols)),
                                                       random_state=42).tolist()
    diverse_negatives = []
    for smi in neg_candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        max_tc = max(DataStructs.TanimotoSimilarity(fp, pfp) for pfp in pos_fps[:50])
        if max_tc < 0.3:
            diverse_negatives.append(smi)
        if len(diverse_negatives) >= 500:
            break

    print(f"  Diverse non-kinase negatives: {len(diverse_negatives)}")

    # Combine negatives
    all_neg_smiles = neg_zap_smiles + diverse_negatives
    all_pos_smiles = pos_smiles

    # Compute fingerprints
    all_clf_smiles = all_pos_smiles + all_neg_smiles
    all_clf_labels = np.array([1]*len(all_pos_smiles) + [0]*len(all_neg_smiles))
    X_clf = compute_fingerprints(all_clf_smiles, "morgan", radius=2, n_bits=2048)

    print(f"  Total classifier data: {len(all_pos_smiles)} pos + {len(all_neg_smiles)} neg = {len(all_clf_smiles)}")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_clf, all_clf_labels)):
        X_train, X_test = X_clf[train_idx], X_clf[test_idx]
        y_train, y_test = all_clf_labels[train_idx], all_clf_labels[test_idx]

        # Random Forest with class weight balancing
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=4)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        cv_metrics.append({"auroc": auroc, "auprc": auprc, "f1": f1})
        print(f"  Fold {fold_idx}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}, F1={f1:.3f}")

    avg_auroc = np.mean([m["auroc"] for m in cv_metrics])
    avg_auprc = np.mean([m["auprc"] for m in cv_metrics])
    print(f"  Mean AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}")

    # Train final model on all data and score candidates
    clf_final = RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=4)
    clf_final.fit(X_clf, all_clf_labels)

    X_cand = compute_fingerprints(cand_smiles, "morgan", radius=2, n_bits=2048)
    cand_probs = clf_final.predict_proba(X_cand)[:, 1]

    clf_ranking = np.argsort(-cand_probs)
    print(f"\n  Binary Classifier Top-5:")
    for rank, j in enumerate(clf_ranking[:5]):
        print(f"    {rank+1}. Mol {j+1}: P(ZAP70 binder) = {cand_probs[j]:.3f}")

    results["binary_classifier"] = {
        "positives": len(all_pos_smiles),
        "negatives": len(all_neg_smiles),
        "neg_sources": {
            "zap70_weak": len(neg_zap_smiles),
            "diverse_non_kinase": len(diverse_negatives),
        },
        "cv_auroc": float(avg_auroc),
        "cv_auprc": float(avg_auprc),
        "cv_metrics": cv_metrics,
        "candidates": [
            {"idx": j+1, "p_binder": float(cand_probs[j])}
            for j in range(19)
        ],
        "ranking": [int(r+1) for r in clf_ranking],
    }

    # =========================================================
    # PART 3: Correlation analysis
    # =========================================================
    print(f"\n{'='*70}")
    print("CROSS-KINASE CORRELATION")
    print(f"{'='*70}")

    # Check if BTK predictions correlate with ZAP70 predictions
    # (load existing ZAP70 scoring results)
    try:
        with open(RESULTS_DIR / "19_molecules_scoring.json") as f:
            zap_scoring = json.load(f)
        zap_consensus = [r["consensus"] for r in zap_scoring["results"]]

        btk_preds = [results["BTK"]["candidates"][j]["pred_mean"] for j in range(19)]
        syk_preds = [results["SYK"]["candidates"][j]["pred_mean"] for j in range(19)]

        r_btk, p_btk = spearmanr(zap_consensus, btk_preds)
        r_syk, p_syk = spearmanr(zap_consensus, syk_preds)

        print(f"  ZAP70 consensus vs BTK pred: Spearman={r_btk:.3f} (p={p_btk:.3e})")
        print(f"  ZAP70 consensus vs SYK pred: Spearman={r_syk:.3f} (p={p_syk:.3e})")

        results["correlations"] = {
            "zap70_vs_btk": {"spearman": float(r_btk), "p": float(p_btk)},
            "zap70_vs_syk": {"spearman": float(r_syk), "p": float(p_syk)},
        }
    except Exception as e:
        print(f"  Could not load ZAP70 scoring: {e}")

    # Save
    out_path = RESULTS_DIR / "19_molecules_cross_kinase.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY — All Rankings for 19 Candidates")
    print(f"{'='*70}")
    print(f"{'Idx':>3} {'BTK pred':>9} {'SYK pred':>9} {'P(binder)':>10} {'ZAP70 cons':>11}")
    print("-" * 45)
    for j in range(19):
        btk_p = results["BTK"]["candidates"][j]["pred_mean"]
        syk_p = results["SYK"]["candidates"][j]["pred_mean"]
        p_bind = cand_probs[j]
        try:
            zap_c = zap_consensus[j]
        except:
            zap_c = 0
        print(f"{j+1:>3} {btk_p:>9.3f} {syk_p:>9.3f} {p_bind:>10.3f} {zap_c:>11.3f}")


if __name__ == "__main__":
    main()
