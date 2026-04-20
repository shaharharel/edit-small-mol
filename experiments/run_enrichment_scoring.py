#!/usr/bin/env python3
"""
Score 19 candidates with the best enrichment approach (antisymmetric regularization).
Also combines with curriculum pretraining.

Usage:
    conda run -n quris python -u experiments/run_enrichment_scoring.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP, FiLMDeltaPredictor

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "enrichment_pretraining"
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
CANDIDATES_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "19_molecules_scoring.json"

ZAP70_ID = "CHEMBL2803"
EMB_DIM = 2048
HIDDEN_DIMS = [1024, 512, 256]
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
N_SEEDS = 10  # More seeds for final scoring


def compute_morgan_fp(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_fp_obj(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def generate_all_pairs(smiles, pIC50):
    n = len(smiles)
    mol_a, mol_b, deltas = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                mol_a.append(smiles[i])
                mol_b.append(smiles[j])
                deltas.append(float(pIC50[j] - pIC50[i]))
    return pd.DataFrame({'mol_a': mol_a, 'mol_b': mol_b, 'delta': deltas})


def load_zap70_all():
    """Load ALL 280 ZAP70 molecules (no split)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    return mol_data


def load_candidates():
    with open(CANDIDATES_FILE) as f:
        data = json.load(f)
    return [(mol['smiles_clean'], mol['idx'], mol['max_tanimoto']) for mol in data['results']]


def train_and_predict_antisym(
    train_smiles, train_y, test_smiles, emb_dict, seed,
    antisym_reg=0.1, pretrained_state=None, lr=1e-3
):
    """Train FiLMDelta with antisymmetric regularization and predict via anchors."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    pairs = generate_all_pairs(train_smiles, train_y)
    n_val = max(int(len(pairs) * 0.1), 100)
    val_pairs = pairs.sample(n=n_val, random_state=seed)
    trn_pairs = pairs.drop(val_pairs.index)

    def get_emb(smi):
        return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

    train_a = np.array([get_emb(s) for s in trn_pairs['mol_a']])
    train_b = np.array([get_emb(s) for s in trn_pairs['mol_b']])
    train_yp = trn_pairs['delta'].values.astype(np.float32)
    val_a = np.array([get_emb(s) for s in val_pairs['mol_a']])
    val_b = np.array([get_emb(s) for s in val_pairs['mol_b']])
    val_yp = val_pairs['delta'].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
    train_a = scaler.transform(train_a).astype(np.float32)
    train_b = scaler.transform(train_b).astype(np.float32)
    val_a = scaler.transform(val_a).astype(np.float32)
    val_b = scaler.transform(val_b).astype(np.float32)

    predictor = FiLMDeltaPredictor(
        hidden_dims=HIDDEN_DIMS, dropout=0.2,
        learning_rate=lr, batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=DEVICE,
    )
    predictor.fit(
        train_a, train_b, train_yp,
        val_a, val_b, val_yp,
        verbose=False,
        antisymmetric_aug=False,
        antisym_reg_weight=antisym_reg,
    )

    # Load pretrained and override (if provided and needed) - not used here
    # The pretrained_state approach was tested separately

    # Anchor prediction
    n_train = len(train_smiles)
    train_embs = scaler.transform(
        np.array([get_emb(s) for s in train_smiles])).astype(np.float32)
    test_embs = scaler.transform(
        np.array([get_emb(s) for s in test_smiles])).astype(np.float32)

    preds = np.zeros(len(test_smiles))
    for j in range(len(test_smiles)):
        pred_deltas = predictor.predict(
            train_embs,
            np.tile(test_embs[j:j+1], (n_train, 1))
        )
        preds[j] = np.median(train_y + pred_deltas)

    del predictor
    gc.collect()
    return preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("SCORING 19 CANDIDATES WITH BEST ENRICHMENT APPROACH")
    print("=" * 70)

    # Load data
    mol_data = load_zap70_all()
    train_smiles = mol_data['smiles'].tolist()
    train_y = mol_data['pIC50'].values
    print(f"ZAP70 training: {len(train_smiles)} molecules")

    candidates = load_candidates()
    cand_smiles = [c[0] for c in candidates]
    cand_idx = [c[1] for c in candidates]
    cand_tc = [c[2] for c in candidates]
    print(f"Candidates: {len(cand_smiles)}")

    # Build embeddings
    all_smi = list(set(train_smiles + cand_smiles))
    emb_dict = {}
    for smi in all_smi:
        emb_dict[smi] = compute_morgan_fp(smi)

    # ============================================================
    # Score with baseline (standard FiLMDelta, no antisym)
    # ============================================================
    print(f"\nScoring with BASELINE (standard FiLMDelta, {N_SEEDS} seeds)...")
    baseline_preds = np.zeros((N_SEEDS, len(cand_smiles)))
    for s in range(N_SEEDS):
        seed = s * 17 + 5
        t1 = time.time()
        preds = train_and_predict_antisym(
            train_smiles, train_y, cand_smiles, emb_dict, seed,
            antisym_reg=0.0  # No regularization = baseline
        )
        baseline_preds[s] = preds
        elapsed = time.time() - t1
        print(f"  Seed {s+1}/{N_SEEDS}: {elapsed:.0f}s")

    baseline_mean = baseline_preds.mean(axis=0)
    baseline_std = baseline_preds.std(axis=0)

    # ============================================================
    # Score with antisymmetric regularization (best approach)
    # ============================================================
    print(f"\nScoring with ANTISYM REG (antisym_reg=0.1, {N_SEEDS} seeds)...")
    antisym_preds = np.zeros((N_SEEDS, len(cand_smiles)))
    for s in range(N_SEEDS):
        seed = s * 17 + 5
        t1 = time.time()
        preds = train_and_predict_antisym(
            train_smiles, train_y, cand_smiles, emb_dict, seed,
            antisym_reg=0.1
        )
        antisym_preds[s] = preds
        elapsed = time.time() - t1
        print(f"  Seed {s+1}/{N_SEEDS}: {elapsed:.0f}s")

    antisym_mean = antisym_preds.mean(axis=0)
    antisym_std = antisym_preds.std(axis=0)

    # ============================================================
    # Score with curriculum + antisym combined
    # ============================================================
    # Load pretrained diverse state
    diverse_state_path = RESULTS_DIR / "diverse_pretrained_state.pt"
    has_diverse = diverse_state_path.exists()

    if has_diverse:
        print(f"\nScoring with CURRICULUM + ANTISYM (diverse pretrain + antisym_reg=0.1, {N_SEEDS} seeds)...")
        diverse_state = torch.load(diverse_state_path, map_location='cpu')

        curriculum_preds = np.zeros((N_SEEDS, len(cand_smiles)))
        for s in range(N_SEEDS):
            seed = s * 17 + 5
            t1 = time.time()

            # Train with curriculum pretrained init + antisym reg
            np.random.seed(seed)
            torch.manual_seed(seed)

            pairs = generate_all_pairs(train_smiles, train_y)
            n_val = max(int(len(pairs) * 0.1), 100)
            val_pairs = pairs.sample(n=n_val, random_state=seed)
            trn_pairs = pairs.drop(val_pairs.index)

            def get_emb(smi):
                return emb_dict.get(smi, np.zeros(EMB_DIM, dtype=np.float32))

            train_a = np.array([get_emb(s_) for s_ in trn_pairs['mol_a']])
            train_b = np.array([get_emb(s_) for s_ in trn_pairs['mol_b']])
            train_yp = trn_pairs['delta'].values.astype(np.float32)
            val_a = np.array([get_emb(s_) for s_ in val_pairs['mol_a']])
            val_b = np.array([get_emb(s_) for s_ in val_pairs['mol_b']])
            val_yp = val_pairs['delta'].values.astype(np.float32)

            scaler = StandardScaler()
            scaler.fit(np.vstack([train_a, train_b, val_a, val_b]))
            train_a = scaler.transform(train_a).astype(np.float32)
            train_b = scaler.transform(train_b).astype(np.float32)
            val_a = scaler.transform(val_a).astype(np.float32)
            val_b = scaler.transform(val_b).astype(np.float32)

            # Create model with pretrained weights
            device = torch.device(DEVICE)
            model = FiLMDeltaMLP(input_dim=EMB_DIM, hidden_dims=HIDDEN_DIMS, dropout=0.2).to(device)
            try:
                model.load_state_dict(diverse_state, strict=True)
            except RuntimeError:
                own = model.state_dict()
                for name, param in diverse_state.items():
                    if name in own and own[name].shape == param.shape:
                        own[name].copy_(param)
                model.load_state_dict(own)
            model = model.to(device)

            # Train with antisym reg
            train_ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(train_a), torch.FloatTensor(train_b), torch.FloatTensor(train_yp))
            val_ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(val_a), torch.FloatTensor(val_b), torch.FloatTensor(val_yp))
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            criterion = nn.MSELoss()
            best_vl, pat, best_st = float('inf'), 0, None

            for epoch in range(MAX_EPOCHS):
                model.train()
                for batch in train_loader:
                    a, b, y = [t.to(device) for t in batch]
                    optimizer.zero_grad()
                    pred = model(a, b)
                    loss = criterion(pred, y)
                    # Antisym reg
                    pred_rev = model(b, a)
                    sym_loss = torch.mean((pred + pred_rev) ** 2)
                    loss = loss + 0.1 * sym_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                vl_sum, nv = 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        a, b, y = [t.to(device) for t in batch]
                        vl_sum += criterion(model(a, b), y).item()
                        nv += 1
                vl = vl_sum / max(nv, 1)
                if vl < best_vl:
                    best_vl = vl; pat = 0
                    best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    pat += 1
                    if pat >= PATIENCE:
                        break

            if best_st:
                model.load_state_dict(best_st)
            model.cpu().eval()

            # Anchor prediction
            n_train = len(train_smiles)
            tr_emb = scaler.transform(
                np.array([get_emb(s_) for s_ in train_smiles])).astype(np.float32)
            te_emb = scaler.transform(
                np.array([get_emb(s_) for s_ in cand_smiles])).astype(np.float32)

            preds = np.zeros(len(cand_smiles))
            with torch.no_grad():
                for j in range(len(cand_smiles)):
                    a = torch.FloatTensor(tr_emb)
                    b = torch.FloatTensor(np.tile(te_emb[j:j+1], (n_train, 1)))
                    deltas = model(a, b).numpy().flatten()
                    preds[j] = np.median(train_y + deltas)

            curriculum_preds[s] = preds
            elapsed = time.time() - t1
            print(f"  Seed {s+1}/{N_SEEDS}: {elapsed:.0f}s")

            del model
            gc.collect()

        curriculum_mean = curriculum_preds.mean(axis=0)
        curriculum_std = curriculum_preds.std(axis=0)
    else:
        curriculum_mean = None

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 70)
    print("CANDIDATE PREDICTIONS")
    print("=" * 70)

    header = f"{'Mol':>4s} {'MaxTc':>6s} {'Baseline':>9s} {'Std':>5s} {'AntisymReg':>11s} {'Std':>5s}"
    if curriculum_mean is not None:
        header += f" {'Curric+AS':>10s} {'Std':>5s}"
    print(header)
    print("-" * len(header))

    for i in range(len(cand_smiles)):
        line = (f"  {cand_idx[i]:2d}  {cand_tc[i]:6.3f}  "
                f"{baseline_mean[i]:8.3f} {baseline_std[i]:5.3f}  "
                f"{antisym_mean[i]:10.3f} {antisym_std[i]:5.3f}")
        if curriculum_mean is not None:
            line += f"  {curriculum_mean[i]:9.3f} {curriculum_std[i]:5.3f}"
        print(line)

    # Save results
    results = {
        'metadata': {
            'n_seeds': N_SEEDS,
            'n_train': len(train_smiles),
            'n_candidates': len(cand_smiles),
            'device': DEVICE,
            'antisym_reg_weight': 0.1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        },
        'candidates': [],
    }

    for i in range(len(cand_smiles)):
        entry = {
            'idx': cand_idx[i],
            'smiles': cand_smiles[i],
            'max_tanimoto': cand_tc[i],
            'baseline_pred': float(baseline_mean[i]),
            'baseline_std': float(baseline_std[i]),
            'antisym_pred': float(antisym_mean[i]),
            'antisym_std': float(antisym_std[i]),
        }
        if curriculum_mean is not None:
            entry['curriculum_antisym_pred'] = float(curriculum_mean[i])
            entry['curriculum_antisym_std'] = float(curriculum_std[i])
        results['candidates'].append(entry)

    # Rankings
    baseline_ranking = np.argsort(-baseline_mean)
    antisym_ranking = np.argsort(-antisym_mean)
    results['baseline_ranking'] = [int(cand_idx[i]) for i in baseline_ranking]
    results['antisym_ranking'] = [int(cand_idx[i]) for i in antisym_ranking]

    if curriculum_mean is not None:
        curric_ranking = np.argsort(-curriculum_mean)
        results['curriculum_ranking'] = [int(cand_idx[i]) for i in curric_ranking]

    out_path = RESULTS_DIR / "candidate_scoring_final.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Rankings comparison
    print(f"\nBaseline ranking:   {results['baseline_ranking'][:5]} ...")
    print(f"AntisymReg ranking: {results['antisym_ranking'][:5]} ...")
    if curriculum_mean is not None:
        print(f"Curric+AS ranking:  {results['curriculum_ranking'][:5]} ...")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
