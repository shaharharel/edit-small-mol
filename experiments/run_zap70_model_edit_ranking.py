"""
ZAP70 (CHEMBL2803) Model-Based Edit Ranking with FiLMDelta.

Phases:
1. Pretrain FiLMDelta on 200K within-assay pairs from the shared dataset
2. Fine-tune on ZAP70 all-pairs (within same assay) with 5-fold molecule-level CV
3. XGB subtraction baseline (same folds)
4. Edit ranking: compare predicted vs actual delta rankings per molecule
5. Find best edits: top-20 predicted pIC50 improvements
"""

import sys
import json
import gc
import time
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

warnings.filterwarnings("ignore")

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP, FiLMDeltaPredictor

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "zap70_model_edit_ranking.json"

DEVICE = "cpu"
SEED = 42
N_FOLDS = 5
PRETRAIN_SAMPLES = 200_000
TARGET_ID = "CHEMBL2803"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------- helpers ----------

def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_embedding_cache():
    """Load chemprop-dmpnn embedding cache."""
    cache_path = PROJECT_ROOT / "data" / "embedding_cache" / "chemprop-dmpnn.npz"
    data = np.load(cache_path, allow_pickle=True)
    smiles_arr = data["smiles"]
    embeddings = data["embeddings"]
    smi2emb = {s: embeddings[i] for i, s in enumerate(smiles_arr)}
    print(f"Loaded embedding cache: {len(smi2emb)} SMILES, dim={embeddings.shape[1]}")
    return smi2emb


def get_embeddings(smiles_list, smi2emb, compute_missing=True):
    """Look up embeddings for a list of SMILES. Computes Morgan FP on-the-fly for missing."""
    embs = []
    found = []
    n_computed = 0
    for s in smiles_list:
        if s in smi2emb:
            embs.append(smi2emb[s])
            found.append(True)
        elif compute_missing:
            fp = compute_morgan_fp(s)
            smi2emb[s] = fp  # cache for reuse
            embs.append(fp)
            found.append(True)
            n_computed += 1
        else:
            embs.append(np.zeros(2048, dtype=np.float32))
            found.append(False)
    if n_computed > 0:
        print(f"  Computed Morgan FP on-the-fly for {n_computed} molecules")
    return np.array(embs, dtype=np.float32), np.array(found)


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return {"mae": mae, "rmse": rmse, "spearman": 0.0, "pearson": 0.0, "r2": 0.0}
    spearman = float(scipy_stats.spearmanr(y_pred, y_true).correlation)
    pearson = float(scipy_stats.pearsonr(y_pred, y_true)[0])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "spearman": spearman, "pearson": pearson, "r2": r2}


# ---------- Phase 1: Pretrain FiLMDelta ----------

def pretrain_filmdelta(smi2emb):
    """Pretrain FiLMDelta on 200K within-assay pairs."""
    print("\n" + "=" * 60)
    print("PHASE 1: Pretrain FiLMDelta on within-assay pairs")
    print("=" * 60)

    t0 = time.time()
    pairs_path = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
    print(f"Loading pairs from {pairs_path.name} ...")
    df = pd.read_csv(pairs_path)
    df_within = df[df["is_within_assay"] == True].copy()
    print(f"Within-assay pairs: {len(df_within):,}")

    # Sample for pretraining
    if len(df_within) > PRETRAIN_SAMPLES:
        df_within = df_within.sample(n=PRETRAIN_SAMPLES, random_state=SEED)
    print(f"Sampled {len(df_within):,} pairs for pretraining")

    # Look up embeddings
    smiles_a = df_within["mol_a"].values
    smiles_b = df_within["mol_b"].values
    deltas = df_within["delta"].values.astype(np.float32)

    emb_a, found_a = get_embeddings(smiles_a, smi2emb)
    emb_b, found_b = get_embeddings(smiles_b, smi2emb)
    mask = found_a & found_b
    print(f"Found embeddings for {mask.sum():,}/{len(mask):,} pairs ({100*mask.mean():.1f}%)")

    emb_a = emb_a[mask]
    emb_b = emb_b[mask]
    deltas = deltas[mask]

    # Train/val split (90/10)
    n = len(deltas)
    idx = np.random.permutation(n)
    n_val = n // 10
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}")

    predictor = FiLMDeltaPredictor(
        hidden_dims=None,  # auto
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=512,
        max_epochs=30,
        patience=10,
        device=DEVICE,
    )

    history = predictor.fit(
        emb_a[train_idx], emb_b[train_idx], deltas[train_idx],
        emb_a[val_idx], emb_b[val_idx], deltas[val_idx],
        verbose=True,
    )

    val_metrics, _, _ = predictor.evaluate(emb_a[val_idx], emb_b[val_idx], deltas[val_idx])
    print(f"Pretrain val metrics: MAE={val_metrics['mae']:.4f}, "
          f"Spearman={val_metrics['spearman']:.4f}, R2={val_metrics['r2']:.4f}")
    print(f"Phase 1 took {time.time()-t0:.1f}s")

    # Clean up large arrays
    del emb_a, emb_b, deltas, df, df_within
    gc.collect()

    return predictor, val_metrics


# ---------- Phase 2 & 3: ZAP70 fine-tuning + XGB baseline ----------

def load_zap70_data(smi2emb):
    """Load ZAP70 molecules and generate within-assay pairs."""
    print("\nLoading ZAP70 data ...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv")
    zap = df[df["target_chembl_id"] == TARGET_ID].copy()

    # Aggregate: mean pIC50 per molecule (across assays)
    mol_data = zap.groupby("molecule_chembl_id").agg(
        smiles=("smiles", "first"),
        pIC50=("pIC50", "mean"),
        n_assays=("assay_chembl_id", "nunique"),
    ).reset_index()

    # Pre-compute embeddings for any molecules not in cache
    n_in_cache = sum(1 for s in mol_data["smiles"] if s in smi2emb)
    n_missing = len(mol_data) - n_in_cache
    if n_missing > 0:
        print(f"Computing Morgan FP for {n_missing} ZAP70 molecules not in cache ...")
        for s in mol_data["smiles"]:
            if s not in smi2emb:
                smi2emb[s] = compute_morgan_fp(s)

    # Drop molecules where RDKit fails to parse SMILES
    mol_data["valid"] = mol_data["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    n_invalid = (~mol_data["valid"]).sum()
    if n_invalid > 0:
        print(f"Dropping {n_invalid} molecules with invalid SMILES")
        mol_data = mol_data[mol_data["valid"]].reset_index(drop=True)

    print(f"ZAP70: {len(mol_data)} molecules with embeddings")

    # Generate all-pairs within this target
    # Use molecule-level pIC50 (mean across assays) for delta computation
    mols = mol_data["molecule_chembl_id"].values
    smiles = mol_data["smiles"].values
    pic50 = mol_data["pIC50"].values

    mol2idx = {m: i for i, m in enumerate(mols)}
    mol2smi = {m: s for m, s in zip(mols, smiles)}
    mol2pic50 = {m: p for m, p in zip(mols, pic50)}

    pairs = []
    for i, j in combinations(range(len(mols)), 2):
        # Both directions
        pairs.append((mols[i], mols[j], pic50[j] - pic50[i]))
        pairs.append((mols[j], mols[i], pic50[i] - pic50[j]))

    pairs_df = pd.DataFrame(pairs, columns=["mol_a_id", "mol_b_id", "delta"])
    pairs_df["smiles_a"] = pairs_df["mol_a_id"].map(mol2smi)
    pairs_df["smiles_b"] = pairs_df["mol_b_id"].map(mol2smi)

    print(f"Generated {len(pairs_df):,} directed pairs")
    return mol_data, pairs_df, mol2smi, mol2pic50


def finetune_and_evaluate(pretrained_predictor, pairs_df, mol_data, smi2emb):
    """5-fold CV: fine-tune FiLMDelta on ZAP70 pairs."""
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tune FiLMDelta on ZAP70 (5-fold CV)")
    print("=" * 60)

    t0 = time.time()
    molecules = mol_data["molecule_chembl_id"].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_preds = []
    all_true = []
    all_pairs_info = []
    fold_metrics = []

    for fold_i, (train_mol_idx, test_mol_idx) in enumerate(kf.split(molecules)):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")
        train_mols = set(molecules[train_mol_idx])
        test_mols = set(molecules[test_mol_idx])

        # Train pairs: both molecules in train set
        train_mask = pairs_df["mol_a_id"].isin(train_mols) & pairs_df["mol_b_id"].isin(train_mols)
        test_mask = pairs_df["mol_a_id"].isin(test_mols) & pairs_df["mol_b_id"].isin(test_mols)

        train_pairs = pairs_df[train_mask]
        test_pairs = pairs_df[test_mask]

        print(f"  Train mols: {len(train_mols)}, Test mols: {len(test_mols)}")
        print(f"  Train pairs: {len(train_pairs):,}, Test pairs: {len(test_pairs):,}")

        if len(test_pairs) == 0:
            print("  No test pairs, skipping fold")
            continue

        # Get embeddings
        emb_a_train, _ = get_embeddings(train_pairs["smiles_a"].values, smi2emb)
        emb_b_train, _ = get_embeddings(train_pairs["smiles_b"].values, smi2emb)
        delta_train = train_pairs["delta"].values.astype(np.float32)

        emb_a_test, _ = get_embeddings(test_pairs["smiles_a"].values, smi2emb)
        emb_b_test, _ = get_embeddings(test_pairs["smiles_b"].values, smi2emb)
        delta_test = test_pairs["delta"].values.astype(np.float32)

        # Create new predictor initialized from pretrained weights
        predictor = FiLMDeltaPredictor(
            hidden_dims=pretrained_predictor.hidden_dims,
            dropout=0.2,
            learning_rate=1e-4,  # lower LR for fine-tuning
            batch_size=512,
            max_epochs=50,
            patience=15,
            device=DEVICE,
        )

        # Initialize model with pretrained weights
        predictor.input_dim = pretrained_predictor.input_dim
        predictor.model = FiLMDeltaMLP(
            input_dim=pretrained_predictor.input_dim,
            hidden_dims=pretrained_predictor.hidden_dims,
            dropout=0.2,
        )
        predictor.model.load_state_dict(pretrained_predictor.model.state_dict())
        predictor.model = predictor.model.to(DEVICE)

        # Fine-tune with early stopping using a val split from train
        n_train = len(delta_train)
        perm = np.random.permutation(n_train)
        n_ft_val = max(n_train // 10, 1)
        ft_val_idx = perm[:n_ft_val]
        ft_train_idx = perm[n_ft_val:]

        # Manual training loop for fine-tuning (since .fit() recreates the model)
        optimizer = torch.optim.Adam(predictor.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        criterion = torch.nn.MSELoss()

        # Tensors
        ea_tr = torch.from_numpy(emb_a_train[ft_train_idx]).float()
        eb_tr = torch.from_numpy(emb_b_train[ft_train_idx]).float()
        dt_tr = torch.from_numpy(delta_train[ft_train_idx]).float()

        ea_v = torch.from_numpy(emb_a_train[ft_val_idx]).float()
        eb_v = torch.from_numpy(emb_b_train[ft_val_idx]).float()
        dt_v = torch.from_numpy(delta_train[ft_val_idx]).float()

        train_ds = torch.utils.data.TensorDataset(ea_tr, eb_tr, dt_tr)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(50):
            predictor.model.train()
            for ba, bb, by in train_loader:
                optimizer.zero_grad()
                pred = predictor.model(ba, bb)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            # Validation
            predictor.model.eval()
            with torch.no_grad():
                val_pred = predictor.model(ea_v, eb_v)
                val_loss = criterion(val_pred, dt_v).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in predictor.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 15:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            predictor.model.load_state_dict(best_state)
            predictor.model = predictor.model.to(DEVICE)

        # Evaluate on test pairs
        preds = predictor.predict(emb_a_test, emb_b_test)
        metrics = compute_metrics(delta_test, preds)
        fold_metrics.append(metrics)
        print(f"  Test: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}, R2={metrics['r2']:.4f}")

        all_preds.extend(preds.tolist())
        all_true.extend(delta_test.tolist())
        for _, row in test_pairs.iterrows():
            all_pairs_info.append({
                "mol_a_id": row["mol_a_id"],
                "mol_b_id": row["mol_b_id"],
                "fold": fold_i,
            })

        del predictor, emb_a_train, emb_b_train, emb_a_test, emb_b_test
        gc.collect()

    # Aggregate metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    overall_metrics = compute_metrics(all_true, all_preds)
    avg_fold = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}

    print(f"\nFiLMDelta overall: MAE={overall_metrics['mae']:.4f}, "
          f"Spearman={overall_metrics['spearman']:.4f}, R2={overall_metrics['r2']:.4f}")
    print(f"FiLMDelta fold-avg: MAE={avg_fold['mae']:.4f}, Spearman={avg_fold['spearman']:.4f}")
    print(f"Phase 2 took {time.time()-t0:.1f}s")

    return {
        "overall": overall_metrics,
        "fold_avg": avg_fold,
        "fold_metrics": fold_metrics,
        "preds": all_preds,
        "true": all_true,
        "pairs_info": all_pairs_info,
    }


def xgb_baseline(pairs_df, mol_data, smi2emb):
    """Phase 3: XGB subtraction baseline with same 5-fold CV."""
    print("\n" + "=" * 60)
    print("PHASE 3: XGB Subtraction Baseline (5-fold CV)")
    print("=" * 60)

    from xgboost import XGBRegressor

    t0 = time.time()
    molecules = mol_data["molecule_chembl_id"].values
    mol2smi = dict(zip(mol_data["molecule_chembl_id"], mol_data["smiles"]))
    mol2pic50 = dict(zip(mol_data["molecule_chembl_id"], mol_data["pIC50"]))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_preds = []
    all_true = []
    fold_metrics = []

    for fold_i, (train_mol_idx, test_mol_idx) in enumerate(kf.split(molecules)):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")
        train_mols = molecules[train_mol_idx]
        test_mols = set(molecules[test_mol_idx])

        # Train XGB on absolute pIC50
        train_smiles = [mol2smi[m] for m in train_mols]
        train_pic50 = np.array([mol2pic50[m] for m in train_mols], dtype=np.float32)
        train_embs, _ = get_embeddings(train_smiles, smi2emb)

        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
        xgb.fit(train_embs, train_pic50)

        # Test: predict delta as f(B) - f(A)
        test_mask = pairs_df["mol_a_id"].isin(test_mols) & pairs_df["mol_b_id"].isin(test_mols)
        test_pairs = pairs_df[test_mask]

        if len(test_pairs) == 0:
            continue

        emb_a_test, _ = get_embeddings(test_pairs["smiles_a"].values, smi2emb)
        emb_b_test, _ = get_embeddings(test_pairs["smiles_b"].values, smi2emb)
        delta_test = test_pairs["delta"].values.astype(np.float32)

        pred_a = xgb.predict(emb_a_test)
        pred_b = xgb.predict(emb_b_test)
        preds = pred_b - pred_a

        metrics = compute_metrics(delta_test, preds)
        fold_metrics.append(metrics)
        print(f"  Test: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}, R2={metrics['r2']:.4f}")

        all_preds.extend(preds.tolist())
        all_true.extend(delta_test.tolist())

        del xgb
        gc.collect()

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    overall_metrics = compute_metrics(all_true, all_preds)
    avg_fold = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}

    print(f"\nXGB overall: MAE={overall_metrics['mae']:.4f}, "
          f"Spearman={overall_metrics['spearman']:.4f}, R2={overall_metrics['r2']:.4f}")
    print(f"XGB fold-avg: MAE={avg_fold['mae']:.4f}, Spearman={avg_fold['spearman']:.4f}")
    print(f"Phase 3 took {time.time()-t0:.1f}s")

    return {
        "overall": overall_metrics,
        "fold_avg": avg_fold,
        "fold_metrics": fold_metrics,
        "preds": all_preds,
        "true": all_true,
    }


# ---------- Phase 4: Edit ranking ----------

def edit_ranking_analysis(film_results, xgb_results, pairs_df, mol_data):
    """Compare per-molecule edit ranking between FiLMDelta and XGB."""
    print("\n" + "=" * 60)
    print("PHASE 4: Edit Ranking Analysis")
    print("=" * 60)

    # We use overall pooled predictions (all folds combined)
    # Build a lookup: (mol_a_id, mol_b_id) -> predicted delta
    film_preds_map = {}
    for i, info in enumerate(film_results["pairs_info"]):
        key = (info["mol_a_id"], info["mol_b_id"])
        film_preds_map[key] = film_results["preds"][i]

    # For XGB we don't have pairs_info stored separately, so retrain a full model
    # Actually, let's just use the pooled predictions which align with the same test pairs
    # Both use the same KFold splits, so the test pairs are identical per fold.
    # Build XGB map from the same ordering
    xgb_preds_map = {}
    # Reconstruct from pairs_info (same pairs in same order)
    for i, info in enumerate(film_results["pairs_info"]):
        key = (info["mol_a_id"], info["mol_b_id"])
        if i < len(xgb_results["preds"]):
            xgb_preds_map[key] = xgb_results["preds"][i]

    # True deltas map
    true_map = {}
    for i, info in enumerate(film_results["pairs_info"]):
        key = (info["mol_a_id"], info["mol_b_id"])
        true_map[key] = film_results["true"][i]

    # Per-molecule ranking analysis
    molecules = mol_data["molecule_chembl_id"].values
    mol2pic50 = dict(zip(mol_data["molecule_chembl_id"], mol_data["pIC50"]))

    film_spearman_per_mol = []
    xgb_spearman_per_mol = []
    film_top10_acc = []
    xgb_top10_acc = []

    for mol_a in molecules:
        # Get all pairs where this molecule is mol_a (outgoing edits)
        true_deltas = []
        film_deltas = []
        xgb_deltas = []
        mol_b_ids = []

        for mol_b in molecules:
            if mol_b == mol_a:
                continue
            key = (mol_a, mol_b)
            if key in true_map and key in film_preds_map and key in xgb_preds_map:
                true_deltas.append(true_map[key])
                film_deltas.append(film_preds_map[key])
                xgb_deltas.append(xgb_preds_map[key])
                mol_b_ids.append(mol_b)

        if len(true_deltas) < 5:
            continue

        true_deltas = np.array(true_deltas)
        film_deltas = np.array(film_deltas)
        xgb_deltas = np.array(xgb_deltas)

        # Spearman correlation
        if np.std(true_deltas) > 1e-8:
            film_sp = scipy_stats.spearmanr(film_deltas, true_deltas).correlation
            xgb_sp = scipy_stats.spearmanr(xgb_deltas, true_deltas).correlation
            if not np.isnan(film_sp):
                film_spearman_per_mol.append(film_sp)
            if not np.isnan(xgb_sp):
                xgb_spearman_per_mol.append(xgb_sp)

        # Top-10 accuracy: what fraction of predicted top-10 are in actual top-10?
        k = min(10, len(true_deltas) // 2)
        if k >= 2:
            true_top_k = set(np.argsort(true_deltas)[-k:])  # highest deltas = best edits
            film_top_k = set(np.argsort(film_deltas)[-k:])
            xgb_top_k = set(np.argsort(xgb_deltas)[-k:])
            film_top10_acc.append(len(true_top_k & film_top_k) / k)
            xgb_top10_acc.append(len(true_top_k & xgb_top_k) / k)

    results = {
        "n_molecules_ranked": len(film_spearman_per_mol),
        "film_mean_spearman": float(np.mean(film_spearman_per_mol)) if film_spearman_per_mol else 0.0,
        "film_median_spearman": float(np.median(film_spearman_per_mol)) if film_spearman_per_mol else 0.0,
        "xgb_mean_spearman": float(np.mean(xgb_spearman_per_mol)) if xgb_spearman_per_mol else 0.0,
        "xgb_median_spearman": float(np.median(xgb_spearman_per_mol)) if xgb_spearman_per_mol else 0.0,
        "film_mean_top10_acc": float(np.mean(film_top10_acc)) if film_top10_acc else 0.0,
        "xgb_mean_top10_acc": float(np.mean(xgb_top10_acc)) if xgb_top10_acc else 0.0,
        "n_molecules_top10": len(film_top10_acc),
    }

    print(f"\nPer-molecule ranking ({results['n_molecules_ranked']} molecules):")
    print(f"  FiLMDelta: mean Spearman={results['film_mean_spearman']:.4f}, "
          f"median={results['film_median_spearman']:.4f}")
    print(f"  XGB:       mean Spearman={results['xgb_mean_spearman']:.4f}, "
          f"median={results['xgb_median_spearman']:.4f}")
    print(f"\nTop-{min(10, 5)} accuracy ({results['n_molecules_top10']} molecules):")
    print(f"  FiLMDelta: {results['film_mean_top10_acc']:.4f}")
    print(f"  XGB:       {results['xgb_mean_top10_acc']:.4f}")

    return results


# ---------- Phase 5: Best edits ----------

def find_best_edits(film_results, mol_data, mol2pic50, smi2emb):
    """Find the top-20 predicted best edits (largest pIC50 increase)."""
    print("\n" + "=" * 60)
    print("PHASE 5: Best Predicted Edits")
    print("=" * 60)

    mol2smi = dict(zip(mol_data["molecule_chembl_id"], mol_data["smiles"]))

    # Build predictions map
    film_preds_map = {}
    true_map = {}
    for i, info in enumerate(film_results["pairs_info"]):
        key = (info["mol_a_id"], info["mol_b_id"])
        film_preds_map[key] = film_results["preds"][i]
        true_map[key] = film_results["true"][i]

    # Collect all predicted edits
    edits = []
    for (mol_a, mol_b), pred_delta in film_preds_map.items():
        actual_delta = true_map.get((mol_a, mol_b), None)
        edits.append({
            "mol_a_id": mol_a,
            "mol_b_id": mol_b,
            "smiles_a": mol2smi.get(mol_a, "?"),
            "smiles_b": mol2smi.get(mol_b, "?"),
            "pIC50_a": float(mol2pic50.get(mol_a, 0)),
            "pIC50_b": float(mol2pic50.get(mol_b, 0)),
            "pred_delta": float(pred_delta),
            "actual_delta": float(actual_delta) if actual_delta is not None else None,
        })

    # Sort by predicted delta (largest improvement first)
    edits.sort(key=lambda x: x["pred_delta"], reverse=True)
    top_20 = edits[:20]

    print(f"\nTop 20 predicted edits (largest pIC50 increase):")
    print(f"{'Rank':>4} | {'mol_a':>14} -> {'mol_b':>14} | {'pIC50_a':>7} -> {'pIC50_b':>7} | "
          f"{'pred_Δ':>7} | {'actual_Δ':>8} | {'error':>6}")
    print("-" * 100)
    for rank, e in enumerate(top_20, 1):
        actual = e["actual_delta"]
        err = abs(e["pred_delta"] - actual) if actual is not None else float("nan")
        print(f"{rank:4d} | {e['mol_a_id']:>14} -> {e['mol_b_id']:>14} | "
              f"{e['pIC50_a']:7.2f} -> {e['pIC50_b']:7.2f} | "
              f"{e['pred_delta']:+7.3f} | {actual:+8.3f} | {err:6.3f}")

    # Summary stats for top-20
    pred_deltas = [e["pred_delta"] for e in top_20]
    actual_deltas = [e["actual_delta"] for e in top_20 if e["actual_delta"] is not None]
    errors = [abs(e["pred_delta"] - e["actual_delta"]) for e in top_20 if e["actual_delta"] is not None]

    summary = {
        "top_20_edits": top_20,
        "top_20_mean_pred_delta": float(np.mean(pred_deltas)),
        "top_20_mean_actual_delta": float(np.mean(actual_deltas)) if actual_deltas else None,
        "top_20_mean_error": float(np.mean(errors)) if errors else None,
        "top_20_actual_in_top20_pct": None,
    }

    # What fraction of predicted top-20 are actually in the true top-20?
    all_actual = [(k, true_map[k]) for k in true_map]
    all_actual.sort(key=lambda x: x[1], reverse=True)
    true_top20_keys = set(k for k, _ in all_actual[:20])
    pred_top20_keys = set((e["mol_a_id"], e["mol_b_id"]) for e in top_20)
    overlap = len(true_top20_keys & pred_top20_keys)
    summary["top_20_actual_in_top20_pct"] = float(overlap / 20)

    print(f"\nTop-20 overlap with actual top-20: {overlap}/20 ({100*overlap/20:.0f}%)")
    print(f"Mean predicted delta: {summary['top_20_mean_pred_delta']:.3f}")
    if summary["top_20_mean_actual_delta"] is not None:
        print(f"Mean actual delta:    {summary['top_20_mean_actual_delta']:.3f}")
        print(f"Mean absolute error:  {summary['top_20_mean_error']:.3f}")

    return summary


# ---------- Main ----------

def main():
    print("=" * 60)
    print("ZAP70 (CHEMBL2803) Model-Based Edit Ranking")
    print("=" * 60)
    t_start = time.time()

    # Load embeddings
    smi2emb = load_embedding_cache()

    # Phase 1: Pretrain
    pretrained, pretrain_metrics = pretrain_filmdelta(smi2emb)

    # Load ZAP70 data
    mol_data, pairs_df, mol2smi, mol2pic50 = load_zap70_data(smi2emb)

    # Phase 2: Fine-tune FiLMDelta
    film_results = finetune_and_evaluate(pretrained, pairs_df, mol_data, smi2emb)

    # Phase 3: XGB baseline
    xgb_results = xgb_baseline(pairs_df, mol_data, smi2emb)

    # Phase 4: Edit ranking comparison
    ranking_results = edit_ranking_analysis(film_results, xgb_results, pairs_df, mol_data)

    # Phase 5: Best edits
    best_edits = find_best_edits(film_results, mol_data, mol2pic50, smi2emb)

    # Compile and save results
    results = {
        "target": "CHEMBL2803 (ZAP70)",
        "n_molecules": len(mol_data),
        "n_pairs": len(pairs_df),
        "n_folds": N_FOLDS,
        "pretrain_samples": PRETRAIN_SAMPLES,
        "pretrain_metrics": pretrain_metrics,
        "filmdelta": {
            "overall": film_results["overall"],
            "fold_avg": film_results["fold_avg"],
            "fold_metrics": film_results["fold_metrics"],
        },
        "xgb_subtraction": {
            "overall": xgb_results["overall"],
            "fold_avg": xgb_results["fold_avg"],
            "fold_metrics": xgb_results["fold_metrics"],
        },
        "edit_ranking": ranking_results,
        "best_edits": best_edits,
        "runtime_seconds": time.time() - t_start,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"FiLMDelta:       MAE={film_results['overall']['mae']:.4f}, "
          f"Spearman={film_results['overall']['spearman']:.4f}, "
          f"R2={film_results['overall']['r2']:.4f}")
    print(f"XGB Subtraction: MAE={xgb_results['overall']['mae']:.4f}, "
          f"Spearman={xgb_results['overall']['spearman']:.4f}, "
          f"R2={xgb_results['overall']['r2']:.4f}")
    print(f"\nEdit Ranking (per-molecule Spearman):")
    print(f"  FiLMDelta: {ranking_results['film_mean_spearman']:.4f}")
    print(f"  XGB:       {ranking_results['xgb_mean_spearman']:.4f}")
    print(f"\nTop-10 Accuracy:")
    print(f"  FiLMDelta: {ranking_results['film_mean_top10_acc']:.4f}")
    print(f"  XGB:       {ranking_results['xgb_mean_top10_acc']:.4f}")
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
