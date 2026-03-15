#!/usr/bin/env python3
"""
Run CheMeleon pretrained D-MPNN through Phase 1 evaluation.

CheMeleon is a foundation model for molecular property prediction,
pretrained on diverse ChEMBL data using ChemProp v2 D-MPNN architecture.
Checkpoint: https://zenodo.org/records/15460715

Usage:
    python -u experiments/run_chemeleon_phase1.py
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from run_paper_evaluation import (
    load_data, split_data, run_single_experiment,
    SEEDS, RESULTS_DIR, CACHE_DIR,
)

CHEMELEON_CKPT = PROJECT_ROOT / "data" / "pretrained_models" / "chemeleon_mp.pt"


def aggregate_seeds(seed_runs):
    """Aggregate metrics across seeds."""
    agg = {}
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_r"]:
        vals = [r[metric] for r in seed_runs if metric in r]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
    agg["n_seeds"] = len(seed_runs)
    return agg


def cache_chemeleon_embeddings(smiles_list):
    """Compute and cache CheMeleon embeddings for all molecules."""
    cache_name = "chemeleon"
    cache_file = CACHE_DIR / f"{cache_name}.npz"

    if cache_file.exists():
        print(f"Cache exists at {cache_file}, loading...")
        data = np.load(cache_file, allow_pickle=True)
        cached_smiles = data['smiles'].tolist()
        cached_embs = data['embeddings']
        emb_dim = int(data['emb_dim'])
        print(f"  {len(cached_smiles)} molecules cached (dim={emb_dim})")

        missing = set(smiles_list) - set(cached_smiles)
        if not missing:
            emb_dict = {smi: cached_embs[i] for i, smi in enumerate(cached_smiles)}
            return emb_dict, emb_dim
        print(f"  {len(missing)} molecules missing from cache, recomputing all...")

    print(f"Computing CheMeleon embeddings...")
    print(f"  Loading checkpoint: {CHEMELEON_CKPT}")

    # Load CheMeleon model following official pattern
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.nn import BondMessagePassing, MeanAggregation
    from chemprop.data import BatchMolGraph, MoleculeDatapoint
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN

    ckpt = torch.load(str(CHEMELEON_CKPT), weights_only=True, map_location='cpu')
    mp = BondMessagePassing(**ckpt['hyper_parameters'])
    mp.load_state_dict(ckpt['state_dict'])

    agg = MeanAggregation()
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=RegressionFFN(input_dim=mp.output_dim),
    )
    model.eval()
    model.to('cpu')

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    emb_dim = mp.output_dim
    print(f"  Model loaded: d_h={ckpt['hyper_parameters']['d_h']}, depth={ckpt['hyper_parameters']['depth']}, output_dim={emb_dim}")

    # Encode in batches
    batch_size = 64
    all_embs = []
    valid_smiles = []
    failed = 0

    from rdkit.Chem import MolFromSmiles

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        try:
            mol_graphs = []
            batch_valid = []
            for smi in batch:
                try:
                    mol = MolFromSmiles(smi)
                    if mol is None:
                        failed += 1
                        continue
                    mg = featurizer(mol)
                    mol_graphs.append(mg)
                    batch_valid.append(smi)
                except Exception:
                    failed += 1

            if mol_graphs:
                bmg = BatchMolGraph(mol_graphs)
                bmg.to('cpu')
                with torch.no_grad():
                    embs = model.fingerprint(bmg).numpy(force=True)

                # Check for NaN or all-zero
                nan_mask = np.isnan(embs).any(axis=1)
                zero_mask = (embs == 0).all(axis=1)
                for j, smi in enumerate(batch_valid):
                    if not nan_mask[j] and not zero_mask[j]:
                        valid_smiles.append(smi)
                        all_embs.append(embs[j])
                    else:
                        failed += 1

        except Exception as e:
            print(f"  Batch {i} failed: {e}")
            failed += len(batch)

        if (i // batch_size) % 100 == 0:
            print(f"  Encoded {min(i + batch_size, len(smiles_list)):,}/{len(smiles_list):,} "
                  f"(valid: {len(valid_smiles):,}, failed: {failed})")

    all_embs = np.array(all_embs, dtype=np.float32)
    print(f"  Final: {len(valid_smiles):,} valid embeddings, {failed} failed")

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        smiles=np.array(valid_smiles, dtype=object),
        embeddings=all_embs,
        emb_dim=emb_dim,
    )
    print(f"  Cache saved: {cache_file} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Free model memory
    del model, mp, agg, featurizer
    gc.collect()

    emb_dict = {smi: all_embs[i] for i, smi in enumerate(valid_smiles)}
    return emb_dict, emb_dim


def main():
    print("=" * 60)
    print("CheMeleon Phase 1 Evaluation")
    print("=" * 60)

    # Load data
    df = load_data()
    all_smiles = list(set(df["mol_a"].tolist() + df["mol_b"].tolist()))
    print(f"  {len(all_smiles):,} unique molecules")

    # Step 1: Cache embeddings
    t0 = time.time()
    emb_dict, emb_dim = cache_chemeleon_embeddings(all_smiles)
    cache_time = time.time() - t0
    print(f"  Embedding time: {cache_time:.0f}s")

    # Check coverage
    missing = [s for s in all_smiles if s not in emb_dict]
    if missing:
        print(f"  WARNING: {len(missing)} molecules not embedded, using zero vectors")
        zero = np.zeros(emb_dim, dtype=np.float32)
        for s in missing:
            emb_dict[s] = zero

    # Step 2: Run Phase 1 evaluation (EditDiff, within-assay, 3 seeds)
    print(f"\nRunning Phase 1 evaluation (EditDiff, within-assay, 3 seeds)...")
    seed_runs = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...", end=" ", flush=True)
        try:
            train_df, val_df, test_df = split_data(df, "assay_within", seed)
            metrics, _ = run_single_experiment(
                train_df, val_df, test_df, emb_dict, emb_dim, "EditDiff", seed
            )
            seed_runs.append(metrics)
            print(f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                  f"Pearson={metrics['pearson_r']:.4f}, R²={metrics['r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            gc.collect()

    if not seed_runs:
        print("ERROR: No successful runs!")
        return

    # Step 3: Save results
    result = {
        "aggregated": aggregate_seeds(seed_runs),
        "per_seed": seed_runs,
        "emb_dim": emb_dim,
    }

    results_file = RESULTS_DIR / "all_results.json"
    try:
        with open(results_file) as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    if "phase1" not in all_results:
        all_results["phase1"] = {}
    all_results["phase1"]["chemeleon"] = result

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    agg = result["aggregated"]
    print(f"\n{'=' * 60}")
    print(f"CheMeleon Phase 1 Results:")
    print(f"  MAE:      {agg['mae_mean']:.4f} ± {agg['mae_std']:.4f}")
    print(f"  Spearman: {agg['spearman_r_mean']:.4f} ± {agg['spearman_r_std']:.4f}")
    print(f"  Pearson:  {agg['pearson_r_mean']:.4f} ± {agg['pearson_r_std']:.4f}")
    print(f"  R²:       {agg['r2_mean']:.4f} ± {agg['r2_std']:.4f}")
    print(f"  Dim:      {emb_dim}")
    print(f"  Saved to: {results_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
