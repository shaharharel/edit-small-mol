#!/usr/bin/env python3
"""
Pre-compute and cache embeddings for all unique molecules in the shared pairs dataset.

Saves embeddings as .npz files for fast loading during experiments.

Usage:
    conda run -n quris python -u scripts/precompute_embeddings.py
    conda run -n quris python -u scripts/precompute_embeddings.py --embedder morgan
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

import argparse
import time
import numpy as np
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False

CACHE_DIR = Path("data/embedding_cache")
DATA_FILE = Path("data/overlapping_assays/extracted/shared_pairs_deduped.csv")


def get_unique_smiles(data_file):
    """Get all unique SMILES from the dataset."""
    print(f"Loading molecules from {data_file}...")
    df = pd.read_csv(data_file, usecols=['mol_a', 'mol_b'])
    smiles = sorted(set(df['mol_a'].tolist() + df['mol_b'].tolist()))
    print(f"  {len(smiles):,} unique molecules")
    return smiles


def compute_and_save(smiles_list, embedder_name):
    """Compute embeddings and save to cache."""
    cache_file = CACHE_DIR / f"{embedder_name}.npz"
    if cache_file.exists():
        print(f"  Cache exists: {cache_file} — skipping")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if embedder_name == "morgan":
        from src.embedding.fingerprints import FingerprintEmbedder
        embedder = FingerprintEmbedder(fp_type="morgan", radius=2, n_bits=2048)
        embeddings = np.array([embedder.encode(smi) for smi in smiles_list])
        emb_dim = 2048

    elif embedder_name == "chemberta2-mlm":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name='chemberta2-mlm', device='cpu', batch_size=128)
        embeddings = embedder.encode(smiles_list)
        emb_dim = embedder.embedding_dim

    elif embedder_name == "chemberta2-mtr":
        from src.embedding.chemberta import ChemBERTaEmbedder
        embedder = ChemBERTaEmbedder(model_name='chemberta2-mtr', device='cpu', batch_size=128)
        embeddings = embedder.encode(smiles_list)
        emb_dim = embedder.embedding_dim

    elif embedder_name == "chemprop-dmpnn":
        from src.embedding.chemprop import ChemPropEmbedder
        embedder = ChemPropEmbedder(featurizer_type='morgan')
        embeddings = embedder.encode(smiles_list)
        emb_dim = embedder.embedding_dim

    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")

    elapsed = time.time() - t0

    # Save: embeddings array + smiles index
    np.savez_compressed(
        cache_file,
        embeddings=embeddings.astype(np.float32),
        smiles=np.array(smiles_list),
        emb_dim=emb_dim,
    )
    size_mb = cache_file.stat().st_size / 1e6
    print(f"  Saved {embedder_name}: {embeddings.shape} ({size_mb:.1f} MB) in {elapsed:.1f}s")


def load_cached_embeddings(embedder_name):
    """Load pre-computed embeddings from cache."""
    cache_file = CACHE_DIR / f"{embedder_name}.npz"
    data = np.load(cache_file, allow_pickle=True)
    smiles = data['smiles'].tolist()
    embeddings = data['embeddings']
    emb_dim = int(data['emb_dim'])
    emb_dict = {smi: embeddings[i] for i, smi in enumerate(smiles)}
    return emb_dict, emb_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default=None,
                        help="Compute only this embedder (default: all)")
    args = parser.parse_args()

    smiles = get_unique_smiles(DATA_FILE)

    embedders = ["morgan", "chemberta2-mlm", "chemberta2-mtr", "chemprop-dmpnn"]
    if args.embedder:
        embedders = [args.embedder]

    for emb_name in embedders:
        print(f"\n--- {emb_name} ---")
        compute_and_save(smiles, emb_name)

    print("\nAll embeddings cached!")
    for f in sorted(CACHE_DIR.glob("*.npz")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
