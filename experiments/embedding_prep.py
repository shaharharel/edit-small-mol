import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.utils import EmbeddingCache, compute_all_embeddings_with_fragments, map_fragment_embeddings_to_pairs


def prepare_embeddings(
    train_data: Dict,
    embedder,
    use_fragments: bool = False,
    cache_dir: str = '.embeddings_cache'
) -> Dict:

    cache = EmbeddingCache(cache_dir=cache_dir)

    all_train_dfs = []
    all_val_dfs = []
    all_test_dfs = []

    for prop, splits in train_data.items():
        all_train_dfs.append(splits['train'])
        all_val_dfs.append(splits['val'])
        all_test_dfs.append(splits['test'])

    train_combined = pd.concat(all_train_dfs, ignore_index=True)
    val_combined = pd.concat(all_val_dfs, ignore_index=True)
    test_combined = pd.concat(all_test_dfs, ignore_index=True)

    train_baseline = train_combined[['mol_b']].rename(columns={'mol_b': 'smiles'})
    val_baseline = val_combined[['mol_b']].rename(columns={'mol_b': 'smiles'})
    test_baseline = test_combined[['mol_b']].rename(columns={'mol_b': 'smiles'})

    emb_lookup = compute_all_embeddings_with_fragments(
        train_edit=train_combined,
        val_edit=val_combined,
        test_edit=test_combined,
        train_baseline=train_baseline,
        val_baseline=val_baseline,
        test_baseline=test_baseline,
        embedder=embedder,
        cache=cache,
        include_edit_fragments=use_fragments
    )

    embeddings = {}

    for prop, splits in train_data.items():
        embeddings[prop] = {}

        for split_name in ['train', 'val', 'test']:
            df = splits[split_name]

            mol_a_emb = np.array([emb_lookup[smi] for smi in df['mol_a']])
            mol_b_emb = np.array([emb_lookup[smi] for smi in df['mol_b']])

            embeddings[prop][split_name] = {
                'mol_a': mol_a_emb,
                'mol_b': mol_b_emb
            }

            if use_fragments:
                try:
                    frag_a_emb, frag_b_emb = map_fragment_embeddings_to_pairs(
                        df, emb_lookup, f"{prop}_{split_name}"
                    )
                    embeddings[prop][split_name]['edit_frag_a'] = frag_a_emb
                    embeddings[prop][split_name]['edit_frag_b'] = frag_b_emb
                except Exception as e:
                    # Handle cases where fragment mapping fails (e.g., no edit_smiles)
                    print(f"  Warning: Could not map fragments for {prop}_{split_name}: {e}")
                    embeddings[prop][split_name]['edit_frag_a'] = np.array([])
                    embeddings[prop][split_name]['edit_frag_b'] = np.array([])

    return embeddings
