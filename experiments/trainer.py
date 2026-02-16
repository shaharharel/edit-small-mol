import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
import numpy as np
import pandas as pd
import gc
from src.models import PropertyPredictor, EditEffectPredictor
from src.models.predictors import TrainablePropertyPredictor, TrainableEditEffectPredictor


def train_all_models(models: Dict, train_data: Dict, config) -> Dict:
    """
    Train all models. Each method uses its own embedder to compute embeddings on-the-fly.
    """
    trained_models = {}
    task_names = list(train_data.keys())
    num_tasks = len(task_names)

    for method_name, method_info in models.items():
        print(f"\nTraining {method_name}...")

        method_type = method_info['type']
        model = method_info['model']
        method_config = method_info['config']

        # Get the embedder for this method
        if method_type in ['baseline_property', 'trainable_baseline_property']:
            embedder = method_info.get('embedder')
        else:
            embedder = method_info.get('mol_embedder')

        print(f"  Embedder: {method_config.get('embedder_type', 'default')}, dim={embedder.embedding_dim if embedder else 'N/A'}")

        split_name = config.splitter_type
        checkpoint_path = Path(config.models_dir) / f"{method_name}_{split_name}.pt"

        load_checkpoint = method_config.get('load_checkpoint')
        if load_checkpoint and Path(load_checkpoint).exists():
            print(f"Loading from {load_checkpoint}")
            if method_type == 'baseline_property':
                model = PropertyPredictor.load_checkpoint(load_checkpoint, embedder=embedder)
            elif method_type == 'edit_framework':
                model = EditEffectPredictor.load_checkpoint(load_checkpoint, mol_embedder=embedder)
            method_info['model'] = model
            print("✓ Model loaded successfully")

        elif method_type == 'baseline_property':
            # PropertyPredictor with frozen embedder - compute embeddings on-the-fly
            train_smiles_list = []
            train_val_list = []
            val_smiles_list = []
            val_val_list = []

            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                for idx in range(len(train_prop)):
                    # mol_a sample
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = train_prop.iloc[idx]['value_a']
                    train_val_list.append(y_row)
                    train_smiles_list.append(train_prop.iloc[idx]['mol_a'])

                    # mol_b sample
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = train_prop.iloc[idx]['value_b']
                    train_val_list.append(y_row)
                    train_smiles_list.append(train_prop.iloc[idx]['mol_b'])

                for idx in range(len(val_prop)):
                    # mol_a sample
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = val_prop.iloc[idx]['value_a']
                    val_val_list.append(y_row)
                    val_smiles_list.append(val_prop.iloc[idx]['mol_a'])

                    # mol_b sample
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = val_prop.iloc[idx]['value_b']
                    val_val_list.append(y_row)
                    val_smiles_list.append(val_prop.iloc[idx]['mol_b'])

            y_train = np.array(train_val_list, dtype=np.float32)
            y_val = np.array(val_val_list, dtype=np.float32)

            # Compute embeddings on-the-fly with progress bar
            print(f"  Computing embeddings for {len(train_smiles_list)} training samples...")
            mol_emb_train = np.array(embedder.encode(train_smiles_list, show_progress=True), dtype=np.float32)
            if val_smiles_list:
                print(f"  Computing embeddings for {len(val_smiles_list)} validation samples...")
                mol_emb_val = np.array(embedder.encode(val_smiles_list, show_progress=True), dtype=np.float32)
            else:
                mol_emb_val = None

            model.fit(
                mol_emb_train=mol_emb_train,
                y_train=y_train,
                mol_emb_val=mol_emb_val,
                y_val=y_val if mol_emb_val is not None else None,
                verbose=True
            )

            del mol_emb_train, mol_emb_val, y_train, y_val
            del train_smiles_list, val_smiles_list, train_val_list, val_val_list
            gc.collect()

        elif method_type == 'trainable_baseline_property':
            # TrainablePropertyPredictor: End-to-end trainable GNN + MLP
            train_smiles_list = []
            train_val_list = []
            val_smiles_list = []
            val_val_list = []

            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                for idx in range(len(train_prop)):
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = train_prop.iloc[idx]['value_a']
                    train_val_list.append(y_row)
                    train_smiles_list.append(train_prop.iloc[idx]['mol_a'])

                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = train_prop.iloc[idx]['value_b']
                    train_val_list.append(y_row)
                    train_smiles_list.append(train_prop.iloc[idx]['mol_b'])

                for idx in range(len(val_prop)):
                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = val_prop.iloc[idx]['value_a']
                    val_val_list.append(y_row)
                    val_smiles_list.append(val_prop.iloc[idx]['mol_a'])

                    y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                    y_row[i] = val_prop.iloc[idx]['value_b']
                    val_val_list.append(y_row)
                    val_smiles_list.append(val_prop.iloc[idx]['mol_b'])

            y_train = np.array(train_val_list, dtype=np.float32)
            y_val = np.array(val_val_list, dtype=np.float32)

            print(f"  Training end-to-end with {len(train_smiles_list)} samples (trainable GNN)...")
            model.fit(
                smiles_train=train_smiles_list,
                y_train=y_train,
                smiles_val=val_smiles_list if val_smiles_list else None,
                y_val=y_val if len(val_smiles_list) > 0 else None,
                verbose=True
            )

            del train_smiles_list, val_smiles_list, train_val_list, val_val_list
            del y_train, y_val
            gc.collect()

        elif method_type == 'trainable_edit_framework':
            # TrainableEditEffectPredictor: End-to-end trainable GNN + MLP for edit prediction
            train_df_list = []
            val_df_list = []
            for prop in task_names:
                train_df_list.append(train_data[prop]['train'])
                val_df_list.append(train_data[prop]['val'])

            train_combined = pd.concat(train_df_list, ignore_index=True)
            val_combined = pd.concat(val_df_list, ignore_index=True)

            delta_train = np.full((len(train_combined), num_tasks), np.nan, dtype=np.float32)
            delta_val = np.full((len(val_combined), num_tasks), np.nan, dtype=np.float32)

            train_idx = 0
            val_idx = 0
            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                delta_train[train_idx:train_idx+len(train_prop), i] = train_prop['delta'].values
                delta_val[val_idx:val_idx+len(val_prop), i] = val_prop['delta'].values

                train_idx += len(train_prop)
                val_idx += len(val_prop)

            print(f"  Training end-to-end with {len(train_combined)} pairs (trainable GNN)...")

            if len(val_combined) > 0:
                model.fit(
                    smiles_a_train=train_combined['mol_a'].tolist(),
                    smiles_b_train=train_combined['mol_b'].tolist(),
                    delta_y_train=delta_train,
                    smiles_a_val=val_combined['mol_a'].tolist(),
                    smiles_b_val=val_combined['mol_b'].tolist(),
                    delta_y_val=delta_val,
                    verbose=True
                )
            else:
                model.fit(
                    smiles_a_train=train_combined['mol_a'].tolist(),
                    smiles_b_train=train_combined['mol_b'].tolist(),
                    delta_y_train=delta_train,
                    verbose=True
                )

            del train_combined, val_combined, delta_train, delta_val
            gc.collect()

        elif method_type == 'edit_framework':
            # EditEffectPredictor with frozen embedder - compute embeddings on-the-fly
            train_df_list = []
            val_df_list = []
            for prop in task_names:
                train_df_list.append(train_data[prop]['train'])
                val_df_list.append(train_data[prop]['val'])

            train_combined = pd.concat(train_df_list, ignore_index=True)
            val_combined = pd.concat(val_df_list, ignore_index=True)

            delta_train = np.full((len(train_combined), num_tasks), np.nan, dtype=np.float32)
            delta_val = np.full((len(val_combined), num_tasks), np.nan, dtype=np.float32)

            train_idx = 0
            val_idx = 0
            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                delta_train[train_idx:train_idx+len(train_prop), i] = train_prop['delta'].values
                delta_val[val_idx:val_idx+len(val_prop), i] = val_prop['delta'].values

                train_idx += len(train_prop)
                val_idx += len(val_prop)

            # Compute embeddings on-the-fly with progress bar
            print(f"  Computing embeddings for {len(train_combined)} training pairs...")
            print("    mol_a embeddings:")
            mol_emb_a_train = np.array(embedder.encode(train_combined['mol_a'].tolist(), show_progress=True), dtype=np.float32)
            print("    mol_b embeddings:")
            mol_emb_b_train = np.array(embedder.encode(train_combined['mol_b'].tolist(), show_progress=True), dtype=np.float32)

            if len(val_combined) > 0:
                print(f"  Computing embeddings for {len(val_combined)} validation pairs...")
                mol_emb_a_val = np.array(embedder.encode(val_combined['mol_a'].tolist(), show_progress=True), dtype=np.float32)
                mol_emb_b_val = np.array(embedder.encode(val_combined['mol_b'].tolist(), show_progress=True), dtype=np.float32)

                model.fit(
                    mol_emb_a=mol_emb_a_train,
                    mol_emb_b=mol_emb_b_train,
                    delta_y=delta_train,
                    mol_emb_a_val=mol_emb_a_val,
                    mol_emb_b_val=mol_emb_b_val,
                    delta_y_val=delta_val,
                    verbose=True
                )
                del mol_emb_a_val, mol_emb_b_val
            else:
                model.fit(
                    mol_emb_a=mol_emb_a_train,
                    mol_emb_b=mol_emb_b_train,
                    delta_y=delta_train,
                    verbose=True
                )

            del mol_emb_a_train, mol_emb_b_train, delta_train, delta_val
            del train_combined, val_combined
            gc.collect()

        elif method_type == 'edit_framework_structured':
            # StructuredEditEffectPredictor: uses MMP structural info
            from src.data.mmp_parser import parse_mmp_batch

            train_df_list = []
            val_df_list = []
            for prop in task_names:
                train_df_list.append(train_data[prop]['train'])
                val_df_list.append(train_data[prop]['val'])

            train_combined = pd.concat(train_df_list, ignore_index=True)
            val_combined = pd.concat(val_df_list, ignore_index=True)

            # Check if MMP columns exist
            required_cols = ['removed_atoms_A', 'added_atoms_B', 'attach_atoms_A']
            missing_cols = [c for c in required_cols if c not in train_combined.columns]
            if missing_cols:
                raise ValueError(
                    f"Structured edit framework requires MMP columns: {missing_cols}. "
                    "Use data from mmpdb format (e.g., chembl_pairs_mmpdb_*.csv)"
                )

            train_mmp = parse_mmp_batch(train_combined)
            val_mmp = parse_mmp_batch(val_combined) if len(val_combined) > 0 else None

            delta_train = np.full((len(train_combined), num_tasks), np.nan, dtype=np.float32)
            delta_val = np.full((len(val_combined), num_tasks), np.nan, dtype=np.float32)

            train_idx = 0
            val_idx = 0
            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                delta_train[train_idx:train_idx+len(train_prop), i] = train_prop['delta'].values
                if len(val_prop) > 0:
                    delta_val[val_idx:val_idx+len(val_prop), i] = val_prop['delta'].values

                train_idx += len(train_prop)
                val_idx += len(val_prop)

            print(f"Training structured edit model on {len(train_combined)} samples...")

            if val_mmp is not None and len(val_combined) > 0:
                model.fit(
                    smiles_A=train_combined['mol_a'].tolist(),
                    smiles_B=train_combined['mol_b'].tolist(),
                    removed_atoms_A=train_mmp['removed_atom_indices_A'],
                    added_atoms_B=train_mmp['added_atom_indices_B'],
                    attach_atoms_A=train_mmp['attach_atom_indices_A'],
                    delta_y=delta_train,
                    mapped_pairs=train_mmp['mapped_atom_pairs'],
                    smiles_A_val=val_combined['mol_a'].tolist(),
                    smiles_B_val=val_combined['mol_b'].tolist(),
                    removed_atoms_A_val=val_mmp['removed_atom_indices_A'],
                    added_atoms_B_val=val_mmp['added_atom_indices_B'],
                    attach_atoms_A_val=val_mmp['attach_atom_indices_A'],
                    delta_y_val=delta_val,
                    mapped_pairs_val=val_mmp['mapped_atom_pairs'],
                    verbose=True
                )
            else:
                model.fit(
                    smiles_A=train_combined['mol_a'].tolist(),
                    smiles_B=train_combined['mol_b'].tolist(),
                    removed_atoms_A=train_mmp['removed_atom_indices_A'],
                    added_atoms_B=train_mmp['added_atom_indices_B'],
                    attach_atoms_A=train_mmp['attach_atom_indices_A'],
                    delta_y=delta_train,
                    mapped_pairs=train_mmp['mapped_atom_pairs'],
                    verbose=True
                )

            gc.collect()

        if config.save_models:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(str(checkpoint_path))
            print(f"Saved to {checkpoint_path}")

        trained_models[method_name] = {
            'type': method_type,
            'model': model,
            'config': method_config
        }
        if method_type in ['baseline_property', 'trainable_baseline_property']:
            trained_models[method_name]['embedder'] = embedder
        else:
            trained_models[method_name]['mol_embedder'] = embedder
            if method_type == 'edit_framework':
                trained_models[method_name]['edit_embedder'] = method_info.get('edit_embedder')

    return trained_models
