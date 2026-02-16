import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils import get_splitter


def load_datasets(config) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    df = pd.read_csv(config.data_file)
    print(f"\nInitial dataset: {len(df):,} pairs")

    train_data = {}
    test_datasets = {}

    # Store few-shot metadata in config for reporting
    config.few_shot_metadata = None

    df = df[df['edit_smiles'].notna() & df['mol_a'].notna() & df['mol_b'].notna() & df['delta'].notna()]
    print(f"After removing missing values: {len(df):,} pairs")

    # Remove duplicates based on (mol_a, mol_b, property_name)
    n_before = len(df)
    df = df.drop_duplicates(subset=['mol_a', 'mol_b', 'property_name'], keep='first')
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"After removing duplicates (mol_a, mol_b, property_name): {len(df):,} pairs ({n_removed:,} removed)")

    property_counts = df.groupby('property_name').size()
    valid_properties = property_counts[property_counts >= config.min_pairs_per_property].index.tolist()
    df = df[df['property_name'].isin(valid_properties)]
    print(f"After filtering properties with >={config.min_pairs_per_property} pairs: {len(df):,} pairs across {len(valid_properties)} properties")

    properties = df['property_name'].unique()
    selected_properties = properties[:config.num_tasks] if len(properties) > config.num_tasks else properties

    df = df[df['property_name'].isin(selected_properties)]
    print(f"After selecting top {config.num_tasks} properties: {len(df):,} pairs")

    edit_property_counts = df.groupby('edit_smiles')['property_name'].nunique()
    multi_property_edits = edit_property_counts[edit_property_counts >= config.min_properties_per_edit].index
    df = df[df['edit_smiles'].isin(multi_property_edits)].copy()

    print(f"After filtering edits appearing in >={config.min_properties_per_edit} properties: {len(df):,} pairs")
    print(f"Final dataset: {len(df):,} pairs with {len(df['edit_smiles'].unique()):,} unique edits\n")

    # Get splitter params, separating split() method args from __init__() args
    splitter_params = config.splitter_params.get(config.splitter_type, {})

    # Parameters that go to split() method, not __init__()
    split_method_params = {}
    init_params = {}

    # Known split() method parameters (vary by splitter type)
    split_param_names = {'property_col', 'smiles_col', 'target_col', 'time_col', 'core_col'}

    for key, value in splitter_params.items():
        if key in split_param_names:
            split_method_params[key] = value
        else:
            init_params[key] = value

    # PROPERTY-LEVEL FEW-SHOT LOGIC
    # For few_shot_target splitter with single-target properties, apply few-shot at property level
    if config.splitter_type == 'few_shot_target':
        print(f"{'='*70}")
        print("PROPERTY-LEVEL FEW-SHOT SPLIT")
        print(f"{'='*70}")

        # Extract few-shot parameters
        few_shot_fraction = init_params.get('few_shot_target_fraction', 0.3)
        few_shot_samples = init_params.get('few_shot_samples', 100)

        # Select properties for few-shot learning
        n_properties = len(selected_properties)
        n_few_shot_properties = max(1, int(n_properties * few_shot_fraction))

        np.random.seed(config.random_seed)
        few_shot_properties = np.random.choice(selected_properties, size=n_few_shot_properties, replace=False)
        regular_properties = [p for p in selected_properties if p not in few_shot_properties]

        print(f"Total properties: {n_properties}")
        print(f"Few-shot properties ({few_shot_fraction*100:.0f}%): {n_few_shot_properties}")
        print(f"  → {list(few_shot_properties)}")
        print(f"Regular properties: {len(regular_properties)}")
        print(f"  → {list(regular_properties)}")
        print(f"Few-shot samples per property: {few_shot_samples}")
        print(f"{'='*70}\n")

        # Store few-shot metadata for reporting
        config.few_shot_metadata = {
            'few_shot_properties': list(few_shot_properties),
            'regular_properties': regular_properties,
            'few_shot_samples': few_shot_samples,
            'few_shot_fraction': few_shot_fraction
        }

        # Use random splitter for within-property splits
        within_property_splitter = get_splitter(
            'random',  # Use random split within each property
            train_size=config.train_ratio,
            val_size=config.val_ratio,
            test_size=config.test_ratio,
            random_state=config.random_seed
        )

        # Split each property
        for prop in selected_properties:
            prop_data = df[df['property_name'] == prop].copy()

            if prop in few_shot_properties:
                # Few-shot property: limit training samples
                print(f"Property: {prop} [FEW-SHOT]")
                print(f"  Total pairs: {len(prop_data):,}")

                # Shuffle data
                prop_data = prop_data.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)

                # Take limited samples for training
                n_train = min(few_shot_samples, len(prop_data))
                train = prop_data.iloc[:n_train].copy()

                # Split remaining between val and test
                remaining = prop_data.iloc[n_train:].copy()
                val_fraction = config.val_ratio / (config.val_ratio + config.test_ratio)
                n_val = int(len(remaining) * val_fraction)

                val = remaining.iloc[:n_val].copy()
                test = remaining.iloc[n_val:].copy()

                print(f"  Train: {len(train):,} (limited to {few_shot_samples})")
                print(f"  Val: {len(val):,}")
                print(f"  Test: {len(test):,}\n")
            else:
                # Regular property: normal 70/15/15 split
                print(f"Property: {prop} [REGULAR]")
                print(f"  Total pairs: {len(prop_data):,}")

                train, val, test = within_property_splitter.split(prop_data, smiles_col='mol_a')

                print(f"  Train: {len(train):,} ({len(train)/len(prop_data)*100:.1f}%)")
                print(f"  Val: {len(val):,} ({len(val)/len(prop_data)*100:.1f}%)")
                print(f"  Test: {len(test):,} ({len(test)/len(prop_data)*100:.1f}%)\n")

            train_data[prop] = {
                'train': train,
                'val': val,
                'test': test
            }

        # Summary statistics
        total_train = sum(len(splits['train']) for splits in train_data.values())
        total_val = sum(len(splits['val']) for splits in train_data.values())
        total_test = sum(len(splits['test']) for splits in train_data.values())

        print(f"{'='*70}")
        print("SUMMARY:")
        print(f"  Total training samples: {total_train:,}")
        print(f"  Total validation samples: {total_val:,}")
        print(f"  Total test samples: {total_test:,}")
        print(f"{'='*70}\n")

    else:
        # Original logic for other splitters
        splitter = get_splitter(
            config.splitter_type,
            train_size=config.train_ratio,
            val_size=config.val_ratio,
            test_size=config.test_ratio,
            random_state=config.random_seed,
            **init_params
        )

        for prop in selected_properties:
            prop_data = df[df['property_name'] == prop].copy()

            train, val, test = splitter.split(prop_data, smiles_col='mol_a', **split_method_params)

            train_data[prop] = {
                'train': train,
                'val': val,
                'test': test
            }

    for test_dataset_name in config.test_datasets:
        try:
            test_df = pd.read_csv(test_dataset_name)
            test_datasets[test_dataset_name] = test_df
        except:
            pass

    return train_data, test_datasets
