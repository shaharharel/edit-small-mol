import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
from experiment_config import ExperimentConfig
from run_experiment import run_experiment


def main():
    # Splitters to run (need to optimize 'butina')
    #splitters = ['random', 'scaffold', 'target', 'few_shot_target']
    splitters = ['random', 'scaffold']
    #splitters = ['core']
    for splitter_type in splitters:
        print(f"\n{'='*80}")
        print(f"Running experiment with {splitter_type.upper()} split")
        print(f"{'='*80}\n")

        # Configure splitter-specific parameters
        splitter_params = {}
        if splitter_type == 'target':
            splitter_params[splitter_type] = {'target_col': 'target_chembl_id'}
        elif splitter_type == 'stratified':
            splitter_params[splitter_type] = {'property_col': 'delta'}
        elif splitter_type == 'few_shot_target':
            # Few-shot learning: 30% of targets with only 100 training examples each
            splitter_params[splitter_type] = {
                'target_col': 'target_chembl_id',
                'few_shot_target_fraction': 0.3,
                'few_shot_samples': 100  # Can also try 1000
            }
        elif splitter_type == 'core':
            # Core-based split: ensure unique cores in val/test
            splitter_params[splitter_type] = {'core_col': 'core'}

        config = ExperimentConfig(
            experiment_name=f"small_molecule_edit_prediction_{splitter_type}",

            # For standard methods: use chembl_pairs_long_sample.csv
            # For Structured Edit methods: use chembl_pairs_mmpdb.csv (has MMP atom-level columns)
            #data_file="../data/pairs/chembl_pairs_long_sample.csv",
            data_file="../data/pairs/mmpdb/chembl_pairs_mmpdb.csv",  # For structured edit

            splitter_type=splitter_type,
            splitter_params=splitter_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,

            num_tasks=10,
            min_pairs_per_property=0,  # Minimum number of pairs required per property
            min_properties_per_edit=1,  # Minimum number of properties an edit must appear in

            # Per-method configuration with embedder_type and trainable_encoder
            # Embedder types: 'chemprop_dmpnn' (D-MPNN GNN, 300-dim, GPU)
            #                 'chemprop_morgan' (Morgan fingerprints, 2048-dim, CPU)
            #                 'chemprop_rdkit' (RDKit 2D descriptors, 217-dim, CPU)
            #                 'chemberta' (ChemBERTa transformer, 768-dim, GPU)
            # Note: edit_framework_structured requires 'chemprop_dmpnn'
            methods=[
                {
                    'name': 'Structured Edit - D-MPNN',
                    'type': 'edit_framework_structured',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': False,
                    'encoder_device': 'auto',
                    'edit_mlp_dims': [512, 512, 300],  # Edit embedding MLP
                    'delta_mlp_dims': [512, 256, 128],  # Delta prediction MLP
                    'k_hop_env': 2,  # K-hop neighborhood for local environment
                    'use_local_delta': True,  # Use local delta from k-hop env
                    'use_rdkit_descriptors': True,  # Include RDKit fragment descriptors
                    'dropout': 0.1,
                    'lr': 0.001,
                    'encoder_lr': 1e-5,
                    'batch_size': 32,
                    'max_epochs': 2
                },
                {
                    'name': 'Structured Edit - D-MPNN Trainable',
                    'type': 'edit_framework_structured',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': True,
                    'encoder_device': 'auto',
                    'edit_mlp_dims': [512, 512, 300],
                    'delta_mlp_dims': [512, 256, 128],
                    'k_hop_env': 2,
                    'use_local_delta': True,
                    'use_rdkit_descriptors': True,
                    'dropout': 0.1,
                    'lr': 0.001,
                    'encoder_lr': 1e-5,
                    'batch_size': 128,
                    'max_epochs': 2
                },
            ],

            metrics=['mae', 'rmse', 'r2', 'pearson_r', 'spearman_r'],

            output_dir=f'results/{splitter_type}',

            additional_test_files={},

            include_cluster_analysis=True,
            n_clusters=4,

            include_edit_embedding_comparison=True
        )

        results, report_path = run_experiment(config)

        print(f"\n{'='*80}")
        print(f"Experiment '{config.experiment_name}' completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")

        # Free memory after each splitter to avoid accumulation
        del results, report_path, config
        gc.collect()
        print(f"Memory freed after {splitter_type} splitter\n")

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print("\nResults directories:")
    for splitter_type in splitters:
        print(f"  - experiments/results/{splitter_type}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
    '''
                # Structured Edit Framework - requires MMPDB data format
                # Uses atom-level MMP information: removed_atoms_A, added_atoms_B, attach_atoms_A, mapped_pairs
                # Data file must be changed to mmpdb format (e.g., chembl_pairs_mmpdb.csv)
                {
                    'name': 'Structured Edit - D-MPNN',
                    'type': 'edit_framework_structured',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': False,
                    'encoder_device': 'auto',
                    'edit_mlp_dims': [512, 512, 300],  # Edit embedding MLP
                    'delta_mlp_dims': [512, 256, 128],  # Delta prediction MLP
                    'k_hop_env': 2,  # K-hop neighborhood for local environment
                    'use_local_delta': True,  # Use local delta from k-hop env
                    'use_rdkit_descriptors': True,  # Include RDKit fragment descriptors
                    'dropout': 0.1,
                    'lr': 0.001,
                    'encoder_lr': 1e-5,
                    'batch_size': 32,
                    'max_epochs': 2
                },
                {
                    'name': 'Structured Edit - D-MPNN Trainable',
                    'type': 'edit_framework_structured',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': True,
                    'encoder_device': 'auto',
                    'edit_mlp_dims': [512, 512, 300],
                    'delta_mlp_dims': [512, 256, 128],
                    'k_hop_env': 2,
                    'use_local_delta': True,
                    'use_rdkit_descriptors': True,
                    'dropout': 0.1,
                    'lr': 0.001,
                    'encoder_lr': 1e-5,
                    'batch_size': 128,
                    'max_epochs': 2
                },
                                {
                    'name': 'Edit Framework - D-MPNN',
                    'type': 'edit_framework',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': False,
                    'encoder_device': 'auto',
                    'use_edit_fragments': False,
                    'hidden_dims': [512, 256],  # Shared backbone
                    'head_hidden_dims': [256, 128],  # Task heads (no dropout on last layer)
                    'dropout': 0.1,
                    'lr': 0.001,  # Learning rate for MLP heads
                    'encoder_lr': 1e-5,  # Learning rate for encoder (if trainable)
                    'batch_size': 128,
                    'max_epochs': 2
                },
                {
                    'name': 'Edit Framework - D-MPNN Trainable',
                    'type': 'edit_framework',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': True,
                    'encoder_device': 'auto',
                    'use_edit_fragments': False,
                    'hidden_dims': [512, 256],  # Shared backbone
                    'head_hidden_dims': [256, 128],  # Task heads (no dropout on last layer)
                    'dropout': 0.1,
                    'lr': 0.001,  # Learning rate for MLP heads
                    'encoder_lr': 1e-5,  # Learning rate for encoder (GNN/transformer)
                    'batch_size': 128,
                    'max_epochs': 2
                },
                {
                    'name': 'Edit Framework - Morgan',
                    'type': 'edit_framework',
                    'embedder_type': 'chemprop_morgan',
                    'trainable_encoder': False,  # Morgan FP is not trainable
                    'encoder_device': 'auto',
                    'use_edit_fragments': False,
                    'hidden_dims': [512, 256],  # Shared backbone
                    'head_hidden_dims': [256, 128],  # Task heads (no dropout on last layer)
                    'dropout': 0.1,
                    'lr': 0.001,  # Learning rate for MLP heads
                    'batch_size': 128,
                    'max_epochs': 2
                },

                {
                    'name': 'Baseline - ChemBERTa',
                    'type': 'baseline_property',
                    'embedder_type': 'chemberta',
                    'trainable_encoder': False,  # Frozen transformer
                    'encoder_device': 'auto',
                    'hidden_dims': [512, 256],
                    'head_hidden_dims': [256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 128,
                    'max_epochs': 2
                },
                {
                    'name': 'Baseline - D-MPNN Trainable',
                    'type': 'baseline_property',
                    'embedder_type': 'chemprop_dmpnn',
                    'trainable_encoder': True,  # End-to-end trainable GNN
                    'encoder_device': 'auto',
                    'hidden_dims': [512, 256],
                    'head_hidden_dims': [256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'encoder_lr': 1e-5,  # Lower LR for encoder fine-tuning
                    'batch_size': 128,
                    'max_epochs': 2
                },
                {
                    'name': 'Baseline - Morgan',
                    'type': 'baseline_property',
                    'embedder_type': 'chemprop_morgan',  # Morgan fingerprints (2048-dim)
                    'trainable_encoder': False,  # Morgan FP is not trainable
                    'hidden_dims': [512, 256],
                    'head_hidden_dims': [256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 128,
                    'max_epochs': 2
                }

    '''