from experiment_config import ExperimentConfig


config = ExperimentConfig(
    experiment_name="edit_framework_comparison",

    data_file="data/pairs/chembl_pairs_long_sample.csv",
    min_pairs_per_property=1000,
    num_tasks=5,

    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,

    splitter_type='scaffold',
    splitter_params={
        'scaffold': {'use_generic': True},
        'target': {'target_col': 'target_id'},
        'butina': {'cutoff': 0.35, 'fp_radius': 2, 'fp_size': 2048},
        'stratified': {'property_col': 'delta', 'n_bins': 5},
        'temporal': {'time_col': 'timestamp'}
    },

    methods=[
        {
            'name': 'baseline_property_predictor',
            'type': 'baseline_property',
            'hidden_dims': None,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'epochs': 50
        },
        {
            'name': 'edit_framework_mode1',
            'type': 'edit_framework',
            'use_edit_fragments': False,
            'trainable_edit_embeddings': True,
            'trainable_edit_dims': [512, 256],
            'hidden_dims': None,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'epochs': 50
        },
        {
            'name': 'edit_framework_mode2',
            'type': 'edit_framework',
            'use_edit_fragments': True,
            'trainable_edit_embeddings': True,
            'trainable_edit_dims': [512, 256],
            'hidden_dims': None,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
    ],

    metrics=['mae', 'rmse', 'r2', 'pearson_r'],

    test_datasets=[],

    save_models=True,
    models_dir="experiments/saved_models",
    output_dir="experiments/results"
)
