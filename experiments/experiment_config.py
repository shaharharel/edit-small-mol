from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ExperimentConfig:
    # Required parameters
    experiment_name: str
    data_file: str
    num_tasks: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    splitter_type: str
    methods: List[Dict]
    metrics: List[str]
    output_dir: str

    # Optional parameters with defaults
    min_pairs_per_property: int = 0
    min_properties_per_edit: int = 1  # Minimum number of properties an edit must appear in
    splitter_params: Dict = field(default_factory=dict)
    test_datasets: List[str] = field(default_factory=list)
    save_models: bool = False
    models_dir: str = "models"
    embedder_type: str = 'chemprop'
    trainable_gnn: bool = False  # Whether to make GNN trainable (only for chemprop_dmpnn)
    gnn_device: str = 'auto'  # Device for GNN ('cpu', 'cuda', or 'auto' for auto-detect)
    include_cluster_analysis: bool = True
    n_clusters: int = 4
    include_edit_embedding_comparison: bool = True
    additional_test_files: Dict[str, str] = field(default_factory=dict)
