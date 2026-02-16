import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import ExperimentConfig
from data_loader import load_datasets
from model_factory import create_models
from trainer import train_all_models
from evaluator import evaluate_all_models, print_comparison_table
from report_generator import generate_report


def run_experiment(config: ExperimentConfig):
    print(f"Running experiment: {config.experiment_name}")

    train_data, test_datasets = load_datasets(config)

    # Create models - each method has its own embedder
    models = create_models(config, train_data, embedder=None)

    # Train all models (embeddings computed on-the-fly per method)
    trained_models = train_all_models(models, train_data, config)

    # Evaluate all models (embeddings computed on-the-fly per method)
    results = evaluate_all_models(trained_models, train_data, test_datasets, config)

    # Print comparison table to console for quick review
    print_comparison_table(results, metrics=config.metrics[:4] if hasattr(config, 'metrics') else None)

    report_path = generate_report(results, config, trained_models, {}, train_data)

    print(f"\nExperiment complete! Report saved to: {report_path}")
    return results, report_path
