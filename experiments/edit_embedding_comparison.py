import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, List, Tuple
import torch


def compare_edit_embeddings(
    model,
    test_df: pd.DataFrame,
    mol_emb_a_test: np.ndarray,
    mol_emb_b_test: np.ndarray,
    task_names: List[str],
    device: str = 'cpu'
) -> Tuple[List[plt.Figure], pd.DataFrame]:

    model.model.eval()
    model.model.to(device)

    with torch.no_grad():
        reactant_tensor = torch.FloatTensor(mol_emb_a_test).to(device)
        product_tensor = torch.FloatTensor(mol_emb_b_test).to(device)

        baseline_edit = product_tensor - reactant_tensor

        trained_edit = model.model.trainable_edit_layer(
            reactant_tensor, product_tensor
        )

        mol_tensor = reactant_tensor

        baseline_input = torch.cat([mol_tensor, baseline_edit], dim=-1)
        trained_input = torch.cat([mol_tensor, trained_edit], dim=-1)

        baseline_preds = model.model.multi_task_network(baseline_input)
        trained_preds = model.model.multi_task_network(trained_input)

    num_tasks = len(task_names)
    delta_test = np.full((len(test_df), num_tasks), np.nan, dtype=np.float32)

    for i, task_name in enumerate(task_names):
        mask = test_df['property_name'] == task_name if 'property_name' in test_df.columns else np.ones(len(test_df), dtype=bool)
        if mask.sum() > 0:
            delta_test[mask, i] = test_df.loc[mask, 'delta'].values

    results = []

    for i, property_name in enumerate(task_names):
        true_values_all = delta_test[:, i]

        mask = ~np.isnan(true_values_all)

        if mask.sum() == 0:
            continue

        true_values = true_values_all[mask]
        baseline_pred = baseline_preds[property_name][mask].cpu().numpy()
        trained_pred = trained_preds[property_name][mask].cpu().numpy()

        baseline_r2 = r2_score(true_values, baseline_pred)
        trained_r2 = r2_score(true_values, trained_pred)

        baseline_mae = mean_absolute_error(true_values, baseline_pred)
        trained_mae = mean_absolute_error(true_values, trained_pred)

        results.append({
            'property': property_name,
            'n_samples': mask.sum(),
            'baseline_r2': baseline_r2,
            'trained_r2': trained_r2,
            'r2_improvement': trained_r2 - baseline_r2,
            'r2_improvement_pct': ((trained_r2 - baseline_r2) / abs(baseline_r2) * 100) if baseline_r2 != 0 else 0,
            'baseline_mae': baseline_mae,
            'trained_mae': trained_mae,
            'mae_improvement': baseline_mae - trained_mae,
            'mae_improvement_pct': ((baseline_mae - trained_mae) / baseline_mae * 100) if baseline_mae != 0 else 0
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_improvement', ascending=False)

    def clean_name(name):
        return name.replace('_', ' ').title()[:35]

    results_df['property_clean'] = results_df['property'].apply(clean_name)

    figures = []

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df['baseline_r2'], width,
                    label='Baseline (simple difference)', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, results_df['trained_r2'], width,
                    label='Trained (learned transformation)', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Property', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('R² Comparison: Trained vs Baseline Edit Embeddings\n(Higher is better)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['property_clean'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

    avg_r2_improvement = results_df['r2_improvement'].mean()
    summary_text = f"Average R² improvement: {avg_r2_improvement:+.4f}"
    ax.text(0.5, 0.02, summary_text, transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='black'))

    plt.tight_layout()
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(14, 8))

    results_sorted = results_df.sort_values('r2_improvement_pct', ascending=True)

    y_pos = np.arange(len(results_sorted))
    colors = ['lightblue' if x > 0 else 'red' for x in results_sorted['r2_improvement_pct']]

    bars = ax.barh(y_pos, results_sorted['r2_improvement_pct'],
                   color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['property_clean'], fontsize=11)
    ax.set_xlabel('R² Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('R² Relative Improvement by Property\nTrained vs Baseline Edit Embeddings',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, results_sorted['r2_improvement_pct'])):
        ax.text(val, i, f' {val:+.1f}%',
                va='center', fontsize=10,
                color='darkgreen' if val > 0 else 'darkred',
                fontweight='bold')

    avg_improvement_pct = results_sorted['r2_improvement_pct'].mean()
    summary_text = f"Average improvement: {avg_improvement_pct:+.1f}%"
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
            ha='right', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(14, 8))

    results_sorted = results_df.sort_values('mae_improvement_pct', ascending=True)

    y_pos = np.arange(len(results_sorted))
    colors = ['green' if x > 0 else 'red' for x in results_sorted['mae_improvement_pct']]

    bars = ax.barh(y_pos, results_sorted['mae_improvement_pct'],
                   color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['property_clean'], fontsize=11)
    ax.set_xlabel('MAE Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('MAE Relative Improvement by Property\nTrained vs Baseline Edit Embeddings',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, results_sorted['mae_improvement_pct'])):
        ax.text(val, i, f' {val:+.1f}%',
                va='center', fontsize=10,
                color='darkgreen' if val > 0 else 'darkred',
                fontweight='bold')

    avg_improvement_pct = results_sorted['mae_improvement_pct'].mean()
    summary_text = f"Average improvement: {avg_improvement_pct:+.1f}%"
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
            ha='right', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    figures.append(fig)

    return figures, results_df
