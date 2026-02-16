import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter, defaultdict
from typing import Dict, Tuple
import torch


def perform_cluster_analysis(
    model,
    test_df: pd.DataFrame,
    mol_emb_a_test: np.ndarray,
    mol_emb_b_test: np.ndarray,
    n_clusters: int = 4,
    device: str = 'cpu'
) -> Tuple[plt.Figure, Dict]:

    edit_layer = model.model.trainable_edit_layer

    reactant_tensor = torch.FloatTensor(mol_emb_a_test).to(device)
    product_tensor = torch.FloatTensor(mol_emb_b_test).to(device)

    edit_layer.eval()
    with torch.no_grad():
        all_edit_emb = edit_layer(reactant_tensor, product_tensor).cpu().numpy()

    unique_edits = test_df['edit_smiles'].unique()
    edit_to_idx = {edit: test_df[test_df['edit_smiles'] == edit].index[0]
                   for edit in unique_edits}

    edit_embeddings = np.array([all_edit_emb[idx] for idx in edit_to_idx.values()])
    edit_names = list(edit_to_idx.keys())

    pca = PCA(n_components=min(50, edit_embeddings.shape[0]))
    edit_pca = pca.fit_transform(edit_embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(edit_pca[:, :10])

    cluster_centers = kmeans.cluster_centers_

    overall_edit_counts = Counter(test_df['edit_smiles'])

    cluster_edits = defaultdict(list)
    for idx, (edit_name, cluster_id) in enumerate(zip(edit_names, cluster_labels)):
        cluster_edits[cluster_id].append(edit_name)

    cluster_top_edits = {}
    for cluster_id in range(n_clusters):
        edits_in_cluster = cluster_edits[cluster_id]

        edit_counts_in_cluster = []
        for edit in edits_in_cluster:
            count = overall_edit_counts.get(edit, 0)
            edit_counts_in_cluster.append((edit, count))

        edit_counts_in_cluster.sort(key=lambda x: x[1], reverse=True)

        cluster_top_edits[cluster_id] = edit_counts_in_cluster[:5]

    fig, ax = plt.subplots(figsize=(18, 12))

    cluster_cmap = cm.get_cmap('Set2')

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(edit_pca[mask, 0], edit_pca[mask, 1],
                   c=[cluster_cmap(cluster_id)], alpha=0.6, s=60,
                   label=f'Cluster {cluster_id}',
                   edgecolors='white', linewidth=0.5)

    pca_vis = PCA(n_components=2)
    pca_vis.fit(edit_pca[:, :10])
    centers_pc = pca_vis.transform(cluster_centers)

    ax.scatter(centers_pc[:, 0], centers_pc[:, 1],
               c='red', marker='X', s=600, edgecolors='black',
               linewidth=3, label='Cluster Centers', zorder=10)

    for cluster_id, (cx, cy) in enumerate(centers_pc):
        ax.annotate(f'{cluster_id}', xy=(cx, cy), fontsize=16, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='red',
                            edgecolor='black', linewidth=2),
                   zorder=11)

    label_positions = {
        0: ('left', 'center', 1.5, 0),
        1: ('right', 'bottom', -0.5, 1.2),
        2: ('right', 'center', -1.5, 0),
        3: ('center', 'bottom', 0, 1.2)
    }

    x_range = edit_pca[:, 0].max() - edit_pca[:, 0].min()
    y_range = edit_pca[:, 1].max() - edit_pca[:, 1].min()

    for cluster_id in range(min(n_clusters, 4)):
        cx, cy = centers_pc[cluster_id]
        ha, va, x_offset_factor, y_offset_factor = label_positions.get(cluster_id, ('center', 'bottom', 0, 1.2))

        label_x = cx + (x_offset_factor * x_range * 0.15)
        label_y = cy + (y_offset_factor * y_range * 0.15)

        top_edits = cluster_top_edits[cluster_id]
        label_lines = [f"Cluster {cluster_id} - Top 5 Edits:"]
        for rank, (edit_name, count) in enumerate(top_edits, 1):
            display_name = edit_name if len(edit_name) <= 35 else edit_name[:32] + "..."
            label_lines.append(f"{rank}. {display_name} ({count:,})")

        label_text = "\n".join(label_lines)

        bbox_props = dict(
            boxstyle='round,pad=0.9',
            facecolor=cluster_cmap(cluster_id),
            alpha=0.85,
            edgecolor='black',
            linewidth=2.25
        )

        ax.annotate(label_text, xy=(cx, cy), xytext=(label_x, label_y),
                   fontsize=13.5, ha=ha, va=va,
                   bbox=bbox_props,
                   arrowprops=dict(arrowstyle='->', lw=2.25, color='black'),
                   zorder=12, family='monospace')

    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_title(f'K-Means Clustering of Edit Embeddings with Top 5 Edits per Cluster ({n_clusters} clusters)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    cluster_stats = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        size = mask.sum()
        pct = 100 * size / len(cluster_labels)
        cluster_stats[cluster_id] = {
            'size': int(size),
            'pct': float(pct),
            'top_edits': cluster_top_edits[cluster_id]
        }

    return fig, cluster_stats
