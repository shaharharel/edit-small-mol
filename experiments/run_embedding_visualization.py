#!/usr/bin/env python3
"""
Embedding space visualization for edit effect framework.

Generates PCA, t-SNE, and UMAP projections comparing base molecule embeddings
vs edit embeddings (emb_B - emb_A), colored by:
  1. Effect size (delta pIC50)
  2. Target (biological context)
  3. Within vs cross-assay
  4. Effect magnitude (|delta|)

Uses cached ChemProp D-MPNN embeddings (the Phase 1 winner).

Usage:
    conda run -n quris python -u experiments/run_embedding_visualization.py
"""

import sys
import gc
import json
import base64
import warnings
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"
CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

SHARED_PAIRS_FILE = DATA_DIR / "shared_pairs_deduped.csv"
EMBEDDER = "chemprop-dmpnn"

# Visualization parameters
MAX_PAIRS = 15000      # Subsample for visualization (t-SNE/UMAP are slow)
MAX_TARGETS = 20       # Top targets to color
SEED = 42


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def load_data_and_embeddings():
    """Load shared pairs + cached embeddings."""
    print("Loading data...")
    df = pd.read_csv(SHARED_PAIRS_FILE)
    # Filter to real MMPs
    if "mol_a_id" in df.columns and "mol_b_id" in df.columns:
        df = df[df["mol_a_id"] != df["mol_b_id"]].copy()

    print(f"  {len(df):,} pairs")

    # Load cached embeddings
    cache_file = CACHE_DIR / f"{EMBEDDER}.npz"
    print(f"  Loading {EMBEDDER} embeddings from {cache_file.name}...")
    data = np.load(cache_file, allow_pickle=True)
    cached_smiles = data['smiles'].tolist()
    cached_embs = data['embeddings']
    emb_dim = int(data['emb_dim'])
    emb_dict = {smi: cached_embs[i] for i, smi in enumerate(cached_smiles)}
    print(f"  {len(emb_dict):,} molecules in cache (dim={emb_dim})")

    return df, emb_dict, emb_dim


def prepare_visualization_data(df, emb_dict):
    """Subsample and build embedding matrices."""
    # Filter pairs where both molecules have embeddings
    valid = df['mol_a'].isin(emb_dict) & df['mol_b'].isin(emb_dict)
    df = df[valid].copy()
    print(f"  Valid pairs: {len(df):,}")

    # Subsample stratified by target
    if len(df) > MAX_PAIRS:
        rng = np.random.RandomState(SEED)
        # Sample proportionally from top targets
        target_counts = df['target_chembl_id'].value_counts()
        top_targets = target_counts.head(MAX_TARGETS).index.tolist()

        # Take pairs from top targets + random sample from rest
        top_df = df[df['target_chembl_id'].isin(top_targets)]
        rest_df = df[~df['target_chembl_id'].isin(top_targets)]

        n_top = min(len(top_df), MAX_PAIRS * 2 // 3)
        n_rest = min(len(rest_df), MAX_PAIRS - n_top)

        sampled = pd.concat([
            top_df.sample(n=n_top, random_state=rng),
            rest_df.sample(n=n_rest, random_state=rng) if n_rest > 0 else rest_df.head(0),
        ], ignore_index=True)
        df = sampled
        print(f"  Subsampled to {len(df):,} pairs")

    # Build matrices
    emb_a = np.array([emb_dict[s] for s in df['mol_a']])
    emb_b = np.array([emb_dict[s] for s in df['mol_b']])
    edit_emb = emb_b - emb_a
    delta = df['delta'].values
    targets = df['target_chembl_id'].values
    is_within = df['is_within_assay'].values if 'is_within_assay' in df.columns else None

    return emb_a, emb_b, edit_emb, delta, targets, is_within


def run_projections(emb_a, edit_emb):
    """Run PCA, t-SNE on both base and edit embeddings."""
    results = {}

    # PCA
    print("  PCA...")
    pca = PCA(n_components=2, random_state=SEED)
    results['pca_base'] = pca.fit_transform(emb_a)
    results['pca_base_var'] = pca.explained_variance_ratio_
    pca2 = PCA(n_components=2, random_state=SEED)
    results['pca_edit'] = pca2.fit_transform(edit_emb)
    results['pca_edit_var'] = pca2.explained_variance_ratio_

    # t-SNE
    print("  t-SNE (this takes a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    results['tsne_base'] = tsne.fit_transform(emb_a)
    tsne2 = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    results['tsne_edit'] = tsne2.fit_transform(edit_emb)

    # Try UMAP
    try:
        import umap
        print("  UMAP...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=SEED)
        results['umap_base'] = reducer.fit_transform(emb_a)
        reducer2 = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=SEED)
        results['umap_edit'] = reducer2.fit_transform(edit_emb)
        results['has_umap'] = True
    except ImportError:
        print("  UMAP not available (pip install umap-learn)")
        results['has_umap'] = False

    return results


def make_comparison_figure(base_2d, edit_2d, delta, title_prefix, base_label, edit_label):
    """Side-by-side comparison of base vs edit embeddings colored by delta."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax = np.percentile(np.abs(delta), 95)

    for ax, coords, title in [(axes[0], base_2d, base_label), (axes[1], edit_2d, edit_label)]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=delta, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, s=2, alpha=0.4, rasterized=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        plt.colorbar(sc, ax=ax, label='ΔpIC50', shrink=0.8)

    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def make_target_figure(coords_2d, targets, title):
    """Embeddings colored by biological target."""
    fig, ax = plt.subplots(figsize=(10, 8))

    target_counts = pd.Series(targets).value_counts()
    top = target_counts.head(MAX_TARGETS).index.tolist()
    cmap = plt.cm.get_cmap('tab20', min(len(top), 20))

    # Plot "other" first
    other_mask = ~np.isin(targets, top)
    if other_mask.sum() > 0:
        ax.scatter(coords_2d[other_mask, 0], coords_2d[other_mask, 1],
                  c='#e0e0e0', s=2, alpha=0.2, rasterized=True, label='Other')

    # Plot top targets
    for i, t in enumerate(top):
        mask = targets == t
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=[cmap(i % 20)], s=3, alpha=0.5, rasterized=True,
                  label=f'{t} (n={mask.sum()})')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(loc='best', fontsize=6, ncol=2, markerscale=4)
    plt.tight_layout()
    return fig


def make_assay_figure(coords_2d, is_within, delta, title):
    """Embeddings colored by within vs cross-assay."""
    if is_within is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by assay type
    w_mask = is_within.astype(bool)
    c_mask = ~w_mask
    axes[0].scatter(coords_2d[c_mask, 0], coords_2d[c_mask, 1],
                   c='#e74c3c', s=2, alpha=0.3, label=f'Cross-assay (n={c_mask.sum():,})', rasterized=True)
    axes[0].scatter(coords_2d[w_mask, 0], coords_2d[w_mask, 1],
                   c='#27ae60', s=2, alpha=0.3, label=f'Within-assay (n={w_mask.sum():,})', rasterized=True)
    axes[0].set_title('By Assay Context')
    axes[0].legend(markerscale=5)
    axes[0].set_xlabel('Dim 1')
    axes[0].set_ylabel('Dim 2')

    # Right: spread comparison
    abs_delta = np.abs(delta)
    sc = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], c=abs_delta, cmap='viridis',
                        vmin=0, vmax=np.percentile(abs_delta, 95),
                        s=2, alpha=0.4, rasterized=True)
    axes[1].set_title('By Effect Magnitude |ΔpIC50|')
    axes[1].set_xlabel('Dim 1')
    axes[1].set_ylabel('Dim 2')
    plt.colorbar(sc, ax=axes[1], label='|ΔpIC50|', shrink=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def generate_html_report(figures):
    """Generate self-contained HTML with embedded figures."""
    h = []
    h.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    h.append("<title>Edit Embedding Visualization</title>")
    h.append("<style>")
    h.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
               max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2c3e50; margin-top: 40px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }
        .note { background: #f0f7ff; border-left: 4px solid #3498db; padding: 12px 16px; margin: 15px 0; }
        .insight { background: #e8f8e8; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 15px 0; }
        img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        .fig-caption { font-size: 0.9em; color: #555; margin-bottom: 20px; }
    """)
    h.append("</style></head><body>")
    h.append(f"<h1>Edit Embedding Space Visualization</h1>")
    h.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"Embedder: {EMBEDDER} | Pairs: {MAX_PAIRS:,} subsample</p>")

    h.append("<div class='note'>")
    h.append("<strong>Key question</strong>: Do edit embeddings (emb_B − emb_A) organize chemical transformations "
             "into a meaningful geometric structure that is <em>not</em> present in the base molecule embedding space?")
    h.append("</div>")

    for section_title, section_figures in figures:
        h.append(f"<h2>{section_title}</h2>")
        for fig_title, fig_b64, caption in section_figures:
            h.append(f"<h3>{fig_title}</h3>")
            h.append(f"<img src='data:image/png;base64,{fig_b64}' alt='{fig_title}'>")
            h.append(f"<p class='fig-caption'>{caption}</p>")

    h.append("</body></html>")

    report_path = RESULTS_DIR / "embedding_visualization.html"
    report_path.write_text("\n".join(h))
    print(f"\nReport saved to: {report_path}")
    return str(report_path)


def main():
    print(f"{'='*60}")
    print(f"Embedding Space Visualization")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Load data
    df, emb_dict, emb_dim = load_data_and_embeddings()
    emb_a, emb_b, edit_emb, delta, targets, is_within = prepare_visualization_data(df, emb_dict)
    del df; gc.collect()
    print(f"  Final: {len(delta):,} pairs, emb_dim={emb_dim}")

    # Run projections
    print("\nRunning dimensionality reduction...")
    proj = run_projections(emb_a, edit_emb)

    # Generate figures
    print("\nGenerating figures...")
    all_figures = []

    # PCA section
    pca_figs = []
    fig = make_comparison_figure(
        proj['pca_base'], proj['pca_edit'], delta,
        'PCA: Base Molecule vs Edit Embeddings',
        f"Base emb (PC1={proj['pca_base_var'][0]:.1%}, PC2={proj['pca_base_var'][1]:.1%})",
        f"Edit emb (PC1={proj['pca_edit_var'][0]:.1%}, PC2={proj['pca_edit_var'][1]:.1%})"
    )
    pca_figs.append(("Effect Size Comparison", fig_to_base64(fig),
                     "Base molecule embeddings encode molecular identity; edit embeddings (emb_B − emb_A) "
                     "encode the chemical transformation. If edit space shows more structure correlated with "
                     "delta, it suggests edits carry distinct SAR information."))

    fig = make_target_figure(proj['pca_edit'], targets, 'PCA: Edit Embeddings by Target')
    pca_figs.append(("Target Clustering", fig_to_base64(fig),
                     "Target-specific clustering in edit space would indicate that similar chemical "
                     "transformations have target-dependent effects — the same edit can have different "
                     "meanings in different biological contexts."))

    if is_within is not None:
        fig = make_assay_figure(proj['pca_edit'], is_within, delta, 'PCA: Assay Context & Effect Magnitude')
        if fig:
            pca_figs.append(("Assay Context", fig_to_base64(fig),
                             "Within-assay and cross-assay pairs in edit embedding space. "
                             "Cross-assay pairs may show higher spread due to lab-to-lab measurement noise."))

    all_figures.append(("PCA Projections", pca_figs))

    # t-SNE section
    tsne_figs = []
    fig = make_comparison_figure(
        proj['tsne_base'], proj['tsne_edit'], delta,
        't-SNE: Base Molecule vs Edit Embeddings',
        'Base molecule embeddings', 'Edit embeddings (emb_B − emb_A)'
    )
    tsne_figs.append(("Effect Size Comparison", fig_to_base64(fig),
                      "t-SNE preserves local neighborhood structure. Clusters in edit space "
                      "correspond to groups of transformations with similar effect sizes."))

    fig = make_target_figure(proj['tsne_edit'], targets, 't-SNE: Edit Embeddings by Target')
    tsne_figs.append(("Target Clustering", fig_to_base64(fig),
                      "t-SNE projection colored by biological target."))

    all_figures.append(("t-SNE Projections", tsne_figs))

    # UMAP section
    if proj.get('has_umap'):
        umap_figs = []
        fig = make_comparison_figure(
            proj['umap_base'], proj['umap_edit'], delta,
            'UMAP: Base Molecule vs Edit Embeddings',
            'Base molecule embeddings', 'Edit embeddings (emb_B − emb_A)'
        )
        umap_figs.append(("Effect Size Comparison", fig_to_base64(fig),
                          "UMAP balances local and global structure. It typically reveals "
                          "more interpretable clusters than t-SNE."))

        fig = make_target_figure(proj['umap_edit'], targets, 'UMAP: Edit Embeddings by Target')
        umap_figs.append(("Target Clustering", fig_to_base64(fig),
                          "UMAP projection colored by biological target."))

        if is_within is not None:
            fig = make_assay_figure(proj['umap_edit'], is_within, delta, 'UMAP: Assay Context & Effect Magnitude')
            if fig:
                umap_figs.append(("Assay Context", fig_to_base64(fig),
                                  "Within-assay vs cross-assay in UMAP edit space."))

        all_figures.append(("UMAP Projections", umap_figs))

    # Generate HTML report
    report = generate_html_report(all_figures)

    print(f"\n{'='*60}")
    print(f"COMPLETE — {report}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
