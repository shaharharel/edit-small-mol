#!/usr/bin/env python3
"""
Generate pharmacophore-style SHAP atom maps for the ZAP70 (ZAP70) case study.

For 4 selected molecules, trains an XGBoost model on Morgan FP (ECFP4, 2048-bit)
to predict pIC50, computes SHAP values per bit, maps bit-level SHAP contributions
back to atoms via RDKit bitInfo, and renders molecules with atoms colored by their
aggregated SHAP contribution (red = positive/activating, blue = negative/deactivating).

Output: results/paper_evaluation/figures/zap70_pharmacophore_shap.png

Usage:
    conda run -n quris python -u experiments/generate_pharmacophore_map.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from io import BytesIO
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_FILE = FIGURES_DIR / "zap70_pharmacophore_shap.png"
ZAP70_ID = "CHEMBL2803"


def load_zap70_molecules():
    """Load ZAP70 molecule-level data (averaged across assays)."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  Loaded {len(mol_data)} ZAP70 molecules, "
          f"pIC50 range: {mol_data['pIC50'].min():.2f} - {mol_data['pIC50'].max():.2f}")
    return mol_data


def compute_morgan_fp(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints as numpy array."""
    from rdkit.Chem import DataStructs
    X = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            X.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    return np.array(X, dtype=np.float32)


def get_atom_shap_contributions(smiles, shap_values_for_mol, radius=2, n_bits=2048):
    """Map SHAP bit contributions back to individual atoms.

    For each Morgan FP bit that is 'on' for this molecule, we know which atoms
    contributed to it (via bitInfo). We distribute the SHAP value for that bit
    equally among its contributing center atoms.

    Returns dict: atom_idx -> aggregated SHAP contribution.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

    atom_contributions = np.zeros(mol.GetNumAtoms(), dtype=np.float64)

    for bit_idx, atom_envs in bit_info.items():
        shap_val = shap_values_for_mol[bit_idx]
        if abs(shap_val) < 1e-10:
            continue
        # Each atom_env is (center_atom, radius)
        # Distribute SHAP value among all center atoms for this bit
        center_atoms = set(ae[0] for ae in atom_envs)
        contribution = shap_val / len(center_atoms)
        for atom_idx in center_atoms:
            atom_contributions[atom_idx] += contribution

    return atom_contributions


def render_molecule_with_shap(smiles, atom_contributions, title="", mol_size=(450, 350)):
    """Render a single molecule with atoms colored by SHAP contribution.

    Red = positive SHAP (increases predicted pIC50, favorable for potency)
    Blue = negative SHAP (decreases predicted pIC50, unfavorable)
    White = near zero contribution
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)

    n_atoms = mol.GetNumAtoms()

    # Normalize contributions for color mapping
    max_abs = max(np.max(np.abs(atom_contributions)), 1e-6)
    # Clip to symmetric range for balanced coloring
    vmax = min(max_abs, np.percentile(np.abs(atom_contributions[atom_contributions != 0]), 95) * 1.5) if np.any(atom_contributions != 0) else max_abs
    vmax = max(vmax, 0.01)

    # Create atom colors: blue-white-red diverging colormap
    cmap = plt.cm.RdBu_r  # Red for positive (good), Blue for negative (bad)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    atom_colors = {}
    atom_radii = {}
    for i in range(n_atoms):
        val = atom_contributions[i]
        rgba = cmap(norm(val))
        atom_colors[i] = (rgba[0], rgba[1], rgba[2])
        # Radius proportional to |SHAP|
        atom_radii[i] = 0.25 + 0.25 * min(abs(val) / vmax, 1.0)

    # Use RDKit drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.padding = 0.15
    opts.bondLineWidth = 2.0
    opts.multipleBondOffset = 0.15

    # Highlight all atoms with their SHAP colors
    highlight_atoms = list(range(n_atoms))
    highlight_bonds = []
    highlight_atom_colors = atom_colors
    highlight_bond_colors = {}
    highlight_atom_radii = atom_radii

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=highlight_atom_colors,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_atom_radii,
    )
    drawer.FinishDrawing()

    # Convert to PIL Image
    png_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(png_data))
    return img, vmax


def select_interesting_molecules(mol_data, shap_values, X_fp):
    """Select 4 molecules with interesting SHAP patterns:
    1. Most potent molecule
    2. Least potent molecule
    3. Most positive total SHAP (model finds most evidence for potency)
    4. Most negative total SHAP (model finds most evidence against potency)
    """
    y = mol_data["pIC50"].values
    total_shap = shap_values.sum(axis=1)

    most_potent_idx = np.argmax(y)
    least_potent_idx = np.argmin(y)
    most_pos_shap_idx = np.argmax(total_shap)
    most_neg_shap_idx = np.argmin(total_shap)

    # Avoid duplicates — replace with next best if needed
    selected = []
    labels = []
    candidates = [
        (most_potent_idx, "Most potent"),
        (least_potent_idx, "Least potent"),
        (most_pos_shap_idx, "Highest SHAP sum"),
        (most_neg_shap_idx, "Lowest SHAP sum"),
    ]

    # If there are duplicates, pick molecules with high SHAP variance instead
    used_indices = set()
    for idx, label in candidates:
        if idx not in used_indices:
            selected.append(idx)
            labels.append(label)
            used_indices.add(idx)

    # Fill remaining slots with high-variance SHAP molecules (interesting patterns)
    if len(selected) < 4:
        atom_shap_variance = []
        for i in range(len(mol_data)):
            smi = mol_data.iloc[i]["smiles"]
            ac = get_atom_shap_contributions(smi, shap_values[i])
            atom_shap_variance.append(np.var(ac))
        var_ranking = np.argsort(atom_shap_variance)[::-1]
        for idx in var_ranking:
            if idx not in used_indices:
                selected.append(idx)
                labels.append(f"High SHAP contrast (pIC50={y[idx]:.1f})")
                used_indices.add(idx)
                if len(selected) >= 4:
                    break

    return selected, labels


def main():
    print("=" * 70)
    print("ZAP70 Pharmacophore SHAP Map")
    print("=" * 70)

    # Load data
    mol_data = load_zap70_molecules()
    all_smiles = mol_data["smiles"].tolist()
    y_all = mol_data["pIC50"].values.astype(np.float32)

    # Compute fingerprints
    print("  Computing Morgan FP (ECFP4, 2048-bit)...")
    X_fp = compute_morgan_fp(all_smiles, radius=2, n_bits=2048)

    # Load v3 optimized XGBoost params
    v3_file = RESULTS_DIR / "zap70_v3_results.json"
    xgb_params = {}
    if v3_file.exists():
        with open(v3_file) as f:
            v3 = json.load(f)
        xgb_params = v3.get("phase_4", {}).get("xgboost_optimized", {}).get("best_params", {})
        print(f"  Loaded v3 XGBoost params (n_estimators={xgb_params.get('n_estimators', '?')})")

    # Train XGBoost on all data
    print("  Training XGBoost...")
    import xgboost as xgb
    model = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=8)
    model.fit(X_fp, y_all, verbose=False)

    # Compute SHAP values
    print("  Computing SHAP values...")
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fp)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Select 4 interesting molecules
    print("  Selecting molecules...")
    selected_indices, selected_labels = select_interesting_molecules(
        mol_data, shap_values, X_fp
    )

    for idx, label in zip(selected_indices, selected_labels):
        smi = all_smiles[idx]
        pic50 = y_all[idx]
        pred = model.predict(X_fp[idx:idx+1])[0]
        print(f"    {label}: pIC50={pic50:.2f} (pred={pred:.2f}), SMILES={smi[:60]}...")

    # Generate per-molecule atom-level SHAP maps
    print("  Rendering molecules with SHAP atom coloring...")
    mol_images = []
    mol_titles = []
    vmaxes = []

    for idx, label in zip(selected_indices, selected_labels):
        smi = all_smiles[idx]
        pic50 = y_all[idx]
        pred = model.predict(X_fp[idx:idx+1])[0]
        atom_contribs = get_atom_shap_contributions(smi, shap_values[idx])
        img, vmax = render_molecule_with_shap(smi, atom_contribs, mol_size=(500, 400))
        mol_images.append(img)
        mol_titles.append(f"{label}\npIC50={pic50:.2f} (pred={pred:.2f})")
        vmaxes.append(vmax)

    # Compose final figure: 2x2 grid with shared colorbar
    print("  Composing publication figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Atom-Level SHAP Contributions to Predicted pIC50\n"
        "ZAP70 (CHEMBL2803) — XGBoost on Morgan FP (ECFP4, 2048-bit)",
        fontsize=14, fontweight='bold', y=0.98
    )

    global_vmax = max(vmaxes)

    for i, (ax, img, title) in enumerate(zip(axes.flat, mol_images, mol_titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis('off')

    # Add colorbar
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-global_vmax, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Position colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('SHAP contribution to predicted pIC50', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Add legend
    legend_elements = [
        Patch(facecolor='#d73027', alpha=0.7, label='Positive (increases potency)'),
        Patch(facecolor='white', edgecolor='gray', label='Neutral'),
        Patch(facecolor='#4575b4', alpha=0.7, label='Negative (decreases potency)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.04),
        fontsize=9,
        framealpha=0.9,
        title="Atom SHAP",
        title_fontsize=10,
    )

    plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.15, wspace=0.05)

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
