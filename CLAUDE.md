# CLAUDE.md - Project Guidelines for AI Assistants

## Environment Setup

**Use the `quris` conda environment for all Python operations.**
**Use CPU for all transformer-based experiments** (MPS crashes with ChemBERTa after prolonged use).

---

## Causal Edit Effect Framework

The central idea is to explicitly model how molecular systems respond to **interventions**: given a baseline molecule and a defined edit (matched molecular pair transformation), predict the resulting change in a property of interest.

### Edit Embeddings

Edit embeddings encode edits as **structured, context-aware interventions**, decoupling:
- The **representation of the molecule** (the background system)
- The **representation of the chemical transformation** applied to it

### Primary Validation: Edit Effect vs. Subtraction Baseline

| Approach | Method | Formula |
|----------|--------|---------|
| **Subtraction baseline** | Predict property independently, then subtract | `F(mol_after) - F(mol_before) = Δproperty` |
| **Edit effect framework** | Learn directly from supervised delta signal | `F(mol_before, edit) → Δproperty` |

### Validation Criteria

1. Edit effect **outperforms subtraction baseline** across scenarios
2. **Generalization** across scaffolds, targets, and challenging splits
3. **Noise robustness**: leveraging within-assay pairs to avoid cross-lab noise
4. **Data efficiency**: better performance in low-data regimes

---

## Current Experiment Plan (March 2026)

### Paper: "Edit Effect Framework for Noise-Robust Bioactivity Prediction"

**Canonical experiment script**: `experiments/run_paper_evaluation.py`

#### Phase 1: Embedder Selection (DONE — shared pairs, 1.7M)
- **Winner: Morgan FP (via ChemProp featurizer)** (2048-dim) — MAE=0.631
- Note: "ChemProp D-MPNN" is actually Morgan FP, NOT a pretrained GNN
- Morgan FP (RDKit) very close (MAE=0.637), all pretrained models lagged
- CheMeleon (pretrained D-MPNN): MAE=0.659, best pretrained but still worse than fingerprints
- MoLFormer-XL: MAE=0.691, Uni-Mol v1: MAE=0.690 — last place

#### Phase 2: Architecture Comparison (DONE — 8 architectures, within-assay)
- **Winner: FiLMDelta** (FiLM-conditioned f(B|δ)−f(A|δ)) — 7.8% MAE reduction over Subtraction
- FiLMDelta MAE=0.616±0.022, EditDiff MAE=0.631±0.016, Subtraction MAE=0.668±0.019
- Attention architectures (GatedCrossAttn, AttnThenFiLM) underperformed even vs Subtraction

#### Phase 3: Generalization (DONE — 7 splits)
- Splits: assay_within, assay_cross, assay_mixed, scaffold, random, cross-target, strict_scaffold, pair_random
- Methods: FiLMDelta, EditDiff, DeepDelta, Subtraction (all with Morgan FP 2048d)
- **CRITICAL**: Random split has 71% exact pair duplicates — useless for generalization claims
- **CRITICAL**: Old scaffold split only uses mol_a scaffolds — asymmetric, misleading
- Assay-within is the primary evaluation (42% both-mols-new, genuine generalization)

#### Noise Robustness Analysis (DONE)
- **Realistic noise tiers** (40 targets, noise ratios 0.35x–3.3x): FiLMDelta wins 40/40 (69.6% avg advantage)
  - Spearman/Pearson/R² gaps all grow significantly with noise (p<0.001)
  - FiLMDelta trains on within-assay pairs; Subtraction trains on all data (realistic usage)
  - Script: `experiments/run_fair_noise_tiers.py`
- **Controlled noise injection** (σ=0→1.5): FiLMDelta degrades 3.2% vs Subtraction 12.3%
  - Script: `experiments/run_noise_injection.py`
- Advantage decomposes: ~8% architecture (Phase 2) + ~62% data curation (within-assay pairs)

#### Results location
- `results/paper_evaluation/all_results.json` — Phase 1-3 results
- `results/paper_evaluation/fair_noise_tiers_results.json` — per-target noise tier results
- `results/paper_evaluation/noise_injection_results.json` — controlled noise injection
- `results/paper_evaluation/evaluation_report.html` — unified HTML report

### Metrics Policy
- **Primary**: MAE (lower is better)
- **Secondary**: Spearman rank correlation (higher is better)
- **Also report**: Pearson r, R² (per-target averaged, NOT pooled multi-target)
- **Never report**: pooled multi-target R² (misleading artifact of prediction shrinkage)

---

## Project Overview

edit-small-mol is a framework for predicting how matched molecular pair (MMP) transformations affect molecular properties. It extracts molecular pairs from ChEMBL, computes edit embeddings, and trains predictors to estimate property changes from chemical transformations.

---

## Code Organization Rules

### Strict Source Code Policy

**All code must be written within the project's source structure:**

```
src/           # Core library code (data loaders, embedders, models, utils)
experiments/   # Experiment runners and configurations
scripts/       # Data preprocessing and extraction scripts
tests/         # Unit tests
```

**Do NOT write code outside these directories.**

### Requesting Exceptions

If a one-off script is necessary:
1. **Ask for permission first**
2. **Provide rationale** and propose integration path

---

## Project Structure

```
src/
├── data/                  # Data loading and extraction
│   ├── base_extractor.py  # Abstract base for data extractors
│   ├── chembl_extractor.py # ChEMBL data extraction pipeline
│   ├── mmp_long_format.py # MMP extraction in long format
│   ├── mmp_atom_mapping_fast.py # Fast atom mapping (84x speedup)
│   ├── scalable_mmp.py    # Scalable MMP extraction
│   └── utils/
│       └── chemistry.py   # RDKit utilities + compute_edit_features(28d)
├── embedding/             # Molecule and edit embedders
│   ├── base.py            # MoleculeEmbedder abstract base
│   ├── fingerprints.py    # Morgan, RDKit, MACCS, Atom Pair
│   ├── chemberta.py       # ChemBERTa-2 (MLM/MTR, 77M params)
│   ├── chemprop.py        # ChemProp D-MPNN
│   ├── molformer.py       # MoLFormer-XL (needs transformers fix)
│   ├── edit_embedder.py   # Simple edit differences
│   └── trainable_edit_embedder.py # Learnable edit embeddings
├── models/                # Neural network architectures
│   ├── predictors/        # Edit effect and property predictors
│   └── trainer.py         # Training utilities
└── utils/
    ├── splits.py          # Random, Scaffold, Target, Butina, Assay, FewShot, Core
    └── metrics.py         # Regression and ranking metrics

experiments/
├── run_paper_evaluation.py     # CANONICAL: Phase 1→2→3 pipeline
├── generate_report.py          # HTML report generator
├── run_fair_noise_tiers.py     # Realistic noise tier experiment (40 targets)
├── run_noise_injection.py      # Controlled noise injection experiment
├── run_data_efficiency.py      # Learning curve analysis
├── run_embedding_visualization.py # PCA/t-SNE/UMAP visualization
├── run_noise_performance_analysis.py # 2×2 factorial noise analysis
└── model_factory.py            # Embedder/model creation

scripts/extraction/             # ChEMBL and MMP extraction scripts
data/                           # Raw and processed datasets (gitignored)
tests/                          # Unit tests
```

---

## Key Data

- **Canonical dataset (shared pairs)**: `data/overlapping_assays/extracted/shared_pairs_deduped.csv` — 1.7M pairs (856K within + 844K cross), 88K molecules, 751 targets
- **Full dataset**: `data/overlapping_assays/extracted/overlapping_assay_pairs_minimal_mmp.csv` — 5M pairs, 758 targets, 105K molecules
- **50K subsample**: `data/overlapping_assays/extracted/overlapping_assay_pairs_minimal_mmp_50k.csv`
- **Embedding cache**: `data/embedding_cache/*.npz` (morgan, chemberta2-mlm, chemberta2-mtr, chemprop-dmpnn)
- **Columns**: mol_a, mol_b, mol_a_id, mol_b_id, edit_smiles, delta, is_within_assay, value_a, value_b, target_chembl_id, assay_id_a, assay_id_b
- **Shared pairs**: pairs appearing in BOTH within-assay and cross-assay contexts — enables perfectly matched noise comparison

---

## Key Patterns

- **Embedders**: Inherit from `MoleculeEmbedder` base class in `src/embedding/base.py`
- **Splits**: Use `get_splitter()` factory in `src/utils/splits.py`
- **Edit modes**: Full molecule difference (context-aware) and edit fragment difference (scaffold-independent)
