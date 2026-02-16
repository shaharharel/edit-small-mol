# CLAUDE.md - Project Guidelines for AI Assistants

## Environment Setup

**Use the `quris` conda environment for all Python operations.**

---

## Causal Edit Effect Framework

The central idea of the causal edit effect framework is to explicitly model how molecular systems respond to **interventions**, rather than only learning correlations over static inputs. In small molecule optimization, the question of interest is not "what is the property of this molecule," but "what will happen if I change this part of it."

We formulate prediction as an **edit effect problem**: given a baseline molecule and a defined edit (matched molecular pair transformation), predict the resulting change in a property of interest.

### Edit Embeddings

Edit embeddings serve as the representation layer that makes this possible: they encode edits as **structured, context-aware interventions**, capturing how local chemical changes propagate through molecular structure. This abstraction decouples:
- The **representation of the molecule** (the background system)
- The **representation of the chemical transformation** applied to it

This alignment with causal reasoning makes the learning problem closer to how medicinal chemistry optimization is performed in practice.

### Advantages Over Direct Prediction

Compared to direct prediction approaches that map molecules to properties in a single step, the edit effect framework introduces **strong inductive structure**:

- **Information sharing**: Learn across many related perturbations
- **Consistency enforcement**: Between base predictions and predicted effects
- **Compositional reasoning**: Over multiple edits

This leads to:
- Improved **data efficiency**
- Better **generalization** to unseen transformations
- Clearer **interpretability**: similar edits have similar representations and effects, even across different molecular contexts

### Primary Validation: Edit Effect vs. Subtraction Baseline

**The most critical validation** is demonstrating that the edit effect framework significantly outperforms the common subtraction approach:

| Approach | Method | Formula |
|----------|--------|---------|
| **Subtraction baseline** | Predict property independently for each molecule, then subtract | `F(mol_after) - F(mol_before) = Δproperty` |
| **Edit effect framework** | Learn directly from supervised delta signal with edit embeddings | `F(mol_before, edit) → Δproperty` |

We must show that learning with the supervised delta signal and specialized edit embedding modeling outperforms independent property prediction followed by subtraction.

### Additional Validation Criteria

Beyond raw predictive performance, we should demonstrate:
1. Edit embeddings organize interventions into **meaningful geometric structure**
2. Representations **transfer across scaffolds and datasets**
3. Accurate prediction in **low-data regimes**
4. Generalization to **out-of-distribution** edits
5. **Chemically sensible clustering** of similar transformations

Together, these establish edit effect prediction with edit embeddings as a principled, causal alternative to direct property prediction.

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

**Do NOT write code outside these directories** (e.g., standalone scripts in root, temporary notebooks, one-off analysis files).

### Requesting Exceptions

If you believe a one-off script or analysis is necessary:

1. **Ask for permission first** - Do not create the file without explicit approval
2. **Provide rationale** - Explain why this cannot fit within the existing structure
3. **Propose integration path** - Describe how/if this will be incorporated into the framework later, or why it should remain a one-off
4. **Suggest location** - If approved, propose the most appropriate location within the project structure

---

## Project Structure

```
src/
├── data/                  # Data loading and extraction
│   ├── base_extractor.py  # Abstract base for data extractors
│   ├── chembl_extractor.py # ChEMBL data extraction pipeline
│   ├── mmp_long_format.py # MMP extraction in long format
│   ├── mmp_atom_mapping.py # Atom mapping for MMPs
│   ├── mmp_atom_mapping_fast.py # Fast atom mapping (84x speedup)
│   ├── mmp_parser.py      # Parse MMP structural columns
│   ├── parallel_extraction.py # Parallel extraction utilities
│   ├── scalable_mmp.py    # Scalable MMP extraction
│   ├── structured_dataset.py # Structured dataset wrapper
│   └── utils/
│       └── chemistry.py   # RDKit-based chemistry utilities
├── embedding/             # Molecule and edit embedders
│   ├── base.py            # MoleculeEmbedder abstract base
│   ├── fingerprints.py    # Morgan, RDKit, MACCS, Atom Pair
│   ├── edit_embedder.py   # Simple edit differences
│   ├── trainable_edit_embedder.py # Learnable edit embeddings
│   ├── structured_edit_base.py # Structured embedder base
│   ├── structured_edit_embedder.py # Rich structured edit representations
│   ├── chemberta.py       # ChemBERTa transformer embeddings
│   ├── chemprop.py        # ChemProp D-MPNN embeddings
│   ├── graphormer.py      # Graphormer graph embeddings
│   ├── molfm.py           # MolFM foundation model
│   ├── unimol.py          # UniMol embeddings
│   └── *_structured.py    # Structured variants of above
├── models/                # Neural network architectures
│   ├── predictors/        # Edit effect and property predictors
│   ├── architectures/     # Multi-task learning components
│   ├── dataset.py         # PyTorch dataset wrappers
│   └── trainer.py         # Training utilities
└── utils/                 # Splits, metrics, caching, logging
    ├── splits.py          # Scaffold, target, Butina, etc.
    ├── metrics.py         # Regression and ranking metrics
    ├── embedding_cache.py # Embedding caching utilities
    └── logging.py         # Logging setup

experiments/               # Experiment runners
├── main.py                # Main entry point
├── experiment_config.py   # Configuration dataclass
├── data_loader.py         # Dataset loading and splitting
├── model_factory.py       # Embedder and model creation
├── trainer.py             # Training loop
├── evaluator.py           # Evaluation and metrics
├── report_generator.py    # Visualization and reports
├── cluster_analysis.py    # Edit embedding clustering
└── edit_embedding_comparison.py # Embedding comparison

scripts/
├── extraction/            # ChEMBL and MMP extraction scripts
│   ├── extract_chembl_data.py
│   ├── download_chembl_long_format.py
│   ├── build_pairs_long_format.py
│   ├── run_chembl_pair_extraction.py
│   └── mmpdb/             # mmpdb-based extraction
└── add_mmp_columns.py     # Compute MMP structural columns

data/                      # Raw and processed datasets (gitignored)
tests/                     # Unit tests
```

---

## Key Patterns

- **Embedders**: Inherit from `MoleculeEmbedder` base class in `src/embedding/base.py`
- **Experiments**: Config-driven via `ExperimentConfig` dataclass, placed in `experiments/`
- **Preprocessing**: Scripts go in `scripts/extraction/`
- **Edit modes**: Two modes supported — full molecule difference (context-aware) and edit fragment difference (scaffold-independent)
