"""
Integration tests for the baseline property predictor.

Verifies that:
1. BaselinePropertyPredictor trains on absolute values and predicts delta via subtraction
2. PropertyPredictor (used by the experiment pipeline) trains on absolute values
3. The trainer data preparation extracts value_a/value_b correctly
4. Both baseline_property and edit_framework methods can coexist in the same run
5. End-to-end pipeline works with synthetic MMP data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedding.fingerprints import FingerprintEmbedder
from src.embedding.edit_embedder import EditEmbedder
from src.models.predictors.baseline_property_predictor import (
    BaselinePropertyPredictor,
    BaselinePropertyMLP,
)
from src.models.predictors.property_predictor import PropertyPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small set of real SMILES for testing
SMILES_POOL = [
    "CCO",                # ethanol
    "CCCO",               # propanol
    "CC(=O)O",            # acetic acid
    "CC(=O)OC",           # methyl acetate
    "c1ccccc1",           # benzene
    "c1ccc(O)cc1",        # phenol
    "c1ccc(N)cc1",        # aniline
    "c1ccc(C)cc1",        # toluene
    "CC(C)O",             # isopropanol
    "CCOC",               # diethyl ether
    "CCN",                # ethylamine
    "CCNC",               # N-methylethylamine
    "c1ccc2ccccc2c1",     # naphthalene
    "c1ccc(Cl)cc1",       # chlorobenzene
    "c1ccc(F)cc1",        # fluorobenzene
    "c1ccc(Br)cc1",       # bromobenzene
]


def make_synthetic_mmp_dataset(
    n_pairs: int = 60,
    n_properties: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic MMP dataset with realistic columns.

    Generates pairs from SMILES_POOL with synthetic pIC50 values
    (value_a, value_b) and delta = value_b - value_a.
    """
    rng = np.random.RandomState(seed)
    rows = []
    property_names = [f"IC50_TARGET_{i}" for i in range(n_properties)]

    for i in range(n_pairs):
        prop = property_names[i % n_properties]
        idx_a = rng.randint(0, len(SMILES_POOL))
        idx_b = rng.randint(0, len(SMILES_POOL))
        while idx_b == idx_a:
            idx_b = rng.randint(0, len(SMILES_POOL))

        mol_a = SMILES_POOL[idx_a]
        mol_b = SMILES_POOL[idx_b]

        # Synthetic absolute pIC50 values in realistic range [4.0, 9.0]
        value_a = 4.0 + rng.rand() * 5.0
        value_b = value_a + rng.randn() * 0.8  # small delta
        delta = value_b - value_a

        rows.append({
            "mol_a": mol_a,
            "mol_b": mol_b,
            "value_a": round(value_a, 3),
            "value_b": round(value_b, 3),
            "delta": round(delta, 3),
            "edit_smiles": f"[H]>>{SMILES_POOL[idx_b]}",  # placeholder
            "property_name": prop,
        })

    return pd.DataFrame(rows)


def split_dataset(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split a dataframe into train/val/test."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return (
        shuffled.iloc[:n_train],
        shuffled.iloc[n_train : n_train + n_val],
        shuffled.iloc[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# Tests for BaselinePropertyPredictor (standalone class)
# ---------------------------------------------------------------------------


class TestBaselinePropertyPredictor:
    """Tests for the fixed BaselinePropertyPredictor."""

    def test_fit_accepts_absolute_values(self):
        """fit() should accept (mol_emb, y_absolute) not (wt_emb, mut_emb, delta)."""
        predictor = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=5, device="cpu"
        )
        mol_emb = np.random.randn(20, 64).astype(np.float32)
        y_abs = np.random.randn(20).astype(np.float32) + 6.0  # pIC50-like

        history = predictor.fit(mol_emb_train=mol_emb, y_train=y_abs, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_predict_returns_delta(self):
        """predict(mol_a, mol_b) should return f(mol_b) - f(mol_a)."""
        predictor = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=10, device="cpu"
        )
        mol_emb = np.random.randn(40, 64).astype(np.float32)
        y_abs = np.random.randn(40).astype(np.float32) + 6.0

        predictor.fit(mol_emb_train=mol_emb, y_train=y_abs, verbose=False)

        mol_a = np.random.randn(5, 64).astype(np.float32)
        mol_b = np.random.randn(5, 64).astype(np.float32)
        delta_pred = predictor.predict(mol_a, mol_b)

        assert delta_pred.shape == (5,)
        # Delta should be f(b) - f(a), not all zeros
        assert not np.allclose(delta_pred, 0.0, atol=1e-4)

    def test_identical_molecules_give_zero_delta(self):
        """Predicting delta for identical mol_a and mol_b should be near zero."""
        predictor = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=10, device="cpu"
        )
        mol_emb = np.random.randn(40, 64).astype(np.float32)
        y_abs = np.random.randn(40).astype(np.float32) + 6.0

        predictor.fit(mol_emb_train=mol_emb, y_train=y_abs, verbose=False)

        same_mol = np.random.randn(5, 64).astype(np.float32)
        delta_pred = predictor.predict(same_mol, same_mol)

        np.testing.assert_allclose(delta_pred, 0.0, atol=1e-6)

    def test_evaluate_computes_metrics(self):
        """evaluate() should return a metrics dict with standard regression metrics."""
        predictor = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=10, device="cpu"
        )
        mol_emb = np.random.randn(40, 64).astype(np.float32)
        y_abs = np.random.randn(40).astype(np.float32) + 6.0

        predictor.fit(mol_emb_train=mol_emb, y_train=y_abs, verbose=False)

        mol_a = np.random.randn(10, 64).astype(np.float32)
        mol_b = np.random.randn(10, 64).astype(np.float32)
        delta_true = np.random.randn(10).astype(np.float32)

        metrics, y_true, y_pred = predictor.evaluate(mol_a, mol_b, delta_true)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pearson" in metrics
        assert y_true.shape == (10,)
        assert y_pred.shape == (10,)

    def test_fit_signature_requires_two_args(self):
        """fit() requires exactly 2 positional args (mol_emb, y), not 3."""
        predictor = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=5, device="cpu"
        )
        mol_emb = np.random.randn(10, 64).astype(np.float32)
        y_abs = np.random.randn(10).astype(np.float32) + 6.0

        # The new signature should work with 2 positional args
        history = predictor.fit(mol_emb, y_abs, verbose=False)
        assert len(history["train_loss"]) > 0

        # Verify the new fit() accepts keyword args correctly
        predictor2 = BaselinePropertyPredictor(
            hidden_dims=[32], max_epochs=5, device="cpu"
        )
        history2 = predictor2.fit(
            mol_emb_train=mol_emb, y_train=y_abs, verbose=False
        )
        assert len(history2["train_loss"]) > 0


# ---------------------------------------------------------------------------
# Tests for PropertyPredictor (used by the experiment pipeline)
# ---------------------------------------------------------------------------


class TestPropertyPredictorPipeline:
    """Tests for PropertyPredictor as used by experiments/trainer.py."""

    def test_trains_on_absolute_values_with_precomputed_embeddings(self):
        """PropertyPredictor should train on absolute pIC50 values.

        Uses 2 tasks to match the real pipeline which always passes
        task_names (multi-task mode).
        """
        embedder = FingerprintEmbedder(fp_type="morgan", n_bits=256)
        task_names = ["IC50_TARGET_0", "IC50_TARGET_1"]
        predictor = PropertyPredictor(
            embedder=embedder,
            task_names=task_names,
            hidden_dims=[64, 32],
            max_epochs=3,
            device="cpu",
        )

        df = make_synthetic_mmp_dataset(n_pairs=60, n_properties=2)
        train_df, val_df, _ = split_dataset(df)

        def _extract_absolute_multi(sub_df):
            smiles, vals = [], []
            for idx in range(len(sub_df)):
                row = sub_df.iloc[idx]
                prop = row["property_name"]
                task_idx = task_names.index(prop)

                # mol_a
                y_row_a = np.full(len(task_names), np.nan, dtype=np.float32)
                y_row_a[task_idx] = row["value_a"]
                vals.append(y_row_a)
                smiles.append(row["mol_a"])

                # mol_b
                y_row_b = np.full(len(task_names), np.nan, dtype=np.float32)
                y_row_b[task_idx] = row["value_b"]
                vals.append(y_row_b)
                smiles.append(row["mol_b"])
            return smiles, np.array(vals, dtype=np.float32)

        train_smiles, y_train = _extract_absolute_multi(train_df)
        val_smiles, y_val = _extract_absolute_multi(val_df)

        mol_emb_train = np.array(embedder.encode(train_smiles), dtype=np.float32)
        mol_emb_val = np.array(embedder.encode(val_smiles), dtype=np.float32)

        predictor.fit(
            mol_emb_train=mol_emb_train,
            y_train=y_train,
            mol_emb_val=mol_emb_val,
            y_val=y_val,
            verbose=False,
        )

        # Predict on test molecules
        test_smiles_a = [df.iloc[0]["mol_a"]]
        test_smiles_b = [df.iloc[0]["mol_b"]]

        preds_a = predictor.predict(test_smiles_a)
        preds_b = predictor.predict(test_smiles_b)

        # Multi-task returns dict
        assert "IC50_TARGET_0" in preds_a
        assert "IC50_TARGET_0" in preds_b

        # Delta via subtraction (same as evaluator.py)
        delta_pred = preds_b["IC50_TARGET_0"] - preds_a["IC50_TARGET_0"]
        assert np.isfinite(delta_pred).all()

    def test_multi_task_with_sparse_labels(self):
        """Multi-task training with NaN labels (sparse tasks) should work."""
        embedder = FingerprintEmbedder(fp_type="morgan", n_bits=256)
        predictor = PropertyPredictor(
            embedder=embedder,
            task_names=["IC50_TARGET_0", "IC50_TARGET_1"],
            hidden_dims=[64, 32],
            max_epochs=3,
            device="cpu",
        )

        df = make_synthetic_mmp_dataset(n_pairs=60, n_properties=2)
        train_df, val_df, _ = split_dataset(df)

        def _extract_multi_task(sub_df):
            smiles_list, val_list = [], []
            for idx in range(len(sub_df)):
                row = sub_df.iloc[idx]
                prop = row["property_name"]
                task_idx = 0 if prop == "IC50_TARGET_0" else 1

                y_row_a = np.full(2, np.nan, dtype=np.float32)
                y_row_a[task_idx] = row["value_a"]
                val_list.append(y_row_a)
                smiles_list.append(row["mol_a"])

                y_row_b = np.full(2, np.nan, dtype=np.float32)
                y_row_b[task_idx] = row["value_b"]
                val_list.append(y_row_b)
                smiles_list.append(row["mol_b"])
            return smiles_list, np.array(val_list, dtype=np.float32)

        train_smiles, y_train = _extract_multi_task(train_df)
        val_smiles, y_val = _extract_multi_task(val_df)

        mol_emb_train = np.array(embedder.encode(train_smiles), dtype=np.float32)
        mol_emb_val = np.array(embedder.encode(val_smiles), dtype=np.float32)

        predictor.fit(
            mol_emb_train=mol_emb_train,
            y_train=y_train,
            mol_emb_val=mol_emb_val,
            y_val=y_val,
            verbose=False,
        )

        preds = predictor.predict(["c1ccccc1"])
        assert "IC50_TARGET_0" in preds
        assert "IC50_TARGET_1" in preds


# ---------------------------------------------------------------------------
# Tests for trainer data preparation logic
# ---------------------------------------------------------------------------


class TestTrainerDataPreparation:
    """
    Tests that the trainer's data preparation for baseline_property correctly
    extracts absolute values from MMP pairs.
    """

    def test_value_a_value_b_extraction(self):
        """
        Verify that the trainer extracts value_a and value_b (absolute pIC50)
        from each pair, not delta values.

        This replicates the logic in experiments/trainer.py lines 50-86.
        """
        df = make_synthetic_mmp_dataset(n_pairs=20, n_properties=2)
        task_names = sorted(df["property_name"].unique())

        train_data = {}
        for prop in task_names:
            prop_df = df[df["property_name"] == prop].reset_index(drop=True)
            train_data[prop] = {"train": prop_df, "val": prop_df[:5], "test": prop_df[5:10]}

        num_tasks = len(task_names)

        # Replicate trainer.py baseline_property logic
        train_smiles_list = []
        train_val_list = []

        for i, prop in enumerate(task_names):
            train_prop = train_data[prop]["train"]

            for idx in range(len(train_prop)):
                # mol_a sample
                y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                y_row[i] = train_prop.iloc[idx]["value_a"]
                train_val_list.append(y_row)
                train_smiles_list.append(train_prop.iloc[idx]["mol_a"])

                # mol_b sample
                y_row = np.full(num_tasks, np.nan, dtype=np.float32)
                y_row[i] = train_prop.iloc[idx]["value_b"]
                train_val_list.append(y_row)
                train_smiles_list.append(train_prop.iloc[idx]["mol_b"])

        y_train = np.array(train_val_list, dtype=np.float32)

        # Verify: each row should have exactly one non-NaN value
        for row in y_train:
            non_nan = np.sum(~np.isnan(row))
            assert non_nan == 1, f"Expected 1 non-NaN per row, got {non_nan}"

        # Verify: non-NaN values should be in a pIC50-like range,
        # NOT zero (the old bug) and NOT delta values (near zero mean)
        non_nan_values = y_train[~np.isnan(y_train)]
        # With value_a in [4,9] and value_b = value_a + N(0,0.8),
        # values should mostly be in [1, 12] range
        assert non_nan_values.mean() > 3.0, (
            f"Mean value {non_nan_values.mean():.3f} is too low for pIC50 -- "
            "possibly using delta or zero labels"
        )

        # Verify: no zero labels (the old bug assigned y=0 to wild-type)
        zero_count = np.sum(np.abs(non_nan_values) < 0.01)
        total = len(non_nan_values)
        zero_frac = zero_count / total
        assert zero_frac < 0.05, (
            f"{zero_frac:.1%} of labels are near-zero -- "
            "possible old bug where wild-type gets y=0"
        )


# ---------------------------------------------------------------------------
# End-to-end test: both methods in same experiment
# ---------------------------------------------------------------------------


class TestDualMethodExperiment:
    """
    Verify that both baseline_property and edit_framework can coexist
    in the same experiment with shared data.
    """

    def test_baseline_and_edit_framework_coexist(self):
        """
        Both methods should train and produce valid predictions on the
        same dataset, using the same multi-task setup as the real pipeline.
        """
        embedder = FingerprintEmbedder(fp_type="morgan", n_bits=256)
        edit_embedder = EditEmbedder(embedder)

        df = make_synthetic_mmp_dataset(n_pairs=80, n_properties=2)
        task_names = ["IC50_TARGET_0", "IC50_TARGET_1"]
        train_df, val_df, test_df = split_dataset(df)

        # --- Baseline: PropertyPredictor on absolute values (multi-task) ---
        baseline = PropertyPredictor(
            embedder=embedder,
            task_names=task_names,
            hidden_dims=[64, 32],
            max_epochs=3,
            device="cpu",
        )

        def _extract_absolute_mt(sub_df):
            smiles, ys = [], []
            for idx in range(len(sub_df)):
                row = sub_df.iloc[idx]
                prop = row["property_name"]
                task_idx = task_names.index(prop)

                y_a = np.full(len(task_names), np.nan, dtype=np.float32)
                y_a[task_idx] = row["value_a"]
                ys.append(y_a)
                smiles.append(row["mol_a"])

                y_b = np.full(len(task_names), np.nan, dtype=np.float32)
                y_b[task_idx] = row["value_b"]
                ys.append(y_b)
                smiles.append(row["mol_b"])
            emb = np.array(embedder.encode(smiles), dtype=np.float32)
            y = np.array(ys, dtype=np.float32)
            return emb, y

        baseline_emb_train, baseline_y_train = _extract_absolute_mt(train_df)
        baseline_emb_val, baseline_y_val = _extract_absolute_mt(val_df)

        baseline.fit(
            mol_emb_train=baseline_emb_train,
            y_train=baseline_y_train,
            mol_emb_val=baseline_emb_val,
            y_val=baseline_y_val,
            verbose=False,
        )

        # Baseline prediction: delta = f(mol_b) - f(mol_a)
        # Filter test to first property for evaluation
        test_prop = test_df[test_df["property_name"] == "IC50_TARGET_0"]
        if len(test_prop) == 0:
            test_prop = test_df.head(5)

        test_smiles_a = test_prop["mol_a"].tolist()
        test_smiles_b = test_prop["mol_b"].tolist()

        test_emb_a = np.array(embedder.encode(test_smiles_a), dtype=np.float32)
        test_emb_b = np.array(embedder.encode(test_smiles_b), dtype=np.float32)

        preds_a = baseline.predict(test_smiles_a, mol_emb=test_emb_a)
        preds_b = baseline.predict(test_smiles_b, mol_emb=test_emb_b)
        baseline_delta = preds_b["IC50_TARGET_0"] - preds_a["IC50_TARGET_0"]

        assert baseline_delta.shape == (len(test_prop),)
        assert np.isfinite(baseline_delta).all()

        # --- Edit framework: EditEffectPredictor (multi-task) ---
        from src.models.predictors.edit_effect_predictor import EditEffectPredictor

        edit_predictor = EditEffectPredictor(
            mol_embedder=embedder,
            edit_embedder=edit_embedder,
            task_names=task_names,
            hidden_dims=[64, 32],
            max_epochs=3,
            device="cpu",
        )

        def _extract_delta_mt(sub_df):
            mol_a = sub_df["mol_a"].tolist()
            mol_b = sub_df["mol_b"].tolist()
            emb_a = np.array(embedder.encode(mol_a), dtype=np.float32)
            emb_b = np.array(embedder.encode(mol_b), dtype=np.float32)
            delta = np.full((len(sub_df), len(task_names)), np.nan, dtype=np.float32)
            for idx in range(len(sub_df)):
                row = sub_df.iloc[idx]
                task_idx = task_names.index(row["property_name"])
                delta[idx, task_idx] = row["delta"]
            return emb_a, emb_b, delta

        emb_a_train, emb_b_train, delta_train = _extract_delta_mt(train_df)
        emb_a_val, emb_b_val, delta_val = _extract_delta_mt(val_df)

        edit_predictor.fit(
            mol_emb_a=emb_a_train,
            mol_emb_b=emb_b_train,
            delta_y=delta_train,
            mol_emb_a_val=emb_a_val,
            mol_emb_b_val=emb_b_val,
            delta_y_val=delta_val,
            verbose=False,
        )

        # Edit framework prediction
        edit_preds = edit_predictor.predict(
            test_smiles_a,
            test_smiles_b,
            mol_emb_a=test_emb_a,
            mol_emb_b=test_emb_b,
        )
        edit_delta = edit_preds["IC50_TARGET_0"]

        assert edit_delta.shape == (len(test_prop),)
        assert np.isfinite(edit_delta).all()

        # Both should produce non-trivial predictions (not all constant)
        assert np.std(baseline_delta) > 1e-6, "Baseline predictions are constant"
        assert np.std(edit_delta) > 1e-6, "Edit framework predictions are constant"


# ---------------------------------------------------------------------------
# Test for BaselinePropertyPredictor with real SMILES
# ---------------------------------------------------------------------------


class TestBaselineWithRealSMILES:
    """End-to-end test using real SMILES with FingerprintEmbedder."""

    def test_end_to_end_with_fingerprints(self):
        """Train on FP embeddings of real molecules, predict deltas."""
        embedder = FingerprintEmbedder(fp_type="morgan", n_bits=256)
        predictor = BaselinePropertyPredictor(
            hidden_dims=[64, 32],
            max_epochs=20,
            patience=5,
            device="cpu",
        )

        df = make_synthetic_mmp_dataset(n_pairs=40, n_properties=1)
        train_df, val_df, test_df = split_dataset(df)

        # Prepare training data: stack mol_a and mol_b with their absolute values
        train_smiles = []
        train_y = []
        for idx in range(len(train_df)):
            row = train_df.iloc[idx]
            train_smiles.append(row["mol_a"])
            train_y.append(row["value_a"])
            train_smiles.append(row["mol_b"])
            train_y.append(row["value_b"])

        mol_emb_train = np.array(embedder.encode(train_smiles), dtype=np.float32)
        y_train = np.array(train_y, dtype=np.float32)

        # Validation
        val_smiles = []
        val_y = []
        for idx in range(len(val_df)):
            row = val_df.iloc[idx]
            val_smiles.append(row["mol_a"])
            val_y.append(row["value_a"])
            val_smiles.append(row["mol_b"])
            val_y.append(row["value_b"])

        mol_emb_val = np.array(embedder.encode(val_smiles), dtype=np.float32)
        y_val = np.array(val_y, dtype=np.float32)

        history = predictor.fit(
            mol_emb_train=mol_emb_train,
            y_train=y_train,
            mol_emb_val=mol_emb_val,
            y_val=y_val,
            verbose=False,
        )

        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

        # Predict delta on test set
        mol_emb_a = np.array(
            embedder.encode(test_df["mol_a"].tolist()), dtype=np.float32
        )
        mol_emb_b = np.array(
            embedder.encode(test_df["mol_b"].tolist()), dtype=np.float32
        )

        delta_pred = predictor.predict(mol_emb_a, mol_emb_b)
        assert delta_pred.shape == (len(test_df),)
        assert np.isfinite(delta_pred).all()

        # Evaluate
        metrics, y_true, y_pred = predictor.evaluate(
            mol_emb_a, mol_emb_b, test_df["delta"].values
        )
        assert "mae" in metrics
        assert np.isfinite(metrics["mae"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
