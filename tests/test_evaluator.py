"""Tests for customer_churn_ml.training.evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from customer_churn_ml.training.evaluator import (
    build_comparison_table,
    evaluate_classifier,
    plot_roc_curves,
)


# ---------------------------------------------------------------------------
# Fixtures — simple binary classification outputs
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_preds():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])
    return y_true, y_pred, y_prob


@pytest.fixture
def imperfect_preds():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])  # two errors
    y_prob = np.array([0.1, 0.6, 0.8, 0.4, 0.2, 0.9])
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# evaluate_classifier — return keys
# ---------------------------------------------------------------------------

class TestEvaluateClassifier:
    def test_returns_required_keys_with_proba(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        for key in ("accuracy", "f1", "roc_auc", "confusion_matrix",
                    "classification_report", "fpr", "tpr"):
            assert key in result, f"Missing key: {key}"

    def test_returns_required_keys_without_proba(self, perfect_preds):
        y_true, y_pred, _ = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob=None)
        assert "accuracy" in result
        assert result["roc_auc"] is None

    def test_perfect_accuracy(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_roc_auc(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert result["roc_auc"] == pytest.approx(1.0)

    def test_imperfect_accuracy_below_1(self, imperfect_preds):
        y_true, y_pred, y_prob = imperfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert result["accuracy"] < 1.0

    def test_model_name_stored(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob, model_name="My Model")
        assert result["model_name"] == "My Model"

    def test_confusion_matrix_is_list(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert isinstance(result["confusion_matrix"], list)

    def test_fpr_tpr_are_lists(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert isinstance(result["fpr"], list)
        assert isinstance(result["tpr"], list)

    def test_accuracy_between_0_and_1(self, imperfect_preds):
        y_true, y_pred, y_prob = imperfect_preds
        result = evaluate_classifier(y_true, y_pred, y_prob)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    @pytest.fixture
    def sample_results(self):
        return {
            "Model A": {"accuracy": 0.80, "roc_auc": 0.85},
            "Model B": {"accuracy": 0.75, "roc_auc": 0.90},
            "Model C": {"accuracy": 0.70, "roc_auc": None},
        }

    def test_sorted_by_roc_auc_descending(self, sample_results):
        table = build_comparison_table(sample_results)
        aucs = [row["ROC_AUC"] for row in table if row["ROC_AUC"] is not None]
        assert aucs == sorted(aucs, reverse=True)

    def test_none_roc_auc_placed_last(self, sample_results):
        table = build_comparison_table(sample_results)
        assert table[-1]["Model"] == "Model C"

    def test_all_models_included(self, sample_results):
        table = build_comparison_table(sample_results)
        assert len(table) == 3

    def test_required_columns_present(self, sample_results):
        table = build_comparison_table(sample_results)
        for row in table:
            assert "Model" in row
            assert "Accuracy" in row
            assert "ROC_AUC" in row


# ---------------------------------------------------------------------------
# plot_roc_curves
# ---------------------------------------------------------------------------

class TestPlotRocCurves:
    @pytest.fixture
    def results_with_probs(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        metrics = evaluate_classifier(y_true, y_pred, y_prob, model_name="Model A")
        return {"Model A": metrics}, y_true

    def test_returns_axes(self, results_with_probs):
        import matplotlib
        matplotlib.use("Agg")
        results, y_true = results_with_probs
        ax = plot_roc_curves(results, y_true)
        assert ax is not None

    def test_uses_provided_axes(self, results_with_probs):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results, y_true = results_with_probs
        _, ax_in = plt.subplots()
        ax_out = plot_roc_curves(results, y_true, ax=ax_in)
        assert ax_out is ax_in

    def test_handles_results_without_probs_gracefully(self, perfect_preds):
        import matplotlib
        matplotlib.use("Agg")
        y_true, y_pred, _ = perfect_preds
        metrics = evaluate_classifier(y_true, y_pred, y_prob=None)
        results = {"Model A": metrics}
        ax = plot_roc_curves(results, y_true)  # must not raise
        assert ax is not None
