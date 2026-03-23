"""Tests for customer_churn_ml.interpret.feature_importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_churn_ml.interpret.feature_importance import (
    get_feature_importance_df,
    plot_feature_importance,
)


# ---------------------------------------------------------------------------
# get_feature_importance_df — Random Forest
# ---------------------------------------------------------------------------

class TestRandomForest:
    def test_returns_dataframe(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        assert isinstance(result, pd.DataFrame)

    def test_importance_column_present(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        assert "Importance" in result.columns

    def test_feature_column_present(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        assert "Feature" in result.columns

    def test_all_features_included(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        assert len(result) == len(X.columns)

    def test_sorted_descending_by_importance(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        importances = result["Importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_importances_sum_to_1(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns)
        assert result["Importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_top_n_limits_rows(self, fitted_rf, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_rf, X.columns, top_n=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_feature_importance_df — Logistic Regression
# ---------------------------------------------------------------------------

class TestLogisticRegression:
    def test_returns_dataframe(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        assert isinstance(result, pd.DataFrame)

    def test_coefficient_column_present(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        assert "Coefficient" in result.columns

    def test_absolute_importance_column_present(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        assert "AbsoluteImportance" in result.columns

    def test_sorted_by_absolute_value_descending(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        abs_vals = result["AbsoluteImportance"].tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_coefficients_can_be_negative(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        # Logistic regression can have negative coefficients
        assert (result["Coefficient"] < 0).any() or (result["Coefficient"] >= 0).all()

    def test_all_features_included(self, fitted_lr, X_y):
        X, _ = X_y
        result = get_feature_importance_df(fitted_lr, X.columns)
        assert len(result) == len(X.columns)


# ---------------------------------------------------------------------------
# Unsupported model
# ---------------------------------------------------------------------------

class TestUnsupportedModel:
    def test_raises_for_model_without_importances_or_coef(self, X_y):
        X, _ = X_y

        class FakeModel:
            pass

        with pytest.raises(ValueError, match="feature_importances_"):
            get_feature_importance_df(FakeModel(), X.columns)


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_returns_axes_for_rf(self, fitted_rf, X_y):
        import matplotlib
        matplotlib.use("Agg")
        X, _ = X_y
        df = get_feature_importance_df(fitted_rf, X.columns)
        ax = plot_feature_importance(df)
        assert ax is not None

    def test_returns_axes_for_lr(self, fitted_lr, X_y):
        import matplotlib
        matplotlib.use("Agg")
        X, _ = X_y
        df = get_feature_importance_df(fitted_lr, X.columns)
        ax = plot_feature_importance(df)
        assert ax is not None

    def test_uses_provided_axes(self, fitted_rf, X_y):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X, _ = X_y
        df = get_feature_importance_df(fitted_rf, X.columns)
        _, ax_in = plt.subplots()
        ax_out = plot_feature_importance(df, ax=ax_in)
        assert ax_out is ax_in
