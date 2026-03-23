"""Tests for customer_churn_ml.preprocessing.scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from customer_churn_ml.constants import NUMERIC_COLS
from customer_churn_ml.preprocessing.scalar import NumericScaler, ScalerConfig


def _make_scaler(path=None) -> NumericScaler:
    config = ScalerConfig(scaler_path=path) if path else ScalerConfig()
    return NumericScaler(config)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_is_fitted_after_fit(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        scaler = _make_scaler()
        scaler.fit(X)
        assert scaler.is_fitted_

    def test_raises_on_missing_numeric_col(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes", "tenure"])
        scaler = _make_scaler()
        with pytest.raises(ValueError, match="missing required columns"):
            scaler.fit(X)

    def test_raises_transform_before_fit(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        scaler = _make_scaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(X)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestTransform:
    def test_numeric_cols_are_scaled(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        scaler = _make_scaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        for col in NUMERIC_COLS:
            # After StandardScaler the mean should be ~0
            assert abs(X_scaled[col].mean()) < 1e-9, f"{col} mean not ~0 after scaling"

    def test_non_numeric_cols_unchanged(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        non_numeric = [c for c in X.columns if c not in NUMERIC_COLS]
        scaler = _make_scaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        for col in non_numeric:
            pd.testing.assert_series_equal(X[col], X_scaled[col], check_names=True)

    def test_does_not_mutate_input(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        original_values = X[list(NUMERIC_COLS)].copy()
        scaler = _make_scaler()
        scaler.fit(X)
        scaler.transform(X)

        pd.testing.assert_frame_equal(X[list(NUMERIC_COLS)], original_values)

    def test_output_shape_matches_input(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        scaler = _make_scaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        assert X_scaled.shape == X.shape


# ---------------------------------------------------------------------------
# Leakage prevention — fit only on train
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_test_stats_differ_from_train_stats(self, cleaned_df):
        """Test data should NOT be zero-mean after scaling — only train data is."""
        X = cleaned_df.drop(columns=["Churn_Yes"])
        y = cleaned_df["Churn_Yes"]
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.4, random_state=42)

        scaler = _make_scaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Train cols should be ~zero mean; test cols will differ (small dataset)
        for col in NUMERIC_COLS:
            train_mean = abs(X_train_scaled[col].mean())
            assert train_mean < 1e-9, f"Train {col} mean not ~0: {train_mean}"

        # At least one test column mean should be non-zero (test != train distribution)
        test_means = [abs(X_test_scaled[col].mean()) for col in NUMERIC_COLS]
        assert any(m > 1e-6 for m in test_means), "Test means are all zero — possible leakage"


# ---------------------------------------------------------------------------
# fit_transform convenience method
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_single_df_returns_dataframe(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        scaler = _make_scaler()
        result = scaler.fit_transform(X)
        assert isinstance(result, pd.DataFrame)

    def test_train_test_pair_returns_two_dataframes(self, cleaned_df):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        X_train, X_test = X.iloc[:5], X.iloc[5:]
        scaler = _make_scaler()
        train_scaled, test_scaled = scaler.fit_transform(X_train, X_test)
        assert train_scaled.shape == X_train.shape
        assert test_scaled.shape  == X_test.shape


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_raises_when_unfitted(self, tmp_path):
        scaler = NumericScaler(ScalerConfig(scaler_path=tmp_path / "scaler.pkl"))
        with pytest.raises(RuntimeError, match="unfitted"):
            scaler.save()

    def test_roundtrip_produces_same_output(self, cleaned_df, tmp_path):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        path = tmp_path / "scaler.pkl"

        scaler = NumericScaler(ScalerConfig(scaler_path=path))
        scaler.fit(X)
        scaler.save(path)

        loaded = NumericScaler.load(path)
        X_original = scaler.transform(X)
        X_loaded   = loaded.transform(X)

        pd.testing.assert_frame_equal(X_original, X_loaded)

    def test_loaded_scaler_is_fitted(self, cleaned_df, tmp_path):
        X = cleaned_df.drop(columns=["Churn_Yes"])
        path = tmp_path / "scaler.pkl"
        scaler = NumericScaler(ScalerConfig(scaler_path=path))
        scaler.fit(X)
        scaler.save(path)

        loaded = NumericScaler.load(path)
        assert loaded.is_fitted_
