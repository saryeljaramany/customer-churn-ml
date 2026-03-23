"""Tests for customer_churn_ml.preprocessing.preprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_churn_ml.constants import TARGET_ENCODED_COL
from customer_churn_ml.preprocessing.preprocessor import (
    Preprocessor,
    PreprocessorConfig,
    preprocess_dataframe,
)


def _make_config(save=False) -> PreprocessorConfig:
    return PreprocessorConfig(save_processed_data=save)


# ---------------------------------------------------------------------------
# Basic output shape and schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_customer_id_is_dropped(self, cleaned_df):
        assert "customerID" not in cleaned_df.columns

    def test_original_churn_column_is_dropped(self, cleaned_df):
        assert "Churn" not in cleaned_df.columns

    def test_target_encoded_column_present(self, cleaned_df):
        assert TARGET_ENCODED_COL in cleaned_df.columns

    def test_target_is_binary(self, cleaned_df):
        assert set(cleaned_df[TARGET_ENCODED_COL].unique()).issubset({0, 1})

    def test_no_object_dtypes_remain(self, cleaned_df):
        obj_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns.tolist()
        assert obj_cols == [], f"Object columns remain: {obj_cols}"

    def test_numeric_cols_present(self, cleaned_df):
        for col in ("tenure", "MonthlyCharges", "TotalCharges"):
            assert col in cleaned_df.columns

    def test_row_count_unchanged(self, raw_df, cleaned_df):
        assert len(cleaned_df) == len(raw_df)


# ---------------------------------------------------------------------------
# Churn encoding
# ---------------------------------------------------------------------------

class TestChurnEncoding:
    def test_yes_maps_to_1(self, raw_df, cleaned_df):
        churn_yes_indices = raw_df.index[raw_df["Churn"] == "Yes"].tolist()
        assert all(cleaned_df.loc[churn_yes_indices, TARGET_ENCODED_COL] == 1)

    def test_no_maps_to_0(self, raw_df, cleaned_df):
        churn_no_indices = raw_df.index[raw_df["Churn"] == "No"].tolist()
        assert all(cleaned_df.loc[churn_no_indices, TARGET_ENCODED_COL] == 0)


# ---------------------------------------------------------------------------
# TotalCharges coercion
# ---------------------------------------------------------------------------

class TestTotalChargesCoercion:
    def test_empty_string_filled_with_median(self, cleaned_df):
        # Row 2 has an empty TotalCharges string — must not be NaN after preprocessing
        assert cleaned_df["TotalCharges"].isna().sum() == 0

    def test_total_charges_is_numeric(self, cleaned_df):
        assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])

    def test_fill_value_is_median_not_zero(self, raw_df):
        config = _make_config()
        preprocessor = Preprocessor(config)
        preprocessor.fit(raw_df)
        # Median of valid values should not be 0
        assert preprocessor.numeric_fill_value_ > 0


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

class TestCategoricalEncoding:
    def test_contract_one_hot_columns_created(self, cleaned_df):
        # "Month-to-month" is dropped (first category alphabetically)
        # "One year" and "Two year" get their own columns
        assert any("Contract" in c for c in cleaned_df.columns)

    def test_drop_first_respected(self, cleaned_df):
        # Month-to-month is the first alphabetically — must not appear as a column
        assert "Contract_Month_to_month" not in cleaned_df.columns

    def test_one_hot_values_are_binary(self, cleaned_df):
        ohe_cols = [c for c in cleaned_df.columns if c.startswith("Contract_")]
        for col in ohe_cols:
            assert set(cleaned_df[col].unique()).issubset({0, 1})

    def test_slugified_column_names(self, cleaned_df):
        # Spaces and special chars must be replaced with underscores
        for col in cleaned_df.columns:
            assert " " not in col
            assert "-" not in col


# ---------------------------------------------------------------------------
# Transform on unseen data (train/test schema consistency)
# ---------------------------------------------------------------------------

class TestTransformConsistency:
    def test_transform_produces_same_columns_as_fit_transform(self, raw_df):
        config = _make_config()
        preprocessor = Preprocessor(config)
        train_df = preprocessor.fit_transform(raw_df)

        # Transform a subset — schema must match
        test_df = preprocessor.transform(raw_df.iloc[:3])
        train_feature_cols = [c for c in train_df.columns if c != TARGET_ENCODED_COL]
        test_feature_cols  = [c for c in test_df.columns  if c != TARGET_ENCODED_COL]
        assert train_feature_cols == test_feature_cols

    def test_unseen_category_becomes_zero(self, raw_df):
        config = _make_config()
        preprocessor = Preprocessor(config)
        preprocessor.fit(raw_df)

        # Inject a category that was not seen during fit
        new_row = raw_df.iloc[[0]].copy()
        new_row["Contract"] = "Three year"  # unseen category
        result = preprocessor.transform(new_row)

        # All one-hot contract columns must be 0
        contract_cols = [c for c in result.columns if c.startswith("Contract_")]
        assert all(result[contract_cols].iloc[0] == 0)

    def test_transform_raises_before_fit(self, raw_df):
        config = _make_config()
        preprocessor = Preprocessor(config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(raw_df)


# ---------------------------------------------------------------------------
# fit requires target column
# ---------------------------------------------------------------------------

class TestFitValidation:
    def test_fit_raises_without_churn_column(self, raw_df):
        config = _make_config()
        preprocessor = Preprocessor(config)
        df_no_target = raw_df.drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Churn"):
            preprocessor.fit(df_no_target)


# ---------------------------------------------------------------------------
# No disk writes when save_processed_data=False
# ---------------------------------------------------------------------------

class TestNoDiskWrites:
    def test_no_file_written_when_save_false(self, raw_df, tmp_path):
        output_path = tmp_path / "cleaned.csv"
        config = PreprocessorConfig(
            save_processed_data=False,
            processed_data_path=output_path,
        )
        preprocessor = Preprocessor(config)
        preprocessor.fit_transform(raw_df)
        assert not output_path.exists()

    def test_file_written_when_save_true(self, raw_df, tmp_path):
        output_path = tmp_path / "cleaned.csv"
        config = PreprocessorConfig(
            save_processed_data=True,
            processed_data_path=output_path,
        )
        preprocessor = Preprocessor(config)
        preprocessor.fit_transform(raw_df)
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------

class TestPreprocessDataframe:
    def test_returns_tuple_of_df_and_preprocessor(self, raw_df):
        result = preprocess_dataframe(raw_df, save_processed_data=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preprocessor_in_tuple_is_fitted(self, raw_df):
        _, preprocessor = preprocess_dataframe(raw_df, save_processed_data=False)
        assert preprocessor.is_fitted_
