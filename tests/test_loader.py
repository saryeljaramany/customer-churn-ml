"""Tests for customer_churn_ml.data.loader."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_churn_ml.data.loader import load_csv


class TestLoadCsv:
    def test_loads_valid_csv(self, tmp_path, raw_df):
        path = tmp_path / "data.csv"
        raw_df.to_csv(path, index=False)

        df = load_csv(path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == raw_df.shape

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")

    def test_validates_required_columns(self, tmp_path, raw_df):
        path = tmp_path / "data.csv"
        raw_df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(path, required_columns=["tenure", "col_that_does_not_exist"])

    def test_passes_when_required_columns_present(self, tmp_path, raw_df):
        path = tmp_path / "data.csv"
        raw_df.to_csv(path, index=False)

        df = load_csv(path, required_columns=["tenure", "Churn"])
        assert "tenure" in df.columns

    def test_no_validation_when_required_columns_none(self, tmp_path, raw_df):
        path = tmp_path / "data.csv"
        raw_df.to_csv(path, index=False)
        df = load_csv(path, required_columns=None)
        assert df is not None
