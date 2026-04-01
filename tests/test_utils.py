"""Tests for customer_churn_ml.utils."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from customer_churn_ml.utils import (
    check_data_integrity,
    ensure_directory,
    load_pickle,
    resolve_path,
    save_pickle,
    validate_columns,
)

# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_path_returned_unchanged(self, tmp_path):
        result = resolve_path(tmp_path)
        assert result == tmp_path

    def test_relative_path_anchored_to_project_root(self):
        result = resolve_path("data/raw/file.csv")
        assert result.is_absolute()
        assert result.parts[-3:] == ("data", "raw", "file.csv")

    def test_string_input_accepted(self):
        result = resolve_path("some/path.csv")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------


class TestEnsureDirectory:
    def test_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "subdir"
        assert not new_dir.exists()
        ensure_directory(new_dir)
        assert new_dir.is_dir()

    def test_does_not_raise_if_directory_exists(self, tmp_path):
        ensure_directory(tmp_path)  # already exists — must not raise


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------


class TestValidateColumns:
    def test_passes_when_all_columns_present(self, raw_df):
        validate_columns(raw_df, ["tenure", "MonthlyCharges", "Churn"])

    def test_raises_on_missing_column(self, raw_df):
        with pytest.raises(ValueError, match="missing required columns"):
            validate_columns(raw_df, ["tenure", "nonexistent_col"])

    def test_empty_required_list_always_passes(self, raw_df):
        validate_columns(raw_df, [])

    def test_frame_name_appears_in_error(self, raw_df):
        with pytest.raises(ValueError, match="my_frame"):
            validate_columns(raw_df, ["bad_col"], frame_name="my_frame")


# ---------------------------------------------------------------------------
# check_data_integrity
# ---------------------------------------------------------------------------


class TestCheckDataIntegrity:
    def test_returns_required_keys(self, raw_df):
        report = check_data_integrity(raw_df)
        assert "shape" in report
        assert "duplicate_rows" in report
        assert "missing_values" in report

    def test_shape_matches_dataframe(self, raw_df):
        report = check_data_integrity(raw_df)
        assert report["shape"] == raw_df.shape

    def test_duplicate_rows_counted_correctly(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        report = check_data_integrity(df)
        assert report["duplicate_rows"] == 1

    def test_duplicate_ids_reported_when_id_col_present(self, raw_df):
        report = check_data_integrity(raw_df, id_col="customerID")
        assert report["duplicate_ids"] == 0

    def test_duplicate_ids_none_when_id_col_absent(self):
        df = pd.DataFrame({"a": [1, 2]})
        report = check_data_integrity(df, id_col="customerID")
        assert report["duplicate_ids"] is None

    def test_missing_values_detected(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, None]})
        report = check_data_integrity(df)
        assert report["missing_values"]["a"] == 1
        assert report["missing_values"]["b"] == 2


# ---------------------------------------------------------------------------
# save_pickle / load_pickle
# ---------------------------------------------------------------------------


class TestPickleRoundtrip:
    def test_roundtrip_dict(self, tmp_path):
        obj = {"key": [1, 2, 3]}
        path = tmp_path / "test.pkl"
        save_pickle(obj, path)
        loaded = load_pickle(path)
        assert loaded == obj

    def test_roundtrip_dataframe(self, tmp_path, raw_df):
        path = tmp_path / "df.pkl"
        save_pickle(raw_df, path)
        loaded = load_pickle(path)
        pd.testing.assert_frame_equal(loaded, raw_df)

    def test_save_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "obj.pkl"
        save_pickle({"x": 1}, path)
        assert path.exists()

    def test_save_returns_path(self, tmp_path):
        path = tmp_path / "obj.pkl"
        returned = save_pickle(42, path)
        assert returned == path
