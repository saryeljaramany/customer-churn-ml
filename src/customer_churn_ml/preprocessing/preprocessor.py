"""Preprocessing pipeline for the churn dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import CARDINALITY_THRESHOLD, PATHS
from ..constants import ID_COL, MISSING_CATEGORY, TARGET_COL, TARGET_ENCODED_COL
from ..utils import ensure_directory, get_logger, save_pickle

logger = get_logger(__name__)


def _safe_numeric_median(series: pd.Series) -> float:
    """Return a usable median even if the series is all missing."""

    median = pd.to_numeric(series, errors="coerce").median()
    if pd.isna(median):
        return 0.0
    return float(median)


def _slugify_category(value: str) -> str:
    """Create a stable column suffix for one-hot encoded categories."""

    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "category"


@dataclass(slots=True)
class PreprocessorConfig:
    """Configuration for the churn preprocessing pipeline."""

    id_col: str = ID_COL
    target_col: str = TARGET_COL
    target_encoded_col: str = TARGET_ENCODED_COL
    cardinality_threshold: int = CARDINALITY_THRESHOLD
    processed_data_path: Path = PATHS.cleaned_data_file
    save_processed_data: bool = True


@dataclass(slots=True)
class Preprocessor:
    """Fit/transform preprocessing compatible with the notebook pipeline."""

    config: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    numeric_fill_value_: float | None = None
    categorical_columns_: list[str] = field(default_factory=list)
    low_cardinality_columns_: list[str] = field(default_factory=list)
    high_cardinality_columns_: list[str] = field(default_factory=list)
    low_cardinality_values_: dict[str, list[str]] = field(default_factory=dict)
    high_cardinality_frequency_: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    is_fitted_: bool = False

    def _prepare_base_frame(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        frame = df.copy()

        if self.config.id_col in frame.columns:
            frame = frame.drop(columns=[self.config.id_col])
            logger.info("Dropped %s", self.config.id_col)

        if "TotalCharges" in frame.columns:
            frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
            if fit:
                self.numeric_fill_value_ = _safe_numeric_median(frame["TotalCharges"])
            fill_value = self.numeric_fill_value_ if self.numeric_fill_value_ is not None else 0.0
            frame["TotalCharges"] = frame["TotalCharges"].fillna(fill_value)

        if self.config.target_col in frame.columns:
            frame[self.config.target_encoded_col] = (frame[self.config.target_col] == "Yes").astype(
                int
            )
            frame = frame.drop(columns=[self.config.target_col])

        return frame

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn encoding metadata from a dataframe."""

        if self.config.target_col not in df.columns:
            raise ValueError(f"Training data must contain '{self.config.target_col}'.")

        base = self._prepare_base_frame(df, fit=True)
        self.categorical_columns_ = base.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
        if self.config.target_encoded_col in self.categorical_columns_:
            self.categorical_columns_.remove(self.config.target_encoded_col)

        self.low_cardinality_columns_.clear()
        self.high_cardinality_columns_.clear()
        self.low_cardinality_values_.clear()
        self.high_cardinality_frequency_.clear()

        for column in self.categorical_columns_:
            series = base[column].fillna(MISSING_CATEGORY).astype(str)
            unique_values = sorted(series.unique().tolist())
            if len(unique_values) <= self.config.cardinality_threshold:
                self.low_cardinality_columns_.append(column)
                self.low_cardinality_values_[column] = unique_values
            else:
                self.high_cardinality_columns_.append(column)
                self.high_cardinality_frequency_[column] = series.value_counts(
                    normalize=True
                ).to_dict()

        transformed = self._transform_frame(base, allow_unfitted=True)
        if self.config.target_encoded_col in transformed.columns:
            self.feature_names_ = [
                column for column in transformed.columns if column != self.config.target_encoded_col
            ]
        else:
            self.feature_names_ = transformed.columns.tolist()

        self.is_fitted_ = True
        logger.info(
            "Fitted preprocessor with %d low-cardinality and %d high-cardinality columns",
            len(self.low_cardinality_columns_),
            len(self.high_cardinality_columns_),
        )
        return self

    def _transform_frame(self, df: pd.DataFrame, *, allow_unfitted: bool) -> pd.DataFrame:
        """Internal transformer that can run before ``is_fitted_`` is set."""

        if not allow_unfitted and not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        frame = self._prepare_base_frame(df, fit=False)

        for column in self.low_cardinality_columns_:
            if column not in frame.columns:
                continue
            categories = self.low_cardinality_values_[column]
            values = frame[column].fillna(MISSING_CATEGORY).astype(str)
            categorical = pd.Categorical(
                values.where(values.isin(categories), other=MISSING_CATEGORY), categories=categories
            )
            dummies = pd.get_dummies(categorical, prefix=column, drop_first=True)
            rename_map = {
                f"{column}_{category}": f"{column}_{_slugify_category(category)}"
                for category in categories[1:]
            }
            dummies = dummies.rename(columns=rename_map)
            expected_columns = list(rename_map.values())
            for expected_column in expected_columns:
                if expected_column not in dummies.columns:
                    dummies[expected_column] = 0
            dummies = dummies[expected_columns]
            frame = frame.drop(columns=[column])
            frame = pd.concat([frame, dummies], axis=1)

        for column in self.high_cardinality_columns_:
            if column not in frame.columns:
                continue
            values = frame[column].fillna(MISSING_CATEGORY).astype(str)
            frequencies = self.high_cardinality_frequency_.get(column, {})
            frame[f"{column}_freq"] = values.map(frequencies).fillna(0.0)
            frame = frame.drop(columns=[column])

        if self.feature_names_:
            aligned_columns = list(self.feature_names_)
            if self.config.target_encoded_col in frame.columns:
                aligned_columns = aligned_columns + [self.config.target_encoded_col]
            for column in aligned_columns:
                if column not in frame.columns:
                    frame[column] = 0
            extra_columns = [column for column in frame.columns if column not in aligned_columns]
            if extra_columns:
                frame = frame.drop(columns=extra_columns)
            frame = frame.reindex(columns=aligned_columns)

        return frame

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing metadata to a dataframe."""

        return self._transform_frame(df, allow_unfitted=False)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessing pipeline and transform the same dataframe."""

        self.fit(df)
        transformed = self.transform(df)
        if self.config.save_processed_data:
            ensure_directory(self.config.processed_data_path.parent)
            transformed.to_csv(self.config.processed_data_path, index=False)
            logger.info("Saved processed data to %s", self.config.processed_data_path)
        return transformed

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the learned preprocessor metadata."""

        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        target_path = path or (PATHS.model_dir / "preprocessor.pkl")
        return save_pickle(self, target_path)


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    config: PreprocessorConfig | None = None,
    save_processed_data: bool | None = None,
) -> tuple[pd.DataFrame, Preprocessor]:
    """Convenience wrapper around :class:`Preprocessor`."""

    cfg = config or PreprocessorConfig()
    if save_processed_data is not None:
        cfg = PreprocessorConfig(
            id_col=cfg.id_col,
            target_col=cfg.target_col,
            target_encoded_col=cfg.target_encoded_col,
            cardinality_threshold=cfg.cardinality_threshold,
            processed_data_path=cfg.processed_data_path,
            save_processed_data=save_processed_data,
        )
    preprocessor = Preprocessor(cfg)
    transformed = preprocessor.fit_transform(df)
    return transformed, preprocessor


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # CLI entry point for preprocessing.
    #
    # Usage:
    #     python -m customer_churn_ml.preprocessing.preprocessor
    #
    # Behavior:
    # - Reads the raw data CSV from config.PATHS.raw_data_file
    # - Runs the Preprocessor.fit_transform to produce cleaned data
    # - Saves the cleaned CSV to config.PATHS.cleaned_data_file
    #   (controlled by PreprocessorConfig.save_processed_data)
    # - Logs progress using the module logger
    logger.info("Starting preprocessing CLI")
    try:
        raw_path = PATHS.raw_data_file
        logger.info("Loading raw data from %s", raw_path)
        df_raw = pd.read_csv(raw_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load raw data from %s: %s", PATHS.raw_data_file, exc)
        raise SystemExit(1) from exc

    cfg = PreprocessorConfig(processed_data_path=PATHS.cleaned_data_file, save_processed_data=True)
    preprocessor = Preprocessor(cfg)
    logger.info("Fitting preprocessor on loaded data")
    processed_df = preprocessor.fit_transform(df_raw)
    logger.info("Preprocessing complete. Processed data shape: %s", processed_df.shape)
