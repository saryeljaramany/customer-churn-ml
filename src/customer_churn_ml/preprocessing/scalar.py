"""Scaling helpers for numeric columns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import PATHS
from ..constants import NUMERIC_COLS
from ..utils import get_logger, load_pickle, save_pickle, validate_columns

logger = get_logger(__name__)


@dataclass(slots=True)
class ScalerConfig:
    """Configuration for numeric feature scaling."""

    numeric_columns: tuple[str, ...] = NUMERIC_COLS
    scaler_path: Path = PATHS.scaler_file


class NumericScaler:
    """Fit and apply a StandardScaler to selected numeric columns."""

    def __init__(self, config: ScalerConfig | None = None) -> None:
        self.config = config or ScalerConfig()
        self.scaler_ = StandardScaler()
        self.is_fitted_ = False

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return self.config.numeric_columns

    def fit(self, df: pd.DataFrame) -> NumericScaler:
        validate_columns(df, self.numeric_columns, frame_name="feature frame")
        self.scaler_.fit(df.loc[:, self.numeric_columns])
        self.is_fitted_ = True
        logger.info("Fitted scaler on columns: %s", list(self.numeric_columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Scaler has not been fitted yet.")
        validate_columns(df, self.numeric_columns, frame_name="feature frame")

        transformed = df.copy()
        for col in self.numeric_columns:
            transformed[col] = transformed[col].astype("float64")
        transformed.loc[:, self.numeric_columns] = self.scaler_.transform(
            transformed.loc[:, self.numeric_columns]
        )
        return transformed

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Fit on the training frame and transform train/test copies."""

        self.fit(train_df)
        train_scaled = self.transform(train_df)
        if test_df is None:
            return train_scaled
        test_scaled = self.transform(test_df)
        return train_scaled, test_scaled

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the fitted scaler to disk."""

        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted scaler.")
        target_path = path or self.config.scaler_path
        saved_path = save_pickle(self.scaler_, target_path)
        logger.info("Saved scaler to %s", saved_path)
        return saved_path

    @classmethod
    def load(
        cls, path: str | Path | None = None, config: ScalerConfig | None = None
    ) -> NumericScaler:
        """Load a previously saved scaler."""

        cfg = config or ScalerConfig()
        scaler = cls(cfg)
        scaler.scaler_ = load_pickle(path or cfg.scaler_path)
        scaler.is_fitted_ = True
        return scaler


def fit_scale_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = NUMERIC_COLS,
    scaler_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, NumericScaler]:
    """Convenience wrapper that mirrors the notebook split-and-scale step."""

    scaler_config = ScalerConfig(
        numeric_columns=tuple(numeric_columns),
        scaler_path=Path(scaler_path) if scaler_path else PATHS.scaler_file,
    )
    scaler = NumericScaler(scaler_config)
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    if scaler_path is not None:
        scaler.save(scaler_path)
    else:
        scaler.save()
    return train_scaled, test_scaled, scaler
