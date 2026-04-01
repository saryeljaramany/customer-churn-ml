"""Data loading helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..config import PATHS
from ..utils import get_logger, resolve_path, validate_columns

logger = get_logger(__name__)


def load_csv(
    path: str | Path,
    *,
    required_columns: Iterable[str] | None = None,
    frame_name: str = "data",
) -> pd.DataFrame:
    """Load a CSV file and validate it if required."""

    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{frame_name} file not found: {file_path}")

    df = pd.read_csv(file_path)
    if required_columns is not None:
        validate_columns(df, required_columns, frame_name=frame_name)

    logger.info("Loaded %s from %s with shape %s", frame_name, file_path, df.shape)
    return df


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw churn dataset."""

    return load_csv(path or PATHS.raw_data_file, frame_name="raw churn data")


def load_processed_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed churn dataset."""

    return load_csv(path or PATHS.cleaned_data_file, frame_name="processed churn data")
