"""Shared helpers for logging, validation, persistence, and diagnostics."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import PROJECT_ROOT
from .constants import ID_COL

_LOGGER_NAME = "customer_churn_ml"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package logger with a simple stream handler."""

    logger_name = name or _LOGGER_NAME
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str], *, frame_name: str = "DataFrame") -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def check_data_integrity(df: pd.DataFrame, *, id_col: str = ID_COL) -> dict[str, Any]:
    """Return a small integrity report for a dataframe."""

    report: dict[str, Any] = {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
    }
    if id_col in df.columns:
        report["duplicate_ids"] = int(df[id_col].duplicated().sum())
    else:
        report["duplicate_ids"] = None
    return report


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist an object with pickle."""

    file_path = resolve_path(path)
    ensure_directory(file_path.parent)
    with file_path.open("wb") as handle:
        pickle.dump(obj, handle)
    return file_path


def load_pickle(path: str | Path) -> Any:
    """Load a pickled artifact."""

    file_path = resolve_path(path)
    with file_path.open("rb") as handle:
        return pickle.load(handle)
