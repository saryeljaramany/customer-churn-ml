"""Central configuration for the customer churn package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .constants import NUMERIC_COLS, TARGET_COL, TARGET_ENCODED_COL

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Filesystem locations used by the project."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    model_dir: Path = PROJECT_ROOT / "model"
    notebook_dir: Path = PROJECT_ROOT / "notebooks"

    raw_data_file: Path = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    cleaned_data_file: Path = PROJECT_ROOT / "data" / "processed" / "cleaned_telco_churn.csv"
    train_features_file: Path = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
    test_features_file: Path = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
    train_target_file: Path = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
    test_target_file: Path = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

    scaler_file: Path = PROJECT_ROOT / "model" / "scaler.pkl"
    model_file: Path = PROJECT_ROOT / "model" / "churn_model.pkl"
    feature_names_file: Path = PROJECT_ROOT / "model" / "feature_names.pkl"
    id_col: str = "customerID"


PATHS: Final[PathConfig] = PathConfig()

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CARDINALITY_THRESHOLD: Final[int] = 30
SAVE_SPLITS: Final[bool] = False

MODEL_PARAMS: Final[dict[str, dict[str, Any]]] = {
    "Logistic Regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
    },
}

PREPROCESSING_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLS
TARGET_NAME: Final[str] = TARGET_COL
TARGET_ENCODED_NAME: Final[str] = TARGET_ENCODED_COL
