"""customer_churn_ml public package API."""

from __future__ import annotations

from .config import CARDINALITY_THRESHOLD, MODEL_PARAMS, PATHS, RANDOM_STATE, SAVE_SPLITS, TEST_SIZE
from .constants import CONTRACT_ORDER, ID_COL, INTERNET_ORDER, NUMERIC_COLS, TARGET_COL, TARGET_ENCODED_COL
from .data.loader import load_csv, load_processed_data, load_raw_data
from .interpret.feature_importance import get_feature_importance_df, plot_feature_importance
from .preprocessing.preprocessor import Preprocessor, PreprocessorConfig, preprocess_dataframe
from .preprocessing.scalar import NumericScaler, ScalerConfig, fit_scale_train_test
from .training.evaluator import build_comparison_table, evaluate_classifier, plot_roc_curves
from .training.trainer import ModelTrainer, TrainingConfig, TrainingOutcome, train_models
from .utils import check_data_integrity, ensure_directory, get_logger, load_pickle, resolve_path, save_pickle

__all__ = [
    "CARDINALITY_THRESHOLD",
    "MODEL_PARAMS",
    "PATHS",
    "RANDOM_STATE",
    "SAVE_SPLITS",
    "TEST_SIZE",
    "CONTRACT_ORDER",
    "ID_COL",
    "INTERNET_ORDER",
    "NUMERIC_COLS",
    "TARGET_COL",
    "TARGET_ENCODED_COL",
    "load_csv",
    "load_processed_data",
    "load_raw_data",
    "get_feature_importance_df",
    "plot_feature_importance",
    "Preprocessor",
    "PreprocessorConfig",
    "preprocess_dataframe",
    "NumericScaler",
    "ScalerConfig",
    "fit_scale_train_test",
    "build_comparison_table",
    "evaluate_classifier",
    "plot_roc_curves",
    "ModelTrainer",
    "TrainingConfig",
    "TrainingOutcome",
    "train_models",
    "check_data_integrity",
    "ensure_directory",
    "get_logger",
    "load_pickle",
    "resolve_path",
    "save_pickle",
]
