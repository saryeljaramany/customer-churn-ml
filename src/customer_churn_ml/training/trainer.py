"""Model training orchestration for the churn project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import MODEL_PARAMS, PATHS, RANDOM_STATE
from ..utils import get_logger, save_pickle
from .evaluator import build_comparison_table, evaluate_classifier

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training."""

    random_state: int = RANDOM_STATE
    model_params: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(MODEL_PARAMS))
    model_path: Path = PATHS.model_file
    feature_names_path: Path = PATHS.feature_names_file


@dataclass(slots=True)
class TrainingOutcome:
    """Results from a training run."""

    best_model_name: str
    best_model: Any
    results: dict[str, dict[str, Any]]
    comparison_table: list[dict[str, Any]]
    feature_names: list[str]


def _build_default_models(config: TrainingConfig) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(**config.model_params["Logistic Regression"]),
        "Random Forest": RandomForestClassifier(**config.model_params["Random Forest"]),
    }


class ModelTrainer:
    """Train and select the best binary classifier by ROC-AUC."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_test: pd.DataFrame,
        y_test,
        *,
        feature_names: list[str] | None = None,
        models: dict[str, Any] | None = None,
        save_artifacts: bool = True,
    ) -> TrainingOutcome:
        """Fit candidate models, evaluate them, and select the best one."""

        candidate_models = models or _build_default_models(self.config)
        results: dict[str, dict[str, Any]] = {}

        for model_name, model in candidate_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                y_prob = 1.0 / (1.0 + np.exp(-scores))
            else:
                y_prob = None

            metrics = evaluate_classifier(y_test, y_pred, y_prob, model_name=model_name)
            metrics["model"] = model
            metrics["y_pred"] = np.asarray(y_pred)
            metrics["y_prob"] = None if y_prob is None else np.asarray(y_prob)
            results[model_name] = metrics

            logger.info("Trained %s", model_name)

        if not results:
            raise ValueError("No models were trained.")

        best_name = max(results.keys(), key=lambda name: results[name].get("roc_auc") or -1.0)
        best_model = results[best_name]["model"]
        comparison_table = build_comparison_table(results)
        resolved_feature_names = feature_names or list(X_train.columns)

        if save_artifacts:
            save_pickle(best_model, self.config.model_path)
            save_pickle(resolved_feature_names, self.config.feature_names_path)
            logger.info("Saved best model to %s", self.config.model_path)
            logger.info("Saved feature names to %s", self.config.feature_names_path)

        return TrainingOutcome(
            best_model_name=best_name,
            best_model=best_model,
            results=results,
            comparison_table=comparison_table,
            feature_names=resolved_feature_names,
        )


def train_models(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    *,
    feature_names: list[str] | None = None,
    models: dict[str, Any] | None = None,
    save_artifacts: bool = True,
    config: TrainingConfig | None = None,
) -> TrainingOutcome:
    """Functional wrapper around :class:`ModelTrainer`."""

    trainer = ModelTrainer(config)
    return trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names=feature_names,
        models=models,
        save_artifacts=save_artifacts,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # CLI-only imports moved here to avoid import-time coupling.
    from sklearn.model_selection import train_test_split
    from ..config import TEST_SIZE, SAVE_SPLITS
    from ..constants import TARGET_ENCODED_COL
    from ..preprocessing.scalar import NumericScaler

    # CLI entry point for model training.
    #
    # Usage:
    #     python -m customer_churn_ml.training.trainer
    #
    # Behavior:
    # - Loads the cleaned/processed CSV from config.PATHS.cleaned_data_file
    # - Splits into train/test using config.TEST_SIZE and config.RANDOM_STATE
    # - Scales numeric features using preprocessing.NumericScaler (fit on train, applied to both splits)
    # - Trains default models and selects the best by ROC_AUC
    # - Saves the best model and feature names to model paths defined in config.PATHS
    # - Logs progress and prints a compact comparison table
    logger.info("Starting training CLI")

    try:
        cleaned_path = PATHS.cleaned_data_file
        logger.info("Loading cleaned data from %s", cleaned_path)
        df = pd.read_csv(cleaned_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load cleaned data from %s: %s", PATHS.cleaned_data_file, exc)
        raise SystemExit(1)

    target_col = TARGET_ENCODED_COL
    if target_col not in df.columns:
        logger.error("Expected target column '%s' in cleaned data; found columns: %s", target_col, list(df.columns))
        raise SystemExit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # split
    logger.info("Splitting data with test_size=%s, random_state=%s", TEST_SIZE, RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # optionally save splits (controlled by config.SAVE_SPLITS)
    if SAVE_SPLITS:
        try:
            logger.info("Saving data splits to %s", PATHS.processed_data_dir)
            X_train.to_csv(PATHS.train_features_file, index=False)
            X_test.to_csv(PATHS.test_features_file, index=False)
            y_train.to_csv(PATHS.train_target_file, index=False)
            y_test.to_csv(PATHS.test_target_file, index=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not save splits to disk: %s", exc)

    # scale numeric features before training
    logger.info("Fitting scaler on training data and transforming train/test splits")
    try:
        scaler = NumericScaler()
        X_train, X_test = scaler.fit_transform(X_train, X_test)
        try:
            scaler.save()
            logger.info("Saved scaler to configured location")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not save scaler: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to fit/transform scaler: %s", exc)
        raise SystemExit(1)

    # train
    trainer = ModelTrainer()
    logger.info("Training models")
    outcome = trainer.train(X_train, y_train, X_test, y_test, save_artifacts=True)

    logger.info("Best model: %s", outcome.best_model_name)
    logger.info("Comparison table:")
    for row in outcome.comparison_table:
        logger.info("  %s: accuracy=%s, roc_auc=%s", row["Model"], row["Accuracy"], row["ROC_AUC"])