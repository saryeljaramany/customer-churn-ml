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
