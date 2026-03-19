"""Evaluation utilities for binary classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Metrics produced for a single model."""

    model_name: str
    accuracy: float
    roc_auc: float | None
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: str
    fpr: list[float] | None = None
    tpr: list[float] | None = None
    thresholds: list[float] | None = None


def evaluate_classifier(
    y_true,
    y_pred,
    y_prob=None,
    *,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute the standard churn metrics used in the notebook."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()
        metrics["thresholds"] = thresholds.tolist()
    else:
        metrics["roc_auc"] = None

    logger.info(
        "Evaluated %s: accuracy=%.4f, roc_auc=%s",
        model_name,
        metrics["accuracy"],
        "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}",
    )
    return metrics


def build_comparison_table(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact comparison table from per-model metrics."""

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "ROC_AUC": metrics.get("roc_auc"),
            }
        )
    rows.sort(key=lambda row: row["ROC_AUC"] if row["ROC_AUC"] is not None else -1.0, reverse=True)
    return rows


def plot_roc_curves(results: Mapping[str, Mapping[str, Any]], y_true, ax=None):
    """Plot ROC curves for a set of model results."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    y_true_arr = np.asarray(y_true)
    for model_name, metrics in results.items():
        y_prob = metrics.get("y_prob")
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true_arr, np.asarray(y_prob))
        roc_value = metrics.get("roc_auc")
        label = model_name if roc_value is None else f"{model_name} (AUC = {roc_value:.3f})"
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Churn Prediction Models")
    ax.legend(loc="lower right")
    return ax
